# okc: problem=muse_bf16_proj
import torch
import triton
import triton.language as tl


@triton.jit
def project(X, W, L, P, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr, WN: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, S: tl.constexpr, ROUTER: tl.constexpr):
    mi = tl.program_id(0) * BM + tl.arange(0, BM)
    ni = tl.program_id(1) * BN + tl.arange(0, BN)
    si = tl.program_id(2)
    ki = si * tl.cdiv(K, S) + tl.arange(0, BK)
    acc = tl.zeros((BM, BN), tl.float32)
    low = tl.zeros((BM, BN), tl.float32)
    for step in range(tl.cdiv(tl.cdiv(K, S), BK)):
        kk = ki + step * BK
        a = tl.load(X + mi[:, None] * K + kk[None, :], (mi[:, None] < M) & (kk[None, :] < K) & (kk[None, :] < (si + 1) * tl.cdiv(K, S)), 0)
        b = tl.load(W + kk[:, None] * WN + ni[None, :], (ni[None, :] < N) & (kk[:, None] < K) & (kk[:, None] < (si + 1) * tl.cdiv(K, S)), 0)
        acc = tl.dot(a, b, acc)
        if ROUTER:
            c = tl.load(L + kk[:, None] * WN + ni[None, :], (ni[None, :] < N) & (kk[:, None] < K) & (kk[:, None] < (si + 1) * tl.cdiv(K, S)), 0)
            low = tl.dot(a, c, low)
    tl.store(P + si * M * N + mi[:, None] * N + ni[None, :], acc + low, (mi[:, None] < M) & (ni[None, :] < N))


@triton.jit
def combine(P, Y, T: tl.constexpr, S: tl.constexpr, B: tl.constexpr):
    i = tl.program_id(0) * B + tl.arange(0, B)
    s = tl.arange(0, S)
    v = tl.load(P + s[:, None] * T + i[None, :], i[None, :] < T, 0)
    tl.store(Y + i, tl.sum(v, axis=0), i < T)


@triton.jit
def vector_project(X, W, Y, N: tl.constexpr, K: tl.constexpr, WK: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    n = tl.program_id(0) * BN + tl.arange(0, BN)
    k = tl.arange(0, BK)
    a = tl.load(X + k, k < K, 0).to(tl.float32)
    b = tl.load(W + n[:, None] * WK + k[None, :], (n[:, None] < N) & (k[None, :] < K), 0).to(tl.float32)
    y = tl.sum(a[None, :] * b, axis=1)
    tl.store(Y + n, y, n < N)


@triton.jit
def add_parts(P, Y, M: tl.constexpr, N: tl.constexpr, B: tl.constexpr):
    i = tl.program_id(0) * B + tl.arange(0, B)
    r = i // N
    c = i % N
    a = tl.load(P + r * (2 * N) + c, r < M, 0)
    b = tl.load(P + r * (2 * N) + c + N, r < M, 0)
    tl.store(Y + i, a + b, r < M)

from torch.utils.cpp_extension import load_inline


CUDA_SOURCE = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cublasLt.h>
#include <map>
#include <tuple>

struct Plan {
    cublasLtMatmulDesc_t operation;
    cublasLtMatrixLayout_t a, b, c;
    cublasLtMatmulAlgo_t algorithm;
};

void lt_matmul(torch::Tensor x, torch::Tensor w, torch::Tensor y, torch::Tensor workspace, int64_t choice) {
    static cublasLtHandle_t handle = []() {
        cublasLtHandle_t value;
        TORCH_CHECK(cublasLtCreate(&value) == CUBLAS_STATUS_SUCCESS, "cuBLASLt initialization failed");
        return value;
    }();
    static std::map<std::tuple<int,int,int,int>, Plan> plans;
    int m = x.size(0), k = x.size(1), n = y.size(1);
    auto key = std::make_tuple(m,n,k,int(choice));
    auto found = plans.find(key);
    if (found == plans.end()) {
        Plan p;
        cublasLtMatmulDescCreate(&p.operation, CUBLAS_COMPUTE_32F, CUDA_R_32F);
        cublasOperation_t trans = CUBLAS_OP_N;
        cublasLtMatmulDescSetAttribute(p.operation, CUBLASLT_MATMUL_DESC_TRANSA, &trans, sizeof(trans));
        cublasLtMatrixLayoutCreate(&p.a, CUDA_R_16BF, n, k, w.size(1));
        cublasLtMatrixLayoutCreate(&p.b, CUDA_R_16BF, k, m, k);
        cublasLtMatrixLayoutCreate(&p.c, CUDA_R_16BF, n, m, n);
        cublasLtMatmulPreference_t pref;
        cublasLtMatmulPreferenceCreate(&pref);
        size_t available = workspace.numel();
        cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &available, sizeof(available));
        cublasLtMatmulHeuristicResult_t results[16];
        int returned = 0;
        auto status = cublasLtMatmulAlgoGetHeuristic(handle, p.operation, p.a, p.b, p.c, p.c, pref, 16, results, &returned);
        TORCH_CHECK(status == CUBLAS_STATUS_SUCCESS && returned > 0, "No cuBLASLt algorithm available");
        p.algorithm = results[std::min(int(choice), returned - 1)].algo;
        cublasLtMatmulPreferenceDestroy(pref);
        found = plans.emplace(key, p).first;
    }
    Plan &p = found->second;
    float alpha = 1.0f, beta = 0.0f;
    auto status = cublasLtMatmul(handle, p.operation, &alpha, w.data_ptr(), p.a, x.data_ptr(), p.b, &beta, y.data_ptr(), p.c, y.data_ptr(), p.c, &p.algorithm, workspace.data_ptr(), workspace.numel(), at::cuda::getCurrentCUDAStream());
    TORCH_CHECK(status == CUBLAS_STATUS_SUCCESS, "cuBLASLt matmul failed: ", int(status));
}
"""

def _prepare_base(weights):
    state = {name: w.T.contiguous() for name, w in weights.items()}
    r = state["router"]
    state["hi"] = r.to(torch.bfloat16)
    state["lo"] = (r - state["hi"].float()).to(torch.bfloat16)
    state["original"] = weights
    state["vector_weights"] = {name: torch.nn.functional.pad(w, (0, 64)) for name, w in weights.items()}
    state["packed"] = torch.cat((state["hi"], state["lo"]), dim=1).contiguous()
    state["lt"] = load_inline(name="projection_lt_transposed", cpp_sources="void lt_matmul(torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, int64_t);", cuda_sources=CUDA_SOURCE, functions=["lt_matmul"], extra_cuda_cflags=["-O3"], extra_ldflags=["-lcublasLt"], verbose=False)
    state["workspace"] = torch.empty((32 * 1024 * 1024,), device=r.device, dtype=torch.uint8)
    return state



@triton.jit
def blocked_vector_project(X, W, Y, N: tl.constexpr, K: tl.constexpr, WK: tl.constexpr,
                           BN: tl.constexpr, BK: tl.constexpr):
    n = tl.program_id(0) * BN + tl.arange(0, BN)
    k = tl.arange(0, BK)
    acc = tl.zeros((BN, BK), tl.float32)
    for step in range(tl.cdiv(K, BK)):
        kk = k + step * BK
        a = tl.load(X + kk, kk < K, 0).to(tl.float32)
        b = tl.load(W + n[:, None] * WK + kk[None, :],
                    (n[:, None] < N) & (kk[None, :] < K), 0).to(tl.float32)
        acc += a[None, :] * b
    tl.store(Y + n, tl.sum(acc, axis=1), n < N)


def _prepare_algorithms(weights):
    import statistics
    state = _prepare_base(weights)
    state["choices"] = {}
    eviction = torch.empty((128 * 1024 * 1024,), device=state["workspace"].device, dtype=torch.uint8)
    begin = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    for name, weight in weights.items():
        if name == "router":
            continue
        n, k = weight.shape
        for m in (8, 16, 32, 64, 128, 256, 512, 1024):
            x = torch.randn((m, k), device=weight.device, dtype=torch.bfloat16)
            y = torch.empty((m, n), device=weight.device, dtype=torch.bfloat16)
            timings = []
            for choice in range(16):
                try:
                    state["lt"].lt_matmul(x, state[name], y, state["workspace"], choice)
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph):
                        state["lt"].lt_matmul(x, state[name], y, state["workspace"], choice)
                    samples = []
                    for repeat in range(5):
                        eviction.add_(1)
                        begin.record()
                        graph.replay()
                        end.record()
                        end.synchronize()
                        samples.append(begin.elapsed_time(end))
                    timings.append((statistics.median(samples), choice))
                except RuntimeError:
                    continue
            if timings:
                finalists = sorted(timings)[:3]
                verified = []
                for _, choice in finalists:
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph):
                        state["lt"].lt_matmul(x, state[name], y, state["workspace"], choice)
                    samples = []
                    for repeat in range(11):
                        eviction.add_(1)
                        begin.record()
                        graph.replay()
                        end.record()
                        end.synchronize()
                        samples.append(begin.elapsed_time(end))
                    verified.append((statistics.median(samples), choice))
                state["choices"][(name, m)] = min(verified)[1]
    torch.cuda.synchronize()
    return state


def _dense_base(data, state):
    x = data["x"]
    cfg = data["config"]
    m, n, k = x.shape[0], cfg["n"], x.shape[1]
    router = cfg["layer"] == "router"
    if m == 1:
        output = torch.empty((m, n), device=x.device, dtype=torch.float32 if router else torch.bfloat16)
        width, warps = state["vector_config"][cfg["layer"]]
        vector_project[(triton.cdiv(n, width),)](x, state["vector_weights"][cfg["layer"]], output, n, k, k + 64, width, triton.next_power_of_2(k), num_warps=warps)
        return output
    if not router and m <= 8 and n <= 2304 and k >= 4096:
        output = torch.empty((m, n), device=x.device, dtype=torch.bfloat16)
        partial = torch.empty((8, m, n), device=x.device, dtype=torch.float32)
        w = state[cfg["layer"]]
        project[(triton.cdiv(m, 16), triton.cdiv(n, 64), 8)](x, w, w, partial, m, n, k, w.shape[1], 16, 64, 128, 8, False, num_warps=4, num_stages=3)
        combine[(triton.cdiv(m * n, 256),)](partial, output, m * n, 8, 256)
        return output
    if not router:
        if cfg["layer"] == "qkv_gate" and m >= 192:
            return x @ state["original"][cfg["layer"]][:n].T
        choice = 2 if cfg["layer"] == "qkv_gate" and 64 <= m < 192 else (1 if cfg["layer"] == "pooler" and m <= 32 else 0)
        choice = state["choices"].get((cfg["layer"], m), choice)
        output = torch.empty((m, n), device=x.device, dtype=torch.bfloat16)
        state["lt"].lt_matmul(x, state[cfg["layer"]], output, state["workspace"], choice)
        return output
    if router and (16 < m < 64 or m >= 192):
        parts = torch.mm(x, state["packed"], out_dtype=torch.float32)
        output = torch.empty((m, n), device=x.device, dtype=torch.float32)
        add_parts[(triton.cdiv(m * n, 256),)](parts, output, m, n, 256)
        return output
    w = state["hi"]
    lo = state["lo"]
    bm = 16 if m <= 16 else 128
    bn = 32
    splits = 16
    partial = torch.empty((splits, m, n), device=x.device, dtype=torch.float32)
    output = torch.empty((m, n), device=x.device, dtype=torch.float32 if router else torch.bfloat16)
    project[(triton.cdiv(m, bm), triton.cdiv(n, bn), splits)](x, w, lo, partial, m, n, k, w.shape[1], bm, bn, 128, splits, True, num_warps=4, num_stages=3)
    combine[(triton.cdiv(m * n, 256),)](partial, output, m * n, splits, 256)
    return output



def _prepare_tuning_base(weights):
    import statistics
    state = _prepare_algorithms(weights)
    device = state["workspace"].device
    eviction = torch.empty((128 * 1024 * 1024,), device=device, dtype=torch.uint8)
    begin = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    state["vector_config"] = {}
    for name, weight in weights.items():
        n, k = weight.shape
        if name == "unpooler":
            n = min(n, 1024)
        x = torch.randn((1, k), device=device, dtype=torch.bfloat16)
        y = torch.empty((1, n), device=device, dtype=torch.float32 if name == "router" else torch.bfloat16)
        widths = (1, 2, 4, 8) if k <= 1024 else ((1, 2, 4) if k <= 4096 else (1, 2))
        measured = []
        for width in widths:
            for warps in (4, 8, 16):
                try:
                    vector_project[(triton.cdiv(n, width),)](
                        x, state["vector_weights"][name], y, n, k, k + 64, width,
                        triton.next_power_of_2(k), num_warps=warps,
                    )
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph):
                        vector_project[(triton.cdiv(n, width),)](
                            x, state["vector_weights"][name], y, n, k, k + 64, width,
                            triton.next_power_of_2(k), num_warps=warps,
                        )
                    times = []
                    for repeat in range(7):
                        eviction.add_(1)
                        begin.record()
                        graph.replay()
                        end.record()
                        end.synchronize()
                        times.append(begin.elapsed_time(end))
                    measured.append((statistics.median(times), width, warps))
                except Exception:
                    continue
        if measured:
            finalists = []
            for _, width, warps in sorted(measured)[:3]:
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    vector_project[(triton.cdiv(n, width),)](
                        x, state["vector_weights"][name], y, n, k, k + 64, width,
                        triton.next_power_of_2(k), num_warps=warps,
                    )
                times = []
                for repeat in range(13):
                    eviction.add_(1)
                    begin.record()
                    graph.replay()
                    end.record()
                    end.synchronize()
                    times.append(begin.elapsed_time(end))
                finalists.append((statistics.median(times), width, warps))
            _, width, warps = min(finalists)
            state["vector_config"][name] = (width, warps)
        else:
            state["vector_config"][name] = (4 if k <= 1024 else 1, 8 if k <= 1024 else 4)
    torch.cuda.synchronize()
    return state

def _run_small_tile(x, weight, tile):
    m, k = x.shape
    n = tile[-1]
    bm, bn, bk, splits, warps = tile[:-1]
    output = torch.empty((m, n), device=x.device, dtype=torch.bfloat16)
    partial = torch.empty((splits, m, n), device=x.device, dtype=torch.float32)
    project[(triton.cdiv(m, bm), triton.cdiv(n, bn), splits)](
        x, weight, weight, partial, m, n, k, weight.shape[1], bm, bn, bk, splits,
        False, num_warps=warps, num_stages=3,
    )
    combine[(triton.cdiv(m * n, 256),)](partial, output, m * n, splits, 256)
    return output


def _prepare_dense_tiles(weights):
    import statistics
    state = _prepare_tuning_base(weights)
    state["small_tiles"] = {}
    device = state["workspace"].device
    eviction = torch.empty((128 * 1024 * 1024,), device=device, dtype=torch.uint8)
    start = torch.cuda.Event(enable_timing=True)
    finish = torch.cuda.Event(enable_timing=True)
    def measure(operation):
        operation()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            operation()
        samples = []
        for repeat in range(9):
            eviction.add_(1)
            start.record()
            graph.replay()
            finish.record()
            finish.synchronize()
            samples.append(start.elapsed_time(finish))
        return statistics.median(samples)
    for name, weight in weights.items():
        if name == "router":
            continue
        full_n, k = weight.shape
        for m in (8, 16, 32):
            n = min(full_n, 1024) if name == "unpooler" and m <= 8 else full_n
            x = torch.randn((m, k), device=device, dtype=torch.bfloat16)
            data = {"x": x, "config": {"layer": name, "n": n}}
            reference_time = measure(lambda: _dense_base(data, state))
            choices = []
            for bn, bk, splits, warps in ((64, 256, 4, 4), (128, 256, 4, 8),
                                         (64, 256, 2, 4), (128, 128, 2, 4),
                                         (32, 256, 8, 4), (64, 512, 4, 8)):
                tile = (max(16, triton.next_power_of_2(m)), bn, bk, splits, warps, n)
                try:
                    elapsed = measure(lambda: _run_small_tile(x, state[name], tile))
                    choices.append((elapsed, tile))
                except RuntimeError:
                    continue
            if choices:
                elapsed, tile = min(choices)
                if elapsed < reference_time * 0.97:
                    elapsed = measure(lambda: _run_small_tile(x, state[name], tile))
                    reference_time = measure(lambda: _dense_base(data, state))
                    if elapsed < reference_time * 0.97:
                        state["small_tiles"][(name, m)] = tile
    return state


def _dense_tiles(data, state):
    tile = state["small_tiles"].get((data["config"]["layer"], data["x"].shape[0]))
    if tile is not None and tile[-1] == data["config"]["n"]:
        return _run_small_tile(data["x"], state[data["config"]["layer"]], tile)
    return _dense_base(data, state)



def _run_router_tile(x, state, n, tile):
    m, k = x.shape
    bm, bn, bk, splits, warps = tile
    output = torch.empty((m, n), device=x.device, dtype=torch.float32)
    partial = torch.empty((splits, m, n), device=x.device, dtype=torch.float32)
    high, low = state["hi"], state["lo"]
    stages = 2 if 6 * bk * (bm + 2 * bn) > 196608 else 3
    project[(triton.cdiv(m, bm), triton.cdiv(n, bn), splits)](
        x, high, low, partial, m, n, k, high.shape[1],
        bm, bn, bk, splits, True, num_warps=warps, num_stages=stages,
    )
    combine[(triton.cdiv(m * n, 256),)](partial, output, m * n, splits, 256)
    return output


def prepare(weights):
    import statistics
    state = _prepare_dense_tiles(weights)
    state["router_tiles"] = {}
    weight = weights["router"]
    n, k = weight.shape
    device = weight.device
    eviction = torch.empty((128 * 1024 * 1024,), device=device, dtype=torch.uint8)
    start = torch.cuda.Event(enable_timing=True)
    finish = torch.cuda.Event(enable_timing=True)

    def capture(operation):
        operation()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            operation()
        return graph

    def measure(graph, repeats):
        samples = []
        for repeat in range(repeats):
            eviction.add_(1)
            start.record()
            graph.replay()
            finish.record()
            finish.synchronize()
            samples.append(start.elapsed_time(finish))
        return statistics.median(samples)

    for m in (8, 16, 32, 128, 256):
        x = torch.randn((m, k), device=device, dtype=torch.bfloat16)
        data = {"x": x, "config": {"layer": "router", "n": n}}
        reference_graph = capture(lambda: _dense_tiles(data, state))
        reference_time = measure(reference_graph, 9)
        bm = min(64, max(16, triton.next_power_of_2(m)))
        choices = (
            (bm, 32, 128, 8, 4),
            (bm, 32, 128, 16, 4),
            (bm, 32, 256, 8, 4),
            (bm, 32, 256, 16, 4),
            (bm, 64, 128, 16, 4),
            (bm, 64, 256, 8, 4),
            (max(16, bm // 2), 32, 128, 8, 4),
            (bm, 32, 256, 32, 8),
        )
        measured = []
        graphs = {}
        for tile in dict.fromkeys(choices):
            try:
                graph = capture(lambda: _run_router_tile(x, state, n, tile))
                elapsed = measure(graph, 7)
                measured.append((elapsed, tile))
                graphs[tile] = graph
            except Exception:
                continue
        if measured:
            elapsed, tile = min(measured)
            if elapsed < reference_time * 0.97:
                ratios = []
                for repeat in range(3):
                    if repeat % 2:
                        candidate_time = measure(graphs[tile], 9)
                        baseline_time = measure(reference_graph, 9)
                    else:
                        baseline_time = measure(reference_graph, 9)
                        candidate_time = measure(graphs[tile], 9)
                    ratios.append(candidate_time / baseline_time)
                if max(ratios) < 0.97:
                    state["router_tiles"][m] = tile
    return state


def kernel_fn(data, state):
    if data["config"]["layer"] == "router":
        m = data["x"].shape[0]
        tile = state["router_tiles"].get(m)
        if tile is not None:
            return _run_router_tile(data["x"], state, data["config"]["n"], tile)
    return _dense_tiles(data, state)
