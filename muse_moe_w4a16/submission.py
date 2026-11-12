# okc: problem=muse_moe_w4a16
"""SGLang's NVFP4 expert path for Muse Spark Open on H200 (Marlin MoE runner), at SGLang commit
a55ab7d5ae1bba59ac351e55026fe7f7644ecc5f: MuseSpark1xFp4Experts._forward_marlin, starting after the pre-expert norm.

prepare() repacks the experts into Marlin's layout with prepare_moe_nvfp4_layer_for_marlin,
as SGLang does at load. kernel_fn runs moe_align_block_size, two moe_wna16_marlin_gemm
(W4A16, FP32 reduce), silu_and_mul and the Triton post-expert-norm weighted sum.

Differs from SGLang: the expert all-reduce is not run."""

from typing import TypedDict

import types

import torch
from sgl_kernel.scalar_type import scalar_types

from sglang.kernels.ops.activation.activation import silu_and_mul
from sglang.kernels.ops.moe.moe_wna16_marlin import moe_wna16_marlin_gemm
from sglang.kernels.ops.muse_spark_v12.fused_post_expert_norm_weighted_sum_triton import (
    fused_post_expert_norm_weighted_sum,
)
from sglang.srt.layers.moe.fused_moe_triton import moe_align_block_size
from sglang.srt.layers.quantization.marlin_utils import marlin_make_workspace
from sglang.srt.layers.quantization.marlin_utils_fp4 import prepare_moe_nvfp4_layer_for_marlin


class Weights(TypedDict):
    w13: torch.Tensor          # uint8 [256, 2I, H/2], gate rows [0, I) then up rows [I, 2I), packed E2M1
    w13_sf: torch.Tensor       # float8_e4m3fn [256, 2I, H/16]
    w13_gs: torch.Tensor       # float32 [256]
    w2: torch.Tensor           # uint8 [256, H, I/2]
    w2_sf: torch.Tensor        # float8_e4m3fn [256, H, I/16]
    w2_gs: torch.Tensor        # float32 [256]
    post_norm_w: torch.Tensor  # bfloat16 [H], the post-expert RMSNorm's weight (applied as 1 + w)


class Config(TypedDict):
    bs: int           # tokens, M
    d_hidden: int     # H = 4096, the width the MoE block runs its experts at
    d_expert: int     # I = 4096
    n_experts: int    # 256 experts in the model
    local_start: int  # this GPU computes experts [local_start, local_start + local_count)
    local_count: int  # 32: one GPU under 8-way expert parallelism
    top_k: int        # 8
    sharding: str     # "ep8"


class Inputs(TypedDict):
    hidden_states: torch.Tensor  # bfloat16 [M, H]
    topk_ids: torch.Tensor       # int32 [M, 8], global expert ids (0-255)
    topk_weights: torch.Tensor   # float32 [M, 8]
    config: Config


N_EXPERTS = 256
TOP_K = 8
# MuseSpark1xFp4Experts' candidates, smallest first.
BLOCK_SIZE_M_CANDIDATES = (8, 16, 32, 48, 64)
POST_NORM_EPS = 1e-8


def _marlin_layer(weights, start, count):
    """The parameters prepare_moe_nvfp4_layer_for_marlin reads, for experts [start, start+count)."""
    sl = slice(start, start + count)
    layer = types.SimpleNamespace(
        quant_config=types.SimpleNamespace(group_size=16),
        moe_runner_config=types.SimpleNamespace(is_gated=True),
        intermediate_size_per_partition=weights["w13"].shape[1] // 2,
        params_dtype=torch.bfloat16,
        w13_weight=weights["w13"][sl].contiguous(),
        w2_weight=weights["w2"][sl].contiguous(),
        w13_weight_scale=weights["w13_sf"][sl].contiguous(),
        w2_weight_scale=weights["w2_sf"][sl].contiguous(),
        # One global scale per expert for gate and up, as in SGLang's Marlin path.
        w13_weight_scale_2=weights["w13_gs"][sl].contiguous(),
        w2_weight_scale_2=weights["w2_gs"][sl].contiguous(),
    )
    prepare_moe_nvfp4_layer_for_marlin(layer)
    return layer


def _prepare_native(weights):
    device = weights["w13"].device
    # One repacked bank: this GPU's 32 experts under EP8.
    banks = {}
    for start, count in ((0, 32),):
        layer = _marlin_layer(weights, start, count)
        expert_map = None
        if count != N_EXPERTS:
            expert_map = torch.full((N_EXPERTS,), -1, dtype=torch.int32, device=device)
            expert_map[start:start + count] = torch.arange(count, dtype=torch.int32, device=device)
        banks[(start, count)] = {
            "w13": layer.w13_weight.data, "w13_s": layer.w13_weight_scale.data, "w13_gs": layer.w13_weight_scale_2.data,
            "w2": layer.w2_weight.data, "w2_s": layer.w2_weight_scale.data, "w2_gs": layer.w2_weight_scale_2.data,
            "expert_map": expert_map, "count": count, "size_n": layer.intermediate_size_per_partition,
        }
    torch.cuda.synchronize()
    return {
        "banks": banks,
        "workspace": marlin_make_workspace(device, max_blocks_per_sm=4),
        # MuseSpark1xFp4Experts binds the effective (1 + w) weight of the post-expert norm.
        "norm_w": (weights["post_norm_w"].float() + 1.0).to(torch.bfloat16),
    }


def kernel_fn(data, state):
    cfg = data["config"]
    bank = state["banks"][(cfg["local_start"], cfg["local_count"])]
    x = data["hidden_states"].contiguous()
    num_tokens, size_k = x.shape
    size_n = bank["size_n"]
    gemm1_n = 2 * size_n
    expert_map = bank["expert_map"]

    block_size_m = 8 if num_tokens < 16 else 16
    topk_weights = data["topk_weights"].to(torch.float32).contiguous()
    if num_tokens <= 32 and expert_map is not None:
        slots = num_tokens * TOP_K
        maximum = triton.cdiv(slots + bank["count"] * (block_size_m - 1), block_size_m) * block_size_m
        local_ids = torch.empty_like(data["topk_ids"], dtype=torch.int32)
        sorted_ids = torch.empty((maximum,), device=x.device, dtype=torch.int32)
        block_experts = torch.empty((maximum // block_size_m,), device=x.device, dtype=torch.int32)
        n_padded = torch.empty((1,), device=x.device, dtype=torch.int32)
        _align_local_routes[(1,)](data["topk_ids"], local_ids, sorted_ids, block_experts, n_padded,
            slots, bank["count"], cfg["local_start"], block_size_m, triton.next_power_of_2(slots),
            maximum, triton.next_power_of_2(maximum), triton.next_power_of_2(maximum // block_size_m), num_warps=8)
    else:
        local_ids = data["topk_ids"].to(torch.int32)
        if expert_map is not None:
            local_ids = expert_map[local_ids]
        local_ids = local_ids.contiguous()
        sorted_ids, block_experts, n_padded = moe_align_block_size(local_ids, block_size_m, bank["count"], ignore_invalid_expert=True)

    cache2 = torch.empty((num_tokens * TOP_K, size_n), device=x.device, dtype=x.dtype)
    cache13 = torch.empty((num_tokens * TOP_K * max(gemm1_n, size_k),), device=x.device, dtype=x.dtype)
    cache1 = cache13[: num_tokens * TOP_K * gemm1_n].view(-1, gemm1_n)
    cache3 = cache13[: num_tokens * TOP_K * size_k].view(-1, size_k)
    common = dict(moe_block_size=block_size_m, mul_topk_weights=False, is_ep=False,
                  b_q_type=scalar_types.float4_e2m1f, is_k_full=True, use_atomic_add=False,
                  use_fp32_reduce=True, is_zp_float=False)
    ws = state["workspace"]
    cache1 = state["marlin_operation"](
        x, cache1, bank["w13"], None, bank["w13_s"], bank["w13_gs"], None, None, None, ws,
        sorted_ids, block_experts, n_padded, topk_weights, top_k=TOP_K,
        size_m=num_tokens, size_n=gemm1_n, size_k=size_k, **common)
    silu_and_mul(cache1.view(-1, gemm1_n), cache2, expert_ids=local_ids.view(-1), expert_step=1)
    y = state["marlin_operation"](
        cache2, cache3, bank["w2"], None, bank["w2_s"], bank["w2_gs"], None, None, None, ws,
        sorted_ids, block_experts, n_padded, topk_weights, top_k=1,
        size_m=num_tokens * TOP_K, size_n=size_k, size_k=size_n, **common).view(num_tokens, TOP_K, size_k)
    return _finalize(
        y, topk_weights, norm_weight=state["norm_w"], eps=POST_NORM_EPS, weight_first=True,
        slot_expert_ids=local_ids)


import triton
import triton.language as tl


@triton.jit
def _route_norm(Y, W, Scores, IDs, Terms, H: tl.constexpr, K: tl.constexpr, B: tl.constexpr):
    token = tl.program_id(0)
    slot = tl.program_id(1)
    offsets = tl.arange(0, B)
    live = tl.load(IDs + token * K + slot) >= 0
    if live:
        value = tl.load(Y + (token * K + slot) * H + offsets, offsets < H, 0).to(tl.float32)
        weight = tl.load(W + offsets, offsets < H, 0).to(tl.float32)
        value = value * weight
        inverse = tl.rsqrt(tl.sum(value * value, axis=0) / H + 1.0e-8)
        score = tl.load(Scores + token * K + slot)
        tl.store(Terms + (token * K + slot) * H + offsets, value * inverse * score, offsets < H)
    else:
        tl.store(Terms + (token * K + slot) * H + offsets, 0.0, offsets < H)


@triton.jit
def _route_sum(Terms, Out, M: tl.constexpr, H: tl.constexpr, K: tl.constexpr, B: tl.constexpr):
    index = tl.program_id(0) * B + tl.arange(0, B)
    token = index // H
    column = index % H
    slots = tl.arange(0, K)
    values = tl.load(Terms + (token[None, :] * K + slots[:, None]) * H + column[None, :],
                     index[None, :] < M * H, 0.0)
    tl.store(Out + index, tl.sum(values, axis=0), index < M * H)


@triton.jit
def _route_norm_one(Y, W, Scores, IDs, Out, H: tl.constexpr, K: tl.constexpr, B: tl.constexpr):
    token = tl.program_id(0)
    offsets = tl.arange(0, B)
    slots = tl.arange(0, K)
    ids = tl.load(IDs + token * K + slots)
    values = tl.load(Y + (token * K + slots[:, None]) * H + offsets[None, :],
                     (ids[:, None] >= 0) & (offsets[None, :] < H), 0.0).to(tl.float32)
    weight = tl.load(W + offsets, offsets < H, 0.0).to(tl.float32)
    values = values * weight[None, :]
    inverse = tl.rsqrt(tl.sum(values * values, axis=1) / H + 1.0e-8)
    scores = tl.load(Scores + token * K + slots)
    result = tl.sum(values * (inverse * scores)[:, None], axis=0)
    tl.store(Out + token * H + offsets, result, offsets < H)


def _finalize(expert_output, topk_scores, norm_weight=None, eps=1e-8,
              weight_first=True, slot_expert_ids=None):
    tokens, topk, hidden = expert_output.shape
    if tokens >= 32:
        return fused_post_expert_norm_weighted_sum(
            expert_output, topk_scores, norm_weight=norm_weight, eps=eps,
            weight_first=weight_first, slot_expert_ids=slot_expert_ids,
        )
    out = torch.empty((tokens, hidden), device=expert_output.device, dtype=expert_output.dtype)
    if tokens == 1:
        _route_norm_one[(tokens,)](
            expert_output, norm_weight, topk_scores, slot_expert_ids, out,
            hidden, topk, triton.next_power_of_2(hidden), num_warps=16,
        )
    else:
        terms = torch.empty((tokens, topk, hidden), device=expert_output.device, dtype=torch.float32)
        _route_norm[(tokens, topk)](
            expert_output, norm_weight, topk_scores, slot_expert_ids, terms,
            hidden, topk, triton.next_power_of_2(hidden), num_warps=4,
        )
        _route_sum[(triton.cdiv(tokens * hidden, 256),)](
            terms, out, tokens, hidden, topk, 256, num_warps=4,
        )
    return out

EXPERT_LAUNCH_SOURCE = r"""
#include "gemm/marlin_moe/moe_wna16_marlin.cuh"
bool launch_experts(
    tvm::ffi::TensorView a, tvm::ffi::TensorView c,
    tvm::ffi::TensorView b, tvm::ffi::TensorView scales, tvm::ffi::TensorView global,
    tvm::ffi::TensorView workspace, tvm::ffi::TensorView sorted,
    tvm::ffi::TensorView experts, tvm::ffi::TensorView padded,
    tvm::ffi::TensorView scores, tvm::ffi::TensorView temporary,
    int64_t block_m, int64_t top_k, int64_t n, int64_t k,
    int64_t tile_k, int64_t tile_n, int64_t threads, int64_t occupancy) {
    using namespace sglang;
    using namespace sglang::device::marlin_moe;
    int dev = a.device().device_id;
    cudaStream_t stream = host::LaunchKernel::resolve_device(a.device());
    int sms = 0;
    int available = 0;
    host::RuntimeDeviceCheck(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev));
    host::RuntimeDeviceCheck(cudaDeviceGetAttribute(&available, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev));
    int shared = available / static_cast<int>(occupancy) - kSharedMemoryLaunchReserve;
    thread_config_t config{static_cast<int>(tile_k), static_cast<int>(tile_n), static_cast<int>(threads)};
    bool small = block_m == 8;
    int blocks_m = (static_cast<int>(block_m) + 15) / 16;
    MarlinFuncPtr kernel = nullptr;
    if (blocks_m == 1 && tile_n == 128) {
        if (tile_k == 64 && threads == 128) {
            if (small) {
                kernel = sglang::device::marlin_moe::Marlin<__nv_bfloat16, host::kFE2M1f.id(), host::kFE4M3fn.id(),
                                128, 1, 8, 4, true, pipe_stages, 1, false, false, false>;
            } else {
                kernel = sglang::device::marlin_moe::Marlin<__nv_bfloat16, host::kFE2M1f.id(), host::kFE4M3fn.id(),
                                128, 1, 8, 4, false, pipe_stages, 1, false, false, false>;
            }
        } else if (tile_k == 128 && threads == 256) {
            if (small) {
                kernel = sglang::device::marlin_moe::Marlin<__nv_bfloat16, host::kFE2M1f.id(), host::kFE4M3fn.id(),
                                256, 1, 8, 8, true, pipe_stages, 1, false, false, false>;
            } else {
                kernel = sglang::device::marlin_moe::Marlin<__nv_bfloat16, host::kFE2M1f.id(), host::kFE4M3fn.id(),
                                256, 1, 8, 8, false, pipe_stages, 1, false, false, false>;
            }
        }
    }
    if (kernel == nullptr || !is_valid_config(config, small, blocks_m,
            a.size(0), n, k, 4, 16, false, true, false, false, shared)) {
        return false;
    }
    host::RuntimeDeviceCheck(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, available - kSharedMemoryLaunchReserve));
    kernel<<<sms * occupancy, threads, shared, stream>>>(
        static_cast<const int4*>(a.data_ptr()), static_cast<const int4*>(b.data_ptr()),
        static_cast<int4*>(c.data_ptr()), static_cast<int4*>(temporary.data_ptr()), nullptr,
        static_cast<const int4*>(scales.data_ptr()), static_cast<const uint16_t*>(global.data_ptr()),
        nullptr, nullptr, static_cast<const int32_t*>(sorted.data_ptr()),
        static_cast<const int32_t*>(experts.data_ptr()), static_cast<const int32_t*>(padded.data_ptr()),
        static_cast<const float*>(scores.data_ptr()), top_k, false, false,
        scales.size(1), a.size(0), n, k, static_cast<int*>(workspace.data_ptr()),
        false, false, true, shared);
    host::RuntimeDeviceCheck(cudaGetLastError());
    return true;
}
"""

class _ExpertLauncher:
    def __init__(self, operation, sms, choices=None, fixed=None):
        self.operation = operation
        self.sms = sms
        self.choices = {} if choices is None else choices
        self.fixed = fixed

    def __call__(self, a, c, b, bias, scales, global_scale, zeros, g_idx, permutation,
                 workspace, sorted_ids, expert_ids, padded, scores, **kw):
        block = kw["moe_block_size"]
        n, k = kw["size_n"], kw["size_k"]
        rows = a.shape[0] if kw["top_k"] == TOP_K else a.shape[0] // TOP_K
        bucket = 1 << (max(rows, 1) - 1).bit_length()
        tile = self.fixed or self.choices.get((bucket, kw["top_k"], n, k, block))
        if tile is None:
            return moe_wna16_marlin_gemm(
                a, c, b, bias, scales, global_scale, zeros, g_idx, permutation,
                workspace, sorted_ids, expert_ids, padded, scores, **kw)
        size = min(n * sorted_ids.numel(), self.sms * 4 * block * 256)
        if block == 8:
            size *= 2
        temporary = torch.empty((size,), device=a.device, dtype=torch.float32)
        used = self.operation.launch_experts(a, c, b, scales, global_scale, workspace,
                                      sorted_ids, expert_ids, padded, scores, temporary,
                                      block, kw["top_k"], n, k, *tile)
        if not used:
            return moe_wna16_marlin_gemm(
                a, c, b, bias, scales, global_scale, zeros, g_idx, permutation,
                workspace, sorted_ids, expert_ids, padded, scores, **kw)
        return c


def prepare(weights):
    import math
    import statistics
    import time
    from tvm_ffi.cpp import load_inline
    from sglang.kernels.jit.utils.compile.paths import KERNEL_PATH
    from sglang.kernels.jit.utils.arch import get_default_target_flags

    started = time.monotonic()
    deadline = started + 245.0
    state = _prepare_native(weights)
    state["marlin_operation"] = moe_wna16_marlin_gemm
    try:
        operation = load_inline(
            name="expert_launch_local_only", cuda_sources=EXPERT_LAUNCH_SOURCE,
            functions=["launch_experts"],
            extra_include_paths=[str(KERNEL_PATH / "include"), str(KERNEL_PATH / "csrc")],
            extra_cuda_cflags=["-std=c++20", "-O3"] + get_default_target_flags(),
        )
    except Exception as error:
        print("expert launch compile fallback", str(error)[:240])
        return state
    print("expert launch ready seconds", round(time.monotonic() - started, 2))
    device = weights["w13"].device
    sms = torch.cuda.get_device_properties(device).multi_processor_count
    launcher = _ExpertLauncher(operation, sms)
    state["marlin_operation"] = launcher
    bank = state["banks"][(0, 32)]
    count = bank["count"]
    hidden = weights["w2"].shape[1]
    intermediate = bank["size_n"]
    launcher.choices[(1, TOP_K, intermediate * 2, hidden, 8)] = (128, 128, 256, 2)
    launcher.choices[(1, 1, hidden, intermediate, 8)] = (128, 128, 256, 2)
    if time.monotonic() >= deadline:
        print("expert tuning deadline before search", round(time.monotonic() - started, 2))
        return state
    generator = torch.Generator(device=device).manual_seed(314159)
    eviction = torch.zeros((128 * 1024 * 1024,), device=device, dtype=torch.uint8)
    begin = torch.cuda.Event(enable_timing=True)
    finish = torch.cuda.Event(enable_timing=True)
    tuning_workspace = marlin_make_workspace(device, max_blocks_per_sm=4)
    tiles = [(64, 128, 128, occupancy) for occupancy in (1, 2, 3, 4)]
    tiles += [(128, 128, 256, occupancy) for occupancy in (1, 2)]

    def capture(call):
        call()
        call()
        torch.cuda.synchronize(device)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            call()
        graph.replay()
        torch.cuda.synchronize(device)
        return graph

    def paired_ratio(native_graph, candidate_graph, repeats):
        ratios = []
        for repeat in range(repeats):
            elapsed = [0.0, 0.0]
            order = (0, 1) if repeat % 2 == 0 else (1, 0)
            for side in order:
                eviction.add_(1)
                begin.record()
                (native_graph if side == 0 else candidate_graph).replay()
                finish.record()
                finish.synchronize()
                elapsed[side] = begin.elapsed_time(finish)
            ratios.append(elapsed[1] / max(elapsed[0], 1.0e-9))
        return statistics.median(ratios)

    def valid_output(candidate, reference, live):
        actual = candidate[live].float()
        expected = reference[live].float()
        error = (actual - expected).square().sum(dim=1)
        scale = expected.square().sum(dim=1)
        floor = scale.mean() * 0.01
        overall = error.sum() <= scale.sum().clamp_min(1.0e-20) * 2.5e-5
        per_row = error <= torch.maximum(scale, floor).clamp_min(1.0e-20) * 4.0e-4
        return bool((torch.isfinite(actual).all() & overall & per_row.all()).item())

    for rows in (128, 256, 32, 16, 8, 1, 64):
        if time.monotonic() >= deadline:
            print("expert tuning deadline at batch", rows, round(time.monotonic() - started, 2))
            break
        block = 8 if rows < 16 else 16
        samples = []
        for slope, local_share in ((0.0, 0.25), (1.2, 0.5)):
            popularity = torch.arange(1, count + 1, device=device, dtype=torch.float32).pow(-slope)
            popularity = popularity[torch.randperm(count, device=device, generator=generator)]
            ids = torch.multinomial(popularity.expand(rows, count), TOP_K,
                                    replacement=False, generator=generator).to(torch.int32)
            selected = torch.rand((rows, TOP_K), device=device, generator=generator) < local_share
            selected[0, 0] = True
            ids = torch.where(selected, ids, -1).contiguous()
            scores = torch.rand((rows, TOP_K), device=device, generator=generator)
            scores = (scores / scores.sum(dim=1, keepdim=True)).contiguous()
            sorted_ids, expert_ids, padded = moe_align_block_size(ids, block, count, ignore_invalid_expert=True)
            live = ids.reshape(-1) >= 0
            samples.append((sorted_ids, expert_ids, padded, scores, live))
        for top_k, n, k, label in ((TOP_K, intermediate * 2, hidden, "w13"),
                                    (1, hidden, intermediate, "w2")):
            if time.monotonic() >= deadline:
                break
            common = dict(moe_block_size=block, mul_topk_weights=False, is_ep=False,
                          b_q_type=scalar_types.float4_e2m1f, is_k_full=True,
                          use_atomic_add=False, use_fp32_reduce=True, is_zp_float=False,
                          top_k=top_k, size_m=rows if top_k == TOP_K else rows * TOP_K,
                          size_n=n, size_k=k)
            comparisons = []
            diagnostics = []
            try:
                for sorted_ids, expert_ids, padded, scores, live in samples:
                    x = torch.randn((common["size_m"], k), device=device,
                                    dtype=torch.bfloat16, generator=generator)
                    reference = torch.empty((rows * TOP_K, n), device=device, dtype=torch.bfloat16)
                    candidate = torch.empty_like(reference)
                    tail = (bank[label], None, bank[label + "_s"], bank[label + "_gs"],
                            None, None, None, tuning_workspace, sorted_ids, expert_ids, padded, scores)
                    native_args = (x, reference) + tail
                    candidate_args = (x, candidate) + tail
                    native_graph = capture(lambda args=native_args: moe_wna16_marlin_gemm(*args, **common))
                    comparisons.append((native_graph, candidate_args, candidate, reference, live))
                choices = []
                for tile in tiles:
                    if time.monotonic() >= deadline:
                        break
                    contender = _ExpertLauncher(operation, sms, fixed=tile)
                    ratios = []
                    valid = True
                    for native_graph, args, candidate, reference, live in comparisons:
                        contender(*args, **common)
                        if not valid_output(candidate, reference, live):
                            valid = False
                            break
                        candidate_graph = capture(lambda args=args: contender(*args, **common))
                        ratios.append(paired_ratio(native_graph, candidate_graph, 3))
                        del candidate_graph
                    diagnostics.append((tile, [round(ratio, 4) for ratio in ratios], valid))
                    if valid and len(ratios) == len(comparisons):
                        mean_ratio = math.exp(sum(math.log(ratio) for ratio in ratios) / len(ratios))
                        if mean_ratio < 0.97 and max(ratios) <= 1.02:
                            choices.append((mean_ratio, tile))
                if choices and time.monotonic() < deadline:
                    _, tile = min(choices)
                    contender = _ExpertLauncher(operation, sms, fixed=tile)
                    ratios = []
                    for native_graph, args, candidate, reference, live in comparisons:
                        candidate_graph = capture(lambda args=args: contender(*args, **common))
                        ratios.append(paired_ratio(native_graph, candidate_graph, 9))
                        del candidate_graph
                    mean_ratio = math.exp(sum(math.log(ratio) for ratio in ratios) / len(ratios))
                    if mean_ratio < 0.97 and max(ratios) <= 1.02:
                        launcher.choices[(rows, top_k, n, k, block)] = tile
            except Exception as error:
                print("expert tuning fallback", rows, label, str(error)[:240])
            print("expert tuning", rows, label, "seconds", round(time.monotonic() - started, 2),
                  "selected", launcher.choices.get((rows, top_k, n, k, block)), "trials", diagnostics)
            comparisons.clear()
    torch.cuda.synchronize(device)
    print("expert launch choices", launcher.choices, "seconds", round(time.monotonic() - started, 2))
    return state


@triton.jit
def _align_local_routes(IDs, Local, Sorted, Experts, Padded, S: tl.constexpr, E: tl.constexpr,
                        START: tl.constexpr, BLOCK: tl.constexpr, BS: tl.constexpr,
                        MAX: tl.constexpr, BM: tl.constexpr, BE: tl.constexpr):
    slots = tl.arange(0, BS)
    experts = tl.arange(0, E)
    ids = tl.load(IDs + slots, slots < S, -1).to(tl.int32) - START
    valid = (slots < S) & (ids >= 0) & (ids < E)
    tl.store(Local + slots, tl.where(valid, ids, -1), slots < S)
    matches = (experts[:, None] == ids[None, :]) & valid[None, :]
    counts = tl.sum(matches.to(tl.int32), axis=1)
    padded = tl.cdiv(counts, BLOCK) * BLOCK
    ends = tl.cumsum(padded)
    begins = ends - padded
    indices = tl.arange(0, BM)
    tl.store(Sorted + indices, S, indices < MAX)
    blocks = tl.arange(0, BE)
    locations = blocks * BLOCK
    owner = (locations[None, :] >= begins[:, None]) & (locations[None, :] < ends[:, None])
    owner_ids = tl.sum(tl.where(owner, experts[:, None] + 1, 0), axis=0) - 1
    tl.store(Experts + blocks, owner_ids, blocks < MAX // BLOCK)
    tl.store(Padded, tl.sum(padded))
    tl.debug_barrier()
    ranks = tl.cumsum(matches.to(tl.int32), axis=1) - 1
    destinations = begins[:, None] + ranks
    tl.store(Sorted + destinations, slots[None, :], matches)
