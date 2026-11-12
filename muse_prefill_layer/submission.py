# okc: problem=muse_prefill_layer
"""SGLang's decoder layer for Muse Spark Open, as one H200 runs it under TP8/EP8, at SGLang commit
a55ab7d5ae1bba59ac351e55026fe7f7644ecc5f (MuseSpark1xDecoderLayer.forward), with the same modules:

  qkv_gate projection              bf16_gemm_dispatch (cuBLAS on H200)
  FusedQKNorm (Triton)             per-head RMSNorm of q and k, read from the fused projection,
                                   with the interleaved RoPE fused in on sliding layers
  cache store, q to FP8            as FlashAttentionBackend and set_kv_buffer do for an FP8 cache
                                   (cast, then the store_cache JIT kernel at the step's rows)
  flash_attn_with_kvcache (FA3)    paged, causal, window (2047, 0) on sliding layers
  FusedRMSNormGate (JIT)           per-head output norm x sigmoid(gate)
  o projection                     bf16_gemm_dispatch; then the all-reduce (identity here)
  FusedPostNormResidualPreNorm     post-attention norm + residual gate + pre-FFN norm (JIT);
                                   it also writes the FP32 router input up to 8 tokens
  router                           FP32 F.linear up to 8 tokens, above that two BF16 GEMMs
                                   on a hi/lo split of the weight (MuseSpark1xRouter); on a
                                   side stream, overlapping the pooler
  FusedMuseSparkRouting (JIT)      sigmoid, biased top-8, renormalize (on the side stream)
  pooler                           bf16_gemm_dispatch, replicated on every GPU
  pre-expert norm                  MuseSpark1xRMSNorm (F.rms_norm)
  experts                          moe_align_block_size, two moe_wna16_marlin_gemm (W4A16),
                                   silu_and_mul, Triton post-expert norm weighted sum
                                   (MuseSpark1xFp4Experts._forward_marlin); then the
                                   all-reduce (identity here)
  unpooler                         up to 8 tokens, this GPU's 1/8 slice of the weight, then
                                   the all-gather (identity here: zeros in the other columns);
                                   above that, the whole weight (bf16_gemm_dispatch)
  FusedPostNormResidualPreNorm     post-FFN norm + residual gate + the next layer's pre-attention
                                   norm (JIT)

Differs from SGLang:
  * The step's positions, cache rows, FA3 lengths and FA3 scheduler metadata come in with the
    inputs; SGLang builds them once per step for all layers.
  * All-reduces are the identity; the all-gather is a zero pad of the unpooler's output."""

from typing import TypedDict

import types

import torch
import torch.nn.functional as F
from sgl_kernel.flash_attn import flash_attn_with_kvcache
from sgl_kernel.scalar_type import scalar_types
from torch import nn

from sglang.kernels.ops.activation.activation import silu_and_mul
from sglang.kernels.ops.kvcache.kvcache import store_cache
from sglang.kernels.ops.moe.moe_wna16_marlin import moe_wna16_marlin_gemm
from sglang.kernels.ops.muse_spark_v12.fused_muse_spark_routing import FusedMuseSparkRouting
from sglang.kernels.ops.muse_spark_v12.fused_norm_gate import FusedRMSNormGate
from sglang.kernels.ops.muse_spark_v12.fused_norm_residual import FusedPostNormResidualPreNorm
from sglang.kernels.ops.muse_spark_v12.fused_post_expert_norm_weighted_sum_triton import (
    fused_post_expert_norm_weighted_sum,
)
from sglang.kernels.ops.muse_spark_v12.fused_qk_norm import FusedQKNorm
from sglang.srt.layers.moe.fused_moe_triton import moe_align_block_size
from sglang.srt.layers.muse_spark_layers import (
    MuseSpark1xNormPreservingResidualConnection,
    MuseSpark1xRMSNorm,
    MuseSpark1xRMSNormWithInputScale,
)
from sglang.srt.layers.quantization.marlin_utils import marlin_make_workspace
from sglang.srt.layers.quantization.marlin_utils_fp4 import prepare_moe_nvfp4_layer_for_marlin
from sglang.srt.layers.quantization.unquant import _bf16_gemm_dispatch_impl


class Weights(TypedDict):
    qkv_gate: torch.Tensor            # bfloat16 [2304, 8192]: q 16x64, k 2x64, v 2x64, output gate 16x64
    o: torch.Tensor                   # bfloat16 [8192, 1024]
    attn_residual_gate: torch.Tensor  # bfloat16 [8192], the post-attention residual gate's weight
    pre_ffn_norm: torch.Tensor        # bfloat16 [8192] (applied as 1 + w)
    router: torch.Tensor              # float32 [256, 8192]
    router_bias: torch.Tensor         # float32 [256], added to the scores only to choose experts
    pooler: torch.Tensor              # bfloat16 [4096, 8192]
    pre_expert_norm: torch.Tensor     # bfloat16 [4096] (applied as 1 + w)
    w13: torch.Tensor                 # uint8 [32, 8192, 2048]: this GPU's experts, gate then up rows, packed E2M1
    w13_sf: torch.Tensor              # float8_e4m3fn [32, 8192, 256]
    w13_gs: torch.Tensor              # float32 [32]
    w2: torch.Tensor                  # uint8 [32, 4096, 2048]
    w2_sf: torch.Tensor               # float8_e4m3fn [32, 4096, 256]
    w2_gs: torch.Tensor               # float32 [32]
    post_expert_norm: torch.Tensor    # bfloat16 [4096] (applied as 1 + w, before normalizing)
    unpooler: torch.Tensor            # bfloat16 [8192, 4096]; up to 8 tokens only rows 0-1023 (this GPU's slice)
    post_ffn_norm: torch.Tensor       # bfloat16 [8192] (applied as 1 + w, before normalizing)
    ffn_residual_gate: torch.Tensor   # bfloat16 [8192]
    next_pre_attn_norm: torch.Tensor  # bfloat16 [8192], the next layer's input norm (applied as 1 + w)


class Config(TypedDict):
    batch: int            # B sequences
    q_len: int            # L: 1 for decode, 8 for a verify block (causal inside it); 200-16384 on muse_prefill_layer
    tokens: int           # T = B * L, request-major
    hidden: int           # 8192
    num_q_heads: int      # 16 per GPU under TP8
    num_kv_heads: int     # 2
    head_dim: int         # 64
    page_size: int        # 1 in the scored cases (as SGLang serves Muse on H200), 64 in some tests
    window: int           # 2048 on sliding-window layers (a token and the 2,047 before it); 0 = all
    rope_theta: float     # 500000 on sliding-window layers (interleaved RoPE); 0 = no RoPE
    qk_scale_factor: float
    sm_scale: float       # 1/8
    kv_scale: float       # the cache scales' value: 1 in the scored cases (SGLang leaves them unset)
    n_experts: int        # 256
    top_k: int            # 8
    local_experts: int    # 32: this GPU computes experts local_start .. local_start + 31


class Inputs(TypedDict):
    stream: torch.Tensor      # float32 [T, 8192], the residual stream
    attn_input: torch.Tensor  # bfloat16 [T, 8192], the normalized stream (the previous layer's boundary wrote it)
    k_cache: torch.Tensor     # float8_e4m3fn [pages, page_size, 2, 64], written in place
    v_cache: torch.Tensor     # float8_e4m3fn [pages, page_size, 2, 64], written in place
    positions: torch.Tensor   # int64 [T], each new token's position
    cache_loc: torch.Tensor   # int64 [T], each new token's flat cache row
    cache_seqlens: torch.Tensor  # int32 [B], seq_lens + L
    cu_seqlens_q: torch.Tensor   # int32 [B+1], 0, L, 2L, ...
    scheduler_metadata: torch.Tensor | None  # int32, FA3's tile schedule on full-attention decode steps
    k_scale: torch.Tensor     # float32 [], key = e4m3 * k_scale
    v_scale: torch.Tensor     # float32 []
    page_table: torch.Tensor  # int32 [B, max_pages], physical page of each logical page (-1 unused)
    seq_lens: torch.Tensor    # int32 [B], cached tokens per request; the block starts at this position
    local_start: torch.Tensor # int32 [1], this GPU's first expert (a multiple of 32; its weights are w13[0], ...)
    config: Config


HIDDEN = 8192
RMS_EPS = 1e-5
POST_NORM_EPS = 1e-8
FP8 = torch.float8_e4m3fn
N_EXPERTS = 256
TOP_K = 8
LOCAL_EXPERTS = 32
D_EXPERT = 4096
# MuseSpark1xRouter switches to the split BF16 GEMMs above this many tokens.
ROUTER_FP32_MAX_TOKENS = 8
# Up to this many tokens the unpooler runs on this GPU's slice and is all-gathered.
SLICED_UNPOOL_MAX_TOKENS = 8
UNPOOL_SLICE = HIDDEN // 8
# MuseSpark1xFp4Experts' candidates, smallest first.
BLOCK_SIZE_M_CANDIDATES = (8, 16, 32, 48, 64)
ROPE_POSITIONS = 1 << 21
ROPE_THETA = 500_000.0


def _gemm(x, w):
    return _bf16_gemm_dispatch_impl(x, w, None)


def _param(w):
    return nn.Parameter(w.contiguous(), requires_grad=False)


def _norm(cls, w, size, eps, **kw):
    """A Muse norm module holding w (applied as 1 + w), as the checkpoint loads it."""
    norm = cls(size, eps=eps, has_weight=w is not None, zero_centered_gamma=True, **kw)
    if w is not None:
        norm.weight = _param(w)
        norm.precompute_effective_weight()
    return norm


def _residual(w):
    gate = MuseSpark1xNormPreservingResidualConnection(HIDDEN, temperature=0.3)
    gate.gate = _param(w)
    gate.precompute_coefficients()
    return gate


def _marlin_bank(weights):
    """The local experts in Marlin's layout, via SGLang's own loader step."""
    device = weights["w13"].device
    layer = types.SimpleNamespace(
        quant_config=types.SimpleNamespace(group_size=16),
        moe_runner_config=types.SimpleNamespace(is_gated=True),
        intermediate_size_per_partition=weights["w13"].shape[1] // 2,
        params_dtype=torch.bfloat16,
        w13_weight=weights["w13"].contiguous(),
        w2_weight=weights["w2"].contiguous(),
        w13_weight_scale=weights["w13_sf"].contiguous(),
        w2_weight_scale=weights["w2_sf"].contiguous(),
        w13_weight_scale_2=weights["w13_gs"].contiguous(),
        w2_weight_scale_2=weights["w2_gs"].contiguous(),
    )
    prepare_moe_nvfp4_layer_for_marlin(layer)
    expert_map = torch.full((N_EXPERTS,), -1, dtype=torch.int32, device=device)
    expert_map[:LOCAL_EXPERTS] = torch.arange(LOCAL_EXPERTS, dtype=torch.int32, device=device)
    return {
        "w13": layer.w13_weight.data, "w13_s": layer.w13_weight_scale.data, "w13_gs": layer.w13_weight_scale_2.data,
        "w2": layer.w2_weight.data, "w2_s": layer.w2_weight_scale.data, "w2_gs": layer.w2_weight_scale_2.data,
        "expert_map": expert_map, "size_n": layer.intermediate_size_per_partition,
        "workspace": marlin_make_workspace(device, max_blocks_per_sm=4),
    }


def _prepare_layer(weights):
    device = weights["qkv_gate"].device
    router = weights["router"]
    hi = router.to(torch.bfloat16)
    lo = (router - hi.float()).to(torch.bfloat16)
    # [2^21, 64] float32 cos | sin, as SGLang's RotaryEmbedding builds it.
    inv_freq = 1.0 / (ROPE_THETA ** (torch.arange(0, 64, 2, dtype=torch.float, device=device) / 64))
    freqs = torch.einsum("i,j -> ij", torch.arange(ROPE_POSITIONS, dtype=torch.float, device=device), inv_freq)
    pre_ffn = _norm(MuseSpark1xRMSNorm, weights["pre_ffn_norm"], HIDDEN, RMS_EPS)
    next_pre_attn = _norm(MuseSpark1xRMSNorm, weights["next_pre_attn_norm"], HIDDEN, RMS_EPS)
    post_attn = _norm(MuseSpark1xRMSNorm, None, HIDDEN, POST_NORM_EPS, is_post_norm=True,
                      post_norm_gain_center_type="one", residual_id=1)
    post_ffn = _norm(MuseSpark1xRMSNormWithInputScale, weights["post_ffn_norm"], HIDDEN, POST_NORM_EPS,
                     is_post_norm=True, post_norm_gain_center_type="one", residual_id=2)
    post_expert = _norm(MuseSpark1xRMSNormWithInputScale, weights["post_expert_norm"], D_EXPERT, POST_NORM_EPS,
                        is_post_norm=True, post_norm_gain_center_type="one", residual_id=2)
    state = {
        "dense": {name: weights[name].contiguous() for name in ("qkv_gate", "o", "pooler", "unpooler")},
        "router": router.contiguous(),
        "router_hi_t": hi.t().contiguous(),
        "router_lo_t": lo.t().contiguous(),
        "routing": FusedMuseSparkRouting(expert_bias=weights["router_bias"].float().contiguous(), route_scale=1.0,
                                         router_output_multiplier=0.1767766952966369,
                                         router_score_division_eps=1e-15),
        "cos_sin": torch.cat((freqs.cos(), freqs.sin()), dim=-1),
        "qk_norm": FusedQKNorm(head_dim=64, num_q_heads=16, num_kv_heads=2, eps=RMS_EPS),
        "o_norm_gate": FusedRMSNormGate(MuseSpark1xRMSNorm(64, eps=RMS_EPS, has_weight=False), True, 64),
        "attn_boundary": FusedPostNormResidualPreNorm(post_attn, _residual(weights["attn_residual_gate"]), pre_ffn),
        "ffn_boundary": FusedPostNormResidualPreNorm(post_ffn, _residual(weights["ffn_residual_gate"]), next_pre_attn),
        "pre_expert_norm": _norm(MuseSpark1xRMSNorm, weights["pre_expert_norm"], D_EXPERT, RMS_EPS),
        # MuseSpark1xFp4Experts binds the post-expert norm's effective (1 + w) weight.
        "post_expert_w": post_expert._get_effective_weight().detach().to(torch.bfloat16),
        "bank": _marlin_bank(weights),
        "side_stream": torch.cuda.Stream(device),
        "_modules": (pre_ffn, next_pre_attn, post_attn, post_ffn, post_expert),
    }
    del freqs
    torch.cuda.synchronize()
    return state


def _packed_experts(x, topk_weights, topk_ids, state, raw_output=False):
    """MuseSpark1xFp4Experts._marlin_expert_outputs and its post-expert-norm finalize."""
    bank = state["bank"]
    num_tokens, size_k = x.shape
    size_n = bank["size_n"]
    gemm1_n = 2 * size_n
    local_ids = bank["expert_map"][topk_ids].contiguous()
    topk_weights = topk_weights.to(torch.float32).contiguous()
    for block_size_m in BLOCK_SIZE_M_CANDIDATES:
        if num_tokens * TOP_K / LOCAL_EXPERTS / block_size_m < 0.9:
            break
    sorted_ids, block_experts, n_padded = moe_align_block_size(local_ids, block_size_m, LOCAL_EXPERTS, ignore_invalid_expert=True)
    cache2 = torch.empty((num_tokens * TOP_K, size_n), device=x.device, dtype=x.dtype)
    cache13 = torch.empty((num_tokens * TOP_K * max(gemm1_n, size_k),), device=x.device, dtype=x.dtype)
    cache1 = cache13[: num_tokens * TOP_K * gemm1_n].view(-1, gemm1_n)
    cache3 = cache13[: num_tokens * TOP_K * size_k].view(-1, size_k)
    common = dict(moe_block_size=block_size_m, mul_topk_weights=False, is_ep=False,
                  b_q_type=scalar_types.float4_e2m1f, is_k_full=True, use_atomic_add=False,
                  use_fp32_reduce=True, is_zp_float=False)
    ws = bank["workspace"]
    cache1 = moe_wna16_marlin_gemm(
        x, cache1, bank["w13"], None, bank["w13_s"], bank["w13_gs"], None, None, None, ws,
        sorted_ids, block_experts, n_padded, topk_weights, top_k=TOP_K,
        size_m=num_tokens, size_n=gemm1_n, size_k=size_k, **common)
    silu_and_mul(cache1.view(-1, gemm1_n), cache2, expert_ids=local_ids.view(-1), expert_step=1)
    y = moe_wna16_marlin_gemm(
        cache2, cache3, bank["w2"], None, bank["w2_s"], bank["w2_gs"], None, None, None, ws,
        sorted_ids, block_experts, n_padded, topk_weights, top_k=1,
        size_m=num_tokens * TOP_K, size_n=size_k, size_k=size_n, **common).view(num_tokens, TOP_K, size_k)
    if raw_output:
        return y
    return fused_post_expert_norm_weighted_sum(
        y, topk_weights, norm_weight=state["post_expert_w"], eps=POST_NORM_EPS, weight_first=True,
        slot_expert_ids=local_ids)


def kernel_fn(data, state):
    cfg = data["config"]
    batch, length, tokens = cfg["batch"], cfg["q_len"], cfg["tokens"]
    hq, hkv, d = cfg["num_q_heads"], cfg["num_kv_heads"], cfg["head_dim"]
    dense = state["dense"]
    # Attention: MuseSpark1xAttention.forward.
    qkv = _gemm(data["attn_input"], dense["qkv_gate"])
    if cfg["rope_theta"]:
        q, k = state["qk_norm"](qkv, data["positions"], state["cos_sin"])
    else:
        q, k = state["qk_norm"](qkv)
    v = qkv[:, hq * d + hkv * d: hq * d + 2 * hkv * d]
    gate = qkv[:, hq * d + 2 * hkv * d:]
    k_cache, v_cache = data["k_cache"], data["v_cache"]
    scaled = cfg["kv_scale"] != 1.0
    k = k.reshape(tokens, hkv * d)
    if scaled:
        k, v = k / data["k_scale"], v / data["v_scale"]
    # The pool keeps FP8 rows as bytes; slot 0 is a live row here (SGLang reserves it for padding).
    store_cache(k.to(FP8).view(torch.uint8), v.to(FP8).view(torch.uint8),
                k_cache.view(-1, hkv * d).view(torch.uint8), v_cache.view(-1, hkv * d).view(torch.uint8),
                data["cache_loc"], reserved_skip_index=-1)
    window = cfg["window"]
    descale = {}
    if scaled:
        descale = dict(k_descale=data["k_scale"].reshape(1, 1).expand(batch, hkv),
                       v_descale=data["v_scale"].reshape(1, 1).expand(batch, hkv))
    out = flash_attn_with_kvcache(
        q=q.view(tokens, hq, d).to(FP8),
        k_cache=k_cache,
        v_cache=v_cache,
        page_table=data["page_table"],
        cache_seqlens=data["cache_seqlens"],
        cu_seqlens_q=data["cu_seqlens_q"],
        max_seqlen_q=length,
        softmax_scale=cfg["sm_scale"] * cfg["qk_scale_factor"],
        causal=True,
        window_size=(window - 1, 0) if window else (-1, -1),
        scheduler_metadata=data["scheduler_metadata"],
        **descale,
    )
    if isinstance(out, tuple):
        out = out[0]
    o = state["o_norm_gate"](out.reshape(tokens, hq * d).to(torch.bfloat16), gate)
    attn_output = _gemm(o, dense["o"])  # all-reduce: identity

    # Post-attention boundary, router and MoE block: MuseSpark1xMoE._forward_experts.
    split_router = tokens > ROUTER_FP32_MAX_TOKENS
    ffn_input, stream, router_input = state["attn_boundary"](
        data["stream"], attn_output, branch_dtype=torch.bfloat16, router_dtype=None if split_router else torch.float32)
    # The router and routing run on a side stream, overlapping the pooler on the main one.
    main_stream, side_stream = torch.cuda.current_stream(), state["side_stream"]
    side_stream.wait_stream(main_stream)
    expert_input = _gemm(ffn_input, dense["pooler"])
    with torch.cuda.stream(side_stream):
        if split_router:
            parts = torch.mm(ffn_input, state["router_packed"], out_dtype=torch.float32)
            logits = parts[:, :N_EXPERTS] + parts[:, N_EXPERTS:]
        else:
            logits = F.linear(router_input, state["router"])
        topk_weights, topk_ids = state["routing"](None, logits, TOP_K, True)
    main_stream.wait_stream(side_stream)
    if not torch.cuda.is_current_stream_capturing():
        for tensor in (topk_weights, topk_ids):
            tensor.record_stream(main_stream)
    # This GPU's experts start at local_start: as SGLang's expert_map on that EP rank.
    local_ids = (topk_ids - data["local_start"]) % N_EXPERTS
    # all-reduce: identity
    expert_output = _experts(state["pre_expert_norm"](expert_input), topk_weights, local_ids, state)
    if tokens <= SLICED_UNPOOL_MAX_TOKENS:
        unpooled = _gemm(expert_output, dense["unpooler"][:UNPOOL_SLICE])
        ffn_output = F.pad(unpooled, (0, HIDDEN - UNPOOL_SLICE))  # all-gather: identity
    else:
        ffn_output = _gemm(expert_output, dense["unpooler"])

    # Post-FFN boundary, with the next layer's pre-attention norm.
    next_attn_input, stream, _ = state["ffn_boundary"](stream, ffn_output, branch_dtype=torch.bfloat16)
    return next_attn_input, stream, topk_ids

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

class _DenseWeight:
    def __init__(self, raw, transposed, operation, workspace, choices):
        self.raw = raw
        self.transposed = transposed
        self.operation = operation
        self.workspace = workspace
        self.choices = choices

    def __getitem__(self, index):
        return _DenseWeight(self.raw[index], self.transposed, self.operation, self.workspace, {})


def _gemm(x, w):
    if x.shape[0] < 512:
        return _bf16_gemm_dispatch_impl(x, w.raw, None)
    output = torch.empty((x.shape[0], w.raw.shape[0]), device=x.device, dtype=torch.bfloat16)
    w.operation.lt_matmul(x, w.transposed, output, w.workspace, w.choices.get(x.shape[0], 0))
    return output


def _prepare_grouped_base(weights):
    import statistics
    state = _prepare_layer(weights)
    operation = load_inline(
        name="prefill_projection_lt", cpp_sources="void lt_matmul(torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, int64_t);",
        cuda_sources=CUDA_SOURCE, functions=["lt_matmul"], extra_cuda_cflags=["-O3"],
        extra_ldflags=["-lcublasLt"], verbose=False,
    )
    device = weights["qkv_gate"].device
    workspace = torch.empty((32 * 1024 * 1024,), device=device, dtype=torch.uint8)
    eviction = torch.empty((128 * 1024 * 1024,), device=device, dtype=torch.uint8)
    begin = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    dense = {}
    for name, raw in state["dense"].items():
        transposed = raw.T.contiguous()
        n, k = raw.shape
        choices = {}
        for m in (753, 3012, 16384):
            x = torch.randn((m, k), device=device, dtype=torch.bfloat16)
            y = torch.empty((m, n), device=device, dtype=torch.bfloat16)
            timings = []
            for choice in range(16):
                try:
                    operation.lt_matmul(x, transposed, y, workspace, choice)
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph):
                        operation.lt_matmul(x, transposed, y, workspace, choice)
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
                verified = []
                for _, choice in sorted(timings)[:3]:
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph):
                        operation.lt_matmul(x, transposed, y, workspace, choice)
                    samples = []
                    for repeat in range(11):
                        eviction.add_(1)
                        begin.record()
                        graph.replay()
                        end.record()
                        end.synchronize()
                        samples.append(begin.elapsed_time(end))
                    verified.append((statistics.median(samples), choice))
                choices[m] = min(verified)[1]
        dense[name] = _DenseWeight(raw, transposed, operation, workspace, choices)
    state["dense"] = dense
    state["router_packed"] = torch.cat((state["router_hi_t"], state["router_lo_t"]), dim=1).contiguous()
    torch.cuda.synchronize()
    return state

GROUPED_ROUTE_THRESHOLD = 256

import triton
import triton.language as tl


@triton.jit
def _gmm_fp4(code):
    magnitude = (code & 7).to(tl.int32)
    exponent = (0x3f000000 + (magnitude >> 1) * 0x00800000).to(tl.float32, bitcast=True)
    value = tl.where(magnitude < 2, magnitude * 0.5, exponent * (1.0 + (magnitude & 1) * 0.5))
    return tl.where((code & 8) != 0, -value, value)


@triton.jit
def _grouped_ends(Keys, Ends, S: tl.constexpr, E: tl.constexpr, STEPS: tl.constexpr):
    expert = tl.arange(0, E)
    low = tl.zeros((E,), tl.int32)
    high = tl.full((E,), S, tl.int32)
    for step in range(STEPS):
        middle = (low + high) // 2
        value = tl.load(Keys + middle, middle < S, E)
        go_right = value <= expert
        low = tl.where(go_right, middle + 1, low)
        high = tl.where(go_right, high, middle)
    tl.store(Ends + expert, low)


@triton.jit
def _grouped_inverse(Perm, Inverse, S: tl.constexpr, B: tl.constexpr):
    i = tl.program_id(0) * B + tl.arange(0, B)
    original = tl.load(Perm + i, i < S, 0)
    tl.store(Inverse + original, i, i < S)


@triton.jit
def _decode_for_grouped(W, SF, GS, Ends, Decoded, R: tl.constexpr, K: tl.constexpr,
                        E: tl.constexpr, B: tl.constexpr):
    experts = tl.arange(0, E)
    ends = tl.load(Ends + experts)
    begins = tl.load(Ends + experts - 1, experts > 0, 0)
    active = ends > begins
    ranks = tl.cumsum(active.to(tl.int32), axis=0) - 1
    count = tl.sum(active.to(tl.int32), axis=0)
    blocks: tl.constexpr = tl.cdiv(R * K, B)
    work = tl.program_id(0)
    offsets = tl.arange(0, B)
    while work < count * blocks:
        rank = work // blocks
        expert = rank
        if count < E:
            expert = tl.sum(tl.where(active & (ranks == rank), experts, 0), axis=0)
        index = (work % blocks) * B + offsets
        row = index // K
        column = index % K
        valid = index < R * K
        packed = tl.load(W + expert * R * (K // 2) + row * (K // 2) + column // 2, valid, 0)
        code = (packed >> ((column & 1) * 4)) & 15
        sf = tl.load(SF.to(tl.pointer_type(tl.uint8)) + expert * R * (K // 16) + row * (K // 16) + column // 16,
                     valid, 0).to(tl.float8e4nv, bitcast=True).to(tl.float32)
        value = _gmm_fp4(code) * sf * tl.load(GS + expert)
        tl.store(Decoded + expert * R * K + index, value, valid)
        work += tl.num_programs(0)


@triton.jit
def _gather_for_grouped(X, Perm, Ends, Grouped, H: tl.constexpr, E: tl.constexpr, B: tl.constexpr):
    count = tl.load(Ends + E - 1)
    row = tl.program_id(0)
    offsets = tl.arange(0, B)
    while row < count:
        original = tl.load(Perm + row) // 8
        value = tl.load(X + original * H + offsets, offsets < H, 0)
        tl.store(Grouped + row * H + offsets, value, offsets < H)
        row += tl.num_programs(0)


@triton.jit
def _activate_grouped(GateUp, Ends, Inter, I: tl.constexpr, E: tl.constexpr,
                      GS: tl.constexpr, B: tl.constexpr):
    count = tl.load(Ends + E - 1)
    row = tl.program_id(0)
    offsets = tl.arange(0, B)
    while row < count:
        gate = tl.load(GateUp + row * GS + offsets, offsets < I, 0).to(tl.float32)
        up = tl.load(GateUp + row * GS + I + offsets, offsets < I, 0).to(tl.float32)
        value = gate / (1.0 + tl.exp(-gate)) * up
        tl.store(Inter + row * I + offsets, value, offsets < I)
        row += tl.num_programs(0)


@triton.jit
def _finish_grouped(Y, Cold, Inverse, IDs, AllIDs, Scores, Weight, Out, H: tl.constexpr,
                    YS: tl.constexpr, B: tl.constexpr, USE_COLD: tl.constexpr):
    token = tl.program_id(0)
    offsets = tl.arange(0, B)
    weight = tl.load(Weight + offsets, offsets < H, 0).to(tl.float32)
    acc = tl.zeros((B,), tl.float32)
    for route in tl.static_range(8):
        slot = token * 8 + route
        expert = tl.load(IDs + slot)
        if tl.load(AllIDs + slot) >= 0:
            if expert >= 0:
                row = tl.load(Inverse + slot)
                value = tl.load(Y + row * YS + offsets, offsets < H, 0).to(tl.float32)
            else:
                value = tl.full((B,), 0.0, tl.float32)
                if USE_COLD:
                    value = tl.load(Cold + slot * H + offsets, offsets < H, 0).to(tl.float32)
            value *= weight
            inverse = tl.rsqrt(tl.sum(value * value, axis=0) / H + 1.0e-8)
            acc += value * inverse * tl.load(Scores + slot)
    tl.store(Out + token * H + offsets, acc, offsets < H)


def prepare(weights):
    raw = {key: weights[key].clone() for key in ("w13", "w13_sf", "w13_gs", "w2", "w2_sf", "w2_gs")}
    state = _prepare_grouped_base(weights)
    operation = getattr(F, "grouped_mm", getattr(torch, "_grouped_mm", None))
    if operation is None:
        raise RuntimeError("Grouped BF16 matrix multiplication is unavailable")
    state["raw_grouped"] = raw
    state["grouped_mm"] = operation
    state["grouped_programs"] = torch.cuda.get_device_properties(weights["w13"].device).multi_processor_count * 4
    return state


def _experts(x, topk_weights, topk_ids, state):
    tokens, hidden = x.shape
    if tokens < 1024:
        return _packed_experts(x, topk_weights, topk_ids, state)
    raw = state["raw_grouped"]
    intermediate = raw["w2"].shape[-1] * 2
    all_ids = state["bank"]["expert_map"][topk_ids].contiguous()
    slots = tokens * TOP_K
    block_size = 256
    blocks = triton.cdiv(slots, block_size)
    counts = torch.empty((blocks, LOCAL_EXPERTS), device=x.device, dtype=torch.int32)
    offsets = torch.empty_like(counts)
    ends = torch.empty((LOCAL_EXPERTS,), device=x.device, dtype=torch.int32)
    permutation = torch.empty((slots,), device=x.device, dtype=torch.int32)
    inverse = torch.empty_like(permutation)
    local_ids = torch.empty_like(all_ids)
    cold_ids = torch.empty_like(topk_ids)
    hybrid = tokens < 4096
    _count_grouped_routes[(blocks,)](
        all_ids, counts, slots, LOCAL_EXPERTS, block_size, num_warps=4,
    )
    _prefix_grouped_routes[(1,)](
        counts, offsets, ends, blocks, LOCAL_EXPERTS,
        triton.next_power_of_2(blocks), GROUPED_ROUTE_THRESHOLD if hybrid else 0,
        num_warps=8,
    )
    _scatter_grouped_routes[(blocks,)](
        all_ids, topk_ids, offsets, ends, permutation, inverse, local_ids, cold_ids,
        slots, LOCAL_EXPERTS, block_size, hybrid, num_warps=8,
    )
    cold = _packed_experts(x, topk_weights, cold_ids, state, raw_output=True) if hybrid else None
    weight13 = torch.empty((LOCAL_EXPERTS, 2 * intermediate, hidden), device=x.device, dtype=torch.bfloat16)
    weight2 = torch.empty((LOCAL_EXPERTS, hidden, intermediate), device=x.device, dtype=torch.bfloat16)
    programs = state["grouped_programs"]
    _decode_for_grouped[(programs,)](
        raw["w13"], raw["w13_sf"], raw["w13_gs"], ends, weight13,
        2 * intermediate, hidden, LOCAL_EXPERTS, 4096, num_warps=8,
    )
    _decode_for_grouped[(programs,)](
        raw["w2"], raw["w2_sf"], raw["w2_gs"], ends, weight2,
        hidden, intermediate, LOCAL_EXPERTS, 4096, num_warps=8,
    )
    grouped_x = torch.empty((slots, hidden), device=x.device, dtype=torch.bfloat16)
    _gather_for_grouped[(programs,)](x, permutation, ends, grouped_x, hidden, LOCAL_EXPERTS,
                                    triton.next_power_of_2(hidden), num_warps=4)
    gate_up = state["grouped_mm"](grouped_x, weight13.transpose(-2, -1), offs=ends)
    inter = torch.empty((slots, intermediate), device=x.device, dtype=torch.bfloat16)
    _activate_grouped[(programs,)](gate_up, ends, inter, intermediate, LOCAL_EXPERTS,
                                 gate_up.stride(0), triton.next_power_of_2(intermediate), num_warps=4)
    y = state["grouped_mm"](inter, weight2.transpose(-2, -1), offs=ends)
    out = torch.empty((tokens, hidden), device=x.device, dtype=torch.bfloat16)
    _finish_grouped[(tokens,)](
        y, cold if cold is not None else y, inverse, local_ids, all_ids, topk_weights.contiguous(), state["post_expert_w"], out,
        hidden, y.stride(0), triton.next_power_of_2(hidden), cold is not None, num_warps=8,
    )
    return out

@triton.jit
def _split_busy_routes(IDs, GlobalIDs, Ends, Hot, Cold, S: tl.constexpr,
                        THRESHOLD: tl.constexpr, B: tl.constexpr):
    slot = tl.program_id(0) * B + tl.arange(0, B)
    expert = tl.load(IDs + slot, slot < S, -1)
    global_expert = tl.load(GlobalIDs + slot, slot < S, 255)
    end = tl.load(Ends + expert, (slot < S) & (expert >= 0), 0)
    begin = tl.load(Ends + expert - 1, (slot < S) & (expert > 0), 0)
    busy = (expert >= 0) & (end - begin >= THRESHOLD)
    tl.store(Hot + slot, tl.where(busy, expert, -1), slot < S)
    tl.store(Cold + slot, tl.where(busy, 255, global_expert), slot < S)


@triton.jit
def _count_grouped_routes(IDs, Counts, S: tl.constexpr, E: tl.constexpr, B: tl.constexpr):
    block = tl.program_id(0)
    slot = block * B + tl.arange(0, B)
    expert = tl.load(IDs + slot, slot < S, -1)
    counts = tl.histogram(expert, E, mask=(slot < S) & (expert >= 0) & (expert < E))
    tl.store(Counts + block * E + tl.arange(0, E), counts)


@triton.jit
def _prefix_grouped_routes(Counts, Offsets, Ends, BLOCKS: tl.constexpr,
                           E: tl.constexpr, BC: tl.constexpr, MIN_COUNT: tl.constexpr):
    blocks = tl.arange(0, BC)
    experts = tl.arange(0, E)
    counts = tl.load(Counts + blocks[:, None] * E + experts[None, :], blocks[:, None] < BLOCKS, 0)
    totals = tl.sum(counts, axis=0)
    keep = totals >= MIN_COUNT
    counts = tl.where(keep[None, :], counts, 0)
    totals = tl.where(keep, totals, 0)
    ends = tl.cumsum(totals, axis=0)
    offsets = tl.cumsum(counts, axis=0) - counts + (ends - totals)[None, :]
    tl.store(Offsets + blocks[:, None] * E + experts[None, :], offsets, blocks[:, None] < BLOCKS)
    tl.store(Ends + experts, ends)


@triton.jit
def _scatter_grouped_routes(IDs, GlobalIDs, Offsets, Ends, Perm, Inverse, Hot, Cold,
                            S: tl.constexpr, E: tl.constexpr, B: tl.constexpr,
                            HYBRID: tl.constexpr):
    block = tl.program_id(0)
    slots = block * B + tl.arange(0, B)
    experts = tl.arange(0, E)
    ids = tl.load(IDs + slots, slots < S, -1)
    ends = tl.load(Ends + experts)
    begins = tl.load(Ends + experts - 1, experts > 0, 0)
    active = ends > begins
    matches = (experts[:, None] == ids[None, :]) & active[:, None] & (slots[None, :] < S)
    ranks = tl.cumsum(matches.to(tl.int32), axis=1) - 1
    offsets = tl.load(Offsets + block * E + experts)
    destinations = offsets[:, None] + ranks
    tl.store(Perm + destinations, slots[None, :], matches)
    inverse = tl.sum(tl.where(matches, destinations, 0), axis=0)
    tl.store(Inverse + slots, inverse, slots < S)
    is_hot = tl.sum(matches.to(tl.int32), axis=0) > 0
    tl.store(Hot + slots, tl.where(is_hot, ids, -1), slots < S)
    if HYBRID:
        global_ids = tl.load(GlobalIDs + slots, slots < S, 255)
        tl.store(Cold + slots, tl.where(is_hot, 255, global_ids), slots < S)
