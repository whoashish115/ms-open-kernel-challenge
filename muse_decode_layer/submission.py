# okc: problem=muse_decode_layer
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
import triton
import triton.language as tl
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



@triton.jit
def _layer_project(X, W, P, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
                   BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, S: tl.constexpr):
    mi = tl.program_id(0) * BM + tl.arange(0, BM)
    ni = tl.program_id(1) * BN + tl.arange(0, BN)
    si = tl.program_id(2)
    ki = si * tl.cdiv(K, S) + tl.arange(0, BK)
    acc = tl.zeros((BM, BN), tl.float32)
    for step in range(tl.cdiv(tl.cdiv(K, S), BK)):
        kk = ki + step * BK
        mask_k = (kk < K) & (kk < (si + 1) * tl.cdiv(K, S))
        a = tl.load(X + mi[:, None] * K + kk[None, :], (mi[:, None] < M) & mask_k[None, :], 0)
        b = tl.load(W + ni[None, :] * K + kk[:, None], (ni[None, :] < N) & mask_k[:, None], 0)
        acc = tl.dot(a, b, acc)
    tl.store(P + si * M * N + mi[:, None] * N + ni[None, :], acc,
             (mi[:, None] < M) & (ni[None, :] < N))


@triton.jit
def _layer_combine(P, Y, T: tl.constexpr, S: tl.constexpr, B: tl.constexpr):
    i = tl.program_id(0) * B + tl.arange(0, B)
    s = tl.arange(0, S)
    values = tl.load(P + s[:, None] * T + i[None, :], i[None, :] < T, 0)
    tl.store(Y + i, tl.sum(values, axis=0), i < T)


@triton.jit
def _layer_vector(X, W, Y, N: tl.constexpr, K: tl.constexpr, WK: tl.constexpr,
                  BN: tl.constexpr, BK: tl.constexpr):
    n = tl.program_id(0) * BN + tl.arange(0, BN)
    k = tl.arange(0, BK)
    a = tl.load(X + k, k < K, 0).to(tl.float32)
    b = tl.load(W + n[:, None] * WK + k[None, :],
                (n[:, None] < N) & (k[None, :] < K), 0).to(tl.float32)
    tl.store(Y + n, tl.sum(a[None, :] * b, axis=1), n < N)


def _gemm(x, w, padded=None):
    m, k = x.shape
    n = w.shape[0]
    if m == 1 and padded is not None:
        out = torch.empty((m, n), device=x.device, dtype=torch.bfloat16)
        width = 4 if k <= 1024 else 1
        _layer_vector[(triton.cdiv(n, width),)](
            x, padded, out, n, k, k + 64, width, triton.next_power_of_2(k),
            num_warps=8 if width == 4 else 4,
        )
        return out
    if m <= 8 and n <= 2304 and k >= 4096:
        out = torch.empty((m, n), device=x.device, dtype=torch.bfloat16)
        partial = torch.empty((8, m, n), device=x.device, dtype=torch.float32)
        _layer_project[(triton.cdiv(m, 16), triton.cdiv(n, 64), 8)](
            x, w, partial, m, n, k, 16, 64, 128, 8, num_warps=4, num_stages=3,
        )
        _layer_combine[(triton.cdiv(m * n, 256),)](partial, out, m * n, 8, 256)
        return out
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


def _prepare_decoder_base(weights):
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
        "dense_padded": {name: F.pad(weights[name], (0, 64)) for name in ("qkv_gate", "o", "pooler", "unpooler")},
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


def _experts(x, topk_weights, topk_ids, state, local_start):
    """MuseSpark1xFp4Experts._marlin_expert_outputs and its post-expert-norm finalize."""
    bank = state["bank"]
    num_tokens, size_k = x.shape
    size_n = bank["size_n"]
    gemm1_n = 2 * size_n
    topk_weights = topk_weights.to(torch.float32).contiguous()
    block_size_m = 8 if num_tokens < 16 else (16 if num_tokens <= 256 else 64)
    if num_tokens <= 32:
        slots = num_tokens * TOP_K
        maximum = triton.cdiv(slots + LOCAL_EXPERTS * (block_size_m - 1), block_size_m) * block_size_m
        local_ids = torch.empty_like(topk_ids, dtype=torch.int32)
        sorted_ids = torch.empty((maximum,), device=x.device, dtype=torch.int32)
        block_experts = torch.empty((maximum // block_size_m,), device=x.device, dtype=torch.int32)
        n_padded = torch.empty((1,), device=x.device, dtype=torch.int32)
        _align_local_routes[(1,)](topk_ids, local_ids, sorted_ids, block_experts, n_padded, local_start,
            slots, LOCAL_EXPERTS, block_size_m, triton.next_power_of_2(slots),
            maximum, triton.next_power_of_2(maximum), triton.next_power_of_2(maximum // block_size_m), num_warps=8)
    else:
        local_ids = torch.empty_like(topk_ids, dtype=torch.int32)
        _map_local_routes[(triton.cdiv(num_tokens * TOP_K, 256),)](
            topk_ids, local_start, local_ids, num_tokens * TOP_K, LOCAL_EXPERTS, 256, num_warps=4)
        sorted_ids, block_experts, n_padded = moe_align_block_size(local_ids, block_size_m, LOCAL_EXPERTS, ignore_invalid_expert=True)
    cache2 = torch.empty((num_tokens * TOP_K, size_n), device=x.device, dtype=x.dtype)
    cache13 = torch.empty((num_tokens * TOP_K * max(gemm1_n, size_k),), device=x.device, dtype=x.dtype)
    cache1 = cache13[: num_tokens * TOP_K * gemm1_n].view(-1, gemm1_n)
    cache3 = cache13[: num_tokens * TOP_K * size_k].view(-1, size_k)
    common = dict(moe_block_size=block_size_m, mul_topk_weights=False, is_ep=False,
                  b_q_type=scalar_types.float4_e2m1f, is_k_full=True, use_atomic_add=False,
                  use_fp32_reduce=True, is_zp_float=False)
    ws = bank["workspace"]
    cache1 = state.get("marlin_operation", moe_wna16_marlin_gemm)(
        x, cache1, bank["w13"], None, bank["w13_s"], bank["w13_gs"], None, None, None, ws,
        sorted_ids, block_experts, n_padded, topk_weights, top_k=TOP_K,
        size_m=num_tokens, size_n=gemm1_n, size_k=size_k, **common)
    silu_and_mul(cache1.view(-1, gemm1_n), cache2, expert_ids=local_ids.view(-1), expert_step=1)
    y = state.get("marlin_operation", moe_wna16_marlin_gemm)(
        cache2, cache3, bank["w2"], None, bank["w2_s"], bank["w2_gs"], None, None, None, ws,
        sorted_ids, block_experts, n_padded, topk_weights, top_k=1,
        size_m=num_tokens * TOP_K, size_n=size_k, size_k=size_n, **common).view(num_tokens, TOP_K, size_k)
    return _finalize(
        y, topk_weights, norm_weight=state["post_expert_w"], eps=POST_NORM_EPS, weight_first=True,
        slot_expert_ids=local_ids)


import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice as lib
from sgl_kernel.flash_attn import flash_attn_with_kvcache


@triton.jit
def _pack_qkv(X, Pos, Slots, KS, VS, Q, KC, VC,
              STRIDE: tl.constexpr, HQ: tl.constexpr, HKV: tl.constexpr,
              D: tl.constexpr, THETA: tl.constexpr):
    token = tl.program_id(0)
    head = tl.program_id(1)
    offsets = tl.arange(0, D)
    values = tl.load(X + token * STRIDE + head * D + offsets).to(tl.float32)
    inv = tl.rsqrt(tl.sum(values * values, axis=0) / D + 1.0e-5)
    normalized = values * inv
    if THETA > 0.0:
        partner = tl.load(X + token * STRIDE + head * D + (offsets ^ 1)).to(tl.float32) * inv
        position = tl.load(Pos + token).to(tl.float32)
        frequency = tl.exp(-tl.log(THETA) * (2.0 * (offsets // 2) / D))
        angle = position * frequency
        normalized = normalized * lib.cos(angle) + tl.where((offsets & 1) == 0, -partner, partner) * lib.sin(angle)
    normalized = normalized.to(tl.bfloat16).to(tl.float32)
    if head < HQ:
        tl.store(Q + token * HQ * D + head * D + offsets, normalized)
    else:
        kv_head = head - HQ
        slot = tl.load(Slots + token)
        k_scale = tl.load(KS)
        v_scale = tl.load(VS)
        tl.store(KC + slot * HKV * D + kv_head * D + offsets, normalized / k_scale)
        value = tl.load(X + token * STRIDE + (HQ + HKV + kv_head) * D + offsets).to(tl.float32)
        tl.store(VC + slot * HKV * D + kv_head * D + offsets, value / v_scale)


@triton.jit
def _finish_attn(A, X, Y, AS: tl.constexpr, XS: tl.constexpr,
                 HQ: tl.constexpr, HKV: tl.constexpr, D: tl.constexpr):
    token = tl.program_id(0)
    head = tl.program_id(1)
    offsets = tl.arange(0, D)
    value = tl.load(A + token * AS + head * D + offsets).to(tl.bfloat16).to(tl.float32)
    gate = tl.load(X + token * XS + (HQ + 2 * HKV + head) * D + offsets).to(tl.float32)
    inv = tl.rsqrt(tl.sum(value * value, axis=0) / D + 1.0e-5)
    output = value * inv / (1.0 + tl.exp(-gate))
    tl.store(Y + token * HQ * D + head * D + offsets, output)


def _fa3_attention_step(data, qkv):
    cfg = data["config"]
    batch, length = cfg["batch"], cfg["q_len"]
    hq, hkv, d = cfg["num_q_heads"], cfg["num_kv_heads"], cfg["head_dim"]
    tokens = batch * length
    qkv = qkv.reshape(tokens, -1)
    q = torch.empty((tokens, hq, d), device=qkv.device, dtype=torch.float8_e4m3fn)
    _pack_qkv[(tokens, hq + hkv)](
        qkv, data["positions"], data["cache_loc"], data["k_scale"], data["v_scale"],
        q, data["k_cache"], data["v_cache"], qkv.stride(0), hq, hkv, d,
        cfg["rope_theta"], num_warps=1,
    )
    descale = {}
    if cfg["kv_scale"] != 1.0:
        descale = {
            "k_descale": data["k_scale"].reshape(1, 1).expand(batch, hkv),
            "v_descale": data["v_scale"].reshape(1, 1).expand(batch, hkv),
        }
    window = cfg["window"]
    out = flash_attn_with_kvcache(
        q=q, k_cache=data["k_cache"], v_cache=data["v_cache"],
        page_table=data["page_table"], cache_seqlens=data["cache_seqlens"],
        cu_seqlens_q=data["cu_seqlens_q"], max_seqlen_q=length,
        softmax_scale=cfg["sm_scale"] * cfg["qk_scale_factor"], causal=True,
        window_size=(window - 1, 0) if window else (-1, -1),
        scheduler_metadata=data["scheduler_metadata"], **descale,
    )
    if isinstance(out, tuple):
        out = out[0]
    out = out.reshape(tokens, hq * d)
    result = torch.empty((tokens, hq * d), device=qkv.device, dtype=torch.bfloat16)
    _finish_attn[(tokens, hq)](
        out, qkv, result, out.stride(0), qkv.stride(0), hq, hkv, d, num_warps=1,
    )
    return result


def kernel_fn(data, state):
    cfg = data["config"]
    batch, length, tokens = cfg["batch"], cfg["q_len"], cfg["tokens"]
    hq, hkv, d = cfg["num_q_heads"], cfg["num_kv_heads"], cfg["head_dim"]
    dense = state["dense"]
    # Attention: MuseSpark1xAttention.forward.
    qkv = _gemm(data["attn_input"], dense["qkv_gate"], state["dense_padded"]["qkv_gate"])
    o = _attention_step(data, qkv)
    attn_output = _gemm(o, dense["o"], state["dense_padded"]["o"])  # all-reduce: identity

    # Post-attention boundary, router and MoE block: MuseSpark1xMoE._forward_experts.
    split_router = tokens > ROUTER_FP32_MAX_TOKENS
    ffn_input, stream, router_input = state["attn_boundary"](
        data["stream"], attn_output, branch_dtype=torch.bfloat16, router_dtype=None if split_router else torch.float32)
    # The router and routing run on a side stream, overlapping the pooler on the main one.
    main_stream, side_stream = torch.cuda.current_stream(), state["side_stream"]
    side_stream.wait_stream(main_stream)
    expert_input = _gemm(ffn_input, dense["pooler"], state["dense_padded"]["pooler"])
    with torch.cuda.stream(side_stream):
        logits = _router_logits(ffn_input, router_input, state)
        topk_weights, topk_ids = state["routing"](None, logits, TOP_K, True)
    main_stream.wait_stream(side_stream)
    if not torch.cuda.is_current_stream_capturing():
        for tensor in (topk_weights, topk_ids):
            tensor.record_stream(main_stream)
    # This GPU's experts start at local_start: as SGLang's expert_map on that EP rank.
    # all-reduce: identity
    expert_output = _experts(state["pre_expert_norm"](expert_input), topk_weights, topk_ids, state, data["local_start"])
    if tokens <= SLICED_UNPOOL_MAX_TOKENS:
        unpooled = _gemm(expert_output, dense["unpooler"][:UNPOOL_SLICE], state["dense_padded"]["unpooler"][:UNPOOL_SLICE])
        ffn_output = F.pad(unpooled, (0, HIDDEN - UNPOOL_SLICE))  # all-gather: identity
    else:
        ffn_output = _gemm(expert_output, dense["unpooler"], state["dense_padded"]["unpooler"])

    # Post-FFN boundary, with the next layer's pre-attention norm.
    next_attn_input, stream, _ = state["ffn_boundary"](stream, ffn_output, branch_dtype=torch.bfloat16)
    return next_attn_input, stream, topk_ids


@triton.jit
def _paged_partials(Q, KC, VC, Table, Pos, Lens, KS, VS, Partial, LSE,
                    HQ: tl.constexpr, HKV: tl.constexpr, D: tl.constexpr,
                    L: tl.constexpr, PAGES: tl.constexpr, PAGE: tl.constexpr,
                    NS: tl.constexpr, CHUNK: tl.constexpr, WINDOW: tl.constexpr,
                    SCALE: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr):
    batch = tl.program_id(0)
    head_block = tl.program_id(1)
    split = tl.program_id(2)
    group: tl.constexpr = HQ // HKV
    query_blocks: tl.constexpr = tl.cdiv(L * group, BM)
    head = head_block // query_blocks
    rows = (head_block % query_blocks) * BM + tl.arange(0, BM)
    dims = tl.arange(0, D)
    columns = tl.arange(0, BN)
    query_token = rows // group
    query_head = head * group + rows % group
    q = tl.load(Q + (batch * L + query_token[:, None]) * HQ * D + query_head[:, None] * D + dims[None, :],
                query_token[:, None] < L, 0.0)
    positions = tl.load(Pos + batch * L + query_token, query_token < L, 0)
    first_position = tl.load(Pos + batch * L)
    length = tl.load(Lens + batch)
    base = 0
    if WINDOW > 0:
        base = tl.maximum(first_position - WINDOW + 1, 0)
    start = base + split * CHUNK
    maximum = tl.full((BM,), -float("inf"), tl.float32)
    denominator = tl.zeros((BM,), tl.float32)
    accumulator = tl.zeros((BM, D), tl.float32)
    k_scale = tl.load(KS)
    v_scale = tl.load(VS)
    if start < length:
        for block in range(tl.cdiv(CHUNK, BN)):
            logical = start + block * BN + columns
            valid = (logical < length) & (logical < PAGES * PAGE)
            page = tl.load(Table + batch * PAGES + logical // PAGE, valid, 0)
            cache_row = page * PAGE + logical % PAGE
            key = tl.load(KC + cache_row[None, :] * HKV * D + head * D + dims[:, None],
                          valid[None, :] & (page[None, :] >= 0), 0.0)
            scores = tl.dot(q, key) * (SCALE * 1.4426950408889634) * k_scale
            visible = valid[None, :] & (page[None, :] >= 0) & (query_token[:, None] < L)
            visible = visible & (logical[None, :] <= positions[:, None])
            if WINDOW > 0:
                visible = visible & (logical[None, :] > positions[:, None] - WINDOW)
            scores = tl.where(visible, scores, -float("inf"))
            next_maximum = tl.maximum(maximum, tl.max(scores, axis=1))
            correction = tl.where(next_maximum == -float("inf"), 0.0, tl.exp2(maximum - next_maximum))
            probability = tl.where(visible, tl.exp2(scores - next_maximum[:, None]), 0.0)
            value = tl.load(VC + cache_row[:, None] * HKV * D + head * D + dims[None, :],
                            valid[:, None] & (page[:, None] >= 0), 0.0).to(tl.bfloat16)
            accumulator = accumulator * correction[:, None] + tl.dot(probability.to(tl.bfloat16), value)
            denominator = denominator * correction + tl.sum(probability, axis=1)
            maximum = next_maximum
    normalized = tl.where(denominator[:, None] > 0.0, accumulator / denominator[:, None], 0.0) * v_scale
    log_sum = tl.where(denominator > 0.0, maximum + tl.log2(denominator), -float("inf"))
    base_offset = ((batch * HKV + head) * NS + split) * (L * group)
    tl.store(Partial + (base_offset + rows[:, None]) * D + dims[None, :], normalized, query_token[:, None] < L)
    tl.store(LSE + base_offset + rows, log_sum, query_token < L)


@triton.jit
def _paged_reduce(Partial, LSE, X, Y, HQ: tl.constexpr, HKV: tl.constexpr,
                  D: tl.constexpr, L: tl.constexpr, NS: tl.constexpr,
                  XS: tl.constexpr, BS: tl.constexpr):
    token = tl.program_id(0)
    query_head = tl.program_id(1)
    batch = token // L
    query_token = token % L
    group: tl.constexpr = HQ // HKV
    kv_head = query_head // group
    row = query_token * group + query_head % group
    splits = tl.arange(0, BS)
    dims = tl.arange(0, D)
    base = ((batch * HKV + kv_head) * NS + splits) * (L * group) + row
    log_sum = tl.load(LSE + base, splits < NS, -float("inf"))
    maximum = tl.max(log_sum, axis=0)
    weights = tl.exp2(log_sum - maximum)
    parts = tl.load(Partial + base[:, None] * D + dims[None, :], splits[:, None] < NS, 0.0)
    output = tl.sum(parts * weights[:, None], axis=0) / tl.sum(weights, axis=0)
    output = output.to(tl.bfloat16).to(tl.float32)
    inverse = tl.rsqrt(tl.sum(output * output, axis=0) / D + 1.0e-5)
    gate = tl.load(X + token * XS + (HQ + 2 * HKV + query_head) * D + dims).to(tl.float32)
    output = output * inverse / (1.0 + tl.exp(-gate))
    tl.store(Y + token * HQ * D + query_head * D + dims, output)


def _attention_step(data, qkv):
    cfg = data["config"]
    batch, length = cfg["batch"], cfg["q_len"]
    if length > 1 and not (cfg["window"] > 0 and batch <= 2):
        return _fa3_attention_step(data, qkv)
    hq, hkv, d = cfg["num_q_heads"], cfg["num_kv_heads"], cfg["head_dim"]
    tokens = batch * length
    qkv = qkv.reshape(tokens, -1)
    q = torch.empty((tokens, hq, d), device=qkv.device, dtype=torch.float8_e4m3fn)
    _pack_qkv[(tokens, hq + hkv)](
        qkv, data["positions"], data["cache_loc"], data["k_scale"], data["v_scale"],
        q, data["k_cache"], data["v_cache"], qkv.stride(0), hq, hkv, d,
        cfg["rope_theta"], num_warps=1,
    )
    capacity = data["page_table"].shape[1] * cfg["page_size"]
    if cfg["window"]:
        chunk = 512 if length > 1 and batch > 2 else 128
        splits = triton.cdiv(cfg["window"] + length - 1, chunk)
    else:
        chunk = (512 if batch <= 2 else (2048 if batch >= 8 else 1024)) if capacity <= 131072 else 4096
        splits = triton.cdiv(capacity, chunk)
    group = hq // hkv
    partial = torch.empty((batch, hkv, splits, length * group, d), device=qkv.device, dtype=torch.float32)
    lse = torch.empty((batch, hkv, splits, length * group), device=qkv.device, dtype=torch.float32)
    result = torch.empty((tokens, hq * d), device=qkv.device, dtype=torch.bfloat16)
    rows = 16 if length > 1 else max(16, triton.next_power_of_2(length * group))
    _paged_partials[(batch, hkv * triton.cdiv(length * group, rows), splits)](
        q, data["k_cache"], data["v_cache"], data["page_table"], data["positions"],
        data["cache_seqlens"], data["k_scale"], data["v_scale"], partial, lse,
        hq, hkv, d, length, data["page_table"].shape[1], cfg["page_size"], splits,
        chunk, cfg["window"], cfg["sm_scale"] * cfg["qk_scale_factor"], rows, 128,
        num_warps=4 if rows <= 16 else 8, num_stages=3 if not cfg["window"] and batch >= 8 else 2,
    )
    _paged_reduce[(tokens, hq)](
        partial, lse, qkv, result, hq, hkv, d, length, splits, qkv.stride(0),
        triton.next_power_of_2(splits), num_warps=4 if splits <= 32 else 8,
    )
    return result


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


@triton.jit
def _router_project(X, W, L, P, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr, WN: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, S: tl.constexpr, ROUTER: tl.constexpr):
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
def _router_combine(P, Y, T: tl.constexpr, S: tl.constexpr, B: tl.constexpr):
    i = tl.program_id(0) * B + tl.arange(0, B)
    s = tl.arange(0, S)
    v = tl.load(P + s[:, None] * T + i[None, :], i[None, :] < T, 0)
    tl.store(Y + i, tl.sum(v, axis=0), i < T)

@triton.jit
def _expert_rms(X, W, Y, H: tl.constexpr, B: tl.constexpr):
    token = tl.program_id(0)
    dim = tl.arange(0, B)
    value = tl.load(X + token * H + dim, dim < H, 0).to(tl.float32)
    weight = tl.load(W + dim, dim < H, 0).to(tl.float32)
    result = value * tl.rsqrt(tl.sum(value * value, axis=0) / H + 1.0e-5) * weight
    tl.store(Y + token * H + dim, result, dim < H)


class _ExpertNorm:
    def __init__(self, weight):
        self.weight = (weight.float() + 1.0).to(torch.bfloat16)

    def __call__(self, x):
        result = torch.empty_like(x)
        _expert_rms[(x.shape[0],)](x, self.weight, result, x.shape[1],
                                  triton.next_power_of_2(x.shape[1]), num_warps=4)
        return result


def _router_logits(x, exact, state):
    tokens = x.shape[0]
    if tokens == 1:
        logits = torch.empty((1, N_EXPERTS), device=x.device, dtype=torch.float32)
        _layer_vector[(N_EXPERTS,)](exact, state["router"], logits,
                                    N_EXPERTS, HIDDEN, HIDDEN, 1, HIDDEN, num_warps=4)
        return logits
    if tokens <= ROUTER_FP32_MAX_TOKENS:
        logits = torch.empty((tokens, N_EXPERTS), device=x.device, dtype=torch.float32)
        _router_exact_vector[(N_EXPERTS, tokens)](
            exact, state["router"], logits, HIDDEN, N_EXPERTS, num_warps=4,
        )
        return logits
    if tokens <= 16:
        partial = torch.empty((16, tokens, N_EXPERTS), device=x.device, dtype=torch.float32)
        logits = torch.empty((tokens, N_EXPERTS), device=x.device, dtype=torch.float32)
        _router_project[(1, triton.cdiv(N_EXPERTS, 32), 16)](
            x, state["router_hi_t"], state["router_lo_t"], partial,
            tokens, N_EXPERTS, HIDDEN, N_EXPERTS, 16, 32, 128, 16, True,
            num_warps=4, num_stages=3,
        )
        _router_combine[(triton.cdiv(tokens * N_EXPERTS, 256),)](
            partial, logits, tokens * N_EXPERTS, 16, 256,
        )
        return logits
    parts = torch.mm(x, state["router_packed"], out_dtype=torch.float32)
    return parts[:, :N_EXPERTS] + parts[:, N_EXPERTS:]

def _prepare_decoder_fused(weights):
    state = _prepare_decoder_base(weights)
    state["pre_expert_norm"] = _ExpertNorm(weights["pre_expert_norm"])
    state["router_packed"] = torch.cat((state["router_hi_t"], state["router_lo_t"]), dim=1).contiguous()
    return state

import torch
import triton
import triton.language as tl


@triton.jit
def _dp_project(X, W, L, P, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr, WN: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, S: tl.constexpr, ROUTER: tl.constexpr):
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
def _dp_combine(P, Y, T: tl.constexpr, S: tl.constexpr, B: tl.constexpr):
    i = tl.program_id(0) * B + tl.arange(0, B)
    s = tl.arange(0, S)
    v = tl.load(P + s[:, None] * T + i[None, :], i[None, :] < T, 0)
    tl.store(Y + i, tl.sum(v, axis=0), i < T)


@triton.jit
def _dp_vector_project(X, W, Y, N: tl.constexpr, K: tl.constexpr, WK: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    n = tl.program_id(0) * BN + tl.arange(0, BN)
    k = tl.arange(0, BK)
    a = tl.load(X + k, k < K, 0).to(tl.float32)
    b = tl.load(W + n[:, None] * WK + k[None, :], (n[:, None] < N) & (k[None, :] < K), 0).to(tl.float32)
    y = tl.sum(a[None, :] * b, axis=1)
    tl.store(Y + n, y, n < N)


@triton.jit
def _dp_add_parts(P, Y, M: tl.constexpr, N: tl.constexpr, B: tl.constexpr):
    i = tl.program_id(0) * B + tl.arange(0, B)
    r = i // N
    c = i % N
    a = tl.load(P + r * (2 * N) + c, r < M, 0)
    b = tl.load(P + r * (2 * N) + c + N, r < M, 0)
    tl.store(Y + i, a + b, r < M)

from torch.utils.cpp_extension import load_inline


_DP_CUDA_SOURCE = r"""
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

def _dp__prepare_base(weights):
    state = {name: w.T.contiguous() for name, w in weights.items()}
    r = state["router"]
    state["hi"] = r.to(torch.bfloat16)
    state["lo"] = (r - state["hi"].float()).to(torch.bfloat16)
    state["original"] = weights
    state["vector_weights"] = {name: torch.nn.functional.pad(w, (0, 64)) for name, w in weights.items()}
    state["packed"] = torch.cat((state["hi"], state["lo"]), dim=1).contiguous()
    state["lt"] = load_inline(name="projection_lt_transposed", cpp_sources="void lt_matmul(torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, int64_t);", cuda_sources=_DP_CUDA_SOURCE, functions=["lt_matmul"], extra_cuda_cflags=["-O3"], extra_ldflags=["-lcublasLt"], verbose=False)
    state["workspace"] = torch.empty((32 * 1024 * 1024,), device=r.device, dtype=torch.uint8)
    return state



@triton.jit
def _dp_blocked_vector_project(X, W, Y, N: tl.constexpr, K: tl.constexpr, WK: tl.constexpr,
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


def _dp__prepare_algorithms(weights):
    import statistics
    state = _dp__prepare_base(weights)
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


def _dp__dense_base(data, state):
    x = data["x"]
    cfg = data["config"]
    m, n, k = x.shape[0], cfg["n"], x.shape[1]
    router = cfg["layer"] == "router"
    if m == 1:
        output = torch.empty((m, n), device=x.device, dtype=torch.float32 if router else torch.bfloat16)
        width, warps = state["vector_config"][cfg["layer"]]
        _dp_vector_project[(triton.cdiv(n, width),)](x, state["vector_weights"][cfg["layer"]], output, n, k, k + 64, width, triton.next_power_of_2(k), num_warps=warps)
        return output
    if not router and m <= 8 and n <= 2304 and k >= 4096:
        output = torch.empty((m, n), device=x.device, dtype=torch.bfloat16)
        partial = torch.empty((8, m, n), device=x.device, dtype=torch.float32)
        w = state[cfg["layer"]]
        _dp_project[(triton.cdiv(m, 16), triton.cdiv(n, 64), 8)](x, w, w, partial, m, n, k, w.shape[1], 16, 64, 128, 8, False, num_warps=4, num_stages=3)
        _dp_combine[(triton.cdiv(m * n, 256),)](partial, output, m * n, 8, 256)
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
        _dp_add_parts[(triton.cdiv(m * n, 256),)](parts, output, m, n, 256)
        return output
    w = state["hi"]
    lo = state["lo"]
    bm = 16 if m <= 16 else 128
    bn = 32
    splits = 16
    partial = torch.empty((splits, m, n), device=x.device, dtype=torch.float32)
    output = torch.empty((m, n), device=x.device, dtype=torch.float32 if router else torch.bfloat16)
    _dp_project[(triton.cdiv(m, bm), triton.cdiv(n, bn), splits)](x, w, lo, partial, m, n, k, w.shape[1], bm, bn, 128, splits, True, num_warps=4, num_stages=3)
    _dp_combine[(triton.cdiv(m * n, 256),)](partial, output, m * n, splits, 256)
    return output



def _dp__prepare_tuning_base(weights):
    import statistics
    state = _dp__prepare_algorithms(weights)
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
                    _dp_vector_project[(triton.cdiv(n, width),)](
                        x, state["vector_weights"][name], y, n, k, k + 64, width,
                        triton.next_power_of_2(k), num_warps=warps,
                    )
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph):
                        _dp_vector_project[(triton.cdiv(n, width),)](
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
                    _dp_vector_project[(triton.cdiv(n, width),)](
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

def _dp__run_small_tile(x, weight, tile):
    m, k = x.shape
    n = tile[-1]
    bm, bn, bk, splits, warps = tile[:-1]
    output = torch.empty((m, n), device=x.device, dtype=torch.bfloat16)
    partial = torch.empty((splits, m, n), device=x.device, dtype=torch.float32)
    _dp_project[(triton.cdiv(m, bm), triton.cdiv(n, bn), splits)](
        x, weight, weight, partial, m, n, k, weight.shape[1], bm, bn, bk, splits,
        False, num_warps=warps, num_stages=3,
    )
    _dp_combine[(triton.cdiv(m * n, 256),)](partial, output, m * n, splits, 256)
    return output


def _dp_prepare(weights):
    import statistics
    state = _dp__prepare_tuning_base(weights)
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
            reference_time = measure(lambda: _dp__dense_base(data, state))
            choices = []
            for bn, bk, splits, warps in ((64, 256, 4, 4), (128, 256, 4, 8),
                                         (64, 256, 2, 4), (128, 128, 2, 4),
                                         (32, 256, 8, 4), (64, 512, 4, 8)):
                tile = (max(16, triton.next_power_of_2(m)), bn, bk, splits, warps, n)
                try:
                    elapsed = measure(lambda: _dp__run_small_tile(x, state[name], tile))
                    choices.append((elapsed, tile))
                except RuntimeError:
                    continue
            if choices:
                elapsed, tile = min(choices)
                if elapsed < reference_time * 0.97:
                    elapsed = measure(lambda: _dp__run_small_tile(x, state[name], tile))
                    reference_time = measure(lambda: _dp__dense_base(data, state))
                    if elapsed < reference_time * 0.97:
                        state["small_tiles"][(name, m)] = tile
    return state


def _dp_kernel_fn(data, state):
    tile = state["small_tiles"].get((data["config"]["layer"], data["x"].shape[0]))
    if tile is not None and tile[-1] == data["config"]["n"]:
        return _dp__run_small_tile(data["x"], state[data["config"]["layer"]], tile)
    return _dp__dense_base(data, state)


class _DenseLayerWeight:
    def __init__(self, name, shape, state):
        self.name = name
        self.shape = shape
        self.state = state

    def __getitem__(self, index):
        start, stop, step = index.indices(self.shape[0])
        if start != 0 or step != 1:
            raise ValueError("Unsupported dense weight slice")
        return _DenseLayerWeight(self.name, (stop, self.shape[1]), self.state)


def _gemm(x, w, padded=None):
    return _dp_kernel_fn({"x": x, "config": {"layer": w.name, "n": w.shape[0]}}, w.state)


def _prepare_local_decoder(weights):
    state = _prepare_decoder_fused(weights)
    dense_state = _dp_prepare({name: weights[name] for name in ("qkv_gate", "o", "pooler", "unpooler", "router")})
    state["dense"] = {name: _DenseLayerWeight(name, weights[name].shape, dense_state)
                      for name in ("qkv_gate", "o", "pooler", "unpooler")}
    return state


@triton.jit
def _router_exact_vector(X, W, Y, K: tl.constexpr, N: tl.constexpr):
    n = tl.program_id(0)
    m = tl.program_id(1)
    k = tl.arange(0, K)
    x = tl.load(X + m * K + k).to(tl.float32)
    w = tl.load(W + n * K + k).to(tl.float32)
    tl.store(Y + m * N + n, tl.sum(x * w, axis=0))


@triton.jit
def _align_local_routes(IDs, Local, Sorted, Experts, Padded, Start, S: tl.constexpr, E: tl.constexpr,
                        BLOCK: tl.constexpr, BS: tl.constexpr,
                        MAX: tl.constexpr, BM: tl.constexpr, BE: tl.constexpr):
    slots = tl.arange(0, BS)
    experts = tl.arange(0, E)
    ids = (tl.load(IDs + slots, slots < S, -1).to(tl.int32) - tl.load(Start)) & 255
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
    state = _prepare_local_decoder(weights)
    from tvm_ffi.cpp import load_inline
    from sglang.kernels.jit.utils.compile.paths import KERNEL_PATH
    from sglang.kernels.jit.utils.arch import get_default_target_flags
    try:
        operation = load_inline(name="decoder_local_expert_launch", cuda_sources=EXPERT_LAUNCH_SOURCE,
            functions=["launch_experts"], extra_include_paths=[str(KERNEL_PATH / "include"), str(KERNEL_PATH / "csrc")],
            extra_cuda_cflags=["-std=c++20", "-O3"] + get_default_target_flags())
        device = weights["w13"].device
        sms = torch.cuda.get_device_properties(device).multi_processor_count
        hidden = weights["w2"].shape[1]
        intermediate = state["bank"]["size_n"]
        choices = {(1, TOP_K, intermediate * 2, hidden, 8): (128, 128, 256, 2),
                   (1, 1, hidden, intermediate, 8): (128, 128, 256, 1),
                   (8, TOP_K, intermediate * 2, hidden, 8): (64, 128, 128, 4),
                   (8, 1, hidden, intermediate, 8): (64, 128, 128, 4)}
        state["marlin_operation"] = _ExpertLauncher(operation, sms, choices=choices)
        print("decoder local expert launch ready")
    except Exception as error:
        print("decoder local expert launch fallback", str(error)[:240])
    return state


@triton.jit
def _map_local_routes(IDs, Start, Local, S: tl.constexpr, E: tl.constexpr, B: tl.constexpr):
    slot = tl.program_id(0) * B + tl.arange(0, B)
    expert = (tl.load(IDs + slot, slot < S, -1).to(tl.int32) - tl.load(Start)) & 255
    tl.store(Local + slot, tl.where(expert < E, expert, -1), slot < S)
