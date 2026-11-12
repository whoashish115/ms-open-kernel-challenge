# okc: problem=muse_attention_decode
"""Starter: the attention step in plain PyTorch, correct but slow. Each request attends over
every position its page table can hold in the paged FP8 cache, with positions past its
context or outside the window masked out.

Implement kernel_fn."""

from typing import TypedDict

import torch


class Config(TypedDict):
    batch: int            # B
    q_len: int            # L: 1 for decode, 8 for a speculative verify block (causal inside it)
    num_q_heads: int      # Hq: 16 per GPU under TP8
    num_kv_heads: int     # Hkv: 2; each serves Hq / Hkv query heads
    head_dim: int         # 64
    page_size: int        # 1 in the scored cases (as SGLang serves Muse on H200), 64 in some tests
    window: int           # 2048 on sliding-window layers (a token and the 2,047 before it); 0 = all
    rope_theta: float     # 500000 on sliding-window layers; 0 = no RoPE
    qk_scale_factor: float
    sm_scale: float       # 1/8
    kv_scale: float       # the cache scales' value: 1 in the scored cases (SGLang leaves them unset)


class Inputs(TypedDict):
    qkv: torch.Tensor          # bfloat16 [B, L, (2*Hq + 2*Hkv)*64], 2304 under tp8: per token q (Hq*64),
                               # then the block's keys before norm and RoPE (Hkv*64), values
                               # (Hkv*64) and the output gate's logits (Hq*64)
    positions: torch.Tensor    # int64 [B*L], each new token's position (seq_lens[b] + i)
    cache_loc: torch.Tensor    # int64 [B*L], each new token's flat cache row (page * page_size + offset)
    cache_seqlens: torch.Tensor  # int32 [B], seq_lens + L: what each request attends over
    cu_seqlens_q: torch.Tensor   # int32 [B+1], 0, L, 2L, ...: the block's query offsets
    scheduler_metadata: torch.Tensor | None  # int32, FA3's tile schedule (get_scheduler_metadata) on
                                             # full-attention decode steps, else None
    k_cache: torch.Tensor     # float8_e4m3fn [pages, page_size, Hkv, 64], written in place
    v_cache: torch.Tensor     # float8_e4m3fn [pages, page_size, Hkv, 64], written in place
    k_scale: torch.Tensor     # float32 [], key = e4m3 * k_scale
    v_scale: torch.Tensor     # float32 [], value = e4m3 * v_scale
    page_table: torch.Tensor  # int32 [B, max_pages], physical page of each logical page (-1 unused)
    seq_lens: torch.Tensor    # int32 [B], cached tokens per request; the block starts at this position
    config: Config


def rms(x, eps=1e-5):
    x = x.float()
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)


def rope(x, pos, theta):
    """Interleaved (GPT-J) RoPE: pairs (2i, 2i+1) rotate by pos * theta^(-2i/d)."""
    d = x.shape[-1]
    inv_freq = 1.0 / (theta ** (torch.arange(0, d, 2, device=x.device).float() / d))
    ang = pos.float()[..., None] * inv_freq
    cos, sin = ang.cos().unsqueeze(-2), ang.sin().unsqueeze(-2)
    even, odd = x[..., 0::2], x[..., 1::2]
    return torch.stack([even * cos - odd * sin, even * sin + odd * cos], -1).flatten(-2)


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


def kernel_fn(data):
    cfg = data["config"]
    return _attention_step(data, data["qkv"]).view(cfg["batch"], cfg["q_len"], -1)
