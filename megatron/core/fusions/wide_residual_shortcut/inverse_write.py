# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Inverse-only bridge for ordinary streamwise writes between fused boundaries."""

import torch
import triton
import triton.language as tl


@triton.jit
def _inverse_write(
    CURRENT,
    UPDATE,
    WRITE,
    RETAIN,
    PREVIOUS,
    H: tl.constexpr,
    S: tl.constexpr,
    SIZE: tl.constexpr,
    MAX_FORGET: tl.constexpr,
    HAS_RETENTION: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < SIZE
    stream = offsets // H % S
    row = offsets // (S * H)
    column = offsets % H
    current = tl.load(CURRENT + offsets, mask, other=0).to(tl.float32)
    update = tl.load(UPDATE + row * H + column, mask, other=0).to(tl.float32)
    write = 2.0 * tl.sigmoid(tl.load(WRITE + stream).to(tl.float32))
    previous = current - write * update
    if HAS_RETENTION:
        forget = tl.sigmoid(-tl.load(RETAIN + stream).to(tl.float32))
        # The ordinary writer permits contraction of its retention coefficient.
        retention = tl.fma(-MAX_FORGET, forget, 1.0)
        previous = tl.div_rn(previous, retention)
    tl.store(PREVIOUS + offsets, previous, mask)


@torch.no_grad()
def inverse_write(current, update, write, retain, streams, max_forget):
    """Recover an ordinary writer's input without evaluating its forward again."""
    previous = torch.empty_like(current)
    _inverse_write[(triton.cdiv(current.numel(), 1024),)](
        current,
        update,
        write,
        write if retain is None else retain,
        previous,
        update.shape[-1],
        streams,
        current.numel(),
        max_forget,
        retain is not None,
        BLOCK=1024,
        enable_fp_fusion=False,
    )
    return previous
