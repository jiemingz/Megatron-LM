# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Shared A/B arithmetic for 1–6 streams; S=3 entry points specialize these bodies."""

import triton
import triton.language as tl

from .helpers import _inverse_value, _native_updated


@triton.jit
def _coeff(
    W, T, A, B, C, TWO: tl.constexpr, NUM_STREAMS: tl.constexpr, HAS_RETENTION: tl.constexpr = True
):
    s = tl.arange(0, triton.next_power_of_2(NUM_STREAMS))
    w = tl.sigmoid(tl.load(W + s, s < NUM_STREAMS, 0))
    f = tl.sigmoid(-tl.load(T + s, s < NUM_STREAMS, 0)) if HAS_RETENTION else 0.0
    a = tl.sigmoid(tl.load(A + s, s < NUM_STREAMS, 0))
    b = tl.full((triton.next_power_of_2(NUM_STREAMS),), 0, tl.float32)
    if TWO:
        b = tl.sigmoid(tl.load(B + s, s < NUM_STREAMS, 0))
    tl.store(C + s, 2 * w, s < NUM_STREAMS)
    tl.store(C + 1 * NUM_STREAMS + s, tl.fma(-0.1, f, 1.0), s < NUM_STREAMS)
    tl.store(C + 2 * NUM_STREAMS + s, a, s < NUM_STREAMS)
    tl.store(C + 3 * NUM_STREAMS + s, b, s < NUM_STREAMS)
    tl.store(C + 4 * NUM_STREAMS + s, (2 * w) * (1 - w), s < NUM_STREAMS)
    tl.store(C + 5 * NUM_STREAMS + s, (0.1 * f) * (1 - f), s < NUM_STREAMS)
    tl.store(C + 6 * NUM_STREAMS + s, a * (1 - a), s < NUM_STREAMS)
    tl.store(C + 7 * NUM_STREAMS + s, b * (1 - b), s < NUM_STREAMS)


@triton.jit
def _values(
    R,
    U,
    V,
    C,
    rows,
    cols,
    M: tl.constexpr,
    H: tl.constexpr,
    TWO: tl.constexpr,
    LOW_PRECISION: tl.constexpr,
    INVERSE: tl.constexpr,
    INVERSE_MODE: tl.constexpr = "baseline",
    X_FP32: tl.constexpr = False,
    SUMMED: tl.constexpr = False,
    ROUND_SUM: tl.constexpr = False,
    NUM_STREAMS: tl.constexpr = 3,
    HAS_RETENTION: tl.constexpr = True,
):
    mask = (rows < M) & (cols < H)
    u = tl.load(U + rows * H + cols, mask, 0).to(tl.float32)
    if TWO and not SUMMED:
        u += tl.load(V + rows * H + cols, mask, 0).to(tl.float32)
    if ROUND_SUM:
        u = u.to(tl.bfloat16).to(tl.float32)
    x0 = tl.full(mask.shape, 0, tl.float32)
    x1 = tl.full(mask.shape, 0, tl.float32)
    previous, current, reads = (), (), ()
    for s in tl.static_range(NUM_STREAMS):
        r = tl.load(R + rows * (NUM_STREAMS * H) + s * H + cols, mask, 0).to(tl.float32)
        w, t = tl.load(C + s), (tl.load(C + 1 * NUM_STREAMS + s) if HAS_RETENTION else 1.0)
        if INVERSE:
            old = _inverse_value(r, u, w, t, rows, s, INVERSE_MODE, HAS_RETENTION)
            if LOW_PRECISION:
                old = old.to(R.dtype.element_ty).to(tl.float32)
            value = r
            if LOW_PRECISION and TWO:
                read = _native_updated(old, u, w, t, rows, s)
            else:
                read = r
        else:
            old = r
            value = _native_updated(old, u, w, t, rows, s)
            read = value
            if LOW_PRECISION:
                value = value.to(R.dtype.element_ty).to(tl.float32)
                if not TWO:
                    read = value
        x0 = tl.fma(tl.load(C + 2 * NUM_STREAMS + s), read, x0)
        if TWO:
            x1 = tl.fma(tl.load(C + 3 * NUM_STREAMS + s), read, x1)
        previous += (old,)
        current += (value,)
        reads += (read,)
    if not X_FP32:
        x0 = x0.to(tl.bfloat16)
        x1 = x1.to(tl.bfloat16)
    return previous, current, reads, x0, x1, u


@triton.jit
def _mix(
    R,
    U,
    V,
    C,
    OUT,
    X0,
    X1,
    M: tl.constexpr,
    H: tl.constexpr,
    TWO: tl.constexpr,
    LOW_PRECISION: tl.constexpr,
    INVERSE: tl.constexpr,
    ROWS: tl.constexpr = 8,
    COLS: tl.constexpr = 128,
    INVERSE_MODE: tl.constexpr = "baseline",
    X_FP32: tl.constexpr = False,
    SUMMED: tl.constexpr = False,
    ROUND_SUM: tl.constexpr = False,
    SUM_OUT=None,
    NUM_STREAMS: tl.constexpr = 3,
    HAS_RETENTION: tl.constexpr = True,
):
    rows = (tl.program_id(0) * ROWS + tl.arange(0, ROWS))[:, None]
    cols = (tl.program_id(1) * COLS + tl.arange(0, COLS))[None, :]
    old, current, reads, x0, x1, u = _values(
        R,
        U,
        V,
        C,
        rows,
        cols,
        M,
        H,
        TWO,
        LOW_PRECISION,
        INVERSE,
        INVERSE_MODE,
        X_FP32,
        SUMMED,
        ROUND_SUM,
        NUM_STREAMS=NUM_STREAMS,
        HAS_RETENTION=HAS_RETENTION,
    )
    mask = (rows < M) & (cols < H)
    if ROUND_SUM:
        tl.store(SUM_OUT + rows * H + cols, u, mask)
    for s in tl.static_range(NUM_STREAMS):
        value = old[s] if INVERSE else current[s]
        tl.store(OUT + rows * (NUM_STREAMS * H) + s * H + cols, value, mask)
    tl.store(X0 + rows * H + cols, x0, mask)
    if TWO:
        tl.store(X1 + rows * H + cols, x1, mask)
