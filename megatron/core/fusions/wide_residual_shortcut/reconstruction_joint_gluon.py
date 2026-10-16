# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Explicit-layout FP32 joint backward, retaining matched inverse arithmetic."""

from triton.experimental import gluon as g
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language import BlockedLayout, SliceLayout

from .gluon_helpers import _inverse_value, _reader_gradient


@g.jit
def _joint_gluon(
    R,
    PREV,
    U,
    V,
    C,
    X0,
    X1,
    RS0,
    RS1,
    G0,
    G1,
    DY0,
    DY1,
    CARRY,
    P0,
    P1,
    OUT,
    DR,
    DU,
    DV,
    CP,
    M: gl.constexpr,
    H: gl.constexpr,
    TWO: gl.constexpr,
    LOW_PRECISION: gl.constexpr,
    INLINE: gl.constexpr,
    D0: gl.constexpr,
    D1: gl.constexpr,
    HAS_CARRY: gl.constexpr,
    ROWS: gl.constexpr,
    COLS: gl.constexpr,
    INVERSE_MODE: gl.constexpr,
    X_FP32: gl.constexpr,
    SUMMED: gl.constexpr,
    ROW_MAJOR: gl.constexpr,
    CONFIG: gl.constexpr,
    WARPS: gl.constexpr,
    RETAIN: gl.constexpr = False,
    CACHE: gl.constexpr = "none",
    CX0=None,
    CX1=None,
    WG0=None,
    WG1=None,
    CONTROLLERS: gl.constexpr = True,
    HAS_RETENTION: gl.constexpr = True,
):
    gl.static_assert(not LOW_PRECISION and X_FP32)
    cache: gl.constexpr = CONFIG[4] if len(CONFIG) > 4 else ""
    layout: gl.constexpr = BlockedLayout(
        [1, CONFIG[0]], [32 // CONFIG[1], CONFIG[1]], [WARPS // CONFIG[2], CONFIG[2]], [1, 0]
    )
    row_pid = gl.program_id(1) if ROW_MAJOR else gl.program_id(0)
    col_pid = gl.program_id(0) if ROW_MAJOR else gl.program_id(1)
    rows = (row_pid * ROWS + gl.arange(0, ROWS, layout=gl.constexpr(SliceLayout(1, layout))))[
        :, None
    ]
    cols = (col_pid * COLS + gl.arange(0, COLS, layout=gl.constexpr(SliceLayout(0, layout))))[
        None, :
    ]
    off = rows * H + cols
    mask = (rows < M) & (cols < H)
    retained = ()
    if INLINE and CACHE == "none" and (D0 or D1):
        x0 = gl.full((ROWS, COLS), 0, gl.float32, layout)
        x1 = gl.full((ROWS, COLS), 0, gl.float32, layout)
        for s in gl.static_range(3):
            r = gl.load(R + rows * (3 * H) + s * H + cols, mask, 0, cache_modifier=cache)
            if RETAIN:
                retained += (r,)
            x0 = gl.fma(gl.load(C + 6 + s), r, x0)
            if TWO:
                x1 = gl.fma(gl.load(C + 9 + s), r, x1)
    dx0 = gl.full((ROWS, COLS), 0, gl.float32, layout)
    dx1 = gl.full((ROWS, COLS), 0, gl.float32, layout)
    if D0:
        dx0 = _reader_gradient(
            X0,
            CX0,
            WG0,
            RS0,
            G0,
            DY0,
            P0,
            off,
            rows,
            cols,
            mask,
            M,
            H,
            x0 if INLINE and CACHE == "none" else 0.0,
            INLINE,
            CACHE,
            cache,
        )
    if D1:
        dx1 = _reader_gradient(
            X1,
            CX1,
            WG1,
            RS1,
            G1,
            DY1,
            P1,
            off,
            rows,
            cols,
            mask,
            M,
            H,
            x1 if INLINE and CACHE == "none" else 0.0,
            INLINE,
            CACHE,
            cache,
        )
    u = gl.load(U + off, mask, 0, cache_modifier=cache).to(gl.float32)
    if TWO and not SUMMED:
        u += gl.load(V + off, mask, 0, cache_modifier=cache).to(gl.float32)
    du = gl.full((ROWS, COLS), 0, gl.float32, layout)
    count: gl.constexpr = gl.cdiv(M, ROWS) * gl.cdiv(H, COLS)
    pid = row_pid * gl.cdiv(H, COLS) + col_pid
    # Expose independent gradient loads before the inverse and reductions.
    # The selected register cap trades a small on-chip spill footprint for
    # more resident warps; physical DRAM accounting includes any spill misses.
    prefetched = ()
    if len(CONFIG) > 3 and CONFIG[3] and HAS_CARRY:
        for s in gl.static_range(3):
            prefetched += (
                gl.load(CARRY + rows * 3 * H + s * H + cols, mask, 0, cache_modifier=cache).to(
                    gl.float32
                ),
            )
    # Reload one residual stream at a time; recovered values own separate output.
    for s in gl.static_range(3):
        wide = rows * 3 * H + s * H + cols
        if RETAIN and INLINE and CACHE == "none" and (D0 or D1):
            r = retained[s]
        else:
            r = gl.load(R + wide, mask, 0, cache_modifier=cache).to(gl.float32)
        w, t = gl.load(C + s), (gl.load(C + 3 + s) if HAS_RETENTION else 1.0)
        if INLINE:
            old = _inverse_value(r, u, w, t, rows, s, INVERSE_MODE, HAS_RETENTION)
        else:
            old = gl.load(PREV + wide, mask, 0).to(gl.float32)
        gl.store(OUT + wide, old, mask)
        value = r
        d = gl.full((ROWS, COLS), 0, gl.float32, layout)
        if D0:
            d = gl.load(C + 6 + s) * dx0
        if D1:
            d = d + gl.load(C + 9 + s) * dx1
        if HAS_CARRY:
            if len(CONFIG) > 3 and CONFIG[3]:
                d += prefetched[s]
            else:
                d += gl.load(CARRY + wide, mask, 0, cache_modifier=cache).to(gl.float32)
        gl.store(DR + wide, t * d, mask)
        du = gl.fma(w, d, du)
        if CONTROLLERS:
            gl.store(CP + s * count + pid, gl.sum(gl.sum(d * u, 1), 0))
            if HAS_RETENTION:
                gl.store(CP + (3 + s) * count + pid, gl.sum(gl.sum(d * old, 1), 0))
            if D0:
                gl.store(
                    CP + ((1 + HAS_RETENTION) * 3 + s) * count + pid,
                    gl.sum(gl.sum(value * dx0, 1), 0),
                )
            if D1:
                gl.store(
                    CP + ((2 + HAS_RETENTION) * 3 + s) * count + pid,
                    gl.sum(gl.sum(value * dx1, 1), 0),
                )
    gl.store(DU + off, du, mask)
    if TWO:
        gl.store(DV + off, du, mask)
