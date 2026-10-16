# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Explicit-layout preparation for FP32 residual/readers; no saved activations."""

from triton.experimental import gluon as g
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language import BlockedLayout, SliceLayout


@g.jit
def _reader(
    R,
    C,
    X,
    RS,
    G,
    DY,
    Q,
    SCALE,
    PP,
    GP,
    M: gl.constexpr,
    H: gl.constexpr,
    INLINE: gl.constexpr,
    READER: gl.constexpr,
    HAS_DY: gl.constexpr,
    QUANT: gl.constexpr,
    COLS: gl.constexpr,
    LAYOUT: gl.constexpr,
    ROW_MAJOR: gl.constexpr,
    PACKED: gl.constexpr,
    CX,
    WG,
    CACHE: gl.constexpr,
    SUPPLIED: gl.constexpr = False,
    VALUE=None,
    PROJECT: gl.constexpr = True,
    BF16: gl.constexpr = False,
):
    rp = gl.program_id(1) if ROW_MAJOR else gl.program_id(0)
    cp = gl.program_id(0) if ROW_MAJOR else gl.program_id(1)
    rr = rp * 32 + gl.arange(0, 32, layout=gl.constexpr(SliceLayout(1, LAYOUT)))
    cc = cp * COLS + gl.arange(0, COLS, layout=gl.constexpr(SliceLayout(0, LAYOUT)))
    rows, cols = rr[:, None], cc[None, :]
    mask = (rows < M) & (cols < H)
    if SUPPLIED:
        x = VALUE
    elif INLINE:
        x = gl.full((32, COLS), 0, gl.float32, LAYOUT)
        for s in gl.static_range(3):
            r = gl.load(R + rows * (3 * H) + s * H + cols, mask, 0)
            x = gl.fma(gl.load(C + 6 + READER * 3 + s), r, x)
    else:
        x = gl.load(X + rows * H + cols, mask, 0).to(gl.float32)
    if CACHE != "none":
        gl.store(CX + rows * H + cols, x, mask)
    rs = gl.load(RS + rows, rows < M, 0)
    gamma = gl.load(G + cols, cols < H, 0).to(gl.float32)
    z = x * rs
    if HAS_DY and PROJECT:
        dy = gl.load(DY + rows * H + cols, mask, 0).to(gl.float32)
        if CACHE == "x_dy":
            gl.store(WG + rows * H + cols, dy * gamma, mask)
        projection = gl.sum((dy * gamma) * z, 1) / H
        # Quantization scales pad H to 128 columns. A 64-column CTA may cover
        # only scale padding; it must not write beyond the unpadded PP width.
        projection_mask = rr < M
        if gl.cdiv(gl.cdiv(H, 128) * 128, COLS) > gl.cdiv(H, COLS):
            projection_mask = projection_mask & (cp < gl.cdiv(H, COLS))
        gl.store(PP + rr * gl.cdiv(H, COLS) + cp, projection, projection_mask)
        gl.store(GP + rp * H + cc, gl.sum(dy * z, 0), (cc < H) & (rp * 32 < M))
    if QUANT and BF16:
        gl.store(Q + rows * H + cols, (z * gamma).to(gl.bfloat16), mask)
    elif QUANT:
        y = z * gamma
        amax = gl.max(gl.abs(y), 0)
        exponent = (
            gl.inline_asm_elementwise(
                "cvt.rp.satfinite.ue8m0x2.f32 $0, 0.0, $1;",
                "=h,f",
                [amax * (1.0 / 448.0)],
                dtype=gl.uint16,
                is_pure=True,
                pack=1,
            ).to(gl.uint32)
            & 255
        )
        bits = gl.where(exponent == 254, 0x00400000, (254 - exponent) << 23)
        bits = gl.where(exponent == 255, 0x7FFFFFFF, bits)
        scaled = y * bits.to(gl.float32, bitcast=True)[None, :]
        if PACKED:
            q = scaled.to(gl.float8e4nv).to(gl.uint8, bitcast=True)
        else:
            q = gl.inline_asm_elementwise(
                "cvt.rn.satfinite.e4m3x2.f32 $0, 0.0, $1;",
                "=h,f",
                [scaled],
                dtype=gl.uint16,
                is_pure=True,
                pack=1,
            ).to(gl.uint8)
        gl.store(Q + rows * H + cols, q, mask)
        index = ((cc // 128) * gl.cdiv(M, 128) + rp // 4) * 512
        index += (cc % 32) * 16 + (cc % 128 // 32) * 4 + rp % 4
        gl.store(SCALE + index, exponent.to(gl.uint8), cc < gl.cdiv(H, 128) * 128)


@g.jit
def _prepare_gluon(
    R,
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
    Q0,
    Q1,
    S0,
    S1,
    PP0,
    PP1,
    GP0,
    GP1,
    M: gl.constexpr,
    H: gl.constexpr,
    TWO: gl.constexpr,
    LOW_PRECISION: gl.constexpr,
    INLINE: gl.constexpr,
    D0: gl.constexpr,
    D1: gl.constexpr,
    Q_0: gl.constexpr,
    Q_1: gl.constexpr,
    COLS: gl.constexpr,
    INVERSE_MODE: gl.constexpr,
    X_FP32: gl.constexpr,
    SUMMED: gl.constexpr,
    SHARED_READERS: gl.constexpr,
    CONFIG: gl.constexpr,
    WARPS: gl.constexpr,
    CX0=None,
    CX1=None,
    WG0=None,
    WG1=None,
    CACHE: gl.constexpr = "none",
    PROJECT: gl.constexpr = True,
    BF16: gl.constexpr = False,
):
    gl.static_assert(not LOW_PRECISION and X_FP32)
    layout: gl.constexpr = BlockedLayout(
        [1, CONFIG[0]], [32 // CONFIG[1], CONFIG[1]], [WARPS // CONFIG[2], CONFIG[2]], [1, 0]
    )
    if SHARED_READERS:
        rp = gl.program_id(1) if CONFIG[3] else gl.program_id(0)
        cp = gl.program_id(0) if CONFIG[3] else gl.program_id(1)
        rows = (rp * 32 + gl.arange(0, 32, layout=gl.constexpr(SliceLayout(1, layout))))[:, None]
        cols = (cp * COLS + gl.arange(0, COLS, layout=gl.constexpr(SliceLayout(0, layout))))[
            None, :
        ]
        mask = (rows < M) & (cols < H)
        x0 = gl.full((32, COLS), 0, gl.float32, layout)
        x1 = gl.full((32, COLS), 0, gl.float32, layout)
        for s in gl.static_range(3):
            r = gl.load(R + rows * 3 * H + s * H + cols, mask, 0)
            x0 = gl.fma(gl.load(C + 6 + s), r, x0)
            x1 = gl.fma(gl.load(C + 9 + s), r, x1)
        _reader(
            R,
            C,
            X0,
            RS0,
            G0,
            DY0,
            Q0,
            S0,
            PP0,
            GP0,
            M,
            H,
            INLINE,
            0,
            D0,
            Q_0,
            COLS,
            layout,
            CONFIG[3],
            CONFIG[4],
            CX0,
            WG0,
            CACHE,
            True,
            x0,
            PROJECT,
            BF16=BF16,
        )
        _reader(
            R,
            C,
            X1,
            RS1,
            G1,
            DY1,
            Q1,
            S1,
            PP1,
            GP1,
            M,
            H,
            INLINE,
            1,
            D1,
            Q_1,
            COLS,
            layout,
            CONFIG[3],
            CONFIG[4],
            CX1,
            WG1,
            CACHE,
            True,
            x1,
            PROJECT,
            BF16=BF16,
        )
    elif gl.program_id(2) == 0:
        _reader(
            R,
            C,
            X0,
            RS0,
            G0,
            DY0,
            Q0,
            S0,
            PP0,
            GP0,
            M,
            H,
            INLINE,
            0,
            D0,
            Q_0,
            COLS,
            layout,
            CONFIG[3],
            CONFIG[4],
            CX0,
            WG0,
            CACHE,
            PROJECT=PROJECT,
            BF16=BF16,
        )
    else:
        _reader(
            R,
            C,
            X1,
            RS1,
            G1,
            DY1,
            Q1,
            S1,
            PP1,
            GP1,
            M,
            H,
            INLINE,
            1,
            D1,
            Q_1,
            COLS,
            layout,
            CONFIG[3],
            CONFIG[4],
            CX1,
            WG1,
            CACHE,
            PROJECT=PROJECT,
            BF16=BF16,
        )
