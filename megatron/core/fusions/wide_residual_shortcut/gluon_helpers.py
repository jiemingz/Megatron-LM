# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Inlined Gluon arithmetic shared by standard A/B and cooperative B.

These helpers do not launch kernels or change layouts, reductions, residency or
shared-memory lifetimes. Keep load/synchronization policy in the calling kernel.
"""

from triton.experimental import gluon as g
from triton.experimental.gluon import language as gl


@g.jit
def _inverse_value(
    r, u, w, t, rows, STREAM: gl.constexpr, MODE: gl.constexpr, HAS_RETENTION: gl.constexpr = True
):
    """Value-only inverse; matched modes follow _native_updated contraction.

    FP64 is a diagnostic inverse, rounded back to FP32 before residual storage.
    No mode can recover the rounding error discarded by the forward write.
    Callers disable implicit FP fusion so separately rounded paths stay so.
    """
    if not HAS_RETENTION:
        contracted = (STREAM == 0) | ((STREAM == 1) & (rows % 32 < 12))
        if MODE == "matched_fp64":
            product = gl.where(
                contracted, w.to(gl.float64) * u.to(gl.float64), (w * u).to(gl.float64)
            )
            old = (r.to(gl.float64) - product).to(gl.float32)
        else:
            old = (
                gl.where(contracted, gl.fma(-w, u, r), r - w * u)
                if MODE != "baseline" and MODE != "div_rn"
                else r - w * u
            )
    elif MODE == "baseline":
        old = (r - w * u) / t
    elif MODE == "div_rn":
        old = gl.div_rn(r - w * u, t)
    else:
        contracted = (STREAM == 0) | ((STREAM == 1) & (rows % 32 < 12))
        if MODE == "matched" or MODE == "matched_rcp":
            numerator = gl.where(contracted, gl.fma(-w, u, r), r - w * u)
            if MODE == "matched_rcp":
                old = numerator * gl.div_rn(1.0, t)
            else:
                old = gl.div_rn(numerator, t)
        else:
            # On the separate path preserve the *forward's* rounded product.
            product = gl.where(
                contracted, w.to(gl.float64) * u.to(gl.float64), (w * u).to(gl.float64)
            )
            old = ((r.to(gl.float64) - product) / t.to(gl.float64)).to(gl.float32)
    return old


@g.jit
def _rmsnorm_dx(z, projection, weighted_dy, rstd):
    # One explicit FMA preserves the accepted rounding and avoids a temporary
    # multiply/subtract pair in both A's joint stage and B's cooperative K3.
    return gl.fma(-z, projection, weighted_dy) * rstd


@g.jit
def _reader_gradient(
    X,
    CX,
    WG,
    RS,
    G,
    DY,
    P,
    off,
    rows,
    cols,
    mask,
    M: gl.constexpr,
    H: gl.constexpr,
    VALUE,
    INLINE: gl.constexpr,
    CACHE: gl.constexpr,
    CACHE_MODIFIER: gl.constexpr,
):
    # Reader 0 and reader 1 use identical RMSNorm derivative arithmetic. Static
    # cache/inline flags remove unused loads without merging their live ranges.
    if CACHE != "none":
        x = gl.load(CX + off, mask, 0)
    elif not INLINE:
        x = gl.load(X + off, mask, 0)
    else:
        x = VALUE
    rs = gl.load(RS + rows, rows < M, 0)
    z = x.to(gl.float32) * rs
    if CACHE == "x_dy":
        weighted = gl.load(WG + off, mask, 0)
    else:
        dy = gl.load(DY + off, mask, 0, cache_modifier=CACHE_MODIFIER).to(gl.float32)
        gamma = gl.load(G + cols, cols < H, 0).to(gl.float32)
        weighted = dy * gamma
    projection = gl.load(P + rows, rows < M, 0)
    return _rmsnorm_dx(z, projection, weighted, rs)
