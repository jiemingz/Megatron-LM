# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Shared arithmetic/reduction kernels and lazy optional quantizer construction."""

import triton
import triton.language as tl


@triton.jit
def _native_updated(old, update, write, retention, row, STREAM: tl.constexpr):
    product_sum = tl.inline_asm_elementwise(
        "{.reg .f32 a,b; mul.rn.f32 a,$1,$2; mul.rn.f32 b,$3,$4; add.rn.f32 $0,a,b;}",
        constraints="=f,f,f,f,f",
        args=[write, update, retention, old],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )
    contracted = tl.fma(write, update, retention * old)
    # Current native BF16 kernel: stream 0 contracts throughout; stream 1
    # contracts its first 3 per-thread row groups (rows 0..11 of each 32-row
    # tile). Stream 2 and remaining rows use separately rounded products.
    if STREAM == 0:
        return contracted
    elif STREAM == 1:
        return tl.where(row % 32 < 12, contracted, product_sum)
    else:
        return product_sum


@triton.jit
def _finish_projection(PP, PROJ, M: tl.constexpr, TILES: tl.constexpr, REDUCE: tl.constexpr):
    rows = tl.program_id(0) * 32 + tl.arange(0, 32)
    c = tl.arange(0, REDUCE)
    value = tl.load(
        PP + rows[:, None] * TILES + c[None, :], (rows[:, None] < M) & (c[None, :] < TILES), other=0
    )
    tl.store(PROJ + rows, tl.sum(value, 1), rows < M)


def make_quantizer():
    from transformer_engine.pytorch.constants import DType
    from transformer_engine.pytorch.tensor.mxfp8_tensor import MXFP8Quantizer

    q = MXFP8Quantizer(DType.kFloat8E4M3, rowwise=True, columnwise=False)
    q.optimize_for_gemm = True
    return q


def column_quantizer():
    from transformer_engine.pytorch.constants import DType
    from transformer_engine.pytorch.tensor.mxfp8_tensor import MXFP8Quantizer

    q = MXFP8Quantizer(DType.kFloat8E4M3, rowwise=False, columnwise=True)
    q.optimize_for_gemm = True
    return q


@triton.jit
def _weight_finish(
    PARTIAL, DW, H: tl.constexpr, CHUNKS: tl.constexpr, REDUCE: tl.constexpr, COLS: tl.constexpr
):
    rows = tl.arange(0, REDUCE)
    cols = tl.program_id(0) * COLS + tl.arange(0, COLS)
    values = tl.load(
        PARTIAL + rows[:, None] * H + cols[None, :],
        (rows[:, None] < CHUNKS) & (cols[None, :] < H),
        other=0,
    )
    tl.store(DW + cols, tl.sum(values, axis=0), cols < H)


@triton.jit
def _inverse_value(
    r, u, w, t, rows, STREAM: tl.constexpr, MODE: tl.constexpr, HAS_RETENTION: tl.constexpr = True
):
    """Value-only inverse; matched modes follow _native_updated contraction.

    FP64 is a diagnostic inverse, rounded back to FP32 before residual storage.
    No mode can recover the rounding error discarded by the forward write.
    Callers disable implicit FP fusion so separately rounded paths stay so.
    """
    # Fixed identity carry needs subtraction only; preserve the forward contraction.
    if not HAS_RETENTION:
        if MODE == "baseline" or MODE == "div_rn":
            old = r - w * u
        else:
            contracted = (STREAM == 0) | ((STREAM == 1) & (rows % 32 < 12))
            if MODE == "matched_fp64":
                product = tl.where(
                    contracted, w.to(tl.float64) * u.to(tl.float64), (w * u).to(tl.float64)
                )
                old = (r.to(tl.float64) - product).to(tl.float32)
            else:
                old = tl.where(contracted, tl.fma(-w, u, r), r - w * u)
    elif MODE == "baseline":
        old = (r - w * u) / t
    elif MODE == "div_rn":
        old = tl.div_rn(r - w * u, t)
    else:
        contracted = (STREAM == 0) | ((STREAM == 1) & (rows % 32 < 12))
        if MODE == "matched" or MODE == "matched_rcp":
            numerator = tl.where(contracted, tl.fma(-w, u, r), r - w * u)
            if MODE == "matched_rcp":
                old = numerator * tl.div_rn(1.0, t)
            else:
                old = tl.div_rn(numerator, t)
        else:
            # On the separate path preserve the *forward's* rounded product.
            product = tl.where(
                contracted, w.to(tl.float64) * u.to(tl.float64), (w * u).to(tl.float64)
            )
            old = ((r.to(tl.float64) - product) / t.to(tl.float64)).to(tl.float32)
    return old


@triton.jit
def _reader_gradient(
    X,
    RS,
    G,
    DY,
    P,
    off,
    rows,
    cols,
    mask,
    M: tl.constexpr,
    H: tl.constexpr,
    VALUE,
    INLINE: tl.constexpr,
):
    # A's single reader and both B readers share this derivative. Keeping each
    # invocation inline preserves load ordering and reader-local live ranges.
    x = VALUE if INLINE else tl.load(X + off, mask, 0)
    rs = tl.load(RS + rows, rows < M, 0)
    z = x.to(tl.float32) * rs
    dy = tl.load(DY + off, mask, 0).to(tl.float32)
    gamma = tl.load(G + cols, cols < H, 0).to(tl.float32)
    projection = tl.load(P + rows, rows < M, 0)
    return tl.fma(-z, projection, dy * gamma) * rs
