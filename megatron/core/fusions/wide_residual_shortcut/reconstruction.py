# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Opt-in, first-order A/B residual reconstruction experiments.

No selected dispatch is changed. A SavedWrite owns updates, exact FP32
coefficients and RMS statistics, never X or an input/output residual. References
are explicitly separate objects and must not enter candidate memory captures.
Residual gradient storage is independently selected in reconstruction_backward.
The opt-in x_storage="fp32" changes forward reader rounding; it does not retain X.
"""

from dataclasses import dataclass

import torch
import triton
import triton.language as tl
from triton.experimental import gluon as gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language import BlockedLayout

from .fusion_runtime import resource
from .helpers import _inverse_value
from .helpers import _native_updated as _native_updated
from .helpers import make_quantizer
from .reconstruction_dispatch import (
    QUALIFIED_BF16_FORWARD,
    QUALIFIED_FORWARD,
    ForwardSchedule,
    qualified_device,
)
from .reconstruction_streams import _coeff as _stream_coeff
from .reconstruction_streams import _mix as _stream_mix
from .reconstruction_streams import _values as _stream_values

INVERSE_MODES = ("baseline", "div_rn", "matched", "matched_fp64", "matched_rcp")
RESIDUAL_DTYPES = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}


def validate_num_streams(num_streams):
    if type(num_streams) is not int or not 1 <= num_streams <= 6:
        raise ValueError("num_streams must be an integer from 1 through 6")
    return num_streams


def residual_dtype(storage):
    if storage not in RESIDUAL_DTYPES:
        raise ValueError("choose residual_storage fp32, bf16, or fp16")
    return RESIDUAL_DTYPES[storage]


@dataclass(frozen=True)
class SavedWrite:
    kind: str
    residual_storage: str
    updates: tuple
    coefficients: torch.Tensor  # [write, retain, read0, read1, derivatives] x num_streams
    gammas: tuple
    rstd: tuple
    controller_sizes: tuple
    x_storage: str = "bf16"
    operand_format: str = "mxfp8"
    num_streams: int = 3
    # None is a structural mode, never inferred from a learned coefficient value.
    has_retention: bool = True

    def __post_init__(self):
        validate_num_streams(self.num_streams)
        if type(self.has_retention) is not bool:
            raise ValueError("has_retention must be a static boolean")
        if not self.has_retention and (
            len(self.controller_sizes) < 2 or self.controller_sizes[1] != 0
        ):
            raise ValueError("disabled retention must have no controller storage")

    @property
    def reader_count(self):
        return 2 if self.kind == "B" else 1

    @property
    def summed_update(self):
        return self.kind == "B" and len(self.updates) == 1


@dataclass(frozen=True)
class Reference:
    previous: torch.Tensor
    current: torch.Tensor
    x: tuple


@triton.jit
def _coeff(W, T, A, B, C, TWO: tl.constexpr, HAS_RETENTION: tl.constexpr = True):
    # Keep the historical S=3 entry point; constexpr specialization inlines the
    # shared A/B body with the same tile geometry and arithmetic ordering.
    _stream_coeff(W, T, A, B, C, TWO, NUM_STREAMS=3, HAS_RETENTION=HAS_RETENTION)


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
    HAS_RETENTION: tl.constexpr = True,
):
    # Keep the historical S=3 entry point; constexpr specialization inlines the
    # shared A/B body with the same tile geometry and arithmetic ordering.
    return _stream_values(
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
        NUM_STREAMS=3,
        HAS_RETENTION=HAS_RETENTION,
    )


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
    HAS_RETENTION: tl.constexpr = True,
):
    # Keep the historical S=3 entry point; constexpr specialization inlines the
    # shared A/B body with the same tile geometry and arithmetic ordering.
    _stream_mix(
        R,
        U,
        V,
        C,
        OUT,
        X0,
        X1,
        M,
        H,
        TWO,
        LOW_PRECISION,
        INVERSE,
        ROWS,
        COLS,
        INVERSE_MODE,
        X_FP32,
        SUMMED,
        ROUND_SUM,
        SUM_OUT,
        NUM_STREAMS=3,
        HAS_RETENTION=HAS_RETENTION,
    )


@triton.jit
def _encode(y, axis: tl.constexpr):
    amax = tl.max(tl.abs(y), axis)
    exponent = (
        tl.inline_asm_elementwise(
            "cvt.rp.satfinite.ue8m0x2.f32 $0, 0.0, $1;",
            "=h,f",
            [amax * (1.0 / 448.0)],
            dtype=tl.uint16,
            is_pure=True,
            pack=1,
        ).to(tl.uint32)
        & 255
    )
    bits = tl.where(exponent == 254, 0x00400000, (254 - exponent) << 23)
    bits = tl.where(exponent == 255, 0x7FFFFFFF, bits)
    if axis == 1:
        scaled = y * bits.to(tl.float32, bitcast=True)[:, None]
    else:
        scaled = y * bits.to(tl.float32, bitcast=True)[None, :]
    q = tl.inline_asm_elementwise(
        "cvt.rn.satfinite.e4m3x2.f32 $0, 0.0, $1;",
        "=h,f",
        [scaled],
        dtype=tl.uint16,
        is_pure=True,
        pack=1,
    ).to(tl.uint8)
    return q, exponent.to(tl.uint8)


@triton.jit
def _norm_row(
    X, G, RS, Q, SCALE, M: tl.constexpr, H: tl.constexpr, BLOCK: tl.constexpr, QUANT: tl.constexpr
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    x = tl.load(X + row * H + cols, cols < H, 0).to(tl.float32)
    rs = tl.rsqrt(tl.sum(x * x, 0) / H + 1.0e-5)
    tl.store(RS + row, rs)
    if QUANT:
        g = tl.load(G + cols, cols < H, 0).to(tl.float32)
        y = (x * rs) * g
        q, exponent = _encode(tl.reshape(y, (BLOCK // 32, 32)), 1)
        tl.store(Q + row * H + cols, tl.reshape(q, (BLOCK,)), cols < H)
        group = tl.arange(0, BLOCK // 32)
        index = ((row // 128) * tl.cdiv(H, 128) + group // 4) * 512
        index += (row % 32) * 16 + (row % 128 // 32) * 4 + group % 4
        tl.store(SCALE + index, exponent, group < H // 32)


@gluon.jit
def _norm_row_bf16(X, G, RS, Y, H: gl.constexpr, BLOCK: gl.constexpr, WARPS: gl.constexpr = 4):
    # MXFP8's reshape/encoding fixes its RMS reduction to 16 values per thread.
    # Keep that FP32 summation order explicitly when removing the encoder;
    # otherwise Triton chooses a different layout for BF16 stores.
    layout: gl.constexpr = BlockedLayout([16], [32], [WARPS], [0])
    row = gl.program_id(0)
    cols = gl.arange(0, BLOCK, layout=layout)
    x = gl.load(X + row * H + cols, cols < H, 0).to(gl.float32)
    rs = gl.rsqrt(gl.sum(x * x, 0) / H + 1.0e-5)
    gl.store(RS + row, rs)
    gamma = gl.load(G + cols, cols < H, 0).to(gl.float32)
    gl.store(Y + row * H + cols, ((x * rs) * gamma).to(gl.bfloat16), cols < H)


def _tensor(t, shape, dtype, device, name):
    if (
        not isinstance(t, torch.Tensor)
        or tuple(t.shape) != tuple(shape)
        or t.dtype != dtype
        or t.device != device
        or not t.is_contiguous()
    ):
        raise ValueError(f"{name} must be contiguous {dtype} {tuple(shape)} on {device}")


@torch.no_grad()
def forward(
    residual,
    updates,
    write,
    retain,
    reads,
    gammas,
    *,
    residual_storage,
    reference=False,
    quant=True,
    x_storage="bf16",
    sum_updates=False,
    operand_format=None,
    num_streams=3,
):
    """Return (current residual, saved write, consumer operands, reference).

    Input branch/reader count denotes A or B. sum_updates rounds and saves
    B's sum in the same kernel that writes the residual. X exists only
    as forward workspace unless reference=True. Callers must choose storage.
    Operands belong to forward consumers, not the tape. operand_format="bf16"
    returns contiguous normalized BF16 tensors; omitted format preserves quant=False
    statistics-only behavior and the default rowwise MXFP8 output.

    retain=None means fixed identity retention (t=1), not zero logits (t=0.95).
    This static mode removes retention arithmetic and its parameter gradient.
    A supplied tensor always keeps the learned path, regardless of its values.
    """
    validate_num_streams(num_streams)
    if operand_format not in (None, "mxfp8", "bf16"):
        raise ValueError("operand_format must be mxfp8 or bf16")
    if operand_format is not None and not quant:
        raise ValueError("explicit operand_format requires quant=True")
    operand_format = "mxfp8" if operand_format is None else operand_format
    bf16 = operand_format == "bf16"
    if x_storage not in ("fp32", "bf16"):
        raise ValueError("choose x_storage bf16 or fp32")
    residual_dtype(residual_storage)
    n = len(updates)
    if n not in (1, 2) or len(reads) != n or len(gammas) != n:
        raise ValueError("A requires one update/read; B requires two")
    if not residual.is_cuda or updates[0].ndim != 2:
        raise ValueError("expected CUDA [M,H] updates")
    m, h = updates[0].shape
    if not m or not h or (quant and not bf16 and (m % 32 or h % 32)):
        raise ValueError("nonempty inputs; MXFP8 dimensions must be multiples of 32")
    dtype = residual_dtype(residual_storage)
    device = residual.device
    schedule = ForwardSchedule()
    if (
        num_streams == 3
        and n == 2
        and sum_updates
        and quant
        and residual_storage == "fp32"
        and x_storage == "fp32"
    ):
        if qualified_device(torch.cuda.get_device_properties(device)):
            table = QUALIFIED_BF16_FORWARD if bf16 else QUALIFIED_FORWARD
            schedule = table.get((torch.cuda.get_device_capability(device), m, h), schedule)
    if (
        num_streams != 3
        and n == 2
        and sum_updates
        and quant
        and residual_storage == "fp32"
        and x_storage == "fp32"
    ):
        from .reconstruction_dispatch import QUALIFIED_STREAM_FORWARD

        if qualified_device(torch.cuda.get_device_properties(device)):
            schedule = QUALIFIED_STREAM_FORWARD.get(
                (torch.cuda.get_device_capability(device), m, h, num_streams, operand_format),
                schedule,
            )
    _tensor(residual, (m, num_streams * h), dtype, device, "residual")
    for u in updates:
        _tensor(u, (m, h), torch.bfloat16, device, "update")
    for c in (write, *((retain,) if retain is not None else ()), *reads):
        if c.ndim != 1 or c.numel() < num_streams:
            raise ValueError(f"controllers require at least {num_streams} entries")
        _tensor(c, c.shape, torch.float32, device, "controller")
    for g in gammas:
        _tensor(g, (h,), torch.bfloat16, device, "gamma")
    c = torch.empty((8 * num_streams,), device=device, dtype=torch.float32)
    coeff_kernel, mix_kernel, stream_options = _stream_kernels(num_streams)
    k = coeff_kernel[(1,)](
        write,
        retain,
        reads[0],
        reads[-1],
        c,
        n == 2,
        **stream_options,
        HAS_RETENTION=retain is not None,
        enable_fp_fusion=False,
    )
    resource(f"reconstruction_coefficients_{n}", k)
    current = torch.empty_like(residual)
    xs = tuple(
        torch.empty_like(updates[0], dtype=torch.float32 if x_storage == "fp32" else torch.bfloat16)
        for _ in range(n)
    )
    summed = torch.empty_like(updates[0]) if sum_updates and n == 2 else None
    k = mix_kernel[(triton.cdiv(m, schedule.mix_rows), triton.cdiv(h, schedule.mix_cols))](
        residual,
        updates[0],
        updates[-1],
        c,
        current,
        xs[0],
        xs[-1],
        m,
        h,
        n == 2,
        residual_storage != "fp32",
        False,
        **stream_options,
        HAS_RETENTION=retain is not None,
        X_FP32=x_storage == "fp32",
        ROUND_SUM=summed is not None,
        SUM_OUT=summed,
        ROWS=schedule.mix_rows,
        COLS=schedule.mix_cols,
        num_warps=schedule.mix_warps,
        enable_fp_fusion=False,
    )
    resource(f"reconstruction_forward_{h}_{n}_{residual_storage}", k)
    stats = tuple(torch.empty(m, dtype=torch.float32, device=device) for _ in range(n))
    qs = []
    for x, g, rs in zip(xs, gammas, stats):
        q = (
            (
                torch.empty((m, h), dtype=torch.bfloat16, device=device)
                if bf16
                else make_quantizer().make_empty((m, h), dtype=torch.bfloat16, device=device)
            )
            if quant
            else None
        )
        if q is not None and not bf16 and (not schedule.skip_scale_reset or m % 128 or h % 128):
            q._rowwise_scale_inv.zero_()
        if bf16:
            k = _norm_row_bf16[(m,)](
                x,
                g,
                rs,
                q,
                h,
                triton.next_power_of_2(h),
                WARPS=schedule.norm_warps,
                num_warps=schedule.norm_warps,
                enable_fp_fusion=False,
            )
        else:
            k = _norm_row[(m,)](
                x,
                g,
                rs,
                None if q is None else q._rowwise_data,
                None if q is None else q._rowwise_scale_inv,
                m,
                h,
                triton.next_power_of_2(h),
                quant,
                num_warps=schedule.norm_warps,
                enable_fp_fusion=False,
            )
        resource(f"reconstruction_norm_{h}", k)
        qs.append(q)
    saved = SavedWrite(
        "A" if n == 1 else "B",
        residual_storage,
        (summed,) if summed is not None else tuple(u.detach() for u in updates),
        c,
        tuple(g.detach() for g in gammas),
        stats,
        tuple(0 if v is None else v.numel() for v in (write, retain, *reads)),
        x_storage,
        operand_format,
        num_streams,
        retain is not None,
    )
    ref = Reference(residual, current, xs) if reference else None
    return current, saved, tuple(qs), ref


@torch.no_grad()
def reconstruct(current, saved, *, inverse_mode="baseline"):
    """Tiled control: temporary X and preceding residual, without replaying GEMMs."""
    if inverse_mode not in INVERSE_MODES:
        raise ValueError("invalid inverse_mode")
    _, mix_kernel, stream_options = _stream_kernels(saved.num_streams)
    m, h = saved.updates[0].shape
    dtype = residual_dtype(saved.residual_storage)
    _tensor(current, (m, saved.num_streams * h), dtype, saved.updates[0].device, "current")
    previous = torch.empty_like(current)
    xs = tuple(
        torch.empty_like(
            saved.updates[0], dtype=torch.float32 if saved.x_storage == "fp32" else torch.bfloat16
        )
        for _ in range(saved.reader_count)
    )
    k = mix_kernel[(triton.cdiv(m, 8), triton.cdiv(h, 128))](
        current,
        saved.updates[0],
        saved.updates[-1],
        saved.coefficients,
        previous,
        xs[0],
        xs[-1],
        m,
        h,
        saved.kind == "B",
        saved.residual_storage != "fp32",
        True,
        INVERSE_MODE=inverse_mode,
        **stream_options,
        HAS_RETENTION=saved.has_retention,
        X_FP32=saved.x_storage == "fp32",
        SUMMED=saved.summed_update,
        enable_fp_fusion=False,
    )
    resource(f"reconstruction_inverse_{h}_{saved.kind}_{saved.residual_storage}", k)
    return previous, xs


@torch.no_grad()
def from_checkpoint(previous, saved):
    """Recreate one write/read from an exact segment-start checkpoint.

    Only residual arithmetic is replayed. This never replays a branch GEMM or
    recomputes statistics. Useful at checkpoint boundaries where previous is
    already available exactly, including the initial residual checkpoint.
    """
    _, mix_kernel, stream_options = _stream_kernels(saved.num_streams)
    m, h = saved.updates[0].shape
    dtype = residual_dtype(saved.residual_storage)
    _tensor(previous, (m, saved.num_streams * h), dtype, saved.updates[0].device, "checkpoint")
    current = torch.empty_like(previous)
    xs = tuple(
        torch.empty_like(
            saved.updates[0], dtype=torch.float32 if saved.x_storage == "fp32" else torch.bfloat16
        )
        for _ in range(saved.reader_count)
    )
    mix_kernel[(triton.cdiv(m, 8), triton.cdiv(h, 128))](
        previous,
        saved.updates[0],
        saved.updates[-1],
        saved.coefficients,
        current,
        xs[0],
        xs[-1],
        m,
        h,
        saved.kind == "B",
        saved.residual_storage != "fp32",
        False,
        **stream_options,
        HAS_RETENTION=saved.has_retention,
        X_FP32=saved.x_storage == "fp32",
        SUMMED=saved.summed_update,
        enable_fp_fusion=False,
    )
    return Reference(previous, current, xs)


@torch.no_grad()
def oracle(saved, previous, current, xs, dys, carry=None):
    """FP64 original-formula gradients; inverse is never differentiated.

    Returns gradients in (residual, updates..., write, retain, reads...,
    gammas...) order and absolute-contribution scales for controller checks.
    Uses saved RMS statistics so reconstruction error is isolated from a new
    forward norm calculation. References use original forward X and residuals.
    """
    m, h = saved.updates[0].shape
    n = len(xs)
    c = saved.coefficients.double().view(8, saved.num_streams)
    old = previous.double().view(m, saved.num_streams, h)
    u = sum(v.double() for v in saved.updates)
    if saved.residual_storage != "fp32" and saved.kind == "B":
        read_values = old * c[1][None, :, None] + u[:, None, :] * c[0][None, :, None]
    else:
        read_values = current.double().view_as(old)
    total = torch.zeros_like(old) if carry is None else carry.double().view_as(old)
    read_grads, gamma_grads, read_scales = [], [], []
    for i, (x, rs, g, dy) in enumerate(zip(xs, saved.rstd, saved.gammas, dys)):
        if dy is None:
            read_grads.append(None)
            read_scales.append(None)
            gamma_grads.append(None)
            continue
        z = x.double() * rs.double()[:, None]
        v = dy.double() * g.double()
        dx = (v - z * (v * z).mean(-1, keepdim=True)) * rs.double()[:, None]
        total += c[2 + i][None, :, None] * dx[:, None, :]
        terms = read_values * dx[:, None, :] * c[6 + i][None, :, None]
        read_grads.append(terms.sum((0, 2)))
        read_scales.append(terms.abs().sum((0, 2)))
        gamma_grads.append((dy.double() * z).sum(0))
    if carry is None and all(dy is None for dy in dys):
        return (None,) * (3 + 3 * n), (None,) * (3 + 3 * n)
    dr = (total * c[1][None, :, None]).view_as(previous)
    du = (total * c[0][None, :, None]).sum(1)
    wt = total * u[:, None, :] * c[4][None, :, None]
    tt = total * old * c[5][None, :, None]
    grads = (
        dr,
        *((du,) * n),
        wt.sum((0, 2)),
        tt.sum((0, 2)) if saved.has_retention else None,
        *read_grads,
        *gamma_grads,
    )
    scales = (
        None,
        *((None,) * n),
        wt.abs().sum((0, 2)),
        tt.abs().sum((0, 2)) if saved.has_retention else None,
        *read_scales,
        *((None,) * n),
    )
    return grads, scales


def _stream_kernels(num_streams):
    validate_num_streams(num_streams)
    if num_streams == 3:
        return _coeff, _mix, {}
    from .reconstruction_streams import _coeff as coeff
    from .reconstruction_streams import _mix as mix

    return coeff, mix, {"NUM_STREAMS": num_streams}
