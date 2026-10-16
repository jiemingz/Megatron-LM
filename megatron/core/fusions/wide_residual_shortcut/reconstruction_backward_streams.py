# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Independent standard backward for generalized stream counts.

All modes use the same gradient formula. ``saved`` supplies original forward
values, ``control`` materializes reconstructed X, and ``inline`` reconstructs
tile-local X in preparation and joint backward. No GEMM is invoked here.
"""

from dataclasses import dataclass

import torch
import triton
import triton.language as tl

from .fusion_runtime import resource
from .helpers import (
    _finish_projection,
    _native_updated,
    _reader_gradient,
    _weight_finish,
    column_quantizer,
)
from .reconstruction import (
    INVERSE_MODES,
    _encode,
    _inverse_value,
    _tensor,
    reconstruct,
    residual_dtype,
)
from .reconstruction_stream_reductions import _controller_chunks, _finish_controllers
from .reconstruction_streams import _values

GRADIENT_STORAGE = ("bf16", "fp32_residual", "fp32")


def gradient_dtypes(storage):
    """Residual and producer/dY dtypes; controller and gamma gradients are FP32."""
    if storage not in GRADIENT_STORAGE:
        raise ValueError("choose gradient_storage bf16, fp32_residual, or fp32")
    return (
        torch.bfloat16 if storage == "bf16" else torch.float32,
        torch.float32 if storage == "fp32" else torch.bfloat16,
    )


@triton.jit
def _prepare_one(
    R,
    U,
    V,
    C,
    X,
    RS,
    G,
    DY,
    Q,
    SCALE,
    PP,
    GP,
    M: tl.constexpr,
    H: tl.constexpr,
    TWO: tl.constexpr,
    LOW_PRECISION: tl.constexpr,
    INLINE: tl.constexpr,
    READER: tl.constexpr,
    HAS_DY: tl.constexpr,
    QUANT: tl.constexpr,
    COLS: tl.constexpr,
    INVERSE_MODE: tl.constexpr,
    X_FP32: tl.constexpr,
    SUMMED: tl.constexpr = False,
    TILE_X: tl.constexpr = False,
    BF16: tl.constexpr = False,
    NUM_STREAMS: tl.constexpr = 3,
    HAS_RETENTION: tl.constexpr = True,
):
    rows = (tl.program_id(0) * 32 + tl.arange(0, 32))[:, None]
    cols = (tl.program_id(1) * COLS + tl.arange(0, COLS))[None, :]
    mask = (rows < M) & (cols < H)
    if TILE_X:
        x = X.to(tl.float32)
    elif INLINE:
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
            True,
            INVERSE_MODE,
            X_FP32,
            SUMMED,
            NUM_STREAMS=NUM_STREAMS,
            HAS_RETENTION=HAS_RETENTION,
        )
        x = x0 if READER == 0 else x1
        x = x.to(tl.float32)
    else:
        x = tl.load(X + rows * H + cols, mask, 0).to(tl.float32)
    rs = tl.load(RS + rows, rows < M, 0)
    gamma = tl.load(G + cols, cols < H, 0).to(tl.float32)
    z = x * rs
    if HAS_DY:
        dy = tl.load(DY + rows * H + cols, mask, 0).to(tl.float32)
        projection = tl.sum((dy * gamma) * z, 1) / H
        tl.store(
            PP + tl.reshape(rows, (32,)) * tl.cdiv(H, COLS) + tl.program_id(1),
            projection,
            tl.reshape(rows, (32,)) < M,
        )
        tl.store(
            GP + tl.program_id(0) * H + tl.reshape(cols, (COLS,)),
            tl.sum(dy * z, 0),
            (tl.reshape(cols, (COLS,)) < H) & (tl.program_id(0) * 32 < M),
        )
    if QUANT and BF16:
        tl.store(Q + rows * H + cols, (z * gamma).to(tl.bfloat16), mask)
    elif QUANT:
        q, exponent = _encode(z * gamma, 0)
        tl.store(Q + rows * H + cols, q, mask)
        cc = tl.reshape(cols, (COLS,))
        group = tl.program_id(0)
        index = ((cc // 128) * tl.cdiv(M, 128) + group // 4) * 512
        index += (cc % 32) * 16 + (cc % 128 // 32) * 4 + group % 4
        # The launch covers padded column groups too. Their masked values are
        # zero, so this initializes every scale byte without extra memset launches.
        tl.store(SCALE + index, exponent, cc < tl.cdiv(H, 128) * 128)


@triton.jit
def _prepare(
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
    M: tl.constexpr,
    H: tl.constexpr,
    TWO: tl.constexpr,
    LOW_PRECISION: tl.constexpr,
    INLINE: tl.constexpr,
    D0: tl.constexpr,
    D1: tl.constexpr,
    Q_0: tl.constexpr,
    Q_1: tl.constexpr,
    COLS: tl.constexpr,
    INVERSE_MODE: tl.constexpr,
    X_FP32: tl.constexpr,
    SUMMED: tl.constexpr = False,
    SHARED_READERS: tl.constexpr = False,
    BF16: tl.constexpr = False,
    NUM_STREAMS: tl.constexpr = 3,
    HAS_RETENTION: tl.constexpr = True,
):
    if SHARED_READERS:
        rows = (tl.program_id(0) * 32 + tl.arange(0, 32))[:, None]
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
            True,
            INVERSE_MODE,
            X_FP32,
            SUMMED,
            NUM_STREAMS=NUM_STREAMS,
            HAS_RETENTION=HAS_RETENTION,
        )
        _prepare_one(
            R,
            U,
            V,
            C,
            x0,
            RS0,
            G0,
            DY0,
            Q0,
            S0,
            PP0,
            GP0,
            M,
            H,
            TWO,
            LOW_PRECISION,
            False,
            0,
            D0,
            Q_0,
            COLS,
            INVERSE_MODE,
            X_FP32,
            SUMMED,
            True,
            BF16=BF16,
            NUM_STREAMS=NUM_STREAMS,
            HAS_RETENTION=HAS_RETENTION,
        )
        _prepare_one(
            R,
            U,
            V,
            C,
            x1,
            RS1,
            G1,
            DY1,
            Q1,
            S1,
            PP1,
            GP1,
            M,
            H,
            TWO,
            LOW_PRECISION,
            False,
            1,
            D1,
            Q_1,
            COLS,
            INVERSE_MODE,
            X_FP32,
            SUMMED,
            True,
            BF16=BF16,
            NUM_STREAMS=NUM_STREAMS,
            HAS_RETENTION=HAS_RETENTION,
        )
    elif tl.program_id(2) == 0:
        _prepare_one(
            R,
            U,
            V,
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
            TWO,
            LOW_PRECISION,
            INLINE,
            0,
            D0,
            Q_0,
            COLS,
            INVERSE_MODE,
            X_FP32,
            SUMMED,
            BF16=BF16,
            NUM_STREAMS=NUM_STREAMS,
            HAS_RETENTION=HAS_RETENTION,
        )
    else:
        _prepare_one(
            R,
            U,
            V,
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
            TWO,
            LOW_PRECISION,
            INLINE,
            1,
            D1,
            Q_1,
            COLS,
            INVERSE_MODE,
            X_FP32,
            SUMMED,
            BF16=BF16,
            NUM_STREAMS=NUM_STREAMS,
            HAS_RETENTION=HAS_RETENTION,
        )


@triton.jit
def _finish_reads(
    PP0,
    PP1,
    GP0,
    GP1,
    P0,
    P1,
    DG0,
    DG1,
    M: tl.constexpr,
    H: tl.constexpr,
    COLS: tl.constexpr,
    D0: tl.constexpr,
    D1: tl.constexpr,
    PR: tl.constexpr,
    GR: tl.constexpr,
):
    if D0:
        if tl.program_id(1) == 0 and tl.program_id(0) < tl.cdiv(M, 32):
            _finish_projection(PP0, P0, M, tl.cdiv(H, COLS), PR)
        if tl.program_id(1) == 1 and tl.program_id(0) < tl.cdiv(H, 32):
            _weight_finish(GP0, DG0, H, tl.cdiv(M, 32), GR, 32)
    if D1:
        if tl.program_id(1) == 2 and tl.program_id(0) < tl.cdiv(M, 32):
            _finish_projection(PP1, P1, M, tl.cdiv(H, COLS), PR)
        if tl.program_id(1) == 3 and tl.program_id(0) < tl.cdiv(H, 32):
            _weight_finish(GP1, DG1, H, tl.cdiv(M, 32), GR, 32)


@triton.jit
def _joint(
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
    M: tl.constexpr,
    H: tl.constexpr,
    TWO: tl.constexpr,
    LOW_PRECISION: tl.constexpr,
    INLINE: tl.constexpr,
    D0: tl.constexpr,
    D1: tl.constexpr,
    HAS_CARRY: tl.constexpr,
    ROWS: tl.constexpr,
    COLS: tl.constexpr,
    INVERSE_MODE: tl.constexpr,
    X_FP32: tl.constexpr,
    SUMMED: tl.constexpr = False,
    ROW_MAJOR: tl.constexpr = False,
    NUM_STREAMS: tl.constexpr = 3,
    HAS_RETENTION: tl.constexpr = True,
):
    row_pid = tl.program_id(1) if ROW_MAJOR else tl.program_id(0)
    col_pid = tl.program_id(0) if ROW_MAJOR else tl.program_id(1)
    rows = (row_pid * ROWS + tl.arange(0, ROWS))[:, None]
    cols = (col_pid * COLS + tl.arange(0, COLS))[None, :]
    off = rows * H + cols
    mask = (rows < M) & (cols < H)
    if INLINE and (D0 or D1):
        olds, currents, reads, x0, x1, unused = _values(
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
            True,
            INVERSE_MODE,
            X_FP32,
            SUMMED,
            NUM_STREAMS=NUM_STREAMS,
            HAS_RETENTION=HAS_RETENTION,
        )
    dx0 = tl.full((ROWS, COLS), 0, tl.float32)
    dx1 = tl.full((ROWS, COLS), 0, tl.float32)
    if D0:
        dx0 = _reader_gradient(
            X0, RS0, G0, DY0, P0, off, rows, cols, mask, M, H, x0 if INLINE else 0.0, INLINE
        )
    if D1:
        dx1 = _reader_gradient(
            X1, RS1, G1, DY1, P1, off, rows, cols, mask, M, H, x1 if INLINE else 0.0, INLINE
        )
    u = tl.load(U + off, mask, 0).to(tl.float32)
    if TWO and not SUMMED:
        u += tl.load(V + off, mask, 0).to(tl.float32)
    du = tl.full((ROWS, COLS), 0, tl.float32)
    count: tl.constexpr = tl.cdiv(M, ROWS) * tl.cdiv(H, COLS)
    pid = row_pid * tl.cdiv(H, COLS) + col_pid
    # Reload one residual stream at a time; recovered values own separate output.
    for s in tl.static_range(NUM_STREAMS):
        wide = rows * NUM_STREAMS * H + s * H + cols
        r = tl.load(R + wide, mask, 0).to(tl.float32)
        w, t = tl.load(C + s), (tl.load(C + 1 * NUM_STREAMS + s) if HAS_RETENTION else 1.0)
        if INLINE:
            old = _inverse_value(r, u, w, t, rows, s, INVERSE_MODE, HAS_RETENTION)
            if LOW_PRECISION:
                old = old.to(R.dtype.element_ty).to(tl.float32)
        else:
            old = tl.load(PREV + wide, mask, 0).to(tl.float32)
        tl.store(OUT + wide, old, mask)
        value = r
        if LOW_PRECISION and TWO:
            value = _native_updated(old, u, w, t, rows, s)
        d = tl.full((ROWS, COLS), 0, tl.float32)
        if D0:
            d = tl.load(C + 2 * NUM_STREAMS + s) * dx0
        if D1:
            d = d + tl.load(C + 3 * NUM_STREAMS + s) * dx1
        if HAS_CARRY:
            d += tl.load(CARRY + wide, mask, 0).to(tl.float32)
        tl.store(DR + wide, t * d, mask)
        du = tl.fma(w, d, du)
        tl.store(CP + s * count + pid, tl.sum(tl.sum(d * u, 1), 0))
        if HAS_RETENTION:
            tl.store(CP + (1 * NUM_STREAMS + s) * count + pid, tl.sum(tl.sum(d * old, 1), 0))
        if D0:
            tl.store(
                CP + ((1 + HAS_RETENTION) * NUM_STREAMS + s) * count + pid,
                tl.sum(tl.sum(value * dx0, 1), 0),
            )
        if D1:
            tl.store(
                CP + ((2 + HAS_RETENTION) * NUM_STREAMS + s) * count + pid,
                tl.sum(tl.sum(value * dx1, 1), 0),
            )
    tl.store(DU + off, du, mask)
    if TWO:
        tl.store(DV + off, du, mask)


@dataclass
class Workspace:
    signature: tuple
    previous: torch.Tensor
    grads: tuple
    projections: tuple
    pp: tuple
    gp: tuple
    columns: tuple
    controllers: torch.Tensor
    chunks: torch.Tensor
    cached_x: tuple = ()
    weighted_dy: tuple = ()


def allocate(
    saved,
    *,
    prepare_wgrad,
    gradient_storage="bf16",
    joint_tile=(8, 128, 4),
    prepare_tile=(128, 4),
    experiment=None,
):
    if experiment is not None:
        raise ValueError("experimental layouts require num_streams=3")
    drdtype, gdtype = gradient_dtypes(gradient_storage)
    n = saved.reader_count
    m, h = saved.updates[0].shape
    device = saved.updates[0].device

    def empty(shape, dtype=torch.float32):
        return torch.empty(shape, device=device, dtype=dtype)

    rdtype = residual_dtype(saved.residual_storage)
    grads = (
        empty((m, saved.num_streams * h), drdtype),
        *(empty((m, h), gdtype) for _ in range(n)),
        *(empty((size,)) if size else None for size in saved.controller_sizes),
        *(empty((h,)) for _ in range(n)),
    )
    p = triton.cdiv(m, joint_tile[0]) * triton.cdiv(h, joint_tile[1])
    columns = tuple(
        (
            (
                empty((m, h), torch.bfloat16)
                if getattr(saved, "operand_format", "mxfp8") == "bf16"
                else column_quantizer().make_empty((m, h), dtype=torch.bfloat16, device=device)
            )
            if flag
            else None
        )
        for flag in prepare_wgrad
    )
    signature = (
        m,
        h,
        device,
        saved.residual_storage,
        saved.x_storage,
        getattr(saved, "operand_format", "mxfp8"),
        n,
        saved.num_streams,
        saved.controller_sizes,
        saved.has_retention,
        tuple(prepare_wgrad),
        gradient_storage,
        tuple(joint_tile),
        tuple(prepare_tile),
        experiment,
    )
    return Workspace(
        signature,
        empty((m, saved.num_streams * h), rdtype),
        grads,
        tuple(empty((m,)) for _ in range(n)),
        tuple(empty((m, triton.cdiv(h, prepare_tile[0]))) for _ in range(n)),
        tuple(empty((triton.cdiv(m, 32), h)) for _ in range(n)),
        columns,
        empty(((3 + saved.has_retention) * saved.num_streams, p)),
        empty(((3 + saved.has_retention) * saved.num_streams, triton.cdiv(p, 1024))),
        (
            tuple(empty((m, h)) for _ in range(n))
            if experiment and experiment.cache != "none"
            else ()
        ),
        (
            tuple(empty((m, h)) for _ in range(n))
            if experiment and experiment.cache == "x_dy"
            else ()
        ),
    )


def _workspace_tensors(workspace):
    yield workspace.previous
    for group in (
        workspace.grads,
        workspace.projections,
        workspace.pp,
        workspace.gp,
        workspace.cached_x,
        workspace.weighted_dy,
    ):
        yield from (t for t in group if t is not None)
    for q in workspace.columns:
        if q is not None:
            if type(q) is torch.Tensor:
                yield q
            else:
                yield q._columnwise_data
                yield q._columnwise_scale_inv
    yield workspace.controllers
    yield workspace.chunks


def _check_aliases(workspace, protected):
    # Reject shared storage, including disjoint views: ownership must be explicit.
    owned = list(_workspace_tensors(workspace))
    keys = [t.untyped_storage().data_ptr() for t in owned]
    if len(set(keys)) != len(keys):
        raise ValueError("workspace buffers must own separate storage")
    if set(keys) & {t.untyped_storage().data_ptr() for t in protected if t is not None}:
        raise ValueError("workspace aliases a live input")


def _validate_workspace(
    b, saved, prepare_wgrad, gradient_storage, joint_tile, prepare_tile, experiment=None
):
    drdtype, gdtype = gradient_dtypes(gradient_storage)
    m, h = saved.updates[0].shape
    n, device = saved.reader_count, saved.updates[0].device
    for group, enabled in (
        (b.cached_x, experiment and experiment.cache != "none"),
        (b.weighted_dy, experiment and experiment.cache == "x_dy"),
    ):
        if len(group) != (n if enabled else 0):
            raise ValueError("workspace cache count mismatch")
        for t in group:
            _tensor(t, (m, h), torch.float32, device, "workspace cache")
    rdtype = residual_dtype(saved.residual_storage)
    _tensor(b.previous, (m, saved.num_streams * h), rdtype, device, "workspace previous")
    specs = [((m, saved.num_streams * h), drdtype)] + [((m, h), gdtype)] * n
    specs += [((size,), torch.float32) for size in saved.controller_sizes]
    specs += [((h,), torch.float32)] * n
    if len(b.grads) != len(specs):
        raise ValueError("workspace gradient count mismatch")
    for t, (shape, dtype) in zip(b.grads, specs):
        if shape == (0,):
            if t is not None:
                raise ValueError("disabled retention must not own a gradient")
        else:
            _tensor(t, shape, dtype, device, "workspace gradient")
    for group, shape in (
        (b.projections, (m,)),
        (b.pp, (m, triton.cdiv(h, prepare_tile[0]))),
        (b.gp, (triton.cdiv(m, 32), h)),
    ):
        if len(group) != n:
            raise ValueError("workspace reader count mismatch")
        for t in group:
            _tensor(t, shape, torch.float32, device, "workspace partial")
    p = triton.cdiv(m, joint_tile[0]) * triton.cdiv(h, joint_tile[1])
    _tensor(
        b.controllers,
        ((3 + saved.has_retention) * saved.num_streams, p),
        torch.float32,
        device,
        "workspace controllers",
    )
    _tensor(
        b.chunks,
        ((3 + saved.has_retention) * saved.num_streams, triton.cdiv(p, 1024)),
        torch.float32,
        device,
        "workspace chunks",
    )
    if len(b.columns) != n:
        raise ValueError("workspace column count mismatch")
    for q, flag in zip(b.columns, prepare_wgrad):
        if (q is not None) != flag:
            raise ValueError("workspace column route mismatch")
        if q is not None and getattr(saved, "operand_format", "mxfp8") == "bf16":
            _tensor(q, (m, h), torch.bfloat16, device, "workspace BF16 operand")
        elif q is not None:
            for t, numel in (
                (q._columnwise_data, m * h),
                (q._columnwise_scale_inv, triton.cdiv(h, 128) * triton.cdiv(m, 128) * 512),
            ):
                if (
                    t.numel() != numel
                    or not t.is_contiguous()
                    or t.device != device
                    or t.dtype != torch.uint8
                ):
                    raise ValueError("workspace column storage mismatch")


@torch.no_grad()
def backward(
    saved,
    current,
    dys,
    carry=None,
    *,
    mode="inline",
    reference=None,
    needs=None,
    prepare_wgrad=None,
    workspace=None,
    inverse_mode="baseline",
    gradient_storage="bf16",
    shared_readers=False,
    joint_tile=(8, 128, 4),
    prepare_tile=(128, 4),
    row_major=False,
    tuned=False,
    prepare_layout=None,
    joint_layout=None,
    experiment=None,
    cooperative=False,
):
    """Complete standard A/B backward for independently compiled stream counts.

    Gradient order is (R, updates..., write, retain, reads..., gammas...).
    Controller padding is zeroed. Saved operand_format selects BF16 or MXFP8
    weight-gradient operands. Storage, inactive readers, explicit references,
    shared-reader preparation, and caller-owned workspace replay are supported.
    Three-stream experimental layouts remain in reconstruction_backward.
    """
    if saved.operand_format not in ("bf16", "mxfp8"):
        raise ValueError("invalid saved operand format")
    if cooperative not in (False, "auto"):
        raise ValueError("use the generalized cooperative entry for explicit execution")
    if (
        experiment is not None
        or prepare_layout is not None
        or joint_layout is not None
        or (tuned and saved.has_retention)
    ):
        raise ValueError("experimental three-stream layouts require num_streams=3")
    bf16 = saved.operand_format == "bf16"
    # Identity retention shortens the joint live ranges. The row-major Gluon
    # layout coalesces its stores; leaving registers uncapped avoids the spills
    # seen with the learned-path cap. The learned specialization is unchanged.
    if not saved.has_retention and saved.num_streams == 3:
        row_major = True
    if tuple(prepare_tile) not in ((64, 4), (64, 8), (128, 4), (128, 8), (256, 8)):
        raise ValueError("unsupported preparation tile")
    if tuple(joint_tile) not in ((8, 128, 4), (4, 128, 4), (4, 256, 4), (16, 128, 4)):
        raise ValueError("unsupported joint tile")
    drdtype, gdtype = gradient_dtypes(gradient_storage)
    n = saved.reader_count
    m, h = saved.updates[0].shape
    needs = (True,) * (3 + 3 * n) if needs is None else tuple(needs)
    prepare_wgrad = (False,) * n if prepare_wgrad is None else tuple(prepare_wgrad)
    if (
        inverse_mode not in INVERSE_MODES
        or mode not in ("inline", "control", "saved")
        or len(dys) != n
        or len(needs) != 3 + 3 * n
        or len(prepare_wgrad) != n
    ):
        raise ValueError("invalid mode, reader routes, or gradient needs")
    if not bf16 and any(prepare_wgrad) and (m % 32 or h % 32):
        raise ValueError("MXFP8 requires M and H divisible by 32")
    device = saved.updates[0].device
    dtype = residual_dtype(saved.residual_storage)
    _tensor(current, (m, saved.num_streams * h), dtype, device, "current")
    for dy in dys:
        if dy is not None:
            if dy.dtype not in (torch.bfloat16, gdtype):
                raise ValueError("dY dtype does not match gradient_storage")
            _tensor(dy, (m, h), dy.dtype, device, "dY")
    if carry is not None:
        _tensor(carry, (m, saved.num_streams * h), drdtype, device, "carry gradient")
    previous, xs = None, (None,) * n
    if mode == "control":
        previous, xs = reconstruct(current, saved, inverse_mode=inverse_mode)
    elif mode == "saved":
        if reference is None or reference.current is not current:
            raise ValueError("saved mode requires the matching explicit reference")
        previous, xs = reference.previous, reference.x
    elif reference is not None:
        raise ValueError("candidate must not retain a reference")
    if workspace is None:
        workspace = allocate(
            saved,
            prepare_wgrad=prepare_wgrad,
            gradient_storage=gradient_storage,
            joint_tile=joint_tile,
            prepare_tile=prepare_tile,
            experiment=experiment,
        )
    signature = (
        m,
        h,
        device,
        saved.residual_storage,
        saved.x_storage,
        getattr(saved, "operand_format", "mxfp8"),
        n,
        saved.num_streams,
        saved.controller_sizes,
        saved.has_retention,
        prepare_wgrad,
        gradient_storage,
        tuple(joint_tile),
        tuple(prepare_tile),
        experiment,
    )
    if workspace.signature != signature:
        raise ValueError("workspace configuration mismatch")
    _validate_workspace(
        workspace, saved, prepare_wgrad, gradient_storage, joint_tile, prepare_tile, experiment
    )
    _check_aliases(
        workspace,
        (
            current,
            previous,
            *xs,
            *dys,
            carry,
            *saved.updates,
            saved.coefficients,
            *saved.gammas,
            *saved.rstd,
        ),
    )
    b = workspace

    def endpoints(values):
        return values[0], values[-1]

    def field(name):
        return endpoints(
            tuple(
                (
                    None
                    if q is None
                    else ((q if name == "_columnwise_data" else None) if bf16 else getattr(q, name))
                )
                for q in b.columns
            )
        )

    d0, d1 = dys[0] is not None, n == 2 and dys[1] is not None
    interleave = experiment is not None and experiment.prepare_schedule == "interleave_readers"
    shared = shared_readers and n == 2 and mode == "inline" and not interleave
    prepare_grid = (
        triton.cdiv(m, 128) * 4 if any(prepare_wgrad) else triton.cdiv(m, 32),
        triton.cdiv(triton.cdiv(h, 128) * 128, prepare_tile[0]),
        1 if shared else n,
    )
    prep_kernel, prep_options = _prepare, dict(
        BF16=bf16, NUM_STREAMS=saved.num_streams, HAS_RETENTION=saved.has_retention
    )
    identity_gluon = (
        not saved.has_retention
        and saved.num_streams == 3
        and saved.residual_storage == "fp32"
        and saved.x_storage == "fp32"
    )
    if identity_gluon:
        from .reconstruction_prepare_gluon import _prepare_gluon

        prep_kernel = _prepare_gluon
        prep_options = dict(CONFIG=(4, 32, 1, True, True), WARPS=prepare_tile[1], BF16=bf16)
        prepare_grid = (prepare_grid[1], prepare_grid[0], prepare_grid[2])
    k = prep_kernel[prepare_grid](
        current,
        *endpoints(saved.updates),
        saved.coefficients,
        *endpoints(xs),
        *endpoints(saved.rstd),
        *endpoints(saved.gammas),
        *endpoints(dys),
        *field("_columnwise_data"),
        *field("_columnwise_scale_inv"),
        *endpoints(b.pp),
        *endpoints(b.gp),
        m,
        h,
        n == 2,
        saved.residual_storage != "fp32",
        mode == "inline",
        d0,
        d1,
        prepare_wgrad[0],
        n == 2 and prepare_wgrad[-1],
        prepare_tile[0],
        inverse_mode,
        saved.x_storage == "fp32",
        saved.summed_update,
        shared,
        **prep_options,
        num_warps=prepare_tile[1],
        enable_fp_fusion=False,
    )
    resource(
        f"reconstruction_s{saved.num_streams}_prepare_{h}_{n}_{mode}_{saved.residual_storage}_shared{int(shared)}_c{prepare_tile[0]}w{prepare_tile[1]}_layout{prepare_layout}_experiment{experiment}",
        k,
    )
    k = _finish_reads[(max(triton.cdiv(m, 32), triton.cdiv(h, 32)), 2 * n)](
        *endpoints(b.pp),
        *endpoints(b.gp),
        *endpoints(b.projections),
        *endpoints(b.grads[-n:]),
        m,
        h,
        prepare_tile[0],
        d0,
        d1,
        triton.next_power_of_2(triton.cdiv(h, prepare_tile[0])),
        triton.next_power_of_2(triton.cdiv(m, 32)),
        enable_fp_fusion=False,
    )
    resource(f"reconstruction_s{saved.num_streams}_finish_reads_{h}_{n}", k)
    joint_grid = (triton.cdiv(m, joint_tile[0]), triton.cdiv(h, joint_tile[1]))
    if row_major:
        joint_grid = joint_grid[::-1]
    joint_kernel, joint_options = _joint, dict(
        NUM_STREAMS=saved.num_streams, HAS_RETENTION=saved.has_retention
    )
    if identity_gluon:
        from .reconstruction_joint_gluon import _joint_gluon

        joint_kernel = _joint_gluon
        joint_options = dict(CONFIG=(4, 32, 1, True), WARPS=joint_tile[2], HAS_RETENTION=False)
    k = joint_kernel[joint_grid](
        current,
        previous,
        *endpoints(saved.updates),
        saved.coefficients,
        *endpoints(xs),
        *endpoints(saved.rstd),
        *endpoints(saved.gammas),
        *endpoints(dys),
        carry,
        *endpoints(b.projections),
        b.previous,
        b.grads[0],
        b.grads[1],
        b.grads[n],
        b.controllers,
        m,
        h,
        n == 2,
        saved.residual_storage != "fp32",
        mode == "inline",
        d0,
        d1,
        carry is not None,
        joint_tile[0],
        joint_tile[1],
        inverse_mode,
        saved.x_storage == "fp32",
        saved.summed_update,
        row_major,
        **joint_options,
        num_warps=joint_tile[2],
        enable_fp_fusion=False,
    )
    resource(
        f"reconstruction_s{saved.num_streams}_joint_{h}_{n}_{mode}_{saved.residual_storage}_{joint_tile[0]}x{joint_tile[1]}w{joint_tile[2]}_order{int(row_major)}_layout{joint_layout}_experiment{experiment}",
        k,
    )
    p, chunks = b.controllers.shape[1], b.chunks.shape[1]
    k = _controller_chunks[((3 + saved.has_retention) * saved.num_streams, chunks)](
        b.controllers,
        b.chunks,
        p,
        chunks,
        True,
        saved.has_retention,
        d0,
        d1,
        NUM_STREAMS=saved.num_streams,
        HAS_RETENTION=saved.has_retention,
        enable_fp_fusion=False,
    )
    resource(f"reconstruction_s{saved.num_streams}_controller_chunks_{m}_{h}_{n}", k)
    controllers = b.grads[1 + n : 3 + 2 * n]
    k = _finish_controllers[(saved.num_streams,)](
        b.chunks,
        controllers[0],
        controllers[1],
        controllers[2],
        controllers[-1],
        saved.coefficients,
        True,
        chunks,
        triton.next_power_of_2(max(chunks, *saved.controller_sizes)),
        saved.controller_sizes[0],
        saved.controller_sizes[1],
        saved.controller_sizes[2],
        saved.controller_sizes[-1],
        True,
        saved.has_retention,
        d0,
        d1,
        NUM_STREAMS=saved.num_streams,
        HAS_RETENTION=saved.has_retention,
        enable_fp_fusion=False,
    )
    resource(f"reconstruction_s{saved.num_streams}_controller_finish_{m}_{h}_{n}", k)
    active = (carry is not None or any(dy is not None for dy in dys),) * (3 + n)
    active += tuple(dy is not None for dy in dys) * 2
    grads = tuple(g if need and on else None for g, need, on in zip(b.grads, needs, active))
    return b.previous, grads, b
