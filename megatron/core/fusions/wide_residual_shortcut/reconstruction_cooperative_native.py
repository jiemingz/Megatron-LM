# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Native cooperative B backward for the validated GB300 schedules.

The incumbent Nano/Super/Ultra schedules and width-generalized kernel live here.
Expanded shapes require offline qualification. Historical alternatives and
benchmark baselines remain in reconstruction_cooperative.py and results/.

Each resident CTA (thread block) owns 32 rows x 128 columns. A group of H/128
CTAs spans the full hidden width and persists across row tiles:
  K1 reconstructs two readers, writes projection/gamma partials and consumer operands.
  K2 acquires all column partials and repeats the small reduction in each CTA,
     avoiding a second cross-CTA handoff for the completed projection.
  K3 computes gradients and the matched inverse in 16-row slices, recycling
     shared storage for carry and the next inputs as each lifetime ends.
These are stages of one kernel. R (three residual streams) and dy (two reader
gradients) have one original global read; U is the saved summed update.

Four warps, 128 columns and a 168-register cap preserve three resident CTAs/SM
in the validated full-shape builds. The host queries actual compiled residency
before launch. Complete backward also includes epoch reset and final reductions.
"""

import ctypes
from dataclasses import dataclass

import torch
import triton
import triton.language as tl
from triton.experimental import gluon as g
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language import (
    BlockedLayout,
    NVMMASharedLayout,
    SliceLayout,
    SwizzledSharedLayout,
)
from triton.experimental.gluon.language.nvidia.hopper import fence_async_shared, mbarrier, tma
from triton.experimental.gluon.nvidia.hopper import TensorDescriptor

from .gluon_helpers import _rmsnorm_dx
from .helpers import column_quantizer
from .reconstruction import _tensor
from .reconstruction_dispatch import INCUMBENTS, WIDTHS, Schedule
from .reconstruction_stream_reductions import _controller_chunks


def _inverse_division_ptx():
    # Sixteen independent values share a scalar denominator and one warp-wide
    # domain guard. NaNs propagate into that guard; exceptional magnitudes use
    # native div.rn. In the validated domain, two residual corrections using a
    # cached RN reciprocal preserve the matched quotient with fewer divisions.
    # Operand groups are outputs[0:16], numerators[16:32], denominators[32:48],
    # reciprocals[48:64]. Consume every input before moving output registers.
    lines = [
        "{ .reg .f32 nmin,nmax,dmin,dmax,q<16>,r,nb;",
        ".reg .pred nvalid,dvalid,valid,upper;",
        "mov.b32 nmin,0x7f800000; mov.b32 dmin,0x7f800000; mov.b32 nmax,0; mov.b32 dmax,0;",
    ]
    for i in range(0, 16, 2):
        lines += [
            f"min.NaN.abs.f32 nmin,nmin,${16+i},${17+i};",
            f"max.NaN.abs.f32 nmax,nmax,${16+i},${17+i};",
        ]
    lines += [
        "setp.ge.f32 nvalid,nmin,0f2f800000; setp.le.f32 upper,nmax,0f4f800000;",
        "and.pred nvalid,nvalid,upper;",
        "abs.f32 dmin,$32; mov.f32 dmax,dmin;",
        "setp.ge.f32 dvalid,dmin,0f2f800000; setp.le.f32 upper,dmax,0f4f800000;",
        "and.pred dvalid,dvalid,upper; and.pred valid,nvalid,dvalid;",
        "vote.sync.all.pred valid,valid,0xffffffff; @!valid bra SLOW;",
        ".reg .b64 pa,pb,pr,pn,pq,pe;",
    ]
    for i in range(0, 16, 2):
        a, b, r = 16 + i, 32 + i, 48 + i
        lines += [
            f"mov.b64 pa,{{${a},${a+1}}}; mov.b64 pb,{{${b},${b+1}}}; mov.b64 pr,{{${r},${r+1}}};",
            "xor.b64 pn,pb,0x8000000080000000; mul.rn.f32x2 pq,pa,pr;",
            "fma.rn.f32x2 pe,pn,pq,pa; fma.rn.f32x2 pq,pe,pr,pq;",
            "fma.rn.f32x2 pe,pn,pq,pa; fma.rn.f32x2 pq,pe,pr,pq;",
            f"mov.b64 {{q{i},q{i+1}}},pq;",
        ]
    lines += ["bra DONE; SLOW:"]
    lines += [f"div.rn.f32 q{i},${16+i},${32+i};" for i in range(16)]
    lines += ["DONE:"]
    lines += [f"mov.f32 ${i},q{i};" for i in range(16)]
    return "\n".join(lines + ["}"])


_MARKSTEIN_UNIFORM_PTX = gl.constexpr(_inverse_division_ptx())
_MARKSTEIN_WARP_CONSTRAINTS = gl.constexpr(",".join(["=f"] * 16 + ["f"] * 48))


@g.jit
def _group_publish_counter_pointer(ARRIVAL, WIDTH: gl.constexpr):
    # Match the original release-RMW protocol while exposing address arithmetic
    # to the compiler. The CTA barrier publishes every producer warp's stores.
    # One leader increments once per tile; padded per-group counters keep
    # independent groups apart. epoch is interpreted by the waiter, not written
    # here: counts accumulate throughout this launch and reset between calls.
    pointer = ARRIVAL + (gl.program_id(0) // WIDTH) * 32
    gl.barrier()
    gl.inline_asm_elementwise(
        "{ .reg .u32 tid,old; .reg .pred p; mov.u32 tid,%tid.x; setp.eq.u32 p,tid,0; @p atom.release.gpu.global.add.u32 old,[$1],1; mov.u32 $0,0; }",
        "=r,l",
        [pointer],
        dtype=gl.int32,
        is_pure=False,
        pack=1,
    )


@g.jit
def _group_wait_counter_pointer(ARRIVAL, epoch, WIDTH: gl.constexpr):
    pointer = ARRIVAL + (gl.program_id(0) // WIDTH) * 32
    # The leader acquires all producer releases; the final CTA barrier makes
    # those writes visible to every consumer warp. PP remains double buffered.
    # Use >= rather than ==: fast producers may already publish the next tile.
    # Relaxed polling avoids paying an acquire on every unsuccessful probe;
    # the acquire fence is still required before reading published partials.
    target = epoch * WIDTH
    gl.barrier()
    gl.inline_asm_elementwise(
        "{ .reg .u32 tid,v; .reg .pred leader,ready; mov.u32 tid,%tid.x; setp.eq.u32 leader,tid,0; @!leader bra DONE; POLL: ld.relaxed.gpu.global.u32 v,[$1]; setp.ge.u32 ready,v,$2; @!ready bra POLL; fence.acquire.gpu; DONE: mov.u32 $0,0; }",
        "=r,l,r",
        [pointer, target],
        dtype=gl.int32,
        is_pure=False,
        pack=1,
    )
    gl.barrier()


@g.jit
def _controller_warps(value):
    """Keep lane-group sums; the final controller kernel combines them."""
    # Partial sums trade scratch traffic for fewer in-CTA reduction steps.
    # subwarp16 won in the accepted schedule; the complete-call benchmark also
    # charges the later kernels that combine these partials.
    LC: gl.constexpr = 8
    WC: gl.constexpr = 4
    # Fixed 16x128 ownership gives each element one contributor. Keep eight
    # subwarp sums so the final kernels can finish the controller reduction.
    V: gl.constexpr = 4
    LR: gl.constexpr = 32 // LC
    WR: gl.constexpr = 4 // WC
    RR: gl.constexpr = 16 // (WR * LR)
    CC: gl.constexpr = 128 // (WC * LC * V)
    gl.static_assert(True)
    owned = gl.reshape(value, (RR, WR, LR, CC, WC, LC, V))
    owned = gl.permute(owned, (1, 4, 2, 5, 0, 3, 6))
    owned = gl.reshape(owned, (4, 32, RR * CC * V))
    owned = gl.convert_layout(
        owned,
        gl.constexpr(BlockedLayout([1, 1, RR * CC * V], [1, 32, 1], [4, 1, 1], [2, 1, 0])),
        assert_trivial=True,
    )
    per_thread = gl.sum(owned, 2)
    grouped = gl.reshape(per_thread, (4 * 32 // 16, 16))
    return gl.sum(grouped, 1)


@g.jit
def _markstein_uniform_divide(numerator, denominator, reciprocal):
    # This cheaper scalar-denominator guard is valid only with one shared
    # coefficient; numerator magnitudes still require the full warp guard.
    gl.static_assert(denominator.numel == 1)
    return gl.inline_asm_elementwise(
        _MARKSTEIN_UNIFORM_PTX,
        _MARKSTEIN_WARP_CONSTRAINTS,
        [numerator, denominator, reciprocal],
        dtype=gl.float32,
        is_pure=False,
        pack=16,
    )


@g.jit
def _matched_inverse(r, u, w, t, rows, STREAM, reciprocal, HAS_RETENTION: gl.constexpr = True):
    # Protect the forward's non-contracted path even when gradient FMAs
    # are enabled. Quantization reconstruction already uses explicit FMAs.
    separate = gl.inline_asm_elementwise(
        "{ .reg .f32 p; mul.rn.f32 p,$2,$3; sub.rn.f32 $0,$1,p; }",
        "=f,f,f,f",
        [r, w, u],
        dtype=gl.float32,
        is_pure=True,
        pack=1,
    )
    contracted = (STREAM == 0) | ((STREAM == 1) & (rows % 32 < 12))
    numerator = gl.where(contracted, gl.fma(-w, u, r), separate)
    if HAS_RETENTION:
        return _markstein_uniform_divide(numerator, t, reciprocal)
    return numerator


@g.jit
def _shared_part(tile, part):
    return tile.slice(part * 16, 16)


@g.jit
def _coefficient(cs, index):
    indices = gl.full((1,), index, gl.int32, gl.constexpr(BlockedLayout([1], [32], [4], [0])))
    return gl.sum(cs.gather(indices, 0), 0)


@g.jit
def _fma2(a, b, c):
    # Blackwell's packed instruction performs two independent RN FP32 FMAs.
    # Packing is useful in K1's reader-axis layout; it is not a blanket license
    # to pack K3 arithmetic, where tested variants increased register spills.
    # Two independent round-to-nearest FP32 FMAs, preserving operand order.
    return gl.inline_asm_elementwise(
        """{ .reg .b64 a,b,c,r;
    mov.b64 a,{$2,$3}; mov.b64 b,{$4,$5}; mov.b64 c,{$6,$7};
    fma.rn.f32x2 r,a,b,c; mov.b64 {$0,$1},r; }""",
        "=f,=f,f,f,f,f,f,f",
        [a, b, c],
        dtype=gl.float32,
        is_pure=True,
        pack=2,
    )


@g.jit
def _k3_part(
    U,
    PROJ,
    OUT,
    DR,
    DU,
    DV,
    CP,
    residuals,
    ds,
    rs0,
    rs1,
    rp,
    cp,
    part,
    M: gl.constexpr,
    CARRY_DESC,
    carry_barrier,
    NEXT_R_DESC,
    NEXT_DY0_DESC,
    NEXT_DY1_DESC,
    copy_barrier,
    next_rp,
    prefetch_active,
    carry_zero,
    gs,
    cs,
    reciprocals,
    zero_barrier,
    carry_barrier2,
    H: gl.constexpr,
    HAS_RETENTION: gl.constexpr = True,
):
    # K1 needs 32 rows for exact quantization, but K3 does not. Working on
    # 16-row slices bounds live gradients/inverse temporaries and permits the
    # shared-memory lifetimes below to overlap transfers with useful arithmetic.
    NT: gl.constexpr = H // 128
    LANES: gl.constexpr = 16
    PARTIALS: gl.constexpr = 4 * 32 // LANES
    P: gl.constexpr = gl.cdiv(M, 16) * NT * PARTIALS
    L: gl.constexpr = BlockedLayout([1, 4], [4, 8], [1, 4], [1, 0])
    rs0 = gl.convert_layout(rs0, gl.constexpr(SliceLayout(1, L)))
    rs1 = gl.convert_layout(rs1, gl.constexpr(SliceLayout(1, L)))
    cc = cp * 128 + gl.arange(0, 128, layout=gl.constexpr(SliceLayout(0, L)))
    cols = cc[None, :]
    kr = part * 16 + gl.arange(0, 16, layout=gl.constexpr(SliceLayout(1, L)))
    k3_rows = (rp * 32 + kr)[:, None]
    k3_mask = k3_rows < M
    k3_off = k3_rows * H + cols
    coefficients = ()
    for i in gl.static_range(12):
        coefficients += (_coefficient(cs, i),)
    dxs = ()
    for reader in gl.static_range(2):
        k3_dz = _shared_part(ds.index(reader), part).load(L).to(gl.float32)
        k3_gamma = gs.index(reader).load(gl.constexpr(SliceLayout(0, L)))
        k3_dz = k3_dz * k3_gamma[None, :]
        projection = gl.gather(PROJ.index(reader).load(gl.constexpr(SliceLayout(1, L))), kr, 0)
        k3_rs = gl.gather(rs0 if reader == 0 else rs1, kr, 0)
        k3_x = gl.full((16, 128), 0, gl.float32, L)
        # Recompute z from retained shared R. Extra FMAs cost less
        # here than retaining two more FP32 planes or rereading R
        # globally; the smaller working set preserves residency.
        for s in gl.static_range(3):
            reconstructed_r = _shared_part(residuals[s], part).load(L)
            beta = coefficients[6 + reader * 3 + s]
            k3_x = gl.fma(beta, reconstructed_r, k3_x)
        k3_z = k3_x * k3_rs[:, None]
        dxs += (_rmsnorm_dx(k3_z, projection[:, None], k3_dz, k3_rs[:, None]),)
    u = gl.load(U + k3_off, k3_mask, 0).to(gl.float32)
    du = gl.full((16, 128), 0, gl.float32, L)
    for s in gl.static_range(3):
        k3_r = _shared_part(residuals[s], part).load(L)
        # Finish R*dx controller partials while this stream's R is live.
        # Reducing now avoids carrying beta intermediates across the
        # inverse/carry work; R itself is still needed for the inverse.
        for reader in gl.static_range(2):
            beta_value = k3_r * dxs[reader]
            beta_total = _controller_warps(beta_value)
            beta_lane = gl.arange(0, PARTIALS, layout=beta_total.type.layout)
            gl.store(
                CP
                + ((reader + 1 + HAS_RETENTION) * 3 + s) * P
                + ((rp * (32 // 16) + part) * NT + cp) * PARTIALS
                + beta_lane,
                beta_total,
                rp * 32 < M,
            )
        if s == 0:
            # R0's shared slice is dead once its value and reader gradients are
            # in registers. CTA synchronization plus the async-proxy fence
            # permits TMA to overwrite it with carry1 during inverse0.
            # R0's shared consumers have finished; its register value remains
            # live through the inverse. Start carry1 before that arithmetic.
            gl.barrier()
            fence_async_shared()
            mbarrier.expect(carry_barrier, 16 * 128 * 4)
            tma.async_copy_global_to_shared(
                CARRY_DESC,
                [rp * 32 + part * 16, H + cp * 128],
                carry_barrier,
                residuals[0].slice(part * 16, 16),
            )
        if s == 1:
            # A separate epoch lets carry2 overlap inverse1 without waiting
            # for this stream's carry1 transfer on the first barrier.
            gl.barrier()
            fence_async_shared()
            mbarrier.expect(carry_barrier2, 16 * 128 * 4)
            tma.async_copy_global_to_shared(
                CARRY_DESC,
                [rp * 32 + part * 16, (2 * H) + cp * 128],
                carry_barrier2,
                residuals[1].slice(part * 16, 16),
            )
        if s == 2:
            beta0 = coefficients[6 + s]
            beta1 = coefficients[9 + s]
            d = beta0 * dxs[0] + beta1 * dxs[1]
        wide = k3_rows * (3 * H) + s * H + cols
        if s == 2:
            # Consume carry2 before its shared slice is recycled for next R1.
            mbarrier.wait(carry_barrier2, part & 1, deps=(residuals[1],))
            carry_value = residuals[1].slice(part * 16, 16).load(L)
        if s == 2:
            d = d + carry_value
        if s == 2:
            # Current R2 and carry2 have been read into registers. The reader
            # gradients are now dead; start next R1/R2 before the last inverse.
            gl.barrier()
            fence_async_shared()
            if prefetch_active:
                nr = next_rp * 32 + part * 16
                tma.async_copy_global_to_shared(
                    NEXT_R_DESC, [nr, H + cp * 128], copy_barrier, residuals[1].slice(part * 16, 16)
                )
                tma.async_copy_global_to_shared(
                    NEXT_R_DESC,
                    [nr, (2 * H) + cp * 128],
                    copy_barrier,
                    residuals[2].slice(part * 16, 16),
                )
        w, t = coefficients[s], (coefficients[3 + s] if HAS_RETENTION else 1.0)
        old = _matched_inverse(k3_r, u, w, t, k3_rows, s, reciprocals[s], HAS_RETENTION)
        gl.store(OUT + wide, old, k3_mask)
        if s != 2:
            if s == 0:
                # Defer the carry wait until after the independent inverse.
                # The dedicated carry0 buffer avoids overwriting live R/dy;
                # its mbarrier phase toggles once per 16-row part.
                mbarrier.wait(zero_barrier, part & 1, deps=(carry_zero,))
                carry_value = carry_zero.load(L)
            else:
                mbarrier.wait(carry_barrier, part & 1, deps=(residuals[0],))
                carry_value = residuals[0].slice(part * 16, 16).load(L)
            beta0 = coefficients[6 + s]
            beta1 = coefficients[9 + s]
            d = beta0 * dxs[0] + beta1 * dxs[1] + carry_value
        if s != 2:
            # All current shared readers finish before either asynchronous reuse.
            gl.barrier()
            fence_async_shared()
            if s == 0 and part == 0:
                mbarrier.expect(zero_barrier, 16 * 128 * 4)
                tma.async_copy_global_to_shared(
                    CARRY_DESC, [rp * 32 + 16, cp * 128], zero_barrier, carry_zero
                )
            if prefetch_active:
                nr = next_rp * 32 + part * 16
                if s == 0:
                    tma.async_copy_global_to_shared(
                        NEXT_DY0_DESC,
                        [nr, cp * 128],
                        copy_barrier,
                        ds.index(0).slice(part * 16, 16),
                    )
                    tma.async_copy_global_to_shared(
                        NEXT_DY1_DESC,
                        [nr, cp * 128],
                        copy_barrier,
                        ds.index(1).slice(part * 16, 16),
                    )
                if s == 1:
                    tma.async_copy_global_to_shared(
                        NEXT_R_DESC, [nr, cp * 128], copy_barrier, residuals[0].slice(part * 16, 16)
                    )
        # With identity retention the write derivative is one, but d already
        # includes both reader contributions. Returning/aliasing the incoming
        # carry here would lose those gradients. R is also still needed above
        # for reader-controller gradients and matched residual reconstruction.
        gl.store(DR + wide, t * d, k3_mask)
        # Stream-local outputs can leave registers now. du accumulates across
        # streams; controller partials are written for separate final reduction.
        du = gl.fma(w, d, du)
        for kind in gl.static_range(1 + HAS_RETENTION):
            value = d * u if kind == 0 else d * old
            total = _controller_warps(value)
            warp = gl.arange(0, PARTIALS, layout=total.type.layout)
            gl.store(
                CP + (kind * 3 + s) * P + ((rp * (32 // 16) + part) * NT + cp) * PARTIALS + warp,
                total,
                rp * 32 < M,
            )
    gl.store(DU + k3_off, du, k3_mask)
    # The summed update gives equal gradients to both branches, but the API
    # requires independent output storage rather than aliased tensors.
    gl.store(DV + k3_off, du, k3_mask)


@g.jit
def _partial_offset(
    reader, group, rr, cc, GROUPS: gl.constexpr, NT: gl.constexpr, TILE: gl.constexpr
):
    # Producers scatter one column; reducers read all columns for each row.
    # TILE blocks those partials to change reducer locality without changing
    # their values. Ultra benefits from TILE=16 at 80 participants; Nano/Super
    # keep TILE=0 because transferring Ultra's layout did not improve them.
    if TILE:
        return (
            (reader * GROUPS + group) * 32 * NT + (cc // TILE) * (32 * TILE) + rr * TILE + cc % TILE
        )
    else:
        return ((reader * GROUPS + group) * 32 + rr) * NT + cc


@g.jit
def _quant_pair_axis(y, Q0, Q1, S0, S1, rr, cc, rp, reader, M: gl.constexpr, H: gl.constexpr):
    # TE column quantization groups exactly 32 rows. Reduce only that axis:
    # combining readers would silently change their E8M0 scales/FP8 operands.
    # Preserve upward scale rounding and the original FP8 conversion order.
    amax = gl.max(gl.abs(y), 1)
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
    # Construct the reciprocal scale without division. Exponent 254 needs a
    # subnormal FP32 reciprocal; exponent 255 preserves the NaN encoding.
    bits = gl.where(exponent == 254, 0x00400000, (254 - exponent) << 23)
    bits = gl.where(exponent == 255, 0x7FFFFFFF, bits)
    q = (
        (y * bits.to(gl.float32, bitcast=True)[:, None, :])
        .to(gl.float8e4nv)
        .to(gl.uint8, bitcast=True)
    )
    qp = gl.where(reader == 0, Q0, Q1)
    rc = gl.convert_layout(reader, gl.constexpr(SliceLayout(1, SliceLayout(1, y.type.layout))))
    sp = gl.where(rc == 0, S0, S1)
    gl.store(
        qp[:, None, None] + rr[None, :, None] * H + cc[None, None, :], q, rr[None, :, None] < M
    )
    index = (
        ((cc // 128) * gl.cdiv(M, 128) + rp // 4) * 512
        + (cc % 32) * 16
        + (cc % 128 // 32) * 4
        + rp % 4
    )
    # Write the padded scale extent even when this tile has no live Q rows.
    gl.store(sp[:, None] + index[None, :], exponent.to(gl.uint8), rp < gl.cdiv(M, 128) * 4)


@g.jit
def _reader_axis(
    C,
    PP,
    GP0,
    GP1,
    Q0,
    Q1,
    S0,
    S1,
    residuals,
    ds,
    rs0,
    rs1,
    rp,
    cp,
    group,
    gs,
    ARRIVAL,
    M: gl.constexpr,
    GRID: gl.constexpr,
    H: gl.constexpr,
    PP_TILE: gl.constexpr,
    BF16: gl.constexpr = False,
):
    NT: gl.constexpr = H // 128
    GROUPS: gl.constexpr = GRID // NT
    # Split the two readers across warp groups. R is shared after its sole
    # global TMA read; each warp group loads only its own shared dy plane.
    L: gl.constexpr = BlockedLayout([1, 1, 4], [1, 4, 8], [2, 1, 4 // 2], [2, 1, 0])
    RL: gl.constexpr = SliceLayout(1, SliceLayout(2, L))
    RR: gl.constexpr = SliceLayout(0, SliceLayout(2, L))
    CL: gl.constexpr = SliceLayout(0, SliceLayout(1, L))
    reader = gl.arange(0, 2, layout=gl.constexpr(RL))
    rr0 = gl.arange(0, 32, layout=gl.constexpr(RR))
    rr = rp * 32 + rr0
    cc = cp * 128 + gl.arange(0, 128, layout=gl.constexpr(CL))
    x = gl.full((2, 32, 128), 0, gl.float32, L)
    # Pack independent FP32 FMAs, preserving the three-stream accumulation
    # order and each operation's RN result. This reduces instruction count
    # without changing the values that subsequently determine FP8 scales.
    for stream in gl.static_range(3):
        rv = residuals[stream].load(gl.constexpr(SliceLayout(0, L)))
        beta = gl.load(C + 6 + 3 * reader + stream)
        x = _fma2(beta[:, None, None], rv[None, :, :], x)
    rst = gl.convert_layout(gl.permute(gl.join(rs0, rs1), (1, 0)), gl.constexpr(SliceLayout(2, L)))
    z = x * rst[:, :, None]
    gamma = gs._reinterpret(
        gl.float32, [2, 128], gl.constexpr(SwizzledSharedLayout(1, 1, 1, [1, 0]))
    ).load(gl.constexpr(SliceLayout(1, L)))
    warp_reader = gl.inline_asm_elementwise(
        "{ .reg .u32 tid; mov.u32 tid,%tid.x; div.u32 $0,tid,$1; }",
        "=r,n",
        [4 * 16],
        dtype=gl.int32,
        is_pure=False,
        pack=1,
    )
    dy = ds.index(warp_reader).load(gl.constexpr(SliceLayout(0, L)))[None, :, :].to(gl.float32)
    # Broadcast onto the reader layout without loading both shared dy planes.
    dy = gl.where(reader[:, None, None] >= 0, dy, dy)
    dz = dy * gamma[:, None, :]
    # PP reduces columns for the row-wise RMSNorm projection; GP reduces rows
    # for gamma's final reduction. Divide PP by the full hidden width, not this
    # CTA's column count, so summing participant partials yields the global mean.
    gl.store(
        PP + _partial_offset(reader[:, None], group, rr0[None, :], cp, GROUPS, NT, PP_TILE),
        gl.sum(dz * z, 2) / H,
    )
    rc = gl.convert_layout(reader, gl.constexpr(SliceLayout(1, SliceLayout(1, L))))
    gl.store(
        gl.where(rc == 0, GP0, GP1)[:, None] + rp * H + cc[None, :], gl.sum(dy * z, 1), rp * 32 < M
    )
    # Release completed PP/GP before quantizing. Quantized outputs are CTA-owned,
    # so peers can reduce PP while we finish quantization; our own K3 still
    # follows quantization and all outputs precede kernel completion.
    _group_publish_counter_pointer(ARRIVAL, NT)
    if BF16:
        qp = gl.where(reader == 0, Q0, Q1)
        gl.store(
            qp[:, None, None] + rr[None, :, None] * H + cc[None, None, :],
            (z * gamma[:, None, :]).to(gl.bfloat16),
            rr[None, :, None] < M,
        )
    else:
        _quant_pair_axis(z * gamma[:, None, :], Q0, Q1, S0, S1, rr, cc, rp, reader, M, H)


@g.jit
def _prefetch_u_only(U, rp, cp, M: gl.constexpr, H: gl.constexpr):
    # Warm U in L2 while projection executes; K3 still performs the real load.
    # This hides U load latency without allocating another shared-memory plane.
    # The accepted presets place U hints after arrival acquisition: moving them
    # earlier lost performance in Ultra's paired ablation.
    i = gl.arange(0, 32 * 128 // 64, layout=gl.constexpr(BlockedLayout([1], [32], [4], [0])))
    rr = rp * 32 + i // (128 // 64)
    cc = cp * 128 + (i % (128 // 64)) * 64
    ASM: gl.constexpr = (
        "{ .reg .pred p; setp.ne.u32 p,$2,0; @p prefetch.global." + "L2" + " [$1]; mov.u32 $0,0; }"
    )
    gl.inline_asm_elementwise(
        ASM,
        "=r,l,r",
        [U + rr * H + cc, (rr < M).to(gl.int32)],
        dtype=gl.int32,
        is_pure=False,
        pack=1,
    )


@g.jit
def _prefetch_carry_only(CARRY, rp, cp, M: gl.constexpr, H: gl.constexpr, ROWS: gl.constexpr):
    # Hint upcoming carry rows before their TMA transfers. Only the designated
    # threads issue requests, avoiding duplicate requests from every warp.
    # These are cache hints, not synchronization; the TMA waits remain required.
    rr = rp * 32 + gl.arange(0, ROWS, layout=gl.constexpr(BlockedLayout([1], [32], [4], [0])))
    base = CARRY + rr * (3 * H) + cp * 128
    for stream in gl.static_range(1, 3):
        gl.inline_asm_elementwise(
            "{ .reg .pred p,owner; .reg .u32 tid; mov.u32 tid,%tid.x; setp.lt.u32 owner,tid,$4; setp.ne.u32 p,$2,0; and.pred p,p,owner; @p cp.async.bulk.prefetch.L2.global [$1],$3; mov.u32 $0,0; }",
            "=r,l,r,n,n",
            [base + stream * H, (rr < M).to(gl.int32), 128 * 4, ROWS],
            dtype=gl.int32,
            is_pure=False,
            pack=1,
        )


@g.jit
def _cooperative(
    R_DESC,
    DY0_DESC,
    DY1_DESC,
    U,
    C,
    RS0,
    RS1,
    G0,
    G1,
    CARRY,
    Q0,
    Q1,
    S0,
    S1,
    PP,
    GP0,
    GP1,
    OUT,
    DR,
    DU,
    DV,
    CP,
    ARRIVAL,
    M: gl.constexpr,
    GRID: gl.constexpr,
    CARRY_DESC,
    NEXT_R_DESC,
    NEXT_DY0_DESC,
    NEXT_DY1_DESC,
    PROJECTION_LAYOUT: gl.constexpr,
    CARRY_PREFETCH_ROWS: gl.constexpr,
    CARRY_PREFETCH_BEFORE_WAIT: gl.constexpr,
    PP_TILE: gl.constexpr,
    H: gl.constexpr,
    BF16: gl.constexpr = False,
    HAS_RETENTION: gl.constexpr = True,
):
    # H specializes all strides and group geometry. One group covers the full
    # hidden dimension; whole groups must fit concurrently on the device so
    # their spinning consumers cannot block unscheduled producers indefinitely.
    NT: gl.constexpr = H // 128
    gl.static_assert(H % 128 == 0 and GRID >= NT and GRID % NT == 0)
    if PP_TILE:
        gl.static_assert(NT % PP_TILE == 0)
    L: gl.constexpr = BlockedLayout([1, 4], [4, 8], [1, 4], [1, 0])
    SL: gl.constexpr = NVMMASharedLayout(swizzle_byte_width=128, element_bitwidth=32, rank=2)
    residuals = (
        gl.allocate_shared_memory(gl.float32, [32, 128], SL),
        gl.allocate_shared_memory(gl.float32, [32, 128], SL),
        gl.allocate_shared_memory(gl.float32, [32, 128], SL),
    )
    DT: gl.constexpr = gl.bfloat16
    # Retain BF16 dy and apply gamma again in K3.
    # Keeping FP32 dz instead doubles this state. The 64-byte BF16 swizzle is
    # paired with the reader-axis layout, not an arbitrary TMA default.
    DSL: gl.constexpr = NVMMASharedLayout(swizzle_byte_width=64, element_bitwidth=16, rank=2)
    ds = gl.allocate_shared_memory(DT, [2, 32, 128], DSL)
    carry_zero = gl.allocate_shared_memory(gl.float32, [16, 128], SL)
    # U uses direct loads plus L2 hints. This small FP32 carry0 tile starts
    # its transfer before K1 without clobbering
    # any retained inputs. Separate carry1/2 barriers allow overlap.
    zero_barrier = mbarrier.allocate_mbarrier()
    mbarrier.init(zero_barrier, count=1)
    carry_barrier2 = mbarrier.allocate_mbarrier()
    mbarrier.init(carry_barrier2, count=1)
    carry_barrier = mbarrier.allocate_mbarrier()
    mbarrier.init(carry_barrier, count=1)
    copy_barrier = mbarrier.allocate_mbarrier()
    mbarrier.init(copy_barrier, count=1)
    # Keep the tiny 2x32 result CTA-private in shared memory. A register-only
    # trial spilled and slowed Nano/Ultra despite removing these stores.
    local_projection = gl.allocate_shared_memory(
        gl.float32, [2, 32], gl.constexpr(SwizzledSharedLayout(1, 1, 1, [0]))
    )
    GROUPS: gl.constexpr = GRID // NT
    bid = gl.program_id(0)
    cp, group = bid % NT, bid // NT
    rr0 = gl.arange(0, 32, layout=gl.constexpr(SliceLayout(1, L)))
    cc0 = gl.arange(0, 128, layout=gl.constexpr(SliceLayout(0, L)))
    cc = cp * 128 + cc0
    cs = gl.allocate_shared_memory(
        gl.float32, [16], gl.constexpr(SwizzledSharedLayout(1, 1, 1, [0]))
    )
    ci = gl.arange(0, 16, layout=gl.constexpr(BlockedLayout([1], [32], [4], [0])))
    cs.store(gl.load(C + ci, ci < 12, 0))
    gs = gl.allocate_shared_memory(
        gl.float32, [2, 128], gl.constexpr(SwizzledSharedLayout(1, 1, 1, [0]))
    )
    gs.index(0).store(gl.load(G0 + cc).to(gl.float32))
    gs.index(1).store(gl.load(G1 + cc).to(gl.float32))
    reciprocals = ()
    # The three inverse denominators are launch-invariant scalar
    # coefficients: one RN reciprocal each can serve every tile and slice.
    for stream in gl.static_range(3):
        reciprocals += (gl.div_rn(1.0, gl.load(C + 3 + stream)) if HAS_RETENTION else 1.0,)
    # Quantization scales include the complete 128-row padded extent. Each
    # independent group can stop after its own last tile. Suppress every next
    # input prefetch on that iteration so no TMA operation outlives its storage.
    group_iters = gl.maximum(gl.cdiv(gl.cdiv(M, 128) * 4 - group, GROUPS), 0)
    for iteration in tl.range(group_iters, disable_licm=True):
        rp = iteration * GROUPS + group
        # Two PP slots prevent a fast CTA's next publication from overwriting
        # a slow CTA's current reduction. Reusing slot n for n+2 requires all
        # n+1 arrivals, which implies every producer finished consuming tile n.
        # This replaces a group-wide end-of-iteration barrier, not the acquire.
        pp_active = PP + (iteration % 2) * (2 * GROUPS * 32 * NT)
        rr = rp * 32 + rr0
        # Finish all previous-iteration shared consumers before asynchronous overwrite.
        gl.barrier()
        if iteration == 0:
            # Bootstrap only: later copies
            # were issued by K3 as shared slices died. The wait covers
            # all three FP32 R planes plus both BF16 dy planes (16
            # bytes/element in the accepted schedule), with one phase
            # per iteration and no duplicate original-input loads.
            mbarrier.expect(copy_barrier, 32 * 128 * 16)
            for s in gl.static_range(3):
                tma.async_copy_global_to_shared(
                    R_DESC, [rp * 32, s * H + cp * 128], copy_barrier, residuals[s]
                )
            tma.async_copy_global_to_shared(
                DY0_DESC, [rp * 32, cp * 128], copy_barrier, ds.index(0)
            )
            tma.async_copy_global_to_shared(
                DY1_DESC, [rp * 32, cp * 128], copy_barrier, ds.index(1)
            )
        copy_deps = (ds,)
        for s in gl.static_range(3):
            copy_deps += (residuals[s],)
        mbarrier.wait(copy_barrier, iteration & 1, deps=copy_deps)
        # The previous iteration consumed the second slice before its
        # existing CTA reuse barrier. Start slice zero ahead of K1.
        gl.barrier()
        fence_async_shared()
        mbarrier.expect(zero_barrier, 16 * 128 * 4)
        tma.async_copy_global_to_shared(CARRY_DESC, [rp * 32, cp * 128], zero_barrier, carry_zero)
        rs0 = gl.load(RS0 + rr, rr < M, 0)
        rs1 = gl.load(RS1 + rr, rr < M, 0)
        _reader_axis(
            C,
            pp_active,
            GP0,
            GP1,
            Q0,
            Q1,
            S0,
            S1,
            residuals,
            ds,
            rs0,
            rs1,
            rp,
            cp,
            group,
            gs,
            ARRIVAL,
            M,
            GRID,
            H,
            PP_TILE=PP_TILE,
            BF16=BF16,
        )
        # Ultra can overlap carry cache hints with the group arrival wait.
        # U retains its separately tuned position after arrival acquisition.
        if CARRY_PREFETCH_ROWS and CARRY_PREFETCH_BEFORE_WAIT:
            _prefetch_carry_only(CARRY, rp, cp, M, H, CARRY_PREFETCH_ROWS)
        _group_wait_counter_pointer(ARRIVAL, iteration + 1, NT)
        # Cache hints overlap the projection with upcoming U/carry reads.
        # They do not issue additional reads of original R or dy.
        if CARRY_PREFETCH_ROWS and not CARRY_PREFETCH_BEFORE_WAIT:
            _prefetch_carry_only(CARRY, rp, cp, M, H, CARRY_PREFETCH_ROWS)
        _prefetch_u_only(U, rp, cp, M, H)
        # Projection has a different shape from K1/K3, so it gets its own
        # layout. The per-model row/vector choices balance coalescing,
        # reduction work and register pressure; copying Ultra's full
        # configuration to the other widths caused spills in screening.
        PL: gl.constexpr = (
            BlockedLayout([1, 2], [2, 16], [4, 1], [1, 0])
            if PROJECTION_LAYOUT == "rows2v2"
            else (
                BlockedLayout([1, 2], [4, 8], [4, 1], [1, 0])
                if PROJECTION_LAYOUT == "rows4v2"
                else BlockedLayout([1, 4], [4, 8], [4, 1], [1, 0])
            )
        )
        pcc = gl.arange(0, triton.next_power_of_2(NT), layout=gl.constexpr(SliceLayout(0, PL)))
        # At KCOLS=128 the real/padded participants are Nano 24/32,
        # Super 32/32, Ultra 80/128. Mask padding to zero in every load;
        # reducing just 64 lanes would silently omit Ultra's last columns.
        REDUCE_ROWS: gl.constexpr = 32
        for reader in gl.static_range(2):
            prr = gl.arange(0, REDUCE_ROWS, layout=gl.constexpr(SliceLayout(1, PL)))
            if PP_TILE:
                partial = gl.load(
                    pp_active
                    + _partial_offset(
                        reader, group, prr[:, None], pcc[None, :], GROUPS, NT, PP_TILE
                    ),
                    pcc[None, :] < NT,
                    0,
                    volatile=True,
                )
            else:
                partial = gl.load(
                    pp_active + ((reader * GROUPS + group) * 32 + prr[:, None]) * NT + pcc[None, :],
                    pcc[None, :] < NT,
                    0,
                    volatile=True,
                )
            local_projection.index(reader).store(gl.sum(partial, 1))
        gl.barrier()
        if iteration + 1 < group_iters:
            # K3 issues the next inputs in pieces as buffers become free.
            # Account for the entire next tile here, then wait next loop.
            # Do not arm/issue this transfer on the group's final tile.
            mbarrier.expect(copy_barrier, 32 * 128 * 16)
        for part in gl.static_range(32 // 16):
            _k3_part(
                U,
                local_projection,
                OUT,
                DR,
                DU,
                DV,
                CP,
                residuals,
                ds,
                rs0,
                rs1,
                rp,
                cp,
                part,
                M,
                CARRY_DESC,
                carry_barrier,
                NEXT_R_DESC,
                NEXT_DY0_DESC,
                NEXT_DY1_DESC,
                copy_barrier,
                rp + GROUPS,
                iteration + 1 < group_iters,
                carry_zero,
                gs,
                cs,
                reciprocals,
                zero_barrier,
                carry_barrier2,
                H=H,
                HAS_RETENTION=HAS_RETENTION,
            )


@triton.jit
def _gamma_controller_finish(
    GP0,
    GP1,
    DG0,
    DG1,
    PART,
    GW,
    GT,
    G0,
    G1,
    COEFF,
    M: tl.constexpr,
    GB: tl.constexpr,
    TILE_ROWS: tl.constexpr,
    P: tl.constexpr,
    BLOCK: tl.constexpr,
    LW: tl.constexpr,
    LT: tl.constexpr,
    L0: tl.constexpr,
    L1: tl.constexpr,
    GAMMA_COLS: tl.constexpr,
    H: tl.constexpr,
    HAS_RETENTION: tl.constexpr = True,
):
    # Disjoint CTA ranges finish gamma and the twelve controller scalars in
    # one launch. Controller chunks were reduced separately first, bounding
    # this final tree; fusion saves a launch without adding cross-CTA waits.
    bid = tl.program_id(0)
    if bid < H // GAMMA_COLS:
        cols = bid * GAMMA_COLS + tl.arange(0, GAMMA_COLS)
        rows = tl.arange(0, GB)
        for reader in tl.static_range(2):
            x = tl.load(
                (GP0 if reader == 0 else GP1) + rows[:, None] * H + cols[None, :],
                rows[:, None] < tl.cdiv(M, TILE_ROWS),
                0,
            )
            tl.store((DG0 if reader == 0 else DG1) + cols, tl.sum(x, 0))
    else:
        s = bid - H // GAMMA_COLS
        i = tl.arange(0, BLOCK)
        factor = tl.load(COEFF + 12 + s)
        tl.store(GW + s, tl.sum(tl.load(PART + s * P + i, i < P, other=0)) * factor)
        tl.store(GW + i + 3, 0, (s == 0) & (i + 3 < LW))
        if HAS_RETENTION:
            factor = tl.load(COEFF + 15 + s)
            tl.store(GT + s, tl.sum(tl.load(PART + (3 + s) * P + i, i < P, other=0)) * factor)
            tl.store(GT + i + 3, 0, (s == 0) & (i + 3 < LT))
        factor = tl.load(COEFF + 18 + s)
        tl.store(
            G0 + s,
            tl.sum(tl.load(PART + ((1 + HAS_RETENTION) * 3 + s) * P + i, i < P, other=0)) * factor,
        )
        tl.store(G0 + i + 3, 0, (s == 0) & (i + 3 < L0))
        factor = tl.load(COEFF + 21 + s)
        tl.store(
            G1 + s,
            tl.sum(tl.load(PART + ((2 + HAS_RETENTION) * 3 + s) * P + i, i < P, other=0)) * factor,
        )
        tl.store(G1 + i + 3, 0, (s == 0) & (i + 3 < L1))


@dataclass(frozen=True)
class Config:
    """Fixed launch selection; offline probes may supply a resolved schedule.

    Existing widths retain their frozen defaults. Other widths bootstrap with
    one group for compilation/validation; this is not a qualified auto entry.
    """

    hidden_size: int = 3072
    launch: Schedule | None = None

    def __post_init__(self):
        if self.hidden_size not in WIDTHS:
            raise ValueError("cooperative backward supports H=2048..10240, step 1024")
        if self.launch is not None:
            if not isinstance(self.launch, Schedule):
                raise ValueError("launch must be an offline Schedule")
            self.launch.validate(self.hidden_size)

    @property
    def schedule(self):
        selected = self.launch or INCUMBENTS.get(self.hidden_size)
        if selected is None:
            selected = Schedule(self.hidden_size // 128)
        return selected.as_tuple()


NANO = Config(3072)
SUPER = Config(4096)
ULTRA = Config(10240)


@dataclass
class Workspace:
    """Caller-owned outputs and scratch; reuse only after prior use completes."""

    signature: tuple
    previous: torch.Tensor
    grads: tuple
    columns: tuple
    pp: torch.Tensor
    gp: tuple
    controllers: torch.Tensor
    chunks: torch.Tensor
    epochs: torch.Tensor
    kernel: object = None
    resources: object = None

    def tensors(self):
        yield from (
            t
            for t in (
                self.previous,
                *self.grads,
                self.pp,
                *self.gp,
                self.controllers,
                self.chunks,
                self.epochs,
            )
            if t is not None
        )
        for q in self.columns:
            if type(q) is torch.Tensor:
                yield q
            else:
                yield q._columnwise_data
                yield q._columnwise_scale_inv


def _signature(saved, config):
    if getattr(saved, "num_streams", 3) != 3:
        raise ValueError("use reconstruction_cooperative_streams for num_streams other than 3")
    if getattr(saved, "operand_format", "mxfp8") not in ("mxfp8", "bf16"):
        raise ValueError("invalid saved operand format")
    m, h = saved.updates[0].shape
    return (
        m,
        h,
        saved.updates[0].device,
        tuple(saved.controller_sizes),
        config,
        (config.schedule, getattr(saved, "operand_format", "mxfp8"), saved.has_retention),
    )


def allocate(saved, config=None):
    """Allocate reusable outputs/scratch; warm up backward before graph capture."""
    config = Config(saved.updates[0].shape[1]) if config is None else config
    m, h, device, sizes, _, _ = _signature(saved, config)
    if h != config.hidden_size or m <= 0 or m % 32 or not saved.summed_update:
        raise ValueError("requires matching H, positive M divisible by 32, and summed update")
    if len(sizes) != 4 or any(
        size < 3 for i, size in enumerate(sizes) if i != 1 or saved.has_retention
    ):
        raise ValueError("four controllers with at least three entries are required")
    if device.type != "cuda" or torch.cuda.get_device_capability(device) != (10, 3):
        raise ValueError("cooperative backward is validated on GB300 (SM103) only")
    nt, groups = h // 128, config.schedule[0] // (h // 128)
    if nt * triton.cdiv(triton.cdiv(m, 128) * 4, groups) >= 2**31:
        raise ValueError("arrival counter exceeds signed 32-bit capacity")

    def empty(shape, dtype=torch.float32):
        return torch.empty(shape, device=device, dtype=dtype)

    grads = (
        empty((m, 3 * h)),
        empty((m, h), torch.bfloat16),
        empty((m, h), torch.bfloat16),
        *(empty((s,)) if s else None for s in sizes),
        empty((h,)),
        empty((h,)),
    )
    p = triton.cdiv(m, 16) * nt * 8
    b = Workspace(
        _signature(saved, config),
        empty((m, 3 * h)),
        grads,
        tuple(
            (
                empty((m, h), torch.bfloat16)
                if getattr(saved, "operand_format", "mxfp8") == "bf16"
                else column_quantizer().make_empty((m, h), dtype=torch.bfloat16, device=device)
            )
            for _ in range(2)
        ),
        # Two slots protect PP from fast next-iteration producers. Projection
        # results are CTA-private shared memory, so no global PROJ is allocated.
        empty((4, groups, 32, nt)),
        (empty((triton.cdiv(m, 32), h)), empty((triton.cdiv(m, 32), h))),
        empty(((3 + saved.has_retention) * 3, p)),
        empty(((3 + saved.has_retention) * 3, triton.cdiv(p, 1024))),
        empty((groups * 32,), torch.int32),
    )
    b._descriptors = tuple((tuple(t.shape), t.dtype, t.device, t.stride()) for t in b.tensors())
    return b


def occupancy(kernel, grid, device):
    """Reject a grid whose spinning consumers could exclude resident producers."""
    kernel._init_handles()
    lib = ctypes.CDLL("libcuda.so.1")
    blocks, static_shared = ctypes.c_int(), ctypes.c_int()
    rc = lib.cuOccupancyMaxActiveBlocksPerMultiprocessor(
        ctypes.byref(blocks),
        ctypes.c_void_p(kernel.function),
        ctypes.c_int(kernel.metadata.num_warps * 32),
        ctypes.c_size_t(kernel.metadata.shared),
    )
    if rc:
        raise RuntimeError(f"CUDA occupancy query failed: {rc}")
    rc = lib.cuFuncGetAttribute(
        ctypes.byref(static_shared), ctypes.c_int(1), ctypes.c_void_p(kernel.function)
    )
    if rc:
        raise RuntimeError(f"CUDA static shared query failed: {rc}")
    sm = torch.cuda.get_device_properties(device).multi_processor_count
    info = dict(
        blocks_per_sm=blocks.value,
        sms=sm,
        limit=sm * blocks.value,
        shared=kernel.metadata.shared,
        static_shared=static_shared.value,
        registers=kernel.n_regs,
        spills=kernel.n_spills,
    )
    if grid > info["limit"]:
        raise ValueError(f"grid {grid} exceeds cooperative residency: {info}")
    return info


def compile_workspace(saved, current, dys, carry, *, config=None):
    """Compile at the actual shape without launching an unchecked grid."""
    return _backward(saved, current, dys, carry, config=config, compile_only=True)


def backward(saved, current, dys, carry, *, workspace=None, config=None):
    """Complete two-reader B backward, with exact operands and matched inverse.

    saved.operand_format selects columnwise MXFP8 or contiguous BF16 [M,H]
    consumer operands. Both variants preserve the same FP32 gradient arithmetic.

    Requires FP32 residual/X storage, BF16 dy, a saved summed update and carry.
    Returns (previous, grads, workspace), with independent previous/dR and du/dv.
    The workspace is reusable across input pools and CUDA graph replay, but must
    not be shared by overlapping invocations. Reset and both final reductions
    are included in every call. Unsupported routes should use the normal
    reconstruction_backward.backward entry point.
    """
    return _backward(saved, current, dys, carry, workspace=workspace, config=config)


def _backward(saved, current, dys, carry, *, workspace=None, config=None, compile_only=False):
    config = Config(saved.updates[0].shape[1]) if config is None else config
    m, h, device, sizes, _, _ = _signature(saved, config)
    if (
        h != config.hidden_size
        or m <= 0
        or m % 32
        or not saved.summed_update
        or saved.residual_storage != "fp32"
        or saved.x_storage != "fp32"
        or saved.reader_count != 2
        or len(dys) != 2
        or len(saved.rstd) != 2
        or len(saved.gammas) != 2
        or len(sizes) != 4
        or any(size < 3 for i, size in enumerate(sizes) if i != 1 or saved.has_retention)
    ):
        raise ValueError("unsupported cooperative backward route")
    _tensor(current, (m, 3 * h), torch.float32, device, "current")
    _tensor(carry, (m, 3 * h), torch.float32, device, "carry")
    for dy in dys:
        _tensor(dy, (m, h), torch.bfloat16, device, "dy")
    _tensor(saved.updates[0], (m, h), torch.bfloat16, device, "summed update")
    _tensor(saved.coefficients, (24,), torch.float32, device, "coefficients")
    for rs, gamma in zip(saved.rstd, saved.gammas):
        _tensor(rs, (m,), torch.float32, device, "rstd")
        _tensor(gamma, (h,), torch.bfloat16, device, "gamma")
    b = allocate(saved, config) if workspace is None else workspace
    if not isinstance(b, Workspace) or b.signature != _signature(saved, config):
        raise ValueError("cooperative workspace configuration mismatch")
    owned = list(b.tensors())
    ptrs = [t.untyped_storage().data_ptr() for t in owned]
    inputs = (current, carry, *dys, *saved.updates, saved.coefficients, *saved.rstd, *saved.gammas)
    # Check backing storage, not just view pointers: an alias can overwrite
    # inputs still used by another CTA or outputs consumed after the call.
    if len(set(ptrs)) != len(ptrs) or set(ptrs) & {t.untyped_storage().data_ptr() for t in inputs}:
        raise ValueError("cooperative workspace aliases live storage")
    desc = tuple((tuple(t.shape), t.dtype, t.device, t.stride()) for t in owned)
    if b._descriptors != desc:
        raise ValueError("cooperative workspace tensor mismatch")

    # TMA descriptors include addresses. Cache construction for repeated calls,
    # rebuilding when a new input pool changes any descriptor-backed pointer.
    key = tuple(t.data_ptr() for t in (current, *dys, carry))
    if getattr(b, "_tma_key", None) != key:
        r_layout = NVMMASharedLayout(swizzle_byte_width=128, element_bitwidth=32, rank=2)
        dy_layout = NVMMASharedLayout(swizzle_byte_width=64, element_bitwidth=16, rank=2)
        b._descriptors_tma = (
            TensorDescriptor.from_tensor(current, [32, 128], r_layout),
            *(TensorDescriptor.from_tensor(dy, [32, 128], dy_layout) for dy in dys),
            TensorDescriptor.from_tensor(carry, [16, 128], r_layout),
            TensorDescriptor.from_tensor(current, [16, 128], r_layout),
            *(TensorDescriptor.from_tensor(dy, [16, 128], dy_layout) for dy in dys),
        )
        b._tma_key = key
    r_desc, dy0_desc, dy1_desc, carry_desc, next_r, next_dy0, next_dy1 = b._descriptors_tma
    grid, projection_layout, pp_tile, carry_rows, before_wait = config.schedule
    args = (
        r_desc,
        dy0_desc,
        dy1_desc,
        saved.updates[0],
        saved.coefficients,
        *saved.rstd,
        *saved.gammas,
        carry,
        *(
            (q if getattr(saved, "operand_format", "mxfp8") == "bf16" else q._columnwise_data)
            for q in b.columns
        ),
        *(
            (
                None
                if getattr(saved, "operand_format", "mxfp8") == "bf16"
                else q._columnwise_scale_inv
            )
            for q in b.columns
        ),
        b.pp,
        *b.gp,
        b.previous,
        *b.grads[:3],
        b.controllers,
        b.epochs,
    )
    kw = dict(
        HAS_RETENTION=saved.has_retention,
        M=m,
        H=h,
        BF16=getattr(saved, "operand_format", "mxfp8") == "bf16",
        GRID=grid,
        CARRY_DESC=carry_desc,
        NEXT_R_DESC=next_r,
        NEXT_DY0_DESC=next_dy0,
        NEXT_DY1_DESC=next_dy1,
        PROJECTION_LAYOUT=projection_layout,
        PP_TILE=pp_tile,
        CARRY_PREFETCH_ROWS=carry_rows,
        CARRY_PREFETCH_BEFORE_WAIT=before_wait,
        num_warps=4,
        maxnreg=168,
        enable_fp_fusion=True,
        launch_cooperative_grid=True,
    )
    if b.kernel is None:
        kernel = _cooperative.warmup(*args, **kw, grid=(grid,))
        b.resources = occupancy(kernel, 0 if compile_only else grid, device)
        b.kernel = kernel
    if compile_only:
        return b
    if grid > b.resources["limit"]:
        raise ValueError(f"grid {grid} exceeds cooperative residency: {b.resources}")
    # Reset on every invocation, including graph replay: otherwise an old epoch
    # could release a consumer before the current partials have been published.
    b.epochs.zero_()
    _cooperative[(grid,)](*args, **kw)
    p, chunks = b.controllers.shape[1], b.chunks.shape[1]
    chunks_kernel = _controller_chunks[((3 + saved.has_retention) * 3, chunks)](
        b.controllers,
        b.chunks,
        p,
        chunks,
        True,
        saved.has_retention,
        True,
        True,
        HAS_RETENTION=saved.has_retention,
        enable_fp_fusion=False,
    )
    finish_kernel = _gamma_controller_finish[(h // 8 + 3,)](
        *b.gp,
        *b.grads[-2:],
        b.chunks,
        *b.grads[3:7],
        saved.coefficients,
        m,
        triton.next_power_of_2(triton.cdiv(m, 32)),
        32,
        chunks,
        triton.next_power_of_2(max(chunks, *sizes)),
        *sizes,
        HAS_RETENTION=saved.has_retention,
        H=h,
        GAMMA_COLS=8,
        num_warps=4,
        enable_fp_fusion=False,
    )
    b.final_kernels = {"controller_chunks": chunks_kernel, "gamma_controller_finish": finish_kernel}
    return b.previous, b.grads, b
