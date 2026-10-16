# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Model-facing A/B boundaries; imports of the GPU implementations are lazy."""

import math

import torch
from torch.autograd.function import once_differentiable

from .residual_inverse import current_manager


def validate_input(residual, hidden_size, num_streams, *, fp32_residual_connection):
    """Validate the local flattened token layout before launching fused work."""
    if type(num_streams) is not int or not 1 <= num_streams <= 6:
        raise ValueError("shortcut fusion requires an integer stream count in [1, 6]")
    if type(hidden_size) is not int or not 32 <= hidden_size <= 10240 or hidden_size % 32:
        raise ValueError("shortcut fusion requires H in [32, 10240], divisible by 32")
    if residual.ndim < 2 or residual.shape[-1] != num_streams * hidden_size:
        raise ValueError("shortcut fusion expects residual shape [..., S * H]")
    tokens = math.prod(residual.shape[:-1])
    maximum = 65536 if hidden_size <= 4096 else 16384
    if tokens <= 0 or tokens % 1024 or tokens > maximum:
        raise ValueError(
            f"shortcut fusion requires local flattened T to be a positive multiple of 1024 "
            f"and at most {maximum} for H={hidden_size}; got T={tokens}"
        )
    if not fp32_residual_connection:
        raise ValueError("BF16 shortcut fusion requires fp32_residual_connection=True")
    expected = torch.float32
    if residual.dtype != expected or not residual.is_contiguous():
        raise ValueError(f"shortcut fusion requires contiguous {expected} residuals")
    if not residual.is_cuda:
        raise ValueError("shortcut fusion requires CUDA tensors")
    return tokens


def _backward(saved, current, dys, carry):
    """Select qualified cooperative execution, preserving the independent S=3 kernel."""
    from . import reconstruction_backward_streams as standard
    from .reconstruction_dispatch import (
        CooperativeEligibilityError,
        qualified_device,
        resolve_backward,
    )

    eligible = (
        saved.reader_count == 2
        and saved.residual_storage == "fp32"
        and saved.x_storage == "fp32"
        and all(dy is not None for dy in dys)
        and carry is not None
    )
    if eligible and qualified_device(torch.cuda.get_device_properties(current.device)):
        schedule = resolve_backward(
            saved.updates[0].shape,
            torch.cuda.get_device_capability(current.device),
            operand_format=saved.operand_format,
            num_streams=saved.num_streams,
        )
        if schedule is not None:
            if saved.num_streams == 3:
                from . import reconstruction_cooperative_native as cooperative
            else:
                from . import reconstruction_cooperative_streams as cooperative
            try:
                return cooperative.backward(
                    saved,
                    current,
                    dys,
                    carry,
                    config=cooperative.Config(saved.updates[0].shape[1], launch=schedule),
                )
            except CooperativeEligibilityError:
                pass
    return standard.backward(
        saved,
        current,
        dys,
        carry,
        prepare_wgrad=(True,) * saved.reader_count,
        inverse_mode="matched",
        gradient_storage="fp32_residual",
    )


class _Boundary(torch.autograd.Function):
    """Own one boundary's tensors; no workspace or handoff is stored on a module."""

    @staticmethod
    def forward(
        ctx,
        residual,
        u,
        v,
        write,
        retain,
        read0,
        read1,
        gamma0,
        gamma1,
        streams,
        fp32,
        inverse_manager,
    ):
        from .reconstruction import forward

        count = 1 if v is None else 2
        hidden = u.shape[-1]
        validate_input(residual, hidden, streams, fp32_residual_connection=fp32)
        leading = residual.shape[:-1]
        updates = (u,) if count == 1 else (u, v)
        if any(t.shape != (*leading, hidden) for t in updates):
            raise ValueError("shortcut updates must have the residual's leading dimensions")
        current, saved, outputs, _ = forward(
            residual.view(-1, streams * hidden),
            tuple(t.contiguous().view(-1, hidden) for t in updates),
            write,
            retain,
            (read0,) if count == 1 else (read0, read1),
            (gamma0,) if count == 1 else (gamma0, gamma1),
            residual_storage="fp32",
            x_storage="fp32",
            operand_format="bf16",
            num_streams=streams,
            sum_updates=True,
        )
        tensors = (current, *saved.updates, saved.coefficients, *saved.gammas, *saved.rstd)
        ctx.inverse_link = None
        if inverse_manager is not None:

            def inverse_current(value):
                from .reconstruction import reconstruct

                previous, _ = reconstruct(
                    value.view(-1, streams * hidden), saved, inverse_mode="matched"
                )
                return previous

            ctx.inverse_link = inverse_manager.register(residual, current, inverse_current)
        ctx.save_for_backward(*tensors)
        ctx.boundary_spec = (
            saved.kind,
            saved.residual_storage,
            saved.x_storage,
            saved.controller_sizes,
            saved.has_retention,
            streams,
            len(saved.updates),
            count,
            leading,
        )
        ctx.set_materialize_grads(False)
        return (
            current.view(*leading, streams * hidden),
            *(q.view(*leading, hidden) for q in outputs),
        )

    @staticmethod
    @once_differentiable
    def backward(ctx, carry, *dys):
        from .reconstruction import SavedWrite

        kind, storage, x_storage, sizes, has_retention, streams, updates_count, count, leading = (
            ctx.boundary_spec
        )
        tensors = ctx.saved_tensors
        current = tensors[0]
        if ctx.inverse_link is not None:
            ctx.inverse_link.require_current()
        end = 1 + updates_count
        updates = tensors[1:end]
        coefficients = tensors[end]
        gammas = tensors[end + 1 : end + 1 + count]
        rstd = tensors[end + 1 + count : end + 1 + 2 * count]
        saved = SavedWrite(
            kind,
            storage,
            updates,
            coefficients,
            gammas,
            rstd,
            sizes,
            x_storage=x_storage,
            operand_format="bf16",
            num_streams=streams,
            has_retention=has_retention,
        )
        hidden = updates[0].shape[1]
        dy = tuple(None if t is None else t.contiguous().view(-1, hidden) for t in dys)
        dc = None if carry is None else carry.contiguous().view_as(current)
        previous, grads, _ = _backward(saved, current, dy, dc)
        if ctx.inverse_link is not None:
            ctx.inverse_link.restore(previous)
            ctx.inverse_link.release()
            ctx.inverse_link = None
        grads = list(grads)
        if grads[0] is not None:
            grads[0] = grads[0].view(*leading, streams * hidden)
        for i in range(count):
            if grads[1 + i] is not None:
                grads[1 + i] = grads[1 + i].view(*leading, hidden)
        if count == 1:
            result = (grads[0], grads[1], None, grads[2], grads[3], grads[4], None, grads[5], None)
        else:
            result = tuple(grads)
        return tuple(
            g if needed else None for g, needed in zip(result, ctx.needs_input_grad[:9])
        ) + (None, None, None)


def write_read_norm(
    residual, update, write, retain, read, gamma, *, num_streams, fp32_residual_connection=True
):
    """Fuse a compute-layer write and the shared expert's read/RMSNorm (A)."""
    return _Boundary.apply(
        residual,
        update,
        None,
        write,
        retain,
        read,
        None,
        gamma,
        None,
        num_streams,
        fp32_residual_connection,
        current_manager(),
    )


def boundary_norm(
    residual,
    routed,
    shared,
    write,
    retain,
    reads,
    gammas,
    *,
    num_streams,
    fp32_residual_connection=True,
):
    """Fuse the MoE write and the next pair's two independent read/RMSNorms (B)."""
    if len(reads) != 2 or len(gammas) != 2:
        raise ValueError("boundary_norm requires two read controllers and two RMSNorm weights")
    return _Boundary.apply(
        residual,
        routed,
        shared,
        write,
        retain,
        *reads,
        *gammas,
        num_streams,
        fp32_residual_connection,
        current_manager(),
    )
