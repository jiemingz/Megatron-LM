# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Call-local residual lifetimes for a linear chain of invertible writes.

The successor's backward restores its input before returning any gradients to
readers or branch consumers. No forward operation is replayed. Only FP32,
full-storage, contiguous carried residuals participate; ordinary branch tensors
and their autograd owners are untouched.
"""

from contextlib import contextmanager
from contextvars import ContextVar
from weakref import WeakValueDictionary

import torch

_ACTIVE = ContextVar("wide_residual_inverse", default=None)


def current_manager():
    """Return the current forward's owner, excluding explicit no-grad regions."""
    return _ACTIVE.get() if torch.is_grad_enabled() else None


def watch_residual(tensor):
    """Record a native reader's storage dependency during its autograd forward."""
    owner = _ACTIVE.get()
    return None if owner is None else owner.slot(tensor)


def register_write(previous, current, update, write, retain, streams, max_forget):
    """Register a native ordinary write, keeping detached inverse metadata only."""
    manager = _ACTIVE.get()
    if manager is None:
        return None
    args = (
        update.detach(),
        write.detach(),
        None if retain is None else retain.detach(),
        streams,
        max_forget,
    )

    def inverse(value):
        from .inverse_write import inverse_write

        return inverse_write(value, *args)

    return manager.register(previous, current, inverse)


class ResidualSlot:
    """One StorageImpl, optionally recoverable from the succeeding write.

    Providers point only forward to the next slot, never back to their input or
    its autograd context. This keeps the ownership graph acyclic, including when
    a loss prunes later gradient nodes but still needs their inverses.
    """

    def __init__(self, tensor, anchor=True):
        self.tensor = tensor.detach()
        self.anchor = anchor
        self.provider = None
        self.restoring = False

    def resident(self):
        return self.tensor.untyped_storage().nbytes() != 0

    def restore(self, value):
        if not self.resident():
            from megatron.core.tensor_parallel.random import _get_share_storage

            _get_share_storage()(self.tensor, value)
            self.provider = None

    def ensure(self):
        if self.resident():
            return
        if self.provider is None or self.restoring:
            raise RuntimeError("residual inverse has no ready successor or contains a cycle")
        self.restoring = True
        try:
            current, inverse, counters = self.provider
            current.ensure()
            counters["lazy_inverses"] += 1
            self.restore(inverse(current.tensor))
        finally:
            self.restoring = False


class ResidualLink:
    """Detached storage handles; no autograd tensors live on context attributes."""

    def __init__(self, previous, current):
        self.previous_slot = previous
        self.current_slot = current

    @property
    def current(self):
        return self.current_slot.tensor

    def require_current(self):
        self.current_slot.ensure()

    def needs_previous(self):
        return not self.previous_slot.resident()

    def restore(self, previous):
        self.previous_slot.restore(previous)

    def release(self):
        # The direct kernel handoff is complete; readers now see resident data.
        self.previous_slot.provider = None
        self.previous_slot = None
        self.current_slot = None


class ResidualInverseManager:
    """Keep periodic physical-layer anchors and release intermediate write outputs."""

    def __init__(self, interval):
        if type(interval) is not int or interval < 2:
            raise ValueError("residual inverse interval must be an integer greater than one")
        self.interval = interval
        self.pending = []
        self.last = None
        self.layers = 0
        self.slots = WeakValueDictionary()
        self.counters = {"lazy_inverses": 0}

    def slot(self, tensor, *, output=False):
        key = tensor.untyped_storage()._cdata
        slot = self.slots.get(key)
        if slot is None:
            slot = ResidualSlot(tensor, anchor=not output)
            self.slots[key] = slot
        return slot

    @contextmanager
    def activate(self):
        token = _ACTIVE.set(self)
        try:
            yield self
            self.seal()
        finally:
            _ACTIVE.reset(token)
            self.pending.clear()
            self.last = None

    def begin_layer(self, physical_layers):
        # An execution unit may contain a complete compute/MoE pair.
        if self.layers and self.layers + physical_layers > self.interval:
            self.seal()

    def finish_layer(self, physical_layers):
        self.layers += physical_layers
        if self.layers >= self.interval:
            self.seal()

    def register(self, previous, current, inverse=None):
        for tensor in (previous, current):
            if (
                tensor.dtype != torch.float32
                or not tensor.is_contiguous()
                or tensor.storage_offset() != 0
                or tensor.untyped_storage().nbytes() != tensor.numel() * tensor.element_size()
            ):
                raise ValueError("residual inverse requires contiguous full-storage FP32 tensors")
        if self.last is not None and (
            self.last.untyped_storage()._cdata != previous.untyped_storage()._cdata
        ):
            raise ValueError("residual inverse requires a single connected chain of writes")
        if previous.untyped_storage()._cdata == current.untyped_storage()._cdata:
            raise ValueError("residual inverse writes must own distinct output storage")
        old, new = self.slot(previous), self.slot(current, output=True)
        if not old.anchor:
            if inverse is None:
                raise ValueError("residual inverse needs the successor's inverse metadata")
            old.provider = (new, inverse, self.counters)
        link = ResidualLink(old, new)
        self.pending.append(link)
        self.last = link.current
        return link

    def seal(self):
        if self.pending:
            anchor = self.last.untyped_storage()._cdata
            for link in self.pending:
                if link.current.untyped_storage()._cdata != anchor:
                    link.current.untyped_storage().resize_(0)
                else:
                    link.current_slot.anchor = True
                    link.current_slot.provider = None
            self.pending.clear()
        self.layers = 0


@contextmanager
def residual_inverse_scope(config, training):
    """Create an independent inverse chain for each HybridStack forward."""
    interval = config.residual_stream_recompute_num_layers
    if (
        config.wide_residual_shortcut_moe_kernel_fusion
        and training
        and torch.is_grad_enabled()
        and interval is not None
        and interval > 1
    ):
        with ResidualInverseManager(interval).activate() as manager:
            yield manager
    else:
        # An ordinary or MTP stack nested in another forward must not join its owner.
        token = _ACTIVE.set(None)
        try:
            yield None
        finally:
            _ACTIVE.reset(token)
