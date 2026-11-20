# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""Activation strategies: what happens to a region's activations, one class per technique.

A module returns its strategies from ``MegatronModule.activation_strategies``, built from its config,
each naming the region it acts on; they are installed once the module is constructed.

The runtime knows nothing about recompute or offloading. For every call of a region it:

1. calls :meth:`Strategy.enter` of each active strategy (in order; it may replace the positional
   arguments);
2. runs the region through the first active *wrapping* strategy's :meth:`Strategy.run` (at most
   one strategy owns the call), or calls it directly;
3. calls :meth:`Strategy.exit` of each active strategy (it may replace the output);
4. fires the actions that strategies scheduled, with :meth:`RegionCall.on`, on this region's
   ``"enter"`` / ``"exit"`` event.

If the call raises, :meth:`Strategy.abort` of every strategy that entered runs instead of 3-4.

A strategy is shared by every call of its region; per-call data goes in ``call.data``. At install,
a region's strategies are ordered by :attr:`Strategy.order` (wrapping strategies first, in
precedence order: replay, then the exclusive recompute / checkpoint / full recompute; then the
non-wrapping ones in the given order, the offload exclusion last).

Actions are how one region acts at another region's boundary (release a recomputed output when
its consumer starts, commit an offload group when a later region returns). They fire in stages:
a lower :attr:`Action.stage` fires first, and :attr:`Action.lifo` stages fire newest first.
"""

from __future__ import annotations

import sys
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any, Callable, ClassVar, Hashable, Iterable, Literal, Optional

import torch

from megatron.core.activation_policy import runtime
from megatron.core.activation_policy.offload.te_cpu import TECPUOffload
from megatron.core.activation_policy.recompute import checkpointing as _checkpointing
from megatron.core.utils import make_viewless_tensor

# ---------------------------------------------------------------------------------------------
# Mechanisms (looked up on this module at call time, so tests can patch them here)
# ---------------------------------------------------------------------------------------------


def _offload_interface():
    from megatron.core.activation_policy.offload.fine_grained_activation_offload import (
        FineGrainedActivationOffloadingInterface,
    )

    return FineGrainedActivationOffloadingInterface


def precision_aware_checkpoint(function, config, tp_group, /, *args, **kwargs):
    """The single call site of ``recompute.precision_aware_checkpoint`` for all strategies."""
    return _checkpointing.precision_aware_checkpoint(function, config, tp_group, *args, **kwargs)


_BORROWED_ATTR = "_activation_policy_borrowed"


def mark_borrowed(tensor: Optional[torch.Tensor]) -> None:
    """Declare that the region consuming ``tensor`` does not own its storage.

    An offload commit never force-releases such a tensor (for example a caller-owned
    communication buffer).
    """
    if tensor is not None:
        setattr(tensor, _BORROWED_ATTR, True)


def _releasable(tensor: Optional[torch.Tensor]) -> bool:
    """Whether an offload commit may force-release ``tensor``'s storage.

    Quantized tensors and GroupedTensors are wrappers: releasing their (wrapper) storage would
    not free their data.
    """
    if tensor is None or getattr(tensor, _BORROWED_ATTR, False):
        return False
    from megatron.core.fp8_utils import is_float8tensor, is_grouped_tensor

    return not (is_float8tensor(tensor) or is_grouped_tensor(tensor))


# ---------------------------------------------------------------------------------------------
# Interface
# ---------------------------------------------------------------------------------------------


# ---------------------------------------------------------------------------------------------
# Regions, events and boundary tensors
# ---------------------------------------------------------------------------------------------
#
# A *region* (or point) is a call inside an *owner* module: a child module, given as the module
# itself (``self.input_layernorm``, ``self.linear_fc2``), or a method of the owner, given as the
# bound method (``self._mlp_bda``, ``self._gated_norm_and_a2a``).

When = Literal["enter", "exit"]


@dataclass(frozen=True)
class Event:
    """Entering or leaving a region of the same owner.

    The tensor an event refers to is the first tensor of the region's first argument
    (``"enter"``) or of its output (``"exit"``).
    """

    region: Any  # a child module, or a method of the owner
    when: When


@dataclass(frozen=True)
class TensorRef:
    """A tensor seen at a region boundary, resolved when the action that uses it fires.

    Build with :meth:`input` (positional argument ``index``, after any offload group start was
    applied to it) or :meth:`output` (element ``index`` of the output tuple, or the output).
    """

    region: Any  # a child module, or a method of the owner
    side: Literal["input", "output"]
    index: int

    @classmethod
    def input(cls, region: Any, index: int = 0) -> TensorRef:
        """Positional argument ``index`` of ``region``."""
        return cls(region, "input", index)

    @classmethod
    def output(cls, region: Any, index: int = 0) -> TensorRef:
        """Output element ``index`` of ``region``."""
        return cls(region, "output", index)


def _same_region(a: Any, b: Any) -> bool:
    return a is b or a == b


def region_name(region: Any) -> str:
    """A readable name for a region (a module, or a method)."""
    if isinstance(region, torch.nn.Module):
        return type(region).__name__
    return getattr(region, "__name__", None) or getattr(region, "name", type(region).__name__)


class RegionCall:
    """One call of a region, as the strategies see it.

    Args:
        owner: The module that owns the region.
        name: The region name.
        state: The owner's per-forward state (event bus and captured tensors).
    """

    __slots__ = ("owner", "name", "state", "data")

    def __init__(self, owner: torch.nn.Module, name: Any, state):
        self.owner, self.name, self.state = owner, name, state
        # Per-call data of the strategies, keyed by strategy.
        self.data: dict = {}

    def on(self, event: Event, source: Hashable, action: Action) -> None:
        """Schedule ``action`` for ``event`` of the same owner.

        Scheduling again with the same ``source`` replaces the earlier action, so an action left
        over from a forward whose event never fired (an aborted micro-batch) never fires late.
        """
        self.state.defer(event, source, action)

    def capture(self, refs: Iterable[TensorRef]) -> None:
        """Record the tensors ``refs`` point at when they pass their region's boundary."""
        self.state.armed.update(refs)

    def uncapture(self, refs: Iterable[TensorRef]) -> None:
        """Stop recording ``refs`` and drop what was recorded for them."""
        for ref in refs:
            self.state.take_record(ref)

    def take(self, ref: TensorRef) -> Any:
        """The tensor recorded for ``ref`` (``None`` if it was not seen); stops recording it."""
        return self.state.take_record(ref)


class Action:
    """Something a strategy does at another region's boundary (see :meth:`RegionCall.on`)."""

    __slots__ = ()
    stage: ClassVar[int] = 0
    lifo: ClassVar[bool] = False

    def on_enter(self, hook_tensors: list) -> None:
        """Fire at a region's enter event; ``hook_tensors`` are taken from its first argument."""
        raise NotImplementedError(f"{type(self).__name__} cannot fire on an enter event.")

    def on_exit(self, value: Any, state) -> Any:
        """Fire at a region's exit event with its output; returns the (possibly new) output."""
        raise NotImplementedError(f"{type(self).__name__} cannot fire on an exit event.")


class Strategy:
    """What happens to one region's activations (see the module docstring).

    Subclasses set ``wraps = True`` if they run the region themselves (:meth:`run`); at most one
    such strategy is active per call, the first in :attr:`order`. Wrapping strategies with the
    same ``exclusive`` group cannot share a region. A strategy with ``replays = True`` replays
    the region with its block when active; the region's other strategies see that (``replayed``).
    A strategy with an ``owner`` attribute acts on that module's region (see
    ``runtime.install_activation_strategies``).

    Args:
        region: The region it acts on: a child module or a method of the owner.
    """

    __slots__ = ("region",)
    wraps: ClassVar[bool] = False
    replays: ClassVar[bool] = False
    order: ClassVar[int] = 10
    exclusive: ClassVar[Optional[str]] = None

    def __init__(self, region: Any):
        self.region = region

    def __repr__(self) -> str:
        return f"{type(self).__name__}({region_name(self.region)})"

    def expand(self) -> Iterable[Strategy]:
        """The strategies installed for this one (itself, or the hooks it creates)."""
        return (self,)

    def observes(self) -> Iterable[str]:
        """Other regions whose boundaries this strategy acts on (they must be observed)."""
        return ()

    def refs(self) -> Iterable[TensorRef]:
        """Boundary tensors this strategy may capture (see :meth:`RegionCall.capture`)."""
        return ()

    def active(self, owner: torch.nn.Module, replayed: bool) -> bool:
        """Whether the strategy applies to a call of ``owner``; ``replayed`` tells whether a
        replaying strategy of the region is active for the call.

        Called on every region call before anything is allocated for it, so it must be cheap.
        """
        return True

    def enter(self, call: RegionCall, args: tuple, kwargs: dict) -> tuple:
        """Before the region runs; returns its (possibly replaced) positional arguments."""
        return args

    def run(
        self, call: RegionCall, forward: Callable, recompute: Callable, args: tuple, kwargs: dict
    ) -> Any:
        """Run the region (wrapping strategies only). ``recompute`` reruns it in backward."""
        raise NotImplementedError

    def exit(self, call: RegionCall, out: Any) -> Any:
        """After the region returned ``out``; returns the (possibly replaced) output."""
        return out

    def abort(self, call: RegionCall) -> None:
        """The region raised after :meth:`enter`: undo what ``enter`` started."""


# ---------------------------------------------------------------------------------------------
# Recompute flavors (wrapping strategies)
# ---------------------------------------------------------------------------------------------


class _ForwardThenRecompute:
    """Call ``forward`` the first time and ``recompute`` on later calls.

    For a wrapped child module, the module's hooks already ran around the recorded forward, so
    the first call goes straight to the original ``forward``; a recompute re-enters through
    ``module.__call__`` so its hooks (for example parameter all-gathers) run again, exactly as
    when model code checkpointed the module itself.
    """

    __slots__ = ("forward", "recompute", "kwargs", "called")

    def __init__(self, forward: Callable, recompute: Callable, kwargs: Optional[dict] = None):
        self.forward, self.recompute, self.kwargs, self.called = forward, recompute, kwargs, False

    def __call__(self, *args, **kwargs):
        if self.kwargs:
            kwargs = {**self.kwargs, **kwargs}
        if not self.called:
            self.called = True
            return self.forward(*args, **kwargs)
        with runtime.recompute_guard():
            return self.recompute(*args, **kwargs)


class _Release(Action):
    """Release a recomputed output and hook its rebuild on the event's tensor."""

    __slots__ = ("ckpt",)
    stage = 1

    def __init__(self, ckpt):
        self.ckpt = ckpt

    def on_enter(self, hook_tensors: list) -> None:
        for tensor in hook_tensors:
            if tensor is not None:
                self.ckpt.discard_output_and_register_recompute(tensor)

    def on_exit(self, value: Any, state) -> Any:
        self.on_enter([runtime._first_tensor(value)])
        return value


class CheckpointWithoutOutputStrategy(Strategy):
    """Discard the region's output in forward and rebuild it in backward.

    Implemented with ``CheckpointWithoutOutput``.

    Args:
        region: The region.
        release_on: When the output storage is released; the rebuild hooks on the gradient of
            that event's tensor, which must be computed before the output is used in backward.
        retain_input_tensors: Keep outputs that share storage with an input when releasing.
        training_only: Only recompute while the owner is in training mode.
    """

    __slots__ = ("release_on", "retain_input_tensors", "training_only", "source")
    wraps = True
    order = 1
    exclusive = "execution"

    def __init__(
        self,
        region: Any,
        release_on: Event,
        *,
        retain_input_tensors: bool = False,
        training_only: bool = False,
    ):
        super().__init__(region)
        self.release_on = release_on
        self.retain_input_tensors = retain_input_tensors
        self.training_only = training_only
        self.source = (region, "recompute")

    def observes(self) -> Iterable[str]:
        return (self.release_on.region,)

    def active(self, owner, replayed) -> bool:
        return torch.is_grad_enabled() and (owner.training or not self.training_only)

    def run(self, call, forward, recompute, args, kwargs):
        ckpt = _checkpointing.CheckpointWithoutOutput(
            retain_input_tensors=self.retain_input_tensors
        )
        out = ckpt.checkpoint(_ForwardThenRecompute(forward, recompute, kwargs), *args)
        call.on(self.release_on, self.source, _Release(ckpt))
        return out


def _is_tensor_tuple(value: Any) -> bool:
    return (
        isinstance(value, (tuple, list))
        and any(torch.is_tensor(v) for v in value)
        and all(v is None or torch.is_tensor(v) for v in value)
    )


def _flatten_tensors(
    args: tuple, kwargs: dict, *, tensor_tuples: bool = False
) -> tuple[list, Callable[[tuple], tuple]]:
    """Split a call's tensors from its other arguments.

    Returns the tensors in order and ``rebuild(tensors) -> (args, kwargs)``. A reentrant
    checkpoint saves only positional tensors, so tensors must be passed to it positionally (to be
    detached for recompute) while everything else is captured. With ``tensor_tuples``, tuples and
    lists of tensors are split into their elements too.
    """
    names = list(kwargs)
    values = list(args) + [kwargs[name] for name in names]
    slots: list[tuple[int, Optional[int]]] = []  # (value index, index within a tuple value)
    tensors: list = []
    for i, value in enumerate(values):
        if torch.is_tensor(value):
            slots.append((i, None))
            tensors.append(value)
        elif tensor_tuples and _is_tensor_tuple(value):
            for j, item in enumerate(value):
                slots.append((i, j))
                tensors.append(item)
    nargs = len(args)

    def rebuild(flat: tuple) -> tuple:
        rebuilt = list(values)
        for (i, j), tensor in zip(slots, flat):
            if j is None:
                rebuilt[i] = tensor
            else:
                items = rebuilt[i] if isinstance(rebuilt[i], list) else list(rebuilt[i])
                items[j] = tensor
                rebuilt[i] = items
        for i in {i for i, j in slots if j is not None}:
            rebuilt[i] = type(values[i])(rebuilt[i])
        return tuple(rebuilt[:nargs]), dict(zip(names, rebuilt[nargs:]))

    return tensors, rebuild


def _tp_group(owner: torch.nn.Module):
    pg_collection = getattr(owner, "pg_collection", None)
    return pg_collection.tp if pg_collection is not None else getattr(owner, "tp_group", None)


class CheckpointStrategy(Strategy):
    """Recompute the whole region in backward, keeping its output.

    Args:
        region: The region.
        impl: ``"precision_aware"`` uses Transformer Engine's checkpoint under FP8/FP4 and the
            reentrant ``checkpointing.checkpoint`` otherwise; ``"tensor_parallel"`` always the
            latter; ``"non_reentrant"`` uses PyTorch's non-reentrant checkpoint, and only when
            grad is enabled and a tensor input requires grad.
        training_only: Only checkpoint while the owner is in training mode.
        distribute_saved_activations: Shard the saved first input across the TP group.
        delayed_fp8_context: Enter the delayed-scaling FP8 context around the checkpoint (TE
            samples it at checkpoint entry), for regions outside a block's FP8 context.
        preserve_rng_state: ``"non_reentrant"`` only: restore the RNG state for the recompute.
    """

    __slots__ = (
        "impl",
        "training_only",
        "distribute_saved_activations",
        "delayed_fp8_context",
        "preserve_rng_state",
    )
    wraps = True
    order = 1
    exclusive = "execution"

    def __init__(
        self,
        region: Any,
        *,
        impl: str = "precision_aware",
        training_only: bool = False,
        distribute_saved_activations: bool = False,
        delayed_fp8_context: bool = False,
        preserve_rng_state: bool = True,
    ):
        super().__init__(region)
        if impl not in ("precision_aware", "tensor_parallel", "non_reentrant"):
            raise ValueError(f"Unknown checkpoint implementation {impl!r}.")
        self.impl = impl
        self.training_only = training_only
        self.distribute_saved_activations = distribute_saved_activations
        self.delayed_fp8_context = delayed_fp8_context
        self.preserve_rng_state = preserve_rng_state

    def active(self, owner, replayed) -> bool:
        return owner.training or not self.training_only

    def run(self, call, forward, recompute, args, kwargs):
        return self.checkpoint(call.owner, forward, recompute, args, kwargs)

    def checkpoint(
        self,
        owner: torch.nn.Module,
        forward: Callable,
        recompute: Callable,
        args: tuple,
        kwargs: dict,
        *,
        tensor_tuples: bool = False,
    ) -> Any:
        """Run ``forward(*args, **kwargs)`` of ``owner`` under this checkpoint policy.

        ``recompute`` is called instead of ``forward`` when the checkpoint recomputes in
        backward. ``tensor_tuples`` also passes the elements of tensor tuples positionally (see
        ``_flatten_tensors``).
        """
        policy = self
        from megatron.core.enums import Fp8Recipe

        tensors, rebuild = _flatten_tensors(args, kwargs, tensor_tuples=tensor_tuples)
        call = _ForwardThenRecompute(forward, recompute)

        def flat_forward(*flat):
            flat_args, flat_kwargs = rebuild(flat)
            return call(*flat_args, **flat_kwargs)

        if policy.impl == "non_reentrant":
            if not (
                torch.is_grad_enabled() and any(t is not None and t.requires_grad for t in tensors)
            ):
                return forward(*args, **kwargs)
            return torch.utils.checkpoint.checkpoint(
                flat_forward,
                *tensors,
                use_reentrant=False,
                preserve_rng_state=policy.preserve_rng_state,
            )

        distribute = policy.distribute_saved_activations
        if policy.impl == "tensor_parallel":
            return _checkpointing.checkpoint(flat_forward, distribute, *tensors)

        config = owner.config
        outer = nullcontext()
        if policy.delayed_fp8_context and config.fp8 and config.fp8_recipe == Fp8Recipe.delayed:
            from megatron.core.fp8_utils import get_fp8_context

            outer = get_fp8_context(config)
        with outer:
            return precision_aware_checkpoint(
                flat_forward,
                config,
                _tp_group(owner),
                *tensors,
                distribute_saved_activations=distribute,
            )


class LayerWiseCheckpointStrategy(Strategy):
    """Full activation recompute (``recompute_granularity="full"``) of a block's layers.

    Its region is the block's per-layer step, a method region: ``run_layer(index, hidden_states,
    context, **kwargs) -> (hidden_states, context)`` runs layer ``index`` with the block's work
    around it (for example context-parallel layout conversions). The block calls its steps in
    order, starting at 0; while training, the strategy decides from the index how each runs:

    * ``"uniform"``: the steps form chunks of ``recompute_num_layers``. At a chunk's first step
      the whole chunk runs inside one checkpoint (only the chunk input is saved; the chunk is
      recomputed in backward from it); the chunk's other steps return the outputs that run
      produced, and the outputs inside the chunk are released once the chunk ends.
    * ``"block"``: the first ``recompute_num_layers`` steps are checkpointed, one each; the
      others run normally.

    Args:
        run_layer: The block's per-layer step (a bound method of the block).
        layers: The block's layers, one per step.
        method: ``"uniform"`` or ``"block"``.
        recompute_num_layers: ``recompute_num_layers``.
        distribute_saved_activations: Shard the saved checkpoint input across the TP group.
        viewless_checkpoint_input: Make the checkpoint input viewless (mHC expands the block
            input and its BDA can return views; distributed checkpointing replaces ``.data``).
        quantized: FP8/FP4 training. ``"block"`` then skips steps whose input does not yet
            require grad (the reentrant checkpoint needs one input that does), shifting its
            window past them.

    When the block's DSA layers share top-k indices across layers and the forward has no
    ``packed_seq_params``, the steps of one forward, and their recompute, run with one per-forward
    index-share carrier, so a recompute finds the indices of its own forward.
    """

    __slots__ = (
        "num_layers",
        "method",
        "recompute_num_layers",
        "viewless_checkpoint_input",
        "quantized",
        "checkpointer",
        "skipped",
        "chunk",
        "carrier",
    )
    wraps = True
    order = 1
    exclusive = "execution"

    def __init__(
        self,
        run_layer: Callable,
        layers,
        *,
        method: str,
        recompute_num_layers: int,
        distribute_saved_activations: bool = False,
        viewless_checkpoint_input: bool = False,
        quantized: bool = False,
    ):
        super().__init__(run_layer)
        if method not in ("uniform", "block"):
            raise ValueError(f"Invalid activation recompute method {method!r}.")
        self.num_layers = len(layers)
        self.method = method
        self.recompute_num_layers = recompute_num_layers
        self.viewless_checkpoint_input = viewless_checkpoint_input
        self.quantized = quantized
        self.checkpointer = CheckpointStrategy(
            None, distribute_saved_activations=distribute_saved_activations
        )
        # Per forward: the "block" steps skipped under FP8/FP4, the open "uniform" chunk, and the
        # DSA index-share carrier.
        self.skipped = 0
        self.chunk: Optional[_Chunk] = None
        self.carrier = None

    def active(self, owner, replayed) -> bool:
        return owner.training

    def run(self, call, forward, recompute, args, kwargs):
        index, hidden_states, context = args
        if index == 0:
            self.skipped, self.chunk, self.carrier = 0, None, None
            config = call.owner.config
            if (
                kwargs.get("packed_seq_params") is None
                and getattr(config, "experimental_attention_variant", None) == "dsa"
                and (getattr(config, "dsa_indexer_topk_freq", 1) or 1) > 1
            ):
                from megatron.core.transformer.experimental_attention_variant.dsa import (
                    _DSAIndexShareCarrier,
                )

                self.carrier = _DSAIndexShareCarrier()
        if self.carrier is not None:
            forward = _in_dsa_carrier_scope(forward, self.carrier)
        if self.method == "block":
            if self.quantized and not hidden_states.requires_grad:
                self.skipped += 1
            if self.skipped <= index < self.recompute_num_layers + self.skipped:
                return self._checkpoint(call.owner, forward, index, index + 1, args, kwargs)
            return forward(*args, **kwargs)

        start = index - index % self.recompute_num_layers
        end = min(start + self.recompute_num_layers, self.num_layers)
        if end - start == 1:
            return self._checkpoint(call.owner, forward, start, end, args, kwargs)
        if index == start:
            self.chunk = _Chunk(start, end, hidden_states)
            self.chunk.outputs[end - 1] = self._checkpoint(
                call.owner, forward, start, end, args, kwargs, chunk=self.chunk
            )
        chunk = self.chunk
        if chunk is None or not chunk.start <= index < chunk.end:
            raise RuntimeError(f"Layer step {index} ran outside its full-recompute chunk.")
        if index > start and hidden_states is not chunk.outputs[index - 1][0]:
            raise RuntimeError(
                f"The input of layer step {index} is not the output of step {index - 1}: work "
                "between the steps of a full-recompute chunk would not be recomputed."
            )
        out = chunk.outputs[index]
        if index == end - 1:
            chunk.release_inner()
            self.chunk = None
        return out

    def _checkpoint(self, owner, forward, start, end, args, kwargs, chunk=None):
        """Run steps ``start..end-1`` inside one checkpoint; returns the last step's output.

        With ``chunk``, the first (forward) run records each step's output in it.
        """
        _, hidden_states, context = args
        if self.viewless_checkpoint_input:
            hidden_states = make_viewless_tensor(
                inp=hidden_states, requires_grad=True, keep_graph=True
            )

        def run_steps(record):
            def steps(hidden_states, context, **shared):
                for index in range(start, end):
                    hidden_states, context = forward(index, hidden_states, context, **shared)
                    if record and index < end - 1:
                        chunk.outputs[index] = (hidden_states, context)
                return hidden_states, context

            return steps

        return self.checkpointer.checkpoint(
            owner,
            run_steps(chunk is not None),
            run_steps(False),
            (hidden_states, context),
            kwargs,
            tensor_tuples=True,
        )


def _in_dsa_carrier_scope(fn: Callable, carrier: object) -> Callable:
    """``fn`` run with ``carrier`` as the DSA index-share carrier (in forward and recompute)."""
    from megatron.core.transformer.experimental_attention_variant.dsa import (
        _dsa_index_share_carrier_scope,
    )

    def scoped(*args, **kwargs):
        with _dsa_index_share_carrier_scope(carrier):
            return fn(*args, **kwargs)

    return scoped


class _Chunk:
    """A "uniform" chunk of :class:`LayerWiseCheckpointStrategy`: the outputs of its steps."""

    __slots__ = ("start", "end", "input", "outputs")

    def __init__(self, start: int, end: int, chunk_input: torch.Tensor):
        self.start, self.end, self.input = start, end, chunk_input
        self.outputs: dict = {}

    def release_inner(self) -> None:
        """Release the outputs inside the chunk (they are rebuilt in backward, as in a
        checkpointed chunk, and must not be used), keeping any storage they share with the
        chunk's input or output."""
        kept = {
            tensor.untyped_storage().data_ptr()
            for tensor in (self.input, *self.outputs[self.end - 1])
            if torch.is_tensor(tensor)
        }
        for index in range(self.start, self.end - 1):
            hidden_states = self.outputs[index][0]
            storage = hidden_states.untyped_storage()
            if storage.data_ptr() not in kept:
                storage.resize_(0)


# ---------------------------------------------------------------------------------------------
# Offloading
# ---------------------------------------------------------------------------------------------


class _CloseHooks(Action):
    """Remove an offload group's saved-tensor hooks (a group spanning several regions)."""

    __slots__ = ("group",)
    stage, lifo = 0, True

    def __init__(self, group):
        self.group = group

    def on_exit(self, value: Any, state) -> Any:
        self.group.__exit__(None, None, None)
        return value


class _Commit(Action):
    """Copy an offload group to host and force-release its captured tensors."""

    __slots__ = ("group", "policy")
    stage, lifo = 2, True  # inner groups commit first

    def __init__(self, group, policy: OffloadStrategy):
        self.group, self.policy = group, policy

    def on_exit(self, value: Any, state) -> Any:
        released = [state.take_record(ref) for ref in self.policy.release]
        return self.group.group_offload(
            value,
            forced_released_tensors=[t for t in released if _releasable(t)],
            delay_offload=self.policy.delay,
        )


class OffloadStrategy(Strategy):
    """Offload what the region saves for backward to host memory (fine-grained offloading).

    The group starts when the region is entered: its first argument goes through the offload
    group-start marker and the saved-tensor hooks are installed.

    Args:
        region: The region the group starts at.
        group: Offload group name, shared with the offload manager's scheduling.
        commit_on: Region whose exit commits the group (default: this region).
        context_until: Region whose exit removes the saved-tensor hooks (default: this region),
            for a group that spans consecutive regions.
        release: Tensors whose storage is force-released once the group is copied. Tensors whose
            storage the region does not own are skipped (see ``mark_borrowed``).
        delay: Queue the copy until CUDA-graph replay returns.
        training_only: Only offload while the owner is in training mode.
        skip_when_replayed: Do not offload a call that is replayed with its block.
        validate: Check run on the region's ``(args, kwargs)`` before the group starts.
    """

    __slots__ = (
        "group",
        "commit_on",
        "context_until",
        "release",
        "delay",
        "training_only",
        "skip_when_replayed",
        "validate",
        "close_event",
        "commit_event",
        "source",
    )

    def __init__(
        self,
        region: Any,
        group: str,
        *,
        commit_on: Any = None,
        context_until: Any = None,
        release: tuple[TensorRef, ...] = (),
        delay: bool = False,
        training_only: bool = False,
        skip_when_replayed: bool = False,
        validate: Optional[Callable[[tuple, dict], None]] = None,
    ):
        super().__init__(region)
        self.group = group
        self.commit_on = commit_on
        self.context_until = context_until
        self.release = tuple(release)
        self.delay = delay
        self.training_only = training_only
        self.skip_when_replayed = skip_when_replayed
        self.validate = validate
        until = region if context_until is None else context_until
        # None: the hooks close when this region returns.
        self.close_event = None if _same_region(until, region) else Event(until, "exit")
        self.commit_event = Event(region if commit_on is None else commit_on, "exit")
        self.source = (region, group)

    def observes(self) -> Iterable[str]:
        regions = [r for r in (self.commit_on, self.context_until) if r is not None]
        return regions + [ref.region for ref in self.release]

    def refs(self) -> Iterable[TensorRef]:
        return self.release

    def active(self, owner, replayed) -> bool:
        if self.training_only and not owner.training:
            return False
        return not (self.skip_when_replayed and replayed)

    def enter(self, call: RegionCall, args: tuple, kwargs: dict) -> tuple:
        if self.validate is not None:
            self.validate(args, kwargs)
        group = _offload_interface()(True, args[0], self.group)
        args = (group.__enter__(),) + args[1:]
        group.tensor = None  # the group needs no reference to its input after starting
        call.capture(self.release)
        call.data[self] = group
        return args

    def exit(self, call: RegionCall, out: Any) -> Any:
        group = call.data.pop(self, None)
        if group is None:  # a signalled region: no group started here
            return out
        if self.close_event is None:
            group.__exit__(None, None, None)
        else:
            # Each group gets its own source: hooks must close even if an earlier group's did not.
            call.on(self.close_event, (self.source, id(group)), _CloseHooks(group))
        call.on(self.commit_event, self.source, _Commit(group, self))
        return out

    def abort(self, call: RegionCall) -> None:
        group = call.data.pop(self, None)
        if group is not None:
            group.__exit__(None, None, None)
            call.uncapture(self.release)


class LayerCPUOffloadStrategy(Strategy):
    """Transformer Engine's legacy per-layer CPU offloading of a block's layers.

    Its region is one layer of the block: the layer runs inside TE's offload context; with
    grad enabled, its output ``(hidden_states, ...)`` then commits the layer's group.

    Args:
        region: The layer.
        offload: The block's TE offload context and commit.
        unless_full_recompute: Not while the block runs under full recompute (training with
            full recompute configured), which never combined with this offloading.
    """

    __slots__ = ("offload", "unless_full_recompute")

    def __init__(self, region: Any, offload: TECPUOffload, *, unless_full_recompute: bool = False):
        super().__init__(region)
        self.offload = offload
        self.unless_full_recompute = unless_full_recompute

    def active(self, owner, replayed) -> bool:
        return not (self.unless_full_recompute and owner.training)

    def enter(self, call: RegionCall, args: tuple, kwargs: dict) -> tuple:
        self.offload.context.__enter__()
        call.data[self] = True
        return args

    def exit(self, call: RegionCall, out: Any) -> Any:
        if call.data.pop(self, None) is None:
            return out
        self.offload.context.__exit__(None, None, None)
        if torch.is_grad_enabled():
            hidden_states, *rest = out
            out = (self.offload.commit(hidden_states), *rest)
        return out

    def abort(self, call: RegionCall) -> None:
        if call.data.pop(self, None) is not None:
            self.offload.context.__exit__(*sys.exc_info())


class ExcludeFromOffloadStrategy(Strategy):
    """When the region is replayed with its block, keep its output out of other offload groups."""

    __slots__ = ()
    order = 20  # after the region's own offload groups closed their hooks

    def active(self, owner, replayed) -> bool:
        return replayed

    def exit(self, call: RegionCall, out: Any) -> Any:
        _offload_interface().mark_not_offload(runtime._first_tensor(out))
        return out


# ---------------------------------------------------------------------------------------------
# Installing
# ---------------------------------------------------------------------------------------------


def full_recompute_segments(method: str, recompute_num_layers: int, num_layers: int):
    """The layer ranges full recompute checkpoints (without the FP8/FP4 ``"block"`` shift).

    ``"uniform"`` checkpoints every chunk of ``recompute_num_layers`` layers; ``"block"`` the
    first ``recompute_num_layers`` layers, one per checkpoint. This is the single definition of
    the segmentation, shared by :class:`LayerWiseCheckpointStrategy` and the EP-overlap
    schedule.
    """
    if method == "uniform":
        return tuple(
            (start, min(start + recompute_num_layers, num_layers))
            for start in range(0, num_layers, recompute_num_layers)
        )
    if method == "block":
        return tuple((i, i + 1) for i in range(min(recompute_num_layers, num_layers)))
    raise ValueError(f"Invalid activation recompute method {method!r}.")


def order_region(region: Any, strategies: Iterable[Strategy]) -> tuple[Strategy, ...]:
    """A region's strategies in the order the runtime applies them (see the module docstring).

    Raises if two strategies of the same ``exclusive`` group (or two replays) share the region.
    """
    strategies = list(strategies)
    seen: dict = {}
    for strategy in strategies:
        key = strategy.exclusive or ("replay" if strategy.replays else None)
        if key is None:
            continue
        if key in seen:
            raise ValueError(f"Region {region!r} cannot have both {seen[key]!r} and {strategy!r}.")
        seen[key] = strategy
    return tuple(sorted(strategies, key=lambda strategy: strategy.order))


def fire_exit_actions(actions: list[Action], value: Any, state) -> Any:
    """Fire exit actions by stage (see the module docstring); returns the final output."""
    if len(actions) > 1:
        stages = sorted({a.stage for a in actions})
        ordered: list[Action] = []
        for stage in stages:
            group = [a for a in actions if a.stage == stage]
            ordered.extend(reversed(group) if group[0].lifo else group)
        actions = ordered
    for action in actions:
        value = action.on_exit(value, state)
    return value


__all__ = [
    "Action",
    "CheckpointStrategy",
    "ExcludeFromOffloadStrategy",
    "LayerWiseCheckpointStrategy",
    "LayerCPUOffloadStrategy",
    "OffloadStrategy",
    "CheckpointWithoutOutputStrategy",
    "RegionCall",
    "Strategy",
    "full_recompute_segments",
    "order_region",
]
