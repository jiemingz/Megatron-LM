# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""Residual replay: discard a block of layers' residual-stream activations, replay them in backward.

mHC and wide-residual layers read a branch input from a wide residual stream and write the branch
output back. Those reads and writes, and the cheap ops between a read and its branch (norms,
mHC's ``h_post``), are recorded into one :class:`CheckpointWithoutOutputManager` per block of
layers. The layer output of a block's last layer stays live: when the layer step that ends the
block returns, the block's recorded outputs are released, and the whole block is rebuilt in
forward order when that step output's gradient arrives.

:class:`ResidualReplayStrategy` is installed by the layer stack, like full recompute: it is given
the stack's per-layer step and its layers, takes each layer's :class:`ResidualBranch` list
(``layer.get_residual_branches_by_layer()``) and hooks those regions on their owners.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, NamedTuple, Optional

import torch

from megatron.core.activation_policy import runtime
from megatron.core.activation_policy.recompute.checkpointing import (
    CheckpointWithoutOutput,
    CheckpointWithoutOutputManager,
)
from megatron.core.activation_policy.strategies import Strategy, _ForwardThenRecompute


class ReplayRegion(NamedTuple):
    """A region of ``owner``: a child module, or a method of ``owner``.

    ``retain_input_tensors`` keeps outputs that share storage with an input when releasing (a
    norm that returns its residual input).
    """

    owner: torch.nn.Module
    region: Any
    retain_input_tensors: bool = False


class ResidualBranch(NamedTuple):
    """One branch's residual read, the ops between it and the branch, and its write.

    The ops are every link of the chain from the read to the branch input and from the branch
    output to the write: an op left out keeps its output (and pins the tensor it feeds).
    """

    read: Optional[ReplayRegion] = None
    ops: Sequence[ReplayRegion] = ()
    write: Optional[ReplayRegion] = None


def build_recompute_block_end_plan(
    num_layers: int, block_size: int | None, *, atomic_layer_pairs: Sequence[tuple[int, int]] = ()
) -> tuple[bool, ...]:
    """Mark block ends without splitting disjoint adjacent atomic layer pairs.

    Indices use the caller's local layer order. A pair may exceed a block size of
    one; otherwise blocks contain at most ``block_size`` layers. ``None`` uses
    the entire local stack. Empty stacks return an empty plan.
    """
    if num_layers < 0:
        raise ValueError("Residual recompute plan requires a non-negative layer count.")
    if num_layers == 0:
        return ()
    if block_size is not None and (
        isinstance(block_size, bool) or not isinstance(block_size, int) or block_size < 1
    ):
        raise ValueError("Residual recompute block size must be a positive integer or None.")

    effective_block_size = block_size or num_layers
    if not atomic_layer_pairs:
        return tuple(
            (layer_index + 1) % effective_block_size == 0 or layer_index == num_layers - 1
            for layer_index in range(num_layers)
        )

    pair_starts: dict[int, int] = {}
    paired_indices: set[int] = set()
    for pair in atomic_layer_pairs:
        if not isinstance(pair, tuple) or len(pair) != 2:
            raise ValueError("Atomic residual recompute layer pairs must be two-item tuples.")
        start, end = pair
        if (
            isinstance(start, bool)
            or not isinstance(start, int)
            or isinstance(end, bool)
            or not isinstance(end, int)
        ):
            raise ValueError("Atomic residual recompute layer-pair indices must be integers.")
        if start < 0 or end >= num_layers or end != start + 1:
            raise ValueError(
                "Atomic residual recompute layer pairs must contain adjacent in-range indices."
            )
        if start in paired_indices or end in paired_indices:
            raise ValueError("Atomic residual recompute layer pairs must not overlap.")
        pair_starts[start] = end
        paired_indices.update((start, end))

    # Treat each unpaired layer or complete pair as one indivisible unit.
    atomic_units = []
    layer_index = 0
    while layer_index < num_layers:
        unit_end = pair_starts.get(layer_index, layer_index)
        atomic_units.append((layer_index, unit_end))
        layer_index = unit_end + 1

    # Pack whole units in order, shortening a block instead of splitting a pair.
    block_ends: set[int] = set()
    layers_in_block = 0
    for unit_start, unit_end in atomic_units:
        unit_size = unit_end - unit_start + 1
        if layers_in_block and layers_in_block + unit_size > effective_block_size:
            block_ends.add(unit_start - 1)
            layers_in_block = 0
        layers_in_block += unit_size
        if layers_in_block >= effective_block_size:
            block_ends.add(unit_end)
            layers_in_block = 0
    block_ends.add(num_layers - 1)
    return tuple(layer_index in block_ends for layer_index in range(num_layers))


class ResidualReplayStrategy(Strategy):
    """Replay blocks of layers' residual reads, ops and writes (mHC, wide residual).

    Args:
        run_layer: The layer stack's per-layer step (a bound method of the stack,
            ``run_layer(index, hidden_states, ...)`` returning the step's hidden states first).
        layers: The stack's layers, one per step; each gives its physical layers' residual
            branches (``get_residual_branches_by_layer``, two for a ShortcutMoE pair).
        block_size: Physical layers per replay block (``None``: all of them).
        atomic_pairs: Adjacent physical layer pairs a block must not split (ShortcutMoE).

    Every region is replayed with its block (:class:`_ReplayHook`) except the layer output of a
    block's last layer (its last write), which stays live. When the step that ends a block
    returns, the block's recorded outputs are released and their rebuild is hooked on the
    step's output. Each block's manager is created by its first replayed region of a forward.
    """

    __slots__ = ("hooks", "managers", "block_end_steps", "steps")

    def __init__(
        self,
        run_layer,
        layers,
        *,
        block_size: Optional[int],
        atomic_pairs: Sequence[tuple[int, int]] = (),
    ):
        super().__init__(run_layer)
        # Each physical layer's branches, and the step that runs it.
        physical: list[Sequence[ResidualBranch]] = []
        steps: list[int] = []
        for step, layer in enumerate(layers):
            for branches in layer.get_residual_branches_by_layer():
                physical.append(branches)
                steps.append(step)
        block_ends = build_recompute_block_end_plan(
            len(physical), block_size, atomic_layer_pairs=atomic_pairs
        )
        hooks: list[Strategy] = [self]
        self.block_end_steps: dict[int, int] = {}
        block = 0
        for index, (branches, is_block_end) in enumerate(zip(physical, block_ends)):
            regions = []
            for branch in branches:
                regions.extend(r for r in (branch.read, *branch.ops) if r is not None)
                if branch.write is not None:
                    regions.append(branch.write)
            writes = [branch.write for branch in branches if branch.write is not None]
            # The layer output of the block's last layer is the block's live boundary.
            live = writes[-1] if is_block_end and writes else None
            hooks.extend(
                _ReplayHook(self, region, block) for region in regions if region is not live
            )
            if is_block_end:
                if index + 1 < len(steps) and steps[index + 1] == steps[index]:
                    raise ValueError("A residual replay block cannot end inside a layer step.")
                self.block_end_steps[steps[index]] = block
                block += 1
        self.hooks = tuple(hooks)
        self.managers: list[Optional[CheckpointWithoutOutputManager]] = [None] * block
        # Whether each running step (innermost last) replays. A region inside a step follows
        # the step's decision, not a grad scope the region is called in (for example a residual
        # write under ``bias_dropout_add_exec_handler``, while the step runs under no_grad).
        self.steps: list[bool] = []

    def expand(self):
        return self.hooks

    def manager(self, block: int) -> CheckpointWithoutOutputManager:
        """The block's manager for this forward (created by its first replayed region)."""
        manager = self.managers[block]
        if manager is None:
            manager = self.managers[block] = CheckpointWithoutOutputManager()
        return manager

    def release(self, block: int) -> Optional[CheckpointWithoutOutputManager]:
        """Take the block's manager; the block's next replayed region starts a new one."""
        manager, self.managers[block] = self.managers[block], None
        return manager

    # The strategy itself acts on the layer step: it finalizes the block a step ends.

    def active(self, owner, replayed) -> bool:
        return True

    def enter(self, call, args, kwargs):
        replaying = _replaying(call.owner)
        call.data[self] = (args[0] if args else kwargs["index"]) if replaying else None
        self.steps.append(replaying)
        return args

    def exit(self, call, out: Any) -> Any:
        self.steps.pop()
        block = self.block_end_steps.get(call.data.pop(self, None))
        if block is not None:
            manager = self.release(block)
            if manager is not None:
                manager.discard_all_outputs_and_register_unified_recompute(
                    runtime._first_tensor(out)
                )
        return out

    def abort(self, call) -> None:
        call.data.pop(self, None)
        if self.steps:
            self.steps.pop()
        self.managers = [None] * len(self.managers)


def _replaying(owner: torch.nn.Module) -> bool:
    # Without grad nothing is saved for backward: nothing to discard or rebuild. A recompute
    # for backward reruns forward code whose regions were already recorded.
    return owner.training and torch.is_grad_enabled() and not runtime.recomputing()


class _ReplayHook(Strategy):
    """Record a region into its block's manager; its output is released when the block ends."""

    __slots__ = ("replay", "owner", "block", "retain_input_tensors")
    wraps = True
    replays = True
    order = 0

    def __init__(self, replay: ResidualReplayStrategy, region: ReplayRegion, block: int):
        super().__init__(region.region)
        self.replay = replay
        self.owner = region.owner
        self.block = block
        self.retain_input_tensors = region.retain_input_tensors

    def active(self, owner, replayed) -> bool:
        steps = self.replay.steps
        return (not steps or steps[-1]) and _replaying(owner)

    def run(self, call, forward, recompute, args, kwargs):
        ckpt = CheckpointWithoutOutput(
            ckpt_manager=self.replay.manager(self.block),
            retain_input_tensors=self.retain_input_tensors,
        )
        return ckpt.checkpoint(_ForwardThenRecompute(forward, recompute, kwargs), *args)
