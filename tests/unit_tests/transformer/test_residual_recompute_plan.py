# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""CPU logic contracts for shared residual-recompute planning and its adapters."""

from unittest.mock import Mock

import pytest
import torch
from torch import nn

from megatron.core.activation_policy import (
    ReplayRegion,
    ResidualBranch,
    ResidualReplayStrategy,
    install_activation_strategies,
)
from megatron.core.tensor_parallel.random import CheckpointWithoutOutputManager
from megatron.core.transformer import residual_recompute


@pytest.mark.parametrize(
    "num_layers,block_size,atomic_layer_pairs,expected",
    [
        pytest.param(0, None, (), (), id="empty"),
        pytest.param(1, None, (), (True,), id="singleton"),
        pytest.param(4, None, (), (False, False, False, True), id="whole-stack"),
        pytest.param(4, 2, (), (False, True, False, True), id="uniform"),
        pytest.param(5, 2, (), (False, True, False, True, True), id="partial-tail"),
        pytest.param(3, 1, (), (True, True, True), id="single-layer-blocks"),
        pytest.param(3, 8, (), (False, False, True), id="oversized-block"),
        pytest.param(
            4, 3, ((2, 3), (0, 1)), (False, True, False, True), id="unsorted-atomic-pairs"
        ),
        pytest.param(
            5, 1, ((1, 2), (3, 4)), (True, False, True, False, True), id="pairs-exceed-target"
        ),
        pytest.param(
            8,
            4,
            ((3, 4), (5, 6)),
            (False, False, True, False, False, False, True, True),
            id="shorten-before-pair",
        ),
        pytest.param(4, None, ((1, 2),), (False, False, False, True), id="atomic-whole-stack"),
    ],
)
def test_block_end_plan(num_layers, block_size, atomic_layer_pairs, expected):
    block_ends = residual_recompute.build_recompute_block_end_plan(
        num_layers, block_size, atomic_layer_pairs=atomic_layer_pairs
    )

    assert isinstance(block_ends, tuple)
    assert block_ends == expected
    assert all(isinstance(is_end, bool) for is_end in block_ends)


@pytest.mark.parametrize("block_size", [False, True, 0, -1, 1.5])
def test_block_end_plan_rejects_invalid_block_size(block_size):
    with pytest.raises(ValueError, match="positive integer"):
        residual_recompute.build_recompute_block_end_plan(3, block_size)


@pytest.mark.parametrize(
    "pairs,message",
    [
        ([(0,)], "two-item tuples"),
        ([(0, 1, 2)], "two-item tuples"),
        ([[0, 1]], "two-item tuples"),
        ([(False, 1)], "indices must be integers"),
        ([(0, True)], "indices must be integers"),
        ([(0.0, 1)], "indices must be integers"),
        ([(0, 1.0)], "indices must be integers"),
        ([(-1, 0)], "adjacent in-range"),
        ([(1, 3)], "adjacent in-range"),
        ([(2, 3)], "adjacent in-range"),
        ([(1, 0)], "adjacent in-range"),
        ([(0, 1), (1, 2)], "must not overlap"),
        ([(0, 1), (0, 1)], "must not overlap"),
    ],
)
def test_block_end_plan_rejects_malformed_pairs(pairs, message):
    with pytest.raises(ValueError, match=message):
        residual_recompute.build_recompute_block_end_plan(3, 2, atomic_layer_pairs=pairs)


def test_block_end_plan_rejects_negative_layer_count():
    with pytest.raises(ValueError, match="non-negative layer count"):
        residual_recompute.build_recompute_block_end_plan(-1, None)


class _Layer(nn.Module):
    """A residual read and write, each a method point."""

    def forward(self, x):
        branch = self.read(x)
        return self.write(branch, x)

    def read(self, t):
        return t * 2

    def write(self, b, t):
        return b + t

    def get_residual_branches_by_layer(self):
        return [
            [
                ResidualBranch(
                    read=ReplayRegion(self, self.read), write=ReplayRegion(self, self.write)
                )
            ]
        ]


class _Stack(nn.Module):
    def __init__(self, num_layers, block_size):
        super().__init__()
        self.layers = nn.ModuleList(_Layer() for _ in range(num_layers))
        self.__dict__["replay"] = ResidualReplayStrategy(
            self._run_layer, self.layers, block_size=block_size
        )
        install_activation_strategies(self, [self.replay])

    def _run_layer(self, index, x):
        return self.layers[index](x)

    def forward(self, x):
        for index in range(len(self.layers)):
            x = self._run_layer(index, x)
        return x


@pytest.mark.parametrize("block_size", [False, True, 0, -1, 1.5])
def test_replay_strategy_rejects_sizes_disallowed_by_config(block_size):
    for num_layers in (1, 3):
        with pytest.raises(ValueError, match="positive integer"):
            _Stack(num_layers, block_size)


@pytest.mark.parametrize("num_layers,block_size", [(0, None), (5, None), (5, 2), (3, 1)])
def test_replay_blocks_get_fresh_managers_every_forward(monkeypatch, num_layers, block_size):
    """Each block records into one manager, created by its first replayed region of a forward and
    released when the step that ends the block returns."""
    from megatron.core.activation_policy.recompute import replay as replay_module

    created = []

    def manager():
        created.append(CheckpointWithoutOutputManager())
        return created[-1]

    monkeypatch.setattr(replay_module, "CheckpointWithoutOutputManager", manager)
    stack = _Stack(num_layers, block_size)
    num_blocks = sum(residual_recompute.build_recompute_block_end_plan(num_layers, block_size))
    x = torch.ones(2, 3, requires_grad=True)
    for forward in range(2):
        stack(x).sum().backward()
        assert len(created) == num_blocks * (forward + 1)
        assert stack.replay.managers == [None] * num_blocks
    assert len({id(m) for m in created}) == len(created)


@pytest.mark.parametrize("training,grad", [(False, True), (True, False)])
def test_replay_is_off_in_eval_and_without_grad(monkeypatch, training, grad):
    from megatron.core.activation_policy.recompute import replay as replay_module

    constructor = Mock(wraps=CheckpointWithoutOutputManager)
    monkeypatch.setattr(replay_module, "CheckpointWithoutOutputManager", constructor)
    stack = _Stack(3, 2).train(training)
    with torch.set_grad_enabled(grad):
        stack(torch.ones(2, 3, requires_grad=True))
    constructor.assert_not_called()


def test_each_block_finalizes_on_its_own_boundary_output(monkeypatch):
    stack = _Stack(5, 2)
    finalized = []
    original = CheckpointWithoutOutputManager.discard_all_outputs_and_register_unified_recompute

    def record(manager, hook_tensor):
        finalized.append((len(manager.checkpoints), hook_tensor))
        return original(manager, hook_tensor)

    monkeypatch.setattr(
        CheckpointWithoutOutputManager, "discard_all_outputs_and_register_unified_recompute", record
    )
    outputs = []
    for layer in stack.layers:
        layer.register_forward_hook(lambda module, args, out: outputs.append(out))
    x = torch.ones(2, 3, requires_grad=True)
    stack(x).sum().backward()
    # Blocks [0, 1], [2, 3], [4]: every region but the block's last write is recorded.
    assert [count for count, _ in finalized] == [3, 3, 1]
    assert [tensor is outputs[i] for (_, tensor), i in zip(finalized, (1, 3, 4))] == [True] * 3
