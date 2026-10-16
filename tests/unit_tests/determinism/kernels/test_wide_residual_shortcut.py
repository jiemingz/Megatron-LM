# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Bit-exact A/B output and gradient replay for stream-count specializations."""

import pytest
import torch

from megatron.core.fusions.wide_residual_shortcut import api
from tests.unit_tests.determinism.kernels.harness import assert_replays_bit_exact, seeded
from tests.unit_tests.fusions.test_wide_residual_shortcut import inputs

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA shortcut fusion")


@pytest.mark.parametrize("streams", range(1, 7))
@pytest.mark.parametrize("readers", [1, 2])
@pytest.mark.parametrize("retention", [False, True])
def test_standard_boundaries_replay(streams, readers, retention):
    seeded()
    raw = inputs(streams, readers, 128, 1024)
    if not retention:
        raw = (raw[0], raw[1], (raw[2][0], None, *raw[2][2:]), raw[3])

    def execute(residual, updates, controllers, gammas):
        if readers == 1:
            return api.write_read_norm(
                residual, updates[0], *controllers, gammas[0], num_streams=streams
            )
        return api.boundary_norm(
            residual, *updates, *controllers[:2], controllers[2:], gammas, num_streams=streams
        )

    assert_replays_bit_exact(execute, raw, replays=3)


@pytest.mark.parametrize("streams", [2, 3, 5, 6])
def test_production_dispatch_replay(streams):
    from megatron.core.fusions.wide_residual_shortcut.reconstruction_dispatch import (
        qualified_device,
    )

    if not qualified_device(torch.cuda.get_device_properties(torch.cuda.current_device())):
        pytest.skip("qualified GB300 production schedules")
    seeded()
    raw = inputs(streams, 2, 4096, 8192)

    def execute(residual, updates, controllers, gammas):
        return api.boundary_norm(
            residual, *updates, *controllers[:2], controllers[2:], gammas, num_streams=streams
        )

    assert_replays_bit_exact(execute, raw, replays=3)


@pytest.mark.parametrize("streams", [1, 2, 3, 5, 6])
@pytest.mark.parametrize("retention", [False, True])
@pytest.mark.parametrize("hidden", [96, 128])
def test_inverse_chain_replay(streams, retention, hidden):
    from megatron.core.fusions.wide_residual_shortcut import inverse_write
    from megatron.core.fusions.wide_residual_shortcut.residual_inverse import ResidualInverseManager
    from megatron.core.transformer.streamwise_residual_ops import streamwise_sigmoid_writeback

    assert callable(inverse_write.inverse_write)  # This bridge is exercised by backward below.
    seeded()
    raw = inputs(streams, 1, hidden, 1024)
    if not retention:
        raw = (raw[0], raw[1], (raw[2][0], None, *raw[2][2:]), raw[3])

    def execute(residual, updates, controllers, gammas):
        with ResidualInverseManager(16).activate():
            current, q0 = api.write_read_norm(
                residual, updates[0], *controllers, gammas[0], num_streams=streams
            )
            current = streamwise_sigmoid_writeback(
                current,
                updates[0],
                controllers[0],
                streams,
                retention_logits=controllers[1],
                retention_max_forget=0.1,
            )
            current, q1 = api.write_read_norm(
                current, updates[0], *controllers, gammas[0], num_streams=streams
            )
        # q0 is also an output so its independent reader gradient participates.
        return current, q0, q1

    assert_replays_bit_exact(execute, raw, replays=3)
