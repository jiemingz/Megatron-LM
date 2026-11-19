# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""The fused TEGroupedMLP path under an offload strategy."""

from types import SimpleNamespace

import pytest
import torch

from megatron.core.activation_policy import (
    OffloadStrategy,
    TensorRef,
    install_activation_strategies,
)
from megatron.core.activation_policy import strategies as strategies_module
from megatron.core.transformer.moe.experts import TEGroupedMLP


class _FakeOffloadInterface:
    log = []

    def __init__(self, offload, tensor, name):
        self.tensor, self.name = tensor, name

    def __enter__(self):
        self.log.append(("start", self.name))
        return self.tensor

    def __exit__(self, *args):
        self.log.append(("close", self.name))

    def group_offload(self, tensor, forced_released_tensors=None, delay_offload=False):
        self.log.append(("commit", self.name, tuple(forced_released_tensors), delay_offload))
        return tensor


def _fake_fused_experts(offload):
    class FakeFusedOps:
        def __call__(self, *args, **kwargs):
            _FakeOffloadInterface.log.append(("ops",))
            return args[0] + 1

    module = TEGroupedMLP.__new__(TEGroupedMLP)
    torch.nn.Module.__init__(module)
    module.config = SimpleNamespace(
        fp8=False,
        fp4=False,
        moe_router_padding_for_quantization=False,
        moe_token_dispatcher_type=None,
        moe_flex_dispatcher_backend=None,
        moe_use_grouped_tensor=True,
        moe_paged_stash=False,
    )
    module._with_fused_impl = True
    module._use_grouped_tensor = True
    module.offload_fused_group_mlp = offload
    module.quantization_padding = lambda tensor, token_counts: (tensor, token_counts)
    module.quantization_unpadding = lambda tensor, token_counts: tensor
    module._fused_ops = (FakeFusedOps(),)
    # No per-linear precision override (see TEGroupedMLP._fused_forward).
    module.linear_fc1 = SimpleNamespace(te_quant_params=None, training=True)
    module.linear_fc2 = SimpleNamespace(use_bias=False, te_quant_params=None, training=True)
    return module


@pytest.mark.parametrize("zero_copy", [False, True])
def test_fused_forward_offloads_and_releases_owned_input(monkeypatch, zero_copy):
    """The group wraps the fused op; its input is force-released only when the experts own it."""
    _FakeOffloadInterface.log = []
    monkeypatch.setattr(strategies_module, "_offload_interface", lambda: _FakeOffloadInterface)
    module = _fake_fused_experts(offload=True)
    install_activation_strategies(
        module,
        [
            OffloadStrategy(
                module._fused_group_mlp,
                "fused_group_mlp",
                release=(TensorRef.input(module._fused_group_mlp),),
            )
        ],
    )
    hidden_states = torch.zeros(2, 4)
    output_buffer = torch.empty(2, 4) if zero_copy else None

    output = module._fused_forward(
        hidden_states, torch.tensor([1, 1]), torch.ones(2), output_buffer=output_buffer
    )

    torch.testing.assert_close(output, torch.ones_like(hidden_states))
    log = _FakeOffloadInterface.log
    assert [entry[:2] for entry in log] == [
        ("start", "fused_group_mlp"),
        ("ops",),
        ("close", "fused_group_mlp"),
        ("commit", "fused_group_mlp"),
    ]
    released, delay = log[-1][2], log[-1][3]
    assert delay is False
    if zero_copy:
        assert released == ()
    else:
        assert len(released) == 1 and released[0] is hidden_states
