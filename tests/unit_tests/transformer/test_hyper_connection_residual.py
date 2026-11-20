# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""The mHC adapter's ownership, selected state, and replay region contract."""

import copy
from unittest.mock import patch

import pytest
import torch
from torch import nn

from megatron.core.activation_policy import ReplayRegion, ResidualBranch
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.hyper_connection import (
    HyperConnectionExecutionContext,
    HyperConnectionModule,
    HyperConnectionResidualConnection,
)
from megatron.core.transformer.transformer_config import TransformerConfig
from tests.unit_tests.test_utilities import Utils, install_residual_replay


def _model():
    config = TransformerConfig(
        num_layers=1,
        hidden_size=8,
        num_attention_heads=2,
        use_cpu_initialization=True,
        enable_mhc_connections=True,
        mhc_num_residual_streams=4,
        mhc_sinkhorn_iterations=3,
        use_fused_mhc=False,
    )
    model = nn.Module()
    model.owner = HyperConnectionModule(config, layer_number=1)
    model.read = HyperConnectionResidualConnection(config, operation="read")
    model.write = HyperConnectionResidualConnection(config, operation="write")
    return model.to(device="cuda" if torch.cuda.is_available() else "cpu")


def _step(model, value, bias=None, dropout=0.25):
    execution = HyperConnectionExecutionContext(model.owner)
    branch, state = model.read(value, execution_context=execution)
    return model.write(
        (branch.sin(), bias),
        state=state,
        dropout_probability=dropout,
        training=True,
        execution_context=execution,
    )


def _rng():
    return [torch.get_rng_state()] + (
        [torch.cuda.get_rng_state()] if torch.cuda.is_available() else []
    )


@pytest.mark.parametrize("deepcopy", [False, True])
@torch.compiler.set_stance("force_eager")
def test_adapter_owns_no_parameters_and_resolves_replaced_owner(deepcopy):
    original = _model()
    value = torch.randn(2, 1, 32, device=next(original.parameters()).device, requires_grad=True)
    _step(original, value)  # Populate any accidental cached dependency.
    model = copy.deepcopy(original) if deepcopy else original
    old_owner = original.owner
    if not deepcopy:
        model.owner = copy.deepcopy(old_owner)
    old_owner.register_forward_hook(lambda *args: pytest.fail("The stale owner was invoked"))
    for adapter in (model.read, model.write):
        assert not list(adapter.parameters()) and not list(adapter.children())
    assert list(model.state_dict()) == ["owner." + name for name in model.owner.state_dict()]
    _step(model, value).square().sum().backward()
    assert all(parameter.grad is None for parameter in old_owner.parameters())
    assert all(parameter.grad is not None for parameter in model.owner.parameters())


@torch.compiler.set_stance("force_eager")
def test_adapter_preserves_owner_selected_state_across_interleaved_reads():
    model = _model()
    selected, calls = [], []

    def select_alias(module, args, output):
        branch, h_res, h_post, residual = output
        selected.append((branch, h_res, h_post, residual.view_as(residual)))
        return selected[-1]

    model.owner.register_forward_hook(select_alias)
    for adapter in (model.read, model.write):
        adapter.register_forward_hook(lambda module, args, output: calls.append(module.operation))
    execution = HyperConnectionExecutionContext(model.owner)
    values = [
        torch.randn(
            2,
            1,
            32,
            device=next(model.parameters()).device,
            dtype=torch.bfloat16,
            requires_grad=True,
        )
        for _ in range(2)
    ]
    prepared = [
        model.read(
            value,
            fp32_residual_connection=True,
            branch_input_dtype=dtype,
            execution_context=execution,
        )
        for value, dtype in zip(values, (None, torch.bfloat16))
    ]
    for value, (branch, state), chosen in zip(values, prepared, selected):
        assert branch is chosen[0]
        assert all(a is b for a, b in zip(state, (chosen[3], chosen[1], chosen[2])))
        assert state[0] is not value and state[0].data_ptr() == value.data_ptr()
        assert all(t.dtype == value.dtype for t in state)
        assert all(t.grad_fn is not None for t in state[1:])
    with patch.object(
        model.owner, "fused_h_res_h_post_bda", wraps=model.owner.fused_h_res_h_post_bda
    ) as write:
        for branch, state in reversed(prepared):
            model.write(
                branch,
                state=state,
                dropout_probability=0.0,
                training=True,
                execution_context=execution,
            )
        for call, (_, state) in zip(write.call_args_list, reversed(prepared)):
            assert all(a is b for a, b in zip(call.args[:3], (state[1], state[0], state[2])))
    assert calls == ["read", "read", "write", "write"]
    with pytest.raises(TypeError, match="returned branch dtype.*expected"):
        model.read(values[0], branch_input_dtype=torch.float32, execution_context=execution)


@pytest.mark.parametrize("has_bias", [False, True])
@pytest.mark.parametrize("layer_output", [False, True])
@torch.compiler.set_stance("force_eager")
def test_owner_points_receive_flat_operands_and_bound_write_options(has_bias, layer_output):
    owner = _model().owner
    value = torch.randn(2, 1, 32, device=next(owner.parameters()).device, requires_grad=True)
    write_name = "_h_res_h_post_bda_layer_output" if layer_output else "_h_res_h_post_bda"
    other_name = "_h_res_h_post_bda" if layer_output else "_h_res_h_post_bda_layer_output"
    with (
        patch.object(owner, "aggregate", wraps=owner.aggregate) as aggregate,
        patch.object(owner, write_name, wraps=getattr(owner, write_name)) as write,
        patch.object(owner, other_name, wraps=getattr(owner, other_name)) as other,
    ):
        branch, h_res, h_post, residual = owner(value, return_residual=True)
        bias = torch.randn(8, device=value.device) if has_bias else None
        options = dict(dropout_prob=0.25, training=True, fused=False)
        owner.fused_h_res_h_post_bda(
            h_res, residual, h_post, (branch, bias), layer_output=layer_output, **options
        )
    assert aggregate.call_count == 1
    assert write.call_count == 1 and other.call_count == 0
    operands = write.call_args.args
    expected = (h_res, residual, h_post, branch) + ((bias,) if has_bias else ())
    assert len(operands) == len(expected)
    assert all(a is b for a, b in zip(operands, expected))
    assert write.call_args.kwargs == options


@pytest.fixture
def gpu_model_parallel():
    Utils.initialize_model_parallel(1, 1)
    model_parallel_cuda_manual_seed(123)
    yield
    Utils.destroy_model_parallel()


class _Step(nn.Module):
    """One replay layer: the mHC read (``aggregate``) and write, via the adapter or the owner."""

    def __init__(self, model, *, through_adapter):
        super().__init__()
        self.__dict__["model"] = model  # not registered: the model keeps its owner
        self.through_adapter = through_adapter

    def get_residual_branches_by_layer(self):
        owner = self.model.owner
        read = ReplayRegion(owner, owner.aggregate)
        write = ReplayRegion(owner, owner._h_res_h_post_bda)
        return [[ResidualBranch(read=read, write=write)]]

    def forward(self, value, bias, dropout):
        if self.through_adapter:
            return _step(self.model, value, bias, dropout)
        owner = self.model.owner
        branch, h_res, h_post, residual = owner(value, return_residual=True)
        return owner.fused_h_res_h_post_bda(
            h_res, residual, h_post, (branch.sin(), bias), dropout, True, False
        )


class _Boundary(nn.Module):
    """A live layer output after the mHC layer: ends its replay block."""

    def get_residual_branches_by_layer(self):
        return [[ResidualBranch(write=ReplayRegion(self, self.boundary))]]

    def forward(self, value):
        return self.boundary(value)

    def boundary(self, value):
        return torch.square(value)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Output-discard replay needs CUDA")
@pytest.mark.parametrize("mode", ["eager", "nonterminal", "terminal"])
@pytest.mark.parametrize("dropout", [0.0, 0.25])
@torch.compiler.set_stance("force_eager")
def test_adapter_replay_matches_direct_owner(mode, dropout, gpu_model_parallel):
    actual = _model()
    reference = copy.deepcopy(actual)
    template = torch.randn(2, 1, 32, device=next(actual.parameters()).device)
    bias_template = torch.randn(8, device=template.device)
    replay = mode != "eager"
    results = []
    for model in (reference, actual):
        value = template.detach().clone().requires_grad_()
        bias = bias_template.detach().clone().requires_grad_()
        owner = model.owner
        hooks = []
        for adapter in (model.read, model.write):
            adapter.register_forward_hook(
                lambda module, args, output: hooks.append(module.operation)
            )
        # The reference calls the owner directly, under the same replay schedule.
        layers = [_Step(model, through_adapter=model is actual), _Boundary()]
        stack = (
            install_residual_replay(layers, block_size=2 if mode == "nonterminal" else 1)
            if replay
            else None
        )
        run = stack._run_layer if replay else (lambda index, *args: layers[index](*args))
        with (
            patch.object(owner, "compute_mappings", wraps=owner.compute_mappings) as mappings,
            # The aggregate point's arithmetic (forward and replay both look it up on the owner).
            patch.object(owner, "_h_aggregate_op", wraps=owner._h_aggregate_op) as aggregate,
            patch.object(owner, "_compute_write", wraps=owner._compute_write) as write,
        ):
            torch.manual_seed(93)
            output = run(0, value, bias, dropout)
            # A separate live boundary exercises both replay regions when the block goes on.
            boundary = run(1, output)
            forward_rng = _rng()
            terminal = mode == "terminal"
            # The block's last write stays live; replayed outputs are released at its end.
            assert bool(output.untyped_storage().nbytes()) == (not replay or terminal)
            boundary.sum().backward()
            # Projection has its own checkpoint; the mapping API is not replayed.
            assert mappings.call_count == 1
            assert aggregate.call_count == 1 + int(replay)
            assert write.call_count == 1 + int(replay and not terminal)
        assert hooks == (["read", "write"] if model is actual else [])
        gradients = [value.grad, bias.grad] + [p.grad for p in model.parameters()]
        assert all(gradient is not None for gradient in gradients)
        torch.testing.assert_close(_rng(), forward_rng, rtol=0, atol=0)
        results.append((output.detach().clone(), gradients, forward_rng))
    torch.testing.assert_close(results[1], results[0], rtol=0, atol=0)
