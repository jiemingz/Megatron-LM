# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Shared residual-layer layout checks and mHC scheduling with small branch fixtures."""

from contextlib import nullcontext
from unittest.mock import patch

import pytest
import torch
from torch import nn

from megatron.core.activation_policy import ReplayRegion, ResidualBranch
from megatron.core.activation_policy.recompute.checkpointing import CheckpointWithoutOutputManager
from megatron.core.fusions.fused_bias_dropout import get_bias_dropout_add
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.enums import CudaGraphModule
from megatron.core.transformer.hyper_connection import HyperConnectionModule
from megatron.core.transformer.hyper_connection_transformer_layer import (
    HyperConnectionTransformerLayer,
)
from megatron.core.transformer.identity_op import IdentityFuncOp, IdentityOp
from megatron.core.transformer.module import TwoStageAttentionLayer
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.transformer_layer import TransformerLayer, TransformerLayerSubmodules
from megatron.core.transformer.wide_residual_config import WideResidualConfig
from megatron.core.transformer.wide_residual_transformer_layer import WideResidualTransformerLayer
from tests.unit_tests.test_utilities import Utils, install_residual_replay


class _NormContext(nullcontext):
    def group_offload(self, value, **kwargs):
        return value


class _Branch(nn.Module):
    """A small trainable branch; residual scheduling and mHC/WR operations remain real."""

    def __init__(self, config, **kwargs) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        self.projection = nn.Linear(config.hidden_size, config.hidden_size)

    def forward(self, value, **kwargs):
        assert value.shape[-1] == self.hidden_size
        return self.projection(value).sin(), None


class _Norm(nn.LayerNorm):
    def __init__(self, config, hidden_size, eps) -> None:
        super().__init__(hidden_size, eps=eps)


def _offload_interface(enabled, value, tag):
    assert not enabled
    return _NormContext(value)


def _make_layer(stages="both", *, layer_type=None, cuda_graph_impl="none", **submodule_overrides):
    layer_type = layer_type or HyperConnectionTransformerLayer
    is_mhc = issubclass(layer_type, HyperConnectionTransformerLayer)
    is_wide = issubclass(layer_type, WideResidualTransformerLayer)
    config = TransformerConfig(
        num_layers=1,
        hidden_size=8,
        num_attention_heads=2,
        use_cpu_initialization=True,
        enable_mhc_connections=is_mhc,
        wide_residual=WideResidualConfig(num_streams=4) if is_wide else None,
        mhc_num_residual_streams=4,
        mhc_sinkhorn_iterations=3,
        use_fused_mhc=False,
        hidden_dropout=0.25,
        bias_dropout_fusion=False,
        cuda_graph_impl=cuda_graph_impl,
    )
    has_attention, has_mlp = stages in ("both", "attention"), stages in ("both", "mlp")
    stage_specs = dict(
        input_layernorm=_Norm if has_attention else IdentityOp,
        self_attention=_Branch if has_attention else IdentityOp,
        self_attn_bda=get_bias_dropout_add if has_attention else IdentityFuncOp,
        self_attention_hyper_connection=(
            HyperConnectionModule if has_attention and is_mhc else IdentityOp
        ),
        pre_mlp_layernorm=_Norm if has_mlp else IdentityOp,
        mlp=_Branch if has_mlp else IdentityOp,
        mlp_bda=get_bias_dropout_add if has_mlp else IdentityFuncOp,
        mlp_hyper_connection=HyperConnectionModule if has_mlp and is_mhc else IdentityOp,
    )
    submodules = TransformerLayerSubmodules(**(stage_specs | submodule_overrides))
    pg_collection = ProcessGroupCollection()
    pg_collection.tp = pg_collection.pp = None
    # Offloading is disabled; keep its external backend out of these residual tests.
    with patch(
        "megatron.core.transformer.transformer_layer._get_offloading_interface",
        return_value=_offload_interface,
    ):
        layer = layer_type(config, submodules, pg_collection=pg_collection)
    return layer.to("cuda" if torch.cuda.is_available() else "cpu")


def test_dedicated_inherits_dispatch_and_parameter_layout():
    layer = _make_layer()
    for name in (
        "_get_residual_connection_for_self_attention",
        "_get_residual_connection_for_mlp",
        "_is_last_residual_write",
        "_validate_residual_branch_layout",
        "_apply_self_attn_bda_step",
        "_apply_mlp_bda_step",
    ):
        assert getattr(type(layer), name) is getattr(TransformerLayer, name)
    assert {name.split(".")[0] for name in layer.state_dict()} == {
        "input_layernorm",
        "pre_mlp_layernorm",
        "self_attention",
        "mlp",
        "self_attention_hyper_connection",
        "mlp_hyper_connection",
    }


class _LegacyPostMlpLayer(HyperConnectionTransformerLayer):
    """Defines the legacy hook that the base layer would otherwise route the MLP write through."""

    def _forward_post_mlp(self, mlp_output_with_bias, residual):
        raise AssertionError("mHC must not route the MLP write through the legacy hook")


def test_dedicated_layer_clears_the_legacy_post_mlp_hook():
    layer = _make_layer(layer_type=_LegacyPostMlpLayer)
    assert layer._legacy_forward_post_mlp is None
    # A real forward and backward must not reach the legacy hook.
    value = torch.randn(2, 1, 32, device=next(layer.parameters()).device, requires_grad=True)
    output, _ = layer(value, attention_mask=None)
    output.square().sum().backward()
    assert value.grad is not None


@pytest.mark.parametrize("stages", ["attention", "mlp", "both"])
def test_optional_stages_construct_and_run_the_inherited_forward(stages):
    layer = _make_layer(stages)
    owners = []
    for active, owner_name, stage in (
        (stages != "mlp", "self_attention_hyper_connection", "self_attn"),
        (stages != "attention", "mlp_hyper_connection", "mlp"),
    ):
        owner = getattr(layer, owner_name)
        assert (owner is not None) == active
        assert (getattr(layer, f"residual_read_{stage}") is not None) == active
        assert (getattr(layer, f"residual_write_{stage}") is not None) == active
        assert any(key.startswith(owner_name + ".") for key in layer.state_dict()) == active
        if active:
            owners.append(owner)

    calls = []
    for owner in owners:
        owner.register_forward_hook(lambda module, args, output: calls.append(module))
    value = torch.randn(2, 1, 32, device=next(layer.parameters()).device, requires_grad=True)
    output, context = layer(value, attention_mask=None)
    assert output.shape == value.shape and context is None
    assert calls == owners
    output.square().sum().backward()
    assert value.grad is not None
    assert all(parameter.grad is not None for parameter in layer.parameters())

    # Exercise the real module enumeration and hook setup, including absent branches.
    layer = _make_layer(stages, cuda_graph_impl="transformer_engine")
    for scopes in (
        [CudaGraphModule.attn],
        [CudaGraphModule.mlp],
        [CudaGraphModule.attn, CudaGraphModule.mlp],
        [],
    ):
        layer.config.cuda_graph_modules = scopes
        hooked = []
        layer.setup_manual_hooks(lambda: hooked.append)
        for hook, args in layer.te_cuda_graph_manual_hooks:
            hook(*args)
        assert len(hooked) == len(set(hooked))
        for scope, owner in (
            (CudaGraphModule.attn, layer.self_attention_hyper_connection),
            (CudaGraphModule.mlp, layer.mlp_hyper_connection),
        ):
            if owner is not None:
                owner_parameters = {
                    module for module in owner.modules() if list(module.parameters(recurse=False))
                }
                assert owner_parameters
                if not scopes or scope in scopes:
                    assert owner_parameters.issubset(hooked)
                else:
                    assert owner_parameters.isdisjoint(hooked)


@pytest.mark.parametrize(
    "stages,overrides,message",
    [
        (
            "attention",
            {"self_attention_hyper_connection": IdentityOp},
            "^The active self-attention",
        ),
        ("mlp", {"mlp_hyper_connection": IdentityOp}, "^The active MLP"),
        (
            "attention",
            {"mlp_hyper_connection": HyperConnectionModule},
            "inactive MLP branch requires an identity hyper connection",
        ),
        (
            "mlp",
            {"self_attention_hyper_connection": HyperConnectionModule},
            "inactive self-attention branch requires an identity hyper connection",
        ),
        ("neither", {}, "requires an attention or MLP branch"),
    ],
)
def test_optional_stages_reject_inconsistent_specs(stages, overrides, message):
    with pytest.raises(ValueError, match=message):
        _make_layer(stages, **overrides)


class _TwoStageBranch(_Branch, TwoStageAttentionLayer):
    """A branch that advertises two-stage attention, as a plain SelfAttention does."""


def test_attention_only_layers_do_not_advertise_two_stage_attention():
    # mHC layers run only through __call__; they must not advertise the split entry points.
    layer = _make_layer("attention", self_attention=_TwoStageBranch)
    assert TransformerLayer.supports_two_stage_attention(layer)
    assert not layer.supports_two_stage_attention()


@pytest.mark.parametrize(
    "layer_type", [HyperConnectionTransformerLayer, WideResidualTransformerLayer]
)
@pytest.mark.parametrize(
    "stages,overrides,error,message",
    [
        ("attention", {"pre_mlp_layernorm": _Norm}, ValueError, "inactive MLP"),
        ("mlp", {"input_layernorm": _Norm}, ValueError, "inactive self-attention"),
        ("attention", {"mlp_bda": get_bias_dropout_add}, ValueError, "inactive MLP"),
        ("mlp", {"self_attn_bda": get_bias_dropout_add}, ValueError, "inactive self-attention"),
        ("both", {"cross_attention": _Branch}, NotImplementedError, "cross-attention branches"),
        ("both", {"pre_cross_attn_layernorm": _Norm}, ValueError, "identity cross-attention path"),
        (
            "both",
            {"cross_attn_bda": get_bias_dropout_add},
            ValueError,
            "identity cross-attention path",
        ),
    ],
)
def test_residual_layers_share_branch_layout_validation(
    layer_type, stages, overrides, error, message
):
    assert (
        layer_type._validate_residual_branch_layout
        is TransformerLayer._validate_residual_branch_layout
    )
    with pytest.raises(error, match=message):
        _make_layer(stages, layer_type=layer_type, **overrides)


def test_shared_layout_validation_preserves_empty_wr_and_ordinary_norm_paths():
    empty = _make_layer("neither", layer_type=WideResidualTransformerLayer)
    value = torch.randn(2, 1, 32, requires_grad=True)
    output, _ = empty(value, attention_mask=None)
    torch.testing.assert_close(output, value, rtol=0, atol=0)
    assert not list(empty.parameters())
    output.sum().backward()
    torch.testing.assert_close(value.grad, torch.ones_like(value))

    # An ordinary-width layer can legitimately normalize an identity attention branch.
    ordinary = _make_layer("mlp", layer_type=TransformerLayer, input_layernorm=_Norm)
    value = torch.randn(2, 1, 8, device=next(ordinary.parameters()).device, requires_grad=True)
    output, _ = ordinary(value, attention_mask=None)
    assert output.shape == value.shape
    output.square().sum().backward()
    assert value.grad is not None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Output-discard replay needs CUDA")
@pytest.mark.parametrize("stages", ["attention", "mlp", "both"])
@torch.compiler.set_stance("force_eager")
def test_shared_layout_validation_preserves_wr_replay(stages, gpu_model_parallel):
    eager = _make_layer(stages, layer_type=WideResidualTransformerLayer)
    replay = _make_layer(stages, layer_type=WideResidualTransformerLayer)
    replay.load_state_dict(eager.state_dict(), strict=True)
    initial = torch.randn(2, 1, 32, device=next(eager.parameters()).device)
    expected = _forward_backward([eager], initial, None)
    actual = _forward_backward([replay], initial, 1)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def _finalize_with_storage_checks(finalize, manager, boundary, restoration_checks):
    # Pytest must not repr tensor arguments whose storage may have been released.
    __tracebackhide__ = True
    # These fixtures produce fresh checkpoint outputs with no retained-input exemptions.
    # Detached aliases observe the original storage; clones only record expected values.
    managed = [
        output.detach()
        for checkpoint in manager.checkpoints
        for output in checkpoint.outputs
        if isinstance(output, torch.Tensor)
    ]
    assert managed, "Replay must register actual checkpoint outputs"
    assert all(
        output.untyped_storage().nbytes() > 0 for output in managed
    ), "Registered outputs must be live before finalize"
    expected = [output.clone() for output in managed]
    finalize(manager, boundary)
    assert all(
        output.untyped_storage().nbytes() == 0 for output in managed
    ), "Finalize must release every registered output"
    # Compare an int: pytest's assertion rewriting would repr a released boundary tensor.
    boundary_storage_bytes = boundary.untyped_storage().nbytes()
    assert boundary_storage_bytes > 0, "Replay must keep the terminal boundary storage live"
    checked = []

    def check_restored(gradient):
        # This hook follows the production replay hook and guards reads of released storage.
        assert all(
            output.untyped_storage().nbytes() > 0 for output in managed
        ), "Replay must restore registered outputs before the boundary gradient is used"
        torch.testing.assert_close(managed, expected, rtol=0, atol=0)
        checked.append(True)
        return gradient

    boundary.register_hook(check_restored)
    restoration_checks.append(checked)


def _run_layers(layers, value, block_size):
    """Run ``layers`` as a stack; ``block_size`` residual-replays them (``None``: eager)."""
    # The boundary can alias this frame's value after a failed storage check.
    __tracebackhide__ = True
    if block_size is None:
        for layer in layers:
            value, _ = layer(value, attention_mask=None)
        return value, []
    stack = install_residual_replay(layers, block_size=block_size)
    block_outputs, restoration_checks, finalized = [], [], []
    finalize = CheckpointWithoutOutputManager.discard_all_outputs_and_register_unified_recompute

    def checked_finalize(manager, boundary):
        finalized.append(boundary)
        _finalize_with_storage_checks(finalize, manager, boundary, restoration_checks)

    with patch.object(
        CheckpointWithoutOutputManager,
        "discard_all_outputs_and_register_unified_recompute",
        checked_finalize,
    ):
        for index in range(len(layers)):
            value, _ = stack._run_layer(index, value, attention_mask=None)
            block_outputs.append(value)
            if index in stack.replay.block_end_steps:
                assert finalized and finalized[-1] is value, "The block ends on this output"
                assert all(
                    output.untyped_storage().nbytes() == 0 for output in block_outputs[:-1]
                ), "Finalize must release the block's nonterminal layer outputs"
                block_outputs.clear()
    assert not block_outputs, "Every replay block needs a terminal boundary"
    return value, restoration_checks


def _forward_backward(layers, initial, block_size):
    # Caller locals can retain released tensors too; preserve the raw traceback only.
    __tracebackhide__ = True
    torch.manual_seed(812)
    value = initial.detach().clone().requires_grad_()
    output, restoration_checks = _run_layers(layers, value, block_size)
    forward_rng = torch.get_rng_state()
    forward_cuda_rng = torch.cuda.get_rng_state(value.device) if value.is_cuda else None
    # Nonuniform weights expose changes that a uniform sum of residual streams can hide.
    weights = torch.linspace(0.25, 1.25, output.numel(), device=output.device).reshape_as(output)
    (output.square() * weights).sum().backward()
    assert all(
        checks == [True] for checks in restoration_checks
    ), "Every replay block's restoration check must run exactly once"
    torch.testing.assert_close(torch.get_rng_state(), forward_rng, rtol=0, atol=0)
    if value.is_cuda:
        torch.testing.assert_close(
            torch.cuda.get_rng_state(value.device), forward_cuda_rng, rtol=0, atol=0
        )
    gradients = {}
    for index, layer in enumerate(layers):
        for name, parameter in layer.named_parameters():
            assert parameter.grad is not None, (index, name)
            assert torch.isfinite(parameter.grad).all(), (index, name)
            gradients[f"{index}.{name}"] = parameter.grad.detach().clone()
    assert gradients and value.grad is not None
    return output.detach(), value.grad, gradients, forward_rng, forward_cuda_rng


def _assert_mhc_replay_matches_eager(actual, expected):
    """Require exact forward/RNG results and numerical FP32 gradient agreement."""
    torch.testing.assert_close(actual[0], expected[0], rtol=0, atol=0)
    torch.testing.assert_close(actual[3:], expected[3:], rtol=0, atol=0)
    # Checkpointed writes can change the gradient strides entering native Sinkhorn.
    # CUDA FP32 reductions can then round differently; use the real-branch layer
    # test's tolerances for input and parameter gradients. Storage checks stay exact.
    torch.testing.assert_close(actual[1:3], expected[1:3], rtol=1e-5, atol=1e-6)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Output-discard replay needs CUDA")
@pytest.mark.parametrize("stages", ["attention", "mlp", "both"])
@torch.compiler.set_stance("force_eager")
def test_optional_stages_replay_matches_eager_at_a_block_boundary(stages, gpu_model_parallel):
    eager = _make_layer(stages)
    replay = _make_layer(stages)
    replay.load_state_dict(eager.state_dict(), strict=True)
    calls = []
    for stage in ("self_attn", "mlp"):
        adapter = getattr(replay, f"residual_write_{stage}")
        if adapter is not None:
            adapter.register_forward_pre_hook(
                lambda module, args, kwargs: calls.append(kwargs["execution_context"].layer_output),
                with_kwargs=True,
            )
    initial = torch.randn(2, 1, 32, device=next(eager.parameters()).device)
    expected = _forward_backward([eager], initial, None)
    actual = _forward_backward([replay], initial, 1)
    _assert_mhc_replay_matches_eager(actual, expected)
    # Only the layer's last write produces its (live) output.
    assert calls == ([False, True] if stages == "both" else [True])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Output-discard replay needs CUDA")
@pytest.mark.parametrize(
    "stages",
    [("attention", "mlp", "attention"), ("mlp", "attention", "mlp"), ("both", "attention", "mlp")],
)
@pytest.mark.parametrize("block_size", [1, 2])
@torch.compiler.set_stance("force_eager")
def test_optional_stages_share_replay_blocks(stages, block_size, gpu_model_parallel):
    eager = [_make_layer(stage) for stage in stages]
    replay = [_make_layer(stage) for stage in stages]
    for source, target in zip(eager, replay):
        target.load_state_dict(source.state_dict(), strict=True)
    initial = torch.randn(2, 1, 32, device=next(eager[0].parameters()).device)
    expected = _forward_backward(eager, initial, None)
    actual = _forward_backward(replay, initial, block_size)
    _assert_mhc_replay_matches_eager(actual, expected)


@pytest.fixture
def gpu_model_parallel():
    Utils.initialize_model_parallel(1, 1)
    model_parallel_cuda_manual_seed(123)
    yield
    Utils.destroy_model_parallel()


class _Boundary(nn.Module):
    """A live layer output after the mHC layer: ends its replay block."""

    def get_residual_branches_by_layer(self):
        return [[ResidualBranch(write=ReplayRegion(self, self.boundary))]]

    def forward(self, value):
        return self.boundary(value)

    def boundary(self, value):
        return torch.square(value)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Output-discard replay needs CUDA")
@pytest.mark.parametrize(
    "terminal,no_grad",
    [(None, False), (False, False), (True, False), (None, True), (False, True), (True, True)],
)
@torch.compiler.set_stance("force_eager")
def test_dedicated_shared_schedule_preserves_mhc_policy(terminal, no_grad, gpu_model_parallel):
    layer = _make_layer()
    # Sharing adapters must not confuse stages, and owners remain replaceable after construction.
    device = next(layer.parameters()).device
    layer.self_attention_hyper_connection = HyperConnectionModule(layer.config, 2).to(device)
    layer.residual_read_mlp = layer.residual_read_self_attn
    layer.residual_write_mlp = layer.residual_write_self_attn
    layer.config.params_dtype = torch.bfloat16  # Preserve the owner's natural FP32 read dtype.
    owners = [layer.self_attention_hyper_connection, layer.mlp_hyper_connection]
    owner_outputs, calls = [], []

    def owner_hook(module, args, output):
        assert len(output) == 3
        selected = (output[0] + 0.125, output[1], output[2] * 0.75)
        owner_outputs.append(selected)
        return selected

    for owner in owners:
        owner.register_forward_pre_hook(lambda module, args: (args[0] * 2,))
        owner.register_forward_hook(owner_hook)
    for adapter in (layer.residual_read_self_attn, layer.residual_write_self_attn):
        adapter.register_forward_pre_hook(
            lambda module, args, kwargs: calls.append(
                (kwargs["execution_context"].owner, kwargs["execution_context"].layer_output)
            ),
            with_kwargs=True,
        )
    initial = torch.randn(2, 1, 32, device=device, requires_grad=True)
    biases = [torch.randn(8, device=device, requires_grad=True) for _ in owners]

    def run_schedule():
        value = initial
        with torch.no_grad() if no_grad else nullcontext():
            for index, (read, write) in enumerate(
                (
                    (layer._run_input_layernorm, layer._apply_self_attn_bda_step),
                    (layer._pre_mlp_layernorm_and_residual, layer._apply_mlp_bda_step),
                )
            ):
                branch, residual, state = read(value)
                assert residual is value and state[0] is value and branch.dtype == value.dtype
                assert state[1] is owner_outputs[-1][1] and state[2] is owner_outputs[-1][2]
                branch_module = layer.self_attention if index == 0 else layer.mlp
                branch_output, _ = branch_module(branch)
                value = write((branch_output, biases[index]), residual, state)
        return value

    # The layer is its block's end (terminal), or a later live boundary ends the block.
    layers = [layer] if terminal else [layer, _Boundary()]
    stack = (
        install_residual_replay(layers, block_size=len(layers)) if terminal is not None else None
    )
    stack_steps = stack.steps if stack is not None else None
    if stack_steps is not None:
        stack_steps[0] = run_schedule
    recorded = []
    finalize = CheckpointWithoutOutputManager.discard_all_outputs_and_register_unified_recompute

    def record(manager, boundary):
        recorded.append(len(manager.checkpoints))
        return finalize(manager, boundary)

    with patch.object(
        CheckpointWithoutOutputManager, "discard_all_outputs_and_register_unified_recompute", record
    ):
        value = run_schedule() if stack is None else stack._run_layer(0)
        boundary = value.square() if terminal in (None, True) else stack._run_layer(1, value)
    assert [owner for owner, _ in calls] == [owners[0], owners[0], owners[1], owners[1]]
    # Only the layer's last write produces its output.
    assert [layer_output for _, layer_output in calls] == [False, False, False, True]
    if stack is not None:
        # Each stage's aggregation and norm (with grad) and every write but a terminal one.
        expected = (0 if no_grad else 4) + 2 - int(terminal)
        assert recorded == [expected]
        assert bool(value.untyped_storage().nbytes()) == terminal
    else:
        assert recorded == []
    boundary.sum().backward()
    assert initial.grad is not None and all(bias.grad is not None for bias in biases)
    assert all((parameter.grad is None) == no_grad for parameter in layer.parameters())
    assert len(calls) == 4 and len(owner_outputs) == 2
