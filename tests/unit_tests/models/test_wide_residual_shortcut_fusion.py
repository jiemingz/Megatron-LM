# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Model construction and A/B handoffs with real wide-residual Mamba layers."""

import copy
from contextlib import nullcontext

import pytest
import torch
from torch import nn

from megatron.core.fusions import wide_residual_shortcut as kernels
from megatron.core.models.hybrid.hybrid_block import HybridStack, HybridStackSubmodules
from megatron.core.models.hybrid.hybrid_layer_allocation import Symbols
from megatron.core.models.hybrid.layers.shortcut_fusion import (
    FusedShortcutMoEBlock,
    PreparedShortcutInput,
)
from megatron.core.models.hybrid.shortcut_block import group_layers_into_shortcut_blocks
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.ssm.mamba_layer import MambaLayerSubmodules
from megatron.core.ssm.wide_residual_mamba_layer import WideResidualMambaLayer
from megatron.core.transformer.module import TwoStageAttentionLayer
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.wide_residual_config import WideResidualConfig
from megatron.core.transformer.wide_residual_layer import StreamwiseSigmoidWideResidualConnection
from tests.unit_tests.test_utilities import Utils


@pytest.fixture(autouse=True)
def bind_local_cuda_device():
    # Bind before the fake module tests create the process-wide overlap stream.
    torch.cuda.set_device(Utils.local_rank % torch.cuda.device_count())


class _Norm(nn.RMSNorm):
    def __init__(self, config, hidden_size, eps):
        super().__init__(hidden_size, eps=eps, device="cuda", dtype=torch.bfloat16)


class _Mixer(nn.Module, TwoStageAttentionLayer):
    def __init__(self, config, d_model, **kwargs):
        super().__init__()
        self.input = nn.Linear(d_model, 32, bias=False, device="cuda", dtype=torch.bfloat16)
        self.output = nn.Linear(32, d_model, bias=False, device="cuda", dtype=torch.bfloat16)

    def forward(self, x, **kwargs):
        return self.forward_post_core_attn(self.forward_pre_attn_and_core_attn(x))

    def supports_two_stage_attention(self):
        return True

    def forward_pre_attn_and_core_attn(self, x, **kwargs):
        return torch.tanh(self.input(x))

    def forward_post_core_attn(self, x):
        return self.output(x), None


class _Experts(nn.Module):
    use_shared_expert = True
    shared_expert_overlap = False
    tp_group = None

    def __init__(self, hidden):
        super().__init__()
        self.router = nn.Linear(hidden, 1, bias=False, device="cuda", dtype=torch.bfloat16)
        self.shared = nn.Linear(hidden, hidden, bias=False, device="cuda", dtype=torch.bfloat16)
        self.routed = nn.Linear(hidden, hidden, bias=False, device="cuda", dtype=torch.bfloat16)
        self.combine_calls = 0

    def route(self, x, padding_mask=None, input_ids=None, packed_seq_params=None):
        return self.router(x).sigmoid(), None

    def preprocess(self, x, probs, routing, padding_mask=None):
        return x, probs

    def dispatch(self, x, probs):
        return x, probs

    def routed_experts_compute(self, x, probs):
        return self.routed(x) * probs, None

    def combine(self, x):
        return x

    def shared_experts_compute(self, x):
        return self.shared(x)

    def postprocess(self, x, shared):
        self.combine_calls += 1
        return x if shared is None else x + shared


class _MoE(nn.Module):
    supports_wide_residual_connections = True
    is_mtp_layer = False
    is_last_layer = False

    def __init__(self, config, layer_number, pg=None, pg_collection=None, **kwargs):
        pg = pg_collection if pg_collection is not None else pg
        super().__init__()
        self.config, self.layer_number = config, layer_number
        self.residual_connection_mlp = StreamwiseSigmoidWideResidualConnection(
            config, layer_number, "mlp", pg
        )
        self.pre_mlp_layernorm = _Norm(config, config.hidden_size, 1e-5)
        self.mlp = _Experts(config.hidden_size)

    def _get_mlp_residual_connection(self):
        return self.residual_connection_mlp

    def _pre_mlp_layernorm_and_residual(self, x):
        x, state = self.residual_connection_mlp(
            x,
            operation="read",
            fp32_residual_connection=self.config.fp32_residual_connection,
            branch_input_dtype=self.config.params_dtype,
        )
        return self.pre_mlp_layernorm(x), state[0], state

    def _maybe_unflatten_for_moe(self, x, padding_mask=None, **kwargs):
        return x, padding_mask, None, None

    def _maybe_reflatten_from_moe(self, x, *args):
        return x

    def _apply_mlp_bda_step(self, output, residual, state):
        return self.residual_connection_mlp(
            output, operation="write", state=state, dropout_probability=0.0, training=self.training
        )


def make_config(streams=3, enabled=True, **kwargs):
    options = dict(
        num_layers=4,
        hidden_size=128,
        num_attention_heads=4,
        normalization="RMSNorm",
        layernorm_epsilon=1e-5,
        params_dtype=torch.bfloat16,
        bf16=True,
        fp32_residual_connection=True,
        hidden_dropout=0.0,
        attention_dropout=0.0,
        num_moe_experts=1,
        moe_shared_expert_intermediate_size=128,
        add_bias_linear=False,
        moe_shortcut_connection=True,
        wide_residual_shortcut_moe_kernel_fusion=enabled,
        wide_residual=WideResidualConfig(num_streams=streams, learned_retention=True),
    )
    options.update(kwargs)
    return TransformerConfig(**options)


def make_blocks(streams=3, enabled=True, retention=True, pairs=2):
    config = make_config(
        streams,
        enabled,
        wide_residual=WideResidualConfig(num_streams=streams, learned_retention=retention),
    )
    pg = ProcessGroupCollection(tp=None)
    physical = nn.ModuleList()
    for index in range(1, 2 * pairs, 2):
        compute = WideResidualMambaLayer(
            config,
            MambaLayerSubmodules(mixer=_Mixer, norm=_Norm),
            layer_number=index,
            pg_collection=pg,
        )
        physical.extend([compute, _MoE(config, index + 1, pg)])
    return group_layers_into_shortcut_blocks(
        physical, [Symbols.GDN, Symbols.MOE] * pairs, config
    ).cuda()


def run(block, x, next_block=None):
    kwargs = (
        dict(next_shortcut_block=next_block) if isinstance(block, FusedShortcutMoEBlock) else {}
    )
    return block(x, None, None, None, None, None, None, lambda *a: nullcontext(), **kwargs)


def explicit_pair(blocks, residual):
    prepared = None
    for i, block in enumerate(blocks):
        compute, moe = block.compute_layer, block.moe_layer
        if prepared is None:
            shortcut = block._read_shortcut_hidden(residual, recompute_context=None)
            route, probs = block._moe_router_preprocess(shortcut)
            compute_input, _, _, _ = compute._prepare_mixer_state(residual)
        else:
            compute_input, route = prepared
            probs, _ = moe.mlp.route(route)
        routed, _ = moe.mlp.routed_experts_compute(route, probs)
        update, _ = compute.mixer.forward_post_core_attn(
            compute.mixer.forward_pre_attn_and_core_attn(compute_input)
        )
        producer, consumer = compute.residual_connection, moe.residual_connection_mlp
        residual, shared_input = kernels.write_read_norm(
            residual,
            update,
            producer.write_map(return_logits=True),
            block._retention_logits(producer),
            consumer.read_map(return_logits=True),
            moe.pre_mlp_layernorm.weight,
            num_streams=block.config.wide_residual.num_streams,
        )
        shared = moe.mlp.shared_experts_compute(shared_input)
        if i + 1 < len(blocks):
            nxt = blocks[i + 1]
            residual, compute_input, shortcut = kernels.boundary_norm(
                residual,
                routed,
                shared,
                consumer.write_map(return_logits=True),
                block._retention_logits(consumer),
                (
                    nxt.compute_layer.residual_connection.read_map(return_logits=True),
                    nxt.shortcut_residual_read.read_map(return_logits=True),
                ),
                (nxt.compute_layer.norm.weight, nxt.shortcut_pre_mlp_layernorm.weight),
                num_streams=block.config.wide_residual.num_streams,
            )
            prepared = compute_input, shortcut
        else:
            residual = block._postprocess(residual, routed, shared, mlp_state=(residual,))
    return residual


@pytest.mark.parametrize("streams", range(2, 7))
@pytest.mark.parametrize("retention", [False, True])
@pytest.mark.parametrize("overlap", [False, True])
def test_model_schedule_and_gradients(streams, retention, overlap):
    blocks = make_blocks(streams, retention=retention)
    explicit = copy.deepcopy(blocks)
    if overlap:
        for block in blocks:
            block.overlap_mode = True
            block.route_ready_event = torch.cuda.Event()
    r = torch.randn(1024, 1, streams * 128, device="cuda", requires_grad=True)
    other = r.detach().clone().requires_grad_(True)
    handoff = run(blocks[0], r, blocks[1])
    assert isinstance(handoff, PreparedShortcutInput)
    actual = run(blocks[1], handoff)
    expected = explicit_pair(explicit, other)
    assert isinstance(actual, torch.Tensor) and actual.dtype == torch.float32
    assert torch.equal(actual, expected)
    incoming = torch.randn_like(actual)
    actual.backward(incoming)
    expected.backward(incoming)
    torch.testing.assert_close(r.grad, other.grad, rtol=0, atol=0)
    for (name, a), (other_name, b) in zip(blocks.named_parameters(), explicit.named_parameters()):
        assert name == other_name
        if a.grad is None or b.grad is None:
            assert a.grad is None and b.grad is None
        else:
            torch.testing.assert_close(a.grad, b.grad, rtol=0, atol=0, msg=name)


def test_state_keys_and_independent_handoffs():
    blocks = make_blocks()
    ordinary = make_blocks(enabled=False)
    ordinary.load_state_dict(blocks.state_dict(), strict=True)
    assert set(ordinary.state_dict()) == set(blocks.state_dict())
    assert all(not isinstance(b, FusedShortcutMoEBlock) for b in ordinary)
    xs = [torch.randn(1024, 1, 384, device="cuda", requires_grad=True) for _ in range(2)]
    prepared = [run(blocks[0], x, blocks[1]) for x in xs]
    assert prepared[0].residual.data_ptr() != prepared[1].residual.data_ptr()
    for value in reversed(prepared):
        run(blocks[1], value).sum().backward()
    assert all(x.grad is not None and torch.isfinite(x.grad).all() for x in xs)
    with pytest.raises(ValueError, match="another block"):
        run(blocks[0], prepared[0])


@pytest.mark.parametrize("interval", [None, 1, 2, 3, 4, 16])
@pytest.mark.parametrize("unpaired", [False, True])
def test_hybrid_stack_runs_both_boundaries(interval, unpaired):
    Utils.initialize_model_parallel(tensor_model_parallel_size=1)
    try:
        pattern = (
            [Symbols.MAMBA, Symbols.MAMBA, Symbols.MOE] * 2
            if unpaired
            else [Symbols.MAMBA, Symbols.MOE] * 2
        )
        config = make_config(num_layers=len(pattern), residual_stream_recompute_num_layers=interval)
        spec = ModuleSpec(
            module=WideResidualMambaLayer, submodules=MambaLayerSubmodules(mixer=_Mixer, norm=_Norm)
        )
        stack = HybridStack(
            config,
            HybridStackSubmodules(mamba_layer=spec, moe_layer=_MoE),
            layer_type_list=pattern,
            pre_process=True,
            post_process=False,
            post_layer_norm=False,
            pg_collection=ProcessGroupCollection.use_mpu_process_groups(),
        ).cuda()
        x = torch.randn(1024, 1, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        result = stack(x, attention_mask=None)
        assert result.shape == (1024, 1, 384) and result.dtype == torch.float32
        # Two live forwards must own independent replay managers and storages.
        other = x.detach().clone().requires_grad_(True)
        second = stack(other, attention_mask=None)
        second.sum().backward()
        result.sum().backward()
        assert x.grad is not None and torch.isfinite(x.grad).all()
        assert other.grad is not None and torch.isfinite(other.grad).all()
        torch.testing.assert_close(x.grad, other.grad, rtol=0, atol=0)
    finally:
        Utils.destroy_model_parallel()


@pytest.mark.parametrize(
    "overrides,match",
    [
        (dict(moe_shortcut_connection=False), "shortcut MoE"),
        (dict(wide_residual=None), "wide_residual"),
        (dict(fp32_residual_connection=False), "fp32_residual_connection"),
        (dict(wide_residual=WideResidualConfig(num_streams=7)), "num_streams"),
        (dict(hidden_size=132, num_attention_heads=4), "divisible by 32"),
        (dict(hidden_dropout=0.1), "hidden_dropout"),
        (dict(add_bias_linear=True), "add_bias_linear"),
        (dict(moe_shortcut_post_norm=True), "moe_shortcut_post_norm"),
        (dict(normalization="LayerNorm"), "RMSNorm"),
        (dict(moe_shared_expert_intermediate_size=None), "shared expert"),
        (dict(recompute_granularity="full"), "recompute_granularity"),
    ],
)
def test_invalid_configurations(overrides, match):
    with pytest.raises(ValueError, match=match):
        make_config(**overrides)


def test_cli_flag_defaults_off_and_maps_to_config_field():
    import argparse
    from dataclasses import fields

    from megatron.training.arguments import _add_wide_residual_args

    parser = _add_wide_residual_args(argparse.ArgumentParser())
    name = "wide_residual_shortcut_moe_kernel_fusion"
    assert not getattr(parser.parse_args([]), name)
    parsed = parser.parse_args(
        ["--wide-residual-shortcut-moe-kernel-fusion", "--wide-residual", "3"]
    )
    assert getattr(parsed, name)
    assert name in {field.name for field in fields(TransformerConfig)}
    assert parsed.wide_residual_num_streams == 3


@pytest.mark.parametrize("dispatcher", ["allgather", "alltoall"])
@pytest.mark.parametrize("grouped", [False, True])
def test_real_moe_consumers_train_step(dispatcher, grouped):
    _real_moe_train_step(dispatcher, grouped)


def _real_moe_train_step(
    dispatcher, grouped, hidden=128, tokens=1024, require_cooperative=False, config_overrides=None
):
    from megatron.core.extensions.transformer_engine import TENorm
    from megatron.core.fusions.fused_bias_dropout import get_bias_dropout_add
    from megatron.core.models.gpt.moe_module_specs import get_moe_module_spec
    from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
    from megatron.core.transformer.transformer_layer import TransformerLayerSubmodules
    from megatron.core.transformer.wide_residual_layer import WideResidualTransformerLayer

    ep_size = 2 if Utils.world_size % 2 == 0 else 1
    Utils.initialize_model_parallel(
        tensor_model_parallel_size=1, expert_model_parallel_size=ep_size
    )
    try:
        model_parallel_cuda_manual_seed(123)
        config = make_config(
            hidden_size=hidden,
            num_moe_experts=2,
            moe_router_topk=2,
            moe_token_dispatcher_type=dispatcher,
            moe_grouped_gemm=grouped,
            expert_model_parallel_size=ep_size,
            gradient_accumulation_fusion=False,
            **(config_overrides or {}),
        )
        compute = ModuleSpec(
            module=WideResidualMambaLayer, submodules=MambaLayerSubmodules(mixer=_Mixer, norm=_Norm)
        )
        moe = ModuleSpec(
            module=WideResidualTransformerLayer,
            submodules=TransformerLayerSubmodules(
                pre_mlp_layernorm=TENorm,
                mlp=get_moe_module_spec(use_te=True, num_experts=2, moe_grouped_gemm=grouped),
                mlp_bda=get_bias_dropout_add,
            ),
        )
        stack = HybridStack(
            config,
            HybridStackSubmodules(mamba_layer=compute, moe_layer=moe),
            layer_type_list=[Symbols.MAMBA, Symbols.MOE, Symbols.MAMBA, Symbols.MOE],
            pre_process=True,
            post_process=False,
            post_layer_norm=False,
            pg_collection=ProcessGroupCollection.use_mpu_process_groups(),
        ).cuda()
        optimizer = torch.optim.SGD(stack.parameters(), lr=0.01)
        updated_parameter = stack.layers[0].compute_layer.residual_connection.write_map.logit
        before = updated_parameter.detach().clone()
        x = torch.randn(tokens, 1, hidden, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        from unittest.mock import patch

        from megatron.core.fusions.wide_residual_shortcut import (
            api,
            reconstruction_cooperative_native,
        )

        original_backward = api._backward
        selected = []

        def record_backend(*args, **kwargs):
            out = original_backward(*args, **kwargs)
            selected.append(isinstance(out[2], reconstruction_cooperative_native.Workspace))
            return out

        with patch.object(api, "_backward", record_backend):
            result = stack(x, attention_mask=None)
            result.square().mean().backward()
        if require_cooperative:
            assert any(selected), "production model did not select the S=3 cooperative kernel"
        for parameter in stack.parameters():
            if parameter.grad is not None:
                assert torch.isfinite(parameter.grad).all()
        optimizer.step()
        assert x.grad is not None and torch.isfinite(x.grad).all()
        assert stack.layers[0].moe_layer.mlp.router.weight.grad is not None
        assert not torch.equal(before, updated_parameter)
    finally:
        Utils.destroy_model_parallel()


def test_reject_non_hybrid_stack():
    from megatron.core.transformer.transformer_block import TransformerBlock

    with pytest.raises(ValueError, match="requires HybridStack"):
        TransformerBlock(make_config(), spec=None)


def test_reject_missing_shortcut_pairs():
    with pytest.raises(ValueError, match="adjacent Mamba/GDN and MoE pair"):
        group_layers_into_shortcut_blocks(
            nn.ModuleList([nn.Identity()]), [Symbols.MAMBA], make_config()
        )


def test_reject_unsupported_compute_layer():
    with pytest.raises(ValueError, match="wide-residual Mamba/GDN predecessor"):
        FusedShortcutMoEBlock(object(), object(), overlap_a2a=False)


def test_cli_fusion_flag_cannot_be_discarded_by_yaml(monkeypatch):
    import sys

    from megatron.training.arguments import parse_args

    monkeypatch.setattr(
        sys,
        "argv",
        ["train", "--wide-residual-shortcut-moe-kernel-fusion", "--yaml-cfg", "unused.yaml"],
    )
    with pytest.raises(ValueError, match="cannot be combined"):
        parse_args()


@pytest.mark.parametrize("dispatcher", ["allgather", "alltoall"])
@pytest.mark.parametrize("interval", [None, 16])
def test_production_cooperative_moe_train_step(dispatcher, interval):
    from megatron.core.fusions.wide_residual_shortcut.reconstruction_dispatch import (
        qualified_device,
    )

    if not qualified_device(torch.cuda.get_device_properties(torch.cuda.current_device())):
        pytest.skip("qualified GB300 cooperative kernel")
    _real_moe_train_step(
        dispatcher,
        True,
        hidden=4096,
        tokens=8192,
        require_cooperative=True,
        config_overrides=dict(residual_stream_recompute_num_layers=interval),
    )


def test_static_fusion_spec_selection_preserves_incumbents():
    from megatron.core.extensions.transformer_engine import (
        TEColumnParallelLinear,
        TELayerNormColumnParallelLinear,
        TENorm,
    )
    from megatron.core.models.hybrid import hybrid_layer_specs as specs
    from megatron.core.models.hybrid.layers.shortcut_fusion import resolve_fusion_submodules

    for original in (
        specs.hybrid_stack_spec,
        specs.wide_residual_hybrid_stack_spec,
        specs.gated_delta_product_stack_spec,
        specs.wide_residual_gated_delta_product_stack_spec,
    ):
        selected = resolve_fusion_submodules(original.submodules)
        assert selected is not original.submodules
        for field, branch, norm in (
            ("mamba_layer", "mixer", "norm"),
            ("gdn_layer", "self_attention", "input_layernorm"),
            ("gdn2_layer", "self_attention", "input_layernorm"),
        ):
            old = getattr(original.submodules, field).submodules
            new = getattr(selected, field).submodules
            assert getattr(old, branch).submodules.in_proj is TELayerNormColumnParallelLinear
            assert getattr(new, branch).submodules.in_proj is TEColumnParallelLinear
            assert getattr(new, norm) is TENorm
            assert new.sharded_state_dict_keys_map[norm + "."] == branch + ".in_proj.layer_norm_"
    custom = HybridStackSubmodules()
    assert resolve_fusion_submodules(custom) is custom


@pytest.mark.parametrize("variant", ["gdn", "gdn2", "mamba", "gdp"])
def test_builtin_specs_train_step(variant):
    from megatron.core.models.hybrid.hybrid_layer_specs import (
        wide_residual_gated_delta_product_stack_spec,
        wide_residual_hybrid_stack_spec,
    )
    from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed

    Utils.initialize_model_parallel(tensor_model_parallel_size=1)
    try:
        model_parallel_cuda_manual_seed(123)
        config = make_config(
            experimental_attention_variant=(variant if variant in ("gdn", "gdn2") else None),
            activation_func=torch.nn.functional.silu,
            linear_attention_freq=1,
            mamba_num_groups=1,
            mamba_num_heads=4,
            moe_router_pre_softmax=True,
            linear_key_head_dim=32,
            linear_value_head_dim=32,
            linear_num_key_heads=4,
            linear_num_value_heads=4,
            linear_conv_kernel_dim=4,
            moe_router_topk=1,
            gradient_accumulation_fusion=False,
        )
        spec = (
            wide_residual_gated_delta_product_stack_spec
            if variant == "gdp"
            else wide_residual_hybrid_stack_spec
        )
        symbol = Symbols.GDN if variant in ("gdn", "gdn2") else Symbols.MAMBA
        stack = HybridStack(
            config,
            spec.submodules,
            layer_type_list=[symbol, Symbols.MOE, symbol, Symbols.MOE],
            pre_process=True,
            post_process=False,
            post_layer_norm=False,
            pg_collection=ProcessGroupCollection.use_mpu_process_groups(),
        ).cuda()
        x = torch.randn(1024, 1, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        result = stack(x, attention_mask=None)
        result.square().mean().backward()
        assert x.grad is not None and torch.isfinite(x.grad).all()
        for block in stack.layers:
            assert isinstance(block, FusedShortcutMoEBlock)
            assert block._compute_norm().weight.grad is not None
            assert block._compute_connection().read_map.logit.grad is not None
        assert all(torch.isfinite(p.grad).all() for p in stack.parameters() if p.grad is not None)
        # Splitting the norm preserves the canonical distributed checkpoint key.
        state = stack.layers[0].compute_layer.sharded_state_dict(
            metadata={"dp_cp_group": stack.pg_collection.dp_cp}
        )
        if symbol == Symbols.GDN:
            norm = state["input_layernorm.weight"]
            assert norm.key == "self_attention.in_proj.layer_norm_weight"
        else:
            norm = state["norm.weight"]
            assert norm.key == "mixer.in_proj.layer_norm_weight"
    finally:
        Utils.destroy_model_parallel()


@pytest.mark.parametrize("dispatcher", ["allgather", "alltoall"])
def test_latent_overlap_and_selective_recompute(dispatcher):
    _real_moe_train_step(
        dispatcher,
        True,
        config_overrides=dict(
            moe_latent_size=64,
            moe_use_norm_before_up_proj=True,
            moe_shortcut_parallel=True,
            recompute_granularity="selective",
            recompute_modules=["moe_act"],
        ),
    )


@pytest.mark.parametrize("interval", [None, 16])
def test_mtp_uses_ordinary_shortcut_with_fused_decoder(interval):
    from megatron.core.models.hybrid.hybrid_layer_specs import wide_residual_hybrid_stack_spec
    from megatron.core.models.hybrid.hybrid_model import HybridModel
    from megatron.core.models.hybrid.shortcut_block import ShortcutMoEBlock
    from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed

    Utils.initialize_model_parallel(tensor_model_parallel_size=1)
    try:
        model_parallel_cuda_manual_seed(123)
        config = make_config(
            residual_stream_recompute_num_layers=interval,
            mtp_num_layers=2,
            mtp_use_repeated_layer=True,
            mtp_loss_scaling_factor=0.1,
            moe_router_topk=1,
            moe_router_pre_softmax=True,
            gradient_accumulation_fusion=False,
            moe_shortcut_parallel=True,
        )
        model = HybridModel(
            config=config,
            hybrid_stack_spec=wide_residual_hybrid_stack_spec,
            vocab_size=128,
            max_sequence_length=1024,
            hybrid_layer_pattern="*E*E/*E/*E",
            parallel_output=True,
            position_embedding_type="none",
        ).cuda()
        tokens = torch.randint(128, (1, 1024), device="cuda")
        positions = torch.arange(1024, device="cuda").unsqueeze(0)
        logits = model(input_ids=tokens, position_ids=positions, attention_mask=None)
        logits.float().square().mean().backward()
        blocks = [m for m in model.modules() if isinstance(m, ShortcutMoEBlock)]
        assert any(isinstance(m, FusedShortcutMoEBlock) for m in blocks)
        mtp_blocks = [m for m in blocks if m.is_mtp_layer]
        assert mtp_blocks and all(type(m) is ShortcutMoEBlock for m in mtp_blocks)
        assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    finally:
        Utils.destroy_model_parallel()


def test_bf16_module_conversion_casts_controller_logits_for_kernels():
    blocks = make_blocks().bfloat16()
    assert blocks[0]._compute_connection().write_map.logit.dtype == torch.bfloat16
    residual = torch.randn(1024, 1, 384, device="cuda", requires_grad=True)
    output = run(blocks[1], run(blocks[0], residual, blocks[1]))
    output.square().mean().backward()
    for parameter in blocks.parameters():
        if parameter.grad is not None:
            assert torch.isfinite(parameter.grad).all()
    assert residual.grad is not None and torch.isfinite(residual.grad).all()


def test_fusion_allows_expert_leaf_activation_offloading():
    config = make_config(
        fine_grained_activation_offloading=True,
        offload_modules=["fused_group_mlp"],
        use_transformer_engine_op_fuser=True,
    )
    assert config.wide_residual_shortcut_moe_kernel_fusion


@pytest.mark.parametrize("modules", [["expert_fc1"], ["moe_act"], ["fused_group_mlp", "moe_act"]])
def test_fusion_rejects_unqualified_activation_offloading(modules):
    with pytest.raises(ValueError, match="fine_grained_activation_offloading"):
        make_config(
            fine_grained_activation_offloading=True,
            offload_modules=modules,
            use_transformer_engine_op_fuser=True,
        )


@pytest.mark.parametrize("interval", [None, 1, 2, 3, 4, 16])
@pytest.mark.parametrize("streams", range(2, 7))
@pytest.mark.parametrize("retention", [False, True])
def test_fused_residual_inverse_storage_and_gradients(interval, streams, retention, monkeypatch):
    """Keep only anchors and hand off inverse outputs, with zero forward replay."""
    from megatron.core.fusions.wide_residual_shortcut.residual_inverse import residual_inverse_scope

    torch.manual_seed(1234)
    blocks = make_blocks(pairs=10, streams=streams, retention=retention)
    explicit = copy.deepcopy(blocks)
    config = blocks[0].config
    config.residual_stream_recompute_num_layers = interval
    residuals = []
    calls = []
    snapshots = []
    for name in ("write_read_norm", "boundary_norm"):
        original = getattr(kernels, name)

        def record(*args, _original=original, _name=name, **kwargs):
            result = _original(*args, **kwargs)
            residuals.append((_name, result[0]))
            snapshots.append(result[0].detach().clone())
            calls.append(_name)
            return result

        monkeypatch.setattr(kernels, name, record)

    def execute(chain, x):
        with residual_inverse_scope(chain[0].config, True) as manager:
            for i, block in enumerate(chain):
                if manager is not None:
                    manager.begin_layer(2)
                x = run(block, x, chain[i + 1] if i + 1 < len(chain) else None)
                if manager is not None:
                    manager.finish_layer(2)
        return x

    x = torch.randn(1024, 1, streams * 128, device="cuda", requires_grad=True)
    other = x.detach().clone().requires_grad_(True)
    actual = execute(blocks, x)
    original_calls = len(calls)
    live = [i for i, (_, value) in enumerate(residuals) if value.untyped_storage().nbytes()]
    pairs_per_block = max(1, (interval or 1) // 2)
    expected_live = (
        list(range(19))
        if interval is None or interval == 1
        else [2 * i + 1 for i in range(9) if (i + 1) % pairs_per_block == 0]
    )
    assert live == expected_live
    expected = execute(explicit, other)
    assert torch.equal(actual, expected)
    incoming = torch.randn_like(actual)
    before_backward = len(calls)
    actual.backward(incoming)
    assert len(calls) == before_backward, "inverse handoff must not replay A/B forward"
    assert original_calls == 19
    expected.backward(incoming)

    errors = []

    def close_gradient(a, b, name):
        # Inversion cannot recover rounding discarded by the FP32 forward.
        # Bound accumulated error relative to each complete gradient, including
        # near-zero entries where elementwise relative error is misleading.
        if interval is None or interval == 1:
            torch.testing.assert_close(a, b, rtol=0, atol=0, msg=name)
        else:
            error = (a.float() - b.float()).norm()
            scale = b.float().norm().clamp_min(1e-6)
            relative = float(error / scale)
            errors.append((relative, name))
            if "read_map.logit" in name:
                # Nearly cancelling read-controller sums can be smaller than
                # BF16 rounding noise in the branch gradients. Bound absolute
                # error as well as relative error, and check padding exactly.
                torch.testing.assert_close(a, b, rtol=0.01, atol=0.05, msg=name)
                assert torch.equal(a[streams:], b[streams:]), name
            else:
                assert error < 0.005 * (1.0 + scale), (name, relative)

    close_gradient(x.grad, other.grad, "input")
    for (name, a), (_, b) in zip(blocks.named_parameters(), explicit.named_parameters()):
        if a.grad is None or b.grad is None:
            assert a.grad is None and b.grad is None, name
        else:
            close_gradient(a.grad, b.grad, name)

    reconstruction_errors = []
    for (_, value), snapshot in zip(residuals[:original_calls], snapshots[:original_calls]):
        relative = (value.detach() - snapshot).norm() / snapshot.norm().clamp_min(1e-6)
        reconstruction_errors.append(float(relative))
        assert relative < 1e-6
    print(
        "INVERSE_ERROR",
        interval,
        streams,
        retention,
        "max_residual_relative_l2",
        max(reconstruction_errors),
        "max_gradient_relative_l2",
        max(errors, default=(0.0, "exact")),
    )


@pytest.mark.parametrize("interval", [1, 4, 16])
def test_fusion_accepts_explicit_residual_interval_without_selective_flags(interval):
    config = make_config(residual_stream_recompute_num_layers=interval)
    assert config.residual_stream_recompute_num_layers == interval
    from megatron.core.transformer.residual_recompute import residual_stream_recompute_enabled

    # The legacy replay planner is disabled; the inverse owner reads the interval.
    assert not residual_stream_recompute_enabled(config, True)
    from megatron.core.fusions.wide_residual_shortcut.residual_inverse import residual_inverse_scope

    with residual_inverse_scope(config, True) as manager:
        assert (manager is not None) == (interval > 1)
    assert not residual_stream_recompute_enabled(config, False)
    with torch.no_grad():
        assert not residual_stream_recompute_enabled(config, True)


@pytest.mark.parametrize("interval", [0, -1, True, 1.5])
def test_fusion_rejects_invalid_residual_intervals(interval):
    with pytest.raises(ValueError, match="positive integer"):
        make_config(residual_stream_recompute_num_layers=interval)


@pytest.mark.parametrize("interval", [4, 16])
@pytest.mark.parametrize("dispatcher", ["allgather", "alltoall"])
def test_real_moe_residual_inverse_chain(interval, dispatcher):
    _real_moe_train_step(
        dispatcher,
        True,
        config_overrides=dict(
            moe_latent_size=64,
            moe_use_norm_before_up_proj=True,
            moe_shortcut_parallel=True,
            recompute_granularity="selective",
            recompute_modules=["moe_act"],
            residual_stream_recompute_num_layers=interval,
        ),
    )


@pytest.mark.parametrize("selected", [False, True])
def test_fused_default_keeps_every_residual_even_with_residual_stream_selected(selected):
    from megatron.core.transformer.residual_recompute import residual_stream_recompute_enabled

    config = make_config(
        **(
            dict(recompute_granularity="selective", recompute_modules=["residual_stream"])
            if selected
            else {}
        )
    )
    assert config.residual_stream_recompute_num_layers is None
    assert not residual_stream_recompute_enabled(config, True)


@pytest.mark.parametrize("interval", [2, 16])
def test_inverse_scope_is_call_local_and_resets_after_exception(interval):
    from megatron.core.fusions.wide_residual_shortcut.residual_inverse import (
        current_manager,
        residual_inverse_scope,
    )

    config = make_config(residual_stream_recompute_num_layers=interval)
    assert current_manager() is None
    with pytest.raises(RuntimeError, match="test exception"):
        with residual_inverse_scope(config, True) as outer:
            assert current_manager() is outer
            with residual_inverse_scope(config, True) as inner:
                assert inner is not outer and current_manager() is inner
            assert current_manager() is outer
            raise RuntimeError("test exception")
    assert current_manager() is None


def test_inverse_rejects_partial_storage_and_disconnected_write_chains():
    from megatron.core.fusions.wide_residual_shortcut.residual_inverse import ResidualInverseManager

    manager = ResidualInverseManager(16)
    x = torch.randn(1024, 1, 384, device="cuda")
    y = torch.empty_like(x)
    manager.register(x, y)
    with pytest.raises(ValueError, match="connected chain"):
        manager.register(x.clone(), torch.empty_like(x))
    with pytest.raises(ValueError, match="full-storage"):
        ResidualInverseManager(16).register(x[:512], torch.empty_like(x[:512]))


@pytest.mark.parametrize("interval", [2, 16])
def test_inverse_rejects_non_native_narrow_writer_at_construction(interval):
    with pytest.raises(ValueError, match="native streamwise writer"):
        make_config(hidden_size=32, residual_stream_recompute_num_layers=interval)
    assert make_config(hidden_size=32, residual_stream_recompute_num_layers=1)


def test_inverse_rejects_non_native_writer_before_launch():
    from megatron.core.fusions.wide_residual_shortcut.residual_inverse import ResidualInverseManager
    from megatron.core.transformer.streamwise_residual_ops import streamwise_sigmoid_writeback

    r = torch.empty(1024, 96, device="cuda", requires_grad=True)
    u = torch.empty(1024, 32, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    w = torch.zeros(128, device="cuda", requires_grad=True)
    with ResidualInverseManager(16).activate():
        with pytest.raises(ValueError, match="qualified native streamwise writer"):
            streamwise_sigmoid_writeback(r, u, w, 3)


@pytest.mark.parametrize("retention", [False, True])
def test_inverse_restores_early_routed_reader_without_running_later_backward(
    retention, monkeypatch
):
    """An auxiliary routed loss can reach an old read before A backward is ready."""
    from megatron.core.fusions.wide_residual_shortcut import reconstruction
    from megatron.core.fusions.wide_residual_shortcut.residual_inverse import ResidualInverseManager
    from megatron.core.transformer.streamwise_residual_ops import (
        streamwise_sigmoid_read,
        streamwise_sigmoid_writeback,
    )

    torch.manual_seed(91)
    r = torch.randn(1024, 384, device="cuda", requires_grad=True)
    u = torch.randn(1024, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    w = torch.zeros(128, device="cuda", requires_grad=True)
    t = torch.full((128,), 4.59511985, device="cuda", requires_grad=True) if retention else None
    read = torch.zeros(128, device="cuda", requires_grad=True)
    gamma = torch.ones(128, device="cuda", dtype=torch.bfloat16)
    # The MoonEP benchmark checkout expresses identity carry with fixed +inf logits.
    fixed = torch.full_like(w, float("inf"))
    fixed_or_t = fixed if t is None else t
    with ResidualInverseManager(16).activate() as manager:
        first = streamwise_sigmoid_writeback(
            r, u, w, 3, retention_logits=t, retention_max_forget=0.1
        )
        routed = streamwise_sigmoid_read(first, read, 3, output_dtype=torch.bfloat16)
        carried, _ = kernels.write_read_norm(first, u, w, fixed_or_t, read, gamma, num_streams=3)
        final = streamwise_sigmoid_writeback(
            carried, u, w, 3, retention_logits=t, retention_max_forget=0.1
        )
    assert first.untyped_storage().nbytes() == 0
    assert carried.untyped_storage().nbytes() == 0

    def forbidden(*args, **kwargs):
        raise AssertionError("forward replay or later A/B backward was used for an early read")

    monkeypatch.setattr(reconstruction, "forward", forbidden)
    from megatron.core.fusions.wide_residual_shortcut import api

    monkeypatch.setattr(api, "_backward", forbidden)
    routed.float().square().mean().backward()
    assert manager.counters["lazy_inverses"] == 2
    assert torch.isfinite(r.grad).all() and torch.isfinite(read.grad).all()
    assert torch.isfinite(final).all()

    reference_values = [v.detach().clone().requires_grad_(True) for v in (r, u, w, read)]
    rr, uu, ww, rd = reference_values
    tt = None if t is None else t.detach().clone().requires_grad_(True)
    ref = streamwise_sigmoid_writeback(rr, uu, ww, 3, retention_logits=tt, retention_max_forget=0.1)
    ref_read = streamwise_sigmoid_read(ref, rd, 3, output_dtype=torch.bfloat16)
    ref_read.float().square().mean().backward()
    torch.testing.assert_close(r.grad, rr.grad, rtol=0, atol=0)
    torch.testing.assert_close(read.grad, rd.grad, rtol=2e-5, atol=2e-6)
