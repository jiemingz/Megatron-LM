# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Fused A/B seams for adjacent wide-residual Mamba/GDN shortcut MoE pairs."""

from dataclasses import dataclass

import torch

from megatron.core.fusions import wide_residual_shortcut as kernels
from megatron.core.inference.utils import InferenceMode
from megatron.core.models.hybrid.layers.shortcut_fusion_validation import (
    validate_shortcut_fusion_config,
)
from megatron.core.models.hybrid.shortcut_block import ShortcutMoEBlock
from megatron.core.ssm.wide_residual_mamba_layer import WideResidualMambaLayer
from megatron.core.transformer.identity_op import IdentityOp
from megatron.core.transformer.moe.shared_experts import set_tensor_grad_fn_sequence_sr
from megatron.core.transformer.wide_residual_layer import (
    StreamwiseSigmoidWideResidualConnection,
    WideResidualTransformerLayer,
)


@dataclass(frozen=True)
class PreparedShortcutInput:
    """One producer's residual and the adjacent consumer's independent norm outputs."""

    residual: torch.Tensor
    compute: torch.Tensor
    shortcut: torch.Tensor
    consumer: int


class FusedShortcutMoEBlock(ShortcutMoEBlock):
    """Keep existing projection/router/expert owners and fuse their residual seams."""

    fuses_shortcut_boundaries = True

    def __init__(
        self, compute_layer, moe_layer, overlap_a2a, attn_local_idx=None, moe_local_idx=None
    ) -> None:
        if not isinstance(compute_layer, (WideResidualMambaLayer, WideResidualTransformerLayer)):
            raise ValueError(
                "shortcut kernel fusion requires a wide-residual Mamba/GDN predecessor"
            )
        super().__init__(compute_layer, moe_layer, overlap_a2a, attn_local_idx, moe_local_idx)
        validate_shortcut_fusion_config(self.compute_layer.config)
        validate_shortcut_fusion_config(self.moe_layer.config)
        if self.is_mtp_layer:
            raise ValueError(
                "shortcut kernel fusion requires a non-MTP wide-residual Mamba/GDN pair"
            )
        if isinstance(compute_layer, WideResidualTransformerLayer) and (
            not isinstance(compute_layer.mlp, IdentityOp)
            or not isinstance(compute_layer.cross_attention, IdentityOp)
        ):
            raise ValueError("shortcut kernel fusion requires a compute-only predecessor")
        connections = (self._compute_connection(), self.moe_layer._get_mlp_residual_connection())
        if any(not isinstance(c, StreamwiseSigmoidWideResidualConnection) for c in connections):
            raise ValueError("shortcut kernel fusion requires streamwise sigmoid connections")
        if any(c.num_streams != self.config.wide_residual.num_streams for c in connections):
            raise ValueError("shortcut kernel fusion connections must use the same stream count")
        for norm in (
            self._compute_norm(),
            self.moe_layer.pre_mlp_layernorm,
            self.shortcut_pre_mlp_layernorm,
        ):
            if (
                getattr(norm, "eps", None) != 1e-5
                or getattr(norm, "zero_centered_gamma", False)
                or getattr(norm, "returns_residual", False)
            ):
                raise ValueError(
                    "shortcut kernel fusion requires ordinary RMSNorm with epsilon 1e-5"
                )
        if not self.moe_layer.mlp.use_shared_expert:
            raise ValueError("shortcut kernel fusion requires a shared expert")

    def _compute_connection(self):
        compute = self.compute_layer
        if isinstance(compute, WideResidualMambaLayer):
            return compute.residual_connection
        return compute._get_self_attention_residual_connection()

    def _compute_norm(self):
        compute = self.compute_layer
        return (
            compute.norm if isinstance(compute, WideResidualMambaLayer) else compute.input_layernorm
        )

    def _retention_logits(self, connection):
        if connection.retention is None:
            # None selects the fixed identity branch; it is not a trainable logit.
            return None
        return connection.retention(return_logits=True).float()

    def forward(
        self,
        hidden_states,
        attention_mask,
        inference_context,
        rotary_pos_emb,
        sequence_len_offset,
        packed_seq_params,
        padding_mask,
        quant_context_factory,
        cp_layout_state=None,
        packed_sequence_cp_metadata=None,
        attn_recompute_context=None,
        moe_recompute_context=None,
        next_shortcut_block=None,
    ):
        """Execute A, and B when the next physical pair can consume the handoff."""
        if (
            InferenceMode.is_active()
            or inference_context is not None
            or any(
                x is not None
                for x in (
                    packed_seq_params,
                    padding_mask,
                    cp_layout_state,
                    packed_sequence_cp_metadata,
                )
            )
        ):
            raise ValueError("shortcut kernel fusion requires eager, unpacked training")
        if attn_recompute_context is not None or moe_recompute_context is not None:
            raise ValueError("fused residuals require inverse handoff, not forward replay")
        prepared = isinstance(hidden_states, PreparedShortcutInput)
        if prepared and hidden_states.consumer != id(self):
            raise ValueError("prepared shortcut input belongs to another block")
        residual = hidden_states.residual if prepared else hidden_states
        fp32 = self.config.fp32_residual_connection
        if fp32:
            residual = residual.float()
        streams, hidden = self.config.wide_residual.num_streams, self.config.hidden_size
        kernels.validate_input(residual, hidden, streams, fp32_residual_connection=fp32)
        select = getattr(self.moe_layer.mlp, "select_token_dispatcher", None)
        if select is not None:
            select()
        with quant_context_factory(self.moe_layer.config, self.moe_layer_idx):
            if prepared:
                shortcut = hidden_states.shortcut
                probs, routing = self.moe_layer.mlp.route(shortcut, None)
                route_input, route_probs = self.moe_layer.mlp.preprocess(
                    shortcut, probs, routing, None
                )
            else:
                shortcut = self._read_shortcut_hidden(
                    residual, recompute_context=attn_recompute_context
                )
                route_input, route_probs = self._moe_router_preprocess(
                    shortcut, recompute_context=attn_recompute_context
                )
        if self.overlap_mode:
            self.route_ready_event.record(torch.cuda.current_stream())
        compute = self.compute_layer
        with quant_context_factory(compute.config, self.attn_layer_idx):
            if prepared:
                compute_input = hidden_states.compute
            else:
                if isinstance(compute, WideResidualMambaLayer):
                    compute_input, _, _, _ = compute._prepare_mixer_state(
                        residual, residual_stream_recompute_context=attn_recompute_context
                    )
                else:
                    compute_input, _, _ = compute._run_input_layernorm(
                        residual, residual_stream_recompute_context=attn_recompute_context
                    )
            if isinstance(compute, WideResidualMambaLayer):
                mixer = compute.mixer
                core = mixer.forward_pre_attn_and_core_attn(
                    compute_input, packed_seq_params=None, packed_sequence_cp_metadata=None
                )
            else:
                mixer = compute.self_attention
                core = mixer.forward_pre_attn_and_core_attn(
                    compute_input,
                    attention_mask=attention_mask,
                    rotary_pos_emb=rotary_pos_emb,
                    attention_bias=None,
                    packed_seq_params=None,
                    packed_sequence_cp_metadata=None,
                )
        with quant_context_factory(self.moe_layer.config, self.moe_layer_idx):
            dispatched, probs = self._launch_dispatch(
                route_input, route_probs, async_op=self.overlap_mode
            )
            if self.overlap_mode:
                dispatched, probs = self._wait_dispatch(dispatched, probs)
            routed, _ = self.moe_layer.mlp.routed_experts_compute(dispatched, probs)
            combined = self._launch_combine(routed, async_op=self.overlap_mode)
        with quant_context_factory(compute.config, self.attn_layer_idx):
            update, bias = mixer.forward_post_core_attn(core)
            if bias is not None:
                raise ValueError("shortcut kernel fusion requires a bias-free mixer projection")
        producer = self._compute_connection()
        consumer = self.moe_layer._get_mlp_residual_connection()
        options = dict(num_streams=streams, fp32_residual_connection=fp32)
        with quant_context_factory(self.moe_layer.config, self.moe_layer_idx):

            def run_a(carried, branch):
                return kernels.write_read_norm(
                    carried,
                    branch,
                    producer.write_map(return_logits=True).float(),
                    self._retention_logits(producer),
                    consumer.read_map(return_logits=True).float(),
                    self.moe_layer.pre_mlp_layernorm.weight,
                    **options,
                )

            wide, shared_input = run_a(residual, update)
            shared = self.moe_layer.mlp.shared_experts_compute(shared_input)
            if self.overlap_mode:
                combined = self._wait_combine(combined)
            set_tensor_grad_fn_sequence_sr(combined, torch.iinfo(torch.int).max)
            if isinstance(next_shortcut_block, FusedShortcutMoEBlock):
                nxt = next_shortcut_block
                if (
                    nxt.config.hidden_size != hidden
                    or nxt.config.wide_residual.num_streams != streams
                    or nxt.config.fp32_residual_connection != fp32
                ):
                    raise ValueError("adjacent fused shortcut blocks must agree on residual layout")
                routed = self.moe_layer.mlp.postprocess(combined, None)

                def run_b(carried, routed_update, shared_update):
                    return kernels.boundary_norm(
                        carried,
                        routed_update,
                        shared_update,
                        consumer.write_map(return_logits=True).float(),
                        self._retention_logits(consumer),
                        (
                            nxt._compute_connection().read_map(return_logits=True).float(),
                            nxt.shortcut_residual_read.read_map(return_logits=True).float(),
                        ),
                        (nxt._compute_norm().weight, nxt.shortcut_pre_mlp_layernorm.weight),
                        **options,
                    )

                wide, normalized, shortcut = run_b(wide, routed, shared)
                return PreparedShortcutInput(wide, normalized, shortcut, id(nxt))
            return self._postprocess(wide, combined, shared, mlp_state=(wide,))


def resolve_fusion_submodules(submodules):
    """Select static fusion-ready specs for the built-in training stacks."""
    from megatron.core.models.hybrid import hybrid_layer_specs as specs

    for original, fused in (
        (specs.hybrid_stack_spec, specs.wide_residual_shortcut_fusion_stack_spec),
        (specs.wide_residual_hybrid_stack_spec, specs.wide_residual_shortcut_fusion_stack_spec),
        (specs.gated_delta_product_stack_spec, specs.wide_residual_shortcut_fusion_gdp_stack_spec),
        (
            specs.wide_residual_gated_delta_product_stack_spec,
            specs.wide_residual_shortcut_fusion_gdp_stack_spec,
        ),
    ):
        if submodules is original.submodules:
            return fused.submodules
    return submodules
