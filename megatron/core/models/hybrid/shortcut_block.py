# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Implement shortcut-connected MoE blocks based on the ScMoE architecture introduced in
`Shortcut-connected Expert Parallelism for Accelerating Mixture of Experts
<https://arxiv.org/abs/2404.05019>`_.

ScMoE routes the preceding layer's representation through sparse experts while the paired layer
computes attention and the shared expert, breaking the usual sequential dependency so
expert-parallel All-to-All communication can overlap with computation.
Our variant adds two LayerNorms to the paper's design: `shortcut_pre_mlp_layernorm` normalizes
the shortcut representation before routing and expert dispatch, while `shortcut_post_norm`
normalizes the merged routed- and shared-expert result before the MoE residual/BDA step.
Also supports selective activation recomputation and activation offloading on
shortcut_pre_mlp_layernorm and shortcut_post_norm, respectively, to reduce the additional
device-memory pressure from enabling Shortcut-MoE.
"""

from __future__ import annotations

from typing import Sequence

import torch

from megatron.core.activation_policy import (
    CheckpointWithoutOutputStrategy,
    Event,
    ExcludeFromOffloadStrategy,
    OffloadStrategy,
    ReplayRegion,
    ResidualBranch,
    TensorRef,
)
from megatron.core.extensions.transformer_engine import TENorm
from megatron.core.inference.utils import InferenceMode
from megatron.core.models.hybrid.hybrid_layer_allocation import Symbols as LayerSymbols
from megatron.core.ssm.gdn_layer_config import GDNLayerConfig
from megatron.core.ssm.mamba_layer import MambaLayer
from megatron.core.ssm.mamba_layer_config import MambaLayerConfig
from megatron.core.transformer.identity_op import IdentityOp, is_identity
from megatron.core.transformer.module import MegatronModule, TwoStageAttentionLayer
from megatron.core.transformer.moe.experts import InferenceGroupedMLP
from megatron.core.transformer.moe.moe_layer_config import MoELayerConfig
from megatron.core.transformer.moe.router import InferenceTopKRouter
from megatron.core.transformer.moe.shared_experts import set_tensor_grad_fn_sequence_sr
from megatron.core.transformer.moe.token_dispatcher_inference import NVLSAllGatherVDispatcher
from megatron.core.transformer.spec_utils import build_module
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.transformer_layer import TransformerLayer
from megatron.core.transformer.wide_residual import StreamwiseSigmoidWideResidualRead
from megatron.core.typed_torch import apply_module
from megatron.core.utils import get_pg_size

# Built-in config types that may precede a shortcut MoE.
SUPPORTED_SHORTCUT_PREDECESSOR_CONFIGS = frozenset(
    {MambaLayerConfig, GDNLayerConfig} | LayerSymbols.ATTENTION_LAYER_CONFIGS
)


def group_layers_into_shortcut_blocks(
    layers: torch.nn.ModuleList,
    layer_config_list: Sequence[TransformerConfig],
    config: TransformerConfig,
    pp_layer_offset: int = 0,
) -> torch.nn.ModuleList:
    """Group physical layers into their registered shortcut-block hierarchy.

    Grouping updates the layer names through the returned `ModuleList` hierarchy. Layers not
    followed by an MoE remain direct children of the returned `ModuleList`.

    Args:
        layers: Physical layers in execution order.
        layer_config_list: Physical layer configs in execution order.
        config: Transformer configuration controlling shortcut scheduling.
        pp_layer_offset: Global offset of this pipeline stage's first physical layer.

    Returns:
        The registered logical layers.

    Raises:
        ValueError: If an MoE has an unsupported predecessor or its shortcut pair crosses a
            pipeline-stage boundary.
    """

    if pp_layer_offset > 0 and layer_config_list and type(layer_config_list[0]) is MoELayerConfig:
        raise ValueError(
            "Shortcut MoE pairs cannot cross pipeline-stage boundaries; this pipeline stage "
            f"begins with an MoE layer (global layer {pp_layer_offset + 1}). Move the stage "
            "boundary so each MoE layer stays on the same stage as the layer before it."
        )

    grouped_layers = torch.nn.ModuleList()
    physical_index = 0
    while physical_index < len(layers):
        next_is_moe = (
            physical_index + 1 < len(layers)
            and type(layer_config_list[physical_index + 1]) is MoELayerConfig
        )
        if not next_is_moe:
            grouped_layers.append(layers[physical_index])
            physical_index += 1
            continue

        compute_layer = layers[physical_index]
        paired_type = type(layer_config_list[physical_index])
        if paired_type not in SUPPORTED_SHORTCUT_PREDECESSOR_CONFIGS:
            raise ValueError(
                "Shortcut MoE must be preceded by a Mamba, GDN, or supported attention layer"
            )

        supports_two_stage = (
            isinstance(compute_layer, TwoStageAttentionLayer)
            and compute_layer.supports_two_stage_attention()
        )
        if not supports_two_stage:
            raise ValueError(
                f"Shortcut layer {paired_type.__name__} does not support two-stage attention"
            )
        moe_layer = layers[physical_index + 1]
        grouped_layers.append(
            ShortcutMoEBlock(
                compute_layer,
                moe_layer,
                overlap_a2a=config.moe_shortcut_parallel,
                attn_local_idx=physical_index,
                moe_local_idx=physical_index + 1,
            )
        )
        physical_index += 2

    return grouped_layers


class ShortcutMoEBlock(MegatronModule):
    """Own and execute one attention-layer/shortcut-MoE pair."""

    _parallel_stream: torch.cuda.Stream | None = None

    @classmethod
    def _get_a2a_overlap_stream(cls) -> torch.cuda.Stream:
        """Return the process-wide high-priority shortcut stream."""
        if cls._parallel_stream is None:
            cls._parallel_stream = torch.cuda.Stream(priority=-1)
        return cls._parallel_stream

    # Inference uses all-gather/reduce-scatter on the same MoE stream.
    _get_moe_overlap_stream = _get_a2a_overlap_stream

    def __init__(
        self,
        compute_layer,
        moe_layer,
        overlap_a2a: bool,
        attn_local_idx: int | None = None,
        moe_local_idx: int | None = None,
    ):
        super().__init__(compute_layer.config)

        self.overlap_mode = overlap_a2a
        self.layer_number = compute_layer.layer_number
        self.attn_layer_idx = compute_layer.layer_number - 1
        self.attn_local_idx = attn_local_idx
        self.moe_layer_idx = moe_layer.layer_number - 1
        self.moe_local_idx = moe_local_idx

        self.is_first_layer = getattr(compute_layer, "is_first_layer", False)
        self.is_last_layer = getattr(moe_layer, "is_last_layer", False)
        self.tp_group = moe_layer.mlp.tp_group
        self.compute_layer = compute_layer
        self.moe_layer = moe_layer

        compute_is_mtp = bool(getattr(compute_layer, "is_mtp_layer", False))
        moe_is_mtp = bool(getattr(moe_layer, "is_mtp_layer", False))
        if compute_is_mtp != moe_is_mtp:
            raise ValueError("A ShortcutMoE pair must agree on whether it belongs to MTP.")
        self.is_mtp_layer = moe_is_mtp
        if not self.is_mtp_layer and (
            (compute_layer.config.wide_residual is None) != (moe_layer.config.wide_residual is None)
        ):
            raise ValueError(
                "A ShortcutMoE pair must agree on whether it carries a wide residual stream."
            )

        self.shortcut_residual_read = None
        if moe_layer.config.wide_residual is not None and not self.is_mtp_layer:
            self.shortcut_residual_read = StreamwiseSigmoidWideResidualRead(
                config=moe_layer.config,
                layer_number=moe_layer.layer_number,
                branch_name="shortcut_routed",
            )
            outer_read = moe_layer._get_residual_connection_for_mlp(operation="read")
            if outer_read is None:
                raise ValueError(
                    "A wide-residual shortcut read requires an outer MoE residual connection."
                )
            if (
                self.shortcut_residual_read.residual_stream_hidden_size
                != outer_read.residual_stream_hidden_size
            ):
                raise ValueError(
                    "The shortcut read and outer MoE residual connection must consume the "
                    "same residual-stream width."
                )
            predecessor_stream_width = getattr(compute_layer, "residual_stream_hidden_size", None)
            if predecessor_stream_width != self.shortcut_residual_read.residual_stream_hidden_size:
                raise ValueError(
                    "The shortcut predecessor and routed read must carry the same "
                    "residual-stream width."
                )

        shortcut_norm_spec = TENorm
        self.shortcut_pre_mlp_layernorm = build_module(
            shortcut_norm_spec,
            config=self.config,
            hidden_size=self.config.hidden_size,
            eps=self.config.layernorm_epsilon,
            has_residual=False,
        )
        self.shortcut_post_norm = (
            build_module(
                shortcut_norm_spec,
                config=self.config,
                hidden_size=self.config.hidden_size,
                eps=self.config.layernorm_epsilon,
                has_residual=False,
            )
            if self.config.moe_shortcut_post_norm
            else IdentityOp()
        )
        self.route_ready_event = torch.cuda.Event() if self.overlap_mode else None
        # Activation strategies for the shortcut norms of the attention/MoE pair (_launch_combine
        # is the routed-output combine). Under residual-stream recompute the
        # shortcut read and both norms replay with their block (see
        # get_residual_branches_by_layer); they are then not offloaded and their outputs are kept
        # out of other offload groups.
        config = self.config
        exclude = config.residual_stream_replay() and config.fine_grained_activation_offloading
        activation_strategies = []
        if config.recompute_selected("shortcut_pre_mlp_layernorm"):
            # The pre-norm output is no longer needed once the routed output is combined.
            activation_strategies.append(
                CheckpointWithoutOutputStrategy(
                    self.shortcut_pre_mlp_layernorm, release_on=Event(self._launch_combine, "exit")
                )
            )
        if exclude:
            activation_strategies.append(
                ExcludeFromOffloadStrategy(self.shortcut_pre_mlp_layernorm)
            )
        if config.offload_selected("shortcut_post_norm"):
            activation_strategies.append(
                OffloadStrategy(
                    self.shortcut_post_norm,
                    "shortcut_post_norm",
                    release=(TensorRef.input(self.shortcut_post_norm),),
                    skip_when_replayed=True,
                )
            )
        if exclude:
            activation_strategies.append(ExcludeFromOffloadStrategy(self.shortcut_post_norm))
        self.install_activation_strategies(activation_strategies)

    def setup_inference_overlap(self) -> None:
        """Create synchronization state and check stream ownership before graph capture.

        At TP>1 the shortcut stream must not issue any TP-group collective (see
        `_forward_inference_overlap`). The inference router and grouped experts communicate
        only through the NVLS dispatcher, so require them rather than trusting a custom spec.
        """
        if self.route_ready_event is None:
            self.route_ready_event = torch.cuda.Event()
        if get_pg_size(self.tp_group) > 1:
            mlp = self.moe_layer.mlp
            if not isinstance(getattr(mlp, "router", None), InferenceTopKRouter) or not (
                isinstance(getattr(mlp, "experts", None), InferenceGroupedMLP)
            ):
                raise RuntimeError(
                    "Shortcut MoE inference overlap at TP>1 requires InferenceTopKRouter and "
                    "InferenceGroupedMLP, which issue no TP-group collectives"
                )

    def get_residual_branches_by_layer(self) -> list:
        """The paired layers' residual branches; the MoE layer's also has the shortcut read and
        norms (its last write ends the pair, so a replay block never ends inside it)."""
        shortcut_ops = [ReplayRegion(self, self.shortcut_pre_mlp_layernorm)]
        if not is_identity(self.shortcut_post_norm):
            shortcut_ops.append(ReplayRegion(self, self.shortcut_post_norm))
        shortcut = ResidualBranch(
            read=(
                ReplayRegion(self, self.shortcut_residual_read)
                if self.shortcut_residual_read is not None
                else None
            ),
            ops=shortcut_ops,
        )
        return [
            self.compute_layer.get_residual_branches(),
            [*self.moe_layer.get_residual_branches(), shortcut],
        ]

    def _read_shortcut_hidden(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Read the pair input used only by shortcut routing."""

        if self.shortcut_residual_read is None:
            return hidden_states
        return apply_module(self.shortcut_residual_read)(hidden_states)

    def _moe_router_preprocess(self, shortcut_hidden, padding_mask=None, packed_seq_params=None):
        """Run shortcut normalization, routing, and dispatch preprocessing."""
        shortcut_input = apply_module(self.shortcut_pre_mlp_layernorm)(shortcut_hidden)
        shortcut_input, padding_mask, _, _ = self.moe_layer._maybe_unflatten_for_moe(
            shortcut_input, padding_mask, input_ids=None, packed_seq_params=packed_seq_params
        )
        probs, routing_map = self.moe_layer.mlp.route(
            shortcut_input, padding_mask, packed_seq_params=packed_seq_params
        )
        return self.moe_layer.mlp.preprocess(
            shortcut_input, probs, routing_map, padding_mask, packed_seq_params=packed_seq_params
        )

    def _moe_shared_experts(self, hidden_states, padding_mask=None, packed_seq_params=None):
        """Run the paired MoE layer's pre-MLP norm and shared experts.

        Returns:
            `(shared_expert_output, moe_unflatten_mbs, residual, mlp_state)`.
        """
        pre_mlp_output, residual, mlp_state = self.moe_layer._pre_mlp_layernorm_and_residual(
            hidden_states
        )
        pre_mlp_output, _, _, moe_unflatten_mbs = self.moe_layer._maybe_unflatten_for_moe(
            pre_mlp_output, padding_mask, input_ids=None, packed_seq_params=packed_seq_params
        )
        shared_expert_output = self.moe_layer.mlp.shared_experts_compute(pre_mlp_output)
        return shared_expert_output, moe_unflatten_mbs, residual, mlp_state

    def _postprocess(
        self,
        residual,
        combined_output,
        shared_expert_output,
        packed_seq_params=None,
        moe_unflatten_mbs=None,
        mlp_state=(),
    ):
        """Join routed/shared output, apply shortcut post-norm, and finish residual/BDA."""
        output = self.moe_layer.mlp.postprocess(combined_output, shared_expert_output)
        output = apply_module(self.shortcut_post_norm)(output)
        output = self.moe_layer._maybe_reflatten_from_moe(
            output, packed_seq_params, moe_unflatten_mbs
        )
        output = self.moe_layer._apply_mlp_bda_step((output, None), residual, mlp_state)
        return output[0] if isinstance(output, tuple) else output

    def _launch_dispatch(
        self, hidden_states: torch.Tensor, probs: torch.Tensor, async_op: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Launch dispatch on the current stream or the shortcut side stream."""
        if not async_op:
            return self.moe_layer.mlp.dispatch(hidden_states, probs)

        assert self.route_ready_event is not None
        dispatch_stream = self._get_a2a_overlap_stream()
        dispatch_stream.wait_event(self.route_ready_event)
        hidden_states.record_stream(dispatch_stream)
        probs.record_stream(dispatch_stream)

        with torch.cuda.stream(dispatch_stream):
            return self.moe_layer.mlp.dispatch(hidden_states, probs)

    def _wait_dispatch(
        self, dispatched_input: torch.Tensor, dispatched_probs: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Wait for dispatch and return its outputs on the main stream."""
        assert self.overlap_mode

        torch.cuda.current_stream().wait_stream(self._get_a2a_overlap_stream())
        dispatched_input.record_stream(torch.cuda.current_stream())
        dispatched_probs.record_stream(torch.cuda.current_stream())
        return dispatched_input, dispatched_probs

    def _dispatch_and_compute_routed_experts(
        self, route_input: torch.Tensor, route_probs: torch.Tensor, overlap: bool
    ) -> torch.Tensor:
        """Dispatch tokens and compute local experts within one ownership scope."""
        dispatched_input, dispatched_probs = self._launch_dispatch(
            route_input, route_probs, async_op=overlap
        )
        if overlap:
            dispatched_input, dispatched_probs = self._wait_dispatch(
                dispatched_input, dispatched_probs
            )
        output, _ = self.moe_layer.mlp.routed_experts_compute(dispatched_input, dispatched_probs)
        return output

    def _launch_combine(self, output: torch.Tensor, async_op: bool = False) -> torch.Tensor:
        """Launch combine on the current stream or the shortcut side stream."""
        if not async_op:
            return self.moe_layer.mlp.combine(output)

        combine_stream = self._get_a2a_overlap_stream()
        combine_stream.wait_stream(torch.cuda.current_stream())
        output.record_stream(combine_stream)
        with torch.cuda.stream(combine_stream):
            return self.moe_layer.mlp.combine(output)

    def _wait_combine(self, combined_output: torch.Tensor) -> torch.Tensor:
        """Wait for the asynchronous combine and return its output on the main stream."""
        torch.cuda.current_stream().wait_stream(self._get_a2a_overlap_stream())
        combined_output.record_stream(torch.cuda.current_stream())
        return combined_output

    def _forward_compute_atomic(
        self,
        hidden_states,
        *,
        attention_mask,
        inference_context,
        rotary_pos_emb,
        sequence_len_offset,
        packed_seq_params,
        padding_mask,
    ):
        """Run the paired layer's ordinary forward, which updates its KV cache or recurrent state.

        Uses the same per-layer arguments as HybridStack.forward passes to unpaired layers.
        """
        if isinstance(self.compute_layer, MambaLayer):
            return self.compute_layer(
                hidden_states,
                attention_mask=attention_mask,
                inference_context=inference_context,
                packed_seq_params=packed_seq_params,
            )
        hidden_states, _ = self.compute_layer(
            hidden_states,
            attention_mask=attention_mask,
            inference_context=inference_context,
            rotary_pos_emb=rotary_pos_emb,
            sequence_len_offset=sequence_len_offset,
            packed_seq_params=packed_seq_params,
            padding_mask=padding_mask,
        )
        return hidden_states

    def _forward_inference_overlap(
        self,
        hidden_states,
        *,
        attention_mask,
        inference_context,
        rotary_pos_emb,
        sequence_len_offset,
        packed_seq_params,
        padding_mask,
    ):
        """Run independent mixer and routed-expert branches until their outputs join.

        Every phase uses the predecessor's ordinary cache-aware forward on the
        main stream. Routing, dispatch, experts and combine run on the MoE
        stream. CUDA graphs capture the input-ready fork and the join before
        the residual write.

        Collective ownership, which makes TP>1 safe without ordering the two streams:

        - Main stream: every TP-group collective. That is the predecessor's and the shared
          expert's sequence-parallel all-gathers and reduce-scatters, on the "tp" symmetric
          buffer or, when a tensor does not fit it, on the NCCL TP communicator.
        - Shortcut stream: only the NVLS dispatcher's metadata exchange, AllGather-V and
          ReduceScatter-V over the tp_ep group, on its own "ep_*" symmetric buffers. Each
          buffer has its own signal pad, so the two streams share no buffer, barrier slot
          or communicator. The shortcut norm, inference router, latent projections
          ("duplicated") and grouped experts are local. With sequence parallelism each rank
          routes its own token shard, and the experts receive tokens only through the
          AllGather-V, so no TP gather is needed first. Expert-TP partial sums are added by
          the ReduceScatter-V.

        Each stream therefore issues its collectives in program order, which is the same on
        every rank, and nothing on the main stream waits on the shortcut stream before the
        join. Configurations that would break this (shared-expert overlap, fused TP
        communication, batch-invariant collectives, non-NVLS dispatch) are rejected by
        `InferenceConfig.validate_shortcut_moe_model` and `setup_inference_overlap`.
        """
        if inference_context is None or not inference_context.is_dynamic_batching():
            raise RuntimeError("Shortcut MoE inference overlap requires dynamic batching")
        if inference_context.num_speculative_tokens:
            raise RuntimeError(
                "Shortcut MoE overlap with speculative decoding requires further testing"
            )
        shortcut_hidden = hidden_states
        if not isinstance(self.compute_layer, (MambaLayer, TransformerLayer)):
            raise RuntimeError(
                "Shortcut MoE inference overlap requires a Mamba or Transformer layer"
            )
        if not isinstance(self.moe_layer.mlp.token_dispatcher, NVLSAllGatherVDispatcher):
            raise RuntimeError("Shortcut MoE inference overlap requires the active NVLS dispatcher")

        main_stream = torch.cuda.current_stream()
        shortcut_stream = self._get_moe_overlap_stream()
        # Fork before the shortcut read/router. Protect the original pair input
        # and padding mask until side-stream routing has consumed them.
        self.route_ready_event.record(main_stream)
        shortcut_stream.wait_event(self.route_ready_event)
        shortcut_hidden.record_stream(shortcut_stream)
        if isinstance(padding_mask, torch.Tensor) and padding_mask.is_cuda:
            padding_mask.record_stream(shortcut_stream)
        with torch.cuda.stream(shortcut_stream):
            with self.moe_layer.get_inner_quantization_context():
                shortcut_hidden = self._read_shortcut_hidden(shortcut_hidden)
                route_input, route_probs = self._moe_router_preprocess(
                    shortcut_hidden=shortcut_hidden,
                    padding_mask=padding_mask,
                    packed_seq_params=packed_seq_params,
                )
                dispatched_input, dispatched_probs = self._launch_dispatch(
                    route_input, route_probs, async_op=False
                )

                routed_output, _ = self.moe_layer.mlp.routed_experts_compute(
                    dispatched_input, dispatched_probs
                )
                combined_output = self._launch_combine(routed_output, async_op=False)

        with self.compute_layer.get_inner_quantization_context():
            compute_output = self._forward_compute_atomic(
                hidden_states,
                attention_mask=attention_mask,
                inference_context=inference_context,
                rotary_pos_emb=rotary_pos_emb,
                sequence_len_offset=sequence_len_offset,
                packed_seq_params=packed_seq_params,
                padding_mask=padding_mask,
            )

        with self.moe_layer.get_inner_quantization_context():
            shared_expert_output, moe_unflatten_mbs, residual, mlp_state = self._moe_shared_experts(
                hidden_states=compute_output,
                padding_mask=padding_mask,
                packed_seq_params=packed_seq_params,
            )
            combined_output = self._wait_combine(combined_output)
            return self._postprocess(
                residual=residual,
                combined_output=combined_output,
                shared_expert_output=shared_expert_output,
                mlp_state=mlp_state,
                packed_seq_params=packed_seq_params,
                moe_unflatten_mbs=moe_unflatten_mbs,
            )

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        inference_context=None,
        rotary_pos_emb=None,
        sequence_len_offset=None,
        packed_seq_params=None,
        padding_mask=None,
        *,
        cp_layout_state=None,
        packed_sequence_cp_metadata=None,
        padding_mask_by_layout=None,
        input_ids=None,
    ):
        """Run the eager schedule with each physical layer's quantization context.

        The layer stack has prepared the pair's input in the attention layer's CP layout
        (``packed_seq_params`` and ``padding_mask`` match it) and finalizes the output after the
        MoE layer; with ``cp_layout_state`` the pair converts to the MoE layer's layout itself.

        In inference the paired layer runs its ordinary forward instead of the two-stage split,
        which never updates KV-cache or recurrent state, and communication stays on the current
        stream.
        """
        del input_ids  # ShortcutMoE does not support hash routing (see TransformerConfig).
        select_token_dispatcher = getattr(self.moe_layer.mlp, "select_token_dispatcher", None)
        if select_token_dispatcher is not None:
            select_token_dispatcher()
        inference = InferenceMode.is_active()
        if inference:
            if cp_layout_state is not None:
                raise RuntimeError("Shortcut-MoE inference does not support context parallelism.")
        else:
            assert inference_context is None, (
                "Shortcut-MoE received an inference context outside inference mode; the two-stage "
                "schedule does not update KV-cache or recurrent inference state."
            )
        if (
            inference
            and inference_context is not None
            and inference_context.config.inference_shortcut_moe_overlap
        ):
            return self._forward_inference_overlap(
                hidden_states,
                attention_mask=attention_mask,
                inference_context=inference_context,
                rotary_pos_emb=rotary_pos_emb,
                sequence_len_offset=sequence_len_offset,
                packed_seq_params=packed_seq_params,
                padding_mask=padding_mask,
            )
        overlap = self.overlap_mode and not inference

        attn_quantization = self.compute_layer.get_inner_quantization_context
        moe_quantization = self.moe_layer.get_inner_quantization_context
        moe_hidden_states = hidden_states
        moe_packed_seq_params = packed_seq_params
        moe_padding_mask = padding_mask
        if cp_layout_state is not None:
            assert self.attn_local_idx is not None and self.moe_local_idx is not None
            moe_hidden_states, moe_packed_seq_params = cp_layout_state.prepare_layer(
                self.moe_local_idx, moe_hidden_states
            )
            moe_padding_mask = cp_layout_state.get_layer_padding_mask(
                self.moe_local_idx, padding_mask, padding_mask_by_layout
            )

        # Launch the moe_router
        with moe_quantization():
            moe_hidden_states = self._read_shortcut_hidden(moe_hidden_states)
            route_input, route_probs = self._moe_router_preprocess(
                shortcut_hidden=moe_hidden_states,
                padding_mask=moe_padding_mask,
                packed_seq_params=moe_packed_seq_params,
            )
            if overlap:
                self.route_ready_event.record(torch.cuda.current_stream())

        # Launch the input and attn of the attention layer
        paired_state = None  # initialize variable early to make pylint happy
        with attn_quantization():
            if inference:
                attn_layer_output = self._forward_compute_atomic(
                    hidden_states,
                    attention_mask=attention_mask,
                    inference_context=inference_context,
                    rotary_pos_emb=rotary_pos_emb,
                    sequence_len_offset=sequence_len_offset,
                    packed_seq_params=packed_seq_params,
                    padding_mask=padding_mask,
                )
            else:
                paired_state = self.compute_layer.forward_pre_attn_and_core_attn(
                    hidden_states=hidden_states,
                    attention_mask=attention_mask,
                    rotary_pos_emb=rotary_pos_emb,
                    packed_seq_params=packed_seq_params,
                    packed_sequence_cp_metadata=packed_sequence_cp_metadata,
                )

        # Launch the dispatch, experts, and combine
        with moe_quantization():
            output = self._dispatch_and_compute_routed_experts(route_input, route_probs, overlap)
            # The routed output no longer needs the shortcut pre-norm output (_launch_combine).
            combined_output = self._launch_combine(output, async_op=overlap)

        # launch the output layer of the attention layer
        if not inference:
            assert paired_state is not None
            with attn_quantization():
                attn_layer_output = self.compute_layer.forward_post_core_attn(*paired_state)
                if isinstance(attn_layer_output, tuple):
                    attn_layer_output = attn_layer_output[0]

        if cp_layout_state is not None:
            attn_layer_output, moe_packed_seq_params = cp_layout_state.prepare_layer(
                self.moe_local_idx, attn_layer_output
            )

        # launch the moe shared experts and combine attn and moe layer outputs
        with moe_quantization():
            shared_expert_output, moe_unflatten_mbs, mlp_residual, mlp_state = (
                self._moe_shared_experts(
                    hidden_states=attn_layer_output,
                    padding_mask=moe_padding_mask,
                    packed_seq_params=moe_packed_seq_params,
                )
            )
            if overlap:
                combined_output = self._wait_combine(combined_output)

            # Ensure the combine autograd node is scheduled first before shared_experts
            set_tensor_grad_fn_sequence_sr(combined_output, torch.iinfo(torch.int).max)
            output = self._postprocess(
                residual=mlp_residual,
                combined_output=combined_output,
                shared_expert_output=shared_expert_output,
                packed_seq_params=moe_packed_seq_params,
                moe_unflatten_mbs=moe_unflatten_mbs,
                mlp_state=mlp_state,
            )
        return output
