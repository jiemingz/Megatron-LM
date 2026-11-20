# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Transformer layer specialization for manifold-constrained hyper-connections."""

from __future__ import annotations

from typing import Optional

from torch import Tensor

from megatron.core.activation_policy import (
    CheckpointStrategy,
    CheckpointWithoutOutputStrategy,
    Event,
    OffloadStrategy,
    ReplayRegion,
    ResidualBranch,
)
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer.enums import CudaGraphModule
from megatron.core.transformer.hyper_connection import (
    HyperConnectionExecutionContext,
    HyperConnectionResidualConnection,
)
from megatron.core.transformer.identity_op import IdentityOp, is_identity
from megatron.core.transformer.residual_connection import (
    ResidualBranchOutput,
    ResidualConnection,
    ResidualConnectionState,
)
from megatron.core.transformer.spec_utils import build_module
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.transformer_layer import TransformerLayer, TransformerLayerSubmodules
from megatron.core.transformer.transformer_layer_mixins import (
    TECudaGraphHyperConnectionTransformerLayerMixin,
)
from megatron.core.typed_torch import apply_module


class HyperConnectionTransformerLayer(
    TECudaGraphHyperConnectionTransformerLayerMixin, TransformerLayer
):
    """A transformer layer with Manifold-Constrained Hyper-Connections (mHC).

    Extends TransformerLayer by adding hyper connection modules around each active
    self-attention or MLP branch. The n-stream hidden states are aggregated before each
    branch and expanded back afterwards using learned mappings (H_pre, H_post, H_res).

    Under mHC recompute the layer stack's residual replay replays each half's aggregation, norm
    and write with its block (see ``get_residual_branches``).

    Cross-attention hyper connection is not supported.
    """

    supports_mhc_connections: bool = True

    def __init__(
        self,
        config: TransformerConfig,
        submodules: TransformerLayerSubmodules,
        layer_number: int = 1,
        hidden_dropout: Optional[float] = None,
        pg_collection: Optional[ProcessGroupCollection] = None,
        vp_stage: Optional[int] = None,
        name: Optional[str] = None,
    ) -> None:
        super().__init__(
            config=config,
            submodules=submodules,
            layer_number=layer_number,
            hidden_dropout=hidden_dropout,
            pg_collection=pg_collection,
            vp_stage=vp_stage,
            name=name,
        )

        if submodules.cross_attention_hyper_connection is not IdentityOp:
            raise ValueError(
                "HyperConnectionTransformerLayer does not support cross-attention "
                "hyper connections. Use IdentityOp for cross_attention_hyper_connection."
            )

        has_attention, has_mlp = self._validate_residual_branch_layout()
        if not has_attention and not has_mlp:
            raise ValueError("HyperConnectionTransformerLayer requires an attention or MLP branch.")

        for active, owner_spec, stage in (
            (has_attention, submodules.self_attention_hyper_connection, "self-attention"),
            (has_mlp, submodules.mlp_hyper_connection, "MLP"),
        ):
            if active and owner_spec is IdentityOp:
                raise ValueError(f"The active {stage} branch requires a hyper connection module.")
            if not active and owner_spec is not IdentityOp:
                raise ValueError(
                    f"The inactive {stage} branch requires an identity hyper connection."
                )

        unsupported_moe_cuda_graph_modules = {
            CudaGraphModule.moe,
            CudaGraphModule.moe_router,
            CudaGraphModule.moe_preprocess,
        }
        if self.is_moe_layer and unsupported_moe_cuda_graph_modules.intersection(
            self.config.cuda_graph_modules
        ):
            raise NotImplementedError(
                "HyperConnectionTransformerLayer does not support MoE CUDA graph "
                "scopes. Disable the moe, moe_router, and moe_preprocess CUDA graph modules "
                "when combining mHC with a MoE MLP submodule."
            )

        # GraphableMegatronModule calls create_mcore_cudagraph_manager before the MLP is
        # built, so repeat the local-graph decision now that self.is_moe_layer is authoritative.
        if self.config.cuda_graph_impl == "local":
            self.create_mcore_cudagraph_manager(self.config)

        self.self_attention_hyper_connection = None
        self.residual_read_self_attn = None
        self.residual_write_self_attn = None
        if has_attention:
            self.self_attention_hyper_connection = build_module(
                submodules.self_attention_hyper_connection,
                config=self.config,
                layer_number=self.layer_number,
            )
            self.residual_read_self_attn = HyperConnectionResidualConnection(
                self.config, operation="read", use_input_residual=True
            )
            self.residual_write_self_attn = HyperConnectionResidualConnection(
                self.config, operation="write"
            )

        self.mlp_hyper_connection = None
        self.residual_read_mlp = None
        self.residual_write_mlp = None
        if has_mlp:
            self.mlp_hyper_connection = build_module(
                submodules.mlp_hyper_connection, config=self.config, layer_number=self.layer_number
            )
            self.residual_read_mlp = HyperConnectionResidualConnection(
                self.config, operation="read", use_input_residual=True
            )
            self.residual_write_mlp = HyperConnectionResidualConnection(
                self.config, operation="write"
            )

        # mHC has always bypassed this legacy hook, which cannot carry residual state.
        self._legacy_forward_post_mlp = None

    def activation_strategies(self) -> list:
        """Replaces TransformerLayer's. Under mHC recompute the norms replay with their block (see
        get_residual_branches); the mHC kernels consume the residual, so the norm offload groups
        release nothing."""
        config = self.config
        activation_strategies = []
        if config.recompute_selected("layernorm") and not is_identity(self.input_layernorm):
            activation_strategies.append(
                CheckpointWithoutOutputStrategy(
                    self.input_layernorm, release_on=Event(self._self_attn_bda, "enter")
                )
            )
        if config.recompute_selected("layernorm") and not is_identity(self.pre_mlp_layernorm):
            activation_strategies.append(
                CheckpointWithoutOutputStrategy(
                    self.pre_mlp_layernorm, release_on=Event(self._mlp_bda, "enter")
                )
            )
        if config.offload_selected("attn_norm") and not is_identity(self.input_layernorm):
            activation_strategies.append(
                OffloadStrategy(self.input_layernorm, "attn_norm", commit_on=self._self_attn_bda)
            )
        if config.offload_selected("mlp_norm") and not is_identity(self.pre_mlp_layernorm):
            activation_strategies.append(
                OffloadStrategy(self.pre_mlp_layernorm, "mlp_norm", commit_on=self._mlp_bda)
            )
        if config.recompute_selected("mlp") and not self.is_moe_layer:
            activation_strategies.append(CheckpointStrategy(self.mlp))
        return activation_strategies

    def get_residual_branches(self) -> list[ResidualBranch]:
        """Each active half's mHC read (``aggregate``), its norm and H_post expansion, and its
        write. The layer's last write produces the layer output
        (``_h_res_h_post_bda_layer_output``).
        """
        branches = []
        for is_mlp, owner, norm in (
            (False, self.self_attention_hyper_connection, self.input_layernorm),
            (True, self.mlp_hyper_connection, self.pre_mlp_layernorm),
        ):
            if owner is None:
                continue
            ops = [] if is_identity(norm) else [ReplayRegion(self, norm)]
            ops.append(ReplayRegion(owner, owner._expand_h_post))
            write = (
                owner._h_res_h_post_bda_layer_output
                if self._is_last_residual_write(is_mlp=is_mlp)
                else owner._h_res_h_post_bda
            )
            branches.append(
                ResidualBranch(
                    read=ReplayRegion(owner, owner.aggregate),
                    ops=ops,
                    write=ReplayRegion(owner, write),
                )
            )
        return branches

    def supports_two_stage_attention(self) -> bool:
        """mHC layers run only through ``__call__``, where their replay state enters.

        The split two-stage entry points bypass it, so attention-only mHC layers must not
        advertise them (for example, to Shortcut MoE grouping).
        """
        return False

    def create_mcore_cudagraph_manager(self, config):
        """Create only CUDA graph managers compatible with this mHC layer."""
        # The first call comes from GraphableMegatronModule before TransformerLayer has
        # constructed self.mlp and set self.is_moe_layer.
        if not hasattr(self, "mlp"):
            return

        if self.is_moe_layer:
            # Whole-layer and MLP-scope graphs would capture dynamic MoE dispatch. Mixed
            # models may still request MLP graphs globally, so leave only this MoE layer eager.
            # An MLP-only MoE layer has no attention branch for an attn scope to graph, and
            # the base MLP branch would reject it.
            if (
                not config.cuda_graph_modules
                or (
                    CudaGraphModule.mlp in config.cuda_graph_modules
                    and CudaGraphModule.attn not in config.cuda_graph_modules
                )
                or isinstance(self.self_attention, IdentityOp)
            ):
                return

        super().create_mcore_cudagraph_manager(config)

    def _read_residual_connection(
        self, connection: ResidualConnection, hidden_states: Tensor, *, is_mlp: bool
    ) -> tuple[Tensor, ResidualConnectionState]:
        """Read through the current owner, preserving its activation dtype."""
        owner = self.mlp_hyper_connection if is_mlp else self.self_attention_hyper_connection
        return apply_module(connection)(
            hidden_states, execution_context=HyperConnectionExecutionContext(owner)
        )

    def _write_residual_connection(
        self,
        connection: ResidualConnection,
        branch_output: ResidualBranchOutput,
        state: ResidualConnectionState,
        *,
        is_mlp: bool,
    ) -> Tensor:
        """Write through the current owner within the base BDA gradient scope.

        The layer's last write produces the layer output (see ``get_residual_branches``).
        """
        owner = self.mlp_hyper_connection if is_mlp else self.self_attention_hyper_connection
        return apply_module(connection)(
            branch_output,
            state=state,
            dropout_probability=self.hidden_dropout,
            training=self.training,
            execution_context=HyperConnectionExecutionContext(
                owner,
                self.config.bias_dropout_fusion,
                layer_output=self._is_last_residual_write(is_mlp=is_mlp),
            ),
        )
