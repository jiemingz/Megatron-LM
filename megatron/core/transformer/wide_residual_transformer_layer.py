# Copyright (c) 2025-2026, NVIDIA CORPORATION. All rights reserved.
"""Transformer adapter for wide-residual read/write connections."""

from __future__ import annotations

from typing import Optional

from megatron.core.activation_policy import ReplayRegion, ResidualBranch
from megatron.core.dist_checkpointing.mapping import ShardedStateDict
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer.identity_op import is_identity
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.transformer_layer import TransformerLayer, TransformerLayerSubmodules
from megatron.core.transformer.wide_residual import (
    StreamwiseSigmoidWideResidualConnection,
    apply_combined_residual_keys,
    load_residual_state,
)


class WideResidualTransformerLayer(TransformerLayer):
    """Transformer layer carrying a wide stream around ordinary-width branches."""

    supports_wide_residual_connections: bool = True

    def __init__(
        self,
        config: TransformerConfig,
        submodules: TransformerLayerSubmodules,
        layer_number: int = 1,
        hidden_dropout: Optional[float] = None,
        pg_collection: Optional[ProcessGroupCollection] = None,
        vp_stage: Optional[int] = None,
        is_mtp_layer: bool = False,
        add_layer_offset: bool = True,
        pp_layer_offset: Optional[int] = None,
        hash_moe_layer_threshold: Optional[int] = None,
        name: str | None = None,
    ) -> None:
        if is_mtp_layer:
            raise ValueError(
                "MTP auxiliary stacks must use ordinary-width TransformerLayer, not "
                "WideResidualTransformerLayer."
            )
        super().__init__(
            config=config,
            submodules=submodules,
            layer_number=layer_number,
            hidden_dropout=hidden_dropout,
            pg_collection=pg_collection,
            vp_stage=vp_stage,
            is_mtp_layer=is_mtp_layer,
            add_layer_offset=add_layer_offset,
            pp_layer_offset=pp_layer_offset,
            hash_moe_layer_threshold=hash_moe_layer_threshold,
            name=name,
        )

        if config.wide_residual is None:
            raise ValueError("WideResidualTransformerLayer requires wide_residual config.")
        has_attention, has_mlp = self._validate_residual_branch_layout()

        self.residual_read_self_attn = None
        self.residual_write_self_attn = None
        if has_attention:
            self.residual_read_self_attn = StreamwiseSigmoidWideResidualConnection(
                config, self.layer_number, "self_attention", operation="read"
            )
            self.residual_write_self_attn = StreamwiseSigmoidWideResidualConnection(
                config, self.layer_number, "self_attention", operation="write"
            )

        self.residual_read_mlp = None
        self.residual_write_mlp = None
        if has_mlp:
            self.residual_read_mlp = StreamwiseSigmoidWideResidualConnection(
                config, self.layer_number, "mlp", operation="read"
            )
            self.residual_write_mlp = StreamwiseSigmoidWideResidualConnection(
                config, self.layer_number, "mlp", operation="write"
            )

        self.register_load_state_dict_pre_hook(load_residual_state)
        residual_connections = [
            reader
            for reader in (self.residual_read_self_attn, self.residual_read_mlp)
            if reader is not None
        ]

        if self.residual_read_self_attn is not None and self._input_layernorm_returns_residual:
            raise ValueError(
                "A self-attention residual connection cannot be combined with a layer norm "
                "that returns its own residual."
            )
        if self.residual_read_mlp is not None and self._pre_mlp_layernorm_returns_residual:
            raise ValueError(
                "An MLP residual connection cannot be combined with a layer norm that returns "
                "its own residual."
            )
        if residual_connections and self.config.inference_fuse_tp_communication:
            raise NotImplementedError(
                "Wide-residual connections do not support fused tensor-parallel inference."
            )

        self.residual_stream_hidden_size = (
            config.wide_residual.num_streams * self.config.hidden_size
        )
        for connection in residual_connections:
            if connection.residual_stream_hidden_size != self.residual_stream_hidden_size:
                raise ValueError(
                    "All wide-residual connections must carry num_streams * hidden_size "
                    f"features, expected {self.residual_stream_hidden_size}, got "
                    f"{connection.residual_stream_hidden_size}."
                )
            if connection.branch_hidden_size != self.config.hidden_size:
                raise ValueError(
                    "Wide-residual connections around TransformerLayer branches must produce "
                    f"hidden_size={self.config.hidden_size}, got "
                    f"{connection.branch_hidden_size}."
                )

    def _connected_norms_replayed(self) -> bool:
        """Residual-stream recompute replays the connected norms (see get_residual_branches)."""
        return self.config.residual_stream_replay()

    def get_residual_branches(self) -> list[ResidualBranch]:
        """The connected halves' residual read, norm and write (see ``ResidualReplayStrategy``)."""
        branches = []
        for get_connection, norm, norm_returns_residual in (
            (
                self._get_residual_connection_for_self_attention,
                self.input_layernorm,
                self._input_layernorm_returns_residual,
            ),
            (
                self._get_residual_connection_for_mlp,
                self.pre_mlp_layernorm,
                self._pre_mlp_layernorm_returns_residual,
            ),
        ):
            read, write = get_connection("read"), get_connection("write")
            if read is None:
                continue
            ops = () if is_identity(norm) else (ReplayRegion(self, norm, norm_returns_residual),)
            branches.append(
                ResidualBranch(
                    read=ReplayRegion(read, read._compute_read),
                    ops=ops,
                    write=ReplayRegion(write, write._compute_write) if write is not None else None,
                )
            )
        return branches

    def sharded_state_dict(
        self, prefix: str = "", sharded_offsets: tuple = (), metadata: Optional[dict] = None
    ) -> ShardedStateDict:
        """Build the sharded state dict, using combined residual keys when loading them."""
        sharded_state_dict = super().sharded_state_dict(prefix, sharded_offsets, metadata)
        apply_combined_residual_keys(sharded_state_dict, prefix, metadata)
        return sharded_state_dict
