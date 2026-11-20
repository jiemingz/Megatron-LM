# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""Mamba layer specialization for streamwise wide-residual connections."""

from torch import Tensor

from megatron.core.activation_policy import ReplayRegion, ResidualBranch
from megatron.core.dist_checkpointing.mapping import ShardedStateDict
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.ssm.mamba_layer import MambaLayer, MambaLayerSubmodules
from megatron.core.transformer.identity_op import is_identity
from megatron.core.transformer.residual_connection import (
    ResidualBranchInput,
    ResidualConnectionState,
)
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.wide_residual import (
    StreamwiseSigmoidWideResidualConnection,
    apply_combined_residual_keys,
    load_residual_state,
)
from megatron.core.typed_torch import apply_module


class WideResidualMambaLayer(MambaLayer):
    """Mamba layer carrying a wide stream around its ordinary-width mixer."""

    supports_wide_residual_connections: bool = True

    def __init__(
        self,
        config: TransformerConfig,
        submodules: MambaLayerSubmodules,
        layer_number: int = 1,
        pg_collection: ProcessGroupCollection = None,
        pp_layer_offset: int = 0,
        name: str | None = None,
        is_mtp_layer: bool = False,
    ) -> None:
        if is_mtp_layer:
            raise ValueError(
                "MTP auxiliary stacks must use ordinary-width MambaLayer, not "
                "WideResidualMambaLayer."
            )
        super().__init__(
            config=config,
            submodules=submodules,
            layer_number=layer_number,
            pg_collection=pg_collection,
            pp_layer_offset=pp_layer_offset,
            name=name,
            is_mtp_layer=is_mtp_layer,
        )

        if config.wide_residual is None:
            raise ValueError("WideResidualMambaLayer requires wide_residual config.")
        if getattr(self.norm, "returns_residual", False):
            raise ValueError(
                "A Mamba residual connection cannot be combined with a layer norm that "
                "returns its own residual."
            )

        self.residual_read = StreamwiseSigmoidWideResidualConnection(
            config, self.layer_number, "mamba", operation="read"
        )
        self.residual_write = StreamwiseSigmoidWideResidualConnection(
            config, self.layer_number, "mamba", operation="write"
        )
        self.register_load_state_dict_pre_hook(load_residual_state)
        self.residual_stream_hidden_size = (
            config.wide_residual.num_streams * self.config.hidden_size
        )
        if self.residual_read.residual_stream_hidden_size != self.residual_stream_hidden_size:
            raise ValueError(
                "The wide-residual Mamba connection must carry num_streams * hidden_size "
                f"features, expected {self.residual_stream_hidden_size}, got "
                f"{self.residual_read.residual_stream_hidden_size}."
            )
        if self.residual_read.branch_hidden_size != self.config.hidden_size:
            raise ValueError(
                "The wide-residual Mamba connection must produce "
                f"hidden_size={self.config.hidden_size}, got "
                f"{self.residual_read.branch_hidden_size}."
            )

    def get_residual_branches(self) -> list[ResidualBranch]:
        """The residual read, the connected input norm and the write around the mixer."""
        ops = () if is_identity(self.norm) else (ReplayRegion(self, self.norm),)
        return [
            ResidualBranch(
                read=ReplayRegion(self.residual_read, self.residual_read._compute_read),
                ops=ops,
                write=ReplayRegion(self.residual_write, self.residual_write._compute_write),
            )
        ]

    def sharded_state_dict(
        self, prefix: str = "", sharded_offsets: tuple = (), metadata: dict | None = None
    ) -> ShardedStateDict:
        """Build the sharded state dict, using combined residual keys when loading them."""
        sharded_state_dict = super().sharded_state_dict(prefix, sharded_offsets, metadata)
        apply_combined_residual_keys(sharded_state_dict, prefix, metadata)
        return sharded_state_dict

    def _get_residual_connection(self):
        """Return the connection surrounding the Mamba mixer."""

        return self.residual_read, self.residual_write

    def _prepare_mixer_state(
        self, hidden_states: Tensor
    ) -> ResidualBranchInput[ResidualConnectionState]:
        """Read the wide stream and normalize the branch input.

        Under residual replay the read and the connected norm replay with their block (see
        :meth:`get_residual_branches`).
        """

        hidden_states, connection_state = apply_module(self.residual_read)(
            hidden_states,
            fp32_residual_connection=self.config.fp32_residual_connection,
            branch_input_dtype=self.config.params_dtype,
        )
        # Validate the carried state before normalization can consume a hook-modified read.
        self.residual_read.residual_stream(connection_state)

        hidden_states = hidden_states.to(dtype=self.config.params_dtype)
        hidden_states = apply_module(self.norm)(hidden_states)
        return ResidualBranchInput(hidden_states, connection_state)

    def _apply_mixer_update(self, mixer_out_with_bias, state: ResidualConnectionState) -> Tensor:
        """Write the mixer update (replayed with its block unless it is the block's boundary)."""

        if state is None:
            raise RuntimeError("Missing state for the Mamba residual connection.")
        # The write replays with its block unless it is the block's live boundary (decided with
        # the layer step's grad mode, so the grad handler below does not affect it).
        with self.bias_dropout_add_exec_handler():
            return apply_module(self.residual_write)(
                mixer_out_with_bias,
                state=state,
                dropout_probability=self.hidden_dropout,
                training=self.training,
            )

    def _pack_two_stage_output(
        self, ssm_output: Tensor, state: ResidualConnectionState
    ) -> tuple[Tensor, Tensor, ResidualConnectionState]:
        """Keep WR's existing two-stage tuple for ShortcutMoE callers."""

        return ssm_output, state[0], state

    def forward_post_core_attn(
        self, ssm_output: Tensor, residual: Tensor, connection_state: ResidualConnectionState
    ) -> Tensor:
        """Project the mixer output and write it to the saved wide-residual stream."""

        mixer_out_with_bias = self.mixer.forward_post_core_attn(ssm_output)
        return self._apply_mixer_update(mixer_out_with_bias, connection_state)
