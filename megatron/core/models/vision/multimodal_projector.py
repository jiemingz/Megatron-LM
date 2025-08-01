# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
from typing import Optional

import torch

from megatron.core.transformer.mlp import MLP, MLPSubmodules
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.spec_utils import build_module
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.utils import make_viewless_tensor


class MultimodalProjector(MegatronModule):
    """
    MultimodalProjector will take the encoded input with input_size hidden state and project
    it into the hidden size of the language model for multimodal training. When projector is
    type affine linear_fc1 from submodules is used.

    Args:
        transformer_config (TransformerConfig): Transformer config
        submodules (MLPSubmodules): Specifies MLP submodules for mlp type projector
        projector_type (str): Projector type
        input_size (int): Input size from feature encoder
        tp_group (torch.distributed.ProcessGroup): Tensor parallel group
    """

    def __init__(
        self,
        config: TransformerConfig,
        submodules: MLPSubmodules,
        projector_type: str,
        input_size: int,
        tp_group: Optional[torch.distributed.ProcessGroup] = None,
    ):
        super().__init__(config=config)
        self.projector_type = projector_type

        assert submodules is not None, "MLPSubmodules must be provided"

        if self.projector_type == "mlp":
            self.encoder = MLP(
                config=config, submodules=submodules, input_size=input_size, tp_group=tp_group
            )
        elif self.projector_type == "affine":
            self.encoder = build_module(
                submodules.linear_fc1,
                input_size,
                config.hidden_size,
                config=config,
                init_method=config.init_method,
                gather_output=True,
                bias=config.add_bias_linear,
                skip_bias_add=True,
                is_expert=False,
                tp_comm_buffer_name=None,
                tp_group=tp_group,
            )
        else:
            raise Exception(f"Unsupported multimodal projection type {self.projector_type}")

    def forward(self, hidden_states):
        """Run multimodal projector.

        Args:
            hidden_states (torch.Tensor): Input.

        Returns:
            torch.Tensor: The projected output.
        """
        # Run encoder.
        encoder_output, encoder_output_bias = self.encoder(hidden_states)

        if encoder_output_bias is not None:
            encoder_output = encoder_output + encoder_output_bias

        # the encoder produces "viewed" tensor. This will result in schedule.py's
        # deallocate_output_tensor() throwing an error, so a viewless tensor is
        # created to prevent this.
        encoder_output = make_viewless_tensor(
            inp=encoder_output, requires_grad=True, keep_graph=True
        )

        return encoder_output
