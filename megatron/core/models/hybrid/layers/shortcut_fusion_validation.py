# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Construction-time contract for the wide-residual shortcut fusion schedule."""

import torch


def validate_shortcut_fusion_config(config):
    """Reject incompatible numerical and scheduling features before building layers."""
    if not config.wide_residual_shortcut_moe_kernel_fusion:
        return
    prefix = "wide_residual_shortcut_moe_kernel_fusion"
    if not config.moe_shortcut_connection or config.wide_residual is None:
        raise ValueError(f"{prefix} requires shortcut MoE and wide_residual")
    streams = config.wide_residual.num_streams
    if type(streams) is not int or not 2 <= streams <= 6:
        raise ValueError(f"{prefix} requires 2 <= wide_residual.num_streams <= 6")
    if not 32 <= config.hidden_size <= 10240 or config.hidden_size % 32:
        raise ValueError(f"{prefix} requires H in [32, 10240], divisible by 32")
    interval = config.residual_stream_recompute_num_layers
    if type(interval) is int and interval > 1 and config.hidden_size == 32:
        raise ValueError(
            f"{prefix} inverse intervals require a native streamwise writer; "
            "H=32 requires residual_stream_recompute_num_layers=1"
        )
    if not config.fp32_residual_connection:
        raise ValueError(
            f"{prefix} requires fp32_residual_connection=True (--fp32-residual-connection)"
        )
    if config.params_dtype != torch.bfloat16:
        raise ValueError(f"{prefix} requires BF16 branch parameters")
    if (
        config.normalization != "RMSNorm"
        or config.layernorm_epsilon != 1e-5
        or config.layernorm_zero_centered_gamma
    ):
        raise ValueError(f"{prefix} requires ordinary RMSNorm with epsilon 1e-5")
    wr = config.wide_residual
    if wr.learned_retention and wr.retention_max_forget != 0.1:
        raise ValueError(f"{prefix} requires retention_max_forget=0.1 when retention is learned")
    # Expert MLP offloading is owned entirely by the existing MoE leaf module.
    # It does not offload tensors saved by the fused residual boundaries.
    expert_only_offload = bool(config.offload_modules) and set(config.offload_modules) <= {
        "fused_group_mlp"
    }
    unsupported = {
        "hidden_dropout": config.hidden_dropout != 0,
        "add_bias_linear": config.add_bias_linear,
        "moe_shortcut_post_norm": config.moe_shortcut_post_norm,
        "moe_shared_expert_overlap": config.moe_shared_expert_overlap,
        "recompute_granularity": config.recompute_granularity not in (None, "selective"),
        "fine_grained_activation_offloading": (
            config.fine_grained_activation_offloading and not expert_only_offload
        ),
        "cpu_offloading": config.cpu_offloading,
        "context_parallel_size > 1": config.context_parallel_size != 1,
    }
    invalid = [name for name, enabled in unsupported.items() if enabled]
    if invalid:
        raise ValueError(f"{prefix} does not support: {', '.join(invalid)}")
    if not config.moe_shared_expert_intermediate_size:
        raise ValueError(f"{prefix} requires a shared expert")
