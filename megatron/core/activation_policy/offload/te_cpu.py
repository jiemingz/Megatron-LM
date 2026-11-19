# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""Legacy layer-granularity CPU offloading through Transformer Engine (``cpu_offloading``).

A block wraps each layer in TE's offload context and commits the layer's group after it (a
``LayerCPUOffloadStrategy`` on each of its layers). While the context is active, linear layers
without TE mark their inputs with :func:`mark_activation`.
"""

from __future__ import annotations

from typing import Any, Callable, NamedTuple, Optional

import torch


class TECPUOffload(NamedTuple):
    """Transformer Engine's layer-granularity CPU offload context and per-layer commit.

    ``commit`` is ``None`` when CPU offloading is off (the context is then a no-op).
    """

    context: Any
    commit: Optional[Callable[[torch.Tensor], torch.Tensor]]


def build(config) -> tuple[Optional[TECPUOffload], Any]:
    """Create the offload context a block shares among its layers.

    Returns the block's :class:`TECPUOffload` (``None`` without TE) and the context to publish
    as ``config._cpu_offloading_context`` (``None`` when CPU offloading is off).
    """
    try:
        from megatron.core.extensions.transformer_engine import get_cpu_offload_context
    except ImportError:
        get_cpu_offload_context = None
    if get_cpu_offload_context is None:
        assert config.cpu_offloading is False, "CPU Offloading is enabled when TE is not present"
        return None, None
    context, commit = get_cpu_offload_context(
        config.cpu_offloading,
        config.cpu_offloading_num_layers,
        config.num_layers,
        config.cpu_offloading_activations,
        config.cpu_offloading_weights,
        config.cpu_offloading_double_buffering,
        config.cpu_offloading_retain_pinned_cpu_buffers,
    )
    published = context if config.cpu_offloading else None
    return TECPUOffload(context, commit if config.cpu_offloading else None), published


def mark_activation(config, tensor: torch.Tensor, have_te: bool) -> None:
    """Inside the published offload context, mark a linear layer input for TE's offload."""
    context = config._cpu_offloading_context
    if context is None or context.inside_context is not True:
        return
    if not have_te:
        assert (
            config.cpu_offloading is False
        ), "CPU Offloading cannot be enabled while TE is not present"
    else:
        tensor.activation_offloading = config.cpu_offloading_activations
