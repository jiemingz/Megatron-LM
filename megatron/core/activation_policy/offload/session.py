# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""Per-iteration setup and teardown of fine-grained activation offloading.

Models call :func:`prepare_model_chunk` before each forward; pipeline schedules call
:func:`end_iteration` after each forward-backward pass.
"""

from __future__ import annotations

import torch

from megatron.core.activation_policy.offload.fine_grained_activation_offload import (
    FineGrainedActivationOffloadingInterface,
)

_PARAMS_EXCLUDED_ATTR = "_activation_offload_params_excluded"


def prepare_model_chunk(model: torch.nn.Module) -> None:
    """Register ``model`` (one model chunk) with the offload manager before its forward.

    On the first call, the chunk's parameters (decoder, MTP and output layer) are marked as not
    offloadable: only activations are offloaded.
    """
    config = model.config
    interface = FineGrainedActivationOffloadingInterface
    interface.init_chunk_handler(
        pp_rank=model.pg_collection.pp.rank(),
        vp_size=config.virtual_pipeline_model_parallel_size,
        vp_stage=model.vp_stage,
        min_offloaded_tensor_size=config.min_offloaded_tensor_size,
        delta_offload_bytes_across_pp_ranks=config.delta_offload_bytes_across_pp_ranks,
        activation_offload_fraction=config.activation_offload_fraction,
        max_inflight_offloads=config.fine_grained_offloading_max_inflight_offloads,
        buffer_size_gib=config.fine_grained_offloading_buffer_size_gib,
    )
    if model.__dict__.get(_PARAMS_EXCLUDED_ATTR, False):
        return
    modules = [model.decoder]
    if model.mtp_process:
        modules.append(model.mtp)
    if model.post_process:
        modules.append(model.output_layer)
    for module in modules:
        for param in module.parameters():
            interface.mark_not_offload(param)
    model.__dict__[_PARAMS_EXCLUDED_ATTR] = True


def end_iteration(config, pg_collection) -> None:
    """Reset the offload manager after a forward-backward pass, if offloading is enabled.

    With a multi-module (MIMO) process group collection, the language model's group is used, and
    ranks without a language model have nothing to reset.
    """
    if not getattr(config, "fine_grained_activation_offloading", False):
        return
    from megatron.core.process_groups_config import MultiModuleProcessGroupCollection

    if isinstance(pg_collection, MultiModuleProcessGroupCollection):
        if not pg_collection.has_language_model():
            return
        pg_collection = pg_collection.get_language_model_collection()
    FineGrainedActivationOffloadingInterface.reset(process_group=pg_collection.tp_dp_cp)
