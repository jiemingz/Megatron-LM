# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""Per-iteration teardown of fine-grained activation offloading.

Pipeline schedules call :func:`end_offload_iteration` after each forward-backward pass. The
per-chunk setup before each forward is ``strategies.OffloadChunkStrategy``.
"""

from megatron.core.activation_policy.offload.fine_grained_activation_offload import (
    FineGrainedActivationOffloadingInterface,
)


def end_offload_iteration(config, pg_collection) -> None:
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
