# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""Activation recompute: checkpointing and block replay.

* :mod:`.checkpointing`: reentrant checkpoint (:func:`checkpoint`), output-discarding recompute
  (:class:`CheckpointWithoutOutput` and its manager) and :func:`precision_aware_checkpoint`.
* :mod:`.replay`: residual replay, which discards the residual-stream activations of a block of
  layers and replays them together in backward (:class:`ResidualReplayStrategy`).
* :mod:`.segments`: full recompute of the EP A2A overlap schedule, which replays segments of
  layers itself (:class:`RecomputeSegment`).

Attributes are imported on first use: ``megatron.core.tensor_parallel`` re-exports
:mod:`.checkpointing`, so importing this package must not import anything else.
"""

import importlib

_EXPORTS = {
    "CheckpointFunction": "checkpointing",
    "CheckpointWithoutOutput": "checkpointing",
    "CheckpointWithoutOutputManager": "checkpointing",
    "checkpoint": "checkpointing",
    "is_checkpoint_without_output_tensor": "checkpointing",
    "is_checkpointing": "checkpointing",
    "precision_aware_checkpoint": "checkpointing",
    "ReplayRegion": "replay",
    "ResidualBranch": "replay",
    "ResidualReplayStrategy": "replay",
    "build_recompute_block_end_plan": "replay",
}

__all__ = sorted(_EXPORTS)


def __getattr__(name):
    module = _EXPORTS.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(importlib.import_module(f"{__name__}.{module}"), name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(_EXPORTS))
