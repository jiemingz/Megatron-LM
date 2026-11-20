# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""Activation recompute: checkpointing and block replay.

* :mod:`.checkpointing`: reentrant checkpoint (``checkpoint``), output-discarding recompute
  (``CheckpointWithoutOutput`` and its manager) and ``precision_aware_checkpoint``.
* :mod:`.residual_recompute`: residual replay, which discards the residual-stream activations of
  a block of layers and replays them together in backward (``ResidualReplayStrategy``).
* :mod:`.segments`: full recompute of the EP A2A overlap schedule, which replays segments of
  layers itself (``RecomputeSegment``).
"""
