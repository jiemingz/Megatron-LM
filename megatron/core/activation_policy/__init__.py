# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""Activation memory strategies (recompute, replay, checkpoint, offload) applied at points.

Each module returns its :mod:`.strategies` from ``MegatronModule.activation_strategies``, built
from its config (``TransformerConfig.recompute_selected`` / ``offload_selected``); they are
installed once the module is constructed. A strategy acts on a *point*: a child module or a method
of its owner, hooked transparently by ``megatron.core.transformer.module``; model code just calls
its submodules and methods. Layer stacks return the strategies that span layers (full recompute,
residual replay), which hook their layers' points.

The mechanisms behind the strategies live in two subpackages: :mod:`.recompute` (checkpointing,
output-discarding recompute and block replay) and :mod:`.offload` (fine-grained activation
offloading).

Attributes are imported on first use: ``megatron.core.tensor_parallel`` re-exports
:mod:`.recompute.checkpointing`, so importing this package must not import model code.
"""

import importlib

_EXPORTS = {
    "Event": "strategies",
    "ReplayRegion": "recompute.replay",
    "ResidualBranch": "recompute.replay",
    "ResidualReplayStrategy": "recompute.replay",
    "TensorRef": "strategies",
    "CheckpointStrategy": "strategies",
    "CheckpointWithoutOutputStrategy": "strategies",
    "ExcludeFromOffloadStrategy": "strategies",
    "LayerWiseCheckpointStrategy": "strategies",
    "LayerCPUOffloadStrategy": "strategies",
    "OffloadStrategy": "strategies",
    "Strategy": "strategies",
    "has_strategy": "runtime",
    "install_activation_strategies": "runtime",
    "installed_strategies": "runtime",
    "mark_borrowed": "strategies",
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
