# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""Fine-grained activation offloading: the host-copy machinery behind ``OffloadStrategy``.

* :mod:`.fine_grained_activation_offload`: the pipeline offload manager, per-chunk handlers,
  tensor groups and the autograd functions that start and commit offload groups.
* :mod:`.pinned_activation_buffer`: the pinned host buffer the offloaded activations are copied
  into.
* :mod:`.session`: per-chunk offload setup and the end-of-iteration reset.
* :mod:`.te_cpu`: the legacy Transformer Engine CPU offloading setup.
"""
