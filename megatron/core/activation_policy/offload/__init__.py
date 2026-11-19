# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""Fine-grained activation offloading: the host-copy machinery behind ``OffloadStrategy``.

* :mod:`.manager`: the pipeline offload manager, per-chunk handlers, tensor groups and the
  autograd functions that start and commit offload groups.
* :mod:`.pinned_buffer`: the pinned host buffer the offloaded activations are copied into.
"""
