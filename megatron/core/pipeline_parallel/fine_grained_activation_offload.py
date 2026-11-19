# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""Fine-grained activation offloading moved to
:mod:`megatron.core.activation_policy.offload.manager`.

This module is an alias of it, kept so that existing imports (and patches of module
attributes) keep referring to the implementation.
"""

import sys

from megatron.core.activation_policy.offload import manager as _impl

sys.modules[__name__] = _impl
