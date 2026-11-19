# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""The pinned activation buffer moved to
:mod:`megatron.core.activation_policy.offload.pinned_buffer`.

This module is an alias of it, kept so that existing imports (and patches of module
attributes) keep referring to the implementation.
"""

import sys

from megatron.core.activation_policy.offload import pinned_buffer as _impl

sys.modules[__name__] = _impl
