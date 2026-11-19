# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""Block replay planning moved to :mod:`megatron.core.activation_policy.recompute.replay`.

This module is an alias of it, kept so that existing imports (and patches of module
attributes) keep referring to the implementation.
"""

import sys

from megatron.core.activation_policy.recompute import replay as _impl

sys.modules[__name__] = _impl
