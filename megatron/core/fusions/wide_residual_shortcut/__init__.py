# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Fused wide-residual shortcut boundaries with call-local autograd state.

This package is the canonical implementation. Standalone production benchmarks
alias these modules; they do not maintain copies of the kernels. Shared Triton
arithmetic lives in helpers/reconstruction_streams, and shared Gluon arithmetic
in gluon_helpers. Keep scheduling and synchronization in the owning kernel.
"""

from .api import boundary_norm, validate_input, write_read_norm

__all__ = ["boundary_norm", "validate_input", "write_read_norm"]
