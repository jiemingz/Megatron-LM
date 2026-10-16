# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Shared diagnostic registry; contains no activation state or layer ownership."""

RESOURCES = {}
KERNELS = {}


def resource(name, kernel):
    KERNELS[name] = kernel
    RESOURCES[name] = dict(
        registers=kernel.n_regs,
        spills=kernel.n_spills,
        shared=kernel.metadata.shared,
        warps=kernel.metadata.num_warps,
        hash=kernel.metadata.hash,
    )
