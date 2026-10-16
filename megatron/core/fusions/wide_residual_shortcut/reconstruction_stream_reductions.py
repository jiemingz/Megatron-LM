# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Controller reductions for independent stream specializations."""

import triton
import triton.language as tl


@triton.jit
def _controller_chunks(
    PART,
    OUT,
    P: tl.constexpr,
    C: tl.constexpr,
    NW: tl.constexpr,
    NT: tl.constexpr,
    N0: tl.constexpr,
    N1: tl.constexpr,
    NUM_STREAMS: tl.constexpr = 3,
    HAS_RETENTION: tl.constexpr = True,
):
    j = tl.program_id(0)
    i = tl.program_id(1) * 1024 + tl.arange(0, 1024)
    logical = j if HAS_RETENTION else tl.where(j < NUM_STREAMS, j, j + NUM_STREAMS)
    active = (
        ((logical < (1 * NUM_STREAMS)) & NW)
        | ((logical >= (1 * NUM_STREAMS)) & (logical < (2 * NUM_STREAMS)) & NT)
        | ((logical >= (2 * NUM_STREAMS)) & (logical < (3 * NUM_STREAMS)) & N0)
        | ((logical >= (3 * NUM_STREAMS)) & N1)
    )
    v = tl.load(PART + j * P + i, (i < P) & active, other=0)
    tl.store(OUT + j * C + tl.program_id(1), tl.sum(v, 0))


@triton.jit
def _finish_controllers(
    PART,
    GW,
    GT,
    G0,
    G1,
    COEFF,
    FINAL_FACTORS: tl.constexpr,
    P: tl.constexpr,
    BLOCK: tl.constexpr,
    LW: tl.constexpr,
    LT: tl.constexpr,
    L0: tl.constexpr,
    L1: tl.constexpr,
    NW: tl.constexpr,
    NT: tl.constexpr,
    N0: tl.constexpr,
    N1: tl.constexpr,
    NUM_STREAMS: tl.constexpr = 3,
    HAS_RETENTION: tl.constexpr = True,
):
    s = tl.program_id(0)
    i = tl.arange(0, BLOCK)
    if NW:
        factor = tl.load(COEFF + (4 * NUM_STREAMS) + s) if FINAL_FACTORS else 1.0
        tl.store(GW + s, tl.sum(tl.load(PART + s * P + i, i < P, other=0)) * factor)
        tl.store(GW + i + (1 * NUM_STREAMS), 0, (s == 0) & (i + (1 * NUM_STREAMS) < LW))
    if NT:
        factor = tl.load(COEFF + (5 * NUM_STREAMS) + s) if FINAL_FACTORS else 1.0
        tl.store(
            GT + s, tl.sum(tl.load(PART + ((1 * NUM_STREAMS) + s) * P + i, i < P, other=0)) * factor
        )
        tl.store(GT + i + (1 * NUM_STREAMS), 0, (s == 0) & (i + (1 * NUM_STREAMS) < LT))
    if N0:
        factor = tl.load(COEFF + (6 * NUM_STREAMS) + s) if FINAL_FACTORS else 1.0
        tl.store(
            G0 + s,
            tl.sum(
                tl.load(PART + (((1 + HAS_RETENTION) * NUM_STREAMS) + s) * P + i, i < P, other=0)
            )
            * factor,
        )
        tl.store(G0 + i + (1 * NUM_STREAMS), 0, (s == 0) & (i + (1 * NUM_STREAMS) < L0))
    if N1:
        factor = tl.load(COEFF + (7 * NUM_STREAMS) + s) if FINAL_FACTORS else 1.0
        tl.store(
            G1 + s,
            tl.sum(
                tl.load(PART + (((2 + HAS_RETENTION) * NUM_STREAMS) + s) * P + i, i < P, other=0)
            )
            * factor,
        )
        tl.store(G1 + i + (1 * NUM_STREAMS), 0, (s == 0) & (i + (1 * NUM_STREAMS) < L1))
