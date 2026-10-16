# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Offline B schedule geometry and dispatch. No CUDA imports or runtime search.

Schedules are published only after two-process coverage, flagship controls,
numerical checks and unfiltered sanitizer checks pass.
"""

from dataclasses import dataclass

WIDTHS = tuple(range(2048, 10241, 1024))
SHAPES = tuple(
    [(m, h) for m in range(4096, 16385, 1024) for h in WIDTHS]
    + [(m, h) for m in range(17408, 65537, 1024) for h in WIDTHS[:3]]
)
FLAGSHIPS = ((16384, 3072), (8192, 4096), (8192, 10240))


def qualified_device(properties):
    """The measured GB300 residency is not portable to every SM103 device."""
    return (
        "GB300" in properties.name
        and properties.major == 10
        and properties.minor == 3
        and properties.multi_processor_count == 152
    )


@dataclass(frozen=True)
class Schedule:
    grid: int
    projection_layout: str = "rows4v4"
    pp_tile: int = 0
    carry_rows: int = 0
    before_wait: bool = False

    def validate(self, hidden_size):
        participants = hidden_size // 128
        if hidden_size not in WIDTHS:
            raise ValueError("cooperative backward requires H=2048..10240, step 1024")
        if type(self.grid) is not int or self.grid <= 0 or self.grid % participants:
            raise ValueError("grid must contain positive whole participant groups")
        if self.projection_layout not in ("rows2v2", "rows4v2", "rows4v4"):
            raise ValueError("unknown projection layout")
        if self.pp_tile not in (0, 8, 16) or (self.pp_tile and participants % self.pp_tile):
            raise ValueError("PP tile must divide the real participant count")
        if self.carry_rows not in (0, 16, 32) or type(self.before_wait) is not bool:
            raise ValueError("invalid carry prefetch schedule")
        if self.before_wait and not self.carry_rows:
            raise ValueError("before_wait requires a carry prefetch")
        return self

    def as_tuple(self):
        return (self.grid, self.projection_layout, self.pp_tile, self.carry_rows, self.before_wait)


INCUMBENTS = {
    3072: Schedule(456, "rows4v4", 0, 0, False),
    4096: Schedule(448, "rows2v2", 0, 16, False),
    10240: Schedule(400, "rows4v2", 16, 32, True),
}

# (CUDA capability, local M, local H) -> Schedule. Filled by the qualified
# static table below; production performs no search or profiling.
QUALIFIED_BACKWARD = {}
QUALIFIED_FORWARD = {}
# Remain empty until independent BF16 qualification passes.
QUALIFIED_BF16_BACKWARD = {}
QUALIFIED_BF16_FORWARD = {}


@dataclass(frozen=True)
class ForwardSchedule:
    mix_rows: int = 8
    mix_cols: int = 128
    mix_warps: int = 4
    norm_warps: int = 4
    skip_scale_reset: bool = False

    def __post_init__(self):
        if self.mix_rows not in (4, 8, 16, 32) or self.mix_cols not in (64, 128, 256):
            raise ValueError("invalid forward mix tile")
        if self.mix_warps not in (4, 8) or self.norm_warps not in (4, 8, 16):
            raise ValueError("invalid forward warp count")
        if type(self.skip_scale_reset) is not bool:
            raise ValueError("skip_scale_reset must be Boolean")


QUALIFIED_STREAM_BACKWARD = {}
QUALIFIED_STREAM_FORWARD = {}


class CooperativeEligibilityError(ValueError):
    """A compiled generalized schedule fails the cooperative resource policy."""


def stream_cooperative_rejection(resources, grid):
    """Conservative host policy for the independent generalized kernel.

    Keeping S residual planes can halve residency. Require two CTAs per SM
    before admitting a new specialization; do not redesign its buffer schedule
    to recover occupancy. The incumbent S=3 implementation does not use this.
    Actual compiled resources, rather than a stream-count cutoff, own the gate.
    """
    if grid > resources["limit"]:
        return f"grid {grid} exceeds cooperative residency: {resources}"
    if resources["blocks_per_sm"] < 2:
        return (
            "generalized cooperative backward requires at least two resident "
            f"CTAs per SM; use standard backward: {resources}"
        )
    return None


def resolve_backward(shape, capability, *, operand_format="mxfp8", num_streams=3):
    if type(num_streams) is not int or not 1 <= num_streams <= 6:
        raise ValueError("num_streams must be an integer from 1 through 6")
    if operand_format not in ("mxfp8", "bf16"):
        raise ValueError("operand_format must be mxfp8 or bf16")
    shape = tuple(shape)
    if num_streams != 3:
        return QUALIFIED_STREAM_BACKWARD.get(
            (tuple(capability), *shape, num_streams, operand_format)
        )
    if tuple(capability) != (10, 3) or shape not in SHAPES:
        return None
    try:
        table = QUALIFIED_BF16_BACKWARD if operand_format == "bf16" else QUALIFIED_BACKWARD
        return table[(tuple(capability), *shape)]
    except KeyError:
        raise ValueError(
            f"B backward {operand_format} shape {shape} has no offline GB300 qualification"
        ) from None


def candidate_grids(m, h, resident_limit):
    """Use residency from an actual-shape build; recheck each compiled grid.

    Nearest whole groups to 75% and 100%, plus the largest resident group
    count dividing the number of 32-row quantization tiles.
    """
    if h not in WIDTHS or m <= 0 or m % 32 or resident_limit < h // 128:
        raise ValueError("invalid shape or insufficient residency for one group")
    nt, tiles = h // 128, m // 32
    maximum = resident_limit // nt
    nearest = min(range(1, maximum + 1), key=lambda n: (abs(n * nt - 0.75 * resident_limit), -n))
    divisor = max(n for n in range(1, maximum + 1) if tiles % n == 0)
    return tuple(sorted({nt * nearest, nt * maximum, nt * divisor}))


def candidate_schedules(m, h, resident_limit):
    """Offline screen only; these candidates are never automatic selections."""
    for grid in candidate_grids(m, h, resident_limit):
        for layout in ("rows2v2", "rows4v2", "rows4v4"):
            for pp in (0, 8, 16):
                if pp and (h // 128) % pp:
                    continue
                for rows, before in ((0, False), (16, False), (16, True), (32, False), (32, True)):
                    yield Schedule(grid, layout, pp, rows, before).validate(h)


# BEGIN QUALIFIED GB300 TABLE
# Two-process evidence: results/shape_coverage_20261014/REPORT.txt
_GB300_BACKWARD_BY_WIDTH = {
    2048: Schedule(448, "rows4v4", 0, 16, True),
    3072: Schedule(456, "rows4v4", 0, 0, False),
    4096: Schedule(448, "rows2v2", 0, 16, False),
    5120: Schedule(440, "rows4v2", 8, 32, True),
    6144: Schedule(432, "rows4v2", 16, 32, True),
    7168: Schedule(448, "rows4v2", 8, 32, True),
    8192: Schedule(448, "rows4v2", 16, 32, True),
    9216: Schedule(432, "rows4v2", 8, 32, True),
    10240: Schedule(400, "rows4v2", 16, 32, True),
}
QUALIFIED_BACKWARD.update({((10, 3), m, h): _GB300_BACKWARD_BY_WIDTH[h] for m, h in SHAPES})
QUALIFIED_FORWARD.update(
    {
        ((10, 3), m, h): (
            ForwardSchedule() if (m, h) in FLAGSHIPS else ForwardSchedule(4, 256, 4, 4, True)
        )
        for m, h in SHAPES
    }
)

# BEGIN QUALIFIED BF16 TABLE
# Evidence: results/bf16_coverage_20261014/REPORT.txt
_BF16_BACKWARD_GROUPS = (
    (
        Schedule(
            grid=400, projection_layout="rows4v2", pp_tile=16, carry_rows=32, before_wait=True
        ),
        (
            (4096, 10240),
            (5120, 10240),
            (6144, 10240),
            (7168, 10240),
            (8192, 10240),
            (9216, 10240),
            (10240, 10240),
            (11264, 10240),
            (12288, 10240),
            (13312, 10240),
            (14336, 10240),
            (15360, 10240),
            (16384, 10240),
        ),
    ),
    (
        Schedule(
            grid=432, projection_layout="rows4v2", pp_tile=16, carry_rows=32, before_wait=True
        ),
        (
            (4096, 6144),
            (5120, 6144),
            (6144, 6144),
            (7168, 6144),
            (8192, 6144),
            (9216, 6144),
            (10240, 6144),
            (11264, 6144),
            (12288, 6144),
            (13312, 6144),
            (14336, 6144),
            (15360, 6144),
            (16384, 6144),
        ),
    ),
    (
        Schedule(grid=432, projection_layout="rows4v2", pp_tile=8, carry_rows=32, before_wait=True),
        (
            (4096, 9216),
            (5120, 9216),
            (6144, 9216),
            (7168, 9216),
            (8192, 9216),
            (9216, 9216),
            (10240, 9216),
            (11264, 9216),
            (12288, 9216),
            (13312, 9216),
            (14336, 9216),
            (15360, 9216),
            (16384, 9216),
        ),
    ),
    (
        Schedule(grid=440, projection_layout="rows4v2", pp_tile=8, carry_rows=32, before_wait=True),
        (
            (4096, 5120),
            (5120, 5120),
            (6144, 5120),
            (7168, 5120),
            (8192, 5120),
            (9216, 5120),
            (10240, 5120),
            (11264, 5120),
            (12288, 5120),
            (13312, 5120),
            (14336, 5120),
            (15360, 5120),
            (16384, 5120),
        ),
    ),
    (
        Schedule(
            grid=448, projection_layout="rows2v2", pp_tile=0, carry_rows=16, before_wait=False
        ),
        (
            (4096, 4096),
            (5120, 4096),
            (6144, 4096),
            (7168, 4096),
            (8192, 4096),
            (9216, 4096),
            (10240, 4096),
            (11264, 4096),
            (12288, 4096),
            (13312, 4096),
            (14336, 4096),
            (15360, 4096),
            (16384, 4096),
            (17408, 4096),
            (18432, 4096),
            (19456, 4096),
            (20480, 4096),
            (21504, 4096),
            (22528, 4096),
            (23552, 4096),
            (24576, 4096),
            (25600, 4096),
            (26624, 4096),
            (27648, 4096),
            (28672, 4096),
            (29696, 4096),
            (30720, 4096),
            (31744, 4096),
            (32768, 4096),
            (33792, 4096),
            (34816, 4096),
            (35840, 4096),
            (36864, 4096),
            (37888, 4096),
            (38912, 4096),
            (39936, 4096),
            (40960, 4096),
            (41984, 4096),
            (43008, 4096),
            (44032, 4096),
            (45056, 4096),
            (46080, 4096),
            (47104, 4096),
            (48128, 4096),
            (49152, 4096),
            (50176, 4096),
            (51200, 4096),
            (52224, 4096),
            (53248, 4096),
            (54272, 4096),
            (55296, 4096),
            (56320, 4096),
            (57344, 4096),
            (58368, 4096),
            (59392, 4096),
            (60416, 4096),
            (61440, 4096),
            (62464, 4096),
            (63488, 4096),
            (64512, 4096),
            (65536, 4096),
        ),
    ),
    (
        Schedule(
            grid=448, projection_layout="rows4v2", pp_tile=16, carry_rows=32, before_wait=True
        ),
        (
            (4096, 8192),
            (5120, 8192),
            (6144, 8192),
            (7168, 8192),
            (8192, 8192),
            (9216, 8192),
            (10240, 8192),
            (11264, 8192),
            (12288, 8192),
            (13312, 8192),
            (14336, 8192),
            (15360, 8192),
            (16384, 8192),
        ),
    ),
    (
        Schedule(grid=448, projection_layout="rows4v2", pp_tile=8, carry_rows=32, before_wait=True),
        (
            (4096, 7168),
            (5120, 7168),
            (6144, 7168),
            (7168, 7168),
            (8192, 7168),
            (9216, 7168),
            (10240, 7168),
            (11264, 7168),
            (12288, 7168),
            (13312, 7168),
            (14336, 7168),
            (15360, 7168),
            (16384, 7168),
        ),
    ),
    (
        Schedule(grid=448, projection_layout="rows4v4", pp_tile=0, carry_rows=16, before_wait=True),
        (
            (4096, 2048),
            (5120, 2048),
            (6144, 2048),
            (7168, 2048),
            (8192, 2048),
            (9216, 2048),
            (10240, 2048),
            (11264, 2048),
            (12288, 2048),
            (13312, 2048),
            (14336, 2048),
            (15360, 2048),
            (16384, 2048),
            (17408, 2048),
            (18432, 2048),
            (19456, 2048),
            (20480, 2048),
            (21504, 2048),
            (22528, 2048),
            (23552, 2048),
            (24576, 2048),
            (25600, 2048),
            (26624, 2048),
            (27648, 2048),
            (28672, 2048),
            (29696, 2048),
            (30720, 2048),
            (31744, 2048),
            (32768, 2048),
            (33792, 2048),
            (34816, 2048),
            (35840, 2048),
            (36864, 2048),
            (37888, 2048),
            (38912, 2048),
            (39936, 2048),
            (40960, 2048),
            (41984, 2048),
            (43008, 2048),
            (44032, 2048),
            (45056, 2048),
            (46080, 2048),
            (47104, 2048),
            (48128, 2048),
            (49152, 2048),
            (50176, 2048),
            (51200, 2048),
            (52224, 2048),
            (53248, 2048),
            (54272, 2048),
            (55296, 2048),
            (56320, 2048),
            (57344, 2048),
            (58368, 2048),
            (59392, 2048),
            (60416, 2048),
            (61440, 2048),
            (62464, 2048),
            (63488, 2048),
            (64512, 2048),
            (65536, 2048),
        ),
    ),
    (
        Schedule(grid=456, projection_layout="rows4v4", pp_tile=0, carry_rows=0, before_wait=False),
        (
            (4096, 3072),
            (5120, 3072),
            (6144, 3072),
            (7168, 3072),
            (8192, 3072),
            (9216, 3072),
            (10240, 3072),
            (11264, 3072),
            (12288, 3072),
            (13312, 3072),
            (14336, 3072),
            (15360, 3072),
            (16384, 3072),
            (17408, 3072),
            (18432, 3072),
            (19456, 3072),
            (20480, 3072),
            (21504, 3072),
            (22528, 3072),
            (23552, 3072),
            (24576, 3072),
            (25600, 3072),
            (26624, 3072),
            (27648, 3072),
            (28672, 3072),
            (29696, 3072),
            (30720, 3072),
            (31744, 3072),
            (32768, 3072),
            (33792, 3072),
            (34816, 3072),
            (35840, 3072),
            (36864, 3072),
            (37888, 3072),
            (38912, 3072),
            (39936, 3072),
            (40960, 3072),
            (41984, 3072),
            (43008, 3072),
            (44032, 3072),
            (45056, 3072),
            (46080, 3072),
            (47104, 3072),
            (48128, 3072),
            (49152, 3072),
            (50176, 3072),
            (51200, 3072),
            (52224, 3072),
            (53248, 3072),
            (54272, 3072),
            (55296, 3072),
            (56320, 3072),
            (57344, 3072),
            (58368, 3072),
            (59392, 3072),
            (60416, 3072),
            (61440, 3072),
            (62464, 3072),
            (63488, 3072),
            (64512, 3072),
            (65536, 3072),
        ),
    ),
)
QUALIFIED_BF16_BACKWARD.update(
    {((10, 3), m, h): schedule for schedule, shapes in _BF16_BACKWARD_GROUPS for m, h in shapes}
)
_BF16_FORWARD_GROUPS = (
    (
        ForwardSchedule(mix_rows=4, mix_cols=128, mix_warps=4, norm_warps=4, skip_scale_reset=True),
        (
            (4096, 7168),
            (6144, 8192),
            (11264, 3072),
            (11264, 4096),
            (11264, 7168),
            (13312, 4096),
            (14336, 4096),
            (16384, 6144),
            (16384, 8192),
            (21504, 4096),
            (24576, 4096),
            (25600, 4096),
            (26624, 4096),
            (27648, 4096),
            (29696, 4096),
            (32768, 4096),
            (35840, 4096),
            (36864, 2048),
            (38912, 2048),
            (39936, 3072),
            (40960, 3072),
            (40960, 4096),
            (41984, 4096),
            (44032, 4096),
            (47104, 3072),
            (48128, 3072),
            (49152, 3072),
            (49152, 4096),
            (50176, 3072),
            (52224, 3072),
            (54272, 4096),
            (57344, 3072),
            (58368, 4096),
            (59392, 3072),
            (60416, 4096),
            (62464, 4096),
            (65536, 3072),
            (65536, 4096),
        ),
    ),
    (
        ForwardSchedule(mix_rows=4, mix_cols=256, mix_warps=4, norm_warps=4, skip_scale_reset=True),
        (
            (4096, 2048),
            (4096, 3072),
            (4096, 4096),
            (4096, 5120),
            (4096, 6144),
            (4096, 9216),
            (4096, 10240),
            (5120, 2048),
            (5120, 3072),
            (5120, 4096),
            (5120, 5120),
            (5120, 6144),
            (5120, 7168),
            (5120, 9216),
            (5120, 10240),
            (6144, 2048),
            (6144, 3072),
            (6144, 4096),
            (6144, 5120),
            (6144, 6144),
            (6144, 7168),
            (6144, 9216),
            (6144, 10240),
            (7168, 2048),
            (7168, 3072),
            (7168, 4096),
            (7168, 5120),
            (7168, 6144),
            (7168, 7168),
            (7168, 9216),
            (7168, 10240),
            (8192, 2048),
            (8192, 3072),
            (8192, 4096),
            (8192, 5120),
            (8192, 6144),
            (8192, 7168),
            (8192, 8192),
            (8192, 9216),
            (8192, 10240),
            (9216, 2048),
            (9216, 3072),
            (9216, 4096),
            (9216, 5120),
            (9216, 6144),
            (9216, 7168),
            (9216, 8192),
            (9216, 9216),
            (9216, 10240),
            (10240, 2048),
            (10240, 3072),
            (10240, 4096),
            (10240, 5120),
            (10240, 6144),
            (10240, 7168),
            (10240, 8192),
            (10240, 9216),
            (10240, 10240),
            (11264, 2048),
            (11264, 5120),
            (11264, 6144),
            (11264, 10240),
            (12288, 2048),
            (12288, 3072),
            (12288, 4096),
            (12288, 5120),
            (12288, 6144),
            (12288, 7168),
            (12288, 8192),
            (12288, 9216),
            (12288, 10240),
            (13312, 2048),
            (13312, 3072),
            (13312, 5120),
            (13312, 6144),
            (13312, 7168),
            (13312, 8192),
            (13312, 9216),
            (13312, 10240),
            (14336, 2048),
            (14336, 3072),
            (14336, 5120),
            (14336, 6144),
            (14336, 7168),
            (14336, 8192),
            (14336, 9216),
            (14336, 10240),
            (15360, 2048),
            (15360, 3072),
            (15360, 4096),
            (15360, 5120),
            (15360, 6144),
            (15360, 7168),
            (15360, 8192),
            (15360, 9216),
            (15360, 10240),
            (16384, 2048),
            (16384, 3072),
            (16384, 4096),
            (16384, 5120),
            (16384, 7168),
            (16384, 9216),
            (16384, 10240),
            (17408, 2048),
            (17408, 3072),
            (17408, 4096),
            (18432, 2048),
            (18432, 3072),
            (19456, 2048),
            (19456, 3072),
            (20480, 2048),
            (20480, 3072),
            (20480, 4096),
            (21504, 2048),
            (21504, 3072),
            (22528, 2048),
            (22528, 3072),
            (23552, 2048),
            (23552, 3072),
            (24576, 2048),
            (24576, 3072),
            (25600, 2048),
            (25600, 3072),
            (26624, 2048),
            (26624, 3072),
            (27648, 2048),
            (27648, 3072),
            (28672, 2048),
            (28672, 3072),
            (29696, 2048),
            (30720, 2048),
            (30720, 3072),
            (31744, 2048),
            (31744, 3072),
            (31744, 4096),
            (32768, 2048),
            (33792, 2048),
            (33792, 3072),
            (34816, 2048),
            (34816, 3072),
            (35840, 2048),
            (35840, 3072),
            (36864, 3072),
            (37888, 2048),
            (37888, 3072),
            (39936, 2048),
            (40960, 2048),
            (41984, 2048),
            (43008, 2048),
            (43008, 3072),
            (44032, 2048),
            (44032, 3072),
            (45056, 2048),
            (46080, 2048),
            (46080, 3072),
            (46080, 4096),
            (47104, 2048),
            (48128, 2048),
            (49152, 2048),
            (50176, 2048),
            (51200, 2048),
            (52224, 2048),
            (53248, 2048),
            (53248, 3072),
            (53248, 4096),
            (54272, 2048),
            (54272, 3072),
            (55296, 2048),
            (55296, 3072),
            (56320, 2048),
            (56320, 3072),
            (58368, 2048),
            (58368, 3072),
            (59392, 2048),
            (60416, 2048),
            (60416, 3072),
            (61440, 2048),
            (61440, 3072),
            (62464, 2048),
            (63488, 2048),
            (63488, 3072),
            (64512, 2048),
            (65536, 2048),
        ),
    ),
    (
        ForwardSchedule(
            mix_rows=8, mix_cols=128, mix_warps=4, norm_warps=4, skip_scale_reset=False
        ),
        (
            (4096, 8192),
            (5120, 8192),
            (7168, 8192),
            (11264, 8192),
            (11264, 9216),
            (18432, 4096),
            (19456, 4096),
            (22528, 4096),
            (23552, 4096),
            (28672, 4096),
            (29696, 3072),
            (30720, 4096),
            (32768, 3072),
            (33792, 4096),
            (34816, 4096),
            (36864, 4096),
            (37888, 4096),
            (38912, 3072),
            (38912, 4096),
            (39936, 4096),
            (41984, 3072),
            (43008, 4096),
            (45056, 3072),
            (45056, 4096),
            (47104, 4096),
            (48128, 4096),
            (50176, 4096),
            (51200, 3072),
            (51200, 4096),
            (52224, 4096),
            (55296, 4096),
            (56320, 4096),
            (57344, 2048),
            (57344, 4096),
            (59392, 4096),
            (61440, 4096),
            (62464, 3072),
            (63488, 4096),
            (64512, 3072),
            (64512, 4096),
        ),
    ),
)
QUALIFIED_BF16_FORWARD.update(
    {((10, 3), m, h): schedule for schedule, shapes in _BF16_FORWARD_GROUPS for m, h in shapes}
)
# END QUALIFIED BF16 TABLE

# BEGIN QUALIFIED STREAM COUNTS
# Evidence: results/streams_20261014/REPORT.md
QUALIFIED_STREAM_BACKWARD.update(
    {
        ((10, 3), 16384, 3072, 1, "bf16"): Schedule(
            grid=456, projection_layout="rows4v4", pp_tile=0, carry_rows=32, before_wait=True
        ),
        ((10, 3), 16384, 3072, 1, "mxfp8"): Schedule(
            grid=456, projection_layout="rows4v2", pp_tile=8, carry_rows=32, before_wait=True
        ),
        ((10, 3), 16384, 3072, 2, "bf16"): Schedule(
            grid=456, projection_layout="rows4v4", pp_tile=0, carry_rows=32, before_wait=True
        ),
        ((10, 3), 16384, 3072, 2, "mxfp8"): Schedule(
            grid=456, projection_layout="rows4v4", pp_tile=0, carry_rows=32, before_wait=True
        ),
        ((10, 3), 16384, 3072, 4, "bf16"): Schedule(
            grid=288, projection_layout="rows4v4", pp_tile=0, carry_rows=16, before_wait=True
        ),
        ((10, 3), 16384, 3072, 4, "mxfp8"): Schedule(
            grid=288, projection_layout="rows4v2", pp_tile=8, carry_rows=32, before_wait=True
        ),
        ((10, 3), 16384, 3072, 5, "bf16"): Schedule(
            grid=288, projection_layout="rows4v2", pp_tile=8, carry_rows=16, before_wait=True
        ),
        ((10, 3), 16384, 3072, 5, "mxfp8"): Schedule(
            grid=288, projection_layout="rows4v2", pp_tile=8, carry_rows=32, before_wait=True
        ),
        ((10, 3), 16384, 3072, 6, "bf16"): None,
        ((10, 3), 16384, 3072, 6, "mxfp8"): None,
        ((10, 3), 8192, 10240, 1, "bf16"): Schedule(
            grid=400, projection_layout="rows4v2", pp_tile=16, carry_rows=32, before_wait=True
        ),
        ((10, 3), 8192, 10240, 1, "mxfp8"): Schedule(
            grid=400, projection_layout="rows4v2", pp_tile=16, carry_rows=32, before_wait=True
        ),
        ((10, 3), 8192, 10240, 2, "bf16"): Schedule(
            grid=400, projection_layout="rows4v2", pp_tile=16, carry_rows=32, before_wait=True
        ),
        ((10, 3), 8192, 10240, 2, "mxfp8"): Schedule(
            grid=400, projection_layout="rows4v2", pp_tile=16, carry_rows=32, before_wait=True
        ),
        ((10, 3), 8192, 10240, 4, "bf16"): Schedule(
            grid=240, projection_layout="rows4v4", pp_tile=0, carry_rows=32, before_wait=True
        ),
        ((10, 3), 8192, 10240, 4, "mxfp8"): Schedule(
            grid=240, projection_layout="rows4v2", pp_tile=8, carry_rows=32, before_wait=True
        ),
        ((10, 3), 8192, 10240, 5, "bf16"): Schedule(
            grid=240, projection_layout="rows4v2", pp_tile=16, carry_rows=32, before_wait=True
        ),
        ((10, 3), 8192, 10240, 5, "mxfp8"): Schedule(
            grid=240, projection_layout="rows4v2", pp_tile=16, carry_rows=32, before_wait=True
        ),
        ((10, 3), 8192, 10240, 6, "bf16"): None,
        ((10, 3), 8192, 10240, 6, "mxfp8"): None,
        ((10, 3), 8192, 4096, 1, "bf16"): Schedule(
            grid=448, projection_layout="rows4v2", pp_tile=16, carry_rows=32, before_wait=True
        ),
        ((10, 3), 8192, 4096, 1, "mxfp8"): Schedule(
            grid=448, projection_layout="rows4v4", pp_tile=0, carry_rows=32, before_wait=True
        ),
        ((10, 3), 8192, 4096, 2, "bf16"): Schedule(
            grid=448, projection_layout="rows2v2", pp_tile=0, carry_rows=16, before_wait=True
        ),
        ((10, 3), 8192, 4096, 2, "mxfp8"): Schedule(
            grid=448, projection_layout="rows4v4", pp_tile=0, carry_rows=32, before_wait=True
        ),
        ((10, 3), 8192, 4096, 4, "bf16"): Schedule(
            grid=288, projection_layout="rows4v4", pp_tile=0, carry_rows=32, before_wait=True
        ),
        ((10, 3), 8192, 4096, 4, "mxfp8"): Schedule(
            grid=288, projection_layout="rows4v4", pp_tile=0, carry_rows=32, before_wait=True
        ),
        ((10, 3), 8192, 4096, 5, "bf16"): Schedule(
            grid=288, projection_layout="rows2v2", pp_tile=16, carry_rows=32, before_wait=True
        ),
        ((10, 3), 8192, 4096, 5, "mxfp8"): Schedule(
            grid=288, projection_layout="rows4v4", pp_tile=16, carry_rows=32, before_wait=True
        ),
        ((10, 3), 8192, 4096, 6, "bf16"): None,
        ((10, 3), 8192, 4096, 6, "mxfp8"): None,
    }
)
QUALIFIED_STREAM_FORWARD.update(
    {
        ((10, 3), 16384, 3072, 1, "bf16"): ForwardSchedule(
            mix_rows=4, mix_cols=256, mix_warps=4, norm_warps=4, skip_scale_reset=True
        ),
        ((10, 3), 16384, 3072, 1, "mxfp8"): ForwardSchedule(
            mix_rows=4, mix_cols=256, mix_warps=4, norm_warps=4, skip_scale_reset=True
        ),
        ((10, 3), 16384, 3072, 2, "bf16"): ForwardSchedule(
            mix_rows=4, mix_cols=256, mix_warps=4, norm_warps=4, skip_scale_reset=True
        ),
        ((10, 3), 16384, 3072, 2, "mxfp8"): ForwardSchedule(
            mix_rows=4, mix_cols=256, mix_warps=4, norm_warps=4, skip_scale_reset=True
        ),
        ((10, 3), 16384, 3072, 4, "bf16"): ForwardSchedule(
            mix_rows=16, mix_cols=128, mix_warps=4, norm_warps=4, skip_scale_reset=True
        ),
        ((10, 3), 16384, 3072, 4, "mxfp8"): ForwardSchedule(
            mix_rows=8, mix_cols=256, mix_warps=4, norm_warps=4, skip_scale_reset=True
        ),
        ((10, 3), 16384, 3072, 5, "bf16"): ForwardSchedule(
            mix_rows=4, mix_cols=256, mix_warps=4, norm_warps=4, skip_scale_reset=True
        ),
        ((10, 3), 16384, 3072, 5, "mxfp8"): ForwardSchedule(
            mix_rows=4, mix_cols=256, mix_warps=4, norm_warps=4, skip_scale_reset=True
        ),
        ((10, 3), 16384, 3072, 6, "bf16"): ForwardSchedule(
            mix_rows=4, mix_cols=128, mix_warps=4, norm_warps=4, skip_scale_reset=True
        ),
        ((10, 3), 16384, 3072, 6, "mxfp8"): ForwardSchedule(
            mix_rows=8, mix_cols=128, mix_warps=4, norm_warps=4, skip_scale_reset=True
        ),
        ((10, 3), 8192, 10240, 1, "bf16"): ForwardSchedule(
            mix_rows=8, mix_cols=128, mix_warps=4, norm_warps=4, skip_scale_reset=True
        ),
        ((10, 3), 8192, 10240, 1, "mxfp8"): ForwardSchedule(
            mix_rows=8, mix_cols=128, mix_warps=4, norm_warps=4, skip_scale_reset=True
        ),
        ((10, 3), 8192, 10240, 2, "bf16"): ForwardSchedule(
            mix_rows=4, mix_cols=256, mix_warps=4, norm_warps=4, skip_scale_reset=True
        ),
        ((10, 3), 8192, 10240, 2, "mxfp8"): ForwardSchedule(
            mix_rows=4, mix_cols=256, mix_warps=4, norm_warps=4, skip_scale_reset=True
        ),
        ((10, 3), 8192, 10240, 4, "bf16"): ForwardSchedule(
            mix_rows=8, mix_cols=256, mix_warps=4, norm_warps=4, skip_scale_reset=True
        ),
        ((10, 3), 8192, 10240, 4, "mxfp8"): ForwardSchedule(
            mix_rows=8, mix_cols=256, mix_warps=4, norm_warps=4, skip_scale_reset=True
        ),
        ((10, 3), 8192, 10240, 5, "bf16"): ForwardSchedule(
            mix_rows=4, mix_cols=256, mix_warps=4, norm_warps=4, skip_scale_reset=True
        ),
        ((10, 3), 8192, 10240, 5, "mxfp8"): ForwardSchedule(
            mix_rows=4, mix_cols=256, mix_warps=4, norm_warps=4, skip_scale_reset=True
        ),
        ((10, 3), 8192, 10240, 6, "bf16"): ForwardSchedule(
            mix_rows=4, mix_cols=256, mix_warps=4, norm_warps=4, skip_scale_reset=True
        ),
        ((10, 3), 8192, 10240, 6, "mxfp8"): ForwardSchedule(
            mix_rows=4, mix_cols=256, mix_warps=4, norm_warps=4, skip_scale_reset=True
        ),
        ((10, 3), 8192, 4096, 1, "bf16"): ForwardSchedule(
            mix_rows=4, mix_cols=256, mix_warps=4, norm_warps=4, skip_scale_reset=True
        ),
        ((10, 3), 8192, 4096, 1, "mxfp8"): ForwardSchedule(
            mix_rows=4, mix_cols=256, mix_warps=4, norm_warps=4, skip_scale_reset=True
        ),
        ((10, 3), 8192, 4096, 2, "bf16"): ForwardSchedule(
            mix_rows=8, mix_cols=256, mix_warps=4, norm_warps=4, skip_scale_reset=True
        ),
        ((10, 3), 8192, 4096, 2, "mxfp8"): ForwardSchedule(
            mix_rows=4, mix_cols=256, mix_warps=4, norm_warps=4, skip_scale_reset=True
        ),
        ((10, 3), 8192, 4096, 4, "bf16"): ForwardSchedule(
            mix_rows=8, mix_cols=256, mix_warps=4, norm_warps=4, skip_scale_reset=True
        ),
        ((10, 3), 8192, 4096, 4, "mxfp8"): ForwardSchedule(
            mix_rows=4, mix_cols=128, mix_warps=4, norm_warps=4, skip_scale_reset=True
        ),
        ((10, 3), 8192, 4096, 5, "bf16"): ForwardSchedule(
            mix_rows=4, mix_cols=256, mix_warps=4, norm_warps=4, skip_scale_reset=True
        ),
        ((10, 3), 8192, 4096, 5, "mxfp8"): ForwardSchedule(
            mix_rows=4, mix_cols=256, mix_warps=4, norm_warps=4, skip_scale_reset=True
        ),
        ((10, 3), 8192, 4096, 6, "bf16"): ForwardSchedule(
            mix_rows=4, mix_cols=256, mix_warps=4, norm_warps=4, skip_scale_reset=True
        ),
        ((10, 3), 8192, 4096, 6, "mxfp8"): ForwardSchedule(
            mix_rows=4, mix_cols=128, mix_warps=4, norm_warps=4, skip_scale_reset=True
        ),
    }
)
# END QUALIFIED STREAM COUNTS
