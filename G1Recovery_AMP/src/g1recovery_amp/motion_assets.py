"""Paths and equal sampling weights for the selected recovery motions.

The suffixes in the file stems (for example ``f1`` and ``y1``) are human
labels only.  The single-route recovery task does not branch on those labels.
"""

from collections.abc import Mapping
from pathlib import Path
from types import MappingProxyType
from typing import Final

MOTION_DATA_DIR = Path(__file__).resolve().parent / "data" / "motions"
GET_UP_MOTION_DATA_DIR = MOTION_DATA_DIR / "get_up"

# Reproduce the eight-motion manifest saved with the successful model_13699.pt
# experiment.  The insertion order is the runtime loader order.  Every value
# is 1.0, so RecoveryMotionCommand selects each file with probability 1 / 8.
GET_UP_TRAINING_MOTION_WEIGHTS: Final[Mapping[str, float]] = MappingProxyType(
    {
        "fallAndGetUp1_subject1_1060_1150": 1.0,
        "fallAndGetUp1_subject1_1400_1480": 1.0,
        "fallAndGetUp1_subject1_2100_2200": 1.0,
        "fallAndGetUp1_subject5_2500_2600": 1.0,
        "fallAndGetUp2_subject2_850_1050": 1.0,
        "fallAndGetUp6_subject1_530_600": 1.0,
        "fallAndGetUp6_subject1_650_700": 1.0,
        "fallAndGetUp6_subject1_1630_1690": 1.0,
    }
)

ACTIVE_GET_UP_MOTION_NAMES: Final[tuple[str, ...]] = tuple(
    GET_UP_TRAINING_MOTION_WEIGHTS
)

CONVERTED_GET_UP_MOTIONS: Final[Mapping[str, Path]] = MappingProxyType(
    {
        name: GET_UP_MOTION_DATA_DIR / f"{name}.npz"
        for name in ACTIVE_GET_UP_MOTION_NAMES
    }
)

# The first file is required separately by mjlab's parent MotionCommand API.
# It is also the deterministic preview motion when sampling_mode="start".
GET_UP_DEFAULT_MOTION: Final[Path] = next(iter(CONVERTED_GET_UP_MOTIONS.values()))
