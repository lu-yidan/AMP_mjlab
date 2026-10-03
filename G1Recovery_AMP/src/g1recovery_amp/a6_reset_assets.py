"""Validated access to the frozen historical A6 reset bank."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np

A6_RESET_DATA_DIR = Path(__file__).resolve().parent / "data" / "reset_banks" / "a6"
A6_MULTITERRAIN_TRAIN_BANK = A6_RESET_DATA_DIR / "multiterrain_train.npz"
A6_MULTITERRAIN_TRAIN_SHA256 = (
    "287eae8e8840c1b3281e7010a84182b824fefacd34439c4f993e9f130af1027a"
)
A6_NATURAL_CURRICULUM_TRAIN_BANK = (
    A6_RESET_DATA_DIR / "natural_curriculum_train.npz"
)
A6_NATURAL_CURRICULUM_TRAIN_SHA256 = (
    "e9f94540520d7927dee01150a0f0bae39ad2aad5ae008b1c7c71f521650e44b9"
)

A6_QPOS_DIM = 36
A6_ROOT_QPOS_DIM = 7
A6_FLAT_STRATUM = 0
A6_DIRECTION_NAMES = (
    "supine",
    "prone",
    "left_side_down",
    "right_side_down",
)
A6_SOURCE_NAMES = ("natural", "procedural")
A6_STAGE_NAMES = ("low", "middle", "late")


@dataclass(frozen=True)
class A6ResetBank:
    """Arrays stored in the historical multi-terrain reset bank."""

    qpos: np.ndarray
    direction: np.ndarray
    stratum: np.ndarray
    source: np.ndarray


@dataclass(frozen=True)
class A6NaturalCurriculumBank:
    """Natural motion states used for the flat middle/late reset curriculum."""

    qpos: np.ndarray
    labels: np.ndarray
    stages: np.ndarray
    clips: np.ndarray


def file_sha256(path: Path) -> str:
    """Return a streaming SHA256 digest without loading a whole asset at once."""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_a6_multiterrain_bank(
    path: Path = A6_MULTITERRAIN_TRAIN_BANK,
    *,
    expected_sha256: str | None = A6_MULTITERRAIN_TRAIN_SHA256,
) -> A6ResetBank:
    """Load and validate the A6 bank before it can write simulator state."""

    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"A6 reset bank not found: {path}")
    if expected_sha256 is not None:
        actual_sha256 = file_sha256(path)
        if actual_sha256 != expected_sha256:
            raise ValueError(
                f"A6 reset bank SHA256 mismatch: {actual_sha256} != "
                f"{expected_sha256}"
            )

    required = {"qpos", "direction", "stratum", "source"}
    with np.load(path, allow_pickle=False) as archive:
        missing = required.difference(archive.files)
        if missing:
            raise ValueError(f"A6 reset bank is missing arrays: {sorted(missing)}")
        bank = A6ResetBank(
            qpos=np.asarray(archive["qpos"]).copy(),
            direction=np.asarray(archive["direction"]).copy(),
            stratum=np.asarray(archive["stratum"]).copy(),
            source=np.asarray(archive["source"]).copy(),
        )

    count = bank.qpos.shape[0]
    if bank.qpos.shape != (count, A6_QPOS_DIM):
        raise ValueError(f"expected A6 qpos shape (N, {A6_QPOS_DIM}), got {bank.qpos.shape}")
    for name, values in (
        ("direction", bank.direction),
        ("stratum", bank.stratum),
        ("source", bank.source),
    ):
        if values.shape != (count,):
            raise ValueError(f"expected {name} shape ({count},), got {values.shape}")
    if not np.isfinite(bank.qpos).all():
        raise ValueError("A6 reset bank qpos contains NaN or Inf")

    quaternion_norm = np.linalg.norm(bank.qpos[:, 3:7], axis=1)
    if not np.allclose(quaternion_norm, 1.0, rtol=0.0, atol=1.0e-5):
        raise ValueError("A6 reset bank contains non-unit root quaternions")

    for direction in range(len(A6_DIRECTION_NAMES)):
        for source in range(len(A6_SOURCE_NAMES)):
            pool = (
                (bank.stratum == A6_FLAT_STRATUM)
                & (bank.direction == direction)
                & (bank.source == source)
            )
            if not np.any(pool):
                raise ValueError(
                    "A6 flat reset pool is empty for "
                    f"direction={direction}, source={source}"
                )
    return bank


def load_a6_natural_curriculum_bank(
    path: Path = A6_NATURAL_CURRICULUM_TRAIN_BANK,
    *,
    expected_sha256: str | None = A6_NATURAL_CURRICULUM_TRAIN_SHA256,
) -> A6NaturalCurriculumBank:
    """Load and validate the frozen natural L/M/H curriculum bank."""

    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"A6 natural curriculum bank not found: {path}")
    if expected_sha256 is not None:
        actual_sha256 = file_sha256(path)
        if actual_sha256 != expected_sha256:
            raise ValueError(
                f"A6 natural curriculum bank SHA256 mismatch: {actual_sha256} != "
                f"{expected_sha256}"
            )

    required = {"qpos", "labels", "stages", "clips"}
    with np.load(path, allow_pickle=False) as archive:
        missing = required.difference(archive.files)
        if missing:
            raise ValueError(
                "A6 natural curriculum bank is missing arrays: "
                f"{sorted(missing)}"
            )
        bank = A6NaturalCurriculumBank(
            qpos=np.asarray(archive["qpos"]).copy(),
            labels=np.asarray(archive["labels"]).copy(),
            stages=np.asarray(archive["stages"]).copy(),
            clips=np.asarray(archive["clips"]).copy(),
        )

    count = bank.qpos.shape[0]
    if bank.qpos.shape != (count, A6_QPOS_DIM):
        raise ValueError(
            f"expected A6 natural qpos shape (N, {A6_QPOS_DIM}), "
            f"got {bank.qpos.shape}"
        )
    for name, values in (
        ("labels", bank.labels),
        ("stages", bank.stages),
        ("clips", bank.clips),
    ):
        if values.shape != (count,):
            raise ValueError(f"expected {name} shape ({count},), got {values.shape}")
    if not np.isfinite(bank.qpos).all():
        raise ValueError("A6 natural curriculum qpos contains NaN or Inf")
    quaternion_norm = np.linalg.norm(bank.qpos[:, 3:7], axis=1)
    if not np.allclose(quaternion_norm, 1.0, rtol=0.0, atol=1.0e-5):
        raise ValueError("A6 natural curriculum contains non-unit root quaternions")
    for stage in A6_STAGE_NAMES:
        if not np.any(bank.stages == stage):
            raise ValueError(f"A6 natural curriculum has no {stage!r} states")
    return bank


def natural_stage_sampling_pool(
    bank: A6NaturalCurriculumBank, stage: str
) -> tuple[np.ndarray, np.ndarray]:
    """Return rows and clip-balanced weights for one natural reset stage."""

    if stage not in A6_STAGE_NAMES:
        raise ValueError(f"unknown A6 reset stage: {stage!r}")
    rows = np.flatnonzero(bank.stages == stage)
    if len(rows) == 0:
        raise ValueError(f"A6 natural curriculum has no {stage!r} states")

    # Mirror partners represent the same source clip and must share one clip's
    # total probability instead of each receiving a full clip allocation.
    base_clips = np.asarray(
        [str(name).replace("__mirror", "") for name in bank.clips]
    )
    clips = np.unique(base_clips[rows])
    weights = np.zeros(len(rows), dtype=np.float64)
    for clip in clips:
        subset = np.flatnonzero(base_clips[rows] == clip)
        weights[subset] = 1.0 / len(clips) / len(subset)
    if not np.isclose(weights.sum(), 1.0):
        raise ValueError(f"A6 {stage!r} sampling weights do not sum to one")
    return rows, weights


__all__ = [
    "A6_DIRECTION_NAMES",
    "A6_FLAT_STRATUM",
    "A6_MULTITERRAIN_TRAIN_BANK",
    "A6_MULTITERRAIN_TRAIN_SHA256",
    "A6_NATURAL_CURRICULUM_TRAIN_BANK",
    "A6_NATURAL_CURRICULUM_TRAIN_SHA256",
    "A6_QPOS_DIM",
    "A6_ROOT_QPOS_DIM",
    "A6_SOURCE_NAMES",
    "A6_STAGE_NAMES",
    "A6NaturalCurriculumBank",
    "A6ResetBank",
    "file_sha256",
    "load_a6_multiterrain_bank",
    "load_a6_natural_curriculum_bank",
    "natural_stage_sampling_pool",
]
