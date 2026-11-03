"""Merge validated 29-DoF escape states into a training-only reset bank."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("sources", type=Path, nargs="+")
    args = parser.parse_args()
    if args.output in args.sources:
        parser.error("output cannot also be a source")
    keys = ("qpos", "qvel", "plate_xyz")
    expected = {"qpos": 36, "qvel": 35, "plate_xyz": 3}
    arrays: dict[str, list[np.ndarray]] = {key: [] for key in keys}
    manifest = []
    for path in args.sources:
        with path.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        with np.load(path) as archive:
            n = len(archive["qpos"])
            for key in keys:
                values = np.asarray(archive[key])
                if values.shape != (n, expected[key]) or not np.isfinite(values).all():
                    raise ValueError(f"invalid {key} in {path}: {values.shape}")
                arrays[key].append(values.copy())
            if "best_xy_clearance" in archive:
                clearance = archive["best_xy_clearance"]
                if len(clearance) != n or float(clearance.min()) < 0.035:
                    raise ValueError(f"replay state is still under the plate: {path}")
        manifest.append({"source": str(path), "sha256": digest, "samples": n})
    merged = {key: np.concatenate(chunks) for key, chunks in arrays.items()}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **merged)
    report = {
        "samples": len(merged["qpos"]),
        "mean_root_height_m": float(merged["qpos"][:, 2].mean()),
        "sources": manifest,
    }
    args.output.with_suffix(".json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
