"""Verify frozen SMP A6 banks and align qpos to the selected G1 model."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np
from mjlab.asset_zoo.robots.unitree_g1.g1_constants import get_spec

from g1recovery_host.model import (
    HOST29_JOINT_NAMES,
    HOST_JOINT_NAMES,
    get_host_urdf_capsule_spec,
)

SOURCE_COMMIT = "1b6d7e61ddbcd2caf9ea17b02160dd1cf0d119d2"
SOURCE_JOINT_NAMES = (
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
)
EXPECTED_SHA = {
    "multiterrain": "287eae8e8840c1b3281e7010a84182b824fefacd34439c4f993e9f130af1027a",
    "natural_curriculum": "e9f94540520d7927dee01150a0f0bae39ad2aad5ae008b1c7c71f521650e44b9",
    "procedural_low": "e289bed8d1c93e87fbdcf969fc5fc741f2d8500a0908ed145d2a4e02e7fe116d",
}
SOURCE_PATHS = {
    "multiterrain": "outputs/multiterrain_bank/train.npz",
    "natural_curriculum": "datasets/reset_banks/natural_curriculum_v1/train.npz",
    "procedural_low": "datasets/reset_banks/procedural_low_v1/train.npz",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--dof", type=int, choices=(23, 29), default=29)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    names = HOST29_JOINT_NAMES if args.dof == 29 else HOST_JOINT_NAMES
    keep = tuple(SOURCE_JOINT_NAMES.index(name) for name in names)
    if len(keep) != args.dof or len(set(keep)) != args.dof:
        raise ValueError("HoST joint mapping is incomplete or duplicated")
    model = (get_spec() if args.dof == 29 else get_host_urdf_capsule_spec()).compile()
    joint_ids = [
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in names
    ]
    lower, upper = model.jnt_range[joint_ids].T
    if args.output is None:
        args.output = Path(f"assets/a6/host{args.dof}")
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = {
        "source_commit": SOURCE_COMMIT,
        "source_joint_names": SOURCE_JOINT_NAMES,
        "host_joint_names": names,
        "keep_source_indices": keep,
        "banks": {},
    }
    for name, relative in SOURCE_PATHS.items():
        source = args.source_root / relative
        actual = sha256(source)
        if actual != EXPECTED_SHA[name]:
            raise ValueError(f"{source}: SHA256 {actual} != {EXPECTED_SHA[name]}")
        with np.load(source) as bank:
            qpos = bank["qpos"]
            if qpos.ndim != 2 or qpos.shape[1] != 36:
                raise ValueError(f"{source}: expected [N,36] qpos, got {qpos.shape}")
            projected = np.concatenate(
                (qpos[:, :7], qpos[:, 7 + np.array(keep)]), axis=1
            ).astype(np.float32)
            clipped = np.clip(projected[:, 7:], lower, upper)
            clipped_rows = int(np.any(clipped != projected[:, 7:], axis=1).sum())
            max_clip_rad = float(np.max(np.abs(clipped - projected[:, 7:])))
            projected[:, 7:] = clipped
            converted = {
                key: projected if key == "qpos" else bank[key] for key in bank.files
            }
        target = args.output / f"{name}_host{args.dof}.npz"
        np.savez_compressed(target, **converted)
        manifest["banks"][name] = {
            "source_path": str(source.resolve()),
            "source_sha256": actual,
            "output_path": str(target.resolve()),
            "output_sha256": sha256(target),
            "count": len(converted["qpos"]),
            "qpos_width": 7 + args.dof,
            "joint_limit_clipped_rows": clipped_rows,
            "maximum_joint_clip_rad": max_clip_rad,
        }
        print(f"{name}: {len(converted['qpos'])} states -> {target}")
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
