"""Measure recovery behavior from fixed motion starts without rendering or training.

The script keeps one checkpoint fixed while changing only the reset motion.
It reports side-to-side rolling, upright time, and ankle spacing alongside
the corresponding reference clip's final upright spacing.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from itertools import pairwise
from pathlib import Path

import numpy as np
import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_rl_cfg, load_runner_cls
from mjlab.utils.torch import configure_torch_backends

from g1recovery_amp.motion_assets import CONVERTED_GET_UP_MOTIONS
from g1recovery_amp.tasks.recovery import DEV_TASK_ID
from g1recovery_amp.tasks.recovery.a6_env_cfg import apply_a6_timing
from g1recovery_amp.tasks.recovery.env_cfg import g1_recovery_dev_env_cfg


@dataclass(frozen=True)
class PoseSample:
    height: float
    up_y: float
    up_z: float
    hip_roll_abs_max: float
    ankle_spacing_xy: float


def _upright_streak(samples: list[PoseSample]) -> int:
    longest = current = 0
    for sample in samples:
        current = current + 1 if sample.height > 0.65 and sample.up_z > 0.7 else 0
        longest = max(longest, current)
    return longest


def _low_height_side_switches(samples: list[PoseSample]) -> int:
    """Ignore near-neutral orientations, then count changes between clear sides."""

    sides = [
        1 if sample.up_y > 0.5 else -1
        for sample in samples
        if sample.height < 0.65 and abs(sample.up_y) > 0.5
    ]
    return sum(previous != current for previous, current in pairwise(sides))


def _median_upright_spacing(samples: list[PoseSample]) -> float | None:
    spacing = [
        sample.ankle_spacing_xy
        for sample in samples
        if sample.height > 0.65 and sample.up_z > 0.7
    ]
    return float(np.median(spacing)) if spacing else None


def _reference_samples(
    motion_file: Path, left_ankle_id: int, right_ankle_id: int
) -> list[PoseSample]:
    with np.load(motion_file) as motion:
        root_quat = motion["body_quat_w"][:, 0]
        w, x, y, z = root_quat.T
        up_y = 2.0 * (y * z + w * x)
        up_z = 1.0 - 2.0 * (x * x + y * y)
        # The converted motion stores bodies in the same entity order as mjlab.
        ankle_delta = (
            motion["body_pos_w"][:, left_ankle_id, :2]
            - motion["body_pos_w"][:, right_ankle_id, :2]
        )
        ankle_spacing = np.linalg.norm(ankle_delta, axis=1)
        height = motion["body_pos_w"][:, 0, 2] + 0.1
        # Reference hip-roll columns are resolved separately in the live model;
        # this report needs only reference ankle spacing and torso direction.
        return [
            PoseSample(float(h), float(side), float(upright), 0.0, float(gap))
            for h, side, upright, gap in zip(
                height, up_y, up_z, ankle_spacing, strict=True
            )
        ]


def _live_sample(robot, hip_roll_ids: list[int], ankle_ids: list[int]) -> PoseSample:
    up_b = -robot.data.projected_gravity_b[0]
    ankle_pos = robot.data.body_link_pos_w[0, ankle_ids, :2]
    return PoseSample(
        height=float(robot.data.root_link_pos_w[0, 2]),
        up_y=float(up_b[1]),
        up_z=float(up_b[2]),
        hip_roll_abs_max=float(robot.data.joint_pos[0, hip_roll_ids].abs().max()),
        ankle_spacing_xy=float(torch.linalg.vector_norm(ankle_pos[0] - ankle_pos[1])),
    )


def evaluate(
    motion_file: Path,
    checkpoint_file: Path,
    device: str,
    max_steps: int,
    a6_timing: bool = False,
) -> None:
    cfg = g1_recovery_dev_env_cfg(play=True)
    if a6_timing:
        apply_a6_timing(cfg)
    cfg.scene.num_envs = 1
    cfg.seed = 42
    cfg.commands["motion"].motion_file = str(motion_file)
    agent_cfg = load_rl_cfg(DEV_TASK_ID)
    raw_env = ManagerBasedRlEnv(cfg=cfg, device=device)
    env = RslRlVecEnvWrapper(raw_env, clip_actions=agent_cfg.clip_actions)
    try:
        runner_cls = load_runner_cls(DEV_TASK_ID) or MjlabOnPolicyRunner
        runner = runner_cls(env, asdict(agent_cfg), device=device)
        runner.load(
            str(checkpoint_file),
            load_cfg={"actor": True},
            strict=True,
            map_location=device,
        )
        policy = runner.get_inference_policy(device=device)
        robot = env.unwrapped.scene["robot"]
        hip_roll_ids, _ = robot.find_joints(
            ("left_hip_roll_joint", "right_hip_roll_joint"), preserve_order=True
        )
        ankle_ids, _ = robot.find_bodies(
            ("left_ankle_roll_link", "right_ankle_roll_link"),
            preserve_order=True,
        )
        samples = [_live_sample(robot, hip_roll_ids, ankle_ids)]
        observations = env.get_observations()
        with torch.inference_mode():
            for _ in range(max_steps):
                actions = policy(observations)
                observations, _, dones, _ = env.step(actions)
                if bool(dones[0]):
                    break
                samples.append(_live_sample(robot, hip_roll_ids, ankle_ids))

        reference = _reference_samples(motion_file, ankle_ids[0], ankle_ids[1])
        step_dt = cfg.sim.mujoco.timestep * cfg.decimation
        upright_s = _upright_streak(samples) * step_dt
        policy_spacing = _median_upright_spacing(samples)
        reference_spacing = _median_upright_spacing(reference)
        print(f"\n{motion_file.name}")
        print(
            f"  observed_s={len(samples) * step_dt:.2f}, final_height_m={samples[-1].height:.3f}"
        )
        print(
            f"  longest_upright_s={upright_s:.2f}, "
            f"low_height_side_switches: policy={_low_height_side_switches(samples)}, "
            f"reference={_low_height_side_switches(reference)}"
        )
        print(
            f"  upright_ankle_spacing_m: "
            f"policy={policy_spacing}, reference={reference_spacing}"
        )
        print(f"  final_abs_hip_roll_rad={samples[-1].hip_roll_abs_max:.3f}")
    finally:
        env.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-file", type=Path, required=True)
    parser.add_argument(
        "--motion-files",
        type=Path,
        nargs="+",
        default=list(CONVERTED_GET_UP_MOTIONS.values()),
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument(
        "--a6-timing",
        action="store_true",
        help="Use A6's 0.002 s physics and 10 substeps, still 0.02 s per action",
    )
    args = parser.parse_args()
    if not args.checkpoint_file.is_file():
        parser.error(f"checkpoint not found: {args.checkpoint_file}")
    if any(not path.is_file() for path in args.motion_files):
        parser.error("one or more motion files do not exist")
    if args.max_steps <= 0:
        parser.error("--max-steps must be positive")
    configure_torch_backends()
    for motion_file in args.motion_files:
        evaluate(
            motion_file,
            args.checkpoint_file,
            args.device,
            args.max_steps,
            a6_timing=args.a6_timing,
        )


if __name__ == "__main__":
    main()
