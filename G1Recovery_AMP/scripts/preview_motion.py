"""Preview one converted G1 recovery NPZ without running a policy or physics."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.viewer import NativeMujocoViewer, ViserPlayViewer

from g1recovery_amp.g1_model import G1_JOINT_NAMES
from g1recovery_amp.tasks.recovery.env_cfg import g1_recovery_dev_env_cfg

ROBOT_MODEL_JOINT_COUNTS = {
    "recovery-29dof": len(G1_JOINT_NAMES),
}


@dataclass(frozen=True)
class MotionPreviewData:
    """Validated arrays needed to display one converted recovery motion."""

    path: Path
    fps: float
    joint_pos: np.ndarray
    joint_vel: np.ndarray
    root_pos_w: np.ndarray
    root_quat_w: np.ndarray
    root_lin_vel_w: np.ndarray
    root_ang_vel_w: np.ndarray

    @property
    def num_frames(self) -> int:
        return int(self.joint_pos.shape[0])

    @property
    def duration_s(self) -> float:
        return self.num_frames / self.fps

    @property
    def joint_count(self) -> int:
        return int(self.joint_pos.shape[1])


def load_motion_preview(path: str | Path) -> MotionPreviewData:
    """Load and validate the fields used by the reference-only viewer."""

    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"motion file does not exist: {path}")

    required = {
        "fps",
        "joint_pos",
        "joint_vel",
        "body_pos_w",
        "body_quat_w",
        "body_lin_vel_w",
        "body_ang_vel_w",
    }
    with np.load(path, allow_pickle=False) as archive:
        missing = required.difference(archive.files)
        if missing:
            raise ValueError(f"motion file is missing fields: {sorted(missing)}")
        fps_values = np.asarray(archive["fps"], dtype=np.float32).reshape(-1)
        if fps_values.size != 1 or not np.isfinite(fps_values[0]):
            raise ValueError("fps must contain one finite value")
        fps = float(fps_values[0])
        arrays = {name: np.asarray(archive[name]).copy() for name in required - {"fps"}}

    if fps <= 0.0:
        raise ValueError(f"fps must be positive, got {fps}")

    joint_pos = arrays["joint_pos"]
    joint_vel = arrays["joint_vel"]
    if joint_pos.ndim != 2 or joint_pos.shape[1] != len(G1_JOINT_NAMES):
        raise ValueError(
            "joint_pos must have shape (frames, 29), "
            f"got {joint_pos.shape}"
        )
    if joint_vel.shape != joint_pos.shape:
        raise ValueError(
            f"joint_vel shape {joint_vel.shape} does not match {joint_pos.shape}"
        )

    num_frames = joint_pos.shape[0]
    body_shapes = {
        "body_pos_w": (num_frames, 3),
        "body_quat_w": (num_frames, 4),
        "body_lin_vel_w": (num_frames, 3),
        "body_ang_vel_w": (num_frames, 3),
    }
    for name, (expected_frames, final_dim) in body_shapes.items():
        value = arrays[name]
        if value.ndim != 3 or value.shape[0] != expected_frames or value.shape[2] != final_dim:
            raise ValueError(
                f"{name} must have shape (frames, bodies, {final_dim}), got {value.shape}"
            )
    if num_frames == 0:
        raise ValueError("motion must contain at least one frame")
    if not all(np.isfinite(value).all() for value in arrays.values()):
        raise ValueError("motion contains NaN or infinity")

    # Body index zero is the floating root in the converted mjlab motion format.
    return MotionPreviewData(
        path=path,
        fps=fps,
        joint_pos=joint_pos,
        joint_vel=joint_vel,
        root_pos_w=arrays["body_pos_w"][:, 0],
        root_quat_w=arrays["body_quat_w"][:, 0],
        root_lin_vel_w=arrays["body_lin_vel_w"][:, 0],
        root_ang_vel_w=arrays["body_ang_vel_w"][:, 0],
    )


def next_frame_index(frame: int, num_frames: int, *, loop: bool) -> int:
    """Return the following frame, wrapping or holding the final frame."""

    if num_frames <= 0:
        raise ValueError("num_frames must be positive")
    if not 0 <= frame < num_frames:
        raise ValueError(f"frame must be in [0, {num_frames}), got {frame}")
    if frame + 1 < num_frames:
        return frame + 1
    return 0 if loop else frame


def resolve_robot_model(joint_count: int, requested: str = "auto") -> str:
    """Resolve an automatic model choice and reject incompatible overrides."""

    if requested == "auto":
        for model_name, expected_count in ROBOT_MODEL_JOINT_COUNTS.items():
            if joint_count == expected_count:
                return model_name
        raise ValueError(f"no preview robot supports {joint_count} joints")

    if requested not in ROBOT_MODEL_JOINT_COUNTS:
        raise ValueError(f"unknown preview robot model: {requested}")
    expected_count = ROBOT_MODEL_JOINT_COUNTS[requested]
    if joint_count != expected_count:
        raise ValueError(
            f"{requested} expects {expected_count} joints, but motion has "
            f"{joint_count}"
        )
    return requested


class ReferencePlaybackEnv:
    """Small viewer adapter that writes NPZ frames instead of stepping physics."""

    def __init__(
        self,
        env: ManagerBasedRlEnv,
        motion: MotionPreviewData,
        *,
        loop: bool,
        height_offset: float,
    ) -> None:
        self._env = env
        self.motion = motion
        self.loop = loop
        self.height_offset = height_offset
        self.frame = 0
        self._device = torch.device(env.device)
        self._joint_pos = torch.as_tensor(motion.joint_pos, device=self._device)
        self._joint_vel = torch.as_tensor(motion.joint_vel, device=self._device)
        self._root_pos_w = torch.as_tensor(motion.root_pos_w, device=self._device)
        self._root_quat_w = torch.as_tensor(motion.root_quat_w, device=self._device)
        self._root_lin_vel_w = torch.as_tensor(
            motion.root_lin_vel_w, device=self._device
        )
        self._root_ang_vel_w = torch.as_tensor(
            motion.root_ang_vel_w, device=self._device
        )
        self._write_frame(0)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._env, name)

    @property
    def unwrapped(self) -> ReferencePlaybackEnv:
        return self

    @property
    def step_dt(self) -> float:
        return 1.0 / self.motion.fps

    def get_observations(self) -> dict[str, torch.Tensor]:
        return {}

    def reset(self) -> tuple[dict[str, torch.Tensor], dict]:
        self.frame = 0
        self._write_frame(self.frame)
        return {}, {}

    def step(self, actions: torch.Tensor) -> None:
        del actions
        self.frame = next_frame_index(
            self.frame, self.motion.num_frames, loop=self.loop
        )
        self._write_frame(self.frame)

    def _write_frame(self, frame: int) -> None:
        robot = self._env.scene["robot"]
        root_pos = self._root_pos_w[frame].clone()
        root_pos += self._env.scene.env_origins[0]
        root_pos[2] += self.height_offset
        root_state = torch.cat(
            (
                root_pos,
                self._root_quat_w[frame],
                self._root_lin_vel_w[frame],
                self._root_ang_vel_w[frame],
            )
        ).unsqueeze(0)
        robot.write_joint_state_to_sim(
            self._joint_pos[frame].unsqueeze(0),
            self._joint_vel[frame].unsqueeze(0),
        )
        robot.write_root_state_to_sim(root_state)
        self._env.sim.forward()


class ZeroPolicy:
    """Viewer-compatible policy placeholder; its output is intentionally ignored."""

    def __init__(self, device: str) -> None:
        self.device = device

    def __call__(self, observations: dict[str, torch.Tensor]) -> torch.Tensor:
        del observations
        return torch.empty((1, 0), device=self.device)


def build_preview_env(device: str, robot_model: str) -> ManagerBasedRlEnv:
    """Build a one-robot scene without training-time managers."""

    if robot_model == "recovery-29dof":
        cfg = g1_recovery_dev_env_cfg(play=True, disturbance_mode="none")
    else:
        raise ValueError(f"unknown preview robot model: {robot_model}")
    cfg.scene.num_envs = 1
    cfg.commands = {}
    cfg.observations = {}
    cfg.rewards = {}
    cfg.terminations = {}
    cfg.events = {}
    cfg.curriculum = {}
    cfg.metrics = {}
    cfg.recorders = {}
    env = ManagerBasedRlEnv(cfg=cfg, device=device)
    env.reset(seed=0)
    return env


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--motion-file", type=Path, required=True)
    parser.add_argument("--viewer", choices=("viser", "native"), default="viser")
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--robot-model",
        choices=("auto", "recovery-29dof"),
        default="auto",
        help="G1 recovery model used for playback.",
    )
    parser.add_argument(
        "--height-offset",
        type=float,
        default=0.0,
        help="Optional vertical display offset in metres.",
    )
    parser.add_argument(
        "--no-loop",
        action="store_true",
        help="Hold the final frame instead of looping to frame zero.",
    )
    parser.add_argument(
        "--inspect-only",
        action="store_true",
        help="Validate and describe the NPZ without opening a viewer.",
    )
    args = parser.parse_args()

    motion = load_motion_preview(args.motion_file)
    print(f"Motion: {motion.path}")
    print(f"Frames: {motion.num_frames}")
    print(f"FPS: {motion.fps:g}")
    print(f"Duration: {motion.duration_s:.2f} s")
    robot_model = resolve_robot_model(motion.joint_count, args.robot_model)
    print(f"Joints: {motion.joint_count}")
    print(f"Robot model: {robot_model}")
    if args.inspect_only:
        return

    env = build_preview_env(args.device, robot_model)
    playback = ReferencePlaybackEnv(
        env,
        motion,
        loop=not args.no_loop,
        height_offset=args.height_offset,
    )
    policy = ZeroPolicy(args.device)
    try:
        if args.viewer == "viser":
            print("Open the Viser URL printed below in your browser.")
            ViserPlayViewer(playback, policy).run()
        else:
            NativeMujocoViewer(playback, policy).run()
    finally:
        env.close()


if __name__ == "__main__":
    main()
