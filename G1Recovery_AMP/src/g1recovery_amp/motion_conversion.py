"""Convert IsaacLab G1 recovery motions into mjlab's motion format.

The source clips store the floating-base pose and 29 joint angles at their
original sampling rate.  mjlab consumes one motion frame per policy step, so
the clip is resampled to the policy rate before MuJoCo forward kinematics is
used to recompute every link pose and velocity.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import torch
from mjlab.scene import Scene, SceneCfg
from mjlab.sim.sim import MujocoCfg, Simulation, SimulationCfg
from mjlab.terrains import TerrainEntityCfg
from mjlab.utils.lab_api.math import (
    axis_angle_from_quat,
    quat_conjugate,
    quat_mul,
    quat_slerp,
)
from mjlab.viewer.offscreen_renderer import OffscreenRenderer
from mjlab.viewer.viewer_config import ViewerConfig

from g1recovery_amp.g1_model import (
    G1_JOINT_NAMES,
    G1_SOURCE_MOTION_TO_JOINT_ORDER,
    get_recovery_g1_robot_cfg,
)

REQUIRED_SOURCE_KEYS = frozenset(
    {"fps", "loop_mode", "root_pos", "root_rot", "dof_pos", "key_body_pos"}
)
MJLAB_MOTION_KEYS = (
    "joint_pos",
    "joint_vel",
    "body_pos_w",
    "body_quat_w",
    "body_lin_vel_w",
    "body_ang_vel_w",
)


@dataclass(frozen=True)
class SourceMotion:
    """Validated motion fields loaded from one Legged Lab ``.pkl`` clip."""

    fps: float
    loop_mode: int
    root_pos: torch.Tensor
    root_quat_wxyz: torch.Tensor
    joint_pos: torch.Tensor
    key_body_pos_w: torch.Tensor

    @property
    def num_frames(self) -> int:
        return self.joint_pos.shape[0]

    @property
    def duration(self) -> float:
        return (self.num_frames - 1) / self.fps


@dataclass(frozen=True)
class ResampledMotion:
    """Root and joint states sampled once per target policy step."""

    fps: float
    root_pos: torch.Tensor
    root_quat_wxyz: torch.Tensor
    root_lin_vel_w: torch.Tensor
    root_ang_vel_w: torch.Tensor
    joint_pos: torch.Tensor
    joint_vel: torch.Tensor

    @property
    def num_frames(self) -> int:
        return self.joint_pos.shape[0]


def load_source_motion(path: str | Path, device: str = "cpu") -> SourceMotion:
    """Load and validate one source motion without changing it."""

    path = Path(path)
    raw = joblib.load(path)
    if not isinstance(raw, dict):
        raise TypeError(f"motion must contain a dictionary: {path}")

    missing = REQUIRED_SOURCE_KEYS - raw.keys()
    if missing:
        raise ValueError(f"motion is missing fields: {sorted(missing)}")

    fps = float(raw["fps"])
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError(f"fps must be positive and finite, got {fps}")

    def tensor(name: str) -> torch.Tensor:
        value = torch.as_tensor(raw[name], dtype=torch.float32, device=device)
        if not torch.isfinite(value).all():
            raise ValueError(f"motion field {name!r} contains NaN or infinity")
        return value

    root_pos = tensor("root_pos")
    root_quat = tensor("root_rot")
    source_joint_pos = tensor("dof_pos")
    key_body_pos = tensor("key_body_pos")

    num_frames = source_joint_pos.shape[0]
    expected_shapes = {
        "root_pos": (num_frames, 3),
        "root_rot": (num_frames, 4),
        "dof_pos": (num_frames, len(G1_JOINT_NAMES)),
    }
    actual_shapes = {
        "root_pos": tuple(root_pos.shape),
        "root_rot": tuple(root_quat.shape),
        "dof_pos": tuple(source_joint_pos.shape),
    }
    for name, expected in expected_shapes.items():
        if actual_shapes[name] != expected:
            raise ValueError(
                f"motion field {name!r} has shape {actual_shapes[name]}, "
                f"expected {expected}"
            )
    if num_frames < 3:
        raise ValueError("motion needs at least three frames to compute velocities")
    if key_body_pos.ndim != 3 or key_body_pos.shape[0] != num_frames:
        raise ValueError(
            "motion field 'key_body_pos' must have shape (frames, bodies, 3)"
        )
    if key_body_pos.shape[2] != 3:
        raise ValueError(
            "motion field 'key_body_pos' must have shape (frames, bodies, 3)"
        )

    # Source PKL columns follow IsaacLab's interleaved ``lab_dof_names`` order;
    # mjlab's G1 follows Unitree SDK order.  Reorder explicitly by the mapping
    # derived from joint names before any interpolation or MuJoCo state write.
    joint_pos = source_joint_pos[:, G1_SOURCE_MOTION_TO_JOINT_ORDER]

    quat_norm = torch.linalg.vector_norm(root_quat, dim=-1)
    if torch.any(quat_norm < 1.0e-6):
        raise ValueError("motion contains a zero-length root quaternion")
    root_quat = root_quat / quat_norm.unsqueeze(-1)

    return SourceMotion(
        fps=fps,
        loop_mode=int(raw["loop_mode"]),
        root_pos=root_pos,
        root_quat_wxyz=root_quat,
        joint_pos=joint_pos,
        key_body_pos_w=key_body_pos,
    )


def _angular_velocity_w(quat_wxyz: torch.Tensor, dt: float) -> torch.Tensor:
    """Compute world-frame angular velocity using centered differences."""

    previous = quat_wxyz[:-2]
    following = quat_wxyz[2:]
    relative = quat_mul(following, quat_conjugate(previous))
    centered = axis_angle_from_quat(relative) / (2.0 * dt)
    return torch.cat((centered[:1], centered, centered[-1:]), dim=0)


def resample_motion(source: SourceMotion, output_fps: float) -> ResampledMotion:
    """Interpolate a source clip onto mjlab's one-frame-per-policy-step grid."""

    if not np.isfinite(output_fps) or output_fps <= 0:
        raise ValueError(f"output_fps must be positive and finite, got {output_fps}")

    output_dt = 1.0 / output_fps
    times = torch.arange(
        0.0,
        source.duration,
        output_dt,
        dtype=torch.float32,
        device=source.joint_pos.device,
    )
    source_frame = times * source.fps
    index_0 = torch.floor(source_frame).to(torch.long)
    index_1 = torch.clamp(index_0 + 1, max=source.num_frames - 1)
    blend = source_frame - index_0

    def lerp(values: torch.Tensor) -> torch.Tensor:
        weight = blend.reshape((-1,) + (1,) * (values.ndim - 1))
        return values[index_0] * (1.0 - weight) + values[index_1] * weight

    root_quat = torch.stack(
        [
            quat_slerp(source.root_quat_wxyz[i0], source.root_quat_wxyz[i1], float(t))
            for i0, i1, t in zip(index_0, index_1, blend, strict=True)
        ]
    )
    # q and -q describe the same orientation.  A unique sign prevents artificial
    # jumps when angular velocity is computed from neighboring samples.
    root_quat = torch.where(root_quat[:, :1] < 0.0, -root_quat, root_quat)

    root_pos = lerp(source.root_pos)
    joint_pos = lerp(source.joint_pos)
    root_lin_vel = torch.gradient(root_pos, spacing=output_dt, dim=0)[0]
    joint_vel = torch.gradient(joint_pos, spacing=output_dt, dim=0)[0]
    root_ang_vel = _angular_velocity_w(root_quat, output_dt)

    return ResampledMotion(
        fps=float(output_fps),
        root_pos=root_pos,
        root_quat_wxyz=root_quat,
        root_lin_vel_w=root_lin_vel,
        root_ang_vel_w=root_ang_vel,
        joint_pos=joint_pos,
        joint_vel=joint_vel,
    )


def convert_motion(
    source_path: str | Path,
    output_path: str | Path,
    *,
    output_fps: float = 50.0,
    device: str = "cuda:0",
    video_path: str | Path | None = None,
    storyboard_path: str | Path | None = None,
) -> dict[str, np.ndarray]:
    """Convert one source clip and save the resulting mjlab ``.npz`` file."""

    if device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"

    source = load_source_motion(source_path, device=device)
    motion = resample_motion(source, output_fps)

    scene_cfg = SceneCfg(
        num_envs=1,
        terrain=TerrainEntityCfg(terrain_type="plane"),
        entities={"robot": get_recovery_g1_robot_cfg()},
    )
    scene = Scene(scene_cfg, device=device)
    sim_cfg = SimulationCfg(mujoco=MujocoCfg(timestep=1.0 / output_fps))
    sim = Simulation(
        num_envs=1,
        cfg=sim_cfg,
        model=scene.compile(),
        device=device,
    )
    scene.initialize(sim.mj_model, sim.model, sim.data)
    scene.reset()

    robot = scene["robot"]
    joint_indices = robot.find_joints(G1_JOINT_NAMES, preserve_order=True)[0]
    log: dict[str, list[np.ndarray]] = {key: [] for key in MJLAB_MOTION_KEYS}
    video_frames: list[np.ndarray] = []
    storyboard_frames: dict[int, np.ndarray] = {}
    storyboard_indices = {0, motion.num_frames // 2, motion.num_frames - 1}
    renderer: OffscreenRenderer | None = None
    if video_path is not None or storyboard_path is not None:
        renderer = OffscreenRenderer(
            model=sim.mj_model,
            scene=scene,
            cfg=ViewerConfig(
                width=640,
                height=480,
                origin_type=ViewerConfig.OriginType.ASSET_BODY,
                entity_name="robot",
                body_name="torso_link",
                distance=2.8,
                elevation=-8.0,
                azimuth=130.0,
            ),
        )
        renderer.initialize()

    try:
        for frame in range(motion.num_frames):
            root_state = robot.data.default_root_state.clone()
            root_state[:, :3] = motion.root_pos[frame]
            root_state[:, 3:7] = motion.root_quat_wxyz[frame]
            root_state[:, 7:10] = motion.root_lin_vel_w[frame]
            root_state[:, 10:13] = motion.root_ang_vel_w[frame]
            robot.write_root_state_to_sim(root_state)

            joint_pos = robot.data.default_joint_pos.clone()
            joint_vel = robot.data.default_joint_vel.clone()
            joint_pos[:, joint_indices] = motion.joint_pos[frame]
            joint_vel[:, joint_indices] = motion.joint_vel[frame]
            robot.write_joint_state_to_sim(joint_pos, joint_vel)

            sim.forward()
            scene.update(1.0 / output_fps)

            log["joint_pos"].append(robot.data.joint_pos[0].cpu().numpy().copy())
            log["joint_vel"].append(robot.data.joint_vel[0].cpu().numpy().copy())
            log["body_pos_w"].append(
                robot.data.body_link_pos_w[0].cpu().numpy().copy()
            )
            log["body_quat_w"].append(
                robot.data.body_link_quat_w[0].cpu().numpy().copy()
            )
            log["body_lin_vel_w"].append(
                robot.data.body_link_lin_vel_w[0].cpu().numpy().copy()
            )
            log["body_ang_vel_w"].append(
                robot.data.body_link_ang_vel_w[0].cpu().numpy().copy()
            )

            should_render = video_path is not None or (
                storyboard_path is not None and frame in storyboard_indices
            )
            if renderer is not None and should_render:
                renderer.update(sim.data)
                rendered_frame = renderer.render()
                if video_path is not None:
                    video_frames.append(rendered_frame)
                if storyboard_path is not None and frame in storyboard_indices:
                    storyboard_frames[frame] = rendered_frame
    finally:
        if renderer is not None:
            renderer.close()

    converted = {key: np.stack(values) for key, values in log.items()}
    converted["fps"] = np.asarray([output_fps], dtype=np.float32)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        output_path,
        fps=converted["fps"],
        joint_pos=converted["joint_pos"],
        joint_vel=converted["joint_vel"],
        body_pos_w=converted["body_pos_w"],
        body_quat_w=converted["body_quat_w"],
        body_lin_vel_w=converted["body_lin_vel_w"],
        body_ang_vel_w=converted["body_ang_vel_w"],
    )
    if video_path is not None:
        import mediapy

        video_path = Path(video_path)
        video_path.parent.mkdir(parents=True, exist_ok=True)
        mediapy.write_video(str(video_path), video_frames, fps=output_fps)
    if storyboard_path is not None:
        import mediapy

        storyboard_path = Path(storyboard_path)
        storyboard_path.parent.mkdir(parents=True, exist_ok=True)
        selected_frames = tuple(
            storyboard_frames[index] for index in sorted(storyboard_indices)
        )
        mediapy.write_image(
            str(storyboard_path), np.concatenate(selected_frames, axis=1)
        )
    return converted


def main() -> None:
    """Command-line entry point for converting a single recovery clip."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="source Legged Lab .pkl file")
    parser.add_argument("output", type=Path, help="destination mjlab .npz file")
    parser.add_argument("--output-fps", type=float, default=50.0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--video", type=Path, help="optional MP4 preview path")
    parser.add_argument(
        "--storyboard", type=Path, help="optional first/middle/last PNG path"
    )
    args = parser.parse_args()
    convert_motion(
        args.source,
        args.output,
        output_fps=args.output_fps,
        device=args.device,
        video_path=args.video,
        storyboard_path=args.storyboard,
    )


if __name__ == "__main__":
    main()
