"""Play a trained HoST flat-ground policy in mjlab's browser Viser viewer."""

import argparse
from pathlib import Path

import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.viewer import ViserPlayViewer

from g1recovery_host import host_ground29_env_cfg, host_ground_env_cfg
from g1recovery_host.action import PHYSICS_REVISION
from g1recovery_host.ppo import HostPolicy
from g1recovery_host.stand_guard import apply_stand_pose_guard


class DeterministicHostPolicy:
    """Adapt the HoST actor to mjlab's viewer policy interface."""

    def __init__(self, policy: HostPolicy, env=None, stand_pose_guard=False):
        self.policy = policy
        self.env = env
        self.stand_pose_guard = stand_pose_guard

    def __call__(self, observations: dict[str, torch.Tensor]) -> torch.Tensor:
        action = self.policy.deterministic_action(observations["actor"])
        if self.stand_pose_guard:
            action = apply_stand_pose_guard(self.env, action)
        return action


def hide_auxiliary_foot_spheres(spec):
    """Hide HoST's visual-only ankle markers without changing foot contacts."""

    hidden = 0
    for geom in spec.geoms:
        if geom.group == 1 and geom.parent.name.startswith(
            ("auxiliary_left_ankle_roll_link", "auxiliary_right_ankle_roll_link")
        ):
            geom.rgba[3] = 0.0
            hidden += 1
    if hidden != 8:
        raise RuntimeError(f"expected 8 auxiliary foot spheres, found {hidden}")
    return spec


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--task", choices=("flat", "flat29"), default="flat")
    parser.add_argument("--model-source", default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=50)
    parser.add_argument("--action-beta", type=float, default=0.25)
    parser.add_argument("--assist-force-n", type=float, default=0.0)
    parser.add_argument("--episode-seconds", type=float, default=5.0)
    parser.add_argument("--stand-pose-guard", action="store_true")
    args = parser.parse_args()
    if not 0.0 < args.action_beta <= 1.0 or args.assist_force_n < 0.0:
        parser.error("action beta must be in (0, 1] and assist force nonnegative")
    if args.episode_seconds <= 0.0:
        parser.error("episode duration must be positive")
    if args.stand_pose_guard and args.task != "flat29":
        parser.error("--stand-pose-guard requires --task flat29")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        parser.error("CUDA unavailable; use --device cpu or fix GPU access")

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.is_file():
        parser.error(f"checkpoint not found: {checkpoint_path}")
    checkpoint = torch.load(
        checkpoint_path, map_location=args.device, weights_only=False
    )
    checkpoint_reward = checkpoint.get("reward_variant", "")
    pose_version = (
        "v6"
        if checkpoint_reward.endswith("pose_v6")
        else "v5"
        if checkpoint_reward.endswith("pose_v5")
        else "v4"
        if checkpoint_reward.endswith("pose_v4")
        else "v3"
        if checkpoint_reward.endswith("pose_v3")
        else "v2"
    )
    is_29 = args.task == "flat29"
    model_source = args.model_source or (
        "unitree_g1_29" if is_29 else "host_urdf_capsule"
    )
    expected_revision = (
        f"{PHYSICS_REVISION}_flat29_29_v1" if is_29 else PHYSICS_REVISION
    )
    if checkpoint.get("task", "flat") != args.task:
        parser.error("checkpoint and requested task differ")
    if checkpoint.get("model_source", "mjlab") != model_source:
        parser.error("checkpoint and requested model source differ")
    if checkpoint.get("physics_revision", "single_torso_force_v1") != expected_revision:
        parser.error("checkpoint and current physics revision differ")

    policy = HostPolicy(
        obs_dim=564 if is_29 else 456,
        action_dim=29 if is_29 else 23,
        upper_residual=is_29 and pose_version in ("v4", "v5", "v6"),
        waist_residual_scale=5.0 if pose_version in ("v5", "v6") else 2.0,
    ).to(args.device)
    policy.load_state_dict(checkpoint["policy"])
    policy.eval()
    if is_29:
        cfg = host_ground29_env_cfg(
            1,
            args.seed,
            pose_version=pose_version,
        )
    else:
        cfg = host_ground_env_cfg(
            1,
            args.seed,
            model_source=model_source,
            legacy_style=checkpoint.get("reward_variant", "paper")
            in ("paper", "height_guard_v1"),
        )
    cfg.episode_length_s = args.episode_seconds
    cfg.viewer.lookat = (0.0, 0.0, 0.6)
    cfg.viewer.distance = 2.4
    cfg.viewer.azimuth = 120.0
    cfg.viewer.elevation = -18.0
    if model_source in ("host_urdf", "host_urdf_capsule"):
        robot_cfg = cfg.scene.entities["robot"]
        source_spec_fn = robot_cfg.spec_fn
        robot_cfg.spec_fn = lambda: hide_auxiliary_foot_spheres(source_spec_fn())
    env = ManagerBasedRlEnv(cfg, device=args.device)
    try:
        action_term = env.action_manager.get_term("joint_pos")
        action_term.beta.fill_(args.action_beta)
        action_term.assist_force.fill_(args.assist_force_n)
        print("Open the Viser URL shown below (usually http://localhost:8080).")
        ViserPlayViewer(
            env, DeterministicHostPolicy(policy, env, args.stand_pose_guard)
        ).run()
    finally:
        env.close()


if __name__ == "__main__":
    main()
