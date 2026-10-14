"""Replay a 29-DoF HoST A6 checkpoint in mjlab's browser Viser viewer."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.viewer import ViserPlayViewer

from g1recovery_host.a6_geometry import A6_PHYSICS_SUFFIX, DIRECTIONS, quotas
from g1recovery_host.a6_reset import MAX_GUIDED_EDGE_OFFSET_M
from g1recovery_host.a6_reward import A6_RESET_VARIANT
from g1recovery_host.a6_task import host_a6_env_cfg
from g1recovery_host.action import PHYSICS_REVISION
from g1recovery_host.ppo import HostPolicy
from g1recovery_host.phase_switch import reset_prone_history_after_escape

SCENE_IDS = {"flat": 0, "guided": 1, "free": 2}
FOUR_VIEW_LAYOUT = (
    (2, 0, 0),  # free plate, supine
    (1, 1, 0),  # guided (green) plate, prone
    (1, 0, 0),  # guided (green) plate, supine
    (0, 0, 0),  # flat, supine
)


class DeterministicHostPolicy:
    """Adapt the trained actor to mjlab's viewer policy interface."""

    def __init__(self, policy: HostPolicy):
        self.policy = policy

    def __call__(self, observations: dict[str, torch.Tensor]) -> torch.Tensor:
        return self.policy.deterministic_action(observations["actor"])


def selected_env(scene: str, direction: str) -> int:
    """Use the first low-stratum world with this scene and start direction."""

    scenes, directions, strata = quotas(32)
    matches = np.flatnonzero(
        (scenes == SCENE_IDS[scene])
        & (directions == DIRECTIONS.index(direction))
        & (strata == 0)
    )
    if not len(matches):
        raise ValueError(f"A6 has no low reset for {scene}/{direction}")
    return int(matches[0])


def saved_curriculum(checkpoint: dict, name: str, fallback: float) -> float:
    value = checkpoint.get("curriculum", {}).get(name)
    return float(value.float().mean()) if value is not None else fallback


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--scene", choices=tuple(SCENE_IDS), default="guided")
    parser.add_argument("--direction", choices=DIRECTIONS, default="supine")
    parser.add_argument(
        "--four-view",
        action="store_true",
        help="show free, guided prone, guided supine, and flat worlds together",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--episode-seconds", type=float, default=None)
    parser.add_argument("--action-beta", type=float, default=None)
    parser.add_argument("--assist-force-n", type=float, default=None)
    parser.add_argument("--guided-offset-m", type=float, default=0.0)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    if args.episode_seconds is not None and args.episode_seconds <= 0:
        parser.error("episode duration must be positive")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        parser.error("CUDA unavailable; use --device cpu or fix GPU access")

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.is_file():
        parser.error(f"checkpoint not found: {checkpoint_path}")
    checkpoint = torch.load(
        checkpoint_path, map_location=args.device, weights_only=False
    )
    episode_seconds = (
        args.episode_seconds
        if args.episode_seconds is not None
        else float(checkpoint.get("a6_episode_seconds", 10.0))
    )
    if not math.isfinite(episode_seconds) or episode_seconds <= 0:
        parser.error("checkpoint episode duration must be finite and positive")
    expected_revision = f"{PHYSICS_REVISION}_{A6_PHYSICS_SUFFIX}"
    historical_revisions = (f"{PHYSICS_REVISION}_a6_29_v1",)
    if (
        checkpoint.get("task") != "a6"
        or checkpoint.get("model_source") != "unitree_g1_29"
        or checkpoint.get("physics_revision")
        not in (expected_revision, *historical_revisions)
        or checkpoint.get("arm") not in ("G+", "G-")
    ):
        parser.error("checkpoint is not compatible with the 29-DoF A6 task")
    if checkpoint["physics_revision"] == historical_revisions[0]:
        print(
            "WARNING: this checkpoint used a plate mass curriculum (initially "
            "4–6 kg, widening toward 4–12 kg); "
            "the current replay uses fixed 6 kg plates.",
            flush=True,
        )
    if checkpoint.get("reward_variant", "").endswith(
        ("pose_v6_v2", "pose_v6_v2_frozen_flat_teacher_v1")
    ):
        print(
            "WARNING: this checkpoint predates the strict guided-board escape "
            "and overload/lift reset rules; replay uses the new rules.",
            flush=True,
        )
    if checkpoint.get("a6_reset_variant") != A6_RESET_VARIANT:
        print(
            "WARNING: checkpoint was trained with different plate reset rules; "
            "this replay resets at >5000 N plate force or >5 cm guided-plate "
            "lift after motor activation while still covered; "
            "contact depth is diagnostic only; the no-contact grace period "
            "uses real control steps since reset.",
            flush=True,
        )

    beta = (
        args.action_beta
        if args.action_beta is not None
        else saved_curriculum(checkpoint, "beta", 1.0)
    )
    assist = (
        args.assist_force_n
        if args.assist_force_n is not None
        else saved_curriculum(checkpoint, "assist_force", 200.0)
    )
    if not 0.0 < beta <= 1.0 or assist < 0.0:
        parser.error("action beta must be in (0, 1] and assist force nonnegative")

    pose_v6_variants = tuple(
        f"historical_a6_{checkpoint['arm']}_29dof_pose_v6_v{version}"
        for version in (2, 3, 4, 5, 6, 7)
    )
    edge_curriculum = "_v8_edge_curriculum_v1" in checkpoint.get("reward_variant", "")
    raw_variant = checkpoint.get("reward_variant", "")
    safe_actions = "_safe_actions_v1" in raw_variant
    aligned_edge = "_aligned_edge_v1" in raw_variant
    lateral_actions = "_lateral_actions_v1" in raw_variant
    lift_guard = "_lift_guard_v1" in raw_variant
    knee_straight_prior = "_knee_straight_prior_v1" in raw_variant
    footprint_edge = "_footprint_edge_v1" in raw_variant
    arm_extraction = "_arm_extraction_v1" in raw_variant
    prone_full_arms = "_prone_full_arms_v1" in raw_variant
    supine_full_arms = "_supine_full_arms_v1" in raw_variant
    pose_variant = (
        raw_variant.removesuffix("_guided_teacher_v1")
        .removesuffix("_episode_15s_v1")
        .removesuffix("_frozen_supine_teacher_v1")
        .removesuffix("_postescape_live_obs_v1")
        .removesuffix("_phase_history_reset_v1")
        .removesuffix("_bidirectional_recovery_v1")
        .removesuffix("_prone_recovery_v1")
        .removesuffix("_postescape_prone_replay_v1")
        .removesuffix("_supine_full_arms_v1")
        .removesuffix("_prone_full_arms_v1")
        .removesuffix("_arm_extraction_v1")
        .removesuffix("_footprint_edge_v1")
        .removesuffix("_knee_straight_prior_v1")
        .removesuffix("_lift_guard_v1")
        .removesuffix("_lateral_actions_v1")
        .removesuffix("_aligned_edge_v1")
        .removesuffix("_progress_aux_v1")
        .removesuffix("_safe_actions_v1")
        .removesuffix("_focused_guided_v1")
    )
    staged_escape_v1 = pose_variant.endswith("_staged_escape_v1")
    staged_escape_v2 = pose_variant.endswith("_staged_escape_v2")
    staged_escape_v3 = pose_variant.endswith("_staged_escape_v3")
    staged_escape_v4 = pose_variant.endswith("_staged_escape_v4")
    staged_escape_v5 = pose_variant.endswith("_staged_escape_v5")
    pose_variant = (
        pose_variant.removesuffix("_staged_escape_v1")
        .removesuffix("_staged_escape_v2")
        .removesuffix("_staged_escape_v3")
        .removesuffix("_staged_escape_v4")
        .removesuffix("_staged_escape_v5")
    )
    for suffix in ("_preescape_knee_v2", "_preescape_knee_v1"):
        pose_variant = pose_variant.removesuffix(suffix)
    pose_v6 = pose_variant in (
        *pose_v6_variants,
        "historical_a6_G+_29dof_pose_v6_v8_edge_curriculum_v1",
        *(variant + "_frozen_flat_teacher_v1" for variant in pose_v6_variants),
        "historical_a6_G+_29dof_pose_v6_v8_edge_curriculum_v1_frozen_flat_teacher_v1",
    )
    if not 0 <= args.guided_offset_m <= MAX_GUIDED_EDGE_OFFSET_M or (
        args.guided_offset_m > 0 and not edge_curriculum
    ):
        parser.error(
            "guided offset requires a v8 checkpoint and must be in [0, 0.50] m"
        )
    has_residual = "upper_residual.0.weight" in checkpoint["policy"]
    if pose_v6 != has_residual:
        parser.error("checkpoint pose-v6 metadata and residual weights disagree")
    policy = HostPolicy(
        obs_dim=612 if staged_escape_v5 else 564,
        action_dim=29,
        log_std_bounds=(-2.3, math.log(checkpoint.get("action_std_max", 0.7)))
        if pose_v6 else None,
        upper_residual=pose_v6,
        waist_residual_scale=5.0 if pose_v6 else 2.0,
    ).to(args.device)
    policy.load_state_dict(checkpoint["policy"])
    policy.eval()
    num_envs = 4 if args.four_view else 32
    env_idx = 0 if args.four_view else selected_env(args.scene, args.direction)
    cfg = host_a6_env_cfg(
        num_envs,
        args.seed,
        arm=checkpoint["arm"],
        edge_curriculum=edge_curriculum,
        plate_observation=staged_escape_v5,
        safe_actions=safe_actions,
        prone_full_arms=prone_full_arms,
        supine_full_arms=supine_full_arms,
        aligned_edge=aligned_edge,
        lateral_actions=lateral_actions,
        lift_guard=lift_guard,
        footprint_edge=footprint_edge,
        footprint_overlap_m=checkpoint.get("a6_footprint_overlap_m", 0.25),
    )
    if args.four_view:
        cfg.events["reset_scene"].params["layout"] = FOUR_VIEW_LAYOUT
    cfg.episode_length_s = episode_seconds
    cfg.viewer.env_idx = env_idx
    cfg.viewer.max_extra_envs = 3 if args.four_view else 0
    cfg.viewer.lookat = (0.0, 0.0, 0.6)
    cfg.viewer.distance = 8.5 if args.four_view else 2.6
    cfg.viewer.azimuth = 120.0
    cfg.viewer.elevation = -18.0
    env = ManagerBasedRlEnv(cfg, device=args.device)
    try:
        env._a6_staged_escape_v1 = staged_escape_v1
        env._a6_staged_escape_v2 = staged_escape_v2
        env._a6_staged_escape_v3 = staged_escape_v3
        env._a6_staged_escape_v4 = staged_escape_v4
        env._a6_staged_escape_v5 = staged_escape_v5
        env._a6_knee_straight_prior_v1 = knee_straight_prior
        env._a6_arm_extraction_v1 = arm_extraction
        env._a6_prone_recovery_v1 = "_prone_recovery_v1" in raw_variant
        env._a6_bidirectional_recovery_v1 = "_bidirectional_recovery_v1" in raw_variant
        env._a6_postescape_live_obs_v1 = "_postescape_live_obs_v1" in raw_variant
        env._a6_edge_offset_m = args.guided_offset_m
        # Viser's first tick calls get_observations()/step() without resetting
        # the env. A6 scene IDs, plate geom IDs, and reset banks are created by
        # reset_scene, so initialize them before handing control to the viewer.
        observations, _ = env.reset()
        if "_phase_history_reset_v1" in raw_variant:
            physical_step = env.step

            def step_with_recovery_phase(actions):
                escaped_before = env._a6_escaped.clone()
                obs, reward, terminated, truncated, extras = physical_step(actions)
                obs, _ = reset_prone_history_after_escape(
                    env, escaped_before, terminated | truncated
                )
                return obs, reward, terminated, truncated, extras

            env.step = step_with_recovery_phase
        action_term = env.action_manager.get_term("joint_pos")
        action_term.beta.fill_(beta)
        action_term.assist_force.fill_(assist)
        layout_name = (
            "free/supine, guided/prone, guided/supine, flat/supine"
            if args.four_view
            else f"{args.scene}/{args.direction} env_idx={env_idx}"
        )
        print(
            f"arm={checkpoint['arm']} layout={layout_name} "
            f"beta={beta:.4f} assist_force_n={assist:.2f} "
            f"episode_seconds={episode_seconds:g}",
            flush=True,
        )
        if args.validate_only:
            if args.four_view:
                actual = torch.stack(
                    (env._a6_scene, env._a6_direction, env._a6_stratum), dim=1
                )
                expected = torch.tensor(FOUR_VIEW_LAYOUT, device=env.device)
                if not torch.equal(actual, expected):
                    raise RuntimeError("four-view A6 worlds do not match layout")
            elif int(env._a6_scene[env_idx]) != SCENE_IDS[args.scene] or int(
                env._a6_direction[env_idx]
            ) != DIRECTIONS.index(args.direction):
                raise RuntimeError("selected A6 world does not match requested scene")
            actions = policy.deterministic_action(observations["actor"])
            if actions.shape != (num_envs, 29) or not torch.isfinite(actions).all():
                raise RuntimeError("checkpoint actor returned invalid 29-DoF actions")
            env.step(actions)
            print("A6 playback validation passed", flush=True)
            return
        print("Open the Viser URL shown below (usually http://localhost:8080).")
        ViserPlayViewer(env, DeterministicHostPolicy(policy)).run()
    finally:
        env.close()


if __name__ == "__main__":
    main()
