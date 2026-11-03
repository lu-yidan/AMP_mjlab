"""Replay a 29-DoF HoST A6 checkpoint in mjlab's browser Viser viewer."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.viewer import ViserPlayViewer

from g1recovery_host.a6_geometry import (
    A6_PHYSICS_SUFFIX,
    DIRECTIONS,
    footprint,
    plate_footprint_inputs,
    quotas,
)
from g1recovery_host.a6_reset import MAX_GUIDED_EDGE_OFFSET_M
from g1recovery_host.a6_reward import A6_RESET_VARIANT
from g1recovery_host.a6_task import host_a6_env_cfg
from g1recovery_host.a6_distill import (
    apply_side_lateral_gain, effective_guided_side_teacher_action,
)
from g1recovery_host.action import PHYSICS_REVISION
from g1recovery_host.phase_switch import (
    postescape_live_observation_enabled,
    reset_prone_history_after_escape,
)
from g1recovery_host.ppo import HostPolicy
from g1recovery_host.proprio_recovery import (
    InitialProneRouter,
    InitialSideRouter,
    load_prone_specialist,
    load_recovery_controller,
    load_supine_flat_gate,
    validate_controller_provenance,
)
from g1recovery_host.proprio_scene import (
    ProprioGuidedSideGate, ProprioSceneGate, load_scene_classifier,
)
from g1recovery_host.proprio_symmetry import (
    mirror_host29_action, mirror_host29_actor_obs,
)

SCENE_IDS = {"flat": 0, "guided": 1, "free": 2}
FOUR_VIEW_LAYOUT = (
    (2, 0, 0),  # free plate, supine
    (1, 1, 0),  # guided (green) plate, prone
    (1, 0, 0),  # guided (green) plate, supine
    (0, 0, 0),  # flat, supine
)
SIDE_FOUR_VIEW_LAYOUT = (
    (1, 2, 0),  # guided plate, left-side down
    (1, 3, 0),  # guided plate, right-side down
    (2, 3, 0),  # free plate, right-side down
    (0, 3, 0),  # flat ground, right-side down
)


class DeterministicHostPolicy:
    """Adapt the trained actor to mjlab's viewer policy interface."""

    def __init__(self, policy: HostPolicy, env=None, flat_expert=None, flat_gate=None,
                 prone_expert=None, prone_router=None, side_expert=None, side_router=None,
                 guided_side_gate=None, mirror_left_side=False,
                 mirror_right_main=False, side_safe_through_step=0,
                 side_safe_scale=0.5, side_safe_probe_through_step=0,
                 left_side_specialist_probe_through_step=0,
                 side_handoff_steps=0, side_lateral_gain=1.0,
                 prone_flat_gate=None, recovery_controller=None,
                 prone_flat_pulse_step=50, prone_free_gate=None,
                 prone_free_pulse_step=100):
        self.policy = policy
        self.env = env
        self.flat_expert = flat_expert
        self.flat_gate = flat_gate
        self.prone_expert = prone_expert
        self.prone_router = prone_router
        self.side_expert = side_expert
        self.side_router = side_router
        self.guided_side_gate = guided_side_gate
        self.mirror_left_side = mirror_left_side
        self.mirror_right_main = mirror_right_main
        self.side_safe_through_step = side_safe_through_step
        self.side_safe_scale = side_safe_scale
        self.side_safe_probe_through_step = side_safe_probe_through_step
        self.left_side_specialist_probe_through_step = left_side_specialist_probe_through_step
        self.side_handoff_steps = side_handoff_steps
        self.side_lateral_gain = side_lateral_gain
        self.prone_flat_gate = prone_flat_gate
        self.recovery_controller = recovery_controller
        self.prone_flat_pulse_step = prone_flat_pulse_step
        self.prone_free_gate = prone_free_gate
        self.prone_free_pulse_step = prone_free_pulse_step

    def __call__(self, observations: dict[str, torch.Tensor]) -> torch.Tensor:
        actor_obs = observations["actor"]
        if self.prone_flat_gate is not None:
            real_steps = self.env.action_manager.get_term("joint_pos").real_steps
            self.prone_flat_gate.observe(actor_obs, real_steps)
            self.recovery_controller.schedule_pulse(
                self.prone_flat_gate.triggered,
                real_steps, self.prone_flat_pulse_step,
            )
        if self.prone_free_gate is not None:
            real_steps = self.env.action_manager.get_term("joint_pos").real_steps
            self.prone_free_gate.observe(actor_obs, real_steps)
            self.recovery_controller.schedule_pulse(
                self.prone_free_gate.triggered,
                real_steps, self.prone_free_pulse_step,
            )
        action = self.policy.deterministic_action(actor_obs)
        if self.mirror_right_main:
            mirrored_right = (
                self.side_router.initial_side & ~self.side_router.initial_left_side
            )
            if mirrored_right.any():
                mirrored_input = mirror_host29_actor_obs(actor_obs[mirrored_right])
                action[mirrored_right] = mirror_host29_action(
                    self.policy.deterministic_action(mirrored_input)
                )
        if self.prone_expert is not None:
            prone_action = self.prone_expert.deterministic_action(actor_obs)
            action = torch.where(
                self.prone_router.initial_prone[:, None], prone_action, action,
            )
        pre_side_action = action
        if self.side_expert is not None:
            if self.guided_side_gate is not None:
                real_steps = self.env.action_manager.get_term("joint_pos").real_steps
                self.guided_side_gate.observe(actor_obs, real_steps)
            side_action = self.side_expert.deterministic_action(actor_obs)
            if self.mirror_left_side:
                mirrored_left = self.side_router.initial_left_side
                if mirrored_left.any():
                    mirrored_input = mirror_host29_actor_obs(actor_obs[mirrored_left])
                    side_action[mirrored_left] = mirror_host29_action(
                        self.side_expert.deterministic_action(mirrored_input)
                    )
            side_route = (
                self.guided_side_gate.triggered
                if self.guided_side_gate is not None
                else self.side_router.initial_side
            )
            if self.left_side_specialist_probe_through_step:
                real_steps = self.env.action_manager.get_term("joint_pos").real_steps
                side_route = side_route | (
                    self.side_router.initial_left_side
                    & (real_steps <= self.left_side_specialist_probe_through_step)
                )
            action = torch.where(
                side_route[:, None], side_action, action,
            )
        if self.flat_gate is not None:
            real_steps = self.env.action_manager.get_term("joint_pos").real_steps
            self.flat_gate.observe(actor_obs, real_steps)
            flat_action = self.flat_expert.deterministic_action(actor_obs[:, :564])
            action = torch.where(self.flat_gate.triggered[:, None], flat_action, action)
        if self.side_safe_through_step:
            real_steps = self.env.action_manager.get_term("joint_pos").real_steps
            side_safe = self.side_router.initial_side & (real_steps <= self.side_safe_through_step)
            if self.guided_side_gate is not None:
                side_safe &= (
                    self.guided_side_gate.triggered
                    | (real_steps <= self.side_safe_probe_through_step)
                    | (
                        self.side_router.initial_left_side
                        & (real_steps <= self.left_side_specialist_probe_through_step)
                    )
                )
            if side_safe.any():
                action = action.clone()
                action[side_safe] = effective_guided_side_teacher_action(
                    action[side_safe], scale=self.side_safe_scale
                )
        if self.side_lateral_gain > 1.0:
            real_steps = self.env.action_manager.get_term("joint_pos").real_steps
            lateral = (
                self.guided_side_gate.triggered
                & (real_steps <= self.side_safe_through_step)
            )
            action = apply_side_lateral_gain(
                action, lateral, self.side_lateral_gain,
            )
        if self.side_handoff_steps:
            age = self.guided_side_gate.trigger_age
            blending = (age > 0) & (age < self.side_handoff_steps)
            if blending.any():
                alpha = (age.float() / self.side_handoff_steps).clamp(0, 1)
                action = torch.where(
                    blending[:, None],
                    torch.lerp(pre_side_action, action, alpha[:, None]),
                    action,
                )
        return action


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
    parser.add_argument("--side-four-view", action="store_true",
                        help="show guided left/right, free right, and flat right together")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--episode-seconds", type=float, default=None)
    parser.add_argument("--action-beta", type=float, default=None)
    parser.add_argument("--assist-force-n", type=float, default=None)
    parser.add_argument(
        "--guided-offset-m", type=float, default=0.0,
        help="0 centres the green board over the torso; 0.25 selects the easy edge tier",
    )
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--validate-steps", type=int, default=1,
                        help="number of control steps to run with --validate-only")
    parser.add_argument("--proprio-recovery-controller", type=Path,
                        help="prone recovery pulse driven only by actor proprioception")
    parser.add_argument("--recovery-pulse-blend-steps", type=int, default=0,
                        help="blend both ends of the recovery pulse over N control steps")
    parser.add_argument("--proprio-prone-specialist", type=Path,
                        help="738D initial-prone actor selected from reset IMU only")
    parser.add_argument("--proprio-prone-flat-gate", type=Path,
                        help="738D proprio scene cue that schedules an early flat-prone recovery pulse")
    parser.add_argument("--prone-flat-pulse-step", type=int, default=50)
    parser.add_argument("--prone-flat-threshold", type=float, default=0.99)
    parser.add_argument("--prone-flat-consecutive", type=int, default=3)
    parser.add_argument("--proprio-prone-free-gate", type=Path,
                        help="738D proprio scene cue that schedules an early free-prone recovery pulse")
    parser.add_argument("--prone-free-pulse-step", type=int, default=100)
    parser.add_argument("--prone-free-threshold", type=float, default=0.99)
    parser.add_argument("--prone-free-consecutive", type=int, default=3)
    parser.add_argument("--proprio-side-specialist", type=Path,
                        help="738D initial-side actor selected from reset IMU only")
    parser.add_argument("--proprio-guided-side-gate", type=Path,
                        help="738D proprio classifier for a fixed-board side policy")
    parser.add_argument("--side-handoff-steps", type=int, default=0,
                        help="blend main and side-specialist targets for N steps after the gate latches")
    parser.add_argument("--proprio-guided-side-early-gate", type=Path,
                        help="optional second classifier for the early side contact window")
    parser.add_argument("--proprio-guided-right-side-gate", type=Path,
                        help="optional separate classifier for initial right side")
    parser.add_argument("--guided-side-threshold", type=float, default=0.9)
    parser.add_argument("--guided-side-consecutive", type=int, default=2)
    parser.add_argument("--guided-side-ema-alpha", type=float, default=0.2)
    parser.add_argument("--guided-side-min-steps", type=int, default=35)
    parser.add_argument("--guided-side-max-steps", type=int, default=120)
    parser.add_argument("--guided-side-early-max-step", type=int, default=45)
    parser.add_argument("--guided-right-side-min-steps", type=int, default=35)
    parser.add_argument("--guided-right-side-max-steps", type=int, default=120)
    parser.add_argument("--mirror-left-side-specialist", action="store_true")
    parser.add_argument("--mirror-right-initial-side-main", action="store_true")
    parser.add_argument("--side-safe-through-step", type=int, default=0)
    parser.add_argument("--side-safe-probe-through-step", type=int, default=0)
    parser.add_argument("--left-side-specialist-probe-through-step", type=int, default=0)
    parser.add_argument("--side-safe-scale", type=float, default=0.5)
    parser.add_argument("--side-lateral-gain", type=float, default=1.0)
    parser.add_argument("--proprio-supine-flat-gate", type=Path,
                        help="proprio-only supine phase gate for the frozen flat expert")
    parser.add_argument("--flat-expert-checkpoint", type=Path)
    parser.add_argument("--flat-gate-threshold", type=float, default=0.8)
    parser.add_argument("--flat-gate-consecutive", type=int, default=3)
    parser.add_argument("--flat-gate-min-steps", type=int, default=120)
    args = parser.parse_args()
    if args.four_view and args.side_four_view:
        parser.error("choose one four-view layout")
    if (args.proprio_supine_flat_gate is None) != (args.flat_expert_checkpoint is None):
        parser.error("supine flat gate and flat expert checkpoint must be supplied together")
    if args.proprio_guided_side_gate is not None and args.proprio_side_specialist is None:
        parser.error("guided-side gate requires a side specialist")
    if args.side_handoff_steps < 0 or (
        args.side_handoff_steps and args.proprio_guided_side_gate is None
    ):
        parser.error("side handoff blending requires a guided-side gate and nonnegative steps")
    if not 1.0 <= args.side_lateral_gain <= 1.5 or (
        args.side_lateral_gain > 1.0
        and (args.proprio_guided_side_gate is None or not args.side_safe_through_step)
    ):
        parser.error("side lateral gain requires [1, 1.5], a guided-side gate and an action horizon")
    if args.recovery_pulse_blend_steps < 0 or (
        args.recovery_pulse_blend_steps and args.proprio_recovery_controller is None
    ):
        parser.error("recovery pulse blending requires its controller and nonnegative steps")
    if args.proprio_prone_flat_gate is not None and not (
        args.proprio_recovery_controller is not None
        and args.proprio_prone_specialist is not None
        and 31 <= args.prone_flat_pulse_step <= 120
        and 0 < args.prone_flat_threshold < 1
        and 1 <= args.prone_flat_consecutive <= 10
    ):
        parser.error("prone-flat gate needs a prone actor, controller, and valid pulse window")
    if args.proprio_prone_free_gate is not None and not (
        args.proprio_recovery_controller is not None
        and args.proprio_prone_specialist is not None
        and 31 <= args.prone_free_pulse_step <= 300
        and 0 < args.prone_free_threshold < 1
        and 1 <= args.prone_free_consecutive <= 10
    ):
        parser.error("prone-free gate needs a prone actor, controller, and valid pulse window")
    if args.proprio_guided_side_early_gate is not None and (
        args.proprio_guided_side_gate is None
        or not 31 <= args.guided_side_early_max_step <= args.guided_side_max_steps
    ):
        parser.error("early side gate requires the primary gate and a valid early window")
    if args.proprio_guided_right_side_gate is not None and (
        args.proprio_guided_side_gate is None
        or not 31 <= args.guided_right_side_min_steps <= args.guided_right_side_max_steps
    ):
        parser.error("right side gate requires the primary gate and a valid right-side window")
    if args.mirror_left_side_specialist and args.proprio_side_specialist is None:
        parser.error("left-side mirroring requires a side specialist")
    if args.side_safe_through_step and (
        args.proprio_side_specialist is None or args.side_safe_through_step <= 30
    ):
        parser.error("side safety window requires a side specialist and at least 31 control steps")
    if args.side_safe_probe_through_step and not (
        args.proprio_guided_side_gate is not None
        and 30 < args.side_safe_probe_through_step <= args.side_safe_through_step
    ):
        parser.error("side safety probe requires a guided-side gate and a valid safety window")
    if args.left_side_specialist_probe_through_step and not (
        args.proprio_guided_side_gate is not None
        and args.mirror_left_side_specialist
        and 30 < args.left_side_specialist_probe_through_step <= args.side_safe_through_step
    ):
        parser.error("left side expert probe requires the gate, mirrored expert, and a safety window")
    if not 0 < args.side_safe_scale <= 1:
        parser.error("side safe scale must be in (0, 1]")
    if not (0 < args.guided_side_threshold < 1
            and args.guided_side_consecutive > 0
            and 0 < args.guided_side_ema_alpha <= 1
            and 30 < args.guided_side_min_steps <= args.guided_side_max_steps):
        parser.error("invalid guided-side gate parameters")
    if args.episode_seconds is not None and args.episode_seconds <= 0:
        parser.error("episode duration must be positive")
    if args.validate_steps < 1:
        parser.error("validation requires at least one control step")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        parser.error("CUDA unavailable; use --device cpu or fix GPU access")

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.is_file():
        parser.error(f"checkpoint not found: {checkpoint_path}")
    checkpoint = torch.load(
        checkpoint_path, map_location=args.device, weights_only=False
    )
    validate_controller_provenance(checkpoint, args.proprio_recovery_controller)
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
    torque_v2 = checkpoint.get("observation_contract") == "proprio_torque_actor_privileged_critic_v2"
    proprio_actor = torque_v2 or checkpoint.get("observation_contract") == "proprio_actor_privileged_critic_v1"
    if (args.proprio_guided_side_gate is not None
            or args.mirror_left_side_specialist
            or args.mirror_right_initial_side_main
            or args.side_safe_through_step) and not torque_v2:
        parser.error("side-scene controller requires the 738D proprio actor")
    if args.proprio_recovery_controller is not None and (
        not torque_v2 or "_phase_history_reset_v1" in raw_variant
    ):
        parser.error("packaged recovery controller requires the 738D proprio actor without legacy phase reset")
    if args.proprio_supine_flat_gate is not None and (
        not torque_v2 or "_phase_history_reset_v1" in raw_variant
    ):
        parser.error("supine flat gate requires the 738D proprio actor without legacy phase reset")
    if proprio_actor and (
        checkpoint.get("actor_obs_dim") != (738 if torque_v2 else 564)
        or checkpoint.get("critic_obs_dim") != (786 if torque_v2 else 612)
        or checkpoint.get("action_contract") != "scene_independent_v1"
    ):
        parser.error("proprioceptive checkpoint has inconsistent observation or action metadata")
    safe_actions = "_safe_actions_v1" in raw_variant
    aligned_edge = "_aligned_edge_v1" in raw_variant
    lateral_actions = "_lateral_actions_v1" in raw_variant
    lift_guard = "_lift_guard_v1" in raw_variant
    if proprio_actor and (safe_actions or lift_guard):
        parser.error("proprioceptive checkpoint cannot enable legacy scene-dependent actions")
    knee_straight_prior = "_knee_straight_prior_v1" in raw_variant
    footprint_edge = "_footprint_edge_v1" in raw_variant
    arm_extraction = "_arm_extraction_v1" in raw_variant
    prone_full_arms = "_prone_full_arms_v1" in raw_variant
    supine_full_arms = "_supine_full_arms_v1" in raw_variant
    pose_variant = (
        raw_variant.removesuffix("_frozen_preescape_parent_v1")
        .removesuffix("_free_plate_anchor_v1")
        .removesuffix("_guided_supine_teacher_boost_v1")
        .removesuffix("_free_plate_motion_reward_v1")
        .removesuffix("_scene_phase_reward_v1")
        .removesuffix("_supine_postescape_flat_teacher_v1")
        .removesuffix("_prone_flexible_exit_v1")
        .removesuffix("_proprio_recovery_controller_v1")
        .removesuffix("_proprio_actor_v1")
        .removesuffix("_proprio_torque_v2")
        .removesuffix("_recovery_bridge_v2_prone_only")
        .removesuffix("_recovery_bridge_v2")
        .removesuffix("_strong_recovery_v1")
        .removesuffix("_recovery_teacher_v1")
        .removesuffix("_postescape_half_prone_v1")
        .removesuffix("_proprio_postescape_replay_v1")
        .removesuffix("_teacher_forced_quarter_v1")
        .removesuffix("_effective_action_distill_v1")
        .removesuffix("_hard_mix_v1")
        .removesuffix("_guided_teacher_v1")
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
        obs_dim=738 if torque_v2 else 564 if proprio_actor else 612 if staged_escape_v5 else 564,
        critic_obs_dim=(786 if torque_v2 else 612) if proprio_actor else None,
        action_dim=29,
        log_std_bounds=(-2.3, math.log(checkpoint.get("action_std_max", 0.7)))
        if pose_v6 else None,
        upper_residual=pose_v6,
        waist_residual_scale=5.0 if pose_v6 else 2.0,
    ).to(args.device)
    policy.load_state_dict(checkpoint["policy"])
    policy.eval()
    prone_expert = None
    if args.proprio_prone_specialist is not None:
        prone_expert, _ = load_prone_specialist(
            args.proprio_prone_specialist, checkpoint, args.device,
        )
    side_expert = None
    if args.proprio_side_specialist is not None:
        side_expert, _ = load_prone_specialist(
            args.proprio_side_specialist, checkpoint, args.device,
        )
    flat_expert = None
    if args.flat_expert_checkpoint is not None:
        source = torch.load(
            args.flat_expert_checkpoint, map_location=args.device, weights_only=False
        )
        if source.get("task") != "flat29" or source.get("model_source") != "unitree_g1_29":
            parser.error("flat expert must be a 29-DoF flat checkpoint")
        flat_expert = HostPolicy(
            564, 29,
            log_std_bounds=(-2.3, math.log(source.get("action_std_max", 0.7))),
            upper_residual=True, waist_residual_scale=5.0,
        ).to(args.device)
        flat_expert.load_state_dict(source["policy"], strict=True)
        flat_expert.eval()
    four_view = args.four_view or args.side_four_view
    num_envs = 4 if four_view else 32
    env_idx = 0 if four_view else selected_env(args.scene, args.direction)
    cfg = host_a6_env_cfg(
        num_envs,
        args.seed,
        arm=checkpoint["arm"],
        edge_curriculum=edge_curriculum,
        plate_observation=staged_escape_v5 and not proprio_actor,
        critic_plate_observation=torque_v2,
        proprio_motor_torque=torque_v2,
        safe_actions=safe_actions,
        prone_full_arms=prone_full_arms,
        supine_full_arms=supine_full_arms,
        aligned_edge=aligned_edge,
        lateral_actions=lateral_actions,
        lift_guard=lift_guard,
        privileged_guided_actions=not proprio_actor,
        footprint_edge=footprint_edge,
        footprint_overlap_m=checkpoint.get("a6_footprint_overlap_m", 0.25),
    )
    if four_view:
        cfg.events["reset_scene"].params["layout"] = (
            SIDE_FOUR_VIEW_LAYOUT if args.side_four_view else FOUR_VIEW_LAYOUT
        )
    cfg.episode_length_s = episode_seconds
    cfg.viewer.env_idx = env_idx
    cfg.viewer.max_extra_envs = 3 if four_view else 0
    cfg.viewer.lookat = (0.0, 0.0, 0.6)
    cfg.viewer.distance = 8.5 if four_view else 2.6
    cfg.viewer.azimuth = 120.0
    cfg.viewer.elevation = -18.0
    env = ManagerBasedRlEnv(cfg, device=args.device)
    try:
        prone_router = (
            InitialProneRouter(num_envs, args.device)
            if prone_expert is not None else None
        )
        side_router = (
            InitialSideRouter(num_envs, args.device)
            if side_expert is not None or args.mirror_right_initial_side_main else None
        )
        guided_side_gate = None
        if args.proprio_guided_side_gate is not None:
            classifier, scene_artifact = load_scene_classifier(
                args.proprio_guided_side_gate, args.device
            )
            if (scene_artifact["observation_contract"] != checkpoint["observation_contract"]
                    or scene_artifact.get("posture") != "side"):
                parser.error("guided-side classifier has an incompatible observation contract")
            early_classifier = None
            if args.proprio_guided_side_early_gate is not None:
                early_classifier, early_artifact = load_scene_classifier(
                    args.proprio_guided_side_early_gate, args.device
                )
                if (early_artifact["observation_contract"] != checkpoint["observation_contract"]
                        or early_artifact.get("posture") != "side"):
                    parser.error("early guided-side classifier has an incompatible observation contract")
            right_classifier = None
            if args.proprio_guided_right_side_gate is not None:
                right_classifier, right_artifact = load_scene_classifier(
                    args.proprio_guided_right_side_gate, args.device
                )
                if (right_artifact["observation_contract"] != checkpoint["observation_contract"]
                        or right_artifact.get("posture") != "side"):
                    parser.error("right guided-side classifier has an incompatible observation contract")
            guided_side_gate = ProprioGuidedSideGate(
                classifier, num_envs, args.device,
                threshold=args.guided_side_threshold,
                consecutive=args.guided_side_consecutive,
                min_steps=args.guided_side_min_steps,
                max_steps=args.guided_side_max_steps,
                ema_alpha=args.guided_side_ema_alpha,
                early_classifier=early_classifier,
                early_max_step=(args.guided_side_early_max_step if early_classifier is not None else 0),
                right_classifier=right_classifier,
                right_min_steps=(args.guided_right_side_min_steps if right_classifier is not None else 0),
                right_max_steps=(args.guided_right_side_max_steps if right_classifier is not None else 0),
            )
        recovery_controller = None
        if args.proprio_recovery_controller is not None:
            recovery_controller, artifact = load_recovery_controller(
                args.proprio_recovery_controller, num_envs, args.device
            )
            if args.recovery_pulse_blend_steps > recovery_controller.pulse_steps // 2:
                parser.error("recovery pulse blend exceeds half the pulse duration")
            recovery_controller.pulse_blend_steps = args.recovery_pulse_blend_steps
            if artifact["observation_contract"] != checkpoint["observation_contract"]:
                raise ValueError("controller and checkpoint observation contracts differ")
        prone_flat_gate = None
        if args.proprio_prone_flat_gate is not None:
            classifier, gate_artifact = load_scene_classifier(
                args.proprio_prone_flat_gate, args.device,
            )
            if (gate_artifact["observation_contract"] != checkpoint["observation_contract"]
                    or gate_artifact.get("posture") != "prone"):
                raise ValueError("prone-flat gate and actor observation contracts differ")
            prone_flat_gate = ProprioSceneGate(
                classifier, num_envs, args.device,
                threshold=args.prone_flat_threshold,
                consecutive=args.prone_flat_consecutive,
                min_steps=31, max_steps=args.prone_flat_pulse_step,
                target_class=0, posture="prone",
            )
        prone_free_gate = None
        if args.proprio_prone_free_gate is not None:
            classifier, gate_artifact = load_scene_classifier(
                args.proprio_prone_free_gate, args.device,
            )
            if (gate_artifact["observation_contract"] != checkpoint["observation_contract"]
                    or gate_artifact.get("posture") != "prone"):
                raise ValueError("prone-free gate and actor observation contracts differ")
            prone_free_gate = ProprioSceneGate(
                classifier, num_envs, args.device,
                threshold=args.prone_free_threshold,
                consecutive=args.prone_free_consecutive,
                min_steps=31, max_steps=args.prone_free_pulse_step,
                target_class=2, posture="prone",
            )
        flat_gate = None
        if args.proprio_supine_flat_gate is not None:
            flat_gate, gate_artifact = load_supine_flat_gate(
                args.proprio_supine_flat_gate, num_envs, args.device,
                threshold=args.flat_gate_threshold,
                consecutive=args.flat_gate_consecutive,
                min_steps=args.flat_gate_min_steps,
            )
            if gate_artifact["observation_contract"] != checkpoint["observation_contract"]:
                raise ValueError("flat gate and checkpoint observation contracts differ")
        env._a6_staged_escape_v1 = staged_escape_v1
        env._a6_staged_escape_v2 = staged_escape_v2
        env._a6_staged_escape_v3 = staged_escape_v3
        env._a6_staged_escape_v4 = staged_escape_v4
        env._a6_staged_escape_v5 = staged_escape_v5
        env._a6_knee_straight_prior_v1 = knee_straight_prior
        env._a6_arm_extraction_v1 = arm_extraction
        env._a6_prone_flexible_exit_v1 = "_prone_flexible_exit_v1" in raw_variant
        env._a6_scene_phase_reward_v1 = "_scene_phase_reward_v1" in raw_variant
        env._a6_side_lift_anticipation_v1 = "_side_lift_anticipation_v1" in raw_variant
        env._a6_side_directional_progress_v1 = "_side_directional_progress_v1" in raw_variant
        env._a6_free_plate_motion_reward_v1 = (
            "_free_plate_motion_reward_v1" in raw_variant
        )
        env._a6_flat_prone_compact_v1 = "_flat_prone_compact_v1" in raw_variant
        env._a6_free_plate_anchor_v1 = "_free_plate_anchor_v1" in raw_variant
        env._a6_prone_recovery_v1 = "_prone_recovery_v1" in raw_variant
        env._a6_bidirectional_recovery_v1 = "_bidirectional_recovery_v1" in raw_variant
        env._a6_strong_recovery_v1 = "_strong_recovery_v1" in raw_variant
        env._a6_recovery_bridge_v2 = "_recovery_bridge_v2" in raw_variant
        env._a6_postescape_live_obs_v1 = postescape_live_observation_enabled(checkpoint)
        env._a6_edge_offset_m = args.guided_offset_m
        action_term = env.action_manager.get_term("joint_pos")
        action_term.beta.fill_(beta)
        action_term.assist_force.fill_(assist)
        # Viser's first tick calls get_observations()/step() without resetting
        # the env. A6 scene IDs, plate geom IDs, and reset banks are created by
        # reset_scene, so initialize them before handing control to the viewer.
        observations, _ = env.reset()
        if prone_router is not None:
            prone_router.arm_initial_gravity(
                env.scene["robot"].data.projected_gravity_b
            )
        if side_router is not None:
            side_router.arm_initial_gravity(
                env.scene["robot"].data.projected_gravity_b
            )
        if guided_side_gate is not None:
            guided_side_gate.arm_initial_gravity(
                env.scene["robot"].data.projected_gravity_b
            )
        if prone_flat_gate is not None:
            prone_flat_gate.arm_initial_gravity(
                env.scene["robot"].data.projected_gravity_b
            )
        if prone_free_gate is not None:
            prone_free_gate.arm_initial_gravity(
                env.scene["robot"].data.projected_gravity_b
            )
        if flat_gate is not None:
            flat_gate.arm_initial_gravity(
                env.scene["robot"].data.projected_gravity_b
            )
        if four_view:
            pos, ext, plate_pos, plate_ext = plate_footprint_inputs(env)
            core_ids = env._a6_prone_core_collision_indices
            core_covered = footprint(
                pos[:, core_ids], ext[:, core_ids], plate_pos, plate_ext
            )[0]
            all_covered = footprint(pos, ext, plate_pos, plate_ext)[0]
            root_xy = env.scene["robot"].data.root_link_pos_w[:, :2]
            plate_offset = (plate_pos[:, :2] - root_xy).norm(dim=-1)
            for idx in torch.where(env._a6_scene == 1)[0].tolist():
                direction = DIRECTIONS[int(env._a6_direction[idx])]
                print(
                    f"initial guided/{direction} plate: "
                    f"root_offset={plate_offset[idx].item():.3f}m "
                    f"core_covered={int(core_covered[idx])}/{len(core_ids)} "
                    f"all_covered={int(all_covered[idx])}/{pos.shape[1]}",
                    flush=True,
                )
        if (recovery_controller is not None or flat_gate is not None
                or prone_router is not None or side_router is not None
                or guided_side_gate is not None or prone_flat_gate is not None
                or prone_free_gate is not None):
            physical_step = env.step
            physical_reset = env.reset

            def step_with_proprio_recovery(actions):
                if recovery_controller is not None:
                    actions, _ = recovery_controller.apply(actions)
                obs, reward, terminated, truncated, extras = physical_step(actions)
                done = terminated | truncated
                if done.any():
                    ids = done.nonzero(as_tuple=False).squeeze(-1)
                    if recovery_controller is not None:
                        recovery_controller.reset(ids)
                    if flat_gate is not None:
                        flat_gate.reset(ids)
                        flat_gate.arm_initial_gravity(
                            env.scene["robot"].data.projected_gravity_b, ids
                        )
                    if prone_router is not None:
                        prone_router.reset(ids)
                        prone_router.arm_initial_gravity(
                            env.scene["robot"].data.projected_gravity_b, ids
                        )
                    if side_router is not None:
                        side_router.reset(ids)
                        side_router.arm_initial_gravity(
                            env.scene["robot"].data.projected_gravity_b, ids
                        )
                    if guided_side_gate is not None:
                        guided_side_gate.reset(ids)
                        guided_side_gate.arm_initial_gravity(
                            env.scene["robot"].data.projected_gravity_b, ids
                        )
                    if prone_flat_gate is not None:
                        prone_flat_gate.reset(ids)
                        prone_flat_gate.arm_initial_gravity(
                            env.scene["robot"].data.projected_gravity_b, ids
                        )
                    if prone_free_gate is not None:
                        prone_free_gate.reset(ids)
                        prone_free_gate.arm_initial_gravity(
                            env.scene["robot"].data.projected_gravity_b, ids
                        )
                if recovery_controller is not None:
                    recovery_controller.observe(
                        obs["actor"], action_term.real_steps, eligible=~done
                    )
                return obs, reward, terminated, truncated, extras

            def reset_with_proprio_recovery(*reset_args, **reset_kwargs):
                obs, extras = physical_reset(*reset_args, **reset_kwargs)
                ids = reset_kwargs.get("env_ids")
                if ids is None and reset_args:
                    ids = reset_args[0]
                if recovery_controller is not None:
                    recovery_controller.reset(ids)
                if flat_gate is not None:
                    flat_gate.reset(ids)
                    flat_gate.arm_initial_gravity(
                        env.scene["robot"].data.projected_gravity_b, ids
                    )
                if prone_router is not None:
                    prone_router.reset(ids)
                    prone_router.arm_initial_gravity(
                        env.scene["robot"].data.projected_gravity_b, ids
                    )
                if side_router is not None:
                    side_router.reset(ids)
                    side_router.arm_initial_gravity(
                        env.scene["robot"].data.projected_gravity_b, ids
                    )
                if guided_side_gate is not None:
                    guided_side_gate.reset(ids)
                    guided_side_gate.arm_initial_gravity(
                        env.scene["robot"].data.projected_gravity_b, ids
                    )
                if prone_flat_gate is not None:
                    prone_flat_gate.reset(ids)
                    prone_flat_gate.arm_initial_gravity(
                        env.scene["robot"].data.projected_gravity_b, ids
                    )
                if prone_free_gate is not None:
                    prone_free_gate.reset(ids)
                    prone_free_gate.arm_initial_gravity(
                        env.scene["robot"].data.projected_gravity_b, ids
                    )
                return obs, extras

            env.step = step_with_proprio_recovery
            env.reset = reset_with_proprio_recovery
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
        action_term.beta.fill_(beta)
        action_term.assist_force.fill_(assist)
        layout_name = (
            "guided/left, guided/right, free/right, flat/right"
            if args.side_four_view
            else "free/supine, guided/prone, guided/supine, flat/supine"
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
            if four_view:
                actual = torch.stack(
                    (env._a6_scene, env._a6_direction, env._a6_stratum), dim=1
                )
                expected = torch.tensor(
                    SIDE_FOUR_VIEW_LAYOUT if args.side_four_view else FOUR_VIEW_LAYOUT,
                    device=env.device,
                )
                if not torch.equal(actual, expected):
                    raise RuntimeError("four-view A6 worlds do not match layout")
            elif int(env._a6_scene[env_idx]) != SCENE_IDS[args.scene] or int(
                env._a6_direction[env_idx]
            ) != DIRECTIONS.index(args.direction):
                raise RuntimeError("selected A6 world does not match requested scene")
            playback_policy = DeterministicHostPolicy(
                policy, env, flat_expert, flat_gate, prone_expert, prone_router,
                side_expert, side_router, guided_side_gate,
                args.mirror_left_side_specialist,
                args.mirror_right_initial_side_main,
                args.side_safe_through_step, args.side_safe_scale,
                args.side_safe_probe_through_step,
                args.left_side_specialist_probe_through_step,
                args.side_handoff_steps,
                args.side_lateral_gain,
                prone_flat_gate, recovery_controller,
                args.prone_flat_pulse_step,
                prone_free_gate, args.prone_free_pulse_step,
            )
            resets = 0
            for _ in range(args.validate_steps):
                actions = playback_policy(observations)
                if actions.shape != (num_envs, 29) or not torch.isfinite(actions).all():
                    raise RuntimeError("checkpoint actor returned invalid 29-DoF actions")
                observations, _, terminated, truncated, _ = env.step(actions)
                done = terminated | truncated
                resets += int(done.sum())
                if done.any() and guided_side_gate is not None:
                    ids = done.nonzero(as_tuple=False).squeeze(-1)
                    if not torch.equal(
                        guided_side_gate.initial_posture[ids],
                        side_router.initial_side[ids],
                    ):
                        raise RuntimeError("guided-side gate was not rearmed after auto-reset")
            print(
                f"A6 playback validation passed: steps={args.validate_steps} "
                f"auto_resets={resets}", flush=True,
            )
            return
        print("Open the Viser URL shown below (usually http://localhost:8080).")
        ViserPlayViewer(
            env, DeterministicHostPolicy(
                policy, env, flat_expert, flat_gate, prone_expert, prone_router,
                side_expert, side_router, guided_side_gate,
                args.mirror_left_side_specialist,
                args.mirror_right_initial_side_main,
                args.side_safe_through_step, args.side_safe_scale,
                args.side_safe_probe_through_step,
                args.left_side_specialist_probe_through_step,
                args.side_handoff_steps,
                args.side_lateral_gain,
                prone_flat_gate, recovery_controller,
                args.prone_flat_pulse_step,
                prone_free_gate, args.prone_free_pulse_step,
            )
        ).run()
    finally:
        env.close()


if __name__ == "__main__":
    main()
