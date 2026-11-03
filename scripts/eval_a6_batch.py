"""Batch, no-assist A6 checkpoint evaluation in the four playback scenes."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
from mjlab.envs import ManagerBasedRlEnv

from g1recovery_host.a6_distill import (
    SIDE_SCALED_IDS,
    apply_side_lateral_gain,
    effective_guided_teacher_action, effective_guided_side_teacher_action,
    legacy_teacher_observation,
    validate_legacy_teacher,
)
from g1recovery_host.a6_geometry import (
    PRONE_EXIT_CONFIRM_STEPS,
    footprint,
    plate_footprint_inputs,
    prone_exit_geometry,
    prone_flexible_clearance,
)
from g1recovery_host.a6_history_replay import restore_postescape_actor_history
from g1recovery_host.a6_reward import GUIDED_MAX_LIFT_M, PLATE_FORCE_INVALID_N
from g1recovery_host.a6_task import host_a6_env_cfg
from g1recovery_host.model import HOST29_JOINT_NAMES, HOST29_TARGET_JOINT_POS
from g1recovery_host.phase_switch import (
    postescape_live_observation_enabled,
    reset_actor_history_from_proprio,
    reset_guided_history_after_escape,
    reset_observation_history_from_proprio,
    reset_prone_history_after_escape,
)
from g1recovery_host.ppo import HostPolicy
from g1recovery_host.proprio_phase import load_phase_detector
from g1recovery_host.proprio_scene import (
    ProprioEarlyFlatGate, ProprioGuidedProneGate, ProprioGuidedSideGate,
    ProprioSceneGate,
    load_scene_classifier,
)
from g1recovery_host.proprio_symmetry import (
    mirror_host29_action, mirror_host29_actor_obs,
)
from g1recovery_host.proprio_recovery import (
    InitialProneRouter,
    InitialSideRouter,
    load_prone_specialist,
    load_recovery_controller,
    load_supine_flat_gate,
    validate_controller_provenance,
)

LABELS = ("guided_prone", "guided_supine", "free_supine", "flat_supine")
LAYOUT_ROWS = ((1, 1, 0), (1, 0, 0), (2, 0, 0), (0, 0, 0))
ALL_LAYOUT_ROWS = tuple(
    (scene, direction, 0) for scene in (1, 2, 0) for direction in range(4)
)
ALL_LABELS = tuple(
    f"{scene}_{direction}"
    for scene in ("guided", "free", "flat")
    for direction in ("supine", "prone", "left_side_down", "right_side_down")
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--supine-teacher-checkpoint", type=Path)
    parser.add_argument("--postescape-teacher-checkpoint", type=Path)
    parser.add_argument(
        "--proprio-postescape-flat-teacher", type=Path,
        help="diagnostic only: switch a proprio actor to the 564D flat expert after guided escape",
    )
    parser.add_argument(
        "--diagnostic-flat-expert-on-flat", action="store_true",
        help="diagnostic only: use the 564D flat expert on flat worlds from reset",
    )
    parser.add_argument(
        "--diagnostic-flat-expert-after-prone-pulse", action="store_true",
        help="diagnostic oracle: use the flat expert on true flat-prone worlds after the proprio recovery pulse",
    )
    parser.add_argument(
        "--diagnostic-flat-expert-on-free-after-escape", action="store_true",
        help="diagnostic only: use the flat expert after simulator-confirmed free-board escape",
    )
    parser.add_argument(
        "--proprio-postescape-a6-teacher", type=Path,
        help="diagnostic only: switch a proprio actor to the legacy 612D A6 expert after guided escape",
    )
    parser.add_argument(
        "--proprio-preescape-a6-teacher", type=Path,
        help="diagnostic only: let the legacy 612D teacher drive prone/supine guided worlds until escape",
    )
    parser.add_argument(
        "--proprio-diagnostic-reset-prone-history", action="store_true",
        help="diagnostic upper bound: use simulator clearance to reset the student history at prone escape",
    )
    parser.add_argument("--proprio-phase-detector", type=Path)
    parser.add_argument("--proprio-prone-specialist", type=Path,
                        help="738D initial-prone actor selected from reset IMU only")
    parser.add_argument("--proprio-side-specialist", type=Path,
                        help="738D initial-side actor selected from reset IMU only")
    parser.add_argument("--proprio-left-side-specialist", type=Path,
                        help="738D actor for initial positive body-Y gravity, overriding the side specialist")
    parser.add_argument("--mirror-left-side-specialist", action="store_true",
                        help="reflect left-side proprioception and actions through the right-side specialist")
    parser.add_argument("--mirror-right-initial-side-main", action="store_true",
                        help="reflect right initial-side proprioception through the main actor's left-side behavior")
    parser.add_argument("--side-soft-gate-v1", action="store_true",
                        help="blend baseline and side actions by the proprio guided-board probability")
    parser.add_argument("--side-soft-gate-low", type=float, default=0.25)
    parser.add_argument("--side-soft-gate-high", type=float, default=0.85)
    parser.add_argument("--proprio-guided-side-gate", type=Path,
                        help="738D scene classifier that limits the side specialist to likely fixed boards")
    parser.add_argument("--side-handoff-steps", type=int, default=0,
                        help="blend main and side-specialist targets for N steps after the gate latches")
    parser.add_argument("--side-safe-only-when-guided-gate", action="store_true",
                        help="apply initial-side joint scaling only after the proprio guided-side gate latches")
    parser.add_argument("--side-safe-probe-through-step", type=int, default=0,
                        help="scale all initial-side actions briefly before the proprio guided-side gate settles")
    parser.add_argument("--left-side-specialist-probe-through-step", type=int, default=0,
                        help="briefly run the mirrored side expert before classifying an initial left side")
    parser.add_argument("--initial-side-joint-safe-scale", type=float, default=0.25,
                        help="scale hip-pitch and nonlateral arm actions during initial-side safety window")
    parser.add_argument("--guided-side-threshold", type=float, default=0.95)
    parser.add_argument("--guided-side-consecutive", type=int, default=10)
    parser.add_argument("--guided-side-ema-alpha", type=float, default=1.0)
    parser.add_argument("--guided-side-min-steps", type=int, default=35)
    parser.add_argument("--guided-side-max-steps", type=int, default=120)
    parser.add_argument("--proprio-guided-side-early-gate", type=Path,
                        help="second proprio classifier used only during early side contact")
    parser.add_argument("--guided-side-early-max-step", type=int, default=45)
    parser.add_argument("--proprio-guided-right-side-gate", type=Path,
                        help="optional proprio classifier for initial right side")
    parser.add_argument("--guided-right-side-min-steps", type=int, default=35)
    parser.add_argument("--guided-right-side-max-steps", type=int, default=120)
    parser.add_argument("--proprio-guided-prone-specialist", type=Path,
                        help="738D actor selected only by a learned guided-prone cue")
    parser.add_argument("--proprio-guided-prone-gate", type=Path,
                        help="738D scene classifier for initial-prone high-confidence routing")
    parser.add_argument("--proprio-prone-flat-gate", type=Path,
                        help="738D initial-prone scene classifier used only to arm an early flat recovery pulse")
    parser.add_argument("--prone-flat-pulse-step", type=int, default=50)
    parser.add_argument("--prone-flat-threshold", type=float, default=0.95)
    parser.add_argument("--prone-flat-consecutive", type=int, default=2)
    parser.add_argument("--proprio-prone-free-gate", type=Path,
                        help="738D initial-prone scene classifier used only to arm an early free-board recovery pulse")
    parser.add_argument("--prone-free-pulse-step", type=int, default=100)
    parser.add_argument("--prone-free-threshold", type=float, default=0.99)
    parser.add_argument("--prone-free-consecutive", type=int, default=3)
    parser.add_argument("--guided-prone-threshold", type=float, default=0.99)
    parser.add_argument("--guided-prone-consecutive", type=int, default=10)
    parser.add_argument("--guided-prone-min-steps", type=int, default=40)
    parser.add_argument("--guided-prone-max-steps", type=int, default=120)
    parser.add_argument("--proprio-supine-flat-gate", type=Path,
                        help="proprio-only supine gate for the frozen flat expert")
    parser.add_argument("--proprio-early-flat-gate", type=Path,
                        help="diagnostic 738D high-confidence flat cue during steps 40-120")
    parser.add_argument("--early-flat-threshold", type=float, default=0.99)
    parser.add_argument("--early-flat-consecutive", type=int, default=10)
    parser.add_argument("--flat-gate-threshold", type=float, default=0.8)
    parser.add_argument("--flat-gate-consecutive", type=int, default=3)
    parser.add_argument("--flat-gate-min-steps", type=int, default=120)
    parser.add_argument("--supine-neutral-through-step", type=int, default=0,
                        help="diagnostic proprio-only zero-increment wait before supine handoff")
    parser.add_argument("--side-neutral-through-step", type=int, default=0,
                        help="diagnostic initial-IMU side posture wait after motor startup")
    parser.add_argument("--side-action-scale", type=float, default=1.0,
                        help="diagnostic scale of all side-posture actor increments")
    parser.add_argument("--side-lateral-gain", type=float, default=1.0,
                        help="multiply eight lateral hip/shoulder actions after a proprio guided-side gate")
    parser.add_argument("--diagnostic-guided-nonside-lateral-gain", type=float, default=1.0,
                        help="oracle-only: boost lateral joints on true guided prone/supine worlds before escape")
    parser.add_argument("--diagnostic-guided-nonside-safe-scale", type=float, default=1.0,
                        help="oracle-only: scale nonlateral joints on true guided prone/supine worlds before escape")
    parser.add_argument("--initial-side-joint-safe-through-step", type=int, default=0,
                        help="diagnostic IMU-only side joint scaling through this control step")
    parser.add_argument("--diagnostic-zero-plate-obs", action="store_true",
                        help="diagnostic legacy teacher with all six 8D plate frames zeroed")
    parser.add_argument("--diagnostic-guided-side-assist-n", type=float, default=0.0,
                        help="simulator-only horizontal torso force on guided-side worlds; never a policy result")
    parser.add_argument("--supine-probe-v1", action="store_true",
                        help="small alternating shoulder-roll probe during active steps 31-60")
    parser.add_argument("--proprio-recovery-controller", type=Path,
                        help="deployable proprio-only detector, initial-posture gate, and fixed pulse")
    parser.add_argument("--recovery-pulse-blend-steps", type=int, default=0,
                        help="blend both ends of the recovery pulse over N control steps")
    parser.add_argument("--diagnostic-flat-prone-early-pulse-step", type=int, default=0,
                        help="diagnostic oracle: force the existing recovery pulse at this step only on true flat-prone worlds")
    parser.add_argument("--diagnostic-free-prone-early-pulse-step", type=int, default=0,
                        help="diagnostic oracle: force the existing recovery pulse only on true free-prone worlds")
    parser.add_argument("--diagnostic-controller-threshold", type=float,
                        help="diagnostic only: override the packaged prone detector threshold")
    parser.add_argument("--proprio-phase-threshold", type=float, default=0.9)
    parser.add_argument("--proprio-phase-consecutive", type=int, default=2)
    parser.add_argument("--proprio-phase-full-observation-reset", action="store_true")
    parser.add_argument("--proprio-phase-controller-reset", action="store_true")
    parser.add_argument("--proprio-phase-min-steps", type=int, default=60)
    parser.add_argument("--proprio-phase-pulse-steps", type=int, default=0,
                        help="after a proprio-only trigger, apply the frozen policy's zero-observation action")
    parser.add_argument("--proprio-phase-initial-prone-gate", action="store_true",
                        help="arm the proprio phase cue only if the first active IMU frame is prone")
    parser.add_argument("--proprio-oracle-pulse-steps", type=int, default=0,
                        help="diagnostic: use the frozen policy's zero-observation action after true guided-prone escape")
    parser.add_argument("--proprio-oracle-actor-reset-only", action="store_true",
                        help="diagnostic upper bound: reset actor history from simulator escape without modifying other state")
    parser.add_argument("--proprio-oracle-observation-reset-only", action="store_true",
                        help="diagnostic upper bound: reset all observation history, but no controller state")
    parser.add_argument("--proprio-oracle-controller-reset-only", action="store_true",
                        help="diagnostic upper bound: reset controller counter, but no observation history")
    parser.add_argument(
        "--proprio-a6-teacher-reset-history", action="store_true",
        help="diagnostic only: clear the privileged teacher's critic history at guided-prone escape",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--per-scene", type=int, default=8)
    parser.add_argument("--all-postescape", action="store_true")
    parser.add_argument(
        "--all-directions", action="store_true",
        help="evaluate every scene/direction pair, including both side-lying resets",
    )
    parser.add_argument("--postescape-bank", type=Path)
    parser.add_argument("--restore-postescape-bank-history", action="store_true",
                        help="diagnostic: restore the recorded six-frame actor history at replay reset")
    parser.add_argument("--postescape-live-obs-override", action="store_true")
    parser.add_argument("--prone-flexible-exit-override", action="store_true",
                        help="diagnostic: evaluate a legacy checkpoint under the new guided-prone exit rule")
    parser.add_argument("--shadow-prone-flexible-exit", action="store_true",
                        help="compare old and new prone exit labels along the same physical trajectories")
    parser.add_argument("--shadow-side-flexible-exit", action="store_true",
                        help="diagnostic only: label guided-side core clearance with shallow trailing tips")
    parser.add_argument("--postescape-motor-delay-override", action="store_true")
    parser.add_argument("--reset-policy-history-on-escape", action="store_true")
    parser.add_argument("--reset-supine-history-on-escape", action="store_true")
    parser.add_argument("--guided-offset-m", type=float, default=0.25)
    parser.add_argument("--sample-action", action="store_true")
    parser.add_argument("--mean-after-escape", action="store_true")
    parser.add_argument("--knee-diagnostics", action="store_true")
    parser.add_argument("--prone-full-arms-override", action="store_true")
    parser.add_argument("--arm-actuation-scale", type=float, default=None)
    parser.add_argument("--hip-actuation-scale", type=float, default=None)
    parser.add_argument(
        "--no-privileged-actions", action="store_true",
        help="legacy-teacher ablation: run its actor without scene-dependent action scaling or lift guard",
    )
    parser.add_argument("--disable-lift-guard", action="store_true")
    parser.add_argument("--disable-safe-scaling", action="store_true")
    parser.add_argument(
        "--bake-safe-scaling-into-policy",
        action="store_true",
        help="diagnostic: bake the legacy guided hip-pitch scale into raw actions",
    )
    parser.add_argument(
        "--bake-hip-scaling-all-scenes",
        action="store_true",
        help="diagnostic: apply the same 0.25 hip-pitch action scale in every scene",
    )
    args = parser.parse_args()
    if args.prone_flexible_exit_override and args.shadow_prone_flexible_exit:
        parser.error("shadow comparison needs the legacy rule to remain active")
    if args.shadow_prone_flexible_exit and args.shadow_side_flexible_exit:
        parser.error("choose one shadow exit posture per evaluation")
    if (args.bake_safe_scaling_into_policy or args.bake_hip_scaling_all_scenes) and not args.no_privileged_actions:
        parser.error("baked hip scaling requires --no-privileged-actions")
    if args.bake_safe_scaling_into_policy and args.bake_hip_scaling_all_scenes:
        parser.error("choose one baked hip-scaling diagnostic")
    if args.per_scene < 8 or args.per_scene % 8:
        parser.error("--per-scene must be a positive multiple of 8")
    if args.all_postescape and args.per_scene < 32:
        parser.error("--all-postescape needs at least 32 worlds")
    if args.all_postescape and args.all_directions:
        parser.error("--all-postescape and --all-directions cannot be combined")
    if args.restore_postescape_bank_history and not (
        args.all_postescape and args.postescape_bank is not None
    ):
        parser.error("recorded postescape history requires all-postescape and a bank")
    if args.postescape_teacher_checkpoint is not None and not args.all_postescape:
        parser.error("--postescape-teacher-checkpoint requires --all-postescape")
    if args.proprio_postescape_flat_teacher is not None and args.all_postescape:
        parser.error("--proprio-postescape-flat-teacher requires full escape episodes")
    if (args.diagnostic_flat_expert_on_flat
        or args.diagnostic_flat_expert_after_prone_pulse
        or args.diagnostic_flat_expert_on_free_after_escape
    ) and args.proprio_postescape_flat_teacher is None:
        parser.error("flat/free expert diagnostics require --proprio-postescape-flat-teacher")
    if args.diagnostic_flat_expert_after_prone_pulse and args.proprio_recovery_controller is None:
        parser.error("post-pulse flat expert diagnostic requires the proprio recovery controller")
    if args.proprio_supine_flat_gate is not None and (
        args.proprio_postescape_flat_teacher is None
        or args.diagnostic_flat_expert_on_flat
        or args.diagnostic_flat_expert_on_free_after_escape
    ):
        parser.error("supine flat gate requires a flat expert and no oracle scene switches")
    if not (0 < args.flat_gate_threshold < 1 and args.flat_gate_consecutive > 0
            and args.flat_gate_min_steps > 30):
        parser.error("invalid supine flat gate thresholds")
    if args.supine_neutral_through_step and (
        args.proprio_supine_flat_gate is None
        or not 30 < args.supine_neutral_through_step <= args.flat_gate_min_steps
    ):
        parser.error("supine neutral wait requires the gate and 30 < wait <= gate minimum")
    if args.supine_probe_v1 and (
        args.proprio_supine_flat_gate is None or args.supine_neutral_through_step
    ):
        parser.error("supine probe requires the flat gate and no neutral-wait override")
    if args.proprio_postescape_flat_teacher is not None and args.proprio_postescape_a6_teacher is not None:
        parser.error("choose one postescape teacher diagnostic")
    if args.proprio_preescape_a6_teacher is not None and (
        args.proprio_postescape_a6_teacher is not None or args.all_postescape
    ):
        parser.error("preescape teacher requires full episodes and cannot be combined with a postescape teacher")
    if args.proprio_diagnostic_reset_prone_history and args.proprio_preescape_a6_teacher is None:
        parser.error("diagnostic student history reset requires --proprio-preescape-a6-teacher")
    if args.proprio_phase_detector is not None and any((
        args.proprio_oracle_actor_reset_only,
        args.proprio_oracle_observation_reset_only,
        args.proprio_oracle_controller_reset_only,
    )):
        parser.error("choose a learned or oracle trigger")
    if args.proprio_recovery_controller is not None and (
        args.proprio_phase_detector is not None
        or args.proprio_preescape_a6_teacher is not None
        or args.proprio_oracle_pulse_steps
    ):
        parser.error("packaged recovery controller must run as an independent student policy")
    if args.diagnostic_controller_threshold is not None and not (
        args.proprio_recovery_controller is not None
        and 0 < args.diagnostic_controller_threshold < 1
    ):
        parser.error("controller threshold diagnostic requires a controller and a threshold in (0,1)")
    if sum((args.proprio_oracle_actor_reset_only,
            args.proprio_oracle_observation_reset_only,
            args.proprio_oracle_controller_reset_only)) > 1:
        parser.error("choose only one oracle trigger")
    if not 0 < args.proprio_phase_threshold < 1 or args.proprio_phase_consecutive < 1:
        parser.error("phase threshold must be (0,1) and consecutive count positive")
    if args.proprio_phase_min_steps < 31:
        parser.error("phase trigger must wait until after the 30-step motor startup")
    if args.proprio_phase_controller_reset and args.proprio_phase_detector is None:
        parser.error("learned controller reset requires a proprio phase detector")
    if args.proprio_phase_pulse_steps < 0 or (
        args.proprio_phase_pulse_steps and args.proprio_phase_detector is None
    ):
        parser.error("learned pulse requires a detector and a nonnegative length")
    if args.proprio_phase_pulse_steps and args.proprio_phase_controller_reset:
        parser.error("choose either a learned pulse or controller reset")
    if args.proprio_phase_initial_prone_gate and args.proprio_phase_detector is None:
        parser.error("initial posture gate requires a proprio phase detector")
    if args.proprio_oracle_pulse_steps < 0 or (args.proprio_oracle_pulse_steps and args.proprio_phase_detector):
        parser.error("oracle pulse requires a nonnegative length and no learned phase trigger")
    if args.proprio_a6_teacher_reset_history and args.proprio_postescape_a6_teacher is None:
        parser.error("teacher-only history reset requires --proprio-postescape-a6-teacher")
    if args.postescape_bank is not None and not args.all_postescape:
        parser.error("--postescape-bank requires --all-postescape")
    if args.postescape_motor_delay_override and not args.all_postescape:
        parser.error("--postescape-motor-delay-override requires --all-postescape")
    if args.postescape_motor_delay_override and args.postescape_live_obs_override:
        parser.error("motor delay and live observation overrides cannot be combined")
    if args.reset_policy_history_on_escape and args.all_postescape:
        parser.error("history-on-escape reset requires a full escape scene")
    if args.reset_supine_history_on_escape and args.all_postescape:
        parser.error("supine history reset requires a full escape scene")
    if args.mean_after_escape and not args.sample_action:
        parser.error("--mean-after-escape requires --sample-action")
    if args.side_neutral_through_step and args.side_neutral_through_step <= 30:
        parser.error("side neutral wait must end after the 30-step motor startup")
    if not 0 < args.side_action_scale <= 1:
        parser.error("side action scale must lie in (0, 1]")
    if not 1.0 <= args.side_lateral_gain <= 1.5:
        parser.error("side lateral gain must lie in [1, 1.5]")
    if not 1.0 <= args.diagnostic_guided_nonside_lateral_gain <= 1.5:
        parser.error("diagnostic guided nonside lateral gain must lie in [1, 1.5]")
    if not 0 < args.diagnostic_guided_nonside_safe_scale <= 1.0:
        parser.error("diagnostic guided nonside safe scale must lie in (0, 1]")
    if args.side_lateral_gain > 1.0 and (
        args.proprio_guided_side_gate is None
        or not args.initial_side_joint_safe_through_step
    ):
        parser.error("side lateral gain requires a proprio guided-side gate and action horizon")
    if args.initial_side_joint_safe_through_step and not (
        30 < args.initial_side_joint_safe_through_step <= args.steps
    ):
        parser.error("initial side joint scaling must end after startup and within the episode")
    if not 0 <= args.diagnostic_guided_side_assist_n <= 300:
        parser.error("diagnostic guided-side assist must be in [0, 300] N")
    if args.proprio_early_flat_gate is not None and args.proprio_postescape_flat_teacher is None:
        parser.error("early flat gate requires the frozen flat expert")
    if (args.proprio_guided_prone_gate is None) != (args.proprio_guided_prone_specialist is None):
        parser.error("guided-prone scene gate and specialist must be supplied together")
    if args.proprio_guided_prone_gate is not None and args.proprio_prone_specialist is None:
        parser.error("guided-prone scene gate needs a baseline prone specialist")
    if args.proprio_guided_side_gate is not None and args.proprio_side_specialist is None:
        parser.error("guided-side gate requires a side specialist")
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
    if args.side_safe_only_when_guided_gate and (
        args.proprio_guided_side_gate is None or not args.initial_side_joint_safe_through_step
    ):
        parser.error("gate-limited side scaling requires a guided-side gate and scaling horizon")
    if args.side_handoff_steps < 0 or (
        args.side_handoff_steps and (
            args.proprio_guided_side_gate is None or args.side_soft_gate_v1
        )
    ):
        parser.error("side handoff blending requires a hard guided-side gate and nonnegative steps")
    if args.recovery_pulse_blend_steps < 0 or (
        args.recovery_pulse_blend_steps and args.proprio_recovery_controller is None
    ):
        parser.error("recovery pulse blending requires its controller and nonnegative steps")
    if args.diagnostic_flat_prone_early_pulse_step and not (
        args.proprio_recovery_controller is not None
        and 31 <= args.diagnostic_flat_prone_early_pulse_step <= 120
    ):
        parser.error("flat-prone early pulse requires its controller and a step in [31, 120]")
    if args.diagnostic_free_prone_early_pulse_step and not (
        args.proprio_recovery_controller is not None
        and 31 <= args.diagnostic_free_prone_early_pulse_step <= 300
    ):
        parser.error("free-prone early pulse requires its controller and a step in [31, 300]")
    if args.proprio_prone_flat_gate is not None and not (
        args.proprio_recovery_controller is not None
        and args.diagnostic_flat_prone_early_pulse_step == 0
        and 31 <= args.prone_flat_pulse_step <= 120
        and 0 < args.prone_flat_threshold < 1
        and 1 <= args.prone_flat_consecutive <= 10
    ):
        parser.error("prone-flat gate requires its controller, a valid window, and no oracle pulse")
    if args.proprio_prone_free_gate is not None and not (
        args.proprio_recovery_controller is not None
        and args.diagnostic_free_prone_early_pulse_step == 0
        and 31 <= args.prone_free_pulse_step <= 300
        and 0 < args.prone_free_threshold < 1
        and 1 <= args.prone_free_consecutive <= 10
    ):
        parser.error("prone-free gate requires its controller, a valid window, and no oracle pulse")
    if args.side_safe_probe_through_step and not (
        args.side_safe_only_when_guided_gate
        and 30 < args.side_safe_probe_through_step <= args.initial_side_joint_safe_through_step
    ):
        parser.error("side safety probe requires gate-limited scaling and a valid probe horizon")
    if args.left_side_specialist_probe_through_step and not (
        args.proprio_guided_side_gate is not None
        and args.mirror_left_side_specialist
        and 30 < args.left_side_specialist_probe_through_step <= args.initial_side_joint_safe_through_step
    ):
        parser.error("left side expert probe requires the gate, mirrored expert, and a safety window")
    if not 0 < args.initial_side_joint_safe_scale <= 1:
        parser.error("initial side joint safe scale must be in (0, 1]")
    if args.proprio_left_side_specialist is not None and args.proprio_side_specialist is None:
        parser.error("left-side specialist requires a baseline side specialist")
    if args.mirror_left_side_specialist and (
        args.proprio_side_specialist is None or args.proprio_left_side_specialist is not None
    ):
        parser.error("mirrored left side requires only the common side specialist")
    if args.side_soft_gate_v1 and (
        args.proprio_guided_side_gate is None
        or args.proprio_side_specialist is None
        or args.proprio_left_side_specialist is not None
        or args.side_safe_only_when_guided_gate
        or not 0 <= args.side_soft_gate_low < args.side_soft_gate_high <= 1
    ):
        parser.error("side soft gate needs a classifier, side actor, and valid probability limits")
    if not 0 < args.guided_side_ema_alpha <= 1:
        parser.error("guided-side EMA alpha must be in (0, 1]")
    if not 0 <= args.guided_offset_m <= 0.5:
        parser.error("--guided-offset-m must be in [0, 0.5]")
    if any(
        value is not None and not 0 < value <= 1
        for value in (args.arm_actuation_scale, args.hip_actuation_scale)
    ):
        parser.error("actuation scales must be in (0, 1]")
    torch.manual_seed(args.seed)
    state = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    validate_controller_provenance(state, args.proprio_recovery_controller)
    if state.get("task") != "a6" or state.get("model_source") != "unitree_g1_29":
        raise ValueError("A6 evaluation requires a 29-DoF A6 checkpoint")
    variant = state["reward_variant"]
    torque_v2 = state.get("observation_contract") == "proprio_torque_actor_privileged_critic_v2"
    if (args.proprio_phase_detector is not None or args.proprio_recovery_controller is not None) and not torque_v2:
        parser.error("proprio phase detector requires 738D actor observations")
    proprio_actor = torque_v2 or state.get("observation_contract") == "proprio_actor_privileged_critic_v1"
    if args.mirror_right_initial_side_main and (not torque_v2 or args.sample_action):
        parser.error("mirrored right main action requires deterministic 738D proprio evaluation")
    if args.diagnostic_zero_plate_obs and proprio_actor:
        parser.error("zero plate observation applies only to the legacy 612D teacher")
    if proprio_actor and (
        state.get("actor_obs_dim") != (738 if torque_v2 else 564)
        or state.get("critic_obs_dim") != (786 if torque_v2 else 612)
        or state.get("action_contract") != "scene_independent_v1"
    ):
        raise ValueError("proprioceptive checkpoint has inconsistent observation or action metadata")
    if proprio_actor and any((
        args.supine_teacher_checkpoint is not None,
        args.postescape_teacher_checkpoint is not None,
        args.reset_policy_history_on_escape,
        args.reset_supine_history_on_escape,
        args.mean_after_escape,
        args.prone_full_arms_override,
        args.arm_actuation_scale is not None,
        args.hip_actuation_scale is not None,
        args.no_privileged_actions,
        args.disable_lift_guard,
        args.disable_safe_scaling,
        args.bake_safe_scaling_into_policy,
        args.bake_hip_scaling_all_scenes,
    )):
        raise ValueError("proprioceptive evaluation must use one unmodified, scene-independent actor")
    if proprio_actor and ("_safe_actions_v1" in variant or "_lift_guard_v1" in variant):
        raise ValueError("proprioceptive checkpoint cannot enable legacy scene-dependent actions")
    if any(path is not None for path in (
        args.proprio_postescape_flat_teacher,
        args.proprio_postescape_a6_teacher,
        args.proprio_preescape_a6_teacher,
    )) and not proprio_actor:
        parser.error("proprio postescape teacher diagnostics require a proprio actor checkpoint")
    phase_switch = (
        "_phase_history_reset_v1" in variant
        or args.reset_policy_history_on_escape
    )
    staged_v5 = "_staged_escape_v5" in variant
    edge_curriculum = "_v8_edge_curriculum_v1" in variant
    if not staged_v5 or not edge_curriculum:
        raise ValueError("batch evaluation requires staged v5 edge curriculum")
    policy = HostPolicy(
        738 if torque_v2 else 564 if proprio_actor else 612,
        29,
        critic_obs_dim=(786 if torque_v2 else 612) if proprio_actor else None,
        log_std_bounds=(-2.3, math.log(state.get("action_std_max", 0.7))),
        upper_residual=True,
        waist_residual_scale=5.0,
    ).to(args.device)
    policy.load_state_dict(state["policy"], strict=True)
    policy.eval()
    prone_policy = None
    if args.proprio_prone_specialist is not None:
        prone_policy, _ = load_prone_specialist(
            args.proprio_prone_specialist, state, args.device,
        )
    side_policy = None
    if args.proprio_side_specialist is not None:
        side_policy, _ = load_prone_specialist(
            args.proprio_side_specialist, state, args.device,
        )
    left_side_policy = None
    if args.proprio_left_side_specialist is not None:
        left_side_policy, _ = load_prone_specialist(
            args.proprio_left_side_specialist, state, args.device,
        )
    guided_prone_policy = None
    if args.proprio_guided_prone_specialist is not None:
        guided_prone_policy, _ = load_prone_specialist(
            args.proprio_guided_prone_specialist, state, args.device,
        )
    phase_detector = None
    if args.proprio_phase_detector is not None:
        phase_detector, detector_state = load_phase_detector(
            args.proprio_phase_detector, args.device
        )
        if detector_state["observation_contract"] != state["observation_contract"]:
            raise ValueError("phase detector and actor observation contracts differ")
    postescape_teacher = None
    proprio_postescape_teacher = None
    proprio_postescape_a6_teacher = None
    proprio_preescape_a6_teacher = None
    a6_teacher_path = (
        args.proprio_postescape_a6_teacher or args.proprio_preescape_a6_teacher
    )
    if a6_teacher_path is not None:
        source = torch.load(
            a6_teacher_path, map_location=args.device, weights_only=False
        )
        validate_legacy_teacher(
            source, state["physics_revision"], state["a6_reset_variant"]
        )
        a6_teacher = HostPolicy(
            612, 29,
            log_std_bounds=(-2.3, math.log(source.get("action_std_max", 0.7))),
            upper_residual=True, waist_residual_scale=5.0,
        ).to(args.device)
        a6_teacher.load_state_dict(source["policy"], strict=True)
        a6_teacher.eval()
        if args.proprio_preescape_a6_teacher is not None:
            proprio_preescape_a6_teacher = a6_teacher
        else:
            proprio_postescape_a6_teacher = a6_teacher
    if args.proprio_postescape_flat_teacher is not None:
        source = torch.load(
            args.proprio_postescape_flat_teacher, map_location=args.device, weights_only=False
        )
        if source.get("task") != "flat29" or source.get("model_source") != "unitree_g1_29":
            raise ValueError("proprio postescape teacher must be a 29-DoF flat checkpoint")
        proprio_postescape_teacher = HostPolicy(
            564, 29,
            log_std_bounds=(-2.3, math.log(source.get("action_std_max", 0.7))),
            upper_residual=True, waist_residual_scale=5.0,
        ).to(args.device)
        proprio_postescape_teacher.load_state_dict(source["policy"], strict=True)
        proprio_postescape_teacher.eval()
    if args.postescape_teacher_checkpoint is not None:
        source = torch.load(
            args.postescape_teacher_checkpoint,
            map_location=args.device,
            weights_only=False,
        )
        if source.get("task") != "flat29" or source.get("model_source") != "unitree_g1_29":
            raise ValueError("postescape teacher must be a 29-DoF flat checkpoint")
        postescape_teacher = HostPolicy(
            564,
            29,
            log_std_bounds=(-2.3, math.log(source.get("action_std_max", 0.7))),
            upper_residual=True,
            waist_residual_scale=5.0,
        ).to(args.device)
        postescape_teacher.load_state_dict(source["policy"], strict=True)
        postescape_teacher.eval()
    supine_teacher = None
    if args.supine_teacher_checkpoint is not None:
        source = torch.load(
            args.supine_teacher_checkpoint, map_location=args.device, weights_only=False
        )
        if source.get("model_source") != "unitree_g1_29" or not (
            "_staged_escape_v5" in source.get("reward_variant", "")
        ):
            raise ValueError("supine teacher must be a 29-DoF staged-v5 A6 checkpoint")
        supine_teacher = HostPolicy(
            612,
            29,
            log_std_bounds=(-2.3, math.log(source.get("action_std_max", 0.7))),
            upper_residual=True,
            waist_residual_scale=5.0,
        ).to(args.device)
        supine_teacher.load_state_dict(source["policy"], strict=True)
        supine_teacher.eval()
    selected_rows = ALL_LAYOUT_ROWS if args.all_directions else LAYOUT_ROWS
    selected_labels = ALL_LABELS if args.all_directions else LABELS
    layout = (
        np.tile(np.asarray((1, 1, 3), dtype=np.int64), (args.per_scene, 1))
        if args.all_postescape
        else np.repeat(np.asarray(selected_rows, dtype=np.int64), args.per_scene, axis=0)
    )
    cfg = host_a6_env_cfg(
        len(layout),
        args.seed,
        arm=state["arm"],
        edge_curriculum=True,
        plate_observation=not proprio_actor,
        critic_plate_observation=torque_v2,
        proprio_motor_torque=torque_v2,
        safe_actions="_safe_actions_v1" in variant
        and not (args.no_privileged_actions or args.disable_safe_scaling),
        prone_full_arms=(
            ("_prone_full_arms_v1" in variant or args.prone_full_arms_override)
            and not (args.no_privileged_actions or args.disable_safe_scaling)
        ),
        supine_full_arms="_supine_full_arms_v1" in variant
        and not (args.no_privileged_actions or args.disable_safe_scaling),
        aligned_edge="_aligned_edge_v1" in variant,
        lateral_actions="_lateral_actions_v1" in variant
        and not (args.no_privileged_actions or args.disable_safe_scaling),
        lift_guard="_lift_guard_v1" in variant
        and not (args.no_privileged_actions or args.disable_lift_guard),
        privileged_guided_actions=not proprio_actor and not args.no_privileged_actions,
        footprint_edge="_footprint_edge_v1" in variant,
        footprint_overlap_m=state.get("a6_footprint_overlap_m", 0.25),
    )
    cfg.events["reset_scene"].params["layout"] = layout
    cfg.actions["joint_pos"].diagnostic_guided_assist = (
        args.diagnostic_guided_side_assist_n > 0
    )
    if args.arm_actuation_scale is not None:
        cfg.actions["joint_pos"].guided_arm_scale = args.arm_actuation_scale
    if args.hip_actuation_scale is not None:
        cfg.actions["joint_pos"].guided_hip_scale = args.hip_actuation_scale
    cfg.auto_reset = False
    cfg.episode_length_s = args.steps * 0.02
    env = ManagerBasedRlEnv(cfg, device=args.device)
    prone_router = (
        InitialProneRouter(env.num_envs, args.device)
        if prone_policy is not None else None
    )
    side_router = (
        InitialSideRouter(env.num_envs, args.device)
        if side_policy is not None or args.mirror_right_initial_side_main else None
    )
    recovery_controller = None
    if args.proprio_recovery_controller is not None:
        recovery_controller, controller_artifact = load_recovery_controller(
            args.proprio_recovery_controller, env.num_envs, args.device
        )
        if args.recovery_pulse_blend_steps > recovery_controller.pulse_steps // 2:
            parser.error("recovery pulse blend exceeds half the pulse duration")
        recovery_controller.pulse_blend_steps = args.recovery_pulse_blend_steps
        if args.diagnostic_controller_threshold is not None:
            recovery_controller.threshold = args.diagnostic_controller_threshold
        if controller_artifact["observation_contract"] != state["observation_contract"]:
            raise ValueError("recovery controller and actor observation contracts differ")
    supine_flat_gate = None
    if args.proprio_supine_flat_gate is not None:
        supine_flat_gate, gate_artifact = load_supine_flat_gate(
            args.proprio_supine_flat_gate, env.num_envs, args.device,
            threshold=args.flat_gate_threshold,
            consecutive=args.flat_gate_consecutive,
            min_steps=args.flat_gate_min_steps,
        )
        if gate_artifact["observation_contract"] != state["observation_contract"]:
            raise ValueError("flat gate and actor observation contracts differ")
    early_flat_gate = None
    if args.proprio_early_flat_gate is not None:
        classifier, scene_artifact = load_scene_classifier(
            args.proprio_early_flat_gate, args.device,
        )
        if scene_artifact["observation_contract"] != state["observation_contract"]:
            raise ValueError("early flat gate and actor observation contracts differ")
        early_flat_gate = ProprioEarlyFlatGate(
            classifier, env.num_envs, args.device,
            threshold=args.early_flat_threshold,
            consecutive=args.early_flat_consecutive,
        )
    guided_prone_gate = None
    if args.proprio_guided_prone_gate is not None:
        classifier, scene_artifact = load_scene_classifier(
            args.proprio_guided_prone_gate, args.device,
        )
        if (scene_artifact["observation_contract"] != state["observation_contract"]
                or scene_artifact.get("posture") != "prone"):
            raise ValueError("guided-prone scene gate has an incompatible input contract")
        guided_prone_gate = ProprioGuidedProneGate(
            classifier, env.num_envs, args.device,
            threshold=args.guided_prone_threshold,
            consecutive=args.guided_prone_consecutive,
            min_steps=args.guided_prone_min_steps,
            max_steps=args.guided_prone_max_steps,
        )
    prone_flat_gate = None
    if args.proprio_prone_flat_gate is not None:
        classifier, scene_artifact = load_scene_classifier(
            args.proprio_prone_flat_gate, args.device,
        )
        if (scene_artifact["observation_contract"] != state["observation_contract"]
                or scene_artifact.get("posture") != "prone"):
            raise ValueError("prone-flat scene gate has an incompatible input contract")
        prone_flat_gate = ProprioSceneGate(
            classifier, env.num_envs, args.device,
            threshold=args.prone_flat_threshold,
            consecutive=args.prone_flat_consecutive,
            min_steps=31, max_steps=args.prone_flat_pulse_step,
            target_class=0, posture="prone",
        )
    prone_free_gate = None
    if args.proprio_prone_free_gate is not None:
        classifier, scene_artifact = load_scene_classifier(
            args.proprio_prone_free_gate, args.device,
        )
        if (scene_artifact["observation_contract"] != state["observation_contract"]
                or scene_artifact.get("posture") != "prone"):
            raise ValueError("prone-free scene gate has an incompatible input contract")
        prone_free_gate = ProprioSceneGate(
            classifier, env.num_envs, args.device,
            threshold=args.prone_free_threshold,
            consecutive=args.prone_free_consecutive,
            min_steps=31, max_steps=args.prone_free_pulse_step,
            target_class=2, posture="prone",
        )
    guided_side_gate = None
    if args.proprio_guided_side_gate is not None:
        classifier, scene_artifact = load_scene_classifier(
            args.proprio_guided_side_gate, args.device,
        )
        if (scene_artifact["observation_contract"] != state["observation_contract"]
                or scene_artifact.get("posture") != "side"):
            raise ValueError("guided-side scene gate has an incompatible input contract")
        early_classifier = None
        if args.proprio_guided_side_early_gate is not None:
            early_classifier, early_artifact = load_scene_classifier(
                args.proprio_guided_side_early_gate, args.device,
            )
            if (early_artifact["observation_contract"] != state["observation_contract"]
                    or early_artifact.get("posture") != "side"):
                raise ValueError("early guided-side gate has an incompatible input contract")
        right_classifier = None
        if args.proprio_guided_right_side_gate is not None:
            right_classifier, right_artifact = load_scene_classifier(
                args.proprio_guided_right_side_gate, args.device,
            )
            if (right_artifact["observation_contract"] != state["observation_contract"]
                    or right_artifact.get("posture") != "side"):
                raise ValueError("right guided-side gate has an incompatible input contract")
        guided_side_gate = ProprioGuidedSideGate(
            classifier, env.num_envs, args.device,
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
    try:
        # The first actor frame contains beta. Restore it before reset builds
        # the six-frame history, especially for fragile replayed exit poses.
        term = env.action_manager.get_term("joint_pos")
        term.beta.fill_(float(state["curriculum"]["beta"].float().mean()))
        term.assist_force.zero_()
        term.guided_escape_assist_force.zero_()
        term.guided_escape_assist_speed_target.zero_()
        if args.postescape_bank is not None:
            env._a6_postescape_bank_path = args.postescape_bank
        env._a6_postescape_live_obs_v1 = (
            postescape_live_observation_enabled(state)
            or args.postescape_live_obs_override
        )
        env._a6_staged_escape_v5 = staged_v5
        env._a6_prone_recovery_v1 = "_prone_recovery_v1" in variant
        env._a6_bidirectional_recovery_v1 = "_bidirectional_recovery_v1" in variant
        env._a6_strong_recovery_v1 = "_strong_recovery_v1" in variant
        env._a6_recovery_bridge_v2 = "_recovery_bridge_v2" in variant
        env._a6_knee_straight_prior_v1 = "_knee_straight_prior_v1" in variant
        env._a6_arm_extraction_v1 = "_arm_extraction_v1" in variant
        env._a6_prone_flexible_exit_v1 = (
            "_prone_flexible_exit_v1" in variant or args.prone_flexible_exit_override
        )
        env._a6_scene_phase_reward_v1 = "_scene_phase_reward_v1" in variant
        env._a6_side_lift_anticipation_v1 = "_side_lift_anticipation_v1" in variant
        env._a6_side_directional_progress_v1 = "_side_directional_progress_v1" in variant
        env._a6_free_plate_motion_reward_v1 = (
            "_free_plate_motion_reward_v1" in variant
        )
        env._a6_flat_prone_compact_v1 = "_flat_prone_compact_v1" in variant
        env._a6_free_plate_anchor_v1 = "_free_plate_anchor_v1" in variant
        env._a6_edge_offset_m = args.guided_offset_m
        # Network/controller construction consumes global RNG. Reseed just
        # before reset so paired baseline/controller runs share initial poses.
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        obs, _ = env.reset()
        if args.restore_postescape_bank_history:
            if "actor_obs" not in env._a6_postescape_bank:
                raise ValueError("postescape bank has no recorded actor history")
            obs = restore_postescape_actor_history(env)
        initial_gravity_b = env.scene["robot"].data.projected_gravity_b.clone()
        initial_side = (
            (initial_gravity_b[:, 1].abs() > 0.70)
            & (initial_gravity_b[:, 0].abs() < 0.65)
        )
        if prone_router is not None:
            prone_router.arm_initial_gravity(initial_gravity_b)
        if side_router is not None:
            side_router.arm_initial_gravity(initial_gravity_b)
        if guided_side_gate is not None:
            guided_side_gate.arm_initial_gravity(initial_gravity_b)
        plate_rows = torch.arange(env.num_envs, device=env.device)
        plate_ids = env._a6_plate_geoms[env._a6_scene.clamp_min(1) - 1]
        initial_plate_xy = env.sim.data.geom_xpos[
            plate_rows, plate_ids, :2
        ].clone()
        if supine_flat_gate is not None:
            supine_flat_gate.arm_initial_gravity(initial_gravity_b)
        if early_flat_gate is not None:
            early_flat_gate.arm_initial_gravity(initial_gravity_b)
        if guided_prone_gate is not None:
            guided_prone_gate.arm_initial_gravity(initial_gravity_b)
        if prone_flat_gate is not None:
            prone_flat_gate.arm_initial_gravity(initial_gravity_b)
        if prone_free_gate is not None:
            prone_free_gate.arm_initial_gravity(initial_gravity_b)
        if args.postescape_motor_delay_override:
            # Diagnostic: recover the original HoST 30-step motor wait even
            # for replayed escape states, while leaving their poses unchanged.
            env._a6_postescape_start.zero_()
        term.beta.fill_(float(state["curriculum"]["beta"].float().mean()))
        term.assist_force.zero_()
        term.guided_escape_assist_force.zero_()
        term.guided_escape_assist_speed_target.zero_()
        if args.diagnostic_guided_side_assist_n:
            guided_side = (env._a6_scene == 1) & (env._a6_direction >= 2)
            term.guided_escape_assist_force[guided_side] = args.diagnostic_guided_side_assist_n
            term.guided_escape_assist_speed_target[guided_side] = 0.3
        active = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
        first_escape = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
        first_phase_trigger = torch.zeros_like(first_escape)
        first_flat_gate_trigger = torch.zeros_like(first_escape)
        first_early_flat_trigger = torch.zeros_like(first_escape)
        first_guided_prone_scene_trigger = torch.zeros_like(first_escape)
        first_prone_flat_gate_trigger = torch.zeros_like(first_escape)
        forced_prone_flat_pulse = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        first_prone_free_gate_trigger = torch.zeros_like(first_escape)
        forced_prone_free_pulse = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        first_guided_side_scene_trigger = torch.zeros_like(first_escape)
        guided_prone_trigger_contact = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        guided_prone_trigger_coverage = torch.zeros(env.num_envs, device=env.device)
        early_flat_trigger_contact = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        early_flat_trigger_coverage = torch.zeros(env.num_envs, device=env.device)
        initial_supine_ever = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        flat_gate_trigger_contact = torch.zeros_like(initial_supine_ever)
        flat_gate_trigger_coverage = torch.zeros(env.num_envs, device=env.device)
        flat_gate_trigger_core_clearance = torch.zeros_like(flat_gate_trigger_coverage)
        flat_gate_trigger_force = torch.zeros_like(flat_gate_trigger_coverage)
        flat_gate_trigger_lift = torch.zeros_like(flat_gate_trigger_coverage)
        flat_gate_trigger_torque_peak = torch.zeros_like(flat_gate_trigger_coverage)
        flat_gate_trigger_torque_rms = torch.zeros_like(flat_gate_trigger_coverage)
        shadow_age = torch.zeros_like(first_escape)
        first_shadow_escape = torch.zeros_like(first_escape)
        shadow_core_at_escape = torch.zeros(env.num_envs, device=env.device)
        shadow_tip_at_escape = torch.zeros_like(shadow_core_at_escape)
        shadow_lift_at_escape = torch.zeros_like(shadow_core_at_escape)
        phase_streak = torch.zeros_like(first_escape)
        initial_posture_seen = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        initial_prone = torch.zeros_like(initial_posture_seen)
        initial_prone_ever = torch.zeros_like(initial_prone)
        if args.all_postescape:
            first_escape[env._a6_escaped] = 1
        first_invalid = torch.zeros_like(first_escape)
        first_stand = torch.zeros_like(first_escape)
        height_at_escape = torch.zeros(env.num_envs, device=env.device)
        height_after_10 = torch.zeros_like(height_at_escape)
        height_after_20 = torch.zeros_like(height_at_escape)
        escape_root_xy = torch.zeros(env.num_envs, 2, device=env.device)
        max_low_root_travel = torch.zeros(env.num_envs, device=env.device)
        max_postescape_low_travel = torch.zeros_like(max_low_root_travel)
        max_clearance = torch.zeros(env.num_envs, device=env.device)
        max_guided_force = torch.zeros_like(max_clearance)
        max_guided_lift = torch.zeros_like(max_clearance)
        max_preescape_free_plate_shift = torch.zeros_like(max_clearance)
        free_plate_shift_at_escape = torch.zeros_like(max_clearance)
        # Diagnostic only: compare executed joint targets without changing them.
        previous_action = torch.zeros(env.num_envs, 29, device=env.device)
        action_delta_sum = torch.zeros_like(max_clearance)
        action_delta_count = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
        max_action_delta = torch.zeros_like(max_clearance)
        side_gate_action_delta = torch.zeros_like(max_clearance)
        flat_gate_action_delta = torch.zeros_like(max_clearance)
        phase_pulse_action_delta = torch.zeros_like(max_clearance)
        invalid_by_force = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        invalid_by_lift = torch.zeros_like(invalid_by_force)
        invalid_other = torch.zeros_like(invalid_by_force)
        final_height = torch.zeros_like(max_clearance)
        final_upright = torch.zeros_like(max_clearance)
        knee_trace = None
        if args.knee_diagnostics:
            knee_ids = [
                HOST29_JOINT_NAMES.index(f"{side}_knee_joint")
                for side in ("left", "right")
            ]
            standing_knees = torch.tensor(
                [HOST29_TARGET_JOINT_POS[f"{side}_knee_joint"] for side in ("left", "right")],
                device=env.device,
            )
            knee_trace = torch.full(
                (args.steps, env.num_envs), float("nan"), device=env.device
            )
        for step in range(1, args.steps + 1):
            with torch.no_grad():
                if step == args.diagnostic_flat_prone_early_pulse_step:
                    early_pulse = recovery_controller.schedule_pulse(
                        active & (env._a6_scene == 0) & (env._a6_direction == 1),
                        term.real_steps, step,
                    )
                    first_phase_trigger = torch.where(
                        early_pulse, step, first_phase_trigger,
                    )
                if step == args.diagnostic_free_prone_early_pulse_step:
                    early_pulse = recovery_controller.schedule_pulse(
                        active & (env._a6_scene == 2) & (env._a6_direction == 1),
                        term.real_steps, step,
                    )
                    first_phase_trigger = torch.where(
                        early_pulse, step, first_phase_trigger,
                    )
                if prone_flat_gate is not None:
                    flat_prone_trigger = prone_flat_gate.observe(
                        obs["actor"], term.real_steps, eligible=active,
                    )
                    first_prone_flat_gate_trigger = torch.where(
                        flat_prone_trigger, step, first_prone_flat_gate_trigger,
                    )
                    if step == args.prone_flat_pulse_step:
                        early_pulse = recovery_controller.schedule_pulse(
                            active & prone_flat_gate.triggered,
                            term.real_steps, step,
                        )
                        first_phase_trigger = torch.where(
                            early_pulse, step, first_phase_trigger,
                        )
                        forced_prone_flat_pulse |= early_pulse
                if prone_free_gate is not None:
                    free_prone_trigger = prone_free_gate.observe(
                        obs["actor"], term.real_steps, eligible=active,
                    )
                    first_prone_free_gate_trigger = torch.where(
                        free_prone_trigger, step, first_prone_free_gate_trigger,
                    )
                    if step == args.prone_free_pulse_step:
                        early_pulse = recovery_controller.schedule_pulse(
                            active & prone_free_gate.triggered,
                            term.real_steps, step,
                        )
                        first_phase_trigger = torch.where(
                            early_pulse, step, first_phase_trigger,
                        )
                        forced_prone_free_pulse |= early_pulse
                if early_flat_gate is not None:
                    early_trigger = early_flat_gate.observe(
                        obs["actor"], term.real_steps, eligible=active,
                    )
                    first_early_flat_trigger = torch.where(
                        early_trigger, step, first_early_flat_trigger,
                    )
                    if early_trigger.any():
                        pos, ext, plate_pos, plate_ext = plate_footprint_inputs(env)
                        coverage, _, _ = footprint(pos, ext, plate_pos, plate_ext)
                        guided_contact = env.scene["guided_contact"].data
                        free_contact = env.scene["free_contact"].data
                        is_free = env._a6_scene == 2
                        touching = torch.where(
                            is_free,
                            (free_contact.found > 0).any(-1)
                            | (free_contact.force_history.norm(dim=-1) > 1).any(dim=(1, 2)),
                            (guided_contact.found > 0).any(-1)
                            | (guided_contact.force_history.norm(dim=-1) > 1).any(dim=(1, 2)),
                        )
                        early_flat_trigger_contact |= early_trigger & touching
                        early_flat_trigger_coverage = torch.where(
                            early_trigger, coverage.float(), early_flat_trigger_coverage,
                        )
                if guided_prone_gate is not None:
                    guided_scene_trigger = guided_prone_gate.observe(
                        obs["actor"], term.real_steps, eligible=active,
                    )
                    first_guided_prone_scene_trigger = torch.where(
                        guided_scene_trigger, step, first_guided_prone_scene_trigger,
                    )
                    if guided_scene_trigger.any():
                        pos, ext, plate_pos, plate_ext = plate_footprint_inputs(env)
                        coverage, _, _ = footprint(pos, ext, plate_pos, plate_ext)
                        guided_contact = env.scene["guided_contact"].data
                        free_contact = env.scene["free_contact"].data
                        is_free = env._a6_scene == 2
                        touching = torch.where(
                            is_free,
                            (free_contact.found > 0).any(-1)
                            | (free_contact.force_history.norm(dim=-1) > 1).any(dim=(1, 2)),
                            (guided_contact.found > 0).any(-1)
                            | (guided_contact.force_history.norm(dim=-1) > 1).any(dim=(1, 2)),
                        )
                        guided_prone_trigger_contact |= guided_scene_trigger & touching
                        guided_prone_trigger_coverage = torch.where(
                            guided_scene_trigger, coverage.float(), guided_prone_trigger_coverage,
                        )
                if guided_side_gate is not None:
                    side_scene_trigger = guided_side_gate.observe(
                        obs["actor"], term.real_steps, eligible=active,
                    )
                    first_guided_side_scene_trigger = torch.where(
                        side_scene_trigger, step, first_guided_side_scene_trigger,
                    )
                if supine_flat_gate is not None:
                    flat_trigger = supine_flat_gate.observe(
                        obs["actor"], term.real_steps, eligible=active,
                    )
                    first_flat_gate_trigger = torch.where(
                        flat_trigger, step, first_flat_gate_trigger,
                    )
                    initial_supine_ever |= supine_flat_gate.initial_supine
                    if flat_trigger.any():
                        pos, ext, plate_pos, plate_ext = plate_footprint_inputs(env)
                        coverage, _, _ = footprint(pos, ext, plate_pos, plate_ext)
                        core_clearance, _ = prone_exit_geometry(
                            pos, ext, plate_pos, plate_ext,
                            env._a6_prone_core_collision_indices,
                        )
                        guided_contact = env.scene["guided_contact"].data
                        free_contact = env.scene["free_contact"].data
                        is_free = env._a6_scene == 2
                        touching = torch.where(
                            is_free,
                            (free_contact.found > 0).any(-1)
                            | (free_contact.force_history.norm(dim=-1) > 1).any(dim=(1, 2)),
                            (guided_contact.found > 0).any(-1)
                            | (guided_contact.force_history.norm(dim=-1) > 1).any(dim=(1, 2)),
                        )
                        force_now = torch.where(
                            is_free,
                            free_contact.force_history.norm(dim=-1).amax(dim=(1, 2)),
                            guided_contact.force_history.norm(dim=-1).amax(dim=(1, 2)),
                        )
                        flat_gate_trigger_contact |= flat_trigger & touching
                        flat_gate_trigger_coverage = torch.where(
                            flat_trigger, coverage.float(), flat_gate_trigger_coverage,
                        )
                        flat_gate_trigger_core_clearance = torch.where(
                            flat_trigger, core_clearance, flat_gate_trigger_core_clearance,
                        )
                        flat_gate_trigger_force = torch.where(
                            flat_trigger, force_now, flat_gate_trigger_force,
                        )
                        flat_gate_trigger_lift = torch.where(
                            flat_trigger, env._a6_plate_lift_peak, flat_gate_trigger_lift,
                        )
                        current_torque = obs["actor"][:, -29:]
                        flat_gate_trigger_torque_peak = torch.where(
                            flat_trigger, current_torque.abs().amax(-1),
                            flat_gate_trigger_torque_peak,
                        )
                        flat_gate_trigger_torque_rms = torch.where(
                            flat_trigger, current_torque.square().mean(-1).sqrt(),
                            flat_gate_trigger_torque_rms,
                        )
                if args.proprio_phase_initial_prone_gate:
                    first_active = (
                        active & ~initial_posture_seen
                        & (env.action_manager.get_term("joint_pos").real_steps > 30)
                    )
                    initial_prone[first_active] = obs["actor"][first_active, 5 * 94 + 3] > 0.70
                    initial_posture_seen |= first_active
                escaped_before = env._a6_escaped.clone()
                policy_input = obs["actor"]
                if args.diagnostic_zero_plate_obs:
                    if policy_input.shape[1] != 612:
                        raise ValueError("zero-plate diagnostic requires six 102D legacy actor frames")
                    policy_input = policy_input.clone().reshape(env.num_envs, 6, 102)
                    policy_input[:, :, 94:102] = 0
                    policy_input = policy_input.reshape(env.num_envs, 612)
                mean_action = policy.deterministic_action(policy_input)
                if args.mirror_right_initial_side_main:
                    mirrored_right = (
                        side_router.initial_side & ~side_router.initial_left_side
                    )
                    if mirrored_right.any():
                        mirrored_input = mirror_host29_actor_obs(
                            obs["actor"][mirrored_right]
                        )
                        mean_action[mirrored_right] = mirror_host29_action(
                            policy.deterministic_action(mirrored_input)
                        )
                action = (
                    policy.distribution(policy_input).sample()
                    if args.sample_action else mean_action
                )
                if prone_policy is not None:
                    prone_action = prone_policy.deterministic_action(obs["actor"])
                    action = torch.where(
                        prone_router.initial_prone[:, None], prone_action, action,
                    )
                pre_side_action = action
                if side_policy is not None:
                    side_action = side_policy.deterministic_action(obs["actor"])
                    if args.mirror_left_side_specialist:
                        mirrored_left = side_router.initial_left_side
                        if mirrored_left.any():
                            mirrored_input = mirror_host29_actor_obs(
                                obs["actor"][mirrored_left]
                            )
                            side_action[mirrored_left] = mirror_host29_action(
                                side_policy.deterministic_action(mirrored_input)
                            )
                    if args.side_soft_gate_v1:
                        guided_probability = guided_side_gate.classifier(
                            obs["actor"]
                        ).softmax(-1)[:, 1]
                        side_soft_weight = (
                            (guided_probability - args.side_soft_gate_low)
                            / (args.side_soft_gate_high - args.side_soft_gate_low)
                        ).clamp(0, 1) * side_router.initial_side.float()
                        side_route = side_router.initial_side
                        action = torch.lerp(
                            action, side_action, side_soft_weight[:, None]
                        )
                    else:
                        side_route = (
                            guided_side_gate.triggered if guided_side_gate is not None
                            else side_router.initial_side
                        )
                        if args.left_side_specialist_probe_through_step:
                            side_route = side_route | (
                                side_router.initial_left_side
                                & (term.real_steps <= args.left_side_specialist_probe_through_step)
                            )
                        action = torch.where(
                            side_route[:, None], side_action, action,
                        )
                    if left_side_policy is not None:
                        left_action = left_side_policy.deterministic_action(obs["actor"])
                        action = torch.where(
                            (side_route & side_router.initial_left_side)[:, None],
                            left_action, action,
                        )
                if guided_prone_policy is not None:
                    guided_prone_action = guided_prone_policy.deterministic_action(obs["actor"])
                    action = torch.where(
                        guided_prone_gate.triggered[:, None], guided_prone_action, action,
                    )
                if args.mean_after_escape:
                    escaped_guided = (env._a6_scene == 1) & env._a6_escaped
                    action = torch.where(escaped_guided[:, None], mean_action, action)
                if supine_teacher is not None:
                    guided_supine = (env._a6_scene == 1) & (env._a6_direction == 0)
                    teacher_action = supine_teacher.deterministic_action(obs["actor"])
                    action = torch.where(guided_supine[:, None], teacher_action, action)
                if postescape_teacher is not None:
                    action = postescape_teacher.deterministic_action(obs["actor"][:, :564])
                if proprio_postescape_teacher is not None:
                    escaped_guided = (env._a6_scene == 1) & env._a6_escaped
                    flat_action = proprio_postescape_teacher.deterministic_action(
                        obs["actor"][:, :564]
                    )
                    expert_mask = (
                        supine_flat_gate.triggered
                        if supine_flat_gate is not None else escaped_guided
                    )
                    if early_flat_gate is not None:
                        expert_mask = expert_mask | early_flat_gate.triggered
                    if args.diagnostic_flat_expert_on_flat:
                        expert_mask = expert_mask | (env._a6_scene == 0)
                    if args.diagnostic_flat_expert_after_prone_pulse:
                        expert_mask = expert_mask | (
                            (env._a6_scene == 0)
                            & (env._a6_direction == 1)
                            & recovery_controller.triggered
                            & (recovery_controller.pulse_remaining == 0)
                        )
                    if args.diagnostic_flat_expert_on_free_after_escape:
                        expert_mask = expert_mask | (
                            (env._a6_scene == 2) & env._a6_escaped
                        )
                    action = torch.where(expert_mask[:, None], flat_action, action)
                if proprio_postescape_a6_teacher is not None:
                    escaped_guided = (env._a6_scene == 1) & env._a6_escaped
                    teacher_action = proprio_postescape_a6_teacher.deterministic_action(
                        legacy_teacher_observation(obs["critic"])
                    )
                    action = torch.where(escaped_guided[:, None], teacher_action, action)
                if proprio_preescape_a6_teacher is not None:
                    obstructed_guided = (
                        (env._a6_scene == 1)
                        & (env._a6_direction < 2)
                        & ~env._a6_escaped
                    )
                    teacher_action = effective_guided_teacher_action(
                        proprio_preescape_a6_teacher.deterministic_action(
                            legacy_teacher_observation(obs["critic"])
                        )
                    )
                    action = torch.where(
                        obstructed_guided[:, None], teacher_action, action
                    )
                if args.bake_safe_scaling_into_policy:
                    guided_under_board = (
                        (env._a6_scene == 1)
                        & (env._a6_direction < 2)
                        & ~env._a6_escaped
                    )
                    baked_action = effective_guided_teacher_action(action)
                    action = torch.where(guided_under_board[:, None], baked_action, action)
                if args.bake_hip_scaling_all_scenes:
                    action = effective_guided_teacher_action(action)
                if args.proprio_oracle_pulse_steps:
                    pulse = (
                        active & (env._a6_scene == 1) & (env._a6_direction == 1)
                        & (first_escape > 0)
                        & (step - first_escape <= args.proprio_oracle_pulse_steps)
                    )
                    if pulse.any():
                        pulse_action = policy.deterministic_action(
                            torch.zeros_like(obs["actor"])
                        )
                        action = torch.where(pulse[:, None], pulse_action, action)
                if args.proprio_phase_pulse_steps:
                    pulse = (
                        active & (first_phase_trigger > 0)
                        & (step - first_phase_trigger <= args.proprio_phase_pulse_steps)
                    )
                    if pulse.any():
                        pulse_action = policy.deterministic_action(
                            torch.zeros_like(obs["actor"])
                        )
                        action = torch.where(pulse[:, None], pulse_action, action)
                if recovery_controller is not None:
                    action, _ = recovery_controller.apply(action)
                if args.supine_probe_v1:
                    action, _ = supine_flat_gate.probe_action(action, term.real_steps)
                if args.supine_neutral_through_step:
                    neutral = supine_flat_gate.neutral_wait_mask(
                        term.real_steps, args.supine_neutral_through_step,
                    )
                    action = torch.where(neutral[:, None], torch.zeros_like(action), action)
                if args.side_neutral_through_step:
                    side_neutral = (
                        initial_side & (term.real_steps > 30)
                        & (term.real_steps <= args.side_neutral_through_step)
                    )
                    action = torch.where(
                        side_neutral[:, None], torch.zeros_like(action), action
                    )
                if args.side_action_scale < 1.0:
                    action = torch.where(
                        initial_side[:, None], action * args.side_action_scale, action
                    )
                if args.initial_side_joint_safe_through_step and step <= args.initial_side_joint_safe_through_step:
                    if args.side_soft_gate_v1:
                        factor = 1.0 - (
                            1.0 - args.initial_side_joint_safe_scale
                        ) * side_soft_weight
                        action = action.clone()
                        action[:, SIDE_SCALED_IDS] *= factor[:, None]
                    else:
                        side_safe = initial_side & active
                        if args.side_safe_only_when_guided_gate:
                            side_safe &= (
                                guided_side_gate.triggered
                                | (step <= args.side_safe_probe_through_step)
                                | (
                                    side_router.initial_left_side
                                    & (term.real_steps <= args.left_side_specialist_probe_through_step)
                                )
                            )
                        if side_safe.any():
                            action = action.clone()
                            action[side_safe] = effective_guided_side_teacher_action(
                                action[side_safe], scale=args.initial_side_joint_safe_scale
                            )
                if args.side_lateral_gain > 1.0:
                    lateral = (
                        active & guided_side_gate.triggered
                        & (term.real_steps <= args.initial_side_joint_safe_through_step)
                    )
                    action = apply_side_lateral_gain(
                        action, lateral, args.side_lateral_gain,
                    )
                if args.diagnostic_guided_nonside_lateral_gain > 1.0:
                    oracle_lateral = (
                        active & (env._a6_scene == 1) & (env._a6_direction < 2)
                        & ~env._a6_escaped & (term.real_steps > 30)
                        & (term.real_steps <= 500)
                    )
                    action = apply_side_lateral_gain(
                        action, oracle_lateral,
                        args.diagnostic_guided_nonside_lateral_gain,
                    )
                if args.diagnostic_guided_nonside_safe_scale < 1.0:
                    oracle_safe = (
                        active & (env._a6_scene == 1) & (env._a6_direction < 2)
                        & ~env._a6_escaped & (term.real_steps > 30)
                        & (term.real_steps <= 500)
                    )
                    if oracle_safe.any():
                        action = action.clone()
                        action[oracle_safe] = effective_guided_side_teacher_action(
                            action[oracle_safe],
                            scale=args.diagnostic_guided_nonside_safe_scale,
                        )
                if args.side_handoff_steps:
                    handoff_age = guided_side_gate.trigger_age
                    blending = active & (handoff_age > 0) & (
                        handoff_age < args.side_handoff_steps
                    )
                    if blending.any():
                        alpha = (handoff_age.float() / args.side_handoff_steps).clamp(0, 1)
                        action = torch.where(
                            blending[:, None],
                            torch.lerp(pre_side_action, action, alpha[:, None]),
                            action,
                        )
                # The first motor command follows HoST's thirty-step wait;
                # exclude that planned start from the continuity comparison.
                if step > 31:
                    action_delta = (action - previous_action).square().mean(-1).sqrt()
                    action_delta_sum += torch.where(
                        active, action_delta, torch.zeros_like(action_delta)
                    )
                    action_delta_count += active.long()
                    max_action_delta = torch.where(
                        active, torch.maximum(max_action_delta, action_delta),
                        max_action_delta,
                    )
                    side_gate_action_delta = torch.where(
                        active & (first_guided_side_scene_trigger == step),
                        action_delta, side_gate_action_delta,
                    )
                    flat_gate_action_delta = torch.where(
                        active & (first_flat_gate_trigger == step),
                        action_delta, flat_gate_action_delta,
                    )
                    phase_pulse_action_delta = torch.where(
                        active & (first_phase_trigger > 0)
                        & (first_phase_trigger == step - 1),
                        action_delta, phase_pulse_action_delta,
                    )
                previous_action.copy_(action)
                obs, _, terminated, truncated, _ = env.step(action)
                if args.shadow_prone_flexible_exit or args.shadow_side_flexible_exit:
                    pos, ext, plate_pos, plate_ext = plate_footprint_inputs(env)
                    core_clearance, trailing_overlap = prone_exit_geometry(
                        pos, ext, plate_pos, plate_ext,
                        env._a6_prone_core_collision_indices,
                    )
                    legacy_geometry = footprint(
                        pos, ext, plate_pos, plate_ext
                    )[2] >= 0.04
                    rows = torch.arange(env.num_envs, device=env.device)
                    plate_xyz = env.sim.data.geom_xpos[
                        rows, env._a6_plate_geoms[0]
                    ]
                    plate_lift = torch.maximum(
                        (plate_xyz[:, 2] - env._a6_initial_plate_z).clamp_min(0),
                        env._a6_plate_lift_peak,
                    )
                    contact_data = env.scene["guided_contact"].data
                    plate_contact = (
                        (contact_data.found > 0).any(-1)
                        | (contact_data.force_history.norm(dim=-1) > 1.0)
                        .any(dim=(1, 2))
                    )
                    shadow_direction = (
                        (env._a6_direction == 1)
                        if args.shadow_prone_flexible_exit
                        else (env._a6_direction >= 2)
                    )
                    shadow_clear = (
                        active & (env._a6_scene == 1)
                        & shadow_direction
                        & env._a6_ever_contact & ~plate_contact
                        & prone_flexible_clearance(
                            core_clearance, trailing_overlap,
                            plate_lift, legacy_geometry,
                        )
                    )
                    shadow_age = torch.where(
                        shadow_clear, shadow_age + 1, 0
                    )
                    new_shadow = (
                        active & (first_shadow_escape == 0)
                        & (shadow_age >= PRONE_EXIT_CONFIRM_STEPS)
                    )
                    first_shadow_escape = torch.where(
                        new_shadow, step, first_shadow_escape
                    )
                    shadow_core_at_escape = torch.where(
                        new_shadow, core_clearance, shadow_core_at_escape
                    )
                    shadow_tip_at_escape = torch.where(
                        new_shadow, trailing_overlap, shadow_tip_at_escape
                    )
                    shadow_lift_at_escape = torch.where(
                        new_shadow, plate_lift, shadow_lift_at_escape
                    )
                if recovery_controller is not None:
                    trigger = recovery_controller.observe(
                        obs["actor"], term.real_steps,
                        eligible=active & ~(terminated | truncated),
                    )
                    first_phase_trigger = torch.where(
                        trigger, step, first_phase_trigger
                    )
                    initial_prone_ever |= recovery_controller.initial_prone
                if args.proprio_a6_teacher_reset_history:
                    teacher_reset = (
                        (env._a6_scene == 1)
                        & (env._a6_direction == 1)
                        & ~escaped_before
                        & env._a6_escaped
                        & ~(terminated | truncated)
                    )
                    ids = teacher_reset.nonzero(as_tuple=False).squeeze(-1)
                    if len(ids):
                        manager = env.observation_manager
                        for buffer in manager._group_obs_term_history_buffer["critic"].values():
                            buffer.reset(batch_ids=ids)
                        obs["critic"] = manager.compute_group(
                            "critic", update_history=True, env_ids=ids
                        )
                        env.obs_buf["critic"] = obs["critic"]
                guided = env._a6_scene == 1
                force = env.scene["guided_contact"].data.force_history.norm(dim=-1).amax(dim=(1, 2))
                lift = env._a6_plate_lift_peak
                max_guided_force = torch.where(
                    active & guided,
                    torch.maximum(max_guided_force, force),
                    max_guided_force,
                )
                max_guided_lift = torch.where(
                    active & guided,
                    torch.maximum(max_guided_lift, lift),
                    max_guided_lift,
                )
                new_invalid = active & env._a6_invalid & (first_invalid == 0)
                force_trip = force > PLATE_FORCE_INVALID_N
                lift_trip = lift > GUIDED_MAX_LIFT_M
                invalid_by_force |= new_invalid & force_trip
                invalid_by_lift |= new_invalid & lift_trip
                invalid_other |= new_invalid & ~(force_trip | lift_trip)
                if phase_switch:
                    obs, _ = reset_prone_history_after_escape(
                        env, escaped_before, terminated | truncated
                    )
                if args.proprio_diagnostic_reset_prone_history:
                    obs, _ = reset_prone_history_after_escape(
                        env, escaped_before, terminated | truncated
                    )
                if phase_detector is not None:
                    phase_score = phase_detector(obs["actor"]).sigmoid()
                    phase_streak = torch.where(
                        active & (phase_score >= args.proprio_phase_threshold),
                        phase_streak + 1,
                        torch.zeros_like(phase_streak),
                    )
                    trigger = (
                        active & ~(terminated | truncated)
                        & (first_phase_trigger == 0)
                        & (initial_prone if args.proprio_phase_initial_prone_gate else True)
                        & (env.action_manager.get_term("joint_pos").real_steps
                           >= args.proprio_phase_min_steps)
                        & (phase_streak >= args.proprio_phase_consecutive)
                    )
                    ids = trigger.nonzero(as_tuple=False).squeeze(-1)
                    first_phase_trigger = torch.where(
                        trigger, step, first_phase_trigger
                    )
                    if len(ids) and args.proprio_phase_controller_reset:
                        env._a6_postescape_start[ids] = True
                        env.action_manager.get_term("joint_pos").real_steps[ids] = 0
                    if not args.proprio_phase_pulse_steps:
                        obs = (
                            reset_observation_history_from_proprio(env, ids)
                            if args.proprio_phase_full_observation_reset
                            else reset_actor_history_from_proprio(env, ids)
                        )
                if (args.proprio_oracle_actor_reset_only
                    or args.proprio_oracle_observation_reset_only
                    or args.proprio_oracle_controller_reset_only):
                    trigger = (
                        active & (env._a6_scene == 1)
                        & (env._a6_direction == 1)
                        & ~escaped_before & env._a6_escaped
                        & ~(terminated | truncated)
                    )
                    ids = trigger.nonzero(as_tuple=False).squeeze(-1)
                    first_phase_trigger = torch.where(
                        trigger, step, first_phase_trigger
                    )
                    if args.proprio_oracle_actor_reset_only:
                        obs = reset_actor_history_from_proprio(env, ids)
                    elif args.proprio_oracle_observation_reset_only:
                        obs = reset_observation_history_from_proprio(env, ids)
                    elif len(ids):
                        env._a6_postescape_start[ids] = True
                        env.action_manager.get_term("joint_pos").real_steps[ids] = 0
                if args.reset_supine_history_on_escape:
                    obs, _ = reset_guided_history_after_escape(
                        env, escaped_before, terminated | truncated, direction=0
                    )
                robot = env.scene["robot"].data
                plate_xy = env.sim.data.geom_xpos[plate_rows, plate_ids, :2]
                free_plate_shift = (plate_xy - initial_plate_xy).norm(dim=-1)
                max_preescape_free_plate_shift = torch.where(
                    active & (env._a6_scene == 2) & ~escaped_before,
                    torch.maximum(max_preescape_free_plate_shift, free_plate_shift),
                    max_preescape_free_plate_shift,
                )
                if knee_trace is not None:
                    knee_deviation = (
                        robot.joint_pos[:, knee_ids] - standing_knees
                    ).abs().amax(-1)
                    knee_trace[step - 1] = torch.where(
                        active & guided & ~escaped_before,
                        knee_deviation,
                        float("nan"),
                    )
                height = robot.root_link_pos_w[:, 2]
                upright = -robot.projected_gravity_b[:, 2]
                new_escape = active & env._a6_escaped & (first_escape == 0)
                free_plate_shift_at_escape = torch.where(
                    new_escape & (env._a6_scene == 2),
                    free_plate_shift,
                    free_plate_shift_at_escape,
                )
                root_xy = robot.root_link_pos_w[:, :2]
                escape_root_xy = torch.where(
                    new_escape[:, None], root_xy, escape_root_xy
                )
                height_at_escape = torch.where(new_escape, height, height_at_escape)
                first_escape = torch.where(
                    new_escape,
                    step,
                    first_escape,
                )
                escape_age = step - first_escape
                low = active & (height < 0.4)
                root_travel = (root_xy - env._a6_initial_root_xy).norm(dim=-1)
                max_low_root_travel = torch.where(
                    low, torch.maximum(max_low_root_travel, root_travel),
                    max_low_root_travel,
                )
                postescape_low = low & (first_escape > 0)
                postescape_travel = (root_xy - escape_root_xy).norm(dim=-1)
                max_postescape_low_travel = torch.where(
                    postescape_low,
                    torch.maximum(max_postescape_low_travel, postescape_travel),
                    max_postescape_low_travel,
                )
                height_after_10 = torch.where(
                    active & (first_escape > 0) & (escape_age == 10),
                    height,
                    height_after_10,
                )
                height_after_20 = torch.where(
                    active & (first_escape > 0) & (escape_age == 20),
                    height,
                    height_after_20,
                )
                first_invalid = torch.where(
                    active & env._a6_invalid & (first_invalid == 0),
                    step,
                    first_invalid,
                )
                stand = (height > 0.6) & (upright > 0.8)
                first_stand = torch.where(
                    active & stand & (first_stand == 0), step, first_stand
                )
                max_clearance = torch.where(
                    active,
                    torch.maximum(max_clearance, env._a6_best_clearance),
                    max_clearance,
                )
                final_height = torch.where(active, height, final_height)
                final_upright = torch.where(active, upright, final_upright)
                done = (terminated | truncated) & active
                if recovery_controller is not None and done.any():
                    recovery_controller.reset(done.nonzero(as_tuple=False).squeeze(-1))
                if args.diagnostic_guided_side_assist_n and done.any():
                    term.guided_escape_assist_force[done] = 0
                if supine_flat_gate is not None and done.any():
                    supine_flat_gate.reset(done.nonzero(as_tuple=False).squeeze(-1))
                if early_flat_gate is not None and done.any():
                    early_flat_gate.reset(done.nonzero(as_tuple=False).squeeze(-1))
                if guided_prone_gate is not None and done.any():
                    guided_prone_gate.reset(done.nonzero(as_tuple=False).squeeze(-1))
                if prone_flat_gate is not None and done.any():
                    prone_flat_gate.reset(done.nonzero(as_tuple=False).squeeze(-1))
                if prone_free_gate is not None and done.any():
                    prone_free_gate.reset(done.nonzero(as_tuple=False).squeeze(-1))
                if guided_side_gate is not None and done.any():
                    guided_side_gate.reset(done.nonzero(as_tuple=False).squeeze(-1))
                active &= ~done
                if not active.any():
                    break
                # Inactive worlds still advance in the batched simulator.
                # Reset every finished world so later steps stay well-defined.
                finished = (terminated | truncated).nonzero(as_tuple=False).squeeze(-1)
                if len(finished):
                    obs, _ = env.reset(env_ids=finished)
                    if args.restore_postescape_bank_history:
                        obs = restore_postescape_actor_history(env, finished)
                    if supine_flat_gate is not None:
                        supine_flat_gate.arm_initial_gravity(
                            env.scene["robot"].data.projected_gravity_b,
                            finished,
                        )
                    if early_flat_gate is not None:
                        early_flat_gate.arm_initial_gravity(
                            env.scene["robot"].data.projected_gravity_b,
                            finished,
                        )
                    if guided_prone_gate is not None:
                        guided_prone_gate.arm_initial_gravity(
                            env.scene["robot"].data.projected_gravity_b,
                            finished,
                        )
                    if prone_flat_gate is not None:
                        prone_flat_gate.arm_initial_gravity(
                            env.scene["robot"].data.projected_gravity_b,
                            finished,
                        )
                    if prone_free_gate is not None:
                        prone_free_gate.arm_initial_gravity(
                            env.scene["robot"].data.projected_gravity_b,
                            finished,
                        )
                    if guided_side_gate is not None:
                        guided_side_gate.arm_initial_gravity(
                            env.scene["robot"].data.projected_gravity_b,
                            finished,
                        )
                    if prone_router is not None:
                        prone_router.arm_initial_gravity(
                            env.scene["robot"].data.projected_gravity_b,
                            finished,
                        )
                    if side_router is not None:
                        side_router.arm_initial_gravity(
                            env.scene["robot"].data.projected_gravity_b,
                            finished,
                        )
        if recovery_controller is not None:
            initial_prone = initial_prone_ever
        result = {
            "checkpoint": str(args.checkpoint),
            "proprio_prone_specialist": (
                str(args.proprio_prone_specialist)
                if args.proprio_prone_specialist is not None else None
            ),
            "proprio_side_specialist": (
                str(args.proprio_side_specialist)
                if args.proprio_side_specialist is not None else None
            ),
            "proprio_left_side_specialist": (
                str(args.proprio_left_side_specialist)
                if args.proprio_left_side_specialist is not None else None
            ),
            "mirror_left_side_specialist": args.mirror_left_side_specialist,
            "mirror_right_initial_side_main": args.mirror_right_initial_side_main,
            "side_soft_gate_v1": args.side_soft_gate_v1,
            "side_handoff_steps": args.side_handoff_steps,
            "recovery_pulse_blend_steps": args.recovery_pulse_blend_steps,
            "diagnostic_flat_prone_early_pulse_step": (
                args.diagnostic_flat_prone_early_pulse_step
            ),
            "diagnostic_free_prone_early_pulse_step": (
                args.diagnostic_free_prone_early_pulse_step
            ),
            "proprio_prone_flat_gate": (
                str(args.proprio_prone_flat_gate)
                if args.proprio_prone_flat_gate is not None else None
            ),
            "prone_flat_pulse_step": args.prone_flat_pulse_step if prone_flat_gate is not None else None,
            "prone_flat_threshold": args.prone_flat_threshold if prone_flat_gate is not None else None,
            "prone_flat_consecutive": args.prone_flat_consecutive if prone_flat_gate is not None else None,
            "proprio_prone_free_gate": (
                str(args.proprio_prone_free_gate)
                if args.proprio_prone_free_gate is not None else None
            ),
            "prone_free_pulse_step": args.prone_free_pulse_step if prone_free_gate is not None else None,
            "prone_free_threshold": args.prone_free_threshold if prone_free_gate is not None else None,
            "prone_free_consecutive": args.prone_free_consecutive if prone_free_gate is not None else None,
            "side_soft_gate_low": args.side_soft_gate_low,
            "side_soft_gate_high": args.side_soft_gate_high,
            "proprio_guided_side_gate": (
                str(args.proprio_guided_side_gate)
                if args.proprio_guided_side_gate is not None else None
            ),
            "proprio_guided_side_early_gate": (
                str(args.proprio_guided_side_early_gate)
                if args.proprio_guided_side_early_gate is not None else None
            ),
            "guided_side_early_max_step": args.guided_side_early_max_step,
            "proprio_guided_right_side_gate": (
                str(args.proprio_guided_right_side_gate)
                if args.proprio_guided_right_side_gate is not None else None
            ),
            "guided_right_side_min_steps": args.guided_right_side_min_steps,
            "guided_right_side_max_steps": args.guided_right_side_max_steps,
            "side_safe_only_when_guided_gate": args.side_safe_only_when_guided_gate,
            "side_safe_probe_through_step": args.side_safe_probe_through_step,
            "left_side_specialist_probe_through_step": args.left_side_specialist_probe_through_step,
            "initial_side_joint_safe_scale": args.initial_side_joint_safe_scale,
            "guided_side_threshold": args.guided_side_threshold,
            "guided_side_consecutive": args.guided_side_consecutive,
            "guided_side_ema_alpha": args.guided_side_ema_alpha,
            "proprio_guided_prone_specialist": (
                str(args.proprio_guided_prone_specialist)
                if args.proprio_guided_prone_specialist is not None else None
            ),
            "proprio_guided_prone_gate": (
                str(args.proprio_guided_prone_gate)
                if args.proprio_guided_prone_gate is not None else None
            ),
            "guided_prone_threshold": args.guided_prone_threshold if guided_prone_gate is not None else None,
            "guided_prone_consecutive": args.guided_prone_consecutive if guided_prone_gate is not None else None,
            "guided_prone_min_steps": args.guided_prone_min_steps if guided_prone_gate is not None else None,
            "guided_prone_max_steps": args.guided_prone_max_steps if guided_prone_gate is not None else None,
            "iteration": state["iteration"],
            "sample_action": args.sample_action,
            "mean_after_escape": args.mean_after_escape,
            "supine_teacher_checkpoint": (
                str(args.supine_teacher_checkpoint)
                if args.supine_teacher_checkpoint is not None else None
            ),
            "postescape_teacher_checkpoint": (
                str(args.postescape_teacher_checkpoint)
                if args.postescape_teacher_checkpoint is not None else None
            ),
            "seed": args.seed,
            "guided_offset_m": args.guided_offset_m,
            "steps": args.steps,
            "per_scene": args.per_scene,
            "all_postescape": args.all_postescape,
            "all_directions": args.all_directions,
            "postescape_bank": (
                str(args.postescape_bank) if args.postescape_bank is not None else None
            ),
            "restore_postescape_bank_history": args.restore_postescape_bank_history,
            "postescape_live_obs_v1": env._a6_postescape_live_obs_v1,
            "prone_flexible_exit_v1": env._a6_prone_flexible_exit_v1,
            "shadow_prone_flexible_exit": args.shadow_prone_flexible_exit,
            "shadow_side_flexible_exit": args.shadow_side_flexible_exit,
            "initial_side_joint_safe_through_step": args.initial_side_joint_safe_through_step,
            "diagnostic_zero_plate_obs": args.diagnostic_zero_plate_obs,
            "postescape_motor_delay_override": args.postescape_motor_delay_override,
            "reset_policy_history_on_escape": phase_switch,
            "reset_supine_history_on_escape": args.reset_supine_history_on_escape,
            "arm_actuation_scale": cfg.actions["joint_pos"].guided_arm_scale,
            "prone_arm_actuation_scale": cfg.actions["joint_pos"].guided_prone_arm_scale,
            "supine_arm_actuation_scale": cfg.actions["joint_pos"].guided_supine_arm_scale,
            "hip_actuation_scale": cfg.actions["joint_pos"].guided_hip_scale,
            "no_privileged_actions": args.no_privileged_actions,
            "disable_lift_guard": args.disable_lift_guard,
            "disable_safe_scaling": args.disable_safe_scaling,
            "bake_safe_scaling_into_policy": args.bake_safe_scaling_into_policy,
            "bake_hip_scaling_all_scenes": args.bake_hip_scaling_all_scenes,
            "proprio_postescape_flat_teacher": (
                str(args.proprio_postescape_flat_teacher)
                if args.proprio_postescape_flat_teacher is not None else None
            ),
            "diagnostic_flat_expert_on_flat": args.diagnostic_flat_expert_on_flat,
            "diagnostic_flat_expert_after_prone_pulse": (
                args.diagnostic_flat_expert_after_prone_pulse
            ),
            "diagnostic_flat_expert_on_free_after_escape": (
                args.diagnostic_flat_expert_on_free_after_escape
            ),
            "proprio_postescape_a6_teacher": (
                str(args.proprio_postescape_a6_teacher)
                if args.proprio_postescape_a6_teacher is not None else None
            ),
            "proprio_preescape_a6_teacher": (
                str(args.proprio_preescape_a6_teacher)
                if args.proprio_preescape_a6_teacher is not None else None
            ),
            "proprio_diagnostic_reset_prone_history": args.proprio_diagnostic_reset_prone_history,
            "proprio_phase_detector": str(args.proprio_phase_detector) if args.proprio_phase_detector else None,
            "proprio_recovery_controller": str(args.proprio_recovery_controller) if args.proprio_recovery_controller else None,
            "diagnostic_controller_threshold": args.diagnostic_controller_threshold,
            "proprio_supine_flat_gate": str(args.proprio_supine_flat_gate) if args.proprio_supine_flat_gate else None,
            "proprio_early_flat_gate": str(args.proprio_early_flat_gate) if args.proprio_early_flat_gate else None,
            "early_flat_threshold": args.early_flat_threshold if early_flat_gate is not None else None,
            "early_flat_consecutive": args.early_flat_consecutive if early_flat_gate is not None else None,
            "flat_gate_threshold": args.flat_gate_threshold if supine_flat_gate is not None else None,
            "flat_gate_consecutive": args.flat_gate_consecutive if supine_flat_gate is not None else None,
            "flat_gate_min_steps": args.flat_gate_min_steps if supine_flat_gate is not None else None,
            "supine_neutral_through_step": args.supine_neutral_through_step,
            "side_neutral_through_step": args.side_neutral_through_step,
            "side_action_scale": args.side_action_scale,
            "side_lateral_gain": args.side_lateral_gain,
            "diagnostic_guided_nonside_lateral_gain": (
                args.diagnostic_guided_nonside_lateral_gain
            ),
            "diagnostic_guided_nonside_safe_scale": (
                args.diagnostic_guided_nonside_safe_scale
            ),
            "diagnostic_guided_side_assist_n": args.diagnostic_guided_side_assist_n,
            "supine_probe_v1": args.supine_probe_v1,
            "proprio_phase_threshold": args.proprio_phase_threshold if phase_detector is not None else None,
            "proprio_phase_consecutive": args.proprio_phase_consecutive if phase_detector is not None else None,
            "proprio_phase_full_observation_reset": args.proprio_phase_full_observation_reset,
            "proprio_phase_controller_reset": args.proprio_phase_controller_reset,
            "proprio_phase_min_steps": args.proprio_phase_min_steps if phase_detector is not None else None,
            "proprio_phase_pulse_steps": args.proprio_phase_pulse_steps,
            "proprio_phase_initial_prone_gate": args.proprio_phase_initial_prone_gate,
            "proprio_oracle_pulse_steps": args.proprio_oracle_pulse_steps,
            "proprio_oracle_actor_reset_only": args.proprio_oracle_actor_reset_only,
            "proprio_oracle_observation_reset_only": args.proprio_oracle_observation_reset_only,
            "proprio_oracle_controller_reset_only": args.proprio_oracle_controller_reset_only,
            "proprio_a6_teacher_reset_history": args.proprio_a6_teacher_reset_history,
            "assistance_n": 0.0,
            "knee_diagnostics": args.knee_diagnostics,
            "scenes": {},
        }
        labels = ("guided_prone_postescape",) if args.all_postescape else selected_labels
        for index, label in enumerate(labels):
            rows = (
                slice(0, args.per_scene) if args.all_postescape
                else slice(index * args.per_scene, (index + 1) * args.per_scene)
            )
            escaped = first_escape[rows].cpu().tolist()
            triggered = first_phase_trigger[rows].cpu().tolist()
            flat_triggered = first_flat_gate_trigger[rows].cpu().tolist()
            early_triggered = first_early_flat_trigger[rows].cpu().tolist()
            guided_scene_triggered = first_guided_prone_scene_trigger[rows].cpu().tolist()
            prone_flat_triggered = first_prone_flat_gate_trigger[rows].cpu().tolist()
            prone_free_triggered = first_prone_free_gate_trigger[rows].cpu().tolist()
            invalid = first_invalid[rows].cpu().tolist()
            stood = first_stand[rows].cpu().tolist()
            heights = final_height[rows].cpu().tolist()
            uprights = final_upright[rows].cpu().tolist()
            result["scenes"][label] = {
                "escaped": sum(value > 0 for value in escaped),
                "phase_triggered": sum(value > 0 for value in triggered),
                "flat_gate_triggered": sum(value > 0 for value in flat_triggered),
                "early_flat_triggered": sum(value > 0 for value in early_triggered),
                "first_early_flat_trigger_steps": early_triggered,
                "early_flat_trigger_contact": early_flat_trigger_contact[rows].cpu().tolist(),
                "early_flat_trigger_coverage_count": early_flat_trigger_coverage[rows].cpu().tolist(),
                "guided_prone_scene_triggered": sum(value > 0 for value in guided_scene_triggered),
                "prone_flat_gate_triggered": sum(value > 0 for value in prone_flat_triggered),
                "prone_flat_pulse_forced": int(forced_prone_flat_pulse[rows].sum()),
                "first_prone_flat_gate_trigger_steps": prone_flat_triggered,
                "prone_free_gate_triggered": sum(value > 0 for value in prone_free_triggered),
                "prone_free_pulse_forced": int(forced_prone_free_pulse[rows].sum()),
                "first_prone_free_gate_trigger_steps": prone_free_triggered,
                "first_guided_prone_scene_trigger_steps": guided_scene_triggered,
                "guided_prone_scene_trigger_contact": guided_prone_trigger_contact[rows].cpu().tolist(),
                "guided_prone_scene_trigger_coverage_count": guided_prone_trigger_coverage[rows].cpu().tolist(),
                "initial_supine_armed": int(initial_supine_ever[rows].sum()),
                "flat_gate_trigger_before_escape": sum(
                    trigger_step > 0 and (escape_step == 0 or trigger_step < escape_step)
                    for trigger_step, escape_step in zip(flat_triggered, escaped, strict=True)
                ) if label.startswith(("guided_", "free_")) else 0,
                "first_flat_gate_trigger_steps": flat_triggered,
                "initial_gravity_b": [
                    [round(float(axis), 4) for axis in vec]
                    for vec in initial_gravity_b[rows].cpu().tolist()
                ],
                "flat_gate_trigger_contact": flat_gate_trigger_contact[rows].cpu().tolist(),
                "flat_gate_trigger_coverage_count": flat_gate_trigger_coverage[rows].cpu().tolist(),
                "flat_gate_trigger_core_clearance_m": [
                    round(value, 4) for value in flat_gate_trigger_core_clearance[rows].cpu().tolist()
                ],
                "flat_gate_trigger_force_n": [
                    round(value, 1) for value in flat_gate_trigger_force[rows].cpu().tolist()
                ],
                "flat_gate_trigger_lift_m": [
                    round(value, 4) for value in flat_gate_trigger_lift[rows].cpu().tolist()
                ],
                "flat_gate_trigger_torque_peak_scaled": [
                    round(value, 4) for value in flat_gate_trigger_torque_peak[rows].cpu().tolist()
                ],
                "flat_gate_trigger_torque_rms_scaled": [
                    round(value, 4) for value in flat_gate_trigger_torque_rms[rows].cpu().tolist()
                ],
                "initial_prone_armed": int(initial_prone[rows].sum()),
                "phase_trigger_before_escape": sum(
                    trigger_step > 0 and (escape_step == 0 or trigger_step < escape_step)
                    for trigger_step, escape_step in zip(triggered, escaped, strict=True)
                ),
                "phase_trigger_after_escape_20": sum(
                    trigger_step > 0 and escape_step > 0
                    and 0 <= trigger_step - escape_step <= 20
                    for trigger_step, escape_step in zip(triggered, escaped, strict=True)
                ),
                "first_phase_trigger_steps": triggered,
                "invalid": sum(value > 0 for value in invalid),
                "invalid_by_force": int(invalid_by_force[rows].sum()),
                "invalid_by_lift": int(invalid_by_lift[rows].sum()),
                "invalid_other": int(invalid_other[rows].sum()),
                "stood_at_end": sum(
                    height > 0.6 and upright > 0.8
                    for height, upright in zip(heights, uprights, strict=True)
                ),
                "escaped_then_stood_at_end": sum(
                    escaped_step > 0 and height > 0.6 and upright > 0.8
                    for escaped_step, height, upright in zip(
                        escaped, heights, uprights, strict=True
                    )
                ),
                "first_escape_steps": escaped,
                "guided_side_gate_triggers": int((first_guided_side_scene_trigger[rows] > 0).sum()),
                "first_guided_side_gate_steps": first_guided_side_scene_trigger[rows].cpu().tolist(),
                "max_preescape_free_plate_shift_m": [
                    round(value, 4) for value in
                    max_preescape_free_plate_shift[rows].cpu().tolist()
                ],
                "free_plate_shift_at_escape_m": [
                    round(value, 4) for value in
                    free_plate_shift_at_escape[rows].cpu().tolist()
                ],
                "first_shadow_escape_steps": (
                    first_shadow_escape[rows].cpu().tolist()
                    if args.shadow_prone_flexible_exit or args.shadow_side_flexible_exit else None
                ),
                "shadow_core_clearance_at_escape_m": (
                    shadow_core_at_escape[rows].cpu().tolist()
                    if args.shadow_prone_flexible_exit or args.shadow_side_flexible_exit else None
                ),
                "shadow_trailing_overlap_at_escape_m": (
                    shadow_tip_at_escape[rows].cpu().tolist()
                    if args.shadow_prone_flexible_exit or args.shadow_side_flexible_exit else None
                ),
                "shadow_plate_lift_at_escape_m": (
                    shadow_lift_at_escape[rows].cpu().tolist()
                    if args.shadow_prone_flexible_exit or args.shadow_side_flexible_exit else None
                ),
                "first_invalid_steps": invalid,
                "first_stand_steps": stood,
                "mean_action_delta_rms": [
                    round(value, 5) for value in
                    (action_delta_sum[rows] / action_delta_count[rows].clamp_min(1))
                    .cpu().tolist()
                ],
                "max_action_delta_rms": [
                    round(value, 5) for value in max_action_delta[rows].cpu().tolist()
                ],
                "side_gate_action_delta_rms": [
                    round(value, 5) for value in
                    side_gate_action_delta[rows].cpu().tolist()
                ],
                "flat_gate_action_delta_rms": [
                    round(value, 5) for value in
                    flat_gate_action_delta[rows].cpu().tolist()
                ],
                "phase_pulse_action_delta_rms": [
                    round(value, 5) for value in
                    phase_pulse_action_delta[rows].cpu().tolist()
                ],
                "height_at_escape_m": [
                    round(value, 4) for value in height_at_escape[rows].cpu().tolist()
                ],
                "height_10_steps_after_escape_m": [
                    round(value, 4) for value in height_after_10[rows].cpu().tolist()
                ],
                "height_20_steps_after_escape_m": [
                    round(value, 4) for value in height_after_20[rows].cpu().tolist()
                ],
                "final_height_m": [round(value, 4) for value in heights],
                "final_upright": [round(value, 4) for value in uprights],
                "max_low_root_travel_m": [
                    round(value, 4) for value in max_low_root_travel[rows].cpu().tolist()
                ],
                "max_postescape_low_travel_m": [
                    round(value, 4)
                    for value in max_postescape_low_travel[rows].cpu().tolist()
                ],
                "max_xy_clearance_m": [
                    round(value, 4) for value in max_clearance[rows].cpu().tolist()
                ],
                "max_guided_force_n": [
                    round(value, 1) for value in max_guided_force[rows].cpu().tolist()
                ],
                "max_guided_lift_m": [
                    round(value, 4) for value in max_guided_lift[rows].cpu().tolist()
                ],
            }
            if knee_trace is not None and label.startswith("guided_"):
                traces = knee_trace[:, rows].cpu().numpy().T
                tails = [values[np.isfinite(values)][-100:] for values in traces]
                active_escape = [
                    values[30:][np.isfinite(values[30:])] for values in traces
                ]
                median_deviation = [
                    float(np.median(values)) if len(values) else None for values in tails
                ]
                above_20deg = [
                    float(np.mean(values > math.radians(20))) if len(values) else None
                    for values in tails
                ]
                valid_medians = [value for value in median_deviation if value is not None]
                valid_fractions = [value for value in above_20deg if value is not None]
                active_fractions = [
                    float(np.mean(values > math.radians(20)))
                    for values in active_escape if len(values)
                ]
                active_p90 = [
                    float(np.percentile(values, 90))
                    for values in active_escape if len(values)
                ]
                result["scenes"][label].update({
                    "preescape_last100_knee_deviation_median_rad": [
                        round(value, 4) if value is not None else None
                        for value in median_deviation
                    ],
                    "preescape_last100_above20_fraction": [
                        round(value, 4) if value is not None else None
                        for value in above_20deg
                    ],
                    "preescape_knee_deviation_p50_rad": round(
                        float(np.median(valid_medians)), 4
                    ) if valid_medians else None,
                    "preescape_above20_mean_fraction": round(
                        float(np.mean(valid_fractions)), 4
                    ) if valid_fractions else None,
                    "preescape_after30_above20_mean_fraction": round(
                        float(np.mean(active_fractions)), 4
                    ) if active_fractions else None,
                    "preescape_after30_knee_deviation_p90_rad": round(
                        float(np.median(active_p90)), 4
                    ) if active_p90 else None,
                })
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(json.dumps({key: {k: v for k, v in scene.items() if not isinstance(v, list)}
                          for key, scene in result["scenes"].items()}, indent=2))
    finally:
        env.close()


if __name__ == "__main__":
    main()
