"""A6 adaptation configuration for the G1 recovery AMP policy.

This module keeps the AMP network contract (observation/action dimensions,
rewards, and discriminator) checkpoint-compatible while adopting A6's finer
MuJoCo integration step, frozen reset bank, and reset-entry action protection.
Later stages will add constrained scenes and shared task rewards here without
changing the original flat-task baseline.
"""

from __future__ import annotations

import math
from typing import Literal

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as common_mdp
from mjlab.envs.mdp import dr
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.metrics_manager import MetricsTermCfg
from mjlab.managers.observation_manager import (
    ObservationGroupCfg,
    ObservationTermCfg,
)
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg

from g1recovery_amp.a6_reset_assets import (
    A6_MULTITERRAIN_TRAIN_BANK,
    A6_NATURAL_CURRICULUM_TRAIN_BANK,
)
from g1recovery_amp.g1_model import G1_JOINT_NAMES, G1_RECOVERY_ACTION_SCALE
from g1recovery_amp.tasks.recovery.a6_scene import (
    plate_contact_sensor_cfgs,
    plate_entity_cfgs,
)
from g1recovery_amp.tasks.recovery.env_cfg import g1_recovery_dev_env_cfg
from g1recovery_amp.tasks.recovery.mdp import (
    A6_CATASTROPHIC_PLATE_FORCE,
    A6_CLEAR_HOLD_STEPS,
    A6_MAX_PLATE_FORCE,
    A6_MIN_PLANAR_CLEARANCE,
    A6_TASK_NAMES,
    A6_TASK_WEIGHTS,
    A6EntryProtectedJointPositionActionCfg,
    RecoveryMotionCommandCfg,
    a6_action_acceleration,
    a6_action_rate,
    a6_amp_reward_scale,
    a6_foot_motion,
    a6_invalid_plate,
    a6_joint_stall,
    a6_physics_cost,
    a6_scaled_cost,
    a6_task_component,
    plate_clearance,
    plate_completion,
    plate_force,
    plate_geometry_progress,
    plate_no_progress,
    plate_separation,
    prone_lateral_progress,
    reset_a6_dynamics,
    reset_from_a6_three_scene_curriculum,
    sample_a6_cost_substep,
    trapped_shoulders_headward,
)

A6_PHYSICS_TIMESTEP = 0.002
A6_DECIMATION = 10
A6_CONTROL_DT = A6_PHYSICS_TIMESTEP * A6_DECIMATION
A6_ACTION_WARMUP_STEPS = 10
A6_RAW_ACTION_CLIP = 10.0
A6_MAX_COMMAND_DELAY_STEPS = 5
A6_OBSTRUCTED_TASK_SCALE = 0.05
A6_HISTORICAL_SCENE_WEIGHTS = (0.50, 0.25, 0.25)
A6_CONSTRAINED_70_30_SCENE_WEIGHTS = (0.0, 0.70, 0.30)
A6_BALANCED_30_50_20_SCENE_WEIGHTS = (0.30, 0.50, 0.20)
A6_SYMMETRIC_OBSTRUCTED_SCALE = 0.05
A6_CONSTRAINT_AWARE_AMP_SCALE = 0.05
A6_CONSTRAINT_AWARE_001_AMP_SCALE = 0.01
A6_FIXED_PLATE_MASS = 6.0
A6_PLATE_SEPARATION_WEIGHT = 0.10
A6_PLATE_NO_PROGRESS_WEIGHT = -0.15
A6_STRICT_ESCAPE_MIN_PLANAR_CLEARANCE = 0.040
A6_STRICT_ESCAPE_CLEAR_HOLD_S = 0.50
A6_TRAPPED_SHOULDERS_HEADWARD_WEIGHT = 0.15
A6_EXPLORATION_COST_OBSTRUCTED_SCALE = 0.25
A6_FORCE_LIMIT_HOLD_S = 0.06
A6_INVALID_PLATE_TERMINATION_WEIGHT = -100.0
A6_PRONE_LATERAL_PROGRESS_WEIGHT = 0.20
A6GuidanceMode = Literal["minus", "plus"]


def apply_a6_timing(cfg: ManagerBasedRlEnvCfg) -> None:
    """Adopt A6 physics substeps without changing the policy control period."""

    original_control_dt = cfg.sim.mujoco.timestep * cfg.decimation
    if not math.isclose(original_control_dt, A6_CONTROL_DT, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError(
            f"A6 timing would change control period: {original_control_dt} -> "
            f"{A6_CONTROL_DT} s"
        )
    cfg.sim.mujoco.timestep = A6_PHYSICS_TIMESTEP
    cfg.decimation = A6_DECIMATION


def g1_recovery_a6_env_cfg(
    *,
    play: bool = False,
    guidance: A6GuidanceMode = "plus",
    scene_weights: tuple[float, float, float] = A6_HISTORICAL_SCENE_WEIGHTS,
    symmetric_obstructed_scale: float | None = None,
    obstructed_style_reward_scale: float | None = None,
    training_plate_mass_override: float | None = None,
    plate_no_progress_weight: float | None = None,
    escape_min_planar_clearance: float = A6_MIN_PLANAR_CLEARANCE,
    escape_clear_hold_s: float = A6_CLEAR_HOLD_STEPS * A6_CONTROL_DT,
    trapped_shoulders_headward_weight: float | None = None,
    exploration_cost_obstructed_scale: float | None = None,
    plate_force_limit: float = A6_MAX_PLATE_FORCE,
    plate_force_hold_s: float = A6_CONTROL_DT,
    catastrophic_plate_force: float | None = None,
    invalid_plate_termination_weight: float | None = None,
    prone_lateral_progress_weight: float | None = None,
) -> ManagerBasedRlEnvCfg:
    """Build matched G-/G+ AMP-A6 environments with identical policy I/O."""

    if guidance not in ("minus", "plus"):
        raise ValueError(f"unknown A6 guidance mode: {guidance!r}")
    if symmetric_obstructed_scale is not None and not (
        0.0 <= symmetric_obstructed_scale <= 1.0
    ):
        raise ValueError("symmetric obstructed scale must be in [0, 1]")
    if obstructed_style_reward_scale is not None and not (
        0.0 <= obstructed_style_reward_scale <= 1.0
    ):
        raise ValueError("obstructed AMP style reward scale must be in [0, 1]")
    if training_plate_mass_override is not None and training_plate_mass_override <= 0:
        raise ValueError("training plate mass override must be positive")
    if plate_no_progress_weight is not None:
        if guidance != "plus":
            raise ValueError("plate no-progress penalty requires plus guidance")
        if plate_no_progress_weight >= 0.0:
            raise ValueError("plate no-progress weight must be negative")
    if escape_min_planar_clearance < 0.0:
        raise ValueError("escape planar clearance must be nonnegative")
    if escape_clear_hold_s <= 0.0:
        raise ValueError("escape clear hold time must be positive")
    escape_clear_hold_steps = round(escape_clear_hold_s / A6_CONTROL_DT)
    if not math.isclose(
        escape_clear_hold_steps * A6_CONTROL_DT,
        escape_clear_hold_s,
        rel_tol=0.0,
        abs_tol=1.0e-9,
    ):
        raise ValueError("escape clear hold time must align with the control period")
    if trapped_shoulders_headward_weight is not None:
        if guidance != "plus":
            raise ValueError("trapped shoulder reward requires plus guidance")
        if trapped_shoulders_headward_weight <= 0.0:
            raise ValueError("trapped shoulder reward weight must be positive")
    if exploration_cost_obstructed_scale is not None and not (
        0.0 <= exploration_cost_obstructed_scale <= 1.0
    ):
        raise ValueError("exploration cost obstructed scale must be in [0, 1]")
    if plate_force_limit <= 0.0:
        raise ValueError("plate force limit must be positive")
    if plate_force_hold_s <= 0.0:
        raise ValueError("plate force hold time must be positive")
    plate_force_hold_steps = round(plate_force_hold_s / A6_CONTROL_DT)
    if not math.isclose(
        plate_force_hold_steps * A6_CONTROL_DT,
        plate_force_hold_s,
        rel_tol=0.0,
        abs_tol=1.0e-9,
    ):
        raise ValueError("plate force hold time must align with the control period")
    if (
        catastrophic_plate_force is not None
        and catastrophic_plate_force <= plate_force_limit
    ):
        raise ValueError("catastrophic plate force must exceed the ordinary limit")
    if (
        invalid_plate_termination_weight is not None
        and invalid_plate_termination_weight >= 0.0
    ):
        raise ValueError("invalid plate termination weight must be negative")
    if prone_lateral_progress_weight is not None:
        if guidance != "plus":
            raise ValueError("prone lateral progress reward requires plus guidance")
        if prone_lateral_progress_weight <= 0.0:
            raise ValueError("prone lateral progress weight must be positive")

    cfg = g1_recovery_dev_env_cfg(play=play)
    apply_a6_timing(cfg)
    # Keep the AMP policy's 29-D action meaning and 0.25 scale, but adopt the
    # A6 deployment contract's safe hand-over from an arbitrary bank pose.
    cfg.actions["joint_pos"] = A6EntryProtectedJointPositionActionCfg(
        entity_name="robot",
        actuator_names=(".*",),
        scale=G1_RECOVERY_ACTION_SCALE,
        use_default_offset=True,
        warmup_steps=A6_ACTION_WARMUP_STEPS,
        raw_action_clip=A6_RAW_ACTION_CLIP,
        max_delay_steps=A6_MAX_COMMAND_DELAY_STEPS,
    )
    motion_cfg = cfg.commands["motion"]
    if not isinstance(motion_cfg, RecoveryMotionCommandCfg):
        raise TypeError("expected the AMP recovery motion command")
    # The motion command remains the source of AMP demonstrations and reference
    # frames, but the physical robot now starts from the frozen A6 bank.
    motion_cfg.initialize_robot_from_motion = False
    # Frozen A6 has no upward get-up assistance.  Keep the motion command for
    # AMP demonstrations, but remove the separate 200 N assistance chain.
    cfg.commands.pop("get_up_assist_force")
    cfg.events.pop("get_up_assist_force")
    cfg.curriculum.pop("get_up_assist_force_level", None)

    # Replace the original AMP-only dynamics with the frozen A6 adaptation
    # protocol.  These changes stay local to the A6 task IDs.
    for event_name in ("add_base_mass", "actuator_gains", "mixed_disturbance"):
        cfg.events.pop(event_name, None)
    if play:
        cfg.events.pop("physics_material", None)
    else:
        cfg.events["physics_material"] = EventTermCfg(
            func=dr.geom_friction,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot", geom_names=r"^(left|right)_foot[1-7]_collision$"
                ),
                "operation": "abs",
                "ranges": (0.3, 1.2),
                "shared_random": True,
            },
        )
        cfg.events["encoder_bias"] = EventTermCfg(
            func=dr.encoder_bias,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
                "bias_range": (-0.015, 0.015),
            },
        )
        cfg.events["base_com"] = EventTermCfg(
            func=dr.body_com_offset,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names="torso_link"),
                "operation": "add",
                "ranges": {
                    0: (-0.025, 0.025),
                    1: (-0.025, 0.025),
                    2: (-0.03, 0.03),
                },
            },
        )
        cfg.events["push_robot"] = EventTermCfg(
            func=common_mdp.push_by_setting_velocity,
            mode="interval",
            interval_range_s=(1.0, 3.0),
            is_global_time=False,
            params={
                "velocity_range": {
                    "x": (-0.5, 0.5),
                    "y": (-0.5, 0.5),
                    "z": (-0.4, 0.4),
                    "roll": (-0.52, 0.52),
                    "pitch": (-0.52, 0.52),
                    "yaw": (-0.78, 0.78),
                }
            },
        )
    cfg.events["a6_dynamics"] = EventTermCfg(
        func=reset_a6_dynamics,
        mode="reset",
        params={"enabled": not play},
    )
    cfg.scene.entities.update(plate_entity_cfgs())
    cfg.scene.sensors += plate_contact_sensor_cfgs()
    cfg.events["a6_three_scene_curriculum_reset"] = EventTermCfg(
        func=reset_from_a6_three_scene_curriculum,
        mode="reset",
        params={
            "low_bank_path": str(A6_MULTITERRAIN_TRAIN_BANK),
            "natural_bank_path": str(A6_NATURAL_CURRICULUM_TRAIN_BANK),
            "asset_cfg": SceneEntityCfg(
                "robot", joint_names=G1_JOINT_NAMES, preserve_order=True
            ),
            "plate_mass_override": (
                A6_FIXED_PLATE_MASS if play else training_plate_mass_override
            ),
            "scene_weights": scene_weights,
            "escape_min_planar_clearance": escape_min_planar_clearance,
            "escape_clear_hold_steps": escape_clear_hold_steps,
        },
    )
    cfg.sim.nconmax = max(cfg.sim.nconmax, 256)
    cfg.sim.njmax = max(cfg.sim.njmax, 4000)
    cfg.terminations["invalid_plate"] = TerminationTermCfg(
        func=a6_invalid_plate,
        params={
            "force_limit": plate_force_limit,
            "force_hold_steps": plate_force_hold_steps,
            "catastrophic_force": catastrophic_plate_force,
        },
    )
    cfg.metrics["a6_joint_acc_substep_raw"] = MetricsTermCfg(
        func=sample_a6_cost_substep,
        per_substep=True,
        reduce="mean",
    )
    # Replace overlapping baseline rewards and penalties with the A6
    # definitions.  The policy I/O stays unchanged; physics costs average all
    # ten substeps.
    for name in (
        "joint_acc_l2",
        "action_rate_l2",
        "joint_torques_l2",
        "joint_pos_limits",
        "ang_vel_xy",
        "lin_vel_xy",
        "target_orientation",
        "low_height_progress",
        "target_base_height",
        "target_joint_deviation_l2",
    ):
        cfg.rewards.pop(name)
    if symmetric_obstructed_scale is None:
        obstructed_scale = A6_OBSTRUCTED_TASK_SCALE if guidance == "plus" else 1.0
    else:
        obstructed_scale = symmetric_obstructed_scale

    def cost_term(
        *,
        source: str,
        weight: float,
        index: int | None = None,
        scale_for_exploration: bool = False,
    ) -> RewardTermCfg:
        cost_obstructed_scale = symmetric_obstructed_scale
        if (
            cost_obstructed_scale is None
            and scale_for_exploration
            and exploration_cost_obstructed_scale is not None
        ):
            cost_obstructed_scale = exploration_cost_obstructed_scale
        if cost_obstructed_scale is None:
            direct_functions = {
                "action_rate": a6_action_rate,
                "action_acceleration": a6_action_acceleration,
                "joint_limits": common_mdp.joint_pos_limits,
                "physics": a6_physics_cost,
                "foot_motion": a6_foot_motion,
                "joint_stall": a6_joint_stall,
                "plate_force": plate_force,
            }
            params = {"index": index} if source == "physics" else {}
            return RewardTermCfg(
                func=direct_functions[source],
                weight=weight,
                params=params,
            )
        params = {
            "cost_name": source,
            "obstructed_scale": cost_obstructed_scale,
        }
        if index is not None:
            params["index"] = index
        return RewardTermCfg(func=a6_scaled_cost, weight=weight, params=params)

    cfg.rewards.update(
        {
            **{
                f"a6_{name}": RewardTermCfg(
                    func=a6_task_component,
                    weight=weight,
                    params={
                        "index": index,
                        "obstructed_scale": obstructed_scale,
                    },
                )
                for index, (name, weight) in enumerate(
                    zip(A6_TASK_NAMES, A6_TASK_WEIGHTS, strict=True)
                )
            },
            "a6_action_rate": cost_term(
                source="action_rate", weight=-0.0015, scale_for_exploration=True
            ),
            "a6_action_acceleration": cost_term(
                source="action_acceleration",
                weight=-0.0012,
                scale_for_exploration=True,
            ),
            "a6_joint_limits": cost_term(source="joint_limits", weight=-0.1),
            "a6_joint_acceleration": cost_term(
                source="physics", weight=-5.0e-8, index=0
            ),
            "a6_torque": cost_term(source="physics", weight=-1.0e-6, index=1),
            "a6_joint_overspeed": cost_term(
                source="physics",
                weight=-0.02,
                index=2,
                scale_for_exploration=True,
            ),
            "a6_joint_overpower": cost_term(source="physics", weight=-2.0e-6, index=3),
            "a6_head_overspeed": cost_term(
                source="physics",
                weight=-1.0,
                index=4,
                scale_for_exploration=True,
            ),
            "a6_sustained_effort": cost_term(
                source="physics",
                weight=-0.05,
                index=5,
                scale_for_exploration=True,
            ),
            "a6_foot_motion": cost_term(
                source="foot_motion", weight=-0.03, scale_for_exploration=True
            ),
            "a6_joint_stall": cost_term(
                source="joint_stall", weight=-0.20, scale_for_exploration=True
            ),
            "plate_completion": RewardTermCfg(func=plate_completion, weight=0.60),
            "plate_force": cost_term(source="plate_force", weight=-0.03),
        }
    )
    if invalid_plate_termination_weight is not None:
        # Reward weights are multiplied by the 0.02 s control period.  A
        # weight of -100 therefore contributes approximately -2 once, on the
        # control step where the invalid-plate termination fires.
        cfg.rewards["invalid_plate_termination"] = RewardTermCfg(
            func=common_mdp.is_terminated,
            weight=invalid_plate_termination_weight,
        )
    if guidance == "plus":
        cfg.rewards.update(
            {
                "plate_geometry_progress": RewardTermCfg(
                    func=plate_geometry_progress, weight=0.45
                ),
                "plate_clearance": RewardTermCfg(func=plate_clearance, weight=0.08),
                "plate_separation": RewardTermCfg(
                    func=plate_separation,
                    weight=A6_PLATE_SEPARATION_WEIGHT,
                ),
            }
        )
        if plate_no_progress_weight is not None:
            cfg.rewards["plate_no_progress"] = RewardTermCfg(
                func=plate_no_progress,
                weight=plate_no_progress_weight,
            )
        if prone_lateral_progress_weight is not None:
            cfg.rewards["prone_lateral_progress"] = RewardTermCfg(
                func=prone_lateral_progress,
                weight=prone_lateral_progress_weight,
            )
        if trapped_shoulders_headward_weight is not None:
            cfg.rewards["trapped_shoulders_headward"] = RewardTermCfg(
                func=trapped_shoulders_headward,
                weight=trapped_shoulders_headward_weight,
            )
    amp_obstructed_scale = (
        symmetric_obstructed_scale
        if obstructed_style_reward_scale is None
        else obstructed_style_reward_scale
    )
    if amp_obstructed_scale is not None:
        # This group is consumed only by AmpPPO.  It is intentionally absent
        # from actor/critic obs_groups, so checkpoint input dimensions remain
        # unchanged.
        cfg.observations["amp_reward_scale"] = ObservationGroupCfg(
            terms={
                "scale": ObservationTermCfg(
                    func=a6_amp_reward_scale,
                    params={"obstructed_scale": amp_obstructed_scale},
                )
            },
            enable_corruption=False,
            nan_policy="error",
        )
    return cfg


__all__ = [
    "A6_ACTION_WARMUP_STEPS",
    "A6_BALANCED_30_50_20_SCENE_WEIGHTS",
    "A6_CATASTROPHIC_PLATE_FORCE",
    "A6_CONSTRAINED_70_30_SCENE_WEIGHTS",
    "A6_CONSTRAINT_AWARE_001_AMP_SCALE",
    "A6_CONSTRAINT_AWARE_AMP_SCALE",
    "A6_CONTROL_DT",
    "A6_DECIMATION",
    "A6_EXPLORATION_COST_OBSTRUCTED_SCALE",
    "A6_FIXED_PLATE_MASS",
    "A6_FORCE_LIMIT_HOLD_S",
    "A6_HISTORICAL_SCENE_WEIGHTS",
    "A6_INVALID_PLATE_TERMINATION_WEIGHT",
    "A6_MAX_COMMAND_DELAY_STEPS",
    "A6_OBSTRUCTED_TASK_SCALE",
    "A6_PHYSICS_TIMESTEP",
    "A6_PLATE_NO_PROGRESS_WEIGHT",
    "A6_PLATE_SEPARATION_WEIGHT",
    "A6_PRONE_LATERAL_PROGRESS_WEIGHT",
    "A6_RAW_ACTION_CLIP",
    "A6_STRICT_ESCAPE_CLEAR_HOLD_S",
    "A6_STRICT_ESCAPE_MIN_PLANAR_CLEARANCE",
    "A6_SYMMETRIC_OBSTRUCTED_SCALE",
    "A6_TRAPPED_SHOULDERS_HEADWARD_WEIGHT",
    "A6GuidanceMode",
    "apply_a6_timing",
    "g1_recovery_a6_env_cfg",
]
