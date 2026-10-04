"""Register the G1 recovery development task."""

from mjlab.tasks.registry import register_mjlab_task

from g1recovery_amp.tasks.recovery.a6_env_cfg import (
    A6_BALANCED_30_50_20_SCENE_WEIGHTS,
    A6_CATASTROPHIC_PLATE_FORCE,
    A6_CONSTRAINED_70_30_SCENE_WEIGHTS,
    A6_CONSTRAINT_AWARE_001_AMP_SCALE,
    A6_CONSTRAINT_AWARE_AMP_SCALE,
    A6_EXPLORATION_COST_OBSTRUCTED_SCALE,
    A6_FIXED_PLATE_MASS,
    A6_FORCE_LIMIT_HOLD_S,
    A6_INVALID_PLATE_TERMINATION_WEIGHT,
    A6_LOW_TORQUE_TRIAL_SCALE,
    A6_PLATE_NO_PROGRESS_WEIGHT,
    A6_PRONE_LATERAL_PROGRESS_WEIGHT,
    A6_STRICT_ESCAPE_CLEAR_HOLD_S,
    A6_STRICT_ESCAPE_MIN_PLANAR_CLEARANCE,
    A6_SYMMETRIC_OBSTRUCTED_SCALE,
    A6_TRAPPED_SHOULDERS_HEADWARD_WEIGHT,
    g1_recovery_a6_env_cfg,
)
from g1recovery_amp.tasks.recovery.env_cfg import g1_recovery_dev_env_cfg
from g1recovery_amp.tasks.recovery.rl.rl_cfg import (
    g1_recovery_a6_constraint_aware_001_ppo_runner_cfg,
    g1_recovery_a6_constraint_aware_005_ppo_runner_cfg,
    g1_recovery_a6_ppo_runner_cfg,
    g1_recovery_a6_symmetric_005_ppo_runner_cfg,
    g1_recovery_ppo_runner_cfg,
)
from g1recovery_amp.tasks.recovery.rl.runner import RecoveryOnPolicyRunner

DEV_TASK_ID = "Mjlab-Recovery-AMP-Flat-Unitree-G1-Dev"
A6_TASK_ID = "Mjlab-Recovery-AMP-A6-Unitree-G1-Dev"
A6_G_MINUS_TASK_ID = "Mjlab-Recovery-AMP-A6-GMinus-Unitree-G1-Dev"
A6_G_PLUS_TASK_ID = "Mjlab-Recovery-AMP-A6-GPlus-Unitree-G1-Dev"
A6_G_MINUS_70_30_TASK_ID = (
    "Mjlab-Recovery-AMP-A6-GMinus-Constrained70-30-Unitree-G1-Dev"
)
A6_G_PLUS_70_30_TASK_ID = "Mjlab-Recovery-AMP-A6-GPlus-Constrained70-30-Unitree-G1-Dev"
A6_G_MINUS_SYMMETRIC_005_TASK_ID = (
    "Mjlab-Recovery-AMP-A6-GMinus-Symmetric005-30-50-20-Unitree-G1-Dev"
)
A6_G_PLUS_SYMMETRIC_005_TASK_ID = (
    "Mjlab-Recovery-AMP-A6-GPlus-Symmetric005-30-50-20-Unitree-G1-Dev"
)
A6_G_PLUS_CONSTRAINT_AWARE_005_TASK_ID = (
    "Mjlab-Recovery-AMP-A6-GPlus-ConstraintAware005-30-50-20-Unitree-G1-Dev"
)
A6_G_PLUS_CONSTRAINT_AWARE_005_FIXED6_TASK_ID = (
    "Mjlab-Recovery-AMP-A6-GPlus-ConstraintAware005-Fixed6-30-50-20-Unitree-G1-Dev"
)
A6_G_PLUS_CONSTRAINT_AWARE_001_FIXED6_TASK_ID = (
    "Mjlab-Recovery-AMP-A6-GPlus-ConstraintAware001-Fixed6-30-50-20-Unitree-G1-Dev"
)
A6_G_PLUS_CONSTRAINT_AWARE_005_NO_PROGRESS_FIXED6_TASK_ID = (
    "Mjlab-Recovery-AMP-A6-GPlus-ConstraintAware005-NoProgress015-"
    "Fixed6-30-50-20-Unitree-G1-Dev"
)
A6_G_PLUS_STRICT_ESCAPE_HEADWARD_FIXED6_TASK_ID = (
    "Mjlab-Recovery-AMP-A6-GPlus-CA005-NP015-Escape040x050-Head015-"
    "Fixed6-30-50-20-Unitree-G1-Dev"
)
A6_G_PLUS_STRICT_ESCAPE_EXPLORATION_025_FIXED6_TASK_ID = (
    "Mjlab-Recovery-AMP-A6-GPlus-CA005-NP015-Escape040x050-"
    "ExploreCost025-Fixed6-30-50-20-Unitree-G1-Dev"
)
A6_G_PLUS_FORCE_HOLD_INVALID2_FIXED6_TASK_ID = (
    "Mjlab-Recovery-AMP-A6-GPlus-CA005-NP015-Escape040x050-"
    "ExploreCost025-ForceHold060-Emergency2500-Invalid2-"
    "Fixed6-30-50-20-Unitree-G1-Dev"
)
A6_G_PLUS_PRONE_LATERAL_020_FIXED6_TASK_ID = (
    "Mjlab-Recovery-AMP-A6-GPlus-CA005-NP015-Escape040x050-"
    "ExploreCost025-ForceHold060-Emergency2500-Invalid2-"
    "ProneLateral020-Fixed6-30-50-20-Unitree-G1-Dev"
)
LOW_TORQUE_CAP_ONLY_90_TASK_ID = (
    "Mjlab-Recovery-AMP-LowTorque-CapOnly-90-Unitree-G1-Dev"
)

register_mjlab_task(
    task_id=DEV_TASK_ID,
    env_cfg=g1_recovery_dev_env_cfg(),
    play_env_cfg=g1_recovery_dev_env_cfg(play=True),
    rl_cfg=g1_recovery_ppo_runner_cfg(),
    # The training CLI recognizes the reused motion command as tracking-shaped
    # and passes ``registry_name``.  This adapter accepts that argument while
    # retaining the ordinary mjlab runner; AmpPPO owns recovery-specific logic.
    runner_cls=RecoveryOnPolicyRunner,
)

register_mjlab_task(
    task_id=A6_G_PLUS_STRICT_ESCAPE_HEADWARD_FIXED6_TASK_ID,
    env_cfg=g1_recovery_a6_env_cfg(
        guidance="plus",
        scene_weights=A6_BALANCED_30_50_20_SCENE_WEIGHTS,
        obstructed_style_reward_scale=A6_CONSTRAINT_AWARE_AMP_SCALE,
        training_plate_mass_override=A6_FIXED_PLATE_MASS,
        plate_no_progress_weight=A6_PLATE_NO_PROGRESS_WEIGHT,
        escape_min_planar_clearance=A6_STRICT_ESCAPE_MIN_PLANAR_CLEARANCE,
        escape_clear_hold_s=A6_STRICT_ESCAPE_CLEAR_HOLD_S,
        trapped_shoulders_headward_weight=A6_TRAPPED_SHOULDERS_HEADWARD_WEIGHT,
    ),
    play_env_cfg=g1_recovery_a6_env_cfg(
        play=True,
        guidance="plus",
        scene_weights=A6_BALANCED_30_50_20_SCENE_WEIGHTS,
        obstructed_style_reward_scale=A6_CONSTRAINT_AWARE_AMP_SCALE,
        training_plate_mass_override=A6_FIXED_PLATE_MASS,
        plate_no_progress_weight=A6_PLATE_NO_PROGRESS_WEIGHT,
        escape_min_planar_clearance=A6_STRICT_ESCAPE_MIN_PLANAR_CLEARANCE,
        escape_clear_hold_s=A6_STRICT_ESCAPE_CLEAR_HOLD_S,
        trapped_shoulders_headward_weight=A6_TRAPPED_SHOULDERS_HEADWARD_WEIGHT,
    ),
    rl_cfg=g1_recovery_a6_constraint_aware_005_ppo_runner_cfg(),
    runner_cls=RecoveryOnPolicyRunner,
)

register_mjlab_task(
    task_id=A6_G_PLUS_STRICT_ESCAPE_EXPLORATION_025_FIXED6_TASK_ID,
    env_cfg=g1_recovery_a6_env_cfg(
        guidance="plus",
        scene_weights=A6_BALANCED_30_50_20_SCENE_WEIGHTS,
        obstructed_style_reward_scale=A6_CONSTRAINT_AWARE_AMP_SCALE,
        training_plate_mass_override=A6_FIXED_PLATE_MASS,
        plate_no_progress_weight=A6_PLATE_NO_PROGRESS_WEIGHT,
        escape_min_planar_clearance=A6_STRICT_ESCAPE_MIN_PLANAR_CLEARANCE,
        escape_clear_hold_s=A6_STRICT_ESCAPE_CLEAR_HOLD_S,
        exploration_cost_obstructed_scale=A6_EXPLORATION_COST_OBSTRUCTED_SCALE,
    ),
    play_env_cfg=g1_recovery_a6_env_cfg(
        play=True,
        guidance="plus",
        scene_weights=A6_BALANCED_30_50_20_SCENE_WEIGHTS,
        obstructed_style_reward_scale=A6_CONSTRAINT_AWARE_AMP_SCALE,
        training_plate_mass_override=A6_FIXED_PLATE_MASS,
        plate_no_progress_weight=A6_PLATE_NO_PROGRESS_WEIGHT,
        escape_min_planar_clearance=A6_STRICT_ESCAPE_MIN_PLANAR_CLEARANCE,
        escape_clear_hold_s=A6_STRICT_ESCAPE_CLEAR_HOLD_S,
        exploration_cost_obstructed_scale=A6_EXPLORATION_COST_OBSTRUCTED_SCALE,
    ),
    rl_cfg=g1_recovery_a6_constraint_aware_005_ppo_runner_cfg(),
    runner_cls=RecoveryOnPolicyRunner,
)

register_mjlab_task(
    task_id=A6_G_PLUS_FORCE_HOLD_INVALID2_FIXED6_TASK_ID,
    env_cfg=g1_recovery_a6_env_cfg(
        guidance="plus",
        scene_weights=A6_BALANCED_30_50_20_SCENE_WEIGHTS,
        obstructed_style_reward_scale=A6_CONSTRAINT_AWARE_AMP_SCALE,
        training_plate_mass_override=A6_FIXED_PLATE_MASS,
        plate_no_progress_weight=A6_PLATE_NO_PROGRESS_WEIGHT,
        escape_min_planar_clearance=A6_STRICT_ESCAPE_MIN_PLANAR_CLEARANCE,
        escape_clear_hold_s=A6_STRICT_ESCAPE_CLEAR_HOLD_S,
        exploration_cost_obstructed_scale=A6_EXPLORATION_COST_OBSTRUCTED_SCALE,
        plate_force_hold_s=A6_FORCE_LIMIT_HOLD_S,
        catastrophic_plate_force=A6_CATASTROPHIC_PLATE_FORCE,
        invalid_plate_termination_weight=A6_INVALID_PLATE_TERMINATION_WEIGHT,
    ),
    play_env_cfg=g1_recovery_a6_env_cfg(
        play=True,
        guidance="plus",
        scene_weights=A6_BALANCED_30_50_20_SCENE_WEIGHTS,
        obstructed_style_reward_scale=A6_CONSTRAINT_AWARE_AMP_SCALE,
        training_plate_mass_override=A6_FIXED_PLATE_MASS,
        plate_no_progress_weight=A6_PLATE_NO_PROGRESS_WEIGHT,
        escape_min_planar_clearance=A6_STRICT_ESCAPE_MIN_PLANAR_CLEARANCE,
        escape_clear_hold_s=A6_STRICT_ESCAPE_CLEAR_HOLD_S,
        exploration_cost_obstructed_scale=A6_EXPLORATION_COST_OBSTRUCTED_SCALE,
        plate_force_hold_s=A6_FORCE_LIMIT_HOLD_S,
        catastrophic_plate_force=A6_CATASTROPHIC_PLATE_FORCE,
        invalid_plate_termination_weight=A6_INVALID_PLATE_TERMINATION_WEIGHT,
    ),
    rl_cfg=g1_recovery_a6_constraint_aware_005_ppo_runner_cfg(),
    runner_cls=RecoveryOnPolicyRunner,
)

register_mjlab_task(
    task_id=A6_G_PLUS_PRONE_LATERAL_020_FIXED6_TASK_ID,
    env_cfg=g1_recovery_a6_env_cfg(
        guidance="plus",
        scene_weights=A6_BALANCED_30_50_20_SCENE_WEIGHTS,
        obstructed_style_reward_scale=A6_CONSTRAINT_AWARE_AMP_SCALE,
        training_plate_mass_override=A6_FIXED_PLATE_MASS,
        plate_no_progress_weight=A6_PLATE_NO_PROGRESS_WEIGHT,
        escape_min_planar_clearance=A6_STRICT_ESCAPE_MIN_PLANAR_CLEARANCE,
        escape_clear_hold_s=A6_STRICT_ESCAPE_CLEAR_HOLD_S,
        exploration_cost_obstructed_scale=A6_EXPLORATION_COST_OBSTRUCTED_SCALE,
        plate_force_hold_s=A6_FORCE_LIMIT_HOLD_S,
        catastrophic_plate_force=A6_CATASTROPHIC_PLATE_FORCE,
        invalid_plate_termination_weight=A6_INVALID_PLATE_TERMINATION_WEIGHT,
        prone_lateral_progress_weight=A6_PRONE_LATERAL_PROGRESS_WEIGHT,
    ),
    play_env_cfg=g1_recovery_a6_env_cfg(
        play=True,
        guidance="plus",
        scene_weights=A6_BALANCED_30_50_20_SCENE_WEIGHTS,
        obstructed_style_reward_scale=A6_CONSTRAINT_AWARE_AMP_SCALE,
        training_plate_mass_override=A6_FIXED_PLATE_MASS,
        plate_no_progress_weight=A6_PLATE_NO_PROGRESS_WEIGHT,
        escape_min_planar_clearance=A6_STRICT_ESCAPE_MIN_PLANAR_CLEARANCE,
        escape_clear_hold_s=A6_STRICT_ESCAPE_CLEAR_HOLD_S,
        exploration_cost_obstructed_scale=A6_EXPLORATION_COST_OBSTRUCTED_SCALE,
        plate_force_hold_s=A6_FORCE_LIMIT_HOLD_S,
        catastrophic_plate_force=A6_CATASTROPHIC_PLATE_FORCE,
        invalid_plate_termination_weight=A6_INVALID_PLATE_TERMINATION_WEIGHT,
        prone_lateral_progress_weight=A6_PRONE_LATERAL_PROGRESS_WEIGHT,
    ),
    rl_cfg=g1_recovery_a6_constraint_aware_005_ppo_runner_cfg(),
    runner_cls=RecoveryOnPolicyRunner,
)

def _low_torque_cap_only_env_cfg(*, play: bool):
    """Keep the model_21900 task unchanged except for its 139 Nm motor caps."""

    return g1_recovery_a6_env_cfg(
        play=play,
        guidance="plus",
        scene_weights=A6_BALANCED_30_50_20_SCENE_WEIGHTS,
        obstructed_style_reward_scale=A6_CONSTRAINT_AWARE_AMP_SCALE,
        training_plate_mass_override=A6_FIXED_PLATE_MASS,
        plate_no_progress_weight=A6_PLATE_NO_PROGRESS_WEIGHT,
        escape_min_planar_clearance=A6_STRICT_ESCAPE_MIN_PLANAR_CLEARANCE,
        escape_clear_hold_s=A6_STRICT_ESCAPE_CLEAR_HOLD_S,
        exploration_cost_obstructed_scale=A6_EXPLORATION_COST_OBSTRUCTED_SCALE,
        plate_force_hold_s=A6_FORCE_LIMIT_HOLD_S,
        catastrophic_plate_force=A6_CATASTROPHIC_PLATE_FORCE,
        invalid_plate_termination_weight=A6_INVALID_PLATE_TERMINATION_WEIGHT,
        prone_lateral_progress_weight=A6_PRONE_LATERAL_PROGRESS_WEIGHT,
        high_torque_limit_scale=A6_LOW_TORQUE_TRIAL_SCALE,
    )


register_mjlab_task(
    task_id=LOW_TORQUE_CAP_ONLY_90_TASK_ID,
    env_cfg=_low_torque_cap_only_env_cfg(play=False),
    play_env_cfg=_low_torque_cap_only_env_cfg(play=True),
    rl_cfg=g1_recovery_a6_constraint_aware_005_ppo_runner_cfg(),
    runner_cls=RecoveryOnPolicyRunner,
)


register_mjlab_task(
    task_id=A6_TASK_ID,
    env_cfg=g1_recovery_a6_env_cfg(),
    play_env_cfg=g1_recovery_a6_env_cfg(play=True),
    rl_cfg=g1_recovery_a6_ppo_runner_cfg(),
    runner_cls=RecoveryOnPolicyRunner,
)

register_mjlab_task(
    task_id=A6_G_MINUS_TASK_ID,
    env_cfg=g1_recovery_a6_env_cfg(guidance="minus"),
    play_env_cfg=g1_recovery_a6_env_cfg(play=True, guidance="minus"),
    rl_cfg=g1_recovery_a6_ppo_runner_cfg(),
    runner_cls=RecoveryOnPolicyRunner,
)

register_mjlab_task(
    task_id=A6_G_PLUS_TASK_ID,
    env_cfg=g1_recovery_a6_env_cfg(guidance="plus"),
    play_env_cfg=g1_recovery_a6_env_cfg(play=True, guidance="plus"),
    rl_cfg=g1_recovery_a6_ppo_runner_cfg(),
    runner_cls=RecoveryOnPolicyRunner,
)

register_mjlab_task(
    task_id=A6_G_MINUS_70_30_TASK_ID,
    env_cfg=g1_recovery_a6_env_cfg(
        guidance="minus", scene_weights=A6_CONSTRAINED_70_30_SCENE_WEIGHTS
    ),
    play_env_cfg=g1_recovery_a6_env_cfg(
        play=True,
        guidance="minus",
        scene_weights=A6_CONSTRAINED_70_30_SCENE_WEIGHTS,
    ),
    rl_cfg=g1_recovery_a6_ppo_runner_cfg(),
    runner_cls=RecoveryOnPolicyRunner,
)

register_mjlab_task(
    task_id=A6_G_PLUS_70_30_TASK_ID,
    env_cfg=g1_recovery_a6_env_cfg(
        guidance="plus", scene_weights=A6_CONSTRAINED_70_30_SCENE_WEIGHTS
    ),
    play_env_cfg=g1_recovery_a6_env_cfg(
        play=True,
        guidance="plus",
        scene_weights=A6_CONSTRAINED_70_30_SCENE_WEIGHTS,
    ),
    rl_cfg=g1_recovery_a6_ppo_runner_cfg(),
    runner_cls=RecoveryOnPolicyRunner,
)

register_mjlab_task(
    task_id=A6_G_MINUS_SYMMETRIC_005_TASK_ID,
    env_cfg=g1_recovery_a6_env_cfg(
        guidance="minus",
        scene_weights=A6_BALANCED_30_50_20_SCENE_WEIGHTS,
        symmetric_obstructed_scale=A6_SYMMETRIC_OBSTRUCTED_SCALE,
    ),
    play_env_cfg=g1_recovery_a6_env_cfg(
        play=True,
        guidance="minus",
        scene_weights=A6_BALANCED_30_50_20_SCENE_WEIGHTS,
        symmetric_obstructed_scale=A6_SYMMETRIC_OBSTRUCTED_SCALE,
    ),
    rl_cfg=g1_recovery_a6_symmetric_005_ppo_runner_cfg(),
    runner_cls=RecoveryOnPolicyRunner,
)

register_mjlab_task(
    task_id=A6_G_PLUS_SYMMETRIC_005_TASK_ID,
    env_cfg=g1_recovery_a6_env_cfg(
        guidance="plus",
        scene_weights=A6_BALANCED_30_50_20_SCENE_WEIGHTS,
        symmetric_obstructed_scale=A6_SYMMETRIC_OBSTRUCTED_SCALE,
    ),
    play_env_cfg=g1_recovery_a6_env_cfg(
        play=True,
        guidance="plus",
        scene_weights=A6_BALANCED_30_50_20_SCENE_WEIGHTS,
        symmetric_obstructed_scale=A6_SYMMETRIC_OBSTRUCTED_SCALE,
    ),
    rl_cfg=g1_recovery_a6_symmetric_005_ppo_runner_cfg(),
    runner_cls=RecoveryOnPolicyRunner,
)

register_mjlab_task(
    task_id=A6_G_PLUS_CONSTRAINT_AWARE_005_TASK_ID,
    env_cfg=g1_recovery_a6_env_cfg(
        guidance="plus",
        scene_weights=A6_BALANCED_30_50_20_SCENE_WEIGHTS,
        obstructed_style_reward_scale=A6_CONSTRAINT_AWARE_AMP_SCALE,
    ),
    play_env_cfg=g1_recovery_a6_env_cfg(
        play=True,
        guidance="plus",
        scene_weights=A6_BALANCED_30_50_20_SCENE_WEIGHTS,
        obstructed_style_reward_scale=A6_CONSTRAINT_AWARE_AMP_SCALE,
    ),
    rl_cfg=g1_recovery_a6_constraint_aware_005_ppo_runner_cfg(),
    runner_cls=RecoveryOnPolicyRunner,
)

register_mjlab_task(
    task_id=A6_G_PLUS_CONSTRAINT_AWARE_005_FIXED6_TASK_ID,
    env_cfg=g1_recovery_a6_env_cfg(
        guidance="plus",
        scene_weights=A6_BALANCED_30_50_20_SCENE_WEIGHTS,
        obstructed_style_reward_scale=A6_CONSTRAINT_AWARE_AMP_SCALE,
        training_plate_mass_override=A6_FIXED_PLATE_MASS,
    ),
    play_env_cfg=g1_recovery_a6_env_cfg(
        play=True,
        guidance="plus",
        scene_weights=A6_BALANCED_30_50_20_SCENE_WEIGHTS,
        obstructed_style_reward_scale=A6_CONSTRAINT_AWARE_AMP_SCALE,
        training_plate_mass_override=A6_FIXED_PLATE_MASS,
    ),
    rl_cfg=g1_recovery_a6_constraint_aware_005_ppo_runner_cfg(),
    runner_cls=RecoveryOnPolicyRunner,
)

register_mjlab_task(
    task_id=A6_G_PLUS_CONSTRAINT_AWARE_001_FIXED6_TASK_ID,
    env_cfg=g1_recovery_a6_env_cfg(
        guidance="plus",
        scene_weights=A6_BALANCED_30_50_20_SCENE_WEIGHTS,
        obstructed_style_reward_scale=A6_CONSTRAINT_AWARE_001_AMP_SCALE,
        training_plate_mass_override=A6_FIXED_PLATE_MASS,
    ),
    play_env_cfg=g1_recovery_a6_env_cfg(
        play=True,
        guidance="plus",
        scene_weights=A6_BALANCED_30_50_20_SCENE_WEIGHTS,
        obstructed_style_reward_scale=A6_CONSTRAINT_AWARE_001_AMP_SCALE,
        training_plate_mass_override=A6_FIXED_PLATE_MASS,
    ),
    rl_cfg=g1_recovery_a6_constraint_aware_001_ppo_runner_cfg(),
    runner_cls=RecoveryOnPolicyRunner,
)

register_mjlab_task(
    task_id=A6_G_PLUS_CONSTRAINT_AWARE_005_NO_PROGRESS_FIXED6_TASK_ID,
    env_cfg=g1_recovery_a6_env_cfg(
        guidance="plus",
        scene_weights=A6_BALANCED_30_50_20_SCENE_WEIGHTS,
        obstructed_style_reward_scale=A6_CONSTRAINT_AWARE_AMP_SCALE,
        training_plate_mass_override=A6_FIXED_PLATE_MASS,
        plate_no_progress_weight=A6_PLATE_NO_PROGRESS_WEIGHT,
    ),
    play_env_cfg=g1_recovery_a6_env_cfg(
        play=True,
        guidance="plus",
        scene_weights=A6_BALANCED_30_50_20_SCENE_WEIGHTS,
        obstructed_style_reward_scale=A6_CONSTRAINT_AWARE_AMP_SCALE,
        training_plate_mass_override=A6_FIXED_PLATE_MASS,
        plate_no_progress_weight=A6_PLATE_NO_PROGRESS_WEIGHT,
    ),
    rl_cfg=g1_recovery_a6_constraint_aware_005_ppo_runner_cfg(),
    runner_cls=RecoveryOnPolicyRunner,
)

__all__ = [
    "A6_G_MINUS_70_30_TASK_ID",
    "A6_G_MINUS_SYMMETRIC_005_TASK_ID",
    "A6_G_MINUS_TASK_ID",
    "A6_G_PLUS_70_30_TASK_ID",
    "A6_G_PLUS_CONSTRAINT_AWARE_001_FIXED6_TASK_ID",
    "A6_G_PLUS_CONSTRAINT_AWARE_005_FIXED6_TASK_ID",
    "A6_G_PLUS_CONSTRAINT_AWARE_005_NO_PROGRESS_FIXED6_TASK_ID",
    "A6_G_PLUS_CONSTRAINT_AWARE_005_TASK_ID",
    "A6_G_PLUS_FORCE_HOLD_INVALID2_FIXED6_TASK_ID",
    "A6_G_PLUS_PRONE_LATERAL_020_FIXED6_TASK_ID",
    "A6_G_PLUS_STRICT_ESCAPE_EXPLORATION_025_FIXED6_TASK_ID",
    "A6_G_PLUS_STRICT_ESCAPE_HEADWARD_FIXED6_TASK_ID",
    "A6_G_PLUS_SYMMETRIC_005_TASK_ID",
    "A6_G_PLUS_TASK_ID",
    "A6_TASK_ID",
    "DEV_TASK_ID",
    "LOW_TORQUE_CAP_ONLY_90_TASK_ID",
    "g1_recovery_a6_env_cfg",
    "g1_recovery_dev_env_cfg",
]
