"""29-joint flat recovery using the existing HoST port (single-critic PPO)."""
import copy

from mjlab.managers.scene_entity_config import SceneEntityCfg
from src.tasks.host_recovery import mdp
from src.assets.robots.unitree_g1.g1_constants_bp import get_g1_robot_cfg
from src.tasks.host_recovery.config.g1.env_cfgs import (
    HOST_TARGET_UPPER_DOF_POS, UPPER_BODY_JOINTS, unitree_g1_host_standup_env_cfg,
)


def flat29_env_cfg(play=False):
    cfg = unitree_g1_host_standup_env_cfg(play=play)
    cfg.scene.entities = {"robot": copy.deepcopy(get_g1_robot_cfg())}
    # Keep native incremental actions, curricula, observation history and rewards.
    # All 17 upper-body joints participate, including the six added joints.
    targets = dict(zip(UPPER_BODY_JOINTS, HOST_TARGET_UPPER_DOF_POS, strict=True))
    for name in ("waist_roll_joint", "waist_pitch_joint",
                 "left_wrist_pitch_joint", "left_wrist_yaw_joint",
                 "right_wrist_pitch_joint", "right_wrist_yaw_joint"):
        targets[name] = 0.0
    for key in ("target_target_upper_dof_pos", "regu_upper_dof_vel"):
        cfg.rewards[key].params["asset_cfg"] = SceneEntityCfg(
            "robot", joint_names=tuple(targets), preserve_order=True)
    cfg.rewards["target_target_upper_dof_pos"].params["target_upper_dof_pos"] = tuple(targets.values())
    # Reuse gravity settling, not the measured heights of the 23-joint model.
    cfg.events["reset_base"].params.update(posture=None, height=0.5)
    # Mix exact deployment conditions into every batch, then remove assistance
    # on a deterministic schedule.  The global success trigger can otherwise
    # remain stuck at 200 N with synchronized four-posture resets.
    if not play:
        cfg.curriculum["pull_force"].func = mdp.mixed_pull_force_schedule
        cfg.curriculum["pull_force"].params = {
            "initial_force": 200.0,
            "warmup_updates": 2000,
            "anneal_updates": 6000,
            "steps_per_update": 50,
            "zero_force_fraction": 0.25,
        }
        # Anti-jitter continuation for the 29-DoF policy. The inherited HoST
        # regularisation is tiny (regu_dof_vel=-4e-6, regu_action_rate=-0.0001,
        # regu_smoothness=-0.00016), so the recovered policy stands with visible
        # high-frequency joint chatter (final joint speed RMS ~7 rad/s). Boost
        # only the smoothness/velocity terms; leave regu_dof_acc and the task
        # reward untouched so the stand-up objective is not re-shaped.
        cfg.rewards["regu_dof_vel"].weight *= 100.0
        cfg.rewards["regu_upper_dof_vel"].weight *= 100.0
        cfg.rewards["regu_action_rate"].weight *= 20.0
        cfg.rewards["regu_smoothness"].weight *= 20.0
    cfg.sim.mujoco.timestep = 0.002
    cfg.decimation = 10
    return cfg
