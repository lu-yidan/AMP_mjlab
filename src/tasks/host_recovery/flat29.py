"""29-joint flat recovery using the existing HoST port (single-critic PPO)."""
import copy

from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.reward_manager import RewardTermCfg
from src.tasks.host_recovery import mdp
from src.assets.robots.unitree_g1.g1_constants_bp import get_g1_robot_cfg
from src.tasks.host_recovery.config.g1.env_cfgs import (
    HOST_TARGET_UPPER_DOF_POS, UPPER_BODY_JOINTS, unitree_g1_host_standup_env_cfg,
)


LEG_JOINTS = (
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
    "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
    "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
)


ELBOW_JOINTS = (
    "left_elbow_joint", "right_elbow_joint",
)


ARM_TWIST_JOINTS = (
    "left_shoulder_yaw_joint", "right_shoulder_yaw_joint",
)


WRIST_JOINTS = (
    "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
    "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
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
        # Anti-jitter continuation (mild). The first attempt used x100 velocity
        # and x20 action-rate/smoothness boosts, which froze the policy in its
        # reset pose (never stood in any posture). Use a gentle regularisation
        # and instead reinforce the get-up objective: stronger head/torso and
        # waist/base height, plus arm/hand pose stability.
        cfg.rewards["regu_dof_vel"].weight *= 10.0
        cfg.rewards["regu_upper_dof_vel"].weight *= 10.0
        cfg.rewards["regu_action_rate"].weight *= 5.0
        cfg.rewards["regu_smoothness"].weight *= 5.0

        cfg.rewards["standup"].weight = 1.5
        cfg.rewards["target_target_base_height"].weight *= 2.0
        cfg.rewards["target_target_upper_dof_pos"].weight *= 2.0

        # Gated post-stand-up stability: penalise leg jitter and upper-body
        # tremor only once the robot is up, leaving the get-up motion free.
        # Legs get a stronger weight because the current model visibly shakes
        # its right leg and twists its right ankle after standing.
        cfg.rewards["target_leg_dof_vel"] = RewardTermCfg(
            func=mdp.target_dof_vel,
            weight=-0.001,
            params={
                "phase3_height": 0.65,
                "asset_cfg": SceneEntityCfg(
                    "robot", joint_names=LEG_JOINTS, preserve_order=True
                ),
            },
        )
        cfg.rewards["target_upper_dof_vel"] = RewardTermCfg(
            func=mdp.target_dof_vel,
            weight=-0.0005,
            params={
                "phase3_height": 0.65,
                "asset_cfg": SceneEntityCfg(
                    "robot", joint_names=tuple(targets), preserve_order=True
                ),
            },
        )
        cfg.rewards["style_hand_assist"] = RewardTermCfg(
            func=mdp.style_hand_assist,
            weight=0.08,
            params={
                "left_hand_cfg": SceneEntityCfg(
                    "robot", body_names=("left_wrist_yaw_link",)
                ),
                "right_hand_cfg": SceneEntityCfg(
                    "robot", body_names=("right_wrist_yaw_link",)
                ),
            },
        )
        cfg.rewards["target_elbow_up"] = RewardTermCfg(
            func=mdp.target_dof_pos_deviation,
            weight=-0.003,
            params={
                "phase3_height": 0.65,
                "asset_cfg": SceneEntityCfg(
                    "robot", joint_names=ELBOW_JOINTS, preserve_order=True
                ),
                "target_pos": (0.0, 0.0),
            },
        )
        cfg.rewards["target_arm_twist"] = RewardTermCfg(
            func=mdp.target_dof_pos_deviation,
            weight=-0.0015,
            params={
                "phase3_height": 0.65,
                "asset_cfg": SceneEntityCfg(
                    "robot", joint_names=ARM_TWIST_JOINTS, preserve_order=True
                ),
                "target_pos": (0.0, 0.0),
            },
        )
        cfg.rewards["target_wrist_flip"] = RewardTermCfg(
            func=mdp.target_dof_pos_deviation,
            weight=-0.0015,
            params={
                "phase3_height": 0.65,
                "asset_cfg": SceneEntityCfg(
                    "robot", joint_names=WRIST_JOINTS, preserve_order=True
                ),
                "target_pos": (0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
            },
        )
        cfg.rewards["target_elbow_flexion"] = RewardTermCfg(
            func=mdp.target_elbow_flexion,
            weight=-0.002,
            params={
                "phase3_height": 0.65,
                "flexion_limit": 1.2,
                "asset_cfg": SceneEntityCfg(
                    "robot", joint_names=ELBOW_JOINTS, preserve_order=True
                ),
            },
        )
    cfg.sim.mujoco.timestep = 0.002
    cfg.decimation = 10
    return cfg
