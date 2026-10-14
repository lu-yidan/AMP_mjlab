"""Flat-ground HoST get-up task adapted to mjlab's manager environment."""

import torch
import torch.nn.functional as F
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp import observations as obs
from mjlab.envs.mdp import terminations
from mjlab.envs.mdp.events import reset_scene_to_default
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.scene import SceneCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.terrains.terrain_entity import TerrainEntityCfg

from .model import (
    HOST29_STAND_ARM_JOINT_POS,
    HOST29_STAND_WAIST_JOINT_NAMES,
    HOST29_TARGET_JOINT_POS,
    HOST_JOINT_VEL_LIMITS,
    HOST_TARGET_JOINT_POS,
    get_host29_robot_cfg,
    get_host_action_cfg,
    get_host_robot_cfg,
)


def joint_pos(env):
    return env.scene["robot"].data.joint_pos


def joint_vel(env):
    return env.scene["robot"].data.joint_vel


def executed_action(env):
    return env.action_manager.get_term("joint_pos").raw_action


def beta_obs(env):
    return env.action_manager.get_term("joint_pos").beta


def host_frame(env):
    """One 76-value HoST frame in the original feature order."""
    action_term = env.action_manager.get_term("joint_pos")
    frame = torch.cat(
        (
            obs.base_ang_vel(env) * 0.25,
            obs.projected_gravity(env),
            joint_pos(env),
            joint_vel(env) * 0.05,
            executed_action(env),
            beta_obs(env) + 0.05 * (torch.rand_like(beta_obs(env)) - 0.5),
        ),
        dim=-1,
    )
    active = action_term.real_steps > 30
    if getattr(env, "_a6_postescape_live_obs_v1", False):
        # Replayed escape states have live motors from their first step. Give
        # their actor the corresponding live proprioception as well; otherwise
        # the first 30 recovery actions are chosen from an all-zero frame.
        active |= env._a6_postescape_start
    active = active.unsqueeze(1)
    return frame * active


def _head_height(env):
    robot = env.scene["robot"]
    if "keyframe_head_link" in robot.body_names:
        head_id = robot.find_bodies(("keyframe_head_link",))[0][0]
        head_z = robot.data.body_link_pos_w[:, head_id, 2]
    else:
        head_id = robot.find_geoms(("head_collision",))[0][0]
        head_z = robot.data.geom_pos_w[:, head_id, 2]
    feet_ids = robot.find_bodies(("left_ankle_roll_link", "right_ankle_roll_link"))[0]
    return head_z - robot.data.body_link_pos_w[:, feet_ids, 2].mean(dim=1)


def _tolerance_above(x, lower, margin, value_at_margin):
    distance = (lower - x).clamp_min(0.0) / margin
    return torch.exp(
        torch.log(torch.tensor(value_at_margin, device=x.device)) * distance.square()
    )


def task_reward(env):
    """Product of upright and head-height scores, as in HoST."""
    robot = env.scene["robot"]
    up = (-robot.data.projected_gravity_b[:, 2]).clamp(-1.0, 1.0)
    orientation = _tolerance_above(up, 0.99, 1.0, 0.05)
    height = _head_height(env)
    head_score = _tolerance_above(height, 1.0, 1.0, 0.1)
    env._host_head_height = height.clone()
    return orientation * head_score


def task_reward_height_guard(env):
    """Optional MuJoCo adaptation: reject airborne height-reward exploits."""
    score = task_reward(env)
    pelvis_z = env.scene["robot"].data.root_link_pos_w[:, 2]
    excess = (pelvis_z - 0.95).clamp_min(0.0)
    return score * torch.exp(-8.0 * excess.square())


def regularization_reward(env):
    robot = env.scene["robot"]
    data = robot.data
    action = env.action_manager.action
    prev = env.action_manager.prev_action
    prev_prev = env.action_manager.prev_prev_action
    # Actuators are grouped by motor class and do not share joint order.
    # qfrc_actuator is already projected into the joint_vel order.
    torque = data.qfrc_actuator
    vel = data.joint_vel
    if not hasattr(env, "_host_last_vel"):
        env._host_last_vel = torch.zeros_like(vel)
    control_acc = (env._host_last_vel - vel) / env.step_dt
    env._host_last_vel = vel.clone()
    limits = data.soft_joint_pos_limits
    velocity_limits = torch.tensor(
        [
            HOST_JOINT_VEL_LIMITS.get(
                name, 22.0 if "wrist_pitch" in name or "wrist_yaw" in name else 37.0
            )
            for name in robot.joint_names
        ],
        device=env.device,
    )
    velocity_excess = (vel.abs() - 0.9 * velocity_limits).clamp(0.0, 1.0)
    outside = (limits[..., 0] - data.joint_pos).clamp_min(0.0) + (
        data.joint_pos - limits[..., 1]
    ).clamp_min(0.0)
    return env.step_dt * (
        -2.5e-7 * control_acc.square().sum(-1)
        - 0.01 * (action - prev).square().sum(-1)
        - 0.01 * (action - 2 * prev + prev_prev).square().sum(-1)
        - 2.5e-6 * torque.square().sum(-1)
        - 2.5e-5 * (vel.abs() * torque.abs()).sum(-1)
        - 1e-3 * vel.square().sum(-1)
        - 0.00025 * (data.joint_pos_target - data.joint_pos).square().sum(-1)
        - 100.0 * outside.sum(-1)
        - velocity_excess.sum(-1)
    )


def _style_reward(env, use_roll_feet=False):
    robot = env.scene["robot"]
    data = robot.data
    q = data.joint_pos
    ids = {name: i for i, name in enumerate(robot.joint_names)}
    waist = (q[:, ids["waist_yaw_joint"]].abs() > 1.4).float()
    hip_yaw = torch.stack(
        [q[:, ids[f"{side}_hip_yaw_joint"]] for side in ("left", "right")], -1
    )
    hip_roll = torch.stack(
        [q[:, ids[f"{side}_hip_roll_joint"]] for side in ("left", "right")], -1
    )
    shoulder = torch.stack(
        [q[:, ids[f"{side}_shoulder_roll_joint"]] for side in ("left", "right")], -1
    )
    knee = torch.stack(
        [q[:, ids[f"{side}_knee_joint"]] for side in ("left", "right")], -1
    )
    knee_ids = robot.find_bodies(
        ("left_knee_link", "right_knee_link"), preserve_order=True
    )[0]
    knees = data.body_link_pos_w[:, knee_ids]
    pitch_feet_ids = robot.find_bodies(
        ("left_ankle_pitch_link", "right_ankle_pitch_link")
    )[0]
    pitch_feet = data.body_link_pos_w[:, pitch_feet_ids]
    roll_feet_ids = robot.find_bodies(
        ("left_ankle_roll_link", "right_ankle_roll_link")
    )[0]
    roll_feet = data.body_link_pos_w[:, roll_feet_ids]
    # HoST tests the vertical spread of the ankle-pitch and ankle-roll
    # links on each side (the two links matching its "ankle" body filter).
    ankle_height_spread = 50.0 * (
        (pitch_feet[:, :, 2] - roll_feet[:, :, 2]).square()
    ).mean(-1)
    ground_parallel = (ankle_height_spread < 0.05).float()
    # The upstream foot indices use ankle-pitch links. Earlier local runs
    # accidentally used ankle-roll links for these two style terms.
    feet_for_shank = roll_feet if use_roll_feet else pitch_feet
    shank = (
        (knees - feet_for_shank)[:, :, 2]
        / (knees - feet_for_shank).norm(dim=-1).clamp_min(1e-6)
    ).mean(-1)
    base_xy = data.root_link_pos_w[:, :2]
    displacement = (
        (base_xy[:, None, :] - pitch_feet[:, :, :2]).square().sum(-1).clamp_min(0.3)
    )
    foot_style = (torch.exp(-2.0 * displacement) * (pitch_feet[:, :, 2] < 0.3)).sum(-1)
    feet_distance = (feet_for_shank[:, 0] - feet_for_shank[:, 1]).norm(dim=-1)
    ang = data.root_link_ang_vel_b[:, :2]
    standing = (data.root_link_pos_w[:, 2] > 0.45).float()
    tall = (data.root_link_pos_w[:, 2] > 0.65).float()
    return env.step_dt * (
        -10.0 * waist
        - 10.0
        * ((hip_yaw.abs().amax(-1) > 1.4) | (hip_yaw.abs().amin(-1) > 0.9)).float()
        - 10.0
        * ((hip_roll.abs().amax(-1) > 1.4) | (hip_roll.abs().amin(-1) > 0.9)).float()
        - 2.5 * ((shoulder[:, 0] < -0.02) | (shoulder[:, 1] > 0.02)).float()
        + 2.5 * foot_style * tall
        - 0.25 * ((knee.abs().amax(-1) > 2.85) | (knee.amin(-1) < -0.06)).float()
        + 10.0 * _tolerance_above(shank, 0.8, 1.0, 0.1) * standing
        + 20.0 * ground_parallel
        - 10.0 * (feet_distance > 0.9).float()
        + torch.exp(-2.0 * ang.square().sum(-1)) * standing
    )


def style_reward(env):
    return _style_reward(env)


def style_reward_legacy(env):
    """Reproduce pre-v2 checkpoints; do not use for a new paper-aligned run."""
    return _style_reward(env, use_roll_feet=True)


def target_reward(env):
    robot = env.scene["robot"]
    data = robot.data
    standing = (data.root_link_pos_w[:, 2] > 0.65).float()
    target = torch.tensor(
        [HOST_TARGET_JOINT_POS.get(name, 0.0) for name in robot.joint_names],
        device=env.device,
    )
    upper = slice(12, len(robot.joint_names))
    pose = torch.exp(-0.1 * (data.joint_pos[:, upper] - target[upper]).square().sum(-1))
    orientation = torch.exp(-5.0 * data.projected_gravity_b[:, :2].square().sum(-1))
    ang = torch.exp(-2.0 * data.root_link_ang_vel_b[:, :2].square().sum(-1))
    lin = torch.exp(-5.0 * data.root_link_lin_vel_b[:, :2].square().sum(-1))
    base_height = torch.exp(-20.0 * (data.root_link_pos_w[:, 2] - 0.75).abs())
    feet_ids = robot.find_bodies(("left_ankle_pitch_link", "right_ankle_pitch_link"))[0]
    foot_heights = data.body_link_pos_w[:, feet_ids, 2]
    feet_level = torch.exp(
        -2.0 * (10.0 * (foot_heights[:, 0] - foot_heights[:, 1]).abs()).clamp_min(0.2)
    )
    return (
        env.step_dt
        * standing
        * (10.0 * (pose + orientation + ang + lin + base_height) + 2.5 * feet_level)
    )


def target_reward_29(env):
    """Standing reward for the 29-axis robot with reference arms and level torso."""
    robot = env.scene["robot"]
    data = robot.data
    if not hasattr(env, "_host29_stand_cache"):
        names = robot.joint_names
        ids = {name: i for i, name in enumerate(names)}
        env._host29_stand_cache = (
            torch.tensor(
                [HOST29_TARGET_JOINT_POS[name] for name in names], device=env.device
            ),
            [ids[name] for name in HOST29_STAND_ARM_JOINT_POS],
            [ids[name] for name in HOST29_STAND_WAIST_JOINT_NAMES],
            [
                ids[name]
                for name in names
                if "hip" in name or "knee" in name or "ankle" in name
            ],
            robot.find_bodies(("left_ankle_pitch_link", "right_ankle_pitch_link"))[0],
        )
    target, arm_ids, waist_ids, leg_ids, feet_ids = env._host29_stand_cache
    standing = (data.root_link_pos_w[:, 2] > 0.65).float()
    arm_pose = torch.exp(
        -0.75 * (data.joint_pos[:, arm_ids] - target[arm_ids]).square().sum(-1)
    )
    waist_pose = torch.exp(
        -3.0 * (data.joint_pos[:, waist_ids] - target[waist_ids]).square().sum(-1)
    )
    leg_pose = torch.exp(
        -0.3 * (data.joint_pos[:, leg_ids] - target[leg_ids]).square().sum(-1)
    )
    orientation = torch.exp(-5.0 * data.projected_gravity_b[:, :2].square().sum(-1))
    ang = torch.exp(-2.0 * data.root_link_ang_vel_b[:, :2].square().sum(-1))
    lin = torch.exp(-5.0 * data.root_link_lin_vel_b[:, :2].square().sum(-1))
    base_height = torch.exp(-20.0 * (data.root_link_pos_w[:, 2] - 0.75).abs())
    foot_heights = data.body_link_pos_w[:, feet_ids, 2]
    feet_level = torch.exp(
        -2.0 * (10.0 * (foot_heights[:, 0] - foot_heights[:, 1]).abs()).clamp_min(0.2)
    )
    return (
        env.step_dt
        * standing
        * (
            10.0 * (orientation + ang + lin + base_height)
            + 10.0 * arm_pose
            + 6.0 * waist_pose
            + 2.0 * leg_pose
            + 2.5 * feet_level
        )
    )


def target_reward_29_pose_v3(env):
    """Standing posture with per-joint Huber errors that do not saturate together."""
    robot = env.scene["robot"]
    data = robot.data
    if not hasattr(env, "_host29_stand_cache"):
        target_reward_29(env)
    target, arm_ids, waist_ids, _, feet_ids = env._host29_stand_cache
    q = data.joint_pos
    standing = (data.root_link_pos_w[:, 2] > 0.65).float()
    nearly_standing = (
        (data.root_link_pos_w[:, 2] > 0.45) & (data.projected_gravity_b[:, 2] < -0.8)
    ).float()
    arm_error = F.smooth_l1_loss(
        q[:, arm_ids],
        target[arm_ids].expand_as(q[:, arm_ids]),
        beta=0.25,
        reduction="none",
    ).sum(-1)
    waist_error = F.smooth_l1_loss(
        q[:, waist_ids],
        target[waist_ids].expand_as(q[:, waist_ids]),
        beta=0.25,
        reduction="none",
    ).sum(-1)
    upper_ids = arm_ids + waist_ids
    limits = data.soft_joint_pos_limits[:, upper_ids]
    outside = (limits[..., 0] - q[:, upper_ids]).clamp_min(0.0) + (
        q[:, upper_ids] - limits[..., 1]
    ).clamp_min(0.0)
    orientation = torch.exp(-5.0 * data.projected_gravity_b[:, :2].square().sum(-1))
    ang = torch.exp(-2.0 * data.root_link_ang_vel_b[:, :2].square().sum(-1))
    lin = torch.exp(-5.0 * data.root_link_lin_vel_b[:, :2].square().sum(-1))
    base_height = torch.exp(-20.0 * (data.root_link_pos_w[:, 2] - 0.75).abs())
    foot_heights = data.body_link_pos_w[:, feet_ids, 2]
    feet_level = torch.exp(
        -2.0 * (10.0 * (foot_heights[:, 0] - foot_heights[:, 1]).abs()).clamp_min(0.2)
    )
    return env.step_dt * (
        standing
        * (
            10.0 * (orientation + ang + lin + base_height)
            + 2.5 * feet_level
            + 10.0
            - 3.0 * arm_error
            + 6.0
            - 12.0 * waist_error
        )
        - 20.0 * nearly_standing * outside.sum(-1)
    )


def target_reward_29_pose_v4(env):
    """Strengthen waist and weak wrist-axis alignment after standing."""
    reward = target_reward_29_pose_v3(env)
    robot = env.scene["robot"]
    data = robot.data
    target, _, waist_ids, _, _ = env._host29_stand_cache
    wrist_ids = [
        i
        for i, name in enumerate(robot.joint_names)
        if "wrist_pitch" in name or "wrist_yaw" in name
    ]
    standing = (data.root_link_pos_w[:, 2] > 0.65).float()
    waist_error = F.smooth_l1_loss(
        data.joint_pos[:, waist_ids],
        target[waist_ids].expand_as(data.joint_pos[:, waist_ids]),
        beta=0.25,
        reduction="none",
    ).sum(-1)
    wrist_error = F.smooth_l1_loss(
        data.joint_pos[:, wrist_ids],
        target[wrist_ids].expand_as(data.joint_pos[:, wrist_ids]),
        beta=0.25,
        reduction="none",
    ).sum(-1)
    return reward - env.step_dt * standing * (30.0 * waist_error + 8.0 * wrist_error)


def reset_host_scene(env, env_ids):
    """HoST initial pose jitter and XY displacement, after mjlab's default reset."""
    reset_scene_to_default(env, env_ids)
    robot = env.scene["robot"]
    root = robot.data.default_root_state[env_ids].clone()
    root[:, :3] += env.scene.env_origins[env_ids]
    root[:, :2] += 2.0 * torch.rand(len(env_ids), 2, device=env.device) - 1.0
    robot.write_root_state_to_sim(root, env_ids=env_ids)
    default = robot.data.default_joint_pos[env_ids]
    scale = 0.9 + 0.2 * torch.rand_like(default)
    offset = 0.2 * torch.rand_like(default) - 0.1
    pos = default * scale + offset
    limits = robot.data.soft_joint_pos_limits[env_ids]
    pos = pos.clamp(limits[..., 0], limits[..., 1])
    robot.write_joint_state_to_sim(pos, torch.zeros_like(pos), env_ids=env_ids)
    if hasattr(env, "_host_last_vel"):
        env._host_last_vel[env_ids] = 0.0


def abnormal_velocity(env):
    robot = env.scene["robot"]
    active = env.action_manager.get_term("joint_pos").real_steps > 30
    joint_bad = robot.data.joint_vel.abs().amax(-1) > 300.0
    base_bad = robot.data.root_link_lin_vel_w.norm(dim=-1) > 20.0
    return active & (joint_bad | base_bad)


def reduce_assistance(env, env_ids):
    """Match HoST's reset-time curriculum using the final head height."""
    action = env.action_manager.get_term("joint_pos")
    if len(env_ids) == 0 or not hasattr(env, "_host_head_height"):
        return {"beta": action.beta.mean(), "assist_force": action.assist_force.mean()}
    if env._host_head_height[env_ids].mean() > 0.9:
        action.assist_force[env_ids] = (action.assist_force[env_ids] - 40.0).clamp_min(
            0.0
        )
        action.beta[env_ids] = (action.beta[env_ids] - 0.02).clamp_min(0.25)
    env._host_head_height[env_ids] = 0.0
    return {"beta": action.beta.mean(), "assist_force": action.assist_force.mean()}


def host_ground_env_cfg(
    num_envs=1, seed=42, model_source="mjlab", height_guard=False, legacy_style=False
):
    """HoST G1 23-DoF flat get-up at 50 Hz control / 200 Hz physics."""
    actor_terms = {"host_frame": ObservationTermCfg(func=host_frame)}
    return ManagerBasedRlEnvCfg(
        scene=SceneCfg(
            num_envs=num_envs,
            terrain=TerrainEntityCfg(terrain_type="plane"),
            entities={"robot": get_host_robot_cfg(model_source=model_source)},
        ),
        sim=SimulationCfg(
            nconmax=64,
            njmax=250,
            mujoco=MujocoCfg(timestep=0.005, iterations=10, ls_iterations=20),
        ),
        decimation=4,
        episode_length_s=10.0,
        scale_rewards_by_dt=False,
        seed=seed,
        is_finite_horizon=True,
        observations={
            "actor": ObservationGroupCfg(terms=actor_terms, history_length=6)
        },
        actions={"joint_pos": get_host_action_cfg()},
        events={"reset_scene": EventTermCfg(func=reset_host_scene, mode="reset")},
        rewards={
            "task": RewardTermCfg(
                func=task_reward_height_guard if height_guard else task_reward,
                weight=1.0,
            ),
            "regu": RewardTermCfg(func=regularization_reward, weight=1.0),
            "style": RewardTermCfg(
                func=style_reward_legacy if legacy_style else style_reward, weight=1.0
            ),
            "target": RewardTermCfg(func=target_reward, weight=1.0),
        },
        terminations={
            "time_out": TerminationTermCfg(func=terminations.time_out, time_out=True),
            "abnormal_velocity": TerminationTermCfg(func=abnormal_velocity),
        },
        curriculum={"assistance": CurriculumTermCfg(func=reduce_assistance)},
    )


def host_ground29_env_cfg(num_envs=1, seed=42, height_guard=True, pose_version="v2"):
    """29-DoF flat warm-up on the same Unitree MJCF used by A6."""
    cfg = host_ground_env_cfg(
        num_envs, seed, model_source="mjlab", height_guard=height_guard
    )
    cfg.scene.entities["robot"] = get_host29_robot_cfg()
    pose_reward = {
        "v2": target_reward_29,
        "v3": target_reward_29_pose_v3,
        "v4": target_reward_29_pose_v4,
        "v5": target_reward_29_pose_v4,
        "v6": target_reward_29_pose_v4,
    }[pose_version]
    cfg.rewards["target"] = RewardTermCfg(func=pose_reward, weight=1.0)
    return cfg
