"""Historical A6 obstacle transfer, retaining HoST's actor and four critics."""

from __future__ import annotations

import torch
from mjlab.entity import EntityCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.metrics_manager import MetricsTermCfg
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.utils.lab_api.math import quat_apply_inverse

from .a6_geometry import free_plate_spec, guided_plate_spec, scene_spec
from .a6_reset import focused_guided_layout, reset_a6_scene
from .a6_reward import a6_invalid, a6_metric, a6_reward, a6_substep
from .task import host_ground29_env_cfg


def plate_frame(env):
    """Eight board features in the robot frame; zero in flat worlds.

    The original 564 HoST values remain the first observation columns, so a
    flat expert can be transferred without changing its initial actor output.
    """
    result = torch.zeros(env.num_envs, 8, device=env.device)
    if not hasattr(env, "_a6_plate_geoms"):
        return result
    active = env._a6_scene > 0
    if not active.any():
        return result
    rows = torch.arange(env.num_envs, device=env.device)
    ids = env._a6_plate_geoms[env._a6_scene.clamp_min(1) - 1]
    robot = env.scene["robot"].data
    plate_pos = env.sim.data.geom_xpos[rows, ids]
    plate_mat = env.sim.data.geom_xmat[rows, ids]
    root_quat = robot.root_link_quat_w
    relative = plate_pos - robot.root_link_pos_w
    relative[:, 2] = 0.0
    result[:, :2] = quat_apply_inverse(root_quat, relative)[:, :2] / 0.5
    result[:, 2:4] = quat_apply_inverse(root_quat, plate_mat[:, :, 0])[:, :2]
    result[:, 4:6] = quat_apply_inverse(root_quat, plate_mat[:, :, 1])[:, :2]
    result[:, 6] = (
        (plate_pos[:, 2] - env._a6_initial_plate_z).clamp_min(0) / 0.05
    ).clamp_max(2.0)
    result[:, 7] = (env._a6_scene == 1).float()
    return result * active[:, None]


def _contact(
    name: str, primary: ContactMatch, secondary: ContactMatch, fields=("found", "force")
):
    return ContactSensorCfg(
        name=name,
        primary=primary,
        secondary=secondary,
        fields=fields,
        reduce="maxforce",
        history_length=10,
    )


def host_a6_env_cfg(
    num_envs=4096,
    seed=42,
    arm="G+",
    preserve_pose=False,
    edge_curriculum=False,
    plate_observation=False,
    focused_guided=False,
    postescape_prone=False,
    safe_actions=False,
    prone_full_arms=False,
    supine_full_arms=False,
    aligned_edge=False,
    lateral_actions=False,
    lift_guard=False,
    footprint_edge=False,
    footprint_overlap_m=0.25,
):
    if arm not in ("G+", "G-"):
        raise ValueError("A6 arm must be G+ or G-")
    if edge_curriculum and arm != "G+":
        raise ValueError("A6 edge curriculum is currently defined only for G+")
    if aligned_edge and not edge_curriculum:
        raise ValueError("aligned edge requires A6 edge curriculum")
    if footprint_edge and not aligned_edge:
        raise ValueError("footprint edge requires aligned edge")
    if lateral_actions and not safe_actions:
        raise ValueError("lateral actions require guided safe actions")
    if prone_full_arms and not safe_actions:
        raise ValueError("prone full arms require guided safe actions")
    if supine_full_arms and not safe_actions:
        raise ValueError("supine full arms require guided safe actions")
    if postescape_prone and not (focused_guided and plate_observation):
        raise ValueError("postescape prone replay requires focused guided layout and plate observation")
    cfg = host_ground29_env_cfg(num_envs, seed)
    if safe_actions:
        cfg.actions["joint_pos"].guided_arm_scale = 0.25
        cfg.actions["joint_pos"].guided_hip_scale = 0.25
        cfg.actions["joint_pos"].guided_lateral_scale = (
            1.0 if lateral_actions else 0.25
        )
    if prone_full_arms:
        cfg.actions["joint_pos"].guided_prone_arm_scale = 1.0
    if supine_full_arms:
        cfg.actions["joint_pos"].guided_supine_arm_scale = 1.0
    cfg.actions["joint_pos"].guided_lift_guard = lift_guard
    if plate_observation:
        cfg.observations["actor"].terms["plate_frame"] = ObservationTermCfg(
            func=plate_frame
        )
    cfg.scene.spec_fn = scene_spec
    cfg.scene.entities.update(
        escape_obstacle=EntityCfg(
            spec_fn=guided_plate_spec,
            init_state=EntityCfg.InitialStateCfg(
                joint_pos={"escape_plate_slide": -0.5}
            ),
        ),
        free_obstacle=EntityCfg(spec_fn=free_plate_spec),
    )
    ground = ContactMatch(mode="body", pattern="terrain")
    cfg.scene.sensors = (
        _contact(
            "guided_contact",
            ContactMatch(mode="body", pattern="escape_plate", entity="escape_obstacle"),
            ContactMatch(mode="subtree", pattern="pelvis", entity="robot"),
            ("found", "force", "dist"),
        ),
        _contact(
            "free_contact",
            ContactMatch(mode="body", pattern="plate", entity="free_obstacle"),
            ContactMatch(mode="subtree", pattern="pelvis", entity="robot"),
            ("found", "force", "dist"),
        ),
        _contact(
            "a6_hands",
            ContactMatch(
                mode="body",
                pattern=r"(left|right)_wrist_yaw_link",
                entity="robot",
            ),
            ground,
            ("found", "force"),
        ),
        _contact(
            "a6_feet",
            ContactMatch(
                mode="subtree", pattern=r"(left|right)_ankle_roll_link", entity="robot"
            ),
            ground,
        ),
    )
    cfg.sim.mujoco.timestep = 0.002
    cfg.decimation = 10
    cfg.sim.nconmax = 128
    cfg.sim.njmax = 500
    cfg.scale_rewards_by_dt = True
    cfg.events = {
        "reset_scene": EventTermCfg(
            func=reset_a6_scene,
            mode="reset",
            params={
                "edge_curriculum": edge_curriculum,
                "aligned_edge": aligned_edge,
                "footprint_edge": footprint_edge,
                "footprint_overlap_m": footprint_overlap_m,
            },
        )
    }
    if focused_guided:
        cfg.events["reset_scene"].params["layout"] = focused_guided_layout(
            num_envs, postescape_prone=postescape_prone
        )
    cfg.rewards = {
        group: RewardTermCfg(
            func=a6_reward,
            weight=1.0,
            params={"group": group, "arm": arm, "preserve_pose": preserve_pose},
        )
        for group in ("task", "regu", "style", "target")
    }
    cfg.terminations["a6_invalid"] = TerminationTermCfg(func=a6_invalid)
    cfg.metrics = {
        "a6_substep": MetricsTermCfg(func=a6_substep, per_substep=True),
        "a6_escaped": MetricsTermCfg(
            func=a6_metric, params={"key": "escaped"}, reduce="max"
        ),
        "a6_contact": MetricsTermCfg(
            func=a6_metric, params={"key": "ever_contact"}, reduce="max"
        ),
    }
    cfg.curriculum = {"assistance": cfg.curriculum["assistance"]}
    return cfg
