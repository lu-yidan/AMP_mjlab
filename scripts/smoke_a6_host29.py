"""Behavioral preflight for the 29-DoF HoST historical A6 environment."""

from __future__ import annotations

import argparse
import json

import mujoco
import numpy as np
import torch
from mjlab.asset_zoo.robots.unitree_g1.g1_constants import (
    ARMATURE_4010,
    ARMATURE_5020,
    DAMPING_4010,
    DAMPING_5020,
    STIFFNESS_4010,
    STIFFNESS_5020,
    get_spec,
)
from mjlab.envs import ManagerBasedRlEnv

from g1recovery_host.a6_curriculum import SuccessGatedEdgeCurriculum
from g1recovery_host.a6_geometry import PLATE_MASS_KG, footprint, quotas
from g1recovery_host.a6_reset import (
    BANK_ROOT,
    edge_offset_for_iteration,
    focused_guided_layout,
)
from g1recovery_host.a6_reward import (
    EFFORT,
    GUIDED_HIP_SOFT_LIMIT_RAD,
    GUIDED_KNEE_SOFT_LIMIT_RAD,
    PRE_ESCAPE_KNEE_TOLERANCE_RAD,
    TASK_WEIGHTS,
    a6_reward,
    guided_exit_margin,
    guided_leg_posture,
    guided_linear_leg_alignment,
    guided_overcurl,
    hand_supported_lateral_progress,
    headward_upper_arm_score,
    joint_stall_cost,
    lateral_travel,
    new_lateral_progress,
    new_torso_progress,
    no_progress_ramp,
    outward_hand_drive,
    path_progress,
    pre_escape_knee_curl,
    staged_guided_knee_action,
)
from g1recovery_host.a6_task import host_a6_env_cfg
from g1recovery_host.model import HOST29_JOINT_NAMES
from g1recovery_host.ppo import (
    GROUP_WEIGHTS,
    discounted_progress_advantage,
    grouped_gae,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--num-envs", type=int, default=32)
    parser.add_argument("--steps", type=int, default=501)
    parser.add_argument("--edge-curriculum", action="store_true")
    parser.add_argument("--staged-escape", action="store_true")
    parser.add_argument("--staged-escape-v2", action="store_true")
    parser.add_argument("--staged-escape-v3", action="store_true")
    parser.add_argument("--staged-escape-v4", action="store_true")
    parser.add_argument("--staged-escape-v5", action="store_true")
    parser.add_argument("--focused-guided", action="store_true")
    parser.add_argument("--safe-actions", action="store_true")
    parser.add_argument("--aligned-edge", action="store_true")
    parser.add_argument("--lateral-actions", action="store_true")
    parser.add_argument("--lift-guard", action="store_true")
    parser.add_argument("--knee-straight-prior", action="store_true")
    parser.add_argument("--footprint-edge", action="store_true")
    parser.add_argument("--footprint-overlap-m", type=float, default=0.25)
    parser.add_argument("--arm-extraction", action="store_true")
    parser.add_argument("--guided-offset-m", type=float, default=0.0)
    args = parser.parse_args()
    if args.guided_offset_m > 0 and not args.edge_curriculum:
        parser.error("nonzero guided offset requires --edge-curriculum")
    if args.focused_guided and not args.staged_escape_v5:
        parser.error("--focused-guided requires --staged-escape-v5")
    if args.safe_actions and not args.staged_escape_v5:
        parser.error("--safe-actions requires --staged-escape-v5")
    if args.aligned_edge and not args.edge_curriculum:
        parser.error("--aligned-edge requires --edge-curriculum")
    if args.lateral_actions and not args.safe_actions:
        parser.error("--lateral-actions requires --safe-actions")
    if args.lift_guard and not args.safe_actions:
        parser.error("--lift-guard requires --safe-actions")
    if args.knee_straight_prior and not args.staged_escape_v5:
        parser.error("--knee-straight-prior requires --staged-escape-v5")
    if args.footprint_edge and not args.aligned_edge:
        parser.error("--footprint-edge requires --aligned-edge")
    if args.arm_extraction and not args.staged_escape_v5:
        parser.error("--arm-extraction requires --staged-escape-v5")
    if (
        sum(
            (
                args.staged_escape,
                args.staged_escape_v2,
                args.staged_escape_v3,
                args.staged_escape_v4,
                args.staged_escape_v5,
            )
        )
        > 1
    ):
        parser.error("choose only one staged escape version")
    if (
        args.staged_escape
        or args.staged_escape_v2
        or args.staged_escape_v3
        or args.staged_escape_v4
        or args.staged_escape_v5
    ) and not args.edge_curriculum:
        parser.error("staged escape requires --edge-curriculum")
    guide = staged_guided_knee_action(
        torch.tensor([[1.5, 1.5], [1.5, 1.5], [0.45, 0.45]]),
        torch.tensor([1, 1, 0]),
        torch.tensor([0.0, 0.18, 0.0]),
    )
    assert torch.allclose(guide[0], torch.zeros(2))
    assert torch.allclose(guide[1], torch.full((2,), -0.45))
    assert torch.allclose(guide[2], torch.zeros(2))
    strict_guide = staged_guided_knee_action(
        torch.tensor([[1.5, 1.5], [0.35, 0.35]]),
        torch.tensor([1, 0]),
        torch.tensor([0.0, 0.18]),
        strict=True,
    )
    assert torch.allclose(strict_guide[0], torch.full((2,), -0.45))
    assert torch.allclose(strict_guide[1], torch.zeros(2))
    assert (
        staged_guided_knee_action(
            torch.tensor([[1.5, 1.5]]), torch.tensor([1]), torch.tensor([0.0])
        )
        .abs()
        .max()
        < 1e-6
    )
    assert edge_offset_for_iteration(1000, 1000, 0.25, 100, 300) == 0.25
    assert edge_offset_for_iteration(1099, 1000, 0.25, 100, 300) == 0.25
    assert 0 < edge_offset_for_iteration(1100, 1000, 0.25, 100, 300) < 0.25
    assert edge_offset_for_iteration(1399, 1000, 0.25, 100, 300) == 0.0
    focus = focused_guided_layout(256)
    assert np.bincount(focus[:, 0], minlength=3).tolist() == [64, 160, 32]
    assert np.bincount(focus[focus[:, 0] == 1, 1], minlength=4).tolist() == [
        80, 80, 0, 0
    ]
    adaptive = SuccessGatedEdgeCurriculum()
    for _ in range(19):
        assert not adaptive.observe((0, 2, 0), (0, 2, 0))[0]
    assert not adaptive.observe((0, 2, 0), (0, 2, 0))[0]
    assert adaptive.offset_m == 0.25, "failure cannot advance the curriculum"
    advanced = False
    for _ in range(20):
        moved, _ = adaptive.observe((2, 0, 0), (2, 0, 0))
        advanced |= moved
    assert advanced and adaptive.offset_m == 0.225
    restored = SuccessGatedEdgeCurriculum(state=adaptive.state_dict())
    assert restored.offset_m == adaptive.offset_m
    aux = discounted_progress_advantage(
        torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 0.0]]),
        torch.tensor([[False, False], [True, True], [True, True]]),
    )
    assert torch.isfinite(aux).all() and aux[1, 0] > aux[1, 1]
    margin = guided_exit_margin(
        torch.tensor([[0.0, -0.25], [0.0, 0.0]]), torch.zeros(2, 2)
    )
    assert torch.allclose(margin, torch.tensor([0.07, 0.32]))
    signed = lateral_travel(
        torch.tensor([[0.0, -0.1], [0.0, 0.1]]),
        torch.tensor([[0.0, 1.0], [0.0, 1.0]]),
        toward_near_edge=True,
    )
    assert torch.allclose(signed, torch.tensor([0.1, 0.0]))
    torso_best, torso_gain = new_torso_progress(
        torch.tensor([0.0, 0.10, 0.20]), torch.tensor([0.0, 0.05, 0.25])
    )
    assert torch.allclose(torso_best, torch.tensor([0.0, 0.10, 0.25]))
    assert torch.allclose(torso_gain, torch.tensor([0.0, 1.0, 0.0]))
    scene, direction, stratum = quotas(4096)
    assert torch.bincount(torch.tensor(scene), minlength=3).tolist() == [
        2048,
        1024,
        1024,
    ]
    assert torch.bincount(torch.tensor(stratum), minlength=3).tolist() == [
        2868,
        820,
        408,
    ]
    assert all(((direction == d) & (stratum == 0)).sum() == 717 for d in range(4))
    xyz = torch.tensor([[[0.0, 0.0, 0.2]]])
    extent = torch.tensor([[[0.1, 0.1, 0.1]]])
    plate = torch.tensor([[0.0, 0.0, 0.4]])
    board_extent = torch.tensor([[0.2, 0.2, 0.03]])
    low = footprint(xyz, extent, plate, board_extent)
    high = footprint(xyz, extent, plate + torch.tensor([[0.0, 0.0, 1.0]]), board_extent)
    assert all(torch.equal(a, b) for a, b in zip(low, high, strict=True))
    change = torch.ones(3) * 0.025
    gated = path_progress(
        change,
        change,
        torch.tensor([0.5, 0.5, 0.95]),
        torch.tensor([1.0, 0.0, 1.0]),
        torch.ones(3, dtype=torch.bool),
    )
    assert gated[0] > 0 and gated[1:].sum() == 0
    assert (
        joint_stall_cost(torch.tensor([[1.0, 0.0]]), torch.tensor([[0.0, 0.5]])).item()
        == 1.0
    )
    best, gain = new_lateral_progress(
        torch.tensor([[0.03, 0.0], [0.01, 0.0]]),
        torch.tensor([[1.0, 0.0], [1.0, 0.0]]),
        torch.tensor([0.01, 0.02]),
    )
    assert torch.allclose(best, torch.tensor([0.03, 0.02]))
    assert torch.allclose(gain, torch.tensor([0.8, 0.0]))
    assert torch.allclose(
        no_progress_ramp(torch.tensor([0.5, 1.0, 1.5])),
        torch.tensor([0.0, 0.5, 1.0]),
    )
    assert torch.allclose(
        no_progress_ramp(torch.tensor([0.2, 0.55, 0.9]), 0.2, 0.9),
        torch.tensor([0.0, 0.5, 1.0]),
    )
    knee_bend, hip_bend, straight, extension = guided_leg_posture(
        torch.tensor([[2.4, 2.2], [0.4, 0.3]]),
        torch.tensor([[1.5, -1.5], [-0.1, -0.1]]),
        torch.tensor([[2.5, 2.3], [0.4, 0.3]]),
        torch.tensor([[1.5, -1.5], [-0.1, -0.1]]),
    )
    assert knee_bend[0] > 0.5 and knee_bend[1] == 0
    assert hip_bend[0] > 0 and hip_bend[1] == 0
    assert straight[0] < 0.1 and straight[1] == 1
    assert extension[0] > 0 and extension[1] == 0
    _, _, _, recurl = guided_leg_posture(
        torch.tensor([[2.5, 2.3]]),
        torch.tensor([[1.5, -1.5]]),
        torch.tensor([[2.4, 2.2]]),
        torch.tensor([[1.5, -1.5]]),
    )
    assert recurl[0] < 0, "curling back must cancel earlier extension credit"
    alignment = guided_linear_leg_alignment(
        torch.tensor([[2.8, 2.8], [1.5, 1.5], [0.3, 0.3]]),
        torch.tensor([[1.5, -1.5], [1.0, -1.0], [-0.1, -0.1]]),
    )
    assert 0 < alignment[0] < alignment[1] < alignment[2] == 1
    arms = headward_upper_arm_score(
        torch.tensor([[-1.0, -1.0], [0.6, 0.6], [0.6, -1.0]])
    )
    assert torch.allclose(arms, torch.tensor([0.0, 1.0, 0.0]))
    early_arm = headward_upper_arm_score(torch.tensor([[-0.8, -0.8]]))
    assert 0 < early_arm[0] < 1
    push = hand_supported_lateral_progress(
        torch.tensor([0.02, 0.02, 0.01]),
        torch.tensor([0.01, 0.01, 0.02]),
        torch.tensor([1.0, 0.0, 1.0]),
    )
    assert torch.allclose(push, torch.tensor([1.0, 0.0, 0.0]))
    overcurl = guided_overcurl(
        torch.tensor([[0.3, 0.4], [2.8, 2.7]]),
        torch.tensor([[0.0, 0.0], [1.8, -1.8]]),
    )
    assert overcurl[0] == 0 and overcurl[1] > 0.9
    drive = outward_hand_drive(
        torch.tensor([[0.0, 0.1], [0.0, 0.1], [0.0, -0.1]]),
        torch.tensor([[0.0, 0.1], [0.0, -0.1], [0.0, -0.1]]),
        torch.tensor([[0.0, 1.0]]).expand(3, -1),
        torch.tensor([1.0, 1.0, 0.5]),
    )
    assert drive[0] > 0 and drive[1] == 0 and drive[2] > 0
    near_best, near_gain = new_lateral_progress(
        torch.tensor([[0.0, -0.03], [0.0, 0.03]]),
        torch.tensor([[0.0, 1.0], [0.0, 1.0]]),
        torch.zeros(2),
        toward_near_edge=True,
    )
    assert torch.allclose(near_best, torch.tensor([0.03, 0.0]))
    assert near_gain[0] > 0 and near_gain[1] == 0
    near_drive = outward_hand_drive(
        torch.zeros(2, 2),
        torch.tensor([[0.0, -0.1], [0.0, 0.1]]),
        torch.tensor([[0.0, 1.0], [0.0, 1.0]]),
        torch.ones(2),
        toward_near_edge=True,
    )
    assert near_drive[0] > 0 and near_drive[1] == 0
    # Raw reward scaling is normalized away by four-critic GAE; a phase
    # scale applied after normalization must change only the selected rows.
    torch.manual_seed(13)
    rewards = torch.randn(1, 8, 4)
    values = torch.zeros_like(rewards)
    dones = torch.ones(1, 8, dtype=torch.bool)
    _, base_adv = grouped_gae(rewards, values, dones, values[0])
    scales = torch.ones_like(rewards)
    scales[:, 1:4, 0] = 0.05
    scales[:, 1:4, 2] = 0.05
    _, scaled_adv = grouped_gae(
        rewards, values, dones, values[0], advantage_scales=scales
    )
    normalized = (rewards - rewards.mean(dim=(0, 1))) / rewards.std(dim=(0, 1))
    expected_change = -0.95 * (
        GROUP_WEIGHTS[0] * normalized[:, 1:4, 0]
        + GROUP_WEIGHTS[2] * normalized[:, 1:4, 2]
    )
    assert torch.allclose(scaled_adv[:, 1:4] - base_adv[:, 1:4], expected_change)
    assert torch.allclose(scaled_adv[:, 4:], base_adv[:, 4:])
    assert GUIDED_KNEE_SOFT_LIMIT_RAD < 0.8
    assert PRE_ESCAPE_KNEE_TOLERANCE_RAD < np.deg2rad(20)
    knee_cost = pre_escape_knee_curl(
        torch.tensor([[0.3, 0.3], [0.0, 0.0], [0.65, 0.55], [1.5, 0.65], [-0.08, 0.3]])
    )
    assert knee_cost[0] == 0 and knee_cost[1] == 0
    assert 0 < knee_cost[2] < knee_cost[3] < 3
    assert knee_cost[4] > 0
    # Deep flexion must remain distinguishable: 0.1 rad of extension at 2.5 rad
    # must reduce the cost by at least 0.02 for one knee.
    deep_cost = pre_escape_knee_curl(torch.tensor([[2.5, 0.3], [2.4, 0.3]]))
    assert deep_cost[0] - deep_cost[1] > 0.02
    assert GUIDED_HIP_SOFT_LIMIT_RAD < 1.0
    manifest = json.loads((BANK_ROOT / "manifest.json").read_text(encoding="utf-8"))
    assert tuple(manifest["source_joint_names"]) == HOST29_JOINT_NAMES
    assert tuple(manifest["host_joint_names"]) == HOST29_JOINT_NAMES
    assert manifest["keep_source_indices"] == list(range(29))
    source_model = get_spec().compile()
    source_joint_ids = [
        mujoco.mj_name2id(source_model, mujoco.mjtObj.mjOBJ_JOINT, name)
        for name in HOST29_JOINT_NAMES
    ]
    limits = source_model.jnt_range[source_joint_ids]
    for bank_name in manifest["banks"]:
        with np.load(BANK_ROOT / f"{bank_name}_host29.npz") as bank:
            qpos = bank["qpos"]
            assert qpos.shape == (manifest["banks"][bank_name]["count"], 36)
            assert (qpos[:, 7:] >= limits[:, 0] - 1e-6).all()
            assert (qpos[:, 7:] <= limits[:, 1] + 1e-6).all()
    env = ManagerBasedRlEnv(
        host_a6_env_cfg(
            args.num_envs,
            edge_curriculum=args.edge_curriculum,
            plate_observation=args.staged_escape_v5,
            focused_guided=args.focused_guided,
            safe_actions=args.safe_actions,
            aligned_edge=args.aligned_edge,
            lateral_actions=args.lateral_actions,
            lift_guard=args.lift_guard,
            footprint_edge=args.footprint_edge,
            footprint_overlap_m=args.footprint_overlap_m,
        ),
        device=args.device,
    )
    try:
        env._a6_staged_escape_v1 = args.staged_escape
        env._a6_staged_escape_v2 = args.staged_escape_v2
        env._a6_staged_escape_v3 = args.staged_escape_v3
        env._a6_staged_escape_v4 = args.staged_escape_v4
        env._a6_staged_escape_v5 = args.staged_escape_v5
        env._a6_knee_straight_prior_v1 = args.knee_straight_prior
        env._a6_arm_extraction_v1 = args.arm_extraction
        env._a6_edge_offset_m = args.guided_offset_m
        obs, _ = env.reset()
        assert env.scene["a6_hands"].data.force is not None
        plate_masses = env.sim.model.body_mass[:, env._a6_plate_bodies]
        assert torch.allclose(
            plate_masses, torch.full_like(plate_masses, PLATE_MASS_KG)
        ), "both A6 plate bodies must remain 6 kg in every world"
        robot = env.scene["robot"]
        assert tuple(robot.joint_names) == HOST29_JOINT_NAMES
        assert set(robot.actuator_names) == set(HOST29_JOINT_NAMES)
        assert obs["actor"].shape == (
            args.num_envs,
            612 if args.staged_escape_v5 else 564,
        )
        if args.staged_escape_v5:
            plate_history = obs["actor"][:, 564:]
            assert plate_history[env._a6_scene == 1].abs().sum() > 0
            flat_history = plate_history[env._a6_scene == 0]
            assert torch.allclose(flat_history, torch.zeros_like(flat_history))
        assert env.action_manager.total_action_dim == 29
        slide = int(env.scene["escape_obstacle"].indexing.joint_ids[0])
        assert int(env.sim.mj_model.jnt_type[slide]) == int(mujoco.mjtJoint.mjJNT_SLIDE)
        guided = env._a6_scene == 1
        rows = torch.where(guided)[0]
        plate = env.sim.data.geom_xpos[rows, env._a6_plate_geoms[0]]
        root = robot.data.root_link_pos_w[rows]
        plate_delta = plate[:, :2] - root[:, :2]
        projected_offset = (plate_delta * env._a6_initial_lateral_axis_xy[rows]).sum(-1)
        if args.footprint_edge and args.guided_offset_m > 0:
            assert (projected_offset > args.guided_offset_m).all()
        elif args.aligned_edge and args.guided_offset_m > 0:
            x_axis = env._a6_initial_lateral_axis_xy[rows, 0].abs() > 0.5
            expected = torch.where(
                x_axis,
                torch.full_like(
                    projected_offset,
                    min(args.guided_offset_m * (0.38 / 0.25), 0.60),
                ),
                torch.full_like(projected_offset, args.guided_offset_m),
            )
            assert torch.allclose(projected_offset, expected, atol=1e-4)
            if args.guided_offset_m == 0.25:
                assert torch.allclose(
                    guided_exit_margin(root[:, :2], plate[:, :2]),
                    torch.full_like(projected_offset, 0.07),
                    atol=1e-4,
                )
        else:
            assert torch.allclose(
                projected_offset,
                torch.full_like(projected_offset, args.guided_offset_m),
                atol=1e-4,
            )
        assert torch.allclose(
            env._a6_initial_edge_offset_m[rows], projected_offset, atol=1e-4
        )
        if not args.edge_curriculum:
            assert plate_delta.abs().max() < 1e-4
        assert (env._a6_initial_count[env._a6_scene > 0] > 0).all()
        assert len(env._a6_leg_collision_indices) == 18
        assert (env._a6_initial_leg_count >= 0).all()
        assert torch.allclose(
            env._a6_initial_root_xy, robot.data.root_link_pos_w[:, :2], atol=1e-4
        )
        assert torch.allclose(
            env._a6_initial_lateral_axis_xy.norm(dim=-1),
            torch.ones(args.num_envs, device=env.device),
            atol=1e-4,
        )
        action = env.action_manager.get_term("joint_pos")
        assert tuple(action._target_names) == HOST29_JOINT_NAMES
        assert torch.allclose(action._effort_limit.cpu(), EFFORT)
        if hasattr(env, "_a6_effort_limits"):
            assert torch.allclose(env._a6_effort_limits.cpu(), EFFORT)
        # The assembled A6 scene namespaces robot joints. Entity indexing is
        # the authoritative joint order; bare names in mj_name2id return -1.
        joint_ids = robot.indexing.joint_ids
        runtime_limits = robot.data.joint_pos_limits[0].cpu().numpy()
        assert np.allclose(
            env.sim.model.jnt_range[0, joint_ids].cpu().numpy(),
            runtime_limits,
            atol=1e-6,
        )
        if not np.allclose(runtime_limits, limits, atol=1e-6):
            raise AssertionError(
                f"runtime hard limits differ from source MJCF: "
                f"shape={np.shape(runtime_limits)} "
                f"diff={[(name, runtime_limits[i].tolist(), limits[i].tolist()) for i, name in enumerate(HOST29_JOINT_NAMES) if not np.allclose(runtime_limits[i], limits[i], atol=1e-6)]}"
            )
        midpoint = runtime_limits.mean(axis=-1)
        soft_expected = np.stack(
            (
                midpoint - 0.45 * np.diff(runtime_limits, axis=-1).ravel(),
                midpoint + 0.45 * np.diff(runtime_limits, axis=-1).ravel(),
            ),
            axis=-1,
        )
        assert np.allclose(
            robot.data.soft_joint_pos_limits[0].cpu().numpy(), soft_expected, atol=1e-6
        )
        new_expected = {
            "waist_roll_joint": (
                STIFFNESS_5020 * 2,
                DAMPING_5020 * 2,
                ARMATURE_5020 * 2,
            ),
            "waist_pitch_joint": (
                STIFFNESS_5020 * 2,
                DAMPING_5020 * 2,
                ARMATURE_5020 * 2,
            ),
            **{
                f"{side}_wrist_{axis}_joint": (
                    STIFFNESS_4010,
                    DAMPING_4010,
                    ARMATURE_4010,
                )
                for side in ("left", "right")
                for axis in ("pitch", "yaw")
            },
        }
        for name, (kp, kd, armature) in new_expected.items():
            i = HOST29_JOINT_NAMES.index(name)
            dof_id = int(robot.indexing.joint_v_adr[i])
            assert np.isclose(float(action._kp[i]), kp, rtol=1e-5)
            assert np.isclose(float(action._kd[i]), kd, rtol=1e-5)
            assert np.isclose(
                float(env.sim.model.dof_armature[0, dof_id]), armature, rtol=1e-5
            )
        for i, name in enumerate(HOST29_JOINT_NAMES):
            if name in new_expected:
                continue
            expected = (
                (150.0, 4.0)
                if "hip_" in name
                else (200.0, 6.0)
                if "knee" in name
                else (40.0, 2.0)
                if "ankle" in name
                else (100.0, 4.0)
            )
            assert np.allclose((float(action._kp[i]), float(action._kd[i])), expected)
            assert np.isclose(
                float(
                    env.sim.model.dof_armature[0, int(robot.indexing.joint_v_adr[i])]
                ),
                0.01,
                rtol=1e-5,
            )
        assert np.isclose(env.step_dt, 0.02)
        assert np.isclose(env.physics_dt, 0.002)
        new = [
            HOST29_JOINT_NAMES.index(name)
            for name in (
                "waist_roll_joint",
                "waist_pitch_joint",
                "left_wrist_pitch_joint",
                "left_wrist_yaw_joint",
                "right_wrist_pitch_joint",
                "right_wrist_yaw_joint",
            )
        ]
        assert torch.allclose(
            action._effort_limit[new],
            torch.tensor([50.0, 50.0, 5.0, 5.0, 5.0, 5.0], device=env.device),
        )
        print(
            f"model=29DoF observation={obs['actor'].shape[1]} action=29 "
            "six-extra-axis gains/armature/limits aligned; banks=identity order; "
            "scene=2048/1024/1024 low/mid/late=2868/820/408",
            flush=True,
        )
        # A subset reset must not overwrite other worlds' robot state.
        before = robot.data.joint_pos.clone()
        selected = torch.tensor(
            [1, args.num_envs // 2, args.num_envs - 1], device=env.device
        )
        env.reset(env_ids=selected)
        assert torch.allclose(
            env.sim.model.body_mass[:, env._a6_plate_bodies],
            torch.full_like(plate_masses, PLATE_MASS_KG),
        ), "partial resets must preserve the fixed A6 plate mass"
        untouched = torch.ones(args.num_envs, device=env.device, dtype=torch.bool)
        untouched[selected] = False
        assert torch.equal(robot.data.joint_pos[untouched], before[untouched])
        done_total = contact_peak = actor_active_peak = 0
        for step in range(args.steps):
            obs, reward, terminated, truncated, _ = env.step(
                torch.zeros(args.num_envs, 29, device=env.device)
            )
            if not reward.isfinite().all() or not obs["actor"].isfinite().all():
                raise FloatingPointError(f"non-finite A6 signal at step {step}")
            if step == 0:
                plus = a6_reward(env, "task", "G+")
                minus = a6_reward(env, "task", "G-")
                alpha = torch.where(env._a6_scene > 0, 0.05, 1.0)
                guided_obstructed = (env._a6_scene == 1) & ~env._a6_escaped
                suppressed = (
                    env._a6_task_values[:, :4]
                    * torch.tensor(TASK_WEIGHTS[:4], device=env.device)
                ).sum(-1)
                assert torch.allclose(
                    plus, alpha * (minus - guided_obstructed * suppressed), atol=1e-6
                )
                guided_ids = torch.where(env._a6_scene == 1)[0]
                escaped_before = env._a6_escaped[guided_ids].clone()
                env._a6_escaped[guided_ids] = True
                assert torch.allclose(
                    a6_reward(env, "task", "G+")[guided_ids],
                    minus[guided_ids],
                    atol=1e-6,
                )
                env._a6_escaped[guided_ids] = escaped_before
                plus_style = a6_reward(env, "style", "G+")
                minus_style = a6_reward(env, "style", "G-")
                assert torch.allclose(plus_style, alpha * minus_style, atol=1e-6)
                delta = a6_reward(env, "target", "G+") - a6_reward(env, "target", "G-")
                guided_scale = (env._a6_scene == 1).float()
                staged = (
                    args.staged_escape
                    or args.staged_escape_v2
                    or args.staged_escape_v3
                    or args.staged_escape_v4
                    or args.staged_escape_v5
                )
                leg_stage_weight = (
                    env._a6_knee_penalty_weight
                    if args.staged_escape_v2
                    or args.staged_escape_v3
                    or args.staged_escape_v4
                    or args.staged_escape_v5
                    else 1.0
                )
                expected = (
                    (0.45 + 0.20 * guided_scale) * env._a6_escape_values[:, 0]
                    + (0.08 + 0.04 * guided_scale) * env._a6_escape_values[:, 1]
                    + (0.10 + 0.05 * guided_scale) * env._a6_escape_values[:, 3]
                    + (0.50 if args.staged_escape_v5 else 3.00 if staged else 1.00)
                    * env._a6_escape_values[:, 7]
                    - (0.15 + 1.35 * guided_scale) * env._a6_escape_values[:, 8]
                    - (0.10 + 0.02 * guided_scale) * env._a6_escape_values[:, 9]
                    - (5.00 if args.staged_escape_v5 else 0.45)
                    * env._a6_escape_values[:, 10]
                    + (0.50 if args.staged_escape_v5 else 3.00 if staged else 1.00)
                    * env._a6_escape_values[:, 11]
                    - 0.90 * leg_stage_weight * env._a6_escape_values[:, 12]
                    + 0.50 * leg_stage_weight * env._a6_escape_values[:, 13]
                    + 0.25 * leg_stage_weight * env._a6_escape_values[:, 14]
                    + 0.12 * leg_stage_weight * env._a6_escape_values[:, 15]
                    + 0.80 * leg_stage_weight * env._a6_escape_values[:, 18]
                    - 0.35 * leg_stage_weight * env._a6_escape_values[:, 17]
                    + (
                        10.00
                        if args.staged_escape_v5
                        else 15.00
                        if args.staged_escape_v4
                        else 3.00
                        if staged
                        else 0.50
                    )
                    * env._a6_escape_values[:, 19]
                    + 0.60 * env._a6_escape_values[:, 20]
                    + 0.80 * leg_stage_weight * env._a6_escape_values[:, 21]
                    + (3.00 if args.staged_escape_v5 else 2.50 if staged else 1.50)
                    * env._a6_escape_values[:, 22]
                    + (2.50 if args.staged_escape_v5 else 2.00 if staged else 1.20)
                    * env._a6_escape_values[:, 23]
                    + 0.15 * env._a6_escape_values[:, 24]
                    - 1.20 * leg_stage_weight * env._a6_escape_values[:, 25]
                    - 650.0 * guided_scale * env._a6_invalid.float()
                )
                assert torch.allclose(delta, expected, atol=1e-6)
            assert (env._a6_escape_values[env._a6_scene != 1, 11:26] == 0).all()
            knee_gate = (env._a6_scene > 0) & ~env._a6_knee_reward_done
            expected_knee_cost = knee_gate.float() * pre_escape_knee_curl(
                env.scene["robot"].data.joint_pos[:, env._a6_knee_ids]
            )
            if staged:
                expected_knee_cost *= env._a6_knee_penalty_weight
            # With auto-reset, the reward cache describes the terminal frame
            # while joint_pos already describes the new episode for done rows.
            surviving = ~(terminated | truncated)
            assert torch.allclose(
                env._a6_escape_values[surviving, 26],
                expected_knee_cost[surviving],
            )
            knee_cost_saved = env._a6_escape_values[:, 26].clone()
            target_with_knee = a6_reward(env, "target", "G+")
            env._a6_escape_values[:, 26] = 0.0
            target_without_knee = a6_reward(env, "target", "G+")
            env._a6_escape_values[:, 26] = knee_cost_saved
            assert torch.allclose(
                target_without_knee - target_with_knee,
                knee_cost_saved,
                atol=1e-5,
            )
            done_total += int((terminated | truncated).sum())
            contact_peak = max(contact_peak, int(env._a6_ever_contact.sum()))
            actor_active_peak = max(
                actor_active_peak, int((action.real_steps > 30).sum())
            )
            if (step + 1) % 100 == 0 or step + 1 == args.steps:
                print(
                    f"step={step + 1} done_total={done_total} contact_peak={contact_peak} actor_active_peak={actor_active_peak} reward_mean={float(reward.mean()):.5f}",
                    flush=True,
                )
        if args.safe_actions:
            action.real_steps.fill_(31)
            action.process_actions(torch.ones(args.num_envs, 29, device=env.device))
            constrained = (env._a6_scene == 1) & ~env._a6_escaped
            lateral_ids = (
                action._guided_lateral_arm_ids + action._guided_lateral_hip_ids
            )
            arm_ids = [i for i in action._guided_arm_ids if i not in lateral_ids]
            hip_ids = [i for i in action._guided_hip_ids if i not in lateral_ids]
            assert torch.allclose(
                action.raw_action[constrained][:, arm_ids],
                torch.full_like(action.raw_action[constrained][:, arm_ids], 0.25),
            )
            assert torch.allclose(
                action.raw_action[constrained][:, hip_ids],
                torch.full_like(action.raw_action[constrained][:, hip_ids], 0.25),
            )
            assert torch.allclose(
                action.raw_action[constrained][:, lateral_ids],
                torch.full_like(
                    action.raw_action[constrained][:, lateral_ids],
                    1.0 if args.lateral_actions else 0.25,
                ),
            )
            assert torch.allclose(
                action.raw_action[~constrained],
                torch.ones_like(action.raw_action[~constrained]),
            )
        assert actor_active_peak > 0
        if args.steps >= 501:
            assert done_total > 0
    finally:
        env.close()


if __name__ == "__main__":
    main()
