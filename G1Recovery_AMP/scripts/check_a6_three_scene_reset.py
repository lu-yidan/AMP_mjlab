"""Validate A6 flat/guided/free reset mapping and plate placement."""

from __future__ import annotations

import argparse

import torch
from mjlab.envs import ManagerBasedRlEnv

from g1recovery_amp.a6_reset_assets import (
    A6_STAGE_NAMES,
    load_a6_multiterrain_bank,
    load_a6_natural_curriculum_bank,
)
from g1recovery_amp.tasks.recovery.a6_env_cfg import g1_recovery_a6_env_cfg
from g1recovery_amp.tasks.recovery.a6_scene import (
    A6_PLATE_CLEARANCE,
    A6_PLATE_HALF_SIZE,
)
from g1recovery_amp.tasks.recovery.mdp import A6_SCENE_NAMES, audit_a6_dynamics
from g1recovery_amp.tasks.recovery.mdp.a6_resets import _robot_collision_top


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-envs", type=int, default=32)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--training-config",
        action="store_true",
        help="enable frozen A6 training randomization instead of nominal playback",
    )
    args = parser.parse_args()

    cfg = g1_recovery_a6_env_cfg(play=not args.training_config)
    cfg.scene.num_envs = args.num_envs
    cfg.seed = args.seed
    env = ManagerBasedRlEnv(cfg=cfg, device=args.device)
    try:
        observations, _ = env.reset(seed=args.seed)
        dynamics_audit = audit_a6_dynamics(env)
        actor = observations["actor"]
        if actor.shape != (args.num_envs, 480):
            raise RuntimeError(f"unexpected actor shape: {tuple(actor.shape)}")

        scene = env._a6_reset_scene
        stage = env._a6_reset_stage
        low_rows = env._a6_reset_bank_index
        natural_rows = env._a6_reset_natural_bank_index
        low_bank = load_a6_multiterrain_bank()
        natural_bank = load_a6_natural_curriculum_bank()

        low_mask = stage == 0
        low_indices = low_rows[low_mask].cpu().numpy()
        torch.testing.assert_close(
            torch.as_tensor(low_bank.stratum[low_indices], device=env.device),
            scene[low_mask],
        )
        for stage_id, stage_name in ((1, "middle"), (2, "late")):
            selected = natural_rows[stage == stage_id].cpu().numpy()
            if (natural_bank.stages[selected] != stage_name).any():
                raise RuntimeError(f"{stage_name} environments sampled wrong rows")

        robot = env.scene["robot"]
        torch.testing.assert_close(
            robot.data.root_link_vel_w, torch.zeros_like(robot.data.root_link_vel_w)
        )
        torch.testing.assert_close(
            robot.data.joint_vel, torch.zeros_like(robot.data.joint_vel)
        )

        all_ids = torch.arange(args.num_envs, device=env.device)
        collision_top = _robot_collision_top(env, all_ids)
        plate_geom_ids = torch.stack(env._a6_plate_geom_ids)
        active_geom = plate_geom_ids[(scene - 1).clamp(0, 1)]
        plate_center_z = env.sim.data.geom_xpos[all_ids, active_geom, 2]
        active = scene > 0
        bottom_gap = (
            plate_center_z[active]
            - A6_PLATE_HALF_SIZE[2]
            - collision_top[active]
        )
        torch.testing.assert_close(
            bottom_gap,
            torch.full_like(bottom_gap, A6_PLATE_CLEARANCE),
            rtol=0.0,
            atol=1.0e-5,
        )
        active_mass = env._a6_plate_mass[active]
        if args.training_config:
            if torch.any((active_mass < 4.0) | (active_mass > 6.0)):
                raise RuntimeError("initial plate mass is outside the 4-6 kg range")
        else:
            torch.testing.assert_close(active_mass, torch.full_like(active_mass, 6.0))

        actor_history = env.observation_manager._group_obs_term_history_buffer["actor"]
        for term_name, buffer in actor_history.items():
            history = buffer.buffer
            torch.testing.assert_close(
                history,
                history[:, :1].expand_as(history),
                msg=lambda message, name=term_name: (
                    f"actor history was not backfilled for {name}: {message}"
                ),
            )

        partial_ids = torch.arange(0, args.num_envs, 2, device=env.device)
        untouched_ids = torch.arange(1, args.num_envs, 2, device=env.device)
        untouched_qpos = env.sim.data.qpos[untouched_ids].clone()
        untouched_mocap = env.sim.data.mocap_pos[untouched_ids].clone()
        untouched_mass = env.sim.model.body_mass[untouched_ids].clone()
        untouched_low_rows = low_rows[untouched_ids].clone()
        untouched_natural_rows = natural_rows[untouched_ids].clone()
        env.reset(env_ids=partial_ids)
        torch.testing.assert_close(env.sim.data.qpos[untouched_ids], untouched_qpos)
        torch.testing.assert_close(
            env.sim.data.mocap_pos[untouched_ids], untouched_mocap
        )
        torch.testing.assert_close(
            env.sim.model.body_mass[untouched_ids], untouched_mass
        )
        torch.testing.assert_close(low_rows[untouched_ids], untouched_low_rows)
        torch.testing.assert_close(
            natural_rows[untouched_ids], untouched_natural_rows
        )

        scene_counts = torch.bincount(scene, minlength=3).cpu().tolist()
        stage_counts = torch.bincount(stage, minlength=3).cpu().tolist()
        source_counts = torch.bincount(
            env._a6_reset_source, minlength=2
        ).cpu().tolist()
        body_range = (
            float(env._a6_body_factors.min()),
            float(env._a6_body_factors.max()),
        )
        gain_range = (
            float(env._a6_gain_factors.min()),
            float(env._a6_gain_factors.max()),
        )
        lag_values = sorted(set(env._a6_command_lag.cpu().tolist()))
        nominal_count = int(env._a6_dynamics_nominal.sum())
        print(
            "a6_three_scene_reset_valid=True, "
            f"num_envs={args.num_envs}, actor_shape={tuple(actor.shape)}, "
            f"scene_counts={dict(zip(A6_SCENE_NAMES, scene_counts, strict=True))}, "
            f"stage_counts={dict(zip(A6_STAGE_NAMES, stage_counts, strict=True))}, "
            f"source_counts={source_counts}, plate_gap_m={A6_PLATE_CLEARANCE}, "
            f"plate_mass_mode={'4_to_6' if args.training_config else 'fixed_6'}, "
            f"body_factor_range={body_range}, gain_factor_range={gain_range}, "
            f"lag_steps={lag_values}, nominal_count={nominal_count}, "
            f"dynamics_audit={dynamics_audit}, velocities_zero=True, "
            "actor_history_repeated=True, partial_reset_isolated=True"
        )
    finally:
        env.close()


if __name__ == "__main__":
    main()
