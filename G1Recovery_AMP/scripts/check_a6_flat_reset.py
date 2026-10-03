"""Validate A6 flat-bank state mapping and five-frame history backfill."""

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
from g1recovery_amp.tasks.recovery.mdp import reset_from_a6_flat_curriculum


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-envs", type=int, default=32)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    cfg = g1_recovery_a6_env_cfg(play=True)
    flat_reset = cfg.events.pop("a6_three_scene_curriculum_reset")
    flat_reset.func = reset_from_a6_flat_curriculum
    cfg.events["a6_flat_curriculum_reset"] = flat_reset
    cfg.scene.num_envs = args.num_envs
    cfg.seed = args.seed
    env = ManagerBasedRlEnv(cfg=cfg, device=args.device)
    try:
        observations, _ = env.reset(seed=args.seed)
        actor = observations["actor"]
        if actor.shape != (args.num_envs, 480):
            raise RuntimeError(f"unexpected actor shape: {tuple(actor.shape)}")

        stage = env._a6_reset_stage
        low_selected = env._a6_reset_bank_index
        natural_selected = env._a6_reset_natural_bank_index
        low_mask = stage == 0
        natural_mask = stage != 0
        if torch.any(low_selected[low_mask] < 0) or torch.any(
            natural_selected[natural_mask] < 0
        ):
            raise RuntimeError("some environments did not receive a curriculum row")
        if torch.any(low_selected[natural_mask] >= 0) or torch.any(
            natural_selected[low_mask] >= 0
        ):
            raise RuntimeError("an environment received rows from two reset banks")

        low_bank = load_a6_multiterrain_bank()
        natural_bank = load_a6_natural_curriculum_bank()
        expected_qpos = torch.empty(
            (args.num_envs, 36), dtype=torch.float32, device=env.device
        )
        expected_qpos[low_mask] = torch.as_tensor(
            low_bank.qpos[low_selected[low_mask].cpu().numpy()],
            dtype=torch.float32,
            device=env.device,
        )
        expected_qpos[natural_mask] = torch.as_tensor(
            natural_bank.qpos[natural_selected[natural_mask].cpu().numpy()],
            dtype=torch.float32,
            device=env.device,
        )
        robot = env.scene["robot"]
        torch.testing.assert_close(
            robot.data.root_link_pos_w - env.scene.env_origins,
            expected_qpos[:, :3],
        )
        torch.testing.assert_close(robot.data.root_link_quat_w, expected_qpos[:, 3:7])
        torch.testing.assert_close(robot.data.joint_pos, expected_qpos[:, 7:])
        torch.testing.assert_close(
            robot.data.root_link_vel_w, torch.zeros_like(robot.data.root_link_vel_w)
        )
        torch.testing.assert_close(
            robot.data.joint_vel, torch.zeros_like(robot.data.joint_vel)
        )

        low_rows = low_selected[low_mask].cpu().numpy()
        torch.testing.assert_close(
            torch.as_tensor(low_bank.direction[low_rows], device=env.device),
            env._a6_reset_direction[low_mask],
        )
        torch.testing.assert_close(
            torch.as_tensor(low_bank.source[low_rows], device=env.device),
            env._a6_reset_source[low_mask],
        )
        if (low_bank.stratum[low_rows] != 0).any():
            raise RuntimeError("a low reset row did not come from flat stratum 0")
        for stage_id, stage_name in ((1, "middle"), (2, "late")):
            selected = natural_selected[stage == stage_id].cpu().numpy()
            if (natural_bank.stages[selected] != stage_name).any():
                raise RuntimeError(
                    f"a {stage_name} environment sampled the wrong natural stage"
                )

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

        stage_counts = torch.bincount(stage, minlength=3).cpu().tolist()
        direction_counts = torch.bincount(
            env._a6_reset_direction[low_mask], minlength=4
        ).cpu().tolist()
        low_source_counts = torch.bincount(
            env._a6_reset_source[low_mask], minlength=2
        ).cpu().tolist()

        # A real training run resets only the environments that finish. Check
        # that this does not alter the state or bookkeeping of the others.
        partial_ids = torch.arange(0, args.num_envs, 2, device=env.device)
        untouched_ids = torch.arange(1, args.num_envs, 2, device=env.device)
        untouched_qpos = torch.cat(
            (
                robot.data.root_link_pos_w[untouched_ids]
                - env.scene.env_origins[untouched_ids],
                robot.data.root_link_quat_w[untouched_ids],
                robot.data.joint_pos[untouched_ids],
            ),
            dim=-1,
        ).clone()
        untouched_low_rows = env._a6_reset_bank_index[untouched_ids].clone()
        untouched_natural_rows = env._a6_reset_natural_bank_index[
            untouched_ids
        ].clone()
        env.reset(env_ids=partial_ids)
        current_untouched_qpos = torch.cat(
            (
                robot.data.root_link_pos_w[untouched_ids]
                - env.scene.env_origins[untouched_ids],
                robot.data.root_link_quat_w[untouched_ids],
                robot.data.joint_pos[untouched_ids],
            ),
            dim=-1,
        )
        torch.testing.assert_close(current_untouched_qpos, untouched_qpos)
        torch.testing.assert_close(
            env._a6_reset_bank_index[untouched_ids], untouched_low_rows
        )
        torch.testing.assert_close(
            env._a6_reset_natural_bank_index[untouched_ids],
            untouched_natural_rows,
        )

        print(
            "a6_flat_curriculum_reset_valid=True, "
            f"num_envs={args.num_envs}, actor_shape={tuple(actor.shape)}, "
            f"stage_counts={dict(zip(A6_STAGE_NAMES, stage_counts, strict=True))}, "
            f"low_direction_counts={direction_counts}, "
            f"low_source_counts={low_source_counts}, velocities_zero=True, "
            "actor_history_repeated=True, partial_reset_isolated=True"
        )
    finally:
        env.close()


if __name__ == "__main__":
    main()
