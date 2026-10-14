"""Run a continuous mjlab rollout and verify HoST 23-DoF control boundaries."""

import argparse

import mujoco
import torch
from mjlab.envs import ManagerBasedRlEnv

from g1recovery_host import host_ground_env_cfg
from g1recovery_host.model import HOST_JOINT_NAMES
from g1recovery_host.task import reduce_assistance


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--num-envs", type=int, default=2)
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument(
        "--model-source",
        choices=("mjlab", "host_urdf", "host_urdf_capsule"),
        default="mjlab",
    )
    args = parser.parse_args()
    if args.steps < 32:
        parser.error("--steps must be at least 32 to cross the no-actuation phase")

    env = ManagerBasedRlEnv(
        host_ground_env_cfg(args.num_envs, model_source=args.model_source),
        device=args.device,
    )
    try:
        obs, _ = env.reset()
        robot = env.scene["robot"]
        model = env.sim.mj_model
        assert len(robot.joint_names) == len(robot.actuator_names) == 23
        assert tuple(robot.joint_names) == HOST_JOINT_NAMES
        assert set(robot.actuator_names) == set(HOST_JOINT_NAMES)
        action_term = env.action_manager.get_term("joint_pos")
        assert tuple(action_term._target_names) == HOST_JOINT_NAMES
        assert len(action_term._torso_ids) == (
            2 if args.model_source.startswith("host_urdf") else 1
        )
        assert (model.nq, model.nv, model.nu) == (30, 29, 23)
        assert obs["actor"].shape == (args.num_envs, 456)
        mass = float(model.body_mass.sum())
        if args.model_source.startswith("host_urdf"):
            assert abs(mass - 34.13385728) < 1e-4, mass
            assert int((model.geom_contype != 0).sum()) > 1
        if args.model_source == "host_urdf_capsule":
            assert int((model.geom_type == mujoco.mjtGeom.mjGEOM_CAPSULE).sum()) >= 4
        action = torch.full((args.num_envs, 23), 0.5, device=args.device)
        max_before = 0.0
        max_after = 0.0
        resets = 0
        for step in range(args.steps):
            obs, rewards, terminated, truncated, _ = env.step(action)
            assert torch.isfinite(obs["actor"]).all()
            assert torch.isfinite(rewards).all()
            resets += int((terminated | truncated).sum())
            torque = float(robot.data.qfrc_actuator.abs().max())
            if step <= 30:
                max_before = max(max_before, torque)
            else:
                max_after = max(max_after, torque)
            if step in (0, 30, 31, args.steps - 1):
                print(
                    f"step={step} torque_max={torque:.4f} "
                    f"root_height={float(robot.data.root_link_pos_w[:, 2].mean()):.4f} "
                    f"reward_mean={float(rewards.mean()):.4f} "
                    f"done={int((terminated | truncated).sum())}"
                )
            if step == 30 and args.model_source.startswith("host_urdf"):
                assert float(robot.data.root_link_pos_w[:, 2].min()) > -0.1, (
                    "HoST URDF robot fell through the floor"
                )
        assert max_before == 0.0, f"no-actuation phase had torque {max_before}"
        assert max_after > 0.0, "PD motor did not activate after 30 steps"
        action_term.real_steps[:] = 31
        action_term.process_actions(torch.full_like(action, 2.0))
        assert torch.all(action_term.raw_action == 2.0), (
            "Gaussian exploration actions must not be clipped at +/-1"
        )
        env._host_head_height = torch.ones(args.num_envs, device=args.device)
        previous_assist = action_term.assist_force.clone()
        previous_beta = action_term.beta.clone()
        reduce_assistance(env, torch.arange(args.num_envs, device=args.device))
        assert torch.allclose(
            action_term.assist_force, (previous_assist - 40.0).clamp_min(0.0)
        )
        assert torch.allclose(action_term.beta, (previous_beta - 0.02).clamp_min(0.25))
        if args.steps > 500:
            assert resets >= args.num_envs, "Expected the 10-second episode to reset"
        print(
            f"PASS source={args.model_source} mass={mass:.5f}kg, 23 joints, "
            f"456 observations, {args.steps} finite steps, delayed actuation, resets={resets}"
        )
    finally:
        env.close()


if __name__ == "__main__":
    main()
