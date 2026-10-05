"""Read an existing AMP checkpoint with this package; no playback files written."""

from __future__ import annotations

import argparse
from dataclasses import asdict

import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls

from g1recovery_amp.tasks.recovery import (
    A6_G_MINUS_TASK_ID,
    A6_G_PLUS_TASK_ID,
    A6_TASK_ID,
    DEV_TASK_ID,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint")
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--task-id",
        choices=(DEV_TASK_ID, A6_TASK_ID, A6_G_MINUS_TASK_ID, A6_G_PLUS_TASK_ID),
        default=DEV_TASK_ID,
        help="environment contract used for the compatibility check",
    )
    parser.add_argument(
        "--full", action="store_true", help="restore actor, critic, and AMP optimizers"
    )
    args = parser.parse_args()

    env_cfg = load_env_cfg(args.task_id, play=True)
    agent_cfg = load_rl_cfg(args.task_id)
    env_cfg.scene.num_envs = 1
    env = ManagerBasedRlEnv(cfg=env_cfg, device=args.device)
    wrapped = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    try:
        runner_cls = load_runner_cls(args.task_id)
        if runner_cls is None:
            raise RuntimeError("AMP task has no registered runner")
        runner = runner_cls(
            wrapped, asdict(agent_cfg), log_dir=None, device=args.device
        )
        runner.load(
            args.checkpoint,
            load_cfg=None if args.full else {"actor": True},
            strict=True,
            map_location=args.device,
        )
        policy = runner.get_inference_policy(device=args.device)
        observations = wrapped.get_observations()
        with torch.inference_mode():
            actions = policy(observations)
            next_observations, rewards, _, _ = wrapped.step(actions)
        if not torch.isfinite(actions).all() or not torch.isfinite(rewards).all():
            raise FloatingPointError("checkpoint inference produced NaN/Inf")
        if any(not torch.isfinite(x).all() for x in next_observations.values()):
            raise FloatingPointError("checkpoint step produced NaN/Inf observation")
        a6_clock = ""
        if hasattr(env, "_a6_curriculum_start_counter"):
            start = int(env._a6_curriculum_start_counter)
            age = int(env.common_step_counter) - start
            a6_clock = f", a6_curriculum_start={start}, a6_curriculum_age={age}"
        print(
            f"checkpoint_compatible=True, full_restore={args.full}, task={args.task_id}, "
            f"action_shape={tuple(actions.shape)}, reward={float(rewards[0]):.6g}"
            f"{a6_clock}"
        )
    finally:
        wrapped.close()


if __name__ == "__main__":
    main()
