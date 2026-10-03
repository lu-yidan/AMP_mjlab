"""Run short real G1 PPO+AMP updates without writing logs or checkpoints."""

from __future__ import annotations

import argparse
import math
from dataclasses import asdict

import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg
from torch import nn

from g1recovery_amp.tasks.recovery import DEV_TASK_ID
from g1recovery_amp.tasks.recovery.rl import AmpPPO


def _snapshot_parameters(module: nn.Module) -> list[torch.Tensor]:
    """Copy trainable parameters so a later optimizer change can be measured."""

    return [parameter.detach().clone() for parameter in module.parameters()]


def _maximum_parameter_change(
    before: list[torch.Tensor],
    module: nn.Module,
) -> float:
    """Return the largest absolute change over all trainable parameters."""

    after = list(module.parameters())
    if len(before) != len(after):
        raise RuntimeError("parameter list changed during the AMP smoke update")
    return max(
        float((new.detach() - old).abs().max())
        for old, new in zip(before, after, strict=True)
    )


def run_amp_training_smoke(
    *,
    num_envs: int = 1,
    updates: int = 1,
    device: str = "cuda:0",
    seed: int = 0,
) -> dict[str, float]:
    """Collect registered rollouts and apply consecutive PPO+AMP updates."""

    if num_envs <= 0:
        raise ValueError(f"num_envs must be positive, got {num_envs}")
    if updates <= 0:
        raise ValueError(f"updates must be positive, got {updates}")

    env_cfg = load_env_cfg(DEV_TASK_ID)
    agent_cfg_object = load_rl_cfg(DEV_TASK_ID)
    env_cfg.scene.num_envs = num_envs
    env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
    wrapped = RslRlVecEnvWrapper(
        env,
        clip_actions=agent_cfg_object.clip_actions,
    )
    try:
        wrapped.seed(seed)
        agent_cfg = asdict(agent_cfg_object)
        runner = MjlabOnPolicyRunner(
            wrapped,
            agent_cfg,
            log_dir=None,
            device=device,
        )
        algorithm = runner.alg
        if not isinstance(algorithm, AmpPPO):
            raise TypeError(f"expected AmpPPO, got {type(algorithm).__name__}")

        actor_before = _snapshot_parameters(algorithm._raw_actor)
        critic_before = _snapshot_parameters(algorithm._raw_critic)
        discriminator_before = _snapshot_parameters(algorithm.amp_discriminator)

        algorithm.train_mode()
        observations = wrapped.get_observations().to(device)
        task_rewards: list[float] = []
        style_rewards: list[float] = []
        blended_rewards: list[float] = []
        action_maximum = 0.0
        done_count = 0
        update_metrics: list[dict[str, float]] = []

        for update_index in range(updates):
            with torch.inference_mode():
                for _ in range(agent_cfg_object.num_steps_per_env):
                    actions = algorithm.act(observations)
                    action_maximum = max(action_maximum, float(actions.abs().max()))
                    observations, rewards, dones, extras = wrapped.step(actions)
                    observations = observations.to(device)
                    rewards = rewards.to(device)
                    dones = dones.to(device)
                    for name, value in observations.items():
                        if not torch.isfinite(value).all():
                            raise FloatingPointError(
                                f"observation group {name!r} contains NaN or Inf"
                            )
                    if not torch.isfinite(rewards).all():
                        raise FloatingPointError("task reward contains NaN or Inf")

                    algorithm.process_env_step(observations, rewards, dones, extras)
                    if (
                        algorithm.style_rewards is None
                        or algorithm.rewards_lerp is None
                    ):
                        raise RuntimeError("AmpPPO did not expose blended rewards")
                    task_rewards.extend(rewards.tolist())
                    style_rewards.extend(algorithm.style_rewards.tolist())
                    blended_rewards.extend(algorithm.rewards_lerp.tolist())
                    done_count += int(dones.sum())

                algorithm.compute_returns(observations)

            expected_replay_length = (
                update_index + 1
            ) * agent_cfg_object.num_steps_per_env
            if algorithm.disc_obs_buffer.current_length != expected_replay_length:
                raise RuntimeError(
                    "live AMP replay buffer length mismatch: "
                    f"expected {expected_replay_length}, got "
                    f"{algorithm.disc_obs_buffer.current_length}"
                )
            if (
                algorithm.disc_demo_obs_buffer.current_length
                != expected_replay_length
            ):
                raise RuntimeError(
                    "demonstration AMP replay buffer length mismatch: "
                    f"expected {expected_replay_length}, got "
                    f"{algorithm.disc_demo_obs_buffer.current_length}"
                )

            metrics = algorithm.update()
            if not all(math.isfinite(value) for value in metrics.values()):
                raise FloatingPointError(
                    f"update {update_index + 1} contains NaN or Inf: {metrics}"
                )
            if algorithm.storage.step != 0:
                raise RuntimeError(
                    f"PPO storage was not cleared after update {update_index + 1}"
                )
            update_metrics.append(metrics)
            print(f"update={update_index + 1}/{updates}, metrics={metrics}")

        actor_change = _maximum_parameter_change(
            actor_before,
            algorithm._raw_actor,
        )
        critic_change = _maximum_parameter_change(
            critic_before,
            algorithm._raw_critic,
        )
        discriminator_change = _maximum_parameter_change(
            discriminator_before,
            algorithm.amp_discriminator,
        )
        if min(actor_change, critic_change, discriminator_change) <= 0.0:
            raise RuntimeError(
                "one or more networks did not change: "
                f"actor={actor_change}, critic={critic_change}, "
                f"discriminator={discriminator_change}"
            )

        result = {
            "num_envs": float(num_envs),
            "updates": float(updates),
            "rollout_steps": float(agent_cfg_object.num_steps_per_env),
            "done_count": float(done_count),
            "raw_action_abs_max": action_maximum,
            "task_reward_min": min(task_rewards),
            "task_reward_max": max(task_rewards),
            "style_reward_min": min(style_rewards),
            "style_reward_max": max(style_rewards),
            "blended_reward_min": min(blended_rewards),
            "blended_reward_max": max(blended_rewards),
            "actor_parameter_max_change": actor_change,
            "critic_parameter_max_change": critic_change,
            "discriminator_parameter_max_change": discriminator_change,
            **update_metrics[-1],
        }
        print(f"task={DEV_TASK_ID}")
        print(f"device={device}, num_envs={num_envs}")
        print(f"rollout_steps={agent_cfg_object.num_steps_per_env}")
        print(
            "amp_replay_lengths="
            f"({algorithm.disc_obs_buffer.current_length}, "
            f"{algorithm.disc_demo_obs_buffer.current_length})"
        )
        print(f"training_smoke_result={result}")
        return result
    finally:
        wrapped.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument("--updates", type=int, default=1)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    run_amp_training_smoke(
        num_envs=args.num_envs,
        updates=args.updates,
        device=args.device,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
