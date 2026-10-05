"""Bounded added-output-only continuation of the successful 23-DoF HoST actor."""
import argparse
import copy
import hashlib
import json
import os
from dataclasses import asdict
from pathlib import Path

os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")

import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper

from src.tasks.host_recovery.config.g1.rl_cfg import unitree_g1_host_standup_ppo_runner_cfg
from src.tasks.host_recovery.minimal29 import ADDED_JOINTS, minimal29_env_cfg
from src.tasks.host_recovery.rl import HoSTOnPolicyRunner


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--bank", type=Path, required=True)
    p.add_argument("--log-dir", type=Path, required=True)
    p.add_argument("--num-envs", type=int, default=1024)
    p.add_argument("--iterations", type=int, default=200)
    p.add_argument("--seed", type=int, default=20261005)
    p.add_argument("--preflight", action="store_true")
    p.add_argument("--rollout-steps", type=int, default=100)
    a = p.parse_args()
    a.log_dir.mkdir(parents=True, exist_ok=False)
    cfg = minimal29_env_cfg(a.num_envs, a.seed, a.bank)
    agent = unitree_g1_host_standup_ppo_runner_cfg()
    agent.seed = a.seed
    agent.logger = "tensorboard"
    agent.num_steps_per_env = 32
    agent.max_iterations = a.iterations
    agent.save_interval = 50
    agent.algorithm.learning_rate = 1e-5
    agent.algorithm.schedule = "fixed"
    agent.algorithm.num_learning_epochs = 2
    agent.algorithm.entropy_coef = 0.0
    agent.actor.distribution_cfg.update(init_std=.03, max_std=.03)

    env = ManagerBasedRlEnv(cfg=cfg, device="cuda:0")
    try:
        wrapper = RslRlVecEnvWrapper(env)
        obs = wrapper.get_observations()
        assert obs["actor"].shape == (a.num_envs, 564)
        assert wrapper.num_actions == 29
        assert env.sim.data.qvel.abs().max() == 0
        history = obs["actor"].reshape(a.num_envs, 6, 94)
        assert (history - history[:, :1]).abs().max() < 1e-6

        runner = HoSTOnPolicyRunner(wrapper, asdict(agent), str(a.log_dir), "cuda:0")
        fresh_critic = copy.deepcopy(runner.alg.policy.critic.state_dict())
        fresh_critic_norm = copy.deepcopy(runner.privileged_obs_normalizer.state_dict())
        runner.load(str(a.checkpoint), load_optimizer=False,
                    load_cfg={"actor": True, "critic": False}, map_location="cuda:0")
        runner.alg.policy.critic.load_state_dict(fresh_critic)
        runner.privileged_obs_normalizer.load_state_dict(fresh_critic_norm)
        runner.current_learning_iteration = 0
        runner.obs_normalizer.until = int(runner.obs_normalizer.count.item())
        with torch.no_grad():
            runner.alg.policy.std.fill_(.03)
        runner.alg.policy.std.requires_grad_(False)

        actor = runner.alg.policy.actor
        for param in actor.parameters():
            param.requires_grad_(False)
        final = next(layer for layer in reversed(actor) if isinstance(layer, torch.nn.Linear))
        final.weight.requires_grad_(True)
        final.bias.requires_grad_(True)
        names = env.scene["robot"].joint_names
        added_ids = [names.index(name) for name in ADDED_JOINTS]
        row_mask = torch.zeros(29, device=env.device)
        row_mask[added_ids] = 1
        final.weight.register_hook(lambda grad: grad * row_mask[:, None])
        final.bias.register_hook(lambda grad: grad * row_mask)

        launch = {
            "source": str(a.checkpoint.resolve()),
            "source_sha256": hashlib.sha256(a.checkpoint.read_bytes()).hexdigest(),
            "bank_sha256": hashlib.sha256(a.bank.read_bytes()).hexdigest(),
            "num_envs": a.num_envs, "iterations": a.iterations,
            "amp_discriminator": False, "teacher": False, "pull_force_n": 0,
            "shared_23_actor_frozen": True, "trainable_output_rows": added_ids,
            "critic_optimizer_fresh": True, "static_history": True,
        }
        (a.log_dir / "launch.json").write_text(json.dumps(launch, indent=2))

        policy = runner.get_inference_policy(device="cuda:0")
        done_count = 0
        with torch.no_grad():
            for _ in range(a.rollout_steps):
                obs, reward, dones, _ = wrapper.step(policy(obs["actor"]))
                assert torch.isfinite(reward).all()
                assert all(torch.isfinite(value).all() for value in obs.values())
                assert torch.isfinite(env.sim.data.qvel).all()
                done_count += int(dones.sum())
        if a.rollout_steps >= int(cfg.episode_length_s / env.step_dt):
            assert done_count > 0, "full-episode smoke never reached a termination"
        print("HOST29_MINIMAL_PREFLIGHT_PASS", flush=True)
        if a.preflight:
            return
        before = {k: v.detach().clone() for k, v in actor.state_dict().items()}
        runner.learn(num_learning_iterations=a.iterations, init_at_random_ep_len=False)
        after = actor.state_dict()
        for key in before:
            if key not in ("6.weight", "6.bias"):
                assert torch.equal(before[key], after[key]), key
        shared = [i for i in range(29) if i not in added_ids]
        assert torch.equal(before["6.weight"][shared], after["6.weight"][shared])
        assert torch.equal(before["6.bias"][shared], after["6.bias"][shared])
        runner.save(str(a.log_dir / f"model_{a.iterations}_final.pt"))
        print("HOST29_MINIMAL_ADDED_ONLY_DONE", flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    main()
