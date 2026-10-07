"""Preflight and train the paired historical-A6 HoST G-/G+ arms."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import subprocess
from dataclasses import asdict
from pathlib import Path

os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")

import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.utils.os import dump_yaml

from src.tasks.host_recovery.a6_runtime import a6_env_cfg, a6_reset_counts
from src.tasks.host_recovery.config.g1.rl_cfg import unitree_g1_host_standup_ppo_runner_cfg
from src.tasks.host_recovery.rl import HoSTOnPolicyRunner


EXPECTED_BANKS = {
  "outputs/multiterrain_bank/train.npz": "287eae8e8840c1b3281e7010a84182b824fefacd34439c4f993e9f130af1027a",
  "datasets/reset_banks/natural_curriculum_v1/train.npz": "e9f94540520d7927dee01150a0f0bae39ad2aad5ae008b1c7c71f521650e44b9",
  "datasets/reset_banks/procedural_low_v1/train.npz": "e289bed8d1c93e87fbdcf969fc5fc741f2d8500a0908ed145d2a4e02e7fe116d",
}


def sha256(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as stream:
    for block in iter(lambda: stream.read(1024 * 1024), b""):
      digest.update(block)
  return digest.hexdigest()


def tensor_state_sha256(state: dict[str, torch.Tensor]) -> str:
  """Stable digest for an in-memory module state dict."""
  digest = hashlib.sha256()
  for name in sorted(state):
    value = state[name].detach().cpu().contiguous()
    digest.update(name.encode("utf-8"))
    digest.update(str(value.dtype).encode("ascii"))
    digest.update(str(tuple(value.shape)).encode("ascii"))
    digest.update(value.numpy().tobytes())
  return digest.hexdigest()


def audit_reset(env: ManagerBasedRlEnv) -> dict:
  counts = a6_reset_counts(env)
  if float(env.sim.data.qvel.abs().max()) >= 1e-6:
    raise AssertionError("A6 reset velocities are not zero")
  if env.num_envs >= 32:
    ids = torch.tensor((0, env.num_envs // 4, env.num_envs // 2, 3 * env.num_envs // 4), device=env.device)
    untouched = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
    untouched[ids] = False
    before_qpos = env.sim.data.qpos[untouched].clone()
    before_mass = env.sim.model.body_mass[untouched].clone()
    before_gain = env._a6_gain_factors[untouched].clone()
    env._reset_idx(ids)
    assert torch.equal(before_qpos, env.sim.data.qpos[untouched])
    assert torch.equal(before_mass, env.sim.model.body_mass[untouched])
    assert torch.equal(before_gain, env._a6_gain_factors[untouched])
  return {"counts": counts, "partial_reset_untouched": env.num_envs - 4}


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--arm", choices=("gminus", "gplus"), required=True)
  parser.add_argument("--checkpoint", type=Path, required=True)
  parser.add_argument("--log-dir", type=Path, required=True)
  parser.add_argument("--num-envs", type=int, default=4096)
  parser.add_argument("--updates", type=int, default=10000)
  parser.add_argument("--seed", type=int, default=20261013)
  parser.add_argument("--learning-rate", type=float, default=1.0e-4)
  parser.add_argument("--entropy-coef", type=float, default=0.0)
  parser.add_argument("--noise-std", type=float, default=0.8)
  parser.add_argument("--max-noise-std", type=float, default=0.8)
  parser.add_argument("--save-interval", type=int, default=500)
  parser.add_argument(
    "--train-action-std",
    action="store_true",
    help="Allow PPO to optimize exploration std; V4 freezes it by default.",
  )
  parser.add_argument(
    "--update-actor-normalizer",
    action="store_true",
    help="Update the inherited actor normalizer; V4 freezes it by default.",
  )
  parser.add_argument("--forward-smoke", action="store_true")
  parser.add_argument("--preflight", action="store_true")
  args = parser.parse_args()

  args.log_dir.mkdir(parents=True, exist_ok=False)
  bank_hashes = {name: sha256(Path(name)) for name in EXPECTED_BANKS}
  assert bank_hashes == EXPECTED_BANKS, bank_hashes
  checkpoint_hash = sha256(args.checkpoint)
  group = "G+" if args.arm == "gplus" else "G-"
  cfg = a6_env_cfg(seed=args.seed, dynamics=True, group=group)
  cfg.scene.num_envs = args.num_envs
  cfg.seed = args.seed
  agent = unitree_g1_host_standup_ppo_runner_cfg()
  agent.seed = args.seed
  agent.logger = "tensorboard"
  agent.experiment_name = f"g1_host_a6_{args.arm}"
  agent.max_iterations = args.updates
  agent.num_steps_per_env = 24
  agent.save_interval = args.save_interval
  agent.algorithm.learning_rate = args.learning_rate
  agent.algorithm.entropy_coef = args.entropy_coef
  agent.actor.distribution_cfg["init_std"] = args.noise_std
  agent.actor.distribution_cfg["max_std"] = args.max_noise_std
  dump_yaml(args.log_dir / "env.yaml", asdict(cfg))
  dump_yaml(args.log_dir / "agent.yaml", asdict(agent))

  env = ManagerBasedRlEnv(cfg=cfg, device="cuda:0")
  try:
    assert abs(env.physics_dt - 0.002) < 1e-9
    assert abs(env.step_dt - 0.02) < 1e-9
    env.reset()
    reset_audit = audit_reset(env)
    wrapper = RslRlVecEnvWrapper(env, clip_actions=agent.clip_actions)
    runner = HoSTOnPolicyRunner(wrapper, asdict(agent), str(args.log_dir), "cuda:0")
    # V4 is actor-only transfer.  Preserve the freshly seeded critic and critic
    # normalizer while loading the source actor and actor normalizer.  The old
    # critic and Adam moments encode a different reward and caused V3 to erase
    # the four-posture recovery skill within 500 updates.
    fresh_critic = copy.deepcopy(runner.alg.policy.critic.state_dict())
    fresh_critic_normalizer = copy.deepcopy(runner.privileged_obs_normalizer.state_dict())
    runner.load(
      str(args.checkpoint),
      load_optimizer=False,
      load_cfg={"actor": True, "critic": False},
      map_location="cuda:0",
    )
    runner.alg.policy.critic.load_state_dict(fresh_critic)
    runner.privileged_obs_normalizer.load_state_dict(fresh_critic_normalizer)

    source = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    source_actor = source["actor_state_dict"]
    policy_state = runner.alg.policy.state_dict()
    for name, value in source_actor.items():
      if name == "distribution.std_param":
        continue
      mapped = "actor." + name.removeprefix("mlp.")
      assert torch.equal(policy_state[mapped].detach().cpu(), value.cpu()), mapped
    assert len(runner.alg.optimizer.state) == 0

    with torch.no_grad():
      runner.alg.policy.std.fill_(args.noise_std)
    runner.alg.policy.std.requires_grad_(args.train_action_std)
    assert runner.alg.policy.max_noise_std == args.max_noise_std
    if not args.update_actor_normalizer:
      runner.obs_normalizer.until = int(runner.obs_normalizer.count.item())

    # A6 is a new adaptation clock.  The HoST environment step and auxiliary
    # curriculum are retained, while optimization starts at A6 update zero.
    runner.current_learning_iteration = 0
    env._a6_adaptation_start_step = int(env.common_step_counter)
    action = runner.get_inference_policy(device="cuda:0")
    observations = wrapper.get_observations()
    with torch.inference_mode():
      actions = action(observations["actor"])
    assert actions.shape == (args.num_envs, 29)
    assert torch.isfinite(actions).all()

    launch = {
      "arm": args.arm,
      "checkpoint": str(args.checkpoint.resolve()),
      "checkpoint_sha256": checkpoint_hash,
      "method": "existing native 29-joint HoST port (single-critic PPO; not full paper HoST)",
      "checkpoint_state": "actor and actor observation normalizer retained; critic, critic normalizer and optimizer freshly initialized",
      "actor_exact_at_launch": True,
      "fresh_critic": True,
      "fresh_optimizer": True,
      "critic_state_sha256": tensor_state_sha256(runner.alg.policy.critic.state_dict()),
      "noise_std": args.noise_std,
      "max_noise_std": args.max_noise_std,
      "action_std_frozen": not args.train_action_std,
      "actor_normalizer_frozen": not args.update_actor_normalizer,
      "learning_rate": args.learning_rate,
      "entropy_coef": args.entropy_coef,
      "adaptation_clock_reset": True,
      "a6_adaptation_start_step": env._a6_adaptation_start_step,
      "native_env_step_counter_retained": int(env.common_step_counter),
      "host_auxiliary_curriculum_retained": True,
      "num_envs": args.num_envs,
      "updates": args.updates,
      "seed": args.seed,
      "physics_dt": env.physics_dt,
      "control_dt": env.step_dt,
      "bank_sha256": bank_hashes,
      "reset_audit": reset_audit,
      "torch_compile_disabled": os.environ.get("TORCHDYNAMO_DISABLE") == "1",
      "git_commit": subprocess.check_output(("git", "rev-parse", "HEAD"), text=True).strip(),
    }
    (args.log_dir / "launch.json").write_text(json.dumps(launch, indent=2), encoding="utf-8")
    print("A6_FORWARD_SMOKE_OK", json.dumps(reset_audit), flush=True)
    if args.forward_smoke:
      return

    if args.preflight:
      # Cross one complete 10 s episode, then verify a reset has occurred and
      # all reward/observation channels remain finite.
      initial_counter = env.common_step_counter
      for _ in range(510):
        with torch.inference_mode():
          actions = action(observations["actor"])
          observations, reward, _, _ = wrapper.step(actions)
        assert torch.isfinite(reward).all()
        assert all(torch.isfinite(value).all() for value in observations.values())
      assert env.common_step_counter >= initial_counter + 510
      (args.log_dir / "preflight.json").write_text(
        json.dumps({"control_steps": 510, "crossed_full_episode": True}, indent=2),
        encoding="utf-8",
      )
      print("A6_FULL_EPISODE_PREFLIGHT_OK", flush=True)
      return

    runner.save(str(args.log_dir / "model_0.pt"))
    runner.learn(num_learning_iterations=args.updates, init_at_random_ep_len=False)
    runner.save(str(args.log_dir / f"model_{args.updates}_final.pt"))
    print("A6_TRAINING_DONE", args.arm, flush=True)
  finally:
    env.close()


if __name__ == "__main__":
  main()
