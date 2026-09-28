"""Fine-tune no-load HoST with SMP get-up rewards, supine by default."""

import json
import os
from dataclasses import asdict
from pathlib import Path

import mjlab.tasks
import src.tasks
import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.utils.os import dump_yaml
from mjlab.utils.torch import configure_torch_backends


def main():
  checkpoint = Path(os.environ["CKPT"]).resolve(strict=True)
  log_dir = Path(os.environ["LOG_DIR"])
  log_dir.mkdir(parents=True, exist_ok=False)
  configure_torch_backends()
  task = os.environ.get("SMP_TASK", "Unitree-G1-HoST-SupineSmp")
  posture_by_task = {
    "Unitree-G1-HoST-SupineSmp": "supine",
    "Unitree-G1-HoST-ProneSmp": "prone",
  }
  if task not in posture_by_task:
    raise ValueError(f"Unsupported SMP task: {task}")
  posture = posture_by_task[task]
  cfg = load_env_cfg(task)
  cfg.scene.num_envs = int(os.environ.get("NUM_ENVS", "4096"))
  cfg.seed = 42
  agent = load_rl_cfg(task)
  agent.seed = 42
  agent.algorithm.learning_rate = float(os.environ.get("LEARNING_RATE", "5e-5"))
  agent.algorithm.entropy_coef = float(os.environ.get("ENTROPY_COEF", "0.01"))
  agent.save_interval = int(os.environ.get("SAVE_INTERVAL", "500"))
  agent.logger = "tensorboard"
  iterations = int(os.environ.get("MAX_ITER", "6000"))
  agent.max_iterations = iterations
  env = RslRlVecEnvWrapper(ManagerBasedRlEnv(cfg=cfg, device="cuda:0"))
  runner = load_runner_cls(task)(env, asdict(agent), str(log_dir), "cuda:0")
  runner.load(str(checkpoint), load_optimizer=False)
  runner.current_learning_iteration = 0
  freeze_normalizer = os.environ.get("FREEZE_OBS_NORMALIZER", "0") == "1"
  if freeze_normalizer:
    # The source checkpoint's actor normalizer has seen far fewer samples
    # than one 4096-env rollout. Updating it on a single-posture batch shifts
    # the policy input enough to destroy the pretrained recovery behavior.
    runner.obs_normalizer.until = int(runner.obs_normalizer.count.item())
    runner.privileged_obs_normalizer.until = int(runner.privileged_obs_normalizer.count.item())
  noise_std = os.environ.get("ACTION_NOISE_STD")
  if noise_std is not None:
    with torch.no_grad():
      runner.alg.policy.std.fill_(float(noise_std))
  if os.environ.get("FREEZE_ACTION_STD", "0") == "1":
    runner.alg.policy.std.requires_grad_(False)
  (log_dir / "params").mkdir()
  dump_yaml(log_dir / "params/env.yaml", asdict(cfg))
  dump_yaml(log_dir / "params/agent.yaml", asdict(agent))
  (log_dir / "source.json").write_text(json.dumps({
    "checkpoint": str(checkpoint),
    "posture": posture,
    "load": "none",
    "auxiliary_pull_force": 0,
    "action_noise_std_override": float(noise_std) if noise_std is not None else None,
    "freeze_action_std": os.environ.get("FREEZE_ACTION_STD", "0") == "1",
    "freeze_obs_normalizer": freeze_normalizer,
    "learning_rate": agent.algorithm.learning_rate,
    "entropy_coef": agent.algorithm.entropy_coef,
    "smp_reference": "https://github.com/tholin-1007/smp/tree/0e67286fe7df77a73740d237ef36b109136552b6",
    "full_smp_prior": False,
  }, indent=2))
  try:
    runner.learn(num_learning_iterations=iterations, init_at_random_ep_len=False)
  finally:
    env.close()
  print(f"{posture.upper()}_SMP_TRAINING_DONE", flush=True)


if __name__ == "__main__":
  main()
