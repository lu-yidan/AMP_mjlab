"""Resume V12 fine-tune from a checkpoint, curriculum pinned at final stage."""
import os
from dataclasses import asdict
from pathlib import Path

import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.utils.os import dump_yaml
from mjlab.utils.torch import configure_torch_backends

import mjlab.tasks  # noqa: F401
import src.tasks  # noqa: F401

task = "Unitree-G1-HoST-StandUp"
ckpt = os.environ["CKPT"]
log_dir = Path(os.environ["LOG_DIR"])
num_envs = int(os.environ.get("NUM_ENVS", "4096"))
max_iter = int(os.environ.get("MAX_ITER", "2000"))
save_interval = int(os.environ.get("SAVE_INTERVAL", "1000"))
seed = int(os.environ.get("SEED", "1"))

configure_torch_backends()
device = "cuda:0" if torch.cuda.is_available() else "cpu"

env_cfg = load_env_cfg(task, play=False)
env_cfg.scene.num_envs = num_envs
env_cfg.seed = seed
# Pin the curriculum at its final-stage floor so the loaded policy keeps
# training in the exact regime it was optimised for (action_scale=0.25, no pull).
env_cfg.actions["joint_pos"].scale = 0.25
env_cfg.events["init_pull_force"].params["force"] = 0.0

agent_cfg = load_rl_cfg(task)
agent_cfg.max_iterations = max_iter
agent_cfg.save_interval = save_interval
agent_cfg.seed = seed

log_dir.mkdir(parents=True, exist_ok=True)

env = ManagerBasedRlEnv(cfg=env_cfg, device=device, render_mode=None)
env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
runner_cls = load_runner_cls(task) or MjlabOnPolicyRunner
runner = runner_cls(env, asdict(agent_cfg), str(log_dir), device)
print(f"[INFO] Loading checkpoint: {ckpt}", flush=True)
runner.load(ckpt, load_optimizer=True)

params_dir = log_dir / "params"
params_dir.mkdir(parents=True, exist_ok=True)
dump_yaml(params_dir / "env.yaml", asdict(env_cfg))
dump_yaml(params_dir / "agent.yaml", asdict(agent_cfg))

runner.learn(num_learning_iterations=max_iter, init_at_random_ep_len=True)
print("[DONE]", flush=True)
