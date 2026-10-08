"""Train V7 by adapting only the V4 actor output layer."""

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
from mjlab.utils.os import dump_yaml

from src.tasks.host_recovery.config.g1.rl_cfg import (
  unitree_g1_host_standup_ppo_runner_cfg,
)
from src.tasks.host_recovery.posture29_v7 import (
  POST_STAND_HEIGHT,
  posture29_v7_env_cfg,
)
from src.tasks.host_recovery.rl import HoSTOnPolicyRunner


def sha256(path: Path) -> str:
  return hashlib.sha256(path.read_bytes()).hexdigest()


def set_action_scale(env, value: float) -> None:
  env._host_action_rescale = torch.full(
    (env.num_envs, 1), value, device=env.device
  )
  action = env.action_manager.get_term("joint_pos")
  if isinstance(action.cfg.scale, dict):
    action.cfg.scale = {key: value for key in action.cfg.scale}
  else:
    action.cfg.scale = value
  if isinstance(action._scale, torch.Tensor):
    action._scale[:] = value
  else:
    action._scale = value


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--checkpoint", type=Path, required=True)
  parser.add_argument("--log-dir", type=Path, required=True)
  parser.add_argument("--num-envs", type=int, default=4096)
  parser.add_argument("--updates", type=int, default=750)
  parser.add_argument("--seed", type=int, default=20261008)
  parser.add_argument("--preflight", action="store_true")
  args = parser.parse_args()

  args.log_dir.mkdir(parents=True, exist_ok=False)
  cfg = posture29_v7_env_cfg()
  cfg.scene.num_envs = args.num_envs
  cfg.seed = args.seed
  agent = unitree_g1_host_standup_ppo_runner_cfg()
  agent.seed = args.seed
  agent.logger = "tensorboard"
  agent.experiment_name = "g1_host29_posture_v7"
  agent.max_iterations = args.updates
  agent.save_interval = 250
  agent.algorithm.learning_rate = 3.0e-6
  agent.algorithm.schedule = "fixed"
  agent.algorithm.num_learning_epochs = 1
  agent.algorithm.entropy_coef = 0.0
  agent.actor.distribution_cfg.update(init_std=0.01, max_std=0.01)
  dump_yaml(args.log_dir / "env.yaml", asdict(cfg))
  dump_yaml(args.log_dir / "agent.yaml", asdict(agent))

  env = ManagerBasedRlEnv(cfg=cfg, device="cuda:0")
  try:
    wrapper = RslRlVecEnvWrapper(env, clip_actions=agent.clip_actions)
    runner = HoSTOnPolicyRunner(wrapper, asdict(agent), str(args.log_dir), "cuda:0")
    fresh_critic = copy.deepcopy(runner.alg.policy.critic.state_dict())
    fresh_critic_normalizer = copy.deepcopy(
      runner.privileged_obs_normalizer.state_dict()
    )
    runner.load(
      str(args.checkpoint),
      load_optimizer=False,
      load_cfg={"actor": True, "critic": False},
      map_location="cuda:0",
    )
    runner.alg.policy.critic.load_state_dict(fresh_critic)
    runner.privileged_obs_normalizer.load_state_dict(fresh_critic_normalizer)
    runner.current_learning_iteration = 0
    runner.obs_normalizer.until = int(runner.obs_normalizer.count.item())

    source = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    source_actor = source["actor_state_dict"]
    policy_state = runner.alg.policy.state_dict()
    for name, value in source_actor.items():
      if name == "distribution.std_param":
        continue
      mapped = "actor." + name.removeprefix("mlp.")
      assert torch.equal(policy_state[mapped].detach().cpu(), value.cpu()), mapped
    assert len(runner.alg.optimizer.state) == 0

    for parameter in runner.alg.policy.actor.parameters():
      parameter.requires_grad_(False)
    for parameter in runner.alg.policy.actor[6].parameters():
      parameter.requires_grad_(True)

    source_scale = source["infos"]["env_state"]["host_action_rescale"].float()
    action_scale = float(source_scale.mean())
    set_action_scale(env, action_scale)
    with torch.no_grad():
      runner.alg.policy.std.fill_(0.01)
    runner.alg.policy.std.requires_grad_(False)

    launch = {
      "checkpoint": str(args.checkpoint.resolve()),
      "checkpoint_sha256": sha256(args.checkpoint),
      "num_envs": args.num_envs,
      "updates": args.updates,
      "seed": args.seed,
      "actor_exact_at_launch": True,
      "actor_trainable_scope": "final Linear(128, 29) only",
      "actor_normalizer_frozen": True,
      "fresh_critic": True,
      "fresh_optimizer": True,
      "learning_rate": 3.0e-6,
      "learning_epochs": 1,
      "entropy_coef": 0.0,
      "noise_std": 0.01,
      "action_std_frozen": True,
      "action_scale_frozen": action_scale,
      "post_stand_height": POST_STAND_HEIGHT,
      "reason": "V6 kept 100% held but produced 3.31-3.41 rad/s joint RMS and visible stepping; preserve V4 hidden recovery features and adapt only output pose.",
      "reward_adjustments": {
        "upper_pose_multiplier": 1.50,
        "elbow_deviation": -0.010,
        "arm_twist_deviation": -0.006,
        "wrist_deviation": -0.004,
        "leg_velocity_penalty": -0.001,
        "upper_velocity_penalty": -0.0005,
      },
    }
    (args.log_dir / "launch.json").write_text(
      json.dumps(launch, indent=2), encoding="utf-8"
    )

    observations = wrapper.get_observations()
    policy = runner.get_inference_policy(device="cuda:0")
    with torch.inference_mode():
      for _ in range(100):
        actions = policy(observations["actor"])
        observations, rewards, _, _ = wrapper.step(actions)
        assert actions.shape == (args.num_envs, 29)
        assert torch.isfinite(actions).all()
        assert torch.isfinite(rewards).all()
        assert all(torch.isfinite(value).all() for value in observations.values())
    print("HOST29_V7_PREFLIGHT_PASS", flush=True)
    if args.preflight:
      return

    runner.save(str(args.log_dir / "model_0.pt"))
    runner.learn(num_learning_iterations=args.updates, init_at_random_ep_len=True)
    runner.save(str(args.log_dir / f"model_{args.updates}_final.pt"))
    print("HOST29_V7_TRAINING_DONE", flush=True)
  finally:
    env.close()


if __name__ == "__main__":
  main()
