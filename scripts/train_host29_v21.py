"""Continue V17 by adapting a coupled lateral/yaw action subset."""

import argparse
import copy
import json
import os
from dataclasses import asdict
from pathlib import Path

os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")

import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.utils.os import dump_yaml

from scripts.train_host29_v7 import set_action_scale, sha256
from src.tasks.host_recovery.config.g1.rl_cfg import (
  unitree_g1_host_standup_ppo_runner_cfg,
)
from src.tasks.host_recovery.posture29_v16 import POST_MOTION_HEIGHT
from src.tasks.host_recovery.posture29_v21 import posture29_v21_env_cfg
from src.tasks.host_recovery.posture29_v8 import POST_STAND_HEIGHT
from src.tasks.host_recovery.rl import HoSTOnPolicyRunner


COUPLED_JOINTS = (
  "left_hip_roll_joint",
  "left_hip_yaw_joint",
  "left_ankle_roll_joint",
  "right_hip_roll_joint",
  "right_hip_yaw_joint",
  "right_ankle_roll_joint",
  "waist_yaw_joint",
)
EXPECTED_ROWS = [1, 2, 5, 7, 8, 11, 12]


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--checkpoint", type=Path, required=True)
  parser.add_argument("--log-dir", type=Path, required=True)
  parser.add_argument("--num-envs", type=int, default=4096)
  parser.add_argument("--updates", type=int, default=500)
  parser.add_argument("--seed", type=int, default=20261010)
  parser.add_argument("--preflight", action="store_true")
  args = parser.parse_args()

  args.log_dir.mkdir(parents=True, exist_ok=False)
  cfg = posture29_v21_env_cfg()
  cfg.scene.num_envs = args.num_envs
  cfg.seed = args.seed
  agent = unitree_g1_host_standup_ppo_runner_cfg()
  agent.seed = args.seed
  agent.logger = "tensorboard"
  agent.experiment_name = "g1_host29_posture_v21"
  agent.max_iterations = args.updates
  agent.save_interval = 125
  agent.algorithm.learning_rate = 5.0e-6
  agent.algorithm.schedule = "fixed"
  agent.algorithm.num_learning_epochs = 1
  agent.algorithm.entropy_coef = 0.0
  agent.actor.distribution_cfg.update(init_std=0.005, max_std=0.005)
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
    policy_state = runner.alg.policy.state_dict()
    for name, value in source["actor_state_dict"].items():
      if name == "distribution.std_param":
        continue
      mapped = "actor." + name.removeprefix("mlp.")
      assert torch.equal(policy_state[mapped].detach().cpu(), value.cpu()), mapped
    assert len(runner.alg.optimizer.state) == 0

    for parameter in runner.alg.policy.actor.parameters():
      parameter.requires_grad_(False)
    output_layer = runner.alg.policy.actor[6]
    output_layer.weight.requires_grad_(True)
    output_layer.bias.requires_grad_(True)
    joint_names = list(env.scene["robot"].joint_names)
    trainable_rows = [joint_names.index(name) for name in COUPLED_JOINTS]
    assert trainable_rows == EXPECTED_ROWS, (trainable_rows, joint_names)
    weight_mask = torch.zeros_like(output_layer.weight)
    bias_mask = torch.zeros_like(output_layer.bias)
    weight_mask[trainable_rows] = 1.0
    bias_mask[trainable_rows] = 1.0
    output_layer.weight.register_hook(lambda gradient: gradient * weight_mask)
    output_layer.bias.register_hook(lambda gradient: gradient * bias_mask)
    frozen_weight = output_layer.weight.detach().clone()
    frozen_bias = output_layer.bias.detach().clone()

    action_scale = float(
      source["infos"]["env_state"]["host_action_rescale"].float().mean()
    )
    set_action_scale(env, action_scale)
    with torch.no_grad():
      runner.alg.policy.std.fill_(0.005)
    runner.alg.policy.std.requires_grad_(False)

    launch = {
      "checkpoint": str(args.checkpoint.resolve()),
      "checkpoint_sha256": sha256(args.checkpoint),
      "num_envs": args.num_envs,
      "updates": args.updates,
      "seed": args.seed,
      "actor_exact_at_launch": True,
      "actor_trainable_scope": "final actor rows 1, 2, 5, 7, 8, 11, and 12",
      "trainable_action_outputs": list(COUPLED_JOINTS),
      "actor_normalizer_frozen": True,
      "fresh_critic": True,
      "fresh_optimizer": True,
      "learning_rate": 5.0e-6,
      "learning_epochs": 1,
      "entropy_coef": 0.0,
      "noise_std": 0.005,
      "action_std_frozen": True,
      "action_scale_frozen": action_scale,
      "action_scale_curriculum": False,
      "pull_force_curriculum": False,
      "posture_gate_height": POST_STAND_HEIGHT,
      "motion_gate_height": POST_MOTION_HEIGHT,
      "reason": (
        "V20's isolated hip-yaw rows degraded yaw; V21 starts from the best "
        "V17 checkpoint and permits the coupled hip-roll, hip-yaw, ankle-roll, "
        "and waist-yaw outputs to coordinate."
      ),
      "reward_adjustments": {
        "yaw_rate": -1.75,
        "root_horizontal_speed": -0.70,
        "foot_horizontal_speed": -0.45,
        "leg_velocity_penalty": -0.0040,
        "hip_yaw_velocity_penalty": -0.15,
        "lateral_balance_velocity_penalty": -0.015,
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
    print("HOST29_V21_PREFLIGHT_PASS", flush=True)
    if args.preflight:
      return

    runner.save(str(args.log_dir / "model_0.pt"))
    runner.learn(num_learning_iterations=args.updates, init_at_random_ep_len=True)
    frozen_rows = [index for index in range(29) if index not in trainable_rows]
    assert torch.equal(output_layer.weight[frozen_rows], frozen_weight[frozen_rows])
    assert torch.equal(output_layer.bias[frozen_rows], frozen_bias[frozen_rows])
    runner.save(str(args.log_dir / f"model_{args.updates}_final.pt"))
    print("HOST29_V21_TRAINING_DONE", flush=True)
  finally:
    env.close()


if __name__ == "__main__":
  main()
