"""Late-settle damping objective for the stable V37 policy."""

import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg

from src.tasks.host_recovery.posture29_v37 import posture29_v37_env_cfg


SETTLE_HEIGHT = 0.66
SETTLE_UPRIGHT = 0.80
SETTLE_STEPS = 100
ALL_ACTIONS = tuple(range(29))


def _all_joints() -> SceneEntityCfg:
  return SceneEntityCfg("robot", joint_names=(".*",), preserve_order=True)


def _ensure_settle_state(env: ManagerBasedRlEnv) -> None:
  if getattr(env, "_host_v38_settle_steps", None) is not None:
    if env._host_v38_settle_steps.shape[0] == env.num_envs:
      return
  env._host_v38_settle_steps = torch.zeros(
    env.num_envs, device=env.device, dtype=torch.long
  )
  env._host_v38_settle_gate = torch.zeros(
    env.num_envs, device=env.device, dtype=torch.bool
  )
  env._host_v38_settle_update = -1


def _settle_gate(env: ManagerBasedRlEnv) -> torch.Tensor:
  _ensure_settle_state(env)
  current_step = int(env.common_step_counter)
  if env._host_v38_settle_update != current_step:
    robot = env.scene["robot"]
    standing = robot.data.root_link_pos_w[:, 2] > SETTLE_HEIGHT
    standing &= robot.data.projected_gravity_b[:, 2] < -SETTLE_UPRIGHT
    env._host_v38_settle_steps = torch.where(
      standing,
      env._host_v38_settle_steps + 1,
      torch.zeros_like(env._host_v38_settle_steps),
    )
    env._host_v38_settle_gate = env._host_v38_settle_steps >= SETTLE_STEPS
    env._host_v38_settle_update = current_step
  return env._host_v38_settle_gate.float()


def settled_joint_stillness(
  env: ManagerBasedRlEnv,
  velocity_scale: float = 0.50,
  asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", joint_names=()),
) -> torch.Tensor:
  robot = env.scene[asset_cfg.name]
  mean_sq = torch.mean(
    torch.square(robot.data.joint_vel[:, asset_cfg.joint_ids]), dim=1
  )
  return torch.exp(-mean_sq / (velocity_scale * velocity_scale)) * _settle_gate(env)


def settled_joint_velocity(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", joint_names=()),
) -> torch.Tensor:
  robot = env.scene[asset_cfg.name]
  mean_sq = torch.mean(
    torch.square(robot.data.joint_vel[:, asset_cfg.joint_ids]), dim=1
  )
  return mean_sq * _settle_gate(env)


def settled_action_rate(env: ManagerBasedRlEnv) -> torch.Tensor:
  action = env.action_manager.action[:, ALL_ACTIONS]
  previous = env.action_manager.prev_action[:, ALL_ACTIONS]
  return torch.mean(torch.square(action - previous), dim=1) * _settle_gate(env)


def settled_root_linear_speed(env: ManagerBasedRlEnv) -> torch.Tensor:
  robot = env.scene["robot"]
  speed_sq = torch.sum(torch.square(robot.data.root_link_lin_vel_b[:, :2]), dim=1)
  return speed_sq * _settle_gate(env)


def settled_root_angular_speed(env: ManagerBasedRlEnv) -> torch.Tensor:
  robot = env.scene["robot"]
  speed_sq = torch.mean(torch.square(robot.data.root_link_ang_vel_b), dim=1)
  return speed_sq * _settle_gate(env)


def posture29_v38_env_cfg():
  cfg = posture29_v37_env_cfg()

  cfg.rewards["target_all_joint_stillness"].weight = 0.30
  cfg.rewards["target_all_joint_velocity"].weight = -0.0020
  cfg.rewards["target_all_action"].weight = -0.03
  cfg.rewards["target_standing_action_rate"].weight = -0.10
  cfg.rewards["target_root_angular_speed"].weight = -0.30

  cfg.rewards["target_settled_joint_stillness"] = RewardTermCfg(
    func=settled_joint_stillness,
    weight=1.00,
    params={"velocity_scale": 0.50, "asset_cfg": _all_joints()},
  )
  cfg.rewards["target_settled_joint_velocity"] = RewardTermCfg(
    func=settled_joint_velocity,
    weight=-0.0080,
    params={"asset_cfg": _all_joints()},
  )
  cfg.rewards["target_settled_action_rate"] = RewardTermCfg(
    func=settled_action_rate,
    weight=-0.50,
  )
  cfg.rewards["target_settled_root_linear_speed"] = RewardTermCfg(
    func=settled_root_linear_speed,
    weight=-1.00,
  )
  cfg.rewards["target_settled_root_angular_speed"] = RewardTermCfg(
    func=settled_root_angular_speed,
    weight=-1.00,
  )

  return cfg
