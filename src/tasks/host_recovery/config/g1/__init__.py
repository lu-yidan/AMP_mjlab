"""Unitree G1 HoST standing-up task registration."""

from mjlab.tasks.registry import register_mjlab_task
from src.tasks.host_recovery.rl import HoSTOnPolicyRunner

from .env_cfgs import unitree_g1_host_standup_env_cfg
from .rl_cfg import unitree_g1_host_standup_ppo_runner_cfg

register_mjlab_task(
  task_id="Unitree-G1-HoST-StandUp",
  env_cfg=unitree_g1_host_standup_env_cfg(),
  play_env_cfg=unitree_g1_host_standup_env_cfg(play=True),
  rl_cfg=unitree_g1_host_standup_ppo_runner_cfg(),
  runner_cls=HoSTOnPolicyRunner,
)