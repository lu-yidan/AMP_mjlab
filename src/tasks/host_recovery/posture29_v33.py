"""Slow upper correction with sagittal leg outputs kept frozen."""

from src.tasks.host_recovery.posture29_v28 import posture29_v28_env_cfg


def posture29_v33_env_cfg():
  cfg = posture29_v28_env_cfg()
  action = cfg.actions["joint_pos"]
  action.arm_blend = 0.5
  action.arm_gain = 0.3
  action.balance_blend = 0.5
  action.balance_gain = 1.0
  action.delay_steps = 50
  action.ramp_steps = 100
  return cfg
