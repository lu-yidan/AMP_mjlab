"""V9 post-stand rewards for open arms and vertical torso posture."""

from mjlab.managers.reward_manager import RewardTermCfg

from src.tasks.host_recovery import mdp
from src.tasks.host_recovery.posture29_v8 import (
  POST_STAND_HEIGHT,
  _bodies,
  posture29_v8_env_cfg,
)


def posture29_v9_env_cfg():
  cfg = posture29_v8_env_cfg()
  cfg.rewards["target_hands_away_from_torso"] = RewardTermCfg(
    func=mdp.target_hands_away_from_torso,
    weight=0.20,
    params={
      "phase3_height": POST_STAND_HEIGHT,
      "lateral_margin": 0.22,
      "error_scale": 0.08,
      "torso_cfg": _bodies("torso_link"),
      "hand_cfg": _bodies("left_wrist_yaw_link", "right_wrist_yaw_link"),
    },
  )
  cfg.rewards["target_torso_vertical"] = RewardTermCfg(
    func=mdp.target_torso_vertical,
    weight=0.25,
    params={
      "phase3_height": POST_STAND_HEIGHT,
      "sigma": -8.0,
      "torso_cfg": _bodies("torso_link"),
    },
  )
  return cfg
