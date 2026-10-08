"""Record 29-DoF A6 rollout videos using the frozen A6 protocol."""

from src.tasks.host_recovery import a6_runtime

import eval_a6_23_video as protocol


protocol.SCENE_NAMES = a6_runtime.SCENE_NAMES
protocol.a6_env_cfg = a6_runtime.a6_env_cfg
protocol.VIDEO_PREFIX = "a6_29"
protocol.COMPLETE_MARKER = "A6_29_VIDEO_COMPLETE"


if __name__ == "__main__":
  protocol.main()
