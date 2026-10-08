"""Formal 29-DoF A6 evaluation using the frozen A6 protocol."""

from src.tasks.host_recovery import a6_runtime

import eval_a6_23_formal as protocol


protocol.SCENE_NAMES = a6_runtime.SCENE_NAMES
protocol._a6_completion_clearance = a6_runtime._a6_completion_clearance
protocol._a6_ground_force = a6_runtime._a6_ground_force
protocol._a6_height = a6_runtime._a6_height
protocol._a6_upright = a6_runtime._a6_upright
protocol.a6_env_cfg = a6_runtime.a6_env_cfg
protocol.a6_reset_counts = a6_runtime.a6_reset_counts
protocol.REPORT_TITLE = "29-DoF A6 formal evaluation"
protocol.COMPLETE_MARKER = "A6_29_FORMAL_EVAL_COMPLETE"


if __name__ == "__main__":
  protocol.main()
