"""Continue the near-standing homotopy with slower confirmed scale changes."""

from __future__ import annotations

import sys

from scripts.train_a6_host_v8 import main


if __name__ == "__main__":
  sys.argv[1:1] = [
    "--action-scale-decay",
    "0.005",
    "--standing-height",
    "0.70",
    "--standing-fraction",
    "0.90",
    "--required-successes",
    "2",
    "--train-std",
    "0.06",
    "--learning-rate",
    "1.5e-5",
    "--resume-training-state",
    "--method-name",
    "confirmed near-standing action-scale homotopy continuation",
    "--experiment-name",
    "g1_host_a6_v9_near_scale_homotopy",
    "--log-tag",
    "A6_V9",
  ]
  main()
