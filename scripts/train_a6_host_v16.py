"""Continue A6 near-scale homotopy with a pre-reset retention cache."""

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
    "3",
    "--require-retention-gate",
    "--retention-upright",
    "0.80",
    "--retention-step-fraction",
    "0.80",
    "--min-scale-hold-steps",
    "14400",
    "--train-std",
    "0.04",
    "--learning-rate",
    "7.5e-6",
    "--resume-training-state",
    "--method-name",
    "cached-final retained near-scale homotopy",
    "--experiment-name",
    "g1_host_a6_v16_cached_retention_homotopy",
    "--log-tag",
    "A6_V16",
  ]
  main()
