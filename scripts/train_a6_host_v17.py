"""Accelerate cached-retention near-scale homotopy after V16 validation."""

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
    "2400",
    "--train-std",
    "0.04",
    "--learning-rate",
    "7.5e-6",
    "--resume-training-state",
    "--method-name",
    "accelerated cached-final retained near-scale homotopy",
    "--experiment-name",
    "g1_host_a6_v17_accelerated_cached_retention",
    "--log-tag",
    "A6_V17",
  ]
  main()
