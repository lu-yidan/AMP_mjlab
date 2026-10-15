"""Align cached-retention homotopy with the 1,150-step near audit horizon."""

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
    "--episode-length-steps",
    "1150",
    "--preflight-steps",
    "2400",
    "--save-interval",
    "200",
    "--train-std",
    "0.04",
    "--learning-rate",
    "7.5e-6",
    "--resume-training-state",
    "--method-name",
    "1150-step horizon-aligned cached-final retained near-scale homotopy",
    "--experiment-name",
    "g1_host_a6_v18_horizon_aligned_cached_retention",
    "--log-tag",
    "A6_V18",
  ]
  main()
