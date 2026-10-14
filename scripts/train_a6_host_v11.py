"""Consolidate the best V10 near-standing policy before further scale decay."""

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
    "--retention-reward-weight",
    "6.0",
    "--min-scale-hold-steps",
    "14400",
    "--train-std",
    "0.035",
    "--learning-rate",
    "5e-6",
    "--resume-training-state",
    "--method-name",
    "retention-weighted near-standing consolidation",
    "--experiment-name",
    "g1_host_a6_v11_retention_consolidation",
    "--log-tag",
    "A6_V11",
  ]
  main()
