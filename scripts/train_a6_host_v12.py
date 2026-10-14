"""Train a bounded output-layer recovery response after motor activation."""

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
    "4.0",
    "--early-recovery-weight",
    "4.0",
    "--early-recovery-window",
    "120",
    "--actor-scope",
    "final-layer",
    "--min-scale-hold-steps",
    "14400",
    "--train-std",
    "0.04",
    "--learning-rate",
    "1e-5",
    "--resume-training-state",
    "--method-name",
    "output-layer early-recovery curriculum",
    "--experiment-name",
    "g1_host_a6_v12_early_recovery",
    "--log-tag",
    "A6_V12",
  ]
  main()
