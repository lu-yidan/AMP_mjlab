"""Learn a bounded lower-body residual for early post-activation recovery."""

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
    "--trainable-output-rows",
    "0",
    "1",
    "2",
    "3",
    "4",
    "5",
    "6",
    "7",
    "8",
    "9",
    "10",
    "11",
    "12",
    "--max-weight-delta",
    "0.003",
    "--max-bias-delta",
    "0.001",
    "--frozen-std",
    "0.005",
    "--min-scale-hold-steps",
    "14400",
    "--train-std",
    "0.04",
    "--learning-rate",
    "1e-5",
    "--resume-training-state",
    "--reset-optimizer-state",
    "--method-name",
    "bounded lower-body early-recovery residual",
    "--experiment-name",
    "g1_host_a6_v13_bounded_recovery_residual",
    "--log-tag",
    "A6_V13",
  ]
  main()
