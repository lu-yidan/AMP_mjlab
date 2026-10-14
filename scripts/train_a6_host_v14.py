"""Train from mixed near and replayed motor-on recovery states."""

from __future__ import annotations

import sys

from scripts.train_a6_host_v8 import main


if __name__ == "__main__":
  sys.argv[1:1] = [
    "--recovery-bank",
    "datasets/reset_banks/a6_29_motor_recovery_v3/train.npz",
    "--recovery-fraction",
    "0.5",
    "--recovery-successful-only",
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
    "--min-scale-hold-steps",
    "14400",
    "--train-std",
    "0.04",
    "--learning-rate",
    "1e-5",
    "--resume-training-state",
    "--reset-optimizer-state",
    "--method-name",
    "mixed near and replayed motor-on recovery states",
    "--experiment-name",
    "g1_host_a6_v14_motor_recovery_replay",
    "--log-tag",
    "A6_V14",
  ]
  main()
