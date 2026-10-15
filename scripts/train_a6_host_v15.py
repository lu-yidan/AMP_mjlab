"""Seed retained near-standing A6 homotopy from the stable V37 policy."""

from __future__ import annotations

import sys

from scripts.train_a6_host_v8 import main


if __name__ == "__main__":
  sys.argv[1:1] = [
    "--action-scale-decay",
    "0.01",
    "--standing-height",
    "0.70",
    "--standing-fraction",
    "0.90",
    "--required-successes",
    "2",
    "--require-retention-gate",
    "--retention-upright",
    "0.80",
    "--retention-step-fraction",
    "0.80",
    "--train-std",
    "0.05",
    "--learning-rate",
    "1e-5",
    "--method-name",
    "V37-seeded retained high-scale A6 homotopy",
    "--experiment-name",
    "g1_host_a6_v15_v37_retained_near_homotopy",
    "--log-tag",
    "A6_V15",
  ]
  main()
