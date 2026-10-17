"""Train a native-scale obstacle residual through post-escape standing."""

from __future__ import annotations

import sys

from scripts.train_a6_host_v19 import main


if __name__ == "__main__":
  sys.argv[1:1] = [
    "--action-scale",
    "0.9323874",
    "--residual-scale",
    "0.25",
    "--residual-after-escape",
    "--train-std",
    "0.04",
    "--learning-rate",
    "1e-5",
    "--episode-length-steps",
    "1150",
    "--method-name",
    "native-scale obstacle residual through post-escape standing",
    "--experiment-name",
    "g1_host_a6_v20_native_obstacle_residual",
    "--log-tag",
    "A6_V20",
  ]
  main()
