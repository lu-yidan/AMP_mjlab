"""Train a frozen recovery residual plus a symmetric post-escape correction."""

from __future__ import annotations

import sys

from scripts.train_a6_host_v19 import main


if __name__ == "__main__":
  sys.argv[1:1] = [
    "--action-scale",
    "0.9323874",
    "--residual-scale",
    "0.25",
    "--correction-scale",
    "0.15",
    "--symmetric-postescape-correction",
    "--post-escape-height-weight",
    "2.0",
    "--post-escape-stance-weight",
    "3.0",
    "--train-std",
    "0.03",
    "--learning-rate",
    "5e-6",
    "--episode-length-steps",
    "1150",
    "--method-name",
    "frozen recovery with symmetric post-escape sagittal correction",
    "--experiment-name",
    "g1_host_a6_v22_symmetric_postescape",
    "--log-tag",
    "A6_V22",
  ]
  main()
