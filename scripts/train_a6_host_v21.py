"""Continue V20 with post-escape formal-height shaping."""

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
    "--post-escape-height-weight",
    "2.0",
    "--post-escape-stance-weight",
    "3.0",
    "--train-std",
    "0.035",
    "--learning-rate",
    "7.5e-6",
    "--episode-length-steps",
    "1150",
    "--method-name",
    "native-scale post-escape formal-height residual continuation",
    "--experiment-name",
    "g1_host_a6_v21_postescape_height",
    "--log-tag",
    "A6_V21",
  ]
  main()
