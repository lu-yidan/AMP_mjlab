"""Run formal-scale near-standing acquisition with checkpoint-safe normalization."""

import os

os.environ["A6_RUN_LABEL"] = "V7"

from scripts.train_a6_host_v6 import main


if __name__ == "__main__":
  main()
