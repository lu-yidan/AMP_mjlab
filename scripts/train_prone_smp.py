"""Fine-tune prone no-load HoST using the same SMP get-up reward shapes."""

import os

from train_supine_smp import main


if __name__ == "__main__":
  os.environ["SMP_TASK"] = "Unitree-G1-HoST-ProneSmp"
  main()
