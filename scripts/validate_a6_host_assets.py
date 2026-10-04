"""Validate the external assets required by the historical A6 HoST study.

The validator intentionally fails when a required bank is missing or has a
different SHA256.  Using the current flat HoST reset as a silent substitute
would invalidate the paired G-/G+ comparison.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path


EXPECTED = {
  "outputs/multiterrain_bank/train.npz":
    "287eae8e8840c1b3281e7010a84182b824fefacd34439c4f993e9f130af1027a",
  "datasets/reset_banks/natural_curriculum_v1/train.npz":
    "e9f94540520d7927dee01150a0f0bae39ad2aad5ae008b1c7c71f521650e44b9",
  "datasets/reset_banks/procedural_low_v1/train.npz":
    "e289bed8d1c93e87fbdcf969fc5fc741f2d8500a0908ed145d2a4e02e7fe116d",
}


def sha256(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as stream:
    for block in iter(lambda: stream.read(1024 * 1024), b""):
      digest.update(block)
  return digest.hexdigest()


def main() -> int:
  parser = argparse.ArgumentParser()
  parser.add_argument("--root", type=Path, default=Path.cwd())
  args = parser.parse_args()

  failed = False
  for relative, expected in EXPECTED.items():
    path = args.root / relative
    if not path.is_file():
      print(f"MISSING {relative}")
      failed = True
      continue
    actual = sha256(path)
    if actual != expected:
      print(f"SHA256_MISMATCH {relative} expected={expected} actual={actual}")
      failed = True
    else:
      print(f"OK {relative} sha256={actual}")

  if failed:
    print("A6 host migration is blocked: do not substitute another reset bank.")
    return 1
  print("All historical A6 reset banks are present and verified.")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
