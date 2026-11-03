"""Summarize paired A6 proprio-only evaluations without choosing by train reward."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def read_pair(directory: Path, iteration: str, seed: int) -> dict:
    pair = {}
    for offset in ("center", "offset"):
        path = directory / f"model_{iteration}_{offset}.json"
        if seed != 48:
            path = directory / f"model_{iteration}_seed{seed}_{offset}.json"
        data = json.loads(path.read_text(encoding="utf-8"))
        if data["iteration"] != int(iteration) or data["seed"] != seed:
            raise ValueError(f"checkpoint/seed mismatch in {path}")
        expected_offset = 0.0 if offset == "center" else 0.25
        if data["guided_offset_m"] != expected_offset or data["per_scene"] != 128:
            raise ValueError(f"evaluation layout mismatch in {path}")
        pair[offset] = data["scenes"]
    return pair


def summarize(pair: dict) -> dict[str, int]:
    result = {
        "total_stood": 0, "guided_stood": 0, "guided_invalid": 0,
        "guided_prone_stood": 0, "guided_prone_invalid": 0,
        "guided_supine_stood": 0, "free_stood": 0, "flat_stood": 0,
    }
    for scenes in pair.values():
        for name, scene in scenes.items():
            group = name.split("_")[0]
            stood = scene[
                "stood_at_end" if group == "flat" else "escaped_then_stood_at_end"
            ]
            result["total_stood"] += stood
            result[f"{group}_stood"] += stood
            if group == "guided":
                result["guided_invalid"] += scene["invalid"]
                if name == "guided_prone":
                    result["guided_prone_stood"] += stood
                    result["guided_prone_invalid"] += scene["invalid"]
                elif name == "guided_supine":
                    result["guided_supine_stood"] += stood
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--iterations", nargs="+", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[48])
    args = parser.parse_args()
    print("iteration seed stood/3072 guided/1024 invalid/1024 prone/256 "
          "prone_invalid/256 supine/256 free/1024 flat/1024")
    for iteration in args.iterations:
        if not (len(iteration) == 5 and iteration.isdecimal()):
            parser.error(f"expected five-digit checkpoint iteration: {iteration}")
        for seed in args.seeds:
            pair = read_pair(args.directory, iteration, seed)
            result = summarize(pair)
            print(
                iteration, seed, result["total_stood"], result["guided_stood"],
                result["guided_invalid"], result["guided_prone_stood"],
                result["guided_prone_invalid"], result["guided_supine_stood"],
                result["free_stood"], result["flat_stood"],
            )


if __name__ == "__main__":
    main()
