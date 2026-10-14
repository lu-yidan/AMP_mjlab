"""Summarize fixed-seed first-episode A6 diagnostic reports."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean

SCENARIOS = ("guided_prone", "guided_supine", "free_supine", "flat_supine")


def summarize_condition(directory: Path, expected_seeds: tuple[int, ...]) -> dict:
    reports = []
    for seed in expected_seeds:
        path = directory / f"seed{seed}.json"
        report = json.loads(path.read_text(encoding="utf-8"))
        if report["guided_offset_m"] != (0.25 if directory.name == "edge025" else 0):
            raise ValueError(f"guided offset mismatch in {path}")
        reports.append(report)

    scenarios = {}
    for name in SCENARIOS:
        worlds = [report["worlds"][name] for report in reports]
        scenarios[name] = {
            "escaped_count": sum(
                world["first_escaped_step"] is not None for world in worlds
            ),
            "invalid_count": sum(
                world["first_invalid_step"] is not None for world in worlds
            ),
            "lift_invalid_count": sum(
                world["first_invalid_step"] is not None
                and world["first_plate_lift_over_5cm_step"] is not None
                for world in worlds
            ),
            "mean_max_near_edge_m": round(
                mean(world["near_edge_displacement_m"]["max"] for world in worlds),
                4,
            ),
            "mean_max_abs_lateral_m": round(
                mean(world["lateral_displacement_m"]["max"] for world in worlds),
                4,
            ),
            "mean_overcurl": round(
                mean(world["overcurl"]["mean"] for world in worlds), 4
            ),
            "per_seed": [
                {
                    "seed": seed,
                    "escaped_step": world["first_escaped_step"],
                    "invalid_step": world["first_invalid_step"],
                    "lift_over_5cm_step": world["first_plate_lift_over_5cm_step"],
                    "max_near_edge_m": world["near_edge_displacement_m"]["max"],
                }
                for seed, world in zip(expected_seeds, worlds, strict=True)
            ],
        }
        if name == "flat_supine":
            scenarios[name]["stable_final_count"] = sum(
                world["series"][-1]["pelvis_height_m"] >= 0.70
                and world["series"][-1]["upright"] >= 0.95
                for world in worlds
            )
            scenarios[name]["mean_final_pelvis_m"] = round(
                mean(world["series"][-1]["pelvis_height_m"] for world in worlds),
                4,
            )
    return {
        "checkpoint_iteration": reports[0]["iteration"],
        "guided_offset_m": reports[0]["guided_offset_m"],
        "seed_count": len(reports),
        "scenarios": scenarios,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(42, 50)))
    args = parser.parse_args()
    seeds = tuple(args.seeds)
    if len(set(seeds)) != len(seeds):
        parser.error("seed list contains duplicates")
    groups = {}
    for checkpoint in ("01099", "01199"):
        for setting in ("edge025", "center"):
            label = f"{checkpoint}_{setting}"
            groups[label] = summarize_condition(
                args.root / f"eval_{checkpoint}" / setting, seeds
            )
    result = {
        "protocol": {
            "seeds": seeds,
            "steps": 300,
            "assist_force_n": 0,
            "flat_stable_min_pelvis_m": 0.70,
            "flat_stable_min_upright": 0.95,
        },
        "groups": groups,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    for label, group in groups.items():
        print(
            label,
            {
                name: {
                    key: value
                    for key, value in group["scenarios"][name].items()
                    if key != "per_seed"
                }
                for name in SCENARIOS
            },
        )
    print(f"summary={args.output}")


if __name__ == "__main__":
    main()
