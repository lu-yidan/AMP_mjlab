"""Repeat HoST get-up evaluation across independent seeds and summarize it."""

import argparse
import json
import statistics
from pathlib import Path
from types import SimpleNamespace

from eval_host import evaluate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--model-source", default="host_urdf_capsule")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seeds", default="50,51,52,53,54")
    parser.add_argument("--num-envs", type=int, default=50)
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--steps", type=int, default=250)
    parser.add_argument("--action-beta", type=float, default=0.25)
    parser.add_argument("--assist-force-n", type=float, default=0.0)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    seeds = [int(value.strip()) for value in args.seeds.split(",")]
    if len(seeds) < 2 or len(set(seeds)) != len(seeds):
        parser.error("--seeds must contain at least two distinct integers")
    if min(args.num_envs, args.episodes, args.steps) <= 0:
        parser.error("counts must be positive")
    if not 0.0 < args.action_beta <= 1.0 or args.assist_force_n < 0.0:
        parser.error("action beta must be in (0, 1] and assist force nonnegative")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for seed in seeds:
        result = evaluate(
            SimpleNamespace(
                checkpoint=args.checkpoint,
                model_source=args.model_source,
                device=args.device,
                num_envs=args.num_envs,
                episodes=args.episodes,
                steps=args.steps,
                seed=seed,
                action_beta=args.action_beta,
                assist_force_n=args.assist_force_n,
                stochastic=False,
                allow_model_transfer=False,
                allow_physics_transfer=False,
                output=str(output_dir / f"seed_{seed}.json"),
            )
        )
        results.append(result)
    success_rates = [result["success_rate"] for result in results]
    stable_rates = [
        result["successes_sustained_last_second_above_0_7m"] / result["total"]
        for result in results
    ]
    summary = {
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "seeds": seeds,
        "episodes_per_seed": args.num_envs * args.episodes,
        "total_episodes": sum(result["total"] for result in results),
        "action_beta": args.action_beta,
        "assistance_force_n": args.assist_force_n,
        "success_rate_mean": statistics.mean(success_rates),
        "success_rate_sample_std": statistics.stdev(success_rates),
        "successes_total": sum(result["successes"] for result in results),
        "success_and_last_second_standing_rate_mean": statistics.mean(stable_rates),
        "success_and_last_second_standing_rate_sample_std": statistics.stdev(
            stable_rates
        ),
        "success_and_last_second_standing_total": sum(
            result["successes_sustained_last_second_above_0_7m"] for result in results
        ),
        "per_seed": [
            {
                "seed": result["seed"],
                "successes": result["successes"],
                "success_rate": result["success_rate"],
                "success_and_last_second_standing": result[
                    "successes_sustained_last_second_above_0_7m"
                ],
            }
            for result in results
        ],
    }
    path = output_dir / "summary.json"
    path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
