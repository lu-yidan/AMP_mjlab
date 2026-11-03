"""Package the validated proprio detector and frozen recovery action."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import torch

from g1recovery_host.ppo import HostPolicy


def sha256(path: Path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--detector", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=0.95)
    parser.add_argument("--consecutive", type=int, default=2)
    parser.add_argument("--pulse-steps", type=int, default=30)
    parser.add_argument("--prone-gravity-x", type=float, default=0.70)
    parser.add_argument("--min-steps", type=int, default=60)
    args = parser.parse_args()
    source = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    detector = torch.load(args.detector, map_location="cpu", weights_only=False)
    contract = "proprio_torque_actor_privileged_critic_v2"
    if (source.get("observation_contract") != contract
        or detector.get("observation_contract") != contract
        or source.get("action_contract") != "scene_independent_v1"):
        parser.error("source actor and detector must share the 738D proprio contract")
    actor = HostPolicy(
        738, 29, critic_obs_dim=786,
        log_std_bounds=(-2.3, math.log(source.get("action_std_max", 0.7))),
        upper_residual=True, waist_residual_scale=5.0,
    )
    actor.load_state_dict(source["policy"], strict=True)
    actor.eval()
    pulse_action = actor.deterministic_action(torch.zeros(1, 738))[0]
    artifact = {
        "version": "proprio_recovery_controller_v1",
        "observation_contract": contract,
        "action_contract": "scene_independent_v1",
        "detector_mean": detector["mean"],
        "detector_scale": detector["scale"],
        "detector_model": detector["model"],
        "pulse_action": pulse_action,
        "threshold": args.threshold,
        "consecutive": args.consecutive,
        "pulse_steps": args.pulse_steps,
        "prone_gravity_x": args.prone_gravity_x,
        "min_steps": args.min_steps,
        "source_checkpoint_sha256": sha256(args.checkpoint),
        "detector_checkpoint_sha256": sha256(args.detector),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(artifact, args.output)
    summary = {k: v for k, v in artifact.items() if not isinstance(v, (torch.Tensor, dict))}
    summary["pulse_action"] = pulse_action.tolist()
    args.output.with_suffix(".json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
