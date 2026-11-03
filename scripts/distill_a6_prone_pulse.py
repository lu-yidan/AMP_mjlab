"""Distill a successful postescape pulse into the same live-proprio HoST actor.

The pulse and escape-success labels are simulation-only training targets.
The saved policy still consumes the ordinary 738D live actor observation.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from g1recovery_host.ppo import HostPolicy


def digest(path: Path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def model_from_state(state, device):
    model = HostPolicy(
        738, 29, critic_obs_dim=786,
        log_std_bounds=(-2.3, math.log(state.get("action_std_max", 0.7))),
        upper_residual=True, waist_residual_scale=5.0,
    ).to(device)
    model.load_state_dict(state["policy"], strict=True)
    return model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--pulse-data", type=Path, nargs="+", required=True)
    parser.add_argument("--preserve-data", type=Path, nargs="+", required=True)
    parser.add_argument("--valid-pulse-data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--updates", type=int, default=800)
    parser.add_argument("--batch-pulse", type=int, default=1024)
    parser.add_argument("--batch-preserve", type=int, default=2048)
    parser.add_argument("--pulse-weight", type=float, default=5.0)
    parser.add_argument("--preserve-weight", type=float, default=1.0)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if (state.get("observation_contract") != "proprio_torque_actor_privileged_critic_v2"
        or state.get("action_contract") != "scene_independent_v1"):
        parser.error("distillation requires the 738D proprio actor and scene-independent action")
    source = model_from_state(state, args.device).eval()
    for parameter in source.parameters():
        parameter.requires_grad_(False)
    student = copy.deepcopy(source).train()
    for parameter in student.actor.parameters():
        parameter.requires_grad_(True)
    for parameter in student.upper_residual.parameters():
        parameter.requires_grad_(True)
    pulse_arrays = []
    pulse_target = None
    for path in args.pulse_data:
        with np.load(path) as archive:
            pulse = archive["pulse_actor_obs"][archive["pulse_success"]].astype(np.float32)
            target = archive["pulse_target_action"].astype(np.float32)
            if target.shape != (29,) or pulse.shape[1:] != (738,):
                raise ValueError(f"invalid pulse observations or action in {path}")
            if pulse_target is not None and not np.allclose(target, pulse_target, atol=1e-5):
                raise ValueError("pulse target action differs across source datasets")
            pulse_target = target
            pulse_arrays.append(pulse)
    positive = torch.as_tensor(np.concatenate(pulse_arrays), device=args.device)
    if len(positive) < 500:
        raise ValueError("too few successful pulse frames for distillation")
    negative_arrays = []
    for path in args.preserve_data:
        with np.load(path) as archive:
            x = archive["actor_obs"][~archive["escaped_prone"]].astype(np.float32)
            if x.shape[1:] != (738,):
                raise ValueError(f"invalid preservation observations in {path}")
            negative_arrays.append(x)
    negative = torch.as_tensor(np.concatenate(negative_arrays), device=args.device)
    target = torch.as_tensor(pulse_target, device=args.device)
    optimizer = torch.optim.AdamW(
        list(student.actor.parameters()) + list(student.upper_residual.parameters()),
        lr=args.learning_rate, weight_decay=1e-5,
    )
    with np.load(args.valid_pulse_data) as archive:
        valid = torch.as_tensor(
            archive["pulse_actor_obs"][archive["pulse_success"]].astype(np.float32),
            device=args.device,
        )
    if len(valid) < 100:
        raise ValueError("held-out pulse dataset has too few successful frames")
    best = float("inf")
    best_policy = None
    progress = []
    for update in range(1, args.updates + 1):
        selected_positive = positive[torch.randint(len(positive), (args.batch_pulse,), device=args.device)]
        selected_negative = negative[torch.randint(len(negative), (args.batch_preserve,), device=args.device)]
        with torch.no_grad():
            preserve_target = source.mean_action(selected_negative)
        pulse_prediction = student.mean_action(selected_positive)
        pulse_loss = F.mse_loss(pulse_prediction, target.expand_as(pulse_prediction))
        preserve_loss = F.mse_loss(student.mean_action(selected_negative), preserve_target)
        loss = args.pulse_weight * pulse_loss + args.preserve_weight * preserve_loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
        optimizer.step()
        if update % 50 == 0 or update == 1 or update == args.updates:
            student.eval()
            with torch.no_grad():
                valid_pulse_mse = F.mse_loss(
                    student.mean_action(valid), target.expand(len(valid), -1)
                ).item()
                valid_preserve_mse = F.mse_loss(
                    student.mean_action(negative[:4096]),
                    source.mean_action(negative[:4096]),
                ).item()
            student.train()
            score = valid_pulse_mse + 2.0 * valid_preserve_mse
            row = {"update": update, "pulse_mse": valid_pulse_mse,
                   "preserve_mse": valid_preserve_mse, "score": score}
            progress.append(row)
            print(json.dumps(row), flush=True)
            if score < best:
                best = score
                best_policy = {k: v.detach().cpu().clone() for k, v in student.state_dict().items()}
    assert best_policy is not None
    output = dict(state)
    output["policy"] = best_policy
    output["pulse_distillation"] = {
        "source_checkpoint": str(args.checkpoint), "source_sha256": digest(args.checkpoint),
        "train_data": [{"path": str(p), "sha256": digest(p)} for p in args.pulse_data],
        "preserve_data": [{"path": str(p), "sha256": digest(p)} for p in args.preserve_data],
        "valid_data": {"path": str(args.valid_pulse_data), "sha256": digest(args.valid_pulse_data)},
        "successful_pulse_frames": len(positive), "preserve_frames": len(negative),
        "best_score": best, "requires_optimizer_reset": True,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(output, args.output)
    args.output.with_suffix(".json").write_text(
        json.dumps({"pulse_distillation": output["pulse_distillation"], "progress": progress}, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
