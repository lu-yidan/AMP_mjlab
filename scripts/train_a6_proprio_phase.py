"""Fit a proprio-only phase detector; report held-out scene error at each threshold."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from g1recovery_host.proprio_phase import ProprioPhaseDetector


def read(paths: list[Path], label_key: str):
    items = [np.load(path) for path in paths]
    arrays = tuple(np.concatenate([item[key] for item in items]) for key in (
        "actor_obs", label_key, "scene", "direction", "teacher_driven"
    ))
    live = np.concatenate([
        item["live_frame"] if "live_frame" in item
        else np.ones(len(item["actor_obs"]), dtype=np.bool_)
        for item in items
    ])
    return (*arrays, live)


def sha256(path: Path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def metrics(prob, label, scene, direction, driven, label_key="escaped_prone"):
    if label_key == "escaped_prone":
        groups = {
            "all": np.ones_like(label, dtype=bool),
            "guided_prone_teacher": (scene == 1) & (direction == 1) & driven,
            "guided_prone_student": (scene == 1) & (direction == 1) & ~driven,
            "guided_other": (scene == 1) & (direction != 1),
            "free_or_flat": (scene != 1),
        }
    else:
        groups = {
            "all": np.ones_like(label, dtype=bool),
            "guided_supine": (scene == 1) & (direction == 0),
            "guided_prone": (scene == 1) & (direction == 1),
            "free_supine": (scene == 2) & (direction == 0),
            "flat_supine": (scene == 0) & (direction == 0),
            "other": direction > 0,
        }
    result = {}
    for threshold in (0.5, 0.9, 0.95, 0.98, 0.99, 0.995, 0.999):
        selected = prob >= threshold
        by_group = {}
        for name, mask in groups.items():
            positive = int((mask & label).sum())
            negative = int((mask & ~label).sum())
            tp = int((mask & label & selected).sum())
            fp = int((mask & ~label & selected).sum())
            by_group[name] = {
                "n": int(mask.sum()), "positive": positive,
                "recall": round(tp / positive, 4) if positive else None,
                "false_positive_rate": round(fp / negative, 6) if negative else None,
                "false_positive_count": fp,
            }
        result[str(threshold)] = by_group
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, nargs="+", required=True)
    parser.add_argument("--valid", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--supine-only-training", action="store_true",
                        help="fit and select on supine frames; keep all postures in safety metrics")
    parser.add_argument("--guided-negative-weight", type=float, default=1.0)
    parser.add_argument("--free-negative-weight", type=float, default=1.0,
                        help="cost of predicting safe while still under a movable plate")
    parser.add_argument("--free-positive-weight", type=float, default=1.0)
    parser.add_argument(
        "--label-key", choices=("escaped_prone", "supine_flat_recovery_safe"),
        default="escaped_prone",
    )
    args = parser.parse_args()
    if args.supine_only_training and args.label_key != "supine_flat_recovery_safe":
        parser.error("supine-only training requires the supine flat-recovery label")
    if (args.guided_negative_weight < 1 or args.free_negative_weight < 1
            or args.free_positive_weight < 1):
        parser.error("recovery group weights must be at least one")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    x, y, train_scene, train_direction, _, live = read(args.train, args.label_key)
    xv, yv, sv, dv, tv, live_valid = read([args.valid], args.label_key)
    if args.label_key == "supine_flat_recovery_safe":
        train_mask = live & ((train_direction == 0) if args.supine_only_training else True)
        x, y = x[train_mask], y[train_mask]
        train_scene = train_scene[train_mask]
        xv, yv = xv[live_valid], yv[live_valid]
        sv, dv, tv = sv[live_valid], dv[live_valid], tv[live_valid]
    if x.shape[1] != 738 or xv.shape[1] != 738:
        parser.error("dataset and checkpoint must have 738D actor observations")
    train = torch.as_tensor(x.astype(np.float32), device=args.device)
    labels = torch.as_tensor(y.astype(np.float32), device=args.device)
    train_weights = np.ones(len(y), dtype=np.float32)
    train_weights[(train_scene == 1) & ~y] = args.guided_negative_weight
    train_weights[(train_scene == 2) & ~y] = args.free_negative_weight
    train_weights[(train_scene == 2) & y] = args.free_positive_weight
    train_weights = torch.as_tensor(train_weights, device=args.device)
    valid = torch.as_tensor(xv.astype(np.float32), device=args.device)
    mean, scale = train.mean(0), train.std(0).clamp_min(1e-3)
    model = ProprioPhaseDetector(mean, scale).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    best = float("inf")
    best_state = None
    for epoch in range(args.epochs):
        model.train()
        order = torch.randperm(len(train), device=args.device)
        for batch in order.split(args.batch_size):
            per_sample = F.binary_cross_entropy_with_logits(
                model(train[batch]), labels[batch], reduction="none"
            )
            loss = (per_sample * train_weights[batch]).sum() / train_weights[batch].sum()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            logits = torch.cat([model(batch) for batch in valid.split(args.batch_size)])
            selection_np = (
                dv == 0 if args.supine_only_training
                else np.ones(len(dv), dtype=bool)
            )
            selection = torch.as_tensor(selection_np, device=args.device)
            valid_weights = np.ones(len(yv), dtype=np.float32)
            valid_weights[(sv == 1) & ~yv] = args.guided_negative_weight
            valid_weights[(sv == 2) & ~yv] = args.free_negative_weight
            valid_weights[(sv == 2) & yv] = args.free_positive_weight
            valid_weights = torch.as_tensor(valid_weights, device=args.device)
            valid_per_sample = F.binary_cross_entropy_with_logits(
                logits[selection],
                torch.as_tensor(yv[selection_np].astype(np.float32), device=args.device),
                reduction="none",
            )
            valid_loss = (
                (valid_per_sample * valid_weights[selection]).sum()
                / valid_weights[selection].sum()
            ).item()
        print(f"epoch={epoch + 1} valid_bce={valid_loss:.5f}", flush=True)
        if valid_loss < best:
            best = valid_loss
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    assert best_state is not None
    model.load_state_dict(best_state)
    with torch.no_grad():
        prob = torch.cat([model(batch).sigmoid() for batch in valid.split(args.batch_size)]).cpu().numpy()
    result = {
        "train": [str(p) for p in args.train], "valid": str(args.valid),
        "train_samples": len(x), "valid_samples": len(xv),
        "best_valid_bce": best,
        "label_key": args.label_key,
        "supine_only_training": args.supine_only_training,
        "guided_negative_weight": args.guided_negative_weight,
        "free_negative_weight": args.free_negative_weight,
        "free_positive_weight": args.free_positive_weight,
        "thresholds": metrics(prob, yv, sv, dv, tv, args.label_key),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model": best_state, "mean": mean.cpu(), "scale": scale.cpu(),
        "observation_contract": "proprio_torque_actor_privileged_critic_v2",
        "label_key": args.label_key,
        "train_sha256": [sha256(p) for p in args.train],
        "valid_sha256": sha256(args.valid),
        "metrics": result,
    }, args.output)
    args.output.with_suffix(".json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result["thresholds"], indent=2), flush=True)


if __name__ == "__main__":
    main()
