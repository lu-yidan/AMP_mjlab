"""Probe whether body observations can identify a still-constrained guided board."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from g1recovery_host.proprio_phase import ProprioPhaseDetector


def read(paths: list[Path]):
    x, y, scene = [], [], []
    for path in paths:
        with np.load(path) as data:
            mask = data["live_frame"] & (data["direction"] == 0)
            x.append(data["actor_obs"][mask].astype(np.float32))
            scene.append(data["scene"][mask])
            y.append(((data["scene"] == 1) & ~data["supine_flat_recovery_safe"])[mask])
    return np.concatenate(x), np.concatenate(y), np.concatenate(scene)


def report(model, paths: list[Path], device: str):
    result = {}
    for path in paths:
        x, y, scene = read([path])
        with torch.no_grad():
            features = torch.as_tensor(x, device=device)
            prob = torch.cat([
                model(batch).sigmoid() for batch in features.split(4096)
            ]).cpu().numpy()
        thresholds = {}
        for threshold in (0.01, 0.02, 0.05, 0.1, 0.2, 0.5):
            predicts_safe = prob < threshold
            thresholds[str(threshold)] = {
                "guided_trapped_false_safe": int((y & predicts_safe).sum()),
                "guided_trapped_total": int(y.sum()),
                "free_allowed": int(((scene == 2) & predicts_safe).sum()),
                "free_total": int((scene == 2).sum()),
                "flat_allowed": int(((scene == 0) & predicts_safe).sum()),
                "flat_total": int((scene == 0).sum()),
            }
        result[str(path)] = thresholds
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, nargs="+", required=True)
    parser.add_argument("--valid", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    x, y, _ = read(args.train)
    xv, yv, _ = read([args.valid[0]])
    train = torch.as_tensor(x, device=args.device)
    labels = torch.as_tensor(y.astype(np.float32), device=args.device)
    valid = torch.as_tensor(xv, device=args.device)
    valid_labels = torch.as_tensor(yv.astype(np.float32), device=args.device)
    mean, scale = train.mean(0), train.std(0).clamp_min(1e-3)
    model = ProprioPhaseDetector(mean, scale).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    class_balance = (1 - labels.mean()) / labels.mean().clamp_min(1e-3)
    best, best_state = float("inf"), None
    for epoch in range(args.epochs):
        model.train()
        for batch in torch.randperm(len(train), device=args.device).split(2048):
            loss = F.binary_cross_entropy_with_logits(
                model(train[batch]), labels[batch], pos_weight=class_balance
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            valid_logits = torch.cat([model(batch) for batch in valid.split(4096)])
            value = F.binary_cross_entropy_with_logits(
                valid_logits, valid_labels, pos_weight=class_balance
            ).item()
        print(f"epoch={epoch + 1} valid_bce={value:.5f}", flush=True)
        if value < best:
            best = value
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }
    model.load_state_dict(best_state)
    result = {"best_valid_bce": best, "metrics": report(model, args.valid, args.device)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model": best_state, "mean": mean.cpu(), "scale": scale.cpu(),
        "observation_contract": "proprio_torque_actor_privileged_critic_v2",
        "label_key": "guided_trapped_supine",
        "metrics": result,
    }, args.output)
    args.output.with_suffix(".json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
