"""Offline probe: can a 738D proprio history identify the three A6 scenes?"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from g1recovery_host.proprio_scene import ProprioSceneClassifier
from g1recovery_host.proprio_symmetry import mirror_host29_actor_obs


def read(path: Path, include_imu_accel: bool = False, posture: str = "supine",
         min_control_step: int = 0, max_control_step: int = 120,
         canonicalize_side: bool = False):
    with np.load(path) as archive:
        obs = archive["actor_obs"].astype(np.float32)
        if include_imu_accel:
            imu = archive["imu_accel_history"].astype(np.float32)
            if imu.shape != (len(obs), 6, 3):
                raise ValueError("expected six 3D IMU acceleration frames per observation")
            obs = np.concatenate((obs, imu.reshape(len(obs), 18)), axis=1)
        scene = archive["scene"].astype(np.int64)
        direction = archive["direction"].astype(np.int64)
        safe = archive["supine_flat_recovery_safe"].astype(bool)
        live = archive["live_frame"].astype(bool)
        step = archive.get("control_step", None)
        episode_key = archive.get("episode_key", None)
    expected_dim = 756 if include_imu_accel else 738
    if obs.ndim != 2 or obs.shape[1] != expected_dim:
        raise ValueError(f"scene probe requires {expected_dim}D observations")
    # Board scenes after confirmed clearance no longer need a scene-specific
    # escape policy. Measure discrimination only while the board may obstruct.
    if posture == "supine":
        keep = live & (direction == 0) & ((scene == 0) | ~safe)
    elif posture == "prone":
        if step is None:
            raise ValueError("prone scene probe needs control-step labels")
        keep = live & (direction == 1) & (step >= min_control_step) & (step <= max_control_step)
    elif posture == "side":
        if step is None:
            raise ValueError("side scene probe needs control-step labels")
        keep = live & (direction >= 2) & (step >= min_control_step) & (step <= max_control_step)
    else:
        raise ValueError("posture must be supine, prone, or side")
    selected_obs = obs[keep].copy()
    if canonicalize_side:
        if posture != "side" or include_imu_accel:
            raise ValueError("side canonicalization requires 738D side observations")
        right_rows = direction[keep] == 3
        if right_rows.any():
            selected_obs[right_rows] = mirror_host29_actor_obs(
                torch.from_numpy(selected_obs[right_rows])
            ).numpy()
    return (
        selected_obs, scene[keep],
        step[keep] if step is not None else None,
        episode_key[keep] if episode_key is not None else None,
    )


def summarize(prob: np.ndarray, truth: np.ndarray, step: np.ndarray | None):
    predicted = prob.argmax(1)

    def group(mask):
        if not mask.any():
            return None
        y, p = truth[mask], predicted[mask]
        return {
            "count": int(mask.sum()),
            "accuracy": float((y == p).mean()),
            "confusion_true_rows_pred_cols": [
                [int(((y == i) & (p == j)).sum()) for j in range(3)]
                for i in range(3)
            ],
            "board_mistaken_for_flat_confident_095": int(
                ((y > 0) & (prob[mask, 0] >= 0.95)).sum()
            ),
            "flat_confident_095": int(((y == 0) & (prob[mask, 0] >= 0.95)).sum()),
        }

    groups = {"all_live_supine_preescape": group(np.ones(len(truth), bool))}
    if step is not None:
        for name, low, high in (
            ("steps_31_60", 31, 60),
            ("steps_61_90", 61, 90),
            ("steps_91_120", 91, 120),
            ("steps_121_plus", 121, 100000),
        ):
            groups[name] = group((step >= low) & (step <= high))
    return groups


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--valid", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--include-imu-accel", action="store_true")
    parser.add_argument("--save-model", type=Path,
                        help="save a 738D scene classifier for online safety evaluation")
    parser.add_argument("--posture", choices=("supine", "prone", "side"), default="supine")
    parser.add_argument("--min-control-step", type=int, default=0)
    parser.add_argument("--max-control-step", type=int, default=120)
    parser.add_argument("--canonicalize-side", action="store_true",
                        help="mirror right-side proprioception into left-side coordinates")
    args = parser.parse_args()
    if not 0 <= args.min_control_step <= args.max_control_step:
        parser.error("control-step training window is invalid")
    torch.manual_seed(42)
    x_np, y_np, _, _ = read(
        args.train, args.include_imu_accel, args.posture,
        args.min_control_step, args.max_control_step, args.canonicalize_side,
    )
    xv_np, yv_np, steps, episode_keys = read(
        args.valid, args.include_imu_accel, args.posture,
        args.min_control_step, args.max_control_step, args.canonicalize_side,
    )
    device = torch.device(args.device)
    x = torch.from_numpy(x_np).to(device)
    y = torch.from_numpy(y_np).to(device)
    xv = torch.from_numpy(xv_np).to(device)
    mean = x.mean(0)
    scale = x.std(0).clamp_min(1e-3)
    if args.save_model is not None and args.include_imu_accel:
        parser.error("online scene classifier requires the existing 738D actor input")
    x = ((x - mean) / scale).clamp(-10, 10)
    xv = ((xv - mean) / scale).clamp(-10, 10)
    net = nn.Sequential(
        nn.Linear(x.shape[1], 256), nn.SiLU(),
        nn.Linear(256, 128), nn.SiLU(), nn.Linear(128, 3),
    ).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=3e-4, weight_decay=1e-4)
    classes = [(y == i).nonzero(as_tuple=True)[0] for i in range(3)]
    if any(len(indices) == 0 for indices in classes):
        raise ValueError("training data must contain all three scenes")
    for epoch in range(args.epochs):
        net.train()
        # Balance scenes; the collection layout oversamples guided board.
        indices = torch.cat([
            cls[torch.randint(len(cls), (2048,), device=device)] for cls in classes
        ])[torch.randperm(6144, device=device)]
        for batch in indices.split(1024):
            loss = nn.functional.cross_entropy(net(x[batch]), y[batch])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        if epoch % 10 == 9:
            print(f"epoch={epoch + 1} loss={float(loss):.4f}", flush=True)
    net.eval()
    with torch.no_grad():
        prob = torch.softmax(net(xv), -1).cpu().numpy()
    report = {
        "train": str(args.train), "valid": str(args.valid),
        "include_imu_accel": args.include_imu_accel,
        "posture": args.posture,
        "min_control_step": args.min_control_step,
        "max_control_step": args.max_control_step,
        "canonicalize_side": args.canonicalize_side,
        "train_counts": {str(i): int((y_np == i).sum()) for i in range(3)},
        "valid_counts": {str(i): int((yv_np == i).sum()) for i in range(3)},
        "metrics": summarize(prob, yv_np, steps),
        "caveat": "offline scene classification; no online switch or safety claim",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    if episode_keys is not None:
        np.savez_compressed(
            args.output.with_suffix(".npz"),
            probability=prob, scene=yv_np, control_step=steps,
            episode_key=episode_keys,
        )
    if args.save_model is not None:
        model = ProprioSceneClassifier(mean, scale).to(device)
        model.network.load_state_dict(net.state_dict(), strict=True)
        args.save_model.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "observation_contract": "proprio_torque_actor_privileged_critic_v2",
            "label_order": ("flat", "guided", "free"),
            "mean": mean.cpu(), "scale": scale.cpu(),
            "model": model.state_dict(),
            "train": str(args.train), "valid": str(args.valid),
            "offline_report": str(args.output),
            "posture": args.posture,
            "canonicalize_side": args.canonicalize_side,
        }, args.save_model)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
