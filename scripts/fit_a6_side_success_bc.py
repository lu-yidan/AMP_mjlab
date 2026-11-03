"""Fine-tune a 738D side specialist from successful teacher episodes only."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch

from g1recovery_host.ppo import HostPolicy


def load_success(path: Path, seed_index: int):
    with np.load(path) as z:
        keep = z["successful_episode"].astype(bool)
        obs = z["actor_obs"][keep].astype(np.float32)
        action = z["target_action"][keep].astype(np.float32)
        direction = z["direction"][keep].astype(np.int64)
        episode = z["row"][keep].astype(np.int64) + seed_index * 1_000_000
        step = z["control_step"][keep].astype(np.float32)
    if obs.ndim != 2 or obs.shape[1] != 738 or action.shape != (len(obs), 29):
        raise ValueError(f"invalid teacher demonstration: {path}")
    return obs, action, direction, episode, step


def make_policy(state: dict, device: str) -> HostPolicy:
    policy = HostPolicy(
        738, 29, critic_obs_dim=786,
        log_std_bounds=(-2.3, math.log(state.get("action_std_max", 0.7))),
        upper_residual=True, waist_residual_scale=5.0,
    ).to(device)
    policy.load_state_dict(state["policy"], strict=True)
    return policy


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--main", type=Path, required=True)
    parser.add_argument("--demo", type=Path, action="append", required=True)
    parser.add_argument("--anchor", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--anchor-weight", type=float, default=0.3)
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 16 or args.learning_rate <= 0:
        parser.error("invalid training schedule")
    torch.manual_seed(42)
    rng = np.random.default_rng(42)
    parent = torch.load(args.parent, map_location="cpu", weights_only=False)
    main = torch.load(args.main, map_location="cpu", weights_only=False)
    required = (
        "observation_contract", "action_contract", "physics_revision",
        "a6_reset_variant", "model_source", "proprio_recovery_controller_sha256",
    )
    if any(parent.get(key) != main.get(key) for key in required):
        raise ValueError("parent specialist and main actor have different contracts")
    if parent.get("observation_contract") != "proprio_torque_actor_privileged_critic_v2":
        raise ValueError("success BC requires the 738D proprio actor")
    samples = [load_success(path, i) for i, path in enumerate(args.demo)]
    obs_np, action_np, direction_np, episode_np, step_np = (
        np.concatenate([sample[i] for sample in samples]) for i in range(5)
    )
    if len(obs_np) < 256 or len(np.unique(episode_np)) < 8:
        raise ValueError("too few successful side demonstrations")
    episodes = np.unique(episode_np)
    rng.shuffle(episodes)
    valid_episodes = episodes[:max(2, int(0.15 * len(episodes)))]
    valid = np.isin(episode_np, valid_episodes)
    train = ~valid
    if not train.any() or not valid.any():
        raise ValueError("episode split left no training or validation samples")
    with np.load(args.anchor) as z:
        anchor_keep = (
            (z["scene"] != 1) & (z["direction"] >= 2)
            & z["live_frame"].astype(bool)
        )
        anchor_np = z["actor_obs"][anchor_keep].astype(np.float32)
    if anchor_np.shape[1:] != (738,) or len(anchor_np) < 512:
        raise ValueError("non-guided side anchor data are missing")
    device = args.device
    obs = torch.from_numpy(obs_np).to(device)
    action = torch.from_numpy(action_np).to(device)
    anchor = torch.from_numpy(anchor_np).to(device)
    model = make_policy(parent, device)
    reference = make_policy(main, device).eval().requires_grad_(False)
    for critic in model.critics:
        critic.requires_grad_(False)
    model.log_std.requires_grad_(False)
    actor_parameters = list(model.actor.parameters()) + list(model.upper_residual.parameters())
    optimizer = torch.optim.AdamW(actor_parameters, lr=args.learning_rate, weight_decay=1e-4)
    train_ids = torch.from_numpy(np.flatnonzero(train)).to(device)
    valid_ids = torch.from_numpy(np.flatnonzero(valid)).to(device)
    history = []
    best_loss = float("inf")
    best_state = None
    stale = 0
    with torch.no_grad():
        anchor_target = reference.mean_action(anchor)
    for epoch in range(1, args.epochs + 1):
        model.train()
        for indices in train_ids[torch.randperm(len(train_ids), device=device)].split(args.batch_size):
            anchor_ids = torch.randint(len(anchor), (len(indices),), device=device)
            prediction = model.mean_action(obs[indices])
            bc_loss = (prediction - action[indices]).square().mean()
            anchor_loss = (model.mean_action(anchor[anchor_ids])
                           - anchor_target[anchor_ids]).square().mean()
            loss = bc_loss + args.anchor_weight * anchor_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(actor_parameters, 1.0)
            optimizer.step()
        model.eval()
        with torch.no_grad():
            validation = float((model.mean_action(obs[valid_ids])
                                - action[valid_ids]).square().mean())
            anchor_validation = float((model.mean_action(anchor[:1024])
                                       - anchor_target[:1024]).square().mean())
        history.append({"epoch": epoch, "demo_mse": validation,
                        "anchor_mse": anchor_validation})
        if validation < best_loss - 1e-4:
            best_loss = validation
            best_state = {key: value.detach().cpu().clone()
                          for key, value in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
        if epoch == 1 or epoch % 10 == 0:
            print(f"epoch={epoch} demo_mse={validation:.5f} anchor_mse={anchor_validation:.5f}", flush=True)
        if stale >= 15:
            break
    if best_state is None:
        raise RuntimeError("BC produced no finite model")
    output = dict(parent)
    output["policy"] = best_state
    output["reward_variant"] = parent["reward_variant"] + "_success_bc_v1"
    output["bc_success_demo"] = [str(path) for path in args.demo]
    output["bc_parent"] = str(args.parent)
    output["bc_anchor"] = str(args.anchor)
    output["bc_validation_mse"] = best_loss
    output["optimizer_after_bc_invalid"] = True
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(output, args.output)
    report = {
        "parent": str(args.parent), "main": str(args.main),
        "demo": [str(path) for path in args.demo],
        "train_episodes": int(len(episodes) - len(valid_episodes)),
        "valid_episodes": int(len(valid_episodes)),
        "train_samples": int(train.sum()), "valid_samples": int(valid.sum()),
        "train_left_samples": int(((direction_np == 2) & train).sum()),
        "train_right_samples": int(((direction_np == 3) & train).sum()),
        "best_demo_mse": best_loss, "last_anchor_mse": history[-1]["anchor_mse"],
        "history": history,
    }
    args.output.with_suffix(".json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "history"}, indent=2), flush=True)


if __name__ == "__main__":
    main()
