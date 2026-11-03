"""Capture guided-plate failures of a flat-ground HoST expert before A6 training."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.viewer import OffscreenRenderer
from PIL import Image

from g1recovery_host.a6_task import host_a6_env_cfg
from g1recovery_host.ppo import HostPolicy


LAYOUT = ((2, 0, 0), (1, 1, 0), (1, 0, 0), (0, 0, 0))
CASES = {1: "guided_prone", 2: "guided_supine"}
CAPTURE_STEPS = {30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 100, 150, 200, 300, 400, 500}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument(
        "--accept-early-a6",
        action="store_true",
        help="capture an early A6 failure separately; this checkpoint already uses target",
    )
    args = parser.parse_args()
    if args.steps < 1 or args.steps > 750:
        parser.error("steps must be in [1, 750]")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        parser.error("CUDA unavailable")
    path = args.checkpoint.resolve(strict=True)
    checkpoint = torch.load(path, map_location=args.device, weights_only=False)
    task = checkpoint.get("task")
    if (
        task not in (("flat29", "a6") if args.accept_early_a6 else ("flat29",))
        or checkpoint.get("model_source") != "unitree_g1_29"
        or "upper_residual.0.weight" not in checkpoint["policy"]
        or (task == "a6" and not checkpoint.get("reward_variant", "").startswith("historical_a6_"))
    ):
        parser.error("expected a pre-A6 flat expert or an explicitly accepted early A6 checkpoint")
    policy = HostPolicy(
        obs_dim=564,
        action_dim=29,
        log_std_bounds=(-2.3, math.log(checkpoint.get("action_std_max", 0.7))),
        upper_residual=True,
        waist_residual_scale=5.0,
    ).to(args.device)
    policy.load_state_dict(checkpoint["policy"], strict=True)
    policy.eval()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    cfg = host_a6_env_cfg(4, args.seed, arm="G+")
    cfg.events["reset_scene"].params["layout"] = LAYOUT
    # Explicit resets preserve the terminal state long enough to photograph
    # it and record whether the A6 safety rule actually fired.
    cfg.auto_reset = False
    cfg.episode_length_s = args.steps * 0.02
    env = ManagerBasedRlEnv(cfg, device=args.device)
    renderers: dict[int, OffscreenRenderer] = {}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "checkpoint": str(path),
        "checkpoint_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "checkpoint_task": checkpoint["task"],
        "checkpoint_iteration": checkpoint["iteration"],
        "evaluation": (
            "flat expert before A6 reward training in current fixed-6kg guided plate scenes"
            if task == "flat29"
            else "early A6 checkpoint (already uses target reward group) in current fixed-6kg guided plate scenes"
        ),
        "seed": args.seed,
        "device": args.device,
        "assist_force_n": 0.0,
        "scenes": {},
    }
    try:
        obs, _ = env.reset()
        actual = torch.stack((env._a6_scene, env._a6_direction, env._a6_stratum), dim=1)
        if not torch.equal(actual, torch.tensor(LAYOUT, device=env.device)):
            raise RuntimeError(f"unexpected layout: {actual.tolist()}")
        action_term = env.action_manager.get_term("joint_pos")
        beta = float(checkpoint["curriculum"]["beta"].float().mean())
        action_term.beta.fill_(beta)
        action_term.assist_force.zero_()
        action_term.guided_escape_assist_force.zero_()
        action_term.guided_escape_assist_speed_target.zero_()
        metadata["action_beta"] = beta
        for idx in CASES:
            view = copy.deepcopy(env.cfg.viewer)
            view.env_idx = idx
            view.max_extra_envs = 0
            view.width = 640
            view.height = 480
            view.distance = 2.8
            view.azimuth = 120.0
            view.elevation = -18.0
            view.enable_shadows = False
            origin = env.scene.env_origins[idx].cpu().numpy()
            view.lookat = tuple((origin + np.array((0.0, 0.0, 0.6))).tolist())
            renderer = OffscreenRenderer(
                env.sim.mj_model,
                view,
                env.scene,
                sim_model=env.sim.model,
                expanded_fields=env.sim.expanded_fields,
            )
            renderer.initialize()
            renderers[idx] = renderer
            metadata["scenes"][CASES[idx]] = {"frames": [], "first_terminal_step": None}

        for step in range(1, args.steps + 1):
            with torch.inference_mode():
                action = policy.deterministic_action(obs["actor"])
                obs, _, terminated, truncated, _ = env.step(action)
            for idx, label in CASES.items():
                case = metadata["scenes"][label]
                ended = bool(terminated[idx] or truncated[idx])
                if ended and case["first_terminal_step"] is None:
                    case["first_terminal_step"] = step
                    case["invalid_at_terminal"] = bool(env._a6_invalid[idx])
                    case["escaped_at_terminal"] = bool(env._a6_escaped[idx])
                if case["first_terminal_step"] is not None and step > case["first_terminal_step"]:
                    continue
                if step not in CAPTURE_STEPS and not ended:
                    continue
                renderer = renderers[idx]
                renderer.update(env.sim.data)
                frame = np.ascontiguousarray(renderer.render())
                prefix = "pre_target" if task == "flat29" else "early_a6"
                file = args.output_dir / f"{prefix}_{label}_step{step:03d}.png"
                Image.fromarray(frame).save(file)
                case["frames"].append(str(file))
                case["escaped"] = bool(env._a6_escaped[idx])
                case["invalid"] = bool(env._a6_invalid[idx])
                case["pelvis_height_m"] = float(env.scene["robot"].data.root_link_pos_w[idx, 2])
                case["root_xy_m"] = env.scene["robot"].data.root_link_pos_w[idx, :2].tolist()
            if step % 100 == 0:
                print(f"step {step}/{args.steps}", flush=True)
            if all(metadata["scenes"][name]["first_terminal_step"] is not None for name in CASES.values()):
                break
            done_ids = (terminated | truncated).nonzero(as_tuple=False).squeeze(-1)
            if len(done_ids):
                with torch.inference_mode():
                    obs, _ = env.reset(env_ids=done_ids)
        (args.output_dir / "capture_metadata.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print(json.dumps(metadata, ensure_ascii=False, indent=2), flush=True)
    finally:
        for renderer in renderers.values():
            renderer.close()
        env.close()


if __name__ == "__main__":
    main()
