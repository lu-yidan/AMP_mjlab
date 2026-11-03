"""Record the four HoST A6 playback scenes as MP4s and a labeled contact sheet."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path

import imageio_ffmpeg
import numpy as np
import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.viewer import OffscreenRenderer
from PIL import Image, ImageDraw, ImageFont

from g1recovery_host.a6_task import host_a6_env_cfg
from g1recovery_host.phase_switch import (
    postescape_live_observation_enabled,
    reset_prone_history_after_escape,
)
from g1recovery_host.ppo import HostPolicy
from g1recovery_host.proprio_recovery import (
    InitialProneRouter,
    InitialSideRouter,
    load_prone_specialist,
    load_recovery_controller,
    load_supine_flat_gate,
    validate_controller_provenance,
)
from g1recovery_host.proprio_scene import (
    ProprioGuidedSideGate,
    ProprioSceneGate,
    load_scene_classifier,
)
from play_a6_host29 import DeterministicHostPolicy

LAYOUT = ((2, 0, 0), (1, 1, 0), (1, 0, 0), (0, 0, 0))
LABELS = ("Free / supine", "Guided / prone", "Guided / supine", "Flat / supine")
SLUGS = ("free_supine", "guided_prone", "guided_supine", "flat_supine")
V62_COMPONENTS = {
    "main": "logs/host_a6_free_specialist_v17_from374_seed42_525/model_00499.pt",
    "prone": "logs/host_a6_guided_prone_specialist_v18_from449_seed42_600/model_00474.pt",
    "side": "logs/host_a6_side_teacher_all_v33_gpu4_from549_seed42_700/model_00624.pt",
    "side_gate": "artifacts/a6_proprio_phase_detector/side_scene_canonical_v48/side_early31_45_seed48.pt",
    "right_side_gate": "artifacts/a6_proprio_phase_detector/side_scene_mirror_v44/side_v17_mirror_seed48.pt",
    "prone_scene_gate": "artifacts/a6_proprio_phase_detector/scene_id_prone_v1/prone_scene_shuffled_seed42_to44.pt",
    "recovery": "artifacts/a6_proprio_phase_detector/proprio_recovery_controller_v1.pt",
    "supine_flat_gate": "artifacts/a6_proprio_phase_detector/recovery_gate_v1/supine_flat_recovery_gate_unweighted_v1.pt",
    "flat": "logs/host_ground29_pose_v6_from2999_4096_seed42_1000/model_00999.pt",
}


def make_environment(checkpoint: dict, seed: int, device: str,
                     auto_reset: bool = False) -> ManagerBasedRlEnv:
    variant = checkpoint["reward_variant"]
    torque_v2 = checkpoint.get("observation_contract") == "proprio_torque_actor_privileged_critic_v2"
    proprio_actor = torque_v2 or checkpoint.get("observation_contract") == "proprio_actor_privileged_critic_v1"
    if (
        checkpoint.get("task") != "a6"
        or checkpoint.get("model_source") != "unitree_g1_29"
        or "_staged_escape_v5" not in variant
        or "_v8_edge_curriculum_v1" not in variant
    ):
        raise ValueError("recording requires a staged-v5, 29-DoF HoST A6 checkpoint")
    cfg = host_a6_env_cfg(
        4,
        seed,
        arm=checkpoint["arm"],
        edge_curriculum=True,
        plate_observation=not proprio_actor,
        critic_plate_observation=torque_v2,
        proprio_motor_torque=torque_v2,
        safe_actions="_safe_actions_v1" in variant,
        prone_full_arms="_prone_full_arms_v1" in variant,
        supine_full_arms="_supine_full_arms_v1" in variant,
        aligned_edge="_aligned_edge_v1" in variant,
        lateral_actions="_lateral_actions_v1" in variant,
        lift_guard="_lift_guard_v1" in variant,
        privileged_guided_actions=not proprio_actor,
        footprint_edge="_footprint_edge_v1" in variant,
        footprint_overlap_m=checkpoint.get("a6_footprint_overlap_m", 0.25),
    )
    cfg.events["reset_scene"].params["layout"] = LAYOUT
    # Keep the last recorded frame before the episode timeout in v62 mode.
    cfg.episode_length_s = 15.02 if auto_reset else 15.0
    cfg.auto_reset = auto_reset
    env = ManagerBasedRlEnv(cfg, device=device)
    env._a6_staged_escape_v5 = True
    env._a6_knee_straight_prior_v1 = "_knee_straight_prior_v1" in variant
    env._a6_arm_extraction_v1 = "_arm_extraction_v1" in variant
    env._a6_prone_flexible_exit_v1 = "_prone_flexible_exit_v1" in variant
    env._a6_scene_phase_reward_v1 = "_scene_phase_reward_v1" in variant
    env._a6_side_lift_anticipation_v1 = "_side_lift_anticipation_v1" in variant
    env._a6_side_directional_progress_v1 = "_side_directional_progress_v1" in variant
    env._a6_flat_prone_compact_v1 = "_flat_prone_compact_v1" in variant
    env._a6_free_plate_motion_reward_v1 = (
        "_free_plate_motion_reward_v1" in variant
    )
    env._a6_free_plate_anchor_v1 = "_free_plate_anchor_v1" in variant
    env._a6_prone_recovery_v1 = "_prone_recovery_v1" in variant
    env._a6_bidirectional_recovery_v1 = "_bidirectional_recovery_v1" in variant
    env._a6_strong_recovery_v1 = "_strong_recovery_v1" in variant
    env._a6_recovery_bridge_v2 = "_recovery_bridge_v2" in variant
    env._a6_postescape_live_obs_v1 = postescape_live_observation_enabled(checkpoint)
    env._a6_edge_offset_m = 0.0
    return env


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=750)
    parser.add_argument("--render-stride", type=int, default=2)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--composite-v62", action="store_true",
                        help="record the exact proprio-only v62 playback controller")
    parser.add_argument("--proprio-recovery-controller", type=Path)
    parser.add_argument("--proprio-supine-flat-gate", type=Path)
    parser.add_argument("--flat-expert-checkpoint", type=Path)
    parser.add_argument("--flat-gate-threshold", type=float, default=0.8)
    parser.add_argument("--flat-gate-consecutive", type=int, default=3)
    parser.add_argument("--flat-gate-min-steps", type=int, default=120)
    args = parser.parse_args()
    if (args.proprio_supine_flat_gate is None) != (args.flat_expert_checkpoint is None):
        parser.error("supine flat gate and flat expert checkpoint must be supplied together")
    if args.steps <= 0 or args.steps > 750 or args.render_stride <= 0:
        parser.error("steps must be in [1, 750] and render stride must be positive")
    if args.width % 16 or args.height % 16 or args.width < 320 or args.height < 240:
        parser.error("width/height must be multiples of 16 and at least 320x240")
    if not args.dry_run and args.output_dir is None:
        parser.error("--output-dir is required when recording")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        parser.error("CUDA is unavailable")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    checkpoint_path = args.checkpoint.resolve(strict=True)
    if args.composite_v62:
        expected = (Path(__file__).resolve().parents[1] / V62_COMPONENTS["main"]).resolve()
        if checkpoint_path != expected:
            parser.error(f"v62 requires the current main actor: {expected}")
        if args.proprio_recovery_controller or args.proprio_supine_flat_gate or args.flat_expert_checkpoint:
            parser.error("v62 components are fixed; omit legacy controller overrides")
        args.proprio_recovery_controller = Path(V62_COMPONENTS["recovery"])
        args.proprio_supine_flat_gate = Path(V62_COMPONENTS["supine_flat_gate"])
        args.flat_expert_checkpoint = Path(V62_COMPONENTS["flat"])
    checksum = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
    checkpoint = torch.load(checkpoint_path, map_location=args.device, weights_only=False)
    validate_controller_provenance(checkpoint, args.proprio_recovery_controller)
    variant = checkpoint["reward_variant"]
    torque_v2 = checkpoint.get("observation_contract") == "proprio_torque_actor_privileged_critic_v2"
    proprio_actor = torque_v2 or checkpoint.get("observation_contract") == "proprio_actor_privileged_critic_v1"
    if args.proprio_recovery_controller is not None and (
        not torque_v2 or "_phase_history_reset_v1" in variant
    ):
        parser.error("packaged recovery controller requires the 738D proprio actor")
    if args.proprio_supine_flat_gate is not None and (
        not torque_v2 or "_phase_history_reset_v1" in variant
    ):
        parser.error("supine flat gate requires the 738D proprio actor without legacy phase reset")
    if proprio_actor and (
        checkpoint.get("actor_obs_dim") != (738 if torque_v2 else 564)
        or checkpoint.get("critic_obs_dim") != (786 if torque_v2 else 612)
        or checkpoint.get("action_contract") != "scene_independent_v1"
        or "_safe_actions_v1" in variant
        or "_lift_guard_v1" in variant
        or "_phase_history_reset_v1" in variant
    ):
        parser.error("proprioceptive checkpoint has an incompatible observation or action contract")
    policy = HostPolicy(
        obs_dim=738 if torque_v2 else 564 if proprio_actor else 612,
        critic_obs_dim=(786 if torque_v2 else 612) if proprio_actor else None,
        action_dim=29,
        log_std_bounds=(-2.3, math.log(checkpoint.get("action_std_max", 0.7))),
        upper_residual=True,
        waist_residual_scale=5.0,
    ).to(args.device)
    policy.load_state_dict(checkpoint["policy"], strict=True)
    policy.eval()
    flat_expert = None
    if args.flat_expert_checkpoint is not None:
        source = torch.load(
            args.flat_expert_checkpoint, map_location=args.device, weights_only=False
        )
        if source.get("task") != "flat29" or source.get("model_source") != "unitree_g1_29":
            parser.error("flat expert must be a 29-DoF flat checkpoint")
        flat_expert = HostPolicy(
            564, 29,
            log_std_bounds=(-2.3, math.log(source.get("action_std_max", 0.7))),
            upper_residual=True, waist_residual_scale=5.0,
        ).to(args.device)
        flat_expert.load_state_dict(source["policy"], strict=True)
        flat_expert.eval()

    prefix = f"HoST_{'v62_' if args.composite_v62 else ''}model_{checkpoint['iteration']:05d}_seed{args.seed}"
    paths: list[Path] = []
    if not args.dry_run:
        assert args.output_dir is not None
        args.output_dir.mkdir(parents=True, exist_ok=True)
        paths = [args.output_dir / f"{prefix}_{slug}.mp4" for slug in SLUGS]
        paths += [args.output_dir / f"{prefix}_four_scenes.mp4"]
        paths += [args.output_dir / f"{prefix}_poster.png"]
        paths += [args.output_dir / f"{prefix}_metrics.json"]
        existing = [str(path) for path in paths if path.exists()]
        if existing:
            raise FileExistsError(f"Refusing to overwrite existing output: {existing}")

    renderers: list[OffscreenRenderer] = []
    writers = []
    env = make_environment(checkpoint, args.seed, args.device,
                           auto_reset=args.composite_v62)
    recovery_controller = None
    if args.proprio_recovery_controller is not None:
        recovery_controller, artifact = load_recovery_controller(
            args.proprio_recovery_controller, env.num_envs, args.device
        )
        if artifact["observation_contract"] != checkpoint["observation_contract"]:
            raise ValueError("controller and checkpoint observation contracts differ")
    flat_gate = None
    if args.proprio_supine_flat_gate is not None:
        flat_gate, gate_artifact = load_supine_flat_gate(
            args.proprio_supine_flat_gate, env.num_envs, args.device,
            threshold=args.flat_gate_threshold,
            consecutive=args.flat_gate_consecutive,
            min_steps=args.flat_gate_min_steps,
        )
        if gate_artifact["observation_contract"] != checkpoint["observation_contract"]:
            raise ValueError("flat gate and checkpoint observation contracts differ")
    playback_policy = None
    prone_router = side_router = guided_side_gate = prone_flat_gate = prone_free_gate = None
    if args.composite_v62:
        prone_expert, _ = load_prone_specialist(
            V62_COMPONENTS["prone"], checkpoint, args.device,
        )
        side_expert, _ = load_prone_specialist(
            V62_COMPONENTS["side"], checkpoint, args.device,
        )
        prone_router = InitialProneRouter(env.num_envs, args.device)
        side_router = InitialSideRouter(env.num_envs, args.device)
        classifier, artifact = load_scene_classifier(V62_COMPONENTS["side_gate"], args.device)
        right_classifier, right_artifact = load_scene_classifier(
            V62_COMPONENTS["right_side_gate"], args.device,
        )
        if any(item["observation_contract"] != checkpoint["observation_contract"]
               or item.get("posture") != "side" for item in (artifact, right_artifact)):
            raise ValueError("v62 side gates do not match the actor observation contract")
        guided_side_gate = ProprioGuidedSideGate(
            classifier, env.num_envs, args.device,
            threshold=0.90, consecutive=2, min_steps=31, max_steps=45,
            ema_alpha=0.2, right_classifier=right_classifier,
            right_min_steps=35, right_max_steps=120,
        )
        classifier, artifact = load_scene_classifier(
            V62_COMPONENTS["prone_scene_gate"], args.device,
        )
        if (artifact["observation_contract"] != checkpoint["observation_contract"]
                or artifact.get("posture") != "prone"):
            raise ValueError("v62 prone gate does not match the actor observation contract")
        prone_flat_gate = ProprioSceneGate(
            classifier, env.num_envs, args.device, threshold=0.99,
            consecutive=3, min_steps=31, max_steps=50,
            target_class=0, posture="prone",
        )
        prone_free_gate = ProprioSceneGate(
            classifier, env.num_envs, args.device, threshold=0.99,
            consecutive=3, min_steps=31, max_steps=100,
            target_class=2, posture="prone",
        )
        playback_policy = DeterministicHostPolicy(
            policy, env, flat_expert, flat_gate, prone_expert, prone_router,
            side_expert, side_router, guided_side_gate,
            mirror_left_side=True, mirror_right_main=True,
            side_safe_through_step=500, side_safe_scale=0.5,
            side_lateral_gain=1.10, prone_flat_gate=prone_flat_gate,
            recovery_controller=recovery_controller, prone_flat_pulse_step=50,
            prone_free_gate=prone_free_gate, prone_free_pulse_step=100,
        )
    camera_centers: list[np.ndarray] = []
    camera_origins: list[np.ndarray] = []
    try:
        term = env.action_manager.get_term("joint_pos")
        term.beta.fill_(float(checkpoint["curriculum"]["beta"].float().mean()))
        term.assist_force.zero_()
        term.guided_escape_assist_force.zero_()
        term.guided_escape_assist_speed_target.zero_()
        obs, _ = env.reset()
        if args.composite_v62:
            gravity = env.scene["robot"].data.projected_gravity_b
            for gate in (prone_router, side_router, guided_side_gate,
                         prone_flat_gate, prone_free_gate):
                gate.arm_initial_gravity(gravity)
        if flat_gate is not None:
            flat_gate.arm_initial_gravity(
                env.scene["robot"].data.projected_gravity_b
            )
        actual = torch.stack(
            (env._a6_scene, env._a6_direction, env._a6_stratum), dim=1
        )
        expected = torch.tensor(LAYOUT, device=args.device)
        if not torch.equal(actual, expected):
            raise RuntimeError(f"Scene layout changed: {actual.tolist()}")

        if not args.dry_run:
            for index in range(4):
                view = copy.deepcopy(env.cfg.viewer)
                view.env_idx = index
                view.max_extra_envs = 0
                view.width = args.width
                view.height = args.height
                view.distance = 2.8
                view.azimuth = 120.0
                view.elevation = -18.0
                origin = env.scene.env_origins[index].cpu().numpy()
                view.lookat = tuple((origin + np.array((0.0, 0.0, 0.6))).tolist())
                view.enable_shadows = False
                renderer = OffscreenRenderer(
                    env.sim.mj_model,
                    view,
                    env.scene,
                    sim_model=env.sim.model,
                    expanded_fields=env.sim.expanded_fields,
                )
                renderer.initialize()
                renderers.append(renderer)
                camera_origins.append(origin)
                camera_centers.append(origin + np.array((0.0, 0.0, 0.6)))
            for path in paths[:5]:
                size = (
                    (args.width * 2, args.height * 2)
                    if path == paths[4]
                    else (args.width, args.height)
                )
                writer = imageio_ffmpeg.write_frames(
                    str(path),
                    size,
                    fps=50 / args.render_stride,
                    quality=7,
                    codec="libx264",
                    output_params=["-movflags", "+faststart"],
                )
                writer.send(None)
                writers.append(writer)

        first_escape = [0] * 4
        first_flat_gate_trigger = [0] * 4
        first_invalid = [0] * 4
        ended_early = [False] * 4
        reset_counts = [0] * 4
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", 21)
        recorded = 0
        executed = 0
        for step in range(1, args.steps + 1):
            with torch.inference_mode():
                if flat_gate is not None and not args.composite_v62:
                    gate_trigger = flat_gate.observe(obs["actor"], term.real_steps)
                    for index in gate_trigger.nonzero(as_tuple=False).squeeze(-1).tolist():
                        first_flat_gate_trigger[index] = step
                action = (playback_policy(obs) if playback_policy is not None
                          else policy.deterministic_action(obs["actor"]))
                if flat_gate is not None and not args.composite_v62:
                    flat_action = flat_expert.deterministic_action(obs["actor"][:, :564])
                    action = torch.where(flat_gate.triggered[:, None], flat_action, action)
                if recovery_controller is not None:
                    action, _ = recovery_controller.apply(action)
                if not torch.isfinite(action).all():
                    raise FloatingPointError("Non-finite policy action")
                escaped_before = env._a6_escaped.clone()
                obs, _, terminated, truncated, _ = env.step(action)
                done = terminated | truncated
                if args.composite_v62 and done.any():
                    ids = done.nonzero(as_tuple=False).squeeze(-1)
                    gravity = env.scene["robot"].data.projected_gravity_b
                    for gate in (prone_router, side_router, guided_side_gate,
                                 prone_flat_gate, prone_free_gate, flat_gate):
                        gate.reset(ids)
                        gate.arm_initial_gravity(gravity, ids)
                    recovery_controller.reset(ids)
                if recovery_controller is not None:
                    recovery_controller.observe(
                        obs["actor"], term.real_steps,
                        eligible=~done,
                    )
                if flat_gate is not None and not args.composite_v62:
                    if done.any():
                        flat_gate.reset(done.nonzero(as_tuple=False).squeeze(-1))
                if "_phase_history_reset_v1" in variant:
                    obs, _ = reset_prone_history_after_escape(
                        env, escaped_before, terminated | truncated
                    )
            executed = step
            for index in range(4):
                if not ended_early[index] and bool(env._a6_escaped[index]) and not first_escape[index]:
                    first_escape[index] = step
                invalid_now = bool(terminated[index]) if args.composite_v62 else bool(env._a6_invalid[index])
                if invalid_now and not first_invalid[index]:
                    first_invalid[index] = step
                if (bool(terminated[index]) or bool(truncated[index])) and step < args.steps:
                    ended_early[index] = True
                    reset_counts[index] += 1

            if renderers and step % args.render_stride == 0:
                frames = []
                roots = env.scene["robot"].data.root_link_pos_w.detach().cpu().numpy()
                for index, (renderer, writer) in enumerate(zip(renderers, writers[:4], strict=True)):
                    target = camera_origins[index].copy()
                    target[:2] = 0.8 * roots[index, :2] + 0.2 * camera_origins[index][:2]
                    target[2] += 0.6
                    camera_centers[index] += 0.12 * (target - camera_centers[index])
                    renderer._cam.lookat[:] = camera_centers[index]
                    renderer.update(env.sim.data)
                    frame = np.ascontiguousarray(renderer.render())
                    writer.send(frame)
                    frames.append(frame)
                panel = Image.new("RGB", (args.width * 2, args.height * 2))
                for index, frame in enumerate(frames):
                    tile = Image.fromarray(frame)
                    draw = ImageDraw.Draw(tile)
                    draw.rectangle((0, 0, args.width, 34), fill=(20, 25, 30))
                    reset_label = f"   resets={reset_counts[index]}" if reset_counts[index] else ""
                    draw.text((12, 5), f"{LABELS[index]}   t={step / 50:.1f}s{reset_label}", font=font, fill="white")
                    panel.paste(tile, ((index % 2) * args.width, (index // 2) * args.height))
                writers[4].send(np.ascontiguousarray(np.asarray(panel)))
                if recorded == 0:
                    panel.save(paths[5])
                recorded += 1
            if any(ended_early) and not args.composite_v62:
                break
            if step % 100 == 0:
                print(f"seed={args.seed} step={step}/{args.steps}", flush=True)

        robot = env.scene["robot"].data
        heights = robot.root_link_pos_w[:, 2].detach().cpu().tolist()
        upright = (-robot.projected_gravity_b[:, 2]).detach().cpu().tolist()
        scenes = {}
        for index, slug in enumerate(SLUGS):
            stood = heights[index] > 0.6 and upright[index] > 0.8
            scenes[slug] = {
                "escaped": first_escape[index] > 0,
                "first_escape_step": first_escape[index],
                "first_flat_gate_trigger_step": first_flat_gate_trigger[index],
                "invalid": first_invalid[index] > 0,
                "first_invalid_step": first_invalid[index],
                "ended_early": ended_early[index],
                "reset_count": reset_counts[index],
                "stood_at_end": stood,
                "success": stood and not ended_early[index]
                and (index == 3 or first_escape[index] > 0),
                "final_pelvis_height_m": round(heights[index], 4),
                "final_upright": round(upright[index], 4),
            }
        result = {
            "checkpoint": str(checkpoint_path),
            "checkpoint_sha256": checksum,
            "controller_version": "v62" if args.composite_v62 else "legacy_recorder",
            "composite_components": V62_COMPONENTS if args.composite_v62 else None,
            "proprio_recovery_controller": (
                str(args.proprio_recovery_controller)
                if args.proprio_recovery_controller is not None else None
            ),
            "proprio_supine_flat_gate": (
                str(args.proprio_supine_flat_gate)
                if args.proprio_supine_flat_gate is not None else None
            ),
            "flat_expert_checkpoint": (
                str(args.flat_expert_checkpoint)
                if args.flat_expert_checkpoint is not None else None
            ),
            "flat_gate_threshold": (
                args.flat_gate_threshold if flat_gate is not None else None
            ),
            "seed": args.seed,
            "layout": list(SLUGS),
            "control_steps": executed,
            "video_frames": recorded,
            "fps": 50 / args.render_stride,
            "auto_reset": args.composite_v62,
            "guided_offset_m": 0.0,
            "assist_force_n": 0.0,
            "camera": "smooth follow: 80% robot root, 20% scene origin",
            "all_success": all(scene["success"] for scene in scenes.values()),
            "scenes": scenes,
        }
        if paths:
            paths[6].write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    finally:
        for writer in writers:
            writer.close()
        for renderer in renderers:
            renderer.close()
        env.close()


if __name__ == "__main__":
    main()
