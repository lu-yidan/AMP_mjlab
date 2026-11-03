"""Check a proprioceptive A6 checkpoint and its scene-independent action path."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from mjlab.envs import ManagerBasedRlEnv

from g1recovery_host.a6_task import host_a6_env_cfg, motor_torque_frame
from g1recovery_host.task import host_frame


def audit_checkpoint(path: Path) -> tuple[bool, bool]:
    state = torch.load(path, map_location="cpu", weights_only=False)
    torque_v2 = state.get("observation_contract") == "proprio_torque_actor_privileged_critic_v2"
    actor_dim = 738 if torque_v2 else 564
    critic_dim = 786 if torque_v2 else 612
    expected = {
        "task": "a6",
        "model_source": "unitree_g1_29",
        "observation_contract": "proprio_torque_actor_privileged_critic_v2" if torque_v2 else "proprio_actor_privileged_critic_v1",
        "actor_obs_dim": actor_dim,
        "critic_obs_dim": critic_dim,
        "action_contract": "scene_independent_v1",
    }
    for key, value in expected.items():
        if state.get(key) != value:
            raise ValueError(f"checkpoint {key}: expected {value!r}, got {state.get(key)!r}")
    weights = state["policy"]
    if tuple(weights["actor.0.weight"].shape) != (512, actor_dim):
        raise ValueError(f"actor input layer is not {actor_dim}D")
    if tuple(weights["upper_residual.0.weight"].shape) != (64, actor_dim):
        raise ValueError(f"upper-body actor residual is not {actor_dim}D")
    for index in range(4):
        if tuple(weights[f"critics.{index}.0.weight"].shape) != (512, critic_dim):
            raise ValueError(f"critic {index} input layer is not {critic_dim}D")
    print(f"checkpoint contract passed: {path} (iteration {state['iteration']})")
    return torque_v2, "_hard_mix_v1" in state["reward_variant"]


def audit_runtime(hard_mix: bool = False, torque_v2: bool = False) -> None:
    cfg = host_a6_env_cfg(
        num_envs=32,
        edge_curriculum=True,
        critic_plate_observation=True,
        focused_guided=True,
        reset_mix="diverse_v3",
        aligned_edge=True,
        footprint_edge=True,
        footprint_overlap_m=0.03,
        privileged_guided_actions=False,
        hard_mix=hard_mix,
        proprio_motor_torque=torque_v2,
    )
    env = ManagerBasedRlEnv(cfg, device="cpu")
    try:
        if hard_mix:
            env._a6_edge_offset_m = 0.25
        observations, _ = env.reset()
        actor_dim = 738 if torque_v2 else 564
        critic_dim = 786 if torque_v2 else 612
        if observations["actor"].shape != (32, actor_dim):
            raise ValueError(f"actor observation is not {actor_dim}D")
        if observations["critic"].shape != (32, critic_dim):
            raise ValueError(f"critic observation is not {critic_dim}D")
        plate_history = observations["critic"][:, actor_dim:]
        if not (plate_history[env._a6_scene == 1].abs().sum() > 0):
            raise ValueError("critic plate observation is missing in guided worlds")
        if not torch.equal(
            plate_history[env._a6_scene == 0],
            torch.zeros_like(plate_history[env._a6_scene == 0]),
        ):
            raise ValueError("flat critic worlds unexpectedly contain plate features")
        term = env.action_manager.get_term("joint_pos")
        if term.cfg.privileged_guided_actions or term.cfg.guided_lift_guard:
            raise ValueError("action path still uses privileged plate controls")
        if torque_v2 and motor_torque_frame(env).shape != (32, 29):
            raise ValueError("motor torque observation is not 29D")
        if hard_mix:
            guided = env._a6_scene == 1
            tier = env._a6_edge_tier
            offsets = env._a6_initial_edge_offset_m
            counts = [
                int((guided & mask).sum())
                for mask in (tier < 2, tier == 2, tier == 3)
            ]
            if counts != [10, 4, 4]:
                raise ValueError(f"unexpected 32-world guided tiers: {counts}")
            if not (offsets[guided & (tier < 3)] > 0).all():
                raise ValueError("easy or half-offset guided board was not shifted")
            if not torch.equal(offsets[guided & (tier == 3)], torch.zeros_like(offsets[guided & (tier == 3)])):
                raise ValueError("centred guided board was shifted")
            print(f"hard-mix guided reset tiers passed: {counts}")
        actions = torch.full((32, 29), 0.2, device=env.device)
        original_scene = env._a6_scene.clone()
        original_direction = env._a6_direction.clone()
        original_escaped = env._a6_escaped.clone()
        term.real_steps.fill_(31)
        env._a6_scene.fill_(1)
        env._a6_escaped.fill_(False)
        env._a6_direction.fill_(0)
        term.process_actions(actions)
        torch.manual_seed(123)
        guided_frame = host_frame(env)
        guided_action = term.raw_action.clone()
        guided_target = term._processed_actions.clone()
        term.real_steps.fill_(31)
        env._a6_scene.fill_(0)
        env._a6_escaped.fill_(True)
        env._a6_direction.fill_(1)
        term.process_actions(actions)
        torch.manual_seed(123)
        flat_frame = host_frame(env)
        if not torch.equal(guided_frame, flat_frame):
            raise ValueError("actor frame changed when only scene labels changed")
        if not torch.equal(guided_action, term.raw_action) or not torch.equal(
            guided_target, term._processed_actions
        ):
            raise ValueError("action execution changed when only scene labels changed")
        if torque_v2:
            env._a6_scene.copy_(original_scene)
            env._a6_direction.copy_(original_direction)
            env._a6_escaped.copy_(original_escaped)
            term.real_steps.fill_(31)
            next_observations, _, _, _, _ = env.step(actions)
            if not (next_observations["actor"][:, 564:].abs().sum() > 0):
                raise ValueError("actor torque history stayed zero after motor activation")
        print(f"runtime contract passed: actor {actor_dim}D, critic {critic_dim}D, scene-independent actions")
    finally:
        env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--runtime", action="store_true")
    parser.add_argument("--hard-mix", action="store_true")
    parser.add_argument("--torque-v2", action="store_true")
    args = parser.parse_args()
    if args.checkpoint is None and not args.runtime:
        parser.error("provide --checkpoint, --runtime, or both")
    torque_v2, hard_mix = args.torque_v2, args.hard_mix
    if args.checkpoint is not None:
        inferred_torque, inferred_hard_mix = audit_checkpoint(args.checkpoint)
        torque_v2 |= inferred_torque
        hard_mix |= inferred_hard_mix
    if args.runtime:
        audit_runtime(hard_mix=hard_mix, torque_v2=torque_v2)
