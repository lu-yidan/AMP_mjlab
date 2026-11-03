"""Policy-history switch at a physically confirmed prone guided-board exit."""

from __future__ import annotations

import torch


def postescape_live_observation_enabled(checkpoint):
    """Recover the effective training setting, including historical replay runs.

    Early checkpoints stored the CLI flag, although training implicitly
    enabled live proprioception whenever a postescape bank was supplied.
    """
    return bool(
        checkpoint.get("a6_postescape_live_obs_v1")
        or checkpoint.get("a6_proprio_postescape_bank_sha256")
        or "_proprio_postescape_replay_v1" in checkpoint.get("reward_variant", "")
    )


def reset_actor_history_from_proprio(env, ids):
    """Restart only the actor's six-frame memory after a proprio-only cue.

    The decision is made by the caller from actor observations. This function
    never reads board state or changes physical state, reward state, motor
    activation, critic history, or simulator escape latches.
    """
    if len(ids):
        manager = env.observation_manager
        for buffer in manager._group_obs_term_history_buffer["actor"].values():
            buffer.reset(batch_ids=ids)
        actor = manager.compute_group("actor", update_history=True, env_ids=ids)
        env.obs_buf["actor"] = actor
    return env.obs_buf


def reset_observation_history_from_proprio(env, ids):
    """Reset both observation groups, without changing controller or physics."""
    if len(ids):
        manager = env.observation_manager
        manager.reset(env_ids=ids)
        env.obs_buf = manager.compute(update_history=True, env_ids=ids)
    return env.obs_buf


def reset_guided_history_after_escape(env, escaped_before, done=None, *, direction=1):
    """Start a guided recovery phase without moving or resetting the robot.

    Real postescape replay states restart the six-frame actor history and
    real-step counter while keeping motors active. Apply the same controller
    transition when an ordinary guided-prone world first satisfies the 25-step
    clearance latch. Only the selected observation rows are backfilled; the
    physical simulation state, board, and other worlds are untouched.
    """
    if done is None:
        done = torch.zeros_like(escaped_before)
    if direction not in (0, 1):
        raise ValueError("guided phase direction must be supine=0 or prone=1")
    selected = (
        (env._a6_scene == 1)
        & (env._a6_direction == direction)
        & ~escaped_before
        & env._a6_escaped
        & ~done
    )
    ids = selected.nonzero(as_tuple=False).squeeze(-1)
    if len(ids):
        env._a6_postescape_start[ids] = True
        env.action_manager.get_term("joint_pos").real_steps[ids] = 0
        env.observation_manager.reset(env_ids=ids)
        env.obs_buf = env.observation_manager.compute(
            update_history=True, env_ids=ids
        )
    return env.obs_buf, ids


def reset_prone_history_after_escape(env, escaped_before, done=None):
    return reset_guided_history_after_escape(
        env, escaped_before, done, direction=1
    )
