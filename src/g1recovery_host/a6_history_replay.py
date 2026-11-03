"""Restore recorded proprio history for a reached postescape reset state."""

from __future__ import annotations

import torch


def restore_postescape_actor_history(env, ids: torch.Tensor | None = None):
    """Replace reset backfill with the six live frames saved at capture time.

    This operates only on replay-reset worlds and uses actor observations, not
    plate state or critic labels. It does not alter physical state or rewards.
    The pinned mjlab CircularBuffer stores terms separately and its pointer
    identifies the chronological first slot after a reset.
    """
    bank = getattr(env, "_a6_postescape_bank", {})
    if "actor_obs" not in bank:
        return env.obs_buf
    if ids is None:
        ids = torch.arange(env.num_envs, device=env.device)
    replay_ids = ids[
        env._a6_postescape_start[ids]
        & (env._a6_postescape_bank_index[ids] >= 0)
    ]
    if not len(replay_ids):
        return env.obs_buf
    snapshots = bank["actor_obs"][
        env._a6_postescape_bank_index[replay_ids]
    ]
    if snapshots.shape != (len(replay_ids), 738):
        raise ValueError("postescape bank actor history must be 738D")
    buffers = env.observation_manager._group_obs_term_history_buffer["actor"]
    for name, start, width in (("host_frame", 0, 94), ("motor_torque", 564, 29)):
        history = snapshots[:, start : start + 6 * width].reshape(-1, 6, width)
        buffer = buffers[name]
        if buffer.max_length != 6 or buffer._buffer is None:
            raise ValueError(f"unexpected actor history buffer for {name}")
        slots = (buffer._pointer + 1 + torch.arange(6, device=env.device)) % 6
        buffer._buffer[slots[:, None], replay_ids[None, :]] = history.transpose(0, 1)
        buffer._num_pushes[replay_ids] = 6
    env.obs_buf["actor"][replay_ids] = snapshots
    return env.obs_buf
