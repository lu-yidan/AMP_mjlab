"""One evaluation protocol for tracking policies, whatever trained them (SuperMimic, PPO, ...).

Every env starts at the task's own reset (a random motion frame) plus one uncounted warm-up step, runs one
episode, and the result is:
survival (fraction of the episode before the first termination, time-outs count as survived),
full_episode (fraction of envs that never terminate) and the motion command's error metrics while alive.
"""

from __future__ import annotations

from collections.abc import Callable

import torch

METRICS = ("error_anchor_pos", "error_body_pos", "error_body_rot", "error_joint_pos")


@torch.no_grad()
def evaluate_tracking(env, act: Callable, get_obs: Callable, command_name: str = "motion", metrics=METRICS, seed: int = 0) -> dict:
    """env: a manager-based env (mjlab or Isaac Lab, unwrapped). act(obs) -> actions, get_obs() -> obs."""
    env = env.unwrapped
    cmd = env.command_manager.get_term(command_name)
    metrics = [k for k in metrics if k in cmd.metrics]
    torch.manual_seed(seed)
    env.reset()
    # Warm-up step, not counted: Isaac Lab computes terminations before it updates the commands, and
    # BeyondMimic's command does not refresh its robot-anchored reference on reset(), so the first step after a
    # full reset terminates envs on stale reference poses (every env in a fresh process). After one step
    # every manager is current; in-step resets refresh before the next termination check.
    env.step(act(get_obs()))
    T = int(env.max_episode_length)
    alive = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
    survived = torch.zeros(env.num_envs, device=env.device)
    err = {k: 0.0 for k in metrics}
    n = 0.0
    for _ in range(T):
        _, _, terminated, truncated, _ = env.step(act(get_obs()))
        alive &= ~terminated
        survived += alive
        for k in metrics:
            err[k] += (cmd.metrics[k] * alive).sum().item()
        n += alive.sum().item()
        alive &= ~truncated
        if not alive.any():
            break
    out = {"survival": (survived / T).mean().item(), "full_episode": (survived >= T - 1).float().mean().item()}
    out.update({k: err[k] / max(n, 1.0) for k in metrics})
    return out
