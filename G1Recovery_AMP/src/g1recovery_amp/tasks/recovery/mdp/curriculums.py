"""Curriculum terms for gradually removing get-up assistance."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import torch

from g1recovery_amp.tasks.recovery.mdp.commands import GetUpAssistForceCommand

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


GET_UP_ASSIST_FORCE_STEP = 10.0
GET_UP_ASSIST_SUCCESS_FRACTION = 0.6


def get_up_assist_force_level(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor | slice,
    reward_term_name: str,
) -> torch.Tensor:
    """Reduce assistance for reset environments after sufficiently good episodes."""

    force_command = cast(
        GetUpAssistForceCommand,
        env.command_manager.get_term("get_up_assist_force"),
    )
    episode_sums = env.reward_manager._episode_sums[reward_term_name]
    reward_cfg = env.reward_manager.get_term_cfg(reward_term_name)
    mean_reward_rate = torch.mean(episode_sums[env_ids]) / env.max_episode_length_s
    success_threshold = GET_UP_ASSIST_SUCCESS_FRACTION * reward_cfg.weight
    if mean_reward_rate > success_threshold:
        force_command._command[env_ids, 0] = (
            force_command._command[env_ids, 0] - GET_UP_ASSIST_FORCE_STEP
        ).clamp(min=0.0)
    return torch.mean(force_command.command)
