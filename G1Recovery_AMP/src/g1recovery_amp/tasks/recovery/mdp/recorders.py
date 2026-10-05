"""Capture AMP observations that mjlab auto-reset would otherwise replace."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import torch
from mjlab.managers.recorder_manager import RecorderTerm, RecorderTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg

from g1recovery_amp.tasks.recovery.mdp.observations import (
    AMP_DISCRIMINATOR_FRAME_DIM,
    AMP_DISCRIMINATOR_HISTORY_LENGTH,
    amp_discriminator_frame_from_state,
)

if TYPE_CHECKING:
    from mjlab.entity import Entity
    from mjlab.envs import ManagerBasedRlEnv


AMP_TERMINAL_DISC_OBSERVATION_KEY = "amp_terminal_disc_observations"
AMP_TERMINAL_ENV_IDS_KEY = "amp_terminal_env_ids"
AMP_TERMINAL_OBSTRUCTED_KEY = "amp_terminal_obstructed"


def advance_amp_observation_history(
    previous_history: torch.Tensor,
    current_frame: torch.Tensor,
) -> torch.Tensor:
    """Drop the oldest frame and append one post-action frame."""

    expected_history_shape = (
        AMP_DISCRIMINATOR_HISTORY_LENGTH,
        AMP_DISCRIMINATOR_FRAME_DIM,
    )
    if (
        previous_history.ndim != 3
        or tuple(previous_history.shape[1:]) != expected_history_shape
    ):
        raise ValueError(
            "previous AMP history must have shape "
            f"(batch, {expected_history_shape[0]}, {expected_history_shape[1]})"
        )
    expected_frame_shape = (
        previous_history.shape[0],
        AMP_DISCRIMINATOR_FRAME_DIM,
    )
    if tuple(current_frame.shape) != expected_frame_shape:
        raise ValueError(
            f"current AMP frame must have shape {expected_frame_shape}, "
            f"got {tuple(current_frame.shape)}"
        )
    return torch.cat((previous_history[:, 1:], current_frame.unsqueeze(1)), dim=1)


class AmpTerminalObservationRecorder(RecorderTerm):
    """Save post-action AMP windows immediately before automatic reset."""

    def __init__(self, cfg: RecorderTermCfg, env: ManagerBasedRlEnv) -> None:
        super().__init__(cfg, env)
        asset_cfg = cfg.params.get("asset_cfg")
        if not isinstance(asset_cfg, SceneEntityCfg):
            raise TypeError("AmpTerminalObservationRecorder requires asset_cfg")
        self._asset_cfg = asset_cfg
        self._robot = cast("Entity", env.scene[asset_cfg.name])

    def record_pre_reset(self, env_ids: torch.Tensor) -> None:
        previous_history = self._env.obs_buf["disc"]
        if not isinstance(previous_history, torch.Tensor):
            raise TypeError("disc observation group must be a concatenated tensor")

        data = self._robot.data
        current_frame = amp_discriminator_frame_from_state(
            data.root_link_pos_w[env_ids],
            data.root_link_quat_w[env_ids],
            data.root_link_ang_vel_w[env_ids],
            data.joint_pos[env_ids],
            data.joint_vel[env_ids],
            data.body_link_pos_w[env_ids][:, self._asset_cfg.body_ids],
        )
        terminal_history = advance_amp_observation_history(
            previous_history[env_ids],
            current_frame,
        )
        self._env.extras[AMP_TERMINAL_ENV_IDS_KEY] = env_ids.detach().clone()
        self._env.extras[AMP_TERMINAL_DISC_OBSERVATION_KEY] = (
            terminal_history.detach().clone()
        )
        # The observations returned by an auto-reset belong to the next episode.
        # Preserve whether the terminal transition was still under a plate so
        # AmpPPO can apply the reward scale to the correct transition.
        if hasattr(self._env, "_a6_reset_scene"):
            from g1recovery_amp.tasks.recovery.mdp.a6_plate_state import (
                update_a6_plate_state,
            )

            update_a6_plate_state(self._env)
            obstructed = (self._env._a6_reset_scene > 0) & (
                ~self._env._a6_plate_escaped
            )
            self._env.extras[AMP_TERMINAL_OBSTRUCTED_KEY] = (
                obstructed[env_ids].detach().clone()
            )

    def record_post_step(self) -> None:
        if torch.any(self._env.reset_buf):
            return
        self._env.extras.pop(AMP_TERMINAL_ENV_IDS_KEY, None)
        self._env.extras.pop(AMP_TERMINAL_DISC_OBSERVATION_KEY, None)
        self._env.extras.pop(AMP_TERMINAL_OBSTRUCTED_KEY, None)
