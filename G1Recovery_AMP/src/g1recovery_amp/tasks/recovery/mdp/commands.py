"""Reference-motion command used to initialize the G1 recovery task."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from mjlab.managers.command_manager import CommandTerm, CommandTermCfg
from mjlab.tasks.tracking.mdp.commands import (
    MotionCommand,
    MotionCommandCfg,
    MotionLoader,
)

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


RECOVERY_ROOT_HEIGHT_OFFSET = 0.1
AMP_REFERENCE_STEPS = 10
INITIAL_GET_UP_ASSIST_FORCE = 200.0


def _resolve_motion_manifest(
    motion_file: str,
    motion_files: tuple[str, ...],
    motion_weights: tuple[float, ...],
    sampling_mode: str,
) -> tuple[tuple[str, ...], tuple[float, ...]]:
    """Resolve the files loaded by training or deterministic playback.

    Mjlab's play CLI overrides only ``motion_file``.  Playback always uses
    motion zero, so load only that override instead of retaining a stale
    multi-motion training manifest.  Training keeps strict alignment between
    its file and weight sequences.
    """

    if sampling_mode == "start":
        return (motion_file,), (1.0,)

    resolved_files = motion_files or (motion_file,)
    if resolved_files[0] != motion_file:
        raise ValueError("motion_file must be the first entry in motion_files")
    if motion_weights and len(motion_weights) != len(resolved_files):
        raise ValueError("motion_weights must match motion_files length")
    resolved_weights = motion_weights or (1.0,) * len(resolved_files)
    return resolved_files, resolved_weights


class GetUpAssistForceCommand(CommandTerm):
    """Store the current upward get-up assistance for every environment."""

    def __init__(self, cfg: GetUpAssistForceCommandCfg, env: ManagerBasedRlEnv):
        super().__init__(cfg, env)
        self._command = torch.full((self.num_envs, 1), cfg.force, device=self.device)

    @property
    def command(self) -> torch.Tensor:
        return self._command

    def _update_metrics(self) -> None:
        pass

    def _resample_command(self, env_ids: torch.Tensor) -> None:
        # Curriculum changes must survive ordinary command resampling/reset.
        del env_ids

    def _update_command(self, env_ids: torch.Tensor | None) -> None:
        # This command is a stored difficulty level, not a changing target.
        del env_ids


@dataclass(kw_only=True)
class GetUpAssistForceCommandCfg(CommandTermCfg):
    """Configuration for the source task's upward torso assistance."""

    resampling_time_range: tuple[float, float] = (100.0, 100.0)
    force: float = INITIAL_GET_UP_ASSIST_FORCE

    def build(self, env: ManagerBasedRlEnv) -> GetUpAssistForceCommand:
        return GetUpAssistForceCommand(self, env)


class RecoveryMotionCommand(MotionCommand):
    """Motion command with source-compatible recovery reset behavior."""

    def __init__(self, cfg: RecoveryMotionCommandCfg, env: ManagerBasedRlEnv):
        super().__init__(cfg, env)
        self.recovery_cfg = cfg
        motion_files, weights = _resolve_motion_manifest(
            cfg.motion_file,
            cfg.motion_files,
            cfg.motion_weights,
            cfg.sampling_mode,
        )
        if any(weight < 0.0 for weight in weights) or sum(weights) <= 0.0:
            raise ValueError("motion_weights must be non-negative with a positive sum")
        if len(motion_files) > 1 and cfg.sampling_mode == "adaptive":
            raise NotImplementedError(
                "adaptive sampling is not implemented for multiple recovery motions"
            )

        # The parent already loaded cfg.motion_file.  Load only the remaining
        # verified assets, preserving the first loader for parent GUI support.
        self.motions = (
            self.motion,
            *(
                MotionLoader(path, self.body_indexes, device=self.device)
                for path in motion_files[1:]
            ),
        )
        self.motion_weights = torch.tensor(
            weights,
            dtype=torch.float32,
            device=self.device,
        )
        self.motion_weights /= self.motion_weights.sum()
        self.motion_lengths = torch.tensor(
            [motion.time_step_total for motion in self.motions],
            dtype=torch.long,
            device=self.device,
        )
        if torch.any(self.motion_lengths <= cfg.future_reference_steps):
            raise ValueError(
                "every motion must contain more frames than future_reference_steps"
            )
        self.motion_ids = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self.reset_time_steps = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self.reset_motion_ids = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )

    def gather_motion_field(
        self,
        field_name: str,
        motion_ids: torch.Tensor,
        frame_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Read one field from different-length motions without padded frames."""

        if motion_ids.ndim != 1 or frame_ids.shape[0] != motion_ids.shape[0]:
            raise ValueError("motion_ids and frame_ids need the same batch dimension")
        if torch.any(motion_ids < 0) or torch.any(motion_ids >= len(self.motions)):
            raise IndexError("motion id is outside the loaded recovery motion set")
        selected_lengths = self.motion_lengths[motion_ids]
        length_shape = (selected_lengths.shape[0],) + (1,) * (frame_ids.ndim - 1)
        if torch.any(frame_ids < 0) or torch.any(
            frame_ids >= selected_lengths.reshape(length_shape)
        ):
            raise IndexError("frame id is outside its selected recovery motion")

        first_field = getattr(self.motions[0], field_name)
        output = torch.empty(
            (*frame_ids.shape, *first_field.shape[1:]),
            dtype=first_field.dtype,
            device=self.device,
        )
        for motion_id, motion in enumerate(self.motions):
            selected = motion_ids == motion_id
            if torch.any(selected):
                output[selected] = getattr(motion, field_name)[frame_ids[selected]]
        return output

    @property
    def joint_pos(self) -> torch.Tensor:
        return self.gather_motion_field("joint_pos", self.motion_ids, self.time_steps)

    @property
    def joint_vel(self) -> torch.Tensor:
        return self.gather_motion_field("joint_vel", self.motion_ids, self.time_steps)

    @property
    def body_pos_w(self) -> torch.Tensor:
        positions = self.gather_motion_field(
            "body_pos_w", self.motion_ids, self.time_steps
        )
        return positions + self._env.scene.env_origins[:, None, :]

    @property
    def body_quat_w(self) -> torch.Tensor:
        return self.gather_motion_field(
            "body_quat_w", self.motion_ids, self.time_steps
        )

    @property
    def body_lin_vel_w(self) -> torch.Tensor:
        return self.gather_motion_field(
            "body_lin_vel_w", self.motion_ids, self.time_steps
        )

    @property
    def body_ang_vel_w(self) -> torch.Tensor:
        return self.gather_motion_field(
            "body_ang_vel_w", self.motion_ids, self.time_steps
        )

    @property
    def anchor_pos_w(self) -> torch.Tensor:
        return self.body_pos_w[:, self.motion_anchor_body_index]

    @property
    def anchor_quat_w(self) -> torch.Tensor:
        return self.body_quat_w[:, self.motion_anchor_body_index]

    @property
    def anchor_lin_vel_w(self) -> torch.Tensor:
        return self.body_lin_vel_w[:, self.motion_anchor_body_index]

    @property
    def anchor_ang_vel_w(self) -> torch.Tensor:
        return self.body_ang_vel_w[:, self.motion_anchor_body_index]

    def _uniform_sampling(self, env_ids: torch.Tensor) -> None:
        """Sample one weighted motion and one valid reset frame per environment."""

        sampled_motion_ids = torch.multinomial(
            self.motion_weights,
            len(env_ids),
            replacement=True,
        )
        self.motion_ids[env_ids] = sampled_motion_ids
        upper_bounds = (
            self.motion_lengths[sampled_motion_ids]
            - self.recovery_cfg.future_reference_steps
        )
        random_fraction = torch.rand(len(env_ids), device=self.device)
        self.time_steps[env_ids] = torch.floor(
            random_fraction * upper_bounds
        ).long()
        self.metrics["sampling_entropy"][:] = 1.0
        per_frame_probability = self.motion_weights / (
            self.motion_lengths - self.recovery_cfg.future_reference_steps
        )
        self.metrics["sampling_top1_prob"][:] = per_frame_probability.max()
        self.metrics["sampling_top1_bin"][:] = 0.5

    def sample_amp_demo_indices(
        self, history_length: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Choose weighted motions and chronological AMP demonstration windows."""

        upper_bounds = self.motion_lengths - history_length
        if torch.any(upper_bounds <= 0):
            raise ValueError("every motion must contain more frames than AMP history")
        if self.recovery_cfg.sampling_mode == "start":
            motion_ids = self.motion_ids.clone()
            selected_upper_bounds = upper_bounds[motion_ids]
            starts = torch.minimum(self.time_steps, selected_upper_bounds - 1)
        else:
            motion_ids = torch.multinomial(
                self.motion_weights,
                self.num_envs,
                replacement=True,
            )
            selected_upper_bounds = upper_bounds[motion_ids]
            starts = torch.floor(
                torch.rand(self.num_envs, device=self.device)
                * selected_upper_bounds
            ).long()
        offsets = torch.arange(history_length, device=self.device)
        frame_ids = starts.unsqueeze(1) + offsets.unsqueeze(0)
        return motion_ids, frame_ids

    def sample_amp_demo_frame_ids(self, history_length: int) -> torch.Tensor:
        """Compatibility helper returning only sampled chronological frame ids."""

        return RecoveryMotionCommand.sample_amp_demo_indices(self, history_length)[1]

    def _resample_command(self, env_ids: torch.Tensor) -> None:
        if self.recovery_cfg.sampling_mode == "start":
            self.motion_ids[env_ids] = 0
        super()._resample_command(env_ids)
        self.reset_time_steps[env_ids] = self.time_steps[env_ids]
        self.reset_motion_ids[env_ids] = self.motion_ids[env_ids]

    def _update_command(self, env_ids: torch.Tensor | None = None) -> None:
        """Advance reference frames without resetting the physical robot.

        Recovery motions initialize the robot only when the environment resets.
        Once a reference reaches its final standing frame, keep that frame as the
        reference until the next environment reset.  Calling ``_resample_command``
        here would inherit tracking-task behavior and teleport qpos/qvel midway
        through a recovery episode.
        """

        if env_ids is None:
            self.time_steps += 1
            selected_lengths = self.motion_lengths[self.motion_ids]
            self.time_steps.copy_(torch.minimum(self.time_steps, selected_lengths - 1))
        else:
            self.time_steps[env_ids] += 1
            selected_lengths = self.motion_lengths[self.motion_ids[env_ids]]
            self.time_steps[env_ids] = torch.minimum(
                self.time_steps[env_ids], selected_lengths - 1
            )

        if self._pending_forward:
            self._pending_forward = False
            self._env.sim.forward()
        self.update_relative_body_poses()

    def _write_reference_state_to_sim(
        self,
        env_ids: torch.Tensor,
        root_pos: torch.Tensor,
        root_ori: torch.Tensor,
        root_lin_vel: torch.Tensor,
        root_ang_vel: torch.Tensor,
        joint_pos: torch.Tensor,
        joint_vel: torch.Tensor,
    ) -> None:
        """Lift and write the source reference state without soft-limit clipping."""

        if not self.recovery_cfg.initialize_robot_from_motion:
            return

        root_pos = root_pos.clone()
        root_pos[:, 2] += self.recovery_cfg.root_height_offset
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
        root_state = torch.cat((root_pos, root_ori, root_lin_vel, root_ang_vel), dim=-1)
        self.robot.write_root_state_to_sim(root_state, env_ids=env_ids)
        self.robot.reset(env_ids=env_ids)


@dataclass(kw_only=True)
class RecoveryMotionCommandCfg(MotionCommandCfg):
    """Configuration for source-compatible recovery reference initialization."""

    root_height_offset: float = RECOVERY_ROOT_HEIGHT_OFFSET
    future_reference_steps: int = AMP_REFERENCE_STEPS
    motion_files: tuple[str, ...] = ()
    motion_weights: tuple[float, ...] = ()
    initialize_robot_from_motion: bool = True

    def build(self, env: ManagerBasedRlEnv) -> RecoveryMotionCommand:
        return RecoveryMotionCommand(self, env)
