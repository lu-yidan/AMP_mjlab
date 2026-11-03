"""Prone recovery controller using only HoST actor observations and a clock."""

from __future__ import annotations

import hashlib
import math
from pathlib import Path

import torch

from .model import HOST29_JOINT_NAMES
from .ppo import HostPolicy
from .proprio_phase import ProprioPhaseDetector, load_phase_detector


class InitialProneRouter:
    """Remember an initial prone posture from the body-frame IMU only."""

    def __init__(self, num_envs: int, device: str, gravity_x: float = 0.7):
        if num_envs < 1 or not 0 < gravity_x < 1:
            raise ValueError("invalid initial prone router configuration")
        self.num_envs = num_envs
        self.gravity_x = gravity_x
        self.initial_prone = torch.zeros(num_envs, dtype=torch.bool, device=device)

    def reset(self, ids=None):
        if ids is None:
            ids = slice(None)
        self.initial_prone[ids] = False

    @torch.no_grad()
    def arm_initial_gravity(self, projected_gravity_b: torch.Tensor, ids=None):
        if projected_gravity_b.shape != (self.num_envs, 3):
            raise ValueError("initial body gravity must have three axes per environment")
        if ids is None:
            ids = slice(None)
        gravity = projected_gravity_b[ids]
        self.initial_prone[ids] = (
            (gravity[:, 0] > self.gravity_x) & (gravity[:, 1].abs() < 0.65)
        )


class InitialSideRouter:
    """Latch side lying using the robot's reset IMU gravity, without board data."""

    def __init__(self, num_envs: int, device: str, gravity_y: float = 0.6):
        if num_envs < 1 or not 0 < gravity_y < 1:
            raise ValueError("invalid initial side router configuration")
        self.num_envs = num_envs
        self.gravity_y = gravity_y
        self.initial_side = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self.initial_left_side = torch.zeros_like(self.initial_side)

    def reset(self, ids=None):
        if ids is None:
            ids = slice(None)
        self.initial_side[ids] = False
        self.initial_left_side[ids] = False

    @torch.no_grad()
    def arm_initial_gravity(self, projected_gravity_b: torch.Tensor, ids=None):
        if projected_gravity_b.shape != (self.num_envs, 3):
            raise ValueError("initial body gravity must have three axes per environment")
        if ids is None:
            ids = slice(None)
        gravity = projected_gravity_b[ids]
        self.initial_side[ids] = (
            (gravity[:, 1].abs() > self.gravity_y) & (gravity[:, 0].abs() < 0.8)
        )
        self.initial_left_side[ids] = self.initial_side[ids] & (gravity[:, 1] > 0)


def load_prone_specialist(path: str | Path, parent: dict, device: str):
    """Load a compatible 738D specialist without exposing scene truth."""
    state = torch.load(path, map_location=device, weights_only=False)
    required = (
        "task", "arm", "model_source", "physics_revision", "a6_reset_variant",
        "observation_contract", "action_contract", "actor_obs_dim", "critic_obs_dim",
        "proprio_recovery_controller_sha256",
    )
    for name in required:
        if state.get(name) != parent.get(name):
            raise ValueError(f"prone specialist differs in {name}")
    if state.get("task") != "a6" or state.get("arm") != "G+":
        raise ValueError("prone specialist must be a G+ A6 policy")
    if state.get("actor_obs_dim") != 738 or state.get("critic_obs_dim") != 786:
        raise ValueError("prone specialist must use 738D/786D observations")
    policy = HostPolicy(
        738, 29, critic_obs_dim=786,
        log_std_bounds=(-2.3, math.log(state.get("action_std_max", 0.7))),
        upper_residual=True, waist_residual_scale=5.0,
    ).to(device)
    policy.load_state_dict(state["policy"], strict=True)
    policy.eval().requires_grad_(False)
    return policy, state


class ProprioSupineFlatGate:
    """Latch a handoff from initial body IMU, actor history, and motor clock."""

    def __init__(self, detector: ProprioPhaseDetector, num_envs: int, device: str,
                 threshold: float = 0.8, consecutive: int = 3,
                 min_steps: int = 120, supine_gravity_x: float = -0.7):
        if not (0 < threshold < 1 and consecutive > 0 and min_steps > 30
                and -1 < supine_gravity_x < 0):
            raise ValueError("invalid supine flat gate thresholds")
        self.detector = detector
        self.num_envs = num_envs
        self.device = device
        self.threshold = threshold
        self.consecutive = consecutive
        self.min_steps = min_steps
        self.supine_gravity_x = supine_gravity_x
        self.armed = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self.initial_supine = torch.zeros_like(self.armed)
        self.triggered = torch.zeros_like(self.armed)
        self.streak = torch.zeros(num_envs, dtype=torch.long, device=device)

    def reset(self, ids=None):
        if ids is None:
            ids = slice(None)
        self.armed[ids] = False
        self.initial_supine[ids] = False
        self.triggered[ids] = False
        self.streak[ids] = 0

    @torch.no_grad()
    def arm_initial_gravity(self, projected_gravity_b: torch.Tensor, ids=None):
        """Capture a body-frame IMU reading immediately after each scene reset."""
        if projected_gravity_b.shape != (self.num_envs, 3):
            raise ValueError("initial body gravity must have three axes per environment")
        if ids is None:
            ids = slice(None)
        gravity = projected_gravity_b[ids]
        self.initial_supine[ids] = (
            (gravity[:, 0] < self.supine_gravity_x)
            & (gravity[:, 1].abs() < 0.65)
        )
        self.armed[ids] = True

    @torch.no_grad()
    def observe(self, actor_obs: torch.Tensor, real_steps: torch.Tensor,
                eligible: torch.Tensor | None = None) -> torch.Tensor:
        if actor_obs.shape != (self.num_envs, 738) or real_steps.shape != (self.num_envs,):
            raise ValueError("flat gate observations and episode clocks must align")
        if eligible is None:
            eligible = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        score = self.detector(actor_obs).sigmoid()
        qualifying = eligible & self.armed & self.initial_supine & (score >= self.threshold)
        self.streak = torch.where(qualifying, self.streak + 1, 0)
        new_trigger = (
            eligible & self.armed & ~self.triggered & self.initial_supine
            & (real_steps >= self.min_steps) & (self.streak >= self.consecutive)
        )
        self.triggered |= new_trigger
        return new_trigger

    def neutral_wait_mask(self, real_steps: torch.Tensor, through_step: int):
        """Hold only initial-supine joints before a proprio handoff decision."""
        if through_step <= 30 or real_steps.shape != (self.num_envs,):
            raise ValueError("supine neutral wait needs a valid motor clock")
        return (
            self.armed & self.initial_supine & ~self.triggered
            & (real_steps > 30) & (real_steps <= through_step)
        )

    def probe_action(self, base_action: torch.Tensor, real_steps: torch.Tensor):
        """Small zero-net shoulder-roll probe for initial supine worlds only."""
        if base_action.shape != (self.num_envs, 29) or real_steps.shape != (self.num_envs,):
            raise ValueError("supine probe requires 29 actions and a motor clock")
        probing = (
            self.armed & self.initial_supine & ~self.triggered
            & (real_steps >= 31) & (real_steps <= 60)
        )
        if not probing.any():
            return base_action, probing
        phase = ((real_steps - 31) // 5).remainder(2)
        amplitude = torch.where(
            phase == 0, base_action.new_full((self.num_envs,), 0.12),
            base_action.new_full((self.num_envs,), -0.12),
        )
        probe = torch.zeros_like(base_action)
        probe[:, HOST29_JOINT_NAMES.index("left_shoulder_roll_joint")] = amplitude
        probe[:, HOST29_JOINT_NAMES.index("right_shoulder_roll_joint")] = -amplitude
        return torch.where(probing[:, None], probe, base_action), probing


def load_supine_flat_gate(path: str | Path, num_envs: int, device: str, **kwargs):
    detector, artifact = load_phase_detector(path, device)
    if artifact.get("label_key") != "supine_flat_recovery_safe":
        raise ValueError("flat gate requires a supine flat-recovery detector")
    return ProprioSupineFlatGate(detector, num_envs, device, **kwargs), artifact


class ProprioRecoveryController:
    """One episode-local phase latch and a fixed, thirty-step recovery action.

    The initial posture comes from the first live IMU gravity frame. The phase
    detector sees the same 738D history as the actor. No board state, scene
    label, collision sensor, simulator escape flag, or world position is read.
    """

    def __init__(self, artifact: dict, num_envs: int, device: str):
        if artifact.get("observation_contract") != "proprio_torque_actor_privileged_critic_v2":
            raise ValueError("controller requires the 738D proprio actor contract")
        if artifact.get("action_contract") != "scene_independent_v1":
            raise ValueError("controller requires scene-independent actions")
        self.detector = ProprioPhaseDetector(
            artifact["detector_mean"], artifact["detector_scale"]
        ).to(device)
        self.detector.load_state_dict(artifact["detector_model"], strict=True)
        self.detector.eval()
        self.pulse_action = artifact["pulse_action"].to(device).float()
        if self.pulse_action.shape != (29,) or not torch.isfinite(self.pulse_action).all():
            raise ValueError("controller pulse must contain 29 finite joint actions")
        self.threshold = float(artifact["threshold"])
        self.consecutive = int(artifact["consecutive"])
        self.pulse_steps = int(artifact["pulse_steps"])
        self.prone_gravity_x = float(artifact["prone_gravity_x"])
        self.min_steps = int(artifact["min_steps"])
        if not (0 < self.threshold < 1 and self.consecutive > 0
                and 0 < self.pulse_steps <= 50 and self.min_steps > 30
                and 0 < self.prone_gravity_x < 1):
            raise ValueError("invalid proprio recovery controller thresholds")
        self.num_envs = num_envs
        self.device = device
        self.first_live_seen = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self.initial_prone = torch.zeros_like(self.first_live_seen)
        self.triggered = torch.zeros_like(self.first_live_seen)
        self.streak = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.pulse_remaining = torch.zeros_like(self.streak)
        self.pulse_blend_steps = 0

    def reset(self, ids=None):
        if ids is None:
            ids = slice(None)
        self.first_live_seen[ids] = False
        self.initial_prone[ids] = False
        self.triggered[ids] = False
        self.streak[ids] = 0
        self.pulse_remaining[ids] = 0

    @torch.no_grad()
    def schedule_pulse(
        self, eligible: torch.Tensor, real_steps: torch.Tensor, pulse_step: int,
    ) -> torch.Tensor:
        """Arm the existing pulse at one control step from an external body cue."""
        if (eligible.shape != (self.num_envs,) or eligible.dtype != torch.bool
                or real_steps.shape != (self.num_envs,) or pulse_step <= 30):
            raise ValueError("early pulse needs Boolean worlds and a valid control step")
        selected = eligible & (real_steps + 1 == pulse_step) & ~self.triggered
        self.initial_prone[selected] = True
        self.triggered[selected] = True
        self.pulse_remaining[selected] = self.pulse_steps
        return selected

    @torch.no_grad()
    def observe(self, actor_obs: torch.Tensor, real_steps: torch.Tensor,
                eligible: torch.Tensor | None = None) -> torch.Tensor:
        if actor_obs.shape != (self.num_envs, 738) or real_steps.shape != (self.num_envs,):
            raise ValueError("controller observations and episode clocks must align")
        if eligible is None:
            eligible = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        first_live = eligible & ~self.first_live_seen & (real_steps > 30)
        self.initial_prone[first_live] = (
            actor_obs[first_live, 5 * 94 + 3] > self.prone_gravity_x
        )
        self.first_live_seen |= first_live
        score = self.detector(actor_obs).sigmoid()
        qualifying = eligible & self.initial_prone & (score >= self.threshold)
        self.streak = torch.where(qualifying, self.streak + 1, 0)
        new_trigger = (
            eligible & ~self.triggered & self.initial_prone
            & (real_steps >= self.min_steps)
            & (self.streak >= self.consecutive)
        )
        self.triggered |= new_trigger
        self.pulse_remaining[new_trigger] = self.pulse_steps
        return new_trigger

    @torch.no_grad()
    def apply(self, base_action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if base_action.shape != (self.num_envs, 29):
            raise ValueError("base action must contain 29 joints for each environment")
        forced = self.pulse_remaining > 0
        if self.pulse_blend_steps:
            steps = self.pulse_blend_steps
            if not 0 < steps <= self.pulse_steps // 2:
                raise ValueError("pulse blend must fit within both ends of the pulse")
            elapsed = self.pulse_steps - self.pulse_remaining
            weight = torch.minimum(
                (elapsed + 1).float() / steps,
                self.pulse_remaining.float() / steps,
            ).clamp(0, 1)
            action = torch.where(
                forced[:, None],
                torch.lerp(base_action, self.pulse_action, weight[:, None]),
                base_action,
            )
        else:
            action = torch.where(forced[:, None], self.pulse_action, base_action)
        self.pulse_remaining[forced] -= 1
        return action, forced


def load_recovery_controller(path: str | Path, num_envs: int, device: str):
    artifact = torch.load(path, map_location=device, weights_only=False)
    return ProprioRecoveryController(artifact, num_envs, device), artifact


def validate_controller_provenance(checkpoint: dict, path: str | Path | None):
    """Require the exact frozen controller for checkpoints trained with it."""
    expected = checkpoint.get("proprio_recovery_controller_sha256")
    if expected is None:
        return
    if path is None:
        raise ValueError("checkpoint requires its frozen proprio recovery controller")
    with Path(path).open("rb") as stream:
        actual = hashlib.file_digest(stream, "sha256").hexdigest()
    if actual != expected:
        raise ValueError("recovery controller SHA256 differs from training checkpoint")
