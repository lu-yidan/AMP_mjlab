"""Conservative early flat cue using only 738D body and motor history."""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from .proprio_symmetry import mirror_host29_actor_obs


class ProprioSceneClassifier(nn.Module):
    def __init__(self, mean: torch.Tensor, scale: torch.Tensor):
        super().__init__()
        if mean.shape != (738,) or scale.shape != (738,):
            raise ValueError("scene classifier requires the 738D actor observation")
        self.register_buffer("mean", mean.float().clone())
        self.register_buffer("scale", scale.float().clamp_min(1e-3).clone())
        self.network = nn.Sequential(
            nn.Linear(738, 256), nn.SiLU(),
            nn.Linear(256, 128), nn.SiLU(), nn.Linear(128, 3),
        )

    def forward(self, actor_obs: torch.Tensor) -> torch.Tensor:
        if actor_obs.shape[-1] != 738:
            raise ValueError("scene classifier received an observation other than 738D")
        normalized = ((actor_obs - self.mean) / self.scale).clamp(-10, 10)
        return self.network(normalized)


def load_scene_classifier(path: str | Path, device: str):
    artifact = torch.load(path, map_location=device, weights_only=False)
    if artifact.get("observation_contract") != "proprio_torque_actor_privileged_critic_v2":
        raise ValueError("scene classifier has an incompatible observation contract")
    if artifact.get("label_order") != ("flat", "guided", "free"):
        raise ValueError("scene classifier label order is incompatible")
    model = ProprioSceneClassifier(artifact["mean"], artifact["scale"]).to(device)
    model.load_state_dict(artifact["model"], strict=True)
    if artifact.get("canonicalize_side", False) and artifact.get("posture") != "side":
        raise ValueError("side-canonical scene model must use side initial postures")
    model.canonicalize_side = bool(artifact.get("canonicalize_side", False))
    model.eval().requires_grad_(False)
    return model, artifact


class ProprioSceneGate:
    """Latch a high-confidence scene cue from actor history and initial IMU."""

    def __init__(self, classifier: ProprioSceneClassifier, num_envs: int,
                 device: str, threshold: float = 0.99, consecutive: int = 10,
                 min_steps: int = 40, max_steps: int = 120,
                 target_class: int = 0, posture: str = "supine",
                 ema_alpha: float = 1.0,
                 early_classifier: ProprioSceneClassifier | None = None,
                 early_max_step: int = 0,
                 right_classifier: ProprioSceneClassifier | None = None,
                 right_min_steps: int = 0,
                 right_max_steps: int = 0):
        if not (0 < threshold < 1 and consecutive > 0
                and 30 < min_steps <= max_steps
                and target_class in (0, 1, 2)
                and posture in ("supine", "prone", "side")
                and 0 < ema_alpha <= 1):
            raise ValueError("invalid proprio scene gate configuration")
        if (early_classifier is None) != (early_max_step == 0):
            raise ValueError("early classifier and maximum step must be supplied together")
        if right_classifier is not None and not (
            posture == "side" and 30 < right_min_steps <= right_max_steps
        ):
            raise ValueError("right-side classifier requires a valid side decision window")
        self.classifier = classifier
        self.num_envs = num_envs
        self.device = device
        self.threshold = threshold
        self.consecutive = consecutive
        self.min_steps = min_steps
        self.max_steps = max_steps
        self.target_class = target_class
        self.posture = posture
        self.ema_alpha = ema_alpha
        self.early_classifier = early_classifier
        self.early_max_step = early_max_step
        self.right_classifier = right_classifier
        self.right_min_steps = right_min_steps
        self.right_max_steps = right_max_steps
        self.initial_posture = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self.initial_right_side = torch.zeros_like(self.initial_posture)
        self.triggered = torch.zeros_like(self.initial_posture)
        self.trigger_age = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.streak = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.smoothed_probability = torch.zeros(num_envs, device=device)
        self.probability_seen = torch.zeros_like(self.initial_posture)

    def reset(self, ids=None):
        if ids is None:
            ids = slice(None)
        self.initial_posture[ids] = False
        self.initial_right_side[ids] = False
        self.triggered[ids] = False
        self.trigger_age[ids] = 0
        self.streak[ids] = 0
        self.smoothed_probability[ids] = 0
        self.probability_seen[ids] = False

    @torch.no_grad()
    def arm_initial_gravity(self, projected_gravity_b: torch.Tensor, ids=None):
        if projected_gravity_b.shape != (self.num_envs, 3):
            raise ValueError("initial IMU gravity must have three axes per environment")
        if ids is None:
            ids = slice(None)
        gravity = projected_gravity_b[ids]
        if self.posture == "side":
            self.initial_posture[ids] = (
                (gravity[:, 1].abs() > 0.6) & (gravity[:, 0].abs() < 0.8)
            )
            self.initial_right_side[ids] = (
                self.initial_posture[ids] & (gravity[:, 1] < 0)
            )
        else:
            signed_x = gravity[:, 0] if self.posture == "prone" else -gravity[:, 0]
            self.initial_posture[ids] = (
                (signed_x > 0.7) & (gravity[:, 1].abs() < 0.65)
            )

    @torch.no_grad()
    def observe(self, actor_obs: torch.Tensor, real_steps: torch.Tensor,
                eligible: torch.Tensor | None = None) -> torch.Tensor:
        if actor_obs.shape != (self.num_envs, 738) or real_steps.shape != (self.num_envs,):
            raise ValueError("early flat gate observations and clocks must align")
        if eligible is None:
            eligible = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        window = (real_steps >= self.min_steps) & (real_steps <= self.max_steps)
        if self.right_classifier is not None:
            right_window = (
                (real_steps >= self.right_min_steps)
                & (real_steps <= self.right_max_steps)
            )
            window = torch.where(self.initial_right_side, right_window, window)
        classifier_obs = actor_obs
        if self.posture == "side" and self.initial_right_side.any() and (
            getattr(self.classifier, "canonicalize_side", False)
            or (self.early_classifier is not None
                and getattr(self.early_classifier, "canonicalize_side", False))
            or (self.right_classifier is not None
                and getattr(self.right_classifier, "canonicalize_side", False))
        ):
            classifier_obs = actor_obs.clone()
            classifier_obs[self.initial_right_side] = mirror_host29_actor_obs(
                actor_obs[self.initial_right_side]
            )
        scene_probability = self.classifier(
            classifier_obs if getattr(self.classifier, "canonicalize_side", False)
            else actor_obs
        ).softmax(-1)[:, self.target_class]
        if self.early_classifier is not None:
            early_probability = self.early_classifier(
                classifier_obs if getattr(self.early_classifier, "canonicalize_side", False)
                else actor_obs
            ).softmax(-1)[:, self.target_class]
            scene_probability = torch.where(
                real_steps <= self.early_max_step,
                torch.maximum(scene_probability, early_probability),
                scene_probability,
            )
        if self.right_classifier is not None:
            right_probability = self.right_classifier(
                classifier_obs if getattr(self.right_classifier, "canonicalize_side", False)
                else actor_obs
            ).softmax(-1)[:, self.target_class]
            scene_probability = torch.where(
                self.initial_right_side, right_probability, scene_probability,
            )
        self.smoothed_probability = torch.where(
            self.probability_seen,
            self.ema_alpha * scene_probability
            + (1.0 - self.ema_alpha) * self.smoothed_probability,
            scene_probability,
        )
        # Start accumulating at motor activation, even if the gate is not
        # allowed to latch until a later control step.
        self.probability_seen |= (
            eligible & self.initial_posture & (real_steps >= 31)
        )
        qualifying = (
            eligible & self.initial_posture & window
            & (self.smoothed_probability >= self.threshold)
        )
        self.streak = torch.where(qualifying, self.streak + 1, 0)
        new_trigger = (
            eligible & self.initial_posture & window & ~self.triggered
            & (self.streak >= self.consecutive)
        )
        self.triggered |= new_trigger
        self.trigger_age = torch.where(
            self.triggered, self.trigger_age + 1,
            torch.zeros_like(self.trigger_age),
        )
        return new_trigger


class ProprioEarlyFlatGate(ProprioSceneGate):
    """High-confidence flat cue for initial supine posture."""

    def __init__(self, classifier: ProprioSceneClassifier, num_envs: int,
                 device: str, threshold: float = 0.99, consecutive: int = 10,
                 min_steps: int = 40, max_steps: int = 120):
        super().__init__(
            classifier, num_envs, device, threshold, consecutive,
            min_steps, max_steps, target_class=0, posture="supine",
        )


class ProprioGuidedProneGate(ProprioSceneGate):
    """High-confidence guided-board cue for initial prone posture."""

    def __init__(self, classifier: ProprioSceneClassifier, num_envs: int,
                 device: str, threshold: float = 0.99, consecutive: int = 10,
                 min_steps: int = 40, max_steps: int = 120):
        super().__init__(
            classifier, num_envs, device, threshold, consecutive,
            min_steps, max_steps, target_class=1, posture="prone",
        )


class ProprioGuidedSideGate(ProprioSceneGate):
    """Route a side specialist only after a proprio-only fixed-board cue."""

    def __init__(self, classifier: ProprioSceneClassifier, num_envs: int,
                 device: str, threshold: float = 0.95, consecutive: int = 10,
                 min_steps: int = 35, max_steps: int = 120,
                 ema_alpha: float = 1.0,
                 early_classifier: ProprioSceneClassifier | None = None,
                 early_max_step: int = 0,
                 right_classifier: ProprioSceneClassifier | None = None,
                 right_min_steps: int = 0,
                 right_max_steps: int = 0):
        super().__init__(
            classifier, num_envs, device, threshold, consecutive,
            min_steps, max_steps, target_class=1, posture="side",
            ema_alpha=ema_alpha,
            early_classifier=early_classifier,
            early_max_step=early_max_step,
            right_classifier=right_classifier,
            right_min_steps=right_min_steps,
            right_max_steps=right_max_steps,
        )
