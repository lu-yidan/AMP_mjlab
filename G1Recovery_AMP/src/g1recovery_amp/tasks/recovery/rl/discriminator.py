"""AMP discriminator translated to the RSL-RL version used by mjlab."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from rsl_rl.modules import EmpiricalNormalization
from torch import nn
from torch.autograd import grad

from g1recovery_amp.tasks.recovery.mdp.observations import (
    AMP_DISCRIMINATOR_FRAME_DIM,
    AMP_DISCRIMINATOR_HISTORY_LENGTH,
)


@dataclass(frozen=True)
class AmpDiscriminatorCfg:
    """Source hyperparameters for the G1 get-up AMP discriminator."""

    frame_dim: int = AMP_DISCRIMINATOR_FRAME_DIM
    history_length: int = AMP_DISCRIMINATOR_HISTORY_LENGTH
    hidden_dims: tuple[int, ...] = (1024, 512)
    negative_slope: float = 0.01
    style_reward_scale: float = 50.0
    task_style_lerp: float = 0.5
    normalization_clip: float = 10.0
    normalization_eps: float = 1.0e-4
    normalization_until: int = 100_000_000


class AmpDiscriminator(nn.Module):
    """Score whether a ten-frame robot motion window resembles demonstration."""

    def __init__(self, cfg: AmpDiscriminatorCfg | None = None) -> None:
        super().__init__()
        if cfg is None:
            cfg = AmpDiscriminatorCfg()
        if not cfg.hidden_dims:
            raise ValueError("AMP discriminator needs at least one hidden layer")
        if cfg.style_reward_scale < 0.0:
            raise ValueError("style_reward_scale must be non-negative")
        if not 0.0 <= cfg.task_style_lerp <= 1.0:
            raise ValueError("task_style_lerp must be between zero and one")
        self.cfg = cfg
        self.input_dim = cfg.frame_dim * cfg.history_length
        self.disc_obs_normalizer = EmpiricalNormalization(
            shape=cfg.frame_dim,
            eps=cfg.normalization_eps,
            until=cfg.normalization_until,
        )

        layers: list[nn.Module] = []
        in_features = self.input_dim
        for out_features in cfg.hidden_dims:
            layers.extend(
                (
                    nn.Linear(in_features, out_features),
                    nn.LeakyReLU(negative_slope=cfg.negative_slope),
                )
            )
            in_features = out_features
        self.disc_trunk = nn.Sequential(*layers)
        self.disc_linear = nn.Linear(in_features, 1)

    def _validate_observation(self, observation: torch.Tensor) -> None:
        expected = (self.cfg.history_length, self.cfg.frame_dim)
        if observation.ndim != 3 or tuple(observation.shape[1:]) != expected:
            raise ValueError(
                "AMP observation must have shape "
                f"(batch, {expected[0]}, {expected[1]}), got "
                f"{tuple(observation.shape)}"
            )

    def normalize_observation(self, observation: torch.Tensor) -> torch.Tensor:
        """Normalize each 85-value frame and clamp extreme values."""

        self._validate_observation(observation)
        flattened_frames = observation.reshape(-1, self.cfg.frame_dim)
        normalized = self.disc_obs_normalizer(flattened_frames)
        normalized = normalized.clamp(
            -self.cfg.normalization_clip, self.cfg.normalization_clip
        )
        return normalized.reshape_as(observation)

    def update_normalization(self, observation: torch.Tensor) -> None:
        """Update running statistics using unnormalized live/demo frames."""

        self._validate_observation(observation)
        self.disc_obs_normalizer.update(observation.reshape(-1, self.cfg.frame_dim))

    def forward(self, flattened_observation: torch.Tensor) -> torch.Tensor:
        """Map a flattened 850-value window to one unconstrained score."""

        if (
            flattened_observation.ndim != 2
            or flattened_observation.shape[1] != self.input_dim
        ):
            raise ValueError(
                f"flattened AMP input must have shape (batch, {self.input_dim})"
            )
        return self.disc_linear(self.disc_trunk(flattened_observation))

    def score(self, observation: torch.Tensor) -> torch.Tensor:
        """Normalize a three-dimensional window and return one score per sample."""

        normalized = self.normalize_observation(observation)
        return self(normalized.reshape(normalized.shape[0], -1)).squeeze(-1)

    def lsgan_loss(
        self,
        policy_observation: torch.Tensor,
        demonstration_observation: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Train policy samples toward -1 and demonstration samples toward +1."""

        policy_score = self.score(policy_observation)
        demonstration_score = self.score(demonstration_observation)
        policy_loss = torch.mean((policy_score + 1.0).square())
        demonstration_loss = torch.mean((demonstration_score - 1.0).square())
        return (
            0.5 * (policy_loss + demonstration_loss),
            policy_score,
            demonstration_score,
        )

    def gradient_penalty(
        self,
        demonstration_observation: torch.Tensor,
        scale: float = 10.0,
    ) -> torch.Tensor:
        """Penalize sharp discriminator changes around demonstration samples."""

        normalized = self.normalize_observation(demonstration_observation)
        flattened = normalized.reshape(normalized.shape[0], -1).detach()
        flattened.requires_grad_(True)
        score = self(flattened)
        input_gradient = grad(
            outputs=score,
            inputs=flattened,
            grad_outputs=torch.ones_like(score),
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        return scale * input_gradient.norm(2, dim=1).square().mean()

    @torch.no_grad()
    def style_reward(
        self,
        policy_observation: torch.Tensor,
        step_dt: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Convert the source LSGAN score into the source AMP style reward."""

        was_training = self.training
        self.eval()
        score = self.score(policy_observation)
        unscaled_reward = (1.0 - 0.25 * (score - 1.0).square()).clamp(min=0.0)
        reward = step_dt * self.cfg.style_reward_scale * unscaled_reward
        if was_training:
            self.train()
        return reward, score

    def blend_reward(
        self,
        task_reward: torch.Tensor,
        style_reward: torch.Tensor,
        task_weight: torch.Tensor | float | None = None,
    ) -> torch.Tensor:
        """Mix hand-written task reward with learned motion-style reward."""

        weight = self.cfg.task_style_lerp if task_weight is None else task_weight
        weight = torch.as_tensor(
            weight,
            dtype=task_reward.dtype,
            device=task_reward.device,
        )
        if weight.ndim > 0 and weight.shape != task_reward.shape:
            raise ValueError(
                "task_weight must be scalar or match task_reward shape, got "
                f"{tuple(weight.shape)} and {tuple(task_reward.shape)}"
            )
        if torch.any((weight < 0.0) | (weight > 1.0)):
            raise ValueError("task_weight must be between zero and one")
        return weight * task_reward + (1.0 - weight) * style_reward
