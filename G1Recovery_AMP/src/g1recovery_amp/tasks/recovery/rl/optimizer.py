"""Standalone optimization step for the migrated AMP discriminator."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from g1recovery_amp.tasks.recovery.rl.discriminator import AmpDiscriminator


@dataclass(frozen=True)
class AmpDiscriminatorOptimizerCfg:
    """Source optimization settings for the G1 get-up discriminator."""

    learning_rate: float = 1.0e-3
    trunk_weight_decay: float = 1.0e-3
    output_weight_decay: float = 1.0e-1
    gradient_penalty_scale: float = 10.0
    max_gradient_norm: float = 1.0


class AmpDiscriminatorOptimizer:
    """Apply one isolated discriminator update without depending on PPO."""

    def __init__(
        self,
        discriminator: AmpDiscriminator,
        cfg: AmpDiscriminatorOptimizerCfg | None = None,
    ) -> None:
        if cfg is None:
            cfg = AmpDiscriminatorOptimizerCfg()
        if cfg.learning_rate <= 0.0:
            raise ValueError("learning_rate must be positive")
        if cfg.trunk_weight_decay < 0.0 or cfg.output_weight_decay < 0.0:
            raise ValueError("weight decay values must be non-negative")
        if cfg.gradient_penalty_scale < 0.0:
            raise ValueError("gradient_penalty_scale must be non-negative")
        if cfg.max_gradient_norm <= 0.0:
            raise ValueError("max_gradient_norm must be positive")

        self.discriminator = discriminator
        self.cfg = cfg
        self.optimizer = torch.optim.Adam(
            (
                {
                    "name": "disc_trunk",
                    "params": discriminator.disc_trunk.parameters(),
                    "weight_decay": cfg.trunk_weight_decay,
                },
                {
                    "name": "disc_linear",
                    "params": discriminator.disc_linear.parameters(),
                    "weight_decay": cfg.output_weight_decay,
                },
            ),
            lr=cfg.learning_rate,
        )

    def update(
        self,
        policy_observations: torch.Tensor,
        demonstration_observations: torch.Tensor,
    ) -> dict[str, float]:
        """Update weights once and return source-shaped diagnostic metrics."""

        if policy_observations.shape != demonstration_observations.shape:
            raise ValueError(
                "policy and demonstration AMP batches must have equal shapes, got "
                f"{tuple(policy_observations.shape)} and "
                f"{tuple(demonstration_observations.shape)}"
            )

        self.discriminator.train()
        discriminator_loss, policy_score, demonstration_score = (
            self.discriminator.lsgan_loss(
                policy_observations,
                demonstration_observations,
            )
        )
        gradient_penalty = self.discriminator.gradient_penalty(
            demonstration_observations,
            scale=self.cfg.gradient_penalty_scale,
        )
        total_loss = discriminator_loss + gradient_penalty
        if not torch.isfinite(total_loss):
            raise FloatingPointError("AMP discriminator loss is NaN or Inf")

        self.optimizer.zero_grad(set_to_none=True)
        total_loss.backward()
        gradient_norm = nn.utils.clip_grad_norm_(
            self.discriminator.parameters(),
            self.cfg.max_gradient_norm,
        )
        if not torch.isfinite(gradient_norm):
            self.optimizer.zero_grad(set_to_none=True)
            raise FloatingPointError("AMP discriminator gradient is NaN or Inf")
        self.optimizer.step()

        # Match the source ordering: this batch affects normalization only after
        # the network update, so all samples in one update use the same statistics.
        with torch.no_grad():
            self.discriminator.update_normalization(
                torch.cat((policy_observations, demonstration_observations), dim=0)
            )

        return {
            "amp/disc_loss": float(discriminator_loss.detach()),
            "amp/disc_grad_penalty": float(gradient_penalty.detach()),
            "amp/disc_total_loss": float(total_loss.detach()),
            "amp/disc_score": float(policy_score.detach().mean()),
            "amp/disc_demo_score": float(demonstration_score.detach().mean()),
            "amp/disc_grad_norm_before_clip": float(gradient_norm.detach()),
        }
