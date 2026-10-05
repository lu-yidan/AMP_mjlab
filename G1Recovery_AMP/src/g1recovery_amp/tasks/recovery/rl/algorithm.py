"""Connect the migrated AMP components to RSL-RL 5.5 PPO."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import torch
from rsl_rl.algorithms import PPO
from rsl_rl.models import MLPModel
from rsl_rl.storage import RolloutStorage
from tensordict import TensorDict

from g1recovery_amp.tasks.recovery.mdp.recorders import (
    AMP_TERMINAL_DISC_OBSERVATION_KEY,
    AMP_TERMINAL_ENV_IDS_KEY,
    AMP_TERMINAL_OBSTRUCTED_KEY,
)
from g1recovery_amp.tasks.recovery.rl.buffer import (
    AMP_REPLAY_BUFFER_LENGTH,
    AmpReplayBuffer,
)
from g1recovery_amp.tasks.recovery.rl.discriminator import (
    AmpDiscriminator,
    AmpDiscriminatorCfg,
)
from g1recovery_amp.tasks.recovery.rl.optimizer import (
    AmpDiscriminatorOptimizer,
    AmpDiscriminatorOptimizerCfg,
)


@dataclass(frozen=True)
class AmpPpoCfg:
    """AMP-only settings added on top of the ordinary PPO settings."""

    step_dt: float = 0.02
    replay_buffer_length: int = AMP_REPLAY_BUFFER_LENGTH
    discriminator: AmpDiscriminatorCfg = field(default_factory=AmpDiscriminatorCfg)
    optimizer: AmpDiscriminatorOptimizerCfg = field(
        default_factory=AmpDiscriminatorOptimizerCfg
    )
    style_reward_scale_obs_group: str | None = None
    obstructed_style_reward_scale: float = 0.05
    actor_std_range_on_load: tuple[float, float] | None = None
    ppo_learning_rate_on_load: float | None = None


def amp_ppo_cfg_from_dict(config: dict[str, Any]) -> AmpPpoCfg:
    """Rebuild nested dataclasses after the runner converts config to dictionaries."""

    values = dict(config)
    discriminator_values = dict(values.pop("discriminator", {}))
    if "hidden_dims" in discriminator_values:
        discriminator_values["hidden_dims"] = tuple(discriminator_values["hidden_dims"])
    optimizer_values = dict(values.pop("optimizer", {}))
    if values.get("actor_std_range_on_load") is not None:
        values["actor_std_range_on_load"] = tuple(values["actor_std_range_on_load"])
    return AmpPpoCfg(
        discriminator=AmpDiscriminatorCfg(**discriminator_values),
        optimizer=AmpDiscriminatorOptimizerCfg(**optimizer_values),
        **values,
    )


class AmpPPO(PPO):
    """PPO whose task reward is blended with an AMP motion-style reward."""

    def __init__(
        self,
        actor: MLPModel,
        critic: MLPModel,
        storage: RolloutStorage,
        *,
        amp_cfg: AmpPpoCfg | dict[str, Any] | None = None,
        device: str = "cpu",
        multi_gpu_cfg: dict[str, Any] | None = None,
        **ppo_kwargs: Any,
    ) -> None:
        if amp_cfg is None:
            raise ValueError("AmpPPO requires amp_cfg")
        if multi_gpu_cfg is not None:
            raise NotImplementedError(
                "AmpPPO discriminator synchronization is not implemented for multi-GPU"
            )
        if isinstance(amp_cfg, dict):
            amp_cfg = amp_ppo_cfg_from_dict(amp_cfg)
        if amp_cfg.step_dt <= 0.0:
            raise ValueError("AMP step_dt must be positive")
        if not 0.0 <= amp_cfg.obstructed_style_reward_scale <= 1.0:
            raise ValueError("obstructed AMP style reward scale must be in [0, 1]")
        if amp_cfg.actor_std_range_on_load is not None:
            lower, upper = amp_cfg.actor_std_range_on_load
            if lower <= 0.0 or upper < lower:
                raise ValueError("actor std load range must satisfy 0 < lower <= upper")
        if (
            amp_cfg.ppo_learning_rate_on_load is not None
            and amp_cfg.ppo_learning_rate_on_load <= 0.0
        ):
            raise ValueError("PPO load learning rate must be positive")

        super().__init__(
            actor,
            critic,
            storage,
            device=device,
            multi_gpu_cfg=None,
            **ppo_kwargs,
        )
        self.amp_cfg = amp_cfg
        self.amp_discriminator = AmpDiscriminator(amp_cfg.discriminator).to(device)
        self.amp_optimizer = AmpDiscriminatorOptimizer(
            self.amp_discriminator,
            amp_cfg.optimizer,
        )
        self.disc_obs_buffer = AmpReplayBuffer(
            max_length=amp_cfg.replay_buffer_length,
            batch_size=storage.num_envs,
            device=device,
        )
        self.disc_demo_obs_buffer = AmpReplayBuffer(
            max_length=amp_cfg.replay_buffer_length,
            batch_size=storage.num_envs,
            device=device,
        )

        # Exposed for logging and small smoke checks after each environment step.
        self.style_rewards: torch.Tensor | None = None
        self.disc_score: torch.Tensor | None = None
        self.rewards_lerp: torch.Tensor | None = None

    def _replace_reset_observations_with_terminal(
        self,
        disc_observations: torch.Tensor,
        dones: torch.Tensor,
        extras: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Use pre-reset AMP windows for environments that ended this step."""

        done_ids = torch.nonzero(dones.reshape(-1).bool(), as_tuple=False).squeeze(-1)
        if done_ids.numel() == 0:
            return disc_observations

        if (
            AMP_TERMINAL_ENV_IDS_KEY not in extras
            or AMP_TERMINAL_DISC_OBSERVATION_KEY not in extras
        ):
            raise RuntimeError(
                "ended environments are missing pre-reset AMP terminal observations"
            )

        terminal_ids = extras[AMP_TERMINAL_ENV_IDS_KEY].to(
            device=disc_observations.device,
            dtype=torch.long,
        )
        terminal_observations = extras[AMP_TERMINAL_DISC_OBSERVATION_KEY].to(
            device=disc_observations.device,
            dtype=disc_observations.dtype,
        )
        done_ids = done_ids.to(disc_observations.device)
        if not torch.equal(
            torch.sort(terminal_ids).values, torch.sort(done_ids).values
        ):
            raise RuntimeError(
                "terminal AMP environment ids do not match the done environments"
            )
        expected_shape = (terminal_ids.numel(), *disc_observations.shape[1:])
        if tuple(terminal_observations.shape) != expected_shape:
            raise ValueError(
                f"terminal AMP observations must have shape {expected_shape}, got "
                f"{tuple(terminal_observations.shape)}"
            )

        corrected = disc_observations.clone()
        corrected[terminal_ids] = terminal_observations
        return corrected

    def process_env_step(
        self,
        obs: TensorDict,
        rewards: torch.Tensor,
        dones: torch.Tensor,
        extras: dict[str, torch.Tensor],
    ) -> None:
        """Blend rewards and save live/demo motion windows before PPO storage."""

        disc_observations = self._replace_reset_observations_with_terminal(
            obs["disc"],
            dones,
            extras,
        )
        demonstration_observations = obs["disc_demo"]
        self.style_rewards, self.disc_score = self.amp_discriminator.style_reward(
            disc_observations,
            step_dt=self.amp_cfg.step_dt,
        )
        self.style_rewards *= self._style_reward_multiplier(obs, dones, extras)
        self.rewards_lerp = self.amp_discriminator.blend_reward(
            rewards,
            self.style_rewards,
        )
        self.disc_obs_buffer.append(disc_observations)
        self.disc_demo_obs_buffer.append(demonstration_observations)
        super().process_env_step(obs, self.rewards_lerp, dones, extras)

    def _style_reward_multiplier(
        self,
        obs: TensorDict,
        dones: torch.Tensor,
        extras: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Read per-world AMP scaling and repair auto-reset terminal entries."""

        group = self.amp_cfg.style_reward_scale_obs_group
        if group is None:
            return torch.ones_like(dones, dtype=torch.float32)
        if group not in obs:
            raise KeyError(
                f"missing AMP style reward scale observation group {group!r}"
            )
        multiplier = obs[group]
        if not isinstance(multiplier, torch.Tensor):
            raise TypeError("AMP style reward scale group must be a tensor")
        if multiplier.ndim == 2 and multiplier.shape[1] == 1:
            multiplier = multiplier[:, 0]
        if multiplier.shape != dones.reshape(-1).shape:
            raise ValueError(
                "AMP style reward scale must have shape "
                f"{tuple(dones.reshape(-1).shape)}, got {tuple(multiplier.shape)}"
            )
        multiplier = multiplier.to(device=dones.device, dtype=torch.float32).clone()
        if not torch.isfinite(multiplier).all() or torch.any(
            (multiplier < 0.0) | (multiplier > 1.0)
        ):
            raise ValueError("AMP style reward scales must be finite and in [0, 1]")

        done_ids = torch.nonzero(dones.reshape(-1).bool(), as_tuple=False).squeeze(-1)
        if done_ids.numel() == 0:
            return multiplier
        if AMP_TERMINAL_OBSTRUCTED_KEY not in extras:
            raise RuntimeError(
                "ended environments are missing terminal A6 obstruction state"
            )
        terminal_ids = extras[AMP_TERMINAL_ENV_IDS_KEY].to(
            device=dones.device, dtype=torch.long
        )
        terminal_obstructed = extras[AMP_TERMINAL_OBSTRUCTED_KEY].to(
            device=dones.device, dtype=torch.bool
        )
        if terminal_obstructed.shape != terminal_ids.shape:
            raise ValueError("terminal A6 obstruction state must match terminal ids")
        multiplier[terminal_ids] = torch.where(
            terminal_obstructed,
            torch.full_like(
                terminal_obstructed,
                self.amp_cfg.obstructed_style_reward_scale,
                dtype=torch.float32,
            ),
            torch.ones_like(terminal_obstructed, dtype=torch.float32),
        )
        return multiplier

    def _update_amp_discriminator(self) -> dict[str, float]:
        """Run the same number of discriminator mini-batches as PPO mini-batches."""

        fetch_length = self.storage.num_transitions_per_env
        policy_batches = self.disc_obs_buffer.mini_batch_generator(
            fetch_length=fetch_length,
            num_mini_batches=self.num_mini_batches,
            num_epochs=self.num_learning_epochs,
        )
        demonstration_batches = self.disc_demo_obs_buffer.mini_batch_generator(
            fetch_length=fetch_length,
            num_mini_batches=self.num_mini_batches,
            num_epochs=self.num_learning_epochs,
        )

        totals: dict[str, float] = {}
        update_count = 0
        for policy_batch, demonstration_batch in zip(
            policy_batches,
            demonstration_batches,
            strict=True,
        ):
            metrics = self.amp_optimizer.update(policy_batch, demonstration_batch)
            for name, value in metrics.items():
                totals[name] = totals.get(name, 0.0) + value
            update_count += 1
        if update_count == 0:
            raise RuntimeError("AMP discriminator update produced no mini-batches")
        return {name: value / update_count for name, value in totals.items()}

    def update(self) -> dict[str, float]:
        """Update the discriminator and the ordinary actor/critic networks."""

        amp_metrics = self._update_amp_discriminator()
        ppo_metrics = super().update()
        return {**ppo_metrics, **amp_metrics}

    def train_mode(self) -> None:
        super().train_mode()
        self.amp_discriminator.train()

    def eval_mode(self) -> None:
        super().eval_mode()
        self.amp_discriminator.eval()

    def save(self) -> dict:
        saved = super().save()
        saved["amp_discriminator_state_dict"] = self.amp_discriminator.state_dict()
        saved["amp_discriminator_optimizer_state_dict"] = (
            self.amp_optimizer.optimizer.state_dict()
        )
        return saved

    def load(self, loaded_dict: dict, load_cfg: dict | None, strict: bool) -> bool:
        load_amp = load_cfg is None or bool(load_cfg.get("amp", False))
        load_iteration = super().load(loaded_dict, load_cfg, strict)
        if load_amp:
            self.amp_discriminator.load_state_dict(
                loaded_dict["amp_discriminator_state_dict"],
                strict=strict,
            )
            self.amp_optimizer.optimizer.load_state_dict(
                loaded_dict["amp_discriminator_optimizer_state_dict"]
            )
        self._sanitize_loaded_actor_std()
        self._restore_configured_ppo_learning_rate()
        return load_iteration

    def _restore_configured_ppo_learning_rate(self) -> None:
        """Override a resumed optimizer's saved LR when explicitly requested."""

        learning_rate = self.amp_cfg.ppo_learning_rate_on_load
        if learning_rate is None:
            return
        self.learning_rate = learning_rate
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = learning_rate

    def _sanitize_loaded_actor_std(self) -> None:
        """Move a resumed policy's raw std parameter inside the configured range.

        RSL-RL clamps the effective standard deviation during sampling, but a
        raw parameter above the clamp has zero gradient.  Clamping the parameter
        itself and dropping its stale Adam moments lets training recover from a
        previously exploded exploration value.
        """

        bounds = self.amp_cfg.actor_std_range_on_load
        if bounds is None:
            return
        distribution = getattr(self._raw_actor, "distribution", None)
        if distribution is None:
            raise TypeError("actor does not expose a distribution to sanitize")
        lower, upper = bounds
        if getattr(distribution, "std_type", None) == "scalar":
            parameter = distribution.std_param
            clamp_bounds = (lower, upper)
        elif getattr(distribution, "std_type", None) == "log":
            parameter = distribution.log_std_param
            clamp_bounds = (math.log(lower), math.log(upper))
        else:
            raise ValueError("unsupported actor standard-deviation parameterization")
        with torch.no_grad():
            parameter.clamp_(*clamp_bounds)
        self.optimizer.state.pop(parameter, None)
