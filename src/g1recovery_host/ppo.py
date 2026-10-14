"""Small four-critic PPO implementation for the mjlab HoST task."""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.distributions import Normal

GROUPS = ("task", "regu", "style", "target")
GROUP_WEIGHTS = (2.5, 0.1, 1.0, 1.0)
GAMMA = 0.99


def _mlp(in_dim, hidden, out_dim, tanh_output=False):
    layers = []
    for width in hidden:
        layers.extend((nn.Linear(in_dim, width), nn.ELU()))
        in_dim = width
    layers.append(nn.Linear(in_dim, out_dim))
    if tanh_output:
        layers.append(nn.Tanh())
    return nn.Sequential(*layers)


class HostPolicy(nn.Module):
    """Tanh actor, optional upper-body residual, and four reward-group critics."""

    def __init__(
        self,
        obs_dim=456,
        action_dim=23,
        log_std_bounds=None,
        upper_residual=False,
        waist_residual_scale=2.0,
    ):
        super().__init__()
        self.log_std_bounds = log_std_bounds
        self.upper_residual = (
            nn.Sequential(
                nn.Linear(obs_dim, 64), nn.ELU(), nn.Linear(64, action_dim - 12)
            )
            if upper_residual
            else None
        )
        if self.upper_residual is not None:
            nn.init.zeros_(self.upper_residual[-1].weight)
            nn.init.zeros_(self.upper_residual[-1].bias)
            self.register_buffer(
                "upper_residual_scale",
                torch.tensor([waist_residual_scale] * 3 + [2.0] * (action_dim - 15)),
                persistent=False,
            )
        self.actor = _mlp(obs_dim, (512, 256, 128), action_dim, True)
        self.critics = nn.ModuleList(_mlp(obs_dim, (512, 256), 1) for _ in GROUPS)
        self.log_std = nn.Parameter(torch.full((action_dim,), math.log(0.8)))

    def mean_action(self, obs):
        mean = self.actor(obs)
        if self.upper_residual is not None:
            residual = self.upper_residual_scale * torch.tanh(self.upper_residual(obs))
            mean = torch.cat((mean[:, :12], mean[:, 12:] + residual), dim=-1)
        return mean

    def distribution(self, obs):
        log_std = self.log_std
        if self.log_std_bounds is not None:
            log_std = log_std.clamp(*self.log_std_bounds)
        return Normal(self.mean_action(obs), log_std.exp().expand(obs.shape[0], -1))

    def values(self, obs):
        return torch.cat([critic(obs) for critic in self.critics], dim=-1)

    @torch.no_grad()
    def deterministic_action(self, obs):
        return self.mean_action(obs)


def grouped_gae(
    rewards,
    values,
    dones,
    last_values,
    gamma=GAMMA,
    lam=0.95,
    advantage_scales=None,
):
    """Compute returns and normalize each critic's advantage independently."""
    steps, num_envs, num_groups = rewards.shape
    if num_groups != len(GROUPS):
        raise ValueError(f"Expected {len(GROUPS)} reward groups, got {num_groups}")
    advantages = torch.zeros_like(rewards)
    gae = torch.zeros(num_envs, num_groups, device=rewards.device)
    for step in reversed(range(steps)):
        nonterminal = (~dones[step]).float().unsqueeze(-1)
        next_values = last_values if step == steps - 1 else values[step + 1]
        delta = rewards[step] + gamma * next_values * nonterminal - values[step]
        gae = delta + gamma * lam * nonterminal * gae
        advantages[step] = gae
    returns = advantages + values
    flat = advantages.reshape(-1, num_groups)
    normalized = (advantages - flat.mean(0)) / flat.std(0).clamp_min(1e-6)
    if advantage_scales is not None:
        if advantage_scales.shape != normalized.shape:
            raise ValueError("advantage scales must match rollout reward groups")
        if (
            not torch.isfinite(advantage_scales).all()
            or ((advantage_scales < 0) | (advantage_scales > 1)).any()
        ):
            raise ValueError("advantage scales must be finite values in [0, 1]")
        # Scaling a reward group before its own advantage normalization is
        # nearly cancelled by that normalization. Apply the stage weight here.
        normalized = normalized * advantage_scales
    weights = torch.tensor(GROUP_WEIGHTS, device=rewards.device)
    combined = (normalized * weights).sum(-1)
    return returns, combined


def discounted_progress_advantage(rewards, dones, gamma=GAMMA, lam=0.95):
    """Separate policy-gradient signal for safe board-edge movement.

    Its normalization excludes the rare -15 terminal safety penalty in the
    four-critic target group. This is an auxiliary actor advantage, not a new
    value head or a replacement for the original PPO objective.
    """
    if rewards.ndim != 2 or dones.shape != rewards.shape:
        raise ValueError("progress rewards and done flags must be [time, env]")
    result = torch.zeros_like(rewards)
    future = torch.zeros_like(rewards[0])
    for step in reversed(range(rewards.shape[0])):
        future = rewards[step] + gamma * lam * (~dones[step]).float() * future
        result[step] = future
    spread = result.std()
    if spread < 1e-6:
        return torch.zeros_like(result)
    return ((result - result.mean()) / spread).clamp(-5.0, 5.0)


def ppo_update(
    policy,
    optimizer,
    batch,
    epochs=5,
    minibatches=4,
    adaptive_kl=False,
    desired_kl=0.01,
    max_learning_rate=1e-2,
    frozen_teacher=None,
    frozen_teacher_mask=None,
    frozen_teacher_weight=0.0,
    supine_teacher=None,
    supine_teacher_mask=None,
    supine_teacher_weight=0.0,
    knee_prior_actions=None,
    knee_prior_mask=None,
    knee_prior_joint_ids=(3, 9),
    knee_prior_weight=0.0,
    entropy_coef=0.01,
):
    """Clipped PPO objective with separate value losses and HoST group weights."""
    if len(batch) not in (11, 13):
        raise ValueError(f"Expected 11 or 13 rollout tensors, got {len(batch)}")
    pose_teacher = batch[11] if len(batch) == 13 else None
    pose_teacher_mask = batch[12] if len(batch) == 13 else None
    if (frozen_teacher is None) != (frozen_teacher_mask is None):
        raise ValueError(
            "frozen teacher and its rollout mask must be provided together"
        )
    if frozen_teacher is not None and (
        frozen_teacher is policy
        or any(parameter.requires_grad for parameter in frozen_teacher.parameters())
    ):
        raise ValueError(
            "frozen teacher must be a separate model with gradients disabled"
        )
    if frozen_teacher_weight < 0:
        raise ValueError("frozen teacher weight must be nonnegative")
    if (supine_teacher is None) != (supine_teacher_mask is None):
        raise ValueError("supine teacher and mask must be provided together")
    if supine_teacher is not None and (
        supine_teacher is policy
        or any(parameter.requires_grad for parameter in supine_teacher.parameters())
    ):
        raise ValueError("supine teacher must be a separate frozen policy")
    if not math.isfinite(supine_teacher_weight) or supine_teacher_weight < 0:
        raise ValueError("supine teacher weight must be finite and nonnegative")
    if (knee_prior_actions is None) != (knee_prior_mask is None):
        raise ValueError("knee prior actions and mask must be provided together")
    if knee_prior_weight < 0:
        raise ValueError("knee prior weight must be nonnegative")
    if not math.isfinite(entropy_coef) or entropy_coef < 0:
        raise ValueError("entropy coefficient must be finite and nonnegative")
    if not math.isfinite(max_learning_rate) or max_learning_rate < 1e-5:
        raise ValueError("maximum learning rate must be finite and at least 1e-5")
    (
        obs,
        next_obs,
        actions,
        old_logp,
        old_mu,
        old_sigma,
        old_values,
        returns,
        advantages,
        active,
        continue_mask,
    ) = batch[:11]
    n = obs.shape[0]
    # A raw log_std slightly outside the clamp has zero gradient through the
    # distribution bound. Project the parameter itself so later PPO gradients
    # can move exploration back inside the supported interval.
    if policy.log_std_bounds is not None:
        with torch.no_grad():
            policy.log_std.clamp_(*policy.log_std_bounds)
    if frozen_teacher_mask is not None and frozen_teacher_mask.shape != (n,):
        raise ValueError("frozen teacher mask must have one value per rollout sample")
    if supine_teacher_mask is not None and supine_teacher_mask.shape != (n,):
        raise ValueError("supine teacher mask must have one value per rollout sample")
    if knee_prior_actions is not None and (
        knee_prior_actions.shape != (n, 2) or knee_prior_mask.shape != (n,)
    ):
        raise ValueError("knee prior actions/mask must have shapes (n, 2)/(n,)")
    if n < minibatches:
        raise ValueError(f"Batch size {n} is smaller than {minibatches} minibatches")
    losses = []
    kl_values = []
    actor_parameters = [*policy.actor.parameters(), policy.log_std]
    if policy.upper_residual is not None:
        actor_parameters.extend(policy.upper_residual.parameters())
    for _ in range(epochs):
        for idx in torch.randperm(n, device=obs.device).tensor_split(minibatches):
            dist = policy.distribution(obs[idx])
            with torch.no_grad():
                kl = (
                    torch.log(dist.stddev / old_sigma[idx] + 1e-5)
                    + (old_sigma[idx].square() + (old_mu[idx] - dist.mean).square())
                    / (2.0 * dist.stddev.square())
                    - 0.5
                )
                kl_mean = float(kl.sum(-1).mean())
            kl_values.append(kl_mean)
            if adaptive_kl:
                for group in optimizer.param_groups:
                    group["lr"] = min(group["lr"], max_learning_rate)
                    if kl_mean > 2.0 * desired_kl:
                        group["lr"] = max(1e-5, group["lr"] / 1.5)
                    elif 0.0 < kl_mean < 0.5 * desired_kl:
                        group["lr"] = min(max_learning_rate, group["lr"] * 1.5)
            logp = dist.log_prob(actions[idx]).sum(-1)
            ratio = (logp - old_logp[idx]).exp()
            surrogate = torch.minimum(
                ratio * advantages[idx],
                ratio.clamp(0.8, 1.2) * advantages[idx],
            )
            # The initial 30 policy steps have no motor command in HoST.
            policy_loss = -(surrogate * active[idx]).sum() / active[
                idx
            ].sum().clamp_min(1)
            value = policy.values(obs[idx])
            clipped = old_values[idx] + (value - old_values[idx]).clamp(-0.2, 0.2)
            value_loss = torch.maximum(
                (value - returns[idx]).square(),
                (clipped - returns[idx]).square(),
            ).mean()
            entropy = dist.entropy().sum(-1).mean()
            mix_amount = continue_mask[idx].unsqueeze(-1) * (
                2.0 * torch.rand_like(continue_mask[idx]).unsqueeze(-1) - 1.0
            )
            mixed_obs = obs[idx] + mix_amount * (next_obs[idx] - obs[idx])
            policy_smooth = (
                (dist.mean - policy.mean_action(mixed_obs)).square().sum(-1).mean()
            )
            value_smooth = (value - policy.values(mixed_obs)).square().sum(-1).mean()
            smoothness = (1.0 / 9.0) * policy_smooth + (1.0 / 90.0) * value_smooth
            teacher_loss = obs.new_zeros(())
            if pose_teacher is not None:
                waist_error = (dist.mean[:, 12:15] - pose_teacher[idx]).square().sum(-1)
                teacher_loss = (
                    waist_error * pose_teacher_mask[idx]
                ).sum() / pose_teacher_mask[idx].sum().clamp_min(1)
            frozen_teacher_loss = obs.new_zeros(())
            if frozen_teacher is not None:
                selected = frozen_teacher_mask[idx] > 0
                if selected.any():
                    with torch.no_grad():
                        teacher_obs_dim = frozen_teacher.actor[0].in_features
                        target_action = frozen_teacher.mean_action(
                            obs[idx][selected, :teacher_obs_dim]
                        )
                    frozen_teacher_loss = (
                        (dist.mean[selected] - target_action).square().sum(-1).mean()
                    )
            supine_teacher_loss = obs.new_zeros(())
            if supine_teacher is not None:
                selected = supine_teacher_mask[idx] > 0
                if selected.any():
                    with torch.no_grad():
                        teacher_obs_dim = supine_teacher.actor[0].in_features
                        target_action = supine_teacher.mean_action(
                            obs[idx][selected, :teacher_obs_dim]
                        )
                    supine_teacher_loss = (
                        (dist.mean[selected] - target_action).square().sum(-1).mean()
                    )
            knee_prior_loss = obs.new_zeros(())
            if knee_prior_actions is not None:
                selected = (knee_prior_mask[idx] > 0) & (active[idx] > 0)
                if selected.any():
                    knee_logits = policy.actor[:-1](obs[idx][selected])[
                        :, knee_prior_joint_ids
                    ]
                    target_logits = torch.atanh(
                        knee_prior_actions[idx][selected].clamp(-0.95, 0.95)
                    )
                    knee_prior_loss = (
                        (knee_logits - target_logits).square().sum(-1).mean()
                    )
            loss = (
                policy_loss
                + value_loss
                - entropy_coef * entropy
                + smoothness
                + 0.05 * teacher_loss
                + frozen_teacher_weight * frozen_teacher_loss
                + supine_teacher_weight * supine_teacher_loss
                + knee_prior_weight * knee_prior_loss
            )
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite PPO loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            # MuJoCo's regularization returns can make a critic gradient much
            # larger than the actor's; clip the five independent networks
            # independently so the actor still receives an effective update.
            actor_norm = nn.utils.clip_grad_norm_(actor_parameters, 1.0)
            critic_norms = [
                nn.utils.clip_grad_norm_(critic.parameters(), 1.0)
                for critic in policy.critics
            ]
            optimizer.step()
            if policy.log_std_bounds is not None:
                with torch.no_grad():
                    policy.log_std.clamp_(*policy.log_std_bounds)
            losses.append(
                (
                    float(policy_loss.detach()),
                    float(value_loss.detach()),
                    float(actor_norm),
                    max(float(norm) for norm in critic_norms),
                    float(teacher_loss.detach()),
                    float(frozen_teacher_loss.detach()),
                    float(supine_teacher_loss.detach()),
                    float(knee_prior_loss.detach()),
                )
            )
    return (
        *tuple(sum(item[i] for item in losses) / len(losses) for i in range(8)),
        sum(kl_values) / len(kl_values),
    )
