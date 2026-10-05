"""Checks for the layer connecting AMP to RSL-RL 5.5 PPO."""

from __future__ import annotations

import math
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from rsl_rl.algorithms import PPO
from rsl_rl.models import MLPModel
from rsl_rl.storage import RolloutStorage
from tensordict import TensorDict
from torch import nn

from g1recovery_amp.tasks.recovery.mdp.recorders import (
    AMP_TERMINAL_DISC_OBSERVATION_KEY,
    AMP_TERMINAL_ENV_IDS_KEY,
    AMP_TERMINAL_OBSTRUCTED_KEY,
)
from g1recovery_amp.tasks.recovery.rl import AmpPPO, AmpPpoCfg
from g1recovery_amp.tasks.recovery.rl.discriminator import AmpDiscriminatorCfg


class _TinyModel(nn.Module):
    is_recurrent = False

    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(()))


class _TinyGaussianActor(_TinyModel):
    def __init__(self, std: float) -> None:
        super().__init__()
        self.distribution = nn.Module()
        self.distribution.std_type = "scalar"  # type: ignore[attr-defined]
        self.distribution.std_param = nn.Parameter(  # type: ignore[attr-defined]
            torch.full((2,), std)
        )


def _make_algorithm(
    amp_cfg: AmpPpoCfg | None = None,
    actor: nn.Module | None = None,
) -> AmpPPO:
    storage = SimpleNamespace(num_envs=2, num_transitions_per_env=2)
    return AmpPPO(
        actor or _TinyModel(),  # type: ignore[arg-type]
        _TinyModel(),  # type: ignore[arg-type]
        storage,  # type: ignore[arg-type]
        amp_cfg=amp_cfg
        or AmpPpoCfg(
            replay_buffer_length=3,
            discriminator=AmpDiscriminatorCfg(hidden_dims=(16, 8)),
        ),
        num_learning_epochs=1,
        num_mini_batches=1,
    )


class TestAmpPPO(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(17)
        self.algorithm = _make_algorithm()

    def test_process_step_replaces_reset_state_and_blends_reward(self) -> None:
        live = torch.zeros(2, 10, 85)
        demonstration = torch.full_like(live, 0.25)
        terminal = torch.full((1, 10, 85), 2.0)
        rewards = torch.tensor([0.4, 0.8])
        dones = torch.tensor([False, True])
        extras = {
            AMP_TERMINAL_ENV_IDS_KEY: torch.tensor([1]),
            AMP_TERMINAL_DISC_OBSERVATION_KEY: terminal,
        }
        expected_live = live.clone()
        expected_live[1] = terminal[0]
        expected_style, _ = self.algorithm.amp_discriminator.style_reward(
            expected_live,
            step_dt=0.02,
        )

        with patch.object(PPO, "process_env_step") as parent_process:
            self.algorithm.process_env_step(
                {"disc": live, "disc_demo": demonstration},  # type: ignore[arg-type]
                rewards,
                dones,
                extras,
            )

        stored_live = self.algorithm.disc_obs_buffer.chronological()[0]
        stored_demo = self.algorithm.disc_demo_obs_buffer.chronological()[0]
        self.assertTrue(torch.equal(stored_live, expected_live))
        self.assertTrue(torch.equal(stored_demo, demonstration))
        expected_blend = 0.5 * rewards + 0.5 * expected_style
        blended_rewards = self.algorithm.rewards_lerp
        self.assertIsNotNone(blended_rewards)
        assert blended_rewards is not None
        self.assertTrue(torch.allclose(blended_rewards, expected_blend))
        passed_rewards = parent_process.call_args.args[1]
        self.assertTrue(torch.allclose(passed_rewards, expected_blend))

    def test_done_without_terminal_observation_fails_early(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "missing pre-reset"):
            self.algorithm._replace_reset_observations_with_terminal(
                torch.zeros(2, 10, 85),
                torch.tensor([True, False]),
                {},
            )

    def test_obstructed_style_reward_is_scaled_including_terminal_step(self) -> None:
        algorithm = _make_algorithm(
            AmpPpoCfg(
                replay_buffer_length=3,
                discriminator=AmpDiscriminatorCfg(hidden_dims=(16, 8)),
                style_reward_scale_obs_group="amp_reward_scale",
                obstructed_style_reward_scale=0.05,
            )
        )
        live = torch.zeros(2, 10, 85)
        terminal = torch.zeros(1, 10, 85)
        dones = torch.tensor([True, False])
        extras = {
            AMP_TERMINAL_ENV_IDS_KEY: torch.tensor([0]),
            AMP_TERMINAL_DISC_OBSERVATION_KEY: terminal,
            AMP_TERMINAL_OBSTRUCTED_KEY: torch.tensor([True]),
        }
        # The returned observation for env 0 is post-reset and says 1.0.  The
        # terminal extra must repair it back to the obstructed 0.05 scale.
        obs = {
            "disc": live,
            "disc_demo": live.clone(),
            "amp_reward_scale": torch.ones(2, 1),
        }
        with (
            patch.object(
                algorithm.amp_discriminator,
                "style_reward",
                return_value=(torch.tensor([2.0, 4.0]), torch.zeros(2)),
            ),
            patch.object(PPO, "process_env_step"),
        ):
            algorithm.process_env_step(  # type: ignore[arg-type]
                obs,
                torch.zeros(2),
                dones,
                extras,
            )

        assert algorithm.style_rewards is not None
        torch.testing.assert_close(
            algorithm.style_rewards,
            torch.tensor([0.1, 4.0]),
        )

    def test_discriminator_update_uses_both_replay_buffers(self) -> None:
        for _ in range(2):
            self.algorithm.disc_obs_buffer.append(torch.randn(2, 10, 85))
            self.algorithm.disc_demo_obs_buffer.append(torch.randn(2, 10, 85))

        before = [
            parameter.detach().clone()
            for parameter in self.algorithm.amp_discriminator.parameters()
        ]
        metrics = self.algorithm._update_amp_discriminator()

        self.assertTrue(all(math.isfinite(value) for value in metrics.values()))
        self.assertTrue(
            any(
                not torch.equal(old, new)
                for old, new in zip(
                    before,
                    self.algorithm.amp_discriminator.parameters(),
                )
            )
        )

    def test_checkpoint_round_trip_restores_ppo_and_amp_state(self) -> None:
        with torch.no_grad():
            next(self.algorithm._raw_actor.parameters()).fill_(1.25)
            next(self.algorithm._raw_critic.parameters()).fill_(-0.75)
            self.algorithm.amp_discriminator.disc_linear.bias.fill_(0.5)
            self.algorithm.amp_discriminator.update_normalization(
                torch.randn(2, 10, 85)
            )
        checkpoint = self.algorithm.save()
        self.assertIn("amp_discriminator_state_dict", checkpoint)
        self.assertIn("amp_discriminator_optimizer_state_dict", checkpoint)

        restored = _make_algorithm()
        restored.load(checkpoint, load_cfg=None, strict=True)

        self.assertTrue(
            torch.equal(
                next(restored._raw_actor.parameters()),
                next(self.algorithm._raw_actor.parameters()),
            )
        )
        self.assertTrue(
            torch.equal(
                next(restored._raw_critic.parameters()),
                next(self.algorithm._raw_critic.parameters()),
            )
        )
        for name, expected in self.algorithm.amp_discriminator.state_dict().items():
            actual = restored.amp_discriminator.state_dict()[name]
            self.assertIsInstance(expected, torch.Tensor)
            self.assertIsInstance(actual, torch.Tensor)
            assert isinstance(expected, torch.Tensor)
            assert isinstance(actual, torch.Tensor)
            self.assertTrue(
                torch.equal(actual, expected),
                msg=f"AMP checkpoint mismatch for {name}",
            )

    def test_resume_sanitizes_exploded_actor_std_parameter(self) -> None:
        source = _make_algorithm(actor=_TinyGaussianActor(19.5))
        checkpoint = source.save()
        restored = _make_algorithm(
            AmpPpoCfg(
                replay_buffer_length=3,
                discriminator=AmpDiscriminatorCfg(hidden_dims=(16, 8)),
                actor_std_range_on_load=(0.05, 1.5),
            ),
            actor=_TinyGaussianActor(1.0),
        )

        restored.load(checkpoint, load_cfg=None, strict=True)

        parameter = restored._raw_actor.distribution.std_param
        torch.testing.assert_close(parameter, torch.full((2,), 1.5))
        self.assertNotIn(parameter, restored.optimizer.state)

    def test_resume_can_override_checkpoint_ppo_learning_rate(self) -> None:
        source = _make_algorithm()
        checkpoint = source.save()
        for param_group in checkpoint["optimizer_state_dict"]["param_groups"]:
            param_group["lr"] = 5.0625e-5
        restored = _make_algorithm(
            AmpPpoCfg(
                replay_buffer_length=3,
                discriminator=AmpDiscriminatorCfg(hidden_dims=(16, 8)),
                ppo_learning_rate_on_load=2.25e-5,
            )
        )

        restored.load(checkpoint, load_cfg=None, strict=True)

        self.assertEqual(restored.learning_rate, 2.25e-5)
        self.assertTrue(
            all(group["lr"] == 2.25e-5 for group in restored.optimizer.param_groups)
        )

    def test_real_ppo_components_complete_one_tiny_update(self) -> None:
        observation = TensorDict(
            {
                "actor": torch.randn(2, 3),
                "critic": torch.randn(2, 4),
                "disc": torch.randn(2, 10, 85),
                "disc_demo": torch.randn(2, 10, 85),
            },
            batch_size=[2],
        )
        observation_groups = {"actor": ["actor"], "critic": ["critic"]}
        actor = MLPModel(
            observation,
            observation_groups,
            "actor",
            2,
            hidden_dims=(8,),
            distribution_cfg={
                "class_name": "GaussianDistribution",
                "init_std": 1.0,
                "std_type": "scalar",
            },
        )
        critic = MLPModel(
            observation,
            observation_groups,
            "critic",
            1,
            hidden_dims=(8,),
        )
        storage = RolloutStorage("rl", 2, 2, observation, [2])
        algorithm = AmpPPO(
            actor,
            critic,
            storage,
            amp_cfg=AmpPpoCfg(
                replay_buffer_length=3,
                discriminator=AmpDiscriminatorCfg(hidden_dims=(16, 8)),
            ),
            num_learning_epochs=1,
            num_mini_batches=1,
        )

        for _ in range(2):
            algorithm.act(observation)
            observation = observation.clone()
            observation["actor"] = torch.randn(2, 3)
            observation["critic"] = torch.randn(2, 4)
            observation["disc"] = torch.randn(2, 10, 85)
            observation["disc_demo"] = torch.randn(2, 10, 85)
            algorithm.process_env_step(
                observation,
                torch.rand(2),
                torch.zeros(2, dtype=torch.bool),
                {},
            )

        algorithm.compute_returns(observation)
        metrics = algorithm.update()

        self.assertEqual(storage.step, 0)
        self.assertTrue({"value", "surrogate", "entropy"}.issubset(metrics))
        self.assertIn("amp/disc_total_loss", metrics)
        self.assertTrue(all(math.isfinite(value) for value in metrics.values()))

    def test_multi_gpu_is_rejected_until_discriminator_sync_exists(self) -> None:
        with self.assertRaisesRegex(NotImplementedError, "multi-GPU"):
            AmpPPO(
                _TinyModel(),  # type: ignore[arg-type]
                _TinyModel(),  # type: ignore[arg-type]
                SimpleNamespace(num_envs=2),  # type: ignore[arg-type]
                amp_cfg=AmpPpoCfg(),
                multi_gpu_cfg={"global_rank": 0, "world_size": 2},
            )


if __name__ == "__main__":
    unittest.main()
