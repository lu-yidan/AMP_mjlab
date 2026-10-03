"""Checks for recovery-specific RSL-RL and AMP registration."""

import unittest
from dataclasses import asdict
from types import SimpleNamespace

from mjlab.tasks.registry import load_rl_cfg, load_runner_cls
from rsl_rl.utils import resolve_callable

from g1recovery_amp.tasks.recovery import A6_G_PLUS_TASK_ID, DEV_TASK_ID
from g1recovery_amp.tasks.recovery.rl import (
    A6_ADAPTATION_STATE_KEY,
    A6_AMP_STYLE_REWARD_SCALE,
    AMP_PPO_CLASS_PATH,
    AMP_REPLAY_BUFFER_LENGTH,
    AmpPPO,
    RecoveryAmpPpoAlgorithmCfg,
    RecoveryOnPolicyRunner,
    add_a6_adaptation_state,
    g1_recovery_ppo_runner_cfg,
    restore_a6_adaptation_state,
)


class TestRecoveryRlCfg(unittest.TestCase):
    def test_flat_parent_rebases_a6_curriculum_clock(self) -> None:
        env = SimpleNamespace(
            common_step_counter=480_000, _a6_curriculum_start_counter=0
        )
        restore_a6_adaptation_state(env, {"env_state": {}})  # type: ignore[arg-type]
        self.assertEqual(env._a6_curriculum_start_counter, 480_000)

    def test_a6_curriculum_clock_round_trip(self) -> None:
        source = SimpleNamespace(
            common_step_counter=487_200, _a6_curriculum_start_counter=480_000
        )
        infos = add_a6_adaptation_state(source, {"existing": 1})  # type: ignore[arg-type]
        assert infos is not None
        self.assertEqual(infos["existing"], 1)
        self.assertEqual(
            infos[A6_ADAPTATION_STATE_KEY], {"curriculum_start_counter": 480_000}
        )
        target = SimpleNamespace(
            common_step_counter=487_200, _a6_curriculum_start_counter=0
        )
        restore_a6_adaptation_state(target, infos)  # type: ignore[arg-type]
        self.assertEqual(target._a6_curriculum_start_counter, 480_000)

    def test_source_get_up_training_settings(self) -> None:
        cfg = g1_recovery_ppo_runner_cfg()

        self.assertEqual(cfg.num_steps_per_env, 24)
        self.assertEqual(cfg.max_iterations, 500_000)
        self.assertEqual(cfg.save_interval, 50)
        self.assertEqual(cfg.clip_actions, 3.5)
        self.assertEqual(cfg.obs_groups, {"actor": ("actor",), "critic": ("critic",)})
        self.assertEqual(cfg.actor.hidden_dims, (512, 256, 128))
        self.assertEqual(cfg.critic.hidden_dims, (512, 256, 128))
        self.assertFalse(cfg.actor.obs_normalization)
        self.assertFalse(cfg.critic.obs_normalization)

        algorithm = cfg.algorithm
        self.assertIsInstance(algorithm, RecoveryAmpPpoAlgorithmCfg)
        self.assertEqual(algorithm.class_name, AMP_PPO_CLASS_PATH)
        self.assertEqual(algorithm.entropy_coef, 0.01)
        self.assertEqual(algorithm.num_learning_epochs, 5)
        self.assertEqual(algorithm.num_mini_batches, 4)
        assert isinstance(algorithm, RecoveryAmpPpoAlgorithmCfg)
        self.assertEqual(
            algorithm.amp_cfg.replay_buffer_length,
            AMP_REPLAY_BUFFER_LENGTH,
        )
        self.assertEqual(algorithm.amp_cfg.discriminator.hidden_dims, (1024, 512))
        self.assertEqual(algorithm.amp_cfg.discriminator.task_style_lerp, 0.5)

    def test_runner_dictionary_resolves_task_owned_amp_algorithm(self) -> None:
        algorithm_cfg = asdict(g1_recovery_ppo_runner_cfg())["algorithm"]
        self.assertIs(resolve_callable(algorithm_cfg["class_name"]), AmpPPO)
        self.assertEqual(algorithm_cfg["amp_cfg"]["step_dt"], 0.02)

    def test_registry_uses_recovery_runner_adapter(self) -> None:
        cfg = load_rl_cfg(DEV_TASK_ID)
        self.assertEqual(cfg.algorithm.class_name, AMP_PPO_CLASS_PATH)  # type: ignore[attr-defined]
        self.assertIs(load_runner_cls(DEV_TASK_ID), RecoveryOnPolicyRunner)

    def test_a6_registry_preserves_native_style_scale(self) -> None:
        flat_cfg = load_rl_cfg(DEV_TASK_ID)
        a6_cfg = load_rl_cfg(A6_G_PLUS_TASK_ID)
        self.assertEqual(
            flat_cfg.algorithm.amp_cfg.discriminator.style_reward_scale,  # type: ignore[attr-defined]
            50.0,
        )
        self.assertEqual(
            a6_cfg.algorithm.amp_cfg.discriminator.style_reward_scale,  # type: ignore[attr-defined]
            A6_AMP_STYLE_REWARD_SCALE,
        )
        self.assertEqual(a6_cfg.obs_groups, flat_cfg.obs_groups)


if __name__ == "__main__":
    unittest.main()
