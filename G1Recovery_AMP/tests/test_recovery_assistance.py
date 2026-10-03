"""Checks for the source-compatible get-up assistance curriculum."""

import unittest
from types import SimpleNamespace
from typing import cast
from unittest.mock import Mock

import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers.scene_entity_config import SceneEntityCfg

from g1recovery_amp.tasks.recovery.mdp import (
    GET_UP_ASSIST_FORCE_STEP,
    apply_get_up_assist_force,
    get_up_assist_force_level,
)


class TestRecoveryAssistance(unittest.TestCase):
    def test_reset_event_applies_only_upward_torso_force(self) -> None:
        robot = Mock()
        robot.num_bodies = 1
        command = SimpleNamespace(command=torch.tensor([[200.0], [150.0]]))
        command_manager = Mock()
        command_manager.get_term.return_value = command
        env = cast(
            ManagerBasedRlEnv,
            SimpleNamespace(
                scene={"robot": robot},
                command_manager=command_manager,
                num_envs=2,
                device="cpu",
            ),
        )
        asset_cfg = SceneEntityCfg("robot", body_ids=[3])

        apply_get_up_assist_force(env, torch.tensor([0, 1]), asset_cfg)

        call = robot.write_external_wrench_to_sim.call_args
        expected_force = torch.tensor([[[0.0, 0.0, 200.0]], [[0.0, 0.0, 150.0]]])
        torch.testing.assert_close(call.args[0], expected_force)
        torch.testing.assert_close(call.args[1], torch.zeros_like(expected_force))
        self.assertEqual(call.kwargs["body_ids"], [3])
        torch.testing.assert_close(call.kwargs["env_ids"], torch.tensor([0, 1]))

    def test_successful_episode_reduces_only_reset_environments(self) -> None:
        command = SimpleNamespace(command=torch.tensor([[200.0], [120.0]]))
        command._command = command.command
        command_manager = Mock()
        command_manager.get_term.return_value = command
        reward_manager = Mock()
        reward_manager._episode_sums = {"target_base_height": torch.tensor([40.0, 0.0])}
        reward_manager.get_term_cfg.return_value = SimpleNamespace(weight=5.0)
        env = cast(
            ManagerBasedRlEnv,
            SimpleNamespace(
                command_manager=command_manager,
                reward_manager=reward_manager,
                max_episode_length_s=10.0,
            ),
        )

        mean_force = get_up_assist_force_level(
            env, torch.tensor([0]), "target_base_height"
        )

        self.assertEqual(command.command[:, 0].tolist(), [190.0, 120.0])
        self.assertEqual(float(mean_force), 155.0)

    def test_assistance_never_becomes_negative(self) -> None:
        command = SimpleNamespace(command=torch.tensor([[5.0]]))
        command._command = command.command
        command_manager = Mock()
        command_manager.get_term.return_value = command
        reward_manager = Mock()
        reward_manager._episode_sums = {"target_base_height": torch.tensor([50.0])}
        reward_manager.get_term_cfg.return_value = SimpleNamespace(weight=5.0)
        env = cast(
            ManagerBasedRlEnv,
            SimpleNamespace(
                command_manager=command_manager,
                reward_manager=reward_manager,
                max_episode_length_s=10.0,
            ),
        )

        get_up_assist_force_level(env, torch.tensor([0]), "target_base_height")

        self.assertEqual(GET_UP_ASSIST_FORCE_STEP, 10.0)
        self.assertEqual(float(command.command[0, 0]), 0.0)


if __name__ == "__main__":
    unittest.main()
