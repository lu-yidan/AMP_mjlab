"""Checks for AMP observations captured immediately before auto-reset."""

import unittest
from types import SimpleNamespace
from typing import cast

import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers.recorder_manager import RecorderTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg

from g1recovery_amp.tasks.recovery import mdp


class TestAmpTerminalObservations(unittest.TestCase):
    def test_history_drops_oldest_and_appends_terminal_frame(self) -> None:
        history = torch.arange(2 * 10 * 85, dtype=torch.float32).reshape(2, 10, 85)
        current = torch.full((2, 85), -7.0)

        result = mdp.advance_amp_observation_history(history, current)

        torch.testing.assert_close(result[:, :-1], history[:, 1:])
        torch.testing.assert_close(result[:, -1], current)

    def test_recorder_exports_only_resetting_environment(self) -> None:
        num_envs = 2
        previous = torch.zeros(num_envs, 10, 85)
        previous[:, :, 0] = torch.arange(10)
        identity_quat = torch.zeros(num_envs, 4)
        identity_quat[:, 0] = 1.0
        data = SimpleNamespace(
            root_link_pos_w=torch.zeros(num_envs, 3),
            root_link_quat_w=identity_quat,
            root_link_ang_vel_w=torch.zeros(num_envs, 3),
            joint_pos=torch.zeros(num_envs, 29),
            joint_vel=torch.zeros(num_envs, 29),
            body_link_pos_w=torch.zeros(num_envs, 6, 3),
        )
        fake_env = SimpleNamespace(
            scene={"robot": SimpleNamespace(data=data)},
            obs_buf={"disc": previous},
            extras={},
            reset_buf=torch.tensor([False, True]),
        )
        asset_cfg = SceneEntityCfg("robot")
        asset_cfg.body_ids = list(range(6))
        recorder = mdp.AmpTerminalObservationRecorder(
            RecorderTermCfg(
                func=mdp.AmpTerminalObservationRecorder,
                params={"asset_cfg": asset_cfg},
            ),
            cast(ManagerBasedRlEnv, fake_env),
        )

        recorder.record_pre_reset(torch.tensor([1]))

        self.assertEqual(
            fake_env.extras[mdp.AMP_TERMINAL_ENV_IDS_KEY].tolist(),
            [1],
        )
        terminal = fake_env.extras[mdp.AMP_TERMINAL_DISC_OBSERVATION_KEY]
        self.assertEqual(terminal.shape, (1, 10, 85))
        torch.testing.assert_close(terminal[0, :-1], previous[1, 1:])
        expected_current = mdp.amp_discriminator_frame_from_state(
            data.root_link_pos_w[1:2],
            data.root_link_quat_w[1:2],
            data.root_link_ang_vel_w[1:2],
            data.joint_pos[1:2],
            data.joint_vel[1:2],
            data.body_link_pos_w[1:2],
        )
        torch.testing.assert_close(terminal[:, -1], expected_current)

    def test_non_terminal_step_removes_stale_extras(self) -> None:
        fake_env = SimpleNamespace(
            scene={"robot": SimpleNamespace(data=SimpleNamespace())},
            obs_buf={},
            extras={
                mdp.AMP_TERMINAL_ENV_IDS_KEY: torch.tensor([0]),
                mdp.AMP_TERMINAL_DISC_OBSERVATION_KEY: torch.zeros(1, 10, 85),
            },
            reset_buf=torch.tensor([False]),
        )
        recorder = mdp.AmpTerminalObservationRecorder(
            RecorderTermCfg(
                func=mdp.AmpTerminalObservationRecorder,
                params={"asset_cfg": SceneEntityCfg("robot")},
            ),
            cast(ManagerBasedRlEnv, fake_env),
        )

        recorder.record_post_step()

        self.assertNotIn(mdp.AMP_TERMINAL_ENV_IDS_KEY, fake_env.extras)
        self.assertNotIn(mdp.AMP_TERMINAL_DISC_OBSERVATION_KEY, fake_env.extras)


if __name__ == "__main__":
    unittest.main()
