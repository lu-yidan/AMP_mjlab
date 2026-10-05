"""Checks for read-only recovery physics health metrics."""

import unittest
from types import SimpleNamespace
from typing import cast

import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers.scene_entity_config import SceneEntityCfg

from g1recovery_amp.tasks.recovery.mdp.metrics import (
    batch_max_contact_penetration,
    batch_max_joint_acceleration,
    max_contact_penetration,
    max_joint_acceleration,
    max_joint_speed,
    max_root_angular_speed,
    unsafe_contact_penetration,
    unsafe_joint_acceleration,
    unsafe_joint_speed,
    unsafe_physics_state,
    unsafe_root_angular_speed,
)


class TestRecoveryMetrics(unittest.TestCase):
    def setUp(self) -> None:
        robot = SimpleNamespace(
            data=SimpleNamespace(
                root_link_ang_vel_w=torch.tensor([[3.0, 4.0, 0.0], [0.0, 0.0, 6.0]]),
                joint_vel=torch.tensor([[1.0, -7.0], [8.0, 2.0]]),
                joint_acc=torch.tensor([[10.0, -20.0], [30.0, 5.0]]),
            )
        )
        contact = SimpleNamespace(
            dim=torch.tensor([3, 0, 3, 3, 0]),
            dist=torch.tensor([-0.01, -9.0, -0.07, 0.02, -8.0]),
            worldid=torch.tensor([0, 0, 1, 1, 1]),
        )
        self.env = cast(
            ManagerBasedRlEnv,
            SimpleNamespace(
                scene={"robot": robot},
                sim=SimpleNamespace(data=SimpleNamespace(contact=contact)),
                num_envs=2,
                device="cpu",
            ),
        )
        self.asset_cfg = SceneEntityCfg("robot", joint_ids=[0, 1])

    def test_peak_state_and_contact_values_are_per_environment(self) -> None:
        torch.testing.assert_close(
            max_root_angular_speed(self.env, self.asset_cfg), torch.tensor([5.0, 6.0])
        )
        torch.testing.assert_close(
            max_joint_speed(self.env, self.asset_cfg), torch.tensor([7.0, 8.0])
        )
        torch.testing.assert_close(
            max_joint_acceleration(self.env, self.asset_cfg),
            torch.tensor([20.0, 30.0]),
        )
        torch.testing.assert_close(
            max_contact_penetration(self.env), torch.tensor([0.01, 0.07])
        )
        torch.testing.assert_close(
            batch_max_joint_acceleration(self.env, self.asset_cfg),
            torch.tensor([30.0, 30.0]),
        )
        torch.testing.assert_close(
            batch_max_contact_penetration(self.env), torch.tensor([0.07, 0.07])
        )

    def test_individual_warning_indicators_identify_the_cause(self) -> None:
        torch.testing.assert_close(
            unsafe_root_angular_speed(self.env, 5.5, self.asset_cfg),
            torch.tensor([0.0, 1.0]),
        )
        torch.testing.assert_close(
            unsafe_joint_speed(self.env, 7.5, self.asset_cfg),
            torch.tensor([0.0, 1.0]),
        )
        torch.testing.assert_close(
            unsafe_joint_acceleration(self.env, 25.0, self.asset_cfg),
            torch.tensor([0.0, 1.0]),
        )
        torch.testing.assert_close(
            unsafe_contact_penetration(self.env, 0.05),
            torch.tensor([0.0, 1.0]),
        )

    def test_warning_indicator_combines_all_thresholds(self) -> None:
        warning = unsafe_physics_state(
            self.env,
            root_angular_speed_threshold=50.0,
            joint_speed_threshold=50.0,
            joint_acceleration_threshold=50.0,
            contact_penetration_threshold=0.05,
            asset_cfg=self.asset_cfg,
        )
        torch.testing.assert_close(warning, torch.tensor([0.0, 1.0]))


if __name__ == "__main__":
    unittest.main()
