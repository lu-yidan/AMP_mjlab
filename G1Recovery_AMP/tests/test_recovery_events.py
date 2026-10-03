"""Checks for recovery-specific disturbance events."""

import math
import unittest
from types import SimpleNamespace
from typing import cast
from unittest.mock import Mock

import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg

from g1recovery_amp.tasks.recovery.mdp import (
    MixedRecoveryDisturbance,
    SustainedBodyWrench,
    randomize_body_mass_and_inertia,
    sample_base_push_velocity,
)


class TestRecoveryEvents(unittest.TestCase):
    def test_mass_randomization_scales_inertia_with_mass(self) -> None:
        robot = SimpleNamespace(
            indexing=SimpleNamespace(body_ids=torch.tensor([1, 3]))
        )
        default_mass = torch.tensor([0.0, 8.0, 2.0, 4.0])
        default_inertia = torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [1.0, 2.0, 3.0],
                [0.2, 0.3, 0.4],
                [0.5, 0.6, 0.7],
            ]
        )
        model = SimpleNamespace(
            body_mass=default_mass.expand(2, -1).clone(),
            body_inertia=default_inertia.expand(2, -1, -1).clone(),
        )
        defaults = {"body_mass": default_mass, "body_inertia": default_inertia}
        sim = SimpleNamespace(
            model=model,
            per_world_default_fields=set(),
            get_default_field=lambda field: defaults[field],
        )
        env = cast(
            ManagerBasedRlEnv,
            SimpleNamespace(
                scene={"robot": robot},
                sim=sim,
                num_envs=2,
                device="cpu",
            ),
        )

        randomize_body_mass_and_inertia(
            env,
            None,
            mass_delta_range=(1.0, 1.0),
            asset_cfg=SceneEntityCfg("robot", body_ids=[0]),
        )

        torch.testing.assert_close(model.body_mass[:, 1], torch.tensor([9.0, 9.0]))
        torch.testing.assert_close(
            model.body_inertia[:, 1],
            torch.tensor([[1.125, 2.25, 3.375], [1.125, 2.25, 3.375]]),
        )
        torch.testing.assert_close(model.body_mass[:, 3], torch.tensor([4.0, 4.0]))

    def test_base_push_only_replaces_horizontal_and_yaw_velocity(self) -> None:
        torch.manual_seed(7)
        root_velocity = torch.tensor(
            [
                [3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
                [-3.0, -4.0, -5.0, -6.0, -7.0, -8.0],
            ]
        )

        pushed = sample_base_push_velocity(
            root_velocity,
            linear_velocity_range=(-0.8, 0.8),
            yaw_velocity_range=(-1.0, 1.0),
        )

        self.assertTrue(torch.all(pushed[:, 0:2] >= -0.8))
        self.assertTrue(torch.all(pushed[:, 0:2] <= 0.8))
        self.assertTrue(torch.all(pushed[:, 5] >= -1.0))
        self.assertTrue(torch.all(pushed[:, 5] <= 1.0))
        torch.testing.assert_close(pushed[:, 2:5], root_velocity[:, 2:5])
        torch.testing.assert_close(root_velocity[0], torch.arange(3.0, 9.0))

    def test_sustained_wrench_waits_resamples_and_clears_on_reset(self) -> None:
        robot = Mock()
        robot.num_bodies = 3
        robot.indexing = SimpleNamespace(body_ids=torch.tensor([1, 2, 3]))
        half_sqrt = math.sqrt(0.5)
        robot.data = SimpleNamespace(
            body_link_quat_w=torch.tensor(
                [
                    [
                        [half_sqrt, 0.0, 0.0, half_sqrt],
                        [1.0, 0.0, 0.0, 0.0],
                        [1.0, 0.0, 0.0, 0.0],
                    ]
                ]
            )
        )
        asset_cfg = SceneEntityCfg("robot", body_ids=[0, 2])
        cfg = EventTermCfg(
            func=SustainedBodyWrench,
            mode="step",
            params={"asset_cfg": asset_cfg, "period_s": (0.04, 0.04)},
        )
        env = cast(
            ManagerBasedRlEnv,
            SimpleNamespace(
                scene={"robot": robot},
                sim=SimpleNamespace(
                    model=SimpleNamespace(
                        body_mass=torch.tensor([[1.0, 2.0, 3.0, 4.0]])
                    )
                ),
                num_envs=1,
                device="cpu",
                step_dt=0.02,
            ),
        )
        event = SustainedBodyWrench(cfg, env)
        event._time_left.zero_()

        event(
            env,
            None,
            linear_acceleration_range=(10.0, 10.0),
            ramp_s=0.04,
            period_s=(0.04, 0.04),
            asset_cfg=asset_cfg,
        )
        first_write = robot.write_external_wrench_to_sim.call_args
        torch.testing.assert_close(first_write.args[0], torch.zeros((1, 2, 3)))
        torch.testing.assert_close(first_write.args[1], torch.zeros((1, 2, 3)))
        torch.testing.assert_close(
            event.force,
            torch.tensor([[[20.0, 20.0, 20.0], [40.0, 40.0, 40.0]]]),
        )

        event(
            env,
            None,
            linear_acceleration_range=(10.0, 10.0),
            ramp_s=0.04,
            period_s=(0.04, 0.04),
            asset_cfg=asset_cfg,
        )
        write_call = robot.write_external_wrench_to_sim.call_args
        torch.testing.assert_close(
            write_call.args[0],
            torch.tensor([[[-10.0, 10.0, 10.0], [20.0, 20.0, 20.0]]]),
        )
        torch.testing.assert_close(
            write_call.args[1], torch.zeros((1, 2, 3))
        )

        robot.write_external_wrench_to_sim.reset_mock()
        event.reset(torch.tensor([0]))
        reset_call = robot.write_external_wrench_to_sim.call_args
        torch.testing.assert_close(reset_call.args[0], torch.zeros((1, 2, 3)))
        torch.testing.assert_close(reset_call.args[1], torch.zeros((1, 2, 3)))

    def test_mixed_disturbance_partitions_and_triggers_every_mode(self) -> None:
        robot = Mock()
        robot.num_bodies = 3
        robot.indexing = SimpleNamespace(body_ids=torch.tensor([1, 2, 3]))
        identity_quat = torch.zeros((20, 3, 4))
        identity_quat[..., 0] = 1.0
        robot.data = SimpleNamespace(
            root_link_vel_w=torch.zeros((20, 6)),
            body_link_quat_w=identity_quat,
        )
        asset_cfg = SceneEntityCfg("robot", body_ids=[0, 2])
        modes: dict[str, dict[str, object]] = {
            "sustained": {
                "enable": True,
                "ratio": 0.30,
                "linear_acceleration": (10.0, 10.0),
                "ramp_s": 0.04,
                "period_s": (0.02, 0.02),
            },
            "impulse": {
                "enable": True,
                "ratio": 0.30,
                "force": (20.0, 20.0),
                "torque": (2.0, 2.0),
                "duration_s": 0.1,
                "period_s": (0.02, 0.02),
            },
            "base_push": {
                "enable": True,
                "ratio": 0.25,
                "vel": (0.5, 0.5),
                "ang_vel": (0.7, 0.7),
                "period_s": (0.02, 0.02),
            },
            "none": {"enable": True, "ratio": 0.15},
        }
        cfg = EventTermCfg(
            func=MixedRecoveryDisturbance,
            mode="step",
            params={"asset_cfg": asset_cfg, "modes": modes},
        )
        env = cast(
            ManagerBasedRlEnv,
            SimpleNamespace(
                scene={"robot": robot},
                sim=SimpleNamespace(
                    model=SimpleNamespace(
                        body_mass=torch.tensor(
                            [[1.0, 2.0, 3.0, 4.0]]
                        ).expand(20, -1)
                    )
                ),
                num_envs=20,
                device="cpu",
                step_dt=0.02,
            ),
        )
        event = MixedRecoveryDisturbance(cfg, env)

        self.assertEqual(
            event.mode_counts(),
            {"sustained": 6, "impulse": 6, "base_push": 5, "none": 3},
        )
        event(env, None, asset_cfg=asset_cfg, modes=modes)

        expected_sustained = torch.tensor(
            [[[20.0, 20.0, 20.0], [40.0, 40.0, 40.0]]]
        ).expand(6, -1, -1)
        torch.testing.assert_close(event.force[0:6], expected_sustained)
        torch.testing.assert_close(event.force[6:12], torch.full((6, 2, 3), 20.0))
        torch.testing.assert_close(event.force[12:20], torch.zeros((8, 2, 3)))
        torch.testing.assert_close(event.torque[0:6], torch.zeros((6, 2, 3)))
        torch.testing.assert_close(event.impulse_left[6:12], torch.full((6,), 0.1))
        torch.testing.assert_close(
            event.trigger_counts[:17], torch.ones_like(event.trigger_counts[:17])
        )
        torch.testing.assert_close(
            event.trigger_counts[17:], torch.zeros_like(event.trigger_counts[17:])
        )
        base_push_call = robot.write_root_link_velocity_to_sim.call_args
        torch.testing.assert_close(
            base_push_call.args[0],
            torch.tensor([[0.5, 0.5, 0.0, 0.0, 0.0, 0.7]]).expand(5, -1),
        )
        torch.testing.assert_close(
            base_push_call.kwargs["env_ids"], torch.arange(12, 17)
        )

        first_wrench_call = robot.write_external_wrench_to_sim.call_args
        torch.testing.assert_close(
            first_wrench_call.args[0][0:6], torch.zeros((6, 2, 3))
        )
        torch.testing.assert_close(
            first_wrench_call.args[0][6:12], torch.full((6, 2, 3), 20.0)
        )

        event.time_left[0:6] = 1.0
        event(env, None, asset_cfg=asset_cfg, modes=modes)
        second_wrench_call = robot.write_external_wrench_to_sim.call_args
        torch.testing.assert_close(
            second_wrench_call.args[0][0:6], expected_sustained * 0.5
        )

        event.reset(torch.arange(20))
        torch.testing.assert_close(event.force, torch.zeros_like(event.force))
        torch.testing.assert_close(event.torque, torch.zeros_like(event.torque))
        torch.testing.assert_close(
            event.trigger_counts, torch.zeros_like(event.trigger_counts)
        )


if __name__ == "__main__":
    unittest.main()
