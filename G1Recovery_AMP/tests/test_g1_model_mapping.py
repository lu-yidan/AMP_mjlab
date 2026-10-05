"""Structural checks for the mjlab Unitree G1 model used by recovery."""

import unittest

import mujoco
from mjlab.entity.entity import Entity

from g1recovery_amp.g1_model import (
    G1_JOINT_NAMES,
    G1_RECOVERY_ACTION_SCALE,
    G1_RECOVERY_DEFAULT_JOINT_POS,
    G1_SOURCE_MOTION_JOINT_NAMES,
    G1_SOURCE_MOTION_TO_JOINT_ORDER,
    get_recovery_g1_robot_cfg,
    get_recovery_joint_position_action_cfg,
    validate_g1_model,
)


class TestG1ModelMapping(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.model = Entity(get_recovery_g1_robot_cfg()).compile()
        cls.mapping = validate_g1_model(cls.model)

    def test_factory_returns_independent_configs(self) -> None:
        first = get_recovery_g1_robot_cfg()
        second = get_recovery_g1_robot_cfg()
        self.assertIsNot(first, second)
        self.assertIsNot(first.articulation, second.articulation)
        self.assertIsNot(
            first.articulation.actuators[0], second.articulation.actuators[0]
        )

    def test_recovery_action_matches_source_semantics(self) -> None:
        action_cfg = get_recovery_joint_position_action_cfg()
        self.assertEqual(action_cfg.entity_name, "robot")
        self.assertEqual(action_cfg.actuator_names, (".*",))
        self.assertEqual(action_cfg.scale, G1_RECOVERY_ACTION_SCALE)
        self.assertEqual(action_cfg.scale, 0.25)
        self.assertTrue(action_cfg.use_default_offset)

    def test_recovery_action_configs_are_independent(self) -> None:
        first = get_recovery_joint_position_action_cfg()
        second = get_recovery_joint_position_action_cfg()
        self.assertIsNot(first, second)

    def test_recovery_default_pose_matches_source(self) -> None:
        cfg = get_recovery_g1_robot_cfg()
        self.assertEqual(cfg.init_state.pos, (0.0, 0.0, 0.8))
        self.assertEqual(cfg.init_state.joint_pos, G1_RECOVERY_DEFAULT_JOINT_POS)
        self.assertEqual(cfg.init_state.joint_vel, {".*": 0.0})

    def test_compiled_actuators_match_source_parameters(self) -> None:
        def expected(joint_name: str) -> tuple[float, float, float]:
            if "hip_pitch" in joint_name or "hip_yaw" in joint_name:
                return 100.0, 2.0, 88.0
            if joint_name == "waist_yaw_joint":
                return 200.0, 5.0, 88.0
            if "hip_roll" in joint_name:
                return 100.0, 2.0, 139.0
            if "knee" in joint_name:
                return 150.0, 4.0, 139.0
            if "ankle" in joint_name:
                return 40.0, 2.0, 25.0
            if joint_name in ("waist_roll_joint", "waist_pitch_joint"):
                return 40.0, 5.0, 25.0
            if "wrist_pitch" in joint_name or "wrist_yaw" in joint_name:
                return 40.0, 1.0, 5.0
            return 40.0, 1.0, 25.0

        for joint_name in G1_JOINT_NAMES:
            joint_id = self.model.joint(joint_name).id
            actuator_id = next(
                actuator_id
                for actuator_id in range(self.model.nu)
                if int(self.model.actuator_trnid[actuator_id, 0]) == joint_id
            )
            kp, kd, effort = expected(joint_name)
            self.assertAlmostEqual(
                -self.model.actuator_biasprm[actuator_id, 1], kp
            )
            self.assertAlmostEqual(
                -self.model.actuator_biasprm[actuator_id, 2], kd
            )
            self.assertAlmostEqual(
                self.model.actuator_forcerange[actuator_id, 1], effort
            )
            self.assertAlmostEqual(
                self.model.dof_armature[self.model.jnt_dofadr[joint_id]], 0.01
            )

    def test_free_base_and_29_dof_dimensions(self) -> None:
        self.assertEqual(self.model.nq, 7 + 29)
        self.assertEqual(self.model.nv, 6 + 29)
        self.assertEqual(self.model.nu, 29)

    def test_joint_order_matches_source_sdk_order(self) -> None:
        self.assertEqual(self.mapping.joint_names, G1_JOINT_NAMES)

    def test_source_motion_order_has_a_one_to_one_name_mapping(self) -> None:
        self.assertCountEqual(G1_SOURCE_MOTION_JOINT_NAMES, G1_JOINT_NAMES)
        self.assertEqual(
            sorted(G1_SOURCE_MOTION_TO_JOINT_ORDER),
            list(range(len(G1_JOINT_NAMES))),
        )
        self.assertNotEqual(
            G1_SOURCE_MOTION_TO_JOINT_ORDER,
            tuple(range(len(G1_JOINT_NAMES))),
        )

    def test_every_joint_has_exactly_one_actuator(self) -> None:
        self.assertCountEqual(self.mapping.actuator_joint_names, G1_JOINT_NAMES)
        self.assertEqual(
            sorted(self.mapping.joint_order_to_ctrl), list(range(len(G1_JOINT_NAMES)))
        )

    def test_ctrl_order_is_explicitly_mapped(self) -> None:
        # The compiled actuator order is grouped by motor type, not joint order.
        # This guards against accidentally treating qpos[7 + i] as ctrl[i].
        self.assertNotEqual(
            self.mapping.joint_order_to_ctrl, tuple(range(len(G1_JOINT_NAMES)))
        )

    def test_required_recovery_bodies_exist(self) -> None:
        # validate_g1_model performs the actual required-body assertion.
        self.assertIn("torso_link", self.mapping.body_names)

    def test_all_hinge_joint_limits_are_valid(self) -> None:
        for joint_id in range(self.model.njnt):
            if self.model.jnt_type[joint_id] == mujoco.mjtJoint.mjJNT_FREE:
                continue
            self.assertTrue(self.model.jnt_limited[joint_id])
            lower, upper = self.model.jnt_range[joint_id]
            self.assertLess(lower, upper, self.model.joint(joint_id).name)


if __name__ == "__main__":
    unittest.main()
