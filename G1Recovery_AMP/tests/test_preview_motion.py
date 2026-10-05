"""Tests for the standalone converted-motion preview tool."""

import unittest

from g1recovery_amp.motion_assets import GET_UP_DEFAULT_MOTION
from scripts.preview_motion import (
    load_motion_preview,
    next_frame_index,
    resolve_robot_model,
)


class TestPreviewMotion(unittest.TestCase):
    def test_loads_packaged_motion_summary(self) -> None:
        motion = load_motion_preview(GET_UP_DEFAULT_MOTION)

        self.assertEqual(motion.num_frames, 149)
        self.assertEqual(motion.fps, 50.0)
        self.assertAlmostEqual(motion.duration_s, 2.98)
        self.assertEqual(motion.joint_pos.shape, (149, 29))
        self.assertEqual(motion.root_quat_w.shape, (149, 4))
        self.assertEqual(motion.joint_count, 29)

    def test_robot_model_is_inferred_and_override_is_checked(self) -> None:
        self.assertEqual(resolve_robot_model(29), "recovery-29dof")
        with self.assertRaisesRegex(ValueError, "no preview robot supports"):
            resolve_robot_model(23)

    def test_frame_index_wraps_or_holds_at_end(self) -> None:
        self.assertEqual(next_frame_index(0, 3, loop=True), 1)
        self.assertEqual(next_frame_index(2, 3, loop=True), 0)
        self.assertEqual(next_frame_index(2, 3, loop=False), 2)

    def test_frame_index_rejects_invalid_input(self) -> None:
        with self.assertRaisesRegex(ValueError, "num_frames must be positive"):
            next_frame_index(0, 0, loop=True)
        with self.assertRaisesRegex(ValueError, "frame must be"):
            next_frame_index(3, 3, loop=True)


if __name__ == "__main__":
    unittest.main()
