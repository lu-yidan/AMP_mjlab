"""Tests for translating source recovery clips into mjlab motion samples."""

import tempfile
import unittest
from pathlib import Path

import joblib
import numpy as np
import torch
from mjlab.tasks.tracking.mdp.commands import MotionLoader

from g1recovery_amp.g1_model import (
    G1_JOINT_NAMES,
    G1_SOURCE_MOTION_TO_JOINT_ORDER,
)
from g1recovery_amp.motion_assets import (
    ACTIVE_GET_UP_MOTION_NAMES,
    CONVERTED_GET_UP_MOTIONS,
    GET_UP_TRAINING_MOTION_WEIGHTS,
)
from g1recovery_amp.motion_conversion import (
    load_source_motion,
    resample_motion,
)

EXPECTED_CONVERTED_FRAMES = {
    "fallAndGetUp1_subject1_1060_1150": 149,
    "fallAndGetUp1_subject1_1400_1480": 132,
    "fallAndGetUp1_subject1_2100_2200": 165,
    "fallAndGetUp1_subject5_2500_2600": 165,
    "fallAndGetUp2_subject2_850_1050": 332,
    "fallAndGetUp6_subject1_530_600": 115,
    "fallAndGetUp6_subject1_650_700": 82,
    "fallAndGetUp6_subject1_1630_1690": 99,
}


class TestMotionConversion(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.motion_path = Path(self.temp_dir.name) / "motion.pkl"

    def _write_motion(self, **overrides: object) -> None:
        frames = 4
        root_rot = np.zeros((frames, 4), dtype=np.float64)
        root_rot[:, 0] = 1.0
        data = {
            "fps": 10,
            "loop_mode": 0,
            "root_pos": np.column_stack(
                (np.arange(frames) / 10.0, np.zeros((frames, 2)))
            ),
            "root_rot": root_rot,
            "dof_pos": np.zeros((frames, len(G1_JOINT_NAMES))),
            "key_body_pos": np.zeros((frames, 6, 3)),
        }
        data.update(overrides)
        joblib.dump(data, self.motion_path)

    def test_loads_source_conventions(self) -> None:
        self._write_motion()
        motion = load_source_motion(self.motion_path)

        self.assertEqual(motion.fps, 10.0)
        self.assertEqual(motion.loop_mode, 0)
        self.assertEqual(motion.num_frames, 4)
        self.assertAlmostEqual(motion.duration, 0.3)
        self.assertEqual(tuple(motion.joint_pos.shape), (4, 29))
        torch.testing.assert_close(
            motion.root_quat_wxyz[:, 0], torch.ones(4)
        )

    def test_rejects_wrong_joint_count(self) -> None:
        self._write_motion(dof_pos=np.zeros((4, 28)))

        with self.assertRaisesRegex(ValueError, r"expected \(4, 29\)"):
            load_source_motion(self.motion_path)

    def test_reorders_source_columns_into_mujoco_joint_order(self) -> None:
        source_columns = np.broadcast_to(np.arange(29), (4, 29)).copy()
        self._write_motion(dof_pos=source_columns)

        motion = load_source_motion(self.motion_path)

        torch.testing.assert_close(
            motion.joint_pos[0],
            torch.tensor(G1_SOURCE_MOTION_TO_JOINT_ORDER, dtype=torch.float32),
        )

    def test_resamples_once_per_policy_step(self) -> None:
        self._write_motion()
        motion = resample_motion(load_source_motion(self.motion_path), 20.0)

        # torch.arange(0, 0.3, 0.05) produces six policy-time samples.
        self.assertEqual(motion.num_frames, 6)
        self.assertEqual(tuple(motion.joint_pos.shape), (6, 29))
        torch.testing.assert_close(
            motion.root_pos[:, 0],
            torch.tensor([0.00, 0.05, 0.10, 0.15, 0.20, 0.25]),
        )
        torch.testing.assert_close(
            motion.root_lin_vel_w[:, 0], torch.ones(6)
        )
        torch.testing.assert_close(
            motion.root_ang_vel_w, torch.zeros((6, 3))
        )

    def test_normalizes_source_quaternions(self) -> None:
        root_rot = np.zeros((4, 4))
        root_rot[:, 0] = 2.0
        self._write_motion(root_rot=root_rot)

        motion = load_source_motion(self.motion_path)

        torch.testing.assert_close(
            torch.linalg.vector_norm(motion.root_quat_wxyz, dim=-1),
            torch.ones(4),
        )


class TestPackagedMotionAsset(unittest.TestCase):
    def _check_asset(self, path: Path, expected_frames: int) -> None:
        self.assertTrue(path.is_file())
        with np.load(path) as data:
            self.assertEqual(
                set(data.files),
                {
                    "fps",
                    "joint_pos",
                    "joint_vel",
                    "body_pos_w",
                    "body_quat_w",
                    "body_lin_vel_w",
                    "body_ang_vel_w",
                },
            )
            self.assertEqual(data["fps"].tolist(), [50.0])
            self.assertEqual(data["joint_pos"].shape, (expected_frames, 29))
            self.assertEqual(data["joint_vel"].shape, (expected_frames, 29))
            self.assertEqual(data["body_pos_w"].shape, (expected_frames, 30, 3))
            self.assertEqual(data["body_quat_w"].shape, (expected_frames, 30, 4))
            self.assertTrue(all(np.isfinite(data[key]).all() for key in data.files))

    def test_assets_match_mjlab_motion_contract(self) -> None:
        for name, expected_frames in EXPECTED_CONVERTED_FRAMES.items():
            with self.subTest(name=name):
                self._check_asset(
                    CONVERTED_GET_UP_MOTIONS[name], expected_frames
                )

    def test_mjlab_motion_loader_reads_asset(self) -> None:
        for name, expected_frames in EXPECTED_CONVERTED_FRAMES.items():
            path = CONVERTED_GET_UP_MOTIONS[name]
            with self.subTest(path=path.name):
                loader = MotionLoader(str(path), body_indexes=torch.arange(30))
                self.assertEqual(loader.time_step_total, expected_frames)
                self.assertEqual(
                    tuple(loader.joint_pos.shape), (expected_frames, 29)
                )
                self.assertEqual(
                    tuple(loader.body_pos_w.shape), (expected_frames, 30, 3)
                )

    def test_training_manifest_contains_original_eight_equal_weight_motions(self) -> None:
        self.assertEqual(len(GET_UP_TRAINING_MOTION_WEIGHTS), 8)
        self.assertEqual(len(ACTIVE_GET_UP_MOTION_NAMES), 8)
        self.assertEqual(set(GET_UP_TRAINING_MOTION_WEIGHTS.values()), {1.0})
        self.assertEqual(
            set(CONVERTED_GET_UP_MOTIONS),
            set(EXPECTED_CONVERTED_FRAMES),
        )
        self.assertTrue(
            set(CONVERTED_GET_UP_MOTIONS).issubset(ACTIVE_GET_UP_MOTION_NAMES)
        )
        self.assertTrue(all(path.is_file() for path in CONVERTED_GET_UP_MOTIONS.values()))


if __name__ == "__main__":
    unittest.main()
