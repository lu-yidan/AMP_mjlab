"""Checks for the frozen A6 reset asset and flat cohort sampler."""

import unittest

import numpy as np

from g1recovery_amp.a6_reset_assets import (
    A6_FLAT_STRATUM,
    A6_MULTITERRAIN_TRAIN_BANK,
    A6_MULTITERRAIN_TRAIN_SHA256,
    A6_NATURAL_CURRICULUM_TRAIN_BANK,
    A6_NATURAL_CURRICULUM_TRAIN_SHA256,
    file_sha256,
    load_a6_multiterrain_bank,
    load_a6_natural_curriculum_bank,
    natural_stage_sampling_pool,
)
from g1recovery_amp.tasks.recovery.mdp import (
    a6_curriculum_progress,
    balanced_flat_curriculum_groups,
    balanced_flat_reset_groups,
    balanced_three_scene_curriculum_groups,
    balanced_three_scene_reset_groups,
)


class TestA6ResetBank(unittest.TestCase):
    def test_curriculum_progress_uses_adaptation_local_age(self) -> None:
        self.assertEqual(a6_curriculum_progress(480_000, 480_000, 100_000), 0.0)
        self.assertEqual(a6_curriculum_progress(530_000, 480_000, 100_000), 0.5)
        self.assertEqual(a6_curriculum_progress(600_000, 480_000, 100_000), 1.0)
        self.assertEqual(a6_curriculum_progress(470_000, 480_000, 100_000), 0.0)

    def test_frozen_asset_hash_and_schema(self) -> None:
        self.assertEqual(
            file_sha256(A6_MULTITERRAIN_TRAIN_BANK),
            A6_MULTITERRAIN_TRAIN_SHA256,
        )
        bank = load_a6_multiterrain_bank()
        self.assertEqual(bank.qpos.shape, (4096, 36))
        self.assertEqual(bank.direction.shape, (4096,))
        self.assertEqual(bank.stratum.shape, (4096,))
        self.assertEqual(bank.source.shape, (4096,))
        np.testing.assert_allclose(
            np.linalg.norm(bank.qpos[:, 3:7], axis=1),
            1.0,
            rtol=0.0,
            atol=1.0e-5,
        )

    def test_flat_bank_has_every_direction_and_source_pool(self) -> None:
        bank = load_a6_multiterrain_bank()
        for direction in range(4):
            for source in range(2):
                count = np.count_nonzero(
                    (bank.stratum == A6_FLAT_STRATUM)
                    & (bank.direction == direction)
                    & (bank.source == source)
                )
                self.assertEqual(count, 64)

    def test_natural_curriculum_asset_hash_schema_and_stage_counts(self) -> None:
        self.assertEqual(
            file_sha256(A6_NATURAL_CURRICULUM_TRAIN_BANK),
            A6_NATURAL_CURRICULUM_TRAIN_SHA256,
        )
        bank = load_a6_natural_curriculum_bank()
        self.assertEqual(bank.qpos.shape, (3046, 36))
        unique, counts = np.unique(bank.stages, return_counts=True)
        self.assertEqual(
            dict(zip(unique.tolist(), counts.tolist(), strict=True)),
            {"late": 295, "low": 1306, "middle": 1445},
        )

    def test_natural_middle_and_late_weights_balance_source_clips(self) -> None:
        bank = load_a6_natural_curriculum_bank()
        base_clips = np.asarray(
            [str(name).replace("__mirror", "") for name in bank.clips]
        )
        for stage in ("middle", "late"):
            rows, weights = natural_stage_sampling_pool(bank, stage)
            self.assertAlmostEqual(float(weights.sum()), 1.0)
            clips = np.unique(base_clips[rows])
            clip_mass = np.asarray(
                [weights[base_clips[rows] == clip].sum() for clip in clips]
            )
            np.testing.assert_allclose(
                clip_mass,
                np.full(len(clips), 1.0 / len(clips)),
                rtol=0.0,
                atol=1.0e-12,
            )

    def test_fixed_cohorts_balance_directions_and_sources(self) -> None:
        direction, source = balanced_flat_reset_groups(4096)
        np.testing.assert_array_equal(np.bincount(direction), np.full(4, 1024))
        for direction_id in range(4):
            group_source = source[direction == direction_id]
            np.testing.assert_array_equal(
                np.bincount(group_source, minlength=2),
                np.array([768, 256]),
            )

    def test_flat_curriculum_matches_historical_2048_environment_quotas(self) -> None:
        stage, direction, source = balanced_flat_curriculum_groups(2048)
        np.testing.assert_array_equal(
            np.bincount(stage, minlength=3), np.array([820, 820, 408])
        )
        self.assertEqual(int(np.count_nonzero(source == 1)), 204)
        self.assertTrue(np.all(source[stage != 0] == 0))
        self.assertTrue(np.all(direction[stage != 0] == -1))
        for direction_id in range(4):
            low = (stage == 0) & (direction == direction_id)
            self.assertEqual(int(low.sum()), 205)
            np.testing.assert_array_equal(
                np.bincount(source[low], minlength=2), np.array([154, 51])
            )

    def test_three_scene_curriculum_matches_historical_4096_quotas(self) -> None:
        scene, stage, direction, source = balanced_three_scene_curriculum_groups(4096)
        np.testing.assert_array_equal(
            np.bincount(scene, minlength=3), np.array([2048, 1024, 1024])
        )
        np.testing.assert_array_equal(
            np.bincount(stage, minlength=3), np.array([2868, 820, 408])
        )
        np.testing.assert_array_equal(
            np.bincount(source, minlength=2), np.array([3380, 716])
        )
        self.assertTrue(np.all(stage[scene != 0] == 0))
        for scene_id in range(3):
            low = (scene == scene_id) & (stage == 0)
            counts = np.bincount(direction[low], minlength=4)
            self.assertLessEqual(int(counts.max() - counts.min()), 1)

    def test_formal_2048_evaluation_groups_match_exact_requested_quotas(self) -> None:
        scene, stage, direction, source = balanced_three_scene_reset_groups(2048)

        np.testing.assert_array_equal(
            np.bincount(scene, minlength=3), np.array([1024, 512, 512])
        )
        self.assertTrue(np.all(stage == 0))
        for scene_id, per_direction in ((0, 256), (1, 128), (2, 128)):
            scene_mask = scene == scene_id
            np.testing.assert_array_equal(
                np.bincount(direction[scene_mask], minlength=4),
                np.full(4, per_direction),
            )
            for direction_id in range(4):
                cohort = scene_mask & (direction == direction_id)
                np.testing.assert_array_equal(
                    np.bincount(source[cohort], minlength=2),
                    np.array([3 * per_direction // 4, per_direction // 4]),
                )

    def test_three_scene_curriculum_supports_constrained_70_30_mix(self) -> None:
        scene, stage, direction, _source = balanced_three_scene_curriculum_groups(
            512, scene_weights=(0.0, 0.70, 0.30)
        )
        np.testing.assert_array_equal(
            np.bincount(scene, minlength=3), np.array([0, 358, 154])
        )
        self.assertTrue(np.all(stage == 0))
        for scene_id in (1, 2):
            counts = np.bincount(direction[scene == scene_id], minlength=4)
            self.assertLessEqual(int(counts.max() - counts.min()), 1)

    def test_three_scene_curriculum_rejects_invalid_weights(self) -> None:
        for weights in ((0.0, 0.0, 0.0), (-1.0, 1.0, 1.0)):
            with self.assertRaisesRegex(ValueError, "scene_weights"):
                balanced_three_scene_curriculum_groups(32, scene_weights=weights)


if __name__ == "__main__":
    unittest.main()
