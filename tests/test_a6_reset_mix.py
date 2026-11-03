"""A6 training quotas cover the directions that evaluation exercises."""

import numpy as np
import pytest

from g1recovery_host.a6_reset import focused_guided_layout


@pytest.mark.parametrize("num_envs", (32, 4096))
def test_diverse_reset_covers_guided_sides_and_free_prone(num_envs):
    rows = focused_guided_layout(num_envs, postescape_prone=True, mix="diverse_v3")
    unit = num_envs // 32
    expected = (
        (unit, unit, unit, unit),
        (6 * unit, 6 * unit, 3 * unit, 3 * unit),
        (2 * unit, 4 * unit, 2 * unit, 2 * unit),
    )
    assert rows.shape == (num_envs, 3)
    for scene, directions in enumerate(expected):
        assert tuple(
            np.count_nonzero((rows[:, 0] == scene) & (rows[:, 1] == direction))
            for direction in range(4)
        ) == directions
    replay = rows[:, 2] == 3
    assert replay.sum() == (6 * unit + 3) // 4
    assert np.all((rows[replay, 0] == 1) & (rows[replay, 1] == 1))
    assert np.all(rows[rows[:, 0] == 2, 2] == 0)


def test_original_mix_remains_unchanged():
    rows = focused_guided_layout(4096, postescape_prone=True)
    assert np.count_nonzero(rows[:, 0] == 0) == 1024
    assert np.count_nonzero(rows[:, 0] == 1) == 2560
    assert np.count_nonzero(rows[:, 0] == 2) == 512
    assert np.count_nonzero((rows[:, 0] == 1) & (rows[:, 1] >= 2)) == 0
    assert np.count_nonzero(rows[:, 2] == 3) == 320


@pytest.mark.parametrize("num_envs", (32, 4096))
def test_prone_compact_mix_favors_flat_prone_and_keeps_board_prone(num_envs):
    rows = focused_guided_layout(
        num_envs, postescape_prone=True,
        mix="prone_compact_v1", postescape_stride=2,
    )
    unit = num_envs // 32
    assert rows.shape == (num_envs, 3)
    assert tuple(np.count_nonzero(rows[:, 0] == scene) for scene in range(3)) == (
        16 * unit, 8 * unit, 8 * unit,
    )
    assert np.all(rows[rows[:, 0] == 0, 1] == 1)
    for scene in (1, 2):
        assert tuple(
            np.count_nonzero((rows[:, 0] == scene) & (rows[:, 1] == direction))
            for direction in range(4)
        ) == (unit, 5 * unit, unit, unit)
    replay = rows[:, 2] == 3
    assert replay.sum() == (5 * unit + 1) // 2
    assert np.all((rows[replay, 0] == 1) & (rows[replay, 1] == 1))


@pytest.mark.parametrize("num_envs", (32, 4096))
def test_flat_prone_specialist_mix_isolates_flat_starts(num_envs):
    rows = focused_guided_layout(
        num_envs, postescape_prone=True,
        mix="flat_prone_specialist_v1", postescape_stride=2,
    )
    unit = num_envs // 32
    assert np.count_nonzero((rows[:, 0] == 0) & (rows[:, 1] == 1)) == 24 * unit
    for scene in (1, 2):
        assert tuple(
            np.count_nonzero((rows[:, 0] == scene) & (rows[:, 1] == direction))
            for direction in range(4)
        ) == (unit,) * 4
    assert np.count_nonzero(rows[:, 2] == 3) == (unit + 1) // 2


def test_half_guided_prone_replays_recovery_states():
    rows = focused_guided_layout(
        4096, postescape_prone=True, mix="diverse_v3", postescape_stride=2
    )
    prone = (rows[:, 0] == 1) & (rows[:, 1] == 1)
    assert np.count_nonzero(prone) == 768
    assert np.count_nonzero(prone & (rows[:, 2] == 3)) == 384
    assert np.count_nonzero(prone & (rows[:, 2] != 3)) == 384


@pytest.mark.parametrize("num_envs", (32, 4096))
def test_recovery_mix_rehearses_flat_and_free_without_dropping_guided(num_envs):
    rows = focused_guided_layout(
        num_envs, postescape_prone=True,
        mix="diverse_v4_recovery", postescape_stride=2,
    )
    unit = num_envs // 32
    expected = (
        (2 * unit,) * 4,
        (5 * unit, 5 * unit, 2 * unit, 2 * unit),
        (3 * unit, 3 * unit, 2 * unit, 2 * unit),
    )
    for scene, directions in enumerate(expected):
        assert tuple(
            np.count_nonzero((rows[:, 0] == scene) & (rows[:, 1] == direction))
            for direction in range(4)
        ) == directions
    replay = rows[:, 2] == 3
    assert replay.sum() == (5 * unit + 1) // 2
    assert np.all((rows[replay, 0] == 1) & (rows[replay, 1] == 1))
    if num_envs == 4096:
        flat_supine = (rows[:, 0] == 0) & (rows[:, 1] == 0)
        assert np.count_nonzero(flat_supine & (rows[:, 2] == 0)) == 154
        assert np.count_nonzero(flat_supine & (rows[:, 2] == 1)) == 76
        assert np.count_nonzero(flat_supine & (rows[:, 2] == 2)) == 26


def test_free_specialist_mix_keeps_all_directions_and_replay():
    rows = focused_guided_layout(
        4096, postescape_prone=True,
        mix="free_specialist_v1", postescape_stride=2,
    )
    assert tuple(np.count_nonzero(rows[:, 0] == scene) for scene in range(3)) == (
        512, 512, 3072,
    )
    assert all(
        np.count_nonzero((rows[:, 0] == 2) & (rows[:, 1] == direction)) == 768
        for direction in range(4)
    )
    replay = rows[:, 2] == 3
    assert replay.sum() == 64
    assert np.all((rows[replay, 0] == 1) & (rows[replay, 1] == 1))


def test_guided_prone_specialist_retains_all_scenes_and_prone_replay():
    rows = focused_guided_layout(
        4096, postescape_prone=True,
        mix="guided_prone_specialist_v1", postescape_stride=2,
    )
    assert tuple(np.count_nonzero(rows[:, 0] == scene) for scene in range(3)) == (
        512, 2560, 1024,
    )
    assert tuple(
        np.count_nonzero((rows[:, 0] == 1) & (rows[:, 1] == direction))
        for direction in range(4)
    ) == (128, 2176, 128, 128)
    assert all(
        np.count_nonzero((rows[:, 0] == 2) & (rows[:, 1] == direction)) == 256
        for direction in range(4)
    )
    replay = rows[:, 2] == 3
    assert replay.sum() == 1088
    assert np.all((rows[replay, 0] == 1) & (rows[replay, 1] == 1))


def test_side_specialist_mixes_fixed_and_movable_board_sides():
    rows = focused_guided_layout(
        4096, postescape_prone=True,
        mix="side_specialist_v1", postescape_stride=2,
    )
    assert tuple(np.count_nonzero(rows[:, 0] == scene) for scene in range(3)) == (
        512, 2048, 1536,
    )
    assert tuple(
        np.count_nonzero((rows[:, 0] == 1) & (rows[:, 1] == direction))
        for direction in range(4)
    ) == (256, 256, 768, 768)
    assert tuple(
        np.count_nonzero((rows[:, 0] == 2) & (rows[:, 1] == direction))
        for direction in range(4)
    ) == (128, 128, 640, 640)
    replay = rows[:, 2] == 3
    assert replay.sum() == 128
    assert np.all((rows[replay, 0] == 1) & (rows[replay, 1] == 1))


def test_guided_side_specialist_keeps_fixed_board_side_majority():
    rows = focused_guided_layout(
        4096, postescape_prone=True,
        mix="guided_side_specialist_v1", postescape_stride=2,
    )
    assert tuple(np.count_nonzero(rows[:, 0] == scene) for scene in range(3)) == (
        512, 3072, 512,
    )
    assert tuple(
        np.count_nonzero((rows[:, 0] == 1) & (rows[:, 1] == direction))
        for direction in range(4)
    ) == (256, 256, 1280, 1280)
    assert all(
        np.count_nonzero((rows[:, 0] == 2) & (rows[:, 1] == direction)) == 128
        for direction in range(4)
    )


def test_guided_left_side_specialist_reallocates_guided_side_quota():
    rows = focused_guided_layout(
        4096, postescape_prone=True,
        mix="guided_left_side_specialist_v1", postescape_stride=2,
    )
    assert tuple(np.count_nonzero(rows[:, 0] == scene) for scene in range(3)) == (
        512, 3072, 512,
    )
    assert tuple(
        np.count_nonzero((rows[:, 0] == 1) & (rows[:, 1] == direction))
        for direction in range(4)
    ) == (256, 256, 2304, 256)
