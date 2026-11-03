"""The free-board objective rewards board movement rather than robot crawl."""

import torch

from g1recovery_host.a6_reward import (
    free_plate_anchor_cost,
    free_plate_motion_signal,
)


def test_free_board_progress_requires_new_board_motion_and_penalizes_excess_crawl():
    shift = torch.tensor([0.10, 0.10, 0.00, 0.10])
    previous = torch.tensor([0.05, 0.05, 0.00, 0.05])
    root = torch.tensor([0.00, 0.80, 0.80, 0.00])
    eligible = torch.tensor([True, True, True, False])
    best, progress, crawl = free_plate_motion_signal(
        shift, previous, root, eligible
    )
    assert best.tolist() == shift.tolist()
    assert progress[0] > progress[1] > 0
    assert progress[2] == progress[3] == 0
    assert crawl[0] == crawl[3] == 0
    assert crawl[1] > 0 and crawl[2] > 0


def test_anchor_spares_successful_near_origin_behavior():
    cost = free_plate_anchor_cost(
        torch.tensor([0.15, 0.30, 1.20, 1.20]),
        torch.tensor([True, True, True, False]),
    )
    assert cost[0] == cost[3] == 0
    assert 0 < cost[1] < cost[2] == 1
