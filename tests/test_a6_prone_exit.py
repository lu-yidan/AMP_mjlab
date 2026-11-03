"""A trailing hand/foot may skim an edge only after the body core is clear."""

import torch

from g1recovery_host.a6_geometry import (
    PRONE_EXIT_CORE_CLEARANCE_M,
    PRONE_EXIT_TRAILING_OVERLAP_M,
    footprint,
    prone_exit_geometry,
    prone_flexible_clearance,
)


def test_prone_exit_geometry_accepts_only_shallow_trailing_overlap():
    plate_pos = torch.zeros(1, 3)
    plate_ext = torch.tensor([[0.45, 0.32, 0.035]])
    ext = torch.tensor([[[0.05, 0.05, 0.05], [0.05, 0.05, 0.05]]])
    core_ids = torch.tensor([0])
    pos = torch.tensor([[[0.60, 0.0, 0.0], [0.49, 0.0, 0.0]]])

    assert footprint(pos, ext, plate_pos, plate_ext)[2].item() == 0
    core_clearance, tip_overlap = prone_exit_geometry(
        pos, ext, plate_pos, plate_ext, core_ids
    )
    assert core_clearance.item() >= PRONE_EXIT_CORE_CLEARANCE_M
    assert tip_overlap.item() <= PRONE_EXIT_TRAILING_OVERLAP_M

    deep_tip = pos.clone()
    deep_tip[0, 1, 0] = 0.40
    _, deep_overlap = prone_exit_geometry(
        deep_tip, ext, plate_pos, plate_ext, core_ids
    )
    assert deep_overlap.item() > PRONE_EXIT_TRAILING_OVERLAP_M

    trapped_core = pos.clone()
    trapped_core[0, 0, 0] = 0.45
    trapped_clearance, _ = prone_exit_geometry(
        trapped_core, ext, plate_pos, plate_ext, core_ids
    )
    assert trapped_clearance.item() < PRONE_EXIT_CORE_CLEARANCE_M

    lifted_plate = plate_pos.clone()
    lifted_plate[:, 2] = 0.10
    assert prone_exit_geometry(pos, ext, lifted_plate, plate_ext, core_ids) == (
        core_clearance, tip_overlap
    )


def test_prone_exit_never_counts_plate_lifting_as_clearance():
    core_clearance = torch.tensor([0.04, 0.04, 0.01])
    trailing_overlap = torch.tensor([0.01, 0.01, 0.01])
    plate_lift = torch.tensor([0.0, 0.04, 0.0])
    strict_geometry = torch.tensor([False, True, False])
    ready = prone_flexible_clearance(
        core_clearance, trailing_overlap, plate_lift, strict_geometry
    )
    assert ready.tolist() == [True, False, False]
