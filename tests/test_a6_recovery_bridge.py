"""Successful early recovery motions must earn progress after clearance."""

from types import SimpleNamespace

import torch

from g1recovery_host.a6_reward import _guided_recovery_reward
from g1recovery_host.model import HOST29_JOINT_NAMES


class _Scene(dict):
    def __init__(self, robot, count):
        super().__init__(robot=SimpleNamespace(data=robot))
        self.env_origins = torch.zeros(count, 3)


def test_bridge_reward_follows_prone_getup_without_paying_under_board():
    height = torch.tensor([0.108, 0.118, 0.178, 0.242, 0.272, 0.276, 0.524])
    up = torch.tensor([-0.065, -0.379, -0.544, -0.628, -0.661, 0.832, 0.940])
    knees = torch.tensor([
        [0.32, 0.39], [1.01, 1.03], [1.64, 1.56], [2.24, 2.10],
        [2.47, 2.65], [2.48, 2.61], [1.63, 1.58],
    ])
    ids = [HOST29_JOINT_NAMES.index(f"{side}_knee_joint") for side in ("left", "right")]
    joints = torch.zeros(len(height), len(HOST29_JOINT_NAMES))
    joints[:, ids] = knees
    robot = SimpleNamespace(
        root_link_pos_w=torch.stack((torch.zeros_like(height), torch.zeros_like(height), height), -1),
        projected_gravity_b=torch.stack((torch.zeros_like(up), torch.zeros_like(up), -up), -1),
        joint_pos=joints,
    )
    env = SimpleNamespace(
        scene=_Scene(robot, len(height)),
        _a6_scene=torch.ones(len(height), dtype=torch.long),
        _a6_direction=torch.ones(len(height), dtype=torch.long),
        _a6_escaped=torch.ones(len(height), dtype=torch.bool),
        _a6_invalid=torch.zeros(len(height), dtype=torch.bool),
        _a6_knee_ids=ids,
        _a6_recovery_bridge_v2=True,
        _a6_strong_recovery_v1=True,
    )
    scores = _guided_recovery_reward(env)
    assert torch.all(scores[1:] > scores[:-1])
    env._a6_escaped[1] = False
    assert _guided_recovery_reward(env)[1] == 0
    env._a6_direction[0] = 0
    supine_bridge = _guided_recovery_reward(env, include_supine=True)[0]
    env._a6_recovery_bridge_v2 = False
    supine_original = _guided_recovery_reward(env, include_supine=True)[0]
    assert torch.isclose(supine_bridge, supine_original)


def test_scene_phase_reward_starts_flat_immediately_and_waits_for_board_escape():
    heights = torch.tensor([0.15, 0.15, 0.15, 0.15, 0.35])
    upright = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.8])
    robot = SimpleNamespace(
        root_link_pos_w=torch.stack((heights * 0, heights * 0, heights), -1),
        projected_gravity_b=torch.stack(
            (upright * 0, upright * 0, -upright), -1
        ),
        joint_pos=torch.zeros(5, len(HOST29_JOINT_NAMES)),
    )
    env = SimpleNamespace(
        scene=_Scene(robot, 5),
        _a6_scene=torch.tensor([0, 2, 2, 1, 0]),
        _a6_direction=torch.zeros(5, dtype=torch.long),
        _a6_escaped=torch.tensor([False, False, True, True, False]),
        _a6_invalid=torch.zeros(5, dtype=torch.bool),
        _a6_recovery_bridge_v2=False,
        _a6_strong_recovery_v1=True,
    )
    score = _guided_recovery_reward(env, include_supine=True, include_flat_free=True)
    assert score[0] > 0
    assert score[1] == 0
    assert score[2] > 0
    assert score[3] > 0
    assert score[4] > score[0]
    env._a6_invalid[2] = True
    assert _guided_recovery_reward(
        env, include_supine=True, include_flat_free=True
    )[2] == 0
