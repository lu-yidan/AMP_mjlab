"""The 738D body-only mirror must preserve the HoST observation contract."""

import torch

from g1recovery_host.proprio_symmetry import (
    JOINT_MIRROR_IDS, JOINT_MIRROR_SIGNS,
    mirror_host29_action, mirror_host29_actor_obs,
)


def test_mirror_is_an_involution_for_actions_and_six_frame_history():
    torch.manual_seed(3)
    action = torch.randn(7, 29)
    obs = torch.randn(7, 738)
    torch.testing.assert_close(mirror_host29_action(mirror_host29_action(action)), action)
    torch.testing.assert_close(mirror_host29_actor_obs(mirror_host29_actor_obs(obs)), obs)


def test_mirror_joint_axes_and_history_blocks():
    # Left hip roll maps to the negative right hip roll; pitch stays positive.
    assert JOINT_MIRROR_IDS[1] == 7
    assert JOINT_MIRROR_SIGNS[1] == -1
    assert JOINT_MIRROR_IDS[0] == 6
    assert JOINT_MIRROR_SIGNS[0] == 1
    host = torch.zeros(1, 6, 94)
    torque = torch.zeros(1, 6, 29)
    host[:, :, 1] = 0.4  # axial angular velocity Y keeps sign.
    host[:, :, 4] = 0.8  # polar gravity Y changes sign.
    host[:, :, 6 + 1] = 0.3
    torque[:, :, 1] = 0.2
    obs = torch.cat((host.reshape(1, 564), torque.reshape(1, 174)), dim=1)
    mirrored = mirror_host29_actor_obs(obs)
    mirrored_host = mirrored[:, :564].reshape(1, 6, 94)
    mirrored_torque = mirrored[:, 564:].reshape(1, 6, 29)
    torch.testing.assert_close(mirrored_host[:, :, 1], host[:, :, 1])
    torch.testing.assert_close(mirrored_host[:, :, 4], -host[:, :, 4])
    torch.testing.assert_close(mirrored_host[:, :, 6 + 7], -host[:, :, 6 + 1])
    torch.testing.assert_close(mirrored_torque[:, :, 7], -torque[:, :, 1])
