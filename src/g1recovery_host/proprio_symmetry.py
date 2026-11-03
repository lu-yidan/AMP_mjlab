"""Left/right reflection of the 29-DoF HoST proprioception contract."""

from __future__ import annotations

import torch

from .model import HOST29_JOINT_NAMES


def _joint_reflection():
    names = HOST29_JOINT_NAMES
    indices = []
    signs = []
    for name in names:
        if name.startswith("left_"):
            partner = "right_" + name[5:]
        elif name.startswith("right_"):
            partner = "left_" + name[6:]
        else:
            partner = name
        indices.append(names.index(partner))
        # The MJCF axes for roll and yaw are body X and Z. They change sign
        # under a left/right reflection; pitch, knee, and elbow axes are Y.
        signs.append(-1.0 if ("_roll_" in name or "_yaw_" in name) else 1.0)
    return tuple(indices), tuple(signs)


JOINT_MIRROR_IDS, JOINT_MIRROR_SIGNS = _joint_reflection()
HOST_FRAME_DIM = 94
TORQUE_FRAME_DIM = 29
HISTORY = 6


def mirror_host29_action(action: torch.Tensor) -> torch.Tensor:
    """Reflect a batch of 29 joint commands across the sagittal plane."""
    if action.ndim != 2 or action.shape[1] != 29:
        raise ValueError("mirror requires 29-DoF action rows")
    indices = torch.tensor(JOINT_MIRROR_IDS, device=action.device)
    signs = action.new_tensor(JOINT_MIRROR_SIGNS)
    return action.index_select(-1, indices) * signs


def mirror_host29_actor_obs(actor_obs: torch.Tensor) -> torch.Tensor:
    """Reflect 6×94 HoST history followed by 6×29 torque history."""
    if actor_obs.ndim != 2 or actor_obs.shape[1] != HISTORY * (HOST_FRAME_DIM + TORQUE_FRAME_DIM):
        raise ValueError("mirror requires 738D proprio actor observations")
    host = actor_obs[:, :HISTORY * HOST_FRAME_DIM].reshape(-1, HISTORY, HOST_FRAME_DIM)
    torque = actor_obs[:, HISTORY * HOST_FRAME_DIM:].reshape(-1, HISTORY, TORQUE_FRAME_DIM)
    reflected_host = host.clone()
    # Angular velocity is an axial vector; gravity is a polar vector.
    reflected_host[:, :, 0:3] *= host.new_tensor((-1.0, 1.0, -1.0))
    reflected_host[:, :, 3:6] *= host.new_tensor((1.0, -1.0, 1.0))
    indices = torch.tensor(JOINT_MIRROR_IDS, device=actor_obs.device)
    signs = actor_obs.new_tensor(JOINT_MIRROR_SIGNS)
    for start in (6, 35, 64):
        reflected_host[:, :, start:start + 29] = (
            host[:, :, start:start + 29].index_select(-1, indices) * signs
        )
    reflected_torque = torque.index_select(-1, indices) * signs
    return torch.cat(
        (reflected_host.reshape(-1, HISTORY * HOST_FRAME_DIM),
         reflected_torque.reshape(-1, HISTORY * TORQUE_FRAME_DIM)),
        dim=-1,
    )
