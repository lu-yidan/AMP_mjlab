"""Isaac Lab 3.x adapter (Newton / PhysX backends): a manager-based ``ManagerBasedRLEnv`` as a ``TrackingTask``.

    from supermimic.wrappers.isaaclab3 import IsaacLab3TrackingTask
    task = IsaacLab3TrackingTask(env, robot_model="path/to/robot.xml")   # or a URDF
    runner = SuperMimicRunner(task, train_cfg, log_dir, device)

Same machinery as the Isaac Lab 2.x adapter (``isaaclab.py``), with Isaac Lab 3's two API changes:
  * robot data fields are ``ProxyArray`` s read through ``.torch``; imagined fields are served the same way
    (``.torch`` keeps the autograd graph; a term reading ``.warp`` cannot be differentiated and raises);
  * quaternions are xyzw (imagination converts at the boundary);
  * no DR-context readers yet (the 2.x ones read the PhysX view): randomization events are warned about,
    pass ``context_fn`` to condition the world model on them.
"""

from __future__ import annotations

import torch

from .isaaclab import IsaacLabTrackingTask
from .manager_based import ImaginedData, wxyz_to_xyzw


class _TorchField:
    """Stands in for a ProxyArray holding an imagined (differentiable) tensor."""

    __slots__ = ("torch",)

    def __init__(self, t: torch.Tensor):
        self.torch = t

    @property
    def warp(self):
        raise NotImplementedError("an imagined state is a torch tensor with gradients; terms must read .torch, not .warp")

    shape = property(lambda s: s.torch.shape)
    dtype = property(lambda s: s.torch.dtype)
    device = property(lambda s: s.torch.device)

    def __getitem__(self, idx):
        return self.torch[idx]

    def __len__(self):
        return len(self.torch)


def _computed(data: ImaginedData, name: str) -> bool:
    return name in vars(data) or name in ImaginedData.ALIASES or isinstance(getattr(type(data), name, None), property)


class _ProxyData:
    """Imagined robot data in Isaac Lab 3's form: computed fields as ``.torch`` holders with xyzw
    quaternions; everything else (static fields, methods) falls through to the real data unchanged."""

    def __init__(self, data: ImaginedData):
        object.__setattr__(self, "_imagined", data)

    def __getattr__(self, name):
        data = self._imagined
        v = getattr(data, name)
        if name.startswith("_") or not torch.is_tensor(v) or not _computed(data, name):
            return v
        if "quat" in name:
            v = wxyz_to_xyzw(v)
        elif "pose" in name or name.endswith("state_w"):
            v = torch.cat((v[..., :3], wxyz_to_xyzw(v[..., 3:7]), v[..., 7:]), -1)
        return _TorchField(v)


class IsaacLab3TrackingTask(IsaacLabTrackingTask):
    QUAT_ORDER = "xyzw"
    CONTEXT_READERS = {}

    def _state_pre_reset(self):
        # no PhysX view here; the 3.x data buffers are read directly (checked: a read inside _reset_idx does
        # not go stale, see docs/adapters.md)
        return self.get_state()

    def _t(self, x):
        return x.torch if hasattr(x, "torch") else x

    def _wrap_data(self, data):
        return _ProxyData(data)
