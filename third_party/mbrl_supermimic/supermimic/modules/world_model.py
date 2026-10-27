"""SuperMimic world model over the full floating-base state.

s_{t+1}, aux_t = f(s_t, a_t, xi): a deterministic MLP on heading- and position-invariant features that
predicts frame-local state increments (position in the heading frame, rotation and velocities in the root
frame), conditioned on the per-env context xi the policy cannot see. An optional head predicts ``aux_dim``
extra targets the imagined rewards need (e.g. the actuators' requested torque at the last substep, which a
control-rate state cannot reproduce).
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ..env.tracking_task import StateLayout
from ..utils.rotations import mat_to_6d, so3_exp, so3_log, yaw_mat
from .normalizer import RunningNorm, mlp


def invariant_features(S) -> torch.Tensor:
    """Height, tilt (6D, heading removed), root-frame velocities, joints."""
    p, R, v, w, q, qd = S
    H = yaw_mat(R)
    Rt = R.transpose(-1, -2)
    return torch.cat((p[..., 2:3], mat_to_6d(H.transpose(-1, -2) @ R), (Rt @ v.unsqueeze(-1)).squeeze(-1),
                      (Rt @ w.unsqueeze(-1)).squeeze(-1), q, qd), -1)  # fmt: skip


class WorldModel(nn.Module):
    def __init__(self, layout: StateLayout, num_actions: int, xi_dim: int = 0, aux_dim: int = 0,
                 hidden_dims=(2048, 2048, 2048), activation: str = "silu", layer_norm: bool = True):  # fmt: skip
        super().__init__()
        self.layout, self.nj, self.aux_dim = layout, layout.num_joints, aux_dim
        self.in_dim = layout.feature_dim + num_actions + xi_dim
        self.net = mlp(self.in_dim, list(hidden_dims), layout.delta_dim + aux_dim, act=activation, layer_norm=layer_norm)
        # start as the identity dynamics (mean increment), so early imagination is benign
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
        self.in_norm = RunningNorm(self.in_dim)
        self.out_norm = RunningNorm(layout.delta_dim, eps=1e-4)
        self.aux_norm = RunningNorm(max(aux_dim, 1), eps=1e-1)
        self.amp_dtype: torch.dtype | None = None  # autocast dtype for the MLP only (integration stays fp32)

    def inputs(self, S, a, xi):
        return torch.cat((invariant_features(S), a, xi), -1)

    @staticmethod
    def targets(S, S1):
        """Frame-local increments from S to S1 (the regression targets)."""
        p, R, v, w, q, qd = S
        p1, R1, v1, w1, q1, qd1 = S1
        H = yaw_mat(R)
        Rt = R.transpose(-1, -2)
        return torch.cat(((H.transpose(-1, -2) @ (p1 - p).unsqueeze(-1)).squeeze(-1), so3_log(Rt @ R1),
                          (Rt @ (v1 - v).unsqueeze(-1)).squeeze(-1), (Rt @ (w1 - w).unsqueeze(-1)).squeeze(-1),
                          q1 - q, qd1 - qd), -1)  # fmt: skip

    @staticmethod
    def state_error(S_hat, S):
        """Error of S_hat in S's local frames, same layout as ``targets``."""
        return WorldModel.targets(S, S_hat)

    @torch.no_grad()
    def update_normalization(self, S, a, xi, S1, aux=None):
        self.in_norm.update(self.inputs(S, a, xi))
        self.out_norm.update(self.targets(S, S1))
        if self.aux_dim and aux is not None:
            self.aux_norm.update(aux)

    def step(self, S, a, xi):
        """Returns (S1, aux); aux is None without an aux head."""
        p, R, v, w, q, qd = S
        x = self.in_norm(self.inputs(S, a, xi))
        with torch.autocast(x.device.type, dtype=self.amp_dtype or torch.float32, enabled=self.amp_dtype is not None):
            y = self.net(x).float()
        n = self.layout.delta_dim
        d = y[..., :n] * self.out_norm.std() + self.out_norm.mean
        aux = y[..., n:] * self.aux_norm.std() + self.aux_norm.mean if self.aux_dim else None
        H = yaw_mat(R)
        dp, dth, dv, dw, dq, dqd = d.split((3, 3, 3, 3, self.nj, self.nj), -1)
        p1 = p + (H @ dp.unsqueeze(-1)).squeeze(-1)
        R1 = R @ so3_exp(dth)
        v1 = v + (R @ dv.unsqueeze(-1)).squeeze(-1)
        w1 = w + (R @ dw.unsqueeze(-1)).squeeze(-1)
        return (p1, R1, v1, w1, q + dq, qd + dqd), aux
