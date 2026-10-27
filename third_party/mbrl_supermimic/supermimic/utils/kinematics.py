"""Differentiable forward kinematics of a MuJoCo kinematic tree in torch.

One free-joint root, hinge joints (active or passive), bodies without joints rigidly attached to their
parent. The joint order is the caller's (the simulator entity's), so the state vector maps 1:1.

The tree is evaluated one depth level at a time, all bodies of a level in one batched expression: a
humanoid is ~8 levels deep, so a forward pass is ~8 small batched steps rather than one chain of tiny
matmuls per body. Imagination runs this (and its backward) every step, so this is what sets its speed.
"""

from __future__ import annotations

import mujoco
import numpy as np
import torch

from .rotations import axis_angle_to_mat, quat_to_mat


def _rodrigues(axis: torch.Tensor, angle: torch.Tensor) -> torch.Tensor:
    """Rotation matrices (..., L, 3, 3) about unit axes (L, 3) by angles (..., L)."""
    x, y, z = axis.unbind(-1)
    o = torch.zeros_like(x)
    K = torch.stack((o, -z, y, z, o, -x, -y, x, o), -1).reshape(axis.shape[:-1] + (3, 3))
    s, c = torch.sin(angle)[..., None, None], (1 - torch.cos(angle))[..., None, None]
    eye = torch.eye(3, device=axis.device, dtype=axis.dtype)
    return eye + s * K + c * (K @ K)


class TreeKinematics:
    def __init__(self, m: mujoco.MjModel, joint_names: list[str], tracked: list[str], device: str):
        assert np.abs(m.jnt_pos).max() < 1e-9, "joint anchors off the body origin are not supported"
        self.device = device
        jidx = {n: i for i, n in enumerate(joint_names)}
        root = None
        bodies, parent, joint_idx, axes, offs_p, offs_R = [], [], [], [], [], []
        local = {}
        for b in range(1, m.nbody):
            nj = m.body_jntnum[b]
            pb = m.body_parentid[b]
            if nj == 1 and m.jnt_type[m.body_jntadr[b]] == mujoco.mjtJoint.mjJNT_FREE:
                assert root is None, "one free joint expected"
                root = b
                local[b] = len(bodies)
                bodies.append(m.body(b).name), parent.append(-1), joint_idx.append(-1)
                axes.append(np.zeros(3)), offs_p.append(np.zeros(3)), offs_R.append(np.array([1.0, 0, 0, 0]))
                continue
            if pb not in local:
                continue  # world-attached markers / mocap bodies are not part of the robot tree
            assert nj <= 1, f"{m.body(b).name}: multi-joint bodies are not supported"
            if nj == 1:
                jid = m.body_jntadr[b]
                assert m.jnt_type[jid] == mujoco.mjtJoint.mjJNT_HINGE
                joint_idx.append(jidx[m.joint(jid).name])
                axes.append(m.jnt_axis[jid].copy())
            else:
                joint_idx.append(-1)
                axes.append(np.zeros(3))
            local[b] = len(bodies)
            bodies.append(m.body(b).name)
            parent.append(local[pb])
            offs_p.append(m.body_pos[b].copy())
            offs_R.append(m.body_quat[b].copy())
        self.body_names, self.parent, self.joint_idx = bodies, parent, joint_idx
        f32 = dict(dtype=torch.float32, device=device)
        self.axes = torch.tensor(np.array(axes), **f32)
        self.off_p = torch.tensor(np.array(offs_p), **f32)
        self.off_R = quat_to_mat(torch.tensor(np.array(offs_R), **f32))
        self.num_bodies = len(bodies)
        self.tracked = list(tracked)
        self.tracked_idx = [bodies.index(n) for n in tracked]
        lo, hi = np.zeros(len(joint_names)), np.zeros(len(joint_names))
        for j in range(m.njnt):
            n = m.joint(j).name
            if n in jidx:
                lo[jidx[n]], hi[jidx[n]] = m.jnt_range[j]
        self.q_lo, self.q_hi = torch.tensor(lo, **f32), torch.tensor(hi, **f32)
        self._build_levels()

    def _build_levels(self):
        """Bodies grouped by depth; each level holds its parents as columns of the level-ordered tensors."""
        depth = [0] * self.num_bodies
        for i, par in enumerate(self.parent):
            depth[i] = 0 if par < 0 else depth[par] + 1
        order = sorted(range(self.num_bodies), key=lambda i: (depth[i], i))
        col = {b: c for c, b in enumerate(order)}
        self.levels = []
        for d in range(1, max(depth) + 1):
            ids = [i for i in order if depth[i] == d]
            L = lambda x, dt=torch.long: torch.tensor(x, dtype=dt, device=self.device)  # noqa: E731
            self.levels.append({
                "parent": L([col[self.parent[i]] for i in ids]),
                "off_p": self.off_p[ids], "off_R": self.off_R[ids], "axes": self.axes[ids],
                "joint": L([max(self.joint_idx[i], 0) for i in ids]),
                "has_joint": L([float(self.joint_idx[i] >= 0) for i in ids], torch.float32),
            })  # fmt: skip
        self._col_of_body = torch.tensor([col[b] for b in range(self.num_bodies)], device=self.device)
        self._col_of_tracked = torch.tensor([col[b] for b in self.tracked_idx], device=self.device)

    def forward(self, p, R, v, w, q, qd, all_bodies: bool = False):
        """Tracked-body world positions, rotations, link-origin linear and angular velocities.

        Args: root position p (N, 3), rotation R (N, 3, 3), world linear / angular velocity v, w (N, 3),
        joint positions / velocities q, qd (N, J) in the caller's joint order."""
        P, Rs, V, W = p[:, None], R[:, None], v[:, None], w[:, None]
        for lv in self.levels:
            par = lv["parent"]
            Pp, Rp, Vp, Wp = P[:, par], Rs[:, par], V[:, par], W[:, par]
            pi = Pp + (Rp @ lv["off_p"][..., None]).squeeze(-1)
            R_off = Rp @ lv["off_R"]
            Ri = R_off @ _rodrigues(lv["axes"], q[:, lv["joint"]] * lv["has_joint"])
            vi = Vp + torch.cross(Wp, pi - Pp, dim=-1)
            wi = Wp + (R_off @ lv["axes"][..., None]).squeeze(-1) * (qd[:, lv["joint"]] * lv["has_joint"])[..., None]
            P, Rs, V, W = torch.cat((P, pi), 1), torch.cat((Rs, Ri), 1), torch.cat((V, vi), 1), torch.cat((W, wi), 1)
        cols = self._col_of_body if all_bodies else self._col_of_tracked
        return P[:, cols], Rs[:, cols], V[:, cols], W[:, cols]

    def forward_reference(self, p, R, v, w, q, qd, all_bodies: bool = False):
        """The same, one body at a time (the definition the batched forward is tested against)."""
        P, Rs, V, W = [None] * self.num_bodies, [None] * self.num_bodies, [None] * self.num_bodies, [None] * self.num_bodies
        for i in range(self.num_bodies):
            par = self.parent[i]
            if par < 0:
                P[i], Rs[i], V[i], W[i] = p, R, v, w
                continue
            Rp = Rs[par]
            pi = P[par] + (Rp @ self.off_p[i].unsqueeze(-1)).squeeze(-1)
            R_off = Rp @ self.off_R[i]
            vi = V[par] + torch.cross(W[par], pi - P[par], dim=-1)
            j = self.joint_idx[i]
            if j < 0:
                Ri, wi = R_off, W[par]
            else:
                Ri = R_off @ axis_angle_to_mat(self.axes[i], q[..., j])
                wi = W[par] + (R_off @ self.axes[i].unsqueeze(-1)).squeeze(-1) * qd[..., j : j + 1]
            P[i], Rs[i], V[i], W[i] = pi, Ri, vi, wi
        idx = range(self.num_bodies) if all_bodies else self.tracked_idx
        return (
            torch.stack([P[i] for i in idx], -2),
            torch.stack([Rs[i] for i in idx], -3),
            torch.stack([V[i] for i in idx], -2),
            torch.stack([W[i] for i in idx], -2),
        )
