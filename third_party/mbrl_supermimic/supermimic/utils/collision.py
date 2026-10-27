"""Differentiable self-collision proxy from a MuJoCo model's capsule and sphere geoms.

Contact sensors cannot be evaluated on imagined states, so a policy trained in imagination never sees a
self-collision penalty and learns to collide. This proxy gives it one: the signed distance between every
pair of robot collision geoms MuJoCo would collide (contype/conaffinity compatible, different bodies, not
parent-child, not excluded), from the body poses of the imagined state. Pairs that already overlap in the
reference pose given at construction are dropped (they are in permanent contact by design).
"""

from __future__ import annotations

import mujoco
import numpy as np
import torch

from .rotations import quat_to_mat


def _segment_distance(p1, q1, p2, q2, eps: float = 1e-9):
    """Distance between segments [p1, q1] and [p2, q2] (batched, Ericson 5.1.9)."""
    d1, d2, r = q1 - p1, q2 - p2, p1 - p2
    a, e, f = (d1 * d1).sum(-1), (d2 * d2).sum(-1), (d2 * r).sum(-1)
    c, b = (d1 * r).sum(-1), (d1 * d2).sum(-1)
    denom = a * e - b * b
    s = torch.where(denom > eps, ((b * f - c * e) / denom.clamp_min(eps)).clamp(0, 1), torch.zeros_like(a))
    t = (b * s + f) / e.clamp_min(eps)
    s = torch.where(t < 0, (-c / a.clamp_min(eps)).clamp(0, 1), torch.where(t > 1, ((b - c) / a.clamp_min(eps)).clamp(0, 1), s))
    t = t.clamp(0, 1)
    gap = p1 + d1 * s[..., None] - p2 - d2 * t[..., None]
    return torch.sqrt((gap * gap).sum(-1) + eps)


class SelfCollisionProxy:
    def __init__(self, m: mujoco.MjModel, body_names: list[str], device, ref_pose=None, root_body: str | None = None):
        """body_names: the FK body order (MuJoCo names). ref_pose: (P (B, 3), R (B, 3, 3)) of one
        configuration used to drop pairs in permanent contact; None keeps all pairs."""
        bidx = {n: i for i, n in enumerate(body_names)}
        geoms = []
        for g in range(m.ngeom):
            body = m.body(m.geom_bodyid[g]).name
            if body not in bidx or not (m.geom_contype[g] or m.geom_conaffinity[g]):
                continue
            t = m.geom_type[g]
            if t not in (mujoco.mjtGeom.mjGEOM_CAPSULE, mujoco.mjtGeom.mjGEOM_SPHERE):
                raise NotImplementedError(f"geom {m.geom(g).name}: only capsules and spheres are supported")
            half = m.geom_size[g][1] if t == mujoco.mjtGeom.mjGEOM_CAPSULE else 0.0
            geoms.append((g, bidx[body], m.geom_size[g][0], half))
        excluded = {(m.exclude_signature[i] >> 16, m.exclude_signature[i] & 0xFFFF) for i in range(m.nexclude)}
        filter_parent = not (m.opt.disableflags & mujoco.mjtDisableBit.mjDSBL_FILTERPARENT)
        pairs = []
        for i, (ga, ba, _, _) in enumerate(geoms):
            for j in range(i + 1, len(geoms)):
                gb, bb = geoms[j][0], geoms[j][1]
                ma, mb = m.geom_bodyid[ga], m.geom_bodyid[gb]
                if ba == bb or not ((m.geom_contype[ga] & m.geom_conaffinity[gb]) or (m.geom_contype[gb] & m.geom_conaffinity[ga])):
                    continue
                if (min(ma, mb), max(ma, mb)) in excluded or (max(ma, mb), min(ma, mb)) in excluded:
                    continue
                if filter_parent and (m.body_parentid[ma] == mb or m.body_parentid[mb] == ma):
                    continue
                pairs.append((i, j))
        f32 = dict(dtype=torch.float32, device=device)
        self.body = torch.tensor([x[1] for x in geoms], dtype=torch.long, device=device)
        self.radius = torch.tensor([x[2] for x in geoms], **f32)
        self.half = torch.tensor([x[3] for x in geoms], **f32)
        self.pos = torch.tensor(np.array([m.geom_pos[x[0]] for x in geoms]), **f32)
        self.R = quat_to_mat(torch.tensor(np.array([m.geom_quat[x[0]] for x in geoms]), **f32))
        self.pa = torch.tensor([p[0] for p in pairs], dtype=torch.long, device=device)
        self.pb = torch.tensor([p[1] for p in pairs], dtype=torch.long, device=device)
        if ref_pose is not None and len(pairs):
            d = self.distances(ref_pose[0][None], ref_pose[1][None])[0]
            keep = d > 0
            self.pa, self.pb = self.pa[keep], self.pb[keep]
        self.num_pairs = int(self.pa.numel())

    def distances(self, P, R):
        """Signed surface distance per pair (N, K) from body positions (N, B, 3) and rotations (N, B, 3, 3)."""
        Rb = R[:, self.body]
        c = P[:, self.body] + (Rb @ self.pos.unsqueeze(-1)).squeeze(-1)
        axis = (Rb @ self.R)[..., :, 2] * self.half[:, None]
        p0, p1 = c - axis, c + axis
        d = _segment_distance(p0[:, self.pa], p1[:, self.pa], p0[:, self.pb], p1[:, self.pb])
        return d - self.radius[self.pa] - self.radius[self.pb]

    def soft_contact(self, P, R, width: float = 0.005):
        """Differentiable "any pair in contact" indicator in [0, 1] (N,)."""
        if self.num_pairs == 0:
            return torch.zeros(P.shape[0], device=P.device)
        return torch.sigmoid(-self.distances(P, R).amin(-1) / width)
