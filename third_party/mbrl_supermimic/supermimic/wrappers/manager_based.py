"""Shared machinery for manager-based envs (Isaac Lab, mjlab): run the env's own reward, termination and
observation terms on world-model states.

Inside ``ManagerBasedTrackingTask.imagined(...)``:
  * the robot asset's data is replaced by fields computed from S by differentiable forward kinematics
    (static per-env fields such as defaults, limits, encoder bias and COM offsets come from the real data);
  * the action manager shows the imagined action and previous action;
  * the motion command (BeyondMimic's ``MotionCommand`` and its mjlab port) sits at the imagined frame,
    with its relative body poses re-anchored.
Everything is restored when the block exits, so the live env never sees an imagined state.

The ordering mirrors ``ManagerBasedRLEnv.step``. A step's terminations and rewards see the post-step state
and the command as it was before the step (its frame, its poses anchored at the pre-step state). The
observations after the step see the advanced command.

Framework adapters (``mjlab.py``, ``isaaclab.py``) subclass the task and supply the kinematic model, the
data field names, sensors, noise application and the step API.

Domain-randomization context (``context_fn="auto"``): the world model is conditioned on the physical
parameters the env's own events randomize (friction, COM offsets, masses, joint offsets ...), read back per
env from the simulator after the events ran. Each adapter registers a reader per event function
(``CONTEXT_READERS``); events that move the state (resets, pushes) or only change rendering are not
context, and any other event is reported as not represented. The context is also appended to the critic
observation (the paper's V(s, g, xi)), in the real and the imagined observations alike.
"""

from __future__ import annotations

import warnings
from contextlib import contextmanager
from typing import Callable

import torch

from ..env.tracking_task import ImaginedRollout, ImaginedStep, StateLayout, TrackingTask
from ..utils.kinematics import TreeKinematics
from ..utils.rotations import mat_to_quat, quat_to_mat


def rot_apply_inv(R, x):
    return (R.transpose(-1, -2) @ x.unsqueeze(-1)).squeeze(-1)


def quat_mul(a, b):
    aw, ax, ay, az = a.unbind(-1)
    bw, bx, by, bz = b.unbind(-1)
    return torch.stack((aw * bw - ax * bx - ay * by - az * bz, aw * bx + ax * bw + ay * bz - az * by,
                        aw * by - ax * bz + ay * bw + az * bx, aw * bz + ax * by - ay * bx + az * bw), -1)  # fmt: skip


def quat_conj(q):
    return torch.cat((q[..., :1], -q[..., 1:]), -1)


def quat_apply(q, v):
    return (quat_to_mat(q) @ v.unsqueeze(-1)).squeeze(-1)


def yaw_quat(q):
    w, x, y, z = q.unbind(-1)
    yaw = torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    zero = torch.zeros_like(yaw)
    return torch.stack((torch.cos(yaw / 2), zero, zero, torch.sin(yaw / 2)), -1)


class _SafeAcos(torch.autograd.Function):
    """acos with a bounded derivative at |x| = 1: acos(cos)^2 has gradient 0 * inf = NaN when fp32 rounds
    a small angle's cosine to exactly 1."""

    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return _ACOS(x)

    @staticmethod
    def backward(ctx, g):
        (x,) = ctx.saved_tensors
        return -g / torch.sqrt((1 - x * x).clamp_min(1e-6))


_ACOS = torch.acos


class Delegate:
    """Attribute access falls through to the wrapped object unless overridden."""

    def __init__(self, inner, **overrides):
        object.__setattr__(self, "_inner", inner)
        for k, v in overrides.items():
            object.__setattr__(self, k, v)

    def __getattr__(self, name):
        return getattr(self._inner, name)


def _state_fields(kin, body_perm, p, R, v, w, q, qd, origins, com_b, com_q_rel):
    """FK of S and the link / COM fields derived from it (one fused graph under torch.compile)."""
    P, Rs, V, W = kin.forward(p + origins, R, v, w, q, qd, all_bodies=True)
    if body_perm is not None:
        P, Rs, V, W = P[:, body_perm], Rs[:, body_perm], V[:, body_perm], W[:, body_perm]
    quat = mat_to_quat(Rs)
    com_off = (Rs @ com_b.unsqueeze(-1)).squeeze(-1)
    return (Rs, torch.cat((p + origins, mat_to_quat(R)), -1), torch.cat((v, w), -1), torch.cat((P, quat), -1),
            torch.cat((V, W), -1), torch.cat((P + com_off, quat_mul(quat, com_q_rel)), -1),
            torch.cat((V + torch.cross(W, com_off, dim=-1), W), -1))  # fmt: skip


class ImaginedData:
    """Robot data fields computed from S: link and COM poses and velocities of every body, in the asset's
    body order, under both mjlab's and Isaac Lab's names (``ALIASES``). Fields that do not depend on the
    state fall through to the real data; dynamic fields not computed here raise."""

    DYNAMIC_UNSUPPORTED = {"joint_acc", "body_acc_w", "body_lin_acc_w", "body_ang_acc_w", "applied_torque",
                           "computed_torque", "actuator_force", "qfrc_actuator", "joint_torques", "body_external_wrench"}  # fmt: skip

    def __init__(self, real, kin: TreeKinematics, body_perm, S, origins, com_b, com_q_rel, gravity, forward, fields=_state_fields):
        p, R, v, w, q, qd = S
        self._real, self._gravity, self._forward = real, gravity, forward
        self.root_R = R
        self.joint_pos, self.joint_vel = q, qd
        (self._Rs, self.root_link_pose_w, self.root_link_vel_w, self.body_link_pose_w, self.body_link_vel_w,
         self.body_com_pose_w, self.body_com_vel_w) = fields(kin, body_perm, p, R, v, w, q, qd, origins, com_b, com_q_rel)  # fmt: skip

    def __getattr__(self, name):
        if name in ImaginedData.ALIASES:
            return getattr(self, ImaginedData.ALIASES[name])
        if name in ImaginedData.DYNAMIC_UNSUPPORTED:
            raise AttributeError(f"imagined robot data cannot provide '{name}' (not a function of the state)")
        return getattr(self._real, name)

    root_link_pos_w = property(lambda s: s.root_link_pose_w[:, 0:3])
    root_link_quat_w = property(lambda s: s.root_link_pose_w[:, 3:7])
    root_link_lin_vel_w = property(lambda s: s.root_link_vel_w[:, 0:3])
    root_link_ang_vel_w = property(lambda s: s.root_link_vel_w[:, 3:6])
    root_link_lin_vel_b = property(lambda s: rot_apply_inv(s.root_R, s.root_link_lin_vel_w))
    root_link_ang_vel_b = property(lambda s: rot_apply_inv(s.root_R, s.root_link_ang_vel_w))
    root_com_pose_w = property(lambda s: s.body_com_pose_w[:, 0])
    root_com_vel_w = property(lambda s: s.body_com_vel_w[:, 0])
    root_com_pos_w = property(lambda s: s.body_com_pose_w[:, 0, 0:3])
    root_com_quat_w = property(lambda s: s.body_com_pose_w[:, 0, 3:7])
    root_com_lin_vel_w = property(lambda s: s.body_com_vel_w[:, 0, 0:3])
    root_com_ang_vel_w = property(lambda s: s.body_com_vel_w[:, 0, 3:6])
    root_com_lin_vel_b = property(lambda s: rot_apply_inv(s.root_R, s.root_com_lin_vel_w))
    root_com_ang_vel_b = property(lambda s: rot_apply_inv(s.root_R, s.root_com_ang_vel_w))
    body_link_pos_w = property(lambda s: s.body_link_pose_w[..., 0:3])
    body_link_quat_w = property(lambda s: s.body_link_pose_w[..., 3:7])
    body_link_lin_vel_w = property(lambda s: s.body_link_vel_w[..., 0:3])
    body_link_ang_vel_w = property(lambda s: s.body_link_vel_w[..., 3:6])
    body_com_pos_w = property(lambda s: s.body_com_pose_w[..., 0:3])
    body_com_quat_w = property(lambda s: s.body_com_pose_w[..., 3:7])
    body_com_lin_vel_w = property(lambda s: s.body_com_vel_w[..., 0:3])
    body_com_ang_vel_w = property(lambda s: s.body_com_vel_w[..., 3:6])
    projected_gravity_b = property(lambda s: rot_apply_inv(s.root_R, s._gravity))
    root_link_state_w = property(lambda s: torch.cat((s.root_link_pose_w, s.root_link_vel_w), -1))
    root_com_state_w = property(lambda s: torch.cat((s.root_com_pose_w, s.root_com_vel_w), -1))
    root_state_w = property(lambda s: torch.cat((s.root_link_pose_w, s.root_com_vel_w), -1))
    body_link_state_w = property(lambda s: torch.cat((s.body_link_pose_w, s.body_link_vel_w), -1))
    body_com_state_w = property(lambda s: torch.cat((s.body_com_pose_w, s.body_com_vel_w), -1))
    body_state_w = property(lambda s: torch.cat((s.body_link_pose_w, s.body_com_vel_w), -1))

    @property
    def heading_w(self):
        f = (self.root_R @ self._forward.unsqueeze(-1)).squeeze(-1)
        return torch.atan2(f[:, 1], f[:, 0])

    @property
    def joint_pos_biased(self):
        return self.joint_pos + self._real.encoder_bias

    # Isaac Lab's short names: link-frame poses, COM-frame velocities
    ALIASES = {
        "root_pos_w": "root_link_pos_w", "root_quat_w": "root_link_quat_w", "root_vel_w": "root_com_vel_w",
        "root_lin_vel_w": "root_com_lin_vel_w", "root_ang_vel_w": "root_com_ang_vel_w",
        "root_lin_vel_b": "root_com_lin_vel_b", "root_ang_vel_b": "root_com_ang_vel_b",
        "body_pos_w": "body_link_pos_w", "body_quat_w": "body_link_quat_w", "body_vel_w": "body_com_vel_w",
        "body_lin_vel_w": "body_com_lin_vel_w", "body_ang_vel_w": "body_com_ang_vel_w",
    }  # fmt: skip


def _identity(x):
    return x


def xyzw_to_wxyz(q):
    return q[..., [3, 0, 1, 2]]


def wxyz_to_xyzw(q):
    return q[..., [1, 2, 3, 0]]


def relative_body_poses(cmd, to_wxyz=_identity, from_wxyz=_identity):
    """BeyondMimic's reference body poses re-anchored to the robot (xy and yaw of the robot anchor, the
    reference's height), as MotionCommand computes them each step. Quaternions are converted to wxyz and
    back for frameworks that use another order."""
    nb = len(cmd.cfg.body_names)
    anchor_p = cmd.anchor_pos_w[:, None].expand(-1, nb, -1)
    anchor_q = to_wxyz(cmd.anchor_quat_w)[:, None].expand(-1, nb, -1)
    robot_p = cmd.robot_anchor_pos_w[:, None].expand(-1, nb, -1)
    robot_q = to_wxyz(cmd.robot_anchor_quat_w)[:, None].expand(-1, nb, -1)
    delta_p = torch.cat((robot_p[..., :2], anchor_p[..., 2:3]), -1)
    delta_q = yaw_quat(quat_mul(robot_q, quat_conj(anchor_q)))
    body_q = from_wxyz(quat_mul(delta_q, to_wxyz(cmd.body_quat_w)))
    return delta_p + quat_apply(delta_q, cmd.body_pos_w - anchor_p), body_q


# ------------------------------------------------------------------ motion command protocol
class MotionCursor:
    """What imagination needs from the motion command: read / move the per-env frame cursor, know the last
    valid frame, and recompute the robot-anchored reference poses after the robot data was swapped.

    Two conventions are recognized:
      * BeyondMimic / mjlab ``MotionCommand``: ``time_steps`` and ``motion.time_step_total`` (one clip);
      * SONIC-style motion libraries: ``time_steps``, ``motion_num_steps`` and ``motion_start_time_steps``
        (one clip per env, with a start offset).
    Any other command works by passing ``cursor=MotionCursor(cmd, last_frame=lambda cmd: ...)`` to the task,
    or by giving the command those attributes."""

    REQUIRED = ("anchor_pos_w", "anchor_quat_w", "robot_anchor_pos_w", "robot_anchor_quat_w", "body_pos_w",
                "body_quat_w", "robot_body_pos_w", "body_pos_relative_w", "body_quat_relative_w")  # fmt: skip
    # a command method that recomputes body_*_relative_w from the current robot data, if the command has one
    RELATIVE_UPDATERS = ("update_relative_body_poses", "_update_relative_reference")

    def __init__(self, cmd, last_frame: Callable | None = None):
        self.cmd = cmd
        if last_frame is not None:
            self._last = last_frame
        elif hasattr(cmd, "motion") and hasattr(cmd.motion, "time_step_total"):
            self._last = lambda c: c.motion.time_step_total - 1
        elif hasattr(cmd, "motion_num_steps") and hasattr(cmd, "motion_start_time_steps"):
            self._last = lambda c: c.motion_num_steps - 1 - c.motion_start_time_steps
        else:
            raise NotImplementedError(
                f"SuperMimic cannot find the clip length on the motion command ({type(cmd).__name__}): expected "
                "'motion.time_step_total' (BeyondMimic / mjlab) or 'motion_num_steps' + 'motion_start_time_steps' "
                "(SONIC-style). Pass cursor=MotionCursor(cmd, last_frame=lambda cmd: <per-env last frame>).")
        if not hasattr(cmd, "time_steps"):
            raise NotImplementedError(f"motion command {type(cmd).__name__} has no 'time_steps' frame cursor")
        missing = [a for a in self.REQUIRED if not hasattr(cmd, a)]
        if missing:
            warnings.warn(f"SuperMimic: motion command {type(cmd).__name__} lacks {missing}; the stock tracking terms "
                          "and soft terminations will not work on it, custom terms may", stacklevel=3)
        self._updater = next((getattr(cmd, n) for n in self.RELATIVE_UPDATERS if callable(getattr(cmd, n, None))), None)

    @property
    def frames(self):
        return self.cmd.time_steps

    def last_frame(self):
        return self._last(self.cmd)

    def set_frames(self, frames):
        self.cmd.time_steps = torch.minimum(frames.clamp(min=0), torch.as_tensor(self.last_frame(), device=frames.device))

    def update_relative(self, to_wxyz=_identity, from_wxyz=_identity):
        """Recompute the robot-anchored reference poses from the (swapped) robot data."""
        if self._updater is not None:
            self._updater()
        elif hasattr(self.cmd, "body_pos_relative_w"):
            self.cmd.body_pos_relative_w, self.cmd.body_quat_relative_w = relative_body_poses(self.cmd, to_wxyz, from_wxyz)

    def snapshot(self):
        c = self.cmd
        return (c.time_steps, getattr(c, "body_pos_relative_w", None), getattr(c, "body_quat_relative_w", None))

    def restore(self, snap):
        c = self.cmd
        c.time_steps, pos, quat = snap
        if pos is not None:
            c.body_pos_relative_w, c.body_quat_relative_w = pos, quat
        inv = getattr(c, "_invalidate_reference_cache", None)
        if callable(inv):
            inv()


# ------------------------------------------------------------------ soft terminations of the tracking terms
def _gravity(data, t=_identity):
    """Unit gravity direction in the world (Isaac Lab 3 / Newton store it in m/s^2)."""
    g = getattr(data, "gravity_vec_w", None)
    g = t(data.GRAVITY_VEC_W if g is None else g)
    return g / g.norm(dim=-1, keepdim=True)


def _anchor_pos_margin(env, command_name, threshold):
    c = env.command_manager.get_term(command_name)
    return (c.anchor_pos_w - c.robot_anchor_pos_w).norm(dim=-1) - threshold


def _anchor_pos_z_margin(env, command_name, threshold):
    c = env.command_manager.get_term(command_name)
    return (c.anchor_pos_w[:, -1] - c.robot_anchor_pos_w[:, -1]).abs() - threshold


def _anchor_ori_margin(env, asset_cfg, command_name, threshold):
    # env is the imagination view: _sm_t reads a framework field as a tensor, _sm_to_wxyz fixes quat order
    c = env.command_manager.get_term(command_name)
    g = _gravity(env.scene[asset_cfg.name].data, getattr(env, "_sm_t", _identity))
    to_wxyz = getattr(env, "_sm_to_wxyz", _identity)
    gz = lambda qt: rot_apply_inv(quat_to_mat(to_wxyz(qt)), g)[:, 2]  # noqa: E731
    return (gz(c.anchor_quat_w) - gz(c.robot_anchor_quat_w)).abs() - threshold


def _body_idx(c, body_names):
    return [i for i, n in enumerate(c.cfg.body_names) if body_names is None or n in body_names]


def _body_pos_margin(env, command_name, threshold, body_names=None):
    c = env.command_manager.get_term(command_name)
    i = _body_idx(c, body_names)
    return (c.body_pos_relative_w[:, i] - c.robot_body_pos_w[:, i]).norm(dim=-1).amax(-1) - threshold


def _body_pos_z_margin(env, command_name, threshold, body_names=None):
    c = env.command_manager.get_term(command_name)
    i = _body_idx(c, body_names)
    return (c.body_pos_relative_w[:, i, -1] - c.robot_body_pos_w[:, i, -1]).abs().amax(-1) - threshold


# termination function name -> margin function (positive = terminated); the soft surrogate is
# sigmoid(margin / tau). Register your own terms here or pass soft_terminations to the task.
SOFT_TERMINATIONS: dict[str, Callable] = {
    "bad_anchor_pos": _anchor_pos_margin,
    "bad_anchor_pos_z_only": _anchor_pos_z_margin,
    "bad_anchor_ori": _anchor_ori_margin,
    "bad_motion_body_pos": _body_pos_margin,
    "bad_motion_body_pos_z_only": _body_pos_z_margin,
}


class ManagerBasedTrackingTask(TrackingTask):
    DEFAULT_OBS_GROUPS = (("policy", "policy"), ("critic", "critic"))
    # the framework's quaternion order; imagination works in wxyz and converts at the boundary
    QUAT_ORDER = "wxyz"
    # reward function name -> factory(task, term_cfg) of a function evaluating it on imagined states
    IMAGINED_REWARDS: dict[str, Callable] = {}
    # event function name -> factory(task, event_cfg) of fn(env) -> (N, ...): the per-env values it randomized
    CONTEXT_READERS: dict[str, Callable] = {}
    # event functions (name prefixes) that are not context: state resets, pushes, rendering
    CONTEXT_IGNORED: tuple[str, ...] = ("reset_", "push_", "randomize_visual_")

    def __init__(self, env, entity_name: str = "robot", command_name: str = "motion",
                 obs_groups: dict[str, str] | None = None, skip_rewards=(), soft_terminations: dict | None = None,
                 term_tau: float = 0.05, context_fn: Callable | str | None = "auto", context_dim: int | None = None,
                 context_in_critic: bool = True, compile: bool = True, cursor: MotionCursor | None = None,
                 resample_fn: Callable | None = None):  # fmt: skip
        """obs_groups maps SuperMimic keys to the env's observation groups. context_fn(env) -> (N, context_dim)
        supplies DR the policy cannot see to the world model; "auto" reads what the env's events randomized
        (see CONTEXT_READERS), None gives none. context_in_critic: append it to the "critic" observation.
        resample_fn() -> bool: load a new motion subset (see TrackingTask.resample_motions); by default a
        motion command holding a SONIC-style library (``motion_lib.load_motions_for_training``, not all
        motions loaded) is resampled the way the SONIC wrapper does it: load, then reset every env.
        compile: torch.compile the imagined-state
        kinematics (imagination is kernel-launch bound without it). cursor: how to drive a motion command
        that follows neither the BeyondMimic / mjlab nor the SONIC convention (see MotionCursor)."""
        self.env = e = env.unwrapped
        self.entity_name, self.entity = entity_name, e.scene[entity_name]
        self.command = e.command_manager.get_term(command_name)
        self.cursor = cursor or MotionCursor(self.command)
        self.num_envs, self.device = e.num_envs, torch.device(e.device)
        self.num_actions = e.action_manager.total_action_dim
        self.step_dt = e.step_dt
        self.layout = StateLayout(len(self.entity.joint_names))
        self._hook_reset()
        self.term_tau = term_tau
        self.kin, self.body_perm = self._build_kinematics()
        self._fields = torch.compile(_state_fields, dynamic=False) if compile else _state_fields
        self._data_cache: list = []
        om = e.observation_manager
        self.obs_groups = obs_groups or {k: g for k, g in self.DEFAULT_OBS_GROUPS if g in om.active_terms}
        self.obs_terms = {g: list(zip(om._group_obs_term_names[g], om._group_obs_term_cfgs[g], strict=True)) for g in self.obs_groups.values()}
        for g, terms in self.obs_terms.items():
            assert om._group_obs_concatenate[g], f"observation group '{g}' must be concatenated"
            for n, c in terms:
                assert getattr(c, "delay_max_lag", 0) == 0, f"observation delay ({g}/{n}) is not supported"
        group_cfg = (lambda g: om.cfg[g]) if isinstance(om.cfg, dict) else (lambda g: getattr(om.cfg, g))
        self.corrupt = {g: bool(getattr(group_cfg(g), "enable_corruption", False)) for g in self.obs_terms}
        self._set_context(context_fn, context_dim, context_in_critic)
        lib = getattr(self.command, "motion_lib", None)
        partial = lib is not None and hasattr(lib, "load_motions_for_training") and not getattr(lib, "all_motions_loaded", False)
        self.motion_subset = resample_fn is not None or partial
        self._resample_fn = resample_fn
        self.history_length = max((c.history_length for terms in self.obs_terms.values() for _, c in terms), default=0)
        self.sensors = self._build_sensors()
        rm, tm = e.reward_manager, e.termination_manager
        self.reward_scale = e.step_dt if getattr(rm, "_scale_by_dt", True) else 1.0
        skip, self.reward_terms, proxied = set(skip_rewards), [], []
        for n, c in zip(rm._term_names, rm._term_cfgs, strict=True):
            if c.weight == 0.0 or n in skip:
                continue
            factory = self.IMAGINED_REWARDS.get(getattr(c.func, "__name__", ""))
            if factory is not None:
                self.reward_terms.append((n, factory(self, c), c))
                proxied.append(n)
            elif self._reads_sensor(c):
                skip.add(n)
            else:
                self.reward_terms.append((n, c.func, c))
        if skip:
            warnings.warn(f"SuperMimic: reward terms not evaluated in imagination: {sorted(skip)}", stacklevel=2)
        if proxied:
            warnings.warn(f"SuperMimic: reward terms replaced by differentiable proxies in imagination: {proxied}", stacklevel=2)
        self.skipped_idx = [i for i, n in enumerate(rm._term_names) if n in skip]
        self.termination_terms = [(n, c) for n, c in zip(tm._term_names, tm._term_cfgs, strict=True) if not c.time_out]
        soft = {**SOFT_TERMINATIONS, **(soft_terminations or {})}
        self.soft_terms = [(soft.get(getattr(c.func, "__name__", "")), c) for _, c in self.termination_terms]
        hard_only = [n for (f, _), (n, _) in zip(self.soft_terms, self.termination_terms) if f is None]
        if hard_only:
            warnings.warn(f"SuperMimic: no soft surrogate (no gradient) for terminations {hard_only}", stacklevel=2)
        self.refresh_static()

    # entity methods that write the simulated state; outside a reset (pushes, motion resampling ...) such a
    # write makes the step's transition non-dynamics, which the replay must know (extras["teleported"])
    STATE_WRITES = ("write_root_pose_to_sim", "write_root_link_pose_to_sim", "write_root_com_pose_to_sim",
                    "write_root_velocity_to_sim", "write_root_com_velocity_to_sim", "write_root_link_velocity_to_sim",
                    "write_root_link_velocity_b_to_sim", "write_root_state_to_sim", "write_root_link_state_to_sim",
                    "write_root_com_state_to_sim", "write_joint_state_to_sim", "write_joint_position_to_sim",
                    "write_joint_velocity_to_sim")  # fmt: skip

    def _set_context(self, context_fn, context_dim, in_critic):
        self.context_terms, self._xi_cache, static = [], None, False
        if context_fn == "auto":
            readers, static = self._context_readers()
            context_fn = None
            if readers:
                N = self.num_envs

                def context_fn(env, _readers=tuple(fn for _, fn in readers)):
                    return torch.cat([torch.as_tensor(fn(env)).reshape(N, -1).float().to(self.device) for fn in _readers], -1)

                self.context_terms = [(n, torch.as_tensor(fn(self.env)).reshape(N, -1).shape[-1]) for n, fn in readers]
        self.context_fn, self.xi_dim = context_fn, 0
        if context_fn is not None:
            xi = context_fn(self.env)
            self.xi_dim = xi.shape[-1] if context_dim is None else context_dim
            assert xi.shape == (self.num_envs, self.xi_dim), f"context_fn returned {tuple(xi.shape)}, expected (N, {self.xi_dim})"
            self._xi_cache = xi if static else None  # startup-only randomization never changes afterwards
        self.context_in_critic = bool(in_critic and self.xi_dim > 0)
        if self.context_in_critic and "critic" not in self.obs_groups:
            warnings.warn("SuperMimic: no critic observation group; the DR context goes to the world model only", stacklevel=3)
            self.context_in_critic = False

    def _context_readers(self):
        """Readers of the parameters the env's events randomize: ([(event name, fn(env))], all startup-only)."""
        em = getattr(self.env, "event_manager", None)
        readers, unread, pushes, static = [], [], [], True
        for mode, cfgs in (getattr(em, "_mode_term_cfgs", None) or {}).items():
            for name, c in zip(em._mode_term_names[mode], cfgs, strict=True):
                fname = getattr(c.func, "__name__", type(c.func).__name__)  # Isaac Lab class terms are instances
                if fname.startswith("push_"):
                    pushes.append(name)
                if fname.startswith(self.CONTEXT_IGNORED):
                    continue
                factory = self.CONTEXT_READERS.get(fname)
                if factory is None:
                    unread.append(f"{name} ({fname}, {mode})")
                    continue
                readers.append((name, factory(self, c)))
                static &= str(mode) in ("startup", "prestartup")
        if unread:
            warnings.warn(f"SuperMimic: randomization events not in the world model's context (no reader; pass context_fn): {unread}", stacklevel=3)
        if pushes:
            warnings.warn(f"SuperMimic: push events {pushes} write the state inside step(); those transitions are cut from the "
                          "world-model data and pushes are never imagined (the paper trains without them)", stacklevel=3)  # fmt: skip
        return readers, static

    def resample_motions(self):
        if self._resample_fn is not None:
            return bool(self._resample_fn())
        if not self.motion_subset:
            return False
        cmd = self.command
        cap = getattr(cmd, "max_num_load_motions", None) or self.num_envs
        if not cmd.motion_lib.load_motions_for_training(max_num_seqs=min(self.num_envs, cap)):
            return False
        self.env.reset()  # every env restarts on the new motions (a teleport for the replay)
        return True

    def _ids(self, env_ids):
        N = self.num_envs
        if env_ids is None or isinstance(env_ids, slice):
            return torch.arange(N, device=self.device)[env_ids if env_ids is not None else slice(None)]
        return torch.as_tensor(env_ids, device=self.device).reshape(-1)

    def _state_pre_reset(self):
        """The state read inside _reset_idx, before the reset writes. Frameworks whose data buffers are lazy
        caches override it with a raw read (a cached read at this point would go stale after the write)."""
        return self.get_state()

    def _hook_reset(self):
        """Capture the state of the envs about to be reset inside step() (the transition into a termination),
        so the world model can learn it (extras["terminal_state"]), and notice state writes outside a reset.
        All three frameworks reset via _reset_idx."""
        e, N = self.env, self.num_envs
        self._terminal = torch.zeros(N, self.layout.state_dim, device=self.device)
        self._terminal_mask = torch.zeros(N, dtype=torch.bool, device=self.device)
        self._written = torch.zeros(N, dtype=torch.bool, device=self.device)
        self._in_reset = False
        orig = e._reset_idx

        def _reset_idx(env_ids=None, _orig=orig):
            ids = self._ids(env_ids)
            if len(ids):
                self._terminal[ids] = self._state_pre_reset()[ids]
                self._terminal_mask[ids] = True
            self._in_reset = True
            try:
                return _orig(env_ids)
            finally:
                self._in_reset = False

        e._reset_idx = _reset_idx
        for name in self.STATE_WRITES:
            f = getattr(self.entity, name, None)
            if f is None:
                continue

            def write(*a, _f=f, **k):
                if not self._in_reset:
                    self._written[self._ids(k.get("env_ids", a[1] if len(a) > 1 else None))] = True
                return _f(*a, **k)

            setattr(self.entity, name, write)

    # ------------------------------------------------------------------ framework hooks
    def _build_kinematics(self) -> tuple[TreeKinematics, list[int] | None]:
        """FK over the asset's joints (in its joint order) and the permutation from FK to asset body order."""
        raise NotImplementedError

    def _build_sensors(self) -> dict:
        return {}

    def _reads_sensor(self, cfg) -> bool:
        """Whether a reward term reads a sensor imagination cannot reproduce (it is then skipped)."""
        return False

    def _postprocess(self, cfg, x, corrupt: bool):
        """The observation manager's per-term noise / modifiers (out of place, before clip and scale)."""
        if corrupt and cfg.noise is not None:
            x = cfg.noise.apply(x)
        return x

    def _env_step(self, actions):
        """Returns obs groups, rewards, terminated, truncated, extras."""
        return self.env.step(actions)

    def _current_obs(self):
        return self.env.observation_manager.compute()

    def _scene_view(self, data):
        return self.env.scene

    def _t(self, x):
        """A robot data field as a torch tensor (frameworks may wrap fields, e.g. Isaac Lab 3's ProxyArray)."""
        return x

    def _wrap_data(self, data: ImaginedData):
        """Present imagined robot data the way the framework's terms expect it."""
        return data

    def _to_wxyz(self, q):
        return q if self.QUAT_ORDER == "wxyz" else xyzw_to_wxyz(q)

    def _from_wxyz(self, q):
        return q if self.QUAT_ORDER == "wxyz" else wxyz_to_xyzw(q)

    @property
    def _gravity_w(self):
        return _gravity(self.entity.data, self._t)

    @property
    def _forward_b(self):
        d = self.entity.data
        f = getattr(d, "forward_vec_b", None)
        return self._t(d.FORWARD_VEC_B if f is None else f)

    # ------------------------------------------------------------------ simulator
    def _map(self, obs):
        return self.with_context({k: obs[g] for k, g in self.obs_groups.items()}, self.get_context())

    def with_context(self, obs, xi):
        if not self.context_in_critic:
            return obs
        return {**obs, "critic": torch.cat((obs["critic"], xi), -1)}

    def get_observations(self):
        return self._map(self._current_obs())

    def step(self, actions):
        t_prev = self.cursor.frames.clone()
        self._terminal_mask[:] = False
        self._written[:] = False
        obs, rew, terminated, truncated, extras = self._env_step(actions)
        done = terminated | truncated
        terminal_state = torch.where(self._terminal_mask[:, None], self._terminal, self.get_state())
        # the motion command resamples (and teleports the robot) at the end of a motion without a done, and
        # events (pushes ...) write the state inside the step: neither transition is dynamics
        teleported = ~done & ((self.cursor.frames != t_prev + 1) | self._written)
        rm = self.env.reward_manager
        skipped = sum((rm._step_reward[:, i] * self.reward_scale for i in self.skipped_idx), torch.zeros_like(rew))
        terms = {n: rm._step_reward[:, i] * self.reward_scale for i, n in enumerate(rm._term_names)}
        return self._map(obs), rew, done, {"time_outs": truncated, "teleported": teleported, "reward_not_imagined": skipped,
                                           "reward_terms": terms, "terminal_state": terminal_state, "log": extras.get("log", {})}  # fmt: skip

    def reset(self, env_ids=None):
        self.env.reset(env_ids=env_ids)

    # ------------------------------------------------------------------ SuperMimic
    @torch.no_grad()
    def refresh_static(self):
        """Per-env COM offsets in the link frame (startup DR); call again after any model change."""
        d, t = self.entity.data, self._t
        link_q, com_q = self._to_wxyz(t(d.body_link_quat_w)), self._to_wxyz(t(d.body_com_quat_w))
        self.com_b = rot_apply_inv(quat_to_mat(link_q), t(d.body_com_pos_w) - t(d.body_link_pos_w))
        self.com_q_rel = quat_mul(quat_conj(link_q), com_q)

    def get_state(self):
        d, t = self.entity.data, self._t
        return torch.cat((t(d.root_link_pos_w) - self.env.scene.env_origins, self._to_wxyz(t(d.root_link_quat_w)),
                          t(d.root_link_lin_vel_w), t(d.root_link_ang_vel_w), t(d.joint_pos), t(d.joint_vel)), -1)  # fmt: skip

    def get_context(self):
        if self.context_fn is None:
            return super().get_context()
        return self._xi_cache if self._xi_cache is not None else self.context_fn(self.env)

    def get_side_state(self):
        am = self.env.action_manager
        return {"frame": self.cursor.frames.clone(), "last_action": am.action.clone(), "prev_action": am.prev_action.clone()}

    def imagine(self, side=None, past=None):
        return ManagerRollout(self, side, past)

    def imagined_data(self, S):
        # a rollout reads each imagined state up to three times (step, next step's anchor, observe): FK once
        for S_c, data in self._data_cache:
            if S_c is S or all(a is b for a, b in zip(S_c, S)):
                return data
        data = self._wrap_data(ImaginedData(self.entity.data, self.kin, self.body_perm, S, self.env.scene.env_origins,
                                            self.com_b, self.com_q_rel, self._gravity_w, self._forward_b, self._fields))  # fmt: skip
        self._data_cache = [*self._data_cache[-2:], (S, data)]
        return data

    @contextmanager
    def imagined(self, S, action, prev_action, frames, anchor=None):
        """Inside the block the env's terms read state S, the given actions, and the reference at frames
        with relative poses anchored at ``anchor`` (default S). Yields the env view to call terms with."""
        e, cur, ent, am = self.env, self.cursor, self.entity, self.env.action_manager
        saved = (ent._data, cur.snapshot(), am._action, am._prev_action)
        torch.acos = _SafeAcos.apply
        try:
            cur.set_frames(frames)
            data = self.imagined_data(S)
            ent._data = data if anchor is None else self.imagined_data(anchor)
            cur.update_relative(self._to_wxyz, self._from_wxyz)
            ent._data = data
            am._action, am._prev_action = action, prev_action
            yield Delegate(e, scene=self._scene_view(data), _sm_t=self._t, _sm_to_wxyz=self._to_wxyz)
        finally:
            torch.acos = _ACOS
            ent._data, snap, am._action, am._prev_action = saved
            cur.restore(snap)

    def rewards(self, view):
        return {n: fn(view, **c.params) * c.weight * self.reward_scale for n, fn, c in self.reward_terms}

    def terminations(self, view):
        hard = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        alive = torch.ones(self.num_envs, device=self.device)
        for soft_fn, c in self.soft_terms:
            hard |= c.func(view, **c.params).bool()
            if soft_fn is not None:
                alive = alive * (1 - torch.sigmoid(soft_fn(view, **c.params) / self.term_tau))
        return hard, 1 - alive

    def _term_value(self, view, g, c):
        """One observation term on the view, as the manager post-processes it (noise, clip, scale)."""
        x = self._postprocess(c, c.func(view, **c.params), self.corrupt[g])
        if c.clip:
            x = x.clamp(c.clip[0], c.clip[1])
        if c.scale is not None:
            x = x * c.scale
        return x

    def rebuild_history(self, past: dict) -> dict:
        """Observation history buffers at a stored state, recomputed from the stored rows ending there (offset
        0 = the start). Rows before a reset are not part of the episode; like the managers' circular buffers
        after a reset, those slots repeat the oldest value that is."""
        H = self.history_length
        assert past is not None and past["s"].shape[0] >= H, "replay start needs the last history_length rows"
        vals = {}
        for j in range(H):  # newest first
            sd, ok = {k: v[j] for k, v in past["side"].items()}, past["valid"][j][:, None]
            with self.imagined(self.layout.unflatten(past["s"][j]), sd["last_action"], sd["prev_action"], sd["frame"]) as view:
                for g, terms in self.obs_terms.items():
                    for n, c in terms:
                        if c.history_length > 0:
                            x = self._term_value(view, g, c)
                            prev = vals.get((g, n))
                            vals.setdefault((g, n), []).append(x if prev is None else torch.where(ok, x, prev[-1]))
        hist = {g: {} for g in self.obs_terms}
        for (g, n), xs in vals.items():
            c = dict(self.obs_terms[g])[n]
            hist[g][n] = torch.stack(xs[: c.history_length][::-1], 1)  # oldest ... newest
        return hist

    def observe(self, view, hist, append: bool):
        """Observation groups on the view; history buffers advanced (append) or their newest frame replaced."""
        out, new_hist = {}, {}
        for key, g in self.obs_groups.items():
            parts, new_hist[g] = [], {}
            for n, c in self.obs_terms[g]:
                x = self._term_value(view, g, c)
                if c.history_length > 0:
                    h = hist[g][n]
                    h = torch.cat((h[:, 1:] if append else h[:, :-1], x[:, None]), 1)
                    new_hist[g][n] = h
                    x = h.reshape(h.shape[0], -1) if c.flatten_history_dim else h
                parts.append(x.reshape(x.shape[0], -1))
            out[key] = torch.cat(parts, -1)
        return out, new_hist

    def history(self):
        hb = self.env.observation_manager._group_obs_term_history_buffer
        return {g: {n: hb[g][n].buffer.clone() for n, c in terms if c.history_length > 0} for g, terms in self.obs_terms.items()}


class ManagerRollout(ImaginedRollout):
    def __init__(self, task: ManagerBasedTrackingTask, side: dict | None = None, past: dict | None = None):
        self.task = task
        task._data_cache = []
        self.xi = side["xi"] if side is not None and "xi" in side else task.get_context()
        if side is None:  # from the live envs
            self.t0 = task.cursor.frames.clone()
            self.last_action = task.env.action_manager.action.clone()
            self.prev_action = task.env.action_manager.prev_action.clone()
            self.hist = task.history()
        else:  # from stored states; observation histories are rebuilt from the stored rows before them
            self.t0, self.last_action, self.prev_action = side["frame"].clone(), side["last_action"].clone(), side["prev_action"].clone()
            self.hist = task.rebuild_history(past) if task.history_length > 0 else {g: {} for g in task.obs_terms}
        self.hist_k, self.k = 0, 0

    def observe(self, S):
        t = self.task
        with t.imagined(S, self.last_action, self.prev_action, self.t0 + self.k) as view:
            obs, hist = t.observe(view, self.hist, append=self.k > self.hist_k)
        if self.k > self.hist_k:
            self.hist, self.hist_k = hist, self.k
        return t.with_context(obs, self.xi)

    def step(self, S, action, S1, aux=None):
        t = self.task
        with t.imagined(S1, action, self.last_action, self.t0 + self.k, anchor=S) as view:
            terms = t.rewards(view)
            done, done_soft = t.terminations(view)
        self.prev_action, self.last_action = self.last_action, action
        self.k += 1
        return ImaginedStep(sum(terms.values()), done, done_soft, terms)
