"""Isaac Lab adapter: a manager-based Isaac Lab env (``ManagerBasedRLEnv``, e.g. BeyondMimic's tracking
task) as a ``TrackingTask``.

    from supermimic.wrappers.isaaclab import IsaacLabTrackingTask
    task = IsaacLabTrackingTask(env, robot_model="path/to/robot.urdf")   # or an MJCF
    runner = SuperMimicRunner(task, train_cfg, log_dir, device)

See ``manager_based.py`` for how the env's own terms run on imagined states. Isaac Lab specifics:
  * forward kinematics comes from the robot's URDF or MJCF (the file the asset was spawned from), compiled
    by MuJoCo without geoms; joints and bodies are matched to the articulation by name;
  * data fields follow Isaac Lab's conventions (``root_pos_w`` / ``body_pos_w`` are link poses,
    ``root_lin_vel_w`` / ``body_lin_vel_w`` COM velocities);
  * reward terms that read sensors (contact forces) are skipped; observation terms must not read sensors;
  * observation modifiers and noise are applied as the manager applies them;
  * the DR context reads back what the events randomized: rigid-body material (static / dynamic friction,
    restitution of the targeted shapes), COM, mass, default joint positions (BeyondMimic's calibration
    offsets), actuator gains, joint friction / armature / limits; other events are warned about.
"""

from __future__ import annotations

import os
import xml.etree.ElementTree as ET

import mujoco
import torch

from ..utils.kinematics import TreeKinematics
from ..utils.rotations import quat_to_mat
from .manager_based import ManagerBasedTrackingTask


def mjmodel_from_robot_file(path: str) -> mujoco.MjModel:
    """A kinematics-only MjModel of a URDF or MJCF: geoms dropped, a free joint on the root body."""
    if path.endswith(".urdf"):
        root = ET.parse(path).getroot()
        for link in root.iter("link"):
            for tag in ("visual", "collision"):
                for el in link.findall(tag):
                    link.remove(el)
        for el in root.findall("mujoco"):  # Unitree's URDFs carry their own compiler block (mesh dirs we dropped)
            root.remove(el)
        ext = ET.SubElement(root, "mujoco")
        ET.SubElement(ext, "compiler", {"fusestatic": "false", "discardvisual": "true", "boundmass": "1e-6", "boundinertia": "1e-9"})
        spec = mujoco.MjSpec.from_string(ET.tostring(root, encoding="unicode"))
        body = spec.worldbody.first_body()
        if not any(j.type == mujoco.mjtJoint.mjJNT_FREE for j in body.joints):
            body.add_freejoint()
    else:
        spec = mujoco.MjSpec.from_file(path)
        spec.compiler.fusestatic = False
        for g in list(spec.geoms):
            spec.delete(g)
        body = spec.worldbody.first_body()
        if not any(j.type == mujoco.mjtJoint.mjJNT_FREE for j in body.joints):
            body.add_freejoint()  # a fixed-base MJCF (e.g. compiled from a URDF) still describes a floating robot here
    return spec.compile()


def _ids(ids, n):
    return list(range(n)) if ids is None or ids == slice(None) else list(ids)


def _asset(task, cfg):
    acfg = cfg.params.get("asset_cfg")
    return (task.env.scene[acfg.name] if acfg is not None else task.entity), acfg


def _material(task, cfg):
    asset, acfg = _asset(task, cfg)
    per_body = getattr(cfg.func, "num_shapes_per_body", None)  # set by the term when it targets a body subset
    shapes = slice(None)
    if per_body is not None:
        start = [sum(per_body[:b]) for b in range(len(per_body))]
        shapes = [start[b] + s for b in _ids(acfg.body_ids, asset.num_bodies) for s in range(per_body[b])]
    return lambda env: asset.root_physx_view.get_material_properties()[:, shapes]


def _com(task, cfg):
    asset, acfg = _asset(task, cfg)
    ids = _ids(acfg.body_ids, asset.num_bodies)
    return lambda env: asset.root_physx_view.get_coms()[:, ids, :3]


def _mass(task, cfg):
    asset, acfg = _asset(task, cfg)
    ids = _ids(acfg.body_ids, asset.num_bodies)
    return lambda env: asset.root_physx_view.get_masses()[:, ids]


def _default_joint_pos(task, cfg):
    asset, acfg = _asset(task, cfg)
    ids = _ids(acfg.joint_ids, asset.num_joints)
    return lambda env: asset.data.default_joint_pos[:, ids]


def _actuator_gains(task, cfg):
    asset, acfg = _asset(task, cfg)
    ids = _ids(acfg.joint_ids, asset.num_joints)

    def read(env):
        K = torch.zeros(env.num_envs, asset.num_joints, device=task.device)
        D = torch.zeros_like(K)
        for act in asset.actuators.values():
            K[:, act.joint_indices], D[:, act.joint_indices] = act.stiffness, act.damping
        return torch.cat((K[:, ids], D[:, ids]), -1)

    return read


def _joint_parameters(task, cfg):
    asset, acfg = _asset(task, cfg)
    ids, p, d = _ids(acfg.joint_ids, asset.num_joints), cfg.params, asset.data
    fields = []
    if p.get("friction_distribution_params") is not None:
        fields.append("joint_friction_coeff" if hasattr(d, "joint_friction_coeff") else "joint_friction")
    if p.get("armature_distribution_params") is not None:
        fields.append("joint_armature")
    if p.get("lower_limit_distribution_params") is not None or p.get("upper_limit_distribution_params") is not None:
        fields.append("joint_pos_limits")
    return lambda env: torch.cat([getattr(d, f)[:, ids].reshape(env.num_envs, -1) for f in fields], -1)


class IsaacLabTrackingTask(ManagerBasedTrackingTask):
    DEFAULT_OBS_GROUPS = (("policy", "policy"), ("critic", "critic"))
    CONTEXT_READERS = {
        "randomize_rigid_body_material": _material,
        "randomize_rigid_body_com": _com,
        "randomize_rigid_body_mass": _mass,
        "randomize_joint_default_pos": _default_joint_pos,
        "randomize_actuator_gains": _actuator_gains,
        "randomize_joint_parameters": _joint_parameters,
    }

    def __init__(self, env, robot_model: str, **kwargs):
        """robot_model: the URDF or MJCF the robot asset was spawned from."""
        if not os.path.exists(robot_model):
            raise FileNotFoundError(robot_model)
        self.robot_model = robot_model
        super().__init__(env, **kwargs)

    def _state_pre_reset(self):
        """Raw read through the PhysX view: Isaac Lab's root_link_* buffers are lazy caches stamped with the
        sim time, and a reset's writes do not refresh them, so reading them here (same stamp as the reads
        after the step) would leave the reset envs' state stale until the next physics step."""
        view = self.entity.root_physx_view
        tf, vel = view.get_root_transforms(), view.get_root_velocities()  # link pose (quat xyzw), COM velocity
        quat = self._to_wxyz(tf[:, 3:7])
        r_com = (quat_to_mat(quat) @ self.com_b[:, 0, :, None]).squeeze(-1)
        v_link = vel[:, :3] - torch.cross(vel[:, 3:], r_com, dim=-1)
        return torch.cat((tf[:, :3] - self.env.scene.env_origins, quat, v_link, vel[:, 3:], view.get_dof_positions(), view.get_dof_velocities()), -1)

    def _build_kinematics(self):
        m = mjmodel_from_robot_file(self.robot_model)
        kin = TreeKinematics(m, list(self.entity.joint_names), list(self.entity.body_names), self.device)
        missing = set(self.entity.body_names) - set(kin.body_names)
        assert not missing, f"bodies missing from {self.robot_model}: {sorted(missing)}"
        return kin, [kin.body_names.index(n) for n in self.entity.body_names]

    def _sensor_names(self, cfg):
        sensors = getattr(self.env.scene, "sensors", {}) or {}
        return [v.name for v in cfg.params.values() if getattr(v, "name", None) in sensors]

    def _build_sensors(self):
        for g, terms in self.obs_terms.items():
            for n, c in terms:
                if self._sensor_names(c):
                    raise NotImplementedError(f"observation term {g}/{n} reads a sensor; imagination cannot reproduce it")
        return {}

    def _reads_sensor(self, cfg):
        return bool(self._sensor_names(cfg))

    def _postprocess(self, cfg, x, corrupt):
        for mod in cfg.modifiers or ():
            x = mod.func(x, **mod.params)
        n = cfg.noise
        if corrupt and n is not None:
            # a NoiseModelCfg (has noise_cfg) holds an instantiated model called on x alone
            x = n.func(x) if hasattr(n, "noise_cfg") else n.func(x, n)
        return x

    def _current_obs(self):
        return self.env.obs_buf
