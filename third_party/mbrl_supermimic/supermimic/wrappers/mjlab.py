"""mjlab adapter: a manager-based mjlab env (``ManagerBasedRlEnv``) as a ``TrackingTask``.

    from supermimic.wrappers.mjlab import MjlabTrackingTask
    task = MjlabTrackingTask(env)                    # env = ManagerBasedRlEnv(cfg, device) or its rsl_rl wrapper
    runner = SuperMimicRunner(task, train_cfg, log_dir, device)

See ``manager_based.py`` for how the env's own terms run on imagined states. mjlab specifics:
  * forward kinematics comes from the env's compiled MjModel (same joint and body order as the entity);
  * site velocimeters and gyros read by observation terms are recomputed from the imagined kinematics;
  * the self-collision cost (a contact-sensor count) is replaced by a differentiable capsule-distance
    proxy; other contact-sensor terms are skipped (or pass ``skip_rewards``);
  * observation delays and noise models are not supported (plain noise configs are);
  * the DR context reads back the model fields mjlab's ``dr`` events randomize (per-world ``env.sim.model``
    arrays: friction, COM offset, mass, joint damping / armature / friction / offsets ...) and the
    encoder bias; ``pd_gains`` / ``effort_limits`` / ``pseudo_inertia`` have no reader yet (warned);
  * mjlab's rewards and terminations read body poses one physics substep stale (see its step()), while
    imagination uses current ones, so verify_task reports small nonzero reward errors on fast motions.
"""

from __future__ import annotations

import mujoco
import torch

from ..utils.collision import SelfCollisionProxy
from ..utils.kinematics import TreeKinematics
from ..utils.rotations import quat_to_mat
from .manager_based import Delegate, ManagerBasedTrackingTask, rot_apply_inv


class _SiteSensor:
    """A velocimeter or gyro on a site, recomputed from the imagined body kinematics."""

    KINDS = {mujoco.mjtSensor.mjSENS_VELOCIMETER: "vel", mujoco.mjtSensor.mjSENS_GYRO: "gyro"}

    def __init__(self, m: mujoco.MjModel, name: str, kin: TreeKinematics, device):
        sid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SENSOR, name)
        if sid < 0 or m.sensor_type[sid] not in self.KINDS or m.sensor_objtype[sid] != mujoco.mjtObj.mjOBJ_SITE:
            raise NotImplementedError(f"sensor '{name}': only site velocimeters and gyros can be imagined")
        site = m.sensor_objid[sid]
        self.kind = self.KINDS[m.sensor_type[sid]]
        self.body = kin.body_names.index(m.body(m.site_bodyid[site]).name)
        f32 = dict(dtype=torch.float32, device=device)
        self.pos = torch.tensor(m.site_pos[site], **f32)
        self.R = quat_to_mat(torch.tensor(m.site_quat[site], **f32))

    def read_fields(self, Rs, V, W):
        """From body rotations (N, B, 3, 3) and link-origin linear / angular velocities (N, B, 3)."""
        Rb = Rs[:, self.body]
        Rsite = Rb @ self.R
        Wb = W[:, self.body]
        if self.kind == "gyro":
            return rot_apply_inv(Rsite, Wb)
        Vs = V[:, self.body] + torch.cross(Wb, (Rb @ self.pos.unsqueeze(-1)).squeeze(-1), dim=-1)
        return rot_apply_inv(Rsite, Vs)

    def read(self, data):
        return self.read_fields(data._Rs, data.body_link_lin_vel_w, data.body_link_ang_vel_w)


_SENSOR_VIEWS: dict[type, type] = {}


def _sensor_view(real, data):
    """A shallow copy of a sensor (still an instance of its class, which terms assert) reading ``data``."""
    cls = type(real)
    if cls not in _SENSOR_VIEWS:
        _SENSOR_VIEWS[cls] = type(f"Imagined{cls.__name__}", (cls,), {"data": property(lambda s: s._imagined_data)})
    view = object.__new__(_SENSOR_VIEWS[cls])
    view.__dict__.update(real.__dict__)
    view._imagined_data = data
    return view


class _Scene(Delegate):
    def __getitem__(self, name):
        if name in self.sensors:
            return _sensor_view(self._inner[name], self.sensors[name].read(self.robot_data))
        return self._inner[name]


def self_collision_proxy(task, sensor_name: str) -> tuple[int, SelfCollisionProxy]:
    """(history length, proxy) for the env's self-collision contact sensor. ``self_collision_cost`` counts the
    sensor-history substeps with a self contact; in imagination it is history_length * the proxy's soft
    any-contact indicator of the capsule/sphere geoms (``proxy.soft_contact(body_pos_w, body_rotations)``)."""
    sensor = next(s for s in task.env.cfg.scene.sensors if s.name == sensor_name)
    H = max(getattr(sensor, "history_length", 0) or 1, 1)
    q0 = task.entity.data.default_joint_pos[:1]
    eye, zero = torch.eye(3, device=task.device)[None], torch.zeros(1, 3, device=task.device)
    P, R, _, _ = task.kin.forward(zero, eye, zero, zero, q0, torch.zeros_like(q0), all_bodies=True)
    return H, SelfCollisionProxy(task.env.sim.mj_model, task.kin.body_names, task.device, ref_pose=(P[0], R[0]))


def _self_collision_proxy(task, cfg):
    H, proxy = self_collision_proxy(task, cfg.params["sensor_name"])
    entity = task.entity_name

    def fn(env, sensor_name, force_threshold=10.0):
        data = env.scene[entity].data
        return H * proxy.soft_contact(data.body_link_pos_w, data._Rs)

    return fn


def _model_field(field, entity_type, default_axes=None, use_address=False):
    """Reader of a model field a ``mjlab.envs.mdp.dr`` event randomizes, at the entities and axes it targets."""

    def factory(task, cfg):
        from mjlab.envs.mdp.dr._core import _get_entity_indices
        from mjlab.managers.scene_entity_config import SceneEntityCfg

        p = cfg.params
        acfg = p.get("asset_cfg") or SceneEntityCfg(task.entity_name)
        idx = _get_entity_indices(task.env.scene[acfg.name].indexing, acfg, entity_type, use_address)
        if p.get("shared_random"):  # one value shared by all targeted entities
            idx = idx[:1]
        r = p.get("ranges")
        axes = p.get("axes") or (sorted(r) if isinstance(r, dict) and r and isinstance(next(iter(r)), int) else default_axes)

        def read(env):
            x = torch.as_tensor(getattr(env.sim.model, field))[:, idx]
            return x[..., axes] if axes is not None and x.dim() == 3 else x

        return read

    return factory


def _encoder_bias(task, cfg):
    acfg = cfg.params.get("asset_cfg")
    ids = slice(None) if acfg is None else acfg.joint_ids
    entity = task.env.scene[acfg.name] if acfg is not None else task.entity
    return lambda env: entity.data.encoder_bias[:, ids]


class MjlabTrackingTask(ManagerBasedTrackingTask):
    DEFAULT_OBS_GROUPS = (("policy", "actor"), ("critic", "critic"))
    IMAGINED_REWARDS = {"self_collision_cost": _self_collision_proxy}
    CONTEXT_READERS = {
        "geom_friction": _model_field("geom_friction", "geom", [0]),
        "body_com_offset": _model_field("body_ipos", "body"),
        "body_ipos": _model_field("body_ipos", "body"),
        "body_mass": _model_field("body_mass", "body"),
        "body_pos": _model_field("body_pos", "body"),
        "body_quat": _model_field("body_quat", "body"),
        "geom_pos": _model_field("geom_pos", "geom"),
        "geom_quat": _model_field("geom_quat", "geom"),
        "geom_size": _model_field("geom_size", "geom"),
        "joint_damping": _model_field("dof_damping", "dof", use_address=True),
        "joint_armature": _model_field("dof_armature", "dof", use_address=True),
        "joint_friction": _model_field("dof_frictionloss", "dof", use_address=True),
        "joint_stiffness": _model_field("jnt_stiffness", "joint"),
        "joint_limits": _model_field("jnt_range", "joint"),
        "joint_default_pos": _model_field("qpos0", "joint", use_address=True),
        "tendon_damping": _model_field("tendon_damping", "tendon"),
        "tendon_stiffness": _model_field("tendon_stiffness", "tendon"),
        "tendon_friction": _model_field("tendon_frictionloss", "tendon"),
        "tendon_armature": _model_field("tendon_armature", "tendon"),
        "encoder_bias": _encoder_bias,
    }
    CONTEXT_IGNORED = ManagerBasedTrackingTask.CONTEXT_IGNORED + ("cam_", "light_", "mat_", "geom_rgba", "geom_matid", "site_")

    def _build_kinematics(self):
        pre = f"{self.entity_name}/"
        bodies = [pre + n for n in self.entity.body_names]
        kin = TreeKinematics(self.env.sim.mj_model, [pre + n for n in self.entity.joint_names], bodies, self.device)
        assert kin.body_names == bodies, "FK body order differs from the entity"
        return kin, None

    def _build_sensors(self):
        sensors = {}
        for terms in self.obs_terms.values():
            for _, c in terms:
                name = c.params.get("sensor_name")
                if name is not None and name not in sensors:
                    sensors[name] = _SiteSensor(self.env.sim.mj_model, name, self.kin, self.device)
        return sensors

    def _reads_sensor(self, cfg):
        return "sensor_name" in cfg.params and cfg.params["sensor_name"] not in self.sensors

    def _scene_view(self, data):
        return _Scene(self.env.scene, sensors=self.sensors, robot_data=data)
