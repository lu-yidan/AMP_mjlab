"""23-DoF MuJoCo G1 using mjlab geometry and HoST actuator settings."""

from __future__ import annotations

import math
from pathlib import Path

import mujoco
from mjlab.actuator import BuiltinMotorActuatorCfg
from mjlab.asset_zoo.robots.unitree_g1.g1_constants import (
    ARMATURE_4010,
    ARMATURE_5020,
    get_g1_robot_cfg,
    get_spec,
)
from mjlab.entity import EntityCfg

from .action import HostActionCfg

REMOVED_JOINTS = frozenset(
    {
        "waist_roll_joint",
        "waist_pitch_joint",
        "left_wrist_pitch_joint",
        "left_wrist_yaw_joint",
        "right_wrist_pitch_joint",
        "right_wrist_yaw_joint",
    }
)

HOST_URDF = (
    Path(__file__).resolve().parents[2] / "legged_gym/resources/robots/g1/g1_23dof.urdf"
)

HOST_JOINT_NAMES = (
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
)

# The upstream Unitree/mjlab 29-DoF model uses this joint order.  The six
# additional joints keep their published MJCF limits, link inertias and meshes.
HOST29_JOINT_NAMES = (
    *HOST_JOINT_NAMES[:13],
    "waist_roll_joint",
    "waist_pitch_joint",
    *HOST_JOINT_NAMES[13:18],
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    *HOST_JOINT_NAMES[18:],
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
)

HOST_DEFAULT_JOINT_POS = {
    name: (
        -0.1
        if "hip_pitch" in name
        else 0.3
        if "knee" in name
        else -0.2
        if "ankle_pitch" in name
        else 0.8
        if "elbow" in name
        else 0.0
    )
    for name in HOST_JOINT_NAMES
}

HOST_JOINT_VEL_LIMITS = {
    name: (
        20.0 if "knee" in name else 32.0 if "hip" in name or "waist" in name else 37.0
    )
    for name in HOST_JOINT_NAMES
}

HOST_TARGET_JOINT_POS = {
    **{name: 0.0 for name in HOST_JOINT_NAMES},
    "left_hip_pitch_joint": -0.1,
    "right_hip_pitch_joint": -0.1,
    "left_knee_joint": 0.3,
    "right_knee_joint": 0.3,
    "left_ankle_pitch_joint": -0.2,
    "right_ankle_pitch_joint": -0.2,
    "left_shoulder_roll_joint": 0.3,
    "right_shoulder_roll_joint": -0.3,
}

# Final frame (429) of G1Recovery_mjlab's Final_ce1_29dof.npz.  Its arm
# posture places the hands at the sides.  Use symmetric flat-footed legs and
# a centered waist rather than copying the clip's small left/right residuals.
HOST29_STAND_ARM_JOINT_POS = {
    "left_shoulder_pitch_joint": 0.242,
    "left_shoulder_roll_joint": 0.094,
    "left_shoulder_yaw_joint": 0.160,
    "left_elbow_joint": 0.821,
    "left_wrist_roll_joint": 0.690,
    "left_wrist_pitch_joint": 0.0,
    "left_wrist_yaw_joint": 0.0,
    "right_shoulder_pitch_joint": 0.242,
    "right_shoulder_roll_joint": -0.094,
    "right_shoulder_yaw_joint": -0.160,
    "right_elbow_joint": 0.821,
    "right_wrist_roll_joint": -0.690,
    "right_wrist_pitch_joint": 0.0,
    "right_wrist_yaw_joint": 0.0,
}
HOST29_STAND_WAIST_JOINT_NAMES = (
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
)
HOST29_TARGET_JOINT_POS = {
    **{name: HOST_TARGET_JOINT_POS.get(name, 0.0) for name in HOST29_JOINT_NAMES},
    **HOST29_STAND_ARM_JOINT_POS,
    "waist_yaw_joint": 0.0,
    "waist_roll_joint": 0.0,
    "waist_pitch_joint": 0.0,
}


def get_host_spec() -> mujoco.MjSpec:
    """Lock the six joints absent from the physical 23-DoF HoST G1."""

    spec = get_spec()
    found = set()
    for joint in list(spec.joints):
        if joint.name in REMOVED_JOINTS:
            found.add(joint.name)
            spec.delete(joint)
    if found != REMOVED_JOINTS:
        raise ValueError(f"29-DoF source model lacks joints: {REMOVED_JOINTS - found}")
    return spec


def get_host_urdf_spec(replace_cylinders: bool = False) -> mujoco.MjSpec:
    """Load HoST's own URDF as a free-base MuJoCo spec, preserving link inertias."""

    if not HOST_URDF.is_file():
        raise FileNotFoundError(f"HoST 23-DoF URDF missing: {HOST_URDF}")
    xml = HOST_URDF.read_text(encoding="utf-8")
    mesh_dir = (HOST_URDF.parent / "meshes").resolve()
    # The upstream URDF contains both meshdir="meshes" and meshes/ in each
    # filename, which MuJoCo would otherwise combine into meshes/meshes/.
    xml = xml.replace('meshdir="meshes"', f'meshdir="{mesh_dir}"')
    xml = xml.replace('filename="meshes/', 'filename="')
    spec = mujoco.MjSpec.from_string(xml)
    if spec.bodies[1].name != "pelvis":
        raise ValueError("HoST URDF root body is no longer pelvis")
    spec.bodies[1].add_freejoint(name="root")
    # The URDF importer marks every imported geom non-colliding. Group 0
    # corresponds to URDF <collision>; group 1 to <visual>.
    for geom in spec.geoms:
        if geom.group == 0:
            geom.contype = 1
            geom.conaffinity = 1
            if replace_cylinders and geom.type == mujoco.mjtGeom.mjGEOM_CYLINDER:
                geom.type = mujoco.mjtGeom.mjGEOM_CAPSULE
    return spec


def get_host_urdf_capsule_spec() -> mujoco.MjSpec:
    """Match Isaac Gym's replace_cylinder_with_capsule asset option."""

    return get_host_urdf_spec(replace_cylinders=True)


def _motor_actuator(pattern: str, effort: float, armature: float = 0.01):
    return BuiltinMotorActuatorCfg(
        target_names_expr=(pattern,),
        effort_limit=effort,
        armature=armature,
    )


def get_host_robot_cfg(model_source: str = "mjlab") -> EntityCfg:
    """Use HoST's 23 motor groups, default pose and fallen root orientation."""

    cfg = get_g1_robot_cfg()
    if model_source == "mjlab":
        cfg.spec_fn = get_host_spec
    elif model_source in ("host_urdf", "host_urdf_capsule"):
        cfg.spec_fn = (
            get_host_urdf_capsule_spec
            if model_source == "host_urdf_capsule"
            else get_host_urdf_spec
        )
        # The stock 29-DoF G1 collision editor names different geoms and
        # disables all unmatched geoms, so it must not be applied to HoST URDF.
        cfg.collisions = ()
    else:
        raise ValueError(f"Unknown G1 model source: {model_source}")
    half_turn = math.sqrt(0.5)
    # HoST init_state.rot is (x,y,z,w)=(0,-1,0,1), normalized here
    # and converted to MuJoCo's (w,x,y,z) convention.
    cfg.init_state = EntityCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.5),
        rot=(half_turn, 0.0, -half_turn, 0.0),
        joint_pos=HOST_DEFAULT_JOINT_POS.copy(),
        joint_vel={".*": 0.0},
    )
    assert cfg.articulation is not None
    cfg.articulation.soft_joint_pos_limit_factor = 0.9
    cfg.articulation.actuators = (
        _motor_actuator(".*_hip_.*_joint", 88.0),
        _motor_actuator(".*_knee_joint", 139.0),
        _motor_actuator(".*_ankle_.*_joint", 50.0),
        _motor_actuator("waist_yaw_joint", 88.0),
        _motor_actuator(".*_shoulder_.*_joint", 25.0),
        _motor_actuator(".*_elbow_joint", 25.0),
        _motor_actuator(".*_wrist_roll_joint", 25.0),
    )
    return cfg


def get_host_action_cfg() -> HostActionCfg:
    """HoST's position target is current joint position plus beta times action."""

    return HostActionCfg(
        entity_name="robot",
        actuator_names=(".*",),
        scale=1.0,
        clip={".*": (-100.0, 100.0)},
    )


def get_host29_robot_cfg() -> EntityCfg:
    """29-DoF Unitree G1 with HoST effort control and six published extra axes.

    The four small wrist motors are 5 Nm and the two 4-bar waist axes 50 Nm,
    per Unitree's g1_29dof MJCF. All link inertias and joint stops come from
    mjlab's open-source 29-DoF G1 MJCF rather than the 23-DoF HoST URDF.
    """
    cfg = get_g1_robot_cfg()
    cfg.spec_fn = get_spec
    half_turn = math.sqrt(0.5)
    cfg.init_state = EntityCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.5),
        rot=(half_turn, 0.0, -half_turn, 0.0),
        joint_pos={
            **HOST_DEFAULT_JOINT_POS,
            **{
                name: 0.0 for name in HOST29_JOINT_NAMES if name not in HOST_JOINT_NAMES
            },
        },
        joint_vel={".*": 0.0},
    )
    assert cfg.articulation is not None
    cfg.articulation.soft_joint_pos_limit_factor = 0.9
    cfg.articulation.actuators = (
        _motor_actuator(".*_hip_.*_joint", 88.0),
        _motor_actuator(".*_knee_joint", 139.0),
        _motor_actuator(".*_ankle_.*_joint", 50.0),
        _motor_actuator("waist_yaw_joint", 88.0),
        _motor_actuator("waist_(roll|pitch)_joint", 50.0, ARMATURE_5020 * 2),
        _motor_actuator(".*_shoulder_.*_joint", 25.0),
        _motor_actuator(".*_elbow_joint", 25.0),
        _motor_actuator(".*_wrist_roll_joint", 25.0),
        _motor_actuator(".*_wrist_(pitch|yaw)_joint", 5.0, ARMATURE_4010),
    )
    return cfg
