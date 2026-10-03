"""Structural contract between the IsaacLab and mjlab Unitree G1 models.

The source motion files, MuJoCo joints, and MuJoCo actuators use three different
orders.  Conversions therefore use names instead of assuming that column ``i``,
``qpos[7 + i]``, and ``ctrl[i]`` all refer to the same joint.
"""

from copy import deepcopy
from dataclasses import dataclass

import mujoco
from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.asset_zoo.robots import get_g1_robot_cfg
from mjlab.entity import EntityCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg

G1_JOINT_NAMES: tuple[str, ...] = (
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
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
)

# Column order written by G1Recovery's ``gmr_to_lab.py``.  Its retargeting YAML
# calls this ``lab_dof_names``.  It is not the Unitree SDK/MuJoCo joint order.
G1_SOURCE_MOTION_JOINT_NAMES: tuple[str, ...] = (
    "left_hip_pitch_joint",
    "right_hip_pitch_joint",
    "waist_yaw_joint",
    "left_hip_roll_joint",
    "right_hip_roll_joint",
    "waist_roll_joint",
    "left_hip_yaw_joint",
    "right_hip_yaw_joint",
    "waist_pitch_joint",
    "left_knee_joint",
    "right_knee_joint",
    "left_shoulder_pitch_joint",
    "right_shoulder_pitch_joint",
    "left_ankle_pitch_joint",
    "right_ankle_pitch_joint",
    "left_shoulder_roll_joint",
    "right_shoulder_roll_joint",
    "left_ankle_roll_joint",
    "right_ankle_roll_joint",
    "left_shoulder_yaw_joint",
    "right_shoulder_yaw_joint",
    "left_elbow_joint",
    "right_elbow_joint",
    "left_wrist_roll_joint",
    "right_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "right_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_wrist_yaw_joint",
)

# ``target_joint_pos = source_joint_pos[:, G1_SOURCE_MOTION_TO_JOINT_ORDER]``.
G1_SOURCE_MOTION_TO_JOINT_ORDER: tuple[int, ...] = tuple(
    G1_SOURCE_MOTION_JOINT_NAMES.index(name) for name in G1_JOINT_NAMES
)

G1_BASE_BODY_NAME = "torso_link"
G1_RECOVERY_ACTION_SCALE = 0.25

G1_RECOVERY_DEFAULT_JOINT_POS: dict[str, float] = {
    "left_hip_pitch_joint": -0.1,
    "right_hip_pitch_joint": -0.1,
    ".*_knee_joint": 0.3,
    ".*_ankle_pitch_joint": -0.2,
    ".*_shoulder_pitch_joint": 0.3,
    "left_shoulder_roll_joint": 0.25,
    "right_shoulder_roll_joint": -0.25,
    ".*_elbow_joint": 0.97,
    "left_wrist_roll_joint": 0.15,
    "right_wrist_roll_joint": -0.15,
}

G1_KEY_BODY_NAMES: tuple[str, ...] = (
    "left_ankle_roll_link",
    "right_ankle_roll_link",
    "left_wrist_yaw_link",
    "right_wrist_yaw_link",
    "left_shoulder_roll_link",
    "right_shoulder_roll_link",
)


@dataclass(frozen=True)
class G1ModelMapping:
    """Name-based indexing extracted from a compiled MuJoCo model."""

    joint_names: tuple[str, ...]
    actuator_joint_names: tuple[str, ...]
    joint_order_to_ctrl: tuple[int, ...]
    body_names: tuple[str, ...]


def get_recovery_g1_robot_cfg() -> EntityCfg:
    """Return a fresh G1 config with the source recovery controller semantics."""

    cfg = deepcopy(get_g1_robot_cfg())
    cfg.init_state = EntityCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.8),
        joint_pos=deepcopy(G1_RECOVERY_DEFAULT_JOINT_POS),
        joint_vel={".*": 0.0},
    )
    assert cfg.articulation is not None
    cfg.articulation.actuators = (
        BuiltinPositionActuatorCfg(
            target_names_expr=(".*_hip_pitch_joint", ".*_hip_yaw_joint"),
            stiffness=100.0,
            damping=2.0,
            effort_limit=88.0,
            armature=0.01,
        ),
        BuiltinPositionActuatorCfg(
            target_names_expr=("waist_yaw_joint",),
            stiffness=200.0,
            damping=5.0,
            effort_limit=88.0,
            armature=0.01,
        ),
        BuiltinPositionActuatorCfg(
            target_names_expr=(".*_hip_roll_joint",),
            stiffness=100.0,
            damping=2.0,
            effort_limit=139.0,
            armature=0.01,
        ),
        BuiltinPositionActuatorCfg(
            target_names_expr=(".*_knee_joint",),
            stiffness=150.0,
            damping=4.0,
            effort_limit=139.0,
            armature=0.01,
        ),
        BuiltinPositionActuatorCfg(
            target_names_expr=(".*_ankle_.*",),
            stiffness=40.0,
            damping=2.0,
            effort_limit=25.0,
            armature=0.01,
        ),
        BuiltinPositionActuatorCfg(
            target_names_expr=("waist_roll_joint", "waist_pitch_joint"),
            stiffness=40.0,
            damping=5.0,
            effort_limit=25.0,
            armature=0.01,
        ),
        BuiltinPositionActuatorCfg(
            target_names_expr=(
                ".*_shoulder_.*",
                ".*_elbow_joint",
                ".*_wrist_roll_joint",
            ),
            stiffness=40.0,
            damping=1.0,
            effort_limit=25.0,
            armature=0.01,
        ),
        BuiltinPositionActuatorCfg(
            target_names_expr=(".*_wrist_pitch_joint", ".*_wrist_yaw_joint"),
            stiffness=40.0,
            damping=1.0,
            effort_limit=5.0,
            armature=0.01,
        ),
    )
    return cfg


def get_recovery_joint_position_action_cfg() -> JointPositionActionCfg:
    """Configure source-compatible position actions for all 29 G1 joints.

    A raw policy action ``a`` requests ``q_target = q_default + 0.25 * a``.
    The mjlab action term resolves actuator names back into natural joint order.
    """

    return JointPositionActionCfg(
        entity_name="robot",
        actuator_names=(".*",),
        scale=G1_RECOVERY_ACTION_SCALE,
        use_default_offset=True,
    )


def inspect_g1_model(
    model: mujoco.MjModel,  # pyright: ignore[reportAttributeAccessIssue]
) -> G1ModelMapping:
    """Inspect G1 joint/body names and derive SDK-joint-order to ``ctrl`` mapping."""

    joint_names = tuple(
        model.joint(joint_id).name
        for joint_id in range(model.njnt)
        # MuJoCo exposes enums through a native module without Python stubs.
        if model.jnt_type[joint_id]
        != mujoco.mjtJoint.mjJNT_FREE  # pyright: ignore[reportAttributeAccessIssue]
    )

    actuator_joint_names: list[str] = []
    for actuator_id in range(model.nu):
        if model.actuator_trntype[
            actuator_id
        ] != mujoco.mjtTrn.mjTRN_JOINT:  # pyright: ignore[reportAttributeAccessIssue]
            raise ValueError(f"G1 actuator {actuator_id} is not a joint transmission")
        joint_id = int(model.actuator_trnid[actuator_id, 0])
        actuator_joint_names.append(model.joint(joint_id).name)

    missing_actuators = set(joint_names) - set(actuator_joint_names)
    if missing_actuators:
        raise ValueError(f"G1 joints without actuators: {sorted(missing_actuators)}")

    joint_order_to_ctrl = tuple(
        actuator_joint_names.index(joint_name) for joint_name in joint_names
    )
    body_names = tuple(model.body(body_id).name for body_id in range(model.nbody))

    return G1ModelMapping(
        joint_names=joint_names,
        actuator_joint_names=tuple(actuator_joint_names),
        joint_order_to_ctrl=joint_order_to_ctrl,
        body_names=body_names,
    )


def validate_g1_model(
    model: mujoco.MjModel,  # pyright: ignore[reportAttributeAccessIssue]
) -> G1ModelMapping:
    """Validate the structural assumptions needed by recovery-task migration."""

    mapping = inspect_g1_model(model)

    if mapping.joint_names != G1_JOINT_NAMES:
        raise ValueError(
            "mjlab G1 joint order differs from the IsaacLab/Unitree SDK order"
        )
    if len(mapping.actuator_joint_names) != len(G1_JOINT_NAMES):
        raise ValueError(
            f"expected 29 G1 actuators, found {len(mapping.actuator_joint_names)}"
        )
    if len(set(mapping.actuator_joint_names)) != len(G1_JOINT_NAMES):
        raise ValueError("G1 actuators do not map one-to-one onto the 29 joints")

    required_bodies = {G1_BASE_BODY_NAME, *G1_KEY_BODY_NAMES}
    missing_bodies = required_bodies - set(mapping.body_names)
    if missing_bodies:
        raise ValueError(f"required G1 bodies are missing: {sorted(missing_bodies)}")

    return mapping
