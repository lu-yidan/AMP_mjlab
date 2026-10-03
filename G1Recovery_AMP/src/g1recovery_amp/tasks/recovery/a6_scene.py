"""Physical entities for the historical A6 three-scene recovery protocol.

Both plates exist in every MuJoCo world. The reset term parks inactive plates
far from the robot and activates exactly one of flat, guided-plate, or
free-plate recovery for each environment.
"""

from __future__ import annotations

import mujoco
from mjlab.entity import EntityCfg
from mjlab.sensor.contact_sensor import ContactMatch, ContactSensorCfg

A6_PLATE_HALF_SIZE = (0.45, 0.32, 0.035)
A6_PLATE_FRICTION = (1.2, 0.01, 0.001)
A6_PLATE_NOMINAL_MASS = 8.0
A6_PLATE_CLEARANCE = 0.002
A6_ROBOT_COLLISION_PATTERN = r".*_collision"


def guided_plate_spec() -> mujoco.MjSpec:
    """Build the vertically guided A6 plate."""

    spec = mujoco.MjSpec()
    body = spec.worldbody.add_body(name="escape_plate")
    body.add_joint(
        name="escape_plate_slide",
        type=mujoco.mjtJoint.mjJNT_SLIDE,
        axis=(0.0, 0.0, 1.0),
        limited=True,
        range=(-1.2, 0.0),
        damping=60.0,
    )
    geom = body.add_geom(
        name="escape_plate_geom",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=A6_PLATE_HALF_SIZE,
        mass=A6_PLATE_NOMINAL_MASS,
        friction=A6_PLATE_FRICTION,
        rgba=(0.12, 0.72, 0.24, 0.8),
    )
    geom.priority = 1
    geom.solref = (0.01, 1.0)
    geom.solimp = (0.98, 0.995, 0.001, 0.5, 2.0)
    return spec


def free_plate_spec() -> mujoco.MjSpec:
    """Build the passive six-degree-of-freedom A6 plate."""

    spec = mujoco.MjSpec()
    body = spec.worldbody.add_body(name="plate")
    body.add_freejoint(name="free_plate_joint")
    body.add_geom(
        name="plate_geom",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=A6_PLATE_HALF_SIZE,
        mass=A6_PLATE_NOMINAL_MASS,
        friction=A6_PLATE_FRICTION,
        rgba=(0.85, 0.45, 0.12, 0.8),
        solref=(0.01, 1.0),
    )
    return spec


def plate_entity_cfgs() -> dict[str, EntityCfg]:
    """Return both plate entities, initially parked away from the robot."""

    return {
        "escape_obstacle": EntityCfg(
            spec_fn=guided_plate_spec,
            init_state=EntityCfg.InitialStateCfg(
                pos=(20.0, 20.0, 0.8),
                joint_pos={"escape_plate_slide": 0.0},
                joint_vel={"escape_plate_slide": 0.0},
            ),
        ),
        "free_obstacle": EntityCfg(
            spec_fn=free_plate_spec,
            init_state=EntityCfg.InitialStateCfg(pos=(22.0, 20.0, 0.1)),
        ),
    }


def plate_contact_sensor_cfgs() -> tuple[ContactSensorCfg, ...]:
    """Measure plate contact and hand support without changing policy inputs."""

    primary = ContactMatch(
        mode="geom",
        pattern=A6_ROBOT_COLLISION_PATTERN,
        entity="robot",
    )
    return (
        ContactSensorCfg(
            name="guided_contact",
            primary=primary,
            secondary=ContactMatch(
                mode="body", pattern="escape_plate", entity="escape_obstacle"
            ),
            fields=("found", "force", "dist"),
            reduce="mindist",
            num_slots=1,
        ),
        ContactSensorCfg(
            name="free_contact",
            primary=primary,
            secondary=ContactMatch(
                mode="body", pattern="plate", entity="free_obstacle"
            ),
            fields=("found", "force", "dist"),
            reduce="mindist",
            num_slots=1,
        ),
        ContactSensorCfg(
            name="path_hands",
            primary=ContactMatch(
                mode="geom",
                pattern=r"(left|right)_hand_collision$",
                entity="robot",
            ),
            secondary=ContactMatch(mode="body", pattern="terrain"),
            fields=("found", "force"),
            reduce="maxforce",
            num_slots=1,
        ),
        ContactSensorCfg(
            name="quality_feet",
            primary=ContactMatch(
                mode="body",
                pattern=("left_ankle_roll_link", "right_ankle_roll_link"),
                entity="robot",
            ),
            secondary=ContactMatch(mode="geom", pattern="terrain"),
            fields=("force",),
            reduce="netforce",
            num_slots=1,
        ),
    )


__all__ = [
    "A6_PLATE_CLEARANCE",
    "A6_PLATE_FRICTION",
    "A6_PLATE_HALF_SIZE",
    "A6_PLATE_NOMINAL_MASS",
    "A6_ROBOT_COLLISION_PATTERN",
    "free_plate_spec",
    "guided_plate_spec",
    "plate_contact_sensor_cfgs",
    "plate_entity_cfgs",
]
