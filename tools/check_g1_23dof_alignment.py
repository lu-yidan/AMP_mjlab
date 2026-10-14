"""Check HoST's 23-DoF G1 against the sibling 29-DoF recovery model.

This uses only the Python standard library, so it works without mjlab or Isaac Gym.
"""

import argparse
import ast
import math
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
URDF = ROOT / "legged_gym/resources/robots/g1/g1_23dof.urdf"
CONFIG = ROOT / "legged_gym/legged_gym/envs/g1/g1_config_ground.py"
AMP_MODEL = ROOT.parent / "G1Recovery_AMP/src/g1recovery_amp/g1_model.py"
MJCF = ROOT.parent / "mjlab/src/mjlab/asset_zoo/robots/unitree_g1/xmls/g1.xml"
REMOVED_29DOF_JOINTS = {
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
}


def require(condition, message):
    if not condition:
        raise SystemExit("FAIL: " + message)


def class_named(parent, name):
    return next(
        n for n in parent.body if isinstance(n, ast.ClassDef) and n.name == name
    )


def assignment(parent, name):
    for node in parent.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == name
            for target in node.targets
        ):
            return node.value
        if (
            isinstance(node, ast.AnnAssign)
            and isinstance(node.target, ast.Name)
            and node.target.id == name
        ):
            return node.value
    raise SystemExit("FAIL: cannot find " + name)


def urdf_joints():
    root = ET.parse(URDF).getroot()
    joints = [
        joint
        for joint in root.findall("joint")
        if joint.get("type") not in ("fixed", "floating")
    ]
    names = [joint.get("name") for joint in joints]
    require(
        len(joints) == 23 and len(set(names)) == 23,
        "URDF must have 23 unique movable joints",
    )
    missing_meshes = [
        mesh.get("filename")
        for mesh in root.findall(".//mesh")
        if not (URDF.parent / mesh.get("filename")).is_file()
    ]
    require(not missing_meshes, "missing URDF meshes: " + str(missing_meshes))
    return joints


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--amp-model", type=Path, default=AMP_MODEL)
    parser.add_argument(
        "--mjcf",
        type=Path,
        default=MJCF,
        help="optional 29-DoF MuJoCo XML for axis and limit comparison",
    )
    args = parser.parse_args()

    joints = urdf_joints()
    names = [joint.get("name") for joint in joints]
    for joint in joints:
        name = joint.get("name")
        expected_velocity = (
            20.0
            if "knee" in name
            else 32.0
            if "hip" in name or "waist" in name
            else 37.0
        )
        actual_velocity = float(joint.find("limit").get("velocity"))
        require(
            actual_velocity == expected_velocity,
            name + " velocity limit differs from mjlab adaptation",
        )
        expected_effort = (
            139.0
            if "knee" in name
            else 88.0
            if "hip" in name or "waist" in name
            else 50.0
            if "ankle" in name
            else 25.0
        )
        require(
            float(joint.find("limit").get("effort")) == expected_effort,
            name + " torque limit differs from mjlab adaptation",
        )
    require(
        args.amp_model.is_file(), "29-DoF reference missing: " + str(args.amp_model)
    )
    amp_tree = ast.parse(args.amp_model.read_text(encoding="utf-8"))
    amp_names = ast.literal_eval(assignment(amp_tree, "G1_JOINT_NAMES"))
    require(
        len(amp_names) == 29 and len(set(amp_names)) == 29,
        "AMP reference must list 29 unique joints",
    )
    require(
        set(amp_names) - set(names) == REMOVED_29DOF_JOINTS,
        "the six joints removed from the 29-DoF model differ from expectation",
    )
    require(
        not set(names) - set(amp_names), "HoST has joints absent from the AMP model"
    )
    require(
        tuple(name for name in amp_names if name in names) == tuple(names),
        "HoST joint order differs from the 29-DoF model's filtered order",
    )

    cfg_tree = ast.parse(CONFIG.read_text(encoding="utf-8"))
    cfg = class_named(cfg_tree, "G1Cfg")
    init_state = class_named(cfg, "init_state")
    env = class_named(cfg, "env")
    asset = class_named(cfg, "asset")
    for key in ("target_joint_angles", "default_joint_angles"):
        angles = ast.literal_eval(assignment(init_state, key))
        require(
            set(angles) == set(names), key + " must contain exactly the 23 URDF joints"
        )
        for joint in joints:
            limits = joint.find("limit")
            lower, upper = float(limits.get("lower")), float(limits.get("upper"))
            require(
                lower <= angles[joint.get("name")] <= upper,
                key + " is outside the URDF limit for " + joint.get("name"),
            )
    require(
        ast.literal_eval(assignment(env, "num_actions")) == 23, "num_actions must be 23"
    )
    require(ast.literal_eval(assignment(env, "num_dofs")) == 23, "num_dofs must be 23")
    require(
        ast.literal_eval(assignment(env, "num_one_step_observations")) == 76,
        "one-step observation count must be 76 for 23 actions",
    )
    require(
        ast.literal_eval(assignment(env, "num_actor_history")) == 6,
        "actor history must contain current state and five previous states",
    )
    observations = assignment(env, "num_observations")
    require(
        isinstance(observations, ast.BinOp)
        and isinstance(observations.op, ast.Mult)
        and isinstance(observations.left, ast.Name)
        and isinstance(observations.right, ast.Name)
        and {observations.left.id, observations.right.id}
        == {"num_actor_history", "num_one_step_observations"},
        "num_observations must equal history times one-step observations",
    )
    require(
        ast.literal_eval(assignment(asset, "file")).endswith("/g1_23dof.urdf"),
        "ground task must load the 23-DoF URDF",
    )
    for node in asset.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id.endswith("_joints"):
                    referenced = ast.literal_eval(node.value)
                    require(
                        set(referenced) <= set(names),
                        target.id + " references a joint absent from the 23-DoF URDF",
                    )

    if args.mjcf.is_file():
        mjcf_joints = {
            joint.get("name"): joint
            for joint in ET.parse(args.mjcf).getroot().findall(".//joint")
            if joint.get("name")
        }
        require(set(names) <= set(mjcf_joints), "29-DoF MJCF is missing a HoST joint")
        for joint in joints:
            name = joint.get("name")
            other = mjcf_joints[name]
            urdf_axis = [float(x) for x in joint.find("axis").get("xyz").split()]
            mjcf_axis = [float(x) for x in other.get("axis").split()]
            require(
                urdf_axis == mjcf_axis, name + " axis differs between URDF and MJCF"
            )
            limits = joint.find("limit")
            urdf_range = [float(limits.get("lower")), float(limits.get("upper"))]
            mjcf_range = [float(x) for x in other.get("range").split()]
            require(
                all(
                    math.isclose(a, b, abs_tol=1e-4)
                    for a, b in zip(urdf_range, mjcf_range)
                ),
                name + " joint limits differ between URDF and MJCF",
            )

    print(
        "PASS: 23 URDF joints, 23 actions, 76 x 6 observations, all meshes and motor limits present"
    )
    print("PASS: AMP 29->23 names and order match; six omitted joints:")
    print("  " + ", ".join(sorted(REMOVED_29DOF_JOINTS)))
    print(
        "PASS: shared joint axes and limits match MJCF"
        if args.mjcf.is_file()
        else "SKIP: MJCF axis and limit comparison (file absent)"
    )
    print("URDF index  AMP index  Joint")
    for host_index, name in enumerate(names):
        print(f"{host_index:>10}  {amp_names.index(name):>9}  {name}")


if __name__ == "__main__":
    main()
