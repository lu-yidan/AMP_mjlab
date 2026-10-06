"""Read-only 23->29 HoST observation, control, MJCF and torque audit."""
import argparse
import hashlib
import json
import os
import re
from dataclasses import asdict
from pathlib import Path

os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco
import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper

from src.assets.robots.unitree_g1 import g1_23dof_constants as g23
from src.assets.robots.unitree_g1 import g1_constants_bp as g29
from src.tasks.host_recovery.config.g1.rl_cfg import unitree_g1_host_standup_ppo_runner_cfg
from src.tasks.host_recovery.minimal29 import ADDED_JOINTS, minimal29_env_cfg
from src.tasks.host_recovery.rl import HoSTOnPolicyRunner


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def model_names(model, kind, count):
    return [mujoco.mj_id2name(model, kind, i) for i in range(count)]


def movable_joint_names(spec):
    model = spec.compile()
    return [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
            for i in range(model.njnt)
            if model.jnt_type[i] != mujoco.mjtJoint.mjJNT_FREE]


def cfg_for_joint(articulation, name):
    matches = [cfg for cfg in articulation.actuators
               if any(re.fullmatch(expr, name) for expr in cfg.target_names_expr)]
    assert len(matches) == 1, (name, matches)
    cfg = matches[0]
    return {
        "class": type(cfg).__name__,
        "kp": float(cfg.stiffness),
        "kd": float(cfg.damping),
        "effort_limit": float(cfg.effort_limit),
        "armature": float(cfg.armature),
        "x1": getattr(cfg, "X1", None),
        "x2": getattr(cfg, "X2", None),
        "y1": getattr(cfg, "Y1", None),
        "y2": getattr(cfg, "Y2", None),
        "fs": getattr(cfg, "Fs", None),
        "fd": getattr(cfg, "Fd", None),
    }


def mjcf_comparison():
    m23, m29 = g23.get_spec().compile(), g29.get_spec().compile()
    j23 = movable_joint_names(g23.get_spec())
    j29 = movable_joint_names(g29.get_spec())
    shared = [name for name in j23 if name in j29]
    joint_rows = []
    for name in shared:
        a = mujoco.mj_name2id(m23, mujoco.mjtObj.mjOBJ_JOINT, name)
        b = mujoco.mj_name2id(m29, mujoco.mjtObj.mjOBJ_JOINT, name)
        joint_rows.append({
            "name": name,
            "index23": j23.index(name), "index29": j29.index(name),
            "range23": m23.jnt_range[a].tolist(), "range29": m29.jnt_range[b].tolist(),
            "axis23": m23.jnt_axis[a].tolist(), "axis29": m29.jnt_axis[b].tolist(),
            "actuator23": cfg_for_joint(g23.G1_23DOF_ARTICULATION, name),
            "actuator29": cfg_for_joint(g29.G1_ARTICULATION, name),
        })
    body23 = set(filter(None, model_names(m23, mujoco.mjtObj.mjOBJ_BODY, m23.nbody)))
    body29 = set(filter(None, model_names(m29, mujoco.mjtObj.mjOBJ_BODY, m29.nbody)))
    body_rows = []
    for name in sorted(body23 & body29):
        a = mujoco.mj_name2id(m23, mujoco.mjtObj.mjOBJ_BODY, name)
        b = mujoco.mj_name2id(m29, mujoco.mjtObj.mjOBJ_BODY, name)
        body_rows.append({
            "name": name,
            "mass23": float(m23.body_mass[a]), "mass29": float(m29.body_mass[b]),
            "inertia23": m23.body_inertia[a].tolist(),
            "inertia29": m29.body_inertia[b].tolist(),
        })
    return {
        "xml23": str(g23.G1_23DOF_XML), "xml29": str(g29.G1_XML),
        "xml23_sha256": sha256(g23.G1_23DOF_XML),
        "xml29_sha256": sha256(g29.G1_XML),
        "joint_names23": j23, "joint_names29": j29,
        "shared_joint_rows": joint_rows, "common_body_rows": body_rows,
    }


def tensor_stats(value):
    flat = value.detach().abs().float().flatten().cpu()
    return {"mean_abs": float(flat.mean()), "p95_abs": float(torch.quantile(flat, .95)),
            "max_abs": float(flat.max())}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--bank", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--num-envs", type=int, default=64)
    p.add_argument("--steps", type=int, default=200)
    args = p.parse_args()
    assert not args.output.exists()

    static = mjcf_comparison()
    shared_actuator_mismatches = []
    for row in static["shared_joint_rows"]:
        a23, a29 = row["actuator23"], row["actuator29"]
        mismatches = {k: (a23[k], a29[k]) for k in
                      ("class", "kp", "kd", "effort_limit", "armature")
                      if a23[k] != a29[k]}
        if mismatches:
            shared_actuator_mismatches.append({"joint": row["name"], **mismatches})
    assert not shared_actuator_mismatches, json.dumps(shared_actuator_mismatches, indent=2)
    names23, names29 = static["joint_names23"], static["joint_names29"]
    shared29 = [names29.index(name) for name in names23]
    mapping = []
    for history in range(6):
        frame = list(range(6))
        frame += [6 + i for i in shared29]
        frame += [35 + i for i in shared29]
        frame += [64 + i for i in shared29]
        frame += [93]
        assert len(frame) == 76
        mapping += [history * 94 + i for i in frame]

    saved = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    first = saved["actor_state_dict"]["mlp.0.weight"]
    last_w = saved["actor_state_dict"]["mlp.6.weight"]
    last_b = saved["actor_state_dict"]["mlp.6.bias"]
    added29 = [names29.index(name) for name in ADDED_JOINTS]
    unmapped = sorted(set(range(564)) - set(mapping))
    checkpoint_contract = {
        "sha256": sha256(args.checkpoint),
        "mapped_input_count": len(mapping), "unmapped_input_count": len(unmapped),
        "unmapped_first_layer_max_abs": float(first[:, unmapped].abs().max()),
        "added_output_weight_max_abs": float(last_w[added29].abs().max()),
        "added_output_bias_max_abs": float(last_b[added29].abs().max()),
        "shared_action_rows29": shared29, "added_action_rows29": added29,
    }

    cfg = minimal29_env_cfg(args.num_envs, 20261005, args.bank)
    cfg.observations["actor"].terms["host"].params["add_noise"] = False
    cfg.events.pop("foot_friction", None)
    cfg.events.pop("encoder_bias", None)
    cfg.events.pop("base_com", None)
    env = ManagerBasedRlEnv(cfg=cfg, device="cuda:0")
    wrapper = RslRlVecEnvWrapper(env)
    try:
        runner = HoSTOnPolicyRunner(wrapper, asdict(unitree_g1_host_standup_ppo_runner_cfg()),
                                    device="cuda:0")
        runner.load(str(args.checkpoint), load_optimizer=False, map_location="cuda:0")
        policy = runner.get_inference_policy(device="cuda:0")
        robot = env.scene["robot"]
        assert list(robot.joint_names) == names29
        term = env.action_manager.get_term("joint_pos")
        obs = wrapper.get_observations()
        initial_history = obs["actor"].reshape(args.num_envs, 6, 94)
        runtime = {
            "observation_shape": list(obs["actor"].shape),
            "history_static_max_error": float((initial_history - initial_history[:, :1]).abs().max()),
            "action_term_class": type(term).__name__,
            "configured_scale": float(term._scale),
            "joint_names": list(robot.joint_names),
        }
        samples = {key: [] for key in ("action", "target_error", "joint_velocity")}
        actuator_force_samples = []
        with torch.no_grad():
            for _ in range(args.steps):
                action = policy(obs["actor"])
                obs, _, _, _ = wrapper.step(action)
                samples["action"].append(action)
                samples["target_error"].append(robot.data.joint_pos_target - robot.data.joint_pos)
                actuator_force_samples.append(robot.data.actuator_force)
                samples["joint_velocity"].append(robot.data.joint_vel)
        stacked = {key: torch.stack(value) for key, value in samples.items()}
        per_joint = []
        for idx, name in enumerate(names29):
            row = {"index": idx, "name": name}
            for key, value in stacked.items():
                row[key] = tensor_stats(value[..., idx])
            per_joint.append(row)
        runtime["per_joint"] = per_joint
        actuator_force = torch.stack(actuator_force_samples)
        runtime["actuator_force"] = [
            {"index": idx, "joint_name": name, **tensor_stats(actuator_force[..., idx])}
            for idx, name in enumerate(robot.actuator_names)
        ]
    finally:
        env.close()

    report = {
        "checkpoint": checkpoint_contract,
        "mjcf": static,
        "shared_actuator_contract": "matched",  # assert above guarantees no mismatches
        "runtime": runtime,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))
    print("HOST23_TO_29_CONTRACT_AUDIT_DONE", args.output, flush=True)


if __name__ == "__main__":
    main()
