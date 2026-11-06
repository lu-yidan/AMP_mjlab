"""Four-direction deterministic zero-assistance audit for minimal HoST29."""
import argparse
import json
import os
from dataclasses import asdict
from pathlib import Path

os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
os.environ.setdefault("MUJOCO_GL", "egl")

import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.utils.lab_api.math import quat_apply_inverse

from src.tasks.host_recovery.config.g1.rl_cfg import unitree_g1_host_standup_ppo_runner_cfg
from src.tasks.host_recovery.minimal29 import minimal29_env_cfg
from src.tasks.host_recovery.rl import HoSTOnPolicyRunner


NAMES = ("prone", "supine", "left_side", "right_side")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--bank", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--num-envs", type=int, default=256)
    p.add_argument("--steps", type=int, default=1000)
    a = p.parse_args()
    cfg = minimal29_env_cfg(a.num_envs, 20261006, a.bank)
    cfg.observations["actor"].terms["host"].params["add_noise"] = False
    cfg.events.pop("foot_friction", None)
    cfg.events.pop("encoder_bias", None)
    cfg.events.pop("base_com", None)
    agent = unitree_g1_host_standup_ppo_runner_cfg()
    env = ManagerBasedRlEnv(cfg=cfg, device="cuda:0")
    wrapper = RslRlVecEnvWrapper(env)
    try:
        runner = HoSTOnPolicyRunner(wrapper, asdict(agent), device="cuda:0")
        runner.load(str(a.checkpoint), load_optimizer=False, map_location="cuda:0")
        policy = runner.get_inference_policy(device="cuda:0")
        robot = env.scene["robot"]
        torso = robot.joint_names  # force model resolution before index setup
        torso_id = robot.find_bodies("torso_link")[0][0]
        foot_ids = torch.as_tensor(
            robot.find_sites(("left_foot", "right_foot"))[0], device=env.device)
        knee_ids = [robot.joint_names.index(n) for n in
                    ("left_knee_joint", "right_knee_joint")]
        gravity = torch.tensor((0., 0., -1.), device=env.device).expand(a.num_envs, 3)
        results = {}
        for direction, name in enumerate(NAMES):
            env._minimal29_direction.fill_(direction)
            wrapper.reset()
            obs = wrapper.get_observations()
            hold = torch.zeros(a.num_envs, dtype=torch.long, device=env.device)
            held1 = torch.zeros(a.num_envs, dtype=torch.bool, device=env.device)
            held5 = torch.zeros_like(held1)
            recovered = torch.zeros_like(held1)
            abnormal = torch.zeros_like(held1)
            sums = {k: 0. for k in ("height", "pelvis_upright", "torso_upright",
                    "knee", "width", "foreaft", "joint_rms", "base_speed",
                    "angular", "foot_rms", "action_delta")}
            samples = 0
            previous = None
            with torch.no_grad():
                for step in range(a.steps):
                    action = policy(obs["actor"])
                    obs, _, dones, _ = wrapper.step(action)
                    abnormal |= dones.bool() & (step + 1 < a.steps)
                    torso_pos = robot.data.body_link_pos_w[:, torso_id]
                    feet = robot.data.site_pos_w[:, foot_ids]
                    height = torso_pos[:, 2] - feet[..., 2].mean(-1)
                    pelvis_u = (-robot.data.projected_gravity_b[:, 2]).clamp(-1, 1)
                    torso_g = quat_apply_inverse(
                        robot.data.body_link_quat_w[:, torso_id], gravity)
                    torso_u = (-torso_g[:, 2]).clamp(-1, 1)
                    knees = robot.data.joint_pos[:, knee_ids].abs().amax(-1)
                    delta = feet[:, 0] - feet[:, 1]
                    local = quat_apply_inverse(robot.data.root_link_quat_w, delta)
                    width, foreaft = local[:, 1].abs(), local[:, 0].abs()
                    joint = robot.data.joint_vel.square().mean(-1).sqrt()
                    base = robot.data.root_link_lin_vel_w.norm(dim=-1)
                    angular = robot.data.root_link_ang_vel_w.norm(dim=-1)
                    foot = robot.data.site_lin_vel_w[:, foot_ids].square().mean((1, 2)).sqrt()
                    action_delta = (torch.zeros_like(action) if previous is None else
                                    action - previous).square().mean(-1).sqrt()
                    previous = action
                    recovered |= (height >= .60) & (pelvis_u >= .80)
                    strict = ((height >= .75) & (pelvis_u >= .95) & (torso_u >= .95)
                              & (knees <= .30) & (width >= .18) & (width <= .35)
                              & (foreaft <= .12) & (joint < .5) & (base < .15)
                              & (angular < .3) & (foot < .1) & ~abnormal)
                    hold = torch.where(strict, hold + 1, 0)
                    held1 |= hold >= 50
                    held5 |= hold >= 250
                    if step >= max(0, a.steps - 250):
                        values = (height, pelvis_u, torso_u, knees, width, foreaft,
                                  joint, base, angular, foot, action_delta)
                        for key, value in zip(sums, values):
                            sums[key] += float(value.mean())
                        samples += 1
            metrics = {key: value / samples for key, value in sums.items()}
            results[name] = dict(
                recovered=float(recovered.float().mean()),
                hold1=float(held1.float().mean()), hold5=float(held5.float().mean()),
                abnormal=float(abnormal.float().mean()), metrics=metrics)
            print(name, json.dumps(results[name], sort_keys=True), flush=True)
        a.output.mkdir(parents=True, exist_ok=False)
        (a.output / "evaluation.json").write_text(json.dumps({"results": results}, indent=2))
    finally:
        env.close()


if __name__ == "__main__":
    main()
