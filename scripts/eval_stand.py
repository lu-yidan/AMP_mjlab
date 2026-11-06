"""Headless evaluation: does the policy stand up and stay up with zero pull force?"""
import os
os.environ.setdefault("MUJOCO_GL", "egl")
import torch
from dataclasses import asdict

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.utils.torch import configure_torch_backends

import mjlab.tasks  # noqa: F401
import src.tasks  # noqa: F401


def main():
    task = "Unitree-G1-HoST-StandUp"
    ckpt = os.environ["CKPT"]
    num_envs = int(os.environ.get("NUM_ENVS", "64"))
    steps = int(os.environ.get("STEPS", "3000"))
    log_every = int(os.environ.get("LOG_EVERY", "500"))

    configure_torch_backends()
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    env_cfg = load_env_cfg(task, play=True)
    env_cfg.scene.num_envs = num_envs
    if "ACTION_SCALE" in os.environ:
        env_cfg.actions["joint_pos"].scale = float(os.environ["ACTION_SCALE"])
    if "POSTURE" in os.environ:
        posture = os.environ["POSTURE"]
        env_cfg.events["reset_base"].params["posture"] = None if posture == "none" else posture
    agent_cfg = load_rl_cfg(task)

    env = ManagerBasedRlEnv(cfg=env_cfg, device=device, render_mode=None)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    runner_cls = load_runner_cls(task) or MjlabOnPolicyRunner
    runner = runner_cls(env, asdict(agent_cfg), device=device)
    runner.load(ckpt, load_optimizer=False)
    policy = runner.get_inference_policy(device=device)

    asset = env.unwrapped.scene["robot"]
    torso_id = asset.find_bodies("torso_link")[0][0]
    upper_names = (
        "waist_yaw_joint",
        "left_shoulder_pitch_joint", "left_shoulder_roll_joint",
        "left_shoulder_yaw_joint", "left_elbow_joint", "left_wrist_roll_joint",
        "right_shoulder_pitch_joint", "right_shoulder_roll_joint",
        "right_shoulder_yaw_joint", "right_elbow_joint", "right_wrist_roll_joint",
    )
    upper_ids = list(asset.find_joints(upper_names)[0])
    upper_ids_t = torch.tensor(upper_ids, device=device)
    foot_ids = list(asset.find_sites(("left_foot", "right_foot"))[0])
    foot_ids_t = torch.tensor(foot_ids, device=device)

    obs = env.get_observations()
    jitter_sq_sum = 0.0
    jitter_n = 0
    with torch.no_grad():
        for step in range(steps):
            actions = policy(obs)
            obs, rew, dones, infos = env.step(actions)
            torso_z = asset.data.body_link_pos_w[:, torso_id, 2]
            feet_z = asset.data.site_pos_w[:, foot_ids_t, 2].mean(dim=1)
            standing = (torso_z - feet_z) > 0.6
            if standing.any():
                vel = asset.data.joint_vel[:, upper_ids_t]  # (N, 11)
                vel_norm2 = torch.sum(torch.square(vel), dim=1)  # (N,)
                jitter_sq_sum += vel_norm2[standing].sum().item()
                jitter_n += int(standing.sum().item())
            if (step + 1) % log_every == 0 or step == steps - 1:
                torso_z = asset.data.body_link_pos_w[:, torso_id, 2]
                feet_z = asset.data.site_pos_w[:, foot_ids_t, 2].mean(dim=1)
                height = torso_z - feet_z
                grav_z = asset.data.projected_gravity_b[:, 2]
                print(
                    f"step={step + 1:5d} "
                    f"height_mean={height.mean().item():.4f} "
                    f"height_min={height.min().item():.4f} "
                    f"height_max={height.max().item():.4f} "
                    f"frac>0.6={(height > 0.6).float().mean().item():.3f} "
                    f"frac>0.7={(height > 0.7).float().mean().item():.3f} "
                    f"upright(grav_z<-0.8)={(grav_z < -0.8).float().mean().item():.3f} "
                    f"grav_z_mean={grav_z.mean().item():.3f}"
                )
    env.close()
    upper_rms = (jitter_sq_sum / max(jitter_n, 1)) ** 0.5
    print(f"UPPER_JOINT_VEL_RMS(standing, 11 upper joints)={upper_rms:.5f}")
    print("[DONE]")


if __name__ == "__main__":
    main()
