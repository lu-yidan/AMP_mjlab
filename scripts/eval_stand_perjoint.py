"""Per-joint upper-body velocity RMS diagnostic."""
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

task = "Unitree-G1-HoST-StandUp"
ckpt = os.environ["CKPT"]
num_envs = int(os.environ.get("NUM_ENVS", "256"))
steps = int(os.environ.get("STEPS", "6000"))
action_scale = float(os.environ.get("ACTION_SCALE", "0.25"))

configure_torch_backends()
device = "cuda:0" if torch.cuda.is_available() else "cpu"

env_cfg = load_env_cfg(task, play=True)
env_cfg.scene.num_envs = num_envs
env_cfg.actions["joint_pos"].scale = action_scale
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
sq_sum = torch.zeros(len(upper_names), device=device)
n = 0
with torch.no_grad():
    for step in range(steps):
        actions = policy(obs)
        obs, rew, dones, infos = env.step(actions)
        torso_z = asset.data.body_link_pos_w[:, torso_id, 2]
        feet_z = asset.data.site_pos_w[:, foot_ids_t, 2].mean(dim=1)
        standing = (torso_z - feet_z) > 0.6
        if standing.any():
            vel = asset.data.joint_vel[:, upper_ids_t]
            sq_sum += torch.sum(torch.square(vel[standing]), dim=0)
            n += int(standing.sum().item())
env.close()

rms = (sq_sum / max(n, 1)) ** 0.5
for name, r in zip(upper_names, rms.cpu().tolist()):
    print(f"{name}: {r:.5f}")
print(f"TOTAL_UPPER_RMS={torch.sqrt(sq_sum.sum() / max(n, 1)).item():.5f}")
print("[DONE]")
