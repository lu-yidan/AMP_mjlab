"""Diagnostic recovery rollout from baseline reference resets; not a benchmark."""
import argparse
import json
from dataclasses import asdict
from pathlib import Path
import imageio.v2 as imageio
import torch
import src.tasks
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from rsl_rl.runners.amp_on_policy_runner import _unpack_obs, _unpack_step

p = argparse.ArgumentParser()
p.add_argument('--checkpoint', required=True)
p.add_argument('--output', required=True)
a = p.parse_args()
out = Path(a.output); out.mkdir(parents=True, exist_ok=True)
task = 'Unitree-G1-MultiAMP-Flat'
cfg = load_env_cfg(task, play=True)
cfg.seed = 42
cfg.scene.num_envs = 64
cfg.viewer.width = 960; cfg.viewer.height = 640
cfg.viewer.max_extra_envs = 0
agent = load_rl_cfg(task)
env = ManagerBasedRlEnv(cfg=cfg, device='cuda:0', render_mode='rgb_array')
w = RslRlVecEnvWrapper(env, clip_actions=agent.clip_actions)
runner = load_runner_cls(task)(w, asdict(agent), log_dir=None, device='cuda:0')
runner.load(a.checkpoint, load_optimizer=False)
policy = runner.get_inference_policy(device='cuda:0')
obs, _ = _unpack_obs(w.get_observations())
robot = env.scene['robot']
def state():
    return robot.data.root_link_pos_w[:, 2] - env.scene.env_origins[:, 2], robot.data.projected_gravity_b
h, g = state()
initial_h, initial_g = h.clone(), g.clone()
fallen = (h < .5) | (g[:, 2] > -.342)
prone = (h < .5) & (g[:, 0] > .5)
supine = (h < .5) & (g[:, 0] < -.5)
alive = torch.ones(64, dtype=torch.bool, device='cuda:0')
hold = torch.zeros(64, dtype=torch.long, device='cuda:0')
success = torch.zeros_like(alive)
# Show a genuinely low initial state, if present; also record its first termination.
ids = prone.nonzero().flatten()
if not len(ids): ids = fallen.nonzero().flatten()
selected = int(ids[0]) if len(ids) else 0
env.cfg.viewer.env_idx = selected
writer = imageio.get_writer(out / 'recovery.mp4', fps=25)
first_done = None
with torch.inference_mode():
    for step in range(500):
        obs, _, done, _ = _unpack_step(*w.step(policy(obs)))
        alive &= ~done.bool()
        if first_done is None and bool(done[selected]): first_done = (step + 1) * env.step_dt
        h, g = state()
        stable = alive & (h > .65) & (g[:, 2] < -.9)
        hold = torch.where(stable, hold + 1, 0)
        success |= hold >= 50
        if step % 2 == 0: writer.append_data(env.render())
writer.close()
def report(mask):
    return {'count': int(mask.sum()), 'success': int((success & mask).sum())}
result = {'checkpoint': a.checkpoint, 'seed': 42, 'duration_s': 10,
          'success_definition': 'root height >0.65m and gravity_z <-0.9 for 1s before first reset',
          'all': report(torch.ones_like(fallen)), 'fallen': report(fallen),
          'prone_heuristic': report(prone), 'supine_heuristic': report(supine),
          'video_env': selected, 'video_initial_root_height': float(initial_h[selected]),
          'video_initial_gravity': initial_g[selected].tolist(),
          'video_first_reset_s': first_done, 'video_success': bool(success[selected])}
(out / 'summary.json').write_text(json.dumps(result, indent=2))
print(json.dumps(result, indent=2))
env.close()
