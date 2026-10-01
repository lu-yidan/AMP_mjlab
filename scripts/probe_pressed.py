"""Short physics validation of the load task, using a fixed recovery checkpoint."""
import os
import json
import torch
from dataclasses import asdict
import mjlab.tasks
import src.tasks
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.utils.torch import configure_torch_backends
from src.tasks.host_recovery.pressed import load_force, released_standing

configure_torch_backends()
cfg=load_env_cfg('Unitree-G1-HoST-Pressed',play=True)
cfg.scene.num_envs=16
env=ManagerBasedRlEnv(cfg=cfg,device='cuda:0')
wrapped=RslRlVecEnvWrapper(env)
agent=load_rl_cfg('Unitree-G1-HoST-Pressed')
runner=load_runner_cls('Unitree-G1-HoST-Pressed')(wrapped,asdict(agent),device='cuda:0')
runner.load(os.environ['CKPT'],load_optimizer=False)
policy=runner.get_inference_policy(device='cuda:0')
obs=wrapped.get_observations()
with torch.no_grad():
    for step in range(600):
        obs,reward,done,info=wrapped.step(policy(obs))
        assert torch.isfinite(reward).all(), 'Nonfinite reward'
        if step in (0,99,119,129,249,499,599):
            print(json.dumps({'step':step+1,'force_mean':load_force(env).mean().item(),
                'loaded_fraction':env._pressed_loaded.float().mean().item(),
                'released_standing':released_standing(env).mean().item(),
                'box_height':env.scene['load'].data.root_link_pos_w[:,2].mean().item()}),flush=True)
wrapped.close()
print('PROBE_DONE')
