"""From-scratch 29-joint flat HoST-port baseline and runtime smoke audit."""
import argparse
import hashlib
import json
import os
import subprocess
from dataclasses import asdict
from pathlib import Path

os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")

import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.utils.os import dump_yaml
from src.tasks.host_recovery.flat29 import flat29_env_cfg
from src.tasks.host_recovery.config.g1.rl_cfg import unitree_g1_host_standup_ppo_runner_cfg
from src.tasks.host_recovery.rl import HoSTOnPolicyRunner
from src.tasks.host_recovery.mdp.events import POSTURE_QUATS


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--num-envs', type=int, default=4096)
    p.add_argument('--iterations', type=int, default=12000)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--log-dir', type=Path, required=True)
    p.add_argument('--smoke', action='store_true')
    p.add_argument('--verify-checkpoint', type=Path)
    a = p.parse_args()
    if a.smoke:
        # Four rollouts of 50 steps reach beyond the 120-step motor-off period.
        a.iterations = max(a.iterations, 4)
    a.log_dir.mkdir(parents=True, exist_ok=False)
    cfg = flat29_env_cfg()
    cfg.scene.num_envs = a.num_envs
    cfg.seed = a.seed
    agent = unitree_g1_host_standup_ppo_runner_cfg()
    agent.seed = a.seed
    agent.logger = 'tensorboard'
    agent.experiment_name = 'g1_host29_flat'
    agent.max_iterations = a.iterations
    dump_yaml(a.log_dir/'env.yaml', asdict(cfg))
    dump_yaml(a.log_dir/'agent.yaml', asdict(agent))
    env = ManagerBasedRlEnv(cfg=cfg, device='cuda:0')
    try:
        robot = env.scene['robot']
        assert len(robot.joint_names) == 29, robot.joint_names
        wrapper = RslRlVecEnvWrapper(env)
        obs = wrapper.get_observations()
        assert obs['actor'].shape[-1] == 564, obs['actor'].shape
        assert wrapper.num_actions == 29
        report = dict(joints=list(robot.joint_names), actor_dim=564, actions=29,
                      num_envs=a.num_envs, iterations=a.iterations,
                      seed=a.seed, from_scratch=a.verify_checkpoint is None, cuda_visible=os.getenv('CUDA_VISIBLE_DEVICES'),
                      method='existing HoST port, single critic PPO; not full paper HoST',
                      robot='repository G1 29DoF; A6 deployment asset parity not yet established',
                      physics_dt=0.002, control_dt=0.02,
                      git_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                      source_sha256={str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                                     for path in (Path(__file__), Path('src/tasks/host_recovery/flat29.py'),
                                                  Path('src/tasks/host_recovery/mdp/observations.py'))})
        (a.log_dir/'launch.json').write_text(json.dumps(report, indent=2))
        if a.smoke:
            # Exercise each fixed posture, including the motors-off transition.
            reset_cfg = env.event_manager.get_term_cfg('reset_base')
            for posture in ('prone', 'supine', 'left_side', 'right_side'):
                reset_cfg.params['posture'] = posture
                wrapper.reset()
                expected = torch.tensor(POSTURE_QUATS[posture], device=env.device)
                alignment = (robot.data.root_link_quat_w * expected).sum(-1).abs()
                assert (alignment > 0.999).all(), (posture, alignment.min().item())
                for _ in range(510):
                    obs, reward, done, extras = wrapper.step(torch.zeros((a.num_envs,29),device='cuda:0'))
                    assert all(torch.isfinite(v).all() for v in obs.values())
                    assert torch.isfinite(reward).all()
                print('POSTURE_FINITE', posture, flush=True)
            reset_cfg.params['posture'] = None
            wrapper.reset()
        runner = HoSTOnPolicyRunner(wrapper, asdict(agent), str(a.log_dir), 'cuda:0')
        if a.verify_checkpoint:
            runner.load(str(a.verify_checkpoint), map_location='cuda:0')
            policy = runner.get_inference_policy(device='cuda:0')
            with torch.inference_mode():
                actions = policy(wrapper.get_observations()['actor'])
            assert actions.shape == (a.num_envs, 29), actions.shape
            assert torch.isfinite(actions).all()
            print('CHECKPOINT_RELOAD_FINITE', str(a.verify_checkpoint), flush=True)
            return
        runner.learn(num_learning_iterations=a.iterations, init_at_random_ep_len=False)
        print('HOST29_TRAINING_DONE', flush=True)
    finally:
        env.close()


if __name__ == '__main__':
    main()
