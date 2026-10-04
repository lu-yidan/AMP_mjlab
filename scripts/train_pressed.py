"""Fine-tune the independent loaded-recovery task from an explicit checkpoint."""
import json
import os
from pathlib import Path
from dataclasses import asdict
import mjlab.tasks
import src.tasks
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.utils.os import dump_yaml
from mjlab.utils.torch import configure_torch_backends


def main():
    checkpoint = Path(os.environ['CKPT']).resolve(strict=True)
    log_dir = Path(os.environ['LOG_DIR'])
    log_dir.mkdir(parents=True, exist_ok=False)
    configure_torch_backends()
    task = 'Unitree-G1-HoST-Pressed'
    cfg = load_env_cfg(task)
    cfg.scene.num_envs = int(os.environ.get('NUM_ENVS','2048'))
    cfg.seed = 42
    agent = load_rl_cfg(task)
    agent.seed = 42
    agent.algorithm.learning_rate = 5e-5
    agent.save_interval = 500
    agent.logger = 'tensorboard'
    iterations = int(os.environ.get('MAX_ITER','6000'))
    agent.max_iterations = iterations
    env = RslRlVecEnvWrapper(ManagerBasedRlEnv(cfg=cfg,device='cuda:0'))
    runner = load_runner_cls(task)(env,asdict(agent),str(log_dir),'cuda:0')
    runner.load(str(checkpoint),load_optimizer=False)
    runner.current_learning_iteration = 0
    (log_dir/'params').mkdir()
    dump_yaml(log_dir/'params/env.yaml',asdict(cfg))
    dump_yaml(log_dir/'params/agent.yaml',asdict(agent))
    (log_dir/'source.json').write_text(json.dumps({
        'checkpoint':str(checkpoint), 'mass_kg':float(os.environ.get('HOST_LOAD_MASS','12')),
        'target':os.environ.get('HOST_LOAD_TARGET','mixed'),
        'smp_reference':'https://github.com/tholin-1007/smp/tree/0e67286fe7df77a73740d237ef36b109136552b6',
        'full_smp_prior':False},indent=2))
    try:
        runner.learn(num_learning_iterations=iterations,init_at_random_ep_len=False)
    finally:
        env.close()
    print('PRESSED_TRAINING_DONE',flush=True)


if __name__ == '__main__':
    main()
