"""Paired AMP constrained adaptation from a shared, verified flat-ground parent."""
from __future__ import annotations
import argparse, contextlib, hashlib, importlib.metadata, io, json, os, random, subprocess, time
from dataclasses import asdict
from pathlib import Path
import numpy as np
import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.utils.os import dump_yaml
from mjlab.utils.torch import configure_torch_backends
from g1recovery_amp.tasks.recovery import LOW_TORQUE_CAP_ONLY_90_TASK_ID as TASK

GEOMETRY_TERMS=('plate_geometry_progress','plate_clearance','plate_separation')
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write_json(path,value):
    path=Path(path);tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');tmp.replace(path)

def build_config(arm,seed,num_envs,updates):
    cfg=load_env_cfg(TASK,play=False);agent=load_rl_cfg(TASK)
    assert all(cfg.rewards[k].weight>0 for k in GEOMETRY_TERMS)
    if arm=='off':
        for k in GEOMETRY_TERMS:cfg.rewards[k].weight=0.
    cfg.seed=agent.seed=seed;cfg.scene.num_envs=num_envs
    agent.max_iterations=updates;agent.save_interval=500;agent.logger='tensorboard'
    agent.upload_model=False;agent.run_name=f'AMP_geometry_{arm}_seed{seed}'
    assert cfg.observations['actor'].enable_corruption
    assert 'push_robot' in cfg.events and cfg.events['a6_dynamics'].params['enabled']
    return cfg,agent

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--arm',choices=('off','on'),required=True)
    p.add_argument('--checkpoint',type=Path,default=Path('checkpoints/parents/flat_model_19998.pt'))
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--num-envs',type=int,default=4096);p.add_argument('--updates',type=int,default=20000)
    p.add_argument('--seed',type=int,default=20261005);p.add_argument('--device',default='cuda:0')
    p.add_argument('--wandb',choices=('online','offline','disabled'),default='online')
    p.add_argument('--dry-run',action='store_true');a=p.parse_args()
    assert a.updates>0 and a.num_envs>0
    a.out.mkdir(parents=True,exist_ok=False);configure_torch_backends()
    random.seed(a.seed);np.random.seed(a.seed);torch.manual_seed(a.seed)
    cfg,agent=build_config(a.arm,a.seed,a.num_envs,a.updates)
    dump_yaml(a.out/'env.yaml',asdict(cfg));dump_yaml(a.out/'agent.yaml',asdict(agent))
    parent=torch.load(a.checkpoint,weights_only=False,map_location='cpu')
    assert sha(a.checkpoint)=='4280efe9de1d6772e33f93847208896c4f7126e62630fc95b1cfeafe72e9f47c'
    assert 'a6_adaptation_state' not in (parent.get('infos') or {})
    launch=dict(arm=a.arm,seed=a.seed,num_envs=a.num_envs,additional_updates=a.updates,
        control_transitions=a.num_envs*agent.num_steps_per_env*a.updates,
        parent_checkpoint=str(a.checkpoint.resolve()),parent_sha256=sha(a.checkpoint),
        parent_iteration=int(parent['iter']),parent_infos=parent.get('infos'),
        scope='paired constrained adaptation from a flat parent without plate guidance',
        git_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        git_branch=subprocess.check_output(['git','branch','--show-current'],text=True).strip(),
        script_sha256=sha(__file__),task=TASK,episode_length_s=cfg.episode_length_s,
        geometry_terms={k:cfg.rewards[k].weight for k in GEOMETRY_TERMS},
        all_reward_weights={k:v.weight for k,v in cfg.rewards.items()},
        control_dt=cfg.sim.mujoco.timestep*cfg.decimation,physics_dt=cfg.sim.mujoco.timestep,
        save_every_updates=agent.save_interval,
        reset=cfg.events['a6_three_scene_curriculum_reset'].params|{'asset_cfg':'robot: G1_JOINT_NAMES'},
        actor_observation_noise=cfg.observations['actor'].enable_corruption,pushes=True,
        motor_mass_inertia_delay_randomization=True,cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
        packages={x:importlib.metadata.version(x) for x in ('torch','mjlab','mujoco','mujoco-warp','rsl-rl-lib','warp-lang')})
    write_json(a.out/'launch.json',launch);del parent
    if a.dry_run:print('DRY_RUN_PASS',json.dumps(launch),flush=True);return
    wb=None
    if a.wandb!='disabled':
        import wandb
        try:
            wb=wandb.init(project='amp-guidance-luyidan',group='flat19998_guidance_20k_20261005',
                name=agent.run_name,config=launch,dir=str(a.out.resolve()),mode=a.wandb,
                sync_tensorboard=True,settings=wandb.Settings(init_timeout=45))
            write_json(a.out/'wandb.json',dict(id=wb.id,url=wb.url,mode=a.wandb))
        except Exception as e:
            write_json(a.out/'wandb_unavailable.json',dict(error_type=type(e).__name__,
                message='W&B unavailable; TensorBoard and local provenance remain enabled.'))
            print('WANDB_UNAVAILABLE',type(e).__name__,flush=True)
    env=None;wrapper=None
    try:
        env=ManagerBasedRlEnv(cfg=cfg,device=a.device)
        wrapper=RslRlVecEnvWrapper(env,clip_actions=agent.clip_actions)
        runner=load_runner_cls(TASK)(wrapper,asdict(agent),log_dir=str(a.out),device=a.device)
        runner.add_git_repo_to_log(__file__);runner.load(str(a.checkpoint.resolve()),map_location=a.device)
        # Restore the flat parent and rebase the A6 adaptation curriculum; iteration labels are adaptation updates.
        runner.current_learning_iteration=0;env.reset()
        initial={k:v.detach().clone() for k,v in runner.alg.actor.state_dict().items()}
        np.savez_compressed(a.out/'initial.npz',qpos=env.sim.data.qpos.cpu().numpy(),
            qvel=env.sim.data.qvel.cpu().numpy(),scene=env._a6_reset_scene.cpu().numpy(),
            direction=env._a6_reset_direction.cpu().numpy(),source=env._a6_reset_source.cpu().numpy(),
            stage=env._a6_reset_stage.cpu().numpy())
        runner.save(str(a.out/'model_initial.pt'));native_log=runner.logger.log;started=time.time()
        def log(**kw):
            it=kw['it']
            if it%25==0 or it==a.updates-1:
                native_log(**kw);losses={k:float(v) for k,v in kw['loss_dict'].items()}
                assert all(np.isfinite(v) for v in losses.values()),losses
                delta=max(float((v-initial[k]).abs().max()) for k,v in runner.alg.actor.state_dict().items())
                write_json(a.out/'progress.json',dict(status='running',completed_updates=it+1,
                    target_updates=a.updates,elapsed_s=time.time()-started,
                    steps_per_second=a.num_envs*agent.num_steps_per_env/(kw['collect_time']+kw['learn_time']),
                    learning_rate=kw['learning_rate'],actor_max_change=delta,
                    action_std_mean=float(kw['action_std'].mean()),losses=losses))
            else:
                with contextlib.redirect_stdout(io.StringIO()):native_log(**kw)
        runner.logger.log=log;print('TRAINING_STARTED',json.dumps(launch),flush=True)
        runner.learn(num_learning_iterations=a.updates,init_at_random_ep_len=True)
        final=a.out/f'model_{a.updates-1}.pt';assert final.exists()
        write_json(a.out/'completed.json',dict(status='complete',additional_updates=a.updates,
            checkpoint=str(final.resolve()),checkpoint_sha256=sha(final),elapsed_s=time.time()-started))
        print('TRAINING_COMPLETE',str(final),flush=True)
    except Exception as e:
        write_json(a.out/'failed.json',dict(error_type=type(e).__name__,message=str(e)));raise
    finally:
        if wrapper is not None:wrapper.close()
        elif env is not None:env.close()
        if wb is not None:wb.finish()

if __name__=='__main__':main()
