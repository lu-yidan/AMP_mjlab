"""AMP-E: scratch SuperMimic training using AMP-C's environment and experts."""
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'third_party/mbrl_supermimic'))
sys.path.insert(0,str(ROOT))
import argparse,json,time,hashlib
import torch
import src.tasks
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg,load_rl_cfg
from src.amp_supermimic.task import AMPTask
from supermimic.runners import SuperMimicRunner

class Runner(SuperMimicRunner):
    def _log(self,*args):
        args[1].update({'Discriminator/'+k:[v] for k,v in self.task.last_disc_metrics.items()})
        row=super()._log(*args)
        if getattr(self,'wandb_run',None):self.wandb_run.log(row,step=args[0])
        return row
    def save(self,path,infos=None):
        super().save(path,infos)
        d=torch.load(path,weights_only=False)
        d['multi_amp_state_dict']=self.task.disc.state_dict()
        d['multi_amp_optimizers']={k:o.state_dict() for k,o in self.task.disc.optimizers.items()}
        d['experiment']=self.cfg['experiment'];d['task']=self.cfg['task']
        if getattr(self,'policy_ema',None) is not None:
            d['policy_ema']=self.policy_ema.state_dict()
            d['ema_metadata']={'decay':self.policy_ema.decay,'updates':self.policy_ema.updates,'normalizer_buffers':'copied_from_live'}
        torch.save(d,path+'.tmp');Path(path+'.tmp').replace(path)

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--num-envs',type=int,default=4096);p.add_argument('--iterations',type=int,default=100000)
    p.add_argument('--sim-steps',type=float,default=24);p.add_argument('--output',required=True)
    p.add_argument('--verify-only',action='store_true');p.add_argument('--smoke',action='store_true')
    p.add_argument('--terrain',action='store_true')
    p.add_argument('--flat-discriminator',choices=['shared','dual'])
    p.add_argument('--iteration-speed-curriculum',action='store_true')
    p.add_argument('--blink-core',action='store_true')
    p.add_argument('--recovery-height-weight',type=float,default=1.0)
    p.add_argument('--no-compile',action='store_true');p.add_argument('--wandb',action='store_true')
    p.add_argument('--experiment-name')
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--world-model-width',type=int,default=2048)
    p.add_argument('--wm-updates',type=int,default=10)
    p.add_argument('--ema-decay',type=float,default=0.)
    p.add_argument('--save-interval',type=int,default=1000)
    a=p.parse_args()
    if a.flat_discriminator and a.terrain:p.error('flat-discriminator requires flat terrain')
    if a.blink_core and (a.terrain or not a.flat_discriminator or not a.experiment_name):p.error('blink-core requires named flat discriminator experiment')
    if a.sim_steps<1 or a.world_model_width<1 or a.wm_updates<1 or a.save_interval<1:p.error('Invalid workload')
    if not 0<=a.ema_decay<1:p.error('ema-decay must be in [0,1)')
    if not 0 <= a.recovery_height_weight <= 1: p.error('--recovery-height-weight must be between 0 and 1')
    out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    task_id='Unitree-G1-AMP-I-Shared-TerrainCritic' if a.terrain else 'Unitree-G1-MultiAMP-Phase-Tracking-Flat'
    if a.flat_discriminator:
        task_id='Unitree-G1-AMP-Controlled-'+('F-Shared' if a.flat_discriminator=='shared' else 'H-Full')+'-Flat'
    experiment='K-SuperMimic-Shared-Terrain' if a.terrain else 'AMP-E'
    if a.recovery_height_weight != 1.0:
        if not a.terrain or a.recovery_height_weight != .5: p.error('Only the terrain half-height ablation is currently named')
        experiment='L-SuperMimic-HalfHeight-Terrain'
    if a.experiment_name:experiment=a.experiment_name
    elif a.ema_decay or a.world_model_width!=2048 or a.wm_updates!=10:p.error('Modified recipes require --experiment-name')
    cfg=load_env_cfg(task_id);cfg.scene.num_envs=a.num_envs;cfg.seed=a.seed
    if a.iteration_speed_curriculum:
        cfg.commands['twist'].automatic_curriculum=False
        cfg.commands['twist'].full_speed=False
    cfg.rewards['track_root_height'].weight=a.recovery_height_weight
    if a.verify_only:
        cfg.observations['actor'].enable_corruption=False
        cfg.events.pop('push_robot',None)
    torch.manual_seed(a.seed)
    env=ManagerBasedRlEnv(cfg=cfg,device='cuda:0');env.reset()
    agent=load_rl_cfg(task_id)
    expected=json.loads((ROOT/'reports/amp_stability_smp_expert_audit.json').read_text())['sha256']
    if hashlib.sha256(Path(agent.multi_amp['expert_bank']).read_bytes()).hexdigest()!=expected:
        raise ValueError('AMP-C expert bank checksum mismatch')
    if a.terrain:
        from src.amp_supermimic.terrain_task import TerrainAMPTask
        task=TerrainAMPTask(env,agent,ROOT/'logs/terrain_map_cache',capture_verification=a.verify_only)
    else:
        task=AMPTask(env,agent,capture_verification=a.verify_only)
    actor_width=task.get_observations()['policy'].shape[-1]
    if actor_width!=384:raise ValueError(f'Blind actor contract changed: {actor_width}')
    if 'height_scan' in cfg.observations['actor'].terms:raise ValueError('Actor must not receive terrain scan')
    if a.verify_only:
        from src.amp_supermimic.verify import verify
        terrain_report=None
        if a.terrain:
            from src.amp_supermimic.verify_terrain import audit_terrain
            terrain_report=audit_terrain(task)
            print('TERRAIN_AUDIT',terrain_report,flush=True)
        report=verify(task)
        if terrain_report is not None:
            report['terrain_audit']=terrain_report
            report['passed']=report['passed'] and terrain_report['passed']
        (out/'verification.json').write_text(json.dumps(report,indent=2));print(report,flush=True)
        env.close()
        if not report['passed']:raise SystemExit(experiment+' numerical gate failed')
        return
    train=dict(num_steps_per_env=a.sim_steps,save_interval=a.save_interval,log_interval=10,seed=a.seed,logger='none',
        policy=dict(actor_hidden_dims=[512,256,128],critic_hidden_dims=[512,256,128]),
        world_model=dict(hidden_dims=[a.world_model_width]*3),algorithm=dict(horizon=32,
        termination='hard',wm_updates=a.wm_updates,wm_warmup_iters=300,buffer_steps=128,lr_decay_iters=100000,
        compile=not a.no_compile,amp='bf16',fail_on_nonfinite=True))
    if a.smoke:
        train['algorithm'].update(wm_warmup_iters=1)
        train['log_interval']=1
    train.update(experiment=experiment,max_iterations=a.iterations,task=task_id,expert_sha256=expected,
        recovery_height_weight=a.recovery_height_weight,ema_decay=a.ema_decay,actor_input='proprioception_only',actor_observation_width=actor_width)
    train.update(flat_discriminator=a.flat_discriminator,speed_curriculum_iterations=[5000,15000] if a.iteration_speed_curriculum else None)
    if a.blink_core:
        train['world_model']['blink_structured']=True
        train['algorithm'].update(learning_rate=4e-4,wm_learning_rate=4e-4)
        train['blink_core']={'structured_drift':.625,'optimizer':'Muon hidden matrices + Adam','mirror':False,'reference_skip':False,'custom_cuda_graph':False}
    if a.terrain:
        train['terrain_map_sha256']=task.terrain_map.fingerprint
        train['algorithm']['fail_on_nonfinite']=True
    (out/'config.json').write_text(json.dumps(train,indent=2))
    def after_iteration(r,it):
        if a.iteration_speed_curriculum:
            stage=int(it>=5000)+int(it>=15000)
            if task.command.stage!=stage:
                task.command.stage=stage
                print(f'[IterationSpeedCurriculum] iteration={it} stage={stage}',flush=True)
        task.update_discriminators()
        if a.iteration_speed_curriculum and it%10==0:
            c=task.command.command
            metrics={'Curriculum/stage':task.command.stage,
                'Curriculum/command_vx_above_1':float((c[:,0]>1.).float().mean()),
                'Curriculum/command_vx_above_2':float((c[:,0]>2.).float().mean()),
                'Curriculum/target_vx_above_2':float((task.command.target[:,0]>2.).float().mean())}
            if getattr(r,'wandb_run',None):r.wandb_run.log(metrics,step=it)
        if r.policy_ema is not None:r.policy_ema.update(r.policy)
    runner=Runner(task,train,str(out),'cuda:0',callbacks=[after_iteration])
    if a.blink_core:
        from src.amp_supermimic.blink_core import configure_blink_optimizers
        configure_blink_optimizers(runner)
    from src.amp_supermimic.ema import PolicyEMA
    runner.policy_ema=PolicyEMA(runner.policy,a.ema_decay) if a.ema_decay else None
    if a.wandb:
        import wandb
        runner.wandb_run=wandb.init(entity='tabletennis',project='lyd-multi-amp',
            name=f'{experiment}-scratch-{a.num_envs}x{a.sim_steps}-{a.iterations}-seed{a.seed}',
            group='amp-supermimic-terrain' if a.terrain else 'amp-recovery-stability-20261027',config=train)
    for param in task.disc.parameters():param.requires_grad_(False)
    runner.learn(a.iterations)
    if a.wandb:runner.wandb_run.finish()
    env.close()
if __name__=='__main__':main()
