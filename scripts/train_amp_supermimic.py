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
        torch.save(d,path+'.tmp');Path(path+'.tmp').replace(path)

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--num-envs',type=int,default=4096);p.add_argument('--iterations',type=int,default=100000)
    p.add_argument('--sim-steps',type=int,default=24);p.add_argument('--output',required=True)
    p.add_argument('--verify-only',action='store_true');p.add_argument('--smoke',action='store_true')
    p.add_argument('--terrain',action='store_true')
    p.add_argument('--no-compile',action='store_true');p.add_argument('--wandb',action='store_true')
    a=p.parse_args();out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    task_id='Unitree-G1-AMP-I-Shared-TerrainCritic' if a.terrain else 'Unitree-G1-MultiAMP-Phase-Tracking-Flat'
    experiment='K-SuperMimic-Shared-Terrain' if a.terrain else 'AMP-E'
    cfg=load_env_cfg(task_id);cfg.scene.num_envs=a.num_envs;cfg.seed=42
    if a.verify_only:
        cfg.observations['actor'].enable_corruption=False
        cfg.events.pop('push_robot',None)
    torch.manual_seed(42)
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
    train=dict(num_steps_per_env=a.sim_steps,save_interval=1000,log_interval=10,seed=42,logger='none',
        policy=dict(actor_hidden_dims=[512,256,128],critic_hidden_dims=[512,256,128]),
        world_model=dict(hidden_dims=[2048,2048,2048]),algorithm=dict(horizon=32,
        termination='hard',wm_warmup_iters=300,buffer_steps=128,lr_decay_iters=100000,
        compile=not a.no_compile,amp='bf16'))
    if a.smoke:
        train['algorithm'].update(wm_warmup_iters=1)
        train['log_interval']=1
    train.update(experiment=experiment,max_iterations=a.iterations,task=task_id,expert_sha256=expected)
    if a.terrain:
        train['terrain_map_sha256']=task.terrain_map.fingerprint
        train['algorithm']['fail_on_nonfinite']=True
    (out/'config.json').write_text(json.dumps(train,indent=2))
    runner=Runner(task,train,str(out),'cuda:0',callbacks=[lambda r,it:task.update_discriminators()])
    if a.wandb:
        import wandb
        runner.wandb_run=wandb.init(entity='tabletennis',project='lyd-multi-amp',
            name=f'{experiment}-scratch-{a.num_envs}x{a.sim_steps}-{a.iterations}-seed42',
            group='amp-supermimic-terrain' if a.terrain else 'amp-recovery-stability-20261027',config=train)
    for param in task.disc.parameters():param.requires_grad_(False)
    runner.learn(a.iterations)
    if a.wandb:runner.wandb_run.finish()
    env.close()
if __name__=='__main__':main()
