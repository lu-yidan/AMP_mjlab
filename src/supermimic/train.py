"""Collect real transitions, fit dynamics, optimize policy by imagined BPTT.

Adapted from supermimic_vibed train.py (MIT). No AMP/PPO policy update is used.
"""
import argparse
from dataclasses import dataclass, asdict
import json
from pathlib import Path
import torch
from torch import nn
from .buffer import ReplayBuffer
from .reset_bank import DEFAULT_BANK

from .mdp import (OBS_DIM, INV_DIM, unflatten_state, observation, reward,
                  termination, critic_input)
from .networks import WorldModel, Policy, Critic

CHECKPOINT_FORMAT = "supermimic_recovery_velocity_v2"


@dataclass
class Config:
    num_envs: int = 1024
    iterations: int = 100000
    buffer_steps: int = 256
    history: int = 10
    horizon: int = 32
    wm_window: int = 8
    wm_batches: int = 10
    wm_batch: int = 256
    policy_batch: int = 256
    warmup_steps: int = 64
    wm_only_iters: int = 500
    sim_steps: int = 3
    lr: float = 2e-4
    gamma: float = .99
    lam: float = .95
    save_interval: int = 1000
    seed: int = 42
    max_speed: float = 3.
    reset_mode: str = "stratified"
    bank_path: str = DEFAULT_BANK
    episode_seconds: float = 20.
    recovery_fraction: float = .5
    small: bool = False
    device: str = 'cuda:0'

    def validate(self):
        for k in ('num_envs','iterations','buffer_steps','history','horizon','wm_window',
                  'wm_batches','wm_batch','policy_batch','warmup_steps','sim_steps','save_interval'):
            if getattr(self,k)<=0:raise ValueError(f'{k} must be positive')
        if self.warmup_steps<self.wm_window or self.buffer_steps<max(self.wm_window,self.history):
            raise ValueError('warmup/buffer must cover model windows and history')
        if self.reset_mode not in ('standing','stratified'):raise ValueError('Unknown reset mode')
        if not 0<=self.recovery_fraction<=1 or self.episode_seconds<=0:raise ValueError('Invalid recovery configuration')
        if not 0<self.gamma<1:raise ValueError('gamma must be in (0,1)')
        if not .1<=self.max_speed<=3:raise ValueError('max_speed must be in [0.1,3]')


class Trainer:
    def __init__(self, cfg, default_q, dt=.02):
        cfg.validate();self.cfg=cfg;self.default_q=default_q;self.dt=dt
        self.wm=WorldModel(hidden=(64,64) if cfg.small else (2048,2048,2048)).to(cfg.device)
        self.pi=Policy(OBS_DIM,3,hist=cfg.history,hidden=(64,64) if cfg.small else (2048,1024,512,256)).to(cfg.device)
        self.critic=Critic(INV_DIM+3+29).to(cfg.device)
        self.optimizers=[torch.optim.Adam(m.parameters(),lr=cfg.lr) for m in (self.wm,self.pi,self.critic)]
        self.buffer=ReplayBuffer(cfg.buffer_steps,cfg.num_envs,cfg.device)
        self.history=torch.zeros(cfg.num_envs,cfg.history,OBS_DIM,device=cfg.device)
        self.iteration=0;self.steps=0;self.reset_metadata={};self.collect_metrics={}
        self.error_floor=torch.tensor([.003]*3+[.005]*3+[.02]*3+[.05]*3+[.003]*29+[.05]*29,device=cfg.device)

    @torch.no_grad()
    def collect(self, sim, steps):
        values=[];recoveries=[];completed=[];successes=[]
        for _ in range(steps):
            o=observation(sim.state(),sim.a_prev,self.default_q)
            self.history=torch.where((sim.ep_step==0)[:,None,None],o[:,None].expand_as(self.history),
                torch.cat((self.history[:,1:],o[:,None]),1))
            self.pi.obs_norm.update(o);self.pi.goal_norm.update(sim.command)
            a=(self.pi(self.history,sim.command)+.1*torch.randn_like(sim.a_prev)).clamp(-5,5)
            tr=sim.step(a);self.buffer.add(tr)
            S,S1=unflatten_state(tr['s']),unflatten_state(tr['s1'])
            self.wm.in_norm.update(self.wm.inputs(S,tr['a'],tr['xi']))
            self.wm.out_norm.update(self.wm.targets(S,S1))
            values.append(tr['r'].mean());self.steps+=sim.n
            recoveries.append(tr['recovery'].float().mean())
            eligible=tr['done']&(tr['reset_group']>0)
            completed.append(eligible.sum());successes.append((eligible&tr['recovered']).sum())
        count=int(torch.stack(completed).sum())
        self.collect_metrics=dict(recovery_state_fraction=float(torch.stack(recoveries).mean()),
            completed_recovery_episodes=count, successful_recovery_episodes=int(torch.stack(successes).sum()))
        return float(torch.stack(values).mean())

    @staticmethod
    def optimize(loss, model, optimizer):
        if not torch.isfinite(loss):raise FloatingPointError('Non-finite training loss')
        optimizer.zero_grad(set_to_none=True);loss.backward()
        norm=nn.utils.clip_grad_norm_(model.parameters(),1.)
        if not torch.isfinite(norm):raise FloatingPointError('Non-finite gradient')
        optimizer.step()

    def wm_update(self):
        c=self.cfg;w=self.buffer.sample_windows(c.wm_batch,c.wm_window,c.recovery_fraction)
        S=unflatten_state(w['s'][:,0]);losses=[]
        scale=torch.maximum(self.wm.out_norm.std(),self.error_floor)
        for j in range(c.wm_window):
            S=self.wm.step(S,w['a'][:,j],w['xi'][:,j])
            losses.append((self.wm.state_error(S,unflatten_state(w['s1'][:,j]))/scale).square().mean(-1))
        loss=(torch.stack(losses,1)*w['mask']).sum()/w['mask'].sum().clamp_min(1)
        self.optimize(loss,self.wm,self.optimizers[0]);return float(loss.detach())

    def policy_update(self):
        c=self.cfg;s,h=self.buffer.sample_starts(c.policy_batch,c.history,c.recovery_fraction)
        S=unflatten_state(s['s']);a_prev=s['a_prev'];goal=s['command']
        hist=observation(unflatten_state(h['s']),h['a_prev'],self.default_q).detach()
        # Freeze model weights while retaining derivatives with respect to actions/state.
        for model in (self.wm,self.critic):model.requires_grad_(False)
        xs=[critic_input(S,goal,a_prev).detach()];rewards=[];ends=[]
        survival=torch.ones(c.policy_batch,device=c.device);objective=0.
        try:
            for t in range(c.horizon):
                a=self.pi(hist,goal);previous=S;S=self.wm.step(S,a,s['xi'])
                r=reward(S,goal,a,a_prev,self.default_q,self.dt,previous_state=previous,gamma=c.gamma);d=termination(S,soft=True)
                objective=objective+c.gamma**t*survival*r
                survival=survival*(1-d)
                hist=torch.cat((hist[:,1:],observation(S,a,self.default_q)[:,None]),1)
                a_prev=a;rewards.append(r.detach());ends.append(d.detach())
                xs.append(critic_input(S,goal,a).detach())
            objective=objective+c.gamma**c.horizon*survival*self.critic(critic_input(S,goal,a_prev))
            loss=-objective.mean();self.optimize(loss,self.pi,self.optimizers[1])
        finally:
            for model in (self.wm,self.critic):model.requires_grad_(True)
        X=torch.stack(xs);self.critic.norm.update(X)
        with torch.no_grad():
            V=self.critic(X);target=V[-1];targets=[]
            for t in reversed(range(c.horizon)):
                target=rewards[t]+c.gamma*(1-ends[t])*((1-c.lam)*V[t+1]+c.lam*target)
                targets.append(target)
            Y=torch.stack(targets[::-1]).flatten();X=X[:-1].flatten(0,1)
        for _ in range(5):
            for ids in torch.randperm(len(Y),device=c.device).tensor_split(min(4,len(Y))):
                vl=(self.critic(X[ids])-Y[ids]).square().mean()
                self.optimize(vl,self.critic,self.optimizers[2])
        return dict(policy_loss=float(loss.detach()),critic_loss=float(vl.detach()),
                    imagined_survival=float(survival.detach().mean()),
                    imagined_start_recovery_fraction=float(s['recovery'].float().mean()))

    def save(self, path, joint_names):
        path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
        payload=dict(format=CHECKPOINT_FORMAT,reset_metadata=self.reset_metadata,config=asdict(self.cfg),iteration=self.iteration,
            steps=self.steps,dt=self.dt,default_q=self.default_q, joint_names=list(joint_names),
            wm=self.wm.state_dict(),pi=self.pi.state_dict(),critic=self.critic.state_dict(),
            optimizers=[x.state_dict() for x in self.optimizers])
        temp=path.with_suffix('.tmp');torch.save(payload,temp);temp.replace(path)

    def load(self,path,joint_names):
        ck=torch.load(path,map_location=self.cfg.device,weights_only=False)
        if ck['format']!=CHECKPOINT_FORMAT or ck['joint_names']!=list(joint_names):
            raise ValueError('Incompatible checkpoint interface/joint order')
        for key in ('history','small','max_speed','reset_mode','episode_seconds','recovery_fraction','gamma'):
            if ck['config'][key]!=getattr(self.cfg,key):raise ValueError(f'Resume mismatch: {key}')
        if abs(ck['dt']-self.dt)>1e-8 or not torch.allclose(ck['default_q'],self.default_q):
            raise ValueError('Resume robot/default pose/control dt mismatch')
        if self.reset_metadata and ck.get('reset_metadata',{}).get('sha256')!=self.reset_metadata.get('sha256'):
            raise ValueError('Resume reset bank mismatch')
        for key in ('wm','pi','critic'):getattr(self,key).load_state_dict(ck[key])
        for opt,state in zip(self.optimizers,ck['optimizers']):opt.load_state_dict(state)
        self.iteration=ck['iteration'];self.steps=ck['steps']
        # Replay/simulator are intentionally refilled; this is not bitwise continuation.


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name,f in Config.__dataclass_fields__.items():
        p.add_argument('--'+name.replace('_','-'),**({'action':'store_true'} if isinstance(f.default,bool) else {'type':type(f.default)}),default=f.default)
    p.add_argument('--output',default='logs/supermimic/stand_walk_run')
    p.add_argument('--resume');p.add_argument('--wandb-project')
    args=p.parse_args();values=vars(args);cfg=Config(**{k:values[k] for k in Config.__dataclass_fields__})
    cfg.validate();torch.manual_seed(cfg.seed)
    from .env import VelocitySim
    sim=VelocitySim(cfg.num_envs,cfg.device,cfg.seed,cfg.max_speed,reset_mode=cfg.reset_mode,
        bank_path=cfg.bank_path,episode_seconds=cfg.episode_seconds,gamma=cfg.gamma)
    trainer=Trainer(cfg,sim.default_q,sim.env.step_dt)
    if sim.bank is not None:trainer.reset_metadata=dict(path=sim.bank.path,sha256=sim.bank.sha256)
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    wb=None
    try:
        if args.resume:trainer.load(args.resume,sim.robot.joint_names)
        if args.wandb_project:
            import wandb
            wb=wandb.init(project=args.wandb_project,name=out.name,config=asdict(cfg))
        (out/'config.json').write_text(json.dumps(asdict(cfg),indent=2))
        trainer.collect(sim,cfg.warmup_steps)
        with (out/'metrics.jsonl').open('a') as log:
            for it in range(trainer.iteration,cfg.iterations):
                metrics=dict(iteration=it+1,sim_reward=trainer.collect(sim,cfg.sim_steps))
                metrics.update(trainer.collect_metrics)
                metrics['wm_loss']=sum(trainer.wm_update() for _ in range(cfg.wm_batches))/cfg.wm_batches
                if it>=cfg.wm_only_iters:metrics.update(trainer.policy_update())
                trainer.iteration=it+1;metrics['sim_steps']=trainer.steps
                log.write(json.dumps(metrics)+'\n');log.flush()
                if wb:wb.log(metrics,step=it+1)
                if it%10==0:print(json.dumps(metrics),flush=True)
                if (it+1)%cfg.save_interval==0 or it+1==cfg.iterations:
                    trainer.save(out/f'model_{it+1}.pt',sim.robot.joint_names)
    finally:
        sim.close()
        if wb:wb.finish()


if __name__=='__main__':main()
