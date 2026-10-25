"""Evaluate one SuperMimic velocity policy with stand/walk/run command steps."""
import argparse
import json
from pathlib import Path
import torch
from .mdp import OBS_DIM, observation, termination
from .networks import Policy
from .rotations import yaw_mat


def load_policy(checkpoint, device):
    ck=torch.load(checkpoint,map_location=device,weights_only=False)
    if ck.get('format')!='supermimic_recovery_velocity_v2':raise ValueError('Not a SuperMimic velocity checkpoint')
    c=ck['config']
    pi=Policy(OBS_DIM,3,hist=c['history'],hidden=(64,64) if c['small'] else (2048,1024,512,256)).to(device)
    pi.load_state_dict(ck['pi']);pi.eval()
    return pi,ck


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',required=True);p.add_argument('--output',required=True)
    p.add_argument('--device',default='cuda:0');p.add_argument('--seconds',type=float,default=12.)
    p.add_argument('--repeats',type=int,default=8)
    a=p.parse_args()
    if a.seconds<=2 or a.repeats<=0:p.error('seconds must exceed 2; repeats must be positive')
    from .env import VelocitySim
    speeds=[0.,.5,1.,2.,3.]
    pi,ck=load_policy(a.checkpoint,a.device)
    sim=VelocitySim(len(speeds)*a.repeats,a.device,42,reset_mode="standing")
    try:
        if ck['joint_names']!=list(sim.robot.joint_names):raise ValueError('Joint order mismatch')
        if abs(ck['dt']-sim.env.step_dt)>1e-8 or not torch.allclose(ck['default_q'],sim.default_q):
            raise ValueError('Robot/control timestep mismatch')
        target=torch.tensor(speeds,device=a.device).repeat_interleave(a.repeats)
        hist=torch.zeros(sim.n,pi.hist,OBS_DIM,device=a.device)
        actual=[];upright=[];failed=torch.zeros(sim.n,dtype=torch.bool,device=a.device)
        with torch.inference_mode():
            for step in range(round(a.seconds/sim.env.step_dt)):
                sim.command.zero_();sim.command[:,0]=target if step*sim.env.step_dt>=2. else 0.
                S=sim.state();o=observation(S,sim.a_prev,sim.default_q)
                hist=o[:,None].expand_as(hist).clone() if step==0 else torch.cat((hist[:,1:],o[:,None]),1)
                sim.step(pi(hist,sim.command),auto_reset=False,resample=False)
                S=sim.state();v=(yaw_mat(S[1]).transpose(-1,-2)@S[2].unsqueeze(-1)).squeeze(-1)
                actual.append(v.cpu());upright.append(((S[0][:,2]>.65)&(S[1][:,2,2]>.9)).cpu())
                failed|=termination(S)
        x=torch.stack(actual);up=torch.stack(upright)
        count=min(len(x)-round(2/sim.env.step_dt),round(5/sim.env.step_dt));result={}
        for i,speed in enumerate(speeds):
            sl=slice(i*a.repeats,(i+1)*a.repeats);v=x[-count:,sl]
            result[str(speed)]=dict(actual_vx=float(v[...,0].mean()),vx_mae=float((v[...,0]-speed).abs().mean()),
                upright_fraction=float(up[-count:,sl].float().mean()),ever_failed_fraction=float(failed[sl].float().mean()))
        report=dict(checkpoint=a.checkpoint,iteration=ck['iteration'],seconds=a.seconds,
            protocol='Nominal flat dynamics; identical default starts, deterministic policy; no resets. Repeats are not independent trials.',cases=result)
        out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(report,indent=2))
        print(json.dumps(report,indent=2))
    finally:sim.close()


if __name__=='__main__':main()
