"""Held-out recovery test, with an unchanged velocity command throughout getting up."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from .evaluate import load_policy
from .mdp import OBS_DIM, observation, is_recovery, termination
from .reset_bank import ResetBank
from .rotations import yaw_mat


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',required=True);p.add_argument('--bank',required=True)
    p.add_argument('--output',required=True);p.add_argument('--device',default='cuda:0')
    p.add_argument('--seconds',type=float,default=20.);p.add_argument('--speed',type=float,default=0.)
    p.add_argument('--max-cases',type=int,default=0)
    a=p.parse_args()
    if a.seconds<=1 or a.max_cases<0 or not 0<=a.speed<=5:p.error('Invalid duration/cases/speed')
    pi,ck=load_policy(a.checkpoint,a.device)
    bank=ResetBank(a.bank,ck['joint_names'],a.device)
    if bank.sha256==ck.get('reset_metadata',{}).get('sha256'):
        p.error('Use a held-out bank, not the training bank')
    ids=np.flatnonzero(bank.stages=='low')
    if a.max_cases and len(ids)>a.max_cases:
        ids=np.random.default_rng(42).choice(ids,a.max_cases,replace=False)
    if not len(ids):p.error('No low poses in bank')
    from .env import VelocitySim
    sim=VelocitySim(len(ids),a.device,42,reset_mode='standing')
    try:
        if list(sim.robot.joint_names)!=ck['joint_names'] or abs(sim.env.step_dt-ck['dt'])>1e-8:
            raise ValueError('Robot interface mismatch')
        if not torch.allclose(sim.default_q,ck['default_q']):raise ValueError('Default pose mismatch')
        sim.reset_to_bank(bank,torch.tensor(ids,device=a.device))
        sim.command.zero_();sim.command[:,0]=a.speed
        hist=torch.zeros(sim.n,pi.hist,OBS_DIM,device=a.device)
        first=torch.full((sim.n,),float('nan'),device=a.device)
        failed=torch.zeros(sim.n,dtype=torch.bool,device=a.device)
        stable=torch.zeros(sim.n,dtype=torch.long,device=a.device)
        errors=torch.zeros(sim.n,device=a.device);counts=errors.clone();tail=[]
        with torch.inference_mode():
            for step in range(round(a.seconds/sim.env.step_dt)):
                o=observation(sim.state(),sim.a_prev,sim.default_q)
                hist=o[:,None].expand_as(hist).clone() if step==0 else torch.cat((hist[:,1:],o[:,None]),1)
                sim.step(pi(hist,sim.command),auto_reset=False,resample=False)
                S=sim.state();failed|=termination(S)
                up=~is_recovery(S);stable=torch.where(up&~failed,stable+1,0)
                newly=(stable>=round(1/sim.env.step_dt))&first.isnan()
                first[newly]=(step+1)*sim.env.step_dt
                valid=first.isfinite()&~failed
                vx=(yaw_mat(S[1]).transpose(-1,-2)@S[2].unsqueeze(-1)).squeeze(-1)[:,0]
                errors+=torch.where(valid,(vx-a.speed).abs(),0);counts+=valid
                tail.append(up)
        success=first.isfinite();tail=torch.stack(tail[-round(1/sim.env.step_dt):]).float().mean(0)
        rows=[dict(bank_row=int(idx),source=str(bank.sources[idx]),direction=str(bank.labels[idx]),
            success=bool(success[i]),recovery_seconds=float(first[i]) if success[i] else None,
            catastrophic=bool(failed[i]),final_upright_fraction=float(tail[i]),
            post_recovery_vx_mae=float(errors[i]/counts[i]) if counts[i]>0 else None) for i,idx in enumerate(ids)]
        report=dict(checkpoint=a.checkpoint,iteration=ck['iteration'],bank_sha256=bank.sha256,
            command_vx=a.speed,seconds=a.seconds,success_fraction=float(success.float().mean()),
            protocol='No resets. Success: root height >=0.65m and Rzz >=0.9 continuously for 1s. Command remains active from start. Flat nominal dynamics.',cases=rows)
        out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(report,indent=2))
        print(json.dumps({k:v for k,v in report.items() if k!='cases'},indent=2))
    finally:sim.close()


if __name__=='__main__':main()
