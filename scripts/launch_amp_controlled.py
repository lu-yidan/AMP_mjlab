"""Launch F/G/H controls, three seeds each, six GPUs with three queued runs.

A failed run stops its worker queue. A durable manifest records every PID and exit.
The launcher refuses to reuse an output directory containing a launch manifest.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
import uuid

ARMS={'F':'F-Shared','G':'G-Unbalanced','H':'H-Full'}

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True)
    p.add_argument('--source-commit',required=True);p.add_argument('--preflight',action='store_true')
    a=p.parse_args();root=Path(__file__).resolve().parents[1];out=Path(a.output).resolve();out.mkdir(parents=True,exist_ok=True)
    path=out/'launch.json'
    bank=root/'logs/amp_stability/smp_recovery_train.npz'
    bank_sha=hashlib.sha256(bank.read_bytes()).hexdigest()
    if bank_sha!='525d90307cf738b2051aad840abf5bb36ebe8460ad4cf7c0f1a3635b8d8a2823':raise ValueError('Unexpected recovery expert bank')
    reset=root/'src/assets/motions/g1/reset_banks/l4_amp_train.npz'
    record=dict(source_commit=a.source_commit,branch='lyd-c-controlled-ablations',expert_sha256=bank_sha,
        reset_sha256=hashlib.sha256(reset.read_bytes()).hexdigest(),preflight=a.preflight,
        num_envs=4096,steps_per_env=24,iterations=20 if a.preflight else 100000,
        created_at=time.strftime('%Y-%m-%d %H:%M:%S%z'),runs=[])
    record['planned_runs'] = [dict(arm=arm, seed=seed, gpu=i if a.preflight else i*2+(seed==43))
        for i,arm in enumerate(ARMS) for seed in ([42] if a.preflight else [42,43,44])]
    with path.open('x') as f:json.dump(record,f,indent=2)
    lock=threading.Lock()
    def persist():
        tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(record,indent=2));tmp.replace(path)
    def worker(gpu,jobs):
        for arm,seed in jobs:
            run_id=uuid.uuid4().hex[:8];label=f'{arm}-seed{seed}'
            command=[sys.executable,'-u','scripts/train.py',f'Unitree-G1-AMP-Controlled-{ARMS[arm]}-Flat',
                '--env.scene.num-envs','4096','--agent.num-steps-per-env','24',
                '--agent.max-iterations',str(record['iterations']),'--agent.save-interval','1000',
                '--agent.logger','tensorboard' if a.preflight else 'wandb',
                '--agent.wandb-project','lyd-multi-amp','--agent.run-name',
                ('preflight-' if a.preflight else '')+f'AMP-{label}-controlled-scratch-4096x24',
                '--agent.seed',str(seed),'--agent.resume','False','--gpu-ids',f'[{gpu}]','--enable-nan-guard','True']
            env=os.environ.copy();env.update(MUJOCO_GL='egl',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',
                WANDB_BASE_URL='https://api.wandb.ai',WANDB_USERNAME='tabletennis',WANDB_RUN_ID=run_id,
                WANDB_RUN_GROUP='amp-c-controlled-20261030',WANDB_MODE='online')
            item=dict(arm=arm,seed=seed,gpu=gpu,run_id=run_id,command=command,log=str(out/(label+'.log')),
                      wandb=None if a.preflight else f'https://wandb.ai/tabletennis/lyd-multi-amp/runs/{run_id}')
            with open(item['log'],'w') as f:
                proc=subprocess.Popen(command,cwd=root,env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
            with lock:
                item.update(pid=proc.pid,status='running');record['runs'].append(item);persist()
            rc=proc.wait()
            with lock:
                item.update(exit_code=rc,status='completed' if rc==0 else 'failed');persist()
            if rc:raise RuntimeError(f'{label} failed with {rc}; worker queue stopped')
    jobs=([(i,[(arm,42)]) for i,arm in enumerate(ARMS)] if a.preflight else
          [(i*2,[(arm,42),(arm,44)]) for i,arm in enumerate(ARMS)]+
          [(i*2+1,[(arm,43)]) for i,arm in enumerate(ARMS)])
    with ThreadPoolExecutor(max_workers=len(jobs)) as pool:
        futures=[pool.submit(worker,gpu,q) for gpu,q in jobs]
        for f in futures:f.result()

if __name__=='__main__':main()
