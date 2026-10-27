"""Launch one AMP-E scratch run after the numerical and full-size smoke gates."""
import argparse, hashlib, json, math, os, subprocess, sys, uuid
from pathlib import Path


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--source-commit',required=True);p.add_argument('--gpu',type=int,default=4)
    p.add_argument('--output',default='records/amp-e-20261027')
    a=p.parse_args();root=Path(__file__).resolve().parents[1];out=root/a.output
    out.mkdir(parents=True,exist_ok=True);manifest=out/'launch.json'
    if manifest.exists():raise RuntimeError('Launch already recorded; refusing duplicate job')
    verification=json.loads((root/'logs/amp_e_verify/verification.json').read_text())
    if not verification.get('passed') or 'gradient_fd_max_abs' not in verification:
        raise RuntimeError('Numerical/gradient gate missing or failed')
    smoke=[json.loads(x) for x in (root/'logs/amp_e_smoke/supermimic_log.jsonl').read_text().splitlines()]
    if not any(r.get('it',0)>=3 and math.isfinite(r.get('policy/imag_reward',math.nan)) and
               math.isfinite(r.get('policy/grad_norm',math.nan)) for r in smoke):
        raise RuntimeError('Full-size actor/world-model smoke did not complete')
    used=int(subprocess.check_output(['nvidia-smi','-i',str(a.gpu),
        '--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).strip())
    if used>128:raise RuntimeError(f'GPU {a.gpu} is occupied ({used} MiB); no process will be stopped')
    run_id=uuid.uuid4().hex[:8]
    logdir=root/'logs/amp_e_100k'/run_id
    cmd=[sys.executable,'-u','scripts/train_amp_supermimic.py','--num-envs','4096',
        '--sim-steps','24','--iterations','100000','--wandb','--output',str(logdir)]
    env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=str(a.gpu),
        WANDB_BASE_URL='https://api.wandb.ai',WANDB_RUN_ID=run_id,WANDB_MODE='online',
        CPATH=str(root/'runtime-include/python3.12'),TORCHINDUCTOR_COMPILE_THREADS='2',OMP_NUM_THREADS='8')
    log=out/'train.log'
    with log.open('w') as f:
        proc=subprocess.Popen(cmd,cwd=root,env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
    record=dict(code='AMP-E',branch='lyd-amp-c-supermimic',source_commit=a.source_commit,
        upstream_commit='eb7a82bedad9e9e25a93cff3318d24e4a9f88cb2',root=str(root),gpu=a.gpu,pid=proc.pid,
        run_id=run_id,wandb='https://wandb.ai/tabletennis/lyd-multi-amp/runs/'+run_id,
        command=cmd,log=str(log),output=str(logdir),verification=verification,smoke=smoke[-3:],
        python=sys.executable,expert_sha256=hashlib.sha256((root/'logs/amp_stability/smp_recovery_train.npz').read_bytes()).hexdigest(),
        runtime_note='CPython 3.12.13; isolated CPython 3.12.9 public extension headers via CPATH for Triton (same 3.12 ABI).')
    manifest.write_text(json.dumps(record,indent=2));print(manifest.read_text())
if __name__=='__main__':main()
