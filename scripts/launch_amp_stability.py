"""Launch the four approved scratch runs on separate GPUs; record immutable inputs."""
import argparse, hashlib, json, os, subprocess, sys, uuid
from pathlib import Path

TASKS = [
    ('A-single', 'Unitree-G1-AMP-Stratified-Tracking-Flat'),
    ('B-stable', 'Unitree-G1-MultiAMP-Stable-Tracking-Flat'),
    ('C-phase-smp', 'Unitree-G1-MultiAMP-Phase-Tracking-Flat'),
    ('D-smooth-smp', 'Unitree-G1-MultiAMP-Smooth-Tracking-Flat'),
]


def main():
    p=argparse.ArgumentParser();p.add_argument('--source-commit',required=True);p.add_argument('--output',required=True);p.add_argument('--gpus',type=int,nargs=4,default=[0,1,2,3]);p.add_argument('--only',nargs='+',choices=[t[0] for t in TASKS]);a=p.parse_args()
    out=Path(a.output).resolve();out.mkdir(parents=True,exist_ok=True)
    manifest=out/'launch.json'
    if manifest.exists():raise RuntimeError('Existing launch manifest; refusing duplicate launch')
    root=Path(__file__).resolve().parents[1]
    bank=root/'logs/amp_stability/smp_recovery_train.npz'
    sha=hashlib.sha256(bank.read_bytes()).hexdigest()
    audit=json.loads((root/'reports/amp_stability_smp_expert_audit.json').read_text())
    if sha!=audit['sha256']:raise ValueError('Expert bank checksum mismatch')
    records=dict(source_commit=a.source_commit,branch='lyd-amp-recovery-stability',root=str(root),expert_bank_sha256=sha,python=sys.executable,runs=[])
    manifest.write_text(json.dumps(records,indent=2))
    for gpu,(label,task) in zip(a.gpus,TASKS):
        if a.only and label not in a.only:continue
        run_id=uuid.uuid4().hex[:8]
        code='AMP-'+label[0]
        name=code+'-stability-'+label.lower()+'-scratch-4096x24-100k-seed42'
        cmd=[sys.executable,'-u','scripts/train.py',task,'--env.scene.num-envs','4096',
             '--agent.num-steps-per-env','24','--agent.max-iterations','100000',
             '--agent.save-interval','1000','--agent.logger','wandb','--agent.wandb-project','lyd-multi-amp',
             '--agent.run-name',name,'--agent.seed','42','--agent.resume','False','--gpu-ids',f'[{gpu}]']
        env=os.environ.copy();env.update(WANDB_BASE_URL='https://api.wandb.ai',WANDB_USERNAME='tabletennis',
              WANDB_RUN_ID=run_id,WANDB_RUN_GROUP='amp-recovery-stability-20261027',WANDB_MODE='online')
        log=out/(label+'.log')
        with log.open('w') as f:
            proc=subprocess.Popen(cmd,cwd=root,env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
        records['runs'].append(dict(code=code,label=label,task=task,gpu=gpu,pid=proc.pid,run_id=run_id,
              wandb='https://wandb.ai/tabletennis/lyd-multi-amp/runs/'+run_id,command=cmd,log=str(log)))
        manifest.write_text(json.dumps(records,indent=2))
    print(manifest.read_text())
if __name__=='__main__':main()
