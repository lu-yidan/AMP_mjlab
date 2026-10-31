"""Run I/J after existing flat controls, reserving GPUs 5 and 6."""
import argparse,json,os,subprocess,sys,time,uuid,threading,hashlib,re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


def main():
 p=argparse.ArgumentParser();p.add_argument('--dependency',required=True);p.add_argument('--output',required=True);p.add_argument('--source-commit',required=True);a=p.parse_args()
 root=Path(__file__).resolve().parents[1];out=Path(a.output).resolve();out.mkdir(parents=True,exist_ok=True);path=out/'launch.json'
 bank=root/'logs/amp_stability/smp_recovery_train.npz';sha=hashlib.sha256(bank.read_bytes()).hexdigest()
 assert sha=='525d90307cf738b2051aad840abf5bb36ebe8460ad4cf7c0f1a3635b8d8a2823'
 jobs=[dict(label='I-Shared-TerrainCritic',gpu=1,dependencies=[['F',43]]),dict(label='J-Dual-TerrainCritic',gpu=0,dependencies=[['F',42],['F',44]])]
 for j in jobs:j.update(status='queued',seed=42,iterations=100000,num_envs=4096,rollout_steps=24)
 state=dict(source_commit=a.source_commit,expert_sha256=sha,dependency=a.dependency,reserved_gpus=[5,6],jobs=jobs)
 with path.open('x') as f:json.dump(state,f,indent=2)
 lock=threading.Lock()
 def save():
  with lock:
   temp=path.with_suffix('.tmp');temp.write_text(json.dumps(state,indent=2));temp.replace(path)
 def worker(j):
  try:
   while True:
    dep=json.loads(Path(a.dependency).read_text());runs=dep['runs']
    matches=[next((r for r in runs if [r['arm'],r['seed']]==d),None) for d in j['dependencies']]
    if any(r and r['status']=='failed' for r in matches):j['status']='blocked_dependency_failed';save();return
    if all(r and r['status']=='completed' for r in matches):
     used=subprocess.check_output(['nvidia-smi','-i',str(j['gpu']),'--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
     if not used:break
    time.sleep(30)
   for preflight in [True,False]:
    name=j['label']+'-seed42-'+('preflight4096' if preflight else 'scratch-4096x24-100k')
    rid=uuid.uuid4().hex[:8];log=out/(name+'.log')
    cmd=[sys.executable,'-u','scripts/train.py','Unitree-G1-AMP-'+j['label'],'--env.scene.num-envs','4096','--agent.num-steps-per-env','24','--agent.max-iterations','20' if preflight else '100000','--agent.save-interval','1000','--agent.logger','tensorboard' if preflight else 'wandb','--agent.wandb-project','lyd-multi-amp','--agent.run-name',name,'--agent.seed','42','--agent.resume','False','--gpu-ids',f"[{j['gpu']}]",'--enable-nan-guard','True']
    env=os.environ.copy();env.update(MUJOCO_GL='egl',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',WANDB_BASE_URL='https://api.wandb.ai',WANDB_USERNAME='tabletennis',WANDB_RUN_ID=rid,WANDB_RUN_GROUP='amp-terrain-critic-20261101',WANDB_MODE='online')
    with log.open('w') as f:proc=subprocess.Popen(cmd,cwd=root,env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
    j.update(status='preflight' if preflight else 'running',pid=proc.pid,command=cmd,log=str(log),run_id=rid,wandb=None if preflight else f'https://wandb.ai/tabletennis/lyd-multi-amp/runs/{rid}');save()
    rc=proc.wait();j['exit_code']=rc
    if rc:j['status']='failed';save();return
    if preflight:
     skips=[float(v) for v in re.findall(r'Mean skipped_non_finite_batches loss:\s*([\d.]+)',log.read_text())]
     if len(skips)!=20 or max(skips)!=0:raise RuntimeError('Full-size preflight failed validation')
     j['preflight_verified']=True;save()
   j['status']='completed';save()
  except Exception as exc:j.update(status='scheduler_error',error=repr(exc));save();raise
 with ThreadPoolExecutor(max_workers=2) as pool:
  list(pool.map(worker,jobs))

if __name__=='__main__':main()
