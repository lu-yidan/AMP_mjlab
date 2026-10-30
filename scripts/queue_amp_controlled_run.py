"""Defer an interrupted H seed43 until H seed44 completes, preserving two free GPUs."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time
import uuid


def main():
    p=argparse.ArgumentParser();p.add_argument('--records',required=True);a=p.parse_args()
    records=Path(a.records).resolve();path=records/'deferred-H43.json'
    state=json.loads(path.read_text());state.update(status='queued',gpu=4,dependency='H seed44 completed')
    def save():
        tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(state,indent=2));tmp.replace(path)
    save()
    while True:
        launch=json.loads((records/'launch.json').read_text())
        h=[r for r in launch['runs'] if r['arm']=='H' and r['seed'] in [42,44]]
        if any(r['status']=='failed' for r in h):
            state['status']='blocked_by_failed_dependency';save();return
        ready=any(r['seed']==44 and r['status']=='completed' for r in h)
        if ready:
            used=subprocess.check_output(['nvidia-smi','-i','4','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
            if not used:break
        time.sleep(30)
    old=state['stopped_run'];command=list(old['command'])
    command[command.index('--gpu-ids')+1]='[4]'
    run_id=uuid.uuid4().hex[:8];env=os.environ.copy()
    env.update(MUJOCO_GL='egl',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',
        WANDB_BASE_URL='https://api.wandb.ai',WANDB_USERNAME='tabletennis',WANDB_RUN_ID=run_id,
        WANDB_RUN_GROUP='amp-c-controlled-20261030',WANDB_MODE='online')
    root=records.parents[1]
    with (records/'H-seed43-deferred.log').open('w') as f:
        child=subprocess.Popen(command,cwd=root,env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
    state.update(status='running',pid=child.pid,run_id=run_id,command=command,
                 wandb=f'https://wandb.ai/tabletennis/lyd-multi-amp/runs/{run_id}');save()
    rc=child.wait();state.update(status='completed' if rc==0 else 'failed',exit_code=rc);save()

if __name__=='__main__':main()
