import os,subprocess,csv,time,json
from pathlib import Path
import wandb
root=Path('/root/workplace/blinkmimic-d264658');out=root/'logs/native-main-seed0';out.mkdir(parents=True,exist_ok=True)
env=dict(os.environ,CUDA_VISIBLE_DEVICES='3',PYTHONPATH='src',MUJOCO_GL='egl',OMP_NUM_THREADS='4',CPATH='/root/workplace/lyd-supermimic-efficiency-20261104/runtime-include/python3.12',PYTHON=str(root/'.venv/bin/python'))
py=str(root/'.venv/bin/python')
def run(cmd,name):
 with (out/(name+'.log')).open('w')as f:subprocess.run(cmd,cwd=root,env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
run([py,'tools/urdf_to_mjcf.py'],'convert');run([py,'tools/test_env.py'],'sanity')
w=wandb.init(entity='tabletennis',project='lyd-multi-amp',name='BlinkMimic-Native-Flat-LAFAN1-40Clips-Main-Seed0',config={'commit':'d264658d6c9ce61fb52f9f0f8985a369f3ccc409','task':'reference-motion tracking; not velocity-command AMP task','dependencies':'frozen uv.lock'})
(out/'run.json').write_text(json.dumps({'wandb_url':w.url,'pid':os.getpid()}))
with (out/'train-wrapper.log').open('w')as f:
 p=subprocess.Popen(['bash','-o','pipefail','scripts/train.sh',str(out),'--seed','0'],cwd=root,env=env,stdout=f,stderr=subprocess.STDOUT)
 last=-1
 while True:
  path=out/'train_log.csv'
  if path.exists():
   for r in csv.DictReader(path.read_text().splitlines()):
    try:v={k:float(x)for k,x in r.items()};it=int(v['iteration'])
    except (ValueError,TypeError):continue
    if it>last:w.log(v,step=it);last=it
  if p.poll() is not None:break
  time.sleep(10)
 if p.returncode:w.finish(exit_code=p.returncode);raise RuntimeError('native training failed')
run([py,'tools/eval_batch.py','--run_dir',str(out),'--reps','4','--group','13'],'evaluation')
rows=list(csv.DictReader((out/'eval_r4.csv').open()));streak=0;passed=False
for r in rows:
 streak=streak+1 if float(r['norm_return'])>=.82 and float(r['n_failed'])==0 else 0
 passed|=streak>=3
w.summary['fidelity_gate_passed']=passed;w.finish()
(out/'complete.json').write_text(json.dumps({'passed':passed,'iterations':last,'evaluated_checkpoints':len(rows)}))
