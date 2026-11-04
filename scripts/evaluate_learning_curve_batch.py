"""Run checkpoint learning curves on one GPU per worker; persist every result."""
import argparse,concurrent.futures,hashlib,json,os,queue,subprocess,time
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--directory',type=Path,required=True);p.add_argument('--python',required=True);p.add_argument('--gpus',type=int,nargs='+',required=True);a=p.parse_args()
root=Path.cwd();out=a.directory.resolve();jobs=json.loads((out/'jobs.json').read_text());pending=queue.Queue()
for job in jobs:pending.put(job)
task='Unitree-G1-AMP-I-Shared-TerrainCritic'
def worker(gpu):
 while True:
  try:j=pending.get_nowait()
  except queue.Empty:return
  name=j['name'];ck=out/'checkpoints'/(name+'.pt');dest=out/name;dest.mkdir(exist_ok=True)
  deadline=time.time()+1800
  while not ck.exists():
   if time.time()>deadline:raise TimeoutError(f'Checkpoint upload missing: {ck}')
   time.sleep(2)
  report=dict(j,gpu=gpu,status='running',checkpoint_sha256=hashlib.sha256(ck.read_bytes()).hexdigest(),steps=[])
  manifest=dest/'job.json'
  env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),MUJOCO_GL='egl',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2')
  common=['--backend',j['backend'],'--checkpoint',str(ck),'--name',name]
  commands=[('recovery',[a.python,'-u','scripts/evaluate_amp_common.py',*common,'--task',task,'--bank','logs/eval100k_single/validation.npz','--output',str(dest/'recovery')]),('commands',[a.python,'-u','scripts/evaluate_amp_commands_latest.py',*common,'--task',task,'--output',str(dest/'commands')])]
  for deg,height in [(15,.1),(20,.15),(25,.2)]:
   for speed in [1,2]:
    kind=f'terrain-{deg}deg-{speed}mps';commands.append((kind,[a.python,'-u','scripts/evaluate_record_k_terrain.py',*common,'--output',str(dest/kind),'--speed',str(speed),'--slope-degrees',str(deg),'--step-height',str(height),'--no-video']))
  for kind,cmd in commands:
   start=time.time()
   with open(dest/(kind+'.log'),'w') as f:
    proc=subprocess.Popen(cmd,cwd=root,env=env,stdout=f,stderr=subprocess.STDOUT)
    step=dict(kind=kind,pid=proc.pid,command=cmd);report['steps'].append(step);manifest.write_text(json.dumps(report,indent=2));step['returncode']=proc.wait();step['seconds']=time.time()-start
   if step['returncode']:
    report['status']='failed';break
  else:report['status']='completed'
  manifest.write_text(json.dumps(report,indent=2));print(name,report['status'],flush=True)
with concurrent.futures.ThreadPoolExecutor(max_workers=len(a.gpus)) as pool:list(pool.map(worker,a.gpus))
