import os,subprocess,json,concurrent.futures
from pathlib import Path
root=Path('/root/workplace/lyd-supermimic-flat-20261107');out=root/'records';out.mkdir(exist_ok=True)
py='/root/autodl-tmp/cra-throughput-20261014/python/bin/python3.12'
base=dict(os.environ,PYTHONPATH='/root/workplace/lyd-multi-discriminator-amp/runtime:.:/root/autodl-tmp/cra-throughput-20261014/.venv/lib/python3.12/site-packages',MUJOCO_GL='egl',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',TORCHINDUCTOR_COMPILE_THREADS='2',CPATH=str(root/'runtime-include/python3.12'))
specs=[('SuperMimic-Flat-SharedAMP-SpeedCurriculum-2p5Steps-WM2048',0,'shared',2.5),('SuperMimic-Flat-SharedAMP-SpeedCurriculum-5Steps-WM2048',1,'shared',5),('SuperMimic-Flat-DualAMP-SpeedCurriculum-2p5Steps-WM2048',2,'dual',2.5)]
def run(s):
 name,gpu,disc,steps=s
 common=[py,'-u','scripts/train_amp_supermimic.py','--flat-discriminator',disc,'--iteration-speed-curriculum','--sim-steps',str(steps),'--world-model-width','2048','--num-envs','4096','--experiment-name',name,'--seed','42']
 for smoke in [True,False]:
  d=out/(('preflight-' if smoke else '')+name);d.mkdir(exist_ok=False)
  cmd=common+['--output',str(d),'--iterations','8' if smoke else '40000']+(['--smoke'] if smoke else ['--wandb'])
  with (d/'stdout.log').open('w')as f:
   p=subprocess.Popen(cmd,cwd=root,env=dict(base,CUDA_VISIBLE_DEVICES=str(gpu)),stdout=f,stderr=subprocess.STDOUT)
   (d/'process.json').write_text(json.dumps({'pid':p.pid,'command':cmd}));rc=p.wait()
  (d/'exit.json').write_text(json.dumps({'returncode':rc}));print(name,'smoke' if smoke else 'training',rc,flush=True)
  if rc:return
with concurrent.futures.ThreadPoolExecutor(3)as ex:list(ex.map(run,specs))
