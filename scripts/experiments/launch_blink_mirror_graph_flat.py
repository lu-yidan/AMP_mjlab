import os,subprocess,json
from pathlib import Path
root=Path('/root/workplace/lyd-blink-mirror-graph-flat-20261107');out=root/'records';out.mkdir(exist_ok=True)
name='BlinkCore-Flat-SharedAMP-SpeedCurriculum-2p5Steps-WM1024-Structured-Muon-EMA0995-Mirror50-CUDAGraph'
base=dict(os.environ,WANDB_MODE='offline',CUDA_VISIBLE_DEVICES='5',PYTHONPATH='/root/workplace/lyd-multi-discriminator-amp/runtime:.:/root/autodl-tmp/cra-throughput-20261014/.venv/lib/python3.12/site-packages',MUJOCO_GL='egl',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',TORCHINDUCTOR_COMPILE_THREADS='2',CPATH=str(root/'runtime-include/python3.12'))
common=['/root/autodl-tmp/cra-throughput-20261014/python/bin/python3.12','-u','scripts/train_amp_supermimic.py','--flat-discriminator','shared','--iteration-speed-curriculum','--blink-core','--blink-mirror-graph','--sim-steps','2.5','--world-model-width','1024','--wm-updates','5','--ema-decay','.995','--num-envs','4096','--experiment-name',name,'--seed','42']
for smoke in [True,False]:
 d=out/(('preflight-4096-' if smoke else '')+name);d.mkdir(exist_ok=False)
 cmd=common+['--output',str(d),'--iterations','12' if smoke else '40000']+(['--smoke'] if smoke else ['--wandb'])
 with (d/'stdout.log').open('w')as f:
  p=subprocess.Popen(cmd,cwd=root,env=base,stdout=f,stderr=subprocess.STDOUT);(d/'process.json').write_text(json.dumps({'pid':p.pid,'cmd':cmd}));rc=p.wait()
 (d/'exit.json').write_text(json.dumps({'returncode':rc}));print('smoke' if smoke else 'training',rc,flush=True)
 if rc:raise SystemExit(rc)
 if smoke:
  rows=[json.loads(x) for x in (d/'supermimic_log.jsonl').read_text().splitlines()]
  if not any(r.get('graph/wm_captured',0)==1 and r.get('graph/policy_captured',0)==1 for r in rows):
   raise RuntimeError('Smoke ended without both custom update graphs captured')

