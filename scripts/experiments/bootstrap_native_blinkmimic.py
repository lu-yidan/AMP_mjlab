"""Download pinned native dependencies locally, transfer, then validate on 17535."""
import json,os,subprocess,time
from pathlib import Path
root=Path('/tmp/supermimic-review/blinkmimic')
log=Path('/home/luyd/workspace/AMP_mjlab/logs/blink_native_bootstrap');log.mkdir(exist_ok=True)
def status(stage):
 (log/'status.json').write_text(json.dumps({'stage':stage,'time':time.time()}));print(stage,flush=True)
try:
 status('frozen-dependency-install')
 subprocess.run(['/home/luyd/.local/bin/uv','sync','--frozen','--python','/home/luyd/miniconda3/bin/python'],cwd=root,check=True,env=dict(os.environ,UV_HTTP_TIMEOUT='600'))
 status('transfer-pinned-site-packages')
 subprocess.run(['rsync','-az','-e','ssh -p 17535',str(root/'.venv/lib/python3.12/site-packages')+'/', 'root@192.168.11.91:/root/workplace/blinkmimic-d264658/.venv/lib/python3.12/site-packages/'],check=True)
 status('check-native-versions')
 subprocess.run(['ssh','-p','17535','root@192.168.11.91',"cd /root/workplace/blinkmimic-d264658 && PYTHONPATH=src .venv/bin/python -c 'import torch,warp,mjlab; assert torch.__version__==\"2.9.0+cu128\"; assert warp.__version__==\"1.16.0\"; print(torch.__version__,warp.__version__,mjlab.__file__)'"],check=True)
 status('start-native-sanity-training-evaluation')
 subprocess.run(['ssh','-p','17535','root@192.168.11.91',"cd /root/workplace/blinkmimic-d264658 && nohup env WANDB_MODE=offline .venv/bin/python -u native_blink_run.py > native_pipeline.log 2>&1 < /dev/null &"],check=True,timeout=30)
 status('native-process-dispatched-check-remote-log')
except Exception as e:
 status('failed: '+repr(e));raise
