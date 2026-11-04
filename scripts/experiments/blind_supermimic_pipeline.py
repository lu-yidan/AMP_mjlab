"""17535-only native fidelity gate followed by isolated blind-actor ablations.

Run with the existing AMP Python environment (for wandb); native subprocesses
use their own frozen uv environment. Fail closed; never restart this over a run.
"""
import csv
import json
import os
from pathlib import Path
import subprocess
import time

ROOT = Path('/root/workplace/lyd-supermimic-efficiency-20261104')
NATIVE = Path('/root/workplace/blinkmimic-d264658')
PY = '/root/autodl-tmp/cra-throughput-20261014/python/bin/python3.12'
RECORDS = ROOT / 'records'
NATIVE_RUN = NATIVE / 'logs/native-main-seed0'
BASE = dict(os.environ, MUJOCO_GL='egl', OMP_NUM_THREADS='4',
            MKL_NUM_THREADS='4', TORCHINDUCTOR_COMPILE_THREADS='2',
            CPATH=str(ROOT / 'runtime-include/python3.12'))
NENV = dict(BASE, CUDA_VISIBLE_DEVICES='0', PYTHONPATH='src',
            PYTHON=str(NATIVE / '.venv/bin/python'))
SPECS = [
    ('SuperMimic-BlindActor-SharedAMP-Terrain-Control-24Steps-WM2048', 2, []),
    ('SuperMimic-BlindActor-SharedAMP-Terrain-EMA0995-24Steps-WM2048', 3, ['--ema-decay', '.995']),
    ('SuperMimic-BlindActor-SharedAMP-Terrain-UTD04-2p5Steps-WM2048', 4, ['--sim-steps', '2.5']),
    ('SuperMimic-BlindActor-SharedAMP-Terrain-24Steps-WM1024', 5, ['--world-model-width', '1024']),
]


def status(stage, **extra):
    data = dict(stage=stage, time=time.strftime('%Y-%m-%d %H:%M:%S'), **extra)
    path = RECORDS / 'pipeline-status.json'
    path.with_suffix('.tmp').write_text(json.dumps(data, indent=2))
    path.with_suffix('.tmp').replace(path)
    print(json.dumps(data), flush=True)


def command(args, label, env=NENV):
    status(label)
    with (RECORDS / (label + '.log')).open('w') as stream:
        subprocess.run(args, cwd=NATIVE, env=env, stdout=stream,
                       stderr=subprocess.STDOUT, check=True)


def require_idle(gpu):
    uuid = subprocess.check_output(['nvidia-smi', '-i', str(gpu),
        '--query-gpu=uuid', '--format=csv,noheader'], text=True).strip()
    used = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid',
        '--format=csv,noheader'], text=True)
    if uuid in used:
        raise RuntimeError(f'GPU {gpu} occupied; refusing to compete with another job')


def main():
    # O_EXCL prevents accidental duplicate launches, including after completion.
    with (RECORDS / 'pipeline.lock').open('x') as f:
        f.write(str(os.getpid()))
    for p in RECORDS.glob('preflight-*/exit.json'):
        assert json.loads(p.read_text())['returncode'] == 0, str(p)
    assert len(list(RECORDS.glob('preflight-*/exit.json'))) == 5
    command(['/root/tools/uv', 'sync', '--frozen', '--python', PY],
            'install-native', dict(BASE, UV_HTTP_TIMEOUT='600'))
    native_py = str(NATIVE / '.venv/bin/python')
    command([native_py, 'tools/urdf_to_mjcf.py'], 'convert-native-robot')
    require_idle(0)
    command([native_py, 'tools/test_env.py'], 'native-env-sanity')
    import wandb
    run = wandb.init(entity='tabletennis', project='lyd-multi-amp',
        name='Native-BlinkMimic-LAFAN1-40Clips-ReleasedMain-Seed0',
        config={'upstream_commit': 'd264658d6c9ce61fb52f9f0f8985a369f3ccc409',
                'task': 'reference motion tracking; separate fidelity benchmark',
                'preset': 'main', 'seed': 0, 'dependency_lock': 'upstream uv.lock'})
    status('native-training', wandb_url=run.url)
    last = -1
    with (RECORDS / 'native-train.log').open('w') as f:
        proc = subprocess.Popen(['bash', '-o', 'pipefail', 'scripts/train.sh',
            str(NATIVE_RUN), '--seed', '0'], cwd=NATIVE, env=NENV, stdout=f, stderr=subprocess.STDOUT)
        while True:
            path = NATIVE_RUN / 'train_log.csv'
            if path.exists():
                for row in csv.DictReader(path.read_text().splitlines()):
                    try:
                        vals = {k: float(v) for k, v in row.items()}
                        iteration = int(vals['iteration'])
                    except (ValueError, TypeError):
                        continue  # skip a partially written final line
                    if iteration > last:
                        run.log(vals, step=iteration)
                        last = iteration
            if proc.poll() is not None:
                if proc.returncode:
                    run.finish(exit_code=proc.returncode)
                    raise RuntimeError(f'Native training failed: {proc.returncode}')
                break
            time.sleep(10)
    assert last >= 3400, f'Training ended early: {last}'
    command([native_py, 'tools/eval_batch.py', '--run_dir', str(NATIVE_RUN),
             '--group', '13', '--reps', '4'], 'native-evaluation')
    rows = list(csv.DictReader((NATIVE_RUN / 'eval_r4.csv').open()))
    streak = 0
    passed = False
    for row in sorted(rows, key=lambda r: int(r['iteration'])):
        good = float(row['norm_return']) >= .82 and float(row['n_failed']) == 0
        streak = streak + 1 if good else 0
        passed |= streak >= 3
    run.summary['native_gate_passed'] = passed
    run.summary['evaluated_checkpoints'] = len(rows)
    run.finish()
    if not passed:
        raise RuntimeError('Native fidelity gate failed: need >=0.82 return and zero failed clips for 3 consecutive checkpoints')
    jobs = []
    for name, gpu, flags in SPECS:
        require_idle(gpu)
        out = RECORDS / name
        out.mkdir(exist_ok=False)
        args = [PY, '-u', 'scripts/train_amp_supermimic.py', '--terrain',
                '--num-envs', '4096', '--iterations', '40000', '--seed', '42',
                '--save-interval', '1000', '--wandb', '--output', str(out),
                '--experiment-name', name] + flags
        with (out / 'stdout.log').open('w') as f:
            p = subprocess.Popen(args, cwd=ROOT, env=dict(BASE, CUDA_VISIBLE_DEVICES=str(gpu)),
                                 stdout=f, stderr=subprocess.STDOUT, start_new_session=True)
        jobs.append(dict(name=name, gpu=gpu, pid=p.pid, command=args))
        (RECORDS / 'ablation-jobs.json').write_text(json.dumps(jobs, indent=2))
    status('ablations-launched', jobs=jobs)


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        status('blocked', error=repr(exc))
        raise
