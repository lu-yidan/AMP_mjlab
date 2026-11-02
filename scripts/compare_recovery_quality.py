"""Summarize matched recovery quality evaluations; no causal significance claims."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

p = argparse.ArgumentParser()
p.add_argument('--directory', type=Path, required=True)
p.add_argument('--bank', type=Path, required=True)
p.add_argument('--output', type=Path, required=True)
a = p.parse_args()
bank = np.load(a.bank)
n = len(bank['qpos'])
names = ('C-71k', 'K-61k')
results = {name: json.loads((a.directory / (name+'.json')).read_text()) for name in names}
assert len({r['initial_state_hash'] for r in results.values()}) == 1
assert all(r['bank_sha256'] == hashlib.sha256(a.bank.read_bytes()).hexdigest() for r in results.values())
traces = {name: np.load(a.directory / (name+'_traces.npz')) for name in names}
assert all(np.isfinite(z[k]).all() for z in traces.values() for k in z.files)
masks = {'all': np.ones(n, bool), 'low': bank['stages']=='low'}
for direction in ('supine', 'prone', 'left_side_down', 'right_side_down'):
    masks['procedural/'+direction] = (bank['sources']=='procedural') & (bank['labels']==direction) & (bank['stages']=='low')
comparison = {}
for group, mask in masks.items():
    comparison[group] = {'n': int(mask.sum()), 'metrics': {}}
    for key in traces[names[0]].files:
        if '__' not in key:
            continue
        values = {name: float(traces[name][key][:n][mask].mean()) for name in names}
        values['K_change_percent'] = 100*(values[names[1]]/values[names[0]]-1) if values[names[0]] else None
        comparison[group]['metrics'][key] = values
report = dict(checkpoints={name: r['checkpoint'] for name,r in results.items()},
    initial_state_hash=results[names[0]]['initial_state_hash'],
    results=results, comparison=comparison, all_traces_finite=True,
    reward_audit={'extra_L4_recovery_guidance': False,
        'height_reward': 'Both use delayed-termination-only height shaping, multiplier 3.5 and weight 1.0; disabled when delay counter is zero.',
        'K_imagined_height_reward': 'AMPTask.transition uses the same delay gate and factor; terrain height is relative to measured/predicted ground.'},
    limitations=['C 71k vs K 61k: different algorithm, discriminator architecture and training terrain, not a matched training ablation.',
        'First 3s RMS includes both recovery and standing; faster recovery can lower this statistic.',
        '50Hz joint acceleration and foot resultant force do not measure physics-step peak whole-body collision impulse.',
        'Smoothness proxies do not establish human-like recovery choreography; video assessment remains necessary.',
        'Videos reuse verified matching nominal recordings, half speed; quantitative rollouts rerun locally.'])
a.output.parent.mkdir(parents=True,exist_ok=True)
a.output.write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(comparison['all'],indent=2))
