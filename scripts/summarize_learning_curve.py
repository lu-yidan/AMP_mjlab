"""Audit and plot matched checkpoint evaluation curves by iterations and wall time."""
import argparse,json
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser();p.add_argument('--directory',type=Path,required=True);a=p.parse_args();root=a.directory.resolve()
jobs=json.load(open(root/'jobs.json'));times=json.load(open(root/'training_times.json'));rows=[];hashes=set()
for job in jobs:
 name=job['name'];d=root/name;manifest=json.load(open(d/'job.json'));assert manifest['status']=='completed',name
 rec=json.load(open(d/'recovery'/f'{name}.json'));cmd=json.load(open(d/'commands'/f'{name}.json'));tr=np.load(d/'recovery'/f'{name}_traces.npz');hashes.add(rec['initial_state_hash'])
 assert np.isfinite(tr['height']).all() and np.isfinite(tr['gravity_z']).all(),name
 command_tr=np.load(d/'commands'/f'{name}_traces.npz');assert all(np.isfinite(command_tr[k]).all() for k in command_tr.files),name
 first=tr['first_success_s'][:1268];r=rec['recovery'];assert r['n']==1268
 trans=[v for k,v in cmd['cases'].items() if k not in ['stand','yaw_left','yaw_right']]
 row=dict(name=name,iteration=job['iteration'],**times[name],recovery_success_percent=r['success_1s_percent'],recovery_within5s_percent=float(np.mean(first<=5)*100),recovery_mean_success_time_s=r['mean_success_time_s'],recovery_last5s_upright_percent=r['last5s_upright_percent'],recovery_first3s_joint_acc=rec['motion_quality']['first3s']['joint_acc_control_rms_rad_s2'],recovery_last5s_joint_acc=rec['motion_quality']['last5s']['joint_acc_control_rms_rad_s2'],command_xy_mae=float(np.mean([v['mae'][:2] for v in trans])),command_yaw_mae=float(np.mean([v['mae'][2] for v in cmd['cases'].values()])),command_min_upright_percent=min(v['upright_percent'] for v in cmd['cases'].values()),command_forward3=cmd['cases']['forward3'],terrain={})
 for deg in [15,20,25]:
  group=[]
  for speed in [1,2]:
   td=json.load(open(d/f'terrain-{deg}deg-{speed}mps'/'terrain.json'));group.append(td)
   terrain_tr=np.load(d/f'terrain-{deg}deg-{speed}mps'/'traces.npz');assert all(np.isfinite(terrain_tr[k]).all() for k in terrain_tr.files),name
  row['terrain'][str(deg)]={}
  for label,keys in [('slope',['slope','slope_inverted']),('stairs',['stairs','stairs_inverted'])]:
   cases=[x['cases'][k] for x in group for k in keys]
   row['terrain'][str(deg)][label]={'sustained_falls':sum(c['sustained_fall_trials'] for c in cases),'trials':sum(c['trials'] for c in cases),'vx_mae':float(np.mean([c['mae'][0] for c in cases]))}
 rows.append(row)
assert len(hashes)==1,hashes
# Every geometry/speed must share initial states across checkpoints.
for deg in [15,20,25]:
 for speed in [1,2]:
  assert len({json.load(open(root/j['name']/f'terrain-{deg}deg-{speed}mps'/'terrain.json'))['initial_state_hash'] for j in jobs})==1
report={'evaluation_host':'ssh -p17535 root@192.168.11.91','checkpoint_audit':json.load(open(root/'checkpoint_audit.json')),'training_times':times,'experiment_names':{'AMP-shared-terrain':'AMP－分层reset＋共享单判别器＋恢复均衡采样＋复杂地形高程critic','SuperMimic-original-height':'SuperMimic－共享单判别器＋复杂地形＋原高度奖励','SuperMimic-half-height':'SuperMimic－共享单判别器＋复杂地形＋半强度高度奖励'},'protocol':'12 checkpoints, seed42; flat1268 fixed recovery poses,12 command cases with8 standing jitters each; terrain15/20/25deg paired with10/15/20cm stairs, both upright/inverted,1/2m/s,8 trials per case.','limitations':['Single training seed and fixed evaluation map; descriptive comparisons, not algorithm-level statistical significance.','Training used easier terrain (max11.3deg,7.5cm), so harder terrain is an out-of-distribution stress test.','Actual wall-clock includes setup/export/logging and shared-machine contention; not an exclusive-GPU timing benchmark.','AMP saved index includes one extra rollout vs SuperMimic, difference0.01% at10k.','Joint acceleration is a physical smoothness proxy, not a sufficient measure of naturalness; failure to move can also lower it.','Only10k-spaced checkpoints: earliest observed attainment is not exact convergence time; connecting plot lines are visual guides, not measured interpolation.','All12 checkpoints were rerun on the17535 RTX4090 server; results are not mixed with prior local4070Ti trials. GPU/runtime floating-point differences can change individual contact-rich rollouts.'],'rows':rows}
reportpath=Path('reports/learning_curve_comparison_20261104.json');reportpath.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
models=[('AMP-shared-terrain','AMP: stratified reset + shared discriminator + terrain critic'),('SuperMimic-original-height','SuperMimic: shared discriminator + original height reward'),('SuperMimic-half-height','SuperMimic: shared discriminator + half height reward')]
panels=[('Recovery success (20s), %',lambda r:r['recovery_success_percent']),('Recovery within 5s, %',lambda r:r['recovery_within5s_percent']),('Command xy MAE, m/s (lower better)',lambda r:r['command_xy_mae']),('Recovery first3s joint acceleration, rad/s²',lambda r:r['recovery_first3s_joint_acc']),('10cm stair falls /32 (lower better)',lambda r:r['terrain']['15']['stairs']['sustained_falls']),('20-degree slope falls /32 (lower better)',lambda r:r['terrain']['20']['slope']['sustained_falls'])]
for axis in ['iteration','wall_clock_hours']:
 fig,axs=plt.subplots(2,3,figsize=(15,8))
 for ax,(title,value) in zip(axs.flat,panels):
  for prefix,label in models:
   rr=sorted([r for r in rows if r['name'].startswith(prefix)],key=lambda r:r['iteration']);ax.plot([r[axis]/1000 if axis=='iteration' else r[axis] for r in rr],[value(r) for r in rr],marker='o',label=label)
  ax.set_title(title,fontsize=11);ax.set_xlabel('Training iterations (k)' if axis=='iteration' else 'Observed training wall time (hours)');ax.grid(alpha=.3)
 handles,labels=axs[0,0].get_legend_handles_labels();fig.legend(handles,labels,loc='lower center',ncol=1,fontsize=10);fig.suptitle('Matched learning curves | one training seed | harder-terrain stress test');fig.tight_layout(rect=(0,.12,1,.95));fig.savefig(root/f'learning_curve_{axis}.png',dpi=150);plt.close(fig)
for r in rows:print(r['name'], 'recovery',round(r['recovery_success_percent'],2),'5s',round(r['recovery_within5s_percent'],2),'xyMAE',round(r['command_xy_mae'],3),'upright',round(r['command_min_upright_percent'],1),'stairs10',r['terrain']['15']['stairs']['sustained_falls'],'slope20',r['terrain']['20']['slope']['sustained_falls'])
