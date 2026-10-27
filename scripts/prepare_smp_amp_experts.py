"""Convert only SMP training clips to AMP features on the current robot model."""
import argparse, ast, hashlib, json
from pathlib import Path
import mujoco
import numpy as np
import torch
from mjlab.scene import Scene
from mjlab.utils.lab_api import math as mu
from src.tasks.amp_loco.config.g1.env_cfgs import g1_amp_flat_env_cfg
from src.tasks.amp_loco.config.g1.rl_cfg import g1_amp_ppo_runner_cfg
from scripts.csv_to_npz import MotionLoader
from rsl_rl.modules.stable_multi_amp import recovery_segments, phase_direction


def main():
    p=argparse.ArgumentParser();p.add_argument('--smp-root',required=True);p.add_argument('--train-bank',required=True);p.add_argument('--validation-bank',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();root=Path(a.smp_root)
    train=np.load(a.train_bank);val=np.load(a.validation_bank)
    clips=sorted(set(train['clips'][train['sources']=='natural']))
    held=set(val['clips'][val['sources']=='natural'])
    base=lambda x:x.removesuffix('__mirror')
    if set(map(base,clips)) & set(map(base,held)):raise ValueError('Mirrored source split leakage')
    tree=ast.parse((root/'scripts/csv_to_npz.py').read_text())
    names=next(ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.AnnAssign) and isinstance(n.target,ast.Name) and n.target.id=='JOINT_NAMES')
    if list(names)!=list(train['joint_names']):raise ValueError('CSV joint order differs from reset bank')
    cfg=g1_amp_flat_env_cfg();cfg.scene.num_envs=1
    scene=Scene(cfg.scene,device='cpu');m=scene.compile();d=mujoco.MjData(m)
    rl=g1_amp_ppo_runner_cfg();body_names=list(rl.amp_body_names)
    bids=[m.body('robot/'+n).id for n in body_names];anchor=m.body('robot/'+rl.amp_anchor_name).id
    qadr=np.array([m.joint('robot/'+n).qposadr[0] for n in names]);limits=np.array([m.joint('robot/'+n).range for n in names])
    aa=[];bb=[];gg=[];clipids=[];audit=[]
    jacp=np.empty((3,m.nv));jacr=np.empty((3,m.nv))
    for ci,name in enumerate(clips):
        path=root/'datasets/csv/getup_lafan6_sliced'/(name+'.csv')
        motion=MotionLoader(str(path),30,50,'cpu')
        q=motion.motion_dof_poss.numpy();n=len(q)
        violation=np.maximum(limits[:,0]-q,q-limits[:,1]).clip(min=0)
        # Never clip expert joint angles silently; reject incompatible clips.
        record=dict(clip=name,sha256=hashlib.sha256(path.read_bytes()).hexdigest(),frames=n,max_limit_violation=float(violation.max()))
        if not np.isfinite(q).all() or violation.max()>.1:
            record['rejected']='nonfinite or >0.1rad joint-limit violation';audit.append(record);continue
        qpos=np.tile(m.qpos0,(n,1));qpos[:,:3]=motion.motion_base_poss.numpy();qpos[:,3:7]=motion.motion_base_rots.numpy();qpos[:,qadr]=q
        qvel=np.zeros((n,m.nv))
        for t in range(n):
            lo=max(t-1,0);hi=min(t+1,n-1)
            mujoco.mj_differentiatePos(m,qvel[t],(hi-lo)/50,qpos[lo],qpos[hi])
        pos=[];quat=[];vel=[];ang=[];ap=[];aq=[]
        for t in range(n):
            d.qpos[:]=qpos[t];d.qvel[:]=qvel[t];mujoco.mj_forward(m,d)
            pos.append(d.xpos[bids].copy());quat.append(d.xquat[bids].copy());ap.append(d.xpos[anchor].copy());aq.append(d.xquat[anchor].copy())
            vv=[];ww=[]
            for bid in bids:
                mujoco.mj_jacBody(m,d,jacp,jacr,bid);vv.append(jacp@qvel[t]);ww.append(jacr@qvel[t])
            vel.append(vv);ang.append(ww)
        pos,quat,vel,ang,ap,aq=[torch.tensor(np.asarray(x),dtype=torch.float32)for x in (pos,quat,vel,ang,ap,aq)]
        bp,bq=mu.subtract_frame_transforms(ap[:,None].expand_as(pos),aq[:,None].expand_as(quat),pos,quat)
        features=torch.cat((bp.flatten(1),mu.matrix_from_quat(bq)[...,:,:2].flatten(1),mu.quat_apply_inverse(quat,vel).flatten(1),mu.quat_apply_inverse(quat,ang).flatten(1)),-1)
        off=torch.zeros(n,3);off[:,2]=.43;gravity=torch.zeros_like(off);gravity[:,2]=-1
        head=ap[:,2]+mu.quat_apply(aq,off)[:,2]
        geometry=torch.cat((head[:,None],mu.quat_apply_inverse(aq,gravity)),-1)
        keep=recovery_segments(geometry,50)
        if not torch.isfinite(features).all():raise ValueError('Invalid features '+name)
        record['retained_transitions']=int(keep.sum());audit.append(record)
        aa.append(features[:-1][keep]);bb.append(features[1:][keep]);gg.append(geometry[:-1][keep]);clipids.extend([ci]*int(keep.sum()))
    A,B,G=map(torch.cat,(aa,bb,gg));phase,direction=phase_direction(G)
    out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(out,a=A.numpy(),b=B.numpy(),geometry=G.numpy(),clip_ids=np.array(clipids),body_names=np.array(body_names),joint_names=np.array(names),fps=np.array([50]))
    report=dict(format='amp_recovery_features_v1',train_bank_sha256=hashlib.sha256(Path(a.train_bank).read_bytes()).hexdigest(),validation_bank_sha256=hashlib.sha256(Path(a.validation_bank).read_bytes()).hexdigest(),validation_source_overlap=0,transitions=len(A),phase_direction_counts=torch.bincount(phase*5+direction,minlength=20).reshape(4,5).tolist(),sha256=hashlib.sha256(out.read_bytes()).hexdigest(),clips=audit,note='CSV xyzw converted to wxyz; current AMP model FK; world body-origin velocities from Jacobians and differentiated qpos; 50Hz; phases by head height .6/.9/1.15; direction by dominant torso gravity xy, 4=upright. No collision or dynamic feasibility claim.')
    out.with_suffix('.json').write_text(json.dumps(report,indent=2));print(json.dumps({k:v for k,v in report.items()if k!='clips'},indent=2))
if __name__=='__main__':main()
