"""Validate L4 training poses against the current AMP robot collision model."""
import argparse, hashlib, json
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
import mujoco
from mjlab.scene import Scene
from src.tasks.amp_loco.config.g1.env_cfgs import g1_amp_flat_env_cfg
p=argparse.ArgumentParser();p.add_argument('--banks',required=True);p.add_argument('--source-xml',required=True);p.add_argument('--output',required=True);p.add_argument('--split',default='train',choices=['train','validation','test']);a=p.parse_args()
names=[x.attrib['joint'] for x in ET.parse(a.source_xml).getroot().find('actuator')]
cfg=g1_amp_flat_env_cfg();cfg.scene.num_envs=1
m=Scene(cfg.scene,device='cpu').compile();d=mujoco.MjData(m)
adr=np.array([int(m.joint('robot/'+n).qposadr[0]) for n in names]);limits=np.array([m.joint('robot/'+n).range for n in names])
geoms=np.array([i for i in range(m.ngeom) if (mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_GEOM,i) or "").startswith("robot/") and (m.geom_contype[i] or m.geom_conaffinity[i])])
rows=[];stages=[];labels=[];sources=[];clips=[];report={}
for source,folder in [('natural','natural_curriculum_v1'),('procedural','procedural_low_v1')]:
 path=Path(a.banks)/folder/(a.split+'.npz');bank=np.load(path);reject=0
 for i,q in enumerate(bank['qpos']):
  if not np.isfinite(q).all() or ((q[7:]<limits[:,0])|(q[7:]>limits[:,1])).any():reject+=1;continue
  d.qpos[:]=m.qpos0;d.qpos[:7]=q[:7];d.qpos[:2]=0;d.qpos[3:7]/=np.linalg.norm(d.qpos[3:7]);d.qpos[adr]=q[7:];d.qvel[:]=0
  mujoco.mj_forward(m,d)
  bottoms=[]
  for g in geoms:
   typ=m.geom_type[g];sz=m.geom_size[g];rot=d.geom_xmat[g].reshape(3,3)
   if typ==mujoco.mjtGeom.mjGEOM_SPHERE: ext=sz[0]
   elif typ==mujoco.mjtGeom.mjGEOM_CAPSULE:ext=sz[0]+sz[1]*abs(rot[2,2])
   elif typ==mujoco.mjtGeom.mjGEOM_BOX:ext=np.abs(rot[2])@sz
   elif typ==mujoco.mjtGeom.mjGEOM_CYLINDER:ext=sz[0]*np.sqrt(max(0,1-rot[2,2]**2))+sz[1]*abs(rot[2,2])
   else:raise ValueError(f'Unsupported collision geometry {typ}')
   bottoms.append(d.geom_xpos[g,2]-ext)
  d.qpos[2]+=.003-min(bottoms);mujoco.mj_forward(m,d)
  if d.qpos[2]<.06 or any(c.dist<-.002 for c in d.contact):reject+=1;continue
  rows.append(np.r_[d.qpos[:7],d.qpos[adr]]);stages.append(bank['stages'][i]);labels.append(bank['labels'][i]);sources.append(source);clips.append(str(bank['clips'][i]) if 'clips' in bank.files else str(i))
 report[source]={'input':len(bank['qpos']),'rejected':reject,'source_sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True)
np.savez_compressed(out,qpos=np.array(rows,dtype=np.float32),joint_names=np.array(names),stages=np.array(stages),labels=np.array(labels),sources=np.array(sources),clips=np.array(clips))
report['groups']={f'{s}/{t}/{l}':sum(x==s and y==t and z==l for x,y,z in zip(sources,stages,labels)) for s,t,l in sorted(set(zip(sources,stages,labels)))}
report['sha256']=hashlib.sha256(out.read_bytes()).hexdigest();out.with_suffix('.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
