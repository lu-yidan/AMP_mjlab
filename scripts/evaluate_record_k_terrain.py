"""K nominal terrain traversal: fixed five tiles, no resets, deterministic actor."""
import argparse,json,hashlib
from pathlib import Path
import numpy as np
import torch
import imageio.v2 as imageio
from PIL import Image,ImageDraw
import src.tasks
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.utils.lab_api.math import quat_apply_inverse,yaw_quat
from src.amp_supermimic.inference import load_policy
from src.tasks.amp_loco.mdp.terrain_tracking import ground_height
p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--output',required=True);a=p.parse_args()
out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
labels=['flat','slope','slope_inverted','stairs_inverted','stairs'];n=40

def reset(env,env_ids):
 r=env.scene['robot'];ids=env_ids if env_ids is not None else torch.arange(n,device=env.device)
 # Eight trials per type; first row, midpoint of each 20% column block.
 origins=env.scene.terrain.terrain_origins[0,torch.tensor([0,2,4,6,8],device=env.device)].repeat_interleave(8,0)
 env.scene.env_origins[ids]=origins[ids]
 root=r.data.default_root_state[ids].clone();root[:,:3]+=origins[ids]
 joints=r.data.default_joint_pos[ids].clone();joints+=torch.tensor(np.random.default_rng(12345).normal(0,.01,(n,29)),device=env.device,dtype=joints.dtype)[ids]
 r.write_root_link_pose_to_sim(root[:,:7],env_ids=ids);r.write_root_link_velocity_to_sim(torch.zeros(len(ids),6,device=env.device),env_ids=ids);r.write_joint_state_to_sim(joints,torch.zeros_like(joints),env_ids=ids)
cfg=load_env_cfg('Unitree-G1-AMP-I-Shared-TerrainCritic',play=True);cfg.seed=12345;cfg.scene.num_envs=n
flat=load_env_cfg('Unitree-G1-AMP-Flat',play=True)
cfg.commands=flat.commands;cfg.events={'reset':EventTermCfg(func=reset,mode='reset')};cfg.terminations={};cfg.curriculum={}
c=cfg.commands['twist'];c.heading_command=False;c.ranges.heading=None;c.rel_standing_envs=0;c.rel_heading_envs=0;c.resampling_time_range=(1000.,1000.);c.ranges.lin_vel_x=(0.,0.);c.ranges.lin_vel_y=(0.,0.);c.ranges.ang_vel_z=(0.,0.)
cfg.viewer.width=640;cfg.viewer.height=384;cfg.viewer.max_extra_envs=0;cfg.viewer.distance=4.;cfg.viewer.elevation=-25;cfg.viewer.azimuth=135
with torch.inference_mode():
 env=ManagerBasedRlEnv(cfg,device='cuda:0',render_mode='rgb_array');obs,_=env.reset();policy=load_policy(a.checkpoint,obs['actor'])
 r=env.scene['robot'];body=list(r.body_names).index('torso_link')
 initial=torch.cat((r.data.root_link_pos_w,r.data.root_link_quat_w,r.data.joint_pos),-1).cpu().numpy()
 writers={name:imageio.get_writer(out/(name+'.mp4'),fps=25,codec='libx264',quality=7,macro_block_size=1,ffmpeg_params=['-threads','1','-movflags','+faststart']) for name in labels}
 actual=[];up=[];positions=[];heights=[];miss=[]
 try:
  for step in range(600):
   target=0. if step<100 else 1.
   cmd=env.command_manager.get_command('twist');cmd.zero_();cmd[:,0]=target
   v=quat_apply_inverse(yaw_quat(r.data.body_link_quat_w[:,body]),r.data.body_link_lin_vel_w[:,body]);wz=r.data.body_link_ang_vel_w[:,body,2]
   h=r.data.root_link_pos_w[:,2]-ground_height(env);stable=(h>.65)&(r.data.projected_gravity_b[:,2]<-.9)
   actual.append(torch.cat((v[:,:2],wz[:,None]),-1).cpu().numpy());up.append(stable.cpu().numpy());positions.append(r.data.root_link_pos_w.cpu().numpy());heights.append(h.cpu().numpy());miss.append((env.scene['terrain_scan'].data.distances[:,93]<0).cpu().numpy())
   if step%2==0:
    for i,name in enumerate(labels):
     idx=i*8;env.cfg.viewer.env_idx=idx
     env._offline_renderer._cam.lookat[:]=r.data.root_link_pos_w[idx].cpu().numpy()
     canvas=Image.new('RGB',(640,448),(20,24,30));canvas.paste(Image.fromarray(env.render()),(0,64));d=ImageDraw.Draw(canvas)
     d.text((8,8),f'K-61k | {name} | {step*.02:.2f}s | no resets',fill='white')
     d.text((8,32),f'cmd vx={target:.1f} actual={v[idx,0]:.2f} root/ground={h[idx]:.2f}m',fill='white')
     writers[name].append_data(np.asarray(canvas))
     if step in (0,300,598):canvas.save(out/(name+f'_{step}.jpg'))
   obs,_,term,trunc,_=env.step(policy(obs['actor']));assert not (term|trunc).any()
 finally:
  for w in writers.values():w.close()
  env.close()
x=np.asarray(actual);u=np.asarray(up);pos=np.asarray(positions)
result=dict(checkpoint=a.checkpoint,checkpoint_sha256=hashlib.sha256(Path(a.checkpoint).read_bytes()).hexdigest(),initial_state_hash=hashlib.sha256(initial.tobytes()).hexdigest(),protocol='Nominal generated terrain seed12345, 8 standing starts per type, 2s stand then 10s vx=1 m/s, no resets/push/DR; final5s torso yaw-frame errors; first trial of every type recorded, no selection by outcome.',cases={})
for i,name in enumerate(labels):
 sl=slice(i*8,(i+1)*8);result['cases'][name]=dict(mae=np.abs(x[350:,sl]-[1,0,0]).mean((0,1)).tolist(),upright_percent=float(u[350:,sl].mean()*100),displacement_x_m=float((pos[-1,sl,0]-pos[0,sl,0]).mean()),ground_ray_miss_percent=float(np.asarray(miss)[:,sl].mean()*100))
(out/'terrain.json').write_text(json.dumps(result,indent=2));np.savez_compressed(out/'traces.npz',actual=x,upright=u,positions=pos,height=np.asarray(heights),miss=np.asarray(miss));print(json.dumps(result),flush=True)
