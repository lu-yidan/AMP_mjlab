"""Six fixed nominal scenarios, identical across AMP-A/B/C/D/E; no resets."""
import argparse, hashlib, json
from dataclasses import asdict
from pathlib import Path
import numpy as np
import torch
import imageio.v2 as imageio
from PIL import Image, ImageDraw, ImageFont
import src.tasks
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.managers.event_manager import EventTermCfg
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.utils.lab_api.math import quat_apply_inverse, yaw_quat
from rsl_rl.runners.amp_on_policy_runner import _unpack_obs, _unpack_step

p=argparse.ArgumentParser()
p.add_argument('--backend',choices=['amp','supermimic'],default='amp')
for key in ('task','checkpoint','name','bank','output'):p.add_argument('--'+key,required=True)
a=p.parse_args();bank=np.load(a.bank);out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
directions=['supine','prone','left_side_down','right_side_down']
rows=[int(np.flatnonzero((bank['sources']=='procedural')&(bank['stages']=='low')&(bank['labels']==d))[0])for d in directions]
labels=directions+['turn','walk_run'];n=6

def reset_eval(env,env_ids):
 r=env.scene['robot'];ids=env_ids if env_ids is not None else torch.arange(n,device=env.device)
 root=r.data.default_root_state[ids].clone();root[:,:3]+=env.scene.env_origins[ids];joints=r.data.default_joint_pos[ids].clone();low=ids<4
 q=torch.tensor(bank['qpos'][rows],device=env.device);mapping=[bank['joint_names'].tolist().index(k)for k in r.joint_names]
 root[low,:7]=q[ids[low],:7];root[low,:3]+=env.scene.env_origins[ids[low]];joints[low]=q[ids[low],7:][:,mapping]
 r.write_root_link_pose_to_sim(root[:,:7],env_ids=ids);r.write_root_link_velocity_to_sim(torch.zeros(len(ids),6,device=env.device),env_ids=ids)
 r.write_joint_state_to_sim(joints,torch.zeros_like(joints),env_ids=ids)

cfg=load_env_cfg('Unitree-G1-AMP-Flat',play=True);cfg.seed=12345;cfg.scene.num_envs=n
cfg.terminations={};cfg.curriculum={};cfg.events={'matched_reset':EventTermCfg(func=reset_eval,mode='reset')};cfg.episode_length_s=30
c=cfg.commands['twist'];c.heading_command=False;c.ranges.heading=None;c.rel_standing_envs=0.;c.rel_heading_envs=0.;c.resampling_time_range=(1000.,1000.)
c.ranges.lin_vel_x=(0.,0.);c.ranges.lin_vel_y=(0.,0.);c.ranges.ang_vel_z=(0.,0.)
cfg.viewer.width=640;cfg.viewer.height=384;cfg.viewer.max_extra_envs=0;cfg.viewer.distance=3.;cfg.viewer.elevation=-20;cfg.viewer.azimuth=135
agent=load_rl_cfg(a.task);agent.amp_num_preload_transitions=2048
# Expert preloading is only for constructing the checkpoint-compatible runner; no learning.
env=ManagerBasedRlEnv(cfg=cfg,device='cuda:0',render_mode='rgb_array');w=RslRlVecEnvWrapper(env,clip_actions=agent.clip_actions)
obs,obs_extras=_unpack_obs(w.get_observations())
if a.backend=='supermimic':
 import sys
 sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'third_party/mbrl_supermimic'))
 from supermimic.modules.actor_critic import ActorCritic
 checkpoint=torch.load(a.checkpoint,map_location='cuda:0',weights_only=False)
 model=ActorCritic({'policy':obs,'critic':obs_extras['observations']['critic']},29).to('cuda:0')
 model.load_state_dict(checkpoint['policy'],strict=True);model.eval()
 clip=checkpoint['cfg'].get('action_clip')
 def policy(observations):
  action=model.act_inference({'policy':observations})
  return action if clip is None else action.clamp(-clip,clip)
else:
 runner=load_runner_cls(a.task)(w,asdict(agent),log_dir=None,device='cuda:0');runner.load(a.checkpoint,load_optimizer=False)
 policy=runner.get_inference_policy(device='cuda:0')
r=env.scene['robot'];body=list(r.body_names).index('torso_link')
initial=torch.cat((r.data.root_link_pos_w-env.scene.env_origins,r.data.root_link_quat_w,r.data.joint_pos),-1).cpu().numpy()
report=dict(backend=a.backend,name=a.name,task=a.task,checkpoint=a.checkpoint,checkpoint_sha256=hashlib.sha256(Path(a.checkpoint).read_bytes()).hexdigest(),bank_sha256=hashlib.sha256(Path(a.bank).read_bytes()).hexdigest(),bank_rows=rows,initial_state_hash=hashlib.sha256(initial.tobytes()).hexdigest(),protocol='Nominal physics, deterministic policy, no resets/pushes; first procedural low pose per direction; recovery 10s at zero command; walk/run 0/.5/1/2/3 m/s and yaw 0/1/-1/1/-1 rad/s each 5s; Command changes enter observations after one 0.02s control step, matching the existing evaluation protocol.',fps=25)
font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',18)
writers={label:imageio.get_writer(out/(label+'.mp4'),fps=25,codec='libx264',quality=7,macro_block_size=1,ffmpeg_params=['-threads','1','-movflags','+faststart'])for label in labels}
traces=[]
try:
 with torch.inference_mode():
  for step in range(1250):
   t=step*env.step_dt;phase=min(4,step//250);command=env.command_manager.get_command('twist');command.zero_();command[5,0]=[0.,.5,1.,2.,3.][phase];command[4,2]=[0.,1.,-1.,1.,-1.][phase]
   if step%2==0:
    vx=quat_apply_inverse(yaw_quat(r.data.body_link_quat_w[:,body]),r.data.body_link_lin_vel_w[:,body])[:,0]
    wz=r.data.body_link_ang_vel_w[:,body,2];height=r.data.root_link_pos_w[:,2]-env.scene.env_origins[:,2]
    for i,label in enumerate(labels):
     if i<4 and step>=500:continue
     env.cfg.viewer.env_idx=i
     target=r.data.root_link_pos_w[i].cpu().numpy().copy();target[2]=float(env.scene.env_origins[i,2])+.6
     env._offline_renderer._cam.lookat[:]=target
     frame=env.render();canvas=Image.new('RGB',(640,464),(20,24,30));canvas.paste(Image.fromarray(frame),(0,80));draw=ImageDraw.Draw(canvas)
     draw.text((10,5),a.name+' | '+label+' | %.2fs'%t,font=font,fill='white')
     draw.text((10,30),'cmd vx=%.1f wz=%.1f | actual vx=%.2f wz=%.2f'%(command[i,0],command[i,2],vx[i],wz[i]),font=font,fill='white')
     draw.text((10,55),'root=%.2fm | no reset'%height[i],font=font,fill=(180,210,240))
     writers[label].append_data(np.asarray(canvas))
     if step in (0,150,400,1000):canvas.save(out/(label+'_%04d.jpg'%step))
    traces.append(dict(t=t,vx=vx.tolist(),wz=wz.tolist(),height=height.tolist()))
   obs,_,done,_=_unpack_step(*w.step(policy(obs)))
   if done.any():raise RuntimeError('Unexpected reset')
finally:
 for writer in writers.values():writer.close()
 env.close()
(out/'recording.json').write_text(json.dumps(report,indent=2));(out/'traces.json').write_text(json.dumps(traces))
print('VIDEO_COMPLETE '+a.name,flush=True)
