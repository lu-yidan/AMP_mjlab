"""Matched 60k evaluation: held-out recovery poses and nominal stand-to-walk trials."""
import argparse,json,hashlib
from pathlib import Path
from dataclasses import asdict
import numpy as np
import torch
import src.tasks
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.managers.event_manager import EventTermCfg
from mjlab.tasks.registry import load_env_cfg,load_rl_cfg,load_runner_cls
from mjlab.utils.lab_api.math import quat_apply
from rsl_rl.runners.amp_on_policy_runner import _unpack_obs,_unpack_step

p=argparse.ArgumentParser();p.add_argument('--task',required=True);p.add_argument('--checkpoint',required=True);p.add_argument('--name',required=True);p.add_argument('--bank',required=True);p.add_argument('--output',required=True);a=p.parse_args()
bank=np.load(a.bank);nr=len(bank['qpos']);n=nr+128

def reset_eval(env,env_ids):
 r=env.scene['robot'];ids=env_ids if env_ids is not None else torch.arange(n,device=env.device)
 root=r.data.default_root_state[ids].clone();root[:,:3]+=env.scene.env_origins[ids]
 joints=r.data.default_joint_pos[ids].clone();low=ids<nr
 q=torch.tensor(bank['qpos'],device=env.device)
 mapping=[bank['joint_names'].tolist().index(k) for k in r.joint_names]
 root[low,:7]=q[ids[low],:7];root[low,:3]+=env.scene.env_origins[ids[low]]
 joints[low]=q[ids[low],7:][:,mapping]
 r.write_root_link_pose_to_sim(root[:,:7],env_ids=ids)
 r.write_root_link_velocity_to_sim(torch.zeros(len(ids),6,device=env.device),env_ids=ids)
 r.write_joint_state_to_sim(joints,torch.zeros_like(joints),env_ids=ids)

cfg=load_env_cfg('Unitree-G1-AMP-Flat',play=True);cfg.seed=12345;cfg.scene.num_envs=n
cfg.terminations={};cfg.curriculum={};cfg.events={'matched_reset':EventTermCfg(func=reset_eval,mode='reset')}
cfg.episode_length_s=30
cmdcfg=cfg.commands['twist'];cmdcfg.heading_command=False;cmdcfg.ranges.heading=None;cmdcfg.rel_standing_envs=0.;cmdcfg.rel_heading_envs=0.
cmdcfg.ranges.lin_vel_x=(0.,0.);cmdcfg.ranges.lin_vel_y=(0.,0.);cmdcfg.ranges.ang_vel_z=(0.,0.)
agent=load_rl_cfg(a.task)
env=ManagerBasedRlEnv(cfg=cfg,device='cuda:0');w=RslRlVecEnvWrapper(env,clip_actions=agent.clip_actions)
runner=load_runner_cls(a.task)(w,asdict(agent),log_dir=None,device='cuda:0');runner.load(a.checkpoint,load_optimizer=False)
policy=runner.get_inference_policy(device='cuda:0');obs,_=_unpack_obs(w.get_observations());r=env.scene['robot'];body=list(r.body_names).index('torso_link')
init_hash=hashlib.sha256(torch.cat((r.data.root_link_pos_w-env.scene.env_origins,r.data.root_link_quat_w,r.data.joint_pos),-1).cpu().numpy().tobytes()).hexdigest()
hold=torch.zeros(n,device=env.device);success=torch.zeros(n,dtype=torch.bool,device=env.device);first=torch.full((n,),float('nan'),device=env.device)
bad_count=torch.zeros(n,device=env.device);ever_timeout=torch.zeros_like(success);switch=torch.zeros(n,device=env.device);previous=None;rec_steps=torch.zeros(n,device=env.device)
errors=[];stable_tail=torch.zeros(n,device=env.device);height_trace=[];gravity_trace=[]
with torch.inference_mode():
 for step in range(1000):
  command=env.command_manager.get_command('twist');command.zero_();target=[0.,.5,1.,2.][step//250];command[nr:,0]=target
  obs,_,done,_=_unpack_step(*w.step(policy(obs)))
  if done.any():raise RuntimeError('Evaluation unexpectedly reset')
  h=r.data.root_link_pos_w[:,2]-env.scene.env_origins[:,2];gz=r.data.projected_gravity_b[:,2]
  stable=(h>.65)&(gz<-.9);hold=torch.where(stable,hold+1,0)
  new=(hold>=50)&~success;first[new]=(step+1)*env.step_dt;success|=new
  if step>=750:stable_tail+=stable
  bad=(h<.5)|(gz>-.3420201433);bad_count=torch.where(bad,bad_count+1,0);ever_timeout|=bad_count>=250
  offset=quat_apply(r.data.body_link_quat_w[:,body],torch.tensor([0.,0.,.43],device=env.device).expand(n,-1))
  head=r.data.body_link_pos_w[:,body,2]+offset[:,2]-env.scene.env_origins[:,2]
  route=((gz+1).abs()>.6)|(head<.9);rec_steps+=route
  if previous is not None:switch+=(route!=previous)
  previous=route
  errors.append((r.data.root_link_lin_vel_w[nr:,0]-target).abs().cpu().numpy())
  if step%10==0:height_trace.append(h.cpu().numpy());gravity_trace.append(gz.cpu().numpy())

def report(mask):
 ids=torch.tensor(np.flatnonzero(mask),device=env.device)
 return {'n':len(ids),'success_1s_percent':float(success[ids].float().mean()*100),'mean_success_time_s':float(first[ids][success[ids]].mean()) if success[ids].any() else None,'last5s_upright_percent':float(stable_tail[ids].mean()/250*100),'ever_continuous_failure_5s_percent':float(ever_timeout[ids].float().mean()*100),'route_switches_mean':float(switch[ids].mean()),'recovery_route_percent':float(rec_steps[ids].mean()/1000*100)}
result={'name':a.name,'checkpoint':a.checkpoint,'initial_state_hash':init_hash,'bank_sha256':hashlib.sha256(Path(a.bank).read_bytes()).hexdigest(),'protocol':'20s, no resets, deterministic policy, nominal dynamics; hroot>.65 and gz<-.9 for 1s; zero command for recovery; 128 standing trials commands 0/.5/1/2m/s every 5s','groups':{}}
for source in ['natural','procedural']:
 for stage in ['low','middle','late']:
  mask=(bank['sources']==source)&(bank['stages']==stage)
  if mask.any(): result['groups'][source+'/'+stage]=report(np.r_[mask,np.zeros(128,dtype=bool)])
  if stage=='low':
   for direction in ['supine','prone','left_side_down','right_side_down']:
    sub=mask&(bank['labels']==direction)
    if sub.any():result['groups'][source+'/'+direction]=report(np.r_[sub,np.zeros(128,dtype=bool)])
result['locomotion']=report(np.r_[np.zeros(nr,dtype=bool),np.ones(128,dtype=bool)])
result['locomotion']['vx_mae_by_5s']=np.array(errors).reshape(4,250,128).mean((1,2)).tolist()
out=Path(a.output);out.mkdir(parents=True,exist_ok=True);(out/(a.name+'.json')).write_text(json.dumps(result,indent=2));np.savez_compressed(out/(a.name+'_traces.npz'),height=np.array(height_trace),gravity_z=np.array(gravity_trace));print(json.dumps(result,indent=2));env.close()
