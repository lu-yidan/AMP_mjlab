"""Matched native-policy command tracking diagnostic, no deployment smoothing."""
import argparse,json
from pathlib import Path
from dataclasses import asdict
import numpy as np
import torch
import src.tasks
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.managers.event_manager import EventTermCfg
from mjlab.tasks.registry import load_env_cfg,load_rl_cfg,load_runner_cls
from mjlab.utils.lab_api.math import quat_apply,quat_apply_inverse,yaw_quat
from rsl_rl.runners.amp_on_policy_runner import _unpack_obs,_unpack_step
p=argparse.ArgumentParser();p.add_argument('--task',required=True);p.add_argument('--checkpoint',required=True);p.add_argument('--name',required=True);p.add_argument('--output',required=True);a=p.parse_args()
cases={'stand':[0,0,0],'forward05':[.5,0,0],'forward1':[1,0,0],'forward2':[2,0,0],'backward05':[-.5,0,0],'backward1':[-1,0,0],'left05':[0,.5,0],'right05':[0,-.5,0],'yaw_left':[0,0,1],'yaw_right':[0,0,-1],'forward_turn':[1,0,1]}
n=len(cases)*8
rng=np.random.default_rng(12345);jitter=rng.normal(0,.01,(8,29)).astype(np.float32)
def reset(env,env_ids):
 r=env.scene['robot'];ids=env_ids if env_ids is not None else torch.arange(n,device=env.device)
 root=r.data.default_root_state[ids].clone();root[:,:3]+=env.scene.env_origins[ids]
 joint=r.data.default_joint_pos[ids].clone()+torch.tensor(np.tile(jitter,(len(cases),1)),device=env.device)[ids]
 r.write_root_link_pose_to_sim(root[:,:7],env_ids=ids);r.write_root_link_velocity_to_sim(torch.zeros(len(ids),6,device=env.device),env_ids=ids);r.write_joint_state_to_sim(joint,torch.zeros_like(joint),env_ids=ids)
cfg=load_env_cfg('Unitree-G1-AMP-Flat',play=True);cfg.seed=12345;cfg.scene.num_envs=n;cfg.events={'reset':EventTermCfg(func=reset,mode='reset')};cfg.terminations={};cfg.curriculum={}
c=cfg.commands['twist'];c.heading_command=False;c.ranges.heading=None;c.rel_standing_envs=0.;c.rel_heading_envs=0.;c.resampling_time_range=(1000.,1000.)
c.ranges.lin_vel_x=(0.,0.);c.ranges.lin_vel_y=(0.,0.);c.ranges.ang_vel_z=(0.,0.)
agent=load_rl_cfg(a.task);env=ManagerBasedRlEnv(cfg=cfg,device='cuda:0');w=RslRlVecEnvWrapper(env,clip_actions=agent.clip_actions)
runner=load_runner_cls(a.task)(w,asdict(agent),log_dir=None,device='cuda:0');runner.load(a.checkpoint,load_optimizer=False);policy=runner.get_inference_policy(device='cuda:0');obs,_=_unpack_obs(w.get_observations())
r=env.scene['robot'];body=list(r.body_names).index('torso_link');target=torch.tensor(list(cases.values()),device=env.device).repeat_interleave(8,dim=0)
actual=[];upright=[];guidance=[];route=[]
with torch.inference_mode():
 for step in range(600):
  env.command_manager.get_command('twist')[:]=0 if step<100 else target
  obs,_,done,_=_unpack_step(*w.step(policy(obs)));assert not done.any()
  v=quat_apply_inverse(yaw_quat(r.data.body_link_quat_w[:,body]),r.data.body_link_lin_vel_w[:,body]);wz=r.data.body_link_ang_vel_w[:,body,2]
  actual.append(torch.cat((v[:,:2],wz[:,None]),-1).cpu().numpy())
  gz=r.data.projected_gravity_b[:,2];height=r.data.root_link_pos_w[:,2]-env.scene.env_origins[:,2];upright.append(((height>.65)&(gz<-.9)).cpu().numpy())
  offset=quat_apply(r.data.body_link_quat_w[:,body],torch.tensor([0.,0.,.43],device=env.device).expand(n,-1));head=r.data.body_link_pos_w[:,body,2]+offset[:,2]-env.scene.env_origins[:,2]
  guidance.append(((1.15-head)/.25).clamp(0,1).cpu().numpy());route.append((((gz+1).abs()>.6)|(head<.9)).cpu().numpy())
x=np.asarray(actual);up=np.asarray(upright);g=np.asarray(guidance);rt=np.asarray(route);result={'name':a.name,'checkpoint':a.checkpoint,'protocol':'8 matched initial joint perturbations per command, 2s standing then 10s command; no resets/push/DR/smoothing; torso yaw-frame xy velocity and world torso yaw rate; final5s means','cases':{}}
for i,(name,cmd) in enumerate(cases.items()):
 sl=slice(i*8,(i+1)*8);v=x[350:,sl];result['cases'][name]={'command':cmd,'mean_actual':v.mean((0,1)).tolist(),'mae':np.abs(v-np.array(cmd)).mean((0,1)).tolist(),'upright_percent':float(up[350:,sl].mean()*100),'guidance_gate_mean':float(g[350:,sl].mean()),'recovery_route_percent':float(rt[350:,sl].mean()*100)}
out=Path(a.output);out.mkdir(exist_ok=True,parents=True);(out/(a.name+'.json')).write_text(json.dumps(result,indent=2));np.savez_compressed(out/(a.name+'_traces.npz'),actual=x,upright=up,guidance_gate=g,route=rt);print(json.dumps(result,indent=2));env.close()
