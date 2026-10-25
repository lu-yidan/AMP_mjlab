"""Nominal mjlab adapter. Capture physical next state BEFORE manual reset."""
import torch
from .mdp import flatten_state, unflatten_state, observation, reward, termination


class VelocitySim:
    def __init__(self, num_envs=4096, device='cuda:0', seed=42, max_speed=3., render=False):
        from mjlab.envs import ManagerBasedRlEnv
        from mjlab.managers.event_manager import EventTermCfg
        from src.tasks.amp_loco.config.g1.env_cfgs import g1_amp_flat_env_cfg
        cfg = g1_amp_flat_env_cfg(play=True)
        cfg.seed = seed
        cfg.scene.num_envs = num_envs
        # No auto-reset: terminal physical states must reach the world model.
        cfg.terminations = {}; cfg.curriculum = {}; cfg.rewards = {}; cfg.metrics = {}
        cfg.observations = {}; cfg.commands = {}; cfg.events = {}
        cfg.scene.sensors = ()
        cfg.events['nominal_reset'] = EventTermCfg(func=self._reset_pose, mode='reset')
        self.env = ManagerBasedRlEnv(cfg=cfg, device=device, render_mode='rgb_array' if render else None)
        self.device=device; self.n=num_envs; self.max_speed=max_speed
        self.robot=self.env.scene['robot']
        self.default_q=self.robot.data.default_joint_pos[0].clone()
        self.command=torch.zeros(num_envs,3,device=device)
        self.a_prev=torch.zeros(num_envs,29,device=device)
        self.ep_step=torch.zeros(num_envs,dtype=torch.int32,device=device)
        self.age=torch.zeros(num_envs,dtype=torch.long,device=device)
        self.xi=torch.empty(num_envs,0,device=device)
        self.env.reset()
        self.resample(torch.arange(num_envs,device=device))

    @staticmethod
    def _reset_pose(env, env_ids):
        r=env.scene['robot']
        ids=torch.arange(env.num_envs,device=env.device) if env_ids is None else env_ids
        root=r.data.default_root_state[ids].clone(); root[:,:3]+=env.scene.env_origins[ids]
        r.write_root_link_pose_to_sim(root[:,:7],env_ids=ids)
        r.write_root_link_velocity_to_sim(torch.zeros_like(root[:,7:13]),env_ids=ids)
        q=r.data.default_joint_pos[ids].clone()
        r.write_joint_state_to_sim(q,torch.zeros_like(q),env_ids=ids)

    def state(self):
        from .rotations import quat_to_mat
        d=self.robot.data
        return (d.root_link_pos_w-self.env.scene.env_origins, quat_to_mat(d.root_link_quat_w),
                d.root_link_lin_vel_w,d.root_link_ang_vel_w,d.joint_pos,d.joint_vel)

    def resample(self, ids):
        # Explicit stand / walk / run goals. A single policy covers all modes.
        mode=torch.randint(0,3,(len(ids),),device=self.device)
        speed=torch.rand(len(ids),device=self.device)
        walk=.1+speed*(min(1.,self.max_speed)-.1)
        run=min(1.,self.max_speed)+speed*max(0.,self.max_speed-1.)
        self.command[ids]=0
        self.command[ids,0]=torch.where(mode==0,0.,torch.where(mode==1,walk,run))
        self.age[ids]=torch.randint(150,401,(len(ids),),device=self.device)

    @torch.no_grad()
    def step(self, action, auto_reset=True, resample=True):
        action=action.to(self.device).clamp(-5,5)
        s=flatten_state(self.state()).clone(); prev=self.a_prev.clone(); command=self.command.clone()
        ep=self.ep_step.clone()
        self.env.step(action)
        s1=flatten_state(self.state()).clone()
        S1=unflatten_state(s1)
        terminated=termination(S1)
        if not torch.isfinite(s1).all():
            raise FloatingPointError('Non-finite simulator state; refusing to contaminate replay')
        self.ep_step+=1; self.age-=1
        done=terminated|(self.ep_step>=1000)
        transition=dict(s=s,s1=s1,a=action.clone(),a_prev=prev,xi=self.xi.clone(),
            r=reward(S1,command,action,prev,self.default_q,self.env.step_dt),
            command=command,ep_step=ep,done=done)
        self.a_prev.copy_(action)
        ids=done.nonzero().flatten()
        if auto_reset and len(ids):
            self.env.reset(env_ids=ids)
            self.ep_step[ids]=0; self.a_prev[ids]=0
            if resample:self.resample(ids)
        if resample:
            self.resample((self.age<=0).nonzero().flatten())
        return transition

    def close(self):
        self.env.close()
