"""Nominal mjlab adapter. Capture physical next state BEFORE manual reset."""
import torch
from .mdp import flatten_state, unflatten_state, reward, termination, is_recovery
from .reset_bank import ResetBank, DEFAULT_BANK


class VelocitySim:
    def __init__(self, num_envs=4096, device='cuda:0', seed=42, max_speed=5., render=False, reset_mode='stratified',
                 bank_path=DEFAULT_BANK, episode_seconds=20., gamma=.99, locomotion_grace=.3, recovery_grace=5.):
        from mjlab.envs import ManagerBasedRlEnv
        from mjlab.managers.event_manager import EventTermCfg
        from src.tasks.amp_loco.config.g1.env_cfgs import g1_amp_flat_env_cfg
        if reset_mode not in ('stratified','standing','grouped'):raise ValueError(reset_mode)
        if episode_seconds<=0:raise ValueError('episode_seconds must be positive')
        self.reset_mode=reset_mode;self.bank_path=bank_path;self.bank=None;self.gamma=gamma
        self.locomotion_grace=locomotion_grace;self.recovery_grace=recovery_grace
        self.recovery_env=torch.zeros(num_envs,dtype=torch.bool,device=device)
        self.recovery_env[torch.randperm(num_envs,device=device)[:round(.4*num_envs)]]=True
        self.bad_steps=torch.zeros(num_envs,dtype=torch.long,device=device)
        self.attempt_steps=torch.zeros_like(self.bad_steps)
        self.attempt_active=self.recovery_env.clone()
        self.reset_groups=torch.zeros(num_envs,dtype=torch.long,device=device)
        cfg = g1_amp_flat_env_cfg(play=True)
        cfg.seed = seed
        cfg.scene.num_envs = num_envs
        # No auto-reset: terminal physical states must reach the world model.
        cfg.terminations = {}; cfg.curriculum = {}; cfg.rewards = {}; cfg.metrics = {}
        cfg.observations = {}; cfg.commands = {}; cfg.events = {}
        cfg.scene.sensors = ()
        # EventManager deep-copies term configs. A bound method would copy this
        # adapter and detach reset bookkeeping/configuration from the live object.
        def reset_pose(env, env_ids):
            self._reset_pose(env, env_ids)
        cfg.events['nominal_reset'] = EventTermCfg(func=reset_pose, mode='reset')
        self.env = ManagerBasedRlEnv(cfg=cfg, device=device, render_mode='rgb_array' if render else None)
        self.device=device; self.n=num_envs; self.max_speed=max_speed
        self.robot=self.env.scene['robot']
        self.max_episode_steps=round(episode_seconds/self.env.step_dt)
        self.recovered=torch.zeros(num_envs,dtype=torch.bool,device=device)
        self.stable_steps=torch.zeros(num_envs,dtype=torch.long,device=device)
        self.default_q=self.robot.data.default_joint_pos[0].clone()
        self.command=torch.zeros(num_envs,3,device=device)
        self.a_prev=torch.zeros(num_envs,29,device=device)
        self.ep_step=torch.zeros(num_envs,dtype=torch.int32,device=device)
        self.age=torch.zeros(num_envs,dtype=torch.long,device=device)
        self.xi=torch.empty(num_envs,0,device=device)
        self.env.reset()
        self.resample(torch.arange(num_envs,device=device))

    def _reset_pose(self, env, env_ids):
        r=env.scene['robot']
        ids=torch.arange(env.num_envs,device=env.device) if env_ids is None else env_ids
        root=r.data.default_root_state[ids].clone(); root[:,:3]+=env.scene.env_origins[ids]
        q=r.data.default_joint_pos[ids].clone()
        self.reset_groups[ids]=0
        if self.reset_mode in ('stratified','grouped'):
            if self.bank is None:self.bank=ResetBank(self.bank_path,r.joint_names,env.device)
            groups,indices=self.bank.sample(len(ids),recovery_only=self.reset_mode=='grouped')
            if self.reset_mode=='grouped':
                groups=torch.where(self.recovery_env[ids],groups,0)
                indices=torch.where(self.recovery_env[ids],indices,-1)
            self.reset_groups[ids]=groups
            low=indices>=0;sample=self.bank.qpos[indices[low]]
            root[low,:7]=sample[:,:7];root[low,:3]+=env.scene.env_origins[ids[low]]
            q[low]=sample[:,7:]
        self.bad_steps[ids]=0;self.attempt_steps[ids]=0
        self.attempt_active[ids]=self.recovery_env[ids] if self.reset_mode=='grouped' else False
        r.write_root_link_pose_to_sim(root[:,:7],env_ids=ids)
        r.write_root_link_velocity_to_sim(torch.zeros_like(root[:,7:13]),env_ids=ids)
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
        done=terminated|(self.ep_step>=self.max_episode_steps)
        recovering=is_recovery(S1)
        self.stable_steps=torch.where(~recovering,self.stable_steps+1,0)
        stable=self.stable_steps>=round(1/self.env.step_dt)
        self.recovered|=stable
        budget_reset=torch.zeros_like(done)
        if self.reset_mode=='grouped':
            # Fall threshold differs from the stricter successful-recovery criterion.
            fallen=(S1[0][:,2]<.5)|(S1[1][:,2,2]<.3420201433)
            self.bad_steps=torch.where(fallen,self.bad_steps+1,0)
            self.attempt_active=(self.attempt_active|fallen)&~stable
            self.attempt_steps=torch.where(self.attempt_active,self.attempt_steps+1,0)
            budget_reset=torch.where(self.recovery_env,
                self.attempt_steps>=round(self.recovery_grace/self.env.step_dt),
                self.bad_steps>=round(self.locomotion_grace/self.env.step_dt))
            done|=budget_reset
        transition=dict(s=s,s1=s1,a=action.clone(),a_prev=prev,xi=self.xi.clone(),
            r=reward(S1,command,action,prev,self.default_q,self.env.step_dt,previous_state=unflatten_state(s),gamma=self.gamma),
            command=command,ep_step=ep,done=done,recovery=is_recovery(unflatten_state(s)),
            terminated=terminated, budget_reset=budget_reset, recovery_env=self.recovery_env.clone(), episode_seconds=self.ep_step.float()*self.env.step_dt, reset_group=self.reset_groups.clone(), recovered=self.recovered.clone())
        self.a_prev.copy_(action)
        ids=done.nonzero().flatten()
        if auto_reset and len(ids):
            self.env.reset(env_ids=ids)
            self.ep_step[ids]=0; self.a_prev[ids]=0
            self.stable_steps[ids]=0;self.recovered[ids]=False
            if resample:self.resample(ids)
        if resample:
            self.resample((self.age<=0).nonzero().flatten())
        return transition

    @torch.no_grad()
    def reset_to_bank(self, bank, indices):
        """Reset every environment to explicitly selected held-out bank rows."""
        if len(indices)!=self.n:raise ValueError('One bank row per environment required')
        mode=self.reset_mode;self.reset_mode='standing'
        try:self.env.reset()
        finally:self.reset_mode=mode
        ids=torch.arange(self.n,device=self.device)
        q=bank.qpos[indices].clone();q[:,:3]+=self.env.scene.env_origins
        self.robot.write_root_link_pose_to_sim(q[:,:7],env_ids=ids)
        self.robot.write_root_link_velocity_to_sim(torch.zeros(self.n,6,device=self.device),env_ids=ids)
        self.robot.write_joint_state_to_sim(q[:,7:],torch.zeros_like(q[:,7:]),env_ids=ids)
        self.env.scene.write_data_to_sim();self.env.sim.forward();self.env.sim.sense()
        self.ep_step.zero_();self.a_prev.zero_();self.stable_steps.zero_();self.recovered.zero_()

    def close(self):
        self.env.close()
