"""AMP-C MDP for MBRL SuperMimic. Physical rollouts keep the original manager env.

Imagination carries recovery hysteresis, delayed termination, command smoothing,
observation histories and the sampled world's DR context. Sensor-dependent costs
and substep joint acceleration are predicted as supervised world-model auxiliaries.
Only real transitions train the adversarial discriminators.
"""
import math
import torch
from supermimic.env.tracking_task import TrackingTask, StateLayout, CarryRollout
from supermimic.wrappers.mjlab import MjlabTrackingTask, _SiteSensor
from supermimic.utils.kinematics import TreeKinematics
from supermimic.utils.rotations import quat_to_mat
from rsl_rl.modules.stable_multi_amp import StableMultiAMP
from rsl_rl.utils.motion_loader import AMPLoader


def inv(R, v):
    return (R.transpose(-1, -2) @ v.unsqueeze(-1)).squeeze(-1)


class AMPTask(MjlabTrackingTask):
    # Override the motion-tracking-specific constructor and imagination, reuse
    # the library's verified reset capture, teleport detection and DR readers.
    aux_names = ('joint_acc_l2', 'foot_slip', 'self_collisions')
    aux_dim = 3
    history_length = 0  # exact histories stored in side state, no reconstruction
    compile_mdp = None
    motion_subset = False

    def __init__(self, env, agent, capture_verification=False):
        self.env = env
        self.entity_name = 'robot'
        self.entity = env.scene['robot']
        self.command = env.command_manager.get_term('twist')
        self.num_envs, self.device = env.num_envs, torch.device(env.device)
        self.num_actions, self.step_dt = 29, env.step_dt
        self.layout = StateLayout(29)
        self._hook_reset()
        self._set_context('auto', None, False)
        self.kin = TreeKinematics(env.sim.mj_model,
            ['robot/'+n for n in self.entity.joint_names],
            ['robot/'+n for n in self.entity.body_names], self.device)
        self.body = list(self.entity.body_names).index('torso_link')
        self.amp_ids = [list(self.entity.body_names).index(n) for n in agent.amp_body_names]
        self.gyro = _SiteSensor(env.sim.mj_model, 'robot/imu_ang_vel', self.kin, self.device)
        self.vel_sensor = _SiteSensor(env.sim.mj_model, 'robot/imu_lin_vel', self.kin, self.device)
        self.q0 = self.entity.data.default_joint_pos[:1].clone()
        self.qd0 = self.entity.data.default_joint_vel[:1].clone()
        self.limits = self.entity.data.soft_joint_pos_limits[:1].clone()
        self.default_height = self.entity.data.default_root_state[0, 2].clone()
        self.weights = {n: c.weight for n,c in zip(env.reward_manager._term_names,env.reward_manager._term_cfgs)}
        self.aux_indices = [env.reward_manager._term_names.index(n) for n in self.aux_names]
        self._aux = torch.zeros(self.num_envs, 3, device=self.device)
        self._obs = env.observation_manager.compute()
        self.noise = torch.zeros(96, device=self.device)
        self.noise[:3]=.2; self.noise[3:6]=.05; self.noise[9:38]=.01; self.noise[38:67]=.5
        self.corrupt = env.cfg.observations['actor'].enable_corruption
        loader=AMPLoader(motion_file=agent.amp_motion_files, body_names=agent.amp_body_names,
            anchor_name=agent.amp_anchor_name, all_body_names=self.entity.body_names, device=self.device)
        self.disc=StableMultiAMP(loader,agent.multi_amp,agent.amp_discr_hidden_dims,
            agent.amp_reward_coef,agent.amp_task_reward_lerp,self.device)
        self.last_disc_metrics={}
        compute=env.reward_manager.compute
        slip_cfg=env.reward_manager._term_cfgs[env.reward_manager._term_names.index('foot_slip')].params['asset_cfg']
        def capture(*args,**kwargs):
            if capture_verification:
                self.reward_snapshot={k:v.clone() for k,v in self.prepare_real().items()}
            result=compute(*args,**kwargs)
            self._aux=env.reward_manager._step_reward[:,self.aux_indices].clone()*self.step_dt
            contact=(env.scene['feet_ground_contact'].data.found>0).float()
            vel=self.entity.data.site_lin_vel_w[:,slip_cfg.site_ids,:2]
            self._aux[:,1]=(vel.square().sum(-1)*contact).sum(-1)*self.weights['foot_slip']*self.step_dt
            return result
        env.reward_manager.compute=capture

    def get_observations(self):
        return {'policy': self._obs['actor'], 'critic': self._obs['critic']}

    def get_side_state(self):
        c=self.command; tm=self.env.termination_manager
        return {k:v.clone() for k,v in dict(
            command=c.command, target=c.target, smooth=c.smooth, recovering=c.recovering,
            hold=c.hold, delay=tm._delay_counters, delay_mask=tm._delay_env_mask,
            last_action=self.env.action_manager.action,
            actor_hist=self._obs['actor'].reshape(self.num_envs,4,-1),
            critic_hist=self._obs['critic'].reshape(self.num_envs,4,-1),
        ).items()}

    def prepare(self,S):
        p,R,v,w,q,qd=S
        P,B,V,W=self.kin.forward(*S,all_bodies=True)
        return self.features(P,B,V,W,R,q,qd,p[:,2],w)

    def prepare_real(self):
        d=self.entity.data
        return self.features(d.body_link_pos_w-self.env.scene.env_origins[:,None],
            quat_to_mat(d.body_link_quat_w),d.body_link_lin_vel_w,d.body_link_ang_vel_w,
            quat_to_mat(d.root_link_quat_w),d.joint_pos,d.joint_vel,
            d.root_link_pos_w[:,2]-self.env.scene.env_origins[:,2],d.root_link_ang_vel_w)

    def features(self,P,B,V,W,R,q,qd,height,w):
        ids=self.amp_ids; br=B[:,self.body]; bp=P[:,self.body]
        pos=inv(br[:,None],P[:,ids]-bp[:,None]).flatten(1)
        ori=(br[:,None].transpose(-1,-2)@B[:,ids])[...,:2].flatten(1)
        amp=torch.cat((pos,ori,inv(B[:,ids],V[:,ids]).flatten(1),inv(B[:,ids],W[:,ids]).flatten(1)),-1)
        head=P[:,self.body,2]+.43*br[:,2,2]
        gravity=-R[:,2,:]
        geometry=torch.cat((head[:,None],-br[:,2,:]),-1)
        yaw=torch.atan2(br[:,1,0],br[:,0,0]); co,si=yaw.cos(),yaw.sin(); tv=V[:,self.body]
        xy=torch.stack((co*tv[:,0]+si*tv[:,1],-si*tv[:,0]+co*tv[:,1]),-1)
        return dict(amp=amp,geometry=geometry,head=head,gravity=gravity,height=height,q=q,
            gyro=self.gyro.read_fields(B,V,W), vel=self.vel_sensor.read_fields(B,V,W),
            xy=xy,wz=W[:,self.body,2],vz=tv[:,2],root_w=inv(R,w),qd=qd,pos=pos,ori=ori)

    def frames(self,P,command,action,noise=False):
        a=torch.cat((P['gyro'],P['gravity'],command,P['q']-self.q0,P['qd']-self.qd0,action),-1)
        c=torch.cat((a,P['vel'],P['pos'],P['ori']),-1)
        if noise and self.corrupt:
            a=a+(2*torch.rand_like(a)-1)*self.noise
        return a,c

    def transition(self,carry,P,action,P1,aux):
        # TerminationManager increments delay on consecutive bad states and clears
        # on recovery. Its timeout is a bootstrap boundary, not a physical death.
        bad=(P1['gravity'][:,2]>-math.cos(math.radians(70)))|(P1['height']<.5)
        delay=torch.where(carry['delay_mask']&bad,carry['delay']+1,0)
        done=bad&(~carry['delay_mask']|(delay>=250))
        delay=torch.where(done,0,delay)
        delayed=carry['delay_mask']&(delay>0)
        active=~carry['recovering']
        terms={
            'track_anchor_linear_velocity':torch.exp(-(P1['xy']-carry['command'][:,:2]).square().sum(-1)/.25)*active,
            'track_anchor_angular_velocity':torch.exp(-(P1['wz']-carry['command'][:,2]).square()/.25)*active,
            'track_root_height':torch.exp(-(self.default_height-P1['height']).square()/.09)*delayed*3.5,
            'body_ang_vel_xy_l2':torch.exp(-P1['root_w'][:,:2].square().sum(-1)/(3.14**2))*(~delayed),
            'is_terminated':done.float(),
            'joint_pos_limits':((self.limits[:,:,0]-P1['q']).clamp(min=0)+(P1['q']-self.limits[:,:,1]).clamp(min=0)).sum(-1),
            'action_rate_l2':(action-carry['last_action']).square().sum(-1),
            'vertical_motion_l2':P1['vz'].square()*active,
        }
        terms={k:v*self.weights[k]*self.step_dt for k,v in terms.items()}
        # Aux targets are already weighted per-step contributions; cap positive
        # predictions at zero because all three are costs, never bonuses.
        for i,n in enumerate(self.aux_names):terms[n]=aux[:,i].clamp(max=0)
        slip_active=(carry['command'][:,:2].norm(dim=-1)+carry['command'][:,2].abs())>.1
        terms['foot_slip']=terms['foot_slip']*slip_active
        task_reward=sum(terms.values())
        style=self.style(P['amp'],P1['amp'],carry['recovering'],carry['condition'],carry['supported'],done)
        reward=.75*task_reward+.25*style
        stable=(P1['height']>.65)&(P1['gravity'][:,2]<-.9)&(P1['head']>1.05)
        hold=torch.where(stable,carry['hold']+self.step_dt,0.)
        enter=((P1['gravity'][:,2]+1).abs()>.6)|(P1['head']<.9)
        recovery=(carry['recovering']|enter)&~(stable&(hold>=.5))
        delta=(carry['target']-carry['command']).clamp(-1.5*self.step_dt,1.5*self.step_dt)
        command=torch.where(carry['smooth'][:,None],carry['command']+delta,carry['target'])
        command=torch.where(recovery[:,None],0.,command)
        a,c=self.frames(P1,command,action,noise=True)
        next_carry={**carry,'P':P1,'command':command,'recovering':recovery,'hold':hold,'delay':delay,
            'last_action':action,'actor_hist':torch.cat((carry['actor_hist'][:,1:],a[:,None]),1),
            'critic_hist':torch.cat((carry['critic_hist'][:,1:],c[:,None]),1)}
        terms={**terms,'style':style,'task':task_reward}
        return next_carry,(reward,done,None,terms)

    def style(self,a,b,recovery,condition,supported,done):
        # Both networks frozen during imagination; gradients remain w.r.t. states.
        rec=self.disc.networks['rec'](self.disc._input('rec',a,b,condition)).squeeze(-1)
        loco=self.disc.networks['loco'](self.disc._input('loco',a,b,condition)).squeeze(-1)
        r=.1*(1-.25*(rec-1).square()).clamp(min=0)
        l=.1*(1-.25*(loco-1).square()).clamp(min=0)*supported
        return torch.where(recovery,r,l)*(~done)

    def step(self,actions):
        # Use exactly C's pre-step feature and routing metadata for real style.
        a=self._obs['amp'].clone(); rec=self.command.recovering.clone(); command=self.command.command.clone()
        P=self.prepare(self.layout.unflatten(self.get_state()))
        self._terminal_mask.zero_(); self._written.zero_()
        obs,reward,term,trunc,extras=self.env.step(actions)
        self._obs=obs; done=term|trunc
        reward=self.disc.reward_and_store(a,obs['amp'],reward,rec,command,done,geometry=P['geometry'])
        return self.get_observations(),reward,done,dict(time_outs=trunc,
            terminal_state=torch.where(self._terminal_mask[:,None],self._terminal,self.get_state()),
            teleported=self._written&~done,log=extras.get('log',{}))

    def get_wm_aux(self):return self._aux

    def imagine(self,side=None,past=None):
        side=self.get_side_state() if side is None else side
        # Match the target command once per rollout. Future command resampling
        # is intentionally held fixed over the 0.64s imagination horizon.
        effective=torch.where(side['recovering'][:,None],side['target'],side['command'])
        idx,supported=self.disc._matched_expert(effective)
        c={k:v.clone() for k,v in side.items()}
        c['condition']=self.disc.expert_velocity[idx]/self.disc.scale
        c['supported']=supported
        return AMPRollout(self,c)

    def update_discriminators(self,steps=20):
        for p in self.disc.parameters():p.requires_grad_(True)
        self.last_disc_metrics=self.disc.update(steps)
        for p in self.disc.parameters():p.requires_grad_(False)


class AMPRollout(CarryRollout):
    @staticmethod
    def begin(task,carry,S):return {**carry,'P':task.prepare(S)}
    @staticmethod
    def observe_fn(task,carry,S):
        return {'policy':carry['actor_hist'].flatten(1),'critic':carry['critic_hist'].flatten(1)}
    @staticmethod
    def step_fn(task,carry,S,action,S1,aux):
        return task.transition(carry,carry['P'],action,task.prepare(S1),aux)
