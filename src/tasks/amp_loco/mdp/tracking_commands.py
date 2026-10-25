"""Explicit command modes, recovery hysteresis and measured speed curriculum."""
from dataclasses import dataclass
import torch
from mjlab.tasks.velocity.mdp.velocity_command import UniformVelocityCommand, UniformVelocityCommandCfg
from mjlab.utils.lab_api.math import quat_apply, quat_apply_inverse, yaw_quat

MODES = ('forward', 'backward', 'side', 'turn', 'mixed', 'stand')
PROBABILITIES = (.25, .15, .15, .15, .20, .10)
LIMITS = ((-.5, 1., .3, .5), (-1., 2., .6, 1.), (-1., 3., 1., 1.5))


def sample_commands(modes, stage):
    """Sample signed SI-unit commands; retain small speeds at every stage."""
    lo, hi, side, yaw = LIMITS[stage]
    n=len(modes);device=modes.device
    u=torch.rand(n,3,device=device)
    small=torch.rand(n,device=device)<.35
    forward=.1+u[:,0]*(torch.where(small,torch.full_like(u[:,0],min(.6,hi)),torch.full_like(u[:,0],hi))-.1)
    backward=-(.1+u[:,0]*(-lo-.1))
    vy=(.1+u[:,1]*(side-.1))*torch.where(torch.rand(n,device=device)<.5,-1.,1.)
    wz=(.15+u[:,2]*(yaw-.15))*torch.where(torch.rand(n,device=device)<.5,-1.,1.)
    out=torch.zeros(n,3,device=device)
    out[modes==0,0]=forward[modes==0]
    out[modes==1,0]=backward[modes==1]
    out[modes==2,1]=vy[modes==2]
    out[modes==3,2]=wz[modes==3]
    mixed=modes==4
    # Keep combined turns at <=1 m/s in the first curriculum stages.
    out[mixed,0]=(lo+u[:,0]*(min(hi,1.)-lo))[mixed]
    out[mixed,1]=vy[mixed]
    out[mixed,2]=wz[mixed]
    return out


def recovery_transition(recovering,hold,height,gz,head,dt):
    enter=((gz+1).abs()>.6)|(head<.9)
    stable=(height>.65)&(gz<-.9)&(head>1.05)
    hold=torch.where(stable,hold+dt,0.)
    recovering=(recovering|enter)&~(stable&(hold>=.5))
    return recovering,hold


class TrackingCommand(UniformVelocityCommand):
    def __init__(self,cfg,env):
        super().__init__(cfg,env)
        self.body=list(self.robot.body_names).index('torso_link')
        self.target=torch.zeros_like(self.vel_command_b)
        self.mode=torch.zeros(self.num_envs,dtype=torch.long,device=self.device)
        self.smooth=torch.zeros(self.num_envs,dtype=torch.bool,device=self.device)
        self.recovering=torch.ones_like(self.smooth)
        self.hold=torch.zeros(self.num_envs,device=self.device)
        self.age=torch.zeros_like(self.hold)
        self.stage=2 if cfg.full_speed else 0
        self.count=torch.zeros(6,device=self.device);self.sums=torch.zeros(6,3,device=self.device)
        self.last_means=torch.zeros(6,3,device=self.device)
        self.recovery_sum=torch.zeros((),device=self.device);self.window_steps=0
        for key in ('stage','recovering_fraction',*[f'{m}_{s}' for m in MODES for s in ('xy_mae','yaw_mae','upright')]):
            self.metrics[key]=torch.zeros_like(self.hold)

    def _resample_command(self,ids):
        self.mode[ids]=torch.multinomial(torch.tensor(PROBABILITIES,device=self.device),len(ids),replacement=True)
        self.target[ids]=sample_commands(self.mode[ids],self.stage)
        self.smooth[ids]=torch.rand(len(ids),device=self.device)<.5
        self.age[ids]=0.
        self.is_standing_env[ids]=self.mode[ids]==5
        self.is_heading_env[ids]=False

    def reset(self,env_ids):
        result=super().reset(env_ids)
        self.vel_command_b[env_ids]=0.
        self.recovering[env_ids]=True;self.hold[env_ids]=0.
        return result

    def physical_state(self):
        data=self.robot.data
        q=data.body_link_quat_w[:,self.body]
        offset=quat_apply(q,torch.tensor([0.,0.,.43],device=self.device).expand(self.num_envs,-1))
        head=data.body_link_pos_w[:,self.body,2]+offset[:,2]-self._env.scene.env_origins[:,2]
        height=data.root_link_pos_w[:,2]-self._env.scene.env_origins[:,2]
        return height,data.projected_gravity_b[:,2],head

    def _update_command(self):
        self.recovering,self.hold=recovery_transition(self.recovering,self.hold,*self.physical_state(),self._env.step_dt)
        delta=(self.target-self.vel_command_b).clamp(-self.cfg.acceleration*self._env.step_dt,self.cfg.acceleration*self._env.step_dt)
        self.vel_command_b[:]=torch.where(self.smooth[:,None],self.vel_command_b+delta,self.target)
        self.vel_command_b[self.recovering]=0.
        self.age+=self._env.step_dt
        self.age[self.recovering]=0.

    def _update_metrics(self):
        super()._update_metrics()
        h,gz,_=self.physical_state()
        velocity=quat_apply_inverse(yaw_quat(self.robot.data.body_link_quat_w[:,self.body]),self.robot.data.body_link_lin_vel_w[:,self.body])
        xy=(velocity[:,:2]-self.command[:,:2]).norm(dim=-1)
        yaw=(self.robot.data.body_link_ang_vel_w[:,self.body,2]-self.command[:,2]).abs()
        stable=((h>.65)&(gz<-.9)).float()
        valid=(~self.recovering)&(self.age>1.)
        ids=self.mode[valid]
        self.count.scatter_add_(0,ids,torch.ones_like(xy[valid]))
        self.sums.index_add_(0,ids,torch.stack((xy[valid],yaw[valid],stable[valid]),-1))
        self.recovery_sum+=self.recovering.float().mean();self.window_steps+=1
        self.metrics['stage'][:]=self.stage
        self.metrics['recovering_fraction'][:]=self.recovering.float().mean()
        for i,m in enumerate(MODES):
            for j,k in enumerate(('xy_mae','yaw_mae','upright')):self.metrics[f'{m}_{k}'][:]=self.last_means[i,j]
        if self.window_steps>=5000:
            means=self.sums/self.count[:,None].clamp(min=1)
            self.last_means.copy_(means)
            for i,m in enumerate(MODES):
                for j,k in enumerate(('xy_mae','yaw_mae','upright')):self.metrics[f'{m}_{k}'][:]=means[i,j]
            min_steps=(120000,360000)
            good=(self.count>5000).all()&(means[:,0]<.3).all()&(means[:,1]<.35).all()&(means[:,2]>.95).all()&(self.recovery_sum/self.window_steps<.15)
            if not self.cfg.full_speed and self.stage<2 and self._env.common_step_counter>=min_steps[self.stage] and bool(good):
                self.stage+=1
                print(f'[TrackingCurriculum] stage={self.stage} limits={LIMITS[self.stage]} step={self._env.common_step_counter}',flush=True)
            self.count.zero_();self.sums.zero_();self.recovery_sum.zero_();self.window_steps=0


@dataclass(kw_only=True)
class TrackingCommandCfg(UniformVelocityCommandCfg):
    acceleration: float = 1.5
    full_speed: bool = False
    def build(self,env):return TrackingCommand(self,env)


def track_xy(env,std=.5):
    c=env.command_manager.get_term('twist');r=c.robot
    v=quat_apply_inverse(yaw_quat(r.data.body_link_quat_w[:,c.body]),r.data.body_link_lin_vel_w[:,c.body])
    return torch.exp(-((v[:,:2]-c.command[:,:2])**2).sum(-1)/std**2)*(~c.recovering)


def track_yaw(env,std=.5):
    c=env.command_manager.get_term('twist')
    error=c.robot.data.body_link_ang_vel_w[:,c.body,2]-c.command[:,2]
    return torch.exp(-error.square()/std**2)*(~c.recovering)


def vertical_motion_cost(env):
    c=env.command_manager.get_term('twist')
    return c.robot.data.body_link_lin_vel_w[:,c.body,2].square()*(~c.recovering)
