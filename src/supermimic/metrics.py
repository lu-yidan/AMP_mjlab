"""Real-simulator monitoring only; does not change rewards or learning."""
import torch
from .mdp import is_recovery
from .rotations import yaw_mat


class TrainingMetrics:
    def __init__(self,n,device,dt):
        self.dt=dt
        self.active=torch.zeros(n,dtype=torch.bool,device=device)
        self.elapsed=torch.zeros(n,device=device)
        self.stable=torch.zeros(n,dtype=torch.long,device=device)
        self.sums={}

    def add(self,key,value):
        self.sums[key]=self.sums.get(key,0.)+float(value)

    @torch.no_grad()
    def observe(self,tr,S,S1):
        up=~is_recovery(S1);reset=tr['ep_step']==0
        self.active[reset]=False;self.elapsed[reset]=0;self.stable[reset]=0
        starts=~self.active&(is_recovery(S)|(is_recovery(S1)))
        self.active|=starts
        self.elapsed=torch.where(self.active,self.elapsed+self.dt,0)
        self.stable=torch.where(self.active&up,self.stable+1,0)
        success=self.active&(self.stable>=round(1/self.dt))&~tr['terminated']
        failed=self.active&tr['done']&~success
        self.add('attempts',starts.sum());self.add('successes',success.sum());self.add('failures',failed.sum())
        self.add('recovery_time',self.elapsed[success].sum())
        recovery_reset=tr.get('reset_group',torch.zeros_like(up))>0
        self.add('reset_successes',(success&recovery_reset).sum())
        self.add('reset_failures',(failed&recovery_reset).sum())
        self.active&=~(success|failed)
        velocity=(yaw_mat(S1[1]).transpose(-1,-2)@S1[2].unsqueeze(-1)).squeeze(-1)
        target=tr['command'][:,0];error=(velocity[:,0]-target).abs()
        moving=target>.05;upright_moving=moving&up
        for key,value in dict(steps=len(up),upright=up.sum(),moving=moving.sum(),upright_moving=upright_moving.sum(),
            command=target.sum(),actual=velocity[:,0].sum(),error=error.sum(),moving_error=error[moving].sum(),
            upright_error=error[upright_moving].sum(),vy_error=velocity[:,1].abs().sum(),
            yaw_error=(S1[3][:,2]-tr['command'][:,2]).abs().sum(),
            episodes=tr['done'].sum(),episode_seconds=((tr['ep_step']+1)*self.dt)[tr['done']].sum(),
            catastrophic=tr['terminated'].sum(),time_limit=(tr['done']&~tr['terminated']&~tr.get('budget_reset',torch.zeros_like(up))).sum()).items():self.add(key,value)

    def flush(self):
        s=self.sums;d=max(s.get('steps',0),1);resolved=s.get('successes',0)+s.get('failures',0)
        result={'Tracking/command_vx_mean':s.get('command',0)/d,'Tracking/actual_vx_mean':s.get('actual',0)/d,
            'Tracking/vx_mae_all':s.get('error',0)/d,'Tracking/vy_mae':s.get('vy_error',0)/d,
            'Tracking/yaw_rate_mae':s.get('yaw_error',0)/d,'Survival/upright_fraction':s.get('upright',0)/d,
            'Survival/completed_episodes':s.get('episodes',0),'Survival/catastrophic_resets':s.get('catastrophic',0),
            'Recovery/attempts_started':s.get('attempts',0),'Recovery/successes':s.get('successes',0),
            'Recovery/failures':s.get('failures',0),'Recovery/resolved_attempts':resolved,
            'Recovery/active_fraction':float(self.active.float().mean()),
            'Tracking/upright_moving_coverage':s.get('upright_moving',0)/d}
        for denominator,key,numerator in [('moving','Tracking/vx_mae_moving','moving_error'),
                ('upright_moving','Tracking/vx_mae_upright_moving','upright_error'),
                ('episodes','Survival/mean_episode_seconds','episode_seconds'),
                ('successes','Recovery/mean_success_seconds','recovery_time')]:
            if s.get(denominator,0)>0:result[key]=s[numerator]/s[denominator]
        if resolved:result['Recovery/success_rate']=s['successes']/resolved
        reset_resolved=s.get('reset_successes',0)+s.get('reset_failures',0)
        result['Recovery/reset_pose_resolved_attempts']=reset_resolved
        if reset_resolved:result['Recovery/reset_pose_success_rate']=s['reset_successes']/reset_resolved
        self.sums={}
        return result
