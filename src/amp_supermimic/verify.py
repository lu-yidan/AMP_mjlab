"""Numerical gate against actual transitions, never policy performance evidence."""
import torch

def verify(t):
    obs_err={};reward_err={};stale_reward_err={};term_bad=0;stale_term_bad=0;total=0;amp_error=0.;grad_ok=False;gradient_error=0.
    # Compare current kinematics and observations, then evaluate true next states
    # with the real auxiliary targets. Exclude reset and command resampling rows.
    for step in range(32):
        side=t.get_side_state();s=t.get_state();P=t.prepare(t.layout.unflatten(s))
        a,c=t.frames(P,side['command'],side['last_action'])
        for k,x,y in [('actor',a,side['actor_hist'][:,-1]),('critic',c,side['critic_hist'][:,-1])]:
            err=float((x-y).abs().max())
            if k=='critic' and hasattr(t,'terrain_map') and err>obs_err.get(k,0):
                idx=(x-y).abs().flatten().argmax();row=int(idx//x.shape[1]);col=int(idx%x.shape[1])
                print('TERRAIN_OBS_ERROR',dict(step=step,row=row,col=col,pred=float(x[row,col]),actual=float(y[row,col]),root=t.entity.data.root_link_pos_w[row].tolist(),sensor=t.env.scene['terrain_scan'].data.pos_w[row].tolist()),flush=True)
            obs_err[k]=max(obs_err.get(k,0),err)
        amp_error=max(amp_error,float((P['amp']-t._obs['amp']).abs().max()))
        roll=t.imagine(side);roll._begin(t.layout.unflatten(s))
        action=.1*torch.randn(t.num_envs,29,device=t.device)
        _,_,done,extra=t.step(action)
        raw={n:t.env.reward_manager._step_reward[:,i]*t.step_dt for i,n in enumerate(t.env.reward_manager._term_names)}
        s1=extra['terminal_state'];S1=t.layout.unflatten(s1)
        _,(_,stale_done,_,stale_terms)=t.transition(roll.carry,P,action,t.prepare(S1),t.get_wm_aux())
        nxt,(reward,pred_done,_,terms)=t.transition(roll.carry,P,action,t.reward_snapshot,t.get_wm_aux())
        valid=~done & ~extra['teleported']
        for k in raw:
            if valid.any():
                reward_err[k]=max(reward_err.get(k,0),float((terms[k][valid]-raw[k][valid]).abs().max()))
                stale_reward_err[k]=max(stale_reward_err.get(k,0),float((stale_terms[k][valid]-raw[k][valid]).abs().max()))
        actual_term=done&~extra['time_outs']
        term_bad+=int((pred_done!=actual_term).sum());stale_term_bad+=int((stale_done!=actual_term).sum());total+=t.num_envs
        if step==31:
            s1=s1.detach().requires_grad_(True)
            _,(reward,*_)=t.transition(roll.carry,P,action,t.prepare(t.layout.unflatten(s1)),t.get_wm_aux())
            g=torch.autograd.grad(reward.sum(),s1)[0]
            grad_ok=bool(torch.isfinite(g).all() and g.abs().sum()>0)
            # Central differences on velocities and joint positions, away from
            # boolean routing boundaries. No optimizer or model approximation.
            for col in (7,8,10,13,18,25):
                plus=s1.detach().clone();minus=s1.detach().clone();eps=1e-3
                plus[0,col]+=eps;minus[0,col]-=eps
                def value(x):
                    return t.transition(roll.carry,P,action,t.prepare(t.layout.unflatten(x)),t.get_wm_aux())[1][0][0]
                fd=(value(plus)-value(minus))/(2*eps)
                gradient_error=max(gradient_error,float((fd-g[0,col]).abs()))
    result=dict(obs_max_abs=obs_err,amp_max_abs=amp_error,reward_max_abs=reward_err,
        termination_mismatch=term_bad/total,post_step_state_reward_max_abs=stale_reward_err,post_step_state_termination_mismatch=stale_term_bad/total,gradient_finite_nonzero=grad_ok,gradient_fd_max_abs=gradient_error)
    result['passed']=max(obs_err.values())<1e-3 and amp_error<1e-3 and max(reward_err.values())<1e-5 and term_bad==0 and grad_ok and gradient_error<1e-3
    return result
