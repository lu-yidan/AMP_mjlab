"""Explicit CUDA update graphs following BlinkMimic train.Graphed's lifecycle.

Replay sampling, expert-condition lookup, simulator collection and discriminator
updates stay outside capture. Static inputs are refreshed on every invocation.
Learning rates and running statistics remain live device tensors. Three eager
warmup updates count normally; capture records one update, then replays it once.
"""
import time
import torch
from torch.utils._pytree import tree_map
from .blink_mirror import AMPMirror

def clone_tree(x):return tree_map(lambda t:t.detach().clone() if torch.is_tensor(t) else t,x)
def copy_tree(dst,src):
    tree_map(lambda d,s:d.copy_(s) if torch.is_tensor(d) else None,dst,src)

class UpdateGraph:
    def __init__(self,fn,name):self.fn,self.name,self.calls,self.graph=fn,name,0,None
    def __call__(self):
        if self.graph is None and self.calls<3:
            stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):out=self.fn()
            torch.cuda.current_stream().wait_stream(stream)
            self.calls+=1
            return clone_tree(out)
        if self.graph is None:
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            self.graph=torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph):self.out=self.fn()
            print(f'[CustomCUDAGraph] captured {self.name}',flush=True)
        self.graph.replay()
        return clone_tree(self.out)

def install_mirror_graph(runner,task):
    a=runner.alg;c=a.cfg
    from supermimic.utils.distributed import world_size
    if world_size()!=1:raise ValueError("This graph adapter is single-GPU only")
    task.disc.graph_safe_checks=True
    if torch.device(a.device).type != 'cuda':
        raise ValueError('Custom CUDA Graph requires CUDA')
    mirror=AMPMirror(list(task.entity.joint_names),task.layout,task.context_terms,task.device)
    # Signed permutation is valid only for symmetric target offsets/scales.
    torch.testing.assert_close(mirror.joints(task.q0),task.q0)
    action=task.env.action_manager.get_term('joint_pos')
    scale=torch.as_tensor(action._scale,device=task.device)
    if scale.numel()>1:torch.testing.assert_close(scale[...,mirror.perm],scale)
    # Check the signed joint map against the loaded robot's forward kinematics.
    q=task.q0+.2*torch.sin(torch.arange(task.num_actions,device=task.device)[None])
    z=torch.zeros(1,3,device=task.device);R=torch.eye(3,device=task.device)[None]
    state=(z,R,z,z,q,torch.zeros_like(q))
    reflected=task.layout.unflatten(mirror.state(task.layout.flatten(state)))
    P,B,_,_=task.kin.forward(*state,all_bodies=True)
    Pm,Bm,_,_=task.kin.forward(*reflected,all_bodies=True)
    names=list(task.entity.body_names)
    partner=lambda n:n.replace('left_','@').replace('right_','left_').replace('@','right_')
    ids=[names.index(partner(n)) for n in names]
    pos_error=(Pm-P[:,ids]@mirror.M).abs().max().item()
    rot_error=(Bm-mirror.M@B[:,ids]@mirror.M).abs().max().item()
    if pos_error>1e-4 or rot_error>1e-4:
        raise ValueError(f'Robot mirror FK mismatch: position={pos_error}, rotation={rot_error}')
    print('[Mirror50] context',task.context_terms,'q0/action scale symmetry verified',
          'FK max errors',pos_error,rot_error,flush=True)
    # Capturable fused critic Adam; actor and WM use upstream capturable MuonAdam.
    a.opt_v=torch.optim.Adam(a.critic_params,lr=torch.tensor(c.critic_learning_rate,device=task.device),fused=True,capturable=True)
    original_policy=a.update_policy
    a._capture_finite=torch.ones((),dtype=torch.bool,device=task.device)
    wm_graph=policy_graph=None
    wm_static=policy_static=None
    params=list(a.wm.parameters())
    def check(out):
        if not bool(a._capture_finite):raise FloatingPointError('Non-finite custom CUDA graph update')
        return out
    def update_wm(layout):
        nonlocal wm_graph,wm_static
        if len(a.replay)<=c.wm_window:return {}
        loss_fn=a._compiled(('wm',id(layout)),a._wm_loss_fn(layout))
        def step():
            a._capture_finite.fill_(True)
            w=wm_static
            ls,la,last=loss_fn(w['s'],w['a'],w['xi'],w['s1'],w['aux'],w['mask'])
            a.opt_wm.zero_grad(set_to_none=True);(ls+la).backward()
            gn=torch.nn.utils.clip_grad_norm_(params,c.max_grad_norm)
            ok=torch.isfinite(ls+la)&torch.isfinite(gn)
            a._capture_finite.logical_and_(ok);a.opt_wm.step(ok)
            return {'wm/loss':ls.detach()/c.wm_window,'wm/aux_loss':la.detach()/c.wm_window,'wm/err_last':last}
        for _ in range(c.wm_updates):
            w=mirror.windows(a.replay.sample_windows(c.wm_batch,c.wm_window))
            if wm_static is None:
                wm_static=clone_tree(w);wm_graph=UpdateGraph(step,'world-model forward/backward/optimizer')
            else:copy_tree(wm_static,w)
            out=check(wm_graph())
        return out
    def update_policy(task,obs):
        nonlocal policy_static,policy_graph
        s,xi,side,past=a.replay.sample_starts(task.history_length)
        rollout=task.imagine({**side,'xi':xi},past)
        inputs=(s,xi,rollout.carry)
        if policy_static is None:
            policy_static=clone_tree(inputs)
            rollout.carry=policy_static[2]
            a._prepared_policy=(policy_static[0],policy_static[1],rollout)
            def step():
                a._capture_finite.fill_(True)
                out=original_policy(task,obs)
                return {k:v for k,v in out.items() if not k.startswith('time/')}
            policy_graph=UpdateGraph(step,'actor and critic forward/backward/optimizers')
        else:copy_tree(policy_static,inputs)
        previous=c.fail_on_nonfinite;c.fail_on_nonfinite=False
        t=time.monotonic()
        try:out=check(policy_graph())
        finally:c.fail_on_nonfinite=previous
        out['time/policy_update_wall']=time.monotonic()-t
        out['graph/policy_captured']=float(policy_graph.graph is not None)
        out['graph/wm_captured']=float(wm_graph is not None and wm_graph.graph is not None)
        return out
    a.update_world_model=update_wm;a.update_policy=update_policy
