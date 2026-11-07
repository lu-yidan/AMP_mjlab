"""BlinkMimic sagittal WM augmentation, adapted to AMP's ordered DR context.

Half the sampled windows are reflected, without doubling the batch or real data.
Auxiliary costs are left/right aggregate scalars and are reflection invariant.
"""
import torch

class AMPMirror:
    def __init__(self, names, layout, context_terms, device):
        self.layout=layout
        partner=lambda n:n.replace('left_','@').replace('right_','left_').replace('@','right_')
        self.perm=torch.tensor([names.index(partner(n)) for n in names],device=device)
        self.sign=torch.tensor([-1. if '_roll_' in n or '_yaw_' in n else 1. for n in names],device=device)
        self.M=torch.diag(torch.tensor([1.,-1.,1.],device=device))
        expected={'foot_friction':1,'encoder_bias':len(names),'base_com':3}
        if dict(context_terms)!=expected:raise ValueError(f'Undefined mirror context: {context_terms}')
        self.slices={};start=0
        for name,width in context_terms:self.slices[name]=slice(start,start+width);start+=width
    def joints(self,x):return x[...,self.perm]*self.sign
    def state(self,x):
        p,R,v,w,q,qd=self.layout.unflatten(x);M=self.M
        return self.layout.flatten((p@M,M@R@M,v@M,-(w@M),self.joints(q),self.joints(qd)))
    def xi(self,x):
        y=x.clone();s=self.slices
        y[...,s['encoder_bias']]=self.joints(x[...,s['encoder_bias']])
        y[...,s['base_com'].start+1]*=-1
        return y
    def windows(self,w):
        out=dict(w);n=w['s'].shape[0]//2
        for key,fn in [('s',self.state),('s1',self.state),('a',self.joints),('xi',self.xi)]:
            out[key]=torch.cat((fn(w[key][:n]),w[key][n:]),0)
        return out
