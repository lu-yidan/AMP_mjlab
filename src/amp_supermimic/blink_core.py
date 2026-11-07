"""BlinkMimic structured drift adapted to the AMP state/auxiliary interface.

Reference: fuyichen0725/blinkmimic d264658, networks.WorldModel.
No reference-pose action skip or mirror augmentation is implied by this module.
"""
import torch
from supermimic.modules.world_model import WorldModel
from supermimic.modules.normalizer import RunningNorm
from supermimic.utils.rotations import yaw_mat, so3_exp


def mv(R, v):
    return (R @ v.unsqueeze(-1)).squeeze(-1)


class BlinkStructuredWorldModel(WorldModel):
    def __init__(self, *args, dt=.02, **kwargs):
        super().__init__(*args, **kwargs)
        self.dt = dt
        self.inc_norm = RunningNorm(self.layout.delta_dim, eps=1e-4)

    def drift(self, S, v1, w1, qd1):
        _, R, v, w, _, qd = S
        a, b = self.dt * .375, self.dt * .625
        return (mv(yaw_mat(R).transpose(-1, -2), a*v+b*v1),
                mv(R.transpose(-1, -2), a*w+b*w1), a*qd+b*qd1)

    def targets(self, S, S1):
        dp, dr, dv, dw, dq, dqd = WorldModel.targets(S, S1).split((3,3,3,3,self.nj,self.nj), -1)
        gp, gr, gq = self.drift(S, S1[2], S1[3], S1[5])
        return torch.cat((dp-gp, dr-gr, dv, dw, dq-gq, dqd), -1)

    @torch.no_grad()
    def update_normalization(self, S, a, xi, S1, aux=None):
        super().update_normalization(S, a, xi, S1, aux)
        self.inc_norm.update(WorldModel.targets(S, S1))

    def integrate(self, S, d):
        p,R,v,w,q,qd = S
        dp,dr,dv,dw,dq,dqd = d.split((3,3,3,3,self.nj,self.nj), -1)
        v1,w1,qd1 = v+mv(R,dv),w+mv(R,dw),qd+dqd
        gp,gr,gq = self.drift(S,v1,w1,qd1)
        return p+mv(yaw_mat(R),dp+gp), R@so3_exp(dr+gr),v1,w1,q+dq+gq,qd1

    def step(self, S, a, xi):
        x=self.in_norm(self.inputs(S,a,xi))
        with torch.autocast(x.device.type,dtype=self.amp_dtype or torch.float32,enabled=self.amp_dtype is not None):
            y=self.net(x).float()
        n=self.layout.delta_dim
        d=y[...,:n]*self.out_norm.std()+self.out_norm.mean
        aux=y[...,n:]*self.aux_norm.std()+self.aux_norm.mean if self.aux_dim else None
        return self.integrate(S,d),aux


def configure_blink_optimizers(runner):
    from third_party.blinkmimic_core.optim import MuonAdam
    # Use actor-only module: do not inadvertently give MuonAdam critic parameters.
    a=runner.alg
    a.opt_pi=MuonAdam(runner.policy.actor,[runner.policy.actor],lr=a.cfg.learning_rate,betas=(.9,.95))
    a.opt_pi.defaults={'fused':False}
    a.opt_wm=MuonAdam(a.wm,[a.wm.net],lr=a.cfg.wm_learning_rate,betas=(.9,.999))
