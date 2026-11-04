"""Evaluation-only EMA weights; live normalizer buffers keep their exact statistics."""
import torch

class PolicyEMA:
    def __init__(self, policy, decay):
        if not 0 < decay < 1:
            raise ValueError('EMA decay must be in (0,1)')
        self.decay = decay
        self.updates = 0
        self.parameters = set(dict(policy.named_parameters()))
        self.weights = {k:v.detach().clone() for k,v in policy.state_dict().items()}

    @torch.no_grad()
    def update(self, policy):
        for key,value in policy.state_dict().items():
            if key in self.parameters:
                self.weights[key].lerp_(value.detach(), 1-self.decay)
            else:
                self.weights[key].copy_(value)
        self.updates += 1

    def state_dict(self):
        return self.weights
