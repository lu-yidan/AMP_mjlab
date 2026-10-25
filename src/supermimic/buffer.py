"""Fixed-size GPU replay buffer of simulator transitions, laid out as (time, env)."""

from __future__ import annotations

import torch

from .mdp import STATE_DIM, XI_DIM, is_recovery, unflatten_state

FIELDS = {
    "s": (STATE_DIM, torch.float32),
    "s1": (STATE_DIM, torch.float32),
    "a": (29, torch.float32),
    "a_prev": (29, torch.float32),
    "xi": (XI_DIM, torch.float32),
    "r": ((), torch.float32),
    "command": (3, torch.float32),
    "ep_step": ((), torch.int32),
    "done": ((), torch.bool),
    "recovery": ((), torch.bool),
}


class ReplayBuffer:
    def __init__(self, T: int, N: int, device: str):
        self.T, self.N, self.device = T, N, device
        self.data = {}
        for k, (shape, dt) in FIELDS.items():
            shp = (T, N) + ((shape,) if isinstance(shape, int) else shape)
            self.data[k] = torch.zeros(shp, dtype=dt, device=device)
        self.ptr, self.size = 0, 0

    def add(self, tr: dict):
        if "recovery" not in tr:
            tr = {**tr, "recovery": is_recovery(unflatten_state(tr["s"]))}
        for k in FIELDS:
            self.data[k][self.ptr] = tr[k]
        self.ptr = (self.ptr + 1) % self.T
        self.size = min(self.size + 1, self.T)

    def _row(self, age: torch.Tensor) -> torch.Tensor:
        """Map age-ordered time index (0 = oldest) to the storage row."""
        return (self.ptr - self.size + age) % self.T

    def _sample(self, B, size, fraction):
        if size <= 0:raise ValueError('Insufficient replay history')
        if fraction is None:
            flat=torch.randint(size*self.N,(B,),device=self.device)
        else:
            flags=self.data['recovery'][self._row(torch.arange(size,device=self.device))].flatten()
            pools=[(~flags).nonzero().flatten(),flags.nonzero().flatten()]
            counts=[B-round(B*fraction),round(B*fraction)]
            draws=[]
            for pool,count in zip(pools,counts):
                if not len(pool):pool=torch.arange(size*self.N,device=self.device)
                draws.append(pool[torch.randint(len(pool),(count,),device=self.device)])
            flat=torch.cat(draws);flat=flat[torch.randperm(B,device=self.device)]
        return flat//self.N,flat%self.N

    def sample_windows(self, B: int, k: int, recovery_fraction=None):
        """k-step windows; returns dict of (B, k, ...) tensors and a validity mask (B, k)."""
        t0,env = self._sample(B,self.size-k+1,recovery_fraction)
        ages = t0.unsqueeze(-1) + torch.arange(k, device=self.device)
        rows = self._row(ages)
        e = env.unsqueeze(-1).expand_as(rows)
        out = {key: self.data[key][rows, e] for key in ("s", "s1", "a", "xi", "done", "recovery")}
        # transition j is valid if no episode ended at transitions 0..j-1
        d = out["done"].float()
        mask = torch.cumprod(torch.cat((torch.ones(B, 1, device=self.device), 1 - d[:, :-1]), -1), -1)
        out["mask"] = mask
        return out

    def sample_starts(self, B: int, hist: int, recovery_fraction=None):
        """Start points for imagination plus the indices of their observation history (oldest first)."""
        age,env = self._sample(B,self.size,recovery_fraction)
        row = self._row(age)
        ep_step = self.data["ep_step"][row, env].long()
        back = torch.arange(hist - 1, -1, -1, device=self.device)  # hist-1 ... 0
        back = torch.minimum(back.unsqueeze(0), ep_step.unsqueeze(-1))  # stay inside the episode
        back = torch.minimum(back, age.unsqueeze(-1))  # stay inside the buffer
        hrows = self._row(age.unsqueeze(-1) - back)
        henv = env.unsqueeze(-1).expand_as(hrows)
        start = {k: self.data[k][row, env] for k in ("s", "a_prev", "xi", "command", "recovery")}
        hist_d = {k: self.data[k][hrows, henv] for k in ("s", "a_prev", "xi", "command", "recovery")}
        return start, hist_d
