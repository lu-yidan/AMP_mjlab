"""GPU ring buffer of the last ``capacity`` vectorized simulator transitions, laid out as (time, env),
sampled as k-step windows for the world model's multi-step loss."""

from __future__ import annotations

import torch


class TransitionReplay:
    def __init__(self, capacity: int, num_envs: int, state_dim: int, num_actions: int, xi_dim: int = 0,
                 aux_dim: int = 0, device="cpu"):  # fmt: skip
        self.T, self.N, self.device = capacity, num_envs, device
        z = lambda *s, dtype=torch.float32: torch.zeros(capacity, num_envs, *s, dtype=dtype, device=device)  # noqa: E731
        self.data = {"s": z(state_dim), "s1": z(state_dim), "a": z(num_actions), "xi": z(xi_dim), "aux": z(aux_dim),
                     # valid: s1 is the true successor of s; cont: the next row continues this one
                     "valid": z(dtype=torch.bool), "cont": z(dtype=torch.bool)}  # fmt: skip
        self.ptr, self.size = 0, 0
        self.startable = 0  # newest rows imagination may start from (the reference changes on a motion resample)

    def __len__(self):
        return self.size

    def stale_references(self):
        """The task loaded other motions: rows stored so far keep training the world model, but their reference
        cursors point into motions no longer loaded, so imagination no longer starts from them."""
        self.startable = 0

    @torch.no_grad()
    def add(self, s, a, s1, xi, aux, valid, cont, side: dict[str, torch.Tensor] | None = None, gap_tol: float = 1e-4):
        """side: the task's side state at s (see TrackingTask.get_side_state), stored next to the transition."""
        for k, v in (side or {}).items():
            if "side/" + k not in self.data:
                self.data["side/" + k] = torch.zeros(self.T, *v.shape, dtype=v.dtype, device=self.device)
            self.data["side/" + k][self.ptr] = v
        if self.size:
            # a teleport outside step() (motion resampling, reset_all) is not a done: cut the window where
            # this state does not continue the previous transition
            prev = (self.ptr - 1) % self.T
            self.data["cont"][prev] &= (s - self.data["s1"][prev]).abs().amax(-1) <= gap_tol
        row = {"s": s, "s1": s1, "a": a, "xi": xi, "valid": valid, "cont": cont}
        if aux is not None:
            row["aux"] = aux
        for k, v in row.items():
            self.data[k][self.ptr] = v
        self.ptr = (self.ptr + 1) % self.T
        self.size = min(self.size + 1, self.T)
        self.startable = min(self.startable + 1, self.T)

    def sample_starts(self, history: int = 0, newest: bool = False):
        """One stored state per env (a random time among its column's ``startable`` newest rows, or the newest
        row): (s, xi, side, past).
        Every stored s is a real simulator state, so any row can start an imagined rollout. With history > 0,
        past holds the ``history`` rows ending at the start (offset 0 = the start row itself, then one step
        back per row): ``s`` (H, N, D), ``side`` {k: (H, N, ...)}, ``valid`` (H, N): the row belongs to the same
        episode as the start (no reset or teleport in between), so observation histories can be rebuilt."""
        env = torch.arange(self.N, device=self.device)
        lo = self.size - self.startable
        t = torch.full((self.N,), self.size - 1, device=self.device) if newest else torch.randint(lo, self.size, (self.N,), device=self.device)
        rows = (self.ptr - self.size + t) % self.T
        side = {k[5:]: v[rows, env] for k, v in self.data.items() if k.startswith("side/")}
        past = None
        if history > 0:
            tj = t[None, :] - torch.arange(history, device=self.device)[:, None]  # (H, N), offset j rows back
            exists = tj >= 0
            rows_j = (self.ptr - self.size + tj.clamp(min=0)) % self.T
            e = env[None, :].expand_as(rows_j)
            # row t-j is in the start's episode iff rows t-1 ... t-j each continue into the next one
            # (cont of row t-i says row t-i continues into t-i+1; the start row's own cont is irrelevant)
            cont_j = self.data["cont"][rows_j, e] & exists
            chain = torch.cat((torch.ones_like(cont_j[:1]), cont_j[1:]), 0)
            past = {"s": self.data["s"][rows_j, e], "valid": torch.cumprod(chain.long(), 0).bool() & exists,
                    "side": {k[5:]: v[rows_j, e] for k, v in self.data.items() if k.startswith("side/")}}  # fmt: skip
        return self.data["s"][rows, env], self.data["xi"][rows, env], side, past

    def sample_windows(self, batch: int, k: int) -> dict[str, torch.Tensor]:
        """(batch, k, ...) windows and their mask: transition j counts if it is valid and every earlier
        transition of the window continued into the next."""
        t0 = torch.randint(0, self.size - k + 1, (batch,), device=self.device)
        env = torch.randint(0, self.N, (batch,), device=self.device)
        rows = (self.ptr - self.size + t0[:, None] + torch.arange(k, device=self.device)) % self.T
        e = env[:, None].expand_as(rows)
        out = {key: v[rows, e] for key, v in self.data.items() if not key.startswith("side/")}
        cont = out["cont"].float()
        reach = torch.cumprod(torch.cat((torch.ones(batch, 1, device=self.device), cont[:, :-1]), -1), -1)
        out["mask"] = reach * out["valid"].float()
        return out
