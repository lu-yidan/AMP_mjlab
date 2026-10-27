"""SuperMimic: fit a world model on real transitions, then train the policy by backpropagating the
return through imagined rollouts of it (the role PPO plays in rsl_rl).

Per iteration the runner calls:
  update_world_model()   k-step windows from the replay, multi-step frame-local state loss (+ aux head);
  update_policy(task)    BPTT through ``horizon`` imagined steps from replay states (with their recorded DR
                         context), evaluating the task's reward, termination and observations on the imagined
                         states, with the critic as terminal value; then the critic is fit to TD(lambda) targets
                         of the imagined trajectory.

The defaults are the paper's (arXiv 2610.28480, sec. 3 and table A.5): soft survival weight, H=32, gamma 0.99,
lambda 0.95, Adam 2e-4 (actor betas 0.9 / 0.95) with cosine decay to 0.03x over 250k iterations, gradient clip 1
(actor: then elementwise 0.1), world model 10 x 2048 windows of k=8 per update, replay 1000 steps, critic 5 epochs
x 4 shuffled minibatches, TF32. The paper does not state a world-model warm-up or an action clip; those defaults
are this library's. Optional guards not in the paper (off by default): the wild-state filter and the
critic-target clip. Non-finite imagined states and non-finite updates are always dropped.

Speed (``compile=True``): imagination is thousands of small kernels per update, so what sets its time is the
number of launches and host syncs, not FLOPs. The world-model window loss is one compiled graph; with a
``CarryRollout`` (functional route) a whole imagined step -- observation, policy, world model, non-finite filter, reward,
termination, return bookkeeping -- is one compiled graph, forward and backward; the critic is fit on stacked
imagined observations; the optimizers are fused and skip non-finite steps on the device, so an update runs
without a host sync. Other rollouts (manager-based) run the same loop with their own observe / step.
"""

from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass

import torch
import torch.nn as nn

from ..env.tracking_task import CarryRollout
from ..storage.transition_replay import TransitionReplay
from ..utils.distributed import allreduce_grads, allreduce_list, sync_norms, world_size


@dataclass
class SuperMimicCfg:
    horizon: int = 32
    # where imagined rollouts start: "replay" (a stored state per env, as in the paper) or "live" (the envs
    # current states; also the fallback until the replay has data)
    imagine_from: str = "replay"
    # how termination enters the imagined return: "soft" (the paper's survival weight in both passes),
    # "straight_through" (forward: hard done, backward: the soft surrogate gradient), "hard" (no gradient)
    termination: str = "soft"
    gamma: float = 0.99
    lam: float = 0.95
    action_noise: float = 0.1  # std of the Gaussian exploration noise during collection
    action_clip: float | None = 20.0  # not in the paper
    learning_rate: float = 2e-4  # actor
    critic_learning_rate: float = 2e-4
    wm_learning_rate: float = 2e-4
    lr_schedule: str = "cosine"  # "cosine" (to lr_min_ratio over lr_decay_iters, then constant) | "constant"
    lr_decay_iters: int = 250_000
    lr_min_ratio: float = 0.03
    wm_warmup_iters: int = 300  # world-model-only iterations at the start (not stated in the paper)
    wm_window: int = 8
    wm_batch: int = 2048
    wm_updates: int = 10  # batches per iteration
    buffer_steps: int = 1000  # replay length in vectorized steps
    critic_epochs: int = 5
    critic_minibatches: int = 4  # shuffled over all imagined (step, env) samples
    max_grad_norm: float = 1.0
    max_grad_value: float = 0.1
    value_target_clip: float | None = None  # optional guard, not in the paper
    aux_loss_clip: float = 1.0  # only for policies with an aux_loss (not in the paper)
    # optional wild-state filter, not in the paper: imagined states beyond these end the rollout there
    wild_joint_vel: float | None = None  # rad/s
    wild_root_vel: float | None = None  # m/s
    wild_height: tuple[float, float] | None = None
    # precision and speed: amp=None runs the world model in TF32 as the paper does ("bf16": bf16 autocast for its
    # MLP); torch.compile of the update graphs (compile_mode: "default" | "reduce-overhead" also CUDA graphs)
    amp: str | None = None
    compile: bool = True
    compile_mode: str = "default"
    tf32: bool = True


def _set_grad(params, flag):
    for p in params:
        p.requires_grad_(flag)


def _detach(obs):
    return {k: v.detach() for k, v in obs.items()}


def _adam(params, lr, device, **kw):
    fused = torch.device(device).type == "cuda"
    return torch.optim.Adam(params, lr=lr, fused=fused, **kw)


def _step_if_finite(opt, ok):
    """opt.step() unless ``ok`` (a device bool) is False, without a host sync: fused Adam reads ``found_inf`` on the
    device (the GradScaler mechanism); otherwise fall back to a synced check."""
    if opt.defaults.get("fused"):
        opt.found_inf = (~ok).float()
        opt.step()
        del opt.found_inf
    elif bool(ok):
        opt.step()


class SuperMimic:
    def __init__(self, policy: nn.Module, world_model: nn.Module, cfg: SuperMimicCfg, step_dt: float, device="cpu"):
        self.policy, self.wm, self.cfg = policy, world_model, cfg
        self.step_dt, self.device = step_dt, device
        if cfg.tf32:
            torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = True
        if cfg.amp is not None and torch.device(device).type == "cuda":
            world_model.amp_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}[cfg.amp]
        self._wm_step = world_model.step  # raw: traced into the larger graphs
        if cfg.compile:
            world_model.step = torch.compile(world_model.step, dynamic=False)  # the generic (non-carry) loop
        self.actor_params = [p for p in policy.actor_parameters() if p.requires_grad]
        self.critic_params = [p for p in policy.critic_parameters() if p.requires_grad]
        self.lr_scale = math.sqrt(world_size())  # the paper's sqrt(K) learning rate on K GPUs
        self.opt_pi = _adam(self.actor_params, cfg.learning_rate * self.lr_scale, device, betas=(0.9, 0.95))
        self.opt_v = _adam(self.critic_params, cfg.critic_learning_rate * self.lr_scale, device)
        self.opt_wm = _adam(list(world_model.parameters()), cfg.wm_learning_rate * self.lr_scale, device)
        self.replay: TransitionReplay | None = None
        self._graphs: dict = {}
        self._disc = None

    def init_storage(self, num_envs, state_dim, num_actions, xi_dim, aux_dim):
        self.replay = TransitionReplay(self.cfg.buffer_steps, num_envs, state_dim, num_actions, xi_dim, aux_dim, self.device)

    def _compiled(self, key, fn):
        """fn compiled once per key (cfg.compile_mode); eager when compile is off."""
        if key not in self._graphs:
            c = self.cfg
            self._graphs[key] = torch.compile(fn, mode=c.compile_mode, dynamic=False) if c.compile else fn
        return self._graphs[key]

    def _keep(self, x):
        """Outputs kept past the next replay of a CUDA graph must be copied out of its memory pool."""
        if self.cfg.compile and self.cfg.compile_mode == "reduce-overhead":
            return torch.utils._pytree.tree_map(lambda t: t.clone() if torch.is_tensor(t) else t, x)
        return x

    @staticmethod
    def _mark_step():
        if hasattr(torch, "compiler") and hasattr(torch.compiler, "cudagraph_mark_step_begin"):
            torch.compiler.cudagraph_mark_step_begin()

    # ------------------------------------------------------------------ collection
    def _clip(self, a):
        c = self.cfg.action_clip
        return a if c is None else a.clamp(-c, c)

    @torch.no_grad()
    def act(self, obs):
        if hasattr(self.policy, "update_normalization"):
            self.policy.update_normalization(obs)
        a = self.policy.act_inference(obs)
        return self._clip(a + self.cfg.action_noise * torch.randn_like(a))

    @torch.no_grad()
    def process_env_step(self, layout, s, a, s1, xi, aux, dones, terminal_state=None, teleported=None, side=None):
        """Store one vectorized transition; s1 is the state after step() (post-reset for done envs); side is
        the task's side state at s."""
        done = dones.bool()
        valid = ~done
        if terminal_state is not None:
            s1 = torch.where(done[:, None], terminal_state, s1)
            valid = torch.ones_like(done)
        if teleported is not None:
            valid = valid & ~teleported.bool()
        self.replay.add(s, a, s1, xi, aux, valid, ~done, side)
        if valid.any():
            S0, S1 = layout.unflatten(s[valid]), layout.unflatten(s1[valid])
            self.wm.update_normalization(S0, a[valid], xi[valid], S1, None if aux is None else aux[valid])

    def sync_statistics(self):
        sync_norms(self.wm, self.policy)

    # ------------------------------------------------------------------ world model
    def _wm_loss_fn(self, layout):
        c, wm, step = self.cfg, self.wm, self._wm_step
        has_aux = wm.aux_dim > 0

        def loss(s, a, xi, s1, aux_t, mask):
            """k-step window loss: state error in the target's local frames over the running-norm std, masked."""
            S = layout.unflatten(s[:, 0])
            std = wm.out_norm.std()
            loss_s = loss_aux = torch.zeros((), device=s.device)
            for j in range(c.wm_window):
                S, aux = step(S, a[:, j], xi[:, j])
                err = ((wm.state_error(S, layout.unflatten(s1[:, j])) / std) ** 2).mean(-1)
                loss_s = loss_s + (err * mask[:, j]).sum() / c.wm_batch
                if has_aux:
                    err_aux = (((aux - aux_t[:, j]) / wm.aux_norm.std()) ** 2).mean(-1)
                    loss_aux = loss_aux + (err_aux * mask[:, j]).sum() / c.wm_batch
            last = (err.detach() * mask[:, -1]).sum() / mask[:, -1].sum().clamp(min=1)
            return loss_s, loss_aux, last

        return loss

    def update_world_model(self, layout) -> dict:
        c, wm = self.cfg, self.wm
        if len(self.replay) <= c.wm_window:
            return {}
        loss_fn = self._compiled(("wm", id(layout)), self._wm_loss_fn(layout))
        params = list(wm.parameters())
        for _ in range(c.wm_updates):
            w = self.replay.sample_windows(c.wm_batch, c.wm_window)
            self._mark_step()
            loss_s, loss_aux, last = loss_fn(w["s"], w["a"], w["xi"], w["s1"], w["aux"], w["mask"])
            self.opt_wm.zero_grad(set_to_none=True)
            (loss_s + loss_aux).backward()
            allreduce_grads(params)
            nn.utils.clip_grad_norm_(params, c.max_grad_norm)
            self.opt_wm.step()
        out = {"wm/loss": loss_s.detach() / c.wm_window, "wm/err_last": last}
        if wm.aux_dim:
            out["wm/aux_loss"] = loss_aux.detach() / c.wm_window
        return self._keep(out)

    # ------------------------------------------------------------------ policy (BPTT) + critic
    def _value(self, obs):
        return self.policy.evaluate(obs).reshape(-1)

    def _critic_keys(self, obs):
        keys = getattr(self.policy, "critic_obs_keys", None)
        return tuple(keys) if keys else tuple(obs)

    def _wild(self, S1, aux):
        c = self.cfg
        p, _, v, _, _, qd = S1
        parts = [x.flatten(1) for x in S1] + ([aux] if aux is not None else [])
        wild = ~torch.isfinite(torch.cat(parts, -1)).all(-1)
        if c.wild_joint_vel is not None:
            wild |= qd.abs().amax(-1) > c.wild_joint_vel
        if c.wild_root_vel is not None:
            wild |= v.norm(dim=-1) > c.wild_root_vel
        if c.wild_height is not None:
            wild |= (p[:, 2] < c.wild_height[0]) | (p[:, 2] > c.wild_height[1])
        return wild

    def _step_body(self, observe, transition, keys):
        """One imagined step as a function of tensors (the unit that is compiled for carry rollouts):
        (carry, S, xi, alive, J, disc) -> carry1, S1, alive1, J1, and what the critic and the logs need."""
        c, pol, wm_step = self.cfg, self.policy, self._wm_step
        if c.termination not in ("straight_through", "soft", "hard"):
            raise ValueError(f"termination={c.termination!r}")

        def body(carry, S, xi, alive, J, disc):
            obs = observe(carry, S)
            act = self._clip(pol.act_inference(obs))
            S1, aux = wm_step(S, act, xi)
            wild = self._wild(S1, aux)
            S1 = tuple(torch.where(wild.view(-1, *[1] * (x.dim() - 1)), x0.detach(), x) for x, x0 in zip(S1, S))
            if aux is not None:
                aux = torch.where(wild[:, None], torch.zeros_like(aux), aux)
            carry1, (r, done, done_soft, terms) = transition(carry, S, act, S1, aux)
            r = torch.where(wild, torch.zeros_like(r), r)
            d = done.float()
            if done_soft is not None and c.termination == "straight_through":
                d = d + done_soft - done_soft.detach()
            elif done_soft is not None and c.termination == "soft":
                d = done_soft
            d = torch.where(wild, torch.ones_like(d), d)
            # critic samples: under the soft survival weight every imagined step (the paper's Eq. 3), else the live ones
            live = ~wild if (done_soft is not None and c.termination == "soft") else alive > 1e-6
            stats = {n: (alive * torch.where(wild, 0.0, v)).detach().mean() for n, v in terms.items()}
            stats["_wild"] = (wild & (alive > 0)).float().mean()
            stats["_r"] = (alive * r).detach()
            rec = ({k: obs[k].detach() for k in keys}, r.detach(), d.detach(), live.detach())
            return carry1, S1, alive * (1 - d), J + disc * alive * r, rec, stats

        return body

    def _final_body(self, observe, keys):
        def final(carry, S):
            obs = observe(carry, S)
            return self._value(obs), {k: obs[k].detach() for k in keys}

        return final

    def update_policy(self, task, obs0) -> dict:
        c, wm, pol = self.cfg, self.wm, self.policy
        t0 = time.time()
        self._mark_step()
        if c.imagine_from == "replay" and self.replay is not None and self.replay.startable > 0:
            s0, xi, side, past = self.replay.sample_starts(getattr(task, "history_length", 0))
            if xi.shape[-1] > 0:
                side = {**side, "xi": xi}
            rollout = task.imagine(side, past)
            S = task.layout.unflatten(s0)
        elif c.imagine_from in ("replay", "live"):
            rollout = task.imagine()
            S = task.layout.unflatten(task.get_state())
            xi = task.get_context()
        else:
            raise ValueError(f"imagine_from={c.imagine_from!r}")
        N, dev = S[0].shape[0], S[0].device
        keys = self._critic_keys(obs0)
        if isinstance(rollout, CarryRollout):
            cls = type(rollout)
            key = ("carry", cls, id(task), keys)
            if key not in self._graphs:
                obs_f = lambda carry, S: cls.observe_fn(task, carry, S)  # noqa: E731
                tr_f = lambda carry, S, a, S1, aux: cls.step_fn(task, carry, S, a, S1, aux)  # noqa: E731
                self._compiled(key + ("begin",), lambda carry, S: cls.begin(task, carry, S))
                self._compiled(key + ("final",), self._final_body(obs_f, keys))
                self._compiled(key + ("obs",), obs_f)
                self._compiled(key, self._step_body(obs_f, tr_f, keys))
            body, final = self._graphs[key], self._graphs[key + ("final",)]
            carry = self._graphs[key + ("begin",)](rollout.carry, S)
            observe0 = self._graphs[key + ("obs",)]
        else:  # an opaque rollout (manager-based): same loop, its own observe / step
            def observe(_, S):
                return rollout.observe(S)

            def transition(_, S, a, S1, aux):
                out = rollout.step(S, a, S1, aux)
                return None, (out.reward, out.done, out.done_soft, out.terms)

            body, final, carry = self._step_body(observe, transition, keys), self._final_body(observe, keys), None
            observe0 = observe
        if self._disc is None or self._disc.shape[0] != c.horizon + 1:
            self._disc = c.gamma ** torch.arange(c.horizon + 1, device=dev, dtype=torch.float32)
        _set_grad(wm.parameters(), False)
        _set_grad(self.critic_params, False)
        aux_loss = pol.aux_loss(observe0(carry, S)) if hasattr(pol, "aux_loss") else None
        alive, J = torch.ones(N, device=dev), torch.zeros(N, device=dev)
        trace, stats = [], []
        for k in range(c.horizon):
            carry, S, alive, J, rec, st = body(carry, S, xi, alive, J, self._disc[k])
            trace.append(rec)
            stats.append(st)
        v_last, obs_last = final(carry, S)
        J = J + self._disc[c.horizon] * alive * v_last
        # rewards are per step (dt-scaled, as Isaac Lab / mjlab return them): report and clip per second
        loss = -J.mean() / self.step_dt
        t_fwd = time.time()
        self.opt_pi.zero_grad(set_to_none=True)
        g_aux, aux_gn = None, torch.zeros((), device=dev)
        if aux_loss is not None and aux_loss.requires_grad:
            g_aux = torch.autograd.grad(aux_loss, self.actor_params, retain_graph=True, allow_unused=True)
            g_aux = [torch.zeros_like(p) if g is None else g for g, p in zip(g_aux, self.actor_params)]
        loss.backward()
        t_bwd = time.time()
        allreduce_grads(self.actor_params)
        gn = nn.utils.clip_grad_norm_(self.actor_params, c.max_grad_norm)
        nn.utils.clip_grad_value_(self.actor_params, c.max_grad_value)
        if g_aux is not None:
            allreduce_list(g_aux)
            aux_gn = torch.linalg.vector_norm(torch.stack([torch.linalg.vector_norm(g) for g in g_aux]))
            scale = torch.where(torch.isfinite(aux_gn), (c.aux_loss_clip / (aux_gn + 1e-6)).clamp(max=1.0), 0.0)
            for p, g in zip(self.actor_params, g_aux):
                p.grad = g * scale if p.grad is None else p.grad.add_(g * scale)
        _step_if_finite(self.opt_pi, torch.isfinite(gn))
        _set_grad(wm.parameters(), True)
        _set_grad(self.critic_params, True)

        # everything read from the step graphs is copied out (stacked) before the critic graphs run
        X = {k: torch.stack([o[k] for o, _, _, _ in trace] + [obs_last[k]]) for k in keys}  # (H+1, N, ...)
        rews, dones = torch.stack([t[1] for t in trace]), torch.stack([t[2] for t in trace])
        live = torch.stack([t[3] for t in trace]).float()
        st = {n: torch.stack([s[n] for s in stats]) for n in stats[0]}
        out = {"policy/loss": loss.detach(), "policy/imag_reward": (st.pop("_r").sum(0) / c.horizon / self.step_dt).mean(),
               "policy/imag_survival": alive.detach().mean(), "policy/grad_norm": gn.detach().clone(),
               "policy/wild_frac": st.pop("_wild").mean()}  # fmt: skip
        out.update({f"imag/{n}": v.mean() / self.step_dt for n, v in st.items()})
        if aux_loss is not None:
            out.update({"policy/aux_loss": aux_loss.detach().clone(), "policy/aux_grad_norm": aux_gn})
        del trace, stats, carry, S, J, alive, rec, st
        # critic: TD(lambda) over the imagined transitions, fit on the stacked imagined observations
        v_loss, V0 = self._fit_critic(X, rews, dones, live, N)
        out.update({"critic/loss": v_loss, "critic/v0": V0, "time/imagine_fwd": t_fwd - t0, "time/imagine_bwd": t_bwd - t_fwd,
                    "time/critic": time.time() - t_bwd})  # fmt: skip
        return self._keep(out)

    def _critic_fns(self, keys, H, N):
        c = self.cfg

        def values(X):
            return self._value({k: v.flatten(0, 1) for k, v in X.items()}).view(H + 1, N)

        def targets(rews, dones, V):
            R, out = V[-1], []
            for t in range(H - 1, -1, -1):
                R = rews[t] + c.gamma * (1 - dones[t]) * ((1 - c.lam) * V[t + 1] + c.lam * R)
                out.append(R)
            T = torch.stack(out[::-1])
            return T if c.value_target_clip is None else T.clamp(-c.value_target_clip, c.value_target_clip)

        def loss(X, T, L, idx):
            """Masked squared error over a minibatch of flat (step, env) indices into the first H steps."""
            pred = self._value({k: v[:H].flatten(0, 1)[idx] for k, v in X.items()})
            Lm = L.flatten()[idx]
            return (((pred - T.flatten()[idx]) ** 2) * Lm).sum() / Lm.sum().clamp(min=1)

        return values, targets, loss

    def _fit_critic(self, X, rews, dones, live, N):
        c, H = self.cfg, self.cfg.horizon
        key = ("critic", tuple(X), H, N)
        if key not in self._graphs:
            values, targets, loss = self._critic_fns(tuple(X), H, N)
            self._compiled(key + ("values",), values)
            self._compiled(key + ("targets",), targets)
            self._compiled(key, loss)
        with torch.no_grad():
            self._mark_step()
            V = self._graphs[key + ("values",)](X)
            T = self._graphs[key + ("targets",)](rews, dones, V)
            V0 = V[0].mean().clone()
        T = self._keep(T)
        loss_fn, v_loss = self._graphs[key], torch.zeros((), device=V.device)
        for _ in range(c.critic_epochs):
            for group in torch.randperm(H * N, device=V.device).chunk(c.critic_minibatches):
                self._mark_step()
                v_loss = loss_fn(X, T, live, group)
                self.opt_v.zero_grad(set_to_none=True)
                v_loss.backward()
                allreduce_grads(self.critic_params)
                nn.utils.clip_grad_norm_(self.critic_params, c.max_grad_norm)
                _step_if_finite(self.opt_v, torch.isfinite(v_loss))
        return v_loss.detach(), V0

    # ------------------------------------------------------------------ schedule and state
    def set_learning_rate(self, it: int):
        c = self.cfg
        if c.lr_schedule == "constant":
            return
        f = c.lr_min_ratio if it >= c.lr_decay_iters else c.lr_min_ratio + (1 - c.lr_min_ratio) * 0.5 * (1 + math.cos(math.pi * it / c.lr_decay_iters))
        for opt, lr in ((self.opt_pi, c.learning_rate), (self.opt_v, c.critic_learning_rate), (self.opt_wm, c.wm_learning_rate)):
            for g in opt.param_groups:
                g["lr"] = lr * f * self.lr_scale

    def state_dict(self) -> dict:
        return {"policy": self.policy.state_dict(), "world_model": self.wm.state_dict(), "opt_pi": self.opt_pi.state_dict(),
                "opt_v": self.opt_v.state_dict(), "opt_wm": self.opt_wm.state_dict(), "cfg": asdict(self.cfg)}  # fmt: skip

    def load_state_dict(self, d: dict, load_optimizer: bool = True, load_world_model: bool = True):
        self.policy.load_state_dict(d["policy"])
        if load_world_model and "world_model" in d:
            self.wm.load_state_dict(d["world_model"])
            if load_optimizer and "opt_wm" in d:
                self.opt_wm.load_state_dict(d["opt_wm"])
        if load_optimizer:
            self.opt_pi.load_state_dict(d["opt_pi"])
            self.opt_v.load_state_dict(d["opt_v"])
