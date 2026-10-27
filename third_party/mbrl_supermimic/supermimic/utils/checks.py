"""Adapter gates: imagined reward, termination and observations must equal the simulator's on real states
(``verify_task``), and their gradients must be right (``verify_gradients``).

SuperMimic only trains well if ``task.imagine()`` evaluates exactly what ``task.step()`` rewards, and BPTT
only works if the gradients through those evaluations are correct: a detach or a non-differentiable op
inside a term leaves the values intact and the policy blind. Run both on a new adapter or a hand-written
MDP (with observation noise off) before training; value errors should be at float precision.
"""

from __future__ import annotations

import torch


@torch.no_grad()
def verify_task(task, steps: int = 20, action_scale: float = 0.3, seed: int = 0) -> dict[str, float]:
    """Step the real task with random actions and replay each transition through ``task.imagine()`` on the
    true next state. Returns the max abs errors over envs that did not reset or teleport in this step or the
    previous one, and the fraction of them the imagined termination ends (all should be ~0);
    ``obs0_after_reset`` is the observation error on envs reset in the previous step, which is informational
    (see the comment in the loop)."""
    g = torch.Generator(device=task.device).manual_seed(seed)
    err = {"reward": 0.0, "obs0": 0.0, "obs1": 0.0, "done_mismatch_frac": 0.0, "compared_envs": 0.0, "obs0_after_reset": 0.0}
    mismatches = 0.0
    layout = task.layout
    # steps since a reset; the start counts as one: a command may fill its anchored reference only at its first
    # update (BeyondMimic's body_pos_relative_w is zeros until then), so the first reward is not gated either
    since_reset = torch.zeros(task.num_envs, dtype=torch.long, device=task.device)
    for _ in range(steps):
        obs = task.get_observations()
        S0 = layout.unflatten(task.get_state())
        rollout = task.imagine()
        fresh = since_reset < 1
        for k, v in rollout.observe(S0).items():
            if k in obs:
                e = (v - obs[k]).abs().max(-1).values
                # the observation an env gets right after its state was written inside step() (a reset, or a
                # motion resampled at its end) is the framework's business (Isaac Lab computes it before the
                # written state is visible to its lazy data buffers), and so is the reference it anchors for
                # the next reward: both are reported apart, not gated
                err["obs0"] = max(err["obs0"], e[~fresh].max().item() if (~fresh).any() else 0.0)
                err["obs0_after_reset"] = max(err["obs0_after_reset"], e[fresh].max().item() if fresh.any() else 0.0)
        a = action_scale * torch.randn(task.num_envs, task.num_actions, device=task.device, generator=g)
        obs1, r, dones, extras = task.step(a)
        S1 = layout.unflatten(task.get_state())
        out = rollout.step(S0, a, S1, task.get_wm_aux())
        live = ~dones.bool() & ~fresh
        if "teleported" in extras:
            live &= ~extras["teleported"].bool()
        moved = dones.bool() | (extras["teleported"].bool() if "teleported" in extras else False)
        since_reset = torch.where(moved, torch.zeros_like(since_reset), since_reset + 1)
        if not live.any():
            continue
        err["compared_envs"] += live.sum().item()
        r = r - extras.get("reward_not_imagined", 0.0)
        err["reward"] = max(err["reward"], (out.reward - r)[live].abs().max().item())
        mismatches += out.done[live].float().sum().item()
        for n, v in extras.get("reward_terms", {}).items():
            if n in out.terms:
                e = (out.terms[n] - v)[live].abs().max().item()
                err[f"reward/{n}"] = max(err.get(f"reward/{n}", 0.0), e)
        for k, v in rollout.observe(S1).items():
            if k in obs1:
                err["obs1"] = max(err["obs1"], (v - obs1[k])[live].abs().max().item())
    err["done_mismatch_frac"] = mismatches / max(err["compared_envs"], 1)
    return err


def verify_gradients(task, eps: float = 1e-3, action_scale: float = 0.3, seed: int = 0) -> dict[str, float]:
    """Autograd against central finite differences, with respect to the next state, of every imagined reward
    term, the soft termination and the observations (summed over envs). Probes the root position (x, z), a
    root quaternion component, and two joints. Returns, per quantity, the relative error at the 95th
    percentile over envs, max over probes (float precision expected; ~1e-2 on the soft termination, whose
    sigmoid is steep; the percentile keeps the few envs whose probe crosses a kink such as a clipped joint
    limit from masking the rest, where a real defect shows on every env), its plain max (``<name>/max``) and
    whether it has a gradient at all (``<name>/has_grad``: 0 means detached or constant)."""
    g = torch.Generator(device=task.device).manual_seed(seed)
    layout = task.layout
    nj = layout.num_joints
    probes = {"root_x": 0, "root_z": 2, "quat_x": 4, "joint_mid": 13 + nj // 2, "joint_last": 13 + nj - 1}
    S0 = layout.unflatten(task.get_state())
    a = action_scale * torch.randn(task.num_envs, task.num_actions, device=task.device, generator=g)
    task.step(a)
    s1 = task.get_state()
    aux = task.get_wm_aux()

    def evaluate(s1_flat):
        rollout = task.imagine()
        out = rollout.step(S0, a, layout.unflatten(s1_flat), aux)
        vals = {f"reward/{n}": v for n, v in out.terms.items()}
        vals["reward"] = out.reward
        if out.done_soft is not None:
            vals["done_soft"] = out.done_soft
        for k, v in rollout.observe(layout.unflatten(s1_flat)).items():
            vals[f"obs/{k}"] = v.sum(-1)
        return vals

    with torch.enable_grad():
        s1v = s1.clone().requires_grad_(True)
        vals = evaluate(s1v)
        grads = {}
        for name, v in vals.items():
            if torch.is_tensor(v) and v.requires_grad:
                grads[name] = torch.autograd.grad(v.sum(), s1v, retain_graph=True)[0]
            else:
                grads[name] = None
    out = {}
    with torch.no_grad():
        fd = {name: {} for name in vals}
        for pname, idx in probes.items():
            sp, sm = s1.clone(), s1.clone()
            sp[:, idx] += eps
            sm[:, idx] -= eps
            vp, vm = evaluate(sp), evaluate(sm)
            for name in vals:
                fd[name][pname] = (vp[name].float() - vm[name].float()) / (2 * eps)
        for name in vals:
            moves = max(d.abs().max().item() for d in fd[name].values())
            if grads[name] is None:
                # no autograd path: fine for a quantity that does not depend on the state (FD flat), a bug otherwise
                out[name], out[f"{name}/has_grad"] = moves, float(moves < 1e-6)
                continue
            # one scale per quantity, so a probe it does not depend on (FD noise ~1e-9) cannot dominate
            scale = max(moves, max(grads[name][:, i].abs().max().item() for i in probes.values()), 1e-6)
            rel, rel_max, worst = 0.0, 0.0, None
            for pname, idx in probes.items():
                e = (fd[name][pname] - grads[name][:, idx]).abs() / scale
                q, m = torch.quantile(e, 0.95).item(), e.max().item()
                rel, worst = (q, pname) if q > rel else (rel, worst)
                rel_max = max(rel_max, m)
            out[name], out[f"{name}/max"], out[f"{name}/has_grad"] = rel, rel_max, 1.0
            if rel > 1e-2:
                out[f"{name}/worst_probe"] = worst
    return out
