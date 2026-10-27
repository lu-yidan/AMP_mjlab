"""Data-parallel helpers: each rank simulates its own envs; gradients and running statistics are averaged
so every replica stays identical. All are no-ops without an initialized multi-rank process group."""

from __future__ import annotations

import torch
import torch.distributed as dist

from ..modules.normalizer import RunningNorm


def is_multi() -> bool:
    return dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1


def world_size() -> int:
    return dist.get_world_size() if is_multi() else 1


def is_main() -> bool:
    return not is_multi() or dist.get_rank() == 0


def allreduce_list(tensors):
    """Average a list of tensors in place with one flat all-reduce."""
    tensors = [t for t in tensors if t is not None]
    if not is_multi() or not tensors:
        return
    flat = torch.cat([t.reshape(-1) for t in tensors])
    dist.all_reduce(flat)
    flat /= dist.get_world_size()
    i = 0
    for t in tensors:
        t.copy_(flat[i:i + t.numel()].view_as(t))
        i += t.numel()


def allreduce_grads(params):
    allreduce_list([p.grad for p in params])


def broadcast_module(m: torch.nn.Module):
    if not is_multi():
        return
    for t in list(m.parameters()) + list(m.buffers()):
        dist.broadcast(t.data, 0)


@torch.no_grad()
def sync_norms(*modules):
    """Merge RunningNorm statistics (exact when every rank saw the same sample count)."""
    if not is_multi():
        return
    w = dist.get_world_size()
    for mod in modules:
        for norm in mod.modules():
            if not isinstance(norm, RunningNorm):
                continue
            stats = torch.cat((norm.mean, norm.var + norm.mean**2, norm.count.reshape(1)))
            dist.all_reduce(stats)
            stats /= w
            d = norm.mean.numel()
            norm.mean.copy_(stats[:d])
            norm.var.copy_((stats[d:2 * d] - norm.mean**2).clamp_min(0))
            norm.count.copy_(stats[2 * d])
