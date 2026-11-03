"""A phase cue computed exclusively from the 738D actor observation."""

from __future__ import annotations

import torch
from torch import nn


def supine_flat_recovery_label(
    scene: torch.Tensor, direction: torch.Tensor, escaped: torch.Tensor
) -> torch.Tensor:
    """Training label for a supine-only flat-recovery gate.

    Flat supine is eligible from motor activation; board worlds become
    eligible only after physical clearance. No simulator label is an input to
    the detector or runtime gate.
    """
    supine = direction == 0
    return supine & ((scene == 0) | (((scene == 1) | (scene == 2)) & escaped))


class ProprioPhaseDetector(nn.Module):
    def __init__(self, mean: torch.Tensor, scale: torch.Tensor):
        super().__init__()
        if mean.shape != (738,) or scale.shape != (738,):
            raise ValueError("the phase detector requires a 738D actor observation")
        self.register_buffer("mean", mean.float().clone())
        self.register_buffer("scale", scale.float().clamp_min(1e-3).clone())
        self.network = nn.Sequential(
            nn.Linear(738, 256), nn.SiLU(),
            nn.Linear(256, 128), nn.SiLU(),
            nn.Linear(128, 1),
        )

    def forward(self, actor_obs: torch.Tensor) -> torch.Tensor:
        if actor_obs.shape[-1] != 738:
            raise ValueError("phase detector received an observation other than 738D")
        normalized = ((actor_obs - self.mean) / self.scale).clamp(-10, 10)
        return self.network(normalized).squeeze(-1)


def load_phase_detector(path, device: str):
    state = torch.load(path, map_location=device, weights_only=False)
    if state.get("observation_contract") != "proprio_torque_actor_privileged_critic_v2":
        raise ValueError("phase detector checkpoint does not match the actor contract")
    model = ProprioPhaseDetector(state["mean"], state["scale"]).to(device)
    model.load_state_dict(state["model"])
    model.eval()
    return model, state
