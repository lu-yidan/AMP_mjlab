"""Export the actor for deployment (TorchScript / ONNX), as rsl_rl's exporter does."""

from __future__ import annotations

import copy
import os

import torch
import torch.nn as nn


class _Actor(nn.Module):
    def __init__(self, policy, key: str):
        super().__init__()
        self.policy, self.key = copy.deepcopy(policy).cpu().eval(), key

    def forward(self, x):
        return self.policy.act_inference({self.key: x})


def export_policy_as_jit(policy, path: str, num_obs: int, obs_key: str = "policy"):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    m = torch.jit.trace(_Actor(policy, obs_key), torch.zeros(1, num_obs))
    m.save(path)


def export_policy_as_onnx(policy, path: str, num_obs: int, obs_key: str = "policy", opset: int = 17):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    torch.onnx.export(_Actor(policy, obs_key), torch.zeros(1, num_obs), path, input_names=["obs"],
                      output_names=["actions"], dynamic_axes={"obs": {0: "batch"}, "actions": {0: "batch"}}, opset_version=opset)  # fmt: skip
