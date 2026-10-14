"""Expand the trained 23-DoF HoST actor to 29 DoF by joint name."""

from __future__ import annotations

from copy import deepcopy

import torch

from .action import PHYSICS_REVISION
from .model import HOST29_JOINT_NAMES, HOST_JOINT_NAMES

OLD_FRAME, NEW_FRAME, HISTORY = 76, 94, 6


def observation_columns() -> tuple[torch.Tensor, torch.Tensor]:
    """Columns for common proprioception in all six HoST history frames."""
    old, new = [], []
    joint = [HOST29_JOINT_NAMES.index(name) for name in HOST_JOINT_NAMES]
    for frame in range(HISTORY):
        a, b = frame * OLD_FRAME, frame * NEW_FRAME
        old += list(range(a, a + 6))
        new += list(range(b, b + 6))
        for old_start, new_start in ((6, 6), (29, 35), (52, 64)):
            old += list(range(a + old_start, a + old_start + 23))
            new += [b + new_start + i for i in joint]
        old.append(a + 75)
        new.append(b + 93)
    return torch.tensor(old), torch.tensor(new)


@torch.no_grad()
def transfer_actor_23_to_29(policy, checkpoint: dict) -> None:
    """Keep shared-joint actor outputs exact for mapped observations.

    The four critics and optimizer are deliberately left fresh because A6
    changes reward targets. New joint output rows start at zero; exploration
    can learn them from the 29-DoF environment.
    """
    if checkpoint.get("num_envs") is None or checkpoint.get("model_source") not in (
        "host_urdf_capsule",
        "host_urdf",
        "mjlab",
    ):
        raise ValueError("Expected a native HoST 23-DoF training checkpoint")
    parent = checkpoint["policy"]
    if tuple(parent["actor.0.weight"].shape) != (512, 456):
        raise ValueError("Parent actor observation size is not HoST 23-DoF 456")
    if tuple(parent["actor.6.weight"].shape) != (23, 128):
        raise ValueError("Parent actor action size is not 23")
    if tuple(policy.actor[0].weight.shape) != (512, 564) or tuple(
        policy.actor[6].weight.shape
    ) != (29, 128):
        raise ValueError("Target must be a 564-to-29 HoST policy")
    old_cols, new_cols = observation_columns()
    policy.actor[0].weight.zero_()
    policy.actor[0].weight[:, new_cols] = parent["actor.0.weight"][:, old_cols]
    policy.actor[0].bias.copy_(parent["actor.0.bias"])
    for index in (2, 4):
        policy.actor[index].weight.copy_(parent[f"actor.{index}.weight"])
        policy.actor[index].bias.copy_(parent[f"actor.{index}.bias"])
    policy.actor[6].weight.zero_()
    policy.actor[6].bias.zero_()
    new_joint = [HOST29_JOINT_NAMES.index(name) for name in HOST_JOINT_NAMES]
    policy.actor[6].weight[new_joint] = parent["actor.6.weight"]
    policy.actor[6].bias[new_joint] = parent["actor.6.bias"]
    policy.log_std[new_joint] = parent["log_std"]


def _copy_expanded(name: str, target: torch.Tensor, source: torch.Tensor) -> None:
    """Copy a 23-DoF parameter or Adam moment into the 29-DoF layout."""

    if target.shape == source.shape:
        target.copy_(source)
        return
    old_cols, new_cols = observation_columns()
    new_joint = [HOST29_JOINT_NAMES.index(name) for name in HOST_JOINT_NAMES]
    if name == "actor.0.weight" or (
        name.startswith("critics.") and name.endswith(".0.weight")
    ):
        if source.shape != (512, 456) or target.shape != (512, 564):
            raise ValueError(f"incompatible first-layer shape for {name}")
        target.zero_()
        target[:, new_cols] = source[:, old_cols]
    elif name == "actor.6.weight":
        if source.shape != (23, 128) or target.shape != (29, 128):
            raise ValueError("incompatible actor output weight shape")
        target.zero_()
        target[new_joint] = source
    elif name in ("actor.6.bias", "log_std"):
        if source.shape != (23,) or target.shape != (29,):
            raise ValueError(f"incompatible action vector shape for {name}")
        # Keep the target's initialization for the six new log standard
        # deviations; all new action-output weights and moments start at zero.
        if name != "log_std":
            target.zero_()
        target[new_joint] = source
    else:
        raise ValueError(
            f"cannot expand parameter {name}: {source.shape} -> {target.shape}"
        )


@torch.no_grad()
def continue_full_23_to_29(policy, optimizer, checkpoint: dict) -> None:
    """Continue the 23-DoF flat checkpoint into 29-DoF A6 as far as shapes allow.

    Actor, all four critics, exploration variance, and Adam moments/step are
    transferred. The six added axes have zero actor outputs and Adam moments;
    their exploration variance keeps the new policy's default initialization.
    """

    source_policy = checkpoint["policy"]
    target_policy = policy.state_dict()
    if (
        checkpoint.get("model_source")
        not in ("host_urdf_capsule", "host_urdf", "mjlab")
        or checkpoint.get("iteration") is None
        or checkpoint.get("curriculum") is None
        or tuple(source_policy["actor.0.weight"].shape) != (512, 456)
        or tuple(source_policy["actor.6.weight"].shape) != (23, 128)
        or list(source_policy) != list(target_policy)
    ):
        raise ValueError("expected a complete 23-DoF HoST flat checkpoint")
    for name, target in target_policy.items():
        _copy_expanded(name, target, source_policy[name])
    policy.load_state_dict(target_policy)

    source_optimizer = checkpoint["optimizer"]
    target_optimizer = optimizer.state_dict()
    source_groups = source_optimizer["param_groups"]
    target_groups = target_optimizer["param_groups"]
    names = [name for name, _ in policy.named_parameters()]
    if (
        len(source_groups) != 1
        or len(target_groups) != 1
        or len(source_groups[0]["params"]) != len(names)
        or len(target_groups[0]["params"]) != len(names)
    ):
        raise ValueError("HoST optimizer parameter order is incompatible")
    for key, value in source_groups[0].items():
        if key != "params":
            target_groups[0][key] = deepcopy(value)
    for name, source_id, target_id in zip(
        names,
        source_groups[0]["params"],
        target_groups[0]["params"],
        strict=True,
    ):
        mapped = {}
        for key, value in source_optimizer["state"].get(source_id, {}).items():
            if torch.is_tensor(value) and value.shape == source_policy[name].shape:
                expanded = torch.zeros_like(target_policy[name])
                _copy_expanded(name, expanded, value)
                mapped[key] = expanded
            else:
                mapped[key] = deepcopy(value)
        target_optimizer["state"][target_id] = mapped
    optimizer.load_state_dict(target_optimizer)


def continue_full_29_to_a6(
    policy, optimizer, checkpoint: dict, extra_obs_dim: int = 0
) -> None:
    """Carry the complete pose-v6 29-DoF PPO state into the A6 task.

    A6 changes the reward and contacts, but its 564 inputs, 29 outputs and
    policy parameter order are identical to the pose-v6 flat expert.  Do not
    silently discard the upper-body residual or Adam state as actor-only
    initialization would do.
    """
    if (
        checkpoint.get("task") != "flat29"
        or checkpoint.get("model_source") != "unitree_g1_29"
        or checkpoint.get("physics_revision")
        != f"{PHYSICS_REVISION}_flat29_29_v1"
        or checkpoint.get("reward_variant") != "height_guard_style_v2_29dof_pose_v6"
        or checkpoint.get("iteration") is None
        or checkpoint.get("curriculum") is None
    ):
        raise ValueError("expected a complete 29-DoF flat pose-v6 checkpoint")
    source = checkpoint["policy"]
    target = policy.state_dict()
    if source.keys() != target.keys() or extra_obs_dim not in (0, 48):
        raise ValueError("pose-v6 policy topology differs from the A6 policy")
    expanded_input_names = {
        "actor.0.weight",
        "upper_residual.0.weight",
        *(f"critics.{index}.0.weight" for index in range(4)),
    }
    for name, tensor in target.items():
        source_shape = source[name].shape
        if extra_obs_dim and name in expanded_input_names:
            if (
                len(source_shape) != 2
                or source_shape[1] != 564
                or tensor.shape != (source_shape[0], 564 + extra_obs_dim)
            ):
                raise ValueError(f"invalid expanded observation layer {name}")
        elif source_shape != tensor.shape:
            raise ValueError(f"pose-v6 parameter shape differs for {name}")
    if policy.upper_residual is None or policy.upper_residual_scale[:3].tolist() != [
        5.0,
        5.0,
        5.0,
    ]:
        raise ValueError("A6 policy needs the pose-v6 upper-body residual")
    if any(not torch.isfinite(tensor).all() for tensor in source.values()):
        raise ValueError("parent policy contains non-finite weights")
    source_optimizer = checkpoint["optimizer"]
    target_optimizer = optimizer.state_dict()
    source_groups = source_optimizer["param_groups"]
    target_groups = target_optimizer["param_groups"]
    names = [name for name, _ in policy.named_parameters()]
    if (
        len(source_groups) != 1
        or len(target_groups) != 1
        or len(source_groups[0]["params"]) != len(names)
        or len(target_groups[0]["params"]) != len(names)
    ):
        raise ValueError("pose-v6 optimizer parameter count differs from A6")
    for name, parameter_id in zip(names, source_groups[0]["params"], strict=True):
        state = source_optimizer["state"].get(parameter_id, {})
        if not state or any(
            key not in state or state[key].shape != source[name].shape
            for key in ("exp_avg", "exp_avg_sq")
        ):
            raise ValueError(f"missing or incompatible Adam moments for {name}")
        if any(not torch.isfinite(state[key]).all() for key in ("exp_avg", "exp_avg_sq")):
            raise ValueError(f"non-finite Adam moments for {name}")
    for name in ("beta", "assist_force"):
        value = checkpoint["curriculum"].get(name)
        if value is None or not torch.isfinite(value).all():
            raise ValueError(f"invalid parent curriculum {name}")
    if extra_obs_dim:
        # Preserve the complete flat policy and its Adam history; only newly
        # observed plate columns and their moments start at zero.
        expanded = {}
        copied_optimizer = deepcopy(source_optimizer)
        for name, tensor in target.items():
            if name in expanded_input_names:
                value = torch.zeros_like(tensor)
                value[:, :564] = source[name]
            else:
                value = source[name]
            expanded[name] = value
        for name, parameter_id in zip(
            names, source_groups[0]["params"], strict=True
        ):
            if name not in expanded_input_names:
                continue
            for key in ("exp_avg", "exp_avg_sq"):
                value = torch.zeros_like(expanded[name])
                value[:, :564] = source_optimizer["state"][parameter_id][key]
                copied_optimizer["state"][parameter_id][key] = value
        policy.load_state_dict(expanded, strict=True)
        optimizer.load_state_dict(copied_optimizer)
    else:
        policy.load_state_dict(source, strict=True)
        optimizer.load_state_dict(source_optimizer)
