"""Minimal 23-to-29 DoF HoST adapter without AMP or training assistance."""
from pathlib import Path

import numpy as np
import torch

from mjlab.envs.mdp.actions import JointPositionAction
from mjlab.managers.event_manager import EventTermCfg

from .flat29 import flat29_env_cfg
from .mdp.actions import IncrementalJointPositionActionCfg


ADDED_JOINTS = (
    "waist_roll_joint", "waist_pitch_joint",
    "left_wrist_pitch_joint", "left_wrist_yaw_joint",
    "right_wrist_pitch_joint", "right_wrist_yaw_joint",
)


class SharedIncrementAddedHomeAction(JointPositionAction):
    """Keep the 23 shared axes incremental and home-centre only new axes."""

    def __init__(self, cfg, env):
        super().__init__(cfg=cfg, env=env)
        self._offset = torch.zeros(self.num_envs, self.action_dim, device=self.device)
        self._added_ids = [
            i for i, name in enumerate(self._entity.joint_names) if name in ADDED_JOINTS
        ]
        assert self.action_dim == 29 and len(self._added_ids) == 6

    def process_actions(self, actions):
        self._raw_actions[:] = actions
        self._offset[:] = self._entity.data.joint_pos[:, self._target_ids]
        self._processed_actions = self._offset + self._raw_actions * self._scale
        # The source actor never controlled these six axes.  Zero output must
        # therefore mean a finite neutral target, not damping at a folded pose.
        self._processed_actions[:, self._added_ids] = (
            self._raw_actions[:, self._added_ids] * 0.25
        )

    def apply_actions(self):
        # Static deployment resets act from the first policy step; the original
        # HoST motor-off settling window is intentionally absent here.
        encoder_bias = self._entity.data.encoder_bias[:, self._target_ids]
        self._entity.set_joint_position_target(
            self._processed_actions - encoder_bias, joint_ids=self._target_ids
        )


class SharedIncrementAddedHomeActionCfg(IncrementalJointPositionActionCfg):
    def build(self, env):
        return SharedIncrementAddedHomeAction(self, env)


def static_bank_reset(env, env_ids=None, bank_path=None):
    """Reset from the verified native 29-DoF four-direction resting bank."""
    if not hasattr(env, "_minimal29_bank"):
        data = np.load(Path(bank_path), allow_pickle=False)
        qpos = data["qpos"].astype(np.float32)
        direction = data["direction"].astype(np.int64)
        assert qpos.shape[1] == 36 and len(qpos) == len(direction)
        env._minimal29_bank = torch.as_tensor(qpos, device=env.device)
        labels = torch.as_tensor(direction, device=env.device)
        env._minimal29_pools = [torch.nonzero(labels == i).flatten() for i in range(4)]
        assert all(len(pool) > 0 for pool in env._minimal29_pools)
        env._minimal29_direction = torch.arange(env.num_envs, device=env.device) % 4
        env._minimal29_rng = torch.Generator(device=env.device).manual_seed(2026100501)
    ids = (torch.arange(env.num_envs, device=env.device)
           if env_ids is None else env_ids)
    if len(ids) == 0:
        return
    family = env._minimal29_direction[ids]
    draw = torch.empty(len(ids), dtype=torch.long, device=env.device)
    for i, pool in enumerate(env._minimal29_pools):
        mask = family == i
        draw[mask] = pool[torch.randint(
            len(pool), (int(mask.sum()),), generator=env._minimal29_rng,
            device=env.device)]
    qpos = env._minimal29_bank[draw]
    robot = env.scene["robot"]
    root = robot.data.default_root_state[ids].clone()
    root[:, :3] = qpos[:, :3] + env.scene.env_origins[ids]
    root[:, 3:7] = qpos[:, 3:7]
    root[:, 7:] = 0
    robot.write_root_state_to_sim(root, env_ids=ids)
    robot.write_joint_state_to_sim(
        qpos[:, 7:], torch.zeros_like(qpos[:, 7:]), env_ids=ids)
    robot.set_joint_position_target(qpos[:, 7:], env_ids=ids)
    env.sim.forward()


def minimal29_env_cfg(num_envs, seed, bank_path):
    cfg = flat29_env_cfg(play=False)
    cfg.scene.num_envs = num_envs
    cfg.seed = seed
    cfg.episode_length_s = 20.0
    cfg.events.pop("reset_base", None)
    cfg.events.pop("reset_joints", None)
    cfg.events.pop("init_pull_force", None)
    cfg.events.pop("apply_pull_force", None)
    cfg.events["minimal29_static_reset"] = EventTermCfg(
        func=static_bank_reset, mode="reset", params={"bank_path": str(bank_path)})
    cfg.curriculum = {}
    old = cfg.actions["joint_pos"]
    cfg.actions["joint_pos"] = SharedIncrementAddedHomeActionCfg(**old.__dict__)
    cfg.actions["joint_pos"].scale = 0.25
    return cfg
