"""HoST's relative joint targets and get-up assistance curriculum."""

from dataclasses import dataclass

import torch
from mjlab.asset_zoo.robots.unitree_g1.g1_constants import (
    DAMPING_4010,
    DAMPING_5020,
    STIFFNESS_4010,
    STIFFNESS_5020,
)
from mjlab.envs.mdp.actions import (
    RelativeJointPositionAction,
    RelativeJointPositionActionCfg,
)

PHYSICS_REVISION = "split_torso_force_v2"


@dataclass(kw_only=True)
class HostActionCfg(RelativeJointPositionActionCfg):
    unactuated_steps: int = 30
    initial_assist_force: float = 200.0
    guided_arm_scale: float = 1.0
    guided_prone_arm_scale: float | None = None
    guided_supine_arm_scale: float | None = None
    guided_hip_scale: float = 1.0
    guided_lateral_scale: float = 1.0
    guided_lift_guard: bool = False

    def build(self, env):
        return HostAction(self, env)


class HostAction(RelativeJointPositionAction):
    """Clipped action: q_target = q_now + beta * action, after 30 steps."""

    cfg: HostActionCfg

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        if not (
            0 < cfg.guided_arm_scale <= 1
            and 0 < cfg.guided_hip_scale <= 1
            and 0 < cfg.guided_lateral_scale <= 1
            and (
                cfg.guided_prone_arm_scale is None
                or 0 < cfg.guided_prone_arm_scale <= 1
            )
            and (
                cfg.guided_supine_arm_scale is None
                or 0 < cfg.guided_supine_arm_scale <= 1
            )
        ):
            raise ValueError("guided action scales must be in (0, 1]")
        self._guided_arm_ids = [
            i
            for i, name in enumerate(self._target_names)
            if any(part in name for part in ("shoulder", "elbow", "wrist"))
        ]
        self._guided_hip_ids = [
            i for i, name in enumerate(self._target_names) if "hip" in name
        ]
        self._guided_lateral_arm_ids = [
            i
            for i, name in enumerate(self._target_names)
            if "shoulder_roll" in name or "shoulder_yaw" in name
        ]
        self._guided_lateral_hip_ids = [
            i
            for i, name in enumerate(self._target_names)
            if "hip_roll" in name or "hip_yaw" in name
        ]
        self._guided_nonlateral_arm_ids = [
            i for i in self._guided_arm_ids if i not in self._guided_lateral_arm_ids
        ]
        gains = []
        for name in self._target_names:
            if "wrist_pitch" in name or "wrist_yaw" in name:
                # Unitree 4010 wrist axes: 5 Nm, substantially weaker than
                # the 25 Nm wrist-roll motor.  Use a matching low PD gain.
                gains.append((STIFFNESS_4010, DAMPING_4010, 5.0))
            elif "waist_roll" in name or "waist_pitch" in name:
                gains.append((STIFFNESS_5020 * 2, DAMPING_5020 * 2, 50.0))
            elif "hip_" in name:
                gains.append((150.0, 4.0, 88.0))
            elif "knee" in name:
                gains.append((200.0, 6.0, 139.0))
            elif "ankle" in name:
                gains.append((40.0, 2.0, 50.0))
            elif "waist" in name:
                gains.append((100.0, 4.0, 88.0))
            else:
                gains.append((100.0, 4.0, 25.0))
        gain_tensor = torch.tensor(gains, device=self.device)
        self._kp = gain_tensor[:, 0]
        self._kd = gain_tensor[:, 1]
        self._effort_limit = gain_tensor[:, 2]
        self.beta = torch.ones(self.num_envs, 1, device=self.device)
        self.assist_force = torch.full(
            (self.num_envs, 1), cfg.initial_assist_force, device=self.device
        )
        # Optional horizontal feasibility/training aid. Zero in every ordinary
        # run and replay; it is applied only in guided, still-obstructed worlds.
        self.guided_escape_assist_force = torch.zeros(
            self.num_envs, 1, device=self.device
        )
        self.guided_escape_assist_speed_target = torch.zeros(
            self.num_envs, 1, device=self.device
        )
        self.guided_lift_guard_factor = torch.ones(
            self.num_envs, 1, device=self.device
        )
        self.real_steps = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self._active = torch.zeros(
            self.num_envs, 1, dtype=torch.bool, device=self.device
        )
        body_names = ("torso_link",)
        if "keyframe_torso_link" in self._entity.body_names:
            body_names += ("keyframe_torso_link",)
        ids, matched = self._entity.find_bodies(body_names, preserve_order=True)
        if tuple(matched) != body_names:
            raise ValueError(f"Unexpected assist bodies: {matched}")
        self._torso_ids = ids

    def process_actions(self, actions):
        # Preserve the executed (zeroed during delay) action for the actor observation.
        actuated = self.real_steps > self.cfg.unactuated_steps
        if hasattr(self._env, "_a6_postescape_start"):
            # Replay poses were captured after motors were already active.
            # Turning them off for the normal 30-step reset delay would make
            # these low-to-ground states collapse before the policy can act.
            actuated |= self._env._a6_postescape_start
        self._active[:] = actuated.unsqueeze(1)
        # Upstream HoST inherits clip_actions=100 from LeggedRobotCfg; a
        # Gaussian exploration sample should not be clipped at +/-1 here.
        self._raw_actions[:] = actions.clamp(-100.0, 100.0) * self._active
        if (
            (
                self.cfg.guided_arm_scale < 1
                or self.cfg.guided_hip_scale < 1
                or self.cfg.guided_lateral_scale < 1
                or self.cfg.guided_prone_arm_scale is not None
                or self.cfg.guided_supine_arm_scale is not None
            )
            and hasattr(self._env, "_a6_scene")
        ):
            rows = torch.where(
                (self._env._a6_scene == 1) & ~self._env._a6_escaped
            )[0]
            if len(rows):
                self._raw_actions[rows[:, None], self._guided_arm_ids] *= (
                    self.cfg.guided_arm_scale
                )
                self._raw_actions[rows[:, None], self._guided_hip_ids] *= (
                    self.cfg.guided_hip_scale
                )
                self._raw_actions[rows[:, None], self._guided_lateral_arm_ids] *= (
                    self.cfg.guided_lateral_scale / self.cfg.guided_arm_scale
                )
                self._raw_actions[rows[:, None], self._guided_lateral_hip_ids] *= (
                    self.cfg.guided_lateral_scale / self.cfg.guided_hip_scale
                )
                if self.cfg.guided_prone_arm_scale is not None:
                    prone_rows = rows[self._env._a6_direction[rows] == 1]
                    if len(prone_rows):
                        self._raw_actions[
                            prone_rows[:, None], self._guided_nonlateral_arm_ids
                        ] *= self.cfg.guided_prone_arm_scale / self.cfg.guided_arm_scale
                if self.cfg.guided_supine_arm_scale is not None:
                    supine_rows = rows[self._env._a6_direction[rows] == 0]
                    if len(supine_rows):
                        self._raw_actions[
                            supine_rows[:, None], self._guided_nonlateral_arm_ids
                        ] *= self.cfg.guided_supine_arm_scale / self.cfg.guided_arm_scale
        self._processed_actions[:] = self._raw_actions * self.beta
        # Action-rate rewards use the same executed, masked action as HoST.
        self._env.action_manager._action[:] = self._raw_actions
        self.real_steps += 1

    def apply_actions(self):
        joint_vel = self._entity.data.joint_vel[:, self._target_ids]
        torque = self._kp * self._processed_actions - self._kd * joint_vel
        torque = (
            torch.clamp(torque, -self._effort_limit, self._effort_limit) * self._active
        )
        position_delta = self._processed_actions
        self.guided_lift_guard_factor.fill_(1.0)
        if self.cfg.guided_lift_guard and hasattr(self._env, "_a6_scene"):
            guided = (self._env._a6_scene == 1) & ~self._env._a6_escaped
            if guided.any():
                current_z = self._env.sim.data.geom_xpos[
                    guided, self._env._a6_plate_geoms[0], 2
                ]
                lift = (current_z - self._env._a6_initial_plate_z[guided]).clamp_min(0)
                factor = ((0.045 - lift) / 0.020).clamp(0.0, 1.0)
                self.guided_lift_guard_factor[guided, 0] = factor
                torque[guided] = torque[guided] * factor[:, None]
                position_delta = self._processed_actions.clone()
                position_delta[guided] *= factor[:, None]
        self._entity.set_joint_effort_target(torque, joint_ids=self._target_ids)
        self._entity.set_joint_position_target(
            self._entity.data.joint_pos[:, self._target_ids] + position_delta,
            joint_ids=self._target_ids,
        )
        upright = self._entity.data.projected_gravity_b[:, 2:3] < -0.8
        forces = torch.zeros(self.num_envs, len(self._torso_ids), 3, device=self.device)
        force_per_body = (
            self.assist_force * (self._active & upright) / len(self._torso_ids)
        )
        forces[:, :, 2] = force_per_body
        if hasattr(self._env, "_a6_scene"):
            guided = (
                (self._env._a6_scene == 1)
                & ~self._env._a6_escaped
                & self._active[:, 0]
            )
            if guided.any():
                outward = -self._env._a6_initial_lateral_axis_xy[guided]
                magnitude = self.guided_escape_assist_force[guided, 0]
                speed_target = self.guided_escape_assist_speed_target[guided, 0]
                if (speed_target > 0).any():
                    outward_speed = (
                        self._entity.data.root_link_lin_vel_w[guided, :2]
                        * outward
                    ).sum(-1)
                    speed_gain = (
                        (speed_target - outward_speed)
                        / speed_target.clamp_min(1e-6)
                    ).clamp(-1.0, 1.0)
                    magnitude = magnitude * torch.where(
                        speed_target > 0, speed_gain, 1.0
                    )
                forces[guided, :, :2] = (
                    outward[:, None, :] * magnitude[:, None, None]
                    / len(self._torso_ids)
                )
        self._entity.write_external_wrench_to_sim(
            forces, torch.zeros_like(forces), body_ids=self._torso_ids
        )

    def reset(self, env_ids=None):
        super().reset(env_ids)
        self.real_steps[env_ids] = 0
        self._active[env_ids] = False
