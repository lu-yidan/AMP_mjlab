"""Historical A6 runtime entities and exact reset for the HoST port.

This module is the executable boundary between the existing 29-joint flat HoST
baseline and the historical three-scene A6 protocol. It intentionally:

- adds the guided plate, free plate, and scene contact sensors to ``flat29``;
- allocates the fixed 50/25/25 scene quota, four balanced low directions, and
  the 70/20/10 low/middle/near-standing strata;
- samples the exact verified reset banks and places boards only under scene 1/2;
- zeroes all velocities and clears per-environment runtime buffers on reset;
- randomises grouped body mass/inertia and Kp/Kd, with 0--10 ms command delay,
  leaving 25% of episodes nominal;
- keeps partial resets confined to the requested worlds.

Reward registration and per-substep metrics are deliberately not in this layer.
Until those layers exist, this configuration remains preflight-only.
"""

from __future__ import annotations

import copy
import dataclasses
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import mujoco
import numpy as np
import torch

from mjlab.entity import EntityCfg
from mjlab.envs.mdp import push_by_setting_velocity
from mjlab.envs.mdp.rewards import joint_pos_limits
from mjlab.managers.event_manager import (
  EventTermCfg,
  RecomputeLevel,
  requires_model_fields,
)
from mjlab.managers.metrics_manager import MetricsTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.sensor.contact_sensor import ContactMatch, ContactSensorCfg
from mjlab.utils.lab_api.math import quat_apply

from src.assets.robots.unitree_g1.unitree_actuators import (
  UnitreeActuator,
  UnitreeActuatorCfg,
)
from src.tasks.host_recovery.flat29 import flat29_env_cfg
from src.tasks.host_recovery.mdp import a6_geometry
from src.tasks.host_recovery.mdp.host_math import HOST_CONSTRAINT_DT
from src.tasks.host_recovery.mdp.actions import (
  IncrementalJointPositionAction,
  IncrementalJointPositionActionCfg,
)

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

#: Historical A6 bank locations, relative to the repository root.
A6_ROOT = Path(os.environ.get("A6_ROOT", Path.cwd()))
LOW_BANK_PATH = Path("outputs/multiterrain_bank/train.npz")
NATURAL_BANK_PATH = Path("datasets/reset_banks/natural_curriculum_v1/train.npz")
PROCEDURAL_BANK_PATH = Path("datasets/reset_banks/procedural_low_v1/train.npz")

#: A6 scene names and index values.
SCENE_NAMES = ("flat", "vertical_plate", "free_plate")
#: Reset strata, matching the historical kind indices used by R2/A6.
STRATUM_NAMES = ("low", "middle", "near_standing")

#: Flat scene internal low/middle/near-standing quotas.
FLAT_LOW_FRACTION = 0.40
FLAT_MIDDLE_FRACTION = 0.40
FLAT_NEAR_FRACTION = 0.20
#: Low states use 75% natural and 25% procedural rows (encoded as ``source``
#: in the pre-screened multiterrain bank).
LOW_PROCEDURAL_FRACTION = 0.25

#: Robot collision pattern. The repository G1 exposes all collision geoms with
#: this suffix, so this is the native HoST-side equivalent of SMP's frozen
#: deployment collision contract.
ROBOT_COLLISION_PATTERN = r".*_collision"
HAND_COLLISION_PATTERN = r"(left|right)_hand_collision$"

#: Board placement gap above the highest covered robot geom, in metres.
PLATE_GROUND_CLEARANCE = 0.002

#: Dynamics layer, matched to the A6 PPO ``num_steps_per_env=24``.
A6_STEPS_PER_UPDATE = 24
A6_DYNAMICS_WARMUP_UPDATES = 2000
PLATE_MASS_RAMP_STEPS = 100_000
#: 25% of episodes keep nominal mass/gains/delay.
A6_NOMINAL_FRACTION = 0.25
#: Six 2 ms physics-substep slots cover 0/2/4/6/8/10 ms command delay.
A6_DELAY_RING_LEN = 6

#: SMP's deployed G1 has a massless ``head`` site at this offset on torso_link.
A6_HEAD_Z_OFFSET = 0.43
#: Rolling joint-history window used by the L (single-joint stall) cost.
A6_MOTION_WINDOW = 25

#: A6 common positive recovery tasks.  Weights are pre-dt values; the HoST
#: reward manager has ``scale_rewards_by_dt=False``, so registration folds in
#: :data:`HOST_CONSTRAINT_DT` exactly once.
A6_POSITIVE_NAMES = (
  "stage_pose",
  "head_velocity",
  "height",
  "upright",
  "feet_quiet",
  "base_quiet",
  "angular_quiet",
  "joint_quiet",
  "action_quiet",
)
A6_POSITIVE_WEIGHTS = (0.22, 0.18, 0.10, 0.15, 0.08, 0.07, 0.07, 0.06, 0.07)

#: A6 common motion/actuator costs.
A6_COST_NAMES = (
  "action_rate",
  "action_acc",
  "joint_acc",
  "torque",
  "joint_speed",
  "joint_power",
  "head_overspeed",
  "sustained_effort",
)
A6_COST_WEIGHTS = (-0.0015, -0.0012, -5e-8, -1e-6, -0.02, -2e-6, -1.0, -0.05)

#: Flat29/HoST reward terms replaced by the A6 shared task/cost block.  They
#: are zeroed rather than removed so the mapping remains visible in config.
A6_NATIVE_OVERLAP_TERMS = (
  "standup",
  "regu_dof_acc",
  "regu_action_rate",
  "regu_smoothness",
  "regu_dof_vel",
  "regu_upper_dof_vel",
  "regu_dof_pos_limits",
  "style_style_ang_vel_xy",
  "style_ground_parallel",
  "target_ang_vel_xy",
  "target_lin_vel_xy",
  "target_feet_height_var",
  "target_target_orientation",
  "target_target_base_height",
)


def _a6_actuator_group(name: str) -> int:
  """Historical A6 Kp/Kd group: waist/hip/knee/ankle/arm/wrist."""
  if "waist" in name:
    return 0
  if "hip" in name:
    return 1
  if "knee" in name:
    return 2
  if "ankle" in name:
    return 3
  if "shoulder" in name or "elbow" in name:
    return 4
  return 5


@dataclass(kw_only=True)
class A6UnitreeActuatorCfg(UnitreeActuatorCfg):
  """Unitree torque-speed actuator whose Kp/Kd are per-world and per-target."""

  def build(self, entity, target_ids, target_names):
    return A6UnitreeActuator(self, entity, target_ids, target_names)


class A6UnitreeActuator(UnitreeActuator):
  """Unitree actuator with grouped Kp/Kd randomisation.

  The underlying MuJoCo actuator is a built-in position actuator.  Keeping the
  per-target ``stiffness``/``damping`` tensors and the model
  ``actuator_gainprm``/``actuator_biasprm`` in lockstep preserves the
  torque-speed/friction compensation while randomising effective PD gains.
  """

  stiffness: torch.Tensor | None
  damping: torch.Tensor | None
  default_stiffness: torch.Tensor | None
  default_damping: torch.Tensor | None

  def initialize(self, mj_model, model, data, device):
    super().initialize(mj_model, model, data, device)
    num_envs = data.nworld
    num_targets = len(self.target_names)
    self.stiffness = torch.full(
      (num_envs, num_targets), self.cfg.stiffness, dtype=torch.float, device=device
    )
    self.damping = torch.full(
      (num_envs, num_targets), self.cfg.damping, dtype=torch.float, device=device
    )
    self.default_stiffness = self.stiffness.clone()
    self.default_damping = self.damping.clone()

  def compute(self, cmd):
    self._joint_vel[:] = cmd.vel
    effort = self.stiffness * (cmd.position_target - cmd.pos)
    effort += self.damping * (cmd.velocity_target - cmd.vel)
    effort += cmd.effort_target
    effort = self._clip_effort(effort)
    effort -= (
      self._friction_static * torch.tanh(cmd.vel / self._activation_vel)
      + self._friction_dynamic * cmd.vel
    )
    kp = torch.clamp(self.stiffness, min=1e-6)
    kd = self.damping
    return cmd.pos + (effort + kd * cmd.vel) / kp

  def set_gains(
    self,
    env_ids,
    kp=None,
    kd=None,
    env=None,
    kp_factor=None,
    kd_factor=None,
  ):
    """Set per-world gains and synchronise the native model actuator fields.

    ``kp``/``kd`` are the actual per-target gains written to this Python
    actuator.  ``*_factor`` are the multiplicative factors relative to the
    compiled MuJoCo defaults, used to update ``actuator_gainprm`` and
    ``actuator_biasprm`` without double-counting the default stiffness.
    """
    if kp is not None:
      self.stiffness[env_ids] = kp
    if kd is not None:
      self.damping[env_ids] = kd
    if env is None:
      return
    ctrl_ids = self.global_ctrl_ids
    default_gainprm = env.sim.get_default_field("actuator_gainprm")
    default_biasprm = env.sim.get_default_field("actuator_biasprm")
    if kp_factor is not None:
      env.sim.model.actuator_gainprm[env_ids[:, None], ctrl_ids, 0] = (
        default_gainprm[ctrl_ids, 0] * kp_factor
      )
      env.sim.model.actuator_biasprm[env_ids[:, None], ctrl_ids, 1] = (
        default_biasprm[ctrl_ids, 1] * kp_factor
      )
    if kd_factor is not None:
      env.sim.model.actuator_biasprm[env_ids[:, None], ctrl_ids, 2] = (
        default_biasprm[ctrl_ids, 2] * kd_factor
      )


def _as_a6_actuator_cfg(cfg):
  """Convert a concrete Unitree actuator config to the A6 dynamic variant."""
  if isinstance(cfg, A6UnitreeActuatorCfg):
    return cfg
  fields = dataclasses.fields(cfg)
  return A6UnitreeActuatorCfg(
    **{field.name: getattr(cfg, field.name) for field in fields if field.init}
  )


def _a6_dynamics_robot_cfg(robot_cfg):
  """Return a robot config whose Unitree actuators support grouped Kp/Kd."""
  robot = copy.deepcopy(robot_cfg)
  if robot.articulation is None:
    return robot
  robot.articulation.actuators = tuple(
    _as_a6_actuator_cfg(actuator) for actuator in robot.articulation.actuators
  )
  return robot


@dataclass(kw_only=True)
class A6DelayedPositionActionCfg(IncrementalJointPositionActionCfg):
  """Incremental joint-position action with historical 0--10 ms command delay."""

  def build(self, env):
    return A6DelayedPositionAction(self, env)


class A6DelayedPositionAction(IncrementalJointPositionAction):
  """Six-slot ring buffer over the processed position targets.

  ``apply_actions`` runs once per 2 ms physics substep, so lag 0..5 is exactly
  0/2/4/6/8/10 ms.  Only the requested reset rows are re-primed, leaving other
  worlds' delay buffers untouched on partial reset.
  """

  def __init__(self, cfg, env):
    super().__init__(cfg=cfg, env=env)
    self._a6_delay_buffer = self._processed_actions.unsqueeze(0).repeat(
      A6_DELAY_RING_LEN, 1, 1
    )
    self._a6_delay_cursor = 0
    self._a6_delay_rows = torch.arange(self.num_envs, device=self.device)
    self._a6_entry = self._entity.data.joint_pos[:, self.target_ids].clone()

  def reset(self, env_ids=None):
    super().reset(env_ids)
    ids = slice(None) if env_ids is None else env_ids
    self._a6_entry[ids] = self._entity.data.joint_pos[ids][:, self.target_ids]
    self._a6_delay_buffer[:, ids] = self._a6_entry[ids].unsqueeze(0)

  def apply_actions(self):
    lag = getattr(
      self._env,
      "_a6_lag",
      torch.zeros(self.num_envs, dtype=torch.long, device=self.device),
    )
    self._a6_delay_buffer[self._a6_delay_cursor] = self._processed_actions
    target = self._a6_delay_buffer[
      (self._a6_delay_cursor - lag) % A6_DELAY_RING_LEN, self._a6_delay_rows
    ]
    self._a6_delay_cursor = (self._a6_delay_cursor + 1) % A6_DELAY_RING_LEN
    original = self._processed_actions
    self._processed_actions = target
    super().apply_actions()
    self._processed_actions = original
    if getattr(self._env, "_a6_substep_enabled", True):
      _a6_sample_substep(self._env)


def _flat_strata(num_envs: int) -> torch.Tensor:
  """Return deterministic low/middle/near-standing labels inside flat scenes.

  Obstacle scenes are handled separately by the caller as all-low. For flat
  scenes the historical allocation is 40/40/20 inside each of the four
  directions, which yields the global 70/20/10 mixture once the 25/25 obstacle
  scenes are included.
  """
  scene_np, direction_np, _ = a6_geometry.quotas(num_envs)
  scene = torch.as_tensor(scene_np, dtype=torch.long)
  direction = torch.as_tensor(direction_np, dtype=torch.long)
  stratum = torch.zeros(num_envs, dtype=torch.long)

  for dir_id in range(len(a6_geometry.DIRECTIONS)):
    flat_ids = torch.nonzero(
      (scene == 0) & (direction == dir_id), as_tuple=False
    ).flatten()
    count = int(flat_ids.numel())
    low = round(count * FLAT_LOW_FRACTION)
    near = round(count * FLAT_NEAR_FRACTION)
    middle = count - low - near
    stratum[flat_ids[:low]] = 0
    stratum[flat_ids[low : low + middle]] = 1
    stratum[flat_ids[low + middle :]] = 2

  return scene, direction, stratum


def a6_env_cfg(
  play: bool = False,
  seed: int | None = None,
  dynamics: bool | None = None,
  group: str = "G+",
  bank_split: str = "train",
) -> "ManagerBasedRlEnvCfg":
  """Return the A6 three-scene environment configuration.

  ``play=True`` keeps the flat29 play-mode overrides (no pull force, no
  curriculum, fixed action scale) so preflight can inspect the runtime without
  starting a training loop.  ``dynamics`` defaults to ``not play``: nominal in
  evaluation, randomised during training.  ``group`` selects the dense escape
  guidance arm: ``"G+"`` keeps G/clearance/separation and obstacle alpha=0.05;
  ``"G-"`` disables those three dense terms and keeps alpha=1.0.  Completion,
  plate force, Q and L stay on in both groups. ``bank_split`` is explicitly
  restricted to the archived train or historical-validation assets.
  """
  from mjlab.envs import ManagerBasedRlEnvCfg

  if group not in ("G+", "G-"):
    raise ValueError(f"unknown A6 group {group!r}")
  if bank_split not in ("train", "validation"):
    raise ValueError(f"unknown A6 bank split {bank_split!r}")
  dense_guide = group == "G+"
  if dynamics is None:
    dynamics = not play

  cfg: ManagerBasedRlEnvCfg = flat29_env_cfg(play=play)

  # Historical A6 keeps all environment origins at the world origin. The bank
  # rows are already expressed relative to the local environment origin.
  cfg.scene.env_spacing = 0.0
  cfg.scene.spec_fn = a6_geometry.scene_spec

  # Swap the Unitree actuator implementation for one that supports per-world,
  # per-joint grouped Kp/Kd while preserving the torque-speed/friction model.
  cfg.scene.entities["robot"] = _a6_dynamics_robot_cfg(
    cfg.scene.entities["robot"]
  )
  action_cfg = cfg.actions["joint_pos"]
  cfg.actions["joint_pos"] = A6DelayedPositionActionCfg(
    entity_name=action_cfg.entity_name,
    actuator_names=action_cfg.actuator_names,
    scale=action_cfg.scale,
  )

  # Guided plate: mjlab auto-wraps the slide-joint spec in a mocap body, so the
  # reset event can move it per environment while the slide joint enforces the
  # historical vertical-only constraint.
  cfg.scene.entities["escape_obstacle"] = EntityCfg(
    spec_fn=a6_geometry.guided_plate_spec,
    init_state=EntityCfg.InitialStateCfg(
      pos=(20.0, 20.0, 0.8),
      joint_pos={"escape_plate_slide": 0.0},
      joint_vel={"escape_plate_slide": 0.0},
    ),
  )
  cfg.scene.entities["free_obstacle"] = EntityCfg(
    spec_fn=a6_geometry.free_plate_spec,
    init_state=EntityCfg.InitialStateCfg(pos=(20.0, 20.0, 0.1)),
  )

  # Obstacle/robot contact sensors used by the reset audit and later by escape
  # reward/termination code.
  cfg.scene.sensors = tuple(cfg.scene.sensors) + (
    ContactSensorCfg(
      name="guided_contact",
      primary=ContactMatch(
        mode="geom", pattern=ROBOT_COLLISION_PATTERN, entity="robot"
      ),
      secondary=ContactMatch(
        mode="body", pattern="escape_plate", entity="escape_obstacle"
      ),
      fields=("found", "force", "dist"),
      reduce="mindist",
      num_slots=1,
    ),
    ContactSensorCfg(
      name="free_contact",
      primary=ContactMatch(
        mode="geom", pattern=ROBOT_COLLISION_PATTERN, entity="robot"
      ),
      secondary=ContactMatch(
        mode="body", pattern="plate", entity="free_obstacle"
      ),
      fields=("found", "force", "dist"),
      reduce="mindist",
      num_slots=1,
    ),
    ContactSensorCfg(
      name="path_hands",
      primary=ContactMatch(
        mode="geom", pattern=HAND_COLLISION_PATTERN, entity="robot"
      ),
      secondary=ContactMatch(mode="body", pattern="terrain"),
      fields=("found", "force"),
      reduce="maxforce",
      num_slots=1,
    ),
    ContactSensorCfg(
      name="quality_feet",
      primary=ContactMatch(
        mode="body",
        pattern=("left_ankle_roll_link", "right_ankle_roll_link"),
        entity="robot",
      ),
      secondary=ContactMatch(mode="body", pattern="terrain"),
      fields=("force",),
      reduce="netforce",
    ),
    ContactSensorCfg(
      name="quality_other",
      primary=ContactMatch(
        mode="geom",
        pattern=ROBOT_COLLISION_PATTERN,
        entity="robot",
        exclude=("left_foot[1-7]_collision", "right_foot[1-7]_collision"),
      ),
      secondary=ContactMatch(mode="body", pattern="terrain"),
      fields=("force",),
      reduce="netforce",
    ),
  )

  cfg.sim.nconmax = max(int(getattr(cfg.sim, "nconmax", 0) or 0), 256)
  cfg.sim.njmax = max(int(getattr(cfg.sim, "njmax", 0) or 0), 4000)

  # This reset must run after the flat reset events, so it is appended after
  # the existing configuration has been fully constructed.
  cfg.events["a6_reset"] = EventTermCfg(
    func=a6_reset, mode="reset", params={"bank_split": bank_split}
  )
  cfg.events["a6_dynamics_reset"] = EventTermCfg(
    func=a6_dynamics_reset,
    mode="reset",
    params={"dynamics": dynamics},
  )
  cfg.events["a6_phase"] = EventTermCfg(
    func=_a6_phase_update, mode="step", params={}
  )
  if play:
    cfg.events.pop("push_robot", None)
  else:
    cfg.events["push_robot"] = EventTermCfg(
      func=push_by_setting_velocity,
      mode="interval",
      interval_range_s=(1.0, 3.0),
      params={
        "velocity_range": {
          "x": (-0.5, 0.5),
          "y": (-0.5, 0.5),
          "z": (-0.4, 0.4),
          "roll": (-0.52, 0.52),
          "pitch": (-0.52, 0.52),
          "yaw": (-0.78, 0.78),
        }
      },
    )

  # The paired HoST arms retain the method's auxiliary pull curriculum, but its
  # update clock must match this protocol's 24-step PPO rollout.
  if "pull_force" in cfg.curriculum:
    cfg.curriculum["pull_force"].params["steps_per_update"] = A6_STEPS_PER_UPDATE

  # Replace overlapping flat29/HoST terms explicitly instead of silently
  # double-counting them next to the shared A6 task/cost block.
  for name in A6_NATIVE_OVERLAP_TERMS:
    if name in cfg.rewards:
      cfg.rewards[name].weight = 0.0

  dt = HOST_CONSTRAINT_DT
  for index, (name, weight) in enumerate(
    zip(A6_POSITIVE_NAMES, A6_POSITIVE_WEIGHTS)
  ):
    cfg.rewards[f"a6_{name}"] = RewardTermCfg(
      func=a6_task_reward,
      params={"index": index, "dense_guide": dense_guide},
      weight=weight * dt,
    )
  for index, (name, weight) in enumerate(
    zip(A6_COST_NAMES, A6_COST_WEIGHTS)
  ):
    cfg.rewards[f"a6_{name}"] = RewardTermCfg(
      func=a6_cost_reward, params={"index": index}, weight=weight * dt
    )
  cfg.rewards["a6_soft_limits"] = RewardTermCfg(
    func=joint_pos_limits,
    params={"asset_cfg": SceneEntityCfg("robot", joint_names=(".*",))},
    weight=-0.10 * dt,
  )

  # Common obstacle interaction terms.  Completion is paid gradually through
  # the escaped boolean and re-contact revokes it by resetting the hold.
  cfg.rewards["plate_completion"] = RewardTermCfg(
    func=a6_completion_reward, weight=0.60 * dt
  )
  cfg.rewards["plate_force"] = RewardTermCfg(
    func=a6_force_reward, weight=-0.03 * dt
  )
  if dense_guide:
    cfg.rewards["plate_geometry_progress"] = RewardTermCfg(
      func=a6_path_reward, params={"index": 0}, weight=0.45 * dt
    )
    cfg.rewards["plate_clearance"] = RewardTermCfg(
      func=a6_path_reward, params={"index": 1}, weight=0.08 * dt
    )
    cfg.rewards["plate_separation"] = RewardTermCfg(
      func=a6_separation_reward, weight=0.01 * dt
    )

  # Q and L are common costs and also apply on flat ground.
  cfg.rewards["path_quiet_feet"] = RewardTermCfg(
    func=a6_path_reward, params={"index": 2}, weight=-0.03 * dt
  )
  cfg.rewards["path_joint_stall"] = RewardTermCfg(
    func=a6_path_reward, params={"index": 3}, weight=-0.20 * dt
  )

  cfg.terminations["invalid_plate"] = TerminationTermCfg(func=a6_invalid)

  cfg.metrics["a6_stage"] = MetricsTermCfg(func=a6_stage_metric, params={})
  cfg.metrics["a6_standing_gate"] = MetricsTermCfg(
    func=a6_gate_metric, params={}
  )
  cfg.metrics["a6_escaped"] = MetricsTermCfg(func=a6_escaped_metric, params={})
  cfg.metrics["a6_clear_hold"] = MetricsTermCfg(
    func=a6_clear_hold_metric, params={}
  )
  cfg.metrics["a6_joint_stall"] = MetricsTermCfg(
    func=a6_load_metric, params={}
  )

  if seed is not None:
    cfg.seed = seed
  return cfg

def _natural_weights(bank: dict[str, np.ndarray]) -> np.ndarray:
  """Mirror SMP's natural-curriculum weighting for middle/late pools."""
  stages = ("late", "middle", "low")
  directions = ("supine", "prone", "left_side_down", "right_side_down")
  mixes = (0.5, 0.4, 0.1)  # ``weights_for(bank, stage=0)`` in SMP.
  weights = np.zeros(len(bank["qpos"]), dtype=np.float64)
  groups = np.array([str(x).replace("__mirror", "") for x in bank["clips"]])
  for stage_name, mass in zip(stages, mixes):
    ids = np.flatnonzero(bank["stages"] == stage_name)
    assert len(ids)
    labels = directions if stage_name == "low" else (None,)
    for label in labels:
      rows = ids if label is None else ids[bank["labels"][ids] == label]
      assert len(rows), (stage_name, label)
      clips = sorted(set(groups[rows]))
      for clip in clips:
        subset = rows[groups[rows] == clip]
        weights[subset] = mass / len(labels) / len(clips) / len(subset)
  assert np.isclose(weights.sum(), 1.0)
  return weights


def _ensure_a6_state(
  env: "ManagerBasedRlEnv", bank_split: str | None = None
) -> None:
  """Allocate per-environment A6 buffers and bank pools once per env."""
  if getattr(env, "_a6_initialized", False):
    if bank_split is not None and env._a6_bank_split != bank_split:
      raise RuntimeError("A6 bank split cannot change after initialization")
    return
  bank_split = bank_split or "train"
  if bank_split not in ("train", "validation"):
    raise ValueError(f"unknown A6 bank split {bank_split!r}")

  num_envs = env.num_envs
  device = env.device
  if num_envs < 32 or num_envs % 32:
    raise ValueError("A6 requires num_envs >= 32 and divisible by 32")

  scene, direction, stratum = _flat_strata(num_envs)
  env._a6_scene = scene.to(device)
  env._a6_direction = direction.to(device)
  env._a6_stratum = stratum.to(device)
  env._a6_source = torch.zeros(num_envs, dtype=torch.long, device=device)

  # Source=1 means procedural low state. The split is applied inside each
  # scene/direction low cell, exactly as in the pre-screened bank.
  for scene_id in range(len(SCENE_NAMES)):
    for dir_id in range(len(a6_geometry.DIRECTIONS)):
      ids = torch.nonzero(
        (env._a6_scene == scene_id)
        & (env._a6_direction == dir_id)
        & (env._a6_stratum == 0),
        as_tuple=False,
      ).flatten()
      procedural = round(int(ids.numel()) * LOW_PROCEDURAL_FRACTION)
      env._a6_source[ids[:procedural]] = 1

  # Verified low obstacle/natural/procedural bank. The multiterrain bank has
  # already merged natural and procedural low rows into one screened dataset;
  # ``source`` is the 0/1 provenance column.
  low_np = np.load(
    A6_ROOT / LOW_BANK_PATH.with_name(f"{bank_split}.npz"), allow_pickle=False
  )
  nat_np = np.load(
    A6_ROOT / NATURAL_BANK_PATH.with_name(f"{bank_split}.npz"), allow_pickle=False
  )
  np.load(
    A6_ROOT / PROCEDURAL_BANK_PATH.with_name(f"{bank_split}.npz"),
    allow_pickle=False,
  )

  env._a6_low_bank = torch.as_tensor(
    low_np["qpos"], dtype=torch.float32, device=device
  )
  env._a6_natural_bank = torch.as_tensor(
    nat_np["qpos"], dtype=torch.float32, device=device
  )

  env._a6_low_pools: dict[tuple[int, int, int], torch.Tensor] = {}
  for scene_id in range(len(SCENE_NAMES)):
    for dir_id in range(len(a6_geometry.DIRECTIONS)):
      for source in (0, 1):
        mask = (
          (low_np["stratum"] == scene_id)
          & (low_np["direction"] == dir_id)
          & (low_np["source"] == source)
        )
        pool = torch.as_tensor(np.flatnonzero(mask), device=device)
        if int(pool.numel()) == 0:
          raise RuntimeError(
            f"empty low bank pool scene={scene_id} direction={dir_id} source={source}"
          )
        env._a6_low_pools[(scene_id, dir_id, source)] = pool

  natural_weights = _natural_weights(nat_np)
  env._a6_nat_pools: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
  for kind, stage_name in ((1, "middle"), (2, "late")):
    mask = nat_np["stages"] == stage_name
    pool = torch.as_tensor(np.flatnonzero(mask), device=device)
    weights = torch.as_tensor(
      natural_weights[mask], dtype=torch.float32, device=device
    )
    weights = weights / weights.sum()
    env._a6_nat_pools[kind] = (pool, weights)

  env._a6_low_rng = torch.Generator(device=device).manual_seed(
    int(env.cfg.seed) + 91019
  )
  env._a6_nat_rng = torch.Generator(device=device).manual_seed(
    int(env.cfg.seed) + 62019
  )

  # Native entity/geom indices. These are cheap to resolve once and avoid
  # re-matching names on every reset.
  robot = env.scene["robot"]
  robot_geom_local, _ = robot.find_geoms(ROBOT_COLLISION_PATTERN)
  if not robot_geom_local:
    raise RuntimeError(f"no robot geoms match {ROBOT_COLLISION_PATTERN!r}")
  env._a6_robot_geoms = robot.indexing.geom_ids[
    torch.as_tensor(robot_geom_local, device=device)
  ].long()

  guided = env.scene["escape_obstacle"]
  free = env.scene["free_obstacle"]
  guided_local, _ = guided.find_geoms("escape_plate_geom")
  free_local, _ = free.find_geoms("plate_geom")
  if not guided_local or not free_local:
    raise RuntimeError("A6 plate geom names were not compiled")
  env._a6_guided_geom = guided.indexing.geom_ids[
    torch.as_tensor(guided_local, device=device)
  ].long()
  env._a6_free_geom = free.indexing.geom_ids[
    torch.as_tensor(free_local, device=device)
  ].long()
  env._a6_plate_geoms = torch.stack(
    [env._a6_guided_geom.squeeze(), env._a6_free_geom.squeeze()]
  )
  guided_body_local, _ = guided.find_bodies(
    ["escape_plate"], preserve_order=True
  )
  free_body_local, _ = free.find_bodies(["plate"], preserve_order=True)
  if len(guided_body_local) != 1 or len(free_body_local) != 1:
    raise RuntimeError("A6 plate body names were not compiled")
  env._a6_plate_bodies = torch.stack(
    [
      guided.indexing.body_ids[guided_body_local[0]],
      free.indexing.body_ids[free_body_local[0]],
    ]
  ).long()

  # Dynamics-grouped body and actuator indices.  The body grouping follows
  # balanced_dynamics: legs (0), upper subtree (1), pelvis/waist (2); column 3
  # is reserved for audit and never scaled.
  model = env.sim.mj_model
  body_names = [
    model.body(int(body_id)).name.split("/")[-1]
    for body_id in robot.indexing.body_ids
  ]

  def descendants(name: str) -> set[int]:
    root = model.body("robot/" + name).id
    result = []
    for local_id, body_id in enumerate(robot.indexing.body_ids.tolist()):
      cursor = body_id
      while cursor > 0 and cursor != root:
        cursor = int(model.body_parentid[cursor])
      if cursor == root:
        result.append(local_id)
    return set(result)

  legs = descendants("left_hip_pitch_link") | descendants(
    "right_hip_pitch_link"
  )
  upper = descendants("torso_link")
  env._a6_bodies = robot.indexing.body_ids.long()
  env._a6_body_groups = torch.tensor(
    [
      0 if local_id in legs else 1 if local_id in upper else 2
      for local_id in range(len(body_names))
    ],
    device=device,
  )
  env._a6_act_groups = [
    torch.tensor(
      [_a6_actuator_group(name) for name in actuator.target_names],
      device=device,
    )
    for actuator in robot.actuators
  ]

  # Runtime buffers required by the A6 reward/termination layers.  They are
  # allocated now so reset can already clear them per env and partial resets
  # can be audited without reallocating tensors.
  num_joints = robot.data.joint_pos.shape[-1]

  head_local, _ = robot.find_bodies(["torso_link"], preserve_order=True)
  feet_local, _ = robot.find_bodies(
    ["left_ankle_roll_link", "right_ankle_roll_link"], preserve_order=True
  )
  knee_local, _ = robot.find_joints(
    ["left_knee_joint", "right_knee_joint"], preserve_order=True
  )
  if not head_local or len(feet_local) != 2 or len(knee_local) != 2:
    raise RuntimeError("A6 reward body/joint indices were not compiled")
  env._a6_head_body = torch.as_tensor(head_local[0], device=device, dtype=torch.long)
  env._a6_feet_bodies = torch.as_tensor(feet_local, device=device, dtype=torch.long)
  env._a6_knee_joints = torch.as_tensor(knee_local, device=device, dtype=torch.long)

  tau_limits = torch.zeros(num_joints, device=device)
  for actuator in robot.actuators:
    if not isinstance(actuator, A6UnitreeActuator):
      raise RuntimeError(
        f"expected A6UnitreeActuator, got {type(actuator).__name__}"
      )
    tau_limits[actuator.target_ids] = actuator._effort_y1[0]
  if torch.any(tau_limits <= 0):
    raise RuntimeError("A6 actuator torque limits are not positive")
  env._a6_tau_limits = tau_limits

  zeros = torch.zeros(num_envs, device=device)
  zeros_bool = torch.zeros(num_envs, dtype=torch.bool, device=device)
  zeros_long = torch.zeros(num_envs, dtype=torch.long, device=device)
  zeros_joints = torch.zeros(num_envs, num_joints, device=device)
  env._a6_best_score = zeros.clone()
  env._a6_best_clearance = zeros.clone()
  env._a6_initial_count = zeros.clone()
  env._a6_progress = zeros.clone()
  env._a6_clearance_score = zeros.clone()
  env._a6_gate = zeros.clone()
  env._a6_load_acc = zeros.clone()
  env._a6_load = zeros.clone()
  env._a6_support = zeros.clone()
  env._a6_force = zeros.clone()
  env._a6_separation_progress = zeros.clone()
  env._a6_best_distance = zeros.clone()
  env._a6_clearance = zeros.clone()
  env._a6_ever_contact = zeros_bool.clone()
  env._a6_invalid = zeros_bool.clone()
  env._a6_escaped = zeros_bool.clone()
  env._a6_no_progress = zeros_bool.clone()
  env._a6_clear_hold = zeros_long.clone()
  env._a6_stage = zeros_long.clone()
  env._a6_hold = zeros_long.clone()
  env._a6_timer = zeros_joints.clone()
  env._a6_task = torch.zeros(num_envs, len(A6_POSITIVE_NAMES), device=device)
  env._a6_cost = torch.zeros(num_envs, len(A6_COST_NAMES), device=device)
  env._a6_accum = torch.zeros(num_envs, 6, device=device)
  env._a6_f_effort_ms = zeros_joints.clone()
  env._a6_joint_acc = zeros_joints.clone()
  env._a6_last_action = zeros_joints.clone()
  env._a6_last_delta = zeros_joints.clone()
  env._a6_motion = torch.zeros(
    A6_MOTION_WINDOW, num_envs, num_joints + 2, device=device
  )
  env._a6_motion_cursor = 0
  env._a6_tick = -1
  env._a6_phase_tick = -1
  env._a6_subtick = 0
  env._a6_substep_enabled = True
  env._a6_reset_draws = torch.zeros(3, dtype=torch.long, device=device)

  env._a6_dynamics_rng = torch.Generator(device=device).manual_seed(
    int(env.cfg.seed) + 43019
  )
  env._a6_lag = torch.zeros(num_envs, dtype=torch.long, device=device)
  env._a6_mass_factors = torch.ones((num_envs, 4), device=device)
  env._a6_gain_factors = torch.ones((num_envs, 6), device=device)
  env._a6_nominal = torch.ones(num_envs, dtype=torch.bool, device=device)
  env._a6_plate_mass = torch.full((num_envs,), 6.0, device=device)
  env._a6_bank_split = bank_split

  env._a6_initialized = True


def _draw_qpos(env: "ManagerBasedRlEnv", ids: torch.Tensor) -> torch.Tensor:
  """Sample bank rows for the requested envs, respecting strata and source."""
  device = env.device
  scene = env._a6_scene[ids]
  direction = env._a6_direction[ids]
  stratum = env._a6_stratum[ids]
  source = env._a6_source[ids]
  indices = torch.empty(len(ids), dtype=torch.long, device=device)

  low_mask = stratum == 0
  for (scene_id, dir_id, source_id), pool in env._a6_low_pools.items():
    mask = (
      low_mask
      & (scene == scene_id)
      & (direction == dir_id)
      & (source == source_id)
    )
    count = int(mask.sum())
    if count:
      indices[mask] = pool[
        torch.randint(0, len(pool), (count,), generator=env._a6_low_rng, device=device)
      ]

  for kind, (pool, weights) in env._a6_nat_pools.items():
    mask = stratum == kind
    count = int(mask.sum())
    if count:
      indices[mask] = pool[
        torch.multinomial(
          weights, count, replacement=True, generator=env._a6_nat_rng
        )
      ]

  qpos = torch.empty((len(ids), 36), dtype=torch.float32, device=device)
  if torch.any(low_mask):
    qpos[low_mask] = env._a6_low_bank[indices[low_mask]]
  middle_mask = stratum == 1
  if torch.any(middle_mask):
    qpos[middle_mask] = env._a6_natural_bank[indices[middle_mask]]
  near_mask = stratum == 2
  if torch.any(near_mask):
    qpos[near_mask] = env._a6_natural_bank[indices[near_mask]]

  env._a6_reset_draws += torch.bincount(
    stratum, minlength=env._a6_reset_draws.numel()
  )
  return qpos


def _robot_bounds(env: "ManagerBasedRlEnv") -> tuple[torch.Tensor, torch.Tensor]:
  """World-XY projected extents of the robot collision geoms."""
  ids = env._a6_robot_geoms
  pos = env.sim.data.geom_xpos[:, ids]
  mat = env.sim.data.geom_xmat[:, ids]
  size = env.sim.model.geom_size[:, ids]
  typ = env.sim.model.geom_type[ids]

  ext = torch.einsum("ngij,ngj->ngi", mat.abs(), size)
  ext = torch.where(
    (typ == int(mujoco.mjtGeom.mjGEOM_SPHERE))[None, :, None],
    size[:, :, 0, None],
    ext,
  )
  ext = torch.where(
    (typ == int(mujoco.mjtGeom.mjGEOM_CAPSULE))[None, :, None],
    size[:, :, 0, None] + size[:, :, 1, None] * mat[:, :, :, 2].abs(),
    ext,
  )
  return pos, ext


def _plate_geometry(
  env: "ManagerBasedRlEnv",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
  """Conservative XY coverage score and clearance for the active board."""
  pos, ext = _robot_bounds(env)
  rows = torch.arange(env.num_envs, device=env.device)
  gid = env._a6_plate_geoms[(env._a6_scene == 2).long()]
  pp = env.sim.data.geom_xpos[rows, gid]
  rot = env.sim.data.geom_xmat[rows, gid]
  size = env.sim.model.geom_size[rows, gid]
  pe = torch.einsum("nij,nj->ni", rot.abs(), size)

  overlap = ext[..., :2] + pe[:, None, :2] - (pos[..., :2] - pp[:, None, :2]).abs()
  covered = (overlap > 0.0).all(-1)
  score = torch.where(
    covered,
    overlap.clamp_min(0.0).amin(-1),
    torch.zeros_like(overlap[..., 0]),
  ).sum(-1)
  clearance = torch.where(
    covered,
    torch.zeros_like(overlap[..., 0]),
    (-overlap).clamp_min(0.0).norm(dim=-1),
  ).amin(-1)
  return covered.sum(-1), score, clearance

def _a6_smooth_gate(value, low, high):
  """Smoothstep gate used by the A6 reward transfer."""
  t = ((value - low) / (high - low)).clamp(0.0, 1.0)
  return t * t * (3.0 - 2.0 * t)


def _a6_gate_ramp(value, low, high):
  """Linear ramp gate used by the historical R2/L reward code."""
  return ((value - low) / (high - low)).clamp(0.0, 1.0)


def _a6_height(env):
  """Reconstructed SMP head-site height above the local env origin."""
  robot = env.scene["robot"]
  return (
    robot.data.body_link_pos_w[:, env._a6_head_body, 2]
    + _a6_head_offset(env)[:, 2]
    - env.scene.env_origins[:, 2]
  )


def _a6_head_offset(env):
  robot = env.scene["robot"]
  local = torch.zeros((env.num_envs, 3), device=env.device)
  local[:, 2] = A6_HEAD_Z_OFFSET
  return quat_apply(robot.data.body_link_quat_w[:, env._a6_head_body], local)


def _a6_head_velocity(env):
  robot = env.scene["robot"]
  return robot.data.body_link_lin_vel_w[:, env._a6_head_body] + torch.cross(
    robot.data.body_link_ang_vel_w[:, env._a6_head_body],
    _a6_head_offset(env),
    dim=-1,
  )


def _a6_vertical_overspeed(vz, stage):
  up = vz.new_tensor((0.30, 0.30, 0.30, 0.20))[stage]
  return (vz - up).clamp_min(0.0).square() + (-vz - 0.20).clamp_min(0.0).square()


def _a6_completion_clearance(env):
  """Historical completion geometry; deliberately distinct from planar G."""
  pos, ext = _robot_bounds(env)
  rows = torch.arange(env.num_envs, device=env.device)
  gid = env._a6_plate_geoms[(env._a6_scene == 2).long()]
  pp = env.sim.data.geom_xpos[rows, gid]
  pe = torch.einsum(
    "nij,nj->ni", env.sim.data.geom_xmat[rows, gid].abs(),
    env.sim.model.geom_size[rows, gid],
  )
  delta = (pos[..., :2] - pp[:, None, :2]).abs() - ext[..., :2] - pe[:, None, :2]
  sep = delta.amax(-1)
  above = (pos[..., 2] - ext[..., 2]) > pp[:, None, 2] + pe[:, None, 2] + 0.025
  return torch.where(above, torch.ones_like(sep), sep).amin(-1)


def _a6_upright(env):
  return (-env.scene["robot"].data.projected_gravity_b[:, 2]).clamp(0.0, 1.0)


def _a6_ground_force(env, name):
  """Return global-frame contact force for a named robot/terrain sensor."""
  return env.scene[name].data.force


def _a6_sustained_update(previous, normalized_tau, dt):
  """First-order EMA of squared normalised torque, from historical R1."""
  return previous + (1.0 - math.exp(-dt / 0.5)) * (
    normalized_tau.square() - previous
  )


def _a6_effort_cost(mean_square):
  """Top-3 sustained-effort cost from historical R1."""
  root = mean_square.clamp_min(0.0).sqrt()
  return ((root - 0.45).clamp_min(0.0) / 0.55).square().topk(
    3, dim=-1
  ).values.mean(-1)


def _a6_sample_substep(env):
  """Accumulate physical-substep costs once per 2 ms physics step.

  Called from :class:`A6DelayedPositionAction` because this mjlab version has
  no per-substep metric hook.  One control step contains ``decimation`` calls,
  so the accumulator is zeroed at the first substep of each control step.
  """
  _ensure_a6_state(env)
  if not getattr(env, "_a6_substep_enabled", True):
    return
  if env._a6_subtick % env.cfg.decimation == 0:
    env._a6_accum.zero_()
    env._a6_load_acc.zero_()
    env._a6_joint_acc.zero_()

  robot = env.scene["robot"]
  # Generalised torque is in joint/DoF order, matching dq and torque limits.
  tau = env.sim.data.qfrc_actuator[:, robot.indexing.joint_v_adr]
  dq = robot.data.joint_vel
  stage = env._a6_stage

  env._a6_f_effort_ms.copy_(
    _a6_sustained_update(
      env._a6_f_effort_ms,
      tau / env._a6_tau_limits[None],
      env.physics_dt,
    )
  )

  speed_limits = tau.new_tensor((6.0, 5.0, 4.0, 3.5))[stage, None]
  power_limits = tau.new_tensor((140.0, 110.0, 90.0, 75.0))[stage, None]
  joint_acc_weight = tau.new_tensor((0.35, 0.65, 1.0, 1.0))[stage]
  torque_weight = tau.new_tensor((0.5, 0.75, 1.0, 1.0))[stage]
  head_vz = _a6_head_velocity(env)[:, 2]

  joint_acc = robot.data.joint_acc.square().sum(-1) * joint_acc_weight
  torque_cost = tau.square().sum(-1) * torque_weight
  speed_cost = (dq.abs() - speed_limits).clamp_min(0.0).square().sum(-1)
  power_cost = ((tau * dq).abs() - power_limits).clamp_min(0.0).square().mean(-1)
  head_overspeed = _a6_vertical_overspeed(head_vz, stage)
  sustained = _a6_effort_cost(env._a6_f_effort_ms)
  env._a6_accum += torch.stack(
    [
      joint_acc,
      torque_cost,
      speed_cost,
      power_cost,
      head_overspeed,
      sustained,
    ],
    -1,
  )

  ratio = (tau / env._a6_tau_limits[None]).abs()
  high_load = ratio > 0.7
  env._a6_timer = torch.where(
    high_load,
    env._a6_timer + env.physics_dt,
    torch.zeros_like(env._a6_timer),
  )
  stall = (
    _a6_gate_ramp(ratio, 0.7, 0.95).square()
    * _a6_gate_ramp(env._a6_timer, 0.3, 1.0)
    * _a6_gate_ramp(0.3 - dq.abs(), 0.0, 0.3)
  )
  env._a6_load_acc += stall.amax(-1)
  env._a6_joint_acc += stall
  if hasattr(env, "_a6_eval_tau_peak"):
    active = env._a6_eval_active[:, None]
    env._a6_eval_tau_peak.copy_(
      torch.maximum(env._a6_eval_tau_peak, tau.abs() * active)
    )
    env._a6_eval_speed_peak.copy_(
      torch.maximum(env._a6_eval_speed_peak, dq.abs() * active)
    )
    env._a6_eval_power_peak.copy_(
      torch.maximum(env._a6_eval_power_peak, (tau * dq).abs() * active)
    )
    env._a6_eval_high_load_time += high_load * active * env.physics_dt
    env._a6_eval_longest_high_load.copy_(
      torch.maximum(env._a6_eval_longest_high_load, env._a6_timer * active)
    )
  env._a6_subtick += 1
  return head_vz.abs()


def _a6_phase_update(env, env_ids=None):
  """Update obstacle contact, completion, invalid-plate and separation state.

  This mirrors historical ``multiterrain.update`` and is intentionally
  idempotent per control step because both the step event and reward/metric
  terms can request it in the same environment step.
  """
  _ensure_a6_state(env)
  if env._a6_phase_tick == env.common_step_counter:
    return
  env._a6_phase_tick = env.common_step_counter

  active = env._a6_scene > 0
  which = env._a6_scene == 2
  contacts = []
  forces = []
  depths = []
  for name in ("guided_contact", "free_contact"):
    sensor_data = env.scene[name].data
    contacts.append((sensor_data.found > 0).any(-1))
    forces.append(sensor_data.force.norm(dim=-1).amax(-1))
    depths.append(sensor_data.dist.amin(-1))

  contact = torch.where(which, contacts[1], contacts[0])
  force = torch.where(which, forces[1], forces[0])
  depth = torch.where(which, depths[1], depths[0])
  env._a6_force = force
  env._a6_ever_contact |= contact & active

  clearance = _a6_completion_clearance(env)
  env._a6_clearance = clearance

  rows = torch.arange(env.num_envs, device=env.device)
  gid = env._a6_plate_geoms[which.long()]
  plate_xy = env.sim.data.geom_xpos[rows, gid, :2]
  distance = (env.scene["robot"].data.root_link_pos_w[:, :2] - plate_xy).norm(
    dim=-1
  )
  env._a6_separation_progress = torch.where(
    active & env._a6_ever_contact,
    ((distance - env._a6_best_distance) / 0.025).clamp(0.0, 1.0),
    torch.zeros_like(distance),
  )
  env._a6_best_distance = torch.maximum(env._a6_best_distance, distance)

  clear = active & env._a6_ever_contact & ~contact & (clearance >= 0.025)
  env._a6_clear_hold = torch.where(
    clear, env._a6_clear_hold + 1, torch.zeros_like(env._a6_clear_hold)
  )
  env._a6_escaped = env._a6_clear_hold >= 15
  env._a6_invalid = active & (
    (depth < -0.02)
    | (force > 1500.0)
    | ((env.episode_length_buf > 25) & ~env._a6_ever_contact)
  )


def _a6_path_update(env):
  """Compute the G, dense clearance, quiet-feet gate and L reward buffers."""
  _a6_phase_update(env)

  robot = env.scene["robot"]
  height = _a6_height(env)
  upright = _a6_upright(env)
  motion = torch.cat([robot.data.joint_pos, height[:, None], upright[:, None]], -1)
  env._a6_motion[env._a6_motion_cursor] = motion
  env._a6_motion_cursor = (env._a6_motion_cursor + 1) % A6_MOTION_WINDOW
  span = env._a6_motion.amax(0) - env._a6_motion.amin(0)
  env._a6_no_progress = (
    (span[:, : robot.data.joint_pos.shape[-1]].amax(-1) < 0.10)
    & (span[:, -2] < 0.04)
    & (span[:, -1] < 0.08)
  )

  count, score, clearance = _plate_geometry(env)
  coverage_delta = (env._a6_best_score - score).clamp_min(0.0)
  clearance_delta = (clearance - env._a6_best_clearance).clamp_min(0.0)
  env._a6_best_score = torch.minimum(env._a6_best_score, score)
  env._a6_best_clearance = torch.maximum(env._a6_best_clearance, clearance)

  support = (env.scene["path_hands"].data.found > 0).float().mean(-1)
  env._a6_support = support
  active = (
    (env._a6_scene > 0)
    & env._a6_ever_contact
    & ~env._a6_escaped
    & ~env._a6_invalid
  )
  env._a6_progress = (
    active
    * (height <= 0.90)
    * support
    * (
      (coverage_delta / 0.025).clamp(0.0, 1.0)
      + 0.5 * (clearance_delta / 0.02).clamp(0.0, 1.0)
    )
  )
  env._a6_clearance_score = (
    (env._a6_scene > 0)
    * ~env._a6_invalid
    * (
      0.85 * (1.0 - count / env._a6_initial_count.clamp_min(1.0)).clamp(0.0, 1.0)
      + 0.15 * (clearance / 0.04).clamp(0.0, 1.0)
    )
  )
  env._a6_gate = _a6_smooth_gate(height, 0.85, 1.15) * _a6_smooth_gate(
    upright, 0.70, 0.93
  )

  joint_span = span[:, : robot.data.joint_pos.shape[-1]]
  stall_gate = (1.0 - joint_span / 0.10).clamp(0.0, 1.0)
  env._a6_load = (
    env._a6_joint_acc / env.cfg.decimation * stall_gate
  ).amax(-1)


def _a6_components(env):
  """Compute the nine positive recovery tasks and eight A6 costs."""
  _ensure_a6_state(env)
  robot = env.scene["robot"]
  height = _a6_height(env)
  upright = _a6_upright(env)
  head_vz = _a6_head_velocity(env)[:, 2]
  knees = robot.data.joint_pos[:, env._a6_knee_joints]
  stage = env._a6_stage

  fallen = (height < 0.65) & (upright < 0.45)
  stage[fallen] = 0
  env._a6_hold[fallen] = 0

  load = _a6_ground_force(env, "quality_feet")[..., 2].abs().amin(-1)
  ready = (
    ((stage == 0) & (height >= 0.55) & (upright >= 0.55) & (knees.amin(-1) >= 0.8) & (head_vz.abs() <= 0.16))
    | ((stage == 1) & (height >= 0.78) & (upright >= 0.72) & (knees.amin(-1) >= 0.6) & (head_vz.abs() <= 0.18))
    | ((stage == 2) & (height >= 1.08) & (upright >= 0.85) & (knees.abs().amax(-1) < 0.8) & (load > 20.0) & (head_vz.abs() <= 0.12))
  )
  env._a6_hold = torch.where(ready, env._a6_hold + 1, torch.zeros_like(env._a6_hold))
  advance = env._a6_hold >= torch.where(stage == 2, 25, 10)
  stage.copy_(torch.where(advance, (stage + 1).clamp_max(3), stage))
  env._a6_hold[advance] = 0

  height_target = height.new_tensor((0.62, 0.86, 1.15, 1.15))[stage]
  upright_target = height.new_tensor((0.60, 0.76, 0.93, 0.93))[stage]
  knee_low = height.new_tensor((0.8, 0.6, 0.0, 0.0))[stage, None]
  knee_high = height.new_tensor((1.8, 1.6, 0.65, 0.65))[stage, None]
  knee_error = (
    (knee_low - knees).clamp_min(0.0).square()
    + (knees - knee_high).clamp_min(0.0).square()
  ).mean(-1)
  pose = torch.exp(
    -8.0 * (height_target - height).clamp_min(0.0).square()
    - 6.0 * (upright_target - upright).clamp_min(0.0).square()
    - 5.0 * knee_error
  )

  target_speed = height.new_tensor((0.10, 0.15, 0.20, 0.0))[stage]
  up_limit = height.new_tensor((0.25, 0.30, 0.30, 0.12))[stage]
  down_limit = height.new_tensor((0.16, 0.18, 0.18, 0.12))[stage]
  target_speed = target_speed * ((height_target - height) / 0.2).clamp(0.0, 1.0)
  excess = (head_vz - up_limit).clamp_min(0.0).square() + (
    -head_vz - down_limit
  ).clamp_min(0.0).square()
  velocity = torch.exp(
    -45.0 * (head_vz - target_speed).square() - 140.0 * excess
  )

  gate = _a6_smooth_gate(height, 0.85, 1.15) * _a6_smooth_gate(
    upright, 0.70, 0.93
  )
  foot_speed = (
    robot.data.body_link_lin_vel_w[:, env._a6_feet_bodies, :]
    .square()
    .sum(-1)
    .mean(-1)
  )
  raw_action = env.action_manager.action
  delta = raw_action - env._a6_last_action
  acc = delta - env._a6_last_delta
  fresh = env.episode_length_buf <= 1
  delta = torch.where(fresh[:, None], torch.zeros_like(delta), delta)
  acc = torch.where(fresh[:, None], torch.zeros_like(acc), acc)

  stage_action = height.new_tensor((0.5, 0.75, 1.0, 1.0))[stage]
  stage_acc = height.new_tensor((0.3, 0.6, 1.0, 1.0))[stage]
  env._a6_task.copy_(
    torch.stack(
      [
        pose,
        velocity,
        torch.exp(-2.0 * (1.15 - height).clamp_min(0.0).square()),
        upright.square(),
        1.0 - gate + gate * torch.exp(-20.0 * foot_speed),
        1.0 - gate + gate * torch.exp(-8.0 * robot.data.root_link_lin_vel_w.square().sum(-1)),
        torch.exp(-0.8 * robot.data.root_link_ang_vel_w.square().sum(-1)),
        torch.exp(-0.04 * robot.data.joint_vel.square().mean(-1)),
        torch.exp(-4.0 * delta.square().mean(-1)),
      ],
      -1,
    )
  )
  env._a6_cost[:, :2] = torch.stack(
    [
      delta.square().sum(-1) * stage_action,
      acc.square().sum(-1) * stage_acc,
    ],
    -1,
  )
  env._a6_cost[:, 2:] = env._a6_accum / env.cfg.decimation
  env._a6_last_action.copy_(raw_action)
  env._a6_last_delta.copy_(delta)


def _a6_update(env):
  """Refresh all cached A6 reward/termination buffers once per control step."""
  _ensure_a6_state(env)
  if env._a6_tick == env.common_step_counter:
    return
  env._a6_tick = env.common_step_counter
  _a6_path_update(env)
  _a6_components(env)


def _a6_alpha(env, dense_guide):
  """0.05 under an unescaped obstacle for G+, otherwise 1.0."""
  if not dense_guide:
    return torch.ones(env.num_envs, device=env.device)
  constrained = (env._a6_scene > 0) & ~env._a6_escaped
  return torch.where(constrained, 0.05, 1.0)


def a6_task_reward(env, index, dense_guide=True):
  _a6_update(env)
  return env._a6_task[:, index] * _a6_alpha(env, dense_guide)


def a6_cost_reward(env, index):
  _a6_update(env)
  return env._a6_cost[:, index]


def a6_path_reward(env, index):
  _a6_update(env)
  if index == 0:
    return env._a6_progress
  if index == 1:
    return env._a6_clearance_score
  if index == 2:
    foot = (
      env.scene["robot"]
      .data.body_link_lin_vel_w[:, env._a6_feet_bodies, :]
      .square()
      .sum(-1)
      .mean(-1)
    )
    return env._a6_gate * (foot / 0.1**2).clamp_max(10.0)
  return env._a6_load


def a6_completion_reward(env):
  _a6_update(env)
  return env._a6_escaped.float()


def a6_force_reward(env):
  _a6_update(env)
  active = env._a6_scene > 0
  return active.float() * ((env._a6_force - 300.0).clamp_min(0.0) / 300.0).square()


def a6_separation_reward(env):
  _a6_update(env)
  return env._a6_separation_progress


def a6_invalid(env):
  _a6_update(env)
  return env._a6_invalid


def a6_stage_metric(env):
  _a6_update(env)
  return env._a6_stage.float()


def a6_gate_metric(env):
  _a6_update(env)
  return env._a6_gate


def a6_escaped_metric(env):
  _a6_update(env)
  return env._a6_escaped.float()


def a6_clear_hold_metric(env):
  _a6_update(env)
  return env._a6_clear_hold.float()


def a6_load_metric(env):
  _a6_update(env)
  return env._a6_load


def _park_and_place_plates(
  env: "ManagerBasedRlEnv", ids: torch.Tensor
) -> None:
  """Park both boards, then place the active board under the sampled robot."""
  robot = env.scene["robot"]
  guided = env.scene["escape_obstacle"]
  free = env.scene["free_obstacle"]
  device = env.device
  count = len(ids)

  park = torch.empty((count, 7), device=device)
  park[:, :3] = env.scene.env_origins[ids] + torch.tensor(
    [20.0, 20.0, 0.1], device=device
  )
  park[:, 3:] = torch.tensor([1.0, 0.0, 0.0, 0.0], device=device)

  guided.write_mocap_pose_to_sim(park, env_ids=ids)
  guided_joint_zeros = torch.zeros(
    (count, guided.data.joint_pos.shape[-1]), device=device
  )
  guided.write_joint_state_to_sim(
    guided_joint_zeros, guided_joint_zeros.clone(), env_ids=ids
  )

  free_state = free.data.default_root_state[ids].clone()
  free_state[:, :7] = park
  free_state[:, 0] += 2.0
  free_state[:, 7:] = 0.0
  free.write_root_state_to_sim(free_state, env_ids=ids)

  # Refresh kinematics so board placement uses the final sampled robot pose.
  env.sim.forward()

  pos, ext = _robot_bounds(env)
  board = park.clone()
  board[:, :2] = robot.data.root_link_pos_w[ids, :2]
  covered = (
    (pos[ids, :, :2] - board[:, None, :2]).abs()
    < ext[ids, :, :2] + board.new_tensor(list(a6_geometry.PLATE_HALF_SIZE[:2]))
  ).all(-1)
  top = torch.where(
    covered,
    pos[ids, :, 2] + ext[ids, :, 2],
    torch.full_like(pos[ids, :, 2], -torch.inf),
  ).amax(-1)
  board[:, 2] = top + a6_geometry.PLATE_HALF_SIZE[2] + PLATE_GROUND_CLEARANCE

  for scene_id, entity in ((1, guided), (2, free)):
    choose = env._a6_scene[ids] == scene_id
    env_subset = ids[choose]
    if not len(env_subset):
      continue
    if scene_id == 1:
      entity.write_mocap_pose_to_sim(board[choose], env_ids=env_subset)
      entity_joint_zeros = torch.zeros(
        (len(env_subset), entity.data.joint_pos.shape[-1]), device=device
      )
      entity.write_joint_state_to_sim(
        entity_joint_zeros, entity_joint_zeros.clone(), env_ids=env_subset
      )
    else:
      free_subset_state = entity.data.default_root_state[env_subset].clone()
      free_subset_state[:, :7] = board[choose]
      free_subset_state[:, 7:] = 0.0
      entity.write_root_state_to_sim(free_subset_state, env_ids=env_subset)

  env.sim.forward()


def _clear_a6_runtime_buffers(
  env: "ManagerBasedRlEnv", ids: torch.Tensor
) -> None:
  """Clear per-env runtime state touched by the A6 reward/termination code."""
  for name in (
    "_a6_best_score",
    "_a6_best_clearance",
    "_a6_initial_count",
    "_a6_progress",
    "_a6_clearance_score",
    "_a6_gate",
    "_a6_load_acc",
    "_a6_load",
    "_a6_support",
    "_a6_force",
    "_a6_separation_progress",
    "_a6_best_distance",
    "_a6_clearance",
    "_a6_accum",
    "_a6_task",
    "_a6_cost",
  ):
    getattr(env, name)[ids] = 0.0
  for name in ("_a6_ever_contact", "_a6_invalid", "_a6_escaped", "_a6_no_progress"):
    getattr(env, name)[ids] = False
  env._a6_clear_hold[ids] = 0
  env._a6_hold[ids] = 0
  env._a6_timer[ids] = 0.0
  env._a6_f_effort_ms[ids] = 0.0
  env._a6_joint_acc[ids] = 0.0
  env._a6_last_action[ids] = 0.0
  env._a6_last_delta[ids] = 0.0
  env._a6_motion[:, ids] = 0.0


@requires_model_fields(
  "body_mass",
  "body_inertia",
  "actuator_gainprm",
  "actuator_biasprm",
  recompute=RecomputeLevel.set_const,
)
def a6_dynamics_reset(
  env: "ManagerBasedRlEnv",
  env_ids: torch.Tensor | None = None,
  dynamics: bool = True,
) -> None:
  """Grouped A6 dynamics randomisation for the requested reset worlds.

  Width grows linearly from +/-10% to +/-20% over the first 2000 PPO updates.
  25% of episodes remain nominal.  Body mass/inertia are scaled coherently by
  subtree and actuator Kp/Kd by joint group; effort/force limits are untouched.
  Command delay is sampled in the 0/2/4/6/8/10 ms grid.
  """
  _ensure_a6_state(env)
  if env_ids is None:
    env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.long)
  else:
    env_ids = env_ids.to(device=env.device, dtype=torch.long)
  if not len(env_ids):
    return

  device = env.device
  num_ids = len(env_ids)
  adaptation_start = int(getattr(env, "_a6_adaptation_start_step", 0))
  adaptation_steps = max(int(env.common_step_counter) - adaptation_start, 0)
  update_count = adaptation_steps / A6_STEPS_PER_UPDATE
  width = 0.1 + 0.1 * min(
    update_count / A6_DYNAMICS_WARMUP_UPDATES, 1.0
  )

  rand = lambda shape: torch.rand(  # noqa: E731
    shape, generator=env._a6_dynamics_rng, device=device
  )

  if dynamics:
    enabled = rand((num_ids,)) >= A6_NOMINAL_FRACTION
    mass_factors = 1.0 + (2.0 * rand((num_ids, 4)) - 1.0) * width
    gain_factors = 1.0 + (2.0 * rand((num_ids, 6)) - 1.0) * width
    mass_factors[~enabled] = 1.0
    gain_factors[~enabled] = 1.0
    mass_factors[:, 3] = 1.0
    lag = torch.randint(
      0, A6_DELAY_RING_LEN, (num_ids,), generator=env._a6_dynamics_rng,
      device=device,
    )
  else:
    enabled = torch.zeros(num_ids, dtype=torch.bool, device=device)
    mass_factors = torch.ones((num_ids, 4), device=device)
    gain_factors = torch.ones((num_ids, 6), device=device)
    lag = torch.zeros(num_ids, dtype=torch.long, device=device)

  if dynamics:
    plate_mass_max = 6.0 + 6.0 * min(
      adaptation_steps / PLATE_MASS_RAMP_STEPS, 1.0
    )
    plate_mass = 4.0 + (plate_mass_max - 4.0) * rand((num_ids,))
  else:
    plate_mass = torch.full((num_ids,), 6.0, device=device)

  env._a6_mass_factors[env_ids] = mass_factors
  env._a6_gain_factors[env_ids] = gain_factors
  env._a6_nominal[env_ids] = ~enabled
  env._a6_lag[env_ids] = torch.where(enabled, lag, 0)
  env._a6_plate_mass[env_ids] = plate_mass

  bodies = env._a6_bodies
  body_scale = mass_factors[:, env._a6_body_groups]
  for field in ("body_mass", "body_inertia"):
    base = env.sim.get_default_field(field)[bodies]
    if field == "body_mass":
      values = base[None] * body_scale
    else:
      values = base[None] * body_scale[:, :, None]
    getattr(env.sim.model, field)[env_ids[:, None], bodies[None, :]] = values

  robot = env.scene["robot"]
  for actuator, group_ids in zip(robot.actuators, env._a6_act_groups):
    if not isinstance(actuator, A6UnitreeActuator):
      raise RuntimeError(
        f"expected A6UnitreeActuator, got {type(actuator).__name__}"
      )
    per_target = gain_factors[:, group_ids]
    actuator.set_gains(
      env_ids,
      kp=actuator.default_stiffness[env_ids] * per_target,
      kd=actuator.default_damping[env_ids] * per_target,
      env=env,
      kp_factor=per_target,
      kd_factor=per_target,
    )

  plate_bodies = env._a6_plate_bodies
  default_plate_mass = env.sim.get_default_field("body_mass")[plate_bodies]
  plate_scale = plate_mass[:, None] / default_plate_mass[None]
  env.sim.model.body_mass[env_ids[:, None], plate_bodies[None, :]] = (
    default_plate_mass[None] * plate_scale
  )
  default_plate_inertia = env.sim.get_default_field("body_inertia")[plate_bodies]
  env.sim.model.body_inertia[env_ids[:, None], plate_bodies[None, :]] = (
    default_plate_inertia[None] * plate_scale[:, :, None]
  )


def a6_audit_dynamics(env: "ManagerBasedRlEnv") -> dict[str, bool]:
  """Validate the dynamics layer against stored per-world randomisation."""
  _ensure_a6_state(env)
  device = env.device
  ids = torch.arange(env.num_envs, device=device)

  bodies = env._a6_bodies
  body_scale = env._a6_mass_factors[:, env._a6_body_groups]
  expected_mass = env.sim.get_default_field("body_mass")[bodies][None] * body_scale
  expected_inertia = (
    env.sim.get_default_field("body_inertia")[bodies][None]
    * body_scale[:, :, None]
  )
  assert torch.allclose(env.sim.model.body_mass[:, bodies], expected_mass)
  assert torch.allclose(env.sim.model.body_inertia[:, bodies], expected_inertia)

  plate_bodies = env._a6_plate_bodies
  expected_plate_mass = env._a6_plate_mass[:, None].expand(-1, 2)
  assert torch.allclose(
    env.sim.model.body_mass[:, plate_bodies], expected_plate_mass
  )

  robot = env.scene["robot"]
  default_gainprm = env.sim.get_default_field("actuator_gainprm")
  default_biasprm = env.sim.get_default_field("actuator_biasprm")
  default_forcerange = env.sim.get_default_field("actuator_forcerange")
  for actuator, group_ids in zip(robot.actuators, env._a6_act_groups):
    per_target = env._a6_gain_factors[:, group_ids]
    assert torch.allclose(
      actuator.stiffness, actuator.default_stiffness * per_target
    )
    assert torch.allclose(
      actuator.damping, actuator.default_damping * per_target
    )
    ctrl_ids = actuator.global_ctrl_ids
    assert torch.allclose(
      env.sim.model.actuator_gainprm[:, ctrl_ids, 0],
      default_gainprm[ctrl_ids, 0] * per_target,
    )
    assert torch.allclose(
      env.sim.model.actuator_biasprm[:, ctrl_ids, 1],
      default_biasprm[ctrl_ids, 1] * per_target,
    )
    assert torch.allclose(
      env.sim.model.actuator_biasprm[:, ctrl_ids, 2],
      default_biasprm[ctrl_ids, 2] * per_target,
    )
    assert torch.allclose(
      env.sim.model.actuator_forcerange[:, ctrl_ids],
      default_forcerange[ctrl_ids],
    )

  action = env.action_manager.get_term("joint_pos")
  saved_lag = env._a6_lag.clone()
  saved_buffer = action._a6_delay_buffer.clone()
  saved_cursor = action._a6_delay_cursor
  saved_processed = action._processed_actions.clone()
  saved_episode_length = env.episode_length_buf.clone()
  saved_substep_enabled = getattr(env, "_a6_substep_enabled", True)
  env._a6_substep_enabled = False
  try:
    env.episode_length_buf.fill_(100)
    env._a6_lag.fill_(5)
    env._a6_lag[0] = 0
    action._a6_delay_buffer.zero_()
    action._processed_actions.fill_(0.123)
    action._a6_delay_cursor = 0
    for step in range(A6_DELAY_RING_LEN):
      action.apply_actions()
      target = (
        robot.data.joint_pos_target[:, action.target_ids]
        + robot.data.encoder_bias[:, action.target_ids]
      )
      assert torch.allclose(
        target[0], torch.full_like(target[0], 0.123), atol=1e-6
      )
      expected = 0.123 if step == A6_DELAY_RING_LEN - 1 else 0.0
      assert torch.allclose(
        target[1:], torch.full_like(target[1:], expected), atol=1e-6
      )
  finally:
    env._a6_lag[:] = saved_lag
    action._a6_delay_buffer[:] = saved_buffer
    action._a6_delay_cursor = saved_cursor
    action._processed_actions[:] = saved_processed
    env.episode_length_buf[:] = saved_episode_length
    env._a6_substep_enabled = saved_substep_enabled

  return {
    "body_mass_inertia_verified": True,
    "actuator_gain_verified": True,
    "effort_caps_unchanged": True,
    "delay_0_to_10ms_verified": True,
    "plate_mass_verified": True,
  }


def a6_reset(
  env: "ManagerBasedRlEnv",
  env_ids: torch.Tensor | None = None,
  bank_split: str = "train",
) -> None:
  """Exact A6 reset event.

  Runs after the flat ``reset_base``/``reset_joints`` events and overwrites
  only the requested worlds with the selected bank pose, board placement, zero
  velocities, and cleared runtime buffers.
  """
  _ensure_a6_state(env, bank_split=bank_split)
  if env_ids is None:
    env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.long)
  else:
    env_ids = env_ids.to(device=env.device, dtype=torch.long)
  if not len(env_ids):
    return

  qpos = _draw_qpos(env, env_ids)
  robot = env.scene["robot"]
  root = robot.data.default_root_state[env_ids].clone()
  root[:, :3] = qpos[:, :3] + env.scene.env_origins[env_ids]
  root[:, 3:7] = qpos[:, 3:7]
  root[:, 7:] = 0.0
  robot.write_root_state_to_sim(root, env_ids=env_ids)
  robot.write_joint_state_to_sim(
    qpos[:, 7:], torch.zeros_like(qpos[:, 7:]), env_ids=env_ids
  )

  _park_and_place_plates(env, env_ids)
  _clear_a6_runtime_buffers(env, env_ids)

  # Historical eligibility requires actual contact for both board types.
  env._a6_ever_contact[env_ids] = False

  coverage, score, clearance = _plate_geometry(env)
  env._a6_best_score[env_ids] = score[env_ids]
  env._a6_best_clearance[env_ids] = clearance[env_ids]
  env._a6_initial_count[env_ids] = coverage[env_ids].float()

  height = _a6_height(env)[env_ids]
  upright = _a6_upright(env)[env_ids]
  stage = torch.zeros_like(env_ids)
  stage = torch.where((height >= 0.55) & (upright >= 0.55), 1, stage)
  stage = torch.where((height >= 0.78) & (upright >= 0.72), 2, stage)
  stage = torch.where((height >= 1.08) & (upright >= 0.85), 3, stage)
  env._a6_stage[env_ids] = stage


def a6_reset_counts(env: "ManagerBasedRlEnv") -> dict[str, list[int]]:
  """Return deterministic allocation counts for preflight manifests."""
  return {
    "scene": torch.bincount(env._a6_scene, minlength=3).tolist(),
    "direction": torch.bincount(env._a6_direction, minlength=4).tolist(),
    "stratum": torch.bincount(env._a6_stratum, minlength=3).tolist(),
    "source": torch.bincount(env._a6_source, minlength=2).tolist(),
  }
