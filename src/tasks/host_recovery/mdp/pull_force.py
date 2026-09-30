"""HoST's vertical pull force -- the second training wheel.

HoST applies an upward force on the torso while the robot is still tipped over
(``projected_gravity[:, 2] < -0.8``, i.e. "lying down"), and only after the
motors are switched on. The force decays to zero as the policy learns, so the
robot ends up getting up entirely on its own.

Config in HoST::

    class curriculum:
        pull_force = True
        force = 100      # x2 because the URDF carries a real and a virtual torso
        threshold_height = 0.9
        no_orientation = False

so the force actually applied is 200 N, roughly 60 % of a G1's weight.

NOTE: applying an external wrench every physics step needs a hook that this
port cannot verify offline -- ``mjlab`` is a pip dependency (``mjlab==1.2.0``,
Linux only) and is not installed in the environment this port was written in.
``apply_pull_force`` therefore resolves the writer at runtime and fails with an
explicit message naming the hook point. The env config ships with
``pull_force.force = 0`` so training runs without it; switch it on once the hook
is confirmed.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from .observations import UNACTUATED_STEPS

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

#: HoST ``curriculum.force``; doubled for the real + virtual torso links.
PULL_FORCE = 200.0
#: HoST ``curriculum.no_orientation``.
NO_ORIENTATION = False
#: Body the force is applied to (HoST uses its virtual torso link).
PULL_FORCE_BODY = "torso_link"


class PullForceState:
  """Per-environment pull force, HoST's ``self.force``."""

  _instance: "PullForceState | None" = None

  def __init__(self) -> None:
    self.force: torch.Tensor | None = None

  @classmethod
  def get(cls) -> "PullForceState":
    if cls._instance is None:
      cls._instance = cls()
    return cls._instance

  def init(
    self, env: ManagerBasedRlEnv, force: float = PULL_FORCE
  ) -> None:
    if self.force is not None:
      return
    self.force = torch.full(
      (env.num_envs, 1), force, device=env.device, dtype=torch.float
    )

  def decay(self, env_ids: torch.Tensor, amount: float, minimum: float = 0.0) -> None:
    if self.force is None:
      return
    self.force[env_ids] = (self.force[env_ids] - amount).clamp(min=minimum)


def init_pull_force(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None,
  force: float = PULL_FORCE,
) -> None:
  """Startup event: create the per-environment force tensor."""
  PullForceState.get().init(env, force)


def pull_force_decay(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor,
  threshold_height: float = 0.9,
  decay: float = 20.0,
  minimum: float = 0.0,
) -> torch.Tensor:
  """Curriculum: take 20 N away once the robot reliably gets up."""
  from .metrics import HoSTMetrics

  state = PullForceState.get()
  if state.force is None:
    state.init(env)
  assert state.force is not None

  if env_ids is None or len(env_ids) == 0:
    env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.int)

  metrics = HoSTMetrics.get()
  reached = False
  if metrics.initialized and metrics.last_episode_head_height is not None:
    reached = bool(torch.mean(metrics.last_episode_head_height[env_ids]) > threshold_height)
  if reached:
    state.decay(env_ids, decay, minimum)
  return state.force.mean()


def _resolve_wrench_writer(env: ManagerBasedRlEnv, body_name: str):
  """Find the entity method that writes an external wrench, if this mjlab has one."""
  asset = env.scene["robot"]
  for name in (
    "write_external_wrench_to_sim",
    "write_body_wrench_to_sim",
    "write_wrench_to_sim",
  ):
    writer = getattr(asset, name, None)
    if writer is not None:
      body_ids = getattr(asset, "find_bodies", None)
      if body_ids is not None:
        body_ids = body_ids(body_name)[0]
      else:
        body_ids = None
      return writer, body_ids
  raise NotImplementedError(
    "HoST's pull force needs a per-step external-wrench hook that was not "
    "found on this mjlab version. Expected an Entity writer such as "
    "'write_external_wrench_to_sim' / 'write_body_wrench_to_sim' "
    "(or an IsaacLab-style apply_external_force_torque action term). "
    "Wire it here once confirmed, then set pull_force.force > 0 in the env cfg."
  )


def apply_pull_force(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None = None,
  body_name: str = PULL_FORCE_BODY,
  no_orientation: bool = NO_ORIENTATION,
  force_when_down: bool = False,
  unactuated_steps: int = UNACTUATED_STEPS,
) -> None:
  """Apply HoST's upward torso force for the current step.

  Faithful to ``LeggedRobot.step``: the force is zero while the motors are off,
  and zero as well unless the base is already upright-ish (HoST gates it on
  ``projected_gravity[:, 2] < -0.8``; in both Isaac Gym and mjlab the body-frame
  gravity is ``(0, 0, -1)`` when upright and ``(0, 0, +1)`` when prone/supine).
  """
  del env_ids  # step-mode events always pass ``None`` here.
  state = PullForceState.get()
  if state.force is None:
    return

  asset = env.scene["robot"]
  force = state.force.squeeze(-1)
  actuated = (env.episode_length_buf > unactuated_steps).float()
  force = force * actuated
  if not no_orientation:
    if force_when_down:
      down = (asset.data.projected_gravity_b[:, 2] > -0.8).float()
      force = force * down
    else:
      upright = (asset.data.projected_gravity_b[:, 2] < -0.8).float()
      force = force * upright

  body_ids = asset.find_bodies(body_name)[0]
  forces = torch.zeros(env.num_envs, len(body_ids), 3, device=env.device)
  forces[:, :, 2] = force.unsqueeze(-1)
  torques = torch.zeros_like(forces)
  asset.write_external_wrench_to_sim(forces, torques, body_ids=body_ids)
