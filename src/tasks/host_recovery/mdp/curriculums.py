"""HoST curriculum terms: the "training wheels".

Getting a humanoid to stand up from the floor is hard enough that HoST starts
the task easier than it really is and then progressively removes the help. Two
wheels are used:

1. ``action_scale`` -- the PD position target is
   ``dof_pos + action * action_rescale``; while a small rescaler makes the
   robot sluggish it is also much easier to control. HoST starts at
   ``control.action_scale`` (= 1.0) and, every time the resetting environments
   have reached ``threshold_height`` on average, drops it by 0.02 down to 0.25.
2. ``pull_force`` -- a vertical force of ``curriculum.force`` (100 N, doubled
   because HoST's URDF has two torso links, so 200 N ~= 60 % of G1's weight) is
   applied to the torso while the base is still tipped over. It decays by 20 N
   on the same trigger, down to 0.

Both triggers are HoST's ``update_force_curriculum``:

.. code-block:: python

    if torch.mean(self.old_headheight[env_ids]) > self.cfg.curriculum.threshold_height:
        self.force[env_ids] -= 20
        self.action_rescale[env_ids] -= 0.02

NOTE: HoST keeps ``force`` and ``action_rescale`` as per-environment tensors.
The port keeps the per-env state in :mod:`..mdp.pull_force` /
``_host_action_rescale`` but applies the decay from the mean over the resetting
environments, because mjlab's curriculum terms are called with ``env_ids`` on
reset only -- the observable behaviour (one global decay step per reset batch)
is what a single-process run of HoST produces as well. Because mjlab's reset
event clears the running metric before the curriculum runs, the episode-final
height is first copied into ``HoSTMetrics.last_episode_head_height``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from .metrics import HoSTMetrics

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

#: HoST ``curriculum.threshold_height``.
THRESHOLD_HEIGHT = 0.7
#: HoST's decay increments and floors.
ACTION_SCALE_DECAY = 0.02
ACTION_SCALE_MIN = 0.25
PULL_FORCE_DECAY = 20.0
PULL_FORCE_MIN = 0.0


def action_scale_decay(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor,
  threshold_height: float = THRESHOLD_HEIGHT,
  decay: float = ACTION_SCALE_DECAY,
  min_scale: float = ACTION_SCALE_MIN,
  term_name: str = "joint_pos",
) -> torch.Tensor:
  """Lower the action rescaler once the robot reliably gets up.

  Returns the current rescaler so it shows up in the training logs.
  """
  if env_ids is None or len(env_ids) == 0:
    env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.int)

  state = HoSTMetrics.get()
  reached = False
  if state.initialized and state.last_episode_head_height is not None:
    reached = bool(
      torch.mean(state.last_episode_head_height[env_ids]) > threshold_height
    )

  rescale = getattr(env, "_host_action_rescale", None)
  if rescale is None:
    term = env.action_manager.get_term(term_name)
    scale = term.cfg.scale
    value = float(sum(scale.values()) / len(scale)) if isinstance(scale, dict) else float(scale)
    rescale = torch.full((env.num_envs, 1), value, device=env.device)
    env._host_action_rescale = rescale  # type: ignore[attr-defined]

  if reached:
    rescale[env_ids] = (rescale[env_ids] - decay).clamp(min=min_scale)

  term = env.action_manager.get_term(term_name)
  scale_cfg = term.cfg.scale
  new_value = float(rescale.mean())
  if isinstance(scale_cfg, dict):
    term.cfg.scale = {key: new_value for key in scale_cfg}
  else:
    term.cfg.scale = new_value
  if isinstance(term._scale, torch.Tensor):
    term._scale[:] = new_value
  else:
    term._scale = new_value
  return rescale.mean()
