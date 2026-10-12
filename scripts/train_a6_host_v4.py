"""Train A6 with a checkpoint-relative forced action-scale bridge."""

from __future__ import annotations

import torch

from mjlab.envs import ManagerBasedRlEnv

from scripts import train_a6_host_v3 as v3


def relative_action_scale_bridge(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor,
  start_scale: float,
  min_scale: float = v3.FORMAL_ACTION_SCALE,
  hold_steps: int = v3.BRIDGE_HOLD_STEPS,
  duration_steps: int = v3.BRIDGE_DURATION_STEPS,
  term_name: str = "joint_pos",
) -> torch.Tensor:
  """Bridge action scale relative to the counter restored from the source."""
  if env_ids is None or len(env_ids) == 0:
    env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.int)

  bridge_start = int(getattr(env, "_a6_scale_bridge_start_step", 0))
  elapsed = max(int(env.common_step_counter) - bridge_start - hold_steps, 0)
  progress = min(elapsed / duration_steps, 1.0)
  scheduled_scale = start_scale + (min_scale - start_scale) * progress

  rescale = getattr(env, "_host_action_rescale", None)
  if rescale is None:
    rescale = torch.full(
      (env.num_envs, 1), start_scale, device=env.device, dtype=torch.float
    )
    env._host_action_rescale = rescale
  rescale[env_ids] = scheduled_scale

  action_term = env.action_manager.get_term(term_name)
  scale_cfg = action_term.cfg.scale
  scale_value = float(rescale.mean())
  if isinstance(scale_cfg, dict):
    action_term.cfg.scale = {key: scale_value for key in scale_cfg}
  else:
    action_term.cfg.scale = scale_value
  if isinstance(action_term._scale, torch.Tensor):
    action_term._scale[:] = scale_value
  else:
    action_term._scale = scale_value
  return rescale.mean()


class RelativeBridgeRunner(v3.HoSTOnPolicyRunner):
  def load(self, *args, **kwargs):
    infos = super().load(*args, **kwargs)
    env = self.env.unwrapped
    env._a6_scale_bridge_start_step = int(env.common_step_counter)
    source_scale = float(env._host_action_rescale.mean())
    env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.int)
    anchored_scale = relative_action_scale_bridge(
      env,
      env_ids,
      start_scale=source_scale,
    )
    assert abs(float(anchored_scale) - source_scale) < 1e-6
    print(
      "A6_V4_BRIDGE_ANCHOR",
      env._a6_scale_bridge_start_step,
      source_scale,
      flush=True,
    )
    return infos


def main() -> None:
  v3.scheduled_action_scale_bridge = relative_action_scale_bridge
  v3.HoSTOnPolicyRunner = RelativeBridgeRunner
  v3.main()


if __name__ == "__main__":
  main()
