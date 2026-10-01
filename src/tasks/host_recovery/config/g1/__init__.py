"""Unitree G1 HoST standing-up task registration."""

from mjlab.tasks.registry import register_mjlab_task
from src.tasks.host_recovery.rl import HoSTOnPolicyRunner

from .env_cfgs import unitree_g1_host_standup_env_cfg
from .rl_cfg import unitree_g1_host_standup_ppo_runner_cfg

register_mjlab_task(
  task_id="Unitree-G1-HoST-StandUp",
  env_cfg=unitree_g1_host_standup_env_cfg(),
  play_env_cfg=unitree_g1_host_standup_env_cfg(play=True),
  rl_cfg=unitree_g1_host_standup_ppo_runner_cfg(),
  runner_cls=HoSTOnPolicyRunner,
)

from src.tasks.host_recovery.pressed import pressed_env_cfg

_pressed_rl = unitree_g1_host_standup_ppo_runner_cfg()
_pressed_rl.experiment_name = "g1_host_pressed"
register_mjlab_task(
  task_id="Unitree-G1-HoST-Pressed",
  env_cfg=pressed_env_cfg(),
  play_env_cfg=pressed_env_cfg(play=True),
  rl_cfg=_pressed_rl,
  runner_cls=HoSTOnPolicyRunner,
)

from src.tasks.host_recovery.supine_smp import supine_smp_env_cfg

_supine_smp_rl = unitree_g1_host_standup_ppo_runner_cfg()
_supine_smp_rl.experiment_name = "g1_host_supine_smp"
register_mjlab_task(
  task_id="Unitree-G1-HoST-SupineSmp",
  env_cfg=supine_smp_env_cfg(),
  play_env_cfg=supine_smp_env_cfg(play=True),
  rl_cfg=_supine_smp_rl,
  runner_cls=HoSTOnPolicyRunner,
)

from src.tasks.host_recovery.prone_smp import prone_smp_env_cfg

_prone_smp_rl = unitree_g1_host_standup_ppo_runner_cfg()
_prone_smp_rl.experiment_name = "g1_host_prone_smp"
register_mjlab_task(
  task_id="Unitree-G1-HoST-ProneSmp",
  env_cfg=prone_smp_env_cfg(),
  play_env_cfg=prone_smp_env_cfg(play=True),
  rl_cfg=_prone_smp_rl,
  runner_cls=HoSTOnPolicyRunner,
)
