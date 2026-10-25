from mjlab.tasks.registry import register_mjlab_task
from src.tasks.amp_loco.rl import AMPOnPolicyRunner

from .env_cfgs import (
  g1_amp_flat_env_cfg,
  g1_amp_rough_env_cfg,
)
from .rl_cfg import g1_amp_ppo_runner_cfg

register_mjlab_task(
  task_id="Unitree-G1-AMP-Rough",
  env_cfg=g1_amp_rough_env_cfg(),
  play_env_cfg=g1_amp_rough_env_cfg(play=True),
  rl_cfg=g1_amp_ppo_runner_cfg(),
  runner_cls=AMPOnPolicyRunner,
)

register_mjlab_task(
  task_id="Unitree-G1-AMP-Flat",
  env_cfg=g1_amp_flat_env_cfg(),
  play_env_cfg=g1_amp_flat_env_cfg(play=True),
  rl_cfg=g1_amp_ppo_runner_cfg(),
  runner_cls=AMPOnPolicyRunner,
)

from .rl_cfg import g1_multi_amp_ppo_runner_cfg

register_mjlab_task(
  task_id="Unitree-G1-MultiAMP-Flat",
  env_cfg=g1_amp_flat_env_cfg(),
  play_env_cfg=g1_amp_flat_env_cfg(play=True),
  rl_cfg=g1_multi_amp_ppo_runner_cfg(),
  runner_cls=AMPOnPolicyRunner,
)

from pathlib import Path
from src.tasks.amp_loco.mdp.stratified_reset import init_stratified, reset_stratified


def stratified_env_cfg(play=False):
  cfg = g1_amp_flat_env_cfg(play=play)
  cfg.events['init_motion_loader'].func = init_stratified
  cfg.events['init_motion_loader'].params['bank_path'] = str(
    Path(__file__).resolve().parents[4] / 'assets/motions/g1/reset_banks/l4_amp_train.npz')
  cfg.events['reset_from_motion'].func = reset_stratified
  return cfg


for task_id, factory in (
  ('Unitree-G1-AMP-Stratified-Flat', g1_amp_ppo_runner_cfg),
  ('Unitree-G1-MultiAMP-Stratified-Flat', g1_multi_amp_ppo_runner_cfg),
):
  rl_cfg = factory()
  rl_cfg.save_interval = 1000
  rl_cfg.experiment_name += '_stratified'
  register_mjlab_task(task_id=task_id, env_cfg=stratified_env_cfg(),
                     play_env_cfg=stratified_env_cfg(play=True),
                     rl_cfg=rl_cfg, runner_cls=AMPOnPolicyRunner)

from mjlab.managers.reward_manager import RewardTermCfg
from src.tasks.amp_loco.mdp.recovery_guidance import recovery_guidance


def guided_stratified_env_cfg(play=False):
  cfg = stratified_env_cfg(play=play)
  for component, weight in (('rise', .7), ('height', .3)):
    cfg.rewards['recovery_guidance_'+component] = RewardTermCfg(
      func=recovery_guidance, weight=weight,
      params={'component': component, 'low_height': .9, 'standing_height': 1.15})
  return cfg


guided_rl_cfg = g1_multi_amp_ppo_runner_cfg()
guided_rl_cfg.experiment_name = 'g1_multi_amp_stratified_guidance'
register_mjlab_task(
  task_id='Unitree-G1-MultiAMP-Stratified-Guidance-Flat',
  env_cfg=guided_stratified_env_cfg(), play_env_cfg=guided_stratified_env_cfg(play=True),
  rl_cfg=guided_rl_cfg, runner_cls=AMPOnPolicyRunner)

# Isolated scratch experiment; previous task definitions remain reproducible.
from src.tasks.amp_loco.mdp.tracking_commands import TrackingCommandCfg, track_xy, track_yaw, vertical_motion_cost


def tracking_stratified_env_cfg(play=False):
  cfg = stratified_env_cfg(play=play)
  cfg.commands['twist'] = TrackingCommandCfg(
    entity_name='robot', resampling_time_range=(3., 8.),
    ranges=TrackingCommandCfg.Ranges(lin_vel_x=(-1.,3.),lin_vel_y=(-1.,1.),ang_vel_z=(-1.5,1.5)),
    full_speed=play, debug_vis=True)
  cfg.curriculum.pop('command_vel',None)
  cfg.rewards['track_anchor_linear_velocity'] = RewardTermCfg(func=track_xy,weight=1.,params={'std':.5})
  cfg.rewards['track_anchor_angular_velocity'] = RewardTermCfg(func=track_yaw,weight=1.,params={'std':.5})
  cfg.rewards['vertical_motion_l2'] = RewardTermCfg(func=vertical_motion_cost,weight=-.2)
  return cfg


tracking_rl_cfg=g1_amp_ppo_runner_cfg()
tracking_rl_cfg.experiment_name='g1_amp_stratified_tracking'
tracking_rl_cfg.save_interval=1000
tracking_rl_cfg.max_iterations=100000
register_mjlab_task(task_id='Unitree-G1-AMP-Stratified-Tracking-Flat',
  env_cfg=tracking_stratified_env_cfg(),play_env_cfg=tracking_stratified_env_cfg(play=True),
  rl_cfg=tracking_rl_cfg,runner_cls=AMPOnPolicyRunner)


def dual_tracking_env_cfg(play=False, guidance=False):
  cfg = tracking_stratified_env_cfg(play=play)
  if guidance:
    for component, weight in (('rise', .7), ('height', .3)):
      cfg.rewards['recovery_guidance_'+component] = RewardTermCfg(
        func=recovery_guidance, weight=weight,
        params={'component': component, 'recovery_only': True})
  return cfg


for guided in (False, True):
  cfg = g1_multi_amp_ppo_runner_cfg()
  cfg.multi_amp.update(condition_mode='matched_velocity',
                       condition_scale=[3., 1., 1.5], match_tolerance=[.3, .2, .3])
  cfg.max_iterations = 100000
  cfg.experiment_name = 'g1_multi_amp_stratified_tracking' + ('_guidance' if guided else '')
  register_mjlab_task(
    task_id='Unitree-G1-MultiAMP-Stratified-Tracking' + ('-Guidance' if guided else '') + '-Flat',
    env_cfg=dual_tracking_env_cfg(guidance=guided),
    play_env_cfg=dual_tracking_env_cfg(play=True, guidance=guided),
    rl_cfg=cfg, runner_cls=AMPOnPolicyRunner)
