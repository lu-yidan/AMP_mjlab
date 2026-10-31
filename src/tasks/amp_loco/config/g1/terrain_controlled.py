"""Critic-only terrain scan pilots, built from the matched F/H controls."""
from dataclasses import fields
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, register_mjlab_task
from mjlab.envs import mdp
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.sensor import ObjRef, GridPatternCfg
from src.tasks.amp_loco.mdp.terrain_sensor import FreshTerrainRayCastSensorCfg
from mjlab.terrains import BoxFlatTerrainCfg, BoxPyramidStairsTerrainCfg, BoxInvertedPyramidStairsTerrainCfg, HfPyramidSlopedTerrainCfg
from mjlab.terrains.terrain_generator import TerrainGeneratorCfg
from src.tasks.amp_loco.mdp.terrain_tracking import TerrainTrackingCommandCfg, terrain_root_low, terrain_height_reward
from src.tasks.amp_loco.rl import AMPOnPolicyRunner


def terrain_env(base_task, play=False):
    cfg = load_env_cfg(base_task, play=play)
    cfg.scene.sensors = (*cfg.scene.sensors, FreshTerrainRayCastSensorCfg(
        name='terrain_scan', frame=ObjRef(type='body', name='pelvis', entity='robot'),
        pattern=GridPatternCfg(size=(1.6, 1.0), resolution=.1), ray_alignment='yaw',
        max_distance=30., include_geom_groups=(0,), debug_vis=False))
    # The actor stays proprioceptive; critic retains the existing four-frame history.
    cfg.observations['critic'].terms['height_scan'] = ObservationTermCfg(
        func=mdp.height_scan, params={'sensor_name':'terrain_scan'}, clip=(-20.,20.))
    cfg.scene.terrain.terrain_type='generator'
    cfg.scene.terrain.terrain_generator=TerrainGeneratorCfg(
        size=(8.,8.), border_width=10., num_rows=3, num_cols=10, curriculum=False,
        sub_terrains={
            'flat':BoxFlatTerrainCfg(proportion=.2),
            'slope':HfPyramidSlopedTerrainCfg(proportion=.2,slope_range=(.05,.2),platform_width=3.,border_width=.25),
            'slope_inverted':HfPyramidSlopedTerrainCfg(proportion=.2,slope_range=(.05,.2),platform_width=3.,border_width=.25,inverted=True),
            'stairs_inverted':BoxInvertedPyramidStairsTerrainCfg(proportion=.2,step_height_range=(.025,.075),step_width=.4,platform_width=3.,border_width=1.),
            'stairs':BoxPyramidStairsTerrainCfg(proportion=.2,step_height_range=(.025,.075),step_width=.4,platform_width=3.,border_width=1.),
        })
    cfg.scene.terrain.max_init_terrain_level=2
    old=cfg.commands['twist']
    cfg.commands['twist']=TerrainTrackingCommandCfg(**{f.name:getattr(old,f.name) for f in fields(old) if f.init})
    cfg.terminations['bad_base_height'].func=terrain_root_low
    cfg.rewards['track_root_height'].func=terrain_height_reward
    cfg.sim.nconmax=128
    return cfg


for label,base in [('I-Shared-TerrainCritic','F-Shared'),('J-Dual-TerrainCritic','H-Full')]:
    task='Unitree-G1-AMP-Controlled-'+base+'-Flat'
    agent=load_rl_cfg(task);agent.experiment_name='g1_'+label.lower()
    register_mjlab_task(task_id='Unitree-G1-AMP-'+label,env_cfg=terrain_env(task),
        play_env_cfg=terrain_env(task,True),rl_cfg=agent,runner_cls=AMPOnPolicyRunner)
