"""Movable chest/pelvis load baseline; SMP-inspired task shaping, no SMP prior.

Reference: tholin-1007/smp, commit 0e67286fe7df77a73740d237ef36b109136552b6.
Privileged contact and object state are used only for rewards and diagnostics.
"""
import os
import mujoco
import torch
from mjlab.entity import EntityCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.metrics_manager import MetricsTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.utils.lab_api.math import quat_apply
from src.assets.robots.unitree_g1.g1_23dof_constants import get_spec
from .config.g1.env_cfgs import unitree_g1_host_standup_env_cfg


def robot_spec():
    spec = get_spec()
    spec.body('torso_link').add_site(name='pressed_head', pos=(0, 0, 0.43))
    return spec


def load_spec():
    mass = float(os.environ.get('HOST_LOAD_MASS', '12'))
    if not 1 <= mass <= 40:
        raise ValueError('HOST_LOAD_MASS must be in [1, 40] kg')
    return mujoco.MjSpec.from_string(f'''<mujoco><worldbody>
      <body name="load" pos="0 0 2"><freejoint/>
        <geom name="load_box" type="box" size="0.20 0.22 0.06"
          mass="{mass}" friction="0.7 0.01 0.001" condim="3"
          contype="1" conaffinity="1" rgba="0.8 0.25 0.12 1"/>
      </body></worldbody></mujoco>''')


def reset_load(env, env_ids):
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    if not hasattr(env, '_pressed_pending'):
        env._pressed_pending = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
        env._pressed_chest = torch.zeros_like(env._pressed_pending)
        env._pressed_loaded = torch.zeros_like(env._pressed_pending)
    env._pressed_pending[env_ids] = True
    env._pressed_loaded[env_ids] = False
    target = os.environ.get('HOST_LOAD_TARGET', 'mixed')
    if target not in ('chest', 'pelvis', 'mixed'):
        raise ValueError('HOST_LOAD_TARGET must be chest, pelvis or mixed')
    env._pressed_chest[env_ids] = (torch.rand(len(env_ids),device=env.device) < 0.5
                                  if target == 'mixed' else target == 'chest')
    pose = torch.zeros(len(env_ids), 7, device=env.device)
    pose[:, :3] = env.scene.env_origins[env_ids]
    pose[:, 2] += 2.0
    pose[:, 3] = 1
    env.scene['load'].write_root_link_pose_to_sim(pose, env_ids=env_ids)
    env.scene['load'].write_root_link_velocity_to_sim(torch.zeros(len(env_ids),6,device=env.device),env_ids=env_ids)


def load_force(env):
    force = env.scene['pressed_contact'].data.force
    return torch.linalg.vector_norm(force,dim=-1).sum(dim=-1)


def place_and_track_load(env, env_ids=None):
    """Place once after reset FK; subsequently only physics moves the box."""
    ids = env._pressed_pending.nonzero(as_tuple=False).flatten()
    robot = env.scene['robot']
    if ids.numel():
        chest = env._pressed_chest[ids]
        torso = robot.find_bodies('torso_link')[0][0]
        pelvis = robot.find_bodies('pelvis')[0][0]
        bodies = torch.where(chest, torso, pelvis)
        offsets = torch.zeros(len(ids),3,device=env.device)
        offsets[:,0] = torch.where(chest, 0.01, 0.0)
        offsets[:,2] = torch.where(chest, 0.14, -0.08)
        pos = robot.data.body_link_pos_w[ids,bodies] + quat_apply(robot.data.body_link_quat_w[ids,bodies],offsets)
        # Supine start: capsule axis is horizontal. Put box 2 cm above the
        # selected capsule/sphere; both bodies settle during the motors-off phase.
        pos[:,2] += torch.where(chest, 0.09, 0.07) + 0.06 + 0.02
        pose = torch.zeros(len(ids),7,device=env.device)
        pose[:,:3] = pos; pose[:,3] = 1
        env.scene['load'].write_root_link_pose_to_sim(pose, env_ids=ids)
        env.scene['load'].write_root_link_velocity_to_sim(torch.zeros(len(ids),6,device=env.device),env_ids=ids)
        env._pressed_pending[ids] = False
    # Only count a real loaded condition near motor activation. Passive box
    # fall-off episodes cannot earn release/success rewards.
    eligible = (env.episode_length_buf >= 100) & (env.episode_length_buf <= 130)
    env._pressed_loaded |= eligible & (load_force(env) > 20)


def free_gate(env):
    return torch.exp(-load_force(env) / 30.0)


def smp_task_reward(env):
    """Reimplement SMP getup's two scalar shapes, with contact feasibility gate."""
    robot = env.scene['robot']
    idx = robot.find_sites('pressed_head')[0][0]
    z = robot.data.site_pos_w[:,idx,2] - env.scene.env_origins[:,2]
    vz = robot.data.site_lin_vel_w[:,idx,2]
    height = torch.exp(-torch.clamp(1.1-z,min=0).square())
    upward = torch.where(z < 0.9, torch.exp(-100*torch.clamp(0.25-vz,min=0).square()), torch.ones_like(z))
    active = (env.episode_length_buf > 130).float()
    return active * (0.3*height + 0.7*free_gate(env)*upward)


def escape_reward(env):
    robot = env.scene['robot']
    torso = robot.find_bodies('torso_link')[0][0]
    pelvis = robot.find_bodies('pelvis')[0][0]
    body = torch.where(env._pressed_chest,torso,pelvis)
    idx = torch.arange(env.num_envs,device=env.device)
    delta = robot.data.body_link_pos_w[idx,body,:2] - env.scene['load'].data.root_link_pos_w[:,:2]
    separation = (torch.linalg.vector_norm(delta,dim=-1)/0.45).clamp(0,1)
    return (env.episode_length_buf > 130).float()*env._pressed_loaded.float()*separation*free_gate(env)


def released_standing(env):
    robot = env.scene['robot']
    idx = robot.find_sites('pressed_head')[0][0]
    high = robot.data.site_pos_w[:,idx,2] - env.scene.env_origins[:,2] > 1.0
    upright = robot.data.projected_gravity_b[:,2] < -0.8
    return ((escape_reward(env)>0.8) & high & upright).float()


def loaded_fraction(env):
    return env._pressed_loaded.float()


def pressed_env_cfg(play=False):
    cfg = unitree_g1_host_standup_env_cfg(play=play)
    cfg.scene.entities['robot'].spec_fn = robot_spec
    cfg.scene.entities['load'] = EntityCfg(spec_fn=load_spec)
    cfg.scene.sensors = tuple(cfg.scene.sensors) + (ContactSensorCfg(
        name='pressed_contact', primary=ContactMatch(mode='body',pattern=('torso_link','pelvis'),entity='robot'),
        secondary=ContactMatch(mode='geom',pattern='load_box',entity='load'),
        fields=('found','force'), reduce='netforce'),)
    cfg.events['reset_base'].params.update(posture='supine',height=None,xy_range=0.0)
    cfg.events['reset_load'] = EventTermCfg(func=reset_load,mode='reset')
    cfg.events['place_load'] = EventTermCfg(func=place_and_track_load,mode='step')
    cfg.events.pop('init_pull_force',None); cfg.events.pop('apply_pull_force',None)
    cfg.curriculum = {}
    cfg.actions['joint_pos'].scale = 0.25
    cfg.episode_length_s = 20.0
    cfg.rewards['smp_task_shapes'] = RewardTermCfg(func=smp_task_reward,weight=1.0)
    cfg.rewards['escape_load'] = RewardTermCfg(func=escape_reward,weight=2.0)
    cfg.rewards['released_standing'] = RewardTermCfg(func=released_standing,weight=2.0)
    cfg.metrics['load_contact_force'] = MetricsTermCfg(func=load_force)
    cfg.metrics['loaded_fraction'] = MetricsTermCfg(func=loaded_fraction)
    cfg.metrics['released_standing'] = MetricsTermCfg(func=released_standing)
    return cfg
