"""Fixed stage/direction/source reset quotas; AMP expert data are unchanged."""
from pathlib import Path
import hashlib
import numpy as np
import torch
from mjlab.managers.scene_entity_config import SceneEntityCfg
from .events import MotionResetManager, init_motion_loader

DIRECTIONS = ('supine', 'prone', 'left_side_down', 'right_side_down')


def make_groups(n, seed=42):
    # Integer rounding preserves equal low-direction quotas and the total size.
    low = int(n * .05); natural = round(low * .75)
    counts = [n - 2 * int(n * .1) - 4 * low, int(n * .1), int(n * .1)]
    for _ in DIRECTIONS: counts.extend([natural, low-natural])
    group = torch.empty(n, dtype=torch.long)
    order = torch.randperm(n, generator=torch.Generator().manual_seed(seed))
    start = 0
    for i, count in enumerate(counts):
        group[order[start:start+count]] = i; start += count
    return group


def init_stratified(env, env_ids, motion_dir, recovery_dir, bank_path,
                    delay_reset_env_ratio=.4, max_delay_steps=250):
    init_motion_loader(env, env_ids, motion_dir, recovery_dir,
                       delay_reset_env_ratio, max_delay_steps)
    print('[StratifiedReset] bank sha256:', hashlib.sha256(Path(bank_path).read_bytes()).hexdigest())
    bank = np.load(bank_path)
    robot = env.scene['robot']
    names = bank['joint_names'].tolist()
    mapping = [names.index(n) for n in robot.joint_names]
    q = torch.tensor(bank['qpos'], device=env.device, dtype=torch.float32)
    fields = dict(root_pos=q[:,:3], root_quat=q[:,3:7],
                  root_lin_vel=torch.zeros_like(q[:,:3]), root_ang_vel=torch.zeros_like(q[:,:3]),
                  joint_pos=q[:,7:][:,mapping], joint_vel=torch.zeros_like(q[:,7:][:,mapping]))
    stages, labels, sources = bank['stages'],bank['labels'],bank['sources']
    selectors = [(sources=='natural') & (stages==stage) for stage in ('late','middle')]
    for direction in DIRECTIONS:
        for source in ('natural','procedural'):
            selectors.append((stages=='low') & (labels==direction) & (sources==source))
    pools = []
    for select in selectors:
        if not select.any(): raise ValueError('Empty stratified reset pool')
        ids = np.flatnonzero(select)
        # Natural clips are equally likely, then frames within the clip are uniform.
        groups = [ids[bank['clips'][ids]==clip] for clip in np.unique(bank['clips'][ids])]
        pools.append((fields, [torch.tensor(g,device=env.device) for g in groups]))
    env._stratified_pools = pools
    env._stratified_groups = make_groups(env.num_envs, env.cfg.seed or 0).to(env.device)
    env.termination_manager._delay_env_mask = env._stratified_groups > 0
    print('[StratifiedReset] group counts:',torch.bincount(env._stratified_groups,minlength=11).tolist())


def reset_stratified(env, env_ids, motion_dir,
                     asset_cfg=SceneEntityCfg('robot',joint_names=('.*',))):
    if env_ids is None: env_ids=torch.arange(env.num_envs,device=env.device)
    manager=MotionResetManager.get()
    for group in range(11):
        ids=env_ids[env._stratified_groups[env_ids]==group]
        if not len(ids):continue
        if group==0:
            manager._write_reset_state(env,ids,manager.walk_run_frames[motion_dir],asset_cfg)
            continue
        fields, clips=env._stratified_pools[group-1]
        choices=torch.randint(len(clips),(len(ids),),device=env.device)
        rows=torch.empty(len(ids),dtype=torch.long,device=env.device)
        for c in choices.unique().tolist():
            mask=choices==c;pool=clips[c]
            rows[mask]=pool[torch.randint(len(pool),(int(mask.sum()),),device=env.device)]
        robot=env.scene['robot']
        root=fields['root_pos'][rows].clone();root[:,:2]=0;root+=env.scene.env_origins[ids]
        robot.write_root_link_pose_to_sim(torch.cat((root,fields['root_quat'][rows]),-1),env_ids=ids)
        robot.write_root_link_velocity_to_sim(torch.zeros(len(ids),6,device=env.device),env_ids=ids)
        robot.write_joint_state_to_sim(fields['joint_pos'][rows],fields['joint_vel'][rows],env_ids=ids)
