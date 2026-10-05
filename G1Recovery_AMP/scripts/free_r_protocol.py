"""Transfer the frozen Table IV Free-R obstacle draws into native AMP physics."""
import copy
import hashlib
import json
from pathlib import Path
import mujoco
import numpy as np
import torch
from mjlab.managers.event_manager import requires_model_fields, RecomputeLevel


def install(cfg, source):
    source = Path(source)
    frozen = dict(np.load(source))
    original = cfg.events['a6_three_scene_curriculum_reset'].func

    @requires_model_fields('body_mass', 'body_inertia', 'geom_size', 'geom_aabb',
                           'geom_rbound', recompute=RecomputeLevel.set_const)
    def reset(env, env_ids=None, **kwargs):
        original(env, env_ids, **kwargs)
        ids = torch.arange(env.num_envs, device=env.device) if env_ids is None else env_ids
        ids = ids[env._a6_reset_scene[ids] == 2]
        if not len(ids):
            return
        if not hasattr(env, '_free_r_parameters'):
            assert np.array_equal(np.flatnonzero(env._a6_reset_scene.cpu().numpy() == 2), frozen['env_ids'])
            assert np.array_equal(env._a6_reset_direction[torch.as_tensor(frozen['env_ids'],device=env.device)].cpu().numpy(), frozen['direction'])
            values = np.zeros((env.num_envs, 6), np.float32)
            values[frozen['env_ids']] = frozen['parameters']
            env._free_r_parameters = torch.as_tensor(values, device=env.device)
        par = env._free_r_parameters[ids]
        free = env.scene['free_obstacle']
        gid = env._a6_plate_geom_ids[1]
        bid = free.indexing.body_ids[-1].long()
        size = par.new_tensor([.45,.32,.035]).expand(len(ids),3).clone()
        size[:,:2] *= par[:,:2]
        sq = size.square()
        inertia = par[:,2,None]/3 * torch.stack((sq[:,1]+sq[:,2],sq[:,0]+sq[:,2],sq[:,0]+sq[:,1]),-1)
        env.sim.model.geom_size[ids,gid] = size
        env.sim.model.geom_aabb[ids,gid,0] = 0
        env.sim.model.geom_aabb[ids,gid,1] = size
        env.sim.model.geom_rbound[ids,gid] = size.norm(dim=-1)
        env.sim.model.body_mass[ids,bid] = par[:,2]
        env.sim.model.body_inertia[ids,bid] = inertia
        env._a6_plate_mass[ids] = par[:,2]
        # Same conservative bounds and 2 mm clearance as the SMP Free-R reset.
        gids = env._a6_robot_collision_geom_ids
        pos = env.sim.data.geom_xpos[ids[:,None],gids[None,:]]
        mat = env.sim.data.geom_xmat[ids[:,None],gids[None,:]]
        sz = env.sim.model.geom_size[ids[:,None],gids[None,:]]
        typ = env.sim.model.geom_type[gids]
        ext = torch.einsum('ngij,ngj->ngi',mat.abs(),sz)
        ext = torch.where((typ == int(mujoco.mjtGeom.mjGEOM_SPHERE))[None,:,None],sz[:,:,:1],ext)
        ext = torch.where((typ == int(mujoco.mjtGeom.mjGEOM_CAPSULE))[None,:,None],
                          sz[:,:,:1]+sz[:,:,1:2]*mat[:,:,:,2].abs(),ext)
        board = free.data.default_root_state[ids].clone()
        board[:,7:] = 0
        board[:,:2] = env.scene['robot'].data.root_link_pos_w[ids,:2]+par[:,3:5]
        angle=par[:,5];c=angle.cos();s=angle.sin()
        delta=pos[:,:,:2]-board[:,None,:2]
        x=c[:,None]*delta[:,:,0]+s[:,None]*delta[:,:,1]
        y=-s[:,None]*delta[:,:,0]+c[:,None]*delta[:,:,1]
        ex=c.abs()[:,None]*ext[:,:,0]+s.abs()[:,None]*ext[:,:,1]
        ey=s.abs()[:,None]*ext[:,:,0]+c.abs()[:,None]*ext[:,:,1]
        covered=(x.abs()<ex+size[:,None,0])&(y.abs()<ey+size[:,None,1])
        assert covered.any(-1).all()
        board[:,2]=torch.where(covered,pos[:,:,2]+ext[:,:,2],-torch.inf).amax(-1)+size[:,2]+.002
        board[:,3:7]=0;board[:,3]=(angle/2).cos();board[:,6]=(angle/2).sin()
        free.write_root_state_to_sim(board,env_ids=ids)
        env.sim.forward()
        from g1recovery_amp.tasks.recovery.mdp.a6_plate_state import reset_a6_plate_state
        from g1recovery_amp.tasks.recovery.mdp.a6_shared_costs import reset_a6_shared_cost_state
        reset_a6_plate_state(env,ids)
        reset_a6_shared_cost_state(env,ids)
    cfg.events['a6_three_scene_curriculum_reset'].func = reset


def audit(env, source, out):
    source=Path(source);out=Path(out);f=dict(np.load(source))
    cpu=lambda x:x.detach().cpu().numpy()
    ids=torch.where(env._a6_reset_scene==2)[0]
    gid=int(env._a6_plate_geom_ids[1]);bid=int(env.scene['free_obstacle'].indexing.body_ids[-1])
    actual=dict(env_ids=cpu(ids),parameters=cpu(env._free_r_parameters[ids]),
                half_size=cpu(env.sim.model.geom_size[ids,gid]),mass=cpu(env.sim.model.body_mass[ids,bid]),
                inertia=cpu(env.sim.model.body_inertia[ids,bid]),
                board_pose=cpu(env.scene['free_obstacle'].data.root_link_pose_w[ids]),
                qpos=cpu(env.sim.data.qpos),qvel=cpu(env.sim.data.qvel),
                mocap_pos=cpu(env.sim.data.mocap_pos),mocap_quat=cpu(env.sim.data.mocap_quat))
    for key in ('env_ids','parameters','half_size','mass','inertia'):
        assert np.allclose(actual[key],f[key],rtol=1e-6,atol=1e-7),key
    model=copy.copy(env.sim.mj_model);data=mujoco.MjData(model);minimum=[]
    for k,j in enumerate(actual['env_ids']):
        model.geom_size[gid]=actual['half_size'][k]
        model.geom_aabb.reshape(-1,2,3)[gid,0]=0
        model.geom_aabb.reshape(-1,2,3)[gid,1]=actual['half_size'][k]
        model.geom_rbound[gid]=np.linalg.norm(actual['half_size'][k])
        model.body_mass[bid]=actual['mass'][k];model.body_inertia[bid]=actual['inertia'][k]
        data.qpos[:]=actual['qpos'][j];data.qvel[:]=actual['qvel'][j]
        data.mocap_pos[:]=actual['mocap_pos'][j];data.mocap_quat[:]=actual['mocap_quat'][j]
        mujoco.mj_forward(model,data)
        minimum.append(min([0.]+[float(c.dist) for c in data.contact if gid in (c.geom1,c.geom2)]))
    assert min(minimum)>-1e-5,min(minimum)
    np.savez_compressed(out/'free_initial_parameters.npz',**actual)
    result=dict(source=str(source),source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        parameters_sha256=hashlib.sha256(actual['parameters'].tobytes()).hexdigest(),
        free_n=len(ids),all_board_parameters_match=True,initial_board_min_distance=min(minimum),
        friction=model.geom_friction[gid].tolist(),native_robot_physics=True,
        inactive_termination_exclusions=False)
    (out/'free_audit.json').write_text(json.dumps(result,indent=2)+'\n')
    print('FREE_R_AUDIT_PASS',json.dumps(result),flush=True)
