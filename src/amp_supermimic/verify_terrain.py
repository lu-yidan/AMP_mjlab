"""Compare the differentiable terrain surrogate to physical ray casts off-platform."""
import torch


@torch.no_grad()
def audit_terrain(task):
    env=task.env;robot=task.entity;errors=[];misses=[]
    # Inspect every generated tile, including both signs of stairs and slopes.
    origins=env.scene.terrain.terrain_origins.reshape(-1,3)
    for start in range(0,len(origins),env.num_envs):
        batch=origins[start:start+env.num_envs]
        base=batch[torch.arange(env.num_envs,device=task.device)%len(batch)]
        for offset in (0.,1.7,2.3,3.):
            pose=torch.zeros(env.num_envs,7,device=task.device)
            pose[:,:3]=base;pose[:,0]+=offset+.013;pose[:,1]+=.017;pose[:,2]+=2.;pose[:,3]=1.
            robot.write_root_link_pose_to_sim(pose)
            env.sim.forward();env.sim.sense();env.scene.update(dt=env.step_dt)
            sensor=env.scene['terrain_scan']
            R=torch.eye(3,device=task.device)[None].expand(env.num_envs,-1,-1)
            predicted,_,valid=task.terrain_map.scan(pose[:,:3],R,torch.zeros_like(base))
            real=sensor.data.pos_w[:,2:3]-sensor.data.hit_pos_w[...,2]
            actual_valid=sensor.data.distances>=0
            misses.append((valid!=actual_valid).float().mean())
            if (valid!=actual_valid).any():
                print('TERRAIN_VALIDITY',dict(offset=offset,predicted_valid=float(valid.float().mean()),actual_valid=float(actual_valid.float().mean()),sensor_pose_error=float((sensor.data.pos_w-pose[:,:3]).abs().max())),flush=True)
            errors.append((predicted-real).abs()[actual_valid&valid])
    err=torch.cat(errors)
    result=dict(points=len(err),mean_abs_m=float(err.mean()),p99_abs_m=float(torch.quantile(err,.99)),max_abs_m=float(err.max()),validity_mismatch=float(torch.stack(misses).max()))
    # 5 cm grid smooths stair discontinuities; reject errors beyond one maximum step.
    result['passed']=result['mean_abs_m']<.005 and result['p99_abs_m']<.04 and result['max_abs_m']<.08 and result['validity_mismatch']==0
    task._obs,_=env.reset()
    return result
