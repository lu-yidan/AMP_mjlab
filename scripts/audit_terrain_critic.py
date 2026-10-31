import json,torch,numpy as np
from pathlib import Path
import src.tasks
from mjlab.tasks.registry import load_env_cfg
from mjlab.envs import ManagerBasedRlEnv
from src.tasks.amp_loco.mdp.terrain_tracking import ground_height
cfg=load_env_cfg('Unitree-G1-AMP-J-Dual-TerrainCritic');cfg.scene.num_envs=128
cfg.seed=42
with torch.inference_mode():
 env=ManagerBasedRlEnv(cfg,device='cuda:0');obs,_=env.reset()
 scan=env.scene['terrain_scan'];r=env.scene['robot']
 print('SHAPES',{k:list(v.shape) for k,v in obs.items()},flush=True)
 print('GROUPS',[(int(g),int((env.sim.mj_model.geom_group==g).sum())) for g in range(6)],flush=True)
 result={'shapes':{k:list(v.shape) for k,v in obs.items()},'rays':scan.data.distances.shape[1],
 'reset_center_miss_fraction':float((scan.data.distances[:,93]<0).float().mean()),
 'reset_ground_origin_max_error':float((ground_height(env)-env.scene.env_origins[:,2]).abs().max())}
 assert obs['actor'].shape[1]==384
 assert obs['critic'].shape[1]==864+187*4
 assert scan.data.distances.shape[1]==187
 assert all(torch.isfinite(v).all() for v in obs.values())
 # Inspect terrain at displaced positions on the same tiles, without advancing physics.
 p=r.data.root_link_pos_w.clone();q=r.data.root_link_quat_w.clone();p[:,0]+=2.;p[:,2]+=1.
 r.write_root_link_pose_to_sim(torch.cat((p,q),-1));env.sim.forward();env.sim.sense();env.scene.update(dt=env.step_dt)
 g=ground_height(env);result['displaced_ground_origin_max_difference']=float((g-env.scene.env_origins[:,2]).abs().max())
 result['displaced_miss_fraction']=float((scan.data.distances[:,93]<0).float().mean())
 assert result['displaced_ground_origin_max_difference']>.02
 print(json.dumps(result),flush=True);Path('reports/terrain_critic_sensor_audit_20261101.json').write_text(json.dumps(result,indent=2))
 env.close()
