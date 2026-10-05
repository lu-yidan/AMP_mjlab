"""Read-only AMP evaluation: separate standing, escape, and quiet-hold gates.

Uses the checkpoint's native cap-only task and native AMP reset bank. This is
not a matched SMP benchmark: it audits success semantics on one set of AMP
trajectories. Saves exact initial states, per-step gate masks, and per-trial
substep peaks, allowing the reported rates to be recomputed.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import importlib.metadata
import json
from pathlib import Path
import random
import subprocess

import numpy as np
import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.sensor.contact_sensor import ContactMatch, ContactSensorCfg
from mjlab.managers.metrics_manager import MetricsTermCfg

from g1recovery_amp.tasks.recovery import LOW_TORQUE_CAP_ONLY_90_TASK_ID
from g1recovery_amp.tasks.recovery.mdp.a6_shared_costs import a6_head_state
from check_a6_flat_rollout import (
    _relaxed_capability_invalid_plate,
    _stable_standing_gates,
)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def sample_loads(env):
    if not hasattr(env, '_audit_alive'):
        return torch.zeros(env.num_envs, device=env.device)
    robot = env.scene['robot']
    tau, dq = robot.data.qfrc_actuator, robot.data.joint_vel
    values = torch.stack((tau.abs(), dq.abs(), (tau*dq).abs()), -1)
    env._audit_peaks = torch.maximum(env._audit_peaks, values*env._audit_alive[:, None, None])
    return values[..., 0].amax(-1)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, default=Path('checkpoints/model_22899.pt'))
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--num-envs', type=int, default=2048)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--horizon', type=float, default=20.)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--sr3', action='store_true')
    p.add_argument('--frozen-reset', type=Path)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=False)
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    cfg = load_env_cfg(LOW_TORQUE_CAP_ONLY_90_TASK_ID, play=True)
    cfg.seed = a.seed; cfg.scene.num_envs = a.num_envs
    dt = cfg.sim.mujoco.timestep*cfg.decimation
    steps = round(a.horizon/dt)
    reset = cfg.events['a6_three_scene_curriculum_reset']
    reset.params['scene_weights'] = (.5,.25,.25)
    reset.params['low_only'] = True
    cfg.episode_length_s = (steps+1)*dt
    cfg.terminations['invalid_plate'].func = _relaxed_capability_invalid_plate
    cfg.terminations['invalid_plate'].params = dict(
        max_force=5000., max_penetration=.05, max_no_contact_steps=round(10./dt))
    if a.sr3:
        assert a.horizon == 23.
        cfg.terminations.pop('invalid_plate', None)
    if a.frozen_reset:
        assert a.sr3
        # Transfer the actual robot poses by joint name; native AMP observations,
        # actuators and physics remain native and are explicitly documented.
        from g1recovery_amp.tasks.recovery.mdp import a6_resets
        frozen = dict(np.load(a.frozen_reset))
        labels = dict(np.load(a.frozen_reset.with_name('initial.npz')))
        assert len(frozen['scene']) == a.num_envs
        original_write = a6_resets._write_a6_qpos
        def frozen_write(env, ids, ignored_qpos, asset_cfg):
            names = list(frozen['joint_names'])
            order = [names.index(x) for x in env.scene['robot'].joint_names]
            local = np.concatenate((frozen['robot_pos_local'], frozen['robot_quat'],
                                    frozen['joint_pos'][:,order]),axis=1)
            qpos = torch.as_tensor(local, device=env.device)[ids]
            for target, key in (('_a6_reset_scene','scene'),('_a6_reset_direction','direction')):
                getattr(env,target)[ids] = torch.as_tensor(frozen[key],device=env.device)[ids]
            env._a6_reset_source[ids] = torch.as_tensor(labels['source'],device=env.device)[ids]
            env._a6_reset_stage[ids] = 0
            original_write(env, ids, qpos, asset_cfg)
        a6_resets._write_a6_qpos = frozen_write
    # Diagnostic only: same per-geom non-foot vertical ground-force sum as SMP.
    cfg.scene.sensors += (ContactSensorCfg(
        name='audit_other',
        primary=ContactMatch(mode='geom', pattern=r'(?!.*foot).*_collision$', entity='robot'),
        secondary=ContactMatch(mode='geom', pattern='terrain'),
        fields=('force',), reduce='netforce', num_slots=1),)
    cfg.metrics['audit_actual_loads'] = MetricsTermCfg(func=sample_loads, per_substep=True)
    env = ManagerBasedRlEnv(cfg=cfg, device=a.device)
    agent = load_rl_cfg(LOW_TORQUE_CAP_ONLY_90_TASK_ID)
    wrapper = RslRlVecEnvWrapper(env, clip_actions=agent.clip_actions)
    try:
        runner = load_runner_cls(LOW_TORQUE_CAP_ONLY_90_TASK_ID)(
            wrapper, asdict(agent), log_dir=None, device=a.device)
        runner.load(str(a.checkpoint.resolve()), load_cfg={'actor':True,'amp':True},
                    strict=True, map_location=a.device)
        policy = runner.get_inference_policy(device=a.device)
        obs = wrapper.get_observations()
        assert not cfg.observations['actor'].enable_corruption
        robot = env.scene['robot']; n = a.num_envs; dev = a.device
        cpu = lambda v: v.detach().cpu().numpy()
        initial = dict(qpos=cpu(env.sim.data.qpos), qvel=cpu(env.sim.data.qvel),
            scene=cpu(env._a6_reset_scene), direction=cpu(env._a6_reset_direction),
            source=cpu(env._a6_reset_source), bank_index=cpu(env._a6_reset_bank_index),
            stage=cpu(env._a6_reset_stage))
        np.savez_compressed(a.out/'initial.npz', **initial)
        if a.frozen_reset:
            names=list(frozen['joint_names']); order=[names.index(x) for x in robot.joint_names]
            pose_error=float(np.max(np.abs(cpu(robot.data.root_link_pos_w-env.scene.env_origins)-frozen['robot_pos_local'])))
            joint_error=float(np.max(np.abs(cpu(robot.data.joint_pos)-frozen['joint_pos'][:,order])))
            assert pose_error < 1e-5 and joint_error < 1e-6, (pose_error,joint_error)
            assert np.array_equal(initial['scene'],frozen['scene'])
            assert np.array_equal(initial['direction'],frozen['direction'])
            (a.out/'matched_reset.json').write_text(json.dumps(dict(robot_position_max_error=pose_error,
                joint_position_max_error=joint_error, manifest_sha256=sha(a.frozen_reset),
                scope='identical robot reset poses and cohorts; AMP native plate placement and dynamics')))
        scene = env._a6_reset_scene.clone()
        assert torch.all(env._a6_reset_stage == 0)
        assert torch.all(env._a6_plate_mass[scene > 0] == 6.)
        env._audit_alive = torch.ones(n, dtype=torch.bool, device=dev)
        env._audit_peaks = torch.zeros(n, 29, 3, device=dev)
        meta = dict(checkpoint=str(a.checkpoint.resolve()), checkpoint_sha256=sha(a.checkpoint),
            git_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
            script_sha256=sha(__file__), initial_file_sha256=sha(a.out/'initial.npz'),
            task=LOW_TORQUE_CAP_ONLY_90_TASK_ID, seed=a.seed, num_envs=n,
            actor_observation_shape=list(obs['actor'].shape), actor_noise=False,
            horizon_s=a.horizon, step_dt=dt, physics_dt=cfg.sim.mujoco.timestep,
            bank_path=str(env._a6_reset_bank_path), bank_sha256=sha(env._a6_reset_bank_path),
            gate_scope='SMP numeric quiet gates with AMP native head/body/contact definitions; native AMP escape; not matched SMP initial states/dynamics',
            invalid=(dict(rule='actual termination or numerical failure only; no plate history exclusion') if a.sr3 else dict(force_n=5000, penetration_m=.05, initial_no_contact_s=10)),
            frozen_reset=str(a.frozen_reset) if a.frozen_reset else None,
            other_sensor_primary_names=list(env.scene['audit_other'].primary_names),
            packages={x:importlib.metadata.version(x) for x in ('torch','mjlab','mujoco','mujoco-warp','warp-lang','rsl-rl-lib')})
        (a.out/'launch.json').write_text(json.dumps(meta,indent=2)+'\n')
        (a.out/'evaluate_success_protocols.py').write_text(Path(__file__).read_text())
        print('AUDIT_LAUNCH',json.dumps(meta),flush=True)
        names = ['basic','quiet_available','quiet_full','basic_escape','quiet_escape',
                 'quiet_smp_clear','quiet_smp_clear_valid']
        gate_names = ['height','upright','knees','base_speed','angular_speed','joint_speed','foot_speed','foot_load','stance_width','nonfoot']
        names += ['quiet_escape_without_'+k for k in gate_names]
        holds = torch.zeros(n,len(names),device=dev); best=holds.clone()
        first = torch.full((n,len(names)),-1,device=dev,dtype=torch.long)
        escaped_ever = torch.zeros(n,device=dev,dtype=torch.bool)
        invalid = escaped_ever.clone()
        smp_invalid = escaped_ever.clone()
        smp_invalid_reasons = torch.zeros(n,3,dtype=torch.bool,device=dev)
        smp_clear_hold = torch.zeros(n,device=dev,dtype=torch.long)
        gate_bits=[]; escape_bits=[]; alive_bits=[]; physical=[]; snapshots={}
        xy0=robot.data.root_link_pos_w[:,:2].clone()
        with torch.inference_mode():
            for step in range(steps):
                alive_before=env._audit_alive.clone()
                actions=policy(obs)
                assert torch.isfinite(actions).all()
                actions=torch.where(alive_before[:,None],actions,0.)
                obs,_,done,_=wrapper.step(actions)
                invalid |= alive_before & done.bool()
                env._audit_alive &= ~done.bool()
                live=env._audit_alive
                head,_=a6_head_state(env)
                upright=(-robot.data.projected_gravity_b[:,2]).clamp(0,1)
                feet=robot.data.body_link_pos_w[:,env._a6_cost_foot_ids]
                width=(feet[:,0,:2]-feet[:,1,:2]).norm(dim=-1)
                foot_load=env.scene['quality_feet'].data.force[...,2].abs().amin(-1)
                foot_vel=robot.data.body_link_lin_vel_w[:,env._a6_cost_foot_ids]
                gates=_stable_standing_gates(head_height=head,upright=upright,
                    knee_position=robot.data.joint_pos[:,env._a6_cost_knee_ids],
                    base_linear_velocity=robot.data.root_link_lin_vel_w,
                    base_angular_velocity=robot.data.root_link_ang_vel_w,
                    joint_velocity=robot.data.joint_vel, foot_linear_velocity=foot_vel,
                    minimum_foot_load=foot_load, stance_width=width)
                other=env.scene['audit_other'].data.force[...,2].abs().sum(-1)
                gates['nonfoot']=other<20.
                basic=(head>=1.15)&(upright>=.90)&(foot_load>20)
                available=gates['stable']; full=available&gates['nonfoot']
                escape=((scene==0)|env._a6_plate_escaped)&live
                escaped_ever |= escape
                sensor_g=env.scene['guided_contact'].data; sensor_f=env.scene['free_contact'].data
                free=scene==2; active=scene>0
                contact=torch.where(free,(sensor_f.found>0).any(-1),(sensor_g.found>0).any(-1))
                force=torch.where(free,sensor_f.force.norm(dim=-1).amax(-1),sensor_g.force.norm(dim=-1).amax(-1))
                depth=torch.where(free,sensor_f.dist.amin(-1),sensor_g.dist.amin(-1))
                smp_invalid_reasons |= torch.stack((force>1500,depth<-.02,
                    (env.episode_length_buf>25)&~env._a6_plate_ever_contact),-1)&active[:,None]&live[:,None]
                smp_invalid |= smp_invalid_reasons.any(-1)
                clear_now=active&env._a6_plate_ever_contact&~contact&(env._a6_plate_clearance>=.025)&live
                smp_clear_hold=torch.where(clear_now,smp_clear_hold+1,0)
                smp_clear=((scene==0)|(smp_clear_hold>=15))&live
                matrix=torch.stack([gates[k] for k in gate_names],-1)
                variants=[basic,available,full,basic&escape,full&escape,full&smp_clear,full&smp_clear&~smp_invalid]
                for i in range(len(gate_names)):
                    variants.append((matrix.sum(-1)-matrix[:,i].long()==len(gate_names)-1)&escape)
                good=torch.stack(variants,-1)&live[:,None]
                holds=torch.where(good,holds+dt,0.);best=torch.maximum(best,holds)
                newly=(holds>=1-1e-4)&(first<0);first[newly]=step
                bits=(matrix.long()*(2**torch.arange(len(gate_names),device=dev))).sum(-1)
                gate_bits.append(cpu(bits).astype(np.uint16));escape_bits.append(cpu(escape));alive_bits.append(cpu(live))
                physical.append(cpu(torch.stack((head,upright,width,foot_load,other,
                    robot.data.root_link_lin_vel_w.norm(dim=-1),robot.data.root_link_ang_vel_w.norm(dim=-1),
                    robot.data.joint_vel.square().mean(-1).sqrt(),foot_vel.norm(dim=-1).amax(-1),
                    robot.data.joint_pos[:,env._a6_cost_knee_ids].abs().amax(-1),
                    (robot.data.root_link_pos_w[:,:2]-xy0).norm(dim=-1)),-1)))
                if step+1 in (round(10/dt),round(20/dt),steps):
                    snapshots[str(round((step+1)*dt,3))]=dict(best_hold=cpu(best),first_1s_step=cpu(first),
                        escaped=cpu(escaped_ever), invalid=cpu(invalid),peaks=cpu(env._audit_peaks),
                        smp_invalid_reasons=cpu(smp_invalid_reasons))
                if (step+1)%100==0:
                    print('STEP',step+1,'basic1',int((best[:,0]>=1-1e-4).sum()),
                          'quiet_escape10',int((best[:,4]>=10-1e-4).sum()),flush=True)
        summary={}
        for horizon,values in snapshots.items():
            cohorts={}
            for scene_id,scene_name in [(None,'all'),(-1,'plates'),(0,'flat'),(1,'guided'),(2,'free')]:
                mask=np.ones(n,dtype=bool) if scene_id is None else initial['scene']>0 if scene_id==-1 else initial['scene']==scene_id
                row={'n':int(mask.sum()),'escaped_n':int(values['escaped'][mask].sum()),'invalid_n':int(values['invalid'][mask].sum())}
                row['smp_invalid_reason_counts']=values['smp_invalid_reasons'][mask].sum(axis=0).tolist()
                for k,name in enumerate(names):
                    for duration in (1,10):
                        ok=values['best_hold'][:,k]>=duration-1e-4
                        row[f'{name}_{duration}s_n']=int(ok[mask].sum())
                row['peak_p95_tau_speed_power']=np.quantile(values['peaks'][mask].max(axis=1),.95,axis=0).tolist()
                cohorts[scene_name]=row
            summary[horizon]=cohorts
            np.savez_compressed(a.out/f'per_trial_{horizon}s.npz',**initial,**values)
        np.savez_compressed(a.out/'control_trace.npz',gate_bits=np.asarray(gate_bits),
            escaped=np.asarray(escape_bits),alive=np.asarray(alive_bits),physical=np.asarray(physical),
            gate_names=np.asarray(gate_names),metric_names=np.asarray(names))
        (a.out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
        if a.sr3:
            from sr3_protocol import PROTOCOL, score, summarize, self_test
            self_test()
            x=np.asarray(physical)
            good=(x[...,0]>=1.15)&(x[...,1]>=.90)&(x[...,3]>20)
            result=score(good,np.asarray(alive_bits),dt)
            sr3=summarize(result,initial['scene'],initial['direction'])
            np.savez_compressed(a.out/'sr3_per_trial.npz',**initial,**result)
            (a.out/'sr3_summary.json').write_text(json.dumps(sr3,indent=2)+'\n')
            (a.out/'sr3_protocol.json').write_text(json.dumps(PROTOCOL,indent=2)+'\n')
            print('SR3_COMPLETE',json.dumps(sr3),flush=True)
        print('EVALUATION_COMPLETE',json.dumps(summary),flush=True)
    finally:
        wrapper.close()


if __name__=='__main__':
    main()
