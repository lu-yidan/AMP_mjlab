"""Diagnostic: run the AMP 29-DoF actor through the HoST A6 physics scene.

The AMP actor observes only its own joint/IMU/action history. Its absolute
joint target is converted to HoST's relative target every control step. This
cross-project adapter is a feasibility probe, not a HoST policy result.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from mjlab.envs import ManagerBasedRlEnv

from g1recovery_amp.g1_model import (
    G1_JOINT_NAMES, G1_RECOVERY_DEFAULT_JOINT_POS,
)
from g1recovery_amp.tasks.recovery.mdp.observations import (
    root_local_rot_tan_norm_from_quat,
)
from g1recovery_host.a6_reward import guided_exit_margin
from g1recovery_host.a6_task import host_a6_env_cfg
from g1recovery_host.model import HOST29_JOINT_NAMES


def load_actor(path: Path, device: str) -> nn.Sequential:
    state = torch.load(path, map_location=device, weights_only=False)
    actor = nn.Sequential(
        nn.Linear(480,512), nn.ELU(), nn.Linear(512,256), nn.ELU(),
        nn.Linear(256,128), nn.ELU(), nn.Linear(128,29),
    ).to(device)
    actor.load_state_dict({k.removeprefix("mlp."): v
                           for k,v in state["actor_state_dict"].items()
                           if k.startswith("mlp.")}, strict=True)
    actor.eval()
    return actor


def update_history(history: list[torch.Tensor] | None, robot, last_action):
    terms = (
        robot.data.root_link_ang_vel_b,
        root_local_rot_tan_norm_from_quat(robot.data.root_link_quat_w),
        robot.data.joint_pos,
        robot.data.joint_vel,
        last_action,
    )
    if history is None:
        history = [term[:,None,:].expand(-1,5,-1).clone() for term in terms]
    else:
        for slot, term in zip(history, terms, strict=True):
            slot[:,:-1] = slot[:,1:].clone()
            slot[:,-1] = term
    return history, torch.cat([term.reshape(len(last_action),-1) for term in history],-1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=44)
    parser.add_argument("--per-side", type=int, default=64)
    parser.add_argument("--steps", type=int, default=750)
    args = parser.parse_args()
    if args.per_side < 16 or (2*args.per_side)%32:
        parser.error("total world count must be a multiple of 32")
    if G1_JOINT_NAMES != HOST29_JOINT_NAMES:
        raise ValueError("AMP and HoST 29-DoF joint order differs")
    torch.manual_seed(args.seed)
    actor = load_actor(args.checkpoint,args.device)
    n = args.per_side
    layout = np.zeros((2*n,3),dtype=np.int64)
    layout[:,0] = 1
    layout[:n,1] = 2
    layout[n:,1] = 3
    cfg = host_a6_env_cfg(2*n,args.seed,arm="G+",edge_curriculum=True,
                          aligned_edge=True,footprint_edge=True,
                          footprint_overlap_m=.03,
                          privileged_guided_actions=False)
    cfg.events["reset_scene"].params["layout"] = layout
    cfg.auto_reset = False
    cfg.episode_length_s = args.steps*.02
    env = ManagerBasedRlEnv(cfg,device=args.device)
    try:
        env._a6_staged_escape_v5 = True
        env._a6_arm_extraction_v1 = True
        env._a6_edge_offset_m = 0.
        env.reset()
        robot = env.scene["robot"]
        term = env.action_manager.get_term("joint_pos")
        term.beta.fill_(.25)
        term.assist_force.zero_()
        default = torch.tensor(
            [G1_RECOVERY_DEFAULT_JOINT_POS.get(name, 0.0) for name in G1_JOINT_NAMES],
            device=args.device,
        )
        last_action = torch.zeros(2*n,29,device=args.device)
        entry_q = robot.data.joint_pos.clone()
        history = None
        active = torch.ones(2*n,dtype=torch.bool,device=args.device)
        escaped = torch.zeros_like(active)
        invalid = torch.zeros_like(active)
        first_escape = torch.zeros(2*n,dtype=torch.long,device=args.device)
        first_invalid = first_escape.clone()
        lift_peak = torch.zeros(2*n,device=args.device)
        progress_peak = lift_peak.clone()
        rows = torch.arange(2*n,device=args.device)
        final_height = lift_peak.clone()
        final_upright = lift_peak.clone()
        for step in range(1,args.steps+1):
            with torch.no_grad():
                history, amp_obs = update_history(history,robot,last_action)
                amp_action = actor(amp_obs).clamp(-10.,10.)
                last_action = torch.where(active[:,None],amp_action,torch.zeros_like(amp_action))
                target = default[None,:]+.25*last_action
                blend = ((step-30)/10.) if step>30 else 0.
                blend = min(1.,max(0.,blend))
                target = (1.-blend)*entry_q+blend*target
                q = robot.data.joint_pos
                command = ((target-q)/.25).clamp(-10.,10.)
                command[~active] = 0.
                _,_,terminated,truncated,_ = env.step(command)
                margin = guided_exit_margin(
                    robot.data.root_link_pos_w[:,:2],
                    env.sim.data.geom_xpos[rows,env._a6_plate_geoms[0],:2],
                )
                progress = (env._a6_initial_exit_margin-margin).clamp_min(0)
                progress_peak = torch.where(active,torch.maximum(progress_peak,progress),progress_peak)
                lift_peak = torch.where(active,torch.maximum(lift_peak,env._a6_plate_lift_peak),lift_peak)
                new_escape = active & env._a6_escaped & ~env._a6_invalid & ~escaped
                escaped |= new_escape
                first_escape = torch.where(new_escape,step,first_escape)
                new_invalid = active & env._a6_invalid & ~invalid
                invalid |= new_invalid
                first_invalid = torch.where(new_invalid,step,first_invalid)
                final_height = torch.where(active,robot.data.root_link_pos_w[:,2],final_height)
                final_upright = torch.where(active,-robot.data.projected_gravity_b[:,2],final_upright)
                done = (terminated|truncated)&active
                active &= ~done
                if done.any():
                    env.reset(env_ids=done.nonzero().squeeze(-1))
        results = {}
        for label,sl in (("guided_left_side_down",slice(0,n)),
                         ("guided_right_side_down",slice(n,2*n))):
            results[label] = {
                "escaped":int(escaped[sl].sum()),
                "invalid":int(invalid[sl].sum()),
                "escaped_then_stood_at_end":int((escaped[sl]&
                    (final_height[sl]>.6)&(final_upright[sl]>.8)).sum()),
                "max_progress_m":float(progress_peak[sl].max()),
                "median_progress_m":float(progress_peak[sl].median()),
                "median_lift_m":float(lift_peak[sl].median()),
                "first_escape_steps":first_escape[sl].cpu().tolist(),
                "first_invalid_steps":first_invalid[sl].cpu().tolist(),
            }
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps(results,indent=2),encoding="utf-8")
        print(json.dumps({k:{t:v[t] for t in ("escaped","invalid","escaped_then_stood_at_end","max_progress_m","median_progress_m","median_lift_m")}
                          for k,v in results.items()}),flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    main()
