"""Bounded task shaping inspired by L4, independent of termination counters."""
import torch
from mjlab.utils.lab_api.math import quat_apply


def guidance_scores(height, vertical_velocity, low_height=.9, standing_height=1.15):
    """Fade low-state guidance near standing; penalize excessive upward speed."""
    gate = ((standing_height-height)/(standing_height-low_height)).clamp(0, 1)
    speed = torch.exp(-100 * torch.relu(.10-vertical_velocity).square()
                      -25 * torch.relu(vertical_velocity-.30).square())
    progress = (height / standing_height).clamp(0, 1)
    return gate * speed, gate * progress


def recovery_guidance(env, component, low_height=.9, standing_height=1.15, recovery_only=False):
    robot = env.scene['robot']
    body = list(robot.body_names).index('torso_link')
    q = robot.data.body_link_quat_w[:,body]
    offset = quat_apply(q, torch.tensor([0.,0.,.43],device=q.device).expand(len(q),-1))
    height = robot.data.body_link_pos_w[:,body,2]+offset[:,2]-env.scene.env_origins[:,2]
    # Point velocity includes torso rotation, not just its origin's translation.
    velocity = robot.data.body_link_lin_vel_w[:,body] + torch.cross(
        robot.data.body_link_ang_vel_w[:,body],offset,dim=-1)
    scores = guidance_scores(height,velocity[:,2],low_height,standing_height)
    if component not in ('rise','height'): raise ValueError(component)
    reward = scores[0 if component=='rise' else 1]
    if recovery_only:
        reward = reward * env.command_manager.get_term('twist').recovering
    return reward
