"""Differentiable velocity-task heads; adapted from SuperMimic (see LICENSE).

State is pelvis p, rotation R, world linear/angular velocities, motor-order q/qd.
Unlike upstream, goals are commanded [vx, vy, wz], not future reference frames.
"""
import torch
from .rotations import mat_to_6d, mat_to_quat, quat_to_mat, yaw_mat

NUM_J = 29
STATE_DIM = 71
INV_DIM = 71
XI_DIM = 0  # First version has nominal dynamics; no unobserved random context.
OBS_DIM = 93


def flatten_state(S):
    p, R, v, w, q, qd = S
    return torch.cat((p, mat_to_quat(R), v, w, q, qd), -1)


def unflatten_state(x):
    return x[..., :3], quat_to_mat(x[..., 3:7]), x[..., 7:10], x[..., 10:13], x[..., 13:42], x[..., 42:71]


def invariant_features(S):
    p, R, v, w, q, qd = S
    Rt = R.transpose(-1, -2)
    return torch.cat((p[..., 2:3], mat_to_6d(yaw_mat(R).transpose(-1, -2) @ R),
                      (Rt @ v.unsqueeze(-1)).squeeze(-1),
                      (Rt @ w.unsqueeze(-1)).squeeze(-1), q, qd), -1)


def observation(S, previous_action, default_q):
    _, R, _, w, q, qd = S
    return torch.cat((q-default_q, qd*.1,
        (R.transpose(-1,-2) @ w.unsqueeze(-1)).squeeze(-1)*.25,
        -R[..., 2, :], previous_action), -1)


def locomotion_readiness(S):
    """Smooth state-based reward blend; no hidden timer or command switching.

    The actor always receives the original command, including when lying down.
    No privileged height or mode gate is required in the deployed actor.
    """
    p, R, *_ = S
    height = ((p[..., 2]-.45)/.25).clamp(0,1)
    upright = ((R[..., 2, 2]-.5)/.4).clamp(0,1)
    smooth = lambda x: x.square()*(3-2*x)
    return smooth(height)*smooth(upright)


def recovery_potential(S):
    p, R, *_ = S
    return .7*(p[...,2]/.75).clamp(0,1)+.3*((R[...,2,2]+1)/2).clamp(0,1)


def is_recovery(S):
    p, R, *_ = S
    return (p[...,2]<.65)|(R[...,2,2]<.9)


def reward(S, command, action, previous_action, default_q, dt=.02,
           previous_state=None, gamma=.99):
    p, R, v, w, q, qd = S
    velocity = (yaw_mat(R).transpose(-1,-2) @ v.unsqueeze(-1)).squeeze(-1)
    xy = torch.exp(-((velocity[..., :2]-command[..., :2])**2).sum(-1)/.25)
    yaw = torch.exp(-(w[..., 2]-command[..., 2]).square()/.25)
    upright = ((R[..., 2, 2]+1)/2).clamp(0, 1)
    height = torch.exp(-((p[..., 2]-.75)/.2).square())
    stand = (command.abs().sum(-1)<.05).float()
    posture = (q-default_q).square().mean(-1)
    ready = locomotion_readiness(S)
    phi = recovery_potential(S)
    # During recovery, permit rolling/lateral motion and knee flexion. Standing
    # posture and low vertical-speed penalties must not oppose getting up.
    cost = .02*action.square().mean(-1)+.05*(action-previous_action).square().mean(-1)
    cost = cost + .002*qd.square().mean(-1)+ready*(.2*v[...,2].square()+.2*stand*posture)
    rise = torch.exp(-((v[...,2]-.2)/.4).square())
    task = ready*(xy+yaw+.5*height)+(1-ready)*(2*phi+.25*rise)+.5*upright
    result = dt*(task-cost)
    if previous_state is not None:
        # Same potential shaping in real and imagined transitions. Zero terminal
        # potential prevents a catastrophic termination paying progress reward.
        phi_next = phi*(1-termination(S,soft=True))
        result = result + 2*(gamma*phi_next-recovery_potential(previous_state))
    return result


def termination(S, soft=False):
    """Falling is NOT terminal. Allow recovery until the episode time limit.

    Only gross physical escape below the floor or above 3m is terminal.
    Non-finite states are rejected by the simulator adapter, not fed to replay.
    """
    p, *_ = S
    if soft:
        return 1-(1-torch.sigmoid((-.2-p[...,2])/.02))*(1-torch.sigmoid((p[...,2]-3.)/.05))
    return (p[...,2]<-.2)|(p[...,2]>3.)


def critic_input(S, command, previous_action):
    # Action-rate cost makes the previous action part of the value state.
    return torch.cat((invariant_features(S), command, previous_action), -1)
