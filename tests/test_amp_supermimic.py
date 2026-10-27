"""Stateful MDP regression checks independent of simulation and expert files."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'third_party/mbrl_supermimic'))
import torch
from src.amp_supermimic.task import AMPTask


def fixture():
    t=object.__new__(AMPTask);t.step_dt=.02;t.default_height=torch.tensor(.78)
    names=['track_anchor_linear_velocity','track_anchor_angular_velocity','track_root_height',
        'body_ang_vel_xy_l2','is_terminated','joint_pos_limits','action_rate_l2','vertical_motion_l2']
    t.weights={n:1. for n in names};t.weights['is_terminated']=-200
    t.limits=torch.tensor([[[-2.,2.]]]).expand(1,29,2)
    t.style=lambda a,b,r,c,s,d:torch.zeros(2)
    t.frames=lambda P,c,a,noise=False:(torch.zeros(2,96),torch.zeros(2,216))
    P=dict(gravity=torch.tensor([[0.,0.,0.],[0.,0.,0.]]),height=torch.full((2,),.3),
        head=torch.full((2,),.4),xy=torch.zeros(2,2),wz=torch.zeros(2),vz=torch.zeros(2),
        root_w=torch.zeros(2,3),q=torch.zeros(2,29),amp=torch.zeros(2,195))
    carry=dict(delay=torch.tensor([249,0]),delay_mask=torch.tensor([True,False]),
        recovering=torch.ones(2,dtype=torch.bool),hold=torch.zeros(2),command=torch.zeros(2,3),
        target=torch.ones(2,3),smooth=torch.zeros(2,dtype=torch.bool),last_action=torch.zeros(2,29),
        actor_hist=torch.zeros(2,4,96),critic_hist=torch.zeros(2,4,216),
        condition=torch.zeros(2,3),supported=torch.ones(2,dtype=torch.bool))
    return t,P,carry


def test_delayed_termination_and_counter_clear():
    t,P,c=fixture()
    nxt,(_,done,_,_)=t.transition(c,P,torch.zeros(2,29),P,torch.zeros(2,3))
    assert done.tolist()==[True,True]
    assert nxt['delay'].tolist()==[0,0]
    c['delay'][0]=248
    nxt,(_,done,_,terms)=t.transition(c,P,torch.zeros(2,29),P,torch.zeros(2,3))
    assert done.tolist()==[False,True]
    assert nxt['delay'].tolist()==[249,0]
    assert terms['track_root_height'][0]>0 and terms['track_root_height'][1]==0


def test_recovery_exit_hold_and_command_ramp():
    t,P,c=fixture();P.update(height=torch.ones(2),head=torch.full((2,),1.2),gravity=torch.tensor([[0.,0.,-1.]]).expand(2,3))
    c['hold'][:]=.48;c['smooth'][:]=True
    nxt,(_,done,_,_)=t.transition(c,P,torch.zeros(2,29),P,torch.zeros(2,3))
    assert not done.any() and not nxt['recovering'].any()
    torch.testing.assert_close(nxt['command'],torch.full((2,3),.03))
    assert not nxt['delay'].any()
    c['hold'].zero_()
    nxt,_=t.transition(c,P,torch.zeros(2,29),P,torch.zeros(2,3))
    assert nxt['recovering'].all() and not nxt['command'].any()


def test_auxiliary_costs_cannot_be_positive_and_slip_needs_command():
    t,P,c=fixture()
    _,(_,_,_,terms)=t.transition(c,P,torch.zeros(2,29),P,torch.ones(2,3))
    assert all(not terms[n].any() for n in t.aux_names)
    _,(_,_,_,terms)=t.transition(c,P,torch.zeros(2,29),P,-torch.ones(2,3))
    assert not terms['foot_slip'].any()
    c['command'][:,0]=1
    _,(_,_,_,terms)=t.transition(c,P,torch.zeros(2,29),P,-torch.ones(2,3))
    assert (terms['foot_slip']==-1).all()
