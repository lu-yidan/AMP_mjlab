import tempfile
import unittest
from pathlib import Path
import torch
from src.supermimic.buffer import ReplayBuffer
from src.supermimic.mdp import flatten_state,unflatten_state,observation,reward,OBS_DIM
from src.supermimic.networks import WorldModel
from src.supermimic.train import Trainer,Config


def state(n=4):
    p=torch.zeros(n,3);p[:,2]=.75
    return p,torch.eye(3).repeat(n,1,1),torch.zeros(n,3),torch.zeros(n,3),torch.zeros(n,29),torch.zeros(n,29)


def transition(n=4,step=0,done=False):
    s=flatten_state(state(n))
    return dict(s=s,s1=s.clone(),a=torch.zeros(n,29),a_prev=torch.zeros(n,29),xi=torch.empty(n,0),
                r=torch.zeros(n),command=torch.zeros(n,3),ep_step=torch.full((n,),step,dtype=torch.int32),done=torch.full((n,),done))


class SuperMimicTest(unittest.TestCase):
    def test_state_and_reward_gradients(self):
        S=state();x=flatten_state(S)
        for a,b in zip(S,unflatten_state(x)):torch.testing.assert_close(a,b)
        a=torch.zeros(4,29,requires_grad=True);wm=WorldModel((16,))
        with torch.no_grad():wm.net[-1].weight.normal_(0,.01)
        result=wm.step(S,a,torch.empty(4,0))
        objective=reward(result,torch.ones(4,3),a,torch.zeros_like(a),torch.zeros(29)).sum()
        objective.backward()
        self.assertTrue(torch.isfinite(a.grad).all());self.assertGreater(float(a.grad.abs().sum()),0)
        self.assertEqual(observation(S,a,torch.zeros(29)).shape,(4,OBS_DIM))

    def test_windows_include_terminal_but_not_reset(self):
        b=ReplayBuffer(3,4,'cpu')
        b.add(transition(step=0,done=True));b.add(transition(step=0));b.add(transition(step=1))
        w=b.sample_windows(10,3)
        torch.testing.assert_close(w['mask'],torch.tensor([1.,0.,0.]).expand(10,-1))
        # Ring wrap also maintains temporal order.
        b.add(transition(step=2));w=b.sample_windows(10,3)
        torch.testing.assert_close(w['mask'],torch.ones(10,3))

    def test_history_does_not_cross_episode(self):
        b=ReplayBuffer(4,4,'cpu')
        for i in range(4):
            tr=transition(step=0);tr['a_prev'].fill_(i);b.add(tr)
        s,h=b.sample_starts(40,3)
        torch.testing.assert_close(h['a_prev'],s['a_prev'][:,None].expand(-1,3,-1))

    def test_imagination_and_checkpoint_roundtrip(self):
        torch.manual_seed(1)
        c=Config(num_envs=4,buffer_steps=8,history=2,horizon=2,wm_window=2,wm_batch=4,policy_batch=4,small=True,device='cpu')
        t=Trainer(c,torch.zeros(29))
        for i in range(4):t.buffer.add(transition(step=i))
        before=[p.detach().clone() for p in t.pi.parameters()]
        t.wm_update();metrics=t.policy_update()
        self.assertTrue(all(torch.isfinite(torch.tensor(v)) for v in metrics.values()))
        self.assertTrue(any(not torch.equal(a,b) for a,b in zip(before,t.pi.parameters())))
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'model.pt';names=[str(i) for i in range(29)]
            t.iteration=2;t.save(path,names);u=Trainer(c,torch.zeros(29));u.load(path,names)
            self.assertEqual(u.iteration,2)
            for k,v in t.pi.state_dict().items():torch.testing.assert_close(v,u.pi.state_dict()[k])
            with self.assertRaises(ValueError):u.load(path,names[::-1])
