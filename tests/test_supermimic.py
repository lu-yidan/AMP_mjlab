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
            u.reset_metadata={'sha256':'wrong-bank'}
            with self.assertRaisesRegex(ValueError,'bank mismatch'):u.load(path,names)
            u.reset_metadata={}
            ck=torch.load(path,weights_only=False);ck['format']='supermimic_velocity_v1';torch.save(ck,path)
            with self.assertRaisesRegex(ValueError,'Incompatible'):u.load(path,names)

    def test_fallen_states_survive_and_progress_is_rewarded(self):
        from src.supermimic.mdp import termination,locomotion_readiness,is_recovery
        S=list(state());S[0][:,2]=.2
        S[1][:]=torch.diag(torch.tensor([1.,-1.,-1.]))
        self.assertTrue(is_recovery(S).all());self.assertFalse(termination(S).any())
        self.assertTrue((locomotion_readiness(S)==0).all())
        up=[x.clone() for x in S];up[0][:,2]=.4;up[1][:]=torch.eye(3)
        z=torch.zeros(4,29);cmd=torch.ones(4,3)
        stay=reward(S,cmd,z,z,z[0],previous_state=S)
        rise=reward(up,cmd,z,z,z[0],previous_state=S)
        self.assertTrue((rise>stay).all())
        up[0][:,2]=-1;self.assertTrue(termination(up).all())

    def test_balanced_replay_with_missing_class_fallback(self):
        b=ReplayBuffer(4,4,'cpu')
        for i in range(4):
            tr=transition(step=i);tr['s'][0,2]=.2;b.add(tr)
        w=b.sample_windows(100,2,.5)
        self.assertEqual(int(w['recovery'][:,0].sum()),50)
        s,h=b.sample_starts(100,3,.5)
        self.assertEqual(int(s['recovery'].sum()),50)
        b=ReplayBuffer(4,4,'cpu');b.add(transition())
        s,_=b.sample_starts(100,3,.5)
        self.assertFalse(s['recovery'].any())

    def test_reset_bank_mapping_and_distribution(self):
        import numpy as np
        from src.supermimic.reset_bank import ResetBank,DEFAULT_BANK
        with np.load(DEFAULT_BANK) as data:
            names=data['joint_names'].tolist();q=torch.tensor(data['qpos'],dtype=torch.float32)
        bank=ResetBank(DEFAULT_BANK,names[::-1],'cpu')
        torch.testing.assert_close(bank.qpos[:,7:],q[:,7:].flip(-1))
        torch.manual_seed(42);groups,ids=bank.sample(20000)
        self.assertLess(abs(float((groups==0).float().mean())-.3),.02)
        self.assertLess(abs(float((groups>=3).float().mean())-.4),.02)
        self.assertTrue((ids[groups==0]==-1).all())
        self.assertEqual(len(groups.unique()),11)

    def test_upside_down_world_model_has_finite_gradients(self):
        S=list(state());S[0][:,2]=.2;S[1][:]=torch.diag(torch.tensor([1.,-1.,-1.]))
        wm=WorldModel((16,));a=torch.zeros(4,29,requires_grad=True)
        with torch.no_grad():wm.net[-1].weight.normal_(0,.01)
        nxt=wm.step(S,a,torch.empty(4,0))
        reward(nxt,torch.ones(4,3),a,a.detach(),torch.zeros(29),previous_state=S).sum().backward()
        self.assertTrue(torch.isfinite(a.grad).all())

    def test_speed_curriculum_requires_sustained_competence(self):
        from src.supermimic.curriculum import SpeedCurriculum
        c=SpeedCurriculum(5,1)
        c.add(steps=10000,upright=1000,moving=5000,tracking=500,episodes=300,length=300)
        c.update();self.assertEqual(c.speed,1)
        for _ in range(2):
            c.add(steps=10000,upright=9500,moving=5000,tracking=4500,episodes=300,length=5700)
            c.update()
        self.assertEqual(c.speed,1.5)
        d=SpeedCurriculum(5,1);d.load_state_dict(c.state_dict());self.assertEqual(d.speed,1.5)
        for _ in range(30):
            c.add(steps=10000,upright=9500,moving=5000,tracking=4500,episodes=300,length=5700);c.update()
        self.assertEqual(c.speed,5)

    def test_monitor_recovery_denominator_and_tracking(self):
        from src.supermimic.metrics import TrainingMetrics
        m=TrainingMetrics(4,'cpu',.5)
        low=list(state());low[0][:,2]=.2;up=state()
        for step in range(2):
            tr=transition(step=step);tr['terminated']=torch.zeros(4,dtype=torch.bool)
            tr['command'][:,0]=1
            m.observe(tr,low if step==0 else up,up)
        r=m.flush()
        self.assertEqual(r['Recovery/success_rate'],1.)
        self.assertEqual(r['Recovery/successes'],4)
        self.assertEqual(r['Recovery/mean_success_seconds'],1.)
        self.assertEqual(r['Tracking/vx_mae_moving'],1.)
        tr=transition(step=0,done=True);tr['terminated']=torch.zeros(4,dtype=torch.bool)
        m.observe(tr,low,low);r=m.flush()
        self.assertEqual(r['Recovery/success_rate'],0.)
        self.assertEqual(r['Recovery/failures'],4)
        self.assertNotIn('Recovery/mean_success_seconds',r)
        self.assertEqual(r['Survival/mean_episode_seconds'],.5)
