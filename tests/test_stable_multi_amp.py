import copy
import unittest
import torch
from tests.test_multi_amp import make_model
from rsl_rl.modules.stable_multi_amp import StableMultiAMP, FreshReplay, recovery_segments


def model(phase=False, smooth=False, cls=StableMultiAMP, **overrides):
    base = make_model(True)
    from types import SimpleNamespace
    loader = SimpleNamespace(observation_dim=4, motion_names=['fallAndGetUp1', 'walk_forward', 'jog_forward'], fps=torch.tensor([10.]))
    for attr in ('_body_pos_b_list', '_body_ori_b_list', '_body_lin_vel_b_list', '_body_ang_vel_b_list'):
        setattr(loader, attr, [torch.randn(16, 1, 1) for _ in range(3)])
    loader._command_velocity_list = [torch.zeros(16, 3), torch.tensor([[1., 0., 0.]]).repeat(16, 1), torch.tensor([[2., 0., 0.]]).repeat(16, 1)]
    h = torch.tensor([1.2, 1.1, .8, .5, .3, .4, .5, .65, .8, .95, 1., 1.1, 1.2, 1.2, 1.2, 1.2])
    loader._recovery_geometry_list = [torch.stack((h, h*0, h*0, h*0-1), -1)] * 3
    cfg = dict(base.cfg, stability_version=1, phase_experts=phase, smooth_transition=smooth, max_reuse=2., max_age=2)
    cfg.update(overrides)
    return cls(loader, cfg, [8], .1, .75, 'cpu')


class StableTests(unittest.TestCase):
    def test_fall_excluded_recovery_retained(self):
        h = torch.tensor([1.2, 1., .6, .2, .3, .6, .9, 1.1, 1.2, 1.2, 1.2])
        g = torch.stack((h, h*0, h*0, h*0-1), -1)
        keep = recovery_segments(g, 10)
        self.assertFalse(keep[:3].any())
        self.assertTrue(keep[3:9].all())

    def test_task_scale_missing_experts_and_terminal(self):
        m = model()
        a = torch.zeros(3, 4)
        # Unsupported velocity, terminal recovery, supported loco.
        r = m.reward_and_store(a, a, torch.ones(3), torch.tensor([False, True, False]),
            torch.tensor([[-1.,0,0],[0.,0,0],[1.,0,0]]), torch.tensor([False, True, False]),
            geometry=torch.tensor([[1.2,0,0,-1.]]).repeat(3,1))
        torch.testing.assert_close(r[:2], torch.tensor([.75,.75]))
        self.assertGreaterEqual(float(r[2]), .75)
        self.assertLessEqual(float(r[2]), .775)

    def test_update_budget_and_checkpoint(self):
        m = model(True)
        a = torch.randn(16,4)
        geometry = torch.tensor([[.5, 0, 0, -1.]]).repeat(16,1)
        with torch.inference_mode():
            m.reward_and_store(a,a,torch.ones(16),torch.ones(16,dtype=torch.bool),torch.zeros(16,3),torch.zeros(16,dtype=torch.bool),geometry=geometry)
        metrics = m.update(20)
        self.assertEqual(metrics['rec_updates'],4)
        self.assertLessEqual(metrics['rec_reuse'],2.)
        self.assertTrue(all(torch.isfinite(torch.tensor(v)) for v in metrics.values()))
        self.assertEqual(m.update(20)['rec_updates'],0)
        other = model(True)
        other.restore(copy.deepcopy(m.checkpoint()))
        for k,v in m.state_dict().items():
            torch.testing.assert_close(v,other.state_dict()[k])

    def test_smooth_endpoints(self):
        m = model(True,True)
        a=torch.zeros(1,4); c=torch.tensor([[1.,0,0.]])
        for n in m.networks.values():
            for p in n.parameters(): p.data.zero_()
            n.amp_linear.bias.data.fill_(1.)
        low=m.reward_and_store(a,a,torch.ones(1),torch.tensor([True]),c,torch.tensor([False]),geometry=torch.tensor([[.5,0,0,-1.]]))
        m.previous_weights = None
        high=m.reward_and_store(a,a,torch.ones(1),torch.tensor([True]),c,torch.tensor([False]),geometry=torch.tensor([[1.2,0,0,-1.]]))
        torch.testing.assert_close(low, high)
        self.assertEqual(m.replay['rec'].num_samples,1)
        self.assertEqual(m.replay['loco'].num_samples,1)

    def test_temporal_blend_and_reset_isolation(self):
        m = model(True, True)
        a=torch.zeros(1,4);c=torch.tensor([[1.,0,0.]])
        def step(height, recovering=True, done=False):
            m.reward_and_store(a,a,torch.ones(1),torch.tensor([recovering]),c,torch.tensor([done]),
                              geometry=torch.tensor([[height,0,0,-1.]]))
            return float(m.previous_weights[0])
        self.assertEqual(step(.5),1.)
        self.assertAlmostEqual(step(1.2,False),.9,places=6)
        self.assertAlmostEqual(step(1.2,False,True),.8,places=6)
        self.assertEqual(step(.5),1.)  # new episode initializes from its own geometry

    def test_expired_replay_and_wrap(self):
        b=FreshReplay(4,2,'cpu');a=torch.randn(7,2)
        b.insert(a,a,torch.zeros(7,3),torch.zeros(7,4),0)
        self.assertEqual(b.num_samples,4)
        torch.testing.assert_close(b.a,a[-4:])
        self.assertIsNone(b.sample(8,4,2))

if __name__=='__main__': unittest.main()
