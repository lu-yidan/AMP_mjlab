import unittest
import torch
from types import SimpleNamespace
from src.tasks.amp_loco.mdp.recovery_guidance import recovery_guidance
from src.tasks.amp_loco.mdp.recovery_guidance import guidance_scores

class GuidanceTest(unittest.TestCase):
    def test_bounds_and_speed(self):
        h=torch.tensor([.3,.3,.3,.3,1.15,1.3])
        v=torch.tensor([-.2,.2,2.,.0,.2,.2])
        rise,height=guidance_scores(h,v)
        self.assertTrue(bool(((rise>=0)&(rise<=1)).all()))
        self.assertTrue(bool(((height>=0)&(height<=1)).all()))
        self.assertGreater(rise[1],rise[0]); self.assertGreater(rise[1],rise[2])
        self.assertEqual(rise[1].item(),1.)
        self.assertEqual(rise[-2:].sum().item(),0.)
        self.assertEqual(height[-2:].sum().item(),0.)

    def test_height_progress_below_taper(self):
        _,score=guidance_scores(torch.tensor([.2,.5,.8]),torch.zeros(3))
        self.assertTrue(bool((score[1:]>score[:-1]).all()))

    def test_locomotion_guidance_is_exactly_zero(self):
        q = torch.tensor([[[1., 0., 0., 0.]], [[1., 0., 0., 0.]]])
        data = SimpleNamespace(body_link_quat_w=q,
            body_link_pos_w=torch.tensor([[[0., 0., .2]], [[0., 0., .2]]]),
            body_link_lin_vel_w=torch.tensor([[[0., 0., .2]], [[0., 0., .2]]]),
            body_link_ang_vel_w=torch.zeros(2, 1, 3))
        class Scene(dict):
            env_origins = torch.zeros(2, 3)
        env = SimpleNamespace(scene=Scene(robot=SimpleNamespace(body_names=['torso_link'], data=data)),
            command_manager=SimpleNamespace(get_term=lambda name: SimpleNamespace(recovering=torch.tensor([False, True]))))
        for component in ('rise', 'height'):
            reward = recovery_guidance(env, component, recovery_only=True)
            self.assertEqual(reward[0].item(), 0.)
            self.assertGreater(reward[1].item(), 0.)
