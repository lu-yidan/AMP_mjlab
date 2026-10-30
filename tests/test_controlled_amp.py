import copy
import unittest
import torch
from tests.test_stable_multi_amp import model
from rsl_rl.modules.controlled_amp import ControlledAMP, require_finite


def make(shared=False, balanced=True):
    return model(True, cls=ControlledAMP, shared_discriminator=shared,
                 balance_recovery=balanced, fail_on_nonfinite=True)


class ControlledTests(unittest.TestCase):
    def test_only_shared_arm_ties_parameters_and_optimizer(self):
        for shared in [False, True]:
            m = make(shared)
            self.assertEqual(m.networks['rec'] is m.networks['loco'], shared)
            self.assertEqual(m.optimizers['rec'] is m.optimizers['loco'], shared)
            self.assertIsNot(m.norms['rec'], m.norms['loco'])
            a=torch.randn(4,4); c=torch.randn(4,3)
            x=m._input('rec',a,a,c)
            self.assertEqual(x.shape, (4,11 if shared else 8))
            if shared:torch.testing.assert_close(x[:,-3:],torch.zeros(4,3))
            self.assertEqual(m._input('loco',a,a,c).shape,(4,11))

    def test_uniform_experts_ignore_requested_phase(self):
        m=make(balanced=False)
        # Fixed RNG gives identical samples for different phase requests.
        g=torch.tensor([[.3,0.,0.,-1.]]).repeat(128,1)
        torch.manual_seed(31);a,b=m._recovery_expert(g)
        g[:,0]=1.3
        torch.manual_seed(31);aa,bb=m._recovery_expert(g)
        torch.testing.assert_close(a,aa);torch.testing.assert_close(b,bb)

    def test_update_and_restore_all_arms(self):
        for shared,balanced in [(True,True),(False,False),(False,True)]:
            m=make(shared,balanced);a=torch.randn(32,4)
            rec=torch.arange(32)%2==0
            c=torch.zeros(32,3);c[~rec,0]=1.
            g=torch.tensor([[.5,0.,0.,-1.]]).repeat(32,1)
            m.reward_and_store(a,a,torch.ones(32),rec,c,torch.zeros(32,dtype=torch.bool),geometry=g)
            stats=m.update(20)
            self.assertEqual(stats['rec_updates'],4);self.assertEqual(stats['loco_updates'],4)
            self.assertTrue(all(torch.isfinite(p).all() for p in m.parameters()))
            other=make(shared,balanced);other.restore(copy.deepcopy(m.checkpoint()))
            for k,v in m.state_dict().items():torch.testing.assert_close(v,other.state_dict()[k])
            self.assertEqual(other.optimizers['rec'] is other.optimizers['loco'],shared)

    def test_full_arm_matches_original_c(self):
        torch.manual_seed(7);old=model(True)
        torch.manual_seed(7);new=make(False,True)
        for k,v in old.state_dict().items():torch.testing.assert_close(v,new.state_dict()[k])
        a=torch.randn(32,4);rec=torch.arange(32)%2==0
        c=torch.zeros(32,3);c[~rec,0]=1.
        g=torch.tensor([[.5,0.,0.,-1.]]).repeat(32,1)
        results=[]
        for m in [old,new]:
            torch.manual_seed(9)
            results.append(m.reward_and_store(a,a,torch.ones(32),rec,c,torch.zeros(32,dtype=torch.bool),geometry=g))
            m.update(2)
        torch.testing.assert_close(*results)
        for k,v in old.state_dict().items():torch.testing.assert_close(v,new.state_dict()[k])

    def test_nonfinite_input_and_gradient_abort(self):
        with self.assertRaises(FloatingPointError):require_finite('test',torch.tensor([float('nan')]))
        m=make();a=torch.randn(16,4)
        m.reward_and_store(a,a,torch.ones(16),torch.ones(16,dtype=torch.bool),torch.zeros(16,3),
                          torch.zeros(16,dtype=torch.bool),geometry=torch.tensor([[.5,0.,0.,-1.]]).repeat(16,1))
        next(m.networks['rec'].parameters()).register_hook(lambda g: g*float('nan'))
        with self.assertRaises(RuntimeError):m.update(1)

if __name__=='__main__':unittest.main()
