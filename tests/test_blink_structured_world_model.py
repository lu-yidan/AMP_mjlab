import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'third_party/mbrl_supermimic'))
import torch
from supermimic.env.tracking_task import StateLayout
from supermimic.utils.rotations import so3_exp
from supermimic.modules.world_model import WorldModel
from src.amp_supermimic.blink_core import BlinkStructuredWorldModel
class StructuredTests(unittest.TestCase):
 def test_target_integrate_inverse_and_gradients(self):
  torch.manual_seed(12);m=BlinkStructuredWorldModel(StateLayout(29),29,hidden_dims=(32,32))
  S=(torch.randn(7,3),so3_exp(torch.randn(7,3)*.3),torch.randn(7,3),torch.randn(7,3),torch.randn(7,29),torch.randn(7,29))
  d=(torch.randn(7,70)*.03).requires_grad_();S1=m.integrate(S,d)
  torch.testing.assert_close(m.targets(S,S1),d,atol=1e-6,rtol=1e-4)
  sum(x.square().sum() for x in S1).backward();self.assertTrue(torch.isfinite(d.grad).all());self.assertGreater(d.grad.abs().sum().item(),0)
  torch.testing.assert_close(m.state_error(S1,S),WorldModel.targets(S,S1))
 def test_zero_residual_follows_constant_velocity(self):
  m=BlinkStructuredWorldModel(StateLayout(29),29,hidden_dims=(32,32));z=torch.zeros(1,3);q=torch.zeros(1,29)
  S=(z,torch.eye(3)[None],torch.ones(1,3),z,q,torch.ones(1,29))
  T=m.integrate(S,torch.zeros(1,70));torch.testing.assert_close(T[0],torch.full((1,3),.02));torch.testing.assert_close(T[4],torch.full((1,29),.02))
if __name__=='__main__':unittest.main()
