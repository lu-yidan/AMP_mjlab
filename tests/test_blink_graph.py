"""GPU regression: refreshed input/statistics/LR must remain live after capture."""
import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'third_party/mbrl_supermimic'))
import torch
from supermimic.modules.normalizer import RunningNorm
from src.amp_supermimic.blink_graph import UpdateGraph
class GraphTests(unittest.TestCase):
 def test_normalizer_preserves_addresses(self):
  n=RunningNorm(5);ptr=[x.data_ptr() for x in n.buffers()]
  n.update(torch.randn(64,5));self.assertEqual(ptr,[x.data_ptr() for x in n.buffers()])
 @unittest.skipUnless(torch.cuda.is_available(),'CUDA required')
 def test_updated_inputs_statistics_and_lr(self):
  torch.manual_seed(42)
  p=torch.nn.Linear(5,2).cuda();q=torch.nn.Linear(5,2).cuda();q.load_state_dict(p.state_dict())
  n=RunningNorm(5).cuda();x=torch.randn(16,5,device='cuda');lr=torch.tensor(.01,device='cuda')
  opt=torch.optim.Adam(p.parameters(),lr=lr,fused=True,capturable=True)
  ref=torch.optim.Adam(q.parameters(),lr=.01,fused=True)
  def step():
   loss=p(n(x)).square().mean();opt.zero_grad(set_to_none=True);loss.backward();opt.step();return {'loss':loss.detach()}
  graph=UpdateGraph(step,'regression')
  for i in range(8):
   x.copy_(torch.randn_like(x)+i);n.update(x);lr.fill_(.01/(i+1));ref.param_groups[0]['lr']=.01/(i+1)
   loss=q(n(x)).square().mean();ref.zero_grad(set_to_none=True);loss.backward();ref.step()
   out=graph();torch.testing.assert_close(out['loss'],loss,atol=2e-5,rtol=2e-5)
   for a,b in zip(p.parameters(),q.parameters()):torch.testing.assert_close(a,b,atol=2e-5,rtol=2e-5)
  self.assertIsNotNone(graph.graph)
if __name__=='__main__':unittest.main()
