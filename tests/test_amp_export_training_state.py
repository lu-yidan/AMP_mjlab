import tempfile,unittest
from types import SimpleNamespace
from unittest.mock import patch
import torch
from rsl_rl.modules import EmpiricalNormalization
from src.tasks.amp_loco.rl.runner import AMPOnPolicyRunner

class Policy(torch.nn.Module):
 def __init__(self):
  super().__init__();self.actor=torch.nn.Sequential(torch.nn.Linear(4,2))
 def act_inference(self,x):return self.actor(x)

class ExportStateTest(unittest.TestCase):
 def test_export_keeps_live_normalizer_updating(self):
  runner=object.__new__(AMPOnPolicyRunner)
  runner.alg=SimpleNamespace(policy=Policy().train())
  runner.alg.policy.runtime_cache=torch.ones(2,requires_grad=True)*2
  runner.empirical_normalization=True
  runner.obs_normalizer=EmpiricalNormalization([4]).train()
  runner.device='cpu'
  runner.obs_normalizer(torch.ones(3,4));before=runner.obs_normalizer.count.item()
  def export_copy(model,example,*args,**kwargs):
   self.assertFalse(model.training);model(example)
  with tempfile.TemporaryDirectory() as directory,patch('torch.onnx.export',side_effect=export_copy):
   runner._export_policy_to_onnx(directory)
  self.assertTrue(runner.alg.policy.training)
  self.assertTrue(runner.obs_normalizer.training)
  self.assertEqual(runner.obs_normalizer.count.item(),before)
  runner.obs_normalizer(torch.ones(3,4)*2)
  self.assertEqual(runner.obs_normalizer.count.item(),before+3)

if __name__=='__main__':unittest.main()
