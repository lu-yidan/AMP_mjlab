import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'third_party/mbrl_supermimic'))
import torch
from supermimic.env.tracking_task import StateLayout
from supermimic.utils.rotations import so3_exp
from src.amp_supermimic.blink_mirror import AMPMirror
class MirrorTests(unittest.TestCase):
 def test_involution_and_window_labels(self):
  names=['left_hip_roll_joint','right_hip_roll_joint','waist_yaw_joint']
  l=StateLayout(3);m=AMPMirror(names,l,[('foot_friction',1),('encoder_bias',3),('base_com',3)],'cpu')
  S=(torch.randn(8,4,3),so3_exp(torch.randn(8,4,3)),torch.randn(8,4,3),torch.randn(8,4,3),torch.randn(8,4,3),torch.randn(8,4,3))
  s=l.flatten(S)
  for x,y in zip(l.unflatten(m.state(m.state(s))),S):torch.testing.assert_close(x,y,atol=2e-6,rtol=2e-6)
  xi=torch.randn(8,4,7);torch.testing.assert_close(m.xi(m.xi(xi)),xi)
  a=torch.randn(8,4,3);torch.testing.assert_close(m.joints(m.joints(a)),a)
  w=dict(s=s,s1=s.clone(),a=a,xi=xi,mask=torch.rand(8,4),aux=torch.randn(8,4,3))
  v=m.windows(w)
  for k in ('mask','aux'):self.assertIs(v[k],w[k])
  for k in ('s','s1','a','xi'):torch.testing.assert_close(v[k][4:],w[k][4:])
 def test_unknown_context_rejected(self):
  with self.assertRaises(ValueError):AMPMirror(['waist_yaw_joint'],StateLayout(1),[('unknown',1)],'cpu')
if __name__=='__main__':unittest.main()
