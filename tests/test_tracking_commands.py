import unittest
import torch
from src.tasks.amp_loco.mdp.tracking_commands import sample_commands,recovery_transition,LIMITS

class TrackingCommandsTest(unittest.TestCase):
 def test_modes_and_limits(self):
  torch.manual_seed(42)
  modes=torch.arange(6).repeat_interleave(1000)
  for stage,(lo,hi,vy,wz) in enumerate(LIMITS):
   c=sample_commands(modes,stage)
   self.assertTrue(((c[:,0]>=lo)&(c[:,0]<=hi)).all())
   self.assertTrue((c[:,1].abs()<=vy).all());self.assertTrue((c[:,2].abs()<=wz).all())
   self.assertTrue((c[modes==0,0]>0).all());self.assertTrue((c[modes==1,0]<0).all())
   self.assertTrue((c[modes==2,0]==0).all());self.assertTrue((c[modes==2,2]==0).all())
   self.assertTrue((c[modes==3,:2]==0).all());self.assertTrue((c[modes==5]==0).all())
   self.assertTrue((c[modes==0,0]<.6).float().mean()>.3)
  self.assertTrue((c[modes==0,0]>2.8).any())
 def test_recovery_hysteresis(self):
  rec=torch.tensor([False,False,False]);hold=torch.zeros(3)
  h=torch.tensor([.3,.8,.8]);gz=torch.tensor([0.,-1.,-1.]);head=torch.tensor([.5,1.2,1.2])
  rec,hold=recovery_transition(rec,hold,h,gz,head,.02)
  self.assertEqual(rec.tolist(),[True,False,False])
  h[:]=.8;gz[:]=-1;head[:]=1.2
  for _ in range(24):rec,hold=recovery_transition(rec,hold,h,gz,head,.02)
  self.assertTrue(rec[0])
  for _ in range(2):rec,hold=recovery_transition(rec,hold,h,gz,head,.02)
  self.assertFalse(rec.any())
  head[1]=.89;rec,hold=recovery_transition(rec,hold,h,gz,head,.02)
  self.assertTrue(rec[1])

if __name__=='__main__':unittest.main()
