"""EMA must smooth parameters without corrupting live weights or normalization."""
import unittest
import torch
from src.amp_supermimic.ema import PolicyEMA

class TestPolicyEMA(unittest.TestCase):
    def test_parameters_smoothed_buffers_copied_and_live_unchanged(self):
        model=torch.nn.Linear(2,1)
        model.register_buffer('running_mean',torch.zeros(2))
        model.register_buffer('count',torch.tensor(0,dtype=torch.long))
        with torch.no_grad():
            model.weight.zero_();model.bias.zero_()
        ema=PolicyEMA(model,.5)
        with torch.no_grad():
            model.weight.fill_(2);model.bias.fill_(4)
            model.running_mean.fill_(9);model.count.fill_(10)
        ema.update(model)
        self.assertTrue(torch.equal(ema.weights['weight'],torch.ones_like(model.weight)))
        self.assertTrue(torch.equal(ema.weights['bias'],torch.full_like(model.bias,2)))
        self.assertTrue(torch.equal(ema.weights['running_mean'],model.running_mean))
        self.assertEqual(ema.weights['count'].item(),10)
        self.assertTrue(torch.equal(model.weight,torch.full_like(model.weight,2)))
        ema.update(model)
        self.assertTrue(torch.equal(ema.weights['weight'],torch.full_like(model.weight,1.5)))
        restored=torch.nn.Linear(2,1)
        restored.register_buffer('running_mean',torch.zeros(2));restored.register_buffer('count',torch.tensor(0))
        restored.load_state_dict(ema.state_dict(),strict=True)
        self.assertEqual(ema.updates,2)

    def test_reject_invalid_decay(self):
        for decay in [0,1,-.1]:
            with self.assertRaises(ValueError):PolicyEMA(torch.nn.Linear(1,1),decay)

if __name__=='__main__':unittest.main()
