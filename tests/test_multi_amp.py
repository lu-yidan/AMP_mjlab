import copy
import unittest
from types import SimpleNamespace
import torch
from rsl_rl.modules.multi_amp import MultiAMP, recovery_gate


def make_model(vector=False):
    loader = SimpleNamespace(observation_dim=4, motion_names=['fallAndGetUp1', 'walk_forward', 'jog_forward'])
    for attr in ('_body_pos_b_list', '_body_ori_b_list', '_body_lin_vel_b_list', '_body_ang_vel_b_list'):
        setattr(loader, attr, [torch.randn(8, 1, 1) for _ in range(3)])
    cfg = dict(speed_max=3., batch_size=8, replay_size=32, learning_rate=1e-3)
    if vector:
        loader._command_velocity_list = [torch.zeros(8, 3), torch.tensor([[1., 0., 0.]]).repeat(8, 1), torch.tensor([[2., 0., 0.]]).repeat(8, 1)]
        cfg.update(condition_mode='matched_velocity', condition_scale=[3., 1., 1.5], match_tolerance=[.3, .2, .3])
    return MultiAMP(loader, cfg, [8], .1, .75, 'cpu')


class MultiAMPTest(unittest.TestCase):
    def test_height_routes_upright_low_state(self):
        gz = torch.tensor([-1., -1., 0.])
        h = torch.tensor([1.2, .5, .5])
        self.assertEqual(recovery_gate(gz, h, .6, .9).tolist(), [False, True, True])
        self.assertEqual(recovery_gate(gz, h, .6).tolist(), [False, False, True])

    def test_training_routes_and_checkpoint(self):
        torch.manual_seed(2)
        model = make_model()
        a, b = torch.randn(4, 4), torch.randn(4, 4)
        rec = torch.tensor([True, False, True, False])
        condition = torch.tensor([0., 1., .5, 0.])
        done = torch.tensor([False, False, True, False])
        with torch.inference_mode():
            reward = model.reward_and_store(a, b, torch.ones(4), rec, condition, done)
        self.assertEqual(reward[2].item(), 1.)
        self.assertEqual(model.replay['rec'].num_samples, 1)
        self.assertEqual(model.replay['loco'].num_samples, 2)
        self.assertEqual(model.replay['loco'].states[:2, -1].tolist(), [1., 0.])
        before = {k: v.clone() for k, v in model.state_dict().items()}
        metrics = model.update(2)
        self.assertTrue(all(torch.isfinite(torch.tensor(v)) for v in metrics.values()))
        for kind in ('rec', 'loco'):
            key = f'networks.{kind}.amp_linear.weight'
            self.assertFalse(torch.equal(before[key], model.state_dict()[key]))
        restored = make_model()
        restored.restore(copy.deepcopy(model.checkpoint()))
        for k, v in model.state_dict().items():
            torch.testing.assert_close(v, restored.state_dict()[k])
        self.assertTrue(restored.optimizers['rec'].state)

    def test_signed_expert_matching_and_unsupported_commands(self):
        model = make_model(vector=True)
        commands = torch.tensor([[1., 0., 0.], [-1., 0., 0.], [0., .5, 0.], [0., 0., 1.]])
        idx, supported = model._matched_expert(commands)
        self.assertEqual(supported.tolist(), [True, False, False, False])
        torch.testing.assert_close(model.expert_velocity[idx[0]], commands[0])
        a, b = torch.randn(4, 4), torch.randn(4, 4)
        rewards = model.reward_and_store(a, b, torch.ones(4), torch.zeros(4, dtype=torch.bool), commands, torch.zeros(4, dtype=torch.bool))
        torch.testing.assert_close(rewards[1:], torch.ones(3))
        self.assertEqual(model.replay['loco'].num_samples, 1)
        metrics = model.update(1)
        self.assertTrue(all(torch.isfinite(torch.tensor(v)) for v in metrics.values()))
        self.assertEqual(metrics['loco_expert_coverage'], .25)
        restored = make_model(vector=True)
        restored.restore(copy.deepcopy(model.checkpoint()))

    def test_empty_branch(self):
        model = make_model()
        self.assertEqual(model.update(1), {'rec_samples': 0., 'loco_samples': 0.})

if __name__ == '__main__':
    unittest.main()
