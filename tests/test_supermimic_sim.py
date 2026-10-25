"""Opt-in real CUDA simulator regression: RUN_SUPERMIMIC_SIM=1."""
import os
import unittest
import torch
from src.supermimic.env import VelocitySim
from src.supermimic.mdp import unflatten_state,reward,is_recovery,flatten_state

@unittest.skipUnless(os.environ.get('RUN_SUPERMIMIC_SIM')=='1','Requires compatible mjlab/CUDA runtime')
class SimulatorTest(unittest.TestCase):
    def test_reset_distribution_terminal_state_and_heldout_reset(self):
        sim=VelocitySim(128)
        try:
            initial=sim.state();assert is_recovery(initial).any()
            groups=sim.reset_groups.bincount(minlength=11).tolist()
            assert sim.bank is not None and len(sim.reset_groups.unique())>5
            tr=sim.step(torch.zeros(128,29,device='cuda:0'))
            assert not tr['terminated'].any(), 'ordinary fallen reset must survive'
            assert (sim.ep_step==1).all()
            expected=reward(unflatten_state(tr['s1']),tr['command'],tr['a'],tr['a_prev'],sim.default_q,sim.env.step_dt,previous_state=unflatten_state(tr['s']),gamma=sim.gamma)
            torch.testing.assert_close(tr['r'],expected)
            sim.ep_step.fill_(sim.max_episode_steps-1)
            tr=sim.step(torch.ones(128,29,device='cuda:0')*.1)
            assert tr['done'].all() and (sim.ep_step==0).all() and (sim.a_prev==0).all()
            assert not torch.allclose(tr['s1'],flatten_state(sim.state()))
            bank=sim.bank
            ids=torch.arange(sim.n,device=sim.device)
            sim.reset_to_bank(bank,ids)
            torch.testing.assert_close(flatten_state(sim.state())[:,:3],bank.qpos[ids,:3],atol=1e-5,rtol=1e-5)
            assert sim.bank.sha256 and (sim.reset_groups==0).all()
        finally:sim.close()

    def test_grouped_reset_budgets(self):
        sim=VelocitySim(128,reset_mode='grouped',max_speed=1)
        try:
            assert sim.recovery_env.sum()==round(.4*128)
            assert (sim.reset_groups[~sim.recovery_env]==0).all()
            assert (sim.reset_groups[sim.recovery_env]>0).all()
            counters=torch.zeros(2,dtype=torch.long,device=sim.device)
            for i in range(260):
                tr=sim.step(torch.zeros(128,29,device=sim.device))
                budget=tr['budget_reset'];rec=tr['recovery_env']
                counters[0]+=(budget&~rec).sum();counters[1]+=(budget&rec).sum()
                if i<249:assert not (budget&rec).any()
                if budget.any():assert (sim.ep_step[budget]==0).all()
            assert (counters>0).all(),counters
            print('PASS fixed groups, recovery-only initial bank, 0.3s/5s budgets, post-reset episode counters:',counters.tolist())
        finally:sim.close()
