"""Early flat handoff uses only body history, initial IMU, and a clock."""

import torch

from g1recovery_host.proprio_scene import (
    ProprioEarlyFlatGate, ProprioGuidedProneGate, ProprioGuidedSideGate,
    ProprioSceneGate,
)


class FixedSceneScores(torch.nn.Module):
    def forward(self, obs):
        return torch.stack((obs[:, 0], torch.zeros_like(obs[:, 0]),
                            torch.zeros_like(obs[:, 0])), -1)


def test_early_flat_gate_requires_initial_supine_and_sustained_confidence():
    gate = ProprioEarlyFlatGate(
        FixedSceneScores(), 3, "cpu", threshold=0.99,
        consecutive=2, min_steps=40, max_steps=120,
    )
    gate.arm_initial_gravity(torch.tensor([
        [-0.9, 0.0, 0.0], [0.9, 0.0, 0.0], [0.0, 0.9, 0.0],
    ]))
    obs = torch.zeros(3, 738)
    obs[:, 0] = 8.0
    assert not gate.observe(obs, torch.tensor([39, 39, 39])).any()
    assert not gate.observe(obs, torch.tensor([40, 40, 40])).any()
    fired = gate.observe(obs, torch.tensor([41, 41, 41]))
    assert fired.tolist() == [True, False, False]
    assert not gate.observe(obs, torch.tensor([42, 42, 42])).any()
    gate.reset(torch.tensor([0]))
    assert not gate.triggered[0]


def test_early_flat_gate_cannot_trigger_after_window():
    gate = ProprioEarlyFlatGate(
        FixedSceneScores(), 1, "cpu", threshold=0.99,
        consecutive=2, min_steps=40, max_steps=120,
    )
    gate.arm_initial_gravity(torch.tensor([[-0.9, 0.0, 0.0]]))
    obs = torch.zeros(1, 738)
    obs[:, 0] = 8.0
    assert not gate.observe(obs, torch.tensor([119])).any()
    assert not gate.observe(obs, torch.tensor([121])).any()
    assert not gate.triggered.any()


class FixedGuidedScores(torch.nn.Module):
    def forward(self, obs):
        return torch.stack((torch.zeros_like(obs[:, 0]), obs[:, 0],
                            torch.zeros_like(obs[:, 0])), -1)


class FixedFreeScores(torch.nn.Module):
    def forward(self, obs):
        return torch.stack((torch.zeros_like(obs[:, 0]),
                            torch.zeros_like(obs[:, 0]), obs[:, 0]), -1)


def test_free_prone_gate_uses_initial_posture_and_three_confident_steps():
    gate = ProprioSceneGate(
        FixedFreeScores(), 2, "cpu", target_class=2, posture="prone",
        threshold=0.99, consecutive=3, min_steps=31, max_steps=100,
    )
    gate.arm_initial_gravity(torch.tensor([[0.9, 0.0, 0.0], [-0.9, 0.0, 0.0]]))
    obs = torch.zeros(2, 738)
    obs[:, 0] = 8.0
    assert not gate.observe(obs, torch.tensor([97, 97])).any()
    assert not gate.observe(obs, torch.tensor([98, 98])).any()
    assert gate.observe(obs, torch.tensor([99, 99])).tolist() == [True, False]
    gate.reset(torch.tensor([0]))
    assert not gate.triggered.any()


def test_guided_prone_gate_ignores_supine_and_latches_prone():
    gate = ProprioGuidedProneGate(
        FixedGuidedScores(), 2, "cpu", threshold=0.99,
        consecutive=2, min_steps=40, max_steps=120,
    )
    gate.arm_initial_gravity(torch.tensor([[0.9, 0.0, 0.0], [-0.9, 0.0, 0.0]]))
    obs = torch.zeros(2, 738)
    obs[:, 0] = 8.0
    assert not gate.observe(obs, torch.tensor([40, 40])).any()
    assert gate.observe(obs, torch.tensor([41, 41])).tolist() == [True, False]


def test_guided_side_gate_only_arms_side_postures():
    gate = ProprioGuidedSideGate(
        FixedGuidedScores(), 3, "cpu", threshold=0.99,
        consecutive=2, min_steps=35, max_steps=120,
    )
    gate.arm_initial_gravity(torch.tensor([
        [0.1, 0.9, 0.0], [0.0, -0.9, 0.0], [0.9, 0.0, 0.0],
    ]))
    obs = torch.zeros(3, 738)
    obs[:, 0] = 8.0
    assert not gate.observe(obs, torch.tensor([35, 35, 35])).any()
    assert gate.observe(obs, torch.tensor([36, 36, 36])).tolist() == [True, True, False]


def test_guided_side_ema_rejects_one_step_probability_spike():
    gate = ProprioGuidedSideGate(
        FixedGuidedScores(), 1, "cpu", threshold=0.85,
        consecutive=2, min_steps=35, max_steps=120, ema_alpha=0.2,
    )
    gate.arm_initial_gravity(torch.tensor([[0.0, 0.9, 0.0]]))
    high = torch.zeros(1, 738)
    high[:, 0] = 8.0
    low = high.clone()
    low[:, 0] = -8.0
    assert not gate.observe(high, torch.tensor([35])).any()
    assert not gate.observe(low, torch.tensor([36])).any()
    assert not gate.observe(high, torch.tensor([37])).any()
    assert not gate.observe(high, torch.tensor([38])).any()
    assert gate.observe(high, torch.tensor([39])).item()


def test_guided_side_ema_uses_pre_gate_motor_history():
    gate = ProprioGuidedSideGate(
        FixedGuidedScores(), 1, "cpu", threshold=0.5,
        consecutive=1, min_steps=45, max_steps=120, ema_alpha=0.1,
    )
    gate.arm_initial_gravity(torch.tensor([[0.0, 0.9, 0.0]]))
    high = torch.zeros(1, 738)
    high[:, 0] = 8.0
    low = high.clone()
    low[:, 0] = -8.0
    assert not gate.observe(high, torch.tensor([31])).any()
    for step in range(32, 46):
        assert not gate.observe(low, torch.tensor([step])).any()
    assert gate.smoothed_probability.item() < 0.5


def test_guided_side_early_classifier_only_contributes_in_its_window():
    gate = ProprioGuidedSideGate(
        FixedSceneScores(), 1, "cpu", threshold=0.9,
        consecutive=2, min_steps=31, max_steps=120,
        early_classifier=FixedGuidedScores(), early_max_step=35,
    )
    gate.arm_initial_gravity(torch.tensor([[0.0, 0.9, 0.0]]))
    obs = torch.zeros(1, 738)
    obs[:, 0] = 8.0
    assert not gate.observe(obs, torch.tensor([34])).any()
    assert gate.observe(obs, torch.tensor([35])).item()
    gate.reset()
    gate.arm_initial_gravity(torch.tensor([[0.0, 0.9, 0.0]]))
    assert not gate.observe(obs, torch.tensor([36])).any()
    assert not gate.observe(obs, torch.tensor([37])).any()


def test_side_canonical_classifier_reflects_only_initial_right_observation():
    classifier = FixedGuidedScores()
    classifier.canonicalize_side = True
    gate = ProprioGuidedSideGate(
        classifier, 2, "cpu", threshold=0.9,
        consecutive=1, min_steps=35, max_steps=45,
    )
    gate.arm_initial_gravity(torch.tensor([
        [0.0, 0.9, 0.0], [0.0, -0.9, 0.0],
    ]))
    obs = torch.zeros(2, 738)
    obs[:, 0] = torch.tensor([8.0, -8.0])
    assert gate.observe(obs, torch.tensor([35, 35])).tolist() == [True, True]


def test_side_gate_keeps_separate_left_and_right_decision_windows():
    gate = ProprioGuidedSideGate(
        FixedGuidedScores(), 2, "cpu", threshold=0.9,
        consecutive=1, min_steps=31, max_steps=35,
        right_classifier=FixedGuidedScores(),
        right_min_steps=40, right_max_steps=45,
    )
    gate.arm_initial_gravity(torch.tensor([
        [0.0, 0.9, 0.0], [0.0, -0.9, 0.0],
    ]))
    obs = torch.zeros(2, 738)
    obs[:, 0] = 8.0
    assert gate.observe(obs, torch.tensor([35, 35])).tolist() == [True, False]
    assert gate.observe(obs, torch.tensor([40, 40])).tolist() == [False, True]
