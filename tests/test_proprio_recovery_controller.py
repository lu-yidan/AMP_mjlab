"""The deployable recovery switch reads only actor history and a local clock."""

import hashlib

import pytest
import torch

from g1recovery_host.proprio_phase import (
    ProprioPhaseDetector,
    supine_flat_recovery_label,
)
from g1recovery_host.proprio_recovery import (
    InitialProneRouter,
    InitialSideRouter,
    ProprioRecoveryController,
    ProprioSupineFlatGate,
    validate_controller_provenance,
)


def test_initial_prone_router_uses_reset_imu_and_clears_per_episode():
    router = InitialProneRouter(3, "cpu")
    gravity = torch.tensor([[0.9, 0.0, 0.0], [-0.9, 0.0, 0.0], [0.9, 0.8, 0.0]])
    router.arm_initial_gravity(gravity)
    assert router.initial_prone.tolist() == [True, False, False]
    router.reset(torch.tensor([0]))
    assert not router.initial_prone.any()


def test_initial_side_router_uses_reset_imu_and_clears_per_episode():
    router = InitialSideRouter(4, "cpu")
    gravity = torch.tensor([
        [0.0, 0.9, 0.0], [0.0, -0.9, 0.0], [0.9, 0.0, 0.0], [0.8, 0.8, 0.0],
    ])
    router.arm_initial_gravity(gravity)
    assert router.initial_side.tolist() == [True, True, False, False]
    assert router.initial_left_side.tolist() == [True, False, False, False]
    router.reset(torch.tensor([0, 1]))
    assert not router.initial_side.any()
    assert not router.initial_left_side.any()


class HighScore:
    def __call__(self, obs):
        return torch.full((len(obs),), 10.0)


def test_early_pulse_requires_body_gate_and_exact_control_step():
    model = ProprioPhaseDetector(torch.zeros(738), torch.ones(738))
    controller = ProprioRecoveryController({
        "observation_contract": "proprio_torque_actor_privileged_critic_v2",
        "action_contract": "scene_independent_v1",
        "detector_mean": model.mean,
        "detector_scale": model.scale,
        "detector_model": model.state_dict(),
        "pulse_action": torch.ones(29),
        "threshold": 0.95,
        "consecutive": 2,
        "pulse_steps": 30,
        "prone_gravity_x": 0.70,
        "min_steps": 60,
    }, 3, "cpu")
    eligible = torch.tensor([True, False, True])
    assert not controller.schedule_pulse(
        eligible, torch.tensor([48, 49, 48]), 50,
    ).any()
    selected = controller.schedule_pulse(
        eligible, torch.tensor([49, 49, 50]), 50,
    )
    assert selected.tolist() == [True, False, False]
    assert controller.pulse_remaining.tolist() == [30, 0, 0]
    assert controller.schedule_pulse(
        eligible, torch.tensor([49, 49, 49]), 50,
    ).tolist() == [False, False, True]
    controller.reset(torch.tensor([0]))
    assert controller.schedule_pulse(
        eligible, torch.tensor([49, 49, 49]), 50,
    ).tolist() == [True, False, False]


def test_supine_flat_recovery_labels_exclude_trapped_and_prone_worlds():
    scene = torch.tensor([0, 0, 1, 1, 1, 2, 2])
    direction = torch.tensor([0, 1, 0, 0, 1, 0, 0])
    escaped = torch.tensor([False, False, False, True, True, False, True])
    assert supine_flat_recovery_label(scene, direction, escaped).tolist() == [
        True, False, False, True, False, False, True,
    ]


def test_prone_only_trigger_and_exact_pulse_length():
    model = ProprioPhaseDetector(torch.zeros(738), torch.ones(738))
    controller = ProprioRecoveryController({
        "observation_contract": "proprio_torque_actor_privileged_critic_v2",
        "action_contract": "scene_independent_v1",
        "detector_mean": model.mean,
        "detector_scale": model.scale,
        "detector_model": model.state_dict(),
        "pulse_action": torch.arange(29, dtype=torch.float),
        "threshold": 0.95,
        "consecutive": 2,
        "pulse_steps": 30,
        "prone_gravity_x": 0.70,
        "min_steps": 60,
    }, 2, "cpu")
    controller.detector = HighScore()
    obs = torch.zeros(2, 738)
    obs[0, 5 * 94 + 3] = 0.9
    obs[1, 5 * 94 + 3] = -0.9
    assert not controller.observe(obs, torch.tensor([31, 31])).any()
    assert controller.observe(obs, torch.tensor([60, 60])).tolist() == [True, False]
    assert not controller.observe(obs, torch.tensor([61, 61])).any()
    for _ in range(30):
        action, forced = controller.apply(torch.zeros(2, 29))
        assert forced.tolist() == [True, False]
        assert torch.equal(action[0], torch.arange(29, dtype=torch.float))
        assert not action[1].any()
    _, forced = controller.apply(torch.zeros(2, 29))
    assert not forced.any()
    controller.reset(torch.tensor([0]))
    assert not controller.triggered[0]
    assert not controller.initial_prone.any()


def test_checkpoint_requires_exact_controller(tmp_path):
    path = tmp_path / "controller.pt"
    path.write_bytes(b"original")
    checkpoint = {
        "proprio_recovery_controller_sha256": hashlib.sha256(b"original").hexdigest()
    }
    validate_controller_provenance(checkpoint, path)
    with pytest.raises(ValueError, match="requires"):
        validate_controller_provenance(checkpoint, None)
    path.write_bytes(b"different")
    with pytest.raises(ValueError, match="SHA256"):
        validate_controller_provenance(checkpoint, path)


def test_supine_gate_requires_initial_posture_and_consecutive_safe_frames():
    detector = ProprioPhaseDetector(torch.zeros(738), torch.ones(738))
    gate = ProprioSupineFlatGate(detector, 2, "cpu", threshold=0.9,
                                 consecutive=3, min_steps=35)
    gate.detector = HighScore()
    obs = torch.zeros(2, 738)
    gravity = torch.tensor([[-0.9, 0.0, 0.0], [0.0, 1.0, 0.0]])
    gate.arm_initial_gravity(gravity)
    assert gate.neutral_wait_mask(torch.tensor([31, 31]), 60).tolist() == [True, False]
    base = torch.ones(2, 29)
    probed, mask = gate.probe_action(base, torch.tensor([31, 31]))
    assert mask.tolist() == [True, False]
    assert torch.count_nonzero(probed[0]) == 2
    assert torch.equal(probed[1], base[1])
    assert torch.isclose(probed[0].sum(), torch.tensor(0.0))
    assert not gate.observe(obs, torch.tensor([31, 31])).any()
    assert not gate.observe(obs, torch.tensor([34, 34])).any()
    assert gate.observe(obs, torch.tensor([35, 35])).tolist() == [True, False]
    assert gate.triggered.tolist() == [True, False]
    assert not gate.neutral_wait_mask(torch.tensor([35, 35]), 60).any()
    gate.reset(torch.tensor([0]))
    assert not gate.triggered.any()
    assert not gate.armed[0]
