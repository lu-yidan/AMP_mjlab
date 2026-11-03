"""A replay reset must expose the same six proprio frames as its capture."""

from types import SimpleNamespace

import torch
from mjlab.utils.buffers.circular_buffer import CircularBuffer

from g1recovery_host.a6_history_replay import restore_postescape_actor_history


def test_recorded_history_restores_buffer_order_and_preserves_other_worlds():
    host = CircularBuffer(6, 2, "cpu")
    torque = CircularBuffer(6, 2, "cpu")
    host.append(torch.zeros(2, 94))
    torque.append(torch.zeros(2, 29))
    snapshot = torch.arange(738, dtype=torch.float32).unsqueeze(0)
    env = SimpleNamespace(
        num_envs=2,
        device="cpu",
        _a6_postescape_bank={"actor_obs": snapshot},
        _a6_postescape_start=torch.tensor([True, False]),
        _a6_postescape_bank_index=torch.tensor([0, -1]),
        observation_manager=SimpleNamespace(
            _group_obs_term_history_buffer={
                "actor": {"host_frame": host, "motor_torque": torque}
            }
        ),
        obs_buf={"actor": torch.zeros(2, 738)},
    )
    obs = restore_postescape_actor_history(env)
    assert torch.equal(obs["actor"][0], snapshot[0])
    assert torch.count_nonzero(obs["actor"][1]) == 0
    assert torch.equal(host.buffer[0], snapshot[0, :564].reshape(6, 94))
    assert torch.equal(torque.buffer[0], snapshot[0, 564:].reshape(6, 29))
    assert host.current_length[0] == torque.current_length[0] == 6
