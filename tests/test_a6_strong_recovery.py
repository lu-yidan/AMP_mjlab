"""Recovery shaping must move a low escaped torso without paying pre-escape."""

from types import SimpleNamespace

import torch

from g1recovery_host.a6_reward import _guided_recovery_reward


class _Scene(dict):
    def __init__(self):
        super().__init__()
        self.env_origins = torch.zeros(3, 3)


def test_strong_recovery_pays_height_and_upright_only_after_escape():
    scene = _Scene()
    scene["robot"] = SimpleNamespace(
        data=SimpleNamespace(
            root_link_pos_w=torch.tensor(
                [[0.0, 0.0, 0.11], [0.0, 0.0, 0.21], [0.0, 0.0, 0.21]]
            ),
            projected_gravity_b=torch.tensor(
                [[0.0, 0.0, 0.0], [0.0, 0.0, -0.5], [0.0, 0.0, -0.5]]
            ),
        )
    )
    env = SimpleNamespace(
        scene=scene,
        _a6_scene=torch.ones(3, dtype=torch.long),
        _a6_escaped=torch.tensor([True, True, False]),
        _a6_invalid=torch.zeros(3, dtype=torch.bool),
        _a6_direction=torch.ones(3, dtype=torch.long),
        _a6_strong_recovery_v1=True,
    )
    strong = _guided_recovery_reward(env)
    env._a6_strong_recovery_v1 = False
    original = _guided_recovery_reward(env)
    assert strong[0] > 10 * original[0]
    assert strong[1] > strong[0]
    assert strong[2] == original[2] == 0
