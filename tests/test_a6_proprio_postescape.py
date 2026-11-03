"""A reached exit-state reset remains proprioceptive and motor-active."""

import torch
from mjlab.envs import ManagerBasedRlEnv

from g1recovery_host.a6_task import host_a6_env_cfg
from g1recovery_host.phase_switch import postescape_live_observation_enabled


def test_historical_replay_checkpoint_restores_effective_live_observation():
    assert postescape_live_observation_enabled({
        "a6_postescape_live_obs_v1": False,
        "a6_proprio_postescape_bank_sha256": "recorded-replay-bank",
    })
    assert postescape_live_observation_enabled({
        "reward_variant": "historical_a6_proprio_postescape_replay_v1"
    })
    assert not postescape_live_observation_enabled({
        "a6_postescape_live_obs_v1": False,
        "reward_variant": "historical_a6",
    })


def test_proprio_postescape_reset_has_live_actor_observation():
    cfg = host_a6_env_cfg(
        num_envs=32,
        edge_curriculum=True,
        critic_plate_observation=True,
        proprio_motor_torque=True,
        focused_guided=True,
        reset_mix="diverse_v3",
        postescape_prone=True,
        aligned_edge=True,
        footprint_edge=True,
        footprint_overlap_m=0.03,
        hard_mix=True,
        privileged_guided_actions=False,
    )
    env = ManagerBasedRlEnv(cfg, device="cpu")
    try:
        env._a6_postescape_live_obs_v1 = True
        env._a6_edge_offset_m = 0.25
        obs, _ = env.reset()
        replay = env._a6_postescape_start.clone()
        assert int(replay.sum()) == 2
        assert env._a6_escaped[replay].all()
        assert obs["actor"].shape == (32, 738)
        assert obs["critic"].shape == (32, 786)
        assert obs["actor"][replay, :564].abs().sum() > 0
        term = env.action_manager.get_term("joint_pos")
        assert not term.cfg.privileged_guided_actions
        next_obs, _, _, _, _ = env.step(torch.full((32, 29), 0.1))
        assert term._active[replay].all()
        assert next_obs["actor"][replay, 564:].abs().sum() > 0
    finally:
        env.close()
