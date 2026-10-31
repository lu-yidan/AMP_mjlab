"""Terrain-relative heights for the isolated critic-scan pilot."""
from dataclasses import dataclass
import torch
from .tracking_commands import TrackingCommand, TrackingCommandCfg
from .rewards import _apply_delay_env_reward_mask_only


def ground_height(env):
    sensor = env.scene['terrain_scan']
    # 17 x 11 yaw-aligned grid: the central ray lies under the pelvis.
    center = sensor.data.distances.shape[1] // 2
    valid = sensor.data.distances[:, center] >= 0
    # Misses (e.g. below a surface) use the tile origin; the pilot audit counts them.
    return torch.where(valid, sensor.data.hit_pos_w[:, center, 2], env.scene.env_origins[:, 2])


class TerrainTrackingCommand(TrackingCommand):
    def physical_state(self):
        height, gz, head = super().physical_state()
        delta = ground_height(self._env) - self._env.scene.env_origins[:, 2]
        return height - delta, gz, head - delta


@dataclass(kw_only=True)
class TerrainTrackingCommandCfg(TrackingCommandCfg):
    def build(self, env):
        return TerrainTrackingCommand(self, env)


def terrain_root_low(env, minimum_height):
    return env.scene['robot'].data.root_link_pos_w[:, 2] - ground_height(env) < minimum_height


def terrain_height_reward(env, std, mask_delay=False, delay_env_rew_ratio=1.0):
    data = env.scene['robot'].data
    height = data.root_link_pos_w[:, 2] - ground_height(env)
    reward = torch.exp(-(height-data.default_root_state[:, 2]).square()/std**2)
    return _apply_delay_env_reward_mask_only(env, reward, mask_delay, delay_env_rew_ratio)
