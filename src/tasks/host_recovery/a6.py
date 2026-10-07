"""Registered wrappers for the historical A6 G-/G+ HoST adaptation."""

from __future__ import annotations

from src.tasks.host_recovery.a6_runtime import a6_env_cfg


def a6_gminus_env_cfg(play: bool = False):
  """A6 control arm without dense geometric escape guidance."""
  return a6_env_cfg(play=play, group="G-")


def a6_gplus_env_cfg(play: bool = False):
  """A6 treatment arm with G, clearance, separation, and alpha gating."""
  return a6_env_cfg(play=play, group="G+")
