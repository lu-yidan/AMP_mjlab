"""Prone-start variant of the no-load HoST + SMP get-up reward task."""

from .supine_smp import supine_smp_env_cfg


def prone_smp_env_cfg(play=False):
  cfg = supine_smp_env_cfg(play=play)
  cfg.events["reset_base"].params["posture"] = "prone"
  return cfg
