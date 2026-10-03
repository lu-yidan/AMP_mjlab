"""Runner adapter for the recovery task's local motion command."""

from __future__ import annotations

from typing import Any

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner
from rsl_rl.env import VecEnv

A6_ADAPTATION_STATE_KEY = "a6_adaptation_state"


def add_a6_adaptation_state(env: ManagerBasedRlEnv, infos: dict | None) -> dict | None:
    """Persist the local A6 curriculum origin when this is an A6 environment."""

    if not hasattr(env, "_a6_curriculum_start_counter"):
        return infos
    return {
        **(infos or {}),
        A6_ADAPTATION_STATE_KEY: {
            "curriculum_start_counter": int(env._a6_curriculum_start_counter)
        },
    }


def restore_a6_adaptation_state(env: ManagerBasedRlEnv, infos: dict | None) -> None:
    """Restore an A6 clock, or rebase it when loading a flat parent checkpoint."""

    if not hasattr(env, "_a6_curriculum_start_counter"):
        return
    state = (infos or {}).get(A6_ADAPTATION_STATE_KEY)
    if state is None:
        env._a6_curriculum_start_counter = int(env.common_step_counter)
        return
    start = int(state["curriculum_start_counter"])
    if start < 0 or start > int(env.common_step_counter):
        raise ValueError(
            "invalid A6 curriculum start counter: "
            f"{start} for common step {env.common_step_counter}"
        )
    env._a6_curriculum_start_counter = start


class RecoveryOnPolicyRunner(MjlabOnPolicyRunner):
    """Accept mjlab's tracking-only argument while using the ordinary runner.

    The recovery motion command reuses mjlab's tracking command configuration,
    so the generic training entry point passes ``registry_name``.  Recovery uses
    local converted motion assets and does not need that value after startup.
    """

    def __init__(
        self,
        env: VecEnv,
        train_cfg: dict[str, Any],
        log_dir: str | None = None,
        device: str = "cpu",
        registry_name: str | None = None,
    ) -> None:
        del registry_name
        super().__init__(env, train_cfg, log_dir, device)

    def save(self, path: str, infos: dict | None = None) -> None:
        infos = add_a6_adaptation_state(self.env.unwrapped, infos)
        super().save(path, infos)

    def load(
        self,
        path: str,
        load_cfg: dict | None = None,
        strict: bool = True,
        map_location: str | None = None,
    ) -> dict:
        infos = super().load(path, load_cfg, strict, map_location)
        restore_a6_adaptation_state(self.env.unwrapped, infos)
        return infos


__all__ = [
    "A6_ADAPTATION_STATE_KEY",
    "RecoveryOnPolicyRunner",
    "add_a6_adaptation_state",
    "restore_a6_adaptation_state",
]
