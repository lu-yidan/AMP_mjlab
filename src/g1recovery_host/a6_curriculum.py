"""Advance guided-board difficulty only after both starting postures succeed."""

from __future__ import annotations

from collections import deque

import torch


def guided_side_target_advantage_scale(scene, direction, escaped, tier):
    """Emphasize harder fixed-board side samples only in PPO's target group."""
    if not (scene.shape == direction.shape == escaped.shape == tier.shape):
        raise ValueError("guided side advantage inputs must align")
    guided_side_trapped = (scene == 1) & (direction >= 2) & ~escaped
    difficulty = torch.where(tier == 3, 3.0, torch.where(tier == 2, 2.0, 1.0))
    return torch.where(guided_side_trapped, difficulty, 1.0)


class SuccessGatedEdgeCurriculum:
    """Move the 6 kg board toward centre in measured 2.5 cm increments.

    A resolved attempt is a first escape or a pre-escape episode ending.
    Unresolved short rollouts do not count as successful attempts.
    """

    def __init__(self, start_offset_m=0.25, state=None):
        self.offset_m = float(start_offset_m)
        self.history = deque(maxlen=20)
        self.advances = 0
        if state is not None:
            self.offset_m = float(state["offset_m"])
            self.history.extend(state["history"])
            self.advances = int(state["advances"])
        if not 0.0 <= self.offset_m <= 0.50:
            raise ValueError("invalid success-gated guided-board offset")

    def observe(self, prone: tuple[int, int, int], supine: tuple[int, int, int],
                *sides: tuple[int, int, int]):
        """Record easy outcomes, optionally followed by both half-tier sides."""
        postures = (prone, supine, *sides)
        if any(len(row) != 3 or any(value < 0 for value in row) for row in postures):
            raise ValueError("A6 outcome counts must be nonnegative")
        if self.history and any(len(entry) != len(postures) for entry in self.history):
            raise ValueError("A6 curriculum posture count changed without a restart")
        self.history.append(tuple(tuple(row) for row in postures))
        totals = self.totals()
        if len(self.history) < self.history.maxlen or self.offset_m == 0:
            return False, totals
        ready = True
        for index, (escaped, invalid, other) in enumerate(totals):
            resolved = escaped + invalid + other
            # Six entries add left/right half. Eight entries add left/right
            # 0.75-offset bridge, then left/right 0.5-offset bridge.
            half_side = (len(totals) == 6 and index >= 4) or (
                len(totals) == 8 and index >= 6
            )
            medium_side = len(totals) == 8 and 4 <= index < 6
            ready &= (
                resolved >= 20
                and escaped / resolved >= (0.20 if half_side else 0.35)
                and invalid / resolved <= (0.75 if half_side else 0.60 if medium_side else 0.50)
            )
        if not ready:
            return False, totals
        self.offset_m = round(max(0.0, self.offset_m - 0.025), 6)
        self.advances += 1
        self.history.clear()
        return True, totals

    def totals(self):
        num_postures = len(self.history[0]) if self.history else 2
        return tuple(
            tuple(sum(entry[posture][kind] for entry in self.history) for kind in range(3))
            for posture in range(num_postures)
        )

    def state_dict(self):
        return {
            "offset_m": self.offset_m,
            "history": list(self.history),
            "advances": self.advances,
        }
