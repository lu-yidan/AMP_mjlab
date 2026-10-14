"""Advance guided-board difficulty only after both starting postures succeed."""

from __future__ import annotations

from collections import deque


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

    def observe(self, prone: tuple[int, int, int], supine: tuple[int, int, int]):
        """Record (escaped, invalid, other-ending) counts for one PPO update."""
        if any(value < 0 for row in (prone, supine) for value in row):
            raise ValueError("A6 outcome counts must be nonnegative")
        self.history.append((tuple(prone), tuple(supine)))
        totals = self.totals()
        if len(self.history) < self.history.maxlen or self.offset_m == 0:
            return False, totals
        ready = True
        for escaped, invalid, other in totals:
            resolved = escaped + invalid + other
            ready &= (
                resolved >= 20
                and escaped / resolved >= 0.35
                and invalid / resolved <= 0.50
            )
        if not ready:
            return False, totals
        self.offset_m = round(max(0.0, self.offset_m - 0.025), 6)
        self.advances += 1
        self.history.clear()
        return True, totals

    def totals(self):
        return tuple(
            tuple(sum(entry[posture][kind] for entry in self.history) for kind in range(3))
            for posture in range(2)
        )

    def state_dict(self):
        return {
            "offset_m": self.offset_m,
            "history": list(self.history),
            "advances": self.advances,
        }
