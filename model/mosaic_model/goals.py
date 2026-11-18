"""Goal and failure detection.

Goal types (durations in sim time units):
  {"type": "extinct", "color": c}                         colour c has count 0
  {"type": "hold", "bands": [[lo, hi], ...], "duration": T} every fraction within its band
                                                           continuously for T
  {"type": "reach", "target": [...], "tol": e, "duration": T}  max |x - target| <= e
                                                           continuously for T
Each goal latches once met; the level is won when all goals have latched.
Failure conditions use the same types and fail the level as soon as one is met
(e.g. {"type": "extinct", "color": 1} during a hold).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class _Cond:
    spec: dict
    since: float | None = None  # time the condition became continuously true
    done: bool = False

    def instant(self, x: np.ndarray, counts: np.ndarray) -> bool:
        s = self.spec
        if s["type"] == "extinct":
            return counts[s["color"]] == 0
        if s["type"] == "hold":
            return all(lo <= xi <= hi for xi, (lo, hi) in zip(x, s["bands"], strict=True))
        if s["type"] == "reach":
            return float(np.max(np.abs(x - np.asarray(s["target"])))) <= s["tol"]
        raise ValueError(f"unknown goal type {s['type']!r}")

    def update(self, t: float, x: np.ndarray, counts: np.ndarray) -> bool:
        if self.done:
            return True
        if not self.instant(x, counts):
            self.since = None
            return False
        if self.since is None:
            self.since = t
        if t - self.since >= float(self.spec.get("duration", 0.0)):
            self.done = True
        return self.done

    def progress(self, t: float) -> float:
        """Fraction of the required duration achieved (0..1)."""
        if self.done:
            return 1.0
        if self.since is None:
            return 0.0
        d = float(self.spec.get("duration", 0.0))
        return 1.0 if d == 0 else min(1.0, (t - self.since) / d)


@dataclass
class GoalTracker:
    goals: list[dict]
    fail: list[dict] = field(default_factory=list)
    status: str = "running"  # running | won | failed

    def __post_init__(self) -> None:
        self._goals = [_Cond(g) for g in self.goals]
        self._fails = [_Cond(f) for f in self.fail]

    def update(self, t: float, counts: np.ndarray) -> str:
        if self.status != "running":
            return self.status
        counts = np.asarray(counts)
        x = counts / counts.sum()
        if any(f.update(t, x, counts) for f in self._fails):
            self.status = "failed"
            return self.status
        met = [g.update(t, x, counts) for g in self._goals]
        if met and all(met):
            self.status = "won"
        return self.status

    def progress(self, t: float) -> list[float]:
        return [g.progress(t) for g in self._goals]
