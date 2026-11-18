"""Random sequential update of a reaction mosaic.

One attempt: pick an element uniformly among all eligible elements (cells, edges, vertices of
the shapes that carry at least one reaction); read its colour code; with probability
total / K fire one matching channel, chosen with probability proportional to its rate.
K is the largest total channel rate of any element state (ReactionSet.max_propensity).

One sweep = one attempt per eligible element; sim time advances by 1 / K per sweep, so every
channel fires at its rate k per unit time (continuous-time Markov chain in the limit of small
change per sweep).

Shuffled control (well-mixed), two variants:
  shuffle="attempt" (default): every attempt draws the element's cells uniformly at random
      (distinct), i.e. a full random permutation before every attempt. This is the exact
      well-mixed process whose large-N limit is the mean-field ODE.
  shuffle="sweep": the brief's original rule, cell colours permuted after each sweep. It
      only approaches mean field when each sweep changes a small fraction of cells
      (K large compared with the per-cell reaction rates); kept for comparison.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np

from .graph import Graph
from .reactions import Block, ReactionSet


class Sim:
    def __init__(
        self,
        graph: Graph,
        reactions: ReactionSet,
        colours: Iterable[int],
        seed: int,
        shuffled: bool = False,
        shuffle: str = "attempt",
    ) -> None:
        self.graph = graph
        self.reactions = reactions
        self.palette = reactions.palette
        self.col: list[int] = [int(c) for c in colours]
        if len(self.col) != graph.n_cells:
            raise ValueError("colour array does not match the graph")
        self.rng = np.random.default_rng(seed)
        self.shuffled = shuffled
        if shuffle not in ("attempt", "sweep"):
            raise ValueError("shuffle must be 'attempt' or 'sweep'")
        self.shuffle = shuffle
        self.t = 0.0
        self.sweeps = 0
        self._pos = 0  # attempts done in the current sweep
        self._carry = 0.0  # fractional attempts owed by advance()
        self.fires = [0] * len(reactions.reactions)
        self._cells = [(i,) for i in range(graph.n_cells)]
        self._edges = [tuple(e) for e in graph.edges.tolist()]
        self._verts = [tuple(v) for v in graph.vertices.tolist()]
        self.rebuild()

    # ---- set-up -------------------------------------------------------------------------
    def rebuild(self) -> None:
        """Rebuild lookup tables and K (after any change to the reactions)."""
        rs = self.reactions
        tabs = rs.tables()
        self._tab = {s: tabs.get(s) for s in ("cell", "edge", "vertex")}
        used = rs.shapes_used()
        self._nc = self.graph.n_cells if "cell" in used else 0
        self._ne = len(self._edges) if "edge" in used else 0
        self._nv = len(self._verts) if "vertex" in used else 0
        self.n_elig = self._nc + self._ne + self._nv
        self.K = rs.max_propensity()

    def swap(self, a: Block, b: Block) -> None:
        self.reactions.swap(a, b)
        self.rebuild()

    # ---- state --------------------------------------------------------------------------
    @property
    def colours(self) -> np.ndarray:
        return np.asarray(self.col, dtype=np.int8)

    def counts(self) -> np.ndarray:
        return np.bincount(self.col, minlength=self.palette)

    def fractions(self) -> np.ndarray:
        return self.counts() / self.graph.n_cells

    # ---- dynamics -----------------------------------------------------------------------
    def sweep(self, n: int = 1) -> None:
        """Complete n sweeps (the first one finishes a partial sweep, if any)."""
        for _ in range(n):
            self.attempts(self.n_elig - self._pos)

    def attempts(self, n: int) -> None:
        """Perform n attempts; sim time advances by 1 / (K * n_elig) per attempt. Sweep
        boundaries (sweep counter, per-sweep shuffling) are honoured inside the call."""
        while n > 0:
            chunk = min(n, self.n_elig - self._pos)
            if self.K > 0 and chunk > 0:
                if self.shuffled and self.shuffle == "attempt":
                    self._attempts_mixed(chunk)
                else:
                    self._attempts(chunk)
                self.t += chunk / (self.K * self.n_elig)
            self._pos += chunk
            n -= chunk
            if self._pos >= self.n_elig:
                self._pos = 0
                self.sweeps += 1
                if self.shuffled and self.shuffle == "sweep":
                    self.col = self.rng.permutation(self.col).tolist()

    def advance(self, dt: float) -> int:
        """Advance sim time by about dt (whole attempts; the remainder is carried over).
        Returns the number of attempts made."""
        if self.K <= 0:
            return 0
        self._carry += dt * self.K * self.n_elig
        n = int(self._carry)
        self._carry -= n
        self.attempts(n)
        return n

    def _attempts(self, n: int) -> None:
        rng = self.rng
        idx = rng.integers(0, self.n_elig, size=n).tolist()
        us = (rng.random(n) * self.K).tolist()
        col = self.col
        fires = self.fires
        p = self.palette
        nc, ne = self._nc, self._ne
        cells, edges, verts = self._cells, self._edges, self._verts
        tc, te, tv = self._tab["cell"], self._tab["edge"], self._tab["vertex"]
        for e, u in zip(idx, us, strict=True):
            if e < nc:
                a = cells[e][0]
                ent = tc[col[a]]
                if ent is None or u >= ent.total:
                    continue
                j = 0
                while u >= ent.cum[j]:
                    j += 1
                col[a] = ent.new[j][0]
            elif e < nc + ne:
                a, b = edges[e - nc]
                ent = te[col[a] * p + col[b]]
                if ent is None or u >= ent.total:
                    continue
                j = 0
                while u >= ent.cum[j]:
                    j += 1
                col[a], col[b] = ent.new[j]
            else:
                a, b, c = verts[e - nc - ne]
                ent = tv[(col[a] * p + col[b]) * p + col[c]]
                if ent is None or u >= ent.total:
                    continue
                j = 0
                while u >= ent.cum[j]:
                    j += 1
                col[a], col[b], col[c] = ent.new[j]
            fires[ent.rid[j]] += 1

    def _attempts_mixed(self, n: int) -> None:
        """Well-mixed attempts: the element's shape is drawn with the same weights as in
        the normal run; its cells are drawn uniformly (an attempt with a repeated cell is
        skipped, a bias of order 1 / n_cells)."""
        rng = self.rng
        idx = rng.integers(0, self.n_elig, size=n).tolist()
        us = (rng.random(n) * self.K).tolist()
        nc_cells = self.graph.n_cells
        ra = rng.integers(0, nc_cells, size=n).tolist()
        rb = rng.integers(0, nc_cells, size=n).tolist()
        rc = rng.integers(0, nc_cells, size=n).tolist()
        col = self.col
        fires = self.fires
        p = self.palette
        nc, ne = self._nc, self._ne
        tc, te, tv = self._tab["cell"], self._tab["edge"], self._tab["vertex"]
        for e, u, a, b, c in zip(idx, us, ra, rb, rc, strict=True):
            if e < nc:
                ent = tc[col[a]]
                if ent is None or u >= ent.total:
                    continue
                j = 0
                while u >= ent.cum[j]:
                    j += 1
                col[a] = ent.new[j][0]
            elif e < nc + ne:
                if a == b:
                    continue
                ent = te[col[a] * p + col[b]]
                if ent is None or u >= ent.total:
                    continue
                j = 0
                while u >= ent.cum[j]:
                    j += 1
                col[a], col[b] = ent.new[j]
            else:
                if a == b or b == c or a == c:
                    continue
                ent = tv[(col[a] * p + col[b]) * p + col[c]]
                if ent is None or u >= ent.total:
                    continue
                j = 0
                while u >= ent.cum[j]:
                    j += 1
                col[a], col[b], col[c] = ent.new[j]
            fires[ent.rid[j]] += 1


Switch = tuple[int, Block, Block]  # (sweep index at which to swap, block a, block b)


def run_timeline(
    sim: Sim, sweeps: int, timeline: Iterable[Switch] = (), record_every: int = 1
) -> tuple[np.ndarray, np.ndarray]:
    """Run `sweeps` sweeps, applying each switch before the sweep with that index
    (counted from the start of this call). Returns (times, fractions) sampled every
    `record_every` sweeps, including the initial state."""
    pending = sorted(timeline, key=lambda s: s[0])
    ts = [sim.t]
    fr = [sim.fractions()]
    k = 0
    for s in range(sweeps):
        while k < len(pending) and pending[k][0] == s:
            sim.swap(pending[k][1], pending[k][2])
            k += 1
        sim.sweep()
        if (s + 1) % record_every == 0:
            ts.append(sim.t)
            fr.append(sim.fractions())
    return np.array(ts), np.array(fr)
