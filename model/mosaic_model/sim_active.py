"""Rejection-reduced variant of Sim: keep the set of *active* elements (those whose colours
match at least one channel) and draw attempts only among them.

Same Markov chain as Sim (normal mode): every active element is attempted at rate K and fires
channel c with probability k_c / K, so each channel fires at rate k_c. Inactive elements have
zero rate and are never drawn. Time advances by 1 / (K * M) per attempt, M = number of
active elements at that attempt.

Bookkeeping after a firing: every element that contains a cell whose colour changed is
re-classified (a cell sits in 1 cell element, 6 edges and 6 vertices).
Kept separate from Sim for benchmarking (tools/bench_active.py); no shuffled mode.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np

from .graph import Graph
from .reactions import Block, ReactionSet


class ActiveSim:
    def __init__(
        self, graph: Graph, reactions: ReactionSet, colours: Iterable[int], seed: int
    ) -> None:
        self.graph = graph
        self.reactions = reactions
        self.palette = reactions.palette
        self.col: list[int] = [int(c) for c in colours]
        self.rng = np.random.default_rng(seed)
        self.t = 0.0
        self.fires = [0] * len(reactions.reactions)
        self.attempts_done = 0
        used = reactions.shapes_used()
        elems: list[tuple[int, ...]] = []
        shape: list[int] = []
        if "cell" in used:
            elems += [(i,) for i in range(graph.n_cells)]
            shape += [0] * graph.n_cells
        if "edge" in used:
            elems += [tuple(e) for e in graph.edges.tolist()]
            shape += [1] * len(graph.edges)
        if "vertex" in used:
            elems += [tuple(v) for v in graph.vertices.tolist()]
            shape += [2] * len(graph.vertices)
        self._elems = elems
        self._shape = shape
        inc: list[list[int]] = [[] for _ in range(graph.n_cells)]
        for e, cells in enumerate(elems):
            for c in set(cells):
                inc[c].append(e)
        self._inc = inc
        self.rebuild()

    @property
    def colours(self) -> np.ndarray:
        return np.asarray(self.col, dtype=np.int8)

    def counts(self) -> np.ndarray:
        return np.bincount(self.col, minlength=self.palette)

    def fractions(self) -> np.ndarray:
        return self.counts() / self.graph.n_cells

    def rebuild(self) -> None:
        tabs = self.reactions.tables()
        self._tabs = [tabs.get("cell"), tabs.get("edge"), tabs.get("vertex")]
        self.K = self.reactions.max_propensity()
        self._act: list[int] = []
        self._pos = [-1] * len(self._elems)
        for e in range(len(self._elems)):
            if self._entry(e) is not None:
                self._pos[e] = len(self._act)
                self._act.append(e)

    def swap(self, a: Block, b: Block) -> None:
        self.reactions.swap(a, b)
        self.rebuild()

    def _entry(self, e: int):
        col, p = self.col, self.palette
        cells = self._elems[e]
        k = 0
        for c in cells:
            k = k * p + col[c]
        return self._tabs[self._shape[e]][k]

    @property
    def n_active(self) -> int:
        return len(self._act)

    def advance(self, dt: float, batch: int = 1024) -> None:
        """Run attempts until sim time has advanced by dt (or nothing is active)."""
        t_end = self.t + dt
        col, p, K = self.col, self.palette, self.K
        elems, shape, tabs, inc = self._elems, self._shape, self._tabs, self._inc
        act, pos, fires = self._act, self._pos, self.fires
        rng = self.rng
        while self.t < t_end and act:
            r1 = rng.random(batch).tolist()
            r2 = (rng.random(batch) * K).tolist()
            t = self.t
            n_done = 0
            for x, u in zip(r1, r2, strict=True):
                m = len(act)
                if m == 0 or t >= t_end:
                    break
                t += 1.0 / (K * m)
                n_done += 1
                e = act[int(x * m)]
                cells = elems[e]
                k = 0
                for c in cells:
                    k = k * p + col[c]
                ent = tabs[shape[e]][k]
                if u >= ent.total:
                    continue
                j = 0
                while u >= ent.cum[j]:
                    j += 1
                new = ent.new[j]
                fires[ent.rid[j]] += 1
                changed = []
                for c, v in zip(cells, new, strict=True):
                    if col[c] != v:
                        col[c] = v
                        changed.append(c)
                # re-classify every element touching a changed cell
                for c in changed:
                    for e2 in inc[c]:
                        cs = elems[e2]
                        k2 = 0
                        for c2 in cs:
                            k2 = k2 * p + col[c2]
                        on = tabs[shape[e2]][k2] is not None
                        i2 = pos[e2]
                        if on and i2 < 0:
                            pos[e2] = len(act)
                            act.append(e2)
                        elif not on and i2 >= 0:
                            last = act.pop()
                            if last != e2:
                                act[i2] = last
                                pos[last] = i2
                            pos[e2] = -1
            self.t = t
            self.attempts_done += n_done
        if not act:  # absorbing state: nothing can fire, time just passes
            self.t = max(self.t, t_end)
