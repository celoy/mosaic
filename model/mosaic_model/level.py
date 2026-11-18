"""JSON level loader (format shared with Godot, see CLAUDE.md, Levels)."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .goals import GoalTracker
from .graph import Graph, make_graph
from .reactions import ReactionSet, reactions_from_json
from .sim import Sim


@dataclass
class Level:
    id: str
    grid: dict
    palette: int
    reactions: list[dict]
    locked: list[list]
    start: dict
    goals: list[dict]
    fail: list[dict] = field(default_factory=list)
    meta: dict = field(default_factory=dict)
    title: str = ""
    empty: int | None = None  # colour index drawn as empty space (optional)
    path: Path | None = None

    def reaction_set(self) -> ReactionSet:
        return ReactionSet(
            reactions_from_json(self.reactions),
            self.palette,
            {(int(b[0]), str(b[1]), int(b[2])) for b in self.locked},
        )

    def graph(self) -> Graph:
        return make_graph(self.grid)

    def tracker(self) -> GoalTracker:
        return GoalTracker(self.goals, self.fail)


def load_level(path: str | Path) -> Level:
    path = Path(path)
    d = json.loads(path.read_text())
    return Level(
        id=d["id"],
        grid=d["grid"],
        palette=int(d["palette"]),
        reactions=d["reactions"],
        locked=d.get("locked", []),
        start=d["start"],
        goals=d.get("goals", []),
        fail=d.get("fail", []),
        meta=d.get("meta", {}),
        title=d.get("title", ""),
        empty=d.get("empty"),
        path=path,
    )


def initial_colours(n_cells: int, fractions: list[float], seed: int) -> np.ndarray:
    """Independent random colours with the given probabilities."""
    rng = np.random.default_rng(seed)
    p = np.asarray(fractions, dtype=float)
    return rng.choice(len(p), size=n_cells, p=p / p.sum())


def make_sim(
    level: Level,
    seed: int | None = None,
    shuffled: bool = False,
    burn_in: bool = True,
    graph: Graph | None = None,
) -> Sim:
    """Sim at the level's start: random colours, then burn_in_sweeps of the initial
    arrangement. Time and firing counters are reset after the burn-in."""
    seed = int(level.start.get("seed", 0)) if seed is None else seed
    g = graph if graph is not None else level.graph()
    cols = initial_colours(g.n_cells, level.start["fractions"], seed)
    sim = Sim(g, level.reaction_set(), cols, seed=seed + 1, shuffled=shuffled)
    if burn_in:
        sim.sweep(int(level.start.get("burn_in_sweeps", 0)))
    sim.t = 0.0
    sim.sweeps = 0
    sim.fires = [0] * len(sim.fires)
    return sim
