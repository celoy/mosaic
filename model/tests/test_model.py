"""Phase 1 tests 1-7 from CLAUDE.md, plus grid geometry checks."""

from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np
import pytest

from mosaic_model import (
    GoalTracker,
    Reaction,
    ReactionSet,
    Sim,
    hex_torus,
    initial_colours,
    integrate,
    load_level,
    make_sim,
    rhs,
    run_timeline,
)
from mosaic_model.reactions import ARITY, code

LEVELS = Path(__file__).resolve().parents[2] / "levels"


# ---- grid -------------------------------------------------------------------------------
@pytest.mark.parametrize("n", [4, 8, 64])
def test_hex_grid_adjacency(n: int) -> None:
    g = hex_torus(n)
    assert g.edges.shape == (3 * n * n, 2)
    assert g.vertices.shape == (2 * n * n, 3)
    nb = g.neighbours()
    for i in range(g.n_cells):
        assert len(set(nb[i])) == 6 and i not in nb[i]
    # every unordered edge appears once
    assert len({frozenset(e) for e in g.edges.tolist()}) == 3 * n * n
    # every vertex is a triangle of mutual neighbours, listed clockwise on screen (y down)
    seen = set()
    for a, b, c in g.vertices.tolist():
        assert b in nb[a] and c in nb[a] and c in nb[b]
        u, v = g.displacement(a, b), g.displacement(a, c)
        assert u[0] * v[1] - u[1] * v[0] > 0  # positive cross product in y-down = clockwise
        seen.add(frozenset((a, b, c)))
    assert len(seen) == 2 * n * n
    # each cell touches 6 vertices
    deg = np.bincount(g.vertices.ravel(), minlength=g.n_cells)
    assert (deg == 6).all()


# ---- test 1: lookup tables vs brute force -----------------------------------------------
def _brute(rs: ReactionSet, shape: str, cols: tuple[int, ...]) -> list:
    """Channels by direct enumeration of every relabelling of the element's cells."""
    out = []
    m = ARITY[shape]
    for i, r in enumerate(rs.reactions):
        if r.shape != shape:
            continue
        for perm in itertools.permutations(range(m)):
            # perm[q] = element position of reaction position q
            if shape == "vertex":
                # clockwise order kept iff perm is a cyclic rotation (even permutation)
                even = sum(perm[x] > perm[y] for x in range(3) for y in range(x + 1, 3)) % 2 == 0
                if not even and r.chiral:
                    continue
            if all(cols[perm[q]] == r.inp[q] for q in range(m)):
                nc = list(cols)
                for q in range(m):
                    nc[perm[q]] = r.out[q]
                out.append((i, tuple(nc), r.rate))
    return sorted(out)


def _random_set(rng: np.random.Generator, palette: int) -> ReactionSet:
    rx = []
    for shape in ("cell", "edge", "vertex", "edge", "vertex"):
        m = ARITY[shape]
        rx.append(
            Reaction(
                shape,
                rng.integers(0, palette, m).tolist(),
                rng.integers(0, palette, m).tolist(),
                float(rng.choice([0.5, 1.0, 2.0])),
                chiral=bool(rng.random() < 0.5),
            )
        )
    return ReactionSet(rx, palette)


@pytest.mark.parametrize("seed", range(20))
def test_lookup_tables_match_brute_force(seed: int) -> None:
    rng = np.random.default_rng(seed)
    palette = int(rng.integers(2, 5))
    rs = _random_set(rng, palette)
    tabs = rs.tables()
    k = 0.0
    for shape, tab in tabs.items():
        for cols in itertools.product(range(palette), repeat=ARITY[shape]):
            ent = tab[code(cols, palette)]
            got = []
            if ent is not None:
                got = sorted(zip(ent.rid, ent.new, np.diff([0.0, *ent.cum]), strict=True))
            exp = _brute(rs, shape, cols)
            assert [(a, b) for a, b, _ in got] == [(a, b) for a, b, _ in exp]
            assert np.allclose([c for *_, c in got], [c for *_, c in exp])
            if ent is not None:
                k = max(k, ent.total)
    assert rs.max_propensity() == pytest.approx(k)


# ---- test 2: chirality -------------------------------------------------------------------
def test_clockwise_vertex_reaction_ignores_mirror() -> None:
    chiral = ReactionSet([Reaction("vertex", [0, 1, 2], [2, 0, 1], 1.0, chiral=True)], 3)
    achiral = ReactionSet([Reaction("vertex", [0, 1, 2], [2, 0, 1], 1.0, chiral=False)], 3)
    tc, ta = chiral.tables()["vertex"], achiral.tables()["vertex"]
    for cw in [(0, 1, 2), (1, 2, 0), (2, 0, 1)]:
        assert tc[code(cw, 3)] is not None
    for ccw in [(0, 2, 1), (2, 1, 0), (1, 0, 2)]:
        assert tc[code(ccw, 3)] is None
        assert ta[code(ccw, 3)] is not None
    # in a simulation: a grid holding only one vertex pattern of each handedness
    g = hex_torus(4)
    a, b, c = g.vertices[0].tolist()
    for pattern, fires in [((0, 1, 2), True), ((0, 2, 1), False)]:
        cols = [3] * g.n_cells  # colour 3 matches nothing
        for cell, col in zip((a, b, c), pattern, strict=True):
            cols[cell] = col
        sim = Sim(g, ReactionSet(chiral.copy().reactions, 4), cols, seed=0)
        sim.sweep(50)
        assert (sum(sim.fires) > 0) == fires


# ---- test 3: conservation by swap reactions ----------------------------------------------
def test_swap_reactions_conserve_counts() -> None:
    rs = ReactionSet(
        [
            Reaction("edge", [0, 1], [1, 0], 1.0),
            Reaction("edge", [1, 2], [2, 1], 0.7),
            Reaction("vertex", [0, 1, 2], [2, 0, 1], 0.5),
            Reaction("vertex", [0, 0, 1], [0, 1, 0], 0.5, chiral=False),
            Reaction("cell", [2], [2], 1.0),
        ],
        3,
    )
    g = hex_torus(16)
    sim = Sim(g, rs, initial_colours(g.n_cells, [0.5, 0.3, 0.2], 1), seed=2)
    c0 = sim.counts().copy()
    for _ in range(20):
        sim.sweep()
        assert (sim.counts() == c0).all()
    assert sum(sim.fires) > 0


# ---- test 4: shuffled control vs mean field -----------------------------------------------
def test_shuffled_matches_mean_field() -> None:
    rs = ReactionSet(
        [
            Reaction("cell", [0], [1], 0.4),
            Reaction("edge", [1, 2], [2, 2], 0.2),
            Reaction("vertex", [2, 2, 2], [2, 2, 0], 0.3),
        ],
        3,
    )
    g = hex_torus(128)
    sim = Sim(g, rs, initial_colours(g.n_cells, [0.6, 0.1, 0.3], 3), seed=4, shuffled=True)
    ts, fr = run_timeline(sim, 30, record_every=2)
    mf = integrate(rhs(rs), fr[0], ts)
    assert ts[-1] > 30  # long enough to approach the fixed point
    assert np.abs(fr - mf).max() < 0.02


# ---- test 5: determinism -------------------------------------------------------------------
def test_determinism_with_seed_and_timeline() -> None:
    level = load_level(LEVELS / "T02_rps.json")
    timeline = [(3, (0, "out", 1), (1, "out", 1)), (7, (0, "out", 1), (1, "out", 1))]

    def run() -> tuple[np.ndarray, np.ndarray, list[int]]:
        sim = make_sim(level, graph=hex_torus(16))
        ts, fr = run_timeline(sim, 12, timeline)
        return ts, fr, sim.fires

    a, b = run(), run()
    assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1]) and a[2] == b[2]
    sim = make_sim(level, seed=99, graph=hex_torus(16))
    _, fr = run_timeline(sim, 12, timeline)
    assert not np.array_equal(fr, a[1])


# ---- test 6: goals -------------------------------------------------------------------------
def test_goal_extinct() -> None:
    gt = GoalTracker([{"type": "extinct", "color": 1}])
    assert gt.update(0.0, np.array([5, 1, 4])) == "running"
    assert gt.update(1.0, np.array([6, 0, 4])) == "won"


def test_goal_hold_and_fail() -> None:
    goal = {"type": "hold", "bands": [[0.2, 0.5], [0.2, 0.5], [0.2, 0.5]], "duration": 2.0}
    gt = GoalTracker([goal], fail=[{"type": "extinct", "color": 2}])
    assert gt.update(0.0, np.array([3, 3, 4])) == "running"
    assert gt.update(1.5, np.array([3, 3, 4])) == "running"
    assert gt.update(1.8, np.array([6, 2, 2])) == "running"  # leaves the band: clock resets
    assert gt.update(2.0, np.array([3, 3, 4])) == "running"
    assert gt.progress(3.0) == [0.5]
    assert gt.update(4.0, np.array([3, 3, 4])) == "won"
    gt2 = GoalTracker([goal], fail=[{"type": "extinct", "color": 2}])
    assert gt2.update(0.0, np.array([5, 5, 0])) == "failed"


def test_goal_reach_and_all_goals() -> None:
    reach = {"type": "reach", "target": [0.5, 0.5, 0.0], "tol": 0.05, "duration": 1.0}
    gt = GoalTracker([reach, {"type": "extinct", "color": 2}])
    assert gt.update(0.0, np.array([52, 48, 0])) == "running"
    assert gt.update(0.5, np.array([47, 53, 0])) == "running"
    assert gt.update(1.0, np.array([50, 50, 0])) == "won"
    gt = GoalTracker([reach])
    gt.update(0.0, np.array([60, 40, 0]))  # 0.1 away: outside tolerance
    assert gt.update(5.0, np.array([60, 40, 0])) == "running"


# ---- test 7: swaps --------------------------------------------------------------------------
def test_swaps_conserve_multiset_and_respect_locks() -> None:
    rs = ReactionSet(
        [Reaction("edge", [0, 1], [0, 0], 1.0), Reaction("vertex", [0, 1, 2], [2, 0, 1], 0.3)],
        3,
        locked={(0, "in", 0)},
    )
    ms = rs.multiset()
    rng = np.random.default_rng(0)
    blocks = rs.blocks()
    free = [b for b in blocks if b not in rs.locked]
    for _ in range(200):
        i, j = rng.choice(len(free), 2, replace=False)
        rs.swap(free[i], free[j])
        assert rs.multiset() == ms
    assert rs.get((0, "in", 0)) == 0
    with pytest.raises(ValueError):
        rs.swap((0, "in", 0), (1, "out", 2))


def test_sim_swap_rebuilds_tables() -> None:
    g = hex_torus(8)
    rs = ReactionSet([Reaction("edge", [0, 1], [1, 1], 1.0)], 2)
    sim = Sim(g, rs, [1] * g.n_cells, seed=0)  # all colour 1: 0|1 never matches
    sim.sweep(5)
    assert sum(sim.fires) == 0
    sim.swap((0, "in", 0), (0, "out", 0))  # now 1|1 -> 0|1
    assert [rs.reactions[0].inp, rs.reactions[0].out] == [[1, 1], [0, 1]]
    sim.sweep(2)
    assert sum(sim.fires) > 0


# ---- level files ------------------------------------------------------------------------------
@pytest.mark.parametrize("path", sorted(LEVELS.glob("*.json")), ids=lambda p: p.name)
def test_level_files_load(path: Path) -> None:
    level = load_level(path)
    d = json.loads(path.read_text())
    assert level.id == d["id"]
    rs = level.reaction_set()
    assert rs.max_propensity() > 0
    sim = make_sim(level, graph=hex_torus(8), burn_in=False)
    sim.sweep(2)


# ---- raster -----------------------------------------------------------------------------------
def test_hex_pixel_map_covers_cells_at_their_centres() -> None:
    from mosaic_model.raster import hex_pixel_map

    g = hex_torus(8)
    size = 10.0
    pm = hex_pixel_map(g, size)
    ids = pm[pm >= 0]
    counts = np.bincount(ids, minlength=g.n_cells)
    # every cell drawn with at least the area of an inset hexagon (edge cells also get the
    # wrapped partial hexes at the image border)
    inset = 3 * np.sqrt(3) / 2 * size**2 * (1 - 0.12) ** 2
    assert (counts > 0.95 * inset).all()
    ys, xs = np.nonzero(pm >= 0)
    for cell in [9, 27, 36, 50]:  # interior cells
        sel = pm[ys, xs] == cell
        cx = xs[sel].mean() + 0.5 - size * np.sqrt(3) / 2
        cy = ys[sel].mean() + 0.5 - size
        assert np.allclose([cx, cy], g.centers[cell] * size, atol=1.0)


# ---- partial sweeps ---------------------------------------------------------------------------
def test_advance_partial_sweeps_keeps_time_and_sweep_count() -> None:
    g = hex_torus(8)
    rs = ReactionSet([Reaction("edge", [0, 1], [1, 0], 2.0), Reaction("cell", [0], [0], 1.0)], 2)
    sim = Sim(g, rs, initial_colours(g.n_cells, [0.5, 0.5], 0), seed=1)
    for _ in range(37):
        sim.advance(0.013)
    assert sim.t == pytest.approx(37 * 0.013, abs=1.0 / (sim.K * sim.n_elig))
    assert sim.sweeps == int(sim.t * sim.K)
    sim.sweep()  # finishes the partial sweep: time lands on a whole number of sweeps
    assert sim.t * sim.K == pytest.approx(round(sim.t * sim.K))
