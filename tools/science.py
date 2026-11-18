"""Headless science checks (CLAUDE.md, Python twin, checks 1-4). CSV + PNG snapshots + summary
in runs/science/<check>_<timestamp>/.

    python tools/science.py voter|rps|chiral|levels|all [--quick]
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import tomllib
from collections.abc import Callable
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from mosaic_model import (  # noqa: E402
    Reaction,
    ReactionSet,
    Sim,
    hex_torus,
    initial_colours,
    integrate,
    load_level,
    make_sim,
    rhs,
)
from mosaic_model.raster import hex_pixel_map, render_rgb  # noqa: E402
from runlog import new_run_dir, write_csv  # noqa: E402

CFG = tomllib.loads((ROOT / "toy" / "viewer.toml").read_text())


# ---- observables ---------------------------------------------------------------------------
def unlike_edges(sim: Sim) -> float:
    c = sim.colours
    e = sim.graph.edges
    return float(np.mean(c[e[:, 0]] != c[e[:, 1]]))


def vertex_stats(sim: Sim) -> tuple[float, float]:
    """(fraction of trichromatic vertices, chirality (n_cw - n_ccw) / (n_cw + n_ccw)),
    cw meaning colours 0, 1, 2 read clockwise."""
    c = sim.colours.astype(np.int64)
    v = sim.graph.vertices
    a, b, d = c[v[:, 0]], c[v[:, 1]], c[v[:, 2]]
    tri = (a != b) & (b != d) & (a != d) & (a < 3) & (b < 3) & (d < 3)
    cw = tri & (((a + 1) % 3) == b)
    n_tri = int(tri.sum())
    n_cw = int(cw.sum())
    chi = (2 * n_cw - n_tri) / n_tri if n_tri else 0.0
    return n_tri / len(v), chi


def trace(
    sim: Sim,
    t_max: float,
    dt_rec: float,
    stop_when_one_left: bool = True,
    extra: Callable[[Sim], list[float]] | None = None,
) -> list[list[float]]:
    """Run to t_max, recording [t, fractions..., extra...] every dt_rec."""
    rows = []
    next_rec = sim.t
    while True:
        if sim.t >= next_rec - 1e-12:
            rows.append([sim.t, *sim.fractions().tolist(), *(extra(sim) if extra else [])])
            next_rec += dt_rec
        if sim.t >= t_max:
            break
        if stop_when_one_left and (sim.counts() > 0).sum() <= 1:
            rows.append([sim.t, *sim.fractions().tolist(), *(extra(sim) if extra else [])])
            break
        sim.sweep()
    return rows


def first_extinction(rows: list[list[float]], palette: int) -> float:
    for r in rows:
        if min(r[1 : 1 + palette]) == 0.0:
            return r[0]
    return float("nan")


def snapshot(sim: Sim, path: Path, size: float = 5.0, empty: int | None = None) -> None:
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    import pygame

    pal = np.array([pygame.Color(h)[:3] for h in CFG["palette"]["colours"]])
    if empty is not None:
        pal[empty] = pygame.Color(CFG["palette"]["empty"])[:3]
    pm = hex_pixel_map(sim.graph, size)
    rgb = render_rgb(pm, sim.colours, pal, pygame.Color(CFG["palette"]["background"])[:3])
    pygame.image.save(pygame.surfarray.make_surface(rgb.swapaxes(0, 1)), str(path))


def _summary(d: Path, lines: list[str]) -> None:
    text = "\n".join(lines) + "\n"
    (d / "summary.txt").write_text(text)
    print(text)


# ---- check 1: voter --------------------------------------------------------------------
def check_voter(quick: bool) -> None:
    p = {"n": 64 if quick else 128, "seeds": 2 if quick else 3, "t_max": 100.0 if quick else 500.0}
    d = new_run_dir("science", "voter", p)
    rs = ReactionSet(
        [Reaction("edge", [0, 1], [0, 0], 1.0), Reaction("edge", [1, 0], [1, 1], 1.0)], 2
    )
    g = hex_torus(p["n"])
    rows = []
    for seed in range(p["seeds"]):
        sim = Sim(g, rs, initial_colours(g.n_cells, [0.5, 0.5], seed), seed=seed + 100)
        next_t = 0.5
        while sim.t < p["t_max"]:
            sim.sweep()
            if sim.t >= next_t:
                rows.append([seed, sim.sweeps, sim.t, unlike_edges(sim), sim.fractions()[0]])
                next_t *= 1.25
        snapshot(sim, d / f"voter_seed{seed}_t{int(sim.t)}.png", 3.0 if p["n"] > 64 else 5.0)
    write_csv(d / "voter.csv", ["seed", "sweep", "t", "rho_unlike", "x0"], rows)
    a = np.array(rows, dtype=float)
    late = a[a[:, 2] >= 5.0]
    t, rho = late[:, 2], late[:, 3]
    # 1/rho vs ln t (voter, d = 2) against log rho vs log t (power law)
    c1 = np.polyfit(np.log(t), 1 / rho, 1)
    r1 = np.corrcoef(np.log(t), 1 / rho)[0, 1] ** 2
    c2 = np.polyfit(np.log(t), np.log(rho), 1)
    r2 = np.corrcoef(np.log(t), np.log(rho))[0, 1] ** 2
    lines = [
        f"Voter model, hex torus n={p['n']}, {p['seeds']} seeds, t <= {p['t_max']}",
        f"K = {rs.max_propensity()} (one sweep = {1 / rs.max_propensity()} time units)",
        "rho(t), fraction of unlike edges, mean over seeds:",
    ]
    for tt in [1, 5, 20, 50, 100, 200, 500]:
        sel = np.abs(np.log(a[:, 2] / tt)) < 0.12
        if sel.any():
            lines.append(f"  t ~ {tt:4d}: rho = {a[sel, 3].mean():.4f}")
    lines += [
        f"fit 1/rho = {c1[0]:.3f} ln t + {c1[1]:.3f}   R^2 = {r1:.4f}",
        f"fit rho ~ t^{c2[0]:.3f}                     R^2 = {r2:.4f}",
        "expected (recalled): slow decay rho ~ 1/ln t in 2D, much slower than the t^(-1/2)",
        "of curvature-driven coarsening.",
    ]
    _summary(d, lines)


# ---- check 2: rock-paper-scissors with exchange ---------------------------------------------
def rps_set(eps: float) -> ReactionSet:
    rx = [
        Reaction("edge", [0, 1], [0, 0], 1.0),
        Reaction("edge", [1, 2], [1, 1], 1.0),
        Reaction("edge", [2, 0], [2, 2], 1.0),
    ]
    if eps > 0:
        rx += [
            Reaction("edge", [0, 1], [1, 0], eps),
            Reaction("edge", [1, 2], [2, 1], eps),
            Reaction("edge", [2, 0], [0, 2], eps),
        ]
    return ReactionSet(rx, 3)


def check_rps(quick: bool) -> None:
    p = {
        "grids": [[32, 60.0]] if quick else [[32, 1000.0], [64, 300.0]],  # [n, t_max]
        "seeds": 2 if quick else 4,
        "eps": [0, 1, 10] if quick else [0, 0.1, 0.3, 1, 3, 10, 30],
    }
    d = new_run_dir("science", "rps", p)
    rows, lines = (
        [],
        [
            "Cyclic invasion 0>1>2>0 (rate 1) + exchange of every unlike pair at rate eps,",
            f"hex torus, uniform start, {p['seeds']} seeds; 'mixed' = well-mixed control, eps = 0.",
            "n    t_max  eps      mode    extinct/seeds  mean t_ext  rho_unlike(end)  wall s",
        ],
    )
    for n, t_max in p["grids"]:
        g = hex_torus(n)
        cases = [(e, False) for e in p["eps"]] + [(0, True)]
        for eps, mixed in cases:
            ext, rho_end, t0 = [], [], time.perf_counter()
            for seed in range(p["seeds"]):
                cols = initial_colours(g.n_cells, [1 / 3] * 3, seed)
                sim = Sim(g, rps_set(eps), cols, seed + 7, mixed)
                r = trace(sim, t_max, 1.0, extra=lambda s: [unlike_edges(s)])
                rows += [[n, eps, int(mixed), seed, *row] for row in r]
                ext.append(first_extinction(r, 3))
                rho_end.append(r[-1][-1])
                if seed == 0:
                    tag = "mixed" if mixed else f"eps{eps:g}"
                    snapshot(sim, d / f"rps_n{n}_{tag}_t{int(sim.t)}.png")
            n_ext = int(np.sum(~np.isnan(ext)))
            mt = np.nanmean(ext) if n_ext else float("nan")
            lines.append(
                f"{n:<4d} {t_max:<6g} {eps:<8g} {'mixed' if mixed else 'grid':7s} "
                f"{n_ext}/{p['seeds']:<13d}{mt:10.1f}  {np.mean(rho_end):15.3f}  "
                f"{time.perf_counter() - t0:7.1f}"
            )
    write_csv(
        d / "rps.csv", ["n", "eps", "mixed", "seed", "t", "x0", "x1", "x2", "rho_unlike"], rows
    )
    lines += [
        "Expected (Reichenbach, Mobilia & Frey 2007, qualitatively): coexistence at low mobility,",
        "loss of species at high mobility. Their model has empty sites (May-Leonard); ours is",
        "the zero-sum cyclic invasion, whose mean field has neutral cycles (x0 x1 x2 conserved).",
    ]
    _summary(d, lines)


# ---- check 2b: May-Leonard (empty colour) ----------------------------------------------------
def may_leonard_set(eps: float, exchange: str = "all", sigma: float = 1.0, mu: float = 1.0):
    """Species 0, 1, 2 and empty 3: selection i|i+1 -> i|E, reproduction i|E -> i|i,
    exchange at rate eps between all unlike pairs ("all") or only with empty ("holes")."""
    e = 3
    rx = []
    for i in range(3):
        rx.append(Reaction("edge", [i, (i + 1) % 3], [i, e], sigma))
    for i in range(3):
        rx.append(Reaction("edge", [i, e], [i, i], mu))
    if eps > 0:
        pairs = [(0, 1), (1, 2), (2, 0)] if exchange == "all" else []
        pairs += [(i, e) for i in range(3)]
        rx += [Reaction("edge", [a, b], [b, a], eps) for a, b in pairs]
    return ReactionSet(rx, 4)


def check_mayleonard(quick: bool) -> None:
    p = {
        "n": 64,
        "seeds": 1 if quick else 2,
        "t_max": 60.0 if quick else 200.0,
        "eps": [0, 1] if quick else [0, 0.3, 1, 3, 10],
        "exchange": ["all", "holes"],
    }
    d = new_run_dir("science", "mayleonard", p)
    g = hex_torus(p["n"])
    rows, lines = (
        [],
        [
            "May-Leonard: selection i|i+1 -> i|E, reproduction i|E -> i|i (rates 1), exchange eps",
            f"n={p['n']}, start 1/4 each, {p['seeds']} seeds, t <= {p['t_max']}.",
            "exchange  eps    extinct  mean t_ext  x_empty(end)  rho_unlike(end)",
        ],
    )
    for ex in p["exchange"]:
        for eps in p["eps"]:
            if ex == "holes" and eps == 0:
                continue
            ext, xe, rho = [], [], []
            for seed in range(p["seeds"]):
                sim = Sim(
                    g,
                    may_leonard_set(eps, ex),
                    initial_colours(g.n_cells, [0.25] * 4, seed),
                    seed + 3,
                )
                r = trace(sim, p["t_max"], 1.0, extra=lambda s: [unlike_edges(s)])
                rows += [[ex, eps, seed, *row] for row in r]
                # extinction of a species (colours 0..2), not of the empty colour
                ext.append(next((row[0] for row in r if min(row[1:4]) == 0.0), float("nan")))
                xe.append(r[-1][4])
                rho.append(r[-1][5])
                if seed == 0:
                    snapshot(sim, d / f"ml_{ex}_eps{eps:g}_t{int(sim.t)}.png", empty=3)
            n_ext = int(np.sum(~np.isnan(ext)))
            mt = np.nanmean(ext) if n_ext else float("nan")
            lines.append(
                f"{ex:9s} {eps:<6g} {n_ext}/{p['seeds']}  {mt:10.1f}  "
                f"{np.mean(xe):12.3f}  {np.mean(rho):15.3f}"
            )
    write_csv(
        d / "mayleonard.csv",
        ["exchange", "eps", "seed", "t", "x0", "x1", "x2", "xE", "rho_unlike"],
        rows,
    )
    _summary(d, lines)


# ---- check 3: chiral vertex rotation --------------------------------------------------------
def check_chiral(quick: bool) -> None:
    p = {"n": 64, "seeds": 2 if quick else 3, "t_max": 60.0 if quick else 200.0, "k_rot": 0.3}
    d = new_run_dir("science", "chiral", p)
    g = hex_torus(p["n"])
    rot = [0, 1, 2], [2, 0, 1]
    cases = {
        "rot_chiral": [Reaction("vertex", *rot, 1.0, chiral=True)],
        "rot_achiral": [Reaction("vertex", *rot, 1.0, chiral=False)],
        "rps": rps_set(0).reactions,
        "rps+rot_chiral": rps_set(0).reactions + [Reaction("vertex", *rot, p["k_rot"], True)],
        "rps+rot_achiral": rps_set(0).reactions + [Reaction("vertex", *rot, p["k_rot"], False)],
    }
    rows = []
    lines = [
        f"Vertex rotation (0,1,2) -> (2,0,1), alone (k=1) and added to cyclic invasion "
        f"(k={p['k_rot']}), n={p['n']}, {p['seeds']} seeds, t <= {p['t_max']}.",
        "case               extinct  mean t_ext  rho_unlike  trichromatic  chirality  (end, mean)",
    ]
    for name, rx in cases.items():
        ext, ends = [], []
        for seed in range(p["seeds"]):
            rs = ReactionSet(
                [Reaction(r.shape, list(r.inp), list(r.out), r.rate, r.chiral) for r in rx], 3
            )
            sim = Sim(g, rs, initial_colours(g.n_cells, [1 / 3] * 3, seed), seed + 11)

            def extra(s: Sim) -> list[float]:
                return [unlike_edges(s), *vertex_stats(s)]

            r = trace(sim, p["t_max"], 1.0, extra=extra)
            rows += [[name, seed, *row] for row in r]
            ext.append(first_extinction(r, 3))
            ends.append(r[-1][4:])
            if seed == 0:
                snapshot(sim, d / f"chiral_{name}_t{int(sim.t)}.png")
        e = np.array(ends)
        n_ext = int(np.sum(~np.isnan(ext)))
        mt = np.nanmean(ext) if n_ext else float("nan")
        lines.append(
            f"{name:18s} {n_ext}/{p['seeds']}  {mt:10.1f}  {e[:, 0].mean():10.3f}  "
            f"{e[:, 1].mean():12.3f}  {e[:, 2].mean():9.3f}"
        )
    lines += [
        "Note: chirality is identically 0 on a torus: #cw - #ccw trichromatic triangles is the",
        "degree of the colouring map (Sperner-type argument), zero on a closed surface.",
        "Rotation alone: each firing is a bijection, so the uniform random state is stationary",
        "(chiral or not); no pattern forms.",
    ]
    write_csv(
        d / "chiral.csv",
        ["case", "seed", "t", "x0", "x1", "x2", "rho_unlike", "trichromatic", "chirality"],
        rows,
    )
    _summary(d, lines)


# ---- check 4: normal against shuffled for each level ----------------------------------------
def check_levels(quick: bool) -> None:
    paths = sorted((ROOT / "levels").glob("*.json"))
    p = {
        "seeds": 2 if quick else 4,
        "t_max": 60.0 if quick else 200.0,
        "levels": [x.name for x in paths],
    }
    d = new_run_dir("science", "levels", p)
    rows = []
    lines = [
        "Initial arrangement of each level, no switches; normal vs shuffled (well-mixed per",
        f"attempt), {p['seeds']} seeds, burn-in as in the level, then t <= {p['t_max']}.",
        "level mode     extinct  mean t_ext  final fractions (mean)     mean-field at t_max",
    ]
    for path in paths:
        lev = load_level(path)
        pal = lev.palette
        for mixed in (False, True):
            ext, fin, mf_end = [], [], []
            for seed in range(p["seeds"]):
                sim = make_sim(lev, seed=1000 + seed, shuffled=mixed)
                x0 = sim.fractions()
                r = trace(sim, p["t_max"], 1.0)
                rows += [[lev.id, int(mixed), seed, *row, *([0.0] * (4 - pal))] for row in r]
                ext.append(first_extinction(r, pal))
                fin.append(r[-1][1 : 1 + pal])
                mf = integrate(rhs(lev.reaction_set()), x0, np.array([0.0, p["t_max"]]), 1e-2)
                mf_end.append(mf[-1])
                if seed == 0:
                    snapshot(sim, d / f"{lev.id}_{'mixed' if mixed else 'grid'}.png")
            n_ext = int(np.sum(~np.isnan(ext)))
            mt = np.nanmean(ext) if n_ext else float("nan")
            lines.append(
                f"{lev.id:5s} {'mixed' if mixed else 'grid':8s} {n_ext}/{p['seeds']}  {mt:10.1f}  "
                f"{np.round(np.mean(fin, axis=0), 3)!s:26s} {np.round(np.mean(mf_end, axis=0), 3)}"
            )
    write_csv(d / "levels.csv", ["level", "mixed", "seed", "t", "x0", "x1", "x2", "x3"], rows)
    _summary(d, lines)


CHECKS = {
    "voter": check_voter,
    "rps": check_rps,
    "mayleonard": check_mayleonard,
    "chiral": check_chiral,
    "levels": check_levels,
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("check", choices=[*CHECKS, "all"])
    ap.add_argument("--quick", action="store_true", help="small, fast version")
    a = ap.parse_args()
    for name, fn in CHECKS.items():
        if a.check in (name, "all"):
            t0 = time.perf_counter()
            fn(a.quick)
            print(f"[{name}] {time.perf_counter() - t0:.1f} s\n")


if __name__ == "__main__":
    main()
