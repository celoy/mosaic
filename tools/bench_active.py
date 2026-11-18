"""Compare the rejection algorithm (Sim) with the active-set algorithm (ActiveSim).

1. Equivalence: same initial state, many seeds, compare observables at fixed times.
2. Speed: wall time per unit of sim time on each test level, n = 64 and 128, early and late.

    python tools/bench_active.py [--quick]
"""

from __future__ import annotations

import argparse
import sys
import time
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
    load_level,
    make_sim,
)
from mosaic_model.sim_active import ActiveSim  # noqa: E402
from runlog import new_run_dir, write_csv  # noqa: E402


def unlike(col: np.ndarray, edges: np.ndarray) -> float:
    return float(np.mean(col[edges[:, 0]] != col[edges[:, 1]]))


def run_to(sim, t: float) -> None:
    if isinstance(sim, ActiveSim):
        sim.advance(t - sim.t)
    else:
        while sim.t < t - 1e-12:
            sim.advance(min(t - sim.t, 1.0 / sim.K))


# ---- 1. equivalence ---------------------------------------------------------------------
def equivalence(seeds: int) -> list[str]:
    g = hex_torus(32)
    half = (g.centers[:, 0] >= g.centers[:, 0].max() / 2).astype(int)
    cases = {
        # deterministic-ish: invasion fronts from a half-plane
        "front 0|1->0|0": (ReactionSet([Reaction("edge", [0, 1], [0, 0], 1.0)], 2), half, 1.5),
        # all three shapes, interior fixed point
        "cell+edge+vertex": (
            ReactionSet(
                [
                    Reaction("cell", [0], [1], 0.4),
                    Reaction("edge", [1, 2], [2, 2], 0.2),
                    Reaction("vertex", [2, 2, 2], [2, 2, 0], 0.3),
                ],
                3,
            ),
            initial_colours(g.n_cells, [0.6, 0.1, 0.3], 0),
            10.0,
        ),
        # chiral vertex reaction that changes counts (the two handednesses differ)
        "chiral (0,1,2)->(0,0,2)": (
            ReactionSet(
                [
                    Reaction("vertex", [0, 1, 2], [0, 0, 2], 1.0, chiral=True),
                    Reaction("edge", [0, 2], [2, 2], 0.3),
                    Reaction("cell", [2], [1], 0.2),
                ],
                3,
            ),
            initial_colours(g.n_cells, [1 / 3] * 3, 0),
            10.0,
        ),
        "RPS (T03 set)": (
            load_level(ROOT / "levels" / "T03_chiral.json").reaction_set(),
            initial_colours(g.n_cells, [1 / 3] * 3, 0),
            10.0,
        ),
    }
    lines = [
        f"Equivalence on n=32, {seeds} seeds per algorithm, same initial state; mean (s.e.)",
        "case                      observable   rejection          active             z",
    ]
    for name, (rs, cols0, t_end) in cases.items():
        obs = {"rej": [], "act": []}
        for seed in range(seeds):
            for kind in obs:
                rs_k = rs.copy()
                sim = (Sim if kind == "rej" else ActiveSim)(g, rs_k, cols0, seed=1000 + seed)
                run_to(sim, t_end)
                c = sim.colours.astype(np.int64)
                fr = np.bincount(c, minlength=rs.palette) / g.n_cells
                obs[kind].append([*fr[: min(3, rs.palette)], unlike(c, g.edges)])
        a, b = np.array(obs["rej"]), np.array(obs["act"])
        labels = [f"x{i}" for i in range(min(3, rs.palette))] + ["rho"]
        for j, lab in enumerate(labels):
            ma, mb = a[:, j].mean(), b[:, j].mean()
            sa, sb = a[:, j].std(ddof=1) / np.sqrt(seeds), b[:, j].std(ddof=1) / np.sqrt(seeds)
            z = (ma - mb) / np.hypot(sa, sb) if np.hypot(sa, sb) > 0 else 0.0
            lines.append(
                f"{name:25s} {lab:5s}   {ma:.4f} ({sa:.4f})    {mb:.4f} ({sb:.4f})   {z:+.2f}"
            )
    return lines


# ---- 2. speed ---------------------------------------------------------------------------------
def speed(sizes: list[int], t_run: float) -> tuple[list[str], list[list]]:
    rows = []
    lines = [
        f"Wall time per unit sim time, after the level's burn-in, run for t = {t_run:g}.",
        "'start' = the level's stationary start; 'front' = sparse activity: one straight",
        "invasion front A|B -> A|A crossing a half-filled grid (about 2/n of edges active).",
        "level  n    stage  fire%(rej)  active%  rej s/t   act s/t   speed-up (rej/act)",
    ]
    for n in sizes:
        g = hex_torus(n)
        for path in sorted((ROOT / "levels").glob("*.json")):
            lev = load_level(path)
            base = make_sim(lev, graph=g)
            stages = [("start", base.col)]
            for stage, cols in stages:
                res = {}
                for kind, cls in (("rej", Sim), ("act", ActiveSim)):
                    sim = cls(g, lev.reaction_set(), cols, seed=5)
                    f0 = sum(sim.fires)
                    a0 = (sim.sweeps * sim.n_elig + sim._pos) if kind == "rej" else 0
                    if kind == "act":
                        a0 = sim.attempts_done
                        m0 = sim.n_active
                    t0 = time.perf_counter()
                    run_to(sim, sim.t + t_run)
                    wall = time.perf_counter() - t0
                    nf = sum(sim.fires) - f0
                    if kind == "rej":
                        na = sim.sweeps * sim.n_elig + sim._pos - a0
                        res["fire"] = nf / max(na, 1)
                        res["n_elig"] = sim.n_elig
                    else:
                        res["active"] = 0.5 * (m0 + sim.n_active) / res["n_elig"]
                    res[kind] = wall / t_run
                sp = res["rej"] / res["act"]
                rows.append(
                    [lev.id, n, stage, res["fire"], res["active"], res["rej"], res["act"], sp]
                )
                lines.append(
                    f"{lev.id:5s}  {n:<4d} {stage:6s} {100 * res['fire']:9.2f}  "
                    f"{100 * res['active']:7.2f}  {res['rej']:8.4f}  {res['act']:8.4f}  {sp:8.2f}"
                )
        # sparse case: a straight front (one reaction, two interfaces on the torus)
        half = (g.centers[:, 0] >= g.centers[:, 0].max() * 0.25).astype(int).tolist()
        res = {}
        for kind, cls in (("rej", Sim), ("act", ActiveSim)):
            rs = ReactionSet([Reaction("edge", [0, 1], [0, 0], 1.0)], 2)
            sim = cls(g, rs, half, seed=5)
            t_front = min(t_run, 0.1 * n)  # well before the fronts meet
            t0 = time.perf_counter()
            run_to(sim, t_front)
            res[kind] = (time.perf_counter() - t0) / t_front
            if kind == "rej":
                res["fire"] = sum(sim.fires) / (sim.sweeps * sim.n_elig + sim._pos)
                res["n_elig"] = sim.n_elig
            else:
                res["active"] = sim.n_active / res["n_elig"]
        sp = res["rej"] / res["act"]
        rows.append(["front", n, "front", res["fire"], res["active"], res["rej"], res["act"], sp])
        lines.append(
            f"front  {n:<4d} front  {100 * res['fire']:9.2f}  {100 * res['active']:7.2f}  "
            f"{res['rej']:8.4f}  {res['act']:8.4f}  {sp:8.2f}"
        )
    return lines, rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    p = {
        "seeds": 6 if a.quick else 24,
        "sizes": [64] if a.quick else [64, 128],
        "t_run": 5.0 if a.quick else 20.0,
    }
    d = new_run_dir("bench", "active_vs_rejection", p)
    lines = equivalence(p["seeds"])
    lines.append("")
    sl, rows = speed(p["sizes"], p["t_run"])
    lines += sl
    write_csv(
        d / "speed.csv",
        [
            "level",
            "n",
            "stage",
            "fire_frac",
            "active_frac",
            "rej_s_per_t",
            "act_s_per_t",
            "speedup",
        ],
        rows,
    )
    text = "\n".join(lines) + "\n"
    (d / "summary.txt").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
