"""Edge-invasion networks (A|B -> A|A or B|B): mean-field coexistence point against the grid.

For pure invasion reactions the mean-field equations are replicator (antisymmetric
Lotka-Volterra) equations, dx_i/dt = 6 x_i (A x)_i with A_ij = k(i beats j) = -A_ji
(edge weight 3 x 2 orientations). The interior fixed point solves A x* = 0, sum x* = 1; it is
neutrally stable (H = sum_i x*_i ln x_i is conserved), so mean-field orbits cycle around x*
and x* is their time average.

Usage:
  .venv/bin/python tools/tournament.py levels/sandbox/S20261118-184837.json [--t 200] [--seeds 2]
Also runs the same network with every rate set to 1 (control). CSV in runs/tournament/.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "model"))

from mosaic_model.level import load_level, make_sim  # noqa: E402


def invasion_matrix(level) -> np.ndarray:
    """A_ij = rate at which i invades j (edge reactions i|j -> i|i), antisymmetrised."""
    n = level.palette
    a = np.zeros((n, n))
    for r in level.reactions:
        if r["rate"] == 0:
            continue
        if r["shape"] != "edge":
            raise ValueError("only edge reactions are handled")
        p, q = r["in"]
        w = r["out"][0]
        if r["out"] != [w, w] or w not in (p, q) or p == q:
            raise ValueError(f"not an invasion reaction: {r}")
        lo = q if w == p else p
        a[w, lo] += r["rate"]
        a[lo, w] -= r["rate"]
    return a


def fixed_point(a: np.ndarray) -> np.ndarray:
    """x with A x = 0, normalised to sum 1 (one-dimensional kernel assumed)."""
    _, s, vt = np.linalg.svd(a)
    if s[-2] < 1e-9 * s[0]:
        raise ValueError("kernel of A has dimension > 1: no unique coexistence point")
    x = vt[-1]
    return x / x.sum()


def run(level, seed: int, shuffled: bool, t_end: float, dt: float = 1.0) -> np.ndarray:
    """Fractions every dt (after the level's burn-in)."""
    sim = make_sim(level, seed=seed, shuffled=shuffled)
    out = [sim.fractions()]
    while sim.t < t_end - 1e-9:
        sim.advance(dt)
        out.append(sim.fractions())
    return np.array(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("level")
    ap.add_argument("--t", type=float, default=200.0, help="run length (time units)")
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--no-control", action="store_true", help="skip the all-k=1 control")
    a = ap.parse_args()

    level = load_level(a.level)
    cases = [("level k", level)]
    if not a.no_control:
        ctrl = load_level(a.level)
        ctrl.reactions = [dict(r, rate=1.0 if r["rate"] > 0 else 0.0) for r in ctrl.reactions]
        cases.append(("all k=1", ctrl))

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir = ROOT / "runs" / "tournament" / f"{Path(a.level).stem}_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)
    git = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True)
    (out_dir / "params.json").write_text(
        json.dumps({"level": a.level, "t": a.t, "seeds": a.seeds, "git": git.stdout.strip()})
    )
    rows = []
    n = level.palette
    for name, lv in cases:
        xs = fixed_point(invasion_matrix(lv))
        print(f"\n{name}: mean-field x* = {np.round(xs, 3)}")
        rows.append([name, "mean field x*", "", *np.round(xs, 4)])
        for shuffled in (False, True):
            mode = "shuffled" if shuffled else "grid"
            for s in range(a.seeds):
                t0 = time.perf_counter()
                x = run(lv, seed=100 + s, shuffled=shuffled, t_end=a.t)
                half = x[len(x) // 2 :]
                m, sd = half.mean(axis=0), half.std(axis=0)
                gone = [c for c in range(n) if x[-1, c] == 0]
                print(
                    f"  {mode:8s} seed {s}: mean (2nd half) {np.round(m, 3)}  "
                    f"sd {np.round(sd, 3)}  extinct {gone}  ({time.perf_counter() - t0:.0f} s)",
                    flush=True,
                )
                rows.append([name, mode, s, *np.round(m, 4)])
                np.savetxt(out_dir / f"{name.replace(' ', '_')}_{mode}_{s}.csv", x, delimiter=",")
    with open(out_dir / "summary.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["case", "mode", "seed", *[f"x{c}" for c in range(n)]])
        w.writerows(rows)
    print(f"\n-> {out_dir}")


if __name__ == "__main__":
    main()
