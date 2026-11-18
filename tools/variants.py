"""What does each ingredient of a level do? Run the level against simple variants.

Variants:
  level        as saved
  no-vertex    vertex reactions off (k = 0)
  mirror       every chiral vertex reaction replaced by its mirror image
               (a, b, c) -> (d, e, f)  becomes  (a, c, b) -> (d, f, e)
  shuffled     the level, well-mixed control (= mean field on a large grid)

Per run: time-averaged fractions over the second half and their sd, colours lost (among
those present at the start), density of unlike edges at the end (low = large domains),
and a PNG snapshot at the end.

  .venv/bin/python tools/variants.py levels/sandbox/S20261118-190742.json [--t 300] [--seeds 2]
Output: runs/variants/<level>_<stamp>/ (CSV per run, PNGs, summary.txt, params.json).
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from mosaic_model import load_level, make_sim  # noqa: E402
from science import snapshot, unlike_edges  # noqa: E402


def variant(path: str, name: str):
    lv = load_level(path)
    if name == "no-vertex":
        lv.reactions = [dict(r, rate=0.0) if r["shape"] == "vertex" else r for r in lv.reactions]
    elif name == "mirror":
        out = []
        for r in lv.reactions:
            if r["shape"] == "vertex" and r.get("chiral", True):
                a, b, c = r["in"]
                d, e, f = r["out"]
                r = dict(r, **{"in": [a, c, b], "out": [d, f, e]})
            out.append(r)
        lv.reactions = out
    return lv


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("level")
    ap.add_argument("--t", type=float, default=300.0)
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--dt", type=float, default=1.0, help="recording interval")
    a = ap.parse_args()

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out = ROOT / "runs" / "variants" / f"{Path(a.level).stem}_{stamp}"
    out.mkdir(parents=True, exist_ok=True)
    git = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True)
    (out / "params.json").write_text(
        json.dumps({"level": a.level, "t": a.t, "seeds": a.seeds, "git": git.stdout.strip()})
    )
    lines = [f"{a.level}  t = {a.t:g}  seeds = {a.seeds}", ""]
    for name in ("level", "no-vertex", "mirror", "shuffled"):
        lv = variant(a.level, "level" if name == "shuffled" else name)
        for s in range(a.seeds):
            t0 = time.perf_counter()
            sim = make_sim(lv, seed=100 + s, shuffled=(name == "shuffled"))
            present = sim.counts() > 0
            rows = [[sim.t, *sim.fractions(), unlike_edges(sim)]]
            t_lost: dict[int, float] = {}
            while sim.t < a.t - 1e-9:
                sim.advance(a.dt)
                x = sim.fractions()
                rows.append([sim.t, *x, unlike_edges(sim)])
                for c in np.flatnonzero(present & (x == 0)):
                    t_lost.setdefault(int(c), round(sim.t))
            r = np.array(rows)
            np.savetxt(out / f"{name}_{s}.csv", r, delimiter=",", header="t,x...,unlike_edges")
            snapshot(sim, out / f"{name}_{s}.png", size=3.0)
            half = r[len(r) // 2 :, 1:-1][:, present]
            line = (
                f"{name:9s} seed {s}: mean {np.round(half.mean(0), 3)} sd {np.round(half.std(0), 3)}"
                f"  unlike edges {r[-1, -1]:.3f}  lost {t_lost or '-'}"
                f"  ({time.perf_counter() - t0:.0f} s)"
            )
            print(line, flush=True)
            lines.append(line)
    lines += ["", "lost: {colour: time of extinction}; mean/sd over colours present at start"]
    (out / "summary.txt").write_text("\n".join(lines) + "\n")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
