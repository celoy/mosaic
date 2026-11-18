"""Run directories for headless outputs: never overwrite, record parameters and git state."""

from __future__ import annotations

import datetime as _dt
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"


def git_state() -> dict:
    def _git(*args: str) -> str:
        try:
            return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()
        except (OSError, subprocess.CalledProcessError):
            return ""

    return {"commit": _git("rev-parse", "HEAD"), "dirty": bool(_git("status", "--porcelain"))}


def new_run_dir(kind: str, name: str, params: dict) -> Path:
    """Create runs/<kind>/<name>_<timestamp>/ with params.json; append a line to runs/LOG.md."""
    stamp = _dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    d = RUNS / kind / f"{name}_{stamp}"
    k = 1
    while d.exists():
        d = RUNS / kind / f"{name}_{stamp}-{k}"
        k += 1
    d.mkdir(parents=True)
    meta = {"name": name, "time": stamp, "git": git_state(), "params": params}
    (d / "params.json").write_text(json.dumps(meta, indent=2))
    RUNS.mkdir(exist_ok=True)
    with open(RUNS / "LOG.md", "a") as f:
        g = meta["git"]
        f.write(
            f"- {stamp} {kind}/{d.name}: git {g['commit'][:8]}{'+dirty' if g['dirty'] else ''} "
            f"params {json.dumps(params)}\n"
        )
    return d


def write_csv(path: Path, header: list[str], rows) -> None:
    with open(path, "w") as f:
        f.write(",".join(header) + "\n")
        for r in rows:
            f.write(",".join(f"{v:.6g}" if isinstance(v, float) else str(v) for v in r) + "\n")
