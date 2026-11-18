"""Minimal pygame-ce viewer for a level (Phase 1).

    python toy/viewer.py levels/T02_rps.json [--seed S] [--shuffled]
    python toy/viewer.py levels/T02_rps.json --smoke 120   # headless test, saves a PNG

Click two unlocked blocks to swap them. Space: pause. 1/2/3: speed x1/x4/x16. R: restart.
S: screenshot. Esc: quit. Swaps are logged to runs/viewer/<level>_<time>/session.json.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
import tomllib
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from mosaic_model import GoalTracker, Level, Sim, load_level, make_sim  # noqa: E402
from mosaic_model.raster import hex_pixel_map, render_rgb  # noqa: E402
from mosaic_model.reactions import ARITY, Block  # noqa: E402

B = 30  # block size, px
GAP = 3
WHITE = (235, 235, 240)
GREY = (140, 140, 150)
DARK = (25, 25, 30)
LAMP_OFF = (60, 55, 40)
LAMP_ON = (255, 220, 90)


def load_cfg() -> dict:
    return tomllib.loads((ROOT / "toy" / "viewer.toml").read_text())


class Viewer:
    def __init__(self, level: Level, cfg: dict, seed: int | None, shuffled: bool) -> None:
        import pygame

        self.pg = pygame
        self.level = level
        self.cfg = cfg
        self.shuffled = shuffled
        self.seed = level.start.get("seed", 0) if seed is None else seed
        w, h = cfg["window"]["width"], cfg["window"]["height"]
        self.screen = pygame.display.set_mode((w, h))
        pygame.display.set_caption(f"Mosaic Shift viewer: {level.id}")
        self.font = pygame.font.SysFont("helvetica,arial", 15)
        self.small = pygame.font.SysFont("helvetica,arial", 12)
        self.big = pygame.font.SysFont("helvetica,arial", 28, bold=True)
        pal = cfg["palette"]
        self.pal = np.array([pygame.Color(c)[:3] for c in pal["colours"]])
        self.bg = pygame.Color(pal["background"])[:3]
        self.glyphs = pal["glyphs"]

        t0 = time.perf_counter()
        print(f"burn-in: {level.start.get('burn_in_sweeps', 0)} sweeps ...", flush=True)
        self.sim0 = make_sim(level, seed=self.seed, shuffled=shuffled)
        print(f"burn-in done in {time.perf_counter() - t0:.1f} s", flush=True)
        self.start_cols = list(self.sim0.col)
        g = self.sim0.graph
        size = min(
            cfg["window"]["grid_width"] / (math.sqrt(3) * (g.n + 0.5)),
            (h - 20) / (1.5 * g.n + 0.5),
        )
        self.pixmap = hex_pixel_map(g, size)
        self.grid_rect = pygame.Rect(10, 10, self.pixmap.shape[1], self.pixmap.shape[0])
        self.grid_surf = pygame.Surface(self.grid_rect.size)
        self.panel_x = self.grid_rect.right + 25
        self.restarts = 0
        self.session: list[dict] = []
        self.run_dir: Path | None = None
        self.reset(first=True)

    # ---- state ------------------------------------------------------------------------------
    def reset(self, first: bool = False) -> None:
        if not first:
            self.restarts += 1
            self.session.append({"event": "restart", "wall": time.time()})
        self.sim = Sim(
            self.sim0.graph,
            self.level.reaction_set(),
            self.start_cols,
            seed=self.seed + 1 + 7919 * self.restarts,
            shuffled=self.shuffled,
        )
        self.tracker = GoalTracker(self.level.goals, self.level.fail)
        self.hist_t: list[float] = [0.0]
        self.hist_x: list[np.ndarray] = [self.sim.fractions()]
        self.lamps = np.zeros(len(self.sim.fires))
        self.last_fires = list(self.sim.fires)
        self.last_t = 0.0
        self.selected: Block | None = None
        self.paused = False
        self.speed_i = 0
        self.due = 0.0
        self.rate_meas = 0.0

    def swap(self, a: Block, b: Block) -> None:
        self.sim.swap(a, b)
        ev = {
            "event": "swap",
            "sweep": self.sim.sweeps,
            "t": self.sim.t,
            "a": a,
            "b": b,
            "restart": self.restarts,
        }
        self.session.append(ev)
        print(f"swap at sweep {self.sim.sweeps} (t={self.sim.t:.2f}): {a} <-> {b}", flush=True)

    def step(self, dt_wall: float) -> None:
        if self.paused:
            return
        sp = self.cfg["speed"]
        target = sp["base_sweeps_per_s"] * sp["multipliers"][self.speed_i]
        self.due = min(self.due + dt_wall * target, max(2.0, 0.25 * target))
        t_end = time.perf_counter() + sp["frame_budget_ms"] * 1e-3
        done = 0
        while self.due >= 1.0 and time.perf_counter() < t_end:
            self.sim.sweep()
            self.due -= 1.0
            done += 1
            self.hist_t.append(self.sim.t)
            self.hist_x.append(self.sim.fractions())
            self.tracker.update(self.sim.t, self.sim.counts())
        if dt_wall > 0:
            self.rate_meas = 0.9 * self.rate_meas + 0.1 * done / dt_wall
        # lamps: firing rate per cell per unit time, exponentially smoothed
        dt = self.sim.t - self.last_t
        if dt > 0:
            df = np.array(self.sim.fires) - np.array(self.last_fires)
            inst = df / (dt * self.sim.graph.n_cells)
            a = 1.0 - math.exp(-dt / self.cfg["lamps"]["tau"])
            self.lamps += a * (inst - self.lamps)
            self.last_fires = list(self.sim.fires)
            self.last_t = self.sim.t
        # trim history
        w = self.cfg["chart"]["window"] * 1.2
        while len(self.hist_t) > 2 and self.hist_t[1] < self.sim.t - w:
            self.hist_t.pop(0)
            self.hist_x.pop(0)

    # ---- drawing ------------------------------------------------------------------------------
    def draw(self) -> None:
        pg = self.pg
        self.screen.fill(self.bg)
        rgb = render_rgb(self.pixmap, self.sim.colours, self.pal, self.bg)
        pg.surfarray.blit_array(self.grid_surf, rgb.swapaxes(0, 1))
        self.screen.blit(self.grid_surf, self.grid_rect)
        y = self._draw_header(10)
        y = self._draw_equations(y + 10)
        self._draw_chart(y + 10)
        self._draw_status()

    def _text(self, s: str, pos, font=None, col=WHITE) -> int:
        surf = (font or self.font).render(s, True, col)
        self.screen.blit(surf, pos)
        return surf.get_height()

    def _draw_header(self, y: int) -> int:
        lv, sim = self.level, self.sim
        x = self.panel_x
        y += self._text(f"{lv.id}  {lv.title}", (x, y))
        mode = "SHUFFLED (well-mixed)" if self.shuffled else "grid"
        mult = self.cfg["speed"]["multipliers"][self.speed_i]
        state = "PAUSED" if self.paused else f"x{mult}"
        y += self._text(
            f"t = {sim.t:7.1f}   sweep {sim.sweeps}   {state}   "
            f"{self.rate_meas:4.1f} sweeps/s   K={sim.K:g}   {mode}",
            (x, y + 2),
            self.small,
            GREY,
        )
        return y + 4

    def _block_rects(self, y: int) -> tuple[dict, list[tuple[int, int, int]]]:
        """Rects for every block, and (reaction index, row y, row height)."""
        rects: dict[Block, object] = {}
        rows = []
        x0 = self.panel_x + 40
        for i, r in enumerate(self.sim.reactions.reactions):
            m = ARITY[r.shape]
            h = 2 * B + GAP if r.shape == "vertex" else B
            for side, xs in (("in", x0), ("out", x0 + 2 * (B + GAP) + 50)):
                for p in range(m):
                    if r.shape == "vertex":
                        # clockwise: 0 top, 1 bottom right, 2 bottom left
                        dx, dy = [((B + GAP) // 2, 0), (B + GAP, B + GAP), (0, B + GAP)][p]
                    else:
                        dx, dy = p * (B + GAP), 0
                    rects[(i, side, p)] = self.pg.Rect(xs + dx, y + dy, B, B)
            rows.append((i, y, h))
            y += h + 14
        return rects, rows

    def _draw_glyph(self, kind: str, rect, col) -> None:
        pg = self.pg
        c = rect.center
        s = rect.width * 0.22
        if kind == "circle":
            pg.draw.circle(self.screen, col, c, s)
        elif kind == "triangle":
            pg.draw.polygon(
                self.screen,
                col,
                [(c[0], c[1] - s), (c[0] + s, c[1] + s * 0.8), (c[0] - s, c[1] + s * 0.8)],
            )
        elif kind == "square":
            pg.draw.rect(
                self.screen, col, pg.Rect(c[0] - s * 0.85, c[1] - s * 0.85, 1.7 * s, 1.7 * s)
            )
        else:
            pg.draw.polygon(
                self.screen,
                col,
                [
                    (c[0], c[1] - s * 1.2),
                    (c[0] + s, c[1]),
                    (c[0], c[1] + s * 1.2),
                    (c[0] - s, c[1]),
                ],
            )

    def _draw_block(self, b: Block, rect) -> None:
        pg = self.pg
        rs = self.sim.reactions
        c = rs.get(b)
        pg.draw.rect(self.screen, self.pal[c], rect, border_radius=4)
        self._draw_glyph(self.glyphs[c], rect, DARK)
        if b in rs.locked:
            # padlock badge on the top-right corner
            lx, ly = rect.right - 2, rect.top + 2
            pg.draw.circle(self.screen, DARK, (lx, ly), 8)
            pg.draw.rect(self.screen, WHITE, pg.Rect(lx - 4, ly - 1, 8, 6))
            pg.draw.arc(self.screen, WHITE, pg.Rect(lx - 3, ly - 6, 6, 9), 0, math.pi, 2)
        if b == self.selected:
            pg.draw.rect(self.screen, WHITE, rect.inflate(6, 6), 3, border_radius=6)

    def _draw_equations(self, y: int) -> int:
        pg = self.pg
        rects, rows = self._block_rects(y)
        self.rects = rects
        half = self.cfg["lamps"]["half"]
        for i, ry, h in rows:
            r = self.sim.reactions.reactions[i]
            # lamp
            lv = float(self.lamps[i])
            b = lv / (lv + half) if lv > 0 else 0.0
            col = [int(LAMP_OFF[k] + b * (LAMP_ON[k] - LAMP_OFF[k])) for k in range(3)]
            cy = ry + h // 2
            pg.draw.circle(self.screen, col, (self.panel_x + 14, cy), 11)
            pg.draw.circle(self.screen, GREY, (self.panel_x + 14, cy), 11, 1)
            # arrow
            ax = self.panel_x + 40 + 2 * (B + GAP) + 8
            pg.draw.line(self.screen, WHITE, (ax, cy), (ax + 32, cy), 2)
            pg.draw.polygon(
                self.screen, WHITE, [(ax + 36, cy), (ax + 28, cy - 5), (ax + 28, cy + 5)]
            )
            for b_, rect in rects.items():
                if b_[0] == i:
                    self._draw_block(b_, rect)
            info = f"k={r.rate:g}   {lv:.3f}/cell/t"
            if r.shape == "vertex":
                info = ("clockwise only   " if r.chiral else "either hand   ") + info
            self._text(info, (self.panel_x + 40 + 4 * (B + GAP) + 60, cy - 7), self.small, GREY)
        return rows[-1][1] + rows[-1][2] + 14 if rows else y

    def _draw_chart(self, y: int) -> None:
        pg = self.pg
        x = self.panel_x
        w = self.screen.get_width() - x - 15
        h = self.cfg["chart"]["height"]
        rect = pg.Rect(x, y, w, h)
        pg.draw.rect(self.screen, (40, 40, 48), rect)
        win = self.cfg["chart"]["window"]
        t1 = max(self.sim.t, win)
        t0 = t1 - win

        def px(t: float, v: float) -> tuple[float, float]:
            return x + (t - t0) / win * w, y + h - v * h

        overlay = pg.Surface(rect.size, pg.SRCALPHA)
        for goal in self.level.goals:
            if goal["type"] == "hold":
                for c, (lo, hi) in enumerate(goal["bands"]):
                    band = pg.Rect(0, h - hi * h, w, (hi - lo) * h)
                    pg.draw.rect(overlay, (*self.pal[c], 45), band)
            elif goal["type"] == "reach":
                for c, tg in enumerate(goal["target"]):
                    lo, hi = tg - goal["tol"], tg + goal["tol"]
                    pg.draw.rect(
                        overlay, (*self.pal[c], 60), pg.Rect(0, h - hi * h, w, (hi - lo) * h)
                    )
            elif goal["type"] == "extinct":
                pg.draw.rect(overlay, (*self.pal[goal["color"]], 160), pg.Rect(0, h - 4, w, 4))
        self.screen.blit(overlay, rect)
        for v in (0.25, 0.5, 0.75):
            pg.draw.line(self.screen, (60, 60, 70), px(t0, v), px(t1, v))
        ts = np.array(self.hist_t)
        xs = np.array(self.hist_x)
        sel = ts >= t0
        if sel.sum() >= 2:
            for c in range(xs.shape[1]):
                pts = [px(t, v) for t, v in zip(ts[sel], xs[sel, c], strict=True)]
                pg.draw.lines(self.screen, self.pal[c], False, pts, 2)
        self._text("fractions vs time", (x + 4, y + 3), self.small, GREY)
        # goals
        yy = y + h + 8
        for k, goal in enumerate(self.level.goals):
            pr = self.tracker.progress(self.sim.t)[k]
            yy += self._text(f"goal: {self._goal_text(goal)}   [{pr:4.0%}]", (x, yy)) + 2
        for f in self.level.fail:
            yy += self._text(f"fail if: {self._goal_text(f)}", (x, yy), col=GREY) + 2
        help_ = (
            "click two blocks to swap | space pause | 1/2/3 speed | R restart | "
            "S screenshot | Esc quit"
        )
        self._text(help_, (x, self.screen.get_height() - 22), self.small, GREY)

    @staticmethod
    def _goal_text(g: dict) -> str:
        if g["type"] == "extinct":
            return f"colour {g['color']} extinct"
        if g["type"] == "hold":
            b = ", ".join(f"{lo:g}-{hi:g}" for lo, hi in g["bands"])
            return f"hold fractions in [{b}] for t={g['duration']:g}"
        if g["type"] == "reach":
            return f"reach {g['target']} +/- {g['tol']:g} for t={g['duration']:g}"
        return json.dumps(g)

    def _draw_status(self) -> None:
        st = self.tracker.status
        if st == "running":
            return
        msg, col = ("GOAL REACHED", (120, 230, 140)) if st == "won" else ("FAILED", (240, 110, 90))
        s = self.big.render(f"{msg}  (R to restart)", True, col)
        r = s.get_rect(center=self.grid_rect.center)
        self.pg.draw.rect(self.screen, DARK, r.inflate(30, 20), border_radius=8)
        self.screen.blit(s, r)

    # ---- input --------------------------------------------------------------------------------
    def click(self, pos) -> None:
        hit = next((b for b, r in getattr(self, "rects", {}).items() if r.collidepoint(pos)), None)
        if hit is None:
            self.selected = None
            return
        if hit in self.sim.reactions.locked:
            return
        if self.selected is None:
            self.selected = hit
        elif hit == self.selected:
            self.selected = None
        else:
            self.swap(self.selected, hit)
            self.selected = None

    def save(self, screenshot: bool = False) -> Path:
        if self.run_dir is None:
            from runlog import new_run_dir

            self.run_dir = new_run_dir(
                "viewer",
                self.level.id,
                {"level": str(self.level.path), "seed": self.seed, "shuffled": self.shuffled},
            )
        (self.run_dir / "session.json").write_text(json.dumps(self.session, indent=1))
        if screenshot:
            p = self.run_dir / f"screen_{self.sim.sweeps:06d}_{self.restarts}.png"
            self.pg.image.save(self.screen, str(p))
            print(f"saved {p}", flush=True)
            return p
        return self.run_dir

    def loop(self, smoke_frames: int = 0) -> None:
        pg = self.pg
        clock = pg.time.Clock()
        frame = 0
        running = True
        while running:
            dt = clock.tick(self.cfg["window"]["fps"]) / 1000.0
            for ev in pg.event.get():
                if ev.type == pg.QUIT:
                    running = False
                elif ev.type == pg.MOUSEBUTTONDOWN and ev.button == 1:
                    self.click(ev.pos)
                elif ev.type == pg.KEYDOWN:
                    if ev.key == pg.K_ESCAPE:
                        running = False
                    elif ev.key == pg.K_SPACE:
                        self.paused = not self.paused
                    elif ev.key in (pg.K_1, pg.K_2, pg.K_3):
                        self.speed_i = {pg.K_1: 0, pg.K_2: 1, pg.K_3: 2}[ev.key]
                    elif ev.key == pg.K_r:
                        self.reset()
                    elif ev.key == pg.K_s:
                        self.save(screenshot=True)
            if smoke_frames:
                dt = 1.0 / self.cfg["window"]["fps"]
                if frame == 1:
                    self.speed_i = 2
                if frame == smoke_frames // 2:
                    free = [
                        b for b in self.sim.reactions.blocks() if b not in self.sim.reactions.locked
                    ]
                    self.click(self.rects[free[0]].center)
                    self.draw()
                    self.click(self.rects[free[-1]].center)
                if frame >= smoke_frames:
                    self.click(self.rects[free[0]].center)  # leave one block selected
                    self.draw()
                    self.save(screenshot=True)
                    running = False
            self.step(dt)
            self.draw()
            pg.display.flip()
            frame += 1
        if any(e["event"] == "swap" for e in self.session) or self.run_dir:
            print(f"session saved to {self.save()}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("level", type=Path)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--shuffled", action="store_true", help="well-mixed control")
    ap.add_argument("--smoke", type=int, default=0, help="headless: run N frames, save PNG")
    a = ap.parse_args()
    if a.smoke:
        os.environ["SDL_VIDEODRIVER"] = "dummy"
    import pygame

    pygame.init()
    v = Viewer(load_level(a.level), load_cfg(), a.seed, a.shuffled)
    v.loop(a.smoke)
    pygame.quit()


if __name__ == "__main__":
    main()
