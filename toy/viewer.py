"""Minimal pygame-ce viewer for a level (Phase 1).

    python toy/viewer.py levels/T02_rps.json [--seed S] [--shuffled]
    python toy/viewer.py levels/T02_rps.json --smoke 120   # headless test, saves a PNG

Click two unlocked blocks to swap them (only while running, not while paused).
Buttons or keys: Space pause, 1/2/3/4 speed x1/4, x1, x4, x16, R restart, S screenshot, Esc quit.
Zoom: mouse wheel or +/- (about the cursor), drag the grid to pan, 0 to see the whole world.
--n N plays on an N x N rectangular torus instead of the level's world.
--world torus:64 | open:64 | hextorus:37 | hexagon:37 overrides the world shape
(hextorus = hexagon with opposite sides glued; hexagon = open hexagon).
Swaps are logged to runs/viewer/<level>_<time>/session.json.
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
sys.path.insert(0, str(ROOT / "toy"))

from mosaic_model import GoalTracker, Level, Sim, load_level, make_sim  # noqa: E402
from mosaic_model.graph import SQRT3  # noqa: E402
from mosaic_model.raster import fit_size, render_rgb, view_pixel_map, world_centre  # noqa: E402
from mosaic_model.reactions import ARITY, Block  # noqa: E402
from palette import colour_names, glyphs, names, palette_rgb  # noqa: E402

RB = 16.0  # circumradius of an equation block (hexagon), px
GAP = 1.08  # spacing factor between neighbouring blocks (1 = touching)
WHITE = (235, 235, 240)
GREY = (140, 140, 150)
DARK = (25, 25, 30)
METER_FACE = (232, 222, 190)
METER_INK = (40, 36, 30)
NEEDLE = (200, 40, 30)
EQ_DX = 64  # x offset of the first equation block from the panel edge (room for the meter)


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
        # pygame's bundled font: no system font lookup (fc-list can hang on macOS)
        self.font = pygame.font.Font(None, 20)
        self.small = pygame.font.Font(None, 17)
        self.big = pygame.font.Font(None, 36)
        pal = cfg["palette"]
        self.palette_name = pal["default"]
        self.pal = palette_rgb(cfg, level.palette, self.palette_name)
        self.cnames = colour_names(cfg, level.palette, self.palette_name)
        self.bg = pygame.Color(pal["background"])[:3]
        self.glyphs = glyphs(cfg, level.palette)
        self.show_glyphs = bool(pal.get("show_glyphs", False))
        self.menu_open = False

        t0 = time.perf_counter()
        print(f"burn-in: {level.start.get('burn_in_sweeps', 0)} sweeps ...", flush=True)
        self.sim0 = make_sim(level, seed=self.seed, shuffled=shuffled)
        print(f"burn-in done in {time.perf_counter() - t0:.1f} s", flush=True)
        self.start_cols = list(self.sim0.col)
        g = self.sim0.graph
        gw = cfg["window"]["grid_width"]
        self.fit = fit_size(g, gw - 20, h - 20)  # whole world visible; periodic worlds tile out
        self.zoom_reset()
        self.grid_rect = pygame.Rect(0, 0, gw, h)
        self.dragging = False
        self.grid_surf = pygame.Surface(self.grid_rect.size)
        self.panel_x = self.grid_rect.right + 20
        self.restarts = 0
        self.session: list[dict] = []
        self.run_dir: Path | None = None
        self.buttons: dict[str, object] = {}
        self.notice = ""
        self.notice_until = 0.0
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
        self.needles = np.zeros(len(self.sim.fires))  # dial position in [0, 1]
        self.last_fires = list(self.sim.fires)
        self.last_t = 0.0
        self.selected: Block | None = None
        self.paused = False
        mults = self.cfg["speed"]["multipliers"]
        self.speed_i = mults.index(1) if 1 in mults else 0
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

    # ---- zoom and pan -------------------------------------------------------------------------
    def zoom_reset(self) -> None:
        self.size = self.fit
        self.centre = world_centre(self.sim0.graph)
        self.map_dirty = True

    def zoom(self, factor: float, at: tuple[int, int] | None = None) -> None:
        """Zoom by `factor` keeping the world point under pixel `at` fixed."""
        gw, h = self.cfg["window"]["grid_width"], self.cfg["window"]["height"]
        new = min(max(self.size * factor, self.fit), self.cfg["grid"]["max_cell_px"])
        if at is not None:
            wx = self.centre[0] + (at[0] - gw / 2) / self.size
            wy = self.centre[1] + (at[1] - h / 2) / self.size
            self.centre = (wx - (at[0] - gw / 2) / new, wy - (at[1] - h / 2) / new)
        self.size = new
        if self.size <= self.fit * 1.0001:
            self.centre = world_centre(self.sim0.graph)
        self.map_dirty = True

    def pan(self, dx: float, dy: float) -> None:
        self.centre = (self.centre[0] - dx / self.size, self.centre[1] - dy / self.size)
        g = self.sim0.graph
        if not g.periodic:  # keep some of an open world in view
            lo, hi = g.centers.min(axis=0), g.centers.max(axis=0)
            self.centre = (
                float(np.clip(self.centre[0], lo[0], hi[0])),
                float(np.clip(self.centre[1], lo[1], hi[1])),
            )
        self.map_dirty = True

    def _remap(self) -> None:
        if self.map_dirty:
            gw, h = self.cfg["window"]["grid_width"], self.cfg["window"]["height"]
            self.pixmap = view_pixel_map(
                self.sim0.graph, self.size, gw, h, self.cfg["grid"]["border"], self.centre
            )
            self.map_dirty = False

    def move_needles(self, dt_wall: float) -> None:
        """Needles swing toward the smoothed firing rate (log dial), with a short lag."""
        lc = self.cfg["lamps"]
        lo, hi = math.log10(lc["lo"]), math.log10(lc["hi"])
        target = np.clip((np.log10(np.maximum(self.lamps, 1e-12)) - lo) / (hi - lo), 0.0, 1.0)
        a = 1.0 - math.exp(-dt_wall / lc["needle_tau"]) if dt_wall > 0 else 1.0
        self.needles += a * (target - self.needles)

    def step(self, dt_wall: float) -> None:
        """Advance base_time_per_s * multiplier sim time units per wall second, in chunks of
        attempts (partial sweeps), within a per-frame time budget."""
        if self.paused:
            return
        sp = self.cfg["speed"]
        sim = self.sim
        rate = sp["base_time_per_s"] * sp["multipliers"][self.speed_i]
        self.due = min(self.due + dt_wall * rate, 0.25 * rate + 0.1)  # cap the backlog
        t_end = time.perf_counter() + sp["frame_budget_ms"] * 1e-3
        t_start = sim.t
        chunk = sp["chunk_attempts"]
        while self.due > 0 and time.perf_counter() < t_end:
            n = min(chunk, max(1, int(self.due * sim.K * sim.n_elig)))
            t0 = sim.t
            sim.attempts(n)
            self.due -= sim.t - t0
            self.hist_t.append(sim.t)
            self.hist_x.append(sim.fractions())
            self.tracker.update(sim.t, sim.counts())
        if dt_wall > 0:
            self.rate_meas = 0.9 * self.rate_meas + 0.1 * (sim.t - t_start) / dt_wall
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
        self._remap()
        rgb = render_rgb(self.pixmap, self.sim.colours, self.pal, self.bg)
        pg.surfarray.blit_array(self.grid_surf, rgb.swapaxes(0, 1))
        self.screen.blit(self.grid_surf, self.grid_rect)
        y = self._draw_header(10)
        y = self._draw_equations(y + 10)
        self._draw_chart(y + 10)
        self._draw_status()
        self._draw_menu()

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
        state = "PAUSED" if self.paused else _mult_label(mult)
        zoom = f"zoom {self.size / self.fit:.1f}x   " if self.size > self.fit * 1.001 else ""
        y += self._text(
            f"t = {sim.t:7.1f}   sweep {sim.sweeps}   {state}   {zoom}"
            f"{self.rate_meas:4.2f} t/s   K={sim.K:g}   {mode}",
            (x, y + 2),
            self.small,
            GREY,
        )
        return self._draw_buttons(y + 8)

    def _draw_buttons(self, y: int) -> int:
        pg = self.pg
        x = self.panel_x
        mults = self.cfg["speed"]["multipliers"]
        labels = [("pause", "Play" if self.paused else "Pause", 70)]
        labels += [(f"speed{i}", _mult_label(m), 48) for i, m in enumerate(mults)]
        labels += [("restart", "Restart", 70)]
        self.buttons = {}
        for key, text, w in labels:
            r = pg.Rect(x, y, w, 26)
            active = (key == "pause" and self.paused) or key == f"speed{self.speed_i}"
            pg.draw.rect(self.screen, (80, 80, 96) if active else (50, 50, 60), r, border_radius=5)
            pg.draw.rect(self.screen, GREY, r, 1, border_radius=5)
            t = self.font.render(text, True, WHITE)
            self.screen.blit(t, t.get_rect(center=r.center))
            self.buttons[key] = r
            x += w + 8
        # second row: palette menu and symbols toggle
        x = self.panel_x
        y2 = y + 32
        for key, text, w in (
            ("palette", f"Palette: {self.palette_name}", 170),
            ("glyphs", f"Symbols: {'on' if self.show_glyphs else 'off'}", 110),
        ):
            r = pg.Rect(x, y2, w, 26)
            pg.draw.rect(self.screen, (50, 50, 60), r, border_radius=5)
            pg.draw.rect(self.screen, GREY, r, 1, border_radius=5)
            t = self.font.render(text, True, WHITE)
            self.screen.blit(t, t.get_rect(center=r.center))
            self.buttons[key] = r
            x += w + 8
        x = self.panel_x
        if self.notice and time.time() < self.notice_until:
            self._text(self.notice, (x, y2 + 32), self.small, (240, 180, 90))
        elif self.paused:
            self._text("paused: no switching", (x, y2 + 32), self.small, GREY)
        return y2 + 50

    def _draw_menu(self) -> None:
        """Palette dropdown, drawn last (on top); each entry shows a swatch."""
        if not self.menu_open:
            return
        pg = self.pg
        r0 = self.buttons["palette"]
        y = r0.bottom + 2
        for name in names(self.cfg):
            r = pg.Rect(r0.x, y, 260, 26)
            pg.draw.rect(self.screen, (35, 35, 44), r)
            pg.draw.rect(self.screen, GREY, r, 1)
            self._text(name, (r.x + 8, r.y + 6), self.font)
            sw = palette_rgb(self.cfg, self.level.palette, name)
            for k, c in enumerate(sw):
                pg.draw.rect(self.screen, c, pg.Rect(r.x + 120 + k * 22, r.y + 5, 18, 16))
            self.buttons[f"pal:{name}"] = r
            y += 26

    @staticmethod
    def _hex_points(c: tuple[float, float], rad: float) -> list[tuple[float, float]]:
        """Pointy-top hexagon around c."""
        return [
            (
                c[0] + rad * math.cos(math.radians(60 * k - 90)),
                c[1] + rad * math.sin(math.radians(60 * k - 90)),
            )
            for k in range(6)
        ]

    def _block_layout(self, y: int) -> tuple[dict, list[tuple[int, int, int]]]:
        """Centre of every block, and (reaction index, row top y, row height).

        Blocks sit like cells on the grid: an edge is two horizontal neighbours, a vertex
        is three hexes around their shared corner, clockwise 0 (top), 1 (lower right),
        2 (lower left)."""
        cents: dict[Block, tuple[float, float]] = {}
        rows = []
        w = SQRT3 * RB * GAP  # centre spacing of neighbours
        x_in = self.panel_x + EQ_DX + RB
        x_out = x_in + 2 * w + 60
        for i, r in enumerate(self.sim.reactions.reactions):
            m = ARITY[r.shape]
            h = int(2 * RB + (1.5 * RB * GAP if r.shape == "vertex" else 0))
            cy0 = y + RB
            for side, xs in (("in", x_in), ("out", x_out)):
                for p in range(m):
                    if r.shape == "vertex":
                        dx, dy = [(w / 2, 0), (w, 1.5 * RB * GAP), (0, 1.5 * RB * GAP)][p]
                    else:
                        dx, dy = p * w, 0
                    cents[(i, side, p)] = (xs + dx, cy0 + dy)
            rows.append((i, y, h))
            y += h + 10
        return cents, rows

    def _draw_glyph(self, kind: str, c: tuple[float, float], col) -> None:
        pg = self.pg
        s = RB * 0.42
        x, y = c
        if kind == "circle":
            pg.draw.circle(self.screen, col, c, s)
        elif kind == "ring":
            pg.draw.circle(self.screen, col, c, s * 1.1, 3)
        elif kind == "triangle":
            tri = [(x, y - s), (x + s, y + s * 0.8), (x - s, y + s * 0.8)]
            pg.draw.polygon(self.screen, col, tri)
        elif kind == "square":
            pg.draw.rect(self.screen, col, pg.Rect(x - s * 0.8, y - s * 0.8, 1.6 * s, 1.6 * s))
        elif kind == "diamond":
            pg.draw.polygon(
                self.screen, col, [(x, y - s * 1.2), (x + s, y), (x, y + s * 1.2), (x - s, y)]
            )
        elif kind == "cross":
            pg.draw.line(self.screen, col, (x - s, y), (x + s, y), 3)
            pg.draw.line(self.screen, col, (x, y - s), (x, y + s), 3)
        elif kind == "x":
            pg.draw.line(self.screen, col, (x - s, y - s), (x + s, y + s), 3)
            pg.draw.line(self.screen, col, (x - s, y + s), (x + s, y - s), 3)
        elif kind == "bar":
            pg.draw.rect(self.screen, col, pg.Rect(x - s, y - s * 0.3, 2 * s, 0.6 * s))
        else:  # colour index as text
            t = self.small.render(kind, True, col)
            self.screen.blit(t, t.get_rect(center=c))

    def _draw_block(self, b: Block, c: tuple[float, float]) -> None:
        pg = self.pg
        rs = self.sim.reactions
        col = rs.get(b)
        if b == self.selected:  # white ring with a gap: visible on any block colour
            pg.draw.polygon(self.screen, WHITE, self._hex_points(c, RB + 7))
            pg.draw.polygon(self.screen, self.bg, self._hex_points(c, RB + 4))
        locked = b in rs.locked
        if locked:  # a fixed block sits in a grey frame, like a tile set in the panel
            pg.draw.polygon(self.screen, GREY, self._hex_points(c, RB + 3))
        pg.draw.polygon(self.screen, self.pal[col], self._hex_points(c, RB))
        if locked:
            pg.draw.polygon(self.screen, DARK, self._hex_points(c, RB), 2)
        if self.show_glyphs:
            self._draw_glyph(self.glyphs[col], c, DARK)

    def _draw_equations(self, y: int) -> int:
        pg = self.pg
        cents, rows = self._block_layout(y)
        self.block_centres = cents
        w = SQRT3 * RB * GAP
        for i, ry, h in rows:
            r = self.sim.reactions.reactions[i]
            lv = float(self.lamps[i])
            cy = ry + h // 2
            self._draw_meter(self.panel_x + 2, cy, float(self.needles[i]))
            # arrow: thin line + open chevrons; 1-3 chevrons for slow / medium / fastest k
            ax = self.panel_x + EQ_DX + 2 * RB + w + 8
            kmax = max(q.rate for q in self.sim.reactions.reactions) or 1.0
            frac = r.rate / kmax
            n_chev = 3 if frac > 0.75 else (2 if frac > 0.4 else 1)
            ac = (200, 200, 210)
            tip = ax + 42
            pg.draw.line(self.screen, ac, (ax, cy), (tip - 1, cy), 2)
            for k in range(n_chev):
                tx = tip - 6 * k
                pg.draw.aalines(
                    self.screen, ac, False, [(tx - 7, cy - 6), (tx, cy), (tx - 7, cy + 6)]
                )
                pg.draw.aalines(
                    self.screen, ac, False, [(tx - 8, cy - 6), (tx - 1, cy), (tx - 8, cy + 6)]
                )
            for b_, c in cents.items():
                if b_[0] == i:
                    self._draw_block(b_, c)
            info = f"k={r.rate:g}  {lv:.3f}"
            if r.shape == "vertex":
                info += "  cw only" if r.chiral else "  either hand"
            self._text(info, (self.panel_x + EQ_DX + 4 * w + 90, cy - 7), self.small, GREY)
        return rows[-1][1] + rows[-1][2] + 14 if rows else y

    def _draw_meter(self, x: float, cy: float, f: float) -> None:
        """Small VU-style dial at (x, cy - 14), 52 x 30 px: log firing rate, one tick per
        decade, needle at fraction f of the scale."""
        pg = self.pg
        lc = self.cfg["lamps"]
        r = pg.Rect(x, cy - 15, 52, 30)
        pg.draw.rect(self.screen, METER_FACE, r, border_radius=4)
        px, py = r.centerx, r.bottom - 3
        rad = 22.0
        a0, a1 = math.radians(150), math.radians(30)  # left and right ends of the scale
        decades = round(math.log10(lc["hi"] / lc["lo"]))
        for k in range(decades + 1):
            a = a0 + (a1 - a0) * k / decades
            ca, sa = math.cos(a), math.sin(a)
            pg.draw.line(
                self.screen,
                METER_INK,
                (px + 0.80 * rad * ca, py - 0.80 * rad * sa),
                (px + rad * ca, py - rad * sa),
                1,
            )
        pts = [
            (
                px + rad * math.cos(a0 + (a1 - a0) * t / 12),
                py - rad * math.sin(a0 + (a1 - a0) * t / 12),
            )
            for t in range(13)
        ]
        pg.draw.aalines(self.screen, METER_INK, False, pts)
        a = a0 + (a1 - a0) * f
        tip = (px + (rad - 1) * math.cos(a), py - (rad - 1) * math.sin(a))
        pg.draw.line(self.screen, NEEDLE, (px, py), tip, 2)
        pg.draw.circle(self.screen, METER_INK, (px, py), 3)

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
        yy = y + h + 10
        for k, goal in enumerate(self.level.goals):
            pr = self.tracker.progress(self.sim.t)[k]
            yy += self._draw_segments(self._goal_segments(goal, f"[{pr:4.0%}]  goal:"), x, yy)
        for f in self.level.fail:
            yy += self._draw_segments(self._goal_segments(f, "fail if:", fail=True), x, yy, GREY)
        help_ = "click 2 blocks to swap | wheel zoom, drag pan, 0 reset | spc 1-4 R S"
        self._text(help_, (x, self.screen.get_height() - 22), self.small, GREY)

    def _goal_segments(self, g: dict, prefix: str, fail: bool = False) -> list:
        """Goal as text pieces and colour swatches (ints), e.g. ["goal: make", 2, "coral",
        "extinct"]."""
        nm = self.cnames
        if g["type"] == "extinct":
            c = g["color"]
            return [prefix, *([] if fail else ["make"]), c, nm[c], "extinct"]
        if g["type"] == "hold":
            seg: list = [prefix, "hold"]
            for c, (lo, hi) in enumerate(g["bands"]):
                if lo > 0 or hi < 1:
                    seg += [c, f"{lo:g}-{hi:g}"]
            return seg + [f"for t = {g['duration']:g}"]
        if g["type"] == "reach":
            seg = [prefix, "reach"]
            for c, v in enumerate(g["target"]):
                seg += [c, f"{v:g}"]
            return seg + [f"(+/- {g['tol']:g}) for t = {g['duration']:g}"]
        return [prefix, json.dumps(g)]

    def _draw_segments(self, segs: list, x: float, y: float, col=WHITE) -> int:
        """Draw text pieces separated by spaces; an int is a hex swatch of that colour."""
        for s_ in segs:
            if isinstance(s_, int):
                c = (x + 8, y + 7)
                pg = self.pg
                pg.draw.polygon(self.screen, GREY, self._hex_points(c, 9))
                pg.draw.polygon(self.screen, self.pal[s_], self._hex_points(c, 8))
                x += 20
            else:
                surf = self.font.render(s_, True, col)
                self.screen.blit(surf, (x, y))
                x += surf.get_width() + 6
        return 22

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
        if self.menu_open:
            for key, r in self.buttons.items():
                if key.startswith("pal:") and r.collidepoint(pos):
                    self.set_palette(key[4:])
                    break
            self.menu_open = False
            return
        for key, r in self.buttons.items():
            if not key.startswith("pal:") and r.collidepoint(pos):
                self.button(key)
                return
        if self.grid_rect.collidepoint(pos):
            self.dragging = True
            return
        hit = None
        for b, c in getattr(self, "block_centres", {}).items():
            if (pos[0] - c[0]) ** 2 + (pos[1] - c[1]) ** 2 <= (RB * SQRT3 / 2) ** 2:
                hit = b
                break
        if hit is None:
            self.selected = None
            return
        if self.paused:
            self._notify("paused: press Play to switch blocks")
            return
        if hit in self.sim.reactions.locked:
            self._notify("this block is locked")
            return
        if self.selected is None:
            self.selected = hit
        elif hit == self.selected:
            self.selected = None
        else:
            self.swap(self.selected, hit)
            self.selected = None

    def set_palette(self, name: str) -> None:
        self.palette_name = name
        self.pal = palette_rgb(self.cfg, self.level.palette, name)
        self.cnames = colour_names(self.cfg, self.level.palette, name)

    def _notify(self, msg: str) -> None:
        self.notice = msg
        self.notice_until = time.time() + 2.0

    def button(self, key: str) -> None:
        if key == "pause":
            self.paused = not self.paused
            if self.paused:
                self.selected = None
        elif key.startswith("speed"):
            self.speed_i = int(key[5:])
        elif key == "restart":
            self.reset()
        elif key == "palette":
            self.menu_open = not self.menu_open
        elif key == "glyphs":
            self.show_glyphs = not self.show_glyphs

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
                elif ev.type == pg.MOUSEBUTTONUP and ev.button == 1:
                    self.dragging = False
                elif ev.type == pg.MOUSEMOTION and self.dragging:
                    self.pan(*ev.rel)
                elif ev.type == pg.MOUSEWHEEL:
                    mp = pg.mouse.get_pos()
                    if self.grid_rect.collidepoint(mp) and ev.y:
                        self.zoom(self.cfg["grid"]["zoom_step"] ** (1 if ev.y > 0 else -1), mp)
                elif ev.type == pg.KEYDOWN:
                    if ev.key == pg.K_ESCAPE:
                        running = False
                    elif ev.key == pg.K_SPACE:
                        self.button("pause")
                    elif pg.K_1 <= ev.key <= pg.K_9:
                        if ev.key - pg.K_1 < len(self.cfg["speed"]["multipliers"]):
                            self.button(f"speed{ev.key - pg.K_1}")
                    elif ev.key in (pg.K_EQUALS, pg.K_PLUS, pg.K_KP_PLUS):
                        self.zoom(self.cfg["grid"]["zoom_step"], pg.mouse.get_pos())
                    elif ev.key in (pg.K_MINUS, pg.K_KP_MINUS):
                        self.zoom(1 / self.cfg["grid"]["zoom_step"], pg.mouse.get_pos())
                    elif ev.key in (pg.K_0, pg.K_KP0):
                        self.zoom_reset()
                    elif ev.key == pg.K_r:
                        self.button("restart")
                    elif ev.key == pg.K_s:
                        self.save(screenshot=True)
            if smoke_frames:
                dt = 1.0 / self.cfg["window"]["fps"]
                if frame == 1:
                    self.speed_i = len(self.cfg["speed"]["multipliers"]) - 1
                    self.zoom(4.0, (300, 300))
                    self.pan(37, -12)
                if frame == smoke_frames // 2 - 2:
                    # paused: a block click must not select or swap anything
                    before = (
                        self.sim.reactions.multiset(),
                        [(r.inp[:], r.out[:]) for r in self.sim.reactions.reactions],
                    )
                    self.click(self.buttons["pause"].center)
                    self.draw()
                    blk = next(b for b in self.block_centres if b not in self.sim.reactions.locked)
                    self.click(self.block_centres[blk])
                    assert self.selected is None and self.paused
                    after = (
                        self.sim.reactions.multiset(),
                        [(r.inp[:], r.out[:]) for r in self.sim.reactions.reactions],
                    )
                    assert before == after
                    self.click(self.buttons["pause"].center)
                    assert not self.paused
                    print("smoke: pause blocks switching OK", flush=True)
                if frame == smoke_frames // 2:
                    free = [
                        b for b in self.sim.reactions.blocks() if b not in self.sim.reactions.locked
                    ]
                    self.click(self.block_centres[free[0]])
                    self.draw()
                    self.click(self.block_centres[free[-1]])
                if frame >= smoke_frames:
                    self.click(self.block_centres[free[0]])  # leave one block selected
                    self.draw()
                    self.save(screenshot=True)
                    running = False
            self.step(dt)
            self.move_needles(dt)
            self.draw()
            pg.display.flip()
            frame += 1
        if any(e["event"] == "swap" for e in self.session) or self.run_dir:
            print(f"session saved to {self.save()}", flush=True)


def _mult_label(m: float) -> str:
    return f"x1/{round(1 / m)}" if m < 1 else f"x{m:g}"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("level", type=Path)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--shuffled", action="store_true", help="well-mixed control")
    ap.add_argument("--n", type=int, default=None, help="override the grid size")
    ap.add_argument("--palette", default=None, help="palette name from toy/viewer.toml")
    ap.add_argument("--world", default=None, help="torus:N | open:N | hextorus:S | hexagon:S")
    ap.add_argument("--smoke", type=int, default=0, help="headless: run N frames, save PNG")
    a = ap.parse_args()
    if a.smoke:
        os.environ["SDL_VIDEODRIVER"] = "dummy"
    import pygame

    pygame.init()
    level = load_level(a.level)
    if a.n:
        level.grid = {"type": "hex", "n": a.n}  # rectangular torus N x N
    if a.world:
        kind, size = a.world.split(":")
        level.grid = {
            "torus": {"type": "hex", "n": int(size)},
            "open": {"type": "hex", "n": int(size), "periodic": False},
            "hextorus": {"type": "hexagon", "side": int(size)},
            "hexagon": {"type": "hexagon", "side": int(size), "periodic": False},
        }[kind]
    cfg = load_cfg()
    if a.palette:
        if a.palette not in cfg["palettes"]:
            ap.error(f"unknown palette {a.palette!r}; choose from {list(cfg['palettes'])}")
        cfg["palette"]["default"] = a.palette
    v = Viewer(level, cfg, a.seed, a.shuffled)
    v.loop(a.smoke)
    pygame.quit()


if __name__ == "__main__":
    main()
