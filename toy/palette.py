"""Palettes for any number of colours (shared by the viewer and the science snapshots)."""

from __future__ import annotations

import colorsys

import numpy as np


def _hex(h: str) -> tuple[int, int, int]:
    h = h.lstrip("#")
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


def names(cfg: dict) -> list[str]:
    return list(cfg["palettes"])


def palette_rgb(cfg: dict, n: int, name: str | None = None) -> np.ndarray:
    """(n, 3) uint8 colours of the named palette (default from the config). Greyscale is
    evenly spaced in lightness; other palettes are extended with pastel hues if too short."""
    name = name or cfg["palette"]["default"]
    if name == "greyscale":
        levels = np.linspace(0.30, 0.95, n) if n > 1 else np.array([0.7])
        return np.array([[int(255 * v)] * 3 for v in levels], dtype=np.uint8)
    out = [_hex(c) for c in cfg["palettes"][name]][:n]
    extra = n - len(out)
    for k in range(extra):
        r, g, b = colorsys.hsv_to_rgb((k + 0.5) / extra, 0.35, 0.9)
        out.append((int(255 * r), int(255 * g), int(255 * b)))
    return np.array(out, dtype=np.uint8)


def glyphs(cfg: dict, n: int) -> list[str]:
    """One glyph name per colour; colours beyond the list get their index as text."""
    g = list(cfg["palette"]["glyphs"])[:n]
    return g + [str(i) for i in range(len(g), n)]
