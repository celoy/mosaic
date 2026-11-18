"""Palette for any number of colours (shared by the viewer and the science snapshots)."""

from __future__ import annotations

import colorsys

import numpy as np


def _hex(h: str) -> tuple[int, int, int]:
    h = h.lstrip("#")
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


def palette_rgb(cfg: dict, n: int) -> np.ndarray:
    """(n, 3) uint8 colours: the configured list first, then evenly spaced extra hues."""
    base = [_hex(c) for c in cfg["palette"]["colours"]]
    out = base[:n]
    extra = n - len(out)
    for k in range(extra):
        r, g, b = colorsys.hsv_to_rgb((k + 0.5) / extra, 0.55, 0.9)
        out.append((int(255 * r), int(255 * g), int(255 * b)))
    return np.array(out, dtype=np.uint8)


def glyphs(cfg: dict, n: int) -> list[str]:
    """One glyph name per colour; colours beyond the list get their index as text."""
    g = list(cfg["palette"]["glyphs"])[:n]
    return g + [str(i) for i in range(len(g), n)]
