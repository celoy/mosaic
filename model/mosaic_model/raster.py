"""Pixel -> cell map for drawing a hex torus as an image (numpy only, no pygame).

render(colours) is then a single fancy-indexing operation: rgb = palette_ext[colours_ext[map]].
"""

from __future__ import annotations

import numpy as np

from .graph import SQRT3, Graph


def hex_pixel_map(g: Graph, size: float, border: float = 0.12) -> np.ndarray:
    """(H, W) int array of cell indices, -1 on borders, for hexes of circumradius `size` px.

    Cell (row 0, col 0) is centred at (size * sqrt3 / 2, size) so the block starts at (0, 0).
    `border` is the fraction of the inradius drawn as a dark gap.
    """
    if g.kind != "hex":
        raise ValueError("hex_pixel_map needs a hex grid")
    n = g.n
    w = int(np.ceil(SQRT3 * size * (n + 0.5)))
    h = int(np.ceil(size * (1.5 * n + 0.5)))
    ys, xs = np.mgrid[0:h, 0:w].astype(float)
    x = (xs + 0.5 - size * SQRT3 / 2) / size
    y = (ys + 0.5 - size) / size
    # fractional axial coordinates (pointy-top), then cube rounding
    q = SQRT3 / 3 * x - y / 3
    r = 2.0 / 3.0 * y
    cx, cz = q, r
    cy = -cx - cz
    rx, ry, rz = np.round(cx), np.round(cy), np.round(cz)
    dx, dy, dz = np.abs(rx - cx), np.abs(ry - cy), np.abs(rz - cz)
    fix_x = (dx > dy) & (dx > dz)
    fix_y = ~fix_x & (dy > dz)
    rx = np.where(fix_x, -ry - rz, rx)
    ry = np.where(fix_y, -rx - rz, ry)
    rz = np.where(~fix_x & ~fix_y, -rx - ry, rz)
    row = rz.astype(int)
    col = (rx + (row - (row & 1)) // 2).astype(int)
    # distance to the cell centre in the hex (inradius) metric
    px = SQRT3 * (col + 0.5 * (row & 1))
    py = 1.5 * row
    ddx, ddy = x - px, y - py
    inr = SQRT3 / 2
    d = np.maximum.reduce(
        [np.abs(ddx), np.abs(0.5 * ddx + SQRT3 / 2 * ddy), np.abs(0.5 * ddx - SQRT3 / 2 * ddy)]
    )
    idx = (row % n) * n + (col % n)
    return np.where(d <= inr * (1 - border), idx, -1)


def render_rgb(
    pixel_map: np.ndarray, colours: np.ndarray, palette_rgb: np.ndarray, bg=(30, 30, 36)
) -> np.ndarray:
    """(H, W, 3) uint8 image from cell colours."""
    lut = np.vstack([np.asarray(palette_rgb, dtype=np.uint8), np.asarray(bg, dtype=np.uint8)])
    c = np.append(np.asarray(colours, dtype=np.int64), len(palette_rgb))  # index -1 -> bg
    return lut[c[pixel_map]]
