"""Pixel -> cell map for drawing any hex world as an image (numpy only, no pygame).

render(colours) is then a single fancy-indexing operation: rgb = palette_ext[colours_ext[map]].
Each cell is drawn once, at its axial position; pixels outside the world are background.
"""

from __future__ import annotations

import numpy as np

from .graph import SQRT3, Graph


def world_pixel_size(g: Graph, size: float) -> tuple[int, int]:
    """(width, height) in pixels of the drawn world for hexes of circumradius `size`."""
    x, y = g.centers[:, 0], g.centers[:, 1]
    w = (x.max() - x.min() + SQRT3) * size
    h = (y.max() - y.min() + 2.0) * size
    return int(np.ceil(w)), int(np.ceil(h))


def fit_size(g: Graph, width: float, height: float) -> float:
    """Largest hex circumradius such that the world fits in width x height pixels."""
    w1, h1 = world_pixel_size(g, 1.0)
    return min(width / w1, height / h1)


def hex_pixel_map(g: Graph, size: float, border: float = 0.0) -> np.ndarray:
    """(H, W) int array of cell indices, -1 on borders and outside the world, for hexes of
    circumradius `size` px. `border` is the fraction of the inradius drawn as a dark gap."""
    w, h = world_pixel_size(g, size)
    x0 = g.centers[:, 0].min() - SQRT3 / 2
    y0 = g.centers[:, 1].min() - 1.0
    ys, xs = np.mgrid[0:h, 0:w].astype(float)
    x = (xs + 0.5) / size + x0
    y = (ys + 0.5) / size + y0
    # fractional axial coordinates (pointy-top), then cube rounding
    cx = SQRT3 / 3 * x - y / 3
    cz = 2.0 / 3.0 * y
    cy = -cx - cz
    rx, ry, rz = np.round(cx), np.round(cy), np.round(cz)
    dx, dy, dz = np.abs(rx - cx), np.abs(ry - cy), np.abs(rz - cz)
    fix_x = (dx > dy) & (dx > dz)
    fix_y = ~fix_x & (dy > dz)
    rx = np.where(fix_x, -ry - rz, rx)
    rz = np.where(~fix_x & ~fix_y, -rx - ry, rz)
    q = rx.astype(np.int64)
    r = rz.astype(np.int64)
    # lookup table on the axial bounding box
    qa, ra = g.axial[:, 0], g.axial[:, 1]
    qmin, rmin = qa.min(), ra.min()
    lut = -np.ones((qa.max() - qmin + 1, ra.max() - rmin + 1), dtype=np.int64)
    lut[qa - qmin, ra - rmin] = np.arange(g.n_cells)
    inside = (q >= qmin) & (q < qmin + lut.shape[0]) & (r >= rmin) & (r < rmin + lut.shape[1])
    idx = np.full(q.shape, -1, dtype=np.int64)
    idx[inside] = lut[q[inside] - qmin, r[inside] - rmin]
    if border > 0:
        px = SQRT3 * (q + 0.5 * r)
        py = 1.5 * r
        ddx, ddy = x - px, y - py
        d = np.maximum.reduce(
            [np.abs(ddx), np.abs(0.5 * ddx + SQRT3 / 2 * ddy), np.abs(0.5 * ddx - SQRT3 / 2 * ddy)]
        )
        idx = np.where(d <= SQRT3 / 2 * (1 - border), idx, -1)
    return idx


def view_pixel_map(
    g: Graph, size: float, width: int, height: int, border: float = 0.0
) -> np.ndarray:
    """(height, width) cell-index map of a viewport with the world centred in it.

    Periodic worlds are tiled to the edges of the viewport (each pixel shows the periodic
    image of its cell); for open worlds pixels outside the world are -1 (background)."""
    cxw = 0.5 * (g.centers[:, 0].min() + g.centers[:, 0].max())
    cyw = 0.5 * (g.centers[:, 1].min() + g.centers[:, 1].max())
    ys, xs = np.mgrid[0:height, 0:width].astype(float)
    x = (xs + 0.5 - width / 2) / size + cxw
    y = (ys + 0.5 - height / 2) / size + cyw
    cx = SQRT3 / 3 * x - y / 3
    cz = 2.0 / 3.0 * y
    cy = -cx - cz
    rx, ry, rz = np.round(cx), np.round(cy), np.round(cz)
    dx, dy, dz = np.abs(rx - cx), np.abs(ry - cy), np.abs(rz - cz)
    fix_x = (dx > dy) & (dx > dz)
    fix_y = ~fix_x & (dy > dz)
    rx = np.where(fix_x, -ry - rz, rx)
    rz = np.where(~fix_x & ~fix_y, -rx - ry, rz)
    q = rx.astype(np.int64)
    r = rz.astype(np.int64)
    qa, ra = g.axial[:, 0], g.axial[:, 1]
    qmin, rmin = qa.min(), ra.min()
    lut = -np.ones((qa.max() - qmin + 1, ra.max() - rmin + 1), dtype=np.int64)
    lut[qa - qmin, ra - rmin] = np.arange(g.n_cells)

    def look(qq: np.ndarray, rr: np.ndarray) -> np.ndarray:
        ok = (qq >= qmin) & (qq < qmin + lut.shape[0]) & (rr >= rmin) & (rr < rmin + lut.shape[1])
        out = np.full(qq.shape, -1, dtype=np.int64)
        out[ok] = lut[qq[ok] - qmin, rr[ok] - rmin]
        return out

    idx = look(q, r)
    if g.periodic and len(g.shifts_axial) == 2:
        s1, s2 = g.shifts_axial.astype(float)
        m = np.array([[s1[0], s2[0]], [s1[1], s2[1]]])
        q0, r0 = qa.mean(), ra.mean()
        k = np.linalg.solve(m, np.stack([(q - q0).ravel(), (r - r0).ravel()]))
        k1 = np.round(k[0]).reshape(q.shape).astype(np.int64)
        k2 = np.round(k[1]).reshape(q.shape).astype(np.int64)
        a1, a2 = g.shifts_axial
        for i in range(-2, 3):
            for j in range(-2, 3):
                todo = idx < 0
                if not todo.any():
                    break
                kk1, kk2 = k1[todo] + i, k2[todo] + j
                idx[todo] = look(
                    q[todo] - kk1 * a1[0] - kk2 * a2[0], r[todo] - kk1 * a1[1] - kk2 * a2[1]
                )
    if border > 0:
        px = SQRT3 * (q + 0.5 * r)
        py = 1.5 * r
        ddx, ddy = x - px, y - py
        d = np.maximum.reduce(
            [np.abs(ddx), np.abs(0.5 * ddx + SQRT3 / 2 * ddy), np.abs(0.5 * ddx - SQRT3 / 2 * ddy)]
        )
        idx = np.where(d <= SQRT3 / 2 * (1 - border), idx, -1)
    return idx


def render_rgb(
    pixel_map: np.ndarray, colours: np.ndarray, palette_rgb: np.ndarray, bg=(30, 30, 36)
) -> np.ndarray:
    """(H, W, 3) uint8 image from cell colours."""
    lut = np.vstack([np.asarray(palette_rgb, dtype=np.uint8), np.asarray(bg, dtype=np.uint8)])
    c = np.append(np.asarray(colours, dtype=np.int64), len(palette_rgb))  # index -1 -> bg
    return lut[c[pixel_map]]
