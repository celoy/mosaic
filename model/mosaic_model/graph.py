"""Cell graphs: cells, edges (pairs of neighbouring cells) and vertices (triples of cells
meeting at a point, listed clockwise as seen on screen with the y axis pointing down).

The hexagonal grid matches the Godot benchmark grid (game/sim/hex_grid.gd): n x n pointy-top
hexes in "odd-r" offset rows (odd rows shifted right by half a cell), periodic in both
directions. Cell index i = r * n + c. Each cell owns three edges (E, SW, SE) and two
vertices (its lower-right and bottom corners), so E = 3 n^2 and V = 2 n^2.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

SQRT3 = 3.0**0.5


@dataclass(frozen=True)
class Graph:
    """Cell graph with full adjacency.

    Attributes:
        n_cells: number of cells.
        edges: (E, 2) int array of neighbouring cell pairs.
        vertices: (V, 3) int array of cell triples, clockwise on screen (y down).
        centers: (n_cells, 2) cell centres for a hex of unit circumradius (y down).
        period: (width, height) of the periodic box, or None if not periodic.
    """

    n_cells: int
    edges: np.ndarray
    vertices: np.ndarray
    centers: np.ndarray
    period: tuple[float, float] | None
    kind: str = "hex"
    n: int = 0

    def neighbours(self) -> list[list[int]]:
        nb: list[list[int]] = [[] for _ in range(self.n_cells)]
        for a, b in self.edges.tolist():
            nb[a].append(b)
            nb[b].append(a)
        return nb

    def displacement(self, a: int, b: int) -> np.ndarray:
        """Vector from centre a to centre b (minimal image if periodic)."""
        d = self.centers[b] - self.centers[a]
        if self.period is not None:
            w, h = self.period
            d = d - np.array([w, h]) * np.round(d / np.array([w, h]))
        return d


def hex_torus(n: int) -> Graph:
    """n x n periodic hexagonal grid (n even, n >= 4)."""
    if n < 4 or n % 2:
        raise ValueError("hex_torus: n must be even and >= 4")
    edges = np.empty((3 * n * n, 2), dtype=np.int64)
    verts = np.empty((2 * n * n, 3), dtype=np.int64)
    centers = np.empty((n * n, 2))
    for r in range(n):
        r2 = (r + 1) % n
        for c in range(n):
            i = r * n + c
            east = r * n + (c + 1) % n
            if r % 2 == 0:
                sw, se = r2 * n + (c - 1) % n, r2 * n + c
            else:
                sw, se = r2 * n + c, r2 * n + (c + 1) % n
            edges[3 * i] = (i, east)
            edges[3 * i + 1] = (i, sw)
            edges[3 * i + 2] = (i, se)
            # Lower-right corner: i (upper left), east (upper right), se (below): clockwise.
            verts[2 * i] = (i, east, se)
            # Bottom corner: i (above), se (lower right), sw (lower left): clockwise.
            verts[2 * i + 1] = (i, se, sw)
            centers[i] = (SQRT3 * (c + 0.5 * (r & 1)), 1.5 * r)
    return Graph(
        n_cells=n * n,
        edges=edges,
        vertices=verts,
        centers=centers,
        period=(SQRT3 * n, 1.5 * n),
        kind="hex",
        n=n,
    )


def make_graph(spec: dict) -> Graph:
    """Build a graph from a level's "grid" entry."""
    if spec.get("type") == "hex":
        return hex_torus(int(spec["n"]))
    raise ValueError(f"unsupported grid type: {spec.get('type')!r}")
