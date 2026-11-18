"""Reactions, colour blocks, swaps and lookup tables.

Conventions (see also meanfield.py):

* A reaction has a shape (cell, edge, vertex), input and output colour tuples and a rate k.
* An element (cell, edge, vertex) stores its cells in a fixed order; for vertices the order
  is clockwise. An *orientation* maps reaction position p to element position perm[p].
  Edge reactions use both orientations; vertex reactions use the three cyclic rotations,
  plus the three mirrored ones if the reaction is achiral (``chiral = False``).
* Every (reaction, orientation) that matches an element's colours is a channel firing at
  rate k. So A|A -> A|B fires at 2k on an A|A edge, and an achiral vertex reaction behaves as
  the reaction plus its mirror image, each at rate k.
* Blocks: every colour in every equation is a block addressed as (reaction, side, position),
  side in {"in", "out"}. The player swaps two unlocked blocks; the multiset is conserved.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field

ARITY = {"cell": 1, "edge": 2, "vertex": 3}
SHAPES = ("cell", "edge", "vertex")

Block = tuple[int, str, int]


@dataclass
class Reaction:
    shape: str
    inp: list[int]
    out: list[int]
    rate: float
    chiral: bool = True

    def __post_init__(self) -> None:
        m = ARITY[self.shape]
        if len(self.inp) != m or len(self.out) != m:
            raise ValueError(f"{self.shape} reaction needs {m} colours per side")
        if self.rate < 0:
            raise ValueError("negative rate")

    def label(self) -> str:
        sep = {"cell": "", "edge": "|", "vertex": ","}[self.shape]
        s = f"{sep.join(map(str, self.inp))} -> {sep.join(map(str, self.out))}"
        if self.shape == "vertex":
            s = f"({s.replace(' -> ', ') -> (')})" + ("" if self.chiral else " achiral")
        return f"{s}  k={self.rate:g}"


def orientations(shape: str, chiral: bool = True) -> list[tuple[int, ...]]:
    """Element positions hit by reaction positions 0..m-1, one tuple per orientation."""
    if shape == "cell":
        return [(0,)]
    if shape == "edge":
        return [(0, 1), (1, 0)]
    rot = [(0, 1, 2), (1, 2, 0), (2, 0, 1)]
    if chiral:
        return rot
    return rot + [(0, 2, 1), (2, 1, 0), (1, 0, 2)]


@dataclass
class Entry:
    """Channels matching one colour tuple of one shape.

    total: sum of channel rates; cum: cumulative rates; new: new colour tuples (element order);
    rid: reaction index of each channel.
    """

    total: float
    cum: list[float]
    new: list[tuple[int, ...]]
    rid: list[int]


@dataclass
class ReactionSet:
    reactions: list[Reaction]
    palette: int
    locked: set[Block] = field(default_factory=set)

    def __post_init__(self) -> None:
        self.locked = {tuple(b) for b in self.locked}  # type: ignore[misc]
        for r in self.reactions:
            for c in r.inp + r.out:
                if not 0 <= c < self.palette:
                    raise ValueError(f"colour {c} outside palette {self.palette}")
        for b in self.locked:
            self.get(b)  # validates the address

    # ---- blocks -------------------------------------------------------------------------
    def blocks(self) -> list[Block]:
        out: list[Block] = []
        for i, r in enumerate(self.reactions):
            for side in ("in", "out"):
                out += [(i, side, p) for p in range(ARITY[r.shape])]
        return out

    def get(self, b: Block) -> int:
        r = self.reactions[b[0]]
        return (r.inp if b[1] == "in" else r.out)[b[2]]

    def _set(self, b: Block, c: int) -> None:
        r = self.reactions[b[0]]
        (r.inp if b[1] == "in" else r.out)[b[2]] = c

    def swap(self, a: Block, b: Block) -> None:
        """Swap the colours of two unlocked blocks."""
        a, b = tuple(a), tuple(b)  # type: ignore[assignment]
        if a in self.locked or b in self.locked:
            raise ValueError(f"locked block in swap {a} <-> {b}")
        ca, cb = self.get(a), self.get(b)
        self._set(a, cb)
        self._set(b, ca)

    def multiset(self) -> list[int]:
        return sorted(self.get(b) for b in self.blocks())

    def shapes_used(self) -> list[str]:
        return [s for s in SHAPES if any(r.shape == s for r in self.reactions)]

    # ---- lookup tables ------------------------------------------------------------------
    def tables(self) -> dict[str, list[Entry | None]]:
        """For each shape used, a list indexed by colour code (c0*P + c1)*P + c2 ..."""
        p = self.palette
        out: dict[str, list[Entry | None]] = {}
        for shape in self.shapes_used():
            m = ARITY[shape]
            tab: list[Entry | None] = []
            for cols in itertools.product(range(p), repeat=m):
                tab.append(self._entry(shape, cols))
            out[shape] = tab
        return out

    def _entry(self, shape: str, cols: tuple[int, ...]) -> Entry | None:
        cum: list[float] = []
        new: list[tuple[int, ...]] = []
        rid: list[int] = []
        tot = 0.0
        for i, r in enumerate(self.reactions):
            if r.shape != shape or r.rate == 0:
                continue
            for perm in orientations(shape, r.chiral):
                if all(cols[perm[q]] == r.inp[q] for q in range(len(perm))):
                    nc = list(cols)
                    for q in range(len(perm)):
                        nc[perm[q]] = r.out[q]
                    tot += r.rate
                    cum.append(tot)
                    new.append(tuple(nc))
                    rid.append(i)
        return Entry(tot, cum, new, rid) if rid else None

    def max_propensity(self) -> float:
        """K: the largest total channel rate of any element in any colour state."""
        k = 0.0
        for tab in self.tables().values():
            for e in tab:
                if e is not None:
                    k = max(k, e.total)
        return k

    def copy(self) -> ReactionSet:
        return ReactionSet(
            [Reaction(r.shape, list(r.inp), list(r.out), r.rate, r.chiral) for r in self.reactions],
            self.palette,
            set(self.locked),
        )


def code(cols: tuple[int, ...] | list[int], palette: int) -> int:
    c = 0
    for x in cols:
        c = c * palette + x
    return c


def reactions_from_json(items: list[dict]) -> list[Reaction]:
    return [
        Reaction(
            shape=d["shape"],
            inp=list(d["in"]),
            out=list(d["out"]),
            rate=float(d["rate"]),
            chiral=bool(d.get("chiral", True)),
        )
        for d in items
    ]
