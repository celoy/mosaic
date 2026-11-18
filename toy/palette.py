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


_HUES = [
    (15, "red"),
    (40, "orange"),
    (65, "yellow"),
    (90, "lime"),
    (150, "green"),
    (175, "teal"),
    (200, "cyan"),
    (250, "blue"),
    (290, "violet"),
    (330, "magenta"),
    (345, "pink"),
    (360, "red"),
]


def _auto_name(rgb) -> str:
    """Plain colour name from hue, saturation and lightness."""
    r, g, b = (v / 255 for v in rgb)
    h, lum, s = colorsys.rgb_to_hls(r, g, b)
    if s < 0.15 or (lum > 0.93) or lum < 0.07:
        return "white" if lum > 0.85 else "black" if lum < 0.15 else "grey"
    hue = next(name for edge, name in _HUES if h * 360 <= edge)
    if lum < 0.3:
        return f"dark {hue}"
    if lum > 0.75:
        return f"pale {hue}"
    return hue


def colour_names(cfg: dict, n: int, name: str | None = None) -> list[str]:
    """One short, unique name per colour: [palette_names] from the config if given,
    otherwise generated from the colour."""
    name = name or cfg["palette"]["default"]
    if name == "greyscale":
        return [f"grey {k + 1}" for k in range(n)]  # 1 = darkest
    given = list(cfg.get("palette_names", {}).get(name, []))[:n]
    rgb = palette_rgb(cfg, n, name)
    out = given + [_auto_name(rgb[k]) for k in range(len(given), n)]
    seen: dict[str, int] = {}
    for k, s in enumerate(out):
        seen[s] = seen.get(s, 0) + 1
        if seen[s] > 1:
            out[k] = f"{s} {seen[s]}"
    return out
