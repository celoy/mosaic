"""Palettes for any number of colours (shared by the viewer and the science snapshots)."""

from __future__ import annotations

import colorsys

import numpy as np


def _hex(h: str) -> tuple[int, int, int]:
    h = h.lstrip("#")
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


def names(cfg: dict) -> list[str]:
    return list(cfg["palettes"])


UI_SPLIT = 5  # listed palettes with more colours give their darkest and lightest to the UI


def luminance(rgb) -> float:
    """WCAG relative luminance of an sRGB colour (0-255)."""
    lin = [
        (v / 255) / 12.92 if v / 255 <= 0.04045 else ((v / 255 + 0.055) / 1.055) ** 2.4 for v in rgb
    ]
    return 0.2126 * lin[0] + 0.7152 * lin[1] + 0.0722 * lin[2]


def _split(cfg: dict, name: str) -> tuple[list[int], tuple | None]:
    """Indices of the cell colours in the palette list, and (bg, fg) taken from it if the
    palette is listed in [palette] split_ui and has more than UI_SPLIT colours (darkest ->
    background, lightest -> text)."""
    cols = [_hex(c) for c in cfg["palettes"][name]]
    idx = list(range(len(cols)))
    if len(cols) <= UI_SPLIT or name not in cfg["palette"].get("split_ui", []):
        return idx, None
    order = sorted(idx, key=lambda k: luminance(cols[k]))
    dark, light = order[0], order[-1]
    return [k for k in idx if k not in (dark, light)], (cols[dark], cols[light])


def ui_colours(cfg: dict, name: str | None = None) -> dict[str, tuple[int, int, int]]:
    """Interface colours: bg, fg (text) and tones mixed between them (mid for secondary
    text, btn / btn_on for buttons, chart / line for the chart)."""
    name = name or cfg["palette"]["default"]
    split = None if name == "greyscale" else _split(cfg, name)[1]
    bg, fg = split or (_hex(cfg["palette"]["background"]), (235, 235, 240))

    def mix(f: float) -> tuple[int, int, int]:
        return tuple(int(round(b + f * (t - b))) for b, t in zip(bg, fg, strict=True))  # type: ignore[return-value]

    return {
        "bg": bg,
        "fg": fg,
        "mid": mix(0.55),
        "btn": mix(0.13),
        "btn_on": mix(0.30),
        "chart": mix(0.07),
        "line": mix(0.18),
    }


def palette_rgb(cfg: dict, n: int, name: str | None = None) -> np.ndarray:
    """(n, 3) uint8 cell colours of the named palette (default from the config). Greyscale is
    evenly spaced in lightness; other palettes are extended with pastel hues if too short."""
    name = name or cfg["palette"]["default"]
    if name == "greyscale":
        levels = np.linspace(0.30, 0.95, n) if n > 1 else np.array([0.7])
        return np.array([[int(255 * v)] * 3 for v in levels], dtype=np.uint8)
    allc = [_hex(c) for c in cfg["palettes"][name]]
    out = [allc[k] for k in _split(cfg, name)[0]][:n]
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
    given_all = list(cfg.get("palette_names", {}).get(name, []))
    keep = _split(cfg, name)[0]
    given = [given_all[k] for k in keep if k < len(given_all)][:n]
    rgb = palette_rgb(cfg, n, name)
    out = given + [_auto_name(rgb[k]) for k in range(len(given), n)]
    seen: dict[str, int] = {}
    for k, s in enumerate(out):
        seen[s] = seen.get(s, 0) + 1
        if seen[s] > 1:
            out[k] = f"{s} {seen[s]}"
    return out
