"""Palettes for any number of colours (shared by the viewer and the science snapshots)."""

from __future__ import annotations

import colorsys
import functools
import itertools

import numpy as np


def _hex(h: str) -> tuple[int, int, int]:
    h = h.lstrip("#")
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


def names(cfg: dict) -> list[str]:
    return list(cfg["palettes"])


# ---- colour difference (CIE76 in Lab) and colour-vision-deficiency simulation ---------------
# Machado et al. (2009) severity-1 matrices on linear RGB, recalled (not verified this session).
CVD = {
    "deuteranopia": np.array(
        [
            [0.367322, 0.860646, -0.227968],
            [0.280085, 0.672501, 0.047413],
            [-0.011820, 0.042940, 0.968881],
        ]
    ),
    "protanopia": np.array(
        [
            [0.152286, 1.052583, -0.204868],
            [0.114503, 0.786281, 0.099216],
            [-0.003882, -0.048116, 1.051998],
        ]
    ),
}


def _linear(rgb) -> np.ndarray:
    c = np.asarray(rgb, dtype=float) / 255
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def _lab(lin: np.ndarray) -> np.ndarray:
    m = np.array([[0.4124, 0.3576, 0.1805], [0.2126, 0.7152, 0.0722], [0.0193, 0.1192, 0.9505]])
    x, y, z = m @ lin / np.array([0.95047, 1.0, 1.08883])
    f = [t ** (1 / 3) if t > 0.008856 else 7.787 * t + 16 / 116 for t in (x, y, z)]
    return np.array([116 * f[1] - 16, 500 * (f[0] - f[1]), 200 * (f[1] - f[2])])


def lab_views(rgb) -> list[np.ndarray]:
    """Lab coordinates as seen with normal vision, deuteranopia, protanopia."""
    lin = _linear(rgb)
    return [_lab(lin)] + [_lab(np.clip(m @ lin, 0, 1)) for m in CVD.values()]


def min_delta_e(cols, view: int = 0) -> float:
    """Smallest CIE76 difference between any two colours (view 0 normal, 1 deut., 2 prot.)."""
    labs = [lab_views(c)[view] for c in cols]
    return min(
        (float(np.linalg.norm(a - b)) for a, b in itertools.combinations(labs, 2)), default=np.inf
    )


def luminance(rgb) -> float:
    """WCAG relative luminance of an sRGB colour (0-255)."""
    lin = _linear(rgb)
    return float(0.2126 * lin[0] + 0.7152 * lin[1] + 0.0722 * lin[2])


@functools.lru_cache(maxsize=256)
def _best_subset(cols: tuple, n: int) -> tuple[int, ...]:
    """Indices (in palette order) of the n colours whose smallest pairwise Delta E (normal
    vision) is largest; ties broken by the colour-blind views."""
    labs = [lab_views(c) for c in cols]

    def score(sub):
        return tuple(
            min(
                float(np.linalg.norm(labs[i][v] - labs[j][v]))
                for i, j in itertools.combinations(sub, 2)
            )
            for v in range(3)
        )

    return max(itertools.combinations(range(len(cols)), n), key=score)


def _pick(cfg: dict, name: str, n: int) -> list[int]:
    """Indices of the palette colours used for n cell colours. Palettes listed in
    [palette] best_subset with more than n colours use their best-separated n colours."""
    cols = [_hex(c) for c in cfg["palettes"][name]]
    if len(cols) > n >= 2 and name in cfg["palette"].get("best_subset", []):
        return list(_best_subset(tuple(cols), n))
    return list(range(len(cols)))


def ui_colours(cfg: dict, name: str | None = None, n: int = 5) -> dict:
    """Interface colours taken from the palette's n cell colours: bg = the darkest, fg (text,
    meter face) = the lightest, accent (meter needle) = the most chromatic of the others;
    tones mixed between bg and fg: mid (secondary text), btn / btn_on (buttons); chart and
    line: a mid grey with the hue of the darkest colour. Also the cell-colour indices
    "dark" and "light" (the darkest colour gets a light contour on the interface)."""
    cols = [tuple(int(v) for v in c) for c in palette_rgb(cfg, n, name)]
    lum = [luminance(c) for c in cols]
    dark = int(np.argmin(lum))
    light = int(np.argmax(lum))
    bg, fg = cols[dark], cols[light]
    others = [k for k in range(len(cols)) if k not in (dark, light)]

    def chroma(c) -> float:
        lab = lab_views(c)[0]
        return float(np.hypot(lab[1], lab[2]))

    accent = cols[max(others, key=lambda k: chroma(cols[k]))] if others else fg

    def mix(f: float) -> tuple[int, int, int]:
        return tuple(int(round(b + f * (t - b))) for b, t in zip(bg, fg, strict=True))  # type: ignore[return-value]

    # chart: a mid grey tinted with the hue of the darkest colour, so that dark and light
    # curves both stand out
    h, _, sat = colorsys.rgb_to_hls(*(v / 255 for v in bg))
    cs = cfg["palette"].get("chart_tint", 0.25) * sat

    def tinted(lightness: float) -> tuple[int, int, int]:
        return tuple(int(round(255 * v)) for v in colorsys.hls_to_rgb(h, lightness, cs))  # type: ignore[return-value]

    lc = cfg["palette"].get("chart_lightness", 0.5)
    return {
        "bg": bg,
        "fg": fg,
        "accent": accent,
        "mid": mix(0.6),
        "btn": mix(0.15),
        "btn_on": mix(0.32),
        "chart": tinted(lc),
        "line": tinted(lc + 0.07),
        "dark": dark,
        "light": light,
        "contrast": (max(lum) + 0.05) / (min(lum) + 0.05),
    }


def palette_rgb(cfg: dict, n: int, name: str | None = None) -> np.ndarray:
    """(n, 3) uint8 cell colours of the named palette (default from the config). Greyscale is
    evenly spaced in lightness; other palettes are extended with pastel hues if too short."""
    name = name or cfg["palette"]["default"]
    if name == "greyscale":
        levels = np.linspace(0.30, 0.95, n) if n > 1 else np.array([0.7])
        return np.array([[int(255 * v)] * 3 for v in levels], dtype=np.uint8)
    allc = [_hex(c) for c in cfg["palettes"][name]]
    out = [allc[k] for k in _pick(cfg, name, n)][:n]
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
    keep = _pick(cfg, name, n)
    given = [given_all[k] for k in keep if k < len(given_all)][:n]
    rgb = palette_rgb(cfg, n, name)
    out = given + [_auto_name(rgb[k]) for k in range(len(given), n)]
    seen: dict[str, int] = {}
    for k, s in enumerate(out):
        seen[s] = seen.get(s, 0) + 1
        if seen[s] > 1:
            out[k] = f"{s} {seen[s]}"
    return out


def worst_delta_e(cfg: dict, n: int, name: str) -> float:
    """Smallest pairwise Delta E of the palette's n cell colours, in the worst of normal
    vision, deuteranopia and protanopia."""
    cols = list(palette_rgb(cfg, n, name))
    return min(min_delta_e(cols, v) for v in range(3))


def menu_names(cfg: dict, n: int) -> list[tuple[str, float]]:
    """Palettes offered in the viewer menu ([palette] menu, default first), each with its
    worst_delta_e for n colours."""
    names = list(cfg["palette"].get("menu", cfg["palettes"]))
    d = cfg["palette"]["default"]
    names = [d] + [nm for nm in names if nm != d]
    return [(nm, worst_delta_e(cfg, n, nm)) for nm in names]
