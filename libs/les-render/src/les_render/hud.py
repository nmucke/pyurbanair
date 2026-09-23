"""HUD overlay rendering: sim clock, inflow compass, colorbars, captions.

Writes ``hud/hud.<FFFF>.png`` -- transparent RGBA PNGs at render resolution,
composited over the rendered frames by :mod:`les_render.compose` (or by
either renderer directly). ``<FFFF>`` is a *file* index, not a video frame
index, exactly like every other layer in the bundle (see
``docs/les_render.md#bundle-layout``): file ``k`` shows video frame
``k * frame_step``, and the renderer displays file ``floor(i / frame_step)``
for video frame ``i``.

Design: dark, semi-transparent thin outline behind white text/glyphs (via
PIL's ``stroke_width``/``stroke_fill``) so every element reads on both dark
LES-volume backgrounds and bright daylight-preset ones, without an opaque
panel behind it. Static content (the compass dial's face, the playback-speed
tag, the title, per-shot colorbars and captions) is rendered once and cached
-- only the sim clock, the inflow arrow and the live speed readout are
redrawn per frame -- to stay well under the ~50 ms/frame budget at 1500+
frames. Pass ``spec["workers"] > 1`` to fan frames out across a
``forkserver`` process pool for large sequences (``n_workers`` is accepted
as a deprecated alias for ``workers``, for consistency with the other
exporters).
"""

from __future__ import annotations

import copy
import dataclasses
import functools
import math
import multiprocessing
import pathlib
import re
from concurrent.futures import ProcessPoolExecutor
from typing import Any, Optional

import numpy as np
from PIL import Image, ImageDraw, ImageFont

# -- defaults ---------------------------------------------------------------

DEFAULT_SPEC: dict[str, Any] = {
    "frame_step": 1,
    "title": None,  # e.g. "Idealized urban array -- run 0000"
    "captions": {},  # {shot_name: caption_text}
    "caption": None,  # fallback caption when a shot has no entry in "captions"
    "show_clock": True,
    "show_compass": bool,  # placeholder, overwritten below (avoid mypy noise)
    "show_colorbars": True,
    "show_scale_hint": True,
    "margin_frac": 0.02,  # UI margin as a fraction of min(width, height)
    "text_color": (255, 255, 255, 255),
    "outline_color": (0, 0, 0, 175),
    "accent_color": (255, 205, 90, 255),  # compass needle / highlights
    "workers": 1,
}
DEFAULT_SPEC["show_compass"] = True

_FONT_CANDIDATES = {
    "regular": [
        "/usr/share/fonts/liberation-sans/LiberationSans-Regular.ttf",
        "/usr/share/fonts/dejavu-sans-fonts/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ],
    "bold": [
        "/usr/share/fonts/liberation-sans/LiberationSans-Bold.ttf",
        "/usr/share/fonts/dejavu-sans-fonts/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    ],
}


def _merge_spec(spec: Optional[dict[str, Any]]) -> dict[str, Any]:
    spec = dict(spec or {})
    if "n_workers" in spec:  # deprecated alias for "workers"
        spec.setdefault("workers", spec.pop("n_workers"))
    merged = copy.deepcopy(DEFAULT_SPEC)
    for k, v in spec.items():
        if isinstance(v, dict) and isinstance(merged.get(k), dict):
            merged[k] = {**merged[k], **v}
        else:
            merged[k] = v
    return merged


@functools.lru_cache(maxsize=64)
def _find_font(kind: str, size: int) -> ImageFont.FreeTypeFont:
    """Cached: loading+parsing a TTF is the dominant per-frame cost otherwise
    (each frame only needs a handful of fixed sizes)."""
    for path in _FONT_CANDIDATES[kind]:
        if pathlib.Path(path).is_file():
            return ImageFont.truetype(path, size)
    # Fallback only reachable if none of the candidate TTFs exist on the
    # host; callers rely on the FreeTypeFont API (.size, .getmetrics()),
    # which the bitmap default font doesn't have.
    return ImageFont.load_default()  # type: ignore[return-value, unused-ignore]


def _linear_to_srgb(c: np.ndarray) -> np.ndarray:
    c = np.clip(c, 0.0, 1.0)
    return np.where(c <= 0.0031308, c * 12.92, 1.055 * (c ** (1.0 / 2.4)) - 0.055)


def _small_caps(text: str) -> str:
    """A cheap small-caps look without a real small-caps font: upper-case,
    tracked out with thin spaces so it doesn't look shouty."""
    return " ".join(text.upper())


_UNITS_RE = re.compile(r"^(.*?)\s*\[([^\[\]]+)\]\s*$")


def _split_variable_units(variable: str) -> tuple[str, str]:
    """``"wind speed [m/s]"`` -> ``("wind speed", "m/s")``; no bracketed
    suffix -> ``(variable, "")``. Only the name gets the letter-spaced caps
    treatment -- units render in their original case."""
    m = _UNITS_RE.match(variable)
    if m:
        return m.group(1), m.group(2)
    return variable, ""


# Curated human label + unit for the flat colour keys layers actually carry
# (``layer["variable"]``, e.g. "speed", "vorticity_magnitude" -- not a
# hand-authored "name [units]" string). ``_TRANSFORM_LABELS`` overrides this
# when a volume applies a transform (``layer["transform"]``, e.g. a
# "speed" volume with ``transform: "abs_excess"`` is really a deviation
# field, not raw speed).
_VARIABLE_LABELS: dict[str, tuple[str, str]] = {
    "speed": ("wind speed", "m/s"),
    "u": ("u velocity", "m/s"),
    "v": ("v velocity", "m/s"),
    "w": ("vertical velocity", "m/s"),
    "vorticity_magnitude": ("vorticity", "1/s"),
    "q_criterion": ("Q-criterion", "1/s²"),
    "pressure": ("pressure (kinematic)", "m²/s²"),
    "gamma_norm": ("vorticity (normalised)", ""),
}

_TRANSFORM_LABELS: dict[str, tuple[str, str]] = {
    "abs_excess": ("speed deviation |U − Ū|", "m/s"),
}


def _variable_label(layer: dict) -> tuple[str, str]:
    """Human label + unit for a layer's colour variable.

    Known ``transform``/``variable`` keys map to a curated name (see
    ``_TRANSFORM_LABELS``/``_VARIABLE_LABELS``); anything else falls back to
    parsing a hand-authored ``"name [units]"`` string, so test fixtures and
    hand-written ``render.yaml`` overrides keep working unchanged.
    """
    transform = layer.get("transform")
    if transform in _TRANSFORM_LABELS:
        return _TRANSFORM_LABELS[transform]
    variable = str(layer.get("variable", layer.get("name", "")))
    if variable in _VARIABLE_LABELS:
        return _VARIABLE_LABELS[variable]
    return _split_variable_units(variable)


def _fmt_num(x: float) -> str:
    """2-significant-figure, plain (non-scientific) display, with near-zero
    values collapsed to a clean ``"0"`` (e.g. ``0.016`` -> ``"0"``) -- a HUD
    range readout doesn't need to show numerical noise near a wall."""
    if not math.isfinite(x):
        return str(x)
    if abs(x) < 0.1:
        return "0"
    magnitude = math.floor(math.log10(abs(x)))
    decimals = max(0, 1 - magnitude)
    rounded = round(x, decimals)
    s = f"{rounded:.{decimals}f}" if decimals > 0 else f"{rounded:.0f}"
    if "." in s:
        s = s.rstrip("0").rstrip(".")
    return s


# Short tag naming a layer's origin, used to disambiguate two colorbars that
# share a variable label but differ in colormap/range (e.g. two "wind
# speed" bars from different layers).
_LAYER_TAG_WORDS: dict[str, str] = {
    "streaklines": "smoke",
    "trails": "tracers",
    "ground_speed": "ground",
    "vortices": "vortex cores",
    "speed_glow": "glow",
}


def _layer_tag(layer: dict) -> tuple[str, str]:
    """``(tag word, technical suffix)``, e.g. ``("ground", " (z = 2 m)")``
    for a ground-level slice layer."""
    name = str(layer.get("name", "")) or "layer"
    word = _LAYER_TAG_WORDS.get(name, name.replace("_", " "))
    suffix = ""
    axis, position = layer.get("axis"), layer.get("position")
    if layer.get("type") == "slice" and axis and position is not None:
        suffix = f" ({axis} = {_fmt_num(float(position))} m)"
    return word, suffix


def _build_colorbar_entries(layers: list[dict]) -> list[dict[str, Any]]:
    """Group ``layers`` into the bars ``_colorbar_stack`` should draw.

    Layers that fully agree on ``(label, colormap, rounded range)`` are
    merged into one bar. Layers that share a label but differ in colormap or
    range stay separate and get their label prefixed with a short tag
    naming the originating layer, so two differently-scaled "wind speed"
    bars don't both just read "SPEED".
    """
    groups: dict[tuple[str, Any, str, str], list[dict]] = {}
    order: list[tuple[str, Any, str, str]] = []
    for layer in layers:
        name_label, _unit = _variable_label(layer)
        vmin, vmax = layer.get("range", [0.0, 1.0])
        key = (
            name_label,
            layer.get("colormap"),
            _fmt_num(float(vmin)),
            _fmt_num(float(vmax)),
        )
        if key not in groups:
            groups[key] = []
            order.append(key)
        groups[key].append(layer)

    labels_seen: dict[str, set] = {}
    for key in order:
        labels_seen.setdefault(key[0], set()).add(key)

    entries = []
    for key in order:
        name_label, _cmap, vmin_s, vmax_s = key
        layer0 = groups[key][0]
        _, unit = _variable_label(layer0)
        segments: list[tuple[str, bool]]
        if len(labels_seen[name_label]) > 1:
            tag_word, tag_suffix = _layer_tag(layer0)
            segments = [
                (tag_word, True),
                (tag_suffix, False),
                (" · ", False),
                (name_label, True),
            ]
        else:
            segments = [(name_label, True)]
        if unit:
            segments.append(("  " + unit, False))
        entries.append(
            {
                "segments": segments,
                "vmin_s": vmin_s,
                "vmax_s": vmax_s,
                "lut": layer0.get("lut_linear_rgb"),
            }
        )
    return entries


def _draw_rich_label(
    draw: ImageDraw.ImageDraw,
    xy: tuple[float, float],
    segments: list[tuple[str, bool]],
    font: ImageFont.FreeTypeFont,
    spec: dict,
) -> None:
    """Draw ``segments`` (``text``, ``is_caps``) left to right from ``xy``.

    ``is_caps`` segments get the letter-spaced small-caps treatment at full
    opacity (e.g. a variable name or a disambiguating layer tag); other
    segments (units, a technical parenthetical like ``" (z = 2 m)"``, a
    separator) are drawn verbatim, slightly dimmed.
    """
    x, y = xy
    stroke_width = max(1, font.size // 9)
    for text, is_caps in segments:
        if not text:
            continue
        rendered = _small_caps(text) if is_caps else text
        fill = (
            tuple(spec["text_color"])
            if is_caps
            else tuple(spec["text_color"][:3]) + (200,)
        )
        _draw_text(draw, (x, y), rendered, font, spec, anchor="la", fill=fill)
        bbox = draw.textbbox(
            (x, y), rendered, font=font, anchor="la", stroke_width=stroke_width
        )
        x = bbox[2] + max(3, round(font.size * 0.3))


def _draw_variable_label(
    draw: ImageDraw.ImageDraw,
    xy: tuple[float, float],
    variable: str,
    font: ImageFont.FreeTypeFont,
    spec: dict,
) -> None:
    """Draw a plain ``"name [units]"``-style label as ``NAME  units`` --
    small-caps name, plain-case units (e.g. ``"WIND SPEED  m/s"``, not
    ``"WIND SPEED [M/S]"``)."""
    name, units = _split_variable_units(variable)
    segments: list[tuple[str, bool]] = [(name, True)]
    if units:
        segments.append(("  " + units, False))
    _draw_rich_label(draw, xy, segments, font, spec)


# -- geometry/layout helpers -------------------------------------------------


@dataclasses.dataclass(frozen=True)
class _Layout:
    width: int
    height: int
    margin: int
    unit: int  # base UI unit (px), everything else scales off this

    @classmethod
    def make(cls, width: int, height: int, margin_frac: float) -> "_Layout":
        unit = max(8, round(min(width, height) * 0.018))
        margin = max(unit, round(min(width, height) * margin_frac))
        return cls(width, height, margin, unit)


def _draw_text(
    draw: ImageDraw.ImageDraw,
    xy: tuple[float, float],
    text: str,
    font: ImageFont.FreeTypeFont,
    spec: dict,
    anchor: str = "la",
    fill: Optional[tuple[int, int, int, int]] = None,
) -> None:
    draw.text(
        xy,
        text,
        font=font,
        fill=fill or tuple(spec["text_color"]),
        anchor=anchor,
        stroke_width=max(1, font.size // 9),
        stroke_fill=tuple(spec["outline_color"]),
    )


# -- inflow -------------------------------------------------------------


def inflow_block(params_ds: Any) -> Optional[dict[str, list[float]]]:
    """Build the manifest ``inflow`` block from a params dataset
    (``inflow_angle`` [deg], ``velocity_magnitude`` [m/s] vs ``time``).

    Returns ``None`` if ``params_ds`` is ``None`` or lacks the variables.
    Angle convention: degrees from +x, counter-clockwise, matching
    ``pyudales.utils.inflow_utils.angle_to_velocity``
    (``u = speed*cos(angle), v = speed*sin(angle)``) -- ``angle_deg=0`` blows
    toward ``+x``.
    """
    if params_ds is None:
        return None
    if "inflow_angle" not in params_ds or "velocity_magnitude" not in params_ds:
        return None
    time = np.asarray(params_ds["time"].values, dtype=float)
    angle = np.asarray(params_ds["inflow_angle"].values, dtype=float)
    speed = np.asarray(params_ds["velocity_magnitude"].values, dtype=float)
    if angle.ndim == 0:
        time = np.array([0.0, 1.0])
        angle = np.full(2, float(angle))
        speed = np.full(2, float(speed))
    return {
        "time": [round(float(v), 4) for v in time],
        "angle_deg": [round(float(v), 4) for v in angle],
        "speed": [round(float(v), 4) for v in speed],
    }


def _interp_inflow(inflow: dict, t: float) -> tuple[float, float]:
    time = np.asarray(inflow["time"], dtype=float)
    angle = np.asarray(inflow["angle_deg"], dtype=float)
    speed = np.asarray(inflow["speed"], dtype=float)
    # unwrap so interpolation across a wrap (e.g. 179 -> -179) takes the short way
    angle_unwrapped = np.degrees(np.unwrap(np.radians(angle)))
    a = float(np.interp(t, time, angle_unwrapped))
    s = float(np.interp(t, time, speed))
    return a, s


# -- static elements (cached) ------------------------------------------------


def _compass_face(
    layout: _Layout, spec: dict
) -> tuple[Image.Image, tuple[int, int], int]:
    """The compass dial background (circle, ticks, axis labels), pre-rendered.

    Returns ``(image, centre_xy, radius)``; ``image`` is ``layout``-sized so
    it can be alpha-composited directly onto the frame canvas.
    """
    im = Image.new("RGBA", (layout.width, layout.height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(im)
    r = layout.unit * 3
    ring_width = max(1, layout.unit // 6)
    # PIL draws an ellipse outline centred on the path, so it bleeds
    # ring_width/2 outward from the nominal radius -- inset the centre by
    # that much too, or the stroke itself pokes past the margin even though
    # the "radius" math alone would land exactly on it.
    inset = math.ceil(ring_width / 2)
    cx = layout.width - layout.margin - r - inset
    cy = layout.margin + r + inset
    ring = tuple(spec["text_color"][:3]) + (150,)
    draw.ellipse((cx - r, cy - r, cx + r, cy + r), outline=ring, width=ring_width)
    draw.ellipse(
        (cx - r * 0.06, cy - r * 0.06, cx + r * 0.06, cy + r * 0.06),
        fill=tuple(spec["text_color"]),
    )
    # Axis reference: subtle "x"/"y" ticks drawn *inside* the rim (not
    # outside it, like the old "+X"/"+Y" labels were) so the whole dial
    # stays within the margin on every edge -- nothing can clip the canvas.
    tick_font = _find_font("regular", max(8, int(layout.unit * 0.65)))
    tick_fill = tuple(spec["text_color"][:3]) + (170,)
    pad = max(2, round(layout.unit * 0.35))
    _draw_text(
        draw, (cx + r - pad, cy), "x", tick_font, spec, anchor="rm", fill=tick_fill
    )
    _draw_text(
        draw, (cx, cy - r + pad), "y", tick_font, spec, anchor="ma", fill=tick_fill
    )
    label_font = _find_font("regular", max(9, int(layout.unit * 0.75)))
    _draw_text(
        draw, (cx, cy + r + layout.unit * 0.5), "inflow", label_font, spec, anchor="ma"
    )
    return im, (cx, cy), r


def _colorbar_stack(layout: _Layout, layers: list[dict], spec: dict) -> Image.Image:
    """Vertically stacked colorbars in the bottom-left corner -- one per
    distinct ``(label, colormap, range)`` among the visible layers (see
    :func:`_build_colorbar_entries`)."""
    im = Image.new("RGBA", (layout.width, layout.height), (0, 0, 0, 0))
    if not layers:
        return im
    entries = _build_colorbar_entries(layers)
    draw = ImageDraw.Draw(im)
    bar_w = max(60, round(layout.width * 0.16))
    bar_h = max(6, round(layout.unit * 0.55))
    font = _find_font("regular", max(10, int(layout.unit * 0.85)))
    # Measured text line height (ascent + descent) for this font, so blocks
    # are spaced from real glyph metrics instead of a guessed constant --
    # guessed constants are what caused labels/range numbers to overlap the
    # neighbouring bar in the stack.
    ascent, descent = font.getmetrics()
    line_h = ascent + descent
    pad = max(2, round(layout.unit * 0.25))

    x0 = layout.margin
    y = layout.height - layout.margin  # bottom of the whole stack; grows upward
    for entry in reversed(entries):  # lay out bottom-up: first entry ends up on top
        y -= line_h + pad  # room for the min/max range line below the bar
        range_y = y + pad
        y -= bar_h
        bar_top = y
        y -= pad + line_h  # room for the label above the bar

        _draw_rich_label(draw, (x0, y), entry["segments"], font, spec)

        lut = np.asarray(entry["lut"], dtype=float)
        if lut.size:
            srgb = (_linear_to_srgb(lut) * 255.0).astype(np.uint8)
            grad = Image.fromarray(srgb[None, :, :], mode="RGB").resize(
                (bar_w, bar_h), Image.Resampling.BILINEAR
            )
            im.paste(grad, (x0, bar_top))
            outline = tuple(spec["text_color"][:3]) + (120,)
            draw.rectangle(
                (x0, bar_top, x0 + bar_w, bar_top + bar_h), outline=outline, width=1
            )

        _draw_text(draw, (x0, range_y), entry["vmin_s"], font, spec, anchor="la")
        _draw_text(
            draw, (x0 + bar_w, range_y), entry["vmax_s"], font, spec, anchor="ra"
        )

        y -= round(layout.unit * 0.6)  # gap before the next block
    return im


def _title_and_caption(
    layout: _Layout, title: Optional[str], caption: Optional[str], spec: dict
) -> Image.Image:
    im = Image.new("RGBA", (layout.width, layout.height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(im)
    if title:
        font = _find_font("bold", max(14, int(layout.unit * 1.3)))
        _draw_text(
            draw,
            (layout.width / 2, layout.margin),
            _small_caps(title),
            font,
            spec,
            anchor="ma",
        )
    if caption:
        font = _find_font("regular", max(11, int(layout.unit * 0.95)))
        _draw_text(
            draw,
            (layout.width / 2, layout.height - layout.margin),
            caption,
            font,
            spec,
            anchor="mb",
            fill=tuple(spec["text_color"][:3]) + (230,),
        )
    return im


def _scale_hint(layout: _Layout, manifest: dict, spec: dict) -> Image.Image:
    im = Image.new("RGBA", (layout.width, layout.height), (0, 0, 0, 0))
    if not spec.get("show_scale_hint", True):
        return im
    geometry = manifest.get("geometry") or {}
    h = geometry.get("max_building_height")
    if h is None:
        return im
    draw = ImageDraw.Draw(im)
    font = _find_font("regular", max(9, int(layout.unit * 0.75)))
    text = f"buildings ≈ {float(h):.0f} m tall"
    _draw_text(
        draw,
        (layout.width - layout.margin, layout.height - layout.margin),
        text,
        font,
        spec,
        anchor="rb",
        fill=tuple(spec["text_color"][:3]) + (200,),
    )
    return im


# -- per-frame dynamic elements ----------------------------------------------


def _draw_clock(
    canvas: Image.Image,
    layout: _Layout,
    t: float,
    playback_speed: Optional[float],
    spec: dict,
) -> None:
    draw = ImageDraw.Draw(canvas)
    font = _find_font("bold", max(16, int(layout.unit * 1.5)))
    _draw_text(
        draw, (layout.margin, layout.margin), f"t = {t:0.1f} s", font, spec, anchor="la"
    )
    if playback_speed:
        tag_font = _find_font("regular", max(11, int(layout.unit * 0.95)))
        _draw_text(
            draw,
            (layout.margin, layout.margin + font.size + layout.unit * 0.3),
            f"{playback_speed:g}x playback",
            tag_font,
            spec,
            anchor="la",
            fill=tuple(spec["text_color"][:3]) + (210,),
        )


def _draw_inflow(
    canvas: Image.Image,
    layout: _Layout,
    centre: tuple[int, int],
    radius: int,
    angle_deg: float,
    speed: float,
    spec: dict,
) -> None:
    draw = ImageDraw.Draw(canvas)
    theta = math.radians(angle_deg)
    dx, dy = math.cos(theta), math.sin(
        theta
    )  # world (x, y); +y drawn "up" (screen y flipped below)
    cx, cy = centre
    tip = (cx + dx * radius * 0.85, cy - dy * radius * 0.85)
    tail = (cx - dx * radius * 0.55, cy + dy * radius * 0.55)
    accent = tuple(spec["accent_color"])
    draw.line([tail, tip], fill=accent, width=max(2, radius // 8))
    # arrowhead
    head = math.atan2(-(tip[1] - tail[1]), tip[0] - tail[0])
    for side in (1.0, -1.0):
        a = head + side * math.radians(150)
        pt = (
            tip[0] + math.cos(a) * radius * 0.28,
            tip[1] - math.sin(a) * radius * 0.28,
        )
        draw.line([tip, pt], fill=accent, width=max(2, radius // 8))
    font = _find_font("bold", max(11, int(layout.unit * 0.95)))
    _draw_text(
        draw,
        (cx, cy + radius + layout.unit * 1.6),
        f"{speed:0.1f} m/s",
        font,
        spec,
        anchor="ma",
    )


# -- shot lookup ------------------------------------------------------------


def _shot_for_frame(shots: list[dict], frame: int) -> Optional[dict]:
    for s in shots:
        if s["start"] <= frame <= s["end"]:
            return s
    return shots[-1] if shots else None


def _visible_layers(
    shot: Optional[dict], all_layers: list[dict], spec: dict
) -> list[dict]:
    if not spec.get("show_colorbars", True):
        return []
    if shot is None or shot.get("layers") is None:
        names = None
    else:
        names = set(shot["layers"])
    out = []
    for layer in all_layers:
        if "range" not in layer or "lut_linear_rgb" not in layer:
            continue  # not a coloured layer (shouldn't happen per the manifest contract, but be defensive)
        if names is None or layer.get("name") in names:
            out.append(layer)
    return out


# -- core render --------------------------------------------------------


class _Renderer:
    """Holds the caches so repeated frames (same process) stay cheap."""

    def __init__(self, manifest: dict, spec: dict):
        self.manifest = manifest
        self.spec = spec
        self.width = int(manifest["render"]["width"])
        self.height = int(manifest["render"]["height"])
        self.layout = _Layout.make(self.width, self.height, spec["margin_frac"])
        self.shots = manifest.get("shots") or []
        self.layers = manifest.get("layers") or []
        self.inflow = manifest.get("inflow")
        self.timeline = manifest["timeline"]
        self.playback_speed = self.timeline.get("playback_speed")

        self.compass_face: Optional[Image.Image]
        self.compass_centre: Optional[tuple[int, int]]
        self.compass_radius: Optional[int]
        if not spec.get("show_compass", True) or self.inflow is None:
            self.compass_face, self.compass_centre, self.compass_radius = (
                None,
                None,
                None,
            )
        else:
            self.compass_face, self.compass_centre, self.compass_radius = _compass_face(
                self.layout, spec
            )
        self.scale_hint = _scale_hint(self.layout, manifest, spec)
        self._shot_cache: dict[tuple, Image.Image] = {}

    def _static_base(self, shot: Optional[dict]) -> Image.Image:
        key = (shot["name"] if shot else None,)
        cached = self._shot_cache.get(key)
        if cached is not None:
            return cached.copy()

        base = Image.new("RGBA", (self.width, self.height), (0, 0, 0, 0))
        if self.compass_face is not None:
            base = Image.alpha_composite(base, self.compass_face)
        base = Image.alpha_composite(base, self.scale_hint)

        layers = _visible_layers(shot, self.layers, self.spec)
        base = Image.alpha_composite(
            base, _colorbar_stack(self.layout, layers, self.spec)
        )

        shot_name = shot["name"] if shot else None
        caption = (
            self.spec["captions"].get(shot_name, self.spec.get("caption"))
            if shot_name
            else self.spec.get("caption")
        )
        base = Image.alpha_composite(
            base,
            _title_and_caption(self.layout, self.spec.get("title"), caption, self.spec),
        )

        self._shot_cache[key] = base
        return base.copy()

    def render(self, video_frame: int, t: float) -> Image.Image:
        shot = _shot_for_frame(self.shots, video_frame)
        canvas = self._static_base(shot)

        if self.spec.get("show_clock", True):
            _draw_clock(canvas, self.layout, t, self.playback_speed, self.spec)

        if (
            self.compass_face is not None
            and self.compass_centre is not None
            and self.compass_radius is not None
            and self.inflow is not None
        ):
            angle, speed = _interp_inflow(self.inflow, t)
            _draw_inflow(
                canvas,
                self.layout,
                self.compass_centre,
                self.compass_radius,
                angle,
                speed,
                self.spec,
            )

        return canvas


def _file_indices(n_frames: int, frame_step: int) -> list[tuple[int, int]]:
    """``[(file_index, video_frame), ...]`` -- file ``k`` renders video frame
    ``k * frame_step`` (clamped to the last frame)."""
    n_files = max(1, math.ceil(n_frames / frame_step))
    return [(k, min(k * frame_step, n_frames - 1)) for k in range(n_files)]


def _render_chunk(
    manifest: dict, spec: dict, out_dir: str, items: list[tuple[int, int]]
) -> list[str]:
    renderer = _Renderer(manifest, spec)
    frame_times = manifest["timeline"].get("frame_times")
    t_start, dt = manifest["timeline"]["t_start"], manifest["timeline"]["dt"]
    hud_dir = pathlib.Path(out_dir) / "hud"
    written = []
    for file_index, video_frame in items:
        t = frame_times[video_frame] if frame_times else t_start + dt * video_frame
        img = renderer.render(video_frame, float(t))
        path = hud_dir / f"hud.{file_index:04d}.png"
        img.save(path)
        written.append(str(path))
    return written


def render_hud(
    manifest: dict, spec: Optional[dict], out_dir: pathlib.Path | str
) -> dict[str, Any]:
    """Render the HUD overlay sequence for a bundle.

    Writes ``<out_dir>/hud/hud.<FFFF>.png`` (RGBA, transparent background) and
    returns the manifest ``"hud"`` block: ``{"pattern": "hud/hud.{frame:04d}.png",
    "frame_step": N}``.
    """
    spec = _merge_spec(spec)
    out_dir = pathlib.Path(out_dir)
    (out_dir / "hud").mkdir(parents=True, exist_ok=True)

    n_frames = int(manifest["timeline"]["n_frames"])
    frame_step = max(1, int(spec["frame_step"]))
    items = _file_indices(n_frames, frame_step)

    n_workers = max(1, int(spec.get("workers", 1)))
    if n_workers > 1 and len(items) > n_workers:
        ctx = multiprocessing.get_context("forkserver")
        chunks = [items[i::n_workers] for i in range(n_workers)]
        with ProcessPoolExecutor(max_workers=n_workers, mp_context=ctx) as pool:
            futures = [
                pool.submit(_render_chunk, manifest, spec, str(out_dir), chunk)
                for chunk in chunks
            ]
            for f in futures:
                f.result()
    else:
        _render_chunk(manifest, spec, str(out_dir), items)

    return {"pattern": "hud/hud.{frame:04d}.png", "frame_step": frame_step}


__all__ = ["render_hud", "inflow_block", "DEFAULT_SPEC"]
