"""Shared PIL crop/compose helpers for the experiments report figures.

All section scripts (``make_esmda_figures.py``, ``make_filtering_figures.py``,
``make_filter_smoothing_figures.py``, ``make_comparison_figures.py``) import
this module so every figure in the report is cropped, labelled and scaled the
same way.

The raw per-run PNGs live in::

    presentations/isda_new/experiments/{esmda,filtering,filter_smoothing}/<run>/

Panels are *detected*, not hard-coded: :func:`panel_bands` finds the all-white
row gaps of a matplotlib figure and returns one vertical band per subplot
(figure suptitle dropped, per-panel titles / tick labels / xlabels kept with
their panel).  This works for all three campaigns even though the figures
differ slightly (e.g. the filter-smoothing sensor figure carries a "State at
validation sensors" suptitle and a cycle-index x-axis).

Typical use::

    import figlib as fl

    ids = fl.run_ids("esmda")                      # {"E1": "pypalm_...", ...}
    d = fl.run_dir("esmda", ids["E6"])
    fl.save(fl.crop_validation(d), out / "valid_E6.png")

Run ``python experiments_report/scripts/figlib_demo.py`` for a worked example
of every crop.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib
import numpy as np
from PIL import Image, ImageDraw, ImageFont

__all__ = [
    "REPO",
    "EXP",
    "METHODS",
    "ID_PREFIX",
    "run_ids",
    "run_dir",
    "panel_bands",
    "crop_validation",
    "crop_assimilation",
    "crop_param_error",
    "crop_marginals",
    "crop_rank_hist",
    "crop_slices",
    "crop_param_evolution",
    "label",
    "hstack",
    "vstack",
    "grid",
    "trim",
    "save",
]

REPO = Path(__file__).resolve().parents[2]
EXP = REPO / "presentations" / "isda_new" / "experiments"

METHODS = ("esmda", "filtering", "filter_smoothing")
ID_PREFIX = {"esmda": "E", "filtering": "F", "filter_smoothing": "H"}

WHITE = (255, 255, 255)
STRIP_BG = (236, 239, 245)
STRIP_FG = (15, 15, 15)
FONT_PATH = Path(matplotlib.get_data_path()) / "fonts" / "ttf" / "DejaVuSans-Bold.ttf"

# Directory entries that are not runs.
_NOT_RUNS = {"_logs", "_scratch", ".DS_Store"}


# --------------------------------------------------------------------------- #
# run bookkeeping
# --------------------------------------------------------------------------- #
def run_ids(method: str) -> dict[str, str]:
    """Ordered ``{ID: run_name}`` map for one campaign (STYLE_SPEC section 1).

    IDs are assigned by sorting the run-directory names alphabetically and
    numbering from 1, with the campaign prefix E / F / H.
    """
    if method not in METHODS:
        raise ValueError(f"unknown method {method!r}; expected one of {METHODS}")
    root = EXP / method
    names = sorted(
        p.name for p in root.iterdir() if p.is_dir() and p.name not in _NOT_RUNS
    )
    pre = ID_PREFIX[method]
    return {f"{pre}{i}": n for i, n in enumerate(names, start=1)}


def run_dir(method: str, run_name: str) -> Path:
    """Path of one run directory (``run_name`` may also be an ID such as ``E6``)."""
    if run_name and run_name[0] in ID_PREFIX.values() and run_name[1:].isdigit():
        run_name = run_ids(method)[run_name]
    return EXP / method / run_name


def _as_dir(run_dir_: Path | str) -> Path:
    return Path(run_dir_)


# --------------------------------------------------------------------------- #
# panel detection
# --------------------------------------------------------------------------- #
def _open(path: Path) -> Image.Image:
    if not Path(path).exists():
        raise FileNotFoundError(path)
    return Image.open(path).convert("RGB")


def _runs(blank: np.ndarray, min_run: int = 3) -> list[tuple[int, int]]:
    """Start/stop indices of the maximal non-blank runs of ``blank``."""
    out: list[tuple[int, int]] = []
    n = len(blank)
    i = 0
    while i < n:
        if blank[i]:
            i += 1
            continue
        j = i
        while j < n and not blank[j]:
            j += 1
        if j - i >= min_run:
            out.append((i, j))
        i = j
    return out


def _bands(
    arr: np.ndarray, tol: int, panel_titles: bool = True
) -> tuple[tuple[int, int], ...]:
    h, w = arr.shape
    rows = _runs(np.asarray((arr >= tol).all(axis=1)))
    if not rows:
        return ((0, h),)

    # "Bodies" are the axes rectangles: much taller than titles / tick labels.
    tallest = max(b - a for a, b in rows)
    bodies = [(a, b) for a, b in rows if (b - a) >= 0.4 * tallest]

    # Gap below which a run above a body still belongs to that body (panel
    # title).  Absolute-ish, scaled with the figure width so it survives a
    # different dpi; the suptitle always sits further away than this.
    near = max(8, round(14 * w / 1667))

    # Top of the first band: walk up from the first body through runs that are
    # "near" enough to belong to it (its own title), stopping at the suptitle.
    top = bodies[0][0]
    if panel_titles:
        for a, b in reversed([r for r in rows if r[1] <= bodies[0][0]]):
            if top - b <= near:
                top = a
            else:
                break

    edges = [max(0, top - 4)]
    for (_, prev_end), (next_start, _) in zip(bodies[:-1], bodies[1:]):
        # split at the middle of the widest blank gap between the two bodies
        inner = [r for r in rows if r[0] >= prev_end and r[1] <= next_start]
        marks = [prev_end] + [x for r in inner for x in r] + [next_start]
        gaps = [
            (marks[k + 1] - marks[k], (marks[k] + marks[k + 1]) // 2)
            for k in range(0, len(marks) - 1, 2)
        ]
        edges.append(max(gaps)[1])
    edges.append(h)
    return tuple((edges[k], edges[k + 1]) for k in range(len(edges) - 1))


@lru_cache(maxsize=128)
def _bands_cached(
    path: str, mtime: float, tol: int, panel_titles: bool
) -> tuple[tuple[int, int], ...]:
    return _bands(np.asarray(Image.open(path).convert("L")), tol, panel_titles)


def panel_bands(
    im_or_path: Image.Image | Path | str, tol: int = 250, panel_titles: bool = True
) -> list[tuple[int, int]]:
    """Vertical ``(y0, y1)`` band of every subplot row of a matplotlib figure.

    The figure suptitle is excluded; per-panel titles, tick labels and x-labels
    stay with their panel.  Accepts a path (cached) or an in-memory image.
    """
    if isinstance(im_or_path, (str, Path)):
        p = Path(im_or_path)
        return list(_bands_cached(str(p), p.stat().st_mtime, tol, panel_titles))
    return list(_bands(np.asarray(im_or_path.convert("L")), tol, panel_titles))


def _panels(
    path: Path, keep: Sequence[int], panel_titles: bool = True
) -> list[Image.Image]:
    im = _open(path)
    bands = panel_bands(path, panel_titles=panel_titles)
    out = []
    for i in keep:
        y0, y1 = bands[i]
        out.append(im.crop((0, y0, im.width, y1)))
    return out


def _stack_panels(
    path: Path, keep: Sequence[int], gap: int = 6, panel_titles: bool = True
) -> Image.Image:
    return vstack(_panels(path, keep, panel_titles=panel_titles), gap=gap)


# --------------------------------------------------------------------------- #
# crops
# --------------------------------------------------------------------------- #
def crop_validation(
    run_dir_: Path | str,
    sensors: Iterable[int] = (0, 3),
    with_error: bool = True,
    title: str | None = None,
) -> Image.Image:
    """Validation-sensor panels + the sensor-error panel, stacked.

    The reference layout of ``figures/comparison/cmp_valid_timeseries.png``:
    sensor 0 (ground), sensor 3 (elevated) and the RMSE/CRPS error panel.
    """
    return _sensor_fig(
        _as_dir(run_dir_) / "sensor_timeseries_validation.png",
        sensors,
        with_error,
        title,
    )


def crop_assimilation(
    run_dir_: Path | str,
    sensors: Iterable[int] = (0,),
    with_error: bool = True,
    title: str | None = None,
) -> Image.Image:
    """Assimilated-sensor panels + the sensor-error panel, stacked."""
    return _sensor_fig(
        _as_dir(run_dir_) / "sensor_timeseries_assimilation.png",
        sensors,
        with_error,
        title,
    )


def _sensor_fig(
    path: Path, sensors: Iterable[int], with_error: bool, title: str | None
) -> Image.Image:
    bands = panel_bands(path)
    n_sensor = len(bands) - 1  # last band is the sensor-error panel
    keep = [s for s in sensors if 0 <= s < n_sensor]
    missing = [s for s in sensors if s not in keep]
    if missing:
        print(
            f"  figlib: {path.parent.name}: no sensor panel {missing} (found {n_sensor})"
        )
    if with_error:
        keep.append(len(bands) - 1)
    im = _stack_panels(path, keep)
    return label(im, title) if title else im


def crop_param_error(run_dir_: Path | str, title: str | None = None) -> Image.Image:
    """Parameter-error time series: inflow angle panel over |U| panel.

    Returns the single panel unchanged if the raw figure only has one.
    """
    path = _as_dir(run_dir_) / "parameter_error.png"
    im = _stack_panels(path, range(len(panel_bands(path))))
    return label(im, title) if title else im


def crop_marginals(run_dir_: Path | str, title: str | None = None) -> Image.Image:
    """Final-knot prior/posterior violins: alpha and |U| side by side.

    The raw figure is a single 1x2 row, so this simply drops the suptitle.
    """
    path = _as_dir(run_dir_) / "parameter_marginals.png"
    im = trim(_stack_panels(path, range(len(panel_bands(path)))))
    return label(im, title) if title else im


def crop_rank_hist(run_dir_: Path | str, title: str | None = None) -> Image.Image:
    """Rank histogram (assimilation + validation panels), titles dropped.

    ``panel_titles=False`` keeps the crop identical across runs: the raw figure
    puts a "Posterior" column title above the first axes in some runs only.
    """
    path = _as_dir(run_dir_) / "rank_histogram.png"
    n = len(panel_bands(path, panel_titles=False))
    im = trim(_stack_panels(path, range(n), panel_titles=False))
    return label(im, title) if title else im


def crop_slices(run_dir_: Path | str, title: str | None = None) -> Image.Image | None:
    """Time-mean horizontal slices, as-is with the white margins trimmed.

    ``None`` when the run has no ``mean_slices.png`` -- the PALM-truth runs do
    not write one (no matching truth field to difference against).
    """
    path = _as_dir(run_dir_) / "mean_slices.png"
    if not path.exists():
        return None
    im = trim(_open(path))
    return label(im, title) if title else im


def crop_param_evolution(
    run_dir_: Path | str, title: str | None = None
) -> Image.Image | None:
    """Parameter trajectories (alpha and |U|); ``None`` if the run has none.

    Only the filtering campaign writes ``parameter_evolution.png``; the trailing
    state-error panel is dropped.
    """
    path = _as_dir(run_dir_) / "parameter_evolution.png"
    if not path.exists():
        return None
    bands = panel_bands(path)
    keep = range(min(2, len(bands)))
    im = _stack_panels(path, keep)
    return label(im, title) if title else im


# --------------------------------------------------------------------------- #
# composition
# --------------------------------------------------------------------------- #
@lru_cache(maxsize=32)
def _font(size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(str(FONT_PATH), size)


def label(img: Image.Image, text: str, height: int | None = None) -> Image.Image:
    """Prepend a bold label strip carrying ``text`` (e.g. "E6 . inflow_turb . loc, obs15")."""
    h = height or max(30, round(img.width / 26))
    font = _font(max(12, int(h * 0.58)))
    out = Image.new("RGB", (img.width, img.height + h), WHITE)
    d = ImageDraw.Draw(out)
    d.rectangle([0, 0, img.width, h - 5], fill=STRIP_BG)
    d.text((round(h * 0.3), round(h * 0.16)), text, fill=STRIP_FG, font=font)
    out.paste(img, (0, h))
    return out


def _labelled(
    imgs: Sequence[Image.Image], labels: Sequence[str] | None
) -> list[Image.Image]:
    if labels is None:
        return list(imgs)
    if len(labels) != len(imgs):
        raise ValueError(f"{len(labels)} labels for {len(imgs)} images")
    # one strip height for the whole row/column so the labels line up
    h = max(30, round(max(i.width for i in imgs) / 26))
    return [label(im, t, height=h) for im, t in zip(imgs, labels)]


def hstack(
    imgs: Sequence[Image.Image], labels: Sequence[str] | None = None, gap: int = 20
) -> Image.Image:
    """Side by side, bottom-padded to a common height (no distortion)."""
    imgs = _labelled(imgs, labels)
    h = max(i.height for i in imgs)
    w = sum(i.width for i in imgs) + gap * (len(imgs) - 1)
    out = Image.new("RGB", (w, h), WHITE)
    x = 0
    for i in imgs:
        out.paste(i, (x, 0))
        x += i.width + gap
    return out


def vstack(
    imgs: Sequence[Image.Image], labels: Sequence[str] | None = None, gap: int = 20
) -> Image.Image:
    """Stacked, scaled to a common width so shared x-axes stay aligned."""
    imgs = _labelled(imgs, labels)
    w = max(i.width for i in imgs)
    imgs = [
        (
            i
            if i.width == w
            else i.resize((w, round(i.height * w / i.width)), Image.Resampling.LANCZOS)
        )
        for i in imgs
    ]
    h = sum(i.height for i in imgs) + gap * (len(imgs) - 1)
    out = Image.new("RGB", (w, h), WHITE)
    y = 0
    for i in imgs:
        out.paste(i, (0, y))
        y += i.height + gap
    return out


def grid(
    imgs: Sequence[Image.Image],
    ncols: int,
    labels: Sequence[str] | None = None,
    gap: int = 20,
) -> Image.Image:
    imgs = _labelled(imgs, labels)
    rows = [hstack(imgs[i : i + ncols], gap=gap) for i in range(0, len(imgs), ncols)]
    return vstack(rows, gap=gap)


def trim(img: Image.Image, tol: int = 250, pad: int = 6) -> Image.Image:
    """Remove uniform near-white margins, keeping a small pad."""
    arr = np.asarray(img.convert("L"))
    h, w = arr.shape
    rows = np.where(~(arr >= tol).all(axis=1))[0]
    cols = np.where(~(arr >= tol).all(axis=0))[0]
    if rows.size == 0 or cols.size == 0:
        return img
    box = (
        max(0, int(cols[0]) - pad),
        max(0, int(rows[0]) - pad),
        min(w, int(cols[-1]) + pad + 1),
        min(h, int(rows[-1]) + pad + 1),
    )
    return img.crop(box)


def save(img: Image.Image, path: Path | str, max_width: int = 1600) -> Path:
    """Downscale to ``max_width`` and write a PNG (parent dirs created)."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if img.width > max_width:
        img = img.resize(
            (max_width, round(img.height * max_width / img.width)), Image.Resampling.LANCZOS
        )
    img.save(p, optimize=True)
    return p


# --------------------------------------------------------------------------- #
if __name__ == "__main__":  # `python figlib.py` prints the ID tables
    for m in METHODS:
        print(f"\n{m}")
        for k, v in run_ids(m).items():
            print(f"  {k:<4} {v}")
