"""WP1.5: the five evaluation figures (metrics doc §7, figure IDs P1/S1/S5/F1/D1).

A figure test that only asserts "matplotlib drew pixels" is worthless -- the
renderer is not what breaks. What breaks is the *content*: a rank histogram
whose uniform reference is a flat line across unequal bins, a field panel whose
colour scale was set by a solid cell holding a sentinel, an axis rescaled into
uselessness by one ``inf``, a figure that raises on the 2-member CI shape
instead of degrading. So what is pinned here is:

  * **every figure writes a real PNG and returns its path** -- the WP1.5 return
    contract, which differs from the WP0.2 legacy plots' unconditional ``None``
    (``..._writes_a_png_and_returns_its_path``);
  * **master-plan invariant 3**: absent, empty and degenerate inputs return
    ``None`` and log, never raise -- an old run dir, a flag-off run and the
    smoke shape all reach these functions
    (``..._no_ops_...``, ``..._survives_the_smoke_shape``);
  * **D1's coarsening is the figure**. ``M+1`` rank bins are grouped with
    ``np.array_split``, which gives *unequal* group widths whenever ``M+1`` is
    not divisible by the bin count, so the uniform reference is per-group
    (``expected_g = N·size_g/(M+1)``) and a flat line at ``N/n_bins`` is simply
    wrong (``..._coarsens_with_array_split_semantics``,
    ``..._reference_is_proportional_to_the_group_width``);
  * **non-finite values never set a limit**: one ``inf`` member, one ``inf``
    cell (``..._axis_limits_stay_finite_...``);
  * **F1 masks the solid cells before it scales the colours** -- they are never
    scored and must never be colour-scaled either
    (``..._solid_cells_are_outside_the_shared_norm``);
  * **identity, not only content**: which row, column, station, sensor or knot
    a drawn value belongs to. A figure whose numbers are right but attached to
    the wrong panel renders exactly like a correct one, so the sign of F1's
    difference, its row labels against the levels they draw, S1's captions
    against their own stations, S5's per-row truth and per-panel surviving
    member count, and P1's two marginals against the knots their tick labels
    claim are each pinned to their own panel rather than to the figure;
  * the two schema-driven degradations a reader would not guess: F1 drops the
    prior column when the prior state was not saved
    (``..._drops_the_prior_column_...``) and S1 drops colocation's
    extrapolated top level when ``extrapolated_edges`` names the vertical axis
    (``..._excludes_the_extrapolated_top_level``).

Inputs are built by hand rather than by running a metric stage: the schema of
``eval_fields.nc`` is fixed by ``scripts/esmda/compute_esmda_metrics.py``
(``_FIELD_DIMS``, ``_eval_fields_dataset``, ``STATION_QUANTILES``) and
fabricating it directly keeps this file a unit suite -- ``run_esmda`` e2e
coverage of the writer lives in ``test_evaluation_mean_fields.py``.

Figure internals are read back through a ``Figure.savefig`` spy
(``captured_figures``) rather than through a helper the implementation would
have to expose: it needs no change to the five frozen signatures, and a
``plt.close``d figure keeps its artists, so the bars, reference lines and
colour norms are all still readable after the function returns.

Specification: ``docs/plans/esmda_evaluation/phase1_metrics_and_figures.md``
(WP1.5); conventions: ``docs/plans/esmda_turbulence_evaluation.md`` §7.
"""

from __future__ import annotations

import logging
import pathlib
import re
from collections.abc import Iterator, Sequence

import numpy as np
import pytest
import xarray

# ``_coarsen_rank_counts`` is private, and imported anyway: it is pure
# arithmetic (no logging, no no-op -- the guards live in its caller), and D1's
# binomial band is drawn as a ``fill_between`` whose y-data cannot be read back
# off the figure, so the formula is pinned on the helper directly. Everything
# else is asserted through the rendered artists.
from evaluation.figures import (
    _coarsen_rank_counts,
    plot_mean_slices,
    plot_parameter_marginals,
    plot_rank_histogram,
    plot_sensor_fans,
    plot_station_profiles,
)
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle

# The schema constants ``eval_fields.nc`` is written with. Duplicated from
# ``scripts/esmda/compute_esmda_metrics.py`` on purpose: ``libs/evaluation`` is
# a leaf (invariant 5) and a test of it must not import a script package to
# learn the file format it reads.
_COMPONENTS = ("u", "v", "w")
_QUANTILES = (0.05, 0.25, 0.5, 0.75, 0.95)


# ---------------------------------------------------------------------------
# Fixtures and artifact readers
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)  # type: ignore[misc]
def _close_figures() -> Iterator[None]:
    """No test leaks a figure into the next one (or into the suite's memory)."""
    import matplotlib.pyplot as plt

    yield
    plt.close("all")


@pytest.fixture  # type: ignore[misc]
def captured_figures(monkeypatch: pytest.MonkeyPatch) -> list[Figure]:
    """Every ``Figure`` a plot function saves, in save order.

    ``style.save_png`` closes the figure it wrote, but closing only drops it
    from pyplot's registry -- the axes, patches and lines survive, so this is
    enough to read a figure's content back without the implementation growing
    an accessor for the tests' benefit.
    """
    saved: list[Figure] = []
    original = Figure.savefig

    def _spy(self: Figure, *args: object, **kwargs: object) -> object:
        saved.append(self)
        return original(self, *args, **kwargs)

    monkeypatch.setattr(Figure, "savefig", _spy)
    return saved


def _assert_png(returned: object, expected: pathlib.Path) -> None:
    """The WP1.5 return contract: the written path comes back, and it is a PNG."""
    assert returned is not None, "the figure no-oped on inputs that are not degenerate"
    assert pathlib.Path(returned) == expected  # type: ignore[arg-type]
    assert expected.is_file()
    payload = expected.read_bytes()
    assert payload[:8] == b"\x89PNG\r\n\x1a\n"
    assert len(payload) > 1024, "a PNG this small is an empty canvas"


def _assert_no_op(returned: object, path: pathlib.Path) -> None:
    """Invariant 3: degenerate input returns ``None`` and leaves no half-figure."""
    assert returned is None
    assert not path.exists()


def _rectangles(ax: object) -> list[Rectangle]:
    return [p for p in ax.patches if isinstance(p, Rectangle)]  # type: ignore[attr-defined]


def _bar_axes(fig: Figure) -> list[object]:
    """The panels that actually carry bars (D1 draws counts as bars)."""
    return [ax for ax in fig.axes if _rectangles(ax)]


def _bar_heights(ax: object) -> list[float]:
    return [float(p.get_height()) for p in _rectangles(ax)]


def _line_ydata(ax: object) -> list[np.ndarray]:
    """Candidate reference-line y-data: ``plot``/``step`` lines and ``hlines``.

    Both are plausible ways to draw D1's per-group expected level, and which
    one the figure uses is a rendering detail the test should not dictate --
    so read the y values out of either.
    """
    candidates = []
    for line in ax.lines:  # type: ignore[attr-defined]
        y = np.asarray(line.get_ydata(), dtype=float)
        if y.size:
            candidates.append(y)
    for collection in ax.collections:  # type: ignore[attr-defined]
        if not hasattr(collection, "get_segments"):
            continue
        segments = collection.get_segments()
        if segments:
            candidates.append(np.concatenate([np.asarray(s)[:, 1] for s in segments]))
    return candidates


def _filled_bands(ax: Axes, orient: str = "vertical") -> list[tuple[np.ndarray, ...]]:
    """``(coord, lo, hi)`` of every ``fill_between`` band drawn on ``ax``.

    ``style.nested_bands`` draws each quantile pair as a filled polygon, and a
    polygon is the only place the *pairing* is recorded -- the numbers never
    reach a line artist. ``fill_between`` closes the region by walking one edge
    forward and the other back, so every vertex sits on one of the two edges and
    the two y values present at a given coordinate ARE ``(lo, hi)``.

    ``orient="horizontal"`` reads a ``fill_betweenx`` band (S1's profiles), whose
    coordinate axis is y and whose edges are x.
    """
    from matplotlib.collections import PolyCollection

    bands: list[tuple[np.ndarray, ...]] = []
    c_axis, v_axis = (0, 1) if orient == "vertical" else (1, 0)
    for collection in ax.collections:
        if not isinstance(collection, PolyCollection):
            continue
        paths = collection.get_paths()
        if not paths:
            continue
        vertices = np.concatenate([np.asarray(p.vertices, float) for p in paths])
        coord = np.unique(np.round(vertices[:, c_axis], 9))
        at = [vertices[np.isclose(vertices[:, c_axis], c), v_axis] for c in coord]
        bands.append(
            (coord, np.array([v.min() for v in at]), np.array([v.max() for v in at]))
        )
    return bands


def _mappables(fig: Figure) -> list[object]:
    """Everything on the figure that carries a colour norm."""
    found: list[object] = []
    for ax in fig.axes:
        found.extend(ax.images)
        found.extend(c for c in ax.collections if hasattr(c, "get_clim"))
    return found


def _clims(fig: Figure) -> list[tuple[float, float]]:
    """Colour limits of every mappable that has any.

    An unscaled collection (matplotlib puts a few on a figure -- the legend
    proxies for the ``set_bad`` colour, for one) reports ``(None, None)``; it
    carries no norm, so it is not a norm to check.
    """
    limits = []
    for mappable in _mappables(fig):
        low, high = mappable.get_clim()  # type: ignore[attr-defined]
        if low is None or high is None:
            continue
        limits.append((float(low), float(high)))
    return limits


def _titles(ax: object) -> list[str]:
    """All three title slots -- ``get_title()`` alone only returns the centred one."""
    return [ax.get_title(loc=loc) for loc in ("left", "center", "right")]  # type: ignore[attr-defined]


def _axes_text(fig: Figure) -> str:
    """Titles, axis labels and panel annotations, lower-cased and joined."""
    parts: list[str] = [t.get_text() for t in fig.texts]
    for ax in fig.axes:
        parts.extend(_titles(ax))
        parts.append(ax.get_xlabel())
        parts.append(ax.get_ylabel())
        parts.extend(t.get_text() for t in ax.texts)
    return " | ".join(p for p in parts if p).lower()


def _axes_titles(fig: Figure) -> str:
    return " | ".join(t.lower() for ax in fig.axes for t in _titles(ax) if t)


def _panel_axes(fig: Figure) -> list[object]:
    """Axes holding a mappable -- the image grid, plus any colourbar axes."""
    return [ax for ax in fig.axes if ax.images or ax.collections]


def _tick_labels(fig: Figure) -> str:
    """Every x tick label on the figure, lower-cased and joined.

    P1 names its two marginals in the x tick labels, and the *knot* each one
    comes from is part of that name.
    """
    return " | ".join(
        t.get_text().lower()
        for ax in fig.axes
        for t in ax.get_xticklabels()
        if t.get_text()
    )


def _labelled_lines(fig: Figure, label: str) -> list[object]:
    """Every ``Line2D`` the figure drew under ``label`` (the legend label)."""
    return [
        line
        for ax in fig.axes
        for line in ax.lines
        if str(line.get_label()).lower() == label.lower()
    ]


def _z_annotations(fig: Figure) -> list[str]:
    """P1's per-panel ``z`` annotations, lower-cased.

    The z annotation is the only panel text that starts with a bare ``z``, which
    is what separates it from titles and axis labels without the test having to
    know how the figure lays its panels out.
    """
    found = []
    for ax in fig.axes:
        for text in ax.texts:
            body = text.get_text().strip().lower()
            if body.startswith("z") and not body.startswith("z/"):
                found.append(body)
    return found


def _annotated_z(fig: Figure) -> float:
    """The single numeric z P1 annotated, as a float."""
    annotations = _z_annotations(fig)
    assert len(annotations) == 1, f"expected one z annotation, got {annotations}"
    match = re.search(r"(-?\d+(?:\.\d+)?)", annotations[0])
    assert match is not None, f"no number in the z annotation {annotations[0]!r}"
    return float(match.group(1))


def _image_of(fig: Figure, title: str) -> np.ndarray:
    """The image data of the panel whose title contains ``title``, NaN-filled.

    ``imshow`` masks the non-finite entries it is handed, so the mask is put
    back as NaN -- masked and NaN are the same statement here ("this cell has no
    value"), and only one of them is comparable with ``np.isnan``.
    """
    axes = [ax for ax in fig.axes if title.lower() in " | ".join(_titles(ax)).lower()]
    assert len(axes) == 1, f"expected one {title!r} panel, found {len(axes)}"
    images = axes[0].images
    assert images, f"the {title!r} panel carries no image"
    return np.ma.filled(images[0].get_array().astype(float), np.nan)


def _station_panel_titles(fig: Figure) -> list[str]:
    """S1's column titles -- the ones naming a station, in draw order."""
    return [t for ax in fig.axes for t in _titles(ax) if t and "#" in t]


# The readers below all answer the same question -- *which* row, column, sensor
# or knot does this artist belong to? -- because that is the class of error the
# figure-wide readers above cannot see: a figure whose content is right but
# attached to the wrong panel renders exactly like a correct one.


def _panel_text(ax: Axes) -> str:
    """One panel's own annotations, lower-cased and joined."""
    return " | ".join(t.get_text() for t in ax.texts if t.get_text()).lower()


def _image_array(ax: Axes) -> np.ndarray:
    """The panel's image data, NaN-filled (see :func:`_image_of`)."""
    assert ax.images, "the panel carries no image"
    return np.ma.filled(ax.images[0].get_array().astype(float), np.nan)


def _row_label_axes(fig: Figure) -> list[Axes]:
    """F1's leftmost column in row order -- the panels carrying the ``z`` label."""
    return [ax for ax in fig.axes if ax.images and ax.get_ylabel().startswith("z =")]


def _labelled_z(ax: Axes) -> float:
    """The z height F1 printed on a row, as a float."""
    match = re.search(r"z = (-?\d+(?:\.\d+)?)", ax.get_ylabel())
    assert match is not None, f"no z in the row label {ax.get_ylabel()!r}"
    return float(match.group(1))


def _sensor_row(fig: Figure, row: int) -> Axes:
    """S5's panel for sensor ``row`` (single-column figures)."""
    axes = [ax for ax in fig.axes if ax.get_ylabel().startswith(f"sensor {row}\n")]
    assert len(axes) == 1, f"expected one panel for sensor {row}, found {len(axes)}"
    return axes[0]


def _vertical_line_positions(ax: Axes) -> list[float]:
    """The x of every full-height vertical rule on the panel (``axvline``)."""
    found = []
    for line in ax.lines:
        x = np.asarray(line.get_xdata(), dtype=float)
        y = np.asarray(line.get_ydata(), dtype=float)
        if x.size == 2 and y.size == 2 and x[0] == x[1] and y.tolist() == [0.0, 1.0]:
            found.append(float(x[0]))
    return sorted(found)


def _tick_labels_by_position(ax: Axes) -> dict[float, str]:
    """The panel's x tick labels keyed by the tick they are attached to.

    P1's two marginals come from different knots, so *which* label sits on which
    tick is the statement; a figure-wide substring search cannot see them swap.
    """
    return {
        float(position): label.get_text().lower()
        for position, label in zip(ax.get_xticks(), ax.get_xticklabels())
        if label.get_text()
    }


# ---------------------------------------------------------------------------
# Synthetic ``eval_fields.nc`` (S1, F1)
# ---------------------------------------------------------------------------


def _eval_fields(
    *,
    n_zlev: int = 2,
    ny: int = 4,
    nx: int = 5,
    n_z: int = 5,
    station_sets: Sequence[str] = ("assimilation", "validation"),
    with_prior: bool = True,
    slabs: bool = True,
    stations: bool = True,
    fluid: np.ndarray | None = None,
    slab_mean: dict[str, np.ndarray] | None = None,
    extrapolated: str = "",
    seed: int = 0,
) -> xarray.Dataset:
    """An ``eval_fields.nc`` of the shape WP1.4's metric stage writes.

    Same variable names, dims, coords and attrs as
    ``compute_esmda_metrics._eval_fields_dataset``: ``{source}_{slab,station}_
    {mean,tke,uw}`` with ``_spread`` on the ensemble sources and ``_quantile``
    at the station columns only, an int8 ``slab_fluid`` mask, and the
    ``horizontal_stride`` / ``extrapolated_edges`` / ``t_start`` / ``t_end``
    attributes a figure has to read to draw an honest plot.
    """
    rng = np.random.default_rng(seed)
    n_stations = len(station_sets)
    sources = ["truth", "posterior"] + (["prior"] if with_prior else [])
    slab_shape = (len(_COMPONENTS), n_zlev, ny, nx)
    station_shape = (len(_COMPONENTS), n_z, n_stations)
    quantile_offsets = (np.asarray(_QUANTILES) - 0.5).reshape(-1, 1, 1, 1)

    variables: dict[str, tuple] = {}
    for source in sources:
        ensemble = source != "truth"
        if slabs:
            mean = (
                slab_mean[source]
                if slab_mean is not None and source in slab_mean
                else rng.normal(size=slab_shape) + 3.0
            )
            variables[f"{source}_slab_mean"] = (
                ("component", "zlev", "y", "x"),
                np.asarray(mean, dtype=np.float32),
            )
            for quantity in ("tke", "uw"):
                variables[f"{source}_slab_{quantity}"] = (
                    ("zlev", "y", "x"),
                    rng.random((n_zlev, ny, nx)).astype(np.float32),
                )
            if ensemble:
                variables[f"{source}_slab_mean_spread"] = (
                    ("component", "zlev", "y", "x"),
                    rng.random(slab_shape).astype(np.float32),
                )
                for quantity in ("tke", "uw"):
                    variables[f"{source}_slab_{quantity}_spread"] = (
                        ("zlev", "y", "x"),
                        rng.random((n_zlev, ny, nx)).astype(np.float32),
                    )
        if stations:
            profile = np.linspace(1.0, 4.0, n_z).reshape(1, -1, 1)
            column_mean = (profile + rng.normal(scale=0.1, size=station_shape)).astype(
                np.float32
            )
            column_spread = rng.random(station_shape).astype(np.float32) * 0.2 + 0.05
            variables[f"{source}_station_mean"] = (
                ("component", "z", "station"),
                column_mean,
            )
            for quantity in ("tke", "uw"):
                variables[f"{source}_station_{quantity}"] = (
                    ("z", "station"),
                    rng.random((n_z, n_stations)).astype(np.float32),
                )
            if ensemble:
                variables[f"{source}_station_mean_spread"] = (
                    ("component", "z", "station"),
                    column_spread,
                )
                variables[f"{source}_station_mean_quantile"] = (
                    ("quantile", "component", "z", "station"),
                    (column_mean + 2.0 * quantile_offsets * column_spread).astype(
                        np.float32
                    ),
                )
                for quantity in ("tke", "uw"):
                    base = rng.random((n_z, n_stations)).astype(np.float32)
                    variables[f"{source}_station_{quantity}_spread"] = (
                        ("z", "station"),
                        base * 0.1,
                    )
                    variables[f"{source}_station_{quantity}_quantile"] = (
                        ("quantile", "z", "station"),
                        (base + quantile_offsets.reshape(-1, 1, 1) * 0.2 * base).astype(
                            np.float32
                        ),
                    )

    if slabs and fluid is not None:
        variables["slab_fluid"] = (("zlev", "y", "x"), np.asarray(fluid, dtype=np.int8))

    coords: dict[str, object] = {"component": list(_COMPONENTS)}
    if slabs:
        coords["zlev"] = np.arange(n_zlev, dtype=float) * 2.0 + 1.0
        coords["y"] = np.arange(ny, dtype=float)
        coords["x"] = np.arange(nx, dtype=float)
    if stations:
        coords["z"] = np.arange(1, n_z + 1, dtype=float)
        coords["station"] = np.arange(n_stations)
        coords["station_x"] = ("station", np.arange(n_stations, dtype=float) * 5.0)
        coords["station_y"] = ("station", np.arange(n_stations, dtype=float) * 2.0)
        coords["station_set"] = ("station", list(station_sets))
        coords["quantile"] = list(_QUANTILES)

    return xarray.Dataset(
        variables,
        coords=coords,
        attrs={
            "description": "synthetic eval_fields for the WP1.5 figure tests",
            "horizontal_stride": 1,
            "extrapolated_edges": extrapolated,
            "t_start": 10.0,
            "t_end": 40.0,
        },
    )


# ---------------------------------------------------------------------------
# Synthetic parameter datasets (P1) and sensor series (S5)
# ---------------------------------------------------------------------------


def _param_dataset(
    times: Sequence[float] = (0.0,), **members: Sequence[Sequence[float]]
) -> xarray.Dataset:
    n_members = len(next(iter(members.values())))
    return xarray.Dataset(
        {
            name: (("ensemble", "time"), np.asarray(rows))
            for name, rows in members.items()
        },
        coords={"ensemble": np.arange(n_members), "time": list(times)},
    )


def _truth_dataset(
    times: Sequence[float] = (0.0,), **values: Sequence[float]
) -> xarray.Dataset:
    return xarray.Dataset(
        {name: (("time",), np.asarray(series)) for name, series in values.items()},
        coords={"time": list(times)},
    )


def _toy_params(
    n_members: int = 6, seed: int = 0
) -> tuple[xarray.Dataset, xarray.Dataset, xarray.Dataset]:
    """A contracted posterior around a slightly offset truth, plus a wide prior."""
    rng = np.random.default_rng(seed)
    posterior = _param_dataset(
        inflow_angle=(270.0 + rng.normal(scale=1.0, size=(n_members, 1))).tolist(),
        velocity_magnitude=(5.0 + rng.normal(scale=0.2, size=(n_members, 1))).tolist(),
    )
    prior = _param_dataset(
        inflow_angle=(270.0 + rng.normal(scale=8.0, size=(n_members, 1))).tolist(),
        velocity_magnitude=(5.0 + rng.normal(scale=1.5, size=(n_members, 1))).tolist(),
    )
    truth = _truth_dataset(inflow_angle=[271.0], velocity_magnitude=[5.1])
    return posterior, prior, truth


def _sensor_series(
    *,
    n_members: int = 5,
    n_times: int = 16,
    n_sensors: int = 2,
    sets: Sequence[str] = ("assimilation", "validation"),
    seed: int = 0,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """``sensor_magnitude``-shaped series: truth ``(T, S)``, ensemble ``(M, T, S)``."""
    rng = np.random.default_rng(seed)
    truth = {
        name: 4.0 + rng.normal(scale=0.3, size=(n_times, n_sensors)) for name in sets
    }
    ensemble = {
        name: truth[name][None]
        + rng.normal(scale=0.4, size=(n_members, n_times, n_sensors))
        for name in sets
    }
    return truth, ensemble


def _rank_counts(
    counts: Sequence[int],
    *,
    sets: Sequence[str] = ("assimilation",),
    halves: Sequence[str] = ("prior", "posterior"),
    statistics: Sequence[str] = ("mean_u",),
) -> dict[str, dict[str, dict[str, list[int]]]]:
    """``run_summary.yaml``'s ``sensor_statistics`` rank counts, as D1 takes them."""
    return {
        s: {h: {stat: list(counts) for stat in statistics} for h in halves}
        for s in sets
    }


# ---------------------------------------------------------------------------
# P1 -- plot_parameter_marginals
# ---------------------------------------------------------------------------


def test_parameter_marginals_writes_a_png_and_returns_its_path(
    tmp_path: pathlib.Path,
) -> None:
    posterior, prior, truth = _toy_params()
    out = tmp_path / "parameter_marginals.png"

    result = plot_parameter_marginals(posterior, truth, out, prior_params=prior)

    _assert_png(result, out)


def test_parameter_marginals_y_limits_include_the_prior(
    tmp_path: pathlib.Path, captured_figures: list[Figure]
) -> None:
    # The whole point of drawing the prior beside the posterior is that the
    # *contraction* is visible; a y-limit set from the posterior alone shows a
    # wide blob next to a clipped one and reads as no contraction at all.
    posterior = _param_dataset(inflow_angle=[[269.9], [270.0], [270.1], [270.2]])
    prior = _param_dataset(inflow_angle=[[250.0], [265.0], [275.0], [290.0]])
    truth = _truth_dataset(inflow_angle=[270.0])

    plot_parameter_marginals(posterior, truth, tmp_path / "p1.png", prior_params=prior)

    assert captured_figures, "no figure was saved"
    fig = captured_figures[-1]
    covered = any(
        ax.get_ylim()[0] <= 250.0 and ax.get_ylim()[1] >= 290.0 for ax in fig.axes
    )
    assert (
        covered
    ), f"prior range not inside any panel's y-limits: {[a.get_ylim() for a in fig.axes]}"


def test_parameter_marginals_annotate_the_z_score(
    tmp_path: pathlib.Path, captured_figures: list[Figure]
) -> None:
    posterior, prior, truth = _toy_params()

    plot_parameter_marginals(posterior, truth, tmp_path / "p1.png", prior_params=prior)

    assert "z" in _axes_text(captured_figures[-1])


def test_parameter_marginals_report_a_collapsed_z_score_as_not_available(
    tmp_path: pathlib.Path, captured_figures: list[Figure]
) -> None:
    # sigma_a = 0: the z-score is +-inf, and printing "inf" in a panel is the
    # failure the contract calls out. It has to read ``n/a``.
    posterior = _param_dataset(inflow_angle=[[270.0], [270.0], [270.0], [270.0]])
    truth = _truth_dataset(inflow_angle=[271.0])

    result = plot_parameter_marginals(posterior, truth, tmp_path / "p1.png")

    assert result is not None, "a collapsed posterior is still a figure worth drawing"
    text = _axes_text(captured_figures[-1])
    assert "n/a" in text
    assert "inf" not in text.replace("inflow", "")


def test_parameter_marginals_axis_limits_stay_finite_with_an_inf_member(
    tmp_path: pathlib.Path, captured_figures: list[Figure]
) -> None:
    # One diverged member carrying ``inf`` must not rescale the panel: the axis
    # limits are computed over the finite values only (phase1 plan, WP1.2).
    posterior = _param_dataset(
        inflow_angle=[[269.0], [270.0], [271.0], [np.inf]],
    )
    truth = _truth_dataset(inflow_angle=[270.0])

    plot_parameter_marginals(posterior, truth, tmp_path / "p1.png")

    assert captured_figures, "an inf member made the figure no-op entirely"
    for ax in captured_figures[-1].axes:
        assert np.all(
            np.isfinite(ax.get_ylim())
        ), f"non-finite y-limits: {ax.get_ylim()}"
        assert np.all(
            np.isfinite(ax.get_xlim())
        ), f"non-finite x-limits: {ax.get_xlim()}"


def test_parameter_marginals_plot_the_final_knot_of_a_dynamic_parameter(
    tmp_path: pathlib.Path, captured_figures: list[Figure]
) -> None:
    # ``K > 1`` is a *dynamic* parameter; the contract picks the final knot to
    # match the ``final`` convention used everywhere in ``run_summary.yaml``.
    # Knot 0 sits two orders of magnitude away, so a figure that drew K violins
    # (or the first knot) cannot keep its y-limits near the final one.
    posterior = _param_dataset(
        times=(0.0, 1.0),
        velocity_magnitude=[[500.0, 5.0], [500.0, 5.2], [500.0, 4.8], [500.0, 5.1]],
    )
    truth = _truth_dataset(times=(0.0, 1.0), velocity_magnitude=[500.0, 5.0])

    plot_parameter_marginals(posterior, truth, tmp_path / "p1.png")

    fig = captured_figures[-1]
    assert all(ax.get_ylim()[1] < 100.0 for ax in fig.axes), (
        "the panel spans knot 0 as well -- the final knot is not what was drawn: "
        f"{[ax.get_ylim() for ax in fig.axes]}"
    )


def test_parameter_marginals_annotate_the_z_of_the_knot_they_draw(
    tmp_path: pathlib.Path, captured_figures: list[Figure]
) -> None:
    # The annotated z must belong to the knot the panel DRAWS. Knot 0 here sits
    # ~390 posterior sigmas off its own truth while the final knot is centred on
    # its own, so an annotation lifted from ``z_score[0]`` prints "z = 386" over
    # a picture of a perfectly calibrated posterior -- and a reader believes the
    # number, not the violin.
    from evaluation.scores import parameter_bundle

    members = np.array(
        [[0.0, 4.0], [1.0, 5.0], [2.0, 6.0], [3.0, 7.0]]
    )  # (member, knot)
    truth_values = np.array([500.0, 5.5])
    posterior = _param_dataset(times=(0.0, 1.0), velocity_magnitude=members.tolist())
    truth = _truth_dataset(times=(0.0, 1.0), velocity_magnitude=truth_values.tolist())
    z_score = parameter_bundle(members, None, truth_values)["z_score"]

    plot_parameter_marginals(posterior, truth, tmp_path / "p1.png")

    assert abs(float(z_score[0])) > 1.0, "the fixture's two knots score the same"
    assert _annotated_z(captured_figures[-1]) == pytest.approx(
        float(z_score[-1]), abs=0.005
    ), "the annotated z is not the drawn knot's"


def test_parameter_marginals_draw_the_run_prior_when_the_truth_is_static(
    tmp_path: pathlib.Path, captured_figures: list[Figure]
) -> None:
    # ``prior_params.nc`` stacks EVERY window's prior along ``time``, so for a
    # static parameter over W windows knot ``w > 0`` is window ``w-1``'s
    # POSTERIOR. Taking the marginal at the final knot therefore draws an
    # already-updated ensemble as "the prior", and the panel reads as almost no
    # contraction. The truth being constant is what identifies this case.
    posterior = _param_dataset(
        times=(0.0, 1.0),
        velocity_magnitude=[[5.0, 4.9], [5.0, 5.1], [5.0, 5.0], [5.0, 5.05]],
    )
    prior = _param_dataset(
        times=(0.0, 1.0),
        velocity_magnitude=[[-20.0, 4.8], [30.0, 5.2], [-5.0, 5.0], [15.0, 5.1]],
    )
    truth = _truth_dataset(times=(0.0, 1.0), velocity_magnitude=[5.0, 5.0])

    plot_parameter_marginals(posterior, truth, tmp_path / "p1.png", prior_params=prior)

    fig = captured_figures[-1]
    covered = any(
        ax.get_ylim()[0] <= -20.0 and ax.get_ylim()[1] >= 30.0 for ax in fig.axes
    )
    assert covered, (
        "the panel does not span the run's actual (knot 0) prior -- a later "
        f"window's prior was drawn instead: {[a.get_ylim() for a in fig.axes]}"
    )
    # ...and the reader is told which knot each marginal came from, because the
    # two branches below pick different ones.
    assert "knot 0" in _tick_labels(fig)


def test_parameter_marginals_keep_the_same_knot_pair_when_the_truth_varies(
    tmp_path: pathlib.Path, captured_figures: list[Figure]
) -> None:
    # A truth that varies across knots is a genuinely time-varying parameter:
    # knot 0 is a different physical time (500 m/s vs 5), so a knot-0 prior would
    # put two different quantities in one panel. Per-window contraction is all
    # there is here -- it just has to be labelled as such.
    posterior = _param_dataset(
        times=(0.0, 1.0),
        velocity_magnitude=[[500.0, 5.0], [500.0, 5.2], [500.0, 4.8], [500.0, 5.1]],
    )
    prior = _param_dataset(
        times=(0.0, 1.0),
        velocity_magnitude=[[400.0, 3.0], [600.0, 7.0], [450.0, 4.0], [550.0, 6.0]],
    )
    truth = _truth_dataset(times=(0.0, 1.0), velocity_magnitude=[500.0, 5.0])

    plot_parameter_marginals(posterior, truth, tmp_path / "p1.png", prior_params=prior)

    fig = captured_figures[-1]
    assert all(ax.get_ylim()[1] < 100.0 for ax in fig.axes), (
        "a knot-0 marginal reached the panel, which spans a different physical "
        f"time: {[a.get_ylim() for a in fig.axes]}"
    )
    assert "final" in _tick_labels(fig)


def _static_knot_pair() -> tuple[xarray.Dataset, xarray.Dataset, xarray.Dataset]:
    """A static parameter over two knots: knot 0's prior is nowhere near knot 1's."""
    posterior = _param_dataset(
        times=(0.0, 1.0),
        velocity_magnitude=[[5.0, 4.9], [5.0, 5.1], [5.0, 5.0], [5.0, 5.05]],
    )
    prior = _param_dataset(
        times=(0.0, 1.0),
        velocity_magnitude=[[-20.0, 4.8], [30.0, 5.2], [-5.0, 5.0], [15.0, 5.1]],
    )
    truth = _truth_dataset(times=(0.0, 1.0), velocity_magnitude=[5.0, 5.0])
    return posterior, prior, truth


def test_parameter_marginals_attach_each_knot_label_to_its_own_marginal(
    tmp_path: pathlib.Path, captured_figures: list[Figure]
) -> None:
    # The static panel's two marginals come from DIFFERENT knots -- the prior
    # from knot 0, the posterior from the final one -- so the labels are not
    # decoration: swapped, they attribute the run's entire contraction to the
    # wrong end of it, and the picture is unchanged. Asserting that "0" and
    # "final" appear *somewhere* on the figure cannot see that, so each label is
    # read off the tick it is attached to.
    posterior, prior, truth = _static_knot_pair()

    plot_parameter_marginals(posterior, truth, tmp_path / "p1.png", prior_params=prior)

    fig = captured_figures[-1]
    for ax in fig.axes:
        labels = _tick_labels_by_position(ax)
        assert set(labels) == {0.0, 1.0}, f"unexpected tick positions: {labels}"
        prior_label, post_label = labels[0.0], labels[1.0]
        assert prior_label.startswith("prior") and post_label.startswith(
            "posterior"
        ), f"the marginals are not where their labels say: {labels}"
        assert "0" in prior_label and "final" not in prior_label, (
            "the prior marginal is drawn at knot 0 and must say so, not "
            f"'final': {prior_label!r}"
        )
        assert (
            "final" in post_label
        ), f"the posterior marginal is the final knot's: {post_label!r}"


def test_parameter_marginals_count_the_static_knots_never_windows(
    tmp_path: pathlib.Path, captured_figures: list[Figure]
) -> None:
    # ``K`` is a KNOT count. It equals the window count only on a pure static
    # run: a ``static_parameters`` entry inside a dynamic run (the shipped
    # ``dynamic_sine`` truth has one, and so does the CLI example in
    # ``conf/run_esmda.yaml``) is broadcast onto the dynamic knot grid, where a
    # 2-window run carries four knots. "4 windows" is then a wrong number rather
    # than a loose word, and the tick labels name the wrong axis with it.
    posterior, prior, truth = _static_knot_pair()

    plot_parameter_marginals(posterior, truth, tmp_path / "p1.png", prior_params=prior)

    fig = captured_figures[-1]
    printed = f"{_axes_titles(fig)} | {_tick_labels(fig)}"
    assert "static, 2 knots" in printed, f"the static title miscounts: {printed!r}"
    assert "knot 0" in printed and "final knot" in printed, printed
    assert "window" not in printed, (
        "the panel calls its knots windows -- they are only the same thing on a "
        f"pure static run: {printed!r}"
    )


def test_parameter_marginals_need_two_finite_knots_to_call_a_truth_static(
    tmp_path: pathlib.Path, captured_figures: list[Figure]
) -> None:
    # ``np.allclose`` over a length-1 array is trivially True, so a truth with a
    # single finite knot reads as "constant" and takes the knot-0 prior branch.
    # Here that would draw a 400-600 prior beside a posterior at ~5 in one
    # panel -- the exact "two different quantities in one panel" failure the
    # branch exists to prevent. One finite knot is no evidence of staticness, so
    # the conservative same-knot pair is the answer.
    posterior = _param_dataset(
        times=(0.0, 1.0, 2.0),
        velocity_magnitude=[
            [500.0, 50.0, 5.0],
            [500.0, 50.0, 5.2],
            [500.0, 50.0, 4.8],
            [500.0, 50.0, 5.1],
        ],
    )
    prior = _param_dataset(
        times=(0.0, 1.0, 2.0),
        velocity_magnitude=[
            [400.0, 40.0, 4.0],
            [600.0, 60.0, 6.0],
            [450.0, 45.0, 4.5],
            [550.0, 55.0, 5.5],
        ],
    )
    truth = _truth_dataset(
        times=(0.0, 1.0, 2.0), velocity_magnitude=[np.nan, np.nan, 5.0]
    )

    plot_parameter_marginals(posterior, truth, tmp_path / "p1.png", prior_params=prior)

    fig = captured_figures[-1]
    assert all(ax.get_ylim()[1] < 100.0 for ax in fig.axes), (
        "a knot-0 marginal reached the panel on the strength of one finite "
        f"truth knot: {[a.get_ylim() for a in fig.axes]}"
    )


def test_parameter_marginals_read_an_all_nan_truth_as_no_truth_at_all(
    tmp_path: pathlib.Path, captured_figures: list[Figure]
) -> None:
    # A truth variable that is present but never finite (a parameter the truth
    # run did not carry, or one whose knots did not align) is not a *dynamic*
    # truth -- it is no truth. ``z = n/a`` reads as "the posterior is degenerate"
    # and sends a reader to look at the ensemble; "no truth" sends them to the
    # run dir, which is where the problem is.
    posterior = _param_dataset(
        velocity_magnitude=[[5.0], [5.1], [4.9], [5.05]],
    )
    truth = _truth_dataset(velocity_magnitude=[np.nan])

    result = plot_parameter_marginals(posterior, truth, tmp_path / "p1.png")

    assert result is not None, "an unscoreable truth is not a reason to skip P1"
    annotations = _z_annotations(captured_figures[-1])
    assert annotations and all(
        "no truth" in a for a in annotations
    ), f"an all-NaN truth is reported as something other than a missing one: {annotations}"


def test_parameter_marginals_tell_a_pinned_parameter_from_a_collapsed_one(
    tmp_path: pathlib.Path, captured_figures: list[Figure]
) -> None:
    # Both cases have sigma_a = 0 and so no z, and the two diagnoses are
    # opposite: a *pinned* parameter was never estimated (it had no prior spread
    # either, and nothing is wrong), a *collapsed* one started wide and lost its
    # spread to the update (which is the failure ESMDA actually has). The prior
    # is the only thing that separates them, so it has to reach the diagnosis --
    # a bundle scored without it can only ever say "collapsed".
    pinned = _param_dataset(velocity_magnitude=[[5.0], [5.0], [5.0], [5.0]])
    truth = _truth_dataset(velocity_magnitude=[5.5])

    plot_parameter_marginals(
        pinned, truth, tmp_path / "pinned.png", prior_params=pinned
    )
    pinned_annotations = _z_annotations(captured_figures[-1])

    wide_prior = _param_dataset(velocity_magnitude=[[3.0], [4.0], [6.0], [7.0]])
    plot_parameter_marginals(
        pinned, truth, tmp_path / "collapsed.png", prior_params=wide_prior
    )
    collapsed_annotations = _z_annotations(captured_figures[-1])

    assert pinned_annotations and all(
        "pinned" in a for a in pinned_annotations
    ), f"a pinned parameter is not diagnosed as one: {pinned_annotations}"
    assert collapsed_annotations and all(
        "collapsed" in a for a in collapsed_annotations
    ), collapsed_annotations
    assert set(pinned_annotations).isdisjoint(collapsed_annotations)


def test_parameter_marginals_distinguish_a_missing_truth_from_a_collapsed_sigma(
    tmp_path: pathlib.Path, captured_figures: list[Figure]
) -> None:
    # One ``n/a`` covering both cases makes two opposite diagnoses read the same:
    # "this run saved no truth to score against" and "the posterior collapsed
    # onto a point" call for completely different follow-ups.
    posterior, prior, _truth = _toy_params()
    plot_parameter_marginals(
        posterior, None, tmp_path / "no_truth.png", prior_params=prior
    )
    without_truth = _z_annotations(captured_figures[-1])

    collapsed = _param_dataset(inflow_angle=[[270.0], [270.0], [270.0], [270.0]])
    plot_parameter_marginals(
        collapsed, _truth_dataset(inflow_angle=[271.0]), tmp_path / "collapsed.png"
    )
    collapsed_sigma = _z_annotations(captured_figures[-1])

    assert without_truth and collapsed_sigma
    assert all("no truth" in a for a in without_truth), without_truth
    assert all("n/a" in a for a in collapsed_sigma), collapsed_sigma
    assert set(without_truth).isdisjoint(collapsed_sigma)
    assert not any("inf" in a for a in without_truth + collapsed_sigma)


def test_parameter_marginals_survive_the_smoke_shape(tmp_path: pathlib.Path) -> None:
    # M = 2: a KDE (and therefore a violin) is meaningless, but the CI shape
    # must still produce a figure rather than a traceback.
    posterior = _param_dataset(inflow_angle=[[269.0], [271.0]])
    prior = _param_dataset(inflow_angle=[[260.0], [280.0]])
    truth = _truth_dataset(inflow_angle=[270.0])
    out = tmp_path / "p1.png"

    _assert_png(
        plot_parameter_marginals(posterior, truth, out, prior_params=prior), out
    )


def test_parameter_marginals_render_without_a_prior(tmp_path: pathlib.Path) -> None:
    # ``run.save_prior_state`` off -- there is no prior ensemble to draw.
    posterior, _prior, truth = _toy_params()
    out = tmp_path / "p1.png"

    _assert_png(plot_parameter_marginals(posterior, truth, out), out)


def test_parameter_marginals_render_without_a_truth(
    tmp_path: pathlib.Path, captured_figures: list[Figure]
) -> None:
    # A run against measured data has no true parameters. The marginals are
    # still worth drawing -- prior vs posterior contraction does not need a
    # truth -- so this renders rather than no-ops; only the dashed truth line
    # and the z-score are dropped, the latter annotated with the reason rather
    # than silently omitted.
    posterior, prior, _truth = _toy_params()
    out = tmp_path / "p1.png"

    _assert_png(plot_parameter_marginals(posterior, None, out, prior_params=prior), out)
    annotations = _z_annotations(captured_figures[-1])
    assert annotations and all("no truth" in a for a in annotations), annotations


def test_parameter_marginals_no_op_on_a_dataset_with_no_estimable_parameters(
    tmp_path: pathlib.Path,
) -> None:
    out = tmp_path / "p1.png"

    _assert_no_op(
        plot_parameter_marginals(
            xarray.Dataset(), xarray.Dataset(), out, prior_params=None
        ),
        out,
    )


# ---------------------------------------------------------------------------
# S1 -- plot_station_profiles
# ---------------------------------------------------------------------------


def test_station_profiles_write_a_png_and_return_its_path(
    tmp_path: pathlib.Path,
) -> None:
    fields = _eval_fields()
    out = tmp_path / "station_profiles.png"

    _assert_png(
        plot_station_profiles(fields, out, u_ref=5.0, building_height=10.0), out
    )


def test_station_profiles_render_without_a_reference_velocity(
    tmp_path: pathlib.Path,
) -> None:
    # ``u_ref``/``building_height`` are optional: unnormalised is a valid plot,
    # it just has to say so on the axis.
    fields = _eval_fields()
    out = tmp_path / "s1.png"

    _assert_png(plot_station_profiles(fields, out), out)


def _one_row_fields(quantity: str, values: np.ndarray, n_z: int) -> xarray.Dataset:
    """An ``eval_fields`` carrying exactly one profile row, with a known truth.

    Dropping the other quantity leaves ``plot_station_profiles`` a single row,
    so the panel under test is identified by the dataset rather than by matching
    an axis label -- the labels are prose and a reworded one must not decide
    which numbers a test reads.
    """
    fields = _eval_fields(
        n_z=n_z, station_sets=("assimilation",), with_prior=False, slabs=False
    )
    other = "tke" if quantity == "mean" else "mean"
    fields = fields.drop_vars(
        [
            v
            for v in map(str, fields.data_vars)
            if f"_station_{other}" in v or "_station_uw" in v
        ]
    )
    dims = ("component", "z", "station") if quantity == "mean" else ("z", "station")
    payload = values[None, :, None] if quantity == "mean" else values[:, None]
    if quantity == "mean":
        payload = np.repeat(payload, len(_COMPONENTS), axis=0)
    fields[f"truth_station_{quantity}"] = (dims, payload.astype(np.float64))
    return fields


def test_station_profiles_divide_tke_by_the_squared_reference_velocity(
    tmp_path: pathlib.Path, captured_figures: list[Figure]
) -> None:
    # ``k`` is an energy, so its non-dimensionalisation is ``U_ref^2`` while the
    # mean row's is ``U_ref``. Dividing ``k`` by ``U_ref`` instead leaves a
    # correctly shaped profile under a correct-looking ``k/U_ref^2`` label, off
    # by a whole velocity scale -- on every published figure at once. The
    # reference is chosen so the two divisors disagree (2 vs 4) and the values
    # are exact in binary, so the drawn numbers pin which one was used.
    n_z, u_ref = 4, 2.0
    tke = np.array([4.0, 8.0, 16.0, 32.0])
    fields = _one_row_fields("tke", tke, n_z)

    plot_station_profiles(fields, tmp_path / "s1.png", u_ref=u_ref)

    lines = _labelled_lines(captured_figures[-1], "Truth")
    assert len(lines) == 1, f"expected one truth profile, got {len(lines)}"
    drawn = np.asarray(lines[0].get_xdata(), dtype=float)  # type: ignore[attr-defined]
    assert drawn == pytest.approx(tke / u_ref**2), (
        f"k was not divided by U_ref^2: drew {drawn.tolist()}, expected "
        f"{(tke / u_ref**2).tolist()} (dividing by U_ref alone gives "
        f"{(tke / u_ref).tolist()})"
    )


def test_station_profiles_divide_the_mean_row_by_the_reference_velocity(
    tmp_path: pathlib.Path, captured_figures: list[Figure]
) -> None:
    # The mean row's counterpart to the check above: ``u/U_ref``, first power.
    n_z, u_ref = 4, 2.0
    mean = np.array([1.0, 2.0, 4.0, 8.0])
    fields = _one_row_fields("mean", mean, n_z)

    plot_station_profiles(fields, tmp_path / "s1.png", u_ref=u_ref)

    lines = _labelled_lines(captured_figures[-1], "Truth")
    assert len(lines) == 1, f"expected one truth profile, got {len(lines)}"
    drawn = np.asarray(lines[0].get_xdata(), dtype=float)  # type: ignore[attr-defined]
    assert drawn == pytest.approx(mean / u_ref), (
        f"the mean row was not divided by U_ref: drew {drawn.tolist()}, "
        f"expected {(mean / u_ref).tolist()}"
    )


def test_station_profiles_divide_the_height_by_the_building_height(
    tmp_path: pathlib.Path, captured_figures: list[Figure]
) -> None:
    # ``z/H`` is what makes the roof line sit at 1, and it is the axis every
    # canopy comparison in the literature is read against. Multiplying by ``H``
    # instead rescales the whole ordinate: the profile keeps its shape, the
    # label still says ``z/H``, and the dotted roof line still lands at 1 --
    # just at the wrong physical height. ``H = 2`` keeps ``z/H`` and ``z*H``
    # apart at every level (the two agree only at z = 1).
    n_z, height = 4, 2.0
    fields = _one_row_fields("mean", np.array([1.0, 2.0, 4.0, 8.0]), n_z)
    z = np.asarray(fields["z"].values, dtype=float)

    plot_station_profiles(fields, tmp_path / "s1.png", building_height=height)

    lines = _labelled_lines(captured_figures[-1], "Truth")
    assert len(lines) == 1, f"expected one truth profile, got {len(lines)}"
    drawn = np.asarray(lines[0].get_ydata(), dtype=float)  # type: ignore[attr-defined]
    assert drawn == pytest.approx(z / height), (
        f"the ordinate is not z/H: drew {drawn.tolist()}, expected "
        f"{(z / height).tolist()} (z*H would give {(z * height).tolist()})"
    )
    # The roof line is drawn at z/H = 1, i.e. at the physical building height:
    # it has to fall inside the plotted levels for the panel to mean anything.
    assert float(np.min(drawn)) < 1.0 < float(np.max(drawn))


def test_station_profiles_normalise_the_ensemble_bands_too(
    tmp_path: pathlib.Path, captured_figures: list[Figure]
) -> None:
    # The quantile bands take the same scale as the line they surround. A scale
    # applied to the median but not the bands (or the reverse) draws an envelope
    # in one unit system around a profile in another, which renders as a wildly
    # over- or under-dispersed ensemble rather than as a units bug.
    n_z, u_ref = 4, 2.0
    fields = _one_row_fields("tke", np.array([4.0, 8.0, 16.0, 32.0]), n_z)
    raw = np.asarray(fields["posterior_station_tke_quantile"].values, dtype=float)
    raw = raw[:, :, 0]  # (quantile, z) at the single station
    outer = (raw[0], raw[-1])
    z = np.asarray(fields["z"].values, dtype=float)

    plot_station_profiles(fields, tmp_path / "s1.png", u_ref=u_ref, building_height=2.0)

    bands = [
        band
        for ax in captured_figures[-1].axes
        for band in _filled_bands(ax, orient="horizontal")
    ]
    assert bands, "no quantile band was drawn around the posterior profile"
    coord, lo, hi = max(bands, key=lambda b: float(np.mean(b[2] - b[1])))
    assert coord == pytest.approx(z / 2.0), "the band's ordinate is not z/H"
    assert lo == pytest.approx(outer[0] / u_ref**2)
    assert hi == pytest.approx(outer[1] / u_ref**2)


def test_station_profiles_draw_the_streamwise_u_component(
    tmp_path: pathlib.Path, captured_figures: list[Figure]
) -> None:
    # The mean row is the STREAMWISE profile: ``u``, because every shipped case
    # puts the inflow along +x. Nothing about the panel changes if the figure
    # reads ``v`` or ``w`` instead -- the axis label and the shape both survive
    # -- so the component index is pinned on the drawn values. The other two
    # components are offset by +-100 so a wrong one cannot be mistaken for
    # noise. Only the mean row is kept here: the TKE variables carry no
    # component axis and so cannot discriminate.
    n_z = 4
    fields = _eval_fields(
        n_z=n_z, station_sets=("assimilation",), with_prior=False, slabs=False
    )
    fields = fields.drop_vars(
        [v for v in fields.data_vars if "_station_tke" in v or "_station_uw" in v]
    )
    u_profile = np.arange(1.0, n_z + 1.0)
    components = np.stack([u_profile, u_profile + 100.0, u_profile - 100.0])
    fields["truth_station_mean"] = (
        ("component", "z", "station"),
        components[:, :, None].astype(np.float32),
    )

    plot_station_profiles(fields, tmp_path / "s1.png")

    lines = _labelled_lines(captured_figures[-1], "Truth")
    assert len(lines) == 1, f"expected one truth profile, got {len(lines)}"
    drawn = np.asarray(lines[0].get_xdata(), dtype=float)  # type: ignore[attr-defined]
    assert drawn == pytest.approx(
        u_profile
    ), f"the profile is not the u component: {drawn.tolist()}"


@pytest.mark.parametrize("spelling", ["z", "zt"])  # type: ignore[misc]
def test_station_profiles_exclude_the_extrapolated_top_level(
    tmp_path: pathlib.Path, captured_figures: list[Figure], spelling: str
) -> None:
    # Colocation linearly extrapolates the LAST index of every axis it moved,
    # which leaves inflated second moments there. ``extrapolated_edges`` names
    # those axes, and a profile that includes the top cell plots an artefact.
    #
    # Both vertical spellings must trim. ``zt`` is uDALES's and is the one that
    # matters in practice: PALM never extrapolates the vertical, so uDALES is
    # the backend where the trim actually fires -- a check written only against
    # ``z`` would pass while the real case plotted the artefact.
    n_z = 5
    with_edge = _eval_fields(
        n_z=n_z, extrapolated=spelling, station_sets=("assimilation",)
    )
    without_edge = _eval_fields(
        n_z=n_z, extrapolated="", station_sets=("assimilation",)
    )
    top, penultimate = float(with_edge["z"][-1]), float(with_edge["z"][-2])

    plot_station_profiles(without_edge, tmp_path / "keep.png")
    kept = captured_figures[-1]
    plot_station_profiles(with_edge, tmp_path / "drop.png")
    dropped = captured_figures[-1]

    def _highest_plotted_z(fig: Figure) -> float:
        highest = float("-inf")
        for ax in fig.axes:
            for y in _line_ydata(ax):
                finite = y[np.isfinite(y)]
                if finite.size:
                    highest = max(highest, float(finite.max()))
        return highest

    assert _highest_plotted_z(kept) == pytest.approx(top), (
        "the control figure does not reach the top level either -- the "
        "comparison below would be vacuous"
    )
    assert _highest_plotted_z(dropped) <= penultimate + 1e-6, (
        "the extrapolated top level is still plotted despite "
        f"extrapolated_edges={spelling!r}"
    )


def test_station_profiles_keep_the_validation_columns_when_truncating(
    tmp_path: pathlib.Path,
    caplog: pytest.LogCaptureFixture,
    captured_figures: list[Figure],
) -> None:
    # More columns than fit: the held-out sensors are the ones a reader came
    # for, so they are kept first -- and the drop is logged, never silent.
    #
    # Which columns survive is the whole assertion. A figure that kept the
    # assimilated ones instead still writes a PNG and still logs a drop, so
    # "a file appeared" says nothing: the panel titles carry the station index
    # and its set, and those are what get read back.
    fields = _eval_fields(
        station_sets=("assimilation",) * 4 + ("validation",) * 4, n_z=4
    )
    out = tmp_path / "s1.png"

    with caplog.at_level(logging.INFO, logger="evaluation.figures"):
        result = plot_station_profiles(fields, out, max_stations=3)

    _assert_png(result, out)
    titles = _station_panel_titles(captured_figures[-1])
    assert len(titles) == 3, f"expected 3 station columns, got {titles}"
    assert all(
        t.lower().startswith("validation") for t in titles
    ), f"an assimilated column took a held-out column's slot: {titles}"
    drawn = {int(re.search(r"#(\d+)", t).group(1)) for t in titles}  # type: ignore[union-attr]
    assert drawn == {4, 5, 6}, f"the validation columns are 4-7, drawn: {sorted(drawn)}"
    assert caplog.records, "columns were dropped without a word about it"


def test_station_profiles_title_each_column_with_its_own_stations_position(
    tmp_path: pathlib.Path, captured_figures: list[Figure]
) -> None:
    # ``_station_order`` puts the held-out columns first, so the column position
    # and the station index disagree on every run that has any -- and the
    # ``(x, y)`` in the title is what a reader uses to find the sensor in the
    # plan view and in the case config. A title reading the position instead
    # sends them to another sensor's coordinates while the index beside it says
    # otherwise, and nothing about the drawn profile looks wrong.
    fields = _eval_fields(
        n_z=4,
        station_sets=("assimilation", "assimilation", "validation"),
        slabs=False,
    )

    plot_station_profiles(fields, tmp_path / "s1.png")

    titles = _station_panel_titles(captured_figures[-1])
    assert len(titles) == 3, titles
    assert (
        titles[0].lower().startswith("validation")
    ), f"the held-out column lost the first slot, so index != position: {titles}"
    for title in titles:
        match = re.search(r"#(\d+)\D+x=(-?\d+)\D+y=(-?\d+)", title)
        assert match is not None, f"unreadable station title {title!r}"
        station = int(match.group(1))
        drawn = (float(match.group(2)), float(match.group(3)))
        # ``_eval_fields`` places station ``i`` at ``(5i, 2i)``.
        assert drawn == (5.0 * station, 2.0 * station), (
            f"panel {title!r} is captioned with another station's position "
            f"(station {station} is at {(5.0 * station, 2.0 * station)})"
        )


def test_station_profiles_no_op_without_a_station_block(
    tmp_path: pathlib.Path,
) -> None:
    # A pre-WP1.4 ``eval_fields.nc`` (or one written from a run with no sensor
    # sets) carries the slabs only.
    fields = _eval_fields(stations=False)
    out = tmp_path / "s1.png"

    _assert_no_op(plot_station_profiles(fields, out), out)


def test_station_profiles_no_op_with_zero_stations(tmp_path: pathlib.Path) -> None:
    fields = _eval_fields(station_sets=())
    out = tmp_path / "s1.png"

    _assert_no_op(plot_station_profiles(fields, out), out)


def test_station_profiles_survive_the_smoke_shape(tmp_path: pathlib.Path) -> None:
    # The 2-member spread reaches the quantile bands as five nearly-coincident
    # levels; a degenerate band is not a reason to fail.
    fields = _eval_fields(n_z=2, station_sets=("assimilation",), with_prior=False)
    out = tmp_path / "s1.png"

    _assert_png(plot_station_profiles(fields, out), out)


# ---------------------------------------------------------------------------
# F1 -- plot_mean_slices
# ---------------------------------------------------------------------------


def test_mean_slices_write_a_png_and_return_its_path(tmp_path: pathlib.Path) -> None:
    fields = _eval_fields(n_zlev=3)
    out = tmp_path / "mean_slices.png"

    _assert_png(plot_mean_slices(fields, out), out)


def test_mean_slices_solid_cells_are_outside_the_shared_norm(
    tmp_path: pathlib.Path, captured_figures: list[Figure]
) -> None:
    # ``slab_fluid == 0`` marks a building interior. Those cells are never
    # scored, and they must never set a colour limit either: here they hold a
    # sentinel four orders of magnitude above the flow, so a norm computed
    # before the mask makes every fluid cell the same shade.
    n_zlev, ny, nx = 1, 4, 5
    fluid = np.ones((n_zlev, ny, nx), dtype=np.int8)
    fluid[0, :2, :2] = 0
    rng = np.random.default_rng(3)
    means = {}
    for source, offset in (("truth", 0.0), ("posterior", 0.1), ("prior", -0.2)):
        slab = rng.random((len(_COMPONENTS), n_zlev, ny, nx)) + offset
        slab[:, fluid == 0] = 1.0e6
        means[source] = slab
    fields = _eval_fields(n_zlev=n_zlev, ny=ny, nx=nx, fluid=fluid, slab_mean=means)

    result = plot_mean_slices(fields, tmp_path / "f1.png")

    assert result is not None
    clims = _clims(captured_figures[-1])
    assert clims, "no mappable on the figure to check a norm on"
    for low, high in clims:
        assert np.isfinite(low) and np.isfinite(high)
        assert (
            max(abs(low), abs(high)) < 100.0
        ), f"the solid-cell sentinel (1e6) reached a colour norm: {clims}"


def test_mean_slices_mask_the_solid_cells_out_of_the_difference(
    tmp_path: pathlib.Path, captured_figures: list[Figure]
) -> None:
    # The difference panel has to be differenced from the MASKED fields, not
    # from the raw ones. The test above cannot see this: it gives both sources
    # the same 1e6 sentinel, so an unmasked ``posterior - truth`` is exactly 0
    # inside the buildings and sails through a limits check. Here the two
    # sentinels differ, so an unmasked difference paints a 1e6 building
    # interior on a diverging map centred at zero -- every fluid cell then
    # renders as the same shade of white.
    n_zlev, ny, nx = 1, 4, 5
    fluid = np.ones((n_zlev, ny, nx), dtype=np.int8)
    fluid[0, :2, :2] = 0
    rng = np.random.default_rng(7)
    means = {}
    for source, sentinel in (("truth", 1.0e6), ("posterior", 2.0e6)):
        slab = rng.random((len(_COMPONENTS), n_zlev, ny, nx))
        slab[:, fluid == 0] = sentinel
        means[source] = slab
    fields = _eval_fields(
        n_zlev=n_zlev, ny=ny, nx=nx, with_prior=False, fluid=fluid, slab_mean=means
    )

    result = plot_mean_slices(fields, tmp_path / "f1.png")

    assert result is not None
    difference = _image_of(captured_figures[-1], "posterior - truth")
    solid = fluid[0] == 0
    assert np.all(
        np.isnan(difference[solid])
    ), "the difference carries a value inside the buildings"
    assert np.all(
        np.isfinite(difference[~solid])
    ), "the fluid cells have no difference either -- the check is vacuous"


def test_mean_slices_difference_is_the_posterior_minus_the_truth(
    tmp_path: pathlib.Path, captured_figures: list[Figure]
) -> None:
    # The panel is titled "Posterior - truth" and its norm is symmetric
    # (``diff_max`` is a ``max |.|``), so an inverted subtraction changes
    # nothing about the figure except that every bias on it reads backwards: a
    # posterior blowing too fast is drawn as one blowing too slow. Nothing above
    # can see it -- the masking test only asks where the difference is finite --
    # so the sign is pinned on a known offset.
    n_zlev, ny, nx = 1, 4, 5
    fluid = np.ones((n_zlev, ny, nx), dtype=np.int8)
    fluid[0, :2, :2] = 0
    offset = 0.75  # exact in float32, so the assertion below is not about dtype
    rng = np.random.default_rng(11)
    truth_slab = rng.random((len(_COMPONENTS), n_zlev, ny, nx)) + 2.0
    fields = _eval_fields(
        n_zlev=n_zlev,
        ny=ny,
        nx=nx,
        with_prior=False,
        fluid=fluid,
        slab_mean={"truth": truth_slab, "posterior": truth_slab + offset},
    )

    result = plot_mean_slices(fields, tmp_path / "f1.png")

    assert result is not None
    difference = _image_of(captured_figures[-1], "posterior - truth")
    solid = fluid[0] == 0
    assert difference[~solid] == pytest.approx(offset, abs=1e-5), (
        "the difference panel is not posterior - truth: a posterior "
        f"{offset} m/s ABOVE the truth is drawn as {difference[~solid].tolist()}"
    )
    assert np.all(np.isnan(difference[solid])), "a building interior was differenced"


def test_mean_slices_label_each_row_with_the_level_it_draws(
    tmp_path: pathlib.Path, captured_figures: list[Figure]
) -> None:
    # ``max_levels`` subsamples the slab axis, so the row index and the level
    # index part company on every real run (``eval_fields.nc`` carries four
    # z-levels, the figure draws three). A row label taken from the row index
    # then prints the height of a level the panel is not showing, and the panel
    # is a perfectly ordinary picture of the wrong height. Each level is filled
    # with its own constant, so the panel says which one it is.
    n_zlev, ny, nx = 5, 3, 4
    per_level = np.arange(n_zlev, dtype=float)[None, :, None, None]
    slab = np.broadcast_to(per_level, (len(_COMPONENTS), n_zlev, ny, nx)).copy()
    fields = _eval_fields(
        n_zlev=n_zlev,
        ny=ny,
        nx=nx,
        with_prior=False,
        slab_mean={"truth": slab, "posterior": slab},
    )
    z_values = np.asarray(fields["zlev"].values, dtype=float)

    plot_mean_slices(fields, tmp_path / "f1.png", max_levels=2)

    rows = _row_label_axes(captured_figures[-1])
    assert len(rows) == 2, f"expected two labelled rows, got {len(rows)}"
    drawn_levels = []
    for ax in rows:
        level = int(round(float(np.nanmax(_image_array(ax)))))
        drawn_levels.append(level)
        assert _labelled_z(ax) == pytest.approx(z_values[level]), (
            f"the row drawing level {level} (z = {z_values[level]:.1f} m) is "
            f"labelled {ax.get_ylabel()!r}"
        )
    assert drawn_levels == [0, n_zlev - 1], (
        "the two rows do not straddle the slab axis, so a row-index label would "
        f"have been right by accident: {drawn_levels}"
    )


def test_mean_slices_keep_the_difference_out_of_the_shared_norm(
    tmp_path: pathlib.Path, captured_figures: list[Figure]
) -> None:
    # The shared norm exists so the truth/prior/posterior panels can be compared
    # by eye; the difference is a different quantity on its own diverging map.
    # Folding it into the shared limits stretches all three field panels over a
    # range no field cell occupies -- here down to -3 m/s, which no panel holds
    # -- and flattens exactly the contrast the figure is for.
    n_zlev, ny, nx = 1, 3, 4
    truth_level, posterior_level = 5.0, 2.0
    shape = (len(_COMPONENTS), n_zlev, ny, nx)
    fields = _eval_fields(
        n_zlev=n_zlev,
        ny=ny,
        nx=nx,
        with_prior=False,
        slab_mean={
            "truth": np.full(shape, truth_level),
            "posterior": np.full(shape, posterior_level),
        },
    )

    plot_mean_slices(fields, tmp_path / "f1.png")

    clims = _clims(captured_figures[-1])
    assert any(
        low == pytest.approx(posterior_level) and high == pytest.approx(truth_level)
        for low, high in clims
    ), (
        "no panel is scaled to the fields alone "
        f"({posterior_level}, {truth_level}); the -3 m/s difference reached the "
        f"shared norm: {clims}"
    )


def test_mean_slices_never_render_a_column_with_no_finite_cell(
    tmp_path: pathlib.Path,
    caplog: pytest.LogCaptureFixture,
    captured_figures: list[Figure],
) -> None:
    # One diverged member is enough to leave a whole accumulated mean field
    # non-finite, and the union-of-columns finiteness check does not fire on
    # that -- the other two columns are finite. The dead column then renders
    # entirely in the solid-cell grey, under a caption saying grey means solid:
    # the figure claims the domain is one big building. Grey must not mean two
    # things on one figure, so the column is either dropped or explicitly
    # labelled, and either way it is logged.
    n_zlev, ny, nx = 1, 4, 5
    rng = np.random.default_rng(8)
    means = {
        source: rng.random((len(_COMPONENTS), n_zlev, ny, nx))
        for source in ("truth", "posterior")
    }
    means["prior"] = np.full((len(_COMPONENTS), n_zlev, ny, nx), np.nan)
    fields = _eval_fields(n_zlev=n_zlev, ny=ny, nx=nx, slab_mean=means)
    out = tmp_path / "f1.png"

    with caplog.at_level(logging.INFO, logger="evaluation.figures"):
        result = plot_mean_slices(fields, out)

    _assert_png(result, out)
    fig = captured_figures[-1]
    titles, text = _axes_titles(fig), _axes_text(fig)
    assert "truth" in titles and "posterior" in titles, (
        "the finite columns stopped rendering: one dead column must not cost "
        f"the others ({titles})"
    )
    assert "prior" not in titles or "no finite" in text, (
        "the all-NaN prior column renders in the solid-cell grey, which the "
        "caption says means a building"
    )
    assert caplog.records, "a column was dropped without a word about it"


def test_mean_slices_keep_a_dead_subject_column_in_the_grid_and_say_why(
    tmp_path: pathlib.Path,
    caplog: pytest.LogCaptureFixture,
    captured_figures: list[Figure],
) -> None:
    # The diverged-member case, on the column the figure is ABOUT. Dropping it
    # leaves "Truth | Prior mean" under the suptitle "Time-mean u on horizontal
    # slices" and a PNG that looks complete: nothing on the figure says the
    # posterior is missing, and the log line is in a console the reader of the
    # PNG does not have. A missing column must be visible as a missing column.
    n_zlev, ny, nx = 1, 4, 5
    rng = np.random.default_rng(12)
    means = {
        source: rng.random((len(_COMPONENTS), n_zlev, ny, nx))
        for source in ("truth", "prior")
    }
    means["posterior"] = np.full((len(_COMPONENTS), n_zlev, ny, nx), np.nan)
    fields = _eval_fields(n_zlev=n_zlev, ny=ny, nx=nx, slab_mean=means)
    out = tmp_path / "f1.png"

    with caplog.at_level(logging.INFO, logger="evaluation.figures"):
        result = plot_mean_slices(fields, out)

    _assert_png(result, out)
    fig = captured_figures[-1]
    assert "posterior mean" in _axes_titles(fig), (
        "the posterior column vanished from the grid: the figure reads as a "
        "complete truth-vs-prior comparison"
    )
    assert "no finite" in _axes_text(fig), (
        "the empty posterior panel carries no on-figure explanation, so it is "
        "indistinguishable from a rendering failure"
    )
    assert "truth" in _axes_titles(fig), "the surviving columns stopped rendering"
    assert caplog.records, "the dead column was not logged"


def test_mean_slices_share_one_norm_across_truth_prior_and_posterior(
    tmp_path: pathlib.Path, captured_figures: list[Figure]
) -> None:
    # Three panels on three different scales cannot be compared by eye, which
    # is the only thing this figure is for.
    fields = _eval_fields(n_zlev=1)

    plot_mean_slices(fields, tmp_path / "f1.png")

    clims = _clims(captured_figures[-1])
    shared = max(clims.count(c) for c in clims)
    assert shared >= 3, f"fewer than three panels share a norm: {clims}"


def test_mean_slices_difference_column_uses_a_symmetric_norm(
    tmp_path: pathlib.Path, captured_figures: list[Figure]
) -> None:
    # ``posterior - truth`` on a diverging map only reads correctly when zero is
    # the centre of the norm.
    fields = _eval_fields(n_zlev=1)

    plot_mean_slices(fields, tmp_path / "f1.png")

    clims = _clims(captured_figures[-1])
    assert any(
        high > 0 and low == pytest.approx(-high) for low, high in clims
    ), f"no symmetric norm among {clims}"


def test_mean_slices_drop_the_prior_column_when_the_prior_is_absent(
    tmp_path: pathlib.Path, captured_figures: list[Figure]
) -> None:
    # The prior mean field only exists under ``run.save_prior_state``; without
    # it the figure is three columns wide, not four with a blank.
    n_zlev = 2
    plot_mean_slices(_eval_fields(n_zlev=n_zlev, with_prior=True), tmp_path / "a.png")
    with_prior = captured_figures[-1]
    plot_mean_slices(_eval_fields(n_zlev=n_zlev, with_prior=False), tmp_path / "b.png")
    without_prior = captured_figures[-1]

    n_with, n_without = len(_panel_axes(with_prior)), len(_panel_axes(without_prior))
    assert n_with - n_without == n_zlev, (
        f"dropping the prior removed {n_with - n_without} panels, expected one "
        f"per z-level ({n_zlev})"
    )
    assert "prior" in _axes_titles(with_prior)
    assert "prior" not in _axes_titles(without_prior)


def test_mean_slices_annotate_the_averaging_window(
    tmp_path: pathlib.Path, captured_figures: list[Figure]
) -> None:
    # "Never instantaneous": the figure has to say which window it averaged, or
    # a reader cannot tell it from a snapshot.
    fields = _eval_fields(n_zlev=1)

    plot_mean_slices(fields, tmp_path / "f1.png")

    text = _axes_text(captured_figures[-1])
    assert "10" in text and "40" in text, f"t_start/t_end not annotated: {text!r}"


def test_mean_slices_cap_the_number_of_levels(tmp_path: pathlib.Path) -> None:
    fields = _eval_fields(n_zlev=6)
    out = tmp_path / "f1.png"

    _assert_png(plot_mean_slices(fields, out, max_levels=2), out)


def test_mean_slices_axis_limits_stay_finite_with_an_inf_cell(
    tmp_path: pathlib.Path, captured_figures: list[Figure]
) -> None:
    # One diverged member leaves an ``inf`` in the accumulated mean. It must be
    # excluded from the norm, exactly like a solid cell.
    n_zlev, ny, nx = 1, 4, 5
    rng = np.random.default_rng(4)
    means = {
        source: rng.random((len(_COMPONENTS), n_zlev, ny, nx))
        for source in ("truth", "posterior", "prior")
    }
    means["posterior"][0, 0, 1, 1] = np.inf
    fields = _eval_fields(n_zlev=n_zlev, ny=ny, nx=nx, slab_mean=means)

    result = plot_mean_slices(fields, tmp_path / "f1.png")

    assert result is not None, "an inf cell made the whole figure no-op"
    for low, high in _clims(captured_figures[-1]):
        assert np.isfinite(low) and np.isfinite(
            high
        ), f"non-finite norm: ({low}, {high})"


def test_mean_slices_no_op_on_an_all_nan_slab(tmp_path: pathlib.Path) -> None:
    # Nothing finite survives the fluid mask, so there is no norm to compute
    # and no field to draw. A blank panel with invented limits would be worse
    # than no figure: it looks like a result.
    n_zlev, ny, nx = 1, 4, 5
    means = {
        source: np.full((len(_COMPONENTS), n_zlev, ny, nx), np.nan)
        for source in ("truth", "posterior", "prior")
    }
    fields = _eval_fields(n_zlev=n_zlev, ny=ny, nx=nx, slab_mean=means)
    out = tmp_path / "f1.png"

    _assert_no_op(plot_mean_slices(fields, out), out)


def test_mean_slices_drop_the_difference_column_without_a_truth(
    tmp_path: pathlib.Path, captured_figures: list[Figure]
) -> None:
    # No truth mean field, no ``posterior - truth`` column to draw (and no
    # truth column either) -- the remaining panels are still a valid figure.
    with_truth = _eval_fields(n_zlev=1)
    without_truth = with_truth.drop_vars(
        [v for v in with_truth.data_vars if v.startswith("truth_slab")]
    )

    plot_mean_slices(with_truth, tmp_path / "with.png")
    truthful = captured_figures[-1]
    result = plot_mean_slices(without_truth, tmp_path / "without.png")
    truthless = captured_figures[-1]

    assert result is not None, "a truth-less run still has prior/posterior means"
    assert "truth" in _axes_titles(truthful)
    assert "truth" not in _axes_titles(truthless)
    assert len(_panel_axes(truthless)) < len(_panel_axes(truthful))


def test_mean_slices_no_op_without_a_slab_block(tmp_path: pathlib.Path) -> None:
    fields = _eval_fields(slabs=False)
    out = tmp_path / "f1.png"

    _assert_no_op(plot_mean_slices(fields, out), out)


def test_mean_slices_no_op_on_an_empty_dataset(tmp_path: pathlib.Path) -> None:
    out = tmp_path / "f1.png"

    _assert_no_op(plot_mean_slices(xarray.Dataset(), out), out)


def test_mean_slices_survive_the_smoke_shape(tmp_path: pathlib.Path) -> None:
    # One level, no prior, a 2-member spread: the CI run's mean fields.
    fields = _eval_fields(n_zlev=1, ny=2, nx=2, with_prior=False, stations=False)
    out = tmp_path / "f1.png"

    _assert_png(plot_mean_slices(fields, out), out)


# ---------------------------------------------------------------------------
# S5 -- plot_sensor_fans
# ---------------------------------------------------------------------------


def test_sensor_fans_write_a_png_and_return_its_path(tmp_path: pathlib.Path) -> None:
    truth, ensemble = _sensor_series()
    out = tmp_path / "sensor_fans.png"

    result = plot_sensor_fans(
        truth,
        ensemble,
        out,
        times=np.arange(16, dtype=float),
        obs_error_std=0.2,
        window_edges=np.array([0.0, 8.0, 15.0]),
    )

    _assert_png(result, out)


def test_sensor_fans_label_each_sensor_set(
    tmp_path: pathlib.Path, captured_figures: list[Figure]
) -> None:
    # Assimilated and held-out sensors mean very different things; a fan that
    # does not say which column is which is misleading, not merely terse.
    truth, ensemble = _sensor_series(sets=("assimilation", "validation"))

    plot_sensor_fans(truth, ensemble, tmp_path / "s5.png", obs_error_std=0.2)

    text = _axes_text(captured_figures[-1])
    assert "assimilation" in text
    assert "validation" in text


def test_sensor_fans_caption_the_pre_wp21_observation_caveat(
    tmp_path: pathlib.Path, captured_figures: list[Figure]
) -> None:
    # The realized noisy observations are not persisted before WP2.1, so the
    # markers are the CLEAN truth +- sigma_o. The contract requires that to be
    # visible in the figure, not only in a docstring.
    truth, ensemble = _sensor_series()

    plot_sensor_fans(truth, ensemble, tmp_path / "s5.png", obs_error_std=0.2)

    text = _axes_text(captured_figures[-1])
    assert any(
        marker in text for marker in ("clean", "not persisted", "noisy", "wp2.1")
    ), f"no caveat about the clean observations in the figure text: {text!r}"


def test_sensor_fans_draw_the_truth_on_the_ensemble_time_axis(
    tmp_path: pathlib.Path, captured_figures: list[Figure]
) -> None:
    # The truth and the fan must sit on ONE axis, the ensemble's physical time.
    # A truth drawn against a raw frame index looks completely normal on a run
    # whose frames happen to start at t=0 with unit spacing, and silently slides
    # the whole comparison on any other -- which is every multi-window run,
    # where window w starts at ``w * sim_time``.
    n_times = 8
    truth, ensemble = _sensor_series(
        n_times=n_times, n_sensors=1, sets=("assimilation",)
    )
    times = np.linspace(100.0, 135.0, n_times)

    plot_sensor_fans(truth, ensemble, tmp_path / "s5.png", times=times)

    lines = _labelled_lines(captured_figures[-1], "Truth")
    assert lines, "no truth line on the figure"
    for line in lines:
        drawn = np.asarray(line.get_xdata(), dtype=float)  # type: ignore[attr-defined]
        assert drawn == pytest.approx(
            times
        ), f"the truth is not on the ensemble's time axis: {drawn.tolist()}"


def _two_sensor_series(
    *, n_members: int = 5, n_times: int = 6, seed: int = 0
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """One set, two sensors whose truths are metres per second apart.

    Separating the two sensors by 8 m/s is what makes "row ``r`` drew sensor
    ``r``" an assertion: with the usual noise around a common mean, a row that
    plots the wrong sensor's series looks exactly like a row that plots its own.
    """
    rng = np.random.default_rng(seed)
    base = np.array([1.0, 9.0])
    truth = base[None, :] + 0.1 * np.arange(n_times, dtype=float)[:, None]
    members = truth[None] + rng.normal(scale=0.05, size=(n_members, n_times, 2))
    return {"assimilation": truth}, {"assimilation": members}


def test_sensor_fans_draw_each_rows_own_sensor_truth(
    tmp_path: pathlib.Path, captured_figures: list[Figure]
) -> None:
    # Every row of S5 is a different sensor, and the truth is the thing the fan
    # is judged against. A truth column index frozen at 0 puts sensor 0's series
    # in every row: the fan then sits metres away from "its" truth on every row
    # but the first, which reads as a catastrophically biased posterior rather
    # than as a plotting error -- and on a run where the sensors happen to agree
    # it reads as nothing at all.
    truth, ensemble = _two_sensor_series()

    plot_sensor_fans(truth, ensemble, tmp_path / "s5.png")

    fig = captured_figures[-1]
    for row in range(truth["assimilation"].shape[1]):
        ax = _sensor_row(fig, row)
        lines = [line for line in ax.lines if str(line.get_label()) == "Truth"]
        assert len(lines) == 1, f"expected one truth line on row {row}"
        drawn = np.asarray(lines[0].get_ydata(), dtype=float)
        assert drawn == pytest.approx(
            truth["assimilation"][:, row]
        ), f"row {row} is drawn against another sensor's truth: {drawn.tolist()}"


def test_sensor_fans_count_the_surviving_members_against_the_ensemble_size(
    tmp_path: pathlib.Path, captured_figures: list[Figure]
) -> None:
    # ``M = kept/total`` is the only thing on the figure that says a member was
    # dropped, so both halves have to be real: ``kept/kept`` always reads
    # "M = 4/4", a complete ensemble, on exactly the runs where one diverged.
    # The count is per sensor -- one member can be finite at one sensor and not
    # at another -- so it is read off each panel rather than off the figure.
    n_members = 5
    truth, ensemble = _two_sensor_series(n_members=n_members)
    ensemble["assimilation"][2, 3, 0] = np.nan  # mid-window, sensor 0 only

    plot_sensor_fans(truth, ensemble, tmp_path / "s5.png")

    fig = captured_figures[-1]
    assert f"m = {n_members - 1}/{n_members}" in _panel_text(
        _sensor_row(fig, 0)
    ), f"sensor 0 dropped a member without saying so: {_panel_text(_sensor_row(fig, 0))}"
    assert f"m = {n_members}/{n_members}" in _panel_text(
        _sensor_row(fig, 1)
    ), "sensor 1 lost a member it never had a non-finite value at"


def test_sensor_fans_filter_members_on_the_last_frame_too(
    tmp_path: pathlib.Path, captured_figures: list[Figure]
) -> None:
    # The last frame is where a member diverges: it is the end of the window,
    # after the instability has had the whole window to grow. A filter that
    # stops one frame short keeps that member, and ``np.quantile`` then NaNs the
    # final point of every band -- the fan stops short of the axis and the panel
    # still claims the full ensemble.
    n_members = 5
    truth, ensemble = _two_sensor_series(n_members=n_members)
    ensemble["assimilation"][2, -1, 0] = np.nan

    plot_sensor_fans(truth, ensemble, tmp_path / "s5.png")

    ax = _sensor_row(captured_figures[-1], 0)
    assert f"m = {n_members - 1}/{n_members}" in _panel_text(
        ax
    ), f"a member non-finite only at the last frame was counted: {_panel_text(ax)}"
    medians = [line for line in ax.lines if str(line.get_label()) == "Posterior median"]
    assert medians, "the fan was not drawn at all"
    for line in medians:
        y = np.asarray(line.get_ydata(), dtype=float)
        assert np.all(np.isfinite(y)), "the fan ends in NaN: the member was kept"


def _ranked_fan(
    n_members: int = 21, n_times: int = 4
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], np.ndarray]:
    """One sensor whose members are ``t + 0 .. t + M-1`` at every frame.

    With ``M = 21`` every level in ``_BAND_QUANTILES`` lands on a member exactly
    (``np.quantile``'s linear rule puts level ``q`` at index ``20 q``), so the
    5/25/50/75/95 % values are the integers ``t + 1, 5, 10, 15, 19`` -- all
    distinct, all exact, and none of them equal to any other level's. A median
    read off the wrong row, or a band paired with the wrong partner, therefore
    cannot coincide with the right answer.
    """
    times = np.arange(float(n_times))
    members = times[None, :, None] + np.arange(float(n_members))[:, None, None]
    truth = times[:, None] + 10.0
    return {"assimilation": truth}, {"assimilation": members}, times


def test_sensor_fans_draw_the_median_not_another_quantile(
    tmp_path: pathlib.Path, captured_figures: list[Figure]
) -> None:
    # The line labelled "<fan> median" is the ONLY number a reader takes off S5
    # by eye, and every quantile row is a plausible, smooth, correctly-shaped
    # curve: drawing the 5th percentile under a "median" label renders exactly
    # like the real thing. So the drawn y-data is checked against the 0.5
    # quantile of the members themselves, computed here rather than by the
    # figure, and against the neighbouring levels it must NOT be.
    truth, ensemble, times = _ranked_fan()
    panel = ensemble["assimilation"][:, :, 0]

    plot_sensor_fans(truth, ensemble, tmp_path / "s5.png", times=times)

    ax = _sensor_row(captured_figures[-1], 0)
    medians = [line for line in ax.lines if str(line.get_label()) == "Posterior median"]
    assert len(medians) == 1, f"expected one median line, got {len(medians)}"
    drawn = np.asarray(medians[0].get_ydata(), dtype=float)
    assert np.asarray(medians[0].get_xdata(), dtype=float) == pytest.approx(times)
    assert drawn == pytest.approx(np.quantile(panel, 0.5, axis=0)), (
        f"the 'median' line is not the 0.5 quantile: drew {drawn.tolist()}, "
        f"expected {np.quantile(panel, 0.5, axis=0).tolist()}"
    )
    for level in (0.05, 0.25, 0.75, 0.95):
        other = np.quantile(panel, level, axis=0)
        assert not np.allclose(drawn, other), f"the median line is the {level} quantile"


def test_sensor_fans_pair_each_band_with_its_opposite_quantile(
    tmp_path: pathlib.Path, captured_figures: list[Figure]
) -> None:
    # The fan's caption claims nested 5--95 % and 25--75 % intervals. Pairing
    # adjacent rows instead (5--25 under the outer alpha, 25--50 under the
    # inner) draws a fan that is monotone, nested and entirely plausible while
    # understating the ensemble's spread by roughly half -- the one thing S5
    # exists to show. The pairs are read off the filled polygons because that is
    # where the pairing lives; nothing else on the figure records it.
    truth, ensemble, times = _ranked_fan()
    panel = ensemble["assimilation"][:, :, 0]
    wanted = {
        (0.05, 0.95): (
            np.quantile(panel, 0.05, axis=0),
            np.quantile(panel, 0.95, axis=0),
        ),
        (0.25, 0.75): (
            np.quantile(panel, 0.25, axis=0),
            np.quantile(panel, 0.75, axis=0),
        ),
    }

    plot_sensor_fans(truth, ensemble, tmp_path / "s5.png", times=times)

    bands = _filled_bands(_sensor_row(captured_figures[-1], 0))
    assert len(bands) == len(
        wanted
    ), f"{len(bands)} bands drawn, expected {len(wanted)}"
    for coord, lo, hi in bands:
        assert coord == pytest.approx(times)
    unmatched = dict(wanted)
    for coord, lo, hi in bands:
        for levels, (want_lo, want_hi) in list(unmatched.items()):
            if np.allclose(lo, want_lo) and np.allclose(hi, want_hi):
                del unmatched[levels]
                break
    assert not unmatched, (
        "no band spans "
        + ", ".join(f"{a:.0%}-{b:.0%}" for a, b in unmatched)
        + "; the bands drawn were "
        + "; ".join(f"({lo.tolist()}, {hi.tolist()})" for _, lo, hi in bands)
    )
    # The outer (widest) band is the lighter one: a fan whose alphas run the
    # other way reads as a 25--75 % interval with a faint halo.
    widths = [float(np.mean(hi - lo)) for _, lo, hi in bands]
    alphas = [
        float(c.get_alpha() or 1.0)
        for c in _sensor_row(captured_figures[-1], 0).collections
        if c.get_paths()
    ]
    assert alphas[int(np.argmax(widths))] < alphas[int(np.argmin(widths))]


def test_sensor_fans_mark_the_window_boundaries_they_are_given(
    tmp_path: pathlib.Path, captured_figures: list[Figure]
) -> None:
    # The boundaries are what makes a multi-window fan readable -- a jump at a
    # window edge is an update, a jump inside one is the model. The wiring test
    # only checks the kwarg the script computes, so a figure that accepts
    # ``window_edges`` and never draws them passes it while the PNG carries no
    # window structure at all.
    truth, ensemble = _two_sensor_series()
    edges = [0.0, 2.0, 5.0]

    plot_sensor_fans(
        truth,
        ensemble,
        tmp_path / "s5.png",
        times=np.arange(truth["assimilation"].shape[0], dtype=float),
        window_edges=np.asarray(edges),
    )

    fig = captured_figures[-1]
    for row in range(2):
        ax = _sensor_row(fig, row)
        assert _vertical_line_positions(ax) == pytest.approx(
            edges
        ), f"row {row} carries no window boundaries: {_vertical_line_positions(ax)}"
    assert "w0" in _panel_text(_sensor_row(fig, 0)), "the windows are not numbered"


def test_sensor_fans_y_limits_include_the_observation_envelope(
    tmp_path: pathlib.Path, captured_figures: list[Figure]
) -> None:
    # sigma_o is routinely wider than the fan (the shipped
    # ``esmda.obs_error_std`` is 0.25 against sensor series that vary by far
    # less), and an envelope outside the y-limits does not clip politely: it
    # fills the panel edge to edge and reads as a background tint, so the one
    # thing the figure says about the observations -- how wide their error is
    # against the spread -- is lost exactly when it matters.
    n_times = 6
    truth = {"assimilation": 4.0 + 0.01 * np.arange(n_times, dtype=float)[:, None]}
    ensemble = {
        "assimilation": truth["assimilation"][None]
        + 0.02 * np.linspace(-1.0, 1.0, 4)[:, None, None]
    }
    obs_error_std = 2.0

    plot_sensor_fans(truth, ensemble, tmp_path / "s5.png", obs_error_std=obs_error_std)

    ax = _sensor_row(captured_figures[-1], 0)
    low, high = ax.get_ylim()
    series = truth["assimilation"][:, 0]
    assert low <= float(series.min()) - obs_error_std, (
        f"the -sigma_o edge of the envelope ({series.min() - obs_error_std}) is "
        f"outside the panel: {(low, high)}"
    )
    assert high >= float(series.max()) + obs_error_std, (
        f"the +sigma_o edge of the envelope ({series.max() + obs_error_std}) is "
        f"outside the panel: {(low, high)}"
    )


def test_sensor_fans_skip_a_truth_whose_cadence_does_not_match(
    tmp_path: pathlib.Path,
    caplog: pytest.LogCaptureFixture,
    captured_figures: list[Figure],
) -> None:
    # A truth of a different length is a truth on a different axis, and the only
    # honest options are to be told what that axis is or to leave it out.
    # Stretching it over ``linspace(t[0], t[-1], T_truth)`` invents one: it
    # assumes the truth is uniform over exactly the ensemble's span, and when it
    # is not the drawn curve is wrong by O(1) on a unit-amplitude signal -- a
    # wrong truth being far worse than a missing one.
    truth, ensemble = _sensor_series(n_times=8, n_sensors=1, sets=("assimilation",))
    truth["assimilation"] = truth["assimilation"][:5]
    out = tmp_path / "s5.png"

    with caplog.at_level(logging.INFO, logger="evaluation.figures"):
        result = plot_sensor_fans(truth, ensemble, out, times=np.linspace(0.0, 7.0, 8))

    _assert_png(result, out)
    assert not _labelled_lines(
        captured_figures[-1], "Truth"
    ), "a truth on a mismatched cadence was drawn on an invented axis"
    assert caplog.records, "the truth was dropped without a word about it"


def test_sensor_fans_drop_a_non_finite_member_and_still_draw_the_fan(
    tmp_path: pathlib.Path,
    caplog: pytest.LogCaptureFixture,
    captured_figures: list[Figure],
) -> None:
    # ``np.quantile`` propagates NaN, so with an ``.any()``-shaped guard a single
    # diverged member empties the bands and makes the median all-NaN: the panel
    # renders as a bare truth line and reads as "the ensemble was never plotted"
    # rather than "one member blew up". WP1.4 records non-finite values through
    # the ensemble reductions as expected on real runs, so this is the common
    # case, not the exotic one.
    n_members = 5
    truth, ensemble = _sensor_series(
        n_members=n_members, n_times=8, n_sensors=1, sets=("assimilation",)
    )
    ensemble["assimilation"][2, 4, 0] = np.nan
    out = tmp_path / "s5.png"

    with caplog.at_level(logging.INFO, logger="evaluation.figures"):
        result = plot_sensor_fans(truth, ensemble, out)

    _assert_png(result, out)
    fig = captured_figures[-1]
    medians = _labelled_lines(fig, "Posterior median")
    assert medians, "the fan was not drawn at all"
    for line in medians:
        y = np.asarray(line.get_ydata(), dtype=float)  # type: ignore[attr-defined]
        assert np.all(
            np.isfinite(y)
        ), "one non-finite member emptied the whole fan through np.quantile"
    assert str(n_members - 1) in _axes_text(
        fig
    ), "the panel does not say how many members are behind the fan"
    assert caplog.records, "a member was dropped without a word about it"


def test_sensor_fans_survive_the_smoke_shape(tmp_path: pathlib.Path) -> None:
    # M = 2: the 0.05 and 0.25 quantiles coincide and the fan collapses to a
    # ribbon between the two members.
    truth, ensemble = _sensor_series(n_members=2, n_times=4, n_sensors=1)
    out = tmp_path / "s5.png"

    _assert_png(plot_sensor_fans(truth, ensemble, out, obs_error_std=0.1), out)


def test_sensor_fans_render_without_an_observation_error(
    tmp_path: pathlib.Path,
) -> None:
    truth, ensemble = _sensor_series()
    out = tmp_path / "s5.png"

    _assert_png(plot_sensor_fans(truth, ensemble, out), out)


def test_sensor_fans_cap_the_number_of_sensors(
    tmp_path: pathlib.Path, caplog: pytest.LogCaptureFixture
) -> None:
    truth, ensemble = _sensor_series(n_sensors=7, sets=("assimilation",))
    out = tmp_path / "s5.png"

    with caplog.at_level(logging.INFO, logger="evaluation.figures"):
        result = plot_sensor_fans(truth, ensemble, out, max_sensors=2)

    _assert_png(result, out)
    assert caplog.records, "sensors were dropped without a word about it"


def test_sensor_fans_no_op_on_empty_inputs(tmp_path: pathlib.Path) -> None:
    out = tmp_path / "s5.png"

    _assert_no_op(plot_sensor_fans({}, {}, out), out)


def test_sensor_fans_no_op_with_zero_sensors(tmp_path: pathlib.Path) -> None:
    # A sensor set that ended up empty (every sensor outside the domain -- the
    # smoke shape's validation set does exactly this).
    truth = {"validation": np.zeros((8, 0))}
    ensemble = {"validation": np.zeros((3, 8, 0))}
    out = tmp_path / "s5.png"

    _assert_no_op(plot_sensor_fans(truth, ensemble, out), out)


def test_sensor_fans_no_op_with_no_ensemble_members(tmp_path: pathlib.Path) -> None:
    truth = {"assimilation": np.zeros((8, 2))}
    ensemble = {"assimilation": np.zeros((0, 8, 2))}
    out = tmp_path / "s5.png"

    _assert_no_op(plot_sensor_fans(truth, ensemble, out), out)


# ---------------------------------------------------------------------------
# D1 -- plot_rank_histogram
# ---------------------------------------------------------------------------


def test_rank_histogram_writes_a_png_and_returns_its_path(
    tmp_path: pathlib.Path,
) -> None:
    counts = _rank_counts([4, 6, 5, 7, 5, 6, 4, 5, 6, 4, 8])
    out = tmp_path / "rank_histogram.png"

    _assert_png(plot_rank_histogram(counts, out), out)


def test_rank_histogram_coarsens_with_array_split_semantics(
    tmp_path: pathlib.Path, captured_figures: list[Figure]
) -> None:
    # The coarsening IS the figure: ``M + 1`` rank bins (11 at M = 10, 129 at
    # M = 128) are unreadable and the pooled counts are far too small to fill
    # them, so they are grouped into ``min(n_bins, M+1)`` contiguous groups
    # with ``np.array_split`` -- which, for 7 bins into 3 groups, gives widths
    # [3, 2, 2], NOT [2, 2, 3] and not an even split it cannot make.
    #
    # Powers of two make every group sum unique, so a wrong grouping cannot
    # accidentally produce the right numbers.
    counts = [1, 2, 4, 8, 16, 32, 64]
    n_bins = 3
    total = sum(counts)
    groups = np.array_split(np.asarray(counts), n_bins)
    expected_heights = sorted(float(g.sum()) for g in groups)

    plot_rank_histogram(_rank_counts(counts), tmp_path / "d1.png", n_bins=n_bins)

    panels = _bar_axes(captured_figures[-1])
    assert panels, "no bars on the figure"
    for ax in panels:
        heights = sorted(_bar_heights(ax))
        assert len(heights) == n_bins, f"{len(heights)} bars, expected {n_bins}"
        assert heights == pytest.approx(expected_heights), (
            f"bar heights {heights} do not match the array_split grouping "
            f"{expected_heights} (group widths {[g.size for g in groups]})"
        )
        assert sum(heights) == pytest.approx(total), "the coarsening lost counts"


def test_rank_histogram_bars_span_the_rank_bins_they_group(
    tmp_path: pathlib.Path, captured_figures: list[Figure]
) -> None:
    # Heights alone do not make a histogram: WHERE each bar sits and how wide it
    # is are what tie a group of counts to a range of ranks, and the x axis is
    # in rank units so the unequal ``array_split`` widths are visible. Bars
    # placed by their centres rather than their left edges shift the entire
    # histogram by half a bin -- the shape, the totals and the reference step
    # all survive, but the leftmost bar starts at a negative rank and every
    # group is read against the wrong ranks. That is exactly the misreading D1
    # exists to prevent (a U-shape read one bin over is a different diagnosis).
    #
    # Expected geometry is written out rather than recomputed with
    # ``array_split``: for 7 rank bins in 3 groups the widths are [3, 2, 2], so
    # the bars occupy ranks [0,3), [3,5), [5,7) and carry 1+2+4, 8+16, 32+64.
    counts = [1, 2, 4, 8, 16, 32, 64]
    # (left edge, width, height) per bar, flattened so a mismatch prints as
    # numbers rather than as a nest of tuples.
    expected = [0.0, 3.0, 7.0, 3.0, 2.0, 24.0, 5.0, 2.0, 96.0]

    plot_rank_histogram(_rank_counts(counts), tmp_path / "d1.png", n_bins=3)

    panels = _bar_axes(captured_figures[-1])
    assert panels, "no bars on the figure"
    for ax in panels:
        bars = sorted(
            (float(p.get_x()), float(p.get_width()), float(p.get_height()))
            for p in _rectangles(ax)
        )
        drawn = [value for bar in bars for value in bar]
        assert drawn == pytest.approx(expected), (
            f"bar geometry {drawn} is not the [0,3)/[3,5)/[5,7) rank groups "
            f"{expected}; centred bars would start at {expected[0] - 1.5}"
        )
        # The bars tile the rank axis exactly: no gap, no overlap, and the last
        # one ends at ``M + 1``.
        assert bars[0][0] == pytest.approx(0.0)
        assert bars[-1][0] + bars[-1][1] == pytest.approx(len(counts))
    # The reference step and its band are drawn on the same rank axis, so a
    # half-bin shift of the bars alone would also put them out of register.
    for ax in panels:
        left, right = ax.get_xlim()  # type: ignore[attr-defined]
        assert left <= 0.0 and right >= len(counts)


def test_rank_histogram_reference_is_proportional_to_the_group_width(
    tmp_path: pathlib.Path, captured_figures: list[Figure]
) -> None:
    # With unequal group widths the uniform reference is NOT a flat line at
    # ``N / n_bins``: a group covering three of seven rank bins should hold
    # three sevenths of the mass. Drawing the flat line makes the widest group
    # look over-populated on a perfectly calibrated ensemble.
    counts = [1, 2, 4, 8, 16, 32, 64]
    n_bins = 3
    total = sum(counts)
    sizes = np.array([g.size for g in np.array_split(np.arange(len(counts)), n_bins)])
    expected = total * sizes / len(counts)
    flat = total / n_bins

    plot_rank_histogram(_rank_counts(counts), tmp_path / "d1.png", n_bins=n_bins)

    panels = _bar_axes(captured_figures[-1])
    assert panels, "no bars on the figure"
    wanted = np.unique(np.round(expected, 6))
    assert wanted.size == 2 and not np.allclose(
        wanted, flat
    ), "the fixture is not diagnostic"
    for ax in panels:
        found = [
            y
            for y in _line_ydata(ax)
            if np.all(np.isfinite(y))
            and np.array_equal(np.unique(np.round(y, 6)), wanted)
        ]
        assert found, (
            "no artist carries the per-group expected levels "
            f"{expected.tolist()} (a flat reference would sit at {flat}); "
            f"y-data found: {[np.unique(np.round(y, 6)).tolist() for y in _line_ydata(ax)]}"
        )


def test_rank_histogram_binomial_band_matches_the_group_probability(
    tmp_path: pathlib.Path,
) -> None:
    # The consistency band is drawn as a ``fill_between`` step, which carries no
    # readable y-data, so it is pinned on the arithmetic the figure calls:
    # ``2*sqrt(N*p_g*(1-p_g))`` with ``p_g = size_g/(M+1)``. Using a single
    # ``p = 1/G`` would give every group the same band despite unequal widths,
    # which is the same error as the flat reference line one test above.
    counts = np.array([1, 2, 4, 8, 16, 32, 64], dtype=float)
    n_bins = 3
    total, n_ranks = counts.sum(), counts.size

    sizes, grouped, expected, band = _coarsen_rank_counts(counts, n_bins)

    p = sizes / n_ranks
    assert sizes.tolist() == [3.0, 2.0, 2.0]
    assert grouped.tolist() == [7.0, 24.0, 96.0]
    assert expected == pytest.approx(total * p)
    assert band == pytest.approx(2.0 * np.sqrt(total * p * (1.0 - p)))
    assert band[0] != pytest.approx(band[1]), "the band ignores the group widths"
    assert expected.sum() == pytest.approx(total)


def test_rank_histogram_never_asks_for_more_groups_than_rank_bins(
    tmp_path: pathlib.Path,
) -> None:
    # ``min(n_bins, M+1)``: the smoke shape has three rank bins and cannot be
    # split into ten.
    sizes, grouped, expected, band = _coarsen_rank_counts(np.array([2.0, 1.0, 3.0]), 10)

    assert sizes.size == grouped.size == expected.size == band.size == 3
    assert sizes.tolist() == [1.0, 1.0, 1.0]


def test_rank_histogram_labels_the_pooled_sample_size(
    tmp_path: pathlib.Path, captured_figures: list[Figure]
) -> None:
    # Counts, not densities, and ``N`` in the panel -- a rank histogram over 30
    # ranks and one over 3000 look identical otherwise.
    counts = [1, 2, 4, 8, 16, 32, 64]

    plot_rank_histogram(_rank_counts(counts), tmp_path / "d1.png", n_bins=3)

    assert "127" in _axes_text(captured_figures[-1])


def test_rank_histogram_labels_the_rank_axis_with_the_ranks_that_exist(
    tmp_path: pathlib.Path, captured_figures: list[Figure]
) -> None:
    # ``M + 1`` bins are ranks ``0 .. M``. Naming an extra rank on the axis is
    # the difference between "the ensemble has M members" and "M + 1", which is
    # the whole quantity a rank histogram is read against -- and the coarsened
    # bars cannot be counted back to it by eye.
    counts = [3, 4, 5, 6, 7]  # M + 1 = 5 rank bins, i.e. ranks 0-4

    plot_rank_histogram(_rank_counts(counts), tmp_path / "d1.png", n_bins=3)

    text = _axes_text(captured_figures[-1])
    assert f"(0-{len(counts) - 1})" in text, f"the rank axis is mislabelled: {text!r}"
    assert f"(0-{len(counts)})" not in text, f"the rank axis is off by one: {text!r}"


def test_rank_histogram_pools_over_the_statistics(
    tmp_path: pathlib.Path, captured_figures: list[Figure]
) -> None:
    # One (set, half) cell holds a count vector per statistic (``mean_u``,
    # ``variance_magnitude``, ...); they are separate samples of the same
    # calibration and are summed, not drawn as separate panels.
    counts = [2, 3, 5, 7]
    statistics = ("mean_u", "variance_magnitude", "mean_magnitude")

    plot_rank_histogram(
        _rank_counts(counts, statistics=statistics),
        tmp_path / "d1.png",
        n_bins=4,
    )

    for ax in _bar_axes(captured_figures[-1]):
        assert sum(_bar_heights(ax)) == pytest.approx(sum(counts) * len(statistics))


def test_rank_histogram_drops_the_prior_column_when_there_is_no_prior_half(
    tmp_path: pathlib.Path, captured_figures: list[Figure]
) -> None:
    counts = [3, 4, 5, 6]
    sets = ("assimilation", "validation")

    plot_rank_histogram(
        _rank_counts(counts, sets=sets, halves=("prior", "posterior")),
        tmp_path / "both.png",
    )
    both = len(_bar_axes(captured_figures[-1]))
    plot_rank_histogram(
        _rank_counts(counts, sets=sets, halves=("posterior",)),
        tmp_path / "one.png",
    )
    posterior_only = len(_bar_axes(captured_figures[-1]))

    assert both == 2 * len(sets)
    assert posterior_only == len(sets)


def test_rank_histogram_survives_the_smoke_shape(tmp_path: pathlib.Path) -> None:
    # M = 2 gives three rank bins, fewer than the ten requested. That is a
    # legitimate (if uninformative) plot: ``min(n_bins, M+1)`` groups.
    out = tmp_path / "d1.png"

    result = plot_rank_histogram(_rank_counts([2, 1, 3]), out, n_bins=10)

    _assert_png(result, out)


def test_rank_histogram_at_the_smoke_shape_keeps_every_rank_bin(
    tmp_path: pathlib.Path, captured_figures: list[Figure]
) -> None:
    counts = [2, 1, 3]

    plot_rank_histogram(_rank_counts(counts), tmp_path / "d1.png", n_bins=10)

    for ax in _bar_axes(captured_figures[-1]):
        assert sorted(_bar_heights(ax)) == pytest.approx(
            sorted(float(c) for c in counts)
        )


def test_rank_histogram_no_op_on_empty_counts(tmp_path: pathlib.Path) -> None:
    out = tmp_path / "d1.png"

    _assert_no_op(plot_rank_histogram({}, out), out)


def test_rank_histogram_no_op_when_nothing_was_ranked(tmp_path: pathlib.Path) -> None:
    # ``N == 0``: every rank was non-finite (a window with no frames), so the
    # bincount is all zeros. Normalising by N here is a divide by zero.
    out = tmp_path / "d1.png"

    _assert_no_op(plot_rank_histogram(_rank_counts([0, 0, 0, 0]), out), out)


def test_rank_histogram_no_op_on_a_single_rank_bin(tmp_path: pathlib.Path) -> None:
    # ``M + 1 < 2`` -- a one-member "ensemble" has no rank distribution.
    out = tmp_path / "d1.png"

    _assert_no_op(plot_rank_histogram(_rank_counts([7]), out), out)


def test_rank_histogram_no_op_when_a_set_carries_no_statistics(
    tmp_path: pathlib.Path,
) -> None:
    out = tmp_path / "d1.png"

    _assert_no_op(plot_rank_histogram({"assimilation": {"posterior": {}}}, out), out)


# ---------------------------------------------------------------------------
# The wiring -- scripts/esmda/make_esmda_figures.py
# ---------------------------------------------------------------------------
#
# Everything above tests the library; none of it executes the stage that calls
# it. That stage is where the run-dir layout, the YAML shapes and the config
# reads live, and no e2e ESMDA test reaches it either -- they all run under
# ``run.skip_viz=true``, whose first act is to return. WP1.3's second review
# round recorded exactly this: "the test that 'covered' this called the library
# directly and bypassed the script, which is exactly how the gap survived the
# round it was written in."
#
# So the two YAML/config readers are unit-tested on the shapes an old run dir
# actually produces, and ``make_figures`` itself is driven end to end on a
# synthetic run dir -- once complete, once missing the two artifacts an older
# metric stage never wrote.

_RUN_WINDOWS = 3  # >1, so the window boundaries are a real thing to get wrong
_RUN_SIM_TIME = 4.0
_RUN_FRAMES = 4  # per window


def _state_dataset(
    n_ensemble: int | None, n_time: int, seed: int, n_cells: int = 5
) -> xarray.Dataset:
    """A ``(time, z, y, x)`` state, with an ``ensemble`` axis when asked."""
    rng = np.random.default_rng(seed)
    axis = np.linspace(0.0, 20.0, n_cells)
    dims: tuple[str, ...] = ("time", "z", "y", "x")
    shape: tuple[int, ...] = (n_time, n_cells, n_cells, n_cells)
    coords: dict[str, object] = {
        "time": np.arange(n_time, dtype=float) * (_RUN_SIM_TIME / n_time),
        "z": axis,
        "y": axis,
        "x": axis,
    }
    if n_ensemble is not None:
        dims = ("ensemble",) + dims
        shape = (n_ensemble,) + shape
        coords["ensemble"] = np.arange(n_ensemble)
    return xarray.Dataset(
        {n: (dims, rng.normal(size=shape) + 2.0) for n in ("u", "v", "w")},
        coords=coords,
    )


def _figure_run_dir(
    tmp_path: pathlib.Path,
    *,
    n_members: int = 4,
    is_dynamic: bool = True,
    skip_viz: bool = False,
    eval_fields: bool = True,
    rank_counts: bool = True,
    building_height: float | None = None,
) -> pathlib.Path:
    """An ESMDA run dir complete enough for the non-``skip_viz`` figure stage.

    Same layout ``run_esmda.py`` writes -- ``truth_access.yaml`` plus the
    assembled parameter/state artifacts -- with the two post-hoc metric
    artifacts (``eval_fields.nc``, ``run_summary.yaml``) switchable, because
    "the run dir predates them" is the degradation the stage exists to survive.
    """
    from scripts.esmda._esmda_common import write_yaml

    run_dir = tmp_path / "run"
    (run_dir / "windows").mkdir(parents=True)
    n_total = _RUN_WINDOWS * _RUN_FRAMES

    config: dict[str, object] = {
        "run": {"skip_viz": skip_viz},
        "esmda": {"obs_error_std": 0.2},
        "obs": {
            "mode": "points",
            "x_points": [4.0, 12.0],
            "y_points": [5.0, 13.0],
            "z_points": [6.0, 7.0],
            "validation_x_points": [8.0],
            "validation_y_points": [9.0],
            "validation_z_points": [10.0],
        },
    }
    if building_height is not None:
        config["geometry"] = {"building_height": building_height}
    write_yaml(config, run_dir / "config.yaml")

    truth = _state_dataset(None, n_total, seed=1).assign_coords(
        time=np.linspace(0.0, _RUN_WINDOWS * _RUN_SIM_TIME, n_total, endpoint=False)
    )
    truth.to_netcdf(run_dir / "truth_state.nc")
    write_yaml(
        {
            "true_state_path": str(run_dir / "truth_state.nc"),
            "n_total": n_total,
            "x_offset": 0.0,
            "start_idx": 0,
            "t_offset": 0.0,
            "num_windows": _RUN_WINDOWS,
            "n_per_window": _RUN_FRAMES,
            "sim_time": _RUN_SIM_TIME,
            "is_dynamic": is_dynamic,
            "truth_solver_name": "pylbm",
            "assim_solver_name": "pylbm",
        },
        run_dir / "truth_access.yaml",
    )

    for w in range(_RUN_WINDOWS):
        _state_dataset(n_members, _RUN_FRAMES, seed=10 + w).to_netcdf(
            run_dir / "windows" / f"window_{w}_posterior_state.nc"
        )

    # ``posterior_state_mean.nc`` carries the ensemble mean/std of |U| beside the
    # mean state; the animation and the final-state plot read only those two.
    mean_state = _state_dataset(None, n_total, seed=20)
    magnitude = np.sqrt(sum(mean_state[c] ** 2 for c in ("u", "v", "w")))
    mean_state["vel_mean"] = magnitude
    mean_state["vel_std"] = magnitude * 0.1
    mean_state.to_netcdf(run_dir / "posterior_state_mean.nc")

    members = np.linspace(-1.0, 1.0, n_members)[:, None]
    knots = np.array([0.0, 1.0, 2.0])
    for name, spread in (("posterior_params", 1.0), ("prior_params", 4.0)):
        xarray.Dataset(
            {
                "velocity_magnitude": (
                    ("ensemble", "time"),
                    5.0 + spread * members * np.ones_like(knots),
                )
            },
            coords={"ensemble": np.arange(n_members), "time": knots},
        ).to_netcdf(run_dir / f"{name}.nc")
    xarray.Dataset(
        {"velocity_magnitude": (("time",), np.full(knots.size, 5.0))},
        coords={"time": knots},
    ).to_netcdf(run_dir / "true_params.nc")

    if eval_fields:
        _eval_fields(n_zlev=2, ny=4, nx=5, n_z=4).to_netcdf(run_dir / "eval_fields.nc")
    if rank_counts:
        write_yaml(
            {
                "sensor_statistics": {
                    "assimilation": {
                        "n_members": n_members,
                        "posterior": {"mean_u": {"rank_counts": [1, 2, 3, 2, 1]}},
                    }
                }
            },
            run_dir / "run_summary.yaml",
        )
    return run_dir


def test_reference_velocity_reads_the_truth_inflow_speed() -> None:
    from scripts.esmda.make_esmda_figures import _reference_velocity

    # A time-varying truth is averaged over its knots: one profile figure covers
    # the whole run, so it gets one reference speed.
    present = _truth_dataset(times=(0.0, 1.0), velocity_magnitude=[4.0, 6.0])

    assert _reference_velocity(present) == pytest.approx(5.0)


@pytest.mark.parametrize(  # type: ignore[misc]
    "values",
    [
        pytest.param(None, id="absent"),
        pytest.param([np.nan, np.nan], id="all_non_finite"),
        pytest.param([0.0, 0.0], id="zero"),
        pytest.param([-3.0, -3.0], id="negative"),
    ],
)
def test_reference_velocity_is_none_when_it_cannot_normalise(
    values: list[float] | None,
) -> None:
    # ``None`` means "plot in m/s". A pressure-gradient driven case estimates no
    # inflow speed at all, and a zero or negative one would divide the profiles
    # by nothing or flip their sign -- all three are the same answer here.
    from scripts.esmda.make_esmda_figures import _reference_velocity

    truth = (
        _truth_dataset(times=(0.0, 1.0), inflow_angle=[270.0, 270.0])
        if values is None
        else _truth_dataset(times=(0.0, 1.0), velocity_magnitude=values)
    )

    assert _reference_velocity(truth) is None


def test_rank_counts_reads_the_summary_block() -> None:
    from scripts.esmda.make_esmda_figures import _rank_counts as read_rank_counts

    summary = {
        "sensor_statistics": {
            "assimilation": {
                "n_members": 4,
                "prior": {"mean_u": {"rank_counts": [1, 2, 1], "crps": 0.4}},
                "posterior": {
                    "mean_u": {"rank_counts": [2, 1, 1], "crps": 0.2},
                    "variance_magnitude": {"rank_counts": [1, 1, 2]},
                },
            }
        }
    }

    counts = read_rank_counts(summary)

    # Only the rank vectors survive: the ensemble sizes and the other scores are
    # this script's business, not the figure's (invariant 5).
    assert counts == {
        "assimilation": {
            "prior": {"mean_u": [1, 2, 1]},
            "posterior": {"mean_u": [2, 1, 1], "variance_magnitude": [1, 1, 2]},
        }
    }


@pytest.mark.parametrize(  # type: ignore[misc]
    "summary",
    [
        pytest.param({}, id="no_summary_at_all"),
        pytest.param({"sensor_statistics": None}, id="empty_block"),
        pytest.param(
            {"sensor_statistics": {"assimilation": {"n_members": 4}}},
            id="set_with_no_halves",
        ),
        pytest.param(
            {"sensor_statistics": {"assimilation": {"posterior": {"mean_u": 0.3}}}},
            id="half_whose_entries_are_bare_scores",
        ),
        pytest.param(
            {
                "sensor_statistics": {
                    "assimilation": {"posterior": {"mean_u": {"crps": 0.3}}}
                }
            },
            id="entry_predating_the_rank_counts",
        ),
    ],
)
def test_rank_counts_drops_a_summary_it_cannot_read(summary: dict) -> None:
    # An old run dir's summary is not malformed, it is *earlier*: WP1.3 added the
    # halves and the rank vectors. Guessing at a shape that is not there would
    # draw a histogram of nothing, so an empty result -- the caller's cue to skip
    # D1 with a printed line -- is the answer.
    from scripts.esmda.make_esmda_figures import _rank_counts as read_rank_counts

    assert read_rank_counts(summary) == {}


def test_make_figures_draws_the_whole_set_on_a_complete_run_dir(
    tmp_path: pathlib.Path,
) -> None:
    from scripts.esmda.make_esmda_figures import make_figures

    run_dir = _figure_run_dir(tmp_path)

    make_figures(run_dir)

    for name in (
        "rollout_time_evolution.png",
        "parameter_error.png",
        "final_state_with_obs.png",
        "sensor_timeseries_assimilation.png",
        "sensor_timeseries_validation.png",
        "parameter_marginals.png",
        "station_profiles.png",
        "mean_slices.png",
        "sensor_fans.png",
        "rank_histogram.png",
    ):
        assert (run_dir / name).is_file(), f"{name} was not written"
    # ffmpeg is optional in this env; the writer falls back to a GIF without it.
    assert (run_dir / "rollout_animation.mp4").is_file() or (
        run_dir / "rollout_animation.gif"
    ).is_file()


def test_make_figures_skips_what_an_old_run_dir_cannot_support(
    tmp_path: pathlib.Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # The degradation the stage exists for: a run whose metric stage predates
    # ``eval_fields.nc`` (WP1.4) and the ``sensor_statistics`` block (WP1.3).
    # Three figures cannot be drawn; the other seven must be, and each skip has
    # to name the artifact and how to get it.
    from scripts.esmda.make_esmda_figures import make_figures

    run_dir = _figure_run_dir(tmp_path, eval_fields=False, rank_counts=False)

    make_figures(run_dir)

    printed = capsys.readouterr().out
    for name in ("station_profiles.png", "mean_slices.png", "rank_histogram.png"):
        assert not (run_dir / name).exists(), f"{name} was drawn from nothing"
        assert name in printed, f"{name} was skipped silently"
    assert "compute_esmda_metrics.py" in printed, "the skip does not say how to fix it"
    # A skip never costs the figures after it (master-plan invariant 3).
    for name in ("parameter_marginals.png", "sensor_fans.png"):
        assert (run_dir / name).is_file(), f"{name} was lost to an earlier skip"


def test_make_figures_is_a_no_op_under_skip_viz(tmp_path: pathlib.Path) -> None:
    from scripts.esmda.make_esmda_figures import make_figures

    run_dir = _figure_run_dir(tmp_path, skip_viz=True)

    make_figures(run_dir)

    assert not list(run_dir.glob("*.png"))


def _captured_kwargs(
    monkeypatch: pytest.MonkeyPatch, *names: str
) -> dict[str, dict[str, object]]:
    """Stub out figure calls in the wiring and record the kwargs they were given.

    What this pins is the *arguments the script computes*, which is where the
    run-dir and config knowledge lives; the figures themselves are covered
    above. ``animate_rollout_state`` goes too -- it is not under test here and
    it is the slowest thing in the stage.
    """
    import scripts.esmda.make_esmda_figures as wiring

    recorded: dict[str, dict[str, object]] = {}

    def _record(name: str) -> object:
        def _stub(*args: object, **kwargs: object) -> None:
            recorded[name] = kwargs
            return None

        return _stub

    for name in names + ("animate_rollout_state",):
        monkeypatch.setattr(wiring, name, _record(name))
    return recorded


@pytest.mark.parametrize("is_dynamic", [True, False])  # type: ignore[misc]
def test_make_figures_marks_the_sensor_window_boundaries_on_any_multi_window_run(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, is_dynamic: bool
) -> None:
    # S5's x-axis is physical time on every run -- ``ensemble_sensor_series``
    # rebases window ``w`` onto ``w * sim_time`` whether or not the parameters
    # are dynamic -- so a static multi-window run (conf/run_esmda.yaml documents
    # one) has real boundaries to mark. The *parameter* plots are the opposite
    # case: a static parameter's x-axis is a window index, so shading it at
    # ``w * sim_time`` would mark the wrong places, and it correctly gets none.
    from scripts.esmda.make_esmda_figures import make_figures

    recorded = _captured_kwargs(
        monkeypatch, "plot_sensor_fans", "plot_rollout_time_evolution"
    )
    run_dir = _figure_run_dir(tmp_path, is_dynamic=is_dynamic)

    make_figures(run_dir)

    edges = recorded["plot_sensor_fans"]["window_edges"]
    assert edges is not None, "S5 got no window boundaries on a multi-window run"
    assert list(edges) == pytest.approx(  # type: ignore[call-overload]
        [w * _RUN_SIM_TIME for w in range(_RUN_WINDOWS + 1)]
    )
    parameter_edges = recorded["plot_rollout_time_evolution"]["window_edges"]
    assert (parameter_edges is not None) is is_dynamic, (
        "the parameter plots' shading follows the parameter x-axis, which is a "
        "window index on a static run"
    )


@pytest.mark.parametrize("height", [None, 12.5])  # type: ignore[misc]
def test_make_figures_passes_a_configured_building_height_to_the_profiles(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, height: float | None
) -> None:
    # S1's ``z/H`` axis and its roof line need a canopy height, and until a case
    # config carries one they are dead on every real run. Optional, so an absent
    # key must still leave the profiles in metres rather than raise (CLAUDE.md's
    # no-op-when-absent rule).
    from scripts.esmda.make_esmda_figures import make_figures

    recorded = _captured_kwargs(monkeypatch, "plot_station_profiles")
    run_dir = _figure_run_dir(tmp_path, building_height=height)

    make_figures(run_dir)

    assert recorded["plot_station_profiles"]["building_height"] == height
    # ``U_ref`` comes off the truth's ``velocity_magnitude`` on the same call.
    assert recorded["plot_station_profiles"]["u_ref"] == pytest.approx(5.0)


def test_make_figures_drops_the_fan_time_axis_when_the_sensor_sets_disagree(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    # S5 draws every sensor set against ONE x-axis. The sets agree on it today
    # -- they are all extracted from the same window state files -- and the loop
    # that collects it keeps only the last set's. If they ever stopped agreeing,
    # that would draw the other columns against a clock that is not theirs, and
    # (since S5 no longer falls back to a frame index internally) could raise
    # mid-stage and cost every figure after it, which invariant 3 forbids.
    import scripts.esmda.make_esmda_figures as wiring

    recorded = _captured_kwargs(monkeypatch, "plot_sensor_fans")
    aligner = wiring.compute_sensor_metrics
    offsets = iter((0.0, 1000.0))

    def _shifted(truth: object, ensemble: object) -> dict:
        aligned = dict(aligner(truth, ensemble))
        aligned["time"] = aligned["time"] + next(offsets)
        return aligned

    monkeypatch.setattr(wiring, "compute_sensor_metrics", _shifted)
    run_dir = _figure_run_dir(tmp_path)

    wiring.make_figures(run_dir)

    assert (
        recorded["plot_sensor_fans"]["times"] is None
    ), "one set's clock was handed to S5 for all of them"
    assert "frame index" in capsys.readouterr().out
