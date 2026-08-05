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
    # and the z-score are dropped, the latter annotated ``n/a`` rather than
    # silently omitted.
    posterior, prior, _truth = _toy_params()
    out = tmp_path / "p1.png"

    _assert_png(plot_parameter_marginals(posterior, None, out, prior_params=prior), out)
    assert "n/a" in _axes_text(captured_figures[-1])


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


def test_station_profiles_exclude_the_extrapolated_top_level(
    tmp_path: pathlib.Path, captured_figures: list[Figure]
) -> None:
    # Colocation linearly extrapolates the LAST index of every axis it moved,
    # which leaves inflated second moments there. ``extrapolated_edges`` names
    # those axes, and a profile that includes the top cell plots an artefact.
    n_z = 5
    with_edge = _eval_fields(n_z=n_z, extrapolated="z", station_sets=("assimilation",))
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
    assert (
        _highest_plotted_z(dropped) <= penultimate + 1e-6
    ), "the extrapolated top level is still plotted despite extrapolated_edges='z'"


def test_station_profiles_keep_the_validation_columns_when_truncating(
    tmp_path: pathlib.Path, caplog: pytest.LogCaptureFixture
) -> None:
    # More columns than fit: the held-out sensors are the ones a reader came
    # for, so they are kept first -- and the drop is logged, never silent.
    fields = _eval_fields(
        station_sets=("assimilation",) * 4 + ("validation",) * 4, n_z=4
    )
    out = tmp_path / "s1.png"

    with caplog.at_level(logging.INFO, logger="evaluation.figures"):
        result = plot_station_profiles(fields, out, max_stations=3)

    _assert_png(result, out)
    assert caplog.records, "columns were dropped without a word about it"


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
