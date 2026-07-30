"""Correctness tests for finite-ensemble data-assimilation metrics."""

from __future__ import annotations

import ast
import inspect
import logging
import math
import pathlib
import textwrap
from typing import Any

import numpy as np
import pytest
import xarray
from data_assimilation.interpolation import interpolate_dataarray_at_points
from data_assimilation.observation_operator import ObservationOperator
from xarray.backends.netCDF4_ import NetCDF4ArrayWrapper

from pyurbanair.plotting import _crps_ensemble, compute_parameter_metrics
from pyurbanair.utils.da_metrics import (
    ensemble_uniqueness,
    per_knot_crps,
    spread_skill_ratio,
    summary_scalars,
)
from scripts.esmda._esmda_common import (
    _energy_score,
    parameter_metric_summary,
    read_yaml,
    sensor_magnitude,
    write_yaml,
)
from scripts.figspec.metrics import spread_skill
from tests.test_esmda_metrics_wiring import (
    ASSIM_POINTS,
    ENSEMBLE_FRAMES_PER_WINDOW,
    GRID_X,
    GRID_Y,
    GRID_Z,
    N_MEMBERS,
    N_WINDOWS,
    VALIDATION_POINTS,
    _write_state_artifacts,
)


def _normal_crps(y: float, mean: float = 0.0, std: float = 1.0) -> float:
    """Analytic CRPS of a Gaussian forecast at a deterministic observation."""
    z = (y - mean) / std
    phi = math.exp(-(z**2) / 2.0) / math.sqrt(2.0 * math.pi)
    cdf = 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))
    return std * (z * (2.0 * cdf - 1.0) + 2.0 * phi - 1.0 / math.sqrt(math.pi))


def test_fair_crps_matches_analytic_gaussian() -> None:
    rng = np.random.default_rng(42)
    members = rng.normal(size=(10_000, 1))
    truth = np.array([0.7])

    score = per_knot_crps(members, truth)[0]

    assert score == pytest.approx(_normal_crps(float(truth[0])), abs=1e-2)


def test_fair_crps_removes_small_ensemble_upward_bias() -> None:
    rng = np.random.default_rng(7)
    n_members = 8
    n_trials = 20_000
    members = rng.normal(size=(n_members, n_trials))
    truth = rng.normal(size=n_trials)

    fair = float(per_knot_crps(members, truth).mean())
    diffs = np.abs(members[:, None, :] - members[None, :, :])
    biased = float(
        (
            np.mean(np.abs(members - truth[None, :]), axis=0)
            - 0.5 * diffs.mean(axis=(0, 1))
        ).mean()
    )
    population_score = 1.0 / math.sqrt(math.pi)

    assert fair == pytest.approx(population_score, abs=1e-2)
    assert biased > population_score + 0.04


def test_all_fair_score_sites_exclude_the_zero_diagonal() -> None:
    members = np.array([[0.0], [2.0]])
    truth = np.array([0.0])
    # term1 = 1; fair pairwise half-term = (0 + 2 + 2 + 0) / (2*2*1) = 1.
    assert per_knot_crps(members, truth)[0] == pytest.approx(0.0)
    assert _crps_ensemble(members, truth)[0] == pytest.approx(0.0)

    vector_members = members.T[:, :, None, None]  # (component, ensemble, time, sensor)
    vector_truth = truth[:, None, None]  # (component, time, sensor)
    assert _energy_score(vector_members, vector_truth)[0] == pytest.approx(0.0)


def test_single_member_scores_reduce_to_absolute_error() -> None:
    members = np.array([[2.5]])
    truth = np.array([1.0])
    assert per_knot_crps(members, truth)[0] == pytest.approx(1.5)
    assert _crps_ensemble(members, truth)[0] == pytest.approx(1.5)

    vector_members = members.T[:, :, None, None]
    vector_truth = truth[:, None, None]
    assert _energy_score(vector_members, vector_truth)[0] == pytest.approx(1.5)


def test_summary_spread_is_root_mean_variance() -> None:
    members = np.array([[0.0, 0.0], [0.0, 2.0]])
    truth = np.zeros(2)
    expected = math.sqrt(np.mean(np.std(members, axis=0, ddof=1) ** 2))

    assert summary_scalars(members, truth)["time_avg_spread"] == pytest.approx(expected)


def test_spread_skill_fortin_factor_calibrates_exchangeable_ensemble() -> None:
    rng = np.random.default_rng(123)
    n_members = 10
    members = rng.normal(size=(n_members, 100_000))
    truth = rng.normal(size=members.shape[1])
    spread = members.std(axis=0, ddof=1)
    error = members.mean(axis=0) - truth

    corrected = spread_skill(spread, error, n_members)
    uncorrected = float(np.sqrt(np.mean(spread**2)) / np.sqrt(np.mean(error**2)))

    assert corrected == pytest.approx(1.0, abs=1e-2)
    assert uncorrected == pytest.approx(
        math.sqrt(n_members / (n_members + 1)), abs=1e-2
    )


def test_spread_skill_delegates_to_the_shared_ratio() -> None:
    """``figspec.spread_skill`` is an adapter, not a second implementation.

    It must equal ``spread_skill_ratio`` on the squared inputs exactly -- for
    ordinary series, for masked (NaN) ones, and on the degenerate inputs the
    shared function guards.
    """
    rng = np.random.default_rng(4711)
    for shape in [(7,), (100,), (12, 31), (4, 6, 9)]:
        for n_members in (1, 2, 10, 512):
            spread = np.abs(rng.normal(size=shape))
            error = rng.normal(size=shape)
            assert spread_skill(spread, error, n_members) == spread_skill_ratio(
                spread**2, error**2, n_members
            )

            masked_spread = np.where(rng.random(shape) < 0.3, np.nan, spread)
            assert spread_skill(masked_spread, error, n_members) == spread_skill_ratio(
                masked_spread**2, error**2, n_members
            )

    assert np.isnan(spread_skill(np.ones(4), np.zeros(4), 5))  # zero error norm
    assert np.isnan(spread_skill(np.full(4, np.nan), np.ones(4), 5))  # fully masked
    assert np.isnan(spread_skill(np.ones(4), np.full(4, np.nan), 5))
    with pytest.raises(ValueError, match="n_members must be positive"):
        spread_skill(np.ones(4), np.ones(4), 0)


def test_ensemble_uniqueness_detects_exact_clone() -> None:
    members = np.array(
        [
            [0.0, 1.0],
            [2.0, 3.0],
            [2.0, 3.0],
            [4.0, 5.0],
        ]
    )

    health = ensemble_uniqueness(members)

    assert health["n_members"] == 4
    assert health["n_unique"] == 3
    assert health["min_pairwise"] == 0.0
    median = health["median_pairwise"]
    assert median is not None and median > 0.0


def test_parameter_metrics_include_crps_skill_against_prior() -> None:
    coords = {"ensemble": np.arange(3), "time": [0.0, 1.0]}
    posterior = xarray.Dataset(
        {
            "inflow_angle": (
                ("ensemble", "time"),
                [[0.0, 0.0], [0.5, 0.5], [-0.5, -0.5]],
            )
        },
        coords=coords,
    )
    prior = xarray.Dataset(
        {
            "inflow_angle": (
                ("ensemble", "time"),
                [[-2.0, -2.0], [2.0, 2.0], [3.0, 3.0]],
            )
        },
        coords=coords,
    )
    truth = xarray.Dataset(
        {"inflow_angle": (("time",), [0.0, 0.0])},
        coords={"time": coords["time"]},
    )

    metrics = compute_parameter_metrics(posterior, truth, prior)
    summary = parameter_metric_summary(posterior, truth, prior)

    assert "prior_crps" in metrics["inflow_angle"]
    assert summary["inflow_angle"]["prior_crps_mean"] > 0.0
    assert summary["inflow_angle"]["crps_reduction_vs_prior"] > 0.0


def test_skip_viz_summary_has_version_and_ensemble_health(
    tmp_path: pathlib.Path,
) -> None:
    from scripts.esmda.compute_esmda_metrics import compute_metrics

    run_dir = tmp_path / "run"
    windows_dir = run_dir / "windows"
    windows_dir.mkdir(parents=True)
    write_yaml({"run": {"skip_viz": True}}, run_dir / "config.yaml")
    write_yaml({"configuration": {"ensemble_size": 4}}, run_dir / "run_info.yaml")

    coords = {"ensemble": np.arange(4), "time": [0.0, 1.0]}
    posterior = xarray.Dataset(
        {
            "inflow_angle": (
                ("ensemble", "time"),
                [[0.0, 0.0], [1.0, 1.0], [1.0, 1.0], [2.0, 2.0]],
            )
        },
        coords=coords,
    )
    prior = xarray.Dataset(
        {
            "inflow_angle": (
                ("ensemble", "time"),
                [[-2.0, -2.0], [-1.0, -1.0], [1.0, 1.0], [2.0, 2.0]],
            )
        },
        coords=coords,
    )
    truth = xarray.Dataset(
        {"inflow_angle": (("time",), [0.0, 0.0])},
        coords={"time": coords["time"]},
    )
    posterior.to_netcdf(run_dir / "posterior_params.nc")
    posterior.to_netcdf(windows_dir / "window_0_posterior_params.nc")
    prior.to_netcdf(run_dir / "prior_params.nc")
    truth.to_netcdf(run_dir / "true_params.nc")

    compute_metrics(run_dir)

    summary = read_yaml(run_dir / "run_summary.yaml")
    assert summary["metrics_version"] == 2
    assert summary["ensemble_health"] == {
        "n_members": 4,
        "n_unique": 3,
        "n_unique_per_window": [3],
        "min_over_median_pairwise": 0.0,
    }
    assert (
        summary["parameter_metrics"]["inflow_angle"]["crps_reduction_vs_prior"]
        is not None
    )


def test_sweep_comparison_warns_on_mixed_metric_versions(
    tmp_path: pathlib.Path,
) -> None:
    from scripts.figure_creation.compare_sweep_results import load_runs

    for name, version in (
        ("pylbm_nx10_ny10_nz4_ens10_steps2", None),
        ("pylbm_nx10_ny10_nz4_ens20_steps2", 2),
    ):
        run_dir = tmp_path / name
        run_dir.mkdir()
        summary: dict[str, object] = {"configuration": {"assimilation_model": "pylbm"}}
        if version is not None:
            summary["metrics_version"] = version
        write_yaml(summary, run_dir / "run_summary.yaml")

    with pytest.warns(UserWarning, match="mismatched metrics versions"):
        runs = load_runs(tmp_path, models=None)

    assert set(runs["metrics_version"]) == {1, 2}


def _write_param_artifacts(run_dir: pathlib.Path) -> None:
    """The three tiny parameter NetCDFs ``process_run`` reads before the sensors."""
    coords = {"ensemble": [0, 1]}
    xarray.Dataset(
        {"inflow_angle": (("ensemble",), [0.0, 1.0])}, coords=coords
    ).to_netcdf(run_dir / "posterior_params.nc")
    xarray.Dataset(
        {"inflow_angle": (("ensemble",), [-1.0, 2.0])}, coords=coords
    ).to_netcdf(run_dir / "prior_params.nc")
    xarray.Dataset({"inflow_angle": 0.5}).to_netcdf(run_dir / "true_params.nc")


def test_sweep_metrics_omit_legacy_sensor_scores_without_truth_access(
    tmp_path: pathlib.Path,
) -> None:
    from scripts.figure_creation.compute_sweep_metrics import process_run

    run_dir = tmp_path / "legacy_run"
    out_dir = tmp_path / "sweep_metrics"
    run_dir.mkdir()
    write_yaml(
        {
            "configuration": {"assimilation_model": "pylbm"},
            "sensor_metrics": {
                "assimilation": {
                    "vel_magnitude_crps": {"mean": 123.0},
                }
            },
        },
        run_dir / "run_summary.yaml",
    )
    write_yaml(
        {
            "obs": {
                "mode": "points",
                "x_points": [0.0],
                "y_points": [0.0],
                "z_points": [0.0],
            }
        },
        run_dir / "config.yaml",
    )

    _write_param_artifacts(run_dir)

    status = process_run(run_dir, out_dir)
    metrics = read_yaml(out_dir / "metrics.yaml")

    assert metrics["metrics_version"] == 2
    assert "parameter_metrics" in metrics
    assert "sensor_metrics" not in metrics
    assert status["note"] == (
        "no truth_access.yaml -> sensor metrics omitted (re-run ESMDA)"
    )


def test_sweep_metrics_survive_a_window_state_file_with_no_ensemble_dim(
    tmp_path: pathlib.Path, caplog: pytest.LogCaptureFixture
) -> None:
    """An unreadable sensor stage must not delete the run from the sweep.

    ``stream_window_members`` raises ``ValueError`` on a window state file with
    no ``ensemble`` dimension -- unreachable through ``run_esmda``, reachable for
    a hand-made or pre-streaming file. Uncaught, that exception reaches
    ``main``'s per-run ``except``/``continue``, and the run ends with *no*
    ``metrics.yaml``: the comparison script then reads it as "never processed"
    rather than "broken", and the parameter metrics -- which were already
    computed and have nothing to do with the sensors -- are lost with it.

    So: the run is still written, the sensor block degrades to its
    ``num_sensors`` skeleton, and the cause is on the status *and* in the log
    naming the run.
    """
    from scripts.figure_creation.compute_sweep_metrics import process_run

    run_dir = tmp_path / "no_ensemble_dim_run"
    run_dir.mkdir()
    obs_config, truth_access = _write_state_artifacts(
        run_dir, np.random.default_rng(11)
    )
    for window in range(N_WINDOWS):
        path = run_dir / "windows" / f"window_{window}_posterior_state.nc"
        with xarray.open_dataset(path) as ds:
            collapsed = ds.isel(ensemble=0, drop=True).load()
        collapsed.to_netcdf(path)
    write_yaml(obs_config, run_dir / "config.yaml")
    write_yaml(truth_access, run_dir / "truth_access.yaml")
    write_yaml(
        {"configuration": {"assimilation_model": "pylbm"}},
        run_dir / "run_summary.yaml",
    )
    _write_param_artifacts(run_dir)

    out_dir = tmp_path / "sweep_metrics"
    with caplog.at_level(logging.WARNING):
        status = process_run(run_dir, out_dir)
    metrics = read_yaml(out_dir / "metrics.yaml")

    assert "parameter_metrics" in metrics
    assert status["components"] is False
    assert status["sensor_timeseries"] is False
    assert "ValueError" in status["note"] and "ensemble" in status["note"]
    assert run_dir.name in caplog.text and "ensemble" in caplog.text
    # The skeleton, not the scores: a sensor count with no metric under it is
    # what tells the comparison stage the set existed and was not scored.
    assert set(metrics["sensor_metrics"]) == {"assimilation", "validation"}
    assert set(metrics["sensor_metrics"]["assimilation"]) == {"num_sensors"}
    assert not list(out_dir.glob("sensor_timeseries_*.nc"))


def test_split_quantities_rejects_a_component_set_it_cannot_key(
    tmp_path: pathlib.Path,
) -> None:
    """A fourth component must be loud, not dropped.

    Every artifact this stage writes is keyed per component (``QUANTITIES``,
    ``_Q_KEY``, the comparison script's columns) and |U| is a sum over exactly
    three, so silently taking ``u``/``v``/``w`` out of a wider series would drop
    the extra one from the summary and from |U| with nothing to read anywhere.
    """
    from scripts.figure_creation.compute_sweep_metrics import _split_quantities

    values = np.arange(4 * 2 * 3, dtype=float).reshape(4, 2, 3)
    four = xarray.DataArray(
        values,
        dims=("component", "time", "sensor"),
        coords={"component": ["u", "v", "w", "tke"]},
    )
    with pytest.raises(ValueError, match="exactly the components"):
        _split_quantities(four)
    with pytest.raises(ValueError, match="no component coordinate"):
        _split_quantities(four.drop_vars("component"))
    # And the accepted set still round-trips.
    assert set(_split_quantities(four.isel(component=slice(0, 3)))) == {
        "u",
        "v",
        "w",
        "vel",
    }


def test_sweep_velocity_magnitude_is_the_shared_definition() -> None:
    """|U| is ``_esmda_common.sensor_magnitude``, bit for bit.

    Pins the D5 correction: the sweep stage used to keep its own elementwise
    ``sqrt(u**2 + v**2 + w**2)``, on the stated grounds that the shared helper
    inherits the stacked array's ``name`` and ``rename`` could not clear it.
    ``rename()`` with no argument does clear it, so the two definitions are now
    one -- and the reduction over three elements sums in the same order as the
    elementwise chain, which is why the artifacts did not move.
    """
    from scripts.figure_creation.compute_sweep_metrics import _split_quantities

    rng = np.random.default_rng(5)
    stacked = xarray.DataArray(
        rng.normal(size=(3, 2, 4)),
        dims=("component", "time", "sensor"),
        coords={"component": ["u", "v", "w"]},
        name="u",
    )
    quantities = _split_quantities(stacked)
    elementwise = np.sqrt(
        quantities["u"] ** 2 + quantities["v"] ** 2 + quantities["w"] ** 2
    )

    assert quantities["vel"].values.tobytes() == elementwise.values.tobytes()
    assert quantities["vel"].identical(sensor_magnitude(stacked).rename())
    # The name is the only thing the shared helper changes, and it is cleared:
    # nothing downstream reads it, but a stale "u" would make `identical()` lie.
    assert quantities["vel"].name is None
    assert [quantities[q].name for q in ("u", "v", "w")] == ["u", "v", "w"]


# ---------------------------------------------------------------------------
# The sweep stage's sensor extraction, after it was moved onto the streamed read
# ---------------------------------------------------------------------------
#
# `compute_sweep_metrics._ensemble_series` used to be an independent copy of the
# ESMDA stage's extraction, and it still did the full-ensemble
# `xr.open_dataset(path).load()` that WP1.3 removed there. It now delegates to
# `_esmda_common.ensemble_sensor_series` / `truth_sensor_series` and only
# reshapes the result per quantity.
#
# Two things need pinning, and neither is covered by
# `tests/test_esmda_stream_members.py` (which pins the shared helpers, not this
# caller). First, that the sweep artifacts did not move: they feed *cross-run*
# comparisons, which is the whole reason `metrics_version` exists, so "close" is
# not good enough and the reference implementations below are verbatim copies of
# the pre-port code (`git show d7a169b:scripts/figure_creation/
# compute_sweep_metrics.py`, lines 79-190). Second, that the `.load()` is
# actually gone -- phase 1's acceptance criterion is a grep, and a grep cannot
# tell code from the docstrings that explain what was removed.


_REFERENCE_X_COORDS = ("x", "xt", "xm")
_REFERENCE_QUANTITIES = ("u", "v", "w", "vel")


def _reference_open_truth(
    true_state_path: Any,
    n_total: Any,
    x_offset: float = 0.0,
    start_idx: int = 0,
    t_offset: float = 0.0,
) -> Any:
    """``compute_sweep_metrics._open_truth`` as it was. Do not modernise."""
    ds = xarray.open_dataset(true_state_path)
    if n_total is not None:
        ds = ds.isel(time=slice(start_idx, start_idx + n_total))
    elif start_idx:
        ds = ds.isel(time=slice(start_idx, None))
    if t_offset and "time" in ds.coords:
        ds = ds.assign_coords(time=ds["time"] - t_offset)
    if x_offset:
        shifted = {c: ds[c] + x_offset for c in _REFERENCE_X_COORDS if c in ds.coords}
        if shifted:
            ds = ds.assign_coords(shifted)
    return ds


def _reference_sensor_components(
    state: Any, obs_x: Any, obs_y: Any, obs_z: Any, solver_name: str
) -> dict[str, Any]:
    """``compute_sweep_metrics._sensor_components`` as it was. Do not modernise."""
    op = ObservationOperator(
        obs_x=list(np.asarray(obs_x, dtype=float)),
        obs_y=list(np.asarray(obs_y, dtype=float)),
        obs_z=list(np.asarray(obs_z, dtype=float)),
        obs_states=["u", "v", "w"],
        solver_name=solver_name,
    )
    comps = {}
    for var in ("u", "v", "w"):
        dims = op.dim_mapping[var]
        comps[var] = interpolate_dataarray_at_points(
            state[var],
            x_dim=dims["x"],
            y_dim=dims["y"],
            z_dim=dims["z"],
            obs_x=op.obs_x,
            obs_y=op.obs_y,
            obs_z=op.obs_z,
        )
    comps["vel"] = np.sqrt(comps["u"] ** 2 + comps["v"] ** 2 + comps["w"] ** 2)
    return comps


def _reference_concat(parts: list[Any]) -> Any:
    return (
        parts[0]
        if len(parts) == 1
        else xarray.concat(parts, dim="time", join="override")
    )


def _reference_ensemble_series(
    state_paths: list[pathlib.Path],
    sensor_sets: dict[str, Any],
    solver_name: str,
    sim_time: float,
) -> Any:
    """``compute_sweep_metrics._ensemble_series`` as it was, ``.load()`` included."""
    if not all(p.exists() for p in state_paths):
        return None
    pieces: dict[str, dict[str, list[Any]]] = {
        name: {q: [] for q in _REFERENCE_QUANTITIES} for name in sensor_sets
    }
    for w, path in enumerate(state_paths):
        ds = xarray.open_dataset(path).load()
        t = np.asarray(ds["time"].values, dtype=float) if "time" in ds.coords else None
        for name, (ox, oy, oz) in sensor_sets.items():
            fields = _reference_sensor_components(ds, ox, oy, oz, solver_name)
            for q, da in fields.items():
                if t is not None and "time" in da.dims:
                    da = da.assign_coords(time=(t - t[0]) + w * sim_time)
                pieces[name][q].append(da)
        ds.close()
    return {
        n: {q: _reference_concat(parts) for q, parts in by_q.items()}
        for n, by_q in pieces.items()
    }


def _reference_truth_series(
    ta: dict[str, Any], sensor_sets: dict[str, Any], solver_name: str
) -> Any:
    """``compute_sweep_metrics._truth_series`` as it was. Do not modernise."""
    pieces: dict[str, dict[str, list[Any]]] = {
        name: {q: [] for q in _REFERENCE_QUANTITIES} for name in sensor_sets
    }
    for w in range(ta["num_windows"]):
        ts = _reference_open_truth(
            ta["true_state_path"],
            ta["n_total"],
            ta["x_offset"],
            ta["start_idx"],
            ta["t_offset"],
        ).isel(time=slice(w * ta["n_per_window"], (w + 1) * ta["n_per_window"]))
        for name, (ox, oy, oz) in sensor_sets.items():
            fields = _reference_sensor_components(ts, ox, oy, oz, solver_name)
            for q, da in fields.items():
                pieces[name][q].append(da)
        ts.close()
    return {
        n: {q: _reference_concat(parts) for q, parts in by_q.items()}
        for n, by_q in pieces.items()
    }


SWEEP_SENSOR_SETS = {"assimilation": ASSIM_POINTS, "validation": VALIDATION_POINTS}


@pytest.fixture  # type: ignore[misc]
def sweep_run(tmp_path: pathlib.Path) -> tuple[pathlib.Path, dict[str, Any]]:
    """The multi-window state artifacts plus the ``truth_access`` keys for them."""
    run_dir = tmp_path / "sweep_run"
    run_dir.mkdir()
    _obs, truth_access = _write_state_artifacts(
        run_dir, np.random.default_rng(11), with_prior=True
    )
    truth_access["num_windows"] = N_WINDOWS
    return run_dir, truth_access


def _window_paths(run_dir: pathlib.Path, kind: str) -> list[pathlib.Path]:
    return [
        run_dir / "windows" / f"window_{w}_{kind}_state.nc" for w in range(N_WINDOWS)
    ]


def _assert_identical_quantities(new: Any, old: Any) -> None:
    """Same sets, same numbers, same coords/name -- and the same memory layout.

    The layout is asserted through the reductions rather than only through the
    strides because the reductions are what the artifact is made of: numpy walks
    a reduction in memory order and float addition is not associative, so a
    re-laid-out buffer moves the sweep's CRPS entries in their last digits.
    Measured while porting this: forcing the per-quantity arrays C-contiguous
    moved 16 of ``metrics.yaml``'s 92 leaves, by at most 2.1e-15 relative.
    """
    assert set(new) == set(old)
    for name in new:
        assert set(new[name]) == set(old[name]) == set(_REFERENCE_QUANTITIES)
        for q, a in new[name].items():
            b = old[name][q]
            assert a.dims == b.dims, (name, q, a.dims, b.dims)
            assert np.array_equal(a.values, b.values, equal_nan=True), (name, q)
            assert a.values.tobytes() == b.values.tobytes(), (name, q)
            # Coordinates too: the sweep aligns truth against ensemble by time
            # *value*, so the right numbers on a shifted axis are still wrong.
            assert a.identical(b), (name, q)
            for axes in (("time",), ("sensor",)):
                if all(d in a.dims for d in axes):
                    assert np.array_equal(a.sum(axes).values, b.sum(axes).values), (
                        name,
                        q,
                        axes,
                    )
            if "ensemble" in a.dims:
                assert np.array_equal(
                    a.sum(("ensemble", "time")).values,
                    b.sum(("ensemble", "time")).values,
                ), (name, q)


def test_sweep_ensemble_series_is_bit_identical_to_the_preload_version(
    sweep_run: tuple[pathlib.Path, dict[str, Any]],
) -> None:
    from scripts.figure_creation.compute_sweep_metrics import _ensemble_series

    run_dir, ta = sweep_run
    for kind in ("posterior", "prior"):
        paths = _window_paths(run_dir, kind)
        new = _ensemble_series(paths, SWEEP_SENSOR_SETS, "pylbm", ta["sim_time"])
        old = _reference_ensemble_series(
            paths, SWEEP_SENSOR_SETS, "pylbm", ta["sim_time"]
        )
        _assert_identical_quantities(new, old)
        # Guard the guard: a fixture whose members were all equal, or whose
        # windows all started at 0, would make the comparison vacuous.
        series = new["assimilation"]["vel"]
        assert series.dims == ("ensemble", "time", "sensor")
        assert float(series.std("ensemble").max()) > 0.0
        assert float(series["time"].max()) > ta["sim_time"]


def test_sweep_truth_series_is_bit_identical_to_the_preload_version(
    sweep_run: tuple[pathlib.Path, dict[str, Any]],
) -> None:
    from scripts.figure_creation.compute_sweep_metrics import _truth_series

    run_dir, ta = sweep_run
    new = _truth_series(ta, SWEEP_SENSOR_SETS, ta["truth_solver_name"])
    old = _reference_truth_series(ta, SWEEP_SENSOR_SETS, ta["truth_solver_name"])
    _assert_identical_quantities(new, old)
    # The truth path's own contract: `x_offset` / `start_idx` / `t_offset` are
    # applied, so a truth saved in its own frame and with a spin-up still lines
    # up with the ensemble's rebased axis.
    assert float(new["assimilation"]["vel"]["time"].min()) == pytest.approx(0.0)


def test_sweep_ensemble_series_is_none_when_a_window_file_is_missing(
    sweep_run: tuple[pathlib.Path, dict[str, Any]],
) -> None:
    """The pre-`save_prior_state` runs the module docstring promises to tolerate."""
    from scripts.figure_creation.compute_sweep_metrics import _ensemble_series

    run_dir, ta = sweep_run
    paths = _window_paths(run_dir, "posterior")
    paths[-1] = paths[-1].with_name("window_absent_state.nc")
    assert _ensemble_series(paths, SWEEP_SENSOR_SETS, "pylbm", ta["sim_time"]) is None


def test_sweep_ensemble_series_never_reads_a_whole_ensemble(
    sweep_run: tuple[pathlib.Path, dict[str, Any]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The invariant, observed on the reads rather than inferred from the code.

    Counted through ``NetCDF4ArrayWrapper._getitem``, the single funnel every
    lazily-indexed netCDF4-backed read passes through. What it proves is that no
    single read pulled more than one member's velocity fields; it says nothing
    about process RSS, which the OS page cache and NumPy's allocator put out of
    view.
    """
    from scripts.figure_creation.compute_sweep_metrics import _ensemble_series

    run_dir, ta = sweep_run
    member_elements = (
        ENSEMBLE_FRAMES_PER_WINDOW * GRID_Z.size * GRID_Y.size * GRID_X.size
    )
    reads: list[tuple[str, int]] = []
    original = NetCDF4ArrayWrapper._getitem

    def counting_getitem(wrapper: Any, key: Any) -> Any:
        out = original(wrapper, key)
        reads.append((wrapper.variable_name, int(np.size(out))))
        return out

    monkeypatch.setattr(NetCDF4ArrayWrapper, "_getitem", counting_getitem)

    _ensemble_series(
        _window_paths(run_dir, "posterior"),
        SWEEP_SENSOR_SETS,
        "pylbm",
        ta["sim_time"],
    )

    velocity_reads = [(n, s) for n, s in reads if n in ("u", "v", "w")]
    assert velocity_reads, "nothing was read -- the counter is not wired up"
    assert max(s for _n, s in velocity_reads) <= member_elements
    # And the pass is not paying for the streaming with repeat reads: each
    # member's three components, once, per window.
    assert sum(s for _n, s in velocity_reads) == (
        3 * N_WINDOWS * N_MEMBERS * member_elements
    )


def test_sweep_metrics_module_no_longer_loads_a_window_state_file() -> None:
    """Pins phase 1's acceptance grep for this module, on the AST not the text.

    Parsed rather than grepped so the ``.load()`` mentions in the module's own
    docstrings -- which exist to record what was removed -- cannot make it pass
    or fail for the wrong reason.

    ``OmegaConf.load`` is allowed by name rather than by pattern: it reads the
    run's ``config.yaml``, and the invariant is about the multi-GB *state*
    files. Whitelisting the one safe receiver keeps the check failing on
    ``xr.open_dataset(...).load()`` however it is spelled, which a
    "``.load()`` on an ``xr`` call" rule would not.
    """
    from scripts.figure_creation import compute_sweep_metrics

    tree = ast.parse(textwrap.dedent(inspect.getsource(compute_sweep_metrics)))
    allowed_receivers = {"OmegaConf", "yaml"}
    offenders = [
        ast.unparse(node)
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr in ("load", "load_dataset")
        and not (
            isinstance(node.func.value, ast.Name)
            and node.func.value.id in allowed_receivers
        )
    ]
    assert offenders == []
