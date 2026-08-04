"""WP1.4: streaming mean fields, colocation and the hit rate.

Three layers, in the order the metric stage uses them: the moment accumulator
that turns streamed frames into time-mean velocity / TKE / ``<u'w'>``, the
colocation that puts the components on one grid first, and the VDI 3783/9 hit
rate that scores the mean field against the truth. The last section drives the
whole thing through ``compute_esmda_metrics`` on a synthetic run dir, because
the e2e ESMDA tests all run under ``run.skip_viz`` and never reach this block.

Specification: ``docs/plans/esmda_evaluation/phase1_metrics_and_figures.md``
(WP1.4); metric definitions: ``docs/plans/esmda_turbulence_evaluation.md`` §4.1.
"""

from __future__ import annotations

import pathlib

import numpy as np
import pytest
import xarray
from evaluation.scores import hit_rate
from evaluation.turbulence import (
    MomentAccumulator,
    colocate_components,
    evenly_spaced_levels,
)

# ---------------------------------------------------------------------------
# MomentAccumulator
# ---------------------------------------------------------------------------


def _chunks(values: np.ndarray, size: int) -> list[np.ndarray]:
    return [values[i : i + size] for i in range(0, values.shape[0], size)]


def test_accumulator_matches_a_direct_mean_and_variance() -> None:
    # The whole point of streaming is that it changes nothing but the memory.
    rng = np.random.default_rng(0)
    u, v, w = (rng.normal(size=(48, 3, 4)) + 5.0 for _ in range(3))

    accumulator = MomentAccumulator()
    for cu, cv, cw in zip(_chunks(u, 7), _chunks(v, 7), _chunks(w, 7)):
        accumulator.update(cu, cv, cw)

    assert np.allclose(accumulator.mean()[0], u.mean(axis=0))
    assert np.allclose(accumulator.mean()[2], w.mean(axis=0))
    assert np.allclose(accumulator.reynolds_stress()[0, 0], u.var(axis=0, ddof=1))
    assert np.allclose(accumulator.count(), 48)


def test_accumulator_tke_is_half_the_summed_component_variances() -> None:
    # ``tke`` bypasses ``reynolds_stress`` for cost, so the identity is worth
    # pinning: it is what makes the field TKE and WP1.3's sensor TKE the same
    # estimator.
    rng = np.random.default_rng(1)
    fields = [rng.normal(size=(30, 5)) * scale for scale in (1.0, 2.0, 0.5)]

    accumulator = MomentAccumulator()
    for i in range(0, 30, 6):
        accumulator.update(*(f[i : i + 6] for f in fields))

    expected = 0.5 * sum(f.var(axis=0, ddof=1) for f in fields)
    assert np.allclose(accumulator.tke(), expected)


def test_accumulator_cross_moment_is_the_uw_covariance() -> None:
    # ``<u'w'>`` is entry [0, 2]; the metrics doc singles it out because it
    # discriminates anisotropy TKE alone cannot.
    rng = np.random.default_rng(2)
    u = rng.normal(size=(40, 2))
    w = 0.6 * u + rng.normal(size=(40, 2))

    accumulator = MomentAccumulator()
    accumulator.update(u, u, w)
    stress = accumulator.reynolds_stress()

    expected = ((u - u.mean(axis=0)) * (w - w.mean(axis=0))).sum(axis=0) / 39
    assert np.allclose(stress[0, 2], expected)
    assert np.allclose(stress[0, 2], stress[2, 0])  # symmetric by construction


def test_accumulator_survives_the_offset_that_defeats_the_naive_form() -> None:
    # A field with a large offset relative to its fluctuation is not
    # pathological -- it is urban flow -- and ``sum uu - (sum u)**2/n`` loses
    # the variance entirely there. This is the whole argument for the Chan
    # combine, so it is a test and not just a docstring.
    rng = np.random.default_rng(3)
    series = 1e4 + 1e-3 * rng.normal(size=(360, 1))
    chunks = _chunks(series, 36)

    accumulator = MomentAccumulator()
    for chunk in chunks:
        accumulator.update(chunk)

    exact = float(
        (
            (
                np.asarray(series, dtype=np.longdouble)
                - np.asarray(series, dtype=np.longdouble).mean()
            )
            ** 2
        ).sum()
        / (series.shape[0] - 1)
    )
    n = sum(c.shape[0] for c in chunks)
    total = sum(c.sum() for c in chunks)
    naive = (sum((c**2).sum() for c in chunks) - total**2 / n) / (n - 1)

    streamed_error = abs(float(accumulator.reynolds_stress()[0, 0][0]) - exact) / exact
    naive_error = abs(float(naive) - exact) / exact
    assert streamed_error < 1e-8
    assert naive_error > 1e-3  # measured ~4e-2


def test_accumulator_counts_per_cell_and_deletes_casewise() -> None:
    # A masked cell must cost its own samples and nothing else, and one
    # component's gap must take the *frame* out of every co-moment at that cell
    # -- otherwise ``R`` mixes sample sets and can go indefinite.
    values = np.ones((4, 2))
    other = np.ones((4, 2)) * 2.0
    other[1, 0] = np.nan

    accumulator = MomentAccumulator()
    accumulator.update(values, other)

    assert list(accumulator.count()) == [3, 4]
    assert np.allclose(accumulator.mean()[0], [1.0, 1.0])  # the nan frame is gone


def test_accumulator_reports_nan_where_no_frame_was_finite() -> None:
    field = np.ones((3, 2))
    field[:, 1] = np.nan

    accumulator = MomentAccumulator()
    accumulator.update(field, field, field)

    assert accumulator.count()[1] == 0
    assert np.isnan(accumulator.mean()[0, 1])
    assert np.isnan(accumulator.tke()[1])


def test_accumulator_rejects_a_cell_shape_that_changed_between_chunks() -> None:
    # The failure this catches is a caller that re-selected its z-levels
    # mid-pass, which would otherwise average two different regions together.
    accumulator = MomentAccumulator()
    accumulator.update(np.ones((2, 3)), np.ones((2, 3)))
    with pytest.raises(ValueError, match="fixed by the first update"):
        accumulator.update(np.ones((2, 4)), np.ones((2, 4)))


def test_accumulator_refuses_to_report_before_it_has_seen_anything() -> None:
    with pytest.raises(ValueError, match="no frames accumulated"):
        MomentAccumulator().mean()


def test_accumulator_ignores_an_empty_chunk() -> None:
    # A window whose frames were all filtered away must not disturb the run's
    # totals (invariant 3: it costs its own frames, not the statistic).
    accumulator = MomentAccumulator()
    accumulator.update(np.ones((3, 2)))
    accumulator.update(np.ones((0, 2)))
    assert list(accumulator.count()) == [3, 3]


def test_evenly_spaced_levels_span_the_column_and_never_repeat() -> None:
    assert list(evenly_spaced_levels(10, 4)) == [0, 3, 6, 9]
    # Fewer levels than asked for: deduplicated rather than repeated, so the
    # slab dim always matches the number of distinct heights stored.
    assert list(evenly_spaced_levels(2, 4)) == [0, 1]


# ---------------------------------------------------------------------------
# Colocation
# ---------------------------------------------------------------------------


def _udales_state(n: int = 5, gradient: float = 0.1) -> xarray.Dataset:
    """A uDALES-staggered state whose ``u`` is linear in x, ``w`` linear in z."""
    faces = np.arange(n, dtype=float)
    centres = faces + 0.5
    u = np.broadcast_to(faces[None, None, None, :], (1, n, n, n)) * gradient
    w = np.broadcast_to(faces[None, :, None, None], (1, n, n, n)) * gradient
    return xarray.Dataset(
        {
            "u": (("time", "zt", "yt", "xm"), u.copy()),
            "v": (("time", "zt", "ym", "xt"), np.zeros((1, n, n, n))),
            "w": (("time", "zm", "yt", "xt"), w.copy()),
        },
        coords={
            "time": [0.0],
            "zt": centres,
            "zm": faces,
            "yt": centres,
            "ym": faces,
            "xt": centres,
            "xm": faces,
        },
    )


def test_colocation_is_a_pass_through_on_an_unstaggered_grid() -> None:
    axis = np.linspace(0.0, 4.0, 5)
    ds = xarray.Dataset(
        {n: (("z", "y", "x"), np.arange(125.0).reshape(5, 5, 5)) for n in "uvw"},
        coords={"z": axis, "y": axis, "x": axis},
    )
    for moved, name in zip(colocate_components(ds, "pylbm"), "uvw"):
        assert np.array_equal(moved.values, ds[name].values)


def test_colocation_moves_faces_to_centres_exactly_on_a_linear_field() -> None:
    # The mean-field half of the argument: by-index reports U half a cell off
    # (an error of 0.5*dx*dU/dx, first order and not averaged away), colocation
    # is exact.
    ds = _udales_state()
    u, _, w = colocate_components(ds, "udales")

    assert u.dims == ("time", "zt", "yt", "xt")
    # u(x) = 0.1*x on the faces -> 0.1*(i + 0.5) at the centres.
    assert np.allclose(
        u.isel(time=0, zt=0, yt=0).values[:-1], 0.1 * (np.arange(4) + 0.5)
    )
    assert np.allclose(
        w.isel(time=0, yt=0, xt=0).values[:-1], 0.1 * (np.arange(4) + 0.5)
    )
    assert np.array_equal(u["xt"].values, ds["xt"].values)


def test_colocation_extrapolates_the_last_column_rather_than_dropping_it() -> None:
    # Dropping it would silently shorten the returned grid, which every
    # consumer then has to re-derive; the cost is a column whose second moments
    # are inflated, documented on the function.
    ds = _udales_state()
    u = colocate_components(ds, "udales")[0]

    faces = ds["u"].isel(time=0, zt=0, yt=0).values
    assert u.sizes["xt"] == ds.sizes["xm"]
    assert np.isclose(
        float(u.isel(time=0, zt=0, yt=0).values[-1]), 1.5 * faces[-1] - 0.5 * faces[-2]
    )


def test_colocation_refuses_a_state_that_was_subset_before_colocating() -> None:
    # Interpolating between two z-levels five cells apart is arithmetically
    # fine and physically meaningless.
    ds = _udales_state(n=6).isel(zm=[0, 2, 4], zt=[0, 2, 4])
    with pytest.raises(ValueError, match="not the midpoints"):
        colocate_components(ds, "udales")


def test_colocation_skips_a_staggered_axis_the_state_does_not_carry() -> None:
    # pypalm's postprocess already interpolates w from zw onto z, so the table
    # entry has to be a no-op on a post-processed state.
    axis = np.linspace(0.0, 4.0, 5)
    faces = axis - 0.5
    ds = xarray.Dataset(
        {
            "u": (("z", "y", "xu"), np.zeros((5, 5, 5))),
            "v": (("z", "yv", "x"), np.zeros((5, 5, 5))),
            "w": (("z", "y", "x"), np.arange(125.0).reshape(5, 5, 5)),
        },
        coords={"z": axis, "y": axis, "x": axis, "xu": faces, "yv": faces},
    )
    w = colocate_components(ds, "palm")[2]
    assert np.array_equal(w.values, ds["w"].values)


def test_colocation_rejects_an_unknown_solver() -> None:
    with pytest.raises(ValueError, match="unknown solver"):
        colocate_components(_udales_state(), "not_a_solver")


# ---------------------------------------------------------------------------
# Hit rate
# ---------------------------------------------------------------------------


def test_hit_rate_counts_a_point_within_the_relative_tolerance() -> None:
    observed = np.array([1.0, 1.0, 1.0, 1.0])
    predicted = np.array([1.2, 1.3, 0.8, 0.7])  # 20%, 30%, 20%, 30%
    assert hit_rate(predicted, observed)["q"] == 0.5


def test_hit_rate_absolute_tolerance_rescues_a_near_zero_reference() -> None:
    # The reason the criterion is an ``or``: a relative test alone is
    # unreachable wherever the observed velocity passes through zero, which in
    # a canopy is every recirculation boundary.
    observed = np.array([0.0, 0.0])
    predicted = np.array([0.05, 0.5])
    assert hit_rate(predicted, observed)["q"] == 0.0
    assert hit_rate(predicted, observed, tolerance_w=0.1)["q"] == 0.5


def test_hit_rate_broadcasts_a_per_component_tolerance() -> None:
    # One call scores a stacked (component, ...) field, each component against
    # its own sampling floor.
    observed = np.zeros((2, 2))
    predicted = np.array([[0.05, 0.05], [0.05, 0.05]])
    scored = hit_rate(predicted, observed, tolerance_w=np.array([0.1, 0.0])[:, None])
    assert scored["q"] == 0.5
    assert scored["n_points"] == 4


def test_hit_rate_excludes_masked_points_from_both_halves() -> None:
    # A solid cell (or an edge an interpolated truth could not reach) is
    # neither a hit nor a miss; counting it either way biases q.
    observed = np.array([1.0, np.nan, 1.0])
    predicted = np.array([1.0, 1.0, 5.0])
    scored = hit_rate(predicted, observed)
    assert scored["n_points"] == 2
    assert scored["q"] == 0.5


def test_hit_rate_is_null_rather_than_zero_when_nothing_is_scorable() -> None:
    scored = hit_rate(np.full(3, np.nan), np.ones(3))
    assert scored == {"q": None, "n_points": 0}


def test_hit_rate_falls_back_to_the_relative_test_on_an_unmeasured_floor() -> None:
    # ``W`` is ``nan`` whenever the run was too short to block-bootstrap one --
    # routine, including in CI -- and that must not disqualify every point.
    observed = np.array([1.0, 1.0])
    predicted = np.array([1.1, 2.0])
    assert hit_rate(predicted, observed, tolerance_w=np.nan)["q"] == 0.5


def test_hit_rate_rejects_a_negative_tolerance() -> None:
    with pytest.raises(ValueError):
        hit_rate(np.ones(2), np.ones(2), tolerance_d=-0.1)
    with pytest.raises(ValueError):
        hit_rate(np.ones(2), np.ones(2), tolerance_w=-1.0)


# ---------------------------------------------------------------------------
# The collector, on the shared streaming pass
# ---------------------------------------------------------------------------

_SENSOR_SETS = {"assimilation": (np.array([4.0]), np.array([8.0]), np.array([6.0]))}


def _member_state(
    path: pathlib.Path, n_ensemble: int, n_time: int, seed: int, n_cells: int = 5
) -> pathlib.Path:
    rng = np.random.default_rng(seed)
    axis = np.linspace(0.0, 20.0, n_cells)
    shape = (n_ensemble, n_time, n_cells, n_cells, n_cells)
    xarray.Dataset(
        {
            name: (("ensemble", "time", "z", "y", "x"), rng.normal(size=shape) + 3.0)
            for name in ("u", "v", "w")
        },
        coords={
            "ensemble": np.arange(n_ensemble),
            "time": np.arange(n_time, dtype=float),
            "z": axis,
            "y": axis,
            "x": axis,
        },
    ).to_netcdf(path)
    return path


def test_collector_matches_a_direct_time_mean_over_both_windows(
    tmp_path: pathlib.Path,
) -> None:
    # The statistics span windows: a member's mean field covers every frame of
    # every window it was fed, which is what makes it comparable with a truth
    # pass over the same range.
    from scripts.esmda._esmda_common import MeanFieldCollector, ensemble_sensor_series

    paths = [
        _member_state(tmp_path / f"window_{w}_posterior_state.nc", 3, 6, seed=w)
        for w in range(2)
    ]
    collector = MeanFieldCollector("pylbm", *_SENSOR_SETS["assimilation"][:2], 3)
    ensemble_sensor_series(
        paths, _SENSOR_SETS, "pylbm", 6.0, on_member=collector.add_member
    )
    fields = collector.result()

    both = xarray.concat(
        [xarray.open_dataset(p) for p in paths], dim="time", join="override"
    )
    z_index = evenly_spaced_levels(5, 4)
    expected = both["u"].isel(ensemble=1, z=z_index).mean("time").values

    assert fields["n_members"] == 3
    assert fields["n_windows"] == 2
    assert fields["frames_per_member"] == [12, 12, 12]
    assert np.allclose(fields["slab_mean"][1, 0], expected)


def test_collector_rides_along_without_changing_the_sensor_series(
    tmp_path: pathlib.Path,
) -> None:
    # The callback exists so the mean fields cost no second read of the window
    # files; it must not perturb what that pass returns.
    from scripts.esmda._esmda_common import MeanFieldCollector, ensemble_sensor_series

    paths = [
        _member_state(tmp_path / f"window_{w}_posterior_state.nc", 2, 4, seed=10 + w)
        for w in range(2)
    ]
    plain = ensemble_sensor_series(paths, _SENSOR_SETS, "pylbm", 4.0)["assimilation"]
    collector = MeanFieldCollector("pylbm", *_SENSOR_SETS["assimilation"][:2], 2)
    with_fields = ensemble_sensor_series(
        paths, _SENSOR_SETS, "pylbm", 4.0, on_member=collector.add_member
    )["assimilation"]

    assert np.array_equal(plain.values, with_fields.values)
    assert plain.dims == with_fields.dims
    assert collector.result()["n_members"] == 2


def test_collector_puts_a_foreign_truth_grid_onto_the_assimilation_one(
    tmp_path: pathlib.Path,
) -> None:
    # Cross-grid runs are the default (a PALM truth against a surrogate
    # ensemble), so the truth is interpolated onto the assimilation grid before
    # anything is scored on it.
    from scripts.esmda._esmda_common import MeanFieldCollector, ensemble_sensor_series

    paths = [_member_state(tmp_path / "window_0_posterior_state.nc", 2, 4, seed=1)]
    collector = MeanFieldCollector("pylbm", *_SENSOR_SETS["assimilation"][:2], 2)
    ensemble_sensor_series(
        paths, _SENSOR_SETS, "pylbm", 4.0, on_member=collector.add_member
    )

    # A twice-as-fine truth carrying an exactly linear field: interpolation is
    # then exact, so any difference is a grid-handling bug rather than error.
    fine = np.linspace(0.0, 20.0, 9)
    field = np.broadcast_to(fine[None, None, None, :], (4, 9, 9, 9))
    truth = xarray.Dataset(
        {name: (("time", "z", "y", "x"), field.copy()) for name in ("u", "v", "w")},
        coords={"time": np.arange(4.0), "z": fine, "y": fine, "x": fine},
    )
    truth_collector = MeanFieldCollector(
        "pylbm",
        *_SENSOR_SETS["assimilation"][:2],
        target=collector.target,
        keep_samples=True,
    )
    truth_collector.add(0, 0, truth)
    fields = truth_collector.result()

    # One "member" (the truth), on the ensemble's cells.
    assert fields["slab_mean"].shape[1:] == collector.result()["slab_mean"].shape[1:]
    # u = x on the assimilation grid's own x coordinates.
    assert np.allclose(fields["slab_mean"][0, 0, 0, 0], collector.target["x"])


# ---------------------------------------------------------------------------
# run_summary.yaml / eval_fields.nc wiring
# ---------------------------------------------------------------------------

_RUN_WINDOWS = 2
_RUN_SIM_TIME = 4.0
_RUN_FRAMES = 30  # enough frames per window to block-bootstrap a floor


def _state_dataset(
    n_ensemble: int | None, n_time: int, seed: int, n_cells: int = 5
) -> xarray.Dataset:
    rng = np.random.default_rng(seed)
    axis = np.linspace(0.0, 20.0, n_cells)
    dims: tuple[str, ...] = ("time", "z", "y", "x")
    shape: tuple[int, ...] = (n_time, n_cells, n_cells, n_cells)
    coords = {
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


def _full_run_dir(
    tmp_path: pathlib.Path, n_members: int = 4, save_prior_state: bool = False
) -> pathlib.Path:
    """An ESMDA run dir complete enough for the non-``skip_viz`` metric path."""
    from scripts.esmda._esmda_common import write_yaml

    run_dir = tmp_path / "run"
    (run_dir / "windows").mkdir(parents=True)
    write_yaml(
        {
            "run": {"skip_viz": False},
            "obs": {
                "mode": "points",
                "x_points": [4.0, 12.0],
                "y_points": [5.0, 13.0],
                "z_points": [6.0, 7.0],
            },
        },
        run_dir / "config.yaml",
    )
    write_yaml(
        {"configuration": {"ensemble_size": n_members}}, run_dir / "run_info.yaml"
    )

    truth = _state_dataset(None, _RUN_WINDOWS * _RUN_FRAMES, seed=1).assign_coords(
        time=np.linspace(
            0.0,
            _RUN_WINDOWS * _RUN_SIM_TIME,
            _RUN_WINDOWS * _RUN_FRAMES,
            endpoint=False,
        )
    )
    truth.to_netcdf(run_dir / "truth_state.nc")
    write_yaml(
        {
            "true_state_path": str(run_dir / "truth_state.nc"),
            "n_total": _RUN_WINDOWS * _RUN_FRAMES,
            "x_offset": 0.0,
            "start_idx": 0,
            "t_offset": 0.0,
            "num_windows": _RUN_WINDOWS,
            "n_per_window": _RUN_FRAMES,
            "sim_time": _RUN_SIM_TIME,
            "truth_solver_name": "pylbm",
            "assim_solver_name": "pylbm",
        },
        run_dir / "truth_access.yaml",
    )

    for w in range(_RUN_WINDOWS):
        _state_dataset(n_members, _RUN_FRAMES, seed=10 + w).to_netcdf(
            run_dir / "windows" / f"window_{w}_posterior_state.nc"
        )
        if save_prior_state:
            _state_dataset(n_members, _RUN_FRAMES, seed=50 + w).to_netcdf(
                run_dir / "windows" / f"window_{w}_prior_state.nc"
            )
    _state_dataset(None, _RUN_WINDOWS * _RUN_FRAMES, seed=20).to_netcdf(
        run_dir / "posterior_state_mean.nc"
    )

    members = np.linspace(-1.0, 1.0, n_members)
    for name, values in (
        ("posterior_params", members),
        ("prior_params", members * 4.0),
    ):
        xarray.Dataset(
            {"inflow_angle": (("ensemble", "time"), np.repeat(values[:, None], 2, 1))},
            coords={"ensemble": np.arange(n_members), "time": [0.0, 1.0]},
        ).to_netcdf(run_dir / f"{name}.nc")
    xarray.Dataset(
        {"inflow_angle": (("time",), [0.0, 0.0])}, coords={"time": [0.0, 1.0]}
    ).to_netcdf(run_dir / "true_params.nc")
    return run_dir


def test_run_summary_carries_the_field_metrics_and_writes_eval_fields(
    tmp_path: pathlib.Path,
) -> None:
    from scripts.esmda._esmda_common import read_yaml
    from scripts.esmda.compute_esmda_metrics import compute_metrics

    run_dir = _full_run_dir(tmp_path)
    compute_metrics(run_dir)
    summary = read_yaml(run_dir / "run_summary.yaml")

    block = summary["field_metrics"]
    assert block["n_windows"] == _RUN_WINDOWS
    assert block["frames_per_member"] == [_RUN_WINDOWS * _RUN_FRAMES] * 4
    assert len(block["z_levels"]) == 4
    assert 0.0 <= block["hit_rate_posterior"]["q"] <= 1.0
    assert block["hit_rate_posterior"]["n_points"] == 3 * 4 * 5 * 5
    # A 60-frame truth blocks fine, so W is a measured floor rather than a null.
    assert all(v > 0 for v in block["hit_rate_tolerance_w"].values())
    # Prior states are off by default, so there is no prior half to score.
    assert "hit_rate_prior" not in block

    fields = xarray.open_dataset(run_dir / "eval_fields.nc")
    assert set(fields["component"].values) == {"u", "v", "w"}
    assert fields["truth_slab_mean"].dims == ("component", "zlev", "y", "x")
    assert fields["posterior_station_mean_quantile"].sizes["quantile"] == 5
    assert fields["posterior_slab_mean"].dtype == np.float32
    # Reductions only: no per-member field survives into the artifact.
    assert all("ensemble" not in v.dims for v in fields.data_vars.values())
    fields.close()

    # Additive: the WP1.1-1.3 blocks are untouched.
    assert summary["metrics_version"] == 2
    assert summary["parameter_metrics"] and summary["sensor_statistics"]


def _constant_state(
    n_ensemble: int | None, n_time: int, values: dict[str, float], n_cells: int = 5
) -> xarray.Dataset:
    axis = np.linspace(0.0, 20.0, n_cells)
    dims: tuple[str, ...] = ("time", "z", "y", "x")
    shape: tuple[int, ...] = (n_time, n_cells, n_cells, n_cells)
    coords = {
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
        {n: (dims, np.full(shape, v)) for n, v in values.items()}, coords=coords
    )


def test_field_metrics_score_each_component_against_its_own_truth(
    tmp_path: pathlib.Path,
) -> None:
    # End-to-end alignment: a posterior that matches the truth in u and w and is
    # 100 % out in v must score exactly that way. A transposed component axis
    # anywhere in the accumulate -> reduce -> score chain is invisible on
    # symmetric random data and obvious here.
    from scripts.esmda._esmda_common import read_yaml
    from scripts.esmda.compute_esmda_metrics import compute_metrics

    run_dir = _full_run_dir(tmp_path)
    truth = _constant_state(
        None, _RUN_WINDOWS * _RUN_FRAMES, {"u": 1.0, "v": 1.0, "w": 1.0}
    )
    truth.assign_coords(
        time=np.linspace(
            0.0,
            _RUN_WINDOWS * _RUN_SIM_TIME,
            _RUN_WINDOWS * _RUN_FRAMES,
            endpoint=False,
        )
    ).to_netcdf(run_dir / "truth_state.nc")
    for w in range(_RUN_WINDOWS):
        _constant_state(4, _RUN_FRAMES, {"u": 1.0, "v": 2.0, "w": 1.0}).to_netcdf(
            run_dir / "windows" / f"window_{w}_posterior_state.nc"
        )

    compute_metrics(run_dir)
    block = read_yaml(run_dir / "run_summary.yaml")["field_metrics"]

    # A constant truth has a *measured* zero sampling floor, so the relative
    # criterion decides on its own.
    assert block["hit_rate_tolerance_w"] == {"u": 0.0, "v": 0.0, "w": 0.0}
    assert block["hit_rate_posterior"]["u"] == 1.0
    assert block["hit_rate_posterior"]["w"] == 1.0
    assert block["hit_rate_posterior"]["v"] == 0.0
    assert block["hit_rate_posterior"]["q"] == pytest.approx(2 / 3)


def test_field_metrics_score_the_prior_when_its_states_were_saved(
    tmp_path: pathlib.Path,
) -> None:
    from scripts.esmda._esmda_common import read_yaml
    from scripts.esmda.compute_esmda_metrics import compute_metrics

    run_dir = _full_run_dir(tmp_path, save_prior_state=True)
    compute_metrics(run_dir)
    block = read_yaml(run_dir / "run_summary.yaml")["field_metrics"]

    assert set(block["hit_rate_prior"]) == set(block["hit_rate_posterior"])
    fields = xarray.open_dataset(run_dir / "eval_fields.nc")
    assert "prior_slab_mean" in fields and "prior_slab_mean_spread" in fields
    fields.close()


def test_field_metrics_are_dropped_without_a_members_axis(
    tmp_path: pathlib.Path, caplog: pytest.LogCaptureFixture
) -> None:
    # An ensemble-mean-only artifact: the mean fields are per member, so there
    # is nothing to accumulate. Invariant 3 -- that costs this block alone.
    from scripts.esmda._esmda_common import read_yaml
    from scripts.esmda.compute_esmda_metrics import compute_metrics

    run_dir = _full_run_dir(tmp_path)
    for w in range(_RUN_WINDOWS):
        _state_dataset(None, _RUN_FRAMES, seed=10 + w).to_netcdf(
            run_dir / "windows" / f"window_{w}_posterior_state.nc"
        )

    with caplog.at_level("WARNING"):
        compute_metrics(run_dir)
    summary = read_yaml(run_dir / "run_summary.yaml")

    assert "No mean-field metrics" in caplog.text
    assert "field_metrics" not in summary
    assert not (run_dir / "eval_fields.nc").exists()
    assert summary["parameter_metrics"] and summary["state_metrics"]


def test_collector_disables_itself_rather_than_breaking_the_shared_pass(
    caplog: pytest.LogCaptureFixture,
) -> None:
    # The mean fields ride on the sensor extraction's pass, so a state whose
    # layout colocation refuses -- here a uDALES state subset along z before
    # colocation -- must cost this layer alone. The callback runs inside the
    # sensor loop; an exception escaping it would take the sensor blocks with
    # it (invariant 3).
    from scripts.esmda._esmda_common import MeanFieldCollector

    collector = MeanFieldCollector("udales", np.array([1.0]), np.array([1.0]))
    with caplog.at_level("WARNING"):
        collector.add(0, 0, _udales_state(n=6).isel(zm=[0, 2, 4], zt=[0, 2, 4]))

    assert "Mean fields disabled" in caplog.text
    assert collector.reason and "midpoints" in collector.reason
    assert collector.result() is None
    # Still inert on everything that follows it in the pass.
    collector.add(0, 1, _udales_state())
    assert collector.result() is None
