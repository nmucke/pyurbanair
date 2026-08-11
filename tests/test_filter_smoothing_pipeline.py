"""The filter-smoothing pipeline's metric and figure stages.

``scripts/filter_smoothing/run_filter_smoothing.py`` writes filtering-SHAPED
per-cycle artifacts -- its inner state filter is the EnKF -- so stages 2 and 3
reuse ``scripts.filtering``'s machinery wholesale. What this file is about is
therefore not that machinery (``tests/test_filtering_evaluation.py`` owns it)
but the two places the reuse is *not* free:

  * **The moving window.** With ``num_windows = W > 1`` the run assimilates
    ``T = L + (W-1)*s`` cycles, but every window's inner filter rewrites the
    same ``_ensemble_states/cycle_0 … cycle_{L-1}`` and the final pass
    overwrites ``state_history.nc`` whole, so only the LAST window's ``L``
    cycles are on disk -- while ``truth_access.yaml``'s ``num_cycles`` is ``T``.
    Handing that ``T`` to ``cycle_state_source`` makes ``_forecast_cycle_paths``
    look for cycles that were never written, find nothing, and silently fall
    back to the analyzed frames on a run that paid for the forecasts. The tests
    below pin the mapping that avoids it (``window_layout`` ->
    ``final_window_truth_access``), that it is not vacuous (the raw ``ta``
    really does degrade), and that the surviving cycles are LABELLED with their
    global indices rather than renumbered from zero.

  * **The outer ESMDA loop.** ``iteration_diagnostics.yaml`` /
    ``window_diagnostics.yaml`` / ``params_iterations.nc`` have no filtering
    counterpart, and neither does the parameter TRAJECTORY: the posterior spans
    the whole horizon while the prior spans the first window only, so the
    knot-by-knot prior comparison is available on a single-window run and
    genuinely absent on a moving one.

Everything runs on a synthetic run dir in the shape
``tests/test_filtering_evaluation.py`` established -- no solver, no Hydra
compose -- so the file stays in the fast set.
"""

from __future__ import annotations

import pathlib
from collections.abc import Iterator

import numpy as np
import pytest
import xarray

# The horizon: 3 truth cycles of 4 frames. The moving-window layout runs two
# 2-cycle windows shifting by one (so the surviving artifacts are cycles 1-2 of
# 3, and a stage that renumbered them 0-1 would be wrong by a whole cycle); the
# single-window layout is one 3-cycle window, where every mapping below has to
# reduce to the plain filtering reading.
_TOTAL_CYCLES = 3
_SIM_TIME = 4.0
_FRAMES = 4
_TOTAL_FRAMES = _TOTAL_CYCLES * _FRAMES
_NUM_STEPS = 2  # outer ESMDA iterations
_MEMBERS = 4

_N_CELLS = 5
_AXIS = np.linspace(0.0, 20.0, _N_CELLS)
_SOLID_COLUMNS = 1


def _layout(num_windows: int) -> tuple[int, int]:
    """``(window_length L, window_shift s)`` for a ``W``-window horizon of T."""
    if num_windows == 1:
        return _TOTAL_CYCLES, _TOTAL_CYCLES
    shift = 1
    return _TOTAL_CYCLES - (num_windows - 1) * shift, shift


def _velocity_dataset(
    dims: tuple[str, ...],
    sizes: dict[str, int],
    seed: int,
    coords: dict[str, object],
) -> xarray.Dataset:
    """u/v/w over ``dims`` on the shared pylbm-shaped grid."""
    rng = np.random.default_rng(seed)
    shape = tuple(sizes[d] for d in dims)
    return xarray.Dataset(
        {name: (dims, rng.normal(size=shape) + 2.0) for name in ("u", "v", "w")},
        coords=coords,
    )


def _with_blanking(state: xarray.Dataset) -> xarray.Dataset:
    """Add pylbm's obstacle indicator (non-zero = solid) beside the velocities."""
    blanking = np.zeros((_N_CELLS, _N_CELLS, _N_CELLS))
    blanking[..., :_SOLID_COLUMNS] = 1.0
    state["blanking"] = (("z", "y", "x"), blanking)
    return state


def _truth_dataset() -> xarray.Dataset:
    """The truth over the FULL horizon, one frame per second."""
    return _velocity_dataset(
        ("time", "z", "y", "x"),
        {"time": _TOTAL_FRAMES, "z": _N_CELLS, "y": _N_CELLS, "x": _N_CELLS},
        1,
        {
            "time": np.arange(_TOTAL_FRAMES, dtype=float),
            "z": _AXIS,
            "y": _AXIS,
            "x": _AXIS,
        },
    )


def _state_history(window_length: int) -> xarray.Dataset:
    """``state_history.nc``: the final pass's analyzed frame per WINDOW cycle."""
    history = _velocity_dataset(
        ("cycle", "ensemble", "z", "y", "x"),
        {
            "cycle": window_length,
            "ensemble": _MEMBERS,
            "z": _N_CELLS,
            "y": _N_CELLS,
            "x": _N_CELLS,
        },
        10,
        {"ensemble": np.arange(_MEMBERS), "z": _AXIS, "y": _AXIS, "x": _AXIS},
    )
    return _with_blanking(history.assign_coords(time=_SIM_TIME))


def _forecast_segment(cycle: int, member: int) -> xarray.Dataset:
    """One member's forecast segment for one WINDOW-local cycle.

    The segment's last frame is the one the analysis updates, so it sits at the
    cycle's end on the solver's own clock; ``_rebased_forecast_times`` anchors on
    it, which is why the local index is what matters here and the window offset
    is applied to the *truth* view instead.
    """
    times = (cycle + 1) * _SIM_TIME - np.arange(_FRAMES - 1, -1, -1, dtype=float)
    return _with_blanking(
        _velocity_dataset(
            ("time", "z", "y", "x"),
            {"time": _FRAMES, "z": _N_CELLS, "y": _N_CELLS, "x": _N_CELLS},
            100 + 10 * cycle + member,
            {"time": times, "z": _AXIS, "y": _AXIS, "x": _AXIS},
        )
    )


def _trajectory(n_knots: int, spread: float, seed: int) -> xarray.Dataset:
    """A parameter trajectory ensemble on a one-knot-per-cycle grid."""
    rng = np.random.default_rng(seed)
    knots = np.arange(n_knots, dtype=float) * _SIM_TIME
    mean = 5.0 + 0.25 * np.arange(n_knots)
    return xarray.Dataset(
        {
            "velocity_magnitude": (
                ("ensemble", "time"),
                mean[None, :] + spread * rng.normal(size=(_MEMBERS, n_knots)),
            )
        },
        coords={"ensemble": np.arange(_MEMBERS), "time": knots},
    )


def _iteration_rows(offset: float = 0.0) -> list[dict]:
    """One record per outer ESMDA iteration, with a converging ``obs_rmse``."""
    return [
        {
            "iteration": i,
            "alpha": float(_NUM_STEPS),
            "obs_rmse": 1.0 + offset - 0.25 * i,
            "innovation_chi2": 1.4 - 0.1 * i,
            "param_spread_prior": 0.5 - 0.05 * i,
            "param_spread_posterior": 0.4 - 0.05 * i,
        }
        for i in range(_NUM_STEPS)
    ]


def _run_dir(
    tmp_path: pathlib.Path,
    *,
    num_windows: int = 2,
    forecast_states: bool = False,
    with_params: bool = True,
    with_iterations: bool = True,
    with_window_diagnostics: bool = True,
    metrics: bool = False,
) -> pathlib.Path:
    """A filter-smoothing run dir complete enough for stages 2 and 3.

    The layout ``scripts/filter_smoothing/run_filter_smoothing.py`` writes: the
    filtering-shaped per-cycle artifacts of the LAST window beside the
    full-horizon ``truth_access.yaml``, the trajectory parameter datasets, and
    the outer loop's own records. Every input a run may legitimately lack is
    switchable, because "this run was configured without it" is the degradation
    the two stages exist to survive.
    """
    from scripts.esmda._esmda_common import write_yaml

    length, shift = _layout(num_windows)
    first_cycle = (num_windows - 1) * shift

    run_dir = tmp_path / "run"
    run_dir.mkdir(parents=True)

    write_yaml(
        {
            "run": {},
            "filter_smoothing": {"obs_error_std": 0.2},
            "obs": {
                "mode": "points",
                "x_points": [4.0, 12.0],
                "y_points": [5.0, 13.0],
                "z_points": [6.0, 7.0],
                "validation_x_points": [8.0],
                "validation_y_points": [9.0],
                "validation_z_points": [10.0],
            },
        },
        run_dir / "config.yaml",
    )
    write_yaml(
        {
            "configuration": {
                "estimator": "FilterSmoothingESMDA",
                "num_cycles": length,
                "num_windows": num_windows,
                "window_shift": shift,
                "total_cycles": _TOTAL_CYCLES,
                "final_pass_first_cycle": first_cycle,
                "num_steps": _NUM_STEPS,
                "ensemble_size": _MEMBERS,
            },
            "timing": {"total_seconds": 1.0},
        },
        run_dir / "run_info.yaml",
    )

    _truth_dataset().to_netcdf(run_dir / "true_state.nc")
    write_yaml(
        {
            "true_state_path": str(run_dir / "true_state.nc"),
            "x_offset": 0.0,
            "start_idx": 0,
            "t_offset": 0.0,
            "n_total": _TOTAL_FRAMES,
            "n_per_cycle": _FRAMES,
            # The FULL horizon, as the runner records it -- this is the number
            # that mis-maps the on-disk cycles when it is used unmodified.
            "num_cycles": _TOTAL_CYCLES,
            "sim_time": _SIM_TIME,
            "truth_solver_name": "pylbm",
            "assim_solver_name": "pylbm",
        },
        run_dir / "truth_access.yaml",
    )

    # The final consistency pass's per-cycle artifacts: the LAST window's.
    history = _state_history(length)
    history.to_netcdf(run_dir / "state_history.nc")
    history.isel(cycle=-1, drop=True).to_netcdf(run_dir / "posterior_state.nc")
    write_yaml(
        [
            {
                "obs_prior_rmse": 1.0 + cycle,
                "obs_posterior_rmse": 0.5 + cycle,
                "innovation_chi2": 1.1,
                "state_spread_prior": 0.3,
                "state_spread_posterior": 0.2,
                # The inner pass is state-only, so it records no parameter
                # spreads -- those live in iteration_diagnostics.yaml instead.
                "param_spread_prior": None,
                "param_spread_posterior": None,
            }
            for cycle in range(length)
        ],
        run_dir / "cycle_diagnostics.yaml",
    )
    if forecast_states:
        for cycle in range(length):
            cycle_dir = run_dir / "_ensemble_states" / f"cycle_{cycle}"
            cycle_dir.mkdir(parents=True, exist_ok=True)
            for member in range(_MEMBERS):
                _forecast_segment(cycle, member).to_netcdf(
                    cycle_dir / f"state_{member}.nc"
                )

    # The outer loop's records. ``iteration_diagnostics.yaml`` is the LAST
    # window's; ``window_diagnostics.yaml`` exists only when the window moved.
    if with_iterations:
        write_yaml(_iteration_rows(), run_dir / "iteration_diagnostics.yaml")
    if num_windows > 1 and with_window_diagnostics:
        write_yaml(
            [
                {
                    "window": w,
                    "first_cycle": w * shift,
                    "last_cycle": w * shift + length - 1,
                    "iteration_diagnostics": _iteration_rows(offset=0.1 * w),
                    "window_time": 2.0 + w,
                }
                for w in range(num_windows)
            ],
            run_dir / "window_diagnostics.yaml",
        )

    # The trajectory artifacts. The posterior spans the FULL horizon; the prior
    # spans the first window only, which is why the knot-by-knot prior columns
    # are available on a single-window run and absent on a moving one.
    if with_params:
        _trajectory(_TOTAL_CYCLES + 1, 1.0, 3).to_netcdf(
            run_dir / "posterior_params.nc"
        )
        _trajectory(length + 1, 4.0, 4).to_netcdf(run_dir / "prior_params.nc")
        iterations = xarray.concat(
            [
                _trajectory(length + 1, 4.0 - step, 5 + step)
                for step in range(_NUM_STEPS + 1)
            ],
            dim="esmda_step",
        )
        iterations.to_netcdf(run_dir / "params_iterations.nc")
    truth = _trajectory(_TOTAL_CYCLES + 1, 0.0, 6).isel(ensemble=[0])
    truth.to_netcdf(run_dir / "true_params.nc")

    if metrics:
        from scripts.filter_smoothing.compute_filter_smoothing_metrics import (
            compute_metrics,
        )

        compute_metrics(run_dir)
    return run_dir


@pytest.fixture(autouse=True)  # type: ignore[misc]
def _close_figures() -> Iterator[None]:
    """No test leaks a figure into the next one (or into the suite's memory)."""
    import matplotlib.pyplot as plt

    yield
    plt.close("all")


# ---------------------------------------------------------------------------
# The window layout, and the cycle mapping built on it
# ---------------------------------------------------------------------------


def test_window_layout_reads_the_moving_window_off_the_runs_own_records(
    tmp_path: pathlib.Path,
) -> None:
    # ``final_pass_first_cycle`` is the whole mapping: the on-disk ``cycle_k`` is
    # the run's cycle ``first_cycle + k``. Derived by the runner, read back here,
    # and never recomputed from a config literal in either stage.
    from scripts.filter_smoothing._filter_smoothing_common import (
        global_cycle_indices,
        read_yaml,
        window_layout,
    )

    run_dir = _run_dir(tmp_path, num_windows=2)
    layout = window_layout(run_dir, read_yaml(run_dir / "truth_access.yaml"))

    assert (layout.num_windows, layout.window_length, layout.window_shift) == (2, 2, 1)
    assert layout.total_cycles == _TOTAL_CYCLES
    assert layout.first_cycle == 1
    assert layout.spans == [(0, 1), (1, 2)]
    # The surviving artifacts are cycles 1 and 2 of a 3-cycle horizon; a stage
    # that renumbered them 0 and 1 would place the whole run a cycle early.
    assert global_cycle_indices(layout) == [1, 2]


def test_window_layout_of_a_single_window_run_is_the_plain_filtering_reading(
    tmp_path: pathlib.Path,
) -> None:
    # The default path has to degenerate exactly: one window over the whole
    # horizon, offset zero, so ``final_window_truth_access`` is a no-op and every
    # reused filtering helper sees the ``ta`` it always saw.
    from scripts.filter_smoothing._filter_smoothing_common import (
        final_window_truth_access,
        global_cycle_indices,
        read_yaml,
        window_layout,
    )

    run_dir = _run_dir(tmp_path, num_windows=1)
    ta = read_yaml(run_dir / "truth_access.yaml")
    layout = window_layout(run_dir, ta)

    assert layout.num_windows == 1
    assert layout.first_cycle == 0
    assert layout.window_length == _TOTAL_CYCLES
    assert global_cycle_indices(layout) == [0, 1, 2]
    window_ta = final_window_truth_access(ta, layout)
    assert window_ta["start_idx"] == ta["start_idx"]
    assert window_ta["t_offset"] == ta["t_offset"]
    assert window_ta["num_cycles"] == ta["num_cycles"]
    assert window_ta["n_total"] == ta["n_total"]


def test_window_layout_prefers_the_orchestrators_own_per_window_spans(
    tmp_path: pathlib.Path, caplog: pytest.LogCaptureFixture
) -> None:
    # ``window_diagnostics.yaml`` is what the windows DID; run_info's scalars are
    # what the config asked for. They agree on every real run, so the only way to
    # tell which one is authoritative is to make them disagree -- and a
    # disagreement means the run directory mixes two runs, which has to be logged
    # rather than resolved silently.
    from scripts.esmda._esmda_common import write_yaml
    from scripts.filter_smoothing._filter_smoothing_common import window_layout

    run_dir = _run_dir(tmp_path, num_windows=2)
    write_yaml(
        [
            {
                "window": w,
                "first_cycle": 2 * w,
                "last_cycle": 2 * w + 1,
                "iteration_diagnostics": _iteration_rows(),
                "window_time": 1.0,
            }
            for w in range(2)
        ],
        run_dir / "window_diagnostics.yaml",
    )

    with caplog.at_level("WARNING"):
        layout = window_layout(run_dir)

    assert layout.spans == [(0, 1), (2, 3)]
    assert layout.first_cycle == 2 and layout.window_shift == 2
    assert "window_diagnostics.yaml" in layout.source
    assert "disagree" in caplog.text


def test_window_layout_says_so_when_a_run_dir_records_no_layout_at_all(
    tmp_path: pathlib.Path,
) -> None:
    # An artifact set predating the layout keys, or a hand-made fixture. Reading
    # it as one window is the only sane default, but an ASSUMED layout must be
    # distinguishable from a recorded one in run_summary.yaml -- silently
    # degrading is exactly what mis-maps the cycles in the first place.
    from scripts.filter_smoothing._filter_smoothing_common import window_layout

    run_dir = _run_dir(tmp_path, num_windows=2)
    (run_dir / "run_info.yaml").unlink()

    layout = window_layout(run_dir)

    # The window_diagnostics record still stands on its own.
    assert layout.spans == [(0, 1), (1, 2)]
    (run_dir / "window_diagnostics.yaml").unlink()
    bare = window_layout(run_dir, {"num_cycles": _TOTAL_CYCLES, "n_per_cycle": _FRAMES})
    assert bare.num_windows == 1
    assert bare.window_length == _TOTAL_CYCLES
    assert "assumed" in bare.source


def test_final_window_truth_access_rebases_the_truth_onto_the_last_window(
    tmp_path: pathlib.Path,
) -> None:
    # The re-basing is what makes every reused helper correct without any of them
    # learning that windows exist: the view starts at the window's first truth
    # frame AND opens at t=0, because ``evaluation.sensors.window_masks`` bins by
    # the absolute time coordinate and would put the whole window in its last bin
    # otherwise.
    from scripts.esmda._esmda_common import open_truth
    from scripts.filter_smoothing._filter_smoothing_common import (
        final_window_truth_access,
        read_yaml,
        truth_end_of_cycle,
        window_layout,
    )

    run_dir = _run_dir(tmp_path, num_windows=2)
    ta = read_yaml(run_dir / "truth_access.yaml")
    window_ta = final_window_truth_access(ta, window_layout(run_dir, ta))

    assert window_ta["num_cycles"] == 2
    assert window_ta["start_idx"] == _FRAMES  # one whole cycle of truth dropped
    assert window_ta["n_total"] == _TOTAL_FRAMES - _FRAMES
    assert window_ta["t_offset"] == pytest.approx(_SIM_TIME)

    with open_truth(
        window_ta["true_state_path"],
        window_ta["n_total"],
        window_ta["x_offset"],
        window_ta["start_idx"],
        window_ta["t_offset"],
    ) as view:
        # Global frames 4..11, rebased so the window opens at t = 0.
        assert list(view["time"].values) == pytest.approx(
            list(np.arange(_TOTAL_FRAMES - _FRAMES, dtype=float))
        )

    # And the end-of-cycle frames are the last window's, not the horizon's first.
    truth_end = truth_end_of_cycle(window_ta)
    assert truth_end.sizes["time"] == 2
    expected = _truth_dataset().isel(time=[7, 11])["u"].values
    assert np.allclose(truth_end["u"].values, expected)


def test_the_raw_truth_access_silently_loses_the_on_disk_forecasts(
    tmp_path: pathlib.Path, caplog: pytest.LogCaptureFixture
) -> None:
    # The trap this whole mapping exists for, pinned in BOTH directions so the
    # fix cannot become vacuous. ``truth_access.yaml``'s ``num_cycles`` is the
    # horizon T, but only L cycle directories were ever written, so
    # ``_forecast_cycle_paths`` looks for ``cycle_2``, finds nothing, and falls
    # back to the one analyzed frame per cycle -- on a run that paid the disk to
    # have the forecasts. With the window-scoped view it picks them up.
    from scripts.filter_smoothing._filter_smoothing_common import (
        cycle_state_source,
        final_window_truth_access,
        read_yaml,
        window_layout,
    )

    run_dir = _run_dir(tmp_path, num_windows=2, forecast_states=True)
    ta = read_yaml(run_dir / "truth_access.yaml")

    with caplog.at_level("INFO"):
        degraded = cycle_state_source(run_dir, ta)
    assert degraded.kind == "analysis"
    assert "No forecast states for cycle 2" in caplog.text

    window_ta = final_window_truth_access(ta, window_layout(run_dir, ta))
    source = cycle_state_source(run_dir, window_ta)
    assert source.kind == "forecast"
    assert source.num_cycles == 2
    assert source.n_members == _MEMBERS


def test_the_truth_and_the_ensemble_bin_onto_the_same_frames_in_the_last_window(
    tmp_path: pathlib.Path,
) -> None:
    # The invariant every statistic rests on, restated for a window that does not
    # start at cycle 0: the truth's frames for global cycles 1-2 have to land in
    # the same two bins as the ensemble's segments, with the same count in each.
    # A mis-set ``t_offset`` produces a complete, plausible, wrong summary rather
    # than an error.
    from evaluation.sensors import window_masks

    from scripts.filter_smoothing._filter_smoothing_common import (
        build_sensor_sets,
        cycle_sensor_series,
        cycle_state_source,
        final_window_truth_access,
        load_run_config,
        read_yaml,
        truth_cycle_statistics_series,
        window_layout,
    )

    run_dir = _run_dir(tmp_path, num_windows=2, forecast_states=True)
    ta = read_yaml(run_dir / "truth_access.yaml")
    window_ta = final_window_truth_access(ta, window_layout(run_dir, ta))
    sensor_sets = build_sensor_sets(load_run_config(run_dir))
    source = cycle_state_source(run_dir, window_ta)
    truth_series = truth_cycle_statistics_series(window_ta, source, sensor_sets)
    ensemble_series = cycle_sensor_series(run_dir, window_ta, source, sensor_sets)

    def _per_bin(series: xarray.DataArray) -> list[int]:
        masks = window_masks(
            series["time"].values, source.bin_seconds, source.num_cycles
        )
        return [int(mask.sum()) for mask in masks]

    assert _per_bin(truth_series["assimilation"]) == [_FRAMES, _FRAMES]
    assert _per_bin(ensemble_series["assimilation"]) == [_FRAMES, _FRAMES]


# ---------------------------------------------------------------------------
# run_summary.yaml -- compute_filter_smoothing_metrics.py
# ---------------------------------------------------------------------------


def test_run_summary_records_the_window_layout_and_the_cycles_it_scored(
    tmp_path: pathlib.Path,
) -> None:
    # The provenance block. A reader holding run_summary.yaml has no other way to
    # tell that the state/sensor numbers cover cycles 1-2 of a 3-cycle run while
    # the trajectory numbers cover all of it.
    from scripts.esmda._esmda_common import read_yaml
    from scripts.filter_smoothing.compute_filter_smoothing_metrics import (
        compute_metrics,
    )

    run_dir = _run_dir(tmp_path, num_windows=2)
    compute_metrics(run_dir)
    summary = read_yaml(run_dir / "run_summary.yaml")
    block = summary["window_layout"]

    assert block["num_windows"] == 2
    assert block["window_length"] == 2
    assert block["window_shift"] == 1
    assert block["total_cycles"] == _TOTAL_CYCLES
    assert block["final_pass_first_cycle"] == 1
    assert block["evaluated_cycles"] == [1, 2]
    assert block["window_spans"] == [[0, 1], [1, 2]]
    assert "window_diagnostics.yaml" in block["source"]
    # ...and every per-cycle block carries those same global indices, so no
    # consumer has to join them back to the layout by hand.
    assert summary["filter_diagnostics"]["cycles"] == [1, 2]
    assert summary["state_metrics"]["cycles"] == [1, 2]
    assert summary["cycle_states"]["cycles"] == [1, 2]


def test_run_summary_carries_every_expected_block_on_a_complete_run_dir(
    tmp_path: pathlib.Path,
) -> None:
    # The metric stage's contract: the filtering pipeline's blocks (reused
    # wholesale) plus the trajectory and outer-loop ones that have no filtering
    # counterpart.
    from scripts.esmda._esmda_common import read_yaml
    from scripts.filter_smoothing.compute_filter_smoothing_metrics import (
        compute_metrics,
    )

    run_dir = _run_dir(tmp_path, num_windows=2)
    compute_metrics(run_dir)
    summary = read_yaml(run_dir / "run_summary.yaml")

    assert summary["metrics_version"] == 2
    assert summary["configuration"]["ensemble_size"] == _MEMBERS
    for key in (
        "window_layout",
        "iteration_metrics",
        "window_metrics",
        "filter_diagnostics",
        "parameter_metrics",
        "trajectory_metrics",
        "ensemble_health",
        "state_metrics",
        "sensor_metrics",
        "cycle_states",
        "sensor_statistics",
        "field_metrics",
    ):
        assert key in summary, f"{key} is missing from run_summary.yaml"
    assert (run_dir / "eval_fields.nc").is_file()


def test_run_summary_reports_the_outer_loops_convergence_per_window(
    tmp_path: pathlib.Path,
) -> None:
    # ``obs_rmse`` is measured before each iteration's update, so it is the
    # loop's cost function and its prior->final drop is the one number that says
    # the extra forecasts bought something. Per window as well as pooled: a run
    # whose later windows are steadily harder to fit is invisible in one average.
    from scripts.esmda._esmda_common import read_yaml
    from scripts.filter_smoothing.compute_filter_smoothing_metrics import (
        compute_metrics,
    )

    run_dir = _run_dir(tmp_path, num_windows=2)
    compute_metrics(run_dir)
    summary = read_yaml(run_dir / "run_summary.yaml")

    iterations = summary["iteration_metrics"]
    assert iterations["num_iterations"] == _NUM_STEPS
    assert iterations["obs_rmse"]["final"] == pytest.approx(0.75)
    assert iterations["obs_rmse_reduction"] == pytest.approx(0.25)
    assert iterations["param_spread_posterior"]["final"] == pytest.approx(0.35)

    windows = summary["window_metrics"]
    assert [w["window"] for w in windows] == [0, 1]
    assert [(w["first_cycle"], w["last_cycle"]) for w in windows] == [(0, 1), (1, 2)]
    assert [w["window_seconds"] for w in windows] == [2.0, 3.0]
    assert windows[1]["obs_rmse"]["max"] == pytest.approx(1.1)


def test_run_summary_without_the_outer_records_keeps_everything_else(
    tmp_path: pathlib.Path, caplog: pytest.LogCaptureFixture
) -> None:
    # Invariant 3 for the two blocks this pipeline adds: a run dir whose
    # diagnostics YAMLs were never written costs those blocks and nothing else.
    from scripts.esmda._esmda_common import read_yaml
    from scripts.filter_smoothing.compute_filter_smoothing_metrics import (
        compute_metrics,
    )

    run_dir = _run_dir(
        tmp_path,
        num_windows=2,
        with_iterations=False,
        with_window_diagnostics=False,
    )
    with caplog.at_level("INFO"):
        compute_metrics(run_dir)
    summary = read_yaml(run_dir / "run_summary.yaml")

    assert "iteration_metrics" not in summary
    assert "window_metrics" not in summary
    assert "No iteration_diagnostics.yaml" in caplog.text
    for key in ("parameter_metrics", "state_metrics", "sensor_metrics", "cycle_states"):
        assert key in summary, f"{key} was lost with the outer-loop records"
    # With window_diagnostics.yaml gone the layout falls back to run_info's
    # scalars, which say the same thing.
    assert summary["window_layout"]["evaluated_cycles"] == [1, 2]
    assert "run_info.yaml" in summary["window_layout"]["source"]


@pytest.mark.parametrize("num_windows", [1, 2])  # type: ignore[misc]
def test_trajectory_metrics_compare_the_prior_only_where_the_knots_line_up(
    tmp_path: pathlib.Path, num_windows: int
) -> None:
    # The prior is sampled over the FIRST window; the posterior spans the whole
    # horizon and every later window's prior is the previous window's posterior.
    # So on a moving-window run there is no knot-by-knot prior to compare
    # against, and an absent prior column must be distinguishable from a prior
    # that scored badly.
    from scripts.esmda._esmda_common import read_yaml
    from scripts.filter_smoothing.compute_filter_smoothing_metrics import (
        compute_metrics,
    )

    run_dir = _run_dir(tmp_path, num_windows=num_windows)
    compute_metrics(run_dir)
    block = read_yaml(run_dir / "run_summary.yaml")["trajectory_metrics"]
    entry = block["velocity_magnitude"]

    # One entry per knot over the full horizon, on the horizon's own clock.
    assert entry["knot"] == pytest.approx(
        [k * _SIM_TIME for k in range(_TOTAL_CYCLES + 1)]
    )
    assert len(entry["rmse"]) == _TOTAL_CYCLES + 1
    assert len(entry["posterior_std"]) == _TOTAL_CYCLES + 1
    assert entry["prior_comparable"] is (num_windows == 1)
    assert ("prior_rmse" in entry) is (num_windows == 1)
    assert ("contraction_ratio" in entry) is (num_windows == 1)


def test_ensemble_health_counts_the_outer_iterations_not_the_inner_cycles(
    tmp_path: pathlib.Path, caplog: pytest.LogCaptureFixture
) -> None:
    # The one filtering block that is NOT reused, and why: the inner pass is
    # state-only, so ``params_history.nc``'s cycle entries are the same
    # trajectory repeated and counting them would report one number ``L+1``
    # times and call it a per-cycle series. ``params_iterations.nc`` is the axis
    # that actually varies -- entry 0 the window's prior, then one per outer
    # update.
    from scripts.esmda._esmda_common import read_yaml
    from scripts.filter_smoothing.compute_filter_smoothing_metrics import (
        compute_metrics,
    )

    run_dir = _run_dir(tmp_path, num_windows=2)
    compute_metrics(run_dir)
    health = read_yaml(run_dir / "run_summary.yaml")["ensemble_health"]

    assert health["n_members"] == _MEMBERS and health["n_unique"] == _MEMBERS
    assert len(health["n_unique_per_iteration"]) == _NUM_STEPS + 1

    # ...and its absence (run.save_history off) costs the list, never the block.
    (run_dir / "params_iterations.nc").unlink()
    with caplog.at_level("INFO"):
        compute_metrics(run_dir)
    health = read_yaml(run_dir / "run_summary.yaml")["ensemble_health"]
    assert health["n_unique_per_iteration"] == []
    assert health["n_members"] == _MEMBERS
    assert "No params_iterations.nc" in caplog.text


# ---------------------------------------------------------------------------
# The figures -- make_filter_smoothing_figures.py
# ---------------------------------------------------------------------------


def _stub_animation(monkeypatch: pytest.MonkeyPatch) -> dict:
    """Stub the rollout animation (not under test, and the slowest thing here)."""
    import scripts.filter_smoothing.make_filter_smoothing_figures as wiring

    recorded: dict = {}

    def _record(name: str) -> object:
        def _stub(*args: object, **kwargs: object) -> None:
            recorded[name] = kwargs
            return None

        return _stub

    monkeypatch.setattr(wiring, "animate_rollout_state", _record("animation"))
    return recorded


def test_make_figures_draws_the_whole_set_on_a_complete_run_dir(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from scripts.filter_smoothing.make_filter_smoothing_figures import make_figures

    _stub_animation(monkeypatch)
    run_dir = _run_dir(tmp_path, num_windows=2, metrics=True)

    make_figures(run_dir)

    for name in (
        "parameter_evolution.png",
        "parameter_error.png",
        "parameter_marginals.png",
        "iteration_convergence.png",
        "final_state_with_obs.png",
        "sensor_timeseries_assimilation.png",
        "sensor_timeseries_validation.png",
        "station_profiles.png",
        "mean_slices.png",
        "sensor_fans.png",
        "rank_histogram.png",
    ):
        assert (run_dir / name).is_file(), f"{name} was not written"


def test_make_figures_places_the_sensor_fan_on_the_horizons_clock(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The truth view was re-based so the window opens at t=0, which is what makes
    # the binning work -- but ``plot_sensor_fans`` labels its axis "Time [s]",
    # and the honest seconds are the HORIZON's. Window 1 starts at cycle 1, so
    # its analyzed frames sit at 8 s and 12 s, not at 4 s and 8 s, and its cycle
    # boundaries at 4/8/12 s.
    import scripts.filter_smoothing.make_filter_smoothing_figures as wiring
    from scripts.filter_smoothing.make_filter_smoothing_figures import make_figures

    recorded = _stub_animation(monkeypatch)

    def _record_fans(*args: object, **kwargs: object) -> None:
        recorded["fans"] = kwargs
        return None

    monkeypatch.setattr(wiring, "plot_sensor_fans", _record_fans)
    run_dir = _run_dir(tmp_path, num_windows=2, metrics=True)

    make_figures(run_dir)

    times = recorded["fans"]["times"]
    assert list(times) == pytest.approx([2 * _SIM_TIME, 3 * _SIM_TIME])
    assert list(recorded["fans"]["window_edges"]) == pytest.approx(
        [_SIM_TIME, 2 * _SIM_TIME, 3 * _SIM_TIME]
    )


def test_make_figures_skips_what_a_run_dir_without_the_metric_stage_cannot_support(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    # The figure stage runs on run dirs whose metric stage never ran. Three
    # figures cannot be drawn then; the rest must be, and each skip has to name
    # the script that would produce the missing artifact.
    from scripts.filter_smoothing.make_filter_smoothing_figures import make_figures

    _stub_animation(monkeypatch)
    run_dir = _run_dir(tmp_path, num_windows=2, metrics=False)

    make_figures(run_dir)

    printed = capsys.readouterr().out
    for name in ("station_profiles.png", "mean_slices.png", "rank_histogram.png"):
        assert not (run_dir / name).exists(), f"{name} was drawn from nothing"
        assert name in printed, f"{name} was skipped silently"
    assert (
        printed.count("compute_filter_smoothing_metrics.py") >= 2
    ), "the skips do not say how to fix themselves"
    # A skip never costs the figures after it.
    for name in (
        "parameter_marginals.png",
        "iteration_convergence.png",
        "sensor_fans.png",
    ):
        assert (run_dir / name).is_file(), f"{name} was lost to an earlier skip"
    # ...and the moving window's span is stated, since no PNG says it.
    assert "cycles 1-2" in printed


def test_the_convergence_figure_draws_one_line_per_window(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # ``iteration_diagnostics.yaml`` is the LAST window's record, not a run-wide
    # one, so a stage that appended it to the per-window series would draw the
    # last window twice.
    import scripts.filter_smoothing.make_filter_smoothing_figures as wiring
    from scripts.filter_smoothing.make_filter_smoothing_figures import make_figures

    recorded = _stub_animation(monkeypatch)

    def _record(per_window: object, output_path: object, **kwargs: object) -> None:
        recorded["convergence"] = (per_window, kwargs)
        return None

    monkeypatch.setattr(wiring, "plot_iteration_convergence", _record)
    run_dir = _run_dir(tmp_path, num_windows=2)

    make_figures(run_dir)

    per_window, kwargs = recorded["convergence"]
    assert len(per_window) == 2
    assert kwargs["window_labels"] == [0, 1]
    assert len(per_window[0]["obs_rmse"]) == _NUM_STEPS
