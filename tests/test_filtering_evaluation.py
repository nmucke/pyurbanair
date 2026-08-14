"""WP1.5 for the filter: the cycle-state contract and the blocks built on it.

The ESMDA pipeline has exactly one thing its evaluation blocks can reduce over
-- ``windows/window_{w}_posterior_state.nc``, the whole window rolled out at the
output cadence -- so its metric stage never has to choose. The filter has two,
and which one exists is a property of how the run was configured:

  * ``forecast`` -- ``_ensemble_states/cycle_{k}/state_{m}.nc``, every frame of
    every member's forecast segment, written only under
    ``run.ensemble_save_on_disk=true``;
  * ``analysis`` -- the default: ``state_history.nc``, ONE analyzed frame per
    cycle, from which no within-cycle statistic can be formed.

``cycle_state_source`` makes that choice once and everything downstream consumes
it without branching again, so this file is mostly about the choice and its
consequences:

  * the choice itself, including the three ways a forecast tree can be present
    but unusable -- a missing cycle, ragged member counts, and the member index
    that must sort numerically because it is what pairs a state file with its
    parameter row (``..._picks_...``, ``..._fall_back_...``, ``..._sort_...``);
  * the rebasing that puts a forecast segment on the run's global clock. It
    anchors on the segment's LAST frame because the analysis updates that frame,
    which is also what makes cycle 0 -- a cold start carrying the model's
    spin-up in front of the cycle proper -- land inside its own cycle rather
    than past it (``..._rebase_...``, ``..._drop_the_cold_start_lead_in``);
  * **frame matching**, the invariant the whole comparison rests on: under
    either source the truth and the ensemble bin into the same cycles with the
    same number of frames in each. Everything the statistics block reports is a
    truth statistic against a member statistic, so a source that shifted one
    side by a frame would still produce a full, plausible, wrong summary
    (``..._bin_the_truth_and_the_ensemble_onto_the_same_frames``);
  * the documented degradation of the analysis source -- a ddof=1 variance of
    one sample is null -- pinned as a fact rather than left as prose
    (``..._null_the_per_cycle_variance_...``);
  * where that degradation has to be VISIBLE: it reaches ``eval_fields.nc`` as
    ``moment_sampling`` and from there the rendered S1, F1 and S5 (the last
    section below). The moments are correctly computed and the truth is sampled
    identically, so the comparison is fair -- what was wrong was calling a
    3-point across-cycle variance "resolved TKE, time-averaged over 12 s". Those
    tests read the figure's own artists back, because a note threaded through
    three layers and then dropped by the renderer produces exactly the PNG this
    is about;
  * the two stages end to end on a synthetic run dir, in the shape
    ``tests/test_evaluation_sensor_statistics.py`` and
    ``tests/test_evaluation_figures.py`` established for the ESMDA pipeline:
    once complete, and once for each way a run dir can be short of an input, so
    that a block degrades to absent-with-a-line and never takes its neighbours
    with it.
"""

from __future__ import annotations

import pathlib
import re
import shutil
from collections.abc import Iterator

import numpy as np
import pytest
import xarray
from evaluation.sensors import window_masks, window_statistics
from matplotlib.figure import Figure

# One cycle of the smoke shape: three cycles so a cycle boundary is a real thing
# to get wrong, four truth frames in each, and four forecast frames per segment
# so the two sources' bins hold the same frames.
_RUN_CYCLES = 3
_RUN_SIM_TIME = 4.0
_RUN_FRAMES = 4
_RUN_TOTAL = _RUN_CYCLES * _RUN_FRAMES
_LEAD_IN = 2  # cycle 0's cold-start spin-up frames, dropped by the rebasing

_N_CELLS = 5
_AXIS = np.linspace(0.0, 20.0, _N_CELLS)

# One x-column of solid cells, so the hit rate takes pylbm's own ``blanking``
# route rather than the truth-variance fallback.
_SOLID_COLUMNS = 1


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


def _truth_dataset(seed: int = 1) -> xarray.Dataset:
    """The truth over the whole horizon, one frame per second."""
    return _velocity_dataset(
        ("time", "z", "y", "x"),
        {"time": _RUN_TOTAL, "z": _N_CELLS, "y": _N_CELLS, "x": _N_CELLS},
        seed,
        {
            "time": np.arange(_RUN_TOTAL, dtype=float),
            "z": _AXIS,
            "y": _AXIS,
            "x": _AXIS,
        },
    )


def _state_history(n_members: int, seed: int = 10) -> xarray.Dataset:
    """``state_history.nc``: the analyzed end-of-cycle frame of every cycle.

    Dimensioned ``(cycle, ensemble, z, y, x)`` and carrying the SCALAR ``time``
    coordinate the filter's concat leaves behind -- the first cycle's
    end-of-segment time, which cannot index the cycles and which
    ``load_analyzed_states`` therefore has to drop.
    """
    history = _velocity_dataset(
        ("cycle", "ensemble", "z", "y", "x"),
        {
            "cycle": _RUN_CYCLES,
            "ensemble": n_members,
            "z": _N_CELLS,
            "y": _N_CELLS,
            "x": _N_CELLS,
        },
        seed,
        {"ensemble": np.arange(n_members), "z": _AXIS, "y": _AXIS, "x": _AXIS},
    )
    return _with_blanking(history.assign_coords(time=_RUN_SIM_TIME))


def _forecast_segment(cycle: int, member: int, lead_in: int) -> xarray.Dataset:
    """One member's forecast segment for one cycle, on the solver's own clock.

    The segment's last frame is the one the analysis updates, so it sits at the
    cycle's end; ``lead_in`` extra frames in front of it stand for the cold
    start's spin-up, which cycle 0 really does carry.
    """
    n_time = _RUN_FRAMES + lead_in
    times = (cycle + 1) * _RUN_SIM_TIME - np.arange(n_time - 1, -1, -1, dtype=float)
    return _with_blanking(
        _velocity_dataset(
            ("time", "z", "y", "x"),
            {"time": n_time, "z": _N_CELLS, "y": _N_CELLS, "x": _N_CELLS},
            100 + 10 * cycle + member,
            {"time": times, "z": _AXIS, "y": _AXIS, "x": _AXIS},
        )
    )


def _write_forecast_tree(
    run_dir: pathlib.Path,
    n_members: int,
    *,
    cycles: list[int] | None = None,
    members_per_cycle: dict[int, int] | None = None,
) -> None:
    """The ``_ensemble_states/cycle_{k}/state_{m}.nc`` tree, whole or partial."""
    for cycle in list(range(_RUN_CYCLES)) if cycles is None else cycles:
        cycle_dir = run_dir / "_ensemble_states" / f"cycle_{cycle}"
        cycle_dir.mkdir(parents=True, exist_ok=True)
        count = (members_per_cycle or {}).get(cycle, n_members)
        for member in range(count):
            _forecast_segment(cycle, member, _LEAD_IN if cycle == 0 else 0).to_netcdf(
                cycle_dir / f"state_{member}.nc"
            )


def _window_forecast_state(
    window: int, n_members: int, frames_per_window: int
) -> xarray.Dataset:
    """One window's ``window_{w}_forecast_state.nc``.

    What ``run.save_forecast_history=true`` writes: the whole window's forecast
    frames for every member, ``(ensemble, time, ...)``, already carrying the
    truth's own frame times — the writer rebases them there, so nothing
    downstream has to.
    """
    times = np.arange(
        window * frames_per_window, (window + 1) * frames_per_window, dtype=float
    )
    return _with_blanking(
        _velocity_dataset(
            ("ensemble", "time", "z", "y", "x"),
            {
                "ensemble": n_members,
                "time": frames_per_window,
                "z": _N_CELLS,
                "y": _N_CELLS,
                "x": _N_CELLS,
            },
            300 + window,
            {
                "ensemble": np.arange(n_members),
                "time": times,
                "z": _AXIS,
                "y": _AXIS,
                "x": _AXIS,
            },
        )
    )


def _write_window_forecast_files(
    run_dir: pathlib.Path,
    n_members: int,
    num_windows: int,
    *,
    windows: list[int] | None = None,
) -> None:
    """The ``windows/window_{w}_forecast_state.nc`` set, whole or partial."""
    windows_dir = run_dir / "windows"
    windows_dir.mkdir(parents=True, exist_ok=True)
    frames_per_window = _RUN_TOTAL // num_windows
    for window in range(num_windows) if windows is None else windows:
        _window_forecast_state(window, n_members, frames_per_window).to_netcdf(
            windows_dir / f"window_{window}_forecast_state.nc"
        )


def _read_truth_access(run_dir: pathlib.Path) -> dict:
    """``truth_access.yaml`` as the two stages read it back."""
    from scripts.esmda._esmda_common import read_yaml

    return read_yaml(run_dir / "truth_access.yaml")


def _filtering_run_dir(
    tmp_path: pathlib.Path,
    *,
    n_members: int = 4,
    forecast_states: bool = False,
    window_forecast_states: bool = False,
    num_windows: int = 1,
    with_params: bool = True,
    with_params_history: bool = True,
    with_ensemble_axis: bool = True,
    eval_fields: bool = True,
    run_summary: bool = True,
    building_height: float | None = None,
) -> pathlib.Path:
    """A filtering run dir complete enough for the metric and figure stages.

    The layout ``scripts/filtering/run_filtering.py`` writes -- note the
    cycle-flavoured ``truth_access.yaml`` keys (``n_per_cycle``, ``num_cycles``)
    beside the slicing keys ``open_truth`` shares with the ESMDA pipeline -- with
    every input a run may legitimately lack made switchable, because "this run
    was configured without it" is the degradation the two stages exist to
    survive: the forecast tree (``run.ensemble_save_on_disk``), the parameter
    artifacts (``filtering.mode=state``), ``params_history.nc``
    (``run.save_history``), and the two artifacts the metric stage itself writes.
    """
    from scripts.esmda._esmda_common import write_yaml

    run_dir = tmp_path / "run"
    run_dir.mkdir(parents=True)

    config: dict[str, object] = {
        "run": {},
        "filtering": {"mode": "joint", "obs_error_std": 0.2},
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
    write_yaml(
        {
            "configuration": {"mode": "joint", "ensemble_size": n_members},
            "timing": {"filter_total_seconds": 1.0},
        },
        run_dir / "run_info.yaml",
    )
    write_yaml(
        [
            {
                "obs_prior_rmse": 1.0 + cycle,
                "obs_posterior_rmse": 0.5 + cycle,
                "innovation_chi2": 1.1,
                "state_spread_prior": 0.3,
                "state_spread_posterior": 0.2,
                "param_spread_prior": 0.4 if with_params else None,
                "param_spread_posterior": 0.3 if with_params else None,
            }
            for cycle in range(_RUN_CYCLES)
        ],
        run_dir / "cycle_diagnostics.yaml",
    )

    _truth_dataset().to_netcdf(run_dir / "true_state.nc")
    write_yaml(
        {
            "true_state_path": str(run_dir / "true_state.nc"),
            "x_offset": 0.0,
            "start_idx": 0,
            "t_offset": 0.0,
            "n_total": _RUN_TOTAL,
            "n_per_cycle": _RUN_FRAMES,
            "num_cycles": _RUN_CYCLES,
            "sim_time": _RUN_SIM_TIME,
            "truth_solver_name": "pylbm",
            "assim_solver_name": "pylbm",
            # The window geometry only the per-window forecast artifact needs
            # (the other sources are indexed by cycle).
            "num_windows": num_windows,
            "n_per_window": _RUN_TOTAL // num_windows,
        },
        run_dir / "truth_access.yaml",
    )

    history = _state_history(n_members)
    if not with_ensemble_axis:
        history = history.isel(ensemble=0, drop=True)
    history.to_netcdf(run_dir / "state_history.nc")
    history.isel(cycle=-1, drop=True).to_netcdf(run_dir / "posterior_state.nc")

    if forecast_states:
        _write_forecast_tree(run_dir, n_members)
    if window_forecast_states:
        _write_window_forecast_files(run_dir, n_members, num_windows)

    # The filter's parameter artifacts: a SCALAR estimate per member (no time
    # axis -- the metric stage is the thing that labels it with the run's end
    # time), the prior it started from, and the stacked
    # ``[prior, cycle 0, cycle 1, ...]`` history.
    members = np.linspace(-1.0, 1.0, n_members)
    if with_params:
        for name, spread in (("posterior_params", 1.0), ("prior_params", 4.0)):
            xarray.Dataset(
                {"velocity_magnitude": (("ensemble",), 5.0 + spread * members)},
                coords={"ensemble": np.arange(n_members)},
            ).to_netcdf(run_dir / f"{name}.nc")
        if with_params_history:
            xarray.Dataset(
                {
                    "velocity_magnitude": (
                        ("cycle", "ensemble"),
                        5.0 + np.outer(np.linspace(4.0, 1.0, _RUN_CYCLES + 1), members),
                    )
                },
                coords={"ensemble": np.arange(n_members)},
            ).to_netcdf(run_dir / "params_history.nc")
    xarray.Dataset(
        {"velocity_magnitude": (("ensemble",), np.array([5.0]))},
        coords={"ensemble": [0]},
    ).to_netcdf(run_dir / "true_params.nc")

    if eval_fields or run_summary:
        from scripts.filtering.compute_filtering_metrics import compute_metrics

        compute_metrics(run_dir)
        if not eval_fields:
            (run_dir / "eval_fields.nc").unlink()
        if not run_summary:
            (run_dir / "run_summary.yaml").unlink()
    return run_dir


# ---------------------------------------------------------------------------
# Which per-cycle ensemble states the evaluation reduces over
# ---------------------------------------------------------------------------

_TRUTH_ACCESS = {
    "n_per_cycle": _RUN_FRAMES,
    "num_cycles": _RUN_CYCLES,
    "sim_time": _RUN_SIM_TIME,
}

# The same horizon as a WINDOWED run records it: ``sim_time`` keeps the ESMDA
# meaning the ESMDA stages read off this file (seconds per assimilation window),
# and one window holds several cycles -- one per observation frame -- so the
# cycle length is its own key. Here three cycles of _RUN_SIM_TIME make one
# window, which is the ratio the two keys must not be confused across.
_CYCLES_PER_WINDOW = 3
_WINDOWED_TRUTH_ACCESS = {
    "n_per_cycle": _RUN_FRAMES,
    "num_cycles": _RUN_CYCLES,
    "sim_time": _RUN_SIM_TIME * _CYCLES_PER_WINDOW,
    "cycle_seconds": _RUN_SIM_TIME,
}


def test_the_cycle_length_falls_back_to_sim_time_on_a_pre_windowing_run_dir() -> None:
    # Every filtering run dir written before the windowing refactor has one
    # ``sim_time`` that IS the cycle length, and no ``cycle_seconds`` at all.
    # The fallback is what keeps those dirs reading exactly as they did.
    from scripts.filtering._filtering_common import cycle_seconds

    assert cycle_seconds(_TRUTH_ACCESS) == _RUN_SIM_TIME


def test_the_cycle_length_is_the_cycle_key_not_the_window_key_when_both_exist() -> None:
    # The collision this helper exists for: on a windowed dir both keys are
    # present and mean different numbers, and reading ``sim_time`` here would
    # stretch every cycle by the cycles per window -- silently, since the result
    # is still a plausible time axis.
    from scripts.filtering._filtering_common import cycle_seconds

    assert cycle_seconds(_WINDOWED_TRUTH_ACCESS) == _RUN_SIM_TIME
    assert _WINDOWED_TRUTH_ACCESS["sim_time"] != _RUN_SIM_TIME


def test_the_windowed_keys_put_the_parameter_history_on_the_cycle_clock(
    tmp_path: pathlib.Path,
) -> None:
    # The site the fallback matters most at: a drifting truth is interpolated
    # onto this axis, so a cycle clock read as a window clock samples the truth
    # at the wrong times (and produces a full, plausible, wrong parameter error).
    from scripts.filtering._filtering_common import load_params_history

    run_dir = _filtering_run_dir(tmp_path, eval_fields=False, run_summary=False)
    windowed = load_params_history(run_dir, _WINDOWED_TRUTH_ACCESS)
    legacy = load_params_history(run_dir, _TRUTH_ACCESS)

    # Entry 0 is the prior at t=0 and entry i is cycle (i-1)'s posterior at
    # ``i * cycle_seconds`` -- the same axis both dirs describe, since the two
    # runs' cycles are the same length; only the KEY it is read from differs.
    expected = np.arange(_RUN_CYCLES + 1, dtype=float) * _RUN_SIM_TIME
    np.testing.assert_allclose(windowed["time"].values, expected)
    np.testing.assert_allclose(legacy["time"].values, expected)


def test_the_windowed_keys_bin_the_forecast_states_by_the_cycle(
    tmp_path: pathlib.Path,
) -> None:
    # ``bin_seconds`` is what makes one bin one cycle in the window statistics.
    # Read off ``sim_time`` on a windowed dir it would be a window, pooling
    # several cycles into one bin.
    from scripts.filtering._filtering_common import cycle_state_source

    run_dir = _filtering_run_dir(
        tmp_path, forecast_states=True, eval_fields=False, run_summary=False
    )
    source = cycle_state_source(run_dir, dict(_WINDOWED_TRUTH_ACCESS))

    assert source.kind == "forecast"
    assert source.bin_seconds == _RUN_SIM_TIME


def test_cycle_state_source_picks_the_forecasts_when_the_run_saved_them(
    tmp_path: pathlib.Path,
) -> None:
    # The stronger source, and the only one that can carry a within-cycle
    # statistic: cycle k's forecast is the ensemble rolled out under cycle k-1's
    # analysis, resolved in time. Picking the analyzed frames instead when these
    # are on disk would silently null every variance the run paid to produce.
    from scripts.filtering._filtering_common import cycle_state_source

    run_dir = _filtering_run_dir(
        tmp_path, forecast_states=True, eval_fields=False, run_summary=False
    )

    source = cycle_state_source(run_dir, {**_TRUTH_ACCESS})

    assert source.kind == "forecast"
    assert source.num_cycles == _RUN_CYCLES
    assert source.n_members == 4
    assert source.bin_seconds == _RUN_SIM_TIME
    assert source.paths is not None and len(source.paths) == _RUN_CYCLES


def test_cycle_state_source_falls_back_to_the_analyzed_frames_by_default(
    tmp_path: pathlib.Path,
) -> None:
    # ``run.ensemble_save_on_disk`` is off by default, so this is what almost
    # every run dir actually holds. The bin unit changes with it: the analyzed
    # series carry an integer cycle index, not seconds, and binning those by the
    # physical cycle length would put every cycle in bin 0.
    from scripts.filtering._filtering_common import cycle_state_source

    run_dir = _filtering_run_dir(tmp_path, eval_fields=False, run_summary=False)

    source = cycle_state_source(run_dir, {**_TRUTH_ACCESS})

    assert source.kind == "analysis"
    assert source.bin_seconds == 1.0
    assert source.num_cycles == _RUN_CYCLES
    assert source.paths is None
    assert "ONE frame per cycle" in source.description


def test_cycle_state_source_falls_back_when_a_cycle_of_forecasts_is_missing(
    tmp_path: pathlib.Path, caplog: pytest.LogCaptureFixture
) -> None:
    # A job killed mid-run, or a tree pruned by hand. Scoring the cycles that
    # survived would put a shorter horizon in the same statistic as the full
    # ones, and nothing downstream would say so -- the fallback is the honest
    # answer, and it has to be logged or the operator cannot tell which source
    # produced their numbers.
    from scripts.filtering._filtering_common import cycle_state_source

    run_dir = _filtering_run_dir(tmp_path, eval_fields=False, run_summary=False)
    _write_forecast_tree(run_dir, 4, cycles=[0, 2])  # cycle 1 never landed

    with caplog.at_level("INFO"):
        source = cycle_state_source(run_dir, {**_TRUTH_ACCESS})

    assert source.kind == "analysis"
    assert "No forecast states for cycle 1" in caplog.text


def test_cycle_state_source_falls_back_when_the_cycles_hold_different_member_counts(
    tmp_path: pathlib.Path, caplog: pytest.LogCaptureFixture
) -> None:
    # Ragged cycles are worse than no forecasts at all: the fair scores' M(M-1)
    # corrections are per statistic, so a three-member cycle scored beside a
    # four-member one reports a spread neither ensemble has.
    from scripts.filtering._filtering_common import cycle_state_source

    run_dir = _filtering_run_dir(tmp_path, eval_fields=False, run_summary=False)
    _write_forecast_tree(run_dir, 4, members_per_cycle={1: 3})

    with caplog.at_level("WARNING"):
        source = cycle_state_source(run_dir, {**_TRUTH_ACCESS})

    assert source.kind == "analysis"
    assert "different member counts per cycle" in caplog.text


def test_cycle_state_source_picks_the_window_forecasts_without_the_on_disk_tree(
    tmp_path: pathlib.Path,
) -> None:
    # ``run.save_forecast_history=true`` carries the same frames as the on-disk
    # tree in one file per window, so a run that saved them must be scored as a
    # ``forecast`` -- same kind, same bin unit, same truth pairing -- and not
    # dropped to the analyzed frames just because the per-member tree is absent.
    from scripts.filtering._filtering_common import cycle_state_source

    run_dir = _filtering_run_dir(
        tmp_path, window_forecast_states=True, eval_fields=False, run_summary=False
    )

    source = cycle_state_source(run_dir, _read_truth_access(run_dir))

    assert source.kind == "forecast"
    assert source.bin_seconds == _RUN_SIM_TIME
    assert source.num_cycles == _RUN_CYCLES
    assert source.n_members == 4
    assert source.paths is None
    assert source.window_paths is not None and len(source.window_paths) == 1
    assert "full output cadence" in source.description


def test_the_on_disk_forecast_tree_is_preferred_over_the_window_artifact(
    tmp_path: pathlib.Path,
) -> None:
    # Both hold the same frames, so the choice is only about which route runs;
    # keeping the per-member tree first is what makes a run that has always had
    # it score exactly as it did before the artifact existed.
    from scripts.filtering._filtering_common import cycle_state_source

    run_dir = _filtering_run_dir(
        tmp_path,
        forecast_states=True,
        window_forecast_states=True,
        eval_fields=False,
        run_summary=False,
    )

    source = cycle_state_source(run_dir, _read_truth_access(run_dir))

    assert source.paths is not None
    assert source.window_paths is None


def test_cycle_state_source_falls_back_when_a_window_forecast_file_is_missing(
    tmp_path: pathlib.Path, caplog: pytest.LogCaptureFixture
) -> None:
    # Same reasoning as the missing cycle above: a partial set would score one
    # window over the full horizon and another over none of it.
    from scripts.filtering._filtering_common import cycle_state_source

    run_dir = _filtering_run_dir(
        tmp_path, eval_fields=False, run_summary=False, num_windows=3
    )
    _write_window_forecast_files(run_dir, 4, 3, windows=[0, 2])  # window 1 never landed

    with caplog.at_level("WARNING"):
        source = cycle_state_source(run_dir, _read_truth_access(run_dir))

    assert source.kind == "analysis"
    assert "2 of 3 window forecast artifacts" in caplog.text


def test_cycle_state_source_falls_back_when_a_window_forecast_file_is_short(
    tmp_path: pathlib.Path, caplog: pytest.LogCaptureFixture
) -> None:
    # The frames are paired with the truth's per-cycle blocks POSITIONALLY, so a
    # window file that does not tile its window would slide every statistic by
    # the difference and still produce a full summary.
    from scripts.filtering._filtering_common import cycle_state_source

    run_dir = _filtering_run_dir(
        tmp_path, window_forecast_states=True, eval_fields=False, run_summary=False
    )
    path = run_dir / "windows" / "window_0_forecast_state.nc"
    short = xarray.open_dataset(path).isel(time=slice(0, _RUN_TOTAL - 1)).load()
    path.unlink()
    short.to_netcdf(path)

    with caplog.at_level("WARNING"):
        source = cycle_state_source(run_dir, _read_truth_access(run_dir))

    assert source.kind == "analysis"
    assert "truth frame(s)" in caplog.text


def test_the_window_forecast_route_feeds_the_collector_one_cycle_at_a_time(
    tmp_path: pathlib.Path,
) -> None:
    # ``on_member`` is how the mean fields are accumulated inside the sensor
    # pass, and its first argument is a CYCLE index on the on-disk route. One
    # window holding several cycles must not turn that into a window index: the
    # collector would then report the run's chunk count as the window count and
    # average over cycle boundaries it is supposed to resolve.
    from scripts.filtering._filtering_common import (
        build_sensor_sets,
        cycle_sensor_series,
        cycle_state_source,
        load_run_config,
    )

    run_dir = _filtering_run_dir(
        tmp_path, window_forecast_states=True, eval_fields=False, run_summary=False
    )
    ta = _read_truth_access(run_dir)
    source = cycle_state_source(run_dir, ta)
    seen: list[tuple[int, int, int]] = []
    cycle_sensor_series(
        run_dir,
        ta,
        source,
        build_sensor_sets(load_run_config(run_dir)),
        on_member=lambda cycle, member, state: seen.append(
            (cycle, member, int(state.sizes["time"]))
        ),
    )

    # One chunk per (cycle, member), each holding exactly the cycle's frames --
    # from ONE file covering all three cycles.
    assert sorted(seen) == sorted(
        (cycle, member, _RUN_FRAMES)
        for cycle in range(_RUN_CYCLES)
        for member in range(4)
    )


def test_forecast_cycle_paths_sort_the_members_numerically_not_lexically(
    tmp_path: pathlib.Path,
) -> None:
    # The member index is what pairs a forecast file with its parameter row, and
    # ``paths[k][m]`` is handed straight to ``expand_dims(ensemble=[m])``. Sorted
    # as text, ``state_10.nc`` lands before ``state_2.nc`` and every member from
    # 2 up is relabelled -- silently, and only on ensembles of eleven or more,
    # which is every production run and no smoke test.
    from scripts.filtering._filtering_common import _forecast_cycle_paths

    for cycle in range(_RUN_CYCLES):
        cycle_dir = tmp_path / "_ensemble_states" / f"cycle_{cycle}"
        cycle_dir.mkdir(parents=True)
        for member in range(12):
            (cycle_dir / f"state_{member}.nc").touch()

    paths = _forecast_cycle_paths(tmp_path, _RUN_CYCLES)

    assert paths is not None
    assert [p.stem for p in paths[0]] == [f"state_{m}" for m in range(12)]


def test_forecast_cycle_paths_ignore_a_file_that_carries_no_member_index(
    tmp_path: pathlib.Path,
) -> None:
    # The numeric sort above reads the member index out of the filename, so a
    # file that matches ``state_*.nc`` without carrying an integer -- a partial
    # write renamed by hand, a stray ``state_final.nc`` -- used to raise
    # ``ValueError`` out of the sort key itself. That is the worst place for it:
    # it is not a scoring failure that costs one block, it is an exception from
    # inside the source-selection step, so one stray file in a scratch directory
    # took the whole metric or figure stage down. It must be skipped instead.
    from scripts.filtering._filtering_common import _forecast_cycle_paths

    for cycle in range(_RUN_CYCLES):
        cycle_dir = tmp_path / "_ensemble_states" / f"cycle_{cycle}"
        cycle_dir.mkdir(parents=True)
        for member in range(3):
            (cycle_dir / f"state_{member}.nc").touch()
        (cycle_dir / "state_final.nc").touch()

    paths = _forecast_cycle_paths(tmp_path, _RUN_CYCLES)

    assert paths is not None
    assert [p.stem for p in paths[0]] == ["state_0", "state_1", "state_2"]


# ---------------------------------------------------------------------------
# Placing a forecast segment on the run's global clock
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("cycle", [0, 1, 2])  # type: ignore[misc]
def test_rebased_forecast_times_land_a_segment_inside_its_own_cycle(
    cycle: int,
) -> None:
    # The span is the contract the binning downstream is written against: cycle
    # k owns ``[k*sim_time, (k+1)*sim_time)``, the same half-open block the
    # truth's ``[k*n_per_cycle, (k+1)*n_per_cycle)`` frames cover. Anchoring on
    # the segment's first frame instead would push every cycle one dt late and
    # spill its last frame into the next bin.
    from scripts.filtering._filtering_common import _rebased_forecast_times

    times = (cycle + 1) * _RUN_SIM_TIME - np.arange(_RUN_FRAMES - 1, -1, -1)

    rebased, keep = _rebased_forecast_times(times, cycle, _RUN_SIM_TIME)

    assert keep.all()
    assert rebased[0] == pytest.approx(cycle * _RUN_SIM_TIME)
    assert rebased[-1] == pytest.approx((cycle + 1) * _RUN_SIM_TIME - 1.0)
    assert (rebased < (cycle + 1) * _RUN_SIM_TIME).all()


def test_rebased_forecast_times_drop_exactly_the_cold_start_lead_in() -> None:
    # Cycle 0 is a cold start, so its segment carries the model's spin-up in
    # front of the cycle proper. Those frames are not part of the horizon the
    # truth's first block covers, and averaging them into cycle 0's statistic
    # would compare a spun-up truth against a spinning-up ensemble -- but only
    # the lead-in may go, or cycle 0 is scored over fewer frames than its peers.
    from scripts.filtering._filtering_common import _rebased_forecast_times

    times = _RUN_SIM_TIME - np.arange(_RUN_FRAMES + _LEAD_IN - 1, -1, -1, dtype=float)

    rebased, keep = _rebased_forecast_times(times, 0, _RUN_SIM_TIME)

    assert int(keep.sum()) == _RUN_FRAMES
    assert list(np.flatnonzero(keep)) == list(range(_LEAD_IN, _RUN_FRAMES + _LEAD_IN))
    assert (rebased[keep] >= 0.0).all()
    assert (rebased[~keep] < 0.0).all()


# ---------------------------------------------------------------------------
# Frame matching: the invariant the whole comparison rests on
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(  # type: ignore[misc]
    "route", ["on_disk_forecast", "window_forecast", "analysis"]
)
def test_the_truth_and_the_ensemble_bin_onto_the_same_frames_under_either_source(
    tmp_path: pathlib.Path, route: str
) -> None:
    # Every number in ``sensor_statistics`` is a truth statistic scored against
    # a member statistic over the same cycle, so the two series have to bin the
    # same way: the same number of bins, and the same number of frames in each.
    # A source that paired the truth's four frames per cycle against a single
    # analyzed frame -- or against a segment still carrying its spin-up -- would
    # produce a complete, plausible, wrong summary rather than an error.
    from scripts.filtering._filtering_common import (
        build_sensor_sets,
        cycle_sensor_series,
        cycle_state_source,
        load_run_config,
        read_yaml,
        truth_cycle_statistics_series,
    )

    run_dir = _filtering_run_dir(
        tmp_path,
        forecast_states=route == "on_disk_forecast",
        window_forecast_states=route == "window_forecast",
        eval_fields=False,
        run_summary=False,
    )
    ta = read_yaml(run_dir / "truth_access.yaml")
    sensor_sets = build_sensor_sets(load_run_config(run_dir))
    source = cycle_state_source(run_dir, ta)
    truth_series = truth_cycle_statistics_series(ta, source, sensor_sets)
    ensemble_series = cycle_sensor_series(run_dir, ta, source, sensor_sets)

    def _per_bin(series: xarray.DataArray) -> list[int]:
        masks = window_masks(
            series["time"].values, source.bin_seconds, source.num_cycles
        )
        return [int(mask.sum()) for mask in masks]

    truth_bins = _per_bin(truth_series["assimilation"])
    assert len(truth_bins) == _RUN_CYCLES
    assert truth_bins == _per_bin(ensemble_series["assimilation"])
    # Not vacuously equal: both forecast routes really do resolve the cycle, and
    # the analysis source really does hold one frame of it.
    expected = 1 if route == "analysis" else _RUN_FRAMES
    assert truth_bins == [expected] * _RUN_CYCLES


def test_the_esmda_state_block_pairs_the_truth_with_the_analyzed_frames() -> None:
    # The same invariant one stage over: the ESMDA metric/figure stages reduce a
    # filtering run dir too, and their state block pairs truth frame k with
    # ``posterior_state_mean.nc`` frame k positionally. A run that analyzes one
    # frame per cycle of ``n_per_cycle`` observation frames keeps every truth
    # frame, so the two series are that factor apart -- paired raw, the whole
    # horizon's analyses would be scored against the truth's leading 1/n, which
    # is a complete, plausible, wrong RMSE rather than an error.
    from evaluation.turbulence import streaming_state_rmse

    from scripts.esmda._esmda_common import analyzed_truth_frames

    truth = _truth_dataset()
    # What the per-window state files hold: the last frame of each cycle's
    # block, and nothing else.
    analyzed = truth.isel(time=[(c + 1) * _RUN_FRAMES - 1 for c in range(_RUN_CYCLES)])

    selected = analyzed_truth_frames(
        truth, {"n_per_cycle": _RUN_FRAMES, "num_cycles": _RUN_CYCLES}
    )
    np.testing.assert_array_equal(
        np.asarray(selected["time"].values), np.asarray(analyzed["time"].values)
    )
    assert float(np.max(streaming_state_rmse(selected, analyzed))) == 0.0
    # Not vacuous: the unselected pairing really does score the wrong frames.
    assert float(np.max(streaming_state_rmse(truth, analyzed))) > 0.0

    # An ESMDA run dir carries no cycle keys at all and an every-frame filtering
    # run carries 1; both must get their truth back untouched.
    for unstrided in ({}, {"n_per_cycle": 1}):
        assert analyzed_truth_frames(truth, unstrided) is truth


def test_the_analysis_source_nulls_the_per_cycle_variance_and_keeps_the_mean(
    tmp_path: pathlib.Path,
) -> None:
    # The documented degradation, pinned so it cannot quietly become something
    # else: one frame per cycle gives a ddof=1 variance of a single sample, so
    # the variance statistics are null by construction while the mean is a real
    # (if unaveraged) number. A reduction that answered ddof=0 here would report
    # a per-cycle variance of exactly zero -- an ensemble that agrees perfectly
    # about the turbulence -- instead of admitting it measured nothing.
    from scripts.filtering._filtering_common import (
        build_sensor_sets,
        cycle_sensor_series,
        cycle_state_source,
        load_run_config,
        read_yaml,
    )

    run_dir = _filtering_run_dir(tmp_path, eval_fields=False, run_summary=False)
    ta = read_yaml(run_dir / "truth_access.yaml")
    sensor_sets = build_sensor_sets(load_run_config(run_dir))
    source = cycle_state_source(run_dir, ta)
    series = cycle_sensor_series(run_dir, ta, source, sensor_sets)["assimilation"]

    stats = window_statistics(series, source.bin_seconds, source.num_cycles)

    assert np.isfinite(stats["mean"].values).all()
    assert np.isnan(stats["variance"].values).all()


def test_the_forecast_source_measures_a_real_within_cycle_variance(
    tmp_path: pathlib.Path,
) -> None:
    # The other half of the same statement, and the reason the forecast source
    # is worth resolving at all: with the segment's frames in the bin the
    # variance is a genuine within-cycle number, not a null.
    from scripts.filtering._filtering_common import (
        build_sensor_sets,
        cycle_sensor_series,
        cycle_state_source,
        load_run_config,
        read_yaml,
    )

    run_dir = _filtering_run_dir(
        tmp_path, forecast_states=True, eval_fields=False, run_summary=False
    )
    ta = read_yaml(run_dir / "truth_access.yaml")
    sensor_sets = build_sensor_sets(load_run_config(run_dir))
    source = cycle_state_source(run_dir, ta)
    series = cycle_sensor_series(run_dir, ta, source, sensor_sets)["assimilation"]

    stats = window_statistics(series, source.bin_seconds, source.num_cycles)

    assert np.isfinite(stats["mean"].values).all()
    assert np.isfinite(stats["variance"].values).all()


# ---------------------------------------------------------------------------
# run_summary.yaml -- scripts/filtering/compute_filtering_metrics.py
# ---------------------------------------------------------------------------


def test_run_summary_records_which_cycle_states_the_blocks_reduced_over(
    tmp_path: pathlib.Path,
) -> None:
    # The two sources answer the same questions at materially different
    # fidelity, and a reader of run_summary.yaml has no other way to tell which
    # numbers they are holding. Recorded, not implied.
    from scripts.esmda._esmda_common import read_yaml
    from scripts.filtering.compute_filtering_metrics import compute_metrics

    run_dir = _filtering_run_dir(
        tmp_path, forecast_states=True, eval_fields=False, run_summary=False
    )
    compute_metrics(run_dir)
    block = read_yaml(run_dir / "run_summary.yaml")["cycle_states"]

    assert block["kind"] == "forecast"
    assert block["num_cycles"] == _RUN_CYCLES
    assert block["n_members"] == 4
    assert "_ensemble_states" in block["description"]


def test_run_summary_carries_the_cycle_blocks_beside_the_pre_existing_ones(
    tmp_path: pathlib.Path,
) -> None:
    # The WP1.5 blocks are additive: the parameter / state / sensor / filter
    # diagnostics that shipped before them have to come out untouched, because
    # the metric stage is re-run over old run dirs and its output is diffed.
    from scripts.esmda._esmda_common import read_yaml
    from scripts.filtering.compute_filtering_metrics import compute_metrics

    run_dir = _filtering_run_dir(tmp_path, eval_fields=False, run_summary=False)
    compute_metrics(run_dir)
    summary = read_yaml(run_dir / "run_summary.yaml")

    assert summary["metrics_version"] == 2
    assert summary["configuration"]["ensemble_size"] == 4
    for key in (
        "filter_diagnostics",
        "parameter_metrics",
        "ensemble_health",
        "state_metrics",
        "sensor_metrics",
        "cycle_states",
        "sensor_statistics",
        "field_metrics",
    ):
        assert key in summary, f"{key} is missing from run_summary.yaml"
    assert summary["cycle_states"]["kind"] == "analysis"
    # The per-cycle uniqueness list is indexed by cycle, so the prior that sits
    # at ``params_history``'s entry 0 must not be counted as cycle 0.
    assert len(summary["ensemble_health"]["n_unique_per_cycle"]) == _RUN_CYCLES


def test_run_summary_scores_every_sensor_set_including_the_held_out_one(
    tmp_path: pathlib.Path,
) -> None:
    # A profile scored only where the filter was fitted is the least informative
    # one available; the held-out set is the point of §4.2. And the filter saves
    # no comparable prior rollout, so the prior half is absent by property of the
    # method rather than by omission -- an empty ``prior`` key would read as a
    # prior that scored nothing.
    from scripts.esmda._esmda_common import read_yaml
    from scripts.filtering.compute_filtering_metrics import compute_metrics

    run_dir = _filtering_run_dir(tmp_path, eval_fields=False, run_summary=False)
    compute_metrics(run_dir)
    block = read_yaml(run_dir / "run_summary.yaml")["sensor_statistics"]

    assert set(block) == {"assimilation", "validation"}
    assert block["assimilation"]["n_members"] == 4
    # The shared library's name for the bin count; here one bin is one cycle.
    assert block["assimilation"]["n_windows"] == _RUN_CYCLES
    assert block["validation"]["num_sensors"] == 1
    assert block["validation"]["posterior"]["mean_magnitude"]["crps"]["mean"] > 0
    for half in block.values():
        assert "prior" not in half


def test_run_summary_field_metrics_record_the_cycle_count_the_chunk_count_hides(
    tmp_path: pathlib.Path,
) -> None:
    # ``n_windows`` counts the distinct chunk indices the accumulators were fed,
    # and the analysis source arrives as ONE chunk however many cycles ran -- the
    # whole cycle axis is a single lazily-opened file. Overwriting it would lie
    # about what the collector saw, so the horizon is recorded beside it, and
    # that is the number a reader needs.
    from scripts.esmda._esmda_common import read_yaml
    from scripts.filtering.compute_filtering_metrics import compute_metrics

    run_dir = _filtering_run_dir(tmp_path, eval_fields=False, run_summary=False)
    compute_metrics(run_dir)
    block = read_yaml(run_dir / "run_summary.yaml")["field_metrics"]

    assert block["n_windows"] == 1
    assert block["n_cycles"] == _RUN_CYCLES
    # pylbm ships ``blanking`` beside the state, and the sensor pass hands the
    # collector ``ds[["u", "v", "w"]]``, so it is only found if the stage points
    # the collector at a state FILE. Falling through to the truth-variance rule
    # here would dilute q toward the built-up fraction.
    assert block["solid_cell_source"] == "blanking"
    assert block["solid_fraction"] == pytest.approx(_SOLID_COLUMNS / _N_CELLS)
    assert (run_dir / "eval_fields.nc").is_file()


def test_run_summary_field_metrics_count_the_cycles_as_chunks_under_the_forecasts(
    tmp_path: pathlib.Path,
) -> None:
    # The forecast source feeds the accumulators cycle by cycle, so the two
    # counts agree there. Asserting it is what makes the analysis case above a
    # statement about the source rather than about the collector.
    from scripts.esmda._esmda_common import read_yaml
    from scripts.filtering.compute_filtering_metrics import compute_metrics

    run_dir = _filtering_run_dir(
        tmp_path, forecast_states=True, eval_fields=False, run_summary=False
    )
    compute_metrics(run_dir)
    block = read_yaml(run_dir / "run_summary.yaml")["field_metrics"]

    assert block["n_windows"] == _RUN_CYCLES
    assert block["n_cycles"] == _RUN_CYCLES
    # Every member saw every frame of every segment, the lead-in excluded.
    assert block["frames_per_member"] == [_RUN_CYCLES * _RUN_FRAMES] * 4


def test_run_summary_in_state_mode_loses_the_parameter_blocks_and_nothing_else(
    tmp_path: pathlib.Path,
) -> None:
    # ``filtering.mode=state`` estimates no parameters, so it writes no
    # posterior_params.nc at all. Invariant 3: that costs the two blocks derived
    # from the parameters and leaves the state, sensor and field blocks -- which
    # is the whole of the WP1.5 evaluation -- intact.
    from scripts.esmda._esmda_common import read_yaml
    from scripts.filtering.compute_filtering_metrics import compute_metrics

    run_dir = _filtering_run_dir(
        tmp_path, with_params=False, eval_fields=False, run_summary=False
    )
    compute_metrics(run_dir)
    summary = read_yaml(run_dir / "run_summary.yaml")

    assert "parameter_metrics" not in summary
    assert "ensemble_health" not in summary
    for key in ("cycle_states", "sensor_statistics", "field_metrics"):
        assert key in summary, f"{key} was lost with the parameters"
    # The parameter spreads are the only diagnostics a state run cannot record;
    # the observation-space ones are still there.
    assert "obs_prior_rmse" in summary["filter_diagnostics"]
    assert "param_spread_prior" not in summary["filter_diagnostics"]


def test_run_summary_without_a_params_history_keeps_the_run_wide_health_counts(
    tmp_path: pathlib.Path, caplog: pytest.LogCaptureFixture
) -> None:
    # ``run.save_history`` may be off, and the per-cycle uniqueness counts live
    # only inside params_history.nc -- the filter writes no per-cycle parameter
    # files the way ESMDA does. Their absence costs the list, never the block:
    # the run-wide counts and the pairwise-distance ratio still stand.
    from scripts.esmda._esmda_common import read_yaml
    from scripts.filtering.compute_filtering_metrics import compute_metrics

    run_dir = _filtering_run_dir(
        tmp_path, with_params_history=False, eval_fields=False, run_summary=False
    )
    with caplog.at_level("INFO"):
        compute_metrics(run_dir)
    health = read_yaml(run_dir / "run_summary.yaml")["ensemble_health"]

    assert health["n_unique_per_cycle"] == []
    assert health["n_members"] == 4 and health["n_unique"] == 4
    assert health["min_over_median_pairwise"] is not None
    assert "No params_history.nc" in caplog.text


def test_the_cycle_blocks_drop_a_sensor_set_with_no_members_and_keep_the_fields(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    # Every score in the statistics block is probabilistic, so a cycle series
    # with no ``ensemble`` axis has nothing to score -- and that must cost the
    # set alone: the mean fields need no members and the held-out set's loss
    # must not take the assimilation set's numbers with it.
    #
    # Driven at ``_cycle_evaluation_blocks`` rather than through
    # ``compute_metrics``, because no run dir can reach the guard today: both
    # cycle-state sources derive their members from the same artifacts the
    # unguarded ``vector_sensor_metrics`` call above them reads, so an
    # ensemble-mean-only run dies before it gets here (see the test below).
    import scripts.filtering.compute_filtering_metrics as metrics_stage
    from scripts.filtering._filtering_common import (
        build_sensor_sets,
        cycle_sensor_series,
        load_run_config,
        read_yaml,
    )

    run_dir = _filtering_run_dir(tmp_path, eval_fields=False, run_summary=False)
    ta = read_yaml(run_dir / "truth_access.yaml")
    sensor_sets = build_sensor_sets(load_run_config(run_dir))

    def _without_validation_members(*args: object, **kwargs: object) -> dict:
        series = dict(cycle_sensor_series(*args, **kwargs))  # type: ignore[arg-type]
        series["validation"] = series["validation"].isel(ensemble=0, drop=True)
        return series

    monkeypatch.setattr(
        metrics_stage, "cycle_sensor_series", _without_validation_members
    )
    with caplog.at_level("WARNING"):
        blocks = metrics_stage._cycle_evaluation_blocks(run_dir, ta, sensor_sets)

    assert "No ensemble dimension in the validation cycle sensor series" in caplog.text
    assert set(blocks["sensor_statistics"]) == {"assimilation"}  # type: ignore[call-overload]
    assert blocks["field_metrics"]["n_cycles"] == _RUN_CYCLES  # type: ignore[index]


def test_a_state_history_without_a_members_axis_costs_only_the_ensemble_blocks(
    tmp_path: pathlib.Path, caplog: pytest.LogCaptureFixture
) -> None:
    # An ensemble-mean-only state artifact. EVERY sensor block is probabilistic
    # and needs the members -- ``vector_sensor_metrics`` raises outright without
    # them, before the library's own guard is reached -- so the guard has to sit
    # at the script level, as it does in the ESMDA stage. Invariant 3 is the
    # whole point: this must cost the blocks that need an ensemble and nothing
    # else. It used to take the entire stage down, so that the parameter, health,
    # diagnostic and state blocks were all lost and no run_summary.yaml was
    # written at all -- a run dir with no metrics rather than one with fewer.
    from scripts.filtering._filtering_common import read_yaml
    from scripts.filtering.compute_filtering_metrics import compute_metrics

    run_dir = _filtering_run_dir(
        tmp_path,
        with_ensemble_axis=False,
        eval_fields=False,
        run_summary=False,
    )

    with caplog.at_level("WARNING"):
        compute_metrics(run_dir)
    summary = read_yaml(run_dir / "run_summary.yaml")

    assert "No ensemble dimension in the assimilation sensor series" in caplog.text
    # The reason the cycle blocks stood down is stated, not left as a bare
    # KeyError on ``sizes["ensemble"]``.
    assert "no 'ensemble' axis" in caplog.text

    # Dropped: everything that needs members.
    assert summary["sensor_metrics"] == {}
    assert "sensor_statistics" not in summary
    assert "field_metrics" not in summary
    assert "cycle_states" not in summary
    assert not (run_dir / "eval_fields.nc").exists()

    # Kept: everything that does not.
    assert summary["metrics_version"] == 2
    assert summary["parameter_metrics"] and summary["state_metrics"]
    assert summary["ensemble_health"] and summary["filter_diagnostics"]


# ---------------------------------------------------------------------------
# The figures -- scripts/filtering/make_filtering_figures.py
# ---------------------------------------------------------------------------


def test_make_figures_draws_the_whole_set_on_a_complete_run_dir(
    tmp_path: pathlib.Path,
) -> None:
    from scripts.filtering.make_filtering_figures import make_figures

    run_dir = _filtering_run_dir(tmp_path)

    make_figures(run_dir)

    for name in (
        "parameter_evolution.png",
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


def test_every_figure_is_written_on_an_opaque_background(
    tmp_path: pathlib.Path,
) -> None:
    # These PNGs are opened straight out of a run directory, so a transparent
    # background takes the viewer's -- and on a dark-mode viewer the dark axis
    # labels, tick text and truth lines land on dark grey and the figure cannot
    # be read at all. ``evaluation.style.apply_style`` defaults to transparent
    # because it was written for the paper/slide figures in
    # scripts/figure_creation/, which ARE composited onto a page the document
    # owns; ``figures._styled`` pins white over it and each save site passes
    # ``transparent=False``. Both halves are needed -- an explicit ``savefig``
    # keyword overrides ``savefig.transparent``, so the rcParam alone is not
    # enough -- which is exactly the kind of pair that silently comes undone.
    from PIL import Image

    from scripts.filtering.make_filtering_figures import make_figures

    run_dir = _filtering_run_dir(tmp_path)
    make_figures(run_dir)

    for name in (
        "parameter_marginals.png",
        "station_profiles.png",
        "mean_slices.png",
        "sensor_fans.png",
        "rank_histogram.png",
    ):
        with Image.open(run_dir / name) as image:
            alpha = image.convert("RGBA").getchannel("A")
        assert alpha.getextrema()[0] == 255, f"{name} has transparent pixels"


def test_make_figures_skips_what_a_run_dir_without_the_metric_stage_cannot_support(
    tmp_path: pathlib.Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # The figure stage runs on run dirs whose metric stage never ran, or predates
    # these blocks. Three figures cannot be drawn then; the rest must be, and
    # each skip has to name the script that would produce the missing artifact --
    # a line that says only "skipped" leaves the operator nowhere to go.
    from scripts.filtering.make_filtering_figures import make_figures

    run_dir = _filtering_run_dir(tmp_path, eval_fields=False, run_summary=False)

    make_figures(run_dir)

    printed = capsys.readouterr().out
    for name in ("station_profiles.png", "mean_slices.png", "rank_histogram.png"):
        assert not (run_dir / name).exists(), f"{name} was drawn from nothing"
        assert name in printed, f"{name} was skipped silently"
    assert (
        printed.count("compute_filtering_metrics.py") >= 2
    ), "the skips do not say how to fix themselves"
    # A skip never costs the figures after it (master-plan invariant 3).
    for name in ("parameter_marginals.png", "sensor_fans.png"):
        assert (run_dir / name).is_file(), f"{name} was lost to an earlier skip"


def test_make_figures_skips_the_parameter_marginals_in_state_mode(
    tmp_path: pathlib.Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # ``filtering.mode=state`` writes no posterior_params.nc, so P1 has no
    # marginal to draw -- and the sensor and field figures, which do not touch
    # the parameters, must be unaffected.
    from scripts.filtering.make_filtering_figures import make_figures

    run_dir = _filtering_run_dir(tmp_path, with_params=False)

    make_figures(run_dir)

    printed = capsys.readouterr().out
    assert not (run_dir / "parameter_marginals.png").exists()
    assert "filtering.mode=state" in printed
    for name in ("sensor_fans.png", "station_profiles.png", "mean_slices.png"):
        assert (run_dir / name).is_file(), f"{name} was lost with the parameters"


def _captured_kwargs(
    monkeypatch: pytest.MonkeyPatch, *names: str
) -> dict[str, dict[str, object]]:
    """Stub out figure calls in the wiring and record the kwargs they were given.

    What this pins is the *arguments the script computes*, which is where the
    run-dir, config and cycle-state knowledge lives; the figures themselves are
    covered in ``tests/test_evaluation_figures.py``. ``animate_rollout_state``
    goes too -- it is not under test here and it is the slowest thing in the
    stage.
    """
    import scripts.filtering.make_filtering_figures as wiring

    recorded: dict[str, dict[str, object]] = {}

    def _record(name: str) -> object:
        def _stub(*args: object, **kwargs: object) -> None:
            recorded[name] = kwargs
            return None

        return _stub

    for name in names + ("animate_rollout_state",):
        monkeypatch.setattr(wiring, name, _record(name))
    return recorded


def test_make_figures_draws_the_analysis_fan_on_physical_end_of_cycle_seconds(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Under the analysis source the extracted series carry an integer CYCLE
    # INDEX -- state_history.nc records no per-cycle time -- while
    # ``plot_sensor_fans`` labels whatever axis it is handed "Time [s]". Passing
    # the raw indices would print 0, 1, 2 as seconds on a run whose cycles are
    # four seconds long, and would put the shaded cycle boundaries nowhere near
    # the samples. The filter analyzes the END of each segment, so cycle k's
    # frame is the state at ``(k+1)*sim_time``, and converting to that is what
    # makes the label true and puts both sources on one clock.
    from scripts.filtering.make_filtering_figures import make_figures

    recorded = _captured_kwargs(monkeypatch, "plot_sensor_fans")
    run_dir = _filtering_run_dir(tmp_path)

    make_figures(run_dir)

    times = recorded["plot_sensor_fans"]["times"]
    assert times is not None, "S5 fell back to a frame index on a well-formed run"
    assert list(times) == pytest.approx(  # type: ignore[call-overload]
        [(k + 1) * _RUN_SIM_TIME for k in range(_RUN_CYCLES)]
    )
    edges = recorded["plot_sensor_fans"]["window_edges"]
    assert list(edges) == pytest.approx(  # type: ignore[call-overload]
        [k * _RUN_SIM_TIME for k in range(_RUN_CYCLES + 1)]
    )


def test_make_figures_draws_the_forecast_fan_on_the_segments_own_clock(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The forecast series already carry real seconds rebased onto the run's
    # global clock, so they are handed through untouched: every frame of every
    # segment, not one point per cycle. The two sources therefore share one
    # x-axis, which is what makes the cycle boundaries mean the same thing in
    # both figures.
    from scripts.filtering.make_filtering_figures import make_figures

    recorded = _captured_kwargs(monkeypatch, "plot_sensor_fans")
    run_dir = _filtering_run_dir(tmp_path, forecast_states=True)

    make_figures(run_dir)

    times = recorded["plot_sensor_fans"]["times"]
    assert list(times) == pytest.approx(  # type: ignore[call-overload]
        list(np.arange(_RUN_CYCLES * _RUN_FRAMES, dtype=float))
    )
    edges = recorded["plot_sensor_fans"]["window_edges"]
    assert list(edges) == pytest.approx(  # type: ignore[call-overload]
        [k * _RUN_SIM_TIME for k in range(_RUN_CYCLES + 1)]
    )


@pytest.mark.parametrize("height", [None, 12.5])  # type: ignore[misc]
def test_make_figures_passes_a_configured_building_height_to_the_profiles(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, height: float | None
) -> None:
    # S1's ``z/H`` axis and its roof line need a canopy height, which no shipped
    # case carries today (the geometry is an STL, not a config scalar), so an
    # absent key has to leave the profiles in metres rather than raise --
    # CLAUDE.md's no-op-when-absent rule. Note the config path: the ``case``
    # group is ``# @package _global_``, so the scalar sits at the root under
    # ``geometry``, not under a ``case:`` node.
    from scripts.filtering.make_filtering_figures import make_figures

    recorded = _captured_kwargs(monkeypatch, "plot_station_profiles")
    run_dir = _filtering_run_dir(tmp_path, building_height=height)

    make_figures(run_dir)

    assert recorded["plot_station_profiles"]["building_height"] == height
    # ``U_ref`` comes off the truth's ``velocity_magnitude`` on the same call.
    assert recorded["plot_station_profiles"]["u_ref"] == pytest.approx(5.0)


# ---------------------------------------------------------------------------
# Sampling provenance: which frames the moments came from, and whether the
# figures say so
# ---------------------------------------------------------------------------
#
# The blocker these cover: on a DEFAULT filtering run
# (``run.ensemble_save_on_disk=false``) the mean-field collector is fed the
# analyzed end-of-cycle frames alone, so ``eval_fields.nc``'s ``*_tke`` and
# ``*_uw`` are moments over ~``num_cycles`` decorrelated snapshots -- with the
# analysis increments folded into the variance -- and not resolved turbulence
# over a window. The numbers are correctly computed and the truth is sampled
# identically, so the comparison is fair; what was wrong is that S1 labelled the
# row ``k`` and captioned it "Time-averaged profiles" while F1 annotated
# ``t = 0-N s``, and nothing anywhere on either PNG said the average was N
# points rather than N seconds. ``CycleStates.description`` already stated it
# exactly, but it reached only the log and ``run_summary.yaml`` -- neither of
# which the figure stage opens.
#
# These read back the RENDERED figure (the ``captured_figures`` spy, as
# ``tests/test_evaluation_figures.py`` does) rather than the kwargs the wiring
# passed: a note threaded correctly through three layers and then dropped on the
# floor by the renderer produces exactly the misleading PNG this is about.


@pytest.fixture  # type: ignore[misc]
def captured_figures(monkeypatch: pytest.MonkeyPatch) -> list[Figure]:
    """Every ``Figure`` a plot function saves, in save order.

    ``style.save_png`` closes the figure it wrote, but closing only drops it
    from pyplot's registry -- the axes, lines and text survive -- so this reads a
    figure's content back without the figure functions growing an accessor for
    the tests' benefit. Same spy as ``tests/test_evaluation_figures.py``.
    """
    saved: list[Figure] = []
    original = Figure.savefig

    def _spy(self: Figure, *args: object, **kwargs: object) -> object:
        saved.append(self)
        return original(self, *args, **kwargs)

    monkeypatch.setattr(Figure, "savefig", _spy)
    return saved


@pytest.fixture(autouse=True)  # type: ignore[misc]
def _close_figures() -> Iterator[None]:
    """No test leaks a figure into the next one (or into the suite's memory)."""
    import matplotlib.pyplot as plt

    yield
    plt.close("all")


def _figure_text(fig: Figure) -> str:
    """Every string the figure renders, lower-cased, on one whitespace-run.

    Newlines and runs of spaces are collapsed because the sampling note is
    wrapped for the PNG, so the phrase a reader sees as one sentence is not one
    string -- and a test that asserted on the unwrapped form would pass or fail
    on the wrap width.
    """
    parts: list[str] = [t.get_text() for t in fig.texts]
    for ax in fig.axes:
        parts.extend(ax.get_title(loc=loc) for loc in ("left", "center", "right"))
        parts.append(ax.get_xlabel())
        parts.append(ax.get_ylabel())
        parts.extend(t.get_text() for t in ax.texts)
    return " ".join(" | ".join(p for p in parts if p).lower().split())


def _figure_titled(figures: list[Figure], marker: str) -> Figure:
    """The one captured figure whose suptitle contains ``marker``."""
    found = [
        fig
        for fig in figures
        if any(marker.lower() in t.get_text().lower() for t in fig.texts)
    ]
    assert len(found) == 1, f"expected one {marker!r} figure, found {len(found)}"
    return found[0]


# The two claims S1's caption can make about its profiles, matched as patterns
# rather than as literals. The wording is prose that a copy edit may legitimately
# reflow or hyphenate ("continuous time-average"); what these tests are about is
# WHICH of the two claims is made, so they pin the claim and let the phrasing
# move. (``_figure_text`` already collapses the PNG's wrap, so only the hyphen is
# left to absorb.) They stay non-vacuous because they are asserted in opposite
# directions on the two sources: the same text cannot satisfy both.
_DENIES_TIME_AVERAGE = re.compile(r"not a continuous time[- ]averag")
_CLAIMS_TIME_AVERAGED_PROFILES = re.compile(r"time[- ]averaged profiles")


def _fan_median_labels(fig: Figure) -> list[str]:
    """The legend labels S5 attached to its fan medians."""
    return sorted(
        {
            str(line.get_label()).lower()
            for ax in fig.axes
            for line in ax.lines
            if "median" in str(line.get_label()).lower()
        }
    )


@pytest.mark.parametrize("forecast_states", [False, True])  # type: ignore[misc]
def test_eval_fields_records_which_frames_the_moments_were_taken_over(
    tmp_path: pathlib.Path, forecast_states: bool
) -> None:
    # ``eval_fields.nc`` is the ONLY artifact the figure stage opens for S1 and
    # F1, so a caveat that lives only in run_summary.yaml cannot reach them.
    # ``CycleStates.description`` is the single authority on the caveat (its
    # docstring says so), so it is what travels -- not a second phrasing that
    # could drift from it.
    from scripts.filtering._filtering_common import cycle_state_source, read_yaml

    run_dir = _filtering_run_dir(tmp_path, forecast_states=forecast_states)
    source = cycle_state_source(run_dir, read_yaml(run_dir / "truth_access.yaml"))

    with xarray.open_dataset(run_dir / "eval_fields.nc") as fields:
        recorded = fields.attrs["moment_sampling"]
        # The machine-readable half, beside the prose: WHETHER those frames
        # leave gaps in ``t_start``--``t_end``. Recorded separately because both
        # sources carry a note, so the note's presence says nothing about which
        # one this is -- and a figure branching on it would qualify the source
        # that needs no qualification exactly as loudly as the one that does.
        sparse = bool(int(fields.attrs["moment_sampling_is_sparse"]))
        # The generic description is still there: this is an addition, not a
        # replacement, and the ESMDA reader of the same schema depends on it.
        assert "resolved TKE" in fields.attrs["description"]

    assert recorded == source.description
    if forecast_states:
        assert "every frame of every cycle's forecast segment" in recorded
        assert not sparse, (
            "the forecast source accumulates every frame of every cycle's "
            "segment and the segments tile the run, so its moments ARE a "
            "continuous time average"
        )
    else:
        # The default run, and the one the labels were lying about.
        assert "ONE frame per cycle" in recorded
        assert sparse, "one analyzed frame per cycle is not a time average"


def test_the_default_run_marks_the_tke_profiles_as_sampled_not_time_averaged(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    captured_figures: list[Figure],
) -> None:
    # S1's second row is the one that changes meaning: ``k`` over one analyzed
    # frame per cycle is an across-cycle variance carrying the analysis
    # increments, and it renders exactly like resolved turbulence. A reader with
    # only the PNG has to be able to tell, so the row itself is marked and the
    # caption carries the sampling and stops saying "time-averaged".
    from scripts.filtering.make_filtering_figures import make_figures

    _captured_kwargs(monkeypatch)  # the animation only; every figure still draws
    run_dir = _filtering_run_dir(tmp_path)

    make_figures(run_dir)

    fig = _figure_titled(captured_figures, "Vertical profiles at the station columns")
    text = _figure_text(fig)
    assert "one frame per cycle" in text, (
        "the PNG does not say the moments are one frame per cycle: a reader "
        f"comparing the TKE profiles cannot tell ({text!r})"
    )
    # Matched loosely on purpose: what has to survive a copy edit is the DENIAL
    # of a continuous time average, not one spelling of it. ``_figure_text``
    # already absorbs the wrap width; these absorb the hyphen.
    assert _DENIES_TIME_AVERAGE.search(text), (
        "the caption does not deny a continuous time average anywhere a reader "
        f"can see it ({text!r})"
    )
    assert not _CLAIMS_TIME_AVERAGED_PROFILES.search(text), (
        "the caption still claims a time average over a run that sampled "
        f"{_RUN_CYCLES} frames"
    )
    # The mark sits on the TKE row's own axis label, not only in the caption:
    # the caption is one line under a grid of panels and says nothing about
    # WHICH row it qualifies. The mean row is deliberately left unmarked -- it
    # survives sparse sampling as a mean of what was sampled.
    labels = {ax.get_xlabel() for ax in fig.axes if ax.get_xlabel()}
    tke_labels = [label for label in labels if "$k" in label]
    assert tke_labels, f"no TKE row label on the figure ({sorted(labels)})"
    assert all(
        label.rstrip().endswith("*") for label in tke_labels
    ), f"the TKE row is unmarked: {tke_labels}"
    assert not any(
        label.rstrip().endswith("*") for label in labels if "bar{u}" in label
    ), "the mean row was marked too, which blurs what the caption qualifies"


def test_the_default_run_does_not_claim_a_continuous_average_over_the_f1_span(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    captured_figures: list[Figure],
) -> None:
    # F1 annotates ``t_start``/``t_end``, which the metric stage sets to the
    # whole run horizon under BOTH sources -- correctly, since that is the
    # stretch the frames were drawn from. But "Time-mean u on horizontal slices
    # (t = 0-12 s)" asserts the 12 seconds were covered, and on a default run
    # they were visited three times. The span stays (it is the right horizon);
    # what changes is the claim made about it.
    from scripts.filtering.make_filtering_figures import make_figures

    _captured_kwargs(monkeypatch)
    run_dir = _filtering_run_dir(tmp_path)

    make_figures(run_dir)

    fig = _figure_titled(captured_figures, "on horizontal slices")
    text = _figure_text(fig)
    span = f"0-{_RUN_CYCLES * _RUN_SIM_TIME:.0f} s"
    assert (
        f"sampled over t = {span}" in text
    ), f"the span still reads as a continuous average over {span} ({text!r})"
    assert f"(t = {span})" not in text
    assert (
        "time-mean" not in text
    ), "the title and/or the colourbar still call the sampled mean a time mean"
    assert "one frame per cycle" in text, "the sampling itself is not on the figure"


def test_the_forecast_run_keeps_the_f1_time_mean_wording(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    captured_figures: list[Figure],
) -> None:
    # The counterpart, and the reason the qualification is a parameter rather
    # than a constant: with the forecasts on disk the frames ARE every frame of
    # every cycle's segment, the segments tile the run, and the moments are a
    # genuine time average over the annotated span. Qualifying THIS figure is
    # its own misstatement -- and, because F1's qualified wording is identical
    # whichever source produced it, it is the one that costs the qualification
    # its meaning on the run that needs it.
    #
    # So this pins the unqualified wording positively, not merely the absence of
    # the sparse note: the regression it guards against says "every frame of
    # every cycle's forecast segment" one line under a caption reading "means
    # over the sampled frames only".
    from scripts.filtering.make_filtering_figures import make_figures

    _captured_kwargs(monkeypatch)
    run_dir = _filtering_run_dir(tmp_path, forecast_states=True)

    make_figures(run_dir)

    text = _figure_text(_figure_titled(captured_figures, "on horizontal slices"))
    # Where the frames came from is still on the figure -- the note is
    # provenance, and provenance is not a caveat.
    assert "every frame of every cycle's forecast segment" in text
    assert "one frame per cycle" not in text
    # ...and every place the mean is named still names it a time mean: the
    # title, the colourbar, and the span, which was covered continuously here.
    span = f"0-{_RUN_CYCLES * _RUN_SIM_TIME:.0f} s"
    assert "time-mean" in text, f"the mean is no longer named a time mean ({text!r})"
    assert "sample-mean" not in text, (
        "the forecast run's continuous time average is labelled a sample mean, "
        f"which is false and reads identically to the default run's ({text!r})"
    )
    assert f"(t = {span})" in text
    assert f"sampled over t = {span}" not in text
    assert "sampled frames only" not in text, (
        "the caption calls a mean over every frame of every segment a mean over "
        "sampled frames only"
    )


def test_the_forecast_run_keeps_the_s1_time_averaged_wording(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    captured_figures: list[Figure],
) -> None:
    # S1's half of the same statement. Its TKE row is the one that changes
    # meaning under sparse sampling, so under a source that is NOT sparse the
    # row must come back unmarked and the caption must go back to claiming the
    # time average it is entitled to: a ``*`` on every run marks nothing.
    from scripts.filtering.make_filtering_figures import make_figures

    _captured_kwargs(monkeypatch)
    run_dir = _filtering_run_dir(tmp_path, forecast_states=True)

    make_figures(run_dir)

    fig = _figure_titled(captured_figures, "Vertical profiles at the station columns")
    text = _figure_text(fig)
    assert _CLAIMS_TIME_AVERAGED_PROFILES.search(
        text
    ), f"the caption no longer claims the time average these frames are ({text!r})"
    assert not _DENIES_TIME_AVERAGE.search(text), (
        "the caption denies a continuous time average over frames that cover "
        f"every segment of every cycle ({text!r})"
    )
    assert "every frame of every cycle's forecast segment" in text
    labels = {ax.get_xlabel() for ax in fig.axes if ax.get_xlabel()}
    tke_labels = [label for label in labels if "$k" in label]
    assert tke_labels, f"no TKE row label on the figure ({sorted(labels)})"
    assert not any(
        label.rstrip().endswith("*") for label in tke_labels
    ), f"the TKE row is marked on a run whose moments are resolved: {tke_labels}"


def test_the_figures_label_the_frames_the_metrics_actually_saw(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    captured_figures: list[Figure],
) -> None:
    # The reason both halves are read off ``eval_fields.nc`` rather than
    # recomputed from ``cycle_state_source`` in the figure stage: the two CAN
    # disagree. Here the metrics ran with the forecasts on disk and the tree was
    # pruned afterwards (they are the bulkiest thing a run writes), so a figure
    # stage that re-derived its labels would describe frames that never entered
    # the numbers -- calling a continuous time average a three-frame sample, on
    # a file whose numbers are the time average.
    from scripts.filtering.make_filtering_figures import make_figures

    _captured_kwargs(monkeypatch)
    run_dir = _filtering_run_dir(tmp_path, forecast_states=True)
    shutil.rmtree(run_dir / "_ensemble_states")

    make_figures(run_dir)

    text = _figure_text(_figure_titled(captured_figures, "on horizontal slices"))
    assert "every frame of every cycle's forecast segment" in text, (
        "the figure names frames the metric stage never saw: it re-derived the "
        f"sampling from the pruned run dir instead of reading the file ({text!r})"
    )
    assert "one frame per cycle" not in text
    assert "time-mean" in text and "sample-mean" not in text, (
        "the figure re-derived the SPARSENESS from the pruned run dir and "
        f"qualified moments that are a time average ({text!r})"
    )


def test_the_sensor_fan_is_labelled_the_forecast_when_it_draws_the_forecasts(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    captured_figures: list[Figure],
) -> None:
    # S5's fan is whatever the cycle-state source hands it. Under ``forecast``
    # that is every frame of each cycle's forecast segment -- the PRIOR side of
    # each analysis -- so a hardcoded "Posterior median" makes one PNG mean two
    # different things depending only on which artifacts the run wrote.
    from scripts.filtering.make_filtering_figures import make_figures

    _captured_kwargs(monkeypatch)
    run_dir = _filtering_run_dir(tmp_path, forecast_states=True)

    make_figures(run_dir)

    fig = _figure_titled(captured_figures, "quantile fan vs truth")
    assert _fan_median_labels(fig) == [
        "forecast median"
    ], "the fan draws the forecast segments but names them the posterior"
    assert "forecast quantile fan" in _figure_text(fig)


def test_the_sensor_fan_stays_the_posterior_on_the_analyzed_frames(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    captured_figures: list[Figure],
) -> None:
    # The other half of the same statement: on the default source the fan really
    # is the analyzed states, so the label it always had is the correct one and
    # must not have been swapped for a blanket rename.
    from scripts.filtering.make_filtering_figures import make_figures

    _captured_kwargs(monkeypatch)
    run_dir = _filtering_run_dir(tmp_path)

    make_figures(run_dir)

    fig = _figure_titled(captured_figures, "quantile fan vs truth")
    assert _fan_median_labels(fig) == ["posterior median"]
    assert "posterior quantile fan" in _figure_text(fig)
