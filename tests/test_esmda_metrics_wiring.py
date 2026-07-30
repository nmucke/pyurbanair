"""End-to-end wiring of the WP1.1-1.3 bundles through ``compute_metrics``.

``tests/test_parameter_bundle.py`` covers the bundle's math on synthetic arrays
and the ESMDA pipeline test covers the run stage, but neither exercises the
*wiring in between*: which arguments ``compute_metrics`` passes, whether the
`truth_access.yaml` key it reads is spelled the way the run stage writes it, and
whether the flatteners it hands the joint block actually agree in shape on
multi-window artifacts. A typo in ``ta.get("num_windows")`` would silently null
`n_knots_effective` for every static parameter and no test would notice.

The pipeline test cannot cover this: its smoke shape is a 2-member ensemble, so
every numeric key in the bundle takes the ``MIN_MEMBERS_CALIBRATION`` null path
and the math never runs. What makes the gap closable cheaply is that the WP1.1
block sits BEFORE the ``skip_viz`` early return -- it reads only the three
parameter datasets, never the truth state -- so a synthetic run dir plus
``skip_viz: true`` drives the whole wiring with no solver and no truth. The
fixture shape follows ``test_da_metrics.test_skip_viz_summary_has_version_and_
ensemble_health``, widened to M = 8 and a multi-window parameter artifact.

WP1.2's ``sensor_statistics`` sits on the *other* side of that early return, so
it needs the rest of a run dir: a truth state file, per-window ensemble state
files and an obs config. That is what ``_write_state_artifacts`` fabricates --
still no solver, because every artifact the truth-consuming layers read is a
NetCDF file this module writes by hand. Two properties of the fixture are
load-bearing rather than incidental: the truth and the ensemble are saved at
**different cadences** (6 vs 8 frames per window), which is the case the two
window-slicing rules exist for, and the window files carry **window-local**
time coordinates, so ``ensemble_sensor_series``' rebasing onto
``[w*sim_time, (w+1)*sim_time)`` is genuinely exercised rather than assumed.

WP1.3's ``mean_field_metrics`` reads the same artifacts through the *same*
per-member pass as WP1.2 (one shared read of the window state files, which is a
production-scale requirement rather than an optimization), so it needs no new
fixture -- only two extensions. ``_with_blanking`` adds pylbm's solid-cell mask,
because it is the one mask any shipped backend writes and the unmasked branch
flatters a run rather than breaking it; and ``run.metrics.stations`` is exercised
explicitly, since nothing else in the pipeline reads that knob.
"""

from __future__ import annotations

import pathlib

import numpy as np
import pytest
import xarray

from scripts.esmda._esmda_common import (
    PIT_BINS,
    MeanFieldAccumulator,
    mean_field_scores,
    mean_field_summary,
    read_yaml,
    write_yaml,
)
from scripts.esmda.compute_esmda_metrics import compute_metrics

N_MEMBERS = 8
N_WINDOWS = 3
KNOTS_PER_WINDOW = 5
SECONDS_PER_KNOT = 30.0
CORRELATION_LENGTH = 200.0

# What a `parameter_metrics.<name>` entry must contain at `level: standard`:
# the pre-phase-1 keys (hard-coded in scripts/figure_creation/) plus WP1.1's.
BASE_PARAM_KEYS = {
    "rmse",
    "crps",
    "prior_rmse_mean",
    "rmse_reduction_vs_prior",
    "prior_crps_mean",
    "crps_reduction_vs_prior",
}
BUNDLE_PARAM_KEYS = {
    "zscore",
    "pit_counts",
    "pit",
    "sampling",
    "coverage",
    "contraction_ratio",
}
JOINT_KEYS = {
    "n_members",
    "n_parameters",
    "n_sample_directions",
    "rank_deficient",
    "posterior_variance_retained",
    "n_constrained_directions",
    "generalized_eigenvalues",
    "eigenvalue_quantiles",
    "most_constrained",
    "least_constrained",
    "posterior_corr",
    "prior_corr",
    "corr_summary",
    "prior_reference",
    "vs_initial_prior",
}

# The per-window spread ratio the fixture below builds in exactly.
WINDOW_RATIO = 0.6

# --- WP1.2: the synthetic state artifacts the truth-consuming layers read ----

# A pylbm-shaped grid: `ObservationOperator("pylbm")` maps every component onto
# the identity dims {z, y, x}. udales/palm would need staggered coordinates
# (`xm`/`zt`/`yv`) on the fabricated files and `neural_surrogate` is rejected by
# the operator outright, so pylbm is the only solver name a hand-written fixture
# can use without also hand-writing a staggered grid.
GRID_X = np.linspace(0.0, 50.0, 6)
GRID_Y = np.linspace(0.0, 40.0, 5)
GRID_Z = np.linspace(2.0, 20.0, 4)

SIM_TIME = 6.0  # seconds per assimilation window
ENSEMBLE_FRAMES_PER_WINDOW = 8
TRUTH_FRAMES_PER_WINDOW = 6  # deliberately NOT the ensemble's cadence
TRUTH_SPINUP_FRAMES = 2  # dropped by `start_idx`, rebased away by `t_offset`
TRUTH_DT = SIM_TIME / TRUTH_FRAMES_PER_WINDOW
# The truth is saved in its own x frame and shifted back by `x_offset`, so the
# offset is exercised rather than defaulted to a no-op 0.0.
TRUTH_X_OFFSET = 5.0

# `run.metrics.bootstrap_blocks`, chosen against ENSEMBLE_FRAMES_PER_WINDOW so
# the block length lands at 8 / 4 = 2 and the identifiability bootstrap actually
# runs. The shipped default of 20 would put it below `block_bootstrap_std`'s
# resolution (L < 2) and every identifiability number would be null -- the smoke
# behaviour, which is covered by the unit tests, not by this one.
BOOTSTRAP_BLOCKS = 4

# Sensor sets, both strictly inside the grid. The validation set exists so the
# held-out branch of `build_sensor_sets` is covered: it is only present when the
# obs config defines `validation_*_points`, and nothing else in this module
# would notice it being dropped.
ASSIM_POINTS = ([5.0, 25.0, 40.0], [8.0, 20.0, 32.0], [4.0, 10.0, 16.0])
VALIDATION_POINTS = ([12.0, 38.0], [12.0, 28.0], [6.0, 14.0])

# `sensor_statistics.<set>.<stat>` key sets, asserted as SETS for the same
# reason the WP1.1 ones are: figure code and sweep comparisons index these paths
# by name, so a renamed or dropped key is as much a regression as a wrong value.
SENSOR_STATISTIC_NAMES = ("window_mean", "window_variance", "tke", "velmag_mean")
SENSOR_SET_KEYS = {"n_members", "n_windows", "n_sensors", "wasserstein"} | set(
    SENSOR_STATISTIC_NAMES
)
STATISTIC_KEYS = {
    "crps",
    "crpss_vs_prior",
    "zscore",
    "coverage",
    "pit_counts",
    "pit",
    "identifiability",
    "n_samples",
    "n_windows_scored",
    "reason",
}
STATISTIC_ZSCORE_KEYS = {"mean", "std", "max_abs", "max_abs_calibrated_median"}
STATISTIC_COVERAGE_KEYS = {
    "alpha_50",
    "nominal_alpha_50",
    "alpha_90",
    "nominal_alpha_90",
    "max_nominal_alpha",
}
STATISTIC_PIT_KEYS = {"n_bins", "n_samples", "ranks_per_bin", "tie_seed"}
IDENTIFIABILITY_KEYS = {
    "ratio_median",
    "ratio_min",
    "sampling_noise_dominated",
    "threshold",
}
WASSERSTEIN_KEYS = {
    "w1_over_sigma_pooled",
    "w1_over_sigma_member_mean",
    "self_floor",
    "w1_over_floor",
    "w1_over_floor_calibrated_median",
    "reason",
}

# --- WP1.3: the mean-field layer's key sets, pinned for the same reason ------
MEAN_FIELD_KEYS = frozenset(
    {
        "hit_rate",
        "fac2_velmag",
        "fb",
        "nmse",
        "tke_rmse",
        "uw_stress_rmse",
        "averaging",
        "masking",
        "eval_fields",
        "reason",
    }
)
HIT_RATE_KEYS = frozenset({"u", "v", "w", "per_z", "allowance", "relative_tolerance"})
NMSE_KEYS = frozenset({"total", "systematic", "unsystematic"})
AVERAGING_KEYS = frozenset(
    {
        "n_time",
        "n_time_truth",
        "n_windows",
        "n_members",
        "n_members_scored",
        "n_stations",
        "n_stations_with_truth",
        "stride",
        "z_levels",
        "z_indices",
        "z_levels_extrapolated",
        "truth_z_levels",
        "z_offset_max",
        "n_cells",
        "n_scored_cells",
        "n_aggregate_cells",
        "n_stress_cells",
        "n_bootstrap_cells",
        "n_bootstrap_cells_max",
        "bootstrap_seed",
    }
)
MASKING_KEYS = frozenset(
    {
        "source",
        "truth_source",
        "ensemble_source",
        "fluid_fraction",
        "scored_fraction",
        "ensemble_fluid_fraction",
        "truth_finite_fraction",
        "note",
    }
)
EVAL_FIELD_VARS = frozenset(
    {
        "truth_mean",
        "ensemble_mean",
        "ensemble_std",
        "truth_velmag",
        "ensemble_velmag",
        "ensemble_velmag_std",
        "truth_tke",
        "ensemble_tke",
        "ensemble_tke_std",
        "truth_uw",
        "ensemble_uw",
        "ensemble_uw_std",
        "fluid_mask",
        "station_truth_mean",
        "station_ensemble_mean",
        "station_ensemble_std",
        "station_truth_rms",
        "station_ensemble_rms",
        "station_ensemble_rms_std",
        "station_truth_tke",
        "station_ensemble_tke",
        "station_ensemble_tke_std",
        "station_truth_uw",
        "station_ensemble_uw",
        "station_ensemble_uw_std",
        "station_ensemble_mean_quantile",
        "station_ensemble_rms_quantile",
        "station_ensemble_tke_quantile",
        "station_ensemble_uw_quantile",
        "station_fluid_mask",
        "station_x",
        "station_y",
    }
)

# The TOP-LEVEL summary keys a `level: basic` run must have, and only those.
# `run_summary.yaml` key paths are hard-coded across `scripts/figure_creation/`,
# so `basic` reproducing the pre-phase-1 set exactly is the invariant that lets
# an old run dir keep processing -- and the one a "compute it, then drop it"
# gate would silently break.
BASIC_SUMMARY_KEYS = frozenset(
    {
        "configuration",  # the fixture's run_info.yaml payload
        "metrics_version",
        "metrics_level",
        "parameter_metrics",
        "ensemble_health",
        "state_metrics",
        "sensor_metrics",
    }
)
STANDARD_SUMMARY_KEYS = BASIC_SUMMARY_KEYS | {
    "sensor_statistics",
    "mean_field_metrics",
}

# The synthetic building the masking test blanks out: x indices {2, 3} and y
# indices {1, 2}, every z. 4 of the 30 cells in a plane, so the fluid fraction
# is exactly 26/30 -- a closed-form assertion rather than a measured one.
BLANK_X = (2, 3)
BLANK_Y = (1, 2)
BLANK_FLUID_FRACTION = 1.0 - (len(BLANK_X) * len(BLANK_Y)) / (GRID_X.size * GRID_Y.size)


def _write_run_dir(
    run_dir: pathlib.Path,
    posterior: xarray.Dataset,
    prior: xarray.Dataset,
    truth: xarray.Dataset,
    *,
    level: str = "standard",
    num_windows: int | None = N_WINDOWS,
    skip_viz: bool = True,
    extra_config: dict | None = None,
    truth_access: dict | None = None,
    stations: list[list[float]] | None = None,
    stride: int = 1,
) -> None:
    """A run dir carrying exactly what the WP1.1 block reads, and nothing else.

    ``skip_viz`` / ``extra_config`` / ``truth_access`` default to the
    parameter-only shape above. The WP1.2 tests flip ``skip_viz`` and supply the
    obs block and the truth-access keys the truth-consuming layers read; see
    :func:`_write_state_artifacts`, which writes the files they point at.
    ``stations`` is WP1.3's ``run.metrics.stations``; ``None`` (the shipped
    default) means the mean-field layer profiles the sensor positions.
    ``stride`` is ``run.metrics.mean_field_stride``, which both the accumulator
    and the truth's sampling floors have to honour.
    """
    windows_dir = run_dir / "windows"
    windows_dir.mkdir(parents=True, exist_ok=True)
    write_yaml(
        {
            # With `skip_viz` true this returns before any truth-state read,
            # which is why the WP1.1 tests need no solver and no truth file.
            "run": {
                "skip_viz": skip_viz,
                "metrics": {
                    "level": level,
                    "bootstrap_blocks": BOOTSTRAP_BLOCKS,
                    "stations": stations,
                    "mean_field_stride": stride,
                },
            },
            # Where the GP knot spacing actually lives in a saved config.
            "prior_params": {
                "correlation_length": CORRELATION_LENGTH,
                "seconds_per_knot": SECONDS_PER_KNOT,
            },
            **(extra_config or {}),
        },
        run_dir / "config.yaml",
    )
    write_yaml(
        {"configuration": {"ensemble_size": N_MEMBERS}}, run_dir / "run_info.yaml"
    )
    write_yaml(
        ({} if num_windows is None else {"num_windows": num_windows})
        | (truth_access or {}),
        run_dir / "truth_access.yaml",
    )
    posterior.to_netcdf(run_dir / "posterior_params.nc")
    posterior.to_netcdf(windows_dir / "window_0_posterior_params.nc")
    prior.to_netcdf(run_dir / "prior_params.nc")
    truth.to_netcdf(run_dir / "true_params.nc")


def _unit_spread(rng: np.random.Generator, shape: tuple[int, int]) -> np.ndarray:
    """Anomalies whose per-column sample spread is exactly 1 (ddof=1).

    Exact rather than sampled, so the spread ratios asserted below are
    closed-form; independent per call, so the stacked blocks stay full rank
    (a *scalar* multiple of one block would collapse the joint covariance's
    rank and make the joint assertions measure the fixture instead).
    """
    raw = rng.normal(size=shape)
    raw = raw - raw.mean(axis=0, keepdims=True)
    return raw / raw.std(axis=0, ddof=1, keepdims=True)


def _chained_blocks(
    rng: np.random.Generator, n_per_window: int, initial_spread: float
) -> tuple[np.ndarray, np.ndarray]:
    """Prior/posterior anomaly blocks with a real multi-window prior chain.

    This is the structure that makes `contraction_ratio` ambiguous, and the one
    a real run always has: `run_esmda.py` seeds window `w`'s prior from window
    `w - 1`'s posterior, so **only block 0 of `prior_params.nc` is a genuine
    prior** and the spread ratchets down window by window:

        prior spread[w] = initial * WINDOW_RATIO**w
        posterior spread[w] = initial * WINDOW_RATIO**(w + 1)

    A prior that is uniformly wider than the posterior at every knot -- what
    this fixture used to build -- is the one shape a multi-window run never
    produces, and it makes the per-window and cumulative ratios coincide, so
    `0 < contraction_ratio["mean"] < 1` passes whichever one is reported.
    """
    prior_blocks, posterior_blocks = [], []
    spread = initial_spread
    for _ in range(N_WINDOWS):
        prior_blocks.append(_unit_spread(rng, (N_MEMBERS, n_per_window)) * spread)
        spread *= WINDOW_RATIO
        posterior_blocks.append(_unit_spread(rng, (N_MEMBERS, n_per_window)) * spread)
    return (
        np.concatenate(prior_blocks, axis=1),
        np.concatenate(posterior_blocks, axis=1),
    )


def _dynamic_run_artifacts() -> tuple[xarray.Dataset, xarray.Dataset, xarray.Dataset]:
    """A dynamic run: knots concatenated across windows, plus a broadcast static.

    This is the shape `run_esmda._concat_windows` produces -- a time-varying
    `inflow_angle` on a `time` coordinate, and the `static_parameters` block
    (`vertical_inflow_exponent`) broadcast to every knot, i.e. piecewise
    constant with one step per window. Both PIT branches of the bundle fire on
    one dataset, and the joint vector spans both parameters. The prior carries
    the real window chain (see `_chained_blocks`).
    """
    rng = np.random.default_rng(7)
    n_knots = N_WINDOWS * KNOTS_PER_WINDOW
    times = np.arange(n_knots) * SECONDS_PER_KNOT

    centre = 270.0 + np.cumsum(rng.normal(0.0, 0.5, size=n_knots))
    angle_prior_a, angle_post_a = _chained_blocks(rng, KNOTS_PER_WINDOW, 3.0)
    angle_post = centre[None, :] + angle_post_a
    angle_prior = centre[None, :] + angle_prior_a
    angle_truth = centre + rng.normal(0.0, 1.0, size=n_knots)

    # One level per window, then broadcast to that window's knots -- so the
    # static parameter's chain lives on the window axis, not the knot axis.
    levels_prior, levels_post = _chained_blocks(rng, 1, 0.08)
    exp_prior = 0.3 + np.repeat(levels_prior, KNOTS_PER_WINDOW, axis=1)
    exp_post = 0.3 + np.repeat(levels_post, KNOTS_PER_WINDOW, axis=1)

    coords = {"ensemble": np.arange(N_MEMBERS), "time": times}
    posterior = xarray.Dataset(
        {
            "inflow_angle": (("ensemble", "time"), angle_post),
            "vertical_inflow_exponent": (("ensemble", "time"), exp_post),
        },
        coords=coords,
    )
    prior = xarray.Dataset(
        {
            "inflow_angle": (("ensemble", "time"), angle_prior),
            "vertical_inflow_exponent": (("ensemble", "time"), exp_prior),
        },
        coords=coords,
    )
    # Truth on its own, coarser knot grid -- the routine cross-grid case.
    truth_times = np.linspace(times[0], times[-1], n_knots - 4)
    truth = xarray.Dataset(
        {
            "inflow_angle": (("time",), np.interp(truth_times, times, angle_truth)),
            "vertical_inflow_exponent": (("time",), np.full(truth_times.size, 0.31)),
        },
        coords={"time": truth_times},
    )
    return posterior, prior, truth


def _velocity_arrays(
    times: np.ndarray,
    gains: np.ndarray,
    phases: np.ndarray,
    rng: np.random.Generator,
) -> dict[str, np.ndarray]:
    """``{u, v, w}`` of shape ``(member, time, z, y, x)`` for a set of members.

    Deliberately built from three separable pieces, because WP1.2 scores each
    of them and a fixture that carried only one would leave a statistic
    degenerate:

      * a member-dependent sheared mean (``gains``) -- the across-member spread
        the window statistics are supposed to see;
      * a travelling wave in ``t`` with a member-dependent ``phase`` -- so
        members decorrelate pointwise while sharing statistics, which is the
        whole premise of scoring in statistics space;
      * iid noise -- the within-member sampling noise the identifiability
        bootstrap measures. Without it ``block_bootstrap_std`` would return ~0
        and the ratio would be infinite rather than a number.

    The truth is the same generator at ``gains = [1.0]``, ``phases = [0.0]``, so
    truth and ensemble are commensurable by construction rather than by
    coincidence.
    """
    member = np.asarray(gains, dtype=float)[:, None, None, None, None]
    phase = np.asarray(phases, dtype=float)[:, None, None, None, None]
    t = np.asarray(times, dtype=float)[None, :, None, None, None]
    zz = GRID_Z[None, None, :, None, None]
    yy = GRID_Y[None, None, None, :, None]
    xx = GRID_X[None, None, None, None, :]
    shape = (member.shape[0], np.size(times), GRID_Z.size, GRID_Y.size, GRID_X.size)

    shear = np.log1p(zz / 2.0)
    fields = {}
    for name, scale, lag in (("u", 3.0, 0.0), ("v", 1.0, 1.1), ("w", 0.4, 2.3)):
        wave = np.sin(0.35 * t + phase + lag + 0.04 * xx) * np.cos(0.05 * yy)
        fields[name] = (
            scale * member * shear + 0.6 * wave + rng.normal(0.0, 0.2, size=shape)
        )
    return fields


def _with_blanking(fields: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Add pylbm's ``blanking`` variable (1 = solid) for one synthetic building.

    The only solid-cell mask any shipped backend writes, and the reason the
    WP1.3 masking block exists: pyudales ships none and pypalm fills PALM's NaN
    with 0.0, so a fixture that never carries it would only ever exercise the
    unmasked branch. The velocities inside the building are left as they are
    rather than zeroed -- pylbm writes near-zeros there, but the *point* is
    that nothing about the velocity field identifies a solid cell, so the mask
    has to come from this variable alone.
    """
    blanking = np.zeros_like(next(iter(fields.values())))
    for i in BLANK_X:
        for j in BLANK_Y:
            blanking[..., j, i] = 1.0
    return {**fields, "blanking": blanking}


def _state_dataset(
    fields: dict[str, np.ndarray], times: np.ndarray, x: np.ndarray
) -> xarray.Dataset:
    """``(ensemble, time, z, y, x)`` Dataset on the pylbm identity grid."""
    dims = ("ensemble", "time", "z", "y", "x")
    return xarray.Dataset(
        {name: (dims, values) for name, values in fields.items()},
        coords={
            "ensemble": np.arange(next(iter(fields.values())).shape[0]),
            "time": np.asarray(times, dtype=float),
            "z": GRID_Z,
            "y": GRID_Y,
            "x": x,
        },
    )


def _write_state_artifacts(
    run_dir: pathlib.Path,
    rng: np.random.Generator,
    *,
    with_prior: bool = False,
    with_blanking: bool = False,
) -> tuple[dict, dict]:
    """Fabricate the truth + ensemble state files WP1.2 reads, no solver involved.

    Returns the ``obs`` config block and the ``truth_access.yaml`` keys that
    point at them, to be handed to :func:`_write_run_dir`. Writing the two
    together is the point: the key *names* here are the contract between the run
    stage and the metrics stage, and a typo in either would show up as a missing
    layer rather than an error.

    ``with_prior`` additionally writes ``windows/window_{w}_prior_state.nc`` for
    a deliberately biased, over-wide prior. That file set is what
    ``compute_esmda_metrics._prior_sensor_series`` looks for, and it is absent on
    a normal run (``conf/run_esmda.yaml`` ships ``run.save_prior_state: false``),
    so both branches of that guard need a test.
    """
    windows_dir = run_dir / "windows"
    windows_dir.mkdir(parents=True, exist_ok=True)

    # --- truth: global time axis, its own x frame, its own cadence -----------
    n_truth = N_WINDOWS * TRUTH_FRAMES_PER_WINDOW
    file_times = np.arange(TRUTH_SPINUP_FRAMES + n_truth) * TRUTH_DT
    truth_fields = _velocity_arrays(file_times, np.array([1.0]), np.array([0.0]), rng)
    if with_blanking:
        truth_fields = _with_blanking(truth_fields)
    truth_path = run_dir / "true_state.nc"
    _state_dataset(truth_fields, file_times, GRID_X - TRUTH_X_OFFSET).isel(
        ensemble=0, drop=True
    ).to_netcdf(truth_path)

    # --- ensemble: one file per window, window-LOCAL time coordinates --------
    window_times = np.arange(ENSEMBLE_FRAMES_PER_WINDOW) * (
        SIM_TIME / ENSEMBLE_FRAMES_PER_WINDOW
    )
    gains = 1.0 + rng.normal(0.0, 0.12, size=N_MEMBERS)
    phases = rng.normal(0.0, 0.25, size=N_MEMBERS)
    # A prior that is both biased and over-wide, so it loses on the means AND on
    # the second moments -- an unbiased-but-wide prior would leave the variance
    # statistic's skill at ~0 and the assertion would be measuring noise.
    prior_gains = 1.6 + rng.normal(0.0, 0.5, size=N_MEMBERS)
    prior_phases = rng.normal(0.0, 1.2, size=N_MEMBERS)

    means = []
    for window in range(N_WINDOWS):
        global_times = window_times + window * SIM_TIME
        posterior = _velocity_arrays(global_times, gains, phases, rng)
        if with_blanking:
            posterior = _with_blanking(posterior)
        _state_dataset(posterior, window_times, GRID_X).to_netcdf(
            windows_dir / f"window_{window}_posterior_state.nc"
        )
        if with_prior:
            _state_dataset(
                _velocity_arrays(global_times, prior_gains, prior_phases, rng),
                window_times,
                GRID_X,
            ).to_netcdf(windows_dir / f"window_{window}_prior_state.nc")
        means.append(_state_dataset(posterior, global_times, GRID_X).mean("ensemble"))
    xarray.concat(means, dim="time").to_netcdf(run_dir / "posterior_state_mean.nc")

    obs_config = {
        "obs": {
            "mode": "points",
            "x_points": ASSIM_POINTS[0],
            "y_points": ASSIM_POINTS[1],
            "z_points": ASSIM_POINTS[2],
            "validation_x_points": VALIDATION_POINTS[0],
            "validation_y_points": VALIDATION_POINTS[1],
            "validation_z_points": VALIDATION_POINTS[2],
        }
    }
    truth_access = {
        "true_state_path": str(truth_path),
        "n_total": n_truth,
        "x_offset": TRUTH_X_OFFSET,
        "start_idx": TRUTH_SPINUP_FRAMES,
        "t_offset": float(TRUTH_SPINUP_FRAMES * TRUTH_DT),
        "num_windows": N_WINDOWS,
        "n_per_window": TRUTH_FRAMES_PER_WINDOW,
        "truth_solver_name": "pylbm",
        "assim_solver_name": "pylbm",
        "sim_time": SIM_TIME,
    }
    return obs_config, truth_access


def _full_run_dir(
    run_dir: pathlib.Path,
    *,
    level: str = "standard",
    with_prior: bool = False,
    with_blanking: bool = False,
    stations: list[list[float]] | None = None,
    stride: int = 1,
    seed: int = 11,
) -> None:
    """The WP1.1 parameter artifacts plus everything the WP1.2 layer reads."""
    obs_config, truth_access = _write_state_artifacts(
        run_dir,
        np.random.default_rng(seed),
        with_prior=with_prior,
        with_blanking=with_blanking,
    )
    _write_run_dir(
        run_dir,
        *_dynamic_run_artifacts(),
        level=level,
        skip_viz=False,
        extra_config=obs_config,
        truth_access=truth_access,
        stations=stations,
        stride=stride,
    )


def _iter_numbers(obj: object, path: str = "") -> list[tuple[str, float]]:
    """Every numeric leaf of a summary, with its key path (for the error message)."""
    if isinstance(obj, dict):
        return [n for k, v in obj.items() for n in _iter_numbers(v, f"{path}.{k}")]
    if isinstance(obj, (list, tuple)):
        return [n for i, v in enumerate(obj) for n in _iter_numbers(v, f"{path}[{i}]")]
    if isinstance(obj, bool) or obj is None or isinstance(obj, str):
        return []
    if isinstance(obj, (int, float)):
        return [(path, float(obj))]
    return []


def test_standard_level_wiring_emits_the_whole_bundle(tmp_path: pathlib.Path) -> None:
    """Every WP1.1 key is present and finite on a multi-window artifact at M = 8."""
    run_dir = tmp_path / "run"
    _write_run_dir(run_dir, *_dynamic_run_artifacts())

    compute_metrics(run_dir)
    summary = read_yaml(run_dir / "run_summary.yaml")

    assert summary["metrics_level"] == "standard"
    parameters = summary["parameter_metrics"]
    assert set(parameters) == {"inflow_angle", "vertical_inflow_exponent", "joint"}

    for name in ("inflow_angle", "vertical_inflow_exponent"):
        entry = parameters[name]
        # The key SET, not just presence: a renamed or dropped key is as much a
        # regression as a wrong number, and figure code indexes these by name.
        assert set(entry) == BASE_PARAM_KEYS | BUNDLE_PARAM_KEYS, name
        assert set(entry["zscore"]) == {
            "mean",
            "std",
            "max_abs",
            "max_abs_calibrated_median",
            "exceedance",
            "overconfident",
            "overconfident_rule",
        }
        # Every calibration number travels with the reference it must be read
        # against: the z-score null is a scaled t, not a normal, and an
        # order-statistic band cannot hit an arbitrary alpha.
        assert set(entry["zscore"]["exceedance"]) == {
            "n_samples",
            "df",
            "null_scale",
            "thresholds",
            "counts",
            "observed",
            "nominal",
            "nominal_normal",
        }
        assert entry["zscore"]["exceedance"]["df"] == N_MEMBERS - 1
        assert set(entry["coverage"]) == {
            "alpha_50",
            "nominal_alpha_50",
            "alpha_90",
            "nominal_alpha_90",
            "max_nominal_alpha",
        }
        # M = 8 -> bands are multiples of 1/9, so neither request is attainable
        # and comparing `alpha_50` against 0.5 would read discretization as
        # miscalibration.
        assert entry["coverage"]["nominal_alpha_50"] == pytest.approx(4.0 / 9.0)
        assert entry["coverage"]["nominal_alpha_90"] == pytest.approx(7.0 / 9.0)
        assert set(entry["sampling"]) == {
            "n_samples",
            "n_knots_effective",
            "pooling",
        }
        assert set(entry["contraction_ratio"]) == {
            "mean",
            "min",
            "vs_window_prior",
            "vs_initial_prior",
        }
        assert set(entry["pit"]) == {
            "n_bins",
            "n_samples",
            "n_knots_effective",
            "pooling",
            "tie_seed",
            "ranks_per_bin",
        }
        assert len(entry["pit_counts"]) == PIT_BINS
        assert sum(entry["pit_counts"]) == entry["pit"]["n_samples"]
        # M = 8: 9 rank values over 10 bins, so the last bin is unreachable --
        # a flat reference would read as a hole in a calibrated histogram.
        assert entry["pit"]["ranks_per_bin"] == [1] * 9 + [0]
        # The `sampling` block mirrors what `pit` carries -- one computation,
        # hoisted because zscore and coverage pool over the same knots.
        assert entry["sampling"] == {
            key: entry["pit"][key]
            for key in ("n_samples", "n_knots_effective", "pooling")
        }

        # Finding 3: the two contraction readings must NOT coincide. The prior
        # chains window to window, so the elementwise ratio measures only the
        # last update while the run as a whole contracted WINDOW_RATIO**w.
        contraction = entry["contraction_ratio"]
        assert contraction["vs_window_prior"]["mean"] == pytest.approx(WINDOW_RATIO)
        assert contraction["vs_window_prior"]["min"] == pytest.approx(WINDOW_RATIO)
        # Cumulative: window w sits at WINDOW_RATIO**(w + 1) against block 0.
        cumulative = [WINDOW_RATIO ** (w + 1) for w in range(N_WINDOWS)]
        assert contraction["vs_initial_prior"]["mean"] == pytest.approx(
            float(np.mean(cumulative))
        )
        assert contraction["vs_initial_prior"]["min"] == pytest.approx(
            WINDOW_RATIO**N_WINDOWS
        )
        assert contraction["vs_initial_prior"]["reason"] is None
        # A run that cut spread by 78% must not report 40%.
        assert contraction["vs_initial_prior"]["mean"] < contraction["mean"]
        # `mean`/`min` stay aliases of the per-window block (schema stability).
        assert contraction["mean"] == contraction["vs_window_prior"]["mean"]
        assert contraction["min"] == contraction["vs_window_prior"]["min"]

    # The `truth_access.yaml` key name is load-bearing for the static branch and
    # the GP config for the dynamic one; both must have resolved to a number.
    assert parameters["inflow_angle"]["pit"]["pooling"] == "knots_correlated"
    # ceil(15 knots * 30 s / 200 s) = 3, unclamped: every knot differs.
    assert parameters["inflow_angle"]["pit"]["n_knots_effective"] == 3
    # Broadcast static: clamped by its N_WINDOWS piecewise-constant segments.
    assert parameters["vertical_inflow_exponent"]["pit"]["n_knots_effective"] == 3

    joint = parameters["joint"]
    # K = 30 is over JOINT_CORR_MAX_K, which is the routine case: the matrices
    # are replaced by the note + `corr_summary` rather than 500 lines of YAML.
    # No `reason` key, though -- the block itself is not degraded.
    assert set(joint) == JOINT_KEYS | {"corr_matrices_omitted"}
    assert joint["posterior_corr"] is None and joint["prior_corr"] is None
    assert set(joint["corr_summary"]) == {
        "posterior_offdiag_abs_mean",
        "posterior_offdiag_abs_max",
        "prior_offdiag_abs_mean",
        "prior_offdiag_abs_max",
    }
    assert joint["n_members"] == N_MEMBERS
    # Both parameters made it into one vector -- i.e. the two flatteners agreed.
    assert joint["n_parameters"] == 2 * N_WINDOWS * KNOTS_PER_WINDOW
    assert joint["n_sample_directions"] == N_MEMBERS - 1
    assert joint["rank_deficient"] is True
    assert 0.0 < joint["posterior_variance_retained"] <= 1.0
    assert len(joint["generalized_eigenvalues"]) == N_MEMBERS - 1
    # The same per-window/cumulative split as `contraction_ratio`, said out loud
    # rather than left to the reader: the concatenated prior is a per-window
    # reference, so the parent block cannot answer "what did the run constrain".
    assert joint["prior_reference"] == "per_window_prior"

    cumulative_joint = joint["vs_initial_prior"]
    assert set(cumulative_joint) == {
        "n_members",
        "n_parameters",
        "n_sample_directions",
        "rank_deficient",
        "posterior_variance_retained",
        "n_constrained_directions",
        "eigenvalue_quantiles",
        "reason",
    }
    assert cumulative_joint["reason"] is None
    # One window wide: both parameters' knots for a single window.
    assert cumulative_joint["n_parameters"] == 2 * KNOTS_PER_WINDOW
    assert cumulative_joint["n_members"] == N_MEMBERS
    # Rank, and the reason for it, pins that the block was sliced along the
    # window axis and not somewhere else: one window's block holds
    # KNOTS_PER_WINDOW independent `inflow_angle` knots plus a SINGLE
    # `vertical_inflow_exponent` level (broadcast across that window's knots, so
    # its 5 columns are one direction). 5 + 1 = 6, below the M - 1 = 7 the
    # ensemble could otherwise support -- a mis-slice spanning two windows would
    # show 7.
    assert cumulative_joint["n_sample_directions"] == KNOTS_PER_WINDOW + 1
    # Three windows of contraction, so the cumulative spectrum sits below the
    # per-window one (~WINDOW_RATIO**6 against ~WINDOW_RATIO**2 in expectation).
    # The ordering is the regression; the exact spectrum is two independent
    # sample covariances at M = 8, which is noisy, so the count is only bounded.
    assert (
        cumulative_joint["eigenvalue_quantiles"]["median"]
        < joint["eigenvalue_quantiles"]["median"]
    )
    assert (
        1
        <= cumulative_joint["n_constrained_directions"]
        <= cumulative_joint["n_sample_directions"]
    )

    for path, value in _iter_numbers(parameters):
        assert np.isfinite(value), f"non-finite value at parameter_metrics{path}"


def test_static_run_wiring_reads_num_windows_from_truth_access(
    tmp_path: pathlib.Path,
) -> None:
    """A purely static run pools over windows, and the depth comes from disk.

    `_concat_windows` stacks per-window files along `time` WITHOUT a `time`
    coordinate here, so the bundle's static branch fires and the only source of
    the pooling depth is `truth_access.yaml`. This pins the key name
    `compute_metrics` reads it under.
    """
    rng = np.random.default_rng(8)
    coords = {"ensemble": np.arange(N_MEMBERS)}  # deliberately no `time` coord
    posterior = xarray.Dataset(
        {
            "sgs_constant": (
                ("ensemble", "time"),
                0.15 + rng.normal(0.0, 0.01, size=(N_MEMBERS, N_WINDOWS)),
            )
        },
        coords=coords,
    )
    prior = xarray.Dataset(
        {
            "sgs_constant": (
                ("ensemble", "time"),
                0.15 + rng.normal(0.0, 0.05, size=(N_MEMBERS, N_WINDOWS)),
            )
        },
        coords=coords,
    )
    truth = xarray.Dataset({"sgs_constant": 0.16})

    run_dir = tmp_path / "run"
    _write_run_dir(run_dir, posterior, prior, truth)
    compute_metrics(run_dir)

    pit = read_yaml(run_dir / "run_summary.yaml")["parameter_metrics"]["sgs_constant"][
        "pit"
    ]
    assert pit["pooling"] == "windows_correlated"
    assert pit["n_knots_effective"] == N_WINDOWS


def test_missing_num_windows_degrades_to_null_rather_than_raising(
    tmp_path: pathlib.Path,
) -> None:
    """An old run dir whose truth_access.yaml predates the key still processes."""
    rng = np.random.default_rng(9)
    posterior = xarray.Dataset(
        {"sgs_constant": (("ensemble", "time"), rng.normal(size=(N_MEMBERS, 3)))},
        coords={"ensemble": np.arange(N_MEMBERS)},
    )
    prior = xarray.Dataset(
        {"sgs_constant": (("ensemble", "time"), rng.normal(size=(N_MEMBERS, 3)))},
        coords={"ensemble": np.arange(N_MEMBERS)},
    )
    truth = xarray.Dataset({"sgs_constant": 0.0})

    run_dir = tmp_path / "run"
    _write_run_dir(run_dir, posterior, prior, truth, num_windows=None)
    compute_metrics(run_dir)

    pit = read_yaml(run_dir / "run_summary.yaml")["parameter_metrics"]["sgs_constant"][
        "pit"
    ]
    assert pit["n_knots_effective"] is None
    assert pit["pooling"] == "windows_correlated"


@pytest.mark.parametrize("level", ["basic", "standard"])  # type: ignore[misc]
def test_metrics_level_is_recorded_in_the_summary(
    tmp_path: pathlib.Path, level: str
) -> None:
    """Without this key an absent `joint` is ambiguous three ways."""
    run_dir = tmp_path / f"run_{level}"
    _write_run_dir(run_dir, *_dynamic_run_artifacts(), level=level)

    compute_metrics(run_dir)
    summary = read_yaml(run_dir / "run_summary.yaml")

    assert summary["metrics_level"] == level
    assert summary["metrics_version"] == 2  # unchanged: same estimator semantics
    # `basic` is still exactly the pre-phase-1 metric set.
    assert ("joint" in summary["parameter_metrics"]) is (level == "standard")
    entry = summary["parameter_metrics"]["inflow_angle"]
    assert (BUNDLE_PARAM_KEYS <= set(entry)) is (level == "standard")


def test_standard_level_wiring_emits_the_sensor_statistics_for_every_set(
    tmp_path: pathlib.Path,
) -> None:
    """WP1.2 is wired past `skip_viz`, on both sensor sets, with a real schema.

    The unit tests build the `{set_name: DataArray}` mappings by hand, so what
    is only checkable here is that `compute_metrics` hands
    `sensor_statistic_scores` the series it actually extracted and the windowing
    keys spelled the way the run stage writes them. A swapped `n_per_window` /
    `sim_time`, or a truth-access key read under the wrong name, produces a
    plausible-looking summary rather than an error -- which is why the shape
    numbers below are asserted against the fixture's construction and not merely
    for finiteness.
    """
    run_dir = tmp_path / "run"
    _full_run_dir(run_dir)

    compute_metrics(run_dir)
    summary = read_yaml(run_dir / "run_summary.yaml")

    statistics = summary["sensor_statistics"]
    # Both sets, i.e. the held-out branch of `build_sensor_sets` fired. It is
    # only present when the obs config defines `validation_*_points`, and the
    # WP1.1 fixture has no obs config at all.
    assert set(statistics) == {"assimilation", "validation"}
    assert set(statistics) == set(summary["sensor_metrics"])

    for name, points in (
        ("assimilation", ASSIM_POINTS),
        ("validation", VALIDATION_POINTS),
    ):
        entry = statistics[name]
        assert set(entry) == SENSOR_SET_KEYS, name
        assert entry["n_members"] == N_MEMBERS
        assert entry["n_windows"] == N_WINDOWS
        assert entry["n_sensors"] == len(points[0])

        for statistic in SENSOR_STATISTIC_NAMES:
            block = entry[statistic]
            assert set(block) == STATISTIC_KEYS, (name, statistic)
            # Nothing degraded: M = 8 is above MIN_MEMBERS_CALIBRATION and every
            # window has both ensemble and truth frames.
            assert block["reason"] is None, (name, statistic)
            assert set(block["zscore"]) == STATISTIC_ZSCORE_KEYS
            assert set(block["coverage"]) == STATISTIC_COVERAGE_KEYS
            assert set(block["pit"]) == STATISTIC_PIT_KEYS
            assert set(block["identifiability"]) == IDENTIFIABILITY_KEYS

            # M = 8 -> order-statistic bands are multiples of 1/9, so neither
            # requested level is attainable and comparing `alpha_50` against 0.5
            # would read discretization as miscalibration.
            assert block["coverage"]["nominal_alpha_50"] == pytest.approx(4.0 / 9.0)
            assert block["coverage"]["nominal_alpha_90"] == pytest.approx(7.0 / 9.0)
            # 9 rank values over 10 bins: the last bin is unreachable, so a flat
            # reference would read as a hole in a calibrated histogram.
            assert block["pit"]["ranks_per_bin"] == [1] * 9 + [0]
            assert len(block["pit_counts"]) == PIT_BINS
            assert sum(block["pit_counts"]) == block["pit"]["n_samples"]
            assert block["pit"]["n_samples"] == block["n_samples"]
            # Nothing dropped: this fixture gives every window both ensemble
            # and truth frames, and 8 frames per window is enough for every
            # statistic including the ddof=1 ones. The assertion is here so a
            # future silent drop -- the failure mode `n_windows_scored` was
            # added for, where the summary still says `n_windows: 3` and
            # `reason: null` while only `n_samples` shrinks -- fails loudly
            # rather than moving `n_samples` to another plausible number.
            assert block["n_windows_scored"] == N_WINDOWS, (name, statistic)

        # The pooling axes, which are the one thing a wrong reduction changes
        # silently: the per-component statistics pool component x sensor x
        # window, the two magnitude-like ones only sensor x window.
        n_sensors, n_components = len(points[0]), 3
        for statistic in ("window_mean", "window_variance"):
            assert (
                entry[statistic]["n_samples"] == n_components * n_sensors * N_WINDOWS
            ), statistic
        for statistic in ("tke", "velmag_mean"):
            assert entry[statistic]["n_samples"] == n_sensors * N_WINDOWS, statistic

        wasserstein = entry["wasserstein"]
        assert set(wasserstein) == WASSERSTEIN_KEYS
        assert wasserstein["reason"] is None
        for key in (
            "w1_over_sigma_pooled",
            "self_floor",
            "w1_over_floor",
            "w1_over_floor_calibrated_median",
        ):
            assert set(wasserstein[key]) == {"median", "max"}
        # `w1_over_floor` is not calibrated at 1 -- a perfect model of this
        # series scores ~0.55 at the shipped shape and a real error grows as
        # sqrt(n) -- so the ratio is only readable against this reference.
        # Asserted present *and* finite here (8 frames per window clears
        # `wasserstein_self_floor`'s four-sample minimum) so a reference that
        # stops being emitted, or starts coming out nan on a case that otherwise
        # scores fine, fails loudly instead of quietly removing the only thing
        # that makes the headline ratio interpretable.
        for reduction in ("median", "max"):
            reference = wasserstein["w1_over_floor_calibrated_median"][reduction]
            assert reference is not None, reduction
            assert np.isfinite(reference) and reference > 0.0, reduction
        # W1 is convex in its first argument, so pooling the members can only
        # help. This holds for any ensemble, which is exactly why it is a wiring
        # assertion: it fails if the two are computed off different samples.
        assert (
            wasserstein["w1_over_sigma_pooled"]["median"]
            <= wasserstein["w1_over_sigma_member_mean"]["median"] + 1e-9
        )

    # The identifiability bootstrap really ran: `run.metrics.bootstrap_blocks`
    # reached `block_bootstrap_std` (4 blocks over 8 frames -> L = 2). At the
    # shipped default of 20 it would be null everywhere and this layer's headline
    # caveat would be untested.
    identifiability = statistics["assimilation"]["window_mean"]["identifiability"]
    assert identifiability["threshold"] == pytest.approx(3.0)
    assert identifiability["ratio_median"] is not None
    assert identifiability["ratio_min"] is not None
    assert isinstance(identifiability["sampling_noise_dominated"], bool)

    # `write_yaml` cannot emit nan/inf, so a non-finite leaf here means a number
    # bypassed `_finite_or_none` on its way out.
    for path, value in _iter_numbers(statistics):
        assert np.isfinite(value), f"non-finite value at sensor_statistics{path}"


def test_absent_prior_state_nulls_crpss_rather_than_raising(
    tmp_path: pathlib.Path,
) -> None:
    """The common case: `run.save_prior_state` is false, so there is no prior.

    `ensemble_sensor_series` opens every path it is given unconditionally and
    has no guard of its own, so without the existence check in
    `_prior_sensor_series` this run raises `FileNotFoundError` -- on the default
    configuration, i.e. on nearly every run. The degradation is a `null`
    `crpss_vs_prior` and nothing else: the rest of the block must still score.
    """
    run_dir = tmp_path / "run"
    _full_run_dir(run_dir, with_prior=False)
    assert not list((run_dir / "windows").glob("*_prior_state.nc"))

    compute_metrics(run_dir)
    statistics = read_yaml(run_dir / "run_summary.yaml")["sensor_statistics"]

    for entry in statistics.values():
        for statistic in SENSOR_STATISTIC_NAMES:
            block = entry[statistic]
            assert block["crpss_vs_prior"] is None, statistic
            # A missing prior is not a degraded block -- `reason` stays null and
            # the CRPS next to it is a real number.
            assert block["reason"] is None
            assert block["crps"] is not None


def test_saved_prior_state_is_discovered_and_scored_against(
    tmp_path: pathlib.Path,
) -> None:
    """With every `window_{w}_prior_state.nc` present, `crpss_vs_prior` is a number.

    The negative branch above cannot distinguish "guard works" from "the call
    site never builds the paths at all", so the positive branch has to exist
    too. The prior written here is biased *and* over-wide, so the skill is
    positive for a structural reason rather than by sampling luck.
    """
    run_dir = tmp_path / "run"
    _full_run_dir(run_dir, with_prior=True)
    assert len(list((run_dir / "windows").glob("*_prior_state.nc"))) == N_WINDOWS

    compute_metrics(run_dir)
    statistics = read_yaml(run_dir / "run_summary.yaml")["sensor_statistics"]

    for name, entry in statistics.items():
        for statistic in SENSOR_STATISTIC_NAMES:
            skill = entry[statistic]["crpss_vs_prior"]
            assert skill is not None, (name, statistic)
            # CRPSS is 1 - post/prior on non-negative scores, so it is bounded
            # above by 1 whatever the prior is.
            assert skill <= 1.0
        # The two statistics the injected bias hits head-on. `window_variance`
        # and `tke` respond to the width, which the fixture also inflates, but
        # their sampling error at 8 frames per window is large enough that
        # asserting a sign there would be asserting on noise.
        for statistic in ("window_mean", "velmag_mean"):
            assert entry[statistic]["crpss_vs_prior"] > 0.0, (name, statistic)


def test_basic_level_omits_sensor_statistics_entirely(tmp_path: pathlib.Path) -> None:
    """`basic` is the pre-phase-1 key set even when the truth IS readable.

    `skip_viz` is false here, so `sensor_metrics` and `state_metrics` are both
    computed -- which is what makes this a test of the level gate rather than of
    the early return. A gate written as "compute it, then drop it" would pass a
    key-presence check while still paying for the bootstrap.

    The top-level key SET is asserted, not merely the absence of the two phase-1
    blocks: `run_summary.yaml` key paths are hard-coded across
    `scripts/figure_creation/`, so "additive only" means `basic` has to stay
    byte-for-byte the same shape a pre-phase-1 consumer indexes.
    """
    run_dir = tmp_path / "run"
    _full_run_dir(run_dir, level="basic")

    compute_metrics(run_dir)
    summary = read_yaml(run_dir / "run_summary.yaml")

    assert summary["metrics_level"] == "basic"
    assert set(summary) == BASIC_SUMMARY_KEYS
    assert "sensor_statistics" not in summary
    assert "mean_field_metrics" not in summary
    assert "sensor_metrics" in summary and "state_metrics" in summary
    assert summary["metrics_version"] == 2  # unchanged: WP1.3 is additive
    # The layer's sidecar is not written either -- a gate that computed the
    # fields and then dropped the summary block would leave this behind.
    assert not (run_dir / "eval_fields.nc").exists()


def test_standard_level_wiring_emits_the_mean_field_layer(
    tmp_path: pathlib.Path,
) -> None:
    """WP1.3's schema, its `eval_fields.nc` sidecar and its bookkeeping.

    What is only checkable here (and not in `tests/test_turbulence_stats.py`,
    which covers the estimators on synthetic arrays) is the *wiring*: that the
    mean-field accumulators are driven off the same `stream_window_members`
    pass as the sensor extraction, that the truth-access keys are read under
    the names the run stage writes, and that the averaging bookkeeping reports
    the fixture's actual shape rather than a plausible-looking one. A swapped
    `n_total` / `n_per_window` would produce a summary that looks fine.

    The fixture's cadence is what makes the sampling floors real here: 18 truth
    frames against `bootstrap_blocks = 4` gives a block length of 5, so the
    hit-rate allowance and the two RMSE floors are numbers rather than the
    smoke shape's nulls (3 frames against the shipped 20 blocks).
    """
    run_dir = tmp_path / "run"
    _full_run_dir(run_dir)

    compute_metrics(run_dir)
    summary = read_yaml(run_dir / "run_summary.yaml")

    assert set(summary) == STANDARD_SUMMARY_KEYS
    block = summary["mean_field_metrics"]
    assert set(block) == MEAN_FIELD_KEYS
    assert block["reason"] is None

    hit = block["hit_rate"]
    assert set(hit) == HIT_RATE_KEYS
    assert hit["relative_tolerance"] == pytest.approx(0.25)
    # One entry per selected z-level, each naming the level it scored.
    assert len(hit["per_z"]) == GRID_Z.size
    assert [entry["z"] for entry in hit["per_z"]] == [pytest.approx(v) for v in GRID_Z]
    for name in ("u", "v", "w"):
        assert 0.0 <= hit[name] <= 1.0, name
        # A real floor, not the `nan` degradation: `hit_rate` documents `nan`
        # as "no allowance available", and on this cadence one IS available.
        assert hit["allowance"][name] > 0.0, name
    assert hit["allowance"]["reason"] is None
    assert 0.0 <= block["fac2_velmag"] <= 1.0

    for name in ("u", "v", "w", "velmag"):
        assert set(block["nmse"][name]) == NMSE_KEYS, name
        assert name in block["fb"], name
    # The split is an identity when both parts exist, and `velmag` is the one
    # component where NMSE is always well posed (positive by construction), so
    # it is the entry that must never be null.
    velmag = block["nmse"]["velmag"]
    assert velmag["total"] is not None
    assert velmag["systematic"] <= velmag["total"] + 1e-12
    assert velmag["systematic"] + velmag["unsystematic"] == pytest.approx(
        velmag["total"]
    )

    for key in ("tke_rmse", "uw_stress_rmse"):
        assert set(block[key]) == {"value", "sampling_floor"}
        assert block[key]["value"] >= 0.0, key
        # The floor is what makes the RMSE readable: an error below the truth's
        # own sampling spread is the window length, not skill.
        assert block[key]["sampling_floor"] > 0.0, key

    averaging = block["averaging"]
    assert set(averaging) == AVERAGING_KEYS
    assert averaging["n_members"] == N_MEMBERS
    assert averaging["n_windows"] == N_WINDOWS
    # Accumulated across ALL windows, per member -- not one window's worth.
    assert averaging["n_time"] == N_WINDOWS * ENSEMBLE_FRAMES_PER_WINDOW
    assert averaging["n_time_truth"] == N_WINDOWS * TRUTH_FRAMES_PER_WINDOW
    assert averaging["z_levels"] == [pytest.approx(v) for v in GRID_Z]
    # Truth and assimilation grids coincide here, so the nearest-level pairing
    # is exact; a non-zero offset on this fixture would mean a mis-selection.
    assert averaging["z_offset_max"] == pytest.approx(0.0)
    assert averaging["n_cells"] == GRID_Z.size * GRID_Y.size * GRID_X.size
    assert averaging["n_scored_cells"] == averaging["n_cells"]
    # Default stations are the sensor x/y of BOTH sets, deduplicated.
    assert averaging["n_stations"] == len(ASSIM_POINTS[0]) + len(VALIDATION_POINTS[0])

    # This fixture is a pylbm grid with no `blanking`, i.e. the unmasked branch:
    # the scores cover building interiors and the block has to say so rather
    # than quietly reporting a fluid fraction of 1 as if it were meaningful.
    masking = block["masking"]
    assert set(masking) == MASKING_KEYS
    assert masking["source"] == "none"
    assert masking["ensemble_fluid_fraction"] is None
    assert masking["note"] is not None

    for path, value in _iter_numbers(block):
        assert np.isfinite(value), f"non-finite value at mean_field_metrics{path}"


def test_standard_level_writes_eval_fields_with_the_expected_variables(
    tmp_path: pathlib.Path,
) -> None:
    """`eval_fields.nc` is the sanctioned handoff to the figure stage.

    Stage 3 re-derives its inputs from the raw artifacts today; every field
    here costs a streaming pass over the window state files, so the variable
    NAMES are as much a contract as the summary's key paths -- WP1.4's F1/F2/S1
    index them directly.
    """
    run_dir = tmp_path / "run"
    _full_run_dir(run_dir)

    compute_metrics(run_dir)

    path = run_dir / "eval_fields.nc"
    assert path.exists()
    assert (
        read_yaml(run_dir / "run_summary.yaml")["mean_field_metrics"]["eval_fields"]
        == path.name
    )
    with xarray.open_dataset(path) as fields:
        assert set(fields.data_vars) == EVAL_FIELD_VARS
        # Two vertical grids in one file, and the distinction is the point:
        # `zlev` is the scored planes, `z` is the FULL column carried only at
        # the stations, because four planes are not a vertical profile.
        assert fields["ensemble_mean"].dims == ("component", "zlev", "y", "x")
        assert fields["station_ensemble_mean"].dims == ("component", "z", "station")
        assert fields.sizes["z"] == GRID_Z.size
        assert fields.sizes["station"] == len(ASSIM_POINTS[0]) + len(
            VALIDATION_POINTS[0]
        )
        assert list(fields["component"].values) == ["u", "v", "w"]
        # M = 8, so the across-member spread exists everywhere.
        assert bool(np.all(np.isfinite(fields["ensemble_std"].values)))
        assert bool(np.all(fields["ensemble_std"].values >= 0.0))
        assert fields.attrs["n_members"] == N_MEMBERS
        assert fields.attrs["mask_source"] == "none"


def test_blanking_supplies_the_fluid_mask_on_both_sides(
    tmp_path: pathlib.Path,
) -> None:
    """The one mask that exists is used, on the truth and on the ensemble.

    The plan's premise -- "solid cells are the NaN cells" -- does not hold for
    any shipped backend (measured NaN fraction 0.0 in every `.temp` artifact),
    so `blanking` is the whole of the mask story and this is the branch that
    must not silently stop firing: an unmasked FAC2 counts building interiors,
    where both fields are near zero and every ratio test passes, so losing the
    mask makes a run look BETTER. The fluid fraction is asserted in closed form
    against the synthetic building rather than measured.
    """
    run_dir = tmp_path / "run"
    _full_run_dir(run_dir, with_blanking=True)

    compute_metrics(run_dir)
    block = read_yaml(run_dir / "run_summary.yaml")["mean_field_metrics"]

    masking = block["masking"]
    assert masking["source"] == "blanking"
    assert masking["truth_source"] == "blanking"
    assert masking["ensemble_source"] == "blanking"
    # Truth and assimilation grids coincide on this fixture, so no interpolation
    # erodes the mask and all three fractions are the building's complement
    # exactly. On a cross-grid run the truth-side number is strictly smaller --
    # interpolating a NaN-masked truth drops every cell whose stencil touched a
    # building, which is the conservative combination rule.
    assert masking["fluid_fraction"] == pytest.approx(BLANK_FLUID_FRACTION)
    assert masking["ensemble_fluid_fraction"] == pytest.approx(BLANK_FLUID_FRACTION)
    assert masking["truth_finite_fraction"] == pytest.approx(BLANK_FLUID_FRACTION)
    assert masking["note"] is None
    # Nothing beyond the mask went non-finite on a healthy run, so the mask's
    # fraction and the scored sample's coincide here. They are separate keys
    # because they are separate samples, not because they usually differ.
    assert masking["scored_fraction"] == pytest.approx(masking["fluid_fraction"])

    averaging = block["averaging"]
    assert averaging["n_scored_cells"] == round(
        BLANK_FLUID_FRACTION * averaging["n_cells"]
    )
    assert masking["scored_fraction"] == pytest.approx(
        averaging["n_scored_cells"] / averaging["n_cells"]
    )

    with xarray.open_dataset(run_dir / "eval_fields.nc") as fields:
        mask = fields["fluid_mask"].values
        assert mask.dtype == np.int8
        assert set(np.unique(mask)) == {0, 1}
        assert float(mask.mean()) == pytest.approx(BLANK_FLUID_FRACTION)
        # WP1.4 greys buildings with `set_bad`, so the mask has to line up with
        # the NaN the truth fields already carry there.
        assert not np.any(np.isfinite(fields["truth_velmag"].values[mask == 0]))
        assert fields["fluid_mask"].attrs["source"] == "blanking"
        # A station inside the building (25, 20) is blanked out, so the column
        # mask is exercised rather than trivially all-ones.
        assert int(fields["station_fluid_mask"].values.min()) == 0


def test_configured_stations_replace_the_default_sensor_columns(
    tmp_path: pathlib.Path,
) -> None:
    """`run.metrics.stations` is a real knob, not a documented one.

    Nothing else in the pipeline reads it, so without this test the config
    block could ship, validate, and be silently ignored. Configured stations
    win outright rather than being added to the sensor positions: they are the
    case's own reference locations (wind-tunnel probes), which need not
    coincide with the assimilated sensors at all.
    """
    run_dir = tmp_path / "run"
    stations = [[10.0, 10.0], [30.0, 30.0]]
    _full_run_dir(run_dir, stations=stations)

    compute_metrics(run_dir)
    block = read_yaml(run_dir / "run_summary.yaml")["mean_field_metrics"]

    assert block["averaging"]["n_stations"] == len(stations)
    with xarray.open_dataset(run_dir / "eval_fields.nc") as fields:
        assert fields.sizes["station"] == len(stations)
        assert list(fields["station_x"].values) == [s[0] for s in stations]
        assert list(fields["station_y"].values) == [s[1] for s in stations]


def _staggered_member(rng: np.random.Generator, gain: float) -> xarray.Dataset:
    """One uDALES-staggered member: ``u`` on ``xm``, ``v`` on ``ym``, ``w`` on ``zm``.

    The whole fixture above is pylbm, which writes one unstaggered grid — so it
    can never exercise colocation's edge extrapolation, and the aggregate
    exclusion that exists because of it would never run. This is the smallest
    state that does: ``w`` lives on ``zm``, so the top ``zt`` level is filled by
    extrapolation from the last two faces and is not a normal cell.
    """
    zt, yt, xt = np.arange(4) + 0.5, np.arange(5) + 0.5, np.arange(6) + 0.5
    zm, ym, xm = zt - 0.5, yt - 0.5, xt - 0.5
    times = np.arange(6.0)

    def field(z: np.ndarray, y: np.ndarray, x: np.ndarray, scale: float) -> np.ndarray:
        base = (
            2.0 * x[None, None, None, :]
            + 3.0 * y[None, None, :, None]
            + 5.0 * z[None, :, None, None]
        )
        wave = np.sin(times)[:, None, None, None]
        noise = rng.normal(0.0, 0.05, size=(times.size, z.size, y.size, x.size))
        return np.asarray(scale * gain * (base + wave) + noise, dtype=float)

    return xarray.Dataset(
        {
            "u": (("time", "zt", "yt", "xm"), field(zt, yt, xm, 1.0)),
            "v": (("time", "zt", "ym", "xt"), field(zt, ym, xt, 0.5)),
            "w": (("time", "zm", "yt", "xt"), field(zm, yt, xt, 0.2)),
        },
        coords={
            "zt": zt,
            "yt": yt,
            "xt": xt,
            "zm": zm,
            "ym": ym,
            "xm": xm,
            "time": times,
        },
    )


def _truth_matching(ensemble: dict, *, top_level_error: float) -> dict:
    """A truth mapping that matches the ensemble exactly except on the TOP level.

    Built from the ensemble's own reduced fields so "matches" is exact rather
    than approximate, then poisoned on the extrapolated level alone. That makes
    every aggregate score a clean statement about whether that level was
    excluded: with the exclusion the layer sees a perfect run, without it the
    contamination is the only thing it sees.
    """
    slab = ensemble["slab"]
    mean = np.array(slab["mean"], dtype=float)
    mean[:, -1] += top_level_error
    tke = np.array(slab["tke"], dtype=float)
    # +10 on one of four levels is an RMSE of 0 when that level is excluded and
    # exactly sqrt(100/4) = 5 when it is not.
    tke[-1] += 10.0
    station = ensemble["station"]
    return {
        "n_time": 6,
        "y": ensemble["y"],
        "x": ensemble["x"],
        "z_levels": ensemble["z_levels"],
        "z_all": ensemble["z_all"],
        "mean": mean,
        "velmag": np.sqrt((mean**2).sum(axis=0)),
        "tke": tke,
        "uw": np.array(slab["uw"], dtype=float),
        "station_mean": np.array(station["mean"], dtype=float),
        "station_rms": np.array(station["rms"], dtype=float),
        "station_tke": np.array(station["tke"], dtype=float),
        "station_uw": np.array(station["uw"], dtype=float),
        "fluid_slab": None,
        "mask_source": "none",
        "allowance": dict.fromkeys(("u", "v", "w"), 0.01),
        "tke_floor": 0.001,
        "uw_floor": 0.001,
        "n_bootstrap_cells": 120,
        "n_bootstrap_cells_max": 4096,
        "bootstrap_seed": 0,
    }


def test_the_extrapolated_top_level_is_reported_per_z_and_kept_out_of_aggregates(
    tmp_path: pathlib.Path,
) -> None:
    """A7: the exclusion has to actually fire, on a genuinely staggered state.

    `evenly_spaced_levels` uses `np.linspace(0, nz-1, k)`, which ALWAYS includes
    `nz-1` — and for uDALES/PALM `w` that is exactly the level colocation had to
    extrapolate, whose second moments are inflated by up to 5x. So the edge is
    not an unlikely corner: it is 1 of the 4 shipped z-levels on every staggered
    backend, and it silently biased `hit_rate`, `tke_rmse` and `uw_stress_rmse`
    until it was excluded.

    The contrast is the assertion. The truth here matches the ensemble exactly
    on the three interior levels and is wrong only on the extrapolated one, so:
    with the exclusion the aggregates say "perfect" (`hit_rate.u = 1.0`,
    `tke_rmse = 0`) while `per_z` still shows the bad level; without it they say
    `0.75` and `5`. Nothing is hidden and nothing is silently biased.
    """
    rng = np.random.default_rng(5)
    accumulator = MeanFieldAccumulator(
        "udales", 4, 1, np.array([2.5, 4.0]), np.array([2.0, 3.0])
    )
    for member in range(3):
        accumulator.add_member(0, member, _staggered_member(rng, 1.0 + 0.05 * member))
    ensemble = accumulator.result()

    # The wiring half: the flag comes off the real staggering, not a fixture
    # constant, and the level selection really does land on the edge.
    assert ensemble["z_edge_extrapolated"] is True
    assert list(ensemble["z_index"]) == [0, 1, 2, 3]

    truth = _truth_matching(ensemble, top_level_error=100.0)
    block, fields = mean_field_summary(ensemble, truth, 1)

    top = float(ensemble["z_levels"][-1])
    assert block["averaging"]["z_levels_extrapolated"] == [pytest.approx(top)]
    # Reported per z -- the bad level is visible, flagged, and not dropped.
    per_z = block["hit_rate"]["per_z"]
    assert [entry["u"] for entry in per_z] == [1.0, 1.0, 1.0, 0.0]
    assert [entry["extrapolated"] for entry in per_z] == [False, False, False, True]
    # ... and kept out of every aggregate.
    assert block["hit_rate"]["u"] == pytest.approx(1.0)
    assert block["fac2_velmag"] == pytest.approx(1.0)
    assert block["tke_rmse"]["value"] == pytest.approx(0.0)
    # The aggregate sample is three levels of the four the layer scored.
    averaging = block["averaging"]
    assert averaging["n_scored_cells"] == 4 * averaging["n_aggregate_cells"] / 3
    assert fields.sizes["zlev"] == 4  # the sidecar still carries every level

    # The same inputs with the exclusion off: this is what the layer reported
    # before A7, and the numbers the reviewer measured by hand.
    unexcluded = dict(ensemble, z_edge_extrapolated=False)
    before, _ = mean_field_summary(
        unexcluded, _truth_matching(ensemble, top_level_error=100.0), 1
    )
    assert before["averaging"]["z_levels_extrapolated"] == []
    assert before["hit_rate"]["u"] == pytest.approx(0.75)
    assert before["tke_rmse"]["value"] == pytest.approx(5.0)
    assert before["averaging"]["n_aggregate_cells"] == averaging["n_scored_cells"]


def test_a_single_extrapolated_level_is_scored_rather_than_nulled(
    tmp_path: pathlib.Path,
) -> None:
    """N2: when the exclusion would empty the aggregate, say so and score anyway.

    At `nz == 1` the only scored level IS `nz - 1`, so excluding it leaves the
    aggregates with no cells at all — `null` for every score, with no `reason`
    and no log line, which is exactly the signature B1 was blocked for. A
    contaminated number that announces its contamination beats a silent null,
    so every level is kept and `reason` carries why.
    """
    zt, yt, xt = np.array([0.5]), np.arange(3) + 0.5, np.arange(3) + 0.5
    zm = np.array([0.0, 1.0])
    times = np.arange(6.0)
    rng = np.random.default_rng(7)

    def one_level(z: np.ndarray, y: np.ndarray, x: np.ndarray) -> np.ndarray:
        shape = (times.size, z.size, y.size, x.size)
        wave = np.sin(times)[:, None, None, None]
        return np.asarray(5.0 + wave + rng.normal(0, 0.1, shape), dtype=float)

    state = xarray.Dataset(
        {
            "u": (("time", "zt", "yt", "xm"), one_level(zt, yt, xt - 0.5)),
            "v": (("time", "zt", "ym", "xt"), one_level(zt, yt - 0.5, xt)),
            "w": (("time", "zm", "yt", "xt"), one_level(zm, yt, xt)),
        },
        coords={
            "zt": zt,
            "yt": yt,
            "xt": xt,
            "zm": zm,
            "ym": yt - 0.5,
            "xm": xt - 0.5,
            "time": times,
        },
    )
    accumulator = MeanFieldAccumulator("udales", 4, 1, np.array([1.5]), np.array([1.5]))
    for member in range(2):
        accumulator.add_member(0, member, state * (1.0 + 0.01 * member))
    ensemble = accumulator.result()
    assert ensemble["z_edge_extrapolated"] is True
    assert list(ensemble["z_index"]) == [0]  # index 0 and nz-1 are the same level

    block, _ = mean_field_summary(
        ensemble, _truth_matching(ensemble, top_level_error=0.0), 1
    )

    # Still reported as extrapolated -- that is a fact about the grid...
    assert len(block["averaging"]["z_levels_extrapolated"]) == 1
    # ... but scored rather than nulled, with the aggregate sample intact.
    assert block["hit_rate"]["u"] is not None
    assert block["averaging"]["n_aggregate_cells"] == (
        block["averaging"]["n_scored_cells"]
    )
    assert block["averaging"]["n_aggregate_cells"] > 0
    # And the summary says why, per the degenerate-shape rule.
    assert block["reason"] is not None
    assert "extrapolated" in block["reason"]


def _diverge_member(
    run_dir: pathlib.Path,
    member: int,
    *,
    region: tuple[slice, slice] | None = None,
) -> None:
    """NaN out one member's velocities in every window, wholly or in a region.

    What a diverged CFD member looks like on disk. ``region`` is a ``(y, x)``
    slice pair: a member that blew up over part of the domain and stayed finite
    elsewhere, which is the harder of the two cases to notice, because the
    ensemble mean then goes NaN over exactly those cells while every count in
    the summary keeps describing the whole slab.
    """
    for path in sorted((run_dir / "windows").glob("window_*_posterior_state.nc")):
        with xarray.open_dataset(path) as opened:
            state = opened.load()
        for name in ("u", "v", "w"):
            if region is None:
                state[name][{"ensemble": member}] = np.nan
            else:
                ys, xs = region
                state[name][{"ensemble": member, "y": ys, "x": xs}] = np.nan
        state.to_netcdf(path.with_suffix(".rewritten.nc"))
        path.unlink()
        path.with_suffix(".rewritten.nc").rename(path)


def test_a_diverged_member_is_excluded_instead_of_nulling_the_layer(
    tmp_path: pathlib.Path,
) -> None:
    """One NaN member must not take the whole mean-field block down with it.

    A diverged member is routine, and `np.stack(...).mean(axis=0)` propagates
    its NaN to every cell: before this was fixed the layer reported `null` for
    every score with **no** `reason` and no log line, while `averaging` happily
    said `n_members: 8` and `n_scored_cells: 104` for a comparison that scored
    nothing. Both halves are pinned here -- the scores survive on the members
    that are fine, and the summary says which ensemble they came from.
    """
    run_dir = tmp_path / "run"
    _full_run_dir(run_dir, with_blanking=True)
    _diverge_member(run_dir, 0)

    compute_metrics(run_dir)
    block = read_yaml(run_dir / "run_summary.yaml")["mean_field_metrics"]

    averaging = block["averaging"]
    assert averaging["n_members"] == N_MEMBERS
    assert averaging["n_members_scored"] == N_MEMBERS - 1
    # Not silent: the block says what was dropped, and names it.
    assert block["reason"] is not None
    assert "0" in block["reason"]
    # And the layer still reports real numbers for the 7 members that are fine.
    for name in ("u", "v", "w"):
        assert block["hit_rate"][name] is not None, name
    assert block["fac2_velmag"] is not None
    assert block["tke_rmse"]["value"] is not None

    with xarray.open_dataset(run_dir / "eval_fields.nc") as fields:
        # The band a figure draws is the SCORED ensemble's, so the attribute a
        # caption would read has to be too.
        assert fields.attrs["n_members_scored"] == N_MEMBERS - 1
        assert fields.attrs["n_members"] == N_MEMBERS
        assert bool(np.all(np.isfinite(fields["ensemble_velmag"].values)))


def test_a_partially_diverged_member_is_excluded_and_the_counts_follow(
    tmp_path: pathlib.Path,
) -> None:
    """The sub-region case: the reported sample must be the scored sample.

    A member that goes NaN over part of the domain silently removes those cells
    from every score. Excluding the member wholesale is the same casewise rule
    `StreamingMoments` applies in time -- an ensemble mean and an ensemble
    spread taken over different member sets at neighbouring cells are not a
    field -- and `n_scored_cells` is derived from the pairs that survived, so it
    cannot drift away from what was scored whichever way the NaN falls.
    """
    run_dir = tmp_path / "run"
    _full_run_dir(run_dir, with_blanking=True)
    _diverge_member(run_dir, 3, region=(slice(0, 2), slice(0, 2)))

    compute_metrics(run_dir)
    block = read_yaml(run_dir / "run_summary.yaml")["mean_field_metrics"]

    averaging = block["averaging"]
    assert averaging["n_members_scored"] == N_MEMBERS - 1
    assert "3" in block["reason"]
    # The partially-NaN member is gone, so the scored set is the mask's again
    # -- the count and `scored_fraction` agree with each other by construction,
    # not by coincidence, and `scored_fraction` is the key that describes the
    # SCORES (`fluid_fraction` describes the mask, and the two are separate
    # names precisely so neither can quietly stand in for the other).
    assert averaging["n_scored_cells"] == round(
        BLANK_FLUID_FRACTION * averaging["n_cells"]
    )
    assert block["masking"]["scored_fraction"] == pytest.approx(
        averaging["n_scored_cells"] / averaging["n_cells"]
    )
    assert block["masking"]["scored_fraction"] <= block["masking"]["fluid_fraction"]
    with xarray.open_dataset(run_dir / "eval_fields.nc") as fields:
        # `isfinite(truth_velmag)` IS the scored set, which is what a figure
        # needs to grey out the cells no number was computed on.
        scored = np.isfinite(fields["truth_velmag"].values)
        assert int(scored.sum()) == averaging["n_scored_cells"]


def test_every_member_diverged_nulls_the_layer_with_a_reason(
    tmp_path: pathlib.Path,
) -> None:
    """All members non-finite is the degenerate shape, and it degrades properly."""
    run_dir = tmp_path / "run"
    _full_run_dir(run_dir, with_blanking=True)
    for member in range(N_MEMBERS):
        _diverge_member(run_dir, member)

    compute_metrics(run_dir)
    block = read_yaml(run_dir / "run_summary.yaml")["mean_field_metrics"]

    assert block["reason"] is not None
    assert block["hit_rate"]["u"] is None
    assert block["fac2_velmag"] is None
    assert block["averaging"]["n_members_scored"] is None
    # No sidecar for a layer that scored nothing.
    assert not (run_dir / "eval_fields.nc").exists()
    assert block["eval_fields"] is None


def test_the_sampling_floors_are_computed_on_the_cells_the_scores_are(
    tmp_path: pathlib.Path,
) -> None:
    """The floors must see the fluid mask and the stride, or they gate nothing.

    `hit_rate`'s allowance and the two RMSE floors are the truth's own sampling
    uncertainty *at the cells being scored*. Bootstrapping the raw slab instead
    pulls them down with the near-constant near-zero series inside buildings,
    and a floor that is too low is anti-conservative for its stated purpose: an
    RMSE genuinely inside the truth's sampling noise gets reported as above its
    floor. The regression this catches is precise -- the floors used to come
    out **byte-identical** with and without `blanking`.
    """
    masked_dir = tmp_path / "masked"
    unmasked_dir = tmp_path / "unmasked"
    _full_run_dir(masked_dir, with_blanking=True)
    _full_run_dir(unmasked_dir, with_blanking=False)

    compute_metrics(masked_dir)
    compute_metrics(unmasked_dir)
    masked = read_yaml(masked_dir / "run_summary.yaml")["mean_field_metrics"]
    unmasked = read_yaml(unmasked_dir / "run_summary.yaml")["mean_field_metrics"]

    for name in ("u", "v", "w"):
        assert masked["hit_rate"]["allowance"][name] != (
            unmasked["hit_rate"]["allowance"][name]
        ), name
    assert masked["tke_rmse"]["sampling_floor"] != (
        unmasked["tke_rmse"]["sampling_floor"]
    )
    assert masked["uw_stress_rmse"]["sampling_floor"] != (
        unmasked["uw_stress_rmse"]["sampling_floor"]
    )
    # The bootstrap ran over the truth's fluid cells and nothing else. The truth
    # grid here is the assimilation grid, so that count is exactly the
    # building's complement.
    cells = GRID_Z.size * GRID_Y.size * GRID_X.size
    assert masked["averaging"]["n_bootstrap_cells"] == round(
        BLANK_FLUID_FRACTION * cells
    )
    assert unmasked["averaging"]["n_bootstrap_cells"] == cells
    # The cap and the seed travel with the number, so it is reproducible rather
    # than merely stable.
    assert masked["averaging"]["n_bootstrap_cells_max"] >= (
        masked["averaging"]["n_bootstrap_cells"]
    )
    assert masked["averaging"]["bootstrap_seed"] is not None


def test_the_sampling_floors_follow_the_mean_field_stride(
    tmp_path: pathlib.Path,
) -> None:
    """A strided run scores fewer cells, so its floors describe fewer cells too.

    Same defect as the mask half: the floors used to be identical at
    `mean_field_stride=3` and at 1, i.e. they described a run that was not
    performed.
    """
    strided_dir = tmp_path / "strided"
    plain_dir = tmp_path / "plain"
    _full_run_dir(strided_dir, with_blanking=True, stride=3)
    _full_run_dir(plain_dir, with_blanking=True)

    compute_metrics(strided_dir)
    compute_metrics(plain_dir)
    strided = read_yaml(strided_dir / "run_summary.yaml")["mean_field_metrics"]
    plain = read_yaml(plain_dir / "run_summary.yaml")["mean_field_metrics"]

    assert strided["averaging"]["stride"] == 3
    assert strided["averaging"]["n_bootstrap_cells"] < (
        plain["averaging"]["n_bootstrap_cells"]
    )
    assert strided["hit_rate"]["allowance"]["u"] != plain["hit_rate"]["allowance"]["u"]


def test_stations_without_a_truth_column_are_counted_and_quantiles_persisted(
    tmp_path: pathlib.Path,
) -> None:
    """S3: the strict all-fluid stencil rule stays, but stops being silent.

    `_column_indicator` drops a station's ENTIRE truth column when its 2x2
    horizontal stencil touches a solid cell -- the right trade, since the
    alternative contaminates a profile with building values -- and stations
    default to the sensor x/y, which in an urban case sit on facades and roofs.
    So an empty truth column is the common case and WP1.4 would otherwise draw
    a posterior band with no truth line and no explanation.

    The quantiles are the other half: a "posterior band" can only be mean +-
    k*sigma unless the empirical fan is persisted, and recomputing it means
    re-reading the window states, which is what `eval_fields.nc` exists to
    prevent.
    """
    run_dir = tmp_path / "run"
    _full_run_dir(run_dir, with_blanking=True)

    compute_metrics(run_dir)
    block = read_yaml(run_dir / "run_summary.yaml")["mean_field_metrics"]

    averaging = block["averaging"]
    n_stations = len(ASSIM_POINTS[0]) + len(VALIDATION_POINTS[0])
    assert averaging["n_stations"] == n_stations
    # The fixture's building swallows three of the five default stations, so
    # this is a real count rather than a copy of `n_stations`.
    assert 0 < averaging["n_stations_with_truth"] < n_stations

    with xarray.open_dataset(run_dir / "eval_fields.nc") as fields:
        assert list(fields["quantile"].values) == [
            pytest.approx(q) for q in (0.05, 0.25, 0.5, 0.75, 0.95)
        ]
        columns = fields["station_ensemble_mean_quantile"]
        assert columns.dims == ("quantile", "component", "z", "station")
        values = columns.values
        # Monotone across the quantile axis, and the median is not the mean.
        assert bool(np.all(np.diff(values, axis=0) >= -1e-12))
        assert bool(
            np.any(np.abs(values[2] - fields["station_ensemble_mean"].values) > 1e-12)
        )
        # The station count in `averaging` is the one this file carries.
        finite = np.isfinite(fields["station_truth_tke"].values).any(axis=0)
        assert int(finite.sum()) == averaging["n_stations_with_truth"]


def test_mean_field_layer_nulls_every_key_when_it_never_ran(
    tmp_path: pathlib.Path,
) -> None:
    """The degraded path emits the same key TREE, not a single `null`.

    The master plan's rule is that a degenerate shape produces `null` plus a log
    line, never special-cased math -- and the whole value of that rule is that a
    consumer can index `mean_field_metrics.nmse.u.systematic` on an old run dir
    exactly as on a production one. Driven through `mean_field_scores(None, ...)`
    rather than through a broken run dir because every *other* reachable failure
    (a missing truth, an unknown solver name) already raises in WP1.2's sensor
    layer, several hundred lines earlier: this is the branch that survives.
    """
    healthy_dir = tmp_path / "run"
    _full_run_dir(healthy_dir)
    compute_metrics(healthy_dir)
    healthy = read_yaml(healthy_dir / "run_summary.yaml")["mean_field_metrics"]

    block, dataset = mean_field_scores(None, "unused.nc", None, 0.0, 0, 0.0, "pylbm")

    assert dataset is None  # nothing to persist, so no `eval_fields.nc`
    assert block["reason"] is not None
    assert block["eval_fields"] is None
    assert _key_tree(block) == _key_tree(healthy)
    for path, value in _iter_numbers(block):
        # The only numbers that survive are configuration, not measurement:
        # the hit rate's relative tolerance, which nulling would hide.
        assert path.endswith("relative_tolerance"), path


def _key_tree(obj: object) -> object:
    """The nested key structure of a summary block, with the values dropped.

    Leaves collapse to ``None`` whatever they hold, lists included: the whole
    point of the degraded block is that a *value* becomes ``null`` while the
    key path stays indexable, so `hit_rate.per_z` being a list of z-levels on
    one side and ``null`` on the other is the contract, not a difference.
    """
    if isinstance(obj, dict):
        return {key: _key_tree(value) for key, value in obj.items()}
    return None
