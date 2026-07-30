"""End-to-end wiring of the WP1.1 and WP1.2 bundles through ``compute_metrics``.

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
"""

from __future__ import annotations

import pathlib

import numpy as np
import pytest
import xarray

from scripts.esmda._esmda_common import PIT_BINS, read_yaml, write_yaml
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
    "reason",
}


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
) -> None:
    """A run dir carrying exactly what the WP1.1 block reads, and nothing else.

    ``skip_viz`` / ``extra_config`` / ``truth_access`` default to the
    parameter-only shape above. The WP1.2 tests flip ``skip_viz`` and supply the
    obs block and the truth-access keys the truth-consuming layers read; see
    :func:`_write_state_artifacts`, which writes the files they point at.
    """
    windows_dir = run_dir / "windows"
    windows_dir.mkdir(parents=True, exist_ok=True)
    write_yaml(
        {
            # With `skip_viz` true this returns before any truth-state read,
            # which is why the WP1.1 tests need no solver and no truth file.
            "run": {
                "skip_viz": skip_viz,
                "metrics": {"level": level, "bootstrap_blocks": BOOTSTRAP_BLOCKS},
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
    run_dir: pathlib.Path, rng: np.random.Generator, *, with_prior: bool = False
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
    seed: int = 11,
) -> None:
    """The WP1.1 parameter artifacts plus everything the WP1.2 layer reads."""
    obs_config, truth_access = _write_state_artifacts(
        run_dir, np.random.default_rng(seed), with_prior=with_prior
    )
    _write_run_dir(
        run_dir,
        *_dynamic_run_artifacts(),
        level=level,
        skip_viz=False,
        extra_config=obs_config,
        truth_access=truth_access,
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
        for key in ("w1_over_sigma_pooled", "self_floor", "w1_over_floor"):
            assert set(wasserstein[key]) == {"median", "max"}
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
    """
    run_dir = tmp_path / "run"
    _full_run_dir(run_dir, level="basic")

    compute_metrics(run_dir)
    summary = read_yaml(run_dir / "run_summary.yaml")

    assert summary["metrics_level"] == "basic"
    assert "sensor_statistics" not in summary
    assert "sensor_metrics" in summary and "state_metrics" in summary
    assert summary["metrics_version"] == 2  # unchanged: WP1.2 is additive
