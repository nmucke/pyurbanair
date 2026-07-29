"""End-to-end wiring of the WP1.1 bundle through ``compute_metrics``.

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
BUNDLE_PARAM_KEYS = {"zscore", "pit_counts", "pit", "coverage", "contraction_ratio"}
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
}


def _write_run_dir(
    run_dir: pathlib.Path,
    posterior: xarray.Dataset,
    prior: xarray.Dataset,
    truth: xarray.Dataset,
    *,
    level: str = "standard",
    num_windows: int | None = N_WINDOWS,
) -> None:
    """A run dir carrying exactly what the WP1.1 block reads, and nothing else."""
    windows_dir = run_dir / "windows"
    windows_dir.mkdir(parents=True)
    write_yaml(
        {
            # `skip_viz` returns before any truth-state read, which is why this
            # works without a solver or a truth file.
            "run": {"skip_viz": True, "metrics": {"level": level}},
            # Where the GP knot spacing actually lives in a saved config.
            "prior_params": {
                "correlation_length": CORRELATION_LENGTH,
                "seconds_per_knot": SECONDS_PER_KNOT,
            },
        },
        run_dir / "config.yaml",
    )
    write_yaml(
        {"configuration": {"ensemble_size": N_MEMBERS}}, run_dir / "run_info.yaml"
    )
    write_yaml(
        {} if num_windows is None else {"num_windows": num_windows},
        run_dir / "truth_access.yaml",
    )
    posterior.to_netcdf(run_dir / "posterior_params.nc")
    posterior.to_netcdf(windows_dir / "window_0_posterior_params.nc")
    prior.to_netcdf(run_dir / "prior_params.nc")
    truth.to_netcdf(run_dir / "true_params.nc")


def _dynamic_run_artifacts() -> tuple[xarray.Dataset, xarray.Dataset, xarray.Dataset]:
    """A dynamic run: knots concatenated across windows, plus a broadcast static.

    This is the shape `run_esmda._concat_windows` produces -- a time-varying
    `inflow_angle` on a `time` coordinate, and the `static_parameters` block
    (`vertical_inflow_exponent`) broadcast to every knot, i.e. piecewise
    constant with one step per window. Both PIT branches of the bundle fire on
    one dataset, and the joint vector spans both parameters.
    """
    rng = np.random.default_rng(7)
    n_knots = N_WINDOWS * KNOTS_PER_WINDOW
    times = np.arange(n_knots) * SECONDS_PER_KNOT

    centre = 270.0 + np.cumsum(rng.normal(0.0, 0.5, size=n_knots))
    angle_post = centre[None, :] + rng.normal(0.0, 1.0, size=(N_MEMBERS, n_knots))
    angle_prior = centre[None, :] + rng.normal(0.0, 3.0, size=(N_MEMBERS, n_knots))
    angle_truth = centre + rng.normal(0.0, 1.0, size=n_knots)

    levels_post = 0.3 + rng.normal(0.0, 0.02, size=(N_MEMBERS, N_WINDOWS))
    levels_prior = 0.3 + rng.normal(0.0, 0.08, size=(N_MEMBERS, N_WINDOWS))
    exp_post = np.repeat(levels_post, KNOTS_PER_WINDOW, axis=1)
    exp_prior = np.repeat(levels_prior, KNOTS_PER_WINDOW, axis=1)

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
        assert set(entry["zscore"]) == {"mean", "std", "max_abs", "overconfident"}
        assert set(entry["coverage"]) == {"alpha_50", "alpha_90", "max_nominal_alpha"}
        assert set(entry["contraction_ratio"]) == {"mean", "min"}
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
        # The update shrank the spread, so the ratio is a real number below 1.
        assert 0.0 < entry["contraction_ratio"]["mean"] < 1.0

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
