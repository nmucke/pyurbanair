"""The dtype contract of ``evaluation.scores.crps_ensemble``.

WP0.2 merged two CRPS implementations that looked identical but were not:
``da_metrics.per_knot_crps`` scored in whatever dtype it was handed, while
``plotting._crps_ensemble`` upcast to float64 first. Parameter artifacts are
float32 on disk, and the pairwise term differs between the two at the ~1e-7
level, so the merge had to preserve *both* behaviours: the function itself no
longer casts, and the one caller that relied on the upcast
(:func:`compute_parameter_metrics`) does it explicitly.

Nothing else pins that split. Without these tests, "tidying up" by moving the
cast back inside the function — or deleting it at the call site — silently
shifts every emitted CRPS number, and no other test in the suite notices.
"""

import numpy as np
import xarray
from evaluation.scores import compute_parameter_metrics, crps_ensemble


def _param_datasets(
    dtype: str,
) -> tuple[xarray.Dataset, xarray.Dataset]:
    """A tiny (ensemble, time) posterior and its matching truth, in ``dtype``."""
    rng = np.random.default_rng(0)
    time = np.array([0.0, 1.0, 2.0, 3.0])
    members = rng.standard_normal((6, time.size)).astype(dtype)
    truth = rng.standard_normal(time.size).astype(dtype)

    posterior = xarray.Dataset(
        {"inflow_angle": (("ensemble", "time"), members)},
        coords={"time": time, "ensemble": np.arange(members.shape[0])},
    )
    true = xarray.Dataset(
        {"inflow_angle": (("time",), truth)},
        coords={"time": time},
    )
    return posterior, true


def test_crps_ensemble_does_not_upcast() -> None:
    # The merged function inherited per_knot_crps's behaviour: dtype is the
    # caller's business. scripts/_common.py relies on this -- it passes raw
    # float32 parameter history and its metrics.csv digits depend on it.
    ens = np.random.default_rng(1).standard_normal((8, 5)).astype(np.float32)
    truth = np.random.default_rng(2).standard_normal(5).astype(np.float32)

    out = crps_ensemble(ens, truth)

    assert out.dtype == np.float32, (
        "crps_ensemble must not cast; float64 here would silently shift "
        "metrics.csv for every float32 parameter run"
    )


def test_crps_ensemble_float32_and_float64_actually_differ() -> None:
    # Guards the premise of the whole split. If these ever agree bitwise, the
    # explicit cast below is pointless and this file can go.
    ens = np.random.default_rng(3).standard_normal((16, 12)).astype(np.float32)
    truth = np.random.default_rng(4).standard_normal(12).astype(np.float64)

    in_float32 = crps_ensemble(ens, truth.astype(np.float32))
    in_float64 = crps_ensemble(np.asarray(ens, dtype=float), truth)

    assert not np.array_equal(in_float32, in_float64)


def test_compute_parameter_metrics_scores_crps_in_float64() -> None:
    # The compensating half: this caller upcast implicitly before the merge
    # (via _crps_ensemble), so it must upcast explicitly after it.
    posterior, true = _param_datasets("float32")

    metrics = compute_parameter_metrics(posterior, true)
    crps = metrics["inflow_angle"]["crps"]

    assert crps.dtype == np.float64

    members = np.asarray(posterior["inflow_angle"].transpose("ensemble", "time").values)
    truth = np.asarray(true["inflow_angle"].values, dtype=float)
    expected = crps_ensemble(np.asarray(members, dtype=float), truth)

    np.testing.assert_array_equal(crps, expected)
