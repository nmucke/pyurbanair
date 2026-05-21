"""Time-varying external priors: mean/std may be per-window profiles.

The parameter_time_series models draw a unit-variance anomaly and apply a
mean(t) + std(t)·z envelope, so a list-valued external mean/std varies over the
window while a scalar reproduces the historical constant behavior.
"""

from __future__ import annotations

import jax
import numpy as np
import pytest

from pyurbanair.parameter_time_series import build_parameter_time_series

_METHODS = [
    ("ar2_relaxation", {"correlation_length": 300.0}),
    ("ar1", {"correlation_length": 300.0}),
    ("ornstein_uhlenbeck", {"correlation_length": 300.0}),
    ("gp_linear_trend", {"correlation_length": 300.0}),
]


@pytest.mark.parametrize("method,kwargs", _METHODS)
def test_profile_mean_and_std_track_control_points(method, kwargs) -> None:
    external = {
        "inflow_angle": {"mean": [0.0, 40.0, 10.0], "std": [2.0, 10.0, 2.0]},
        "velocity_magnitude": {"mean": 5.0, "std": 0.5, "min": 0.1},
    }
    model = build_parameter_time_series(
        method=method, external_priors=external, ensemble_size=4000,
        method_kwargs=kwargs,
    )
    # window endpoints + midpoint coincide with the three control points
    time_coords = np.array([0.0, 300.0, 600.0])
    ds = model.sample_prior(time_coords, jax.random.PRNGKey(0))
    angle = ds["inflow_angle"].transpose("time", "ensemble").values

    np.testing.assert_allclose(angle.mean(1), [0.0, 40.0, 10.0], atol=1.0)
    np.testing.assert_allclose(angle.std(1), [2.0, 10.0, 2.0], rtol=0.15)

    # the constant-spec parameter stays constant in mean
    vel = ds["velocity_magnitude"].transpose("time", "ensemble").values
    np.testing.assert_allclose(vel.mean(1), 5.0, atol=0.1)


@pytest.mark.parametrize("method,kwargs", _METHODS)
def test_scalar_specs_are_unchanged_by_refactor(method, kwargs) -> None:
    """A scalar mean/std must give a stationary (constant-in-mean) series."""
    external = {"inflow_angle": {"mean": 12.0, "std": 3.0}}
    model = build_parameter_time_series(
        method=method, external_priors=external, ensemble_size=4000,
        method_kwargs=kwargs,
    )
    ds = model.sample_prior(np.linspace(0, 600, 5), jax.random.PRNGKey(1))
    angle = ds["inflow_angle"].transpose("time", "ensemble").values
    # ensemble mean ~ 12 at every time; ensemble std ~ 3
    np.testing.assert_allclose(angle.mean(1), 12.0, atol=0.6)
    np.testing.assert_allclose(angle.std(1), 3.0, rtol=0.2)
