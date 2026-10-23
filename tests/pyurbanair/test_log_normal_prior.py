"""The LogNormal prior: sampling statistics, bounds and the log-space lookup."""

from __future__ import annotations

import numpy as np
import pytest

from pyurbanair.dynamic_parameters.ar2_relaxation import AR2RelaxationModel
from pyurbanair.static_parameters import (
    Constant,
    LogNormal,
    Normal,
    ParameterSampler,
    log_space_parameter_names,
)


def test_log_normal_sampling_statistics() -> None:
    sampler = ParameterSampler(
        {"z0": LogNormal(median=0.05, log_std=0.7), "a": Normal(0.0, 1.0)}, seed=3
    )
    z0 = np.asarray(sampler.sample(20000).z0.values, dtype=float)
    assert np.all(z0 > 0.0)
    assert np.median(z0) == pytest.approx(0.05, rel=0.03)
    assert np.log(z0).std() == pytest.approx(0.7, rel=0.03)


def test_log_normal_truncates_and_validates() -> None:
    import jax

    prior = LogNormal(0.05, 0.7, min=0.02, max=0.1)
    values = np.asarray(prior.sample(jax.random.PRNGKey(0), 5000))
    # Truncated, not clamped: no point mass at the bounds, the tails are filled.
    assert values.min() > 0.02 and values.max() < 0.1
    assert values.min() < 0.0205 and values.max() > 0.0975
    assert prior.bounds == (0.02, 0.1)
    assert LogNormal(0.05, 0.7, max=0.1).bounds is None
    with pytest.raises(ValueError, match="median"):
        LogNormal(0.0, 0.7)
    with pytest.raises(ValueError, match="log_std"):
        LogNormal(0.05, -1.0)


@pytest.mark.parametrize(  # type: ignore[misc]
    "bounds, message",
    [
        ({"min": 0.3, "max": 0.005}, "min < median"),
        ({"min": 0.1, "max": 0.3}, "min < median"),
        ({"min": 0.005, "max": 0.05}, "median < max"),
        ({"min": 0.0}, "0 < min"),
    ],
)
def test_log_normal_rejects_bounds_around_the_median(
    bounds: dict, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        LogNormal(0.05, 0.7, **bounds)


def test_log_normal_without_bounds_samples_as_before() -> None:
    """No min/max: the plain draw exp(log(median) + log_std * N(0, 1))."""
    import jax
    import jax.numpy as jnp

    key = jax.random.PRNGKey(5)
    expected = jnp.exp(jax.random.normal(key, (64,)) * 0.7 + jnp.log(0.05))
    np.testing.assert_array_equal(LogNormal(0.05, 0.7).sample(key, 64), expected)


def test_log_space_parameter_names() -> None:
    static = ParameterSampler(
        {"z0": LogNormal(0.05, 0.7), "a": Normal(0.0, 1.0), "b": Constant(1.0)}
    )
    assert log_space_parameter_names(static) == {"z0": None}
    bounded = ParameterSampler({"z0": LogNormal(0.05, 0.7, min=0.005, max=0.3)})
    assert log_space_parameter_names(bounded) == {"z0": (0.005, 0.3)}
    dynamic = AR2RelaxationModel(
        external_parameters={"inflow_angle": Normal(0.0, 10.0)},
        simulation_time=60.0,
        seconds_per_knot=30.0,
        correlation_length=100.0,
        static_parameters={"z0": LogNormal(0.05, 0.7)},
    )
    assert log_space_parameter_names(dynamic) == {"z0": None}
    assert np.all(dynamic.sample(4).z0.values > 0.0)


def test_log_normal_is_not_a_time_varying_external_prior() -> None:
    model = AR2RelaxationModel(
        external_parameters={"z0": LogNormal(0.05, 0.7)},
        simulation_time=60.0,
        seconds_per_knot=30.0,
        correlation_length=100.0,
    )
    with pytest.raises(TypeError, match="not usable as a dynamic external prior"):
        model.sample(4)
