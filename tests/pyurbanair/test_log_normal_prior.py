"""The LogNormal prior: sampling statistics and the log-space name lookup."""

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


def test_log_normal_clamps_and_validates() -> None:
    import jax

    values = LogNormal(0.05, 0.7, min=0.02, max=0.1).sample(jax.random.PRNGKey(0), 5000)
    assert float(values.min()) == pytest.approx(0.02)
    assert float(values.max()) == pytest.approx(0.1)
    with pytest.raises(ValueError, match="median"):
        LogNormal(0.0, 0.7)
    with pytest.raises(ValueError, match="log_std"):
        LogNormal(0.05, -1.0)


def test_log_space_parameter_names() -> None:
    static = ParameterSampler(
        {"z0": LogNormal(0.05, 0.7), "a": Normal(0.0, 1.0), "b": Constant(1.0)}
    )
    assert log_space_parameter_names(static) == ["z0"]
    dynamic = AR2RelaxationModel(
        external_parameters={"inflow_angle": Normal(0.0, 10.0)},
        simulation_time=60.0,
        seconds_per_knot=30.0,
        correlation_length=100.0,
        static_parameters={"z0": LogNormal(0.05, 0.7)},
    )
    assert log_space_parameter_names(dynamic) == ["z0"]
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
