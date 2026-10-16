"""Log-space parameters: the Kalman update works on log(value), every forecast
and returned Dataset holds the physical (positive) value."""

from typing import Any, cast

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import xarray
from data_assimilation.filtering import EnsembleKalmanFilter, RandomWalkEvolution
from data_assimilation.parameter_selection import merge_parameters, select_parameters
from data_assimilation.smoothing.esmda import (
    ParameterESMDA,
    StateAndParameterESMDA,
    StateAndTimeVaryingParameterESMDA,
    TimeVaryingParameterESMDA,
)

TRUE_Z0 = 0.1
N_E = 200


class _Forecast:
    """Observes log(z0) and a: linear in log space, so ESMDA recovers log(z0).

    Like the LES backends, a non-positive z0 is an error, not a member failure.
    """

    save_on_disk = False
    results_dir = None

    def __init__(self) -> None:
        self.inputs: list[xarray.Dataset] = []

    def run_ensemble(self, state: Any = None, params: Any = None) -> xarray.Dataset:
        self.inputs.append(params.copy(deep=True))
        z0 = np.asarray(params.z0.values, dtype=float)
        if np.any(z0 <= 0.0):
            raise ValueError("roughness length must be positive")
        a = params.a
        a = np.asarray(a.isel(time=-1).values if "time" in a.dims else a.values)
        values = np.stack([np.log(z0), a], axis=1)
        values = np.stack([values, values], axis=1)  # two frames
        return xarray.Dataset(
            {"u": (("ensemble", "time", "x"), values)},
            coords={"ensemble": params.ensemble, "time": [0.0, 1.0], "x": [0.0, 1.0]},
        )

    def apply_failure_substitutions_to_params(self, params: Any) -> Any:
        return params

    def apply_failure_substitutions_to_state(self, state: Any) -> Any:
        return state


class _Observation:
    def __call__(self, state: xarray.Dataset) -> jnp.ndarray:
        return jnp.asarray(state.u.isel(time=-1).values)


def _params(dynamic: bool = False) -> xarray.Dataset:
    """Log-normal z0 (median 0.05, log_std 0.7) and a Normal a."""
    k1, k2 = jax.random.split(jax.random.PRNGKey(0))
    z0 = np.exp(np.log(0.05) + 0.7 * np.asarray(jax.random.normal(k1, (N_E,))))
    a = np.asarray(jax.random.normal(k2, (N_E,)))
    params = xarray.Dataset(
        {"z0": ("ensemble", z0), "a": ("ensemble", a)},
        coords={"ensemble": np.arange(N_E)},
    )
    if dynamic:
        params["a"] = (("ensemble", "time"), np.stack([a, a], axis=1))
        params = params.assign_coords(time=[0.0, 1.0])
    params.z0.attrs = {"units": "m"}
    return params


OBS = jnp.array([np.log(TRUE_Z0), 0.5])


def _smoother(cls: Any, model: Any, log_names: Any, num_steps: int = 4) -> Any:
    extra = {"num_time_points": 2} if issubclass(cls, TimeVaryingParameterESMDA) else {}
    return cls(
        observation_operator=_Observation(),
        forward_model=cast(Any, model),
        C_D=jnp.array([0.01, 0.01]),
        num_steps=num_steps,
        rng_key=jax.random.PRNGKey(1),
        parameter_names_to_estimate=["z0", "a"],
        log_parameter_names=log_names,
        **extra,
    )


def test_select_and_merge_round_trip() -> None:
    params = _params()
    selected = select_parameters(params, ["z0"], ["z0"])
    np.testing.assert_allclose(selected.z0.values, np.log(params.z0.values))
    assert "a" not in selected
    merged = merge_parameters(params, selected, ["z0"])
    np.testing.assert_allclose(merged.z0.values, params.z0.values, rtol=1e-6)
    assert merged.z0.attrs == {"units": "m"}
    # A log name that is not estimated is left alone.
    xarray.testing.assert_identical(
        select_parameters(params, ["a"], ["z0"]), select_parameters(params, ["a"])
    )


def test_untransformed_parameters_are_unchanged() -> None:
    params = _params()
    selected = select_parameters(params, None, ["z0"])
    xarray.testing.assert_identical(selected.a, params.a)
    updated = selected.assign(a=selected.a + 1.0)
    merged = merge_parameters(params, updated, ["z0"])
    xarray.testing.assert_identical(
        merged.a, merge_parameters(params, updated[["a"]]).a
    )


@pytest.mark.parametrize(  # type: ignore[misc]
    "z0, message",
    [
        (np.zeros(N_E), "strictly positive"),
        (-np.ones(N_E), "strictly positive"),
    ],
)
def test_non_positive_values_are_rejected(z0: Any, message: str) -> None:
    params = _params().assign(z0=("ensemble", z0))
    with pytest.raises(ValueError, match=message):
        select_parameters(params, None, ["z0"])


def test_time_varying_and_absent_log_parameters_are_rejected() -> None:
    params = _params(dynamic=True)
    with pytest.raises(ValueError, match="must be static"):
        select_parameters(params, None, ["a"])
    with pytest.raises(ValueError, match="absent from params"):
        select_parameters(params, None, ["missing"])


@pytest.mark.parametrize(  # type: ignore[misc]
    "cls",
    [
        ParameterESMDA,
        StateAndParameterESMDA,
        TimeVaryingParameterESMDA,
        StateAndTimeVaryingParameterESMDA,
    ],
)
def test_esmda_recovers_a_log_scale_truth_and_stays_positive(cls: Any) -> None:
    params = _params(dynamic=issubclass(cls, TimeVaryingParameterESMDA))
    model = _Forecast()
    output = _smoother(cls, model, ["z0"])(
        params=params, observations=OBS, return_params_history=True
    )
    history = output[0] if isinstance(output, tuple) else output
    # Every forecast saw a physical, positive z0, and so does every saved step.
    assert all(bool((p.z0 > 0).all()) for p in model.inputs)
    assert bool((history.z0 > 0).all())
    posterior = history.isel(esmda_step=-1)
    # Physical units: the posterior median is z0 itself, not log(z0).
    assert float(np.median(posterior.z0)) == pytest.approx(TRUE_Z0, rel=0.1)
    assert posterior.z0.attrs == {"units": "m"}


def test_physical_update_of_the_same_problem_crashes_the_forecast() -> None:
    """The failure the transform removes: additive updates push z0 <= 0."""
    small = jnp.array([np.log(0.01), 0.5])
    with pytest.raises(ValueError, match="must be positive"):
        _smoother(ParameterESMDA, _Forecast(), None)(
            params=_params(), observations=small
        )
    history = _smoother(ParameterESMDA, _Forecast(), ["z0"])(
        params=_params(), observations=small, return_params_history=True
    )
    assert bool((history[0].z0 > 0).all())


def test_esmda_untransformed_parameter_is_bit_identical() -> None:
    """With one step both runs share the prior forecast: `a` must not change."""
    runs = [
        _smoother(ParameterESMDA, _Forecast(), names, num_steps=1)(
            params=_params(),
            observations=OBS,
            return_params_history=True,
            final_forecast=False,
        )
        for names in (None, ["z0"])
    ]
    xarray.testing.assert_identical(runs[0].a, runs[1].a)
    assert not np.allclose(runs[0].z0.values, runs[1].z0.values)


def test_filter_keeps_log_parameters_positive() -> None:
    model = _Forecast()
    enkf = EnsembleKalmanFilter(
        observation_operator=_Observation(),
        forward_model=cast(Any, model),
        C_D=jnp.array([0.01, 0.01]),
        mode="parameter",
        rng_key=jax.random.PRNGKey(1),
        log_parameter_names=["z0"],
        # In log units for z0: a multiplicative random walk.
        parameter_evolution=RandomWalkEvolution(std={"z0": 0.05, "a": 0.05}),
    )
    result = enkf.run(
        params=_params(), observations=jnp.stack([OBS, OBS]), return_history=True
    )
    assert result.params is not None and result.params_history is not None
    assert all(bool((p.z0 > 0).all()) for p in model.inputs)
    assert bool((result.params_history.z0 > 0).all())
    assert float(np.median(result.params.z0)) == pytest.approx(TRUE_Z0, rel=0.1)
