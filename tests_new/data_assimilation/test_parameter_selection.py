"""Applied forecast fields are independent of estimated DA parameters."""

from typing import Any, cast

import jax.numpy as jnp
import numpy as np
import pytest
import xarray
from data_assimilation.filter_smoothing import FilterSmoothing
from data_assimilation.filtering import EnsembleKalmanFilter, RandomWalkEvolution
from data_assimilation.filtering.base import FilterMode
from data_assimilation.inflation import MultiplicativeInflation
from data_assimilation.localization.correlation import CorrelationLocalization
from data_assimilation.localization.distance import DistanceLocalization
from data_assimilation.smoothing.esmda import (
    ParameterESMDA,
    StateAndParameterESMDA,
    StateAndTimeVaryingParameterESMDA,
    StateESMDA,
    TimeVaryingParameterESMDA,
)


class _Forecast:
    save_on_disk = False
    results_dir = None

    def __init__(self) -> None:
        self.inputs: list[xarray.Dataset] = []

    def run_ensemble(self, state: Any = None, params: Any = None) -> xarray.Dataset:
        self.inputs.append(params.copy(deep=True))
        a = params.a
        start = a.isel(time=0).values if "time" in a.dims else a.values
        end = a.isel(time=-1).values if "time" in a.dims else a.values
        bias = params.bias
        fixed = bias.isel(time=-1).values if "time" in bias.dims else bias.values
        values = np.stack([start + fixed, end + fixed], axis=1)
        return xarray.Dataset(
            {"u": (("ensemble", "time", "x"), values[:, :, None])},
            coords={"ensemble": params.ensemble, "time": [0.0, 1.0], "x": [0.0]},
        )

    def apply_failure_substitutions_to_params(self, params: Any) -> Any:
        return params

    def apply_failure_substitutions_to_state(self, state: Any) -> Any:
        return state


class _Observation:
    def __call__(self, state: xarray.Dataset) -> jnp.ndarray:
        return jnp.asarray(state.u.isel(time=-1).values)


def _params(dynamic: bool = False, fixed_dynamic: bool = False) -> xarray.Dataset:
    a = np.linspace(-2, 2, 8).astype(np.float32)
    bias = np.linspace(-20.123456789, -18.123456789, 8)
    params = xarray.Dataset(
        {"a": ("ensemble", a), "bias": ("ensemble", bias)},
        coords={
            "ensemble": np.arange(8),
            "member_label": ("ensemble", np.arange(8) + 10),
        },
        attrs={"model_config": "full"},
    )
    if dynamic:
        params["a"] = (("ensemble", "time"), np.stack([a, a + 1], axis=1))
    if fixed_dynamic:
        params["bias"] = (("ensemble", "time"), np.stack([bias, bias + 0.25], axis=1))
    if dynamic or fixed_dynamic:
        params = params.assign_coords(time=[0.0, 1.0])
    params.bias.attrs = {"units": "dimensionless", "role": "fixed"}
    return params


def _smoother(cls: Any, model: Any, selection: Any = None, **kwargs: Any) -> Any:
    if issubclass(cls, TimeVaryingParameterESMDA):
        kwargs["num_time_points"] = 2
    return cls(
        observation_operator=_Observation(),
        forward_model=cast(Any, model),
        C_D=jnp.array([0.2]),
        num_steps=1,
        parameter_names_to_estimate=selection,
        **kwargs,
    )


@pytest.mark.parametrize(  # type: ignore[misc]
    "cls",
    [
        ParameterESMDA,
        StateAndParameterESMDA,
        StateESMDA,
        TimeVaryingParameterESMDA,
        StateAndTimeVaryingParameterESMDA,
    ],
)
@pytest.mark.parametrize("selection", [["a"], []])  # type: ignore[misc]
def test_smoothers_forecast_all_fields_and_update_only_selected(
    cls: Any, selection: Any
) -> None:
    dynamic = issubclass(cls, TimeVaryingParameterESMDA)
    params = _params(dynamic=dynamic, fixed_dynamic=dynamic)
    model = _Forecast()
    smoother = _smoother(cls, model, selection)
    history = smoother(
        params=params,
        observations=jnp.array([0.0]),
        return_params_history=True,
        final_forecast=False,
    )
    posterior = history.isel(esmda_step=-1, drop=True)
    xarray.testing.assert_identical(posterior.bias, params.bias)
    assert posterior.attrs == params.attrs
    if not selection or cls is StateESMDA:
        xarray.testing.assert_identical(posterior, params)
    else:
        assert not np.array_equal(posterior.a.values, params.a.values)
    assert len(model.inputs) == 1
    xarray.testing.assert_identical(model.inputs[0], params)
    for step in range(history.sizes["esmda_step"]):
        xarray.testing.assert_identical(
            history.bias.isel(esmda_step=step, drop=True), params.bias
        )


@pytest.mark.parametrize("mode", ["state", "parameter", "joint"])  # type: ignore[misc]
@pytest.mark.parametrize("selection", [["a"], []])  # type: ignore[misc]
def test_filter_excludes_fixed_fields_from_analysis_inflation_and_evolution(
    mode: FilterMode, selection: Any
) -> None:
    params = _params(fixed_dynamic=True)
    model = _Forecast()
    filter = EnsembleKalmanFilter(
        observation_operator=_Observation(),
        forward_model=cast(Any, model),
        C_D=jnp.array([0.2]),
        mode=mode,
        parameter_names_to_estimate=selection,
        inflation=MultiplicativeInflation(1.3),
        parameter_evolution=None if mode == "state" else RandomWalkEvolution(2.0),
    )
    result = filter.run(
        params=params, observations=jnp.array([[0.0], [1.0]]), return_history=True
    )
    assert result.params is not None and result.params_history is not None
    xarray.testing.assert_identical(result.params.bias, params.bias)
    for forecast_input in model.inputs:
        xarray.testing.assert_identical(forecast_input.bias, params.bias)
    if not selection or mode == "state":
        xarray.testing.assert_identical(result.params, params)
    else:
        assert not np.array_equal(result.params.a.values, params.a.values)
    for cycle in range(result.params_history.sizes["cycle"]):
        xarray.testing.assert_identical(
            result.params_history.bias.isel(cycle=cycle, drop=True), params.bias
        )


@pytest.mark.parametrize(  # type: ignore[misc]
    "cls", [ParameterESMDA, TimeVaryingParameterESMDA, EnsembleKalmanFilter]
)
def test_missing_selection_fails_before_forecast(cls: Any) -> None:
    model = _Forecast()
    if cls is EnsembleKalmanFilter:
        instance = cls(
            observation_operator=_Observation(),
            forward_model=cast(Any, model),
            C_D=jnp.array([0.2]),
            mode="state",
            parameter_names_to_estimate=["missing"],
        )
        call = instance.run
    else:
        instance = _smoother(cls, model, ["missing"])
        call = instance
    with pytest.raises(ValueError, match="absent from params"):
        call(params=_params(), observations=jnp.array([[0.0]]))
    assert not model.inputs


@pytest.mark.parametrize("cls", [ParameterESMDA, EnsembleKalmanFilter])  # type: ignore[misc]
@pytest.mark.parametrize("selection", [["a", "a"], "a", [""]])  # type: ignore[misc]
def test_invalid_selection_rejected_at_construction(cls: Any, selection: Any) -> None:
    with pytest.raises(ValueError, match="parameter_names_to_estimate"):
        if cls is EnsembleKalmanFilter:
            cls(
                observation_operator=_Observation(),
                forward_model=cast(Any, _Forecast()),
                C_D=jnp.array([0.2]),
                mode="state",
                parameter_names_to_estimate=selection,
            )
        else:
            _smoother(cls, _Forecast(), selection)


def test_dynamic_explicit_group_ids_are_restricted_to_selected_knot_rows() -> None:
    params = _params(dynamic=True, fixed_dynamic=True)
    smoother = _smoother(
        TimeVaryingParameterESMDA,
        _Forecast(),
        ["a"],
        localization=CorrelationLocalization(truncation_correlation=0.1),
    )
    groups = smoother._time_varying_group_ids(params)
    pred_obs = jnp.asarray(params.a.isel(time=-1).values)[None, :]
    updated = smoother.update_params_from_pred_obs(
        params, pred_obs, jnp.array([0.0]), group_ids=groups
    )
    xarray.testing.assert_identical(updated.bias, params.bias)
    assert not np.array_equal(updated.a.values, params.a.values)


def test_distance_localization_ignores_unestimated_dynamic_fields() -> None:
    params = _params(dynamic=True)
    smoother = _smoother(
        TimeVaryingParameterESMDA,
        _Forecast(),
        ["bias"],
        localization=DistanceLocalization(localization_radius=5),
        global_parameter_names=["bias"],
    )
    posterior = smoother(
        params=params, observations=jnp.array([0.0]), final_forecast=False
    )
    xarray.testing.assert_identical(posterior.a, params.a)
    assert not np.array_equal(posterior.bias.values, params.bias.values)


@pytest.mark.parametrize(  # type: ignore[misc]
    "cls", [ParameterESMDA, TimeVaryingParameterESMDA, EnsembleKalmanFilter]
)
def test_none_selection_matches_explicit_all_values_and_rng(cls: Any) -> None:
    params = _params(dynamic=cls is TimeVaryingParameterESMDA)
    instances = []
    for names in [None, list(params.data_vars)]:
        if cls is EnsembleKalmanFilter:
            instance = cls(
                observation_operator=_Observation(),
                forward_model=cast(Any, _Forecast()),
                C_D=jnp.array([0.2]),
                mode="parameter",
                parameter_evolution=RandomWalkEvolution(0.1),
                parameter_names_to_estimate=names,
            )
        else:
            instance = _smoother(cls, _Forecast(), names)
        instances.append(instance)
    if cls is EnsembleKalmanFilter:
        results = [
            i.run(params=params, observations=jnp.array([[0.0]])).params
            for i in instances
        ]
    else:
        results = [
            i(params=params, observations=jnp.array([0.0]), final_forecast=False)
            for i in instances
        ]
    xarray.testing.assert_equal(results[0], results[1])
    np.testing.assert_array_equal(instances[0].rng_key, instances[1].rng_key)


def test_empty_joint_filter_selection_needs_no_parameter_spread_maintenance() -> None:
    filter = EnsembleKalmanFilter(
        observation_operator=_Observation(),
        forward_model=cast(Any, _Forecast()),
        C_D=jnp.array([0.2]),
        mode="joint",
        parameter_names_to_estimate=[],
    )
    params = _params()
    result = filter.run(params=params, observations=jnp.array([[0.0]]))
    xarray.testing.assert_identical(result.params, params)


class _WindowObservation:
    def __call__(self, state: xarray.Dataset) -> xarray.DataArray:
        return cast(xarray.DataArray, state.u.rename(x="obs"))


@pytest.mark.parametrize("dynamic", [False, True])  # type: ignore[misc]
@pytest.mark.parametrize("selection", [["a"], []])  # type: ignore[misc]
def test_hybrid_retains_unestimated_fields_across_both_phases(
    dynamic: bool, selection: Any
) -> None:
    params = _params(dynamic=dynamic)
    smoother_model, filter_model = _Forecast(), _Forecast()
    smoother_kwargs: dict[str, Any] = {"num_time_points": 2} if dynamic else {}
    smoother_cls = TimeVaryingParameterESMDA if dynamic else ParameterESMDA
    smoother = smoother_cls(
        observation_operator=cast(Any, _WindowObservation()),
        forward_model=cast(Any, smoother_model),
        C_D=jnp.array([0.2, 0.2]),
        num_steps=1,
        parameter_names_to_estimate=selection,
        **smoother_kwargs,
    )
    filter = EnsembleKalmanFilter(
        observation_operator=_Observation(),
        forward_model=cast(Any, filter_model),
        C_D=jnp.array([0.2]),
        mode="joint",
        parameter_evolution=RandomWalkEvolution(0.1),
        parameter_names_to_estimate=selection,
    )
    hybrid = FilterSmoothing(smoother=smoother, filter=filter)
    observations = [
        xarray.DataArray(
            [[0.0]], dims=("time", "obs"), coords={"time": [float(t)], "obs": [0.0]}
        )
        for t in [1, 2]
    ]
    result = hybrid.run(params=params, observations=observations, return_history=True)
    xarray.testing.assert_identical(result.esmda_params.bias, params.bias)
    assert result.params is not None
    xarray.testing.assert_identical(result.params.bias, params.bias)
    for forecast_input in smoother_model.inputs + filter_model.inputs:
        xarray.testing.assert_identical(forecast_input.bias, params.bias)
    assert len(smoother_model.inputs) == 1 and len(filter_model.inputs) == 2
