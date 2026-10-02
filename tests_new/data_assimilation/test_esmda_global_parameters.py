"""Global static parameters retain the full update under localization."""

from types import SimpleNamespace
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import xarray as xr
from data_assimilation.filtering import (
    EnsembleKalmanFilter,
    ETKFAnalysis,
    IdentityEvolution,
    LETKFAnalysis,
)
from data_assimilation.localization.correlation import CorrelationLocalization
from data_assimilation.localization.distance import DistanceLocalization
from data_assimilation.smoothing.esmda import (
    ParameterESMDA,
    StateAndParameterESMDA,
    StateAndTimeVaryingParameterESMDA,
    TimeVaryingParameterESMDA,
)


def _prior() -> xr.Dataset:
    a = np.arange(8, dtype=float) - 3.5
    result = xr.Dataset(
        {"global_bias": ("ensemble", a), "local_bias": ("ensemble", a)},
        coords={"ensemble": np.arange(8)},
        attrs={"experiment": "global metadata"},
    )
    result.global_bias.attrs = {"units": "1", "localization": "global"}
    return result


def _predicted() -> jnp.ndarray:
    # Both parameters have a finite, weak correlation with the observation.
    a = np.arange(8, dtype=float) - 3.5
    noise = np.array([1, -1, -1, 1, 1, -1, -1, 1])
    return jnp.asarray((a + 20 * noise)[None, :])


def _smoother(cls: Any = ParameterESMDA, **kwargs: Any) -> Any:
    if issubclass(cls, TimeVaryingParameterESMDA):
        kwargs.setdefault("num_time_points", 2)
    return cls(
        observation_operator=lambda state: state.u.values,
        forward_model=SimpleNamespace(save_on_disk=False, results_dir=None),
        C_D=jnp.asarray([0.5]),
        num_steps=1,
        rng_key=jax.random.PRNGKey(12),
        **kwargs,
    )


def _localization(block_grouping: bool = False) -> CorrelationLocalization:
    return CorrelationLocalization(
        truncation_correlation=0.99, block_grouping=block_grouping
    )


@pytest.mark.parametrize("block_grouping", [False, True])  # type: ignore[misc]
def test_global_row_matches_unlocalized_update_and_local_row_is_excluded(
    block_grouping: bool,
) -> None:
    params, obs = _prior(), jnp.asarray([5.0])
    smoother = _smoother(
        localization=_localization(block_grouping),
        global_parameter_names=["global_bias"],
    )
    expected = _smoother().update_params_from_pred_obs(params, _predicted(), obs)
    result = smoother.update_params_from_pred_obs(params, _predicted(), obs)
    np.testing.assert_allclose(result.global_bias, expected.global_bias, atol=1e-6)
    np.testing.assert_array_equal(result.local_bias, params.local_bias)
    assert not np.array_equal(result.global_bias, params.global_bias)
    assert result.attrs == params.attrs
    assert result.global_bias.attrs == params.global_bias.attrs
    # The same metadata applies to subsequent analyses, including fresh Datasets.
    repeated = smoother.update_params_from_pred_obs(result, _predicted(), obs)
    np.testing.assert_array_equal(repeated.local_bias, params.local_bias)
    assert smoother.global_parameter_names == ("global_bias",)


@pytest.mark.parametrize("localization", [None, _localization()])  # type: ignore[misc]
def test_empty_global_declaration_preserves_values_and_rng(localization: Any) -> None:
    legacy = _smoother(localization=localization)
    explicit = _smoother(localization=localization, global_parameter_names=())
    for _ in range(2):
        left = legacy.update_params_from_pred_obs(_prior(), _predicted(), jnp.ones(1))
        right = explicit.update_params_from_pred_obs(
            _prior(), _predicted(), jnp.ones(1)
        )
        xr.testing.assert_identical(left, right)
        np.testing.assert_array_equal(legacy.rng_key, explicit.rng_key)


def test_all_global_distance_needs_no_coordinates_and_equals_global_exactly() -> None:
    params = _prior()
    smoother = _smoother(
        localization=DistanceLocalization(0.01),
        global_parameter_names=list(params.data_vars),
    )
    global_smoother = _smoother(global_parameter_names=list(params.data_vars))
    result = smoother.update_params_from_pred_obs(params, _predicted(), jnp.ones(1))
    expected = global_smoother.update_params_from_pred_obs(
        params, _predicted(), jnp.ones(1)
    )
    xr.testing.assert_identical(result, expected)
    np.testing.assert_array_equal(smoother.rng_key, global_smoother.rng_key)


@pytest.mark.parametrize("global_names", [["missing"], ["global_bias"]])  # type: ignore[misc]
def test_distance_or_missing_metadata_rejected_before_forecast(
    global_names: list[str],
) -> None:
    smoother = _smoother(
        localization=DistanceLocalization(1.0), global_parameter_names=global_names
    )
    # The forward double has no run_ensemble: any forecast before validation fails.
    with pytest.raises(ValueError, match="absent|every parameter"):
        smoother(params=_prior(), observations=jnp.ones(1))


@pytest.mark.parametrize("names", ["global_bias", [""], ["x", "x"], [1]])  # type: ignore[misc]
def test_invalid_global_declaration(names: Any) -> None:
    with pytest.raises(ValueError, match="global_parameter_names"):
        _smoother(global_parameter_names=names)


@pytest.mark.parametrize(  # type: ignore[misc]
    "cls", [TimeVaryingParameterESMDA, StateAndTimeVaryingParameterESMDA]
)
def test_global_static_parameter_survives_dynamic_flattening(cls: Any) -> None:
    params = _prior()
    params["local_bias"] = xr.DataArray(
        np.stack([params.local_bias.values, params.local_bias.values + 1]),
        dims=("time", "ensemble"),
        coords={"time": [0.0, 1.0], "ensemble": params.ensemble},
    )
    smoother = _smoother(
        cls, localization=_localization(True), global_parameter_names=["global_bias"]
    )
    if cls is TimeVaryingParameterESMDA:
        result = smoother.update_params_from_pred_obs(params, _predicted(), jnp.ones(1))
    else:
        state = xr.Dataset(
            {"u": (("ensemble", "time", "x"), np.zeros((8, 1, 1)))},
            coords={"ensemble": params.ensemble, "time": [0], "x": [10.0]},
        )
        smoother._observation_step = lambda **kw: _predicted().T
        _, result = smoother._one_step(params, jnp.ones(1), state)
    np.testing.assert_array_equal(result.local_bias, params.local_bias)
    assert not np.array_equal(result.global_bias, params.global_bias)
    assert result.global_bias.dims == ("ensemble",)
    assert result.global_bias.attrs == params.global_bias.attrs


def test_joint_state_localizes_while_global_parameter_receives_full_update() -> None:
    params = _prior()
    state = xr.Dataset(
        {"u": (("ensemble", "x"), params.local_bias.values[:, None])},
        coords={"ensemble": params.ensemble, "x": [10.0]},
    )
    smoother = _smoother(
        StateAndParameterESMDA,
        localization=_localization(True),
        global_parameter_names=["global_bias"],
    )
    updated_state, updated_params = smoother._augmented_state_update(
        state, params, _predicted(), jnp.ones(1), 8
    )
    reference = _smoother().update_params_from_pred_obs(
        params, _predicted(), jnp.ones(1)
    )
    np.testing.assert_array_equal(updated_state.u, state.u)
    np.testing.assert_array_equal(updated_params.local_bias, params.local_bias)
    np.testing.assert_allclose(
        updated_params.global_bias, reference.global_bias, atol=1e-6
    )


def test_global_time_array_rejected_explicitly() -> None:
    params = _prior().expand_dims(time=[0.0, 1.0])
    smoother = _smoother(
        TimeVaryingParameterESMDA, global_parameter_names=["global_bias"]
    )
    with pytest.raises(ValueError, match="must be static"):
        smoother.update_params_from_pred_obs(params, _predicted(), jnp.ones(1))


class _FilterModel:
    save_on_disk = False
    results_dir = None

    def __init__(self, donor: bool = False) -> None:
        self.donor = donor
        self.applied: list[xr.Dataset] = []

    def apply_failure_substitutions_to_params(self, params: xr.Dataset) -> xr.Dataset:
        result = params.copy(deep=True)
        if self.donor:
            for name in result.data_vars:
                values = np.array(result[name].values, copy=True)
                values[0] = values[1]
                result[name].data = values
        return result

    def apply_failure_substitutions_to_state(self, state: Any) -> Any:
        return state

    def run_ensemble(self, state: Any, params: xr.Dataset) -> xr.Dataset:
        used = self.apply_failure_substitutions_to_params(params)
        self.applied.append(used.copy(deep=True))
        return xr.Dataset(
            {"u": (("ensemble", "time", "x"), used.global_bias.values[:, None, None])},
            coords={"ensemble": used.ensemble, "time": [0.0], "x": [0.0]},
        )


def _filter(model: Any = None, **kwargs: Any) -> EnsembleKalmanFilter:
    forward_model: Any = model or _FilterModel()
    return EnsembleKalmanFilter(
        observation_operator=lambda state: state.u.isel(time=-1).values,
        forward_model=forward_model,
        C_D=jnp.asarray([0.5]),
        parameter_evolution=IdentityEvolution(),
        rng_key=jax.random.PRNGKey(12),
        **kwargs,
    )


@pytest.mark.parametrize("deterministic", [False, True])  # type: ignore[misc]
def test_filter_global_mask_and_metadata(deterministic: bool) -> None:
    params = _prior()
    state = xr.Dataset(
        {"u": (("ensemble", "x"), params.local_bias.values[:, None])},
        coords={"ensemble": params.ensemble, "x": [0.0]},
    )
    localized = _filter(
        mode="joint",
        analysis=LETKFAnalysis() if deterministic else None,
        localization=_localization(True),
        global_parameter_names=["global_bias"],
    )
    reference = _filter(
        mode="joint", analysis=ETKFAnalysis() if deterministic else None
    )
    _, expected, _ = reference._analysis_cycle(
        0, state, params, _predicted(), jnp.ones((1, 1))
    )
    updated_state, result, _ = localized._analysis_cycle(
        0, state, params, _predicted(), jnp.ones((1, 1))
    )
    assert result is not None and expected is not None
    np.testing.assert_allclose(result.global_bias, expected.global_bias, atol=1e-6)
    np.testing.assert_array_equal(result.local_bias, params.local_bias)
    np.testing.assert_array_equal(updated_state.u, state.u)
    assert result.global_bias.attrs == params.global_bias.attrs


@pytest.mark.parametrize("enabled", [False, True])  # type: ignore[misc]
def test_filter_records_accepted_forecast_params_separately(enabled: bool) -> None:
    model = _FilterModel(donor=True)
    filter_ = _filter(
        model,
        mode="parameter",
        global_parameter_names=["global_bias"] if enabled else (),
    )
    result = filter_.run(
        params=_prior(), observations=jnp.asarray([[2.0], [4.0]]), return_history=True
    )
    if not enabled:
        assert result.applied_params_history is None
        return
    assert result.applied_params_history is not None
    assert result.params_history is not None
    assert result.applied_params_history.sizes["cycle"] == 2
    assert result.params_history.sizes["cycle"] == 3
    for cycle, used in enumerate(model.applied):
        xr.testing.assert_identical(
            result.applied_params_history.isel(cycle=cycle), used
        )
    assert not np.array_equal(
        result.applied_params_history.global_bias.isel(cycle=0),
        result.params_history.global_bias.isel(cycle=0),
    )


def test_filter_default_metadata_is_value_and_rng_noop() -> None:
    params = _prior()
    legacy = _filter(mode="parameter", localization=_localization())
    explicit = _filter(
        mode="parameter", localization=_localization(), global_parameter_names=()
    )
    expected = legacy.run(
        params=params, observations=jnp.ones((2, 1)), return_history=True
    )
    actual = explicit.run(
        params=params, observations=jnp.ones((2, 1)), return_history=True
    )
    xr.testing.assert_identical(actual.params_history, expected.params_history)
    np.testing.assert_array_equal(explicit.rng_key, legacy.rng_key)
    assert actual.applied_params_history is None


def test_filter_parameter_distance_restriction_is_unchanged() -> None:
    with pytest.raises(ValueError, match="parameter-only filter"):
        _filter(
            mode="parameter",
            localization=DistanceLocalization(1.0),
            global_parameter_names=["global_bias"],
        )


def test_filter_unknown_global_name_fails_before_forecast() -> None:
    model = _FilterModel()
    filter_ = _filter(model, mode="parameter", global_parameter_names=["missing"])
    with pytest.raises(ValueError, match="absent"):
        filter_.run(params=_prior(), observations=jnp.ones((1, 1)))
    assert model.applied == []
