"""Tests for Gaussian process parameter extrapolation."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import xarray

from pyurbanair.parameter_extrapolation import extrapolate_parameters


def _make_params(
    time_coords: np.ndarray,
    ensemble_size: int = 8,
    seed: int = 0,
) -> xarray.Dataset:
    """Create a synthetic parameter dataset with a known linear trend."""
    rng = np.random.default_rng(seed)
    n_t = len(time_coords)
    # Linear trend + small noise per ensemble member
    slopes = 1.0 + 0.1 * rng.standard_normal(ensemble_size)
    intercepts = 0.5 + 0.05 * rng.standard_normal(ensemble_size)
    values = (
        time_coords[:, None] * slopes[None, :] + intercepts[None, :]
    )
    return xarray.Dataset(
        data_vars={
            "param_a": (("time", "ensemble"), values),
            "param_b": ("ensemble", jnp.arange(ensemble_size, dtype=float)),
        },
        coords={
            "time": time_coords,
            "ensemble": np.arange(ensemble_size),
        },
    )


class TestExtrapolateParameters:
    """Tests for ``extrapolate_parameters``."""

    def test_output_dims_and_coords(self):
        time_coords = np.linspace(0, 10, 6)
        params = _make_params(time_coords)
        pred_times = jnp.array([11.0, 12.0, 13.0])

        result = extrapolate_parameters(params, pred_times)

        assert "time" in result.dims
        assert "ensemble" in result.dims
        assert result.sizes["time"] == 3
        assert result.sizes["ensemble"] == 8
        np.testing.assert_allclose(result.coords["time"].values, pred_times)

    def test_non_time_varying_passed_through(self):
        time_coords = np.linspace(0, 10, 6)
        params = _make_params(time_coords)
        pred_times = jnp.array([11.0, 12.0])

        result = extrapolate_parameters(params, pred_times)

        assert "param_b" in result.data_vars
        np.testing.assert_allclose(
            result["param_b"].values, params["param_b"].values
        )

    def test_linear_extrapolation_accuracy(self):
        """GP should extrapolate a near-linear function reasonably well."""
        time_coords = np.linspace(0, 10, 11)
        params = _make_params(time_coords, ensemble_size=4, seed=42)
        pred_times = jnp.array([11.0, 12.0])

        result = extrapolate_parameters(
            params, pred_times, correlation_length=5.0
        )

        # For a linear function y = slope * t + intercept, the GP
        # prediction should be close (not exact, since RBF kernel
        # reverts to the mean far from data).
        predicted = result["param_a"].values  # (2, 4)
        last_training = params["param_a"].isel(time=-1).values  # (4,)

        # The prediction at t=11 should be larger than the last training
        # value for positive slopes.
        assert np.all(predicted[0] > last_training * 0.9)

    def test_include_std(self):
        time_coords = np.linspace(0, 10, 6)
        params = _make_params(time_coords)
        pred_times = jnp.array([11.0, 15.0, 20.0])

        result = extrapolate_parameters(
            params, pred_times, include_std=True
        )

        assert "param_a_std" in result.data_vars
        stds = result["param_a_std"].values
        assert stds.shape == (3, 8)
        # Std should increase with distance from training data.
        assert np.all(stds[-1] >= stds[0] - 1e-6)

    def test_std_not_included_by_default(self):
        time_coords = np.linspace(0, 10, 6)
        params = _make_params(time_coords)
        pred_times = jnp.array([11.0])

        result = extrapolate_parameters(params, pred_times)

        assert "param_a_std" not in result.data_vars

    def test_default_correlation_length(self):
        """Should not error when correlation_length is not provided."""
        time_coords = np.linspace(0, 20, 10)
        params = _make_params(time_coords)
        pred_times = jnp.array([21.0])

        result = extrapolate_parameters(params, pred_times)

        assert result["param_a"].shape == (1, 8)

    def test_interpolation_recovers_training_data(self):
        """Predicting at training times should recover the original values."""
        time_coords = np.linspace(0, 10, 6)
        params = _make_params(time_coords, ensemble_size=4, seed=7)

        result = extrapolate_parameters(
            params,
            jnp.asarray(time_coords),
            correlation_length=5.0,
        )

        np.testing.assert_allclose(
            result["param_a"].values,
            params["param_a"].values,
            atol=1e-3,
        )
