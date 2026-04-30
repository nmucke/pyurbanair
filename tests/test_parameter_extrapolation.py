"""Tests for the pyurbanair.parameter_time_series classes."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import xarray

from pyurbanair.parameter_time_series import (
    AR1Model,
    AR2RelaxationModel,
    GPLinearTrendModel,
    OrnsteinUhlenbeckModel,
    build_parameter_time_series,
)


EXTERNAL_PRIORS = {
    "param_a": {"mean": 0.5, "std": 1.0},
}


def _make_posterior(
    time_coords: np.ndarray,
    ensemble_size: int = 8,
    seed: int = 0,
) -> xarray.Dataset:
    """Synthetic posterior with a known per-member linear trend."""
    rng = np.random.default_rng(seed)
    slopes = 1.0 + 0.1 * rng.standard_normal(ensemble_size)
    intercepts = 0.5 + 0.05 * rng.standard_normal(ensemble_size)
    values = time_coords[:, None] * slopes[None, :] + intercepts[None, :]
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


# ---------------------------------------------------------------------------
# Common shape / contract tests parametrized over methods
# ---------------------------------------------------------------------------


@pytest.fixture(params=["gp_linear_trend", "ar1", "ornstein_uhlenbeck", "ar2_relaxation"])
def model(request):
    method_kwargs = {
        "gp_linear_trend": {"correlation_length": 5.0},
        "ar1": {"correlation_length": 5.0},
        "ornstein_uhlenbeck": {"correlation_length": 5.0},
        "ar2_relaxation": {"correlation_length": 5.0},
    }[request.param]
    return build_parameter_time_series(
        method=request.param,
        external_priors=EXTERNAL_PRIORS,
        ensemble_size=8,
        method_kwargs=method_kwargs,
    )


class TestSamplePrior:
    def test_output_dims(self, model):
        time_coords = jnp.linspace(0.0, 10.0, 6)
        ds = model.sample_prior(time_coords, jax.random.PRNGKey(0))
        assert ds.sizes["time"] == 6
        assert ds.sizes["ensemble"] == 8
        assert "param_a" in ds.data_vars

    def test_marginal_mean_close_to_prior(self, model):
        time_coords = jnp.linspace(0.0, 10.0, 50)
        ds = model.sample_prior(time_coords, jax.random.PRNGKey(0))
        ensemble_mean = ds["param_a"].mean("ensemble").values
        # Per-time-point ensemble mean should hover near the prior mean
        # for a moderate ensemble size.  Loose tolerance.
        assert np.abs(ensemble_mean.mean() - 0.5) < 1.5


class TestExtrapolate:
    def test_output_dims_and_coords(self, model):
        # Some methods carry state set up in sample_prior; call it first.
        prior_times = jnp.linspace(0.0, 10.0, 6)
        model.sample_prior(prior_times, jax.random.PRNGKey(0))

        time_coords = np.linspace(0, 10, 6)
        post = _make_posterior(time_coords)
        pred_times = jnp.array([10.0, 11.0, 12.0])

        ds = model.extrapolate(post, pred_times, jax.random.PRNGKey(1))
        assert ds.sizes["time"] == 3
        assert ds.sizes["ensemble"] == 8
        np.testing.assert_allclose(
            np.asarray(ds.coords["time"].values), np.asarray(pred_times)
        )

    def test_non_time_varying_passed_through(self, model):
        prior_times = jnp.linspace(0.0, 10.0, 6)
        model.sample_prior(prior_times, jax.random.PRNGKey(0))

        time_coords = np.linspace(0, 10, 6)
        post = _make_posterior(time_coords)
        pred_times = jnp.array([10.0, 11.0])

        ds = model.extrapolate(post, pred_times, jax.random.PRNGKey(1))
        assert "param_b" in ds.data_vars
        np.testing.assert_allclose(ds["param_b"].values, post["param_b"].values)


# ---------------------------------------------------------------------------
# Method-specific behavior
# ---------------------------------------------------------------------------


class TestGPLinearTrend:
    def test_linear_extrapolation_increases(self):
        """GP+trend should keep pushing past the last value for positive slopes."""
        time_coords = np.linspace(0, 10, 11)
        post = _make_posterior(time_coords, ensemble_size=4, seed=42)
        pred_times = jnp.array([10.0, 11.0, 12.0])

        m = GPLinearTrendModel(
            external_priors=EXTERNAL_PRIORS, ensemble_size=4,
            correlation_length=5.0,
        )
        ds = m.extrapolate(post, pred_times, jax.random.PRNGKey(0))
        predicted = ds["param_a"].values  # (3, 4)
        last_training = post["param_a"].isel(time=-1).values  # (4,)
        assert np.all(predicted[1] > last_training * 0.9)

    def test_continuity_at_first_prediction_point(self):
        """When pred_times[0] == train_times[-1], output matches per-member."""
        time_coords = np.linspace(0, 10, 6)
        post = _make_posterior(time_coords, ensemble_size=4, seed=7)
        pred_times = jnp.asarray(time_coords[-1:])

        m = GPLinearTrendModel(
            external_priors=EXTERNAL_PRIORS, ensemble_size=4,
            correlation_length=5.0,
        )
        ds = m.extrapolate(post, pred_times, jax.random.PRNGKey(0))
        np.testing.assert_allclose(
            ds["param_a"].values[0],
            post["param_a"].isel(time=-1).values,
            atol=1e-3,
        )


class TestAR2Relaxation:
    def test_state_carried_across_calls(self):
        """Two consecutive extrapolate calls should evolve the AR(2) state."""
        m = AR2RelaxationModel(
            external_priors=EXTERNAL_PRIORS, ensemble_size=8,
            correlation_length=5.0,
        )
        prior_times = jnp.linspace(0.0, 10.0, 6)
        m.sample_prior(prior_times, jax.random.PRNGKey(0))
        state_after_prior = m._state["param_a"]

        post = _make_posterior(np.linspace(0, 10, 6))
        m.extrapolate(post, jnp.linspace(10.0, 20.0, 6), jax.random.PRNGKey(1))
        state_after_extrap = m._state["param_a"]

        # The terminal (z, w) should change after another integration.
        assert not np.allclose(
            np.asarray(state_after_prior[0]),
            np.asarray(state_after_extrap[0]),
        )

    def test_relaxation_returns_to_external_at_long_lead(self):
        """Far into the next window α(t)→0, so the prior reverts to x_ext."""
        m = AR2RelaxationModel(
            external_priors=EXTERNAL_PRIORS, ensemble_size=64,
            correlation_length=1.0,  # short → α decays fast
        )
        prior_times = jnp.linspace(0.0, 1.0, 4)
        m.sample_prior(prior_times, jax.random.PRNGKey(0))

        # Posterior with mean far away from x_ext = 0.5.
        time_coords = np.linspace(0, 1.0, 4)
        post = xarray.Dataset(
            data_vars={
                "param_a": (
                    ("time", "ensemble"),
                    np.full((4, 64), 100.0),
                ),
            },
            coords={"time": time_coords, "ensemble": np.arange(64)},
        )
        pred_times = jnp.linspace(1.0, 21.0, 21)  # 20 l_corr long
        ds = m.extrapolate(post, pred_times, jax.random.PRNGKey(1))
        ensemble_mean_far = float(ds["param_a"].isel(time=-1).mean("ensemble"))
        # Should be much closer to x_ext (0.5) than to posterior mean (100).
        assert abs(ensemble_mean_far - 0.5) < 5.0

    def test_min_clip_applied(self):
        priors = {"v": {"mean": 0.5, "std": 1.0, "min": 0.1}}
        m = AR2RelaxationModel(
            external_priors=priors, ensemble_size=128, correlation_length=2.0,
        )
        ds = m.sample_prior(jnp.linspace(0.0, 10.0, 5), jax.random.PRNGKey(0))
        assert float(ds["v"].min()) >= 0.1 - 1e-6


class TestFactory:
    def test_unknown_method_raises(self):
        with pytest.raises(ValueError, match="Unknown"):
            build_parameter_time_series(
                method="not_a_method",
                external_priors=EXTERNAL_PRIORS,
                ensemble_size=4,
                method_kwargs={},
            )

    def test_missing_kwargs_uses_defaults(self):
        # GPLinearTrendModel requires correlation_length — missing should fail.
        with pytest.raises(TypeError):
            build_parameter_time_series(
                method="gp_linear_trend",
                external_priors=EXTERNAL_PRIORS,
                ensemble_size=4,
                method_kwargs={},
            )
