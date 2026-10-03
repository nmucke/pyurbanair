"""Small scalar oracle for observation-noise aggregation and tempering.

Run with ``pixi run -e dev python scripts/examples/observation_likelihood.py``.
The truth measurement contains instrument noise only. Representation
uncertainty is added to the assimilation likelihood, and independent frame
errors are averaged with variance ``sigma**2 / m``. The example resolves covariance with
the production observation-error API and applies the production ETKF update to
an ensemble with exact unit sample variance.
"""

from __future__ import annotations

from math import sqrt
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import xarray as xr
from data_assimilation.filtering.etkf import ETKFAnalysis
from data_assimilation.observation_error import ObservationErrorSpec
from data_assimilation.observation_operator import AggregateObservations


def gaussian_update(
    prior_mean: float,
    prior_variance: float,
    observation: float,
    observation_variance: float,
) -> tuple[float, float]:
    """Exact scalar linear-Gaussian posterior for H=1."""
    posterior_variance = 1.0 / (1.0 / prior_variance + 1.0 / observation_variance)
    posterior_mean = posterior_variance * (
        prior_mean / prior_variance + observation / observation_variance
    )
    return posterior_mean, posterior_variance


def main() -> None:
    prior_mean, prior_variance = 0.0, 1.0
    raw_instrument_std = 0.5
    raw_representation_std = 0.2
    num_independent_frames = 4

    # Fixed illustrative noisy measurements; only instrument noise is
    # generated in the synthetic observation product.
    raw_measurements = (1.2, 1.4, 1.6, 1.8)
    observation = sum(raw_measurements) / num_independent_frames

    observations = (
        xr.DataArray(
            list(raw_measurements),
            dims=("time",),
        )
        .expand_dims(obs=[0])
        .transpose("time", "obs")
        .assign_coords(time=range(num_independent_frames))
    )
    # A minimal labelled operator is enough for this scalar oracle.
    operator = SimpleNamespace(
        num_sensors=1,
        obs_states=("u",),
        obs_z=(2.0,),
    )
    spec = ObservationErrorSpec(
        instrument_std=raw_instrument_std,
        representation_std=raw_representation_std,
    )
    resolved = spec.resolve(
        observations,
        operator,
        AggregateObservations(interval_seconds=num_independent_frames),
    )
    instrument_variance = float(resolved.instrument_variance[0, 0])
    representation_variance = float(resolved.representation_variance[0, 0])
    physical_variance = float(resolved.variance[0, 0])

    corrected = gaussian_update(
        prior_mean, prior_variance, observation, physical_variance
    )
    retained = ObservationErrorSpec(
        instrument_std=raw_instrument_std,
        representation_std=raw_representation_std,
        aggregation="none",
    ).resolve(
        observations,
        operator,
        AggregateObservations(interval_seconds=num_independent_frames),
    )
    retained_variance = float(retained.variance[0, 0])
    unscaled = gaussian_update(
        prior_mean, prior_variance, observation, retained_variance
    )

    alpha = 3.0
    tempered_step = gaussian_update(
        prior_mean, prior_variance, observation, alpha * physical_variance
    )

    # A four-member ensemble with exactly zero sample mean and unit sample
    # variance lets the production ETKF transform be compared to the oracle.
    a = sqrt(3.0) / 2.0
    prior_ensemble = jnp.asarray([[-a, -a, a, a]])
    pred_obs = prior_ensemble.copy()
    analysis = ETKFAnalysis()
    etkf_corrected = np.asarray(
        analysis(
            prior_ensemble,
            pred_obs,
            jnp.asarray([observation]),
            jnp.asarray([physical_variance]),
            jax.random.PRNGKey(0),
        )
    )[0]
    etkf_unscaled = np.asarray(
        analysis(
            prior_ensemble,
            pred_obs,
            jnp.asarray([observation]),
            jnp.asarray([retained_variance]),
            jax.random.PRNGKey(0),
        )
    )[0]
    etkf_corrected_stats = (
        float(np.mean(etkf_corrected)),
        float(np.var(etkf_corrected, ddof=1)),
    )
    etkf_unscaled_stats = (
        float(np.mean(etkf_unscaled)),
        float(np.var(etkf_unscaled, ddof=1)),
    )
    assert np.allclose(etkf_corrected_stats, corrected, atol=2e-6)
    assert np.allclose(etkf_unscaled_stats, unscaled, atol=2e-6)

    print(f"mean observation: {observation:.3f}")
    print(f"instrument variance after averaging: {instrument_variance:.4f}")
    print(f"representation variance in likelihood: {representation_variance:.4f}")
    print(f"physical likelihood variance: {physical_variance:.4f}")
    print(f"aggregation=none likelihood variance: {retained_variance:.4f}")
    print(
        "corrected posterior: " f"mean={corrected[0]:.4f}, std={sqrt(corrected[1]):.4f}"
    )
    print(
        "aggregation=none posterior: "
        f"mean={unscaled[0]:.4f}, std={sqrt(unscaled[1]):.4f}"
    )
    print(
        f"one alpha={alpha:g} tempered step: "
        f"mean={tempered_step[0]:.4f}, std={sqrt(tempered_step[1]):.4f}"
    )
    print(
        "production ETKF corrected: "
        f"mean={etkf_corrected_stats[0]:.4f}, "
        f"std={sqrt(etkf_corrected_stats[1]):.4f}"
    )
    print(
        "production ETKF unscaled: "
        f"mean={etkf_unscaled_stats[0]:.4f}, "
        f"std={sqrt(etkf_unscaled_stats[1]):.4f}"
    )
    print("physical likelihood variance remains 0.0725 under tempering")


if __name__ == "__main__":
    main()
