"""Tests for deterministic sine/cosine dynamic parameter profiles."""

import jax
import numpy as np

from pyurbanair.dynamic_parameters.harmonic import HarmonicParameterModel


def test_harmonic_profiles_are_shared_and_continuous_across_rollout() -> None:
    model = HarmonicParameterModel(
        profiles={
            "inflow_angle": {
                "waveform": "sine",
                "offset": 3.0,
                "amplitude": 2.0,
                "frequency": 0.1,
            },
            "velocity_magnitude": {
                "waveform": "cosine",
                "offset": 7.5,
                "amplitude": 1.0,
                "frequency": 0.1,
                "min": 7.0,
            },
        },
        simulation_time=10.0,
        seconds_per_knot=5.0,
    )

    first = model.sample(ensemble_size=3)
    second = model.extrapolate(first, model.time_coords, jax.random.PRNGKey(0))

    np.testing.assert_allclose(first["inflow_angle"].isel(ensemble=0), [3.0, 3.0, 3.0])
    np.testing.assert_allclose(
        first["inflow_angle"].isel(ensemble=0), first["inflow_angle"].isel(ensemble=2)
    )
    np.testing.assert_allclose(
        first["inflow_angle"].isel(time=-1), second["inflow_angle"].isel(time=0)
    )
    assert float(first["velocity_magnitude"].min()) >= 7.0
