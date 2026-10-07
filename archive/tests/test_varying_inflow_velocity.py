"""Test that pylbm produces consistent output timesteps across varying inflow velocities.

The number of output timesteps should be determined solely by simulation_time
and output_frequency, regardless of the inflow velocity (which affects C_u and
therefore the internal timestep size).
"""

from collections.abc import Callable
from typing import Any

import pytest
import xarray
from hydra.utils import instantiate

from pyurbanair.config.hydra_helpers import clean_outputs

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")  # type: ignore[misc]
def pylbm_cfg(compose_module_cfg: Callable[..., Any]) -> Any:
    """Compose a single-model pylbm test config once for this module."""
    return compose_module_cfg(
        [
            "model=pylbm",
            "model.forward_model.cuda=false",
        ]
    )


@pytest.fixture(scope="module")  # type: ignore[misc]
def pylbm_model(pylbm_cfg: Any) -> Any:
    """Create and compile a pylbm forward model once for all tests."""
    model = instantiate(pylbm_cfg.model.forward_model)
    instantiate(pylbm_cfg.model.prepare, forward_model=model)
    return model


def test_output_timesteps_across_velocity_changes(
    pylbm_cfg: Any, pylbm_model: Any
) -> None:
    """Changing lattice scaling must not collect stale output from the last run."""
    clean_outputs(model_name="pylbm", forward_model=pylbm_model)
    expected_num_outputs = round(
        pylbm_cfg.time.simulation_time / pylbm_cfg.time.output_frequency
    )
    reference_time: Any = None

    # Exercise both increases and decreases in C_u without external cleanup.
    # Checking each run against the expected count also subsumes the old
    # separate test that reran all velocities just to compare their counts.
    for velocity in (2.0, 10.0, 5.0):
        params = xarray.Dataset(
            data_vars={"inflow_angle": 0.0, "velocity_magnitude": velocity}
        )
        state = pylbm_model.run_single(params=params)
        assert state.sizes["time"] == expected_num_outputs, f"velocity={velocity}"
        if reference_time is None:
            reference_time = state.time.copy(deep=True)
        else:
            xarray.testing.assert_equal(state.time, reference_time)
