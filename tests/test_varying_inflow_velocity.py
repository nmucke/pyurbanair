"""Test that pylbm produces consistent output timesteps across varying inflow velocities.

The number of output timesteps should be determined solely by simulation_time
and output_frequency, regardless of the inflow velocity (which affects C_u and
therefore the internal timestep size).
"""

from typing import Any

import pytest
import xarray
from hydra.utils import instantiate

from pyurbanair.config.hydra_helpers import clean_outputs


@pytest.fixture(scope="module")  # type: ignore[misc]
def pylbm_cfg(compose_module_cfg: Any) -> Any:
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


VELOCITIES = [2.0, 5.0, 10.0]

pytestmark = pytest.mark.integration


def test_all_velocities_without_cleaning(pylbm_cfg: Any, pylbm_model: Any) -> None:
    """Run velocities sequentially WITHOUT cleaning between runs.

    This reproduces the real-world scenario where _clean_output is a no-op
    and leftover files from a previous run with different iout may be
    collected by the next run.
    """
    clean_outputs(model_name="pylbm", forward_model=pylbm_model)

    time_dims: dict[float, int] = {}
    output_files_info: dict[float, list[str]] = {}

    for velocity in VELOCITIES:
        params = xarray.Dataset(
            data_vars={
                "inflow_angle": 0.0,
                "velocity_magnitude": velocity,
            }
        )

        state = pylbm_model.run_single(params=params)
        time_dims[velocity] = state.sizes["time"]

        # List remaining output files
        output_dir = pylbm_model.dirs.output_dir
        nc_files = sorted(output_dir.glob("out_0000_F*.nc"))
        output_files_info[velocity] = [f.name for f in nc_files]

        print(
            f"velocity={velocity}: time_dim={time_dims[velocity]}, "
            f"C_u={pylbm_model.C_u}, "
            f"iout={pylbm_model.output_frequency_timesteps}, "
            f"nt1={pylbm_model.num_timesteps}, "
            f"files={[f.name for f in nc_files]}"
        )

    expected = round(pylbm_cfg.time.simulation_time / pylbm_cfg.time.output_frequency)
    assert set(time_dims.values()) == {expected}, (
        f"Inconsistent time dimensions across velocities (no cleaning): {time_dims}\n"
        f"Output files: {output_files_info}"
    )
