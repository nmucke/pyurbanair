"""Test that pylbm produces consistent output timesteps across varying inflow velocities.

The number of output timesteps should be determined solely by simulation_time
and output_frequency, regardless of the inflow velocity (which affects C_u and
therefore the internal timestep size).
"""

import pathlib
import sys

import numpy as np
import pytest
import xarray

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import tests.config as tests_config

sys.modules["scripts.config"] = tests_config

from pyurbanair.utils.config_utils import (
    clean_forward_model_outputs,
    create_forward_model,
    prepare_forward_model,
)


@pytest.fixture(scope="module")
def pylbm_model():
    """Create and compile a pylbm forward model once for all tests."""
    model = create_forward_model(model_name="pylbm")
    prepare_forward_model(model_name="pylbm", forward_model=model)
    return model


VELOCITIES = [2.0, 3.0, 5.0, 7.0, 10.0]


@pytest.mark.parametrize("velocity", VELOCITIES)
def test_output_timesteps_consistent(pylbm_model, velocity: float) -> None:
    """Each velocity should produce the same number of output timesteps."""
    clean_forward_model_outputs(model_name="pylbm", forward_model=pylbm_model)

    params = xarray.Dataset(
        data_vars={
            "inflow_angle": 0.0,
            "velocity_magnitude": velocity,
        }
    )

    state = pylbm_model.run_single(params=params)

    expected_num_outputs = round(
        tests_config.TIME["simulation_time"] / tests_config.TIME["output_frequency"]
    )

    actual_time_steps = state.sizes["time"]
    print(
        f"velocity={velocity}, C_u={pylbm_model.C_u}, "
        f"spt={pylbm_model.seconds_per_timestep:.6f}, "
        f"iout={pylbm_model.output_frequency_timesteps}, "
        f"num_timesteps={pylbm_model.num_timesteps}, "
        f"time_dim={actual_time_steps}, expected={expected_num_outputs}"
    )

    assert actual_time_steps == expected_num_outputs, (
        f"velocity={velocity}: got {actual_time_steps} time steps, "
        f"expected {expected_num_outputs} "
        f"(C_u={pylbm_model.C_u}, iout={pylbm_model.output_frequency_timesteps}, "
        f"num_timesteps={pylbm_model.num_timesteps})"
    )


def test_all_velocities_same_time_dim(pylbm_model) -> None:
    """Run all velocities and verify they all produce the same time dimension."""
    time_dims: dict[float, int] = {}

    for velocity in VELOCITIES:
        clean_forward_model_outputs(model_name="pylbm", forward_model=pylbm_model)

        params = xarray.Dataset(
            data_vars={
                "inflow_angle": 0.0,
                "velocity_magnitude": velocity,
            }
        )

        state = pylbm_model.run_single(params=params)
        time_dims[velocity] = state.sizes["time"]

        print(
            f"velocity={velocity}: time_dim={time_dims[velocity]}, "
            f"C_u={pylbm_model.C_u}, "
            f"iout={pylbm_model.output_frequency_timesteps}, "
            f"num_timesteps={pylbm_model.num_timesteps}"
        )

    unique_dims = set(time_dims.values())
    assert (
        len(unique_dims) == 1
    ), f"Inconsistent time dimensions across velocities: {time_dims}"


def test_all_velocities_without_cleaning(pylbm_model) -> None:
    """Run velocities sequentially WITHOUT cleaning between runs.

    This reproduces the real-world scenario where _clean_output is a no-op
    and leftover files from a previous run with different iout may be
    collected by the next run.
    """
    clean_forward_model_outputs(model_name="pylbm", forward_model=pylbm_model)

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

    unique_dims = set(time_dims.values())
    assert len(unique_dims) == 1, (
        f"Inconsistent time dimensions across velocities (no cleaning): {time_dims}\n"
        f"Output files: {output_files_info}"
    )
