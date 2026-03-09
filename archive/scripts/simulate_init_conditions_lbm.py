import os
import pathlib
import pdb
import shutil

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import xarray

# from pylbm.compile_program import compile_lbm
from pylbm.forward_model import ForwardModel

NUM_SIMULATIONS = 500


def main() -> None:
    if os.path.exists(".temp"):
        shutil.rmtree(".temp")

    stl_path = pathlib.Path("examples/lbm/experiments/xie_castro_2008_STL.stl")
    # stl_path = pathlib.Path("examples/lbm/experiments/geom.STL")

    forward_model = ForwardModel(
        stl_path=stl_path,
        nx=80,
        ny=80,
        nz=8,
        simulation_time=3000 * 0.0538,
        bounds=((0, 80), (0, 80), (0, 40)),
        output_frequency=3000 * 0.0538,
        cuda=True,
        verbose=False,
    )
    forward_model.compile()
    random_key = jax.random.PRNGKey(42)

    random_key, subkey = jax.random.split(random_key)
    inflow_angle = jax.random.normal(subkey, (NUM_SIMULATIONS,)) * 8

    random_key, subkey = jax.random.split(random_key)
    velocity_magnitude = jax.random.normal(subkey, (NUM_SIMULATIONS,)) * 2 + 8
    velocity_magnitude = jnp.maximum(velocity_magnitude, 0.1)
    params = xarray.Dataset(
        data_vars={
            "inflow_angle": ("ensemble", inflow_angle),
            "velocity_magnitude": ("ensemble", velocity_magnitude),
        },
    )
    params.to_netcdf(f"esmda_init_conditions/lbm/params.nc")
    for i in range(NUM_SIMULATIONS):
        print(f"SIMULATION {i} OF {NUM_SIMULATIONS}")
        state = forward_model(params=params.isel(ensemble=i))

        state = state.isel(time=-1)  # type: ignore[union-attr]

        state.to_netcdf(f"esmda_init_conditions/lbm/state_{i}.nc")


if __name__ == "__main__":
    main()
