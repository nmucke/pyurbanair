import os
import pathlib
import pdb
import shutil

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import pylbm
import xarray
from animation import animate_ensemble_state, animate_state

# from pylbm.compile_program import compile_lbm
from pylbm.forward_model import ForwardModel
from pylbm.stl_to_lbm import stl_to_lbm_geometry

NUM_SIMULATIONS = 500


def main() -> None:
    if os.path.exists(".temp"):
        shutil.rmtree(".temp")

    stl_path = pathlib.Path("examples/lbm/experiments/xie_castro_2008_STL.stl")
    # stl_path = pathlib.Path("examples/lbm/experiments/geom.STL")

    forward_model = ForwardModel(
        stl_path=stl_path,
        nx=120,
        ny=120,
        nz=8,
        num_timesteps=500,
        bounds=((0, 160), (0, 160), (0, 40)),
        output_frequency=1000,
    )
    random_key = jax.random.PRNGKey(42)

    random_key, subkey = jax.random.split(random_key)
    inflow_angle = jax.random.normal(subkey, (NUM_SIMULATIONS,)) * 8

    random_key, subkey = jax.random.split(random_key)
    velocity_magnitude = jax.random.normal(subkey, (NUM_SIMULATIONS,)) * 2 + 10
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
        state.to_netcdf(f"esmda_init_conditions/lbm/state_{i}.nc")


if __name__ == "__main__":
    main()
