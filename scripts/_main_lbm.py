import pathlib
import pdb

import matplotlib.pyplot as plt
import numpy as np
import pylbm
import xarray

# from pylbm.compile_program import compile_lbm
from pylbm.forward_model import ForwardModel
from pylbm.stl_to_lbm import stl_to_lbm_geometry


def main() -> None:
    stl_path = pathlib.Path("examples/lbm/experiments/xie_castro_2008_STL.stl")
    # stl_path = pathlib.Path("examples/lbm/experiments/geom.STL")

    forward_model = ForwardModel(
        stl_path=stl_path,
        nx=128,
        ny=128,
        nz=8,
        num_timesteps=500,
        bounds=((0, 160), (0, 160), (0, 40)),
    )
    # inflow_angle = np.array([1, 10, 20])
    # velocity_magnitude = np.array([3, 5, 7])
    # params = xarray.Dataset(
    #     data_vars={
    #         "inflow_angle": ("ensemble", inflow_angle),
    #         "velocity_magnitude": ("ensemble", velocity_magnitude),
    #     },
    #     coords={"ensemble": np.arange(len(inflow_angle))},
    # )
    params = xarray.Dataset(
        data_vars={
            "inflow_angle": 15,
            "velocity_magnitude": 3,
        },
    )
    state = forward_model(params=params)
    # state = xarray.load_dataset(".temp/lbm/out005000.nc")

    vel_magnitude = np.sqrt(state.u.values**2 + state.v.values**2 + state.w.values**2)
    print(vel_magnitude.shape)
    print(vel_magnitude.min(), vel_magnitude.max())
    plt.figure()
    plt.imshow(vel_magnitude[0, 1, :, :])
    plt.colorbar()
    plt.savefig("figures/u.png")
    plt.show()


if __name__ == "__main__":
    main()
