import os
import pathlib
import pdb
import shutil

import matplotlib.pyplot as plt
import numpy as np
import pylbm
import xarray
from animation import animate_3d, animate_ensemble_state, animate_state

# from pylbm.compile_program import compile_lbm
from pylbm.forward_model import ForwardModel
from pylbm.stl_to_lbm import stl_to_lbm_geometry


def main() -> None:

    if os.path.exists(".temp"):
        shutil.rmtree(".temp")

    stl_path = pathlib.Path("examples/lbm/experiments/xie_castro_2008_STL.stl")
    # stl_path = pathlib.Path("examples/lbm/experiments/geom.STL")

    forward_model = ForwardModel(
        stl_path=stl_path,
        nx=120,
        ny=120,
        nz=16,
        simulation_time=1500 * 0.0538,
        bounds=((0, 160), (0, 160), (0, 16)),
        output_frequency=10 * 0.0538,
        cuda=False,
        results_dir=pathlib.Path(".temp/lbm"),
    )
    forward_model.compile()
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
            "inflow_angle": 10,
            "velocity_magnitude": 10,
        },
    )
    state = forward_model(params=params)
    if state is None:
        raise RuntimeError("Expected in-memory state from forward_model run.")

    vel_magnitude = np.sqrt(state.u.values**2 + state.v.values**2 + state.w.values**2)
    state = state.assign(vel_magnitude=(("time", "z", "y", "x"), vel_magnitude))

    # Remove the blanking data_var from the xarray.Dataset if present
    if "blanking" in state.data_vars:
        state = state.drop_vars("blanking")

    # if "rho" in state.data_vars:
    #     state = state.drop_vars("rho")

    animate_3d(
        state=state,
        output_path=pathlib.Path("figures/lbm_animation_3d.mp4"),
        variable="vel_magnitude",
    )

    animate_state(
        state=state,
        output_path=pathlib.Path("figures/lbm_animation.mp4"),
        z_level=1,
        # vmin={"u": -3.0, "v": -2.0, "w": -2.0, "pres": 0.0, "vel_magnitude": 0.0},
        # vmax={"u": 3.0, "v": 2.0, "w": 2.0, "pres": 1.0, "vel_magnitude": 3.0},
    )

    vel_magnitude = state.v.values
    print(vel_magnitude.shape)
    print(vel_magnitude.min(), vel_magnitude.max())
    plt.figure()
    plt.imshow(vel_magnitude[0, 1, :, :])
    # plt.imshow(state.blanking.values[0, 1, :, :])
    plt.colorbar()
    plt.savefig("figures/vel_magnitude_lbm.png")
    plt.close()
    # plt.show()


if __name__ == "__main__":
    main()
