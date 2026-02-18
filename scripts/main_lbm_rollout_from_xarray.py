import os
import pathlib
import pdb
import shutil

import matplotlib.pyplot as plt
import numpy as np
import pylbm
import xarray
from animation import animate_state
from pylbm.rollout_forward_model import RolloutForwardModel


def main() -> None:

    if os.path.exists(".temp"):
        shutil.rmtree(".temp")

    stl_path = pathlib.Path("examples/lbm/experiments/xie_castro_2008_STL.stl")
    # stl_path = pathlib.Path("examples/lbm/experiments/geom.STL")

    forward_model = RolloutForwardModel(
        stl_path=stl_path,
        nx=120,
        ny=120,
        nz=8,
        num_timesteps=100,
        bounds=((0, 160), (0, 160), (0, 40)),
        output_frequency=2,
        # results_dir=pathlib.Path(".temp/results"),
    )
    params = xarray.Dataset(
        data_vars={
            "inflow_angle": -40,
            "velocity_magnitude": 10,
        },
    )
    # save_on_disk is applied automatically when results_dir is passed above

    # Run 3 rollout steps (cold → warm → warm)
    states = []
    for i in range(3):
        if i == 0:
            state_new = None

        state = forward_model.run_single(state=state_new, params=params)
        states.append(state.copy())  # type: ignore[union-attr]

        if i == 0:
            state_new = xarray.Dataset(
                data_vars={
                    "rho": (("time", "z", "y", "x"), state.rho.values),  # type: ignore[union-attr]
                    "u": (("time", "z", "y", "x"), state.u.values * 1.3),  # type: ignore[union-attr]
                    "v": (("time", "z", "y", "x"), state.v.values * 1.3),  # type: ignore[union-attr]
                    "w": (("time", "z", "y", "x"), state.w.values * 1.3),  # type: ignore[union-attr]
                    "blanking": (("time", "z", "y", "x"), state.blanking.values),  # type: ignore[union-attr]
                },
            )

        if i == 1:
            state_new = xarray.Dataset(
                data_vars={
                    "rho": (("time", "z", "y", "x"), state.rho.values),  # type: ignore[union-attr]
                    "u": (("time", "z", "y", "x"), state.u.values * 0.3),  # type: ignore[union-attr]
                    "v": (("time", "z", "y", "x"), state.v.values * 0.3),  # type: ignore[union-attr]
                    "w": (("time", "z", "y", "x"), state.w.values * 0.3),  # type: ignore[union-attr]
                    "blanking": (("time", "z", "y", "x"), state.blanking.values),  # type: ignore[union-attr]
                },
            )
        # params = xarray.Dataset(
        #     data_vars={
        #         "inflow_angle": 40,
        #         "velocity_magnitude": 10,
        #     },
        # )

    # state = xarray.load_dataset(forward_model.results_dir / "my_sim.nc")
    # state = forward_model.get_states()

    state = xarray.concat(states, dim="time", join="override")
    vel_magnitude = np.sqrt(state.u.values**2 + state.v.values**2 + state.w.values**2)
    state = state.assign(vel_magnitude=(("time", "z", "y", "x"), vel_magnitude))

    # Remove the blanking data_var from the xarray.Dataset if present
    if "blanking" in state.data_vars:
        state = state.drop_vars("blanking")

    if "rho" in state.data_vars:
        state = state.drop_vars("rho")

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
    plt.show()


if __name__ == "__main__":
    main()
