import pathlib
import pdb

import matplotlib.pyplot as plt
import numpy as np
import pylbm
import xarray
from animation import animate_state
from pylbm.rollout_forward_model import RolloutForwardModel


def main() -> None:
    stl_path = pathlib.Path("examples/lbm/experiments/xie_castro_2008_STL.stl")
    # stl_path = pathlib.Path("examples/lbm/experiments/geom.STL")

    forward_model = RolloutForwardModel(
        stl_path=stl_path,
        nx=140,
        ny=120,
        nz=8,
        num_timesteps=100,
        bounds=((-40, 160), (0, 160), (0, 40)),
        output_frequency=10,
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
    for _ in range(3):
        state = forward_model.run_single(sim_name="my_sim")
        states.append(state)

    pdb.set_trace()
    # state = xarray.load_dataset(forward_model.results_dir / "my_sim.nc")
    state = forward_model.get_states()
    pdb.set_trace()
    # state = xarray.concat(states, dim="time", join="override")
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
