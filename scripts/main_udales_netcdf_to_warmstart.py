import os
import pathlib

import xarray
from pyudales.rollout_forward_model import RolloutForwardModel
from pyudales.utils.forward_model_utils import create_new_forward_model
from pyudales.rollout_forward_model import RolloutForwardModel
from pyudales.utils.warm_start_utils import generate_warmstart_file
from pyudales.utils.namoptions_utils import NamoptionsFile
from animation import animate_state
import logging
logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# Random seed
SEED = 42

# Directory settings
# MATLAB_BIN = "/Applications/MATLAB_R2025b.app/bin/matlab"
MATLAB_BIN = "/opt/sw/matlab-2023b/bin/matlab"
CASE_DIR = "examples/udales/experiments/xie_and_castro"
EXPERIMENT_NAME = "999"
RESULTS_DIR = pathlib.Path(".temp/udales")

FIGURES_DIR = "figures"
os.makedirs(FIGURES_DIR, exist_ok=True)

ENSEMBLE_SIZE = 4

# Compute ressources
NCPU_PER_PROCESS = 4
NUM_PARALLEL_PROCESSES = 2

# Forward model settings
FIXED_INPUT = {
    "save_only_last_timestep": False,
    "output_frequency": 2.0,
    "ncpu": NCPU_PER_PROCESS,
    "matlab_bin": MATLAB_BIN,
    "case_dir": CASE_DIR,
    "experiment_name": EXPERIMENT_NAME,
    "verbose": False,
}


def main() -> None:

    forward_model = RolloutForwardModel(**FIXED_INPUT)  # type: ignore[arg-type]
    params = xarray.Dataset(
        data_vars={
            "inflow_angle": -45,
            "velocity_magnitude": 3,
            "pressure_gradient_magnitude": 0.0041912,
        },
    )
    forward_model.run_preprocessing(python_or_matlab="python")
    state = forward_model(params=params)

    # At this point, warmstart files have been generated in output_dir/experiment_name
    # and are ready to be modified

    # Select last timestep for the warm start
    state_warm_start = state.isel(time=slice(-1, None))
    
    # Modify the state (example: add 1.0 to all values)
    state_warm_start = state_warm_start + 1.0

    # The forward_model.run_single will automatically detect existing warmstart files
    # and modify them with the new state values
    state_new = forward_model.run_single(state=state_warm_start)

    state = xarray.concat([state, state_new], dim="time")

    animate_state(
        state=state,
        output_path=pathlib.Path("figures/udales_animation.mp4"),
        z_level=0,
        vmin={"u": -3.0, "v": -2.0, "w": -2.0, "pres": 0.0, "vel_magnitude": 0.0},
        vmax={"u": 3.0, "v": 2.0, "w": 2.0, "pres": 1.0, "vel_magnitude": 3.0},
    )




if __name__ == "__main__":
    
    main()
