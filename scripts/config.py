import pathlib

from pyurbanair.utils.config_utils import (
    clean_forward_model_outputs,
    clean_forward_model_restarts,
    create_C_D,
    create_ensemble_forward_model,
    create_forward_model,
    create_initial_state_ensemble,
    create_observation_operator,
    create_observation_points,
    create_parameter_ensemble,
    create_rollout_forward_model,
    create_true_params,
    model_args,
    prepare_forward_model,
    solver_name,
)

BASE_RESULTS_DIR = pathlib.Path(".temp/scripts")

DOMAIN = {
    "nx": 120,
    "ny": 80,
    "nz": 8,
    "bounds": ((-40.0, 80.0), (0.0, 80.0), (0.0, 40.0)),
}

TIME = {
    "simulation_time": 50.0,  # 3000 * 0.0538,
    "output_frequency": 0.5,  # 3000 * 0.0538,
    "spinup_time": 20.0,
}

LBM_ARGS = {
    "stl_path": "examples/lbm/experiments/xie_castro_2008_STL.stl",
    "experiment_name": "runcase",
    "cuda": False,
    "verbose": False,
    "boundary_condition": "inflow_outflow",
}

UDALES_ARGS = {
    "case_dir": "examples/udales/experiments/xie_and_castro",
    "experiment_name": "999",
    "matlab_bin": "/opt/sw/matlab-2023b/bin/matlab",
    "ncpu": 1,
    "save_only_last_timestep": False,
    "verbose": False,
    "boundary_condition": "inflow_outflow",
}

ENSEMBLE = {
    "ensemble_size": 4,
    "num_parallel_processes": 1,
    "num_cpus_per_process": 1,
}

OBS = {
    "x_min": 10.0,
    "x_max": 30.0,
    "y_min": 10.0,
    "y_max": 30.0,
    "n_per_axis": 4,
    "z": 2.0,
    "states": ["u", "v", "w"],
    "temporal_mode": "mean",
}

ESMDA = {
    "num_steps": 2,
    "num_assimilation_windows": 2,
    "seed": 42,
    "obs_error_std": 0.1,
    "init_conditions_dir": "esmda_init_conditions",
    "true_sim_id": 0,
}

TRUE_PARAMS = {
    "inflow_angle": 10.0,
    "velocity_magnitude": 3.0,
    "pressure_gradient_magnitude": 0.0041912,
}

PARAM_PRIORS = {
    "inflow_angle_mean": 0.0,
    "inflow_angle_std": 8.0,
    "velocity_mean": 3.0,
    "velocity_std": 1.0,
    "pressure_mean": 0.0041912,
    "pressure_std": 0.001,
}
