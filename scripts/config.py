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
    create_time_varying_parameter_ensemble,
    create_time_varying_true_params,
    create_true_params,
    model_args,
    prepare_forward_model,
    sample_smooth_ensemble,
    solver_name,
)

BASE_RESULTS_DIR = pathlib.Path(".temp/scripts")

DOMAIN = {
    "nx": 100,
    "ny": 80,
    "nz": 8,
    "bounds": ((-20.0, 80.0), (0.0, 80.0), (0.0, 40.0)),
}

TIME = {
    "simulation_time": 10.0,  # 3000 * 0.0538,
    "output_frequency": 1.0,  # 3000 * 0.0538,
    "spinup_time": 5.0,
}

LBM_ARGS = {
    "stl_path": "examples/lbm/experiments/xie_castro_2008_STL.stl",
    "experiment_name": "runcase",
    "cuda": False,
    "verbose": False,
    "boundary_condition": "inflow_outflow",
    "compile": True,
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
    "x_points": [20.0, 20.0, 40.0, 40.0, 60.0],
    "y_points": [20.0, 60.0, 10.0, 30.0, 60.0],
    "z_points": [1.0, 1.0, 1.0, 1.0, 1.0],
    "states": ["u", "v", "w"],
    "temporal_mode": "intervals",
    "interval_size": 5,
    "aggregation_mode": "mean",
}

ESMDA = {
    "num_steps": 1,
    "num_assimilation_windows": 3,
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

TIME_VARYING_PARAMS = {
    "num_time_points": 10,
    "prior_correlation_length": 5.0,  # seconds — controls smoothness of GP prior
    "truth_correlation_length": 10.0,  # seconds — different from prior to avoid inverse crime
    "inflow_angle_start": -20.0,
    "inflow_angle_end": 20.0,
    "inflow_angle_sigmoid_center": 0.5,
    "inflow_angle_sigmoid_width": 0.1,
    "velocity_magnitude_start": 3.0,
    "velocity_magnitude_end": 5.0,
    "velocity_magnitude_sigmoid_center": 0.5,
    "velocity_magnitude_sigmoid_width": 0.1,
}
