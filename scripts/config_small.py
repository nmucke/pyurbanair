"""Tiny smoke-test variant of :mod:`scripts.config`.

Same shape as ``config.py`` so the production scripts (which import
``from scripts import config``) work after renaming this file to
``config.py``. Values are sized so the rollout ESMDA finishes in
minutes on a small partition while still exercising every code path
(>= 2 windows, > 1 ensemble member, > 1 ESMDA step).
"""

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
    create_time_varying_true_params,
    create_true_params,
    model_args,
    prepare_forward_model,
    sample_smooth_ensemble,
    solver_name,
)

BASE_RESULTS_DIR = pathlib.Path(".temp/scripts")

# Small but non-degenerate grid: matches the shape used by the test suite
# in tests/config.py, which is known to run end-to-end with uDALES.
DOMAIN = {
    "nx": 40,
    "ny": 40,
    "nz": 4,
    "bounds": ((0.0, 40.0), (0.0, 40.0), (0.0, 10.0)),
}

# 2-minute window so num_time_points = int(120 / 60) = 2 (the minimum that
# gives a non-trivial time-varying parameter trajectory).
TIME = {
    "simulation_time": 120.0,
    "output_frequency": 10.0,
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
    "nudging_config": {
        "tnudge": 15.0,
        "nnudge": 2,
        "profile_config": {"type": "power_law", "alpha": 0.25},
    },
}

PALM_ARGS = {
    "case_dir": "examples/palm/experiments/xie_and_castro_palm",
    "stl_path": "examples/palm/experiments/xie_and_castro_palm/xie_castro_2008_STL.stl",
    "experiment_name": "urban_run",
    "ncpu": 1,
    "save_only_last_timestep": False,
    "verbose": False,
    "boundary_condition": "inflow_outflow",
    "nudging_config": {
        "profile_config": {"type": "uniform"},
    },
    "compile": False,
}

# 8-member ensemble with 4-way parallelism: two batches of 4 keeps the
# parallel path exercised without saturating a small partition.
ENSEMBLE = {
    "ensemble_size": 8,
    "num_parallel_processes": 4,
    "num_cpus_per_process": 1,
    "failure_policy": "resample_from_successes",
    "failure_jitter_scale": 0.05,
    "failure_seed": 0,
}

OBS = {
    "x_points": [10.0, 20.0, 30.0, 38.0, 10.0],
    "y_points": [20.0, 25.0, 10.0, 30.0, 2.0],
    "z_points": [1.0, 1.0, 1.0, 1.0, 1.0],
    "states": ["u", "v", "w"],
    "temporal_mode": "intervals",
    "interval_size": 3,
    "aggregation_mode": "mean",
}

# 2 windows is the minimum that still tests the inter-window handoff
# (state warm-start and prior extrapolation).
ESMDA = {
    "num_steps": 1,
    "num_assimilation_windows": 2,
    "seed": 42,
    "obs_error_std": 0.25,
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
    "inflow_angle_std": 10.0,
    "velocity_mean": 3.0,
    "velocity_std": 1.0,
    "pressure_mean": 0.0041912,
    "pressure_std": 0.001,
}

EXTERNAL_PRIORS = {
    "inflow_angle":       {"mean": 0.0, "std": 10.0},
    "velocity_magnitude": {"mean": 5.0, "std": 0.5, "min": 0.1},
}

TIME_VARYING_PARAMS = {
    "num_time_points": 4,
    "method": "ar2_relaxation",
    "method_kwargs": {
        "gp_linear_trend": {
            "correlation_length": 10.0,
            "slope_damping_time": None,
        },
        "ar1": {
            "correlation_length": 10.0,
            "phi_max": 0.999,
        },
        "ornstein_uhlenbeck": {
            "correlation_length": 200.0,
            "phi_max": 0.999,
        },
        "ar2_relaxation": {
            "correlation_length": 200.0,
        },
    },
    "truth_method": "ar2_relaxation",
    "truth_method_kwargs": {
        "gp_linear_trend": {
            "correlation_length": 100.0,
        },
        "ar1": {
            "correlation_length": 100.0,
            "phi_max": 0.999,
        },
        "ornstein_uhlenbeck": {
            "correlation_length": 200.0,
            "phi_max": 0.999,
        },
        "ar2_relaxation": {
            "correlation_length": 200.0,
        },
    },
}
