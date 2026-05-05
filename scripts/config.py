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

DOMAIN = {
    "nx": 50,
    "ny": 40,
    "nz": 16,
    "bounds": ((-10.0, 40.0), (0.0, 40.0), (0.0, 40.0)),
}
# DOMAIN = {
#     "nx": 90,
#     "ny": 80,
#     "nz": 16,
#     "bounds": ((-10.0, 80.0), (0.0, 80.0), (0.0, 40.0)),
# }

TIME = {
    "simulation_time": 5*60.0,  # 3000 * 0.0538,
    "output_frequency": 5.0,  # 3000 * 0.0538,
    "spinup_time": 10.0,
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
    "ncpu": 4,
    "save_only_last_timestep": False,
    "verbose": False,
    "boundary_condition": "inflow_outflow",
    "nudging_config": {
        "tnudge": 15.0,
        "nnudge": 4,
        # Vertical inflow profile.  Omit for uniform (back-compat).  Supported:
        #   {"type": "uniform"},
        #   {"type": "power_law", "alpha": 0.25, "z_ref": 40.0}
        # velocity_magnitude is interpreted as the speed at z_ref.
        # "profile_config": {"type": "power_law", "alpha": 0.25},
        "profile_config": {"type": "uniform"},
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
        # "profile_config": {"type": "power_law", "alpha": 0.25},
        "profile_config": {"type": "uniform"},
    },
    "compile": False,
}

ENSEMBLE = {
    "ensemble_size": 64,
    "num_parallel_processes": 64,
    "num_cpus_per_process": 1,
    # Failure handling for individual ensemble members. With
    # "resample_from_successes", a per-member CalledProcessError is logged
    # and that slot is replaced by a clone of a randomly chosen successful
    # member (state cloned, params cloned + jittered). "raise" preserves
    # the historical fail-the-whole-ensemble behavior.
    "failure_policy": "resample_from_successes",
    "failure_jitter_scale": 0.05,
    "failure_seed": 0,
}

OBS = {
    # "x_points": [20.0, 20.0, 40.0, 50.0, 60.0],
    # "y_points": [20.0, 60.0, 10.0, 40.0, 60.0],
    "x_points": [10.0, 20.0, 30.0, 38.0, 10.0],
    "y_points": [20.0, 25.0, 10.0, 30.0, 2.0],
    "z_points": [1.0, 1.0, 1.0, 1.0, 1.0],
    "states": ["u", "v", "w"],
    "temporal_mode": "intervals",
    "interval_size": 3,
    "aggregation_mode": "mean",
}

ESMDA = {
    "num_steps": 3,
    "num_assimilation_windows": 6,
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

# External-prior estimates (the paper's x_ext and Σ_ext).  Consumed by the
# pyurbanair.parameter_time_series classes for both prior sampling and
# (where applicable) between-window relaxation toward x_ext.  Optional
# "min"/"max" entries clip generated values.
EXTERNAL_PRIORS = {
    "inflow_angle":       {"mean": 0.0, "std": 10.0},
    "velocity_magnitude": {"mean": 5.0, "std": 0.5, "min": 0.1},
}

# Time-varying parameter prior + extrapolation.
#
# ``method`` selects one of the pyurbanair.parameter_time_series classes;
# ``method_kwargs[<method>]`` carries that class's hyperparameters.  Each
# class exposes ``sample_prior`` (cold-start initial-window draw) and
# ``extrapolate`` (next window's prior given the previous posterior), so
# the rollout loop is agnostic to the choice.
#
# Available methods:
#   "gp_linear_trend"     RBF GP prior; per-member linear-trend + GP
#                         residual extrapolation, optional slope damping.
#   "ar1"                 Stationary AR(1) prior; per-member AR(1) fit
#                         rolled forward deterministically from the last
#                         posterior value.
#   "ornstein_uhlenbeck"  Stationary OU prior; per-member OU-with-drift
#                         fit rolled forward stochastically (E-M) with
#                         ensemble-pooled diffusion.
#   "ar2_relaxation"      Critically-damped AR(2) prior (Evensen 2024)
#                         carrying state across windows; next-window
#                         prior blended with posterior mean via
#                         exponential relaxation toward x_ext.
TIME_VARYING_PARAMS = {
    "num_time_points": int(TIME['simulation_time'] / 60),
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
    # Truth-trajectory generator.  Selects one of the same methods to
    # draw the underlying true parameter trajectory (ensemble_size=1).
    # Use a different correlation length than the assimilation prior to
    # avoid the inverse crime of identical generative processes.
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
