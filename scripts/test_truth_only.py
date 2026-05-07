"""Reproduce window 0 of the truth_model in run_time_varying_parameters_rollout_esmda.

Same code path as ``truth_model(params=true_params_0, state=None)`` in the
rollout script: builds ``truth_ts_model`` from ``EXTERNAL_PRIORS``, draws
one window's truth trajectory, and runs the uDALES forward model once.

Used to bisect the cleanup-time SIGSEGV without paying for the ensemble
(~5 min vs hours).
"""

import argparse
import pathlib
import sys

import jax
import jax.numpy as jnp
import numpy as np
import xarray

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from pyurbanair.parameter_time_series import build_parameter_time_series

from scripts import config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="pyudales", choices=["pyudales", "pylbm", "pypalm"])
    args = parser.parse_args()

    num_windows = int(config.ESMDA["num_assimilation_windows"])
    num_time_points = config.TIME_VARYING_PARAMS["num_time_points"]
    sim_time = config.TIME["simulation_time"]
    truth_method = config.TIME_VARYING_PARAMS["truth_method"]
    truth_method_kwargs = config.TIME_VARYING_PARAMS["truth_method_kwargs"][truth_method]

    rng_key = jax.random.PRNGKey(config.ESMDA["seed"])

    truth_ts_model = build_parameter_time_series(
        method=truth_method,
        external_priors=config.EXTERNAL_PRIORS,
        ensemble_size=1,
        method_kwargs=truth_method_kwargs,
    )

    rng_key, subkey = jax.random.split(rng_key)
    step = max(num_time_points - 1, 1)
    n_unique = num_windows * step + 1
    full_time = jnp.linspace(0.0, num_windows * sim_time, n_unique)
    full_ds = truth_ts_model.sample_prior(full_time, subkey).isel(ensemble=0, drop=True)

    local_time = np.asarray(jnp.linspace(0.0, sim_time, num_time_points))
    end = num_time_points
    data_vars = {
        name: ("time", np.asarray(full_ds[name].values[0:end]))
        for name in truth_ts_model.param_names
    }
    if args.model == "pyudales":
        data_vars["pressure_gradient_magnitude"] = config.TRUE_PARAMS[
            "pressure_gradient_magnitude"
        ]
    true_params_0 = xarray.Dataset(data_vars=data_vars, coords={"time": local_time})

    print(f"Window 0 truth params (drawn from EXTERNAL_PRIORS via {truth_method}):")
    print(true_params_0)

    truth_model = config.create_forward_model(args.model)
    config.prepare_forward_model(args.model, truth_model)
    truth_model = config.create_rollout_forward_model(args.model, truth_model)

    state = truth_model(params=true_params_0, state=None)
    print(f"Forward model finished. State dims: {dict(state.sizes)}")


if __name__ == "__main__":
    main()
