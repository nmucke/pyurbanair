"""Run an ESMDA smoother over consecutive assimilation windows.

Config: configs/assimilation.yaml (the `assimilation`, `observation` and
`smoothing` blocks). The smoother variant is `smoothing.smoother`, e.g.

    python scripts/run_smoother.py
    python scripts/run_smoother.py 'smoothing.smoother=${smoother.dynamic}' \
        params@truth_params=dynamic_sine params@prior_params=dynamic

Each window assimilates its slice of the truth's observations; the next window
starts from the posterior's last state and the posterior parameters (time-varying
parameters are extrapolated into the next window).

Outputs, in `<paths.results_dir>/smoother/`:
    config.yaml, run_info.yaml, true_state.nc (inline truth), true_params.nc
    windows/window_{w}_prior_params.nc, _posterior_params.nc,
            _posterior_state.nc, _prior_state.nc (assimilation.save_states: prior
            or all), _step_{i}_state.nc for ESMDA steps 1..num_steps-1 (all),
            _obs.nc (observations and predicted observations per ESMDA step;
                     obs_std is the configured std; with
                     observation.adaptive_error also obs_std_assimilated, the
                     inflated std the window used, and obs_error_group)
    run_info.yaml also lists the adaptive error's per-window inflation.
"""

from __future__ import annotations

import pyurbanair.quiet_jax  # noqa: F401  (silences JAX CPU-fallback noise)

import pathlib
import shutil
import sys
import time
from typing import Any

import hydra
import jax
import jax.numpy as jnp
import numpy as np
from data_assimilation.adaptive_observation_error import observation_heights
from hydra.utils import instantiate
from omegaconf import DictConfig
from tqdm import tqdm

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent / "utils"))

from helper_functions import (  # noqa: E402
    flatten_obs,
    make_aggregation,
    make_ensemble_model,
    make_observation_error,
    make_observation_operator,
    make_run_dir,
    make_truth,
    next_window_params,
    observe,
    parameter_names,
    save_obs,
    save_yaml,
)
from inconsistency_check import check_config  # noqa: E402


def run(cfg: DictConfig) -> None:
    check_config(cfg, "smoother")
    da = cfg.assimilation
    num_windows = int(da.num_windows)
    sim_time = float(cfg.time.simulation_time)
    ensemble_size = int(cfg.ensemble.ensemble_size)
    rng_key = jax.random.PRNGKey(da.seed)

    out_dir, windows_dir = make_run_dir(cfg, "smoother")

    # --- Truth and its observations, one block per window ----------------------
    truth = make_truth(cfg, out_dir)
    frames_per_window = truth.sizes["time"] // num_windows
    truth_operator = make_observation_operator(cfg, cfg.truth_model.solver_name)
    error = make_observation_error(cfg)
    aggregation = make_aggregation(cfg)
    adaptive = instantiate(cfg.observation.get("adaptive_error"))
    observations = []
    for w in range(num_windows):
        rng_key, key = jax.random.split(rng_key)
        window_truth = truth.isel(
            time=slice(w * frames_per_window, (w + 1) * frames_per_window)
        )
        observations.append(
            observe(window_truth, truth_operator, error, aggregation, key)
        )
    truth.close()

    # --- Prior and smoother -----------------------------------------------------
    on_disk = bool(da.ensemble_save_on_disk)
    states_dir = out_dir / "_ensemble_states"
    ensemble_model = make_ensemble_model(cfg, states_dir if on_disk else None)
    prior_sampler = instantiate(cfg.prior_params)
    params = prior_sampler.sample(ensemble_size)
    is_dynamic = "time" in params.dims

    selected, global_names, log_names = parameter_names(cfg, params, prior_sampler)
    extra = {"num_time_points": int(params.sizes["time"])} if is_dynamic else {}
    rng_key, key = jax.random.split(rng_key)
    smoother = instantiate(
        cfg.smoothing.smoother,
        observation_operator=make_observation_operator(
            cfg, cfg.assim_model.solver_name
        ),
        forward_model=ensemble_model,
        C_D=jnp.asarray(observations[0][2].covariance_diag),
        rng_key=key,
        aggregate_observations=aggregation,
        parameter_names_to_estimate=selected,
        global_parameter_names=global_names,
        log_parameter_names=log_names,
        **extra,
    )
    smoother.collect_obs_diagnostics = True
    # ESMDA steps saved besides the posterior (0 the prior); the others are
    # freed as soon as their update is done.
    num_steps = int(smoother.num_steps)
    count = {"posterior": 0, "prior": 1, "all": num_steps}[da.save_states]
    keep_steps = list(range(count))
    smoother.keep_steps = keep_steps
    state_names = {0: "prior", num_steps: "posterior"}

    # --- Window loop --------------------------------------------------------------
    state = None  # window 0 cold-starts
    seconds_per_window = []
    for w in tqdm(range(num_windows), desc="windows"):
        start = time.perf_counter()
        obs, obs_clean, resolved = observations[w]
        params.to_netcdf(windows_dir / f"window_{w}_prior_params.nc")
        if hasattr(smoother, "pin_initial_time_point"):
            # Keep time-varying parameters continuous across window boundaries.
            smoother.pin_initial_time_point = w > 0

        covariance = resolved.covariance_diag
        if adaptive is not None:
            heights = observation_heights(resolved, truth_operator)
            covariance = adaptive.variance(covariance, heights)

        output = smoother(
            state=state,
            params=params,
            observations=obs,
            observation_covariance=jnp.asarray(covariance),
            return_params_history=True,
        )
        params_history = output if on_disk else output[0]
        del output  # the posterior state is read back with get_state
        posterior_params = params_history.isel(esmda_step=-1)
        posterior_params.to_netcdf(windows_dir / f"window_{w}_posterior_params.nc")
        flat_obs = flatten_obs(obs, aggregation)
        obs_extra: dict[str, np.ndarray] | None = None
        if adaptive is not None:
            adaptive.record(
                flat_obs,
                smoother.pred_obs_history[0],
                smoother.pred_obs_history[-1],
                resolved.covariance_diag,
                heights,
            )
            obs_extra = {
                "obs_std_assimilated": np.sqrt(covariance),
                "obs_error_group": adaptive.groups(heights)[0],
            }
        save_obs(
            windows_dir / f"window_{w}_obs.nc",
            flat_obs,
            flatten_obs(obs_clean, aggregation),
            resolved.std,
            smoother.pred_obs_history,
            obs_extra,
        )

        # Saved ensemble states, one at a time, the posterior last; the next
        # window starts from its final frame (a copy, so no window state stays
        # referenced into the next window).
        for step in [*keep_steps, num_steps]:
            name = state_names.get(step, f"step_{step}")
            states = smoother.get_state(step)
            states.to_netcdf(windows_dir / f"window_{w}_{name}_state.nc")
        state = states.isel(time=-1).copy(deep=True)
        del states

        # Next window's prior: the posterior, extrapolated if time-varying.
        if is_dynamic and w < num_windows - 1:
            rng_key, key = jax.random.split(rng_key)
            params = next_window_params(prior_sampler, posterior_params, sim_time, key)
        else:
            params = posterior_params
        seconds_per_window.append(time.perf_counter() - start)

    if on_disk:
        shutil.rmtree(states_dir, ignore_errors=True)
    info: dict[str, Any] = {}
    if adaptive is not None:
        info["adaptive_observation_error"] = adaptive.history
    save_yaml(
        {
            "smoother": type(smoother).__name__,
            "num_windows": num_windows,
            "window_length_seconds": sim_time,
            "ensemble_size": ensemble_size,
            "num_esmda_steps": num_steps,
            "truth_dir": da.truth_dir,
            "truth_start_time": da.truth_start_time,
            "observation_error_model": observations[0][2].provenance,
            "seconds_per_window": [float(s) for s in seconds_per_window],
            **info,
        },
        out_dir / "run_info.yaml",
    )
    print(f"Saved outputs in {out_dir}")


@hydra.main(version_base=None, config_path="../configs", config_name="assimilation")  # type: ignore[misc, unused-ignore]
def main(cfg: DictConfig) -> None:
    run(cfg)


if __name__ == "__main__":
    main()
