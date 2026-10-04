"""Run the hybrid: ESMDA for the parameters, a filter for the state, per window.

Config: configs/assimilation.yaml (the `assimilation`, `observation`,
`smoothing`, `filtering` and `hybrid` blocks). In each window the smoother
estimates the parameters from the whole window's observations, then the filter
runs over the same observations, cycle by cycle, with those parameters and
produces the window's state (FilterSmoothing in data_assimilation).

    python scripts/run_hybrid.py
    python scripts/run_hybrid.py 'smoothing.smoother=${smoother.static}' \
        params@truth_params=static_truth params@prior_params=static

The outputs follow the same per-window layout as run_smoother.py, so
compute_metrics.py and visualize_assimilation.py read them the same way.

Outputs, in `<paths.results_dir>/hybrid/`:
    config.yaml, run_info.yaml, true_state.nc (inline truth), true_params.nc
    windows/window_{w}_prior_params.nc      the smoother's prior
            window_{w}_posterior_params.nc  the smoother's posterior
            window_{w}_posterior_state.nc   the filter's analyzed state per cycle
            window_{w}_obs.nc               the smoother's observations and
                                            predicted observations per ESMDA step
            window_{w}_filter_obs.nc        the filter's; esmda_step 0 is the
                                            forecast, 1 the analysis
            window_{w}_filter_params.nc     the filter's corrected parameters
                                            per cycle (only in mode=joint)
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
import xarray
from data_assimilation.filter_smoothing import FilterSmoothing, resolve_tempering_policy
from hydra.utils import instantiate
from omegaconf import DictConfig
from tqdm import tqdm

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent / "utils"))

from helper_functions import (  # noqa: E402
    StridedOperator,
    cycle_observations,
    cycles_to_time,
    flatten_obs,
    make_aggregation,
    make_ensemble_model,
    make_observation_error,
    make_observation_operator,
    make_run_dir,
    make_truth,
    next_window_params,
    parameter_names,
    save_obs,
    save_yaml,
)
from inconsistency_check import check_config  # noqa: E402


def run(cfg: DictConfig) -> None:
    check_config(cfg, "hybrid")
    da = cfg.assimilation
    num_windows = int(da.num_windows)
    sim_time = float(cfg.time.simulation_time)
    ensemble_size = int(cfg.ensemble.ensemble_size)
    stride = int(da.assimilate_every_n_step)
    cycle_seconds = stride * float(cfg.time.output_frequency)
    rng_key = jax.random.PRNGKey(da.seed)

    out_dir, windows_dir = make_run_dir(cfg, "hybrid")

    # --- Truth and one observation per cycle ----------------------------------------
    truth = make_truth(cfg, out_dir)
    cycles_per_window = truth.sizes["time"] // num_windows // stride
    rng_key, key = jax.random.split(rng_key)
    observations = cycle_observations(cfg, truth, num_windows * cycles_per_window, key)
    truth.close()
    # The hybrid works on the window clock: cycle k of a window ends at
    # (k + 1) * cycle_seconds, whatever the truth's own frame times are.
    windows: list[dict[str, Any]] = []
    for w in range(num_windows):
        cycles = observations[w * cycles_per_window : (w + 1) * cycles_per_window]
        clock = [(k + 1) * cycle_seconds for k in range(cycles_per_window)]
        windows.append(
            {
                "noisy": [c[0].assign_coords(time=[t]) for c, t in zip(cycles, clock)],
                "clean": [c[1].assign_coords(time=[t]) for c, t in zip(cycles, clock)],
                "errors": [c[2] for c in cycles],
                "times": np.array([c[3] for c in cycles]),
            }
        )

    # The smoother assimilates a whole window (possibly aggregated): its own C_D.
    truth_operator = make_observation_operator(cfg, cfg.truth_model.solver_name)
    error = make_observation_error(cfg)
    aggregation = make_aggregation(cfg)
    for window in windows:
        clean = xarray.concat(window["clean"], dim="time", join="override")
        window["smoother_error"] = error.resolve(clean, truth_operator, aggregation)

    # --- Prior, smoother, filter and the hybrid around them --------------------------
    prior_sampler = instantiate(cfg.prior_params)
    params = prior_sampler.sample(ensemble_size)
    is_dynamic = "time" in params.dims
    selected, global_names = parameter_names(cfg, params)
    operator = make_observation_operator(cfg, cfg.assim_model.solver_name)
    if stride > 1:
        operator = StridedOperator(operator, stride)
    policy = resolve_tempering_policy(
        cfg.filtering.beta, cfg.hybrid.likelihood_allocation
    )

    # Two model stacks with separate scratch dirs (for the backends that have
    # one): the smoother forecasts whole windows, the filter one cycle at a time.
    # On disk, each stack writes its member files under its own root.
    scratch = pathlib.Path(cfg.paths.experiment_dir)

    def temp_dir(name: str) -> dict:
        has_scratch = "temp_dir" in cfg.assim_model.forward_model
        return {"temp_dir": scratch / name} if has_scratch else {}

    states_dir = out_dir / "_ensemble_states"
    on_disk = bool(da.ensemble_save_on_disk)
    smoother_model = make_ensemble_model(
        cfg,
        states_dir / "smoother" if on_disk else None,
        **temp_dir("hybrid_smoother"),
    )
    filter_model = make_ensemble_model(
        cfg,
        states_dir / "filter" if on_disk else None,
        **temp_dir("hybrid_filter"),
        simulation_time=cycle_seconds,
    )

    rng_key, smoother_key, filter_key = jax.random.split(rng_key, 3)
    smoother = instantiate(
        cfg.smoothing.smoother,
        observation_operator=operator,
        forward_model=smoother_model,
        C_D=jnp.asarray(windows[0]["smoother_error"].covariance_diag),
        rng_key=smoother_key,
        aggregate_observations=aggregation,
        parameter_names_to_estimate=selected,
        global_parameter_names=global_names,
        likelihood_weight=policy.smoother_weight,
        **({"num_time_points": int(params.sizes["time"])} if is_dynamic else {}),
    )
    smoother.collect_obs_diagnostics = True
    # On disk, delete forecasts once used: the filter makes the window's state.
    smoother.prune_disk_steps = True
    smoother.keep_prior_disk_step = False
    enkf = instantiate(
        cfg.filtering,
        observation_operator=operator,
        forward_model=filter_model,
        C_D=jnp.asarray(windows[0]["errors"][0].covariance_diag),
        rng_key=filter_key,
        parameter_names_to_estimate=selected,
        global_parameter_names=global_names,
    )
    enkf.collect_pred_obs = True
    enkf.prune_disk_cycles = True
    enkf.keep_first_disk_cycle = False
    hybrid = FilterSmoothing(smoother=smoother, filter=enkf, tempering=policy)

    # --- Window loop ------------------------------------------------------------------
    state = None  # cold start
    seconds_per_window = []
    for w, window in enumerate(tqdm(windows, desc="windows")):
        start = time.perf_counter()
        params.to_netcdf(windows_dir / f"window_{w}_prior_params.nc")
        if hasattr(smoother, "pin_initial_time_point"):
            # Keep time-varying parameters continuous across window boundaries.
            smoother.pin_initial_time_point = w > 0
        smoother.set_observation_covariance(window["smoother_error"].covariance_diag)
        enkf.set_observation_covariance(window["errors"][0].covariance_diag)

        result = hybrid.run(
            state=state,
            params=params,
            observations=window["noisy"],
            return_history=True,
        )

        posterior_params = result.esmda_params
        posterior_params.to_netcdf(windows_dir / f"window_{w}_posterior_params.nc")
        cycles_to_time(result.state_history, window["times"]).to_netcdf(
            windows_dir / f"window_{w}_posterior_state.nc"
        )
        if result.params_history is not None:  # filtering.mode=joint
            cycles_to_time(result.params_history, window["times"]).to_netcdf(
                windows_dir / f"window_{w}_filter_params.nc"
            )
        noisy = xarray.concat(window["noisy"], dim="time", join="override")
        clean = xarray.concat(window["clean"], dim="time", join="override")
        save_obs(
            windows_dir / f"window_{w}_obs.nc",
            flatten_obs(noisy, aggregation),
            flatten_obs(clean, aggregation),
            window["smoother_error"].std,
            smoother.pred_obs_history,
        )
        save_obs(
            windows_dir / f"window_{w}_filter_obs.nc",
            np.concatenate([flatten_obs(o, None) for o in window["noisy"]]),
            np.concatenate([flatten_obs(o, None) for o in window["clean"]]),
            np.concatenate([e.std for e in window["errors"]]),
            [
                np.concatenate(hybrid.pred_obs_history),
                np.concatenate(hybrid.pred_obs_post_history),
            ],
        )

        # Next window: the filter's state, and the smoother's posterior as prior
        # (extrapolated if time-varying).
        state = result.state
        if is_dynamic and w < num_windows - 1:
            rng_key, key = jax.random.split(rng_key)
            params = next_window_params(prior_sampler, posterior_params, sim_time, key)
        else:
            params = posterior_params
        seconds_per_window.append(time.perf_counter() - start)

    if on_disk:
        shutil.rmtree(states_dir, ignore_errors=True)
    save_yaml(
        {
            "smoother": type(smoother).__name__,
            "filter": type(enkf).__name__,
            "mode": str(cfg.filtering.mode),
            "likelihood_allocation": policy.likelihood_allocation,
            "beta": float(policy.beta),
            "smoother_weight": float(policy.smoother_weight),
            "num_windows": num_windows,
            "window_length_seconds": sim_time,
            "cycles_per_window": cycles_per_window,
            "cycle_seconds": cycle_seconds,
            "ensemble_size": ensemble_size,
            "num_esmda_steps": int(smoother.num_steps),
            "truth_dir": da.truth_dir,
            "truth_start_time": da.truth_start_time,
            "observation_error_model": observations[0][2].provenance,
            "seconds_per_window": [float(s) for s in seconds_per_window],
        },
        out_dir / "run_info.yaml",
    )
    print(f"Saved outputs in {out_dir}")


@hydra.main(version_base=None, config_path="../configs", config_name="assimilation")  # type: ignore[misc, unused-ignore]
def main(cfg: DictConfig) -> None:
    run(cfg)


if __name__ == "__main__":
    main()
