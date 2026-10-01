"""Run an ESMDA smoother over consecutive assimilation windows.

Config: configs_new/assimilation.yaml (the `assimilation`, `observation` and
`smoothing` blocks). The smoother variant is `smoothing.smoother`, e.g.

    python scripts_new/run_smoother.py
    python scripts_new/run_smoother.py 'smoothing.smoother=${smoother.dynamic}' \
        params@truth_params=dynamic_sine params@prior_params=dynamic

Each window assimilates its slice of the truth's observations; the next window
starts from the posterior's last state and the posterior parameters (time-varying
parameters are extrapolated into the next window).

Outputs, in `<paths.results_dir>/smoother/`:
    config.yaml, run_info.yaml, true_state.nc (inline truth), true_params.nc
    windows/window_{w}_prior_params.nc, _posterior_params.nc,
            _posterior_state.nc, _prior_state.nc (if assimilation.save_prior_state),
            _obs.nc (observations and predicted observations per ESMDA step)
"""

from __future__ import annotations

import pathlib
import shutil
import sys
import time

import hydra
import jax
import jax.numpy as jnp
import numpy as np
import xarray
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from helper_functions import (  # noqa: E402
    concat_member_files,
    flatten_obs,
    last_frames,
    make_aggregation,
    make_ensemble_model,
    make_observation_error,
    make_observation_operator,
    make_truth,
    member_files,
    observe,
    save_obs,
    save_yaml,
)
from inconsistency_check import check_config  # noqa: E402

from pyurbanair.config.discrepancy import SGS_BIAS_PARAMETER_NAMES  # noqa: E402


def run(cfg: DictConfig) -> None:
    check_config(cfg, "smoother")
    da = cfg.assimilation
    num_windows = int(da.num_windows)
    sim_time = float(cfg.time.simulation_time)
    ensemble_size = int(cfg.ensemble.ensemble_size)
    rng_key = jax.random.PRNGKey(da.seed)

    out_dir = pathlib.Path(cfg.paths.results_dir) / "smoother"
    windows_dir = out_dir / "windows"
    windows_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, out_dir / "config.yaml", resolve=True)

    # --- Truth and its observations, one block per window ----------------------
    truth = make_truth(cfg, out_dir)
    frames_per_window = truth.sizes["time"] // num_windows
    truth_operator = make_observation_operator(cfg, cfg.truth_model.solver_name)
    error = make_observation_error(cfg)
    aggregation = make_aggregation(cfg)
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
    save_prior_state = bool(da.save_prior_state)
    ensemble_model = make_ensemble_model(
        cfg, out_dir / "_ensemble_states" if on_disk else None
    )
    prior_sampler = instantiate(cfg.prior_params)
    params = prior_sampler.sample(ensemble_size)
    is_dynamic = "time" in params.dims

    selected = da.params_to_estimate
    discrepancy = OmegaConf.select(
        cfg, "assim_model.forward_model.model_discrepancy.enabled", default=False
    )
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
        parameter_names_to_estimate=None if selected is None else list(selected),
        # SGS-discrepancy coefficients are global: never localized.
        global_parameter_names=[
            n for n in SGS_BIAS_PARAMETER_NAMES if discrepancy and n in params
        ],
        **extra,
    )
    smoother.collect_obs_diagnostics = True
    # On disk, delete each ESMDA step's forecasts once used (keep prior + posterior).
    smoother.prune_disk_steps = True
    smoother.keep_prior_disk_step = save_prior_state

    # --- Window loop --------------------------------------------------------------
    state = None  # window 0 cold-starts
    window_seconds = []
    for w in tqdm(range(num_windows), desc="windows"):
        start = time.perf_counter()
        obs, obs_clean, resolved = observations[w]
        params.to_netcdf(windows_dir / f"window_{w}_prior_params.nc")
        if hasattr(smoother, "pin_initial_time_point"):
            # Keep time-varying parameters continuous across window boundaries.
            smoother.pin_initial_time_point = w > 0

        output = smoother(
            state=state,
            params=params,
            observations=obs,
            observation_covariance=jnp.asarray(resolved.covariance_diag),
            return_params_history=True,
            return_state_history=save_prior_state and not on_disk,
        )
        params_history = output if on_disk else output[0]
        posterior_params = params_history.isel(esmda_step=-1)
        posterior_params.to_netcdf(windows_dir / f"window_{w}_posterior_params.nc")
        save_obs(
            windows_dir / f"window_{w}_obs.nc",
            flatten_obs(obs, aggregation),
            flatten_obs(obs_clean, aggregation),
            resolved.std,
            smoother.pred_obs_history,
        )

        # Posterior (and prior) ensemble states; next window starts from the end.
        if on_disk:
            steps_dir = smoother.base_results_dir
            if save_prior_state:
                concat_member_files(
                    member_files(steps_dir / "step_0", ensemble_size),
                    windows_dir / f"window_{w}_prior_state.nc",
                )
            posterior_files = member_files(
                steps_dir / f"step_{smoother.num_steps}", ensemble_size
            )
            concat_member_files(
                posterior_files, windows_dir / f"window_{w}_posterior_state.nc"
            )
            state = last_frames(posterior_files)
            for step in range(smoother.num_steps + 1):
                shutil.rmtree(steps_dir / f"step_{step}", ignore_errors=True)
        else:
            states = output[1]
            if save_prior_state:
                states.isel(esmda_step=0).to_netcdf(
                    windows_dir / f"window_{w}_prior_state.nc"
                )
                states = states.isel(esmda_step=-1)
            states.to_netcdf(windows_dir / f"window_{w}_posterior_state.nc")
            state = states.isel(time=-1)

        # Next window's prior: the posterior, extrapolated if time-varying.
        if is_dynamic and w < num_windows - 1:
            knot_times = np.asarray(prior_sampler.time_coords)
            rng_key, key = jax.random.split(rng_key)
            params = prior_sampler.extrapolate(
                posterior_params, jnp.asarray(knot_times) + sim_time, key
            ).assign_coords(time=knot_times)
        else:
            params = posterior_params
        window_seconds.append(time.perf_counter() - start)

    save_yaml(
        {
            "smoother": type(smoother).__name__,
            "num_windows": num_windows,
            "window_seconds": sim_time,
            "ensemble_size": ensemble_size,
            "num_esmda_steps": int(smoother.num_steps),
            "truth_dir": da.truth_dir,
            "truth_start_time": da.truth_start_time,
            "observation_error_model": observations[0][2].provenance,
            "seconds_per_window": [float(s) for s in window_seconds],
        },
        out_dir / "run_info.yaml",
    )
    print(f"Saved outputs in {out_dir}")


@hydra.main(version_base=None, config_path="../configs_new", config_name="assimilation")  # type: ignore[misc, unused-ignore]
def main(cfg: DictConfig) -> None:
    run(cfg)


if __name__ == "__main__":
    main()
