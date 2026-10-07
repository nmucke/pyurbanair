"""Run an ensemble Kalman filter over consecutive assimilation windows.

Config: configs/assimilation.yaml (the `assimilation`, `observation` and
`filtering` blocks). `filtering` is the EnsembleKalmanFilter itself, e.g.

    python scripts/run_filtering.py params@prior_params=static \
        'filtering.analysis=${analysis.letkf}' \
        'filtering.localization=${localization.distance}'

Each window is a sequence of cycles: forecast `assimilate_every_n_step` output
frames, then assimilate the last one. Windows only chunk the run (state,
parameters and the random stream carry over), so they keep the outputs in the
same per-window layout as the smoother and the same compute_metrics.py and
visualize_assimilation.py read both.

Outputs, in `<paths.results_dir>/filtering/`:
    config.yaml, run_info.yaml, true_state.nc (inline truth), true_params.nc
    windows/window_{w}_prior_params.nc      parameters entering each cycle
            window_{w}_posterior_params.nc  parameters after each analysis
            window_{w}_posterior_state.nc   analyzed state at each cycle
            window_{w}_obs.nc               observations; esmda_step 0 is the
                                            forecast, 1 the analysis
            window_{w}_forecast_state.nc    every forecast frame (only with
                                            assimilation.save_forecast_history)
Both parameter files have a `time` dim: the cycles' analysis times.
"""

from __future__ import annotations

import pyurbanair.quiet_jax  # noqa: F401  (silences JAX CPU-fallback noise)

import pathlib
import shutil
import sys
import time

import hydra
import jax
import jax.numpy as jnp
import numpy as np
from hydra.utils import instantiate
from omegaconf import DictConfig
from tqdm import tqdm

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent / "utils"))

from helper_functions import (  # noqa: E402
    StridedOperator,
    cycle_observations,
    cycles_to_time,
    flatten_obs,
    make_ensemble_model,
    make_observation_operator,
    make_run_dir,
    make_truth,
    parameter_names,
    save_obs,
    save_yaml,
)
from inconsistency_check import check_config  # noqa: E402


def run(cfg: DictConfig) -> None:
    check_config(cfg, "filtering")
    da = cfg.assimilation
    num_windows = int(da.num_windows)
    sim_time = float(cfg.time.simulation_time)
    ensemble_size = int(cfg.ensemble.ensemble_size)
    stride = int(da.assimilate_every_n_step)
    cycle_seconds = stride * float(cfg.time.output_frequency)
    rng_key = jax.random.PRNGKey(da.seed)

    out_dir, windows_dir = make_run_dir(cfg, "filtering")

    # --- Truth and one observation per cycle ---------------------------------------
    # A cycle covers `stride` truth frames and assimilates the last one.
    truth = make_truth(cfg, out_dir)
    cycles_per_window = truth.sizes["time"] // num_windows // stride
    rng_key, key = jax.random.split(rng_key)
    observations = cycle_observations(cfg, truth, num_windows * cycles_per_window, key)
    truth.close()

    # --- Prior and filter -----------------------------------------------------------
    on_disk = bool(da.ensemble_save_on_disk)
    states_dir = out_dir / "_ensemble_states"
    ensemble_model = make_ensemble_model(
        cfg, states_dir if on_disk else None, simulation_time=cycle_seconds
    )
    params = instantiate(cfg.prior_params).sample(ensemble_size)

    operator = make_observation_operator(cfg, cfg.assim_model.solver_name)
    selected, global_names = parameter_names(cfg, params)
    rng_key, key = jax.random.split(rng_key)
    enkf = instantiate(
        cfg.filtering,
        observation_operator=(
            StridedOperator(operator, stride) if stride > 1 else operator
        ),
        forward_model=ensemble_model,
        C_D=jnp.asarray(observations[0][2].covariance_diag),
        rng_key=key,
        parameter_names_to_estimate=selected,
        global_parameter_names=global_names,
    )
    enkf.collect_pred_obs = True
    enkf.collect_forecast_frames = bool(da.save_forecast_history)
    # On disk, delete each cycle's forecasts once its analysis is done.
    enkf.prune_disk_cycles = True
    enkf.keep_first_disk_cycle = False

    # --- Window loop ------------------------------------------------------------------
    state = None  # cycle 0 cold-starts
    seconds_per_window = []
    for w in tqdm(range(num_windows), desc="windows"):
        start = time.perf_counter()
        cycles = observations[w * cycles_per_window : (w + 1) * cycles_per_window]
        times = np.array([c[3] for c in cycles])

        result = enkf.run(
            state=state,
            params=params,
            observations=[c[0] for c in cycles],
            observation_covariances=np.stack([c[2].covariance_diag for c in cycles]),
            return_history=True,
        )

        # params_history: what entered cycle 0, then the analysis after each cycle.
        history = result.params_history
        for name, part in (("prior", slice(0, -1)), ("posterior", slice(1, None))):
            cycles_to_time(history.isel(cycle=part), times).to_netcdf(
                windows_dir / f"window_{w}_{name}_params.nc"
            )
        cycles_to_time(result.state_history, times).to_netcdf(
            windows_dir / f"window_{w}_posterior_state.nc"
        )
        if result.forecast_history is not None:
            frames = cycles_per_window * stride
            result.forecast_history.assign_coords(
                time=w * sim_time + cycle_seconds / stride * (np.arange(frames) + 1)
            ).transpose("ensemble", "time", ...).to_netcdf(
                windows_dir / f"window_{w}_forecast_state.nc"
            )
        save_obs(
            windows_dir / f"window_{w}_obs.nc",
            np.concatenate([flatten_obs(c[0], None) for c in cycles]),
            np.concatenate([flatten_obs(c[1], None) for c in cycles]),
            np.concatenate([c[2].std for c in cycles]),
            [
                np.concatenate(enkf.pred_obs_history),
                np.concatenate(enkf.pred_obs_post_history),
            ],
        )

        state, params = result.state, result.params
        seconds_per_window.append(time.perf_counter() - start)

    if on_disk:
        shutil.rmtree(states_dir, ignore_errors=True)
    save_yaml(
        {
            "filter": type(enkf).__name__,
            "mode": str(cfg.filtering.mode),
            "num_windows": num_windows,
            "window_length_seconds": sim_time,
            "cycles_per_window": cycles_per_window,
            "cycle_seconds": cycle_seconds,
            "ensemble_size": ensemble_size,
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
