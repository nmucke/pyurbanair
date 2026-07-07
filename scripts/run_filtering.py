"""Run sequential ensemble filtering (EnKF): state / parameter / joint modes.

The filtering counterpart of ``scripts/run_esmda.py``: instead of ESMDA's
multiple tempered updates per assimilation window, the ensemble Kalman filter
forecasts the ensemble one segment at a time and applies ONE full-weight
analysis per segment (cycle), warm-starting the next cycle from the analyzed
end-of-segment state (see ``docs/data_assimilation.md`` and
``libs/data-assimilation/src/data_assimilation/filtering/``).

Mode and machinery are declarative axes (see conf/run_filtering.yaml):

  * ``filtering.mode=state|parameter|joint``
        which augmented blocks the analysis updates.
  * ``filtering/analysis=stochastic``
        the update math (deterministic ETKF/LETKF land here later).
  * ``filtering/localization=none|correlation|distance``
        reuses the smoother's localization strategies unchanged.
  * ``filtering/inflation=none|multiplicative|rtps|rtpp``
        ensemble spread maintenance.
  * ``filtering/evolution=none|random_walk``
        the parameters' forecast model between cycles (required — or an
        inflation — for the parameter-updating modes ``parameter``/``joint``).

and the truth source mirrors run_esmda.py:

  * ``run.truth_dir=null``    simulate the truth inline (default).
  * ``run.truth_dir=<path>``  load a state.nc/params.nc artifact written by
                              run_forward_model.py.

One cycle consumes one forecast segment of ``time.simulation_time`` seconds:
the truth is generated over ``filtering.num_cycles`` such segments up front,
each segment's observations are extracted with the case's temporal
observation operator, and the filter consumes the resulting
``(num_cycles, N_d)`` batch matrix in a single ``run()`` call. To assimilate
one observation interval per analysis (the plan's default cadence), set
``time.simulation_time`` equal to ``obs.interval_seconds``.

Only static scalar parameters are supported (the filter estimates the
parameter value *now*; time-varying/AR(2) priors stay with the ESMDA
smoothers for the time being) — pair with ``params@prior_params=static``.

Examples::

    python scripts/run_filtering.py filtering.mode=joint filtering.num_cycles=4
    python scripts/run_filtering.py filtering.mode=parameter \
        filtering/evolution=random_walk filtering/inflation=none
    python scripts/run_filtering.py filtering.mode=state \
        filtering/localization=correlation
"""

import dataclasses
import pathlib
import sys
import time

import hydra
import jax
import jax.numpy as jnp
import numpy as np
import xarray
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

import pyurbanair.quiet_jax  # noqa: F401  (suppress JAX CPU-fallback noise; must precede `import jax`)
from pyurbanair.config.hydra_helpers import (
    clean_outputs,
    create_observation_operator,
    filter_parameter_config,
)

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from scripts._esmda_common import open_truth, truth_x_min, write_yaml


def run(cfg: DictConfig) -> None:
    num_cycles = int(cfg.filtering.num_cycles)
    if num_cycles < 1:
        raise ValueError(f"filtering.num_cycles must be >= 1, got {num_cycles}.")
    sim_time = float(cfg.time.simulation_time)
    ensemble_size = int(cfg.ensemble.ensemble_size)
    final_time = sim_time * num_cycles
    domain_x_min = float(cfg.domain.bounds[0][0])
    rng_key = jax.random.PRNGKey(cfg.filtering.seed)

    # The filter estimates static scalar parameters only (Phase 1): a
    # time-varying (AR(2)) prior belongs to the dynamic ESMDA smoothers.
    for name, params_cfg in (
        ("truth_params", cfg.truth_params),
        ("prior_params", cfg.prior_params),
    ):
        if "seconds_per_knot" in list(params_cfg.keys()):
            raise ValueError(
                f"{name} is a time-varying (dynamic) sampler, which the filter "
                "does not support yet. Use params@prior_params=static / "
                "params@truth_params=static_truth, or the ESMDA smoothers "
                "(scripts/run_esmda.py) for time-varying parameters."
            )

    # Select which parameters the filter estimates (same contract as
    # run_esmda.py `params_to_estimate`): null -> every parameter the sampler
    # configs define; a list -> that subset, applied to prior AND truth.
    selected = cfg.get("params_to_estimate", None)
    selected = list(selected) if selected is not None else None
    truth_params_cfg = filter_parameter_config(cfg.truth_params, selected)
    prior_params_cfg = filter_parameter_config(cfg.prior_params, selected)
    if selected is not None:
        print(f"Estimating parameters: {selected}")

    # --- Output dir -----------------------------------------------------------
    out_dir = pathlib.Path(cfg.paths.results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(config=cfg, f=out_dir / "config.yaml")

    # --- Truth (simulated inline, or loaded from disk) ------------------------
    if cfg.run.truth_dir is None:
        true_forward_model = instantiate(
            cfg.truth_model.forward_model,
            results_dir=None,
            simulation_time=final_time,
        )
        truth_sampler = instantiate(truth_params_cfg)
        true_params = truth_sampler.sample(1)

        instantiate(cfg.truth_model.prepare, forward_model=true_forward_model)
        clean_outputs(model_name=cfg.truth_model.name, forward_model=true_forward_model)
        true_state = true_forward_model(params=true_params.isel(ensemble=0))

        true_state_path = out_dir / "true_state.nc"
        true_state.to_netcdf(true_state_path)
        n_total = int(true_state.sizes["time"])
        x_offset = domain_x_min - truth_x_min(true_state)
        start_idx = 0
        t_offset = 0.0
        del true_state
    else:
        truth_dir = pathlib.Path(cfg.run.truth_dir)
        true_params = xarray.load_dataset(truth_dir / "params.nc")

        # Optionally begin the horizon partway into a pre-simulated truth
        # (skip a spin-up); frames are rebased so that time becomes t=0.
        start_time = float(cfg.run.truth_start_time or 0.0)
        true_state_path = truth_dir / "state.nc"
        with xarray.open_dataset(true_state_path) as _truth_meta:
            true_times = np.asarray(_truth_meta["time"].values, dtype=float)
            x_offset = domain_x_min - truth_x_min(_truth_meta)
        start_idx = int((true_times < start_time).sum())
        t_offset = start_time
        n_total = int(((true_times[start_idx:] - t_offset) < final_time).sum())

    if x_offset:
        print(
            f"Shifting truth x by {x_offset:+g} to align with domain x_min={domain_x_min:g}"
        )

    # Number of truth frames per cycle (contiguous, half-open blocks). Guard
    # the degenerate slicings explicitly: zero frames per cycle would feed the
    # observation operator empty segments, and a remainder would be dropped
    # silently.
    if n_total < num_cycles:
        raise ValueError(
            f"The truth provides {n_total} frame(s) within the {final_time:g}s "
            f"horizon, fewer than filtering.num_cycles={num_cycles} (each "
            "cycle needs at least one frame). Increase time.simulation_time, "
            "reduce filtering.num_cycles, or point run.truth_dir at a longer "
            "truth."
        )
    n_per_cycle = n_total // num_cycles
    n_dropped = n_total - n_per_cycle * num_cycles
    if n_dropped:
        print(
            f"Truth frames ({n_total}) do not divide evenly into {num_cycles} "
            f"cycles of {n_per_cycle}; the trailing {n_dropped} frame(s) are "
            "not assimilated."
        )
    true_params.to_netcdf(out_dir / "true_params.nc")

    # --- Assimilation ensemble model ------------------------------------------
    assim_results_dir = (
        pathlib.Path(cfg.run.results_dir) if cfg.run.results_dir is not None else None
    )
    assim_model = instantiate(
        cfg.assim_model.forward_model, results_dir=assim_results_dir
    )
    instantiate(cfg.assim_model.prepare, forward_model=assim_model)

    # Optional on-disk ensemble forecasts (one NetCDF per member per cycle,
    # under cycle_{k}/ dirs) — keeps a large ensemble field off host RAM. The
    # analyzed end-of-cycle state is carried in memory either way.
    ensemble_states_dir = (
        out_dir / "_ensemble_states"
        if bool(cfg.run.get("ensemble_save_on_disk", False))
        else None
    )
    ensemble_model = instantiate(
        cfg.assim_model.ensemble_model,
        forward_model=assim_model,
        results_dir=ensemble_states_dir,
    )

    # --- Prior parameter ensemble ---------------------------------------------
    prior_sampler = instantiate(prior_params_cfg)
    prior_params = prior_sampler.sample(ensemble_size)
    prior_params.to_netcdf(out_dir / "prior_params.nc")

    # --- Observation operators and per-cycle observations ----------------------
    truth_obs_op = create_observation_operator(cfg.obs, cfg.truth_model.solver_name)
    assim_obs_op = create_observation_operator(cfg.obs, cfg.assim_model.solver_name)

    obs_rows = []
    for cycle in range(num_cycles):
        cycle_truth = open_truth(
            true_state_path, n_total, x_offset, start_idx, t_offset
        ).isel(time=slice(cycle * n_per_cycle, (cycle + 1) * n_per_cycle))
        obs_rows.append(jnp.asarray(truth_obs_op(cycle_truth)))
        cycle_truth.close()
    observations = jnp.stack(obs_rows, axis=0)  # (num_cycles, N_d)

    obs_error_std = float(cfg.filtering.obs_error_std)
    C_D_diag = (obs_error_std**2) * jnp.ones(observations.shape[1])
    rng_key, subkey = jax.random.split(rng_key)
    observations = observations + obs_error_std * jax.random.normal(
        subkey, observations.shape
    )

    # --- Filter ----------------------------------------------------------------
    rng_key, filter_key = jax.random.split(rng_key)
    enkf = instantiate(
        cfg.filtering.filter,
        observation_operator=assim_obs_op,
        forward_model=ensemble_model,
        C_D=C_D_diag,
        rng_key=filter_key,
    )

    save_history = bool(cfg.run.get("save_history", True))
    filter_start = time.perf_counter()
    result = enkf.run(
        state=None,  # cold start; the first segment includes the model's spin-up
        params=prior_params,
        observations=observations,
        return_history=save_history,
    )
    filter_seconds = time.perf_counter() - filter_start

    # --- Outputs ----------------------------------------------------------------
    # The analyzed end-of-run state (final-frame ensemble) and parameters; the
    # per-cycle histories when requested; the per-cycle diagnostics always.
    if result.params is not None:
        result.params.to_netcdf(out_dir / "posterior_params.nc")
    if result.state is not None:
        result.state.to_netcdf(out_dir / "posterior_state.nc")
    if result.params_history is not None:
        result.params_history.to_netcdf(out_dir / "params_history.nc")
    if result.state_history is not None:
        result.state_history.to_netcdf(out_dir / "state_history.nc")
    write_yaml(
        [dataclasses.asdict(d) for d in result.diagnostics],
        out_dir / "cycle_diagnostics.yaml",
    )

    write_yaml(
        {
            "configuration": {
                "filter": type(enkf).__name__,
                "mode": str(cfg.filtering.mode),
                "num_cycles": int(num_cycles),
                "ensemble_size": int(ensemble_size),
                "simulation_time_per_cycle": float(sim_time),
                "final_time": float(final_time),
                "observation_error_std": obs_error_std,
                "seed": int(cfg.filtering.seed),
                "truth_model": str(cfg.truth_model.name),
                "assimilation_model": str(cfg.assim_model.name),
                "truth_source": "disk" if cfg.run.truth_dir is not None else "inline",
                "truth_dir": (
                    str(cfg.run.truth_dir) if cfg.run.truth_dir is not None else None
                ),
                "num_truth_frames": int(n_total),
            },
            "timing": {
                "filter_total_seconds": float(filter_seconds),
                "mean_cycle_seconds": float(filter_seconds / max(num_cycles, 1)),
            },
        },
        out_dir / "run_info.yaml",
    )
    print(f"Saved outputs in {out_dir}")


@hydra.main(  # type: ignore[misc]
    version_base=None, config_path="../conf", config_name="run_filtering"
)
def main(cfg: DictConfig) -> None:
    run(cfg)


if __name__ == "__main__":
    main()
