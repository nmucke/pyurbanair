"""Run a forward model: single member or ensemble, one window or a rollout,
with static or time-varying inflow.

This single script replaces the former
run_{ensemble_,rollout_,ensemble_rollout_,time_varying_}forward_model.py
family. Three `run.*` knobs select the mode:

  * ``run.ensemble`` (bool)     -> run an N-member ensemble instead of one member.
  * ``run.num_steps`` (int)     -> roll the state forward over this many windows
                                   (num_steps=1 is a single forward).
  * ``run.time_varying`` (bool) -> drive a single member with a time-varying
                                   inflow drawn from the external-prior AR(2)
                                   truth model; persists state.nc + params.nc as
                                   a ground-truth artifact and plots the derived
                                   inflow angle. (Not combinable with
                                   ``run.ensemble``.)

Examples::

    python scripts/run_forward_model.py model=pylbm
    python scripts/run_forward_model.py model=pyudales run.ensemble=true
    python scripts/run_forward_model.py run.num_steps=3                # rollout
    python scripts/run_forward_model.py run.ensemble=true run.num_steps=3
    python scripts/run_forward_model.py model=pylbm run.time_varying=true
"""

import pathlib
import sys
import time

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import hydra
import numpy as np
import xarray
from hydra.utils import instantiate
from omegaconf import DictConfig
from pyurbanair.config.hydra_helpers import (
    clean_outputs,
    resolve_output_dir,
)
from pyurbanair.utils.run_utils import add_velocity_magnitude

from scripts._common import (
    plot_derived_inflow_angle,
    plot_derived_velocity_magnitude,
    resolve_results_dir,
    visualize_forward_state,
)


def get_stepper(model, is_ensemble):
    if is_ensemble:
        def step(params, state=None):
            out = model.run_ensemble(params=params, state=state, sim_name="state")
            return out if out is not None else model.get_states()
        return step
    else:
        def step(params, state=None):
            out = model(params=params, state=state)
            return out if out is not None else model.get_states()
        return step

# The sampler always emits an `ensemble` dim. A single-member run must hand
# the forward model params WITHOUT it (scalar for static, (time,) for
# dynamic) -- the solver's inflow application can't handle a size-1 ensemble
# axis. Keep the ensemble dim in params_list so extrapolate() still sees it;
# drop it only at the model call.
def _member_params(p, is_ensemble):
    if not is_ensemble and "ensemble" in p.dims:
        return p.isel(ensemble=0, drop=True)
    return p

# Stitch rollout windows onto a single, monotonic global time axis. Solvers
# report a per-window local clock (each window restarts near 0), so re-base
# window w to start at w * simulation_time. This puts the coarse params grid
# and the fine state grid on the same axis, so the derived-vs-prescribed
# inflow-angle plot lines up. A single window is returned unchanged.
def _concat_windows(window_list, cfg):
    if len(window_list) == 1:
        return window_list[0]
    sim = float(cfg.time.simulation_time)
    rebased = []
    for w, ds in enumerate(window_list):
        t = np.asarray(ds["time"].values, dtype=float)
        rebased.append(ds.assign_coords(time=(t - t[0]) + w * sim))
    return xarray.concat(rebased, dim="time", join="override")


def run(cfg: DictConfig) -> None:
    import pdb

    import jax
    rng_key = jax.random.PRNGKey(int(cfg.params.get("seed", 0)))

    is_ensemble = cfg.run.ensemble
    model_name = cfg.model.name
    rollout_steps = cfg.run.rollout_steps

    # Instantiate parameters
    params_sampler = instantiate(cfg.params)
    params = params_sampler.sample(
        cfg.ensemble.ensemble_size if is_ensemble else 1
    )
    is_dynamic_params = "time" in params.coords

    # Instantiate forward model.
    forward_model = instantiate(
        cfg.model.forward_model,
        results_dir=resolve_results_dir(cfg),
    )
    instantiate(cfg.model.prepare, forward_model=forward_model)

    # Clean outputs from previous runs
    clean_outputs(model_name=model_name, forward_model=forward_model)

    # Instantiate ensemble model if running ensemble
    if is_ensemble:
        forward_model = instantiate(cfg.model.ensemble_model, forward_model=forward_model)

    t1 = time.time()

    # Simulate
    stepper = get_stepper(forward_model, is_ensemble)
    out = stepper(params=_member_params(params, is_ensemble))

    # Rollout simulation if rollout_steps > 0
    sim = float(cfg.time.simulation_time)
    state = [out]
    params_list = [params]
    for _ in range(rollout_steps):
        if is_dynamic_params:
            next_window_times = np.linspace(
                0.0, sim, cfg.params.time_coords.num
            )
            rng_key, subkey = jax.random.split(rng_key)
            params_next = params_sampler.extrapolate(
                params_list[-1], next_window_times, subkey
            )
            params_list.append(params_next)
        out = stepper(params=_member_params(params_list[-1], is_ensemble), state=state[-1])
        state.append(out)

    t2 = time.time()
    elapsed = t2-t1

    ##### Post processing and plotting #####
    params = _concat_windows(params_list, cfg)
    state = _concat_windows(state, cfg)
    state = add_velocity_magnitude(state)

    print(f"Model: {model_name}{' (time-varying inflow)' if is_dynamic_params else ''}")
    if is_ensemble:
        print(f"Ensemble size: {cfg.ensemble.ensemble_size}")
    if rollout_steps > 1:
        print(f"Rollout: {rollout_steps} steps")
    print(f"Elapsed: {elapsed:.2f} seconds")
    print(f"Dims: {dict(state.sizes)}")
    print(f"Vars: {list(state.data_vars)}")
    if rollout_steps:
        # Report the start -> end of the (time-varying) inflow. Reduce the
        # ensemble dim to a representative mean and coerce to 1-D so this works
        # whether or not params carry `ensemble` / `time` dims.
        inflow = params["inflow_angle"]
        velocity = params["velocity_magnitude"]
        if "ensemble" in inflow.dims:
            inflow = inflow.mean("ensemble")
            velocity = velocity.mean("ensemble")
        inflow = np.atleast_1d(inflow.values)
        velocity = np.atleast_1d(velocity.values)
        print(f"Inflow angle: {float(inflow[0]):.1f} -> {float(inflow[-1]):.1f} deg")
        print(
            f"Velocity magnitude: {float(velocity[0]):.1f} -> "
            f"{float(velocity[-1]):.1f} m/s"
        )

    # Nothing to write or draw.
    if cfg.run.skip_viz and not is_dynamic_params:
        return

    # Single output folder for everything this run produces.
    suffix = model_name
    if is_ensemble:
        suffix += "_ensemble"
    if rollout_steps > 1:
        suffix += "_rollout"
    if is_dynamic_params:
        suffix += "_time_varying"
    out_dir = resolve_output_dir(cfg, "forward_model") / suffix
    out_dir.mkdir(parents=True, exist_ok=True)

    if is_dynamic_params:
        # The ground-truth artifact / derived-inflow plot are single-member
        # concepts; pick member 0 when this was an ensemble run. The state/params
        # are persisted regardless of the viz toggle (a downstream twin / DA
        # experiment reads them).
        if "ensemble" in state.dims:
            state = state.isel(ensemble=0, drop=True)
        if "ensemble" in params.dims:
            params = params.isel(ensemble=0, drop=True)
        state.to_netcdf(out_dir / "state.nc")
        params.to_netcdf(out_dir / "params.nc")
        print(f"Saved state -> {out_dir / 'state.nc'}")
        print(f"Saved params -> {out_dir / 'params.nc'}")

    if cfg.run.skip_viz:
        return

    # Visualize once, into the single out_dir. For a plain ensemble show the
    # ensemble mean; a dynamic run has already been reduced to member 0 above.
    label = model_name
    if is_dynamic_params:
        label += " time-varying" + (" (member 0)" if is_ensemble else "")
    elif is_ensemble:
        label += " ensemble (mean)"
    if rollout_steps > 1:
        label += " rollout"

    viz_state = state.mean(dim="ensemble") if "ensemble" in state.dims else state
    visualize_forward_state(viz_state, model_name, out_dir, label)
    if is_dynamic_params:
        plot_derived_inflow_angle(state, params, out_dir)
        plot_derived_velocity_magnitude(state, params, out_dir)

@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    run(cfg)


if __name__ == "__main__":
    main()
