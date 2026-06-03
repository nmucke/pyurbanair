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
import xarray
from hydra.utils import instantiate
from omegaconf import DictConfig
from pyurbanair.config.hydra_helpers import (
    clean_outputs,
    configure_failure_policy,
    create_parameter_ensemble,
    create_time_varying_true_params,
    create_true_params,
    resolve_output_dir,
    resolve_parameter_schema,
)
from pyurbanair.utils.run_utils import add_velocity_magnitude

from scripts._common import (
    plot_derived_inflow_angle,
    resolve_results_dir,
    visualize_forward_state,
)


def run(cfg: DictConfig) -> None:
    model_name = cfg.model.name
    is_ensemble = bool(cfg.run.ensemble)
    is_time_varying = bool(cfg.run.time_varying)
    num_steps = int(cfg.run.num_steps)
    if is_ensemble and is_time_varying:
        raise ValueError(
            "run.ensemble and run.time_varying are mutually exclusive "
            "(time-varying inflow is a single-member ground-truth run)."
        )
    results_dir = resolve_results_dir(cfg)
    param_names = resolve_parameter_schema(model_name)

    forward_model = instantiate(cfg.model.forward_model, results_dir=results_dir)
    instantiate(cfg.model.prepare, forward_model=forward_model)
    clean_outputs(model_name=model_name, forward_model=forward_model)

    if is_ensemble:
        runner = instantiate(cfg.model.ensemble_model, forward_model=forward_model)
        configure_failure_policy(runner, cfg.ensemble.failure)
        for member in runner.ensemble_forward_models:
            clean_outputs(model_name=model_name, forward_model=member)
        params = create_parameter_ensemble(
            model_name=model_name,
            prior_cfg=cfg.params.prior,
            ensemble_size=cfg.ensemble.ensemble_size,
            seed=cfg.esmda.seed,
            param_names=param_names,
        )

        def step(state):
            out = runner.run_ensemble(params=params, state=state, sim_name="state")
            return out if out is not None else runner.get_states()

    else:
        if is_time_varying:
            params = create_time_varying_true_params(
                model_name=model_name,
                tv_cfg=cfg.time_varying,
                true_cfg=cfg.params.true,
                external_cfg=cfg.params.external,
                simulation_time=cfg.time.simulation_time,
                num_time_points=int(cfg.time_varying.num_time_points),
                seed=cfg.esmda.seed,
            )
        else:
            params = create_true_params(model_name, cfg.params.true, param_names)

        def step(state):
            out = forward_model(params=params, state=state)
            return out if out is not None else forward_model.get_states()

    t1 = time.time()
    state = step(None)
    state_list = [state]
    for _ in range(num_steps - 1):
        state = step(state)
        state_list.append(state)
    if num_steps > 1:
        state = xarray.concat(state_list, dim="time", join="override")
    elapsed = time.time() - t1

    state = add_velocity_magnitude(state)

    print(f"Model: {model_name}{' (time-varying inflow)' if is_time_varying else ''}")
    if is_ensemble:
        print(f"Ensemble size: {cfg.ensemble.ensemble_size}")
    if num_steps > 1:
        print(f"Rollout: {num_steps} steps")
    print(f"Elapsed: {elapsed:.2f} seconds")
    print(f"Dims: {dict(state.sizes)}")
    print(f"Vars: {list(state.data_vars)}")
    if is_time_varying:
        print(
            f"Inflow angle: {params['inflow_angle'].values[0]:.1f} -> "
            f"{params['inflow_angle'].values[-1]:.1f} deg"
        )
        print(
            f"Velocity magnitude: {params['velocity_magnitude'].values[0]:.1f} -> "
            f"{params['velocity_magnitude'].values[-1]:.1f} m/s"
        )

    if is_time_varying:
        # Persist the simulated state and the parameter profile that produced
        # it: this is the ground-truth artifact a downstream twin / DA
        # experiment reads, so it is written regardless of the viz toggle.
        out_dir = (
            resolve_output_dir(cfg, "forward_model") / f"{model_name}_time_varying"
        )
        out_dir.mkdir(parents=True, exist_ok=True)
        state.to_netcdf(out_dir / "state.nc")
        params.to_netcdf(out_dir / "params.nc")
        print(f"Saved state -> {out_dir / 'state.nc'}")
        print(f"Saved params -> {out_dir / 'params.nc'}")
        if not cfg.run.skip_viz:
            visualize_forward_state(
                state, model_name, out_dir, f"{model_name} time-varying"
            )
            plot_derived_inflow_angle(state, params, out_dir)
        return

    if not cfg.run.skip_viz:
        suffix = model_name
        if is_ensemble:
            suffix += "_ensemble"
        if num_steps > 1:
            suffix += "_rollout"
        out_dir = resolve_output_dir(cfg, "forward_model") / suffix

        viz_state = state.mean(dim="ensemble") if "ensemble" in state.dims else state
        label = model_name
        if is_ensemble:
            label += " ensemble (mean)"
        if num_steps > 1:
            label += " rollout"
        visualize_forward_state(viz_state, model_name, out_dir, label)


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    run(cfg)


if __name__ == "__main__":
    main()
