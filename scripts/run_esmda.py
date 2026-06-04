"""Run ESMDA: parameter-only / joint state+parameter, static / time-varying
parameters, single-window / multi-window rollout, with truth simulated inline
or loaded from disk.

This single script replaces the former
run_{parameter,state_and_parameter,rollout,time_varying_parameter,
time_varying_parameters_rollout}_esmda.py family. Three declarative axes select
the mode (see conf/run_esmda.yaml):

  * ``esmda/smoother=parameter|state_and_parameter|time_varying``
        which augmented state the Kalman update acts on.
  * ``params@prior_params=static|dynamic``
        static scalar parameters vs a time-varying (AR(2)) prior.
  * ``esmda.num_assimilation_windows=1|N``
        a single assimilation window vs an N-window rollout.

and the truth source:

  * ``run.ground_truth_dir=null``    simulate the truth inline (default).
  * ``run.ground_truth_dir=<path>``  load a state.nc/params.nc artifact written
                                     by run_forward_model.py run.time_varying=true.

Truth (states + parameters) for every window is generated up front, before any
assimilation runs. The window loop then consumes the precomputed truth.

Examples::

    python scripts/run_esmda.py esmda/smoother=parameter \
        params@prior_params=static params@truth_params=static_truth
    python scripts/run_esmda.py esmda/smoother=state_and_parameter \
        params@prior_params=static esmda.num_assimilation_windows=3
    python scripts/run_esmda.py esmda/smoother=time_varying \
        params@prior_params=dynamic params@truth_params=dynamic_truth \
        esmda.num_assimilation_windows=3
"""

import pathlib
import sys
import time

import pyurbanair.quiet_jax  # noqa: F401  (suppress JAX CPU-fallback noise; must precede `import jax`)

import hydra
import jax
import jax.numpy as jnp
import numpy as np
import xarray
from data_assimilation.smoothing.esmda import (
    StateAndParameterESMDA,
)
from hydra.utils import instantiate
from omegaconf import DictConfig
from tqdm import tqdm

from pyurbanair.config.hydra_helpers import (
    clean_outputs,
    create_observation_operator,
    create_observation_points,
    resolve_output_dir,
)
from pyurbanair.plotting import (
    plot_final_state_with_obs,
    plot_rollout_time_evolution,
)
from pyurbanair.utils.animation_utils import animate_rollout_state
from pyurbanair.utils.run_utils import add_velocity_magnitude, get_ensemble_mean_field


if __package__ is None or __package__ == "":
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))


# ---------------------------------------------------------------------------
# Small helpers (in the style of run_forward_model.py)
# ---------------------------------------------------------------------------

def _concat_windows(paths, sim_time, rebase, transform=None):
    """Concatenate per-window NetCDF files along ``time``.

    Files are opened one at a time and (optionally) reduced by ``transform``
    before being appended, so a memory-heavy per-window reduction (e.g. taking
    the ensemble mean of a state field) keeps only the reduced result instead of
    holding every window's full data in memory at once.

    ``rebase`` (used for the time-varying case) shifts each window's local time
    onto a single monotonic global axis (window ``w`` starts at ``w*sim_time``);
    the static case stacks the windows as-is, matching the old rollout script.
    """
    pieces = []
    for w, path in enumerate(paths):
        ds = xarray.open_dataset(path).load()
        if transform is not None:
            ds = transform(ds)
        if rebase and "time" in ds.dims:
            t = np.asarray(ds["time"].values, dtype=float)
            ds = ds.assign_coords(time=(t - t[0]) + w * sim_time)
        pieces.append(ds)
    if len(pieces) == 1:
        return pieces[0]
    return xarray.concat(pieces, dim="time", join="override")

# ---------------------------------------------------------------------------
# Output / plotting
# ---------------------------------------------------------------------------

def _finish_rollout(cfg, out_dir, windows_dir, num_windows, sim_time, is_dynamic):
    """Assemble rollout outputs from the per-window files in ``windows_dir``.

    State files are reduced one window at a time to the ensemble mean state plus
    the ensemble mean/std of the velocity magnitude, so the full ensemble is
    never held across windows. Parameter files are small enough to load in full,
    so the whole ensemble is kept in memory for distribution plotting.
    """
    state_paths = [
        windows_dir / f"window_{w}_posterior_state.nc" for w in range(num_windows)
    ]
    posterior_param_paths = [
        windows_dir / f"window_{w}_posterior_params.nc" for w in range(num_windows)
    ]
    prior_param_paths = [
        windows_dir / f"window_{w}_prior_params.nc" for w in range(num_windows)
    ]

    # Parameters: full ensemble in memory.
    posterior_params = _concat_windows(posterior_param_paths, sim_time, rebase=is_dynamic)
    prior_params = _concat_windows(prior_param_paths, sim_time, rebase=is_dynamic)
    posterior_params.to_netcdf(out_dir / "posterior_params.nc")
    prior_params.to_netcdf(out_dir / "prior_params.nc")

    # States: reduce each window's ensemble in a single pass before
    # concatenating, so the full ensemble is never held across windows. Keep the
    # mean state (u/v/w) plus the ensemble mean/std of the velocity magnitude
    # (``vel_mean``/``vel_std``).
    def _state_summary(ds):
        vmag = add_velocity_magnitude(ds)["vel_magnitude"]
        reduced = ds.mean(dim="ensemble")
        reduced["vel_mean"] = vmag.mean(dim="ensemble")
        reduced["vel_std"] = vmag.std(dim="ensemble")
        return reduced

    posterior_state = _concat_windows(
        state_paths, sim_time, rebase=is_dynamic, transform=_state_summary
    )
    posterior_state.to_netcdf(out_dir / "posterior_state_mean.nc")

    if cfg.run.skip_viz:
        return

    true_state = xarray.open_dataset(out_dir / "true_state.nc")
    true_params = xarray.open_dataset(out_dir / "true_params.nc")
    obs_x, obs_y, _ = create_observation_points(cfg.obs)

    # Boundaries between assimilation windows on the (rebased) global time axis,
    # used to lightly shade alternating windows in the parameter plot.
    window_edges = (
        list(np.linspace(0.0, sim_time * num_windows, num_windows + 1))
        if is_dynamic and num_windows > 1
        else None
    )

    plot_rollout_time_evolution(
        esmda_params=posterior_params,
        true_params=true_params,
        esmda_state=posterior_state,
        true_state=true_state,
        output_path=out_dir / "rollout_time_evolution.png",
        prior_params=prior_params,
        window_edges=window_edges,
    )
    animate_rollout_state(
        true_state=true_state,
        mean_vel=posterior_state["vel_mean"],
        std_vel=posterior_state["vel_std"],
        output_path=out_dir / "rollout_animation.mp4",
        z_level=0,
    )
    true_vel = add_velocity_magnitude(true_state)["vel_magnitude"]
    plot_final_state_with_obs(
        mean_vel=posterior_state["vel_mean"],
        std_vel=posterior_state["vel_std"],
        output_path=out_dir / "final_state_with_obs.png",
        true_vel=true_vel,
        obs_x=obs_x,
        obs_y=obs_y,
        z_level=0,
    )

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(cfg: DictConfig) -> None:
    import pdb
    num_windows = int(cfg.esmda.num_assimilation_windows)
    sim_time = float(cfg.time.simulation_time)
    ensemble_size = int(cfg.ensemble.ensemble_size)
    is_dynamic = "time_coords" in list(cfg.truth_params.keys())
    rng_key = jax.random.PRNGKey(cfg.esmda.seed)


    # --- True parameter sampler and state -----------------------------------------------------------
    if cfg.run.truth_dir is None:
        truth_sampler = instantiate(cfg.truth_params)
        if num_windows > 0:
            true_forward_model = instantiate(
                cfg.truth_model.forward_model, 
                results_dir=None, 
                simulation_time=sim_time * num_windows
            )
            if is_dynamic:
                # Match the assimilation grid: each window owns `num` control
                # points spaced `sim_time/(num-1)` (conf/params/dynamic.yaml's
                # per-window linspace(0, sim_time, num)). Sampling the truth on
                # the same spacing puts its control points on the window
                # boundaries instead of on a `num*num_windows`-point grid whose
                # spacing (sim_time*num_windows/(num*num_windows-1)) drifts off
                # the window edges, so truth and assim curves share x-locations.
                num = cfg.truth_params.time_coords.num
                time_coords = jnp.linspace(
                    0, sim_time * num_windows, (num - 1) * num_windows + 1
                )
                truth_sampler = instantiate(cfg.truth_params, time_coords=time_coords)
        else:
            true_forward_model = instantiate(
                cfg.truth_model.forward_model, 
                results_dir=None, 
                simulation_time=sim_time
            )

        true_params = truth_sampler.sample(1)

        instantiate(cfg.truth_model.prepare, forward_model=true_forward_model)
        clean_outputs(model_name=cfg.truth_model.name, forward_model=true_forward_model)
        true_state = true_forward_model(params=true_params.isel(ensemble=0))

    else:
        truth_dir = pathlib.Path(cfg.run.truth_dir)
        true_params = xarray.load_dataset(truth_dir / "params.nc")
        true_state = xarray.load_dataset(truth_dir / "state.nc")

    # --- Output and windows dir ---------------------------------------------------------


    out_dir = resolve_output_dir(cfg, "esmda")
    out_dir.mkdir(parents=True, exist_ok=True)
    windows_dir = out_dir / "windows"
    

    # --- Assimilation ensemble model -------------
    assim_results_dir = (
        pathlib.Path(cfg.run.results_dir) if cfg.run.results_dir is not None else None
    )
    assim_model = instantiate(
        cfg.assim_model.forward_model, results_dir=assim_results_dir
    )
    instantiate(cfg.assim_model.prepare, forward_model=assim_model)
    ensemble_model = instantiate(
        cfg.assim_model.ensemble_model, forward_model=assim_model
    )

    # --- Prior parameter sampler -----------------------------------------------------------
    prior_sampler = instantiate(cfg.prior_params)
    prior_params = prior_sampler.sample(ensemble_size)

    # --- Observation operator -----------------------------------------------------------
    truth_obs_op = create_observation_operator(cfg.obs, cfg.truth_model.solver_name)
    assim_obs_op = create_observation_operator(cfg.obs, cfg.assim_model.solver_name)

    # --- Observation error covariance ---------
    # Truth frames sit on a uniform grid over [0, sim_time*num_windows); each
    # window owns exactly `n_per_window` of them. Size C_D from the first such
    # block so it matches every window's observation vector (and the per-window
    # count the assimilation model emits).
    n_per_window = true_state.sizes["time"] // max(num_windows, 1)
    obs = jnp.asarray(truth_obs_op(true_state.isel(time=slice(0, n_per_window))))
    C_D = jnp.diag((cfg.esmda.obs_error_std**2) * jnp.ones(obs.shape[0]))

    # --- Save true state and params ---------
    true_state.to_netcdf(out_dir / f"true_state.nc")
    true_params.to_netcdf(out_dir / f"true_params.nc")
    del true_state

    # --- Smoother -----------------------------------------------------------
    rng_key, esmda_key = jax.random.split(rng_key)
    esmda = instantiate(
        cfg.esmda.smoother,
        observation_operator=assim_obs_op,
        forward_model=ensemble_model,
        C_D=C_D,
        rng_key=esmda_key,
    )
    include_state = isinstance(esmda, StateAndParameterESMDA)

    # --- Run ESMDA -----------------------------------------------------------
    state_input = None
    for window in tqdm(range(num_windows)):
        windows_dir.mkdir(parents=True, exist_ok=True)
        prior_params.to_netcdf(windows_dir / f"window_{window}_prior_params.nc")

        # Pin the t=0 knot from window 1 onward so the Kalman update preserves
        # the cross-window continuity that the GP extrapolation established at
        # each window boundary. Window 0's prior t=0 is just a cold-start GP
        # draw (over a spun-up flow), so ESMDA is free to fit it. Only the
        # time-varying smoother carries this flag.
        if hasattr(esmda, "pin_initial_time_point"):
            esmda.pin_initial_time_point = window > 0

        # Get observations in window and add noise. Select the w-th contiguous
        # block of frames (half-open) rather than an inclusive time-slice: the
        # frame at the next window's start (t=(window+1)*sim_time) must NOT be
        # double-counted, or interior windows would be one frame longer than the
        # assimilation model emits and the observation vector would misalign.
        window_true_state = xarray.open_dataset(out_dir / "true_state.nc").isel(
            time=slice(window * n_per_window, (window + 1) * n_per_window)
        )
        window_obs = jnp.asarray(truth_obs_op(window_true_state))
        rng_key, subkey = jax.random.split(rng_key)
        window_obs = window_obs + jnp.sqrt(C_D) @ jax.random.normal(subkey, window_obs.shape)
        
        # Sample posterior
        output = esmda(
            state=state_input,
            params=prior_params,
            observations=window_obs,
            return_params_history=True,
            return_state_history=False,
        )

        posterior_params = output[0].isel(esmda_step=-1)
        posterior_params.to_netcdf(windows_dir / f"window_{window}_posterior_params.nc")
        output[1].to_netcdf(windows_dir / f"window_{window}_posterior_state.nc")

        state_input = output[1].isel(time=-1)

        # Next window's prior: extrapolate the posterior
        if window < num_windows - 1:
            if is_dynamic:
                prediction_times = jnp.linspace(
                    sim_time, 2.0 * sim_time, cfg.prior_params.time_coords.num
                )
                rng_key, subkey = jax.random.split(rng_key)
                extrapolated = prior_sampler.extrapolate(
                    posterior_params, prediction_times, subkey
                )
                prior_params = extrapolated.assign_coords(
                    time=np.asarray(jnp.linspace(0.0, sim_time, cfg.prior_params.time_coords.num))
                )
            else:
                prior_params = posterior_params

        del output

    _finish_rollout(
        cfg,
        out_dir=out_dir,
        windows_dir=windows_dir,
        num_windows=num_windows,
        sim_time=sim_time,
        is_dynamic=is_dynamic,
    )
    print(f"Saved outputs in {out_dir}")


@hydra.main(version_base=None, config_path="../conf", config_name="run_esmda")
def main(cfg: DictConfig) -> None:
    run(cfg)


if __name__ == "__main__":
    main()
