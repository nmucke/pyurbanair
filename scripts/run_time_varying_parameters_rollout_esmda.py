"""Rollout ESMDA with time-varying parameters across multiple windows.

Each assimilation window uses :class:`TimeVaryingParameterESMDA` to
estimate time-varying inflow parameters.  Between windows the next
window's prior parameter ensemble is produced by a
:class:`pyurbanair.parameter_time_series.ParameterTimeSeries` instance
selected via the Hydra ``time_varying.method`` config: that object both
draws the initial-window prior and propagates the posterior into the next
window's prior.

Window 0 starts cold (``state=None``) with spin-up enabled.  Subsequent
windows warm-start from the previous window's final forecast state, with
spin-up disabled.
"""

import pathlib
import sys
from typing import Any

import hydra
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import xarray
from hydra.utils import instantiate
from omegaconf import DictConfig
from tqdm import tqdm

from pyurbanair.parameter_time_series import ParameterTimeSeries
from pyurbanair.plotting import plot_state_init_and_terminal
from pyurbanair.config.hydra_helpers import (
    build_truth_ts_model,
    configure_failure_policy,
    create_C_D,
    create_observation_operator,
    create_observation_points,
    make_rng_key,
    resolve_output_dir,
)
from pyurbanair.utils.animation_utils import animate_rollout_state

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

# ---------------------------------------------------------------------------
# Truth generation
# ---------------------------------------------------------------------------


def _generate_truth_params_all_windows(
    num_windows: int,
    num_time_points: int,
    sim_time: float,
    truth_ts_model: ParameterTimeSeries,
    rng_key: jax.random.PRNGKey,
) -> list[xarray.Dataset]:
    """Generate time-varying true parameters across all windows.

    A single ``sample_prior`` draw from ``truth_ts_model`` (with
    ``ensemble_size=1``) spans the full time horizon ``[0,
    num_windows * sim_time]`` on the union of all windows' time grids
    (sharing boundary points between adjacent windows), then is sliced
    into per-window datasets so that each slice matches the ESMDA
    window grid exactly and the profile is continuous across boundaries.

    Using a separate model instance for the truth (typically with a
    different correlation length than the assimilation prior) avoids
    the inverse crime of identical generative processes for truth and
    prior.

    Args:
        num_windows: Number of assimilation windows.
        num_time_points: Discrete parameter time points *per window*.
        sim_time: Duration of one window in seconds.
        truth_ts_model: Single-member ParameterTimeSeries instance used
            to draw the truth trajectory.
        rng_key: JAX random key.

    Returns:
        List of ``num_windows`` :class:`xarray.Dataset` objects, each
        with dims ``(time,)`` and time coordinates matching the ESMDA
        window grid for that window.
    """
    # Shared-boundary grid: window w spans indices
    # [w*(N_t-1) : w*(N_t-1) + N_t], so window w's last point == window
    # w+1's first point (both at t = (w+1)*sim_time).
    step = max(num_time_points - 1, 1)
    n_unique = num_windows * step + 1
    full_time = jnp.linspace(0, num_windows * sim_time, n_unique)

    full_ds = truth_ts_model.sample_prior(full_time, rng_key)
    # ensemble_size=1 by construction; squeeze the ensemble dim.
    full_ds = full_ds.isel(ensemble=0, drop=True)

    # Each window's simulation runs with its own clock starting at t=0,
    # so assign LOCAL time coords [0, sim_time] to each window's dataset.
    # Absolute times are reconstructed at save/plot time from the window index.
    local_time = np.asarray(jnp.linspace(0.0, sim_time, num_time_points))

    datasets: list[xarray.Dataset] = []
    for w in range(num_windows):
        start = w * step
        end = start + num_time_points
        data_vars = {
            name: ("time", np.asarray(full_ds[name].values[start:end]))
            for name in truth_ts_model.param_names
        }
        datasets.append(
            xarray.Dataset(data_vars=data_vars, coords={"time": local_time})
        )
    return datasets


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def _plot_time_varying_params_rollout(
    true_params_list: list[xarray.Dataset],
    posterior_params_list: list[xarray.Dataset],
    prior_params_list: list[xarray.Dataset],
    output_path: pathlib.Path,
) -> None:
    """Plot true vs prior vs posterior parameters across all windows.

    For each parameter the true profile is shown as a solid line, the
    per-window prior ensemble (cold-start for window 0, extrapolated for
    later windows) is shown in dashed green, and the posterior ensemble in
    orange.  Window boundaries are marked with vertical dashed lines.
    """
    param_names = [
        name
        for name in true_params_list[0].data_vars
        if "time" in true_params_list[0][name].dims
    ]
    n_params = len(param_names)
    fig, axes = plt.subplots(n_params, 1, figsize=(12, 4 * n_params), squeeze=False)

    for ax, name in zip(axes[:, 0], param_names):
        # True profile (concatenated across windows)
        true_times = np.concatenate(
            [np.asarray(ds.coords["time"].values) for ds in true_params_list]
        )
        true_vals = np.concatenate(
            [np.asarray(ds[name].values) for ds in true_params_list]
        )
        ax.plot(true_times, true_vals, color="C0", linewidth=2, label="True")

        # Prior per window: individual members + ensemble mean.  Window 0
        # is the cold-start prior; windows 1+ are extrapolated from the
        # previous posterior.
        for w, prior_ds in enumerate(prior_params_list):
            t = np.asarray(prior_ds.coords["time"].values)
            members = np.asarray(prior_ds[name].transpose("time", "ensemble").values)
            ax.plot(t, members, color="C2", linewidth=0.8, linestyle="--", alpha=0.50)
            ens_mean = members.mean(axis=1)
            mean_label = "Prior mean" if w == 0 else None
            ax.plot(
                t,
                ens_mean,
                color="C2",
                linewidth=2,
                linestyle="--",
                label=mean_label,
            )

        # Posterior per window: individual members + ensemble mean
        for w, post_ds in enumerate(posterior_params_list):
            t = np.asarray(post_ds.coords["time"].values)
            members = np.asarray(post_ds[name].transpose("time", "ensemble").values)
            ax.plot(t, members, color="C1", linewidth=0.8, alpha=0.50)
            ens_mean = members.mean(axis=1)
            mean_label = "Posterior mean" if w == 0 else None
            ax.plot(t, ens_mean, color="C1", linewidth=4, label=mean_label)

        # Window boundaries
        for w in range(1, len(true_params_list)):
            boundary = float(true_params_list[w].coords["time"].values[0])
            ax.axvline(boundary, color="gray", linestyle=":", linewidth=1)

        ax.set_xlabel("Time [s]")
        ax.set_ylabel(name)
        ax.legend()
        ax.set_title(name)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved time-varying parameter rollout plot to {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def run(cfg: DictConfig) -> None:
    # ---- Config -----------------------------------------------------------
    num_windows = int(cfg.esmda.num_assimilation_windows)
    num_time_points = int(cfg.time_varying.num_time_points)
    sim_time = cfg.time.simulation_time

    rng_key = make_rng_key(cfg.esmda.seed)

    # ---- Parameter time-series models ------------------------------------
    ts_model = instantiate(cfg.time_varying.prior_model)
    truth_ts_model = build_truth_ts_model(
        tv_cfg=cfg.time_varying,
        external_cfg=cfg.params.external,
        ensemble_size=1,
    )

    # ---- Truth parameters across all windows ------------------------------
    rng_key, subkey = jax.random.split(rng_key)
    true_params_per_window = _generate_truth_params_all_windows(
        num_windows=num_windows,
        num_time_points=num_time_points,
        sim_time=sim_time,
        truth_ts_model=truth_ts_model,
        rng_key=subkey,
    )

    # ---- Forward models ---------------------------------------------------
    truth_model = instantiate(cfg.truth_model.forward_model)
    instantiate(cfg.truth_model.prepare, forward_model=truth_model)

    assim_results_dir = (
        pathlib.Path(cfg.run.results_dir)
        if cfg.run.results_dir is not None
        else None
    )
    assim_model = instantiate(
        cfg.assim_model.forward_model,
        results_dir=assim_results_dir,
    )
    instantiate(cfg.assim_model.prepare, forward_model=assim_model)

    # ---- Initial parameter ensemble for window 0 -------------------------
    initial_time_coords = jnp.linspace(0.0, sim_time, num_time_points)
    rng_key, prior_key = jax.random.split(rng_key)
    params_ensemble = ts_model.sample_prior(initial_time_coords, prior_key)

    # ---- Observation setup ------------------------------------------------
    truth_obs_op = create_observation_operator(cfg.obs, cfg.truth_model.solver_name)
    ensemble_model = instantiate(
        cfg.assim_model.ensemble_model,
        forward_model=assim_model,
    )
    configure_failure_policy(ensemble_model, cfg.ensemble.failure)
    assim_obs_op = create_observation_operator(cfg.obs, cfg.assim_model.solver_name)

    # ---- Storage ----------------------------------------------------------
    # ``prior_params_list[w]`` is the prior ensemble for window ``w``:
    # cold-start from ``ts_model.sample_prior`` for window 0, and
    # ``ts_model.extrapolate(posterior_{w-1})`` for windows 1+.
    true_state_list: list[xarray.Dataset] = []
    prior_params_list: list[xarray.Dataset] = []
    posterior_params_list: list[xarray.Dataset] = []
    esmda_state_list: list[xarray.Dataset] = []

    # ---- State tracking for handoff between windows -----------------------
    true_state: xarray.Dataset | None = None
    esmda_final_state: xarray.Dataset | None = None
    C_D: jnp.ndarray | None = None
    esmda: Any | None = None

    # ---- Window loop ------------------------------------------------------
    for w in tqdm(range(num_windows), desc="Assimilation windows"):
        true_params_w = true_params_per_window[w]
        # Record the prior used for this window before ESMDA updates it.
        prior_params_list.append(params_ensemble)

        # --- Truth forward -------------------------------------------------
        true_state = truth_model(params=true_params_w, state=true_state)
        if true_state is None:
            raise RuntimeError("Expected in-memory truth rollout state.")
        true_state_list.append(true_state)

        # --- Noisy observations --------------------------------------------
        true_obs = jnp.asarray(truth_obs_op(true_state))
        if C_D is None:
            C_D = create_C_D(true_obs.shape[0], cfg.esmda.obs_error_std)
        rng_key, subkey = jax.random.split(rng_key)
        noisy_obs = true_obs + jnp.sqrt(C_D) @ jax.random.normal(subkey, true_obs.shape)

        # --- Window time coords (local; each window's sim runs 0..sim_time) -
        local_time_coords = jnp.linspace(0.0, sim_time, num_time_points)

        # --- Construct ESMDA (local time coords are the same every window) -
        if esmda is None:
            esmda = instantiate(
                cfg.esmda.smoother,
                observation_operator=assim_obs_op,
                forward_model=ensemble_model,
                C_D=C_D,
                time_coords=local_time_coords,
                rng_key=rng_key,
            )
        # Pin t=0 from window 1 onward so the Kalman update preserves the
        # continuity that the GP extrapolation established at each window
        # boundary. Window 0's prior t=0 is a free parameter (just a GP
        # draw), so we let ESMDA fit it.
        esmda.pin_initial_time_point = w > 0

        # --- Determine initial state for ESMDA -----------------------------
        if esmda_final_state is not None and "time" in esmda_final_state.dims:
            state_input = esmda_final_state.isel(time=-1)
        else:
            state_input = esmda_final_state  # None for window 0

        # --- Run ESMDA -----------------------------------------------------
        # ``_analysis`` returns either ``(params_history, final_state)`` (in
        # memory) or just ``params_history`` (when the ensemble forward model
        # has save_on_disk=True). The rollout flow needs the in-memory final
        # state to seed the next window, so reject the on-disk contract loudly
        # instead of letting it crash later as an opaque AttributeError on a
        # string return value.
        output = esmda(
            state=state_input,
            params=params_ensemble,
            observations=noisy_obs,
            return_params_history=True,
            return_state_history=False,
        )
        if not isinstance(output, tuple):
            raise RuntimeError(
                "Rollout ESMDA requires an in-memory ensemble forward model "
                "so the final window state can seed the next window. Got a "
                "single return value, which means the forward model is "
                "configured with save_on_disk=True."
            )
        params_history, esmda_final_state = output

        # Extract final posterior (last ESMDA step)
        posterior_params = params_history.isel(esmda_step=-1)
        posterior_params_list.append(posterior_params)
        esmda_state_list.append(esmda_final_state)

        # --- Build next window's prior ------------------------------------
        # Methods that fit-and-roll-forward (gp_linear_trend, ar1, OU) want
        # prediction times that start at the posterior's last training
        # point so the rollout is continuous; we extrapolate over
        # [sim_time, 2*sim_time] and relabel back to local [0, sim_time]
        # for the next window's forward model.  AR(2) relaxation is
        # translation-invariant in local time, so this works there too.
        if w < num_windows - 1:
            prediction_times = jnp.linspace(sim_time, 2.0 * sim_time, num_time_points)
            rng_key, extrap_key = jax.random.split(rng_key)
            extrapolated = ts_model.extrapolate(
                posterior_params,
                prediction_times=prediction_times,
                rng_key=extrap_key,
            )
            extrapolated = extrapolated.assign_coords(
                time=np.asarray(local_time_coords)
            )
            params_ensemble = extrapolated

    # ---- Save outputs -----------------------------------------------------
    out_dir = resolve_output_dir(
        cfg,
        f"time_varying_rollout_esmda_{cfg.truth_model.name}_{cfg.assim_model.name}",
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    # Each window's datasets carry LOCAL time coords [0, sim_time]; shift to
    # absolute times before concatenating so the global time axis is monotonic.
    def _shift_to_abs(ds: xarray.Dataset, w: int) -> xarray.Dataset:
        return ds.assign_coords(time=ds["time"].values + w * sim_time)

    true_state_abs = [_shift_to_abs(ds, w) for w, ds in enumerate(true_state_list)]
    esmda_state_abs = [_shift_to_abs(ds, w) for w, ds in enumerate(esmda_state_list)]
    true_params_abs = [
        _shift_to_abs(ds, w) for w, ds in enumerate(true_params_per_window)
    ]
    posterior_params_abs = [
        _shift_to_abs(ds, w) for w, ds in enumerate(posterior_params_list)
    ]
    prior_params_abs = [
        _shift_to_abs(ds, w) for w, ds in enumerate(prior_params_list)
    ]

    true_state_all = xarray.concat(true_state_abs, dim="time", join="override")
    esmda_state_all = xarray.concat(esmda_state_abs, dim="time", join="override")
    true_params_all = xarray.concat(true_params_abs, dim="time", join="override")
    posterior_params_all = xarray.concat(
        posterior_params_abs, dim="time", join="override"
    )

    true_state_all.to_netcdf(out_dir / "true_state.nc")
    esmda_state_all.to_netcdf(out_dir / "esmda_state.nc")
    true_params_all.to_netcdf(out_dir / "true_params.nc")
    posterior_params_all.to_netcdf(out_dir / "posterior_params.nc")

    prior_params_all = xarray.concat(prior_params_abs, dim="time", join="override")
    prior_params_all.to_netcdf(out_dir / "prior_params.nc")

    # ---- Plotting ---------------------------------------------------------
    if not cfg.run.skip_viz:
        _plot_time_varying_params_rollout(
            true_params_list=true_params_abs,
            posterior_params_list=posterior_params_abs,
            prior_params_list=prior_params_abs,
            output_path=out_dir / "time_varying_parameters_rollout.png",
        )
        obs_x, obs_y, _ = create_observation_points(cfg.obs)
        esmda_mean_all = (
            esmda_state_all.mean(dim="ensemble")
            if "ensemble" in esmda_state_all.dims
            else esmda_state_all
        )
        plot_state_init_and_terminal(
            true_state=true_state_all,
            estimated_state=esmda_mean_all,
            output_path=out_dir / "state_init_and_terminal.png",
            obs_x=obs_x,
            obs_y=obs_y,
            z_level=0,
        )
        animate_rollout_state(
            true_state=true_state_all,
            esmda_state=esmda_state_all,
            output_path=out_dir / "rollout_animation.mp4",
            z_level=0,
        )

    print(f"Saved outputs in {pathlib.Path(out_dir)}")


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    run(cfg)


if __name__ == "__main__":
    # Print tracebacks to ``sys.__stderr__`` (the original fd) before
    # re-raising. tqdm wraps ``sys.stderr``; on shutdown its buffered ``\r``
    # progress line can be the only thing that ever reaches the SLURM .err
    # file, hiding the actual traceback. Bypassing the wrapper guarantees
    # the traceback lands on disk even if tqdm swallows it.
    import sys as _sys
    import traceback as _traceback
    try:
        main()
    except BaseException:
        _traceback.print_exc(file=_sys.__stderr__)
        _sys.__stderr__.flush()
        raise
