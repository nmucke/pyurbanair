"""Rollout ESMDA with time-varying parameters and GP extrapolation between windows.

Each assimilation window uses :class:`TimeVaryingParameterESMDA` to estimate
time-varying inflow parameters.  Between windows the posterior parameter
ensemble is extrapolated to the next window's time coordinates using a
Gaussian process, providing a physically informed prior for the next window.

Window 0 starts cold (``state=None``) with spin-up enabled.  Subsequent
windows warm-start from the previous window's final forecast state, with
spin-up disabled.
"""

import argparse
import pathlib
import sys

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import xarray
from data_assimilation.smoothing.esmda import TimeVaryingParameterESMDA
from tqdm import tqdm

from pyurbanair.parameter_extrapolation import extrapolate_parameters

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from scripts import config


# ---------------------------------------------------------------------------
# Truth generation
# ---------------------------------------------------------------------------


def _generate_truth_params_all_windows(
    num_windows: int,
    num_time_points: int,
    sim_time: float,
    truth_correlation_length: float,
    rng_key: jax.random.PRNGKey,
) -> list[xarray.Dataset]:
    """Generate time-varying true parameters across all windows from a GP draw.

    A single Gaussian process draw (per parameter) spans the full time
    horizon ``[0, num_windows * sim_time]``.  The result is sliced into
    per-window datasets so that the truth profile is smooth across window
    boundaries.

    Args:
        num_windows: Number of assimilation windows.
        num_time_points: Discrete parameter time points *per window*.
        sim_time: Duration of one window in seconds.
        truth_correlation_length: RBF kernel length scale for truth GP.
        rng_key: JAX random key.

    Returns:
        List of ``num_windows`` :class:`xarray.Dataset` objects, each with
        dims ``(time,)`` and time coordinates local to that window.
    """
    total_points = num_windows * num_time_points
    full_time = jnp.linspace(0, num_windows * sim_time, total_points)

    rng_key, subkey = jax.random.split(rng_key)
    inflow_angle = config.sample_smooth_ensemble(
        rng_key=subkey,
        time_coords=full_time,
        mean=config.PARAM_PRIORS["inflow_angle_mean"],
        std=config.PARAM_PRIORS["inflow_angle_std"],
        ensemble_size=1,
        correlation_length=truth_correlation_length,
    )[:, 0]  # squeeze ensemble dim

    rng_key, subkey = jax.random.split(rng_key)
    velocity_magnitude = config.sample_smooth_ensemble(
        rng_key=subkey,
        time_coords=full_time,
        mean=config.PARAM_PRIORS["velocity_mean"],
        std=config.PARAM_PRIORS["velocity_std"],
        ensemble_size=1,
        correlation_length=truth_correlation_length,
    )[:, 0]
    velocity_magnitude = jnp.maximum(velocity_magnitude, 0.1)

    # Slice into per-window datasets
    datasets: list[xarray.Dataset] = []
    for w in range(num_windows):
        start = w * num_time_points
        end = (w + 1) * num_time_points
        window_time = full_time[start:end]
        datasets.append(
            xarray.Dataset(
                data_vars={
                    "inflow_angle": ("time", np.asarray(inflow_angle[start:end])),
                    "velocity_magnitude": (
                        "time",
                        np.asarray(velocity_magnitude[start:end]),
                    ),
                },
                coords={"time": np.asarray(window_time)},
            )
        )
    return datasets


# ---------------------------------------------------------------------------
# Boundary pinning
# ---------------------------------------------------------------------------


def _pin_boundary_values(
    params: xarray.Dataset,
    boundary_values: dict[str, jnp.ndarray],
) -> xarray.Dataset:
    """Overwrite the first time index of time-varying params with boundary values.

    This ensures continuity between consecutive assimilation windows by
    setting each parameter's leftmost time point to the rightmost value
    from the previous window's posterior.

    Args:
        params: Parameter ensemble with dims ``(time, ensemble)``.
        boundary_values: Maps parameter name to array of shape
            ``(ensemble,)`` — the rightmost values from the previous
            window's posterior.

    Returns:
        New Dataset with boundary values pinned at time index 0.
    """
    data_vars: dict = {}
    for name in params.data_vars:
        if name in boundary_values and "time" in params[name].dims:
            values = jnp.asarray(params[name].values)  # (time, ensemble)
            values = values.at[0].set(boundary_values[name])
            data_vars[name] = (params[name].dims, values)
        else:
            data_vars[name] = params[name]
    return xarray.Dataset(data_vars=data_vars, coords=params.coords)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def _plot_time_varying_params_rollout(
    true_params_list: list[xarray.Dataset],
    posterior_params_list: list[xarray.Dataset],
    extrapolated_params_list: list[xarray.Dataset],
    output_path: pathlib.Path,
) -> None:
    """Plot true vs estimated vs extrapolated parameters across all windows.

    For each parameter the true profile is shown as a solid line, posterior
    ensemble mean +/- 1 std as a shaded band, and the GP-extrapolated prior
    for each subsequent window as a dashed line with lighter shading.
    Window boundaries are marked with vertical dashed lines.
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

        # Posterior per window — individual ensemble trajectories + mean
        for w, post_ds in enumerate(posterior_params_list):
            t = np.asarray(post_ds.coords["time"].values)
            n_ens = post_ds.sizes["ensemble"]
            for e in range(n_ens):
                vals = np.asarray(post_ds[name].isel(ensemble=e).values)
                label = "Posterior members" if w == 0 and e == 0 else None
                ax.plot(t, vals, color="C1", linewidth=0.5, alpha=0.4, label=label)
        for w, post_ds in enumerate(posterior_params_list):
            t = np.asarray(post_ds.coords["time"].values)
            ens_mean = np.asarray(post_ds[name].mean(dim="ensemble").values)
            label = "Posterior mean" if w == 0 else None
            ax.plot(t, ens_mean, color="C3", linewidth=2.5, zorder=10, label=label)

        # Extrapolated prior per window — individual ensemble trajectories + mean
        for w, extrap_ds in enumerate(extrapolated_params_list):
            t = np.asarray(extrap_ds.coords["time"].values)
            n_ens = extrap_ds.sizes["ensemble"]
            for e in range(n_ens):
                vals = np.asarray(extrap_ds[name].isel(ensemble=e).values)
                label = "Extrapolated members" if w == 0 and e == 0 else None
                ax.plot(
                    t, vals, color="C2", linewidth=0.5, alpha=0.3,
                    linestyle="--", label=label,
                )
        for w, extrap_ds in enumerate(extrapolated_params_list):
            t = np.asarray(extrap_ds.coords["time"].values)
            ens_mean = np.asarray(extrap_ds[name].mean(dim="ensemble").values)
            label = "Extrapolated mean" if w == 0 else None
            ax.plot(
                t, ens_mean, color="C4", linewidth=2.5, linestyle="--",
                zorder=10, label=label,
            )

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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Rollout ESMDA with time-varying parameters and GP extrapolation."
    )
    parser.add_argument(
        "--truth-model", choices=["pylbm", "pyudales"], default="pylbm"
    )
    parser.add_argument(
        "--assim-model", choices=["pylbm", "pyudales"], default="pylbm"
    )
    parser.add_argument(
        "--skip-viz",
        action="store_true",
        help="Skip plotting outputs.",
    )
    parser.add_argument(
        "--num-par-time-points",
        type=int,
        default=None,
        help="Number of discrete time points per window for time-varying parameters. "
        "Defaults to config.TIME_VARYING_PARAMS['num_time_points'].",
    )
    args = parser.parse_args()

    # ---- Config -----------------------------------------------------------
    num_windows = int(config.ESMDA["num_assimilation_windows"])
    num_time_points = (
        args.num_par_time_points
        if args.num_par_time_points is not None
        else config.TIME_VARYING_PARAMS["num_time_points"]
    )
    sim_time = config.TIME["simulation_time"]
    prior_corr_length = config.TIME_VARYING_PARAMS["prior_correlation_length"]
    truth_corr_length = config.TIME_VARYING_PARAMS["truth_correlation_length"]

    rng_key = jax.random.PRNGKey(config.ESMDA["seed"])

    # ---- Truth parameters across all windows ------------------------------
    rng_key, subkey = jax.random.split(rng_key)
    true_params_per_window = _generate_truth_params_all_windows(
        num_windows=num_windows,
        num_time_points=num_time_points,
        sim_time=sim_time,
        truth_correlation_length=truth_corr_length,
        rng_key=subkey,
    )

    # ---- Forward models ---------------------------------------------------
    truth_model = config.create_forward_model(args.truth_model)
    config.prepare_forward_model(args.truth_model, truth_model)
    truth_model = config.create_rollout_forward_model(args.truth_model, truth_model)

    # No results_dir: states stay in memory for state handoff
    assim_model = config.create_forward_model(args.assim_model)
    config.prepare_forward_model(args.assim_model, assim_model)
    assim_model = config.create_rollout_forward_model(args.assim_model, assim_model)

    # ---- Initial parameter ensemble for window 0 -------------------------
    params_ensemble = config.create_time_varying_parameter_ensemble(
        args.assim_model, num_time_points
    )

    # ---- Run truth for window 0 (cold start with spinup) ------------------
    true_state = truth_model(params=true_params_per_window[0], state=None)
    if true_state is None:
        raise RuntimeError("Expected in-memory truth rollout state.")

    # ---- Observation setup ------------------------------------------------
    truth_obs_op = config.create_observation_operator(args.truth_model)
    true_obs_sample = jnp.asarray(truth_obs_op(true_state))
    C_D = config.create_C_D(true_obs_sample.shape[0])

    ensemble_model = config.create_ensemble_forward_model(args.assim_model, assim_model)
    assim_obs_op = config.create_observation_operator(args.assim_model)

    # ---- ESMDA instance ---------------------------------------------------
    window_0_time_coords = jnp.linspace(0, sim_time, num_time_points)
    esmda = TimeVaryingParameterESMDA(
        observation_operator=assim_obs_op,
        forward_model=ensemble_model,
        C_D=C_D,
        num_time_points=num_time_points,
        time_coords=window_0_time_coords,
        num_steps=config.ESMDA["num_steps"],
        alpha=config.ESMDA["num_steps"],
        rng_key=rng_key,
    )

    # ---- Storage ----------------------------------------------------------
    true_state_list: list[xarray.Dataset] = []
    true_params_list: list[xarray.Dataset] = list(true_params_per_window)
    posterior_params_list: list[xarray.Dataset] = []
    extrapolated_params_list: list[xarray.Dataset] = []
    esmda_state_list: list[xarray.Dataset] = []

    # ---- State tracking for handoff between windows -----------------------
    esmda_final_state: xarray.Dataset | None = None
    boundary_values: dict[str, jnp.ndarray] = {}

    # ---- Window loop ------------------------------------------------------
    for w in tqdm(range(num_windows), desc="Assimilation windows"):
        true_params_w = true_params_per_window[w]

        # --- Truth forward -------------------------------------------------
        if w == 0:
            # Already computed above for observation setup
            pass
        else:
            true_state = truth_model(params=true_params_w, state=true_state)
            if true_state is None:
                raise RuntimeError("Expected in-memory truth rollout state.")
        true_state_list.append(true_state)

        # --- Noisy observations --------------------------------------------
        true_obs = jnp.asarray(truth_obs_op(true_state))
        rng_key, subkey = jax.random.split(rng_key)
        noisy_obs = true_obs + jnp.sqrt(C_D) @ jax.random.normal(
            subkey, true_obs.shape
        )

        # --- Update ESMDA time coords for this window ----------------------
        window_time_coords = jnp.linspace(
            w * sim_time, (w + 1) * sim_time, num_time_points
        )
        esmda.time_coords = window_time_coords

        # --- Determine initial state for ESMDA -----------------------------
        if esmda_final_state is not None and "time" in esmda_final_state.dims:
            state_input = esmda_final_state.isel(time=-1)
        else:
            state_input = esmda_final_state  # None for window 0

        # --- Run ESMDA -----------------------------------------------------
        output = esmda(
            state=state_input,
            params=params_ensemble,
            observations=noisy_obs,
            return_params_history=True,
            return_state_history=False,
        )
        params_history, esmda_final_state = output

        # Extract final posterior (last ESMDA step)
        posterior_params = params_history.isel(esmda_step=-1)

        # Pin leftmost posterior point to previous window's rightmost value
        # to ensure exact continuity across window boundaries.
        if boundary_values:
            posterior_params = _pin_boundary_values(posterior_params, boundary_values)

        posterior_params_list.append(posterior_params)
        esmda_state_list.append(esmda_final_state)

        # Store rightmost posterior values for next window's boundary pinning
        boundary_values = {
            name: jnp.asarray(posterior_params[name].isel(time=-1).values)
            for name in posterior_params.data_vars
            if "time" in posterior_params[name].dims
        }

        print(f"Window {w} completed")

        # --- GP extrapolation to next window's time coords -----------------
        if w < num_windows - 1:
            next_time_coords = jnp.linspace(
                (w + 1) * sim_time, (w + 2) * sim_time, num_time_points
            )
            params_ensemble = extrapolate_parameters(
                posterior_params,
                prediction_times=next_time_coords,
                correlation_length=prior_corr_length,
                include_std=False,
            )
            # Pin extrapolated prior's leftmost point to previous posterior
            params_ensemble = _pin_boundary_values(params_ensemble, boundary_values)
            # Clamp velocity to avoid C_u=0 in LBM (same guard as
            # create_time_varying_parameter_ensemble).
            if "velocity_magnitude" in params_ensemble.data_vars:
                params_ensemble["velocity_magnitude"] = params_ensemble[
                    "velocity_magnitude"
                ].clip(min=0.1)
            extrapolated_params_list.append(params_ensemble)

    # ---- Save outputs -----------------------------------------------------
    out_dir = config.BASE_RESULTS_DIR / "time_varying_rollout_esmda"
    out_dir.mkdir(parents=True, exist_ok=True)

    true_state_all = xarray.concat(true_state_list, dim="time", join="override")
    esmda_state_all = xarray.concat(esmda_state_list, dim="time", join="override")

    true_params_all = xarray.concat(true_params_list, dim="time", join="override")
    posterior_params_all = xarray.concat(
        posterior_params_list, dim="time", join="override"
    )

    true_state_all.to_netcdf(out_dir / "true_state.nc")
    esmda_state_all.to_netcdf(out_dir / "esmda_state.nc")
    true_params_all.to_netcdf(out_dir / "true_params.nc")
    posterior_params_all.to_netcdf(out_dir / "posterior_params.nc")

    # Save extrapolated priors for inspection
    if extrapolated_params_list:
        extrap_all = xarray.concat(
            extrapolated_params_list, dim="time", join="override"
        )
        extrap_all.to_netcdf(out_dir / "extrapolated_params.nc")

    # ---- Plotting ---------------------------------------------------------
    if not args.skip_viz:
        _plot_time_varying_params_rollout(
            true_params_list=true_params_list,
            posterior_params_list=posterior_params_list,
            extrapolated_params_list=extrapolated_params_list,
            output_path=out_dir / "time_varying_parameters_rollout.png",
        )

    print(f"Saved outputs in {pathlib.Path(out_dir)}")


if __name__ == "__main__":
    main()
