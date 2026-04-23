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
from pyurbanair.utils.animation_utils import animate_rollout_state

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
    horizon ``[0, num_windows * sim_time]``.  The draw is evaluated on
    the union of all windows' time grids (sharing boundary points
    between adjacent windows) so that each per-window slice matches the
    ESMDA window grid ``linspace(w*sim_time, (w+1)*sim_time, N_t)``
    exactly and the profile is continuous across boundaries.

    Args:
        num_windows: Number of assimilation windows.
        num_time_points: Discrete parameter time points *per window*.
        sim_time: Duration of one window in seconds.
        truth_correlation_length: RBF kernel length scale for truth GP.
        rng_key: JAX random key.

    Returns:
        List of ``num_windows`` :class:`xarray.Dataset` objects, each with
        dims ``(time,)`` and time coordinates matching the ESMDA window
        grid for that window.
    """
    # Shared-boundary grid: window w spans indices
    # [w*(N_t-1) : w*(N_t-1) + N_t], so window w's last point == window
    # w+1's first point (both at t = (w+1)*sim_time).
    step = max(num_time_points - 1, 1)
    n_unique = num_windows * step + 1
    full_time = jnp.linspace(0, num_windows * sim_time, n_unique)

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

    # Each window's simulation runs with its own clock starting at t=0,
    # so assign LOCAL time coords [0, sim_time] to each window's dataset.
    # Absolute times are reconstructed at save/plot time from the window index.
    local_time = np.asarray(jnp.linspace(0.0, sim_time, num_time_points))

    datasets: list[xarray.Dataset] = []
    for w in range(num_windows):
        start = w * step
        end = start + num_time_points
        datasets.append(
            xarray.Dataset(
                data_vars={
                    "inflow_angle": ("time", np.asarray(inflow_angle[start:end])),
                    "velocity_magnitude": (
                        "time",
                        np.asarray(velocity_magnitude[start:end]),
                    ),
                },
                coords={"time": local_time},
            )
        )
    return datasets


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

        # Posterior per window: individual members + ensemble mean
        for w, post_ds in enumerate(posterior_params_list):
            t = np.asarray(post_ds.coords["time"].values)
            members = np.asarray(post_ds[name].transpose("time", "ensemble").values)
            ax.plot(t, members, color="C1", linewidth=0.8, alpha=0.25)
            ens_mean = members.mean(axis=1)
            mean_label = "Posterior mean" if w == 0 else None
            ax.plot(t, ens_mean, color="C1", linewidth=2, label=mean_label)

        # Extrapolated prior per window: individual members + ensemble mean
        for w, extrap_ds in enumerate(extrapolated_params_list):
            t = np.asarray(extrap_ds.coords["time"].values)
            members = np.asarray(extrap_ds[name].transpose("time", "ensemble").values)
            ax.plot(t, members, color="C2", linewidth=0.8, linestyle="--", alpha=0.25)
            ens_mean = members.mean(axis=1)
            mean_label = "Extrapolated posterior" if w == 0 else None
            ax.plot(
                t,
                ens_mean,
                color="C2",
                linewidth=1.5,
                linestyle="--",
                label=mean_label,
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
    parser.add_argument("--truth-model", choices=["pylbm", "pyudales"], default="pylbm")
    parser.add_argument("--assim-model", choices=["pylbm", "pyudales"], default="pylbm")
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
    parser.add_argument(
        "--results-dir",
        type=pathlib.Path,
        default=None,
        help="Override results directory for assimilation model outputs.",
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
    extrap_method = config.TIME_VARYING_PARAMS.get(
        "extrapolation_method", "linear_trend_gp"
    )
    slope_damping_time = config.TIME_VARYING_PARAMS.get("slope_damping_time", None)
    ar1_phi_max = config.TIME_VARYING_PARAMS.get("ar1_phi_max", 0.999)
    ou_phi_max = config.TIME_VARYING_PARAMS.get("ou_phi_max", 0.999)

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

    assim_results_dir = (
        pathlib.Path(args.results_dir) if args.results_dir is not None else None
    )
    assim_model = config.create_forward_model(
        args.assim_model, results_dir=assim_results_dir
    )
    config.prepare_forward_model(args.assim_model, assim_model)
    assim_model = config.create_rollout_forward_model(args.assim_model, assim_model)

    # ---- Initial parameter ensemble for window 0 -------------------------
    params_ensemble = config.create_time_varying_parameter_ensemble(
        args.assim_model, num_time_points
    )

    # ---- Observation setup ------------------------------------------------
    truth_obs_op = config.create_observation_operator(args.truth_model)
    ensemble_model = config.create_ensemble_forward_model(args.assim_model, assim_model)
    assim_obs_op = config.create_observation_operator(args.assim_model)

    # ---- Storage ----------------------------------------------------------
    true_state_list: list[xarray.Dataset] = []
    posterior_params_list: list[xarray.Dataset] = []
    extrapolated_params_list: list[xarray.Dataset] = []
    esmda_state_list: list[xarray.Dataset] = []

    # ---- State tracking for handoff between windows -----------------------
    true_state: xarray.Dataset | None = None
    esmda_final_state: xarray.Dataset | None = None
    C_D: jnp.ndarray | None = None
    esmda: TimeVaryingParameterESMDA | None = None

    # ---- Window loop ------------------------------------------------------
    for w in tqdm(range(num_windows), desc="Assimilation windows"):
        true_params_w = true_params_per_window[w]

        # --- Truth forward -------------------------------------------------
        true_state = truth_model(params=true_params_w, state=true_state)
        if true_state is None:
            raise RuntimeError("Expected in-memory truth rollout state.")
        true_state_list.append(true_state)

        # --- Noisy observations --------------------------------------------
        true_obs = jnp.asarray(truth_obs_op(true_state))
        if C_D is None:
            C_D = config.create_C_D(true_obs.shape[0])
        rng_key, subkey = jax.random.split(rng_key)
        noisy_obs = true_obs + jnp.sqrt(C_D) @ jax.random.normal(subkey, true_obs.shape)

        # --- Window time coords (local; each window's sim runs 0..sim_time) -
        local_time_coords = jnp.linspace(0.0, sim_time, num_time_points)

        # --- Construct ESMDA (local time coords are the same every window) -
        if esmda is None:
            esmda = TimeVaryingParameterESMDA(
                observation_operator=assim_obs_op,
                forward_model=ensemble_model,
                C_D=C_D,
                num_time_points=num_time_points,
                time_coords=local_time_coords,
                num_steps=config.ESMDA["num_steps"],
                alpha=config.ESMDA["num_steps"],
                rng_key=rng_key,
                pin_initial_time_point=False,
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
        posterior_params_list.append(posterior_params)
        esmda_state_list.append(esmda_final_state)

        # --- GP extrapolation to next window ------------------------------
        # Extrapolate forward to [sim_time, 2*sim_time] relative to the
        # posterior's local time axis (RBF kernel is translation-invariant),
        # then relabel the result to local [0, sim_time] for the next
        # window's forward model.
        if w < num_windows - 1:
            prediction_times = jnp.linspace(sim_time, 2.0 * sim_time, num_time_points)
            rng_key, extrap_key = jax.random.split(rng_key)
            extrapolated = extrapolate_parameters(
                posterior_params,
                prediction_times=prediction_times,
                method=extrap_method,
                correlation_length=prior_corr_length,
                include_std=False,
                slope_damping_time=slope_damping_time,
                ar1_phi_max=ar1_phi_max,
                rng_key=extrap_key,
                ou_phi_max=ou_phi_max,
            )
            extrapolated = extrapolated.assign_coords(
                time=np.asarray(local_time_coords)
            )
            extrapolated["velocity_magnitude"] = extrapolated[
                "velocity_magnitude"
            ].clip(min=0.1)
            params_ensemble = extrapolated
            extrapolated_params_list.append(params_ensemble)

    # ---- Save outputs -----------------------------------------------------
    out_dir = config.BASE_RESULTS_DIR / "time_varying_rollout_esmda"
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
    # extrapolated_params_list[w] is the prior for window w+1
    extrapolated_params_abs = [
        _shift_to_abs(ds, w + 1) for w, ds in enumerate(extrapolated_params_list)
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

    # Save extrapolated priors for inspection
    if extrapolated_params_abs:
        extrap_all = xarray.concat(extrapolated_params_abs, dim="time", join="override")
        extrap_all.to_netcdf(out_dir / "extrapolated_params.nc")

    # ---- Plotting ---------------------------------------------------------
    if not args.skip_viz:
        _plot_time_varying_params_rollout(
            true_params_list=true_params_abs,
            posterior_params_list=posterior_params_abs,
            extrapolated_params_list=extrapolated_params_abs,
            output_path=out_dir / "time_varying_parameters_rollout.png",
        )
        animate_rollout_state(
            true_state=true_state_all,
            esmda_state=esmda_state_all,
            output_path=out_dir / "rollout_animation.mp4",
            z_level=0,
        )

    print(f"Saved outputs in {pathlib.Path(out_dir)}")


if __name__ == "__main__":
    main()
