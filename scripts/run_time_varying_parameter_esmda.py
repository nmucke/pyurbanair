import argparse
import csv
import pathlib
import sys
import time

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from data_assimilation.smoothing.esmda import TimeVaryingParameterESMDA
from pyurbanair.plotting import (
    plot_state_init_and_terminal,
    plot_true_vs_estimated_state,
)
from pyurbanair.utils.animation_utils import _visualize_state_history
from pyurbanair.utils.da_metrics import (
    per_knot_crps,
    per_knot_error,
    per_knot_in_band,
    per_knot_spread,
    summary_scalars,
)
from pyurbanair.utils.run_utils import get_ensemble_mean_field

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from scripts import config


def _plot_time_varying_params(
    params_history: "xarray.Dataset",
    true_params: "xarray.Dataset",
    time_coords: np.ndarray,
    output_path: pathlib.Path,
) -> None:
    """Plot true vs estimated time-varying parameters.

    For each parameter, the true profile is shown as a solid line and the
    final ESMDA step's ensemble mean is shown with a shaded +/- 1 std band.
    """
    import xarray  # noqa: F811 – local import to keep type hint lazy

    param_names = [
        name for name in true_params.data_vars if "time" in true_params[name].dims
    ]
    n_params = len(param_names)
    fig, axes = plt.subplots(n_params, 1, figsize=(8, 4 * n_params), squeeze=False)

    for ax, name in zip(axes[:, 0], param_names):
        true_vals = np.asarray(true_params[name].values)
        ax.plot(time_coords, true_vals, color="C0", linewidth=2, label="True")

        # Use the final ESMDA step
        final = params_history[name].isel(esmda_step=-1)
        ens_mean = np.asarray(final.mean(dim="ensemble").values)
        ens_std = np.asarray(final.std(dim="ensemble").values)

        ax.plot(time_coords, ens_mean, color="C1", linewidth=2, label="Estimated mean")
        ax.fill_between(
            time_coords,
            ens_mean - ens_std,
            ens_mean + ens_std,
            color="C1",
            alpha=0.3,
            label="Estimated std",
        )

        ax.set_xlabel("Time [s]")
        ax.set_ylabel(name)
        ax.legend()
        ax.set_title(name)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved time-varying parameter plot to {output_path}")


def _compute_time_varying_metrics(
    params_history: "xarray.Dataset",
    true_params: "xarray.Dataset",
    time_coords: np.ndarray,
) -> tuple[list[dict], list[dict]]:
    """Compute per-knot and per-step summary metrics.

    Returns ``(rows, summary_rows)``: long-format per-knot records and
    per-step summary records, ready for CSV writing.
    """
    param_names = [
        name for name in true_params.data_vars if "time" in true_params[name].dims
    ]
    rows: list[dict] = []
    summary_rows: list[dict] = []
    n_steps = int(params_history.sizes["esmda_step"])
    for k in range(n_steps):
        for name in param_names:
            ens = np.asarray(
                params_history[name].isel(esmda_step=k).transpose("ensemble", "time").values
            )
            truth = np.asarray(true_params[name].values)
            err = per_knot_error(ens, truth)
            spr = per_knot_spread(ens)
            crps = per_knot_crps(ens, truth)
            band = per_knot_in_band(ens, truth)
            for t_idx, t in enumerate(time_coords):
                rows.append(
                    {
                        "esmda_step": k,
                        "parameter": name,
                        "time": float(t),
                        "error": float(err[t_idx]),
                        "spread": float(spr[t_idx]),
                        "crps": float(crps[t_idx]),
                        "in_band": int(bool(band[t_idx])),
                    }
                )
            summary = summary_scalars(ens, truth)
            summary_rows.append({"esmda_step": k, "parameter": name, **summary})
    return rows, summary_rows


def _write_metrics_csv(rows: list[dict], path: pathlib.Path) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _print_summary(summary_rows: list[dict]) -> None:
    by_step: dict[int, list[dict]] = {}
    for row in summary_rows:
        by_step.setdefault(int(row["esmda_step"]), []).append(row)
    for step in sorted(by_step):
        print(f"--- ESMDA step {step} ---")
        for row in by_step[step]:
            print(
                f"  {row['parameter']:20s} "
                f"rmse={row['time_avg_error']:.4f}  "
                f"spread={row['time_avg_spread']:.4f}  "
                f"crps={row['mean_crps']:.4f}  "
                f"coverage={row['coverage']:.2f}"
            )


def _plot_time_varying_metrics(
    params_history: "xarray.Dataset",
    true_params: "xarray.Dataset",
    time_coords: np.ndarray,
    output_path: pathlib.Path,
) -> None:
    """Per-parameter diagnostic: error/spread/CRPS/in-band over time, one
    line per ESMDA step (color-graded so step 0 is light, final is dark)."""
    param_names = [
        name for name in true_params.data_vars if "time" in true_params[name].dims
    ]
    n_params = len(param_names)
    n_steps = int(params_history.sizes["esmda_step"])
    metric_titles = ["|mean - truth|", "ensemble spread", "CRPS", "in 90% band"]
    fig, axes = plt.subplots(
        n_params, 4, figsize=(16, 3.5 * n_params), squeeze=False
    )
    cmap = plt.get_cmap("viridis")
    for row_idx, name in enumerate(param_names):
        truth = np.asarray(true_params[name].values)
        for k in range(n_steps):
            ens = np.asarray(
                params_history[name].isel(esmda_step=k).transpose("ensemble", "time").values
            )
            err = per_knot_error(ens, truth)
            spr = per_knot_spread(ens)
            crps = per_knot_crps(ens, truth)
            band = per_knot_in_band(ens, truth).astype(float)
            shade = 0.2 + 0.8 * (k / max(n_steps - 1, 1))
            color = cmap(shade)
            label = f"step {k}"
            axes[row_idx, 0].plot(time_coords, err, color=color, label=label)
            axes[row_idx, 1].plot(time_coords, spr, color=color, label=label)
            axes[row_idx, 2].plot(time_coords, crps, color=color, label=label)
            axes[row_idx, 3].plot(
                time_coords, band, "o-", color=color, label=label, alpha=0.7
            )
        for col, title in enumerate(metric_titles):
            ax = axes[row_idx, col]
            ax.set_xlabel("Time [s]")
            ax.set_title(f"{name}: {title}")
            if col == 3:
                ax.set_ylim(-0.1, 1.1)
        axes[row_idx, 0].legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved time-varying metrics plot to {output_path}")


def _apply_config_overrides(args: argparse.Namespace) -> None:
    """Apply CLI overrides to the imported ``config`` module in place.

    Must run before any forward-model construction so downstream callers
    see the override.
    """
    if args.esmda_num_steps is not None:
        config.ESMDA["num_steps"] = int(args.esmda_num_steps)
    if args.obs_error_std is not None:
        config.ESMDA["obs_error_std"] = float(args.obs_error_std)
    if args.truth_corr_length is not None:
        config.TIME_VARYING_PARAMS["truth_correlation_length"] = float(
            args.truth_corr_length
        )
    if args.prior_corr_length is not None:
        config.TIME_VARYING_PARAMS["prior_correlation_length"] = float(
            args.prior_corr_length
        )
    if args.obs_interval is not None:
        config.OBS["interval_size"] = int(args.obs_interval)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--truth-model", choices=["pylbm", "pyudales", "pypalm"], default="pylbm")
    parser.add_argument("--assim-model", choices=["pylbm", "pyudales", "pypalm"], default="pylbm")
    parser.add_argument("--skip-viz", action="store_true")
    parser.add_argument(
        "--results-dir",
        type=pathlib.Path,
        default=None,
        help="Override results directory for assimilation model outputs.",
    )
    parser.add_argument(
        "--num-par-time-points",
        type=int,
        default=None,
        help="Number of discrete time points for time-varying parameters. "
        "Defaults to config.TIME_VARYING_PARAMS['num_time_points'].",
    )
    parser.add_argument(
        "--esmda-num-steps",
        type=int,
        default=None,
        help="Override config.ESMDA['num_steps'] for sweeps.",
    )
    parser.add_argument(
        "--obs-error-std",
        type=float,
        default=None,
        help="Override config.ESMDA['obs_error_std'] for sweeps.",
    )
    parser.add_argument(
        "--truth-corr-length",
        type=float,
        default=None,
        help="Override config.TIME_VARYING_PARAMS['truth_correlation_length'].",
    )
    parser.add_argument(
        "--prior-corr-length",
        type=float,
        default=None,
        help="Override config.TIME_VARYING_PARAMS['prior_correlation_length'].",
    )
    parser.add_argument(
        "--obs-interval",
        type=int,
        default=None,
        help="Override config.OBS['interval_size'] for sweeps.",
    )
    args = parser.parse_args()
    _apply_config_overrides(args)

    num_time_points = (
        args.num_par_time_points
        if args.num_par_time_points is not None
        else config.TIME_VARYING_PARAMS["num_time_points"]
    )
    sim_time = config.TIME["simulation_time"]
    time_coords = jnp.linspace(0, sim_time, num_time_points)

    truth_model = config.create_forward_model(args.truth_model)
    config.prepare_forward_model(args.truth_model, truth_model)
    true_params = config.create_time_varying_true_params(
        args.truth_model, num_time_points
    )
    true_state = truth_model(params=true_params)
    if true_state is None:
        raise RuntimeError("Expected in-memory truth state.")

    truth_obs_op = config.create_observation_operator(args.truth_model)
    true_obs = jnp.asarray(truth_obs_op(true_state))

    C_D = config.create_C_D(true_obs.shape[0])
    rng_key = jax.random.PRNGKey(config.ESMDA["seed"])
    rng_key, subkey = jax.random.split(rng_key)
    true_obs = true_obs + jnp.sqrt(C_D) @ jax.random.normal(subkey, true_obs.shape)

    assim_results_dir = (
        pathlib.Path(args.results_dir) if args.results_dir is not None else None
    )
    assim_model = config.create_forward_model(
        args.assim_model,
        results_dir=assim_results_dir,
    )
    config.prepare_forward_model(args.assim_model, assim_model)

    ensemble_model = config.create_ensemble_forward_model(args.assim_model, assim_model)
    assim_obs_op = config.create_observation_operator(args.assim_model)
    params_ensemble = config.create_time_varying_parameter_ensemble(
        args.assim_model, num_time_points
    )

    esmda = TimeVaryingParameterESMDA(
        observation_operator=assim_obs_op,
        forward_model=ensemble_model,
        C_D=C_D,
        num_time_points=num_time_points,
        time_coords=time_coords,
        num_steps=config.ESMDA["num_steps"],
        alpha=config.ESMDA["num_steps"],
        rng_key=rng_key,
    )

    t1 = time.time()
    output = esmda(
        params=params_ensemble,
        observations=true_obs,
        return_params_history=True,
        return_state_history=True,
    )
    t2 = time.time()
    print(f"ESMDA time: {t2 - t1:.2f} seconds")
    ensemble_mean_field, _ = get_ensemble_mean_field(
        output=output,
        esmda=esmda,
        num_esmda_steps=int(config.ESMDA["num_steps"]),
        ensemble_size=int(config.ENSEMBLE["ensemble_size"]),
    )

    out_dir = config.BASE_RESULTS_DIR / "time_varying_parameter_esmda"
    out_dir.mkdir(parents=True, exist_ok=True)
    if isinstance(output, tuple):
        params_history, state_history = output
        params_history.to_netcdf(out_dir / "params_history.nc")
        state_history.to_netcdf(out_dir / "state_history.nc")
    else:
        params_history = output
        params_history.to_netcdf(out_dir / "params_history.nc")
    ensemble_mean_field.to_netcdf(out_dir / "state_mean_history.nc")

    metric_rows, summary_rows = _compute_time_varying_metrics(
        params_history=params_history,
        true_params=true_params,
        time_coords=np.asarray(time_coords),
    )
    _write_metrics_csv(metric_rows, out_dir / "time_varying_metrics.csv")
    _write_metrics_csv(summary_rows, out_dir / "summary_metrics.csv")
    _print_summary(summary_rows)

    if not args.skip_viz:
        obs_x, obs_y, _ = config.create_observation_points()
        plot_true_vs_estimated_state(
            true_state=true_state,
            estimated_state=ensemble_mean_field,
            output_path=out_dir / "state_comparison.png",
            obs_x=obs_x,
            obs_y=obs_y,
            z_level=0,
        )
        plot_state_init_and_terminal(
            true_state=true_state,
            estimated_state=ensemble_mean_field,
            output_path=out_dir / "state_init_and_terminal.png",
            obs_x=obs_x,
            obs_y=obs_y,
            z_level=0,
        )
        state_for_viz = (
            state_history if isinstance(output, tuple) else ensemble_mean_field
        )
        _visualize_state_history(
            state_history=state_for_viz,
            out_dir=out_dir,
            title_prefix="time_varying_parameter_esmda",
            z_level=0,
        )

        _plot_time_varying_params(
            params_history=params_history,
            true_params=true_params,
            time_coords=np.asarray(time_coords),
            output_path=out_dir / "time_varying_parameters.png",
        )
        _plot_time_varying_metrics(
            params_history=params_history,
            true_params=true_params,
            time_coords=np.asarray(time_coords),
            output_path=out_dir / "time_varying_metrics.png",
        )

    print(f"Saved outputs in {pathlib.Path(out_dir)}")


if __name__ == "__main__":
    main()
