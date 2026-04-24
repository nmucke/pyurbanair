import argparse
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
    args = parser.parse_args()

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
        output.to_netcdf(out_dir / "params_history.nc")
    ensemble_mean_field.to_netcdf(out_dir / "state_mean_history.nc")

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

        if isinstance(output, tuple):
            _plot_time_varying_params(
                params_history=params_history,
                true_params=true_params,
                time_coords=np.asarray(time_coords),
                output_path=out_dir / "time_varying_parameters.png",
            )

    print(f"Saved outputs in {pathlib.Path(out_dir)}")


if __name__ == "__main__":
    main()
