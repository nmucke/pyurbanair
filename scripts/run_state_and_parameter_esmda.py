import argparse
import pathlib
import sys

import jax
import jax.numpy as jnp
from data_assimilation.smoothing.esmda import StateAndParameterESMDA

from pyurbanair.plotting import (
    plot_parameter_distributions,
    plot_state_init_and_terminal,
    plot_true_vs_estimated_state,
)
from pyurbanair.utils.animation_utils import _visualize_state_history
from pyurbanair.utils.config_utils import load_init_conditions_for_esmda
from pyurbanair.utils.run_utils import get_ensemble_mean_field

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from scripts import config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--truth-model", choices=["pylbm", "pyudales"], default="pylbm")
    parser.add_argument("--assim-model", choices=["pylbm", "pyudales"], default="pylbm")
    parser.add_argument("--skip-viz", action="store_true")
    parser.add_argument(
        "--init-conditions",
        action="store_true",
        help="Load init states and params from esmda_init_conditions/{lbm|udales}/",
    )
    parser.add_argument(
        "--true-sim-id",
        type=int,
        default=None,
        help="Ensemble member ID to use as truth (default: config ESMDA.true_sim_id)",
    )
    args = parser.parse_args()

    use_init_conditions = args.init_conditions

    ensemble_size = int(config.ENSEMBLE["ensemble_size"])  # type: ignore[call-overload]
    true_sim_id = (
        args.true_sim_id
        if args.true_sim_id is not None
        else int(config.ESMDA.get("true_sim_id", 0))  # type: ignore[call-overload]
    )

    if use_init_conditions:
        # Load truth init (true_params, true_init_state) from truth-model dir
        truth_data = load_init_conditions_for_esmda(
            args.truth_model,
            ensemble_size=max(1, true_sim_id + 1),
            true_sim_id=true_sim_id,
        )
        if truth_data is None:
            truth_subdir = "lbm" if args.truth_model == "pylbm" else "udales"
            init_dir = pathlib.Path(config.ESMDA["init_conditions_dir"]) / truth_subdir  # type: ignore[arg-type]
            raise FileNotFoundError(
                f"Truth init conditions not found or incomplete in {init_dir}. "
                f"Need params.nc and state_{true_sim_id}.nc"
            )
        _, _, true_params, true_init_state = truth_data

        # Load assim init (init_state_ensemble, init_params_ensemble) from assim-model dir
        assim_data = load_init_conditions_for_esmda(
            args.assim_model,
            ensemble_size,
            true_sim_id,
        )
        if assim_data is None:
            assim_subdir = "lbm" if args.assim_model == "pylbm" else "udales"
            init_dir = pathlib.Path(config.ESMDA["init_conditions_dir"]) / assim_subdir  # type: ignore[arg-type]
            raise FileNotFoundError(
                f"Assim init conditions not found or incomplete in {init_dir}. "
                f"Need params.nc and state_0.nc .. state_{ensemble_size - 1}.nc"
            )
        init_state_ensemble, init_params_ensemble, _, _ = assim_data
        rollout = True
    else:
        true_params = config.create_true_params(args.truth_model)
        init_params_ensemble = None
        init_state_ensemble = None
        true_init_state = None
        rollout = False

    truth_model_name = args.truth_model
    truth_model = config.create_forward_model(truth_model_name, rollout=rollout)
    config.prepare_forward_model(truth_model_name, truth_model)
    if use_init_conditions:
        true_state = truth_model(params=true_params, state=true_init_state)
    else:
        true_state = truth_model(params=true_params)
    if true_state is None:
        raise RuntimeError("Expected in-memory truth state.")

    truth_obs_op = config.create_observation_operator(truth_model_name)
    true_obs = jnp.asarray(truth_obs_op(true_state))
    C_D = config.create_C_D(true_obs.shape[0])

    rng_key = jax.random.PRNGKey(config.ESMDA["seed"])
    rng_key, subkey = jax.random.split(rng_key)
    true_obs = true_obs + jnp.sqrt(C_D) @ jax.random.normal(subkey, true_obs.shape)

    assim_results_dir = (
        config.BASE_RESULTS_DIR / "state_and_parameter_esmda" / "assim_states"
    )
    assim_model = config.create_forward_model(
        args.assim_model,
        rollout=rollout,
        results_dir=assim_results_dir,
    )
    config.prepare_forward_model(args.assim_model, assim_model)
    config.clean_forward_model_outputs(args.assim_model, assim_model)
    if use_init_conditions and args.assim_model == "pylbm":
        config.clean_forward_model_restarts(args.assim_model, assim_model)

    if not use_init_conditions:
        assim_model.set_results_dir(None)
        assim_ref_state = assim_model(
            params=config.create_true_params(args.assim_model)
        )
        assim_model.set_results_dir(assim_results_dir)
        config.clean_forward_model_outputs(args.assim_model, assim_model)
        if assim_ref_state is None:
            raise RuntimeError("Expected in-memory assimilation reference state.")
        init_state_ensemble = config.create_initial_state_ensemble(assim_ref_state)
        init_params_ensemble = config.create_parameter_ensemble(args.assim_model)

    ensemble_model = config.create_ensemble_forward_model(args.assim_model, assim_model)
    assim_obs_op = config.create_observation_operator(args.assim_model)

    esmda = StateAndParameterESMDA(
        observation_operator=assim_obs_op,
        forward_model=ensemble_model,
        C_D=C_D,
        num_steps=config.ESMDA["num_steps"],
        alpha=1 / config.ESMDA["num_steps"],  # type: ignore[operator]
        rng_key=rng_key,
    )

    output = esmda(
        state=init_state_ensemble,
        params=init_params_ensemble,
        observations=true_obs,
        return_params_history=True,
        return_state_history=True,
    )
    ensemble_mean_field, _ = get_ensemble_mean_field(
        output=output,
        esmda=esmda,
        num_esmda_steps=int(config.ESMDA["num_steps"]),  # type: ignore[call-overload]
        ensemble_size=int(config.ENSEMBLE["ensemble_size"]),  # type: ignore[call-overload]
    )

    out_dir = config.BASE_RESULTS_DIR / "state_and_parameter_esmda"
    out_dir.mkdir(parents=True, exist_ok=True)
    state_for_viz = ensemble_mean_field
    params_for_plot = None
    if isinstance(output, tuple):
        params_history, state_history = output
        params_history.to_netcdf(out_dir / "params_history.nc")
        state_history.to_netcdf(out_dir / "state_history.nc")
        state_for_viz = state_history
        params_for_plot = params_history
    else:
        output.to_netcdf(out_dir / "params_history.nc")
        params_for_plot = output
    ensemble_mean_field.to_netcdf(out_dir / "state_mean_history.nc")

    if params_for_plot is not None:
        plot_parameter_distributions(
            params_history=params_for_plot,
            true_params=true_params,
            output_path=out_dir / "parameter_distributions.png",
        )

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
        _visualize_state_history(
            state_history=state_for_viz,
            out_dir=out_dir,
            title_prefix="state_and_parameter_esmda",
            z_level=0,
        )

    print(f"Saved outputs in {pathlib.Path(out_dir)}")


if __name__ == "__main__":
    main()
