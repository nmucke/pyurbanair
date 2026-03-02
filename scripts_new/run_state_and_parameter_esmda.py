import argparse
import pathlib
import sys

import jax
import jax.numpy as jnp
from data_assimilation.smoothing.esmda import StateAndParameterESMDA

from pyurbanair.utils.animation_utils import _visualize_state_history
from pyurbanair.utils.run_utils import get_ensemble_mean_field

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from scripts_new import config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--truth-model", choices=["pylbm", "pyudales"], default="pylbm")
    parser.add_argument("--assim-model", choices=["pylbm", "pyudales"], default="pylbm")
    parser.add_argument(
        "--skip-viz",
        action="store_true",
        help="Skip plotting and animation outputs.",
    )
    args = parser.parse_args()

    truth_model = config.create_forward_model(args.truth_model, rollout=False)
    config.prepare_forward_model(args.truth_model, truth_model)
    true_params = config.create_true_params(args.truth_model)
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
        config.BASE_RESULTS_DIR / "state_and_parameter_esmda" / "assim_states"
    )
    assim_model = config.create_forward_model(
        args.assim_model,
        rollout=False,
        results_dir=assim_results_dir,
    )
    config.prepare_forward_model(args.assim_model, assim_model)

    assim_ref_state = assim_model(params=config.create_true_params(args.assim_model))
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
        alpha=1 / config.ESMDA["num_steps"],
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
        num_esmda_steps=int(config.ESMDA["num_steps"]),
        ensemble_size=int(config.ESMDA["ensemble_size"]),
    )

    out_dir = config.BASE_RESULTS_DIR / "state_and_parameter_esmda"
    out_dir.mkdir(parents=True, exist_ok=True)
    state_for_viz = ensemble_mean_field
    if isinstance(output, tuple):
        params_history, state_history = output
        params_history.to_netcdf(out_dir / "params_history.nc")
        state_history.to_netcdf(out_dir / "state_history.nc")
        state_for_viz = state_history
    else:
        output.to_netcdf(out_dir / "output.nc")
    ensemble_mean_field.to_netcdf(out_dir / "state_mean_history.nc")

    if not args.skip_viz:
        _visualize_state_history(
            state_history=state_for_viz,
            out_dir=out_dir,
            title_prefix="state_and_parameter_esmda",
        )

    print(f"Saved outputs in {pathlib.Path(out_dir)}")


if __name__ == "__main__":
    main()
