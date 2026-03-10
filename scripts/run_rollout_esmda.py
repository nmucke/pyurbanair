import argparse
import pathlib
import sys

import jax
import jax.numpy as jnp
import xarray
from data_assimilation.smoothing.esmda import StateAndParameterESMDA
from tqdm import tqdm

from pyurbanair.plotting import plot_rollout_time_evolution
from pyurbanair.utils.animation_utils import animate_rollout_state
from pyurbanair.utils.config_utils import load_init_conditions_for_esmda

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from scripts import config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--truth-model", choices=["pylbm", "pyudales"], default="pyudales"
    )
    parser.add_argument("--assim-model", choices=["pylbm", "pyudales"], default="pylbm")
    parser.add_argument(
        "--skip-viz",
        action="store_true",
        help="Skip plotting and animation outputs.",
    )
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
    ensemble_size = int(config.ENSEMBLE["ensemble_size"])
    num_assimilation_windows = int(config.ESMDA["num_assimilation_windows"])
    true_sim_id = (
        args.true_sim_id
        if args.true_sim_id is not None
        else int(config.ESMDA.get("true_sim_id", 0))
    )

    if use_init_conditions:
        truth_data = load_init_conditions_for_esmda(
            args.truth_model,
            ensemble_size=max(1, true_sim_id + 1),
            true_sim_id=true_sim_id,
        )
        if truth_data is None:
            truth_subdir = "lbm" if args.truth_model == "pylbm" else "udales"
            init_dir = pathlib.Path(config.ESMDA["init_conditions_dir"]) / truth_subdir
            raise FileNotFoundError(
                f"Truth init conditions not found or incomplete in {init_dir}. "
                f"Need params.nc and state_{true_sim_id}.nc"
            )
        _, _, true_params, true_init_state = truth_data

        assim_data = load_init_conditions_for_esmda(
            args.assim_model,
            ensemble_size,
            true_sim_id,
        )
        if assim_data is None:
            assim_subdir = "lbm" if args.assim_model == "pylbm" else "udales"
            init_dir = pathlib.Path(config.ESMDA["init_conditions_dir"]) / assim_subdir
            raise FileNotFoundError(
                f"Assim init conditions not found or incomplete in {init_dir}. "
                f"Need params.nc and state_0.nc .. state_{ensemble_size - 1}.nc"
            )
        init_state_ensemble, init_params_ensemble, _, _ = assim_data
    else:
        true_params = config.create_true_params(args.truth_model)
        true_init_state = None
        init_state_ensemble = None
        init_params_ensemble = None

    truth_model = config.create_forward_model(args.truth_model, rollout=True)
    config.prepare_forward_model(args.truth_model, truth_model)

    # No results_dir: states must stay in memory so we can feed them back each window
    assim_model = config.create_forward_model(args.assim_model, rollout=True)
    config.prepare_forward_model(args.assim_model, assim_model)

    if not use_init_conditions:
        assim_init_model = config.create_forward_model(args.assim_model, rollout=True)
        config.prepare_forward_model(args.assim_model, assim_init_model)
        assim_ref_state = assim_init_model(
            params=config.create_true_params(args.assim_model)
        )
        if assim_ref_state is None:
            raise RuntimeError("Expected in-memory assimilation rollout reference state.")
        init_state_ensemble = config.create_initial_state_ensemble(assim_ref_state)
        init_params_ensemble = config.create_parameter_ensemble(args.assim_model)

    if use_init_conditions:
        # Use the loaded init state directly; no need to run a warmup step
        true_state = true_init_state
    else:
        # Run one truth step to get the initial true state
        true_state = truth_model(params=true_params, state=None)
        if true_state is None:
            raise RuntimeError("Expected in-memory truth rollout state.")

    truth_obs_op = config.create_observation_operator(args.truth_model)
    _state_for_sample = (
        true_state.expand_dims(time=[0]) if "time" not in true_state.dims else true_state
    )
    true_obs_sample = jnp.asarray(truth_obs_op(_state_for_sample))
    C_D = config.create_C_D(true_obs_sample.shape[0])

    ensemble_model = config.create_ensemble_forward_model(args.assim_model, assim_model)
    assim_obs_op = config.create_observation_operator(args.assim_model)

    rng_key = jax.random.PRNGKey(config.ESMDA["seed"])

    esmda = StateAndParameterESMDA(
        observation_operator=assim_obs_op,
        forward_model=ensemble_model,
        C_D=C_D,
        num_steps=config.ESMDA["num_steps"],
        alpha=1 / config.ESMDA["num_steps"],
        rng_key=rng_key,
    )

    true_state_list = []
    true_params_list = [true_params]
    esmda_state_list = []
    esmda_params_list = [init_params_ensemble]

    for _ in tqdm(range(num_assimilation_windows)):
        # Perturb true params (zero perturbation by default)
        rng_key, subkey = jax.random.split(rng_key)
        vel_magnitude_perturbation = jax.random.normal(subkey) * 0.0
        rng_key, subkey = jax.random.split(rng_key)
        angle_perturbation = jax.random.normal(subkey) * 0.0

        true_params = xarray.Dataset(
            data_vars={
                "velocity_magnitude": (
                    [],
                    float(true_params.velocity_magnitude.values + vel_magnitude_perturbation),
                ),
                "inflow_angle": (
                    [],
                    float(true_params.inflow_angle.values + angle_perturbation),
                ),
            },
        )

        # Run truth simulation from previous state
        true_state = truth_model(params=true_params, state=true_state)
        if true_state is None:
            raise RuntimeError("Expected in-memory truth rollout state.")
        true_state_list.append(true_state)
        true_params_list.append(true_params)

        # Generate noisy observations
        true_obs = jnp.asarray(truth_obs_op(true_state))
        rng_key, subkey = jax.random.split(rng_key)
        true_obs = true_obs + jnp.sqrt(C_D) @ jax.random.normal(subkey, true_obs.shape)

        # Run ESMDA for this window
        state_input = (
            init_state_ensemble.isel(time=-1)
            if "time" in init_state_ensemble.dims
            else init_state_ensemble
        )
        init_params_ensemble, init_state_ensemble = esmda(
            state=state_input,
            params=init_params_ensemble,
            observations=true_obs,
            return_params_history=False,
            return_state_history=False,
        )
        esmda_state_list.append(init_state_ensemble)
        esmda_params_list.append(init_params_ensemble)

    out_dir = config.BASE_RESULTS_DIR / "rollout_esmda"
    out_dir.mkdir(parents=True, exist_ok=True)

    true_state_all = xarray.concat(true_state_list, dim="time", join="override")
    true_params_all = xarray.concat(true_params_list, dim="time", join="override")
    esmda_state_all = xarray.concat(esmda_state_list, dim="time", join="override")
    esmda_params_all = xarray.concat(esmda_params_list, dim="time", join="override")

    true_state_all.to_netcdf(out_dir / "true_state.nc")
    true_params_all.to_netcdf(out_dir / "true_params.nc")
    esmda_state_all.to_netcdf(out_dir / "esmda_state.nc")
    esmda_params_all.to_netcdf(out_dir / "esmda_params.nc")

    if not args.skip_viz:
        plot_rollout_time_evolution(
            esmda_params=esmda_params_all,
            true_params=true_params_all,
            esmda_state=esmda_state_all,
            true_state=true_state_all,
            output_path=out_dir / "rollout_time_evolution.png",
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
