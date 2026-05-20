import pathlib
import sys

import pyurbanair.quiet_jax  # noqa: F401  (suppress JAX CPU-fallback noise; must precede `import jax`)

import hydra
import jax
import jax.numpy as jnp
import xarray
from hydra.utils import instantiate
from omegaconf import DictConfig
from tqdm import tqdm

from pyurbanair.plotting import plot_rollout_time_evolution
from pyurbanair.config.hydra_helpers import (
    clean_outputs,
    configure_failure_policy,
    create_C_D,
    create_initial_state_ensemble,
    create_observation_operator,
    create_parameter_ensemble,
    create_true_params,
    make_rng_key,
    resolve_output_dir,
)
from pyurbanair.utils.animation_utils import animate_rollout_state

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))


def run(cfg: DictConfig) -> None:
    num_assimilation_windows = int(cfg.esmda.num_assimilation_windows)

    true_params = create_true_params(cfg.truth_model.name, cfg.params.true)

    truth_model = instantiate(cfg.truth_model.forward_model)
    instantiate(cfg.truth_model.prepare, forward_model=truth_model)

    # No results_dir: states must stay in memory so we can feed them back each window
    assim_model = instantiate(cfg.assim_model.forward_model)
    instantiate(cfg.assim_model.prepare, forward_model=assim_model)

    # Spin up the assim model once to build the initial state ensemble.
    assim_ref_state = assim_model(
        params=create_true_params(cfg.assim_model.name, cfg.params.true)
    )
    # Drop spin-up side-effects (per-iter NetCDF dumps, warm-start seed files
    # on pylbm) before the per-window loop reuses the same model.
    clean_outputs(cfg.assim_model.name, assim_model)
    if assim_ref_state is None:
        raise RuntimeError("Expected in-memory assimilation rollout reference state.")
    # Every member starts from the same spin-up trajectory; parameter spread
    # is what drives trajectory divergence on the first window.
    init_state_ensemble = create_initial_state_ensemble(
        assim_ref_state,
        cfg.ensemble.ensemble_size,
    )
    init_params_ensemble = create_parameter_ensemble(
        model_name=cfg.assim_model.name,
        prior_cfg=cfg.params.prior,
        ensemble_size=cfg.ensemble.ensemble_size,
        seed=cfg.esmda.seed,
    )

    # Run one truth step to get the initial true state
    true_state = truth_model(params=true_params, state=None)
    if true_state is None:
        raise RuntimeError("Expected in-memory truth rollout state.")

    truth_obs_op = create_observation_operator(cfg.obs, cfg.truth_model.solver_name)
    _state_for_sample = (
        true_state.expand_dims(time=[0])
        if "time" not in true_state.dims
        else true_state
    )
    true_obs_sample = jnp.asarray(truth_obs_op(_state_for_sample))
    C_D = create_C_D(true_obs_sample.shape[0], cfg.esmda.obs_error_std)

    ensemble_model = instantiate(
        cfg.assim_model.ensemble_model,
        forward_model=assim_model,
    )
    configure_failure_policy(ensemble_model, cfg.ensemble.failure)
    assim_obs_op = create_observation_operator(cfg.obs, cfg.assim_model.solver_name)

    rng_key = make_rng_key(cfg.esmda.seed)

    esmda = instantiate(
        cfg.esmda.smoother,
        observation_operator=assim_obs_op,
        forward_model=ensemble_model,
        C_D=C_D,
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
                    float(
                        true_params.velocity_magnitude.values
                        + vel_magnitude_perturbation
                    ),
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

    out_dir = resolve_output_dir(cfg, "rollout_esmda")
    out_dir.mkdir(parents=True, exist_ok=True)

    true_state_all = xarray.concat(true_state_list, dim="time", join="override")
    true_params_all = xarray.concat(true_params_list, dim="time", join="override")
    esmda_state_all = xarray.concat(esmda_state_list, dim="time", join="override")
    esmda_params_all = xarray.concat(esmda_params_list, dim="time", join="override")

    true_state_all.to_netcdf(out_dir / "true_state.nc")
    true_params_all.to_netcdf(out_dir / "true_params.nc")
    esmda_state_all.to_netcdf(out_dir / "esmda_state.nc")
    esmda_params_all.to_netcdf(out_dir / "esmda_params.nc")

    if not cfg.run.skip_viz:
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


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    run(cfg)


if __name__ == "__main__":
    main()
