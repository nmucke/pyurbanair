import pathlib
import sys

import pyurbanair.quiet_jax  # noqa: F401  (suppress JAX CPU-fallback noise; must precede `import jax`)

import hydra
import jax
import jax.numpy as jnp
from hydra.utils import instantiate
from omegaconf import DictConfig

from pyurbanair.plotting import (
    plot_parameter_distributions,
    plot_state_init_and_terminal,
    plot_true_vs_estimated_state,
)
from pyurbanair.config.hydra_helpers import (
    clean_outputs,
    configure_failure_policy,
    create_C_D,
    create_initial_state_ensemble,
    create_observation_operator,
    create_observation_points,
    create_parameter_ensemble,
    create_true_params,
    resolve_parameter_schema,
    make_rng_key,
    resolve_output_dir,
)
from pyurbanair.utils.animation_utils import _visualize_state_history
from pyurbanair.utils.run_utils import get_ensemble_mean_field

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))


def run(cfg: DictConfig) -> None:
    true_params = create_true_params(
        cfg.truth_model.name,
        cfg.params.true,
        resolve_parameter_schema(cfg.truth_model.name),
    )

    truth_model_name = cfg.truth_model.name
    truth_model = instantiate(cfg.truth_model.forward_model)
    instantiate(cfg.truth_model.prepare, forward_model=truth_model)

    clean_outputs(model_name=truth_model_name, forward_model=truth_model)
    true_state = truth_model(params=true_params)
    if true_state is None:
        raise RuntimeError("Expected in-memory truth state.")

    truth_obs_op = create_observation_operator(cfg.obs, cfg.truth_model.solver_name)
    true_obs = jnp.asarray(truth_obs_op(true_state))
    C_D = create_C_D(true_obs.shape[0], cfg.esmda.obs_error_std)

    rng_key = make_rng_key(cfg.esmda.seed)
    rng_key, subkey = jax.random.split(rng_key)
    true_obs = true_obs + jnp.sqrt(C_D) @ jax.random.normal(subkey, true_obs.shape)

    out_dir = resolve_output_dir(cfg, "state_and_parameter_esmda")
    assim_results_dir = out_dir / "assim_states"
    out_dir.mkdir(parents=True, exist_ok=True)
    assim_results_dir.mkdir(parents=True, exist_ok=True)
    assim_model = instantiate(
        cfg.assim_model.forward_model,
        results_dir=assim_results_dir,
    )
    instantiate(cfg.assim_model.prepare, forward_model=assim_model)
    clean_outputs(cfg.assim_model.name, assim_model)

    # Spin up the assim model once to seed the initial state ensemble.
    assim_model.set_results_dir(None)
    assim_ref_state = assim_model(
        params=create_true_params(
            cfg.assim_model.name,
            cfg.params.true,
            resolve_parameter_schema(cfg.assim_model.name),
        )
    )
    assim_model.set_results_dir(assim_results_dir)
    clean_outputs(cfg.assim_model.name, assim_model)
    if assim_ref_state is None:
        raise RuntimeError("Expected in-memory assimilation reference state.")
    init_state_ensemble = create_initial_state_ensemble(
        assim_ref_state,
        cfg.ensemble.ensemble_size,
    )
    init_params_ensemble = create_parameter_ensemble(
        model_name=cfg.assim_model.name,
        prior_cfg=cfg.params.prior,
        ensemble_size=cfg.ensemble.ensemble_size,
        seed=cfg.esmda.seed,
        param_names=resolve_parameter_schema(cfg.assim_model.name),
    )

    ensemble_model = instantiate(
        cfg.assim_model.ensemble_model,
        forward_model=assim_model,
    )
    configure_failure_policy(ensemble_model, cfg.ensemble.failure)
    assim_obs_op = create_observation_operator(cfg.obs, cfg.assim_model.solver_name)

    esmda = instantiate(
        cfg.esmda.smoother,
        observation_operator=assim_obs_op,
        forward_model=ensemble_model,
        C_D=C_D,
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
        num_esmda_steps=int(cfg.esmda.num_steps),  # type: ignore[call-overload]
        ensemble_size=int(cfg.ensemble.ensemble_size),
    )

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

    if not cfg.run.skip_viz:
        obs_x, obs_y, _ = create_observation_points(cfg.obs)
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


@hydra.main(version_base=None, config_path="../conf", config_name="run_state_and_parameter_esmda")
def main(cfg: DictConfig) -> None:
    run(cfg)


if __name__ == "__main__":
    main()
