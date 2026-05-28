import pathlib
import sys

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import hydra
import matplotlib.pyplot as plt
import xarray
from hydra.utils import instantiate
from omegaconf import DictConfig
from pyudales.utils.grid_utils import interpolate_grid
from pyurbanair.config.hydra_helpers import (
    clean_outputs,
    create_true_params,
    resolve_output_dir,
    resolve_parameter_schema,
)
from pyurbanair.utils.animation_utils import animate_state
from pyurbanair.utils.run_utils import add_velocity_magnitude, extract_2d_slice


def run(cfg: DictConfig) -> None:
    model_name = cfg.model.name
    num_steps = int(cfg.run.num_steps)
    results_dir = (
        pathlib.Path(cfg.run.results_dir) if cfg.run.results_dir is not None else None
    )

    forward_model = instantiate(cfg.model.forward_model, results_dir=results_dir)
    instantiate(cfg.model.prepare, forward_model=forward_model)
    clean_outputs(model_name=model_name, forward_model=forward_model)

    true_params = create_true_params(
        model_name,
        cfg.params.true,
        resolve_parameter_schema(model_name),
    )

    state_list = []
    state = forward_model(params=true_params)
    if state is None:
        state = forward_model.get_states()
    state_list.append(state)
    for _ in range(num_steps - 1):
        state = forward_model(params=true_params, state=state)
        if state is None:
            state = forward_model.get_states()
        state_list.append(state)

    state = xarray.concat(state_list, dim="time", join="override")
    state = add_velocity_magnitude(state)

    print(f"Model: {model_name}")
    print(f"Rollout: {num_steps} steps")
    print(f"Dims: {dict(state.sizes)}")
    print(f"Vars: {list(state.data_vars)}")

    if not cfg.run.skip_viz:
        out_dir = resolve_output_dir(cfg, "forward_model") / f"{model_name}_rollout"
        out_dir.mkdir(parents=True, exist_ok=True)

        plot_var = "vel_magnitude" if "vel_magnitude" in state.data_vars else "u"
        plot_2d = extract_2d_slice(state[plot_var], z_level=0)
        plt.figure(figsize=(6, 5))
        plt.imshow(plot_2d, origin="lower")
        plt.colorbar(label=plot_var)
        plt.title(f"{model_name} rollout - {plot_var} (last time, mid z)")
        plt.tight_layout()
        plt.savefig(out_dir / "field_snapshot.png")
        plt.close()

        if model_name == "pyudales":
            state = interpolate_grid(state)

        animate_state(
            state=state,
            output_path=out_dir / "state_animation.mp4",
            z_level=0,
        )
        print(f"Saved visualization outputs in {out_dir}")


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    run(cfg)


if __name__ == "__main__":
    main()
