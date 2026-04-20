import argparse
import pathlib
import sys

import xarray

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt
from pyudales.utils.grid_utils import interpolate_grid
from pyurbanair.utils.animation_utils import animate_state
from pyurbanair.utils.run_utils import add_velocity_magnitude, extract_2d_slice

from scripts import config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        choices=["pylbm", "pyudales"],
        default="pylbm",
        help="Forward model backend.",
    )
    parser.add_argument(
        "--skip-viz",
        action="store_true",
        help="Skip plotting and animation outputs.",
    )
    parser.add_argument(
        "--results-dir",
        type=pathlib.Path,
        default=None,
        help="Override results directory for model outputs.",
    )
    args = parser.parse_args()

    model_name = args.model
    results_dir = (
        pathlib.Path(args.results_dir) if args.results_dir is not None else None
    )

    forward_model = config.create_forward_model(
        model_name=model_name,
        results_dir=results_dir,
    )
    config.prepare_forward_model(model_name=model_name, forward_model=forward_model)
    config.clean_forward_model_outputs(
        model_name=model_name, forward_model=forward_model
    )
    forward_model = config.create_rollout_forward_model(
        model_name=model_name,
        forward_model=forward_model,
    )

    state_list = []
    true_params = config.create_true_params(model_name)
    state = forward_model(params=true_params)
    if state is None:
        state = forward_model.get_states()
    state_list.append(state)
    state = forward_model(params=true_params, state=state)
    if state is None:
        state = forward_model.get_states()
    state_list.append(state)

    state = xarray.concat(state_list, dim="time", join="override")
    state = add_velocity_magnitude(state)

    print(f"Model: {model_name}")
    print(f"Rollout: 2 steps")
    print(f"Dims: {dict(state.sizes)}")
    print(f"Vars: {list(state.data_vars)}")

    if not args.skip_viz:
        out_dir = config.BASE_RESULTS_DIR / "forward_model" / f"{model_name}_rollout"
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


if __name__ == "__main__":
    main()
