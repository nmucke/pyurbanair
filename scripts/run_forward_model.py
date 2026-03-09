import argparse
import pathlib
import sys

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt

from pyurbanair.utils.animation_utils import animate_state
from pyurbanair.utils.run_utils import add_velocity_magnitude, extract_2d_slice
from scripts import config
from pyudales.utils.grid_utils import interpolate_grid


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        choices=["pylbm", "pyudales"],
        default="pylbm",
        help="Forward model backend.",
    )
    parser.add_argument(
        "--rollout",
        action="store_true",
        help="Use rollout forward model variant.",
    )
    parser.add_argument(
        "--skip-viz",
        action="store_true",
        help="Skip plotting and animation outputs.",
    )
    args = parser.parse_args()

    model_name = args.model
    forward_model = config.create_forward_model(
        model_name=model_name,
        rollout=args.rollout,
        results_dir=None,
    )
    config.prepare_forward_model(model_name=model_name, forward_model=forward_model)
    config.clean_forward_model_outputs(
        model_name=model_name, forward_model=forward_model
    )

    true_params = config.create_true_params(model_name)
    state = forward_model(params=true_params)
    if state is None:
        raise RuntimeError("Expected in-memory state from forward model run.")
    state = add_velocity_magnitude(state)

    print(f"Model: {model_name}")
    print(f"Rollout: {args.rollout}")
    print(f"Dims: {dict(state.sizes)}")
    print(f"Vars: {list(state.data_vars)}")

    if not args.skip_viz:
        run_type = "rollout" if args.rollout else "single"
        out_dir = config.BASE_RESULTS_DIR / "forward_model" / f"{model_name}_{run_type}"
        out_dir.mkdir(parents=True, exist_ok=True)

        plot_var = "vel_magnitude" if "vel_magnitude" in state.data_vars else "u"
        n_times = state.sizes.get("time", 1)
        time_indices = (
            [0, (n_times - 1) // 2, n_times - 1]
            if n_times >= 3
            else list(range(n_times))
        )
        fig, axes = plt.subplots(1, len(time_indices), figsize=(5 * len(time_indices), 5))
        if len(time_indices) == 1:
            axes = [axes]
        for ax, ti in zip(axes, time_indices):
            plot_2d = extract_2d_slice(
                state[plot_var].isel(time=ti) if n_times > 1 else state[plot_var],
                z_level=0,
            )
            im = ax.imshow(plot_2d, origin="lower")
            plt.colorbar(im, ax=ax, label=plot_var)
            t_label = f"t={ti}" if n_times > 1 else "t=0"
            ax.set_title(f"{plot_var} ({t_label}, mid z)")
        fig.suptitle(f"{model_name} {run_type} - {plot_var} at equidistant times")
        plt.tight_layout()
        plt.savefig(out_dir / "field_snapshot.png")
        plt.close()


        if model_name == "pyudales":
            state = interpolate_grid(state)

        animate_state(
            state=state,
            output_path=out_dir / "state_animation.mp4",
            z_level=1,
        )
        print(f"Saved visualization outputs in {out_dir}")


if __name__ == "__main__":
    main()
