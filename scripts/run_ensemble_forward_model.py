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
        rollout=args.rollout,
        results_dir=results_dir,
    )
    config.prepare_forward_model(model_name=model_name, forward_model=forward_model)
    config.clean_forward_model_outputs(
        model_name=model_name, forward_model=forward_model
    )

    ensemble_model = config.create_ensemble_forward_model(
        model_name=model_name, forward_model=forward_model
    )
    for member in ensemble_model.ensemble_forward_models:
        config.clean_forward_model_outputs(
            model_name=model_name, forward_model=member
        )

    params_ensemble = config.create_parameter_ensemble(model_name)
    state = ensemble_model.run_ensemble(params=params_ensemble, sim_name="state")
    if state is None:
        state = ensemble_model.get_states()
    state = add_velocity_magnitude(state)

    print(f"Model: {model_name}")
    print(f"Rollout: {args.rollout}")
    print(f"Ensemble size: {config.ENSEMBLE['ensemble_size']}")
    print(f"Dims: {dict(state.sizes)}")
    print(f"Vars: {list(state.data_vars)}")

    if not args.skip_viz:
        run_type = "rollout" if args.rollout else "ensemble"
        out_dir = config.BASE_RESULTS_DIR / "forward_model" / f"{model_name}_{run_type}"
        out_dir.mkdir(parents=True, exist_ok=True)

        # Plot ensemble mean
        state_mean = state.mean(dim="ensemble")
        plot_var = "vel_magnitude" if "vel_magnitude" in state_mean.data_vars else "u"
        plot_2d = extract_2d_slice(state_mean[plot_var], z_level=0)
        plt.figure(figsize=(6, 5))
        plt.imshow(plot_2d, origin="lower")
        plt.colorbar(label=plot_var)
        plt.title(f"{model_name} {run_type} - {plot_var} (ensemble mean, last time, mid z)")
        plt.tight_layout()
        plt.savefig(out_dir / "field_snapshot.png")
        plt.close()

        if model_name == "pyudales":
            state = interpolate_grid(state)

        # Animate ensemble mean
        state_for_anim = state.mean(dim="ensemble")
        animate_state(
            state=state_for_anim,
            output_path=out_dir / "state_animation.mp4",
            z_level=0,
        )
        print(f"Saved visualization outputs in {out_dir}")


if __name__ == "__main__":
    main()
