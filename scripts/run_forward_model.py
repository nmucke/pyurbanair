import argparse
import pathlib
import sys

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
    forward_model = config.create_forward_model(
        model_name=model_name,
        results_dir=(
            pathlib.Path(args.results_dir) if args.results_dir is not None else None
        ),
    )
    config.prepare_forward_model(model_name=model_name, forward_model=forward_model)
    config.clean_forward_model_outputs(
        model_name=model_name, forward_model=forward_model
    )

    true_params = config.create_true_params(model_name)

    state = forward_model(params=true_params)
    if state is None:
        state = forward_model.get_states()
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
        plot_2d = extract_2d_slice(state[plot_var], z_level=0)
        plt.figure(figsize=(6, 5))
        plt.imshow(plot_2d, origin="lower")
        plt.colorbar(label=plot_var)
        plt.title(f"{model_name} {run_type} - {plot_var} (last time, mid z)")
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

        import numpy as np

        plt.figure()
        for idx in [10, 40, 70]:
            u_at_left_end = state.u.isel(z=1, y=idx, x=0).values
            v_at_left_end = state.v.isel(z=1, y=idx, x=0).values
            inflow_angle_from_state = (
                np.arctan2(v_at_left_end, u_at_left_end) * 180.0 / np.pi
            )
            plt.plot(inflow_angle_from_state, label=f"idx={idx}")
        plt.legend()
        plt.title("inflow angle from state")
        plt.xlabel("time")
        plt.ylabel("inflow angle")
        plt.show()

        print(f"\n--- Inflow angle diagnostics ---")
        print(f"state.u.dims = {state.u.dims}")
        print(f"state.u.shape = {state.u.shape}")

        t_mid = state.sizes["time"] // 2
        sample = state.isel(time=t_mid, x=0)
        for zi in range(min(state.sizes["z"], 4)):
            s = sample.isel(z=zi)
            u_mean = float(s.u.mean())
            v_mean = float(s.v.mean())
            w_mean = float(s.w.mean())
            angle = np.degrees(np.arctan2(v_mean, u_mean))
            mag = np.sqrt(u_mean**2 + v_mean**2)
            print(
                f"  z={zi}: u={u_mean:.4f}, v={v_mean:.4f}, w={w_mean:.4f}, "
                f"|uv|={mag:.4f}, angle(v,u)={angle:.2f}°"
            )

        print(f"\nAngle vs x-position (z=0, y=40, mid-time):")
        for xi in [0, 1, 2, 5, 10, 60, 119]:
            s = state.isel(time=t_mid, z=0, y=40, x=xi)
            u_val = float(s.u)
            v_val = float(s.v)
            w_val = float(s.w)
            angle = np.degrees(np.arctan2(v_val, u_val))
            print(
                f"  x={xi}: u={u_val:.4f}, v={v_val:.4f}, w={w_val:.4f}, "
                f"angle={angle:.2f}°"
            )

        print(f"--- End diagnostics ---\n")


if __name__ == "__main__":
    main()
