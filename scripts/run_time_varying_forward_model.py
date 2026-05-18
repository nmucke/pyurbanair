"""Run a forward model with time-varying inflow parameters.

Supports both pyudales (via time-dependent nudging) and pylbm (via
``uvel_time.dat``).  Use ``model=`` to select which solver to run.
"""

import pathlib
import sys

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import hydra
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from hydra.utils import instantiate
from omegaconf import DictConfig

from pyurbanair.config.hydra_helpers import (
    clean_outputs,
    resolve_output_dir,
)
from pyurbanair.utils.animation_utils import animate_state
from pyurbanair.utils.run_utils import add_velocity_magnitude, extract_2d_slice


def run(cfg: DictConfig) -> None:
    model_name = cfg.model.name

    # Time is relative to the simulation interval (not including spinup).
    # The forward model internally prepends a constant plateau for spinup.
    sim_time = cfg.time.simulation_time
    n_snapshots = 100
    time_seconds = np.linspace(0, sim_time, n_snapshots)

    if bool(cfg.run.use_true_params):
        angle = float(cfg.params.true.inflow_angle)
        vel = float(cfg.params.true.velocity_magnitude)
        print(f"Using TRUE_PARAMS: angle={angle}, vel={vel}")
        inflow_vec = np.full(n_snapshots, angle)
    else:
        angle = -40.0
        vel = 3.0

        x = np.linspace(angle, -angle, n_snapshots)

        def sigmoid(x, center, width, min_val, max_val):
            return min_val + (max_val - min_val) / (1 + np.exp(-(x - center) / width))

        inflow_vec = sigmoid(x, center=0.0, width=5.0, min_val=angle, max_val=-angle)

    data_vars: dict = {
        "inflow_angle": ("time", inflow_vec),
        "velocity_magnitude": ("time", np.full(n_snapshots, vel)),
    }

    # pressure_gradient_magnitude is only relevant for pyudales
    if model_name == "pyudales":
        data_vars["pressure_gradient_magnitude"] = float(
            cfg.params.true.pressure_gradient_magnitude
        )

    params = xr.Dataset(data_vars=data_vars, coords={"time": time_seconds})

    results_dir = (
        pathlib.Path(cfg.run.results_dir) if cfg.run.results_dir is not None else None
    )
    forward_model = instantiate(cfg.model.forward_model, results_dir=results_dir)
    instantiate(cfg.model.prepare, forward_model=forward_model)
    clean_outputs(model_name=model_name, forward_model=forward_model)

    state = forward_model.run_single(params=params)
    if state is None:
        state = forward_model.get_states()

    state = add_velocity_magnitude(state)

    print(f"Model: {model_name} (time-varying inflow)")
    print(f"Dims: {dict(state.sizes)}")
    print(f"Vars: {list(state.data_vars)}")
    print(
        f"Inflow angle: {params['inflow_angle'].values[0]:.1f} -> "
        f"{params['inflow_angle'].values[-1]:.1f} deg"
    )
    print(
        f"Velocity magnitude: {params['velocity_magnitude'].values[0]:.1f} -> "
        f"{params['velocity_magnitude'].values[-1]:.1f} m/s"
    )

    if not cfg.run.skip_viz:
        out_dir = resolve_output_dir(cfg, "forward_model") / f"{model_name}_time_varying"
        out_dir.mkdir(parents=True, exist_ok=True)

        plot_var = "vel_magnitude" if "vel_magnitude" in state.data_vars else "u"
        plot_2d = extract_2d_slice(state[plot_var], z_level=0)
        plt.figure(figsize=(6, 5))
        plt.imshow(plot_2d, origin="lower")
        plt.colorbar(label=plot_var)
        plt.title(f"{model_name} time-varying - {plot_var} (last time, mid z)")
        plt.tight_layout()
        plt.savefig(out_dir / "field_snapshot.png")
        plt.close()

        if model_name == "pyudales":
            from pyudales.utils.grid_utils import interpolate_grid

            state = interpolate_grid(state)

        animate_state(
            state=state,
            output_path=out_dir / "state_animation.mp4",
            z_level=0,
        )

        # Derived inflow angle at three probes near the left x-boundary.
        # PALM stores u/v on staggered grids (u on xu, v on yv), so resolve
        # the spatial dim names per-variable.
        def _pick_dim(da, candidates):
            return next(d for d in candidates if d in da.dims)

        x_cands = ("x", "xt", "xm", "xu")
        y_cands = ("y", "yt", "ym", "yv")
        z_cands = ("z", "zt", "zm", "zu")
        u_x_dim = _pick_dim(state["u"], x_cands)
        u_y_dim = _pick_dim(state["u"], y_cands)
        u_z_dim = _pick_dim(state["u"], z_cands)
        v_x_dim = _pick_dim(state["v"], x_cands)
        v_y_dim = _pick_dim(state["v"], y_cands)
        v_z_dim = _pick_dim(state["v"], z_cands)

        x_left = float(state[u_x_dim].min())
        y_min = float(state[u_y_dim].min())
        y_max = float(state[u_y_dim].max())
        y_probes = [
            y_min + 0.2 * (y_max - y_min),
            y_min + 0.5 * (y_max - y_min),
            y_min + 0.8 * (y_max - y_min),
        ]
        z_probe = 0.5 * (float(state[u_z_dim].min()) + float(state[u_z_dim].max()))

        fig, ax = plt.subplots(figsize=(8, 4))
        for y_p in y_probes:
            u_sel = {u_x_dim: x_left, u_y_dim: y_p, u_z_dim: z_probe}
            v_sel = {v_x_dim: x_left, v_y_dim: y_p, v_z_dim: z_probe}
            u_t = state["u"].sel(**u_sel, method="nearest")
            v_t = state["v"].sel(**v_sel, method="nearest")
            angle_sim = np.degrees(np.arctan2(v_t.values, u_t.values))
            ax.plot(state.time.values, angle_sim, label=f"y={y_p:.1f} m")
        ax.plot(
            params["time"].values,
            params["inflow_angle"].values,
            "k--",
            alpha=0.5,
            label="prescribed",
        )
        ax.set_xlabel("time [s]")
        ax.set_ylabel("inflow angle [deg]")
        ax.set_title(f"Derived inflow angle near x={x_left:.1f} m (z={z_probe:.1f} m)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(out_dir / "derived_inflow_angle.png")
        plt.close()

        print(f"Saved visualization outputs in {out_dir}")


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    run(cfg)


if __name__ == "__main__":
    main()
