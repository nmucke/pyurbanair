"""Animation helpers used by scripts_new runners."""

import pathlib

import matplotlib.pyplot as plt
import xarray

from pyurbanair.animation import animate_3d, animate_ensemble_state, animate_state
from pyurbanair.utils.run_utils import add_velocity_magnitude, extract_2d_slice


def _visualize_state_history(
    state_history: xarray.Dataset,
    out_dir: pathlib.Path,
    title_prefix: str,
) -> None:
    state_viz = state_history
    for step_dim in ("esmda_step", "assimilation_step", "step", "window", "iteration"):
        if step_dim in state_viz.dims:
            state_viz = state_viz.isel({step_dim: -1})
            break

    state_viz = add_velocity_magnitude(state_viz)
    if not state_viz.data_vars:
        return
    plot_var = "vel_magnitude" if "vel_magnitude" in state_viz.data_vars else "u"
    if plot_var not in state_viz.data_vars:
        plot_var = list(state_viz.data_vars)[0]

    snapshot_state = (
        state_viz.mean(dim="ensemble") if "ensemble" in state_viz.dims else state_viz
    )
    if "time" in snapshot_state.dims:
        plot_2d = extract_2d_slice(snapshot_state[plot_var])
        if plot_2d.ndim == 2:
            plt.figure(figsize=(6, 5))
            plt.imshow(plot_2d, origin="lower")
            plt.colorbar(label=plot_var)
            plt.title(f"{title_prefix} - {plot_var} (last step)")
            plt.tight_layout()
            plt.savefig(out_dir / "state_history_snapshot.png")
            plt.close()

    if "time" not in state_viz.dims:
        return
    if "ensemble" in state_viz.dims:
        animate_ensemble_state(
            state=state_viz,
            output_path=out_dir / "state_history_animation.mp4",
            z_level=None,
        )
    else:
        animate_state(
            state=state_viz,
            output_path=out_dir / "state_history_animation.mp4",
            z_level=None,
        )


__all__ = [
    "animate_state",
    "animate_3d",
    "animate_ensemble_state",
    "_visualize_state_history",
]
