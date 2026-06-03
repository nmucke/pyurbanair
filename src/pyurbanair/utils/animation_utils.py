"""Animation helpers used by the scripts/ runners."""

import pathlib

import matplotlib.pyplot as plt
import numpy as np
import xarray

from pyurbanair.animation import _get_writer_and_output_path, animate_state
from pyurbanair.utils.run_utils import add_velocity_magnitude, extract_2d_slice


def _visualize_state_history(
    state_history: xarray.Dataset,
    out_dir: pathlib.Path,
    title_prefix: str,
    z_level: int | None = None,
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
        plot_2d = extract_2d_slice(snapshot_state[plot_var], z_level=z_level)
        if plot_2d.ndim == 2:
            plt.figure(figsize=(6, 5))
            plt.imshow(plot_2d, origin="lower")
            plt.colorbar(label=plot_var)
            plt.title(f"{title_prefix} - {plot_var} (last step)")
            plt.tight_layout()
            plt.savefig(out_dir / "state_history_snapshot.png")
            plt.close()


def animate_rollout_state(
    true_state: xarray.Dataset,
    esmda_state: xarray.Dataset,
    output_path: str | pathlib.Path,
    z_level: int | None = None,
    fps: int = 5,
    dpi: int = 100,
    cmap: str = "viridis",
) -> None:
    """Animate 4-panel rollout comparison over time windows.

    Panels per frame:
      1. Truth velocity magnitude
      2. Ensemble mean velocity magnitude
      3. Ensemble std velocity magnitude
      4. |Ensemble mean − truth| velocity magnitude
    """
    true_with_vel = add_velocity_magnitude(true_state)
    esmda_with_vel = add_velocity_magnitude(esmda_state)

    if (
        "vel_magnitude" not in true_with_vel.data_vars
        or "vel_magnitude" not in esmda_with_vel.data_vars
    ):
        raise ValueError(
            "Could not compute vel_magnitude (need u, v, w in both datasets)"
        )

    true_vel = true_with_vel["vel_magnitude"]
    esmda_vel = esmda_with_vel["vel_magnitude"]

    if "time" not in true_vel.dims or "time" not in esmda_vel.dims:
        raise ValueError("Both true_state and esmda_state must have a 'time' dimension")
    if "ensemble" not in esmda_vel.dims:
        raise ValueError("esmda_state must have an 'ensemble' dimension")

    n_times = min(true_vel.sizes["time"], esmda_vel.sizes["time"])

    # Resolve z_level
    z_dim = next((d for d in ("z", "zm", "zt") if d in true_vel.dims), None)
    if z_level is None:
        z_level = (true_vel.sizes[z_dim] // 2) if z_dim is not None else 0

    def _get_2d(da: xarray.DataArray, t: int) -> np.ndarray:
        sl = da.isel(time=t)
        if z_dim is not None and z_dim in sl.dims:
            sl = sl.isel({z_dim: z_level})
        while sl.ndim > 2:
            sl = sl.isel({sl.dims[0]: 0})
        return np.asarray(sl.values)

    # Pre-compute all frames to get consistent colour limits
    frames_truth, frames_mean, frames_std, frames_diff = [], [], [], []
    for t in range(n_times):
        truth_2d = _get_2d(true_vel, t)
        mean_2d = _get_2d(esmda_vel.mean(dim="ensemble"), t)
        std_2d = _get_2d(esmda_vel.std(dim="ensemble"), t)
        diff_2d = np.abs(mean_2d - truth_2d)
        frames_truth.append(truth_2d)
        frames_mean.append(mean_2d)
        frames_std.append(std_2d)
        frames_diff.append(diff_2d)

    all_vel = np.concatenate([f.ravel() for f in frames_truth + frames_mean])
    vmin_vel = float(np.nanmin(all_vel))
    vmax_vel = float(np.nanmax(all_vel))
    vmax_std = float(np.nanmax([f for f in frames_std]))
    vmax_diff = float(np.nanmax([f for f in frames_diff]))

    fig, axes = plt.subplots(1, 4, figsize=(22, 5), constrained_layout=True)
    titles = ["Truth |U|", "Ensemble mean |U|", "Ensemble std |U|", "Abs error |U|"]

    im_truth = axes[0].imshow(
        frames_truth[0], origin="lower", cmap=cmap, vmin=vmin_vel, vmax=vmax_vel
    )
    im_mean = axes[1].imshow(
        frames_mean[0], origin="lower", cmap=cmap, vmin=vmin_vel, vmax=vmax_vel
    )
    im_std = axes[2].imshow(
        frames_std[0], origin="lower", cmap="plasma", vmin=0.0, vmax=vmax_std
    )
    im_diff = axes[3].imshow(
        frames_diff[0], origin="lower", cmap="Reds", vmin=0.0, vmax=vmax_diff
    )

    for ax, title, im in zip(axes, titles, [im_truth, im_mean, im_std, im_diff]):
        ax.set_title(title)
        ax.set_axis_off()
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    suptitle = fig.suptitle("Time step 0", fontsize=12)

    output_path = pathlib.Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path, writer = _get_writer_and_output_path(output_path=output_path, fps=fps)

    with writer.saving(fig, str(output_path), dpi=dpi):
        for t in range(n_times):
            im_truth.set_array(frames_truth[t])
            im_mean.set_array(frames_mean[t])
            im_std.set_array(frames_std[t])
            im_diff.set_array(frames_diff[t])
            suptitle.set_text(f"Time step {t}")
            writer.grab_frame()

    plt.close(fig)


__all__ = [
    "animate_state",
    "animate_rollout_state",
    "_visualize_state_history",
]
