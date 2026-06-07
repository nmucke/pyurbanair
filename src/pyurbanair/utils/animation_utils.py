"""Animation helpers used by the scripts/ runners."""

import pathlib

import matplotlib.pyplot as plt
import numpy as np
import xarray

from pyurbanair.animation import _get_writer_and_output_path, animate_state
from pyurbanair.utils.run_utils import add_velocity_magnitude, extract_2d_slice


def _regrid_horizontal(src: xarray.DataArray, tgt: xarray.DataArray) -> xarray.DataArray:
    """Interpolate ``src``'s horizontal plane onto ``tgt``'s grid.

    The last two dims of each array are treated as ``(y, x)`` and interpolation
    is by physical coordinate value, so a truth and an assimilation state living
    on different resolutions can be differenced. Returns ``src`` unchanged when
    the grids already match or coordinates are unavailable.
    """
    sy, sx = src.dims[-2], src.dims[-1]
    ty, tx = tgt.dims[-2], tgt.dims[-1]
    if src.sizes[sy] == tgt.sizes[ty] and src.sizes[sx] == tgt.sizes[tx]:
        return src
    if not all(c in src.coords for c in (sy, sx)) or not all(c in tgt.coords for c in (ty, tx)):
        return src
    return src.interp(
        {sy: np.asarray(tgt[ty].values), sx: np.asarray(tgt[tx].values)},
        kwargs={"bounds_error": False, "fill_value": None},
    )


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
    mean_vel: xarray.DataArray,
    std_vel: xarray.DataArray,
    output_path: str | pathlib.Path,
    z_level: int | None = None,
    fps: int = 5,
    dpi: int = 120,
    cmap: str = "viridis",
) -> None:
    """Animate 4-panel rollout comparison over time windows.

    Panels per frame:
      1. Truth velocity magnitude
      2. Ensemble mean velocity magnitude
      3. Ensemble std velocity magnitude
      4. |Ensemble mean − truth| velocity magnitude

    ``mean_vel`` and ``std_vel`` are the precomputed ensemble mean and std of the
    velocity magnitude (no ``ensemble`` dimension), so the full ensemble need not
    be held in memory here.
    """
    true_with_vel = add_velocity_magnitude(true_state)

    if "vel_magnitude" not in true_with_vel.data_vars:
        raise ValueError(
            "Could not compute vel_magnitude for true_state (need u, v, w)"
        )

    true_vel = true_with_vel["vel_magnitude"]
    if "ensemble" in true_vel.dims:
        true_vel = true_vel.mean(dim="ensemble")

    # Truth and assimilation states may sit on different resolutions; interpolate
    # the truth onto the (mean) assimilation grid so the error panel
    # (|mean - truth|) can be differenced point-for-point.
    true_vel = _regrid_horizontal(true_vel, mean_vel)

    if any("time" not in da.dims for da in (true_vel, mean_vel, std_vel)):
        raise ValueError("true_state, mean_vel and std_vel must have a 'time' dimension")

    n_times = min(true_vel.sizes["time"], mean_vel.sizes["time"], std_vel.sizes["time"])

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
        mean_2d = _get_2d(mean_vel, t)
        std_2d = _get_2d(std_vel, t)
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

    # Real time labels when available, else a frame counter.
    times = np.asarray(mean_vel["time"].values) if "time" in mean_vel.coords else None

    def _frame_label(t: int) -> str:
        return f"t = {times[t]:.2f}" if times is not None else f"Frame {t + 1} / {n_times}"

    fig, axes = plt.subplots(1, 4, figsize=(21, 5.4), constrained_layout=True)
    fig.set_facecolor("white")
    panels = [
        ("Truth  |U|", frames_truth, cmap, vmin_vel, vmax_vel, "Velocity magnitude"),
        ("Ensemble mean  |U|", frames_mean, cmap, vmin_vel, vmax_vel, "Velocity magnitude"),
        ("Ensemble std  |U|", frames_std, "magma", 0.0, vmax_std, "Ensemble std"),
        ("Absolute error  |U|", frames_diff, "Reds", 0.0, vmax_diff, "|mean − truth|"),
    ]

    images = []
    for ax, (title, frames, cm, vmn, vmx, cb_label) in zip(axes, panels):
        im = ax.imshow(frames[0], origin="lower", cmap=cm, vmin=vmn, vmax=vmx)
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.set_axis_off()
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cb.set_label(cb_label, fontsize=10)
        images.append(im)
    im_truth, im_mean, im_std, im_diff = images

    suptitle = fig.suptitle(_frame_label(0), fontsize=15, fontweight="bold")

    output_path = pathlib.Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path, writer = _get_writer_and_output_path(output_path=output_path, fps=fps)

    with writer.saving(fig, str(output_path), dpi=dpi):
        for t in range(n_times):
            im_truth.set_array(frames_truth[t])
            im_mean.set_array(frames_mean[t])
            im_std.set_array(frames_std[t])
            im_diff.set_array(frames_diff[t])
            suptitle.set_text(_frame_label(t))
            writer.grab_frame()

    plt.close(fig)


__all__ = [
    "animate_state",
    "animate_rollout_state",
    "_visualize_state_history",
]
