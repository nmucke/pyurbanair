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


def _resolve_dim(da: xarray.DataArray, candidates: tuple[str, ...]) -> str | None:
    return next((d for d in candidates if d in da.dims), None)


def animate_height_panels(
    state: xarray.Dataset,
    output_path: str | pathlib.Path,
    heights: tuple[float, ...] = (2.0, 10.0, 32.0),
    fps: int = 10,
    dpi: int = 120,
    speed_cmap: str = "viridis",
    vort_cmap: str = "RdBu_r",
) -> None:
    """Animate velocity magnitude and vorticity at several heights, one file.

    Layout is a 2 x ``len(heights)`` grid of panels sharing a single time axis:
    the top row is the horizontal velocity magnitude and the bottom row the
    vertical vorticity ``omega_z = dv/dx - du/dy``; each column is one height.
    The requested heights are matched to the nearest available z-level.

    ``state`` must already be reduced to a single member (no ``ensemble`` dim)
    and carry ``u``/``v`` with a vertical and two horizontal coordinates.
    """
    if not all(v in state.data_vars for v in ("u", "v")):
        raise ValueError("animate_height_panels needs 'u' and 'v' in the state")
    if "time" not in state.dims:
        raise ValueError("Dataset must have a 'time' dimension")

    z_dim = _resolve_dim(state["u"], ("z", "zm", "zt", "zu"))
    if z_dim is None:
        raise ValueError("Could not find a vertical dimension on 'u'")

    # Speed source: reuse the precomputed magnitude when present, else fall back
    # to the horizontal speed sqrt(u^2 + v^2).
    speed_da = state["vel_magnitude"] if "vel_magnitude" in state.data_vars else None

    times = np.asarray(state["time"].values)
    n_times = len(times)
    z_coord = np.asarray(state[z_dim].values)

    # Per-height precomputed (time, y, x) numpy arrays for both rows.
    speed_frames: list[np.ndarray] = []
    vort_frames: list[np.ndarray] = []
    actual_heights: list[float] = []
    extent = None
    for h in heights:
        k = int(np.argmin(np.abs(z_coord - h)))
        u_h = state["u"].isel({z_dim: k})
        v_h = state["v"].isel({z_dim: k})
        actual_heights.append(float(z_coord[k]))

        x_dim, y_dim = u_h.dims[-1], u_h.dims[-2]
        x_coord = np.asarray(u_h[x_dim].values)
        y_coord = np.asarray(u_h[y_dim].values)
        if extent is None:
            extent = [x_coord.min(), x_coord.max(), y_coord.min(), y_coord.max()]

        u_arr = np.asarray(u_h.values)
        v_arr = np.asarray(v_h.values)
        # omega_z = dv/dx - du/dy, differentiated over physical coordinates.
        dv_dx = np.gradient(v_arr, x_coord, axis=-1)
        du_dy = np.gradient(u_arr, y_coord, axis=-2)
        vort_frames.append(dv_dx - du_dy)

        if speed_da is not None:
            speed_frames.append(np.asarray(speed_da.isel({z_dim: k}).values))
        else:
            speed_frames.append(np.hypot(u_arr, v_arr))

    # Shared colour limits per row: speed 0..max, vorticity symmetric about 0.
    speed_max = float(np.nanmax([np.nanmax(f) for f in speed_frames]))
    vort_lim = float(
        np.nanmax([np.nanpercentile(np.abs(f), 99) for f in vort_frames])
    )

    n_cols = len(heights)
    fig, axes = plt.subplots(
        2, n_cols, figsize=(5 * n_cols, 9), squeeze=False, constrained_layout=True
    )
    fig.set_facecolor("white")

    speed_ims, vort_ims = [], []
    for col in range(n_cols):
        ax_top, ax_bot = axes[0, col], axes[1, col]
        im_s = ax_top.imshow(
            speed_frames[col][0], origin="lower", aspect="auto",
            cmap=speed_cmap, vmin=0.0, vmax=speed_max, extent=extent,
        )
        im_v = ax_bot.imshow(
            vort_frames[col][0], origin="lower", aspect="auto",
            cmap=vort_cmap, vmin=-vort_lim, vmax=vort_lim, extent=extent,
        )
        ax_top.set_title(f"z = {actual_heights[col]:.0f} m", fontweight="bold")
        for ax in (ax_top, ax_bot):
            ax.set_xticks([])
            ax.set_yticks([])
        speed_ims.append(im_s)
        vort_ims.append(im_v)

    axes[0, 0].set_ylabel("Velocity magnitude [m/s]", fontweight="bold")
    axes[1, 0].set_ylabel(r"Vorticity $\omega_z$ [1/s]", fontweight="bold")
    fig.colorbar(speed_ims[-1], ax=axes[0, :].tolist(), fraction=0.046, pad=0.02)
    fig.colorbar(vort_ims[-1], ax=axes[1, :].tolist(), fraction=0.046, pad=0.02)

    suptitle = fig.suptitle(f"t = {times[0]:.2f} s", fontsize=15, fontweight="bold")

    output_path = pathlib.Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path, writer = _get_writer_and_output_path(output_path=output_path, fps=fps)

    with writer.saving(fig, str(output_path), dpi=dpi):
        for t in range(n_times):
            for col in range(n_cols):
                speed_ims[col].set_array(speed_frames[col][t])
                vort_ims[col].set_array(vort_frames[col][t])
            suptitle.set_text(f"t = {times[t]:.2f} s")
            writer.grab_frame()

    plt.close(fig)


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
    "animate_height_panels",
    "animate_rollout_state",
    "_visualize_state_history",
]
