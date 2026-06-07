import pathlib
import warnings
from typing import Optional

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import xarray


def _get_writer_and_output_path(
    output_path: pathlib.Path,
    fps: int,
) -> tuple[pathlib.Path, animation.AbstractMovieWriter]:
    """
    Return a usable animation writer and output path.

    Falls back to PillowWriter/GIF when ffmpeg is not available.
    """
    if animation.writers.is_available("ffmpeg"):
        return output_path, animation.FFMpegWriter(fps=fps)

    gif_path = output_path.with_suffix(".gif")
    warnings.warn(
        "ffmpeg is not available. Saving animation as GIF instead.",
        RuntimeWarning,
        stacklevel=3,
    )
    return gif_path, animation.PillowWriter(fps=fps)


def _save_projection_animation(
    volume_4d: np.ndarray,
    output_path: pathlib.Path,
    fps: int,
    dpi: int,
    width: int,
    height: int,
    cmap: str,
) -> None:
    """
    Save a CPU-only fallback animation using max-intensity projection over depth.
    """
    # Infer a likely depth axis from the smallest spatial dimension.
    # This keeps fallback robust even if spatial dimensions are reordered upstream.
    depth_axis = int(np.argmin(volume_4d.shape[1:])) + 1
    projection = np.nanmax(volume_4d, axis=depth_axis)

    finite = np.isfinite(projection)
    if not np.any(finite):
        warnings.warn(
            "Projection contains no finite values; writing an empty fallback animation.",
            RuntimeWarning,
            stacklevel=2,
        )
        projection = np.zeros_like(projection, dtype=np.float32)
        vmin, vmax = 0.0, 1.0
    else:
        valid_values = projection[finite]
        # Use robust percentiles to avoid a near-flat image due to outliers.
        vmin = float(np.percentile(valid_values, 2))
        vmax = float(np.percentile(valid_values, 98))
        if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
            vmin = float(np.nanmin(valid_values))
            vmax = float(np.nanmax(valid_values))
            if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
                vmin, vmax = 0.0, 1.0

    projection = np.nan_to_num(projection, nan=vmin)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path, writer = _get_writer_and_output_path(output_path=output_path, fps=fps)

    fig, ax = plt.subplots(figsize=(width / dpi, height / dpi), dpi=dpi)
    ax.set_axis_off()
    im = ax.imshow(
        projection[0],
        cmap=cmap,
        origin="lower",
        aspect="auto",
        vmin=vmin,
        vmax=vmax,
    )
    with writer.saving(fig, str(output_path), dpi=dpi):
        writer.grab_frame()
        for frame in projection[1:]:
            im.set_array(frame)
            writer.grab_frame()
    plt.close(fig)


def animate_state(
    state: xarray.Dataset,
    output_path: str | pathlib.Path,
    z_level: Optional[int] = None,
    fps: int = 10,
    dpi: int = 100,
    cmap: str = "viridis",
    vmin: Optional[float | dict[str, float]] = None,
    vmax: Optional[float | dict[str, float]] = None,
) -> None:
    """
    Animate all data variables from an xarray Dataset, with each frame being a time instance.

    Parameters
    ----------
    state : xarray.Dataset
        Dataset containing data variables to animate. Expected to have a 'time' dimension.
    output_path : str | pathlib.Path
        Path where the .mp4 file will be saved.
    z_level : Optional[int], default=None
        Z-level index to use for 3D variables. If None, uses the middle z-level.
    fps : int, default=10
        Frames per second for the animation.
    dpi : int, default=100
        Resolution (dots per inch) for the animation.
    cmap : str, default="viridis"
        Colormap to use for the plots.
    vmin : Optional[float | dict[str, float]], default=None
        Minimum value for color scale. If a float, applies to all variables.
        If a dict, maps variable names to their vmin values. If None, auto-calculates.
    vmax : Optional[float | dict[str, float]], default=None
        Maximum value for color scale. If a float, applies to all variables.
        If a dict, maps variable names to their vmax values. If None, auto-calculates.
    """
    # Get all data variables
    data_vars = list(state.data_vars.keys())
    n_vars = len(data_vars)

    if n_vars == 0:
        raise ValueError("Dataset has no data variables to animate")

    # Get time dimension
    if "time" not in state.dims:
        raise ValueError("Dataset must have a 'time' dimension")

    time_values = state.time.values
    n_times = len(time_values)

    # Determine z_level if not provided
    if z_level is None:
        # Find a common z dimension (zm or zt)
        z_dims = ["zm", "zt"]
        z_dim = None
        for zd in z_dims:
            if zd in state.dims:
                z_dim = zd
                break

        if z_dim is not None:
            z_level = len(state[z_dim]) // 2
        else:
            z_level = 0

    # Create figure with subplots
    fig, axes = plt.subplots(
        1, n_vars, figsize=(5 * n_vars, 5), constrained_layout=True
    )
    if n_vars == 1:
        axes = [axes]

    # Initialize imshow objects for each variable
    ims = []
    vmin_max = {}

    # Determine vmin/vmax for each variable
    # Check if vmin/vmax are provided as dictionaries (per-variable) or floats (global)
    vmin_dict = vmin if isinstance(vmin, dict) else None
    vmax_dict = vmax if isinstance(vmax, dict) else None
    global_vmin = vmin if isinstance(vmin, (int, float)) else None
    global_vmax = vmax if isinstance(vmax, (int, float)) else None

    # First pass: determine vmin/vmax for each variable across all times
    for var_name in data_vars:
        # Calculate auto values first
        var_data = state[var_name]
        var_values = []

        for t_idx in range(n_times):
            var_slice = var_data.isel(time=t_idx)

            # Handle z dimension
            z_dims_in_var = [d for d in ["zm", "zt"] if d in var_slice.dims]
            if z_dims_in_var:
                z_dim_name = z_dims_in_var[0]
                if len(var_slice[z_dim_name]) > z_level:
                    var_slice = var_slice.isel({z_dim_name: z_level})
                else:
                    var_slice = var_slice.isel({z_dim_name: 0})

            # Get 2D slice (handle different x/y coordinate systems)
            if len(var_slice.dims) == 2:
                var_2d = var_slice.values
            elif len(var_slice.dims) == 1:
                # 1D variable, skip or handle differently
                continue
            else:
                # Take first slice if more than 2D remains
                while len(var_slice.dims) > 2:
                    var_slice = var_slice.isel({var_slice.dims[0]: 0})
                var_2d = var_slice.values

            var_values.append(var_2d)

        if var_values:
            all_values = np.concatenate([v.flatten() for v in var_values])
            auto_vmin = np.nanmin(all_values)
            auto_vmax = np.nanmax(all_values)

            # Determine final vmin: per-variable dict > global value > auto-calculated
            if vmin_dict is not None and var_name in vmin_dict:
                final_vmin = vmin_dict[var_name]
            elif global_vmin is not None:
                final_vmin = global_vmin
            else:
                final_vmin = auto_vmin

            # Determine final vmax: per-variable dict > global value > auto-calculated
            if vmax_dict is not None and var_name in vmax_dict:
                final_vmax = vmax_dict[var_name]
            elif global_vmax is not None:
                final_vmax = global_vmax
            else:
                final_vmax = auto_vmax

            vmin_max[var_name] = (final_vmin, final_vmax)

    # Initialize plots
    for idx, var_name in enumerate(data_vars):
        var_data = state[var_name]
        var_slice = var_data.isel(time=0)

        # Handle z dimension
        z_dims_in_var = [d for d in ["zm", "zt"] if d in var_slice.dims]
        if z_dims_in_var:
            z_dim_name = z_dims_in_var[0]
            if len(var_slice[z_dim_name]) > z_level:
                var_slice = var_slice.isel({z_dim_name: z_level})
            else:
                var_slice = var_slice.isel({z_dim_name: 0})

        # Get 2D slice
        if len(var_slice.dims) == 2:
            var_2d = var_slice.values
            x_coord = var_slice.coords[var_slice.dims[1]].values
            y_coord = var_slice.coords[var_slice.dims[0]].values
        elif len(var_slice.dims) == 1:
            # Skip 1D variables
            continue
        else:
            # Take first slice if more than 2D remains
            while len(var_slice.dims) > 2:
                var_slice = var_slice.isel({var_slice.dims[0]: 0})
            var_2d = var_slice.values
            x_coord = var_slice.coords[var_slice.dims[1]].values
            y_coord = var_slice.coords[var_slice.dims[0]].values

        vmin, vmax = vmin_max.get(var_name, (None, None))
        im = axes[idx].imshow(
            var_2d,
            origin="lower",
            aspect="auto",
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            extent=[
                x_coord.min(),
                x_coord.max(),
                y_coord.min(),
                y_coord.max(),
            ],
        )
        axes[idx].set_title(f"{var_name} (t={time_values[0]:.2f})")
        axes[idx].set_xlabel(var_slice.dims[1])
        axes[idx].set_ylabel(var_slice.dims[0])
        plt.colorbar(im, ax=axes[idx])
        ims.append(im)

    def animate(frame: int) -> list:
        """Update function for animation."""
        for idx, var_name in enumerate(data_vars):
            var_data = state[var_name]
            var_slice = var_data.isel(time=frame)

            # Handle z dimension
            z_dims_in_var = [d for d in ["zm", "zt"] if d in var_slice.dims]
            if z_dims_in_var:
                z_dim_name = z_dims_in_var[0]
                if len(var_slice[z_dim_name]) > z_level:
                    var_slice = var_slice.isel({z_dim_name: z_level})
                else:
                    var_slice = var_slice.isel({z_dim_name: 0})

            # Get 2D slice
            if len(var_slice.dims) == 2:
                var_2d = var_slice.values
            elif len(var_slice.dims) == 1:
                continue
            else:
                while len(var_slice.dims) > 2:
                    var_slice = var_slice.isel({var_slice.dims[0]: 0})
                var_2d = var_slice.values

            ims[idx].set_array(var_2d)
            axes[idx].set_title(f"{var_name} (t={time_values[frame]:.2f})")

        return ims

    # Create animation
    anim = animation.FuncAnimation(
        fig, animate, frames=n_times, interval=1000 / fps, blit=False
    )

    # Save animation
    output_path = pathlib.Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path, writer = _get_writer_and_output_path(output_path=output_path, fps=fps)

    anim.save(str(output_path), writer=writer, dpi=dpi)

    plt.close(fig)
