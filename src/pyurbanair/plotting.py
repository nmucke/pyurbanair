import pathlib

import matplotlib.pyplot as plt
import numpy as np
import xarray

from pyurbanair.utils.run_utils import add_velocity_magnitude


def _extract_2d_slice_with_extent(
    data_array: xarray.DataArray,
    z_level: int | None = None,
) -> tuple[np.ndarray, tuple[float, float, float, float]]:
    da = data_array
    if "time" in da.dims:
        da = da.isel(time=-1)
    for z_dim in ("z", "zm", "zt"):
        if z_dim in da.dims:
            da = da.isel(
                {z_dim: z_level if z_level is not None else len(da[z_dim]) // 2}
            )
            break
    if da.ndim > 2:
        indexers = {dim: 0 for dim in da.dims[:-2]}
        da = da.isel(indexers)

    values = np.asarray(da.values)
    if da.ndim != 2:
        return values, (0.0, float(values.shape[-1]), 0.0, float(values.shape[-2]))

    y_dim, x_dim = da.dims[0], da.dims[1]
    if x_dim in da.coords and y_dim in da.coords:
        x_vals = np.asarray(da.coords[x_dim].values)
        y_vals = np.asarray(da.coords[y_dim].values)
        extent = (
            float(np.min(x_vals)),
            float(np.max(x_vals)),
            float(np.min(y_vals)),
            float(np.max(y_vals)),
        )
    else:
        extent = (0.0, float(values.shape[1]), 0.0, float(values.shape[0]))
    return values, extent


def plot_parameter_distributions(
    params_history: xarray.Dataset,
    true_params: xarray.Dataset | None,
    output_path: str | pathlib.Path,
    bins: int = 25,
) -> None:
    """Plot per-step parameter distributions for an ESMDA parameter history."""
    step_dim = None
    for candidate in ("esmda_step", "assimilation_step", "step", "window", "iteration"):
        if candidate in params_history.dims:
            step_dim = candidate
            break

    if step_dim is None:
        params_history = params_history.expand_dims(esmda_step=[0])
        step_dim = "esmda_step"

    num_steps = int(params_history.sizes[step_dim])
    param_names = list(params_history.data_vars)
    if not param_names:
        raise ValueError("No parameters found in params_history.")

    # Compute fixed x-axis limits per parameter (across all steps)
    param_xlims: dict[str, tuple[float, float]] = {}
    for param_name in param_names:
        all_vals: list[float] = []
        for i in range(num_steps):
            step_slice = params_history.isel({step_dim: i})
            vals = np.asarray(step_slice[param_name].values).reshape(-1)
            vals = vals[np.isfinite(vals)]
            if vals.size > 0:
                all_vals.extend(vals.tolist())
        if true_params is not None and param_name in true_params.data_vars:
            true_val = np.asarray(true_params[param_name].values).reshape(-1)
            if true_val.size > 0 and np.isfinite(true_val[0]):
                all_vals.append(float(true_val[0]))
        if all_vals:
            vmin, vmax = float(np.min(all_vals)), float(np.max(all_vals))
            margin = max((vmax - vmin) * 0.05, 1e-10)
            param_xlims[param_name] = (vmin - margin, vmax + margin)
        else:
            param_xlims[param_name] = (0.0, 1.0)

    fig, axes = plt.subplots(
        num_steps,
        len(param_names),
        figsize=(5 * len(param_names), 3.5 * num_steps),
        squeeze=False,
        constrained_layout=True,
    )

    for i in range(num_steps):
        step_slice = params_history.isel({step_dim: i})
        for j, param_name in enumerate(param_names):
            ax = axes[i, j]
            values = np.asarray(step_slice[param_name].values).reshape(-1)
            values = values[np.isfinite(values)]
            if values.size == 0:
                ax.text(0.5, 0.5, "No finite values", ha="center", va="center")
                ax.set_axis_off()
                continue

            ax.hist(values, bins=bins, alpha=0.7, label=f"Step {i}")
            ax.axvline(
                float(np.mean(values)),
                color="black",
                linestyle="--",
                linewidth=2,
                label="ESMDA mean",
            )

            if true_params is not None and param_name in true_params.data_vars:
                true_value = np.asarray(true_params[param_name].values).reshape(-1)
                if true_value.size > 0 and np.isfinite(true_value[0]):
                    ax.axvline(
                        float(true_value[0]),
                        color="red",
                        linewidth=2,
                        label="True",
                    )

            ax.set_xlim(param_xlims[param_name])

            if i == 0:
                ax.set_title(f"{param_name} distribution")
            ax.set_ylabel(f"Step {i}")
            ax.legend()

    output_path = pathlib.Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)


def plot_true_vs_estimated_state(
    true_state: xarray.Dataset,
    estimated_state: xarray.Dataset,
    output_path: str | pathlib.Path,
    obs_x: np.ndarray | None = None,
    obs_y: np.ndarray | None = None,
    z_level: int | None = None,
) -> None:
    """Plot estimated vs true state snapshots, error, and RMSE by step."""
    true_for_plot = add_velocity_magnitude(true_state)
    est_for_plot = add_velocity_magnitude(estimated_state)

    plot_var = "vel_magnitude" if "vel_magnitude" in est_for_plot.data_vars else "u"
    if plot_var not in est_for_plot.data_vars:
        plot_var = list(est_for_plot.data_vars)[0]
    if plot_var not in true_for_plot.data_vars:
        true_plot_var = (
            "vel_magnitude" if "vel_magnitude" in true_for_plot.data_vars else None
        )
        if true_plot_var is None:
            true_plot_var = list(true_for_plot.data_vars)[0]
    else:
        true_plot_var = plot_var

    step_dim = None
    for candidate in ("esmda_step", "assimilation_step", "step", "window", "iteration"):
        if candidate in est_for_plot.dims:
            step_dim = candidate
            break

    if step_dim is None:
        est_for_plot = est_for_plot.expand_dims(esmda_step=[0])
        step_dim = "esmda_step"

    true_2d, true_extent = _extract_2d_slice_with_extent(
        true_for_plot[true_plot_var], z_level=z_level
    )
    num_steps = int(est_for_plot.sizes[step_dim])
    rmse_vals: list[float] = []
    est_slices: list[np.ndarray] = []
    true_slices: list[np.ndarray] = []
    err_slices: list[np.ndarray] = []
    est_extents: list[tuple[float, float, float, float]] = []
    true_extents: list[tuple[float, float, float, float]] = []
    err_extents: list[tuple[float, float, float, float]] = []

    for i in range(num_steps):
        est_2d, est_extent = _extract_2d_slice_with_extent(
            est_for_plot.isel({step_dim: i})[plot_var],
            z_level=z_level,
        )
        true_2d_aligned = true_2d
        true_extent_aligned = true_extent
        min_y = min(est_2d.shape[0], true_2d.shape[0])
        min_x = min(est_2d.shape[1], true_2d.shape[1])
        if est_2d.shape != true_2d.shape:
            est_2d = est_2d[:min_y, :min_x]
            true_2d_aligned = true_2d[:min_y, :min_x]
            x0 = max(est_extent[0], true_extent[0])
            x1 = min(est_extent[1], true_extent[1])
            y0 = max(est_extent[2], true_extent[2])
            y1 = min(est_extent[3], true_extent[3])
            if x1 > x0 and y1 > y0:
                est_extent = (x0, x1, y0, y1)
                true_extent_aligned = (x0, x1, y0, y1)
        err_2d = est_2d - true_2d_aligned
        rmse_vals.append(float(np.sqrt(np.mean(err_2d**2))))
        est_slices.append(est_2d)
        true_slices.append(true_2d_aligned)
        err_slices.append(err_2d)
        est_extents.append(est_extent)
        true_extents.append(true_extent_aligned)
        err_extents.append(est_extent)

    true_vmin = float(np.nanmin(true_2d))
    true_vmax = float(np.nanmax(true_2d))
    err_abs = max(float(np.nanmax(np.abs(e))) for e in err_slices)
    err_vmin, err_vmax = -err_abs, err_abs

    fig, axes = plt.subplots(
        num_steps,
        3,
        figsize=(12, 4 * num_steps),
        squeeze=False,
        constrained_layout=True,
    )
    for i in range(num_steps):
        im0 = axes[i, 0].imshow(
            est_slices[i],
            origin="lower",
            vmin=true_vmin,
            vmax=true_vmax,
            extent=est_extents[i],
        )
        im1 = axes[i, 1].imshow(
            true_slices[i],
            origin="lower",
            vmin=true_vmin,
            vmax=true_vmax,
            extent=true_extents[i],
        )
        im2 = axes[i, 2].imshow(
            err_slices[i],
            origin="lower",
            cmap="RdBu_r",
            vmin=err_vmin,
            vmax=err_vmax,
            extent=err_extents[i],
        )

        if obs_x is not None and obs_y is not None:
            axes[i, 0].scatter(obs_x, obs_y, color="red", s=12)
            axes[i, 1].scatter(obs_x, obs_y, color="red", s=12)

        if i == 0:
            axes[i, 0].set_title(f"Estimated ({plot_var})")
            axes[i, 1].set_title(f"True ({true_plot_var})")
            axes[i, 2].set_title("Error")
        axes[i, 2].set_ylabel(f"Step {i}\nRMSE={rmse_vals[i]:.4f}")

        fig.colorbar(im0, ax=axes[i, 0])
        fig.colorbar(im1, ax=axes[i, 1])
        fig.colorbar(im2, ax=axes[i, 2])

    output_path = pathlib.Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)
