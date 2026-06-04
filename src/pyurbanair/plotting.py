import pathlib

import matplotlib.pyplot as plt
import numpy as np
import xarray
from matplotlib.lines import Line2D

from pyurbanair.utils.run_utils import add_velocity_magnitude

# --- Shared figure style ----------------------------------------------------
# Semantic colours used consistently across every figure.
_COLOR_TRUTH = "#222222"        # near-black, drawn dashed
_COLOR_PRIOR = "#ff7f0e"        # orange
_COLOR_POSTERIOR = "#1f77b4"    # blue
_COLOR_OBS = "#e6194b"          # crimson markers

# Colourmaps by physical meaning.
_CMAP_FIELD = "viridis"         # velocity magnitude
_CMAP_STD = "magma"             # ensemble spread
_CMAP_ERROR = "Reds"            # absolute error

# rcParams applied (locally, via rc_context) inside each plotting function so we
# never mutate global matplotlib state.
_RC = {
    "figure.dpi": 120,
    "savefig.dpi": 150,
    "savefig.bbox": "tight",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.titleweight": "bold",
    "axes.labelsize": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "axes.axisbelow": True,
    "grid.alpha": 0.3,
    "grid.linewidth": 0.6,
    "legend.frameon": True,
    "legend.framealpha": 0.9,
    "legend.fontsize": 9,
    "image.cmap": _CMAP_FIELD,
}

_PARAM_LABELS = {
    "inflow_angle": "Inflow angle",
    "velocity_magnitude": "Velocity magnitude",
}


def _save(fig, output_path: str | pathlib.Path) -> None:
    output_path = pathlib.Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)


def _shade_windows(ax, edges) -> None:
    """Lightly shade alternating assimilation windows with dotted dividers."""
    if edges is None or len(edges) < 2:
        return
    for k in range(len(edges) - 1):
        if k % 2 == 1:
            ax.axvspan(edges[k], edges[k + 1], color="0.5", alpha=0.06, zorder=0)
    for e in edges[1:-1]:
        ax.axvline(e, color="0.75", linewidth=0.6, linestyle=":", zorder=0)


def _param_legend_handles(has_prior: bool) -> list[Line2D]:
    handles: list[Line2D] = []
    if has_prior:
        handles += [
            Line2D([0], [0], color=_COLOR_PRIOR, lw=2.5, label="Prior mean"),
            Line2D([0], [0], color=_COLOR_PRIOR, lw=0.9, alpha=0.5, label="Prior members"),
        ]
    handles += [
        Line2D([0], [0], color=_COLOR_POSTERIOR, lw=2.5, label="Posterior mean"),
        Line2D([0], [0], color=_COLOR_POSTERIOR, lw=0.9, alpha=0.5, label="Posterior members"),
        Line2D([0], [0], color=_COLOR_TRUTH, lw=2.0, ls="--", label="Truth"),
    ]
    return handles


def _extract_2d_slice_with_extent(
    data_array: xarray.DataArray,
    z_level: int | None = None,
    time_index: int | None = None,
) -> tuple[np.ndarray, tuple[float, float, float, float]]:
    da = data_array
    if "time" in da.dims:
        idx = time_index if time_index is not None else -1
        da = da.isel(time=idx)
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


def plot_state_init_and_terminal(
    true_state: xarray.Dataset,
    estimated_state: xarray.Dataset,
    output_path: str | pathlib.Path,
    obs_x: np.ndarray | None = None,
    obs_y: np.ndarray | None = None,
    z_level: int | None = None,
) -> None:
    """Plot estimated vs true state with both initial and terminal time for each step."""
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

    has_time_est = "time" in est_for_plot.dims
    has_time_true = "time" in true_for_plot.dims

    true_init, _ = _extract_2d_slice_with_extent(
        true_for_plot[true_plot_var],
        z_level=z_level,
        time_index=0 if has_time_true else None,
    )
    true_terminal, _ = _extract_2d_slice_with_extent(
        true_for_plot[true_plot_var],
        z_level=z_level,
        time_index=-1 if has_time_true else None,
    )

    num_steps = int(est_for_plot.sizes[step_dim])
    init_slices: list[np.ndarray] = []
    terminal_slices: list[np.ndarray] = []
    init_extents: list[tuple[float, float, float, float]] = []
    terminal_extents: list[tuple[float, float, float, float]] = []
    init_rmse: list[float] = []
    terminal_rmse: list[float] = []

    for i in range(num_steps):
        step_slice = est_for_plot.isel({step_dim: i})
        est_init, est_init_ext = _extract_2d_slice_with_extent(
            step_slice[plot_var],
            z_level=z_level,
            time_index=0 if has_time_est else None,
        )
        est_terminal, est_terminal_ext = _extract_2d_slice_with_extent(
            step_slice[plot_var],
            z_level=z_level,
            time_index=-1 if has_time_est else None,
        )

        min_y = min(est_init.shape[0], true_init.shape[0])
        min_x = min(est_init.shape[1], true_init.shape[1])
        true_init_aligned = true_init[:min_y, :min_x]
        est_init_crop = est_init[:min_y, :min_x]
        init_err = est_init_crop - true_init_aligned
        init_rmse.append(float(np.sqrt(np.mean(init_err**2))))

        min_y = min(est_terminal.shape[0], true_terminal.shape[0])
        min_x = min(est_terminal.shape[1], true_terminal.shape[1])
        true_terminal_aligned = true_terminal[:min_y, :min_x]
        est_terminal_crop = est_terminal[:min_y, :min_x]
        terminal_err = est_terminal_crop - true_terminal_aligned
        terminal_rmse.append(float(np.sqrt(np.mean(terminal_err**2))))

        init_slices.append(est_init_crop)
        terminal_slices.append(est_terminal_crop)
        init_extents.append(est_init_ext)
        terminal_extents.append(est_terminal_ext)

    true_vmin = float(
        np.nanmin(np.concatenate([true_init.ravel(), true_terminal.ravel()]))
    )
    true_vmax = float(
        np.nanmax(np.concatenate([true_init.ravel(), true_terminal.ravel()]))
    )
    err_abs = 0.0
    for i in range(num_steps):
        sy, sx = init_slices[i].shape
        err_abs = max(
            err_abs, float(np.nanmax(np.abs(init_slices[i] - true_init[:sy, :sx])))
        )
        sy, sx = terminal_slices[i].shape
        err_abs = max(
            err_abs,
            float(np.nanmax(np.abs(terminal_slices[i] - true_terminal[:sy, :sx]))),
        )
    err_vmin, err_vmax = -err_abs, err_abs

    fig, axes = plt.subplots(
        num_steps,
        6,
        figsize=(18, 4 * num_steps),
        squeeze=False,
        constrained_layout=True,
    )
    for i in range(num_steps):
        sy, sx = init_slices[i].shape
        im0 = axes[i, 0].imshow(
            init_slices[i],
            origin="lower",
            vmin=true_vmin,
            vmax=true_vmax,
            extent=init_extents[i],
        )
        im1 = axes[i, 1].imshow(
            true_init[:sy, :sx],
            origin="lower",
            vmin=true_vmin,
            vmax=true_vmax,
            extent=init_extents[i],
        )
        im2 = axes[i, 2].imshow(
            init_slices[i] - true_init[:sy, :sx],
            origin="lower",
            cmap="RdBu_r",
            vmin=err_vmin,
            vmax=err_vmax,
            extent=init_extents[i],
        )
        sy, sx = terminal_slices[i].shape
        im3 = axes[i, 3].imshow(
            terminal_slices[i],
            origin="lower",
            vmin=true_vmin,
            vmax=true_vmax,
            extent=terminal_extents[i],
        )
        im4 = axes[i, 4].imshow(
            true_terminal[:sy, :sx],
            origin="lower",
            vmin=true_vmin,
            vmax=true_vmax,
            extent=terminal_extents[i],
        )
        im5 = axes[i, 5].imshow(
            terminal_slices[i] - true_terminal[:sy, :sx],
            origin="lower",
            cmap="RdBu_r",
            vmin=err_vmin,
            vmax=err_vmax,
            extent=terminal_extents[i],
        )

        if obs_x is not None and obs_y is not None:
            for col in (0, 1, 3, 4):
                axes[i, col].scatter(obs_x, obs_y, color="red", s=12)

        if i == 0:
            axes[i, 0].set_title(f"Estimated init ({plot_var})")
            axes[i, 1].set_title(f"True init ({true_plot_var})")
            axes[i, 2].set_title("Init error")
            axes[i, 3].set_title(f"Estimated terminal ({plot_var})")
            axes[i, 4].set_title(f"True terminal ({true_plot_var})")
            axes[i, 5].set_title("Terminal error")
        axes[i, 2].set_ylabel(f"Step {i}\nInit RMSE={init_rmse[i]:.4f}")
        axes[i, 5].set_ylabel(f"Terminal RMSE={terminal_rmse[i]:.4f}")

    output_path = pathlib.Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)


def plot_rollout_time_evolution(
    esmda_params: xarray.Dataset,
    true_params: xarray.Dataset,
    esmda_state: xarray.Dataset,
    true_state: xarray.Dataset,
    output_path: str | pathlib.Path,
    prior_params: xarray.Dataset | None = None,
    window_edges: list[float] | None = None,
) -> None:
    """Plot parameter and RMSE time evolution over rollout assimilation windows.

    For each parameter every ensemble member is drawn faintly (``alpha=0.35``)
    for both the prior (if ``prior_params`` is given) and the posterior, with the
    ensemble mean overlaid on top (opaque, thicker). The truth is a dashed line.
    ``window_edges`` (if given) lightly shades alternating assimilation windows.
    """
    from pyurbanair.utils.state_utils import get_velocity_magnitude_field

    def _members_and_x(da: xarray.DataArray):
        """Return (x, members) with members shaped (n_ensemble, n_x)."""
        da = da.transpose("ensemble", ...)
        members = np.asarray(da.values).reshape(da.sizes["ensemble"], -1)
        non_ens = [d for d in da.dims if d != "ensemble"]
        if non_ens == ["time"] and "time" in da.coords:
            x = np.asarray(da["time"].values, dtype=float)
        else:
            x = np.arange(members.shape[1])
        return x, members

    def _plot_ensemble(ax, ds, param_name, color):
        x, members = _members_and_x(ds[param_name])
        ax.plot(x, members.T, color=color, alpha=0.35, linewidth=0.9)
        ax.plot(x, members.mean(axis=0), color=color, alpha=1.0, linewidth=2.5)

    true_state_mean = true_state.mean(dim="ensemble") if "ensemble" in true_state.dims else true_state
    esmda_state_mean = esmda_state.mean(dim="ensemble") if "ensemble" in esmda_state.dims else esmda_state

    true_vel = np.asarray(get_velocity_magnitude_field(true_state_mean))
    esmda_vel = np.asarray(get_velocity_magnitude_field(esmda_state_mean))
    min_t = min(true_vel.shape[0], esmda_vel.shape[0])
    rmse = np.sqrt(np.mean((true_vel[:min_t] - esmda_vel[:min_t]) ** 2, axis=tuple(range(1, true_vel.ndim))))

    param_names = [p for p in ("inflow_angle", "velocity_magnitude") if p in esmda_params.data_vars]
    n_params = len(param_names)
    has_prior = prior_params is not None

    with plt.rc_context(_RC):
        fig, axes = plt.subplots(
            n_params + 1, 1, figsize=(11, 3.2 * (n_params + 1)), constrained_layout=True
        )
        axes = np.atleast_1d(axes)

        for i, param_name in enumerate(param_names):
            ax = axes[i]
            _shade_windows(ax, window_edges)
            if has_prior and param_name in prior_params.data_vars:
                _plot_ensemble(ax, prior_params, param_name, _COLOR_PRIOR)
            _plot_ensemble(ax, esmda_params, param_name, _COLOR_POSTERIOR)
            if param_name in true_params.data_vars:
                true_da = true_params[param_name]
                if "ensemble" in true_da.dims:
                    true_da = true_da.isel(ensemble=0)
                x_true, true_members = _members_and_x(true_da.expand_dims("ensemble"))
                ax.plot(
                    x_true, true_members[0], color=_COLOR_TRUTH, linewidth=2.0,
                    linestyle="--", zorder=5,
                )
            ax.set_ylabel(_PARAM_LABELS.get(param_name, param_name))
            ax.set_xlabel("Time")
            ax.margins(x=0.01)
            ax.legend(handles=_param_legend_handles(has_prior), loc="best", ncol=1)

        ax_rmse = axes[n_params]
        ax_rmse.plot(
            np.arange(len(rmse)), rmse, color=_COLOR_POSTERIOR, linewidth=2.0,
            marker="o", markersize=4,
        )
        ax_rmse.set_xlabel("Time step")
        ax_rmse.set_ylabel("RMSE  |U|")
        ax_rmse.set_title("State error")
        ax_rmse.margins(x=0.01)

        fig.suptitle("Parameter evolution over assimilation windows", fontsize=15, fontweight="bold")
        _save(fig, output_path)


def plot_final_state_with_obs(
    mean_vel: xarray.DataArray,
    std_vel: xarray.DataArray,
    output_path: str | pathlib.Path,
    true_vel: xarray.DataArray | None = None,
    obs_x: np.ndarray | None = None,
    obs_y: np.ndarray | None = None,
    z_level: int | None = None,
) -> None:
    """Plot the velocity magnitude at the final time with observation locations.

    Always shows the posterior ensemble mean and std; if ``true_vel`` is given,
    the truth is added as a leading panel that shares the colour scale of the
    posterior mean for a fair comparison.
    """
    mean_2d, mean_extent = _extract_2d_slice_with_extent(mean_vel, z_level=z_level)
    std_2d, std_extent = _extract_2d_slice_with_extent(std_vel, z_level=z_level)

    true_2d = true_extent = None
    if true_vel is not None:
        true_2d, true_extent = _extract_2d_slice_with_extent(true_vel, z_level=z_level)

    # Share the colour scale between truth and posterior mean.
    field_stack = [mean_2d] + ([true_2d] if true_2d is not None else [])
    vmin = float(np.nanmin([np.nanmin(f) for f in field_stack]))
    vmax = float(np.nanmax([np.nanmax(f) for f in field_stack]))

    with plt.rc_context(_RC):
        n_panels = 3 if true_2d is not None else 2
        fig, axes = plt.subplots(
            1, n_panels, figsize=(6.3 * n_panels, 5.4), constrained_layout=True
        )

        col = 0
        if true_2d is not None:
            im_true = axes[col].imshow(
                true_2d, origin="lower", cmap=_CMAP_FIELD, extent=true_extent,
                aspect="equal", vmin=vmin, vmax=vmax,
            )
            axes[col].set_title("Truth  |U|")
            cb = fig.colorbar(im_true, ax=axes[col], fraction=0.046, pad=0.04)
            cb.set_label("Velocity magnitude")
            col += 1

        im_mean = axes[col].imshow(
            mean_2d, origin="lower", cmap=_CMAP_FIELD, extent=mean_extent,
            aspect="equal", vmin=vmin, vmax=vmax,
        )
        axes[col].set_title("Posterior mean  |U|")
        cb0 = fig.colorbar(im_mean, ax=axes[col], fraction=0.046, pad=0.04)
        cb0.set_label("Velocity magnitude")
        col += 1

        im_std = axes[col].imshow(
            std_2d, origin="lower", cmap=_CMAP_STD, extent=std_extent, aspect="equal"
        )
        axes[col].set_title("Posterior std  |U|")
        cb1 = fig.colorbar(im_std, ax=axes[col], fraction=0.046, pad=0.04)
        cb1.set_label("Ensemble std")

        for ax in axes:
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.grid(False)

        if obs_x is not None and obs_y is not None:
            for ax in axes:
                ax.scatter(
                    obs_x, obs_y, s=40, marker="o", facecolor=_COLOR_OBS,
                    edgecolor="white", linewidth=0.8, zorder=5, label="Observations",
                )
            axes[0].legend(loc="upper right", framealpha=0.9)

        fig.suptitle("State at final time", fontsize=15, fontweight="bold")
        _save(fig, output_path)
