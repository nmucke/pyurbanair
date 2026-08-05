"""Figure builders: one function per figure ID, plus the general plots.

Here today (moved in WP0.2): the general state / parameter / sensor plots that
came out of ``pyurbanair.plotting``. WP1.5 adds the evaluation figure set
proper -- P1, S1, S5, F1, D1, then D3 and S4 in phases 2--3 — listed in
``docs/plans/esmda_turbulence_evaluation.md`` §7.

That new figure set will take time averages or statistics only -- never
instantaneous fields, which decorrelate after a Lyapunov horizon and measure
chaos rather than parameter quality. The general plots moved from
``pyurbanair.plotting`` predate that rule: several of them do plot snapshots,
and they take an ``output_path`` and write the file themselves rather than
returning a ``Figure``. WP0.2 is a pure refactor and keeps both properties;
changing either is a later cleanup.

Populated in WP0.2 (move), extended in WP1.5.
"""

# mypy: ignore-errors
# Moved wholesale in WP0.2 from ``src/pyurbanair/plotting.py`` -- largely
# unannotated code predating the strict mypy config. Waived rather than
# annotated as part of a pure refactor; dropping the waiver is later cleanup.
# The WP1.5 figure set below is fully annotated and passes the strict config on
# its own (checked by deleting this line: the 18 remaining errors are all in the
# moved code above it) -- but it is *inside* the waiver, so nothing enforces
# that. Dropping the waiver needs the 18 legacy errors fixed first.

import contextlib
import logging
import pathlib
import warnings
from typing import Iterator, cast

import matplotlib.pyplot as plt
import numpy as np
import xarray
from evaluation.scores import (
    _aligned_parameter_members,
    _param_members_and_x,
    _plotted_param_names,
    compute_parameter_metrics,
    compute_sensor_metrics,
    parameter_bundle,
)
from evaluation.style import (
    CMAP_DIFF,
    CMAP_FIELD,
    COLORS,
    PARAM_LABELS,
    PARAM_UNITS,
    apply_style,
    finite_limits,
    mark_windows,
    nested_bands,
    save_png,
)
from evaluation.turbulence import evenly_spaced_levels
from matplotlib.axes import Axes
from matplotlib.collections import PolyCollection
from matplotlib.colors import Colormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

logger = logging.getLogger(__name__)

# --- Shared figure style ----------------------------------------------------
# Semantic colours used consistently across every figure.
_COLOR_TRUTH = "#222222"  # near-black, drawn dashed
_COLOR_PRIOR = "#ff7f0e"  # orange
_COLOR_POSTERIOR = "#1f77b4"  # blue
_COLOR_OBS = "#e6194b"  # crimson markers

# Colourmaps by physical meaning.
_CMAP_FIELD = "viridis"  # velocity magnitude
_CMAP_STD = "magma"  # ensemble spread
_CMAP_ERROR = "Reds"  # absolute error

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
    "vertical_inflow_exponent": "Vertical inflow exponent (α)",
    "sgs_constant": "SGS constant",
}


# Velocity-magnitude helpers duplicated from ``pyurbanair.utils.run_utils`` and
# ``pyurbanair.utils.state_utils``: the originals also serve non-evaluation
# flows and stay put, and this leaf library must not import ``pyurbanair``.
# The arithmetic is identical to the originals.
def _add_velocity_magnitude(state: xarray.Dataset) -> xarray.Dataset:
    if not all(v in state.data_vars for v in ("u", "v", "w")):
        return state
    vel_magnitude = np.sqrt(state.u.values**2 + state.v.values**2 + state.w.values**2)
    return state.assign(vel_magnitude=(state["u"].dims, vel_magnitude))


def _get_velocity_magnitude_field(state: xarray.Dataset) -> np.ndarray:
    """Get the velocity magnitude field from a state."""
    u = state.u.values
    v = state.v.values
    w = state.w.values
    return np.sqrt(u**2 + v**2 + w**2)


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
            Line2D(
                [0], [0], color=_COLOR_PRIOR, lw=0.9, alpha=0.5, label="Prior members"
            ),
        ]
    handles += [
        Line2D([0], [0], color=_COLOR_POSTERIOR, lw=2.5, label="Posterior mean"),
        Line2D(
            [0],
            [0],
            color=_COLOR_POSTERIOR,
            lw=0.9,
            alpha=0.5,
            label="Posterior members",
        ),
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
    true_for_plot = _add_velocity_magnitude(true_state)
    est_for_plot = _add_velocity_magnitude(estimated_state)

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
    true_for_plot = _add_velocity_magnitude(true_state)
    est_for_plot = _add_velocity_magnitude(estimated_state)

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
    esmda_state: xarray.Dataset | None,
    true_state: xarray.Dataset | None,
    output_path: str | pathlib.Path,
    prior_params: xarray.Dataset | None = None,
    window_edges: list[float] | None = None,
    rmse: np.ndarray | None = None,
) -> None:
    """Plot parameter and RMSE time evolution over rollout assimilation windows.

    For each parameter every ensemble member is drawn faintly (``alpha=0.35``)
    for both the prior (if ``prior_params`` is given) and the posterior, with the
    ensemble mean overlaid on top (opaque, thicker). The truth is a dashed line.
    ``window_edges`` (if given) lightly shades alternating assimilation windows.

    ``rmse`` may be supplied precomputed (one value per time step). Callers
    handling a large truth should pass a streamed ``rmse`` here so the full 4-D
    velocity field is never materialised; ``esmda_state``/``true_state`` are then
    unused and may be ``None``. If ``rmse`` is ``None`` it is computed in full
    from the two states (the original whole-domain behaviour).
    """

    def _plot_ensemble(ax, ds, param_name, color):
        x, members = _param_members_and_x(ds[param_name])
        ax.plot(x, members.T, color=color, alpha=0.35, linewidth=0.9)
        ax.plot(x, members.mean(axis=0), color=color, alpha=1.0, linewidth=2.5)

    if rmse is None:
        # Fallback: whole-domain RMSE between the ensemble-mean state and the
        # truth. This materialises the full velocity fields; callers handling a
        # large truth should precompute a streamed ``rmse`` and pass it in.
        true_state_mean = (
            true_state.mean(dim="ensemble")
            if "ensemble" in true_state.dims
            else true_state
        )
        esmda_state_mean = (
            esmda_state.mean(dim="ensemble")
            if "ensemble" in esmda_state.dims
            else esmda_state
        )

        true_vel = np.asarray(_get_velocity_magnitude_field(true_state_mean))
        esmda_vel = np.asarray(_get_velocity_magnitude_field(esmda_state_mean))
        min_t = min(true_vel.shape[0], esmda_vel.shape[0])
        rmse = np.sqrt(
            np.mean(
                (true_vel[:min_t] - esmda_vel[:min_t]) ** 2,
                axis=tuple(range(1, true_vel.ndim)),
            )
        )
    else:
        rmse = np.asarray(rmse)

    param_names = _plotted_param_names(esmda_params)
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
                x_true, true_members = _param_members_and_x(
                    true_da.expand_dims("ensemble")
                )
                truth = true_members[0]
                # A constant-in-time (static) parameter has a single truth value;
                # draw it as a horizontal line spanning the panel rather than a
                # lone point so it reads against the posterior trajectory.
                if truth.size == 1 or np.allclose(truth, truth[0]):
                    ax.axhline(
                        float(truth[0]),
                        color=_COLOR_TRUTH,
                        linewidth=2.0,
                        linestyle="--",
                        zorder=5,
                    )
                else:
                    ax.plot(
                        x_true,
                        truth,
                        color=_COLOR_TRUTH,
                        linewidth=2.0,
                        linestyle="--",
                        zorder=5,
                    )
            ax.set_ylabel(_PARAM_LABELS.get(param_name, param_name))
            ax.set_xlabel("Time")
            ax.margins(x=0.01)
            ax.legend(handles=_param_legend_handles(has_prior), loc="best", ncol=1)

        ax_rmse = axes[n_params]
        ax_rmse.plot(
            np.arange(len(rmse)),
            rmse,
            color=_COLOR_POSTERIOR,
            linewidth=2.0,
            marker="o",
            markersize=4,
        )
        ax_rmse.set_xlabel("Time step")
        ax_rmse.set_ylabel("RMSE  |U|")
        ax_rmse.set_title("State error")
        ax_rmse.margins(x=0.01)

        fig.suptitle(
            "Parameter evolution over assimilation windows",
            fontsize=15,
            fontweight="bold",
        )
        _save(fig, output_path)


def plot_parameter_error(
    esmda_params: xarray.Dataset,
    true_params: xarray.Dataset,
    output_path: str | pathlib.Path,
    window_edges: list[float] | None = None,
) -> None:
    """Plot per-parameter estimation error of the posterior ensemble vs truth.

    One panel per parameter, each showing the RMSE and CRPS error series from
    :func:`evaluation.scores.compute_parameter_metrics` on a shared axis.
    ``window_edges`` (if given) shades the windows.
    """
    metrics = compute_parameter_metrics(esmda_params, true_params)
    if not metrics:
        return

    param_names = list(metrics.keys())
    x_is_time = "time" in esmda_params.coords

    with plt.rc_context(_RC):
        fig, axes = plt.subplots(
            len(param_names),
            1,
            figsize=(11, 3.2 * len(param_names)),
            constrained_layout=True,
        )
        axes = np.atleast_1d(axes)

        for ax, param_name in zip(axes, param_names):
            x_est = metrics[param_name]["x"]
            rmse = metrics[param_name]["rmse"]
            crps = metrics[param_name]["crps"]

            _shade_windows(ax, window_edges)
            ax.plot(
                x_est,
                rmse,
                color=_COLOR_POSTERIOR,
                linewidth=2.0,
                marker="o",
                markersize=4,
                label=f"RMSE (mean {np.mean(rmse):.3g})",
            )
            ax.plot(
                x_est,
                crps,
                color=_COLOR_PRIOR,
                linewidth=2.0,
                marker="s",
                markersize=4,
                label=f"CRPS (mean {np.mean(crps):.3g})",
            )
            ax.set_ylabel(f"{_PARAM_LABELS.get(param_name, param_name)} error")
            ax.set_xlabel("Time" if x_is_time else "Assimilation window")
            ax.margins(x=0.01)
            ax.set_ylim(bottom=0.0)
            ax.legend(loc="best")

        fig.suptitle("Parameter estimation error", fontsize=15, fontweight="bold")
        _save(fig, output_path)


def plot_sensor_timeseries(
    true_sensor: xarray.DataArray,
    ensemble_sensor: xarray.DataArray,
    output_path: str | pathlib.Path,
    title: str,
    sensor_x: np.ndarray | None = None,
    sensor_y: np.ndarray | None = None,
    sensor_z: np.ndarray | None = None,
) -> None:
    """Plot the true vs ensemble |U| time series at a set of sensor locations.

    One panel per sensor shows the truth (dashed black), every ensemble member
    (faint blue) and the ensemble mean (opaque blue) as a function of time. A
    final panel shows the per-time RMSE and CRPS over those sensors (see
    :func:`evaluation.scores.compute_sensor_metrics`).
    """
    m = compute_sensor_metrics(true_sensor, ensemble_sensor)
    t_ens = m["time"]
    members = m["members"]
    ens_mean = m["ens_mean"]
    truth = m["truth"]
    rmse = m["rmse"]
    crps = m["crps"]
    n_sensors = members.shape[2]

    def _sensor_label(i: int) -> str:
        if sensor_x is not None and sensor_y is not None and sensor_z is not None:
            return (
                f"Sensor {i}  "
                f"(x={float(sensor_x[i]):.0f}, y={float(sensor_y[i]):.0f}, "
                f"z={float(sensor_z[i]):.0f})"
            )
        return f"Sensor {i}"

    handles = [
        Line2D(
            [0],
            [0],
            color=_COLOR_POSTERIOR,
            lw=0.9,
            alpha=0.5,
            label="Ensemble members",
        ),
        Line2D([0], [0], color=_COLOR_POSTERIOR, lw=2.5, label="Ensemble mean"),
        Line2D([0], [0], color=_COLOR_TRUTH, lw=2.0, ls="--", label="Truth"),
    ]

    with plt.rc_context(_RC):
        fig, axes = plt.subplots(
            n_sensors + 1,
            1,
            figsize=(11, 2.6 * (n_sensors + 1)),
            constrained_layout=True,
        )
        axes = np.atleast_1d(axes)

        for i in range(n_sensors):
            ax = axes[i]
            ax.plot(
                t_ens,
                members[:, :, i].T,
                color=_COLOR_POSTERIOR,
                alpha=0.35,
                linewidth=0.9,
            )
            ax.plot(
                t_ens, ens_mean[:, i], color=_COLOR_POSTERIOR, alpha=1.0, linewidth=2.5
            )
            ax.plot(
                t_ens,
                truth[:, i],
                color=_COLOR_TRUTH,
                linewidth=2.0,
                linestyle="--",
                zorder=5,
            )
            ax.set_ylabel("|U|")
            ax.set_title(_sensor_label(i), loc="left")
            ax.margins(x=0.01)
            if i == 0:
                ax.legend(handles=handles, loc="best", ncol=1)

        ax_err = axes[n_sensors]
        ax_err.plot(
            t_ens,
            rmse,
            color=_COLOR_POSTERIOR,
            linewidth=2.0,
            marker="o",
            markersize=4,
            label=f"RMSE (mean {np.mean(rmse):.3g})",
        )
        ax_err.plot(
            t_ens,
            crps,
            color=_COLOR_PRIOR,
            linewidth=2.0,
            marker="s",
            markersize=4,
            label=f"CRPS (mean {np.mean(crps):.3g})",
        )
        ax_err.set_ylabel("|U| error")
        ax_err.set_xlabel("Time")
        ax_err.set_title("Sensor error", loc="left")
        ax_err.set_ylim(bottom=0.0)
        ax_err.margins(x=0.01)
        ax_err.legend(loc="best")

        fig.suptitle(title, fontsize=15, fontweight="bold")
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
                true_2d,
                origin="lower",
                cmap=_CMAP_FIELD,
                extent=true_extent,
                aspect="equal",
                vmin=vmin,
                vmax=vmax,
            )
            axes[col].set_title("Truth  |U|")
            cb = fig.colorbar(im_true, ax=axes[col], fraction=0.046, pad=0.04)
            cb.set_label("Velocity magnitude")
            col += 1

        im_mean = axes[col].imshow(
            mean_2d,
            origin="lower",
            cmap=_CMAP_FIELD,
            extent=mean_extent,
            aspect="equal",
            vmin=vmin,
            vmax=vmax,
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
                    obs_x,
                    obs_y,
                    s=40,
                    marker="o",
                    facecolor=_COLOR_OBS,
                    edgecolor="white",
                    linewidth=0.8,
                    zorder=5,
                    label="Observations",
                )
            axes[0].legend(loc="upper right", framealpha=0.9)

        fig.suptitle("State at final time", fontsize=15, fontweight="bold")
        _save(fig, output_path)


# ===========================================================================
# The WP1.5 evaluation figure set -- P1, S1, F1, S5, D1
# (docs/plans/esmda_turbulence_evaluation.md section 7)
#
# A different contract from the general plots above, which stay as they are:
# these take already-opened objects, write the file themselves and return the
# path written -- or ``None`` when their inputs are absent, empty or degenerate
# (master-plan invariant 3), which is never an exception. Every colour, band
# and save goes through ``evaluation.style`` rather than the legacy ``_COLOR_*``
# constants above, so a prior/posterior pair cannot end up on two scales.
# ===========================================================================

# Nested quantile bands: the levels ``compute_esmda_metrics`` stores at the
# station columns (its ``STATION_QUANTILES``), and S5's default fan.
_BAND_QUANTILES = (0.05, 0.25, 0.5, 0.75, 0.95)

# Solid (masked) cells are drawn *outside* the colour scale, never inside it.
_SOLID_COLOR = "#D9D9D9"

# Below this many members a KDE says more about the kernel than about the
# posterior, so the marginal becomes a box + strip of the members themselves.
_MIN_VIOLIN_MEMBERS = 8

# Cell-centre vertical dim names, as ``eval_fields.nc``'s ``extrapolated_edges``
# spells them (``z`` for PALM/pylbm, ``zt`` for uDALES).
_VERTICAL_CENTRE_DIMS = ("z", "zt")

# The streamwise velocity component, which S1 profiles and F1 slices. ``u`` is
# streamwise only because every shipped case puts the inflow along +x; a case
# with a different inflow axis needs its profiles rethought, not relabelled, so
# this is a stated assumption rather than a caller's knob.
_STREAMWISE = "u"


@contextlib.contextmanager
def _styled() -> Iterator[None]:
    """The shared rcParams, scoped to one figure.

    ``apply_style`` mutates the global rcParams; the surrounding ``rc_context``
    is what keeps a figure from leaking its style into the caller's process.
    """
    with plt.rc_context():
        apply_style()
        yield


def _finite(values: np.ndarray) -> np.ndarray:
    """The finite entries of ``values``, flattened."""
    flat = np.asarray(values, dtype=float).ravel()
    kept: np.ndarray = flat[np.isfinite(flat)]  # boolean indexing is typed Any
    return kept


# ---------------------------------------------------------------------------
# P1 -- parameter marginals
# ---------------------------------------------------------------------------


def _marginal_members(
    posterior_params: xarray.Dataset,
    true_params: xarray.Dataset | None,
    prior_params: xarray.Dataset | None,
) -> Iterator[tuple[str, np.ndarray, np.ndarray | None, np.ndarray | None]]:
    """Yield ``(name, posterior (M, K), truth (K,) | None, prior (M, K) | None)``.

    With a truth this is :func:`evaluation.scores._aligned_parameter_members`,
    which owns the knot-grid alignment and the guard that drops a prior sampled
    on a different grid -- so P1 annotates the z-score of exactly the knots the
    metric block scored. Without a truth there is nothing to align, so the same
    parameter selection and the same prior guard are applied directly: a run
    whose truth was never saved still has a contraction worth drawing.
    """
    if true_params is not None:
        for name, _x, members, truth, prior in _aligned_parameter_members(
            posterior_params, true_params, prior_params
        ):
            yield name, members, truth, prior
        return

    for name in _plotted_param_names(posterior_params):
        _x, members = _param_members_and_x(posterior_params[name])
        prior = None
        if prior_params is not None and name in prior_params.data_vars:
            _, candidate = _param_members_and_x(prior_params[name])
            if candidate.shape[1] == members.shape[1]:
                prior = candidate
        yield name, members, None, prior


def _param_axis_label(name: str) -> str:
    """The parameter's labelled axis title, units included when they are known."""
    if name in PARAM_LABELS:
        return PARAM_LABELS[name]
    label = name.replace("_", " ").capitalize()
    return f"{label} [{PARAM_UNITS[name]}]" if name in PARAM_UNITS else label


def _draw_marginal(ax: Axes, position: float, values: np.ndarray, color: str) -> None:
    """One marginal at ``position``: a violin, or a box + strip when a KDE lies.

    A violin needs both enough members for the KDE and a non-zero spread -- a
    *pinned* parameter has every member identical by construction, and
    ``gaussian_kde`` raises on that singular covariance rather than drawing a
    line.
    """
    values = _finite(values)
    if values.size == 0:
        return

    if values.size >= _MIN_VIOLIN_MEMBERS and np.ptp(values) > 0:
        parts = ax.violinplot(
            [values], positions=[position], widths=0.7, showextrema=False
        )
        # matplotlib types every entry of that dict as a single ``Collection``;
        # ``bodies`` is in fact one polygon per violin.
        for body in cast(list[PolyCollection], parts["bodies"]):
            body.set_facecolor(color)
            body.set_edgecolor(color)
            body.set_alpha(0.45)
    else:
        ax.boxplot(
            [values],
            positions=[position],
            widths=0.45,
            patch_artist=True,
            boxprops={"facecolor": color, "alpha": 0.35, "edgecolor": color},
            medianprops={"color": color, "linewidth": 1.6},
            whiskerprops={"color": color},
            capprops={"color": color},
            showfliers=False,
        )
        # The members themselves, jittered off the box so ties stay countable.
        jitter = np.linspace(-0.13, 0.13, values.size)
        ax.scatter(
            position + jitter,
            values,
            s=14,
            color=color,
            alpha=0.85,
            edgecolor="white",
            linewidth=0.4,
            zorder=4,
        )

    ax.scatter(
        [position],
        [values.mean()],
        marker="_",
        s=260,
        color=color,
        linewidth=1.8,
        zorder=5,
    )


def _truth_is_static(truth: np.ndarray | None) -> bool:
    """Whether a parameter's truth is the same value at every knot.

    The discriminator between the two knot pairings P1 can draw. A *static*
    parameter estimated over ``W`` windows still arrives with ``K = W`` knots
    (``run_esmda.py`` stacks one point per window along ``time``), so the knot
    count alone cannot tell the two apart -- but a constant truth can.
    """
    if truth is None:
        return False
    finite = _finite(truth)
    return bool(finite.size) and bool(np.allclose(finite, finite[0]))


def _z_annotation(bundle: dict[str, np.ndarray] | None, knot: int) -> str:
    """P1's z label, distinguishing the reasons a z cannot be shown.

    A bare ``z = n/a`` used to cover three opposite diagnoses -- no truth was
    saved, the posterior collapsed, the parameter was pinned -- so a reader could
    not tell a missing artifact from a failed assimilation. Each gets its own
    string, told apart by the bundle's own scales: a pinned parameter has no
    prior spread either, a collapsed one started with some.
    """
    if bundle is None:
        return "z: no truth"

    z_value = float(bundle["z_score"][knot])
    if np.isfinite(z_value):
        return f"z = {z_value:.2f}"

    post_std = float(bundle["posterior_std"][knot])
    if not np.isfinite(post_std) or post_std > 0.0:
        # A finite, non-zero spread with a non-finite z means the *truth* is
        # missing at this knot; a non-finite spread means the members are.
        return "z = n/a"
    prior_std = bundle.get("prior_std")
    if prior_std is not None and float(prior_std[knot]) == 0.0:
        return "z = n/a (pinned)"
    return "z = n/a (posterior collapsed)"


def plot_parameter_marginals(
    posterior_params: xarray.Dataset,
    true_params: xarray.Dataset | None,
    output_path: str | pathlib.Path,
    *,
    prior_params: xarray.Dataset | None = None,
) -> pathlib.Path | None:
    """P1: prior vs posterior marginal per parameter, truth dashed, z annotated.

    One panel per estimated parameter, each holding the prior marginal (grey)
    beside the posterior one (teal) -- a violin where a KDE is meaningful, a box
    plus the members themselves below :data:`_MIN_VIOLIN_MEMBERS`. The y-limits
    span **both** marginals and the truth: autoscaling to the posterior alone
    hides the contraction, which is the one thing this figure exists to show.

    **Which knot each marginal comes from depends on what the parameter is**,
    and both are labelled so no reader has to guess. ``prior_params.nc`` stacks
    *every* window's prior along ``time``, so on a multi-window run its final
    knot is window ``W-1``'s prior -- which is window ``W-2``'s posterior, and
    shows almost no contraction against the posterior beside it:

    * A **static** parameter (:func:`_truth_is_static`: the truth is the same at
      every knot, so the ``K`` knots are ``K`` windows of one quantity) draws the
      prior at **knot 0** -- the run's actual prior -- against the posterior at
      the final knot. That is the total contraction the run achieved.
    * A genuinely **time-varying** parameter keeps the **same (final) knot** for
      both. Knot 0 is a different physical time there, so a knot-0 prior would
      compare two different quantities; per-window contraction is all that is
      available, and the labels say so.

    Without a truth the two cannot be told apart, so the same-knot pair is drawn
    (the conservative reading) and labelled as such.

    Note this is *not* the same ``final`` as ``run_summary.yaml``'s
    ``contraction_ratio``, which is per knot and so stays per window on a static
    multi-window run. The figure's subject is the run; the YAML's is the window.

    The annotated ``z = (theta* - mean_post)/sigma_post`` comes from
    :func:`evaluation.scores.parameter_bundle` for the posterior's knot, and
    :func:`_z_annotation` names the reason whenever it cannot be shown rather
    than printing an infinity.

    Returns the path written, or ``None`` when no parameter carries finite
    members.
    """
    entries = [
        entry
        for entry in _marginal_members(posterior_params, true_params, prior_params)
        if np.isfinite(entry[1]).any()
    ]
    if not entries:
        logger.info("plot_parameter_marginals: no parameter with finite members")
        return None

    with _styled():
        fig, axes = plt.subplots(
            1,
            len(entries),
            figsize=(3.5 * len(entries), 4.4),
            squeeze=False,
            constrained_layout=True,
        )
        for ax, (name, post, truth, prior) in zip(axes[0], entries):
            n_knots = post.shape[1]
            knot = n_knots - 1  # the posterior's final knot; see the docstring
            static = _truth_is_static(truth)
            prior_knot = 0 if (static and n_knots > 1) else knot

            post_k = post[:, knot]
            prior_k = None if prior is None else prior[:, prior_knot]
            truth_k = None if truth is None else float(truth[knot])

            if n_knots == 1:
                prior_label, post_label, title = "Prior", "Posterior", name
            elif static:
                prior_label = "Prior\n(window 0)"
                post_label = "Posterior\n(final window)"
                title = f"{name} (static, {n_knots} windows)"
            else:
                prior_label = "Prior\n(final knot)"
                post_label = "Posterior\n(final knot)"
                title = f"{name} (knot {knot} of {n_knots})"

            positions = []
            if prior_k is not None:
                _draw_marginal(ax, 0.0, prior_k, COLORS["prior"])
                positions.append((0.0, prior_label))
            _draw_marginal(ax, 1.0, post_k, COLORS["posterior"])
            positions.append((1.0, post_label))

            if truth_k is not None and np.isfinite(truth_k):
                ax.axhline(
                    truth_k,
                    color=COLORS["truth"],
                    linestyle="--",
                    linewidth=1.5,
                    zorder=3,
                )

            bundle = None
            if truth is not None:
                with warnings.catch_warnings():
                    # An ``inf`` member makes the mean and std non-finite; the
                    # result is displayed as a named "no z" string, so the
                    # numpy warning would only be noise on the operator's
                    # console.
                    warnings.simplefilter("ignore", RuntimeWarning)
                    bundle = parameter_bundle(post, prior, truth)
            ax.annotate(
                _z_annotation(bundle, knot),
                xy=(0.5, 0.98),
                xycoords="axes fraction",
                ha="center",
                va="top",
                fontsize=9,
                color=COLORS["charcoal"],
            )

            limits = finite_limits(
                post_k,
                prior_k,
                None if truth_k is None else np.array([truth_k]),
                pad=0.10,
            )
            if limits is not None:
                ax.set_ylim(*limits)
            ax.set_xticks([p for p, _ in positions])
            ax.set_xticklabels([label for _, label in positions], fontsize=8)
            ax.set_xlim(-0.6, 1.6)
            ax.set_ylabel(_param_axis_label(name))
            ax.set_title(title, loc="left")

        handles = [
            Patch(facecolor=COLORS["prior"], alpha=0.45, label="Prior"),
            Patch(facecolor=COLORS["posterior"], alpha=0.45, label="Posterior"),
            Line2D([0], [0], color=COLORS["truth"], ls="--", lw=1.5, label="Truth"),
        ]
        axes[0, 0].legend(handles=handles, loc="lower left", fontsize=8)
        fig.suptitle("Parameter marginals: prior vs posterior")
        return save_png(fig, output_path)


# ---------------------------------------------------------------------------
# S1 -- vertical profiles at the station columns
# ---------------------------------------------------------------------------


def _extrapolated_axes(fields: xarray.Dataset) -> tuple[str, ...]:
    """The centre axes whose last index ``eval_fields.nc`` flags as extrapolated."""
    raw = str(fields.attrs.get("extrapolated_edges", ""))
    return tuple(name for name in (part.strip() for part in raw.split(",")) if name)


def _station_order(fields: xarray.Dataset, max_stations: int) -> list[int]:
    """Station indices to draw: held-out columns first, then the assimilated ones.

    A profile drawn only where the assimilation was fitted is the least
    informative one available, so a validation column never loses its slot to an
    assimilation column. Anything dropped is logged -- silent truncation would
    make a figure that omits half the evidence look complete.
    """
    sets = (
        [str(s) for s in np.asarray(fields["station_set"].values).ravel()]
        if "station_set" in fields.coords
        else [""] * int(fields.sizes["station"])
    )
    held_out = [i for i, s in enumerate(sets) if s != "assimilation"]
    assimilated = [i for i, s in enumerate(sets) if s == "assimilation"]
    order = held_out + assimilated
    if len(order) > max_stations:
        dropped = order[max_stations:]
        logger.info(
            "plot_station_profiles: %d of %d station columns not drawn (max_stations"
            "=%d): %s",
            len(dropped),
            len(order),
            max_stations,
            ", ".join(f"{i} ({sets[i]})" for i in dropped),
        )
    return order[:max_stations]


def _station_profile(
    fields: xarray.Dataset,
    variable: str,
    station: int,
    component: str | None = None,
) -> np.ndarray | None:
    """A station column's profile, ``(z,)`` or ``(quantile, z)``; ``None`` if absent.

    A *requested* component that the variable does not carry returns ``None``
    rather than falling back to component 0: silently drawing ``u`` under a ``w``
    label is worse than drawing nothing, and both callers already treat ``None``
    as "this curve is not available".
    """
    if variable not in fields.data_vars:
        return None
    da = fields[variable].isel(station=station)
    if "component" in da.dims:
        available = [str(c) for c in np.asarray(da["component"].values).ravel()]
        if component is None:
            da = da.isel(component=0)
        elif component in available:
            da = da.sel(component=component)
        else:
            logger.info(
                "plot_station_profiles: %s has no component %r (has %s); "
                "the profile is not drawn",
                variable,
                component,
                ", ".join(available) or "none",
            )
            return None
    return np.asarray(da.transpose(..., "z").values, dtype=float)


def plot_station_profiles(
    fields: xarray.Dataset,
    output_path: str | pathlib.Path,
    *,
    u_ref: float | None = None,
    building_height: float | None = None,
    max_stations: int = 6,
) -> pathlib.Path | None:
    """S1: mean-velocity and TKE profiles at the station columns (the LES figure).

    ``fields`` is an already-opened ``eval_fields.nc`` (WP1.4). Rows are the
    streamwise time-mean velocity and the resolved TKE, columns are station
    columns labelled with their sensor set and ``(x, y)``; the truth is a black
    line, the ensembles are nested quantile bands (5--95 % light, 25--75 % dark)
    about their median, teal for the posterior and grey for the prior. An inset
    plan view marks which column each panel draws.

    Both axes are non-dimensionalised when the caller supplies the scales:
    ``z/H`` with a roof line at 1 given ``building_height``, ``u/U_ref`` (and
    ``k/U_ref^2``) given ``u_ref``. Without them the panels are in metres and
    m/s and say so.

    The **last z index is dropped** when ``extrapolated_edges`` names the
    vertical axis: colocation fills that cell by extrapolating from the two
    faces below it, which inflates its second moments by ~20 % and up to 5x, and
    it is the TKE row that reads it. The caption records the exclusion.

    The mean row is :data:`_STREAMWISE` (``u``), which is the streamwise
    component only because every shipped case puts the inflow along +x. That is
    an assumption of the figure, not a knob: a case with a different inflow axis
    needs its profiles rethought rather than relabelled.

    Returns the path written, or ``None`` when the dataset carries no station
    columns or no profile variables.
    """
    if "station" not in fields.dims or int(fields.sizes["station"]) == 0:
        logger.info("plot_station_profiles: no station columns in the dataset")
        return None

    prefixes = [
        p
        for p in ("truth", "prior", "posterior")
        if any(f"{p}_station_{quantity}" in fields for quantity in ("mean", "tke"))
    ]
    rows = [
        quantity
        for quantity in ("mean", "tke")
        if any(f"{p}_station_{quantity}" in fields for p in prefixes)
    ]
    if not rows:
        logger.info("plot_station_profiles: no station profile variables present")
        return None

    z = np.asarray(fields["z"].values, dtype=float)
    trimmed = bool(set(_extrapolated_axes(fields)) & set(_VERTICAL_CENTRE_DIMS))
    keep = slice(0, -1) if trimmed and z.size > 1 else slice(None)
    z = z[keep]
    z_plot = z / building_height if building_height else z
    z_label = "z/H [-]" if building_height else "z [m]"

    scale = {
        "mean": 1.0 if u_ref is None else 1.0 / u_ref,
        "tke": 1.0 if u_ref is None else 1.0 / u_ref**2,
    }
    row_label = {
        "mean": r"$\bar{u}/U_{ref}$ [-]" if u_ref else r"$\bar{u}$ [m/s]",
        "tke": r"$k/U_{ref}^2$ [-]" if u_ref else r"$k$ [m$^2$/s$^2$]",
    }
    stations = _station_order(fields, max_stations)
    sets = (
        [str(s) for s in np.asarray(fields["station_set"].values).ravel()]
        if "station_set" in fields.coords
        else ["" for _ in range(int(fields.sizes["station"]))]
    )
    station_x = np.asarray(fields["station_x"].values, dtype=float)
    station_y = np.asarray(fields["station_y"].values, dtype=float)
    levels = (
        np.asarray(fields["quantile"].values, dtype=float)
        if "quantile" in fields.coords
        else np.asarray(_BAND_QUANTILES)
    )

    with _styled():
        fig, axes = plt.subplots(
            len(rows),
            len(stations),
            figsize=(2.9 * len(stations), 3.6 * len(rows)),
            squeeze=False,
            sharey=True,
            constrained_layout=True,
        )
        for r, quantity in enumerate(rows):
            for c, station in enumerate(stations):
                ax = axes[r][c]
                drawn: list[np.ndarray] = []
                for prefix in prefixes:
                    name = f"{prefix}_station_{quantity}"
                    color = COLORS["truth" if prefix == "truth" else prefix]
                    bands = _station_profile(
                        fields, f"{name}_quantile", station, _STREAMWISE
                    )
                    if bands is not None:
                        bands = bands[:, keep] * scale[quantity]
                        nested_bands(
                            ax,
                            z_plot,
                            bands,
                            levels,
                            color,
                            orient="horizontal",
                            label=prefix.capitalize(),
                        )
                        drawn.append(bands)
                        continue
                    profile = _station_profile(fields, name, station, _STREAMWISE)
                    if profile is None:
                        continue
                    profile = profile[keep] * scale[quantity]
                    ax.plot(
                        profile,
                        z_plot,
                        color=color,
                        lw=1.8,
                        label=prefix.capitalize(),
                        zorder=5 if prefix == "truth" else 4,
                    )
                    drawn.append(profile)

                if building_height:
                    ax.axhline(1.0, color=COLORS["charcoal"], ls=":", lw=1.0, zorder=2)
                    ax.annotate(
                        "z/H = 1",
                        xy=(0.02, 1.0),
                        xycoords=("axes fraction", "data"),
                        va="bottom",
                        fontsize=7,
                        color=COLORS["charcoal"],
                    )
                limits = finite_limits(*drawn, pad=0.05)
                if limits is not None:
                    ax.set_xlim(*limits)
                ax.set_xlabel(row_label[quantity])
                if c == 0:
                    ax.set_ylabel(z_label)
                if r == 0:
                    ax.set_title(
                        f"{sets[station] or 'station'} #{station}\n"
                        f"(x={station_x[station]:.0f}, y={station_y[station]:.0f})",
                        fontsize=9,
                    )
                    # Plan view in the lower right: a boundary-layer profile
                    # leaves that corner empty, and the inset needs its own
                    # opaque background against the bands behind it.
                    inset = ax.inset_axes((0.64, 0.06, 0.34, 0.30))
                    inset.scatter(station_x, station_y, s=6, color=COLORS["prior"])
                    inset.scatter(
                        [station_x[station]],
                        [station_y[station]],
                        s=22,
                        color=COLORS["orange"],
                        zorder=3,
                    )
                    inset.set_facecolor("white")
                    inset.patch.set_alpha(0.9)
                    inset.margins(0.2)
                    inset.set_xticks([])
                    inset.set_yticks([])
                    inset.set_xlabel("plan view", fontsize=6, labelpad=1)

        handles, labels = axes[0][0].get_legend_handles_labels()
        if handles:
            axes[0][0].legend(handles, labels, loc="upper left", fontsize=8)

        caption = "Time-averaged profiles; ensembles as 5-95 % / 25-75 % bands."
        if trimmed:
            caption += (
                " Top z cell excluded: colocation extrapolates it "
                f"({', '.join(_extrapolated_axes(fields))})."
            )
        fig.supxlabel(caption, fontsize=8, color=COLORS["charcoal"])
        fig.suptitle("Vertical profiles at the station columns")
        return save_png(fig, output_path)


# ---------------------------------------------------------------------------
# F1 -- time-averaged horizontal slices
# ---------------------------------------------------------------------------


def _masked_cmap(name: str) -> Colormap:
    """A copy of ``name`` whose masked (solid) cells get their own flat colour."""
    cmap = plt.get_cmap(name).copy()
    cmap.set_bad(_SOLID_COLOR)
    return cmap


def _slab_component(
    fields: xarray.Dataset, variable: str, component: str
) -> np.ndarray | None:
    """A slab variable as ``(zlev, y, x)`` for one component; ``None`` if absent.

    A variable that does not carry the requested component returns ``None``
    rather than component 0 -- drawing ``u`` under a ``w`` label is worse than
    drawing nothing, and the caller already handles a missing column.
    """
    if variable not in fields.data_vars:
        return None
    da = fields[variable]
    if "component" in da.dims:
        available = [str(c) for c in np.asarray(da["component"].values).ravel()]
        if component not in available:
            logger.info(
                "plot_mean_slices: %s has no component %r (has %s); "
                "the column is not drawn",
                variable,
                component,
                ", ".join(available) or "none",
            )
            return None
        da = da.sel(component=component)
    return np.asarray(da.transpose("zlev", "y", "x").values, dtype=float)


def plot_mean_slices(
    fields: xarray.Dataset,
    output_path: str | pathlib.Path,
    *,
    max_levels: int = 3,
) -> pathlib.Path | None:
    """F1: time-mean horizontal slices, truth | prior | posterior | difference.

    Rows are up to ``max_levels`` evenly spaced z-levels of ``eval_fields.nc``'s
    slabs; columns are the truth, the prior ensemble mean (dropped when the run
    did not save its prior state), the posterior ensemble mean and the
    posterior-minus-truth difference, all of the :data:`_STREAMWISE` component.
    **The first three columns share one ``Normalize``** over the finite fluid
    cells of all of them -- an unshared colour scale across a prior/posterior
    pair is one of the three mistakes the metrics doc names -- and the difference
    gets its own symmetric diverging norm centred on zero.

    Solid cells (``slab_fluid == 0``) are masked to :data:`_SOLID_COLOR` before
    anything is scaled, so an obstacle interior held at ~0 neither colours a
    panel nor compresses the scale of the flow around it.

    Finiteness is checked **per column**, not over their union: a single
    diverged member makes ``posterior_slab_mean`` all-NaN while the truth beside
    it is fine, and an all-NaN column would otherwise render entirely in the
    solid-cell grey under a caption saying grey means solid. Such a column is
    dropped with a log line instead -- grey must not mean two things on one
    figure.

    Unlike S1 this figure deliberately does **not** apply
    ``extrapolated_edges``: these are first moments, where the extrapolation
    artefact is small. The exclusion matters for the second moments S1 draws.

    These are the accumulated time means only -- **never an instantaneous
    field**, which decorrelates after a Lyapunov horizon and would measure chaos
    rather than parameter quality. The averaging window is annotated from the
    file's ``t_start`` / ``t_end``.

    Returns the path written, or ``None`` without a posterior slab or without a
    single finite fluid cell to scale.
    """
    posterior = _slab_component(fields, "posterior_slab_mean", _STREAMWISE)
    if posterior is None:
        logger.info("plot_mean_slices: no posterior_slab_mean in the dataset")
        return None
    if "zlev" not in fields.dims or int(fields.sizes["zlev"]) == 0:
        # ``evenly_spaced_levels`` raises on a zero-length axis, which would
        # abort the whole figure stage rather than skip this figure.
        logger.info("plot_mean_slices: no z-levels in the slabs")
        return None
    truth = _slab_component(fields, "truth_slab_mean", _STREAMWISE)
    prior = _slab_component(fields, "prior_slab_mean", _STREAMWISE)

    solid = None
    if "slab_fluid" in fields.data_vars:
        solid = np.asarray(fields["slab_fluid"].values) == 0

    def masked(values: np.ndarray | None) -> np.ndarray | None:
        if values is None or solid is None:
            return values
        return np.where(solid, np.nan, values)

    masked_truth = masked(truth)
    masked_posterior = masked(posterior)

    def drawable(label: str, values: np.ndarray) -> bool:
        """Whether a column has anything to show, logged per column when not."""
        if not np.isfinite(values).any():
            logger.info(
                "plot_mean_slices: %s has no finite fluid cell and is not drawn "
                "(check for a diverged member)",
                label,
            )
            return False
        return True

    columns: list[tuple[str, np.ndarray]] = []
    for label, values in (
        ("Truth", masked_truth),
        ("Prior mean", masked(prior)),
        ("Posterior mean", masked_posterior),
    ):
        if values is not None and drawable(label, values):
            columns.append((label, values))
    difference = None
    if masked_truth is not None and masked_posterior is not None:
        candidate = masked_posterior - masked_truth
        if drawable("Posterior - truth", candidate):
            difference = candidate

    limits = finite_limits(*[values for _, values in columns])
    if limits is None:
        logger.info("plot_mean_slices: no finite fluid cell in the slabs")
        return None
    vmin, vmax = limits
    if vmax <= vmin:  # a uniform field: a zero-width norm colours nothing
        vmin, vmax = vmin - 0.5, vmax + 0.5
    diff_max = 1.0
    if difference is not None:
        finite_diff = _finite(difference)
        if finite_diff.size and np.max(np.abs(finite_diff)) > 0:
            diff_max = float(np.max(np.abs(finite_diff)))

    n_levels = int(fields.sizes["zlev"])
    levels = evenly_spaced_levels(n_levels, min(max_levels, n_levels))
    z_values = np.asarray(fields["zlev"].values, dtype=float)
    x = np.asarray(fields["x"].values, dtype=float)
    y = np.asarray(fields["y"].values, dtype=float)
    extent = (float(x.min()), float(x.max()), float(y.min()), float(y.max()))
    n_cols = len(columns) + (1 if difference is not None else 0)

    with _styled():
        fig, axes = plt.subplots(
            len(levels),
            n_cols,
            figsize=(3.4 * n_cols, 3.2 * len(levels)),
            squeeze=False,
            constrained_layout=True,
        )
        field_image = difference_image = None
        for r, level in enumerate(levels):
            for c, (label, values) in enumerate(columns):
                field_image = axes[r][c].imshow(
                    values[level],
                    origin="lower",
                    extent=extent,
                    aspect="equal",
                    cmap=_masked_cmap(CMAP_FIELD),
                    vmin=vmin,
                    vmax=vmax,
                )
                if r == 0:
                    axes[r][c].set_title(label, fontsize=10)
            if difference is not None:
                ax = axes[r][len(columns)]
                difference_image = ax.imshow(
                    difference[level],
                    origin="lower",
                    extent=extent,
                    aspect="equal",
                    cmap=_masked_cmap(CMAP_DIFF),
                    vmin=-diff_max,
                    vmax=diff_max,
                )
                if r == 0:
                    ax.set_title("Posterior - truth", fontsize=10)
            axes[r][0].set_ylabel(f"z = {z_values[level]:.1f} m\ny [m]")
            for ax in axes[r]:
                ax.set_xlabel("x [m]")
                ax.grid(False)

        unit = f"{_STREAMWISE} [m/s]"
        if field_image is not None:
            fig.colorbar(
                field_image,
                ax=[
                    axes[r][c] for r in range(len(levels)) for c in range(len(columns))
                ],
                fraction=0.03,
                pad=0.02,
                label=f"time-mean {unit}",
            )
        if difference_image is not None:
            fig.colorbar(
                difference_image,
                ax=[axes[r][len(columns)] for r in range(len(levels))],
                fraction=0.05,
                pad=0.02,
                label=f"difference {unit}",
            )

        span = ""
        if "t_start" in fields.attrs and "t_end" in fields.attrs:
            span = (
                f"  (t = {float(fields.attrs['t_start']):.0f}"
                f"-{float(fields.attrs['t_end']):.0f} s)"
            )
        fig.suptitle(f"Time-mean {_STREAMWISE} on horizontal slices{span}")

        caption = "Time averages, never instantaneous."
        if solid is not None:
            caption += " Solid cells masked (grey) and excluded from the norm."
        stride = int(fields.attrs.get("horizontal_stride", 1) or 1)
        if stride > 1:
            caption += f" Horizontal stride {stride}."
        fig.supxlabel(caption, fontsize=8, color=COLORS["charcoal"])
        return save_png(fig, output_path)


# ---------------------------------------------------------------------------
# S5 -- sensor time-series fans
# ---------------------------------------------------------------------------


def _sensor_time_axis(times: np.ndarray | None, n_time: int) -> tuple[np.ndarray, str]:
    """``(x, axis label)`` for a sensor panel: the given times, or a frame index."""
    if times is not None:
        return np.asarray(times, dtype=float).ravel(), "Time [s]"
    return np.arange(n_time, dtype=float), "Frame"


def plot_sensor_fans(
    truth: dict[str, np.ndarray],
    ensemble: dict[str, np.ndarray],
    output_path: str | pathlib.Path,
    *,
    times: np.ndarray | None = None,
    obs_error_std: float | None = None,
    window_edges: np.ndarray | list[float] | None = None,
    max_sensors: int = 4,
) -> pathlib.Path | None:
    """S5: quantile fans at the sensors, assimilated and held-out side by side.

    ``truth[set]`` is ``(time, sensor)`` and ``ensemble[set]`` is
    ``(ensemble, time, sensor)`` -- what
    :func:`evaluation.sensors.sensor_magnitude` produces. **Columns are the
    sensor sets**, labelled: the held-out column beside the assimilated one is
    the strongest anti-overfitting evidence in the suite, which it only is if a
    reader can see which is which. Rows are sensors, up to ``max_sensors``;
    anything dropped is logged. Window boundaries are marked, and the truth
    carries a ``+/- obs_error_std`` envelope when the caller knows that width.

    The fan is :data:`_BAND_QUANTILES` of the members that are **finite across
    the whole window**. A member that diverged mid-window is dropped, logged, and
    subtracted from the surviving-member count annotated on the panel: quantiles
    propagate NaN, so one such member used to erase the whole fan and leave a
    bare truth line with nothing to say why. A panel no-ops only when no member
    survives.

    The truth is drawn on the ensemble's own time axis. A truth series of a
    different length is skipped with a log rather than resampled onto an invented
    axis -- the only honest options are the ensemble's axis or none, and
    :func:`evaluation.scores.compute_sensor_metrics` (which the wiring goes
    through) has already aligned them.

    **Pre-WP2.1 caveat, stated in the figure itself:** the realized noisy
    observations the filter actually assimilated are not persisted yet, so the
    envelope is the *clean* truth plus or minus the nominal observation error --
    the assimilated values scatter around it. WP2.1 swaps the source.

    Returns the path written, or ``None`` when no sensor set carries a finite
    ensemble series.
    """
    sets = []
    for name, members in ensemble.items():
        values = np.asarray(members, dtype=float)
        if values.ndim == 3 and values.size and np.isfinite(values).any():
            sets.append((name, values))
        else:
            logger.info("plot_sensor_fans: sensor set %r has no finite series", name)
    if not sets:
        logger.info("plot_sensor_fans: no sensor set with a finite ensemble series")
        return None

    n_rows = min(max_sensors, max(values.shape[2] for _, values in sets))
    for name, values in sets:
        if values.shape[2] > n_rows:
            logger.info(
                "plot_sensor_fans: %d of %d sensors not drawn for set %r "
                "(max_sensors=%d)",
                values.shape[2] - n_rows,
                values.shape[2],
                name,
                max_sensors,
            )

    with _styled():
        fig, axes = plt.subplots(
            n_rows,
            len(sets),
            figsize=(5.4 * len(sets), 2.5 * n_rows),
            squeeze=False,
            constrained_layout=True,
        )
        obs_drawn = False
        edges = None if window_edges is None else np.asarray(window_edges, dtype=float)
        for c, (name, members) in enumerate(sets):
            t, t_label = _sensor_time_axis(times, members.shape[1])
            truth_series = truth.get(name)
            if truth_series is not None:
                truth_series = np.asarray(truth_series, dtype=float)
                if truth_series.shape[0] != t.size:
                    # The only axes this figure can honestly put the truth on are
                    # the ensemble's own and none at all; resampling it onto
                    # ``linspace(t[0], t[-1], ...)`` assumes a uniform cadence
                    # over exactly the ensemble's span, and is wrong by O(1) on a
                    # unit signal when that does not hold.
                    logger.info(
                        "plot_sensor_fans: set %r has %d truth frames but %d "
                        "ensemble frames; the truth line is not drawn",
                        name,
                        truth_series.shape[0],
                        t.size,
                    )
                    truth_series = None

            for r in range(n_rows):
                ax = axes[r][c]
                if r >= members.shape[2]:
                    ax.set_axis_off()
                    continue

                panel = members[:, :, r]
                # Quantiles propagate NaN, so a member that is non-finite
                # anywhere in the window would blank the whole fan. Drop it
                # instead and say how many are left.
                usable = np.isfinite(panel).all(axis=1)
                n_kept, n_total = int(usable.sum()), int(panel.shape[0])
                if n_kept < n_total:
                    logger.info(
                        "plot_sensor_fans: set %r sensor %d: %d of %d members are "
                        "not finite across the window and are dropped from the fan",
                        name,
                        r,
                        n_total - n_kept,
                        n_total,
                    )

                drawn: list[np.ndarray] = []
                if n_kept:
                    bands = np.quantile(panel[usable], _BAND_QUANTILES, axis=0)
                    nested_bands(
                        ax,
                        t,
                        bands,
                        _BAND_QUANTILES,
                        COLORS["posterior"],
                        label="Posterior median",
                    )
                    drawn.append(bands)
                ax.annotate(
                    f"M = {n_kept}/{n_total}",
                    xy=(0.99, 0.97),
                    xycoords="axes fraction",
                    ha="right",
                    va="top",
                    fontsize=7,
                    color=COLORS["charcoal"],
                )

                truth_r = None
                if truth_series is not None and r < truth_series.shape[1]:
                    truth_r = truth_series[:, r]
                    ax.plot(
                        t,
                        truth_r,
                        color=COLORS["truth"],
                        lw=1.5,
                        label="Truth",
                        zorder=6,
                    )
                    drawn.append(truth_r)

                if truth_r is not None and obs_error_std:
                    obs_drawn = True
                    ax.fill_between(
                        t,
                        truth_r - obs_error_std,
                        truth_r + obs_error_std,
                        color=COLORS["orange"],
                        alpha=0.18,
                        lw=0,
                        label=r"Truth $\pm\sigma_o$",
                        zorder=2,
                    )

                limits = finite_limits(*drawn, pad=0.08)
                if limits is not None:
                    ax.set_ylim(*limits)
                mark_windows(ax, edges, annotate=(r == 0))
                ax.set_ylabel(f"sensor {r}\n|U| [m/s]")
                ax.margins(x=0.01)
                if r == 0:
                    ax.set_title(f"{name} sensors", loc="left")
                    # Lower left: the window indices sit along the top edge and
                    # the fan through the middle.
                    ax.legend(loc="lower left", fontsize=8, ncol=3)
                if r == n_rows - 1:
                    ax.set_xlabel(t_label)

        fig.suptitle("Sensor time series: posterior quantile fan vs truth")
        if obs_drawn:
            # Only claimed when the envelope was actually drawn -- a caption
            # about observations on a figure that has none is its own kind of
            # wrong.
            fig.supxlabel(
                "The orange envelope is the clean truth +/- the nominal sigma_o: "
                "the realized noisy assimilated observations are not persisted "
                "before WP2.1.",
                fontsize=8,
                color=COLORS["charcoal"],
            )
        return save_png(fig, output_path)


# ---------------------------------------------------------------------------
# D1 -- rank histogram
# ---------------------------------------------------------------------------


def _pool_rank_counts(statistics: dict[str, list[int]]) -> np.ndarray | None:
    """Sum the per-statistic rank-count vectors of one (set, half) cell.

    Per-window statistics are already ~independent samples, so pooling over
    statistics, sensors and windows is what makes the counts numerous enough to
    read.

    Every vector in one cell has the same length by construction: the writer
    (``_score_window_statistic``) bins with ``minlength=n_members + 1`` and one
    ``(set, half)`` cell has a single ensemble size, so a ragged cell is a
    programming error rather than a run-dir degradation and is left to raise.
    """
    vectors = [
        np.asarray(counts, dtype=float).ravel()
        for counts in (statistics or {}).values()
        if counts is not None and np.size(counts) > 0
    ]
    if not vectors:
        return None

    pooled: np.ndarray = np.sum(vectors, axis=0)
    if pooled.size < 2 or pooled.sum() <= 0:
        return None
    return pooled


def _coarsen_rank_counts(
    counts: np.ndarray, n_bins: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Group ``M+1`` rank bins into ``min(n_bins, M+1)`` contiguous groups.

    Pooled rank counts are small (~50--300), so ``M+1`` bins would draw sampling
    noise rather than a shape. ``np.array_split`` leaves the groups **unequal**
    in width whenever ``M+1`` is not divisible, so the uniform reference is
    per group: ``expected_g = N * size_g/(M+1)`` with a binomial consistency
    band ``2*sqrt(N*p_g*(1-p_g))``, ``p_g = size_g/(M+1)``. A flat line across
    unequal groups would flag a calibrated ensemble as biased.

    Returns ``(sizes, grouped_counts, expected, band)``, all ``(G,)``.
    """
    counts = np.asarray(counts, dtype=float).ravel()
    n_ranks = counts.size
    groups = np.array_split(np.arange(n_ranks), min(max(n_bins, 1), n_ranks))
    sizes = np.array([g.size for g in groups], dtype=float)
    grouped = np.array([counts[g].sum() for g in groups], dtype=float)
    total = grouped.sum()
    p = sizes / n_ranks
    return sizes, grouped, total * p, 2.0 * np.sqrt(total * p * (1.0 - p))


def plot_rank_histogram(
    rank_counts: dict[str, dict[str, dict[str, list[int]]]],
    output_path: str | pathlib.Path,
    *,
    n_bins: int = 10,
) -> pathlib.Path | None:
    """D1: rank of the truth within the ensemble, prior | posterior, per sensor set.

    ``rank_counts`` is ``run_summary.yaml``'s ``sensor_statistics`` rank block:
    ``[set][half][statistic]`` holding the ``M+1`` counts
    :func:`evaluation.scores._score_window_statistic` writes. Rows are sensor
    sets, columns the prior and posterior halves (the prior column is dropped
    when no set has one). Counts are pooled over statistics, sensors and windows
    and coarsened by :func:`_coarsen_rank_counts`; the panels plot **counts**,
    not densities, with ``N`` labelled, over the per-group uniform reference and
    its binomial consistency band.

    A U shape means the ensemble is over-confident, a dome means it is
    over-dispersed, a slope means bias.

    Returns the path written, or ``None`` when no cell has usable counts (an old
    run dir, a run whose sensor block never ran, an ensemble of one).
    """
    cells: dict[tuple[str, str], np.ndarray] = {}
    for set_name, by_half in (rank_counts or {}).items():
        for half, statistics in (by_half or {}).items():
            pooled = _pool_rank_counts(statistics)
            if pooled is not None:
                cells[(set_name, half)] = pooled
    if not cells:
        logger.info("plot_rank_histogram: no usable rank counts")
        return None

    set_names = [s for s in rank_counts if any(key[0] == s for key in cells)]
    known = [h for h in ("prior", "posterior") if any(key[1] == h for key in cells)]
    halves = known + sorted({key[1] for key in cells} - set(known))

    with _styled():
        fig, axes = plt.subplots(
            len(set_names),
            len(halves),
            figsize=(4.2 * len(halves), 3.0 * len(set_names)),
            squeeze=False,
            constrained_layout=True,
        )
        for r, set_name in enumerate(set_names):
            for c, half in enumerate(halves):
                ax = axes[r][c]
                pooled = cells.get((set_name, half))
                if pooled is None:
                    ax.set_axis_off()
                    continue

                sizes, grouped, expected, band = _coarsen_rank_counts(pooled, n_bins)
                edges = np.concatenate([[0.0], np.cumsum(sizes)])
                ax.bar(
                    edges[:-1],
                    grouped,
                    width=sizes,
                    align="edge",
                    color=COLORS["posterior" if half == "posterior" else "prior"],
                    edgecolor="white",
                    linewidth=0.6,
                    zorder=3,
                )
                # Step + band in rank units, so the unequal group widths show.
                stepped = np.concatenate([expected, expected[-1:]])
                low = np.clip(
                    np.concatenate([expected - band, (expected - band)[-1:]]), 0.0, None
                )
                high = np.concatenate([expected + band, (expected + band)[-1:]])
                ax.fill_between(
                    edges, low, high, step="post", color=COLORS["window"], alpha=0.25
                )
                ax.step(
                    edges,
                    stepped,
                    where="post",
                    color=COLORS["charcoal"],
                    lw=1.2,
                    ls="--",
                    zorder=4,
                )
                ax.set_xlim(0.0, edges[-1])
                ax.annotate(
                    f"N = {int(grouped.sum())}",
                    xy=(0.02, 0.94),
                    xycoords="axes fraction",
                    va="top",
                    fontsize=8,
                    color=COLORS["charcoal"],
                )
                if r == 0:
                    ax.set_title(half.capitalize(), loc="left")
                if c == 0:
                    ax.set_ylabel(f"{set_name}\ncount")
                if r == len(set_names) - 1:
                    ax.set_xlabel(f"rank of truth (0-{int(pooled.size) - 1})")

        handles = [
            Line2D(
                [0], [0], color=COLORS["charcoal"], ls="--", lw=1.2, label="Uniform"
            ),
            Patch(facecolor=COLORS["window"], alpha=0.25, label=r"$\pm 2\sigma$ band"),
        ]
        axes[0][-1].legend(handles=handles, loc="upper right", fontsize=8)
        fig.suptitle("Rank histogram of the truth within the ensemble")
        return save_png(fig, output_path)
