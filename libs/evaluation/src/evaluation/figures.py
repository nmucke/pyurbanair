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

Populated in WP0.2 (move), extended in WP1.5 (P1, S1, S5, F1, D1), phase 3
(S4) and phase 2 (D3).
"""

# WP0.2 moved the general plots here out of ``src/pyurbanair/plotting.py``
# under a file-level ``# mypy: ignore-errors``. The waiver is gone: the moved
# helpers are annotated and the module passes the repo's strict config, so the
# WP1.5/phase-2/phase-3 figure set is now actually enforced rather than merely
# believed to be clean.

import contextlib
import logging
import pathlib
import textwrap
import warnings
from typing import Iterator, Sequence, cast

import matplotlib.pyplot as plt
import numpy as np
import xarray
from evaluation.scores import (
    DATA_MISMATCH_TARGET,
    _aligned_parameter_members,
    _param_members_and_x,
    _plotted_param_names,
    compute_parameter_metrics,
    compute_sensor_metrics,
    data_mismatch_target_band,
    parameter_bundle,
)

# ``_BAND_ALPHAS`` is imported rather than restated so S4's legend patch cannot
# drift from the alpha the envelope is actually drawn with.
from evaluation.style import (
    _BAND_ALPHAS,
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
from evaluation.turbulence import (
    evenly_spaced_levels,
    log_spectral_distance,
    median_spectrum,
)
from matplotlib.axes import Axes
from matplotlib.collections import PolyCollection
from matplotlib.colors import Colormap, LinearSegmentedColormap
from matplotlib.figure import Figure
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
#
# ``_add_velocity_magnitude`` has no caller left in this module -- its only two
# call sites were in the callerless snapshot plots deleted with the WP0.2 dead
# code. It is kept because ``tests/test_evaluation_library.py`` pins it by name
# as the drift check against ``pyurbanair.utils.run_utils``.
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


def _save(fig: Figure, output_path: str | pathlib.Path) -> None:
    output_path = pathlib.Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)


def _shade_windows(ax: Axes, edges: Sequence[float] | None) -> None:
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

    def _plot_ensemble(
        ax: Axes, ds: xarray.Dataset, param_name: str, color: str
    ) -> None:
        x, members = _param_members_and_x(ds[param_name])
        ax.plot(x, members.T, color=color, alpha=0.35, linewidth=0.9)
        ax.plot(x, members.mean(axis=0), color=color, alpha=1.0, linewidth=2.5)

    if rmse is None:
        # Fallback: whole-domain RMSE between the ensemble-mean state and the
        # truth. This materialises the full velocity fields; callers handling a
        # large truth should precompute a streamed ``rmse`` and pass it in.
        #
        # The two states are declared optional because the streamed-``rmse``
        # path -- which is what every caller in the repo uses -- leaves them
        # unset. This branch is exactly the case where they are required, so
        # they are cast rather than guarded: a ``raise`` here would change what
        # a mis-call does at run time, and this pass adds types only. Calling
        # with ``rmse=None`` and no states still fails, as it always has.
        truth_states = cast(xarray.Dataset, true_state)
        esmda_states = cast(xarray.Dataset, esmda_state)
        true_state_mean = (
            truth_states.mean(dim="ensemble")
            if "ensemble" in truth_states.dims
            else truth_states
        )
        esmda_state_mean = (
            esmda_states.mean(dim="ensemble")
            if "ensemble" in esmda_states.dims
            else esmda_states
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
            # ``prior_params is not None`` rather than the ``has_prior`` alias
            # it is assigned from: identical at run time, but only the direct
            # test narrows the Optional for the type checker.
            if prior_params is not None and param_name in prior_params.data_vars:
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

    **Opaque, unlike ``apply_style``'s own default.** That default is transparent
    because it was written for the paper/slide figures in
    ``scripts/figure_creation/``, which are composited onto a page whose colour
    the document owns. Every figure in this module is instead a PNG an operator
    opens straight out of a run directory, where a transparent background takes
    the viewer's -- and on a dark-mode viewer the dark axis labels, tick text and
    truth lines land on dark grey and the figure is unreadable. So the three
    facecolors are pinned white here, and the matching ``transparent=False`` goes
    to ``save_png`` at each save site (an explicit ``savefig`` keyword overrides
    ``savefig.transparent``, so setting the rcParam alone would not be enough).
    """
    with plt.rc_context():
        apply_style()
        plt.rcParams.update(
            {
                "figure.facecolor": "white",
                "axes.facecolor": "white",
                "savefig.facecolor": "white",
                "savefig.transparent": False,
            }
        )
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

    An estimated parameter that the truth file does not carry has no panel --
    the alignment yields only the parameters both datasets hold -- so it is
    logged: "one panel per estimated parameter" is otherwise silently untrue.
    """
    if true_params is not None:
        scored = set(_plotted_param_names(posterior_params, true_params))
        unscored = [
            name
            for name in _plotted_param_names(posterior_params)
            if name not in scored
        ]
        if unscored:
            logger.info(
                "plot_parameter_marginals: %d estimated parameter(s) absent from "
                "the truth and so not drawn: %s",
                len(unscored),
                ", ".join(unscored),
            )
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
    """Whether a parameter's truth is the same value at **two or more** knots.

    The discriminator between the two knot pairings P1 can draw. A *static*
    parameter estimated over ``W`` windows still arrives with ``K = W`` knots
    (``run_esmda.py`` stacks one point per window along ``time``), so the knot
    count alone cannot tell the two apart -- but a constant truth can.

    Two finite knots are the minimum evidence for that claim: ``np.allclose``
    over a single value is trivially true, so a truth known at one knot only
    (the rest NaN) would read as static and pair a knot-0 prior against a
    final-knot posterior -- two different quantities in one panel, the exact
    failure this branch exists to prevent. Below that threshold the conservative
    same-knot pair is drawn instead.
    """
    if truth is None:
        return False
    finite = _finite(truth)
    return finite.size >= 2 and bool(np.allclose(finite, finite[0]))


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
      two or more finite knots, so those knots are repeats of one quantity)
      draws the prior at **knot 0** -- the run's actual prior -- against the
      posterior at the final knot. That is the total contraction the run
      achieved.
    * A genuinely **time-varying** parameter keeps the **same (final) knot** for
      both. Knot 0 is a different physical time there, so a knot-0 prior would
      compare two different quantities; per-window contraction is all that is
      available, and the labels say so.

    Without a truth the two cannot be told apart, so the same-knot pair is drawn
    (the conservative reading) and labelled as such. A truth whose every knot is
    non-finite counts as no truth: it is not drawn, not scored, and does not
    pick the knot pair.

    Both branches label their marginals by **knot**, never by window: a
    ``static_parameters`` entry inside a dynamic run is broadcast onto the
    dynamic knot grid, so its knot count is not the run's window count.

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
            if truth is not None and not np.isfinite(truth).any():
                # A truth with no finite knot is no truth: it can neither be
                # drawn, nor scored, nor used to pick the knot pair. Reading it
                # as a time-varying one would label the panel with a property it
                # does not have; ``None`` reaches the honest "z: no truth".
                logger.info(
                    "plot_parameter_marginals: %s has no finite truth knot; "
                    "the panel is drawn as if no truth were saved",
                    name,
                )
                truth = None

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
                # Knots, not windows: a ``static_parameters`` entry inside a
                # *dynamic* run is broadcast onto the dynamic knot grid, so its
                # knot count is not the window count.
                prior_label = "Prior\n(knot 0)"
                post_label = "Posterior\n(final knot)"
                title = f"{name} (static, {n_knots} knots)"
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
        return save_png(fig, output_path, transparent=False)


# ---------------------------------------------------------------------------
# S1 -- vertical profiles at the station columns
# ---------------------------------------------------------------------------


# S1's TKE row and F1's mean are the two places a reader reads "time average"
# off a figure, and the frames behind them are not always a time average: the
# filter's default source hands the collectors ONE analyzed frame per cycle, so
# the second moments are an across-cycle variance carrying the analysis
# increments. The writer of ``eval_fields.nc`` knows which it was and records it
# there; these two figures take that line as ``sampling_note`` and render it,
# because the numbers themselves are indistinguishable either way.
#
# WHICH frames and WHETHER they were sparse are two separate inputs on purpose
# (``sampling_note`` and ``sampling_is_sparse``). Deriving the second from the
# first -- "there is a note, so the frames must have been sparse" -- reads the
# provenance of a run that saved every forecast frame as a caveat and stamps
# "sample-mean" on a genuine time average, which is both false and the exact
# signal that stops discriminating between the two sources once it fires on
# both. A note is provenance; only the flag is a caveat.
_SAMPLING_WRAP = 108


def _sampling_caption(note: str, marker: str = "") -> str:
    """The provenance line as it goes under a figure: prefixed and wrapped.

    Wrapped here rather than left to matplotlib, which does not wrap a
    ``supxlabel``: the notes come from the metric stage and run to a couple of
    hundred characters, and one that runs off both edges of the PNG is a
    qualification a reader cannot read.

    ``marker`` is the footnote symbol tying the line back to a row label that
    carries the same symbol (S1's ``k *``). It leads the line -- a footnote
    marker reads as one at the start of its footnote and as a typo anywhere
    else, and the line's own first word is what the marker refers to.
    """
    return textwrap.fill(f"{marker}Moments sampled from {note}", width=_SAMPLING_WRAP)


def _extrapolated_axes(fields: xarray.Dataset) -> tuple[str, ...]:
    """The centre axes whose last index ``eval_fields.nc`` flags as extrapolated."""
    raw = str(fields.attrs.get("extrapolated_edges", ""))
    return tuple(name for name in (part.strip() for part in raw.split(",")) if name)


def _held_out_first(
    sensor_sets: Sequence[str], max_items: int, *, figure: str, noun: str
) -> list[int]:
    """Indices to draw: the held-out points first, then the assimilated ones.

    A panel drawn only where the assimilation was fitted is the least informative
    one available, so a validation point never loses its slot to an assimilation
    point. Anything dropped is logged -- silent truncation would make a figure that
    omits half the evidence look complete.

    Shared by S1's station columns and S4's probe sensors: the rule is a figure
    convention, and two figures applying it two ways is how the held-out column
    quietly stops being first in one of them. ``figure`` and ``noun`` only shape
    the log line.
    """
    sets = [str(s) for s in sensor_sets]
    held_out = [i for i, s in enumerate(sets) if s != "assimilation"]
    assimilated = [i for i, s in enumerate(sets) if s == "assimilation"]
    order = held_out + assimilated
    if len(order) > max_items:
        dropped = order[max_items:]
        logger.info(
            "%s: %d of %d %s not drawn (max=%d): %s",
            figure,
            len(dropped),
            len(order),
            noun,
            max_items,
            ", ".join(f"{i} ({sets[i] or 'unlabelled'})" for i in dropped),
        )
    return order[:max_items]


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
    sampling_note: str | None = None,
    sampling_is_sparse: bool = False,
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

    ``sampling_note`` is the caller's one-line record of WHICH frames the
    moments were reduced over (``eval_fields.nc``'s ``moment_sampling``). It is
    a plain string -- this function learns nothing about run directories from it
    -- and it is printed as a provenance line under the panels whatever it says.

    ``sampling_is_sparse`` is the separate claim that those frames do NOT cover
    the horizon continuously (``eval_fields.nc``'s
    ``moment_sampling_is_sparse``), and it alone decides the wording: the TKE
    row is marked ``*``, tied to the note's own leading ``*``, and the caption
    stops calling the profiles a time average. That row is the reason: ``k``
    from one analyzed frame per cycle is an across-cycle variance, not resolved
    turbulence, and it renders identically to the real thing. A dense source
    that names its frames gets the provenance line without the caveat -- saying
    "not a continuous time average" over one that is would mislabel the better
    of two runs.

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
    # Only the vertical axis is trimmed, so only it is named in the caption:
    # ``extrapolated_edges`` lists every extrapolated axis (x and y among them)
    # and printing all of them claims exclusions that did not happen.
    vertical_extrapolated = tuple(
        name for name in _extrapolated_axes(fields) if name in _VERTICAL_CENTRE_DIMS
    )
    trimmed = bool(vertical_extrapolated) and z.size > 1
    keep = slice(0, -1) if trimmed else slice(None)
    z = z[keep]
    z_plot = z / building_height if building_height else z
    z_label = "z/H [-]" if building_height else "z [m]"

    scale = {
        "mean": 1.0 if u_ref is None else 1.0 / u_ref,
        "tke": 1.0 if u_ref is None else 1.0 / u_ref**2,
    }
    # Only the second-moment row is marked: the mean row survives sparse
    # sampling as a mean of what was sampled, while ``k`` changes meaning
    # outright, and marking both would blur which one the caption is about.
    # The mark needs a footnote to point at, so it needs both inputs -- without
    # a note the caption still says the profiles are not a time average, which
    # is the same warning without a dangling symbol.
    tke_mark = " *" if sampling_is_sparse and sampling_note else ""
    row_label = {
        "mean": r"$\bar{u}/U_{ref}$ [-]" if u_ref else r"$\bar{u}$ [m/s]",
        "tke": (r"$k/U_{ref}^2$ [-]" if u_ref else r"$k$ [m$^2$/s$^2$]") + tke_mark,
    }
    # Extracted once and handed to the ordering rule, which is shared with S4.
    sets = (
        [str(s) for s in np.asarray(fields["station_set"].values).ravel()]
        if "station_set" in fields.coords
        else ["" for _ in range(int(fields.sizes["station"]))]
    )
    stations = _held_out_first(
        sets, max_stations, figure="plot_station_profiles", noun="station columns"
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

        if sampling_is_sparse:
            caption = (
                "Profiles averaged over the sampled frames only -- NOT a "
                "continuous time average; ensembles as 5-95 % / 25-75 % bands."
            )
        else:
            caption = "Time-averaged profiles; ensembles as 5-95 % / 25-75 % bands."
        if trimmed:
            caption += (
                " Top z cell excluded: colocation extrapolates it "
                f"({', '.join(vertical_extrapolated)})."
            )
        if sampling_note:
            caption += "\n" + _sampling_caption(
                sampling_note, marker="* " if tke_mark else ""
            )
        fig.supxlabel(caption, fontsize=8, color=COLORS["charcoal"])
        fig.suptitle("Vertical profiles at the station columns")
        return save_png(fig, output_path, transparent=False)


# ---------------------------------------------------------------------------
# F1 -- time-averaged horizontal slices
# ---------------------------------------------------------------------------


def _masked_cmap(name: str) -> Colormap:
    """A copy of ``name`` whose masked (solid) cells get their own flat colour."""
    cmap = plt.get_cmap(name).copy()
    cmap.set_bad(_SOLID_COLOR)
    return cmap


# Why a column can carry no finite fluid cell, per column: an ensemble mean is
# blanked by a diverged member, the truth has no ensemble to diverge, and the
# difference only inherits its sources' gaps. One hint for all four would send a
# reader hunting a member that does not exist.
_EMPTY_SLAB_HINT = {
    "Truth": "the truth field is all NaN",
    "Prior mean": "check for a diverged member",
    "Posterior mean": "check for a diverged member",
    "Posterior - truth": "its truth or posterior source has none",
}


def _empty_slab_panel(
    ax: Axes, extent: tuple[float, float, float, float], hint: str
) -> None:
    """Draw an empty F1 panel that says *why* it is empty, on the field's extent.

    The alternative -- dropping the column -- leaves nothing on the PNG to
    distinguish "the posterior is missing" from "this figure only ever had three
    columns", while the suptitle keeps promising the missing subject.
    """
    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    ax.set_aspect("equal")
    ax.annotate(
        f"no finite cell\n({hint})",
        xy=(0.5, 0.5),
        xycoords="axes fraction",
        ha="center",
        va="center",
        fontsize=9,
        color=COLORS["charcoal"],
    )


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
    sampling_note: str | None = None,
    sampling_is_sparse: bool = False,
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
    solid-cell grey under a caption saying grey means solid -- grey must not mean
    two things on one figure. Such a column instead **keeps its slot in the grid
    and is drawn blank, labelled with why** (and logged): a column silently
    dropped is a figure that looks complete while missing its subject. It
    contributes nothing to the shared norm. Only a column the run never wrote
    (no prior state saved) is absent from the grid altogether.

    Unlike S1 this figure deliberately does **not** apply
    ``extrapolated_edges``: these are first moments, where the extrapolation
    artefact is small. The exclusion matters for the second moments S1 draws.

    These are the accumulated time means only -- **never an instantaneous
    field**, which decorrelates after a Lyapunov horizon and would measure chaos
    rather than parameter quality. The averaging window is annotated from the
    file's ``t_start`` / ``t_end``.

    ``sampling_note`` is the caller's one-line record of WHICH frames were
    accumulated (``eval_fields.nc``'s ``moment_sampling``), a plain string,
    printed as a provenance line under the panels whatever it says.

    ``sampling_is_sparse`` (``eval_fields.nc``'s ``moment_sampling_is_sparse``)
    is the separate claim that those frames do not cover the span, and it alone
    changes the wording: the title says "sample-mean" rather than "time-mean"
    and the span reads "sampled over t = a-b s", because ``t_start`` /
    ``t_end`` are the horizon the frames were drawn from, not proof that the
    horizon was covered continuously, and the filter's default source draws one
    frame per cycle from it. A run that saved every forecast frame covered it,
    so it keeps the time-mean wording and merely says where the frames came
    from.

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
                "plot_mean_slices: %s has no finite fluid cell and is drawn empty "
                "(%s)",
                label,
                _EMPTY_SLAB_HINT[label],
            )
            return False
        return True

    # A column present in the dataset but carrying nothing finite keeps its slot
    # and is drawn as a labelled blank (``values`` is ``None`` below). Dropping
    # it from the grid instead leaves the PNG reading "Truth | Prior mean" under
    # an unchanged suptitle, with nothing on the figure to say the subject of the
    # whole figure is missing. A column the run never wrote at all (no prior
    # state saved) is genuinely absent and stays out of the grid.
    columns: list[tuple[str, np.ndarray | None]] = []
    for label, values in (
        ("Truth", masked_truth),
        ("Prior mean", masked(prior)),
        ("Posterior mean", masked_posterior),
    ):
        if values is not None:
            columns.append((label, values if drawable(label, values) else None))
    difference = None
    has_difference = masked_truth is not None and masked_posterior is not None
    # Re-tested directly rather than through ``has_difference``: same condition,
    # but only this form narrows the two Optionals for the type checker.
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
    n_cols = len(columns) + (1 if has_difference else 0)

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
                if values is None:
                    _empty_slab_panel(axes[r][c], extent, _EMPTY_SLAB_HINT[label])
                else:
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
            if has_difference:
                ax = axes[r][len(columns)]
                if difference is None:
                    _empty_slab_panel(ax, extent, _EMPTY_SLAB_HINT["Posterior - truth"])
                else:
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
        # "time-mean" is a claim about how the frames were drawn, not only about
        # what was averaged, so it goes everywhere the mean is named or nowhere.
        mean_label = "sample-mean" if sampling_is_sparse else "time-mean"
        if field_image is not None:
            fig.colorbar(
                field_image,
                ax=[
                    axes[r][c] for r in range(len(levels)) for c in range(len(columns))
                ],
                fraction=0.03,
                pad=0.02,
                label=f"{mean_label} {unit}",
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
            # ``t = 0-40 s`` reads as "averaged continuously over 40 seconds".
            # It is only ever the horizon the frames came FROM, and under a
            # sparse source that horizon was visited a handful of times, so the
            # preposition changes with the sampling.
            edges = (
                f"{float(fields.attrs['t_start']):.0f}"
                f"-{float(fields.attrs['t_end']):.0f} s"
            )
            span = (
                f"  (sampled over t = {edges})"
                if sampling_is_sparse
                else f"  (t = {edges})"
            )
        title_prefix = "Sample-mean" if sampling_is_sparse else "Time-mean"
        fig.suptitle(f"{title_prefix} {_STREAMWISE} on horizontal slices{span}")

        if sampling_is_sparse:
            caption = (
                "Means over the sampled frames only -- never instantaneous, but "
                "NOT a continuous time average either."
            )
        else:
            caption = "Time averages, never instantaneous."
        if solid is not None:
            caption += " Solid cells masked (grey) and excluded from the norm."
        stride = int(fields.attrs.get("horizontal_stride", 1) or 1)
        if stride > 1:
            caption += f" Horizontal stride {stride}."
        if sampling_note:
            caption += "\n" + _sampling_caption(sampling_note)
        fig.supxlabel(caption, fontsize=8, color=COLORS["charcoal"])
        return save_png(fig, output_path, transparent=False)


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
    fan_label: str = "Posterior",
) -> pathlib.Path | None:
    """S5: quantile fans at the sensors, assimilated and held-out side by side.

    ``truth[set]`` is ``(time, sensor)`` and ``ensemble[set]`` is
    ``(ensemble, time, sensor)`` -- what
    :func:`evaluation.sensors.sensor_magnitude` produces. **Columns are the
    sensor sets**, labelled: the held-out column beside the assimilated one is
    the strongest anti-overfitting evidence in the suite, which it only is if a
    reader can see which is which. Rows are sensors, up to ``max_sensors``;
    anything dropped is logged. Window boundaries are marked, and the truth
    carries a ``+/- obs_error_std`` envelope when the caller knows that width --
    inside the y-limits, which span it: an envelope wider than the fan would
    otherwise clip to the panel edges and read as a background tint rather than
    as the observation error it is being compared against.

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

    ``fan_label`` names what the fan IS, in the legend and the title. It is not
    always the posterior: the filtering pipeline hands this function every frame
    of each cycle's *forecast* segment when the run saved them -- the prior side
    of each analysis -- and the same fan under a "Posterior" legend would be the
    same PNG meaning two different things depending on which artifacts the run
    happened to write. The caller knows which; this function cannot.

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
                        label=f"{fan_label} median",
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
                    # The envelope sets the limits too: at the shipped
                    # ``obs_error_std`` a narrow sensor series has a sigma_o wider
                    # than the fan, and an envelope clipped to limits taken from
                    # the fan alone fills the panel edge to edge and reads as a
                    # background tint instead of the width it is.
                    drawn.append(truth_r - obs_error_std)
                    drawn.append(truth_r + obs_error_std)

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

        fig.suptitle(f"Sensor time series: {fan_label.lower()} quantile fan vs truth")
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
        return save_png(fig, output_path, transparent=False)


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
        return save_png(fig, output_path, transparent=False)


# ---------------------------------------------------------------------------
# S4 -- premultiplied energy spectra at the probes
# ---------------------------------------------------------------------------

# Premultiplying turns the inertial range's -5/3 into -2/3, since
# ``f*E ~ f*f**(-5/3)``. It also makes equal areas equal energy on a log
# frequency axis, which is why the metrics doc asks for this form rather than for
# a bare ``E(f)``.
_INERTIAL_SLOPE = -2.0 / 3.0

# The guide segment spans this factor in frequency and sits this many times above
# the highest curve inside that span. Both exist to keep it a *reference* rather
# than a fit: a slope line drawn through the data reads as one, and no shipped
# probe record is long enough to claim a fitted inertial range.
_GUIDE_SPAN = 3.0
_GUIDE_OFFSET = 4.0


def _log_limits(*arrays: np.ndarray | None) -> tuple[float, float] | None:
    """Axis limits for a log scale: :func:`finite_limits` applied in decades.

    ``finite_limits`` pads *additively*, which on a decade-spanning spectrum would
    be either invisible at the top or negative at the bottom; padding in
    ``log10`` and exponentiating back is the same 5 % margin measured the way the
    axis measures it. Non-positive values become non-finite in the log and are
    dropped by the shared helper, which is the correct treatment on a log axis --
    they cannot be drawn either.
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        decades = [
            np.log10(np.asarray(a, dtype=float)) for a in arrays if a is not None
        ]
    limits = finite_limits(*decades, pad=0.05)
    return None if limits is None else (10.0 ** limits[0], 10.0 ** limits[1])


def _member_bands(members: np.ndarray, sensor: int) -> np.ndarray | None:
    """``(Q, K)`` nested-band quantiles of one sensor's member spectra.

    Quantiles are taken **per frequency bin, ignoring non-finite members**: a
    member whose probe series had a gap has an all-``nan`` spectrum (see
    :func:`evaluation.turbulence.welch_spectrum`) and would otherwise blank the
    whole envelope, which is the same failure mode S5's fan guards against.
    ``None`` when no member is finite anywhere at this sensor.
    """
    values = np.asarray(members, dtype=float)[:, sensor]
    if not np.isfinite(values).any():
        return None
    with warnings.catch_warnings():
        # An all-nan bin is a legitimate outcome and ``nan`` the right answer for
        # it; the per-bin RuntimeWarning is what has to go.
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.asarray(np.nanquantile(values, _BAND_QUANTILES, axis=0))


def _guide_segment(
    frequencies: np.ndarray, drawn: list[np.ndarray], upper: float
) -> tuple[np.ndarray, np.ndarray] | None:
    """A short ``-2/3`` reference segment, offset above the curves it sits beside.

    Anchored at the *geometric* middle of the compared band -- the panel's axis is
    logarithmic, so an index-based anchor on a linearly spaced frequency grid would
    put it in the top decade every time -- spanning :data:`_GUIDE_SPAN` in
    frequency at :data:`_GUIDE_OFFSET` times the highest value any curve reaches
    inside that span. So it is never drawn *through* the data (metrics doc §7,
    figure S4).
    ``None`` when the band is too narrow to carry it or nothing finite was drawn.
    """
    f = np.asarray(frequencies, dtype=float)
    if f.size < 3 or not (upper > f[0] > 0.0):
        return None
    start = float(np.sqrt(f[0] * upper))
    stop = min(upper, start * _GUIDE_SPAN)
    if not (stop > start > 0.0):
        return None
    inside = (f >= start) & (f <= stop)
    peak = finite_limits(*[np.asarray(d)[..., inside] for d in drawn])
    if peak is None or peak[1] <= 0.0:
        return None
    segment = np.array([start, stop])
    return segment, _GUIDE_OFFSET * peak[1] * (segment / start) ** _INERTIAL_SLOPE


def plot_spectra(
    f: np.ndarray,
    truth: np.ndarray,
    posterior: np.ndarray,
    output_path: str | pathlib.Path,
    *,
    prior: np.ndarray | None = None,
    truth_halves: np.ndarray | None = None,
    variance: np.ndarray | None = None,
    f_cutoff: float | None = None,
    sensor_sets: Sequence[str] | None = None,
    max_sensors: int = 6,
) -> pathlib.Path | None:
    """S4: premultiplied energy spectra at the probes, truth vs posterior vs prior.

    The figure the mean-field metrics cannot replace: an over-smoothed or
    collapsed flow can carry the right time-mean and even the right total
    variance while putting that variance at the wrong frequencies, and this is the
    only panel in the suite that shows *where* the energy sits. One panel per probe
    sensor (held-out first, see :func:`_held_out_first`), log--log, premultiplied
    ``f*E(f)/sigma^2``. The default ``max_sensors`` is the same 6 as S1's
    ``max_stations``: the shipped case defines 6 assimilation + 4 validation probe
    points, and a smaller budget would fill the figure with held-out panels alone
    and leave no in-sample reference beside them.

    Args:
        f: ``(K,)`` frequencies, as :func:`evaluation.turbulence.probe_spectra`
            returns them -- running past ``f_cutoff``, so the panel shows the whole
            resolved band and marks where scoring stops.
        truth: ``(sensor, K)`` truth spectra (the trace ``E_uu + E_vv + E_ww``).
        posterior: ``(M, sensor, K)`` member spectra.
        output_path: PNG to write.
        prior: ``(M, sensor, K)`` prior-member spectra. Prior probe reruns are
            optional, so ``None`` simply drops the grey envelope -- the figure is
            truth vs posterior then, not a failure.
        truth_halves: ``(2, sensor, K)`` spectra of the two halves of the truth
            record, annotated beside each panel's distance so the reader has
            *something* to read it against -- two finite records of the same flow
            are already some dB apart. It is **not** a pass threshold and it is not
            the like-for-like scatter either: it runs ~2x that (see
            :func:`evaluation.turbulence.probe_spectra`), which is why the panel
            labels it "truth halves" rather than "floor" and the caption says so.
        variance: ``(sensor,)`` normalisation, the truth's total resolved variance.
            **One shared scale for every curve in a panel** -- normalising each
            curve by its own variance would let a member carrying half the energy
            overlay the truth exactly. ``None`` plots the raw premultiplied
            spectrum in m^2/s^2 and says so on the axis. A sensor whose variance is
            not finite and positive is **dropped with a log line** rather than
            drawn unnormalized: the axis label and the shared scale are per figure,
            so one raw panel among normalized ones is mislabelled by construction.
        f_cutoff: Where comparisons stop (``f_Nyquist/4``); drawn as a dotted
            vertical line, and the band the annotated LSD is computed over.
        sensor_sets: ``(sensor,)`` set labels, used to order and title the panels.
        max_sensors: Panels drawn; anything dropped is logged.

    The posterior is nested quantile bands about its median (5--95 % light,
    25--75 % dark, teal); the prior is its 5--95 % **envelope** only (grey), since
    two full band stacks in one panel obscure the comparison the figure is for.
    The ``-2/3`` guide is a reference slope offset above the curves, never a fit.

    Each panel annotates the log-spectral distance of that sensor's
    posterior-median spectrum and the truth's own halves distance, both computed by
    :func:`evaluation.turbulence.log_spectral_distance` over ``f < f_cutoff`` --
    the same function and the same band as ``run_summary.yaml``'s
    ``spectral_metrics``, whose entries are these numbers reduced over sensors by
    the median.

    Returns the path written, or ``None`` when there is nothing to draw (no
    sensor, no finite spectrum, no sensor with a usable normalisation, a truth and
    an ensemble on different frequency grids).
    """
    frequencies = np.asarray(f, dtype=float).ravel()
    truth_e = np.asarray(truth, dtype=float)
    posterior_e = np.asarray(posterior, dtype=float)
    if truth_e.ndim != 2 or posterior_e.ndim != 3:
        logger.info(
            "plot_spectra: expected (sensor, freq) truth and (member, sensor, freq) "
            "posterior spectra, got %s and %s",
            truth_e.shape,
            posterior_e.shape,
        )
        return None
    if not frequencies.size or truth_e.shape != posterior_e.shape[1:]:
        logger.info(
            "plot_spectra: %d frequencies against truth %s and posterior %s -- the "
            "spectra are not on one grid",
            frequencies.size,
            truth_e.shape,
            posterior_e.shape,
        )
        return None
    if not (np.isfinite(truth_e).any() and np.isfinite(posterior_e).any()):
        logger.info("plot_spectra: no finite truth or posterior spectrum")
        return None

    n_sensors = truth_e.shape[0]
    sets = (
        [str(s) for s in sensor_sets]
        if sensor_sets is not None and len(sensor_sets) == n_sensors
        else ["" for _ in range(n_sensors)]
    )
    normalized = variance is not None and np.asarray(variance).size == n_sensors
    scale = (
        np.asarray(variance, dtype=float).ravel() if normalized else np.ones(n_sensors)
    )
    # A sensor whose normalisation is not usable is dropped here rather than drawn
    # raw: ``normalized`` sets the axis label and the panels share one scale, so a
    # single unnormalized panel would be mislabelled *and* off-scale. It happens
    # for real -- a probe inside a solid cell has zero variance, and one with a gap
    # in its series has none at all.
    usable = [
        i
        for i in range(n_sensors)
        if not normalized or (np.isfinite(scale[i]) and scale[i] > 0.0)
    ]
    if len(usable) < n_sensors:
        logger.info(
            "plot_spectra: %d of %d probe sensors have no usable variance to "
            "normalize by (zero or non-finite) and are not drawn: %s",
            n_sensors - len(usable),
            n_sensors,
            [i for i in range(n_sensors) if i not in usable],
        )
    # Dropped *before* the ordering, so an unusable sensor costs itself rather than
    # a panel slot that another sensor could have filled.
    order = [
        usable[i]
        for i in _held_out_first(
            [sets[i] for i in usable],
            max_sensors,
            figure="plot_spectra",
            noun="probe sensors",
        )
    ]
    if not order:
        logger.info("plot_spectra: no probe sensors to draw")
        return None

    band = frequencies < f_cutoff if f_cutoff else np.ones(frequencies.size, bool)
    median = median_spectrum(posterior_e)

    with _styled():
        fig, axes = plt.subplots(
            1,
            len(order),
            figsize=(3.5 * len(order), 3.7),
            squeeze=False,
            sharex=True,
            # One y scale for every panel: the curves are all divided by their own
            # sensor's truth variance, so they *are* comparable across sensors, and
            # per-panel autoscaling would hide that one probe carries an order of
            # magnitude less energy than another.
            sharey=True,
            constrained_layout=True,
        )
        panels: list[np.ndarray] = []
        for column, sensor in enumerate(order):
            ax = axes[0][column]
            # Premultiplied and on one scale per panel: the same divisor for the
            # truth, the posterior and the prior (see ``variance`` above). Every
            # sensor left in ``order`` has a finite positive scale.
            weight = frequencies / scale[sensor]
            drawn: list[np.ndarray] = []

            prior_bands = None if prior is None else _member_bands(prior, sensor)
            if prior_bands is not None:
                # The outer envelope only: the comparison is posterior vs truth,
                # and a second full band stack buries it.
                envelope = prior_bands[[0, -1]] * weight
                nested_bands(
                    ax,
                    frequencies,
                    envelope,
                    (_BAND_QUANTILES[0], _BAND_QUANTILES[-1]),
                    COLORS["prior"],
                )
                drawn.append(envelope)

            posterior_bands = _member_bands(posterior_e, sensor)
            if posterior_bands is not None:
                bands = posterior_bands * weight
                nested_bands(
                    ax,
                    frequencies,
                    bands,
                    _BAND_QUANTILES,
                    COLORS["posterior"],
                    label="Posterior median",
                )
                drawn.append(bands)

            truth_curve = truth_e[sensor] * weight
            ax.plot(
                frequencies,
                truth_curve,
                color=COLORS["truth"],
                lw=1.6,
                label="Truth",
                zorder=6,
            )
            drawn.append(truth_curve)

            guide = _guide_segment(
                frequencies,
                drawn,
                float(f_cutoff) if f_cutoff else float(frequencies[-1]),
            )
            if guide is not None:
                ax.plot(
                    *guide,
                    color=COLORS["charcoal"],
                    lw=1.2,
                    ls="--",
                    zorder=5,
                )
                ax.annotate(
                    r"$-2/3$",
                    xy=(guide[0][-1], guide[1][-1]),
                    xytext=(3, 2),
                    textcoords="offset points",
                    fontsize=8,
                    color=COLORS["charcoal"],
                )
                drawn.append(guide[1])

            if f_cutoff:
                ax.axvline(
                    float(f_cutoff),
                    color=COLORS["charcoal"],
                    ls=":",
                    lw=1.0,
                    zorder=2,
                )

            ax.set_xscale("log")
            ax.set_yscale("log")
            panels.extend(drawn)
            ax.set_xlabel("f [Hz]")
            if column == 0:
                ax.set_ylabel(
                    r"$f\,E(f)/\sigma^2$ [-]"
                    if normalized
                    else r"$f\,E(f)$ [m$^2$/s$^2$]"
                )
            ax.set_title(f"{sets[sensor] or 'probe'} #{sensor}", loc="left", fontsize=9)
            ax.annotate(
                _lsd_label(truth_e, median, truth_halves, sensor, band),
                xy=(0.03, 0.05),
                xycoords="axes fraction",
                fontsize=8,
                color=COLORS["charcoal"],
                # A premultiplied spectrum dips towards the bottom-left of the
                # panel exactly where this sits, so on a real run the curve runs
                # straight through the text. The number is the panel's headline
                # and has to stay readable, hence the backing box rather than a
                # different corner (every corner is occupied in some panel).
                bbox={
                    "facecolor": "white",
                    "edgecolor": "none",
                    "alpha": 0.75,
                    "pad": 1.5,
                },
                zorder=5,
            )

        handles, labels = axes[0][0].get_legend_handles_labels()
        if prior is not None:
            handles.append(
                Patch(
                    facecolor=COLORS["prior"],
                    # ``nested_bands`` draws a lone band at its inner alpha; the
                    # legend swatch has to be the same shade as what it labels.
                    alpha=_BAND_ALPHAS[1],
                    label="Prior 5-95 %",
                )
            )
        if f_cutoff:
            handles.append(
                Line2D(
                    [0],
                    [0],
                    color=COLORS["charcoal"],
                    ls=":",
                    lw=1.0,
                    label=r"$f_{Nyq}/4$ cutoff",
                )
            )
        # A decade of headroom above the shared limits, and the legend inside it.
        # A premultiplied spectrum runs from the top left to the bottom right and
        # the guide segment and the per-panel LSD take the corners it leaves, so
        # every in-panel corner is occupied by *something*; making room is the only
        # placement that cannot land on a curve. (An ``outside`` figure legend is
        # the other option and collides with the suptitle or the caption instead.)
        limits = _log_limits(*panels)
        if limits is not None:
            axes[0][0].set_ylim(limits[0], limits[1] * (10.0 if handles else 1.0))
        if handles:
            axes[0][-1].legend(handles=handles, loc="upper right", fontsize=8)
        fig.suptitle("Premultiplied energy spectra at the probes")
        fig.supxlabel(
            "Spectra are the trace E_uu+E_vv+E_ww: Welch (Hann, 50 % overlap, "
            "linear detrend), one segment length for the truth and every member.\n"
            "Normalized by the TRUTH's resolved variance, so an energy deficit "
            "stays visible. Comparisons stop at the dotted cutoff; the -2/3 line "
            "is a reference slope, not a fit.\n"
            "'truth halves' is the LSD between the two halves of the truth's own "
            "record. It runs ~2x the like-for-like scatter of this comparison, so "
            "it is a reference, NOT a pass threshold: halve it before reading an "
            "LSD as indistinguishable from the truth.",
            fontsize=8,
            color=COLORS["charcoal"],
        )
        return save_png(fig, output_path, transparent=False)


def _lsd_label(
    truth_e: np.ndarray,
    median: np.ndarray,
    truth_halves: np.ndarray | None,
    sensor: int,
    band: np.ndarray,
) -> str:
    """``LSD = x dB (truth halves y)`` for one panel, or that it is unmeasurable.

    The halves distance is the whole reason the number is annotated at all: a bare
    "3 dB" invites a reader to compare it against zero, which is not the reachable
    value at any finite record length. It is deliberately **not** called a floor
    here -- it runs ~2x the like-for-like scatter (derivation in
    :func:`evaluation.turbulence.probe_spectra`), so a panel under it is not thereby
    a passing panel, and the caption repeats the factor.
    """
    distance = float(log_spectral_distance(truth_e[sensor, band], median[sensor, band]))
    if not np.isfinite(distance):
        return "LSD: no comparable bin"
    label = f"LSD = {distance:.2f} dB"
    if truth_halves is not None and np.asarray(truth_halves).shape[0] == 2:
        halves = np.asarray(truth_halves, dtype=float)
        reference = float(
            log_spectral_distance(halves[0, sensor, band], halves[1, sensor, band])
        )
        if np.isfinite(reference):
            label += f" (truth halves {reference:.2f})"
    return label


# ---------------------------------------------------------------------------
# D3 -- data-mismatch decay
# ---------------------------------------------------------------------------

# Switch to a log y-axis once the per-step MEDIANS span at least this many
# decades: a healthy MDA run drops O_N by one to two orders of magnitude from the
# prior, and on a linear axis every posterior iteration is then flattened onto
# zero -- the part of the figure the target band exists to resolve. Judged on the
# medians, not on every member: one outlier member two decades below its step's
# median would otherwise flip the whole figure to log.
_D3_LOG_DECADES = 1.5

# Fraction of an iteration slot the per-window box cluster may occupy, box widths
# included. Below 1.0 with a margin, so consecutive iterations' clusters never
# touch and every box stays on its own tick.
_D3_CLUSTER_SPAN = 0.72


def _d3_box_layout(n_steps: int, n_windows: int) -> tuple[np.ndarray, float]:
    """``(positions, width)`` for ``n_windows`` boxes per iteration.

    ``positions`` is ``(n_windows, n_steps)``: iteration ``i`` sits at integer
    ``i`` and a rollout's windows are offset around it. The offsets are placed so
    the cluster's full extent -- **outer box edges included** -- stays within
    ``_D3_CLUSTER_SPAN`` of one slot, which is what keeps window ``w``'s box at
    iteration ``i+1`` clear of window ``w'``'s at iteration ``i``. A single
    window lands exactly on the integers.
    """
    steps = np.arange(n_steps, dtype=float)[None, :]
    if n_windows <= 1:
        return steps, _D3_CLUSTER_SPAN * 0.6

    # n_windows boxes of width ``w`` spanning centres [-h, +h] occupy
    # 2h + w; solving 2h + w = span with w = span/(n_windows + 1) leaves a
    # box-width gap between adjacent clusters.
    width = _D3_CLUSTER_SPAN / (n_windows + 1)
    half = (_D3_CLUSTER_SPAN - width) / 2.0
    offsets = np.linspace(-half, half, n_windows)
    return steps + offsets[:, None], width


def plot_data_mismatch_decay(
    per_window: Sequence[np.ndarray] | None,
    output_path: str | pathlib.Path,
    *,
    num_observations: int = 0,
    window_indices: Sequence[int] | None = None,
) -> pathlib.Path | None:
    """D3: per-member ``O_N`` vs ESMDA iteration against the ½ target band.

    ``per_window`` is a list of ``(n_steps, M)`` arrays of
    :func:`evaluation.scores.data_mismatch` values, one per assimilation window;
    ``num_observations`` is ``N_d``, which sets the band; ``window_indices``
    says which window each entry is, so a run whose window 1 could not be read
    labels its boxes 0 and 2 rather than silently renumbering them 0 and 1.
    Windows are drawn as separate boxes per iteration rather than pooled —
    window 0's prior is a cold-start draw and a later window's is an
    extrapolated posterior, so pooling their step-0 boxes would conflate two
    different objects.

    A healthy run's boxes descend from the prior and settle inside the band. The
    two failure modes the figure exists to separate: boxes settling **above** the
    band are under-fitting, boxes settling **below** it mean the ensemble is
    fitting observation noise. The band's χ² target assumes ``C_D`` includes
    representativeness error, which it does not here — annotated on the figure,
    so the trend and the box heights are read before the absolute placement.

    Returns the path written, or ``None`` when there is nothing to draw (an old
    run dir, or ``esmda.save_obs_diagnostics=false``).
    """
    supplied = list(per_window if per_window is not None else [])
    windows = [np.atleast_2d(np.asarray(w, dtype=float)) for w in supplied]
    keep = [i for i, w in enumerate(windows) if w.size and np.any(np.isfinite(w))]
    windows = [windows[i] for i in keep]
    if not windows:
        logger.info("plot_data_mismatch_decay: no usable data-mismatch values")
        return None

    labels = [window_indices[i] for i in keep] if window_indices is not None else keep
    n_steps = max(w.shape[0] for w in windows)
    positions, box_width = _d3_box_layout(n_steps, len(windows))
    band = data_mismatch_target_band(num_observations)

    # Colour the prior end grey and the posterior end teal, interpolating in
    # between, so the direction of travel is legible without reading the axis.
    ramp = _step_colors(n_steps)

    # Decided before anything is drawn: the band's placement depends on it.
    # Judged on the per-step medians (see ``_D3_LOG_DECADES``) but only allowed
    # when every plotted value is strictly positive, since a single O_N of
    # exactly 0 -- a member reproducing the observations exactly -- has no place
    # on a log axis.
    pooled = np.concatenate([_finite(w) for w in windows])
    step_values = [
        _finite(np.concatenate([w[s] for w in windows if w.shape[0] > s]))
        for s in range(n_steps)
    ]
    # ``np.median`` of an empty step warns on stdout before returning nan. The
    # nan is the right answer and ``_spans_decades`` drops it, so the guard here
    # suppresses only the noise.
    medians = np.array([np.median(v) if v.size else np.nan for v in step_values])
    use_log = _spans_decades(medians, _D3_LOG_DECADES) and bool(np.all(pooled > 0.0))

    with _styled():
        fig, ax = plt.subplots(figsize=(1.6 + 1.5 * n_steps, 4.2))

        for w, values in enumerate(windows):
            for step in range(values.shape[0]):
                finite = _finite(values[step])
                if finite.size == 0:
                    continue
                box = ax.boxplot(
                    finite,
                    positions=[positions[w][step]],
                    widths=box_width,
                    showfliers=True,
                    patch_artist=True,
                    medianprops={"color": COLORS["charcoal"], "lw": 1.4},
                    flierprops={
                        "marker": ".",
                        "markersize": 3,
                        "markerfacecolor": COLORS["charcoal"],
                        "markeredgecolor": "none",
                    },
                    whiskerprops={"color": COLORS["charcoal"], "lw": 1.0},
                    capprops={"color": COLORS["charcoal"], "lw": 1.0},
                    zorder=3,
                )
                box["boxes"][0].set_facecolor(ramp[step])
                box["boxes"][0].set_edgecolor(COLORS["charcoal"])
                box["boxes"][0].set_alpha(0.85)

        if use_log:
            ax.set_yscale("log")

        band_drawn = band is not None
        if band is not None:
            lower, upper = DATA_MISMATCH_TARGET - band, DATA_MISMATCH_TARGET + band
            if lower <= 0.0:
                # ``band >= 1/2``, i.e. 18 or fewer observations (3/sqrt(2*18)
                # is exactly 1/2). The true lower edge is at or below zero,
                # which a log axis cannot take and which on a linear one just
                # means "no lower bound".
                if not use_log:
                    lower = 0.0
                elif float(pooled.min()) < upper:
                    lower = float(pooled.min())
                else:
                    # A band with no drawable lower edge, and nothing plotted
                    # inside it: a decoration rather than a reference.
                    band_drawn = False
            # Whenever ``lower > 0`` -- every N_d >= 19, which is every real
            # run -- the true band is drawn as-is and the axis grows to include
            # it. That is deliberate: on an off-target run the reader needs to
            # see HOW FAR above the band the boxes sit, and clipping the band to
            # the data was what made the span invert.

            if band_drawn:
                ax.axhspan(lower, upper, color=COLORS["window"], alpha=0.25, zorder=1)
        ax.axhline(
            DATA_MISMATCH_TARGET, color=COLORS["charcoal"], ls="--", lw=1.2, zorder=2
        )

        ax.set_xticks(np.arange(n_steps))
        ax.set_xticklabels(_d3_step_labels(n_steps))
        ax.set_xlim(-0.6, n_steps - 0.4)
        ax.set_xlabel("ESMDA iteration")
        ax.set_ylabel(r"$O_N$")
        ax.set_title("Normalized data mismatch per iteration", loc="left")

        handles = [
            Line2D(
                [0],
                [0],
                color=COLORS["charcoal"],
                ls="--",
                lw=1.2,
                label=r"target $1/2$",
            )
        ]
        if band_drawn:
            handles.append(
                Patch(
                    facecolor=COLORS["window"],
                    alpha=0.25,
                    label=r"$1/2 \pm 3/\sqrt{2N_d}$",
                )
            )
        if len(windows) > 1:
            # The actual window numbers, so a run that lost one to a read error
            # does not present the survivors as a contiguous 0..N-1.
            handles.append(
                Line2D(
                    [0],
                    [0],
                    color="none",
                    label="windows "
                    + ", ".join(str(w) for w in labels)
                    + " (left to right)",
                )
            )
        ax.legend(handles=handles, loc="best", fontsize=8)
        # Only when a band is on the figure -- otherwise the note explains a
        # reference the reader cannot see.
        if band_drawn:
            ax.annotate(
                "band assumes $C_D$ includes representativeness error (it does "
                "not);\nread the trend and the box heights before the absolute "
                "placement",
                xy=(0.0, -0.22),
                xycoords="axes fraction",
                fontsize=7,
                color=COLORS["charcoal"],
                va="top",
            )
        return save_png(fig, output_path, transparent=False)


def _d3_step_labels(n_steps: int) -> list[str]:
    """Iteration tick labels: ``prior``, the interior indices, ``posterior``."""
    labels = [str(i) for i in range(n_steps)]
    labels[0] = "0\nprior"
    if n_steps > 1:
        labels[-1] = f"{n_steps - 1}\nposterior"
    return labels


def _step_colors(n_steps: int) -> list[tuple[float, float, float, float]]:
    """Prior-grey to posterior-teal ramp, one colour per iteration."""
    ramp = LinearSegmentedColormap.from_list(
        "d3", [COLORS["prior"], COLORS["posterior"]]
    )
    if n_steps <= 1:
        return [ramp(1.0)]
    return [ramp(i / (n_steps - 1)) for i in range(n_steps)]


def _spans_decades(values: np.ndarray, decades: float) -> bool:
    """True when the finite ``values`` are positive and span ``decades`` of them.

    Non-positive values have no ratio to take, so their presence answers the
    question ``False`` rather than raising.
    """
    finite = _finite(values)
    if finite.size == 0 or np.any(finite <= 0.0):
        return False
    return bool(np.log10(finite.max() / finite.min()) >= decades)
