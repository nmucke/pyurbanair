"""Run several solver backends over one shared specification and compare them.

Every selected backend is driven with the *same* case and parameter trajectory
within each named scenario. Each scenario is then replayed for every model, so
the four (or more) solver/scenario combinations isolate solver response from
the prescribed forcing.

Written into the run directory:

  * ``scenario_parameters.png``     -- prescribed trajectories for every
                                      scenario, before any solver is run.
  * ``<scenario>/parameters.png``   -- prescribed inflow vs the inflow each model
                                      actually realises at the inlet.
  * ``state_snapshots_z<h>.png``    -- |U| at height ``h``, models x snapshot times.
  * ``state_difference_z<h>.png``   -- the same, as (model - reference).
  * ``field_rmse.png``              -- |U| RMSE against the reference over time.
  * ``state_animation.mp4``         -- |U| animation, models x heights.
  * ``sensor_timeseries_<set>.png`` -- u/v/w/|U| at the assimilation and the
                                      held-out validation sensors, one line per model.
  * ``sensor_rolling_<set>.png``    -- the same sensors smoothed by a sliding
                                      window of the observation operator's
                                      ``obs.interval_seconds`` length.
  * ``sensor_metrics_<set>.png`` / ``sensor_metrics.csv`` -- rolling-window
                                      bias, MAE, RMSE, correlation, and spread
                                      ratio against the reference.
  * ``field_{mean,std}_z<h>.png``   -- windowed mean and temporal-spread maps,
                                      excluding geometry and its first wall cell.
  * ``vertical_profiles_<region>.png`` -- mean speed/direction and component
                                      variability in upstream, canopy, and wake
                                      regions inferred from the STL.
  * ``field_{pdf,cdf}_<region>.png`` -- fluid-cell velocity distributions.
  * ``sensor_spectra_<set>.png``    -- sensor-averaged power spectra and
                                      autocorrelations.
  * ``measurement_overview.png``    -- time-mean state slices annotated with
                                      every sensor, inlet probe, and regional
                                      diagnostic footprint.
  * ``wake_metrics.csv`` / ``wake_profiles.png`` -- velocity deficit,
                                      recirculation, vertical exchange, and
                                      wake-recovery diagnostics.
  * ``<scenario>/summary.csv``      -- runtime + error scalars per model.
  * ``scenario_summary.csv``        -- the same scalars, with scenario labels.
  * ``scenario_error_summary.png``  -- solver discrepancy metrics across all
                                      scenarios in one figure.
  * ``within_model/<model>/``       -- the same core state/sensor diagnostics,
                                      comparing parameter scenarios within one
                                      solver.

The backends live on different (partly staggered) grids, so every field figure is
drawn after interpolating each model onto one common cell-centred grid
(``compare.grid``) and onto one common time axis. Sensor series are interpolated
at the physical sensor points on each model's own grid, so they are
grid-independent by construction.

Usage::

    python scripts/compare_models.py
    python scripts/compare_models.py 'compare.models=[pypalm,pyudales,pylbm]'
    python scripts/compare_models.py run.rollout_steps=3
    python scripts/compare_models.py 'compare.parameter_scenarios=[sine,cosine]'
"""

import copy
import csv
import pathlib
import sys
import textwrap
import time
import warnings
from dataclasses import dataclass

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import hydra
import matplotlib.pyplot as plt
import numpy as np
import xarray
from hydra.utils import instantiate
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
from omegaconf import DictConfig

import pyurbanair.quiet_jax  # noqa: F401  (suppress JAX CPU-fallback noise)
from pyurbanair.animation import _get_writer_and_output_path
from pyurbanair.config.hydra_helpers import clean_outputs, resolve_output_dir
from pyurbanair.utils.run_utils import add_velocity_magnitude
from scripts._common import resolve_results_dir

# _sensor_component_timeseries is private but is the canonical grid-independent
# sensor extraction (it resolves each backend's staggered dim mapping); the
# alternative is duplicating it here.
from scripts.esmda._esmda_common import (
    _sensor_component_timeseries,
    build_sensor_sets,
    sensor_magnitude,
)

# Candidate spatial dim names across the backends: pylbm/the surrogate are
# cell-centred (x/y/z), uDALES and PALM stagger each component on its own axis.
_X_DIMS = ("x", "xt", "xm", "xu")
_Y_DIMS = ("y", "yt", "ym", "yv")
_Z_DIMS = ("z", "zt", "zm", "zu")


@dataclass(frozen=True)
class GeometryInfo:
    """STL-derived roof heights and the horizontal extent of the buildings."""

    roof_heights: np.ndarray  # (y, x), metres above the domain floor
    x_min: float
    x_max: float
    y_min: float
    y_max: float


@dataclass(frozen=True)
class Region:
    """A named horizontal diagnostic region."""

    name: str
    x_min: float
    x_max: float
    y_min: float
    y_max: float


# ---------------------------------------------------------------------------
# Shared-specification execution
# ---------------------------------------------------------------------------


def _pick_dim(da: xarray.DataArray, candidates: tuple[str, ...]) -> str:
    return next(d for d in candidates if d in da.dims)


def sample_shared_params(
    cfg: DictConfig, params_cfg: DictConfig
) -> list[xarray.Dataset]:
    """Sample the parameters once; every model is driven with these.

    Returns one Dataset per rollout window (a single entry for a static prior,
    which is replayed unchanged for every window). Sampling here rather than
    per-model is what makes the comparison a controlled experiment.
    """
    import jax

    params_sampler = instantiate(params_cfg)
    params = params_sampler.sample(
        cfg.ensemble.ensemble_size if cfg.run.ensemble else 1
    )
    params_list = [params]
    if "time" not in params.coords:
        return params_list

    rng_key = jax.random.PRNGKey(int(params_cfg.get("seed", 0)))
    next_window_times = np.asarray(params_sampler.time_coords)
    for _ in range(cfg.run.rollout_steps):
        rng_key, subkey = jax.random.split(rng_key)
        params_list.append(
            params_sampler.extrapolate(params_list[-1], next_window_times, subkey)
        )
    return params_list


# The sampler always emits an `ensemble` dim. A single-member run must hand the
# forward model params WITHOUT it -- the solvers' inflow application can't handle
# a size-1 ensemble axis. Mirrors run_forward_model.py.
def _member_params(p: xarray.Dataset, is_ensemble: bool) -> xarray.Dataset:
    if not is_ensemble and "ensemble" in p.dims:
        return p.isel(ensemble=0, drop=True)
    return p


def _concat_windows(
    window_list: list[xarray.Dataset], cfg: DictConfig
) -> xarray.Dataset:
    """Stitch rollout windows onto one monotonic global time axis.

    Solvers report a per-window local clock (each window restarts near 0), so
    window ``w`` is re-based to start at ``w * simulation_time``.
    """
    if len(window_list) == 1:
        return window_list[0]
    sim = float(cfg.time.simulation_time)
    rebased = []
    for w, ds in enumerate(window_list):
        t = np.asarray(ds["time"].values, dtype=float)
        rebased.append(ds.assign_coords(time=(t - t[0]) + w * sim))
    return xarray.concat(rebased, dim="time", join="override")


def run_one_model(
    cfg: DictConfig,
    key: str,
    params_list: list[xarray.Dataset],
    scenario_name: str,
) -> tuple[xarray.Dataset, float]:
    """Run backend ``cfg.models[key]`` over ``params_list``; return (state, seconds)."""
    is_ensemble = cfg.run.ensemble

    # Per-model view of the config: the backends all interpolate the *global*
    # ${paths.experiment_dir} for their scratch dir, so give each one its own
    # before instantiating. Same for an on-disk results dir.
    run_cfg = copy.deepcopy(cfg)
    model_name = str(cfg.models[key].name)
    run_cfg.paths.experiment_dir = (
        f"{cfg.paths.experiment_root}_{model_name}_{scenario_name}"
    )
    if run_cfg.run.results_dir is not None:
        run_cfg.run.results_dir = str(
            pathlib.Path(str(cfg.run.results_dir)) / scenario_name / model_name
        )
    model_cfg = run_cfg.models[key]

    forward_model = instantiate(
        model_cfg.forward_model,
        results_dir=resolve_results_dir(run_cfg),
    )
    instantiate(model_cfg.prepare, forward_model=forward_model)
    clean_outputs(model_name=model_name, forward_model=forward_model)
    if is_ensemble:
        forward_model = instantiate(
            model_cfg.ensemble_model, forward_model=forward_model
        )

    def step(
        params: xarray.Dataset, state: xarray.Dataset | None = None
    ) -> xarray.Dataset:
        if is_ensemble:
            out = forward_model.run_ensemble(
                params=params, state=state, sim_name="state"
            )
        else:
            out = forward_model(params=params, state=state)
        return out if out is not None else forward_model.get_states()

    t0 = time.time()
    windows = [step(params=_member_params(params_list[0], is_ensemble))]
    for w in range(1, int(cfg.run.rollout_steps) + 1):
        params_w = params_list[min(w, len(params_list) - 1)]
        windows.append(
            step(
                params=_member_params(params_w, is_ensemble),
                state=windows[-1],
            )
        )
    elapsed = time.time() - t0

    state = add_velocity_magnitude(_concat_windows(windows, cfg))
    return state, elapsed


# ---------------------------------------------------------------------------
# Common grid / common time axis
# ---------------------------------------------------------------------------


def build_common_grid(cfg: DictConfig) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Cell-centred (x, y) plotting grid over the domain plus the slice heights.

    Cell *centres* (not the bounds themselves): every solver's outermost sample
    also sits half a cell inside the domain, so a centred -- and deliberately
    coarser -- plotting grid keeps the interpolation an interpolation.
    """
    (x_lo, x_hi), (y_lo, y_hi), (z_lo, z_hi) = [
        tuple(float(v) for v in pair) for pair in cfg.domain.bounds
    ]

    def centres(lo: float, hi: float, n: int) -> np.ndarray:
        edges = np.linspace(lo, hi, n + 1)
        return 0.5 * (edges[:-1] + edges[1:])

    x = centres(x_lo, x_hi, int(cfg.compare.grid.nx))
    y = centres(y_lo, y_hi, int(cfg.compare.grid.ny))
    # Heights are absolute (m); clip to the domain so a mis-set height can't ask
    # for a plane outside it.
    heights = np.clip(np.asarray(cfg.compare.heights, dtype=float), z_lo, z_hi)
    return x, y, heights


def _clamped(da: xarray.DataArray, dim: str, target: np.ndarray) -> np.ndarray:
    """``target`` clipped into ``da[dim]``'s range.

    xarray's multi-dimensional ``interp`` silently returns NaN outside the source
    coordinates -- ``fill_value=None`` does *not* extrapolate on this path. The
    grid above keeps requests inside the domain, but the solvers' first/last
    sample can still sit a fraction of a cell inside a requested point (and the
    lowest requested height can fall below the first level of a coarse grid), so
    clamp rather than punch NaN holes into the comparison.
    """
    coord = np.asarray(da.coords[dim].values, dtype=float)
    return np.clip(target, coord.min(), coord.max())


def regrid_state(
    state: xarray.Dataset,
    x: np.ndarray,
    y: np.ndarray,
    heights: np.ndarray,
    times: np.ndarray,
) -> xarray.Dataset:
    """Interpolate u/v/w onto the common (time, height, y, x) grid and add |U|.

    Each component is interpolated on *its own* staggered axes and by physical
    coordinate value, so the staggering is undone in the same pass that puts the
    models on a shared grid. The state must already be single-member.
    """
    components = {}
    for var in ("u", "v", "w"):
        da = state[var]
        z_dim, y_dim, x_dim = (
            _pick_dim(da, _Z_DIMS),
            _pick_dim(da, _Y_DIMS),
            _pick_dim(da, _X_DIMS),
        )
        sel = {
            z_dim: _clamped(da, z_dim, heights),
            y_dim: _clamped(da, y_dim, y),
            x_dim: _clamped(da, x_dim, x),
        }
        interpolated = da.interp(sel, kwargs={"bounds_error": False})
        # Re-label onto the shared coordinate values so the components (each
        # clamped against its own staggered axis) align exactly.
        components[var] = interpolated.rename(
            dict(zip((z_dim, y_dim, x_dim), ("height", "y", "x")))
        ).assign_coords(height=heights, y=y, x=x)

    regridded = xarray.Dataset(components).interp(
        time=times, kwargs={"bounds_error": False}
    )
    magnitude = np.sqrt(regridded["u"] ** 2 + regridded["v"] ** 2 + regridded["w"] ** 2)
    return regridded.assign(vel_magnitude=magnitude)


def common_time_axis(states: dict[str, xarray.Dataset]) -> np.ndarray:
    """Time axis shared by every model: the finest cadence over the common span."""
    axes = {
        name: np.asarray(s["time"].values, dtype=float) for name, s in states.items()
    }
    t_lo = max(float(t.min()) for t in axes.values())
    t_hi = min(float(t.max()) for t in axes.values())
    n = max(len(t) for t in axes.values())
    return np.linspace(t_lo, t_hi, n)


def _ensemble_mean(state: xarray.Dataset) -> xarray.Dataset:
    return state.mean(dim="ensemble") if "ensemble" in state.dims else state


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------


def _model_colors(names: list[str]) -> dict[str, str]:
    return {name: f"C{i}" for i, name in enumerate(names)}


def plot_parameters(
    params: xarray.Dataset,
    fields: dict[str, xarray.Dataset],
    output_path: pathlib.Path,
) -> None:
    """Prescribed parameters vs the inflow each model actually realises.

    The prescribed values are identical for every model by construction (one
    shared draw), so the interesting signal is the *derived* inflow: the angle
    and speed recovered from (u, v) at inlet probes. A model that responds
    sluggishly to the prescribed inflow shows up here as a lagged/damped curve.
    """
    param_names = list(params.data_vars)
    n_rows = len(param_names)
    colors = _model_colors(list(fields))

    # Inlet probes: three y-positions at the highest requested slice height,
    # well above the buildings, one grid column in from the inflow face.
    derived: dict[str, dict[str, np.ndarray]] = {}
    for name, field in fields.items():
        probe = field.isel(height=-1, x=0)
        y_vals = np.asarray(probe["y"].values, dtype=float)
        y_idx = [int(round(f * (len(y_vals) - 1))) for f in (0.2, 0.5, 0.8)]
        u = np.asarray(probe["u"].isel(y=y_idx).values)  # (time, probe)
        v = np.asarray(probe["v"].isel(y=y_idx).values)
        derived[name] = {
            "time": np.asarray(field["time"].values, dtype=float),
            "inflow_angle": np.degrees(np.arctan2(v, u)).mean(axis=-1),
            "velocity_magnitude": np.hypot(u, v).mean(axis=-1),
        }

    # Shared x-range so the constant-in-time parameters (drawn as a single
    # horizontal line) span the same axis as the time-varying ones.
    t_span = (
        min(float(s["time"].min()) for s in derived.values()),
        max(float(s["time"].max()) for s in derived.values()),
    )

    fig, axes = plt.subplots(
        n_rows, 1, figsize=(9, 3.0 * max(n_rows, 1)), squeeze=False
    )
    for ax, param in zip(axes[:, 0], param_names):
        da = params[param]
        if "time" in da.dims:
            ax.plot(
                np.asarray(da["time"].values, dtype=float),
                np.asarray(da.values),
                "k--",
                lw=2.0,
                label="prescribed",
            )
        else:
            ax.axhline(
                float(np.atleast_1d(da.values)[0]),
                color="k",
                ls="--",
                lw=2.0,
                label="prescribed",
            )
        for name, series in derived.items():
            if param in series:
                ax.plot(
                    series["time"],
                    series[param],
                    color=colors[name],
                    lw=1.6,
                    label=f"{name} (derived)",
                )
        ax.set_ylabel(param)
        ax.set_xlabel("time [s]")
        ax.set_xlim(*t_span)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=8)

    fig.suptitle("Prescribed parameters and the inflow realised at the inlet")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_scenario_parameters(
    params_by_scenario: dict[str, xarray.Dataset], output_path: pathlib.Path
) -> None:
    """Plot the prescribed dynamic trajectories that define each scenario."""
    param_names = list(
        dict.fromkeys(
            name
            for params in params_by_scenario.values()
            for name, da in params.data_vars.items()
            if "time" in da.dims
        )
    )
    if not param_names:
        return
    colors = _model_colors(list(params_by_scenario))
    fig, axes = plt.subplots(
        len(param_names), 1, figsize=(9, 3.0 * len(param_names)), squeeze=False
    )
    for axis, name in zip(axes[:, 0], param_names):
        for scenario, params in params_by_scenario.items():
            if name not in params or "time" not in params[name].dims:
                continue
            values = params[name]
            if "ensemble" in values.dims:
                values = values.mean("ensemble")
            axis.plot(
                np.asarray(values["time"].values, dtype=float),
                np.asarray(values.values, dtype=float),
                color=colors[scenario],
                label=scenario,
            )
        axis.set(ylabel=name.replace("_", " "), title=f"Prescribed {name}")
        axis.grid(True, alpha=0.3)
        axis.legend(fontsize=8)
    axes[-1, 0].set_xlabel("time [s]")
    fig.suptitle("Parameter scenarios replayed by every solver")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_parameter_scenario_response(
    params_by_scenario: dict[str, xarray.Dataset],
    fields: dict[str, xarray.Dataset],
    model_name: str,
    output_path: pathlib.Path,
) -> None:
    """Prescribed and realised inflow for several scenarios of one solver."""
    param_names = list(
        dict.fromkeys(
            name
            for params in params_by_scenario.values()
            for name, da in params.data_vars.items()
            if "time" in da.dims
        )
    )
    if not param_names:
        return
    colors = _model_colors(list(fields))
    derived: dict[str, dict[str, np.ndarray]] = {}
    for scenario, field in fields.items():
        probe = field.isel(height=-1, x=0)
        y_idx = [
            int(round(fraction * (probe.sizes["y"] - 1)))
            for fraction in (0.2, 0.5, 0.8)
        ]
        u = np.asarray(probe["u"].isel(y=y_idx).values)
        v = np.asarray(probe["v"].isel(y=y_idx).values)
        derived[scenario] = {
            "time": np.asarray(field["time"].values, dtype=float),
            "inflow_angle": np.degrees(np.arctan2(v, u)).mean(axis=-1),
            "velocity_magnitude": np.hypot(u, v).mean(axis=-1),
        }

    fig, axes = plt.subplots(
        len(param_names), 1, figsize=(9, 3.0 * len(param_names)), squeeze=False
    )
    for axis, param in zip(axes[:, 0], param_names):
        for scenario, params in params_by_scenario.items():
            prescribed = params[param]
            if "ensemble" in prescribed.dims:
                prescribed = prescribed.mean("ensemble")
            axis.plot(
                np.asarray(prescribed["time"].values, dtype=float),
                np.asarray(prescribed.values, dtype=float),
                color=colors[scenario],
                ls="--",
                lw=2.0,
                label=f"{scenario} prescribed",
            )
            if param in derived[scenario]:
                axis.plot(
                    derived[scenario]["time"],
                    derived[scenario][param],
                    color=colors[scenario],
                    lw=1.6,
                    label=f"{scenario} realised",
                )
        axis.set(ylabel=param.replace("_", " "), xlabel="time [s]")
        axis.grid(True, alpha=0.3)
        axis.legend(fontsize=8, ncol=2)
    fig.suptitle(f"Parameter-scenario response: {model_name}")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_scenario_error_summary(rows: list[dict], output_path: pathlib.Path) -> None:
    """Compare solver-discrepancy scalars across all forcing scenarios."""
    metric_names = [
        name
        for name in dict.fromkeys(key for row in rows for key in row)
        if name.startswith("field_rmse_") or name == "rolling_sensor_rmse_speed"
    ]
    if not metric_names:
        return
    scenarios = list(dict.fromkeys(str(row["scenario"]) for row in rows))
    models = list(dict.fromkeys(str(row["model"]) for row in rows))
    colors = _model_colors(models)
    fig, axes = plt.subplots(
        len(metric_names),
        1,
        figsize=(9, 2.9 * len(metric_names)),
        sharex=True,
        squeeze=False,
    )
    x = np.arange(len(scenarios))
    for axis, metric in zip(axes[:, 0], metric_names):
        for model in models:
            values = []
            for scenario in scenarios:
                row = next(
                    item
                    for item in rows
                    if item["scenario"] == scenario and item["model"] == model
                )
                values.append(float(row.get(metric, np.nan)))
            axis.plot(x, values, marker="o", color=colors[model], label=model)
        axis.set_ylabel(metric.replace("_", " ") + " [m/s]")
        axis.grid(True, alpha=0.3)
        axis.legend(fontsize=8)
    axes[-1, 0].set_xticks(x, scenarios)
    axes[-1, 0].set_xlabel("parameter scenario")
    fig.suptitle("Solver discrepancy across parameter scenarios")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_state_snapshots(
    fields: dict[str, xarray.Dataset],
    height_idx: int,
    height: float,
    n_snapshots: int,
    output_path: pathlib.Path,
) -> None:
    """|U| at one height: one row per model, one column per snapshot time."""
    names = list(fields)
    times = np.asarray(next(iter(fields.values()))["time"].values, dtype=float)
    t_idx = np.unique(np.linspace(0, len(times) - 1, n_snapshots).round().astype(int))

    planes = {
        n: np.asarray(f["vel_magnitude"].isel(height=height_idx).values)
        for n, f in fields.items()
    }
    vmax = float(np.nanmax([np.nanmax(p) for p in planes.values()]))

    fig, axes = plt.subplots(
        len(names),
        len(t_idx),
        figsize=(3.1 * len(t_idx), 2.9 * len(names)),
        squeeze=False,
    )
    for row, name in enumerate(names):
        for col, t in enumerate(t_idx):
            ax = axes[row, col]
            im = ax.imshow(planes[name][t], origin="lower", vmin=0.0, vmax=vmax)
            ax.set_xticks([])
            ax.set_yticks([])
            if row == 0:
                ax.set_title(f"t={times[t]:.0f} s", fontsize=10)
            if col == 0:
                ax.set_ylabel(name, fontsize=10)
    fig.colorbar(im, ax=axes, fraction=0.02, label="|U| [m/s]")
    fig.suptitle(f"|U| at z={height:.1f} m")
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_state_difference(
    fields: dict[str, xarray.Dataset],
    reference: str,
    height_idx: int,
    height: float,
    n_snapshots: int,
    output_path: pathlib.Path,
) -> None:
    """(model - reference) |U| at one height, one row per non-reference model."""
    others = [n for n in fields if n != reference]
    if not others:
        return
    times = np.asarray(fields[reference]["time"].values, dtype=float)
    t_idx = np.unique(np.linspace(0, len(times) - 1, n_snapshots).round().astype(int))
    ref_plane = np.asarray(
        fields[reference]["vel_magnitude"].isel(height=height_idx).values
    )
    diffs = {
        n: np.asarray(fields[n]["vel_magnitude"].isel(height=height_idx).values)
        - ref_plane
        for n in others
    }
    lim = float(np.nanmax([np.nanmax(np.abs(d)) for d in diffs.values()])) or 1.0

    fig, axes = plt.subplots(
        len(others),
        len(t_idx),
        figsize=(3.1 * len(t_idx), 2.9 * len(others)),
        squeeze=False,
    )
    for row, name in enumerate(others):
        for col, t in enumerate(t_idx):
            ax = axes[row, col]
            im = ax.imshow(
                diffs[name][t], origin="lower", vmin=-lim, vmax=lim, cmap="RdBu_r"
            )
            ax.set_xticks([])
            ax.set_yticks([])
            if row == 0:
                ax.set_title(f"t={times[t]:.0f} s", fontsize=10)
            if col == 0:
                ax.set_ylabel(f"{name}\n- {reference}", fontsize=9)
    fig.colorbar(im, ax=axes, fraction=0.02, label="Δ|U| [m/s]")
    fig.suptitle(f"|U| difference against {reference} at z={height:.1f} m")
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_field_rmse(
    fields: dict[str, xarray.Dataset],
    reference: str,
    heights: np.ndarray,
    output_path: pathlib.Path,
) -> dict[str, dict[str, float]]:
    """Per-height |U| RMSE against the reference over time.

    Returns ``{"z<height>": {model: time-mean RMSE}}``, the reference included at
    0.0 (it is the baseline), so the summary table has a column per height for
    every model.
    """
    others = [n for n in fields if n != reference]
    times = np.asarray(fields[reference]["time"].values, dtype=float)
    colors = _model_colors(list(fields))
    ref = fields[reference]["vel_magnitude"]

    fig, axes = plt.subplots(
        len(heights), 1, figsize=(9, 2.8 * len(heights)), squeeze=False, sharex=True
    )
    time_means: dict[str, dict[str, float]] = {}
    for row, height in enumerate(heights):
        ax = axes[row, 0]
        time_means[f"z{height:.0f}"] = {reference: 0.0}
        for name in others:
            diff = np.asarray(
                (fields[name]["vel_magnitude"] - ref).isel(height=row).values
            )
            rmse = np.sqrt(np.nanmean(diff**2, axis=(-2, -1)))
            ax.plot(times, rmse, color=colors[name], lw=1.8, label=name)
            time_means[f"z{height:.0f}"][name] = float(np.nanmean(rmse))
        ax.set_ylabel("|U| RMSE [m/s]")
        ax.set_title(f"z = {height:.1f} m", loc="left", fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(bottom=0.0)
        if row == 0:
            ax.legend(loc="best", fontsize=8)
    axes[-1, 0].set_xlabel("time [s]")
    fig.suptitle(f"Field |U| RMSE against {reference}")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return time_means


def animate_states(
    fields: dict[str, xarray.Dataset],
    heights: np.ndarray,
    output_path: pathlib.Path,
    fps: int,
) -> pathlib.Path:
    """Animate |U|: one column per model, one row per height, shared colour scale."""
    import matplotlib.animation as animation

    names = list(fields)
    times = np.asarray(fields[names[0]]["time"].values, dtype=float)
    frames = {
        n: np.asarray(f["vel_magnitude"].values) for n, f in fields.items()
    }  # (time, height, y, x)
    vmax = float(np.nanmax([np.nanmax(f) for f in frames.values()]))

    fig, axes = plt.subplots(
        len(heights),
        len(names),
        figsize=(3.4 * len(names), 3.1 * len(heights)),
        squeeze=False,
        constrained_layout=True,
    )
    images = []
    for row in range(len(heights)):
        for col, name in enumerate(names):
            ax = axes[row, col]
            im = ax.imshow(frames[name][0, row], origin="lower", vmin=0.0, vmax=vmax)
            ax.set_xticks([])
            ax.set_yticks([])
            if row == 0:
                ax.set_title(name)
            if col == 0:
                ax.set_ylabel(f"z={heights[row]:.0f} m")
            images.append((im, name, row))
    fig.colorbar(images[0][0], ax=axes, fraction=0.02, label="|U| [m/s]")
    suptitle = fig.suptitle(f"|U|   t={times[0]:.0f} s")

    def update(frame: int) -> list:
        artists: list = [suptitle]
        for im, name, row in images:
            im.set_array(frames[name][frame, row])
            artists.append(im)
        suptitle.set_text(f"|U|   t={times[frame]:.0f} s")
        return artists

    # h264 rejects odd frame dimensions, and the panel grid above sizes the
    # figure from the model/height counts -- round both axes up to an even pixel
    # count at the save dpi so any layout encodes.
    dpi = 120
    fig.set_size_inches(
        *(np.ceil(np.asarray(fig.get_size_inches()) * dpi / 2) * 2 / dpi)
    )

    # Falls back to a .gif path when ffmpeg is unavailable.
    raw_path, writer = _get_writer_and_output_path(output_path, fps)
    save_path = pathlib.Path(raw_path)
    anim = animation.FuncAnimation(fig, update, frames=len(times), blit=False)
    anim.save(str(save_path), writer=writer, dpi=dpi)
    plt.close(fig)
    return save_path


_AGGREGATIONS = ("mean", "median", "max", "min")


def rolling_aggregate(
    series: xarray.DataArray, window_seconds: float, aggregation_mode: str
) -> xarray.DataArray:
    """Smooth a sensor series with a sliding window of the observation length.

    The observation operator (``TemporalObservationOperator``, ``mode="intervals"``)
    reduces each ``interval_seconds`` window to one number by ``aggregation_mode``.
    Here the *same window length and reduction* slide over every frame instead of
    tiling disjoint bins, so the result is a smooth curve on the original time
    axis rather than a staircase -- it shows the same suppression of sub-interval
    fluctuation without the arbitrary phase of a fixed bin grid.

    The window is centred (no lag against the raw series) and partial at the two
    ends (``min_periods=1``), so the curve spans the full series.

    Aggregating the *sensor series* rather than the state field is exact for
    ``mean`` (the spatial interpolation is linear, so it commutes with the time
    average); for median/max/min the two orders differ slightly, and the
    per-sensor value plotted here is the more directly interpretable one.

    Returns a series on the same ``time`` axis as the input.
    """
    if aggregation_mode not in _AGGREGATIONS:
        raise ValueError(
            f"Invalid aggregation_mode {aggregation_mode!r}; "
            f"must be one of {list(_AGGREGATIONS)}."
        )
    times = np.asarray(series["time"].values, dtype=float)
    if times.size == 0:
        raise ValueError("Sensor series has no time steps to aggregate.")

    # Window length in frames, from this series' own cadence -- the backends emit
    # at a uniform (but not necessarily equal) output frequency, so the same
    # physical window is a different frame count per model.
    cadence = float(np.median(np.diff(times))) if times.size > 1 else 0.0
    if cadence <= 0.0:
        return series
    window = max(1, int(round(window_seconds / cadence)))
    # Force an odd frame count. A centred even window spans [i - w//2, i + (w+1)//2),
    # i.e. half a frame left of centre -- and since w is derived from each model's
    # own cadence, that offset would differ per model and show up as a spurious
    # relative phase shift between the very curves this figure compares. An odd
    # window is exactly symmetric at the cost of at most one frame of length.
    window += 1 - window % 2

    rolling = series.rolling(time=window, center=True, min_periods=1)
    return getattr(rolling, aggregation_mode)()


def plot_sensor_rolling(
    smoothed: dict[str, xarray.DataArray],
    raw: dict[str, xarray.DataArray],
    sensor_points: tuple[np.ndarray, np.ndarray, np.ndarray],
    title: str,
    output_path: pathlib.Path,
) -> None:
    """Sliding-window sensor observations, one model per colour.

    The smoothed curve is drawn over a faint copy of the raw per-frame series, so
    the sub-interval fluctuation the observation operator averages away stays
    visible behind it. ``|U|`` is formed from the *smoothed* components, matching
    the operator's order (aggregate, then observe).
    """
    sx, sy, sz = sensor_points
    names = list(smoothed)
    colors = _model_colors(names)
    n_sensors = int(next(iter(smoothed.values())).sizes["sensor"])
    columns = ["u", "v", "w", "|U|"]

    fig, axes = plt.subplots(
        n_sensors,
        len(columns),
        figsize=(3.6 * len(columns), 2.4 * n_sensors),
        squeeze=False,
        sharex=True,
    )
    for row in range(n_sensors):
        for col, component in enumerate(columns):
            ax = axes[row, col]
            for name in names:
                smooth_da, raw_da = smoothed[name], raw[name]

                def _pick(da: xarray.DataArray) -> xarray.DataArray:
                    return (
                        sensor_magnitude(da)
                        if component == "|U|"
                        else da.sel(component=component)
                    )

                ax.plot(
                    np.asarray(raw_da["time"].values, dtype=float),
                    np.asarray(_pick(raw_da).isel(sensor=row).values),
                    color=colors[name],
                    lw=0.8,
                    alpha=0.3,
                )
                ax.plot(
                    np.asarray(smooth_da["time"].values, dtype=float),
                    np.asarray(_pick(smooth_da).isel(sensor=row).values),
                    color=colors[name],
                    lw=2.2,
                    label=name,
                )
            ax.grid(True, alpha=0.3)
            if row == 0:
                ax.set_title(component)
                if col == 0:
                    ax.legend(loc="best", fontsize=8)
            if col == 0:
                ax.set_ylabel(
                    f"sensor {row}\n(x={sx[row]:.0f}, y={sy[row]:.0f}, z={sz[row]:.0f})",
                    fontsize=8,
                )
    for ax in axes[-1, :]:
        ax.set_xlabel("time [s]")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_sensor_comparison(
    series: dict[str, xarray.DataArray],
    sensor_points: tuple[np.ndarray, np.ndarray, np.ndarray],
    title: str,
    output_path: pathlib.Path,
) -> None:
    """u/v/w/|U| at each sensor, one line per model.

    ``series`` maps a model name to its ``(component, time, sensor)`` series,
    interpolated at the physical sensor points on that model's own grid -- so the
    panels are directly comparable despite the differing grids.
    """
    sx, sy, sz = sensor_points
    names = list(series)
    colors = _model_colors(names)
    n_sensors = int(next(iter(series.values())).sizes["sensor"])
    columns = ["u", "v", "w", "|U|"]

    fig, axes = plt.subplots(
        n_sensors,
        len(columns),
        figsize=(3.6 * len(columns), 2.4 * n_sensors),
        squeeze=False,
        sharex=True,
    )
    for row in range(n_sensors):
        for col, component in enumerate(columns):
            ax = axes[row, col]
            for name in names:
                da = series[name]
                values = (
                    sensor_magnitude(da)
                    if component == "|U|"
                    else da.sel(component=component)
                )
                ax.plot(
                    np.asarray(da["time"].values, dtype=float),
                    np.asarray(values.isel(sensor=row).values),
                    color=colors[name],
                    lw=1.5,
                    label=name,
                )
            ax.grid(True, alpha=0.3)
            if row == 0:
                ax.set_title(component)
                if col == 0:
                    ax.legend(loc="best", fontsize=8)
            if col == 0:
                ax.set_ylabel(
                    f"sensor {row}\n(x={sx[row]:.0f}, y={sy[row]:.0f}, z={sz[row]:.0f})",
                    fontsize=8,
                )
    for ax in axes[-1, :]:
        ax.set_xlabel("time [s]")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _cell_centres(lo: float, hi: float, n: int) -> np.ndarray:
    edges = np.linspace(lo, hi, n + 1)
    return 0.5 * (edges[:-1] + edges[1:])


def load_geometry_info(
    stl_path: str | pathlib.Path,
    x: np.ndarray,
    y: np.ndarray,
    z_min: float,
    z_max: float,
) -> GeometryInfo:
    """Rasterise the STL into a common-grid solid mask without solver imports.

    PALM's topography is itself a height map, while uDALES has a full immersed
    boundary representation. For a cross-solver statistic the conservative
    common definition of fluid is therefore the part of a column above the
    highest STL intersection. This also works exactly for the benchmark's
    extruded cubes. If ray casting is unavailable, retain all cells and warn
    rather than failing an otherwise valid comparison.
    """
    roof = np.zeros((len(y), len(x)), dtype=float)
    x_min, x_max = float(x.min()), float(x.max())
    y_min, y_max = float(y.min()), float(y.max())
    try:
        import trimesh

        mesh = trimesh.load(pathlib.Path(stl_path))
        if isinstance(mesh, trimesh.Scene):
            meshes = [
                g for g in mesh.geometry.values() if isinstance(g, trimesh.Trimesh)
            ]
            mesh = trimesh.util.concatenate(meshes)
        if not isinstance(mesh, trimesh.Trimesh):
            raise TypeError("STL did not contain a triangular mesh")
        bounds = np.asarray(mesh.bounds, dtype=float)
        x_min, y_min = bounds[0, :2]
        x_max, y_max = bounds[1, :2]

        xx, yy = np.meshgrid(x, y, indexing="xy")
        origins = np.column_stack(
            [xx.ravel(), yy.ravel(), np.full(xx.size, z_max + 1.0)]
        )
        directions = np.tile(np.array([0.0, 0.0, -1.0]), (xx.size, 1))
        locations, rays, _ = trimesh.ray.ray_triangle.RayMeshIntersector(
            mesh
        ).intersects_location(origins, directions, multiple_hits=True)
        flat = np.zeros(xx.size, dtype=float)
        for location, ray in zip(locations, rays):
            flat[ray] = max(flat[ray], float(location[2]) - z_min)
        roof = np.clip(flat.reshape(yy.shape), 0.0, None)
    except Exception as exc:  # geometry masking improves diagnostics, not availability
        warnings.warn(
            f"Could not rasterise {stl_path} for fluid-cell masking ({exc}); "
            "statistics include all common-grid cells.",
            stacklevel=2,
        )
    return GeometryInfo(roof, x_min, x_max, y_min, y_max)


def _dilate(mask: np.ndarray, cells: int) -> np.ndarray:
    """Chebyshev-dilate a 2-D mask without wrapping across domain edges."""
    out = np.asarray(mask, dtype=bool)
    for _ in range(max(0, cells)):
        padded = np.pad(out, 1, constant_values=False)
        out = np.logical_or.reduce(
            [
                padded[i : i + out.shape[0], j : j + out.shape[1]]
                for i in range(3)
                for j in range(3)
            ]
        )
    return out


def fluid_masks(
    geometry: GeometryInfo, heights: np.ndarray, wall_exclusion_cells: int
) -> np.ndarray:
    """Return a shared ``(height, y, x)`` fluid mask, excluding wall neighbours."""
    masks = []
    for height in heights:
        solid = np.asarray(height <= geometry.roof_heights, dtype=bool)
        masks.append(~_dilate(solid, wall_exclusion_cells))
    return np.asarray(masks)


def auto_regions(
    geometry: GeometryInfo,
    x: np.ndarray,
    y: np.ndarray,
) -> list[Region]:
    """Infer stable upstream/canopy/wake x-regions from the STL bounding box."""
    domain_x0, domain_x1 = float(x.min()), float(x.max())
    domain_y0, domain_y1 = float(y.min()), float(y.max())
    bx0 = float(np.clip(geometry.x_min, domain_x0, domain_x1))
    bx1 = float(np.clip(geometry.x_max, domain_x0, domain_x1))
    by0 = float(np.clip(geometry.y_min, domain_y0, domain_y1))
    by1 = float(np.clip(geometry.y_max, domain_y0, domain_y1))
    if bx1 <= bx0:  # degraded no-STL fallback: use central thirds of the domain
        bx0 = domain_x0 + (domain_x1 - domain_x0) / 3.0
        bx1 = domain_x0 + 2.0 * (domain_x1 - domain_x0) / 3.0
    if by1 <= by0:
        by0, by1 = domain_y0, domain_y1
    wake_mid = bx1 + 0.5 * max(domain_x1 - bx1, 0.0)
    return [
        Region("upstream", domain_x0, max(bx0, domain_x0), domain_y0, domain_y1),
        Region("canopy", bx0, bx1, by0, by1),
        Region("near_wake", bx1, wake_mid, domain_y0, domain_y1),
        Region("far_wake", wake_mid, domain_x1, domain_y0, domain_y1),
    ]


def _region_mask(region: Region, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    xx, yy = np.meshgrid(x, y, indexing="xy")
    return (
        (xx >= region.x_min)
        & (xx <= region.x_max)
        & (yy >= region.y_min)
        & (yy <= region.y_max)
    )


def statistic_windows(
    times: np.ndarray, window_seconds: float, max_windows: int
) -> list[tuple[str, np.ndarray]]:
    """Select evenly distributed non-overlapping statistic windows."""
    if window_seconds <= 0:
        raise ValueError("compare.analysis.statistics_window_seconds must be positive.")
    if times.size == 0:
        return []
    cadence = float(np.median(np.diff(times))) if times.size > 1 else 1.0
    edges = np.arange(times[0], times[-1] + cadence, window_seconds)
    if edges[-1] < times[-1] + 0.5 * cadence:
        edges = np.append(edges, times[-1] + cadence)
    windows = []
    for start, stop in zip(edges[:-1], edges[1:]):
        idx = np.flatnonzero((times >= start) & (times < stop))
        if idx.size:
            windows.append((f"{start:.0f}–{times[idx[-1]]:.0f} s", idx))
    if not windows:
        return [(f"{times[0]:.0f} s", np.arange(times.size))]
    if len(windows) > max_windows:
        pick = np.unique(
            np.linspace(0, len(windows) - 1, max_windows).round().astype(int)
        )
        windows = [windows[i] for i in pick]
    return windows


def plot_measurement_overview(
    fields: dict[str, xarray.Dataset],
    reference: str,
    heights: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    geometry: GeometryInfo,
    regions: list[Region],
    sensor_sets: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]],
    inlet_probe_height: float,
    profile_heights: np.ndarray,
    wall_exclusion_cells: int,
    output_path: pathlib.Path,
) -> None:
    """Map every comparison footprint onto time-mean reference-state slices.

    Region rectangles are areas, not arbitrary visual guides: profiles, PDFs,
    CDFs, and regional wake metrics reduce all fluid cells in them. The wake
    curve itself averages over the full fluid y-span at each x, so its legend
    explicitly says it is a crosswind average rather than a single line.
    """
    region_colors = {
        "upstream": "tab:blue",
        "canopy": "tab:orange",
        "near_wake": "tab:green",
        "far_wake": "tab:red",
    }
    sensor_colors = {"assimilation": "tab:purple", "validation": "tab:brown"}
    mean_speed = np.asarray(fields[reference]["vel_magnitude"].mean("time").values)
    vmax = float(np.nanmax(mean_speed)) if np.isfinite(mean_speed).any() else 1.0
    fig, axes = plt.subplots(
        1,
        len(heights),
        figsize=(5.0 * len(heights), 6.6),
        squeeze=False,
        constrained_layout=False,
    )
    height_spacing = (
        float(np.min(np.diff(np.sort(heights)))) if len(heights) > 1 else 0.5
    )
    extent = (float(x.min()), float(x.max()), float(y.min()), float(y.max()))
    for height_idx, height in enumerate(heights):
        ax = axes[0, height_idx]
        image = ax.imshow(
            mean_speed[height_idx],
            origin="lower",
            extent=extent,
            vmin=0.0,
            vmax=vmax,
            aspect="equal",
        )
        solid = np.ma.masked_where(
            height > geometry.roof_heights, geometry.roof_heights
        )
        ax.imshow(
            solid,
            origin="lower",
            extent=extent,
            cmap="Greys",
            alpha=0.75,
            aspect="equal",
        )
        for region in regions:
            if region.x_max <= region.x_min or region.y_max <= region.y_min:
                continue
            ax.add_patch(
                Rectangle(
                    (region.x_min, region.y_min),
                    region.x_max - region.x_min,
                    region.y_max - region.y_min,
                    fill=False,
                    lw=2.0,
                    ec=region_colors.get(region.name, "black"),
                )
            )
        for set_name, (sx, sy, sz) in sensor_sets.items():
            color = sensor_colors.get(set_name, "tab:pink")
            on_slice = (
                np.abs(np.asarray(sz, dtype=float) - height) <= 0.5 * height_spacing
            )
            ax.scatter(
                np.asarray(sx)[~on_slice],
                np.asarray(sy)[~on_slice],
                s=52,
                facecolors="none",
                edgecolors=color,
                linewidths=1.5,
                alpha=0.45,
                marker="o",
            )
            ax.scatter(
                np.asarray(sx)[on_slice],
                np.asarray(sy)[on_slice],
                s=52,
                color=color,
                edgecolors="white",
                linewidths=0.6,
                marker="o",
            )
        # plot_parameters derives inflow from three points at the highest
        # requested comparison level, one x column inside the domain.
        if np.isclose(height, inlet_probe_height):
            probe_y = np.asarray(
                [y[int(round(fraction * (len(y) - 1)))] for fraction in (0.2, 0.5, 0.8)]
            )
            ax.scatter(
                np.full_like(probe_y, x[0]),
                probe_y,
                marker="*",
                color="black",
                edgecolors="white",
                linewidths=0.5,
                s=120,
                zorder=5,
            )
        ax.set(title=f"z={height:.1f} m", xlabel="x [m]")
        if height_idx == 0:
            ax.set_ylabel("y [m]")

    handles: list = [
        Patch(facecolor="0.5", edgecolor="0.2", label="STL solid at this z"),
        Line2D(
            [],
            [],
            marker="o",
            color="none",
            markerfacecolor=sensor_colors["assimilation"],
            markeredgecolor="white",
            markersize=7,
            label="Assimilation sensors (filled on their z slice)",
        ),
        Line2D(
            [],
            [],
            marker="o",
            color="none",
            markerfacecolor=sensor_colors["validation"],
            markeredgecolor="white",
            markersize=7,
            label="Validation sensors (filled on their z slice)",
        ),
        Line2D(
            [],
            [],
            marker="*",
            color="black",
            markersize=10,
            label=f"Inlet probes (z={inlet_probe_height:.1f} m)",
        ),
    ]
    handles.extend(
        Line2D(
            [],
            [],
            color=region_colors.get(region.name, "black"),
            lw=2,
            label=f"{region.name}: regional statistics",
        )
        for region in regions
        if region.x_max > region.x_min and region.y_max > region.y_min
    )
    fig.colorbar(
        image,
        ax=axes,
        fraction=0.025,
        pad=0.02,
        label=f"time-mean |U| [m/s] ({reference})",
    )
    fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.115),
        ncol=3,
        fontsize=8,
        frameon=True,
    )
    fig.suptitle("Comparison measurement overview", y=0.98)
    overview_text = (
        "Filled sensor markers are sampled on the displayed z slice; hollow markers show the same x/y point at another z. "
        f"Regional profiles and wake metrics use all {len(profile_heights)} profile levels "
        f"({profile_heights.min():.1f}–{profile_heights.max():.1f} m). "
        "Wake-recovery curves are crosswind averages over all fluid y at each x. "
        f"Field statistics/distributions exclude solids plus {wall_exclusion_cells} adjacent common-grid cell(s)."
    )
    fig.text(
        0.5,
        0.018,
        textwrap.fill(overview_text, width=46 * len(heights)),
        ha="center",
        va="bottom",
        fontsize=8,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.9, "pad": 2},
    )
    fig.subplots_adjust(bottom=0.33, top=0.88, wspace=0.15)
    fig.savefig(output_path, dpi=170)
    plt.close(fig)


def plot_windowed_field_statistics(
    fields: dict[str, xarray.Dataset],
    heights: np.ndarray,
    masks: np.ndarray,
    windows: list[tuple[str, np.ndarray]],
    statistic: str,
    output_dir: pathlib.Path,
) -> None:
    """Plot windowed mean or temporal standard-deviation maps of speed."""
    if statistic not in ("mean", "std"):
        raise ValueError(f"Unsupported field statistic {statistic!r}")
    names = list(fields)
    reducer = np.nanmean if statistic == "mean" else np.nanstd
    for height_idx, height in enumerate(heights):
        maps = {
            (name, window_idx): np.where(
                masks[height_idx],
                reducer(
                    np.asarray(
                        fields[name]["vel_magnitude"]
                        .isel(time=idx, height=height_idx)
                        .values
                    ),
                    axis=0,
                ),
                np.nan,
            )
            for name in names
            for window_idx, (_, idx) in enumerate(windows)
        }
        finite = [values[np.isfinite(values)] for values in maps.values()]
        vmax = max(
            (float(np.nanmax(values)) for values in finite if values.size), default=1.0
        )
        vmin = (
            0.0
            if statistic == "std"
            else min(
                (float(np.nanmin(values)) for values in finite if values.size),
                default=0.0,
            )
        )
        fig, axes = plt.subplots(
            len(names),
            len(windows),
            figsize=(3.0 * len(windows), 2.8 * len(names)),
            squeeze=False,
        )
        for row, name in enumerate(names):
            for col, (label, _) in enumerate(windows):
                ax = axes[row, col]
                image = ax.imshow(maps[name, col], origin="lower", vmin=vmin, vmax=vmax)
                ax.set_xticks([])
                ax.set_yticks([])
                if row == 0:
                    ax.set_title(label, fontsize=9)
                if col == 0:
                    ax.set_ylabel(name)
        fig.colorbar(image, ax=axes, fraction=0.02, label=f"{statistic} |U| [m/s]")
        fig.suptitle(f"{statistic.title()} |U| at z={height:.1f} m (fluid cells only)")
        fig.savefig(
            output_dir / f"field_{statistic}_z{height:.0f}.png",
            dpi=150,
            bbox_inches="tight",
        )
        plt.close(fig)


def _masked_values(values: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Flatten time and horizontal dimensions after applying a 2-D fluid mask."""
    if not np.any(mask):
        return np.array([], dtype=float)
    return np.asarray(values)[..., mask].reshape(-1)


def _component_values(
    field: xarray.Dataset, height_idx: int, mask: np.ndarray, component: str
) -> np.ndarray:
    if component == "|U|":
        values = field["vel_magnitude"].isel(height=height_idx).values
    else:
        values = field[component].isel(height=height_idx).values
    return _masked_values(np.asarray(values), mask)


def plot_vertical_profiles(
    fields: dict[str, xarray.Dataset],
    heights: np.ndarray,
    masks: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    regions: list[Region],
    output_dir: pathlib.Path,
) -> None:
    """Regional profiles of mean speed/direction and component variability."""
    colors = _model_colors(list(fields))
    for region in regions:
        horizontal = _region_mask(region, x, y)
        fig, axes = plt.subplots(1, 3, figsize=(12, 5.8), sharey=True)
        for name, field in fields.items():
            mean_speed, direction = [], []
            component_std: dict[str, list[float]] = {
                component: [] for component in ("u", "v", "w")
            }
            for height_idx in range(len(heights)):
                mask = horizontal & masks[height_idx]
                u = _component_values(field, height_idx, mask, "u")
                v = _component_values(field, height_idx, mask, "v")
                speed = _component_values(field, height_idx, mask, "|U|")
                mean_speed.append(float(np.nanmean(speed)) if speed.size else np.nan)
                direction.append(
                    float(np.degrees(np.arctan2(np.nanmean(v), np.nanmean(u))))
                    if u.size
                    else np.nan
                )
                for component in component_std:
                    values = _component_values(field, height_idx, mask, component)
                    component_std[component].append(
                        float(np.nanstd(values)) if values.size else np.nan
                    )
            axes[0].plot(mean_speed, heights, color=colors[name], label=name)
            axes[1].plot(direction, heights, color=colors[name], label=name)
            for component, values in component_std.items():
                axes[2].plot(
                    values,
                    heights,
                    color=colors[name],
                    ls={"u": "-", "v": "--", "w": ":"}[component],
                    label=f"{name} {component}",
                )
        axes[0].set(title="Mean speed", xlabel="|U| [m/s]", ylabel="height [m]")
        axes[1].set(
            title="Mean direction", xlabel="atan2(v, u) [deg]", xlim=(-180, 180)
        )
        axes[2].set(title="Component variability", xlabel="temporal/spatial std [m/s]")
        axes[0].legend(fontsize=8)
        axes[2].legend(fontsize=7, ncol=2)
        for ax in axes:
            ax.grid(True, alpha=0.3)
        fig.suptitle(f"Regional vertical profiles: {region.name}")
        fig.tight_layout()
        fig.savefig(output_dir / f"vertical_profiles_{region.name}.png", dpi=150)
        plt.close(fig)


def _sample(values: np.ndarray, max_samples: int) -> np.ndarray:
    values = values[np.isfinite(values)]
    if values.size <= max_samples:
        return values
    return values[np.linspace(0, values.size - 1, max_samples).round().astype(int)]


def plot_field_distributions(
    fields: dict[str, xarray.Dataset],
    heights: np.ndarray,
    masks: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    regions: list[Region],
    max_samples: int,
    kind: str,
    output_dir: pathlib.Path,
) -> None:
    """Plot regional fluid-cell PDFs or CDFs for all velocity components."""
    colors = _model_colors(list(fields))
    components = ("u", "v", "w", "|U|")
    for region in regions:
        horizontal = _region_mask(region, x, y)
        fig, axes = plt.subplots(
            len(heights),
            len(components),
            figsize=(14, 2.8 * len(heights)),
            squeeze=False,
        )
        for height_idx, height in enumerate(heights):
            mask = horizontal & masks[height_idx]
            for comp_idx, component in enumerate(components):
                ax = axes[height_idx, comp_idx]
                samples = {
                    name: _sample(
                        _component_values(field, height_idx, mask, component),
                        max_samples,
                    )
                    for name, field in fields.items()
                }
                nonempty = [values for values in samples.values() if values.size]
                if nonempty:
                    combined = np.concatenate(nonempty)
                    lo, hi = np.nanpercentile(combined, [1.0, 99.0])
                    if np.isclose(lo, hi):
                        lo, hi = lo - 0.5, hi + 0.5
                    for name, values in samples.items():
                        if not values.size:
                            continue
                        if kind == "pdf":
                            density, edges = np.histogram(
                                values, bins=50, range=(lo, hi), density=True
                            )
                            ax.plot(
                                0.5 * (edges[1:] + edges[:-1]),
                                density,
                                color=colors[name],
                                label=name,
                            )
                        else:
                            ordered = np.sort(values)
                            ax.plot(
                                ordered,
                                np.linspace(0.0, 1.0, ordered.size),
                                color=colors[name],
                                label=name,
                            )
                if height_idx == 0:
                    ax.set_title(component)
                if comp_idx == 0:
                    ax.set_ylabel(f"z={height:.1f} m\n{kind}")
                ax.grid(True, alpha=0.3)
                if height_idx == 0 and comp_idx == 0:
                    ax.legend(fontsize=8)
        fig.suptitle(f"Fluid-cell {kind.upper()}s: {region.name}")
        fig.tight_layout()
        fig.savefig(output_dir / f"field_{kind}_{region.name}.png", dpi=150)
        plt.close(fig)


def _sensor_values(series: xarray.DataArray, component: str) -> np.ndarray:
    da = (
        sensor_magnitude(series)
        if component == "|U|"
        else series.sel(component=component)
    )
    return np.asarray(da.transpose("time", "sensor").values, dtype=float)


def _series_metrics(reference: np.ndarray, candidate: np.ndarray) -> dict[str, float]:
    valid = np.isfinite(reference) & np.isfinite(candidate)
    if not np.any(valid):
        return {
            key: np.nan for key in ("bias", "mae", "rmse", "correlation", "std_ratio")
        }
    ref, cand = reference[valid], candidate[valid]
    diff = cand - ref
    ref_std = float(np.std(ref))
    cand_std = float(np.std(cand))
    correlation = (
        np.nan
        if ref_std == 0.0 or cand_std == 0.0
        else float(np.corrcoef(ref, cand)[0, 1])
    )
    return {
        "bias": float(np.mean(diff)),
        "mae": float(np.mean(np.abs(diff))),
        "rmse": float(np.sqrt(np.mean(diff**2))),
        "correlation": correlation,
        "std_ratio": cand_std / ref_std if ref_std else np.nan,
    }


def sensor_metric_rows(
    series: dict[str, xarray.DataArray], reference: str, set_name: str
) -> list[dict]:
    """Calculate pooled sensor/time metrics on already windowed series."""
    rows = []
    ref = series[reference]
    for name, candidate in series.items():
        aligned = candidate.interp(time=ref["time"])
        for component in ("u", "v", "w", "|U|"):
            values = _series_metrics(
                _sensor_values(ref, component), _sensor_values(aligned, component)
            )
            rows.append(
                {
                    "sensor_set": set_name,
                    "model": name,
                    "component": component,
                    **values,
                }
            )
    return rows


def plot_sensor_metrics(
    rows: list[dict], reference: str, set_name: str, output_path: pathlib.Path
) -> None:
    """Compact heatmaps of rolling-window sensor metrics, pooled over sensors."""
    models = list(dict.fromkeys(row["model"] for row in rows))
    components = ["u", "v", "w", "|U|"]
    metrics = ("bias", "mae", "rmse", "correlation", "std_ratio")
    fig, axes = plt.subplots(
        1,
        len(metrics),
        figsize=(3.0 * len(metrics), 0.8 * len(models) + 2.2),
        squeeze=False,
    )
    for axis, metric in zip(axes[0], metrics):
        values = np.array(
            [
                [
                    next(
                        row[metric]
                        for row in rows
                        if row["model"] == model and row["component"] == comp
                    )
                    for comp in components
                ]
                for model in models
            ]
        )
        cmap = "RdBu_r" if metric in ("bias", "correlation") else "viridis"
        image = axis.imshow(values, aspect="auto", cmap=cmap)
        axis.set(
            title=metric,
            xticks=range(len(components)),
            xticklabels=components,
            yticks=range(len(models)),
            yticklabels=models,
        )
        for row_idx in range(len(models)):
            for col_idx in range(len(components)):
                value = values[row_idx, col_idx]
                axis.text(
                    col_idx,
                    row_idx,
                    f"{value:.2g}",
                    ha="center",
                    va="center",
                    fontsize=7,
                )
        fig.colorbar(image, ax=axis, fraction=0.05)
    fig.suptitle(f"Rolling-window sensor metrics against {reference}: {set_name}")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _power_spectrum(values: np.ndarray, dt: float) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(values, dtype=float)
    valid = np.all(np.isfinite(values), axis=0)
    values = values[:, valid]
    if values.shape[0] < 4 or values.shape[1] == 0:
        return np.array([]), np.array([])
    demeaned = values - values.mean(axis=0, keepdims=True)
    transform = np.fft.rfft(demeaned, axis=0)
    return (
        np.fft.rfftfreq(values.shape[0], dt)[1:],
        (dt / values.shape[0] * np.abs(transform) ** 2).mean(axis=1)[1:],
    )


def _autocorrelation(values: np.ndarray) -> np.ndarray:
    correlations = []
    for column in np.asarray(values, dtype=float).T:
        column = column[np.isfinite(column)]
        if column.size < 4 or np.std(column) == 0.0:
            continue
        column = column - column.mean()
        corr = np.correlate(column, column, mode="full")[column.size - 1 :]
        correlations.append(corr / corr[0])
    return (
        np.nanmean(np.asarray(correlations), axis=0) if correlations else np.array([])
    )


def plot_sensor_spectra(
    series: dict[str, xarray.DataArray],
    set_name: str,
    max_lag_seconds: float,
    output_path: pathlib.Path,
) -> None:
    """Compare sensor-averaged PSDs and autocorrelations for all components."""
    colors = _model_colors(list(series))
    components = ("u", "v", "w", "|U|")
    fig, axes = plt.subplots(
        len(components), 2, figsize=(11, 2.6 * len(components)), squeeze=False
    )
    for row, component in enumerate(components):
        for name, da in series.items():
            times = np.asarray(da["time"].values, dtype=float)
            if times.size < 4:
                continue
            dt = float(np.median(np.diff(times)))
            values = _sensor_values(da, component)
            frequency, power = _power_spectrum(values, dt)
            if frequency.size:
                axes[row, 0].loglog(frequency, power, color=colors[name], label=name)
            corr = _autocorrelation(values)
            if corr.size:
                lag = np.arange(corr.size) * dt
                keep = lag <= max_lag_seconds
                axes[row, 1].plot(lag[keep], corr[keep], color=colors[name], label=name)
        axes[row, 0].set(ylabel=component, xlabel="frequency [Hz]")
        axes[row, 1].set(xlabel="lag [s]", ylim=(-0.2, 1.05))
        for axis in axes[row]:
            axis.grid(True, alpha=0.3)
        if row == 0:
            axes[row, 0].legend(fontsize=8)
            axes[row, 0].set_title("Sensor-mean power spectrum")
            axes[row, 1].set_title("Sensor-mean autocorrelation")
    fig.suptitle(f"Temporal scales at {set_name} sensors")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def wake_metric_rows(
    fields: dict[str, xarray.Dataset],
    heights: np.ndarray,
    masks: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    regions: list[Region],
    streamwise_angle_deg: float,
    geometry: GeometryInfo,
    recovery_deficit: float,
) -> list[dict]:
    """Compute regional wake/recovery statistics from time-resolved fields."""
    direction = np.array(
        [
            np.cos(np.deg2rad(streamwise_angle_deg)),
            np.sin(np.deg2rad(streamwise_angle_deg)),
        ]
    )
    regional_masks = {region.name: _region_mask(region, x, y) for region in regions}
    rows = []
    upstream = next(region for region in regions if region.name == "upstream")
    for name, field in fields.items():
        for height_idx, height in enumerate(heights):
            streamwise = direction[0] * np.asarray(
                field["u"].isel(height=height_idx).values
            ) + direction[1] * np.asarray(field["v"].isel(height=height_idx).values)
            w = np.asarray(field["w"].isel(height=height_idx).values)
            speed = np.asarray(field["vel_magnitude"].isel(height=height_idx).values)
            base_mask = masks[height_idx]
            upstream_values = _masked_values(
                streamwise, base_mask & regional_masks[upstream.name]
            )
            upstream_mean = (
                float(np.nanmean(upstream_values)) if upstream_values.size else np.nan
            )
            for region in regions:
                mask = base_mask & regional_masks[region.name]
                u_values = _masked_values(streamwise, mask)
                w_values = _masked_values(w, mask)
                speed_values = _masked_values(speed, mask)
                mean_streamwise = (
                    float(np.nanmean(u_values)) if u_values.size else np.nan
                )
                u_prime = u_values - mean_streamwise
                w_prime = (
                    w_values - float(np.nanmean(w_values))
                    if w_values.size
                    else np.array([])
                )
                rows.append(
                    {
                        "model": name,
                        "region": region.name,
                        "height_m": float(height),
                        "mean_speed_mps": (
                            float(np.nanmean(speed_values))
                            if speed_values.size
                            else np.nan
                        ),
                        "mean_streamwise_mps": mean_streamwise,
                        "velocity_deficit": (
                            (upstream_mean - mean_streamwise) / abs(upstream_mean)
                            if upstream_mean
                            else np.nan
                        ),
                        "recirculation_fraction": (
                            float(np.mean(u_values < 0.0)) if u_values.size else np.nan
                        ),
                        "w_rms_mps": (
                            float(np.nanstd(w_values)) if w_values.size else np.nan
                        ),
                        "uw_flux_m2ps2": (
                            float(np.nanmean(u_prime * w_prime))
                            if u_prime.size
                            else np.nan
                        ),
                    }
                )

            # First downstream x whose time/y-mean streamwise speed recovers to
            # the requested fraction of the upstream reference speed.
            longitudinal = np.nanmean(
                np.where(base_mask[None, :, :], streamwise, np.nan), axis=(0, 1)
            )
            start = np.flatnonzero(x > geometry.x_max)
            threshold = (1.0 - recovery_deficit) * upstream_mean
            recovered = (
                start[longitudinal[start] >= threshold]
                if start.size and np.isfinite(threshold)
                else []
            )
            rows.append(
                {
                    "model": name,
                    "region": "wake_recovery",
                    "height_m": float(height),
                    "mean_speed_mps": np.nan,
                    "mean_streamwise_mps": np.nan,
                    "velocity_deficit": np.nan,
                    "recirculation_fraction": np.nan,
                    "w_rms_mps": np.nan,
                    "uw_flux_m2ps2": np.nan,
                    "wake_recovery_x_m": (
                        float(x[recovered[0]]) if len(recovered) else np.nan
                    ),
                }
            )
    return rows


def plot_wake_profiles(
    fields: dict[str, xarray.Dataset],
    heights: np.ndarray,
    masks: np.ndarray,
    x: np.ndarray,
    streamwise_angle_deg: float,
    geometry: GeometryInfo,
    output_path: pathlib.Path,
) -> None:
    """Plot mean streamwise velocity recovery along x at all comparison heights."""
    direction = np.array(
        [
            np.cos(np.deg2rad(streamwise_angle_deg)),
            np.sin(np.deg2rad(streamwise_angle_deg)),
        ]
    )
    colors = _model_colors(list(fields))
    fig, axes = plt.subplots(
        len(heights), 1, figsize=(9, 2.7 * len(heights)), sharex=True, squeeze=False
    )
    for height_idx, height in enumerate(heights):
        axis = axes[height_idx, 0]
        for name, field in fields.items():
            streamwise = direction[0] * np.asarray(
                field["u"].isel(height=height_idx).values
            ) + direction[1] * np.asarray(field["v"].isel(height=height_idx).values)
            longitudinal = np.nanmean(
                np.where(masks[height_idx][None, :, :], streamwise, np.nan), axis=(0, 1)
            )
            axis.plot(x, longitudinal, color=colors[name], label=name)
        axis.axvspan(
            geometry.x_min,
            geometry.x_max,
            color="0.6",
            alpha=0.25,
            label="building extent",
        )
        axis.set(ylabel=f"z={height:.1f} m\nU∥ [m/s]")
        axis.grid(True, alpha=0.3)
        if height_idx == 0:
            axis.legend(fontsize=8)
    axes[-1, 0].set_xlabel("x [m]")
    fig.suptitle("Time- and crosswind-mean streamwise wake recovery")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def write_summary(rows: list[dict], path: pathlib.Path) -> None:
    if not rows:
        return
    # Union of keys (in first-seen order): a backend that skipped a metric must
    # not truncate the table.
    fieldnames = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, restval="")
        writer.writeheader()
        writer.writerows(rows)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def _run_scenario(
    cfg: DictConfig,
    scenario_name: str,
    params_list: list[xarray.Dataset],
    model_keys: list[str],
    reference: str,
    out_dir: pathlib.Path,
) -> tuple[list[dict], dict[str, xarray.Dataset]]:
    """Run and analyse one forcing scenario across every selected backend."""
    states: dict[str, xarray.Dataset] = {}
    runtimes: dict[str, float] = {}
    for key in model_keys:
        print(f"--- running {key} / {scenario_name} ---")
        states[key], runtimes[key] = run_one_model(cfg, key, params_list, scenario_name)
        print(
            f"{key}: {runtimes[key]:.2f} s  dims={dict(states[key].sizes)}  "
            f"vars={list(states[key].data_vars)}"
        )

    out_dir.mkdir(parents=True, exist_ok=True)

    if cfg.compare.save_states:
        for key, state in states.items():
            model_dir = out_dir / key
            model_dir.mkdir(parents=True, exist_ok=True)
            state.to_netcdf(model_dir / "state.nc")
            print(f"Saved state -> {model_dir / 'state.nc'}")

    if cfg.run.skip_viz:
        rows = [
            {
                "model": key,
                "is_reference": int(key == reference),
                "runtime_s": round(runtimes[key], 2),
            }
            for key in model_keys
        ]
        write_summary(rows, out_dir / "summary.csv")
        return rows, states

    # Every field figure works off the single-member (ensemble-mean) state
    # regridded onto one common grid and one common time axis.
    single = {k: _ensemble_mean(s) for k, s in states.items()}
    x, y, heights = build_common_grid(cfg)
    times = common_time_axis(single)
    fields = {k: regrid_state(s, x, y, heights, times) for k, s in single.items()}

    params = _ensemble_mean(params_list[0])
    if len(params_list) > 1:
        params = _concat_windows([_ensemble_mean(p) for p in params_list], cfg)
    plot_parameters(params, fields, out_dir / "parameters.png")

    analysis = cfg.compare.analysis
    (_, _), (_, _), (z_min, z_max) = [
        tuple(float(v) for v in pair) for pair in cfg.domain.bounds
    ]
    geometry = load_geometry_info(cfg.geometry.stl_path, x, y, z_min, z_max)
    masks = fluid_masks(geometry, heights, int(analysis.wall_exclusion_cells))
    windows = statistic_windows(
        times,
        float(analysis.statistics_window_seconds),
        int(analysis.max_statistics_windows),
    )
    for statistic in ("mean", "std"):
        plot_windowed_field_statistics(
            fields, heights, masks, windows, statistic, out_dir
        )

    n_snapshots = int(cfg.compare.num_snapshots)
    for idx, height in enumerate(heights):
        plot_state_snapshots(
            fields,
            idx,
            float(height),
            n_snapshots,
            out_dir / f"state_snapshots_z{height:.0f}.png",
        )
        plot_state_difference(
            fields,
            reference,
            idx,
            float(height),
            n_snapshots,
            out_dir / f"state_difference_z{height:.0f}.png",
        )
    field_rmse = plot_field_rmse(fields, reference, heights, out_dir / "field_rmse.png")

    # Full-height statistics use a deliberately coarser grid than the field
    # snapshots. This keeps regional profiles and wake diagnostics inexpensive
    # even when the visual comparison grid is high resolution.
    profile_x = _cell_centres(
        float(cfg.domain.bounds[0][0]),
        float(cfg.domain.bounds[0][1]),
        int(analysis.profile_grid.nx),
    )
    profile_y = _cell_centres(
        float(cfg.domain.bounds[1][0]),
        float(cfg.domain.bounds[1][1]),
        int(analysis.profile_grid.ny),
    )
    profile_heights = _cell_centres(z_min, z_max, int(analysis.profile_num_heights))
    profile_fields = {
        key: regrid_state(state, profile_x, profile_y, profile_heights, times)
        for key, state in single.items()
    }
    profile_geometry = load_geometry_info(
        cfg.geometry.stl_path, profile_x, profile_y, z_min, z_max
    )
    profile_masks = fluid_masks(
        profile_geometry, profile_heights, int(analysis.wall_exclusion_cells)
    )
    regions = auto_regions(profile_geometry, profile_x, profile_y)
    field_regions = auto_regions(geometry, x, y)
    sensor_sets = build_sensor_sets(cfg)
    plot_measurement_overview(
        fields,
        reference,
        heights,
        x,
        y,
        geometry,
        field_regions,
        sensor_sets,
        float(heights[-1]),
        profile_heights,
        int(analysis.wall_exclusion_cells),
        out_dir / "measurement_overview.png",
    )
    plot_vertical_profiles(
        profile_fields,
        profile_heights,
        profile_masks,
        profile_x,
        profile_y,
        regions,
        out_dir,
    )
    # PDFs/CDFs intentionally use the high-resolution visual grid and its
    # independent STL mask, preserving more spatial samples than the profiles.
    for kind in ("pdf", "cdf"):
        plot_field_distributions(
            fields,
            heights,
            masks,
            x,
            y,
            field_regions,
            int(analysis.max_distribution_samples),
            kind,
            out_dir,
        )

    angle = 0.0
    if "inflow_angle" in params:
        angle = float(
            np.nanmean(np.asarray(params["inflow_angle"].values, dtype=float))
        )
    wake_rows = wake_metric_rows(
        profile_fields,
        profile_heights,
        profile_masks,
        profile_x,
        profile_y,
        regions,
        angle,
        profile_geometry,
        float(analysis.wake_recovery_deficit),
    )
    write_summary(wake_rows, out_dir / "wake_metrics.csv")
    plot_wake_profiles(
        profile_fields,
        profile_heights,
        profile_masks,
        profile_x,
        angle,
        profile_geometry,
        out_dir / "wake_profiles.png",
    )

    if cfg.compare.animate:
        anim_path = animate_states(
            fields, heights, out_dir / "state_animation.mp4", int(cfg.compare.fps)
        )
        print(f"Saved animation -> {anim_path}")

    # Sensor series: interpolated at the physical points on each model's own grid
    # (using that backend's solver-specific dim mapping), so no regridding bias
    # enters the comparison.
    # The observation operator never sees the per-frame series: in `intervals`
    # mode it reduces each `interval_seconds` window to one number. Draw that
    # view too, as a sliding window of the same length and reduction.
    interval_seconds = cfg.obs.get("interval_seconds")
    aggregation_mode = str(cfg.obs.get("aggregation_mode") or "mean")
    if interval_seconds is None:
        print(
            "obs.interval_seconds is unset; skipping the sliding-window "
            "sensor figures."
        )

    sensor_rmse: dict[str, dict[str, float]] = {}
    all_sensor_metric_rows: list[dict] = []
    for set_name, points in sensor_sets.items():
        series = {
            key: _sensor_component_timeseries(
                single[key],
                points[0],
                points[1],
                points[2],
                str(cfg.models[key].solver_name),
            )
            for key in model_keys
        }
        plot_sensor_comparison(
            series,
            points,
            f"State at the {set_name} sensors",
            out_dir / f"sensor_timeseries_{set_name}.png",
        )
        if interval_seconds is not None:
            smoothed = {
                key: rolling_aggregate(da, float(interval_seconds), aggregation_mode)
                for key, da in series.items()
            }
            plot_sensor_rolling(
                smoothed,
                series,
                points,
                f"Observed state at the {set_name} sensors "
                f"({aggregation_mode} over a sliding "
                f"{float(interval_seconds):.0f} s window)",
                out_dir / f"sensor_rolling_{set_name}.png",
            )
            metric_series = smoothed
        else:
            metric_series = series
        metrics = sensor_metric_rows(metric_series, reference, set_name)
        all_sensor_metric_rows.extend(metrics)
        plot_sensor_metrics(
            metrics,
            reference,
            set_name,
            out_dir / f"sensor_metrics_{set_name}.png",
        )
        plot_sensor_spectra(
            series,
            set_name,
            float(analysis.spectrum_max_lag_seconds),
            out_dir / f"sensor_spectra_{set_name}.png",
        )
        ref_magnitude = sensor_magnitude(series[reference])
        sensor_rmse[set_name] = {
            key: float(
                np.sqrt(
                    np.nanmean(
                        (
                            sensor_magnitude(series[key]).interp(
                                time=ref_magnitude["time"]
                            )
                            - ref_magnitude
                        ).values
                        ** 2
                    )
                )
            )
            for key in model_keys
        }

    write_summary(all_sensor_metric_rows, out_dir / "sensor_metrics.csv")
    rolling_sensor_rmse = {
        key: float(
            np.nanmean(
                [
                    row["rmse"]
                    for row in all_sensor_metric_rows
                    if row["model"] == key and row["component"] == "|U|"
                ]
            )
        )
        for key in model_keys
    }

    rows = [
        {
            "model": key,
            "is_reference": int(key == reference),
            "runtime_s": round(runtimes[key], 2),
            **{
                f"field_rmse_{height_tag}": round(values[key], 4)
                for height_tag, values in field_rmse.items()
            },
            **{
                f"sensor_rmse_{set_name}": round(values[key], 4)
                for set_name, values in sensor_rmse.items()
            },
            "rolling_sensor_rmse_speed": round(rolling_sensor_rmse[key], 4),
        }
        for key in model_keys
    ]
    write_summary(rows, out_dir / "summary.csv")
    print(f"Saved comparison figures in {out_dir}")
    return rows, states


def _compare_parameter_scenarios_within_model(
    cfg: DictConfig,
    model_name: str,
    scenario_names: list[str],
    reference_scenario: str,
    states_by_scenario: dict[str, dict[str, xarray.Dataset]],
    params_by_scenario: dict[str, xarray.Dataset],
    output_dir: pathlib.Path,
) -> list[dict]:
    """Compare forcing scenarios while holding one solver fixed."""
    output_dir.mkdir(parents=True, exist_ok=True)
    single = {
        scenario: _ensemble_mean(states_by_scenario[scenario][model_name])
        for scenario in scenario_names
    }
    x, y, heights = build_common_grid(cfg)
    times = common_time_axis(single)
    fields = {
        scenario: regrid_state(state, x, y, heights, times)
        for scenario, state in single.items()
    }
    scenario_params = {
        scenario: params_by_scenario[scenario] for scenario in scenario_names
    }
    plot_parameter_scenario_response(
        scenario_params, fields, model_name, output_dir / "parameters.png"
    )

    analysis = cfg.compare.analysis
    (_, _), (_, _), (z_min, z_max) = [
        tuple(float(value) for value in pair) for pair in cfg.domain.bounds
    ]
    geometry = load_geometry_info(cfg.geometry.stl_path, x, y, z_min, z_max)
    masks = fluid_masks(geometry, heights, int(analysis.wall_exclusion_cells))
    windows = statistic_windows(
        times,
        float(analysis.statistics_window_seconds),
        int(analysis.max_statistics_windows),
    )
    for statistic in ("mean", "std"):
        plot_windowed_field_statistics(
            fields, heights, masks, windows, statistic, output_dir
        )
    for index, height in enumerate(heights):
        plot_state_snapshots(
            fields,
            index,
            float(height),
            int(cfg.compare.num_snapshots),
            output_dir / f"state_snapshots_z{height:.0f}.png",
        )
        plot_state_difference(
            fields,
            reference_scenario,
            index,
            float(height),
            int(cfg.compare.num_snapshots),
            output_dir / f"state_difference_z{height:.0f}.png",
        )
    field_rmse = plot_field_rmse(
        fields, reference_scenario, heights, output_dir / "field_rmse.png"
    )

    profile_x = _cell_centres(
        float(cfg.domain.bounds[0][0]),
        float(cfg.domain.bounds[0][1]),
        int(analysis.profile_grid.nx),
    )
    profile_y = _cell_centres(
        float(cfg.domain.bounds[1][0]),
        float(cfg.domain.bounds[1][1]),
        int(analysis.profile_grid.ny),
    )
    profile_heights = _cell_centres(z_min, z_max, int(analysis.profile_num_heights))
    profile_fields = {
        scenario: regrid_state(state, profile_x, profile_y, profile_heights, times)
        for scenario, state in single.items()
    }
    profile_geometry = load_geometry_info(
        cfg.geometry.stl_path, profile_x, profile_y, z_min, z_max
    )
    profile_masks = fluid_masks(
        profile_geometry, profile_heights, int(analysis.wall_exclusion_cells)
    )
    regions = auto_regions(profile_geometry, profile_x, profile_y)
    field_regions = auto_regions(geometry, x, y)
    plot_vertical_profiles(
        profile_fields,
        profile_heights,
        profile_masks,
        profile_x,
        profile_y,
        regions,
        output_dir,
    )
    for kind in ("pdf", "cdf"):
        plot_field_distributions(
            fields,
            heights,
            masks,
            x,
            y,
            field_regions,
            int(analysis.max_distribution_samples),
            kind,
            output_dir,
        )

    reference_params = scenario_params[reference_scenario]
    reference_angle = (
        float(np.nanmean(reference_params["inflow_angle"].values))
        if "inflow_angle" in reference_params
        else 0.0
    )
    wake_rows = wake_metric_rows(
        profile_fields,
        profile_heights,
        profile_masks,
        profile_x,
        profile_y,
        regions,
        reference_angle,
        profile_geometry,
        float(analysis.wake_recovery_deficit),
    )
    for row in wake_rows:
        row["physical_model"] = model_name
    write_summary(wake_rows, output_dir / "wake_metrics.csv")
    plot_wake_profiles(
        profile_fields,
        profile_heights,
        profile_masks,
        profile_x,
        reference_angle,
        profile_geometry,
        output_dir / "wake_profiles.png",
    )

    sensor_sets = build_sensor_sets(cfg)
    interval_seconds = cfg.obs.get("interval_seconds")
    aggregation_mode = str(cfg.obs.get("aggregation_mode") or "mean")
    sensor_rmse: dict[str, dict[str, float]] = {}
    metric_rows: list[dict] = []
    solver_name = str(cfg.models[model_name].solver_name)
    for set_name, points in sensor_sets.items():
        series = {
            scenario: _sensor_component_timeseries(
                single[scenario], points[0], points[1], points[2], solver_name
            )
            for scenario in scenario_names
        }
        plot_sensor_comparison(
            series,
            points,
            f"{model_name}: state at the {set_name} sensors",
            output_dir / f"sensor_timeseries_{set_name}.png",
        )
        if interval_seconds is not None:
            smoothed = {
                scenario: rolling_aggregate(
                    values, float(interval_seconds), aggregation_mode
                )
                for scenario, values in series.items()
            }
            plot_sensor_rolling(
                smoothed,
                series,
                points,
                f"{model_name}: observed state at the {set_name} sensors "
                f"({aggregation_mode} over a sliding {float(interval_seconds):.0f} s window)",
                output_dir / f"sensor_rolling_{set_name}.png",
            )
            comparison_series = smoothed
        else:
            comparison_series = series
        rows = sensor_metric_rows(comparison_series, reference_scenario, set_name)
        for row in rows:
            row["physical_model"] = model_name
        metric_rows.extend(rows)
        plot_sensor_metrics(
            rows,
            reference_scenario,
            set_name,
            output_dir / f"sensor_metrics_{set_name}.png",
        )
        plot_sensor_spectra(
            series,
            set_name,
            float(analysis.spectrum_max_lag_seconds),
            output_dir / f"sensor_spectra_{set_name}.png",
        )
        reference_magnitude = sensor_magnitude(series[reference_scenario])
        sensor_rmse[set_name] = {
            scenario: float(
                np.sqrt(
                    np.nanmean(
                        (
                            sensor_magnitude(series[scenario]).interp(
                                time=reference_magnitude["time"]
                            )
                            - reference_magnitude
                        ).values
                        ** 2
                    )
                )
            )
            for scenario in scenario_names
        }
    write_summary(metric_rows, output_dir / "sensor_metrics.csv")
    rolling_sensor_rmse = {
        scenario: float(
            np.nanmean(
                [
                    row["rmse"]
                    for row in metric_rows
                    if row["model"] == scenario and row["component"] == "|U|"
                ]
            )
        )
        for scenario in scenario_names
    }
    summary_rows = [
        {
            "physical_model": model_name,
            "parameter_scenario": scenario,
            "is_reference_parameter_scenario": int(scenario == reference_scenario),
            **{
                f"field_rmse_{height_tag}": round(values[scenario], 4)
                for height_tag, values in field_rmse.items()
            },
            **{
                f"sensor_rmse_{set_name}": round(values[scenario], 4)
                for set_name, values in sensor_rmse.items()
            },
            "rolling_sensor_rmse_speed": round(rolling_sensor_rmse[scenario], 4),
        }
        for scenario in scenario_names
    ]
    write_summary(summary_rows, output_dir / "summary.csv")
    print(f"Saved within-model parameter comparison -> {output_dir}")
    return summary_rows


def run(cfg: DictConfig) -> None:
    """Run every selected solver against every selected parameter scenario."""
    model_keys = [str(k) for k in cfg.compare.models]
    unknown = [key for key in model_keys if key not in cfg.models]
    if unknown:
        raise ValueError(
            f"compare.models refers to unmounted backends {unknown}; add "
            f"`- model@models.<name>: <name>` to conf/compare_models.yaml "
            f"(mounted: {sorted(cfg.models)})."
        )
    reference = str(cfg.compare.reference or model_keys[0])
    if reference not in model_keys:
        raise ValueError(
            f"compare.reference={reference!r} is not in compare.models {model_keys}."
        )

    scenario_names = [str(name) for name in cfg.compare.parameter_scenarios]
    if not scenario_names:
        raise ValueError(
            "compare.parameter_scenarios must select at least one scenario."
        )
    unknown_scenarios = [
        name for name in scenario_names if name not in cfg.parameter_scenarios
    ]
    if unknown_scenarios:
        raise ValueError(
            "compare.parameter_scenarios refers to unmounted parameter scenarios "
            f"{unknown_scenarios}; mounted: {sorted(cfg.parameter_scenarios)}."
        )

    root_out_dir = resolve_output_dir(cfg, "compare_models") / "comparison"
    root_out_dir.mkdir(parents=True, exist_ok=True)
    params_by_scenario: dict[str, xarray.Dataset] = {}
    states_by_scenario: dict[str, dict[str, xarray.Dataset]] = {}
    summary_rows: list[dict] = []
    for scenario_name in scenario_names:
        # One trajectory per scenario, replayed by every backend. Sampling is
        # deliberately outside the model loop to keep this a controlled test.
        params_list = sample_shared_params(cfg, cfg.parameter_scenarios[scenario_name])
        params = _ensemble_mean(params_list[0])
        if len(params_list) > 1:
            params = _concat_windows(
                [_ensemble_mean(param) for param in params_list], cfg
            )
        params_by_scenario[scenario_name] = params
        rows, states = _run_scenario(
            cfg,
            scenario_name,
            params_list,
            model_keys,
            reference,
            root_out_dir / scenario_name,
        )
        if cfg.compare.within_model_parameter_comparisons and not cfg.run.skip_viz:
            states_by_scenario[scenario_name] = states
        summary_rows.extend({"scenario": scenario_name, **row} for row in rows)

    if not cfg.run.skip_viz:
        plot_scenario_parameters(
            params_by_scenario, root_out_dir / "scenario_parameters.png"
        )
        plot_scenario_error_summary(
            summary_rows, root_out_dir / "scenario_error_summary.png"
        )
        if cfg.compare.within_model_parameter_comparisons and len(scenario_names) > 1:
            parameter_reference = str(
                cfg.compare.parameter_reference or scenario_names[0]
            )
            if parameter_reference not in scenario_names:
                raise ValueError(
                    "compare.parameter_reference must be one of "
                    f"compare.parameter_scenarios; got {parameter_reference!r}."
                )
            within_model_rows: list[dict] = []
            for model_name in model_keys:
                within_model_rows.extend(
                    _compare_parameter_scenarios_within_model(
                        cfg,
                        model_name,
                        scenario_names,
                        parameter_reference,
                        states_by_scenario,
                        params_by_scenario,
                        root_out_dir / "within_model" / model_name,
                    )
                )
            write_summary(within_model_rows, root_out_dir / "within_model_summary.csv")
    write_summary(summary_rows, root_out_dir / "scenario_summary.csv")
    print(f"Saved all scenario comparisons in {root_out_dir}")


@hydra.main(  # type: ignore[misc]
    version_base=None, config_path="../conf", config_name="compare_models"
)
def main(cfg: DictConfig) -> None:
    run(cfg)


if __name__ == "__main__":
    main()
