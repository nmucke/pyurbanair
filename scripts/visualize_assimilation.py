"""Draw the figures of a finished assimilation run.

    python scripts/visualize_assimilation.py <run dir>

The run dir is what an assimilation script wrote (e.g.
`<paths.results_dir>/smoother`). Figures, written into `<run dir>/figures/`:

  parameter_evolution.png   parameter trajectories (prior, posterior, truth) and
                            the ensemble-mean |U| RMSE over time
  animation.mp4             truth, ensemble mean, spread and error of |U| over time
  final_state.png           the same at the final time, with the sensors marked
  mean_slices.png           time-mean streamwise velocity at a few heights:
                            truth | prior | posterior | posterior - truth
  tke_slices.png            the same for the resolved TKE
  station_profiles.png      time-mean u and TKE profiles at the sensor columns
  sensor_timeseries_<set>.png  truth vs ensemble |U| at each sensor set
  tke_evolution.png         rolling TKE at the sensors, truth vs ensemble
  rank_histogram.png        rank of the truth in the ensemble, from the window
                            statistics in metrics.yaml (run compute_metrics.py first)

The prior columns/bands are drawn only when the prior states were saved. The
truth and the ensemble are assumed to be on the same grid (same backend).

Each window file is read once, one member at a time; the truth one window at a
time.
"""

from __future__ import annotations

import pyurbanair.quiet_jax  # noqa: F401  (silences JAX CPU-fallback noise)

import argparse
import pathlib
import sys

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import xarray  # noqa: E402
import yaml  # noqa: E402
from evaluation.figures import (  # noqa: E402
    plot_final_state_with_obs,
    plot_rank_histogram,
    plot_rollout_time_evolution,
    plot_sensor_timeseries,
    plot_tke_time_evolution,
)
from evaluation.sensors import sensor_magnitude  # noqa: E402
from evaluation.turbulence import (  # noqa: E402
    colocate_components,
    select_z_plane,
    sensor_tke_evolution,
    streaming_state_rmse,
)
from omegaconf import DictConfig, OmegaConf  # noqa: E402

from pyurbanair.utils.animation_utils import animate_rollout_state  # noqa: E402
from pyurbanair.utils.run_utils import add_velocity_magnitude  # noqa: E402

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent / "utils"))

from helper_functions import (  # noqa: E402
    case_stl_path,
    concat_windows,
    global_time,
    open_truth,
    sensor_series,
    sensor_sets,
)

NUM_LEVELS = 3  # heights drawn in the mean/TKE slices
MAX_STATIONS = 6  # sensor columns drawn in the profiles


def run(run_dir: pathlib.Path) -> None:
    cfg = OmegaConf.load(run_dir / "config.yaml")
    assert isinstance(cfg, DictConfig)
    num_windows = int(cfg.assimilation.num_windows)
    sim_time = float(cfg.time.simulation_time)
    # Truth frames are matched to the ensemble's by time, within half a frame.
    tolerance = 0.5 * float(cfg.time.output_frequency)
    out = run_dir / "figures"
    out.mkdir(exist_ok=True)

    def window_files(name: str) -> list[pathlib.Path]:
        return [
            run_dir / "windows" / f"window_{w}_{name}.nc" for w in range(num_windows)
        ]

    sets = sensor_sets(cfg)
    prior_files = window_files("prior_state")
    has_prior = all(f.exists() for f in prior_files)

    # --- One pass over truth and ensembles, window by window ---------------------
    truth = open_truth(cfg, run_dir)
    stl_path = case_stl_path(cfg)
    frames = truth.sizes["time"] // num_windows
    stations = _station_points(sets)
    truth_c = Collector(cfg.truth_model.solver_name, sets, stations)
    post_c = Collector(cfg.assim_model.solver_name, sets, stations)
    prior_c = Collector(cfg.assim_model.solver_name, sets, stations)
    rmse = []
    for w in range(num_windows):
        window_truth = truth.isel(time=slice(w * frames, (w + 1) * frames))
        truth_c.add_window(window_truth, None)
        with xarray.open_dataset(window_files("posterior_state")[w]) as ds:
            mean_state = post_c.add_window(ds, global_time(ds.time, w, sim_time))
        rmse.append(
            streaming_state_rmse(
                window_truth.sel(
                    time=mean_state.time, method="nearest", tolerance=tolerance
                ),
                mean_state,
                stl_path,
            )
        )
        if has_prior:
            with xarray.open_dataset(prior_files[w]) as ds:
                prior_c.add_window(ds, global_time(ds.time, w, sim_time))
    truth.close()

    # --- Parameters ----------------------------------------------------------------
    posterior_params = concat_windows(window_files("posterior_params"), sim_time)
    prior_params = concat_windows(window_files("prior_params"), sim_time)
    dynamic = "time" in posterior_params.dims
    plot_rollout_time_evolution(
        esmda_params=posterior_params,
        true_params=xarray.load_dataset(run_dir / "true_params.nc"),
        esmda_state=None,
        true_state=None,
        output_path=out / "parameter_evolution.png",
        prior_params=prior_params,
        window_edges=(
            list(np.linspace(0, num_windows * sim_time, num_windows + 1))
            if dynamic and num_windows > 1
            else None
        ),
        rmse=np.concatenate(rmse),
    )

    # --- |U| fields --------------------------------------------------------------------
    truth_plane = xarray.concat(truth_c.planes, dim="time")
    mean_vel, std_vel = post_c.plane_mean_std()
    animate_rollout_state(
        true_state=truth_plane,
        mean_vel=mean_vel,
        std_vel=std_vel,
        output_path=out / "animation.mp4",
        z_level=0,
    )
    plot_final_state_with_obs(
        mean_vel=mean_vel,
        std_vel=std_vel,
        output_path=out / "final_state.png",
        true_vel=add_velocity_magnitude(truth_plane)["vel_magnitude"],
        obs_x=np.asarray(cfg.obs.x_points),
        obs_y=np.asarray(cfg.obs.y_points),
        z_level=0,
    )

    # --- Time-mean and TKE fields ---------------------------------------------------
    columns = {"truth": truth_c, "posterior": post_c}
    if has_prior:
        columns = {"truth": truth_c, "prior": prior_c, "posterior": post_c}
    _plot_slices(columns, "mean_u", "time-mean u [m/s]", out / "mean_slices.png")
    _plot_slices(columns, "tke", "resolved TKE [m²/s²]", out / "tke_slices.png")
    _plot_profiles(columns, stations, out / "station_profiles.png")

    # --- Sensors -------------------------------------------------------------------------
    tke_series = {}
    for name, (sx, sy, sz) in sets.items():
        truth_s = truth_c.sensor_series(name)
        post_s = post_c.sensor_series(name)
        plot_sensor_timeseries(
            true_sensor=sensor_magnitude(truth_s),
            ensemble_sensor=sensor_magnitude(post_s),
            output_path=out / f"sensor_timeseries_{name}.png",
            title=f"|U| at the {name} sensors",
            sensor_x=np.asarray(sx),
            sensor_y=np.asarray(sy),
            sensor_z=np.asarray(sz),
        )
        tke = sensor_tke_evolution(truth_s, post_s, window_seconds=sim_time / 8.0)
        if tke is not None:
            tke_series[name] = tke
    plot_tke_time_evolution(tke_series, out / "tke_evolution.png")

    # --- Rank histogram (from compute_metrics.py) -------------------------------------
    metrics_path = run_dir / "metrics.yaml"
    if metrics_path.exists():
        with open(metrics_path) as f:
            statistics = yaml.safe_load(f).get("sensor_statistics", {})
        counts = {
            name: {
                half: {
                    stat: entry["rank_counts"]
                    for stat, entry in block[half].items()
                    if isinstance(entry, dict) and entry.get("rank_counts")
                }
                for half in ("prior", "posterior")
                if half in block
            }
            for name, block in statistics.items()
        }
        plot_rank_histogram(counts, out / "rank_histogram.png")
    else:
        print("No metrics.yaml: skipping rank_histogram.png (run compute_metrics.py)")

    print(f"Saved figures in {out}")


class Collector:
    """Everything the figures need from one source (truth, prior or posterior).

    Fed one window at a time and, within it, one member at a time:
      * the (u, v, w) series at every sensor set,
      * |U| on the lowest z-plane: kept per frame for the truth, as ensemble
        mean and std for an ensemble,
      * per member, time sums of u, v, w and their squares on a few heights and
        at the sensor columns (cell-centred), for the time-mean and TKE figures.
    """

    def __init__(self, solver_name: str, sets: dict, stations: np.ndarray) -> None:
        self.solver_name = solver_name
        self.sets = sets
        self.stations = stations
        self.series: dict[str, list] = {name: [] for name in sets}
        self.planes: list[xarray.Dataset] = []
        self.plane_sum: list[xarray.DataArray] = []
        self.plane_sumsq: list[xarray.DataArray] = []
        self.n_members = 0
        self.n_frames = 0
        self.sums: dict[str, np.ndarray] = {}  # (member, component, ...) arrays
        self.levels: np.ndarray | None = None
        self.grid: dict[str, np.ndarray] = {}
        self.columns: tuple[np.ndarray, np.ndarray] = (np.array([]), np.array([]))
        self.is_ensemble = True

    def add_window(self, ds: xarray.Dataset, time: np.ndarray | None) -> xarray.Dataset:
        """Add one window; return its ensemble-mean (u, v, w) state.

        `time` is the window's global time axis (see `global_time`); None keeps
        the file's own.
        """
        ds = ds[["u", "v", "w"]]
        if time is not None:
            ds = ds.assign_coords(time=time)
        self.is_ensemble = "ensemble" in ds.dims
        n_members = ds.sizes.get("ensemble", 1)
        self.n_members = n_members
        series: dict[str, list] = {name: [] for name in self.sets}
        total = None
        plane_sum = plane_sumsq = None
        for m in range(n_members):
            member = (ds.isel(ensemble=m) if "ensemble" in ds.dims else ds).load()
            for name, points in self.sets.items():
                series[name].append(sensor_series(member, points, self.solver_name))
            total = member if total is None else total + member
            plane = select_z_plane(member, z_level=0)
            vel = add_velocity_magnitude(plane)["vel_magnitude"]
            plane_sum = vel if plane_sum is None else plane_sum + vel
            plane_sumsq = vel**2 if plane_sumsq is None else plane_sumsq + vel**2
            if not self.is_ensemble:
                self.planes.append(plane)
            self._add_moments(m, member)
        self.n_frames += ds.sizes["time"]
        for name in self.sets:
            stacked = xarray.concat(series[name], dim="ensemble")
            self.series[name].append(stacked.transpose("component", "ensemble", ...))
        assert total is not None and plane_sum is not None and plane_sumsq is not None
        self.plane_sum.append(plane_sum)
        self.plane_sumsq.append(plane_sumsq)
        return total / n_members

    def _add_moments(self, m: int, member: xarray.Dataset) -> None:
        """Accumulate per-member time sums on the slice heights and columns."""
        u, v, w = colocate_components(member, self.solver_name)
        z_dim, y_dim, x_dim = u.dims[-3:]
        if self.levels is None:
            nz = u.sizes[z_dim]
            # Skip the top cell: its centre is extrapolated by the colocation.
            self.levels = np.unique(
                np.linspace(0, max(nz - 2, 0), NUM_LEVELS).astype(int)
            )
            self.grid = {
                "x": u[x_dim].values,
                "y": u[y_dim].values,
                "z": u[z_dim].values,
            }
            ix = np.abs(self.grid["x"][None, :] - self.stations[:, :1]).argmin(1)
            iy = np.abs(self.grid["y"][None, :] - self.stations[:, 1:]).argmin(1)
            self.columns = (iy, ix)
        comps = np.stack([np.asarray(c.values) for c in (u, v, w)])  # (3, t, z, y, x)
        slabs = comps[:, :, self.levels]
        cols = comps[:, :, :, self.columns[0], self.columns[1]]  # (3, t, z, station)
        for key, values in (("slab", slabs), ("col", cols)):
            for power in (1, 2):
                name = f"{key}{power}"
                total = (values**power).sum(axis=1)
                if name not in self.sums:
                    self.sums[name] = np.zeros((self.n_members, *total.shape))
                self.sums[name][m] += total

    def moments(self, key: str) -> dict[str, np.ndarray]:
        """Per-member time-mean u and TKE, (member, ...) arrays."""
        mean = self.sums[f"{key}1"] / self.n_frames
        var = self.sums[f"{key}2"] / self.n_frames - mean**2
        return {"mean_u": mean[:, 0], "tke": 0.5 * var.sum(axis=1)}

    def plane_mean_std(self) -> tuple[xarray.DataArray, xarray.DataArray]:
        mean = xarray.concat(self.plane_sum, dim="time") / self.n_members
        sumsq = xarray.concat(self.plane_sumsq, dim="time") / self.n_members
        return mean, ((sumsq - mean**2).clip(min=0.0)) ** 0.5

    def sensor_series(self, name: str) -> xarray.DataArray:
        series: xarray.DataArray = xarray.concat(self.series[name], dim="time")
        return series if self.is_ensemble else series.isel(ensemble=0)


def _station_points(sets: dict) -> np.ndarray:
    """Unique (x, y) sensor columns over all sensor sets, at most MAX_STATIONS."""
    xy = np.concatenate(
        [np.column_stack([np.asarray(x), np.asarray(y)]) for x, y, _ in sets.values()]
    )
    return np.unique(xy, axis=0)[:MAX_STATIONS]


def _plot_slices(
    columns: dict[str, Collector], quantity: str, label: str, path: pathlib.Path
) -> None:
    """Rows: heights. Columns: each source's ensemble mean, then posterior - truth."""
    fields = {k: c.moments("slab")[quantity].mean(axis=0) for k, c in columns.items()}
    diff = fields["posterior"] - fields["truth"]
    grid = columns["truth"].grid
    levels = columns["truth"].levels
    assert levels is not None
    vmin = float(np.nanmin([np.nanmin(f) for f in fields.values()]))
    vmax = float(np.nanmax([np.nanmax(f) for f in fields.values()]))
    dmax = np.nanmax(np.abs(diff))
    n_cols = len(fields) + 1
    fig, axes = plt.subplots(
        len(levels), n_cols, figsize=(3.2 * n_cols, 2.8 * len(levels)), squeeze=False
    )
    extent = (grid["x"][0], grid["x"][-1], grid["y"][0], grid["y"][-1])
    for row, level in enumerate(levels):
        for col, (name, field) in enumerate(fields.items()):
            im = axes[row, col].imshow(
                field[row], origin="lower", extent=extent, vmin=vmin, vmax=vmax
            )
            axes[row, col].set_title(f"{name}, z = {grid['z'][level]:.1f} m")
        fig.colorbar(im, ax=axes[row, : n_cols - 1].tolist(), label=label)
        d = axes[row, -1].imshow(
            diff[row],
            origin="lower",
            extent=extent,
            cmap="RdBu_r",
            vmin=-dmax,
            vmax=dmax,
        )
        axes[row, -1].set_title("posterior - truth")
        fig.colorbar(d, ax=axes[row, -1])
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def _plot_profiles(
    columns: dict[str, Collector], stations: np.ndarray, path: pathlib.Path
) -> None:
    """Rows: time-mean u, TKE. Columns: sensor columns. Ensembles as min-max bands."""
    z = columns["truth"].grid["z"]
    colors = {"prior": "grey", "posterior": "tab:blue"}
    fig, axes = plt.subplots(
        2, len(stations), figsize=(2.6 * len(stations), 6), squeeze=False, sharey=True
    )
    for row, (quantity, label) in enumerate(
        (("mean_u", "time-mean u [m/s]"), ("tke", "TKE [m²/s²]"))
    ):
        for name, collector in columns.items():
            values = collector.moments("col")[quantity]  # (member, z, station)
            for s in range(len(stations)):
                ax = axes[row, s]
                if name == "truth":
                    ax.plot(values[0, :, s], z, "k--", label="truth")
                else:
                    ax.fill_betweenx(
                        z,
                        values[:, :, s].min(0),
                        values[:, :, s].max(0),
                        color=colors[name],
                        alpha=0.3,
                    )
                    ax.plot(values[:, :, s].mean(0), z, color=colors[name], label=name)
                ax.set_xlabel(label)
                if row == 0:
                    ax.set_title(f"x={stations[s, 0]:g}, y={stations[s, 1]:g}")
    axes[0, 0].set_ylabel("z [m]")
    axes[1, 0].set_ylabel("z [m]")
    axes[0, 0].legend()
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=pathlib.Path)
    run(parser.parse_args().run_dir)


if __name__ == "__main__":
    main()
