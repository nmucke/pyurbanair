"""Draw the figures of a finished forward run.

    python scripts_new/visualize_forward.py <run dir>

The run dir is what run_forward.py wrote (`config.yaml`, `state.nc`,
`params.nc`). Figures, written into `<run dir>/figures/`:

  field_snapshot.png   |U| at the final time on three heights
  animation.mp4        |U| and vertical vorticity over time on the same heights
  parameters.png       every sampled parameter over time, all members; the
                       inflow angle and speed panels also show the values
                       recovered from the field near the inlet (dotted), as a
                       check that the solver applies the forcing

For an ensemble run the field figures show the ensemble mean.
"""

from __future__ import annotations

import argparse
import pathlib
import sys

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import xarray  # noqa: E402
from evaluation.turbulence import colocate_components  # noqa: E402
from omegaconf import DictConfig, OmegaConf  # noqa: E402

from pyurbanair.utils.animation_utils import animate_height_panels  # noqa: E402

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from helper_functions import sensor_series  # noqa: E402

HEIGHT_FRACTIONS = (0.1, 0.4, 0.7)  # of the domain height, for the field figures


def run(run_dir: pathlib.Path) -> None:
    cfg = OmegaConf.load(run_dir / "config.yaml")
    assert isinstance(cfg, DictConfig)
    solver = cfg.model.solver_name
    out = run_dir / "figures"
    out.mkdir(exist_ok=True)

    state = xarray.open_dataset(run_dir / "state.nc")
    params = xarray.load_dataset(run_dir / "params.nc")
    z_top = float(cfg.domain.bounds[2][1])
    heights = tuple(f * z_top for f in HEIGHT_FRACTIONS)

    mean = _ensemble_mean_centred(state, solver)
    _plot_snapshot(mean, heights, out / "field_snapshot.png")
    animate_height_panels(
        state=mean, output_path=out / "animation.mp4", heights=heights
    )
    _plot_parameters(state, params, cfg, out / "parameters.png")
    state.close()
    print(f"Saved figures in {out}")


def _ensemble_mean_centred(state: xarray.Dataset, solver: str) -> xarray.Dataset:
    """(u, v, w) on the cell centres, averaged over members one at a time."""
    members = state.sizes.get("ensemble", 1)
    total = None
    for m in range(members):
        member = state.isel(ensemble=m) if "ensemble" in state.dims else state
        u, v, w = colocate_components(member[["u", "v", "w"]].load(), solver)
        centred = xarray.Dataset({"u": u, "v": v, "w": w})
        total = centred if total is None else total + centred
    assert total is not None
    return total / members


def _plot_snapshot(
    mean: xarray.Dataset, heights: tuple[float, ...], path: pathlib.Path
) -> None:
    speed = np.sqrt(mean.u**2 + mean.v**2 + mean.w**2).isel(time=-1)
    z_dim, y_dim, x_dim = speed.dims[-3:]
    fig, axes = plt.subplots(1, len(heights), figsize=(4.5 * len(heights), 4))
    for ax, h in zip(axes, heights):
        plane = speed.sel({z_dim: h}, method="nearest")
        im = ax.imshow(
            plane.values,
            origin="lower",
            extent=(
                float(plane[x_dim][0]),
                float(plane[x_dim][-1]),
                float(plane[y_dim][0]),
                float(plane[y_dim][-1]),
            ),
        )
        ax.set_title(
            f"|U|, z = {float(plane[z_dim]):.1f} m, t = {float(speed.time):.0f} s"
        )
        fig.colorbar(im, ax=ax, label="m/s")
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def _plot_parameters(
    state: xarray.Dataset, params: xarray.Dataset, cfg: DictConfig, path: pathlib.Path
) -> None:
    """One panel per parameter: every member's sampled value over time.

    Inflow angle and speed also get the value recovered from (u, v) at three
    probes across the inlet (first cell, mid height), averaged over the probes.
    """
    t_end = float(state.time[-1])
    derived = _inlet_flow(state, cfg)
    names = list(params.data_vars)
    fig, axes = plt.subplots(
        len(names), 1, figsize=(8, 2.6 * len(names)), squeeze=False
    )
    for ax, name in zip(axes[:, 0], names):
        values = params[name]
        if "ensemble" not in values.dims:
            values = values.expand_dims(ensemble=[0])
        for m in range(values.sizes["ensemble"]):
            member = values.isel(ensemble=m)
            color = f"C{m % 10}"
            if "time" in member.dims:
                ax.plot(member.time, member, color=color, lw=1)
            else:
                ax.hlines(float(member), 0, t_end, color=color, lw=1)
            if str(name) in derived:
                series = derived[str(name)]
                series = (
                    series.isel(ensemble=m) if "ensemble" in series.dims else series
                )
                ax.plot(series.time, series, ":", color=color, lw=1)
        ax.set_ylabel(name)
        ax.grid(alpha=0.3)
    axes[-1, 0].set_xlabel("time [s]")
    axes[0, 0].set_title("Sampled parameters (solid); recovered at the inlet (dotted)")
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def _inlet_flow(state: xarray.Dataset, cfg: DictConfig) -> dict[str, xarray.DataArray]:
    """Inflow angle [deg] and speed recovered at the inlet, (([ensemble,] time)."""
    (x0, x1), (y0, y1), (z0, z1) = (tuple(b) for b in cfg.domain.bounds)
    x = x0 + 0.5 * (x1 - x0) / cfg.domain.nx
    ys = [y0 + f * (y1 - y0) for f in (0.2, 0.5, 0.8)]
    points = ([x] * 3, ys, [0.5 * (z0 + z1)] * 3)
    series = sensor_series(state[["u", "v", "w"]], points, cfg.model.solver_name)
    u, v = series.sel(component="u"), series.sel(component="v")
    return {
        "inflow_angle": xarray.apply_ufunc(np.arctan2, v, u).mean("sensor")
        * 180
        / np.pi,
        "velocity_magnitude": ((u**2 + v**2) ** 0.5).mean("sensor"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=pathlib.Path)
    run(parser.parse_args().run_dir)


if __name__ == "__main__":
    main()
