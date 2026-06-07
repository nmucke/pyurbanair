"""Visualize a ground-truth artifact (``state.nc`` + ``params.nc``).

Produces three groups of figures in an output directory:

  1. ``params_*``        - the prescribed parameters stored in ``params.nc``
                           (time-varying ``inflow_angle`` / ``velocity_magnitude``
                           and any scalar parameters such as
                           ``pressure_gradient_magnitude``).
  2. ``field_snapshot_*`` - a mid-z, last-time slice of each state field
                           (u, v, w, pres, ...).
  3. ``derived_*``       - parameters recovered *from* the simulated state near
                           the inlet, overlaid on the prescribed values, so you
                           can check the field actually carries the inflow it
                           was driven with.

The state file can be tens of GB, so it is opened lazily and only the small
slices needed for each figure are ever read into memory.

Examples::

    python scripts/visualize_ground_truth.py
    python scripts/visualize_ground_truth.py ground_truth_spunup/large
    python scripts/visualize_ground_truth.py ground_truth --out-dir figs/gt
"""

from __future__ import annotations

import argparse
import pathlib
import sys

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt
import numpy as np
import xarray

from pyurbanair.utils.run_utils import extract_2d_slice

from scripts._common import (
    plot_derived_inflow_angle,
    plot_derived_velocity_magnitude,
)


def plot_params(params: xarray.Dataset, out_dir: pathlib.Path) -> None:
    """Plot the prescribed parameters stored in ``params.nc``.

    Time-varying parameters (those with a ``time`` dim) get one line plot each;
    scalar parameters are reported in a small text panel.
    """
    time_vars = [n for n in params.data_vars if "time" in params[n].dims]
    scalar_vars = [n for n in params.data_vars if "time" not in params[n].dims]

    n_panels = len(time_vars) + (1 if scalar_vars else 0)
    if n_panels == 0:
        print("params.nc has no plottable variables; skipping.")
        return

    fig, axes = plt.subplots(
        n_panels, 1, figsize=(8, 3 * n_panels), squeeze=False
    )
    axes = axes[:, 0]

    time = np.asarray(params["time"].values) if "time" in params else None
    for ax, name in zip(axes, time_vars):
        ax.plot(time, np.asarray(params[name].values), color="C0")
        ax.set_xlabel("time [s]")
        ax.set_ylabel(name)
        ax.set_title(f"prescribed {name}")
        ax.grid(True, alpha=0.3)

    if scalar_vars:
        ax = axes[len(time_vars)]
        ax.axis("off")
        lines = [
            f"{name} = {float(params[name].values):.6g}" for name in scalar_vars
        ]
        ax.text(
            0.02,
            0.95,
            "scalar parameters\n\n" + "\n".join(lines),
            va="top",
            ha="left",
            family="monospace",
            transform=ax.transAxes,
        )

    fig.tight_layout()
    out = out_dir / "params.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved {out}")


def plot_state_fields(
    state: xarray.Dataset, out_dir: pathlib.Path, z_level: int | None = None
) -> None:
    """Plot a mid-z, last-time horizontal slice of each state field."""
    field_vars = [
        n for n in state.data_vars if state[n].ndim >= 3 and "time" in state[n].dims
    ]
    if not field_vars:
        print("state.nc has no time-varying field variables; skipping.")
        return

    n = len(field_vars)
    ncols = min(n, 2)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(6 * ncols, 5 * nrows), squeeze=False
    )
    flat = axes.ravel()

    for ax, name in zip(flat, field_vars):
        slice_2d = extract_2d_slice(state[name], z_level=z_level)
        im = ax.imshow(slice_2d, origin="lower")
        fig.colorbar(im, ax=ax, label=name)
        ax.set_title(f"{name} (last time, mid z)")
        ax.set_xlabel("x index")
        ax.set_ylabel("y index")

    for ax in flat[n:]:
        ax.axis("off")

    fig.tight_layout()
    out = out_dir / "field_snapshot.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "gt_dir",
        nargs="?",
        default="ground_truth_spunup",
        type=pathlib.Path,
        help="folder containing state.nc and params.nc (default: ground_truth_spunup)",
    )
    parser.add_argument(
        "--out-dir",
        type=pathlib.Path,
        default=None,
        help="where to write figures (default: <gt_dir>/figures)",
    )
    parser.add_argument(
        "--z-level",
        type=int,
        default=None,
        help="z index for field slices (default: mid-domain)",
    )
    args = parser.parse_args()

    gt_dir: pathlib.Path = args.gt_dir
    state_path = gt_dir / "state.nc"
    params_path = gt_dir / "params.nc"
    for p in (state_path, params_path):
        if not p.exists():
            parser.error(f"missing {p}")

    out_dir = args.out_dir or (gt_dir / "figures")
    out_dir.mkdir(parents=True, exist_ok=True)

    params = xarray.open_dataset(params_path)
    # Lazy: the netCDF backend keeps the (possibly huge) state off-memory; only
    # the small slices touched by the plotting helpers are read on access.
    state = xarray.open_dataset(state_path)

    print(f"params.nc: {list(params.data_vars)}")
    print(f"state.nc:  {list(state.data_vars)}")

    plot_params(params, out_dir)
    plot_state_fields(state, out_dir, z_level=args.z_level)

    # Derived-from-state vs. prescribed (needs u and v on the state grid).
    if all(v in state.data_vars for v in ("u", "v")):
        if "inflow_angle" in params:
            plot_derived_inflow_angle(state, params, out_dir)
            print(f"Saved {out_dir / 'derived_inflow_angle.png'}")
        if "velocity_magnitude" in params:
            plot_derived_velocity_magnitude(state, params, out_dir)
            print(f"Saved {out_dir / 'derived_velocity_magnitude.png'}")
    else:
        print("state lacks u/v; skipping derived-parameter plots.")

    print(f"\nAll figures written to {out_dir}")


if __name__ == "__main__":
    main()
