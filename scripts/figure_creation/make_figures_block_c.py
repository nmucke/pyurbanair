"""Block C figures: full-joint (IC+state+param, reduction) showcase (spec §6).

Produces, under ``<out>/C_full_joint/``:
  C1_state_fields_full.png

The single ``IC+state+param (reduction)`` run (``svd_all`` -> method
``icstate_red``) is the "best we can do" result. C1 shows truth vs the full-joint
estimate vs error of |U| at a few times (window-start, mid, end of a window).

This is a field-based (heavy) figure: it loads the posterior-mean rollout +
truth and only runs with ``--heavy`` (use SLURM). The animation ``anim_C_full``
is produced by the separate animations script -- not here.
"""
from __future__ import annotations

import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import matplotlib

matplotlib.use("Agg")
import numpy as np

from figspec import dataio, figcommon as FC, mask
from figspec import style as S

HORIZON_S = 300.0  # Block C overlapping horizon [s] (10 windows x 30 s)


def full_joint_run():
    """The complete IC+state+param (reduction) Block B run (icstate_red), or None."""
    for r in dataio.discover_block_b():
        if r.method == "icstate_red" and r.complete:
            return r
    return None


def _snapshot_times(run, times_arg):
    """Window-start, mid, end times of the last window (or user-specified)."""
    if times_arg:
        return [float(t) for t in times_arg]
    edges = dataio.window_edges(run)
    if edges is None:
        return [0.5 * HORIZON_S, 0.75 * HORIZON_S, HORIZON_S]
    edges = edges[edges <= HORIZON_S + 1e-6]
    a, b = float(edges[-2]), float(edges[-1])
    # window start, mid, near-end of the last window
    return [a, 0.5 * (a + b), a + 0.95 * (b - a)]


# ---------------------------------------------------------------------------
# C1 -- full-joint state fields at a few times
# ---------------------------------------------------------------------------
def c1_state_fields(run, outdir, *, times, solid2d, assim_xy, val_xy):
    """3-row montage over times: truth / full-joint estimate / error (per-column
    truth, so each time's error compares against truth at the SAME time)."""
    sm = dataio.load_state_mean(run)
    if sm is None:
        print("  (C1 skipped: posterior_state_mean.nc missing)")
        return
    model = dataio.interp_to_truth(dataio.velmag_field(sm))
    zi = mask.nearest_z_index(FC.PED_Z)

    truth_fields, est_fields, extent = [], [], None
    for t in times:
        tf, extent = FC.truth_slice(t)
        truth_fields.append(tf)
        da = model.sel(time=t, method="nearest").isel(z=zi)
        est_fields.append(np.asarray(da.values, dtype=float))
    sm.close()

    FC.plot_field_time_montage(
        outdir / "C1_state_fields_full.png", times=times,
        truth_fields=truth_fields, est_fields=est_fields, extent=extent,
        est_label="full-joint", mask2d=solid2d,
        assim_xy=assim_xy, val_xy=val_xy)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=pathlib.Path,
                    default=pathlib.Path(__file__).resolve().parent.parent / "figures")
    ap.add_argument("--heavy", action="store_true",
                    help="produce the field-based C1 figure; use SLURM")
    ap.add_argument("--no-mask", action="store_true", help="do not mask building cells")
    ap.add_argument("--snapshot-time", type=float, nargs="*", default=None,
                    help="time(s) [s] for the C1 grid (default: window start/mid/end)")
    args = ap.parse_args()

    S.apply_style()
    outdir = args.out / "C_full_joint"
    outdir.mkdir(parents=True, exist_ok=True)

    run = full_joint_run()
    if run is None:
        print("Block C: full-joint (icstate_red) run not found / incomplete; nothing to do.")
        return
    print(f"Block C: full-joint run = {run.name}")

    if not args.heavy:
        print("C1 is field-based. Re-run with --heavy (SLURM) to produce it.")
        return

    solid = None if args.no_mask else mask.truth_solid_mask()
    zi = mask.nearest_z_index(FC.PED_Z)
    solid2d = None if solid is None else solid[zi]

    cfg = dataio.load_config(run)
    assim_xy = dataio.sensor_coords(cfg, "assimilation")
    val_xy = dataio.sensor_coords(cfg, "validation")

    times = _snapshot_times(run, args.snapshot_time)
    print(f"  C1 snapshot times: {[round(t, 1) for t in times]} s")
    c1_state_fields(run, outdir, times=times, solid2d=solid2d,
                    assim_xy=assim_xy, val_xy=val_xy)
    print("Block C heavy figures done.")


if __name__ == "__main__":
    main()
