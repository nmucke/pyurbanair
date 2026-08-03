"""Cheap self-test of the figspec library on the login node (small slices only)."""

from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np

from figspec import dataio, mask, metrics


def main() -> None:
    print("== discover ==")
    a = dataio.discover_block_a()
    b = dataio.discover_block_b()
    for r in a:
        print(
            f"  A {r.name:55s} model={r.model:7s} res={r.res_label:11s} "
            f"dx={r.dx:.1f} complete={r.complete}"
        )
    for r in b:
        print(f"  B {r.name:60s} method={r.method} complete={r.complete}")

    print("\n== truth grid ==")
    g = dataio.truth_grid()
    print("  z", g["z"])
    print("  y[:3]", g["y"][:3], " x[:3]", g["x"][:3])

    print("\n== params (a complete udales A run) ==")
    ud = next(r for r in a if r.model == "udales" and r.complete)
    post = dataio.load_params(ud, "posterior")
    true = dataio.load_params(ud, "true")
    prior = dataio.load_params(ud, "prior")
    for p in dataio.PARAMS:
        print(f"  {p}: {metrics.param_metrics(post, true, p, prior)}")

    print("\n== field interp + rmse (single time) ==")
    sm = dataio.load_state_mean(ud)
    mv = dataio.velmag_field(sm).isel(time=-1)
    mvg = dataio.interp_to_truth(mv)
    tv = dataio.truth_velmag().isel(time=-1)
    print("  model-on-truth shape", mvg.shape, "truth shape", tv.shape)
    # single-slab rmse (no time dim): expand dims
    import xarray as xr

    rmse = metrics.field_rmse(mvg.expand_dims("time"), tv.expand_dims("time"))
    print("  udales field |U| RMSE (final frame, all cells):", rmse)

    print("\n== palm interp 33->16 z ==")
    pl = next((r for r in a if r.model == "palm" and r.complete), None)
    if pl is not None:
        psm = dataio.load_state_mean(pl)
        pv = dataio.velmag_field(psm).isel(time=-1)
        print("  palm raw z size:", pv.sizes.get("z"))
        pvg = dataio.interp_to_truth(pv)
        print(
            "  palm-on-truth shape:",
            pvg.shape,
            "finite frac:",
            float(np.isfinite(pvg.values).mean()),
        )

    print("\n== STL building mask ==")
    m = mask.truth_solid_mask()
    if m is None:
        print("  STL unavailable -> no mask")
    else:
        print(
            "  mask shape",
            m.shape,
            "solid fraction",
            float(m.mean()),
            "per-z solid frac",
            [round(float(m[k].mean()), 3) for k in range(m.shape[0])],
        )

    print("\n== sensors ==")
    cfg = dataio.load_config(ud)
    asx = dataio.sensor_coords(cfg, "assimilation")
    val = dataio.sensor_coords(cfg, "validation")
    print("  assim", None if asx is None else asx.shape, asx)
    print("  valid", None if val is None else val.shape, val)
    if val is not None:
        ts = dataio.sensor_timeseries(
            dataio.interp_to_truth(dataio.velmag_field(sm).isel(time=slice(-3, None))),
            val,
        )
        print("  val-sensor |U| timeseries (last 3 frames):\n", ts)

    print("\nOK")


if __name__ == "__main__":
    main()
