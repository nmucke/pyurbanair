"""Block B figures: IC + parameter / IC+state+parameter ESMDA, by method (spec §5).

Produces, under ``<out>/B_state_estimation/``:
  B1_param_traj_methods.pdf, B2_state_err_vs_time_methods.pdf,
  B2b_valsensor_err_vs_time_methods.pdf, B3_state_fields_methods.png,
  B4_valsensors_methods.pdf, B5_window_prior_post_methods.pdf,
  tables/B_method_comparison.{csv,tex}

Holding the model (uDALES) and truth fixed on the short horizon (10 windows x
30 s = 300 s), this compares ``param-only`` (the finest Block A uDALES run,
restricted to 0..300 s) against the four state-estimation methods.

Cheap (params only) figures always run; the field-based figures (B2/B2b/B3/B4/B5
and the field columns of the method table) load the posterior-mean rollouts +
truth and only run with ``--heavy`` (use SLURM). Per-run field metrics are
cached to ``<out>/_cache/block_b_metrics.json`` for the summary script.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from figspec import dataio, figcommon as FC, mask, metrics
from figspec import style as S

HORIZON_S = 300.0  # Block B overlapping horizon [s] (10 windows x 30 s)


# ---------------------------------------------------------------------------
# Run selection
# ---------------------------------------------------------------------------
def block_b_by_method(runs):
    """{method: Run} for the four state-estimation methods (complete only)."""
    out = {}
    for r in runs:
        if r.method is None or not r.complete:
            continue
        out[r.method] = r
    return out


def param_only_baseline():
    """Finest complete uDALES Block A run (param-only baseline), or None."""
    best = None
    for r in dataio.discover_block_a():
        if r.model != "udales" or not r.complete:
            continue
        if best is None or r.nx > best.nx:
            best = r
    return best


def _ordered_methods(by_method, *, include_baseline=True):
    """Methods present, in canonical METHOD_ORDER (param_only first)."""
    keys = list(by_method)
    if include_baseline and "param_only" in S.METHOD_ORDER:
        keys = keys
    return [m for m in S.METHOD_ORDER if m in by_method]


def _restrict_time(da, t_max=HORIZON_S):
    """Restrict a DataArray with a 'time' dim to t <= t_max (inclusive)."""
    if "time" not in da.dims:
        return da
    return da.sel(time=da["time"] <= t_max + 1e-6)


# ---------------------------------------------------------------------------
# B1 -- parameter trajectories by method (cheap, params only)
# ---------------------------------------------------------------------------
def b1_param_traj(by_method, baseline, outdir):
    """2 stacked panels (alpha, |U|); one posterior-mean line per method +
    param-only baseline + truth (black), restricted to 0..300 s."""
    params = dataio.PARAMS
    # window edges from any method run (all share the same 10x30 s grid)
    ref = next(iter(by_method.values()), baseline)
    if ref is None:
        return
    edges = dataio.window_edges(ref)
    if edges is not None:
        edges = edges[edges <= HORIZON_S + 1e-6]

    fig, axes = plt.subplots(len(params), 1, figsize=(8.4, 2.8 * len(params)),
                             constrained_layout=True, squeeze=False)
    axes = axes[:, 0]

    # series = (method_key, run); param_only baseline first if present
    series = []
    if baseline is not None and baseline.has("posterior_params.nc"):
        series.append(("param_only", baseline))
    for m in _ordered_methods(by_method):
        if by_method[m].has("posterior_params.nc"):
            series.append((m, by_method[m]))

    true_ds = None
    for m, r in series:
        if true_ds is None:
            true_ds = dataio.load_params(r, "true")

    for ax, p in zip(axes, params):
        S.shade_windows(ax, edges)
        for m, r in series:
            post = dataio.load_params(r, "posterior")
            if post is None or p not in post.data_vars:
                if post is not None:
                    post.close()
                continue
            t = np.asarray(post["time"].values, dtype=float)
            keep = t <= HORIZON_S + 1e-6
            pm = np.asarray(post[p].transpose("ensemble", "time").values)
            ax.plot(t[keep], pm.mean(0)[keep], color=S.METHOD_COLORS.get(m),
                    lw=2.0, label=S.METHOD_LABELS.get(m, m))
            post.close()
        if true_ds is not None and p in true_ds.data_vars:
            tt = (np.asarray(true_ds["time"].values, dtype=float)
                  if "time" in true_ds.dims else None)
            tv = np.asarray(true_ds[p].values, dtype=float)
            if tt is not None:
                keep = tt <= HORIZON_S + 1e-6
                ax.plot(tt[keep], tv[keep], color=S.COLORS["truth"], lw=2.0,
                        label="truth", zorder=6)
            else:
                ax.plot([0, HORIZON_S], [tv, tv], color=S.COLORS["truth"],
                        lw=2.0, label="truth", zorder=6)
        S.mark_windows(ax, edges, annotate=False)
        ax.set_ylabel(S.PARAM_LABELS.get(p, p))
        ax.set_xlim(0.0, HORIZON_S)
        ax.legend(loc="upper right", ncol=2, fontsize=7)
        ax.margins(x=0.01)
    axes[-1].set_xlabel("Time [s]")
    if true_ds is not None:
        true_ds.close()
    S.save_pdf(fig, outdir / "B1_param_traj_methods.pdf")


# ---------------------------------------------------------------------------
# Heavy: per-run field |U| series (interp to truth grid, time-aligned)
# ---------------------------------------------------------------------------
def _model_truth_fields(run):
    """Return (model_field, truth_field) standardized to the truth grid and the
    run's (<=300 s) time axis. Caller closes nothing -- the state file is loaded
    lazily and the returned arrays are eager numpy via xarray. Returns None if
    the posterior-mean state is missing."""
    sm = dataio.load_state_mean(run)
    if sm is None:
        return None, None, None
    model = dataio.interp_to_truth(dataio.velmag_field(sm))
    model = _restrict_time(model)
    truth = dataio.align_truth_time(dataio.truth_velmag(), model)
    return model, truth, sm


def compute_field_metrics(run, solid_mask, val_xy):
    """Horizon (0..300 s) |U| field RMSE (fluid cells), normalized, val-sensor."""
    model, truth, sm = _model_truth_fields(run)
    if model is None:
        return None
    rmse = metrics.field_rmse(model, truth, solid_mask)
    nrmse = metrics.normalized_field_rmse(model, truth, solid_mask)
    out = {"field_rmse": rmse, "norm_rmse": nrmse}
    if val_xy is not None:
        mts = dataio.sensor_timeseries(model, val_xy)
        tts = dataio.sensor_timeseries(truth, val_xy)
        out["valsensor_rmse"] = metrics.sensor_rmse(mts, tts)
    sm.close()
    return out


# ---------------------------------------------------------------------------
# B2 -- state |U| field RMSE vs time, by method
# ---------------------------------------------------------------------------
def b2_state_err_vs_time(series, solid_mask, outdir):
    """series: list of (method_key, run). x=time(0..300), y=|U| field RMSE."""
    ref = series[0][1]
    edges = dataio.window_edges(ref)
    if edges is not None:
        edges = edges[edges <= HORIZON_S + 1e-6]
    lines = []
    for m, r in series:
        model, truth, sm = _model_truth_fields(r)
        if model is None:
            continue
        t = np.asarray(model["time"].values, dtype=float)
        y = metrics.field_rmse_timeseries(model, truth, solid_mask)
        lines.append({"time": t, "y": y, "color": S.METHOD_COLORS.get(m),
                      "label": S.METHOD_LABELS.get(m, m)})
        sm.close()
    if not lines:
        return
    FC.plot_lines_vs_time(outdir / "B2_state_err_vs_time_methods.pdf", lines,
                          edges=edges, ylabel=r"$|U|$ field RMSE [m/s]")


def b2b_valsensor_err_vs_time(series, val_xy, outdir):
    """Per-time validation-sensor |U| RMSE (across the validation sensors)."""
    if val_xy is None:
        print("  (B2b skipped: no validation sensors)")
        return
    ref = series[0][1]
    edges = dataio.window_edges(ref)
    if edges is not None:
        edges = edges[edges <= HORIZON_S + 1e-6]
    lines = []
    for m, r in series:
        model, truth, sm = _model_truth_fields(r)
        if model is None:
            continue
        t = np.asarray(model["time"].values, dtype=float)
        mts = dataio.sensor_timeseries(model, val_xy)   # (n_sensor, n_time)
        tts = dataio.sensor_timeseries(truth, val_xy)
        y = np.sqrt(np.nanmean((mts - tts) ** 2, axis=0))  # RMSE across sensors
        lines.append({"time": t, "y": y, "color": S.METHOD_COLORS.get(m),
                      "label": S.METHOD_LABELS.get(m, m)})
        sm.close()
    if not lines:
        return
    FC.plot_lines_vs_time(outdir / "B2b_valsensor_err_vs_time_methods.pdf",
                          lines, edges=edges,
                          ylabel=r"val-sensor $|U|$ RMSE [m/s]")


# ---------------------------------------------------------------------------
# B3 -- state field snapshots by method
# ---------------------------------------------------------------------------
def b3_state_fields(series, outdir, *, time_s, solid2d, assim_xy, val_xy):
    """Columns = truth + methods (METHOD_ORDER); rows = field + error."""
    truth_field, extent = FC.truth_slice(time_s)
    cols = [("truth", truth_field)]
    for m, r in series:
        res = FC.model_slice(r, time_s)
        if res is not None:
            cols.append((S.METHOD_LABELS.get(m, m), res[0]))
    FC.plot_field_error_grid(
        outdir / "B3_state_fields_methods.png", columns=cols,
        truth_field=truth_field, extent=extent, mask2d=solid2d,
        assim_xy=assim_xy, val_xy=val_xy)


# ---------------------------------------------------------------------------
# B4 -- validation-sensor time series by method
# ---------------------------------------------------------------------------
def b4_valsensors(series, val_xy, outdir):
    """Per validation sensor: truth (black) + each method line for |U|."""
    if val_xy is None:
        print("  (B4 skipped: no validation sensors)")
        return
    ref = series[0][1]
    edges = dataio.window_edges(ref)
    if edges is not None:
        edges = edges[edges <= HORIZON_S + 1e-6]

    truth_ts = None
    time = None
    method_series = []
    for m, r in series:
        model, truth, sm = _model_truth_fields(r)
        if model is None:
            continue
        if truth_ts is None:
            time = np.asarray(model["time"].values, dtype=float)
            truth_ts = dataio.sensor_timeseries(truth, val_xy)
        mts = dataio.sensor_timeseries(model, val_xy)
        method_series.append({"label": S.METHOD_LABELS.get(m, m),
                              "color": S.METHOD_COLORS.get(m), "ts": mts})
        sm.close()
    if truth_ts is None:
        return
    FC.plot_sensor_timeseries(
        outdir / "B4_valsensors_methods.pdf", time=time, truth_ts=truth_ts,
        series=method_series, sensor_xyz=val_xy, edges=edges)


# ---------------------------------------------------------------------------
# B5 -- prior vs posterior state error per window (best effort)
# ---------------------------------------------------------------------------
def b5_window_prior_post(series, baseline, solid_mask, outdir):
    """Per-window state |U| RMSE 'before' vs 'after' per method.

    Per-window ensemble files are not loaded here (they are ~2.4 GB each), so
    we approximate the per-window state error from the posterior-mean rollout:
    'after' = field RMSE at the window END (post-update, end of window), and
    'before' = the param-only baseline's field RMSE at the same window end (the
    state error one would have *without* the IC/state update). Best-effort and
    clearly labelled.
    """
    ref = series[0][1]
    edges = dataio.window_edges(ref)
    if edges is None:
        print("  (B5 skipped: no window edges)")
        return
    edges = edges[edges <= HORIZON_S + 1e-6]
    win_ends = edges[1:]

    # baseline ("before" = without IC/state update) field RMSE timeseries
    base_rmse_at = None
    if baseline is not None:
        bmodel, btruth, bsm = _model_truth_fields(baseline)
        if bmodel is not None:
            bt = np.asarray(bmodel["time"].values, dtype=float)
            by = metrics.field_rmse_timeseries(bmodel, btruth, solid_mask)
            base_rmse_at = lambda te: float(np.interp(te, bt, by))
            bsm.close()

    methods = [m for m, _ in series if m != "param_only"]
    fig, ax = plt.subplots(figsize=(8.6, 4.2), constrained_layout=True)
    nwin = len(win_ends)
    xw = np.arange(nwin)
    width = 0.8 / max(1, len(methods))

    if base_rmse_at is not None:
        before = [base_rmse_at(te) for te in win_ends]
        ax.plot(xw, before, "o--", color=S.METHOD_COLORS["param_only"],
                label="before (param-only)", zorder=5)

    for k, m in enumerate(methods):
        run = dict(series)[m]
        model, truth, sm = _model_truth_fields(run)
        if model is None:
            continue
        t = np.asarray(model["time"].values, dtype=float)
        y = metrics.field_rmse_timeseries(model, truth, solid_mask)
        after = [float(np.interp(te, t, y)) for te in win_ends]
        ax.bar(xw + (k - (len(methods) - 1) / 2) * width, after, width,
               color=S.METHOD_COLORS.get(m), label=f"after: {S.METHOD_LABELS.get(m, m)}",
               alpha=0.85)
        sm.close()

    ax.set_xticks(xw)
    ax.set_xticklabels([f"W{i}" for i in range(nwin)])
    ax.set_xlabel("Assimilation window (state at window end)")
    ax.set_ylabel(r"$|U|$ field RMSE [m/s]")
    ax.set_ylim(bottom=0.0)
    ax.legend(loc="best", ncol=2, fontsize=7)
    S.save_pdf(fig, outdir / "B5_window_prior_post_methods.pdf")


# ---------------------------------------------------------------------------
# Table B-I -- method comparison
# ---------------------------------------------------------------------------
def _r(v, nd=3):
    return None if v is None else round(float(v), nd)


def _state_dim_label(run):
    """Descriptive 'updated state dim' for a method (no single fixed number on
    disk; reduction methods adapt the rank online)."""
    if run.method == "param_only":
        return "0 (params only)"
    cfg = dataio.load_config(run)
    dom = cfg.get("domain", {}) or {}
    nx, ny, nz = dom.get("nx"), dom.get("ny"), dom.get("nz")
    full = None
    if nx and ny and nz:
        full = nx * ny * nz * 3  # u,v,w
    red = (cfg.get("esmda", {}) or {}).get("state_reduction")
    if red:
        ef = red.get("energy_fraction")
        return f"reduced (SVD, {ef:g} energy)" if ef is not None else "reduced (SVD)"
    if run.method == "icstate_red":
        return "reduced (SVD)"
    return f"{full} (IC field)" if full else "IC field"


def table_method_comparison(series, field_metrics, baseline, outdir):
    """Rows = methods (param_only + 4). Field columns require --heavy (cached)."""
    rows = []
    for m, r in series:
        s = dataio.load_summary(r)
        pm = s.get("parameter_metrics", {}) or {}
        timing = s.get("timing", {}) or {}
        cfg_s = s.get("configuration", {}) or {}

        # parameter RMSE recomputed over 0..300 s for fairness (param_only too).
        post = dataio.load_params(r, "posterior")
        true = dataio.load_params(r, "true")
        prior = dataio.load_params(r, "prior")
        rmse_a = rmse_u = None
        cov = None
        if post is not None and true is not None:
            post300 = _restrict_time(post)
            cov_d = metrics.param_metrics(post300, true, "inflow_angle", prior)
            mu_d = metrics.param_metrics(post300, true, "velocity_magnitude", prior)
            rmse_a = cov_d.get("rmse")
            rmse_u = mu_d.get("rmse")
            cov = cov_d.get("coverage1")
        for ds in (post, true, prior):
            if ds is not None:
                ds.close()
        # fall back to stored summary RMSE if recompute unavailable
        if rmse_a is None:
            rmse_a = pm.get("inflow_angle", {}).get("rmse", {}).get("mean")
        if rmse_u is None:
            rmse_u = pm.get("velocity_magnitude", {}).get("rmse", {}).get("mean")

        fm = field_metrics.get(r.name, {})
        n_e = cfg_s.get("ensemble_size") or r.ens
        wtw = timing.get("mean_window_seconds")

        rows.append([
            S.METHOD_LABELS.get(m, m),
            _r(rmse_a, 2), _r(rmse_u, 3),
            _r(fm.get("field_rmse")), _r(fm.get("valsensor_rmse")),
            _state_dim_label(r), int(n_e),
            _r(wtw, 1),
            _r(cov, 2),
        ])
    S.write_table(
        outdir / "tables" / "B_method_comparison",
        ["Method", "RMSE alpha [deg]", "RMSE |U| [m/s]", "|U| field RMSE",
         "val-sensor |U| RMSE", "updated state dim", "N_e",
         "wall-time/window [s]", "alpha cov +/-1sig"], rows,
        bold_min_cols=(1, 2, 3, 4))


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=pathlib.Path,
                    default=pathlib.Path(__file__).resolve().parent.parent / "figures")
    ap.add_argument("--heavy", action="store_true",
                    help="also produce field-based figures (B2/B2b/B3/B4/B5 + "
                         "field table columns); use SLURM")
    ap.add_argument("--no-mask", action="store_true", help="do not mask building cells")
    ap.add_argument("--snapshot-time", type=float, default=None,
                    help="time [s] for B3 field grid (default: near end of last window)")
    args = ap.parse_args()

    S.apply_style()
    outdir = args.out / "B_state_estimation"
    outdir.mkdir(parents=True, exist_ok=True)
    cache_dir = args.out / "_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    runs = dataio.discover_block_b()
    by_method = block_b_by_method(runs)
    baseline = param_only_baseline()
    methods = _ordered_methods(by_method)
    print(f"Block B: {len(by_method)} method runs "
          f"({[m for m in methods]}); baseline="
          f"{baseline.name if baseline else 'MISSING'}")

    # ordered (method, run) series including the param-only baseline first
    series = []
    if baseline is not None:
        series.append(("param_only", baseline))
    series += [(m, by_method[m]) for m in methods]

    # --- cheap param figures ------------------------------------------------
    b1_param_traj(by_method, baseline, outdir)

    if not args.heavy:
        # still emit the table with cheap (param + cost) columns; field columns
        # remain blank until --heavy fills the cache.
        table_method_comparison(series, {}, baseline, outdir)
        print("Cheap figures done. Re-run with --heavy (SLURM) for field figures.")
        return

    # --- heavy field figures ------------------------------------------------
    solid = None if args.no_mask else mask.truth_solid_mask()
    zi = mask.nearest_z_index(FC.PED_Z)
    solid2d = None if solid is None else solid[zi]

    # validation sensors from a method run's config (all share the same obs).
    cfg = dataio.load_config(series[-1][1])
    val_xy = dataio.sensor_coords(cfg, "validation")
    assim_xy = dataio.sensor_coords(cfg, "assimilation")

    # per-run field metrics (cached for the summary script)
    field_metrics = {}
    for m, r in series:
        rcfg = dataio.load_config(r)
        rval = dataio.sensor_coords(rcfg, "validation")
        fm = compute_field_metrics(r, solid, rval)
        if fm is None:
            continue
        fm["method"] = m
        field_metrics[r.name] = fm
        print(f"  field metrics {r.name} [{m}]: {fm['field_rmse']:.3f}")
    with open(cache_dir / "block_b_metrics.json", "w") as f:
        json.dump(field_metrics, f, indent=1)

    # snapshot time near the end of the last Block-B window (300 s horizon)
    snap = args.snapshot_time
    if snap is None:
        edges = dataio.window_edges(series[-1][1])
        if edges is not None:
            edges = edges[edges <= HORIZON_S + 1e-6]
            # 90% into the last window (near its end, where methods differ most)
            snap = float(edges[-2] + 0.9 * (edges[-1] - edges[-2]))
        else:
            snap = HORIZON_S
    print(f"  B3 snapshot time: {snap:.1f} s (near end of last window)")

    b2_state_err_vs_time(series, solid, outdir)
    b2b_valsensor_err_vs_time(series, val_xy, outdir)
    b3_state_fields(series, outdir, time_s=snap, solid2d=solid2d,
                    assim_xy=assim_xy, val_xy=val_xy)
    b4_valsensors(series, val_xy, outdir)
    b5_window_prior_post(series, baseline, solid, outdir)
    table_method_comparison(series, field_metrics, baseline, outdir)
    print("Block B heavy figures done.")


if __name__ == "__main__":
    main()
