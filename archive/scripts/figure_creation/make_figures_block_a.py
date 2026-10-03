"""Block A figures: parameter-only ESMDA, model x resolution (spec §4).

Produces, under ``<out>/A_param_only/``:
  A1_param_traj_<model>.pdf, A1_param_traj_grid.pdf, A2_param_err_vs_res.pdf,
  A3_state_fields_truth_vs_models.png, A4_valsensors_<model>.pdf,
  A5_state_err_vs_res.pdf, A6_window_prior_post_udales.pdf,
  tables/A_param_accuracy.{csv,tex}, tables/A_state_accuracy.{csv,tex}

Cheap (params only) figures always run; the field-based figures (A3/A4/A5 and the
state-accuracy table) load the posterior-mean rollouts + truth and only run with
``--heavy`` (use SLURM). Per-run field metrics are cached to
``<out>/_cache/block_a_metrics.json`` for the summary script.
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
from evaluation import scores
from evaluation import style as S
from figspec import dataio
from figspec import figcommon as FC
from figspec import mask


def _teal_shades(n):
    import matplotlib

    if n == 1:
        return [S.COLORS["posterior"]]
    cmap = matplotlib.colormaps["winter"]
    return [cmap(0.15 + 0.7 * i / (n - 1)) for i in range(n)]


def finest(runs):
    """Finest (largest nx) complete run per model -> dict model->Run."""
    out = {}
    for r in runs:
        if not r.complete:
            continue
        if r.model not in out or r.nx > out[r.model].nx:
            out[r.model] = r
    return out


# ---------------------------------------------------------------------------
# A1 -- parameter trajectories
# ---------------------------------------------------------------------------
def a1_per_model(runs_by_model, outdir):
    for model, runs in runs_by_model.items():
        runs = sorted(
            [r for r in runs if r.has("posterior_params.nc")], key=lambda r: r.nx
        )
        if not runs:
            continue
        params = dataio.PARAMS
        fig, axes = plt.subplots(
            len(params),
            1,
            figsize=(8.4, 2.8 * len(params)),
            constrained_layout=True,
            squeeze=False,
        )
        axes = axes[:, 0]
        shades = _teal_shades(len(runs))
        finest_run = runs[-1]
        prior = dataio.load_params(finest_run, "prior")
        true = dataio.load_params(finest_run, "true")
        edges = dataio.param_window_edges(
            dataio.load_params(finest_run, "posterior")["time"].values,
            (dataio.load_summary(finest_run).get("configuration", {}) or {}).get(
                "num_assimilation_windows"
            ),
        )
        for ax, p in zip(axes, params):
            S.shade_windows(ax, edges)
            if prior is not None and p in prior.data_vars:
                prm = np.asarray(prior[p].transpose("ensemble", "time").values)
                S.band(
                    ax,
                    prior["time"].values,
                    prm.mean(0),
                    prm.std(0),
                    S.COLORS["prior"],
                    alpha=0.15,
                    label="prior",
                    lw=1.4,
                    ls="--",
                )
            for r, col in zip(runs, shades):
                post = dataio.load_params(r, "posterior")
                if post is None or p not in post.data_vars:
                    continue
                pm = np.asarray(post[p].transpose("ensemble", "time").values)
                ax.plot(
                    post["time"].values,
                    pm.mean(0),
                    color=col,
                    lw=2.0,
                    label=f"post {r.res_label_dx}",
                )
                post.close()
            if true is not None and p in true.data_vars:
                tr_t = (
                    true["time"].values if "time" in true.dims else post["time"].values
                )
                ax.plot(
                    tr_t,
                    np.asarray(true[p].values),
                    color=S.COLORS["truth"],
                    lw=2.0,
                    label="truth",
                    zorder=6,
                )
            S.mark_windows(ax, edges, annotate=False)
            ax.set_ylabel(S.PARAM_LABELS.get(p, p))
            ax.legend(loc="upper right", ncol=2, fontsize=7)
            ax.margins(x=0.01)
        axes[-1].set_xlabel("Time [s]")
        S.save_pdf(fig, outdir / f"A1_param_traj_{model}.pdf")
        for ds in (prior, true):
            if ds is not None:
                ds.close()


def a1_grid(rep_by_model, outdir):
    """rows = {alpha, |U|}, cols = {palm, udales, lbm} at representative res."""
    models = [m for m in ("palm", "udales", "lbm") if m in rep_by_model]
    params = dataio.PARAMS
    fig, axes = plt.subplots(
        len(params),
        len(models),
        figsize=(4.0 * len(models), 2.8 * len(params)),
        constrained_layout=True,
        squeeze=False,
        sharey="row",
    )
    for j, model in enumerate(models):
        r = rep_by_model[model]
        prior = dataio.load_params(r, "prior")
        post = dataio.load_params(r, "posterior")
        true = dataio.load_params(r, "true")
        nwin = (dataio.load_summary(r).get("configuration", {}) or {}).get(
            "num_assimilation_windows"
        )
        edges = dataio.param_window_edges(post["time"].values, nwin)
        for i, p in enumerate(params):
            ax = axes[i, j]
            prm = (
                np.asarray(prior[p].transpose("ensemble", "time").values)
                if prior is not None and p in prior.data_vars
                else None
            )
            pm = np.asarray(post[p].transpose("ensemble", "time").values)
            tr_t = true["time"].values if "time" in true.dims else post["time"].values
            FC.param_panel(
                ax,
                time=post["time"].values,
                post_mean=pm.mean(0),
                post_std=pm.std(0),
                prior_mean=None if prm is None else prm.mean(0),
                prior_std=None if prm is None else prm.std(0),
                truth_t=tr_t,
                truth=np.asarray(true[p].values) if p in true else None,
                edges=edges,
            )
            if i == 0:
                ax.set_title(f"{S.MODEL_LABELS[model]} ({r.res_label_dx})")
            if j == 0:
                ax.set_ylabel(S.PARAM_LABELS.get(p, p))
            if i == len(params) - 1:
                ax.set_xlabel("Time [s]")
            if i == 0 and j == 0:
                ax.legend(loc="upper right", fontsize=7, ncol=2)
        for ds in (prior, post, true):
            if ds is not None:
                ds.close()
    S.save_pdf(fig, outdir / "A1_param_traj_grid.pdf")


# ---------------------------------------------------------------------------
# A2 -- parameter error vs resolution (stored run_summary RMSE)
# ---------------------------------------------------------------------------
def a2_param_err_vs_res(runs, outdir):
    by = {"inflow_angle": {}, "velocity_magnitude": {}}
    for r in runs:
        if not r.has("run_summary.yaml"):
            continue
        pm = dataio.load_summary(r).get("parameter_metrics", {}) or {}
        for p in by:
            if p in pm:
                by[p].setdefault(r.model, ([], []))
                by[p][r.model][0].append(r.dx)
                by[p][r.model][1].append(pm[p].get("rmse", {}).get("mean"))
    panels = [
        {
            "title": "Inflow angle",
            "ylabel": r"RMSE $\alpha$ [deg]",
            "by_model": by["inflow_angle"],
        },
        {
            "title": "Velocity magnitude",
            "ylabel": r"RMSE $|U|$ [m/s]",
            "by_model": by["velocity_magnitude"],
        },
    ]
    FC.plot_vs_resolution(outdir / "A2_param_err_vs_res.pdf", panels, truth_dx=1.0)


# ---------------------------------------------------------------------------
# Parameter accuracy table (A-I)
# ---------------------------------------------------------------------------
def table_param_accuracy(runs, outdir):
    rows = []
    for r in sorted(runs, key=lambda r: (r.model, -r.dx)):
        if not r.has("run_summary.yaml") or not r.has("posterior_params.nc"):
            continue
        pm = dataio.load_summary(r).get("parameter_metrics", {}) or {}
        post = dataio.load_params(r, "posterior")
        true = dataio.load_params(r, "true")
        prior = dataio.load_params(r, "prior")
        cov = {p: scores.param_metrics(post, true, p, prior) for p in dataio.PARAMS}
        rows.append(
            [
                S.MODEL_LABELS[r.model],
                r.res_label_dx,
                pm.get("inflow_angle", {}).get("rmse", {}).get("mean"),
                pm.get("velocity_magnitude", {}).get("rmse", {}).get("mean"),
                _pct(pm.get("inflow_angle", {}).get("rmse_reduction_vs_prior")),
                _pct(pm.get("velocity_magnitude", {}).get("rmse_reduction_vs_prior")),
                round(cov["inflow_angle"].get("coverage1", float("nan")), 2),
                round(cov["velocity_magnitude"].get("coverage1", float("nan")), 2),
            ]
        )
        for ds in (post, true, prior):
            if ds is not None:
                ds.close()
    S.write_table(
        outdir / "tables" / "A_param_accuracy",
        [
            "Model",
            "Resolution",
            "RMSE alpha [deg]",
            "RMSE |U| [m/s]",
            "alpha reduction [%]",
            "|U| reduction [%]",
            "alpha cov +/-1sig",
            "|U| cov +/-1sig",
        ],
        rows,
        bold_min_cols=(2, 3),
        bold_max_cols=(4, 5),
    )


def _pct(v):
    return None if v is None else round(100.0 * float(v), 1)


# ---------------------------------------------------------------------------
# Heavy: per-run field metrics (cached) + A3/A4/A5 + state table
# ---------------------------------------------------------------------------
def compute_field_metrics(run, solid_mask, val_xy):
    """Horizon-mean |U| field RMSE (fluid cells), normalized, and val-sensor RMSE."""
    sm = dataio.load_state_mean(run)
    if sm is None:
        return None
    model = dataio.interp_to_truth(dataio.velmag_field(sm))  # (time,z,y,x)
    truth = dataio.align_truth_time(dataio.truth_velmag(), model)
    rmse = scores.field_rmse(model, truth, solid_mask)
    nrmse = scores.normalized_field_rmse(model, truth, solid_mask)
    out = {"field_rmse": rmse, "norm_rmse": nrmse}
    if val_xy is not None:
        mts = dataio.sensor_timeseries(model, val_xy)
        tts = dataio.sensor_timeseries(truth, val_xy)
        out["valsensor_rmse"] = scores.sensor_rmse(mts, tts)
    sm.close()
    return out


def a3_state_fields(rep_by_model, outdir, *, time_s, solid2d, assim_xy, val_xy):
    cols = [("truth", FC.truth_slice(time_s)[0])]
    extent = FC.truth_slice(time_s)[1]
    for model in ("palm", "udales", "lbm"):
        if model in rep_by_model:
            res = FC.model_slice(rep_by_model[model], time_s)
            if res is not None:
                cols.append(
                    (
                        f"{S.MODEL_LABELS[model]} ({rep_by_model[model].res_label_dx})",
                        res[0],
                    )
                )
    FC.plot_field_error_grid(
        outdir / "A3_state_fields_truth_vs_models.png",
        columns=cols,
        truth_field=cols[0][1],
        extent=extent,
        mask2d=solid2d,
        assim_xy=assim_xy,
        val_xy=val_xy,
    )


def a4_valsensors(rep_by_model, outdir):
    for model, r in rep_by_model.items():
        cfg = dataio.load_config(r)
        val_xy = dataio.sensor_coords(cfg, "validation")
        if val_xy is None:
            continue
        sm = dataio.load_state_mean(r)
        model_field = dataio.interp_to_truth(dataio.velmag_field(sm))
        truth = dataio.align_truth_time(dataio.truth_velmag(), model_field)
        t = np.asarray(model_field["time"].values)
        mts = dataio.sensor_timeseries(model_field, val_xy)
        tts = dataio.sensor_timeseries(truth, val_xy)
        # prior reconstruction unavailable as a field; show posterior vs truth.
        edges = dataio.window_edges(r)
        FC.plot_sensor_timeseries(
            outdir / f"A4_valsensors_{model}.pdf",
            time=t,
            truth_ts=tts,
            series=[
                {
                    "label": f"{S.MODEL_LABELS[model]} posterior",
                    "color": S.COLORS["posterior"],
                    "ts": mts,
                }
            ],
            sensor_xyz=val_xy,
            edges=edges,
        )
        sm.close()


def a5_state_err_vs_res(field_metrics, outdir):
    by = {}
    for key, m in field_metrics.items():
        model, dx = m["model"], m["dx"]
        if m.get("field_rmse") is None:
            continue
        by.setdefault(model, ([], []))
        by[model][0].append(dx)
        by[model][1].append(m["field_rmse"])
    panels = [
        {
            "title": "State reconstruction",
            "ylabel": r"$|U|$ field RMSE [m/s]",
            "by_model": by,
        }
    ]
    # optional second panel: validation-sensor RMSE
    byv = {}
    for key, m in field_metrics.items():
        if m.get("valsensor_rmse") is None:
            continue
        byv.setdefault(m["model"], ([], []))
        byv[m["model"]][0].append(m["dx"])
        byv[m["model"]][1].append(m["valsensor_rmse"])
    if any(byv.values()):
        panels.append(
            {
                "title": "Validation sensors",
                "ylabel": r"val-sensor $|U|$ RMSE [m/s]",
                "by_model": byv,
            }
        )
    FC.plot_vs_resolution(outdir / "A5_state_err_vs_res.pdf", panels, truth_dx=1.0)


def table_state_accuracy(field_metrics, outdir):
    rows = []
    for key, m in sorted(
        field_metrics.items(), key=lambda kv: (kv[1]["model"], -kv[1]["dx"])
    ):
        rows.append(
            [
                S.MODEL_LABELS[m["model"]],
                m["res_label_dx"],
                _r(m.get("field_rmse")),
                _r(m.get("norm_rmse")),
                _r(m.get("valsensor_rmse")),
            ]
        )
    S.write_table(
        outdir / "tables" / "A_state_accuracy",
        ["Model", "Resolution", "|U| field RMSE", "norm. RMSE", "val-sensor |U| RMSE"],
        rows,
        bold_min_cols=(2, 3, 4),
    )


def _r(v, nd=3):
    return None if v is None else round(float(v), nd)


def a6_window_prior_post(run, outdir):
    """Per-window parameter prior vs posterior ensemble (mean +/- sigma)."""
    wdir = run.path / "windows"
    if not wdir.is_dir():
        return
    import re

    wins = sorted(
        {
            int(re.match(r"window_(\d+)_", p.name).group(1))
            for p in wdir.glob("window_*_posterior_params.nc")
        }
    )
    if not wins:
        return
    params = dataio.PARAMS
    fig, axes = plt.subplots(
        len(params),
        1,
        figsize=(8.4, 2.8 * len(params)),
        constrained_layout=True,
        squeeze=False,
    )
    axes = axes[:, 0]
    import xarray as xr

    for ax, p in zip(axes, params):
        prm_means, prm_lo, prm_hi, post_means, post_lo, post_hi = ([] for _ in range(6))
        for w in wins:
            pr = xr.open_dataset(wdir / f"window_{w}_prior_params.nc")
            po = xr.open_dataset(wdir / f"window_{w}_posterior_params.nc")
            if p in pr.data_vars:
                a = np.asarray(pr[p].values).ravel()
                prm_means.append(a.mean())
                prm_lo.append(a.mean() - a.std())
                prm_hi.append(a.mean() + a.std())
            b = np.asarray(po[p].values).ravel()
            post_means.append(b.mean())
            post_lo.append(b.mean() - b.std())
            post_hi.append(b.mean() + b.std())
            pr.close()
            po.close()
        xw = np.arange(len(wins))
        ax.errorbar(
            xw - 0.08,
            prm_means,
            yerr=[np.array(prm_means) - prm_lo, np.array(prm_hi) - prm_means],
            fmt="o",
            color=S.COLORS["prior"],
            label="prior",
            capsize=3,
        )
        ax.errorbar(
            xw + 0.08,
            post_means,
            yerr=[np.array(post_means) - post_lo, np.array(post_hi) - post_means],
            fmt="s",
            color=S.COLORS["posterior"],
            label="posterior",
            capsize=3,
        )
        ax.set_xticks(xw)
        ax.set_xticklabels([f"W{w}" for w in wins])
        ax.set_ylabel(S.PARAM_LABELS.get(p, p))
        ax.legend(loc="best")
    axes[-1].set_xlabel("Assimilation window")
    S.save_pdf(fig, outdir / "A6_window_prior_post_udales.pdf")


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--out",
        type=pathlib.Path,
        default=pathlib.Path(__file__).resolve().parent.parent / "figures",
    )
    ap.add_argument(
        "--heavy",
        action="store_true",
        help="also produce field-based figures (A3/A4/A5 + state table); use SLURM",
    )
    ap.add_argument("--no-mask", action="store_true", help="do not mask building cells")
    ap.add_argument(
        "--snapshot-time",
        type=float,
        default=None,
        help="time [s] for A3 field grid (default: mid of last window)",
    )
    args = ap.parse_args()

    S.apply_style()
    outdir = args.out / "A_param_only"
    outdir.mkdir(parents=True, exist_ok=True)
    cache_dir = args.out / "_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    runs = [r for r in dataio.discover_block_a()]
    runs_by_model = {}
    for r in runs:
        runs_by_model.setdefault(r.model, []).append(r)
    rep = finest(runs)
    print(
        f"Block A: {sum(r.complete for r in runs)}/{len(runs)} complete runs; "
        f"representative: {[f'{m}:{r.res_label}' for m,r in rep.items()]}"
    )

    # --- cheap param figures ------------------------------------------------
    a1_per_model(runs_by_model, outdir)
    a1_grid(rep, outdir)
    a2_param_err_vs_res(runs, outdir)
    table_param_accuracy(runs, outdir)
    if "udales" in rep:
        a6_window_prior_post(rep["udales"], outdir)

    if not args.heavy:
        print("Cheap figures done. Re-run with --heavy (SLURM) for field figures.")
        return

    # --- heavy field figures ------------------------------------------------
    solid = None if args.no_mask else mask.truth_solid_mask()
    zi = mask.nearest_z_index(FC.PED_Z)
    solid2d = None if solid is None else solid[zi]

    field_metrics = {}
    for r in runs:
        if not r.complete:
            continue
        cfg = dataio.load_config(r)
        val_xy = dataio.sensor_coords(cfg, "validation")
        m = compute_field_metrics(r, solid, val_xy)
        if m is None:
            continue
        m.update(model=r.model, dx=r.dx, res_label_dx=r.res_label_dx)
        field_metrics[r.name] = m
        print(f"  field metrics {r.name}: {m['field_rmse']:.3f}")

    with open(cache_dir / "block_a_metrics.json", "w") as f:
        json.dump(field_metrics, f, indent=1)

    # snapshot time: mid of last window of a representative udales run
    snap = args.snapshot_time
    if snap is None:
        edges = dataio.window_edges(rep.get("udales", runs[0]))
        snap = float(0.5 * (edges[-2] + edges[-1])) if edges is not None else 900.0
    cfg = (
        dataio.load_config(rep["udales"])
        if "udales" in rep
        else dataio.load_config(runs[0])
    )
    assim_xy = dataio.sensor_coords(cfg, "assimilation")
    val_xy = dataio.sensor_coords(cfg, "validation")
    a3_state_fields(
        rep, outdir, time_s=snap, solid2d=solid2d, assim_xy=assim_xy, val_xy=val_xy
    )
    a4_valsensors(rep, outdir)
    a5_state_err_vs_res(field_metrics, outdir)
    table_state_accuracy(field_metrics, outdir)
    print("Block A heavy figures done.")


if __name__ == "__main__":
    main()
