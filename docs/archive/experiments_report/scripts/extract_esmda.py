"""Extract ESMDA ``run_summary.yaml`` numbers into LaTeX table macros.

Run with::

    cd /Users/ntmucke/Code/pyurbanair && \
        pixi run -e dev python experiments_report/scripts/extract_esmda.py

Writes ``experiments_report/sections/esmda_tables.tex``, which defines the three
macros required by STYLE_SPEC section 3: ``\\EsmdaMatrixTable``,
``\\EsmdaMasterTable`` and ``\\EsmdaCalibTable``.

Run IDs come from :func:`figlib.run_ids` (the authority, STYLE_SPEC section 1).
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Any

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
import figlib as fl  # noqa: E402

OUT = fl.REPO / "experiments_report/sections/esmda_tables.tex"

METHOD = "esmda"
CASE_LABEL = {
    "inflow": r"\code{inflow}",
    "inflow_turb": r"\code{inflow\_turb}",
    "periodic": r"\code{periodic}",
}
CASE_ORDER = {"inflow": 0, "inflow_turb": 1, "periodic": 2}
OBS_ORDER = {"7p5": 0, "15": 1, "30": 2}
OBS_LABEL = {"7p5": "7.5", "15": "15", "30": "30"}
TRUTH_LABEL = {"pyudales": "uDALES", "pypalm": "PALM"}
N_MASTER_NUM = 9  # numeric columns of the master table (for the "failed" span)
N_CALIB_NUM = 11


# --------------------------------------------------------------------------- #
# loading
# --------------------------------------------------------------------------- #
def parse_name(name: str) -> dict[str, str]:
    """Split ``<truth>_to_<assim>_w3_<loc>_obs<int>_<case>`` into fields."""
    truth, rest = name.split("_to_", 1)
    assim, rest = rest.split("_w3_", 1)
    loc, rest = rest.split("_obs", 1)
    obs, case = rest.split("_", 1)
    return {
        "name": name,
        "truth": truth,
        "assim": assim,
        "loc": "on" if loc == "loccorrelation" else "off",
        "obs": obs,
        "case": case,
    }


def get(node: Any, *keys: str) -> Any:
    for key in keys:
        if not isinstance(node, dict):
            return None
        node = node.get(key)
    return node


def edge_fraction(counts: list[int] | None) -> float | None:
    """Fraction of rank-histogram mass in the two outer bins (1.0 = truth always outside)."""
    if not counts:
        return None
    total = sum(counts)
    return (counts[0] + counts[-1]) / total if total else None


def load_runs() -> list[dict]:
    """All 20 ESMDA runs with their ID, config fields and (if present) metrics."""
    runs: list[dict] = []
    for rid, name in fl.run_ids(METHOD).items():
        info = parse_name(name)
        info["id"] = rid
        info["n"] = int(rid[1:])
        d = fl.run_dir(METHOD, name)
        summary = d / "run_summary.yaml"
        info["ok"] = summary.exists()
        if info["ok"]:
            info["s"] = yaml.safe_load(summary.read_text())
        runs.append(info)
    return runs


def sort_key(r: dict) -> tuple:
    return (
        r["truth"] != "pyudales",
        CASE_ORDER[r["case"]],
        OBS_ORDER[r["obs"]],
        r["loc"] != "on",
    )


# --------------------------------------------------------------------------- #
# formatting
# --------------------------------------------------------------------------- #
def fmt(x: Any, nd: int = 3) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "--"
    return f"{x:.{nd}f}"


def pct(x: Any) -> str:
    if x is None:
        return "--"
    return f"{100 * x:.0f}"


def bold(text: str, on: bool) -> str:
    return r"\textbf{" + text + "}" if on else text


def metrics(r: dict) -> dict[str, Any]:
    """Flat metric dict for one completed run."""
    s = r["s"]
    pm = get(s, "parameter_metrics") or {}
    sm = get(s, "sensor_metrics") or {}
    ss = get(s, "sensor_statistics") or {}
    hit = get(s, "field_metrics", "hit_rate_posterior") or {}
    dm = get(s, "esmda_diagnostics", "data_mismatch") or {}
    return {
        "assim": get(sm, "assimilation", "velocity_vector_rmse", "mean"),
        "valid": get(sm, "validation", "velocity_vector_rmse", "mean"),
        "assim_es": get(sm, "assimilation", "velocity_vector_energy_score", "mean"),
        "valid_es": get(sm, "validation", "velocity_vector_energy_score", "mean"),
        "field": get(s, "state_metrics", "vel_magnitude_rmse", "mean"),
        "a_rmse": get(pm, "inflow_angle", "rmse", "mean"),
        "a_prior": get(pm, "inflow_angle", "prior_rmse_mean"),
        "a_red": get(pm, "inflow_angle", "rmse_reduction_vs_prior"),
        "a_crps_red": get(pm, "inflow_angle", "crps_reduction_vs_prior"),
        "a_ne": get(pm, "inflow_angle", "normalized_error", "final"),
        "u_rmse": get(pm, "velocity_magnitude", "rmse", "mean"),
        "u_prior": get(pm, "velocity_magnitude", "prior_rmse_mean"),
        "u_red": get(pm, "velocity_magnitude", "rmse_reduction_vs_prior"),
        "u_ne": get(pm, "velocity_magnitude", "normalized_error", "final"),
        "za": get(pm, "inflow_angle", "z_score", "std"),
        "zu": get(pm, "velocity_magnitude", "z_score", "std"),
        "zpar": get(pm, "pooled", "z_score", "std"),
        "ca": get(pm, "inflow_angle", "contraction_ratio", "mean"),
        "cu": get(pm, "velocity_magnitude", "contraction_ratio", "mean"),
        "chi2": get(s, "filter_diagnostics", "innovation_chi2", "mean"),  # None: ESMDA
        "q": hit.get("q"),
        "q_u": hit.get("u"),
        "q_v": hit.get("v"),
        "q_w": hit.get("w"),
        "edge": edge_fraction(
            get(ss, "validation", "posterior", "mean_magnitude", "rank_counts")
        ),
        "wall": get(s, "timing", "esmda_total_seconds") / 3600.0,
        "dm": dm.get("per_step_median"),
        "nobs": dm.get("num_observations"),
        "n_unique": min(get(s, "ensemble_health", "n_unique_per_window") or [0]),
        "pair": get(s, "ensemble_health", "min_over_median_pairwise"),
    }


def mark_best(rows: list[dict], cols: list[str]) -> None:
    """Flag the best (minimum) value of each column within each truth block."""
    for truth in ("pyudales", "pypalm"):
        block = [r for r in rows if r["truth"] == truth and r["ok"]]
        for col in cols:
            cand = [r for r in block if isinstance(r["m"].get(col), (int, float))]
            if not cand:
                continue
            best = min(cand, key=lambda r: r["m"][col])
            best.setdefault("best", set()).add(col)


def blocks(rows: list[dict], render) -> str:
    """Render the uDALES block, a midrule, then the PALM block."""
    out: list[str] = []
    prev = None
    for r in rows:
        if prev is not None and r["truth"] != prev:
            out.append(r"\midrule")
        out.append(render(r) + r" \\")
        prev = r["truth"]
    return "\n".join(out)


# --------------------------------------------------------------------------- #
# tables
# --------------------------------------------------------------------------- #
def table_matrix(runs: list[dict]) -> str:
    def render(r: dict) -> str:
        status = "ok" if r["ok"] else "failed (truth run)"
        folder = r["name"].replace("_", r"\_")
        return (
            f"{r['id']} & \\code{{{folder}}} & {TRUTH_LABEL[r['truth']]} & "
            f"{CASE_LABEL[r['case']]} & {r['loc']} & {OBS_LABEL[r['obs']]} & {status}"
        )

    return (
        r"""\newcommand{\EsmdaMatrixTable}{%
\begin{table}[htbp]
\centering
\caption{ESMDA: experiment matrix. Twenty configurations; the observation
interval is the method knob. \emph{loc} is localization on
(\code{loccorrelation}) or off (\code{locnone}). Run-directory names are given
here only; everywhere else runs are cited by ID. The two failed runs died in the
PALM \emph{truth} simulation before any assimilation started
(Section~\ref{sec:esmda-failed}).}
\label{tab:esmda-matrix}
\footnotesize
\begin{adjustbox}{max width=\textwidth}
\begin{tabular}{llllcc l}
\toprule
ID & run directory & truth & case & loc & obs [s] & status \\
\midrule
"""
        + blocks(runs, render)
        + r"""
\bottomrule
\end{tabular}
\end{adjustbox}
\end{table}%
}
"""
    )


def table_master(rows: list[dict]) -> str:
    def render(r: dict) -> str:
        head = (
            f"{r['id']} & {CASE_LABEL[r['case']]} & {r['loc']} & {OBS_LABEL[r['obs']]}"
        )
        if not r["ok"]:
            return (
                head
                + r" & \multicolumn{"
                + str(N_MASTER_NUM)
                + r"}{c}{\emph{failed (truth run)}}"
            )
        m, b = r["m"], r.get("best", set())
        return (
            head
            + " & "
            + bold(fmt(m["assim"], 3), "assim" in b)
            + " & "
            + bold(fmt(m["valid"], 3), "valid" in b)
            + " & "
            + bold(fmt(m["field"], 3), "field" in b)
            + " & "
            + bold(fmt(m["a_rmse"], 2), "a_rmse" in b)
            + f" ({pct(m['a_red'])}) & "
            + bold(fmt(m["u_rmse"], 3), "u_rmse" in b)
            + f" ({pct(m['u_red'])}) & "
            + fmt(m["chi2"], 2)
            + " & "
            + fmt(m["zpar"], 1)
            + " & "
            + fmt(m["q"], 2)
            + " & "
            + fmt(m["wall"], 2)
        )

    return (
        r"""\newcommand{\EsmdaMasterTable}{%
\begin{table}[htbp]
\centering
\caption{ESMDA: master results for all 20 runs, uDALES-truth block above,
PALM-truth block below. \emph{assim} / \emph{valid} are the vector RMSE
[m/s] of the ensemble mean at the 6 assimilated and 4 validation sensors,
averaged over time; \emph{field} $\Umag$ is the domain-wide RMSE of the velocity
magnitude. The two parameter columns give the knot-mean RMSE with the reduction
vs.\ prior in parentheses (priors are re-drawn per run, so only the reduction is
comparable across rows, and never across methods). ESMDA has no sequential
innovation, so $\chi^2$ is undefined; $z_{\mathrm{par}}$ is the pooled parameter
$z$-score standard deviation (1.03 = calibrated); $q$ is the field hit rate
(uDALES truth only). Bold = best in the column within the truth block; $q$ and
$\chi^2$ are not bolded.}
\label{tab:esmda-master}
\footnotesize
\begin{adjustbox}{max width=\textwidth}
\begin{tabular}{llcc rrr rr rrr r}
\toprule
 & & & obs & \multicolumn{2}{c}{vector RMSE [m/s]} & field & \multicolumn{2}{c}{parameter RMSE (red.\%)} & & & & wall \\
\cmidrule(lr){5-6}\cmidrule(lr){8-9}
ID & case & loc & [s] & assim & valid & $\Umag$ & $\alpha$ [deg] & $\Umag$ [m/s] & $\chi^2$ & $z_{\mathrm{par}}$ & $q$ & [h] \\
\midrule
"""
        + blocks(rows, render)
        + r"""
\bottomrule
\end{tabular}
\end{adjustbox}
\end{table}%
}
"""
    )


def table_calib(rows: list[dict]) -> str:
    def render(r: dict) -> str:
        head = (
            f"{r['id']} & {CASE_LABEL[r['case']]} & {r['loc']} & {OBS_LABEL[r['obs']]}"
        )
        if not r["ok"]:
            return (
                head
                + r" & \multicolumn{"
                + str(N_CALIB_NUM)
                + r"}{c}{\emph{failed (truth run)}}"
            )
        m = r["m"]
        return (
            head
            + " & "
            + " & ".join(
                [
                    fmt(m["chi2"], 2),
                    fmt(m["za"], 1),
                    fmt(m["zu"], 1),
                    fmt(m["zpar"], 1),
                    fmt(m["ca"], 2),
                    fmt(m["cu"], 2),
                    fmt(m["q"], 2),
                    fmt(m["q_u"], 2),
                    fmt(m["q_v"], 2),
                    fmt(m["q_w"], 2),
                    fmt(m["edge"], 2),
                ]
            )
        )

    return (
        r"""\newcommand{\EsmdaCalibTable}{%
\begin{table}[htbp]
\centering
\caption{ESMDA: calibration diagnostics. $\chi^2$ is undefined for a smoother.
std$(z)$ is the standard deviation of the per-knot parameter $z$-scores
$(\theta^\ast-\bar\theta^a)/\sigma^a$; a calibrated 50-member ensemble gives
1.03, so every entry is over-confident. \emph{Contraction} is the knot-mean
posterior/prior spread ratio ($\ll 1$ = the data informed the parameter,
$\approx 1$ = unidentifiable). $q$ and its per-component values are field hit
rates (uDALES truth only). \emph{rank edge frac} is the fraction of the
validation-sensor rank histogram falling in the two outer bins; a calibrated
51-bin histogram gives 0.04.}
\label{tab:esmda-calib}
\footnotesize
\begin{adjustbox}{max width=\textwidth}
\begin{tabular}{llcc r rrr rr rrrr r}
\toprule
 & & & obs & & \multicolumn{3}{c}{std$(z)$} & \multicolumn{2}{c}{contraction} & \multicolumn{4}{c}{field hit rate} & rank edge \\
\cmidrule(lr){6-8}\cmidrule(lr){9-10}\cmidrule(lr){11-14}
ID & case & loc & [s] & $\chi^2$ & $\alpha$ & $\Umag$ & pool & $\alpha$ & $\Umag$ & $q$ & $u$ & $v$ & $w$ & frac (valid) \\
\midrule
"""
        + blocks(rows, render)
        + r"""
\bottomrule
\end{tabular}
\end{adjustbox}
\end{table}%
}
"""
    )


# --------------------------------------------------------------------------- #
def main() -> None:
    runs = load_runs()
    for r in runs:
        if r["ok"]:
            r["m"] = metrics(r)
    by_id = sorted(runs, key=lambda r: r["n"])
    ordered = sorted(runs, key=sort_key)
    mark_best(
        ordered,
        ["assim", "valid", "field", "a_rmse", "u_rmse", "zpar", "wall"],
    )

    n_ok = sum(r["ok"] for r in runs)
    header = (
        "% AUTO-GENERATED by experiments_report/scripts/extract_esmda.py\n"
        "% Do not edit by hand; re-run the script instead.\n"
        f"% {len(runs)} runs, {n_ok} completed, {len(runs) - n_ok} failed.\n\n"
    )
    OUT.write_text(
        header
        + table_matrix(by_id)
        + "\n"
        + table_master(ordered)
        + "\n"
        + table_calib(ordered)
    )
    print(f"wrote {OUT} ({n_ok}/{len(runs)} runs with metrics)")

    # Console digest used while writing the prose.
    for r in ordered:
        if not r["ok"]:
            print(f"{r['id']:<4} {r['name']:<58} FAILED")
            continue
        m = r["m"]
        print(
            f"{r['id']:<4} {TRUTH_LABEL[r['truth']]:<6} {r['case']:<11} "
            f"loc={r['loc']:<3} obs={OBS_LABEL[r['obs']]:<4} "
            f"as={m['assim']:.3f} va={m['valid']:.3f} vES={m['valid_es']:.3f} "
            f"fld={m['field']:.3f} a={m['a_rmse']:6.2f}({100*m['a_red']:+3.0f}%) "
            f"u={m['u_rmse']:.3f}({100*m['u_red']:+3.0f}%) "
            f"z={m['za']:.1f}/{m['zu']:.1f}/{m['zpar']:.1f} "
            f"c={m['ca']:.2f}/{m['cu']:.2f} ne={m['a_ne']:+.2f}/{m['u_ne']:+.2f} "
            f"q={m['q']} edge={m['edge']:.2f} h={m['wall']:.2f} "
            f"dm={[round(v,1) for v in m['dm']]} Nd={m['nobs']} "
            f"uniq={m['n_unique']} pair={m['pair']:.2f} "
            f"acrps={100*m['a_crps_red']:.0f}%"
        )


if __name__ == "__main__":
    main()
