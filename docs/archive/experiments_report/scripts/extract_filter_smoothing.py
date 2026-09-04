"""Extract filter-smoothing (hybrid) run metrics into LaTeX booktabs table bodies.

Run from the repo root::

    pixi run -e dev python experiments_report/scripts/extract_filter_smoothing.py

Emits ``experiments_report/sections/filter_smoothing_tables.tex`` with the three
tables required by ``STYLE_SPEC.md`` section 3: ``\\HybMatrixTable`` (experiment
matrix), ``\\HybMasterTable`` (master results) and ``\\HybCalibTable``
(calibration), plus a handful of inline ``\\Hyb*`` number macros.

Run IDs H1..H20 come from :func:`figlib.run_ids`, which is the single authority
for the ID mapping (sorted run-directory names, numbered from 1).
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Any

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
import figlib as fl  # noqa: E402

OUT = fl.REPO / "experiments_report" / "sections" / "filter_smoothing_tables.tex"

CASE_ORDER = {"inflow": 0, "inflow_turb": 1, "periodic": 2}
KNOB_ORDER = {"7.5": 0, "15": 1, "30": 2}


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def esc(text: str) -> str:
    return text.replace("_", r"\_")


def get(node: Any, *keys: str, default: Any = None) -> Any:
    for key in keys:
        if not isinstance(node, dict) or key not in node:
            return default
        node = node[key]
    return node


def fmt(value: Any, digits: int = 3) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "--"
    return f"{value:.{digits}f}"


def sig3(value: Any) -> str:
    """Three significant digits, without exponent notation."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "--"
    v = abs(value)
    if v == 0:
        return "0.00"
    digits = max(0, 2 - int(math.floor(math.log10(v))))
    return f"{value:.{digits}f}"


def pct(value: Any) -> str:
    """Reduction-vs-prior as an integer percentage."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "--"
    return f"{round(100 * value):d}"


def bold(text: str, on: bool) -> str:
    return rf"\textbf{{{text}}}" if on else text


def decode(name: str) -> dict[str, str]:
    """truth / case / loc / knob from a run-directory name."""
    if name.endswith("inflow_turb"):
        case = "inflow_turb"
    elif name.endswith("periodic"):
        case = "periodic"
    else:
        case = "inflow"
    return {
        "truth": "PALM" if name.startswith("pypalm") else "uDALES",
        "case": case,
        "loc": "on" if "loccorrelation" in name else "off",
        "knob": "7.5" if "obs7p5" in name else ("30" if "obs30" in name else "15"),
    }


def sort_key(rec: dict[str, Any]) -> tuple[int, int, int, int]:
    d = rec["d"]
    return (
        0 if d["truth"] == "uDALES" else 1,
        CASE_ORDER[d["case"]],
        0 if d["loc"] == "on" else 1,
        KNOB_ORDER[d["knob"]],
    )


# --------------------------------------------------------------------------- #
# load
# --------------------------------------------------------------------------- #
def pooled_rank_counts(cell: dict[str, Any] | None) -> list[float] | None:
    """Sum the per-statistic rank-count vectors of one sensor set (as the
    rank-histogram figure does)."""
    vectors = [
        [float(x) for x in counts]
        for counts in (cell or {}).values()
        if isinstance(counts, dict) is False and counts
    ]
    vectors = [v for v in vectors if sum(v) > 0]
    if not vectors:
        return None
    return [sum(col) for col in zip(*vectors)]


def rank_edge_fraction(summary: dict[str, Any], which: str) -> float | None:
    """Fraction of the pooled rank counts landing in the two extreme rank bins.

    Uniform expectation is ``2 / (N + 1) = 0.039`` for a 50-member ensemble; a
    larger value means the truth falls outside the ensemble too often.
    """
    post = get(summary, "sensor_statistics", which, "posterior", default={}) or {}
    counts = {k: v.get("rank_counts") for k, v in post.items() if isinstance(v, dict)}
    pooled = pooled_rank_counts(counts)
    if not pooled:
        return None
    total = sum(pooled)
    return None if total <= 0 else (pooled[0] + pooled[-1]) / total


def cycle_chi2(run_dir: Path) -> float | None:
    """Mean innovation chi-square over the run's cycles (``cycle_diagnostics.yaml``)."""
    path = run_dir / "cycle_diagnostics.yaml"
    if not path.exists():
        return None
    with path.open() as fh:
        cycles = yaml.safe_load(fh) or []
    vals = [
        c["innovation_chi2"]
        for c in cycles
        if isinstance(c, dict) and c.get("innovation_chi2") is not None
    ]
    return sum(vals) / len(vals) if vals else None


def row_metrics(summary: dict[str, Any], run_dir: Path) -> dict[str, Any]:
    pm = summary.get("parameter_metrics", {})
    ang = pm.get("inflow_angle", {})
    vel = pm.get("velocity_magnitude", {})
    sm = summary.get("sensor_metrics", {})
    return {
        "chi2": cycle_chi2(run_dir),
        "assim": get(sm, "assimilation", "velocity_vector_rmse", "mean"),
        "valid": get(sm, "validation", "velocity_vector_rmse", "mean"),
        "field": get(summary, "state_metrics", "vel_magnitude_rmse", "mean"),
        "field_final": get(summary, "state_metrics", "vel_magnitude_rmse", "final"),
        "ang": get(ang, "rmse", "mean"),
        "ang_final": get(ang, "rmse", "final"),
        "ang_prior": get(ang, "prior_rmse_mean"),
        "ang_red": get(ang, "rmse_reduction_vs_prior"),
        "ang_z": get(ang, "z_score", "std"),
        "ang_contr": get(ang, "contraction_ratio", "mean"),
        "vel": get(vel, "rmse", "mean"),
        "vel_final": get(vel, "rmse", "final"),
        "vel_prior": get(vel, "prior_rmse_mean"),
        "vel_red": get(vel, "rmse_reduction_vs_prior"),
        "vel_z": get(vel, "z_score", "std"),
        "vel_contr": get(vel, "contraction_ratio", "mean"),
        "z_par": get(pm, "pooled", "z_score", "std"),
        "q": get(summary, "field_metrics", "hit_rate_posterior", "q"),
        "q_u": get(summary, "field_metrics", "hit_rate_posterior", "u"),
        "q_v": get(summary, "field_metrics", "hit_rate_posterior", "v"),
        "q_w": get(summary, "field_metrics", "hit_rate_posterior", "w"),
        "edge_valid": rank_edge_fraction(summary, "validation"),
        "wall": (get(summary, "timing", "hybrid_total_seconds") or 0.0) / 3600.0,
        "cycle_s": get(summary, "timing", "mean_cycle_seconds"),
    }


def load_runs() -> list[dict[str, Any]]:
    runs: list[dict[str, Any]] = []
    for rid, name in fl.run_ids("filter_smoothing").items():
        path = fl.run_dir("filter_smoothing", name)
        summary_path = path / "run_summary.yaml"
        rec: dict[str, Any] = {
            "id": rid,
            "name": name,
            "d": decode(name),
            "ok": summary_path.exists(),
            "m": {},
        }
        if rec["ok"]:
            with summary_path.open() as fh:
                rec["m"] = row_metrics(yaml.safe_load(fh), path)
        runs.append(rec)
    return runs


# --------------------------------------------------------------------------- #
# bolding
# --------------------------------------------------------------------------- #
def best(rows: list[dict[str, Any]], key: str, mode: str = "min") -> set[int]:
    vals = [(i, r["m"].get(key)) for i, r in enumerate(rows)]
    vals = [(i, v) for i, v in vals if v is not None]
    if not vals:
        return set()
    target = min(v for _, v in vals) if mode == "min" else max(v for _, v in vals)
    return {i for i, v in vals if abs(v - target) < 1e-12}


# --------------------------------------------------------------------------- #
# tables
# --------------------------------------------------------------------------- #
HEAD = [
    r"\footnotesize",
    r"\begin{adjustbox}{max width=\textwidth}",
]
FOOT = [r"\end{tabular}", r"\end{adjustbox}", r"}"]


def matrix_table(runs: list[dict[str, Any]]) -> str:
    lines = [r"\newcommand{\HybMatrixTable}{%", *HEAD]
    lines.append(r"\begin{tabular}{@{}l l l c c l l@{}}")
    lines.append(r"\toprule")
    lines.append(
        r"ID & truth & case & loc & $\Delta t_{\mathrm{E}}$ (s) & status & run directory \\"
    )
    lines.append(r"\midrule")
    for r in runs:
        d = r["d"]
        status = "ok" if r["ok"] else r"\bad{failed (truth run)}"
        lines.append(
            " & ".join(
                [
                    r["id"],
                    d["truth"],
                    esc(d["case"]),
                    d["loc"],
                    d["knob"],
                    status,
                    rf"\code{{{esc(r['name'])}}}",
                ]
            )
            + r" \\"
        )
    lines.append(r"\bottomrule")
    lines.extend(FOOT)
    return "\n".join(lines)


NCOL_MASTER = 13


def master_table(runs: list[dict[str, Any]]) -> str:
    ud = sorted([r for r in runs if r["d"]["truth"] == "uDALES"], key=sort_key)
    pp = sorted([r for r in runs if r["d"]["truth"] == "PALM"], key=sort_key)

    lines = [
        r"% AUTO-GENERATED by experiments_report/scripts/extract_filter_smoothing.py",
        r"\newcommand{\HybMasterTable}{%",
        *HEAD,
    ]
    lines.append(r"\begin{tabular}{@{}l l c c r r r r r r r r r@{}}")
    lines.append(r"\toprule")
    lines.append(
        r"ID & case & loc & $\Delta t_{\mathrm{E}}$ & assim & valid & field $\Umag$ & "
        r"$\alpha$ RMSE & $\Umag$ RMSE & $\chi^2$ & $z_{\mathrm{par}}$ & $q$ & wall \\"
    )
    lines.append(
        r" & & & (s) & RMSE & RMSE & RMSE & [deg] (red \%) & [m/s] (red \%) & & & & [h] \\"
    )
    lines.append(r"\midrule")

    for header, block in (
        (r"\textit{uDALES truth}", ud),
        (r"\textit{PALM truth (model error)}", pp),
    ):
        lines.append(rf"\multicolumn{{{NCOL_MASTER}}}{{@{{}}l}}{{{header}}}\\")
        okrows = [r for r in block if r["ok"]]
        marks = {
            k: best(okrows, k)
            for k in ("assim", "valid", "field", "ang", "vel", "z_par")
        }
        j = 0
        for r in block:
            d = r["d"]
            head = [r["id"], esc(d["case"]), d["loc"], d["knob"]]
            if not r["ok"]:
                lines.append(
                    " & ".join(head)
                    + rf" & \multicolumn{{{NCOL_MASTER - 4}}}{{c}}"
                    rf"{{\bad{{failed (truth run)}}}} \\"
                )
                continue
            m = r["m"]
            lines.append(
                " & ".join(
                    head
                    + [
                        bold(fmt(m["assim"]), j in marks["assim"]),
                        bold(fmt(m["valid"]), j in marks["valid"]),
                        bold(fmt(m["field"]), j in marks["field"]),
                        bold(fmt(m["ang"], 2), j in marks["ang"])
                        + rf" ({pct(m['ang_red'])})",
                        bold(fmt(m["vel"]), j in marks["vel"])
                        + rf" ({pct(m['vel_red'])})",
                        sig3(m["chi2"]),
                        bold(sig3(m["z_par"]), j in marks["z_par"]),
                        fmt(m["q"]),
                        fmt(m["wall"], 2),
                    ]
                )
                + r" \\"
            )
            j += 1
        if block is ud:
            lines.append(r"\midrule")
    lines.append(r"\bottomrule")
    lines.extend(FOOT)
    return "\n".join(lines)


NCOL_CALIB = 15


def calib_table(runs: list[dict[str, Any]]) -> str:
    ud = sorted([r for r in runs if r["d"]["truth"] == "uDALES" and r["ok"]], key=sort_key)
    pp = sorted([r for r in runs if r["d"]["truth"] == "PALM" and r["ok"]], key=sort_key)
    lines = [r"\newcommand{\HybCalibTable}{%", *HEAD]
    lines.append(r"\begin{tabular}{@{}l l c c r r r r r r r r r r r@{}}")
    lines.append(r"\toprule")
    lines.append(
        r"ID & case & loc & $\Delta t_{\mathrm{E}}$ & $\chi^2$ & $z_\alpha$ & "
        r"$z_{\Umag}$ & $z_{\mathrm{pool}}$ & $c_\alpha$ & $c_{\Umag}$ & "
        r"$q$ & $q_u$ & $q_v$ & $q_w$ & edge \\"
    )
    lines.append(
        r" & & & (s) & & & & & & & & & & & frac \\"
    )
    lines.append(r"\midrule")
    for header, block in (
        (r"\textit{uDALES truth}", ud),
        (r"\textit{PALM truth (model error)}", pp),
    ):
        lines.append(rf"\multicolumn{{{NCOL_CALIB}}}{{@{{}}l}}{{{header}}}\\")
        for r in block:
            d, m = r["d"], r["m"]
            lines.append(
                " & ".join(
                    [
                        r["id"],
                        esc(d["case"]),
                        d["loc"],
                        d["knob"],
                        sig3(m["chi2"]),
                        sig3(m["ang_z"]),
                        sig3(m["vel_z"]),
                        sig3(m["z_par"]),
                        fmt(m["ang_contr"], 2),
                        fmt(m["vel_contr"], 2),
                        fmt(m["q"]),
                        fmt(m["q_u"]),
                        fmt(m["q_v"]),
                        fmt(m["q_w"]),
                        fmt(m["edge_valid"]),
                    ]
                )
                + r" \\"
            )
        if block is ud:
            lines.append(r"\midrule")
    lines.append(r"\bottomrule")
    lines.extend(FOOT)
    return "\n".join(lines)


def macros(runs: list[dict[str, Any]]) -> str:
    ok = [r for r in runs if r["ok"]]
    walls = [r["m"]["wall"] for r in ok]
    cyc = [r["m"]["cycle_s"] for r in ok if r["m"]["cycle_s"] is not None]
    zs = [r["m"]["z_par"] for r in ok if r["m"]["z_par"] is not None]
    return "\n".join(
        [
            rf"\newcommand{{\HybNumOk}}{{{len(ok)}}}",
            rf"\newcommand{{\HybNumRuns}}{{{len(runs)}}}",
            rf"\newcommand{{\HybWallMin}}{{{min(walls):.2f}}}",
            rf"\newcommand{{\HybWallMax}}{{{max(walls):.2f}}}",
            rf"\newcommand{{\HybCycleMin}}{{{min(cyc):.0f}}}",
            rf"\newcommand{{\HybCycleMax}}{{{max(cyc):.0f}}}",
            rf"\newcommand{{\HybZMin}}{{{sig3(min(zs))}}}",
            rf"\newcommand{{\HybZMax}}{{{sig3(max(zs))}}}",
        ]
    )


def main() -> None:
    runs = load_runs()
    OUT.write_text(
        "\n\n".join(
            [matrix_table(runs), master_table(runs), calib_table(runs), macros(runs)]
        )
        + "\n"
    )
    print(f"wrote {OUT} ({len(runs)} runs, {sum(r['ok'] for r in runs)} with results)")

    for r in sorted(runs, key=sort_key):
        if not r["ok"]:
            print(f"  {r['id']:<4} FAILED  {r['name']}")
            continue
        m, d = r["m"], r["d"]

        def f(key: str, digits: int = 3) -> str:
            v = m.get(key)
            return "  --" if v is None else f"{v:.{digits}f}"

        print(
            f"  {r['id']:<4} {d['truth']:<6} {d['case']:<12} loc-{d['loc']:<3} "
            f"dt {d['knob']:>4} | assim {f('assim')} valid {f('valid')} "
            f"field {f('field')}/fin {f('field_final')} | "
            f"ANG {f('ang_prior', 2)}->{f('ang', 2)} ({pct(m['ang_red'])}%) "
            f"fin {f('ang_final', 2)} | VEL {f('vel_prior')}->{f('vel')} "
            f"({pct(m['vel_red'])}%) fin {f('vel_final')} | chi2 {f('chi2', 2)} "
            f"z {f('z_par', 2)} q {f('q')} edge {f('edge_valid')} wall {f('wall', 2)}"
        )


if __name__ == "__main__":
    main()
