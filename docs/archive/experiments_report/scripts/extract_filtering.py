"""Extract the LaTeX tables of the sequential-filtering (EnKF) section.

Reads every ``run_summary.yaml`` under
``presentations/isda_new/experiments/filtering/`` plus the definitive Hydra
overrides in ``_logs/<run>.args`` -- mode / inflation / localization are *not*
fully recorded in ``run_summary.yaml:configuration`` -- and writes the booktabs
table bodies to ``experiments_report/sections/filtering_tables.tex``.

Run IDs come from :func:`figlib.run_ids` (STYLE_SPEC section 1).

Run with::

    cd /Users/ntmucke/Code/pyurbanair
    pixi run -e dev python experiments_report/scripts/extract_filtering.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))

import figlib as fl  # noqa: E402

EXP = fl.EXP / "filtering"
OUT = fl.REPO / "experiments_report" / "sections" / "filtering_tables.tex"

STATIC_PRIOR_ALPHA = 19.29  # deg, static prior RMSE carried by state-only runs
STATIC_PRIOR_UMAG = 0.919  # m/s

CASE_ORDER = ["inflow", "inflow_turb", "periodic"]
CASE_TEX = {
    "inflow": r"\code{inflow}",
    "inflow_turb": r"\code{inflow\_turb}",
    "periodic": r"\code{periodic}",
}
KNOB_ORDER = ["state,none", "state,RTPS", "joint,RTPS"]
LOC_ORDER = ["none", "corr"]


# --------------------------------------------------------------------------- #
# loading
# --------------------------------------------------------------------------- #
def esc(text: str) -> str:
    return text.replace("_", r"\_")


def load_args(run: str) -> dict[str, str]:
    path = EXP / "_logs" / f"{run}.args"
    out: dict[str, str] = {}
    if not path.exists():
        return out
    for line in path.read_text().splitlines():
        key, sep, val = line.strip().partition("=")
        if sep:
            out[key] = val
    return out


def load_progress() -> dict[str, dict[str, str]]:
    lines = (EXP / "_logs" / "progress.tsv").read_text().strip().splitlines()
    head = lines[0].split("\t")
    return {
        rec["id"]: rec
        for rec in (dict(zip(head, ln.split("\t"))) for ln in lines[1:])
    }


def get(node: Any, *keys: str) -> Any:
    for key in keys:
        if not isinstance(node, dict):
            return None
        node = node.get(key)
    return node


def edge_fraction(counts: list[int] | None) -> float | None:
    if not counts:
        return None
    total = sum(counts)
    return None if total == 0 else (counts[0] + counts[-1]) / total


class Run:
    """One filtering run: configuration knobs plus the metrics both tables need."""

    def __init__(self, rid: str, name: str, progress: dict[str, dict[str, str]]) -> None:
        self.rid = rid
        self.name = name
        self.dir = EXP / name
        self.args = load_args(name)
        summary_path = self.dir / "run_summary.yaml"
        self.ok = summary_path.exists()
        self.s: dict[str, Any] = (
            yaml.safe_load(summary_path.read_text()) if self.ok else {}
        )
        rec = progress.get(name, {})
        self.rc = rec.get("rc")
        self.elapsed = float(rec["elapsed_s"]) if rec.get("elapsed_s") else None

        self.truth = "PALM" if name.startswith("pypalm") else "uDALES"
        for case in ("inflow_turb", "periodic", "inflow"):
            if f"_{case}" in name or name.endswith(case):
                self.case = case
                break
        self.loc = "corr" if self.args.get("filtering/localization") == "correlation" else "none"
        self.mode = self.args.get("filtering.mode", "joint")
        self.infl = self.args.get("filtering/inflation", "none")
        self.knob = f"{self.mode},{'RTPS' if self.infl == 'rtps' else 'none'}"

    # -- metric accessors --------------------------------------------------- #
    @property
    def assim(self) -> Any:
        return get(self.s, "sensor_metrics", "assimilation", "velocity_vector_rmse", "mean")

    @property
    def valid(self) -> Any:
        return get(self.s, "sensor_metrics", "validation", "velocity_vector_rmse", "mean")

    @property
    def valid_final(self) -> Any:
        return get(self.s, "sensor_metrics", "validation", "velocity_vector_rmse", "final")

    @property
    def field(self) -> Any:
        return get(self.s, "state_metrics", "vel_magnitude_rmse", "mean")

    @property
    def chi2(self) -> Any:
        return get(self.s, "filter_diagnostics", "innovation_chi2", "mean")

    @property
    def chi2_final(self) -> Any:
        return get(self.s, "filter_diagnostics", "innovation_chi2", "final")

    def par(self, p: str, *keys: str) -> Any:
        return get(self.s, "parameter_metrics", p, *keys)

    @property
    def alpha(self) -> Any:
        return self.par("inflow_angle", "rmse", "mean")

    @property
    def umag(self) -> Any:
        return self.par("velocity_magnitude", "rmse", "mean")

    @property
    def z_pool(self) -> Any:
        return get(self.s, "parameter_metrics", "pooled", "z_score", "std")

    @property
    def q(self) -> Any:
        return get(self.s, "field_metrics", "hit_rate_posterior", "q")

    def q_comp(self, c: str) -> Any:
        return get(self.s, "field_metrics", "hit_rate_posterior", c)

    @property
    def wall(self) -> Any:
        t = get(self.s, "timing", "filter_total_seconds")
        return None if t is None else t / 3600.0

    @property
    def edge(self) -> Any:
        return edge_fraction(
            get(
                self.s,
                "sensor_statistics",
                "validation",
                "posterior",
                "mean_magnitude",
                "rank_counts",
            )
        )

    @property
    def z_valid(self) -> Any:
        return get(
            self.s,
            "sensor_statistics",
            "validation",
            "posterior",
            "mean_magnitude",
            "z_score",
            "std",
        )

    @property
    def joint(self) -> bool:
        return self.mode == "joint"


# --------------------------------------------------------------------------- #
# formatting
# --------------------------------------------------------------------------- #
def fmt(value: Any, digits: int = 3, bold: bool = False) -> str:
    if value is None:
        return "--"
    text = f"{value:.{digits}f}"
    return rf"\textbf{{{text}}}" if bold else text


def fmt_par(run: Run, p: str, digits: int, bold: bool) -> str:
    """Parameter RMSE with reduction-vs-prior; carried priors italic with a dagger."""
    val = run.par(p, "rmse", "mean")
    if val is None:
        return "--"
    text = f"{val:.{digits}f}"
    if not run.joint:
        return rf"\textit{{{text}}}$^\dagger$"
    red = run.par(p, "rmse_reduction_vs_prior")
    if bold:
        text = rf"\textbf{{{text}}}"
    return text if red is None else rf"{text} ({red * 100:.0f})"


def best(runs: list[Run], attr: str) -> float | None:
    vals = [getattr(r, attr) for r in runs if getattr(r, attr) is not None]
    return min(vals) if vals else None


def best_par(runs: list[Run], p: str) -> float | None:
    vals = [r.par(p, "rmse", "mean") for r in runs if r.joint and r.par(p, "rmse", "mean")]
    return min(vals) if vals else None


# --------------------------------------------------------------------------- #
# tables
# --------------------------------------------------------------------------- #
def sort_key(r: Run) -> tuple:
    return (
        0 if r.truth == "uDALES" else 1,
        CASE_ORDER.index(r.case),
        KNOB_ORDER.index(r.knob),
        LOC_ORDER.index(r.loc),
    )


def matrix_table(runs: list[Run]) -> str:
    rows = [
        r"\begin{tabular}{l>{\raggedright\arraybackslash\ttfamily\scriptsize}"
        r"p{0.30\textwidth}llllll}",
        r"\toprule",
        r"ID & \normalfont run directory & truth & case & loc & mode & inflation & "
        r"status \\",
        r"\midrule",
    ]
    ordered = sorted(runs, key=sort_key)
    prev_truth = None
    for r in ordered:
        if prev_truth is not None and r.truth != prev_truth:
            rows.append(r"\midrule")
        prev_truth = r.truth
        status = "ok" if r.ok else "failed (truth run)"
        rows.append(
            f"\\code{{{r.rid}}} & {esc(r.name).replace(chr(92) + '_', chr(92) + '_' + chr(92) + 'fbk{}')} & {r.truth} & "
            f"{CASE_TEX[r.case]} & {r.loc} & {r.mode} & "
            f"{'RTPS' if r.infl == 'rtps' else 'none'} & {status} \\\\"
        )
    rows += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(rows)


def master_table(runs: list[Run]) -> str:
    rows = [
        r"\begin{tabular}{llllrrrrrrrrr}",
        r"\toprule",
        r"ID & case & loc & knob & assim & valid & field $\Umag$ & "
        r"$\alpha$ [deg] (red\%) & $\Umag$ [m/s] (red\%) & $\chi^2$ & "
        r"$z_\mathrm{par}$ & $q$ & wall [h] \\",
        r"\midrule",
    ]
    ordered = sorted(runs, key=sort_key)
    for truth in ("uDALES", "PALM"):
        block = [r for r in ordered if r.truth == truth]
        if not block:
            continue
        if truth == "PALM":
            rows.append(r"\midrule")
        okb = [r for r in block if r.ok]
        b = {a: best(okb, a) for a in ("assim", "valid", "field", "z_pool", "wall")}
        ba = best_par(okb, "inflow_angle")
        bu = best_par(okb, "velocity_magnitude")
        for r in block:
            head = (
                f"\\code{{{r.rid}}} & {CASE_TEX[r.case]} & {r.loc} & \\code{{{r.knob}}}"
            )
            if not r.ok:
                rows.append(
                    head + r" & \multicolumn{9}{c}{failed (truth run)} \\"
                )
                continue
            rows.append(
                " & ".join(
                    [
                        head,
                        fmt(r.assim, 3, r.assim == b["assim"]),
                        fmt(r.valid, 3, r.valid == b["valid"]),
                        fmt(r.field, 3, r.field == b["field"]),
                        fmt_par(r, "inflow_angle", 2, r.par("inflow_angle", "rmse", "mean") == ba),
                        fmt_par(r, "velocity_magnitude", 3, r.par("velocity_magnitude", "rmse", "mean") == bu),
                        fmt(r.chi2, 2),
                        fmt(r.z_pool, 2, r.z_pool == b["z_pool"]),
                        fmt(r.q, 3),
                        fmt(r.wall, 2, r.wall == b["wall"]),
                    ]
                )
                + r" \\"
            )
    rows += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(rows)


def calib_table(runs: list[Run]) -> str:
    rows = [
        r"\begin{tabular}{llllrrrrrrrrrrr}",
        r"\toprule",
        r"ID & case & loc & knob & $\chi^2$ & $z_\alpha$ & $z_{\Umag}$ & "
        r"$z_\mathrm{pool}$ & contr.\ $\alpha$ & contr.\ $\Umag$ & "
        r"$q$ & $q_u$ & $q_v$ & $q_w$ & edge \\",
        r"\midrule",
    ]
    ordered = [r for r in sorted(runs, key=sort_key) if r.ok]
    prev_truth = None
    for r in ordered:
        if prev_truth is not None and r.truth != prev_truth:
            rows.append(r"\midrule")
        prev_truth = r.truth
        rows.append(
            " & ".join(
                [
                    f"\\code{{{r.rid}}}",
                    CASE_TEX[r.case],
                    r.loc,
                    f"\\code{{{r.knob}}}",
                    fmt(r.chi2, 2),
                    fmt(r.par("inflow_angle", "z_score", "std"), 2),
                    fmt(r.par("velocity_magnitude", "z_score", "std"), 2),
                    fmt(r.z_pool, 2),
                    fmt(r.par("inflow_angle", "contraction_ratio", "mean"), 2),
                    fmt(r.par("velocity_magnitude", "contraction_ratio", "mean"), 2),
                    fmt(r.q, 3),
                    fmt(r.q_comp("u"), 3),
                    fmt(r.q_comp("v"), 3),
                    fmt(r.q_comp("w"), 3),
                    fmt(r.edge, 3),
                ]
            )
            + r" \\"
        )
    rows += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(rows)


def loc_table(by: dict[tuple, Run]) -> str:
    """Paired localization comparison: rows differing only in ``loc``."""
    rows = [
        r"\begin{tabular}{lllllrrrl}",
        r"\toprule",
        r"truth & case & knob & no loc & loc & valid (none) & valid (corr) & "
        r"$\Delta$ & $\chi^2$ (none $\to$ corr) \\",
        r"\midrule",
    ]
    for truth in ("uDALES", "PALM"):
        for case in CASE_ORDER:
            for knob in KNOB_ORDER:
                a = by.get((truth, case, knob, "none"))
                b = by.get((truth, case, knob, "corr"))
                if not (a and b and a.ok and b.ok):
                    continue
                d = 100.0 * (b.valid - a.valid) / a.valid
                col = r"\bad" if d > 0 else r"\good"
                rows.append(
                    " & ".join(
                        [
                            truth,
                            CASE_TEX[case],
                            f"\\code{{{knob}}}",
                            f"\\code{{{a.rid}}}",
                            f"\\code{{{b.rid}}}",
                            fmt(a.valid, 3),
                            fmt(b.valid, 3),
                            f"{col}{{${d:+.1f}\\%$}}",
                            f"${a.chi2:.2f} \\to {b.chi2:.2f}$",
                        ]
                    )
                    + r" \\"
                )
    rows += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(rows)


def ladder_table(by: dict[tuple, Run]) -> str:
    """The method-knob ladder none -> RTPS -> joint, at ``locnone``, per case."""
    rows = [
        r"\begin{tabular}{lllrrrrrr}",
        r"\toprule",
        r"case & ID & knob & assim & valid & field $\Umag$ & $\alpha$ [deg] & "
        r"$\chi^2$ & edge \\",
        r"\midrule",
    ]
    for i, case in enumerate(CASE_ORDER):
        if i:
            rows.append(r"\midrule")
        for knob in KNOB_ORDER:
            r = by.get(("uDALES", case, knob, "none"))
            if r is None or not r.ok:
                continue
            alpha = (
                rf"\textit{{{r.alpha:.2f}}}$^\dagger$"
                if not r.joint
                else f"{r.alpha:.2f}"
            )
            rows.append(
                " & ".join(
                    [
                        CASE_TEX[case] if knob == KNOB_ORDER[0] else "",
                        f"\\code{{{r.rid}}}",
                        f"\\code{{{knob}}}",
                        fmt(r.assim, 3),
                        fmt(r.valid, 3),
                        fmt(r.field, 3),
                        alpha,
                        fmt(r.chi2, 2),
                        fmt(r.edge, 3),
                    ]
                )
                + r" \\"
            )
    rows += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(rows)


WRAP = (
    "\\newcommand{{\\{name}}}{{%\n"
    "\\footnotesize\n"
    "\\begin{{adjustbox}}{{max width=\\textwidth}}\n"
    "{body}\n"
    "\\end{{adjustbox}}}}\n"
)


def main() -> None:
    progress = load_progress()
    ids = fl.run_ids("filtering")
    runs = [Run(rid, name, progress) for rid, name in ids.items()]
    by = {(r.truth, r.case, r.knob, r.loc): r for r in runs}

    parts = [
        "% Generated by experiments_report/scripts/extract_filtering.py -- do not edit.",
        r"\providecommand{\fbk}{\discretionary{}{}{}}",
        WRAP.format(name="FiltMatrixTable", body=matrix_table(runs)),
        WRAP.format(name="FiltMasterTable", body=master_table(runs)),
        WRAP.format(name="FiltCalibTable", body=calib_table(runs)),
        WRAP.format(name="FiltLocTable", body=loc_table(by)),
        WRAP.format(name="FiltLadderTable", body=ladder_table(by)),
    ]
    OUT.write_text("\n".join(parts))
    print(f"wrote {OUT}")

    # ---- console summary used when writing the prose ---------------------- #
    ok = [r for r in runs if r.ok]
    print(f"{len(ok)}/{len(runs)} runs completed; "
          f"total wall {sum(r.elapsed for r in ok if r.elapsed) / 3600:.1f} h")
    hdr = f"{'ID':<4}{'truth':<7}{'case':<12}{'loc':<6}{'knob':<12}"
    print(hdr + "  assim  valid  field  alpha   |U|    chi2  zpar   q      edge   wall")
    for r in sorted(runs, key=sort_key):
        if not r.ok:
            print(f"{r.rid:<4}{r.truth:<7}{r.case:<12}{r.loc:<6}{r.knob:<12}  FAILED rc={r.rc}")
            continue
        def n(v, d=3):
            return "  --  " if v is None else f"{v:.{d}f}"
        print(
            f"{r.rid:<4}{r.truth:<7}{r.case:<12}{r.loc:<6}{r.knob:<12}"
            f"  {n(r.assim)}  {n(r.valid)}  {n(r.field)}  {n(r.alpha,2):>6}  "
            f"{n(r.umag)}  {n(r.chi2,2):>5}  {n(r.z_pool,2):>5}  {n(r.q)}  "
            f"{n(r.edge)}  {n(r.wall,2)}"
        )
    print("\nextras (final-cycle / spread / z_valid):")
    for r in sorted(runs, key=sort_key):
        if not r.ok:
            continue
        def n(v, d=3):
            return "--" if v is None else f"{v:.{d}f}"

        fd = r.s.get("filter_diagnostics", {})
        print(
            f"{r.rid:<4} valid_final={n(r.valid_final)} chi2_final={n(r.chi2_final,2)} "
            f"zvalid={n(r.z_valid,2)} "
            f"sprd_prior={n(get(fd,'state_spread_prior','mean'))} "
            f"sprd_post={n(get(fd,'state_spread_posterior','mean'))} "
            f"parsprd_post={n(get(fd,'param_spread_posterior','mean'))} "
            f"obsprior={n(get(fd,'obs_prior_rmse','mean'))} "
            f"obspost={n(get(fd,'obs_posterior_rmse','mean'))} "
            f"uniq={get(r.s,'ensemble_health','n_unique')} "
            f"div={n(get(r.s,'ensemble_health','min_over_median_pairwise'),4)} "
            f"crps_a={n(r.par('inflow_angle','crps','mean'),2)} "
            f"assim_final={n(get(r.s,'sensor_metrics','assimilation','velocity_vector_rmse','final'))}"
        )


if __name__ == "__main__":
    main()
