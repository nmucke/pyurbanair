"""Cross-method comparison tables for the ISDA 2026 experiments report.

Reads the ``run_summary.yaml`` of every completed run in the three campaigns
(``esmda/``, ``filtering/``, ``filter_smoothing/``) and emits matched
side-by-side booktabs tables, one row per *method variant* within each
(truth model, case, localization) cell.

Run from the repo root::

    cd /Users/ntmucke/Code/pyurbanair
    pixi run -e dev python experiments_report/scripts/extract_comparison.py

Writes ``experiments_report/sections/comparison_tables.tex``.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
import figlib as fl  # noqa: E402  (shared run-ID / crop library)

REPO = Path(__file__).resolve().parents[2]
EXP = REPO / "presentations" / "isda_new" / "experiments"
OUT = REPO / "experiments_report" / "sections" / "comparison_tables.tex"
DUMP = REPO / "experiments_report" / "scripts" / "comparison_data.json"

# Static prior baselines (the value every state-only filtering run reports,
# because its parameter ensemble is never updated).
PRIOR_ANGLE = 19.29
PRIOR_VMAG = 0.919
Z_EXPECTED = 1.03  # std(z) of a perfectly calibrated 50-member ensemble

CASES = ["inflow", "inflow_turb", "periodic"]
CASE_LABEL = {
    "inflow": r"\code{inflow} (laminar)",
    "inflow_turb": r"\code{inflow\_turb}",
    "periodic": r"\code{periodic}",
}
LOC_DIR = {"corr": "loccorrelation", "none": "locnone"}
LOC_LABEL = {"corr": "correlation loc.", "none": "no loc."}

# (key, table label, campaign, directory template).  Labels are the four
# method names fixed by STYLE_SPEC section 5, plus the uninflated state-only
# variant that only the filtering campaign runs.
METHODS: list[tuple[str, str, str, str]] = [
    ("E", "ESMDA", "esmda", "{truth}_to_pyudales_w3_{loc}_obs15_{case}"),
    (
        "Sn",
        "EnKF state, no infl.",
        "filtering",
        "{truth}_to_pyudales_w3_{loc}_{case}_state_inflnone",
    ),
    (
        "Sr",
        "EnKF state$+$RTPS",
        "filtering",
        "{truth}_to_pyudales_w3_{loc}_{case}_state_inflrtps",
    ),
    ("J", "EnKF joint$+$RTPS", "filtering", "{truth}_to_pyudales_w3_{loc}_{case}"),
    (
        "H",
        "Filter smoothing",
        "filter_smoothing",
        "{truth}_to_pyudales_w3_{loc}_obs15_{case}",
    ),
]

# folder name -> run ID (E*/F*/H*), the identifiers used by Sections 2-4.
RUN_ID: dict[str, dict[str, str]] = {
    m: {name: rid for rid, name in fl.run_ids(m).items()} for m in fl.METHODS
}
# Methods whose parameter ensemble is actually updated. For the others the
# parameter columns and the pooled parameter z-score are carried priors.
PARAM_METHODS = {"E", "J", "H"}
STATIC_PARAM_COLS = {"angle", "vmag", "zpool"}


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def get(node: Any, *keys: str) -> Any:
    for key in keys:
        if not isinstance(node, dict):
            return None
        node = node.get(key)
    return node


def fmt(value: Any, digits: int = 3, bold: bool = False) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "--"
    text = f"{value:.{digits}f}"
    return rf"\textbf{{{text}}}" if bold else text


def edge_fraction(counts: list[int] | None) -> float | None:
    if not counts:
        return None
    total = sum(counts)
    return (counts[0] + counts[-1]) / total if total else None


class Row:
    """One (truth, case, loc, method) cell, loaded from its run_summary.yaml."""

    def __init__(
        self, truth: str, case: str, loc: str, method: tuple[str, str, str, str]
    ):
        self.key, self.label, self.campaign, template = method
        self.truth, self.case, self.loc = truth, case, loc
        self.name = template.format(truth=truth, loc=LOC_DIR[loc], case=case)
        self.dir = EXP / self.campaign / self.name
        self.rid = RUN_ID[self.campaign].get(self.name, "--")
        self.exists = self.dir.is_dir()
        path = self.dir / "run_summary.yaml"
        self.ok = path.exists()
        self.m: dict[str, Any] = {}
        if not self.ok:
            return
        s = yaml.safe_load(path.read_text())
        sm = get(s, "sensor_metrics") or {}
        pm = get(s, "parameter_metrics") or {}
        ss = get(s, "sensor_statistics") or {}
        wall = (
            get(s, "timing", "esmda_total_seconds")
            or get(s, "timing", "filter_total_seconds")
            or get(s, "timing", "hybrid_total_seconds")
        )
        self.m = {
            "assim": get(sm, "assimilation", "velocity_vector_rmse", "mean"),
            "valid": get(sm, "validation", "velocity_vector_rmse", "mean"),
            "assim_es": get(sm, "assimilation", "velocity_vector_energy_score", "mean"),
            "valid_es": get(sm, "validation", "velocity_vector_energy_score", "mean"),
            "state": get(s, "state_metrics", "vel_magnitude_rmse", "mean"),
            "angle": get(pm, "inflow_angle", "rmse", "mean"),
            "angle_prior": get(pm, "inflow_angle", "prior_rmse_mean"),
            "vmag": get(pm, "velocity_magnitude", "rmse", "mean"),
            "vmag_prior": get(pm, "velocity_magnitude", "prior_rmse_mean"),
            "chi2": get(s, "filter_diagnostics", "innovation_chi2", "mean"),
            "zpool": get(pm, "pooled", "z_score", "std"),
            "zval": get(
                ss, "validation", "posterior", "mean_magnitude", "z_score", "std"
            ),
            "edge": edge_fraction(
                get(ss, "validation", "posterior", "mean_magnitude", "rank_counts")
            ),
            "q": get(s, "field_metrics", "hit_rate_posterior", "q"),
            "wall": (wall / 3600.0) if wall else None,
            "onp": get(s, "esmda_diagnostics", "data_mismatch", "per_step_median"),
        }
        # State-only filters never update the parameter ensemble: flag the two
        # parameter columns as "carried prior" so they are not bolded.
        self.static_params = self.key in {"Sn", "Sr"}

    @property
    def group(self) -> tuple[str, str, str]:
        return (self.truth, self.case, self.loc)


def load_rows() -> list[Row]:
    rows: list[Row] = []
    for truth in ("pyudales", "pypalm"):
        for case in CASES:
            for loc in ("corr", "none"):
                for method in METHODS:
                    row = Row(truth, case, loc, method)
                    if row.exists:
                        rows.append(row)
    return rows


# --------------------------------------------------------------------------- #
# bolding: best value per column *within* each (truth, case, loc) group
# --------------------------------------------------------------------------- #
def mark_best(rows: list[Row]) -> None:
    specs: list[tuple[str, str, float | None]] = [
        ("assim", "min", None),
        ("valid", "min", None),
        ("state", "min", None),
        ("valid_es", "min", None),
        ("angle", "min", None),
        ("vmag", "min", None),
        ("chi2", "target", 1.0),
        ("zpool", "target", Z_EXPECTED),
        ("zval", "target", Z_EXPECTED),
        ("q", "max", None),
        ("wall", "min", None),
    ]
    groups: dict[tuple[str, str, str], list[Row]] = {}
    for row in rows:
        if row.ok:
            groups.setdefault(row.group, []).append(row)
    for members in groups.values():
        for col, mode, target in specs:
            cand = [
                r
                for r in members
                if isinstance(r.m.get(col), (int, float))
                and not (col in STATIC_PARAM_COLS and r.static_params)
            ]
            if not cand:
                continue
            if mode == "min":
                best = min(cand, key=lambda r: r.m[col])
            elif mode == "max":
                best = max(cand, key=lambda r: r.m[col])
            else:
                best = min(cand, key=lambda r: abs(r.m[col] - float(target)))
            best.m[col + "_b"] = True


def cell(row: Row, col: str, digits: int) -> str:
    return fmt(row.m.get(col), digits, bool(row.m.get(col + "_b")))


# --------------------------------------------------------------------------- #
# master tables
# --------------------------------------------------------------------------- #
NCOL_MASTER = 13


def master_table(rows: list[Row], truth: str, macro: str) -> list[str]:
    out: list[str] = []
    add = out.append
    add(rf"\newcommand{{\{macro}}}{{%")
    add(r"\footnotesize")
    add(r"\begin{adjustbox}{max width=\textwidth}")
    add(r"\begin{tabular}{@{}ll rrr rr rrrr r@{}}")
    add(r"\toprule")
    add(
        r"ID & method & \multicolumn{3}{c}{sensor / field RMSE} & "
        r"\multicolumn{2}{c}{parameters} & \multicolumn{4}{c}{calibration} & wall \\"
    )
    add(r"\cmidrule(lr){3-5}\cmidrule(lr){6-7}\cmidrule(lr){8-11}")
    add(
        r" & & assim & valid & field $\Umag$ & $\alpha$ [deg] & $\Umag$ [m/s] & "
        r"$\chi^2$ & $z_{\mathrm{par}}$ & $z_{\mathrm{val}}$ & $q$ & [h] \\"
    )
    add(r"\midrule")
    first = True
    for case in CASES:
        for loc in ("corr", "none"):
            block = [r for r in rows if r.group == (truth, case, loc)]
            if not block:
                continue
            if not any(r.ok for r in block):
                if not first:
                    add(r"\addlinespace[2pt]")
                add(
                    rf"\multicolumn{{{NCOL_MASTER - 1}}}{{@{{}}l}}{{{CASE_LABEL[case]}, "
                    rf"{LOC_LABEL[loc]} --- \bad{{all methods failed (PALM truth diverged)}}}} \\"
                )
                first = False
                continue
            if not first:
                add(r"\midrule")
            add(
                rf"\multicolumn{{{NCOL_MASTER - 1}}}{{@{{}}l}}{{\emph{{{CASE_LABEL[case]}, "
                rf"{LOC_LABEL[loc]}}}}} \\"
            )
            first = False
            for row in block:
                if not row.ok:
                    continue
                param_cells = [cell(row, "angle", 2), cell(row, "vmag", 3)]
                zpool_cell = cell(row, "zpool", 2)
                if row.static_params:
                    param_cells = [
                        rf"\textit{{{c}}}$^{{\dagger}}$" for c in param_cells
                    ]
                    zpool_cell = rf"\textit{{{zpool_cell}}}$^{{\dagger}}$"
                add(
                    " & ".join(
                        [
                            row.rid,
                            row.label,
                            cell(row, "assim", 3),
                            cell(row, "valid", 3),
                            cell(row, "state", 3),
                            *param_cells,
                            cell(row, "chi2", 2),
                            zpool_cell,
                            cell(row, "zval", 2),
                            cell(row, "q", 3),
                            cell(row, "wall", 2),
                        ]
                    )
                    + r" \\"
                )
    add(r"\bottomrule")
    add(r"\end{tabular}")
    add(r"\end{adjustbox}")
    add(r"}")
    add("")
    return out


# --------------------------------------------------------------------------- #
# generalisation table (assim vs valid gap)
# --------------------------------------------------------------------------- #
def gap_table(rows: list[Row]) -> list[str]:
    out: list[str] = []
    add = out.append
    add(r"\newcommand{\CmpGapTable}{%")
    add(r"\footnotesize")
    add(r"\begin{adjustbox}{max width=\textwidth}")
    add(r"\begin{tabular}{@{}ll lrrr lrrr@{}}")
    add(r"\toprule")
    add(
        r"case & method & \multicolumn{4}{c}{localization on} & "
        r"\multicolumn{4}{c}{localization off} \\"
    )
    add(r"\cmidrule(lr){3-6}\cmidrule(lr){7-10}")
    add(r" & & ID & assim & valid & valid/assim & ID & assim & valid & valid/assim \\")
    add(r"\midrule")
    first = True
    for truth in ("pyudales", "pypalm"):
        tlabel = (
            "uDALES truth (matched)"
            if truth == "pyudales"
            else "PALM truth (model error)"
        )
        if not first:
            add(r"\midrule")
        add(rf"\multicolumn{{10}}{{@{{}}l}}{{\emph{{{tlabel}}}}} \\")
        first = False
        for case in CASES:
            block_any = any(r.ok for r in rows if r.truth == truth and r.case == case)
            if not block_any:
                continue
            printed_case = False
            for key, label, _, _ in METHODS:
                cells: list[str] = []
                have = False
                for loc in ("corr", "none"):
                    match = [
                        r
                        for r in rows
                        if r.group == (truth, case, loc) and r.key == key and r.ok
                    ]
                    if not match:
                        cells += ["--", "--", "--", "--"]
                        continue
                    have = True
                    row = match[0]
                    a, v = row.m["assim"], row.m["valid"]
                    cells += [
                        row.rid,
                        fmt(a, 3),
                        fmt(v, 3),
                        fmt(v / a if a else None, 2),
                    ]
                if not have:
                    continue
                lead = CASE_LABEL[case] if not printed_case else ""
                printed_case = True
                add(" & ".join([lead, label, *cells]) + r" \\")
    add(r"\bottomrule")
    add(r"\end{tabular}")
    add(r"\end{adjustbox}")
    add(r"}")
    add("")
    return out


# --------------------------------------------------------------------------- #
# observation-interval sweep: ESMDA vs filter smoothing
# --------------------------------------------------------------------------- #
def obs_sweep_table() -> list[str]:
    intervals = [("obs7p5", "7.5"), ("obs15", "15"), ("obs30", "30")]
    combos = [
        ("pyudales", "inflow_turb"),
        ("pyudales", "periodic"),
        ("pypalm", "inflow_turb"),
        ("pypalm", "periodic"),
    ]
    out: list[str] = []
    add = out.append
    add(r"\newcommand{\CmpObsIntervalTable}{%")
    add(r"\footnotesize")
    add(r"\begin{adjustbox}{max width=\textwidth}")
    add(r"\begin{tabular}{@{}lll rrr rrr rrr@{}}")
    add(r"\toprule")
    add(
        r"case & method & IDs & \multicolumn{3}{c}{$\alpha$ RMSE [deg]} & "
        r"\multicolumn{3}{c}{valid.\ RMSE} & \multicolumn{3}{c}{field $\Umag$ RMSE} \\"
    )
    add(r"\cmidrule(lr){4-6}\cmidrule(lr){7-9}\cmidrule(lr){10-12}")
    add(r" & & (7.5/15/30) & 7.5 & 15 & 30 & 7.5 & 15 & 30 & 7.5 & 15 & 30 \\")
    add(r"\midrule")
    for truth, case in combos:
        tl = "uDALES" if truth == "pyudales" else "PALM"
        printed = False
        for camp, label in (
            ("esmda", "ESMDA"),
            ("filter_smoothing", "Filter smoothing"),
        ):
            vals: dict[str, list[Any]] = {"angle": [], "valid": [], "state": []}
            ids: list[str] = []
            for obs, _ in intervals:
                name = f"{truth}_to_pyudales_w3_loccorrelation_{obs}_{case}"
                ids.append(RUN_ID[camp].get(name, "--"))
                path = EXP / camp / name / "run_summary.yaml"
                if not path.exists():
                    for v in vals.values():
                        v.append(None)
                    continue
                s = yaml.safe_load(path.read_text())
                vals["angle"].append(
                    get(s, "parameter_metrics", "inflow_angle", "rmse", "mean")
                )
                vals["valid"].append(
                    get(
                        s,
                        "sensor_metrics",
                        "validation",
                        "velocity_vector_rmse",
                        "mean",
                    )
                )
                vals["state"].append(
                    get(s, "state_metrics", "vel_magnitude_rmse", "mean")
                )
            lead = rf"{tl}, {CASE_LABEL[case]}" if not printed else ""
            printed = True
            add(
                " & ".join(
                    [lead, label, "/".join(ids)]
                    + [fmt(v, 2) for v in vals["angle"]]
                    + [fmt(v, 3) for v in vals["valid"]]
                    + [fmt(v, 3) for v in vals["state"]]
                )
                + r" \\"
            )
        add(r"\addlinespace[2pt]")
    out = out[:-1]  # drop the trailing addlinespace
    add = out.append
    add(r"\bottomrule")
    add(r"\end{tabular}")
    add(r"\end{adjustbox}")
    add(r"}")
    add("")
    return out


# --------------------------------------------------------------------------- #
# cost table
# --------------------------------------------------------------------------- #
STRUCTURE = {
    "E": ("3 (1/window)", "96", "4"),
    "Sn": ("180 (60/window)", "12", "1"),
    "Sr": ("180 (60/window)", "12", "1"),
    "J": ("180 (60/window)", "12", "1"),
    "H": ("183 (61/window)", "12 / 96", "$\\approx 4$"),
}


def cost_table(rows: list[Row]) -> list[str]:
    out: list[str] = []
    add = out.append
    add(r"\newcommand{\CmpCostTable}{%")
    add(r"\footnotesize")
    add(r"\begin{adjustbox}{max width=\textwidth}")
    add(r"\begin{tabular}{@{}l lll rrr r@{}}")
    add(r"\toprule")
    add(
        r"method & analyses & obs per & window & \multicolumn{3}{c}{wall time [h]} & "
        r"rel.\ to \\"
    )
    add(r"\cmidrule(lr){5-7}")
    add(r" & (total) & analysis & passes & min & mean & max & EnKF \\")
    add(r"\midrule")
    enkf_mean = None
    means: dict[str, float] = {}
    for key, _, _, _ in METHODS:
        walls = [r.m["wall"] for r in rows if r.key == key and r.ok and r.m.get("wall")]
        if walls:
            means[key] = sum(walls) / len(walls)
    enkf_mean = means.get("J")
    for key, label, _, _ in METHODS:
        walls = [r.m["wall"] for r in rows if r.key == key and r.ok and r.m.get("wall")]
        if not walls:
            continue
        analyses, nobs, passes = STRUCTURE[key]
        mean = sum(walls) / len(walls)
        rel = mean / enkf_mean if enkf_mean else None
        add(
            " & ".join(
                [
                    label,
                    analyses,
                    nobs,
                    passes,
                    fmt(min(walls), 2),
                    fmt(mean, 2),
                    fmt(max(walls), 2),
                    fmt(rel, 2) + r"$\times$",
                ]
            )
            + r" \\"
        )
    add(r"\bottomrule")
    add(r"\end{tabular}")
    add(r"\end{adjustbox}")
    add(r"}")
    add("")
    return out


# --------------------------------------------------------------------------- #
def main() -> None:
    rows = load_rows()
    mark_best(rows)

    lines: list[str] = [
        "% AUTO-GENERATED by experiments_report/scripts/extract_comparison.py",
        "% Do not edit by hand; re-run the script instead.",
        "",
    ]
    lines += master_table(rows, "pyudales", "CmpMasterUDALES")
    lines += master_table(rows, "pypalm", "CmpMasterPALM")
    lines += gap_table(rows)
    lines += obs_sweep_table()
    lines += cost_table(rows)

    # A few numbers the prose quotes, exported as macros so they cannot drift.
    def macro(
        name: str, truth: str, case: str, loc: str, key: str, col: str, digits: int
    ) -> str:
        match = [
            r for r in rows if r.group == (truth, case, loc) and r.key == key and r.ok
        ]
        value = match[0].m[col] if match else None
        return rf"\newcommand{{\{name}}}{{{fmt(value, digits)}}}"

    lines.append("% --- numbers quoted in the prose --------------------------------")
    lines.append(
        macro("CmpFlagAngleE", "pyudales", "inflow_turb", "corr", "E", "angle", 2)
    )
    lines.append(
        macro("CmpFlagAngleJ", "pyudales", "inflow_turb", "corr", "J", "angle", 2)
    )
    lines.append(
        macro("CmpFlagAngleH", "pyudales", "inflow_turb", "corr", "H", "angle", 2)
    )
    lines.append(
        macro("CmpFlagValidE", "pyudales", "inflow_turb", "corr", "E", "valid", 3)
    )
    lines.append(
        macro("CmpFlagValidJ", "pyudales", "inflow_turb", "corr", "J", "valid", 3)
    )
    lines.append(
        macro("CmpFlagValidH", "pyudales", "inflow_turb", "corr", "H", "valid", 3)
    )
    lines.append(macro("CmpPerAssimE", "pyudales", "periodic", "corr", "E", "assim", 3))
    lines.append(macro("CmpPerAssimH", "pyudales", "periodic", "corr", "H", "assim", 3))
    lines.append(macro("CmpPerVmagJ", "pyudales", "periodic", "corr", "J", "vmag", 3))
    lines.append(macro("CmpPerVmagH", "pyudales", "periodic", "corr", "H", "vmag", 3))
    lines.append("")

    OUT.write_text("\n".join(lines) + "\n")

    dump = [
        {
            "method": r.key,
            "label": r.label,
            "truth": r.truth,
            "case": r.case,
            "loc": r.loc,
            "name": r.name,
            "campaign": r.campaign,
            "ok": r.ok,
            "static_params": getattr(r, "static_params", False),
            **{k: v for k, v in r.m.items() if not k.endswith("_b") and k != "onp"},
        }
        for r in rows
    ]
    DUMP.write_text(json.dumps(dump, indent=1) + "\n")

    n_ok = sum(1 for r in rows if r.ok)
    print(f"wrote {OUT} ({len(lines)} lines)")
    print(f"wrote {DUMP} ({n_ok}/{len(rows)} runs with a summary)")
    print("\n--- matched cells ---")
    for row in rows:
        if not row.ok:
            print(
                f"  {row.truth:9s} {row.case:12s} {row.loc:5s} {row.key:3s}  FAILED  {row.name}"
            )
            continue
        m = row.m
        chi2 = "--" if m["chi2"] is None else f"{m['chi2']:.2f}"
        hitq = "--" if m["q"] is None else f"{m['q']:.3f}"
        print(
            f"  {row.truth:9s} {row.case:12s} {row.loc:5s} {row.key:3s} "
            f"as={m['assim']:.3f} va={m['valid']:.3f} st={m['state']:.3f} "
            f"ang={m['angle']:.2f} vm={m['vmag']:.3f} "
            f"chi2={chi2:>6s} zp={m['zpool']:.2f} zv={m['zval']:.2f} "
            f"q={hitq:>6s} h={m['wall']:.2f}"
        )


if __name__ == "__main__":
    main()
