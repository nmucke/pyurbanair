"""Summary figures for the cross-method comparison section.

Consumes ``experiments_report/scripts/comparison_data.json`` (written by
``extract_comparison.py``) plus the per-run PNGs of the three campaigns.

Run from the repo root::

    cd /Users/ntmucke/Code/pyurbanair
    pixi run -e dev python experiments_report/scripts/extract_comparison.py
    pixi run -e dev python experiments_report/scripts/make_comparison_figures.py

Writes into ``experiments_report/figures/comparison/``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
import figlib as fl  # noqa: E402  (shared crop / compose library)

REPO = Path(__file__).resolve().parents[2]
EXP = REPO / "presentations" / "isda_new" / "experiments"
DATA = REPO / "experiments_report" / "scripts" / "comparison_data.json"
OUTDIR = REPO / "experiments_report" / "figures" / "comparison"
OUTDIR.mkdir(parents=True, exist_ok=True)

PRIOR_ANGLE = 19.29
PRIOR_VMAG = 0.919
Z_EXPECTED = 1.03

# Method names fixed by STYLE_SPEC section 5; used verbatim in every figure.
METHOD_ORDER = ["E", "Sn", "Sr", "J", "H"]
METHOD_LABEL = {
    "E": "ESMDA",
    "Sn": "EnKF state, no infl.",
    "Sr": "EnKF state+RTPS",
    "J": "EnKF joint+RTPS",
    "H": "Filter smoothing",
}
METHOD_COLOR = {
    "E": "#1f77b4",
    "Sn": "#bdbdbd",
    "Sr": "#ff7f0e",
    "J": "#d62728",
    "H": "#2ca02c",
}
METHOD_MARKER = {"E": "o", "Sn": "v", "Sr": "s", "J": "D", "H": "^"}

CASE_SHORT = {"inflow": "laminar", "inflow_turb": "turb", "periodic": "periodic"}
CASE_MARKER = {"inflow": "o", "inflow_turb": "s", "periodic": "^"}
LOC_SHORT = {"corr": "loc", "none": "no loc"}

plt.rcParams.update(
    {
        "figure.dpi": 160,
        "savefig.dpi": 160,
        "font.size": 8,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.linestyle": ":",
        "axes.axisbelow": True,
        "legend.frameon": False,
    }
)


def load() -> list[dict[str, Any]]:
    rows = json.loads(DATA.read_text())
    return [r for r in rows if r["ok"]]


def groups_for(rows: list[dict[str, Any]], truth: str) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    for case in ("inflow", "inflow_turb", "periodic"):
        for loc in ("corr", "none"):
            if any(
                r["truth"] == truth and r["case"] == case and r["loc"] == loc
                for r in rows
            ):
                out.append((case, loc))
    return out


def pick(rows, truth, case, loc, key) -> dict[str, Any] | None:
    for r in rows:
        if (r["truth"], r["case"], r["loc"], r["method"]) == (truth, case, loc, key):
            return r
    return None


def grouped_bars(
    ax,
    rows,
    truth,
    column,
    *,
    skip_static: bool = False,
    log: bool = False,
    title: str = "",
    ylabel: str = "",
) -> None:
    """One panel of grouped bars: x = (case, loc) cells, hue = method."""
    cells = groups_for(rows, truth)
    n = len(METHOD_ORDER)
    width = 0.8 / n
    for j, key in enumerate(METHOD_ORDER):
        xs, ys = [], []
        for i, (case, loc) in enumerate(cells):
            row = pick(rows, truth, case, loc, key)
            if row is None or row.get(column) is None:
                continue
            if skip_static and row["static_params"]:
                continue
            xs.append(i - 0.4 + width * (j + 0.5))
            ys.append(row[column])
        if xs:
            ax.bar(
                xs,
                ys,
                width=width * 0.92,
                color=METHOD_COLOR[key],
                label=METHOD_LABEL[key],
            )
    ax.set_xticks(range(len(cells)))
    ax.set_xticklabels(
        [f"{CASE_SHORT[c]}\n{LOC_SHORT[l]}" for c, l in cells], fontsize=7
    )
    if log:
        ax.set_yscale("log")
    ax.set_title(title, fontsize=8.5)
    ax.set_ylabel(ylabel, fontsize=8)


def legend_handles(keys=METHOD_ORDER) -> list[Patch]:
    return [Patch(facecolor=METHOD_COLOR[k], label=METHOD_LABEL[k]) for k in keys]


# --------------------------------------------------------------------------- #
# Fig 1: sensor / field accuracy
# --------------------------------------------------------------------------- #
def fig_accuracy(rows) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(9.0, 5.4), constrained_layout=True)
    for i, truth in enumerate(("pyudales", "pypalm")):
        tl = (
            "uDALES truth (matched model)"
            if truth == "pyudales"
            else "PALM truth (model error)"
        )
        grouped_bars(
            axes[i, 0],
            rows,
            truth,
            "valid",
            title=f"{tl} --- held-out (validation) sensors",
            ylabel="velocity-vector RMSE [m/s]",
        )
        grouped_bars(
            axes[i, 1],
            rows,
            truth,
            "state",
            title=f"{tl} --- domain field",
            ylabel=r"field $|U|$ RMSE [m/s]",
        )
    fig.legend(
        handles=legend_handles(),
        loc="lower center",
        ncol=5,
        fontsize=7.5,
        bbox_to_anchor=(0.5, -0.035),
    )
    fig.savefig(OUTDIR / "cmp_accuracy.png", bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Fig 2: parameter recovery
# --------------------------------------------------------------------------- #
def _lollipop(ax, rows, truth, column, baseline, ylabel, title) -> None:
    """Log-axis lollipops measured *from the climatological prior*.

    On a logarithmic axis a bar's length is meaningless (the baseline is the
    arbitrary axis minimum), so each marker is stemmed to the static-prior
    value instead: down = better than climatology, up = worse.
    """
    cells = groups_for(rows, truth)
    keys = ["E", "J", "H"]
    width = 0.7 / len(keys)
    for j, key in enumerate(keys):
        for i, (case, loc) in enumerate(cells):
            row = pick(rows, truth, case, loc, key)
            if row is None or row.get(column) is None or row["static_params"]:
                continue
            x = i - 0.35 + width * (j + 0.5)
            y = row[column]
            ax.vlines(
                x, baseline, y, color=METHOD_COLOR[key], lw=3.2, alpha=0.75, zorder=2
            )
            ax.plot(
                x,
                y,
                marker=METHOD_MARKER[key],
                ms=5.5,
                color=METHOD_COLOR[key],
                markeredgecolor="k",
                markeredgewidth=0.4,
                zorder=3,
            )
    ax.axhline(baseline, color="k", ls="--", lw=0.9, zorder=1)
    ax.set_yscale("log")
    ax.set_xticks(range(len(cells)))
    ax.set_xticklabels(
        [f"{CASE_SHORT[c]}\n{LOC_SHORT[l]}" for c, l in cells], fontsize=7
    )
    ax.set_xlim(-0.6, len(cells) - 0.4)
    ax.set_ylabel(ylabel, fontsize=8)
    ax.set_title(title, fontsize=8.5)


def fig_parameters(rows) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(9.0, 5.4), constrained_layout=True)
    for i, truth in enumerate(("pyudales", "pypalm")):
        tl = (
            "uDALES truth (matched)"
            if truth == "pyudales"
            else "PALM truth (model error)"
        )
        _lollipop(
            axes[i, 0],
            rows,
            truth,
            "angle",
            PRIOR_ANGLE,
            r"$\alpha$ RMSE [deg]",
            rf"{tl} --- inflow angle $\alpha$",
        )
        axes[i, 0].text(
            0.006,
            PRIOR_ANGLE * 1.05,
            "static prior (state-only filters stay on this line)",
            fontsize=6.5,
            transform=axes[i, 0].get_yaxis_transform(),
        )
        _lollipop(
            axes[i, 1],
            rows,
            truth,
            "vmag",
            PRIOR_VMAG,
            r"$|U|$ RMSE [m/s]",
            rf"{tl} --- velocity magnitude $|U|$",
        )
        axes[i, 1].text(
            0.006,
            PRIOR_VMAG * 1.05,
            "static prior",
            fontsize=6.5,
            transform=axes[i, 1].get_yaxis_transform(),
        )
    handles = [
        Line2D(
            [],
            [],
            marker=METHOD_MARKER[k],
            color=METHOD_COLOR[k],
            lw=3.2,
            markeredgecolor="k",
            markeredgewidth=0.4,
            label=METHOD_LABEL[k],
        )
        for k in ("E", "J", "H")
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=3,
        fontsize=7.5,
        bbox_to_anchor=(0.5, -0.035),
    )
    fig.savefig(OUTDIR / "cmp_parameters.png", bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Fig 3: calibration
# --------------------------------------------------------------------------- #
def fig_calibration(rows) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(9.0, 6.2), constrained_layout=True)
    specs = [
        ("chi2", r"innovation $\chi^2$ (1 = calibrated)", 1.0, False),
        ("zval", r"validation-sensor $z$ std (1.03 = calibrated)", Z_EXPECTED, False),
        ("zpool", r"pooled parameter $z$ std (1.03 = calibrated)", Z_EXPECTED, True),
    ]
    order = [(t, c, l) for t in ("pyudales", "pypalm") for c, l in groups_for(rows, t)]
    labels = [
        f"{'UD' if t == 'pyudales' else 'PA'}\n{CASE_SHORT[c]}\n{LOC_SHORT[l]}"
        for t, c, l in order
    ]
    n = len(METHOD_ORDER)
    width = 0.8 / n
    for ax, (column, title, target, skip_static) in zip(axes, specs):
        for j, key in enumerate(METHOD_ORDER):
            xs, ys = [], []
            for i, (t, c, l) in enumerate(order):
                row = pick(rows, t, c, l, key)
                if row is None or row.get(column) is None:
                    continue
                if skip_static and row["static_params"]:
                    continue
                xs.append(i - 0.4 + width * (j + 0.5))
                ys.append(max(row[column], 1e-3))
            if xs:
                ax.bar(xs, ys, width=width * 0.92, color=METHOD_COLOR[key])
        ax.axhline(target, color="k", ls="--", lw=0.9)
        ax.set_yscale("log")
        ax.set_xticks(range(len(order)))
        ax.set_xticklabels(labels, fontsize=6.5)
        ax.set_xlim(-0.6, len(order) - 0.4)
        ax.set_title(title, fontsize=8.5)
    fig.legend(
        handles=legend_handles(),
        loc="lower center",
        ncol=5,
        fontsize=7.5,
        bbox_to_anchor=(0.5, -0.03),
    )
    fig.savefig(OUTDIR / "cmp_calibration.png", bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Fig 4: generalisation (fit vs held-out)
# --------------------------------------------------------------------------- #
def fig_generalisation(rows) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 4.0), constrained_layout=True)
    for ax, truth in zip(axes, ("pyudales", "pypalm")):
        sub = [r for r in rows if r["truth"] == truth]
        for r in sub:
            ax.scatter(
                r["assim"],
                r["valid"],
                marker=METHOD_MARKER[r["method"]],
                s=46,
                facecolor=METHOD_COLOR[r["method"]],
                edgecolor="k" if r["loc"] == "corr" else "none",
                linewidth=0.7,
                zorder=3,
            )
        lo = min(min(r["assim"] for r in sub), min(r["valid"] for r in sub)) * 0.7
        hi = max(max(r["assim"] for r in sub), max(r["valid"] for r in sub)) * 1.4
        ax.plot([lo, hi], [lo, hi], "k--", lw=0.9, zorder=1)
        ax.text(hi * 0.62, hi * 0.72, "1:1", fontsize=7, rotation=45)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_xlabel("assimilated-sensor vector RMSE [m/s]")
        ax.set_ylabel("held-out-sensor vector RMSE [m/s]")
        ax.set_title(
            (
                "uDALES truth (matched)"
                if truth == "pyudales"
                else "PALM truth (model error)"
            ),
            fontsize=8.5,
        )
    handles = [
        Line2D(
            [],
            [],
            ls="",
            marker=METHOD_MARKER[k],
            color=METHOD_COLOR[k],
            markersize=6,
            label=METHOD_LABEL[k],
        )
        for k in METHOD_ORDER
    ] + [
        Line2D(
            [],
            [],
            ls="",
            marker="o",
            markerfacecolor="w",
            markeredgecolor="k",
            markersize=6,
            label="black edge = correlation localization",
        ),
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=3,
        fontsize=7.5,
        bbox_to_anchor=(0.5, -0.14),
    )
    fig.savefig(OUTDIR / "cmp_generalisation.png", bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------- #
# PIL composites: one column per method, cropped with figlib
# --------------------------------------------------------------------------- #
CAMPAIGN_OF = {"E": "esmda", "F": "filtering", "H": "filter_smoothing"}

# (method key, run ID) columns of every composite, grouped by case.  The method
# key selects the label, the ID selects the run directory.
#
# Three columns per row, not four: at \textwidth a fourth column pushes the
# axis tick labels below ~3 pt.  The three state-only EnKF runs therefore get
# one figure of their own (VALID_STATEONLY) instead of a column in each case.
VALID_FIGS: list[tuple[str, list[tuple[str, str]]]] = [
    # (output stem, columns)
    ("cmp_valid_ud_inflow", [("E", "E11"), ("J", "F7"), ("H", "H11")]),
    ("cmp_valid_ud_inflow_turb", [("E", "E12"), ("J", "F9"), ("H", "H12")]),
    ("cmp_valid_ud_periodic", [("E", "E13"), ("J", "F11"), ("H", "H13")]),
    ("cmp_valid_palm_inflow_turb", [("E", "E2"), ("J", "F2"), ("H", "H2")]),
    ("cmp_valid_palm_periodic", [("E", "E3"), ("J", "F3"), ("H", "H3")]),
]

# The state-only EnKF companions of the three uDALES cases above.
VALID_STATEONLY: list[tuple[str, str]] = [("Sr", "F8"), ("Sr", "F10"), ("Sr", "F12")]
STATEONLY_CASE = {"F8": "laminar inflow", "F10": "turbulent inflow", "F12": "periodic"}

# localization off, two cases in one figure
VALID_NOLOC: list[tuple[str, list[tuple[str, str]]]] = [
    (r"turbulent inflow, localization off", [("E", "E19"), ("J", "F16"), ("H", "H19")]),
    (r"periodic, localization off", [("E", "E20"), ("J", "F19"), ("H", "H20")]),
]

# Parameter figures: state-only runs are omitted (their parameter ensemble is
# never updated, so every panel would be a flat prior).
PARAM_FIGS: list[tuple[str, str, list[tuple[str, str, list[tuple[str, str]]]]]] = [
    (
        "ud",
        "uDALES truth (matched model), localization on",
        [
            ("inflow", "laminar inflow", [("E", "E11"), ("J", "F7"), ("H", "H11")]),
            (
                "inflow_turb",
                "turbulent inflow",
                [("E", "E12"), ("J", "F9"), ("H", "H12")],
            ),
            ("periodic", "periodic", [("E", "E13"), ("J", "F11"), ("H", "H13")]),
        ],
    ),
    (
        "palm",
        "PALM truth (model error), localization on",
        [
            (
                "inflow_turb",
                "turbulent inflow",
                [("E", "E2"), ("J", "F2"), ("H", "H2")],
            ),
            ("periodic", "periodic", [("E", "E3"), ("J", "F3"), ("H", "H3")]),
        ],
    ),
]

_VALID_RMSE: dict[str, float] = {}


def index_runs(rows) -> None:
    """``{run ID: validation vector RMSE}`` so labels can carry the headline number."""
    ids = {m: {name: rid for rid, name in fl.run_ids(m).items()} for m in fl.METHODS}
    for r in rows:
        rid = ids[r["campaign"]].get(r["name"])
        if rid and r.get("valid") is not None:
            _VALID_RMSE[rid] = r["valid"]


def _rdir(rid: str) -> Path:
    return fl.run_dir(CAMPAIGN_OF[rid[0]], rid)


def _col_label(key: str, rid: str, with_rmse: bool = True) -> str:
    text = f"{METHOD_LABEL[key]} · {rid}"
    if with_rmse and rid in _VALID_RMSE:
        text += f" · valid {_VALID_RMSE[rid]:.3f}"
    return text


def _row(cols, crop, with_rmse: bool = True) -> Image.Image:
    imgs = [crop(_rdir(rid)) for _, rid in cols]
    return fl.hstack(imgs, [_col_label(k, r, with_rmse) for k, r in cols])


def fig_valid_composites() -> None:
    """One figure per case: the same held-out sensors under every method."""
    for stem, cols in VALID_FIGS:
        fl.save(_row(cols, fl.crop_validation), OUTDIR / f"{stem}.png")

    imgs = [fl.crop_validation(_rdir(rid)) for _, rid in VALID_STATEONLY]
    labels = [
        f"{STATEONLY_CASE[rid]} · {_col_label(k, rid)}" for k, rid in VALID_STATEONLY
    ]
    fl.save(fl.hstack(imgs, labels), OUTDIR / "cmp_valid_ud_stateonly.png")

    rows = [_row(cols, fl.crop_validation) for _, cols in VALID_NOLOC]
    fl.save(
        fl.vstack(rows, [t for t, _ in VALID_NOLOC], gap=34),
        OUTDIR / "cmp_valid_ud_noloc.png",
        max_width=1600,
    )


def fig_param_composites() -> None:
    """Parameter-error time series and final-knot marginals, per case row."""
    for stem, _truth_label, cases in PARAM_FIGS:
        err = [_row(cols, fl.crop_param_error, with_rmse=False) for _, _, cols in cases]
        fl.save(
            fl.vstack(err, [t for _, t, _ in cases], gap=34),
            OUTDIR / f"cmp_param_error_{stem}.png",
            max_width=1600,
        )
        mar = [_row(cols, fl.crop_marginals, with_rmse=False) for _, _, cols in cases]
        fl.save(
            fl.vstack(mar, [t for _, t, _ in cases], gap=34),
            OUTDIR / f"cmp_marginals_{stem}.png",
            max_width=1600,
        )


# --------------------------------------------------------------------------- #
def main() -> None:
    rows = load()
    index_runs(rows)
    fig_accuracy(rows)
    fig_parameters(rows)
    fig_calibration(rows)
    fig_generalisation(rows)
    fig_valid_composites()
    fig_param_composites()
    for p in sorted(OUTDIR.glob("*.png")):
        w, h = Image.open(p).size
        print(f"  {p.relative_to(REPO)}  {w}x{h}  (aspect h/w = {h / w:.2f})")


if __name__ == "__main__":
    main()
