"""Technique-effect dumbbell figure for the Conclusions section.

Reads scripts/comparison_data.json (written by extract_comparison.py) plus the
ESMDA / filter-smoothing obs-interval runs, and draws, per technique, how
switching it on moves validation RMSE, inflow-angle RMSE and validation-sensor
calibration (std of z).  Green = improvement, red = regression.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import yaml

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
EXP = ROOT.parent / "presentations/isda_new/experiments"
OUT = ROOT / "figures/conclusions"
OUT.mkdir(parents=True, exist_ok=True)

rows = json.loads((HERE / "comparison_data.json").read_text())
idx = {(r["method"], r["truth"], r["case"], r["loc"]): r for r in rows if r["ok"]}

CASES = [("inflow", "laminar"), ("inflow_turb", "turbulent"), ("periodic", "periodic")]
MLABEL = {
    "E": "ESMDA",
    "Sn": "EnKF state",
    "Sr": "EnKF state+RTPS",
    "J": "EnKF joint+RTPS",
    "H": "Filter smoothing",
}


def summary(campaign: str, name: str) -> dict:
    return yaml.safe_load((EXP / campaign / name / "run_summary.yaml").read_text())


def metrics_from_summary(s: dict) -> dict:
    pm = s["parameter_metrics"]
    sv = s["sensor_statistics"]["validation"]["posterior"]
    zs = [sv[k]["z_score"]["std"] for k in ("mean_u", "mean_v", "mean_w")]
    return {
        "valid": s["sensor_metrics"]["validation"]["velocity_vector_rmse"]["mean"],
        "angle": pm["inflow_angle"]["rmse"]["mean"],
        "zval": sum(zs) / len(zs),
    }


# (technique, [(row label, off-run, on-run, params_estimated)])
pairs: list[tuple[str, list[tuple[str, dict, dict, bool]]]] = []

# 1. Localization off -> on
loc_rows = []
for case, clab in CASES:
    for m in ("E", "Sr", "J", "H"):
        a, b = idx.get((m, "pyudales", case, "none")), idx.get((m, "pyudales", case, "corr"))
        if a and b:
            loc_rows.append((f"{clab} · {MLABEL[m]}", a, b, m in ("E", "J", "H")))
pairs.append(("Localization (off → on)", loc_rows))

# 2. RTPS inflation none -> RTPS (state-only EnKF, no loc)
infl_rows = []
for case, clab in CASES:
    a, b = idx.get(("Sn", "pyudales", case, "none")), idx.get(("Sr", "pyudales", case, "none"))
    if a and b:
        infl_rows.append((f"{clab} · EnKF state", a, b, False))
pairs.append(("RTPS inflation (off → on)", infl_rows))

# 3. Joint parameter estimation (state+RTPS -> joint+RTPS)
joint_rows = []
for case, clab in CASES:
    for loc, ll in (("none", "no loc"), ("corr", "loc")):
        a, b = idx.get(("Sr", "pyudales", case, loc)), idx.get(("J", "pyudales", case, loc))
        if a and b:
            joint_rows.append((f"{clab} · {ll}", a, b, True))
pairs.append(("Joint parameter estimation (state → joint EnKF)", joint_rows))

# 4. Hybrid: add EnKF state update to ESMDA (ESMDA -> filter smoothing)
hyb_rows = []
for case, clab in CASES:
    for loc, ll in (("none", "no loc"), ("corr", "loc")):
        a, b = idx.get(("E", "pyudales", case, loc)), idx.get(("H", "pyudales", case, loc))
        if a and b:
            hyb_rows.append((f"{clab} · {ll}", a, b, True))
pairs.append(("Add EnKF state update to ESMDA (→ filter smoothing)", hyb_rows))

# 5. Observation interval 7.5 s -> 30 s (ESMDA and filter smoothing, loc)
obs_rows = []
for camp, m in (("esmda", "E"), ("filter_smoothing", "H")):
    for case, clab in (("inflow_turb", "turbulent"), ("periodic", "periodic")):
        try:
            a = metrics_from_summary(summary(camp, f"pyudales_to_pyudales_w3_loccorrelation_obs7p5_{case}"))
            b = metrics_from_summary(summary(camp, f"pyudales_to_pyudales_w3_loccorrelation_obs30_{case}"))
        except FileNotFoundError:
            continue
        obs_rows.append((f"{clab} · {MLABEL[m]}", a, b, True))
pairs.append(("Observation interval (7.5 s → 30 s)", obs_rows))

# 6. Model error: uDALES truth -> PALM truth (same method, loc)
me_rows = []
for case, clab in (("inflow_turb", "turbulent"), ("periodic", "periodic")):
    for m in ("E", "J", "H"):
        a, b = idx.get((m, "pyudales", case, "corr")), idx.get((m, "pypalm", case, "corr"))
        if a and b:
            me_rows.append((f"{clab} · {MLABEL[m]}", a, b, True))
pairs.append(("Model error (uDALES truth → PALM truth)", me_rows))

# ---------------------------------------------------------------- drawing
GREEN, RED, GREY = "#2a9d3c", "#c8322b", "#8a8a8a"
cols = [
    ("valid", "Validation vector RMSE [m/s]", "lower is better", True),
    ("angle", "Inflow-angle RMSE [deg]", "lower is better", True),
    ("zval", "Calibration: std(z) at validation sensors", "closer to 1 is better", False),
]
n_rows = sum(len(p[1]) + 1 for p in pairs)
fig, axes = plt.subplots(1, 3, figsize=(15, 0.36 * n_rows + 1.2), sharey=True)

y = 0
yticks, ylabels, group_lines = [], [], []
for tech, prs in pairs:
    yticks.append(y)
    ylabels.append(tech)
    group_lines.append(y)
    y -= 1
    for lab, a, b, has_par in prs:
        yticks.append(y)
        ylabels.append("    " + lab)
        for ax, (key, _, _, lower) in zip(axes, cols):
            if key == "angle" and not has_par:
                ax.text(0.5, y, "n/a (parameters not estimated)", va="center", ha="center",
                        fontsize=7, color=GREY, transform=ax.get_yaxis_transform())
                continue
            va, vb = a[key], b[key]
            if va is None or vb is None:
                continue
            if lower:
                better = vb < va
            else:
                better = abs(vb - 1) < abs(va - 1)
            c = GREEN if better else RED
            ax.annotate("", xy=(vb, y), xytext=(va, y),
                        arrowprops=dict(arrowstyle="-|>", color=c, lw=1.8, shrinkA=3, shrinkB=1))
            ax.plot(va, y, "o", color="white", mec=GREY, ms=6, zorder=3)
            ax.plot(vb, y, "o", color=c, mec=c, ms=6, zorder=3)
            pct = 100 * (vb - va) / va
            ax.text(max(va, vb), y, f"  {pct:+.0f}%", va="center", ha="left", fontsize=7, color=c)
        y -= 1
    y -= 0.4

for ax, (key, title, sub, lower) in zip(axes, cols):
    ax.set_title(f"{title}\n({sub})", fontsize=10)
    ax.set_xscale("log")
    ax.grid(axis="x", alpha=0.3, which="both")
    for gy in group_lines:
        ax.axhline(gy + 0.5, color="black", lw=0.6)
    if key == "zval":
        ax.axvline(1.0, color="black", ls="--", lw=0.8)
    ax.tick_params(axis="x", labelsize=8)
axes[0].set_yticks(yticks)
axes[0].set_yticklabels(ylabels, fontsize=8)
for t, l in zip(axes[0].get_yticklabels(), ylabels):
    if not l.startswith("    "):
        t.set_fontweight("bold")
axes[0].set_ylim(y, 1)
fig.suptitle("Effect of each technique / condition on accuracy and calibration "
             "(open circle = before, filled = after; green = better, red = worse)",
             fontsize=11)
fig.tight_layout(rect=(0, 0, 1, 0.96))
fig.savefig(OUT / "technique_effects.png", dpi=150)
print("wrote", OUT / "technique_effects.png")
