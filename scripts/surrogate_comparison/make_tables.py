"""Write the LaTeX tables of a surrogate comparison from `metrics.nc` and `metrics.yaml`.

    python scripts/surrogate_comparison/make_tables.py <run dir> [key=value ...]

The run dir is what scripts/surrogate_comparison/compute_metrics.py wrote into;
the overrides are applied to its config.yaml (e.g. `metrics.horizons=[1,2,5]`,
`display_names={dft_s: Ours}`). Each table is a booktabs `table` environment
with caption and label, ready to `\\input` (needs only `\\usepackage{booktabs}`).
Rows: the models in config order under their `display_names`, then, below a
rule, persistence and, where it is the reference, the truth. Everything is dimensionless per sample (u_ref: the
canopy-top plane mean of the truth's time-mean u; H: the mean building height;
T: the truth's integral time scale, T-bar its mean over the samples) and then
averaged over the samples.

The best MODEL value per column is bold (persistence and truth never are): the smallest error, the largest ACC / ACC horizon / speedup, the
closest to 0 dB, to 1 (divergence ratio) or to the truth (Lambda/H), compared
at the printed precision, so ties are all bold. Each column has one number of
decimals, three significant digits of its typical value (two for dB). "--": not a number
(an unstable rollout); empty: not defined for that row.

Outputs, in the run dir (the .csv holds the same table at full precision):
  table_accuracy.tex/.csv
    rmse_h<h>, rmse_window  RMSE/u_ref over the fluid cells (metrics.nc `rmse`),
                            RMS over the leads t <= h T-bar for each
                            `metrics.horizons` inside the window, and over the
                            whole window (cf. metrics.yaml `rmse`)
    acc_h<h>, acc_window    the anomaly correlation about the truth's time mean
                            (`acc`), mean over the same leads
    acc_horizon             first lead where ACC < `metrics.acc_threshold`, in
                            t / T-bar (a sample whose ACC never drops counts
                            as the window's end); ">=" when any sample is
                            censored so (`acc_horizon_censored` > 0): a lower bound
    vorticity_rmse          window RMSE of the vorticity, x H / u_ref
    rmse_pedestrian         window RMSE/u_ref at the pedestrian level
  table_physics.tex/.csv
    mean_u_rmse, tke_rmse, profile_rmse_uw
                            RMSE of the time-mean u field / u_ref, of the
                            resolved-TKE field / u_ref^2 and of the -<u'w'>
                            profile / u_ref^2
    spectra_db_u_mid, spectra_db_u_near_cutoff
                            10 log10 of the spanwise u' spectrum energy against
                            the truth's at the canopy top, in the mid and
                            near-cutoff wavelength bands
    integral_length         spanwise integral length Lambda_y / H at the canopy top
    wasserstein_u           W2 distance of the u' distribution at the canopy top / u_ref
    divergence_ratio        window mean RMS divergence / the truth's
  table_cost.tex/.csv (models only)
    n_parameters            parameters (all of model.parameters()), millions
    ms_per_step             inference wall time per lead step, ms (the first
                            step of each rollout excluded)
    speedup                 `metrics.solver_seconds_per_step` / seconds per step
                            (only when compute_metrics.py wrote it)
"""

from __future__ import annotations

import argparse
import csv
import math
import pathlib
from typing import Any, NamedTuple

import numpy as np
import xarray
import yaml
from omegaconf import DictConfig, OmegaConf

PERSISTENCE, TRUTH, HALVES = "persistence", "truth", "truth_halves"
ROW_NAMES = {
    PERSISTENCE: "Persistence",
    TRUTH: "Truth (LES)",
}
NORMALISATION = (
    r"Velocities are scaled by $u_\mathrm{ref}$ (the canopy-top plane mean of the "
    r"time-mean streamwise velocity), lengths by the mean building height $H$ and "
    r"lead times by the mean integral time scale $\bar{T}$ of the truth, per sample "
    r"before averaging over the samples."
)
BOLD = r"Bold: the best model per column; ``--'': unstable rollout."


class Column(NamedTuple):
    """One table column: `values` per row source (None: not defined for it,
    NaN: not a number); `best` is "min", "max", a target value (closest wins)
    or None (no bold); `prefixes` per row source are printed before the number."""

    group: str
    head: str
    name: str
    values: dict[str, float | None]
    best: str | float | None
    prefixes: dict[str, str] | None = None
    digits: int | None = None  # None: decimals() of the column


def run(run_dir: pathlib.Path, overrides: list[str] | None = None) -> None:
    run_dir = pathlib.Path(run_dir)
    cfg = OmegaConf.merge(
        OmegaConf.load(run_dir / "config.yaml"), OmegaConf.from_dotlist(overrides or [])
    )
    assert isinstance(cfg, DictConfig)
    scalars = yaml.safe_load((run_dir / "metrics.yaml").read_text())
    with xarray.open_dataset(run_dir / "metrics.nc") as ds:
        metrics = ds.load()
    # metrics.yaml keeps the config order of the models
    models = [s for s in scalars if s not in (PERSISTENCE, TRUTH, HALVES, "reference")]
    names = ROW_NAMES | {
        str(k): str(v) for k, v in (cfg.get("display_names") or {}).items()
    }

    _accuracy(metrics, scalars, models, cfg, names, run_dir)
    _physics(scalars, models, names, run_dir)
    _cost(scalars, models, names, run_dir)
    print(
        f"Saved table_accuracy, table_physics and table_cost (.tex, .csv) in {run_dir}"
    )


def _accuracy(
    metrics: xarray.Dataset,
    scalars: dict,
    models: list[str],
    cfg: DictConfig,
    names: dict[str, str],
    run_dir: pathlib.Path,
) -> None:
    rows = [*models, PERSISTENCE]
    over_t = metrics["lead_over_T"].values
    window = float(over_t[-1])
    horizons = [
        float(h)
        for h in cfg.metrics.horizons
        if float(h) < window and (over_t <= float(h)).any()
    ]
    t_mean = float(metrics.attrs["T_mean"])
    group = {"rmse": r"$\mathrm{RMSE}/u_\mathrm{ref}$", "acc": "ACC"}

    def lead_means(var: str, best: str) -> list[Column]:
        # RMS (rmse) or mean (acc) of the curve over the leads t <= h T-bar,
        # one decimals for all h
        power = 2 if var == "rmse" else 1
        out = []
        for h in [*horizons, None]:
            keep = over_t <= (h if h is not None else np.inf)
            head = rf"$t \le {h:g}\,\bar{{T}}$" if h is not None else "window"
            values: dict[str, float | None] = {
                s: float(
                    np.mean(metrics[var].sel(source=s).values[keep] ** power)
                    ** (1 / power)
                )
                for s in rows
                if s in metrics.source
            }
            tag = f"h{h:g}" if h is not None else "window"
            out.append(Column(group[var], head, f"{var}_{tag}", values, best))
        digits = decimals([v for c in out for v in c.values.values() if v is not None])
        return [c._replace(digits=digits) for c in out]

    horizon = {s: _mean(scalars, s, "acc_horizon_seconds") for s in rows}
    censored = {  # a lower bound: some sample's ACC never dropped below
        s: r"$\geq$"
        for s in rows
        if (_mean(scalars, s, "acc_horizon_censored") or 0) > 0
    }
    threshold = float(cfg.metrics.acc_threshold)
    columns = [
        *lead_means("rmse", "min"),
        *lead_means("acc", "max"),
        Column(
            "",
            r"$t_\mathrm{ACC}/\bar{T}$",
            "acc_horizon",
            {s: v / t_mean if v is not None else None for s, v in horizon.items()},
            "max",
            prefixes=censored,
        ),
        Column(
            "",
            r"$\mathrm{RMSE}_\omega H/u_\mathrm{ref}$",
            "vorticity_rmse",
            {s: _mean(scalars, s, "vorticity_rmse") for s in rows},
            "min",
        ),
        Column(
            "",
            r"$\mathrm{RMSE}_\mathrm{ped}/u_\mathrm{ref}$",
            "rmse_pedestrian",
            {s: _mean(scalars, s, "rmse_pedestrian") for s in rows},
            "min",
        ),
    ]
    caption = (
        r"Accuracy of the rollouts on the fluid cells: RMSE (RMS over the leads) "
        r"and anomaly correlation (ACC, about the truth's time mean; mean over the "
        r"leads) for the leads $t \le h\,\bar{T}$ and the whole window "
        f"($t \\le {window:.1f}\\,\\bar{{T}}$); $t_\\mathrm{{ACC}}$ is the first "
        f"lead with ACC $< {threshold:g}$, mean over the samples, a sample whose "
        r"ACC never drops counting as the window's end ($\geq$: a lower bound, "
        r"some sample never drops); the vorticity and pedestrian-level RMSE are "
        r"over the window. "
        f"{NORMALISATION} {BOLD}"
    )
    write_table(run_dir / "table_accuracy", columns, rows, models, names, caption)


def _physics(
    scalars: dict, models: list[str], names: dict[str, str], run_dir: pathlib.Path
) -> None:
    rows = [*models, PERSISTENCE, TRUTH]

    def column(group: str, head: str, key: str, best: str | float) -> Column:
        values = {s: _mean(scalars, s, key) for s in rows}
        return Column(group, head, key, values, best)

    truth_length = _mean(scalars, TRUTH, "integral_length")
    columns = [
        column("RMSE", r"$\bar{u}/u_\mathrm{ref}$", "mean_u_rmse", "min"),
        column("RMSE", r"$k/u_\mathrm{ref}^2$", "tke_rmse", "min"),
        column(
            "RMSE",
            r"$-\langle u'w' \rangle/u_\mathrm{ref}^2$",
            "profile_rmse_uw",
            "min",
        ),
        # dB: two decimals in both bands (0.01 dB is far below any difference)
        column(r"$E_{u'}$ [dB]", "mid", "spectra_db_u_mid", 0.0)._replace(digits=2),
        column(
            r"$E_{u'}$ [dB]", "near cutoff", "spectra_db_u_near_cutoff", 0.0
        )._replace(digits=2),
        column(
            "",
            r"$\Lambda_y/H$",
            "integral_length",
            truth_length if truth_length is not None else np.nan,
        ),
        column("", r"$W_2(u')/u_\mathrm{ref}$", "wasserstein_u", "min"),
        column("", r"$D/D_\mathrm{truth}$", "divergence_ratio", 1.0),
    ]
    # The truth is the reference only of Lambda/H; its other entries are trivial.
    for c in columns:
        if c.name != "integral_length":
            c.values[TRUTH] = None
    rows = [s for s in rows if any(c.values.get(s) is not None for c in columns)]
    caption = (
        r"Turbulence statistics over the window. RMSE of the time-mean "
        r"streamwise velocity $\bar{u}$ and of the resolved turbulent kinetic "
        r"energy $k$ (fields) and of the $-\langle u'w' \rangle$ profile; energy "
        r"of the spanwise spectrum of $u'$ at the canopy top against the truth in "
        r"the mid (4--8 cells) and near-cutoff (2--4 cells) wavelength bands; "
        r"spanwise integral length $\Lambda_y$ and Wasserstein-2 distance $W_2$ of "
        r"the $u'$ distribution at the canopy top; RMS divergence $D$ relative to "
        r"the truth's. "
        f"{NORMALISATION} Bold: the best model per column (smallest error, closest "
        r"to 0~dB, to 1 or to the truth); ``--'': unstable rollout."
    )
    write_table(run_dir / "table_physics", columns, rows, models, names, caption)


def _cost(
    scalars: dict, models: list[str], names: dict[str, str], run_dir: pathlib.Path
) -> None:
    def scaled(key: str, factor: float) -> dict[str, float | None]:
        values = {m: _mean(scalars, m, key) for m in models}
        return {m: v * factor if v is not None else None for m, v in values.items()}

    columns = [
        Column(
            "",
            r"Parameters ($10^6$)",
            "n_parameters",
            scaled("n_parameters", 1e-6),
            None,
        ),
        Column("", "ms / step", "ms_per_step", scaled("seconds_per_step", 1e3), "min"),
    ]
    speedup = scaled("speedup", 1.0)
    if any(v is not None for v in speedup.values()):
        columns.append(Column("", "Speedup", "speedup", speedup, "max"))
    caption = (
        r"Cost: parameters and inference wall time per lead step"
        r" (and speedup over the LES solver per output interval, if measured)."
        r" Bold: the fastest model."
    )
    write_table(run_dir / "table_cost", columns, models, models, names, caption)


def _mean(scalars: dict, source: str, key: str) -> Any:
    """The sample mean of a metrics.yaml scalar (plain for the cost entries);
    None if it is not there (or null)."""
    value = scalars.get(source, {}).get(key)
    if isinstance(value, dict):
        value = value.get("mean")
    if value is None or isinstance(value, bool):
        return value
    return float(value)


TEX_SPECIALS = {
    "\\": r"\textbackslash{}",
    "&": r"\&",
    "%": r"\%",
    "$": r"\$",
    "#": r"\#",
    "_": r"\_",
    "{": r"\{",
    "}": r"\}",
    "~": r"\textasciitilde{}",
    "^": r"\textasciicircum{}",
}


def tex_escape(text: str) -> str:
    """`text` as literal LaTeX text."""
    return text.translate({ord(c): e for c, e in TEX_SPECIALS.items()})


def decimals(values: list[float]) -> int:
    """Decimals that give the median |finite value| three significant digits (0..5)."""
    finite = np.abs([v for v in values if math.isfinite(v) and v != 0])
    if not finite.size:
        return 2
    return int(np.clip(2 - math.floor(math.log10(float(np.median(finite)))), 0, 5))


def best_rows(column: Column, models: list[str], digits: int) -> set[str]:
    """The models with the best value of `column` at `digits` decimals."""
    if column.best is None:
        return set()
    scores = {}
    for m in models:
        value = column.values.get(m)
        if value is None or math.isnan(value):
            continue
        value = round(value, digits)
        if column.best == "min":
            scores[m] = value
        elif column.best == "max":
            scores[m] = -value
        elif math.isfinite(float(column.best)):
            scores[m] = round(abs(value - float(column.best)), digits)
    if not scores:
        return set()
    top = min(scores.values())
    return {m for m, s in scores.items() if s == top}


def write_table(
    path_stem: pathlib.Path,
    columns: list[Column],
    rows: list[str],
    models: list[str],
    names: dict[str, str],
    caption: str,
) -> None:
    """`path_stem`.tex (booktabs table, models then a rule then the other rows)
    and `path_stem`.csv (the same, full precision)."""
    digits = {
        c.name: (
            c.digits
            if c.digits is not None
            else decimals([v for v in c.values.values() if v is not None])
        )
        for c in columns
    }

    def cell(column: Column, source: str, bold: bool) -> str:
        value = column.values.get(source)
        if value is None:
            return ""
        if math.isnan(value):
            return "--"
        text = (column.prefixes or {}).get(
            source, ""
        ) + f"{value:.{digits[column.name]}f}"
        text = text.replace("-", "$-$")
        return rf"\textbf{{\boldmath {text}}}" if bold else text

    bold = {c.name: best_rows(c, models, digits[c.name]) for c in columns}
    lines = [
        r"\begin{table}",
        r"\centering",
        rf"\caption{{{caption}}}",
        rf"\label{{tab:comparison_{path_stem.name.removeprefix('table_')}}}",
        r"\begin{tabular}{l" + "r" * len(columns) + "}",
        r"\toprule",
    ]
    if any(c.group for c in columns):
        spans: list[tuple[str, int]] = []  # consecutive columns of one group
        for c in columns:
            if spans and c.group and spans[-1][0] == c.group:
                spans[-1] = (c.group, spans[-1][1] + 1)
            else:
                spans.append((c.group, 1))
        heads, rules, start = [""], [], 2
        for group, n in spans:
            heads.append(rf"\multicolumn{{{n}}}{{c}}{{{group}}}" if group else "")
            if group:
                rules.append(rf"\cmidrule(lr){{{start}-{start + n - 1}}}")
            start += n
        lines += [" & ".join(heads) + r" \\", " ".join(rules)]
    lines.append(" & ".join(["Model", *(c.head for c in columns)]) + r" \\")
    lines.append(r"\midrule")
    for i, source in enumerate(rows):  # the models first
        if i == len(models) > 0:
            lines.append(r"\midrule")
        cells = [cell(c, source, source in bold[c.name]) for c in columns]
        lines.append(
            " & ".join([tex_escape(names.get(source, source)), *cells]) + r" \\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    path_stem.with_suffix(".tex").write_text("\n".join(lines))

    with open(path_stem.with_suffix(".csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["source", *(c.name for c in columns)])
        for source in rows:
            values = [c.values.get(source) for c in columns]
            writer.writerow([source, *("" if v is None else repr(v) for v in values)])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=pathlib.Path)
    parser.add_argument(
        "overrides", nargs="*", help="config overrides, e.g. metrics.horizons=[1,2]"
    )
    args = parser.parse_args()
    run(args.run_dir, args.overrides)


if __name__ == "__main__":
    main()
