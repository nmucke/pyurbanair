"""Build the ESMDA report figures from the raw per-run PNGs.

Run with::

    cd /Users/ntmucke/Code/pyurbanair && \
        pixi run -e dev python experiments_report/scripts/make_esmda_figures.py

Writes into ``experiments_report/figures/esmda/``.  All cropping/composition
goes through the shared library ``figlib`` so the ESMDA figures match the
filtering, filter-smoothing and comparison ones (STYLE_SPEC section 4).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import figlib as fl  # noqa: E402

OUT = fl.REPO / "experiments_report/figures/esmda"

# Reference run of each case: uDALES truth, localization on, obs 15 s.
REF = {"inflow": "E11", "inflow_turb": "E12", "periodic": "E13"}
CASES = ("inflow", "inflow_turb", "periodic")


def parse(name: str) -> tuple[str, str, str]:
    """``(case, loc, obs)`` from a run-directory name."""
    loc, rest = name.split("_w3_", 1)[1].split("_obs", 1)
    obs, case = rest.split("_", 1)
    return case, ("loc" if loc == "loccorrelation" else "no loc"), obs.replace("p", ".")


def strip(rid: str) -> str:
    """Label strip text "ID . case . knob" (STYLE_SPEC section 4)."""
    case, loc, obs = parse(fl.run_ids("esmda")[rid])
    return f"{rid} · {case} · {loc}, obs{obs}"


def d(rid: str) -> Path:
    return fl.run_dir("esmda", rid)


def alpha_panel(run_dir: Path):
    """Only the inflow-angle panel of ``parameter_error.png``.

    Small local helper (figlib's ``crop_param_error`` always returns both
    panels); stacking three single panels keeps the observation-interval sweep
    legible at \\textwidth.
    """
    path = run_dir / "parameter_error.png"
    im = fl._open(path)  # noqa: SLF001 -- figlib's own loader, same contract
    y0, y1 = fl.panel_bands(path)[0]
    return im.crop((0, y0, im.width, y1))


# --------------------------------------------------------------------------- #
# (a)/(b)/(c) per case -- the three reference runs
# --------------------------------------------------------------------------- #
def per_case() -> None:
    for case in CASES:
        rid = REF[case]
        fl.save(
            fl.crop_validation(d(rid), sensors=(0, 3), title=strip(rid)),
            OUT / f"valid_{rid}.png",
        )
        fl.save(fl.crop_param_error(d(rid), title=strip(rid)), OUT / f"perr_{rid}.png")
        fl.save(fl.crop_marginals(d(rid), title=strip(rid)), OUT / f"marg_{rid}.png")


# --------------------------------------------------------------------------- #
# model error, sensitivities, calibration
# --------------------------------------------------------------------------- #
def model_error() -> None:
    """Validation sensors, PALM truth vs uDALES truth, turbulent inflow."""
    imgs = [fl.crop_validation(d(r), sensors=(0, 3)) for r in ("E12", "E2")]
    fl.save(
        fl.vstack(
            imgs, [strip("E12") + " · uDALES truth", strip("E2") + " · PALM truth"]
        ),
        OUT / "valid_modelerror_E12_E2.png",
    )


def localization() -> None:
    """Final-knot marginals with and without localization, periodic case."""
    imgs = [fl.crop_marginals(d(r)) for r in ("E13", "E20")]
    fl.save(fl.vstack(imgs, [strip("E13"), strip("E20")]), OUT / "marg_loc_E13_E20.png")


def obs_interval() -> None:
    """Inflow-angle error across the observation-interval sweep, turbulent inflow."""
    ids = ("E16", "E12", "E14")
    imgs = [alpha_panel(d(r)) for r in ids]
    fl.save(
        fl.vstack(imgs, [strip(r) for r in ids]),
        OUT / "perr_obsinterval_E16_E12_E14.png",
    )


def rank_hist() -> None:
    """Rank histograms of the three reference runs, one row."""
    ids = [REF[c] for c in CASES]
    imgs = [fl.crop_rank_hist(d(r)) for r in ids]
    fl.save(fl.hstack(imgs, [strip(r) for r in ids]), OUT / "rank_E11_E12_E13.png")


# --------------------------------------------------------------------------- #
# appendix: every run
# --------------------------------------------------------------------------- #
def param_error_grid(truth: str, out_name: str) -> None:
    ids = [
        rid
        for rid, name in fl.run_ids("esmda").items()
        if name.startswith(truth) and (d(rid) / "parameter_error.png").exists()
    ]
    ids.sort(key=lambda r: (CASES.index(parse(fl.run_ids("esmda")[r])[0]), r))
    imgs = [fl.crop_param_error(d(r)) for r in ids]
    fl.save(fl.grid(imgs, 2, [strip(r) for r in ids]), OUT / out_name, max_width=1700)


if __name__ == "__main__":
    per_case()
    model_error()
    localization()
    obs_interval()
    rank_hist()
    param_error_grid("pyudales", "perr_grid_udales.png")
    param_error_grid("pypalm", "perr_grid_palm.png")
    print("\n".join(sorted(p.name for p in OUT.glob("*.png"))))
