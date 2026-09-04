"""Build the figure set for the filter-smoothing (hybrid) report section.

Run from the repo root::

    pixi run -e dev python experiments_report/scripts/make_filter_smoothing_figures.py

Every crop goes through the shared library ``figlib`` so the filter-smoothing
figures are cropped, labelled and scaled exactly like the ESMDA, filtering and
comparison ones.  Output lands in ``experiments_report/figures/filter_smoothing/``.

Reference run of each case (STYLE_SPEC section 2): uDALES truth, localization on,
``esmda.interval_seconds = 15`` -- H11 (inflow), H12 (inflow_turb), H13 (periodic).
"""

from __future__ import annotations

import sys
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
import figlib as fl  # noqa: E402

OUTDIR = fl.REPO / "experiments_report" / "figures" / "filter_smoothing"
METHOD = "filter_smoothing"

IDS = fl.run_ids(METHOD)
REFS = {"inflow": "H11", "inflow_turb": "H12", "periodic": "H13"}


# --------------------------------------------------------------------------- #
# small local helpers (figlib stays generic)
# --------------------------------------------------------------------------- #
def decode(run_id: str) -> tuple[str, str, str, str]:
    """(truth, case, loc, knob) of a run ID, from its directory name."""
    name = IDS[run_id]
    if name.endswith("inflow_turb"):
        case = "inflow_turb"
    elif name.endswith("periodic"):
        case = "periodic"
    else:
        case = "inflow"
    return (
        "PALM" if name.startswith("pypalm") else "uDALES",
        case,
        "loc on" if "loccorrelation" in name else "loc off",
        "7.5 s" if "obs7p5" in name else ("30 s" if "obs30" in name else "15 s"),
    )


def strip(run_id: str, with_truth: bool = False) -> str:
    """Label-strip text: "H13 . periodic . loc on, 15 s"."""
    truth, case, loc, knob = decode(run_id)
    head = f"{run_id} · {truth} truth · {case}" if with_truth else f"{run_id} · {case}"
    return f"{head} · {loc}, {knob}"


def d(run_id: str) -> Path:
    return fl.run_dir(METHOD, run_id)


def angle_panel(run_id: str, title: str | None = None) -> Image.Image:
    """Only the inflow-angle panel of ``parameter_error.png`` (band 0)."""
    path = d(run_id) / "parameter_error.png"
    y0, y1 = fl.panel_bands(path)[0]
    im = Image.open(path).convert("RGB").crop((0, y0, Image.open(path).width, y1))
    return fl.label(im, title) if title else im


def out(img: Image.Image, name: str, max_width: int = 1600) -> None:
    p = fl.save(img, OUTDIR / name, max_width=max_width)
    print(f"  {name:<34} {img.width}x{img.height}  {p.stat().st_size // 1024} kB")


# --------------------------------------------------------------------------- #
def main() -> None:
    print("filter-smoothing figures ->", OUTDIR)
    OUTDIR.mkdir(parents=True, exist_ok=True)

    # -- (a)/(b)/(c) per case, for the reference run of each case ------------ #
    for case, rid in REFS.items():
        out(fl.crop_validation(d(rid), sensors=(0, 3), title=strip(rid)), f"valid_{rid}.png")
        out(fl.crop_param_error(d(rid), title=strip(rid)), f"parerr_{rid}.png")
        out(fl.crop_marginals(d(rid), title=strip(rid)), f"marginals_{rid}.png")

    # -- model error: PALM truth vs uDALES truth on inflow_turb -------------- #
    out(
        fl.vstack(
            [
                fl.crop_validation(d(r), sensors=(0,), title=strip(r, with_truth=True))
                for r in ("H12", "H2")
            ]
        ),
        "modelerror_valid_inflow_turb.png",
    )

    # -- sensitivity: localization on/off on periodic ------------------------ #
    out(
        fl.vstack([fl.crop_param_error(d(r), title=strip(r)) for r in ("H13", "H20")]),
        "localization_parerr_periodic.png",
    )

    # -- sensitivity: ESMDA aggregation interval on inflow_turb -------------- #
    out(
        fl.vstack([angle_panel(r, strip(r)) for r in ("H16", "H12", "H14")]),
        "interval_parerr_inflow_turb.png",
    )

    # -- calibration: rank histograms of the three reference runs ------------ #
    out(
        fl.hstack(
            [fl.crop_rank_hist(d(REFS[c])) for c in REFS],
            labels=[strip(REFS[c]) for c in REFS],
        ),
        "rank_hist_cases.png",
    )

    # -- appendix: inflow-angle error of every completed run ----------------- #
    case_rank = {"inflow": 0, "inflow_turb": 1, "periodic": 2}
    knob_rank = {"7.5 s": 0, "15 s": 1, "30 s": 2}

    def order(run_id: str) -> tuple[int, int, int, int]:
        truth, case, loc, knob = decode(run_id)
        return (truth != "uDALES", case_rank[case], loc != "loc on", knob_rank[knob])

    done = sorted(
        (i for i in IDS if (d(i) / "parameter_error.png").exists()), key=order
    )
    out(
        fl.grid([angle_panel(i) for i in done], ncols=2,
                labels=[strip(i, with_truth=True) for i in done]),
        "parerr_grid_all.png",
    )

    print("done")


if __name__ == "__main__":
    main()
