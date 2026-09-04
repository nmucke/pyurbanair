"""Build every figure used by ``sections/filtering.tex``.

All crops go through the shared library ``figlib`` so the filtering figures
match the ESMDA, filter-smoothing and comparison ones.  Run IDs come from
:func:`figlib.run_ids` (STYLE_SPEC section 1).

Run with::

    cd /Users/ntmucke/Code/pyurbanair
    pixi run -e dev python experiments_report/scripts/make_filtering_figures.py
"""

from __future__ import annotations

import sys
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))

import figlib as fl  # noqa: E402

FIG = fl.REPO / "experiments_report" / "figures" / "filtering"

# Reference run of each case: uDALES truth, loccorrelation, joint + RTPS.
REF = {"inflow": "F7", "inflow_turb": "F9", "periodic": "F11"}

# Method-knob ladder (uDALES truth, locnone): none -> RTPS -> joint.
LADDER = {
    "inflow": [("F14", "state, none"), ("F15", "state, RTPS"), ("F13", "joint, RTPS")],
}

# All joint runs, for the appendix parameter-error grid.
JOINT = [
    ("F13", "inflow", "no loc"),
    ("F7", "inflow", "loc"),
    ("F16", "inflow_turb", "no loc"),
    ("F9", "inflow_turb", "loc"),
    ("F19", "periodic", "no loc"),
    ("F11", "periodic", "loc"),
    ("F5", "PALM inflow_turb", "no loc"),
    ("F2", "PALM inflow_turb", "loc"),
    ("F6", "PALM periodic", "no loc"),
    ("F3", "PALM periodic", "loc"),
]


def d(rid: str) -> Path:
    return fl.run_dir("filtering", rid)


def alpha_panel(rid: str) -> Image.Image:
    """Top (inflow-angle) panel of ``parameter_error.png`` only."""
    path = d(rid) / "parameter_error.png"
    y0, y1 = fl.panel_bands(path)[0]
    im = Image.open(path).convert("RGB")
    return im.crop((0, y0, im.width, y1))


def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    written: set[str] = set()

    def out(img: Image.Image, name: str) -> None:
        fl.save(img, FIG / name)
        written.add(name)
        print(f"  {name}")

    # (a)/(b)/(c) per case, for the reference run of that case ---------------- #
    for case, rid in REF.items():
        out(fl.crop_validation(d(rid), sensors=(0, 3)), f"valid_{rid}.png")
        out(fl.crop_param_error(d(rid)), f"parerr_{rid}.png")
        out(fl.crop_marginals(d(rid)), f"marginals_{rid}.png")

    # model error: PALM truth vs uDALES truth, inflow_turb -------------------- #
    out(
        fl.hstack(
            [
                fl.crop_validation(d("F9"), sensors=(0, 3)),
                fl.crop_validation(d("F2"), sensors=(0, 3)),
            ],
            [
                "F9 · inflow_turb · uDALES truth, loc, joint+RTPS",
                "F2 · inflow_turb · PALM truth, loc, joint+RTPS",
            ],
        ),
        "valid_modelerror_inflow_turb.png",
    )

    # sensitivity: the method-knob ladder none -> RTPS -> joint --------------- #
    for case, rungs in LADDER.items():
        out(
            fl.hstack(
                [fl.crop_validation(d(r), sensors=(3,)) for r, _ in rungs],
                [f"{r} · {case} · no loc, {k}" for r, k in rungs],
            ),
            f"ladder_{case}.png",
        )

    # calibration: rank histograms, the three reference runs in one row ------- #
    out(
        fl.hstack(
            [fl.crop_rank_hist(d(REF[c])) for c in ("inflow", "inflow_turb", "periodic")],
            [
                f"{REF[c]} · {c} · loc, joint+RTPS"
                for c in ("inflow", "inflow_turb", "periodic")
            ],
        ),
        "rank_cases.png",
    )

    # appendix: inflow-angle error for every joint run ------------------------ #
    out(
        fl.grid(
            [alpha_panel(rid) for rid, _, _ in JOINT],
            2,
            [f"{rid} · {case} · {loc}, joint+RTPS" for rid, case, loc in JOINT],
        ),
        "parerr_grid_joint.png",
    )

    stale = sorted(p.name for p in FIG.glob("*.png") if p.name not in written)
    for name in stale:
        (FIG / name).unlink()
        print(f"  removed stale {name}")
    print(f"{len(written)} figures in {FIG}")


if __name__ == "__main__":
    main()
