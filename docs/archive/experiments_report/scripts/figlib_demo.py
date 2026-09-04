"""Smoke-test / gallery for ``figlib``: one example of every crop per campaign.

Run from the repo root::

    cd /Users/ntmucke/Code/pyurbanair
    pixi run -e dev python experiments_report/scripts/figlib_demo.py

Writes into ``experiments_report/figures/_demo/`` (scratch: not referenced by
the LaTeX sources) and prints the run-ID tables of the three campaigns.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import figlib as fl  # noqa: E402

OUT = fl.REPO / "experiments_report" / "figures" / "_demo"

DEMO = {
    "esmda": "pyudales_to_pyudales_w3_loccorrelation_obs15_inflow_turb",
    "filtering": "pyudales_to_pyudales_w3_loccorrelation_inflow_turb",
    "filter_smoothing": "pyudales_to_pyudales_w3_loccorrelation_obs15_inflow_turb",
}
LABEL = {
    "esmda": "ESMDA",
    "filtering": "EnKF joint + RTPS",
    "filter_smoothing": "Filter smoothing",
}


def rid_of(method: str, run: str) -> str:
    return next(k for k, v in fl.run_ids(method).items() if v == run)


def main() -> None:
    for method in fl.METHODS:
        print(f"\n{method}")
        for k, v in fl.run_ids(method).items():
            print(f"  {k:<4} {v}")

    print(f"\nwriting demo crops -> {OUT}")
    valids = []
    for method, run in DEMO.items():
        d = fl.run_dir(method, run)
        rid = rid_of(method, run)
        tag = f"{method}_{rid}"
        jobs = {
            "valid": fl.crop_validation(d),
            "assim": fl.crop_assimilation(d),
            "parerr": fl.crop_param_error(d),
            "marginals": fl.crop_marginals(d),
            "rank": fl.crop_rank_hist(d),
            "slices": fl.crop_slices(d),
            "parevol": fl.crop_param_evolution(d),
        }
        for kind, im in jobs.items():
            if im is None:
                print(f"  {kind:<10} {tag:<28} (absent)")
                continue
            p = fl.save(im, OUT / f"{kind}_{tag}.png")
            print(f"  {kind:<10} {tag:<28} {p.name}  {fl.Image.open(p).size}")
        valids.append(fl.crop_validation(d))

    labels = [f"{LABEL[m]} · {rid_of(m, r)} · inflow_turb" for m, r in DEMO.items()]
    p = fl.save(fl.hstack(valids, labels=labels), OUT / "hstack_valid_3way.png")
    print(f"  hstack     3-way validation        {p.name}  {fl.Image.open(p).size}")


if __name__ == "__main__":
    main()
