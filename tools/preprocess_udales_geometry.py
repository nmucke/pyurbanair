#!/usr/bin/env python
"""Precompute uDALES IBM geometry for a case and save it into examples/udales/<case>/.

The expensive part of uDALES preprocessing is the STL->IBM step (solid/fluid
point classification + facet-to-cell matching), run by the Fortran
``IBM_preproc.exe``. For a fixed STL *and* grid it always produces the same
``solid_*``/``fluid_boundary_*``/``facet_sections_*`` files, so it is wasteful to
re-run it on every model instantiation.

This standalone utility runs that step once for a case's configured grid and
saves the resulting geometry files (plus a ``geom_meta.json`` recording the grid
and ``nfcts``) into ``examples/udales/<case>/``. A later run reuses them by
pointing the model's ``precomputed_geom_dir`` at that folder
(``conf/case/<case>/geometry.yaml: udales_precomputed_geom_dir``), which flips
``gen_geom`` to ``.false.`` so preprocessing copies the files instead of
re-running the classifier.

The geometry is grid-specific. The bundle's ``geom_meta.json`` records the grid
it was built for; the model validates it against the active domain and raises on
a mismatch, so regenerate (re-run this script) whenever you change
``domain.nx/ny/nz``/``bounds`` for the case.

Examples
--------
    pixi run python tools/preprocess_udales_geometry.py --case barcelona
    # override the grid for the generated bundle:
    pixi run python tools/preprocess_udales_geometry.py --case barcelona \
        --overrides domain.nx=300 domain.ny=300 domain.nz=64
"""

from __future__ import annotations

import argparse
import pathlib
import sys

from hydra import compose, initialize_config_dir
from hydra.utils import instantiate

from pyudales.forward_model import save_precomputed_geometry

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
CONF_DIR = PROJECT_ROOT / "conf"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--case", default="barcelona", help="Case name (conf/case/<case>).")
    p.add_argument(
        "--output-dir",
        type=pathlib.Path,
        default=None,
        help="Where to save the geometry bundle (default: examples/udales/<case>).",
    )
    p.add_argument(
        "--overrides",
        nargs="*",
        default=[],
        help="Extra Hydra overrides (e.g. domain.nx=600 domain.nz=32).",
    )
    args = p.parse_args(argv)

    out_dir = args.output_dir or (PROJECT_ROOT / "examples" / "udales" / args.case)

    with initialize_config_dir(version_base=None, config_dir=str(CONF_DIR)):
        cfg = compose(
            config_name="config",
            overrides=[
                "model=pyudales",
                f"case={args.case}",
                # Force generation from the STL even if the case already points at
                # a precomputed bundle (otherwise we'd try to reuse what we are
                # about to (re)build).
                "model.forward_model.precomputed_geom_dir=null",
                *args.overrides,
            ],
        )

        results_dir = str(PROJECT_ROOT / ".temp" / "udales_geom_prep")
        fm = instantiate(cfg.model.forward_model, results_dir=results_dir)

        nm = fm.dirs.experiment_dir / f"namoptions.{fm.dirs.experiment_name}"
        print(
            f"Preprocessing uDALES geometry for case={args.case}\n"
            f"  experiment dir : {fm.dirs.experiment_dir}\n"
            f"  STL            : {cfg.geometry.stl_path}\n"
            f"  output bundle  : {out_dir}\n"
            "Running the IBM classifier from the STL — this can take many minutes "
            "for a large grid / facet count ...",
            flush=True,
        )

        # gen_geom defaults to .true. (precomputed_geom_dir is null), so this runs
        # the full STL->IBM Fortran step and writes the geometry files + &WALLS
        # counts into the experiment dir.
        fm.run_preprocessing()

        dest = save_precomputed_geometry(fm.dirs.experiment_dir, out_dir, namoptions_path=nm)

    files = sorted(q.name for q in pathlib.Path(dest).glob("*") if q.is_file())
    print(f"\nSaved {len(files)} geometry files to {dest}:")
    for name in files:
        print(f"  {name}")
    print(
        f"\nTo reuse this bundle, set in conf/case/{args.case}/geometry.yaml:\n"
        f"  udales_precomputed_geom_dir: examples/udales/{args.case}\n"
        "Subsequent uDALES runs will skip the IBM classifier and copy these files."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
