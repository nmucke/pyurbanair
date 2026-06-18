"""Orchestrate the full EnKF-2026 figure pipeline (spec: docs/figure_specs.md).

Runs the block drivers in sequence into a single ``figures/`` tree, then the
summary (which consumes the per-block metric caches) and the NOTES generator.
Animations are separate (they need ffmpeg) -- pass ``--animations`` to include
them, or run ``scripts/make_animations.py`` under the dedicated SLURM wrapper.

    python scripts/make_all_figures.py --out figures --heavy
"""
from __future__ import annotations

import argparse
import pathlib
import subprocess
import sys

HERE = pathlib.Path(__file__).resolve().parent
PY = sys.executable


def run(script: str, extra: list[str]) -> None:
    cmd = [PY, "-u", str(HERE / script), *extra]
    print(f"\n=== {script} {' '.join(extra)} ===", flush=True)
    subprocess.run(cmd, check=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="figures")
    ap.add_argument("--heavy", action="store_true",
                    help="produce field-based figures (needs the big mean files; use SLURM)")
    ap.add_argument("--no-mask", action="store_true")
    ap.add_argument("--animations", action="store_true",
                    help="also render the mp4 animations (needs ffmpeg on PATH/FFMPEG_BIN)")
    args = ap.parse_args()

    common = ["--out", args.out]
    heavy = (["--heavy"] if args.heavy else []) + (["--no-mask"] if args.no_mask else [])

    run("make_figures_block_a.py", common + heavy)
    run("make_figures_block_b.py", common + heavy)
    run("make_figures_block_c.py", common + heavy)
    run("make_figures_summary.py", common)
    run("make_notes.py", common)
    if args.animations:
        run("make_animations.py", common)

    print(f"\nAll figures written under {args.out}/")


if __name__ == "__main__":
    main()
