"""Benchmark pypalm ensemble parallel scaling.

Companion to ``scripts/benchmark_ensemble_scaling.py`` (uDALES variant).
Measures total wall time and per-member runtime as
``num_parallel_processes`` is varied at ``ncpu=1``, to find where the
DRAM-bandwidth cliff lands for PALM. The uDALES benchmark showed the
cliff at workers=4 on the local 3950X box; PALM has more physics per
cell, so the cliff is expected to sit at a higher worker count, but
this script is the only honest way to find out.

Per-member timing is captured by class-level monkey-patching
``pypalm.ForwardModel.run_single`` to write a one-line timings file
into ``BENCH_TIMING_DIR``. Under the ProcessPoolExecutor + fork
context used by ``BaseEnsembleForwardModel._run_parallel``, workers
inherit the patched class.

Requires palmrun on PATH (or PALM_ROOT/PALM_BIN set). Compile is NOT
performed here; assume PALM is already built.

Run:
    pixi run -e dev python scripts/benchmark_palm_ensemble_scaling.py

Limit the sweep to one or two configs while iterating:
    pixi run -e dev python scripts/benchmark_palm_ensemble_scaling.py --sweeps 1:1,1:4
"""

from __future__ import annotations

import argparse
import csv
import os
import pathlib
import shutil
import statistics
import sys
import time
from dataclasses import dataclass

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import pypalm  # noqa: F401, E402  (init-side effects: locates palmrun etc.)
from pypalm.ensemble_forward_model import EnsembleForwardModel  # noqa: E402
from pypalm.forward_model import ForwardModel as _PALMForwardModel  # noqa: E402

# Sweep at ncpu=1 only (matches the recommended production setting for
# this codebase). The uDALES benchmark already showed that mixing
# ncpu>1 with parallel workers is strictly worse on this hardware.
DEFAULT_SWEEPS: list[tuple[int, int]] = [
    (1, 1),
    (1, 2),
    (1, 4),
    (1, 6),
    (1, 8),
    (1, 12),
]

CASE_DIR = pathlib.Path("examples/palm/experiments/xie_and_castro_palm")
STL_PATH = CASE_DIR / "xie_castro_2008_STL.stl"
EXPERIMENT_NAME = "bench_run"

# Tuned to keep one member around 10-30s on a single CPU. Bump if
# per-member time is too short for the contention measurement to be
# stable, or shrink the domain.
ENSEMBLE_SIZE = 8
SIMULATION_TIME = 60.0
OUTPUT_FREQUENCY = 30.0
SPINUP_TIME = 0.0

DOMAIN = {
    "nx": 32,
    "ny": 32,
    "nz": 16,
    "bounds": ((0.0, 50.0), (0.0, 50.0), (0.0, 32.0)),
}


# Monkey-patch ForwardModel.run_single to capture per-member runtime.
# Apply BEFORE the ProcessPoolExecutor forks so workers inherit the
# patched class via copy-on-write.
_orig_run_single = _PALMForwardModel.run_single


def _timed_run_single(self, *args, **kwargs):  # type: ignore[no-untyped-def]
    t0 = time.time()
    try:
        return _orig_run_single(self, *args, **kwargs)
    finally:
        elapsed = time.time() - t0
        bench_dir = pathlib.Path(
            os.environ.get("BENCH_TIMING_DIR", "/tmp/palm_bench_timings")
        )
        bench_dir.mkdir(parents=True, exist_ok=True)
        exp_id = getattr(self, "experiment_name", None) or "unknown"
        try:
            with open(bench_dir / f"{exp_id}.timings", "w") as fh:
                fh.write(f"total {elapsed:.6f}\n")
        except OSError:
            pass


_PALMForwardModel.run_single = _timed_run_single  # type: ignore[assignment]


@dataclass
class StageTimings:
    total: float


def parse_member_timings(bench_dir: pathlib.Path) -> dict[str, StageTimings]:
    out: dict[str, StageTimings] = {}
    for f in sorted(bench_dir.glob("*.timings")):
        kv: dict[str, float] = {}
        for line in f.read_text().splitlines():
            parts = line.split()
            if len(parts) == 2:
                kv[parts[0]] = float(parts[1])
        out[f.stem] = StageTimings(total=kv.get("total", float("nan")))
    return out


def assert_palmrun_available() -> None:
    # pypalm._resolve_palmrun() also checks the auto-installed bundle at
    # libs/pypalm/palm_model_system/bin/palmrun, which shutil.which won't
    # find. Trust pypalm.PALMRUN_BIN.
    if pypalm.PALMRUN_BIN is not None and pathlib.Path(pypalm.PALMRUN_BIN).exists():
        return
    if shutil.which("palmrun") is not None:
        return
    if os.environ.get("PALM_BIN") or os.environ.get("PALM_ROOT"):
        return
    raise RuntimeError(
        "palmrun not found. Set PALM_BIN/PALM_ROOT, put palmrun on PATH, or "
        "let pypalm auto-install palm_model_system."
    )


def build_ensemble(
    project_root: pathlib.Path,
    ncpu: int,
    workers: int,
) -> EnsembleForwardModel:
    fm = _PALMForwardModel(
        case_dir=project_root / CASE_DIR,
        stl_path=project_root / STL_PATH,
        experiment_name=EXPERIMENT_NAME,
        ncpu=ncpu,
        nx=DOMAIN["nx"],
        ny=DOMAIN["ny"],
        nz=DOMAIN["nz"],
        bounds=DOMAIN["bounds"],
        simulation_time=SIMULATION_TIME,
        output_frequency=OUTPUT_FREQUENCY,
        spinup_time=SPINUP_TIME,
        boundary_condition="periodic",
        verbose=False,
    )
    # Assume PALM is already built; calling fm.compile(compile=True)
    # would invoke palmbuild and is the user's job before benchmarking.

    ens = EnsembleForwardModel(
        forward_model=fm,
        ensemble_size=ENSEMBLE_SIZE,
        num_parallel_processes=workers,
        num_cpus_per_process=ncpu,
    )
    return ens


def run_one_sweep(
    project_root: pathlib.Path,
    ncpu: int,
    workers: int,
    bench_dir: pathlib.Path,
) -> dict:
    bench_dir.mkdir(parents=True, exist_ok=True)
    for f in bench_dir.glob("*.timings"):
        f.unlink()

    ens = build_ensemble(project_root, ncpu=ncpu, workers=workers)

    t0 = time.time()
    ens.run_ensemble(sim_name="state")
    wall = time.time() - t0

    timings = parse_member_timings(bench_dir)
    if timings:
        totals = [t.total for t in timings.values()]
        return {
            "ncpu": ncpu,
            "workers": workers,
            "wall_total": wall,
            "n_members": len(timings),
            "member_total_min": min(totals),
            "member_total_mean": statistics.fmean(totals),
            "member_total_max": max(totals),
        }
    return {
        "ncpu": ncpu,
        "workers": workers,
        "wall_total": wall,
        "n_members": ENSEMBLE_SIZE,
        "member_total_min": float("nan"),
        "member_total_mean": float("nan"),
        "member_total_max": float("nan"),
    }


def main() -> None:
    global ENSEMBLE_SIZE, SIMULATION_TIME

    parser = argparse.ArgumentParser()
    parser.add_argument("--ensemble-size", type=int, default=ENSEMBLE_SIZE)
    parser.add_argument("--simulation-time", type=float, default=SIMULATION_TIME)
    parser.add_argument(
        "--sweeps",
        type=str,
        default=None,
        help="Comma-separated ncpu:workers pairs, e.g. '1:1,1:4,1:8'",
    )
    parser.add_argument(
        "--out",
        type=pathlib.Path,
        default=pathlib.Path(".temp/bench/palm_ensemble_scaling.csv"),
    )
    args = parser.parse_args()

    ENSEMBLE_SIZE = args.ensemble_size
    SIMULATION_TIME = args.simulation_time

    if args.sweeps is not None:
        sweeps = []
        for pair in args.sweeps.split(","):
            a, b = pair.split(":")
            sweeps.append((int(a), int(b)))
    else:
        sweeps = DEFAULT_SWEEPS

    assert_palmrun_available()

    project_root = pathlib.Path(__file__).resolve().parent.parent
    bench_dir = project_root / ".temp/bench/palm_timings"
    args.out.parent.mkdir(parents=True, exist_ok=True)

    os.environ["BENCH_TIMING_DIR"] = str(bench_dir)

    rows: list[dict] = []
    for ncpu, workers in sweeps:
        print(f"\n=== sweep ncpu={ncpu} workers={workers} ===", flush=True)
        row = run_one_sweep(project_root, ncpu, workers, bench_dir)
        print(
            f"    wall={row['wall_total']:.2f}s "
            f"member_total mean={row['member_total_mean']:.2f}s "
            f"max={row['member_total_max']:.2f}s",
            flush=True,
        )
        rows.append(row)

    fieldnames = list(rows[0].keys())
    with open(args.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"\nWrote CSV: {args.out}")

    baseline = next(
        (r["wall_total"] for r in rows if r["ncpu"] == 1 and r["workers"] == 1),
        None,
    )

    print("\nSummary (sorted by wall_total):")
    print(f"{'ncpu':>4} {'workers':>7} {'wall':>8} {'member_avg':>10}  speedup")
    for row in sorted(rows, key=lambda r: r["wall_total"]):
        speedup = (
            f"{baseline / row['wall_total']:.2f}x"
            if baseline and row["wall_total"] > 0
            else "n/a"
        )
        print(
            f"{row['ncpu']:>4} {row['workers']:>7} "
            f"{row['wall_total']:>8.2f} {row['member_total_mean']:>10.2f}  {speedup}"
        )


if __name__ == "__main__":
    main()
