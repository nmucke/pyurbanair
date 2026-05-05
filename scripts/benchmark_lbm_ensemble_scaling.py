"""Benchmark pylbm ensemble parallel scaling.

Companion to ``scripts/benchmark_ensemble_scaling.py`` (uDALES) and
``scripts/benchmark_palm_ensemble_scaling.py``.

LBM runs as a single-process (non-MPI) Fortran binary per ensemble
member, so all parallelism comes from Python's ProcessPoolExecutor.
That makes it the cleanest test of where the DRAM-bandwidth ceiling
sits for this codebase: there's no MPI launcher overhead to confound
the measurement.

Compiles LBM once at startup (using the experiment's nx/ny/nz so the
boltzmann binary matches the benchmark domain), then reuses the
compiled binary across sweeps. Per-member timing is captured via a
class-level monkey-patch on ``pylbm.ForwardModel.run_single``.

Run:
    pixi run -e dev python scripts/benchmark_lbm_ensemble_scaling.py

Quick check:
    pixi run -e dev python scripts/benchmark_lbm_ensemble_scaling.py --sweeps 1:1,1:4
"""

from __future__ import annotations

import argparse
import csv
import os
import pathlib
import statistics
import sys
import time
from dataclasses import dataclass

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import pylbm  # noqa: F401, E402
from pylbm.ensemble_forward_model import EnsembleForwardModel  # noqa: E402
from pylbm.forward_model import ForwardModel as _LBMForwardModel  # noqa: E402

DEFAULT_SWEEPS: list[tuple[int, int]] = [
    (1, 1),
    (1, 2),
    (1, 4),
    (1, 6),
    (1, 8),
    (1, 12),
    (1, 16),
]

STL_PATH = pathlib.Path("examples/lbm/experiments/xie_castro_2008_STL.stl")
EXPERIMENT_NAME = "bench_run"

# Tuned to keep one member around 10-30s on a single CPU. Bump
# simulation_time if the per-member time is too short to see
# contention, or shrink the domain.
ENSEMBLE_SIZE = 8
SIMULATION_TIME = 10.0
OUTPUT_FREQUENCY = 5.0
SPINUP_TIME = 0.0

DOMAIN = {
    "nx": 64,
    "ny": 64,
    "nz": 16,
    "bounds": ((0.0, 80.0), (0.0, 80.0), (0.0, 32.0)),
}


_orig_run_single = _LBMForwardModel.run_single


def _timed_run_single(self, *args, **kwargs):  # type: ignore[no-untyped-def]
    t0 = time.time()
    try:
        return _orig_run_single(self, *args, **kwargs)
    finally:
        elapsed = time.time() - t0
        bench_dir = pathlib.Path(
            os.environ.get("BENCH_TIMING_DIR", "/tmp/lbm_bench_timings")
        )
        bench_dir.mkdir(parents=True, exist_ok=True)
        # pylbm uses experiment_dir.name as the per-member id; fall back to pid.
        exp_id = getattr(self.dirs, "experiment_dir", None)
        exp_id = exp_id.name if exp_id is not None else f"pid{os.getpid()}"
        try:
            with open(bench_dir / f"{exp_id}.timings", "w") as fh:
                fh.write(f"total {elapsed:.6f}\n")
        except OSError:
            pass


_LBMForwardModel.run_single = _timed_run_single  # type: ignore[assignment]


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


def build_template_forward_model(project_root: pathlib.Path) -> _LBMForwardModel:
    """Build a single ForwardModel with the benchmark domain and compile once."""
    fm = _LBMForwardModel(
        stl_path=project_root / STL_PATH,
        experiment_name=EXPERIMENT_NAME,
        nx=DOMAIN["nx"],
        ny=DOMAIN["ny"],
        nz=DOMAIN["nz"],
        bounds=DOMAIN["bounds"],
        simulation_time=SIMULATION_TIME,
        output_frequency=OUTPUT_FREQUENCY,
        spinup_time=SPINUP_TIME,
        cuda=False,
        verbose=False,
        boundary_condition="periodic",
    )
    print("Compiling LBM (one-time)…", flush=True)
    fm.compile(compile=True)
    print("Compile complete.", flush=True)
    return fm


def build_ensemble(
    template: _LBMForwardModel,
    workers: int,
) -> EnsembleForwardModel:
    return EnsembleForwardModel(
        forward_model=template,
        ensemble_size=ENSEMBLE_SIZE,
        num_parallel_processes=workers,
        num_cpus_per_process=1,
    )


def run_one_sweep(
    template: _LBMForwardModel,
    ncpu: int,
    workers: int,
    bench_dir: pathlib.Path,
) -> dict:
    bench_dir.mkdir(parents=True, exist_ok=True)
    for f in bench_dir.glob("*.timings"):
        f.unlink()

    ens = build_ensemble(template, workers=workers)

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
        default=pathlib.Path(".temp/bench/lbm_ensemble_scaling.csv"),
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

    project_root = pathlib.Path(__file__).resolve().parent.parent
    bench_dir = project_root / ".temp/bench/lbm_timings"
    args.out.parent.mkdir(parents=True, exist_ok=True)

    os.environ["BENCH_TIMING_DIR"] = str(bench_dir)

    template = build_template_forward_model(project_root)

    rows: list[dict] = []
    for ncpu, workers in sweeps:
        print(f"\n=== sweep ncpu={ncpu} workers={workers} ===", flush=True)
        row = run_one_sweep(template, ncpu, workers, bench_dir)
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
