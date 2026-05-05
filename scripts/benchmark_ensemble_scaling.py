"""Benchmark uDALES ensemble parallel scaling.

Sweeps (ncpu_per_member, num_parallel_processes) and reports total wall
time plus per-stage timings (copy / mpiexec / gather_outputs.sh) so we
can see whether the modest-speedup symptom is dominated by MPI
oversubscription, NCO post-processing, or something else.

Run:
    pixi run -e dev python scripts/benchmark_ensemble_scaling.py
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

import pyudales  # noqa: E402  (sets up UDALES_PATH, builds executable)
import pyudales.forward_model as udales_fm  # noqa: E402
from pyudales.ensemble_forward_model import EnsembleForwardModel  # noqa: E402
from pyudales.forward_model import ForwardModel  # noqa: E402
from pyudales.utils.namoptions_utils import NamoptionsFile  # noqa: E402

# (ncpu_per_member, num_parallel_processes)
DEFAULT_SWEEPS: list[tuple[int, int]] = [
    (1, 1),    # serial baseline
    (1, 2),
    (1, 4),
    (1, 8),
    (1, 16),
    (2, 8),
    (4, 4),
    (4, 8),    # production point
    (8, 2),
    (16, 1),
]

CASE_DIR = pathlib.Path("examples/udales/experiments/xie_and_castro")
EXPERIMENT_NAME = "888"  # avoid clashing with production "999"
ENSEMBLE_SIZE = 16
SIMULATION_TIME = 60.0
OUTPUT_FREQUENCY = 30.0
SPINUP_TIME = 0.0

# Domain divisible by every nprocx, nprocy combination we use:
#   nprocx ∈ {1,2,4} divides 64
#   nprocy ∈ {1,2,4} divides 64, and ktot=16 divisible by nprocy
DOMAIN = {
    "nx": 64,
    "ny": 64,
    "nz": 16,
    "bounds": ((-10.0, 40.0), (0.0, 40.0), (0.0, 32.0)),
}

NPROC_FACTORIZATIONS: dict[int, tuple[int, int]] = {
    1: (1, 1),
    2: (2, 1),
    4: (2, 2),
    8: (4, 2),
    16: (4, 4),
}


@dataclass
class StageTimings:
    copy: float
    mpiexec: float
    gather: float
    total: float


def parse_member_timings(bench_dir: pathlib.Path) -> dict[str, StageTimings]:
    out: dict[str, StageTimings] = {}
    for f in sorted(bench_dir.glob("*.timings")):
        exp_id = f.stem
        kv = {}
        for line in f.read_text().splitlines():
            parts = line.split()
            if len(parts) == 2:
                kv[parts[0]] = float(parts[1])
        out[exp_id] = StageTimings(
            copy=kv.get("copy", float("nan")),
            mpiexec=kv.get("mpiexec", float("nan")),
            gather=kv.get("gather", float("nan")),
            total=kv.get("total", float("nan")),
        )
    return out


def install_timed_local_execute() -> tuple[pathlib.Path, pathlib.Path]:
    """Replace local_execute.sh with the timed wrapper.

    Returns (original_path, backup_path); call restore_local_execute() in
    a finally block to put the original back.
    """
    timed_src = pathlib.Path(__file__).parent / "_bench_local_execute.sh"
    target = pathlib.Path(udales_fm.LOCAL_EXECUTE_SCRIPT)
    backup = target.with_suffix(".sh.bench-bak")
    if not backup.exists():
        shutil.copy(target, backup)
    shutil.copy(timed_src, target)
    target.chmod(0o755)
    return target, backup


def restore_local_execute(target: pathlib.Path, backup: pathlib.Path) -> None:
    if backup.exists():
        shutil.copy(backup, target)
        target.chmod(0o755)
        backup.unlink()


def set_nproc_in_namoptions(case_dir: pathlib.Path, ncpu: int) -> None:
    """Write nprocx/nprocy into the case's namoptions before model construction."""
    nprocx, nprocy = NPROC_FACTORIZATIONS[ncpu]
    candidates = list(case_dir.glob("namoptions.*"))
    if not candidates:
        raise RuntimeError(f"No namoptions in {case_dir}")
    nm = NamoptionsFile(candidates[0])
    nm.set_value("RUN", "nprocx", nprocx)
    nm.set_value("RUN", "nprocy", nprocy)
    nm.write()


def build_ensemble(
    project_root: pathlib.Path,
    ncpu: int,
    workers: int,
) -> EnsembleForwardModel:
    case_dir = project_root / CASE_DIR
    set_nproc_in_namoptions(case_dir, ncpu)

    fm = ForwardModel(
        case_dir=case_dir,
        experiment_name=EXPERIMENT_NAME,
        ncpu=ncpu,
        simulation_time=SIMULATION_TIME,
        output_frequency=OUTPUT_FREQUENCY,
        spinup_time=SPINUP_TIME,
        save_only_last_timestep=False,
        verbose=False,
        boundary_condition="periodic",
        nx=DOMAIN["nx"],
        ny=DOMAIN["ny"],
        nz=DOMAIN["nz"],
        bounds=DOMAIN["bounds"],
    )
    fm.run_preprocessing(python_or_matlab="python")

    ens = EnsembleForwardModel(
        forward_model=fm,
        ensemble_size=ENSEMBLE_SIZE,
        num_parallel_processes=workers,
        num_cpus_per_process=1,
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
    if not timings:
        raise RuntimeError(f"No timing files captured in {bench_dir}")

    copies = [t.copy for t in timings.values()]
    mpis = [t.mpiexec for t in timings.values()]
    gathers = [t.gather for t in timings.values()]
    totals = [t.total for t in timings.values()]

    return {
        "ncpu": ncpu,
        "workers": workers,
        "wall_total": wall,
        "n_members": len(timings),
        "member_total_min": min(totals),
        "member_total_mean": statistics.fmean(totals),
        "member_total_max": max(totals),
        "member_mpiexec_mean": statistics.fmean(mpis),
        "member_gather_mean": statistics.fmean(gathers),
        "member_copy_mean": statistics.fmean(copies),
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
        help="Comma-separated list of ncpu:workers pairs, e.g. '1:1,1:8,4:8'",
    )
    parser.add_argument(
        "--out",
        type=pathlib.Path,
        default=pathlib.Path(".temp/bench/ensemble_scaling.csv"),
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
    bench_dir = project_root / ".temp/bench/timings"
    args.out.parent.mkdir(parents=True, exist_ok=True)

    os.environ["BENCH_TIMING_DIR"] = str(bench_dir)
    target, backup = install_timed_local_execute()

    case_dir = project_root / CASE_DIR
    namoptions_paths = list(case_dir.glob("namoptions.*"))
    namoptions_snapshot = {p: p.read_text() for p in namoptions_paths}

    rows: list[dict] = []
    try:
        for ncpu, workers in sweeps:
            print(f"\n=== sweep ncpu={ncpu} workers={workers} ===", flush=True)
            row = run_one_sweep(project_root, ncpu, workers, bench_dir)
            print(
                f"    wall={row['wall_total']:.2f}s "
                f"member_total mean={row['member_total_mean']:.2f}s "
                f"max={row['member_total_max']:.2f}s | "
                f"mpiexec={row['member_mpiexec_mean']:.2f}s "
                f"gather={row['member_gather_mean']:.2f}s "
                f"copy={row['member_copy_mean']:.2f}s",
                flush=True,
            )
            rows.append(row)
    finally:
        restore_local_execute(target, backup)
        for p, txt in namoptions_snapshot.items():
            p.write_text(txt)

    fieldnames = list(rows[0].keys())
    with open(args.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"\nWrote CSV: {args.out}")

    print("\nSummary (sorted by wall_total):")
    print(
        f"{'ncpu':>4} {'workers':>7} {'wall':>8} {'member_avg':>10} "
        f"{'mpi_avg':>9} {'gather_avg':>10} {'copy_avg':>9}"
    )
    for row in sorted(rows, key=lambda r: r["wall_total"]):
        print(
            f"{row['ncpu']:>4} {row['workers']:>7} "
            f"{row['wall_total']:>8.2f} {row['member_total_mean']:>10.2f} "
            f"{row['member_mpiexec_mean']:>9.2f} "
            f"{row['member_gather_mean']:>10.2f} "
            f"{row['member_copy_mean']:>9.2f}"
        )


if __name__ == "__main__":
    main()
