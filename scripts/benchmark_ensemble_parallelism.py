"""Benchmark ensemble forward-model wall time under different parallelism modes.

Runs ``num_calls`` back-to-back ensemble forward sweeps (no ESMDA, no
data assimilation) and prints per-call + total wall time.  Used to
quantify the speedup from:

  * the persistent worker-pool change in
    ``BaseEnsembleForwardModel`` (gated by ``PYURBANAIR_PERSIST_POOL``),
  * the slurm ``srun --exact`` fan-out in
    ``libs/pyudales/shell_scripts/local_execute.sh`` (gated by
    ``PYURBANAIR_DISABLE_SRUN`` + presence of ``$SLURM_JOB_ID``).

The script mutates ``scripts.config`` in-place with smoke-sized values
so the benchmark fits comfortably in a 30-min slurm wall.
"""

from __future__ import annotations

import argparse
import contextlib
import os
import pathlib
import sys
import time

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from pyudales.utils.namoptions_utils import NamoptionsFile  # noqa: E402

from scripts import config  # noqa: E402


@contextlib.contextmanager
def _ncpu1_namoptions(case_dir: str, experiment_name: str):
    """Temporarily set nprocx=nprocy=1 in the case namoptions and restore on exit.

    `validate_and_sync_ncpu` (libs/pyudales/.../ncpu_utils.py) silently overrides
    `UDALES_ARGS["ncpu"]` to match `nprocx*nprocy` from namoptions. So forcing
    `ncpu=1` requires the namoptions to also have `nprocx=1, nprocy=1` —
    otherwise our setting is silently bumped back up. This snapshot/restore
    keeps the user's case file clean.
    """
    path = pathlib.Path(case_dir) / f"namoptions.{experiment_name}"
    if not path.exists():
        yield
        return
    nm = NamoptionsFile(path)
    saved = (
        nm.get_value("RUN", "nprocx"),
        nm.get_value("RUN", "nprocy"),
    )
    nm.set_value("RUN", "nprocx", 1)
    nm.set_value("RUN", "nprocy", 1)
    nm.write()
    try:
        yield
    finally:
        nm = NamoptionsFile(path)
        if saved[0] is not None:
            nm.set_value("RUN", "nprocx", saved[0])
        if saved[1] is not None:
            nm.set_value("RUN", "nprocy", saved[1])
        nm.write()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ensemble-size", type=int, default=24)
    parser.add_argument("--num-parallel-processes", type=int, default=24)
    parser.add_argument("--num-calls", type=int, default=3)
    parser.add_argument("--simulation-time", type=float, default=60.0)
    parser.add_argument("--spinup-time", type=float, default=5.0)
    parser.add_argument("--label", type=str, default="benchmark")
    parser.add_argument(
        "--ncpu",
        type=int,
        default=1,
        help="MPI ranks per ensemble member. Default 1 (matches user's "
        "production usage; warmstart does not work for ncpu>1).",
    )
    args = parser.parse_args()

    config.TIME["simulation_time"] = args.simulation_time
    config.TIME["spinup_time"] = args.spinup_time
    config.TIME["output_frequency"] = 5.0
    config.ENSEMBLE["ensemble_size"] = args.ensemble_size
    config.ENSEMBLE["num_parallel_processes"] = args.num_parallel_processes
    config.ENSEMBLE["num_cpus_per_process"] = args.ncpu
    config.ENSEMBLE["failure_policy"] = "raise"
    config.UDALES_ARGS["ncpu"] = args.ncpu

    persist = os.environ.get("PYURBANAIR_PERSIST_POOL", "1") != "0"
    srun_disabled = bool(os.environ.get("PYURBANAIR_DISABLE_SRUN"))
    print(
        f"[{args.label}] ensemble={args.ensemble_size} "
        f"parallel={args.num_parallel_processes} "
        f"ncpu={args.ncpu} "
        f"calls={args.num_calls} sim_time={args.simulation_time}s "
        f"persist_pool={persist} srun={'off' if srun_disabled else 'on'} "
        f"slurm_job={os.environ.get('SLURM_JOB_ID', 'none')}"
    )

    case_dir = config.UDALES_ARGS["case_dir"]
    experiment_name = config.UDALES_ARGS["experiment_name"]
    namoptions_ctx = (
        _ncpu1_namoptions(case_dir, experiment_name)
        if args.ncpu == 1
        else contextlib.nullcontext()
    )

    with namoptions_ctx:
        forward_model = config.create_forward_model("pyudales")
        config.prepare_forward_model("pyudales", forward_model)
        ensemble_model = config.create_ensemble_forward_model(
            "pyudales", forward_model
        )

        params = config.create_parameter_ensemble("pyudales")

        per_call: list[float] = []
        t_total_start = time.perf_counter()
        for i in range(args.num_calls):
            t0 = time.perf_counter()
            _ = ensemble_model.run_ensemble(params=params, sim_name=f"bench_{i}")
            dt = time.perf_counter() - t0
            per_call.append(dt)
            print(f"[{args.label}] call {i + 1}/{args.num_calls}: {dt:.2f}s")
        total = time.perf_counter() - t_total_start

        # Drop the first call so the remaining mean reflects steady-state cost
        # (first call still pays one-time fork / module-load if persist=True
        # and the pool was just created).
        steady = per_call[1:] if len(per_call) > 1 else per_call
        mean_steady = sum(steady) / len(steady)
        print(
            f"[{args.label}] SUMMARY total={total:.2f}s "
            f"first={per_call[0]:.2f}s "
            f"steady_mean={mean_steady:.2f}s n_steady={len(steady)}"
        )

        if hasattr(ensemble_model, "close"):
            ensemble_model.close()


if __name__ == "__main__":
    main()
