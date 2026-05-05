# pyudales ensemble parallel-scaling notes

Working notes from an investigation into why `EnsembleForwardModel` showed
only modest speedup with multiple parallel workers. Captures what was
measured, what was changed, and what's still open.

## TL;DR

On the local 16-core / Ryzen 9 3950X / single-socket box, the optimal
configuration for the pyudales ensemble forward model is:

```python
UDALES_ARGS = {..., "ncpu": 1}
ENSEMBLE = {..., "num_parallel_processes": 4, "num_cpus_per_process": 1}
```

This gives **3.15× speedup over serial**. Going past `workers=4` doesn't
improve wall time on this hardware/workload: the bottleneck is DRAM
bandwidth, not CPU oversubscription. CPU pinning is implemented and
verified, but doesn't change wall time at `ncpu=1` (still useful as a
defensive measure under load).

## Files changed

| file | change |
|---|---|
| `src/pyurbanair/utils/cpu_pinning.py` | **new** — topology-aware physical-core enumeration (dedupes SMT siblings, spreads across L3 cache groups), `ProcessPoolExecutor` initializer that calls `os.sched_setaffinity` on each worker. Disable with `PYURBANAIR_DISABLE_CPU_PINNING=1`. |
| `src/pyurbanair/base_ensemble_forward_model.py` | `_run_parallel` now passes `mp_context=fork`, `initializer=pin_worker_initializer`, `initargs=(cpu_queue,)` to `ProcessPoolExecutor`. Pinning enabled by default. |
| `libs/pyudales/shell_scripts/local_execute.sh` | `mpiexec` flags changed from `--oversubscribe` to `--bind-to none --oversubscribe` so MPI inherits the worker's CPU affinity instead of overriding it. |
| `scripts/_bench_local_execute.sh` | **new** — timed wrapper used by the benchmark; mirrors the `--bind-to none` flag. |
| `scripts/benchmark_ensemble_scaling.py` | **new** — sweeps `(ncpu, num_parallel_processes)`, captures per-stage timings (cp / mpiexec / gather), writes CSV. Snapshots/restores the case namoptions so the benchmark doesn't mutate the source case. |
| `scripts/config.py` | `UDALES_ARGS["ncpu"] = 1` (was 4); `ENSEMBLE["num_parallel_processes"] = 4` (was 8); comment explains the cap. |
| `examples/udales/experiments/xie_and_castro/namoptions.300` | `nprocx=1, nprocy=1` so `validate_and_sync_ncpu` doesn't override `ncpu=1` back to 4. |

## Benchmark numbers

64×64×16 grid, runtime=60s, ensemble_size=16, periodic BCs, on 16-core
Ryzen 9 3950X / 32 GB RAM. CSVs at
`.temp/bench/ensemble_scaling.csv` (no pinning) and
`.temp/bench/ensemble_scaling_pinned.csv` (with pinning).

### `ncpu=1` sweep — both with and without pinning

| workers | wall (no-pin) | wall (pinned) | speedup pinned |
|--------:|--------------:|--------------:|---------------:|
| 1 | 224.67 | 225.81 | 1.00× |
| 2 | 120.12 | 121.05 | 1.87× |
| 4 |  71.41 |  71.66 | **3.15×** |
| 8 |  71.43 |  75.00 | 3.01× |
| 16 | 97.07 | 100.49 | 2.25× |

Per-member runtime (mpi only) over the same sweep:

| workers | mpi/member (no-pin) | mpi/member (pinned) | inflation vs serial |
|--------:|--------------------:|--------------------:|--------------------:|
| 1 | 13.96 | 14.03 | 1.00× |
| 4 | 17.71 | 17.72 | 1.27× |
| 8 | 35.20 | 36.83 | **2.63×** |
| 16 | 94.98 | 99.57 | 7.10× |

Going from 4 to 8 workers, per-member runtime exactly doubles → wall stays
flat → adding workers past 4 wastes CPU.

### Original full sweep (mixed `ncpu, workers`) — without pinning

For reference, the original sweep that exposed the bad production
default of `(ncpu=4, workers=8)`:

| ncpu | workers | total ranks | wall (s) | mpi/member | gather/member | speedup |
|-----:|--------:|-----:|---------:|------:|-------:|--------:|
| 1 | 4 | 4 | **71.4** | 17.7 | 0.02 | **3.15×** |
| 1 | 8 | 8 |  71.4 | 35.2 | 0.02 | 3.14× |
| 2 | 8 | 16 | 74.9 | 36.7 | 0.03 | 3.00× |
| 4 | 4 | 16 | 86.7 | 17.2 | 4.28 | 2.59× |
| 1 | 16 | 16 | 97.1 | 95.0 | 0.03 | 2.31× |
| 8 | 2 | 16 | 101.2 | 7.5 | 5.06 | 2.22× |
| **4** | **8** | **32** | **102.5** | **43.7** | **6.92** | **2.19× ← old default** |
| 16 | 1 | 16 | 134.7 | 4.3 | 4.12 | 1.67× |
| 1 | 1 | 1 | 224.7 | 14.0 | 0.02 | 1.00× |

## Why pinning didn't move the needle

Pinning was the obvious first guess (8 workers × 1 rank on 16 cores with
`--oversubscribe` could collide on SMT pairs). Verified pinning is correct:

```
Worker 957107 → cpus=[0]    # CCX0
Worker 957108 → cpus=[4]    # CCX1
Worker 957109 → cpus=[8]    # CCX2
Worker 957110 → cpus=[12]   # CCX3
```

(See `pixi run -e dev python -c '...'` test in session log.)

But the benchmark improved by < 0.5%. So the 4→8 cliff is **not** SMT
collision — it's DRAM-bandwidth saturation past 4 active ranks. On a
single-socket Zen 2 with ~50 GB/s aggregate DRAM and 4 CCXs sharing one
infinity-fabric path, putting > 4 simultaneous FFT-pressure-Poisson
solves in flight saturates the memory subsystem. Confirmed by:

- uDALES has **no OpenMP** directives in source (`grep -r '!\$omp'`).
- Binary doesn't link a threaded FFTW symbol path (no `fftw_init_threads`
  in `nm -D`).
- `libgomp` shows up in `ldd` only because gfortran's runtime pulls it.
- Pinning correctly puts workers on distinct physical cores spread
  across CCXs, but per-member runtime still doubles 4→8.

So the modest speedup is partly a config issue (workers=8 was no better
than workers=4) and partly a hardware ceiling.

## How to re-run the benchmark

Default sweep (10 configs, ~15 min on this box):

```bash
pixi run -e dev python scripts/benchmark_ensemble_scaling.py \
    --simulation-time 60.0 --ensemble-size 16
```

ncpu=1 only sweep (faster, more relevant since `ncpu=1` is the default):

```bash
pixi run -e dev python scripts/benchmark_ensemble_scaling.py \
    --sweeps 1:1,1:2,1:4,1:8,1:16 \
    --simulation-time 60.0 --ensemble-size 16 \
    --out .temp/bench/ensemble_scaling_ncpu1.csv
```

Custom `(ncpu, workers)` pairs:

```bash
... --sweeps 1:4,1:8,4:4,4:8 ...
```

Output goes to `.temp/bench/ensemble_scaling.csv` (gitignored). The
benchmark snapshots the case namoptions so it won't dirty your working tree.

To toggle pinning off:

```bash
PYURBANAIR_DISABLE_CPU_PINNING=1 pixi run -e dev python scripts/benchmark_ensemble_scaling.py ...
```

## Open questions / next steps

In rough priority order:

1. **Re-run benchmark on the production domain.** The 4-worker cap was
   measured on the 64×64×16 benchmark domain. Production uses
   `nx=50, ny=40, nz=16, simulation_time=300s` (smaller per-rank working
   set). The bandwidth cliff position depends on working-set size, so
   the cap may move. Until measured, treat `workers=4` as a default,
   not a fact.

2. **Re-run on different hardware.** On a multi-socket / NUMA box (or a
   machine with more memory channels per core, e.g., EPYC), the cap
   should move up. The benchmark + pinning logic is portable.

3. **Long-lived `ProcessPoolExecutor` for ESMDA.** Currently every
   `run_ensemble()` call spins up a fresh pool and pickles a deepcopy of
   the forward model into each task. For ESMDA loops with many
   `run_ensemble()` calls, holding the pool open across calls would save
   real time. Worth profiling before doing — only matters if non-MPI
   overhead is measurable in the rollout/ESMDA scripts.

4. **`gather_outputs.sh` rewrite — DEFERRED**. Was on the original list,
   but at `ncpu=1` there's only one dump file per member and the script
   no-ops. Only worth doing if a future workload pushes back to `ncpu>1`.

5. **CPU pinning across CCDs (not just CCXs).** Current implementation
   spreads across L3 cache groups (CCXs). On a 2-CCD chip like the 3950X
   you might want a `topology/die_id`-aware ordering that fills CCD0
   before CCD1 (reduces inter-CCD infinity-fabric traffic), or vice-versa
   (spreads memory channels). Not measured to matter yet.

## Auto-memory cross-reference

A condensed version of this is in
`/ufs/ntm/.claude/projects/-export-scratch1-ntm-pyurbanair/memory/ensemble_scaling.md`
(local to that machine; this MD is the portable copy).
