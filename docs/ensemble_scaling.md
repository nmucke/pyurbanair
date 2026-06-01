# Ensemble parallel-scaling notes (uDALES, PALM, LBM)

Working notes from an investigation into why `EnsembleForwardModel`
showed only modest speedup with multiple parallel workers. Started
with uDALES, then extended to PALM and LBM. Captures what was
measured, what was changed, and what's still open.

## TL;DR

On the local 16-core / Ryzen 9 3950X / single-socket box:

| solver | optimal `num_parallel_processes` | best speedup | bottleneck |
|--------|--------------------------------:|-------------:|------------|
| uDALES | 4 | 3.15× | DRAM bandwidth (cliff at 4) |
| PALM | 8 | 2.45× | DRAM bandwidth (cliff at 8) |
| LBM | ≥ 16 (one per physical core) | ≥ 13.5× (≈ 7.5× at N=8) | none observed; compute-bound |

`conf/model/pyudales.yaml` pins `ncpu: 1` (was 4) and
`conf/ensemble.yaml` (and the `size/` overlays) set `num_parallel_processes`.
These defaults are uDALES-optimized; PALM and LBM run sub-optimally
with the global default — they want more workers. See **Open questions
/ next steps** for the per-model-ensemble proposal.

CPU pinning is implemented and verified to put workers on distinct
physical cores spread across CCXs, but doesn't move the needle for
uDALES/PALM (DRAM-bandwidth-bound) and is harmless for LBM
(compute-bound, scales cleanly with or without it). Disable globally
with `PYURBANAIR_DISABLE_CPU_PINNING=1`.

## Files changed

| file | change |
|---|---|
| `src/pyurbanair/utils/cpu_pinning.py` | **new** — topology-aware physical-core enumeration (dedupes SMT siblings, spreads across L3 cache groups), `ProcessPoolExecutor` initializer that calls `os.sched_setaffinity` on each worker. Disable with `PYURBANAIR_DISABLE_CPU_PINNING=1`. |
| `src/pyurbanair/base_ensemble_forward_model.py` | `_run_parallel` now passes `mp_context=fork`, `initializer=pin_worker_initializer`, `initargs=(cpu_queue,)` to `ProcessPoolExecutor`. Pinning enabled by default. |
| `libs/pyudales/shell_scripts/local_execute.sh` | `mpiexec` flags changed from `--oversubscribe` to `--bind-to none --oversubscribe` so MPI inherits the worker's CPU affinity instead of overriding it. |
| `scripts/_bench_local_execute.sh` | **new** — timed wrapper used by the uDALES benchmark; mirrors the `--bind-to none` flag. |
| `scripts/benchmark_ensemble_scaling.py` | **new** — uDALES benchmark. Sweeps `(ncpu, num_parallel_processes)`, captures per-stage timings (cp / mpiexec / gather), writes CSV. |
| `scripts/benchmark_palm_ensemble_scaling.py` | **new** — PALM benchmark. Monkey-patches `pypalm.ForwardModel.run_single` to record per-member timings into `BENCH_TIMING_DIR`; the patch propagates through fork to all workers. |
| `scripts/benchmark_lbm_ensemble_scaling.py` | **new** — LBM benchmark. Same monkey-patch trick; compiles LBM once at startup so the boltzmann binary matches the benchmark domain. |
| `conf/model/pyudales.yaml` / `conf/ensemble.yaml` | `ncpu: 1` (was 4); `num_parallel_processes` capped per `size/` overlay; comment explains the cap. |
| `examples/udales/experiments/xie_and_castro/namoptions.300` | `nprocx=1, nprocy=1` so `validate_and_sync_ncpu` doesn't override `ncpu=1` back to 4. |

## Benchmark numbers

All measurements on a 16-core / 32-thread Ryzen 9 3950X (4 CCXs of 4 cores each, ~50 GB/s aggregate DRAM). Pinning enabled unless noted.

### uDALES — `ncpu=1` sweep, 64×64×16 domain, runtime=60s, ensemble_size=16

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

CSVs: `.temp/bench/ensemble_scaling.csv` (no pinning) and
`.temp/bench/ensemble_scaling_pinned.csv` (with pinning).

### uDALES — original mixed-`(ncpu, workers)` sweep, no pinning

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

### PALM — `ncpu=1` sweep, 32×32×16 domain, sim_time=60s, ensemble_size=8

| workers | wall (s) | member avg (s) | speedup | inflation vs serial |
|--------:|---------:|---------------:|--------:|--------------------:|
| 1 | 78.33 |  9.79 | 1.00× | 1.00× |
| 2 | 48.73 | 12.17 | 1.61× | 1.24× |
| 4 | 38.23 | 18.70 | 2.05× | 1.91× |
| 6 | 36.83 | 22.15 | 2.13× | 2.27× |
| **8** | **31.99** | **31.59** | **2.45×** | 3.23× |
| 12 | 33.53 | 33.26 | 2.34× | 3.40× ← effectively `min(workers, ensemble_size)=8` |

CSV: `.temp/bench/palm_ensemble_scaling.csv`.

### LBM — `ncpu=1` sweep, 64×64×16 domain, sim_time=10s, ensemble_size=8

| workers | wall (s) | member avg (s) | speedup | inflation vs serial |
|--------:|---------:|---------------:|--------:|--------------------:|
| 1 | 233.07 | 29.13 | 1.00× | 1.00× |
| 2 | 115.47 | 28.82 | 2.02× | 0.99× (no inflation) |
| 4 |  60.25 | 29.85 | 3.87× | 1.02× |
| 6 |  60.15 | 30.18 | 3.87× | 1.04× |
| **8** |  **31.00** | 30.66 | **7.52×** | 1.05× |
| 12 |  30.98 | 30.54 | 7.52× | 1.05× ← saturated by N=8 |
| 16 |  30.94 | 30.60 | 7.53× | 1.05× ← saturated by N=8 |

LBM scales near-linearly with workers up to ensemble_size. Re-run at
`ensemble_size=16` to confirm the cliff is past 16 workers:

| workers | wall (s) | member avg (s) |
|--------:|---------:|---------------:|
| 8  | 61.69 | 30.56 |
| 12 | 62.15 | 31.44 ← `ceil(16/12)=2` batches; 2nd batch only fills 4/12 slots |
| **16** | **34.53** | 33.53 |

At workers=16, ensemble_size=16: one batch, 15% per-member inflation,
implied 13.5× speedup over serial. CSVs:
`.temp/bench/lbm_ensemble_scaling.csv`,
`.temp/bench/lbm_ensemble_scaling_n16.csv`.

### DelftBlue (Xeon Gold 6248R, 48 cores/node) — uDALES

Same `ncpu=1` sweep, same 64×64×16 / 60s case, `ensemble_size=32` so the
top of the sweep fills in exactly one batch. Ran with `--exclusive
--mem=0 --cpus-per-task=32`. CSV:
`.temp/bench/ensemble_scaling_delftblue_*.csv`, SLURM script:
`job_scripts/delftblue/bench_udales_scaling.slurm`.

| workers | wall (s)  | mpi/member (s) |    speedup | per-member inflation |
|--------:|----------:|---------------:|-----------:|---------------------:|
|       1 |    746.71 |          21.20 |      1.00× |                1.00× |
|       4 |    207.52 |          23.63 |      3.60× |                1.11× |
|       8 |    123.25 |          28.18 |      6.06× |                1.33× |
|      16 |     68.98 |          29.57 |     10.82× |                1.39× |
|  **32** | **55.64** |      **45.47** | **13.42×** |                2.14× |

Key findings:

- **Optimal at workers=32** for `ensemble_size=32` (one-batch fit is the
  dominant factor). Production `cfg.ensemble.num_parallel_processes` on
  DelftBlue should target `≥ ensemble_size` when feasible, capped by node
  cores.
- **Way more memory headroom than Ryzen.** At workers=16 inflation is
  1.39× (Ryzen at the same point: 7.10×). Xeon Gold 6248R has 6 memory
  channels per CPU × 2 CPUs = 12 channels per node vs Ryzen's 2.
- **No sharp cliff up to 32**, just gradual diminishing returns. 16→32
  inflates per-member by 1.54× but cuts wall time 19% (one batch beats
  two even with the inflation).

Direct answer to "1 node × 16 cpus vs 2 nodes × 8 cpus" for
`ensemble_size=32`:

|             layout | projected wall | reasoning |
|-------------------:|---------------:|-----------|
|   1 node × 32 cpus |       55.6 s   | measured; 1 batch |
|   2 nodes × 8 cpus |        ~61 s   | each node: 16 members in 2 batches × ~28 s |
|   1 node × 16 cpus |       69.0 s   | 2 batches × ~30 s |

A single big node beats both layouts. Multi-node only becomes interesting
once `ensemble_size` grows past ~48.

**Caveat — MPI_Finalize segfaults corrupt netcdf output.** OpenMPI on
DelftBlue compute nodes still SIGSEGVs inside `MPI_Finalize` after a
clean uDALES run, despite the `OMPI_MCA_pml=ob1 / btl=self,vader,tcp`
overrides in `activation_scripts/delftblue_activation.sh`. The bench
wrapper (`scripts/_bench_local_execute.sh`) tolerates exit 139 when the
run log contains `TOTAL CPU time`, and the bench driver
(`scripts/benchmark_ensemble_scaling.py`) tolerates the resulting xarray
concat failure. But the underlying issue corrupts per-member `fielddump`
netcdf mid-close — different members end up with different `xt` sizes.
ESMDA currently masks this via
`cfg.ensemble.failure.policy = "resample_from_successes"`, which only
catches `CalledProcessError` and not corrupted-but-readable netcdf —
some posterior members may silently be running on a truncated grid.
Worth investigating: try `OMPI_MCA_coll_hcoll_enable=0`, disable
`hwloc`/`prted` early shutdown, or switch to a non-conda OpenMPI build.

## Cross-solver comparison

| Solver | Domain (cells) | Best speedup | Best workers | per-member inflation at best |
|--------|---------------:|-------------:|-------------:|-----------------------------:|
| uDALES | 64×64×16 ≈ 65k | 3.15× | 4 | 1.27× |
| PALM   | 32×32×16 ≈ 16k | 2.45× | 8 | 3.23× |
| LBM    | 64×64×16 ≈ 65k | ≥ 13.5× | **≥ 16** | 1.05–1.15× |

**Caveat**: domain sizes differ across solvers. PALM was tested at
32³ vs the others' 64³, so cross-solver numbers are directional, not
exact. The qualitative ordering (LBM ≫ PALM > uDALES) almost certainly
holds because:

- LBM has high arithmetic intensity per byte (stream + collide is
  SIMD-friendly, small working set) → compute-bound.
- uDALES and PALM both run an FFT-based pressure Poisson solve every
  timestep → memory-bandwidth-bound. PALM has slightly more physics
  per cell (radiation, surface energy balance, more scalars), so its
  per-rank arithmetic intensity is higher and the cliff sits a bit
  later (8 vs 4).
- 16 simultaneous LBM workers barely contend on this Zen 2 box; even
  4 simultaneous uDALES workers already start inflating per-member.

## Why pinning didn't move the needle (for uDALES/PALM)

Pinning was the obvious first guess: 8 workers × 1 rank on 16 cores
with `--oversubscribe` could collide on SMT pairs. Verified pinning
is correctly applied:

```
Worker 957107 → cpus=[0]    # CCX0
Worker 957108 → cpus=[4]    # CCX1
Worker 957109 → cpus=[8]    # CCX2
Worker 957110 → cpus=[12]   # CCX3
```

But the benchmark improved by < 0.5% for uDALES. So the 4→8 cliff is
**not** SMT collision — it's DRAM-bandwidth saturation past 4 active
ranks. Confirmed by:

- uDALES has **no OpenMP** directives in source (`grep -r '!\$omp'`).
- Binary doesn't link a threaded FFTW symbol path (no
  `fftw_init_threads` in `nm -D`).
- `libgomp` shows up in `ldd` only because gfortran's runtime pulls it.
- Pinning correctly puts workers on distinct physical cores spread
  across CCXs, but per-member runtime still doubles 4→8.

LBM's near-linear scaling on the same hardware is independent
confirmation: when the workload isn't memory-bandwidth-bound, the
machine has plenty of headroom. The bottleneck is solver-side, not
scheduler-side.

## How to re-run the benchmarks

uDALES default sweep (10 configs, ~15 min on this box):

```bash
pixi run -e dev python scripts/benchmark_ensemble_scaling.py \
    --simulation-time 60.0 --ensemble-size 16
```

uDALES `ncpu=1` only (faster):

```bash
pixi run -e dev python scripts/benchmark_ensemble_scaling.py \
    --sweeps 1:1,1:2,1:4,1:8,1:16 \
    --simulation-time 60.0 --ensemble-size 16 \
    --out .temp/bench/ensemble_scaling_ncpu1.csv
```

PALM (requires `palmrun` — auto-installed under
`libs/pypalm/palm_model_system/` on first import):

```bash
pixi run -e dev python scripts/benchmark_palm_ensemble_scaling.py \
    --simulation-time 60.0 --ensemble-size 8
```

LBM (compiles boltzmann once at startup):

```bash
pixi run -e dev python scripts/benchmark_lbm_ensemble_scaling.py \
    --simulation-time 10.0 --ensemble-size 8
```

Custom sweep selection: `--sweeps 1:4,1:8,1:16` etc.

All output goes to `.temp/bench/*.csv` (gitignored). The uDALES
benchmark snapshots/restores the case namoptions so it doesn't dirty
the working tree.

To toggle pinning off:

```bash
PYURBANAIR_DISABLE_CPU_PINNING=1 pixi run -e dev python scripts/benchmark_ensemble_scaling.py ...
```

## Open questions / next steps

In rough priority order:

1. **Per-model `ensemble` defaults via Hydra**. The current
   `conf/ensemble.yaml` / `size/` overlays set `num_parallel_processes`,
   which is uDALES-optimal but leaves ~17% on the table for PALM (8 → 4:
   38s vs 32s) and ~2× on the table for LBM (8 → 4: 60s vs 31s). Two
   reasonable shapes:

   ```yaml
   # A per-backend ensemble overlay (e.g. selected by model) that overrides
   # num_parallel_processes for its target backend on top of conf/ensemble.yaml.
   ```

   then either (a) select per-run on the CLI
   (`ensemble=lbm model=pylbm`), or (b) have each `conf/model/*.yaml`
   carry a default `ensemble@_global_` selection. Option (b) auto-picks
   the right block from the model choice without an extra CLI flag but
   couples the two groups; option (a) keeps the groups orthogonal but
   relies on the caller to remember to pair them. Not implemented yet
   because it's a config-shape decision worth aligning on first.

2. **Re-run benchmarks on the production domains.** All three
   benchmarks use small domains tuned for fast iteration. Production
   uDALES uses `nx=50, ny=40, nz=16, simulation_time=300s`; PALM
   probably runs larger; LBM may run very large grids. The
   bandwidth-cliff position depends on per-rank working-set size, so
   the cap may move. Until measured on the real domain, treat the
   numbers above as the *order of magnitude* and not the exact
   answer.

3. **Re-run on different hardware.** On a multi-socket / NUMA box
   (e.g. EPYC, dual-socket Xeon) with more memory channels per core,
   the uDALES/PALM caps should move up. The benchmark scripts and
   pinning logic are portable.

4. **Long-lived `ProcessPoolExecutor` for ESMDA.** Currently every
   `run_ensemble()` call spins up a fresh pool and pickles a
   deepcopy of the forward model into each task. For ESMDA loops
   with many `run_ensemble()` calls, holding the pool open across
   calls would save real time. Worth profiling before doing — only
   matters if non-MPI overhead is measurable in the rollout/ESMDA
   scripts.

5. **Push LBM past `workers=16`.** With ensemble_size=16 and
   workers=16 we got 13.5× speedup at 1.15× per-member inflation.
   Worth testing workers=24/32 (using SMT siblings) for LBM only —
   it may continue to benefit since the working set is cache-resident
   per rank. Not worth doing for uDALES or PALM (already past the
   cliff).

6. **`gather_outputs.sh` rewrite — DEFERRED**. Was on the original
   list, but at `ncpu=1` there's only one dump file per member and
   the script no-ops. Only worth doing if a future workload pushes
   back to `ncpu>1`.

7. **CPU pinning across CCDs (not just CCXs).** Current
   implementation spreads across L3 cache groups (CCXs). On a 2-CCD
   chip like the 3950X you might want a `topology/die_id`-aware
   ordering that fills CCD0 before CCD1 (reduces inter-CCD
   infinity-fabric traffic), or vice versa (spreads memory channels).
   Not measured to matter yet.

## Auto-memory cross-reference

A condensed version of this is in
`/ufs/ntm/.claude/projects/-export-scratch1-ntm-pyurbanair/memory/ensemble_scaling.md`
(local to that machine; this MD is the portable copy).
