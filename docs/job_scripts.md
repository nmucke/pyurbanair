# Job scripts — HPC / cluster submission reference

Scripts for running ESMDA, ground-truth, and training-data jobs on Snellius,
DelftBlue, and a local workstation. Three clusters share the same
experiment definition; only the SLURM wrappers differ.

Prerequisite reading: [codebase_guide.md](codebase_guide.md) §5 (Hydra config)
and [neural_surrogates.md](neural_surrogates.md) §1–4 (training-data pipeline).

---

## Layout at a glance

```
job_scripts/
├── snellius/          # SURF Snellius (rome/genoa CPUs, 128/192 cores per node)
│   ├── submit.sh          # high-level ESMDA wrapper (truth-inline mode)
│   ├── common.sh          # shared sweep defaults + COMMON_RUN_FLAGS
│   ├── sweep_base.sh      # sweep engine (submits one sbatch per swept point)
│   ├── templates/esmda.slurm   # generic job body used by submit.sh
│   ├── pylbm/   pyudales/   pypalm/
│   │   ├── rollout_esmda_from_truth.slurm   # direct sbatch runner (loaded-truth mode)
│   │   └── sweep_{domain,ensemble,esmda_steps,interval}_rollout_esmda_from_truth.sh
│   ├── ground_truth.slurm
│   ├── generate_training_data.slurm
│   ├── eval_sweep.slurm
│   ├── visualize_run.slurm, visualize_run_all.slurm
│   ├── trim_and_visualize.slurm
│   ├── make_state_small.slurm
│   ├── plot_state_slices.slurm
│   ├── run_esmda_test.slurm
│   ├── figures_static.slurm, figures_animations.slurm
│   └── compare_state_runs.slurm, visualize_state_run.slurm
│
├── delftblue/         # TU Delft DelftBlue (compute-p1 / compute-p2, max 64 cores)
│   ├── submit.sh
│   ├── common.sh, sweep_base.sh, templates/esmda.slurm
│   ├── pylbm/   pyudales/   pypalm/
│   │   ├── rollout_esmda_from_truth.slurm
│   │   └── sweep_{domain,ensemble,esmda_steps,interval}_rollout_esmda_from_truth.sh
│   ├── sweep_state_estimation_rollout_esmda_from_truth.sh
│   ├── ground_truth.slurm, generate_training_data.slurm
│   ├── eval_sweep.slurm, run_esmda_test.slurm
│   ├── visualize_run.slurm, trim_and_visualize.slurm
│   ├── make_state_small.slurm, plot_state_slices.slurm
│   └── pypalm/m0_capture.{py,slurm}  m0_diff.py
│                       m1_direct_run.{py,slurm}  m2_smoke.slurm
│
└── local/             # workstation (no SLURM, sequential runs)
    ├── common.sh, sweep_base.sh, eval_sweep.sh
    ├── pylbm/   pyudales/   pypalm/
    │   ├── rollout_esmda_from_truth.sh
    │   └── sweep_{domain,ensemble,esmda_steps,interval}_rollout_esmda_from_truth.sh
    └── neural_surrogate/
        ├── rollout_esmda_from_truth.sh
        └── sweep_{ensemble,esmda_steps,interval}_rollout_esmda_from_truth.sh
```

---

## Snellius

Full README: [job_scripts/snellius/README.md](../job_scripts/snellius/README.md)

### `submit.sh` — high-level ESMDA wrapper

[`job_scripts/snellius/submit.sh`](../job_scripts/snellius/submit.sh) is the
entry point for ad-hoc ESMDA runs. It generates the truth **inline** (no
pre-simulated ground truth needed) using `run.truth_dir=null`.

```bash
job_scripts/snellius/submit.sh <model> <size> [hydra overrides...]
```

**Argument pattern**

| Argument | Values |
|----------|--------|
| `<model>` | `pylbm` \| `pyudales` \| `pypalm` — the assimilation forward model (and truth model, unless `TRUTH_MODEL` overrides it) |
| `<size>` | `tiny` \| `small` \| `medium` \| `large` \| `xlarge` |

**Size → SLURM allocation mapping**

| `<size>` | ensemble_size | partition | cores | walltime default |
|---------:|--------------|-----------|------:|-----------------|
| `tiny`   | 4            | rome      | 16    | 00:30:00 |
| `small`  | 32           | rome      | 32    | 16:00:00 |
| `medium` | 64           | rome      | 64    | 24:00:00 |
| `large`  | 64           | rome      | 64    | 48:00:00 |
| `xlarge` | 96           | rome      | 96    | 96:00:00 |
| >128     | —            | genoa     | ≤192  | — |

One core per ensemble member, rounded up to the partition's minimum billable
share (16 on `rome`, 24 on `genoa`). Pass `ensemble.ensemble_size=N` to size the
allocation for a custom N rather than the size label's default.

**Environment variables**

| Variable | Effect |
|----------|--------|
| `TRUTH_MODEL=<m>` | Generate truth with a different solver (twin experiment) |
| `WALLTIME=HH:MM:SS` | Override the per-size default wall time |
| `DRY_RUN=1` | Print sizing and `sbatch` command without submitting |

**Per-job isolation**: each submission creates a private working directory under
`/scratch-shared/$USER/urbanair_runs/<timestamp>-<jobname>-<pid>/` populated
with symlinks back to the repo. Hydra artifacts and cwd-relative writes land in
that per-job dir so back-to-back submissions cannot clobber each other. The
workdir is tiny (symlinks) and `/scratch-shared` auto-purges it.

**pylbm isolation**: because the LBM Fortran build mutates its own source tree
(object files, generated F90, the `boltzmann` binary), the template copies
`libs/pylbm/LBM/` to `$RUN_TEMP_DIR/LBM/` and sets `PYLBM_LBM_PATH` so
concurrent submissions build against their own private copy.

**Results**: final outputs land in
`/projects/prjs2075/urbanair/esmda/<truth>_to_<assim>_<size>_<jobid>`.
Intermediate solver I/O goes to `/scratch-shared/$USER/urbanair_temp/<jobid>`
(cleaned on success, left on failure for post-mortem).

Logs: `job_scripts/snellius/out_files/slurm-<model>_<size>-<jobid>.{out,err}`

### Templates and per-backend `rollout_esmda_from_truth.slurm`

[`job_scripts/snellius/templates/esmda.slurm`](../job_scripts/snellius/templates/esmda.slurm)
is the generic job body used by `submit.sh`. It is driven by
`PUA_SIZE` / `PUA_ENSEMBLE_SIZE` / `PUA_NUM_PARALLEL` / `PUA_TRUTH_MODEL` /
`PUA_ASSIM_MODEL` injected by the wrapper, and calls
`scripts/esmda/run_esmda.py` with `run.truth_dir=null` (truth generated inline).
It is not meant to be `sbatch`-ed directly.

For runs against a **pre-simulated ground truth** use the per-backend SLURM
runners instead:

- [`job_scripts/snellius/pylbm/rollout_esmda_from_truth.slurm`](../job_scripts/snellius/pylbm/rollout_esmda_from_truth.slurm)
- [`job_scripts/snellius/pyudales/rollout_esmda_from_truth.slurm`](../job_scripts/snellius/pyudales/rollout_esmda_from_truth.slurm)
- [`job_scripts/snellius/pypalm/rollout_esmda_from_truth.slurm`](../job_scripts/snellius/pypalm/rollout_esmda_from_truth.slurm)

Each runner sources
[`job_scripts/snellius/common.sh`](../job_scripts/snellius/common.sh) (shared
experiment defaults) and adds only what differs per backend: `ASSIM_MODEL`,
`cuda=false` + private LBM copy (pylbm); `temp_dir`/`output_dir` (pyudales);
`PALM` MPI env + `domain.nz=16` floor + `PYPALM_USE_DIRECT_RUN=1` (pypalm).
The runners are directly `sbatch`-able; sweep values (`NX`, `NY`, `NZ`,
`ENSEMBLE_SIZE`, `NUM_ESMDA_STEPS`, `INTERVAL_SECONDS`) are injected via
`--export` by the sweep launchers but have sensible per-runner defaults.

### Shared sweep infrastructure

[`job_scripts/snellius/common.sh`](../job_scripts/snellius/common.sh) is the
single source of truth for every Hydra override that is identical across all
three backends: domain bounds, sensor coordinates, time horizon
(`SIMULATION_TIME=180s`, `OUTPUT_FREQUENCY=2s`, `SPINUP_TIME=50s`), number
of assimilation windows (`NUM_ASSIM_WINDOWS=6`), dynamic smoother flags
(`esmda/smoother=dynamic`, `params@{truth,prior}_params=dynamic_{truth,}`),
localization flags, and the `COMMON_RUN_FLAGS` array. Ground truth path is
resolved and validated here.

[`job_scripts/snellius/sweep_base.sh`](../job_scripts/snellius/sweep_base.sh)
defines the canonical swept value lists and their per-row wall times, and
**submits one `sbatch` job per swept point**. Partition and `--cpus-per-task`
are auto-sized to the ensemble size (one core per member, no cap).

Canonical sweep value lists (edit here once for all backends):

| Sweep kind | Values |
|------------|--------|
| `domain`   | grid resolutions, e.g. `60x80x32` (ground-truth ratio `25:20:8`, coarse → GT) |
| `ensemble` | sizes `8 16 32 64 96` at fixed `60x80x16` |
| `steps`    | ESMDA iterations `1 2 3 4` at fixed grid + ensemble |
| `interval` | `esmda.interval_seconds` values `10 20 30 60` (s) at fixed grid + ensemble + steps |

Each backend folder holds four thin wrappers that delegate to `sweep_base.sh`:

```bash
# Submit the domain sweep for all three backends:
for m in pyudales pylbm pypalm; do
  bash job_scripts/snellius/$m/sweep_domain_rollout_esmda_from_truth.sh
done
# Extra hydra overrides reach every job:
bash job_scripts/snellius/pylbm/sweep_ensemble_rollout_esmda_from_truth.sh esmda.seed=1
```

Results land in
`/projects/prjs2075/urbanair/assim_from_ground_truth/<RUN_TAG>` where
`RUN_TAG=<model>_nx<NX>_ny<NY>_nz<NZ>_ens<E>_steps<S>_int<I>[_localization]`.
No two configurations or backends collide. Localization is off by default;
enable with `USE_LOCALIZATION=true`.

### Standalone utility jobs

All are self-contained (edit the `CONFIG` block, then `sbatch` directly):

| Script | What it runs |
|--------|-------------|
| [`ground_truth.slurm`](../job_scripts/snellius/ground_truth.slurm) | `scripts/run_forward_model.py params=dynamic_truth` — generates a time-varying ground truth (single or rolled-out forward simulation) |
| [`generate_training_data.slurm`](../job_scripts/snellius/generate_training_data.slurm) | `scripts/neural_surrogate/generate_training_data.py` — surrogate training dataset; 96-core rome node; output under `/projects/prjs2075/urbanair/training_data/` |
| [`eval_sweep.slurm`](../job_scripts/snellius/eval_sweep.slurm) | `compute_sweep_metrics.py` → `compare_sweep_results.py`; `MODELS` env restricts both stages |
| [`visualize_run.slurm`](../job_scripts/snellius/visualize_run.slurm) | Regenerate the figure set for one ESMDA run |
| [`visualize_run_all.slurm`](../job_scripts/snellius/visualize_run_all.slurm) | Same, for all runs in a folder |
| [`trim_and_visualize.slurm`](../job_scripts/snellius/trim_and_visualize.slurm) | `trim_spinup.py` → `visualize_ground_truth.py` |
| [`make_state_small.slurm`](../job_scripts/snellius/make_state_small.slurm) | Stream a reduced copy of a large ground-truth state |
| [`plot_state_slices.slurm`](../job_scripts/snellius/plot_state_slices.slurm) | z-slice plots and mp4 animation |
| [`run_esmda_test.slurm`](../job_scripts/snellius/run_esmda_test.slurm) | Quick ESMDA smoke run against an on-disk truth (`TRUTH_DIR` env) |
| [`figures_static.slurm`](../job_scripts/snellius/figures_static.slurm) / [`figures_animations.slurm`](../job_scripts/snellius/figures_animations.slurm) | Batch figure/animation generation |
| [`compare_state_runs.slurm`](../job_scripts/snellius/compare_state_runs.slurm) / [`visualize_state_run.slurm`](../job_scripts/snellius/visualize_state_run.slurm) | State-estimation run comparison and visualization |

### Snellius quick reference

| Task | Command |
|------|---------|
| Ad-hoc ESMDA run (truth inline) | `job_scripts/snellius/submit.sh pylbm small` |
| Twin experiment (truth=pyudales, assim=pylbm) | `TRUTH_MODEL=pyudales job_scripts/snellius/submit.sh pylbm small` |
| Preview sizing without submitting | `DRY_RUN=1 job_scripts/snellius/submit.sh pypalm xlarge` |
| ESMDA from pre-simulated truth | `sbatch job_scripts/snellius/pyudales/rollout_esmda_from_truth.slurm` |
| Domain sweep (one backend) | `bash job_scripts/snellius/pylbm/sweep_domain_rollout_esmda_from_truth.sh` |
| Ensemble sweep (all three backends) | `for m in pyudales pylbm pypalm; do bash job_scripts/snellius/$m/sweep_ensemble_rollout_esmda_from_truth.sh; done` |
| Generate time-varying ground truth | `sbatch job_scripts/snellius/ground_truth.slurm` |
| Generate surrogate training data | `sbatch job_scripts/snellius/generate_training_data.slurm` |
| Post-process sweep → figures | `sbatch job_scripts/snellius/eval_sweep.slurm` |

---

## DelftBlue

Full README: [job_scripts/delftblue/README.md](../job_scripts/delftblue/README.md)

**Partitions**: `compute-p1` (48 cores / 185 GB per node, 218 nodes) for
requests ≤ 48 cores; `compute-p2` (64 cores / 250 GB, 90 nodes) above that.
The old `compute` partition is drained. Memory ceiling is ~3.9 GB per core.

### `submit.sh` — high-level ESMDA wrapper

[`job_scripts/delftblue/submit.sh`](../job_scripts/delftblue/submit.sh) has the
same interface as the Snellius wrapper but targets DelftBlue's partitions and
memory constraints.

```bash
job_scripts/delftblue/submit.sh <model> <size> [hydra overrides...]
```

**Size → SLURM allocation mapping**

| `<size>` | ensemble_size | cores | mem-per-cpu | walltime default |
|---------:|--------------|------:|------------|-----------------|
| `tiny`   | 4            | 4     | 3G / 2G (pypalm) | 00:30:00 |
| `small`  | 32           | 32    | 3G / 2G (pypalm) | 04:00:00 |
| `medium` | 64           | 64    | 3G / 2G (pypalm) | 12:00:00 |
| `large`  | 64           | 64    | 3G / 2G (pypalm) | 24:00:00 |
| `xlarge` | 96 (pypalm oversubscribed at 64 cores) | 64 | 2G | 24:00:00 |

One core per worker, capped at one 64-core compute node. pypalm tolerates
oversubscription (nested `mpirun` + OpenMPI yield flags), so for pypalm the
wrapper allows up to 96 workers on 64 cores — matching the historical xlarge
pattern.

**Environment variables**: same as Snellius (`TRUTH_MODEL`, `WALLTIME`, `DRY_RUN`)
plus `PYPALM_USE_DIRECT_RUN` (default `1` — bypasses `palmrun`/`palmbuild` and
runs the prebuilt PALM binary directly, saving ~130 s per invocation; clear to
empty to revert to the `palmrun` path).

**DelftBlue-specific environment** (applied by the template when pypalm is
involved): `OMPI_MCA_pml=ob1`/TCP, `OMPI_MCA_osc=pt2pt` (OpenMPI <5 still
needs this; Snellius does not), CMakeCache cleanup, and a `bash -c` wrapper
that strips nvhpc's `CC`/`F90`/... so `palmbuild` picks conda's gfortran.
pypalm's fast-IO working dir routes to node-local `/tmp` for many-small-file
build-tree copies.

**Results**: `/projects/urbanair`. Scratch: `/scratch/$USER/urbanair_temp/<jobid>`.

### Rollout-ESMDA-from-truth sweeps

The per-backend runners and sweep wrappers mirror Snellius exactly (same
`common.sh`/`sweep_base.sh` pattern, same four sweep kinds), but cap at a
64-core node. See Snellius section for the sweep mechanics; the main difference
is the `--mem-per-cpu` (2G for pypalm, 3G otherwise) and partition selection.

Files: [`job_scripts/delftblue/pyudales/`](../job_scripts/delftblue/pyudales/),
[`job_scripts/delftblue/pylbm/`](../job_scripts/delftblue/pylbm/),
[`job_scripts/delftblue/pypalm/`](../job_scripts/delftblue/pypalm/)

On DelftBlue, localization defaults to **off** in `common.sh`
(`USE_LOCALIZATION=false`); set `USE_LOCALIZATION=correlation` (or `distance`)
to enable. The `sweep_state_estimation_rollout_esmda_from_truth.sh` script (below)
demonstrates the localization/SVD methods head-to-head.

### State-estimation methods sweep

[`job_scripts/delftblue/sweep_state_estimation_rollout_esmda_from_truth.sh`](../job_scripts/delftblue/sweep_state_estimation_rollout_esmda_from_truth.sh)
submits the **same experiment once per state-update strategy** of
`esmda/smoother=state_and_dynamic`, using 30 s windows / 2 knots / 13 windows
(≈390 s total) / 10 s observation intervals:

| Method key | Update strategy |
|------------|----------------|
| `corr_ic`  | Correlation-based localization (Vossepoel 2025), updates the window IC |
| `dist_ic`  | Physical-distance-based localization, updates the window IC |
| `svd_ic`   | Reduced-SVD state update (basis from all window snapshots), IC only |
| `svd_all`  | As `svd_ic` + post-loop joint Kalman update of every window time step (`esmda.final_time_smoothing=true`) |

```bash
bash job_scripts/delftblue/sweep_state_estimation_rollout_esmda_from_truth.sh pyudales
bash job_scripts/delftblue/sweep_state_estimation_rollout_esmda_from_truth.sh pylbm esmda.seed=1
```

Each method lands in its own results dir via a `RUN_SUFFIX`. SVD methods
require the in-memory ensemble (`run.results_dir` unset) and are incompatible
with state localization (the constructor raises).

### PALM overhead investigation scripts (`pypalm/m0/m1/m2`)

These are **benchmark / investigation** scripts for the PALM per-invocation
overhead work; they are not part of the standard ESMDA submission pattern:

| Script | Purpose |
|--------|---------|
| [`pypalm/m0_capture.slurm`](../job_scripts/delftblue/pypalm/m0_capture.slurm) + [`m0_capture.py`](../job_scripts/delftblue/pypalm/m0_capture.py) | Capture: run pypalm tiny twice (combine on / off), keep palmrun tempdirs on `/scratch` (no EXIT trap), stash everything for offline diffing |
| [`pypalm/m0_diff.py`](../job_scripts/delftblue/pypalm/m0_diff.py) | Diff the two M0 stash outputs |
| [`pypalm/m1_direct_run.slurm`](../job_scripts/delftblue/pypalm/m1_direct_run.slurm) + [`m1_direct_run.py`](../job_scripts/delftblue/pypalm/m1_direct_run.py) | Unit test: drive `pypalm.direct_palm.run_direct` against the same tiny config as M0, verify u/v/w match the M0 palmrun reference, print phase timings |
| [`pypalm/m2_smoke.slurm`](../job_scripts/delftblue/pypalm/m2_smoke.slurm) | Smoke test: full ESMDA rollout pipeline with `PYPALM_USE_DIRECT_RUN=1`, 1 member / 1 window / 1 ESMDA step — exercises every code path fast |

Submit them directly with `sbatch`; they write to `/scratch/$USER/m0_capture/`
and `/scratch/$USER/m1_direct/` respectively.

### Standalone utility jobs

Same set as Snellius, with DelftBlue paths and partition labels. Note:
**DelftBlue has no `ffmpeg` module** — the mp4 step in `plot_state_slices.slurm`
fails unless you run `pixi add ffmpeg` once on the login node or export
`FFMPEG_BIN`.

| Script | What it runs |
|--------|-------------|
| [`ground_truth.slurm`](../job_scripts/delftblue/ground_truth.slurm) | Time-varying ground truth; outputs under `/projects/urbanair/ground_truth/` |
| [`generate_training_data.slurm`](../job_scripts/delftblue/generate_training_data.slurm) | Surrogate training data; full 64-core `compute-p2` node; outputs under `/projects/urbanair/training_data/pyudales_<size>` |
| [`eval_sweep.slurm`](../job_scripts/delftblue/eval_sweep.slurm) | Post-process sweep → metrics + figures; runs on a compute node (stage 1 opens large states) |
| [`run_esmda_test.slurm`](../job_scripts/delftblue/run_esmda_test.slurm) | Quick ESMDA smoke run against an on-disk truth (`TRUTH_DIR` env); outputs under `test_outputs/` |
| [`visualize_run.slurm`](../job_scripts/delftblue/visualize_run.slurm) | Regenerate figure set for one run |
| [`trim_and_visualize.slurm`](../job_scripts/delftblue/trim_and_visualize.slurm) | `trim_spinup.py` → `visualize_ground_truth.py` |
| [`make_state_small.slurm`](../job_scripts/delftblue/make_state_small.slurm) | Reduced copy of a large ground-truth state |
| [`plot_state_slices.slurm`](../job_scripts/delftblue/plot_state_slices.slurm) | z-slice plots + mp4 animation (needs ffmpeg; see note) |

### DelftBlue quick reference

| Task | Command |
|------|---------|
| Ad-hoc ESMDA run (truth inline) | `job_scripts/delftblue/submit.sh pylbm small` |
| Twin experiment | `TRUTH_MODEL=pyudales job_scripts/delftblue/submit.sh pylbm small` |
| Preview sizing | `DRY_RUN=1 job_scripts/delftblue/submit.sh pypalm xlarge` |
| Revert to palmrun path | `PYPALM_USE_DIRECT_RUN= job_scripts/delftblue/submit.sh pypalm small` |
| ESMDA from pre-simulated truth | `sbatch job_scripts/delftblue/pyudales/rollout_esmda_from_truth.slurm` |
| Domain sweep (one backend) | `bash job_scripts/delftblue/pylbm/sweep_domain_rollout_esmda_from_truth.sh` |
| State-estimation methods sweep | `bash job_scripts/delftblue/sweep_state_estimation_rollout_esmda_from_truth.sh pyudales` |
| Generate ground truth | `sbatch job_scripts/delftblue/ground_truth.slurm` |
| Generate surrogate training data | `sbatch job_scripts/delftblue/generate_training_data.slurm` |
| PALM overhead capture (M0) | `sbatch job_scripts/delftblue/pypalm/m0_capture.slurm` |
| PALM direct-run unit test (M1) | `sbatch job_scripts/delftblue/pypalm/m1_direct_run.slurm` |
| PALM direct-run smoke (M2) | `sbatch job_scripts/delftblue/pypalm/m2_smoke.slurm` |

---

## Local (no SLURM)

Full README: [job_scripts/local/README.md](../job_scripts/local/README.md)

Runs `scripts/esmda/run_esmda.py` **directly** in your shell — no `sbatch`, no
`module`, no wall clock. All three CFD backends plus the neural surrogate are
supported. Runs go **sequentially** (no job scheduler); within each run the
ensemble is parallelised across up to `LOCAL_MAX_PARALLEL` processes (default
16, set in `common.sh`).

### `common.sh` — shared defaults

[`job_scripts/local/common.sh`](../job_scripts/local/common.sh) is the single
source of shared experiment configuration. Every value is env-overridable.

| Group | Key variables |
|-------|--------------|
| Paths | `RESULTS_ROOT` (default `/export/scratch2/ntm/results/assim_from_ground_truth`), `TEMP_ROOT`, `GROUND_TRUTH_DIR`, `GROUND_TRUTH_MODEL` |
| Domain / sensors | `CASE`, `X/Y/Z_BOUNDS`, `X/Y/Z_POINTS` |
| Windows | `NUM_ASSIM_WINDOWS=6` |
| Time horizon | `SIMULATION_TIME=180s`, `OUTPUT_FREQUENCY=2s`, `SPINUP_TIME=50s` |
| Parameters | `NUM_TIME_POINTS=6`, `DYNAMIC_PARAM_FLAGS` (dynamic smoother + param groups) |
| Parallelism | `LOCAL_MAX_PARALLEL=16` (CPU backends), `ENSEMBLE_SIZE=64` |
| Misc | `SEED=0`, `SKIP_VIZ=false`, `USE_LOCALIZATION=false`, `TRUNCATION_CORRELATION=0.3` |
| BLAS | `OMP/MKL/OPENBLAS_NUM_THREADS=1`, `PYTHONUNBUFFERED=1` |

Grid resolution (`NX`/`NY`/`NZ`), `ENSEMBLE_SIZE`, `NUM_ESMDA_STEPS`, and
`INTERVAL_SECONDS` are **not** in `common.sh` — they live in each individual
runner so sweep launchers can inject one value per run.

The file validates that `$GROUND_TRUTH_DIR/state.nc` and `params.nc` exist
before any run starts.

### Per-backend runners

| Backend | pixi env | Parallelism | Notes |
|---------|----------|-------------|-------|
| `pylbm` | `cuda` | **Single process** (`num_parallel=1` hard-pinned) | Ensemble evaluated sequentially on GPU; private LBM build copy via `PYLBM_LBM_PATH` |
| `pyudales` | `dev` | `min(ensemble, LOCAL_MAX_PARALLEL)` workers | Per-run `temp_dir`/`output_dir` |
| `pypalm` | `dev` | `min(ensemble, LOCAL_MAX_PARALLEL)` workers | `nz` floored at 16; nested per-member MPI with oversubscribe; `PYPALM_USE_DIRECT_RUN=1` by default |
| `neural_surrogate` | `cuda` | Hybrid: pyudales CPU spin-up (`min(ensemble, LOCAL_MAX_PARALLEL)` workers) + single batched GPU rollout | Grid **pinned** to trained resolution (60×80×16); `MODEL_DIR` selects weights (default `model_weights/unet_convnext_medium`); `JAX_PLATFORMS=cpu` set to keep JAX off the GPU |

A single run:

```bash
bash job_scripts/local/pyudales/rollout_esmda_from_truth.sh
bash job_scripts/local/pylbm/rollout_esmda_from_truth.sh esmda.num_steps=4
bash job_scripts/local/neural_surrogate/rollout_esmda_from_truth.sh
```

Override the ground-truth location:

```bash
GROUND_TRUTH_DIR=/data/urbanair/truth \
  bash job_scripts/local/pyudales/rollout_esmda_from_truth.sh
```

### Sweeps

[`job_scripts/local/sweep_base.sh`](../job_scripts/local/sweep_base.sh) defines
the same four value lists as the Snellius/DelftBlue engines but runs each point
**sequentially in the foreground** rather than submitting sbatch jobs. A failing
point is reported but does not abort the rest. Sweeps always pass
`run.skip_viz=true` (the comparison figures are drawn afterwards by
`eval_sweep.sh`). The neural surrogate folder omits the domain sweep (grid is
fixed at 60×80×16).

```bash
# Domain sweep, all backends:
for m in pyudales pylbm pypalm; do
  bash job_scripts/local/$m/sweep_domain_rollout_esmda_from_truth.sh
done

# Ensemble sweep, neural surrogate:
bash job_scripts/local/neural_surrogate/sweep_ensemble_rollout_esmda_from_truth.sh

# Control parallelism:
LOCAL_MAX_PARALLEL=8 bash job_scripts/local/pyudales/sweep_domain_rollout_esmda_from_truth.sh
```

### `eval_sweep.sh` — post-processing

[`job_scripts/local/eval_sweep.sh`](../job_scripts/local/eval_sweep.sh) runs the
two-stage evaluation pipeline:

1. `scripts/compute_sweep_metrics.py` → `sweep_metrics/` (reads large posterior states)
2. `scripts/figure_creation/compare_sweep_results.py` → `comparison/` (lightweight)

```bash
bash job_scripts/local/eval_sweep.sh /path/to/assim_from_ground_truth
# With flags to the compare stage:
bash job_scripts/local/eval_sweep.sh /path/to/runs --sweep ensemble
bash job_scripts/local/eval_sweep.sh /path/to/runs --sweep domain --linear-x
# Restrict both stages to certain backends:
MODELS="pyudales pylbm" bash job_scripts/local/eval_sweep.sh /path/to/runs
```

Results: output directories `METRICS_DIR` (default `sweep_metrics/`) and
`COMPARISON_DIR` (default `comparison/`) under the repo root.

### High-rate probe series (`run_probe_series.py`)

[`scripts/esmda/run_probe_series.py`](../scripts/esmda/run_probe_series.py) adds
the probe time series the Welch spectrum / figure S4 need to a **finished** pylbm
ESMDA run: it re-runs one window's truth and posterior members at a high output
cadence (`probes.output_frequency`, default **0.25 s**), extracts the sensor
points from each snapshot, and deletes the snapshots again. Compose it with the
same overrides the ESMDA run used and add the `probes.*` knobs:

```bash
python scripts/esmda/run_probe_series.py \
  case=barcelona model@truth_model=pylbm model@assim_model=pylbm \
  probes.run_dir=/path/to/esmda_run probes.window_index=-1 \
  probes.spinup_time=100 \
  paths.experiment_dir=$PWD/.temp_probes
```

**Leave `probes.output_frequency` alone unless you have re-derived it.** The
width of the band figure S4 scores is fixed by the sample count, and over a
300 s window 1.0 s yields 300 samples — exactly the 4-bin refusal floor, less
than a decade of frequency, in the energy-containing range. The default 0.25 s
yields 1200 samples and 18 bins. The pre-flight reports the bin count before any
solver starts and warns below a decade; `conf/run_probe_series.yaml` carries the
cadence/band table.

Budget it as a **job of its own**, and budget the *scratch* first. The solve
itself does not depend on the output cadence, but the whole window is on disk
before Python reduces any of it, so peak scratch is
`ensemble.num_parallel_processes` × one member's snapshots. At the 0.25 s
default that is **~103 GB per member on `case=barcelona`** (3200 snapshots of
224×224×32 × 5 float32 fields), i.e. **~411 GB at 4 workers** — four times the
old 1.0 s figure of ~26 GB per member. `case=xie_and_castro` goes 0.22 →
0.90 GB per member. The cheapest lever is `probes.spinup_time` (500 s of
barcelona's 800 s solve at the case default, so 5/8 of the snapshots);
`probes.include_prior=true` doubles the member count. The probe files themselves
stay in the megabytes either way.

The compute cost is one window of forward solves per member (plus the discarded
`probes.spinup_time` lead-in), parallelised over
`ensemble.num_parallel_processes` exactly like the assimilation ensemble. Two
practical notes: give it its own `paths.experiment_dir` (it clones per-member
experiment dirs under `<experiment_dir>/probe_experiments/`, and sharing scratch
with a running job is asking for trouble), and expect it to **compile once**.
`compile=false` does *not* let it borrow the assimilation run's binary: the
experiment name is compiled in, and the probe models are built under
`probe_runcase` exactly so they cannot reach the run's `runcase` experiment dir,
so a `runcase`-stamped binary reads as stale and the build is refused.
`compile=false` is only useful for a REPEAT probe run against a build tree that
already holds a probe-stamped binary — and note that moving
`paths.experiment_dir` moves the build tree with it (`<experiment_dir>/lbm_build`),
so point `PYLBM_BUILD_ROOT` (or `PYLBM_LBM_PATH`) at the cached tree rather than
trying to keep both. Use `probes.max_members=<N>` to probe a subset and
`probes.include_prior=true` to add the prior envelope.

Writes into the run dir: `truth_probes.nc`, `windows/window_{w}_probes.nc` and
(opt-in) `windows/window_{w}_probes_prior.nc`. It changes no existing artifact.

### Local quick reference

| Task | Command |
|------|---------|
| Single pyudales ESMDA run | `bash job_scripts/local/pyudales/rollout_esmda_from_truth.sh` |
| Single pypalm ESMDA run with extra hydra override | `bash job_scripts/local/pypalm/rollout_esmda_from_truth.sh esmda.num_steps=4` |
| Neural surrogate ESMDA run | `bash job_scripts/local/neural_surrogate/rollout_esmda_from_truth.sh` |
| Domain sweep (all three backends) | `for m in pyudales pylbm pypalm; do bash job_scripts/local/$m/sweep_domain_rollout_esmda_from_truth.sh; done` |
| Ensemble sweep (neural surrogate) | `bash job_scripts/local/neural_surrogate/sweep_ensemble_rollout_esmda_from_truth.sh` |
| Enable localization for one run | `USE_LOCALIZATION=true bash job_scripts/local/pyudales/rollout_esmda_from_truth.sh` |
| Post-process sweep → figures | `bash job_scripts/local/eval_sweep.sh /path/to/runs` |
| High-rate probe series for one window | `python scripts/esmda/run_probe_series.py probes.run_dir=/path/to/esmda_run` |

---

## Cross-cluster key conventions

**`COMMON_RUN_FLAGS`** — every `common.sh` builds this bash array of Hydra
overrides that are identical across all three backends for a given cluster:
ground-truth path, domain bounds, sensors, windows, time horizon, dynamic
smoother flags, and localization. Each runner expands it verbatim and only
appends what genuinely differs (assim model, grid, ensemble, ESMDA steps,
`hydra.run.dir`, backend solver flags). This is what guarantees that "all three
backends run the exact same experiment."

**RUN_TAG format** — output directories always embed the full configuration:
`<model>_nx<NX>_ny<NY>_nz<NZ>_ens<E>_steps<S>_int<I>[_localization]`. No
two configurations or backends collide.

**Localization** — off by default in every `common.sh`. Enable with
`USE_LOCALIZATION=true` (local/Snellius) or `USE_LOCALIZATION=correlation`
(delftblue, also accepts `distance`). The canonical `LOCALIZATION_FLAGS` array
is built inside `common.sh` and propagated through `COMMON_RUN_FLAGS`.

**pypalm `nz` floor** — PALM requires ≥16 vertical levels. The `nz=8` coarsest
domain-sweep row is automatically raised to `nz=16` for pypalm in every
`sweep_base.sh`, landing in a separate `nz16` output dir so it doesn't collide
with the `nz8` runs of the other backends.

**Scratch cleanup** — every runner traps `EXIT` and removes `RUN_TEMP_DIR` on
success, leaving it intact on failure for post-mortem inspection.
