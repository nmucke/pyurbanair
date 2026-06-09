# Local job scripts

Local (no-SLURM) siblings of `job_scripts/snellius/<backend>/rollout_esmda_from_truth.slurm`.
They run time-varying-parameter **rollout ESMDA against a pre-simulated ground
truth** by invoking `scripts/run_esmda.py` **directly** in this shell — no
`sbatch`, no `module`, no partitions, no wall clock. Everything stays under the
repo by default (`results/`, `.local_runs/temp/`).

The defining property of this folder: **all three backends run the exact same
experiment** at every configuration — same ground truth, domain, windows, time
horizon, sensors and dynamic-parameter settings. Only the assimilation solver
(and its CPU/GPU execution) differs, so the runs are directly comparable. That
guarantee comes from two shared files:

- `common.sh` — every default and every Hydra override that is **identical across
  backends**, in one place (see below). Sourced by each runner.
- `sweep_base.sh` — the one sweep engine, holding the **canonical swept value
  lists**. Every backend's sweep wrappers delegate to it.

## Layout

```
local/
├── common.sh          # shared defaults + COMMON_RUN_FLAGS (sourced by every runner)
├── sweep_base.sh      # shared sweep engine + canonical value lists
├── pylbm/             # GPU backend (cuda pixi env, single process)
│   ├── rollout_esmda_from_truth.sh
│   ├── sweep_domain_rollout_esmda_from_truth.sh
│   ├── sweep_ensemble_rollout_esmda_from_truth.sh
│   └── sweep_esmda_steps_rollout_esmda_from_truth.sh
├── pyudales/          # CPU backend (dev pixi env, multi-process)
│   └── … same four files …
└── pypalm/            # CPU backend (dev pixi env, multi-process, nested MPI)
    └── … same four files …
```

## Usage

A single run with one backend (edit `common.sh` / the runner, or pass overrides):

```bash
bash job_scripts/local/pyudales/rollout_esmda_from_truth.sh
bash job_scripts/local/pylbm/rollout_esmda_from_truth.sh esmda.num_steps=4
```

A full sweep with one backend (runs each point sequentially):

```bash
bash job_scripts/local/pyudales/sweep_domain_rollout_esmda_from_truth.sh
bash job_scripts/local/pylbm/sweep_ensemble_rollout_esmda_from_truth.sh esmda.seed=1
bash job_scripts/local/pypalm/sweep_esmda_steps_rollout_esmda_from_truth.sh
```

The same sweep across **all three backends** (identical configs, comparable runs):

```bash
for m in pyudales pylbm pypalm; do
  bash job_scripts/local/$m/sweep_domain_rollout_esmda_from_truth.sh
done
```

Any extra arguments are forwarded verbatim as Hydra overrides to **every** run.

## Pointing at the ground truth

`common.sh` defaults `GROUND_TRUTH_DIR` to
`/projects/prjs2075/urbanair/ground_truth_small` (the Snellius project space).
Locally, point it at wherever the pre-simulated truth lives — the runner errors
out clearly if `state.nc` + `params.nc` are not found:

```bash
GROUND_TRUTH_DIR=/data/urbanair/ground_truth_small \
  bash job_scripts/local/pyudales/rollout_esmda_from_truth.sh
```

The leaf actually loaded is `${GROUND_TRUTH_DIR}/${GROUND_TRUTH_MODEL}_time_varying`
(`GROUND_TRUTH_MODEL=pyudales` by default); set `GROUND_TRUTH_SUBDIR=""` if
`GROUND_TRUTH_DIR` already points straight at the folder holding the `.nc` files.

## What lives where

### `common.sh` — shared defaults (one place to retune the whole suite)

Every value is env-overridable (`export VAR=… ` before invoking). It holds:

| Group                  | Keys                                                        |
|------------------------|-------------------------------------------------------------|
| Paths                  | `RESULTS_ROOT`, `TEMP_ROOT`, `GROUND_TRUTH_DIR`, `GROUND_TRUTH_MODEL` |
| Domain **size**        | `CASE`, `X/Y/Z_BOUNDS`, `X/Y/Z_POINTS` (sensors)            |
| Assimilation windows   | `NUM_ASSIM_WINDOWS`                                          |
| Time horizon           | `SIMULATION_TIME`, `OUTPUT_FREQUENCY`, `SPINUP_TIME`         |
| Dynamic parameters     | `NUM_TIME_POINTS` + `DYNAMIC_PARAM_FLAGS` (dynamic smoother + param groups) |
| Misc / localization    | `SEED`, `SKIP_VIZ`, `USE_LOCALIZATION`, `TRUNCATION_CORRELATION` |

It then builds **`COMMON_RUN_FLAGS`** — the single array of every
`run_esmda.py` Hydra override that is identical across backends. Each runner
expands it verbatim and only adds what genuinely differs (assim model, the
per-run sweep values, `hydra.run.dir`, backend solver flags). This array is what
makes "the exact same thing" enforceable rather than copy-pasted.

Note the **grid resolution `NX`/`NY`/`NZ` is NOT here** — it is a sweep parameter
and lives in each runner (defaulted, env-overridable). Likewise `ENSEMBLE_SIZE`
and `NUM_ESMDA_STEPS`.

### `sweep_base.sh` — canonical swept values (one place, all backends)

Defines the three value lists used by every backend:

- `RESOLUTIONS` — coarse → ground-truth grid (`25 20 8` … `100 80 32`).
- `ENSEMBLE_SIZES` — `8 16 32 64`, at a fixed grid.
- `ESMDA_STEPS` — `1 2 4 8`, at a fixed grid + ensemble.

Plus the `FIXED_*` values for the dimensions each sweep holds constant. Edit
these once to retune the sweeps for **all** backends. Runs sequentially; a single
failing point is reported but does not abort the rest.

## Backend differences (the only things that vary)

| Backend   | pixi env | Parallelism                                       | Solver specifics |
|-----------|----------|---------------------------------------------------|------------------|
| `pylbm`   | `cuda`   | **GPU, single process** — `num_parallel=1` hard-pinned, ensemble run sequentially (one GPU) | `cuda=true`; private LBM build copy via `PYLBM_LBM_PATH` |
| `pyudales`| `dev`    | CPU, `num_parallel` = min(ensemble, **`LOCAL_MAX_PARALLEL`** = 16 by default) | per-run `temp_dir`/`output_dir` |
| `pypalm`  | `dev`    | CPU, `num_parallel` = min(ensemble, **`LOCAL_MAX_PARALLEL`** = 16 by default) | **nz floored at 16** (PALM minimum); nested per-member MPI: pinning off, OMPI oversubscribe; direct-run by default |

Execution model: no scheduler. Runs go **sequentially**, one after another, in
the shell you launch. For the CPU backends (pyudales, pypalm) **you choose the
maximum number of parallel ensemble processes** via `LOCAL_MAX_PARALLEL` (default
16, set in `common.sh`); the actual worker count is
`min(ensemble_size, LOCAL_MAX_PARALLEL)`. Change it once in `common.sh`, per run
(`LOCAL_MAX_PARALLEL=32 bash …`), or across a whole sweep
(`LOCAL_MAX_PARALLEL=32 bash …/sweep_domain_…sh`); or pin an exact count with
`NUM_PARALLEL=…`. `pylbm` always runs on the GPU with the ensemble evaluated
sequentially — that is fixed, not overridable.

Because `pypalm` requires at least 16 vertical levels, the shared domain sweep's
coarsest row (`25 20 8`) is automatically raised to `25 20 16` for `pypalm` only;
the other backends run it as-is. The run still lands in its own `nz16` output dir,
so nothing collides.

## Outputs

Each run writes to `${RESULTS_ROOT}/<RUN_TAG>` where
`RUN_TAG=<assim>_nx<NX>_ny<NY>_nz<NZ>_ens<E>_steps<S>[_localization]`, so no two
configurations (or backends) collide. Heavy intermediate solver I/O goes to a
private `${TEMP_ROOT}/<RUN_TAG>_<pid>` scratch dir, removed on success and left
behind on failure for post-mortem.
