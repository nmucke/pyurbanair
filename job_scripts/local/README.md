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
├── eval_sweep.sh      # post-process a runs folder -> metrics + comparison figures
├── pylbm/             # GPU backend (cuda pixi env, single process)
│   ├── rollout_esmda_from_truth.sh
│   ├── sweep_domain_rollout_esmda_from_truth.sh
│   ├── sweep_ensemble_rollout_esmda_from_truth.sh
│   ├── sweep_esmda_steps_rollout_esmda_from_truth.sh
│   └── sweep_interval_rollout_esmda_from_truth.sh
├── pyudales/          # CPU backend (dev pixi env, multi-process)
│   └── … same five files …
└── pypalm/            # CPU backend (dev pixi env, multi-process, nested MPI)
    └── … same five files …
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
bash job_scripts/local/pyudales/sweep_interval_rollout_esmda_from_truth.sh
```

The same sweep across **all three backends** (identical configs, comparable runs):

```bash
for m in pyudales pylbm pypalm; do
  bash job_scripts/local/$m/sweep_domain_rollout_esmda_from_truth.sh
done
```

Any extra arguments are forwarded verbatim as Hydra overrides to **every** run.

**Sweeps always skip the per-run visualization** (`run.skip_viz=true`, forced in
`sweep_base.sh`) — the slow animations/plots are wasted work mid-sweep; the
comparison figures are drawn afterwards by `eval_sweep.sh` from the metrics. A
single direct runner call still produces its viz (default `SKIP_VIZ=false`).

## Evaluating a sweep (`eval_sweep.sh`)

Once the runs exist, post-process them into metrics + comparison figures. Just
point it at the **folder holding all the runs** (the sweeps' `RESULTS_ROOT`,
which contains one `<model>_nx..._ens..._steps...` subdir per run):

```bash
bash job_scripts/local/eval_sweep.sh /path/to/assim_from_ground_truth
```

It runs the two-stage pipeline: `compute_sweep_metrics.py` (→ `sweep_metrics/`)
then `compare_sweep_results.py` (→ `comparison/domain` + `comparison/ensemble`).
Eval is self-contained — it needs no ground truth or solver env (each run records
its own truth path), so it just needs the runs folder.

- Folder defaults to `$RESULTS_ROOT`, then the repo-local
  `results/assim_from_ground_truth`, if you omit it.
- Anything after the folder goes to the **compare stage only**:
  ```bash
  bash job_scripts/local/eval_sweep.sh /path/to/runs --sweep ensemble
  bash job_scripts/local/eval_sweep.sh /path/to/runs --sweep domain --linear-x
  ```
- Restrict **both** stages to some backends with `MODELS` (env, not positional):
  ```bash
  MODELS="pyudales pylbm" bash job_scripts/local/eval_sweep.sh /path/to/runs
  ```
- Other env knobs: `ENV` (pixi env, default `dev`), `METRICS_DIR`, `COMPARISON_DIR`.

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
and lives in each runner (defaulted, env-overridable). Likewise `ENSEMBLE_SIZE`,
`NUM_ESMDA_STEPS` and `INTERVAL_SECONDS` (the `obs.interval_seconds` bin width).

### `sweep_base.sh` — canonical swept values (one place, all backends)

Defines the four value lists used by every backend:

- `RESOLUTIONS` — coarse → ground-truth grid (`25 20 8` … `100 80 32`).
- `ENSEMBLE_SIZES` — `8 16 32 64`, at a fixed grid.
- `ESMDA_STEPS` — `1 2 4 8`, at a fixed grid + ensemble.
- `INTERVAL_SECONDS_LIST` — `10 20 30 60`, the `obs.interval_seconds`
  time-aggregation bin width, at a fixed grid + ensemble + steps.

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
`RUN_TAG=<assim>_nx<NX>_ny<NY>_nz<NZ>_ens<E>_steps<S>_int<I>[_localization]`, so no two
configurations (or backends) collide. Heavy intermediate solver I/O goes to a
private `${TEMP_ROOT}/<RUN_TAG>_<pid>` scratch dir, removed on success and left
behind on failure for post-mortem.
