# Snellius job scripts

Submit ESMDA runs on Snellius (CPU-only) with the `submit.sh` wrapper. It sizes
the SLURM allocation from the experiment config, so you tune the run in one
place — `conf/size/<size>.yaml` — and the requested cores follow automatically.

## Usage

```bash
job_scripts/snellius/submit.sh <model> <size> [extra hydra overrides...]
```

- `<model>`: `pylbm` | `pyudales` | `pypalm` — the **assimilation** forward model
  (and the truth model too, unless `TRUTH_MODEL` overrides it).
- `<size>`:  `tiny` | `small` | `medium` | `large` | `xlarge` (a `conf/size/<size>.yaml`)

Examples:

```bash
job_scripts/snellius/submit.sh pylbm small
job_scripts/snellius/submit.sh pyudales medium esmda.num_assimilation_windows=3
job_scripts/snellius/submit.sh pypalm small ensemble.ensemble_size=20   # sizes the job for 20
TRUTH_MODEL=pyudales job_scripts/snellius/submit.sh pylbm small         # twin experiment
```

Any extra arguments are forwarded verbatim as Hydra overrides.

### Choosing the truth model (twin experiments)

By default the truth and assimilation forward models are the same. Set
`TRUTH_MODEL` to generate the ground truth with a *different* solver than the one
used for assimilation — e.g. a high-fidelity truth assimilated with a cheaper
model:

```bash
TRUTH_MODEL=pyudales job_scripts/snellius/submit.sh pylbm small
#   -> truth_model=pyudales, assim_model=pylbm
```

The wrapper composes each role's solver-specific Hydra flags automatically
(`cuda=false` for pylbm, `temp_dir`/`output_dir` for pyudales, `temp_dir` +
`domain.nz=16` for pypalm), so any truth/assim combination works. Mixed runs get
a `..._truth-<model>` suffix in the job name and log files.

## Tuning a run

Edit the per-size knobs in `conf/size/<size>.yaml`:

| Knob                              | Meaning                                  |
|-----------------------------------|------------------------------------------|
| `ensemble.ensemble_size`          | number of ensemble members               |
| `time.simulation_time`            | per-window simulation horizon            |
| `esmda.num_assimilation_windows`  | number of assimilation windows           |
| `*_params.time_coords.num`        | time-varying parameter knots per window  |

Correlation localization is **off by default** (`esmda/localization: none` in
`conf/run_esmda.yaml`). Enable it by appending the `++esmda.localization.*`
overrides — see the `USE_LOCALIZATION` block in `job_scripts/snellius/common.sh`
for the canonical set.

`submit.sh` reads `ensemble.ensemble_size` and requests **one core per member**,
rounded up to the partition's minimum billable share (16 on `rome`, 24 on
`genoa`) and capped at a single node. `ensemble.num_parallel_processes` is set to
the matching worker count. So bumping `ensemble_size` in the config is all it
takes for the next submission to grab more cores — no edits to the job scripts.

| `ensemble_size` | partition | cores requested | parallel workers |
|----------------:|-----------|----------------:|-----------------:|
| 4               | rome      | 16              | 4                |
| 8               | rome      | 16              | 8                |
| 32              | rome      | 32              | 32               |
| 64              | rome      | 64              | 64               |
| 96              | rome      | 96              | 96               |
| 129–192         | genoa     | up to 192       | up to 192        |

## Options (environment variables)

- `WALLTIME=HH:MM:SS` — override the per-size default wall time.
- `DRY_RUN=1` — print the computed sizing and the `sbatch` command without submitting.

```bash
DRY_RUN=1 job_scripts/snellius/submit.sh pylbm xlarge      # preview sizing
WALLTIME=02:00:00 job_scripts/snellius/submit.sh pylbm medium
```

## Layout

- `submit.sh` — the wrapper (compute sizing, submit).
- `templates/esmda.slurm` — one generic job body for all model combinations,
  driven by `PUA_SIZE` / `PUA_NUM_PARALLEL` / `PUA_TRUTH_MODEL` / `PUA_ASSIM_MODEL`
  (set by the wrapper). It runs `scripts/run_esmda.py` in simulate-truth-inline
  mode (`run.truth_dir=null`) with the `+size=<size>` overlay; for runs against a
  pre-simulated on-disk truth use the per-backend
  `<model>/rollout_esmda_from_truth.slurm` runners below instead. Not meant to be
  `sbatch`ed directly.

### Per-job working directories (concurrent-submission safety)

Each submission gets its own working directory under
`/scratch-shared/$USER/urbanair_runs/<timestamp>-<jobname>-<pid>/`, populated
with symlinks back to the read-only parts of the repo (`src`, `scripts`,
`conf`, `libs`, `.pixi`, etc.) and passed to sbatch as `--chdir`. Anything the
template writes at the working-dir root (Hydra artifacts, stray cwd-relative
writes) lives in that per-job dir, not in the shared repo root, so back-to-back
submissions can't clobber each other. The workdirs are tiny (just symlinks)
and `/scratch-shared` auto-purges, so no manual cleanup is needed.

Note: the per-job dir contains symlinks to the repo's `libs/`, so concurrent
**rebuilds** of uDALES / PALM still touch the shared `libs/<solver>/build/`
trees (pylbm is exempt: each job builds in a private copy of the LBM tree via
`PYLBM_LBM_PATH`). After the first successful build, subsequent runs reuse the
cache and don't write there, so this is usually fine — but avoid submitting
jobs in parallel while a code change is forcing a rebuild.
- `out_files/` — SLURM `.out`/`.err` logs, named `slurm-<model>_<size>-<jobid>`
  (gitignored).

Results land in `/projects/prjs2075/urbanair/esmda/<truth>_to_<assim>_<size>_<jobid>`;
intermediate solver I/O goes to `/scratch-shared/$USER/urbanair_temp/<jobid>`
(cleaned up on success, left behind on failure for post-mortem inspection).

## Rollout-ESMDA-from-truth sweeps (per backend)

This suite is the Snellius sibling of `job_scripts/local/` and is organised the
same way: two shared single-source-of-truth files plus thin per-backend runners
and sweep wrappers, so **all three backends run the exact same experiment** at
every configuration — same ground truth, domain, windows, time horizon, sensors
and dynamic-parameter settings. Only the assimilation solver differs.

The one difference from `local/`: there is **no parallel-worker cap**. Each run
sets `ensemble.num_parallel_processes == ensemble_size`, and the sweep launchers
request **one core per ensemble member** (`--cpus-per-task == ENSEMBLE_SIZE`),
so the SLURM allocation tracks the ensemble size directly.

```
snellius/
├── common.sh          # shared defaults + COMMON_RUN_FLAGS (sourced by every runner)
├── sweep_base.sh      # shared sweep engine + canonical value lists; submits one sbatch per point
├── pyudales/  pylbm/  pypalm/
│   ├── rollout_esmda_from_truth.slurm        # slim runner: backend specifics only
│   ├── sweep_domain_rollout_esmda_from_truth.sh
│   ├── sweep_ensemble_rollout_esmda_from_truth.sh
│   ├── sweep_esmda_steps_rollout_esmda_from_truth.sh
│   └── sweep_interval_rollout_esmda_from_truth.sh
```

- **`common.sh`** — every default and every `run_esmda.py` Hydra override that is
  identical across backends, in one place: paths (`/projects/prjs2075/urbanair`
  results, `/scratch-shared` temp), domain bounds + sensors, windows, time
  horizon, dynamic-parameter settings, localization, ground-truth
  resolution/validation, and the `COMMON_RUN_FLAGS` array. Every value is
  env-overridable; edit it once to retune the whole suite.
- **`sweep_base.sh`** — the canonical swept value lists (resolutions, ensemble
  sizes, ESMDA steps, observation intervals) with a per-row `--time`, plus the sizing logic
  (`--cpus-per-task = ensemble`, partition auto-selected: `rome` ≤128 cores,
  `genoa` up to 192). Submits one job per swept value.
- **`<backend>/rollout_esmda_from_truth.slurm`** — sources `common.sh` and adds
  only what differs: `ASSIM_MODEL` (= folder name), `num_parallel = ensemble`,
  `hydra.run.dir`, and the backend solver flags (`cuda=false` +
  `paths.experiment_dir` + private LBM copy for pylbm; `temp_dir`/`output_dir`
  for pyudales; `temp_dir` + PALM env + `nz≥16` floor for pypalm). Directly
  `sbatch`-able too.
- the four sweep wrappers (domain / ensemble / esmda_steps / interval) are thin
  and identical across folders — they delegate to `../sweep_base.sh` with the
  sibling `.slurm`. The `interval` sweep varies `obs.interval_seconds` (the
  observation temporal-aggregation bin width) at a fixed grid + ensemble + steps.

Run a sweep with one backend (submits one job per point), or the same sweep
across all three for directly comparable runs:

```bash
bash job_scripts/snellius/pyudales/sweep_domain_rollout_esmda_from_truth.sh
bash job_scripts/snellius/pylbm/sweep_ensemble_rollout_esmda_from_truth.sh esmda.seed=1
for m in pyudales pylbm pypalm; do
  bash job_scripts/snellius/$m/sweep_domain_rollout_esmda_from_truth.sh
done
```

Each job writes to `/projects/prjs2075/urbanair/assim_from_ground_truth/<RUN_TAG>`
where `RUN_TAG` embeds the assim model, grid, ensemble size, step count and
observation interval, so no
two configurations (or backends) collide. Correlation localization is **off** by
default; set `USE_LOCALIZATION=true` (env var, propagated through the sweep) to
enable it. Because pypalm requires `nz≥16`, the domain sweep's coarsest row
(`25 20 8`) is automatically raised to `25 20 16` for pypalm only, landing in its
own `nz16` output dir.
