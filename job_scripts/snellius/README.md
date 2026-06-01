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
| `time.simulation_time`            | forward-model simulation horizon         |
| `esmda.num_assimilation_windows`  | number of assimilation windows           |

Correlation localization is **on by default** (`esmda.localization` in the
config). The template pins `esmda.localization.truncation_correlation=0.3` so the
update is well-posed at any ensemble size; pass `esmda.localization=null` as an
extra override for the global (unlocalized) update.

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
  (set by the wrapper). Not meant to be `sbatch`ed directly.

### Per-job working directories (concurrent-submission safety)

Each submission gets its own working directory under
`/scratch-shared/$USER/urbanair_runs/<timestamp>-<jobname>-<pid>/`, populated
with symlinks back to the read-only parts of the repo (`src`, `scripts`,
`conf`, `libs`, `.pixi`, etc.) and passed to sbatch as `--chdir`. Anything the
template writes at the working-dir root — most importantly the `.temp` symlink
pylbm uses, which points at a node-local `/scratch-local/<jobid>` path — lives
in that per-job dir, not in the shared repo root. Two jobs submitted back-to-back
no longer clobber each other's `.temp`. The workdirs are tiny (just symlinks)
and `/scratch-shared` auto-purges, so no manual cleanup is needed.

Note: the per-job dir contains symlinks to the repo's `libs/`, so concurrent
**rebuilds** of pylbm / uDALES / PALM still touch the shared `libs/<solver>/build/`
trees. After the first successful build, subsequent runs reuse the cache and
don't write there, so this is usually fine — but avoid submitting jobs in
parallel while a code change is forcing a rebuild.
- `out_files/` — SLURM `.out`/`.err` logs, named `slurm-<model>_<size>-<jobid>`
  (gitignored).

Results land in `/projects/prjs2075/urbanair`; intermediate I/O uses node-local
`$TMPDIR` (auto-purged at job end). To keep a failing run's solver logs for
debugging, point temp at persistent scratch, e.g.
`... +truth_model.forward_model.temp_dir=/scratch-shared/$USER/debug`.
