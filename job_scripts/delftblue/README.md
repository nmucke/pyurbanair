# DelftBlue job scripts

Submit ESMDA runs on DelftBlue (CPU-only, `compute` partition) with the
`submit.sh` wrapper. It sizes the SLURM allocation from the experiment config,
so you tune the run in one place — `conf/size/<size>.yaml` — and the requested
cores follow automatically.

## Usage

```bash
job_scripts/delftblue/submit.sh <model> <size> [extra hydra overrides...]
```

- `<model>`: `pylbm` | `pyudales` | `pypalm` — the **assimilation** forward model
  (and the truth model too, unless `TRUTH_MODEL` overrides it).
- `<size>`:  `tiny` | `small` | `medium` | `large` | `xlarge` (a `conf/size/<size>.yaml`)

Examples:

```bash
job_scripts/delftblue/submit.sh pylbm small
job_scripts/delftblue/submit.sh pyudales medium esmda.num_assimilation_windows=3
job_scripts/delftblue/submit.sh pypalm small ensemble.ensemble_size=20   # sizes the job for 20
TRUTH_MODEL=pyudales job_scripts/delftblue/submit.sh pylbm small         # twin experiment
```

Any extra arguments are forwarded verbatim as Hydra overrides.

### Choosing the truth model (twin experiments)

By default the truth and assimilation forward models are the same. Set
`TRUTH_MODEL` to generate the ground truth with a *different* solver than the
one used for assimilation — e.g. a high-fidelity truth assimilated with a
cheaper model:

```bash
TRUTH_MODEL=pyudales job_scripts/delftblue/submit.sh pylbm small
#   -> truth_model=pyudales, assim_model=pylbm
```

The wrapper composes each role's solver-specific Hydra flags automatically
(`cuda=false` for pylbm, `temp_dir`/`output_dir` for pyudales, `temp_dir` +
`domain.nz=16` for pypalm), so any truth/assim combination works. Mixed runs
get a `..._truth-<model>` suffix in the job name and log files.

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
capped at a single DelftBlue compute node (64 cores). pypalm tolerates
oversubscription (it disables CPU pinning and lets OpenMPI yield), so for
pypalm the wrapper lets workers go up to 96 past the core cap — matching the
historical `xlarge` pattern (96 workers on 64 cores). For pylbm/pyudales it
stays one worker per core.

| `ensemble_size` | model        | cores | parallel workers |
|----------------:|--------------|------:|-----------------:|
| 4               | any          | 4     | 4                |
| 8               | any          | 8     | 8                |
| 32              | any          | 32    | 32               |
| 64              | any          | 64    | 64               |
| 96              | pylbm/pyudales | 64  | 64               |
| 96              | pypalm       | 64    | 96 (oversub)     |

Memory: `--mem-per-cpu` is set to **2G** when pypalm is involved (palm + combine
subprocesses are heavier on RSS) and **3G** otherwise.

## Options (environment variables)

- `WALLTIME=HH:MM:SS` — override the per-size default wall time.
- `DRY_RUN=1` — print the computed sizing and the `sbatch` command without submitting.
- `PYPALM_USE_DIRECT_RUN=` — only applies when pypalm is one of the roles. The
  template defaults this to `1`, which bypasses `palmrun`/`palmbuild` and runs
  the prebuilt PALM binary directly (saves ~130 s of per-invocation overhead;
  see `docs/palm_overhead_plan.md`). Set the variable to **empty or `0`** when
  submitting to revert to the historical palmrun path, e.g.
  `PYPALM_USE_DIRECT_RUN= job_scripts/delftblue/submit.sh pypalm small`.

```bash
DRY_RUN=1 job_scripts/delftblue/submit.sh pylbm xlarge      # preview sizing
WALLTIME=02:00:00 job_scripts/delftblue/submit.sh pylbm medium
```

## Layout

- `submit.sh` — the wrapper (compute sizing, submit).
- `templates/esmda.slurm` — one generic job body for all model combinations,
  driven by `PUA_SIZE` / `PUA_NUM_PARALLEL` / `PUA_TRUTH_MODEL` / `PUA_ASSIM_MODEL`
  (set by the wrapper). Handles the DelftBlue-specific MPI workaround
  (`OMPI_MCA_pml=ob1`/TCP) and, when pypalm is in either role, applies the
  PALM-specific environment (`PYPALM_FAST_IO_CATALOG` on node-local `/tmp`,
  OMPI oversubscribe/yield, CMakeCache cleanup, the `bash -c` wrapper that
  scrubs nvhpc's `CC`/`F90`/... so palmbuild picks conda's gfortran). Not meant
  to be `sbatch`ed directly.
- `out_files/` — SLURM `.out`/`.err` logs, named
  `slurm-esmda_<model>_<size>-<jobid>` (gitignored).
- `pypalm/m0_*.{py,slurm}`, `pypalm/m1_*.{py,slurm}`, `pypalm/m2_*.slurm` —
  investigation/benchmark scripts for the PALM per-invocation-overhead work
  (see `docs/palm_overhead_plan.md`). Not part of the standard ESMDA submission
  pattern; submit directly with `sbatch`. The standard `submit.sh` path now
  defaults to the direct-run; the palmrun fallback is reached by clearing
  `PYPALM_USE_DIRECT_RUN=` (see Options above).

Results land in `/projects/urbanair`; intermediate I/O uses
`/scratch/$USER/urbanair_temp/<jobid>` (beegfs scratch). pypalm additionally
routes its per-run working dir to node-local `/tmp` for the many-small-file
build-tree copy.
