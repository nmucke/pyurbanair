# Job scripts (SLURM)

SLURM wrappers for running the scripts in `scripts/` and the pipelines in
`workflows/` on the clusters. CPU only for now.

**Status:** not yet tested on SLURM. Checked only with `bash -n`,
`tests/scripts/test_job_scripts.py` and a local dry run.

## Layout

```text
job_scripts/
├── snellius/                         # SURF Snellius: partition rome, account tdsr72361
│   ├── env.sh                        # sourced by every job (see below)
│   ├── forward_workflow.slurm        # workflows/forward_workflow.sh
│   ├── assimilation_workflow.slurm   # workflows/assimilation_workflow.sh <smoother|filtering|hybrid>
│   ├── surrogate_generate_data.slurm # scripts/surrogate/generate_data.py
│   ├── surrogate_train.slurm         # scripts/surrogate/train.py
│   ├── surrogate_evaluate_{stepper,autoencoder,latent_generator}.slurm
│   └── out_files/                    # SLURM logs, %x-%j.out (gitignored)
└── delftblue/                        # TU Delft DelftBlue: compute-p1/p2, account innovation; same files
```

Each job is an `#SBATCH` header, `source job_scripts/<machine>/env.sh` and one
`run ... "$@"` line. `env.sh` does `module purge`, keeps BLAS single-threaded,
exports `PYURBANAIR_MACHINE` and `PYURBANAIR_RESULTS_ROOT` (read by
`paths.machine` / `paths.results_root` in `configs/common.yaml`), sets up and
cleans the per-job solver scratch, sets the MPI options for PALM's nested
`mpirun`, and defines `run` (a command in the machine's pixi env).

| Machine | `paths.results_root` | Scratch (`paths.scratch.<machine>`) |
|---|---|---|
| snellius | `/projects/prjs2075/urbanair` | `/scratch-shared/$USER/urbanair_temp/$SLURM_JOB_ID` |
| delftblue | `/projects/urbanair` | `/scratch/$USER/urbanair_temp/$SLURM_JOB_ID` |

Locally there is no job script: run the script or `workflows/*.sh` directly.
`scripts/tools/*` take seconds; run them on a login node.

## Submitting

Submit from the repo root. Every argument after the job script goes straight
to the script or workflow as a Hydra override (a CLI `paths.results_root=...`
beats the env var):

```bash
sbatch job_scripts/snellius/forward_workflow.slurm model=pylbm params=dynamic_truth
sbatch job_scripts/snellius/assimilation_workflow.slurm smoother model@assim_model=pylbm ensemble.ensemble_size=64
sbatch job_scripts/snellius/surrogate_train.slurm --config-name surrogate/train_autoencoder
```

`surrogate_train.slurm` needs `--config-name surrogate/<config>`. A ground
truth is a `forward_workflow.slurm params=dynamic_truth ...` run; load it in an
assimilation with `assimilation.truth_dir=<dir>`.

**Resizing.** Put sbatch flags before the job script:
`sbatch --cpus-per-task=64 --time=48:00:00 job_scripts/snellius/...`. Keep
`ensemble.num_parallel_processes` × `ensemble.num_cpus_per_process` ≤
`--cpus-per-task` (default 8 × 1). The hardware is DRAM-bandwidth-bound past
~4–8 workers: benchmark before raising the worker count.

**Sweeps and shards** are shell loops of `sbatch` calls. Give each sweep point
its own `paths.results_root`, or the outputs collide:

```bash
for n in 32 64 96; do
  sbatch job_scripts/snellius/assimilation_workflow.slurm smoother ensemble.ensemble_size=$n \
    paths.results_root=/projects/prjs2075/urbanair/ens$n
done
for i in 0 1 2 3; do
  sbatch job_scripts/snellius/surrogate_generate_data.slurm data.sharding.num_shards=4 data.sharding.shard_index=$i
done
```

## Example: ESMDA from a loaded ground truth

The old rollout-from-truth experiment (pyudales truth, pylbm assimilation,
time-varying inflow). The case `xie_and_castro` already holds its domain
bounds and sensors:

```bash
sbatch job_scripts/snellius/assimilation_workflow.slurm smoother \
  model@truth_model=pyudales model@assim_model=pylbm \
  params@truth_params=dynamic_truth params@prior_params=dynamic 'smoothing.smoother=${smoother.dynamic}' \
  assimilation.truth_dir=/projects/prjs2075/urbanair/ground_truth_pyudales_wide/pyudales_time_varying \
  assimilation.num_windows=6 assimilation.ensemble_save_on_disk=true \
  domain.nx=50 time.output_frequency=2.0 time.spinup_time=50.0 \
  ensemble.ensemble_size=96 smoothing.num_steps=3 'smoothing.localization=${localization.none}'
```

## Machine notes

- **pixi** must be 0.72.1 or newer (`requires-pixi` in `pyproject.toml`). An
  older one fails on every `pixi run`, often with a manifest parse error
  (`expected a string, found table`) rather than a version message. Run
  `pixi self-update` once per cluster account.
- **Scratch** is per job, removed on success and kept on failure for
  debugging.
- **Snellius:** scratch must be on `/scratch-shared`, not `$TMPDIR`
  (pyudales' `write_inputs.sh` fails under `/scratch-local`). OpenMPI 5: PALM's
  nested `mpirun` needs `--map-by :OVERSUBSCRIBE` (set via
  `PYPALM_MPIRUN_EXTRA_ARGS`). Never set `OMPI_MCA_osc=pt2pt` there: the
  component is gone and `MPI_Init` fails.
- **DelftBlue:** OpenMPI 4 takes `rmaps_base_oversubscribe` instead (and
  does need `osc=pt2pt`, set in its activation script). Each job
  builds LBM in a private copy (`PYLBM_LBM_PATH`) so concurrent jobs don't
  clobber the source tree. When the arguments mention `pypalm`, `env.sh`
  drops PALM's stale CMake caches and nvhpc's compiler variables, so PALM is
  built with conda's gfortran.
- The MPI `pml`/`btl` settings live in `activation_scripts/*_activation.sh`.

The previous job scripts (sweep launchers, `submit.sh`, figure jobs, PALM
debugging) are in `archive/job_scripts/`.
