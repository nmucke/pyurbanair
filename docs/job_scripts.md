# Job scripts (SLURM)

SLURM wrappers for running the scripts in `scripts/` and the pipelines in
`workflows/` on the clusters. CPU by default; a GPU run passes its resources
as sbatch flags (see the Tadpole example below).

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
│   ├── surrogate_prechunk_data.slurm # scripts/surrogate/train.py prechunk.prepare_only=true
│   ├── surrogate_prepare_latents.slurm # train.py latent_cache.prepare_only=true (GPU)
│   ├── surrogate_evaluate_{stepper,autoencoder,latent_generator}.slurm
│   ├── surrogate_baselines_compare.slurm # scripts/surrogate/baselines/compare.py
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

## Example: Tadpole AE (B) on the realistic corpus (GPU)

The size-B field autoencoder on the realistic uDALES corpus, trained on a
re-chunked copy of the data on scratch (`prechunk`, see
[neural_surrogates.md](neural_surrogates.md) §29). First an optional CPU job
that only makes the copy, then the GPU training once it succeeded:

```bash
args=(--config-name surrogate/train_autoencoder name=tadpole_ae_b_realistic
  paths.weights_dir=/projects/urbanair/model_weights
  paths.data_dir=/projects/urbanair/training_data/pyudales_realistic
  architecture.size=B batch_sampler.batch_size=48 batch_sampler.cell_budget=null
  batch_sampler.drop_last=false dataloader.num_workers=12 trainer.checkpoint_every=1
  prechunk.output_root=/scratch/$USER/training_data/pyudales_realistic_rechunked)
prep=$(sbatch --parsable job_scripts/delftblue/surrogate_prechunk_data.slurm "${args[@]}")
sbatch --dependency=afterok:$prep --partition=gpu-a100 --account=research-ceg-gse \
  --gpus-per-task=1 --cpus-per-task=16 --mem-per-cpu=4G --time=48:00:00 \
  job_scripts/delftblue/surrogate_train.slurm "${args[@]}"
```

For another dataset, `surrogate_prechunk_data.slurm paths.data_dir=<dataset>
prechunk.output_root=<copy>` makes its copy; train with the same two
overrides. A job cut off at its time limit resumes where it stopped when
resubmitted. The copy takes about 1.5 min per 700 MB trajectory (about 13 h
for the 520-file corpus); the DFT's whole-frame copy about half that, using up
to about 1 GB of memory. The DFT has its own whole-frame copy: start the
arguments with `--config-name surrogate/train_dft` (the last one wins, and
`key=value` overrides must all come after it) and give it its own
`prechunk.output_root`. Without the CPU job the GPU job makes
the copy itself; with a complete copy it only validates it in seconds, so once
the copy exists submit the GPU command alone. **Resuming:** submit the same GPU command again. It
continues from `checkpoint.pt` (`trainer.resume: true`), so `num_epochs` is
the total, not the epochs to add; `checkpoint_every=1` loses at most one epoch
to the time limit, and `encoder.pt` / `decoder.pt` / `geometry_branch.pt` are
cut from the best `weights.pt` at the start of each job and on every new best. Each submission rewrites
`config.yaml` from the current config (in this layout, also for a run started
with the earlier one), so keep the overrides. `cell_budget`
counts full trajectory grids, not the 64-cell crops, hence `null` here and the
batch size tuned directly. Evaluate on a GPU slice:

```bash
sbatch --partition=gpu-a100-small --account=research-ceg-gse --gpus-per-task=1 \
  --cpus-per-task=2 --mem-per-cpu=4G --time=04:00:00 \
  job_scripts/delftblue/surrogate_evaluate_autoencoder.slurm \
  autoencoder.model_dir=/projects/urbanair/model_weights/tadpole_ae_b_realistic \
  autoencoder.batch_size=1 autoencoder.max_internal_batchsize=2
```

The evaluation, the latent generator (`autoencoder_dir=<model dir>`) and the
DFT stepper (`pretrained_dir=<model dir>`) read only `architecture` and
`dataset` from the autoencoder's `config.yaml`, so artifacts trained with the
earlier layout load as they are; a DFT on this one also needs
`architecture.size=B` and the autoencoder's SDF and `geometry_branch` settings.

## Example: latent generator on precomputed latents (GPU)

The flow-matching generator on the realistic corpus, trained on a cache of the
frozen autoencoder's latents (`latent_cache`, see
[neural_surrogates.md](neural_surrogates.md) §38). First one GPU job encodes
every train/val frame, then the training reads the latents instead of the
states and skips the encoder:

```bash
args=(--config-name surrogate/train_latent_generator name=latent_generator_b_realistic
  paths.weights_dir=/projects/urbanair/model_weights
  paths.data_dir=/projects/urbanair/training_data/pyudales_realistic
  autoencoder_dir=/projects/urbanair/model_weights/tadpole_ae_b_realistic
  latent_cache.output_root=/scratch/$USER/training_data/pyudales_realistic_latents_ae_b
  batch_sampler.cell_budget=null batch_sampler.batch_size=4)
prep=$(sbatch --parsable job_scripts/delftblue/surrogate_prepare_latents.slurm "${args[@]}")
sbatch --dependency=afterok:$prep --partition=gpu-a100 --account=research-ceg-gse \
  --gpus-per-task=1 --cpus-per-task=16 --mem-per-cpu=4G --time=48:00:00 \
  job_scripts/delftblue/surrogate_train.slurm "${args[@]}"
```

Encode only a final autoencoder: the cache is keyed on its `weights.pt` and
settings, so a retrained one is refused (use a new `output_root`). The cache is about 112 GB
(float32) for the 500 train trajectories; a resubmitted job resumes it. The
latent generator's batch is bounded by `max_latent_tokens` (`B` x latent cells
must stay at or below 4096); the default `cell_budget` counts full grids and
gives one frame per batch here, hence `null` and a fixed batch size.

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
- **DelftBlue OpenMPI:** the `delftblue` env pins OpenMPI below 5: 5.0.x
  segfaults uDALES in `MPI_Finalize` (exit 139 after a clean run).
- **DelftBlue GPUs:** `gpu-a100` gives a full 80 GB A100 for up to 48 h;
  `gpu-a100-small` a 10 GB MIG slice (up to 4 CPUs) for up to 4 h.
- **DelftBlue accounts:** `innovation` allows a user 1 running and 10 queued
  jobs of at most 24 h; submit long series under `research-ceg-gse`.
- **DelftBlue throughput:** uDALES data generation costs about 1.6–2.1 µs per
  grid cell per simulated second.
- **DelftBlue BeeGFS** can intermittently report hard-linked pixi env files as
  missing when many jobs import at once; resubmit the failed job.

The previous job scripts (sweep launchers, `submit.sh`, figure jobs, PALM
debugging) are in `archive/job_scripts/`.
