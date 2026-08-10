# pyurbanair

[![CI](https://github.com/nmucke/pyurbanair/actions/workflows/ci.yml/badge.svg)](https://github.com/nmucke/pyurbanair/actions/workflows/ci.yml)

A Python framework for urban air flow simulation and ensemble-based data assimilation. Part of the UrbanAIR project.

> **Note:** This repository is under active development (v0.1.0). Things will change and some functionalities may not work as intended.

## Features

- **Three CFD backends:** pylbm (Lattice Boltzmann Method, wrapping Geir Evensen's LBM), pyudales (wrapping uDALES v2.2.0), and pypalm (wrapping the PALM model system)
- **Neural surrogate backend:** train a learned one-step network on CFD ensembles and run it as a drop-in fourth forward model — several architectures (`SimpleConv`, `UNetConvNeXt`, the transformer-based `UPT`, `P3D`, and a domain-decomposed model that runs on any grid sharing its training cell spacing), full stack (data generation, training, autoregressive rollout) in [`neural-surrogates`](libs/neural-surrogates), documented in [`docs/neural_surrogates.md`](docs/neural_surrogates.md)
- **Signed-distance geometry features** — optional SDF / ∇SDF obstacle channels (`sdf` / `grad` / `both`) shared by the dataloader and the P3D / Tadpole stems, giving the network a bounded, grid-spacing-consistent view of the building field
- **Foundation-model pre-training + fine-tuning** — pre-train a vendored Tadpole (variational) autoencoder on urban-flow snapshots, turn it into an ESMDA time-stepper via a zero-initialised "Dynamic Fine-Tuning" (DFT) head, and adapt any architecture with parameter-efficient **LoRA (PEFT)** fine-tuning that merges back into a plain, ESMDA-loadable `weights.pt`
- **Model comparison** — score several trained surrogates on the same rollout benchmark from one config
- **Ensemble-based data assimilation** using ESMDA (Ensemble Smoother with Multiple Data Assimilation), implemented in JAX
- **Sequential ensemble filtering (EnKF)** — a cycled Ensemble Kalman Filter alongside the smoothers, with pluggable analysis / localization / inflation (multiplicative, RTPS, RTPP) / parameter-evolution groups, for state, parameter, or joint estimation
- **Parameter estimation** and **joint state-parameter estimation**
- **Localization** for ESMDA (adaptive correlation-based observation tapering, with optional "grid block" joint analysis; opt-in)
- **Multi-step rollout simulations** with state carry-over between time windows
- **Cross-model assimilation** (e.g., use LBM as truth model with uDALES — or a neural surrogate — for assimilation)
- **Time-varying parameters** with per-window mean/std profiles for the inflow priors
- **Observation operators** for mapping simulation states to observation space, with held-out validation sensors for out-of-sample scoring
- **Reusable ground-truth artifacts** — simulate a truth once, trim its spin-up, downcast to 32-bit, and feed it to many assimilation runs
- **Benchmark geometry generation** for the Xie and Castro 2008 case

## Installation

All dependencies and environments are handled via [Pixi](https://pixi.sh). Install Pixi on Linux or MacOS by running:

```
curl -fsSL https://pixi.sh/install.sh | sh
```

Four environments are available:

| Environment | Purpose |
|-------------|---------|
| `dev` | Full development environment with all backends, data assimilation, benchmarks, and dev tools |
| `delftblue` | HPC environment for the DelftBlue supercomputer |
| `snellius` | CPU-only HPC environment for the Snellius supercomputer |
| `cuda` | GPU-accelerated environment with CUDA support |

Install and activate the dev environment:

```
pixi run setup-dev
pixi shell --environment=dev
```

> **Why `setup-dev` and not `pixi install -e dev`?** The dev env combines the
> `palm` feature (which depends on `coreutils`) with the `udales` feature
> (which transitively pulls in `tempest-remap` via `nco`). `coreutils` ships
> `bin/test` as a file while `tempest-remap` ships scripts under `bin/test/`
> as a directory, so the two clobber each other and the first `pixi install`
> aborts. The `setup-dev` task runs the install, deletes the conflicting
> `bin/test` file if needed, and re-runs the install so `tempest-remap` can
> claim the path. Run it once after cloning; subsequent `pixi install` /
> `pixi shell` calls work normally.

### LBM specifics

For running the LBM code on MacOS, you have to run the following after initializing the environment:

```
ulimit -s unlimited
```

## Usage

### Configuration

All simulation and assimilation settings live in `conf/`, composed by
[Hydra](https://hydra.cc/). Any field can be overridden from the command line
(`domain.nx=80`, `esmda.num_steps=4`). There are three run entry points,
one per script, and each is **self-contained** —
[`run_forward_model.yaml`](conf/run_forward_model.yaml) for
`scripts/run_forward_model.py`, [`run_esmda.yaml`](conf/run_esmda.yaml) for
`scripts/esmda/run_esmda.py`, and
[`run_filtering.yaml`](conf/run_filtering.yaml) for
`scripts/filtering/run_filtering.py` (sequential EnKF). See
[`conf/README.md`](conf/README.md) for the full overview.

**Inlined base** (each entry point carries its own copy, rather than pulling
shared files):

- **output `paths`** — output roots (everything mutable lands under `.temp/`).
- **`time.seconds_per_knot`** — the spacing (in seconds) between time-varying
  parameter knots; the parameter takes a new value every `seconds_per_knot` s,
  with the last value linearly extrapolated onto the window end when the horizon
  isn't an exact multiple. The per-window horizon — simulation duration, output
  frequency, spinup time — lives in the `case`.
- **`ensemble`** — ensemble size, parallel processes, CPUs/process, failure
  policy.
- **`esmda`** (run_esmda only) — assimilation steps/windows, observation error
  std, seed, plus `localization` / `state_reduction` (default `none`; selected
  via the `esmda/*` groups, see below).
- the **`run:`** namespace and Hydra settings.

**Groups** (one option per structurally-distinct variant):

- **`case/`** — the experiment bundle, and the one file you edit to set up an
  experiment. A case is self-contained: `domain` bounds + grid resolution, `obs`
  sensor layout (incl. held-out `validation_{x,y,z}_points`), `geometry` STL
  paths, and the per-window `time` horizon. `case=xie_and_castro` (default) or
  `case=barcelona`. Override individual fields as usual (`domain.nx=80`).
- **`model/`** — forward + ensemble backend, mounted under a package
  (`model@truth_model=pylbm model@assim_model=pyudales`).
- **`params/`** — parameter samplers: `static` / `dynamic` (assimilation prior)
  and `static_truth` / `dynamic_truth` (the truth generator, kept separate to
  avoid the inverse crime). `dynamic` is `AR2RelaxationModel`, a critically
  damped AR(2) prior relaxing toward an external prior; each external `mean`/
  `std` may be a scalar **or a list of control points** interpolated over the
  window, letting `x_ext(t)` / `Σ_ext(t)` vary in time. Mounted once
  (`params=...`) for forward runs, twice (`params@truth_params=...`
  `params@prior_params=...`) for assimilation.
- **`esmda/smoother/`** — the ESMDA variant: `static` (parameter-only),
  `state` (state-only with fixed parameters), `dynamic` (time-varying
  parameters), `state_and_parameter` / `state_and_dynamic` (joint). Selected
  with `esmda/smoother=...`.
- **`filtering/`** (run_filtering only) — the sequential-EnKF machinery, one
  option per group: `filtering/analysis` (`stochastic`), `filtering/localization`
  (`none` / `correlation` / `distance`), `filtering/inflation` (`none` /
  `multiplicative` / `rtps` / `rtpp`), and `filtering/evolution` (`none` /
  `random_walk`, the parameter forecast model). The estimation target is
  `filtering.mode=state | parameter | joint`.
- **compute budget** — ensemble size / parallelism / ESMDA steps / windows /
  param knots are baked into the two entry points at medium-sized defaults;
  change them with plain CLI overrides (`ensemble.ensemble_size=8`,
  `esmda.num_assimilation_windows=10`). See [`conf/README.md`](conf/README.md).
- **`neural_surrogate/`** — everything for the learned surrogate, in one folder:
  `training.yaml`, `testing.yaml`, `training_data.yaml` (a single data-generation
  config), `architectures/{unet_convnext,upt,p3d,domain_decomposed}/<size>.yaml`,
  a `mode/` group (`standard` / `domain_decomposition`), and the foundation-model
  configs — `pretrain_autoencoder.yaml`, `finetuning.yaml` with a `finetune_mode/`
  group (`lora_nextstep` / `dft`), and `comparison.yaml`. See
  [`docs/neural_surrogates.md`](docs/neural_surrogates.md).

### Forward simulations

A single `run_forward_model.py` covers single/ensemble runs, single-window or
multi-window rollouts, and static or time-varying inflow. The mode is selected
by `run.ensemble`, `run.rollout_steps`, and `params=static|dynamic`:

```bash
# Single forward simulation
python scripts/run_forward_model.py model=pylbm
python scripts/run_forward_model.py model=pyudales

# Ensemble forward simulation
python scripts/run_forward_model.py model=pylbm run.ensemble=true

# Multi-window rollout (run.rollout_steps additional windows after the first)
python scripts/run_forward_model.py model=pylbm run.rollout_steps=3

# Ensemble rollout (combine both flags)
python scripts/run_forward_model.py model=pylbm run.ensemble=true run.rollout_steps=3

# Time-varying inflow (params=dynamic) — writes a state.nc/params.nc ground-truth
# artifact that run_esmda.py can consume via run.truth_dir
python scripts/run_forward_model.py model=pylbm params=dynamic run.rollout_steps=3
```

### Ground-truth artifacts

A truth simulation can be saved once and reused across many assimilation runs.
`run_forward_model.py params=dynamic` writes a `state.nc`/`params.nc` pair; three
helper scripts (plain CLIs, not Hydra) post-process it:

```bash
# Drop the spin-up transient and rebase the time axis to t=0
python scripts/adjust_simulations/trim_spinup.py \
    --state ground_truth/state.nc --params ground_truth/params.nc \
    --spinup-time 50 --output-dir ground_truth_spunup

# Downcast 64-bit NetCDF variables to 32-bit float (streamed; halves on-disk size)
python scripts/adjust_simulations/convert_ground_truth_to_32bit.py   # ground_truth/64_bit -> 32_bit

# Diagnostic figures: prescribed params, a field snapshot, and the inflow
# angle/speed recovered from the flow vs. the prescribed values
python scripts/figure_creation/visualize_ground_truth.py ground_truth_spunup
```

The resulting folder is what `run_esmda.py` loads via `run.truth_dir` (see below).
These (multi-GB) `ground_truth*` folders are gitignored.

### Data assimilation

All data assimilation runs through a **single** script, `run_esmda.py`. The
mode is the cross product of three declarative axes plus a truth source:

- `esmda/smoother=static | state | state_and_parameter | dynamic |
  state_and_dynamic` — parameter-only, state-only, or joint state+parameter,
  with static or time-varying parameters as appropriate.
- `params@prior_params=static | dynamic` (paired with the matching
  `params@truth_params=static_truth | dynamic_truth`) — static scalar
  parameters vs. a time-varying AR(2) prior.
- `esmda.num_assimilation_windows=1 | N` — a single window vs. an N-window
  rollout.
- `run.truth_dir=null` (simulate the truth inline) or `=<path>` to a saved
  `state.nc`/`params.nc` truth artifact. `run_esmda.yaml` defaults this to
  `ground_truth_spunup`. Use `run.truth_start_time=<seconds>` to begin the
  assimilation horizon partway into a disk truth (skips a spin-up and rebases
  that time to t=0). Disk truth is streamed, so multi-GB files never load fully.

Shared ESMDA settings live in the inlined `esmda:` block of `conf/run_esmda.yaml`.
The dynamic multi-window setup
(time-varying inflow over a rollout, with localization) is written up in
[`docs/temp/esmda_dynamic_multiwindow.md`](docs/temp/esmda_dynamic_multiwindow.md).

```bash
# Parameter estimation (parameter-only smoother, static params, single window)
python scripts/esmda/run_esmda.py esmda/smoother=static \
  params@prior_params=static params@truth_params=static_truth \
  model@truth_model=pylbm model@assim_model=pylbm

# Cross-model assimilation (LBM truth, uDALES assimilation)
python scripts/esmda/run_esmda.py esmda/smoother=static \
  params@prior_params=static params@truth_params=static_truth \
  model@truth_model=pylbm model@assim_model=pyudales

# Joint state and parameter estimation
python scripts/esmda/run_esmda.py esmda/smoother=state_and_parameter \
  params@prior_params=static params@truth_params=static_truth

# State-only estimation (parameters are used by forecasts but held fixed)
python scripts/esmda/run_esmda.py esmda/smoother=state \
  params@prior_params=static params@truth_params=static_truth \
  esmda/localization=distance

# Rollout-based ESMDA with multiple assimilation windows
python scripts/esmda/run_esmda.py esmda/smoother=state_and_parameter \
  params@prior_params=static esmda.num_assimilation_windows=3

# Time-varying-parameter ESMDA over a 3-window rollout
python scripts/esmda/run_esmda.py esmda/smoother=dynamic \
  params@prior_params=dynamic params@truth_params=dynamic_truth \
  esmda.num_assimilation_windows=3

# Assimilate against a saved truth instead of simulating it inline
python scripts/esmda/run_esmda.py esmda/smoother=dynamic \
  run.truth_dir=ground_truth_spunup run.truth_start_time=50

# Adaptive correlation localization (Vossepoel et al. 2025) is OFF by default;
# enable it with the esmda/localization group, or set its fields:
python scripts/esmda/run_esmda.py esmda/smoother=static \
  esmda/localization=correlation \
  esmda.localization.truncation_correlation=0.35 esmda.localization.block_grouping=true

# Fast smoke run (small domain, few steps)
python scripts/esmda/run_esmda.py model@truth_model=pylbm model@assim_model=pylbm \
  domain.nx=20 domain.ny=20 domain.nz=4 time.simulation_time=5 \
  ensemble.ensemble_size=4 esmda.num_steps=1 esmda.num_assimilation_windows=1
```

> **Note:** `run_esmda.yaml` defaults to the time-varying rollout
> (`esmda/smoother=dynamic`, `params=dynamic`, `pyudales`↔`pyudales`); set the
> axes above explicitly for the other modes. The smoother group filenames are
> `static`/`state`/`dynamic`/`state_and_parameter`/`state_and_dynamic`.

Each run writes per-window prior/posterior parameters and state, a
`run_summary.yaml` with timing and accuracy metrics (parameter RMSE/CRPS, state
RMSE, assimilated- and validation-sensor RMSE/CRPS, sensor window statistics and
the mean-field hit rate) beside the reduced `eval_fields.nc` the field figures
read, and diagnostic figures (parameter time-evolution and marginals, parameter
error, sensor time series and quantile fans, station profiles, time-mean field
slices, a rank histogram, final state with observations, and an animation). The
figure stage skips whatever a given run dir cannot support rather than failing.
All forward models also generate a `.temp` folder where intermediate files are
stored.

### Sequential filtering (EnKF)

Where ESMDA re-assimilates a whole window at once, `run_filtering.py` runs a
**cycled Ensemble Kalman Filter**: each cycle forecasts one segment of
`time.simulation_time` seconds (set it to `obs.interval_seconds` to assimilate
one observation interval per analysis) and applies a single full-weight analysis.
The mode is `filtering.mode=state | parameter | joint`, and the analysis math /
localization / inflation / parameter-evolution are picked by the `filtering/*`
groups. The **prior must be static** (`params@prior_params=static`) — the filter
re-tracks a scalar each cycle, so time-varying (AR(2)) priors stay with
`run_esmda.py` — but the truth may be dynamic, in which case the filter tracks a
drifting truth. The truth source (`run.truth_dir` / `run.truth_start_time`)
mirrors `run_esmda.py`.

```bash
# Joint state+parameter filtering over 4 cycles
python scripts/filtering/run_filtering.py filtering.mode=joint filtering.num_cycles=4

# Parameter-only, driven by a random-walk parameter forecast (no inflation)
python scripts/filtering/run_filtering.py filtering.mode=parameter \
  filtering/evolution=random_walk filtering/inflation=none

# State estimation with correlation localization
python scripts/filtering/run_filtering.py filtering.mode=state \
  filtering/localization=correlation
```

> **Note:** the parameter-updating modes (`parameter`, `joint`) require spread
> maintenance — pair them with `filtering/evolution=random_walk` or an inflation
> option, or the filter refuses the silently-collapsing configuration at
> construction. `scripts/run_filtering_pipeline.sh` chains run → metrics →
> figures, mirroring `run_esmda_pipeline.sh`.

### Neural surrogates

A learned one-step network can be trained on a CFD ensemble and then used
as a drop-in fourth forward model alongside pylbm, pyudales, and pypalm.
Several architectures are available — `SimpleConv` (baseline), `UNetConvNeXt`,
the transformer-based `UPT` (Universal Physics Transformer), `P3D`, and a
domain-decomposed model that tiles a fixed patch size so one trained instance
runs on any global grid sharing its training cell spacing. The end-to-end stack
(dataset generation → training → autoregressive rollout → use as a
forward/assimilation model) is documented in
[`docs/neural_surrogates.md`](docs/neural_surrogates.md). The headline
commands:

```bash
# 1. Generate a training dataset by driving a CFD ensemble
pixi run -e dev python scripts/neural_surrogate/generate_training_data.py model=pylbm

# 2. Train a surrogate (pick an architecture preset; UNetConvNeXt or UPT)
pixi run -e dev python scripts/neural_surrogate/train_neural_surrogate.py \
    dataset.root_dir=training_data/pylbm_medium \
    'neural_surrogate/architectures/upt@architecture=small'

# 3. Autoregressive rollout on the test split (diagnostic plots + animation)
pixi run -e dev python scripts/neural_surrogate/test_neural_surrogate.py \
    model_dir=model_weights/upt_small sample_idx=0

# 4. Use the trained surrogate as an assimilation model
python scripts/esmda/run_esmda.py esmda/smoother=dynamic \
    params@prior_params=dynamic params@truth_params=dynamic_truth \
    esmda.num_assimilation_windows=3 \
    model@truth_model=pyudales model@assim_model=neural_surrogate \
    assim_model.forward_model.model_dir=model_weights/upt_small
```

`UPT` z-score-normalizes the state and inflow parameters and predicts the
per-step residual; the normalization statistics are computed automatically at the
start of training and baked into the checkpoint, so nothing extra is needed at
inference time.

#### Foundation-model pre-training & fine-tuning

Beyond training an architecture from scratch, the surrogate stack supports a
pre-train → fine-tune workflow (design records under
[`docs/neural_surrogate_plans/`](docs/neural_surrogate_plans/)):

- **Tadpole autoencoder pre-training** — `pretrain_autoencoder.py` trains a
  vendored Tadpole (variational) autoencoder on urban-flow snapshots as pure
  representation learning (no next-step objective); `test_autoencoder.py`
  inspects reconstructions. The AE is *not* an ESMDA model on its own.
- **AE → time-stepper (DFT)** — `finetune_neural_surrogate.py` with
  `finetune_mode=dft` turns a pre-trained AE (passed as `pretrained_model_dir`)
  into a `TadpoleTimeStepper` by training a zero-initialised "Dynamic
  Fine-Tuning" head (skip scales + latent subnetwork) around the frozen
  autoencoder, yielding a next-step forward model.
- **LoRA (PEFT) fine-tuning** — the default `finetune_mode=lora_nextstep` adapts
  an already-trained architecture (e.g. P3D) with low-rank adapters via
  HuggingFace PEFT, then merges them back into a plain `weights.pt` that is
  byte-indistinguishable from a fully-trained checkpoint, so
  `NeuralSurrogateForwardModel` and ESMDA need zero changes. Requires the optional
  extra (`pip install 'neural_surrogates[finetuning]'`; already present in the
  pixi `dev`/`cuda` envs). `lora.variant=balora` (Bayesian LoRA, plan 04) is
  reserved but not yet implemented.
- **Model comparison** — `compare_surrogate_models.py` (config
  `neural_surrogate=comparison`) scores several trained surrogates on one rollout
  benchmark.

```bash
# Pre-train the Tadpole autoencoder on an existing training dataset
pixi run -e dev python scripts/neural_surrogate/pretrain_autoencoder.py \
    dataset.root_dir=training_data/pylbm_barcelona model_name=tadpole_ae_s

# LoRA next-step fine-tune of an already-trained P3D onto new data
pixi run -e dev python scripts/neural_surrogate/finetune_neural_surrogate.py \
    pretrained_model_dir=model_weights/p3d_xie_and_castro \
    model_name=p3d_ft_barcelona \
    dataset.root_dir=training_data/pylbm_barcelona

# AE -> DFT time-stepper from a pre-trained Tadpole autoencoder
pixi run -e dev python scripts/neural_surrogate/finetune_neural_surrogate.py \
    finetune_mode=dft \
    pretrained_model_dir=model_weights/tadpole_ae_s \
    model_name=tadpole_dft_s_barcelona \
    dataset.root_dir=training_data/pylbm_barcelona

# Compare several trained surrogates on the same rollout benchmark
pixi run -e dev python scripts/neural_surrogate/compare_surrogate_models.py
```

Any architecture can be built with SDF geometry features
(`dataset.sdf_features=sdf|grad|both`, default `none`) to feed the network a
bounded signed-distance view of the obstacle field; the same channels are shared
by the dataloader and the model stem. See
[`docs/neural_surrogates.md`](docs/neural_surrogates.md) for the full stack.

### Running on Snellius (SLURM)

The Snellius `snellius` env ships with a one-command submit wrapper that picks
the partition, requests the right number of cores (from the size label), and
sets a sensible wall time. Use it instead of writing your own sbatch files. Full
details: [`job_scripts/snellius/README.md`](job_scripts/snellius/README.md).

```bash
# Pattern
job_scripts/snellius/submit.sh <model> <size> [extra hydra overrides...]
#   <model>   pylbm | pyudales | pypalm     (assimilation forward model)
#   <size>    tiny | small | medium | large | xlarge   (sizes the SLURM allocation)
```

Common launches:

| Goal                                          | Command                                                                       |
|-----------------------------------------------|-------------------------------------------------------------------------------|
| pylbm, small run                              | `job_scripts/snellius/submit.sh pylbm small`                                  |
| pyudales, medium run                          | `job_scripts/snellius/submit.sh pyudales medium`                              |
| pypalm, small run                             | `job_scripts/snellius/submit.sh pypalm small`                                 |
| Twin experiment (truth ≠ assim model)         | `TRUTH_MODEL=pyudales job_scripts/snellius/submit.sh pylbm small`             |
| Ad-hoc Hydra override (per submission)        | `job_scripts/snellius/submit.sh pylbm small esmda.num_assimilation_windows=3` |
| Custom wall time (overrides the scale default)| `WALLTIME=30:00:00 job_scripts/snellius/submit.sh pyudales medium`            |
| Preview only (don't submit)                   | `DRY_RUN=1 job_scripts/snellius/submit.sh pyudales medium`                    |

**Tuning a run.** Runs use the medium-sized defaults baked into
`conf/run_esmda.yaml` (physical setup lives in `conf/case/<case>.yaml`). The
`<size>` label maps to an ensemble size that the wrapper uses to size the SLURM
allocation (one core per member, rounded up to the partition's billing minimum —
16 on `rome`, 24 on `genoa`); pass hydra overrides to change anything else:

| Override                          | Meaning                            |
|-----------------------------------|------------------------------------|
| `ensemble.ensemble_size`          | number of ensemble members         |
| `time.simulation_time`            | per-window forward-model duration  |
| `esmda.num_assimilation_windows`  | number of assimilation windows     |

Results land in `/projects/prjs2075/urbanair/`; SLURM logs in
`job_scripts/snellius/out_files/slurm-<model>_<size>-<jobid>.{out,err}`
(gitignored). Mixed-model runs get a `..._truth-<model>` suffix.

## Repository Structure

The repository uses a monorepo approach. It contains a base project `pyurbanair` and a series of sub-libraries in the `libs/` folder. The general idea is that everything should be run from the `pyurbanair` project, which loads functionalities from the other libraries.

```
pyurbanair/
├── src/
│   └── pyurbanair/                        # Main package
│       ├── base_forward_model.py          # Abstract base class for forward models
│       ├── base_ensemble_forward_model.py # Ensemble execution orchestration
│       ├── base_rollout_forward_model.py  # Legacy multi-step rollout base (file-only, unused)
│       ├── quiet_jax.py                    # Import before jax to silence CPU-fallback noise
│       ├── animation.py                   # Animation utilities
│       ├── static_parameters/             # ParameterSampler + Normal/Uniform/Constant
│       ├── dynamic_parameters/            # AR2RelaxationModel time-varying prior
│       ├── training_data/                 # Sampler skeletons for surrogate data generation
│       ├── config/
│       │   └── hydra_helpers.py           # Helpers consumed by Hydra configs (instantiate targets)
│       └── utils/
│           ├── state_utils.py             # State manipulation utilities
│           ├── run_utils.py               # Runtime utilities
│           └── animation_utils.py         # Animation generation helpers
│
├── libs/                                  # Sub-libraries
│   ├── evaluation/                        # Metrics + figures for DA runs (leaf lib; no JAX,
│   │   ├── pyproject.toml                 #   no pyurbanair/backend imports)
│   │   └── src/evaluation/
│   │       ├── scores.py                  # Ensemble scores (CRPS, energy score) + metric bundles
│   │       ├── turbulence.py              # z-plane selection, streamed |U| state RMSE
│   │       ├── sensors.py                 # Reductions of pre-extracted sensor series
│   │       ├── style.py                   # Figure palette/rcParams/save + STL solid masks
│   │       └── figures.py                 # plot_* for parameters, sensors and state
│   │
│   ├── data-assimilation/                 # Data assimilation library (JAX)
│   │   ├── pyproject.toml
│   │   └── src/data_assimilation/
│   │       ├── observation_operator.py    # Maps states to observation space
│   │       ├── interpolation.py           # Grid interpolation utilities
│   │       ├── augmentation.py            # State/parameter augmentation for joint analysis
│   │       ├── inflation.py               # Covariance inflation (multiplicative, RTPS, RTPP)
│   │       ├── reduction.py               # State-space reduction
│   │       ├── io.py                      # NetCDF read/write helpers
│   │       ├── localization/              # BaseLocalization + CorrelationLocalization
│   │       ├── smoothing/
│   │       │   ├── base.py                # Base smoothing class
│   │       │   └── esmda.py               # ESMDA implementation
│   │       └── filtering/                 # Sequential EnKF: base, analysis, parameter_evolution
│   │
│   ├── pylbm/                             # Lattice Boltzmann Method wrapper
│   │   ├── pyproject.toml
│   │   └── src/pylbm/
│   │       ├── forward_model.py
│   │       ├── ensemble_forward_model.py
│   │       ├── stl_to_lbm.py             # STL geometry conversion
│   │       └── utils/
│   │
│   ├── pyudales/                          # uDALES wrapper
│   │   ├── pyproject.toml
│   │   └── src/pyudales/
│   │       ├── forward_model.py
│   │       ├── ensemble_forward_model.py
│   │       ├── python_udgeom/            # Python preprocessing (Matlab alternative)
│   │       └── utils/                    # namoptions, nudging, ncpu, dt-collapse watchdog (run_monitor.py)
│   │
│   ├── pypalm/                            # PALM model system wrapper (lazy import)
│   │   ├── pyproject.toml
│   │   └── src/pypalm/
│   │       ├── forward_model.py
│   │       ├── ensemble_forward_model.py
│   │       └── utils/
│   │
│   └── neural-surrogates/                 # Learned one-step CFD surrogate
│       ├── pyproject.toml
│       └── src/neural_surrogates/
│           ├── forward_model.py           # NeuralSurrogateForwardModel
│           ├── ensemble_forward_model.py
│           ├── datasets/                  # TransitionDataset, PatchTransitionDataset
│           ├── training/                  # BaseTraining, Trainer, PatchTrainer, AutoencoderTrainer
│           ├── finetuning/                # LoRA/PEFT injection, merge-to-state-dict, target presets
│           ├── decomposition.py / dd_loss.py  # Domain-decomposition operators + Eq-9 loss
│           ├── geometry.py                # STL → voxel geometry channel
│           ├── sdf.py                     # Signed-distance-field geometry features (SDF / ∇SDF)
│           ├── training_spinup.py         # Warm-start ESMDA from training-data trajectories
│           └── architectures/             # SimpleConv, UNetConvNeXt, UPT (_upt/), P3D, DomainDecomposed,
│                                          #   Tadpole AE + AE→DFT time-stepper (tadpole_ae/stepper, _tadpole/)
│
├── conf/                                  # Hydra config (see Configuration)
│   ├── run_forward_model.yaml             # Entry point — forward-model runs (self-contained)
│   ├── run_esmda.yaml                     # Entry point — all ESMDA runs (self-contained)
│   ├── run_filtering.yaml                 # Entry point — sequential EnKF runs (self-contained)
│   ├── README.md                          # Config overview — the axes + recipes
│   ├── case/                              # Experiment bundles: domain+grid+geometry+sensors+time (xie_and_castro, barcelona)
│   ├── model/                             # Backend wiring (pylbm, pyudales, pypalm, neural_surrogate)
│   ├── params/                            # Parameter samplers (static/dynamic + *_truth)
│   ├── esmda/                             # ESMDA smoother/localization/state_reduction groups
│   ├── filtering/                         # EnKF analysis/localization/inflation/evolution groups
│   └── neural_surrogate/                  # Surrogate: training/testing/training_data.yaml, architectures/, mode/,
│                                          #   pretrain_autoencoder.yaml, finetuning.yaml + finetune_mode/, comparison.yaml
│
├── scripts/                               # Main execution scripts (see docs/scripts_and_configs.md)
│   ├── run_forward_model.py               # Forward sim (run.ensemble / run.rollout_steps / params=static|dynamic)
│   ├── _common.py                         # Shared script glue (viz, derived-param plots, metrics)
│   ├── run_esmda_pipeline.sh              # ESMDA: run → metrics → figures
│   ├── run_filtering_pipeline.sh          # Filtering: run → metrics → figures
│   ├── esmda/                             # ESMDA pipeline: run_esmda.py, compute_esmda_metrics.py, make_esmda_figures.py, _esmda_common.py
│   ├── filtering/                         # Filtering (EnKF) pipeline: run_filtering.py, compute_filtering_metrics.py, make_filtering_figures.py, _filtering_common.py
│   ├── neural_surrogate/                  # Surrogate stack (generate/train/test, pretrain/finetune autoencoder, compare)
│   ├── adjust_simulations/                # Ground-truth utilities (trim_spinup, 32-bit, ...)
│   ├── figure_creation/                   # Paper/diagnostic figures (visualize_ground_truth, ...)
│   ├── tools/                             # Case setup CLIs (prepare_case_stl, preprocess_udales_geometry)
│   └── figspec/                           # Figure data-IO + masking helpers for figure_creation/
│
├── job_scripts/                           # HPC submission (see docs/job_scripts.md)
│   ├── snellius/                          # Snellius SLURM wrapper + sweeps
│   ├── delftblue/                         # DelftBlue SLURM wrapper + sweeps
│   └── local/                             # Local multi-process runners
│
├── examples/                              # Example experiments
│   ├── benchmark_geometry/                # Xie and Castro 2008 geometry tools
│   ├── lbm/experiments/                   # LBM experiment configs (STL files)
│   ├── udales/experiments/                # uDALES experiment configs
│   └── palm/                              # PALM experiment configs (_p3d)
│
├── docs/                                  # Documentation
│   ├── codebase_guide.md                  # Orientation sheet for AI coding assistants (entrypoint)
│   ├── pylbm.md / pyudales.md / pypalm.md # Per-backend deep dives
│   ├── data_assimilation.md               # ESMDA / observation operator / localization
│   ├── neural_surrogates.md               # Neural-surrogate stack
│   ├── scripts_and_configs.md             # conf/ + scripts/ reference
│   ├── job_scripts.md                     # HPC job-script reference
│   ├── neural_surrogate_plans/            # Foundation-model / fine-tuning design records (LoRA, AE, DFT, BaLoRA)
│   ├── plans/ · temp/                     # Working notes / design records (theory + history)
│   └── ...
│
├── tests/                                 # Test suite
├── pyproject.toml                         # Project configuration
├── LICENSE                                # MIT License
└── .gitmodules                            # Git submodules (u-dales, LBM)
```

### Libraries

Each library has a dedicated deep-dive doc under [`docs/`](docs/):
[pylbm](docs/pylbm.md), [pyudales](docs/pyudales.md), [pypalm](docs/pypalm.md),
[data-assimilation](docs/data_assimilation.md), and
[neural-surrogates](docs/neural_surrogates.md). The configs and scripts are
covered in [`docs/scripts_and_configs.md`](docs/scripts_and_configs.md), HPC
submission in [`docs/job_scripts.md`](docs/job_scripts.md), and
[`docs/codebase_guide.md`](docs/codebase_guide.md) is the orientation entrypoint
for contributors and AI coding assistants.

#### pyurbanair

The base library. It contains a base forward model, base ensemble forward model, and base rollout forward model. All other libraries that introduce forward models inherit from these base classes. This ensures compatibility throughout the entire repo. The base classes handle common functionality — for example, one only has to implement `run_single` when adding a new forward model, and ensemble simulation is automatically handled by the base class.

#### data-assimilation

Data assimilation functionalities implemented using JAX. Contains an observation operator (for mapping simulation states to observation locations), grid interpolation utilities, a base smoothing class, ESMDA (Ensemble Smoother with Multiple Data Assimilation), a sequential Ensemble Kalman Filter (`filtering/`, cycled state/parameter/joint analysis with pluggable inflation and parameter-evolution models), and optional localization (adaptive correlation-based observation tapering, with an optional "grid block" mode that updates co-located rows jointly). Supports parameter-only, joint state-parameter, and time-varying-parameter estimation. Compatible with every simulation backend.

#### pylbm

A wrapper for Geir Evensen's Lattice Boltzmann simulator. On first import, it automatically downloads the repository from GitHub and compiles the code based on the experiment specifications. Supports STL geometry input and optional CUDA acceleration (via the `cuda` environment).

#### pyudales

A wrapper for the uDALES v2.2.0 simulator. On first import, it automatically downloads the repository from GitHub and compiles the code based on the experiment specifications. Preprocessing can be done with Matlab or with the pure-Python preprocessor in `python_udgeom/`. A timestep watchdog (`utils/run_monitor.py`) detects numerical instability (`dt` collapse) and kills a diverging run early so the ensemble can resample it instead of waiting out a slow crash.

#### pypalm

A wrapper for the PALM model system. It is imported lazily (compiling on first import) so that non-PALM runs never pay the PALM compile cost. Same three-class forward/ensemble shape as the other backends.

#### neural-surrogates

A learned, one-step surrogate of the CFD forward models, built with PyTorch. It provides a dataset generation/loading stack (`TransitionDataset`), architectures (`SimpleConv` baseline, `UNetConvNeXt`, the transformer-based `UPT`, `P3D`, a domain-decomposed model, and a vendored Tadpole autoencoder + its AE→time-stepper), a generic `Trainer` (best-val checkpointing, patience-based early stopping, and the pushforward trick) plus an `AutoencoderTrainer` for representation pre-training, and a `NeuralSurrogateForwardModel` that wraps a trained network as a `BaseForwardModel` so it slots into the ensemble/ESMDA machinery as a fourth backend. `UPT`/`P3D` z-score-normalize their inputs and predict the per-step residual (both required for stable rollouts on dense grids), with normalization statistics computed at training time and stored in the checkpoint. Optional signed-distance-field geometry features (`sdf.py`) give the model a bounded view of the obstacle field. A `finetuning/` module adds parameter-efficient LoRA (PEFT) fine-tuning that merges back into a plain, ESMDA-loadable `weights.pt`; combined with Tadpole autoencoder pre-training and the "Dynamic Fine-Tuning" (DFT) time-stepper head, this supports a foundation-model pre-train → fine-tune workflow (design records in [`docs/neural_surrogate_plans/`](docs/neural_surrogate_plans/)). A cold start is bootstrapped by the CFD backend that generated its training data (or warm-started from saved training trajectories via `training_spinup.py`); warm starts step the network directly. See [`docs/neural_surrogates.md`](docs/neural_surrogates.md) for the full stack.

## Benchmark Geometry

A script to generate the geometry in `stl` (as well as other formats) for the Xie and Castro 2008 benchmark can be found in the `examples/benchmark_geometry/` folder.

By importing `XieCastroBenchmarkGeometry` from `boundary_geometry.py` one can configure and serialize the specific setup. There is also a command-line tool available. The dependencies are available in the `dev` environment. Usage:

```
pixi shell -e dev
python examples/benchmark_geometry/benchmark_geometry.py --help
```

One example for STL is:

```
python examples/benchmark_geometry/benchmark_geometry.py stl output --num-tiles 3 3
```

For Geir Evensen's Lattice Boltzmann code one can also configure a Fortran file, which needs to be compiled subsequently. To change the base resolution one can provide a refinement factor as well:

```
python examples/benchmark_geometry/benchmark_geometry.py stl output --resolution 4 --num-tiles 3 3
```

## Data and File Types

The main data and file types are NetCDF and xarray. All forward models take in parameters as xarray Datasets and output states as xarray Datasets. When the model is configured to save, simulation outputs are always stored in NetCDF format to ensure compatibility across libraries.

### State data

States are always provided and output as xarray Datasets. They should have the following format:

```
Dimensions:  (time: 1, zm: 6, yt: 128, xt: 128, zt: 6, ym: 128, xm: 128)
Coordinates:
  * time     (time) float32 4B 50.27
  * zm       (zm) float32 24B 0.0 6.667 13.33 20.0 26.67 33.33
  * yt       (yt) float32 512B 0.625 1.875 3.125 4.375 ... 156.9 158.1 159.4
  * xt       (xt) float32 512B 0.625 1.875 3.125 4.375 ... 156.9 158.1 159.4
  * zt       (zt) float32 24B 3.333 10.0 16.67 23.33 30.0 36.67
  * ym       (ym) float32 512B 0.0 1.25 2.5 3.75 5.0 ... 155.0 156.2 157.5 158.8
  * xm       (xm) float32 512B 0.0 1.25 2.5 3.75 5.0 ... 155.0 156.2 157.5 158.8
Data variables:
    w        (time, zm, yt, xt) float32 393kB 0.0 0.0 0.0 ... -0.0216 -0.01649
    pres     (time, zt, yt, xt) float32 393kB ...
    v        (time, zt, ym, xt) float32 393kB -0.06821 -0.1152 ... 0.5625 0.5629
    u        (time, zt, yt, xm) float32 393kB -0.05016 0.08196 ... 3.14 3.139
```

Note that `xt` vs `xm` is uDALES-specific (staggered grid). For pylbm there is only `x`, `y`, `z`. However, `time` should always be present, even when only one time step is stored.

Ensembles of states are also in xarray format with an added `ensemble` dimension:

```
Dimensions:  (ensemble: 50, time: 1, zm: 6, yt: 128, xt: 128,
              zt: 6, ym: 128, xm: 128)
Coordinates:
  * time     (time) float32 4B 50.15
  * zm       (zm) float32 24B 0.0 6.667 13.33 20.0 26.67 33.33
  * yt       (yt) float32 512B 0.625 1.875 3.125 4.375 ... 156.9 158.1 159.4
  * xt       (xt) float32 512B 0.625 1.875 3.125 4.375 ... 156.9 158.1 159.4
  * zt       (zt) float32 24B 3.333 10.0 16.67 23.33 30.0 36.67
  * ym       (ym) float32 512B 0.0 1.25 2.5 3.75 5.0 ... 155.0 156.2 157.5 158.8
  * xm       (xm) float32 512B 0.0 1.25 2.5 3.75 5.0 ... 155.0 156.2 157.5 158.8
Dimensions without coordinates: ensemble
Data variables:
    w        (ensemble, time, zm, yt, xt) float32 59MB 0.0 ... -0...
    pres     (ensemble, time, zt, yt, xt) float32 59MB -0.1003 .....
    v        (ensemble, time, zt, ym, xt) float32 59MB -0.38 ... ...
    u        (ensemble, time, zt, yt, xm) float32 59MB 0.3301 ......
```

### Parameter data

Parameters are provided as an xarray Dataset when calling the forward model:

```python
true_params = xarray.Dataset(
    data_vars={
        "inflow_angle": TRUE_ANGLE,
        "velocity_magnitude": TRUE_VELOCITY_MAGNITUDE,
        "pressure_gradient_magnitude": TRUE_PRESSURE_GRADIENT,
    },
)
```

Currently, `inflow_angle`, `velocity_magnitude`, and `pressure_gradient_magnitude` are supported. Note that `pressure_gradient_magnitude` is only used by pyudales.

An ensemble of parameters can be provided in the same manner, with an added `ensemble` dimension:

```python
params_ensemble = xarray.Dataset(
    data_vars={
        "inflow_angle": ("ensemble", inflow_angle_range),
        "velocity_magnitude": ("ensemble", velocity_magnitude_range),
    },
    coords={"ensemble": jnp.arange(len(inflow_angle_range))},
)
```

Running with an ensemble of parameters automatically simulates an ensemble. This is handled by the base forward model.

## Development

When adding to the repository, first create a new branch. Then make the changes you want, commit, and create a pull request.

If you want to add to the repository you should make use of the linting and formatting. These are automatically installed in the dev environment. Simply run:

```
pixi run pre-commit
```

and it will apply formatting and give you errors to be fixed. Note that it only applies to files that are staged. Sometimes the linting gives errors that you don't necessarily want to fix. These errors you can ignore by adding the following after the line in question:

```python
# type: ignore[<something>]
```

There is currently no protection on the main branch. Committing directly is possible without passing pre-commit. Please be mindful before committing.

For AI coding assistants, [`docs/codebase_guide.md`](docs/codebase_guide.md) is a
fast-orientation sheet covering the internal structure, contracts, and
conventions.

## License

MIT License. Copyright (c) 2025 Nikolaj T. Mucke. See [LICENSE](LICENSE) for details.
