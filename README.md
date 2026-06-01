# pyurbanair

A Python framework for urban air flow simulation and ensemble-based data assimilation. Part of the UrbanAIR project.

> **Note:** This repository is under active development (v0.1.0). Things will change and some functionalities may not work as intended.

## Features

- **Three CFD backends:** pylbm (Lattice Boltzmann Method, wrapping Geir Evensen's LBM), pyudales (wrapping uDALES v2.2.0), and pypalm (wrapping the PALM model system)
- **Neural surrogate backend:** train a learned one-step network on CFD ensembles and run it as a drop-in fourth forward model — full stack (data generation, training, autoregressive rollout) in [`neural-surrogates`](libs/neural-surrogates), documented in [`docs/neural_surrogates.md`](docs/neural_surrogates.md)
- **Ensemble-based data assimilation** using ESMDA (Ensemble Smoother with Multiple Data Assimilation), implemented in JAX
- **Parameter estimation** and **joint state-parameter estimation**
- **Localization** for ESMDA (correlation-based observation tapering)
- **Multi-step rollout simulations** with state carry-over between time windows
- **Cross-model assimilation** (e.g., use LBM as truth model with uDALES — or a neural surrogate — for assimilation)
- **Time-varying parameters** with per-window mean/std profiles for the inflow priors
- **Observation operators** for mapping simulation states to observation space
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

All simulation and assimilation settings live in `conf/` as composable
[Hydra](https://hydra.cc/) config groups. The top-level `conf/config.yaml`
selects one option per group; any field can be overridden from the
command line. The groups are:

- **`domain/`** — grid resolution (`nx`, `ny`, `nz`) and spatial bounds
- **`time/`** — simulation duration, output frequency, spinup time
- **`model/`** — per-backend forward-model and ensemble-model
  `_target_` blocks (`pylbm`, `pyudales`, `pypalm`, `neural_surrogate`)
- **`ensemble/`** — ensemble size, parallel processes, CPUs/process,
  failure policy
- **`obs/`** — observation operator schema (points/grid mode, sensor
  locations, observed states, temporal aggregation)
- **`esmda/`** — smoother variant (`parameter`, `state_and_parameter`,
  `rollout`, `time_varying_parameter`, `time_varying_rollout`), number
  of assimilation steps/windows, observation error std, random seed
- **`localization/`** — ESMDA localization: `none` (global update) or
  `correlation` (correlation-based observation tapering)
- **`params/{true,prior,external}/`** — true parameter values,
  assimilation prior, external/expert prior
- **`time_varying/`** — time-varying-parameter method
  (`ar2_relaxation`, `ar1`, `gp_linear_trend`, `ornstein_uhlenbeck`)
  and per-method kwargs for both the assimilation prior and the truth
  trajectory. The external prior `mean`/`std` (`params/external/`) may be
  a scalar **or a list of control points** interpolated over the window,
  letting `x_ext(t)` / `Σ_ext(t)` vary in time (see
  `conf/params/external/time_varying.yaml`)
- **`training_data/`** — neural-surrogate dataset size presets (`tiny`,
  `small`, `medium`, `large`, `xlarge`); see
  [`docs/neural_surrogates.md`](docs/neural_surrogates.md)
- **`neural_surrogate_architectures/`, `neural_surrogate_training/`,
  `neural_surrogate_testing/`** — surrogate architecture presets,
  training loop, and autoregressive-rollout test config
- **`preset/`** — bundled overlays (`small`, `test`) for fast runs

### Forward simulations

```bash
# Single forward simulation
python scripts/run_forward_model.py model=pylbm
python scripts/run_forward_model.py model=pyudales

# Ensemble forward simulation
python scripts/run_ensemble_forward_model.py model=pylbm

# Multi-step rollout simulation
python scripts/run_rollout_forward_model.py model=pylbm run.num_steps=4

# Ensemble rollout simulation
python scripts/run_ensemble_rollout_forward_model.py model=pylbm
```

### Data assimilation

The `esmda` group defaults to `parameter` in `conf/config.yaml`. Every
other smoother variant needs an explicit `esmda=<name>` selector (the
examples below do this for `state_and_parameter`, `rollout`, and
`time_varying_parameter`).

```bash
# Parameter estimation with ESMDA
python scripts/run_parameter_esmda.py \
  model@truth_model=pylbm model@assim_model=pylbm

# Cross-model assimilation (LBM truth, uDALES assimilation)
python scripts/run_parameter_esmda.py \
  model@truth_model=pylbm model@assim_model=pyudales

# Joint state and parameter estimation
python scripts/run_state_and_parameter_esmda.py \
  esmda=state_and_parameter \
  model@truth_model=pylbm model@assim_model=pylbm

# Rollout-based ESMDA with multiple assimilation windows
python scripts/run_rollout_esmda.py \
  esmda=rollout \
  model@truth_model=pylbm model@assim_model=pylbm

# Time-varying-parameter ESMDA
python scripts/run_time_varying_parameter_esmda.py \
  esmda=time_varying_parameter \
  model@truth_model=pylbm model@assim_model=pylbm \
  esmda.num_steps=4 obs.interval_size=2

# Fast test preset (small domain, few steps, CPU-only LBM)
python scripts/run_parameter_esmda.py preset=test
```

All forward models generate a `.temp` folder where intermediate files are stored.

### Neural surrogates

A learned one-step network can be trained on a CFD ensemble and then used
as a drop-in fourth forward model alongside pylbm, pyudales, and pypalm.
The end-to-end stack (dataset generation → training → autoregressive
rollout → use as a forward/assimilation model) is documented in
[`docs/neural_surrogates.md`](docs/neural_surrogates.md). The headline
commands:

```bash
# 1. Generate a training dataset by driving a CFD ensemble
pixi run -e dev python scripts/generate_training_data.py training_data=small model=pylbm

# 2. Train a surrogate (UNetConvNeXt by default; pick a preset/architecture)
pixi run -e dev python scripts/train_neural_surrogate.py \
    dataset.root_dir=training_data/pylbm_small \
    'neural_surrogate_architectures/unet_convnext@architecture=small'

# 3. Autoregressive rollout on the test split (diagnostic plots + animation)
pixi run -e dev python scripts/test_neural_surrogate.py \
    model_dir=model_weights/unet_convnext_small sample_idx=0

# 4. Use the trained surrogate as an assimilation model
python scripts/run_time_varying_parameters_rollout_esmda.py \
    model@truth_model=pyudales model@assim_model=neural_surrogate \
    assim_model.forward_model.model_dir=model_weights/unet_convnext_small
```

### Running on Snellius (SLURM)

The Snellius `snellius` env ships with a one-command submit wrapper that picks
the partition, requests the right number of cores, and sets a sensible wall
time — all from `conf/size/<size>.yaml`. Use it instead of writing your own
sbatch files. Full details: [`job_scripts/snellius/README.md`](job_scripts/snellius/README.md).

```bash
# Pattern
job_scripts/snellius/submit.sh <model> <size> [extra hydra overrides...]
#   <model>   pylbm | pyudales | pypalm     (assimilation forward model)
#   <size>    tiny | small | medium | large | xlarge
```

Common launches:

| Goal                                          | Command                                                                       |
|-----------------------------------------------|-------------------------------------------------------------------------------|
| pylbm, small run                              | `job_scripts/snellius/submit.sh pylbm small`                                  |
| pyudales, medium run                          | `job_scripts/snellius/submit.sh pyudales medium`                              |
| pypalm, small run                             | `job_scripts/snellius/submit.sh pypalm small`                                 |
| Twin experiment (truth ≠ assim model)         | `TRUTH_MODEL=pyudales job_scripts/snellius/submit.sh pylbm small`             |
| Ad-hoc Hydra override (per submission)        | `job_scripts/snellius/submit.sh pylbm small esmda.num_assimilation_windows=3` |
| Custom wall time (overrides the size default) | `WALLTIME=30:00:00 job_scripts/snellius/submit.sh pyudales medium`            |
| Preview only (don't submit)                   | `DRY_RUN=1 job_scripts/snellius/submit.sh pyudales medium`                    |

**Tuning a run.** Edit the three per-size knobs in `conf/size/<size>.yaml`; the
wrapper reads `ensemble.ensemble_size` and sizes the SLURM allocation
automatically (one core per ensemble member, rounded up to the partition's
billing minimum — 16 on `rome`, 24 on `genoa`):

| Knob                              | Meaning                            |
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
│       ├── base_rollout_forward_model.py  # Multi-step rollout simulations
│       ├── animation.py                   # Animation utilities
│       ├── plotting.py                    # Plotting utilities
│       ├── config/
│       │   └── hydra_helpers.py           # Helpers consumed by Hydra configs (instantiate targets)
│       └── utils/
│           ├── state_utils.py             # State manipulation utilities
│           ├── run_utils.py               # Runtime utilities
│           └── animation_utils.py         # Animation generation helpers
│
├── libs/                                  # Sub-libraries
│   ├── data-assimilation/                 # Data assimilation library (JAX)
│   │   ├── pyproject.toml
│   │   └── src/data_assimilation/
│   │       ├── observation_operator.py    # Maps states to observation space
│   │       ├── interpolation.py           # Grid interpolation utilities
│   │       └── smoothing/
│   │           ├── base.py                # Base smoothing class
│   │           └── esmda.py               # ESMDA implementation
│   │
│   ├── pylbm/                             # Lattice Boltzmann Method wrapper
│   │   ├── pyproject.toml
│   │   └── src/pylbm/
│   │       ├── forward_model.py
│   │       ├── ensemble_forward_model.py
│   │       ├── rollout_forward_model.py
│   │       ├── stl_to_lbm.py             # STL geometry conversion
│   │       └── utils/
│   │
│   ├── pyudales/                          # uDALES wrapper
│   │   ├── pyproject.toml
│   │   └── src/pyudales/
│   │       ├── forward_model.py
│   │       ├── ensemble_forward_model.py
│   │       ├── rollout_forward_model.py
│   │       ├── python_udgeom/            # Python preprocessing (Matlab alternative)
│   │       └── utils/
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
│           ├── data.py                    # TransitionDataset
│           ├── training.py                # Trainer (train/val loop, pushforward)
│           ├── geometry.py                # STL → voxel geometry channel
│           └── architectures/             # SimpleConv, UNetConvNeXt
│
├── conf/                                  # Hydra config groups (see Configuration)
│   ├── config.yaml                        # Top-level composition
│   ├── domain/, time/, model/, ensemble/, paths/, size/
│   ├── obs/, esmda/, localization/, params/, time_varying/
│   ├── training_data/                     # Surrogate dataset size presets
│   ├── neural_surrogate_architectures/    # Surrogate architecture presets
│   ├── neural_surrogate_training/, neural_surrogate_testing/
│   └── preset/                            # Bundled overlays (small, test)
│
├── scripts/                               # Main execution scripts
│   ├── run_forward_model.py               # Single forward simulation
│   ├── run_ensemble_forward_model.py      # Ensemble forward simulation
│   ├── run_rollout_forward_model.py       # Multi-step rollout
│   ├── run_ensemble_rollout_forward_model.py  # Ensemble rollout
│   ├── run_time_varying_forward_model.py  # Time-varying inflow
│   ├── run_parameter_esmda.py             # Parameter estimation via ESMDA
│   ├── run_state_and_parameter_esmda.py   # Joint state-parameter estimation
│   ├── run_rollout_esmda.py               # Rollout-based ESMDA
│   ├── run_time_varying_parameter_esmda.py
│   ├── run_time_varying_parameters_rollout_esmda.py
│   ├── generate_training_data.py          # Build surrogate training dataset
│   ├── train_neural_surrogate.py          # Train a surrogate
│   ├── test_neural_surrogate.py           # Autoregressive rollout on test split
│   └── dataloading.py                     # TransitionDataset smoke test
│
├── examples/                              # Example experiments
│   ├── benchmark_geometry/                # Xie and Castro 2008 geometry tools
│   ├── lbm/experiments/                   # LBM experiment configs (STL files)
│   └── udales/experiments/                # uDALES experiment configs
│
├── tests/                                 # Test suite
├── pyproject.toml                         # Project configuration
├── LICENSE                                # MIT License
└── .gitmodules                            # Git submodules (u-dales, LBM)
```

### Libraries

#### pyurbanair

The base library. It contains a base forward model, base ensemble forward model, and base rollout forward model. All other libraries that introduce forward models inherit from these base classes. This ensures compatibility throughout the entire repo. The base classes handle common functionality — for example, one only has to implement `run_single` when adding a new forward model, and ensemble simulation is automatically handled by the base class.

#### data-assimilation

Data assimilation functionalities implemented using JAX. Contains an observation operator (for mapping simulation states to observation locations), grid interpolation utilities, a base smoothing class, ESMDA (Ensemble Smoother with Multiple Data Assimilation), and optional localization (correlation-based observation tapering). Supports parameter-only, joint state-parameter, and time-varying-parameter estimation. Compatible with every simulation backend.

#### pylbm

A wrapper for Geir Evensen's Lattice Boltzmann simulator. On first import, it automatically downloads the repository from GitHub and compiles the code based on the experiment specifications. Supports STL geometry input and optional CUDA acceleration (via the `cuda` environment).

> **Caveat:** The STL-to-LBM geometry conversion has been implemented but may not be completely correct. Do not fully trust outputs from pylbm when using STL geometry.

#### pyudales

A wrapper for the uDALES v2.2.0 simulator. On first import, it automatically downloads the repository from GitHub and compiles the code based on the experiment specifications. Preprocessing can be done with Matlab or with the pure-Python preprocessor in `python_udgeom/`.

#### pypalm

A wrapper for the PALM model system. It is imported lazily (compiling on first import) so that non-PALM runs never pay the PALM compile cost. Same three-class forward/ensemble shape as the other backends.

#### neural-surrogates

A learned, one-step surrogate of the CFD forward models, built with PyTorch. It provides a dataset generation/loading stack (`TransitionDataset`), architectures (`SimpleConv` baseline and `UNetConvNeXt`), a generic `Trainer` (best-val checkpointing, patience-based early stopping, and the pushforward trick), and a `NeuralSurrogateForwardModel` that wraps a trained network as a `BaseForwardModel` so it slots into the ensemble/ESMDA machinery as a fourth backend. A cold start is bootstrapped by the CFD backend that generated its training data; warm starts step the network directly. See [`docs/neural_surrogates.md`](docs/neural_surrogates.md) for the full stack.

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

## License

MIT License. Copyright (c) 2025 Nikolaj T. Mucke. See [LICENSE](LICENSE) for details.
