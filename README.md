# pyurbanair

Urban airflow simulation and ensemble data assimilation, part of the UrbanAIR
project. pyurbanair runs CFD solvers and learned surrogates behind one Python
interface, in ensembles, and estimates inflow parameters and flow states from
sensor data.

[![scripts](https://github.com/nmucke/pyurbanair/actions/workflows/tests-scripts.yml/badge.svg)](https://github.com/nmucke/pyurbanair/actions/workflows/tests-scripts.yml)
[![data-assimilation](https://github.com/nmucke/pyurbanair/actions/workflows/tests-data-assimilation.yml/badge.svg)](https://github.com/nmucke/pyurbanair/actions/workflows/tests-data-assimilation.yml)
[![neural-surrogates](https://github.com/nmucke/pyurbanair/actions/workflows/tests-neural-surrogates.yml/badge.svg)](https://github.com/nmucke/pyurbanair/actions/workflows/tests-neural-surrogates.yml)
[![pyudales](https://github.com/nmucke/pyurbanair/actions/workflows/tests-pyudales.yml/badge.svg)](https://github.com/nmucke/pyurbanair/actions/workflows/tests-pyudales.yml)
[![pylbm](https://github.com/nmucke/pyurbanair/actions/workflows/tests-pylbm.yml/badge.svg)](https://github.com/nmucke/pyurbanair/actions/workflows/tests-pylbm.yml)
[![pypalm](https://github.com/nmucke/pyurbanair/actions/workflows/tests-pypalm.yml/badge.svg)](https://github.com/nmucke/pyurbanair/actions/workflows/tests-pypalm.yml)

> Under active development (v0.1.0): interfaces still change.

## What it does

- **Forward simulation** with three CFD backends, the Lattice Boltzmann
  solver (`pylbm`), [uDALES](https://github.com/uDALES/u-dales) (`pyudales`)
  and [PALM](https://palm.muk.uni-hannover.de) (`pypalm`), plus trained
  neural surrogates as a fast drop-in fourth backend.
- **Data assimilation** in JAX: the ESMDA smoother, ensemble Kalman filters
  (stochastic, ETKF, LETKF) and a hybrid of the two. It estimates static or
  time-varying inflow parameters, the flow state, or both.
- **Neural surrogates**: generate training data, train next-step models,
  autoencoders and latent generators, fine-tune (full or LoRA) and evaluate
  them.

## Install

Everything runs through [Pixi](https://pixi.sh):

```bash
curl -fsSL https://pixi.sh/install.sh | sh   # once
pixi run setup-dev                           # install the dev environment
pixi shell -e dev                            # activate it
```

Other environments: `cuda` (GPU), `snellius` and `delftblue` (HPC), `mcp`
(the agent interface), `rendering` (3D views).

Supported platforms are Linux (linux-64) and macOS on Apple silicon
(osx-arm64). The pixi environment brings the compilers, MPI, NetCDF and FFTW;
nothing else needs installing on Linux. On macOS, also install Xcode's command
line tools once (`xcode-select --install`) for the SDK and Apple's linker. The
solvers build themselves on first use: uDALES and the LBM from the pinned
submodules, PALM from a pinned release it downloads once.

If pixi fails to link a package after an update (e.g. `failed to link
pytorch`), delete the environment and reinstall: `rm -rf .pixi/envs/dev &&
pixi run setup-dev`.

## Quick start

Every run is configured by [Hydra](https://hydra.cc) from `configs/`. Override
anything on the command line. Outputs go to `.temp/` by default.

```bash
# Forward run (uDALES by default), then its figures
python scripts/run_forward.py model=pylbm forward.ensemble=true
bash workflows/forward_workflow.sh model=pylbm   # the same, plus figures

# Data assimilation: run, compute metrics, draw figures
bash workflows/assimilation_workflow.sh smoother
bash workflows/assimilation_workflow.sh filtering params@prior_params=static
bash workflows/assimilation_workflow.sh hybrid

# Neural surrogates
python scripts/surrogate/generate_data.py
python scripts/surrogate/train.py --config-name surrogate/train_stepper
python scripts/surrogate/evaluate_stepper.py
```

Each script's docstring lists its options and outputs.
[configs/README.md](configs/README.md) shows the common overrides.

## Repository layout

```text
configs/     Hydra configs: forward.yaml, assimilation.yaml, surrogate/, and
             the case, model and params groups
scripts/     the scripts you run (run_forward, run_smoother, run_filtering,
             run_hybrid, compute_metrics, visualize_*), surrogate/ and tools/;
             their shared helpers are in scripts/utils/
workflows/   shell pipelines: a run followed by its post-processing
geometries/  case inputs, one folder per case (STL, uDALES and PALM
             templates), plus the UrbanTALES training-geometry templates
src/         pyurbanair: the base classes every backend inherits
libs/        pylbm, pyudales, pypalm, neural-surrogates, data-assimilation,
             evaluation, visualization, mcp-server
tests/       one folder per package, plus tests of scripts/ and configs/
docs/        reference documentation
archive/     the previous configs, scripts and tests (not maintained)
```

## Documentation

| Topic | Doc |
|---|---|
| Orientation and "how do I add X" | [docs/codebase_guide.md](docs/codebase_guide.md) |
| Configs, scripts and workflows | [docs/scripts_and_configs.md](docs/scripts_and_configs.md), config keys in [configs/README.md](configs/README.md) |
| Cases and geometries | [geometries/README.md](geometries/README.md) |
| Data assimilation | [docs/data_assimilation.md](docs/data_assimilation.md) |
| Neural surrogates | [docs/neural_surrogates.md](docs/neural_surrogates.md) |
| Evaluation metrics and figures | [docs/evaluation.md](docs/evaluation.md) |
| Backends | [pylbm](docs/pylbm.md), [pyudales](docs/pyudales.md), [pypalm](docs/pypalm.md) |
| Tests | [tests/README.md](tests/README.md) |
| HPC jobs | [docs/job_scripts.md](docs/job_scripts.md) |
| Agent interface (MCP) | [docs/mcp.md](docs/mcp.md) (`pixi run -e mcp register-claude` adds it to Claude Code) |
| HTML forward-run viewer | [docs/visualization.md](docs/visualization.md) |

Plans, research notes and the archive live in `docs/plans/`, `docs/research/`
and `docs/archive/`; they are working notes, not maintained references.

## Development

```bash
pixi run -e dev py.test            # tests without compiled solvers
pixi run -e dev test-integration   # tiny real-solver runs
pixi run -e dev pre-commit         # black, isort, mypy on staged files
```

Work on a branch and open a pull request. CI runs each package's tests when
that package changes.

## License

MIT. See [LICENSE](LICENSE).
