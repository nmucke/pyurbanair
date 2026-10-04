# pyurbanair — Codebase Guide (for AI coding assistants)

This is a fast-orientation sheet aimed at LLM coding tools. The user-facing
[`README.md`](../README.md) covers install / usage. This sheet covers
**internal structure, contracts, and conventions** so an assistant can land
non-trivial edits without re-deriving them.

## 1. What this repo is

`pyurbanair` is a Python monorepo for urban-airflow CFD ensembles and
ensemble data assimilation. It wraps three Fortran CFD solvers (plus a learned
surrogate) behind a common Python interface and runs them in ensembles for
parameter / state estimation.

- **Three CFD backends**, each in [libs/](../libs/) as an editable subpackage:
  - `pylbm`  — Lattice Boltzmann (Geir Evensen). STL geometry. Optional CUDA.
  - `pyudales` — uDALES v2.2.0. Staggered grid; Matlab or Python preprocessing.
  - `pypalm` — PALM model system. Imports lazily (compiles on first import).
- **`neural-surrogates`** — a learned one-step surrogate usable as a fourth
  backend.
- **`data-assimilation`** implements ESMDA, the EnKF family and the
  ESMDA × filter hybrid in JAX.
- **`pyurbanair`** (top-level package) holds the base classes that *every*
  backend's forward / ensemble model inherits from. Polymorphism is via these
  base classes — data assimilation never depends on a specific solver.
- All public I/O is `xarray.Dataset`. On-disk format is NetCDF.

## 1a. Documentation map — where to read before you edit

This guide is the **entrypoint**; it stays at the level of cross-cutting
structure, contracts, and conventions. When a task targets one area, **read
that doc first** — it has the file-level detail, gotchas, and recipes this
guide only summarizes:

| When the task touches… | Read |
|---|---|
| Hydra configs (`configs/`), the scripts (`scripts/`) and `workflows/` | [scripts_and_configs.md](scripts_and_configs.md); every key and override in [configs/README.md](../configs/README.md) |
| ESMDA, filtering, the hybrid, observation operator, localization, state reduction | [data_assimilation.md](data_assimilation.md) |
| Neural surrogates (architectures, training, rollout, domain decomposition) | [neural_surrogates.md](neural_surrogates.md) |
| The LBM backend (compile, `infile.in`, STL geometry, CUDA, warm starts) | [pylbm.md](pylbm.md) |
| The uDALES backend (`namoptions`, staggered grid, nudging, dt watchdog, discrepancy) | [pyudales.md](pyudales.md) |
| The PALM backend (lazy import, `_p3d` namelists, direct-run path, topography) | [pypalm.md](pypalm.md) |
| DA metrics and figures (`libs/evaluation`) | [evaluation.md](evaluation.md) |
| Local MCP forward jobs, preparation and client setup | [mcp.md](mcp.md) |
| Saved forward visualization and browser bundles (`libs/visualization`) | [visualization.md](visualization.md) |
| Running on HPC clusters (Snellius / DelftBlue) | [job_scripts.md](job_scripts.md) |
| Case inputs (`geometries/`) | [geometries/README.md](../geometries/README.md) |
| Tests (`tests/`: per-package folders, script tests, overlays) | [tests/README.md](../tests/README.md) |

Not maintained references — verify against the code before relying on them:

- `docs/plans/` — open plans; `implemented/` and `rejected/` keep the finished
  and dropped ones as design records.
- `docs/research/` — research notes, reviews and theory.
- `docs/archive/` — superseded docs, including the old setup's
  `scripts_and_configs_archived_setup.md`.

`archive/` at the repo root holds the retired `conf/`, `scripts/` and `tests/`
(dead code: not run, not tested). Docs that still name `conf/...`,
`scripts/esmda/...`, `run_esmda.py` etc. describe that archived setup.

## 2. Monorepo layout

```
src/pyurbanair/                    # Top-level package: base classes + glue
  base_forward_model.py            # BaseForwardModel
  base_ensemble_forward_model.py   # BaseEnsembleForwardModel (parallel/seq, failure policy)
  quiet_jax.py                     # Import before `jax` to suppress CPU-fallback noise
  static_parameters/               # ParameterSampler + Normal/Uniform/Constant distributions
  dynamic_parameters/              # AR2RelaxationModel (time-varying prior),
                                   #   HarmonicParameterModel (smooth truths)
  training_data/                   # UniformExternalAR2Sampler for surrogate training data
  config/
    hydra_helpers.py               # Targets the model configs instantiate: prepare_compile,
                                   #   prepare_udales, prepare_neural_surrogate; plus
                                   #   clean_outputs, resolve_parameter_schema
    discrepancy.py                 # validate_sgs_discrepancy_settings (uDALES discrepancy)
  utils/
    cpu_pinning.py                 # Worker → CPU pinning for parallel ensembles
    solver_process.py              # run_solver: every backend's solver launch
    toolchain.py                   # apple_linker_flags: macOS linking for solver builds
    run_utils.py, animation_utils.py

configs/                           # Hydra config (see §5); keys in configs/README.md
  forward.yaml                     # Entry point for run_forward.py
  assimilation.yaml                # Entry point for run_smoother/run_filtering/run_hybrid.py
  common.yaml                      # paths (per machine), ensemble budget, Hydra run dir
  case/                            # One self-contained file per case (xie_and_castro, barcelona)
  params/                          # Parameter samplers (mounted twice for DA: truth + prior)
  assimilation_settings/           # Every option of each DA component, at top level
  model/                           # forward + ensemble backend (mounted under model@<pkg>)
  surrogate/                       # Neural-surrogate configs (--config-name surrogate/<name>)
  visualization/                   # Render presets of the MCP viewer (quicklook, flow_3d)

libs/data-assimilation/src/data_assimilation/   # see docs/data_assimilation.md
  observation_operator.py          # ObservationOperator, TemporalObservationOperator,
                                   #   AggregateObservations
  observation_error.py             # ObservationErrorSpec (physical R)
  interpolation.py                 # Grid → sensor-point interpolation
  augmentation.py, parameter_selection.py, io.py
  smoothing/                       # BaseSmoothing, the ESMDA variants (esmda.py)
  filtering/                       # EnsembleKalmanFilter, analyses (stochastic, (L)ETKF),
                                   #   parameter evolution
  filter_smoothing/                # FilterSmoothing (the hybrid) + beta tempering
  localization/                    # BaseLocalization, CorrelationLocalization, DistanceLocalization
  reduction.py                     # OnlineStateReduction, StreamingStateReduction
  inflation.py                     # Multiplicative, RTPS, RTPP

libs/evaluation/src/evaluation/    # Metrics + figures for DA runs (docs/evaluation.md).
                                   #   Leaf lib: no jax, no pyurbanair, no backends.
libs/mcp-server/src/mcp_server/    # Optional MCP server for forward runs (docs/mcp.md)
libs/visualization/src/visualization/ # Forward-run rendering + browser viewer (docs/visualization.md)

libs/pylbm/src/pylbm/              # LBM wrapper. __init__ locates/clones the LBM Fortran code.
  forward_model.py                 # ForwardModel(BaseForwardModel)
  ensemble_forward_model.py        # EnsembleForwardModel(BaseEnsembleForwardModel)
  stl_to_lbm.py                    # STL → LBM voxel geometry
  utils/                           # infile.in editing, compile, warm-start, params, ...
libs/pyudales/src/pyudales/        # uDALES wrapper; explicit cached solver preparation.
  forward_model.py, ensemble_forward_model.py
  python_udgeom/                   # Python preprocessing alternative to Matlab
  utils/                           # namoptions, nudging, ncpu, warm-start, run_monitor, ...
libs/pypalm/src/pypalm/            # PALM wrapper. Same shape (+ direct_palm.py, stl_to_palm.py).
libs/neural-surrogates/src/neural_surrogates/   # Learned one-step CFD surrogate (PyTorch)
  forward_model.py, ensemble_forward_model.py   # NeuralSurrogate{,Ensemble}ForwardModel
  architectures/                   # p3d, unet_convnext, upt, simple_conv, domain_decomposed, tadpole_*
  datasets/, training/, finetuning/

scripts/                           # Scripts you run; their shared helpers are in utils/.
  run_forward.py                   # Forward sim — single/ensemble, extra windows
  run_smoother.py                  # ESMDA over consecutive windows
  run_filtering.py                 # Cycled EnKF
  run_hybrid.py                    # Per window: ESMDA params, then filter state
  compute_metrics.py               # metrics.yaml from a finished DA run dir (plain CLI)
  visualize_forward.py, visualize_assimilation.py   # Figures from a run dir (plain CLI)
  surrogate/                       # generate_data.py, train.py, evaluate_*.py
  utils/                           # Helpers only: inconsistency_check.py (check_config),
                                   #   helper_functions.py (truth, observations, ensemble
                                   #   model, I/O), tasks.py (train.py), eval_common.py
  tools/                           # Geometry CLIs
  setup_dev_env.sh, start_mcp, register_claude.sh

workflows/                         # forward_workflow.sh, assimilation_workflow.sh <method>
geometries/                        # Case inputs, one folder per case (+ urbantales/)
tests/                             # One folder per package + scripts/ + configs/ (tiny
                                   #   overlays). See tests/README.md.
job_scripts/                       # SLURM wrappers (snellius/, delftblue/)
archive/                           # Retired conf/, scripts/, tests/ (not run or tested)
.temp/                             # Default scratch dir. Everything mutable lands here.
```

## 3. The core abstraction — forward models

All backends conform to the same two-class shape, declared in
[src/pyurbanair/](../src/pyurbanair/) and inherited by each backend.

### `BaseForwardModel` — single simulation
- File: [src/pyurbanair/base_forward_model.py](../src/pyurbanair/base_forward_model.py)
- Subclasses implement `run_single`, `_apply_inflow_settings`,
  `save_results`, `_clean_output`.
- Save mode is determined by whether `results_dir` was passed:
  - `results_dir=None` → **in-memory** mode → `__call__` returns
    the `xarray.Dataset`.
  - `results_dir=<path>` → **on-disk** mode → state is written to
    `{results_dir}/{sim_name}.nc` and `__call__` returns `None`.
- `__call__(state, params, sim_name)` is the public entry. It calls
  `run_single` then saves and cleans.
- `state` / `params` are always `xarray.Dataset` (or `None`).

### `BaseEnsembleForwardModel` — ensemble of N members
- File: [src/pyurbanair/base_ensemble_forward_model.py](../src/pyurbanair/base_ensemble_forward_model.py)
- Holds `self.ensemble_forward_models: list[BaseForwardModel]` populated via
  `_create_new_forward_model` (subclass implements this — it clones the
  template model into a per-member temp dir).
- Dispatch in `run_ensemble`:
  1. `num_parallel_processes > 1` → `_run_parallel` (ProcessPoolExecutor +
     `forkserver`).
  2. else save_in_memory → sequential, returns concatenated dataset.
  3. else save_on_disk → sequential, writes per-member files.
- **Failure policy** — passed at construction via the `failure=` arg (each
  `configs/model/*.yaml` wires `failure: ${ensemble.failure}`); reconfigurable
  via `configure_failure_policy`:
  - `"raise"` (the default `ensemble.failure.policy` in `common.yaml`) — the
    first failure aborts the whole ensemble.
  - `"resample_from_successes"` — failed members are cloned from a random
    successful donor (in memory or on disk); the *params* ensemble is re-cloned
    (with Gaussian jitter) by `apply_failure_substitutions_to_params(params)`.
- **CPU pinning**: parallel runs pin workers to distinct cores via
  [src/pyurbanair/utils/cpu_pinning.py](../src/pyurbanair/utils/cpu_pinning.py).
  Disable with `PYURBANAIR_DISABLE_CPU_PINNING=1`.
- mp context is **forkserver**, not fork, because JAX starts background
  threads at import.

Multi-window driving lives in the scripts (the DA scripts' window loops and
`run_forward.py`'s `forward.rollout_steps` loop): each window's final state is
fed in as the next window's warm start and time-varying parameters are
extrapolated between windows.

## 4. Data contracts

**State** = `xarray.Dataset` with at least a `time` dimension. Grid axes
depend on backend:
- pylbm / pypalm — `x, y, z` (PALM also uses `xu`, `yv` staggers, unified
  in postprocess).
- pyudales — staggered: `xt, yt, zt, xm, ym, zm`. The observation operator
  carries a `dim_mapping` per solver that selects the right axes per
  variable.

Variables are `u, v, w[, pres]`. Ensembles add an `ensemble` dim.

**Parameters** = `xarray.Dataset` of scalar variables: `inflow_angle`
(degrees), `velocity_magnitude` (m/s), optionally the model-error knobs
`vertical_inflow_exponent` and `sgs_constant` (§7), and uDALES-only
`pressure_gradient_magnitude` and `sgs_bias_b0/b1/b2`
(`resolve_parameter_schema` in `hydra_helpers.py` lists them per backend).

For time-varying parameters, vars have a `time` dim. For ensembles, an
`ensemble` dim. Backends detect time-varying via `is_time_varying_params(params)`.

## 5. Configuration system

Hydra composes two entry points in [`configs/`](../configs/): `forward.yaml`
and `assimilation.yaml`. [`configs/README.md`](../configs/README.md) lists every
key and the common overrides; [scripts_and_configs.md](scripts_and_configs.md)
describes the groups, scripts and workflows.

- `common.yaml` owns `paths` (`paths.machine` picks the solver scratch dir per
  machine), the single `ensemble` budget (incl. `failure` policy) and Hydra's
  run dir.
- `case/` bundles domain, grid, geometry, sensors and the per-window `time`
  settings (incl. `seconds_per_knot`).
- `model/` and `params/` contain constructor targets. Forward runs mount once;
  assimilation mounts truth/assimilation models and truth/prior samplers separately.
- `assimilation.yaml` holds every DA block: `assimilation`, `observation`
  (`operator`, `aggregation`, `error`), `smoothing`, `filtering` (the
  `EnsembleKalmanFilter` constructor block itself) and `hybrid`. Each script
  reads the blocks it needs.
- `assimilation_settings/` holds every option of each component at the top
  level (`smoother.*`, `analysis.*`, `localization.*`, `state_reduction.*`,
  `inflation.*`); a slot selects one by interpolation:
  `'smoothing.smoother=${smoother.static}'`.
- `surrogate/` holds the neural-surrogate configs (`--config-name surrogate/<name>`).

Runners keep `run(cfg)` plus a thin `@hydra.main` wrapper. They call
`check_config(cfg, workflow)` from `scripts/utils/inconsistency_check.py` before any
side effect, so a bad combination fails in seconds, then instantiate the
selected components explicitly.

### Tests

The tests compose the real `configs/` entry points made tiny by an overlay
from `tests/configs/` (`compose("forward", "+test=forward", root=tmp_path)`);
every output goes under `root`. Most script tests run on the neural-surrogate
backend trained once per session on synthetic data, so they need no compiled
solver.

The default test command (`pixi run -e dev py.test`) runs fast tests; real CFD
calls are marked `integration` (`test-integration`, `test-all`). See
[`tests/README.md`](../tests/README.md).

## 6. Data assimilation flow

The library side (classes, math, extension recipes) is in
[data_assimilation.md](data_assimilation.md); this section maps it onto the
scripts.

### Observations
- `ObservationOperator` maps a state Dataset to a flat vector of length
  `num_sensors * len(obs_states)`, index-based (`obs_ids_*`) or
  coordinate-based (`obs_*`, interpolated). The configs use coordinate-based:
  `observation.operator` in `configs/assimilation.yaml` reads the case's
  `obs.*` points and `scripts/utils/helper_functions.py::make_observation_operator`
  instantiates it with the model's `solver_name`.
- `TemporalObservationOperator` applies it per output frame.
  `AggregateObservations` optionally bins frames into `interval_seconds`-wide
  intervals (`observation.aggregation`, `null` = off). `observation.error`
  (`ObservationErrorSpec`) builds the physical observation error.

### The assimilation entry points

All share the observation operator, the augmentation layer, the localization
machinery and the analysis math — they differ in *what* is updated and *when*.
All read [configs/assimilation.yaml](../configs/assimilation.yaml):

| Script | Algorithm |
|---|---|
| [scripts/run_smoother.py](../scripts/run_smoother.py) | ESMDA — re-forecast each window `smoothing.num_steps` times with tempered updates. Variant = `smoothing.smoother` (`static`, `dynamic`, `state`, `state_and_parameter`, `state_and_dynamic`). |
| [scripts/run_filtering.py](../scripts/run_filtering.py) | Sequential EnKF — forecast `assimilation.assimilate_every_n_step` frames, then one analysis; windows are chunking only. `filtering.mode=state\|parameter\|joint`. |
| [scripts/run_hybrid.py](../scripts/run_hybrid.py) | Filter smoothing — per window, ESMDA estimates the parameters, then the filter produces the state over the same observations. |

All three write the same per-window layout under
`<paths.results_dir>/<smoother|filtering|hybrid>/`, read by
`scripts/compute_metrics.py <run dir>` (`metrics.yaml`) and
`scripts/visualize_assimilation.py <run dir>` (figures);
`workflows/assimilation_workflow.sh <method>` chains the three stages.

### ESMDA essentials
- `_BaseESMDA` in
  [smoothing/esmda.py](../libs/data-assimilation/src/data_assimilation/smoothing/esmda.py)
  runs `num_steps` iterations of an **iterated joint update**: each iteration
  forecasts from the *current* initial-condition estimate, and the state-bearing
  variants feed the Kalman-updated IC forward. Parameter-only variants keep the
  caller's pinned IC.
- Ensemble failures are applied to the params ensemble between forecast and
  analysis via `apply_failure_substitutions_to_params`.
- Localization (`smoothing.localization` / `filtering.localization`:
  `none`, `correlation`, `distance`) and reduced SVD/KL state updates
  (`smoothing.state_reduction` / `filtering.state_reduction`) are optional and
  mutually exclusive; see the localization and reduced-state sections of
  [data_assimilation.md](data_assimilation.md).

### Multi-window runs
Each DA script loops over `assimilation.num_windows`. The truth (state +
parameters) for every window is simulated up front (or loaded); per window the
loop slices that window's truth observations, adds noise, assimilates, writes
`windows/window_{w}_*` files, and feeds the final posterior state in as the
next window's `state`. A time-varying prior is extrapolated into the next window
(`prior_sampler.extrapolate(...)`); a static one carries the posterior.

### Truth source — inline vs. on disk
Truth is either **simulated inline** (`assimilation.truth_dir: null`, the
default) or **loaded** from a `state.nc`/`params.nc` pair written by
`scripts/run_forward.py`. `assimilation.truth_start_time` begins the
assimilation horizon partway into a disk truth (drops earlier frames and
rebases that time to `t=0`) to skip a spin-up. Disk truth is read lazily, one
window at a time (`make_truth`, `open_truth` in `scripts/utils/helper_functions.py`).

### Validation sensors
A case's `obs` block may define held-out sensors via
`validation_{x,y,z}_points`. They are **scored but never assimilated** —
`scripts/utils/helper_functions.py::sensor_sets` adds them as the `validation`
set, and `compute_metrics.py` / `visualize_assimilation.py` score and plot them.

### Parameter samplers (static + dynamic)
Both kinds are built with `hydra.utils.instantiate(...)` and share one
interface — all configuration at construction, and **`sample(ensemble_size)`**
returns an `xarray.Dataset` with an `ensemble` dim:

```python
params_sampler = instantiate(cfg.params)          # or cfg.truth_params / cfg.prior_params
params = params_sampler.sample(ensemble_size)
```

`assimilation.params_to_estimate` selects the prior fields DA updates (`null`
= all, `[]` = none). Unselected fields are still applied to the model and saved;
unselected dynamic trajectories still extrapolate between windows. Use a
`Constant` to prescribe a value, or remove the entry to use the model default.

- **Static** ([src/pyurbanair/static_parameters/](../src/pyurbanair/static_parameters/)) —
  `ParameterSampler` holds a `name -> Distribution` mapping (`Normal`,
  `Uniform`, `Constant`); `static.yaml` is a prior, `static_truth.yaml` all
  `Constant`s. Output has an `ensemble` dim only.
- **Dynamic** ([src/pyurbanair/dynamic_parameters/](../src/pyurbanair/dynamic_parameters/)) —
  `AR2RelaxationModel` (critically-damped AR(2) relaxing toward an external
  prior; `dynamic.yaml`, `dynamic_truth.yaml`) and `HarmonicParameterModel`
  (smooth truths; `dynamic_sine.yaml`, `dynamic_cosine.yaml`). Output adds a
  `time` dim with a knot every `time.seconds_per_knot` s; `extrapolate(...)`
  continues a posterior into the next window.

A single-member run drops the `ensemble` dim with `.isel(ensemble=0, drop=True)`.

## 7. Backend-specific notes (gotchas)

### pylbm
- On **first import** the Fortran code is located (or cloned as the LBM
  submodule) into `libs/pylbm/LBM/`
  (see [libs/pylbm/src/pylbm/__init__.py](../libs/pylbm/src/pylbm/__init__.py)).
- Compile is gated by `cfg.model.compile` (consumed by
  `pyurbanair.config.hydra_helpers.prepare_compile` via the `model.prepare`
  block). After a rebuild, stale `seed_*.dat[.orig]` files are wiped (they
  break warm starts).
- Runtime configuration lives in `infile.in`, edited via
  `Infile(...).set_value(...)`. `iprt1` is set to disable the every-iteration
  NetCDF dump that otherwise makes warm starts ~20× slower.
- Output `out_0000_F<timestep>.nc` files are concatenated along `time` and
  the spin-up outputs dropped.
- The solver runs with stdout/stderr to `DEVNULL` unless `verbose=True`.
  **To see swallowed errors, override `model.forward_model.verbose=true`.**
  Silent crashes here are the #1 mystery when LBM "produces no output".
- STL → LBM conversion is functional but not fully validated ([pylbm.md](pylbm.md)).

### pyudales
- Preprocessing is Matlab or pure Python (`python_udgeom/`), selected by
  `python_or_matlab` in the `prepare` block of
  [configs/model/pyudales.yaml](../configs/model/pyudales.yaml)
  (`hydra_helpers.prepare_udales`).
- Runtime config in `namoptions.<exp>` (edited via `NamoptionsFile`).
- Staggered grid: state has `xt/xm`, `yt/ym`, `zt/zm`.
- Inflow profile via the nested `nudging_config` on `cfg.model.forward_model`
  (`apply_time_varying_inflow` in
  [utils/nudging_utils.py](../libs/pyudales/src/pyudales/utils/nudging_utils.py)).
- **Instability watchdog**: `run_single` runs uDALES via
  `run_with_dt_watchdog` ([utils/run_monitor.py](../libs/pyudales/src/pyudales/utils/run_monitor.py)).
  If `dt` stays below `min_dt` for `patience` steps it kills the run and raises
  `CalledProcessError` so the ensemble's failure policy can act. Configured by
  `instability_check` on the forward model.
- `params_utils.py` keeps a whitelist (`INFLOW_PARAM_NAMES`) of inlet
  variables that survive `extract_inflow_params` / `merge_params`; a new inlet
  variable not in it is *silently dropped*. Discrepancy coefficients use their
  own extractor/writer (`utils/discrepancy_utils.py`) — don't whitelist them.

### Model-error compensation knobs
- `vertical_inflow_exponent` (power-law shear exponent α) and `sgs_constant`
  let the smoother absorb truth↔assim solver misspecification. Both are static
  scalars consumed per member in each backend's `_apply_inflow_settings`, and
  each write is a no-op when the parameter is absent.
- **Write sites differ per solver** and `sgs_constant` is not the same
  quantity across them:
  - pylbm: α → `uvel_shear.dat`; SGS → the `infile.in` `ivreman smagor` line
    (`apply_sgs_setting`).
  - pyudales: α → nudging `profile_config`; SGS → `&NAMSUBGRID` `cs`
    (Smagorinsky) or `c_vreman` (Vreman), whichever the active closure reads
    (`_apply_sgs_setting`; see [pyudales.md](pyudales.md)).
  - pypalm: α → `profile_config`; SGS → `km_constant` (a constant eddy
    diffusivity in m²/s that replaces the SGS-TKE closure and forces
    `constant_flux_layer = .false.`), so it needs PALM-appropriate prior ranges.

### pypalm
- Lazy-imported. All `pypalm.*` `_target_` blocks live only in
  [configs/model/pypalm.yaml](../configs/model/pypalm.yaml), so non-PALM runs
  never pay the compile cost.
- Postprocess unifies the vertical staggers (`zu_3d → z`, `w`
  interpolated from `zw_3d`) so all three velocity components share one `z`.

### neural_surrogate (learned fourth backend)
Read [neural_surrogates.md](neural_surrogates.md) before editing here. A few
gotchas worth knowing up front:
- Stepper presets are `<family>_<size>` entries in
  [configs/surrogate/architectures.yaml](../configs/surrogate/architectures.yaml)
  (`p3d`, `unet_convnext`, `upt`).
- UPT's `normalize=True` and `predict_residual=True` are load-bearing (see the
  docstring in [architectures/upt.py](../libs/neural-surrogates/src/neural_surrogates/architectures/upt.py));
  normalization stats are baked into the checkpoint via `set_normalization`.
- `spinup_source: generative` (Tadpole latent generator) samples the window-0
  field instead of running a CFD spin-up; contract in
  [neural_surrogates.md](neural_surrogates.md) Part I.

## 8. Adding a new component — recipes

### Add a new forward-model backend
1. Add a sub-library under `libs/<name>/` mirroring pylbm:
   `pyproject.toml`, `src/<name>/forward_model.py`,
   `ensemble_forward_model.py`, `utils/`, optional `__init__.py` that
   pulls the underlying Fortran/C source if needed.
2. Subclass `BaseForwardModel`: `__init__` (call super with `results_dir`),
   `run_single`, `_apply_inflow_settings`, `save_results`, `_clean_output`.
   Launch the solver with `pyurbanair.utils.solver_process.run_solver`, and
   pass `apple_linker_flags` (`pyurbanair.utils.toolchain`) to any native
   build's link step, so it builds and runs on Linux and macOS.
3. Subclass `BaseEnsembleForwardModel`; the only mandatory override is
   `_create_new_forward_model` (clone the template into a per-member directory).
4. Add a Pixi feature in [pyproject.toml](../pyproject.toml).
5. Add `configs/model/<name>.yaml` mirroring
   [configs/model/pylbm.yaml](../configs/model/pylbm.yaml): `name`,
   `solver_name`, `forward_model`, `prepare`, `ensemble_model` with
   `failure: ${ensemble.failure}`. Reuse or add a prepare helper in
   [src/pyurbanair/config/hydra_helpers.py](../src/pyurbanair/config/hydra_helpers.py),
   and add a `clean_outputs` branch there (its `else` arm raises).
6. Add a `dim_mapping` entry in
   [`ObservationOperator`](../libs/data-assimilation/src/data_assimilation/observation_operator.py)
   for the new `solver_name`.
7. Add a tiny overlay `tests/configs/model/<name>_tiny.yaml` and `integration`
   tests in `tests/<name>/`.

### Add a new parameter
- Add it to the sampler configs in [configs/params/](../configs/params/): a
  `Distribution` block in `static.yaml` / `static_truth.yaml` and/or under
  `external_parameters` in `dynamic.yaml` / `dynamic_truth.yaml`. No Python
  change is needed for sampling. A constant-in-time parameter estimated
  alongside a time-varying inflow goes under `static_parameters:` in the
  dynamic configs.
- If backend-specific, extend `resolve_parameter_schema` in
  [hydra_helpers.py](../src/pyurbanair/config/hydra_helpers.py).
- Write it in each backend's `_apply_inflow_settings` (pylbm / pyudales:
  `utils/params_utils.py`). Read it with `get_param_value(params, name)` and
  **no-op when it is absent** so default runs stay byte-identical. For uDALES
  inlet parameters, add it to `INFLOW_PARAM_NAMES` (§7).
- Select it for estimation with `assimilation.params_to_estimate`.

### Add a new ESMDA variant, localization strategy or filter analysis
Follow the extension recipes in [data_assimilation.md](data_assimilation.md): add the
class, add an entry to the matching `configs/assimilation_settings/*.yaml`, and
teach `scripts/utils/inconsistency_check.py` its rules. No new script is needed;
select it with `'smoothing.smoother=${smoother.<name>}'` (or the matching slot).

### Add a new run script
- Place under [scripts/](../scripts/), mirror an existing one: `def run(cfg)` +
  a thin `@hydra.main(config_path="../configs", ...)` `main`; post-processing
  scripts are plain CLIs taking a run dir. Document usage and outputs in the
  module docstring.
- Call `check_config(cfg, "<workflow>")` from `scripts/utils/inconsistency_check.py`
  first (add the workflow's rules there).
- Share only what several scripts need via `scripts/utils/helper_functions.py`.
- Write under `cfg.paths.results_dir` and save the composed `config.yaml` there.
- Add a test under `tests/scripts/` that composes the entry point with a
  `tests/configs/` overlay and calls `run(cfg)`; add a `workflows/` chain if useful.

## 9. Operational defaults / scaling

- `.temp/` is the default scratch directory. `paths.results_root` defaults to
  `.temp`; `forward.yaml` sets `results_dir` to `<results_root>/${model.name}`,
  `assimilation.yaml` to `<results_root>/${truth_model.name}_to_${assim_model.name}`,
  and each DA script appends its workflow name.
- Pre-commit hooks (`black`, `isort`, `mypy`): `pixi run -e dev pre-commit`.
  They are **not enforced** server-side.
- **Ensemble scaling** is DRAM-bandwidth-bound past ~4–8 workers (archived
  findings: [archive/ensemble_scaling.md](archive/ensemble_scaling.md)). Don't
  raise `ensemble.num_parallel_processes` past 8 without re-benchmarking.

## 10. Things that look optional but aren't

- Every state is expected to have a `time` dim, even when length 1 (the
  observation operators assume it).
- State-bearing ESMDA variants warm-start each iteration's forecast from the
  analyzed IC, skipping a fresh spin-up, so the analyzed field must be a usable
  warm-start state for the forward model.
- `BaseEnsembleForwardModel._set_save_mode` controls whether the ensemble
  result is concatenated or written to disk. Member `results_dir`s are
  overridden to match the ensemble's at dispatch — don't bake per-member paths
  into ensemble code.

## 11. Fastest path to "where is X?"

| You want to change… | Look here |
|---|---|
| Run-time settings (sim time, ensemble size, ESMDA steps) | [configs/README.md](../configs/README.md) |
| Geometry / domain / sensor layout | [configs/case/](../configs/case/) + [geometries/](../geometries/) |
| Per-backend wiring | [configs/model/](../configs/model/) |
| Prepare / clean helpers | [src/pyurbanair/config/hydra_helpers.py](../src/pyurbanair/config/hydra_helpers.py) |
| What an ensemble member does in parallel | [src/pyurbanair/base_ensemble_forward_model.py](../src/pyurbanair/base_ensemble_forward_model.py) |
| How a solver consumes params | `libs/<solver>/src/<solver>/forward_model.py` (`_apply_inflow_settings`), pylbm/pyudales `utils/params_utils.py` |
| ESMDA Kalman update / variants | [smoothing/esmda.py](../libs/data-assimilation/src/data_assimilation/smoothing/esmda.py), [configs/assimilation_settings/smoother.yaml](../configs/assimilation_settings/smoother.yaml) |
| Filter / hybrid | [filtering/](../libs/data-assimilation/src/data_assimilation/filtering/), [filter_smoothing/](../libs/data-assimilation/src/data_assimilation/filter_smoothing/) |
| Config consistency rules | [scripts/utils/inconsistency_check.py](../scripts/utils/inconsistency_check.py) |
| How sensors map to grid points | [observation_operator.py](../libs/data-assimilation/src/data_assimilation/observation_operator.py) |
| Per-window logic | the window loops of `run_smoother.py` / `run_filtering.py` / `run_hybrid.py`; `run_forward.py`'s `forward.rollout_steps` loop |
| Parameter samplers | [static_parameters/](../src/pyurbanair/static_parameters/), [dynamic_parameters/](../src/pyurbanair/dynamic_parameters/), [configs/params/](../configs/params/) |
| Truth source / spin-up skip | `assimilation.truth_dir`, `assimilation.truth_start_time`; `make_truth`, `open_truth` in [helper_functions.py](../scripts/utils/helper_functions.py) |
| DA metrics + figures | [libs/evaluation/](../libs/evaluation/src/evaluation/), [compute_metrics.py](../scripts/compute_metrics.py), [visualize_assimilation.py](../scripts/visualize_assimilation.py) |
| uDALES dt-collapse handling | [run_monitor.py](../libs/pyudales/src/pyudales/utils/run_monitor.py) |
| Neural-surrogate architectures | [architectures/](../libs/neural-surrogates/src/neural_surrogates/architectures/), [configs/surrogate/architectures.yaml](../configs/surrogate/architectures.yaml) |
| Test fixture composition | [tests/conftest.py](../tests/conftest.py) (`compose` + `tests/configs/` overlays) |
