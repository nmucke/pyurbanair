# pyurbanair — Codebase Guide (for AI coding assistants)

This is a fast-orientation sheet aimed at LLM coding tools. The user-facing
[`README.md`](../README.md) covers install / usage. This sheet covers
**internal structure, contracts, and conventions** so an assistant can land
non-trivial edits without re-deriving them.

## 1. What this repo is

`pyurbanair` is a Python monorepo for urban-airflow CFD ensembles and
ensemble data assimilation (ESMDA). It wraps three Fortran CFD solvers
behind a common Python interface and runs them in ensembles for parameter /
state estimation.

- **Three CFD backends**, each in [libs/](../libs/) as an editable subpackage:
  - `pylbm`  — Lattice Boltzmann (Geir Evensen). STL geometry. Optional CUDA.
  - `pyudales` — uDALES v2.2.0. Staggered grid; Matlab or Python preprocessing.
  - `pypalm` — PALM model system. Imports lazily (compiles on first import).
- **`data-assimilation`** library implements ESMDA in JAX.
- **`pyurbanair`** (top-level package) holds the base classes that *every*
  backend's forward / ensemble / rollout model inherits from. Polymorphism
  is via these base classes — ESMDA never depends on a specific solver.
- All public I/O is `xarray.Dataset`. On-disk format is NetCDF.

## 2. Monorepo layout

```
src/pyurbanair/                    # Top-level package: base classes + glue
  base_forward_model.py            # BaseForwardModel
  base_ensemble_forward_model.py   # BaseEnsembleForwardModel (parallel/seq, failure policy)
  base_rollout_forward_model.py    # BaseRolloutForwardModel (multi-window state carry-over)
  parameter_time_series/           # Time-varying parameter priors (AR1/AR2/OU/GP)
  utils/
    config_utils.py                # Factories: create_forward_model, create_ensemble_*, etc.
    cpu_pinning.py                 # Worker → CPU pinning for parallel ensembles
    run_utils.py, state_utils.py, animation_utils.py, da_metrics.py
  plotting.py, animation.py

libs/data-assimilation/src/data_assimilation/
  observation_operator.py          # ObservationOperator + TemporalObservationOperator
  interpolation.py                 # Grid → sensor-point interpolation
  smoothing/
    base.py                        # BaseSmoothing — _forecast_step, _observation_step
    esmda.py                       # ParameterESMDA, StateAndParameterESMDA, TimeVaryingParameterESMDA

libs/pylbm/src/pylbm/              # LBM wrapper. __init__ git-clones the LBM Fortran code.
  forward_model.py                 # ForwardModel(BaseForwardModel)
  ensemble_forward_model.py        # EnsembleForwardModel(BaseEnsembleForwardModel)
  stl_to_lbm.py                    # STL → LBM voxel geometry
  utils/                           # infile.in editing, compile, warm-start, params, ...

libs/pyudales/src/pyudales/        # uDALES wrapper. Similar shape to pylbm.
  forward_model.py, ensemble_forward_model.py
  python_udgeom/                   # Python preprocessing alternative to Matlab
  utils/                           # namoptions, nudging, ncpu, warm-start, etc.

libs/pypalm/src/pypalm/            # PALM wrapper. Similar shape.

scripts/                           # All top-level executables run from here.
  config.py                        # SINGLE source of truth for run-time configuration
  config_small.py                  # Cheaper config preset (for tests / quick runs)
  run_forward_model.py             # Single forward sim
  run_ensemble_forward_model.py    # Ensemble forward sim
  run_rollout_forward_model.py     # Multi-window rollout (state carries between windows)
  run_ensemble_rollout_forward_model.py
  run_parameter_esmda.py           # Parameter-only ESMDA
  run_state_and_parameter_esmda.py # Joint state+parameter ESMDA
  run_rollout_esmda.py             # Multi-window joint ESMDA
  run_time_varying_*.py            # Time-varying inflow params variants
  benchmark_*_ensemble_scaling.py  # Throughput benchmarks (see docs/ensemble_scaling.md)

examples/
  benchmark_geometry/              # Xie & Castro 2008 geometry generator (CLI)
  lbm/, udales/, palm/             # Per-backend experiment dirs (STL, namoptions, p3d, etc.)

tests/                             # pytest suite. tests/config.py = small/fast config used by tests.
.temp/                             # Default scratch dir. Everything mutable lands here.
```

## 3. The core abstraction — forward models

All three solvers conform to the same three-class shape, declared in
[src/pyurbanair/](../src/pyurbanair/) and inherited by each backend.

### `BaseForwardModel` — single simulation
- File: [src/pyurbanair/base_forward_model.py](../src/pyurbanair/base_forward_model.py)
- Subclasses must implement `run_single`, `_apply_inflow_settings`,
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
- **Failure policy** (`configure_failure_policy`):
  - `"raise"` (default) — first failure aborts the whole ensemble.
  - `"resample_from_successes"` — failed members are cloned from a random
    successful donor; the *params* ensemble can be re-cloned (with
    Gaussian jitter) by calling
    `apply_failure_substitutions_to_params(params)`.
  - On-disk parallel runs **do not** support resample — they raise.
- **CPU pinning**: parallel runs pin workers to distinct cores via
  [src/pyurbanair/utils/cpu_pinning.py](../src/pyurbanair/utils/cpu_pinning.py).
  Disable with `PYURBANAIR_DISABLE_CPU_PINNING=1`.
- mp context is **forkserver**, not fork, because JAX starts background
  threads at import.

### `BaseRolloutForwardModel` — multi-window simulations
- File: [src/pyurbanair/base_rollout_forward_model.py](../src/pyurbanair/base_rollout_forward_model.py)
- Wraps a `BaseForwardModel` and runs it repeatedly, feeding each
  window's final state into the next as a warm start.
- `rollout_step` is auto-incremented per call.
- If `spinup_first_step_only=True`, calls `forward_model.disable_spinup()`
  after step 0 so only the cold start pays the spinup cost.
- Subclasses implement `_pre_run_rollout_step` / `_post_run_rollout_step`.

> When the README and `config_utils.create_rollout_forward_model` say
> "rollout forward model", note that the current default returns the
> underlying `forward_model` unchanged — multi-window driving is now
> handled in the scripts (e.g. `run_rollout_esmda.py`) by repeatedly
> invoking the model with state carry-over.

## 4. Data contracts

**State** = `xarray.Dataset` with at least a `time` dimension. Grid axes
depend on backend:
- pylbm / pypalm — `x, y, z` (PALM also uses `xu`, `yv` staggers, unified
  in postprocess).
- pyudales — staggered: `xt, yt, zt, xm, ym, zm`. The observation operator
  carries a `dim_mapping` per solver that selects the right axes per
  variable.

Variables are `u, v, w[, pres]`. Ensembles add an `ensemble` dim.

**Parameters** = `xarray.Dataset` with up to three scalar variables:
- `inflow_angle` (degrees)
- `velocity_magnitude` (m/s)
- `pressure_gradient_magnitude` — **uDALES-only**

For time-varying parameters, vars have a `time` dim. For ensembles, an
`ensemble` dim. Backends detect time-varying via
`is_time_varying_params(params)` in each backend's `utils/params_utils.py`.

## 5. Configuration system

[scripts/config.py](../scripts/config.py) is the **single source of truth**
for all runs. It is a plain Python module with dict-typed constants:
`DOMAIN, TIME, LBM_ARGS, UDALES_ARGS, PALM_ARGS, ENSEMBLE, OBS, ESMDA,
TRUE_PARAMS, PARAM_PRIORS, EXTERNAL_PRIORS, TIME_VARYING_PARAMS`.

All run scripts import this module and call the factory functions
re-exported from
[src/pyurbanair/utils/config_utils.py](../src/pyurbanair/utils/config_utils.py):

```python
forward_model        = config.create_forward_model(model_name, results_dir=None)
config.prepare_forward_model(model_name, forward_model)   # compile / preprocess
ensemble_model       = config.create_ensemble_forward_model(model_name, fm)
true_params          = config.create_true_params(model_name)
params_ensemble      = config.create_parameter_ensemble(model_name)
obs_op               = config.create_observation_operator(model_name)
C_D                  = config.create_C_D(num_obs)
```

`pypalm` is **lazily imported** inside the `model_name == "pypalm"`
branches — its `__init__` compiles the solver, which is too expensive for
runs that never instantiate a PALM model.

Tests use [tests/config.py](../tests/config.py) (smaller domain / shorter
simulation_time). [scripts/config_small.py](../scripts/config_small.py) is
a manual quick-run preset.

## 6. Data assimilation flow

### `ObservationOperator` (data-assimilation lib)
- Maps a state Dataset to a flat observation vector of length
  `num_sensors * len(obs_states)`.
- Two construction modes: **index-based** (`obs_ids_*`) or
  **coordinate-based** (`obs_*`, interpolated). config.py currently uses
  coordinate-based.
- Variable→dim mapping handles each backend's staggered grids.
- `TemporalObservationOperator` wraps it with time aggregation:
  `mean | median | max | min | full | intervals`. `intervals` is the
  config default — observations are aggregated in chunks of
  `interval_size` time steps.

### ESMDA
- `BaseSmoothing` ([libs/data-assimilation/src/data_assimilation/smoothing/base.py](../libs/data-assimilation/src/data_assimilation/smoothing/base.py))
  provides `_forecast_step` (runs the ensemble) and `_observation_step`
  (applies the observation operator).
- `_BaseESMDA` provides the shared Kalman update
  (`_compute_kalman_update`) and the `_analysis` loop that drives
  `num_steps` iterations with the same `initial_state` (pinned to avoid
  spin-up bypass across iterations).
- Three variants:
  - `ParameterESMDA` — augmented state is just parameters.
  - `StateAndParameterESMDA` — augmented state concatenates flattened state
    and parameters; output state is unflattened back to xarray.
  - `TimeVaryingParameterESMDA` — flattens each `(time, ensemble)`
    parameter into `{name}_{t}` scalars before update, then unflattens.
- On-disk mode: each ESMDA step has its own subdirectory
  `step_{i}/state_*.nc`; `get_state(step, ensemble_member)` re-opens them.
- Ensemble failures recorded by the underlying ensemble model are
  applied to the params ensemble between forecast and analysis via
  `apply_failure_substitutions_to_params`.

### Multi-window rollout ESMDA
See [scripts/run_rollout_esmda.py](../scripts/run_rollout_esmda.py): outer
loop over `num_assimilation_windows`; in each window the truth model
forecasts one step and ESMDA updates state+params for the assimilation
model. The window's posterior state is fed in as the next window's
initial state.

### Time-varying parameter priors
[src/pyurbanair/parameter_time_series/](../src/pyurbanair/parameter_time_series/) —
four classes, all subclassing `ParameterTimeSeries` and registered in
`_REGISTRY`:
- `gp_linear_trend` — RBF GP prior; linear-trend + GP residual extrapolation.
- `ar1` — stationary AR(1) prior; deterministic roll-forward.
- `ornstein_uhlenbeck` — OU process; stochastic roll-forward.
- `ar2_relaxation` — critically-damped AR(2); relaxation toward external
  prior `x_ext`. **This is the config default.**

Each exposes `sample_prior(time_coords, rng_key)` (initial window) and
`extrapolate(posterior, prediction_times, rng_key)` (next window).
Construct via `build_parameter_time_series(method, external_priors,
ensemble_size, method_kwargs)`.

## 7. Backend-specific notes (gotchas)

### pylbm
- On **first import** the Fortran code is fetched as a git submodule into
  `libs/pylbm/LBM/` (see [libs/pylbm/src/pylbm/__init__.py](../libs/pylbm/src/pylbm/__init__.py)).
  No network access ⇒ it will silently fall back.
- Compile is gated by `LBM_ARGS["compile"]` and called from
  `prepare_forward_model`. After a rebuild, stale `seed_*.dat[.orig]`
  files are wiped (they break warm starts).
- Runtime configuration lives in `infile.in`. The wrapper edits keys via
  `Infile(...).set_value(...)`. `iprt1` is set to disable the
  every-iteration NetCDF dump that otherwise makes warm starts ~20× slower.
- Output is `out_0000_F<timestep>.nc`. They are concatenated along `time`
  and trimmed to `simulation_time / output_frequency` outputs (the
  spinup_outputs prefix is dropped).
- Failures surface as `subprocess.CalledProcessError`. **To see swallowed
  errors, flip `verbose=True` in `LBM_ARGS`.**
- STL → LBM geometry conversion is implemented but not fully trusted (per
  README caveat).

### pyudales
- Uses Matlab for default preprocessing
  (`UDALES_ARGS["matlab_bin"]`). A pure-Python preprocessor exists in
  `python_udgeom/` and is selected by passing
  `python_or_matlab="python"` to `run_preprocessing`. `prepare_forward_model`
  currently always uses Python.
- Runtime config in `namoptions.<exp>` (edited via `NamoptionsFile`).
- Staggered grid: state has `xt/xm`, `yt/ym`, `zt/zm`. Some plotting
  utilities call `interpolate_grid` to project everything onto a common
  grid before display.
- Inflow profile is configured via the nested `nudging_config` dict
  (`profile_config = {"type": "uniform" | "power_law", "alpha": ...,
  "z_ref": ...}`).
- `pressure_gradient_magnitude` is the third parameter only this backend
  supports.

### pypalm
- Lazy-imported. Same factory pattern as the others but lives behind
  `if model_name == "pypalm": from pypalm... import ...` in
  `config_utils.py` so non-PALM runs never pay its compile cost.
- Postprocess unifies the vertical staggers (`zu_3d → z`, `w`
  interpolated from `zw_3d`) so all three velocity components share a
  single `z` dim.

## 8. Adding a new component — recipes

### Add a new forward-model backend
1. Add a new sub-library under `libs/<name>/` mirroring the pylbm shape:
   `pyproject.toml`, `src/<name>/forward_model.py`,
   `ensemble_forward_model.py`, `utils/`, optional `__init__.py` that
   pulls the underlying Fortran/C source if needed.
2. Subclass `BaseForwardModel`. Implement: `__init__` (call super with
   `results_dir`), `run_single`, `_apply_inflow_settings`,
   `save_results`, `_clean_output`. The base class handles save mode and
   `__call__`.
3. Subclass `BaseEnsembleForwardModel`. Only mandatory override is
   `_create_new_forward_model` (clone the template into a per-member
   directory).
4. Add a Pixi feature in [pyproject.toml](../pyproject.toml) (system
   deps + pypi dependency on the new lib).
5. Add branches to every helper in
   [src/pyurbanair/utils/config_utils.py](../src/pyurbanair/utils/config_utils.py)
   (`model_args`, `create_forward_model`, `prepare_forward_model`,
   `clean_forward_model_outputs`, `create_ensemble_forward_model`,
   `solver_name`).
6. Add the solver name to `ModelName` and to each script's
   `argparse --model choices=[...]` list.
7. Add a `dim_mapping` entry in
   [`ObservationOperator.__init__`](../libs/data-assimilation/src/data_assimilation/observation_operator.py)
   for the new `solver_name`.

### Add a new parameter
- New `data_vars` entry to `TRUE_PARAMS` and matching prior in
  `PARAM_PRIORS` and (if relevant) `EXTERNAL_PRIORS` in
  [scripts/config.py](../scripts/config.py).
- Extend `create_true_params` and `create_parameter_ensemble` in
  [config_utils.py](../src/pyurbanair/utils/config_utils.py).
- Extend each backend's `_apply_inflow_settings` /
  `apply_inflow_settings` in `utils/params_utils.py` so the value gets
  written to that solver's input format.
- Time-varying support: implement reading in the backend (e.g. pylbm's
  `write_uvel_time_file`).

### Add a new ESMDA variant
- Subclass `_BaseESMDA` in
  [smoothing/esmda.py](../libs/data-assimilation/src/data_assimilation/smoothing/esmda.py).
- Override `_one_step(params, obs, state)` — choose what's in the
  augmented vector, call `self._compute_kalman_update(...)`, return
  `(updated_state_or_None, updated_params)`.
- Add a new top-level `scripts/run_<name>.py`. Pattern: build truth
  model, draw truth obs, build assim ensemble + obs op + C_D, instantiate
  the smoother, call it, dump results.

### Add a new run script
- Place under [scripts/](../scripts/), mirror an existing one — `argparse`
  with `--model` (or `--truth-model`/`--assim-model`), pull all config
  from `from scripts import config`, use the `config.create_*` factories.
- Output to `config.BASE_RESULTS_DIR / "<script_name>"`.
- If interactive plots: gate with `--skip-viz`.

## 9. Operational defaults / scaling

- `.temp/` is the default scratch directory. Every backend writes its
  per-experiment dir and per-member dirs underneath.
- Tests use a small domain (see `tests/config.py`) — `pixi run py.test`
  in the dev env.
- Pre-commit hooks (`black`, `isort`, `mypy`) installed via
  `pixi run pre-commit`. They are **not enforced** server-side; commits
  can bypass.
- **Ensemble scaling on this hardware** is DRAM-bandwidth-bound past
  ~4 workers (see [docs/ensemble_scaling.md](ensemble_scaling.md) and
  the comment in `ENSEMBLE` of `config.py`). Don't blindly raise
  `num_parallel_processes` past 8 — re-benchmark first.
- `pyurbanair` deliberately uses `forkserver` not `fork` for parallel
  workers (JAX threads + bare-fork = deadlock).

## 10. Things that look optional but aren't

- Every state is expected to have a `time` dim, even when length 1. Some
  helpers (`extract_2d_slice`, observation operators) assume this.
- ESMDA `_analysis` pins the caller-provided initial state for every
  inner iteration — do **not** "optimize" this by feeding the previous
  iteration's output forward; it would bypass spin-up after the first
  iter and tangle iterates.
- `BaseEnsembleForwardModel._set_save_mode` controls whether the
  ensemble result is concatenated or written to disk. Forward-model
  child `results_dir` is overridden to match the ensemble's at parallel
  dispatch time — don't bake per-member paths into ensemble code.
- `pylbm` runs `subprocess` with `stderr=DEVNULL` unless `verbose=True`.
  Silent crashes here are the #1 mystery when LBM "produces no output".

## 11. Fastest path to "where is X?"

| You want to change… | Look here |
|---|---|
| Run-time parameters (domain, sim time, ensemble size) | [scripts/config.py](../scripts/config.py) |
| What an ensemble member does in parallel | [src/pyurbanair/base_ensemble_forward_model.py](../src/pyurbanair/base_ensemble_forward_model.py) |
| How a solver consumes params | `libs/<solver>/src/<solver>/utils/params_utils.py` |
| ESMDA Kalman update | [libs/data-assimilation/src/data_assimilation/smoothing/esmda.py](../libs/data-assimilation/src/data_assimilation/smoothing/esmda.py) |
| How sensors map to grid points | [libs/data-assimilation/src/data_assimilation/observation_operator.py](../libs/data-assimilation/src/data_assimilation/observation_operator.py) |
| Factory wiring (model_name → class) | [src/pyurbanair/utils/config_utils.py](../src/pyurbanair/utils/config_utils.py) |
| Per-window rollout logic | [src/pyurbanair/base_rollout_forward_model.py](../src/pyurbanair/base_rollout_forward_model.py) and `scripts/run_rollout_*.py` |
| Time-varying parameter prior | [src/pyurbanair/parameter_time_series/](../src/pyurbanair/parameter_time_series/) |
| Benchmark / scaling experiments | `scripts/benchmark_*_ensemble_scaling.py`, [docs/ensemble_scaling.md](ensemble_scaling.md) |
