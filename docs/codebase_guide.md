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
  config/
    hydra_helpers.py               # Targets that Hydra `_target_` blocks instantiate
                                   #   (prepare_*, create_*, configure_failure_policy,
                                   #    build_truth_ts_model, resolve_output_dir, ...)
  utils/
    cpu_pinning.py                 # Worker → CPU pinning for parallel ensembles
    run_utils.py, state_utils.py, animation_utils.py, da_metrics.py
  plotting.py, animation.py

conf/                              # Hydra config (see §5 Configuration system)
  config.yaml                      # Base composition + `run:` namespace
  domain.yaml, time.yaml, ensemble.yaml, obs.yaml, esmda.yaml,
  parameters.yaml, paths.yaml      # One flat file per parameter category
  run_*_esmda.yaml                 # Per-script primary configs (smoother wiring)
  model/, size/, preset/, training_data/, neural_surrogate_*/   # remaining groups/overlays

libs/data-assimilation/src/data_assimilation/
  observation_operator.py          # ObservationOperator + TemporalObservationOperator
  interpolation.py                 # Grid → sensor-point interpolation
  localization/
    base.py                        # BaseLocalization — inflation_factors + localized_update
    correlation.py                 # CorrelationLocalization (adaptive correlation-based)
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
                                   # Each exposes `def run(cfg)` + a thin `@hydra.main` wrapper.
  run_forward_model.py             # Single forward sim
  run_ensemble_forward_model.py    # Ensemble forward sim
  run_rollout_forward_model.py     # Multi-window rollout (state carries between windows)
  run_ensemble_rollout_forward_model.py
  run_time_varying_forward_model.py
  run_parameter_esmda.py           # Parameter-only ESMDA
  run_state_and_parameter_esmda.py # Joint state+parameter ESMDA
  run_rollout_esmda.py             # Multi-window joint ESMDA
  run_time_varying_parameter_esmda.py
  run_time_varying_parameters_rollout_esmda.py
  benchmark_*_ensemble_scaling.py  # Throughput benchmarks (see docs/ensemble_scaling.md)

examples/
  benchmark_geometry/              # Xie & Castro 2008 geometry generator (CLI)
  lbm/, udales/, palm/             # Per-backend experiment dirs (STL, namoptions, p3d, etc.)

tests/                             # pytest suite. tests/conftest.py provides
                                   # `compose_test_cfg` / `compose_module_cfg` fixtures
                                   # that compose `preset=test` + per-test overrides.
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

> The legacy `BaseRolloutForwardModel` is now effectively unused at
> runtime — the explicit `create_rollout_forward_model` factory was
> removed during the Hydra migration. Multi-window driving is handled
> directly in the scripts (e.g. `run_rollout_esmda.py`) by repeatedly
> invoking the forward model with state carry-over.

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

Run-time configuration is a [Hydra](https://hydra.cc/) tree rooted at
[`conf/`](../conf/). The base [`conf/config.yaml`](../conf/config.yaml)
composes **one flat file per parameter category** (not a group of per-variant
files) and bundles a `run:` namespace for generic script-behavior knobs
(`skip_viz`, `results_dir`, `num_steps`, `use_true_params`).

Each flat file uses a `# @package <category>` directive (or `_global_` for
`parameters`) so its body lands at the right runtime key. Each holds the full
set of fields the most involved run case —
[`scripts/run_time_varying_parameters_rollout_esmda.py`](../scripts/run_time_varying_parameters_rollout_esmda.py)
— needs; every other `run_*` script reads a subset.

| File | Runtime key | Notable fields |
|---|---|---|
| [`paths.yaml`](../conf/paths.yaml) | `paths` | `base_results_dir` |
| [`domain.yaml`](../conf/domain.yaml) | `domain` | `nx`, `ny`, `nz`, `bounds` |
| [`time.yaml`](../conf/time.yaml) | `time` | `simulation_time`, `output_frequency`, `spinup_time` |
| [`ensemble.yaml`](../conf/ensemble.yaml) | `ensemble` | `ensemble_size`, `num_parallel_processes`, `failure.{policy, jitter_scale, seed}` |
| [`obs.yaml`](../conf/obs.yaml) | `obs` | `mode: points|grid`, sensor coords, `states`, `temporal_mode`, `interval_size` |
| [`esmda.yaml`](../conf/esmda.yaml) | `esmda` | `num_steps`, `alpha`, `num_assimilation_windows`, `obs_error_std`, `seed`, `localization` (correlation block; `null` = global) |
| [`parameters.yaml`](../conf/parameters.yaml) | `params.*` + `time_varying.*` | per-parameter `{mean, std, min?, max?}`; `method`, `method_kwargs.*`, `truth_method*`, `prior_model._target_` |

Groups/overlays that remain (each option is structurally distinct, not just a
scale):

| Group | Selects | Notable |
|---|---|---|
| `model/` | forward + ensemble backend | `model@truth_model=pylbm model@assim_model=pyudales` |
| `size/` | run-size overlay (`tiny`→`xlarge`) | `# @package _global_`; deep-merges the flat files |
| `preset/` | bundled overlays (`small`, `test`) | smaller domain / fewer steps / CPU-only LBM |
| `training_data/` | data-generation overlay | inlines its own `domain`/`time` + a parameter sampler |

**ESMDA smoother** is the one genuinely per-script piece, so it is *not* in
`esmda.yaml`. Each esmda script has its own primary config
([`conf/run_parameter_esmda.yaml`](../conf/run_parameter_esmda.yaml),
`run_state_and_parameter_esmda.yaml`, `run_rollout_esmda.yaml`,
`run_time_varying_parameter_esmda.yaml`,
[`time_varying_rollout_esmda.yaml`](../conf/time_varying_rollout_esmda.yaml))
that does `defaults: [config]` and supplies `esmda.smoother._target_`; the
script's `@hydra.main` uses that `config_name`. Tests compose the matching
`config_name` (see [tests/conftest.py](../tests/conftest.py)).

**Localization** lives in the esmda namespace: `esmda.localization` is the
adaptive correlation-localization block (Vossepoel et al. 2025) and is applied
by every smoother (`localization: ${..localization}`). Run the global,
unlocalized update with `esmda.localization=null`.

The `size/` overlays are `# @package _global_` and are the single place a run
is sized (`size=medium`). Each inlines `domain`/`time` and overrides only the
fields that scale with the run (`ensemble.ensemble_size`,
`ensemble.num_parallel_processes`, the `obs` sensor coords + `interval_size`,
`esmda.num_steps`/`num_assimilation_windows`, `time_varying.num_time_points`),
deep-merging over the flat base files.

Single-model scripts mount the model under `cfg.model.*`. Assimilation
scripts use Hydra's package-override syntax to mount the same `model/`
group **twice**, once as the truth model and once as the assim model:

```bash
python scripts/run_parameter_esmda.py \
  model@truth_model=pylbm model@assim_model=pyudales
```

Inside the YAMLs, sibling-relative interpolation (`${.foo}`, `${..foo}`)
is used wherever the surrounding group might be re-mounted under
another package; absolute interpolation (`${time.simulation_time}`) is
reserved for cross-group lookups.

### Instantiation vs. helpers

Backend object construction is **declarative** — every forward model,
ensemble model, ESMDA smoother, and time-varying prior model is built
by `hydra.utils.instantiate(cfg.<group>.<target>, ...)` against the
`_target_` block in YAML. The scripts also call a small set of helpers
from [src/pyurbanair/config/hydra_helpers.py](../src/pyurbanair/config/hydra_helpers.py)
for the procedural pieces:

```python
forward_model   = instantiate(cfg.model.forward_model, results_dir=...)
instantiate(cfg.model.prepare, forward_model=forward_model)   # compile / preprocess
ensemble_model  = instantiate(cfg.model.ensemble_model, forward_model=forward_model)
configure_failure_policy(ensemble_model, cfg.ensemble.failure)

true_params     = create_true_params(cfg.model.name, cfg.params.true)
params_ensemble = create_parameter_ensemble(
    model_name=cfg.model.name, prior_cfg=cfg.params.prior,
    ensemble_size=cfg.ensemble.ensemble_size, seed=cfg.esmda.seed,
)
obs_op          = create_observation_operator(cfg.obs, cfg.model.solver_name)
C_D             = create_C_D(num_obs, cfg.esmda.obs_error_std)

esmda           = instantiate(cfg.esmda.smoother, ..., rng_key=rng_key)
```

`pypalm` stays **lazy** because its `_target_` blocks only appear inside
[conf/model/pypalm.yaml](../conf/model/pypalm.yaml); composing a config
with `model=pylbm` never imports `pypalm`. This is asserted by a
regression test in [tests/test_hydra_config.py](../tests/test_hydra_config.py).

### Script structure

Every script in [`scripts/`](../scripts/) follows the same shape:

```python
def run(cfg: DictConfig) -> None:
    ...

@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    run(cfg)
```

`run(cfg)` is the **testable entry point** — tests compose a
`DictConfig` and call `run(cfg)` directly without going through Hydra's
CLI. `main` is just the CLI wrapper.

### Tests

[tests/conftest.py](../tests/conftest.py) exposes two fixtures, both
returning the same composer callable. Pick by fixture scope, not by
behavior:

- **`compose_test_cfg`** (function-scoped): use this in every ordinary
  test that calls `run(cfg)` once. Each `compose(...)` call opens and
  closes a `GlobalHydra`, so the function-scoped lifecycle is fine.
- **`compose_module_cfg`** (module-scoped): use only when a
  module-scoped fixture depends on a composed config (e.g. `pylbm_cfg`
  → `pylbm_model` in the sign / velocity-grid tests, where compiling
  pylbm once per module is the whole point). A function-scoped
  composer can't be invoked from a module-scoped fixture without
  pytest erroring on the scope mismatch.

The `preset=test` overlay sets a small domain / short simulation time /
CPU-only LBM / 4-member ensemble. Override anything per-test:

```python
def test_something(compose_test_cfg) -> None:
    # esmda tests pass the script's own primary config via config_name; the
    # smoother comes from there, not an `esmda=<variant>` selector.
    cfg = compose_test_cfg(
        ["model=pyudales", "esmda.num_steps=1"],
        config_name="run_parameter_esmda",
    )
    run(cfg)
```

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

### Localization (optional)
- [localization/base.py](../libs/data-assimilation/src/data_assimilation/localization/base.py)
  defines `BaseLocalization`. Subclasses implement one method,
  `inflation_factors(aug_dev, pred_obs_dev) -> (N_aug, N_d)`, returning a
  per-(state-row, observation) observation-error inflation factor (`1.0` =
  keep, `>1` = taper, `inf` = exclude). The shared local-analysis math lives
  in `localized_update`, which updates each augmented row with only its
  relevant observations (Vossepoel et al. 2025, MWR-D-24-0269.1).
- `CorrelationLocalization`
  ([localization/correlation.py](../libs/data-assimilation/src/data_assimilation/localization/correlation.py))
  selects observations by ensemble correlation: exclude `|ρ| < ρ_t`, taper the
  rest by correlation distance. Needs no spatial coordinates, so it works for
  both abstract parameter rows and gridded state rows.
- `_BaseESMDA` takes an optional `localization=` arg. When `None`,
  `_compute_kalman_update` does the original global update unchanged; when set,
  it delegates to `localization.localized_update(...)`. The hook is in the
  shared base, so **all** variants (Parameter / TimeVaryingParameter /
  StateAndParameter) get it. Configured via `esmda.localization` in
  [conf/esmda.yaml](../conf/esmda.yaml) (the correlation block, on by default);
  every smoother YAML wires it through with `localization: ${..localization}`,
  and `esmda.localization=null` selects the global update. No script changes
  needed.
- **Cost note**: `localized_update` is `jax.vmap` over augmented rows
  (`N_aug` small `N_d×N_d` solves). Cheap for parameter variants; for
  `StateAndParameterESMDA` (large `N_aug`) it is `O(N_aug·N_d²)` memory — the
  paper's grid-block transition-matrix reuse (§3b) is a documented future
  optimization, not yet implemented.

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
- Compile is gated by `cfg.model.compile` (consumed by
  `pyurbanair.config.hydra_helpers.prepare_compile` via the
  `model.prepare._target_` instantiation). After a rebuild, stale
  `seed_*.dat[.orig]` files are wiped (they break warm starts).
- Runtime configuration lives in `infile.in`. The wrapper edits keys via
  `Infile(...).set_value(...)`. `iprt1` is set to disable the
  every-iteration NetCDF dump that otherwise makes warm starts ~20× slower.
- Output is `out_0000_F<timestep>.nc`. They are concatenated along `time`
  and trimmed to `simulation_time / output_frequency` outputs (the
  spinup_outputs prefix is dropped).
- Failures surface as `subprocess.CalledProcessError`. **To see swallowed
  errors, override `model.forward_model.verbose=true` on the CLI.**
- STL → LBM geometry conversion is implemented but not fully trusted (per
  README caveat).

### pyudales
- The Matlab binary is set on `cfg.model.forward_model.matlab_bin`. A
  pure-Python preprocessor exists in `python_udgeom/` and is selected by
  the `prepare._target_` block in [conf/model/pyudales.yaml](../conf/model/pyudales.yaml)
  via `python_or_matlab: python`, which is what
  `pyurbanair.config.hydra_helpers.prepare_udales` passes through.
- Runtime config in `namoptions.<exp>` (edited via `NamoptionsFile`).
- Staggered grid: state has `xt/xm`, `yt/ym`, `zt/zm`. Some plotting
  utilities call `interpolate_grid` to project everything onto a common
  grid before display.
- Inflow profile is configured via the nested `nudging_config` field on
  `cfg.model.forward_model` (`profile_config = {"type": "uniform" |
  "power_law", "alpha": ..., "z_ref": ...}`).
- `pressure_gradient_magnitude` is the third parameter only this backend
  supports. The helpers `create_true_params` /
  `create_time_varying_true_params` filter it out for non-uDALES
  models.

### pypalm
- Lazy-imported. All `pypalm.*` `_target_` blocks live exclusively in
  [conf/model/pypalm.yaml](../conf/model/pypalm.yaml), so Hydra only
  triggers the PALM import when that config is instantiated. Non-PALM
  runs never pay the compile cost. This invariant is asserted by
  `tests/test_hydra_config.py::test_palm_target_does_not_import_for_non_palm_composition`.
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
5. Add a new `conf/model/<name>.yaml` mirroring
   [conf/model/pylbm.yaml](../conf/model/pylbm.yaml): `name`,
   `solver_name`, `forward_model._target_`, `ensemble_model._target_`,
   `prepare._target_`. Use the existing `prepare_compile` /
   `prepare_udales` helpers in
   [src/pyurbanair/config/hydra_helpers.py](../src/pyurbanair/config/hydra_helpers.py)
   if they fit; add a new prepare helper there otherwise. Add a
   `clean_outputs` branch in the same module for the new `model_name`.
   **Watch out**: today `clean_outputs` is an if/elif chain with an
   `else` arm that falls back to the uDALES cleanup
   ([src/pyurbanair/config/hydra_helpers.py:55-64](../src/pyurbanair/config/hydra_helpers.py#L55-L64)).
   Adding a new backend means both inserting an `elif` branch AND
   making sure unrecognized models no longer silently get uDALES
   cleanup — raise on the else arm instead of leaving the fall-through.
6. Add a `dim_mapping` entry in
   [`ObservationOperator.__init__`](../libs/data-assimilation/src/data_assimilation/observation_operator.py)
   for the new `solver_name`.
7. Add a regression test that the model composes without importing the
   backend lazily (mirror
   `test_palm_target_does_not_import_for_non_palm_composition`).

### Add a new parameter
- Add a default value under `params.true` and a matching prior under
  `params.prior` in [conf/parameters.yaml](../conf/parameters.yaml). If the
  parameter is time-varying and feeds the truth/prior ts-models, also add it to
  `params.external` in the same file.
- Extend `create_true_params` and `create_parameter_ensemble` in
  [src/pyurbanair/config/hydra_helpers.py](../src/pyurbanair/config/hydra_helpers.py)
  to include the new field. Watch out for the model-conditional filter
  (`if model_name == "pyudales": ...`) — `pressure_gradient_magnitude`
  uses it today.
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
- Add a new top-level `scripts/run_<name>.py` with its own primary config
  `conf/run_<name>.py`-style YAML (`defaults: [config]` + an
  `esmda.smoother._target_` block pointing to your class; use
  sibling-relative interpolation `${..num_steps}` / `${..localization}` for
  the shared ESMDA fields). Point the script's `@hydra.main` at that
  `config_name`. Pattern: build truth model, draw truth obs, build assim
  ensemble + obs op + C_D, `instantiate(cfg.esmda.smoother, ...)` with the
  dynamic kwargs, call it, dump results.

### Add a new localization strategy
- Subclass `BaseLocalization` in
  [localization/](../libs/data-assimilation/src/data_assimilation/localization/)
  and implement `inflation_factors(aug_dev, pred_obs_dev)`. Return `(N_aug,
  N_d)`: `1.0` keeps an observation for a row, `>1` tapers it, `jnp.inf`
  excludes it. The shared `localized_update` handles the Kalman math — a
  distance-based strategy (Gaspari–Cohn taper + radius cutoff) drops in by
  implementing only this method.
- Localization lives in [conf/esmda.yaml](../conf/esmda.yaml) under
  `esmda.localization` (the default correlation block). Swap in your strategy
  by overriding `esmda.localization._target_` (and its args), or replace the
  block in `esmda.yaml`. Every smoother already receives it via
  `localization: ${..localization}`; `esmda.localization=null` gives the global
  update.

### Add a new run script
- Place under [scripts/](../scripts/), mirror an existing one. The
  shape is `def run(cfg)` + a thin `@hydra.main`-decorated `main`.
- Use `hydra.utils.instantiate(cfg.model.forward_model, ...)` for
  backend construction; pull procedural helpers from
  [pyurbanair.config.hydra_helpers](../src/pyurbanair/config/hydra_helpers.py).
- Compute outputs via `resolve_output_dir(cfg, "<script_name>")` so the
  script writes under `${paths.base_results_dir}/<script_name>/` when
  invoked directly and under Hydra's auto-managed run dir when invoked
  via `@hydra.main`.
- Gate visualization with `cfg.run.skip_viz`. Per-script behavior
  knobs that don't fit any config group land under `run.*` (e.g.
  `run.num_steps`, `run.use_true_params`).
- Add a test in `tests/test_<name>.py` that uses `compose_test_cfg` to
  invoke `run(cfg)` directly.

## 9. Operational defaults / scaling

- `.temp/` is the default scratch directory. Every backend writes its
  per-experiment dir and per-member dirs underneath. The default
  `paths.base_results_dir` is `.temp/scripts` (see
  [conf/paths.yaml](../conf/paths.yaml)).
- Tests use [conf/preset/test.yaml](../conf/preset/test.yaml) (small
  domain / short simulation_time / CPU-only LBM / 4-member ensemble) —
  `pixi run py.test` in the dev env.
- Pre-commit hooks (`black`, `isort`, `mypy`) installed via
  `pixi run pre-commit`. They are **not enforced** server-side; commits
  can bypass.
- **Ensemble scaling on this hardware** is DRAM-bandwidth-bound past
  ~4 workers (see [docs/ensemble_scaling.md](ensemble_scaling.md) and
  the comments in [conf/ensemble.yaml](../conf/ensemble.yaml)).
  Don't blindly raise `num_parallel_processes` past 8 — re-benchmark
  first.
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
| Run-time parameters (domain, sim time, ensemble size) | [conf/](../conf/) — see §5 |
| Per-backend `model_name → class` wiring | [conf/model/](../conf/model/) (`forward_model._target_` / `ensemble_model._target_` blocks) |
| Hydra `_target_` helpers (prepare, clean, factories) | [src/pyurbanair/config/hydra_helpers.py](../src/pyurbanair/config/hydra_helpers.py) |
| What an ensemble member does in parallel | [src/pyurbanair/base_ensemble_forward_model.py](../src/pyurbanair/base_ensemble_forward_model.py) |
| How a solver consumes params | `libs/<solver>/src/<solver>/utils/params_utils.py` |
| ESMDA Kalman update | [libs/data-assimilation/src/data_assimilation/smoothing/esmda.py](../libs/data-assimilation/src/data_assimilation/smoothing/esmda.py) |
| How sensors map to grid points | [libs/data-assimilation/src/data_assimilation/observation_operator.py](../libs/data-assimilation/src/data_assimilation/observation_operator.py) |
| Per-window rollout logic | [src/pyurbanair/base_rollout_forward_model.py](../src/pyurbanair/base_rollout_forward_model.py) and `scripts/run_rollout_*.py` |
| Time-varying parameter prior | [src/pyurbanair/parameter_time_series/](../src/pyurbanair/parameter_time_series/) and `time_varying.*` in [conf/parameters.yaml](../conf/parameters.yaml) |
| Test fixture composition | [tests/conftest.py](../tests/conftest.py) (`compose_test_cfg`, `compose_module_cfg`) |
| Benchmark / scaling experiments | `scripts/benchmark_*_ensemble_scaling.py`, [docs/ensemble_scaling.md](ensemble_scaling.md) |
