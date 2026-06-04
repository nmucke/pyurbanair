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
  base_rollout_forward_model.py    # BaseRolloutForwardModel (legacy; file-only, unused)
  quiet_jax.py                     # Import before `jax` to suppress CPU-fallback noise
  static_parameters/               # Static parameter sampler (ParameterSampler +
                                   #   Normal/Uniform/Constant Distributions)
  dynamic_parameters/              # Time-varying parameter prior (AR2RelaxationModel
                                   #   + ParameterTimeSeries base). Only method left.
  training_data/                   # Sampler skeletons for surrogate data generation
  config/
    hydra_helpers.py               # Targets that Hydra `_target_` blocks instantiate
                                   #   (prepare_*, clean_outputs, create_observation_*,
                                   #    create_C_D, create_initial_state_ensemble,
                                   #    resolve_output_dir, resolve_parameter_schema, ...)
  utils/
    cpu_pinning.py                 # Worker → CPU pinning for parallel ensembles
    run_utils.py, state_utils.py, animation_utils.py
  plotting.py, animation.py

conf/                              # Hydra config (see §5 Configuration system)
  config.yaml                      # Base composition (forward-model runs) + `run:` namespace
  run_esmda.yaml                   # Primary config for run_esmda.py (smoother + double-mount)
  generate_training_data.yaml      # Primary config for generate_training_data.py
  paths.yaml, time.yaml, ensemble.yaml, esmda.yaml   # Shared flat files (one per category)
  case/                            # Geometry case bundles domain+obs+geometry (xie_and_castro,
                                   #   barcelona). Switch with `case=...`.
  params/                          # Parameter samplers: static, dynamic, static_truth,
                                   #   dynamic_truth (mounted twice: truth + prior)
  esmda/smoother/                  # ESMDA variant group: static, dynamic, state_and_parameter
  model/                           # forward + ensemble backend (mounted under model@<pkg>)
  size/, preset/, training_data/, neural_surrogate_*/   # remaining overlays/groups

libs/data-assimilation/src/data_assimilation/
  observation_operator.py          # ObservationOperator + TemporalObservationOperator
  interpolation.py                 # Grid → sensor-point interpolation
  localization/
    base.py                        # BaseLocalization — inflation_factors + localized_update
    correlation.py                 # CorrelationLocalization (adaptive correlation-based)
  localization/                    # see §6
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
libs/neural-surrogates/src/neural_surrogates/   # Learned one-step CFD surrogate (PyTorch)

scripts/                           # All top-level executables run from here.
                                   # Each exposes `def run(cfg)` + a thin `@hydra.main` wrapper.
  _common.py                       # Shared script glue (results-dir resolution, forward viz,
                                   #   derived-inflow / time-varying-param plots + metrics)
  run_forward_model.py             # Forward sim — single/ensemble (run.ensemble),
                                   #   single-window/rollout (run.rollout_steps), static or
                                   #   time-varying inflow (params=static|dynamic). Replaces the
                                   #   former run_{ensemble_,rollout_,ensemble_rollout_,
                                   #   time_varying_}forward_model.py family.
  run_esmda.py                     # THE single ESMDA entry point. Replaces the former
                                   #   run_{parameter,state_and_parameter,rollout,
                                   #   time_varying_parameter,time_varying_parameters_rollout}_esmda.py
                                   #   family. Mode = (esmda/smoother) × (params@prior_params) ×
                                   #   (esmda.num_assimilation_windows); truth simulated inline or
                                   #   loaded from disk (run.truth_dir).
  generate_training_data.py        # Build surrogate training dataset from a CFD ensemble
  train_neural_surrogate.py        # Train a surrogate
  test_neural_surrogate.py         # Autoregressive rollout on the test split
  dataloading.py                   # TransitionDataset smoke test

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
- **Failure policy** — passed at construction via the `failure=` arg (the
  ensemble model's `_target_` block wires `failure: ${ensemble.failure}`);
  reconfigurable later via `configure_failure_policy` (used by
  `generate_training_data.py` to force `"raise"`):
  - `"raise"` — first failure aborts the whole ensemble.
  - `"resample_from_successes"` (the default in `conf/ensemble.yaml`) — failed
    members are cloned from a random successful donor; the *params* ensemble can
    be re-cloned (with Gaussian jitter) by calling
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
> runtime — the file remains but nothing imports it. Multi-window
> driving is handled directly in the scripts (`run_esmda.py`'s window
> loop and `run_forward_model.py`'s `run.rollout_steps` loop) by
> repeatedly invoking the forward model with state carry-over and
> re-extrapolating the parameter prior between windows.

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
[`conf/`](../conf/). There are **two primary configs**:
[`conf/config.yaml`](../conf/config.yaml) (forward-model runs) and
[`conf/run_esmda.yaml`](../conf/run_esmda.yaml) (all ESMDA runs); a third,
[`conf/generate_training_data.yaml`](../conf/generate_training_data.yaml),
drives surrogate data generation. Each composes a mix of **shared flat files**
(one per category) and **groups** (one option per structurally-distinct
variant), and bundles a `run:` namespace for generic script-behavior knobs
(`skip_viz`, `results_dir`, `ensemble`, `rollout_steps`, `truth_dir`).

Shared flat files use a `# @package <category>` directive so the body lands at
the right runtime key:

| File | Runtime key | Notable fields |
|---|---|---|
| [`paths.yaml`](../conf/paths.yaml) | `paths` | `results_dir` (default `.temp/${model.name}`), `experiment_dir` |
| [`time.yaml`](../conf/time.yaml) | `time` | `simulation_time`, `output_frequency`, `spinup_time` |
| [`ensemble.yaml`](../conf/ensemble.yaml) | `ensemble` | `ensemble_size`, `num_parallel_processes`, `failure.{policy, jitter_scale, seed}` |
| [`esmda.yaml`](../conf/esmda.yaml) | `esmda` | `num_steps`, `alpha`, `num_assimilation_windows`, `obs_error_std`, `seed`, `localization` (correlation block; `null` = global). Mounted only by `run_esmda.yaml`. |

Everything that varies per variant is a **group**:

| Group | Selects | Notable |
|---|---|---|
| `case/` | geometry bundle (`xie_and_castro`, `barcelona`) | bundles `domain` + `obs` + `geometry` for one geometry. `case=xie_and_castro` is the default. `domain.yaml`/`obs.yaml` live here now, not at the conf root. |
| `model/` | forward + ensemble backend | mounted under a package: `model@model=pylbm` (forward), or twice for assimilation (see below) |
| `params/` | parameter sampler | `static`, `dynamic`, `static_truth`, `dynamic_truth` — see below. Mounts at runtime key `params` (forward) or `truth_params`/`prior_params` (esmda). |
| `esmda/smoother/` | ESMDA variant | `static` (`ParameterESMDA`), `dynamic` (`TimeVaryingParameterESMDA`), `state_and_parameter` (`StateAndParameterESMDA`). The one genuinely mode-specific `_target_`. |
| `size/` | run-size overlay (`tiny`→`xlarge`, plus `test`) | `# @package _global_`; deep-merges over the case/flat files |
| `preset/` | bundled overlays (`small`, `test`) | smaller domain / fewer steps / CPU-only LBM |
| `training_data/` | surrogate data-generation overlay | each size pulls `training_data/_base.yaml` (sampler skeleton + horizon) and overrides only what scales |

**Parameter samplers** are a group mounted *by package*, not a flat file. Each
option is a single sampler `_target_`:
- `static` / `dynamic` — the assimilation **prior** (Normal priors / AR(2)
  external prior).
- `static_truth` / `dynamic_truth` — the **truth** generator (Constants / a
  distinct AR(2) seed). Keeping truth and prior as separate configs avoids the
  inverse crime.

`config.yaml` mounts one sampler at key `params` (`params=static|dynamic`).
`run_esmda.yaml` mounts the group **twice** — `params@truth_params` and
`params@prior_params` — exactly mirroring the model double-mount.

**ESMDA smoother** is selected via the `esmda/smoother` group (default
`dynamic`). It is the one genuinely mode-specific piece; the shared
`num_steps`/`alpha`/`localization` come from `esmda.yaml` via
`${esmda.*}` interpolation. The single [`run_esmda.py`](../scripts/run_esmda.py)
script handles every former esmda script — mode is the cross product of
`esmda/smoother`, `params@prior_params`, and `esmda.num_assimilation_windows`
(1 = single window, N = rollout). Tests compose `config_name="run_esmda"` and
pick the smoother via the group override (see [tests/conftest.py](../tests/conftest.py)).

> **Naming drift to watch**: the `run_esmda.yaml` / `run_esmda.py` header
> comments call the smoother options `parameter|state_and_parameter|time_varying`,
> and `tests/test_run_esmda.py` uses those names — but the actual files in
> [`conf/esmda/smoother/`](../conf/esmda/smoother/) are
> `static|state_and_parameter|dynamic`. The CLI selector must match the
> **filenames** (`esmda/smoother=static`, `esmda/smoother=dynamic`).

**Localization** lives in the esmda namespace: `esmda.localization` is the
adaptive correlation-localization block (Vossepoel et al. 2025) and is applied
by every smoother (`localization: ${..localization}`). Run the global,
unlocalized update with `esmda.localization=null`.

The `size/` overlays are `# @package _global_` and are the single place a run
is sized (`size=medium`). Each inlines `domain`/`time` and overrides only the
fields that scale with the run (`ensemble.ensemble_size`,
`ensemble.num_parallel_processes`, the `obs` sensor coords + `interval_seconds`,
`esmda.num_steps`/`num_assimilation_windows`), deep-merging over the
case/flat base files.

Forward-model runs mount the model once at `cfg.model.*`. Assimilation runs use
Hydra's package-override syntax to mount the same `model/` and `params/` groups
**twice**, once as the truth and once as the assim/prior:

```bash
python scripts/run_esmda.py \
  model@truth_model=pylbm model@assim_model=pyudales \
  params@truth_params=static_truth params@prior_params=static \
  esmda/smoother=static
```

Inside the YAMLs, sibling-relative interpolation (`${.foo}`, `${..foo}`)
is used wherever the surrounding group might be re-mounted under
another package; absolute interpolation (`${time.simulation_time}`) is
reserved for cross-group lookups.

### Instantiation vs. helpers

Backend object construction is **declarative** — every forward model,
ensemble model, ESMDA smoother, parameter sampler, and time-varying prior is
built by `hydra.utils.instantiate(cfg.<group>, ...)` against the `_target_`
block in YAML. Both samplers and the smoother are full `_target_` configs (the
old `create_true_params` / `create_parameter_ensemble` / `configure_failure_policy`
/ `build_truth_ts_model` helpers are gone). The scripts call only a small set of
remaining procedural helpers from
[src/pyurbanair/config/hydra_helpers.py](../src/pyurbanair/config/hydra_helpers.py):

```python
# Assim model + ensemble (run_esmda.py)
assim_model     = instantiate(cfg.assim_model.forward_model, results_dir=...)
instantiate(cfg.assim_model.prepare, forward_model=assim_model)   # compile / preprocess
clean_outputs(model_name=cfg.assim_model.name, forward_model=assim_model)
ensemble_model  = instantiate(cfg.assim_model.ensemble_model, forward_model=assim_model)

# Truth + prior samplers, each its own _target_ block (mounted twice)
truth_sampler   = instantiate(cfg.truth_params)        # static_truth | dynamic_truth
true_params     = truth_sampler.sample(1)
prior_sampler   = instantiate(cfg.prior_params)        # static | dynamic
prior_params    = prior_sampler.sample(cfg.ensemble.ensemble_size)

# Observation operator + error covariance (helpers)
obs_op          = create_observation_operator(cfg.obs, cfg.assim_model.solver_name)
C_D             = create_C_D(num_obs, cfg.esmda.obs_error_std)

esmda           = instantiate(cfg.esmda.smoother, ..., rng_key=rng_key)
```

The failure policy is now configured directly in the ensemble model's
`_target_` block (`conf/model/<name>.yaml` → `ensemble_model`), reading
`${ensemble.failure.*}`; there is no separate `configure_failure_policy` call.

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

Every fixture injects `+size=test` (the smallest domain / shortest window /
2-member ensemble; see [conf/size/test.yaml](../conf/size/test.yaml)). Override
anything per-test:

```python
def test_something(compose_test_cfg) -> None:
    # ESMDA tests pass run_esmda as config_name and pick the smoother via the
    # esmda/smoother group override (there is no `esmda=<variant>` selector).
    cfg = compose_test_cfg(
        [
            "model@truth_model=pylbm", "model@assim_model=pyudales",
            "esmda/smoother=static",
            "params@truth_params=static_truth", "params@prior_params=static",
            "esmda.num_steps=1",
        ],
        config_name="run_esmda",
    )
    run(cfg)
```

## 6. Data assimilation flow

### `ObservationOperator` (data-assimilation lib)
- Maps a state Dataset to a flat observation vector of length
  `num_sensors * len(obs_states)`.
- Two construction modes: **index-based** (`obs_ids_*`) or
  **coordinate-based** (`obs_*`, interpolated). The `case/<name>/obs.yaml`
  configs use coordinate-based, built by `create_observation_operator` /
  `create_observation_points` in `hydra_helpers.py`.
- Variable→dim mapping handles each backend's staggered grids.
- `TemporalObservationOperator` wraps it with time aggregation:
  `mean | median | max | min | full | intervals`. `intervals` is the
  config default — observations are binned by their `time` coordinate (in
  seconds) into `interval_seconds`-wide windows and aggregated within each.

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
Handled directly inside [scripts/run_esmda.py](../scripts/run_esmda.py)'s window
loop when `esmda.num_assimilation_windows > 1`. The full truth (state +
parameters) for every window is simulated up front; the loop then, per window:
slices that window's truth observations (a contiguous block of frames), adds
noise, runs the smoother, persists per-window prior/posterior params + state to
`windows_dir`, and feeds the window's final posterior state in as the next
window's `state`. For the **dynamic** (time-varying) case the next window's
prior is `prior_sampler.extrapolate(posterior, ...)`; for the static case it is
just the posterior. `_finish_rollout` then concatenates the per-window files
(rebasing time for the dynamic case) into the final outputs.

### Parameter samplers (static + dynamic)
Both kinds of sampler are built declaratively with
`hydra.utils.instantiate(...)` and share one interface — all configuration is
passed at construction time and **`sample(ensemble_size)`** returns an
`xarray.Dataset` with an `ensemble` dim — so a run draws parameters with two
lines regardless of kind:

```python
params_sampler = instantiate(cfg.params)          # or cfg.truth_params / cfg.prior_params
params = params_sampler.sample(ensemble_size)
```

- **Static** ([src/pyurbanair/static_parameters/](../src/pyurbanair/static_parameters/)) —
  `ParameterSampler` holds a `name -> Distribution` mapping. Each parameter is
  a `Normal` / `Uniform` random prior or a fixed `Constant` (each its own
  `_target_` block), so the same class covers both "sample an ensemble from a
  prior" (`conf/params/static.yaml`) and "use these fixed truth values"
  (`conf/params/static_truth.yaml`, all `Constant`s). Output has an `ensemble`
  dim only (no `time`).
- **Dynamic / time-varying** ([src/pyurbanair/dynamic_parameters/](../src/pyurbanair/dynamic_parameters/)) —
  `AR2RelaxationModel` is the **only** surviving method (the former `ar1`,
  `gp_linear_trend`, `ornstein_uhlenbeck` were removed). Critically-damped AR(2)
  relaxing toward the external prior `x_ext`; output adds a `time` dim. It also
  exposes `extrapolate(posterior, prediction_times, rng_key)` for the next
  rollout window. Configured by `conf/params/dynamic.yaml` (prior) /
  `dynamic_truth.yaml` (truth): `external_parameters` (each a
  `static_parameters` `Distribution`, whose `mean`/`std` may be a scalar or a
  list of control points interpolated over the window), `correlation_length`,
  `seed`, and a `time_coords` built by a nested `numpy.linspace` target.

A single-member run drops the `ensemble` dim with `.isel(ensemble=0, drop=True)`.

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
  supports. The ordered per-model schema comes from `resolve_parameter_schema`
  in `hydra_helpers.py` (adds it for `pyudales` only); the sampler configs
  simply include or omit it (`conf/params/static.yaml` carries it as a
  `Constant`, which non-uDALES backends ignore).

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
   `prepare._target_`, and `ensemble_model.failure: ${ensemble.failure}`. Use
   the existing `prepare_compile` / `prepare_udales` /
   `prepare_neural_surrogate` helpers in
   [src/pyurbanair/config/hydra_helpers.py](../src/pyurbanair/config/hydra_helpers.py)
   if they fit; add a new prepare helper there otherwise. Add a
   `clean_outputs` branch in the same module for the new `model_name`.
   `clean_outputs` is an if/elif chain that now **raises** on the `else`
   arm ([src/pyurbanair/config/hydra_helpers.py:71-89](../src/pyurbanair/config/hydra_helpers.py#L71-L89)),
   so a new backend without its own branch fails loudly rather than silently
   getting uDALES cleanup — add the `elif`.
6. Add a `dim_mapping` entry in
   [`ObservationOperator.__init__`](../libs/data-assimilation/src/data_assimilation/observation_operator.py)
   for the new `solver_name`.
7. Add a regression test that the model composes without importing the
   backend lazily (mirror
   `test_palm_target_does_not_import_for_non_palm_composition`).

### Add a new parameter
- Add the parameter to the sampler configs in
  [conf/params/](../conf/params/): a `Distribution` block in `static.yaml`
  (prior) and `static_truth.yaml` (truth) and/or under `external_parameters`
  in `dynamic.yaml` / `dynamic_truth.yaml`. The samplers pick up any key in the
  mapping — no Python change needed for the sampling side.
- If the parameter is backend-specific (like `pressure_gradient_magnitude`),
  extend `resolve_parameter_schema` in
  [src/pyurbanair/config/hydra_helpers.py](../src/pyurbanair/config/hydra_helpers.py).
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
- Add a new option to the `esmda/smoother` group:
  `conf/esmda/smoother/<name>.yaml` with your class's `_target_` and the
  shared fields wired via `${esmda.num_steps}` / `${esmda.alpha}` /
  `${esmda.localization}`. No new script or primary config is needed —
  [scripts/run_esmda.py](../scripts/run_esmda.py) instantiates whatever
  `cfg.esmda.smoother` resolves to, and you select it with
  `esmda/smoother=<name>`. If the variant needs the augmented state to include
  the flattened field, branch on `isinstance(esmda, StateAndParameterESMDA)` as
  the script already does.

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
  script writes under Hydra's auto-managed run dir when invoked via
  `@hydra.main`, and under `${paths.*}` when `run(cfg)` is called directly.
- Gate visualization with `cfg.run.skip_viz`. Per-script behavior
  knobs that don't fit any config group land under `run.*` (e.g.
  `run.ensemble`, `run.rollout_steps`, `run.truth_dir`).
- Add a test in `tests/test_<name>.py` that uses `compose_test_cfg` to
  invoke `run(cfg)` directly.

## 9. Operational defaults / scaling

- `.temp/` is the default scratch directory. Every backend writes its
  per-experiment dir and per-member dirs underneath. The default
  `paths.results_dir` is `.temp/${model.name}` and `experiment_dir` is
  `.temp` (see [conf/paths.yaml](../conf/paths.yaml)); `run_esmda.yaml`
  overrides `results_dir` to `.temp/${truth_model.name}_to_${assim_model.name}`.
- Tests inject `+size=test` ([conf/size/test.yaml](../conf/size/test.yaml):
  tiny domain / 3 s window / 2-member ensemble) via the conftest fixtures —
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
| Run-time parameters (sim time, ensemble size, esmda steps) | [conf/](../conf/) flat files — see §5 |
| Geometry / domain / sensor layout | [conf/case/](../conf/case/) (`case=<name>` bundles `domain`+`obs`+`geometry`) |
| Per-backend `model_name → class` wiring | [conf/model/](../conf/model/) (`forward_model._target_` / `ensemble_model._target_` blocks) |
| Hydra `_target_` helpers (prepare, clean, obs operator) | [src/pyurbanair/config/hydra_helpers.py](../src/pyurbanair/config/hydra_helpers.py) |
| What an ensemble member does in parallel | [src/pyurbanair/base_ensemble_forward_model.py](../src/pyurbanair/base_ensemble_forward_model.py) |
| How a solver consumes params | `libs/<solver>/src/<solver>/utils/params_utils.py` |
| ESMDA Kalman update / variants | [libs/data-assimilation/src/data_assimilation/smoothing/esmda.py](../libs/data-assimilation/src/data_assimilation/smoothing/esmda.py) |
| Which ESMDA mode runs (smoother/prior/windows) | [scripts/run_esmda.py](../scripts/run_esmda.py) + [conf/run_esmda.yaml](../conf/run_esmda.yaml), [conf/esmda/smoother/](../conf/esmda/smoother/) |
| How sensors map to grid points | [libs/data-assimilation/src/data_assimilation/observation_operator.py](../libs/data-assimilation/src/data_assimilation/observation_operator.py) |
| Per-window rollout logic | `run_esmda.py`'s window loop / `run_forward_model.py`'s `run.rollout_steps` loop |
| Parameter samplers (static + dynamic) | [src/pyurbanair/static_parameters/](../src/pyurbanair/static_parameters/), [src/pyurbanair/dynamic_parameters/](../src/pyurbanair/dynamic_parameters/), [conf/params/](../conf/params/) |
| Test fixture composition | [tests/conftest.py](../tests/conftest.py) (`compose_test_cfg`, `compose_module_cfg`) |
| Benchmark / scaling findings | [docs/ensemble_scaling.md](ensemble_scaling.md) (the one-off benchmark scripts were removed; recover from git history to re-run) |
