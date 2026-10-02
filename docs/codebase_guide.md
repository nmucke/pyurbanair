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

## 1a. Documentation map — where to read before you edit

This guide is the **entrypoint**; it stays at the level of cross-cutting
structure, contracts, and conventions. Each library / area has a dedicated
deep-dive doc. When a task targets one area, **read that doc first** — it has the
file-level detail, gotchas, and recipes this guide only summarizes:

| When the task touches… | Read |
|---|---|
| The LBM backend (compile, `infile.in`, STL/bathymetry geometry, CUDA, warm starts) | [docs/pylbm.md](pylbm.md) |
| The uDALES backend (`namoptions`, staggered grid, nudging, dt-collapse watchdog, ncpu) | [docs/pyudales.md](pyudales.md) |
| The PALM backend (lazy import, `_p3d` namelists, direct-run path, km_constant, topography) | [docs/pypalm.md](pypalm.md) |
| ESMDA / observation operator / localization / state reduction | [docs/data_assimilation.md](data_assimilation.md) |
| Neural surrogates (UNetConvNeXt, UPT, P3D, domain-decomposition, training, rollout) | [docs/neural_surrogates.md](neural_surrogates.md) |
| Hydra configs (`configs/`), the executable scripts (`scripts/`) and `workflows/` | [docs/scripts_and_configs.md](scripts_and_configs.md) |
| Tests (`tests/`: per-package folders, script tests, overlays) | [tests/README.md](../tests/README.md) |
| Local MCP forward jobs, preparation and client setup | [docs/mcp.md](mcp.md) |
| Saved forward visualization and browser bundles (`libs/visualization`) | [docs/visualization.md](visualization.md) |
| Running on HPC clusters (Snellius / DelftBlue / local SLURM) | [docs/job_scripts.md](job_scripts.md) |
| Dynamic multi-window ESMDA theory/config | [docs/temp/esmda_dynamic_multiwindow.md](temp/esmda_dynamic_multiwindow.md) |
| Model-error compensation parameters (α, sgs/km) | [docs/temp/esmda_model_error_parameters.md](temp/esmda_model_error_parameters.md) |
| Reduced SVD/KL state update theory | [docs/temp/reduced_state_da.md](temp/reduced_state_da.md) |
| Ensemble parallel-scaling findings | [docs/temp/ensemble_scaling.md](temp/ensemble_scaling.md) |

> `docs/plans/`, `docs/temp/`, and `docs/domain_decomposition_surrogate/` are
> working notes / design records, not maintained references — useful for theory
> and history, but verify against the code before relying on them.
>
> `archive/` holds the retired `conf/`, `scripts/` and `tests/` (dead code: not
> run, not tested). Docs that still name `conf/...`, `scripts/esmda/...`,
> `run_esmda.py` etc. describe that archived setup.

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
                                   #    resolve_parameter_schema, ...)
  utils/
    cpu_pinning.py                 # Worker → CPU pinning for parallel ensembles
    run_utils.py, state_utils.py, animation_utils.py
  animation.py

configs/                           # Hydra config (see §5 Configuration system)
  README.md                        # Keys + common overrides
  forward.yaml                     # Entry point for run_forward.py
  assimilation.yaml                # Entry point for run_smoother/run_filtering/run_hybrid.py
  common.yaml                      # paths (per machine), ensemble budget, Hydra run dir
  case/                            # Experiment bundle: domain+grid+obs+geometry+time, one self-
                                   #   contained file per case (xie_and_castro, barcelona). `case=...`.
  params/                          # Parameter samplers: static, dynamic, dynamic_sine, dynamic_cosine,
                                   #   static_truth, dynamic_truth (mounted twice: truth + prior),
                                   #   surrogate_training_data
  assimilation_settings/           # Every option of each DA component (smoother, analysis,
                                   #   localization, state_reduction, inflation), at top level
  model/                           # forward + ensemble backend (mounted under model@<pkg>)
  surrogate/                       # surrogate: generate_data, train_*, finetune_stepper, eval,
                                   #   training (shared defaults), architectures
  visualization/                   # Render presets of the MCP viewer (quicklook, flow_3d)

libs/data-assimilation/src/data_assimilation/
  observation_operator.py          # ObservationOperator + TemporalObservationOperator
  interpolation.py                 # Grid → sensor-point interpolation
  localization/                    # see §6 (selected via smoothing/filtering.localization)
    base.py                        # BaseLocalization, taper_inflation, localized_update
    correlation.py                 # CorrelationLocalization (adaptive correlation-based)
    distance.py                    # DistanceLocalization (physical-distance-based)
  smoothing/
    base.py                        # BaseSmoothing — _forecast_step, _observation_step
    esmda.py                       # Parameter/StateAndParameter/TimeVaryingParameter/
                                   #   StateAndTimeVaryingParameter ESMDA

libs/mcp-server/src/mcp_server/    # Optional MCP server (forward runs); see docs/mcp.md
  server.py, tools.py              # MCP SDK v2 tool registration + thin adapters
  jobs/                            # SDK-free: composition of configs/forward.yaml, immutable
                                   #   plans (check_config + limits), private paths, SQLite
                                   #   queue, supervisor, workers (run scripts/run_forward.py)

libs/visualization/src/visualization/ # Saved-state normalization, PNG/MP4 rendering and the
                                   #   local browser viewer for a run's state.nc; see
                                   #   docs/visualization.md

libs/evaluation/src/evaluation/    # Metrics + figures for DA runs. Leaf lib: no jax, no
                                   #   pyurbanair, no backends (see its __init__).
  scores.py                        # Ensemble scores (CRPS, energy score, per-knot skill)
                                   #   + the parameter/sensor metric bundles
  turbulence.py                    # z-plane selection, streaming |U| state RMSE, mean-field
                                   #   moments, Welch probe spectra + log-spectral distance
  sensors.py                       # Reductions of pre-extracted sensor series
  style.py                         # Talk-figure palette/rcParams/save + STL solid masks
  figures.py                       # plot_* for DA runs (parameters, sensors, state,
                                   #   mean-field slices, station profiles, rank hist.,
                                   #   probe spectra)

libs/pylbm/src/pylbm/              # LBM wrapper. __init__ git-clones the LBM Fortran code.
  forward_model.py                 # ForwardModel(BaseForwardModel)
  ensemble_forward_model.py        # EnsembleForwardModel(BaseEnsembleForwardModel)
  stl_to_lbm.py                    # STL → LBM voxel geometry
  utils/                           # infile.in editing, compile, warm-start, params, ...

libs/pyudales/src/pyudales/        # uDALES wrapper; explicit cached solver preparation.
  forward_model.py, ensemble_forward_model.py
  python_udgeom/                   # Python preprocessing alternative to Matlab
  utils/                           # namoptions, nudging, ncpu, warm-start, etc.

libs/pypalm/src/pypalm/            # PALM wrapper. Similar shape.
libs/neural-surrogates/src/neural_surrogates/   # Learned one-step CFD surrogate (PyTorch)
  forward_model.py, ensemble_forward_model.py   # NeuralSurrogate{,Ensemble}ForwardModel
  architectures/                   # simple_conv, unet_convnext, upt (+_upt/), p3d, domain_decomposed
  datasets/                        # transition.py (TransitionDataset), patch.py (PatchTransitionDataset)
  training/                        # base.py (BaseTraining), standard.py (Trainer), patch.py (PatchTrainer)
  decomposition.py, dd_loss.py, geometry.py      # DD operators, Eq-9 loss, STL→voxel channel

scripts/                           # Scripts you run; their shared helpers are in utils/.
                                   # Hydra scripts expose `def run(cfg)` + a thin `@hydra.main` wrapper.
  run_forward.py                   # Forward sim — single/ensemble (forward.ensemble), extra
                                   #   windows (forward.rollout_steps), static or time-varying params.
  run_smoother.py                  # ESMDA over consecutive windows (smoothing.*)
  run_filtering.py                 # Cycled EnKF (filtering.*)
  run_hybrid.py                    # Per window: ESMDA params, then filter state
  compute_metrics.py               # metrics.yaml from a finished DA run dir (plain CLI)
  visualize_forward.py, visualize_assimilation.py   # Figures from a run dir (plain CLI)
  surrogate/                       # generate_data.py, train.py, evaluate_*.py
                                   #   — see docs/neural_surrogates.md
  utils/                           # Helpers only, nothing to run:
                                   #   inconsistency_check.py (check_config, called first in every run_*.py),
                                   #   helper_functions.py (truth, observations, ensemble model, I/O),
                                   #   tasks.py (train.py's per-task setup), eval_common.py (evaluate_*.py)
  tools/                           # Case setup CLIs (prepare_case_stl, preprocess_udales_geometry)
  setup_dev_env.sh, start_mcp      # `pixi run setup-dev`; MCP launcher
  register_claude.sh               # `pixi run -e mcp register-claude`

workflows/                         # forward_workflow.sh, assimilation_workflow.sh <method>:
                                   #   run + post-processing on the same run dir

examples/
  benchmark_geometry/              # Xie & Castro 2008 geometry generator (CLI)
  lbm/, udales/, palm/             # Per-backend experiment dirs (STL, namoptions, p3d, etc.)

tests/                             # pytest suite, one folder per package + scripts/ (for
                                   #   scripts/, configs/, workflows/) + configs/ (tiny overlays)
                                   #   + legacy/ (old fixtures). See tests/README.md.
archive/                           # Retired conf/, scripts/, tests/ (not run or tested)
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
  - `"resample_from_successes"` (the default in `common/runtime.yaml`) — failed
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
> driving is handled directly in the scripts (the DA scripts' window
> loops and `run_forward.py`'s `forward.rollout_steps` loop) by
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

Hydra composes two entry points in [`configs/`](../configs/): `forward.yaml`
and `assimilation.yaml`. [`configs/README.md`](../configs/README.md) lists every
key; [docs/scripts_and_configs.md](scripts_and_configs.md) is the reference.

- `common.yaml` owns the run name, `paths` (`paths.machine` picks the solver
  scratch dir per machine), the single `ensemble` budget (incl. `failure`
  policy) and Hydra's run dir.
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
selected components explicitly. Dynamic smoother knot counts are supplied from
sampled data, not editable YAML values.

### Tests

The script tests compose the real `configs/` entry points made tiny by an
overlay from `tests/configs/` (`compose("forward", "+test=forward",
root=tmp_path)`: 20×20×4 cells, 3 s windows, two members on one worker); every
output goes under `root`. Most script tests run on the neural-surrogate
backend trained once per session on synthetic data, so they need no compiled
solver. Some library tests still use the frozen old-schema configs through the
`compose_test_cfg` / `compose_module_cfg` fixtures in `tests/legacy/`.

The default test command (`pixi run -e dev py.test`) runs fast tests; real CFD
calls are marked `integration` (`test-integration`, `test-all`). See
[`tests/README.md`](../tests/README.md). The tests require `forkserver` for
parallel ensembles.

## 6. Data assimilation flow

### `ObservationOperator` (data-assimilation lib)
- Maps a state Dataset to a flat observation vector of length
  `num_sensors * len(obs_states)`.
- Two construction modes: **index-based** (`obs_ids_*`) or
  **coordinate-based** (`obs_*`, interpolated). The `case/<name>/obs.yaml`
  configs use coordinate-based: `observation.operator` in
  `configs/assimilation.yaml` reads the case's `obs.*` points and
  `scripts/utils/helper_functions.py::make_observation_operator` instantiates it.
- Variable→dim mapping handles each backend's staggered grids.
- `TemporalObservationOperator` wraps it and applies it per output frame,
  returning a time-resolved labelled xarray. Interval aggregation lives in
  `AggregateObservations` (same module), an optional input to the DA classes:
  observations are binned by their `time` coordinate (in seconds) into
  `interval_seconds`-wide windows and aggregated within each — configured by
  `observation.aggregation` in `configs/assimilation.yaml` (`null` = off).

### ESMDA
- `BaseSmoothing` ([libs/data-assimilation/src/data_assimilation/smoothing/base.py](../libs/data-assimilation/src/data_assimilation/smoothing/base.py))
  provides `_forecast_step` (runs the ensemble) and `_observation_step`
  (applies the observation operator).
- `_BaseESMDA` provides the shared Kalman update
  (`_compute_kalman_update`) and the `_analysis` loop that drives
  `num_steps` iterations. The loop is an **iterated joint update**: each
  iteration forecasts from the *current* initial-condition estimate, and the
  Kalman-updated IC from `_one_step` is fed forward to the next iteration (and
  the posterior forecast / next window). Parameter-only variants return no
  state, so their `initial_state` stays the caller's pinned value (behavior
  unchanged); the state-bearing variants warm-start the next forecast from the
  analyzed IC, which is what makes the state estimate actually affect the
  results (it previously did not — the analyzed IC was discarded).
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

### The assimilation entry points

ESMDA is not the only one. All share the observation operator, the
augmentation layer, the localization machinery and the analysis math above —
they differ in *what* is updated and *when*. All read
[configs/assimilation.yaml](../configs/assimilation.yaml):

| Script | Algorithm |
|---|---|
| [scripts/run_smoother.py](../scripts/run_smoother.py) | ESMDA smoothing — re-forecast the window `smoothing.num_steps` times with tempered updates. Variant = `smoothing.smoother` × `params@prior_params` × `assimilation.num_windows`. |
| [scripts/run_filtering.py](../scripts/run_filtering.py) | Sequential EnKF — forecast `assimilation.assimilate_every_n_step` frames, then one analysis; windows are chunking only. Mode = `filtering.mode=state\|parameter\|joint` × `filtering.analysis/localization/state_reduction/inflation/parameter_evolution`. |
| [scripts/run_hybrid.py](../scripts/run_hybrid.py) | Filter smoothing — per window, ESMDA estimates the parameters, then the filter produces the state over the same observations. |

Full detail (cycle semantics, the filter hooks) is in
[docs/data_assimilation.md](data_assimilation.md) §8; the configs and saved
artifacts are in [docs/scripts_and_configs.md](scripts_and_configs.md).

All three write the same per-window layout under
`<paths.results_dir>/<smoother|filtering|hybrid>/`, read by
`scripts/compute_metrics.py <run dir>` (`metrics.yaml`) and
`scripts/visualize_assimilation.py <run dir>` (figures);
`workflows/assimilation_workflow.sh <method>` chains the three stages.

### Localization (optional)
- [localization/base.py](../libs/data-assimilation/src/data_assimilation/localization/base.py)
  defines `BaseLocalization`. Subclasses implement one method,
  `inflation_factors(aug_dev, pred_obs_dev, row_coords=None, obs_coords=None) ->
  (N_aug, N_d)`, returning a per-(state-row, observation) observation-error
  inflation factor (`1.0` = keep, `>1` = taper, `inf` = exclude). Inflation
  multiplies the observation-error *perturbation* (std), so the error variance is
  scaled by `inflation**2`. The shared `taper_inflation(distance, truncation,
  beta, max_inflation)` (Vossepoel Eqs. 9–10) drives **both** strategies. The
  shared local-analysis math lives in `localized_update(..., group_ids=None,
  localize_mask=None, row_coords=None, obs_coords=None)`, which updates each
  augmented row with only its relevant observations (Vossepoel et al. 2025,
  MWR-D-24-0269.1).
- Two strategies (a `class.requires_coordinates` flag tells the smoother whether
  to compute geometry):
  - `CorrelationLocalization`
    ([localization/correlation.py](../libs/data-assimilation/src/data_assimilation/localization/correlation.py))
    selects observations by ensemble correlation: exclude `|ρ| < ρ_t`, taper the
    rest by correlation distance `1-|ρ|`. Needs no spatial coordinates
    (`requires_coordinates=False`).
  - `DistanceLocalization`
    ([localization/distance.py](../libs/data-assimilation/src/data_assimilation/localization/distance.py))
    selects observations by **physical Euclidean distance** between the state
    grid point and the sensor: exclude beyond `localization_radius`, taper the
    rest. `requires_coordinates=True` (`horizontal_only` ignores the vertical);
    only valid on a state-bearing smoother with coordinate-based observations.
- **Strategy-aware joint localization.** State-bearing smoothers localize state
  rows. Correlation also localizes parameter rows; distance sets those parameter
  rows to `localize_mask=False` (all-ones inflation = exact global update). The
  shared `StateAndParameterESMDA._augmented_state_update` also computes
  coordinates for `requires_coordinates` strategies via
  `_state_row_coords` (per-row x/y/z, dim axis from the dim-name prefix, in
  `_flatten_state` order) and `_observation_coords` (sensor xyz tiled, since the
  sensor is the innermost obs-vector axis → obs `j` ↔ sensor `j % num_sensors`).
- `_BaseESMDA` takes an optional `localization=` arg. When `None`,
  `_compute_kalman_update` does the original global update unchanged; when set,
  it delegates to `localization.localized_update(...)`. The hook is in the shared
  base, so **all** variants get it. Selected via `smoothing.localization`
  (`${localization.none|correlation|distance}`, options in
  `configs/assimilation_settings/localization.yaml`); every smoother entry wires
  it through with `localization: ${smoothing.localization}`. No script changes
  needed.
- **Grid-block joint analysis** (`block_grouping`, Vossepoel §3b). When the
  strategy has `block_grouping=True`, co-located augmented rows are updated
  *jointly* with one shared observation selection + transition matrix instead of
  per-row. The grouping (`group_ids`) is built in
  [smoothing/esmda.py](../libs/data-assimilation/src/data_assimilation/smoothing/esmda.py):
  `ParameterESMDA` groups by parameter base name (`_group_ids_by_base_name`);
  `StateAndParameterESMDA._state_group_ids` groups the u/v/w at one cell.
  `_group_inflation` in
  [localization/base.py](../libs/data-assimilation/src/data_assimilation/localization/base.py)
  takes the per-observation min inflation across block members so they share the
  active-observation set; `resolve_row_inflation` (the mask/group ordering around
  it) and `active_observations` (the shared "assimilate this one?" predicate) in
  the same file are called by both `localized_update` and the filter's
  `LETKFAnalysis`, so the stochastic and deterministic local analyses cannot
  drift apart. (Correlation uses `ddof=1` to match the `(N_e-1)`
  covariance denominator — the ratio is the exact sample correlation.)
- **Cost note**: `localized_update` is `jax.vmap` over augmented rows
  (`N_aug` small `N_d×N_d` solves). Cheap for parameter variants; for
  `StateAndParameterESMDA` (large `N_aug`) it is `O(N_aug·N_d²)` memory — the
  paper's grid-block transition-matrix reuse (§3b) is a documented future
  optimization, not yet implemented.

### Reduced SVD/KL state update (optional)
- [reduction.py](../libs/data-assimilation/src/data_assimilation/reduction.py)
  defines `OnlineStateReduction`, taken by the **state-bearing** smoothers via
  the `state_reduction=` constructor arg (default `None` = the full-space
  update, byte-identical to before). When set, the state rows of the augmented
  Kalman vector are replaced by reduced coefficients of an SVD/KL basis
  **refitted each ESMDA iteration** from the forecast ensemble
  (`basis_source`: the `time=0` IC anomalies, or every window frame), and the
  Kalman increment is decoded back onto each member's full state. Parameters
  always keep the global update; incompatible with (state) localization — the
  constructor raises. Theory + implementation notes:
  [docs/reduced_state_da.md](temp/reduced_state_da.md).
- `final_time_smoothing=True` (requires `state_reduction`, in-memory mode
  only) adds one post-loop, un-tempered (`alpha=1`) Kalman update of the state
  at **all window time steps jointly**, reusing the final posterior forecast
  (no extra solve) with the parameters frozen.
- Selected via `smoothing.state_reduction` (`${state_reduction.none|svd|...}`,
  options in `configs/assimilation_settings/state_reduction.yaml`, default
  `none`) + the `smoothing.final_time_smoothing` flag; wired into the
  `state`, `state_and_parameter` and `state_and_dynamic` smoothers in
  `configs/assimilation_settings/smoother.yaml`.

### Multi-window rollout ESMDA
Handled directly inside [scripts/run_smoother.py](../scripts/run_smoother.py)'s window
loop when `assimilation.num_windows > 1`. The full truth (state +
parameters) for every window is simulated up front (or loaded); the loop then,
per window: slices that window's truth observations, adds noise, runs the
smoother, writes `windows/window_{w}_*` files, and feeds the window's final
posterior state in as the next window's `state`. For the **dynamic**
(time-varying) case the next window's prior is
`prior_sampler.extrapolate(posterior, ...)`; for the static case it is just
the posterior.

### Truth source — inline vs. on disk
Truth (state + params) is either **simulated inline**
(`assimilation.truth_dir: null`, the default) or **loaded from a saved
artifact**: a `state.nc`/`params.nc` pair as written by
`scripts/run_forward.py`. `assimilation.truth_start_time` begins the
assimilation horizon partway into a disk truth (drops earlier frames and
rebases that time to `t=0`) to skip a spin-up. Disk truth is read lazily, one
window at a time, so multi-GB truths never load fully. The `ground_truth*`
dirs are gitignored. (The archived spin-up trimming / 32-bit conversion
utilities are in `archive/scripts/adjust_simulations/`.)

### Validation sensors
A case's `obs` block may define a held-out sensor set via
`validation_{x,y,z}_points`. These are **scored but never assimilated** —
`scripts/utils/helper_functions.py::sensor_sets` adds them as the `validation` set,
and `compute_metrics.py` / `visualize_assimilation.py` score and plot them as
an out-of-sample check alongside the assimilated sensors.

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

Truth and prediction samplers keep every configured parameter. The runners'
`params_to_estimate` selects only the prior fields eligible for DA updates:
`null` selects all, and `[]` selects none. Unselected fields are still applied
to the model and saved with the full parameter ensemble. Use a `Constant` to
prescribe a value, or remove its sampler entry to use the model default.
Unselected random priors keep their sampled member values; unselected dynamic
trajectories continue through normal extrapolation between windows.

- **Static** ([src/pyurbanair/static_parameters/](../src/pyurbanair/static_parameters/)) —
  `ParameterSampler` holds a `name -> Distribution` mapping. Each parameter is
  a `Normal` / `Uniform` random prior or a fixed `Constant` (each its own
  `_target_` block), so the same class covers both "sample an ensemble from a
  prior" (`configs/params/static.yaml`) and "use these fixed truth values"
  (`configs/params/static_truth.yaml`, all `Constant`s). Output has an `ensemble`
  dim only (no `time`).
- **Dynamic / time-varying** ([src/pyurbanair/dynamic_parameters/](../src/pyurbanair/dynamic_parameters/)) —
  `AR2RelaxationModel` is the prior model (the former `ar1`,
  `gp_linear_trend`, `ornstein_uhlenbeck` were removed); `HarmonicParameterModel`
  (`configs/params/dynamic_sine.yaml`, `dynamic_cosine.yaml`) generates smooth
  truths. Critically-damped AR(2)
  relaxing toward the external prior `x_ext`; output adds a `time` dim. It also
  exposes `extrapolate(posterior, prediction_times, rng_key)` for the next
  rollout window. Configured by `configs/params/dynamic.yaml` (prior) /
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
  the `prepare._target_` block in [configs/model/pyudales.yaml](../configs/model/pyudales.yaml)
  via `python_or_matlab: python`, which is what
  `pyurbanair.config.hydra_helpers.prepare_udales` passes through.
- Runtime config in `namoptions.<exp>` (edited via `NamoptionsFile`).
- Staggered grid: state has `xt/xm`, `yt/ym`, `zt/zm`. Some plotting
  utilities call `interpolate_grid` to project everything onto a common
  grid before display.
- Inflow profile is configured via the nested `nudging_config` field on
  `cfg.model.forward_model` (`profile_config = {"type": "uniform" |
  "power_law", "alpha": ..., "z_ref": ...}`). `nnudge_meters` (height in m,
  converted to a grid-level count and overriding the raw `nnudge`) is supported
  by `apply_time_varying_inflow` in
  [utils/nudging_utils.py](../libs/pyudales/src/pyudales/utils/nudging_utils.py).
- **Instability watchdog**: `run_single` runs uDALES via
  `run_with_dt_watchdog` ([utils/run_monitor.py](../libs/pyudales/src/pyudales/utils/run_monitor.py))
  instead of bare `subprocess.run`. It tails `run.<exp>.log`, and if the timestep
  `dt` stays below `min_dt` for `patience` consecutive steps (after `warmup_steps`)
  it kills the process tree and raises `CalledProcessError` so the ensemble's
  failure policy can resample — instead of waiting out a slow divergence.
  Configured by the `instability_check` dict on `ForwardModel`
  (`InstabilityCheck.from_config`: `enabled, min_dt, patience, warmup_steps,
  poll_interval_s`).
- The initial inflow speed is written via `_set_infile_value("uini",
  velocity_magnitude)` when params are given (fallback when a time-varying
  `uvel_time.dat` is absent), so the run starts at the requested magnitude rather
  than the template default. pylbm does the same (`uini`).
- `pressure_gradient_magnitude` is the third parameter only this backend
  supports. The ordered per-model schema comes from `resolve_parameter_schema`
  in `hydra_helpers.py` (adds it for `pyudales` only); the sampler configs
  simply include or omit it (`configs/params/static.yaml` carries it as a
  `Constant`, which non-uDALES backends ignore).

### Model-error compensation knobs (cross-model ESMDA)
- Two parameters let the smoother absorb truth↔assim solver misspecification
  instead of corrupting the inflow estimate (see
  [docs/esmda_model_error_parameters.md](temp/esmda_model_error_parameters.md)):
  `vertical_inflow_exponent` (the power-law shear exponent α) and `sgs_constant`
  (the sub-grid-scale mixing constant). Both are static scalars, advertised by
  `resolve_parameter_schema` for every backend, and consumed per-member in each
  backend's `_apply_inflow_settings` (outside the static/time-varying branch).
- **Write sites differ per solver** and the `sgs_constant` value is NOT the same
  quantity across them — they are different closures, intentionally untied:
  - pylbm: α → `uvel_shear.dat` (rewritten via `resolve_profile_config` +
    `write_uvel_shear_file`); SGS → `infile.in` `ivreman smagor` line
    (`apply_sgs_setting`, `const = 2.5*smagorinsky**2` in `m_vreman.F90`).
  - pyudales: α → nudging `profile_config` (`_resolve_nudging_config`); SGS →
    `&NAMSUBGRID cs` (`_apply_sgs_setting`, takes effect under
    `lsmagorinsky=.true.`).
  - pypalm: α → `profile_config` for the driver / `u_profile`; SGS →
    `km_constant` (Option A proxy — a constant eddy diffusivity [m²/s] that
    *replaces* the prognostic SGS-TKE closure, since PALM's `c_0` is hardcoded).
    A fixed `km` also forces `constant_flux_layer = .false.` (PALM rejects the
    pair otherwise — check_parameters PAC0149), so this is a full constant-Km
    regime switch. Note `km_constant` is in m²/s — a *different physical quantity*
    from the dimensionless LBM/uDALES Smagorinsky constants, so a uDALES↔PALM run
    needs PALM-appropriate `sgs_constant` ranges in the prior config, not the
    uDALES `cs` defaults.
- Each write is a no-op when its parameter is absent from `params`, so
  single-model / default runs are unaffected.

### pypalm
- Lazy-imported. All `pypalm.*` `_target_` blocks live exclusively in
  [configs/model/pypalm.yaml](../configs/model/pypalm.yaml), so Hydra only
  triggers the PALM import when that config is instantiated. Non-PALM
  runs never pay the compile cost. (The test asserting this,
  `test_palm_target_does_not_import_for_non_palm_composition`, is in the
  archived `archive/tests/test_hydra_config.py`.)
- Postprocess unifies the vertical staggers (`zu_3d → z`, `w`
  interpolated from `zw_3d`) so all three velocity components share a
  single `z` dim.

### neural_surrogate (learned fourth backend)
> Full stack (data generation, training, rollout, domain decomposition) in
> [docs/neural_surrogates.md](neural_surrogates.md) — read it before editing here.
- Architectures live in
  [architectures/](../libs/neural-surrogates/src/neural_surrogates/architectures/):
  `SimpleConv` (baseline), `UNetConvNeXt`, **`UPT`** (Universal Physics
  Transformer — encoder/approximator/decoder over supernodes + latent tokens),
  **`P3D`** (Holzschuh et al. ICLR 2026 hierarchical conv + windowed-attention
  U-net; wraps the upstream `p3d_surrogate` package, lazy-imported), and
  `DomainDecomposed` (tiles a fixed patch size so one trained model runs on any
  grid sharing its cell spacing). Presets are config groups
  `neural_surrogate/architectures/{unet_convnext,upt,p3d,domain_decomposed}/<size>`.
  `mode=standard` (the default) now pairs `p3d/medium` + `Trainer` + `MSELoss`;
  `mode=domain_decomposition` pairs `domain_decomposed/small` + `PatchTrainer` +
  `DomainDecompositionLoss`. Datasets are under `datasets/` (`transition.py`,
  `patch.py`); the training loop is split into `training/{base,standard,patch}.py`.
- **UPT has two load-bearing knobs — do not flip them off casually**
  ([architectures/upt.py](../libs/neural-surrogates/src/neural_surrogates/architectures/upt.py)):
  - `normalize=True` — z-scores state channels *and* inflow params with buffers
    `state_mean/std`, `param_mean/std` set via `set_normalization(...)`. Raw
    `inflow_angle` (~50× the velocity channels) otherwise swamps the encoder's
    single `input_proj` and the rollout collapses to a constant (see the
    docstring in `upt.py`, which marks both knobs "load-bearing").
  - `predict_residual=True` — predicts `state_{t+1} − state_t` (added back to the
    input), keeping near-identity (correct for a slow transient) as the default.
  - `attention_type` selects the self-attention impl (`dot_product` default;
    `efficient`/`linformer`/`transsolver` via `attention_kwargs`); a per-geometry
    `_geom_cache` avoids rebuilding the supernode neighbour graph each step.
- Normalization stats are computed by `_compute_normalization_stats` in
  [training/data_utils.py](../libs/neural-surrogates/src/neural_surrogates/training/data_utils.py)
  (streamed in f64 over fluid cells only) and **baked into the checkpoint** via
  `model.set_normalization(...)`, so no separate stats file is needed at inference.
- The Tadpole track (`TadpoleAE` pre-training, the `TadpoleTimeStepper` DFT
  fine-tune, and the `TadpoleLatentGenerator` / `spinup_source: generative`
  cold start that samples the window-0 field instead of running a CFD spin-up)
  is documented in [docs/neural_surrogates.md](neural_surrogates.md) Parts G–I;
  the generative spin-up contract (regeneration on every cold forecast, no
  initial-knot pinning, joint-state smoothers rejected) is Part I, §40.

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
5. Add a new `configs/model/<name>.yaml` mirroring
   [configs/model/pylbm.yaml](../configs/model/pylbm.yaml): `name`,
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
   the archived `test_palm_target_does_not_import_for_non_palm_composition`),
   and a tiny overlay in `tests/configs/model/<name>_tiny.yaml`.

### Add a new parameter
- Add the parameter to the sampler configs in
  [configs/params/](../configs/params/): a `Distribution` block in `static.yaml`
  (prior) and `static_truth.yaml` (truth) and/or under `external_parameters`
  in `dynamic.yaml` / `dynamic_truth.yaml`. The samplers pick up any key in the
  mapping — no Python change needed for the sampling side.
- A **constant-in-time** parameter that must still be ESMDA-estimated *alongside*
  the time-varying inflow goes under `static_parameters:` (not
  `external_parameters:`) in `dynamic.yaml` / `dynamic_truth.yaml`. The AR(2)
  sampler draws it once (window 0), emits it with no `time` dim, and the
  time-varying smoother passes time-less vars through its flatten/unflatten
  unchanged — so it can be updated jointly when selected for estimation and is
  carried (not re-randomized) across windows. See
  [docs/esmda_model_error_parameters.md](temp/esmda_model_error_parameters.md) §6.
- If the parameter is backend-specific (like `pressure_gradient_magnitude`),
  extend `resolve_parameter_schema` in
  [src/pyurbanair/config/hydra_helpers.py](../src/pyurbanair/config/hydra_helpers.py).
- Extend each backend's `_apply_inflow_settings` /
  `apply_inflow_settings` in `utils/params_utils.py` so the value gets
  written to that solver's input format. Read the value with
  `get_param_value(params, name)` and **no-op when it is absent** so single-model
  / default runs stay byte-identical.
- **uDALES gotcha:** Discrepancy coefficients use their own extractor/writer
  (`utils/discrepancy_utils.py`); do not add them to the inlet whitelist.
  For inlet parameters, `pyudales/utils/params_utils.py` keeps a whitelist
  (`INFLOW_PARAM_NAMES`) of variables that survive `extract_inflow_params` /
  `merge_params`. A new variable not in it is *silently dropped* before reaching
  the solver — add it there. pylbm and pypalm read `params` directly.
- Time-varying support: implement reading in the backend (e.g. pylbm's
  `write_uvel_time_file`).
- To let a run **choose which parameters ESMDA estimates**, set
  `assimilation.params_to_estimate` in [configs/assimilation.yaml](../configs/assimilation.yaml) (a list,
  `null` for all prior parameters, or `[]` for none). All configured prior and
  truth parameters reach their corresponding forward models. Unselected prior
  fields retain their sampled realization without DA updates; dynamic fields
  still extrapolate normally. Omit a parameter from its sampler config to use
  the corresponding model's default instead.

### Add a new ESMDA variant
- Subclass `_BaseESMDA` in
  [smoothing/esmda.py](../libs/data-assimilation/src/data_assimilation/smoothing/esmda.py).
- Override `_one_step(params, obs, state)` — choose what's in the
  augmented vector, call `self._compute_kalman_update(...)`, return
  `(updated_state_or_None, updated_params)`.
- Add a new entry to
  [configs/assimilation_settings/smoother.yaml](../configs/assimilation_settings/smoother.yaml)
  with your class's `_target_` and the shared fields wired via
  `${smoothing.num_steps}` / `${smoothing.alpha}` / `${smoothing.localization}`.
  No new script is needed — [scripts/run_smoother.py](../scripts/run_smoother.py)
  instantiates whatever `cfg.smoothing.smoother` resolves to, and you select it
  with `'smoothing.smoother=${smoother.<name>}'`. Teach
  `scripts/utils/inconsistency_check.py` which priors it pairs with.

### Add a new localization strategy
- Subclass `BaseLocalization` in
  [localization/](../libs/data-assimilation/src/data_assimilation/localization/)
  and implement `inflation_factors(aug_dev, pred_obs_dev, row_coords=None,
  obs_coords=None)`. Return `(N_aug, N_d)`: `1.0` keeps an observation for a row,
  `>1` tapers it, `jnp.inf` excludes it. Reuse `taper_inflation(distance,
  truncation, beta, max_inflation)` for the Vossepoel Eq. 9–10 taper. If the
  strategy needs grid/sensor geometry (like `DistanceLocalization`), set the
  class attribute `requires_coordinates = True`; the state-bearing smoothers then
  pass `row_coords`/`obs_coords` (and it only works on those smoothers). The
  shared `localized_update` handles the Kalman math.
- Add an entry to
  [configs/assimilation_settings/localization.yaml](../configs/assimilation_settings/localization.yaml)
  (`<name>: {_target_: ..., ...}`) and select it with
  `'smoothing.localization=${localization.<name>}'` (or
  `filtering.localization`). Every smoother already receives it via
  `localization: ${smoothing.localization}`; `${localization.none}` gives the
  global update.

### Add a new run script
- Place under [scripts/](../scripts/), mirror an existing one. The
  shape is `def run(cfg)` + a thin `@hydra.main(config_path="../configs", ...)`
  `main`; post-processing scripts are plain CLIs taking a run dir.
- Call `check_config(cfg, "<workflow>")` from `scripts/utils/inconsistency_check.py` first
  (add the workflow's rules there).
- Use `hydra.utils.instantiate(cfg.model.forward_model, ...)` for backend
  construction; share only what several scripts need via `scripts/utils/helper_functions.py`.
- Write under `cfg.paths.results_dir` and save the composed `config.yaml`
  there.
- Add a test under `tests/scripts/` that composes the entry point with a
  `tests/configs/` overlay (`compose(..., "+test=<name>", root=tmp_path)`) and
  calls `run(cfg)` directly; add a workflow chain under `workflows/` if useful.

## 9. Operational defaults / scaling

- `.temp/` is the default scratch directory. Every backend writes its
  per-experiment dir and per-member dirs underneath. The default
  `paths.results_dir` is `.temp/${model.name}` (`forward.yaml`) and
  `experiment_dir` is the per-machine scratch (`.temp` locally, see
  `common.yaml`); `assimilation.yaml` sets `results_dir` to
  `.temp/${truth_model.name}_to_${assim_model.name}` and each DA script appends
  its workflow name.
- Tests shrink runs with overlays in `tests/configs/`; production edits do not retune them.
  Run `pixi run -e dev py.test`, `test-integration`, or `test-all` as appropriate.
- Pre-commit hooks (`black`, `isort`, `mypy`) installed via
  `pixi run pre-commit`. They are **not enforced** server-side; commits
  can bypass.
- **Ensemble scaling on this hardware** is DRAM-bandwidth-bound past
  ~4 workers (see [docs/ensemble_scaling.md](temp/ensemble_scaling.md) and
  the `ensemble` budget in `configs/common.yaml`).
  Don't blindly raise `num_parallel_processes` past 8 — re-benchmark
  first.
- `pyurbanair` deliberately uses `forkserver` not `fork` for parallel
  workers (JAX threads + bare-fork = deadlock).

## 10. Things that look optional but aren't

- Every state is expected to have a `time` dim, even when length 1. Some
  helpers (`extract_2d_slice`, observation operators) assume this.
- ESMDA `_analysis` feeds the Kalman-updated initial condition forward
  between iterations (state-bearing variants) so the state estimate is actually
  used — the next forecast warm-starts from the analyzed IC. This intentionally
  skips a fresh spin-up after iteration 0, so the analyzed field must be a usable
  warm-start state for the forward model (the cross-window carry-over already
  assumes this). Parameter-only variants return no state and keep the caller's
  pinned IC, so they are unaffected. (Before, the analyzed IC was discarded and
  every forecast re-ran from the pinned IC — which made state estimation and
  state localization a no-op.)
- `BaseEnsembleForwardModel._set_save_mode` controls whether the
  ensemble result is concatenated or written to disk. Forward-model
  child `results_dir` is overridden to match the ensemble's at parallel
  dispatch time — don't bake per-member paths into ensemble code.
- `pylbm` runs `subprocess` with `stderr=DEVNULL` unless `verbose=True`.
  Silent crashes here are the #1 mystery when LBM "produces no output".

## 11. Fastest path to "where is X?"

| You want to change… | Look here |
|---|---|
| Run-time parameters (sim time, ensemble size, ESMDA steps) | [configs/](../configs/) — see §5 |
| Geometry / domain / sensor layout | [configs/case/](../configs/case/) (`case=<name>` bundles `domain`+`obs`+`geometry`) |
| Per-backend `model_name → class` wiring | [configs/model/](../configs/model/) (`forward_model._target_` / `ensemble_model._target_` blocks) |
| Hydra `_target_` helpers (prepare, clean, obs operator) | [src/pyurbanair/config/hydra_helpers.py](../src/pyurbanair/config/hydra_helpers.py) |
| What an ensemble member does in parallel | [src/pyurbanair/base_ensemble_forward_model.py](../src/pyurbanair/base_ensemble_forward_model.py) |
| How a solver consumes params | `libs/<solver>/src/<solver>/utils/params_utils.py` |
| ESMDA Kalman update / variants | [libs/data-assimilation/src/data_assimilation/smoothing/esmda.py](../libs/data-assimilation/src/data_assimilation/smoothing/esmda.py) |
| Which ESMDA mode runs (smoother/prior/windows) | [scripts/run_smoother.py](../scripts/run_smoother.py) + [configs/assimilation.yaml](../configs/assimilation.yaml), [configs/assimilation_settings/smoother.yaml](../configs/assimilation_settings/smoother.yaml) |
| Which assimilation algorithm runs at all (ESMDA / EnKF / hybrid) | the entry points in §6 — `run_smoother.py`, `run_filtering.py`, `run_hybrid.py` |
| How sensors map to grid points | [libs/data-assimilation/src/data_assimilation/observation_operator.py](../libs/data-assimilation/src/data_assimilation/observation_operator.py) |
| Per-window rollout logic | the window loops of `run_smoother.py` / `run_filtering.py` / `run_hybrid.py`; `run_forward.py`'s `forward.rollout_steps` loop |
| Parameter samplers (static + dynamic) | [src/pyurbanair/static_parameters/](../src/pyurbanair/static_parameters/), [src/pyurbanair/dynamic_parameters/](../src/pyurbanair/dynamic_parameters/), [configs/params/](../configs/params/) |
| Truth source / spin-up skip | `assimilation.truth_dir` + `assimilation.truth_start_time` in [assimilation.yaml](../configs/assimilation.yaml); `scripts/utils/helper_functions.py` (`make_truth`, `open_truth`) |
| Localization (correlation/distance/none) / grid-block grouping | [localization/](../libs/data-assimilation/src/data_assimilation/localization/) (`correlation.py`, `distance.py`), [configs/assimilation_settings/localization.yaml](../configs/assimilation_settings/localization.yaml) (`block_grouping`, state-only) |
| Reduced SVD/KL state update / final trajectory smoothing | [reduction.py](../libs/data-assimilation/src/data_assimilation/reduction.py), [configs/assimilation_settings/state_reduction.yaml](../configs/assimilation_settings/state_reduction.yaml), [docs/reduced_state_da.md](temp/reduced_state_da.md) |
| Neural-surrogate architectures (UPT etc.) | [architectures/](../libs/neural-surrogates/src/neural_surrogates/architectures/), [configs/surrogate/architectures.yaml](../configs/surrogate/architectures.yaml) |
| uDALES instability / dt-collapse handling | [libs/pyudales/src/pyudales/utils/run_monitor.py](../libs/pyudales/src/pyudales/utils/run_monitor.py) (`instability_check`) |
| DA metrics + diagnostic plots (RMSE/CRPS, sensor series) | [libs/evaluation/src/evaluation/scores.py](../libs/evaluation/src/evaluation/scores.py) (`compute_parameter_metrics`, `compute_sensor_metrics`) + [figures.py](../libs/evaluation/src/evaluation/figures.py) (`plot_parameter_error`, `plot_sensor_timeseries`) |
| Validation (held-out) sensors | `obs.validation_{x,y,z}_points` in [configs/case/](../configs/case/) + `scripts/utils/helper_functions.py::sensor_sets` |
| Test fixture composition | [tests/conftest.py](../tests/conftest.py) (`compose` + `tests/configs/` overlays; legacy `compose_test_cfg` / `compose_module_cfg`) |
| Dynamic multi-window ESMDA theory / config | [docs/esmda_dynamic_multiwindow.md](temp/esmda_dynamic_multiwindow.md) |
| Benchmark / scaling findings | [docs/ensemble_scaling.md](temp/ensemble_scaling.md) (the one-off benchmark scripts were removed; recover from git history to re-run) |
