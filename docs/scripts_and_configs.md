# pyurbanair — Scripts and Configuration Reference

Detailed reference for [`conf/`](../conf/) (Hydra configs) and [`scripts/`](../scripts/)
(executable entry points). Complements the high-level summary in
[`codebase_guide.md §5`](codebase_guide.md) and the existing
[`conf/README.md`](../conf/README.md) overview — read those first for
orientation, then return here for field-level detail.

---

## Part 1 — `conf/`: Hydra configuration tree

### Overview

The configuration tree has exactly **five primary run entry points**, each
self-contained (they inline the shared base rather than pulling separate
`paths.yaml`/`time.yaml`/`ensemble.yaml` files):

| Entry point | Script | What it adds |
|---|---|---|
| [`conf/run_forward_model.yaml`](../conf/run_forward_model.yaml) | `run_forward_model.py` | `case` + single `model@model` mount + single `params` mount |
| [`conf/run_esmda.yaml`](../conf/run_esmda.yaml) | `run_esmda.py` | same base + `esmda:` scalars + double model mount (`@truth_model`/`@assim_model`) + double params mount (`@truth_params`/`@prior_params`) |
| [`conf/run_filtering.yaml`](../conf/run_filtering.yaml) | `run_filtering.py` | same base + `filtering:` scalars + the `filtering/*` groups + the same double model/params mounts (static params only) |
| [`conf/run_filter_smoothing.yaml`](../conf/run_filter_smoothing.yaml) | `run_filter_smoothing.py` | same base + `filter_smoothing:` scalars + the `filter_smoothing/*` groups + the same double model/params mounts (time-varying prior required) |
| [`conf/compare_models.yaml`](../conf/compare_models.yaml) | `compare_models.py` | same base as `run_forward_model` + `compare:` scalars + an *N-way* model mount (`model@models.<name>`) and named parameter-scenario mounts (`params@parameter_scenarios.<name>`) |

A third entry point, [`conf/neural_surrogate/training_data.yaml`](../conf/neural_surrogate/training_data.yaml),
extends `run_forward_model` with dataset-shape fields for surrogate data
generation. The surrogate train/test scripts use
[`conf/neural_surrogate/training.yaml`](../conf/neural_surrogate/training.yaml)
and [`conf/neural_surrogate/testing.yaml`](../conf/neural_surrogate/testing.yaml).

### `compare_models` diagnostics

[`scripts/compare_models.py`](../scripts/compare_models.py) deliberately treats
instantaneous LES-field error as a qualitative diagnostic, rather than the sole
cross-model score. In addition to snapshots and sensor series it writes
time-windowed fluid-cell mean/spread maps, STL-derived upstream/canopy/wake
profiles, PDFs/CDFs, sensor spectra/autocorrelations, and wake-recovery metrics.
`compare.analysis` controls the statistical window, compact full-height grid,
wall-cell exclusion, distribution sampling, and wake-recovery threshold. Its
regions are inferred from the selected STL and assume the dominant flow is in
the +x direction.

The **mode** of a run is the cross product of its config groups — no separate
mode file is required. `run_esmda.py` handles every former assimilation script
via `esmda/smoother` × `params@prior_params` × `esmda.num_assimilation_windows`.

---

### 1.1 Inlined base blocks

Both entry points inline these shared namespaces directly in their YAML body
rather than pulling them from separate files. The table below summarises each.

#### `paths:`

| Field | Default (fwd / esmda) | Purpose |
|---|---|---|
| `results_dir` | `results/${model.name}` / `/export/scratch2/ntm/${truth_model.name}_to_${assim_model.name}` | Top-level output root; Hydra run dir is nested under it. |
| `experiment_dir` | `${oc.env:PWD}/.temp_${model.name}` | Absolute scratch dir for CFD solvers (they `chdir` into subdirs here). Resolved from `$PWD` rather than Hydra's runtime cwd so bare `compose()` in tests works. |
| `base_results_dir` | `.temp_${model.name}` | Fallback used by `resolve_output_dir()` when Hydra is not initialized (direct `run(cfg)` calls from tests). |

#### `time:`

| Field | Default | Purpose |
|---|---|---|
| `seconds_per_knot` | `30.0` (fwd) / `60.0` (esmda) | Spacing (seconds) between AR(2) parameter knots. The per-window horizon (`simulation_time`/`output_frequency`/`spinup_time`) lives in the `case` group, not here. |

#### `ensemble:`

| Field | Default | Purpose |
|---|---|---|
| `ensemble_size` | 64 (fwd) / 50 (esmda) | Number of ensemble members. |
| `num_parallel_processes` | 1 | `ProcessPoolExecutor` worker count. |
| `num_cpus_per_process` | 15 (fwd) / 1 (esmda) | CPU affinity per worker (passed to `cpu_pinning`). |
| `failure.policy` | `resample_from_successes` | What to do when a member fails: `raise` or `resample_from_successes`. |
| `failure.jitter_scale` | 0.05 | Relative std of Gaussian jitter applied to donor params when resampling. |
| `failure.seed` | 0 | RNG seed for the resampling draw. |

#### `esmda:` (run_esmda only)

| Field | Default | Purpose |
|---|---|---|
| `num_steps` | 3 | Number of ESMDA (Kalman update) iterations per window. |
| `alpha` | `${.num_steps}` | Inflation denominator; defaults to `num_steps` (standard ESMDA). |
| `num_assimilation_windows` | 3 | Number of sequential assimilation windows (1 = single window). |
| `seed` | 42 | JAX RNG seed. |
| `obs_error_std` | 0.25 | Diagonal observation-error standard deviation (same for all sensors). |
| `localization` | `null` | Set by the `esmda/localization` group; `null` = global (unlocalized) update. |
| `state_reduction` | `null` | Set by the `esmda/state_reduction` group; `null` = full-space update. |
| `final_time_smoothing` | `false` | Post-loop Kalman update of the full trajectory (requires `state_reduction`). |

#### `filtering:` (run_filtering only)

| Field | Default | Purpose |
|---|---|---|
| `num_cycles` | 2 | Number of filter cycles; each forecasts one segment of `time.simulation_time` and applies ONE full-weight analysis. |
| `seed` | 42 | JAX RNG seed. |
| `obs_error_std` | 0.25 | Diagonal observation-error standard deviation (same for all sensors). |
| `mode` | `joint` | Which blocks the analysis updates: `state` \| `parameter` \| `joint`. The parameter-updating modes (`parameter`/`joint`) require spread maintenance (evolution or inflation). |
| `analysis` | (group) | Set by `filtering/analysis` (default `stochastic`); `etkf*`/`letkf*` are the deterministic ensemble transforms and constrain `localization` (§1.8). |
| `localization` | (group) | Set by `filtering/localization` (default `none`). |
| `state_reduction` | (group) | Set by `filtering/state_reduction` (default `none`); current/streaming SVD requires `mode=state|joint` and global localization. |
| `inflation` | (group) | Set by `filtering/inflation` (default `rtps`). |
| `parameter_evolution` | (group) | Set by `filtering/evolution` (default `none`). |
| `filter` | `EnsembleKalmanFilter` block | The composed filter `_target_`; normally left alone. |

#### `filter_smoothing:` (run_filter_smoothing only)

| Field | Default | Purpose |
|---|---|---|
| `num_cycles` | 8 | Number of cycles `L` in the window; each forecasts one segment of `time.simulation_time`. Also the number of parameter knots the outer update estimates (plus the trailing knot). |
| `num_steps` | 4 | Number of outer ESMDA iterations `N_a`; each is one full inner filter pass through the window plus one tempered trajectory update. |
| `alpha` | `null` | Tempering denominator; `null` = `num_steps` (equal weights, the standard ESMDA schedule). |
| `seed` | 42 | JAX RNG seed. |
| `obs_error_std` | 0.25 | Diagonal observation-error standard deviation (same for all sensors, per cycle). |
| `common_inner_noise` | `true` | Reuse one inner RNG key across iterations, so the map `Θ → D` is deterministic and the outer cross-covariances are not diluted by fresh Monte Carlo noise. `false` = independent draws. The outer perturbed-observation draws always use fresh subkeys. |
| `inner_analysis` | (group) | Set by `filter_smoothing/inner_analysis` (default `stochastic`); the same `AnalysisScheme` classes as `filtering/analysis`, and they constrain `inner_localization` the same way (§1.9). |
| `inner_localization` | (group) | Set by `filter_smoothing/inner_localization` (default `none`). |
| `inner_inflation` | (group) | Set by `filter_smoothing/inner_inflation` (default `rtps`). |
| `temporal_localization` | (group) | Set by `filter_smoothing/temporal_localization` (default `none`); localizes the **outer** trajectory update in time. |
| `smoother` | `FilterSmoothingESMDA` block | The composed algorithm `_target_`, wiring the four groups plus `num_steps`/`alpha`/`common_inner_noise`; normally left alone. |

#### `run:`

| Field | Default | Purpose |
|---|---|---|
| `skip_viz` | `false` | Skip all figure/animation output. |
| `results_dir` | `null` | Explicit override for the per-run output directory (null = use Hydra's auto dir). |
| `ensemble` | `false` (fwd only) | Run an ensemble rather than a single member. |
| `rollout_steps` | 0 (fwd only) | Number of extra windows to roll forward. |
| `ensemble_save_on_disk` | `false` (fwd) / `true` (esmda) | Write per-member NetCDFs instead of in-memory ensemble Dataset. |
| `truth_dir` | `null` | Path to a saved `state.nc`/`params.nc` truth artifact; `null` = simulate inline. |
| `truth_start_time` | `null` | Drop truth frames before this time (seconds) and rebase. |
| `save_prior_state` | `false` (esmda only) | Persist the per-window prior ensemble state (large; off by default). |

---

### 1.2 Config group: `case/`

Each case bundles everything experiment-specific into one self-contained file —
domain bounds, grid, geometry paths, sensor layout, and per-window time horizon.
**Adding a new experiment = adding one file here.**

Currently two cases are defined:

#### [`case/xie_and_castro.yaml`](../conf/case/xie_and_castro.yaml)

Benchmark staggered cube array (Xie & Castro 2008). Default case.

- **Domain**: `nx=30, ny=40, nz=16`; bounds `x ∈ [-20, 40]`, `y ∈ [0, 80]`, `z ∈ [0, 32]` m.
  Upstream inflow region (`x < 5`) in front of the cube array.
- **Geometry**: shared STL at `examples/xie_and_castro/xie_castro_2008_STL.stl` for all
  three CFD backends.
- **Sensors** (`obs.mode: points`): 6 assimilation sensors in open N-S lanes at street level
  (`z=2 m`), plus 2 held-out validation sensors. States observed: `u, v`.
  Aggregation: `mean` over `interval_seconds=60.0` intervals.
- **Time**: `simulation_time=300 s`, `output_frequency=2 s`, `spinup_time=50 s` per window.

#### [`case/barcelona.yaml`](../conf/case/barcelona.yaml)

Real Barcelona urban geometry (~900 × 870 × 85 m domain).

- **Domain**: `nx=224, ny=224, nz=32` (4 m resolution); bounds `[0,896] × [0,896] × [0,128]` m.
  A 2 m resolution option is commented in the file.
- **Geometry**: shared STL at `examples/barcelona/buildings.stl`; uDALES uses
  `examples/udales/barcelona/` (symlink); precomputed IBM geometry bundle under
  `geometry.udales_precomputed_geom_dir` avoids re-running the expensive STL→IBM
  classifier (~30+ min on 422k facets).
- **Sensors**: 6 assimilation sensors on a 3×2 grid at `z=3 m` (street-canyon level),
  verified ≥ 6 m from any building. States observed: `u, v, w`.
  Aggregation: `mean` over `interval_seconds=30.0` intervals.
- **Time**: `simulation_time=1200 s`, `output_frequency=10 s`, `spinup_time=500 s` per window.

---

### 1.3 Config group: `model/`

Each file wires one CFD backend under a runtime package. Every file provides the
same four top-level keys: `name`, `solver_name`, `forward_model._target_`,
`ensemble_model._target_`, `prepare._target_`. The ensemble model also reads
`${ensemble.failure}` directly from the inlined base.

| File | Backend | `_target_` classes |
|---|---|---|
| [`model/pylbm.yaml`](../conf/model/pylbm.yaml) | Lattice Boltzmann (CUDA optional) | `pylbm.forward_model.ForwardModel` + `pylbm.ensemble_forward_model.EnsembleForwardModel` |
| [`model/pyudales.yaml`](../conf/model/pyudales.yaml) | uDALES v2.2.0 (staggered grid) | `pyudales.forward_model.ForwardModel` + `pyudales.ensemble_forward_model.EnsembleForwardModel` |
| [`model/pypalm.yaml`](../conf/model/pypalm.yaml) | PALM model system (lazy import) | `pypalm.forward_model.ForwardModel` + `pypalm.ensemble_forward_model.EnsembleForwardModel` |
| [`model/neural_surrogate.yaml`](../conf/model/neural_surrogate.yaml) | Learned one-step surrogate | `neural_surrogates.NeuralSurrogateForwardModel` + `neural_surrogates.NeuralSurrogateEnsembleForwardModel` |

**Notable per-model fields:**

- **pylbm**: `cuda: true`, `verbose: false` (flip to surface swallowed stderr),
  `profile_config.type: power_law`, `boundary_condition: inflow_outflow`.
- **pyudales**: `ncpu: 25` (MPI ranks), `nudging_config` (nudging height, profile type),
  `instability_check` (dt-watchdog: kills diverging runs early so the ensemble
  resamples), `precomputed_geom_dir` (IBM geometry cache).
- **pypalm**: `compile: false` (PALM is pre-compiled; `compile` is a no-op for PALM),
  `ncpu: 14` (must divide `domain.nx` for the multigrid solver),
  `boundary_condition: periodic|inflow_outflow`, and `nudging_config` —
  `enabled` (default `true`; `false` restores the old un-driven periodic
  staging), `tnudge` (relaxation timescale in s, default 15.0),
  `nnudge_meters` (no nudging below this height, default 4.0), and
  `profile_config` (vertical shear shape; also used under `inflow_outflow`).
  Under `periodic` BCs these drive PALM's nudging scheme so the domain is
  relaxed toward the `inflow_angle`/`velocity_magnitude` wind — the same
  physics and parameter meaning as pyudales's periodic nudging. See
  [docs/pypalm.md §8](pypalm.md).
- **neural_surrogate**: `model_dir` (checkpoint folder written by `train_neural_surrogate.py`),
  `spinup_source: forward_model|training_data` (cold-start source),
  `spinup_forward_model` (a nested uDALES config for the CFD cold start),
  `default_params` (constant fallbacks for params the caller omits),
  `_recursive_: false` (surrogate builds its spin-up backend itself).
  Uses `solver_name: pylbm` (regular-grid observation mapping) regardless of the
  spin-up backend.

**Double-mount for ESMDA.** `run_esmda.yaml` mounts the `model/` group twice:
`model@truth_model=<name>` and `model@assim_model=<name>`. Selecting different
backends for truth and assimilation introduces genuine model error. The
`solver_name` field tells the observation operator which grid convention to use.

---

### 1.4 Config group: `params/`

Each file is a standalone `_target_` block instantiated directly as either a
static sampler (`ParameterSampler`) or a time-varying sampler
(`AR2RelaxationModel`). Both expose `sample(ensemble_size) → xarray.Dataset`.

| File | Instantiates | Purpose |
|---|---|---|
| [`params/static.yaml`](../conf/params/static.yaml) | `pyurbanair.static_parameters.ParameterSampler` | Assimilation **prior**: Normal priors for `inflow_angle`, `velocity_magnitude`, `vertical_inflow_exponent`, `sgs_constant`; `pressure_gradient_magnitude` as a `Constant`. |
| [`params/static_truth.yaml`](../conf/params/static_truth.yaml) | `pyurbanair.static_parameters.ParameterSampler` | **Truth** generator: all `Constant` distributions (exact fixed values). Avoids inverse crime. |
| [`params/dynamic.yaml`](../conf/params/dynamic.yaml) | `pyurbanair.dynamic_parameters.ar2_relaxation.AR2RelaxationModel` | Time-varying **prior**: AR(2) relaxation for `inflow_angle` + `velocity_magnitude` (both get a `time` dim), plus static `vertical_inflow_exponent`/`sgs_constant` entries estimated jointly. |
| [`params/dynamic_truth.yaml`](../conf/params/dynamic_truth.yaml) | `AR2RelaxationModel` | Time-varying **truth**: same AR(2) structure but different seed to avoid inverse crime. |
| [`params/dynamic_sine.yaml`](../conf/params/dynamic_sine.yaml) | `pyurbanair.dynamic_parameters.harmonic.HarmonicParameterModel` | Deterministic sine forcing for controlled forward comparisons. |
| [`params/dynamic_cosine.yaml`](../conf/params/dynamic_cosine.yaml) | `HarmonicParameterModel` | A second deterministic cosine forcing scenario. |

**Key fields in `dynamic.yaml`:**
- `correlation_length: 100.0` — AR(2) decay length (seconds).
- `seconds_per_knot: ${time.seconds_per_knot}` — knot spacing interpolated from
  the shared `time` block.
- `external_parameters` — time-varying params with `_target_` Normal distributions.
- `static_parameters` — model-error knobs (`vertical_inflow_exponent`, `sgs_constant`)
  that ride in the same Dataset but carry no `time` dim; drawn once (window 0)
  and refined across windows.

**Deterministic profiles.** `dynamic_sine` and `dynamic_cosine` use
`HarmonicParameterModel`. Each entry in `profiles` has `waveform` (`sine` or
`cosine`), `offset`, `amplitude`, `frequency` in Hz, optional `phase` in radians,
and optional `min` / `max` clips. The series is identical for every ensemble
member and continues on a global clock through rollouts, making it suitable for
a controlled solver comparison rather than stochastic parameter inference.

**Mounting in ESMDA.** `run_esmda.yaml` mounts this group twice:
`params@truth_params=static_truth|dynamic_truth` and
`params@prior_params=static|dynamic`. The truth and prior never share a generative
process (the anti-inverse-crime design).

---

### 1.5 Config group: `esmda/smoother/`

Selects the DA variant. The `esmda/smoother` group default in `run_esmda.yaml` is
`dynamic`. The `_target_` is the one genuinely mode-specific field; all shared
fields are wired via `${esmda.*}` interpolation.

> **Naming note:** The actual filenames are `static`, `state`, `state_and_parameter`,
> `dynamic`, `state_and_dynamic`. The `run_esmda.yaml` header and docstring use
> these names. The `run_esmda.py` docstring also uses these names as the CLI values.

| File | `_target_` class | What it estimates |
|---|---|---|
| [`esmda/smoother/static.yaml`](../conf/esmda/smoother/static.yaml) | `data_assimilation.smoothing.esmda.ParameterESMDA` | Static scalar parameters only. |
| [`esmda/smoother/state.yaml`](../conf/esmda/smoother/state.yaml) | `data_assimilation.smoothing.esmda.StateESMDA` | State only; static parameters are held fixed. Also wires `state_reduction` + `final_time_smoothing`. |
| [`esmda/smoother/state_and_parameter.yaml`](../conf/esmda/smoother/state_and_parameter.yaml) | `data_assimilation.smoothing.esmda.StateAndParameterESMDA` | Joint time=0 state IC + static parameters. Also wires `state_reduction` + `final_time_smoothing`. |
| [`esmda/smoother/dynamic.yaml`](../conf/esmda/smoother/dynamic.yaml) | `data_assimilation.smoothing.esmda.TimeVaryingParameterESMDA` | Time-varying (AR(2)) parameters only. Has `pin_initial_time_point: true` (re-toggled per window by `run_esmda.py` for continuity). |
| [`esmda/smoother/state_and_dynamic.yaml`](../conf/esmda/smoother/state_and_dynamic.yaml) | `data_assimilation.smoothing.esmda.StateAndTimeVaryingParameterESMDA` | Joint time=0 state IC + time-varying parameters. Pair with `params@prior_params=dynamic`. |

---

### 1.6 Config group: `esmda/localization/`

Controls Kalman update localization. Default is `none` (global unlocalized update).
Each file uses `# @package esmda` so it sets `esmda.localization`. Every smoother
wires `localization: ${esmda.localization}`.

| File | Strategy | Key fields |
|---|---|---|
| [`esmda/localization/none.yaml`](../conf/esmda/localization/none.yaml) | Global update (`localization: null`) | — |
| [`esmda/localization/correlation.yaml`](../conf/esmda/localization/correlation.yaml) | `CorrelationLocalization` — excludes observations by ensemble correlation | `truncation_correlation: 0.35`, `tapering_beta: 0.5`, `max_inflation: 8.0`, `block_grouping: true` |
| [`esmda/localization/distance.yaml`](../conf/esmda/localization/distance.yaml) | `DistanceLocalization` — excludes by Euclidean sensor–gridpoint distance | `localization_radius: 10.0`, `tapering_beta: 0.5`, `max_inflation: 4.0`, `block_grouping: true`, `horizontal_only: false` |

Correlation localization applies to every estimated state/parameter row.
Distance localization applies to state rows only and keeps joint parameter rows
on the global update. `distance` requires a state-bearing smoother
(`state`, `state_and_parameter`, or `state_and_dynamic`) and is incompatible with
`state_reduction`. See [codebase_guide.md §6](codebase_guide.md) for the math.

---

### 1.7 Config group: `esmda/state_reduction/`

Controls reduced-basis state update. Default is `none`. Only the state-bearing
smoothers consume this field; incompatible with state localization.

| File | Strategy | Key fields |
|---|---|---|
| [`esmda/state_reduction/none.yaml`](../conf/esmda/state_reduction/none.yaml) | Full-space update (`state_reduction: null`) | — |
| [`esmda/state_reduction/svd.yaml`](../conf/esmda/state_reduction/svd.yaml) | `OnlineStateReduction` — online SVD/KL basis refitted each ESMDA iteration | `energy_fraction: 0.99`, `max_rank: null`, `basis_source: initial_condition\|window_snapshots`, `snapshot_stride: 1` |

See [`docs/reduced_state_da.md`](temp/reduced_state_da.md) for theory and
implementation notes.

---

### 1.8 Config groups: `filtering/*` (run_filtering only)

The sequential-filter counterparts of the `esmda/*` groups. Each file uses
`# @package filtering` so it sets the matching `filtering.*` field.

| Group | Options (default first) | Sets |
|---|---|---|
| [`filtering/analysis/`](../conf/filtering/analysis/) | `stochastic`, `etkf`, `etkf_tsvd`, `letkf`, `letkf_tsvd` | `filtering.analysis` — the update math: perturbed-observation `StochasticEnKFAnalysis`, or a deterministic `ETKFAnalysis` (global) / `LETKFAnalysis` (per block). `etkf*` REQUIRES `filtering/localization=none`; `letkf*` REQUIRES a non-null localization |
| [`filtering/localization/`](../conf/filtering/localization/) | `none`, `correlation`, `distance` | `filtering.localization` — the same strategies as `esmda/localization` (reused unchanged); `distance` needs `filtering.mode=state\|joint` |
| [`filtering/state_reduction/`](../conf/filtering/state_reduction/) | `none`, `svd_current`, `svd_streaming` | `filtering.state_reduction` — optional final-forecast-state analysis basis; requires `filtering.mode=state\|joint` and `filtering/localization=none` |
| [`filtering/inflation/`](../conf/filtering/inflation/) | `rtps`, `none`, `multiplicative`, `rtpp` | `filtering.inflation` — ensemble spread maintenance |
| [`filtering/evolution/`](../conf/filtering/evolution/) | `none`, `random_walk` | `filtering.parameter_evolution` — the parameters' forecast model between cycles |

See [data_assimilation.md](data_assimilation.md) for the filtering library
itself (`BaseFilter` / `EnsembleKalmanFilter`, cycle semantics, diagnostics).

The analysis options carry a declared `localization_policy` that
`BaseFilter.__init__` enforces, so an unusable pair fails before the first
forecast rather than running a global update under a localized name:
`stochastic` is `optional`, `etkf*` `forbidden`, `letkf*` `required`. Selecting
a non-default analysis therefore means selecting its localization too — pin both
groups on the CLI rather than relying on whichever `filtering/localization` the
defaults list resolves to:

```bash
python scripts/filtering/run_filtering.py filtering.mode=state \
    filtering/analysis=etkf filtering/localization=none
python scripts/filtering/run_filtering.py filtering.mode=state \
    filtering/analysis=letkf filtering/localization=distance
```

The `*_tsvd` variants enable an observation-space truncated SVD, nested on the
analysis object as an `ObservationTSVD` block (`enabled`, `energy_fraction`,
`max_rank`, `numerical_tolerance`). `enabled` gates the *scientific* truncation
— `energy_fraction` and `max_rank` — so `max_rank` with `enabled: false` is
rejected at construction rather than silently ignored; `numerical_tolerance`
only redefines "numerically zero" and therefore applies either way. It
truncates weak linear combinations of the
*whitened observation anomalies* — a different axis from
`filtering/state_reduction`, which acts on state rows — and never touches the
physical observation-error variances. In `etkf`/`letkf` the whole `tsvd` node is
`null` rather than a disabled block, so turning truncation on there means
selecting the `*_tsvd` group, not overriding a leaf under a null node. `letkf*`
also excludes a state reduction, since the filter already refuses reduction
together with any localization. See
[data_assimilation.md](data_assimilation.md) for the transform math and the
`localize -> R_eff -> whiten -> TSVD -> transform` ordering.

`svd_current` refits an orthonormal basis to each cycle's final forecast
ensemble. Its knobs are `energy_fraction`, `max_rank`, and optional
`variable_scales`. `svd_streaming` incrementally accumulates those anomaly
blocks without retaining snapshots and additionally exposes
`forgetting_factor`, `update_every_n_cycles` and `subspace_warning_threshold`. A forgetting factor below one
has old-block covariance half-life `log(0.5) / log(forgetting_factor)` cycles;
one means equal-weight history, not a frozen basis. Both strategies affect only
the analysis representation—the forward model and observation operator remain
full-space. The shipped default remains `none`.

---

### 1.9 Config groups: `filter_smoothing/*` (run_filter_smoothing only)

The filter-smoothing entry point's groups. Each file uses
`# @package filter_smoothing` so it sets the matching `filter_smoothing.*`
field. Three of the four are the **inner state filter's** knobs — the same
classes as the corresponding `filtering/*` groups (§1.8), reused unchanged and
mounted under a different key; the fourth configures the **outer** trajectory
update.

| Group | Options (default first) | Sets |
|---|---|---|
| [`filter_smoothing/inner_analysis/`](../conf/filter_smoothing/inner_analysis/) | `stochastic`, `etkf`, `etkf_tsvd`, `letkf`, `letkf_tsvd` | `filter_smoothing.inner_analysis` — the per-cycle state update math inside the window. Same `localization_policy` contract as `filtering/analysis`: `etkf*` REQUIRES `filter_smoothing/inner_localization=none`, `letkf*` REQUIRES a non-null one |
| [`filter_smoothing/inner_localization/`](../conf/filter_smoothing/inner_localization/) | `none`, `correlation`, `distance` | `filter_smoothing.inner_localization` — the inner state analysis's localization (the `localization/` strategies reused unchanged) |
| [`filter_smoothing/inner_inflation/`](../conf/filter_smoothing/inner_inflation/) | `rtps`, `none`, `multiplicative`, `rtpp` | `filter_smoothing.inner_inflation` — the inner filter's spread maintenance |
| [`filter_smoothing/temporal_localization/`](../conf/filter_smoothing/temporal_localization/) | `none`, `taper` | `filter_smoothing.temporal_localization` — `TemporalLocalization` on the outer, whole-trajectory ESMDA update |

`taper.yaml` exposes `temporal_radius` (the truncation, counted in **cycles**
— the coordinates are cycle indices, so a radius below 1 freezes the
trajectory; default 3.0), `tapering_beta`, `max_inflation` and
`block_grouping` (keep `False` — a block shares one observation selection
across all of a parameter's knots, erasing the temporal taper). The distance
it tapers on is the separation between a knot (at its segment start) and an
observation batch (at its segment end), and the taper is symmetric in time —
a late observation may update an early knot, which is what makes the method a
smoother; the radius suppresses spurious long-range sampling correlations, not
causality. `none` (the default) leaves the trajectory update global.

Because the inner analysis and its localization are a coupled pair, pin both
groups on the CLI rather than relying on whichever the defaults list resolves
to, exactly as for `run_filtering.py` (§1.8):

```bash
python scripts/filter_smoothing/run_filter_smoothing.py \
    filter_smoothing/inner_analysis=etkf filter_smoothing/inner_localization=none
python scripts/filter_smoothing/run_filter_smoothing.py \
    filter_smoothing/inner_analysis=letkf filter_smoothing/inner_localization=distance \
    filter_smoothing/temporal_localization=taper
```

See [data_assimilation.md §9](data_assimilation.md) for the algorithm itself
(inner pass, trajectory stacking, reset semantics, final consistency pass).

---

### 1.10 Config group: `neural_surrogate/`

Five primary configs (not groups) drive the surrogate scripts:

| Config | Script | Purpose |
|---|---|---|
| [`neural_surrogate/training_data.yaml`](../conf/neural_surrogate/training_data.yaml) | `generate_training_data.py`, `generate_random_geometries_training_data.py` | Extends `run_forward_model` with `training_data:` block: `num_train/val/test`, `output_dir`, `params_sampler` (`UniformExternalAR2Sampler`, incl. a literal `seconds_per_knot`), the generation horizon (`simulation_time`/`output_frequency`/`spinup_time`, set here rather than inherited from the case), and `geometry:` (`source` = `barcelona`/`xie_and_castro` for the single-geometry script, `idealized`/`realistic` pools + `resolution`/`z_size` for the random-geometry script). Also pins the backend via a defaults-list `override /model@model:` entry (CLI `model=` still wins). |
| [`neural_surrogate/training.yaml`](../conf/neural_surrogate/training.yaml) | `train_neural_surrogate.py` | Full surrogate training config: `trainer:` (AMP, `torch.compile`, LR schedule, pushforward curriculum, grad clip, resume, checkpoint), `optimizer:`, `dataset:`, `dataloader:`, `init_weights_path`. Mode selected via `neural_surrogate/mode@_global_`. |
| [`neural_surrogate/pretrain_autoencoder.yaml`](../conf/neural_surrogate/pretrain_autoencoder.yaml) | `pretrain_autoencoder.py` | Tadpole-style (V)AE pre-training (plan 02), no next-step objective. `architecture:` (`neural_surrogates.TadpoleAE`: `size`, `encoder_crop_size`, `latent_type`, `encode_geometry`, `sdf_features`, `pretrained`), `loss:` (`kl_weight`, `geometry_recon_weight`), `trainer:` (`neural_surrogates.AutoencoderTrainer`), `optimizer:`, `dataset:` (`neural_surrogates.SnapshotDataset`), `dataloader:` (`snapshot_collate`). No `mode` group. See [neural_surrogates.md §26–30](neural_surrogates.md#part-g--autoencoder-foundation-model-pre-training-tadpole). |
| [`neural_surrogate/finetuning.yaml`](../conf/neural_surrogate/finetuning.yaml) | `finetune_neural_surrogate.py` | LoRA fine-tuning of a trained surrogate (plan 01). Reuses `training.yaml`'s `trainer`/`dataset`/`dataloader`/`optimizer` shape plus `pretrained_model_dir`, `model_name`, `recompute_normalization`, and a `lora:` block (`variant`, `rank`, `alpha`, `dropout`, `target_preset`, `target_modules`, `modules_to_save`). Architecture + state/param/SDF spec are read from the pretrained `model_dir` (not re-declared). `finetune_mode` selected via `neural_surrogate/finetune_mode@_global_`. See [neural_surrogates.md §21–25](neural_surrogates.md#part-f--parameter-efficient-fine-tuning-lora--peft). |
| [`neural_surrogate/testing.yaml`](../conf/neural_surrogate/testing.yaml) | `test_neural_surrogate.py` | Minimal: `model_dir`, `sample_idx`, `device`, `output_dir`. |
| [`neural_surrogate/comparison.yaml`](../conf/neural_surrogate/comparison.yaml) | `compare_surrogate_models.py` | `models` (list of `{name, dir}`), `data` (`root_dir`, `split`, `sample_indices`, `max_steps`), `device`, `output_dir`, `animate`. |

#### `neural_surrogate/architectures/`

Size presets for each architecture family. Each preset is a single `_target_`
block. Override the family/size on the CLI:
`'neural_surrogate/architectures@architecture=unet_convnext/large'`.

| Subfamily | Presets | `_target_` |
|---|---|---|
| `unet_convnext/` | `tiny`, `small`, `medium`, `large`, `xlarge` | `neural_surrogates.UNetConvNeXt` |
| `upt/` | `tiny`, `small`, `medium`, `large`, `xlarge` | `neural_surrogates.UPT` |
| `p3d/` | `tiny`, `small`, `medium`, `large`, `xlarge` | `neural_surrogates.P3D` |
| `domain_decomposed/` | `tiny`, `small`, `medium` | `neural_surrogates.DomainDecomposed` (with `_recursive_: false`, `_convert_: all`) |

#### `neural_surrogate/mode/`

Bundles the trainer class + loss + architecture default so a single `mode`
selection wires them all.

| File | Trainer | Loss | Architecture default |
|---|---|---|---|
| [`mode/standard.yaml`](../conf/neural_surrogate/mode/standard.yaml) | `neural_surrogates.Trainer` | `torch.nn.MSELoss` | `p3d/medium` |
| [`mode/domain_decomposition.yaml`](../conf/neural_surrogate/mode/domain_decomposition.yaml) | `neural_surrogates.PatchTrainer` | `neural_surrogates.DomainDecompositionLoss` | `domain_decomposed/small` |

The `mode` entry sits **after** `_self_` in `training.yaml`'s defaults list so it
overrides the inline `trainer._target_`. The architecture family/size is still
separately overridable on the CLI on top of the mode default.

#### `neural_surrogate/finetune_mode/`

The fine-tuning analogue of `mode/`: bundles the trainer class + loss for
`finetuning.yaml`, but (unlike `mode/`) sets **no** architecture default — the
architecture comes from the pretrained `model_dir`.

| File | Trainer | Loss |
|---|---|---|
| [`finetune_mode/lora_nextstep.yaml`](../conf/neural_surrogate/finetune_mode/lora_nextstep.yaml) | `neural_surrogates.Trainer` | `torch.nn.MSELoss` |
| [`finetune_mode/dft.yaml`](../conf/neural_surrogate/finetune_mode/dft.yaml) | `neural_surrogates.Trainer` | `torch.nn.MSELoss` |

Both entries sit **after** `_self_` so they override the inline `trainer._target_`.
Unlike `lora_nextstep` (architecture read from the pretrained `model_dir`),
`dft` (plan 03: autoencoder → time-stepper) declares the `TadpoleTimeStepper`
architecture **inline** — its `pretrained_model_dir` is the **AE** dir, and the
script instantiates the stepper fresh with `pretrained_ae_dir` set to it. It also
carries a `lora.target_preset: tadpole_encdec` and a `trainable_modules` list (the
sub-network + γ skips + `latent_residual_scale`, trained fully, not via LoRA). See
[neural_surrogates.md §31–34](neural_surrogates.md#part-g--autoencoder--time-stepper-tadpole-dft).

---

## Part 2 — `scripts/`: Executable entry points

### Standard script shape

Every Hydra-driven script follows:

```python
def run(cfg: DictConfig) -> None:
    ...

@hydra.main(version_base=None, config_path="../conf", config_name="run_forward_model")
def main(cfg: DictConfig) -> None:
    run(cfg)
```

`run(cfg)` is the **testable entry point** — tests compose a `DictConfig` directly
and call `run(cfg)` without going through Hydra's CLI. `main` is the CLI wrapper
only. Output directories are resolved via `resolve_output_dir(cfg, "<script_name>")`
from [`src/pyurbanair/config/hydra_helpers.py`](../src/pyurbanair/config/hydra_helpers.py),
which writes under Hydra's auto-managed run dir when invoked via `@hydra.main`
and under `${paths.*}` when called directly.

Scripts that are **plain argparse CLIs** (not Hydra) are noted explicitly below.

---

### 2.1 Core run scripts

#### [`run_forward_model.py`](../scripts/run_forward_model.py)
**Hydra** — config: [`run_forward_model.yaml`](../conf/run_forward_model.yaml)

The consolidated forward-model entry point. Replaces the former
`run_{ensemble,rollout,ensemble_rollout,time_varying}_forward_model.py` family.
Mode is selected by `run.*` knobs:

- `run.ensemble=true` — N-member ensemble instead of one member.
- `run.num_steps=N` — multi-window rollout (carry final state forward).
- `run.time_varying=true` — single member with time-varying AR(2) inflow; writes
  `state.nc` + `params.nc` as a ground-truth artifact for `run_esmda.py`.

Single member drops the `ensemble` dim (`isel(ensemble=0, drop=True)`) before
passing params to the solver. Multi-window stitch rebases each window's local
time clock onto a global monotonic axis.

**Produces:** field snapshot PNG, velocity-magnitude animation, derived
inflow-angle vs prescribed plot (when `params=dynamic`).

#### [`compare_models.py`](../scripts/compare_models.py)
**Hydra** — config: [`compare_models.yaml`](../conf/compare_models.yaml)

Runs several backends over each selected parameter scenario and draws the
cross-model comparison figures. Every backend is mounted under `models.<name>`;
`compare.models` (default `[pypalm, pyudales]`) selects which ones run and
`compare.reference` (default: the first) is the baseline for the difference
figures. Parameter configs are mounted under `parameter_scenarios.<name>` and
selected by `compare.parameter_scenarios`; the defaults `draw_a` and `draw_b`
are independent AR(2) draws and therefore produce four runs. Replace either
mount with `dynamic_sine` or `dynamic_cosine` for prescribed harmonic forcing.
The general knobs mirror `run_forward_model.yaml`, so `case`, `time`, `ensemble`
and `run.{ensemble,rollout_steps,skip_viz}` behave identically.

The default `compare.within_model_parameter_comparisons=true` also compares
`draw_a` against `draw_b` separately for PALM and uDALES. Those diagnostics live
under `comparison/within_model/<solver>/`; choose a different baseline with
`compare.parameter_reference=<scenario>`.

For another stochastic draw, add a named mount and assign its seed, for example:

```bash
python scripts/compare_models.py \
  +params@parameter_scenarios.draw_c=dynamic \
  parameter_scenarios.draw_c.seed=71 \
  'compare.parameter_scenarios=[draw_a,draw_c]'
```

For the two deterministic profiles, replace the default mounts with
`params@parameter_scenarios.draw_a=dynamic_sine` and
`params@parameter_scenarios.draw_b=dynamic_cosine`. Amplitude and frequency
can then be overridden independently at
`parameter_scenarios.draw_a.profiles.<parameter>.<field>`.

What makes it a controlled comparison:

- The params are sampled **once per scenario** (including each rollout window's
  `extrapolate`) and replayed for every backend.
- Each solver/scenario case gets its own scratch dir — the script rewrites the
  global `paths.experiment_dir` to `${paths.experiment_root}_<model>_<scenario>`
  before instantiating, since the model configs all interpolate that one key.
- Field figures are drawn after interpolating every model onto one common
  cell-centred grid (`compare.grid`, deliberately coarser than the solver grids)
  and one common time axis; each velocity component is interpolated on *its own*
  staggered axes, so staggering is undone in the same pass. Note that xarray's
  multi-dim `interp` returns NaN outside the source coords (`fill_value=None`
  does **not** extrapolate on that path), so targets are clamped per component.
- Sensor series use `_sensor_component_timeseries` at the physical sensor points
  on each model's own grid, so they carry no regridding bias. They are drawn
  twice: raw per-frame, and smoothed by a *sliding* window of the observation
  operator's `obs.interval_seconds` length reduced by `obs.aggregation_mode` —
  the same suppression of sub-interval fluctuation `TemporalObservationOperator`
  applies, without the arbitrary phase of its disjoint bin grid. Skipped when
  `obs.interval_seconds` is unset. The window is centred and rounded to an *odd*
  frame count (an even centred window sits half a frame left of centre, and since
  the count comes from each model's own cadence that would put a spurious
  relative phase shift between the compared curves). Aggregating the sensor
  series rather than the state field is exact for `mean` (linear interpolation
  commutes with the time average) and approximate for median/max/min.

**Produces:** `scenario_parameters.png`, `scenario_error_summary.png`, and
`scenario_summary.csv` at the root, plus one directory per scenario containing
`parameters.png` (prescribed vs each model's realised inlet inflow),
`state_snapshots_z<h>.png`, `state_difference_z<h>.png`,
`field_rmse.png`, `state_animation.mp4`, `sensor_timeseries_{assimilation,
validation}.png` (u/v/w/|U|, one line per model),
`sensor_rolling_{assimilation,validation}.png` (the same sensors smoothed over
the observation operator's window length), `sensor_metrics.csv` plus one
metric heatmap per sensor set, windowed `field_{mean,std}_z<h>.png` maps,
regional vertical-profile and velocity-distribution figures,
sensor spectra/autocorrelations, `measurement_overview.png` (the spatial
footprints and heights of every diagnostic), `wake_profiles.png`,
`wake_metrics.csv`, and `summary.csv`. The statistical diagnostics are controlled by
`compare.analysis`. When enabled, `within_model/<solver>/` contains the same
core field, statistical, wake, and sensor diagnostics with parameter scenarios
as the series and `within_model_summary.csv` aggregates their scalars.

#### [`run_esmda.py`](../scripts/esmda/run_esmda.py)
**Hydra** — config: [`run_esmda.yaml`](../conf/run_esmda.yaml)

The consolidated ESMDA entry point. Replaces `run_{parameter,state_and_parameter,rollout,time_varying_parameter,time_varying_parameters_rollout}_esmda.py`.

Stage 1 of the three-script pipeline (see `run_esmda_pipeline.sh`). Saves:
- Per-window `prior_params.nc`/`posterior_params.nc`, optionally prior state ensemble.
- Assembled `posterior_state_mean.nc`, `posterior_params.nc`, `prior_params.nc`,
  `true_params.nc`, `truth_access.yaml`, `run_info.yaml`, `config.yaml`.
- Under `esmda.save_obs_diagnostics=true` (the default), the per-window
  observation-space arrays: `windows/window_{w}_obs.nc` (`obs` = truth + noise,
  `obs_clean` = the noise-free projection, `obs_error_std` = `sqrt(diag(C_D))`,
  with `obs_sensor`/`obs_state`/`obs_interval` labels and the flattening order in
  the file attrs), `windows/window_{w}_pred_obs.nc`
  (`pred_obs(esmda_step, obs_index, ensemble)`, step 0 the prior forecast and −1
  the posterior forecast) and `windows/window_{w}_params_steps.nc` (the parameter
  ensemble at every iteration, kept for debugging). All KB-scale. The observation
  axis is named `obs_index`, not `obs`, because a variable whose name equals its
  dimension is silently promoted to an index coordinate on the netCDF
  round-trip — which would turn the `obs` data variable into a coordinate on
  read. These feed `run_summary.yaml`'s `esmda_diagnostics` block and figure D3;
  set the flag false to reproduce the pre-phase-2 artifact set exactly. The
  post-processing decides on the **run's own flag**, read back from
  `run_info.yaml`'s `configuration`, not on whether the files are there: a
  flag-off rerun into a `paths.results_dir` an earlier run already populated
  ignores the leftovers (with a warning) instead of republishing the previous
  run's mismatch beside this run's metrics.

Mode is the cross product of:
- `esmda/smoother=static|state|state_and_parameter|dynamic|state_and_dynamic`
- `params@prior_params=static|dynamic`
- `esmda.num_assimilation_windows=1|N`

Truth is generated up front for all windows, then the window loop consumes it.
For the dynamic case, `prior_sampler.extrapolate(posterior, ...)` seeds the next
window's prior. `include_state` and `is_dynamic` flags (derived from the smoother
type) drive the generic window loop without per-combination branching.

#### [`run_filtering.py`](../scripts/filtering/run_filtering.py)
**Hydra** — config: [`run_filtering.yaml`](../conf/run_filtering.yaml)

The sequential-filtering (EnKF) entry point. Where ESMDA re-forecasts one
window `num_steps` times with tempered updates, the filter forecasts one
segment per cycle and applies ONE full-weight analysis to the end-of-segment
state/parameters, warm-starting the next cycle from the analyzed state. The
truth is generated over `filtering.num_cycles` segments up front; per-cycle
observations are extracted with the case's temporal observation operator and
the filter consumes the whole `(num_cycles, N_d)` batch in one `run()` call.

Mode is the cross product of `filtering.mode=state|parameter|joint` and the
`filtering/*` groups (§1.8), including the deterministic ETKF/LETKF analyses
and optional current/streaming state SVD. Reduced state analysis requires
`mode=state|joint` and `filtering/localization=none`; `filtering/analysis` also
constrains the localization choice (§1.8). Truth source (`run.truth_dir`) mirrors
`run_esmda.py`. Static scalar parameters only — a dynamic (AR(2)) params
mount fails loudly; time-varying priors stay with the ESMDA smoothers.

Stage 1 of the single-run filtering pipeline (§2.4), orchestrated by
[`run_filtering_pipeline.sh`](../scripts/run_filtering_pipeline.sh).

Saves: `posterior_params.nc` / `posterior_state.nc` (analyzed final-frame
ensemble), optional `params_history.nc` / `state_history.nc`
(`run.save_history`), per-cycle `cycle_diagnostics.yaml` (innovation χ²,
obs-space prior/posterior RMSE plus the `obs_posterior_rmse_kind` provenance
label, block spreads, `analysis_time`, and stable sets of nullable reduction
(`reduction_*`) and ensemble-transform (`transform_*` for a global ETKF,
`local_*` for a LETKF's per-block counts, ranks, retained/discarded energy and
chunk size) diagnostics),
`prior_params.nc`,
`true_params.nc`, `true_state.nc` (inline truth), `truth_access.yaml` (the
lazy-truth slicing/offsets the metric/figure stages read back), `run_info.yaml`,
`config.yaml`. `run_info.yaml.configuration.state_reduction` records the fully
resolved reduction subtree (or `null`) for benchmark provenance;
`state_reduction_resolved_variable_scales` records the physical scales actually
applied after validating the forecast Dataset's variables.
`run_info.yaml.configuration.analysis` records the fully resolved analysis
subtree (its `_target_` and any nested `ObservationTSVD` settings). It is never
`null`, and it is the only record of which update math ran: `configuration.filter`
stays `EnsembleKalmanFilter` for the ETKF and LETKF too, because the analysis is
injected rather than subclassed. `run_info.yaml.configuration.localization`
records the resolved localization subtree beside it (`null` for the global
update): the two are a coupled pair through `localization_policy`, and the
per-block `local_*` cycle diagnostics only mean anything against the strategy
and radius that produced them.

#### [`run_filter_smoothing.py`](../scripts/filter_smoothing/run_filter_smoothing.py)
**Hydra** — config: [`run_filter_smoothing.yaml`](../conf/run_filter_smoothing.yaml)

The filter-smoothing entry point: a sequential EnKF for the state, wrapped in
an outer ESMDA loop over the **parameter trajectory** of the whole window. Each
of `filter_smoothing.num_steps` iterations runs one full inner filter pass
through the `filter_smoothing.num_cycles` cycles — recording each cycle's
forecast observations *before* its analysis — stacks them into one
`(L·N_d, N_e)` system and applies one tempered Kalman update to the flattened
trajectory; every iteration restarts the inner pass from the same initial
state, and a final pass with the converged trajectory produces the returned
state. See [data_assimilation.md §9](data_assimilation.md).

Truth generation, per-cycle `(num_cycles, N_d)` observation batches, obs noise
and truth source (`run.truth_dir`) are the same as `run_filtering.py`. The one
inverted guard is the prior: a **dynamic (time-varying) params mount is
required** (`params@prior_params=dynamic`), sampled once over the whole window
with `time.seconds_per_knot = time.simulation_time` so its knots align with the
cycles — a knot spacing or knot count that does not cover `num_cycles` fails
loudly. The sampler emits `L+1` knots for an `L`-cycle window; the trailing
knot rides along in the trajectory, updated only through the prior's temporal
correlations.

Mode is the cross product of the `filter_smoothing/*` groups (§1.9) — the inner
analysis / localization / inflation and the outer temporal localization — with
`filter_smoothing.num_steps` and `num_cycles`. Phase 1 is a single window; the
moving-window extension is not implemented. There is no metric/figure pipeline
counterpart yet (§2.3 / §2.4) — this script's artifacts are the whole output.

Saves: `posterior_params.nc` (the smoothed **trajectory** ensemble, with its
`time` knot dim), `posterior_state.nc` (the filtered end-of-window ensemble
from the final consistency pass), `iteration_diagnostics.yaml` (one record per
outer ESMDA iteration: windowed observation-space RMSE, trajectory spread,
innovation χ² on the stacked system), `cycle_diagnostics.yaml` (the final
pass's per-cycle diagnostics, the same `FilterResult` schema as
`run_filtering.py`'s), `prior_params.nc`, `true_params.nc`, `true_state.nc`
(inline truth), `truth_access.yaml`, `run_info.yaml`, `config.yaml`. Under
`run.save_history` (default true) it also writes `params_iterations.nc` — the
trajectory ensemble at every outer iteration, the artifact that shows the
smoothing converge — plus the final pass's `params_history.nc` /
`state_history.nc` over cycles.

---

### 2.2 Shared script libraries

#### [`_common.py`](../scripts/_common.py)

Script-level plumbing shared by `run_forward_model.py` and other callers.
Not a Hydra script; imported directly. Provides:
- `resolve_results_dir(cfg)` — extracts `cfg.run.results_dir` or `None`.
- `visualize_forward_state(state, model_name, out_dir, ...)` — standard field
  snapshot + velocity-magnitude animation. Projects uDALES staggered fields onto
  a common grid before display.
- `plot_derived_inflow_angle(...)` / `plot_derived_velocity_magnitude(...)` —
  derived-vs-prescribed inflow diagnostics.
- Per-parameter skill panels built on `evaluation.scores` (CRPS, per-knot
  error/spread/in-band, summary scalars).

#### [`_esmda_common.py`](../scripts/esmda/_esmda_common.py)

Post-processing helpers shared by `compute_esmda_metrics.py` and
`make_esmda_figures.py` (and reused by the filtering pipeline's
[`_filtering_common.py`](../scripts/filtering/_filtering_common.py), §2.4).
Read-only with respect to the run directory except explicit write calls.
Provides:
- `load_run_config(run_dir)` — re-load the Hydra config saved by `run_esmda.py`.
- `build_sensor_sets(cfg)` — assimilation + optional validation sensor coordinates
  from the obs config.
- `open_truth(cfg, ta)` — lazy truth access (multi-GB `state.nc` never fully loaded).
- `ensemble_sensor_series(...)` / `truth_sensor_series(...)` — interpolate ensemble
  and truth states at sensor locations. Both stream: the truth one window at a
  time, the ensemble **one member at a time** (since WP1.3 — a full-ensemble
  window state file runs to gigabytes and is never read whole).
- `probe_record_paths(run_dir)` / `probe_spectra_bundle(run_dir)` — locate the
  high-rate probe records an optional `run_probe_series.py` rerun wrote (one
  window's truth, posterior and — optionally — prior) and reduce them to the
  matched Welch spectra both the metric and the figure stage consume, so
  `spectral_metrics` and `probe_spectra.png` cannot be computed two different ways.
  `None` when a run dir has no probe records, which is most of them. `truth_probes.nc`
  is unversioned and overwritten by every rerun, so the pairing is decided by the
  truth's own `window_index` and a truth whose window has no posterior record (or a
  member record whose recorded window disagrees) is **refused** rather than paired
  with a stale one — re-probing window 2 after window 1 must not silently compare
  two different windows.
- `read_yaml(...)` / `write_yaml(...)` / `_to_native(...)` — small YAML helpers.

The metric functions themselves (`streaming_state_rmse`, `select_z_plane`,
`sensor_magnitude`, `window_statistics`, `parameter_metric_summary`,
`series_stats`, `vector_sensor_metrics`) live in the
[`evaluation`](../libs/evaluation/src/evaluation/) library; this module keeps
only the run-dir-aware extraction they consume.

---

### 2.3 ESMDA pipeline scripts

These three scripts form the standard single-run pipeline, orchestrated by
[`run_esmda_pipeline.sh`](../scripts/run_esmda_pipeline.sh).

> The metric and figure *computation* lives in the editable leaf library
> [`libs/evaluation`](../libs/evaluation/src/evaluation/) (`scores`,
> `turbulence`, `sensors`, `style`, `figures`), leaving the two metric/figure
> stages below — and their filtering counterparts in §2.4 — as thin
> orchestration: resolve run dirs, open artifacts, call `evaluation`,
> write YAML/PNGs. The library takes arrays and datasets and returns
> numbers, dicts and figures — it knows nothing about Hydra or the run-dir
> layout, and imports neither jax nor `pyurbanair`. See
> [docs/plans/esmda_evaluation/master_plan.md](plans/esmda_evaluation/master_plan.md).

#### [`compute_esmda_metrics.py`](../scripts/esmda/compute_esmda_metrics.py)
**Plain argparse CLI** — usage: `python scripts/esmda/compute_esmda_metrics.py --run-dir <dir>`

Stage 2 of the pipeline. Reads the artifacts saved by `run_esmda.py` and writes
`run_summary.yaml` — the `run_info` metadata augmented with:
- `metrics_version` — estimator-semantics marker (see below).
- `parameter_metrics` — per parameter, accuracy (RMSE/CRPS summary + RMSE
  reduction and CRPSS (`crps_reduction_vs_prior`) against the prior) and
  calibration (`z_score` = `(θ*−θ̄ᵃ)/σᵃ`, `normalized_error` = `(θ̄ᵃ−θ*)/σᵇ`,
  `contraction_ratio` = `σᵃ/σᵇ`), plus a `pooled` entry holding the z-scores
  of every parameter and knot together. Read the two halves jointly: a small
  contraction ratio with a large `|z|` is a *spuriously* confident posterior,
  which no accuracy number reveals. A calibrated posterior has pooled
  `z_score.mean ≈ 0` and `std ≈ expected_std` — **compare against
  `expected_std`, never against 1.** The z-score is standard normal only in
  the limit; at finite `M` it is `√(1+1/M)·t₍M₋₁₎`, whose std is 1.02 at
  `M=64` and 1.26 at `M=8`. Below **four** members the whole `z_score` entry
  is `null`: at `M=2` that t is Cauchy, so neither its variance *nor its mean*
  exists and a perfectly calibrated run would report `|mean| > 5` about 13 %
  of the time. Read `contraction_ratio` there — it is well defined at any
  size. Entries are likewise `null` where the scale is degenerate (a pinned
  parameter has `σᵇ = 0`; a single-member ensemble has no `ddof=1` spread).

  Two caveats on reading `std`. It is a *sample* std over `n` knots and
  carries its own sampling noise, which is wide when `n` is small: for a
  calibrated `M=64` ensemble the 5th–95th percentile range is 0.06–1.99 at
  `n=2` and still 0.57–1.47 at `n=8`, against an `expected_std` of 1.02. Treat
  a single parameter's `std` at `n < ~10` as a coarse flag and prefer the
  `pooled` entry, which aggregates knots across parameters. And the pooled `n`
  is a count, not an effective sample size: adjacent knots of one time-varying
  parameter are strongly correlated, and a many-knot parameter outweighs a
  static one.
- `ensemble_health` — `n_members` / `n_unique` (exact duplicate rows) run-wide
  and per window, plus the min/median pairwise-distance ratio. A resampling
  policy that clones a diverged member (pypalm) leaves an ensemble with fewer
  degrees of freedom than its nominal size.
- `state_metrics` — `|U|` field RMSE summary (streamed z-slice by z-slice).
- `sensor_metrics` — full-vector (u, v, w) RMSE and energy score per sensor set
  (assimilation + validation).
- `sensor_statistics` — the same sensor sets scored on **window statistics**
  rather than on the instantaneous series. Instantaneous turbulence is chaotic:
  two members with identical parameters decorrelate within an eddy turnover, so
  a pointwise error is mostly a phase measurement that no parameter estimate
  controls. The parameters act on the *statistics*, which is what makes those
  the identifiable quantities. Per sensor set: `n_members`, `n_windows`,
  `num_sensors`, and a `posterior` (plus `prior`, when `run.save_prior_state`
  saved the prior states) block with one entry per `statistic × quantity` —
  `mean_u` … `variance_magnitude`, eight in all, scored separately because a
  parameter that fixes the mean wind while leaving the resolved variance halved
  is exactly the failure this block names. Each entry carries:
  - `crps` — the fair CRPS of `{T_m}` against `T*`, averaged over sensors and
    reported as a series over windows (`final` = the last window), plus
    `prior_crps_mean` / `crps_reduction_vs_prior` when the prior was scored.
  - `z_score` — the same reduction and the same `expected_std` caveats as
    `parameter_metrics` above, pooled over sensors and windows.
  - `rank_counts` — how often the truth landed at each rank `0…M` within
    `{T_m}`, pooled over sensors and windows (so the list has `M+1` entries and
    sums to the number of scored knots). Ranks use only the ordering, so pooled
    they see a *shape* failure (a skewed or bimodal posterior with the right
    mean and spread) that neither the CRPS nor the z-score can. Figure D1
    coarsens these to ~10 bins; binning is one-way, so the artifact keeps the
    exact `M+1`.
  - `identifiability` — across-member spread of `T` divided by a typical
    member's own block-bootstrap sampling std of `T`, averaged over sensors and
    reported per window. **Read this first:** below ~3 (which is what the
    warning in the run log thresholds on, using this same `mean`) the statistic
    does not clear its own sampling noise at this window length, and its CRPS
    and ranks are measuring the window, not the parameters. Absent when no floor
    could be measured — the bootstrap needs both ≳15 frames per window *and* a
    window spanning ≳15 integral time scales of the series itself, so short runs
    (including the CI smoke shape) have none, and an unmeasured floor is unknown
    rather than infinitely identifiable. The second condition is the binding one
    on real urban-canopy flow: in-canopy velocity decorrelates over ~140 s, so a
    300 s window holds ~2 independent samples and the floor is refused outright
    rather than under-reported. See the phase-3 note in
    `docs/plans/esmda_evaluation/`.

  A sensor set whose state files carry no `ensemble` dimension (an old
  ensemble-mean-only artifact) is dropped from **both** sensor blocks with a
  log line — every score in them is probabilistic and needs the members — and
  the rest of the summary is written as usual.
- `field_metrics` — the mean **field**, scored with the VDI 3783/9 hit rate `q`:
  the fraction of cells where the posterior ensemble-mean time-mean velocity is
  within a relative tolerance `D = 0.25` **or** an absolute one `W` of the
  truth's. The `or` is the guideline's: a relative test alone is unreachable
  wherever the velocity passes through zero (every recirculation boundary),
  an absolute one alone is meaningless in the free stream. Acceptance is
  `q ≥ 0.66`. `W` is not a convention here — it is the truth's own
  block-bootstrap sampling error on its time-mean (median over sampled cells,
  per component), which turns `q` into "indistinguishable from the truth within
  the truth's own noise". The block carries `hit_rate_posterior` (pooled `q`
  over the three components, `n_points`, and a per-component breakdown),
  `hit_rate_prior` when `run.save_prior_state` saved the prior states,
  `hit_rate_tolerance_w`, the scored `z_levels`, the `horizontal_stride`,
  `n_windows` and the frames each member (and the truth) contributed. A `W` of
  `null` means no floor could be measured (the run is too short to
  block-bootstrap — the CI smoke shape always is), and the hit rate then runs on
  its relative criterion alone; a `q` of `null` means no cell was scorable at
  all. The prior half is all-or-nothing: a prior covering fewer windows or fewer
  frames than the posterior — a job killed mid-write — is dropped rather than
  compared across a different horizon.

  **Fluid cells only**, which matters more than it sounds: a solid cell holds
  ~0 in the truth *and* in every member, so it is a hit whatever the flow does,
  and counting solids drags `q` toward the built-up fraction — at 30 % solid a
  fluid hit rate of 0.52 would report as 0.66 and clear the acceptance
  threshold on a field that fails it. Two rules, in order of authority: the
  backend's own `blanking` indicator when the state files carry one (pylbm), and
  otherwise the truth's own resolved TKE, since a cell a solver held at a
  constant has exactly zero variance while a fluid cell in a turbulent flow does
  not. `solid_cell_source` records which ran, with `n_fluid_cells` and
  `solid_fraction` beside it; `none` means every cell was scored and `q` is
  diluted by whatever obstacles the domain holds. The fallback cannot see an
  obstacle a backend filled with time-varying junk rather than zeros (uDALES),
  and it stands down entirely on a truth with no resolved fluctuation anywhere.

  The fields behind it are written beside the summary as **`eval_fields.nc`**
  (the WP1.5 figures read it rather than re-streaming): per-cell time-mean
  velocity, resolved TKE and `<u'w'>` for the truth and, reduced across the
  ensemble, for the posterior (and prior). Two regions — a few evenly spaced
  z-slabs at full (or strided) horizontal resolution, and full-depth columns at
  the sensors' `(x, y)`, **both** the assimilated and the held-out ones, labelled
  by a `station_set` coordinate — with ensemble mean and `ddof=1` spread on both
  and nested quantiles at the station columns only. Reductions only: no
  per-member field is stored, and everything is float32. The stresses are
  **resolved-only** — the subgrid contribution is not in them and is not
  negligible inside a canopy. The file is self-contained by design: the
  averaging window (`t_start`/`t_end`), the stride, the station labels, the
  fluid mask (`slab_fluid`), **which frames the moments were reduced over**
  (`moment_sampling`, and whether those frames left gaps —
  `moment_sampling_is_sparse`) and which axes carry colocation's extrapolated
  edge (`extrapolated_edges`) are attributes, coordinates or variables here, so a
  figure never reopens the run's other artifacts — or re-derives a mask — to
  draw an honest plot. That last one matters for plotting: every axis colocation
  moves has its **last** index extrapolated from the two faces below it rather
  than interpolated between two, so those cells carry inflated second moments
  (~20 % for a well-resolved field, up to 5× for face-to-face white noise). It
  is not only the vertical — uDALES moves x, y and z, PALM moves x and y — and
  an evenly spaced selection always includes the last index.

  `moment_sampling` is on **every** `eval_fields.nc`, ESMDA and filtering
  alike, because `t_start`/`t_end` alone are a horizon, not a cadence. It is a
  plain sentence naming the frames; ESMDA writes the default one — *"every
  output frame of each window's posterior rollout, so the moments are
  within-window time averages"* — and a pipeline whose frames are not that
  passes its own (the filter does, §2.4). What it exists to disambiguate is
  `*_tke` / `*_uw`: moments reduced over one frame per cycle are an
  across-cycle variance carrying the analysis increments rather than resolved
  turbulence, and nothing in the numbers says which. The figure stage opens
  this file and not `run_summary.yaml`, so the line has to travel with the data
  it qualifies; S1 and F1 take it as their `sampling_note`.

  **`moment_sampling` is provenance; `moment_sampling_is_sparse` is the
  caveat.** The second attribute is beside it on every `eval_fields.nc` — `0` or
  `1`, an int because netCDF attributes have no boolean type — and it records
  whether those frames leave *gaps* in `t_start`–`t_end`, so that the moments
  are not a continuous time average. **That flag, and not the presence of
  `moment_sampling`, is what S1 and F1 qualify their labels on.** The note
  prints on both cycle-state sources, dense and sparse alike; the flag alone
  drives "time-mean" versus "sample-mean" — F1's colorbar label, title prefix,
  span preposition (`t = …` versus `sampled over t = …`) and caption, and S1's
  caption — and it is what marks S1's TKE rows with a `*` (the marker needs a
  footnote to point at, so it also needs the note). The split is deliberate:
  sparseness is a property of the frames, not of whether someone wrote a
  sentence about them. The `forecast` cycle-state source names its frames too
  and genuinely *is* a continuous time average, so a figure that inferred the
  caveat from the note's mere presence would qualify it exactly as loudly as it
  qualifies a one-frame-per-cycle run — and the warning would stop
  discriminating between the two. ESMDA always writes `0`; the filter writes
  `1` for every source but `forecast` (§2.4). One asymmetry worth knowing:
  `make_filtering_figures.py` reads both attributes off the file and forwards
  them, while `make_esmda_figures.py` reads neither, so on an ESMDA run the
  wording is right by default but the provenance line is not drawn.

  Three things worth knowing about how they are produced. The accumulation
  rides on the *same* pass over the window state files as the sensor
  extraction, so the ensemble — the M-times-larger half — costs no extra read
  (the truth is streamed a second time, because it can only be sampled once the
  ensemble pass has fixed the grid); the components are interpolated onto
  cell centres first (a stress is a one-point moment, and combining staggered
  components by array index biases the anisotropy ratio by the staggering
  pattern rather than by the flow); and the truth is interpolated onto the
  assimilation grid, so cross-grid runs are scored cell against cell. Memory is
  bounded by two derived numbers rather than by config, because they answer two
  different questions: a horizontal stride keeping **one ensemble's** persistent
  accumulators inside ~1 GB (that is all M members' — and the posterior and prior
  collectors are alive at once, so the pass's persistent worst case is twice it;
  logged whenever the stride is not 1), and a time sub-chunk keeping one
  accumulation step's transient inside ~256 MB. The transient is the larger term
  and is sized on whichever grid the step touches — the source grid when
  colocation or a cross-grid interpolation runs, the target slab otherwise. The
  whole block is dropped — with a log line, the rest of the summary intact —
  when the state carries no ensemble axis or its layout cannot be co-located.
- `spectral_metrics` — the **frequency spectra** at the probes, and the one thing
  the mean-field metrics cannot see: a flow can carry the right time-mean and even
  the right total variance while putting that variance at the wrong frequencies,
  which is what an over-smoothed or surrogate-collapsed field does. Welch
  estimates (Hann, 50 % overlap, linear detrend) of the trace `E_uu+E_vv+E_ww` at
  each probe, with **one segment duration for the truth and every member** —
  identical `nperseg` at a shared cadence, and at differing cadences the shared
  quantity is the segment's length in *seconds*, with the comparison restricted to
  the band both records resolve. Nothing is ever resampled: interpolating a coarse
  record onto a fine axis invents the high-frequency content under test.
  Comparisons stop at `f_Nyquist/4` (`f_cutoff`), since the top of a sampled band
  carries the discretisation's roll-off and the SGS closure rather than the
  flow. It is **not** an anti-aliasing margin: snapshot probing applies no
  anti-alias filter, and content near `f_s` folds toward *DC* — the bottom of
  the scored band, where no cutoff reaches it. Bounding that is the probe
  cadence's job.
  `lsd_posterior_median` is the log-spectral distance
  `√(mean_k [10·log₁₀(E_t/E_m)]²)` in dB between the truth and the *member-median*
  spectrum, reduced over sensors by the median (`n_sensors`, `n_band_bins`,
  `segment_seconds` and `sample_frequency` record what it ran on).

  **`n_band_bins` is the number to size a probe rerun against**, and it depends
  on the **sample count alone** — `nperseg = n/8` and the cutoff is a fraction
  of `f_s`, so the cadence cancels.
  `evaluation.turbulence.spectral_band_bins(n)` returns the bins a record of
  `n` samples would score and
  `minimum_spectral_samples(band_bins)` inverts it. Its default argument is the
  **refusal floor** — 4 bins, 264 samples, below which `probe_spectra` returns
  `None` — which is not a target: bins run `1..B`, so `B` bins is a frequency
  span of exactly `B`, and four of them is well under a decade.
  `SPECTRAL_BAND_DECADE_BINS` (10 bins, **648 samples**) is the decade a `-5/3`
  reading off figure S4 needs; `run_probe_series.py`'s pre-flight reports the
  bin count and warns below it.

  **Never read it against zero, and not against `lsd_truth_floor` either.**
  `lsd_truth_floor` is the distance between the two halves of the truth's own
  record, and it runs **~2×** the scatter this comparison shows under a null of
  identical flows: halving the record at fixed `nperseg` halves the segment count
  (≈4× the variance in `LSD²`) while the reported number compares the full truth
  against a *median over M members*, whose own scatter is ~1/M of one estimate —
  measured 1.99 (M=8) and 2.08 (M=32) on statistically identical `f^-5/3` flows.
  So `lsd_truth_floor_comparable` (that value halved) is the like-for-like
  reference, and it is the one to read against. **Neither is a pass threshold**:
  the metrics doc sets no acceptance level for the LSD, and a posterior at 2 dB
  under a 2.5 dB halves distance is *not* indistinguishable from the truth — its
  comparable reference is ~1.2 dB. `lsd_prior_median` appears when prior
  probe reruns were done and is what makes the posterior number a change rather
  than an absolute; the optional prior record can never move any other entry (the
  scored sensors, the band and the cutoff come from the truth and posterior alone,
  and a prior that does not fit that grid is dropped with a log line).
  **`n_members` is not bookkeeping**: the median of M spectra is smoother the
  larger M is, so the distance falls with ensemble size on identical flows (1.409
  dB at M=2 against 1.171 dB at M=32) and two runs' LSDs are comparable only at
  equal M — which matters directly for the ensemble-size sweeps in
  [docs/job_scripts.md](job_scripts.md). The whole block is absent unless an explicit
  `scripts/esmda/run_probe_series.py` rerun wrote the high-rate probe records —
  the assimilation's own output cadence is ~30× too coarse for a spectrum — and
  it is the one block computed even under `run.skip_viz`, since it reads only
  those small records and never the truth.
- `esmda_diagnostics.data_mismatch` — the normalized data mismatch
  `O_N = (1/2N_d)·(d−g(θ))ᵀ C_D⁻¹ (d−g(θ))` per member per ESMDA iteration,
  reduced to `per_step_median` / `per_step_iqr` / `per_step_min` (index 0 the
  prior forecast, −1 the posterior). This is the one diagnostic that separates
  the two ways an ES-MDA run fails: `O_N` well above the `target` of ½ is
  under-fitting, `O_N` well *below* it means the ensemble is fitting the
  observation noise — an over-aggressive schedule or a missing localization —
  and no RMSE distinguishes those. `target_band` is `3/√(2N_d)`; the
  `underfit_final` / `overfit_final` / `collapsed` flags compare the last
  iteration that produced values — `final_step_index` says which, since a run
  whose posterior forecast failed outright gets an earlier one — against it.
  `collapsed` fires only when a vanishing across-member IQR is paired with an
  off-target median (identical members *on* target are converged, not collapsed)
  and is `null` when fewer than 8 values back it, since the smoke shape's two
  members have no meaningful IQR. **The flags are advisory**, and the block carries
  `caveat: no_representativeness_error` saying why: the χ² target assumes `C_D`
  covers representativeness error and here it is a single instrument-scale
  `esmda.obs_error_std`, so a too-small `C_D` makes a healthy run look
  under-fitted. Read the trend across iterations and the member spread — neither
  moved by a constant mis-scaling of `C_D` — before the flags. Absent on any run
  dir written before WP2.1 or with `esmda.save_obs_diagnostics=false` — that
  second case is decided by the flag recorded in the run's own `run_info.yaml`
  and not by whether the files exist, so a flag-off rerun into a results dir an
  earlier run populated drops the block (and D3) with a warning instead of
  reporting the earlier run's mismatch. An *absent* flag means unknown, not
  false, and falls through to the files. Like `spectral_metrics` it is computed
  even under `run.skip_viz`, reading only the KB-scale observation-space files.

> **`metrics_version`.** The keys above are additive only; when an existing key
> changes *meaning*, this marker bumps instead. `2` (current) means the fair
> `M(M−1)` pairwise CRPS / energy-score estimators and the root-mean-variance
> spread introduced in WP1.1. An absent marker (or `1`) means the older biased
> `M²` form, whose scores sit ~O(1/M) higher — at M=50 roughly 2 %, enough to
> reorder a sweep. `compare_sweep_results.py` and `compare_state_runs.py` warn
> when the runs they are about to compare mix the two; re-run
> `compute_esmda_metrics.py` on the older dirs to bring them forward.

#### [`make_esmda_figures.py`](../scripts/esmda/make_esmda_figures.py)
**Plain argparse CLI** — usage: `python scripts/esmda/make_esmda_figures.py --run-dir <dir>`

Stage 3 of the pipeline. Reads artifacts and writes into the run directory:
- `rollout_time_evolution.png` — parameter trajectories + state `|U|` RMSE.
- `parameter_error.png` — per-parameter posterior error over time.
- `rollout_animation.mp4` — ensemble-mean `|U|` field vs truth.
- `final_state_with_obs.png` — final `|U|` field with sensor locations.
- `sensor_timeseries_<set>.png` — truth vs ensemble at each sensor set.
- `parameter_marginals.png` — prior vs posterior marginal per parameter, truth
  dashed and the z-score annotated, with the y-limits including the prior so the
  contraction is visible. **Which knots the two marginals come from depends on
  the truth**, and both are labelled with it in the panel: for a *static*
  parameter (the truth is the same at every knot — `prior_params.nc` stacks one
  point per assimilation window) the prior is taken at **knot 0**, the run's
  actual prior, against the posterior at the final knot, so the panel shows the
  total contraction the run achieved; for a genuinely *time-varying* truth both
  come from the **final** knot, since knot 0 is a different physical time and
  the only contraction available is per-window. Note that WP1.2's
  `parameter_metrics.contraction_ratio.final` in `run_summary.yaml` is always
  per-knot, so on a static multi-window run the figure and the YAML deliberately
  describe different things — the figure's subject is the run, the YAML's is the
  window.
- `station_profiles.png` — vertical `ū/U_ref` and TKE profiles at the sensor
  columns (assimilated *and* held-out, labelled by `station_set`): truth line,
  posterior median with nested quantile bands, prior bands, plus an inset plan
  view. `U_ref` is the truth's `velocity_magnitude` parameter when the case
  estimates one, else the profiles are drawn in m/s. The `z/H` axis and its
  roof line need a canopy height, read from the optional `geometry.building_height`
  key; no shipped case defines it (the geometry is an STL, not a scalar), so the
  default is metres.
- `mean_slices.png` — a few z-levels × (truth | prior mean | posterior mean |
  posterior − truth), shared colour norm across the first three and a symmetric
  diverging one for the difference, solid cells masked out. Always the
  accumulated time-mean, never an instantaneous frame; the averaging window is
  annotated from the file's own `t_start`/`t_end`.
- `sensor_fans.png` — the sensor `|U|` series as nested posterior quantile fans,
  one column per sensor set, with the truth, the window boundaries and a
  ± `esmda.obs_error_std` envelope around the truth. Its x-axis is physical time
  on *every* run, so the window boundaries are marked whenever there is more than
  one window — unlike the parameter plots, whose x-axis is a window index on a
  static run. **Pre-WP2.1 caveat, stated in the figure:** the realized noisy
  observations the run actually assimilated are not persisted yet, so the
  envelope is the *clean* truth ± σ, not the values that were assimilated.
- `probe_spectra.png` — premultiplied `f·E(f)/σ²` at the probes, log–log, one panel
  per probe sensor (held-out first): truth line, posterior nested quantile bands,
  prior 5–95 % envelope, a dotted line at the `f_Nyquist/4` comparison cutoff and a
  short `−2/3` reference slope offset above the curves (a reference, not a fit).
  Every curve in a panel is divided by the **truth's** resolved variance, so a
  member carrying half the energy cannot overlay the truth. Each panel annotates
  its own log-spectral distance and the truth's *halves* distance — the same
  function and band as `spectral_metrics`, whose entries are these numbers reduced
  over sensors. The panel says "truth halves" rather than "floor" deliberately: it
  is ~2× the like-for-like scatter and not a pass threshold, which the caption
  repeats. A probe whose truth variance is zero or non-finite (a sensor in a solid
  cell, a series with a gap) is dropped with a log line rather than drawn
  unnormalized under a normalized axis label. Needs the high-rate probe records
  (see `run_probe_series.py`), so it is absent on any run dir without a probe
  rerun.
- `rank_histogram.png` — the rank of the truth's window statistic within the
  members, read out of `run_summary.yaml`'s `sensor_statistics` block and pooled
  over statistics, sensors and windows; rows = sensor sets, columns = prior |
  posterior, with the uniform reference and its binomial band. Coarsened to ~10
  rank bins (the summary stores all `M+1`, since binning down is exact and
  binning up is not).
- `data_mismatch_decay.png` (D3) — per-member `O_N` boxes vs ESMDA iteration
  against the ½ target band, the figure the ES-MDA literature conditions readers
  to expect. A rollout's windows are drawn as separate boxes per iteration rather
  than pooled: window 0's prior is a cold-start draw and a later window's is an
  extrapolated posterior, so one pooled step-0 box would conflate two different
  objects. The y-axis goes log once the per-step medians span ≥1.5 decades and
  every plotted value is positive (a healthy MDA run drops `O_N` by one to two
  orders of magnitude, which a linear axis flattens onto zero exactly where the
  band matters); on a log axis the band is floored at the smallest plotted value,
  and dropped altogether when nothing plotted comes near it, so it can never be
  drawn inverted. The `C_D` caveat above is annotated
  on the figure. Reads the same `obs_diagnostics_bundle` the metric stage scores,
  so the boxes and the YAML come off one reduction; absent whenever
  `esmda_diagnostics` is.

Five of the last seven depend on an artifact a run dir may not have:
`station_profiles.png` and `mean_slices.png` read `eval_fields.nc`,
`rank_histogram.png` reads `run_summary.yaml`'s `sensor_statistics` block — both
written only by a current metric stage — `data_mismatch_decay.png` reads the
per-window observation-space files (WP2.1, `esmda.save_obs_diagnostics`), and
`probe_spectra.png` reads the probe
records only an explicit probe rerun writes, so it is missing from almost every
run dir by design. The other two need nothing extra:
`parameter_marginals.png` reads the parameter datasets and `sensor_fans.png` the
sensor series this stage extracts itself, and both are present on any run dir.
Each figure is **skipped with a printed line** when its input is absent, and a
skip never costs the figures after it — an old run dir still gets every figure it
can support. The prior halves of the profile, slice and rank figures additionally
need `run.save_prior_state`. All the run-dir layout, config reading and YAML
parsing lives in this script: `libs/evaluation` is handed opened datasets and
plain dicts and stays a leaf.

The stage configures `logging` at INFO on its entry point, because the reason a
figure no-oped is logged by `evaluation.figures` while only the *fact* is
printed here — without a root handler the operator would see the skip and never
the why.

Honors `run.skip_viz` from the saved config (no-op if true).

#### [`run_esmda_pipeline.sh`](../scripts/run_esmda_pipeline.sh)
**Shell script** (executable). Runs all three stages in sequence.
Resolves the run dir from `conf/run_esmda.yaml` using the same Hydra overrides
forwarded to `run_esmda.py`, so the metric/figure stages automatically find the
right directory without it being pinned in the script.

```bash
scripts/run_esmda_pipeline.sh esmda/smoother=static \
    params@prior_params=static params@truth_params=static_truth
```

---

### 2.4 Filtering pipeline scripts

The filtering (EnKF) analogue of §2.3, orchestrated by
[`run_filtering_pipeline.sh`](../scripts/run_filtering_pipeline.sh). Stage 1 is
[`run_filtering.py`](../scripts/filtering/run_filtering.py) (§2.1). The metric and
figure stages reuse the ESMDA truth-access / sensor-series helpers via
[`_filtering_common.py`](../scripts/filtering/_filtering_common.py), which adapts
them to the filter's per-**cycle** time axis (the truth is compared at each
cycle's end-of-segment frame).

**Cycle state sources.** The ESMDA evaluation blocks all reduce over the
ensemble states of one assimilation step, and ESMDA has exactly one thing to
reduce (`windows/window_{w}_posterior_state.nc`). The filter has two, and
`_filtering_common.cycle_state_source` picks between them — the choice is
recorded in `run_summary.yaml`'s `cycle_states` block, so the numbers always say
which they came from:

| Source | When | What you get |
|---|---|---|
| `forecast` | `run.ensemble_save_on_disk=true` | Every member's full forecast segment, kept under `_ensemble_states/cycle_{k}/state_{m}.nc` and never pruned. Cycle *k*'s forecast is the ensemble rolled out from cycle *k−1*'s analysis under its analyzed parameters — the exact analogue of an ESMDA window file, and out-of-sample. Within-cycle variance and real resolved turbulence. |
| `analysis` | default | `state_history.nc` alone: **one** analyzed frame per cycle. Every block still runs, but the per-cycle *variance* statistic is null (a `ddof=1` variance of one sample) and the TKE / `<u'w'>` moments are taken *across* cycles, so they carry the analysis increments — read those panels as an upper bound. |

Both are frame-matched against the truth by the same contract, so the statistics
compare like with like either way.

#### [`compute_filtering_metrics.py`](../scripts/filtering/compute_filtering_metrics.py)
**Plain argparse CLI** — usage: `python scripts/filtering/compute_filtering_metrics.py --run-dir <dir>`

Stage 2. Reads the artifacts saved by `run_filtering.py` and writes
`run_summary.yaml` — the `run_info` metadata augmented with:
- `metrics_version` — estimator-semantics marker, shared with the ESMDA
  pipeline (§2.3).
- `filter_diagnostics` — summary stats of the per-cycle innovation χ² and
  observation-space prior/posterior RMSE (always available; every mode).
- `parameter_metrics` — per-parameter RMSE/CRPS of the final analyzed ensemble
  + RMSE reduction and CRPSS vs prior, and the same calibration entries as the
  ESMDA summary (§2.3) (absent in `mode=state`).
- `state_metrics` — per-cycle `|U|` field RMSE vs the truth's end-of-cycle frames.
- `sensor_metrics` — full-vector (u, v, w) RMSE and energy score per sensor set.
- `ensemble_health` — duplicate-member counts of the analyzed parameter
  ensemble, run-wide and per cycle (`n_unique_per_cycle`, read out of
  `params_history.nc`) (absent in `mode=state`).
- `cycle_states` — which of the two cycle state sources above the three blocks
  below reduced over, with its cycle/member counts and a one-line caveat.
- `sensor_statistics` — per sensor set, the per-cycle mean and variance of
  u/v/w/`|U|` scored as the verification object (fair CRPS, z-score, rank
  counts, identifiability) — the statistics-space counterpart of
  `sensor_metrics` (§2.3). Posterior half only: the filter saves no comparable
  prior rollout. Its `n_windows` key is the shared library's name for the bin
  count, which here is the cycle count.
- `field_metrics` — the VDI 3783/9 hit rate `q` of the time-mean velocity field
  over evenly spaced z-slabs, with `W` block-bootstrapped from the truth's own
  sampling error. Writes `eval_fields.nc` beside the summary for the figure
  stage, stamping its `moment_sampling` attribute (§2.3) with the cycle-state
  source's own `description` — the same line the `cycle_states` block above
  carries, put where the figure stage will actually see it — and its
  `moment_sampling_is_sparse` attribute with `source.kind != "forecast"`, i.e.
  `1` under the default `analysis` source and `0` under `forecast`, whose
  segments tile the run and whose moments therefore *are* a continuous time
  average. That flag is what S1 and F1 branch on (§2.3). `n_cycles` is the
  run's cycle count; `n_windows` beside it is the number of accumulator chunks,
  which is 1 under the `analysis` source.

#### [`make_filtering_figures.py`](../scripts/filtering/make_filtering_figures.py)
**Plain argparse CLI** — usage: `python scripts/filtering/make_filtering_figures.py --run-dir <dir>`

Stage 3. Reads artifacts and writes into the run directory:
- `parameter_evolution.png` — parameter trajectories over cycles + per-cycle `|U|` RMSE.
- `parameter_error.png` — per-parameter posterior error over cycles.
- `rollout_animation.mp4` — analyzed ensemble-mean `|U|` field vs truth (one frame/cycle).
- `final_state_with_obs.png` — final analyzed `|U|` field with sensor locations.
- `sensor_timeseries_<set>.png` — truth vs ensemble at each sensor set.
- `parameter_marginals.png` (P1) — prior vs posterior marginal per parameter.
- `station_profiles.png` (S1) — mean-velocity / TKE profiles at the sensor
  columns (needs `eval_fields.nc`).
- `mean_slices.png` (F1) — time-mean field slices, truth vs posterior vs
  difference (needs `eval_fields.nc`).
- `sensor_fans.png` (S5) — sensor quantile fans with the observation-error
  envelope, assimilated and held-out columns side by side.
- `rank_histogram.png` (D1) — rank histogram of the per-cycle statistics (needs
  `run_summary.yaml`'s `sensor_statistics`).

The parameter figures are skipped in `mode=state` (no parameters estimated).
The last five mirror the ESMDA stage's WP1.5 figures (§2.3); each degrades to a
printed skip line when the artifact it reads is absent, so a run dir whose metric
stage predates them still gets every figure it can support, and a skip never
costs the figures after it. There is no filtering counterpart to the ESMDA
suite's `probe_spectra.png` (S4): it needs the high-rate probe records only a
dedicated solver rerun (`scripts/esmda/run_probe_series.py`) writes, and the
filtering pipeline has no such script.

#### [`run_filtering_pipeline.sh`](../scripts/run_filtering_pipeline.sh)
**Shell script** (executable). Runs all three stages in sequence, resolving the
run dir from `conf/run_filtering.yaml` with the same Hydra overrides forwarded to
`run_filtering.py`.

```bash
scripts/run_filtering_pipeline.sh filtering.mode=joint filtering.num_cycles=4
```

---

### 2.5 Ground-truth artifact utilities

Located in [`scripts/adjust_simulations/`](../scripts/adjust_simulations/).
All are **plain argparse or zero-argument CLIs** — not Hydra.

#### [`adjust_simulations/trim_spinup.py`](../scripts/adjust_simulations/trim_spinup.py)

Drops spin-up frames from a `state.nc`/`params.nc` pair and rebases time to `t=0`.
Streams the state in time-chunks via netCDF4 so multi-GB files never fully reside
in RAM. Output directory can be fed directly to `run_esmda.py run.truth_dir=<path>`.

```bash
python scripts/adjust_simulations/trim_spinup.py \
    --state ground_truth/state.nc \
    --params ground_truth/params.nc \
    --spinup-time 25 --output-dir ground_truth_spunup
```

#### [`adjust_simulations/convert_ground_truth_to_32bit.py`](../scripts/adjust_simulations/convert_ground_truth_to_32bit.py)

Copies all `*.nc` files from `ground_truth/64_bit/` to `ground_truth/32_bit/`,
downcasting `float64 → float32`. Streams slice-by-slice along the unlimited
dimension. No CLI arguments — source and destination directories are hardcoded
relative to the script location.

#### [`adjust_simulations/regenerate_ground_truth_params.py`](../scripts/adjust_simulations/regenerate_ground_truth_params.py)

Regenerates `params.nc` from a past run's Hydra overrides without re-running the
solver, using `create_time_varying_true_params` from `hydra_helpers.py`. Useful
when the params file is lost but the config + seed are known.

#### [`adjust_simulations/make_state_small.py`](../scripts/adjust_simulations/make_state_small.py)

Creates a reduced copy of `state.nc` — keeps only `u, v, w` variables and a
specified time range (`[0, 1000]` s by default), streaming in batches. Input/output
paths are hardcoded. Used to shrink a multi-GB full-resolution truth into a
manageable artifact.

---

### 2.6 Figure creation pipeline

Located in [`scripts/figure_creation/`](../scripts/figure_creation/).
All are **plain argparse CLIs** unless noted. These scripts operate on already-saved
ESMDA run artifacts; they do not re-run the forward model.

#### [`figure_creation/visualize_run.py`](../scripts/figure_creation/visualize_run.py)

Generates a consolidated figure set for a **single parameter-estimation ESMDA run**
from its saved small artifacts (parameter NetCDFs, posterior mean rollout,
`run_summary.yaml`). Writes to `result_figures/<case>/`. Optionally loads one final
frame of the truth for a truth-vs-posterior comparison panel.

Figures: `parameter_trajectories.png`, `parameter_error.png`,
`parameter_metrics.png`, `final_state.png`, `state_montage.png`,
`metrics_summary.png`.

#### [`figure_creation/visualize_state_run.py`](../scripts/figure_creation/visualize_state_run.py)

State-estimation counterpart to `visualize_run.py`. Adds state-specific figures
from the per-window ensemble state hyperslabs:
- `state_spread_reduction.png` — per-window ensemble spread, prior vs posterior.
- Handles both `_ic` (IC-only) and `_all` (full trajectory smoothing) flavours.
- Three state-update methods: `svd`, `localization_corr`, `localization_dist_dist`.

#### [`figure_creation/compute_sweep_metrics.py`](../scripts/figure_creation/compute_sweep_metrics.py)

Middle stage of a **sweep comparison pipeline** (runs across ensemble size or domain
resolution). Computes every metric + metric time series from ESMDA posterior results
and writes small artifacts to `pyurbanair/sweep_metrics/<run>/`:
- `metrics.yaml` — `metrics_version` (§2.3) + configuration + parameter/state/sensor
  metrics (u/v/w + `|U|` per component, per sensor set). Every ensemble score in
  it is recomputed here by the current estimators, so the marker is always the
  current version. A run without `truth_access.yaml` (produced before that file
  existed) cannot have its sensor scores recomputed, so what is carried over
  from its `run_summary.yaml` is decided by an **allowlist**
  (`_CARRYABLE_SENSOR_KEYS`): `num_sensors` plus the RMSE keys
  (`vel_magnitude_rmse`, `u/v/w_rmse`, `velocity_vector_rmse`), each of them
  estimator-*independent* and therefore bit-identical across the WP1.1 switch.
  Everything else is dropped rather than mixed into a version-2 file — including
  `velocity_vector_energy_score`, which the suffix denylist this replaced
  (`not k.endswith("_crps")`) let through even though it is a biased `M²`
  pairwise score. An allowlist fails closed: a future score is withheld until
  someone vouches for it. A carried-forward file records
  `sensor_metrics_provenance` next to the block — `recomputed: false`,
  `carried_from_metrics_version`, `dropped_keys` — so an empty CRPS panel reads
  as withheld rather than failed. Re-run ESMDA to restore them.
- `sensor_timeseries_<set>.nc` — truth + prior/posterior ensemble series (small;
  no full fields).
- Copies of `posterior_params.nc`, `prior_params.nc`, `true_params.nc`.

#### [`figure_creation/compare_sweep_results.py`](../scripts/figure_creation/compare_sweep_results.py)

Final stage of the sweep pipeline. Reads `pyurbanair/sweep_metrics/` and draws
comparison figures + a summary CSV. `--sweep domain` compares across grid cells;
`--sweep ensemble` compares across ensemble sizes. `--sweep all` does both.
Warns when the runs it is about to compare carry different `metrics_version`
markers (§2.3), since their CRPS / energy scores are then incomparable.

#### [`figure_creation/compare_state_runs.py`](../scripts/figure_creation/compare_state_runs.py)

Compares multiple **state-estimation** ESMDA runs on shared metrics. Reads each
run's `run_summary.yaml`. Groups bars by method (`svd`/`localization_corr`/
`localization_dist_dist`) and labels by mode. `--mode ic|all|both` filter.
Warns on mixed `metrics_version` markers (§2.3), as the sweep comparison does.

#### [`figure_creation/compare_param_vs_state.py`](../scripts/figure_creation/compare_param_vs_state.py)

Compares parameter-only ESMDA against state+parameter ESMDA runs across the
estimation type boundary. Classifies runs into three categories: `param_only`,
`state_ic`, `state_all`. Runs from different directories can be mixed as CLI
arguments.

#### [`figure_creation/compare_localization.sh`](../scripts/figure_creation/compare_localization.sh)

**Shell script** — runs `run_esmda.py` multiple times (currently SVD-IC,
SVD-snapshot+final-smoothing, and parameter-only baseline) with identical settings
and prints a comparison table from each run's `run_summary.yaml`. Configurable
via env vars: `SIZE`, `TRUTH_DIR`, `TRUTH_MODEL`, `ASSIM_MODEL`,
`ENSEMBLE_SIZE`, `NUM_STEPS`, `NUM_WINDOWS`, `SVD_ENERGY`, etc.

#### Other figure scripts

- [`figure_creation/make_all_figures.py`](../scripts/figure_creation/make_all_figures.py) — orchestrates the
  full EnKF-2026 figure pipeline (block drivers in sequence → summary → notes).
  Pass `--heavy` to include expensive panels.
- [`figure_creation/make_figures_block_a.py`](../scripts/figure_creation/make_figures_block_a.py) /
  [`block_b.py`](../scripts/figure_creation/make_figures_block_b.py) /
  [`block_c.py`](../scripts/figure_creation/make_figures_block_c.py) — per-block figure drivers.
- [`figure_creation/make_figures_summary.py`](../scripts/figure_creation/make_figures_summary.py) — summary metrics figures.
- [`figure_creation/make_notes.py`](../scripts/figure_creation/make_notes.py) — auto-generated notes page.
- [`figure_creation/make_animations.py`](../scripts/figure_creation/make_animations.py) — animation rendering (requires ffmpeg).
- [`figure_creation/plot_state_slices.py`](../scripts/figure_creation/plot_state_slices.py) — 2D velocity
  slices for a given run directory.
- [`figure_creation/visualize_ground_truth.py`](../scripts/figure_creation/visualize_ground_truth.py) —
  diagnostic figures for a truth artifact directory (params, field snapshot,
  derived inflow angle vs prescribed).

---

### 2.7 `figspec/` — shared figure primitives

Located in [`scripts/figspec/`](../scripts/figspec/). A small internal library
imported by the block drivers in `figure_creation/`.

| Module | Purpose |
|---|---|
| [`figspec/dataio.py`](../scripts/figspec/dataio.py) | I/O helpers: lazy truth dataset access, loading parameter/state artifacts, grid metadata. |
| [`figspec/figcommon.py`](../scripts/figspec/figcommon.py) | Reusable plotting primitives (parameter-trajectory panels, error-vs-time lines, field + difference heatmap grids). |
| [`figspec/mask.py`](../scripts/figspec/mask.py) | Building-mask utilities for field plots (fluid/obstacle masking); binds `evaluation.style`'s STL geometry to this repo's data locations. |
| [`figspec/_selftest.py`](../scripts/figspec/_selftest.py) | Quick smoke test for the figspec library. |

The Matplotlib style constants and the metric definitions the block drivers use
live in the `evaluation` library instead:
[`evaluation.style`](../libs/evaluation/src/evaluation/style.py) and
[`evaluation.scores`](../libs/evaluation/src/evaluation/scores.py).

---

### 2.8 Neural surrogate scripts

Located in [`scripts/neural_surrogate/`](../scripts/neural_surrogate/).
Full documentation is in [`docs/neural_surrogates.md`](neural_surrogates.md).
Brief summary:

| Script | Hydra? | Purpose |
|---|---|---|
| [`neural_surrogate/generate_training_data.py`](../scripts/neural_surrogate/generate_training_data.py) | Yes — [`neural_surrogate/training_data.yaml`](../conf/neural_surrogate/training_data.yaml) | Build `train/val/test` dataset from a CFD ensemble on ONE fixed geometry (`training_data.geometry.source=barcelona\|xie_and_castro`). Configures the ensemble with `failure: raise`. |
| [`neural_surrogate/generate_random_geometries_training_data.py`](../scripts/neural_surrogate/generate_random_geometries_training_data.py) | Yes — [`neural_surrogate/training_data.yaml`](../conf/neural_surrogate/training_data.yaml) | Same dataset layout over randomly sampled pool geometries (`source=idealized\|realistic`): per-geometry grid from STL bounds at `geometry.resolution` (nx/ny padded to multiples of 16, fixed `z_size`), geometry-disjoint val/test, direct sequential single-model runs (one prepared forward model per geometry, no ensemble machinery). See `docs/neural_surrogates.md` §2b. |
| [`neural_surrogate/train_neural_surrogate.py`](../scripts/neural_surrogate/train_neural_surrogate.py) | Yes — [`neural_surrogate/training.yaml`](../conf/neural_surrogate/training.yaml) | Train a surrogate; bakes normalization stats into the checkpoint. Writes `model_weights/<model_name>/`. |
| [`neural_surrogate/finetune_neural_surrogate.py`](../scripts/neural_surrogate/finetune_neural_surrogate.py) | Yes — [`neural_surrogate/finetuning.yaml`](../conf/neural_surrogate/finetuning.yaml) | LoRA fine-tune a trained surrogate (plan 01): inject adapters, train only the adapter weights, export a merged `model_dir` (+ `adapter/`) that loads into `NeuralSurrogateForwardModel` unchanged. See `docs/neural_surrogates.md` §21–25. |
| [`neural_surrogate/test_neural_surrogate.py`](../scripts/neural_surrogate/test_neural_surrogate.py) | Yes — [`neural_surrogate/testing.yaml`](../conf/neural_surrogate/testing.yaml) | Autoregressive rollout on the test split; writes trajectory tensors, rollout PNG, RMSE plot, animation. |
| [`neural_surrogate/compare_surrogate_models.py`](../scripts/neural_surrogate/compare_surrogate_models.py) | Yes — [`neural_surrogate/comparison.yaml`](../conf/neural_surrogate/comparison.yaml) | Roll out several trained models on the same trajectories; writes overlaid per-step RMSE, stacked truth/pred/`|err|` slice grids + animation, a summary-metrics bar chart, and `metrics.csv`. |
| [`neural_surrogate/dataloading.py`](../scripts/neural_surrogate/dataloading.py) | No — argparse | `TransitionDataset` smoke test: builds a DataLoader, prints batch shapes, writes diagnostic plots (`states.png`, `params.png`, `geometry.png`). |
| [`neural_surrogate/add_geometry_to_training_data.py`](../scripts/neural_surrogate/add_geometry_to_training_data.py) | No — argparse | Post-processes an existing dataset to add a geometry variable to each state file. |
| [`neural_surrogate/finish_training_data_processing.py`](../scripts/neural_surrogate/finish_training_data_processing.py) | No — argparse | Finalizes a partially-generated dataset (e.g. after a cluster job restart). |

---

## Quick lookup

| You want to… | Go to |
|---|---|
| Add a new experiment (domain/sensors/geometry) | [`conf/case/`](../conf/case/) — one YAML per case |
| Change run size (ensemble, ESMDA steps, windows) | CLI overrides on `ensemble.*`/`esmda.*`; or edit the inlined blocks in [`run_esmda.yaml`](../conf/run_esmda.yaml) |
| Switch CFD backend | `model@model=pylbm|pyudales|pypalm|neural_surrogate` (fwd) or `model@truth_model=...` + `model@assim_model=...` (esmda) |
| Change DA mode | `esmda/smoother=static|state|state_and_parameter|dynamic|state_and_dynamic` |
| Run a sequential filter (EnKF) instead of ESMDA | [`scripts/filtering/run_filtering.py`](../scripts/filtering/run_filtering.py) — `filtering.mode=state|parameter|joint` + `filtering/*` groups (§1.8) |
| Filter the state but smooth a parameter *trajectory* over the window | [`scripts/filter_smoothing/run_filter_smoothing.py`](../scripts/filter_smoothing/run_filter_smoothing.py) — `filter_smoothing/*` groups (§1.9), `params@prior_params=dynamic` required |
| Enable localization | `esmda/localization=correlation|distance` (smoother) or `filtering/localization=...` (filter) + optional field overrides |
| Enable reduced state update | `esmda/state_reduction=svd` (requires state-bearing smoother, incompatible with localization) |
| Run the full ESMDA pipeline | [`scripts/run_esmda_pipeline.sh`](../scripts/run_esmda_pipeline.sh) |
| Run the full filtering pipeline | [`scripts/run_filtering_pipeline.sh`](../scripts/run_filtering_pipeline.sh) |
| Train a surrogate | [`scripts/neural_surrogate/train_neural_surrogate.py`](../scripts/neural_surrogate/train_neural_surrogate.py) — see [`docs/neural_surrogates.md`](neural_surrogates.md) |
| LoRA fine-tune a trained surrogate | [`scripts/neural_surrogate/finetune_neural_surrogate.py`](../scripts/neural_surrogate/finetune_neural_surrogate.py) — see [`docs/neural_surrogates.md` Part F](neural_surrogates.md#part-f--parameter-efficient-fine-tuning-lora--peft) |
| Understand config groups at a glance | [`conf/README.md`](../conf/README.md) |
| Understand the data-assimilation abstractions | [`docs/codebase_guide.md §6`](codebase_guide.md) |
