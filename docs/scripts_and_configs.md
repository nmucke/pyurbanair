# pyurbanair — Scripts and Configuration Reference

Detailed reference for [`conf/`](../conf/) (Hydra configs) and [`scripts/`](../scripts/)
(executable entry points). Complements the high-level summary in
[`codebase_guide.md §5`](codebase_guide.md) and the existing
[`conf/README.md`](../conf/README.md) overview — read those first for
orientation, then return here for field-level detail.

---

## Part 1 — `conf/`: Hydra configuration tree

### Overview

The configuration tree has exactly **four primary run entry points**, each
self-contained (they inline the shared base rather than pulling separate
`paths.yaml`/`time.yaml`/`ensemble.yaml` files):

| Entry point | Script | What it adds |
|---|---|---|
| [`conf/run_forward_model.yaml`](../conf/run_forward_model.yaml) | `run_forward_model.py` | `case` + single `model@model` mount + single `params` mount |
| [`conf/run_esmda.yaml`](../conf/run_esmda.yaml) | `run_esmda.py` | same base + `esmda:` scalars + double model mount (`@truth_model`/`@assim_model`) + double params mount (`@truth_params`/`@prior_params`) |
| [`conf/run_filtering.yaml`](../conf/run_filtering.yaml) | `run_filtering.py` | same base + `filtering:` scalars + the `filtering/*` groups + the same double model/params mounts (static params only) |
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
| `analysis` | (group) | Set by `filtering/analysis` (default `stochastic`). |
| `localization` | (group) | Set by `filtering/localization` (default `none`). |
| `inflation` | (group) | Set by `filtering/inflation` (default `rtps`). |
| `parameter_evolution` | (group) | Set by `filtering/evolution` (default `none`). |
| `filter` | `EnsembleKalmanFilter` block | The composed filter `_target_`; normally left alone. |

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
| `metrics` | (block, esmda only) | Post-processing depth for stages 2–3; see below. Ignored by the run stage. |

##### `run.metrics:` (esmda only)

Read by [`compute_esmda_metrics.py`](#compute_esmda_metricspy) and
`make_esmda_figures.py`, never by the run stage. Saved with the run, so
re-processing a run dir reuses its own settings; run dirs written before this
block existed simply fall back to the defaults below.

| Field | Default | Purpose |
|---|---|---|
| `level` | `standard` | `basic` = the pre-phase-1 summary only; `standard` adds the evaluation layers (parameter calibration bundle, statistics-space sensor scoring, mean-field/Reynolds-stress); `full` is reserved for later phases and currently equals `standard`. Unknown values raise. |
| `n_z_slices` | `4` | Evenly-spaced z-levels the mean-field layer accumulates on. |
| `mean_field_stride` | `1` | Spatial stride for the hit-rate / NMSE maps (scores only comparable at equal stride). |
| `bootstrap_blocks` | `20` | Blocks for the block-bootstrap sampling-error bars. |
| `stations` | `null` | `[[x, y], ...]` columns for the profile figures; `null` = the obs config's sensor x/y. |

`n_z_slices`, `mean_field_stride` and `bootstrap_blocks` must each be `>= 1`:
like an unknown `level`, a smaller value raises when the block is resolved,
rather than surfacing as an empty slice deep inside the layer that reads it.

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

> **Naming note:** The actual filenames are `static`, `state_and_parameter`,
> `dynamic`, `state_and_dynamic`. The `run_esmda.yaml` header and docstring use
> these names. The `run_esmda.py` docstring also uses these names as the CLI values.

| File | `_target_` class | What it estimates |
|---|---|---|
| [`esmda/smoother/static.yaml`](../conf/esmda/smoother/static.yaml) | `data_assimilation.smoothing.esmda.ParameterESMDA` | Static scalar parameters only. |
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

State localization is applied to **state rows only**; parameter rows always get
the global update. `distance` requires a state-bearing smoother
(`state_and_parameter` or `state_and_dynamic`) and is incompatible with
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
| [`filtering/analysis/`](../conf/filtering/analysis/) | `stochastic` | `filtering.analysis` — the update math (`StochasticEnKFAnalysis`; ETKF/LETKF will land here) |
| [`filtering/localization/`](../conf/filtering/localization/) | `none`, `correlation`, `distance` | `filtering.localization` — the same strategies as `esmda/localization` (reused unchanged); `distance` needs `filtering.mode=state\|joint` |
| [`filtering/inflation/`](../conf/filtering/inflation/) | `rtps`, `none`, `multiplicative`, `rtpp` | `filtering.inflation` — ensemble spread maintenance |
| [`filtering/evolution/`](../conf/filtering/evolution/) | `none`, `random_walk` | `filtering.parameter_evolution` — the parameters' forecast model between cycles |

See [data_assimilation.md](data_assimilation.md) for the filtering library
itself (`BaseFilter` / `EnsembleKalmanFilter`, cycle semantics, diagnostics).

---

### 1.9 Config group: `neural_surrogate/`

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

Mode is the cross product of:
- `esmda/smoother=static|state_and_parameter|dynamic|state_and_dynamic`
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
`filtering/*` groups (§1.8). Truth source (`run.truth_dir`) mirrors
`run_esmda.py`. Static scalar parameters only — a dynamic (AR(2)) params
mount fails loudly; time-varying priors stay with the ESMDA smoothers.

Stage 1 of the single-run filtering pipeline (§2.5), orchestrated by
[`run_filtering_pipeline.sh`](../scripts/run_filtering_pipeline.sh).

Saves: `posterior_params.nc` / `posterior_state.nc` (analyzed final-frame
ensemble), optional `params_history.nc` / `state_history.nc`
(`run.save_history`), per-cycle `cycle_diagnostics.yaml` (innovation χ²,
obs-space prior/posterior RMSE, block spreads), `prior_params.nc`,
`true_params.nc`, `true_state.nc` (inline truth), `truth_access.yaml` (the
lazy-truth slicing/offsets the metric/figure stages read back), `run_info.yaml`,
`config.yaml`.

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
- DA metric utilities (CRPS, per-knot error/spread/in-band, summary scalars).

#### [`_esmda_common.py`](../scripts/esmda/_esmda_common.py)

Post-processing helpers shared by `compute_esmda_metrics.py` and
`make_esmda_figures.py` (and reused by the filtering pipeline's
[`_filtering_common.py`](../scripts/filtering/_filtering_common.py), §2.5).
Read-only with respect to the run directory except explicit write calls.
Provides:
- `load_run_config(run_dir)` — re-load the Hydra config saved by `run_esmda.py`.
- `build_sensor_sets(cfg)` — assimilation + optional validation sensor coordinates
  from the obs config.
- `open_truth(cfg, ta)` — lazy truth access (multi-GB `state.nc` never fully loaded).
- `ensemble_sensor_series(...)` / `truth_sensor_series(...)` — interpolate ensemble
  and truth states at sensor locations.
- `streaming_state_rmse(...)` — streamed z-slice RMSE without loading the full field.
- `parameter_metric_summary(...)` / `vector_sensor_metrics(...)` — scalar summaries
  written to `run_summary.yaml`.
- `parameter_bundle_summary(...)` — the WP1.1 calibration bundle (z-score /
  PIT / coverage / contraction / joint directions) layered additively onto the
  above. The scoring math itself lives in
  [`ensemble_scores.py`](../src/pyurbanair/utils/ensemble_scores.py); this
  module is orchestration, and it reads every calibration *reference* from
  there rather than re-deriving it.
- `sensor_statistic_scores(truth_series, ensemble_series, *, num_windows,
  n_per_window, sim_time, bootstrap_blocks, prior_series=None)` — the WP1.2
  statistics-space sensor bundle. Takes the series the caller already extracted
  and **does no file IO**, so the extraction can be restreamed later without
  touching the scoring. Both windowing rules are arguments because the two
  series are sliced differently: the ensemble by *time value*
  (`[w·sim_time, (w+1)·sim_time)`, the rebasing `ensemble_sensor_series`
  applies), the truth by *frame index* (`slice(w·n_per_window, …)`). Sampling
  noise comes from
  [`turbulence_stats.block_bootstrap_std`](../src/pyurbanair/utils/turbulence_stats.py)
  and the distances from `ensemble_scores`' Wasserstein family.

---

### 2.3 ESMDA pipeline scripts

These three scripts form the standard single-run pipeline, orchestrated by
[`run_esmda_pipeline.sh`](../scripts/run_esmda_pipeline.sh).

#### [`compute_esmda_metrics.py`](../scripts/esmda/compute_esmda_metrics.py)
**Plain argparse CLI** — usage: `python scripts/esmda/compute_esmda_metrics.py --run-dir <dir> [--metrics-level basic|standard|full]`

Stage 2 of the pipeline. Reads the artifacts saved by `run_esmda.py` and writes
`run_summary.yaml` — the `run_info` metadata augmented with:
- `metrics_version: 2` — fair finite-ensemble CRPS/energy-score and corrected
  spread semantics (absent/1 denotes the older estimators).
- `metrics_level` — the resolved [`run.metrics.level`](#runmetrics-esmda-only)
  this summary was produced at, so *which layers were computed* is recorded
  rather than inferred. Without it an absent key is ambiguous three ways: a run
  dir processed before phase 1, one processed at `basic`, and a layer that
  no-op'd on missing inputs. Read it before comparing runs — `--metrics-level`
  makes mixed-depth reprocessing of one sweep easy. Absent = pre-phase-1.
- `parameter_metrics` — per-parameter RMSE/CRPS summary + RMSE reduction and
  CRPSS vs prior.
- `ensemble_health` — exact posterior-member uniqueness, per-window unique
  counts, and the minimum/median pairwise-distance ratio.
- `state_metrics` — `|U|` field RMSE summary (streamed z-slice by z-slice).
- `sensor_metrics` — full-vector (u, v, w) RMSE and energy score per sensor set
  (assimilation + validation).
- `sensor_statistics` — the same sensor series scored in *statistics* space
  rather than pointwise: per-window mean / variance / TKE / mean `|U|`, a
  Wasserstein distribution distance, and an identifiability guard.
  `standard` and above only, and it sits *after* the `run.skip_viz` early
  return because it consumes the truth.

Which of those layers run is gated by [`run.metrics.level`](#runmetrics-esmda-only)
from the saved config: `basic` writes exactly the keys above, higher levels add
keys on top (never change existing ones). `--metrics-level` overrides the saved
config for one invocation, so an existing run dir can be re-processed at another
depth (e.g. `basic` to line up with runs post-processed before the evaluation
layers existed) without editing its `config.yaml`. Run dirs saved before the
`run.metrics` block existed resolve to the shipped defaults.

##### `standard`: the parameter calibration bundle (WP1.1)

At `level: standard` each `parameter_metrics.<name>` entry additionally carries
the keys below, and a sibling `parameter_metrics.joint` appears. All of it is
computed before the `run.skip_viz` early return (it reads only the parameter
artifacts), so the fast sweep path gets it too. **Every key is `null` — not
absent — when `M < 3`**, where a `ddof=1` spread, an order-statistic band and a
10-bin PIT are all artifacts of the ensemble size.

The one rule that matters when reading any of it: **a calibration number at
production `M` is meaningless without the reference emitted next to it.** None
of the textbook large-sample references apply, and every one of them errs toward
calling a perfectly calibrated ensemble broken.

| Key | Meaning |
|---|---|
| `zscore.{mean,std,max_abs}` | Pooled per-knot `(truth − mean_m)/std_m`. |
| `zscore.max_abs_calibrated_median` | Where `max\|z\|` sits for a *calibrated* ensemble of this `M` over this many knots. `max_abs` is unreadable without it — the max of `n` draws grows with `n`. |
| `zscore.exceedance` | Observed `\|z\|` tail fractions (`observed`) **beside the levels a calibrated ensemble scores** (`nominal`). The null is `sqrt((M+1)/M)·t(M−1)`, *not* normal: at `M = 32`, `P(\|z\| > 3) = 0.59%`, 2.2× the normal table's 0.27%. `nominal_normal` is the large-`M` limit, for context only. Compare `observed` against `nominal`. |
| `zscore.overconfident` | Screening boolean: `observed[0] > 2 × nominal[0]` (the `\|z\| > 2` cut). Not a test — the pooled knots are correlated. |
| `zscore.overconfident_rule` | The rule as a string, naming the keys it is computed from. |
| `pit_counts` / `pit` | 10-bin rank histogram + metadata. **Divide by `pit.ranks_per_bin`, never by a flat `n/n_bins`**: ranks take `M + 1` values, which rarely divides into 10, so at `M = 32` a calibrated ensemble shows a fixed +21%/−9% three-bin comb. |
| `sampling.{n_samples,n_knots_effective,pooling}` | The pooling caveat, on the entry because it qualifies `pit`, `zscore.exceedance` **and** `coverage` alike — all three pool over the same correlated knots. `n_knots_effective` is a stated *upper bound* on independence. Mirrored under `pit` for schema stability. |
| `coverage.alpha_{50,90}` | Empirical coverage of the central order-statistic band. |
| `coverage.nominal_alpha_{50,90}` | **What a calibrated ensemble actually scores** — compare against this, never against 0.5/0.9. Band edges are order statistics, so attainable levels are multiples of `1/(M+1)`: at `M = 32`, `alpha = 0.5` is nominal 0.4848. |
| `coverage.max_nominal_alpha` | `(M−1)/(M+1)`, the widest band this `M` offers, so a clamped `alpha_90` does not read as a failure. |
| `contraction_ratio.vs_window_prior` | `{mean, min}` of `std_post/std_prior` knot by knot. **Per-window** contraction: `run_esmda.py` seeds window `w`'s prior from window `w−1`'s posterior, so only block 0 of `prior_params.nc` is a genuine prior. |
| `contraction_ratio.vs_initial_prior` | `{mean, min, reason}` against window 0's prior block — the **cumulative** contraction, i.e. what "how much did assimilation shrink the uncertainty" means. The two differ by the window count: a 3-window run with a true per-window ratio of 0.6 reports 0.600 per window and 0.392/0.216 cumulatively. `reason` is non-null exactly when the numbers are null. |
| `contraction_ratio.{mean,min}` | Aliases of `vs_window_prior`, retained for existing consumers. |

`parameter_metrics.joint` scores the flattened `(M, K)` parameter vector:
`n_constrained_directions` (generalized eigenvalues `λ < 0.5`, i.e. directions
whose spread at least halved), `generalized_eigenvalues` /
`eigenvalue_quantiles`, `most_constrained` / `least_constrained` (eigenvalue +
top parameter-space loadings), `n_sample_directions` / `rank_deficient` /
`posterior_variance_retained` (the pencil is rank-truncated onto the prior's
eigenbasis — ensembles are routinely smaller than `K`), `corr_summary`, and
`posterior_corr` / `prior_corr` only when `K ≤ 8` (`corr_matrices_omitted` says
so otherwise). Two keys name *which* question it answers:

- `joint.prior_reference: per_window_prior` — as with `contraction_ratio`, the
  concatenated prior is a per-window reference, so the top-level
  `n_constrained_directions` counts what the per-window updates constrained.
- `joint.vs_initial_prior` — the same decomposition for the **final posterior
  window against the window-0 prior block**, i.e. what the run as a whole
  constrained. A scalar summary only (no loadings or matrices), with `reason`
  non-null when the artifact could not be split into windows.

##### `standard`: statistics-space sensor scoring (WP1.2)

At `level: standard` a top-level `sensor_statistics` mapping appears, keyed by
sensor set (`assimilation`, plus `validation` when the case defines held-out
sensors) exactly as `sensor_metrics` is. Unlike the WP1.1 parameter bundle it
attaches *after* the `run.skip_viz` early return — it reuses the truth and
ensemble sensor series `sensor_metrics` already extracted, so a `skip_viz` run
has no `sensor_statistics` at all.

Why it exists next to `sensor_metrics`: the pointwise scores ask whether the
ensemble matches the truth **at each instant**, which a turbulent flow cannot be
expected to do — two realizations of identical statistics decorrelate within a
few turnover times, after which the pointwise number measures phase rather than
physics. These score quantities that *are* reproducible.

**Every key is `null` — not absent — when `M < 3` or the window count is not
usable**, with a non-null `reason` on that block and a `logger.info` naming what
was skipped. The identifiability sub-block degrades on its own (see below) while
the calibration numbers around it stay real, so a `null` there is not a null run.

Each of `window_mean`, `window_variance`, `tke` and `velmag_mean` carries the
same key set; `<stat>` below stands for any of the four.

| Key | Meaning |
|---|---|
| `<set>.{n_members,n_windows,n_sensors}` | Shape of what was scored. `n_windows` is the *requested* window count from `truth_access.yaml`, not the number that contributed — a window with no ensemble or no truth frames is dropped silently and only shrinks `<stat>.n_samples`. |
| `<stat>` = `window_mean` | Per-window time-mean of each velocity component, pooled over component × sensor × window. The one statistic a biased-inflow run moves; also the one a *long* window can match while every higher moment is wrong. |
| `<stat>` = `window_variance` | Per-window `ddof=1` time-variance of each component, pooled the same way. `null` for any window holding a single frame — one frame has no `ddof=1` variance, and at the smoke shape that is the whole run. |
| `<stat>` = `tke` | `0.5·Σ_c var(u_c)` per sensor per window. Computed as the time-mean of the *instantaneous* `0.5·n/(n−1)·Σ_c(u_c−⟨u_c⟩)²`, which makes it exactly the tabulated `ddof=1` TKE — the Bessel factor is on the integrand so the bootstrap resamples the reported estimator, not a `ddof=0` cousin. |
| `<stat>` = `velmag_mean` | Per-window time-mean of `\|U\|`, per sensor. Not implied by `window_mean`: `⟨\|U\|⟩ ≥ \|⟨U⟩\|`, and the gap is exactly the fluctuation a directionally-wandering flow carries. |
| `<stat>.crps` | Fair (finite-`M` debiased) CRPS of the ensemble's statistic against the truth's, averaged over pooled elements. **In the statistic's own units** — variance and TKE are in m²/s², so the four are not comparable to each other. |
| `<stat>.crpss_vs_prior` | Skill against the prior ensemble's same statistic. **`null` on nearly every run**: it needs `windows/window_*_prior_state.nc` for *every* window, and `conf/run_esmda.yaml` ships `run.save_prior_state: false`. `null` here means "not saved", never "no skill". |
| `<stat>.zscore.{mean,std,max_abs}` | Pooled `(truth − mean_m)/std_m` over the pooled elements. A calibrated ensemble has `std ≈ 1`, but only in expectation — the pooled elements here are **not** independent (one window's sensors see one realization of one flow), so the sampling error on these is larger than `n_samples` suggests. |
| `<stat>.zscore.max_abs_calibrated_median` | Where `max\|z\|` sits for a *calibrated* ensemble of this `M` over this many pooled elements. `max_abs` is unreadable without it — the max of `n` draws grows with `n`, so a bigger sensor set alone raises it. |
| `<stat>.coverage.alpha_{50,90}` | Empirical coverage of the central order-statistic band over the pooled elements. |
| `<stat>.coverage.nominal_alpha_{50,90}` | **What a calibrated ensemble of this `M` actually scores** — compare against this, never against 0.5/0.9. Band edges are member order statistics, so attainable levels are multiples of `1/(M+1)`. |
| `<stat>.coverage.max_nominal_alpha` | `(M−1)/(M+1)`, the widest band this `M` offers, so a clamped `alpha_90` does not read as a failure. |
| `<stat>.pit_counts` / `<stat>.pit` | 10-bin rank histogram plus its metadata (`n_bins`, `n_samples`, `ranks_per_bin`, `tie_seed`). **Divide by `pit.ranks_per_bin`, never by a flat `n/n_bins`** — ranks take `M+1` values, which rarely divides into 10, so a calibrated ensemble shows a fixed comb against a flat reference. `tie_seed` is emitted so the tie-broken counts are reproducible. |
| `<stat>.identifiability.{ratio_median,ratio_min}` | Across-member spread ÷ median within-member block-bootstrap sampling std, per pooled element, reduced by median and min. **This is the number that says whether the calibration keys above mean anything**: at a ratio near 1 the members differ by no more than one member's own finite-window sampling noise, so CRPS/coverage/PIT are scoring the averaging window rather than the assimilation. |
| `<stat>.identifiability.sampling_noise_dominated` | `ratio_median < threshold`. A screen on *how to read the block*, never a gate — the numbers are reported either way. `null` (not `false`) when the ratio is unknown. |
| `<stat>.identifiability.threshold` | `IDENTIFIABILITY_MIN_RATIO = 3.0`; at `r` the within-member sampling variance is `1/r²` of the across-member variance, so 3 is where sampling noise stops contributing more than ~11% of the scored spread. A constant, so it survives the null path. |
| `<stat>.n_samples` | Pooled elements that actually scored (non-finite truth or members dropped once, before every score, so all of them see the same sample). **Not** an effective sample size: these elements are correlated. |
| `<stat>.reason` | Non-null exactly when the statistic's numbers are null, naming which degradation fired. |
| `wasserstein.w1_over_sigma_pooled` | `{median, max}` over sensors of `W1(all members' \|U\| samples pooled, truth \|U\|)/σ_truth`, over the **whole run** — a window holds too few frames for a distance between empirical distributions to mean anything. This is the ensemble's *predictive* distribution. |
| `wasserstein.w1_over_sigma_member_mean` | The same distance with each member scored alone, then averaged. Read it **with** the pooled number, never instead of it: `W1` is convex in its first argument, so pooled ≤ member-mean always. They coincide under a shared bias (pooling cannot cancel it) and separate when the spread brackets the truth. |
| `wasserstein.self_floor` | `W1(first half, second half of the truth)/σ_truth` — what a *perfect* model scores at this sample count and autocorrelation. Halves are contiguous, not random, so each keeps its serial correlation; the raw distance is unreadable without this. |
| `wasserstein.w1_over_floor` | The pooled distance in units of that floor. **~1 is indistinguishable from perfect at this sample size** — this is the headline ratio, not `w1_over_sigma_pooled`. |
| `wasserstein.reason` | Non-null when any of the four reduced to `null`, naming which. A constant truth series has no `σ` and no floor; that is a property of the sensor, not a bug. |

#### [`make_esmda_figures.py`](../scripts/esmda/make_esmda_figures.py)
**Plain argparse CLI** — usage: `python scripts/esmda/make_esmda_figures.py --run-dir <dir>`

Stage 3 of the pipeline. Reads artifacts and writes into the run directory:
- `rollout_time_evolution.png` — parameter trajectories + state `|U|` RMSE.
- `parameter_error.png` — per-parameter posterior error over time.
- `rollout_animation.mp4` — ensemble-mean `|U|` field vs truth.
- `final_state_with_obs.png` — final `|U|` field with sensor locations.
- `sensor_timeseries_<set>.png` — truth vs ensemble at each sensor set.

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

#### [`compute_filtering_metrics.py`](../scripts/filtering/compute_filtering_metrics.py)
**Plain argparse CLI** — usage: `python scripts/filtering/compute_filtering_metrics.py --run-dir <dir>`

Stage 2. Reads the artifacts saved by `run_filtering.py` and writes
`run_summary.yaml` — the `run_info` metadata augmented with
`metrics_version: 2` and:
- `filter_diagnostics` — summary stats of the per-cycle innovation χ² and
  observation-space prior/posterior RMSE (always available; every mode).
- `parameter_metrics` — per-parameter RMSE/CRPS of the final analyzed ensemble
  + reduction vs prior (absent in `mode=state`).
- `state_metrics` — per-cycle `|U|` field RMSE vs the truth's end-of-cycle frames.
- `sensor_metrics` — full-vector (u, v, w) RMSE and energy score per sensor set.

#### [`make_filtering_figures.py`](../scripts/filtering/make_filtering_figures.py)
**Plain argparse CLI** — usage: `python scripts/filtering/make_filtering_figures.py --run-dir <dir>`

Stage 3. Reads artifacts and writes into the run directory:
- `parameter_evolution.png` — parameter trajectories over cycles + per-cycle `|U|` RMSE.
- `parameter_error.png` — per-parameter posterior error over cycles.
- `rollout_animation.mp4` — analyzed ensemble-mean `|U|` field vs truth (one frame/cycle).
- `final_state_with_obs.png` — final analyzed `|U|` field with sensor locations.
- `sensor_timeseries_<set>.png` — truth vs ensemble at each sensor set.

The parameter figures are skipped in `mode=state` (no parameters estimated).

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
- `metrics.yaml` — configuration + parameter/state/sensor metrics (u/v/w + `|U|`
  per component, per sensor set).
- `sensor_timeseries_<set>.nc` — truth + prior/posterior ensemble series (small;
  no full fields).
- Copies of `posterior_params.nc`, `prior_params.nc`, `true_params.nc`.

For legacy runs without `truth_access.yaml`, the stage still recomputes the
parameter bundle with version-2 estimators but omits sensor metrics: copying the
source summary's version-1 sensor scores would create a mixed-semantics
`metrics.yaml`.

#### [`figure_creation/compare_sweep_results.py`](../scripts/figure_creation/compare_sweep_results.py)

Final stage of the sweep pipeline. Reads `pyurbanair/sweep_metrics/` and draws
comparison figures + a summary CSV. `--sweep domain` compares across grid cells;
`--sweep ensemble` compares across ensemble sizes. `--sweep all` does both.

#### [`figure_creation/compare_state_runs.py`](../scripts/figure_creation/compare_state_runs.py)

Compares multiple **state-estimation** ESMDA runs on shared metrics. Reads each
run's `run_summary.yaml`. Groups bars by method (`svd`/`localization_corr`/
`localization_dist_dist`) and labels by mode. `--mode ic|all|both` filter.

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
| [`figspec/style.py`](../scripts/figspec/style.py) | Matplotlib style constants (colors, line styles, fonts) shared across all block drivers. |
| [`figspec/mask.py`](../scripts/figspec/mask.py) | Building-mask utilities for field plots (fluid/obstacle masking). |
| [`figspec/metrics.py`](../scripts/figspec/metrics.py) | Metric computation helpers used by the block drivers. |
| [`figspec/_selftest.py`](../scripts/figspec/_selftest.py) | Quick smoke test for the figspec library. |

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
| Change DA mode | `esmda/smoother=static|state_and_parameter|dynamic|state_and_dynamic` |
| Run a sequential filter (EnKF) instead of ESMDA | [`scripts/filtering/run_filtering.py`](../scripts/filtering/run_filtering.py) — `filtering.mode=state|parameter|joint` + `filtering/*` groups (§1.8) |
| Enable localization | `esmda/localization=correlation|distance` (smoother) or `filtering/localization=...` (filter) + optional field overrides |
| Enable reduced state update | `esmda/state_reduction=svd` (requires state-bearing smoother, incompatible with localization) |
| Run the full ESMDA pipeline | [`scripts/run_esmda_pipeline.sh`](../scripts/run_esmda_pipeline.sh) |
| Run the full filtering pipeline | [`scripts/run_filtering_pipeline.sh`](../scripts/run_filtering_pipeline.sh) |
| Train a surrogate | [`scripts/neural_surrogate/train_neural_surrogate.py`](../scripts/neural_surrogate/train_neural_surrogate.py) — see [`docs/neural_surrogates.md`](neural_surrogates.md) |
| LoRA fine-tune a trained surrogate | [`scripts/neural_surrogate/finetune_neural_surrogate.py`](../scripts/neural_surrogate/finetune_neural_surrogate.py) — see [`docs/neural_surrogates.md` Part F](neural_surrogates.md#part-f--parameter-efficient-fine-tuning-lora--peft) |
| Understand config groups at a glance | [`conf/README.md`](../conf/README.md) |
| Understand the data-assimilation abstractions | [`docs/codebase_guide.md §6`](codebase_guide.md) |
