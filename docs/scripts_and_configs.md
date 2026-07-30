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
  and truth states at sensor locations. `ensemble_sensor_series` **streams
  member-at-a-time** rather than loading each window file whole (it used to do
  `xarray.open_dataset(path).load()` on a file holding the entire ensemble); its
  return value is unchanged and bit-identical, memory layout included, since
  WP1.2's calibration numbers are pinned to it. It now raises on a window file
  with no `ensemble` dimension instead of silently returning an axis-less series.
- `stream_window_members(state_paths, variables=("u","v","w"))` — **the
  sanctioned reader for `windows/window_*_state.nc`**, and the one pass every
  consumer of those files hangs off. Yields `(window_index, member_index,
  member_state)`; opens each file lazily, slices `.isel(ensemble=m)`, and
  materialises **one member** (`variables=None` reads every data variable). The
  member is materialised on purpose: xarray does not cache reads taken through
  `.isel`, so a lazy yield would make `N` consumers cost `N×` the bytes and
  defeat the shared pass. That is a `.load()` at member granularity, which is
  what the never-load-a-window-file rule prescribes — the docstring says so
  rather than hiding behind the spelling.
- `member_sensor_series(...)` / `SensorSeriesAccumulator` — the per-member unit
  of `ensemble_sensor_series` and the accumulator that reassembles it, exposed
  so a caller driving its own `stream_window_members` pass can attach the sensor
  extraction to it. The window-local → global time rebasing
  (`(t − t[0]) + w·sim_time`) lives in `member_sensor_series`.
- `MeanFieldAccumulator(solver_name, n_z_slices, stride, station_x, station_y,
  state_paths=None)` — the WP1.3 ensemble half: two
  `turbulence_stats.StreamingMoments` per member (z-slabs, full-z station
  columns), fed from the same shared pass. Per-cell state only, so it is
  independent of the frame count — but **not** of `M` or the grid: the slab
  accumulators are 0.62 GB at `M = 32` / 4×256×256 and **10 GB** at the plan's
  `M = 128` / 4×512×512 target, held for the whole pass, which is what
  `mean_field_stride` is for. `state_paths` is how it reads the solid-cell mask:
  once per window from `isel(ensemble=0, time=0)`, because `blanking` is static
  geometry that the run stage writes replicated over `(ensemble, time)` — asking
  `stream_window_members` for it instead cost a second full ensemble-sized read
  (+33% on the window-state bytes at Barcelona scale) for one frame's worth of
  information. Omit it and the mask falls back to the member, for an in-memory
  caller. A member whose time-mean slab is non-finite anywhere is excluded from
  every reduction, with a warning and an `n_members_scored` count; other failures
  set `reason` and null the layer rather than taking the sensor pass down.
- `truth_mean_field_stats(..., stride=1, bootstrap_blocks=20)` — the WP1.3 truth
  half, on the truth's own grid and cadence: time-mean fields, Reynolds stresses,
  station columns, and the moving-block sampling floors (hit-rate allowance, TKE
  and `⟨u′w′⟩` RMSE floors). Read chunk-wise; nothing is interpolated here,
  because averaging precedes interpolation. `stride` is not cosmetic — the floors
  are bootstrapped over the truth's fluid cells **at the scored stride**, so that
  they describe the same sample the scores do.
- `STATION_QUANTILES` — the across-member percentiles (5/25/50/75/95) persisted
  per station column in `eval_fields.nc`, so WP1.4's profile band is empirical
  rather than `mean ± kσ` and needs no recomputation.
- `mean_field_summary(...)` / `mean_field_scores(...)` — the reduction and the
  WP1.3 entry point. `mean_field_scores` returns
  `(mean_field_metrics block, eval_fields Dataset)`, the Dataset being `None` on
  every degraded path.
- `evenly_spaced_levels(n_levels, n_slices)` — `_vel_field_4z`'s z-level
  selection rule, shared rather than re-derived, so the mean-field slabs land on
  the same planes as the `state_metrics` `|U|` RMSE slices.
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
  and the distances from `ensemble_scores`' Wasserstein family — including
  `wasserstein_over_floor_reference`, the perfect-model reference `w1_over_floor`
  has to be read against, so like the WP1.1 bundle this layer imports its
  calibration references rather than asserting them in prose.

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
- `mean_field_metrics` — the time-mean velocity field and the resolved Reynolds
  stresses scored against the truth with the urban-CFD standards (hit rate,
  FAC2, fractional bias, NMSE with its systematic/unsystematic split), plus TKE
  and `⟨u′w′⟩` RMSEs beside the truth's own sampling floors. `standard` and
  above only, and likewise after the `skip_viz` return.

It also writes one NetCDF at `standard` and above: **`eval_fields.nc`**, the
reduced mean-field / station-column arrays the stage-3 figures need. It is the
sanctioned handoff — stage 3 otherwise re-derives its inputs from the raw
artifacts, which here would mean repeating the streaming pass over every window
state file. Its variables and the mask it carries are documented in
[`data_assimilation.md`](data_assimilation.md#run-directory-artifacts).

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
| `<set>.{n_members,n_windows,n_sensors}` | Shape of what was *configured*. `n_windows` is the requested window count read from `truth_access.yaml`, **not** the number that contributed — for that, read `<stat>.n_windows_scored`, which is per statistic because windows drop per statistic. |
| `<stat>` = `window_mean` | Per-window time-mean of each velocity component, pooled over component × sensor × window. The one statistic a biased-inflow run moves; also the one a *long* window can match while every higher moment is wrong. |
| `<stat>` = `window_variance` | Per-window `ddof=1` time-variance of each component, pooled the same way. `null` for any window holding a single frame — one frame has no `ddof=1` variance, and at the smoke shape that is the whole run. |
| `<stat>` = `tke` | `0.5·Σ_c var(u_c)` per sensor per window. Computed as the time-mean of the *instantaneous* `0.5·n/(n−1)·Σ_c(u_c−⟨u_c⟩)²`, which makes it exactly the tabulated `ddof=1` TKE — the Bessel factor is on the integrand so the bootstrap resamples the reported estimator, not a `ddof=0` cousin. |
| `<stat>` = `velmag_mean` | Per-window time-mean of `\|U\|`, per sensor. Not implied by `window_mean`: `⟨\|U\|⟩ ≥ \|⟨U⟩\|`, and the gap is exactly the fluctuation a directionally-wandering flow carries. |
| `<stat>.crps` | Fair (finite-`M` debiased) CRPS of the ensemble's statistic against the truth's, averaged over pooled elements. **In the statistic's own units** — variance and TKE are in m²/s², so the four are not comparable to each other. |
| `<stat>.crpss_vs_prior` | **One-window-ahead** skill against the prior ensemble's same statistic. `windows/window_{w}_prior_state.nc` is window `w`'s ESMDA step-0 forecast and `run_esmda.py` seeds window `w`'s prior from window `w−1`'s posterior, so pooling all windows scores each posterior against **the previous window's posterior propagated forward**, not against the run's initial prior. It is therefore *not* the same quantity as `parameter_metrics.<p>.crps_reduction_vs_prior` in the same file, and the distinction is the one WP1.1 makes explicit in `contraction_ratio.vs_window_prior` / `vs_initial_prior` (there is no `vs_initial_prior` here — the initial prior was never rolled out at the sensors). **`null` on nearly every run**: it needs `windows/window_*_prior_state.nc` for *every* window, and `conf/run_esmda.yaml` ships `run.save_prior_state: false`. `null` here means "not saved", never "no skill". Scored on the **windows both ensembles have in common** — if the prior and the posterior drop different windows the pool is rebuilt on the intersection (and is `null` only when that is empty), so the number is never a comparison of two different window subsets. Consequence: this key is **not recoverable as `1 − crps/prior_crps` from the reported keys**, and the prior's CRPS cannot be backed out of the summary. `<stat>.crps` keeps the posterior's **full** window list and its own (truth-and-posterior-members) finiteness filter, while `crpss_vs_prior` is computed on the **intersection** under a **joint** filter (truth *and* every posterior member *and* every prior member finite), so the two CRPS values inside the skill score are means over a different element set than the reported `crps`. |
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
| `<stat>.n_windows_scored` | How many of the `<set>.n_windows` configured windows contributed a finite element to **this** statistic. Per statistic rather than per set because windows drop per statistic: a single-frame window has no `ddof=1` variance, so it kills `window_variance` and `tke` while `window_mean` survives it. `< n_windows` is not an error and does not set `reason` — but it is the difference between `n_samples` being small because the sensor set is small and being small because two thirds of the run silently fell out, so check it before comparing `n_samples` across runs. `0`, not `null`, on the null path (nothing contributed is a measurement). |
| `<stat>.reason` | Non-null exactly when the statistic's numbers are null, naming which degradation fired. |
| `wasserstein.w1_over_sigma_pooled` | `{median, max}` over **(sensor, window)** of `W1(that window's members' \|U\| samples pooled, that window's truth \|U\|)/σ_truth`. Every Wasserstein number below is computed per assimilation window and then reduced over the `n_sensors × n_windows` elements — **not** pooled over the whole run (see the stationarity note below for why). This one is the ensemble's *predictive* distribution. |
| `wasserstein.w1_over_sigma_member_mean` | The same per-(sensor, window) distance with each member scored alone, then averaged over members before the reduction. Read it **with** the pooled number, never instead of it: `W1` is convex in its first argument, so pooled ≤ member-mean always. They coincide under a shared bias (pooling cannot cancel it) and separate when the spread brackets the truth. |
| `wasserstein.self_floor` | `W1(first half, second half of that window's truth)/σ_truth` — the truth's *own* sampling variability at this sample count and autocorrelation, i.e. the scale the raw distance has to be read against. Halves are contiguous, not random, so each keeps its serial correlation; the raw distance is unreadable without this. Computed on the same (sensor, window) elements as the distance above, so floor and distance are like-for-like on sample count — a per-window floor against a whole-run distance would compare an `n`-frame floor to a `W·n`-frame distance. It is **not** a perfect model's expected distance, and dividing by it does not produce a metric calibrated at 1: the floor compares `n/2` truth samples against `n/2`, while the numerator compares `M·n` pooled member samples against `n` truth samples. See the calibration note below. |
| `wasserstein.w1_over_floor` | The `{median, max}` of the **per-(sensor, window) ratios** — each element's own distance divided by its own floor, then reduced. Deliberately **not** `w1_over_sigma_pooled.median / self_floor.median`: a ratio of medians pairs one element's distance with another element's floor, and the two medians need not even be attained at the same sensor. **Not calibrated at 1**: read it against `w1_over_floor_calibrated_median`, never against 1 (calibration note below). This is the headline ratio, not `w1_over_sigma_pooled`. |
| `wasserstein.w1_over_floor_calibrated_median` | `{median, max}` over the same (sensor, window) elements of **the `w1_over_floor` a perfect model of a series like this one — same length, same `M`, same autocorrelation — typically scores**; the reference the row above has to be read against. Per element, `ensemble_scores.wasserstein_over_floor_reference` runs a **two-sample** block bootstrap: each replicate moving-block-resamples the element's truth into `n_members + 1` synthetic series (so they inherit its marginal and, up to the block length, its autocorrelation), pools `n_members` of them as a synthetic perfect model, and scores that against the remaining one as a **stand-in truth window**, dividing by *that stand-in's* own floor; the answer is the median over replicates. Scoring the pooled model against the real truth array instead is wrong by 4–6× and worsens with `M` — `M·n` draws from the truth's own empirical distribution converge to that distribution, so the distance goes to 0 and the term that actually dominates (the truth window's own deviation from its law) is missed. The price of resampling both sides is conditioning: the median runs over the sampling variability of the truth *window* as well as the model's, which is why this reads "a series like this one" and not "this exact window" — the dominant term *is* that window's own sampling error, and no estimator conditioned on the one window can see it. Same nominal-vs-calibrated device as `coverage.nominal_alpha_*` and `zscore.max_abs_calibrated_median`, and for the same reason: the perfect-model value moves with the series' autocorrelation and with `M`, so a prose constant cannot travel with a plotted number. Read `w1_over_floor.median ≈ this` as "no worse than the truth's own sampling noise", and a multiple of it as a genuine discrepancy. |
| `wasserstein.reason` | Non-null when any of the five reduced to `null`, naming which. Individual (sensor, window) elements drop out of the reduction rather than nulling the block: a constant truth series has no `σ` and no floor (a property of the sensor, not a bug), and a window with fewer than four finite frames cannot be split in half for a floor at all. Each entry is `null` only when no element survives *for that entry*, so the five can degrade separately — at the **smoke shape** (3 frames per window) `self_floor`, `w1_over_floor` and `w1_over_floor_calibrated_median` are `null` with a reason while the two distances are still numbers, because a floor needs four samples per window and a distance needs two. |

**`w1_over_floor` is not calibrated at 1 — read it against
`w1_over_floor_calibrated_median`.** Dividing by the floor removes the *scale* of
the sampling noise but not its *sample-count* dependence: the floor is `n/2`
truth samples against `n/2`, the numerator is `M·n` pooled member samples against
`n` truth samples. Measured on `ensemble_scores` directly, no windowing involved
(AR(1) φ = 0.6 stationary series, M = 32, median of 200 trials):

| `n` scored per element | perfect model | +0.5σ mean bias | σ × 0.5 |
|---|---|---|---|
| 18 | 0.77 | 0.93 | 0.75 |
| 36 | 0.56 | 0.99 | 0.80 |
| 108 | 0.53 | 1.48 | 1.25 |
| 216 | 0.50 | 2.02 | 1.72 |
| 432 | 0.54 | 3.00 | 2.47 |

A perfect model sits at **~0.55 and is flat in `n`** — numerator and floor both
shrink as `1/√n`, so the ratio does not move. A genuine error has an
`n`-independent numerator over a `1/√n` floor, so its score **grows as `√n`**.
Two consequences for reading a summary: a model reading exactly `1.0` is already
about **2× worse than perfect**, not "indistinguishable from perfect"; and at the
shipped 36-frame window a +0.5σ bias reads 0.99 against a perfect 0.56, so the
raw ratio alone cannot separate them at all — the separation is only visible
against a reference computed at the same sample count. That reference is
`w1_over_floor_calibrated_median`. The ~0.55 is **not** a constant to memorize:
it was measured at one φ and one `M` and moves with both, which is why the number
ships beside the score rather than in this paragraph.

**How much to trust the reference, and which way it errs.** It is a bootstrap, so
it reuses the truth's own values and has to be validated rather than asserted.
Ratio of `wasserstein_over_floor_reference` to a *directly simulated* independent
perfect model, paired per truth window (AR(1), `M = 32`, 120 windows, 5 model
realizations each):

| `n` | φ = 0 | φ = 0.6 | φ = 0.9 |
|---|---|---|---|
| 36 | 1.02 | 1.05 | 0.55 |
| 108 | 0.99 | 0.91 | 0.88 |
| 432 | 0.98 | 1.06 | 1.08 |

Within 12% at eight of the nine points. The exception has a clean mechanism — at
φ = 0.9 and `n = 36` the resampling block is 2 frames, which cannot carry a
correlation spanning ten, so the synthetic sides under-inherit the slow
excursions that make a short strongly-correlated window a poor sample of its own
law. **Note the direction: this one is not conservative.** An understated
reference makes a good model look bad, unlike the floor's own biases. So on a
strongly autocorrelated probe over a short window, read a score moderately above
its reference as *inconclusive* rather than as a failure; longer windows and
weakly correlated series do not have the problem.

**Reading the Wasserstein rows: what the floor does and does not absorb.**
`self_floor` is calibrated on *stationary* series — its AR(1) validation table in
`ensemble_scores.wasserstein_self_floor` is correct as far as it goes. A
**deterministic trend inside the split window** is a separate failure mode: the
two contiguous halves are then drawn from different parts of the trend, so the
split measures the trend rather than the series' own sampling variability and the
floor inflates.

The table below is the **pre-change, whole-run baseline** — one distance and one
floor over all windows pooled, which is what this layer computed *before* round 1
made every Wasserstein key per `(sensor, window)`. It is kept because it is the
measurement that motivated windowing, **not** because it describes current
output: no number in it lines up with a key in a `run_summary.yaml` written by
the current code. Measured by the reviewer on a 108-frame `|U|` at σ_turb = 0.5,
with "perfect" an independent realization of the same law and "bad" a +20% `|U|`
with half the turbulence:

| truth (pre-change, whole-run) | `self_floor` | perfect `w1_over_floor` | bad `w1_over_floor` |
|---|---|---|---|
| stationary | 0.168 | 0.75 | 18.2 |
| magnitude cosine only | 0.366 | 0.38 | 5.8 |
| both cosines (shipped default) | 0.575 | 0.12 | 1.9 |

The trend is what matters here, not the absolute values. On the shipped default
truth (`params@truth_params: dynamic_cosine` — a 400 s inflow-angle cosine and a
200 s magnitude cosine) the trend inflated the floor ~3.4× and pulled a
clearly-wrong model from 18.2 down to `w1_over_floor ≈ 1.9`, a number that reads
as "essentially at the floor" to anyone taking 1 as the calibrated value (it is
not — see the calibration note above; the perfect model of that same truth read
0.12, so the *pair* still separated, but only if you had both). Computing per
assimilation window (as WP1.2 now does) removes the **cross-window** part of that
inflation, which is the dominant term. It does not remove all of it: a 180 s
window against a 200 s magnitude cosine is still not stationary within the
window, so on the shipped default the floor stays somewhat inflated and
`w1_over_floor` correspondingly deflated.

**What windowing costs in sensitivity (mechanism structural, magnitude
indicative).** Because a real error's score grows as `√n` while a perfect model's
is flat, scoring per window instead of whole-run divides every real error's score
by roughly `√W` — 108 frames → 36 at the shipped cadence, a factor ~1.7. The
reviewer measured, end to end: a +20% mean bias reading **2.02 whole-run → 0.85
per-window**, a perfect model **0.44 → 0.27**, i.e. bad/perfect separation
**4.6× → 3.1×**. Those three pairs come from a **synthetic stand-in for a probe
series, not from a real run**, so take the magnitude as indicative; the `√n`
mechanism behind it is structural and applies to any run. Per-window is retained
anyway — it is what keeps the floor and the distance like-for-like on sample
count, and it removes the cross-window trend term — and with
`w1_over_floor_calibrated_median` shipping beside the ratio the residual cost is
*visible* rather than silent: score and reference are computed on the same
elements at the same sample count, so the per-window-vs-whole-run choice no
longer changes how the number is read, only how much room there is between a good
model and a bad one.

**What decides it is the forcing's period against the split length, not the
split length itself.** The floor inflates when the two contiguous halves have
different means, i.e. when the split interval contains a *net excursion* of the
trend — which is worst near `period ≈ 2 × interval` and vanishes once the
interval holds several full cycles. Measured on the same synthetic probe: a
cosine of period ≈ 2× the interval inflates the floor ~3× at *both* 108 frames
(0.223 → 0.724) and 36 frames (0.369 → 1.426), while a cosine cycling several
times inside the interval does not inflate it at all (0.294 against a stationary
0.369). So "use a shorter window" is not a universal remedy: shortening helps
only while it moves the interval away from half the forcing period, and a window
that lands *on* that ratio is the worst case. On the shipped default the 180 s
window sits at 0.9× the 200 s magnitude cosine and 0.45× the 400 s inflow-angle
cosine, so some inflation survives windowing. WP1.4 plots these numbers: on a
truth with a deterministic trend of period comparable to one window, read a low
`w1_over_floor` as "the floor ate it", not as "the ensemble is perfect".
`w1_over_floor_calibrated_median` does **not** rescue that case, and it is not
meant to: it calibrates the *sample count*, not the stationarity. Its stand-in
window is moving-block resampled (a 2-frame block at the shipped shape), so it
cannot inherit a trend spanning the window — the reference is therefore computed
on an effectively de-trended series while the score itself divides by the
inflated floor. Reasoning from that construction rather than from a measurement:
expect a trending truth to push the *score* below its reference, and read a score
well below its reference as "the floor ate it" rather than as "better than
perfect". Either way, the pair is not a trend test in either direction.

##### `standard`: the mean-field / Reynolds-stress layer (WP1.3)

At `level: standard` a top-level `mean_field_metrics` mapping appears, and
`eval_fields.nc` is written beside it. Like `sensor_statistics` it attaches
*after* the `run.skip_viz` early return (it consumes the truth), so a
`skip_viz` run has neither.

What it scores: each ensemble member's **time-mean** velocity field over the
whole assimilation horizon, reduced to the ensemble mean of those means, against
the truth's time mean on the same grid — with the urban-CFD standard metrics of
[`docs/plans/esmda_turbulence_evaluation.md`](plans/esmda_turbulence_evaluation.md)
§4.1. The ensemble half comes off the **single shared per-member read pass**
that also feeds the sensor extraction (`stream_window_members`), so its
incremental cost is arithmetic, not IO; the truth half is a separate pass over
the truth's own grid and cadence, and averaging always precedes interpolation.

Two rules for reading any of it, both inherited from the layers above.
**A calibration number ships with its reference** — here the references are the
truth's own sampling floors, and an RMSE below its floor is not skill, it is the
truth's window length. And **every key is `null`, not absent, on a degraded
path**: an old run dir, an absent truth, a solver whose staggering cannot be
colocated, or a level at which the layer never ran all produce the same key set.
`reason` is non-null on every one of those — and, unlike the other blocks here,
also on a block that still carries perfectly good numbers, because this layer
has degradations that are partial (a diverged member excluded, an extrapolated
level that could not be dropped) and those should be legible rather than
invisible.

| Key | Meaning |
|---|---|
| `hit_rate.{u,v,w}` | VDI 3783/9 hit rate `q` per signed velocity component: the fraction of scored cells agreeing within `relative_tolerance` **or** within `allowance`. The `or` is the design — a velocity component passes through zero, where a relative test is meaningless. Metrics-doc acceptance is `q >= 0.66`. Like every aggregate here it is reduced over the **non-extrapolated** z-levels only (see `averaging.z_levels_extrapolated`). |
| `hit_rate.per_z` | The same `q` per scored plane, as `[{z, u, v, w, fac2_velmag, extrapolated}, ...]` — **every** plane, including the extrapolated one the aggregates leave out. Read it before the pooled numbers: a run can pass in the free stream and fail in the canopy, and the pooled `q` averages that away. |
| `hit_rate.per_z[*].extrapolated` | `true` on a plane whose colocation was extrapolated rather than interpolated — the top level of a staggered backend's `w`, which an evenly spaced selection always lands on. Its second moments are inflated (5× against an interior cell for face-to-face white noise, ~1.2× for a well-resolved field), so it is reported here but kept out of every aggregate — unless excluding it would leave nothing (`nz == 1`), where it is kept and `reason` says so. |
| `hit_rate.allowance.{u,v,w}` | `W`, the absolute error allowed regardless of relative error, **in m/s** — the truth's own sampling uncertainty, a moving-block bootstrap of the per-cell time mean reduced over cells by the *median* (forced: `hit_rate` takes one scalar, and the median is robust across a wake). This is what turns `q` into "indistinguishable within the truth's sampling error" rather than an arbitrary threshold. Bootstrapped over the truth's **fluid** cells at the scored **stride** (see `averaging.n_bootstrap_cells`) — over the raw slab it would be a floor for a sample the scores never touch. |
| `hit_rate.allowance.reason` | Non-null when *no* component has a finite allowance, naming the degradation. `null` allowances are the sanctioned path, not a fault: the absolute clause is then skipped and `q` degrades to the pure relative test, which can only be **lower** — a missing floor never flatters a run. It fires at the **smoke shape** (3 frames against `bootstrap_blocks: 20` leaves a block length below 2) and on all three of the `.temp` run dirs. |
| `hit_rate.relative_tolerance` | `D`, `0.25` by default. Configuration, so it keeps its value on the null path — a `q` is unreadable without the two tolerances it was computed with. |
| `fac2_velmag` | Fraction of scored cells with `0.5 <= pred/obs <= 2` on `\|U\|`. Acceptance `>= 0.5` for dispersion, `>= 0.3` in practice for urban-LES velocity. `\|U\|` only, never a signed component: `fac2` **raises** on a negative value rather than returning a meaningless ratio test. |
| `fb.{u,v,w,velmag}` | `FB = (ō − p̄)/(0.5(ō + p̄))`; acceptance `\|FB\| <= 0.3`. **Positive means the prediction is too small.** Bounded in `[−2, 2]` for `velmag`; **not bounded at all** for a signed component, whose denominator can pass through zero while both fields are ordinary — a large per-component `\|FB\|` is then a statement about the denominator, not about the run. `fb.velmag` is the one to quote. |
| `nmse.{u,v,w,velmag}.total` | `⟨(o − p)²⟩/(ō·p̄)`; acceptance `NMSE <= 4`. `null` whenever `ō·p̄ <= 0`, which is **routine** for a signed component and unreachable for `velmag`. |
| `nmse.<q>.{systematic,unsystematic}` | `NMSE_s = 4FB²/(4 − FB²)` and the remainder. Exact algebra, not an approximation, so `NMSE_s <= NMSE` is an identity. **This split is the conceptual payoff of the layer**: assimilation should collapse `NMSE_s` (a bias is a parameter error) while `NMSE_u` is chaotic decorrelation and is irreducible — a run that lowers only the total may have done nothing to the estimable part. Both `null` at `\|FB\| >= 2`, where the formula divides by zero. |
| `tke_rmse.value` / `uw_stress_rmse.value` | RMSE of the ensemble-mean **resolved** TKE and `⟨u′w′⟩` maps against the truth's, over the scored cells. Resolved only — no subgrid contribution, which is also stated in `eval_fields.nc`'s `stress_kind` attribute. |
| `tke_rmse.sampling_floor` / `uw_stress_rmse.sampling_floor` | The reference the line above is meaningless without: the truth's own sampling error at this window length, a moving-block bootstrap of the per-timestep integrands (Bessel factor on the integrand, so the resampled statistic is exactly the reported `ddof=1` moment and matches WP1.2's TKE definition), reduced over cells by **RMS** — the RMS of the per-cell sampling errors is the RMSE a perfect model would still score. `value <= sampling_floor` means "at the noise floor", not "skillful". Computed on the same fluid, strided, capped cell sample as the allowance above; over the unmasked slab these come out ~13% low, which is the anti-conservative direction for a number whose whole job is to say "this RMSE is not skill". `null` wherever the bootstrap is undefined, including the smoke shape. |
| `averaging.n_time` / `n_time_truth` | Frames averaged per member, and frames averaged on the truth side. They routinely differ — the truth is saved on its own cadence — and both are time means, so that is not a misalignment. `n_time` is the **smallest** member's count when members carry different frame counts (a `logger.info` says so); the moments weight every frame equally, so a longer member is simply better sampled. |
| `averaging.{n_windows,n_members,n_stations,stride}` | The accumulation's shape, as **configured**. `stride` is `run.metrics.mean_field_stride`: hit rate, FAC2, FB and NMSE are cell-count-weighted, so **scores are only comparable across runs at equal stride**. For what actually contributed, read the two keys below. |
| `averaging.n_members_scored` | Members that entered the mean field. It is `< n_members` when a member's time-mean slab came out non-finite anywhere — a diverged CFD member — in which case that member is dropped from **every** reduction here, `reason` names the excluded indices and a `logger.warning` fires. The exclusion is whole-member on purpose: an ensemble mean and a `ddof=1` spread taken over different member sets at neighbouring cells are not a field, and the spread is what `eval_fields.nc` publishes. `eval_fields.nc` carries the same number as an attr, so a figure captioning a band with `M` reads the right one. |
| `averaging.n_stations_with_truth` | Stations that have a truth column at all. Routinely `< n_stations`: a station whose 2×2 horizontal stencil touches a solid cell loses its **entire** truth column (the all-fluid stencil rule is strict on purpose — the alternative is profiling against a building interior), and stations default to sensor x/y, which in an urban case sit on facades and roofs. A `logger.warning` names the affected stations and their coordinates. **Check this before reading an S1 profile panel**: without it, a panel with a posterior band and no truth line looks like a plotting bug. |
| `averaging.{z_levels,z_indices}` | The scored planes on the assimilation grid, and their indices. The selection is `evenly_spaced_levels`, the same rule `state_metrics`' `\|U\|` RMSE slices use, so the two layers describe the same planes. |
| `averaging.z_levels_extrapolated` | The subset of those planes whose colocation was **extrapolated**, not interpolated. Each solver stores one face per cell, so the last centre of a staggered axis is filled from outside the data with weights (1.5, −0.5) — and `np.linspace(0, nz−1, k)` always includes `nz−1`, so on uDALES/PALM `w` this is not bad luck, it is guaranteed (1 of 4 levels at the shipped `n_z_slices`). It is a fact about the **grid**, reported whatever the aggregate then does with it: normally these levels are excluded from every aggregate and still reported in `per_z`, but at `nz == 1` the only scored level *is* the extrapolated one, and there every level is kept with a `reason` and a warning rather than nulling every score. Empty on pylbm — nothing is staggered. Only the z axis is tracked: the last x-row and y-column carry the same edge and stay in every aggregate, a known limitation left to WP1.4 or later. |
| `averaging.{truth_z_levels,z_offset_max}` | The truth levels actually used, and the largest residual `\|z_truth − z_assim\|`. Vertical alignment for the slabs is **nearest level, not interpolated** — a post-averaging z-interpolation would require carrying the bracketing levels through the streaming pass, i.e. the full 3-D truth moments. `z_offset_max` is how far that compromise moved things (0.0 on all three `.temp` run dirs); a non-trivial value means the slab scores are comparing slightly different heights. The **station columns** have no such constraint and *are* z-interpolated after averaging. |
| `averaging.{n_cells,n_scored_cells}` | Cells in the slab region, and cells that carried a **scoreable pair** — finite on both sides, which is the mask *and* everything else that came out non-finite. Their ratio is `masking.scored_fraction`, and `n_scored_cells` is derived from the retained pairs themselves, so the reported sample is the scored sample by construction rather than by coincidence. |
| `averaging.n_aggregate_cells` | The subset of `n_scored_cells` on the levels the **aggregate** scores reduce over, i.e. excluding any extrapolated plane. It differs from `n_scored_cells` exactly when `z_levels_extrapolated` is non-empty *and* the exclusion was applied; the per-z rows still cover all of them. |
| `averaging.n_stress_cells` | The sample **the two RMSEs** were computed on — a subset of `n_aggregate_cells`, and the only key that describes `tke_rmse` / `uw_stress_rmse` rather than the four urban-CFD scores. It is smaller because those are `ddof=1` moments: a cell needs **two** frames where a mean needs one, so a member that lost all but one frame over a sub-region, or a one-frame window, drops that cell from the stresses while its mean field stays perfectly finite (measured 76 against 104 on a partial-divergence fixture). The base sample is deliberately *not* widened to include this rule — that would let a short window empty the hit rate's sample too — so the count travels instead. `tke` and `⟨u′w′⟩` share one sample by construction (same `StreamingMoments` per-cell count, same mask), so one leaf covers both exactly; it is computed as an intersection regardless, and can therefore only understate. |
| `averaging.n_bootstrap_cells` | Cells actually used for the sampling floors. The floors need the truth *series* per cell (a moving-block bootstrap resamples a series, not a moment), which is the one array here that grows with the frame count — so the series is taken over the truth's **fluid** cells at the **scored stride**, then subsampled by a seeded random draw. Fluid and strided because the floors have to describe the same sample the scores do; **random** rather than a fixed stride because a raveled stride phase-locks (at 4×512×512 the implied stride is 68, `gcd(68, 512) = 4`, so only `x ≡ 0 (mod 4)` is retained — on a regular building array that is one geometric phase). At `stride > 1` this can exceed `n_scored_cells`: the truth-side stride lands on the truth's own cells while the erosion happens on the assimilation grid, and the floors are a property of the truth's sampling *at the same density*, not a cell-for-cell alignment. |
| `averaging.n_bootstrap_cells_max` | The cap (`_TRUTH_BOOTSTRAP_MAX_CELLS = 4096`). A time bound, not a memory bound — see the runtime paragraph below for what it costs. |
| `averaging.bootstrap_seed` | The seed of that draw. Fixed rather than configurable, so the floors are reproducible: a floor that moved between two re-runs of the metrics stage would be indistinguishable from a floor that moved because the run changed. |
| `masking.source` | `blanking` if either side supplied a solid-cell mask, else `"none"`. |
| `masking.{truth_source,ensemble_source}` | Which side supplied it, separately — a cross-model run routinely has one and not the other. |
| `masking.note` | Non-null **exactly when nothing masked**, and it is the caveat that matters most in this block. See the paragraph below. |
| `masking.fluid_fraction` | **The mask's own number**: fluid on the assimilation side **and** resolvable on the interpolated truth side. The second half is not only the truth's buildings — a truth grid that does not cover the whole assimilation domain leaves NaN at the edges too. This describes the *mask*, not the scores. |
| `masking.scored_fraction` | **The scores' number**: the fraction that carried a scoreable pair, i.e. the sample every number in this block was computed on. `= n_scored_cells / n_cells` by construction, and always `<= fluid_fraction`. The two names exist because giving one name to two samples is the failure mode this layer keeps having to defend against; today they are equal on every reachable path (the only non-finiteness sources are the mask and diverged members, and the latter are excluded member-wise), so **a gap between them is a signal**, not noise. |
| `masking.ensemble_fluid_fraction` | The assimilation side's own fluid fraction, before the truth's resolvability is intersected in. The gap between this and `fluid_fraction` is the cross-grid cost quantified below. |
| `masking.truth_finite_fraction` | Fraction of target cells where the interpolated truth is finite. |
| `eval_fields` | File name of the companion NetCDF (`eval_fields.nc`), or `null` when the layer degraded and wrote none. |
| `reason` | Names whatever degraded, and — unlike the other blocks in this file — **it can be non-null on a block that still carries numbers**. Partial degradation is legible on purpose: excluded members and a kept-but-extrapolated aggregate level both produce real scores plus a `reason` describing what they were computed despite. A null block always has one; a block with numbers may. |

**The masking caveat, because it is the one number-moving limitation in this
layer.** The plan assumed solid cells arrive as NaN. They do not, on any
backend: the measured NaN fraction of `u`/`v`/`w` is 0.0 in every state artifact
inspected. Only **pylbm** writes a mask (`blanking`, 1 = solid); **pypalm**
`fillna(0.0)`s PALM's NaN before returning state, so a PALM mask is not
recoverable from the artifact at all; **uDALES** fielddumps carry small non-zero
junk inside buildings and ship no mask (the training-data scripts attach one
from `solid_c.txt` afterwards, which needs case-dir paths this stage does not
have). So on a uDALES or PALM run — `masking.source: none`, and a
`logger.warning` fires — **the scores include building interiors and are
optimistic by roughly the building fraction**. FAC2 is the worst affected: a
cell that is zero on both sides counts as a hit, which is right for a stagnation
point and wrong for a solid cell. Read a `masking.source: none` run's FAC2 and
hit rate as upper bounds, and compare them only against other unmasked runs.
Parsing `solid_c.txt` and preserving PALM's NaN are both out of phase 1.

**The cross-grid cost: small in count, biased in direction.** When truth and
assimilation grids differ, the truth is NaN-masked *before* `.interp`, so any
target cell whose interpolation stencil touched a building drops out. That is
the conservative threshold and it comes for free — but it erodes the perimeter
of every building. Measured on `pylbm_to_pylbm` (truth half-cell-shifted by
`x_offset: -0.5`), 164 of 1200 ensemble-fluid cells are lost, of which **120 are
genuinely solid in the truth's own shifted geometry** and correctly excluded;
only **44 — 3.7% of the fluid set — are erosion**. The count is therefore small.
The *bias* is not: those 44 cells carry **2.65× the mean shear** of the retained
set and the run scores materially worse on them (`q(u)` 0.682 against 0.804,
NMSE(`|U|`) 0.0680 against 0.0128), so dropping them **flatters `q` by
+0.005…+0.010 and understates NMSE(`|U|`) by 6%** at this shape. The effect
scales with perimeter/area, so it is *larger* on a dense array like Barcelona
than on this fixture. The alternative — interpolating field and mask separately
— keeps those cells but scores them partly against building interiors, which is
a wrong number instead of a missing one; the trade stands, but read a cross-grid
run's `q` and NMSE as slightly optimistic and check the gap between
`masking.ensemble_fluid_fraction` and `masking.fluid_fraction` to see how much
perimeter this case has.

**Runtime, and why the smoke number was meaningless.** This layer's dominant
cost is the truth's bootstrap, and `block_bootstrap_std_batch` returns all-`nan`
before doing any work below 4 frames — the smoke truth has 3. So the smoke-scale
benchmark (2.06–2.28× against `basic`, ~90 ms absolute) measured this layer at
the one shape where its expensive half does not run, which is also why dropping
`n_z_slices` to 2 appeared to change nothing. Measured `_truth_sampling_floors`
at 36 frames / 20 blocks: 1600 cells 157.5 ms, 4096 cells 448 ms, 16384 cells
1.9 s, **65536 cells (4×128×128) 13.2 s**. `_TRUTH_SERIES_MAX_BYTES` bounds
memory, not time, and at that shape the stride it implies is 1.
The bootstrap row count is therefore capped at `_TRUTH_BOOTSTRAP_MAX_CELLS =
4096`. End to end at 4×128×128 × 36 frames, `M = 8`, 3 windows: **`basic` 0.35 s
against `standard` 1.62 s**, where uncapped the truth pass alone would add the
13.2 s. **The cap does not materially move the floors:** over 8 seeds at that
shape, capped against all-cell is rms 0.50% (allowance), 0.39% (tke), 0.64%
(uw), worst case 1.29% — against a number that is an order-of-magnitude
statement about the truth's window length. `averaging.n_bootstrap_cells` /
`n_bootstrap_cells_max` / `bootstrap_seed` make the realised sample
reproducible. `n_z_slices` and `mean_field_stride` remain the relief valves for
the *rest* of the stage, whose cost is per cell.

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

**Known remaining full-ensemble `.load()`, deliberately not fixed.** The ESMDA
evaluation effort's phase-1 acceptance criterion is "no `.load()` of window
state files anywhere (grep)", and as of WP1.3 that holds — but the grep is not
the same thing as "no full-ensemble state file is ever loaded". One hit remains:
`_filtering_common.ensemble_cycle_sensor_series` does `analyzed_states.load()`
on `state_history.nc`, which **is** a `(cycle, ensemble, …)` full-ensemble
state. It is out of scope because it is not a `windows/window_*_state.nc` and it
is much smaller — one analyzed frame per cycle rather than a whole rollout — but
it will grow with `filtering.num_cycles × ensemble size`, and the fix when it
matters is the same one the ESMDA stage took: drive it from a
member-at-a-time generator and accumulate. Recorded here so a reader running the
acceptance grep is not misled into thinking the tree is clean of full-ensemble
loads everywhere.

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

Both sensor series are read through the ESMDA stage's helpers
(`_esmda_common.ensemble_sensor_series` / `truth_sensor_series` / `open_truth`)
rather than through this file's own copies, which were deleted. The ensemble
path used to `xr.open_dataset(path).load()` a whole-ensemble window file, the
one thing phase 1 forbids materialising; the truth path was already
window-at-a-time but was character-for-character the ESMDA copy, so keeping it
only created somewhere for the two stages to disagree about what a sensor series
is. Both are bit-identical to what they replaced, **memory layout included** —
which is load-bearing rather than fussy: consumers reduce over these axes, numpy
walks a reduction in memory order and float addition is not associative, so a
re-laid-out buffer moves `metrics.yaml` in the last ULP (measured: forcing
C-contiguity moves 16 of 92 leaves, all sensor CRPS entries, by up to 2.1e-15).
Sweep comparisons are cross-run, and `metrics_version` exists so historical
numbers only move deliberately. `_split_quantities` — the `{u, v, w, vel}`
reshaping — is all that remains specific to this stage, and it deliberately
returns `.sel` **views** of the shared buffer for that reason; `vel` is
`_esmda_common.sensor_magnitude`, the same definition the ESMDA stage uses. It
is loudly three-component (`_COMPONENTS`, and a `ValueError` naming the actual
component set otherwise) rather than silently so: the `metrics.yaml` key names
and the three-term `|U|` sum are per-component anyway, so a "general" splitter
would have moved a silent drop one layer down instead of removing it.

A run whose sensor series cannot be read (a legacy window file with no
`ensemble` dimension; sensor points outside the domain) no longer disappears
from the sweep: `process_run` catches the `ValueError`, logs a warning naming
the run and the cause, records it in `status["note"]`, and still writes
`metrics.yaml` with the parameter/state metrics and the `num_sensors` skeleton
— the same degradation as the missing-`truth_access` branch above.

**Known stale invocations (pre-existing, not fixed here).**
`job_scripts/local/eval_sweep.sh:85`,
`job_scripts/snellius/eval_sweep.slurm:66` and
`job_scripts/delftblue/eval_sweep.slurm:69` all call
`scripts/compute_sweep_metrics.py`, which no longer exists — the script lives at
`scripts/figure_creation/compute_sweep_metrics.py`. The comment headers in those
files and `job_scripts/local/README.md` name the old path too. Unrelated to the
evaluation effort and deliberately left for its own change; a sweep launched
from any of the three will fail at stage 1 until it is corrected.

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
