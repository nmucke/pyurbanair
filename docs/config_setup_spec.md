# Config setup refactor — spec

## Goal

Flatten `conf/`. Before the refactor a run composed ~15 groups across `esmda/`, `filtering/`,
`observation/`, `execution/` and `common/`. After the refactor a run is one
short workflow file plus `assimilation.yaml`.

Status: done. The flattened tree is `configs/`, read by the rewritten
`scripts/` and tested by `tests/`. The old `conf/`, `scripts/` and `tests/`
are archived under `archive/` (not run, not tested). This page is kept as the
design record and the old -> new mapping; for the current tree see
[configs/README.md](../configs/README.md) and
[scripts_and_configs.md](scripts_and_configs.md).

Out of scope: `visualization/`, `compare_models.yaml`, `run_probe_series.yaml`,
named experiments. The neural-surrogate configs are covered in the last section.

## Layout

```
configs/
  forward.yaml            # entry point: forward runs
  assimilation.yaml       # entry point: all three DA scripts (ESMDA, filtering, hybrid)
  common.yaml             # paths (per machine), ensemble budget, Hydra run dir
  model/                  # pylbm, pyudales, pypalm, neural_surrogate
  case/                   # xie_and_castro, barcelona (incl. time.seconds_per_knot)
  params/                 # parameter samplers
  assimilation_settings/  # one file per component, every option inside;
                          # mounted at the top level (`smoother.*`, `analysis.*`, …)
    smoother.yaml         # static, dynamic, state, state_and_parameter, state_and_dynamic
    analysis.yaml         # stochastic, etkf, etkf_tsvd, letkf, letkf_tsvd
    localization.yaml     # none, correlation, distance
    state_reduction.yaml  # none, svd (smoothing), svd_current, svd_streaming (filtering)
    inflation.yaml        # none, multiplicative, rtps, rtpp
```

## Rules

1. **Each setting lives in one file.** A workflow file only overrides what
   differs for that workflow.
2. **Keys are namespaced:** `assimilation.*`, `observation.*`, `smoothing.*`,
   `filtering.*`, `hybrid.*`, `forward.*`.
3. **DA components are chosen by interpolation.** A slot points at an option:
   `localization: ${localization.correlation}`. Override from the CLI with
   `'filtering.localization=${localization.distance}'` (quoted). Each half of
   the hybrid has its own slots. `filtering` is the `EnsembleKalmanFilter`
   constructor block itself; `smoothing.smoother` is the chosen smoother, whose
   options read `smoothing.localization` / `smoothing.state_reduction`.
   Parameter evolution is inline (`filtering.parameter_evolution`, `null` = none).
4. **Only model, case and params are Hydra groups.**
5. **The observation operator and aggregation are inline** in
   `assimilation.yaml` (`observation.operator`, `observation.aggregation`;
   `null` switches aggregation off).
6. **YAML holds `_target_` and user knobs only.** Values the runner computes
   (`num_time_points`, `pin_initial_time_point`, `solver_name`, grid
   coordinates) are passed as kwargs at instantiate time.
7. **Invalid combinations are still rejected by the code** (e.g. letkf without
   localization, a dynamic prior in filtering, the SGS-discrepancy rules).

## Files

| File | Contents |
|---|---|
| `common.yaml` | `paths` (`machine`, `results_root`, per-machine `scratch`, `experiment_dir`); `ensemble` (size, workers, CPUs, `failure`); `hydra.run.dir` |
| `case/<case>.yaml` | `domain`, `geometry`, `obs` (sensor layout), `time` (`simulation_time`, `output_frequency`, `spinup_time`, `seconds_per_knot`) |
| `forward.yaml` | `paths.results_dir`, `forward.*` |
| `assimilation.yaml` | models, truth and prior samplers (static by default: valid for all three scripts), `paths.results_dir`, all `assimilation_settings/` files; `assimilation.*` (incl. all save flags and `assimilate_every_n_step`); `observation.*`; `smoothing`, `filtering` and `hybrid` blocks |

## Old → new

| Old | New |
|---|---|
| `run_forward_model.yaml` | `forward.yaml` | `paths.results_dir`, `forward.*` |
| `run_esmda.yaml`, `run_filtering.yaml`, `run_filter_smoothing.yaml` | `assimilation.yaml` |
| `common/runtime.yaml` | `common.yaml` (without `run.results_dir`, only used by `compare_models.py`, and `hydra.job.chdir`, already Hydra's default) |
| `execution/*` | one `ensemble:` block in `common.yaml` |
| `paths:` in each `run_*.yaml` | `paths` in `common.yaml` + `paths.results_dir` in the entry point; job scripts set `paths.machine` / `paths.results_root` through `PYURBANAIR_MACHINE` / `PYURBANAIR_RESULTS_ROOT` |
| `time.seconds_per_knot` | `time.seconds_per_knot` in the case file |
| `model@model=X` (forward) | `model=X` |
| `params_to_estimate` | `assimilation.params_to_estimate` |
| `*.num_assimilation_windows`, `*.seed` | `assimilation.num_windows`, `assimilation.seed` |
| `run.truth_dir`, `run.truth_start_time`, `run.ensemble_save_on_disk` | `assimilation.*` |
| `run.save_prior_state` | `smoothing.save_prior_state` |
| `run.save_history`, `run.save_forecast_history` | `filtering.*` |
| forward `run.ensemble`, `ground_truth_dir`, `rollout_steps`, `initial_state` | `forward.*` |
| `observation_error.*` | `observation.error.*`; `aggregation` → `propagation`; `representation_time_model` dropped (only `independent` exists) |
| `observation/operator=X` | inline `observation.operator` |
| `observation/aggregation=X`, `esmda.interval_seconds`, `esmda.aggregation_mode` | inline `observation.aggregation` (`interval_seconds`, `mode`) or `null` |
| `esmda.*` | `smoothing.*` |
| `esmda/smoother,localization,state_reduction=X` | `'smoothing.<slot>=${<component>.X}'` |
| `filtering/analysis,localization,state_reduction,inflation=X` | `'filtering.<slot>=${<component>.X}'` |
| `filtering/evolution=X` | inline `filtering.parameter_evolution` (`null` = none) |
| `filtering.filter` | the `filtering` block itself |
| `run.save_*`, `filtering.assimilate_every_n_step` | `assimilation.*` |
| `esmda.save_obs_diagnostics` | removed: always save (KB-scale) |
| forward `run.ensemble_save_on_disk` | removed (rejected by the forward workflow) |
| `filter_smoothing.beta` | `filtering.beta` |
| `filter_smoothing.likelihood_allocation` | `hybrid.likelihood_allocation` |
| `experiment/` | removed |

## Switch-over (done)

Instead of patching the old scripts, new ones were written against
`configs/`: `scripts/run_forward.py` (`forward`), `scripts/run_smoother.py`,
`scripts/run_filtering.py`, `scripts/run_hybrid.py` (all `assimilation`; each
appends its workflow name to `paths.results_dir`), with post-processing in
`scripts/compute_metrics.py`, `scripts/visualize_forward.py` and
`scripts/visualize_assimilation.py`, and config checks in
`scripts/utils/inconsistency_check.py`. Components are built with
`instantiate(cfg.smoothing.smoother, num_time_points=...)`,
`instantiate(cfg.filtering, observation_operator=..., forward_model=..., C_D=...)`
and `instantiate(cfg.observation.error)` (the `ObservationErrorSpec` field is
`propagation`). The old scripts (`run_forward_model.py`, `esmda/run_esmda.py`,
`filtering/run_filtering.py`, `filter_smoothing/run_filter_smoothing.py`,
`preview_config.py`, the sweep/metrics helpers) are in `archive/scripts/`,
the old job scripts in `archive/job_scripts/`.

## Acceptance

- Every workflow composes and fully resolves, with and without component
  overrides (`tests/scripts/test_configs.py`).
- `pixi run -e dev py.test` passes on the new scripts.

## Neural surrogates

| Old | New |
|---|---|
| `neural_surrogate/training_data.yaml` + `training_data/geometry_mode/*` | `surrogate/generate_data.yaml` (`data.geometry.mode: random\|fixed`); its trajectory times override `time.*`; its sampler is `params/surrogate_training_data.yaml` |
| `neural_surrogate/training.yaml` + `mode/*` + `dataset/transition.yaml` | `surrogate/train_stepper.yaml` |
| `neural_surrogate/pretrain_autoencoder.yaml` | `surrogate/train_autoencoder.yaml` |
| `neural_surrogate/train_latent_generator.yaml` | `surrogate/train_latent_generator.yaml` |
| `neural_surrogate/finetuning.yaml` + `finetune_mode/*` | `surrogate/train_dft.yaml` |
| — (new) | `surrogate/finetune_stepper.yaml`: fine-tune a pretrained stepper (`method: full\|lora`) |
| `neural_surrogate/testing*.yaml`, `comparison.yaml` | `surrogate/eval.yaml`, blocks `stepper` (`models`: one or several, so it also compares), `autoencoder`, `latent_generator` |
| `neural_surrogate/architectures/<family>/<size>.yaml` (22 files) | `surrogate/architectures.yaml`, entries `<family>_<size>` (domain-decomposed ones discontinued) |
| losses in `mode/*` | `loss` (MSE) in `surrogate/training.yaml`; the autoencoder's term weights are `loss_weights` |
| per-file `trainer`, `optimizer`, `batch_sampler` + `dataloader` | shared `trainer`, `optimizer`, `batch_sampler` (the only `batch_size`), `dataloader`, `dataset` in `surrogate/training.yaml`; each `surrogate/train_*.yaml` overrides what differs and sets its `collate_fn` |
| hard-coded weight/data paths | `paths.weights_dir`, `paths.training_data_dir` in `common.yaml` |

Scripts, in `scripts/surrogate/` (the old ones are archived in
`archive/scripts/neural_surrogate/`, reading `archive/conf/neural_surrogate/`):

| Old | New |
|---|---|
| `generate_training_data.py`, `generate_random_geometries_training_data.py` | `generate_data.py` |
| `train_neural_surrogate.py`, `pretrain_autoencoder.py`, `train_latent_generator.py`, `finetune_neural_surrogate.py` | `train.py` (`--config-name surrogate/<config>`); per-task setup in `tasks.py` |
| `test_neural_surrogate.py`, `compare_surrogate_models.py` | `evaluate_stepper.py` |
| `test_autoencoder.py` | `evaluate_autoencoder.py` |
| `test_latent_generator.py` | `evaluate_latent_generator.py` |
| — | `eval_common.py`: helpers shared by the evaluation scripts |
