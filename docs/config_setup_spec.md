# Config setup refactor — spec

## Goal

Flatten `conf/`. Today a run composes ~15 groups across `esmda/`, `filtering/`,
`observation/`, `execution/` and `common/`. After the refactor a run is one
short workflow file plus `assimilation.yaml`.

Status: the new tree exists in `configs_new/` next to `conf/`. It composes for
every workflow; the scripts are not wired to it yet. Concrete values are not
final and will be set once the structure is done.

Out of scope: `neural_surrogate/`, `training_data/`, `visualization/`,
`compare_models.yaml`, `run_probe_series.yaml`, named experiments.

## Layout

```
configs_new/
  forward.yaml            # entry point: forward runs
  assimilation.yaml       # entry point: all three DA scripts (ESMDA, filtering, hybrid)
  common.yaml             # run name, paths (per machine), ensemble budget, Hydra run dir
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
| `common.yaml` | `run.name`, `run.skip_viz`; `paths` (`machine`, `results_root`, per-machine `scratch`, `experiment_dir`, `base_results_dir`); `ensemble` (size, workers, CPUs, `failure`); `hydra.run.dir` |
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
| `paths:` in each `run_*.yaml` | `paths` in `common.yaml` + `paths.results_dir` in the entry point; job scripts pass `paths.machine=snellius` / `delftblue` |
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

## Script changes needed (migration)

1. Point each script's `@hydra.main` at the new directory: `forward` for
   `run_forward_model.py`, `assimilation` for the three DA scripts. Each DA
   script appends its workflow name to `paths.results_dir`; the hybrid reads
   its tempering from `filtering.beta`.
2. Read the new keys (table above). Build components with
   `instantiate(cfg.smoothing.smoother, num_time_points=...)` and
   `instantiate(cfg.filtering, observation_operator=..., forward_model=..., C_D=...)`.
3. Build the observation error with `instantiate(cfg.observation.error)` instead of
   `create_observation_error(cfg)`. Rename the `ObservationErrorSpec.aggregation`
   field to `propagation` (3 uses in `observation_error.py`, plus
   `hydra_helpers.py` and two tests). Keep the temporal-operator check from the
   helper as a one-line check in the runners.
4. Update `scripts/preview_config.py`, `run_record.py` (workflow names
   `esmda` → `smoothing`, `filter_smoothing` → `hybrid`), the
   SGS-discrepancy validation keys, and `tests/conftest.py`.
5. Code defaults for removed keys: always write the observation diagnostics;
   the forward workflow no longer reads `ensemble_save_on_disk`.
6. Add `upgrade_legacy_config(cfg)` wherever a saved `config.yaml` is reloaded
   (`_esmda_common.py`, `compute_sweep_metrics.py`) so old runs still
   post-process.
7. Update `job_scripts/` (about 78 files use `esmda.` / `esmda/`).
8. Replace `conf/` with `configs_new/`; update `conf/README.md`,
   `docs/scripts_and_configs.md` and the commands in `CLAUDE.md`.

## Acceptance

- Every workflow composes and fully resolves, with and without component
  overrides (checked for `configs_new/`).
- `pixi run -e dev py.test` passes once the scripts are switched over.
