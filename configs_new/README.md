# configs_new/

Preview of the flattened config tree described in
[docs/config_setup_spec.md](../docs/config_setup_spec.md). The scripts still
read `conf/`; nothing here is wired up yet.

```
forward.yaml        # entry point: forward runs
assimilation.yaml   # entry point: all three DA scripts (smoothing, filtering, hybrid)
common.yaml         # run name, paths (per machine), ensemble budget, Hydra run dir
model/  case/  params/
assimilation_settings/   # one file per component, all options inside
```

All three DA scripts load `assimilation.yaml`; each reads the blocks it needs
(`smoothing`, `filtering`, and for the hybrid also `hybrid`).

## Common overrides

```bash
# Hydra groups
case=barcelona  model=pylbm                          # forward
paths.machine=snellius                                # or delftblue / local
model@assim_model=pylbm  params@prior_params=static   # DA

# DA components: point a slot at an option (quote it so the shell keeps ${...})
'smoothing.smoother=${smoother.static}'  'smoothing.localization=${localization.distance}'
'filtering.analysis=${analysis.letkf}'   'filtering.localization=${localization.distance}'
filtering.parameter_evolution=null

# Tune an option where it is defined
localization.distance.localization_radius=20

# Observations
observation.aggregation=null                          # no binning
observation.aggregation.interval_seconds=30
```

## Keys

| Key | Holds |
|---|---|
| `assimilation.*` | `num_windows`, `seed`, `params_to_estimate`, `truth_dir`, `truth_start_time`, `ensemble_save_on_disk`, smoothing-only `save_prior_state`, filtering-only `assimilate_every_n_step`, `save_history`, `save_forecast_history` |
| `observation.*` | `operator`, `aggregation`, `error` (`instrument_std`, `representation_std`, `propagation`) |
| `smoothing.*` | `smoother` (the instantiable smoother), `localization`, `state_reduction`, `num_steps`, `alpha`, `final_time_smoothing` |
| `filtering` | the `EnsembleKalmanFilter` constructor block: `mode`, `analysis`, `localization`, `state_reduction`, `inflation`, `parameter_evolution`, `beta` |
| `hybrid.*` | `likelihood_allocation` (tempering is `filtering.beta`) |
| `forward.*` | `ensemble`, `ground_truth_dir`, `rollout_steps`, `initial_state` |
| `smoother.*`, `analysis.*`, `localization.*`, `state_reduction.*`, `inflation.*` | every option of each component (from `assimilation_settings/`) |
| `time.*` | from the case, including `seconds_per_knot` |
| `paths.*` | `results_root`, `experiment_dir`, `base_results_dir` from `common.yaml` (scratch chosen by `paths.machine`); `results_dir` from the workflow |
| `ensemble.*` | one budget for every workflow (`common.yaml`), incl. `failure` |
| `run.*` | `name`, `skip_viz` |

## What the scripts must do

```python
smoother = instantiate(cfg.smoothing.smoother, num_time_points=...)
enkf = instantiate(cfg.filtering, observation_operator=..., forward_model=..., C_D=...)
operator = instantiate(cfg.observation.operator, ...)
```
