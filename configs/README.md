# configs/

The Hydra config tree read by the scripts in `scripts/`. The design and the
mapping from the retired `conf/` tree (now `archive/conf/`) are in
[docs/config_setup_spec.md](../docs/config_setup_spec.md).

```
forward.yaml        # entry point: forward runs
assimilation.yaml   # entry point: all three DA scripts (smoothing, filtering, hybrid)
common.yaml         # run name, paths (per machine), ensemble budget, Hydra run dir
model/  case/  params/
assimilation_settings/   # one file per component, all options inside
```

All three DA scripts load `assimilation.yaml`; each reads the blocks it needs
(`smoothing`, `filtering`, and for the hybrid also `hybrid`).

`workflows/` chains a run with its post-processing; the overrides go to the
run script, and the post-processing reads the run dir it wrote:

```bash
bash workflows/forward_workflow.sh model=pylbm                  # run_forward + visualize_forward
bash workflows/assimilation_workflow.sh smoother <overrides>    # or filtering / hybrid; run_<method>
                                                                #   + compute_metrics + visualize_assimilation
```

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

## Neural surrogates

All surrogate configs live in `surrogate/`; run them with
`config_name="surrogate/<name>"`.

```
surrogate/generate_data.yaml           # generate training data (like forward.yaml + `data`)
surrogate/train_stepper.yaml           # train the next-step surrogate
surrogate/train_autoencoder.yaml       # train the field autoencoder
surrogate/train_latent_generator.yaml  # train the latent initial-field generator
surrogate/train_dft.yaml               # fine-tune a pretrained autoencoder into a stepper (DFT)
surrogate/finetune_stepper.yaml        # fine-tune a pretrained stepper (full or LoRA)
surrogate/eval.yaml                    # all evaluation: blocks stepper (one or several
                                       #   models), autoencoder, latent_generator
surrogate/training.yaml                # training defaults every training config shares
surrogate/architectures.yaml           # every stepper architecture, <family>_<size>
params/surrogate_training_data.yaml    # sampler for the training data
```

Each training config loads `training.yaml` (shared `trainer`, `optimizer`,
`batch_sampler`, `dataloader`, `dataset`) and overrides only what differs, so
the composed config is already complete. Examples:

```bash
'architecture=${architectures.unet_convnext_small}'                 # train_stepper
data.geometry.mode=fixed 'data.geometry.name=${case_name}'          # generate_data
```

The scripts are in `scripts/surrogate/`:

```bash
python scripts/surrogate/generate_data.py
python scripts/surrogate/train.py --config-name surrogate/train_<task>   # or finetune_stepper
python scripts/surrogate/evaluate_stepper.py            # all read surrogate/eval.yaml
python scripts/surrogate/evaluate_autoencoder.py
python scripts/surrogate/evaluate_latent_generator.py
```

`train.py` builds the model and datasets for the config's `task` from
`scripts/utils/tasks.py`; the evaluation scripts share
`scripts/utils/eval_common.py`.

Hydra treats `surrogate/` as a config group, so every file there starts with
`# @package _global_` (keys stay at the top level, not under `surrogate.`) and
lists root configs with a leading slash (`- /common`, `- /model: pyudales`).
