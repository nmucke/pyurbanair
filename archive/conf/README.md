# Run configurations

A named experiment is the usual place to set up a run. It records the case,
backends, parameter distributions, execution budget and deliberate deviations
from workflow defaults. Choose one with `experiment=<workflow>/<name>`; use a
CLI override for a one-off variation. An experiment is optional.

```bash
python scripts/esmda/run_esmda.py experiment=esmda/barcelona_dynamic
python scripts/esmda/run_esmda.py experiment=esmda/barcelona_dynamic ensemble.ensemble_size=64
python scripts/run_forward_model.py case=barcelona model@model=pylbm
```

Hydra applies shared runtime policy and selected ingredients first, then the
workflow's own controls (`_self_`), then the named experiment, then CLI value
overrides. A recipe's `experiment.workflow` identifies its intended entry point;
the lightweight preview and runner validate it. Select a different group using
its mounted name, such as `model@assim_model=pylbm` or
`params@prior_params=static`. Add a new recipe under `experiment/<workflow>/`
without chaining it to another recipe.

| Setting | Default owner | Where to edit |
| --- | --- | --- |
| Grid, physical bounds, STL, sensors, window duration | `case/` | Add or select a physical site. |
| Solver construction | `model/` | Select each role with `model@...`. |
| Truth and prior distributions | `params/` | Mount independently for assimilation. |
| Members and workers | `execution/` | Select a preset or override `ensemble.*`. |
| Failure policy, common run flags, Hydra directory | `common/runtime.yaml` | Override only for a specific run. |
| ESMDA/filtering scalars and components | `esmda/`, `filtering/` | Select a component group or override its field. |
| Observation uncertainty | `observation/error.yaml` | Edit shared defaults or override `observation_error.*`. |
| Observation construction | `observation/` | Select operator and aggregation groups. |
| Surrogate data and training | `neural_surrogate/` | Training modes bundle architecture, trainer and loss. |

The physical `time.simulation_time`, `time.output_frequency` and
`time.spinup_time` belong to the case. The workflow keeps
`time.seconds_per_knot`; those fields refer to the same window but have
different owners. Current output locations are preserved in each entry point's
`paths.*` fields. `run.results_dir` is an optional override for an actual run;
`run.name` labels named recipes. Scratch `paths.experiment_dir` stays absolute
for solvers that change directories.

The main entry points are `run_forward_model.yaml`, `run_esmda.yaml`,
`run_filtering.yaml`, `run_filter_smoothing.yaml`, `compare_models.yaml` and
`run_probe_series.yaml`. The probe config composes the same ESMDA ingredients
with separate scratch space. `neural_surrogate/training_data.yaml` composes its
own data-generation inputs: `training_data/geometry_mode=fixed case=barcelona` selects a fixed site.
The default `geometry_mode=random` and `training_data.geometry.source=idealized|realistic`
select a random geometry pool for the separate random-geometry generator.
Training, fine-tuning, testing, autoencoder and latent-generator configs stay
independent.

ESMDA's `esmda.interval_seconds` and `esmda.aggregation_mode` remain the
editable aggregation values for existing artifact readers; the selected
`observation/aggregation` group uses them to construct the object. The filter
selects `observation/aggregation=none` because it assimilates individual
frames. The dynamic smoother's knot count is derived from sampled parameters
at runtime and is deliberately absent from YAML.

Use Hydra's resolved preview to inspect a selected run without starting it:

```bash
python scripts/preview_config.py run_esmda experiment=esmda/barcelona_dynamic
```

The preview lists selected configs and deferred runtime inputs. A normal run
also saves `config.resolved.yaml` with interpolations expanded and
`run_manifest.yaml` with selected groups, CLI overrides and runtime constructor
arguments. Existing `config.yaml` artifacts keep their reader-facing keys.
For random-geometry shards, the resolved config comes from the frozen plan;
`config.requested.yaml` records the shard's launch request separately. Its
manifest labels requested group choices explicitly because the plan does not
preserve the original Hydra defaults list.

Observation uncertainty has one owner: `observation/error.yaml`, included by all
three assimilation workflows and mounted at `observation_error`. Instrument
noise is added to synthetic measurements; representation uncertainty only
widens the likelihood. The algorithm-level `obs_error_std` configuration keys
have been removed. The similarly named field in saved observation artifacts
still records physical marginal standard deviations.
