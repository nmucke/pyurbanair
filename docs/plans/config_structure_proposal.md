# Proposal: readable run configurations

Status: proposal for review, 2026-09-28. No configuration or script migration has
been implemented. `libs/` is outside the change scope, including during the
proposed migration. Constructor and exported-model contracts must remain intact.

## Recommendation

Make **a named experiment file the normal place to set up a run**. It selects the
case, backends, parameter distributions, algorithm and execution preset, and
shows the settings that distinguish that run. Keep reusable ingredients in
recognizable groups, and make each setting have one default owner.

Keep the existing workflow entry points and useful groups (`case/`, `model/`,
`params/`, `esmda/`, `filtering/`). Add `experiment/`, `execution/`, and a small
`common/` directory. Remove repeated default blocks and placeholder values that
another file immediately replaces. Use Hydra targets for selectable components;
keep the sequence of operations visible in Python.

This deliberately avoids both a giant universal config and a directory of tiny
files that must all be opened to understand one run. A user normally reads their
experiment file, the selected case, and the component they want to tune.

## What the audit found

Three Sol agents reviewed config ownership, instantiation boundaries, and
workflow usability. The following findings were checked against source rather
than assuming the existing documentation describes current defaults.

| Current issue | Evidence | Proposed response |
|---|---|---|
| Shared blocks are copied across entry points, with a mixture of intentional and accidental differences. | [Forward](../../conf/run_forward_model.yaml), [ESMDA](../../conf/run_esmda.yaml), [filtering](../../conf/run_filtering.yaml), [hybrid](../../conf/run_filter_smoothing.yaml), [comparison](../../conf/compare_models.yaml). Forward currently has 64 members and one worker; ESMDA has 50 and eight. | Extract common policy once; use mutually exclusive execution presets for different budgets. Preserve current defaults during extraction. |
| The same node is defined by a root placeholder and subsequently by a group. | ESMDA declares `localization: null` but selects `esmda/localization: correlation`; filtering similarly defines analysis/evolution defaults before group overrides. | Let the selected group own the final node; remove the placeholders. |
| Hydra composition can be superseded by script logic. | [`_apply_geometry_source`](../../scripts/neural_surrogate/generate_training_data.py) loads another case after composition; its case fields override even CLI values. | Use `case=` as the sole fixed-geometry selection; eliminate the post-composition case merge. |
| Some apparently configurable values are always replaced at construction. | [`esmda/smoother/dynamic.yaml`](../../conf/esmda/smoother/dynamic.yaml) declares `num_time_points: 1`; ESMDA derives it from sampled parameters. | Omit runtime-only constructor arguments from editable YAML and record the actual values separately. |
| Main objects already use Hydra. | Models, ensembles, samplers, smoothers, filters, and filter components already have `_target_` blocks. | Extend that pattern to remaining genuine construction choices; do not replace working declarative construction with a new factory layer. |
| Documentation repeats defaults and has drifted. | The [config README](../../conf/README.md) and [scripts reference](../scripts_and_configs.md) describe different entry-point inventories and some outdated values. | Keep one ownership/usage guide; derive effective values from composition rather than duplicating them in prose tables. |

Renaming folders alone would leave these problems in place.

## Proposed folder structure

The tree below is the intended organization, not a set of files created by this
proposal. Existing options abbreviated with `...` remain available.

```text
conf/
  README.md                         # where to edit, examples, composition rules

  run_forward_model.yaml            # workflow composition + its run controls
  run_esmda.yaml
  run_filtering.yaml
  run_filter_smoothing.yaml
  compare_models.yaml
  run_probe_series.yaml
  render_les.yaml

  common/
    runtime.yaml                    # shared output policy, Hydra, failure policy

  execution/
    forward.yaml                    # complete member/worker/CPU budget
    assimilation.yaml
    comparison.yaml
    smoke.yaml                      # small execution budget; not physical grid

  experiment/                       # user-facing named runs; one file per run
    forward/
      xie_truth.yaml
    esmda/
      barcelona_dynamic.yaml
    filtering/
      xie_joint.yaml
    filter_smoothing/
      xie_dynamic.yaml
    comparison/
      xie_backends.yaml
    surrogate/
      barcelona_training.yaml

  case/                             # physical setup stays together
    xie_and_castro.yaml
    barcelona.yaml

  model/                            # backend construction and lifecycle hooks
    pylbm.yaml
    pyudales.yaml
    pypalm.yaml
    neural_surrogate.yaml

  params/                           # directly instantiable samplers
    static.yaml
    static_truth.yaml
    dynamic.yaml
    dynamic_truth.yaml
    dynamic_sine.yaml
    dynamic_cosine.yaml

  observation/                      # new: construction, not sensor coordinates
    operator/
      spatial_points.yaml
      temporal_points.yaml
      spatial_grid.yaml
      temporal_grid.yaml
    aggregation/
      none.yaml
      mean.yaml
      ...

  esmda/
    default.yaml                    # shared ESMDA settings, used by hybrid too
    smoother/                       # existing options and targets
    localization/
    state_reduction/

  filtering/
    default.yaml                    # shared filter settings and filter target
    analysis/
    localization/
    state_reduction/
    inflation/
    evolution/

  neural_surrogate/                 # separate training/data workflows
    training_data.yaml
    training.yaml
    finetuning.yaml
    testing.yaml
    comparison.yaml
    pretrain_autoencoder.yaml
    testing_autoencoder.yaml
    train_latent_generator.yaml
    testing_latent_generator.yaml
    dataset/                        # new: dataset paths/splits/loading settings
    mode/                           # compatible architecture/trainer/loss bundles
    finetune_mode/
    architectures/

  render_preset/                    # keep the independent rendering workflow
```

Root workflow files should be short enough to scan: a defaults list followed by
workflow controls. Avoid introducing another `workflow/` or `profile/` layer
between them and their ingredients. `esmda/default.yaml` and
`filtering/default.yaml` exist because the hybrid genuinely reuses those
algorithms, not just to reduce the length of root files.

The `common/runtime.yaml` file is deliberately small. It owns the shared output
directory policy, `hydra.job.chdir: false`, shared run flags, and ensemble failure
policy. It does not own grids, algorithm parameters, or training settings.
Independent training/rendering entry points should only include it if its fields
apply; sharing a directory does not require sharing a universal runtime schema.

## Ownership and precedence

“One owner” means one default definition **per composed run**, not that a YAML
key can only appear once anywhere in the repository. Alternative backends and
presets necessarily expose some of the same keys. An experiment can intentionally
override defaults, and the CLI can override the experiment.

| Question | Default owner | Runtime location / rule |
|---|---|---|
| Where are the buildings, domain, grid and sensors? | `case/<name>.yaml` | Keep `domain`, `geometry`, `obs`, and physical `time` fields together in one file. |
| What backend is used, and with what numerical settings? | `model/<backend>.yaml` | Keep `model`, `truth_model`, `assim_model`, and `models.<name>` mounts. |
| What distributions or forcing profiles are used? | `params/<option>.yaml` | Keep separate `truth_params` and `prior_params`; their seeds and distributions are independent choices. |
| How far apart are parameter knots? | Workflow entry point initially | Keep `time.seconds_per_knot` as one shared input during migration; model/sampler targets reference it. Case files own the other physical time fields. Document this field-level split. |
| How large is the ensemble and how many workers are used? | Selected `execution/<preset>.yaml` | Mount directly at `ensemble`; owns size, workers and CPUs per worker. No automatic parallelism increases. |
| What happens on member failure? | `common/runtime.yaml` | Owns `ensemble.failure`; an experiment may explicitly override it. Data generation's required `raise` policy must be visible and validated. |
| Which algorithm settings apply? | `esmda/default.yaml`, `filtering/default.yaml`, selected component files | Default files own scalar settings; selectable groups exclusively own their component node. |
| Are forecasts saved on disk? How many forward rollout steps? | Workflow entry point | Workflow-specific `run.*`; these are not hardware preset settings. |
| Where are artifacts written? | `common/runtime.yaml` | One output policy, a single optional `run.results_dir` override, and separately defined scratch space. |
| What is special about this particular run? | `experiment/<workflow>/<name>.yaml` | All deliberate selections and deviations are visible together. |
| What depends on loaded or sampled data? | Runtime preparation code | Separate derived metadata and explicit constructor arguments, not hidden config defaults. |

Initially preserve existing runtime key names wherever possible. In particular,
case files may retain their explicit `# @package _global_` because they own a
documented bundle of namespaces. Moving everything under `case.*` would add
unrelated script and saved-artifact churn to the first migration.

For output paths, introduce `run.name` and one shared results-root rule rather
than five fallback policies. Resolve scratch paths relative to a stable launch
directory, including under the Compose API. Preserve old output locations via
explicit compatibility settings until shell pipelines and readers are migrated.
Never change output locations silently as part of extracting common defaults.

Use this order consistently:

1. Shared runtime policy and selected ingredients.
2. Workflow body (`_self_`), defining only workflow-owned values.
3. Optional named experiment, which contains deliberate overrides.
4. CLI value overrides.

Group selection overrides replace selected options rather than adding a second
backend on top. Hydra merges dictionaries and later definitions can overwrite
earlier values; `_self_` makes the body position explicit. The proposal uses
those existing rules, not a new merge engine.
See [Hydra 1.3 defaults-list semantics](https://hydra.cc/docs/1.3/advanced/defaults_list/).

Additional conventions:

- Every file has a short ownership comment; use `_global_` only for documented
  bundles such as common policy, cases and experiments.
- No root `localization: null` when a localization group supplies that node.
  The `none` option owns the null value.
- No experiment-to-experiment inheritance. Reuse ingredients, not chains of runs.
- No automatic hardware detection that changes scientific settings, and no
  custom resolvers that read datasets or construct objects during composition.
- CLI overrides remain authoritative for user-owned settings. Reject invalid
  combinations rather than silently switching them to another algorithm.
- Keep comments about units and non-obvious constraints beside values. Move
  long tutorials and historical alternatives out of YAML.

## What setting up a run would look like

The examples below are proposed configuration excerpts, not runnable replacements
for the current repository. Existing scalar values would be migrated intact.

A root ESMDA file becomes a composition map:

```yaml
# conf/run_esmda.yaml
defaults:
  - common/runtime
  - execution: assimilation
  - case: xie_and_castro
  - model@truth_model: pyudales
  - model@assim_model: pyudales
  - params@truth_params: dynamic_sine
  - params@prior_params: dynamic
  - esmda: default
  - esmda/localization: correlation
  - esmda/state_reduction: none
  - esmda/smoother: dynamic
  - _self_
  - experiment: null

run:
  name: esmda
  ensemble_save_on_disk: true
  save_prior_state: false
  truth_dir: null
  truth_start_time: null

time:
  seconds_per_knot: 60.0

params_to_estimate: [inflow_angle, velocity_magnitude]
```

`execution/assimilation.yaml` declares `# @package ensemble` and contains
`ensemble_size`, `num_parallel_processes`, and `num_cpus_per_process` directly.
`common/runtime.yaml` declares `# @package _global_`. `esmda/default.yaml`
uses its normal `esmda` package and contains shared scalar algorithm settings;
it does not redefine the selectable component nodes.

A named run records choices together:

```yaml
# conf/experiment/esmda/barcelona_dynamic.yaml
# @package _global_
defaults:
  - override /case: barcelona
  - override /execution: assimilation
  - override /model@truth_model: pyudales
  - override /model@assim_model: pylbm
  - override /params@truth_params: dynamic_sine
  - override /params@prior_params: dynamic
  - override /esmda/smoother: dynamic
  - override /esmda/localization: correlation

run:
  name: barcelona_dynamic
ensemble:
  ensemble_size: 32
esmda:
  num_steps: 4
  num_assimilation_windows: 3
```

This is a reviewable setup, not a recommendation that these numerical choices
are appropriate for every Barcelona study. Repeating a selection already used
by the launcher is useful here: it makes the experiment's intent stable when
launcher defaults change. The normal invocation would become:

```bash
pixi run -e dev python scripts/esmda/run_esmda.py experiment=esmda/barcelona_dynamic
```

A one-off variation remains small:

```bash
pixi run -e dev python scripts/esmda/run_esmda.py \
  experiment=esmda/barcelona_dynamic ensemble.ensemble_size=64
```

Adding a physical site still means adding one case file. Adding a run at that
site means adding one experiment file. Switching only the assimilation backend
remains `model@assim_model=pyudales`. Experiment directories indicate their
workflow, and validation should reject an experiment used with the wrong entry
point. Record an explicit expected-workflow field in production recipes for
that validation; the abbreviated example above omits this metadata.

This follows Hydra's existing
[experiment configuration pattern](https://hydra.cc/docs/1.3/patterns/configuring_experiments/).

## Use instantiate for construction

Retain the current direct model, sampler, smoother, and filter targets. Replace
remaining class-selection branches when the selected object is genuinely
configurable. Do not build a generic `create_everything(cfg)` function that
simply hides the same decisions elsewhere.

A fixed-point observation operator can directly target the existing library:

```yaml
# conf/observation/operator/temporal_points.yaml
_target_: data_assimilation.observation_operator.TemporalObservationOperator
observation_operator:
  _target_: data_assimilation.observation_operator.ObservationOperator
  _convert_: all
  obs_x: ${obs.x_points}
  obs_y: ${obs.y_points}
  obs_z: ${obs.z_points}
  obs_states: ${obs.states}
  # solver_name is supplied for the model role at construction time.
```

When introducing this group, add `observation/operator: temporal_points` to the
workflow defaults; this mounts the template at `observation.operator`. The
workflow can instantiate it for each role without embedding truth-specific
wiring in a reusable file:

```python
truth_operator = instantiate(
    cfg.observation.operator,
    observation_operator={"solver_name": cfg.truth_model.solver_name},
)
assim_operator = instantiate(
    cfg.observation.operator,
    observation_operator={"solver_name": cfg.assim_model.solver_name},
)
```

A spatial option targets `ObservationOperator` directly and takes the role's
`solver_name` at the top level. A small role-construction adapter may normalize
this injection if both options are supported by one workflow. Grid sensor
generation still needs a small numeric adapter to create coordinates. Keep that
adapter in `src/pyurbanair/config/`, with explicit inputs; do not alter the
observation library. Validation sensors remain a separate, held-out operator.

Aggregation can target `AggregateObservations` directly, with a `none` option
for unaggregated observations. When migrating `interval_seconds` and
`aggregation_mode`, choose one writable owner and use compatibility projections
for old readers; do not retain two independently editable aggregation configs.

Use partial instantiation only when construction must wait for runtime values;
do not recursively instantiate the whole root. Hydra supports nested targets,
keyword overrides, partial construction, and conversion controls. Existing
`_recursive_: false` on the surrogate must survive because its constructor owns
the nested spin-up configuration.
See [Hydra object instantiation](https://hydra.cc/docs/1.3/advanced/instantiate_objects/overview/).

| Move into declarative construction | Keep explicit in the workflow |
|---|---|
| Observation operator and aggregation variants | Loading truth, sensor coordinate calculations, validating compatibility |
| Existing model, ensemble, sampler and algorithm choices | Compile/preprocess/cleanup lifecycle, with named backend hooks where useful |
| Existing training architecture/trainer/loss variants | Dataset inspection, normalization, training and evaluation loops |
| Backend-specific hook selection where it removes dispatch | PRNG splitting, observation-sized covariance, truth and cycle horizons |
| Constructor parameters with fixed or interpolated inputs | Per-window state changes, warm starts, knot counts, artifact writes |

For example, omit `num_time_points: 1` from the dynamic smoother YAML and pass
the sampled knot count explicitly to `instantiate`. This requires a coordinated
script change: the current ESMDA runner checks whether `num_time_points` exists
in the config before supplying it. Replace that presence check with an explicit
dynamic-mode/capability check in the same change; the constructor requires the
argument. Do not substitute `???`
unless the user really must provide the value: that would make normal resolved
previews fail for a value only the program can compute.

The resulting Python should visibly read as: validate configuration, prepare
runtime inputs, instantiate components, prepare solvers, execute the window
loop, save outputs. Future helpers should describe one operation and accept
narrow inputs. Numerical loops will still have real complexity; the goal is to
remove configuration plumbing from them.

## Make the applied configuration inspectable

For ordinary settings, users should be able to inspect the same experiment
without executing the workflow:

```bash
pixi run -e dev python scripts/esmda/run_esmda.py \
  experiment=esmda/barcelona_dynamic --cfg job --resolve

pixi run -e dev python scripts/esmda/run_esmda.py \
  experiment=esmda/barcelona_dynamic --info defaults
```

Hydra provides resolved configuration output and composition diagnostics.
The defaults listing shows file/package ordering; it is not a per-field
provenance report. See [Hydra debugging tools](https://hydra.cc/docs/1.3/tutorials/basic/running_your_app/debugging/).

Add a lightweight preview/validation entry point during implementation. It must
compose and validate without importing solver modules, compiling, running
preprocessing, sampling large datasets, or invoking `run(cfg)`. Existing script
imports can have side effects even when Hydra later exits for `--cfg`, so do
not advertise current script previews as fully isolated until imports are
audited. Use the lightweight entry point for the guaranteed safe preview.

The preview should show selected files, resolved user settings, and a short list
of deferred runtime arguments. Derived values known from config, such as a
requested horizon, can be shown immediately. Values requiring data should say
what supplies them; a preview should never claim they are known in advance.

Every actual run should save:

- `config.yaml`: retain the schema expected by existing metric and artifact
  readers during migration.
- `config.resolved.yaml`: resolved user configuration at launch, with no later
  silent changes to user-owned settings.
- `run_manifest.yaml`: selected config choices, CLI overrides, code revision,
  resolved artifact/scratch paths, and the runtime constructor overrides that
  affect behavior. Include role and window when values differ by role/window.
- Hydra's normal metadata when invoked through Hydra. Direct `run(cfg)` tests
  must remain supported without depending on `HydraConfig` being initialized.

Do not serialize datasets or live model objects into the manifest. Record
shapes, paths, selected parameter names, seeds and concrete scalar arguments.
These records distinguish configuration intent from data-dependent execution.

## Special workflows and compatibility

**Surrogate data generation:** select a fixed site through `case=` only. Remove
the second fixed-case selector at `training_data.geometry.source`; a transitional
alias must be translated before composition or reject conflicting selections.
Random geometry pools are a different input mode, with their grid and bounds
recorded per generated sample. Do not pretend a single pre-run case config can
describe all sampled geometries. Dataset-generation failure policy and time
horizon overrides must be visible in configuration/manifest, not quietly forced
inside the runner.

**Surrogate training:** preserve mode bundles that jointly select compatible
architecture, trainer and loss. The selected mode owns the trainer target;
remove the inline target that the mode always overrides. `dataset/` owns
reusable paths, splits and loader settings; architecture presets own network
shape. Keep training, testing, fine-tuning, autoencoder and latent-generator
entry points separate. Exported `model_weights/.../config.yaml` is consumed by
libraries and must keep its existing contract. If the launch config changes,
translate at export in script/application code, without modifying `libs/`.

**Hybrid assimilation:** reuse the two algorithm settings groups. The hybrid
retains one owner for shared seed, observation error and window count, with
explicit interpolation/projection into consumers. Preserve the scientific
difference between filtering chunk boundaries and hybrid/ESMDA windows; do not
flatten them into a generic horizon knob that obscures the difference.

**Comparison and probes:** preserve N-way model/scenario mounts for comparison.
Replace probe/data-generation inheritance from another executable's complete
config with direct composition of the needed ingredients. This prevents a
change to a forward or assimilation launch default from changing a diagnostic
workflow unexpectedly.

**Shell pipelines, metrics and figures:** inventory all config readers before
renaming keys. Preserve artifact layouts and CLI selectors in the first stage.
Argparse utilities that process existing artifacts can remain argparse utilities.
No universal Hydra runner is required.

## Migration and acceptance criteria

Implement in reviewable stages, with each stage preserving behavior except for
explicitly documented fixes to misleading precedence.

1. **Capture current composition.** Inventory every `@hydra.main` entry and shell
   composer, including hybrid, comparison, probes, all surrogate workflows and
   rendering. Capture resolved configuration trees and runtime overrides for
   representative runs. Use live YAML defaults, not values copied from docs.
2. **Extract ownership without changing public keys.** Add common runtime and
   execution presets; extract shared algorithm settings; remove overwritten
   placeholders. Preserve different historical workflow budgets and paths.
3. **Add named experiments and preview.** Provide a few complete recipes, enforce
   workflow compatibility, document precedence, and standardize run manifests.
   Do not require experiment files for quick CLI-only runs.
4. **Remove hidden configuration decisions.** Make fixed geometry selection
   declarative, remove fake runtime defaults, and validate invalid combinations
   before side effects. Preserve old artifact formats via explicit projections.
5. **Expand instantiate and simplify runners.** Introduce observation targets and
   selected lifecycle hooks, retaining small adapters only where existing APIs
   require them. Keep all implementation changes outside `libs/`.
6. **Update the maintained guides and examples.** Rewrite `conf/README.md` around
   user journeys and ownership; synchronize `docs/scripts_and_configs.md` and
   `docs/codebase_guide.md`. Keep this proposal a design record, not a second
   reference manual.

Acceptance checks should cover:

- Composition and resolution for every entry point, both cases, each model
  mount, each smoother/filter component family, and representative mixed-model
  and surrogate configurations. These checks must not instantiate solvers.
- Before/after equality of resolved settings and actual constructor arguments,
  normalizing timestamps and temporary paths. Maintain an explicit list of
  intentional schema/precedence changes rather than ignoring broad subtrees.
- CLI values win over recipes, recipes win over defaults, and switching a case
  does not retain fields unique to the previous case. Verify package placement
  for truth, assimilation, comparison, and named parameter scenarios.
- Invalid combinations fail with the setting named: dynamic/static mismatch,
  unsupported hybrid modes, localization/reduction conflicts, wrong experiment
  workflow, inconsistent filter stride, and missing external artifacts.
- Preview imports no backend and creates no run directories; missing
  user-required paths are reported without constructing a model.
- Existing small workflow smoke tests after script changes, plus saved-config
  reader checks and unchanged surrogate export contracts. No full CFD campaign
  is needed to validate a folder refactor.
- No changed files under `libs/`, no changed multiprocessing context, and no
  incidental change to parallelism, scientific defaults, or failure behavior.

The proposal's experiment packaging, group overrides, CLI priority, and
interpolation pattern were checked with an isolated temporary fixture using the
repository's installed Hydra 1.3.2. That verifies the composition pattern, not
the full proposed migration or scientific compatibility of the example run.
No solvers were instantiated for this proposal.
