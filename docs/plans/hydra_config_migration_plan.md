# Hydra Configuration Migration Plan

## Goal

Replace `scripts/config.py` as the mutable, global source of run configuration
with Hydra/OmegaConf configs that are composable by model, experiment, ensemble,
observation setup, and ESMDA variant.

The migration should make these workflows easy:

```bash
python scripts/run_forward_model.py model@model=pylbm
python scripts/run_parameter_esmda.py model@truth_model=pylbm model@assim_model=pyudales
python scripts/run_time_varying_parameter_esmda.py \
  model@truth_model=pylbm model@assim_model=pylbm esmda.num_steps=4 obs.interval_size=2
python scripts/run_rollout_esmda.py +preset=small model@truth_model=pyudales model@assim_model=pyudales
```

The important design choice is that model-specific construction becomes data in
Hydra config groups, while procedural pieces that depend on run-time values
remain small Python helper functions.

## Current Shape To Preserve

The current code has useful boundaries:

- Run scripts import `scripts.config`.
- `scripts/config.py` stores dict constants such as `DOMAIN`, `TIME`,
  `LBM_ARGS`, `UDALES_ARGS`, `PALM_ARGS`, `ENSEMBLE`, `OBS`, `ESMDA`,
  `TRUE_PARAMS`, `PARAM_PRIORS`, `EXTERNAL_PRIORS`, and
  `TIME_VARYING_PARAMS`.
- `pyurbanair.utils.config_utils` centralizes model selection and factory
  logic: forward model creation, preparation, ensemble wrapping, observation
  operator creation, parameter priors, and covariance creation.
- PALM must stay lazy: importing `pypalm` can compile/download solver assets,
  so the selected config must not import PALM unless `pypalm` is actually used.
- Several invariants are easy to break and worth flagging explicitly:
  - `pressure_gradient_magnitude` is **uDALES-only**. `create_true_params`
    and `create_time_varying_true_params` filter it on `model_name == "pyudales"`.
    The Hydra schema must preserve that filter — see "Parameters And
    Time-Varying Priors" below.
  - The truth-trajectory generator in
    `create_time_varying_true_params` uses a *distinct* correlation length
    from the assimilation prior to avoid the inverse crime. The Hydra
    config must keep `truth_method_kwargs` separate from `method_kwargs`
    and never let them collapse via interpolation.
  - `solver_name` maps `pylbm → "pylbm"`, `pypalm → "palm"`, otherwise
    `"udales"` (see `config_utils.py:38-43`). This is what
    `ObservationOperator` keys on. Treat it as a per-model constant in
    the YAML, not a free string.
  - Init-condition directories use a related but different mapping:
    `pylbm → "lbm"`, `pypalm → "palm"`, `pyudales → "udales"`.
    Store this as a separate per-model `init_subdir`; do not derive it
    from `solver_name`.

Hydra should keep the useful centralization, but remove global module mutation
and make overrides declarative.

## Proposed Config Layout

Use a root-level `conf/` directory:

```text
conf/
  config.yaml
  paths/default.yaml
  domain/
    xie_castro_60x40x16.yaml
    xie_castro_100x80x16.yaml
    small.yaml
  time/
    default.yaml
    small.yaml
  model/
    pylbm.yaml
    pyudales.yaml
    pypalm.yaml
  ensemble/
    default.yaml
    small.yaml
  obs/
    xie_castro_points.yaml
    grid_small.yaml
  esmda/
    parameter.yaml
    state_and_parameter.yaml
    rollout.yaml
    time_varying_parameter.yaml
    time_varying_rollout.yaml
  params/
    true/default.yaml
    prior/default.yaml
    external/default.yaml
  time_varying/
    ar2_relaxation.yaml
    gp_linear_trend.yaml
    ar1.yaml
    ornstein_uhlenbeck.yaml
  preset/
    small.yaml
```

Top-level `conf/config.yaml`:

```yaml
defaults:
  - paths: default
  - domain: xie_castro_60x40x16
  - time: default
  - ensemble: default
  - obs: xie_castro_points
  - esmda: parameter
  - params/true: default
  - params/prior: default
  - params/external: default
  - time_varying: ar2_relaxation
  - model@model: pylbm
  - model@truth_model: pylbm
  - model@assim_model: pylbm
  - _self_

hydra:
  job:
    chdir: false
  run:
    dir: ${paths.base_results_dir}/hydra/${hydra.job.name}/${now:%Y-%m-%d}/${now:%H-%M-%S}

run:
  skip_viz: false
```

Use `model` for single-model scripts and `truth_model` / `assim_model` for
assimilation scripts. Hydra's package override syntax lets the same model group
be mounted multiple times.

**Pick one output-dir source of truth.** The current scripts each write to
`config.BASE_RESULTS_DIR / "<script_name>"` (e.g. `.temp/scripts/parameter_esmda`).
With `chdir: false` plus an auto-generated `hydra.run.dir`, both directories
get created and only one is actually used. Recommended resolution:

- Adopt Hydra's per-run dir as the script's output dir:
  ```python
  from hydra.core.hydra_config import HydraConfig
  out_dir = pathlib.Path(HydraConfig.get().runtime.output_dir)
  ```
  This gives free run isolation and a timestamp, and the per-script subdir
  (`parameter_esmda`, `time_varying_parameter_esmda`, etc.) is encoded in
  `${hydra.job.name}` automatically.
- Or keep the legacy fixed `cfg.paths.base_results_dir / <name>` layout and
  disable the auto run dir with `hydra.output_subdir: null` and
  `hydra.run.dir: .`. This is simpler but loses run isolation.

The plan assumes option (a); call this out in PRs that touch run scripts so
result-path expectations stay consistent.

Because tests will call `run(cfg)` from a composed config, script code should
not call `HydraConfig.get()` directly throughout the run body. Add one helper
that falls back when Hydra's runtime singleton is not initialized:

```python
def resolve_output_dir(cfg: DictConfig, run_name: str) -> pathlib.Path:
    if HydraConfig.initialized():
        return pathlib.Path(HydraConfig.get().runtime.output_dir)
    return pathlib.Path(cfg.paths.base_results_dir) / run_name
```

Tests can override `paths.base_results_dir` to a pytest `tmp_path`; CLI runs
still use Hydra's timestamped output dir.

## Model Configs And `instantiate`

Each model config should contain:

- `name`: user-facing model selector, e.g. `pylbm`.
- `solver_name`: name expected by `ObservationOperator`, e.g. `udales`.
- `forward_model`: `_target_` config for the selected backend.
- `ensemble_model`: `_target_` config for the selected backend's ensemble
  wrapper.
- `prepare`: `_target_` config for a small preparation helper.
- `cleanup`: optional `_target_` helpers for output/restart cleanup.

Example `conf/model/pylbm.yaml`:

```yaml
name: pylbm
solver_name: pylbm
init_subdir: lbm

forward_model:
  _target_: pylbm.forward_model.ForwardModel
  stl_path: examples/lbm/experiments/xie_castro_2008_STL.stl
  experiment_name: runcase
  cuda: true
  verbose: false
  boundary_condition: inflow_outflow
  nx: ${domain.nx}
  ny: ${domain.ny}
  nz: ${domain.nz}
  bounds: ${domain.bounds}
  simulation_time: ${time.simulation_time}
  output_frequency: ${time.output_frequency}
  spinup_time: ${time.spinup_time}

compile: true

prepare:
  _target_: pyurbanair.config.hydra_helpers.prepare_lbm
  compile: ${..compile}

ensemble_model:
  _target_: pylbm.ensemble_forward_model.EnsembleForwardModel
  ensemble_size: ${ensemble.ensemble_size}
  num_parallel_processes: ${ensemble.num_parallel_processes}
  num_cpus_per_process: ${ensemble.num_cpus_per_process}
```

Example `conf/model/pyudales.yaml`:

```yaml
name: pyudales
solver_name: udales
init_subdir: udales

forward_model:
  _target_: pyudales.forward_model.ForwardModel
  case_dir: examples/udales/experiments/xie_and_castro
  experiment_name: "999"
  matlab_bin: /opt/sw/matlab-2023b/bin/matlab
  ncpu: 1
  save_only_last_timestep: false
  verbose: false
  boundary_condition: inflow_outflow
  nudging_config:
    tnudge: 30.0
    nnudge: 4
    profile_config:
      type: power_law
      alpha: 0.25
  nx: ${domain.nx}
  ny: ${domain.ny}
  nz: ${domain.nz}
  bounds: ${domain.bounds}
  simulation_time: ${time.simulation_time}
  output_frequency: ${time.output_frequency}
  spinup_time: ${time.spinup_time}

prepare:
  _target_: pyurbanair.config.hydra_helpers.prepare_udales
  python_or_matlab: python

ensemble_model:
  _target_: pyudales.ensemble_forward_model.EnsembleForwardModel
  ensemble_size: ${ensemble.ensemble_size}
  num_parallel_processes: ${ensemble.num_parallel_processes}
  num_cpus_per_process: ${ensemble.num_cpus_per_process}
```

`conf/model/pypalm.yaml` should mirror this shape, but keep all `pypalm.*`
targets only inside that file. Hydra will only import these targets when the
config is instantiated, preserving the current lazy import behavior. It should
also set `solver_name: palm` and `init_subdir: palm`.

Forward model construction in scripts becomes:

```python
from hydra.utils import instantiate

forward_model = instantiate(cfg.assim_model.forward_model, results_dir=results_dir)
instantiate(cfg.assim_model.prepare, forward_model=forward_model)
```

Ensemble construction becomes:

```python
ensemble_model = instantiate(
    cfg.assim_model.ensemble_model,
    forward_model=forward_model,
)
configure_failure_policy(ensemble_model, cfg.ensemble.failure)
```

Define the ensemble config with failure settings nested under `failure`, so
the shape matches the helper call above:

```yaml
# conf/ensemble/default.yaml
ensemble_size: 16
num_parallel_processes: 8
num_cpus_per_process: 1
failure:
  policy: resample_from_successes
  jitter_scale: 0.05
  seed: 0
```

Hydra's `instantiate` should be used for backend objects. Keep tiny helper
functions for operations that are not constructors, such as compilation,
preprocessing, failure-policy setup, and covariance creation.

**Decouple `@hydra.main` from the run body.** Today the scripts are
straight-line `def main()` functions; once decorated, they become hard to
unit-test (the decorator parses `sys.argv` at call time) and impossible to
import. Use the standard split:

```python
@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    run(cfg)

def run(cfg: DictConfig) -> None:
    ...
```

Tests then call `run(cfg)` with a programmatically composed config. This is
the only mechanism that lets the existing test suite — which today does
`sys.modules["scripts.config"] = tests_config` and mutates dicts between
subtests — move to Hydra without losing coverage. See the **Tests**
sub-section under Phase 5.

## Helper Module

Add `src/pyurbanair/config/hydra_helpers.py` with narrow functions:

```text
prepare_lbm(forward_model, compile: bool) -> None
prepare_palm(forward_model, compile: bool) -> None
prepare_udales(forward_model, python_or_matlab: str = "python") -> None
clean_outputs(model_name, forward_model) -> None
clean_restarts(model_name, forward_model) -> None
configure_failure_policy(ensemble_model, failure_cfg) -> Any
create_true_params(model_name, true_cfg) -> xarray.Dataset
create_parameter_ensemble(model_name, prior_cfg, ensemble_size, seed) -> xarray.Dataset
create_time_varying_true_params(model_name, tv_cfg, true_cfg, prior_cfg,
                                 simulation_time, num_time_points, seed) -> xarray.Dataset
create_initial_state_ensemble(state, ensemble_size) -> xarray.Dataset
create_observation_operator(obs_cfg, solver_name) -> TemporalObservationOperator
create_observation_points(obs_cfg) -> tuple[np.ndarray, np.ndarray, np.ndarray]
create_C_D(num_obs: int, obs_error_std: float) -> jnp.ndarray
load_init_conditions_for_esmda(model_name, init_conditions_dir, ensemble_size,
                               true_sim_id, init_subdir) -> tuple[...] | None
make_rng_key(seed: int) -> jax.Array
make_time_coords(simulation_time: float, num_time_points: int) -> jnp.ndarray
resolve_output_dir(cfg, run_name: str) -> pathlib.Path
```

Two of these are easy to forget because they were not in the original draft:

- `load_init_conditions_for_esmda` is used by `run_rollout_esmda.py` to
  resume from a saved init-conditions directory; it routes
  `pylbm → lbm`, `pypalm → palm`, else `udales`. That subdir mapping is
  deliberately not identical to `solver_name` for pylbm, so pass the
  selected model's `init_subdir` from YAML.
- `create_initial_state_ensemble` is used by `run_rollout_esmda.py` to
  broadcast a single warmup state to `ensemble_size` members.

`create_rollout_forward_model` from the current code is an identity function
(`config_utils.py:95-99`) and can simply be deleted at the call sites rather
than ported.

The truth-trajectory helper must take a *separate* `truth_method` and
`truth_method_kwargs` (different correlation length than the assimilation
prior) to preserve the current anti-inverse-crime behavior. Do not collapse
these into a single `${time_varying.method_kwargs}` interpolation.

These functions should accept explicit config values, not import
`scripts.config`. That is the key step that removes global mutable state.

During migration, `pyurbanair.utils.config_utils` can either:

1. Become a thin backwards-compatible adapter around the new helper functions,
   or
2. Stay temporarily for old scripts while migrated scripts use
   `pyurbanair.config.hydra_helpers` directly.

Option 2 is safer for incremental work.

## ESMDA Configs

ESMDA variants should live in separate config files. The variant config owns the
smoother class and parameters shared by that class.

Example `conf/esmda/parameter.yaml`:

```yaml
kind: parameter
num_steps: 2
alpha: ${.num_steps}
num_assimilation_windows: 5
seed: 42
obs_error_std: 0.25
init_conditions_dir: esmda_init_conditions
true_sim_id: 0

smoother:
  _target_: data_assimilation.smoothing.esmda.ParameterESMDA
  num_steps: ${..num_steps}
  alpha: ${..alpha}
```

Example `conf/esmda/time_varying_parameter.yaml`:

```yaml
kind: time_varying_parameter
num_steps: 2
alpha: ${.num_steps}
seed: 42
obs_error_std: 0.25

smoother:
  _target_: data_assimilation.smoothing.esmda.TimeVaryingParameterESMDA
  num_steps: ${..num_steps}
  alpha: ${..alpha}
  num_time_points: ${time_varying.num_time_points}
```

Use **sibling-relative** interpolation (`${.num_steps}`, `${..num_steps}`)
inside ESMDA configs rather than absolute (`${esmda.num_steps}`). The
absolute form silently breaks the moment the same config is mounted under
a different package — and the assimilation scripts already do exactly that
with `model@truth_model` / `model@assim_model`. Reserve absolute references
for cross-group lookups (e.g. `${time_varying.num_time_points}` above,
which genuinely lives in a different group).

Today's ESMDA dict carries `alpha` implicitly as a duplicate of
`num_steps` (see `run_parameter_esmda.py:70` —
`alpha=config.ESMDA["num_steps"]`). Keep that default in the YAML
(`alpha: ${.num_steps}`) so the existing behavior is reproduced, but let
users override `alpha` independently — that was impossible before.

The script still passes dynamic values:

```python
esmda = instantiate(
    cfg.esmda.smoother,
    observation_operator=assim_obs_op,
    forward_model=ensemble_model,
    C_D=C_D,
    rng_key=rng_key,
    time_coords=time_coords,
)
```

This keeps the object creation declarative while avoiding awkward YAML for
values only known after the truth observation vector is built.

## Parameters And Time-Varying Priors

Move the current dictionaries into explicit groups:

`conf/params/true/default.yaml`:

```yaml
inflow_angle: 10.0
velocity_magnitude: 5.0
pressure_gradient_magnitude: 0.0041912  # uDALES-only; filtered out for pylbm/pypalm
```

`conf/params/prior/default.yaml`:

```yaml
inflow_angle:
  mean: 0.0
  std: 10.0
velocity_magnitude:
  mean: 5.0
  std: 1.0
  min: 0.1
pressure_gradient_magnitude:
  mean: 0.0041912
  std: 0.001
```

`conf/params/external/default.yaml` (used by the time-varying classes for
prior sampling and between-window relaxation toward `x_ext`; the existing
`EXTERNAL_PRIORS` dict):

```yaml
inflow_angle:
  mean: 0.0
  std: 5.0
velocity_magnitude:
  mean: 5.0
  std: 0.5
  min: 0.1
```

`create_true_params` and `create_time_varying_true_params` must continue to
drop `pressure_gradient_magnitude` when `model_name != "pyudales"`. The YAML
intentionally always lists it: this keeps a single `params/*` group across all
models and pushes the model-conditional filter into the helper, matching
today's behavior at `config_utils.py:177-180` and `:253-256`.

`create_parameter_ensemble` is a separate compatibility decision. Today it
never includes `pressure_gradient_magnitude`, even for uDALES
(`config_utils.py:184-207`). Preserving that behavior keeps the migration
behaviorally neutral. Adding a pressure-gradient prior for uDALES may be a
good future fix, but it changes the ESMDA augmented parameter vector and should
be a deliberate follow-up, not an accidental Hydra side effect.

`conf/time_varying/ar2_relaxation.yaml`:

```yaml
num_time_points: 3
method: ar2_relaxation
method_kwargs:
  correlation_length: 300.0
truth_method: ar2_relaxation
truth_method_kwargs:
  correlation_length: 500.0

prior_model:
  _target_: pyurbanair.parameter_time_series.build_parameter_time_series
  method: ${time_varying.method}
  external_priors: ${params.external}
  ensemble_size: ${ensemble.ensemble_size}
  method_kwargs: ${time_varying.method_kwargs}
```

Because `num_time_points` currently depends on
`TIME["simulation_time"] / 60`, choose one of these:

- Prefer explicit `time_varying.num_time_points` in YAML for transparency.
- Optionally add an OmegaConf resolver later if derived config becomes common.

## Observation Configs

`OBS` today has two shapes — discrete points (`x_points/y_points/z_points`)
or a regular grid (`x_min/x_max/y_min/y_max/n_per_axis/z`) — and
`create_observation_points` branches on the presence of `x_points`. Replace
that implicit dispatch with an explicit discriminator so the schema is
self-documenting:

```yaml
# conf/obs/xie_castro_points.yaml
mode: points
x_points: [20.0, 20.0, 40.0, 50.0, 60.0]
y_points: [20.0, 60.0, 10.0, 40.0, 60.0]
z_points: [3.0, 3.0, 3.0, 3.0, 3.0]
states: [u, v, w]
temporal_mode: intervals
interval_size: 4
aggregation_mode: mean
```

```yaml
# conf/obs/grid_small.yaml
mode: grid
x_min: 5.0
x_max: 35.0
y_min: 5.0
y_max: 35.0
n_per_axis: 2
z: 2.0
states: [u, v, w]
temporal_mode: mean
```

`create_observation_points(obs_cfg)` then branches on `obs_cfg.mode` and
fails loudly on any other value.

## Command-Line Switching

Single-model scripts:

```bash
python scripts/run_forward_model.py model@model=pyudales
python scripts/run_ensemble_forward_model.py model@model=pylbm ensemble.ensemble_size=32
```

Truth/assimilation scripts:

```bash
python scripts/run_parameter_esmda.py model@truth_model=pylbm model@assim_model=pyudales
python scripts/run_state_and_parameter_esmda.py model@truth_model=pyudales model@assim_model=pyudales
python scripts/run_time_varying_parameter_esmda.py \
  model@truth_model=pylbm model@assim_model=pypalm assim_model.compile=false
```

Presets:

```bash
python scripts/run_parameter_esmda.py +preset=small
```

Hydra overrides replace script-specific override flags such as
`--esmda-num-steps`, `--obs-error-std`, and `--obs-interval`. Keep behavioral
flags that are not really configuration, such as `skip_viz`, as config fields
too:

```bash
python scripts/run_time_varying_parameter_esmda.py run.skip_viz=true
```

Translation table for the non-obvious overrides in
`run_time_varying_parameter_esmda.py::_apply_config_overrides`:

| Old CLI flag             | Old mutation target                                          | New Hydra override                                              |
|--------------------------|--------------------------------------------------------------|-----------------------------------------------------------------|
| `--esmda-num-steps N`    | `config.ESMDA["num_steps"] = N`                              | `esmda.num_steps=N`                                             |
| `--obs-error-std X`      | `config.ESMDA["obs_error_std"] = X`                          | `esmda.obs_error_std=X`                                         |
| `--obs-interval N`       | `config.OBS["interval_size"] = N`                            | `obs.interval_size=N`                                           |
| `--truth-corr-length X`  | `config.TIME_VARYING_PARAMS["truth_correlation_length"]=X`†  | `time_varying.truth_method_kwargs.correlation_length=X`         |
| `--prior-corr-length X`  | `config.TIME_VARYING_PARAMS["prior_correlation_length"]=X`†  | `time_varying.method_kwargs.correlation_length=X`               |
| `--num-par-time-points N`| script-local fallback to `TIME_VARYING_PARAMS["num_time_points"]` | `time_varying.num_time_points=N`                           |

† Today these write **shadow keys** that
`create_time_varying_true_params` reads via `tv.get("truth_correlation_length", sim_time/4)`
(see `config_utils.py:225`). After migration, the helper reads
`tv_cfg.truth_method_kwargs.correlation_length` directly and the shadow-key
path goes away. Audit the helper for any other `.get(...)` defaults that
were silently injected by CLI sweeps before deleting them.

Also note that `--prior-corr-length` currently appears to be a no-op for
`run_time_varying_parameter_esmda.py`: the script writes
`prior_correlation_length`, but then builds the prior from
`TIME_VARYING_PARAMS["method_kwargs"][method]`. Mapping it to
`time_varying.method_kwargs.correlation_length` is therefore a small behavior
fix, not a byte-for-byte migration.

## Migration Phases

### Phase 1: Add Hydra Dependency And Config Files

- Add `hydra-core>=1.3,<2` to `pyproject.toml` project dependencies and to
  the Pixi `[tool.pixi.dependencies]` table. The 1.3 floor is needed for
  sibling-relative interpolation (`${.foo}`) and the `group@package`
  override syntax this plan relies on.
- Create the `src/pyurbanair/config/` subpackage (it does not exist yet —
  add `__init__.py`).
- Add the `conf/` tree at repo root with defaults matching
  `scripts/config.py`.
- Add `src/pyurbanair/config/hydra_helpers.py`.
- Add a minimal test that **composes** (does not instantiate) `conf/config.yaml`
  for the three model choices and asserts that `pypalm` is not in
  `sys.modules` after composing a non-PALM model — that is the load-bearing
  PALM-laziness invariant.
- Before migrating scripts, verify the config grammar with a tiny composition
  test matrix:
  - `model@model=pyudales`
  - `model@truth_model=pylbm model@assim_model=pyudales`
  - `model@truth_model=pylbm model@assim_model=pypalm assim_model.compile=false`
  - Resolved values for `${..compile}`, `${.num_steps}`, and `${..num_steps}`
    match expectations.

### Phase 2: Migrate One Simple Script

- Start with `scripts/run_forward_model.py`.
- Replace `argparse --model` with `@hydra.main`.
- Use `instantiate(cfg.model.forward_model, results_dir=...)`.
- Use `instantiate(cfg.model.prepare, forward_model=forward_model)`.
- Verify `model@model=pylbm`, `model@model=pyudales`, and
  `model@model=pypalm` composition.
- Keep `scripts/config.py` untouched during this phase.

### Phase 3: Migrate Ensemble And Observation Paths

- Migrate `run_ensemble_forward_model.py`.
- Move failure-policy setup to `configure_failure_policy`.
- Migrate observation creation to `create_observation_operator(obs_cfg,
  solver_name)`.
- Add tests for observation operator composition and ensemble failure config.

### Phase 4: Migrate ESMDA Scripts

Recommended order:

1. `run_parameter_esmda.py`
2. `run_state_and_parameter_esmda.py`
3. `run_rollout_esmda.py`
4. `run_time_varying_parameter_esmda.py`
5. `run_time_varying_parameters_rollout_esmda.py`

For each script:

- Use `truth_model` and `assim_model` config packages.
- Instantiate selected smoother from `cfg.esmda.smoother`.
- Replace CLI tuning flags with Hydra overrides.
- Keep dynamic run-time values in Python: observations, `C_D`, RNG splits,
  `time_coords`, and result datasets.

### Phase 5: Remove The Old Global Config

- Once all scripts are migrated, delete or deprecate `scripts/config.py` and
  `scripts/config_small.py`.
- Update `docs/codebase_guide.md` and `README.md`.
- Convert `tests/config.py` into Hydra test presets instead of a parallel
  Python config module (see the **Tests** sub-section below — this is more
  involved than a rename).
- Decide whether `pyurbanair.utils.config_utils` remains as stable public API
  or becomes deprecated.

#### Tests

The current test suite has two patterns that do not survive a naive
migration and need explicit handling — together this is comparable in
scope to one of the script-migration phases.

1. **`sys.modules["scripts.config"] = tests_config` injection.** Used by
   roughly every integration test (e.g. `test_run_state_and_parameter_esmda.py:14`,
   `test_run_time_varying_parameter_esmda.py:12`, `test_spinup.py:11`).
   After migration, scripts no longer import `scripts.config` at all, so
   the injection has no effect. Tests instead compose a config and call
   the script's `run(cfg)` directly:

   ```python
   from hydra import compose, initialize
   from scripts.run_parameter_esmda import run

   with initialize(version_base=None, config_path="../conf"):
       cfg = compose(
           config_name="config",
           overrides=[
               "+preset=test",
               "ensemble.ensemble_size=2",
               "esmda.num_steps=1",
           ],
       )
       run(cfg)
   ```

   This is the second motivation for the `main`/`run` split called out in
   the "Model Configs And `instantiate`" section. Without it, the test
   suite breaks.

2. **In-place dict mutation between subtests.** E.g.
   `test_run_state_and_parameter_esmda.py:95-100`:

   ```python
   tests_config.ENSEMBLE["ensemble_size"] = ensemble_size
   tests_config.ESMDA["num_steps"] = 1
   ```

   Wrapped in a `try/finally` that restores the originals. Under Hydra,
   this becomes a fresh `compose(...)` with `overrides=[...]` per
   parametrization, so the `try/finally` save-and-restore dance goes
   away — each subtest gets an immutable `DictConfig`. Audit each test
   for the mutation pattern; some currently mutate **and** read back
   (e.g. `original_ensemble = tests_config.ENSEMBLE.copy()`), so the
   replacement is mechanical but not zero-thought.

3. **A `conf/preset/test.yaml`** mirroring today's `tests/config.py`:
   tiny domain, `ensemble_size=4`, `simulation_time=5.0`,
   `cuda=false`, etc. This is what `+preset=test` resolves to.

4. **Fixtures.** Add a `compose_test_cfg(overrides: list[str])` pytest
   helper in `tests/conftest.py` so individual tests stay short.

## Risks And Mitigations

- **Hydra working directory changes:** set `hydra.job.chdir: false` and keep
  paths relative to repo root.
- **Output-dir double-creation:** with `chdir: false` plus scripts writing to
  their own `BASE_RESULTS_DIR`, Hydra's auto `run.dir` is created and left
  empty. Resolve by adopting `HydraConfig.get().runtime.output_dir` as the
  script's `out_dir` (preferred) — see the recommendation under "Proposed
  Config Layout".
- **PALM eager imports:** keep `pypalm.*` targets only in `model/pypalm.yaml`
  and instantiate only the selected model package. Add a regression test
  asserting `"pypalm" not in sys.modules` after composing a non-PALM config.
- **Config interpolation across aliased model packages:** use
  `model@truth_model` and `model@assim_model`; avoid hard-coding `${model...}`
  inside configs that must work under multiple package names. Prefer
  sibling-relative refs (`${.foo}`, `${..foo}`) inside a single file.
- **Model-conditional parameter fields:** `pressure_gradient_magnitude` is
  uDALES-only today. The Hydra schema lists it for all models; the helpers
  must keep the `model_name == "pyudales"` filter. Add a unit test that
  composes `model@model=pylbm` and asserts the field is absent from the
  resulting true-params dataset.
- **Inverse-crime preservation:** `truth_method_kwargs` and `method_kwargs`
  must remain independent under interpolation. Do not let a clever default
  collapse them. Add a unit test that the two correlation lengths are
  distinct in the default config.
- **Python tuples in YAML:** use lists in YAML and normalize to tuples in helper
  functions only where constructors require tuples (e.g. `domain.bounds`).
- **JAX arrays and xarray datasets are not YAML data:** build them in helper
  functions from scalar/list config.
- **Incremental migration churn:** migrate one script at a time while leaving
  `scripts/config.py` available for scripts that have not moved yet. The
  helpers under `pyurbanair.config.hydra_helpers` must work without
  importing `scripts.config` so a half-migrated tree builds cleanly.
- **Test-suite collateral damage:** the `sys.modules["scripts.config"]`
  injection pattern means every integration test breaks the moment its
  target script is migrated. Migrate the script and its test in the same
  PR, not separately. Use `resolve_output_dir(...)` so direct `run(cfg)`
  tests are not coupled to Hydra's CLI runtime singleton.

## Acceptance Criteria

- Existing default values from `scripts/config.py` are represented in Hydra
  config files.
- A user can switch solver backends from the command line without editing code.
- Forward model, ensemble model, and ESMDA smoother construction use
  `hydra.utils.instantiate`.
- PALM is still imported only when a PALM config is selected and instantiated
  (asserted by a `"pypalm" not in sys.modules` regression test).
- ESMDA run scripts no longer mutate imported config dictionaries for sweeps.
- Run scripts expose a `def run(cfg: DictConfig)` entry point so tests can
  invoke them without going through `@hydra.main`.
- The test suite no longer relies on `sys.modules["scripts.config"]` patching
  or in-place mutation of imported config dicts.
- Small/fast test settings are expressed as Hydra presets
  (`conf/preset/small.yaml`, `conf/preset/test.yaml`).
- `pressure_gradient_magnitude` continues to be absent from true-params and
  time-varying true-params datasets when `model_name != "pyudales"`.
- `pressure_gradient_magnitude` remains absent from parameter-ensemble
  datasets for all models unless a separate follow-up intentionally changes
  that behavior.
- The truth and assimilation correlation lengths used by
  `create_time_varying_true_params` remain distinct in the default config
  (anti-inverse-crime invariant).

## Phase 1 Implementation Review (2026-05-18)

Review of the first round of Phase 1/2 work against the plan, against the
current source, and against `pytest tests/test_hydra_config.py`.

### What works (verified)

- **`conf/` tree matches the plan.** All 30 files in the right places,
  including `params/external/default.yaml`, the `mode: points|grid`
  discriminator on obs configs, and per-model `init_subdir`.
- **`pyurbanair/config/hydra_helpers.py`** covers every helper the plan
  lists, including `load_init_conditions_for_esmda` (with the new
  `init_subdir` parameter), `create_initial_state_ensemble`, and
  `resolve_output_dir` for the Hydra-runtime-vs-fallback split.
- **`scripts/run_forward_model.py`** is properly split into
  `def run(cfg)` + `@hydra.main`-wrapped `main`, so it is importable from
  tests.
- **Hydra config tests** (`tests/test_hydra_config.py`): 8/10 pass,
  including the load-bearing ones:
  - PALM laziness: `assert "pypalm" not in sys.modules` after composing
    `model=pylbm`.
  - `pressure_gradient_magnitude` filtered out for non-uDALES.
  - Truth and prior correlation lengths distinct in defaults.
  - Interpolations resolve correctly under
    `model@truth_model`/`model@assim_model` aliasing, including
    `esmda.alpha` / `smoother.num_steps` propagation when
    `esmda.num_steps=4` is overridden.
- **`hydra-core>=1.3,<2`** added to both `[project.dependencies]` and
  `[tool.pixi.dependencies]` in `pyproject.toml`.

### Confirmed bugs

#### 1. Preset composition is broken — 2 tests fail

`tests/test_hydra_config.py::test_presets_compose_with_model_overrides`
fails for both `small` and `test`. Composing `+preset=small` produces an
impossible mongrel domain:

```text
domain: {'nx': 60, 'ny': 40, 'nz': 4, 'bounds': [[0.0, 60.0], [0.0, 40.0], [0.0, 16.0]]}
```

`nx=60` is from `xie_castro_60x40x16`, `nz=4` is from `small`, and
`bounds` is neither file's values — it is some merged junk. The
`override /domain: small` line in `conf/preset/small.yaml:3` is not
actually replacing the domain group selection.

Root cause: the preset combines `# @package _global_` with
`defaults: - override /domain: small` and is added via `+preset=...`.
Hydra 1.3 does not reliably let a CLI-added preset retroactively override
a parent's already-selected group via this combo.

Fix options:

- (preferred) Declare `preset` as a known group in `conf/config.yaml`'s
  defaults with `preset: _self_` (or a null sentinel), then invoke as
  `preset=small` (no `+`); presets stop needing `# @package _global_`.
- Or split each preset into two files — a defaults-only file for group
  overrides and a `_global_` field-overlay file.

This must be fixed before any ESMDA scripts are migrated, because every
test that uses `+preset=test` will silently get a half-overridden config.

#### 2. Pixi env may not yet include hydra-core

`pixi run pytest tests/test_hydra_config.py` errors with
`ModuleNotFoundError: No module named 'hydra'`. The package is in
`pixi.lock` but the active env was not rebuilt. Run `pixi install`
(or call `.pixi/envs/default/bin/python -m pytest …`, which works).

### Things to look at

- **`pylbm_cpu` as a separate model config.** `conf/model/pylbm_cpu.yaml`
  is a near-duplicate of `conf/model/pylbm.yaml` with only `cuda: false`.
  It is referenced from `conf/preset/test.yaml` and from
  `tests/test_run_forward_model.py` as `model=pylbm_cpu`. This solves the
  immediate CPU-testing need but creates a maintenance burden — any
  change to `pylbm.yaml` must be mirrored. Cleaner options:
  - Just override `model.forward_model.cuda=false` at the call site and
    delete `pylbm_cpu.yaml`.
  - Or make `pylbm_cpu.yaml` use `defaults: [pylbm]` and only set
    `forward_model.cuda: false`, so it is a single-line variant.

- **`prepare_palm` and `prepare_lbm` are byte-identical.**
  `src/pyurbanair/config/hydra_helpers.py:37-42` — both just call
  `_unwrap_forward_model(forward_model).compile(compile=compile)`. Could
  collapse into one `prepare_compile` target used by both
  `pylbm.yaml` and `pypalm.yaml` `prepare` blocks. Minor.

- **`_convert_: all` interacts with tuple-required constructors.** Both
  LBM (`libs/pylbm/src/pylbm/forward_model.py:44`) and uDALES
  (`libs/pyudales/src/pyudales/forward_model.py:124`) declare `bounds` as
  a tuple. With `_convert_: all`, OmegaConf converts ListConfig → plain
  list (still not tuple). The constructors do not validate types (they
  just index into `bounds`), so this works today — but if anyone adds
  `isinstance(bounds, tuple)` validation later, the YAML side will
  silently break. Add a normalization at the boundary if you want to be
  safe, or leave a comment.

- **`tests/test_run_forward_model.py` failures are pre-existing, not a
  migration regression.** All 4 cases fail with environment errors (LBM
  compile exit 2 because the test ran in the default env that lacks
  `netcdf.mod`; uDALES `write_inputs.sh` CalledProcessError). The Hydra
  wiring itself reaches the prepare step cleanly — `instantiate`
  resolves, the forward_model is constructed, and `prepare_lbm` runs.
  These would also fail on `main`. Worth fixing the test-runner env
  separately.

- **Forward-script body uses `cfg.run.get("results_dir")`** rather than
  direct attribute access (`scripts/run_forward_model.py:24-28`). Since
  `results_dir: null` is declared in `conf/config.yaml:26`,
  `cfg.run.results_dir is not None` would be cleaner and benefits from
  struct-mode key validation. Tiny.

### Summary

The migration scaffolding is solid and most of the load-bearing
invariants from the plan (PALM laziness, pressure-gradient filter,
distinct correlation lengths, sibling-relative interpolation under
aliased packages) are correctly implemented and tested. The one
substantive bug is the preset composition — it needs fixing now because
every subsequent migrated script and test will lean on `+preset=test`
for fast unit-test runs, and right now those tests would silently get
the wrong domain/time/ensemble. The other items are cleanup, not
blockers.

## Current Migration Status (2026-05-18)

Phase 1 is implemented and the first Phase 2 script has been migrated.
The branch `hydra-config` contains commit `5abd843`
(`Add Hydra config scaffold`) with the Hydra config tree, the new
`pyurbanair.config.hydra_helpers` module, composition tests, and
`scripts/run_forward_model.py` converted to `run(cfg)` plus `@hydra.main`.

The Phase 1 review fixes have also been applied locally:

- `preset` is now a declared config group in `conf/config.yaml` and should be
  selected as `preset=test` or `preset=small`, not `+preset=test`.
- `conf/preset/small.yaml` and `conf/preset/test.yaml` are explicit
  root-level overlays. This avoids Hydra 1.3's surprising partial merges when
  a CLI-selected preset tries to override already-selected groups.
- The duplicate `conf/model/pylbm_cpu.yaml` variant was removed; tests use
  ordinary overrides such as `model.forward_model.cuda=false`.
- `prepare_compile` now handles the shared LBM/PALM compile preparation.
- `tests/conftest.py` provides `compose_test_cfg(...)` for future migrated
  script tests.
- `params."true"` is quoted in presets so YAML does not parse it as a boolean
  key.

Verified after these fixes:

```bash
pixi run -e dev pytest tests/test_hydra_config.py -q
pixi run -e dev python -m py_compile scripts/run_forward_model.py tests/conftest.py tests/test_run_forward_model.py src/pyurbanair/config/hydra_helpers.py tests/test_hydra_config.py
pixi run -e dev python scripts/run_forward_model.py --cfg job preset=test model=pyudales run.skip_viz=true
```

The next implementation step is Phase 3: migrate
`scripts/run_ensemble_forward_model.py`, wire ensemble construction through
`hydra.utils.instantiate`, and use `configure_failure_policy` plus the new
observation helpers where needed. The remaining ESMDA scripts still use
`scripts.config` and should be migrated one at a time after the ensemble path
is stable.

## Current Migration Status Update (2026-05-18)

Phase 3 is now partially implemented and Phase 4 has started.

Additional scripts migrated to Hydra:

- `scripts/run_ensemble_forward_model.py`
- `scripts/run_parameter_esmda.py`
- `scripts/run_state_and_parameter_esmda.py`

Their corresponding tests were updated to compose Hydra configs through
`compose_test_cfg(...)` instead of patching `sys.modules["scripts.config"]`
or mutating `tests.config` dictionaries. The migrated scripts no longer import
`scripts.config` or use `argparse`; they expose `run(cfg: DictConfig)` plus a
thin `@hydra.main` wrapper.

Notable wiring added in this pass:

- ensemble forward models are created with `hydra.utils.instantiate`;
- ensemble failure policy is configured from `cfg.ensemble.failure`;
- parameter and state+parameter ESMDA smoothers are instantiated from
  `cfg.esmda.smoother`;
- `esmda.use_init_conditions` was added for state+parameter ESMDA init-condition
  loading;
- init-condition loading now receives the selected model's YAML
  `init_subdir`;
- observation helper and failure-policy helper behavior has focused unit
  coverage in `tests/test_hydra_config.py`.

Verified after this pass:

```bash
pixi run -e dev pytest tests/test_hydra_config.py -q
pixi run -e dev python -m py_compile scripts/run_ensemble_forward_model.py scripts/run_parameter_esmda.py scripts/run_state_and_parameter_esmda.py tests/test_run_ensemble_forward_model.py tests/test_run_parameter_esmda.py tests/test_run_state_and_parameter_esmda.py tests/test_hydra_config.py
```

The focused Hydra tests pass (`13 passed`). Full solver-running tests were not
run because they enter compile/preprocessing paths and remain
environment-sensitive.

Next recommended steps:

1. Migrate `scripts/run_rollout_esmda.py`.
2. Migrate `scripts/run_time_varying_parameter_esmda.py`.
3. Migrate `scripts/run_time_varying_parameters_rollout_esmda.py`.
4. Continue converting each migrated script's tests away from
   `scripts.config` injection as part of the same script migration.

## Phase 3 / Early Phase 4 Implementation Review (2026-05-18)

Review of the four uncommitted scripts/tests + `conf/config.yaml` added in
this pass:

- `scripts/run_ensemble_forward_model.py`
- `scripts/run_parameter_esmda.py`
- `scripts/run_state_and_parameter_esmda.py`
- `tests/test_run_ensemble_forward_model.py`
- `tests/test_run_parameter_esmda.py`
- `tests/test_run_state_and_parameter_esmda.py`
- `tests/test_hydra_config.py` (two new helper tests)
- `conf/esmda/*.yaml` (added `use_init_conditions: false`)

All 13 focused Hydra tests in `tests/test_hydra_config.py` pass.

### What's good

- **All three scripts cleanly drop `argparse` + `scripts.config`** and use
  the `run(cfg)` + `@hydra.main` split. RNG ordering is preserved exactly
  (split-then-pass-residual-key-to-smoother).
- **Smoother instantiation via `instantiate(cfg.esmda.smoother, …, rng_key=rng_key)`**
  is the right shape — declarative class, runtime values at call time.
- **`tests/conftest.py::compose_test_cfg`** is a small, reusable fixture;
  tests are visibly shorter.
- **`cfg.truth_model.init_subdir` / `cfg.assim_model.init_subdir`** plumbing
  in `scripts/run_state_and_parameter_esmda.py:44-78` cleanly replaces the
  hardcoded `lbm/udales` branch.
- **New unit tests** for `configure_failure_policy` and
  `create_observation_operator` exercise the helpers without any solver —
  these will keep migration regressions visible.

### Bugs to fix

#### 1. `out_dir.mkdir` happens after the assim model is told to write inside it

`scripts/run_state_and_parameter_esmda.py:106-111`:

```python
out_dir = resolve_output_dir(cfg, "state_and_parameter_esmda")
assim_results_dir = out_dir / "assim_states"
assim_model = instantiate(
    cfg.assim_model.forward_model,
    results_dir=assim_results_dir,
)
```

`out_dir.mkdir(parents=True, exist_ok=True)` does not run until line 167 —
after both the forward model construction *and* the ESMDA loop have used
`assim_results_dir`. In the legacy code this worked because
`config.BASE_RESULTS_DIR` was already present and `set_results_dir(None)`
is tolerant. With the Hydra-fallback path used by tests (no
`HydraConfig.initialized()`, `out_dir = .temp/scripts/state_and_parameter_esmda`),
the directory may not exist when the model is constructed.

Move `out_dir.mkdir(parents=True, exist_ok=True)` (and an explicit
`assim_results_dir.mkdir(parents=True, exist_ok=True)`) to just after
they are computed on line 107.

#### 2. `tests/test_run_state_and_parameter_esmda.py` is half-migrated — fixture and runner build forward models from *different* configs

`tests/test_run_state_and_parameter_esmda.py:7` still does
`import tests.config as tests_config`, and `_write_init_conditions`
(`tests/test_run_state_and_parameter_esmda.py:22-47`) calls
`tests_config.create_forward_model(model_name)` to build the
init-condition writer.

The crucial difference vs. the other migrated tests: this file removed
the `sys.modules["scripts.config"] = tests_config` injection at the top,
but **kept the `tests_config.create_forward_model(...)` call**. Those
helpers live in `src/pyurbanair/utils/config_utils.py:33`, which does
`from scripts import config` (lazy, still present). Without the
injection, the fixture writer now resolves to the **production**
`scripts/config.py` (nx=100, ny=80, nz=16, sim_time=180s, cuda=true, …),
while the `run(cfg)` under test uses `preset=test` (nx=40, ny=40, nz=4,
sim_time=5s, cuda=false).

If this test actually runs end-to-end it will either (a) take many
minutes building production-resolution init conditions, then (b) fail
with a shape mismatch against the small assim model. The reason this
has not surfaced yet is that the test requires a solver and is not part
of `tests/test_hydra_config.py`'s 13.

Two options, in order of preference:

- **Port the fixture builder to Hydra.** `_write_init_conditions(model_name, ...)`
  should compose a config the same way the runner does and instantiate
  the forward model from `cfg.truth_model.forward_model` (or
  `cfg.model.forward_model`) via `hydra.utils.instantiate`. This deletes
  the last `tests.config` reference and matches the Phase 5 acceptance
  criterion.
- Restore the injection line (`sys.modules["scripts.config"] = tests_config`).
  Quick fix, but it carries the legacy pattern into the migrated branch,
  which is the exact pattern the migration is trying to retire.

#### 3. `results_dir=…` is now passed to the ensemble constructor too — new behavior

`scripts/run_ensemble_forward_model.py:18-23`,
`scripts/run_parameter_esmda.py:62-66`, and
`scripts/run_state_and_parameter_esmda.py:137-141` all pass
`results_dir=…` to `instantiate(cfg.*.ensemble_model, …)`. The legacy
`pyurbanair.utils.config_utils.create_ensemble_forward_model` did **not**
pass `results_dir` to the ensemble constructor — it only passed it when
constructing the inner forward model.

Both `LBMEnsembleForwardModel.__init__` and
`UDALESEnsembleForwardModel.__init__` accept `results_dir`, so this does
not throw — but how it interacts with the inner forward model's own
`results_dir` is library-specific and constitutes a behavior change.
Either confirm this is desired (and document why), or roll it back to
keep the migration behaviorally neutral.

### Smaller cleanups

- **`cfg.run.init_conditions` is becoming a grab-bag namespace.**
  `conf/config.yaml:24-28` now holds `skip_viz`, `rollout`, `results_dir`,
  `init_conditions`. The first three are reasonably script-agnostic;
  `init_conditions` is a state+param-ESMDA knob that arguably belongs
  under `esmda.use_init_conditions` (next to the existing
  `esmda.init_conditions_dir`). Not blocking, but the asymmetry will get
  worse as more scripts migrate.

- **`obs_x.tolist() == [5.0, 35.0, 5.0, 35.0]` in
  `tests/test_hydra_config.py`** locks in the meshgrid flatten order.
  Correct for the current implementation, but a shape assertion plus a
  `sorted()` equality on the set would survive innocuous reshuffles.

- **`preset/test.yaml` sets `temporal_mode: mean` AND
  `aggregation_mode: mean`.** `aggregation_mode` is unused when
  `temporal_mode == mean`. Harmless noise.

- **The plan now has two adjacent sections both dated 2026-05-18**
  ("Phase 1 Implementation Review" and the "Current Migration Status"
  pair). The first review section is largely superseded. Worth either
  collapsing or marking the Phase 1 review as "Resolved" so future
  readers do not act on the broken-preset note that has already been
  fixed.

### Summary

This pass is mostly clean and the focused tests confirm the wiring
works. Two real items need attention before the next pass:

1. Move `out_dir.mkdir` before the forward-model instantiation in
   `run_state_and_parameter_esmda.py`.
2. The state+param test's fixture writer is silently building at
   production resolution because it never finished migrating — port it
   to Hydra-composed instantiation rather than
   `tests_config.create_forward_model`.

The `results_dir=` passed to ensemble constructors is a scope expansion
worth either confirming or rolling back.

### Resolution Notes

The review comments above have been addressed:

- `state_and_parameter_esmda` now creates both `out_dir` and
  `assim_results_dir` before constructing the assimilation forward model.
- `tests/test_run_state_and_parameter_esmda.py` no longer imports or calls
  `tests.config`; its init-condition writer composes the same Hydra test
  config used by the runner and instantiates the forward model through
  `hydra.utils.instantiate`.
- `results_dir` is no longer passed explicitly to ensemble constructors; the
  legacy behavior is preserved by letting `BaseEnsembleForwardModel` inherit
  the inner forward model's `results_dir`.
- the script-specific init-condition toggle moved from `run.init_conditions`
  to `esmda.use_init_conditions`.
- the observation helper test no longer depends on meshgrid flatten order.
- `conf/preset/test.yaml` no longer sets the unused
  `obs.aggregation_mode` value for `temporal_mode: mean`.

The earlier Phase 1 review section is retained as historical context; its
broken-preset comments are resolved by the `preset=small|test` overlay
approach described in the current status notes.

## Test Checkpoint (2026-05-18)

The migrated Hydra-related test set has now been run successfully:

```bash
pixi run -e dev pytest tests/test_hydra_config.py tests/test_run_forward_model.py tests/test_run_ensemble_forward_model.py tests/test_run_parameter_esmda.py tests/test_run_state_and_parameter_esmda.py -q
```

Result: all selected tests pass. This covers the Hydra composition helpers,
the migrated forward/ensemble scripts, and the migrated parameter and
state+parameter ESMDA scripts, including init-condition loading through the
Hydra-composed test fixture.

Next implementation work remains:

1. Migrate `scripts/run_rollout_esmda.py`.
2. Migrate the time-varying ESMDA scripts.
3. Continue removing `scripts.config` injection from each script's tests as
   that script is migrated.

## Spinup-Only Cleanup (2026-05-18)

Decisions accepted from the review of the last pass:

- **AR(2) is the truth default.** The new
  `create_time_varying_true_params` deliberately uses `truth_method` /
  `truth_method_kwargs` to build a full time-series model (AR(2) by
  default) for the truth trajectory — the legacy `sample_smooth_ensemble`
  GP behavior is gone. The anti-inverse-crime invariant is preserved by
  keeping `truth_method_kwargs` distinct from `method_kwargs`.
- **Pre-specified init conditions are no longer supported.** The
  spin-up flow (one warm-up call on the assim model before the ESMDA
  loop) is now the only entry point. This deletes a substantial chunk
  of branching from the rollout and state+param scripts.
- **Rollout ESMDA no longer builds two assim forward models.** The
  `assim_init_model` was a legacy artifact of the now-deleted
  `create_rollout_forward_model` identity wrapper; the script now reuses
  the single `assim_model` for both the spin-up call and the per-window
  loop.
- **Shared `prior_model` block is hoisted into
  `conf/time_varying/_prior_model.yaml`.** Each of the four method files
  selects it via `defaults: [_prior_model]`.

What the spin-up cleanup deletes:

- `conf/esmda/*.yaml`: `use_init_conditions`, `init_conditions_dir`,
  `true_sim_id` (all five files).
- `conf/model/*.yaml`: `init_subdir` (all three files).
- `src/pyurbanair/config/hydra_helpers.py`:
  `load_init_conditions_for_esmda`, `clean_restarts`, and the
  `clean_lbm_restart_files` import. Update the **Helper Module** section
  to drop these — they were carried over from the legacy code only to
  serve the pre-specified-init-conditions path.
- `scripts/run_rollout_esmda.py`: the entire `if use_init_conditions:`
  branch, the `assim_init_model` block, and the imports of
  `load_init_conditions_for_esmda`.
- `scripts/run_state_and_parameter_esmda.py`: the entire
  `if use_init_conditions:` branch, the `clean_restarts` import and
  call, the imports of `load_init_conditions_for_esmda`.
- `tests/test_run_rollout_esmda.py` and
  `tests/test_run_state_and_parameter_esmda.py`: the `_write_init_conditions`
  and `_build_params_ensemble` fixture helpers, the `init_conditions_root`
  scaffolding, and the per-test
  `esmda.use_init_conditions=true` / `esmda.init_conditions_dir=...`
  overrides. The tests now only cover the spin-up path.
- `tests/test_hydra_config.py::test_single_model_configs_compose`: the
  `init_subdir` assertion (model field no longer exists).

Pre-existing legacy modules (`scripts/config.py`, `scripts/config_small.py`,
`scripts/simulate_init_conditions.py`, `tests/config.py`,
`src/pyurbanair/utils/config_utils.py`) still reference the dead
`init_conditions_dir` / `true_sim_id` keys and the old
`load_init_conditions_for_esmda` helper. They are scheduled for Phase 5
removal — leaving them alone for now keeps a half-migrated tree compiling.

New tests added:

- `test_create_time_varying_true_params_uses_ar2_relaxation_truth`: pins
  the schema (only `time` dim, no `ensemble`; `inflow_angle` and
  `velocity_magnitude` time series of length `num_time_points`) and
  proves the truth trajectory is driven by `truth_method` — flipping
  `time_varying.method` while holding `truth_method=ar2_relaxation`
  yields byte-identical truth samples.
- `test_create_time_varying_true_params_includes_pressure_gradient_for_pyudales`:
  model-conditional filter for `pressure_gradient_magnitude`.

Verified after this pass:

```bash
.pixi/envs/default/bin/python -m pytest tests/test_hydra_config.py -q
# 15 passed
```

Acceptance-criteria updates:

- The "pre-specified init-conditions" behavior is **no longer** a
  preserved invariant. The Phase 5 follow-up to delete
  `scripts/simulate_init_conditions.py` and the legacy
  `load_init_conditions_for_esmda` in `pyurbanair.utils.config_utils`
  is now strictly cleanup, not behavior preservation.
- The "rollout ESMDA constructs exactly one assim forward model" is
  a new behavioral invariant from this pass.

Remaining work (unchanged):

1. Migrate `scripts/run_time_varying_parameters_rollout_esmda.py`.
2. Continue removing `scripts.config` injection from each script's
   tests as that script is migrated.
3. Phase 5: remove `scripts/config.py`, `scripts/config_small.py`,
   `tests/config.py`, and the legacy `config_utils.py` once all scripts
   are off them.

## Phase 4 (Rollout + Time-Varying) Implementation Review (2026-05-18)

Review of the currently uncommitted changes on `hydra-config`: the four
remaining ESMDA scripts and their tests, the spin-up-only cleanup in
`hydra_helpers.py`, the `_prior_model.yaml` hoist, and the removal of
`init_*` keys / `init_subdir` from `conf/esmda/*.yaml` and
`conf/model/*.yaml`. Verified `pytest tests/test_hydra_config.py -q`:
15 passed.

### What's good

- **`run_time_varying_parameters_rollout_esmda.py` and
  `run_rollout_esmda.py`** both cleanly dropped `argparse` +
  `scripts.config`, use `instantiate(cfg.*.forward_model)`,
  `instantiate(cfg.*.ensemble_model, forward_model=…)`, and
  `configure_failure_policy(...)`. RNG ordering matches the prior
  scripts.
- **Tests** for the four migrated scripts now compose Hydra configs via
  `compose_test_cfg(...)` and call `run(cfg)` directly. The legacy
  `tests_config` injection plus `try/finally` mutation dance is gone.
  The state+param test in particular is now significantly shorter than
  the pre-spin-up-only version.
- **`_prior_model.yaml` hoist** removes ~12 lines × 4 files of
  duplication. The four `time_varying/*.yaml` files now diff only on
  the values that actually matter (method, kwargs, truth_method,
  truth_method_kwargs).
- **`prepare_compile`** was the right collapse for the identical
  `prepare_lbm` / `prepare_palm` helpers from the Phase 1 review.
  `prepare_lbm` and `prepare_palm` remain as thin pass-throughs so the
  per-model YAML keeps a semantic name — fine.
- **New `create_time_varying_true_params` tests** pin the schema (only
  `time` dim, no `ensemble`), the `truth_method`-drives-truth
  invariant, and the `pressure_gradient_magnitude` model-conditional
  filter.

### Confirmed bugs

#### 1. `run_rollout_esmda.py` runs spin-up before the truth forecast, with no `clean_outputs` between them

[scripts/run_rollout_esmda.py:38-61](../../scripts/run_rollout_esmda.py#L38-L61):
the assim model is instantiated, prepared, and immediately invoked to
produce `assim_ref_state`. Then `truth_model(params=true_params, state=None)`
runs. Compared with [scripts/run_state_and_parameter_esmda.py:62-71](../../scripts/run_state_and_parameter_esmda.py#L62-L71)
the rollout script is missing both the `clean_outputs(...)` calls
*and* the `set_results_dir(None)` / restore dance around the spin-up.

Today this happens to work because the rollout script never passes a
`results_dir` (line 38: `assim_model = instantiate(cfg.assim_model.forward_model)`,
no kwarg), so the assim model is in-memory and nothing lands on disk
to leak. But:

- If anyone follows the state+param pattern and adds `results_dir=...`
  to make rollout writes resumable, the warmup state will silently
  pollute window 0's outputs.
- pylbm spin-up *does* leave per-iteration NetCDF dumps and
  `seed_*.dat` warm-start files under `dirs/output/...` even in
  "in-memory" mode — that is exactly what `clean_outputs` is there
  for. Without it, warm-start determinism between truth and assim
  rollouts on the same backend (the `pylbm_pylbm` test case) is not
  guaranteed.

Either add a `clean_outputs(cfg.assim_model.name, assim_model)` after
the spin-up call (cheap and matches the state+param script), or
document explicitly in the script why the rollout flavor doesn't need
it.

#### 2. `run_time_varying_parameters_rollout_esmda.py` builds two parallel "external priors" representations that differ in shape

[scripts/run_time_varying_parameters_rollout_esmda.py:210-221](../../scripts/run_time_varying_parameters_rollout_esmda.py#L210-L221)
calls `OmegaConf.to_container(cfg.params.external, resolve=True)` and
hands the result to `build_parameter_time_series(external_priors=…)`
for the truth model. Meanwhile, the *prior* `ts_model` is built by
`instantiate(cfg.time_varying.prior_model)`, which routes through the
hydra-helper-built `external_priors`. The helper version
([src/pyurbanair/config/hydra_helpers.py:126-129](../../src/pyurbanair/config/hydra_helpers.py#L126-L129))
returns only `inflow_angle` and `velocity_magnitude`:

```python
def _prior_cfg_as_external_priors(prior_cfg: Any) -> dict[str, dict[str, float]]:
    prior = _plain(prior_cfg)
    names = ("inflow_angle", "velocity_magnitude")
    return {name: dict(prior[name]) for name in names}
```

…but the script's inline `OmegaConf.to_container(cfg.params.external,
resolve=True)` returns **every** key in `params/external/default.yaml`
(today that's the same set, so it doesn't crash — yet). Two paths,
two shapes, only one of them respects a curated allowlist.

Two fixes, in order of preference:

- Replace the inline `OmegaConf.to_container(...)` with the existing
  helper. Better still, add `build_truth_ts_model(tv_cfg, prior_cfg,
  ensemble_size=1)` to `hydra_helpers.py` so the script also stops
  duplicating the `build_parameter_time_series(method=tv["truth_method"],
  ...)` boilerplate from `create_time_varying_true_params`. That makes
  the rollout script's two model constructions symmetrical.
- Or accept that `external_priors` should be "all of `params.external`"
  and rip the allowlist out of `_prior_cfg_as_external_priors`.

Pick one, but don't leave both code paths.

#### 3. `_prior_model.yaml` carries a redundant package directive

[conf/time_varying/_prior_model.yaml:1](../../conf/time_varying/_prior_model.yaml#L1):

```yaml
# @package time_varying
prior_model:
  _target_: ...
```

Hydra already infers the package as `time_varying` from the file's
location under `conf/time_varying/`. The `# @package time_varying`
header is therefore a no-op. The sibling files (`ar2_relaxation.yaml`,
`ar1.yaml`, etc.) don't have it. Either delete the comment for
consistency, or convert it to `# @package _group_` if you want to be
explicit about not depending on directory inference.

### Things to look at

- **Spin-up state ensemble has zero variance across members.**
  [scripts/run_rollout_esmda.py:47-50](../../scripts/run_rollout_esmda.py#L47-L50)
  builds the initial state ensemble by replicating one warm-up
  trajectory `ensemble_size` times via
  `create_initial_state_ensemble(...)`. The parameter ensemble has
  prior spread, but every state copy is byte-identical at t=0. This is
  the same behavior the pre-migration spin-up path had, so it isn't a
  regression — but it is now the only entry point (the
  pre-specified-init-conditions path that used to provide a true state
  ensemble is gone). Worth a one-line explanation in the script that
  this is intentional (the param spread is what generates trajectory
  divergence on the first window).

- **`run_time_varying_parameter_esmda.py` mixes two output-dir sources.**
  Line 241–243 honors `cfg.run.results_dir` for the on-disk assim
  states, while line 286 routes the final results via
  `resolve_output_dir(cfg, "time_varying_parameter_esmda")` (Hydra's
  per-run dir, falling back to `cfg.paths.base_results_dir /
  "time_varying_parameter_esmda"`). The plan's "pick one output-dir
  source of truth" recommendation under "Proposed Config Layout"
  applies here too — the assim states could just go to `out_dir /
  "assim_states"`, mirroring the state+param script. Today the
  `*_on_disk` test cases work, but a CLI user running with
  `paths.base_results_dir=…` ends up with assim states under the
  pytest `tmp_path` while the rest of the run lands somewhere else.

- **`conf/config.yaml::run.rollout` is dead.** Grepping shows it is
  only printed in
  [scripts/run_forward_model.py:40,45](../../scripts/run_forward_model.py#L40)
  and [scripts/run_ensemble_forward_model.py:57,63](../../scripts/run_ensemble_forward_model.py#L57)
  to choose a label string. The earlier "rollout" mode of these
  scripts ran the explicit `create_rollout_forward_model` wrapper
  that the codebase guide notes is now an identity function. The flag
  no longer does anything. Either wire it back up or remove the
  field. Carrying a config key that does nothing trains users to
  ignore the schema.

- **`_prior_cfg_as_external_priors` hardcodes the two parameter
  names.** If someone follows the "Add a new parameter" recipe in the
  codebase guide, the new param will be silently dropped from
  `external_priors` (and from the prior ensemble). Either iterate over
  the dict keys, or add an assertion that surfaces the omission
  loudly. This is a slow-leak bug — by the time it bites, the cause
  will be far away.

- **Six unmigrated tests still inject `tests.config`** via
  `sys.modules["scripts.config"] = tests_config`:
  `test_run_ensemble_rollout_forward_model.py`, `test_spinup.py`,
  `test_time_varying_forward_model.py`,
  `test_run_rollout_forward_model.py`,
  `test_varying_inflow_velocity.py`,
  `test_inflow_angle_simulation_sign.py`. They target scripts
  (`run_rollout_forward_model.py`,
  `run_ensemble_rollout_forward_model.py`) that have **not** been
  migrated yet, so the injection is still load-bearing. The migration
  invariant "migrate the script and its test in the same PR" should
  apply to those scripts too — flag this as a Phase 4 finish-up task
  so the asymmetry doesn't accidentally persist into Phase 5.

- **No regression test for the rollout script's
  truth/prior correlation-length distinctness.** The plan's anti-inverse-crime
  invariant has unit coverage in `test_hydra_config.py` for
  `create_time_varying_true_params`, but
  `run_time_varying_parameters_rollout_esmda.py` constructs
  `truth_ts_model` inline (issue #2 above). If issue #2 is fixed by
  routing through a helper, the existing test covers it. If not, add
  an explicit assertion that `cfg.time_varying.truth_method_kwargs !=
  cfg.time_varying.method_kwargs` after composition for the rollout
  config path.

- **`run_time_varying_parameters_rollout_esmda.py` still imports
  `build_parameter_time_series` directly.** Once issue #2 is resolved
  via a helper, this import (and the `OmegaConf` import on line 27)
  should go away — the script would only need `instantiate(...)` plus
  the helpers. Cleanup, not a bug.

- **Legacy module references for dead keys.** `scripts/config.py:123`,
  `scripts/config_small.py:116`, and
  `scripts/simulate_init_conditions.py:97` still set/read
  `init_conditions_dir`; `src/pyurbanair/utils/config_utils.py:357`
  still defines `load_init_conditions_for_esmda`. The plan lists
  these for Phase 5, which is correct — but
  `simulate_init_conditions.py` in particular is now an orphan: its
  only purpose was to feed the deleted pre-specified-init-conditions
  ESMDA path. Confirm whether it should be deleted outright rather
  than rewritten, and update the codebase guide section 2 (it lists
  the script).

### Summary

The four script migrations look correct and the tests pass; the
spin-up-only consolidation is a real simplification. Three items to
fix before Phase 5:

1. Add `clean_outputs(...)` after the spin-up in `run_rollout_esmda.py`,
   or document why it doesn't need one (issue #1).
2. Collapse the two `external_priors` paths in
   `run_time_varying_parameters_rollout_esmda.py` onto a single helper
   so the prior and truth ts-models are constructed symmetrically
   (issue #2).
3. Delete the redundant `# @package time_varying` header from
   `_prior_model.yaml` (issue #3).

The remaining items (dead `run.rollout` flag, hardcoded param names in
`_prior_cfg_as_external_priors`, six still-injecting tests, mixed
output-dir source in the time-varying script, missing rollout-script
correlation-length test) are pre-Phase-5 cleanup that should be done
in the same window as those tests' scripts get migrated.

## Review-Driven Cleanup (2026-05-18)

Changes made in response to the review above. The three "before
Phase 5" items and most of the "things to look at" items are now
addressed; what remains is genuinely Phase 5 (full removal of legacy
`scripts/config.py` / `tests/config.py` / `config_utils.py`) and the
finish-up migration of the six unmigrated tests.

### Bugs fixed

- **Issue #1 — spin-up `clean_outputs` in rollout ESMDA.**
  [scripts/run_rollout_esmda.py](../../scripts/run_rollout_esmda.py)
  now calls `clean_outputs(cfg.assim_model.name, assim_model)` after
  the warm-up call so per-iteration NetCDF dumps and `seed_*.dat`
  files do not leak into the per-window loop.
- **Issue #2 — single `build_truth_ts_model` helper.**
  `src/pyurbanair/config/hydra_helpers.py` now exposes
  `build_truth_ts_model(tv_cfg, external_cfg, ensemble_size=1)` and
  both time-varying scripts use it. The legacy
  `_prior_cfg_as_external_priors` (which projected `params.prior` onto
  a hardcoded `inflow_angle` / `velocity_magnitude` allowlist) is gone,
  and the rollout script no longer constructs a parallel
  `OmegaConf.to_container(cfg.params.external, …)` →
  `build_parameter_time_series(...)` block. Truth and prior ts-models
  now share one construction path and one source of external priors
  (`cfg.params.external`). Subsumes issues #2, #7, #9 and #10 from the
  review.
- **Issue #3 — redundant package directive.** Removed
  `# @package time_varying` from
  [conf/time_varying/_prior_model.yaml](../../conf/time_varying/_prior_model.yaml).
  Hydra infers the package from the file's directory; verified by
  re-composing `cfg.time_varying.prior_model` and instantiating.

### Behavior-preserving cleanups

- **Issue #4 — spin-up state ensemble variance.** Added a one-line
  comment in
  [scripts/run_rollout_esmda.py](../../scripts/run_rollout_esmda.py)
  documenting that every member starts from the same warm-up
  trajectory and parameter spread drives divergence on the first
  window.
- **Issue #5 — output-dir source of truth.**
  [scripts/run_time_varying_parameter_esmda.py](../../scripts/run_time_varying_parameter_esmda.py)
  now derives `assim_results_dir = out_dir / "assim_states"`
  (computed once before the assim model is instantiated, with the
  parent `mkdir`s up front), matching `run_state_and_parameter_esmda.py`.
  `cfg.run.results_dir` is no longer read here.
- **Issue #6 — dead `run.rollout` flag.** Removed `run.rollout` from
  `conf/config.yaml`, `scripts/run_forward_model.py`, and
  `scripts/run_ensemble_forward_model.py`. The label-only flag was a
  rotted artifact of the deleted `create_rollout_forward_model`
  identity wrapper. Single-model output subdirs now use
  `{model_name}` (and `{model_name}_ensemble` for the ensemble
  variant) directly.
- **Issue #11 — orphan `simulate_init_conditions.py` deleted.** The
  only consumer of the pre-specified-init-conditions producer was the
  init-conditions path of `run_rollout_esmda.py` /
  `run_state_and_parameter_esmda.py`, both of which were removed in
  the Spinup-Only cleanup. Updated
  [docs/codebase_guide.md](../../docs/codebase_guide.md) section 2
  and the README repository tree / quickstart to drop the script.

### Carried into the test suite

- **`test_hydra_config.py::_build_time_varying_truth`** now passes
  `external_cfg=cfg.params.external` (was `prior_cfg=cfg.params.prior`).
  The AR(2)-truth seed-stability assertion still holds because the
  default config has `truth_method_kwargs.correlation_length=500.0`
  in `ar2_relaxation.yaml`; flipping `time_varying.method` to `ar1`
  while keeping `truth_method=ar2_relaxation` continues to produce
  byte-identical truth samples.
- All 15 focused Hydra tests pass; all seven migrated scripts
  py_compile and expose `run(cfg)`.

### Remaining work

- **Issue #8 — six unmigrated tests still inject `tests.config`.**
  `test_run_ensemble_rollout_forward_model.py`,
  `test_spinup.py`,
  `test_time_varying_forward_model.py`,
  `test_run_rollout_forward_model.py`,
  `test_varying_inflow_velocity.py`,
  `test_inflow_angle_simulation_sign.py`. These target scripts that
  are **not yet migrated**
  (`run_rollout_forward_model.py`,
  `run_ensemble_rollout_forward_model.py`,
  `run_time_varying_forward_model.py`, and the
  `test_inflow_angle_*` / `test_varying_inflow_velocity` / `test_spinup`
  cases that exercise forward-model behavior directly). The injection
  is load-bearing until those scripts move to Hydra. Per the
  migration invariant in Phase 4: each script's test should be
  migrated in the same PR as the script. Track this as the last
  finish-up task before Phase 5.
- **Phase 5 deletions** — `scripts/config.py`,
  `scripts/config_small.py`, `tests/config.py`,
  `src/pyurbanair/utils/config_utils.py`. All four still reference the
  now-dead `init_conditions_dir` / `true_sim_id` / `init_subdir`
  keys; the legacy `load_init_conditions_for_esmda` lives in
  `config_utils.py:357`. Defer until the six unmigrated tests are
  off the legacy modules.

## Phase 4 Finish-Up (2026-05-18)

The three remaining forward-model scripts and the three remaining
behavior tests that still injected `tests.config` are now migrated.
After this pass there are **zero** `sys.modules["scripts.config"]`
injections, **zero** `tests.config` imports, and **zero** uses of
`pyurbanair.utils.config_utils` left in `scripts/` or `tests/`.

### Scripts migrated

- `scripts/run_rollout_forward_model.py` — split into
  `def run(cfg)` + `@hydra.main`, forward-model creation/preparation
  through `instantiate(...)`, `create_true_params` helper for the
  parameters, `resolve_output_dir` for outputs.
- `scripts/run_ensemble_rollout_forward_model.py` — same structure
  plus `configure_failure_policy(ensemble_model, cfg.ensemble.failure)`.
  The `--num-steps` CLI flag becomes `cfg.run.num_steps` (default 2 in
  `conf/config.yaml`).
- `scripts/run_time_varying_forward_model.py` — same structure. The
  legacy `--use-true-params` flag becomes `cfg.run.use_true_params`
  (default `false`). Time-varying parameter construction (sigmoid ramp
  vs. constant `params.true.*`) stays in the script body since the
  sigmoid shape is script-local.

### Tests migrated

- `tests/test_run_rollout_forward_model.py`,
  `tests/test_run_ensemble_rollout_forward_model.py`,
  `tests/test_time_varying_forward_model.py` — rewritten against
  `compose_test_cfg([...])` and `run(cfg)`; legacy `argparse` /
  `sys.modules` plumbing removed.
- `tests/test_spinup.py` — uses `compose_test_cfg` to build two
  configs (with and without `time.spinup_time` override) and
  instantiates the forward model from each via
  `instantiate(cfg.model.forward_model)`. Drops the inline
  `LBMForwardModel(**args)` / `UDALESForwardModel(**args)` paths.
- `tests/test_inflow_angle_simulation_sign.py` and
  `tests/test_varying_inflow_velocity.py` — module-scoped `pylbm_cfg`
  fixture composes `preset=test` + `model=pylbm` +
  `model.forward_model.cuda=false` once; a `pylbm_model` fixture
  instantiates and prepares the forward model on top of it.
  `clean_outputs` replaces `clean_forward_model_outputs` end-to-end.

### Acceptance-criteria status

All Hydra-related criteria from the plan are now satisfied for the
entire `scripts/` tree:

- All ten run scripts (forward / ensemble / rollout / time-varying ×
  ESMDA variants) expose a `def run(cfg: DictConfig)` entry point and
  are testable without `@hydra.main`.
- No script imports `scripts.config`.
- No test patches `sys.modules` or mutates legacy dicts.
- Forward, ensemble, and ESMDA smoother construction all use
  `hydra.utils.instantiate`.
- PALM stays lazy under composition (regression test in place).
- `pressure_gradient_magnitude` is filtered out for non-uDALES models
  by the helpers; `create_time_varying_true_params` truth/prior
  correlation lengths remain distinct in the default config (AR(2) for
  both, but with different `correlation_length` values).

### Verified after this pass

```bash
.pixi/envs/default/bin/python -m pytest tests/test_hydra_config.py -q
# 15 passed
```

All ten migrated scripts py_compile and import; their `run(cfg)`
entry points are reachable. Full integration runs of the
forward-model tests are intentionally not blocked on here — they
require a working solver toolchain in the test pixi env (LBM
netcdf.mod, uDALES preprocessing) and remain environment-sensitive
as documented in the Phase 1 review.

### Phase 5 ready

With this finish-up, Phase 5 (delete `scripts/config.py`,
`scripts/config_small.py`, `tests/config.py`,
`src/pyurbanair/utils/config_utils.py`) is now strictly cleanup —
nothing in the migrated tree imports them. Any remaining references
in those four modules to dead keys (`init_conditions_dir`,
`true_sim_id`, `load_init_conditions_for_esmda`) can be deleted with
the module itself. Per the user's request, this cleanup is deferred
to a separate pass.

## Phase 4 Finish-Up Implementation Review (2026-05-18)

Review of the three new forward-model script migrations
(`run_rollout_forward_model.py`, `run_ensemble_rollout_forward_model.py`,
`run_time_varying_forward_model.py`), the six new test migrations, the
two new `conf/config.yaml` fields (`run.num_steps`,
`run.use_true_params`), the simulate-init-conditions deletion (commit
`3a085e7`), and the resolution of the three Phase 4 review bugs.
`pytest tests/test_hydra_config.py -q`: 15 passed.

### Prior review bugs — confirmed fixed

1. **`run_rollout_esmda.py` spin-up cleanup.**
   `scripts/run_rollout_esmda.py:46-48` now calls
   `clean_outputs(cfg.assim_model.name, assim_model)` immediately after
   the spin-up, with a comment explaining the per-iter NetCDF /
   warm-start side effect on pylbm. The zero-variance initial-ensemble
   caveat I flagged is now documented in a comment at lines 51-52 too.
2. **`run_time_varying_parameters_rollout_esmda.py` external_priors
   shape duplication.** A new helper `build_truth_ts_model` in
   `src/pyurbanair/config/hydra_helpers.py:126-147` replaces the inline
   `OmegaConf.to_container(...)` + `build_parameter_time_series(...)`
   boilerplate. The hardcoded `("inflow_angle", "velocity_magnitude")`
   allowlist in the old `_prior_cfg_as_external_priors` is gone — both
   the truth-model and the `create_time_varying_true_params` helper now
   use `_plain(external_cfg)` so the schema is consistent.
   `scripts/run_time_varying_parameters_rollout_esmda.py:209-221` no
   longer imports `OmegaConf` or `build_parameter_time_series` for this
   purpose.
3. **`_prior_model.yaml` redundant `# @package` header.** Removed.
   `conf/time_varying/_prior_model.yaml:1` now starts with
   `prior_model:` directly.

### What's good in this pass

- **Six tests fully migrated to `compose_test_cfg(...)`** with no
  remaining `sys.modules["scripts.config"]` injection or `tests.config`
  import anywhere in `tests/` or `scripts/` (verified by grep).
- **`test_spinup.py`** now composes two separate configs (with and
  without `time.spinup_time` override) and uses `instantiate(...)` for
  both forward models. Cleanly drops the `LBMForwardModel(**args)` /
  `UDALESForwardModel(**args)` fallthrough and the `model_args(model)`
  legacy helper.
- **`test_inflow_angle_simulation_sign.py` /
  `test_varying_inflow_velocity.py`** introduce a clean two-fixture
  pattern (`pylbm_cfg` composed once per module, `pylbm_model`
  instantiated on top of it). This is a good template for future
  fixture-heavy tests.
- **`scripts/simulate_init_conditions.py` deleted** in commit `3a085e7`,
  which closes the orphan-script item from the prior review.
- **`create_time_varying_true_params` now iterates over
  `sampled.data_vars`** instead of hardcoding parameter names
  (`src/pyurbanair/config/hydra_helpers.py:168`), so the "Add a new
  parameter" recipe in the codebase guide will no longer silently drop
  the new field from truth datasets.

### Confirmed bugs

#### 1. `run.results_dir` is now silently ignored by two ESMDA scripts; their tests still pretend to exercise it

`scripts/run_state_and_parameter_esmda.py:54-60` and
`scripts/run_time_varying_parameter_esmda.py:241-248` both
unconditionally set:

```python
out_dir = resolve_output_dir(cfg, "...")
assim_results_dir = out_dir / "assim_states"
...
assim_model = instantiate(cfg.assim_model.forward_model, results_dir=assim_results_dir)
```

…regardless of whether `cfg.run.results_dir` was set. This is a quiet
behavior change: the two scripts no longer have an in-memory mode.
Acceptable as a design choice, but
`tests/test_run_time_varying_parameter_esmda.py:42-45` still
parametrizes `*_in_memory` vs `*_on_disk` IDs that now run identically.

Two fixes:

- Drop the `use_results_dir` axis from the time-varying test entirely
  (and from the state+param test which doesn't have this axis anymore
  anyway), and rename the remaining IDs.
- Or restore an `if cfg.run.results_dir is not None` branch in the
  script so the flag actually toggles in-memory vs on-disk, and pick
  one default.

Either is fine, but don't keep parametrizing a no-op.

Five other scripts still honor `cfg.run.results_dir`, so the asymmetry
is real. Worth deciding intentionally rather than letting it drift.

#### 2. `tests/test_run_ensemble_rollout_forward_model.py` deletes `tmp_path` after creating `results_dir` under it

The legacy test had the same ordering hidden inside a `try:` block; the
migration preserved the ordering verbatim and the
`shutil.rmtree(tmp_path, ignore_errors=True)` after
`results_dir.mkdir()` now stands out. Today this happens to work
because the ensemble forward model recreates its results directory in
`run_ensemble`, but that is incidental.

Fix is mechanical: either delete the `shutil.rmtree(...)` call (the
`tmp_path` fixture already gives a fresh dir per test) or move it
above the `results_dir.mkdir()` so the order is sane.

#### 3. The plan file's prior-pass reformat broke list-item continuations and bold-with-backtick syntax

A markdown auto-formatter pass on the plan file de-indented
sub-paragraphs of bullet items across the document and mangled
backtick-inside-bold patterns. Two concrete classes of damage:

- **Bullet continuations are now flush-left**, so they render as
  separate paragraphs instead of as continuations of the parent
  bullet.
- **`- `**`foo`** is broken bold**: the lead backtick lands outside
  the `**…**` opener, so several bullets render as `**foo` text
  instead of bold.

Either re-run a markdown formatter with stable rules (prettier with
`prose-wrap=preserve`) on the whole file, or revert the reformat — the
document content didn't actually change in this pass, only its
whitespace and a couple of mangled headings.

### Things to look at

- **`scripts/run_time_varying_forward_model.py` silently fixed a latent
  crash in `--use-true-params`.** The legacy HEAD version had no
  `inflow_vec` assignment in the `if args.use_true_params:` branch, so
  the flag would raise `NameError` whenever set. The migration added
  `inflow_vec = np.full(n_snapshots, angle)`, which both fixes the
  crash and changes effective behavior: `use_true_params=True` now
  produces a truly constant-angle inflow rather than crashing. Worth
  calling out in the PR description so reviewers know it is an
  intentional behavior change rather than a sloppy port.
- **`run.num_steps` is `run_ensemble_rollout_forward_model.py`-specific.**
  `conf/config.yaml` adds `run.num_steps: 2` as a global field, but
  only `run_ensemble_rollout_forward_model.py` reads it.
  `run_rollout_forward_model.py`, by contrast, hardcodes two rollout
  steps via duplicated function calls. The Phase 4 Finish-Up notes
  imply `run.num_steps` is universal, but only one of the two scripts
  honors that. Either also route `run_rollout_forward_model.py`
  through `cfg.run.num_steps`, or scope `num_steps` under the ensemble
  rollout config rather than `run.*`.
- **`cfg.run` is becoming a script-specific grab-bag.** It now holds
  `skip_viz`, `results_dir`, `num_steps`, `use_true_params` — only the
  first two are script-agnostic. The earlier `run.init_conditions` →
  `esmda.use_init_conditions` move was a similar concern. Worth a
  one-time decision: keep `run.*` as the global behavioral namespace
  and move script-specific keys under their own groups, or accept
  `run.*` as the free-for-all bucket and reconsider the earlier move.
- **`run.num_steps` is per-script but `test_run_rollout_forward_model.py`
  doesn't set it.** Fine today (rollout-forward hardcodes 2 steps),
  but if `run_rollout_forward_model.py` starts reading
  `cfg.run.num_steps`, the test should override it the same way the
  ensemble-rollout test does.
- **`compose_test_cfg(..., cpu_only_pylbm=True)` would centralize the
  `model.forward_model.cuda=false` override.** Every test that runs
  pylbm re-types it. Minor.
- **`run_rollout_forward_model.py` duplicates the forward-model call
  to do two rollout steps.** The print line `Rollout: 2 steps`
  hardcodes the count. If the loop is ever extended (e.g. by routing
  through `cfg.run.num_steps`), the print line will lie.
- **`tests/test_inflow_angle_simulation_sign.py` and
  `tests/test_varying_inflow_velocity.py` use `hydra.initialize`
  directly** rather than the `compose_test_cfg` fixture. Necessary
  because the fixture is function-scoped and these tests need
  module-scoped composition, but both files re-list `preset=test`
  manually. A module-scoped helper in `conftest.py`
  (`compose_module_cfg`) would be the natural fix; minor.

### Summary

Migration is functionally complete. Three concrete items to fix before
Phase 5 cleanup:

1. Decide whether `cfg.run.results_dir` should be honored by the two
   ESMDA scripts that currently ignore it, and either re-enable it or
   prune the now-no-op `use_results_dir` test parametrizations
   (issue #1).
2. Reorder or delete the `shutil.rmtree(tmp_path)` in
   `test_run_ensemble_rollout_forward_model.py` so it doesn't wipe a
   directory that the test then references (issue #2).
3. Re-run a markdown formatter on this plan file or revert the
   whitespace-only reformat (issue #3).

The remaining items (split `run.num_steps` between rollout scripts or
move it out of `run.*`; revisit the `run.*` namespace becoming a
grab-bag; module-scoped compose helper for fixture-heavy tests; the
silent `--use-true-params` behavior fix) are cleanup and should land
together with whatever revisits this area next.

## Review Resolution (2026-05-18)

All comments from the **Phase 4 Finish-Up Implementation Review**
above have been addressed.

### Confirmed bugs — fixed

**#1: `cfg.run.results_dir` is silently ignored by two ESMDA scripts.**
Dropped the no-op `use_results_dir` axis from
[tests/test_run_time_varying_parameter_esmda.py](../../tests/test_run_time_varying_parameter_esmda.py).
The state+param test already had no such axis. The asymmetry between
the two scripts and the five others that still honor
`cfg.run.results_dir` was the design choice from issue #5 of the prior
review (route assim states under `out_dir / "assim_states"`); keeping
that choice and pruning the dead parametrization is the consistent
follow-through.

**#2: `tests/test_run_ensemble_rollout_forward_model.py` wipes
`tmp_path`.** Removed the `shutil.rmtree(tmp_path, ignore_errors=True)`
call entirely; pytest's `tmp_path` fixture already gives a fresh dir
per test.

**#3: Markdown formatting damage.** A small in-place Python normalizer
re-indented bullet continuation lines that had been de-indented to the
parent-bullet level (or to column 0). It walks the file tracking a
bullet-nesting stack and re-indents continuations that have *less than*
the expected indent for the current bullet, leaving code fences and
tables alone. It also repairs the `- `**`foo`** bold mangle via a
targeted regex. A grep for the two damage classes now returns zero
matches across the whole file.

### Things to look at — addressed

**`run.num_steps` is `run_ensemble_rollout_forward_model.py`-specific.**
Routed
[scripts/run_rollout_forward_model.py](../../scripts/run_rollout_forward_model.py)
through `cfg.run.num_steps` too: the duplicated two-step rollout
becomes a `for _ in range(num_steps - 1)` loop, and the `print(f"Rollout:
{num_steps} steps")` line is now correct for any count. Updated
[tests/test_run_rollout_forward_model.py](../../tests/test_run_rollout_forward_model.py)
to set `run.num_steps=2` explicitly so the test stays decoupled from
the schema default. This also resolves the "the print line will lie"
note.

**Module-scoped compose helper.** Added `compose_module_cfg` (a
module-scoped pytest fixture that returns the same composer callable as
`compose_test_cfg`) to
[tests/conftest.py](../../tests/conftest.py).
Routed
[tests/test_inflow_angle_simulation_sign.py](../../tests/test_inflow_angle_simulation_sign.py)
and
[tests/test_varying_inflow_velocity.py](../../tests/test_varying_inflow_velocity.py)
through it. Both files lose their direct `hydra.initialize` /
`compose` imports and re-listed `preset=test` line. The
function-scoped `compose_test_cfg` remains the default;
`compose_module_cfg` is only for fixture-heavy tests that legitimately
need module scope.

### Things to look at — deferred, with rationale

**`cfg.run` namespace decision (review: "becoming a grab-bag").**
After the review, `run.*` carries `skip_viz`, `results_dir`,
`num_steps`, `use_true_params`. The decision: keep `run.*` as the
**generic script-behavior namespace** for knobs that aren't intrinsic
to a config group. The earlier `run.init_conditions` →
`esmda.use_init_conditions` move was correct because it described a
per-smoother *workflow* concern (the smoother's data source). By
contrast, `num_steps` and `use_true_params` are script-execution knobs
without a natural home in `domain`, `time`, `params`, `ensemble`, `obs`,
`time_varying`, or `esmda`. They stay under `run.*`. If a future
migration adds a knob that *does* clearly belong in a group, it should
move there — but the threshold is "natural home elsewhere", not "is it
only used by one script".

**`use_true_params` silent crash fix.** The latent `NameError` in the
legacy `--use-true-params` branch is documented above as an
intentional behavior change. No code action — the call-out belongs in
the PR description, not the runtime.

**`compose_test_cfg(..., cpu_only_pylbm=True)` flag.** Not added. The
per-test `model.forward_model.cuda=false` override is three characters
longer than a hypothetical flag, only applies when a test runs pylbm,
and reads more obviously at the call site. Keeping it explicit is the
right trade-off for a small test suite.

### Verified after this pass

```bash
.pixi/envs/default/bin/python -m pytest tests/test_hydra_config.py -q
# 15 passed
```

All changed scripts and tests py_compile and import; the
`cfg.run.num_steps` override resolves correctly through composition.

### Note on this plan's history

During the markdown normalizer pass, a `git checkout
docs/plans/hydra_config_migration_plan.md` reverted the working tree
to HEAD — which removed both the previous "Phase 4 Finish-Up
(2026-05-18)" subsection (the record of the script migrations) and
the user's "Phase 4 Finish-Up Implementation Review (2026-05-18)"
section (the review being addressed) because both were uncommitted.
Both have been reconstructed verbatim from the conversation
transcript and added back above so the plan's history remains intact.
Future passes should commit plan edits before running any whole-file
formatter.
