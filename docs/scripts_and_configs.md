# pyurbanair — Scripts and Configuration Reference

Reference for [`configs/`](../configs/) (Hydra configs),
[`scripts/`](../scripts/) (executable entry points) and
[`workflows/`](../workflows/) (run + post-processing chains). Every config key
and the common overrides are in [`configs/README.md`](../configs/README.md),
the single reference for keys; the design record and the old -> new mapping
are in [`config_setup_spec.md`](plans/implemented/config_setup_spec.md). Every
script's module docstring documents its usage and outputs; read it before
changing a script.

The previous setup (`conf/`, `scripts/esmda/`, `scripts/filtering/`,
`scripts/neural_surrogate/`, ...) is archived under `archive/` and documented in
[`archive/scripts_and_configs_archived_setup.md`](archive/scripts_and_configs_archived_setup.md).

---

## Quick lookup

| You want to… | Go to |
|---|---|
| Run a forward simulation | `python scripts/run_forward.py model=pylbm ...` (§2.1) |
| Run ESMDA | `python scripts/run_smoother.py ...`, variant `'smoothing.smoother=${smoother.<name>}'` |
| Run a sequential filter (EnKF) | `python scripts/run_filtering.py ...`, `filtering.mode=state\|parameter\|joint` |
| Run the hybrid (ESMDA params + filter state) | `python scripts/run_hybrid.py ...` |
| Run + metrics + figures in one go | `bash workflows/assimilation_workflow.sh <smoother\|filtering\|hybrid> ...`, `bash workflows/forward_workflow.sh ...` (§2.4) |
| Add a new experiment (domain/sensors/geometry) | [`configs/case/`](../configs/case/) — one YAML per case; inputs in [`geometries/`](../geometries/README.md) |
| Switch CFD backend | `model=...` (forward) or `model@truth_model=... model@assim_model=...` (assimilation) |
| Change a setting (run size, DA components, observations, paths) | every key and the common overrides: [`configs/README.md`](../configs/README.md) |
| Use a saved truth | `assimilation.truth_dir=<forward run dir>` (+ `assimilation.truth_start_time`) |
| Generate surrogate training data | `python scripts/surrogate/generate_data.py` (§2.3) |
| Train / fine-tune a surrogate | `python scripts/surrogate/train.py --config-name surrogate/<config>` |
| Evaluate a surrogate | `python scripts/surrogate/evaluate_{stepper,autoencoder,latent_generator}.py` |
| Run the scripts with tiny test configs | `--config-dir tests/configs +test=forward` (see [`tests/README.md`](../tests/README.md)) |

---

## Part 1 — `configs/`: Hydra configuration tree

```
configs/
  forward.yaml            # entry point: scripts/run_forward.py
  assimilation.yaml       # entry point: run_smoother.py, run_filtering.py, run_hybrid.py
  common.yaml             # run, paths (per machine), ensemble budget, Hydra run dir
  case/                   # xie_and_castro (default), barcelona
  model/                  # pylbm, pyudales, pypalm, neural_surrogate
  params/                 # static, dynamic, dynamic_sine, dynamic_cosine,
                          #   static_truth, dynamic_truth, surrogate_training_data
  assimilation_settings/  # smoother, analysis, localization, state_reduction, inflation
  surrogate/              # generate_data, train_*, finetune_stepper, eval,
                          #   training (shared defaults), architectures
```

### 1.1 Entry points and `common.yaml`

- **`forward.yaml`** — `common` + `case` + `params` + `model`, plus the
  `forward` block.
- **`assimilation.yaml`** — `common` + `case` + `model@truth_model` +
  `model@assim_model` + `params@truth_params` + `params@prior_params` + all of
  `assimilation_settings/`, plus the DA blocks. Each DA script reads the blocks
  it needs and appends its workflow name (`smoother`, `filtering`, `hybrid`) to
  `paths.results_dir`.
- **`common.yaml`** — `paths` (per-machine scratch), the one `ensemble` budget
  and Hydra's run dir.

Every key of these blocks, and the common overrides, are listed in
[`configs/README.md`](../configs/README.md).

### 1.2 `case/`

A case is self-contained (`# @package _global_`): `case_name`, `domain`
(`nx/ny/nz`, `bounds`), `geometry` (`case_dir`, the case's
[geometries/](../geometries/) folder, and `stl_path` inside it), `obs`
(assimilation sensors `x/y/z_points`, held-out `validation_*_points`, `states`) and `time` (`simulation_time` per window,
`output_frequency`, `spinup_time`, `seconds_per_knot`). Keep `nx`/`ny` even
(PALM's FFT pressure solver rejects odd cyclic dimensions). Add a case by
copying one and adjusting.

### 1.3 `model/`

Each file wires one backend: `name`, `solver_name` (the observation grid
convention), `forward_model`, `prepare`, `ensemble_model` (which reads the
`ensemble` budget). Assimilation mounts the group twice
(`model@truth_model=... model@assim_model=...`); different backends give
genuine model error.

| File | Backend | Notable fields |
|---|---|---|
| `pylbm.yaml` | Lattice Boltzmann | `cuda`, `ncpu` (OpenMP threads; CPU build only), `verbose`, `profile_config`, `boundary_condition`, `inlet_turbulence` |
| `pyudales.yaml` | uDALES (staggered grid) | `ncpu`, `nudging_config`, `instability_check` (dt watchdog), `precomputed_geom_dir`, `closure`, `model_discrepancy` (§1.7) |
| `pypalm.yaml` | PALM | `ncpu` (must divide `domain.nx`), `boundary_condition`, `nudging_config` |
| `neural_surrogate.yaml` | learned stepper | `model_dir` (a `scripts/surrogate/train.py` output), `spinup_source` (`forward_model \| training_data \| generative`), `spinup_forward_model`, `generative_spinup`, `default_params`; `solver_name: pylbm` |

### 1.4 `params/`

Standalone `_target_` samplers. Mounted once for forward runs (`params=...`)
and twice for assimilation (`params@truth_params=... params@prior_params=...`),
so truth and prior stay separate (no inverse crime).

| File | Sampler | Use |
|---|---|---|
| `static.yaml` | `ParameterSampler` | static prior |
| `dynamic.yaml` | `AR2RelaxationModel` | time-varying AR(2) prior relaxing to an external mean/std |
| `static_truth.yaml` | `ParameterSampler` | static truth |
| `dynamic_truth.yaml` | `AR2RelaxationModel` | time-varying truth |
| `dynamic_sine.yaml`, `dynamic_cosine.yaml` | `HarmonicParameterModel` | smooth time-varying truths (the assimilation default truth is `dynamic_sine`) |
| `surrogate_training_data.yaml` | `UniformExternalAR2Sampler` | inflow trajectories for `generate_data.py` |

Time-varying samplers take a value every `time.seconds_per_knot` seconds.

### 1.5 `assimilation_settings/`

Each file is mounted at the top level and holds every option of one component
(`smoother`, `analysis`, `localization`, `state_reduction`, `inflation`); the
`smoothing` / `filtering` slots point at one by interpolation. The options and
their slots are listed in [`configs/README.md`](../configs/README.md#keys); the
classes behind them in [data_assimilation.md](data_assimilation.md).

### 1.6 Config checks

`scripts/utils/inconsistency_check.py::check_config(cfg, workflow)` runs first in
every `run_*.py` and reports every problem at once. Among others it enforces:
a time-varying prior pairs with `smoother.dynamic` / `state_and_dynamic` and a
static one with the others; the smoother needs truth and prior both static or
both time-varying; filtering needs a static prior; the hybrid's smoother is
parameter-only and its filter mode `state` or `joint`; localization and state
reduction are exclusive; distance localization in the smoother needs a
state-bearing smoother; LETKF needs a localization and ETKF forbids one;
parameter-updating filter modes need inflation or a parameter evolution;
`assimilate_every_n_step` divides the frames per window;
`hybrid.likelihood_allocation=shared_budget` needs `filtering.beta > 1`, no
aggregation and `assimilate_every_n_step=1`; `params_to_estimate` names only
prior parameters; and the SGS-discrepancy rules of §1.7.

A localized state-bearing smoother is also checked for memory. Its update
holds an `(N_aug, N_d, N_d)` float32 array; `N_aug` is estimated as
`3·nx·ny·nz` (a lower bound). `check_config` refuses the run when the estimate
exceeds the machine's physical memory and warns above half of it. The fix is
`smoothing.state_reduction` instead of a localization, a larger
`observation.aggregation.interval_seconds`, or fewer sensors.

### 1.7 uDALES model discrepancy

`configs/model/pyudales.yaml` has an optional `forward_model.model_discrepancy`
block (height band, regularization, log cap) that needs the Vreman closure.
Its coefficients are static parameter fields `sgs_bias_b0/b1/b2` (default zero)
set in the `params/` files (commented examples in `dynamic.yaml` and
`dynamic_sine.yaml`; set in `dynamic_truth.yaml`). Estimate them by naming them in
`assimilation.params_to_estimate`. The config checks require them to be static
parameters; with discrepancy the smoother must be parameter-only and match the
prior, the hybrid needs `filtering.mode=state` and
`ensemble.failure.policy=raise`, and a filter estimating them needs
`filtering.parameter_evolution=null`. The physics and checkpoint contract are
in [pyudales §4.1](pyudales.md#41-strainrotation-discrepancy). (The old
`experiment/*/sgs_bias_small` recipes are archived in `archive/conf/experiment/`.)

### 1.8 `surrogate/`

Run with `--config-name surrogate/<name>` (the `@package` convention and
example overrides are in [`configs/README.md`](../configs/README.md#neural-surrogates)).

| File | Script | Purpose |
|---|---|---|
| `generate_data.yaml` | `scripts/surrogate/generate_data.py` | training data; `data.geometry.mode: fixed \| random`, `data.sharding`, its own `time.*` |
| `training.yaml` | — | shared `trainer`, `optimizer`, `batch_sampler`, `dataloader`, `dataset`, `loss` |
| `architectures.yaml` | — | every stepper architecture, `<family>_<size>` (`p3d`, `unet_convnext`, `upt`) |
| `train_stepper.yaml` | `train.py` | next-step stepper; `'architecture=${architectures.<name>}'` |
| `train_autoencoder.yaml` | `train.py` | field autoencoder (Tadpole) |
| `train_latent_generator.yaml` | `train.py` | flow-matching latent initial-field generator |
| `train_dft.yaml` | `train.py` | pretrained autoencoder -> stepper (DFT) |
| `finetune_stepper.yaml` | `train.py` | fine-tune a pretrained stepper, `method: full \| lora` |
| `eval.yaml` | `evaluate_*.py` | blocks `stepper` (one or several `models`), `autoencoder`, `latent_generator` |

Each training config loads `training.yaml` and overrides only what differs.
Weights go to `<paths.weights_dir>/<name>/`, data to
`<paths.training_data_dir>/<model>_<geometry>/`.

---

## Part 2 — `scripts/` and `workflows/`

### Standard script shape

Hydra scripts expose `def run(cfg)` plus a thin `@hydra.main` wrapper
(`config_path="../configs"`), so tests call `run(cfg)` on a composed config.
`run` calls `check_config` first and writes under `cfg.paths.results_dir`
(saving the composed `config.yaml` there). Post-processing scripts are plain
CLIs that take a finished run dir. Shared helpers (truth, observation pieces,
ensemble model, I/O) live in `scripts/utils/helper_functions.py`; each script keeps
its own workflow logic.

### 2.1 Run scripts

| Script | Config | Writes |
|---|---|---|
| `run_forward.py` | `forward.yaml` | `<results_dir>/`: `config.yaml`, `state.nc` (all windows; `ensemble` dim for ensembles), `params.nc` |
| `run_smoother.py` | `assimilation.yaml` (`assimilation`, `observation`, `smoothing`) | `<results_dir>/smoother/` |
| `run_filtering.py` | `assimilation.yaml` (`assimilation`, `observation`, `filtering`) | `<results_dir>/filtering/` |
| `run_hybrid.py` | `assimilation.yaml` (+ `smoothing`, `hybrid`) | `<results_dir>/hybrid/` |

The forward run covers 1 + `forward.rollout_steps` windows of
`time.simulation_time`, each starting from the previous window's last state.
Its `state.nc` + `params.nc` are what `assimilation.truth_dir` loads.

All DA run dirs share one layout: `config.yaml`, `run_info.yaml`,
`true_state.nc` (inline truth), `true_params.nc` and
`windows/window_{w}_{prior,posterior}_{params,state}.nc` plus
`window_{w}_obs.nc`. The smoother assimilates each window's observations with
`smoothing.num_steps` ESMDA steps and starts the next window from the
posterior. The filter forecasts `assimilation.assimilate_every_n_step` frames,
then assimilates the last one; windows only chunk the run. The hybrid runs the
smoother for the parameters, then the filter over the same observations for the
state (`window_{w}_filter_obs.nc`, and `window_{w}_filter_params.nc` in
`mode=joint`). See each script's docstring for the exact files.

### 2.2 Post-processing

| Script | Input | Writes |
|---|---|---|
| `compute_metrics.py <run dir>` | a DA run dir | `metrics.yaml`: parameter RMSE/CRPS (+ prior and reduction), ensemble-mean \|U\| RMSE, per sensor set (assimilated and validation) RMSE and energy score, spread–skill, climatology baseline, per-window sensor statistics; observation-space fit and Desroziers per stage |
| `visualize_assimilation.py <run dir>` | a DA run dir (run `compute_metrics.py` first for the rank histogram) | `figures/`: parameter evolution, animation, final state, mean/TKE slices, station profiles, sensor time series, TKE evolution, rank histogram |
| `visualize_forward.py <run dir>` | a forward run dir | `figures/`: field snapshot, animation, parameters (with inlet-recovered angle/speed) |

Both read window files one member at a time, so multi-GB runs fit in memory.

### 2.3 Neural-surrogate scripts (`scripts/surrogate/`)

| Script | Config | Does |
|---|---|---|
| `generate_data.py` | `surrogate/generate_data` | one simulation per sample on the case geometry or random STL layouts; resumable, shardable; writes `state/` and `param/` `{train,val,test}/sample_XXXX.nc` |
| `train.py` | `--config-name surrogate/<train_*\|finetune_stepper>` | builds datasets + model for the config's `task` (`scripts/utils/tasks.py`: `stepper`, `autoencoder`, `latent_generator`, `dft`, `finetune_stepper`), then fits |
| `evaluate_stepper.py` | `surrogate/eval` block `stepper` | rolls one or several steppers out on test trajectories; metrics + figures |
| `evaluate_autoencoder.py` | `surrogate/eval` block `autoencoder` | reconstruction metrics + figures |
| `evaluate_latent_generator.py` | `surrogate/eval` block `latent_generator` | generated-field statistics vs real fields |

`scripts/utils/eval_common.py` holds the loading and slice-figure helpers shared by the
evaluation scripts. See [neural_surrogates.md](neural_surrogates.md) for the
library side.

### 2.4 Workflows

`workflows/*.sh` chain a run with its post-processing on the same run dir.
Hydra overrides go to the run script; the post-processing reads the dir it
wrote.

| Workflow | Args | Runs |
|---|---|---|
| `forward_workflow.sh` | `[overrides...]` | `run_forward.py`, then `visualize_forward.py <paths.results_dir>` and the HTML viewer with its 3D view, `python -m visualization <paths.results_dir> <paths.results_dir>/viewer` in the `rendering` env ([visualization.md](visualization.md)) |
| `assimilation_workflow.sh` | `<smoother\|filtering\|hybrid> [overrides...]` | `run_<method>.py`, then `compute_metrics.py` and `visualize_assimilation.py` on `<paths.results_dir>/<method>` |

```bash
bash workflows/forward_workflow.sh model=pylbm params=static_truth
bash workflows/assimilation_workflow.sh filtering 'filtering.analysis=${analysis.letkf}'
```

Each resolves the run dir first from the same overrides
(`--cfg job --resolve -p paths.results_dir`), runs from the repo root, stops on
the first failing stage (`set -euo pipefail`) and prints `Done: <run dir>`. Run
them inside the dev environment (`pixi shell -e dev`). The SLURM job scripts
wrap them ([job_scripts.md](job_scripts.md)).

### 2.5 Other

- `scripts/tools/` — geometry CLIs: `prepare_case_stl.py`,
  `preprocess_udales_geometry.py`, `benchmark_geometry.py`,
  `download_urbantales_geometries.py`, `rasters_to_stl.py`.
- `scripts/setup_dev_env.sh` — behind `pixi run setup-dev`.
- `scripts/start_mcp` — MCP server launcher; the server runs `run_forward.py`
  on `configs/forward.yaml` (see [mcp.md](mcp.md)).
- `scripts/register_claude.sh` — behind `pixi run -e mcp register-claude`:
  registers `start_mcp` with Claude Code after asking.
- `job_scripts/` — SLURM wrappers around the workflows and surrogate scripts
  for Snellius and DelftBlue (see [job_scripts.md](job_scripts.md)).

### Not ported from the archived setup

The ground-truth utilities (`adjust_simulations/`: trim spin-up, 32-bit
conversion, ...), the figure-creation and sweep-comparison scripts
(`figure_creation/`, `figspec/`), `compare_models.py` and
`preview_config.py` are in `archive/scripts/` only. A spin-up can be skipped
at load time with `assimilation.truth_start_time` instead of trimming.
