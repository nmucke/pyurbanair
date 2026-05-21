# Plan: UPT neural-surrogate forward model (`libs/pyupt`)

Status: **proposal / design doc**. Implementation not started.

This document plans a neural-network surrogate forward model based on
**Universal Physics Transformers** (UPT, [arXiv:2402.12365](https://arxiv.org/abs/2402.12365),
Alkin et al., NeurIPS 2024) and packages it as a fourth backend, `pyupt`,
that drops into the existing `BaseForwardModel` / `BaseEnsembleForwardModel`
machinery alongside `pylbm`, `pyudales`, `pypalm`.

## 0. Why UPT (recap of the decision)

Confirmed requirements: **fixed domain, varying inflow** + **transient
time-series** output feeding multi-window ESMDA. That rules out the
million-scale, steady-state, geometry-generalizing solvers (AB-UPT,
Transolver++) and points at UPT, whose native **latent-space temporal
propagation** maps directly onto our rollout pattern
([`BaseRolloutForwardModel`](../src/pyurbanair/base_rollout_forward_model.py)
and `scripts/run_rollout_esmda.py`). See `docs/codebase_guide.md` §3.

Scope of this surrogate:
- Inputs: an initial state (`u,v,w[,pres]` on the solver grid, `time` length ≥1)
  + scalar parameters (`inflow_angle`, `velocity_magnitude`, and for uDALES
  `pressure_gradient_magnitude`).
- Output: a time-indexed `xarray.Dataset` with the same grid axes and
  variables as the source solver — **identical contract to the Fortran
  backends** so ESMDA, the observation operator, and plotting are unchanged.
- One trained model per (solver, geometry, grid) tuple. Geometry generalization
  is explicitly out of scope (that is what would have justified AB-UPT/Transolver++).

## 1. Key architectural decisions (read these first)

These three decisions shape everything below. The first two are flagged again
in §12 as the decisions to confirm before coding.

### D1 — Framework: JAX + Equinox/Flax (recommended)
The repo is already JAX-centric: `jax>=0.7` is a top-level dependency, ESMDA
is written in JAX, and the ensemble executor deliberately uses `forkserver`
*because* JAX starts threads at import
([`base_ensemble_forward_model.py:417`](../src/pyurbanair/base_ensemble_forward_model.py#L417)).
A JAX surrogate keeps a single accelerator stack and leaves the door open to
differentiating through the forward model inside ESMDA later.
- Recommended: **Equinox** (lightweight, PyTree-native modules) + **Optax**
  (optimizers) + **Orbax** (checkpointing). Flax NNX is an acceptable alternative.
- Trade-off / alternative: the **official UPT reference code is PyTorch**. Using
  PyTorch (DDP/FSDP, mature transformer ecosystem) means maintaining a second
  accelerator framework in the repo and bridging tensors↔JAX at the ESMDA
  boundary. Recommend JAX unless porting effort from the reference repo is the
  dominant cost. **Decision to confirm — see §12.**

### D2 — Ensemble parallelism: batch on-device, do NOT fork N processes
The existing parallel path
([`_run_parallel`](../src/pyurbanair/base_ensemble_forward_model.py#L395)) is
built for **CPU-bound Fortran subprocess** solvers: forkserver workers + CPU
pinning, DRAM-bandwidth-capped at ~4 workers (`docs/ensemble_scaling.md`). For a
GPU NN that model is wrong — N processes each load the weights onto the GPU and
contend for it. The natural parallelism for a NN ensemble is **vectorizing over
the `ensemble` dimension in a single process** (`jax.vmap` / a leading batch
axis), running all members in one or a few batched forward passes.

`pyupt.EnsembleForwardModel` therefore **overrides** `run_ensemble` to:
- Stack the per-member initial states and params into batched arrays.
- Run one batched autoregressive rollout.
- Re-split into a `concat`-along-`ensemble` `xarray.Dataset`, matching the
  return contract of the base class.
- Keep `num_parallel_processes` semantics only as an optional fallback (e.g.
  CPU-only smoke tests); default to in-process batching.

This is the single most important efficiency decision and the main place where
`pyupt` legitimately diverges from the other backends.

### D3 — Common grid / staggering
UPT is grid-agnostic but our training tensors are simplest on a single
collocated grid. Train and predict on the **collocated `x,y,z` grid**:
- pylbm / pypalm already collocated → use directly.
- pyudales staggered (`xt/xm`, …) → interpolate to a common grid with the
  existing `pyudales.utils.grid_utils.interpolate_grid` *before* building
  training tensors, and register `pyupt` in the observation operator with the
  collocated mapping (see §8.4).
A geometry/building **mask** (solid vs fluid cells) is a static input channel.

## 2. Library layout (`libs/pyupt`)

Mirror the `pylbm` shape (`docs/codebase_guide.md` §8 recipe):

```
libs/pyupt/
  pyproject.toml                    # editable install + pixi pkg (see §9)
  src/pyupt/
    __init__.py
    forward_model.py                # ForwardModel(BaseForwardModel)      (§4)
    ensemble_forward_model.py       # EnsembleForwardModel(...) override  (§4, D2)
    model/
      upt.py                        # UPT module: encoder→propagator→decoder
      encoder.py                    # grid/points → latent tokens (perceiver-style pooling)
      propagator.py                 # latent-space transformer time-stepper
      decoder.py                    # latent → field on query positions
      conditioning.py               # param (inflow_angle, velocity_mag, ...) embedding
      layers.py                     # attention blocks, MLPs, pos-encoding
    data/
      generate.py                   # driver: solver ensembles → trajectory corpus (§5)
      dataset.py                    # on-disk corpus → batched windows
      normalization.py              # fit/apply per-variable standardization (§6.3)
      grid.py                       # grid metadata, collocation, mask extraction
    training/
      train.py                      # single + multi-GPU training entry (§6, §7)
      loop.py                       # step fn, rollout/pushforward loss, eval
      sharding.py                   # jax mesh / data-parallel helpers (§6.4)
      checkpoint.py                 # Orbax save/restore + manifest (§7)
    utils/
      state_io.py                   # xarray.Dataset ↔ model tensor (the contract glue)
      params_io.py                  # params Dataset → conditioning vector
      registry.py                   # resolve model id → checkpoint dir
  tests/                            # unit tests local to the lib (tiny model)
models/pyupt/                       # trained-model registry (git-ignored)  (§7)
```

## 3. Data flow at a glance

```
priors ──▶ solver ensembles (pylbm/pyudales/pypalm) ──▶ trajectory corpus (.zarr/.nc)
                                                              │
                                          fit normalization + grid/mask metadata
                                                              │
                                    train UPT (single/multi-GPU) ──▶ checkpoint + manifest
                                                              │
        ESMDA / rollout scripts ◀── pyupt.ForwardModel(checkpoint) ◀──────┘
        (same contract as Fortran backends)
```

## 4. Forward-model contract mapping

`pyupt.ForwardModel(BaseForwardModel)` must implement the four abstract methods
(`docs/codebase_guide.md` §3). Mapping:

| Base method | pyupt behavior |
|---|---|
| `__init__` | Load checkpoint (weights + normalization + grid/mask metadata) via `utils/registry`. Store `simulation_time`, `output_frequency` → derive `num_outputs = round(simulation_time/output_frequency)`. Call `super().__init__(results_dir=...)`. Lazy-init the accelerator/model on first `run_single` so forkserver workers stay importable. |
| `run_single(state, params, sim_name)` | (1) `params` → conditioning vector (`utils/params_io`). (2) `state` (or cold-start prior) → normalized initial latent (`utils/state_io`). (3) autoregressively roll the latent propagator `num_outputs` steps. (4) decode each step, denormalize, reapply the building mask, assemble an `xarray.Dataset` with `time` + the source grid coords. Return it. |
| `_apply_inflow_settings(params)` | Store the conditioning vector on `self` (no files to edit, unlike Fortran). Handle time-varying params (`time` dim) by feeding a per-step conditioning sequence. |
| `save_results` / `_clean_output` | `save_results` = `self._save_results` (NetCDF, base helper). `_clean_output` = no-op (no scratch files). |

Notes:
- **Warm start / rollout**: the multi-window pattern already feeds each window's
  final `state` into the next `run_single`; UPT consumes it natively as the new
  initial condition. No restart-file machinery (unlike pylbm's `.uf` files).
- **`get_states` / on-disk mode** work unchanged via the base class.
- `pyupt.EnsembleForwardModel(BaseEnsembleForwardModel)`: implements
  `_create_new_forward_model` (cheap — share the immutable weights, clone only
  per-member result dirs) **and** overrides `run_ensemble` for on-device
  batching (D2). The failure policy is largely moot (a NN forward pass does not
  raise `CalledProcessError`); keep `"raise"` default.

## 5. Generating simulated training data

Principle: **the existing Fortran backends are the data generator.** Reuse the
ensemble machinery rather than inventing a new runner.

New script `scripts/generate_upt_training_data.py` (standard `run(cfg)` + thin
`@hydra.main` shape, `docs/codebase_guide.md` §5):
1. Build the chosen source solver + ensemble model from Hydra exactly like
   `run_ensemble_forward_model.py`.
2. Sample a large parameter set from the prior — reuse
   [`create_parameter_ensemble`](../src/pyurbanair/config/hydra_helpers.py#L90)
   (and `parameter_time_series/` for time-varying inflow, to teach the
   surrogate transient boundary conditions).
3. Run the ensemble **on disk** (per-member trajectories) over the full
   `simulation_time` at the target `output_frequency`.
4. For each trajectory store: the `(u,v,w[,pres])` field over `time`, the
   parameter vector, the static building mask, grid metadata. Convert staggered
   uDALES output via `interpolate_grid` (D3).
5. Persist to a **corpus directory** as **Zarr** (chunked along `time`/sample;
   better random access for training than many small NetCDFs). Keep a JSON
   manifest: solver, grid, bounds, param ranges, counts, git SHA of the
   generating code.

Sampling guidance:
- Cover the inflow prior generously (and a bit beyond) to avoid extrapolation at
  ESMDA time; stratify over `inflow_angle` × `velocity_magnitude`.
- Include the **spin-up transient**, not just the statistically-steady tail —
  the surrogate must reproduce early-window dynamics for assimilation.
- Split **by trajectory** into train/val/test (never leak frames of one
  trajectory across splits).
- This step is the expensive one; run it under the existing ensemble scaling
  guidance (`num_parallel_processes≈4` on the single box; scale out on a
  cluster). Generation is embarrassingly parallel across parameter samples.

**Anti-inverse-crime note**: when later evaluating ESMDA with `pyupt` as the
*assim* model, draw the *truth* from a Fortran solver (the config already mounts
`model/` twice, `docs/codebase_guide.md` §5) so you measure real surrogate error,
not a model matched to itself.

## 6. Training

### 6.1 Task formulation
- One-step learner: `(state_t, conditioning) → state_{t+1}` in latent space.
- **Rollout/pushforward loss is mandatory** for ESMDA usefulness: train with
  multi-step unrolled predictions (curriculum: grow horizon over training, or
  pushforward with stop-gradient on the warm-up portion) so error does not
  blow up over a window. This is the property AB-UPT/Transolver++ never
  demonstrate and the main risk for a surrogate driven repeatedly by ESMDA.
- Loss: normalized per-variable MSE on the field; optionally a spectral/gradient
  term to preserve wake sharpness; mask out solid cells.

### 6.2 Conditioning
Embed `inflow_angle` (use sin/cos to respect periodicity), `velocity_magnitude`,
and (uDALES) `pressure_gradient_magnitude` via an MLP → FiLM/added tokens in the
propagator. Time-varying params feed a per-step conditioning sequence.

### 6.3 Normalization
Fit per-variable mean/std (and mask-aware stats) over the **training split
only**; store in the checkpoint manifest; apply at inference. Never recompute on
the assimilation data.

### 6.4 Single-GPU and multi-GPU
- **Single GPU**: default. `jit`-compiled step, gradient accumulation if memory-bound.
- **Multi-GPU (data parallel)**: JAX `jax.sharding` with a 1-D device `Mesh`;
  shard the batch axis, replicate params; `jax.lax.pmean` gradients. Encapsulate
  in `training/sharding.py` so the step fn is mesh-agnostic. This is the right
  default — UPT-scale models for our moderate grids fit on one GPU; we parallelize
  over **data**, not model.
- **Model/sharded parallel (optional, later)**: only if a single grid won't fit
  in memory; shard latent tokens across devices. Not needed for the initial
  urban grids (`domain.nx/ny/nz` are modest). Mention as a hook, don't build it.
- Launcher: a pixi task `train-upt` (single) and `train-upt-multi` (sets
  `XLA_FLAGS`/visible devices); multi-node via SLURM on DelftBlue mirrors the
  existing `delftblue` env. PyTorch alternative would use `torchrun` DDP/FSDP.
- Throughput hygiene: mixed precision (bf16 compute, fp32 master), input
  pipeline prefetch from Zarr, log step time + tokens/s.

### 6.5 Observability
TensorBoard/W&B optional; at minimum log train/val loss, **rollout error vs
horizon** (the metric that matters), and periodic field snapshots reusing
`pyurbanair.utils.animation_utils`.

## 7. Storing models and configs

A trained model is **weights + everything needed to reproduce inference**:

```
models/pyupt/<run_id>/
  weights/                 # Orbax checkpoint (params + optimizer state for resume)
  config.yaml              # full resolved Hydra training config (snapshot)
  normalization.json       # per-variable mean/std, mask stats
  grid.json                # solver_name, nx/ny/nz, bounds, coord arrays, mask ref
  manifest.json            # run_id, git SHA, data corpus id, metrics, created_at
  metrics.json             # final/best val + rollout-horizon errors
```

- `run_id` = timestamp + short git SHA (e.g. `20260601T1200_a1b2c3d`); `latest`
  symlink per (solver, geometry) for convenience.
- `models/` is **git-ignored** (large binaries). Document an external sync
  target (object store / shared scratch) for sharing checkpoints across machines.
- Inference resolves a checkpoint by **explicit path** or by `run_id` through
  `utils/registry.py`. The forward-model Hydra config (§8) takes a
  `checkpoint_path`. The manifest's `git SHA` + `config.yaml` make any run
  reproducible; the data `corpus id` ties back to §5's manifest.
- Hydra writes its own run dir for training under
  `${paths.base_results_dir}` — keep that for logs, but the *promoted* artifact
  lives under `models/pyupt/` so inference does not depend on Hydra run dirs.

## 8. Configuration & Hydra wiring

### 8.1 `conf/model/pyupt.yaml`
Mirror [`conf/model/pylbm.yaml`](../conf/model/pylbm.yaml):
```yaml
name: pyupt
solver_name: pyupt          # new entry in the obs-operator dim_mapping (§8.4)
checkpoint_path: ???        # required: models/pyupt/<run_id> or a path
forward_model:
  _target_: pyupt.forward_model.ForwardModel
  _convert_: all
  checkpoint_path: ${..checkpoint_path}
  nx: ${domain.nx}
  ny: ${domain.ny}
  nz: ${domain.nz}
  bounds: ${domain.bounds}
  simulation_time: ${time.simulation_time}
  output_frequency: ${time.output_frequency}
  device: cuda              # or cpu
prepare:
  _target_: pyurbanair.config.hydra_helpers.prepare_pyupt   # load/validate checkpoint
ensemble_model:
  _target_: pyupt.ensemble_forward_model.EnsembleForwardModel
  _convert_: all
  ensemble_size: ${ensemble.ensemble_size}
  num_parallel_processes: 1   # default in-process batching (D2)
```

### 8.2 `hydra_helpers` additions
- Add `prepare_pyupt(forward_model, ...)` (replaces `compile` — validates the
  checkpoint, warms the JIT). Keep it lazy/import-light like the pypalm pattern
  so composing `model=pylbm` never imports `pyupt`/heavy NN deps.
- Extend `clean_outputs` with an explicit `elif model_name == "pyupt"` no-op
  branch. **Important**: the current `else` arm falls through to uDALES cleanup
  ([`hydra_helpers.py:55-64`](../src/pyurbanair/config/hydra_helpers.py#L55-L64));
  per `docs/codebase_guide.md` §8 recipe, add the branch *and* convert the
  fall-through `else` into a raise so unknown models can't silently get uDALES
  cleanup.
- `create_true_params` / `create_parameter_ensemble` already produce the param
  Datasets the surrogate consumes — no change unless a new parameter is added.

### 8.3 Training config group
New `conf/upt/` group (model dims, optimizer, batch, rollout-horizon schedule,
data corpus path, sharding) consumed only by `scripts/generate_upt_training_data.py`
and `scripts/train_upt.py`. Keep training config **out of** the inference
`model/` group; the trained checkpoint snapshots its own training config (§7).

### 8.4 Observation operator
Register `pyupt` in
[`ObservationOperator.__init__`](../libs/data-assimilation/src/data_assimilation/observation_operator.py#L65)
with the collocated mapping (same as `pylbm`):
```python
elif solver_name == "pyupt":
    self.dim_mapping = {v: {"z": "z", "y": "y", "x": "x"} for v in ("u", "v", "w")}
```
(If a surrogate is trained on uDALES data we still output a collocated grid, so
this single mapping suffices — that is the point of D3.)

## 9. Packaging, environment, dependencies

- `libs/pyupt/pyproject.toml` mirrors `libs/pylbm/pyproject.toml`: editable
  install, `packages = ["src/pyupt"]`. Python deps: `jax`, `equinox`/`flax`,
  `optax`, `orbax-checkpoint`, `zarr`, `numpy`, `xarray`. No Fortran/MPI deps.
- Top-level `pyproject.toml`:
  - New pixi feature `[tool.pixi.feature.pyupt.dependencies]` (the NN libs) and
    `[tool.pixi.feature.pyupt.pypi-dependencies]` → `pyupt = { path = "libs/pyupt", editable = true }`.
  - Add `pyupt` to the `cuda` (and `delftblue`) environments for GPU train/infer;
    add to `dev` for CPU smoke tests. **Do not** add NN deps to the lean
    `default` env.
  - GPU JAX: install `jax[cuda12]` via the existing `cuda` feature/activation;
    document that CPU envs keep `JAX_PLATFORMS=cpu`.
- Lazy import: keep all `pyupt.*` `_target_` blocks confined to
  `conf/model/pyupt.yaml` so non-UPT compositions never import the NN stack —
  same invariant pypalm relies on, asserted by a regression test
  (`tests/test_hydra_config.py`).

## 10. Inference / use in the existing pipeline

No changes to ESMDA, rollout, or the scripts beyond config selection:
```bash
# single forward sim with the surrogate
python scripts/run_forward_model.py model=pyupt model.checkpoint_path=models/pyupt/latest

# parameter ESMDA, Fortran truth vs surrogate assim (anti-inverse-crime)
python scripts/run_parameter_esmda.py \
  model@truth_model=pylbm model@assim_model=pyupt \
  assim_model.checkpoint_path=models/pyupt/latest
```
Because `pyupt.ForwardModel` honors the `BaseForwardModel` contract (time-indexed
`xarray.Dataset`, same grid/vars), the observation operator, localization, and
plotting work untouched. The big win: a surrogate ESMDA window costs a batched
GPU forward pass instead of N Fortran simulations.

**Uncertainty caveat**: the surrogate is deterministic; ensemble spread comes
from the parameter and initial-state ensembles, *not* model stochasticity. UPT
will under-disperse relative to the Fortran solver. Plan to (a) document this and
(b) optionally add model-error inflation or an ensemble of surrogates / MC-dropout
later if ESMDA spread collapses.

## 11. Testing & CI

- `libs/pyupt/tests/`: unit tests for `state_io`/`params_io` round-trips, a
  **tiny** UPT (2 layers, small latent) training-step test, a rollout-shape test.
- `tests/` (top-level): a `compose_test_cfg(["model=pyupt", ...])` test that
  `run(cfg)` works end-to-end against a tiny checkpoint fixture; a regression
  test that composing a non-UPT model does **not** import `pyupt`
  (mirror `test_palm_target_does_not_import_for_non_palm_composition`).
- Ship a tiny pretrained checkpoint fixture (or train-on-the-fly for 1 step) so
  CI never needs a GPU. Gate any GPU/multi-GPU training tests behind a marker.

## 12. Decisions to confirm before coding

1. **Framework (D1)**: JAX+Equinox (recommended, repo-consistent, differentiable)
   vs PyTorch (matches the official UPT code, mature DDP/FSDP, but a second
   accelerator stack). *Biggest, least-reversible choice.*
2. **Ensemble execution (D2)**: confirm in-process on-device batching as the
   default (vs reusing the forkserver process pool). Recommended: batching.
3. **Source solver for the first corpus**: pylbm (collocated, CUDA, fastest to
   iterate) vs pyudales/pypalm. Recommended: start with pylbm on the Xie–Castro
   geometry already in `examples/lbm/`.
4. **Corpus storage & budget**: Zarr layout + how many trajectories / how much
   GPU-hours for generation, and where checkpoints sync (shared scratch?).

## 13. Phased roadmap

- **P0 — scaffolding**: `libs/pyupt` skeleton, pyproject + pixi feature, lazy-import
  regression test, `state_io`/`params_io` with round-trip tests. No model yet.
- **P1 — data**: `scripts/generate_upt_training_data.py`, Zarr corpus + manifest,
  normalization fit, a small pylbm corpus end-to-end.
- **P2 — model + single-GPU training**: UPT modules, one-step + pushforward loss,
  `scripts/train_upt.py`, checkpoint/manifest, rollout-error eval.
- **P3 — forward model**: `pyupt.ForwardModel` + `EnsembleForwardModel` (D2),
  `conf/model/pyupt.yaml`, `prepare_pyupt`, `clean_outputs` branch, obs-operator
  registration; `run_forward_model.py model=pyupt` works.
- **P4 — multi-GPU**: `training/sharding.py` data-parallel mesh, pixi tasks,
  DelftBlue/SLURM launch; scale the corpus and model.
- **P5 — ESMDA validation**: surrogate-as-assim vs Fortran-truth runs; measure
  posterior accuracy, rollout stability, ensemble dispersion; document the
  uncertainty caveat and any inflation needed.

## 14. Risks / open issues

- **Rollout drift** over long windows — mitigated by pushforward training (§6.1);
  the primary acceptance metric is rollout-error-vs-horizon, not one-step MSE.
- **Under-dispersion** in ESMDA (§10 caveat).
- **Distribution shift**: ESMDA can push params outside the training prior;
  cover margins in §5 sampling and monitor for extrapolation.
- **Data-generation cost** dominates effort (§5); plan cluster time.
- **Staggered-grid information loss** from interpolation (D3) — acceptable for a
  surrogate, but quantify against held-out uDALES data if that solver is used.
```
