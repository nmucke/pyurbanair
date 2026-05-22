# neural_surrogates

An **architecture-agnostic neural-surrogate framework** for pyurbanair. It learns
a fast, GPU-batched surrogate of a CFD solver that drops into the *exact* same
`BaseForwardModel` / `BaseEnsembleForwardModel` / ESMDA machinery as `pylbm`,
`pyudales`, and `pypalm` (see `docs/codebase_guide.md` §3). Once trained, a
checkpoint is used via `model=neural_surrogate` and nothing downstream — ESMDA,
the observation operator, plotting — has to change.

The design separates the **framework** (everything pyurbanair-specific: the I/O
contract, ensemble batching, geometry handling, data generation, training
curriculum, checkpoint format, Hydra wiring) from the **architecture** (the
neural network itself, behind a tiny interface). The framework is written once;
adding a new network means implementing one interface — not re-touching the
forward model, dataset, or ESMDA glue. The first architecture is a **3D
convolutional UNet**; **UPT** is a planned second implementation of the same
interface.

> Full design rationale and phased roadmap:
> [`docs/neural_surrogate_plan.md`](../../docs/neural_surrogate_plan.md).

---

## Quick start — the two central commands

Run from the repo root. Use `-e dev` on CPU/macOS, `-e cuda` on a GPU box.

**1. Generate a training corpus** (the CFD solver *is* the data generator):

```bash
pixi run -e dev python scripts/generate_neural_surrogate_data.py \
  model=pylbm model.forward_model.cuda=false \
  domain=xie_castro_60x40x16 \
  time.simulation_time=300 time.output_frequency=5 time.spinup_time=30 \
  ensemble.ensemble_size=8 ensemble.num_parallel_processes=4 \
  +generate.n_trajectories=200 \
  +generate.corpus_path=.temp/neural_surrogate/xie_castro
```

Add `params/external=time_varying time_varying=ar2_relaxation +generate.time_varying=true`
for transient inflow (time-varying boundary conditions).

**2. Run the training** (architecture is a config choice; UNet by default):

```bash
pixi run -e cuda python scripts/train_surrogate.py \
  corpus_path=.temp/neural_surrogate/xie_castro \
  run_id=lbm_xie_castro_unet3d_v1 \
  history_len=3
```

Equivalent Pixi tasks:

```bash
pixi run -e dev  generate-surrogate-data model=pylbm model.forward_model.cuda=false \
  domain=xie_castro_60x40x16 +generate.corpus_path=.temp/neural_surrogate/xie_castro \
  +generate.n_trajectories=200
pixi run -e cuda train-surrogate corpus_path=.temp/neural_surrogate/xie_castro \
  run_id=lbm_xie_castro_unet3d_v1
pixi run -e cuda train-surrogate-multi corpus_path=... run_id=...   # data-parallel multi-GPU
```

The trainer reads the grid, channels, param schema, and normalization from the
corpus and builds the network to match — you never set runtime dims. The
checkpoint is written to `models/neural_surrogates/<run_id>/`. See
[§4 End-to-end workflow](#4-end-to-end-workflow) for the full pipeline
(sizing → generate → train → GATE → inference → ESMDA) and all knobs.

---

## Table of contents

0. [Quick start — the two central commands](#quick-start--the-two-central-commands)
1. [Core idea: the architecture interface](#1-core-idea-the-architecture-interface)
2. [Library layout](#2-library-layout)
3. [Installation & environments](#3-installation--environments)
4. [End-to-end workflow](#4-end-to-end-workflow)
5. [Data formats and shapes](#5-data-formats-and-shapes)
6. [Checkpoint format](#6-checkpoint-format)
7. [Configuration reference](#7-configuration-reference)
8. [Conditioning & time-varying parameters](#8-conditioning--time-varying-parameters)
9. [Geometry, cold start, masking](#9-geometry-cold-start-masking)
10. [Ensemble batching (D2)](#10-ensemble-batching-d2)
11. [Key design decisions & defaults](#11-key-design-decisions--defaults)
12. [Adding a new architecture](#12-adding-a-new-architecture)
13. [Testing](#13-testing)
14. [Limitations & scope](#14-limitations--scope)

---

## 1. Core idea: the architecture interface

Every architecture is a JAX/[Equinox](https://docs.kidger.site/equinox/) module
implementing a **field-space stepping interface with an opaque carry**. The
carry is a PyTree the architecture owns; the framework never inspects it.

```python
class SurrogateArchitecture(eqx.Module):
    def init_carry(
        self,
        hist_fields: Float[Array, "K C Z Y X"],   # last K history frames (normalized)
        hist_params: Float[Array, "K P"],          # per-frame conditioning (dense)
        hist_mask:   Float[Array, "K"],             # 1 = real frame, 0 = left-pad
        static:      Float[Array, "S Z Y X"],       # baked SDF + mask channels
    ) -> Carry: ...

    def step(
        self,
        carry:      Carry,
        next_param: Float[Array, "P"],              # boundary condition for t -> t+1
        static:     Float[Array, "S Z Y X"],
    ) -> tuple[Float[Array, "C Z Y X"], Carry]:
        """Advance one step: return the decoded field AND the new carry."""
```

Shape conventions are **channels-first, space-last**: `C` velocity/pressure
channels (`u, v, w[, pres]`), collocated grid axes `(Z, Y, X)`, `K` history
frames, `P` conditioning width, `S` static geometry channels.

**Two architectures, one interface:**

- **3D conv UNet** (`architectures/unet3d.py`, the first/default): the carry is a
  **ring buffer of the last `K` decoded fields**. `step` channel-stacks
  `[K·C fields, S static]`, runs the UNet (FiLM-conditioned on `next_param`),
  and returns the predicted field plus the updated ring buffer. At `K = 1` this
  is the plain Markov `(field_t, params) → field_{t+1}`.
- **UPT** (planned): the carry is a ring buffer of the last `K` **latents**;
  `init_carry` encodes history frames to latents, `step` runs the latent
  temporal propagator and decodes. Same interface, no framework changes.

**Why this cut is load-bearing:**

- **Rollout is architecture-agnostic.** `rollout.py` provides one
  `rollout(arch, carry, future_params, static, n_steps)` built on `jax.lax.scan`
  over `step`. The forward model's inference autoregression *and* the training
  pushforward loop call the same helper. Pushforward feeds the model its own
  output automatically because `step` already returns the carry it needs next.
- **Training stays in field space.** The loss compares decoded `step` outputs to
  true frames; latents (for UPT) are recomputed every gradient step, never
  cached.
- **Batching is `jax.vmap` over the ensemble axis** — the natural GPU-batched
  inference strategy (D2), not process forking.

The `rollout` contract is guarded by a dummy-architecture conformance test
(`tests/test_interface.py`) independent of any real network.

---

## 2. Library layout

```
libs/neural_surrogates/
  pyproject.toml
  src/neural_surrogates/
    __init__.py                 # version only; NO top-level NN imports (lazy invariant)
    forward_model.py            # ForwardModel(BaseForwardModel) — arch-agnostic inference
    ensemble_forward_model.py   # EnsembleForwardModel — on-device vmap batching (D2)
    rollout.py                  # rollout() + rollout_from_history() via jax.lax.scan
    architectures/
      base.py                   # SurrogateArchitecture interface + Carry typing
      registry.py               # resolve_architecture(name, config) -> module
      unet3d.py                 # 3D conv UNet (first architecture)
    data/
      generate.py               # CorpusWriter / ZarrCorpus / open_corpus (Zarr corpus)
      dataset.py                # Corpus protocol, InMemoryCorpus, lazy WindowDataset
      normalization.py          # mask-aware per-variable standardization
      grid.py                   # GridMeta, STL->mask voxelization, mask->SDF, static channels
    training/
      loop.py                   # masked pushforward loss, train_step, eval, per-horizon error
      train.py                  # run(cfg): build arch from corpus, curriculum, checkpoint
      conditioning.py           # ParamEmbedding (MLP) + FiLM modulation
      checkpoint.py             # Orbax save/restore + the §7 artifact set
      sharding.py               # data-parallel mesh helpers (multi-GPU)
    utils/
      state_io.py               # xarray <-> [T,C,Z,Y,X] tensor, history, time-axis trim
      params_io.py              # params Dataset -> dense per-step conditioning (sin/cos)
      schema.py                 # ParamSchema / ContractSchema (source-solver contract)
      registry.py               # checkpoint path/run_id resolution + manifest loading
  tests/                        # CPU-only unit + smoke + e2e tests (tiny models)
```

The top-level `neural_surrogates/__init__.py` is deliberately import-light: it
must **not** pull in JAX/Equinox/Optax/Orbax, so composing a non-surrogate Hydra
config never imports the NN stack (the same lazy-import invariant `pypalm`
relies on; enforced by a regression test in `tests/test_hydra_config.py`).

The matching trained-model registry lives at `models/neural_surrogates/<run_id>/`
in the repo root (git-ignored).

---

## 3. Installation & environments

The library is wired in as a Pixi feature (`neural_surrogates`) added to the
`dev`, `cuda`, and `delftblue` environments — **not** `default`, which stays
free of the NN stack. Dependencies: `jax`, `equinox`, `jaxtyping`, `optax`,
`orbax-checkpoint`, `zarr`, `numpy`, `xarray`, `scipy`.

```bash
pixi run setup-dev            # CPU dev env (also handles the coreutils/tempest-remap clash)
# GPU box:
pixi install -e cuda          # Linux + NVIDIA
```

On macOS there is no CUDA — use the `dev` (CPU) env. CPU envs run under
`JAX_PLATFORMS=cpu`.

There are also Pixi tasks:

```bash
pixi run -e dev  generate-surrogate-data ...
pixi run -e cuda train-surrogate ...
pixi run -e cuda train-surrogate-multi ...     # data-parallel
pixi run -e dev  eval-surrogate-gate ...
```

---

## 4. End-to-end workflow

```
priors -> CFD ensembles (pylbm/pyudales/pypalm) -> trajectory corpus (.zarr)
                                                       |
                                  fit normalization + grid/mask metadata
                                                       |
            train <architecture> (single/multi-GPU) -> checkpoint + manifest
                                                       |
   ESMDA / rollout scripts <- neural_surrogates.ForwardModel(checkpoint) <----'
   (same contract as the CFD backends)
```

The corpus, normalization, training loop, and checkpoint are
architecture-agnostic — only the `architecture: <name>` field and its
`arch/*.yaml` block differ between a UNet run and a future UPT run.

### Step 0 — sizing (do this first)

Data generation dominates the effort. Estimate before generating:

- **Storage** ≈ `N_traj · N_frames · nx·ny·nz · n_vars · bytes`. At 64³, 3 vars,
  fp16, 100 frames ≈ 1.5 MB/frame → ~150 MB/traj → ~150 GB for 1000 trajectories
  (fp32 doubles; `+pres` is +33%; 128³ is ×8).
- **Compute** ≈ `N_traj · wall_time_per_traj / workers`. Minutes-per-trajectory
  transient runs at 1000 trajectories push to days on a single box → justifies
  cluster scale-out.

Start the GATE on a *small* corpus; only commit to the full corpus after it
clears.

### Step 1 — generate a corpus

The existing CFD backend **is** the data generator (reuses the ensemble
machinery). One model per (solver, geometry, grid).

```bash
pixi run -e dev python scripts/generate_neural_surrogate_data.py \
  model=pylbm model.forward_model.cuda=false \
  domain=xie_castro_60x40x16 \
  time.simulation_time=300 time.output_frequency=5 time.spinup_time=30 \
  ensemble.ensemble_size=8 ensemble.num_parallel_processes=4 \
  +generate.n_trajectories=200 \
  +generate.corpus_path=.temp/neural_surrogate/xie_castro
```

It samples parameters from `params.prior`, runs the solver ensemble on disk,
converts each trajectory to a `[T,C,Z,Y,X]` tensor (uDALES is interpolated to
the collocated grid first), records per-frame **effective** conditioning,
voxelizes the geometry once, writes a Zarr corpus + manifest, and fits
normalization on the train split.

**uDALES example** (`size=small` domain; uDALES has no `forward_model.stl_path`,
so `generate.stl_path` must be set explicitly):

```bash
pixi run -e dev python scripts/generate_neural_surrogate_data.py \
  model=pyudales size=small \
  time.simulation_time=300 time.output_frequency=1 \
  ensemble.ensemble_size=4 ensemble.num_parallel_processes=4 \
  +generate.n_trajectories=400 \
  +generate.stl_path=examples/udales/experiments/xie_and_castro/xie_castro_2008_STL.stl \
  +generate.corpus_path=.temp/neural_surrogate/corpus_udales
```

**Time-varying inflow** (transient BCs — recommended for ESMDA-relevant data):

```bash
... params/external=time_varying time_varying=ar2_relaxation \
    +generate.time_varying=true
```

This samples transient parameter series from the configured
`parameter_time_series` model and drives the solver with them. The external
prior `mean`/`std` may be **per-window profiles** (lists) so `x_ext(t)` /
`Σ_ext(t)` vary across the window — see
[§8](#8-conditioning--time-varying-parameters).

> `generate.stl_path` resolves the geometry; it defaults to
> `model.forward_model.stl_path` (pylbm) and must be set explicitly for backends
> without one (e.g. uDALES).

### Step 2 — train

The trainer reads the grid, channels, param schema, and normalization from the
corpus and builds the network to match — you never set runtime dims.

```bash
pixi run -e cuda python scripts/train_surrogate.py \
  corpus_path=.temp/neural_surrogate/xie_castro \
  run_id=lbm_xie_castro_unet3d_v1 \
  history_len=3
```

Key knobs (`conf/neural_surrogate/train.yaml`):

- **`history_len`** (`K`): `1` = Markov; raise for temporal context.
- **`horizon_schedule`**: the **pushforward curriculum** — grows the rollout
  horizon `H` over epochs with a stop-gradient warm-up. This is what makes the
  surrogate stable under repeated ESMDA stepping; it is *mandatory* for ESMDA
  usefulness, and applies to every architecture.
- **`off_manifold_noise_std`**: perturbs history ICs during training to harden
  against analysis-OOD states (the GATE B2 bar). `0` disables.
- **`arch.*`**: depth/width in `conf/neural_surrogate/arch/unet3d.yaml`.

Output → `models/neural_surrogates/<run_id>/` (see [§6](#6-checkpoint-format)).
Multi-GPU (data-parallel) via `train-surrogate-multi`.

### Step 3 — GATE (go/no-go before scaling)

```bash
pixi run -e dev python scripts/eval_surrogate_gate.py \
  checkpoint_path=models/neural_surrogates/lbm_xie_castro_unet3d_v1 \
  corpus_path=.temp/neural_surrogate/xie_castro split=val horizon=8
```

Reports three pre-stated bars: (B1) clean rollout-error-vs-horizon, (B2)
analysis-OOD robustness from perturbed ICs (the load-bearing one), (B3)
cold-start sanity. Commit cluster time only if all three pass. The script is
architecture-agnostic — it re-runs unchanged for UPT.

### Step 4 — inference (forward model)

```bash
pixi run -e dev python scripts/run_forward_model.py model=neural_surrogate \
  model.checkpoint_path=models/neural_surrogates/lbm_xie_castro_unet3d_v1 \
  domain=xie_castro_60x40x16 \
  time.simulation_time=300 time.output_frequency=5
```

The composed `domain` **must match the checkpoint grid** (validated at load;
raises on mismatch). With `state=None` the surrogate cold-starts from the canned
IC bank; otherwise it warm-starts from the supplied state. `time.*` only sets
the number of rollout steps (`num_outputs = simulation_time / output_frequency`).

### Step 5 — ESMDA (surrogate as assim, CFD as truth)

```bash
pixi run -e cuda python scripts/run_parameter_esmda.py \
  model@truth_model=pylbm model@assim_model=neural_surrogate \
  assim_model.checkpoint_path=models/neural_surrogates/lbm_xie_castro_unet3d_v1
```

A surrogate ESMDA window costs one batched GPU forward pass instead of N CFD
simulations. The same pattern works for `run_state_and_parameter_esmda.py`,
`run_rollout_esmda.py`, and `run_time_varying_parameters_rollout_esmda.py`. Keep
truth ≠ assim backend (anti-inverse-crime). Do **not** use `preset=test` /
`size=small` with a surrogate — those change `domain` and would mismatch the
checkpoint grid.

---

## 5. Data formats and shapes

### Tensor convention

All field tensors are **channels-first, space-last**: `[T, C, Z, Y, X]` (a
single frame is `[C, Z, Y, X]`). The spatial grid is the **collocated** `z, y, x`
grid (D3); uDALES staggered grids (`zt/yt/xt`, …) are interpolated/renamed to
this before tensors are built (`state_io.to_collocated_dims`).

### Corpus on disk (Zarr)

```
<corpus>/
  manifest.json            # grid, schema, var_names, splits, counts, source solver, git SHA
  geometry.npy             # solid/fluid mask                       [Z, Y, X]      f4
  static.npy               # baked static channels (SDF + mask)     [S, Z, Y, X]   f4
  normalization.json       # per-variable mean/std (train split)
  trajectories/
    traj_00000.zarr/       # group:
      fields               #   raw state, chunked along time        [T, C, Z, Y, X] f4
      params               #   per-frame ENCODED conditioning       [T, P]          f4
      times                #   output-frame times                   [T]             f8
    traj_00001.zarr/
    ...
```

- Trajectories are stored **whole**, never pre-windowed — overlapping windows
  would duplicate frames and the rollout horizon grows during training.
  Windowing is a lazy index map (`data/dataset.py`).
- Geometry is identical across trajectories (fixed geometry, D5) → stored once
  at the corpus root.
- `params` are the **encoded conditioning** (see [§8](#8-conditioning--time-varying-parameters)),
  not raw physical values.

Worked example for `domain=xie_castro_60x40x16`, 3 channels, no pressure,
`simulation_time/output_frequency = 30`: `(Z,Y,X) = (16,40,60)`, `C = 3`,
`P = 3` (`sin`/`cos` of inflow angle + velocity), `S = 2` (SDF + mask),
`T = 30` → `fields [30,3,16,40,60]`, `params [30,3]`, `geometry.npy [16,40,60]`,
`static.npy [2,16,40,60]`.

### Window records (`WindowDataset.__getitem__`)

Lazily sliced from a flat index `sample_id -> (trajectory_id, t_anchor,
history_len)`; each record is architecture-independent:

```
hist_fields    [K, C, Z, Y, X]   # K history frames; left-padded if history_len < K
hist_params    [K, P]            # dense per-step params at those frames
hist_mask      [K]               # 1 = real frame, 0 = left-pad
future_params  [H, P]            # boundary conditions for the H rollout steps
target_fields  [H, C, Z, Y, X]   # next H *true* frames (pushforward targets)
```

`H` is curriculum-controlled (bumping it just recomputes the index); splitting
is by trajectory before the index is built; short-history start windows are
explicit index entries (so the cold-start regime is actually trained).

### Normalization

Per-variable mean/std, fit **mask-aware** (solid cells excluded) on the
**train split only**, stored in the checkpoint, applied at inference, never
recomputed on assimilation data.

---

## 6. Checkpoint format

A checkpoint is **weights + everything needed to reproduce inference**,
including the architecture identity:

```
models/neural_surrogates/<run_id>/
  weights/               # Orbax checkpoint (the inexact-array param leaves)
  architecture.json      # {"name": "unet3d", "config": {...}, "history_len": K}
  normalization.json     # per-variable mean/std
  grid.json              # nx/ny/nz/bounds  (validated against the composed domain)
  geometry.npy           # baked solid/fluid mask
  static.npy             # baked static channels (SDF + mask)
  schema.json            # source_solver_name, param_schema, state_var_names, dtype
  ic_bank.npz            # canned cold-start ICs: params [N,P], fields [N,C,Z,Y,X]
  manifest.json          # created_at, architecture, history_len, source solver, corpus path
  metrics.json           # train/val loss, rollout-horizon errors
```

- `architecture.json` lets the loader reconstruct the exact network via
  `architectures/registry.py`; load asserts the requested architecture matches.
- `schema.json` is load-bearing: `ForwardModel` and `params_io` use it to decide
  which params are required and which state variables are emitted — so a
  uDALES-trained checkpoint still receives `pressure_gradient_magnitude` while a
  pylbm-trained one does not, **independent of `model.name`**.
- Resolve a checkpoint by explicit path or `run_id` (`utils/registry.py`).
- `models/neural_surrogates/` is git-ignored; sync weights to an external store.

---

## 7. Configuration reference

### Inference: `conf/model/neural_surrogate.yaml`

```yaml
name: neural_surrogate
solver_name: neural_surrogate     # collocated mapping in the obs operator
checkpoint_path: ???              # required
forward_model:
  _target_: neural_surrogates.forward_model.ForwardModel
  checkpoint_path: ${..checkpoint_path}
  nx/ny/nz/bounds: ${domain.*}    # VALIDATED against the checkpoint, not used to size the net
  simulation_time/output_frequency: ${time.*}
  device: cuda                    # or cpu
  cold_start: canned              # canned IC bank | raise | zeros
prepare:
  _target_: pyurbanair.config.hydra_helpers.prepare_neural_surrogate
ensemble_model:
  _target_: neural_surrogates.ensemble_forward_model.EnsembleForwardModel
  num_parallel_processes: 1       # default in-process on-device batching (D2)
```

The grid **and** architecture come from the checkpoint, not from config —
`nx/ny/nz/bounds` are passed only so `prepare_neural_surrogate` can validate the
composed `domain` and raise on mismatch.

### Training: `conf/neural_surrogate/`

- `train.yaml` — optimizer, batch size, `history_len` (`K`), `horizon_schedule`
  (pushforward curriculum), `off_manifold_noise_std`, `corpus_path`, `run_id`,
  `checkpoint_dir`, sharding. Defaults `arch: unet3d`.
- `arch/unet3d.yaml` — `base_channels`, `channel_multipliers` (length =
  number of UNet levels), `num_res_blocks_per_level`, `norm`, `groups`,
  `activation`, `embed_dim`, `residual`. Snapshotted into the checkpoint.
- `data.yaml` — corpus-generation knobs (read under the `generate.*` namespace
  by the data script): `corpus_path`, `n_trajectories`, `time_varying`,
  `stl_path`, split fractions, geometry-static toggles.
- `gate.yaml` — GATE bars (`b1`, `b2`, `cold_start_max`, `horizon`, …).

Grid divisibility: the UNet's pooling needs `nx/ny/nz` divisible by
`2^(num_levels-1)`; the model pads (edge) and crops otherwise.

---

## 8. Conditioning & time-varying parameters

**The framework interpolates; the architecture embeds** (the conditioning split
is clean). `utils/params_io.params_to_conditioning` turns a params `Dataset`
into a dense per-step conditioning sequence `[T, P]`:

- a sparse `time`-dim series is **linearly interpolated** onto the dense
  per-step rollout grid (matching how the CFD solvers interpolate sparse inflow
  series at runtime);
- `inflow_angle` is encoded via **sin/cos** (two channels) so the 359°→1° wrap
  doesn't sweep;
- scalar params broadcast to every step;
- which params are included comes from the checkpoint's `ParamSchema`, not
  `model.name`.

So for pylbm `P = 3` (`sin, cos, velocity`); for uDALES `P = 4` (`+ pressure
gradient`). The architecture only ever receives `next_param` ready to embed
(FiLM for the UNet via `training/conditioning.py`).

**Time-varying external priors.** The `parameter_time_series` models
(`ar2_relaxation`, `ar1`, `ornstein_uhlenbeck`, `gp_linear_trend`) accept
external prior `mean`/`std` that are **scalars (constant) or lists of control
points** spaced evenly over the window and linearly interpolated. Each model
draws a unit-variance anomaly `z` and applies a `mean(t) + std(t)·z(t)` envelope
— so `x_ext(t)` / `Σ_ext(t)` can vary over the window while a scalar spec
reproduces the previous behavior exactly. Example:

```yaml
# conf/params/external/time_varying.yaml
inflow_angle:
  mean: [-30.0, 0.0, 30.0]   # sweep the prior-mean inflow direction across the window
  std:  [3.0, 8.0, 3.0]      # widest spread mid-window
velocity_magnitude:
  mean: [6.0, 9.0, 7.0]
  std:  [0.5, 1.0, 0.5]
  min: 0.1
```

Used by the data generator (`+generate.time_varying=true`) to teach transient
boundary conditions, and by the time-varying ESMDA scripts. Between-window
extrapolation reduces a profile to its mean.

---

## 9. Geometry, cold start, masking

- **Geometry (D5)** is a static, baked-in input — one model per (solver,
  geometry, grid). The STL is voxelized once into a solid/fluid mask on the
  collocated grid (reusing `pylbm.stl_to_lbm.get_building_grid_indices`, so the
  mask matches where the solver places solid cells), and a signed-distance field
  (SDF) is derived from it. Both are stacked into the `static` channels fed every
  step (`data/grid.py`).
- **Cold start (`state=None`).** Default `cold_start=canned`: the checkpoint
  ships a small IC bank (the first frame of each train trajectory keyed by its
  encoded conditioning); `state=None` picks the nearest by conditioning. Other
  modes: `raise` (require an IC) and `zeros`.
- **Solid-cell masking.** After decoding, velocity channels (`u, v, w`) are
  re-zeroed in solid cells (no-penetration). Pressure (`pres`) is **not** masked.

---

## 10. Ensemble batching (D2)

`EnsembleForwardModel` **overrides** `run_ensemble` to batch on-device instead of
forking N processes (which is right for CPU-bound Fortran but wrong for a GPU
NN). It:

- streams members in sub-batches sized by `vmap_chunk_size` (default = full
  ensemble; lower it when memory-bound) and `jax.vmap`s the rollout over the
  ensemble axis;
- honors the **full** `run_ensemble` contract: all three save modes,
  `state` arriving as a `pathlib.Path` to per-member files (loaded lazily via
  `get_member_state`), and the `rollout_step` increment;
- writes per-member `{sim_name}_{i}.nc` on disk or returns a
  concat-along-`ensemble` Dataset in memory.

`num_parallel_processes > 1` remains available only as an optional CPU-smoke
fallback (delegates to the base process-fork path).

---

## 11. Key design decisions & defaults

| Decision | Choice |
|---|---|
| Framework (D1) | JAX + Equinox + Optax + Orbax (repo is JAX-centric; `vmap` *is* the batching path) |
| Ensemble execution (D2) | In-process on-device `vmap` batching (default) |
| Grid / staggering (D3) | Train & predict on the collocated `x,y,z` grid; uDALES interpolated first |
| Temporal context (D4) | A `K`-frame history maintained by the framework; architectures consume it however they like (`K=1` Markov default for the UNet) |
| Conditioning (1.5) | Framework interpolates sparse→dense + sin/cos; architecture embeds |
| Geometry (D5) | STL voxelized offline to mask + SDF static channels |
| Cold start | Canned IC bank (default), or `raise` / `zeros` |
| First architecture | 3D conv UNet (config-driven); UPT later, same interface |

Architectures are deterministic and stateless between calls (no batch
norm/dropout/mutable state) unless the interface is explicitly extended.

---

## 12. Adding a new architecture

1. Implement `SurrogateArchitecture` (`init_carry` + `step`) as an `eqx.Module`
   under `architectures/`. Special-case `K = 1` to skip multi-frame machinery.
2. Register a factory in `architectures/registry.py`
   (`{"unet3d": ..., "your_arch": ...}`).
3. Add `conf/neural_surrogate/arch/<your_arch>.yaml` (its hyperparameters;
   snapshotted into the checkpoint). Runtime dims (`in_state_channels`,
   `static_channels`, `history_len`, `param_dim`) are injected by the trainer.
4. Train with `arch=<your_arch>`. **No** changes to the dataset, training loop,
   forward model, ensemble model, or ESMDA glue.
5. Re-run the CPU smoke stage and the GATE on the existing corpus, then compare
   against the UNet.

The `rollout` conformance test plus the CPU smoke stage exercise the contract
for any architecture before the GATE.

---

## 13. Testing

All tests are CPU-only and use tiny models / synthetic corpora — no GPU, no
committed binaries, no solver runs.

```bash
pixi run -e dev python -m pytest libs/neural_surrogates/tests -q
pixi run -e dev python -m pytest tests/test_neural_surrogate_forward.py -q
```

- `test_io_roundtrip.py` — state↔tensor and params→conditioning round-trips,
  time-less warm-start frames, schema-driven pressure-gradient inclusion, trim.
- `test_interface.py` — dummy-architecture `init_carry`/`step`/`rollout` shapes,
  `K=1` fast path, `vmap` over the ensemble.
- `test_unet3d.py` — grid-divisibility / pad-crop, `K=1`, residual init.
- `test_smoke_training.py` — the §11.1 CPU smoke stage, parametrized over
  `(T, grid, K, H, C)`; plumbing invariants only (shapes, finiteness, params
  change) + a checkpoint round-trip.
- `test_train_run.py` — full Zarr-corpus → curriculum training → checkpoint.
- `test_ensemble_forward.py` — D2 batching: in-memory concat, on-disk files,
  chunked == unchunked.
- `test_review_fixes.py` — uDALES dim renaming, `hist_mask` handling, pressure
  not masked.
- `tests/test_neural_surrogate_forward.py` (repo level) — end-to-end
  `run_forward_model` against a train-on-the-fly tiny checkpoint, plus grid
  mismatch raising; and the lazy-import regression test in `test_hydra_config.py`.

---

## 14. Limitations & scope

- **Single geometry.** A checkpoint is valid only for the (solver, geometry,
  grid) it was trained on; a different geometry is out-of-distribution.
- **Rollout drift** over long windows — mitigated by pushforward training; the
  primary acceptance metric is rollout-error-vs-horizon, not one-step MSE.
- **Analysis-OOD** — the EnKF analysis can produce states off the learned
  manifold; mitigated by off-manifold IC injection (GATE B2).
- **Under-dispersion** — a deterministic surrogate under-disperses in ESMDA;
  spread comes from the parameter/IC ensembles, so treat calibrated spread as a
  pass/fail bar (inflation / ensemble-of-surrogates if needed).
- **Parameter distribution shift** — ESMDA can push params outside the training
  prior; cover margins generously when sampling the corpus.
- **Status:** P0–P3 + the GATE eval are implemented and CPU-tested; multi-GPU
  sharding (P4) and UPT (P-UPT) are stubbed/registered for later. See the plan
  doc for the phased roadmap.
