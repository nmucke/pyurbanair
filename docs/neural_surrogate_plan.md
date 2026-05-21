# Plan: architecture-agnostic neural surrogate library (`libs/neural_surrogates`)

Status: **P0–P3 + P5-gate landed** (framework, UNet3D, training, inference,
Hydra wiring, GATE eval). The data-generation (P1) and training/eval scripts run
on the existing solver/cluster; multi-GPU sharding (P4) and UPT (P-UPT) are
stubbed/registered for later. Library lives in `libs/neural_surrogates`; CPU
tests in `libs/neural_surrogates/tests` + `tests/test_neural_surrogate_forward.py`.

This document supersedes [`docs/upt_surrogate_plan.md`](upt_surrogate_plan.md). That
plan packaged a single architecture (Universal Physics Transformers) as a backend
`pyupt`. This plan keeps **every pyurbanair-specific requirement from it** but
generalizes the deliverable: a `neural_surrogates` library that owns the
**forward-model contract, dataloading, training, checkpointing, and ESMDA glue**,
and treats the neural network itself as a **pluggable architecture**. The first
architecture is a **3D convolutional UNet** (simple, fast to stand the framework up
and test end-to-end); **UPT is added later** as a second implementation of the same
interface, reusing the §1.0 fidelity discipline from the old plan.

The surrogate drops into the existing `BaseForwardModel` / `BaseEnsembleForwardModel`
machinery alongside `pylbm`, `pyudales`, `pypalm` (see `docs/codebase_guide.md` §3),
so ESMDA, the observation operator, and plotting are unchanged.

## 0. Scope and design goal

Confirmed requirements (unchanged from the UPT plan): **fixed domain, varying
inflow** + **transient time-series output** feeding multi-window ESMDA
([`BaseRolloutForwardModel`](../src/pyurbanair/base_rollout_forward_model.py),
`scripts/run_rollout_esmda.py`). One trained model per (solver, geometry, grid)
tuple; geometry generalization is out of scope (§14).

Scope of the surrogate (identical contract to the Fortran backends):
- **Inputs**: an initial state (`u,v,w[,pres]` on the solver grid) + scalar
  parameters (`inflow_angle`, `velocity_magnitude`, and for uDALES-trained
  checkpoints `pressure_gradient_magnitude`). The incoming `state` may be either a
  time-indexed history (`time` length ≥1) or a single time-less warm-start frame,
  because current rollout ESMDA passes `isel(time=-1)` frames between windows.
- **Output**: a time-indexed `xarray.Dataset` with the same grid axes and variables
  as the source solver.

Every checkpoint records the **source solver contract** (`source_solver_name` +
`param_schema` + `state_var_names`) separately from the pyurbanair backend name
`neural_surrogate`. Inference uses that schema to decide which params are required
and which state variables are emitted. Do not key uDALES-only behavior off
`model.name == "pyudales"` once the solver is represented by a surrogate.

**New design goal: separate the framework from the architecture.** Everything that
is pyurbanair-specific (the I/O contract, ensemble batching, geometry handling, data
generation, training curriculum, checkpoint format, Hydra wiring) lives in a shared
**framework layer** and is written **once**. The neural network is an **architecture
layer** behind a small interface (§1.1). Adding UPT later, or any other model, means
implementing that interface — not re-touching the forward model, dataset, or ESMDA
glue.

## 1. Key architectural decisions (read these first)

### 1.1 — The architecture abstraction (the load-bearing new decision)

Every architecture is a JAX/Equinox module implementing a **field-space stepping
interface with an opaque carry**. The carry is a PyTree the architecture owns; the
framework never inspects it.

```python
class SurrogateArchitecture(eqx.Module):
    """Contract every neural architecture implements. Field-space in/out;
    the architecture owns whatever internal state it carries between steps."""

    def init_carry(
        self,
        hist_fields: Float[Array, "K C Z Y X"],   # last K history frames (normalized)
        hist_params: Float[Array, "K P"],          # per-frame conditioning (dense, §1.5)
        hist_mask:   Float[Array, "K"],            # 1 = real frame, 0 = left-pad
        static:      Float[Array, "S Z Y X"],      # baked SDF + mask channels (D5)
    ) -> Carry:
        """Build the initial autoregressive state from the K-frame history."""

    def step(
        self,
        carry:       Carry,
        next_param:  Float[Array, "P"],            # boundary condition for t -> t+1
        static:      Float[Array, "S Z Y X"],
    ) -> tuple[Float[Array, "C Z Y X"], Carry]:
        """Advance one step. Return the decoded field prediction AND the new carry."""
```

Two architectures, one interface:
- **3D conv UNet** (§ first architecture): `Carry` is a **ring buffer of the last K
  *fields*** (+ nothing else). `init_carry` stores the K frames; `step`
  channel-stacks `[K·C fields, static, broadcast(next_param)]`, runs the UNet, and
  returns the predicted field plus a carry with that field appended and the oldest
  dropped. At `K = 1` this is the plain Markov `(field_t, params) → field_{t+1}`.
- **UPT** (later): `Carry` is a **ring buffer of the last K *latents***. `init_carry`
  **encodes** each history frame to a latent (cheap to carry); `step` runs the
  temporal propagator over the latent window conditioned on `next_param`, **decodes**
  to a field, and appends the newly-propagated latent to the buffer. This is exactly
  the K-frame latent-stepping design from the old plan (`upt_surrogate_plan.md`
  D4-ii), now expressed as one implementation of this interface.

Why this interface is the right cut:
- **Rollout is architecture-agnostic.** The framework provides one
  `rollout(arch, carry, future_params, static, n_steps)` built on `jax.lax.scan`
  over `step`. The forward model's inference autoregression (§4) and the training
  pushforward loop (§6) call the *same* helper. Pushforward feeds the model its own
  output automatically, because `step` already returns the carry it needs next (a
  predicted field for the UNet, a propagated latent for UPT) — the loop never has to
  know which.
- **Training stays in field space and never caches latents.** The loss compares
  decoded `step` outputs against true frames (§6.1). For UPT, `init_carry`
  re-encodes raw history frames *inside* the train step, so encoder-weight changes
  are seen every gradient step — no stale-latent problem, and the dataset emits raw
  fields for both architectures (§6.1.1).
- **Conditioning split is clean.** The framework does the sparse→dense param
  interpolation and `inflow_angle` sin/cos encoding (§1.5 / params_io); the
  architecture does the *embedding* (FiLM for the UNet, tokens for UPT) inside
  `step`/`init_carry`. `next_param` arrives ready to embed.
- **Static geometry is uniform.** Both architectures receive the baked SDF+mask
  `static` channels (D5) every step; how they fuse them is internal.

Architectures are selected by name (`architecture: unet3d`) via a small registry and
**baked into the checkpoint manifest** (§7), since the choice fixes the weights'
meaning. The framework validates the requested architecture against the checkpoint at
load time.

> **K=1 fast path.** Architectures should special-case `K = 1` to skip any
> history-attention / multi-frame-stacking machinery, so the simplest model (and the
> UNet default) carries no temporal overhead. For UPT this is the exact Markov
> reduction that the §11 parity test relies on.

### D1 — Framework: JAX + Equinox/Flax (unchanged recommendation)
Same reasoning as the old plan: the repo is JAX-centric (`jax>=0.7` top-level, ESMDA
in JAX, `forkserver` chosen *because* JAX threads at import,
[`base_ensemble_forward_model.py:417`](../src/pyurbanair/base_ensemble_forward_model.py#L417)),
and the natural ensemble parallelism (D2) is `jax.vmap` over the `ensemble` axis — so
JAX *is* the batched-inference strategy. Recommended: **Equinox** (PyTree-native
modules — the `SurrogateArchitecture` above is an `eqx.Module`) + **Optax** + **Orbax**.

A 3D conv UNet is trivial in either framework; UPT's reference is PyTorch and porting
it to JAX is the real cost (the §1.0/§11 parity discipline below). Choosing JAX means
the **UNet is native and the UPT port is the work**, not the reverse. Recommend JAX
unless the eventual UPT cross-framework port is judged the dominant risk. Do **not**
justify JAX by "differentiating through the forward model in ESMDA" — ESMDA is
derivative-free; end-to-end autodiff only pays off for a future gradient-based
assimilator (4D-Var), not this repo today. **Decision to confirm — §12.**

### D2 — Ensemble parallelism: batch on-device, do NOT fork N processes
Unchanged and architecture-independent. The existing parallel path
([`_run_parallel`](../src/pyurbanair/base_ensemble_forward_model.py#L395)) is built for
CPU-bound Fortran subprocesses (forkserver + CPU pinning, DRAM-capped at ~4 workers,
`docs/ensemble_scaling.md`). For a GPU NN that is wrong — N processes contend for one
GPU. `neural_surrogates.EnsembleForwardModel` **overrides** `run_ensemble` to:
- Stream per-member initial states and params into batched arrays chunk by chunk.
- Run the batched autoregressive `rollout` (§1.1) **in sub-batches sized to GPU
  memory** via a `vmap_chunk_size` (members per device pass); default = full
  ensemble, lower it when memory-bound. `ensemble × grid × (K fields or latents)` can
  exceed device memory.
- For each chunk, load only those members' `state`/params via `get_member_state` /
  `get_member_params`, move the chunk to device, run the rollout, and immediately
  append to the in-memory result list or write per-member NetCDF files. Do **not** stack
  the full ensemble in host RAM before chunking, because ESMDA may pass `state` as a
  `Path` to on-disk per-member files.
- Re-split into a `concat`-along-`ensemble` `xarray.Dataset`.
- Keep `num_parallel_processes` only as an optional CPU-smoke fallback; default to
  in-process batching.

The override must honor the **full** `run_ensemble` contract, not just the in-memory
happy path (§4): all three save modes, `state` arriving as a `pathlib.Path` to
per-member files, and the `rollout_step` increment. Only the *execution mechanism*
changes (process-fork → `vmap`).

### D3 — Common grid / staggering (unchanged)
Train and predict on the **collocated `x,y,z` grid**. pylbm/pypalm already collocated;
pyudales staggered (`xt/xm`, …) → `pyudales.utils.grid_utils.interpolate_grid` to a
common grid *before* building training tensors, and register `neural_surrogates` in the
obs operator with the collocated mapping (§8.4). A 3D conv UNet in particular **needs**
a regular collocated grid — this decision is now doubly motivated.

### D4 — Temporal context: a K-frame history is a *framework* concept
The framework maintains the K-frame history (the ring buffer fed to `init_carry`,
grown across a rollout from a single IC). It is **architecture-agnostic**: the UNet
consumes the K frames by channel-stacking, UPT by latent attention. Variable length
(steps `1…K` early in a window / at cold start) is handled by left-padding +
`hist_mask`, and architectures special-case `K=1` (§1.1). `K` is a framework
hyperparameter, baked per checkpoint (§7).

> The old plan's D4-ii (latent ring buffer + temporal transformer) is now **just the
> UPT architecture's implementation** of `init_carry`/`step`. The old plan's D4-i
> (sparse→dense per-step param interpolation in the forward model, matching the
> Fortran solvers) is **framework** behavior — see §1.5.

### 1.5 — Conditioning: framework interpolates, architecture embeds
The Fortran solvers accept a *sparse* time-varying parameter series and interpolate
between values at runtime (e.g. pylbm's
[`write_uvel_time_file`](../libs/pylbm/src/pylbm/utils/params_utils.py#L119) +
`m_inflow.F90`). The framework replicates this in `utils/params_io`, **not** in any
network: a params `Dataset` whose vars carry a `time` dim of `N ≪ num_outputs` sparse
values is **interpolated onto the dense per-step rollout grid**, producing one
conditioning vector per step. Details:
- **Linear** interpolation (`xarray.Dataset.interp` / `numpy.interp`).
- Interpolate `inflow_angle` via **sin/cos** so the 359°→1° wrap doesn't sweep, then
  hand the encoded scalars to the architecture.
- Replicate the **spin-up plateau** convention if spin-up frames are emitted.
- Scalar params (no `time` dim) broadcast to every step.

The network always receives `next_param` (and the K per-frame params) ready to embed;
it never sees the sparse series.

### D5 — Geometry: STL voxelized to a static occupancy/SDF channel (unchanged)
Geometry generalization is out of scope (§14): one model per (solver, geometry, grid).
So geometry is a **static, baked-in input**, never fed STL-wise at inference:
- Voxelize the STL **once, offline**, per (geometry, grid), into a solid/fluid mask on
  the collocated grid (D3), reusing
  [`pylbm.stl_to_lbm.get_building_grid_indices`](../libs/pylbm/src/pylbm/stl_to_lbm.py#L120)
  / `python_udgeom`, so the mask is bit-identical to where the source solver placed
  solid cells.
- **Prefer a signed-distance field (SDF)** as the static channel (smooth wall-proximity
  gradients, pools/convolves more gracefully than a 0/1 edge); keep the binary mask
  alongside for loss-masking and output re-application. (Decision flagged in §12.)
- **Three uses**: (1) static input channel(s) (SDF [+ mask]) every step; (2) loss mask
  (solid cells excluded from per-variable MSE, §6.1); (3) output re-application
  (re-zero velocity in solid cells / no-penetration after decode). A validation check
  asserts the surrogate's solid cells coincide with where the Fortran output is
  identically zero.
- **Storage**: in the checkpoint (`geometry.npy` + STL path/SHA + `nx/ny/nz/bounds` in
  the manifest, §7) **and** once at the corpus root (§5). `prepare_neural_surrogate`
  validates composed geometry/grid against the checkpoint, raising on mismatch.

## 2. Library layout (`libs/neural_surrogates`)

Mirror the `pylbm` shape (`docs/codebase_guide.md` §8 recipe). The **`architectures/`**
subpackage is the only place network code lives; everything else is shared framework.

```
libs/neural_surrogates/
  pyproject.toml                    # editable install + pixi pkg (§9)
  src/neural_surrogates/
    __init__.py                     # version only; NO top-level NN imports (lazy invariant)
    forward_model.py                # ForwardModel(BaseForwardModel) — arch-agnostic   (§4)
    ensemble_forward_model.py       # EnsembleForwardModel(...) override — D2 batching  (§4)
    rollout.py                      # rollout(arch, carry, future_params, static, n)    (§1.1)
    architectures/
      base.py                       # SurrogateArchitecture interface + Carry typing    (§1.1)
      registry.py                   # name -> architecture class ("unet3d", later "upt")
      unet3d.py                     # 3D conv UNet — FIRST architecture (config-driven)  (§1.2)
      upt/                          # UPT — LATER (faithful port; see §1.6 + old plan §1.0)
        upt.py                      #   encoder -> propagator -> decoder
        encoder.py                  #   supernode msg-passing + perceiver pool (port)
        propagator.py               #   latent time-stepper; K=1 = ref Markov (port/new)
        decoder.py                  #   perceiver cross-attn to query positions (port)
        layers.py                   #   attention/MLP/pos-enc, match ref numerics
    data/
      generate.py                   # solver ensembles -> trajectory corpus (§5)
      dataset.py                    # lazy index-map windowing -> field-space windows (§6.1.1)
      normalization.py              # fit/apply per-variable standardization (§6.3)
      grid.py                       # grid metadata, collocation; STL->mask+SDF (D5)
    training/
      train.py                      # def run(cfg) + @hydra.main; single/multi-GPU (§6,§7)
      loop.py                       # train_step, pushforward/rollout loss, eval (§6.1)
      conditioning.py               # shared param-embedding helpers (FiLM/MLP) (§6.2)
      sharding.py                   # jax mesh / data-parallel helpers (§6.4)
      checkpoint.py                 # Orbax save/restore + manifest (§7)
    utils/
      state_io.py                   # xarray.Dataset <-> tensor; K-frame history; trim (§4)
      params_io.py                  # params Dataset -> conditioning; sparse->dense (§1.5)
      registry.py                   # resolve model id / checkpoint dir
  tests/                            # unit tests local to the lib (tiny model)
models/neural_surrogates/           # trained-model registry (git-ignored)  (§7)
```

### 1.2 — First architecture: 3D convolutional UNet (configurable from the start)
`architectures/unet3d.py` implements `SurrogateArchitecture` with a standard encoder/
decoder UNet over the 3D grid. **Config-driven** from day one (per the chosen scope),
read from `conf/neural_surrogate/arch/unet3d.yaml` (§8.3) and snapshotted into the
checkpoint manifest:
- `base_channels`, `channel_multipliers` (depth/width per level), `num_levels`,
  `num_res_blocks_per_level`, `norm` (`group`/`none`) + `groups`, `activation`,
  optional `attention_at_levels` (cheap toggle, off by default). **P2 keeps
  architectures deterministic and stateless**: no batch norm, dropout, or mutable layer
  state unless the `SurrogateArchitecture` contract is explicitly extended with
  mode/RNG/state plumbing.
- **History**: `K`-frame input by channel-stacking (`init_carry` holds K fields; `step`
  concatenates `[K·C, S static, P broadcast-param]` → in-channels). `K=1` skips the
  stacking (Markov fast path, §1.1).
- **Conditioning**: per-step `next_param` embedded via an MLP and injected as **FiLM**
  (per-channel scale/shift) at each level (helpers in `training/conditioning.py`).
- **Output**: predicts the next field (residual `Δfield` option — predict the update
  and add to the last input frame — recommended for stability; config flag).
- **Grid divisibility**: UNet pooling needs `nx/ny/nz` divisible by `2^num_levels`;
  `init_carry`/the model pad-and-crop or assert at construction. Document this against
  the §8.1 grid-validation check.

This is intentionally not claimed optimal — its job is to exercise and prove the whole
framework (data → train → checkpoint → forward model → ESMDA) end-to-end cheaply.

### 1.6 — Adding UPT later (faithful-port discipline)
When UPT is added under `architectures/upt/`, the **§1.0 fidelity discipline of the old
plan applies verbatim to the encoder and decoder**: mirror `ml-jku/UPT` module-for-
module, match hyperparameters/numerics, and gate with the **encoder/decoder + K=1
parity test** (§11). The propagator's K-frame temporal attention is a *new design*
validated by behavior (the GATE, §13), not parity; `K=1` reduces exactly to the
reference Markov approximator. None of this touches the framework — UPT is one more
`SurrogateArchitecture`. See [`upt_surrogate_plan.md`](upt_surrogate_plan.md) §1.0 and
§D4-ii for the full fidelity rationale, reused unchanged.

## 3. Data flow at a glance

```
priors -> solver ensembles (pylbm/pyudales/pypalm) -> trajectory corpus (.zarr)
                                                            |
                                        fit normalization + grid/mask metadata
                                                            |
              train <architecture> (single/multi-GPU) -> checkpoint + manifest
                                                            |
   ESMDA / rollout scripts <- neural_surrogates.ForwardModel(checkpoint) <-----'
   (same contract as Fortran backends)
```

The corpus, normalization, training loop, and checkpoint are **architecture-agnostic** —
only the `architecture: <name>` field and its `arch/*.yaml` block differ between a UNet
run and a UPT run.

## 4. Forward-model contract mapping

`neural_surrogates.ForwardModel(BaseForwardModel)` implements the four abstract methods
(`docs/codebase_guide.md` §3), **independent of architecture** (it instantiates whatever
the checkpoint names):

| Base method | behavior |
|---|---|
| `__init__` | Load checkpoint (weights + architecture name/config + normalization + grid/mask) via `utils/registry`. Store `simulation_time`, `output_frequency` → `num_outputs = round(simulation_time/output_frequency)`. `super().__init__(results_dir=...)`. **Lazy-init** the accelerator + architecture on first `run_single` so forkserver workers stay importable. |
| `run_single(state, params, sim_name)` | (1) `params` → **per-step** conditioning sequence (`utils/params_io`, §1.5). (2) resolve initial field(s) (see **cold-start** below) → normalized history → `arch.init_carry(...)` (D4). (3) `rollout.rollout(arch, carry, future_params, static, num_outputs)` autoregressively (§1.1). (4) decode is inside `step`; denormalize, reapply the building mask, assemble an `xarray.Dataset` with `time` + source grid coords per the **time-axis contract** below. Return it. |
| `_apply_inflow_settings(params)` | Build/store the per-step conditioning **sequence** on `self` (no files to edit). Sparse `time`-dim params interpolated to dense per-step grid (§1.5: linear; angle via sin/cos; spin-up plateau); scalars broadcast. |
| `save_results` / `_clean_output` | Implement a concrete `save_results(...)` method that delegates to `_save_results(...)` (even though `BaseForwardModel.__call__` currently calls `_save_results` directly, the abstract method should still be satisfied clearly). `_clean_output` = no-op. |

Notes:
- **Warm start / rollout**: the multi-window pattern feeds each window's final `state`
  into the next `run_single`; the surrogate consumes it as the new IC. No restart files.
  With the K-frame history, `state_io` seeds the buffer with up to the last `K` frames
  of an incoming `time>1` state so cross-window context is preserved.
- **Time-less warm-start frames**: `state_io` accepts both a Dataset with `time` and a
  single-frame Dataset without it, internally normalizing to `[T,C,Z,Y,X]` before
  extracting the K-frame history. This is required for current rollout ESMDA, which
  passes posterior states with `time` stripped.
- **`get_states` / on-disk mode** work unchanged via the base class.

**Cold-start initial condition (decision — §12).** A NN cannot spin up a physical field
from `state=None`, and ESMDA window 0 frequently passes `state=None`. Options (unchanged
from the old plan):
- **(a) Require an IC** — `run_single` raises on `None`; caller supplies a spun-up field.
- **(b) Canned IC bank** — ship spun-up fields keyed by parameter regime in the
  checkpoint; `state=None` selects nearest by `params`. **Recommended default** (cheap,
  unit-testable without extra training, matches how window-0 ICs are produced).
- **(c) Params→initial-field/latent head** — a *second generative model* whose IC error
  compounds with rollout drift; defer to its own phase only if (b) is insufficient.

Recommended: **(b)** default, **(a)** as the P0/P3 stopgap, **(c)** only if needed.

**Time-axis contract (must match the Fortran backends exactly).** pylbm concatenates
`out_*_F<t>.nc` and **drops the spin-up-output prefix**, trimming to `simulation_time /
output_frequency` outputs (`docs/codebase_guide.md` §7). The
`TemporalObservationOperator` then aggregates in fixed `interval_size` chunks, so the
surrogate's output `time` length **and** spacing must equal the source backend's
*after trimming*. §5 training data includes the spin-up transient, but `run_single`
emits the **trimmed** window (`num_outputs` frames at `output_frequency` spacing).
`utils/state_io` owns this trimming; a round-trip test asserts the `time` coord matches
the Fortran backend's on an identical config.

`neural_surrogates.EnsembleForwardModel(BaseEnsembleForwardModel)`: implements
`_create_new_forward_model` (share immutable weights, clone only per-member result dirs)
**and** overrides `run_ensemble` for on-device batching (D2). The override reproduces the
base method's branching ([`run_ensemble`](../src/pyurbanair/base_ensemble_forward_model.py#L482)):
- **All three save modes** — `save_on_disk` writes per-member `{sim_name}_{i}.nc` into
  `self.results_dir` (so ESMDA's `step_{i}/state_*.nc` re-open via `get_state`/
  `get_states` keeps working); `save_in_memory` returns the `concat`-along-`ensemble`
  dataset. Chunking streams through members instead of preloading the full ensemble.
- **`state` may be a `pathlib.Path`** — chunk-load via the existing `get_member_state`
  (handles `Dataset` | `Path` | `None`) before forming each batched array.
- **`rollout_step` increment** ([L461-464](../src/pyurbanair/base_ensemble_forward_model.py#L461))
  still advances per call.
- **Failure policy** is largely moot (a NN forward pass doesn't raise
  `CalledProcessError`); keep `"raise"` default, skip resample plumbing.

## 5. Generating simulated training data

Principle (unchanged): **the existing Fortran backends are the data generator.** Reuse
the ensemble machinery rather than inventing a runner. Because the corpus is
architecture-agnostic, the *same* corpus trains the UNet now and UPT later.

New script `scripts/generate_neural_surrogate_data.py` (`def run(cfg)` + thin
`@hydra.main`):
1. Build the source solver + ensemble like
   [`run_ensemble_forward_model.py`](../scripts/run_ensemble_forward_model.py).
2. Sample a large parameter set from the prior — reuse
   [`create_parameter_ensemble`](../src/pyurbanair/config/hydra_helpers.py#L90)
   (+ `parameter_time_series/` for transient inflow, to teach transient BCs).
3. Run the ensemble **on disk** over the full `simulation_time` at the target
   `output_frequency`.
4. Per trajectory store: `(u,v,w[,pres])` over `time`, the parameter vector **resampled
   to one value per output frame** (the solver already interpolated the sparse input, so
   record per-frame *effective* params so tensors align with §1.5's dense convention),
   grid metadata, and the source solver's `param_schema` / `state_var_names`. Convert
   staggered uDALES via `interpolate_grid` (D3). Geometry mask/SDF is **identical across
   trajectories** (fixed geometry, D5) — voxelize once, store at the **corpus root**, not
   per-trajectory.
5. Persist to a **corpus directory** as **Zarr** (chunked along `time`/sample). **Store
   each trajectory whole — never pre-windowed** (overlapping windows duplicate frames
   ~`K×` and the rollout horizon `H` grows during training, so a materialized window
   corpus is both a storage blow-up and stale the moment the curriculum advances).
   Windowing is a lazy index map (§6.1.1). Keep a JSON manifest: solver, grid, bounds,
   param ranges, counts, git SHA.

Sampling guidance (unchanged): cover the inflow prior generously (stratify
`inflow_angle × velocity_magnitude`); include the **spin-up transient**; split **by
trajectory** into train/val/test (never leak frames across splits); run under the
ensemble scaling guidance (`num_parallel_processes ≈ 4` on the single box; scale out on
a cluster).

**Sizing (do this back-of-envelope *before* P1 — it decides feasibility).** Dominant cost
(§14); do not enter P1 without numbers:
- **Storage** = `N_traj · N_frames · nx·ny·nz · n_vars · bytes`. At `64³`, `n_vars=3`,
  `fp16`, `N_frames=100`: `≈1.5 MB/frame → ~150 MB/traj → ~150 GB / 1000 traj` (fp32
  doubles; `+pres` is +33%; `128³` is ×8). Pick dtype/grid deliberately.
- **Compute** = `N_traj · wall_time_per_traj / effective_workers`. On the 3950X box
  (`workers≈4`), minutes-per-trajectory transient runs at 1000 trajectories push to
  **days–weeks** on one box → justifies cluster scale-out.
- Start the GATE on a **small** corpus (§13); commit to the full corpus only after the
  GATE clears. Record `N_traj`, grid, dtype, GB, GPU-hours in the §12 decision.

**Anti-inverse-crime guard (enforce structurally).** When evaluating ESMDA with the
surrogate as the *assim* model, draw the *truth* from a Fortran solver (config mounts
`model/` twice, `docs/codebase_guide.md` §5). The P5 harness must **assert** truth and
assim are not the same backend/corpus and refuse to report posterior accuracy when they
coincide — a check, not a comment.

## 6. Training (architecture-agnostic)

The training loop is written **once** and trains any `SurrogateArchitecture`. It only
ever calls `arch.init_carry` / `arch.step` (via `rollout`) and the field-space loss — it
never knows whether the carry holds fields or latents.

### 6.1 Task formulation
- **Field-space pushforward learner.** From a window's K-frame history, `init_carry`
  builds the carry; `rollout` unrolls `H` steps feeding the model its own output;
  the loss is normalized per-variable MSE of decoded predictions vs the next `H` true
  frames. With `K=1, H=1` this is the plain one-step `(field_t, params) → field_{t+1}`.
- **Latents are recomputed every step — never cached.** For UPT, `init_carry`
  re-encodes raw frames inside the train step (encoder weights move every step). The
  dataset stores **raw fields** for both architectures; `data/dataset.py` never
  materializes latents.
- **Rollout/pushforward loss is mandatory** for ESMDA usefulness: curriculum grows the
  horizon `H` over training (stop-gradient on the warm-up portion) so error doesn't
  blow up over a window. This is the main risk for a surrogate driven repeatedly by
  ESMDA, and it applies to the UNet just as much as to UPT.
- **Off-manifold IC injection at step time** (§14): `training/loop.py` optionally
  perturbs `hist_fields` (ensemble-mix / noise / synthetic EnKF increments) before
  `init_carry`, feeding the §13 GATE B2 (analysis-OOD) bar — architecture-agnostic.
- Loss: normalized per-variable MSE; optional spectral/gradient term to preserve wake
  sharpness; **mask out solid cells** (D5).

#### 6.1.1 Windowing the trajectory corpus (`data/dataset.py`)
Window lazily via an index map — do not materialize windows. `dataset.py` stores
trajectories whole and builds a flat index `sample_id → (trajectory_id, t_start,
history_len)`. `__getitem__` slices Zarr lazily, applies stored normalization (§6.3) on
load, and returns one **field-space** window record (architecture-independent):

```
hist_fields    [K, C, Z, Y, X]   # K history frames; left-padded if history_len < K
hist_params    [K, P]            # dense per-step params at those frames (§1.5)
hist_mask      [K]               # 1 = real frame, 0 = left-pad
future_params  [H, P]            # boundary conditions for the H rollout steps
target_fields  [H, C, Z, Y, X]   # next H frames (pushforward targets, true frames only)
```

Mechanics:
- **Sliding window with stride `s`** turns hundreds of trajectories into the
  tens-of-thousands of samples a deep model needs.
- **Variable-length start windows are explicit index entries**, not just masking: emit
  windows anchored at `t_start=0` with `history_len = 1, 2, …, K−1` (left-pad +
  `hist_mask`) so the cold-start / early-window regime is actually trained.
- **`H` is curriculum-controlled** (just lengthens `target_fields`/`future_params`,
  shrinks valid `t_start` range — index recomputed, nothing re-materialized). Targets
  are always *true* frames; predicted intermediates are produced at step time by
  `rollout`.
- **Split by trajectory *before* building the index** (§5).
- **Batching**: stack `B` records; `hist_mask` lets mixed `1…K` windows batch together.

### 6.2 Conditioning
The **dense, interpolated** per-step params (§1.5, framework) reach training tensors as
`hist_params` / `future_params`. The *embedding* is shared (`training/conditioning.py`:
`inflow_angle` sin/cos already applied → MLP → FiLM/added tokens) but **invoked inside
the architecture** so each model fuses it natively (FiLM levels in the UNet, tokens in
UPT). uDALES adds `pressure_gradient_magnitude`; non-uDALES drops it.

The param list is not inferred from `model.name`; it comes from the corpus/checkpoint
`param_schema`. For a uDALES-trained neural surrogate, `pressure_gradient_magnitude`
must be present even though `model.name == "neural_surrogate"`.

### 6.3 Normalization
Fit per-variable mean/std (mask-aware) over the **training split only**; store in the
checkpoint manifest; apply at inference. Never recompute on assimilation data.

### 6.4 Single-GPU and multi-GPU
- **Single GPU**: default. `jit`-compiled step; gradient accumulation if memory-bound.
- **Multi-GPU (data parallel)**: JAX `jax.sharding` 1-D `Mesh`; shard batch axis,
  replicate params; `pmean` grads. Encapsulate in `training/sharding.py` so the step fn
  is mesh-agnostic — works for any architecture. Right default; parallelize over data,
  not model.
- **Model/sharded parallel**: only if one grid won't fit; a hook, don't build it.
- Launchers: pixi tasks `train-surrogate` (single) / `train-surrogate-multi`; SLURM on
  DelftBlue mirrors the `delftblue` env.
- Throughput hygiene: bf16 compute / fp32 master, Zarr prefetch, log step time + tokens/s.

### 6.5 Observability
At minimum log train/val loss and **rollout error vs horizon** (the metric that
matters); periodic field snapshots via `pyurbanair.utils.animation_utils`.

## 7. Storing models and configs

A trained model is **weights + everything needed to reproduce inference**, including the
**architecture identity**:

```
models/neural_surrogates/<run_id>/
  weights/                 # Orbax checkpoint (params + optimizer state for resume)
  config.yaml              # full resolved Hydra training config snapshot
  architecture.json        # {"name": "unet3d", "config": {...}}  + K (history)  <-- NEW
  normalization.json       # per-variable mean/std, mask stats
  grid.json                # source_solver_name, nx/ny/nz, bounds, coord arrays, mask ref
  geometry.npy             # baked static occupancy mask (+ SDF); STL path/SHA in manifest (D5)
  schema.json              # param_schema, state_var_names, output dim mapping, dtype policy
  manifest.json            # run_id, git SHA, data corpus id, metrics, created_at
  metrics.json             # final/best val + rollout-horizon errors
```

- `architecture.json` lets the forward model reconstruct the exact network; the loader
  resolves the name through `architectures/registry.py` and asserts the requested
  architecture matches the checkpoint.
- `run_id` = timestamp + short git SHA; `latest` symlink per (solver, geometry,
  architecture).
- `models/` is **git-ignored**; document an external sync target (object store / shared
  scratch).
- Inference resolves a checkpoint by **explicit path** or `run_id` via `utils/registry.py`.
  The forward-model Hydra config (§8) takes a `checkpoint_path`.
- `schema.json` is load-bearing: `ForwardModel` and `params_io` use it to validate and
  order params, so a uDALES-trained checkpoint still receives
  `pressure_gradient_magnitude` while a pylbm-trained checkpoint does not.

## 8. Configuration & Hydra wiring

### 8.1 `conf/model/neural_surrogate.yaml`
Mirror [`conf/model/pylbm.yaml`](../conf/model/pylbm.yaml):
```yaml
name: neural_surrogate
solver_name: neural_surrogate   # new entry in the obs-operator dim_mapping (§8.4)
checkpoint_path: ???            # required: models/neural_surrogates/<run_id> or a path
forward_model:
  _target_: neural_surrogates.forward_model.ForwardModel
  _convert_: all
  checkpoint_path: ${..checkpoint_path}
  nx: ${domain.nx}              # validated against checkpoint, NOT used to size the model
  ny: ${domain.ny}
  nz: ${domain.nz}
  bounds: ${domain.bounds}
  simulation_time: ${time.simulation_time}
  output_frequency: ${time.output_frequency}
  device: cuda                  # or cpu
prepare:
  _target_: pyurbanair.config.hydra_helpers.prepare_neural_surrogate
ensemble_model:
  _target_: neural_surrogates.ensemble_forward_model.EnsembleForwardModel
  _convert_: all
  ensemble_size: ${ensemble.ensemble_size}
  num_parallel_processes: 1     # default in-process batching (D2)
```

> **Grid AND architecture come from the checkpoint, not from config.** `nx/ny/nz/bounds`
> are passed only so `prepare_neural_surrogate` can **validate** the composed `domain`
> against the checkpoint and **raise on mismatch** — never to size the network. The
> architecture name/config likewise come from `architecture.json` (§7); the inference
> `model/` config does not pick the architecture (the *training* config did, §8.3).

### 8.2 `hydra_helpers` additions
- Add `prepare_neural_surrogate(forward_model, ...)` (replaces `compile`): load/validate
  the checkpoint against `cfg.domain` (grid) and `cfg.time` (output count/spacing),
  raise on mismatch, warm the JIT. **Keep heavy NN imports function-local** — module-top
  imports here would defeat the lazy-import invariant (note
  [hydra_helpers.py:16-17](../src/pyurbanair/config/hydra_helpers.py#L16) already imports
  `pylbm`/`pyudales` at module top and every script imports this module). Mirror the
  function-local pattern `pypalm` uses at
  [L60](../src/pyurbanair/config/hydra_helpers.py#L60).
- Extend `clean_outputs` with an `elif model_name == "neural_surrogate"` no-op branch
  **and turn the fall-through `else` into a raise**
  ([currently L55-64](../src/pyurbanair/config/hydra_helpers.py#L55-L64)) so unknown
  models can't silently get uDALES cleanup (per `docs/codebase_guide.md` §8).
- `create_true_params` / `create_parameter_ensemble` already produce the param Datasets
  the surrogate consumes for native Fortran backends, but the surrogate needs a schema
  bridge: when `model_name == "neural_surrogate"`, helper code or the calling script must
  use the checkpoint's `param_schema` / `source_solver_name` so uDALES-trained
  checkpoints include `pressure_gradient_magnitude`. Prefer adding a small helper such
  as `resolve_parameter_schema(model_or_cfg)` and extending the param factory call sites
  to pass an explicit schema, rather than making `create_parameter_ensemble` open
  checkpoints by itself.

### 8.3 Training config group
New `conf/neural_surrogate/` group, consumed only by
`scripts/generate_neural_surrogate_data.py` and `scripts/train_surrogate.py`:
```
conf/neural_surrogate/
  train.yaml                # optimizer, batch, rollout-horizon schedule, history K, corpus path, sharding
  arch/
    unet3d.yaml             # base_channels, channel_multipliers, num_levels, norm, activation, residual, ...
    upt.yaml                # (later) latent dim, depth, heads, supernode count, ...
  data.yaml                 # corpus generation knobs (sampling, dtype, grid)
```
The selected `arch/*.yaml` block is **snapshotted into the checkpoint manifest** (§7);
`K` lives in `train.yaml` and is baked per checkpoint. Keep training config **out of**
the inference `model/` group.

### 8.4 Observation operator
Register `neural_surrogate` in
[`ObservationOperator.__init__`](../libs/data-assimilation/src/data_assimilation/observation_operator.py#L65)
with the collocated mapping (same as pylbm):
```python
elif solver_name == "neural_surrogate":
    self.dim_mapping = {v: {"z": "z", "y": "y", "x": "x"} for v in ("u", "v", "w")}
```
**Cross-grid caveat for joint state estimation** (unchanged): a `neural_surrogate` assim
model emits collocated `x,y,z` even when truth is staggered uDALES. Irrelevant for
`ParameterESMDA`; self-consistent for `StateAndParameterESMDA` (augmented vector is the
assim state); but any direct truth-vs-assim **state** comparison or overlaid plotting is
cross-grid and must interpolate first. Flag in §11 tooling, don't silently difference.

## 9. Packaging, environment, dependencies

- `libs/neural_surrogates/pyproject.toml` mirrors `libs/pylbm/pyproject.toml`: editable,
  `packages=["src/neural_surrogates"]`. Deps: `jax`, `equinox`/`flax`, `optax`,
  `orbax-checkpoint`, `zarr`, `numpy`, `xarray`. No Fortran/MPI.
- Top-level `pyproject.toml`: new pixi feature
  `[tool.pixi.feature.neural_surrogates.*]` (NN libs + `neural_surrogates =
  {path="libs/neural_surrogates", editable=true}`); add to `cuda`, `delftblue`, `dev`
  — **not** `default`. GPU JAX via the existing `cuda` feature; CPU envs keep
  `JAX_PLATFORMS=cpu`.
- **Lazy import**: keep all `neural_surrogates.*` `_target_` blocks confined to
  `conf/model/neural_surrogate.yaml` so non-surrogate compositions never import the NN
  stack — same invariant pypalm relies on, asserted by a regression test (§11).

## 10. Inference / use in the existing pipeline

The common ESMDA/rollout math does not change, but a few helpers/scripts need surrogate
schema awareness (param factories, anti-inverse-crime checks, and possibly
truth/assim comparison tooling). The user-facing invocation stays config-driven:
```bash
# single forward sim with the surrogate
python scripts/run_forward_model.py model=neural_surrogate \
  model.checkpoint_path=models/neural_surrogates/latest

# parameter ESMDA, Fortran truth vs surrogate assim (anti-inverse-crime)
python scripts/run_parameter_esmda.py \
  model@truth_model=pylbm model@assim_model=neural_surrogate \
  assim_model.checkpoint_path=models/neural_surrogates/latest
```
The architecture (UNet or UPT) is whatever the checkpoint was trained with; the command
line is identical. A surrogate ESMDA window costs a batched GPU forward pass instead of N
Fortran simulations.

**Uncertainty caveat (unchanged, applies to *any* deterministic architecture):** ensemble
spread comes from the parameter/IC ensembles, not model stochasticity, so the surrogate
under-disperses. Treat calibrated spread as a **P5 pass/fail bar** (rank histograms /
spread-skill, §13); add model-error inflation / an ensemble of surrogates / MC-dropout
as needed.

## 11. Testing & CI

- For setup, integration wiring, and smoke-test runs that compose the real pyurbanair
  config, use the `size=small` overlay by default. This is the intended modest
  end-to-end setup for neural-surrogate plumbing; reserve `size=tiny` / synthetic data
  for narrow unit tests and the full/default sizes for post-GATE scaling.
- `libs/neural_surrogates/tests/`: `state_io`/`params_io` round-trips; a **tiny
  architecture** training-step test (default UNet, 2 levels, few channels); a
  rollout-shape test; and the **CPU smoke-training stage (§11.1)**.
- **`rollout`/interface conformance test**: a tiny dummy `SurrogateArchitecture` (e.g.
  identity-ish carry) exercises `init_carry`/`step`/`rollout` shapes and the K=1 fast
  path independently of any real network — guards the contract.
- **Architecture-specific tests** live beside each architecture. The UNet adds a
  grid-divisibility / pad-crop test. **UPT (later)** adds the **encoder/decoder + K=1
  parity test** against `ml-jku/UPT` (faithful-port gate, §1.6 / old plan §11),
  skippable without PyTorch; it guards the *port*, not the surrogate's accuracy.
- `tests/` (top-level): a
  `compose_test_cfg(["model=neural_surrogate", "size=small", ...])` end-to-end test
  against a **train-on-the-fly tiny checkpoint** in `tmp_path` (no committed binary, no
  GPU); and a regression test that composing a non-surrogate model does **not** import
  `neural_surrogates` (mirror `test_palm_target_does_not_import_for_non_palm_composition`).

### 11.1 CPU smoke-training stage (plumbing, not accuracy)
A **CPU-only** stage proving the training code *runs end-to-end* on a **tiny synthetic
corpus** in `tmp_path` (no solver, no GPU): random trajectories → window (§6.1.1) → tiny
UNet → a few optimizer steps. **Parametrized over shape**, since hard-coded/mismatched
dimensions are the likeliest break:
- **Trajectory length `T ∈ {2,3,7,20}`** — exercises the windowing index map, including
  `T` too short for a full `K+H` window (only short/truncated windows, never an
  out-of-range slice) and `T` large enough for many interior windows.
- **Grid shape `(Z,Y,X)`** — a tiny cube, an anisotropic, and a non-cubic case; vary
  the UNet `num_levels` so the divisibility/pad path is hit.
- **History `K ∈ {1,3}` and horizon `H ∈ {1,3}`** — `K=1` Markov fast path, `K=3`
  channel-stack + `hist_mask` (incl. one `history_len < K` window), `H>1` pushforward.
- **Channels `C ∈ {3,4}`** — with/without `pres`.

Assert **plumbing invariants only** (shape + finiteness): window record shapes match
§6.1.1 and the index count; forward pass is `target_fields`-shaped, loss finite;
gradients finite and the param PyTree **changes** after the steps; the `H>1` unroll feeds
output back without shape drift. Keep one combination wired to save+reload a throwaway
checkpoint so the registry/manifest round-trip is smoke-tested. This is the always-on
guard so the §13 GATE is never the *first* thing to exercise the training loop.

Separately, any smoke test that instantiates real solver/surrogate Hydra configs should
compose with `size=small`, e.g.
`compose_test_cfg(["model=neural_surrogate", "size=small", ...])`, so setup exercises
the same modest domain/ensemble/time settings used by quick end-to-end runs.

## 12. Decisions to confirm before coding

1. **Framework (D1)**: JAX+Equinox (recommended; repo-consistent; JAX *is* the D2
   batching path; UNet trivial, UPT is the port cost) vs PyTorch. *Biggest, least
   reversible.* Differentiability is **not** a tiebreaker (ESMDA is derivative-free).
2. **Ensemble execution (D2)**: confirm in-process on-device batching as default; honor
   all three save modes (§4).
3. **Cold-start IC (§4)**: (a) require IC, (b) canned IC bank (recommended default), (c)
   params→IC head (its own phase if needed). *Blocks ESMDA window 0.*
4. **History length `K` (D4)**: framework hyperparameter, baked per checkpoint. UNet
   default `K=1` (Markov) for the first cut; raise later by ablation. UPT targets 3–10.
5. **Geometry representation (D5)**: SDF (+ mask) recommended; baked offline, never fed
   STL-wise.
6. **First architecture config (§1.2)**: confirm UNet hyperparameters
   (`num_levels`/channels/norm/residual) for the first corpus.
7. **Source solver for the first corpus**: pylbm on the Xie–Castro geometry already in
   `examples/lbm/` (collocated, CUDA, fastest) — recommended.
8. **Corpus storage & budget (§5)**: `N_traj`, grid, dtype, GB, GPU-hours, checkpoint
   sync target. Record before generating.
9. **Schema bridge for surrogate params**: confirm the concrete `schema.json` shape and
   helper API (`resolve_parameter_schema` or equivalent) before P3, so uDALES-trained
   checkpoints do not silently drop `pressure_gradient_magnitude`.

## 13. Phased roadmap

Dependency order: **P0 → P1 → P2 → GATE → (P3 ∥ P4) → P5**. The architecture abstraction
means **UPT slots in after the framework is proven** (a later, isolated P-add), without
re-touching P0/P1/P3.

- **P0 — scaffolding**: `libs/neural_surrogates` skeleton, pyproject + pixi feature,
  lazy-import regression test, `state_io`/`params_io` + round-trip tests, the
  `SurrogateArchitecture` interface + `rollout` + conformance test. No real model yet.
  Any repo-integrated config smoke uses `size=small`.
- **P1 — data**: `scripts/generate_neural_surrogate_data.py`, Zarr corpus + manifest,
  normalization fit, a small pylbm corpus end-to-end. **Do the §5 sizing first.**
- **P2 — UNet + single-GPU training**: implement `architectures/unet3d.py` (config-driven,
  §1.2) and the architecture-agnostic `training/loop.py` (one-step + **pushforward** loss
  with off-manifold IC injection) + `train_surrogate.py` + checkpoint/manifest +
  rollout-error eval. **Stand up the §11.1 CPU smoke stage as soon as the loop is wired**,
  before real data or the GATE. Real-config smoke runs use `size=small`.
- **GATE (after P2) — go/no-go before scaling**, on a *small* pylbm corpus. Three bars,
  all pre-stated, over a full assimilation window's worth of steps:
  1. **Clean rollout-error-vs-horizon** on held-out trajectories — below bar B1.
  2. **Analysis-OOD robustness (load-bearing)**: roll from perturbed / ensemble-mixed /
     noised ICs and ≥1 synthetic EnKF analysis increment (§14) — below bar B2. Cheap on
     the small corpus; the only bar that de-risks P4/P5.
  3. **Cold-start sanity**: roll from the §4 IC strategy, confirm no divergence.

  Commit cluster time to P4 only if **all three** clear. The gate is intentionally
  architecture-agnostic — the same `eval_surrogate_gate.py` re-runs for UPT later.
- **P3 — forward model**: `ForwardModel` + `EnsembleForwardModel` (D2),
  `conf/model/neural_surrogate.yaml`, `prepare_neural_surrogate`, `clean_outputs` branch,
  obs-operator registration; `run_forward_model.py model=neural_surrogate` works.
- **P4 — multi-GPU**: `training/sharding.py` data-parallel mesh, pixi tasks,
  DelftBlue/SLURM; scale the corpus and model.
- **P5 — ESMDA validation**: surrogate-as-assim vs Fortran-truth (enforced
  anti-inverse-crime guard, §5); posterior accuracy, multi-window rollout stability, and
  **ensemble dispersion as a pass/fail bar** (rank histograms / spread-skill, §10);
  add inflation if spread collapses. Flag cross-grid truth-vs-assim comparisons (§8.4).
- **P-UPT (later) — second architecture**: implement `architectures/upt/*` (faithful port,
  §1.6), pass the encoder/decoder + K=1 **parity test** (§11), re-run the CPU smoke stage
  and the **same GATE** on the same corpus, then compare against the UNet. No framework
  changes; only a new `arch/upt.yaml` and a retrain.

## 14. Risks / open issues

- **Rollout drift** over long windows — mitigated by pushforward training (§6.1); primary
  acceptance metric is rollout-error-vs-horizon, not one-step MSE. Applies to every
  architecture.
- **State out-of-distribution from the analysis step** (likely the *dominant* risk): the
  EnKF analysis produces states that are linear combinations of ensemble members and need
  not lie on the learned manifold. Feeding such an increment into an autoregressive
  surrogate is where rollout drift detonates. Mitigation: §6.1 pushforward training
  includes perturbed / off-manifold ICs (GATE B2). Architecture-independent.
- **Under-dispersion** in ESMDA (§10) — a deterministic surrogate under-disperses
  regardless of architecture; calibrated spread is a P5 bar.
- **Parameter distribution shift**: ESMDA can push params outside the training prior;
  cover margins in §5 sampling and monitor.
- **Data-generation cost** dominates effort (§5); plan cluster time. The corpus is shared
  across architectures, so this cost is paid once for both UNet and UPT.
- **Staggered-grid information loss** from interpolation (D3) — acceptable for a
  surrogate; quantify against held-out uDALES data if used. (The UNet *requires* the
  collocated grid.)
- **Geometry generalization (cannot transfer to an unseen building)**: a single-geometry
  surrogate (D5) is specialized to one occupancy field, fixed grid, geometry-specific
  normalization — a different geometry is OOD and unreliable. No cheap inference fix.
  Treat one checkpoint as valid for exactly the geometry it was trained on. (Architecture-
  independent: a UNet on a fixed grid is, if anything, *less* transferable than UPT's
  point-based formulation.)
- **Architecture-interface leakage**: the risk specific to this design is the
  field-space/opaque-carry interface (§1.1) proving too narrow for UPT's latent stepping.
  Mitigation: the interface was deliberately specified against *both* the UNet and the
  UPT K-frame design up front; the `rollout` conformance test (§11) and the P-UPT phase
  re-validate it before any framework code is committed to UPT's shape.

## 15. Detailed implementation plan (file-by-file)

Decision defaults (from §12, recommended): **JAX + Equinox (D1)**, **in-process `vmap`
(D2)**, **canned-IC-bank default (§4)**, **SDF + mask geometry (D5)**, **pylbm /
Xie–Castro first corpus (§12.7)**, **3D conv UNet first architecture (§1.2)**. New files
in backticks; existing files linked. Order: **P0 → P1 → P2 → GATE → (P3 ∥ P4) → P5 →
P-UPT**.

### P0 — Scaffolding (no real model, no GPU)
**Exit:** `libs/neural_surrogates` installs, lazy-import invariant holds, `state_io`/
`params_io` round-trip, the architecture interface + `rollout` + conformance test pass.
Use `size=small` for any smoke test that composes the real pyurbanair Hydra config.

| File | Responsibility |
|---|---|
| `libs/neural_surrogates/pyproject.toml` | Mirror [libs/pylbm/pyproject.toml](../libs/pylbm/pyproject.toml): `packages=["src/neural_surrogates"]`; deps `jax, equinox, optax, orbax-checkpoint, zarr, numpy, xarray`. No Fortran/MPI. |
| `libs/neural_surrogates/src/neural_surrogates/__init__.py` | Version only. **No top-level NN imports** (lazy invariant). |
| `.../architectures/base.py` | `SurrogateArchitecture` interface (`init_carry`, `step`) + `Carry` typing (§1.1). |
| `.../architectures/registry.py` | `resolve_architecture(name, config) -> SurrogateArchitecture`; `{"unet3d": ...}` (UPT added later). |
| `.../rollout.py` | `rollout(arch, carry, future_params, static, n_steps)` via `jax.lax.scan`; pushforward stop-gradient hook (§1.1/§6.1). |
| `.../utils/state_io.py` | `state_to_tensor(ds, grid) -> [T,C,Z,Y,X]`; accept both time-indexed histories and time-less single-frame warm starts; `tensor_to_state(arr, grid, var_names, time_coords) -> Dataset`; `trim_to_window(ds, num_outputs)` (§4 time-axis contract, matching [pylbm forward_model.py:401-408](../libs/pylbm/src/pylbm/forward_model.py#L401-L408)); K-frame history extraction. |
| `.../utils/params_io.py` | `params_to_conditioning(params, schema, num_steps, output_frequency, spinup_outputs) -> [T,P]`: sparse→dense **linear** interp, `inflow_angle` sin/cos, spin-up plateau — matching [`write_uvel_time_file`](../libs/pylbm/src/pylbm/utils/params_utils.py#L119); include/drop `pressure_gradient_magnitude` according to the checkpoint `param_schema`, not `model.name`. |
| `.../data/grid.py` | `GridMeta`; `build_occupancy_mask(stl_path, grid)` reusing [`get_building_grid_indices`](../libs/pylbm/src/pylbm/stl_to_lbm.py#L120); `mask_to_sdf(mask)` (signed EDT). |
| `.../utils/registry.py` | `resolve_checkpoint(path_or_run_id) -> Path`; `load_manifest(dir)`. |
| `libs/neural_surrogates/tests/test_io_roundtrip.py` | state↔tensor and params→conditioning round-trips; time-less warm-start frames normalize to one-frame tensors; schema-driven params include uDALES pressure gradient only when requested; trimmed `time` equals pylbm's on a tiny config. |
| `libs/neural_surrogates/tests/test_interface.py` | Dummy architecture exercises `init_carry`/`step`/`rollout` shapes + K=1 fast path. |
| [pyproject.toml](../pyproject.toml) | `[tool.pixi.feature.neural_surrogates.*]` (NN deps + editable lib); add to `cuda`, `delftblue`, `dev` — **not** `default`. |
| [tests/test_hydra_config.py](../tests/test_hydra_config.py) | `test_neural_surrogate_target_does_not_import_for_non_surrogate_composition` (mirror pypalm). |

### P1 — Data generation (small corpus)
**Exit:** small pylbm Xie–Castro corpus + fitted normalization, readable by `dataset.py`.
**Do the §5 sizing table first**; record in `conf/neural_surrogate/data.yaml`.

| File | Responsibility |
|---|---|
| `scripts/generate_neural_surrogate_data.py` | `def run(cfg)` + `@hydra.main`. Build solver+ensemble like [run_ensemble_forward_model.py](../scripts/run_ensemble_forward_model.py); sample params via [`create_parameter_ensemble`](../src/pyurbanair/config/hydra_helpers.py#L90) (+ `parameter_time_series/`); run **on-disk** over full `simulation_time`+spin-up; record per-frame **effective** params and the source solver schema (§5/§1.5). |
| `.../data/generate.py` | `write_trajectory(...)` to Zarr (whole trajectories, chunked along `time`/sample, §5); voxelize geometry **once** to corpus root (`geometry.npy`); corpus manifest JSON. Staggered uDALES via `interpolate_grid` (D3). |
| `.../data/normalization.py` | `fit_normalization(corpus, split="train")` (mask-aware per-var mean/std → `normalization.json`); `apply`/`invert`. Train split only (§6.3). |

### P2 — UNet, single-GPU training, CPU smoke stage, GATE
**Exit:** config-driven 3D UNet, architecture-agnostic pushforward training, **§11.1 CPU
smoke stage green**, **all three §13 GATE bars cleared**.

| File | Responsibility |
|---|---|
| `.../architectures/unet3d.py` | 3D conv UNet implementing `SurrogateArchitecture` (§1.2): config-driven depth/channels/norm/activation/residual; K-frame channel-stacking (K=1 fast path); FiLM conditioning; grid divisibility pad/crop. |
| `.../training/conditioning.py` | Shared param-embedding MLP (sin/cos already applied) → FiLM/tokens (§6.2). |
| `.../data/dataset.py` | **Lazy index-map windowing** (§6.1.1): `sample_id → (trajectory_id, t_start, history_len)`; `__getitem__` slices Zarr, normalizes, returns the field-space window record; explicit short-history start entries; split-by-trajectory before indexing. |
| `.../training/loop.py` | `train_step` (jit); one-step + **pushforward** loss (curriculum on `H`, stop-gradient warm-up) via `rollout`; **inject perturbed/off-manifold ICs** at step time (§6.1/§14, feeds GATE B2); mask solid cells; rollout-error-vs-horizon eval. **Architecture-agnostic** (only calls the interface). |
| `.../training/train.py` | `def run(cfg)` + `@hydra.main`; Optax, bf16 compute / fp32 master, Zarr prefetch. |
| `.../training/checkpoint.py` | Orbax save + the §7 artifact set incl. **`architecture.json`**, **K**, and **`schema.json`** (`source_solver_name`, `param_schema`, `state_var_names`). |
| `conf/neural_surrogate/train.yaml`, `arch/unet3d.yaml`, `data.yaml` | Training group (§8.3); consumed only by the two training scripts. `arch/unet3d.yaml` snapshotted into the checkpoint. |
| `libs/neural_surrogates/tests/test_smoke_training.py` | **§11.1 CPU smoke stage** — parametrized over `(T, grid, K, H, C)`; plumbing invariants only. Stand up as soon as the loop is wired. |
| `scripts/eval_surrogate_gate.py` | Compute + check the **three §13 GATE bars** on the small corpus. Hard go/no-go. Architecture-agnostic (re-runs for UPT). |

### P3 — Forward model + Hydra wiring (inference)
**Exit:** `python scripts/run_forward_model.py model=neural_surrogate
model.checkpoint_path=…` runs and honors the `BaseForwardModel` contract.

| File | Responsibility |
|---|---|
| `.../forward_model.py` | `ForwardModel(BaseForwardModel)`. `__init__`: load checkpoint via `registry`, resolve architecture from `architecture.json`, store schema/grid/normalization/mask, `num_outputs`, **lazy-init on first `run_single`**. `run_single` per §4 (conditioning seq using `param_schema` → resolve IC via canned bank, raise on `None` stopgap → normalize time-less or time-indexed state history → `init_carry` → `rollout` → decode/denorm/reapply mask → trimmed Dataset). Implement concrete `save_results(...)` delegating to `_save_results`; `_clean_output`=no-op. |
| `.../ensemble_forward_model.py` | `EnsembleForwardModel(BaseEnsembleForwardModel)`. `_create_new_forward_model` shares weights, clones result dirs. **Override `run_ensemble`** honoring all 3 save modes + `state` as `Path` (via `get_member_state`) + `rollout_step` increment ([base L482-522](../src/pyurbanair/base_ensemble_forward_model.py#L482-L522)); stream chunk-load members, `vmap` each `vmap_chunk_size` sub-batch, then concat/write outputs (D2). |
| `conf/model/neural_surrogate.yaml` | Per §8.1: `checkpoint_path: ???`, `prepare._target_: …prepare_neural_surrogate`, `num_parallel_processes: 1`; nx/ny/nz/bounds for **validation only**. |
| [src/pyurbanair/config/hydra_helpers.py](../src/pyurbanair/config/hydra_helpers.py) | Add `prepare_neural_surrogate` (**function-local** NN imports, §8.2) validating grid vs `cfg.domain` + output count vs `cfg.time`, then JIT warm-up. Add `elif model_name=="neural_surrogate"` no-op to `clean_outputs` **and turn the `else` fall-through into a raise** ([currently L63-64](../src/pyurbanair/config/hydra_helpers.py#L63-L64)). Add/route schema-aware param factory behavior (`resolve_parameter_schema(model_or_cfg)` or equivalent) so `neural_surrogate` can request the checkpoint's `param_schema` (notably uDALES pressure gradient) without the factory guessing from `model.name`. |
| [observation_operator.py](../libs/data-assimilation/src/data_assimilation/observation_operator.py#L71) | Add `elif solver_name=="neural_surrogate"` collocated mapping (§8.4). |
| `tests/test_neural_surrogate_forward.py` | `compose_test_cfg(["model=neural_surrogate", "size=small", …])` end-to-end against a **train-on-the-fly tiny checkpoint** in `tmp_path` (no committed binary, no GPU; §11). |

### P4 — Multi-GPU + scale corpus
| File | Responsibility |
|---|---|
| `.../training/sharding.py` | 1-D device `Mesh`; shard batch / replicate params; `pmean` grads; mesh-agnostic step fn (§6.4). |
| [pyproject.toml](../pyproject.toml) | pixi tasks `train-surrogate` / `train-surrogate-multi`; DelftBlue SLURM launcher. |
| (corpus) | Scale §5 generation to the full `N_traj`; retrain. |

### P5 — ESMDA validation
- Run [run_parameter_esmda.py](../scripts/run_parameter_esmda.py) /
  [run_state_and_parameter_esmda.py](../scripts/run_state_and_parameter_esmda.py) with
  `model@truth_model=pylbm model@assim_model=neural_surrogate`.
- **Enforced anti-inverse-crime guard** (assert truth ≠ assim backend/corpus, §5).
- Metrics: posterior accuracy, multi-window rollout stability, **ensemble dispersion as a
  pass/fail bar** (rank histograms / spread-skill, §10); inflation if spread collapses;
  flag cross-grid comparisons (§8.4).

### P-UPT (later) — second architecture
- `.../architectures/upt/{upt,encoder,propagator,decoder,layers}.py` — **faithful port**
  of `ml-jku/UPT` encoder/decoder + the K-frame propagator (§1.6, old plan §1.0/D4-ii),
  implementing the same `SurrogateArchitecture` interface.
- `libs/neural_surrogates/tests/test_upt_parity.py` — encoder/decoder + K=1 reference
  parity (§11), skippable without PyTorch; degrades to structural review if no comparable
  transient reference config.
- `conf/neural_surrogate/arch/upt.yaml` — UPT dims; register `"upt"` in
  `architectures/registry.py`.
- Re-run the **CPU smoke stage** and the **same `eval_surrogate_gate.py` GATE** on the
  existing corpus; compare against the UNet. **No framework changes.**

### Resolve before P2 coding
1. **First UNet hyperparameters** (§1.2/§12.6) for the small corpus.
2. **Canned-IC-bank format** (§4) the GATE cold-start bar and P3 `run_single` consume.

### Resolve before P3 coding
1. **Checkpoint schema bridge** (§7/§8.2): exact `schema.json` fields and helper/call-site
   API for creating true/prior params from a `neural_surrogate` checkpoint's
   `source_solver_name` / `param_schema`.

### Resolve before P-UPT coding
1. **§1.0 prerequisite** — does released `ml-jku/UPT` ship a comparable transient config
   (ideally weights) to parity-test against? If not, the parity test degrades to
   structural review + shape/gradient tests (§11).
