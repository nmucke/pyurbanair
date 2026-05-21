# Plan: UPT neural-surrogate forward model (`libs/pyupt`)

Status: **proposal / design doc**. Implementation not started.

This document plans a neural-network surrogate forward model based on
**Universal Physics Transformers** (UPT, [arXiv:2402.12365](https://arxiv.org/abs/2402.12365),
Alkin et al., NeurIPS 2024) and packages it as a fourth backend, `pyupt`,
that drops into the existing `BaseForwardModel` / `BaseEnsembleForwardModel`
machinery alongside `pylbm`, `pyudales`, `pypalm`.

> **Reference implementation (authoritative): [`github.com/ml-jku/UPT`](https://github.com/ml-jku/UPT)** (official PyTorch).
> The **encoder and decoder** under `src/pyupt/model/` are a **faithful, parity-tested
> JAX port of this repo**. The **temporal propagator is a new design** (the K-frame
> history attention, D4-ii); at `K = 1` it ports the reference approximator and is
> parity-testable in that mode, but the K-frame extension is validated by *behavior*
> (rollout / analysis-OOD), not by numeric identity. See
> [§1.0 Implementation fidelity](#10--implementation-fidelity-faithful-port-of-encoderdecoder-new-design-for-the-propagator).

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

These decisions shape everything below. The framework and ensemble-execution
choices are flagged again in §12 as the decisions to confirm before coding.

### 1.0 — Implementation fidelity: faithful port of encoder/decoder, new design for the propagator
Fidelity discipline applies **where we genuinely reuse upstream design** — the
**encoder** and **decoder**. There it is worth reproducing
[`github.com/ml-jku/UPT`](https://github.com/ml-jku/UPT) module-for-module and
op-for-op rather than re-deriving from the paper, because the port crosses
frameworks (PyTorch → JAX, D1) and small unintended divergences (default
initializations, attention scaling, LayerNorm epsilon/placement, residual order,
positional-encoding conventions, pooling/supernode construction) silently change
behavior and are hard to debug later.

The **temporal propagator is explicitly a new design**, not a faithful port: the
K-frame history-attention stepper (D4-ii) replaces the reference's Markov
approximator with a temporal transformer. At `K = 1` it should reduce to the
reference approximator (special-case the K=1 path to bypass the temporal block —
see D4-ii — so the reduction is exact and parity-testable), but the K>1 design is
validated by *behavior* (rollout error, analysis-OOD robustness — §13 GATE), not
by numeric identity. **Do not hold the propagator to op-for-op parity.**

A second, easily-overlooked caveat sets the *ceiling* on what parity can buy:
**we never reuse upstream weights.** Our surrogate is trained from scratch on our
own geometry/grid/physics (geometry generalization is out of scope, §0), so the
parity test does not certify "we reproduce a validated model" — it certifies only
"the encoder/decoder port has no transcription bug." That is worth having, but it
is a code-correctness check, **not** the acceptance gate for the surrogate; the
acceptance gate is the behavioral GATE in §13.

Concretely, for the encoder/decoder:
- **Mirror the reference's module structure and names** so review against upstream
  is mechanical. UPT's stages map to our files (§2): reference **encoder**
  (supernode message-passing + perceiver pooling) → `encoder.py`; reference
  **decoder** (perceiver cross-attention to query positions) → `decoder.py`;
  reference **approximator** → `propagator.py` (the K=1 core; see above). Keep the
  upstream layer names in comments.
- **Match hyperparameters and numerics**: hidden dims, depth, head count, MLP
  ratio, norm type/eps, activation, init scheme, attention scaling, and weight-tying
  exactly as upstream unless a deviation below requires otherwise.
- **Verify with a numerical parity test** (§11), scoped to encoder, decoder, and
  the **K=1** propagator: load identical weights into the reference (PyTorch) and
  the JAX port on a tiny config and assert forward-pass outputs match to tight
  tolerance. This is the acceptance gate **for the port**, separate from — and
  subordinate to — the behavioral GATE (§13).

> **Verify before committing to the parity approach** that the released
> `ml-jku/UPT` actually ships the transient latent-stepping configuration (and,
> ideally, weights) we intend to compare against — not only the architecture for a
> different benchmark. If it does not, the parity test degrades to "structural
> port review" and the encoder/decoder are validated by shape/gradient tests
> instead. This is a P2 prerequisite, not an assumption.

**Sanctioned deviations from upstream** (everything in the encoder/decoder
otherwise tracks the reference):
1. **Framework**: JAX + Equinox/Flax instead of PyTorch (D1).
2. **Ensemble execution**: in-process `vmap` batching over the ensemble axis,
   no process forking (D2) — an *outer* concern, not a change to the model.
3. **Temporal propagator (new design, not a port)**: the `K`-frame
   history-attention stepper (D4-ii). `K = 1` is special-cased to recover the
   reference Markov approximator exactly; `K > 1` is a new module validated by
   behavior, not parity.
4. **Conditioning**: parameter embedding (`inflow_angle` sin/cos, etc.) injected
   into the propagator (§6.2) and the dense per-step param interpolation done in
   the forward model (D4-i) — both upstream of / around the core model.
5. **Geometry input**: the static occupancy/SDF channel (D5).
6. **I/O contract**: the `xarray`/grid glue (`utils/state_io`, `params_io`) and
   collocated-grid choice (D3) wrap the model; they do not alter it.

If any of these forces a change *inside* the ported encoder/decoder, isolate it
behind a flag and document the upstream line it diverges from, so those modules
stay diffable against `ml-jku/UPT`.

### D1 — Framework: JAX + Equinox/Flax (recommended)
The repo is already JAX-centric: `jax>=0.7` is a top-level dependency, ESMDA
is written in JAX, and the ensemble executor deliberately uses `forkserver`
*because* JAX starts threads at import
([`base_ensemble_forward_model.py:417`](../src/pyurbanair/base_ensemble_forward_model.py#L417)).
A JAX surrogate keeps a single accelerator stack.

The decisive technical reason is **D2**: the natural ensemble-parallelism for a
NN is `jax.vmap` over the `ensemble` axis, so JAX *is* the batched-inference
strategy — D1 and D2 reinforce each other. (Note: do **not** justify JAX by
"differentiating through the forward model inside ESMDA" — ESMDA is a
derivative-free ensemble Kalman method and gains nothing from autodiff. End-to-end
differentiability only pays off for a future gradient-based assimilator, e.g.
4D-Var; it is not a reason for *this* repo today.)
- Recommended: **Equinox** (lightweight, PyTree-native modules) + **Optax**
  (optimizers) + **Orbax** (checkpointing). Flax NNX is an acceptable alternative.
- Trade-off / alternative: the **official UPT reference code is PyTorch**
  ([`github.com/ml-jku/UPT`](https://github.com/ml-jku/UPT)). Choosing JAX means
  **porting** that repo across frameworks — the cost lands on faithful
  reproduction (§1.0), not on framework maturity. PyTorch (DDP/FSDP, mature
  transformer ecosystem, *runs the reference as-is*) avoids the port but means
  maintaining a second accelerator framework in the repo and bridging tensors↔JAX
  at the ESMDA boundary. Recommend JAX unless the cross-framework porting/parity
  effort is judged the dominant risk. **Decision to confirm — see §12.**

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
- Run the batched autoregressive rollout **in sub-batches sized to GPU memory**,
  not necessarily one pass for the whole ensemble. `ensemble_size × grid × K`
  latent history can exceed device memory for large ensembles/grids; expose a
  `vmap_chunk_size` (members per device pass) and loop over chunks so a large
  ensemble degrades to a few passes instead of an OOM. Default chunk = full
  ensemble; lower it when memory-bound.
- Re-split into a `concat`-along-`ensemble` `xarray.Dataset`, matching the
  return contract of the base class.
- Keep `num_parallel_processes` semantics only as an optional fallback (e.g.
  CPU-only smoke tests); default to in-process batching.

This is the single most important efficiency decision and the main place where
`pyupt` legitimately diverges from the other backends. A corollary: because D2
keeps everything in one process, `pyupt` inference essentially never forks, so
the forkserver/JAX-threads rationale that motivates the existing parallel path
is moot for the surrogate — the framework choice (D1) is more orthogonal to the
parallel executor than it first appears.

The override must, however, still honor the **full** `run_ensemble` contract,
not just the in-memory happy path (see §4 for the expanded spec): on-disk save
mode, `state` arrived as a `pathlib.Path` to per-member files, and the
`rollout_step` increment all flow through `run_ensemble` today. Only the
*execution mechanism* changes (process-fork → `vmap`); the save-mode and
warm-start semantics are preserved.

### D3 — Common grid / staggering
UPT is grid-agnostic but our training tensors are simplest on a single
collocated grid. Train and predict on the **collocated `x,y,z` grid**:
- pylbm / pypalm already collocated → use directly.
- pyudales staggered (`xt/xm`, …) → interpolate to a common grid with the
  existing `pyudales.utils.grid_utils.interpolate_grid` *before* building
  training tensors, and register `pyupt` in the observation operator with the
  collocated mapping (see §8.4).
A geometry/building **mask** (solid vs fluid cells) is a static input channel —
how that mask is produced from the `.stl` is **D5**.

### D4 — Temporal context: a short, variable-length history of states + params
Two related requirements drive this, and they are split across the **two layers**:

**(i) Network sees a short history; forward model interpolates sparse params.**
The Fortran solvers accept a *sparse* time-varying parameter series and
**interpolate between the values at runtime** — e.g. pylbm writes `(time,
velocity, direction)` rows to `uvel_time.dat` and `m_inflow.F90` interpolates
(see [`write_uvel_time_file`](../libs/pylbm/src/pylbm/utils/params_utils.py#L119)).
`pyupt` replicates this **in the forward model, not the network**:
`_apply_inflow_settings` / `utils/params_io` accept a params `Dataset` whose vars
carry a `time` dim of `N ≪ num_outputs` sparse values and **interpolate them onto
the dense per-step rollout grid**, producing a per-step conditioning sequence. The
UPT network itself only ever consumes **one** conditioning vector per step (or a
short history of them — see (ii)); it never sees the sparse series. Interpolation
details:
- **Linear** interpolation to match the Fortran behavior (`xarray.Dataset.interp`
  / `numpy.interp`).
- Interpolate `inflow_angle` via its **sin/cos components** (or unwrap) so the
  359°→1° wrap doesn't produce a spurious sweep, *then* embed.
- Replicate the **spin-up plateau** convention (the `spinup_time` prepend in
  `write_uvel_time_file`) if spin-up frames are emitted.
- Scalar params (no `time` dim) broadcast to every step, exactly as today.

**(ii) State/param history as model input (architecture extension).** Native UPT
is **Markovian**: it encodes one state and steps the latent forward
(`z_t → z_{t+1}`). The requirement is to condition each step on a short history
(target **K = 3–10** frames) of past states *and* their params, starting from a
single initial condition and growing as the rollout accumulates frames. This is a
genuine architecture change — recommended design:
- Maintain a **sliding window (ring buffer) of the last `K` encoded latent
  frames** `{z_{t-K+1}, …, z_t}`, each tagged with (a) its interpolated param
  embedding and (b) a relative temporal position encoding. The buffer holds
  *latents*, not fields, so it is cheap to carry.
- The propagator becomes a **temporal transformer that attends across these `K`
  latent frames** (plus the target step's param embedding, the boundary condition
  for `t→t+1`) to produce `z_{t+1}`. Spatial structure is already captured inside
  each frame's latent token set by the encoder, so the added attention is over the
  small temporal axis — keep it **factorized / temporal-only** (or use a few
  pooled summary tokens per frame) so cost stays ≈ linear in `K`.
- **Variable length via masking**: at step 0 only the IC exists (history length 1);
  pad to `K` and apply an attention mask so the *same* module handles `1…K`
  frames. **`K = 1` is special-cased to bypass the temporal-attention block** so it
  reduces *exactly* to the reference Markov approximator — without that special-case
  a length-1 temporal transformer still carries the added param cross-attention and
  is only *approximately* the reference, which would break the §11 K=1 parity test.
  With the bypass, the history-aware architecture is a strict superset of the
  baseline and can be ablated against it.
- Decode still happens per step; only the propagator input grows. Memory/compute
  grow modestly because `K` is small and the window stores latents.
- *Alternatives considered & rejected*: channel-stacking `K` states at the
  encoder input (fixed `K`, conflates time with channels, no clean
  variable-length handling); recurrent latent memory (more hidden state to
  manage, doesn't match the explicit short-window request).

### D5 — Geometry: STL voxelized to a static occupancy/SDF channel (baked per model)
Today every backend takes geometry as a `.stl` and voxelizes it onto its grid:
pylbm via [`get_building_grid_indices(stl, nx, ny, nz, bounds)`](../libs/pylbm/src/pylbm/stl_to_lbm.py#L120)
→ solid-object grid boxes → generated Fortran; uDALES via `python_udgeom` (IBM
blocks); the same `xie_castro_2008_STL.stl` drives all of them. The surrogate must
consume geometry **on the same grid as the fields**, and — critically — geometry
generalization is **out of scope** (§0): one trained model per (solver, geometry,
grid) tuple. That collapses the problem to a *static, baked-in* input:

- **The STL is never fed to the network.** It is voxelized **once, offline**, per
  (geometry, grid), into a static **solid/fluid occupancy mask** on the collocated
  grid (D3), reusing the existing tooling
  (`pylbm.stl_to_lbm.get_building_grid_indices` / `python_udgeom`) so the mask is
  bit-identical to where the *source solver* placed solid cells. No STL parsing at
  inference time.
- **Prefer a signed-distance field (SDF) over a raw binary mask** as the static
  channel: an SDF gives the encoder smooth wall-proximity gradients (sharper
  wakes, better near-wall behavior) and pools more gracefully than a hard 0/1
  edge; keep the binary mask alongside it for loss-masking and output
  re-application. (Decision flagged in §12.)
- **Three uses, mirroring the field pipeline:**
  1. **Static input channel** (SDF [+ mask]) concatenated to the state channels at
     the encoder, for every step.
  2. **Loss mask** — solid cells excluded from the per-variable MSE (§6.1) so the
     network is neither penalized nor credited for the trivially-zero interior.
  3. **Output re-application** — after decode, re-zero velocity in solid cells /
     enforce no-penetration (the "reapply the building mask" step in §4). A
     validation check asserts the surrogate's solid cells coincide with where the
     Fortran output is identically zero, so the obs operator, localization, and
     plotting stay consistent.
- **Storage** (geometry is fixed per model, so store it *once*, never
  per-trajectory): in the checkpoint (§7, `geometry.npy` + the STL path/SHA and
  voxelization params `nx/ny/nz/bounds` in the manifest) **and** once at the corpus
  root (§5). `prepare_pyupt` validates the composed geometry/grid against the
  checkpoint's, raising on mismatch (same pattern as the grid check in §8.2).
- **Upgrade path (not now)**: UPT is natively point/mesh-based, so *true* geometry
  generalization would feed STL **surface points as geometry tokens** to the
  encoder — but that is the AB-UPT/Transolver++ regime §0 explicitly ruled out.
  Recorded as a hook; the static channel is the correct choice for the
  fixed-geometry scope.
- **Middle-ground variant — a *fixed set* of known geometries (optional, no
  architecture change):** the static SDF/mask channel already carries the geometry,
  so training a *single* surrogate across a small, enumerated catalogue of
  geometries (each still fixed within a trajectory) is feasible by sampling
  geometry alongside inflow params in §5 and feeding the per-geometry SDF channel.
  This buys interpolation *within* the trained catalogue, **not** extrapolation to
  unseen buildings (see the generalization note at the end of this section). It
  costs more data/training and a richer corpus manifest (geometry id per
  trajectory); defer until the single-geometry surrogate clears the §13 GATE.

**Can a model trained on one geometry run on a different one?** Short answer: *no,
not reliably* — see the dedicated discussion at the end of §0-scope reasoning
captured in §14 (Generalization). A model trained on a single geometry has only
ever seen that one occupancy field, so the encoder/propagator weights are
specialized to its wake structure; presenting a new SDF at inference is
out-of-distribution and the fixed grid + baked normalization were fit to the
trained case. Geometry transfer requires either the *fixed-set* variant above
(works only within the trained catalogue) or the point-cloud UPT upgrade (the
ruled-out AB-UPT regime). Within one trajectory the geometry never changes, which
is consistent with all of the above.

## 2. Library layout (`libs/pyupt`)

Mirror the `pylbm` shape (`docs/codebase_guide.md` §8 recipe):

```
libs/pyupt/
  pyproject.toml                    # editable install + pixi pkg (see §9)
  src/pyupt/
    __init__.py
    forward_model.py                # ForwardModel(BaseForwardModel)      (§4)
    ensemble_forward_model.py       # EnsembleForwardModel(...) override  (§4, D2)
    model/                          # FAITHFUL JAX port of ml-jku/UPT (see §1.0)
      upt.py                        # UPT module: encoder→approximator→decoder
      encoder.py                    # ref "encoder": supernode msg-passing + perceiver pool
      propagator.py                 # ref "approximator": latent transformer time-stepper;
                                    #   + K-frame temporal attention (K=1 ⇒ ref Markov) (D4-ii)
      decoder.py                    # ref "decoder": perceiver cross-attn to query positions
      conditioning.py               # param (inflow_angle, velocity_mag, ...) embedding (deviation)
      layers.py                     # attention blocks, MLPs, pos-encoding (match ref numerics)
    data/
      generate.py                   # driver: solver ensembles → trajectory corpus (§5)
      dataset.py                    # lazy index-map windowing: whole Zarr trajectories →
                                    #   field-space (history, target) windows (§6.1);
                                    #   K-frame history slices + per-step interp params,
                                    #   encoded on the fly in the train step (not cached)
      normalization.py              # fit/apply per-variable standardization (§6.3)
      grid.py                       # grid metadata, collocation; STL→occupancy mask
                                    #   + SDF, reusing stl_to_lbm/python_udgeom    (D5)
    training/
      train.py                      # single + multi-GPU training entry (§6, §7)
      loop.py                       # step fn, rollout/pushforward loss, eval
      sharding.py                   # jax mesh / data-parallel helpers (§6.4)
      checkpoint.py                 # Orbax save/restore + manifest (§7)
    utils/
      state_io.py                   # xarray.Dataset ↔ model tensor; owns the K-frame
                                    #   latent history buffer + time-axis trimming
      params_io.py                  # params Dataset → conditioning; sparse→dense
                                    #   per-step interpolation (matches Fortran)      (D4-i)
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
| `run_single(state, params, sim_name)` | (1) `params` → **per-step** conditioning sequence (`utils/params_io`, interpolating sparse time-varying values — see D4-i). (2) resolve the initial field(s) (see **cold-start** note below) → normalized initial latent(s), seeding the K-frame history buffer (D4-ii). (3) autoregressively roll the latent propagator `num_outputs` steps, sliding the K-frame window and feeding each step its conditioning slice. (4) decode each step, denormalize, reapply the building mask, assemble an `xarray.Dataset` with `time` + the source grid coords, matching the **time-axis contract** below. Return it. |
| `_apply_inflow_settings(params)` | Build/store the per-step conditioning **sequence** on `self` (no files to edit, unlike Fortran). If params carry a `time` dim with `N ≪ num_outputs` sparse values, **interpolate onto the dense per-step grid** (D4-i: linear; angle via sin/cos; spin-up plateau); scalar params broadcast to all steps. |
| `save_results` / `_clean_output` | `save_results` = `self._save_results` (NetCDF, base helper). `_clean_output` = no-op (no scratch files). |

Notes:
- **Warm start / rollout**: the multi-window pattern already feeds each window's
  final `state` into the next `run_single`; UPT consumes it natively as the new
  initial condition. No restart-file machinery (unlike pylbm's `.uf` files).
  With the K-frame history (D4-ii) the warm start is a length-1 history at
  window start and grows to `K` within the window; if a window hands off a state
  with `time` length > 1, `state_io` seeds the buffer with up to the last `K`
  frames so cross-window context is preserved without restart files.
- **`get_states` / on-disk mode** work unchanged via the base class.

**Cold-start initial condition (decision — see §12).** Unlike a Fortran solver,
a NN surrogate cannot synthesize a physically consistent field from `state=None`
by spinning up internally — there is no "cold-start prior" to roll from. ESMDA
window 0 frequently passes `state=None`. The plan must commit to one of:
- **(a) Require an IC**: `run_single` raises if `state is None`; the caller
  always supplies a spun-up field (e.g. a canned per-regime field shipped with
  the checkpoint). Simplest, but pushes the burden onto every entry script.
- **(b) Canned IC bank**: ship a small set of spun-up fields keyed by parameter
  regime in the checkpoint; `state=None` selects the nearest by `params`.
- **(c) Params→initial-latent encoder**: train an extra head mapping the
  conditioning vector to an initial latent, so the surrogate can start from
  `params` alone. Most flexible — but understand what it *is*: **a second
  generative model** (params → plausible latent) with its own training data, its
  own out-of-distribution behavior, and — critically — it manufactures the IC that
  then seeds the autoregressive rollout, so its errors **compound with the
  rollout-drift / analysis-OOD risk** (§14), they don't sit beside it. It is a
  phase, not a head bolted on in a sentence.

Recommended: **(b) canned IC bank** as the default — it is cheap, unit-testable
without any extra training, and closest to how ESMDA window-0 ICs are actually
produced (a spun-up field per regime). **(a)** is the P0/P3 stopgap so the forward
model is usable before the bank exists. Escalate to **(c)** only if (b) proves
insufficient in P5, and then **as its own phase with its own metric** (does the
generated IC, rolled out, stay on-manifold?), budgeted alongside the §13 GATE work
rather than hidden inside P3.

**Time-axis contract (must match the Fortran backends exactly).** pylbm
concatenates `out_*_F<t>.nc` and **drops the spin-up-output prefix**, trimming to
`simulation_time / output_frequency` outputs (`docs/codebase_guide.md` §7). The
`TemporalObservationOperator` then aggregates in fixed `interval_size` chunks, so
the surrogate's output `time` length **and** spacing must equal what the source
backend produces *after trimming* — not the full spun-up series. §5 training data
deliberately includes the spin-up transient, but `run_single` must emit the
**trimmed** window (post-spin-up, `num_outputs = round(simulation_time/
output_frequency)` frames at `output_frequency` spacing). Any off-by-spin-up
mismatch silently misaligns every assimilation window. `utils/state_io` owns this
trimming and a round-trip test must assert the time coord matches the Fortran
backend's on an identical config.

- `pyupt.EnsembleForwardModel(BaseEnsembleForwardModel)`: implements
  `_create_new_forward_model` (cheap — share the immutable weights, clone only
  per-member result dirs) **and** overrides `run_ensemble` for on-device
  batching (D2). The override is **more than "stack → run → split"**; it must
  reproduce the base method's branching ([`run_ensemble`](../src/pyurbanair/base_ensemble_forward_model.py#L482)):
  - **All three save modes.** When `save_on_disk`, write per-member
    `{sim_name}_{i}.nc` into `self.results_dir` exactly as the base does, so
    ESMDA's on-disk path (`step_{i}/state_*.nc`, re-opened via `get_state`/
    `get_states`) keeps working. When `save_in_memory`, return the
    `concat`-along-`ensemble` dataset.
  - **`state` may be a `pathlib.Path`.** Rollout/ESMDA pass a directory of
    per-member files; the batching code must load-and-stack via the existing
    `get_member_state` (which already handles `Dataset` | `Path` | `None`)
    before forming the batched array, not assume an in-memory `ensemble` dim.
  - **`rollout_step` increment** ([L461-464](../src/pyurbanair/base_ensemble_forward_model.py#L461))
    must still advance per call so multi-window driving stays consistent.
  - The failure policy is largely moot (a NN forward pass does not raise
    `CalledProcessError`); keep `"raise"` default and skip the
    resample-from-successes plumbing.

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
   parameter vector **resampled to one value per output frame** (the Fortran
   solver already interpolated the sparse input internally, so record the
   per-frame effective params — not just the sparse knots — so training tensors
   align frame-for-frame with D4-i's dense per-step convention) and grid metadata.
   Convert staggered uDALES output via `interpolate_grid` (D3). Use
   `parameter_time_series/` to drive transient inflow here so the corpus actually
   contains time-varying boundary conditions to learn from. The **geometry
   mask/SDF is identical across all trajectories** (fixed geometry, D5) — voxelize
   the STL once and store it at the **corpus root**, not per-trajectory.
5. Persist to a **corpus directory** as **Zarr** (chunked along `time`/sample;
   better random access for training than many small NetCDFs). **Store each
   trajectory whole — never pre-windowed.** Overlapping `(history, target)`
   windows duplicate every frame ~`K×` and the rollout horizon `H` grows during
   training (§6.1), so a materialized window corpus is both an order-of-magnitude
   storage blow-up and stale the moment the curriculum advances. Windowing is a
   **lazy index map** in `data/dataset.py` (§6.1), not a generation-time artifact.
   Keep a JSON manifest: solver, grid, bounds, param ranges, counts, git SHA of the
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

**Sizing (do this back-of-envelope *before* P1 — it decides feasibility).** This
is the dominant cost (§14) and the plan must not enter P1 without a number.
Compute both axes from the actual target config:

- **Storage** = `N_traj · N_frames · nx·ny·nz · n_vars · bytes`. Worked example at
  a modest `64³` grid, `n_vars=3` (`u,v,w`), `fp16`, `N_frames=100`:
  `64³·3·2 B ≈ 1.5 MB/frame → ~150 MB/trajectory → ~150 GB per 1000 trajectories`
  (fp32 doubles it; adding `pres` is +33%; a `128³` grid is ×8 → multi-TB at 1000
  trajectories). Pick the dtype and grid deliberately — they swing the corpus by an
  order of magnitude.
- **Compute** = `N_traj · wall_time_per_traj / effective_workers`. On the single
  3950X box capped at `workers≈4` (per `docs/ensemble_scaling.md`), even
  1-minute-per-trajectory LBM runs at 1000 trajectories is `1000·1 min / 4 ≈ 4 h`;
  realistic transient runs are minutes each, pushing this to **days–weeks** on one
  box. This is the concrete justification for "scale out on a cluster," and the
  number that decides whether P1 is feasible on available hardware.
- A transformer surrogate of transient turbulent flow realistically needs
  **hundreds–low-thousands** of trajectories for the inflow sweep; start the GATE
  on the **small** corpus (§13) and only commit to the full corpus once the GATE is
  cleared. Record the chosen `N_traj`, grid, dtype, and resulting GB / GPU-hours in
  the §12.7 decision before generating.

**Anti-inverse-crime guard (enforce structurally, not by discipline)**: when
evaluating ESMDA with `pyupt` as the *assim* model, draw the *truth* from a Fortran
solver (the config already mounts `model/` twice, `docs/codebase_guide.md` §5) so
you measure real surrogate error, not a model matched to itself. Because
"accidentally matched to itself" is an easy false win that invalidates the entire
P5 result, the P5 harness should **assert** truth and assim are not the same
backend / corpus (e.g. compare `solver_name` and the corpus manifest id) and refuse
to report a posterior-accuracy number when they coincide — a check, not a comment.

## 6. Training

### 6.1 Task formulation
- History-conditioned learner in latent space:
  `({z_{t-K+1..t}}, {param_{t-K+1..t}}, param_{t+1}) → z_{t+1}` (D4-ii). With
  `K = 1` this is the plain Markov `(state_t, conditioning) → state_{t+1}`. Params
  are already the dense **per-step** sequence (D4-i) by the time they reach training
  tensors.
- **Training is in field space; latents are recomputed every step — not cached.**
  The `{z}` above is the *inference-time* ring buffer (D4-ii). During training the
  encoder weights change every gradient step, so any cached latent is stale after
  one optimizer step. Therefore a training window stores **raw fields**, and the
  encoder runs *inside* the train step on each history frame; the latent buffer is
  reconstructed on the fly. `data/dataset.py` never materializes or caches latents.
- **Rollout/pushforward loss is mandatory** for ESMDA usefulness: train with
  multi-step unrolled predictions (curriculum: grow horizon over training, or
  pushforward with stop-gradient on the warm-up portion) so error does not
  blow up over a window. This is the property AB-UPT/Transolver++ never
  demonstrate and the main risk for a surrogate driven repeatedly by ESMDA.
- Loss: normalized per-variable MSE on the field; optionally a spectral/gradient
  term to preserve wake sharpness; mask out solid cells.

#### 6.1.1 Windowing the trajectory corpus (`data/dataset.py`)
A corpus trajectory has hundreds of frames; training consumes
`(history, target)` windows. **Window lazily via an index map — do not materialize
windows** (see §5 for why). `dataset.py` stores trajectories whole and builds a
flat index

```
sample_id → (trajectory_id, t_start, history_len)
```

`__getitem__(sample_id)` slices the Zarr lazily, applies the stored normalization
(§6.3) on load, and returns one **field-space** window record (`H` = current
rollout horizon, `C` = `len(u,v,w[,pres])`):

```
hist_fields    [K, C, Z, Y, X]   # K history frames; left-padded if history_len < K
hist_params    [K, P]            # dense per-step params at those frames (D4-i)
hist_mask      [K]               # 1 = real frame, 0 = left-pad
future_params  [H, P]            # boundary conditions for the H rollout steps
target_fields  [H, C, Z, Y, X]   # the next H frames (pushforward targets, true frames only)
```

Mechanics, each of which the plan must honor:
- **Sliding window with stride `s`**: a trajectory of `T` frames yields
  ~`(T − K − H + 1)/s` windows — this is what turns hundreds of trajectories into
  the tens-of-thousands of samples the transformer needs.
- **Variable-length start windows are explicit index entries, not just masking.**
  To actually train the `1…K` regime (the cold-start / early-window regime ESMDA
  hits), emit windows anchored at `t_start = 0` with `history_len = 1, 2, …, K−1`
  (left-pad + `hist_mask`), in addition to the interior full-`K` windows. Masking
  alone, applied only to interior windows, never exercises short history.
- **`H` is a curriculum-controlled slice length, not a baked choice.** Growing `H`
  (§6.1 pushforward curriculum) just lengthens `target_fields`/`future_params` and
  shrinks the valid `t_start` range — the index is recomputed, nothing is
  re-materialized. `H = 1` is the one-step learner; `H > 1` unrolls the propagator
  on its own decoded output (stop-gradient on the warm-up portion). Targets are
  always *true* frames; predicted intermediates are produced at step time.
- **Off-manifold IC injection happens at step time, not in the window.** The
  dataset emits clean frames; `training/loop.py` optionally perturbs `hist_fields`
  (ensemble-mix / noise) before encoding, so the §13 GATE B2 (analysis-OOD)
  distribution is tunable without re-windowing and the corpus stays clean.
- **Split by trajectory *before* building the index** (§5) so no window straddles
  the train/val boundary and no two windows of one trajectory land in different
  splits. The index builder is where this is enforced.
- **Batching**: stack `B` window records; `hist_mask` handles variable history
  length within a batch, so mixed `1…K` windows batch together.

### 6.2 Conditioning
Embed `inflow_angle` (use sin/cos to respect periodicity), `velocity_magnitude`,
and (uDALES) `pressure_gradient_magnitude` via an MLP → FiLM/added tokens in the
propagator. Time-varying params feed a per-step conditioning sequence — and that
sequence is the **dense, interpolated** one produced by `params_io` (D4-i), so the
network never sees the sparse `N`-value series; it always receives one
conditioning vector per step. In the history-aware propagator (D4-ii) each of the
`K` latent frames is tagged with the param embedding active at that frame, and the
target step `t+1` additionally receives `param_{t+1}` as the step's boundary
condition.

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
  geometry.npy             # baked static occupancy mask (+ SDF); STL path/SHA in manifest (D5)
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
  nx: ${domain.nx}          # validated against the checkpoint, NOT used to size the model
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

> **The grid comes from the checkpoint, not from `domain`.** A trained model
> pins its grid in `grid.json` (§7). The `nx/ny/nz/bounds` above are passed so
> `prepare_pyupt` can **validate** that the composed `domain` matches the
> checkpoint and **raise on mismatch** — they must *not* be used to size the
> network, or composing with a different `domain` than the checkpoint was
> trained on would silently produce wrong results. The forward model reads its
> true grid from the checkpoint.

### 8.2 `hydra_helpers` additions
- Add `prepare_pyupt(forward_model, ...)` (replaces `compile` — loads/validates
  the checkpoint **against `cfg.domain` (grid match) and `cfg.time` (output
  count/spacing)**, raising on mismatch, then warms the JIT). **Keep the heavy
  NN imports function-local** inside `prepare_pyupt`: note that
  [hydra_helpers.py:16-17](../src/pyurbanair/config/hydra_helpers.py#L16) already
  imports `pylbm` and `pyudales` at module top and every script imports this
  module, so the only way to preserve the "composing `model=pylbm` never imports
  `pyupt`" invariant is the function-local import pattern `pypalm` uses in
  `clean_outputs` ([L60](../src/pyurbanair/config/hydra_helpers.py#L60)) — a
  module-top `import pyupt` here would defeat it.
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
**`history_length` K (D4-ii)**, data corpus path, sharding) consumed only by
`scripts/generate_upt_training_data.py` and `scripts/train_upt.py`. Keep training
config **out of** the inference `model/` group; the trained checkpoint snapshots
its own training config (§7) — and since `K` is baked into the architecture, it
lives in the checkpoint manifest (§7) and `prepare_pyupt` reads it from there, not
from the inference config.

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

**Cross-grid caveat for joint state estimation.** D3 means a `pyupt` *assim*
model emits a collocated `x,y,z` state even when the *truth* is staggered uDALES.
For `ParameterESMDA` this is irrelevant (only the observation vectors are
compared, and both pass through their own dim mappings). For
`StateAndParameterESMDA` the augmented vector is the *assim* model's state, so the
update is self-consistent — but any direct truth-vs-assim **state** comparison or
overlaid plotting is cross-grid and must interpolate one onto the other first.
Flag this in the §11 validation tooling rather than silently differencing arrays
on mismatched axes.

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
will under-disperse relative to the Fortran solver. This is not a polish item: for
a *deterministic* model driving ESMDA, spread collapse makes the posterior
overconfident and can invalidate the whole assimilation. Treat calibrated spread as
a **P5 pass/fail bar** (rank histograms / spread-skill, §13), and add model-error
inflation or an ensemble of surrogates / MC-dropout as needed to pass it — not as
an optional afterthought.

## 11. Testing & CI

- `libs/pyupt/tests/`: unit tests for `state_io`/`params_io` round-trips, a
  **tiny** UPT (2 layers, small latent) training-step test, a rollout-shape test,
  and the **CPU smoke-training stage (§11.1)** that runs the full
  window→encode→propagate→decode→loss→optimizer loop across varied shapes.
- **Reference parity test (§1.0)** — scoped to the **encoder, decoder, and the
  K=1 propagator only** (the parts we actually port; the K>1 propagator is a new
  design validated by behavior, not parity). With the PyTorch reference
  (`ml-jku/UPT`) available, load identical weights into both it and the JAX port on
  a tiny config and assert forward-pass outputs match to tight tolerance (per-stage,
  then end-to-end at K=1). This guards the *port* against silent transcription drift
  — it is **not** the acceptance gate for the surrogate (we never reuse upstream
  weights, §1.0); the behavioral GATE (§13) is. Mark it skippable when
  PyTorch/the reference isn't installed (e.g. lean CI), but run it in the dev/cuda
  envs. If §1.0's prerequisite check finds no comparable transient reference config,
  this degrades to a structural port review + shape/gradient tests.
- `tests/` (top-level): a `compose_test_cfg(["model=pyupt", ...])` test that
  `run(cfg)` works end-to-end against a tiny checkpoint fixture; a regression
  test that composing a non-UPT model does **not** import `pyupt`
  (mirror `test_palm_target_does_not_import_for_non_palm_composition`).
- **Prefer train-on-the-fly (1 step) over a committed checkpoint fixture**: §7
  makes `models/` git-ignored, so a binary checkpoint can't live in the repo
  anyway. A CI fixture that builds a tiny UPT, runs one optimizer step, and
  writes a throwaway checkpoint to `tmp_path` keeps the suite GPU-free and forces
  `state_io`/`params_io` to work without a real trained model. Gate any
  GPU/multi-GPU training tests behind a marker.

### 11.1 CPU smoke-training stage (plumbing, not accuracy)

A dedicated, **CPU-only** stage whose sole purpose is to prove the training code
*runs end-to-end* — **accuracy is explicitly not measured here** (that is the §13
GATE). It runs the full pipeline on a **tiny synthetic corpus** built in `tmp_path`
(no solver, no real data, no GPU): generate a handful of random trajectories,
window them (§6.1.1), build a 2-layer/small-latent UPT, and run **a few optimizer
steps** (e.g. 3–5). Fast enough for ordinary CI under `JAX_PLATFORMS=cpu`.

**Crucially, this is *several* tests, parametrized over shape**, because the most
likely place the plumbing breaks is a hard-coded or mismatched dimension. Cover the
cross-product (kept small):

- **Trajectory length `T`** — e.g. `T ∈ {2, 3, 7, 20}`. This exercises the
  windowing index map (§6.1.1) at different window counts, including:
  - `T` too short for a full `K + H` window → must yield only short-history /
    truncated windows (or be skipped cleanly), never an out-of-range slice;
  - `T` large enough for many interior windows.
- **Number of spatial points / grid shape** — vary `(Z, Y, X)` (e.g. a tiny
  `4×4×4`, an anisotropic `2×8×6`, and a non-cubic `1×16×16`) so the
  grid-agnostic encoder (supernode pooling) and decoder (query positions) are
  exercised at different point counts and aspect ratios, and `state_io`'s
  flatten/unflatten is shape-correct. Where the encoder subsamples supernodes,
  vary the supernode count too.
- **History `K ∈ {1, 3}` and horizon `H ∈ {1, 3}`** — `K = 1` hits the Markov
  bypass (§1.0), `K = 3` hits the temporal-attention path and `hist_mask`;
  `H = 1` is one-step, `H = 3` unrolls the pushforward loop. Include at least one
  variable-length-history window (`history_len < K`) so the mask/left-pad path is
  covered.
- **Channels `C ∈ {3, 4}`** — with and without `pres`.

Per parametrization, assert only **plumbing invariants** (shape + finiteness, not
error magnitude):

1. The window record shapes from `dataset.py` match the spec in §6.1.1 for the
   given `(T, K, H, C, grid)`, and the produced window count matches the index map.
2. A forward pass produces `target_fields`-shaped output; the loss is a finite
   scalar.
3. Gradients are finite (no `NaN`/`Inf`) and the parameter PyTree **changes** after
   the optimizer steps (catches frozen/disconnected params).
4. The unroll (`H > 1`) feeds decoded output back without shape drift across steps.

Keep one combination wired to also save+reload a throwaway checkpoint (§7) so the
`registry`/manifest round-trip is smoke-tested too. This stage is the cheap,
always-on guard that the §13 GATE (which needs a real small corpus) is never the
*first* thing to exercise the training loop.

## 12. Decisions to confirm before coding

1. **Framework (D1)**: JAX+Equinox (recommended; repo-consistent, and JAX *is*
   the D2 batching path via `vmap`) vs PyTorch (matches the official UPT code,
   mature DDP/FSDP, but a second accelerator stack). *Biggest, least-reversible
   choice.* Note the differentiability argument is **not** a tiebreaker —
   ESMDA is derivative-free (see D1).
2. **Ensemble execution (D2)**: confirm in-process on-device batching as the
   default (vs reusing the forkserver process pool). Recommended: batching. The
   override must still honor all three save modes (§4).
3. **Cold-start initial condition (§4)**: (a) require an IC, (b) canned IC bank,
   or (c) params→initial-latent encoder. Recommended: (b) as default (cheap,
   testable, matches how window-0 ICs are produced), (a) as the stopgap; (c) is a
   *second generative model* whose IC errors compound with rollout drift — defer to
   its own P5 phase only if (b) is insufficient. *Blocks ESMDA window 0, which
   often passes `state=None`.*
4. **History length `K` (D4-ii)**: target 3–10. Larger `K` gives more temporal
   context (helps transient/wake memory) at higher cost; `K = 1` is the Markov
   baseline. Pick by the post-P2 rollout-error ablation (the GATE). Baked into the
   architecture, so it is fixed per checkpoint.
5. **Geometry representation (D5)**: binary occupancy mask vs signed-distance
   field as the static channel. Recommended: SDF (+ mask for loss/output
   re-application). Geometry is baked per model; the STL is voxelized offline with
   existing tooling, never fed to the network.
6. **Source solver for the first corpus**: pylbm (collocated, CUDA, fastest to
   iterate) vs pyudales/pypalm. Recommended: start with pylbm on the Xie–Castro
   geometry already in `examples/lbm/`.
7. **Corpus storage & budget**: Zarr layout + how many trajectories / how much
   GPU-hours for generation, and where checkpoints sync (shared scratch?).

## 13. Phased roadmap

- **P0 — scaffolding**: `libs/pyupt` skeleton, pyproject + pixi feature, lazy-import
  regression test, `state_io`/`params_io` with round-trip tests. No model yet.
- **P1 — data**: `scripts/generate_upt_training_data.py`, Zarr corpus + manifest,
  normalization fit, a small pylbm corpus end-to-end.
- **P2 — model + single-GPU training**: port the encoder/decoder faithfully from
  `ml-jku/UPT` and build the new K-frame propagator (§1.0); **pass the
  encoder/decoder + K=1 parity test (§11) first** (after confirming the reference
  ships a comparable transient config — §1.0 prerequisite), then add the one-step +
  pushforward loss, `scripts/train_upt.py`, checkpoint/manifest, rollout-error eval.
  The pushforward loss must already include perturbed / off-manifold ICs (§6.1,
  §14) so the GATE can test analysis-OOD robustness. **Stand up the CPU
  smoke-training stage (§11.1) as soon as the train loop is wired — before any real
  data or the GATE** — so the window→encode→propagate→decode→loss→optimizer path is
  proven across varied trajectory lengths and grid shapes on CPU first.
- **GATE (after P2) — go/no-go before scaling**: P1 (data generation) and P4
  (multi-GPU) are the two most expensive phases, and they currently precede the
  only phase (P5) that proves the surrogate is good enough for ESMDA. Insert an
  explicit gate here, on a *small* pylbm corpus. **Crucially, the gate must test
  the dominant risk (§14), not just the easy case.** Clean-trajectory rollout error
  is necessary but *not sufficient* — held-out trajectories are on-manifold by
  construction, whereas ESMDA feeds the surrogate cold-start ICs and off-manifold
  analysis increments, which is where rollout drift actually detonates. A gate that
  measures only clean rollout can pass green while P5 fails. The gate therefore has
  **three bars, all pre-stated**, evaluated over a full assimilation window's worth
  of steps:
  1. **Clean rollout-error-vs-horizon** on held-out trajectories (the §6.5
     baseline) — must stay below bar B1.
  2. **Analysis-OOD robustness (the load-bearing bar)**: roll from *perturbed /
     ensemble-mixed / noised* ICs and from at least one *synthetic EnKF analysis
     increment* (a linear combination of ensemble members, the §14 failure mode),
     and require rollout error to stay below bar B2. This is cheap to construct on
     the small corpus — no cluster, no real ESMDA loop — and is the only bar that
     actually de-risks P4/P5.
  3. **Cold-start sanity**: roll from the chosen IC strategy (§4 — bank/stopgap),
     not only from corpus ICs, and confirm it does not diverge.

  Only commit cluster time to P4 if **all three** clear their bar; otherwise iterate
  on model/loss/IC handling first. This makes the dominant-risk metric a gating
  milestone, not a P5 surprise.
- **P3 — forward model**: `pyupt.ForwardModel` + `EnsembleForwardModel` (D2),
  `conf/model/pyupt.yaml`, `prepare_pyupt`, `clean_outputs` branch, obs-operator
  registration; `run_forward_model.py model=pyupt` works.
- **P4 — multi-GPU**: `training/sharding.py` data-parallel mesh, pixi tasks,
  DelftBlue/SLURM launch; scale the corpus and model.
- **P5 — ESMDA validation**: surrogate-as-assim vs Fortran-truth runs (with the
  enforced anti-inverse-crime guard, §5); measure posterior accuracy, rollout
  stability over multiple windows, and **ensemble dispersion as a pass/fail bar —
  not a footnote**: compute rank histograms / spread-skill ratio and require
  calibrated spread, since a deterministic surrogate under-disperses (§10) and an
  overconfident posterior invalidates the assimilation. Document the uncertainty
  caveat and any model-error inflation needed to pass the dispersion bar.

## 14. Risks / open issues

- **Rollout drift** over long windows — mitigated by pushforward training (§6.1);
  the primary acceptance metric is rollout-error-vs-horizon, not one-step MSE.
- **Under-dispersion** in ESMDA (§10 caveat).
- **Parameter distribution shift**: ESMDA can push params outside the training
  prior; cover margins in §5 sampling and monitor for extrapolation.
- **State out-of-distribution from the analysis step** (likely the *dominant*
  surrogate risk): the EnKF analysis produces states that are linear combinations
  of ensemble members and need not lie on the data manifold the surrogate learned
  from clean solver trajectories. Feeding such an increment into an
  autoregressive surrogate is exactly where rollout drift detonates, and it is
  distinct from — and worse than — parameter OOD. Mitigation: the §6.1 pushforward
  training should include **perturbed / off-manifold initial states** (e.g.
  ensemble-mixed or noised ICs), not just clean trajectory frames, so the
  propagator is robust to analysis increments.
- **Data-generation cost** dominates effort (§5); plan cluster time.
- **Staggered-grid information loss** from interpolation (D3) — acceptable for a
  surrogate, but quantify against held-out uDALES data if that solver is used.
- **Geometry generalization (cannot transfer to an unseen building)**: a
  single-geometry surrogate (D5) is specialized to one occupancy field, fixed grid,
  and geometry-specific normalization, so running it on a *different* geometry is
  out-of-distribution and unreliable — expect large, unquantified error, not
  graceful degradation. There is no cheap fix at inference. The only supported
  routes are (a) the D5 fixed-set variant, which interpolates **within** an
  enumerated training catalogue but still does not extrapolate to new buildings, or
  (b) the point-cloud UPT upgrade (the AB-UPT regime ruled out in §0). Treat one
  checkpoint as valid for exactly the geometry it was trained on.

## 15. Detailed implementation plan (file-by-file)

This expands the §13 phases into concrete files, signatures, and responsibilities.
Decision defaults are taken from §12 as recommended: **JAX + Equinox (D1),
in-process `vmap` (D2), canned-IC-bank default (§4), SDF + mask geometry (D5),
pylbm / Xie–Castro as the first corpus (§12.6)**. New files are shown in backticks
(they do not exist yet); existing files are linked. Dependency order is
**P0 → P1 → P2 → GATE → (P3 ∥ P4) → P5** (§13).

### P0 — Scaffolding (no model, no GPU)
**Exit:** `libs/pyupt` installs, the lazy-import invariant holds, `state_io` /
`params_io` round-trip, CI is green with no trained weights.

| File | Responsibility |
|---|---|
| `libs/pyupt/pyproject.toml` | Mirror [libs/pylbm/pyproject.toml](../libs/pylbm/pyproject.toml): `packages=["src/pyupt"]`; deps `jax, equinox, optax, orbax-checkpoint, zarr, numpy, xarray`. No Fortran/MPI. |
| `libs/pyupt/src/pyupt/__init__.py` | Version only. **No top-level NN imports** (unlike pylbm's git-clone `__init__`) so importing the package is cheap and the lazy-import invariant holds. |
| `libs/pyupt/src/pyupt/utils/state_io.py` | `state_to_tensor(ds, grid) -> Array[T,C,Z,Y,X]`; `tensor_to_state(arr, grid, var_names, time_coords) -> Dataset`; `trim_to_window(ds, num_outputs)` (the §4 time-axis contract: drop spin-up prefix, reassign `time=range(num_outputs)`, matching [forward_model.py:401-408](../libs/pylbm/src/pylbm/forward_model.py#L401-L408)); `LatentHistory(K)` ring buffer (inference-time, §6.1). |
| `libs/pyupt/src/pyupt/utils/params_io.py` | `params_to_conditioning(params, num_steps, output_frequency, spinup_outputs) -> Array[T,P]`: sparse→dense **linear** interp, `inflow_angle` via sin/cos, spin-up plateau prepend — matching [`write_uvel_time_file`](../libs/pylbm/src/pylbm/utils/params_utils.py#L119); drop `pressure_gradient_magnitude` unless uDALES. |
| `libs/pyupt/src/pyupt/data/grid.py` | `GridMeta` (`solver_name, nx/ny/nz, bounds, coord arrays`); `build_occupancy_mask(stl_path, grid)` reusing [`get_building_grid_indices`](../libs/pylbm/src/pylbm/stl_to_lbm.py#L120); `mask_to_sdf(mask)` (signed EDT). |
| `libs/pyupt/src/pyupt/utils/registry.py` | `resolve_checkpoint(path_or_run_id) -> Path`; `load_manifest(dir)`. |
| `libs/pyupt/tests/test_io_roundtrip.py` | state↔tensor and params→conditioning round-trips; assert the trimmed `time` coord equals pylbm's on an identical tiny config. |
| [pyproject.toml](../pyproject.toml) | Add `[tool.pixi.feature.pyupt.*]` (NN deps + `pyupt = {path=..., editable=true}`); add to `cuda`, `delftblue`, `dev` — **not** `default` (§9). |
| [tests/test_hydra_config.py](../tests/test_hydra_config.py) | Add `test_pyupt_target_does_not_import_for_non_palm_composition`, mirroring the pypalm assertion. |

### P1 — Data generation (small corpus)
**Exit:** a small pylbm Xie–Castro corpus on disk + fitted normalization, readable
by `dataset.py`. **Do the §5 sizing table first** and record `N_traj`, grid, dtype,
GB, est. hours in `conf/upt/` before generating.

| File | Responsibility |
|---|---|
| `scripts/generate_upt_training_data.py` | `def run(cfg)` + `@hydra.main`. Build solver+ensemble like [run_ensemble_forward_model.py](../scripts/run_ensemble_forward_model.py); sample params via [`create_parameter_ensemble`](../src/pyurbanair/config/hydra_helpers.py#L90) (+ `parameter_time_series/` for transient inflow); run **on-disk** over full `simulation_time`+spin-up; record per-frame **effective** params (§5/D4-i), not sparse knots. |
| `libs/pyupt/src/pyupt/data/generate.py` | `write_trajectory(...)` to Zarr (chunked along `time`/sample, **whole trajectories — not windowed**, §5); voxelize geometry **once** to corpus root (`geometry.npy`); write corpus manifest JSON. Convert staggered uDALES via `interpolate_grid` (D3). |
| `libs/pyupt/src/pyupt/data/normalization.py` | `fit_normalization(corpus, split="train")` (mask-aware per-var mean/std → `normalization.json`); `apply` / `invert`. Fit on **train split only** (§6.3). |

### P2 — Model, single-GPU training, CPU smoke stage, GATE
**Exit:** faithful encoder/decoder (parity-tested), new K-frame propagator,
pushforward training, **§11.1 CPU smoke stage green**, and **all three §13 GATE bars
cleared**. *(Pre-req: confirm the §1.0 reference-config check before relying on the
parity test.)*

| File | Responsibility |
|---|---|
| `libs/pyupt/src/pyupt/model/layers.py` | Attention block, MLP, positional encoding — **match upstream numerics** (norm eps/placement, attention scaling, init). |
| `libs/pyupt/src/pyupt/model/encoder.py` | Supernode message-passing + perceiver pool. **Faithful port** of ref encoder (§1.0). |
| `libs/pyupt/src/pyupt/model/decoder.py` | Perceiver cross-attn to query positions. **Faithful port** of ref decoder (§1.0). |
| `libs/pyupt/src/pyupt/model/propagator.py` | `__call__(latent_history[K], param_emb[K], target_param)`. **K=1 path special-cased to bypass the temporal block** → exact ref approximator (§1.0/D4-ii); K>1 = factorized temporal attention + length-1…K masking. |
| `libs/pyupt/src/pyupt/model/conditioning.py` | Param MLP (`inflow_angle` sin/cos, …) → FiLM/added tokens (§6.2). |
| `libs/pyupt/src/pyupt/model/upt.py` | Wire encoder→propagator→decoder; consume the static SDF+mask channel (D5). |
| `libs/pyupt/src/pyupt/data/dataset.py` | **Lazy index-map windowing** per §6.1.1: `sample_id → (trajectory_id, t_start, history_len)`; `__getitem__` slices Zarr, applies normalization, returns the field-space window record (`hist_fields/hist_params/hist_mask/future_params/target_fields`). Explicit short-history start entries; split-by-trajectory before indexing. |
| `libs/pyupt/src/pyupt/training/loop.py` | `train_step` (jit); one-step + **pushforward** loss (curriculum on `H`, stop-gradient warm-up); **inject perturbed / off-manifold ICs** at step time (§6.1/§14, feeds GATE B2); mask solid cells; rollout-error-vs-horizon eval. |
| `libs/pyupt/src/pyupt/training/train.py` | `def run(cfg)` + `@hydra.main`; Optax, bf16 compute / fp32 master, Zarr prefetch. |
| `libs/pyupt/src/pyupt/training/checkpoint.py` | Orbax save + the §7 artifact set (`weights/`, `config.yaml`, `normalization.json`, `grid.json`, `geometry.npy`, `manifest.json` incl. **K**, `metrics.json`). |
| `conf/upt/*.yaml` | New training group: model dims, optimizer, batch, rollout-horizon schedule, **`history_length` K**, corpus path, sharding (§8.3). Consumed only by the two training scripts. |
| `libs/pyupt/tests/test_smoke_training.py` | **§11.1 CPU smoke stage** — parametrized over `(T, grid, K, H, C)`; assert plumbing invariants only. Stand up **as soon as the loop is wired**, before real data/GATE. |
| `libs/pyupt/tests/test_parity.py` | **Encoder/decoder + K=1** reference parity (§11), skippable without PyTorch; degrades to structural review if no comparable transient ref config. |
| `scripts/eval_upt_gate.py` | Compute + check the **three §13 GATE bars** (B1 clean rollout, B2 analysis-OOD, cold-start sanity) on the small corpus. Hard go/no-go before P3/P4. |

### P3 — Forward model + Hydra wiring (inference)
**Exit:** `python scripts/run_forward_model.py model=pyupt model.checkpoint_path=…`
runs and honors the `BaseForwardModel` contract.

| File | Responsibility |
|---|---|
| `libs/pyupt/src/pyupt/forward_model.py` | `ForwardModel(BaseForwardModel)`. `__init__`: load checkpoint via `registry`, store grid/normalization/mask, `num_outputs=round(sim_time/output_freq)`, **lazy-init model on first `run_single`**. `run_single` per the §4 mapping (conditioning seq → resolve IC via canned bank, raise on `None` as stopgap → seed `LatentHistory` → autoregress → decode/denorm/reapply mask → trimmed Dataset). `save_results=self._save_results`; `_clean_output`=no-op. |
| `libs/pyupt/src/pyupt/ensemble_forward_model.py` | `EnsembleForwardModel(BaseEnsembleForwardModel)`. `_create_new_forward_model` shares immutable weights, clones only result dirs. **Override `run_ensemble`** honoring all 3 save modes + `state` as `Path` (via `get_member_state`) + `rollout_step` increment ([base L482-522](../src/pyurbanair/base_ensemble_forward_model.py#L482-L522)); batch via `vmap` with `vmap_chunk_size` sub-batching (D2). |
| `conf/model/pyupt.yaml` | Per §8.1: `checkpoint_path: ???`, `prepare._target_: …prepare_pyupt`, `num_parallel_processes: 1`; nx/ny/nz/bounds passed for **validation only**. |
| [src/pyurbanair/config/hydra_helpers.py](../src/pyurbanair/config/hydra_helpers.py) | Add `prepare_pyupt` (**function-local** NN imports — preserve invariant; §8.2) validating grid vs `cfg.domain` and output count vs `cfg.time`, then JIT warm-up. Add `elif model_name=="pyupt"` no-op to `clean_outputs` **and turn the `else` fall-through into a raise** ([currently L63-64](../src/pyurbanair/config/hydra_helpers.py#L63-L64)). |
| [observation_operator.py](../libs/data-assimilation/src/data_assimilation/observation_operator.py#L71) | Add `elif solver_name=="pyupt"` collocated mapping (same as pylbm). |
| `tests/test_pyupt_forward.py` | `compose_test_cfg(["model=pyupt", …])` end-to-end against a **train-on-the-fly tiny checkpoint** in `tmp_path` (no committed binary, no GPU; §11). |

### P4 — Multi-GPU + scale corpus
| File | Responsibility |
|---|---|
| `libs/pyupt/src/pyupt/training/sharding.py` | 1-D device `Mesh`; shard batch axis / replicate params; `pmean` grads; mesh-agnostic step fn (§6.4). |
| [pyproject.toml](../pyproject.toml) | pixi tasks `train-upt` (single) / `train-upt-multi`; DelftBlue SLURM launcher mirroring the `delftblue` env. |
| (corpus) | Scale §5 generation to the full `N_traj`; retrain. |

### P5 — ESMDA validation
- Run [run_parameter_esmda.py](../scripts/run_parameter_esmda.py) /
  [run_state_and_parameter_esmda.py](../scripts/run_state_and_parameter_esmda.py)
  with `model@truth_model=pylbm model@assim_model=pyupt`.
- **Enforced anti-inverse-crime guard** in the harness (assert truth ≠ assim
  backend/corpus, §5).
- Metrics: posterior accuracy, multi-window rollout stability, **ensemble
  dispersion as a pass/fail bar** (rank histograms / spread-skill, §10/§13); add
  inflation if spread collapses. Flag cross-grid truth-vs-assim state comparisons
  (§8.4).

### Resolve before P2 coding
1. **§1.0 prerequisite** — does released `ml-jku/UPT` ship a comparable transient
   config (ideally weights) to parity-test against? If not, the parity test
   degrades to structural review + shape/gradient tests (§11).
2. **Canned-IC-bank format** (§4) the GATE cold-start bar and P3 `run_single` will
   consume.
