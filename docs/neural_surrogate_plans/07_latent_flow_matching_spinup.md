# Plan 07 — Conditional latent flow matching for generative spin-up

**Status: implemented on feat/latent-flow-matching-spinup (2026-09-10); §3 statistical acceptance on real data not yet run.**

**Goal.** Generate a statistically developed flow state conditioned on geometry
and inflow parameters, then start a surrogate rollout without CFD spin-up.
During the **first ESMDA window, regenerate each member's initial state on every
forward evaluation using that member's current parameter values**, including
updated values after assimilation. This matches the LES cold-start lifecycle.
Later windows warm-start from the carried-forward state.

**Scope.** Train a conditional flow-matching velocity network in a frozen
`TadpoleAE` latent space. Reuse the DFT transformer's architecture family, with
independent generator sizing. Decode through the plain AE decoder. The new DFT
skip mixers, skip gates, and dynamics LoRA weights are not part of generation:
a generated latent has no input-state encoder skips.

The [Tadpole paper, §5.4](https://arxiv.org/html/2605.15284v1#S5.S4) demonstrates
latent flow matching with a separate generative network and an AE decoder,
including AEs adapted on the downstream autoencoding task. Its experiment does
not establish our conditional spin-up or ESMDA behavior; those require the
validation below. Flow time `tau` is an artificial integration coordinate,
independent of physical time and the parameter-history cadence.

## 0. Existing interfaces and decisions

- `TadpoleAE` processes state variables independently, sharing encoder/decoder
  weights. Its latent feature count per variable is `Cl = 256 / 512 / 1024` for
  S / B / L. The generator jointly models state latents as
  `(B, D, Zl, Yl, Xl)`, where `D = n_state_channels * Cl`.
- Spatial processing is selectable: `local`, `global`, or `halo`. Inherit
  `spatial_mode`, `encoder_crop_size`, and `halo_size` from the selected AE
  export and record them in the generator's `ae_kwargs`. They define the
  generator's training representation and cannot change at sampling time.
- Folded geometry (`encode_geometry: true`) has independent mask/SDF latent
  blocks. A separate `geometry_branch` instead supplies a feature pyramid and
  its stride-16 conditioning field. Support both; reject an AE with neither
  geometry path in the initial implementation.
- `ParamConditionedSubnetwork` supplies input FiLM for parameters/time and
  spatial FiLM for geometry. Reuse that class, but **do not inherit the DFT
  `_SUBNET_SIZES` hidden widths**. Input FiLM is the first conditioning design;
  its adequacy is an empirical question, not a guaranteed replacement for
  independent geometry tokens or per-layer conditioning.
- The surrogate has separate single-member and batched-ensemble cold-start
  paths. Both must support generation. A one-off directory of initial states
  prepared by `run_esmda.py` would prevent regeneration after parameter updates
  and is therefore not the integration mechanism.

## 1. Data and physical conditioning contract

Implement `SnapshotHistoryDataset(SnapshotDataset)` in
`datasets/snapshot_history.py`, inheriting geometry/SDF handling, lazy I/O,
`sample_index`, `grid_shape`, and `snapshot_collate` behavior. Add explicit
`param_vars` and `param_history_steps` (`Hp`, positive integer). Hoist
`TransitionDataset._load_params` to a shared reader and retain its existing
behavior for current callers.

Each item adds `params_hist: (Hp, P)`, oldest first, ending at the snapshot's
physical time. `time_stride` thins snapshot anchors only; histories use
contiguous saved times. Scalar parameters broadcast across the history.

Required checks and metadata:

- Pair state/parameter files by sample identifier, not just sorted file count.
  Validate time coordinates, lengths, ordering, finite values, and a consistent
  positive saved cadence `history_dt_seconds`; do not silently align by index.
- Save ordered state/parameter names, their units/conventions, geometry mask
  convention, coordinate ordering, grid spacing, boundary/forcing conventions,
  and supported training geometries/grids in the generator artifact. If source
  files omit units or conventions, require explicit config metadata.
- Require the generator dataset's ordered state variables to match the AE
  export. Validate geometry/SDF settings and physical grid metadata; strict
  weight loading checks tensor compatibility, not these physical contracts.
- `Hp` samples span `(Hp - 1) * history_dt_seconds`: 12 samples at five-second
  spacing span **55 seconds**. Validate cadence across trajectories. Arbitrary
  supplied deployment histories must match it; resampling is explicit.
- Missing leading history may repeat the first recorded parameters **only when
  dataset provenance confirms constant prehistory at those values**. Verify
  spin-up duration and the alignment of the first saved state/parameter time.
  Otherwise require the missing history or exclude anchors that need it.
- Include every varying forcing parameter needed to distinguish target states;
  omitted parameters must be documented constants. Store the resulting
  conditioning schema so deployment cannot substitute a different convention.

Post-spin-up does not imply equilibrium under the current forcing. A changing
inflow produces developed but transient states. Check stationarity of the
constant-forcing spin-up data and select valid targets using corpus-specific
burn-in/quality criteria; a configured spin-up duration alone is insufficient.
Training on transient snapshots with histories is allowed, but acceptance must
include the **constant-history case used by ESMDA cold starts**. Add suitable
constant-forcing data if that case is inadequately represented.

For the first implementation, use full snapshots (`random_crop_size: null`).
Internal local/halo AE processing is still available. Training a global latent
transformer on randomly relocated crops and sampling whole domains requires a
separate assessment of coordinates, global correlations, and conditioning.

## 2. Model: `TadpoleLatentGenerator`

Implement in `architectures/tadpole_latent_flow.py`. Own a frozen nested
`TadpoleAE`, a flow velocity network, and latent/parameter normalization buffers.
Delegate field I/O to the AE and spatial operations to the current shared
`_tadpole_spatial` helpers; do not copy the old local-only rearrange pipeline.

### Frozen AE and spatial layout

Build the AE from its export's `architecture` config, including spatial settings
and geometry branch, inject the saved state count, and strictly load its full
`weights.pt`. Save resolved `ae_kwargs` plus an AE weight fingerprint in the
new artifact. Deployment rebuilds with `skip_pretrained_load: true` and loads
one generator `weights.pt`, including `ae.*`; it does not need the original AE
folder. State normalization remains owned by the AE.

Freeze all AE parameters and override the generator's `train(mode)` so the
nested AE remains in evaluation mode after `BaseTraining` calls `model.train()`.
Use deterministic latent means (`latent_type: mode`) for this first version.
Stochastic posterior targets are a separate future experiment.

Define `encode_latents` and `decode_latents` around one canonical full-domain
latent grid:

1. Assemble masked, normalized working inputs with the AE helpers and pad using
   the inherited policy: global to stride 16, local/halo to the core size.
2. Use `encode_spatial` with the plain AE encoder for all three modes. Local
   regions have zero halo; global uses one whole field; halo retains central
   latent cores. Geometry-branch features must have full spatial grids with
   batch order `B*C_work` before the helper extracts regions; the existing
   local `_fold_geom_feats` crop layout is not the input to this helper.
3. Reshape `(B*C_work, Cl, Zl, Yl, Xl)` into `(B, C_work*Cl, Zl, Yl, Xl)`.
   Normalize and split generated state channels from known geometry channels.
4. For decoding, denormalize only state latents, reshape to
   `(B*C, Cl, Zl, Yl, Xl)`, and call `decode_spatial` with the plain AE decoder
   and matching geometry features/policy. There are no DFT residuals or mixers.
   Crop padding, denormalize physical state variables, and apply the fluid mask.

Encoding, geometry-only encoding, latent-stat estimation, and sampling must use
this same policy and respect encoder/decoder `max_internal_batchsize`. Test
local against the existing AE path and global/halo against their current paths
with numerical tolerances, including non-divisible rectangular domains.

### Velocity network: remove the fixed output-subspace restriction

For three variables and AE size S, `D=768`, while the DFT default hidden width is
144. Its final `Linear(144, 768)` restricts every velocity to a fixed subspace:
for any vector `a` orthogonal to that projection's columns,
`a^T v = a^T bias`. The proposed ODE could only translate those components of
its Gaussian initialization, not remove their noise or learn their conditional
distribution. Equal input/output widths do not resolve this restriction.

Use `ParamConditionedSubnetwork` with these **generator-specific** settings:

- `in_dim = D`; `hidden_size: null` resolves to `D`, rounded up to a multiple of
  `num_heads` if necessary. Reject explicit `hidden_size < D` for this direct
  velocity parameterization. Default `n_layers: 4`, `num_heads: 8`.
- The final projection has input width at least `D`, removing the forced
  low-rank output restriction. Keep its zero initialization; hidden activations
  are nonzero so its weights receive gradients on the first optimization step.
  This removes a structural blocker, not a guarantee of sample quality.
- `n_params = Hp*P + time_embed_dim`, default time embedding width 64. Concatenate
  normalized parameter history with a sinusoidal embedding of `tau` for FiLM.
- Geometry enters spatial FiLM. Fold mode uses deterministic normalized
  geometry latents (`D_geom = n_geometry_channels*Cl`); branch mode uses its
  stride-16 features and passes the full pyramid to the AE decoder.
- Keep the DFT model and defaults unchanged. Do not quote the paper's 12.3M
  parameter count for this newly sized network; report the actual count.

For global latent token count `N = Zl*Yl*Xl`, existing naive attention allocates
quadratic `B*heads*N*N` attention tensors. Profile the intended domains and
reject batches over a configured latent-token/attention budget before training.
A physical-cell budget alone is insufficient. If needed, implement exact
memory-efficient attention as a separately validated backend; do not silently
replace global attention with local windows to fit memory.

### Flow objective, normalization, and sampling

Compute raw encoder latent statistics once over a representative training
subset, accumulating in float64 over batch and spatial axes. Store per-channel
means/stds for all working latent channels; clamp standard deviations with an
explicit epsilon (default `1e-6`), check finite values, and report near-constant
channels. Freeze these buffers after installation and restore them on resume.
Do not recompute them from already-normalized latents. Cache provenance includes
AE weights, spatial policy, dataset/geometry identity, precision, and latent mode.

With normalized target `z1`, independently sample `z0 ~ N(0,I)` and one
`tau ~ U(0,1)` per sample:

```text
z_tau = (1 - tau) * z0 + tau * z1
v_target = z1 - z0
loss = mean_squared_error(velocity(z_tau, tau, history, geometry), v_target)
```

Train over state latent channels, including padded positions; no voxel fluid
mask is directly meaningful in latent space. Padding is not automatically
harmless: latent attention and halo decoding can propagate its errors into the
retained domain. Check padding sensitivity in evaluation.

Provide:

- `encode_latents(...)`: deterministic normalized state latents, geometry
  conditioning, decoder geometry features, and original spatial shape.
- `geometry_condition(...)`: derive the same conditioning without a state,
  under identical geometry/SDF padding and precision.
- `velocity(z, tau, params_hist, geom_cond)`: validate shapes/cadence/schema and
  return the direct flow velocity; no DFT latent residual addition.
- `forward(...)`: draw noise/time and return velocity prediction and target.
- `sample(params_hist, geometry, ..., initial_noise=None, generator=None)`:
  integrate from 0 to 1 using Euler, initially 50 steps, then decode. Reject
  simultaneous `initial_noise` and `generator`. Explicit initial noise enables
  reproducible member batching. Geometry and AE weights may be cached; sampled
  states may not be reused across changed conditioning.
- `set_normalization(...)`: install parameter stats only; AE state stats stay
  unchanged. Reject invalid stats and preserve the saved variable ordering.

Run frozen encoding/geometry conditioning and statistics in fp32 initially;
use bf16 autocast only for the velocity net during training, and fp32 sampling.
Disable autocast explicitly around frozen encoding. A cast after bf16 encoding
does not recover fp32 latents. Faster encoder precision can be added after
measuring representation/parity changes and recording it in the artifact.

Use tolerances for encode/decode normalization round-trips and batched execution;
bitwise equality is not promised. Fixed seeds guarantee identical initial noise
per member, not identical floating-point kernels across batch sizes/devices.

## 3. Training, artifacts, and acceptance gate

- Add `LatentFlowMatchingTrainer(BaseTraining)` in `training/flow_matching.py`.
  Move AE snapshot batch preparation into a shared base helper, preserving
  current geometry/SDF upload caching. The trainer adds `params_hist`, calls
  the generator, and computes fp32 MSE. No pushforward objective or dynamics
  skip/LoRA training is involved.
- Fix validation examples, batch ordering, noise, and flow times across epochs;
  reseeding flow noise alone does not fix shuffled examples or random crops.
  Keep validation RNG separate from training RNG. Fail on empty train/validation
  loaders, including trajectory-bucket samplers with `drop_last`.
- Add `train_latent_generator.yaml` and `train_latent_generator.py` with the
  standard `run(cfg)` + Hydra wrapper. Require AE directory, training root,
  model name, conditioning schema, and physical metadata. Resolve inherited AE
  settings; default to `Hp=12`, generator width as above, AdamW `1e-4`, and a
  configurable representative latent-stat pass. Set `random_crop_size: null`.
- Build geometry-compatible batches with `TrajectoryBatchSampler`, apply both
  physical-cell and latent-attention limits, install normalization, optimize
  only trainable velocity parameters, and train. Verify that the new network's
  output projection actually changes in the smoke test.
- Export `config.yaml`, complete best-validation `weights.pt`, `checkpoint.pt`,
  and `metrics.csv`. Save resolved `ae_kwargs`, generator settings, physical
  schema, normalization, geometry support, AE fingerprint, sampling settings,
  and data provenance. Rebuild without external AE/data directories. Register
  new classes in their subpackage and package-root exports.

**Statistical acceptance precedes ESMDA integration.** Add
`test_latent_generator.py` and compare held-out real states, frozen-AE
reconstructions of those states, and generated states under matched geometry
and histories. Evaluate the constant-history cold-start case separately.

Report conditional mean/RMS profiles, velocity distributions, energy spectra,
cross-component/Reynolds-stress statistics, diversity across noise seeds, and
divergence using grid spacing and a stencil-valid fluid mask near obstacles.
Compare rollout transients in kinetic energy/RMS and other application-relevant
statistics for all three initial-state sources. AE reconstruction error and
generation error are separate; no bound on rollout error follows from AE MSE.

Sweep Euler step counts (e.g. 25/50/100), compare conditioning against shuffled
or omitted history, and report sampling memory/time at deployment shapes.
Declare tolerances relative to held-out sampling variability and the AE baseline
before accepting the generator; finite samples and low flow loss alone do not
establish a useful spin-up distribution. The paper's W1/MMD/PQM metrics are
additional references, not substitutes for these conditional checks.

## 4. Deployment: regenerate on first-window forward evaluations

### Runtime and explicit template

Implement a reusable loader/sampler in `neural_surrogates/generative_spinup.py`.
Use it from `NeuralSurrogateForwardModel` and `NeuralSurrogateEnsembleForwardModel`.
Load the frozen generator lazily once per device/runtime; share read-only weights
where appropriate. Cache only immutable template/geometry/conditioning features.

Require an explicit deployment template carrying canonical coordinates and an
explicit obstacle mask; a training snapshot may supply this metadata, but its
velocity values are not the initial state. Canonicalize through the existing
regular-grid helper, validate state order, dimensions, coordinates, grid spacing,
mask convention and geometry fingerprint against both generator and stepper,
and write generated arrays on those canonical dimensions. Do not choose sample
0 from a possibly multi-geometry corpus or infer obstacles from generated zeros.
Initial scope is a validated supported geometry/grid; unseen geometries require
held-out geometry evaluation and an explicit supported-domain policy.

Add a nested config under `conf/model/neural_surrogate.yaml`:

```yaml
forward_model:
  spinup_source: generative
  generative_spinup:
    model_dir: ???
    template_path: ???
    seed: 0
    sample_batch_size: 8
    num_sampling_steps: null  # use the generator's validated saved setting
```

For each forward call, obtain each member's **current** parameter values from
the provided schedule in the generator's saved order. Take the first knot only
for time-varying variables; static variables already have one value per member.
Resolve missing variables through explicit `default_params` or raise. Validate
units/conventions and form the constant history by repeating those values `Hp`
times. The physical rollout then uses the full current parameter schedule.

Use base noise seeded by `(configured_seed, stable_member_index)` and construct
it per member before batching. **Regenerate on every cold forward call**, while
reusing the same base noise across ESMDA iterations by default. This provides
common random numbers: a changed parameter produces a newly conditioned sample
without introducing unrelated Monte Carlo changes into the assimilation map.
Never seed by batch position or reset all members to one shared draw. Independent
noise resampling across assimilation iterations is a separate future option.

### Single-model, ensemble, and ESMDA lifecycle

- In `_get_template_and_initial_state`, an explicit state always takes the warm
  path. If `state is None` and `spinup_source == "generative"`, sample from the
  current parameters, return the canonical generated snapshot, then use the
  normal history/rollout handling. Single forward runs therefore work too.
- In `NeuralSurrogateEnsembleForwardModel.run_ensemble`, add a generative branch
  before `_spinup_templates`: sample current member conditions in bounded
  batches and call `rollout_batched`. This path must not construct or run the
  CFD spin-up ensemble. Propagate stable member indices for RNG handling.
- Extend constructor validation to accept `generative`. Skip CFD preparation
  and backend cloning for it, as for `training_data`; implement cold-start
  generation rather than copying `training_data`'s cold-start error. Audit
  backend construction and `dirs`/cleanup accesses so a generator needs no CFD
  executable or preprocessing. Keep existing modes unchanged.
- In `run_esmda.py`, **do not pre-generate `_initial_states`, anchor the prior,
  or set `pin_initial_from_spinup` for generative mode**. Window 0 passes
  `state_input=None`. Parameter ESMDA already re-forecasts with updated params
  from the same initial-state argument, so each initial, intermediate, and final
  posterior forecast invokes generation again. Its initial parameter knot stays
  inferable. Tests must cover this lifecycle, not just a direct sampling call.
- Later windows pass their carried-forward states, so no generation occurs;
  keep the existing boundary-knot pinning for cross-window continuity. Supplied
  restart states likewise bypass generation, even in the first window.
- Initial support is parameter-inference ESMDA. A joint smoother that starts
  supplying independently analysed initial states needs an explicit policy for
  reconciling those states with conditional regeneration; reject that unsupported
  combination rather than silently ignoring one source of initial conditions.
- State-history stepping remains outside this plan: generate one frame and use
  the current repeated-frame warning/fallback if `H > 1`. Parameter-history
  conditioning is independent of this limitation.

Optionally save generated snapshots as diagnostics for each forecast, but those
files must not become the fixed warm-start input for later iterations of window
0. Generator failure raises with the member/conditioning context; there is no
silent CFD fallback. Keep `forkserver` and existing ensemble parallelism rules.

## 5. Required tests

1. **Dataset:** sample/time pairing, oldest-first history, verified plateau
   padding, static parameters, explicit variable ordering, cadence validation,
   `time_stride`, shared geometry collate, multi-geometry bucketing, unchanged
   `TransitionDataset` behavior after the reader refactor.
2. **Representation:** all three spatial modes crossed with folded/branch
   geometry; deterministic geometry-only conditioning equals conditioning from
   state encoding; encode/decode matches the AE within tolerance with nontrivial
   normalization, active geometry projections, rectangular padding, and chunked
   execution. Geometry channels are never generated.
3. **Model/training:** reject an undersized velocity hidden width; output shape
   and finite losses; nonzero output-projection gradients and actual updates;
   later gradients reach conditioning; frozen AE stays unchanged and in eval
   mode; robust constant-channel statistics; deterministic validation; attention
   budget failure; strict self-contained checkpoint/resume/reload and sampling.
4. **Sampling/deployment:** noise is identical per member across batch sizes and
   results agree within tolerance; distinct members receive distinct noise;
   changing conditioning reruns the generator; static/default/missing params,
   canonical output coordinates, obstacles, and physical metadata checks.
5. **ESMDA integration:** instrument a tiny conditional generator and run a real
   parameter-ESMDA first window with multiple updates and final forecast. Assert
   a generator call for every cold forecast with the current first parameter
   value, no initial-knot pinning, no fixed seed-state directory, and no CFD
   calls. Assert no generator calls for later warm windows or explicit restart
   states. Cover both single and batched forward paths and early rejection of
   unsupported joint-state assimilation.
6. **Acceptance:** run the held-out conditional/statistical checks in §3 before
   enabling generative spin-up for production assimilation. Record the chosen
   sampling steps and evaluated geometries/grids in the artifact/report.

## 6. Delivery phases

1. Dataset, shared parameter reader, snapshot batch preparation, and parity tests.
2. Spatially correct generator, full-width velocity network, trainer/config,
   self-contained export, unit tests and a real two-epoch training smoke test.
3. Conditional statistical evaluation and sampling-step/memory benchmarks;
   resolve acceptance failures before deployment work is considered complete.
4. Single/ensemble cold-start integration, first-window ESMDA regeneration tests,
   and config documentation. Update maintained `docs/neural_surrogates.md`,
   `docs/scripts_and_configs.md`, and the plan index when implementation lands.

Deferred experiments: latent caching after profiling, Heun, nonuniform flow-time
sampling, classifier-free guidance, per-layer time/history conditioning,
stochastic AE targets, random-crop generator training, and state-history seed
trajectories. None is a prerequisite for the initial validated Euler/FiLM design.
