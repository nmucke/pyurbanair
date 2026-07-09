# Plan 05-alt — Geometry-conditioned pre-training with forecast objective: full implementation plan

**Status: actionable implementation plan (2026-07-09).** This is the merged,
hand-off-ready successor to two documents that are both kept for the record:

- [Plan 05](05_latent_space_pretraining_extensions.md) — the original
  literature-driven extension plan (JEPA family, masking, aux losses). Its
  research basis, citations and anti-collapse recipes are reused here; read
  its Part I–II for the *why* behind several terms below.
- [Plan 06](06_geometry_conditioned_forecast_pretraining.md) — the critique
  of plan 05 and the five reframing ideas this plan is built on.

Like plan 05, this extends the **already-implemented plan-02 code**
(`TadpoleAE`, `AutoencoderTrainer`, `SnapshotDataset`,
`pretrain_autoencoder.yaml`) and feeds [plan 03](03_ae_to_timestepper.md).
It does not block plans 01/04.

**The five design commitments** (argued in plan 06, summarized here):

1. **Gate on adaptation, not probes.** The accept/reject metric for every
   phase is a standardized fine-tune-and-roll-out benchmark on held-out
   geometries. Latent probes / RankMe / spectra are diagnostics only.
2. **Geometry conditions the autoencoder itself**: `z = E(u | g)`,
   `û = D(z | g)`, via zero-initialized pathways that keep the plan-02 model
   byte-identical when disabled and HF weights loadable.
3. **The second pre-training objective is latent forecasting** (temporal,
   geometry-tokens-in-context, EMA targets) — the predictor is
   plan-03-DFT-shaped so pre-training directly warm-starts the downstream
   model. Same-time masked prediction is demoted to a final optional ablation.
4. **Data first**: multi-geometry corpus mechanics, nondimensionalization
   convention and symmetry augmentation land before any objective work.
5. **Physics-aware aux losses** (spectral, divergence, near-wall, denoise);
   coherent KL policy (LDM-style tiny β, **no free-bits**, explicit
   latent-noise smoothing).

---

## Part I — Goal and success criteria

**Goal:** a pre-train → fine-tune pipeline for urban atmospheric flow: one
autoencoder + latent forecaster pre-trained across O(10²) geometries, such
that a short plan-03 DFT fine-tune on a *new* geometry yields a stable
autoregressive surrogate usable as an ESMDA forward model.

**Success criteria (measured by the Part-IV benchmark):**

- S1. On held-out geometries (both tiers), the fixed-budget fine-tuned
  stepper beats (a) the same stepper fine-tuned from a *plain plan-02 AE*
  and (b) a from-scratch P3D trained on the same target-city budget, on
  rollout RMSE at the assimilation horizon.
- S2. Rolled-out fields stay physically valid: bounded divergence, no
  wall-flux violation growth, energy spectra within a band of the reference.
- S3. The sample-efficiency curve shows pre-training reduces target-city
  data needs materially (the headline plot: rollout error vs. fine-tune
  trajectories, pretrained vs. scratch).
- S4. A smoke-scale ESMDA twin experiment on a held-out geometry recovers
  parameters comparably to the CFD-backend baseline at smoke shape.

---

## Part II — What to implement

Every new knob **defaults to off / `none`** and must leave existing runs
byte-identical (repo no-op rule). All file paths below are the actual repo
paths. Register new public classes in the `neural_surrogates` `__init__`
exports like the plan-02 classes.

### A. Dataset layer (`libs/neural-surrogates/src/neural_surrogates/datasets/`)

**A1. Multi-root support + per-geometry batches** — new module
`datasets/multiroot.py`:

- `class MultiRootSnapshotDataset` (thin wrapper): takes
  `root_dirs: list[str]` (or a mapping name→dir), builds one
  `SnapshotDataset` per root with shared kwargs, concatenates. Exposes
  `sub_lengths` and per-item `root_id`. Geometry is per-root (the existing
  per-trajectory dedup already handles within-root sharing).
- `class PerRootBatchSampler(torch.utils.data.Sampler)`: every batch drawn
  from a single sub-dataset; roots sampled **uniformly** (MPP-style
  balancing), `drop_last` per root. This preserves the
  `snapshot_collate` shared-geometry fast path, keeps `torch.compile`
  shapes static per geometry, and is the batch-correctness fix for mixed
  geometries.
- Same wrapper pattern must be reusable for `TransitionDataset` (plan-03
  fine-tuning / multi-geometry doc §6 phase 0 lists the same need — build it
  generically: the wrapper takes the dataset class or pre-built datasets).
- Config: `dataset.root_dir` stays; new optional `dataset.root_dirs`
  (mutually exclusive, script-validated). Held-out tiers are *config
  conventions*, not code: e.g.
  `conf/neural_surrogate/dataset_splits/urban_corpus_v1.yaml` listing
  `train_roots`, `heldout_interp_roots` (unseen layouts from seen families),
  `heldout_extrap_roots` (unseen family, e.g. real city vs. procedural).

**A2. Temporal pairs** — extend `SnapshotDataset`:

- New args: `return_pair: bool = False`, `max_pair_offset: int = 4`
  (M; the pair target is `t + m·time_stride` steps with `m ~ U{1..M}` drawn
  per `__getitem__`), and `pair_params: bool = True`.
- When on, items gain `"state_next"` `(C, *grid)`, `"dt_steps"` (scalar
  tensor, the drawn `m`), and — because the forecaster conditions on
  physical params — `"params"` `(P,)` loaded per trajectory exactly as
  `TransitionDataset` does (reuse its param-loading helper; keep
  `param_names` handling identical so `get_normalization_stats` still
  works). Indexing must exclude the last `M·time_stride` frames per
  trajectory when pairing is on (reuse `TransitionDataset`'s index logic).
- `snapshot_collate` passes the new keys through `default_collate`
  unchanged (shared-geometry logic untouched).

**A3. Nondimensionalization** — extend `SnapshotDataset` (and mirror in
`TransitionDataset` behind the same knob, default off):

- New args: `nondim: bool = False`, `u_ref: float | str = "inflow_speed"`,
  `h_ref: float | None = None`. Velocities divided by the case's `u_ref`
  (a literal, or the name of a params entry / dataset attr to read
  per-trajectory) before anything else; `h_ref` recorded in the item dict
  for diagnostics (length scaling is implicit — grids are already in
  cells). The z-score stats pipeline then operates on nondim'd data
  automatically (stats are computed from dataset output — no change
  needed, but assert in the script that model and dataset agree on the
  `nondim` flag, like the SDF cross-checks).
- Frame-cadence check: the pre-train script logs each root's convective
  step `u_ref·Δt_save/Δx`; warn (not fail) when roots differ by >2× —
  resampling is a data-generation decision, not a training-time one.

**A4. Symmetry augmentation** — extend `SnapshotDataset`:

- New arg `augment: dict = {"y_flip": False, "yaw_90": False}` (per-item
  50% / uniform-in-4 application when enabled). Transforms applied
  consistently to: state (`y_flip`: flip y-axis and negate `v`; `yaw_90`:
  rotate grid about z and rotate `(u, v)` components), `state_next` (same
  transform — one draw per item), geometry mask, SDF channel, **∇SDF
  channels (rotate the vector components too)**, and any inflow-angle param
  (co-rotated). `yaw_90` additionally requires a cubic x/y footprint and
  BC symmetry — assert grid squareness; the config for y-periodic uDALES
  data must keep it off (see multi-geometry doc §4.7; document in the
  config comment).
- Augmented items break the shared-geometry dedup (transformed masks
  differ) — the collate fallback already handles per-sample geometry;
  accept the throughput cost, it's an opt-in.

### B. Architecture: geometry-conditioned `TadpoleAE`
(`libs/neural-surrogates/src/neural_surrogates/architectures/tadpole_ae.py`
+ minimal vendored diffs)

New constructor arg
`geometry_conditioning: str = "none"  # none | encoder | encoder_decoder`,
requiring `encode_geometry=True` and `sdf_features != "none"` when not
`"none"` (the conditioning input is the geometry block: mask + SDF
channels, `n_geom = 1 + n_geom_feature_channels`).

**B1. Encoder conditioning** (`encoder` and `encoder_decoder` modes): each
folded *flow* crop is encoded together with its co-located geometry
channels.

- Vendored change (small, owned): `TadpoleAutoencoder.__init__` gains
  `encoder_kwargs: dict = None` / `decoder_kwargs: dict = None`, forwarded
  to `_KLP3DEncoder(size, **encoder_kwargs)` / `_P3DDecoder(size,
  **decoder_kwargs)`. `P3DEncoder` **already accepts `in_channels`**
  (`_tadpole/architecture/p3d/core.py`), which reaches the first conv
  (`ConditionedEncoder3D.feature_embed`, a 3×3×3 Conv3d) — so encoder
  widening is a kwarg, not surgery. Pass
  `encoder_kwargs={"in_channels": 1 + n_geom}` when conditioning is on.
- Weight surgery helper (in `tadpole_ae.py`):
  `_inflate_input_conv(state_dict, n_extra)` — when loading pretrained /
  plan-02 weights whose `feature_embed.weight` is `(F, 1, 3, 3, 3)`, expand
  to `(F, 1+n_geom, 3, 3, 3)` with **zeros in the new input slices** (bias
  unchanged). A freshly converted model therefore computes exactly what the
  unconditioned model computes.
- Fold logic: the conditional path cannot use `TadpoleAutoencoder.forward`
  (it folds *all* channels to single-channel crops). `TadpoleAE` gets its
  own fold for this mode: rearrange state to
  `(B C U V W) 1 Xc Yc Zc`, rearrange the geometry block to
  `(B U V W) n_geom Xc Yc Zc`, `repeat` it across `C`, concatenate on the
  channel dim → encoder input `(B·C·U·V·W, 1+n_geom, Xc, Yc, Zc)`. Reuse
  the existing `encode()` rearrange as the template; respect
  `max_internal_batchsize` chunking (mirror the upstream loop).
- The geometry block **also still passes through the encoder as its own
  folded crops** (the plan-02 `encode_geometry` path, unchanged) — those
  geometry latents are the DFT/forecaster's geometry tokens. Their crops are
  conditioned on themselves (same concat), which is fine and keeps one
  encoder.

**B2. Decoder conditioning** (`encoder_decoder` mode): additive zero-init
injection at the latent — no decoder weight surgery.

- New module in the wrapper: `self.geo_to_latent = nn.Conv3d(n_geom,
  latent_channels, 1)` with **weights and bias zero-initialized**;
  `latent_channels` read from `self.ae.encoder.latent_size` (the
  `DiagonalGaussianDistribution` mean channel count — verify at
  implementation).
- Per flow crop: pool the geometry crop to the latent grid
  (`F.adaptive_avg_pool3d(g_crop, latent_grid_shape)`), then decode
  `D(z + geo_to_latent(g_pool))`. Zero-init ⇒ exact no-op at conversion.
- `decode()`'s public signature grows an optional `geometry_latent_input`;
  plan 03's stepper must call it accordingly when the AE was trained with
  `encoder_decoder` (record the mode in the AE's `config.yaml`; plan 03
  cross-checks it like the SDF flags).

**B3. Contract preservation:**

- `geometry_conditioning: "none"` must leave the module tree, state-dict
  keys and outputs **byte-identical** to today's model (unit-tested).
- `weights.pt` / `encoder.pt` / `decoder.pt` artifact semantics unchanged;
  `geo_to_latent` lives in the `TadpoleAE` wrapper state (inside
  `weights.pt`, not `decoder.pt`) — plan 03 loads the wrapper anyway. Note
  this in the artifact docs.

### C. Auxiliary losses (`training/aux_losses.py`, new module)

Plan 05's module design, reduced to the physics-first menu. One small class
per term, uniform interface:

```python
class AuxLoss:  # informal protocol
    def __call__(self, ctx: AuxContext) -> tuple[torch.Tensor, dict[str, float]]: ...
```

`AuxContext` is a small dataclass the trainer fills: `state`, `recon`
(working space), `geometry`, `geom_features`, `latent` (sampled z), `model`.
`AutoencoderTrainer` composes whichever terms the config enables, adds
`weight * loss` to the total, and appends each term's logs to the
`_aux_terms` dict (which already lands in `metrics.csv`).

- **`Denoise`** *(kept verbatim from plan 05 R1a — its best
  cost/benefit term)*: with prob `prob`, encode
  `ũ = u + σ_in·‖u‖_rms·η` (per-sample relative noise, DPOT's ablated
  optimum σ_in ≈ 5e-4; sweep 5e-4…1e-2) but reconstruct clean `u`.
  Separately, latent robustness: one extra decode of
  `z + σ_z·std(z)·η` (σ_z a few percent) with recon loss against `u` —
  **this is the term that makes a latent stepper viable** (the stepper's
  error is a latent perturbation; a learned VAE σ collapses, the explicit
  term keeps the decoder flat around data latents). Config:
  `{input_sigma_rel, latent_sigma_rel, prob}`.
- **`NearWall`** *(kept from plan 05 R1c / MARIO)*: per-cell weight
  `w(x) = 1 + w_bl · max(0, (τ − d̂)/τ)²` with `d̂` the normalized wall
  distance from the SDF channel, `τ` ≈ 2 cells…canyon scale. Implemented as
  a *weight map factory* consumed by the trainer's masked-MSE (so it
  composes with every other recon term rather than being its own loss).
  Config: `{weight: w_bl, tau_cells}`.
- **`Spectral`** *(new; the anti-smoothing term, replaces plan 02's GAN as
  first resort)*: radially-binned 3D energy spectrum per crop (rFFT over
  the crop, `E(k)` by shell-sum over channels),
  `L_spec = Σ_k (log(Ê(k)+ε) − log(E(k)+ε))²`, averaged over crops.
  Compute on the working-space recon of *fluid-dominated* crops only
  (skip crops with <50% fluid — walls corrupt the spectrum). Config:
  `{weight, min_fluid_frac}`.
- **`Divergence`** *(new)*: central-difference `∇·û` on the assembled
  (unfolded) physical-units reconstruction, interior fluid cells only
  (mask out any cell whose stencil touches a solid or domain boundary);
  `L_div = mean((∇·û)²) / mean((∇·u)²+ε)` — normalized by the data's own
  discrete divergence so imperfect LES/LBM data doesn't make the target
  unreachable. Couples u/v/w through the loss despite the channel fold.
  Config: `{weight}`.

**KL policy (decision, not a knob):** stay LDM-style — `kl_weight ≈ 1e-6`,
`latent_type: sample`. **No free-bits** (at β=1e-6 a per-channel floor is
numerically invisible — plan 06 §C-5). Dead-channel monitoring is a
diagnostic (per-channel latent std in `metrics.csv`), not a loss.

### D. Latent forecaster (`training/forecaster.py`, new module)

The headline objective. Design constraint that overrides all others:
**the predictor must be weight-compatible with plan 03's DFT sub-network**
(`tadpole` `SequentialModel` — a transformer over flattened crop-latent
tokens; verify the exact tokenization in
`_tadpole/architecture/p3d/transformer.py` and plan 03 §0 *before* writing
this module, and match it).

- **`EMAEncoder`** *(from plan 05 §2.3)*: wraps the online encoder;
  `update(momentum)` after each optimizer step; momentum schedule
  0.996 → 1.0 over training (cosine on epochs; constant 0.9999 fallback
  knob). `sg[·]` on all outputs.
- **`LatentForecaster(nn.Module)`**: transformer over latent tokens with
  the DFT tokenization. Context sequence = **all** crop latents of the
  field at `t` — flow tokens *and* geometry tokens (the plan-02
  `encode_geometry` latents; geometry conditioning comes free as context
  tokens, no cross-attention module needed — a deliberate simplification
  of plan 05's recipe, possible because our AE already tokenizes
  geometry). Conditioning: `FiLM(p, m·Δt)` — params and step offset
  through a small MLP → per-layer scale/shift, **output layers
  zero-initialized** (Tadpole convention; also makes the init condition
  testable). Positional encoding over the (channel, crop-index) grid,
  matching the DFT's scheme.
  Depth/width guidance *(from plan 05 / IWM / I-JEPA)*: start ~8 layers,
  narrow; **depth is the first knob** if conditioning seems ignored.
- **Targets:** `z̄(t+mΔt) = sg[EMA_encoder(u(t+mΔt) | g)]`, *flow tokens
  only* (geometry is static — predicting it is the slow-feature trap;
  plan 05's core insight, kept). Target normalization knob
  `target_norm: layernorm | none | channel_std` (plan 05's caveat about LN
  equalizing token energies — expose, default `layernorm`).
- **Loss:** `L_fc = mean_t ‖ẑ_t − z̄_t‖²` over target flow tokens.
  Optional robustness: `context_dropout: 0.0` — drop a random fraction of
  context tokens per step (this replaces plan 05's voxel-masking machinery
  and sidesteps the zero-fill/obstacle aliasing bug entirely).
- **Total loss and balancing** *(plan 05's recipe kept)*:
  `L = L_rec_total + λ_fc·L_fc`, λ_fc ≈ 0.25–1.0; optional
  `balance: grad_norm` implementing VA-VAE's
  `w_adapt = ‖∇L_rec‖/‖∇L_fc‖` at the encoder's last layer if hand-tuning
  is unstable.
- **Anti-collapse** *(plan 05 §2.3 kept in full)*: the recon term is the
  main guard; RankMe logged every epoch (see §E); escape hatches in order:
  (1) **frozen-target mode** (`targets: frozen` — freeze a plan-02 encoder
  as target, train predictor only; zero collapse risk, the mandated *first*
  experiment), (2) EMA joint (`targets: ema`, default for the real runs),
  (3) SIGReg (`targets: sigreg`) if EMA proves finicky.
- **Trainer integration:** extend `AutoencoderTrainer` (keep one trainer,
  one script): when `forecast.weight > 0`, require `dataset.return_pair`,
  build forecaster + EMA encoder, compute
  recon-on-`u(t)` (+ optionally recon-on-`u(t+m)`, knob) + `L_fc`;
  EMA update hooked after the optimizer step (add a minimal
  `_post_optimizer_step()` hook to `BaseTraining` — default no-op so
  `Trainer`/`PatchTrainer` stay byte-identical; existing tests guard this).
- **Artifacts:** forecaster + EMA state saved in `checkpoint.pt` for
  resume; final forecaster exported as `forecaster.pt` next to
  `encoder.pt`/`decoder.pt`. `weights.pt` remains the plain `TadpoleAE`
  state dict (the ESMDA/plan-03 contract is untouched); plan 03's DFT
  warm-start loads `forecaster.pt` explicitly.

### E. Diagnostics (`training/diagnostics.py`, new module — plan 05 R0's
best parts, demoted to diagnostics)

Logged every `diagnostics_every` epochs into `metrics.csv` columns (cheap,
on a fixed held-out mini-batch):

- **RankMe** effective rank of pooled latents:
  `exp(−Σ p_k log p_k)`, `p_k = σ_k/‖σ‖₁` over singular values.
- Per-channel latent std (dead-channel monitor, replaces free-bits).
- Latent radial spectrum + channel eigenspectrum (the "diffusability"
  diagnostics plan 05 cites).
- When forecasting is on: `L_fc` vs. the **persistence baseline**
  (`‖z(t) − z̄(t+m)‖²`) — the single most informative forecaster health
  number (below-persistence = learning dynamics; collapse shows up as both
  →0 with falling RankMe).

### F. Benchmark: `scripts/neural_surrogate/benchmark_adaptation.py`
(+ `conf/neural_surrogate/benchmark_adaptation.yaml`)

The gate for all phases. `run(cfg)` + thin `@hydra.main` wrapper; output =
one NetCDF/CSV bundle per evaluated checkpoint under the model dir. Three
legs, each independently switchable (the fine-tune leg depends on plan 03;
the other two run from day one):

- **Leg 1 — reconstruction & physics** (no plan-03 dependency): on each
  held-out root: recon RMSE (SDF-banded 0–2 / 2–5 / 5–10 cells — plan 05's
  banding, kept), energy-spectrum distance, divergence norm, wall-flux
  violation (mean |û·n̂| on first-fluid-cell faces via the mask), and a
  nearest-neighbor-retrieval baseline (best train-set snapshot by L2 —
  plan 05's anti-memorization check, kept).
- **Leg 2 — frozen-latent forecastability** (no plan-03 dependency; the
  interim gate until plan 03 lands): freeze the AE, train a *small fixed*
  latent stepper (2-layer transformer or MLP, fixed seed/steps/LR — part
  of this script, not a new training entry point) on each held-out root's
  train split; report latent-space and decoded rollout error vs. horizon
  against persistence. Cheap, and directly probes "is this latent space
  steppable?".
- **Leg 3 — adaptation** (requires plan-03 `finetune_mode: dft`): for each
  held-out root × data budget (e.g. {2, 8, all} trajectories): run the
  plan-03 fine-tune with a **fixed smoke-plus recipe** (fixed epochs, LoRA
  rank, LR; composed from `finetuning.yaml` exactly like the tests
  compose configs), then roll out on held-back trajectories. Report:
  rollout RMSE vs. horizon (per-root, never pooled), pedestrian-level
  (configurable z-index) RMSE, time-averaged-field error over the rollout
  window, and Leg-1's physics metrics on the rolled-out fields. Optional
  `warm_start: forecaster.pt` to A/B idea 3.
- **Diagnostics subcommand** (`benchmark.probes=true`): plan 05's probe
  suite — linear + MLP probes for λ_p, λ_f, mean/max building height,
  pointwise SDF-recon R², mean pedestrian-level speed — reported alongside,
  never gating.

### G. Config additions (`conf/neural_surrogate/pretrain_autoencoder.yaml`)

```yaml
architecture:
  # ... existing keys unchanged ...
  geometry_conditioning: none   # none | encoder | encoder_decoder (B1/B2);
                                # non-none requires encode_geometry + sdf_features

dataset:
  # ... existing keys unchanged ...
  root_dirs: null               # list overrides root_dir (A1)
  return_pair: false            # temporal pairs for the forecaster (A2)
  max_pair_offset: 4
  nondim: false                 # per-case U_ref scaling (A3)
  u_ref: inflow_speed
  augment: {y_flip: false, yaw_90: false}   # A4; keep off for y-periodic uDALES

aux:                            # C — every term defaults to a no-op
  denoise:   {input_sigma_rel: 0.0, latent_sigma_rel: 0.0, prob: 0.5}
  near_wall: {weight: 0.0, tau_cells: 2.0}
  spectral:  {weight: 0.0, min_fluid_frac: 0.5}
  divergence: {weight: 0.0}

forecast:                       # D — weight 0.0 = module never built
  weight: 0.0
  targets: ema                  # frozen | ema | sigreg
  ema_momentum: [0.996, 1.0]
  predictor: {layers: 8, width: default, film: true}   # DFT-compatible shape
  target_norm: layernorm        # layernorm | none | channel_std
  context_dropout: 0.0
  recon_next: false             # also reconstruct u(t+m)
  balance: none                 # none | grad_norm (VA-VAE)

diagnostics_every: 5            # E; 0 disables
```

Plus `conf/neural_surrogate/benchmark_adaptation.yaml` (§F) and dataset
split files (`conf/neural_surrogate/dataset_splits/…`, §A1).

### H. Data generation (corpus workstream, parallel to the code)

- `scripts/data_generation/procedural_layouts.py`: generate random
  building-array STLs (blocks/streets, parameterized λ_p, λ_f, height
  distributions) at the trained cell spacing, plus per-layout Hydra case
  configs consumable by the existing `generate_training_data.py` pipeline
  (multi-geometry doc §6 phase 3). Record the generator seed + morphology
  descriptors per layout in a manifest CSV (this is what the probes and
  tier definitions consume).
- Corpus targets: **v0** = 10–20 layouts (enough for the benchmark's
  held-out tiers to exist; unblocks all code phases), **v1** = O(10²)
  (the A3-phase pre-training run). One backend only at first (pylbm is the
  cheap one — see the ensemble-scaling and OOM memories before sizing
  runs: use `run.ensemble_save_on_disk` at ≥75³, workers ≤ 4–8).
- UrbanTALES ingestion continues per the multi-geometry doc; snapshot-shape
  only.

---

## Part III — Mathematics (consolidated reference)

Notation: snapshot `u ∈ R^{C×X×Y×Z}` (z-scored per channel after optional
nondim), geometry block `g` (mask + clamped SDF (+∇SDF)), params `p`,
fluid mask F, crop size h, latent grid `z ∈ R^{c_l×(h/f)³}` per folded crop
(f = 16/8/4 for S/B/L). `E_θ̄` = EMA encoder, `sg[·]` = stop-grad.

```
# Conditional AE (B1/B2); zero-init pathways ⇒ identity to plan-02 at load
z    = E_θ([u_c ; g_c])
û_c  = D_φ(z + Wg·pool(g_c)),     Wg zero-init 1×1×1 conv

# Reconstruction (near-wall-weighted masked MSE + tiny KL; NO free-bits)
L_rec = (1/Σw) Σ_{x∈F} w(x)·(û−u)²  +  β·KL,        β ≈ 1e-6
w(x)  = 1 + w_bl·max(0, (τ−d̂(x))/τ)²                # MARIO near-wall

# Physics terms
L_spec = Σ_k (log(Ê(k)+ε) − log(E(k)+ε))²            # radial shells, fluid crops
L_div  = ⟨(∇·û)²⟩_F∘ / (⟨(∇·u)²⟩_F∘ + ε)             # interior fluid cells

# Denoising (DPOT/PDE-Refiner lineage)
encode ũ = u + σ_in‖u‖η, reconstruct u;   L_lat = ‖D(z+σ_z·std(z)η | g) − u‖²

# Forecast objective (targets: flow tokens only; geometry tokens = context)
ẑ(t+mΔt)   = P_ψ({z_i(t)}_flow ∪ {z_j}_geo ; FiLM(p, mΔt)),  m ~ U{1..M}
L_fc        = (1/N) Σ_i ‖ẑ_i − sg[E_θ̄(u(t+mΔt)|g)]_i‖²
L_total     = L_rec + λ_s·L_spec + λ_d·L_div + λ_fc·L_fc     (enabled terms only)

# Diagnostics
RankMe = exp(−Σ_k p_k log p_k),  p_k = σ_k/‖σ‖₁      # SVD of pooled latents
persistence baseline: ‖z(t) − z̄(t+m)‖²               # forecaster must beat this
```

Zero-init invariants (all unit-tested): inflated encoder input slices = 0,
`Wg` = 0, forecaster FiLM/output layers = 0 ⇒ (i) converted conditional AE
≡ plan-02 AE, (ii) initial `L_fc` = persistence baseline.

---

## Part IV — Implementation phases

Each phase: **branch → implement → `pixi run -e dev pre-commit` →
`pixi run -e dev py.test` green (mind the ~28 pre-existing failures — see
the WIP-baseline memory; stash-verify before blaming new code) → PR**, docs
updated in the same PR (`docs/neural_surrogates.md` + this file's status
line). Phases gated on the §F benchmark unless noted.

### A0 — corpus v0 + dataset layer + benchmark (no model changes)
*Branch: `feat/pretrain-benchmark`.*

1. §A1–A4 dataset work (multiroot, pairs, nondim, augment) + unit tests.
2. §H procedural-layout generator + manifest; generate corpus v0 (10–20
   layouts, smoke-to-medium preset) and define split configs (tiers).
3. §E diagnostics module (needed by the benchmark's probe leg).
4. §F benchmark script, legs 1–2 + probes subcommand (leg 3 stubbed behind
   a config flag until plan 03 lands).
5. **Acceptance:** benchmark runs end-to-end on the existing plan-02
   checkpoint against corpus-v0 held-out roots; e2e smoke test composes
   the benchmark config on fixture data; single-root configs byte-identical
   to before (no-op checks).

### A1 — conditional AE + physics/denoise losses
*Branch: `feat/geometry-conditioned-ae`. Gate: benchmark legs 1–2.*

1. §B1 encoder conditioning (vendored kwargs passthrough, conditional fold,
   inflation helper) + §B2 decoder injection.
2. §C aux losses + trainer composition.
3. Runs on corpus v0: plain plan-02 baseline vs. `encoder` vs.
   `encoder_decoder`, each ± the aux bundle (denoise + near-wall as one
   bundle; spectral and divergence ablated individually — they are the
   novel terms).
4. **Acceptance:** no-op parity tests pass; conditioned model beats the
   baseline on held-out-root recon (esp. the 0–2-cell SDF band) and does
   not regress leg 2.

### A2 — forecast pre-training
*Branch: `feat/latent-forecast-pretrain`. Gate: benchmark leg 2 (leg 3
A/B once plan 03 is available).*

1. Verify DFT tokenization (plan 03 §0 / vendored `SequentialModel`);
   implement §D to match it.
2. Experiment order *(plan 05's derisking, kept)*: (i) `targets: frozen`
   on the best A1 checkpoint — validates predictor + conditioning with
   zero collapse risk; (ii) joint `targets: ema`; (iii) knobs in priority
   order: predictor depth → λ_fc/balancing → `context_dropout` →
   `target_norm`.
3. Export `forecaster.pt`; wire the benchmark's `warm_start` option.
4. **Acceptance:** forecaster beats persistence on held-out roots in
   latent space; joint training keeps recon within a few % of A1 and
   RankMe stable; leg-2 rollout improves.

### A3 — scale + deployment protocol (the payoff phase)
*Branch: `feat/pretrain-deployment`. Mostly experiments; code = glue.*

1. Corpus v1 (O(10²) layouts) generated while A1/A2 iterate.
2. Full pre-train of the winning A1/A2 recipe on corpus v1 (consider
   size S → B — diverse data rewards capacity; multi-geometry doc §5).
3. Benchmark leg 3 in full: adaptation recipe grid (frozen vs. LoRA vs.
   full × split LRs), sample-efficiency curves on both tiers, DFT
   warm-start A/B (**the** plan-03 synergy measurement).
4. Smoke-scale ESMDA twin experiment on a held-out geometry (S4).
5. **Acceptance:** success criteria S1–S4 evaluated and written up (results
   section appended to this doc or a sibling results note).

### A4 — optional ablations (only if A1–A3 leave measurable headroom)

Plan 05's remaining ideas, ordered by residual promise: same-time
Geo-JEPA / masked-crop inpainting (multi-block contiguous masks, **noise-
fill only, never zero-fill** — zero aliases with obstacles), EQ-VAE
latent-equivariance, β/free-bits sweep (revisiting §C's KL decision),
VICReg-on-pooled-latents for ESMDA regime structure. Each behind its own
config knob, each gated on the benchmark.

---

## Part V — Tests (consolidated)

Unit (CPU, tiny sizes, `pytest.importorskip` on the tadpole deps):

- **No-op parity (the critical one):** plan-02 weights loaded into a
  `geometry_conditioning: encoder_decoder` model produce byte-identical
  outputs (inflation zeros + zero `Wg`); `geometry_conditioning: none`
  module tree/state-dict keys unchanged vs. today.
- Conditional fold shapes: `(B, C, X, Y, Z)` round-trip with conditioning
  on, padded and unpadded grids, `max_internal_batchsize` chunked path ≡
  unchunked.
- Aux terms: each = exact no-op at weight 0 (total loss bit-equal to the
  plain path); divergence loss ≈ 0 on an analytic solenoidal field and > 0
  on a non-solenoidal one; spectral loss = 0 on self and > 0 on a low-pass
  copy; near-wall weights = 1 beyond τ.
- Dataset: pair indexing never crosses trajectory ends; `dt_steps` ∈ 1..M;
  y-flip flips v sign and the mask consistently (assert a flow-through
  invariant on a hand-built field); ∇SDF components rotate under yaw_90;
  nondim round-trip; multiroot sampler emits single-root batches with
  uniform root frequencies; params surface in pair mode and match
  `TransitionDataset`'s for the same file.
- Forecaster: zero-init ⇒ `L_fc` == persistence baseline (to float
  tolerance); EMA update math (momentum 1.0 ⇒ frozen; 0.0 ⇒ copy);
  target flow/geometry token split correct.
- Diagnostics: RankMe ≈ k on synthetic rank-k data.
- Trainer: `_post_optimizer_step` default no-op leaves `Trainer` /
  `PatchTrainer` behavior identical (existing tests must stay green).

e2e smoke (fixture data, 2 epochs, tiny crops, composed configs):

- `pretrain_autoencoder.yaml` + `geometry_conditioning=encoder_decoder` +
  `aux.denoise.input_sigma_rel=1e-3` + `aux.divergence.weight=0.1` — runs,
  new metrics columns appear, `weights.pt`/`encoder.pt` format unchanged.
- Same + `forecast.weight=0.25` + `dataset.return_pair=true` — runs,
  `forecaster.pt` written, resume from `checkpoint.pt` works.
- `benchmark_adaptation.py` legs 1–2 on a fixture checkpoint.

---

## Part VI — Risks and hand-off notes

Risks (mitigations inline; plan 05 Part IV's compute and citation caveats
apply unchanged):

- **DFT tokenization mismatch** would void the warm-start payoff — hence
  the A2 step-1 verification *before* writing the forecaster; if upstream
  `SequentialModel` tokenization is unsuitable, weight-compatibility wins
  over convenience (plan 03 §0 allows a custom sub-network).
- **Vendored diff** (`encoder_kwargs`/`decoder_kwargs`): 5 lines, additive,
  documented in `_tadpole/__init__.py`'s vendoring note.
- **Joint-training instability / collapse**: recon anchor + EMA schedule +
  RankMe monitoring + frozen-target first (plan 05's full recipe); SIGReg
  escape hatch.
- **Benchmark cost**: legs 1–2 are minutes-scale; leg 3 is bounded by the
  fixed smoke-plus recipe. Run leg 3 only at phase gates, legs 1–2 freely.
- **Corpus compute is the long pole** — start §H generation on day one; it
  runs unattended while A0 code lands (respect the DRAM-bandwidth ceiling
  and disk-saving memories when sizing ensembles).
- **Pooling penalty** (DrivAerNet++/MPP): expected; the A3 fine-tune
  protocol is the designed answer and S3 measures whether it pays.

Hand-off checklist for the implementing agent:

1. Read first: `CLAUDE.md`, `docs/neural_surrogates.md`,
   `docs/codebase_guide.md` §(new-parameter recipe), plan 02 + plan 03,
   multi-geometry doc §4–6 — then the plan-02 source files this plan
   extends (`tadpole_ae.py`, `autoencoder.py`, `snapshot.py`,
   `training/base.py`).
2. Conventions that will bite: no-op defaults verified by test, not by
   inspection; `def run(cfg)` + thin `@hydra.main`; branch-first, one
   phase per PR; `pixi run -e dev pre-commit` before every commit; never
   commit `training_data/`, weights, or corpus artifacts; forkserver mp
   context is load-bearing.
3. When a design detail here conflicts with what you find in the code
   (e.g. exact latent shapes, `SequentialModel` tokenization, HF weight
   file layout), **the code wins** — update this doc in the same PR.
