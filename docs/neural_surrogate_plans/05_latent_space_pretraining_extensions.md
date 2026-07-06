# Plan 05 — Extended TadpoleAE pre-training: latent spaces that understand buildings and generalize to new geometries

**Status: research report + implementation plan (2026-07-06). Builds directly on
[plan 02](02_autoencoder_pretraining.md)'s `TadpoleAE` / `AutoencoderTrainer` /
`SnapshotDataset` / `pretrain_autoencoder.yaml` — same base setup, extended
training objectives.** Complements (does not duplicate)
[docs/multi_geometry_surrogate_research.md](../multi_geometry_surrogate_research.md),
which covers geometry *inputs* (SDF > mask), multi-dataset mechanics and
procedural geometry generation; this plan covers the *training objective*:
what loss terms shape the latent space itself.

Research basis: a July-2026 literature sweep (JEPA family, JEPA-on-science,
JEPA+decoder hybrids, non-JEPA SSL for PDEs, geometry generalization in CFD
surrogates). Citations inline; verification caveats in §4.

---

## Part I — Specific recommendations (the overview)

### The two findings that frame everything

1. **Latent-prediction pre-training beats pixel reconstruction for physics —
   but our downstream task needs the decoder.** Qu et al. (Polymathic AI +
   LeCun, arXiv:2603.13227) show JEPA-style latent prediction beats VideoMAE
   by 28–51% on downstream physical-parameter tasks on The Well; V-JEPA shows
   latent prediction yields emergent physics understanding where pixel
   prediction sits at chance (arXiv:2502.11831). At the same time, decoding a
   *pure* JEPA latent recovers gist, not detail (D-JEPA, arXiv:2410.03755) —
   fatal for CFD fields. The resolution is well-precedented: **keep the VAE
   reconstruction objective and add a latent-prediction term.** BootMAE
   (arXiv:2207.07116), D-JEPA and VA-VAE (arXiv:2501.01423) all show the
   combination is non-conflicting and mutually stabilizing — the recon term
   prevents JEPA collapse, the latent term adds structure, and VA-VAE's
   ablation shows latent-alignment terms cost ~nothing in reconstruction while
   dramatically improving downstream latent modeling.
2. **Latents don't keep geometry unless you make them.** "Do Neural Operators
   Forget Geometry?" (arXiv:2605.05862) shows probing classifiers find
   operators *discard* geometric attributes from latents during ordinary
   field-loss training, and auxiliary losses that keep geometry decodable
   improve both representations and task performance. Worse, JEPA objectives
   prefer *slow* features (arXiv:2211.10831) — and in our data the slowest
   "feature" is the static geometry itself, so a naive JEPA term could spend
   its capacity re-encoding buildings while ignoring flow. The fix is to make
   geometry **conditioning, not content**: give the predictor the target
   region's geometry/SDF for free, so the latent objective is forced to be
   about *flow given geometry* — which is exactly the generalization we want.
   IWM (arXiv:2403.00504) is the formal version: an unconditioned predictor
   collapses to invariance (encoder throws the factor away, MRR 0.00 vs 0.82);
   a *deep, conditioned* predictor makes the factor equivariant/actionable.

Direct precedent that this composite works in our domain: **AeroJEPA**
(arXiv:2605.05586) — a JEPA whose context is geometry (point cloud + SDF) +
operating conditions and whose target is the flow-field latent, with geometry
via cross-attention, scalars via AdaLN, an optional decoder, and SIGReg
anti-collapse. And a 3D-voxel JEPA on LES/urban flow **does not exist yet**
(verified gap, July 2026) — this is open territory.

### Recommendations, ranked (E0–E5 = implementation phases in Part III)

| # | Recommendation | Why | Cost |
|---|---|---|---|
| **R0** | **Build the measurement harness first**: held-out-geometry eval splits (interpolative + extrapolative tiers), latent probes (linear+MLP for plan-area density λ_p, frontal density λ_f, heights, pointwise SDF-recon R²), SDF-banded near-wall recon errors, RankMe effective rank + latent-spectrum diagnostics logged during training. | Every published OOD-geometry claim degrades 2–5× on the extrapolative tier; without this harness we can't tell which objective helps. RankMe (arXiv:2210.02885) is the standard label-free collapse/quality diagnostic. | Low |
| **R1** | **Cheap loss upgrades to the plan-02 VAE, immediately**: (a) denoising — relative input noise + latent-noise robustness; (b) free-bits KL per latent channel; (c) MARIO-style near-wall loss up-weighting; (d) masked-crop inpainting phases — mask flow blocks, **never geometry channels**, reconstruct through the existing decoder. | (a) is the highest value-per-effort rollout-stability lever (DPOT ε≈5e-4, PDE-Refiner, Stachenfeld); (b) prevents dead-channel latents that break downstream steppers; (c) 6× effect on force accuracy in ablation (MARIO); (d) turns pre-training itself into "predict flow given geometry" (MAE-PDE arXiv:2403.17728; Inpainting Physics arXiv:2605.08832) using zero new modules. | Low |
| **R2** | **Geo-JEPA term (the headline extension)**: add a geometry-conditioned masked *latent* prediction loss to the VAE objective. Mask flow crops multi-block-contiguous; an EMA target encoder embeds the full field; a deep latent predictor sees context latents + the target crops' geometry/SDF (cross-attention) + global params (FiLM) and regresses the target latents. `L = L_rec + β·KL + λ_jepa·L_pred`, λ_jepa ≈ 0.25–1.0. | The latent-space counterpart of R1(d), with the demonstrated latent-objective advantage for physics (Qu et al.) and the IWM-style conditioning that makes geometry generalization the *training task*. Novel-but-derisked: every ingredient is published (BootMAE joint training, AeroJEPA geometry conditioning, I-JEPA masking recipe). | Medium |
| **R3** | **Anti-forgetting geometry decodability**: keep plan 02's `encode_geometry` reconstruction, and add a small SDF-prediction head from *flow* latents as an auxiliary loss. | Direct fix for the forgetting hypothesis (arXiv:2605.05862); trivial module. | Low |
| **R4** | **Temporal latent structure** (bridge to the plan-03 time-stepper): snapshot pairs (t, t+Δt) → (a) slowness term; (b) LE-PDE-style *normalized* latent-consistency with a small jointly-trained (throwaway or kept) latent evolver; optionally the evolver *is* a temporal Geo-JEPA predictor whose weights warm-start plan 03's DFT sub-network. | Makes latent trajectories short and predictable — precisely what the DFT stage needs (LE-PDE arXiv:2206.07681; tcKAE arXiv:2403.12335; PV-VAE arXiv:2605.02134). The warm-start reuse is free synergy with plan 03. | Medium |
| **R5** | **Optional ablation-gated extras**: EQ-VAE latent-equivariance term (y-flip / 90°-yaw+inflow-rotated latents must decode to transformed fields, arXiv:2502.09509); VICReg-with-symmetry-views on pooled latents for regime structure (Mialon et al. arXiv:2307.05432) — useful for ESMDA parameter estimation. | Cheap manifold smoothing / regime clustering; each behind a config knob, adopted only if R0 metrics improve. | Low–Med |
| **R6** | **Data lever (prerequisite for any of this to generalize)**: multi-geometry snapshot corpus — UrbanTALES (538 layouts, already partially ingested) + procedural layouts, per-geometry homogeneous batches, y-flip (v→−v) and 90°-rotation+inflow augmentation. | O(10²) diverse geometries is where published within-family generalization starts (AB-SWIFT 138, Vargiemezis ~202); no objective rescues a single-geometry corpus. Mechanics live in the multi-geometry doc (§6 phase 0/3); this plan only adds the snapshot-dataset versions. | Medium (mostly data-gen compute) |

**What I recommend *against*:** replacing the VAE with a pure JEPA (decoder is
non-negotiable for plan 03; D-JEPA/RCDM show pure-JEPA latents under-determine
fields); GAN loss as a latent-quality tool (orthogonal, stays the plan-02
optional extra); Koopman/linear-latent hard constraints (too rigid for
full 3D turbulence — at most a soft consistency term via R4); geometry
curricula (no supervised-surrogate evidence); and betting on PI-JEPA
(arXiv:2604.01349) — **withdrawn June 2026**.

**Suggested reading order for the papers**: Qu et al. 2603.13227 →
AeroJEPA 2605.05586 → IWM 2403.00504 → VA-VAE 2501.01423 →
forgetting-hypothesis 2605.05862 → LatentMIM 2407.15837 (masking-redundancy)
→ LeJEPA 2511.08544 (SIGReg, if we want to drop EMA machinery).

---

## Part II — Mathematical details

Notation: snapshot `u ∈ R^{C×X×Y×Z}` (z-scored per channel), geometry mask
`g`, SDF features `s`, params `p ∈ R^P`. Plan-02 TadpoleAE: channels folded to
batch, field tiled into crops of size `h³` (h = `encoder_crop_size`); encoder
`E_θ` → per-crop latent distribution `N(μ, σ²)` with latent grid
`z ∈ R^{c_l × h/f × h/f × h/f}` (f = compression 16/8/4 for S/B/L); decoder
`D_φ`. EMA copy `E_θ̄`.

### 2.1 Baseline (plan 02, unchanged)

```
L_VAE = (1/|F|) Σ_{i∈F} (û_i − u_i)²  +  β · KL[q(z|u) ‖ N(0, I)]
```
F = fluid cells; β ≈ 1e-6/dim (LDM convention).

### 2.2 R1 upgrades

**(a) Denoising.** Input corruption: with prob. ρ_n, encode
`ũ = u + σ_in‖u‖·η, η∼N(0,I)`, reconstruct clean `u` (σ_in ≈ 5e-4…1e-2
relative — DPOT's ablated optimum at 5e-4). Latent robustness:
`L_lat = ‖D_φ(z + σ_z·std(z)·η) − u‖²_F`, σ_z a few percent. Rationale: the
decoder must be flat around data latents because a latent stepper's error *is*
a latent perturbation; a VAE's learned σ tends to collapse, so the explicit
fixed-σ term keeps the property.

**(b) Free-bits KL** per latent channel j:
```
L_KL^FB = Σ_j max(λ_fb, E_u KL[q(z_j|u) ‖ p(z_j)]),   λ_fb ≈ 0.25–1 nat
```
prevents dead channels / few-outlier-channel latents (Kingma et al. 1606.04934).

**(c) Near-wall weight** (MARIO, 2505.14704): with normalized wall distance
`d̂ ∈ [0,1]` from the SDF, per-cell loss weight
`w(x) = 1 + w_bl·(max(0, (τ − d̂)/τ))²`, τ ≈ the first ~2 cells…canyon scale
(MARIO uses the last 2% of the distance range). Their ablation: removing it
inflates drag error ~6×.

**(d) Masked inpainting.** Sample a multi-block 3D mask M over flow channels
only (geometry/SDF channels always visible — conv encoder ⇒ inpainting-style
masking: masked voxels zero/noise-filled at input, not token-dropped):
```
L_mask = λ_m·(1/|M∩F|) Σ_{M∩F} (û−u)² + λ_v·(1/|V∩F|) Σ_{V∩F} (û−u)²,  λ_m ≫ λ_v
```
Masking must be **large contiguous blocks**: smooth, spatially-correlated
fields make scattered masks trivially interpolable (LatentMIM 2407.15837 —
raising ratio + contiguity is what forces non-trivial completion; I-JEPA's
multi-block 54.2% vs random-patch 17.6%). Start ratio 60–75%, block size ~h/2,
sweep upward — MAE-PDE used 75–90% on 1D/2D.

### 2.3 R2 — the Geo-JEPA term

Per training field (one geometry, one time): tile into N crops. Choose target
crop index set T (the masked multi-block region, ~15–25% of crops in 1–4
contiguous blocks) and context set V (the rest, minus a margin). Then:

- **Targets** (no gradient): `z̄_t = sg[E_θ̄(u_t)]` for t ∈ T — the EMA
  encoder sees the *full unmasked* field crops. Target normalization: I-JEPA
  layer-norms targets; note the caveat that LN equalizes token energies
  (arXiv:2508.02829) — expose the choice (LN | none | channel-std) in config.
- **Context**: `z_v = E_θ(u_v)` for v ∈ V (online encoder, gradients on).
- **Predictor** `P_ψ` — a transformer over latent tokens (crop-level tokens =
  flattened `c_l·(h/f)³` vectors, or finer sub-tokens):
  ```
  ẑ_t = P_ψ( {z_v, pos_v}_{v∈V} ;  {pos_t, GEO_t}_{t∈T} ;  FiLM(p) )
  ```
  with conditioning per the AeroJEPA/IWM/V-JEPA-2-AC recipes:
  - `GEO_t`: the target crops' geometry/SDF latents `E_θ̄(g_t, s_t)` (free —
    plan 02 already folds geometry channels through the same encoder) attached
    to the target mask tokens and/or a geometry-token bank consumed by
    cross-attention. Geometry is *input to the predictor*, never a prediction
    target → the encoder has no incentive to become geometry-invariant
    (IWM), and the slow-feature trap (static geometry dominating the latent
    objective) is defused because geometry information is supplied, not
    rewarded.
  - `FiLM(p)`: global params (inflow speed/direction …) through a small MLP →
    per-layer scale/shift. Zero-init the output layers (Tadpole convention).
  - **Depth over width**: IWM equivariance needed 18 predictor layers where 12
    failed; I-JEPA found narrow (384) beats wide (1024) at depth 12. Start
    ~8–12 layers, narrow, and treat depth as the first knob if conditioning
    seems ignored.
- **Loss** (L2, per target token, à la I-JEPA; L1 optional à la V-JEPA):
  ```
  L_pred = (1/|T|) Σ_{t∈T} ‖ ẑ_t − z̄_t ‖²
  ```
- **Total**: `L = L_rec + β·L_KL^FB + λ_jepa·L_pred` with λ_jepa ≈ 0.25–1.0.
  Precedents: BootMAE λ=1 with separate heads; REPA λ=0.5; DreamerV3 weights
  its encoder-shaping term 0.1 vs recon 1.0; VA-VAE 0.1 with gradient-norm
  balancing `w_adapt = ‖∇L_rec‖/‖∇L_pred‖` at the encoder's last layer —
  adopt that balancing if hand-tuning λ is unstable.

**Anti-collapse.** The reconstruction term is itself the main guard (D-JEPA:
"the diffusion loss effectively prevents the representation collapse of the
prediction loss"; Delta-JEPA: aux decode loss λ=0 → collapse, stable
λ∈[1,50]). Belt-and-braces, all cheap:
- EMA momentum schedule 0.996 → 1.0 (I-JEPA default); with a strong recon term
  a constant 0.9999 also works (D-JEPA).
- Monitor **RankMe** each epoch: `RankMe = exp(−Σ_k p_k log p_k)`,
  `p_k = σ_k/‖σ‖₁` over the singular values of a batch of pooled latents —
  a falling effective rank is the early-warning signal.
- If EMA proves finicky, two published escape hatches: **SIGReg / LeJEPA**
  (arXiv:2511.08544) — replace EMA/stop-grad with a sketched
  isotropic-Gaussian regularizer on latents (random 1-D projections pushed to
  N(0,1); this is what AeroJEPA uses, weight 0.01) — or **SALT-style
  sequential** (arXiv:2509.24317): freeze the plan-02 VAE encoder as the
  target and train only the predictor on its latents (zero collapse risk, and
  a good *first* experiment before joint training).

**Variant (temporal Geo-JEPA, feeds R4/plan 03):** identical machinery with
targets from `u(t+Δt)` instead of the same snapshot:
`ẑ_t(t+Δt) = P_ψ(z(t), GEO, FiLM(p, Δt))`. The predictor is then literally a
latent time-stepper trained without a decoder — its weights warm-start the
plan-03 DFT sub-network (both are transformers over the same latent tokens).

### 2.4 R3 — geometry decodability

Small conv head `H_geo` on flow-crop latents:
`L_geo = ‖H_geo(z^flow) − s‖²` (clamped SDF target), weight ~0.05–0.1.
Complements (not replaces) plan 02's geometry-channel reconstruction.

### 2.5 R4 — temporal latent terms

With snapshot pairs (u^k = u(t), u^{k+m} = u(t+mΔt), m = 1…M, M ≈ 4):
- **Slowness** (small weight; recon+KL prevent the classic collapse):
  `L_slow = ‖z^{k+1} − z^k‖²`.
- **Normalized latent consistency** (LE-PDE Eq.; the normalization is
  load-bearing — without it the encoder shrinks latents to cheat):
  ```
  L_consist = Σ_{m=1}^{M} ‖ g^{(m)}(z^k) − z^{k+m} ‖² / ‖ z^{k+m} ‖²
  ```
  with `g` a small latent evolver (the temporal Geo-JEPA predictor of §2.3, or
  a throwaway MLP). tcKAE's all-pairs variant (predictions of step n launched
  from different start times must agree) is the latent analogue of Poseidon's
  all2all trick — adopt if we train g for keeps.

### 2.6 R5 — optional terms

- **EQ-VAE**: for τ ∈ {y-flip with v→−v, 90° yaw + co-rotated (u,v) and
  inflow} (the valid urban subgroup — see multi-geometry doc §4.7):
  `L_EQ = ‖D_φ(τ·E_θ(u)) − τ·u‖²_F` — transforming the *latent grid* must
  decode to the transformed field. One extra decode per batch.
- **VICReg on pooled latents** with symmetry/crop views (coefficients
  λ=25 inv / μ=25 var / ν=1 cov — Mialon et al. used the defaults on PDEs;
  variance hinge `max(0, 1−√(Var+ε))` per dim, covariance = off-diagonal²).
  Apply to pooled μ, never to sampled z; KL and the variance term coexist.

---

## Part III — Implementation plan

All of it extends plan-02 code; **every new term defaults off** (repo no-op
rule) so the plain-VAE path stays byte-identical. One new config sub-block:

```yaml
# conf/neural_surrogate/pretrain_autoencoder.yaml (additions)
aux:
  denoise:   {input_sigma_rel: 0.0, latent_sigma_rel: 0.0, prob: 0.5}
  kl:        {free_bits: 0.0}
  near_wall: {weight: 0.0, tau_cells: 2.0}
  mask:      {ratio: 0.0, block_frac: 0.5, weight_visible: 0.05}
  geo_head:  {weight: 0.0}
  jepa:      {weight: 0.0, mode: same_time,      # same_time | temporal
              targets: ema,                       # ema | frozen | sigreg
              ema_momentum: [0.996, 1.0],
              predictor: {layers: 8, width: 384, geometry: cross_attn, params: film},
              target_norm: layernorm}
  temporal:  {slowness: 0.0, consistency: 0.0, horizon: 4}
  eq:        {weight: 0.0}
  vicreg:    {weight: 0.0, lambda: 25, mu: 25, nu: 1}
```

### New/changed modules

1. `datasets/snapshot.py` (plan 02) gains: multi-block 3D mask sampling
   (returns the crop-level mask), optional temporal pairing
   (`return_pair: m ∈ 1..M`, reusing `TransitionDataset`'s index logic), and
   multi-root support with per-geometry homogeneous batches (thin
   ConcatDataset + batch sampler — the snapshot twin of multi-geometry doc
   §6 phase 0; do it there once, reuse here).
2. `training/aux_losses.py` — one small class per term (Denoise, FreeBits,
   NearWall, MaskedRecon, GeoHead, EQ, VICReg, Slowness/Consistency), each
   `__call__(batch, model, outputs) → (loss, logs)`. `AutoencoderTrainer`
   composes whatever the config enables and appends each term to
   `metrics.csv`.
3. `training/jepa.py` — `EMAEncoder` wrapper (momentum schedule, `sg[]`),
   `LatentPredictor` (transformer over crop latent tokens; FiLM for params,
   cross-attention over geometry latent tokens; zero-init output), the
   `L_pred` computation, and the SIGReg alternative. The predictor is saved in
   the checkpoint (needed for the plan-03 warm start) but excluded from the
   `weights.pt` contract consumed by plan 03's encoder/decoder loading —
   `save_separate_weights` continues to emit clean `encoder.pt`/`decoder.pt`.
4. `training/diagnostics.py` — RankMe on pooled latents, latent radial
   spectrum + channel eigenspectrum (the "diffusability" diagnostics,
   arXiv:2502.14831) — logged every K epochs.
5. `scripts/neural_surrogate/evaluate_latents.py` — the R0 harness: given an
   AE `model_dir` and dataset roots, produce (a) held-out-geometry recon
   metrics, SDF-banded (0–2 / 2–5 / 5–10 cells); (b) linear + MLP probes
   (λ_p, λ_f, mean/max height, pointwise SDF R², mean pedestrian-level
   speed); (c) latent spectra + RankMe; (d) a nearest-neighbor-retrieval
   baseline (anti-memorization check). Follows the `run(cfg)` + `@hydra.main`
   shape; output = a small NetCDF/CSV bundle per model.

### Phases (each a branch/PR, gated by R0 metrics)

- **E0 — harness** (R0): `evaluate_latents.py` + diagnostics + geometry-split
  definitions in the dataset config (`split_by: geometry`, tiers). No model
  changes. *Acceptance: harness runs on a plan-02 baseline checkpoint.*
- **E1 — cheap wins** (R1 + R3): denoise, free-bits, near-wall weight, masked
  inpainting, geometry head. Ablate each against the E0 baseline on held-out
  geometry recon + probes. *These likely ship before any JEPA code.*
- **E2 — Geo-JEPA** (R2): first SALT-style (frozen plan-02 encoder as target,
  predictor-only training — no collapse risk, validates the conditioning),
  then joint EMA training. Gate: RankMe stable, recon within a few % of E1,
  probes and held-out-geometry recon improve.
- **E3 — temporal terms** (R4): paired snapshots, slowness + consistency,
  temporal Geo-JEPA; export predictor for the plan-03 warm start and A/B the
  DFT fine-tune with vs without it. *This is where the payoff for plan 03 is
  measured.*
- **E4 — optional extras** (R5): EQ-VAE, VICReg — only if E1–E3 leave headroom
  on the R0 metrics.
- **E5 — scale** (R6): rerun the winning objective on the multi-geometry
  corpus; final eval = fully-held-out geometries, both tiers, plus a plan-03
  DFT fine-tune on a held-out geometry as the end-to-end test.

### Tests

- Unit: mask sampler (contiguity, ratio, geometry channels never masked); each
  aux loss = 0 / exact no-op when disabled; EMA update math; RankMe on
  synthetic rank-k data; predictor zero-init ⇒ `L_pred` initial value equals
  the trivial-predictor baseline.
- e2e smoke: `pretrain_autoencoder.yaml` + `aux.jepa.weight=0.1` +
  `aux.mask.ratio=0.5` on fixture data, 2 epochs, tiny crops — asserts
  training runs, metrics columns appear, `weights.pt`/`encoder.pt` unchanged
  in format.

## Part IV — Risks & verification caveats

- **Redundancy trap**: smooth flow fields make small masks trivial
  (LatentMIM). If `L_pred` drops near zero within epochs while probes don't
  improve, raise mask ratio/block size before touching anything else — and
  check RankMe (T-JEPA reports JEPA *starting* collapsed on non-image data).
- **Slow-feature trap**: monitor the geometry-vs-flow split of probe scores;
  if flow probes stagnate while geometry probes saturate, the predictor's
  geometry conditioning path is too weak (deepen the predictor — IWM) or the
  latent objective is being paid by geometry content (strengthen masking of
  flow, keep geometry supplied).
- **Compute**: EMA copy ≈ +1 encoder memory; JEPA adds ~1 target-encoder
  forward + predictor per step (~1.5–2× encoder cost, decoder unchanged).
  All terms are single-GPU-friendly at plan-02 crop sizes.
- **Citation caveats**: 2026 preprints (AeroJEPA, forgetting-hypothesis,
  AB-SWIFT, Qu et al., Inpainting Physics) were verified to exist and read at
  abstract/HTML level during the research session, not all at full-PDF depth;
  PI-JEPA is withdrawn; quantitative claims above carry the papers' own
  numbers — re-verify any single number before quoting it externally.
- **Ordering vs the master plan**: this plan extends plan 02 and feeds plan
  03; it does not block plans 01/04 (LoRA/BaLoRA). Sensible insertion: E0–E1
  right after plan 02 lands; E2+ can proceed in parallel with plan 03's
  baseline DFT so the A/B (vanilla AE vs Geo-JEPA AE as DFT init) is clean.
