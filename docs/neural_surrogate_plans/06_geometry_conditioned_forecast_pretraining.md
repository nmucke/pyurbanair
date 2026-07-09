# Plan 06 — Geometry-conditioned forecast pre-training: a critical revision of plan 05

**Status: proposal (2026-07-09). Alternative to — not an extension of —
[plan 05](05_latent_space_pretraining_extensions.md), which is kept unchanged
for comparison.** Builds on the same plan-02 code
(`TadpoleAE` / `AutoencoderTrainer` / `SnapshotDataset`) and the same
literature base; large parts of plan 05 survive here (its R0 diagnostics, R1
denoise/near-wall terms, the EMA anti-collapse machinery, the R6 corpus, the
plan-03 warm-start idea). What changes is **what sits at the center**: the
architecture is made geometry-conditioned at the root, the pre-training
objective becomes the downstream task (latent forecasting), the gate metric
becomes a standardized fine-tune-and-roll-out benchmark, and the data corpus
moves from last to first.

---

## Part I — Critique of plan 05

Plan 05 is a strong literature synthesis, and its risk register is honest.
The problems are structural — in what it optimizes, in what order, and in one
architectural blind spot it inherits without questioning.

### C-1. The gate metric is a proxy, and the plan gates everything on it

Plan 05's phases E1–E4 are accepted or rejected on the R0 harness: linear/MLP
probes for morphology descriptors (λ_p, λ_f, heights), SDF-recon R², RankMe,
held-out-geometry *reconstruction*. None of these is the product. The product
is: **a fine-tuned time-stepper that rolls out stably on a new city and
supports ESMDA**. The causal chain "better probes → better latents → better
DFT fine-tune → better rollout → better assimilation" is long, and the
literature plan 05 itself cites gives no evidence for the middle links on
autoregressive 3D turbulence. The foundation-model field settled this
question years ago: you evaluate a pre-training recipe **by fine-tuning it on
the downstream task** (every result in Poseidon, MPP, DPOT is a fine-tune
curve, not a probe). Probes are cheap *diagnostics* for debugging a failed
run — they must not be *gates*.

**Consequence:** plan 05 could spend months shipping objectives that improve
every R0 number and leave rollout RMSE on a held-out city unchanged — and its
own process would call that success until E5.

### C-2. The data lever is acknowledged as the prerequisite, then scheduled last

R6 says it outright: *"no objective rescues a single-geometry corpus"* — and
then the multi-geometry corpus is phase **E5**, after all objective work.
Every intermediate ablation (E1–E4) would run on essentially one geometry,
where "held-out geometry" tiers don't exist and cross-geometry transfer — the
thing every objective is being designed for — is unmeasurable. The
cross-geometry evidence base (multi-geometry doc §3–§4) is unambiguous that
data diversity dominates objective choice: one-city training fails transfer
regardless of recipe (the Niigata→Montreal FNO), and O(10²) layouts is where
generalization starts. The corpus is not a scaling step; it is the
experimental substrate without which the plan's own questions can't be asked.

### C-3. The architecture is geometry-blind at the core, and plan 05 patches symptoms

This is the deepest issue. The vendored Tadpole autoencoder **folds channels
into the batch and encodes each 64³ crop as an independent single-channel
image** (`tadpole_ae.py`: `(B C U V W) 1 Xc Yc Zc`). Concretely:

- The **encoder** never sees geometry when embedding a flow crop: a `u`-crop
  in a street canyon and the same `u`-values in open terrain are encoded by a
  network that cannot tell them apart except through the flow values
  themselves. Geometry rides along as *separate* folded crops whose latents
  sit next to the flow latents — visible to the plan-03 DFT's attention, but
  not to the flow encoder.
- The **decoder** reconstructs flow crops without being told where the walls
  are. Every near-wall gradient, every recirculation edge must be stored *in
  the flow latent*, because the decoder gets nothing else. The latent is
  forced to spend capacity re-encoding geometry-determined structure —
  precisely the "slow feature trap" plan 05 worries about — and the
  `state * geometry` output masking only zeroes solid cells; it does not give
  the decoder the wall distance that shapes the fluid cells next to them.
- Channels are encoded independently, so `u`/`v`/`w` coupling (continuity,
  the pressure–velocity link) exists nowhere in the AE — only in the
  downstream predictor.

Plan 05 responds to the *symptoms* of this design: an anti-forgetting SDF
head (R3) to keep geometry decodable from latents that were never given it
cheaply; IWM-style predictor conditioning (R2) so at least the *predictor*
sees geometry; monitoring for the slow-feature trap. All of that machinery
treats a problem that should be removed at the source: **condition the
encoder and the decoder on geometry directly** (Part II, idea 2). Then the
latent no longer needs to store geometry (it is supplied at decode time, so
it cannot be "forgotten"), latent capacity goes to flow, and "flow given
geometry" is the AE's definition rather than an auxiliary loss's aspiration.

Plan 05 also inherits the fold's motivation without owning its cost. Tadpole
folds channels to be channel-count-agnostic across PDE *families*. We are
building an urban-flow *family* model with a fixed modality (u, v, w, later
maybe θ). We pay the foundation-model tax (no cross-channel physics, no
geometry conditioning, N× encoder passes per field) for a generality we do
not need.

### C-4. Same-time JEPA is a detour when the data is trajectories

Plan 05's headline (R2) is *same-snapshot* masked latent prediction, with the
temporal variant as a sub-bullet and R4 as a later phase. But all our data
comes from **trajectories** — temporal pairs are free — and the temporal
predictor is *literally the downstream model* (plan 03's DFT sub-network is a
transformer over the same latent tokens; plan 05 itself notes the warm-start
synergy). Spatial-completion pre-training is a proxy task whose value for
dynamics is speculative and which brings the largest pile of new machinery in
the plan: multi-block 3D mask samplers, ratio/contiguity sweeps, the
redundancy-trap failure mode, target-normalization choices. The temporal
objective needs none of that (context = latents at t, target = latents at
t+Δt — no masking required for the task to be non-trivial) and every
gradient it takes is a gradient toward the thing we ship. Pre-train the
forecaster; keep same-time masking as an optional ablation.

There is also a concrete masking bug lying in wait: plan 05 masks by
zero-filling voxels at the conv encoder's input — but in our masked inputs
**zero already means "solid"** (`state * geometry`). Zero-filled mask blocks
are indistinguishable from buildings, and the single-channel fold makes the
standard fix (a mask-indicator channel) impossible. If masking is ever
implemented, it must noise-fill or use a learned fill value.

### C-5. The latent-regularization story is internally inconsistent

R1(b) adds free-bits with λ_fb ≈ 0.25–1 nat *per channel* while keeping the
LDM convention β ≈ 1e-6. Free-bits changes the KL's *shape*, but the KL still
enters the total loss multiplied by β — at 1e-6 the floor is numerically
invisible and the recommendation is a no-op. The two knobs come from two
different philosophies (proper VAE vs. LDM-style barely-regularized AE) and
plan 05 adopts both without choosing. Choose: our downstream is a
*deterministic latent stepper*, not generative sampling, so the LDM stance is
right — tiny/zero KL, with latent smoothness enforced *explicitly* by the
latent-noise robustness term (plan 05's own R1(a), which is the mechanism
that actually matters for a stepper). Drop free-bits, or raise β and own the
consequences; don't ship both defaults.

### C-6. The physics is missing

For an urban-atmospheric-flow model, plan 05's loss menu is strikingly
generic — near-wall weighting (R1c) is the only term that knows this is
fluid dynamics. Three cheap, well-precedented levers are absent:

- **Spectral loss.** The known failure mode of (V)AEs on turbulence is
  high-wavenumber smoothing — plan 02 even reserves a GAN for it. An
  energy-spectrum / log-spectral-distance term on reconstructions is the
  cheap, stable alternative (standard in turbulence SR/closure work; the same
  diagnosis underlies PDE-Refiner's high-frequency analysis) and should be
  tried long before adversarial training.
- **Divergence penalty.** The data is incompressible LES; `‖∇·û‖²` over
  fluid cells is nearly free to compute on the assembled reconstruction and
  couples u/v/w through the loss even where the architecture doesn't.
- **Physical-validity metrics in the harness.** R0 measures recon error and
  probes but never asks: is the decoded/rolled-out field *plausible flow*?
  Divergence norms, wall-normal flux at building faces, energy-spectrum
  match, and — because that is what the application consumes — pedestrian-
  level (z ≈ 2 m) statistics and time-averaged fields (ESMDA's observation
  operators act on these) belong in the benchmark.

Meanwhile R5 spends its optionality budget on VICReg and EQ-VAE. The valid
urban symmetry group (y-flip, 90° yaw with co-rotated velocity and inflow) is
better spent as plain **data augmentation** (multi-geometry doc §4.7 — an
order-of-magnitude sample-efficiency lever) than as latent-equivariance loss
terms: augmentation attacks generalization directly, adds zero modules, and
composes with everything else.

### C-7. There is no deployment story

The stated end goal is *pre-train → fine-tune → predict real urban flow (→
assimilate)*. Plan 05 ends at "rerun the winning objective on the corpus"
(E5) with a single DFT fine-tune as the end-to-end test. Missing entirely:
the **per-city adaptation protocol** (what gets LoRA'd vs. fully trained vs.
frozen; split LRs à la Poseidon; how much target-city data), the
**sample-efficiency curve** (rollout error vs. fine-tune data budget on a
held-out city — the number that justifies pre-training at all), and the
**ESMDA twin experiment** on a held-out geometry. Those are the deliverables
the whole program is for, and they need to be designed early because they
*are* the benchmark (C-1).

---

## Part II — Central ideas of the new plan

Five ideas, each answering a critique above. Everything else in plan 05 that
doesn't conflict (denoise, near-wall weighting, EMA machinery, RankMe-style
diagnostics, the R6 corpus mechanics, the reading list) is retained.

### Idea 1 — The benchmark is the product: gate on adaptation, not probes *(answers C-1, C-7)*

Build one standardized **adaptation benchmark** first, and gate every
subsequent phase on it:

> Given a pre-trained checkpoint: run the plan-03 DFT fine-tune with a
> *fixed, small* recipe (fixed steps, fixed LoRA rank, fixed data budget) on
> each held-out geometry; report (a) rollout RMSE vs. horizon (per-area,
> never pooled), (b) pedestrian-level and time-averaged-field errors, (c)
> physical validity (divergence, wall flux, energy spectra), (d) the same at
> 2–3 fine-tune data budgets → the sample-efficiency curve.

A pre-training change ships iff it improves this benchmark. Latent probes,
RankMe and spectra remain as cheap *diagnostics* logged during training —
useful for explaining failures, never for declaring success. This inverts
plan 05's E0: same harness-first instinct, pointed at the right quantity.
The mini fine-tune is affordable by construction (it is the smoke-shaped
version of plan 03's stage, reusing its config), and it exercises the full
path pretrain → finetune → rollout that the probes skip.

### Idea 2 — Geometry into the autoencoder itself: `z = E(u | g)`, `û = D(z | g)` *(answers C-3)*

Make the AE a **conditional** autoencoder. Geometry (mask + clamped SDF)
stops being "extra channels that happen to ride through the same encoder"
and becomes conditioning supplied to *both* ends:

- **Encoder:** each folded flow crop is encoded together with its
  co-located geometry channels — input conv widens from 1 to
  `1 + n_geom` channels. The **new input weights are zero-initialized**, so
  a freshly converted model is *exactly* the plan-02 model (repo no-op rule
  satisfied at the weight level) and HF pre-trained weights load unchanged
  into the channel-0 slice.
- **Decoder:** the decoder receives the geometry crop alongside the latent —
  concretely, the geometry crop downsampled to the latent grid (or the
  geometry crop's own encoder latent) concatenated to the flow latent, again
  through a zero-initialized projection. The decoder can now place walls,
  boundary layers and canyon structure without the latent having to store
  them.

Why this beats plan 05's approach: the latent objective no longer has to
*fight* the architecture. "Geometry is conditioning, not content" (plan 05's
own correct principle, applied there only to the JEPA predictor) now holds
for the entire model. The anti-forgetting head (R3) becomes unnecessary —
nothing needs to be remembered that is always supplied. The slow-feature
trap loses its fuel — static geometry earns no reconstruction reward through
the latent. And latent capacity is spent exclusively on flow, which is what
the plan-03 stepper must predict. The folded, channel-agnostic structure is
*kept* (HF compatibility, memory behavior, minimal diff to vendored code);
a full multi-channel joint encoder (u,v,w in one crop) is a fallback
ablation, not the default.

Geometry latents are still produced (the geometry crops still pass through
the encoder as today) so the plan-03 DFT keeps its geometry tokens.

### Idea 3 — Pre-train the forecaster, not just the autoencoder *(answers C-4)*

The second pre-training objective is **geometry-conditioned latent
forecasting** — plan 05's "temporal Geo-JEPA variant" promoted from footnote
to headline, replacing same-time masked prediction:

- A latent predictor `P_ψ` (transformer over crop-latent tokens, geometry
  tokens via cross-attention, params + Δt via FiLM — the *same architecture
  contract* as plan 03's DFT sub-network) predicts the latents of
  `u(t+mΔt)` from the latents of `u(t)`, `m ∈ {1..M}`.
- Targets come from an **EMA encoder** (plan 05's anti-collapse recipe
  carries over: momentum schedule, RankMe monitoring, recon loss as the main
  guard, SIGReg/frozen-target escape hatches).
- Optional cheap robustness: drop a random subset of *context* tokens
  (crop-level dropout) — robustness to imperfect context without any
  voxel-masking machinery, sidestepping the zero-fill/obstacle aliasing bug
  entirely (C-4).
- `L = L_rec + λ_fc · L_forecast`, with the gradient-norm balancing plan 05
  already selected (VA-VAE-style) if hand-tuned λ is unstable.

The payoff is structural: **pre-training now contains the downstream
computation.** The exported `P_ψ` warm-starts plan 03's DFT (same tokens,
same conditioning pathways), so "how much does pre-training help?" becomes a
clean A/B: DFT from scratch vs. DFT from `P_ψ`, measured on the Idea-1
benchmark. Multi-Δt conditioning (FiLM on m·Δt) gives Poseidon-style
all2all pairs for free from the trajectories we already have and prepares
for corpora with heterogeneous save cadences.

### Idea 4 — Data and dimensional analysis first *(answers C-2)*

Phase 0 is the corpus, not the harness code:

- **O(10²) geometries** before any objective work: procedural building
  arrays through the existing `generate_training_data.py` pipeline +
  UrbanTALES ingestion, per multi-geometry doc §6 phases 0/3 (multi-root
  datasets, per-geometry homogeneous batches — built once there, reused
  here). Interpolative/extrapolative held-out tiers defined at corpus-build
  time.
- **Nondimensionalization fixed as a data-schema decision now**: velocities
  scaled by per-case U_ref, lengths by building height H, and the convective
  time step per saved frame checked/resampled to a common cadence
  (multi-geometry doc §4.3). Retrofitting this after checkpoints exist means
  re-pre-training; deciding it now costs a config field. Reynolds similarity
  also *shrinks the parameter space* the forecaster must cover — different
  inflow speeds over the same layout collapse toward one sample.
- **Symmetry augmentation as data** (y-flip with v→−v; 90° yaw + co-rotated
  (u,v) and inflow where the backend's BCs allow it): replaces plan 05's
  EQ-VAE/VICReg loss terms with the mechanism that has the strongest
  evidence (Lie-point-symmetry augmentation, ICML 2022) and zero new
  modules.

### Idea 5 — Physics-aware losses over generic SSL extras *(answers C-5, C-6)*

The auxiliary-loss budget goes to terms that know this is incompressible
urban flow, each individually cheap and default-off:

| Term | What | Why |
|---|---|---|
| Spectral | log-energy-spectrum distance on reconstructed crops | the actual VAE-on-turbulence failure mode; the cheap alternative to plan 02's GAN option |
| Divergence | `‖∇·û‖²` on fluid cells of the assembled recon | incompressibility; couples u/v/w through the loss despite the fold |
| Near-wall | SDF-banded up-weighting *(kept from plan 05 R1c)* | canyon/pedestrian accuracy is the application |
| Denoise | input + latent noise robustness *(kept from plan 05 R1a)* | the highest-value rollout-stability lever; the latent-noise term is what makes a latent stepper viable |

KL policy (resolving C-5): stay LDM-style — β tiny, `latent_type: sample`,
**no free-bits** — and rely on the explicit latent-noise term for the
smoothness a stepper needs. Dead-channel monitoring moves to diagnostics
(per-channel latent variance in `metrics.csv`) instead of a loss term.

Dropped from plan 05's menu (with reasons): free-bits (C-5), same-time
masked inpainting and its mask-sampler machinery (C-4; revisit as G4
ablation), the R3 SDF head (obsoleted by Idea 2), EQ-VAE and VICReg
(replaced by augmentation, Idea 4).

---

## Part III — Mathematical sketch

Notation as in plan 05 §2. Geometry block `g` = mask + clamped SDF
(+ ∇SDF), per crop.

**Conditional AE (Idea 2).**
```
z   = E_θ([u_c ; g_c])            # folded crop u_c with its geometry channels; new input weights zero-init
û_c = D_φ([z ; proj(g_c↓)])       # g_c↓ = geometry crop pooled to the latent grid; proj zero-init
L_rec = (1/|F|) Σ_F w(x)·(û − u)²  +  β·KL,     β ≈ 1e-6 (LDM stance, no free-bits)
```
`w(x)` = near-wall weight (plan 05 §2.2c unchanged). Zero-init of both new
pathways ⇒ at load time the model is bit-identical to plan-02 behavior;
training then opens the pathways.

**Physics terms (Idea 5).**
```
L_spec = Σ_k | log Ê(k) − log E(k) |²          # radially-binned energy spectrum per crop/field
L_div  = (1/|F∘|) Σ_{F∘} (∇·û)²                # central differences, interior fluid cells only
```

**Forecast pre-training (Idea 3).** For a trajectory pair (t, t+mΔt):
```
ẑ(t+mΔt) = P_ψ( {z_i(t), pos_i} ; {z^geo_j} via cross-attn ; FiLM(p, mΔt) )
L_forecast = (1/N) Σ_i ‖ ẑ_i(t+mΔt) − sg[E_θ̄([u(t+mΔt) ; g])]_i ‖²
L = L_rec (+ λ_s·L_spec + λ_d·L_div) + λ_fc·L_forecast
```
`E_θ̄` = EMA encoder (momentum 0.996→1.0; recon term is the primary
anti-collapse guard; RankMe logged per epoch as diagnostic; frozen-encoder
SALT-style variant = first experiment, zero collapse risk). `P_ψ`
zero-initialized output ⇒ initial `L_forecast` equals the
persistence-forecast baseline — the unit-testable init condition, and the
identity-init story plan 03 already requires.

---

## Part IV — Implementation phases

Everything default-off / no-op (repo rule); each phase a branch/PR gated on
the Part-II Idea-1 benchmark (except G0, which *builds* it).

- **G0 — corpus + benchmark** *(Ideas 1, 4)*:
  (a) procedural-layout generation + UrbanTALES snapshot ingestion via the
  multi-geometry doc's phase-0 mechanics (multi-root `SnapshotDataset`,
  per-geometry batch sampler — implemented once, shared with
  `TransitionDataset`); held-out tiers defined in dataset config.
  (b) nondimensionalization convention in the data schema (per-case U_ref,
  H; config fields + dataset-side scaling, default off).
  (c) `scripts/neural_surrogate/benchmark_adaptation.py` — the standardized
  mini-DFT fine-tune + rollout + physical-validity + budget-curve harness
  (`run(cfg)` + `@hydra.main`; reuses plan-03's fine-tune config in smoke
  shape). Latent probes/RankMe included as a cheap diagnostics sub-command.
  *Acceptance: benchmark runs end-to-end on a plan-02 baseline checkpoint
  and one held-out procedural geometry.*
- **G1 — conditional AE + physics losses** *(Ideas 2, 5)*: encoder/decoder
  geometry conditioning behind `architecture.geometry_conditioning:
  none | encoder | encoder_decoder` (zero-init; `none` = today's model);
  `aux.spectral / aux.divergence / aux.near_wall / aux.denoise` in
  `training/aux_losses.py` (plan 05's module layout, smaller menu);
  augmentation flags in `SnapshotDataset`. Ablate on the benchmark.
- **G2 — forecast pre-training** *(Idea 3)*: `training/forecaster.py`
  (`EMAEncoder`, `LatentForecaster` = plan-03-DFT-shaped transformer,
  Δt-FiLM, context-token dropout); first frozen-target (SALT-style), then
  joint EMA. Checkpoint carries the forecaster; `encoder.pt`/`decoder.pt`
  contract untouched. *Acceptance: forecaster beats persistence in latent
  space on held-out geometries; benchmark improves with warm start.*
- **G3 — deployment protocol** *(Idea 1 payoff)*: the per-city adaptation
  recipe (frozen/LoRA/full × split LRs), the full sample-efficiency study on
  both held-out tiers, and an ESMDA twin experiment on a held-out geometry
  (smoke-shaped first). This is the end-to-end result the program is for.
- **G4 — optional ablations**: plan 05's same-time Geo-JEPA (with noise-fill
  masking, never zero-fill), masked inpainting, free-bits/β sweep, EQ-VAE —
  only if G1–G3 leave measurable headroom on the benchmark.

### Tests

- Zero-init no-op parity: `geometry_conditioning: encoder_decoder` model
  with plan-02 weights loaded produces byte-identical outputs to the plan-02
  model (the conditioning pathways are exactly zero).
- Divergence loss = 0 on an analytic solenoidal field; spectral loss = 0 on
  self-comparison and > 0 on a low-pass-filtered copy.
- Forecaster-at-init = persistence baseline (zero-init output layer).
- Mask/augmentation correctness: y-flip flips v sign, geometry co-flipped;
  nondim round-trip.
- e2e smoke: G0 benchmark on fixture data; G2 joint training 2 epochs, tiny
  crops, artifacts + metrics columns asserted; `weights.pt`/`encoder.pt`
  format unchanged.

---

## Part V — Risks

- **Vendored-code drift.** Idea 2 modifies the vendored `_tadpole`
  encoder/decoder I/O. Mitigation: zero-init additive pathways only, behind
  the `geometry_conditioning` knob; `none` path stays byte-identical
  (tested); we already own the vendored copy, so upstream divergence is a
  documentation task, not a dependency conflict.
- **HF pre-trained weights lose value under conditioning.** Mitigated by the
  zero-init inflation (weights load unchanged); and plan 02 already treats
  HF-vs-scratch as an open experiment — on urban LES the domain gap makes
  scratch the likely winner anyway. The benchmark decides.
- **Joint AE+forecaster instability.** The recon anchor + EMA schedule is
  the well-trodden mitigation (BootMAE/D-JEPA precedents, per plan 05);
  the frozen-target first step isolates the risk before joint training.
- **Corpus compute.** O(10²) procedural layouts ≈ O(10²) ensemble runs at
  smoke-to-medium preset on DRAM-bandwidth-capped hardware (see the ensemble
  scaling memory) — the long pole, which is exactly why it is G0. Layouts
  can be generated incrementally; the benchmark only needs the held-out
  tiers early.
- **Benchmark cost per ablation.** A mini-DFT fine-tune per candidate is
  heavier than a probe. Kept affordable by fixing a smoke-shaped recipe
  (small crops, short schedule, 2–3 geometries per tier); diagnostics still
  catch gross failures cheaply between benchmark runs.
- **Pooling penalty** (DrivAerNet++/MPP): expect joint pre-training to trail
  specialists per city; the G3 protocol (short per-city fine-tune) is the
  designed answer, and the sample-efficiency curve measures whether the
  trade is worth it.
