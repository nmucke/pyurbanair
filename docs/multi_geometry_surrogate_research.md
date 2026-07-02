# Multi-geometry neural surrogates: research notes & upgrade plan

*Research/design record, 2026-07-02. Focus: the P3D architecture
([libs/neural-surrogates/src/neural_surrogates/architectures/p3d.py](../libs/neural-surrogates/src/neural_surrogates/architectures/p3d.py))
and the training stack ([docs/neural_surrogates.md](neural_surrogates.md)).
Goal: train one surrogate on data from **multiple geometries / urban areas**
and have it perform well on all of them — ideally generalizing to unseen
layouts. This is a survey + plan, not implemented behavior; verify against
the code before relying on details.*

---

## 1. Where the current setup stands

Today the surrogate trains on one `training_data/<backend>_<case>/` tree at a
time (currently `pypalm_barcelona`), i.e. **one geometry per trained model**.
Geometry enters the model as a single binary fluid/obstacle channel, the
output is hard-masked to zero in obstacles, params are broadcast as constant
input channels, the model predicts a z-scored residual, and training uses the
pushforward curriculum with `grad_unroll_steps=2`. That recipe is in line with
best practice for a *single* geometry — normalization + residual prediction is
exactly what fixed the UPT collapse here, and hard masking + masked loss
(`trainer.mask_loss: true`) is the standard, well-supported default.

The user-facing claim "geometry is already an input, so multi-geometry should
technically work" is only true at the `forward()` signature level. Two things
stand in the way:

1. **The pipeline hard-codes a single geometry** in several places (§2) — a
   naively pooled dataset would train against the *wrong mask* silently.
2. **A binary mask is a weak geometry encoding** for generalization: a fluid
   cell 5 m from a tower and one 500 m from anything look identical in that
   channel. The literature is unusually consistent that this representation
   underperforms distance-based encodings across geometries (§3).

Notably, the **P3D paper itself does no geometry conditioning at all**
(Holzschuh et al., arXiv:2509.10186 — all 14 jointly-trained PDE cases are
periodic, obstacle-free domains; the only non-periodic case is channel flow,
handled by the grid setup). Our mask channel is already an extension of the
upstream model, and multi-geometry training is untested territory for this
architecture. One architectural sympathy worth noting: P3D mixes local
(windowed-attention / conv) computation with a global-context mechanism, so a
geometry channel that carries **global** shape information per voxel (an SDF)
composes much better with it than a nearly-information-free local binary mask.

## 2. Concrete blockers in the current code

These are the places that assume one geometry. All were verified against the
code on 2026-07-02.

| # | Location | Assumption | Failure mode with mixed geometries |
|---|---|---|---|
| B1 | `TransitionDataset.__init__` ([transition.py:146](../libs/neural-surrogates/src/neural_surrogates/datasets/transition.py#L146)) | Geometry is loaded **once from the first trajectory** and returned for every item | All samples from a second geometry get the first geometry's mask |
| B2 | `transition_collate` ([transition.py:21](../libs/neural-surrogates/src/neural_surrogates/datasets/transition.py#L21)) | Ships `batch[0]`'s geometry as `(1, *grid)` for the whole batch | Mixed-geometry batches silently use one item's mask for all |
| B3 | `BaseTraining._prepare_batch` ([training/base.py:251](../libs/neural-surrogates/src/neural_surrogates/training/base.py#L251)) | Caches the geometry mask from the **first batch of the run** (`self._geometry is None`) and reuses it for every subsequent batch, including the fluid mask used by `mask_loss` | Even a fixed dataset/collate would be defeated: the whole run trains against batch #1's mask |
| B4 | `_compute_normalization_stats` ([train_neural_surrogate.py:26](../../scripts/neural_surrogate/train_neural_surrogate.py#L26)) | One fluid mask, one train split, single global mean/std buffers | Stats biased toward whichever dataset dominates; per-dataset flow-scale differences absorbed into a single z-score |
| B5 | `TransitionDataset` takes a single `root_dir`; enforces one `param_names` set across samples | One dataset tree per training run | No mechanism to mix datasets, or datasets with different param sets |
| B6 | `P3D.compile_dynamic = False` ([p3d.py:166](../libs/neural-surrogates/src/neural_surrogates/architectures/p3d.py#L166)) + `dataloader.drop_last` rationale | torch.compile specializes on one fixed (padded) grid shape and batch size | Each distinct grid shape triggers a full recompile; fine for a handful of shapes, pathological if shapes alternate per step *(they won't if batches are per-geometry — see R2)* |
| B7 | `NeuralSurrogateForwardModel._check_domain` ([forward_model.py](../libs/neural-surrogates/src/neural_surrogates/forward_model.py)) | Requested grid must equal the **single** trained domain (exception: `domain_flexible` DD models, spacing-only check) | A multi-geometry checkpoint has no notion of "set of trained domains" |
| B8 | Grid divisibility: P3D pads non-periodic axes to a multiple of 16; **periodic axes must already divide 16** | Every geometry's grid must satisfy this | New areas need grid sizes chosen accordingly (esp. the y-periodic uDALES convention) |

None of these are deep — they are all "one geometry" baked into bookkeeping,
not into the math. The model contract `forward(state, params, geometry)`
already carries geometry per batch.

## 3. Literature: geometry representation (what to feed the network)

*(Full citations in §8.)*

**Binary mask → signed distance function (SDF) is the single
best-documented input change for cross-geometry generalization**, replicated
from Guo et al. (KDD 2016, the canonical CNN-aerodynamics-with-SDF paper)
through GINO (NeurIPS 2023) to dedicated 2025 studies (Rabeh et al.,
arXiv:2503.17289: SDF inputs improved boundary-layer accuracy up to 32% over
no-SDF operator baselines; a 2025 J. Comput. Phys. paper is devoted entirely
to SDF domain encoding for shape-family surrogates). The intuition matches
P3D's structure: SDF gives every voxel — and every attention window — global
shape awareness that a binary mask cannot.

Practical grid-encoding menu, evidence-ranked:

1. **Clamped/normalized SDF channel** (strong evidence). Keep the binary mask
   *as well* — it remains the exact fluid/solid partition needed for hard
   masking and the masked loss. Clamping the SDF at a few characteristic
   lengths (building height) bounds the input distribution; this is common
   practice, though no clean truncated-vs-full ablation exists.
2. **∇SDF (normals) as 3 extra channels** (moderate — widely used, no clean
   isolated-benefit ablation on voxel grids).
3. **Multi-directional distance features (MDDF)** — per-voxel distance to
   buildings along several directions. FLUME-FNO (arXiv:2503.19708) used this
   for 3D urban wind + temperature and generalized to unseen urban
   configurations from only **23** simulations. The one on-domain 3D result;
   worth trying if plain SDF saturates.
4. **Wall-proximity fields** (smooth 1-at-wall decaying-to-0 transforms of
   SDF), used by Chen & Thuerey (arXiv:2109.02183) — helpful when output
   fields have very different near-wall gradient scales.

**Boundary handling:** hard masking of solid cells every rollout step +
fluid-masked loss (both already implemented) is the strong practical default.
Partial convolutions / exact-BC operator constructions (BOON, BENO) exist but
have no verified evidence for immersed building geometries — not recommended
as a first move.

**Geometry-encoder architectures** (GINO, DoMINO, Transolver/++, AB-UPT): the
car-aero literature converges on a separate multi-scale geometry encoder
producing latents that condition a field predictor, trained across
O(10²–10³) shapes. All of that evidence is for **steady/statistical**
targets, not autoregressive rollout — integration risk is real. Treat as the
fallback if grid-channel encodings plateau (§6, phase 4). The repo's existing
UPT implementation is the natural in-house baseline from this family.

**Urban-specific state of the art:** multi-geometry generalization has been
demonstrated at pedestrian level in 2D (MLP-mixer on 163 synthetic
geometries × 8 wind directions, Cambridge EDS 2025; Pareto-CNNs on 2,384
geometries, 2024; UrbanTALES-based hierarchical U-Net, arXiv:2510.27101) and
in 3D for steady fields (AB-SWIFT, arXiv:2603.25635, trained on *randomized*
urban geometries). A cautionary direct precedent: an FNO trained on **one**
city (Niigata) failed to transfer to Montreal (arXiv:2501.05499) — patch
training alone did not rescue it. **3D autoregressive multi-city surrogates
are essentially open territory** — nobody has published exactly what we're
attempting, but everything adjacent says: many (cheap, procedurally
generated) layouts + distance-based geometry encoding.

## 4. Literature: training across heterogeneous datasets

The PDE-foundation-model literature (MPP, DPOT, Poseidon, PDE-Transformer,
Aurora) has settled on a consistent set of mechanisms. What breaks with naive
pooling is well documented: DrivAerNet++ reports per-category models at
R² ≈ 0.82–0.9 dropping to ≈ 0.6 when the same models are trained on the
pooled diverse dataset; MPP concedes joint training "does hurt performance on
individual tasks" relative to per-task optima. The mitigations:

1. **Balanced dataset sampling — never size-proportional.** MPP samples one
   *system per micro-batch uniformly* (homogeneous micro-batches + gradient
   accumulation); DPOT uses weighted sampling `p_k ∝ w_k/|D_k|`. Homogeneous
   per-dataset batches are also exactly what our pipeline wants for other
   reasons (B2, B3, B6).
2. **Per-dataset loss normalization.** Relative/normalized MSE
   (MSE / ‖target‖²) so high-wind or high-variance areas don't dominate the
   gradient (MPP's NMSE, Poseidon's relative loss).
3. **Normalization strategy** — three published options:
   - *Global per-variable stats anchored to a reference dataset* (Aurora):
     simplest, keeps absolute scale visible; our current single
     mean/std is this, minus the multi-dataset streaming.
   - *Per-sample reversible instance norm* (RevIN; adopted by MPP): each
     input window z-scored by its own stats, denormalized at the output.
     Auto-nondimensionalizes by the local velocity scale; removes scale
     information from the network input (fine for residual next-step
     prediction, but keep the stats around).
   - *Physics nondimensionalization*: scale velocity by inflow U_ref, length
     by building height H, time by H/U_ref — Reynolds similarity makes
     different-inflow samples of similar geometry *the same* sample.
     Repeatedly asserted/used in ML-CFD, **no controlled large-scale
     ablation** exists; the principle is sound and it matches our own
     UPT-collapse experience (raw scales sabotage training).
     Practical synthesis: nondimensionalize by known physical scales first,
     then z-score the residual variance.
   - *Time-step caveat*: datasets with different Δt or U_ref have different
     convective time steps per network step. Either resample trajectories to
     a common convective cadence at generation time, or condition on the
     step scale.
4. **Explicit conditioning beats implicit inference.** Field/dataset
   embeddings (MPP), DiT-style task conditioning (PDE-Transformer — same
   group as P3D), mask channels (DPOT). For us: the geometry channels *are*
   the conditioning, optionally plus a learned per-area embedding through
   P3D's native adaLN conditioning hook (the `pde_parameters` path our
   wrapper already exposes as `param_conditioning="native"` — currently
   scalar-only, extensible to an embedding).
5. **Rollout stability in mixtures.** No dedicated study of
   pushforward × multi-dataset interaction exists. DPOT is the existence
   proof that noise-injection training scales to large multi-dataset
   autoregressive pretraining — crucially with noise scaled to the sample
   norm (`ε ~ N(0, ϵ‖u‖I)`, ϵ ≈ 5e-4), so the perturbation auto-adapts per
   dataset. Transferable rule: apply pushforward/noise **in normalized
   space** so one stability hyperparameter serves all datasets. Our
   pushforward curriculum + `grad_unroll_steps=2` carries over unchanged.
6. **Joint pretrain → cheap per-area fine-tune** is the deployment recipe
   validated by Poseidon (split learning rates: transferred backbone at
   small LR, fresh heads/embeddings at large LR) and consistent with the
   MPP/DrivAerNet++ pooling penalty: expect joint-only to be a few percent
   worse per city than a specialist; short fine-tuning recovers it. Our
   `init_weights_path` warm start already supports the crude version.
   LoRA/adapters for PDE surrogates exist (MORPH, F-Adapter) but are
   immature — not a first move.
7. **Data augmentation.** Lie-point-symmetry augmentation gives up to an
   order of magnitude sample-efficiency for neural PDE solvers
   (Brandstetter et al., ICML 2022). For urban flow with a ground plane the
   valid subgroup is **rotations about z and horizontal reflections, with
   co-rotated (u, v) components and co-rotated `inflow_angle`** — the urban
   FNO study confirmed rotating geometry + field + inflow consistently works
   (0.34 vs 0.65 m/s error). Two repo-specific cautions: (a) 90° rotations
   swap x/y — only valid if grid + BCs are compatible (the y-periodic
   uDALES convention breaks x↔y symmetry; PALM/pylbm cases may allow it);
   (b) reflections must flip the sign of the reflected velocity component.
8. **How many geometries?** No clean scaling law. Bracketing evidence: one
   geometry fails cross-city (arXiv:2501.05499); 163 synthetic layouts
   suffice at 2D pedestrian level; 10,000 procedural 2D layouts train a
   usable optimization surrogate (arXiv:2603.21210); even 8,000 car shapes
   leave *cross-family* generalization hard (DrivAerNet++). Realistic read
   for us: a handful of real cities won't generalize to unseen layouts, but
   **procedurally generated building arrays** (random blocks/streets at
   matched grid spacing, pushed through the existing
   `generate_training_data.py` pipeline) are the published way to fill
   geometry space cheaply. O(10²) layouts is where 2D urban results start
   working; treat that as the initial target.

## 5. What P3D specifically offers / constrains

- **Conditioning hook:** upstream P3D has a combined embedding layer
  (diffusion-timestep + scalar `pde_parameters` adaLN modulation in every
  block). Our wrapper's default `param_conditioning="channels"` bypasses it.
  For multi-area training this hook is the natural place for a learned
  **area/dataset embedding** or richer global conditioning (wind direction,
  step scale) — a modest wrapper extension (embedding table → the adaLN
  path) rather than a fork of upstream.
- **Patchwise/global structure:** P3D's windowed attention sees small
  neighborhoods; global context flows through a dedicated mechanism. A
  binary mask inside one window is nearly information-free; an SDF channel
  restores global geometry awareness *within every window*. This makes the
  mask→SDF upgrade unusually well-matched to this architecture.
- **Static-shape compilation:** `compile_dynamic=False` is load-bearing
  (inductor backward-stride assert otherwise — see the P3D gotchas memory).
  Per-geometry homogeneous batches keep each distinct grid shape a separate
  static compilation; a few shapes = a few one-off compiles. Alternatively,
  pad/crop all areas to one common grid shape at dataset-generation time and
  avoid the issue entirely.
- **Fixed factories:** upstream S/B/L bake in depth/heads/hidden; capacity
  scaling for a much wider data distribution means moving S→B→L, not tuning
  internals. Multi-geometry training is a good reason to expect the next
  size up to pay off (all foundation-model results above are in the
  "diverse data rewards scale" regime).
- **DD wrapper synergy:** `DomainDecomposed` (docs §13–§19) already accepts
  P3D as the per-patch fine net via `extra_in_channels`, and is the one
  architecture with `domain_flexible=True` (spacing-only domain check). If
  different areas need different **domain sizes**, DD is the existing
  mechanism for that — and its coarse-context channel plays the same
  "global geometry information reaches every patch" role as an SDF.

## 6. Recommended plan (phased, cheapest-first)

**Phase 0 — make multi-geometry mechanically correct** (no research risk):

1. Multi-root dataset: either a thin `ConcatDataset` over per-geometry
   `TransitionDataset`s or a `root_dirs: [...]` extension; geometry loaded
   **per trajectory's dataset**, not from the global first file (B1, B5).
2. **Per-geometry homogeneous batches** via a batch sampler that draws each
   batch from one sub-dataset (uniform over datasets to start — MPP-style
   balancing for free). This preserves the `transition_collate`
   ship-geometry-once optimization (B2), keeps torch.compile shapes static
   per geometry (B6), and respects `drop_last` per dataset.
3. Fix `BaseTraining._prepare_batch` to take the geometry from the *current*
   batch instead of caching the first one (B3). Cache per-geometry masks
   keyed on dataset id if the GPU transfer matters.
4. Normalization stats streamed over **all** train splits with each file's
   own fluid mask (B4). Keep a single global mean/std initially (Aurora
   pattern) — simplest, and z-scored residual prediction already tolerates
   moderate scale spread.
5. Checkpoint metadata: record the *set* of trained domains (or move to a
   spacing-only check) so `_check_domain` can accept any trained geometry
   (B7). Ensure new-area grids satisfy the divisibility-by-16 rule (B8).

*Acceptance test: train on {existing area + one new area}, verify per-area
val loss ≈ the respective single-area runs (small gap expected), and
byte-identical behavior on a single-area config.*

**Phase 1 — geometry representation** (highest-evidence upgrade):

6. Add a **clamped SDF channel** alongside the binary mask
   (`distance_transform_edt` on the voxel mask, signed via inside/outside,
   normalized by a characteristic height, clamped). Computed once per
   geometry — either at data-generation time into the sample files, or
   lazily in the dataset/forward model from the mask (cheap, and keeps old
   datasets usable). The forward model computes it from the voxelized STL
   the same way, so train/inference stay consistent.
7. Optional second step if SDF saturates: ∇SDF channels or FLUME-FNO-style
   multi-directional distances.

**Phase 2 — multi-dataset training strategy:**

8. **Per-dataset normalized (relative) MSE** so no area dominates gradients.
9. Weighted dataset sampling (`p_k ∝ w_k/|D_k|`) once areas have very
   different sizes.
10. **Physics nondimensionalization experiment:** scale each dataset's
    velocities by its reference inflow speed (and check the convective time
    step per network step is comparable across datasets; resample at
    generation time if not). Compare against pure global z-score.
11. DPOT-style per-sample-norm **noise injection** (ϵ ≈ 1e-4–1e-3 in
    normalized space) as a complement to the existing pushforward
    curriculum for rollout stability across regimes.
12. Optional: learned per-area embedding through P3D's native adaLN
    conditioning hook — try *without* it first; the geometry channels may
    suffice, and an area embedding hurts unseen-layout generalization if
    the model leans on it.

**Phase 3 — data scale and augmentation:**

13. Rotation/reflection augmentation about the vertical axis with correctly
    transformed (u, v) and co-rotated `inflow_angle` (respecting each
    backend's BC symmetry).
14. **Procedural geometry generation**: random building-array STLs at the
    trained cell spacing, pushed through the existing
    `generate_training_data.py` pipeline. This — not collecting more real
    cities — is how every published multi-geometry urban result got its
    training set. Target O(10²) layouts initially.
15. Deployment recipe: **joint pretrain on everything → short per-area
    fine-tune** (split LRs à la Poseidon; `init_weights_path` already gives
    the crude version). Evaluate with one geometry fully held out.

**Phase 4 — architecture escalation (only if the above plateaus):**

16. Scale P3D S → B/L (diverse data rewards capacity).
17. `DomainDecomposed` with a P3D fine net for heterogeneous **domain
    sizes** (already implemented, `domain_flexible`).
18. Separate geometry-encoder latents (GINO/DoMINO/AB-UPT family) — highest
    risk; all published evidence is on steady targets, not autoregressive
    rollout. The in-repo UPT is the cheap first probe of this family.

**Evaluation harness to build alongside phase 0:** per-area val/rollout RMSE
(reported separately, never pooled), plus a held-out-geometry track — that is
the metric the whole effort is about, and none of the current tooling
reports it.

## 7. Risks and open questions

- **Pooling penalty is real** (DrivAerNet++ R² 0.9 → 0.6; MPP's admission).
  Budget for per-area fine-tuning rather than expecting one checkpoint to
  match specialists everywhere.
- **No published precedent** for 3D *autoregressive* multi-city urban
  surrogates — the plan above composes adjacent results; expect surprises
  at the rollout-stability × geometry-diversity interaction (unstudied).
- **Param-set heterogeneity** (B5): pyudales adds
  `pressure_gradient_magnitude`, PALM/pylbm don't. Mixing *backends* (not
  just geometries) needs a union-and-zero-pad param convention (DPOT-style)
  — recommend restricting multi-geometry training to one backend first.
- **Different grid shapes** interact with compile/cudnn autotuning; if the
  recompile cost annoys, standardize all areas to one grid shape at
  generation time.
- **How much geometry diversity the smoke-scale hardware can generate** —
  each new area is a full CFD ensemble run; procedural layouts at the
  current `medium` preset cost ~1 ensemble run each. The DRAM-bandwidth
  ceiling (memory: ensemble scaling caps at ~4–8 workers) bounds throughput.

## 8. Key references

*Geometry encoding:* Guo, Li, Iorio, KDD 2016 (CNN + SDF steady flow) ·
Bhatnagar et al. 2019, arXiv:1905.13166 · Chen & Thuerey, arXiv:2109.02183 ·
Rabeh et al. 2025, arXiv:2503.17289 (SDF + derivative constraints) ·
"Shape-informed surrogate models based on SDF domain encoding," JCP 2025 ·
FLUME-FNO, arXiv:2503.19708 (MDDF, 3D urban, 23 sims).

*Geometry-aware operators / car aero:* GINO, NeurIPS 2023, arXiv:2309.00583 ·
DoMINO (NVIDIA), arXiv:2501.13350 · Transolver, ICML 2024, arXiv:2402.02366;
Transolver++, arXiv:2502.02414 · UPT, arXiv:2402.12365; AB-UPT,
arXiv:2502.09692 · DrivAerNet++, arXiv:2406.09624 (pooled-training R² drop) ·
CarBench, arXiv:2512.07847.

*Urban wind surrogates:* MLP-mixer pedestrian wind, Cambridge Env. Data
Science 2025 (163 geometries) · Clemente et al., CACAIE 2024 (2,384
geometries) · Feilian v2 / UrbanTALES, arXiv:2510.27101 · urban FNO
cross-city failure, arXiv:2501.05499 · AB-SWIFT (3D, randomized urban
geometries), arXiv:2603.25635 · video-model urban surrogate (10k procedural
layouts, 2D), arXiv:2603.21210 · graph-diffusion urban flow,
arXiv:2512.14725.

*Multi-dataset training:* MPP, NeurIPS 2024, arXiv:2310.02994 (RevIN,
uniform micro-batch sampling, NMSE) · DPOT, ICML 2024, arXiv:2403.03542
(weighted sampling, per-norm noise injection) · Poseidon, NeurIPS 2024,
arXiv:2405.19101 (all2all pairs, diversity ablation, split-LR fine-tune) ·
PDE-Transformer, ICML 2025, arXiv:2505.24717 (same group as P3D; DiT
conditioning) · Aurora, Nature 641 (2025), arXiv:2405.13063 ·
PDEformer-2, arXiv:2507.15409 · The Well, arXiv:2412.00568 · Zhou et al.,
TMLR 2024, arXiv:2406.08473 (when pretraining pays off) · bias-aware FM
benchmark, arXiv:2605.29283.

*Normalization / augmentation / rollout stability:* RevIN, ICLR 2022 ·
Brandstetter et al., Lie-point-symmetry augmentation, ICML 2022,
arXiv:2202.07643 · Wang, Walters, Yu, ICLR 2021, arXiv:2002.03061 ·
message-passing PDE solvers (pushforward), ICLR 2022, arXiv:2202.03376 ·
Stachenfeld et al. (noise injection), ICLR 2022 · List et al., unrolled
training, arXiv:2402.12971 · PDE-Refiner, NeurIPS 2023, arXiv:2308.05732 ·
P3D, Holzschuh et al., arXiv:2509.10186, github.com/tum-pbs/P3D.

*Caveat: 2512/26xx arXiv ids are 2025–26 preprints verified to exist during
this research session but only partially read; a few well-known ids
(DrivAerNet++, UPT, RevIN venue details) were cited from model knowledge
without re-fetching.*
