# Compact latent generator with full latent reach

**Status: open (2026-10-23).** Follows PR #221 (`feat/small-velocity-net`) and
the study in §1. Supersedes PR #221's default of a 176-wide velocity net.
Phase 1 is implemented on `feat/latent-generator-full-reach`. CPU tests pass.
The idealized GPU proof of concept is still pending: the local RTX 3090 failed
under sustained load on 2026-10-23.

**Goal.** A latent flow-matching generator of **10–40M trainable parameters**
that matches or beats the 177M full-width model
(`latent_generator_b_realistic_adaln`) on validation loss and the acceptance
metrics of `scripts/surrogate/evaluate_latent_generator.py`.

**Problem.** The frozen Tadpole-B AE gives tokens of `D = 3 × 512 = 1536`
channels on the padded grid / 16 (~130–450 tokens per realistic sample). The
velocity net `Linear(D→H) → n blocks of width H → Linear(H→D)` can only write
in an H-dimensional subspace, and its cost grows with H². At `H = D` it has
120–250M parameters; at `H = 176` (PR #221) it has 5.5M but cannot shape
~89% of the token directions.

## 1. Evidence

All measurements use the local test split of `pyudales_realistic` (10
trajectories × 120 frames). The PCA basis was fitted and evaluated on the same
geometries (different frames), which flatters PCA-based results.

1. **The latent is high-dimensional.** PCA of z-scored per-token latents
   (64,512 tokens):
   - The top 176 components hold 37% of the variance. 90% needs 859, 99% needs
     1140.
   - Eigenvalues sit on a plateau near 1 from about component 30 to 1000.
   - About 400 directions have variance below 0.1, falling to about 1e-7.
2. **The decoder uses about 1100–1300 directions.** Real latents projected on
   the top-k components, with the rest set to the mean, then decoded (60
   held-out snapshots):

   | k | rel. L2 error ÷ AE's | mean-profile RMSE ÷ AE's |
   |---|---|---|
   | 176 | 7.9× | 39× |
   | 768 | 4.2× | 15× |
   | 1024 | 2.3× | 6× |
   | 1280 | 1.08× | 1.17× |

   - The vertical velocity `w` lives mostly in the mid- and low-variance
     directions.
   - Filling the unreached directions with unit-variance noise, which is what
     a confined generator produces, gives about 6× the real divergence.
     Filling them with noise of the true variances is about 2× better than
     unit noise.
3. **Even a perfect generator inside the trained h176 output subspace would be
   ~6× the AE error.** The trained `out_proj` span holds 23% of the variance
   (a random 176-dim subspace holds 11.5%). In the decoder test it is worth
   about k ≈ 600.
4. **The h176 run is capacity-limited, not undertrained.**
   - Its val loss is flat from epoch 20: 1.292 → 1.285 by epoch 155.
   - The 177M model reaches 1.256 at epoch 5 and 1.074 at epoch 213.
   - The 232M `input_film` restart run reaches 1.051, but has never been
     evaluated.
5. **Width alone does not pass W1.**
   - The 177M model passes the profile, divergence and diversity checks.
   - Its W1 is 0.093 against a bound of 0.016. Heun sampling at 25 steps only
     brings it to 0.083.
6. **A network-free Gaussian generator fails badly.** Tokens are drawn iid
   from `N(μ, Σ)` (G1) or from a per-channel Gaussian (G2), decoded, and run
   through the acceptance gate on 60 held-out snapshots × 4 seeds:

   | source | profile RMSE | W1 (u / v / w) | div RMS | gate |
   |---|---|---|---|---|
   | AE recon | 0.007 | 0.013 (0.022 / 0.013 / 0.004) | 0.019 | – |
   | G1 full-covariance Gaussian | 1.52 | 0.72 (0.99 / 1.14 / 0.045) | 0.084 | fails profile, W1, divergence |
   | G2 per-channel Gaussian | 1.52 | 0.85 (1.16 / 1.31 / 0.084) | 0.106 | fails profile, W1, divergence |
   | 177M adaLN (stored eval) | 0.062 | 0.093 (0.104 / 0.089 / 0.086) | 0.024 | fails W1 only |

   - The Gaussians are unconditional, so the mean flow is a mix of all inflow
     directions. Conditioning is what the network must supply.
   - Full covariance beats the per-channel one on W1 and divergence, which
     supports the PCA basis over the channel basis for the skip.
   - **On `w`, plain G1 (W1 0.045) beats the trained 177M model (0.086).** The
     trained generator gets the small-scale, low-variance directions wrong,
     and that is where the Gaussian skip should help most.
7. **Pending:** the evaluation of the 232M `input_film` restart run.

**Consequences.**
- Neither of these fixes it:
  - a uniformly wider net in budget (H = 768 is still below 859);
  - the PR #221 per-channel skip, which is full rank but only rescales noise
    in the channel basis.
- The output path must reach all ~1300 useful directions *with context*
  (neighbouring tokens, geometry, parameters).
- Attention, the part that costs H², can stay narrow.

## 2. Design

The new knobs follow AGENTS.md "no-op when absent": when a knob is absent
nothing is built, old artifacts still load strictly, and default runs stay
byte-identical.

### 2.1 Gaussian skip in the PCA basis — `gaussian_skip: true`

- **Buffers.** `latent_pca_basis` (D×D), `latent_pca_eig` (D) and
  `latent_pca_mean` (D), all in the z-scored space.
  - They are filled by `compute_latent_normalization`, which also accumulates
    the raw second moment in float64.
  - The z-scored covariance follows as `Σ_z = S⁻¹ (Σ_raw) S⁻¹`, with
    `S = diag(std)`, so one pass suffices.
  - Eigenvalues are floored, e.g. at `1e-4`.
  - Memory: 9.4 MB in fp32.
- **Skip.** With `y = Pᵀ(z_τ − τμ)` and `m = Pᵀμ`, the exact velocity of a
  per-token Gaussian `N(μ, PΛPᵀ)` under `z_τ = (1−τ)z₀ + τz₁` is

  `v_G = P (m + c(τ) ⊙ y)`, where `c_j(τ) = (τλ_j − (1−τ)) / (τ²λ_j + (1−τ)²)`.

  - It has no `1/(1−τ)` singularity, and `c_j = a(τ)` when `λ_j = 1`.
  - `v = v_G + net(...)`. The net's output layer starts at zero, so at init the
    model *is* the Gaussian, with the correct variance in every direction.
    That includes the ~400 near-empty directions, which the PR #221 skip fills
    with unit noise.
  - Cost: two D×D matmuls per token per evaluation (~0.7 GFLOP at 300 tokens),
    and 0 trainable parameters.
- **Relation to PR #221.** This branch starts from `dev`, so it has no
  `latent_skip`. `gaussian_skip` is independent of it: absent or `false`
  builds no buffers and changes nothing. The Gaussian skip generalises PR
  #221's `a(τ)` (`c_j = a(τ)` at `λ_j = 1`) from the channel basis to the
  eigenbasis. If needed later, a zero-initialised per-direction correction
  `Linear(t_emb → D)` in the eigenbasis can be added.
- **Statistics.** Computed in both places latent statistics come from:
  `compute_latent_normalization` (no cache) and the latent cache
  (`latent_cache.py` per-file sums plus `load_latent_stats`). The cache also
  stores the per-file sum of outer products.
- **Discretisation (measured; the original "no stiffness" claim was wrong).**
  Integrating the exact Gaussian velocity alone, the generated/true variance
  ratio per eigenvalue is:

  | λ | Euler, 50 steps | Heun, 50 steps |
  |---|---|---|
  | 10 | 0.93 | 1.00 |
  | 1 | 0.95 | 1.00 |
  | 1e-2 | 0.84 | 1.01 |
  | 1e-3 | 0.59 | 1.08 |
  | 1e-4 | 0.18 | 1.87 |

  - The error comes from discretising the ODE, not from the skip. A net that
    learned the exact flow would show the same error, so it also applies to
    the existing models.
  - Euler under-disperses the low-variance directions, which is where `w`
    lives (§1.2).
  - The tests now check that Heun at 50 steps (and Euler at 1000) matches the
    covariance within 3%, and that Euler at 50 steps matches the scalar Euler
    recursion.
  - Follow-up, cheap: evaluate with Heun (already in the step sweep) and try a
    grid refined near `τ → 1` (the opposite of `time_shift > 1`, which refines
    near noise). Compare W1(w) for each.

### 2.2 Design A — narrow trunk plus a wide per-token head (`head_hidden`, `head_layers`)

- **Head.** `v_net = out_proj(h) + head(h, z_τ, c)`, where `head` acts on each
  token separately:
  - `Linear(H + D → W)`;
  - `head_layers` adaLN MLP blocks at width W (MLP ratio 2, shift/scale/gate
    from the same conditioning embedding as the trunk);
  - `LayerNorm`, then a zero-initialised `Linear(W → D)`.
- **Why it reaches every direction.** The limit was the linear read-out, not
  the information. The conditioning is low-dimensional (48 parameter values
  plus geometry). A nonlinear head on `concat(h, z_τ)` lets context reach all
  D directions (measured: context-span rank 1536).
- **Prior art.**
  - The wide "DiT^DH" head of RAE [Zheng et al. 2025].
  - The per-token adaLN MLP head of MAR [Li et al. 2024].
  - The encoder/velocity-decoder split of DDT [Wang et al. 2025].
- **Config A.** `hidden_size: 320, n_layers: 6, num_heads: 8, mlp_ratio: 4,
  film_hidden: 256, head_hidden: 1024, head_layers: 2`: 27.6M, measured at
  D = 1536.
  - The first estimate (~19M, with `head_hidden: 1536`) was wrong. That build
    measures 40.6M, because each head block at width W costs about 4W² plus
    its modulation.
  - Other measured builds: head 1536 with 1 layer is 30.0M; head 768 with 2
    layers is 22.6M.
- **Where it lives.** In `latent_velocity_adaln.py`, built only when
  `head_hidden` is set (adaLN conditioning only).

### 2.3 Design B — full-width residual stream with narrow attention (`velocity_conditioning: adaln_bottleneck`)

- **Block.** The residual stream stays at width D. Each block:
  - `LN → Linear(D→H) → attention at width H → Linear(H→D)` (zero-init, gated);
  - then a per-token MLP `D → M → D` (zero-init, gated);
  - with adaLN shift/scale/gate at width D.
- **Why it reaches every direction.** Every block can write any direction,
  with spatial context.
- **Prior art.**
  - Bottleneck adapters [Houlsby et al. 2019].
  - The observation that attention's head/projection width can sit below the
    model width [Bhojanapalli et al. 2020].
  - Narrow-latent/wide-interface computation, as in RIN [Jabri et al. 2022].
- **Config B.** `hidden_size: 256, n_layers: 4, num_heads: 8, mlp_hidden: 512`,
  measured at 26.7M with `film_hidden: 256` (the design study's 24.1M prototype differed slightly).
  - Most of it is the D-wide adaLN modulation (~1.7M per block). If that
    dominates, share one modulation across blocks or feed it from a smaller
    `film_hidden`.
  - At 6 layers it is 34.4M.
- **Where it lives.** A new sibling class in `latent_velocity_adaln.py`,
  selected by the new `velocity_conditioning` value. The existing values are
  untouched.

### 2.4 Training-objective knobs (no parameters)

- **Clean-sample prediction (`prediction: data`; `velocity` when absent).**
  - The net predicts `x̂`. Then `v = (x̂ − z_τ) / max(1−τ, 0.05)`, and the loss
    stays the v-MSE against `z₁ − z₀`, so val losses stay comparable.
  - With the Gaussian skip, `x̂ = x̂_G + net`, where
    `x̂_G = μ + P diag(g) Pᵀ (z_τ − τμ)` and `g_j = τλ_j / (τ²λ_j + (1−τ)²)`.
  - Li & He [2025] show x-prediction stays well posed when the net is narrower
    than the token; v- or ε-prediction then degrades badly.
- **Time shift (`time_shift: α`; 1 when absent).**
  - Training: sample `σ = 1−τ` uniformly, then map
    `σ' = ασ / (1 + (α−1)σ)`.
  - Sampling: use the same warped step grid.
  - The RAE rule `α = sqrt(m / 4096)`, with `m` = tokens × D ≈ 4.6e5, gives
    α ≈ 10 [Zheng et al. 2025; Esser et al. 2024].
  - Sweep α ∈ {1, 3, 6, 10} on the best architecture only.

### 2.5 Not pursued (reasons in the team reports)

- **Uniformly wider nets** (H 384–768): context reaches only H directions.
- **FiLM of `z` by `h`:** context still moves only H directions.
- **Channel-group tokens:**
  - one token per variable is full reach only from H = 512 (32M), at 4× the
    activation memory;
  - 12 groups of 128 need 3.1 GB of activations per sample.
- **Generating only the top-k PCA coordinates:** k must be ~1100–1300 (§1.2),
  so it saves nothing.
- **A smaller-latent AE:** `TadpoleAE` only offers fixed sizes (S gives
  D = 768). A realistic S AE would cost 100+ GPU-h plus a new latent cache, and
  768 is still far above budget widths.
- **Regularising the AE for diffusability** [VA-VAE; EQ-VAE; Skorokhodov et al.
  2025]: kept as the fallback if Phase 2 fails W1 everywhere.

## 3. Phases

**Phase 0: baselines (running 2026-10-23, no new code).**
1. Evaluate the Gaussian-only generator (§1.6a).
2. Evaluate `latent_generator_b_realistic_restart` (232M) with
   `evaluate_latent_generator.py`.
3. Record both in §1.

**Phase 1: implementation.** On a new branch from `dev`, not on top of PR
#221's default config.
1. Write the §2.1 Gaussian skip:
   - extend `compute_latent_normalization` with the covariance and add the
     buffers;
   - add `_gaussian_velocity`;
   - make `gaussian_skip` absent → current behaviour.
2. Add the Design A head and the Design B block.
3. Add `prediction` and `time_shift`. `sample()` uses the same warping.
4. **(done: `latent_spectrum` in `evaluate_latent_generator.py`; generated latents re-encoded until `sample()` returns its latents)** Add an eval diagnostic: the eigen-spectrum of generated vs real latents in
   the real PCA basis (variance ratio per direction band). This shows directly
   whether the low-variance directions come out too noisy.
5. Write tests in `tests/neural_surrogates/test_tadpole_latent_flow.py`:
   - old artifacts still load strictly;
   - at init with `gaussian_skip`, the velocity equals `v_G`, and Euler sampling of the
     untrained model reproduces `N(μ, Σ)`;
   - the context Jacobian of designs A and B has rank D (small D);
   - parameter counts are recorded and `extra_repr` shows the new knobs;
   - `prediction: data` with `net = 0` gives the same `v` as the Gaussian
     velocity;
   - the time-shift map is monotone, and maps 0 → 0 and 1 → 1.
6. **(done)** Update the docs: `docs/neural_surrogates.md` (latent-generator section,
   with the §1 evidence condensed) and `configs/surrogate/train_latent_generator.yaml`.
   The default stays full width until Phase 2 picks a winner.

**Phase 2: training runs.**
- **Setup.**
  - Same B AE, latent cache, val seed, batch 40 and lr 1e-4 as the 177M run,
    so val losses are comparable.
  - One A100 each, under a 48 h limit.
  - An estimated 65–85 s per epoch puts 300 epochs at about 6–7 h.

| run | design | params | purpose |
|---|---|---|---|
| R1 | A + Gaussian skip | 27.6M | main candidate |
| R2 | B + Gaussian skip | 26.7M | main candidate |
| R3 | uniform H512 L6 + Gaussian skip | 32M | control: rank without full reach |
| R4 | best of R1/R2 + `prediction: data` | same | objective |
| R5 | best + `time_shift` α ∈ {3, 6, 10}, 100 epochs each | same | schedule |

- **Comparators:** 5.5M h176, 177M adaLN, 232M restart, and the Gaussian
  baseline.
- **Evaluation:** every run goes through `evaluate_latent_generator.py` (same
  snapshots and seeds) plus the spectrum diagnostic.

**Decision rule.**
- **Adopt** the smallest model that meets all three:
  - val loss within 3% of the 177M model's at the same epoch;
  - W1 ≤ the 177M model's;
  - no acceptance check worse than the 177M model's.
- **If none passes W1 but they match the 177M model:**
  - W1 is not a capacity problem.
  - Next step: the AE side, i.e. a noise-augmented decoder [Zheng et al.
    2025], or diffusability regularisation.
  - Plus a check of the W1 bound itself (bootstrap vs AE).

## 4. Risks

- **The PCA basis comes from the training latent cache.** It must be computed
  over the same snapshots as the normalisation. Check that the train and test
  spectra agree, since §1 used test only.
- **Padding tokens are included in the statistics.** Check the spectrum with
  padding tokens masked out; if they differ much, compute the statistics on
  fluid/valid tokens only.
- **The Gaussian is per token and iid in space.** Spatial correlations must
  still come from attention. Design A's head has no spatial view of its own.
  If A's samples show token seams or weak large-scale structure where B's do
  not, prefer B.
- **Design B's D-wide adaLN is memory-heavy at batch 40.** Measure the peak
  before the long runs; fall back to a shared modulation.
- **PR #221.** Its `latent_skip` (channel) stays as a supported option. Its
  default config (`hidden_size: 176`, `latent_generator_s_h176`) should not
  land. Either drop it from the PR, or close the PR in favour of Phase 1.

## 5. References

All arXiv IDs were checked against the arXiv API on 2026-10-23.

| Topic | Reference |
|---|---|
| Our AE and generator recipe | Q. Liu, F. Koehler, B. Holzschuh, N. Thuerey. *Tadpole: Autoencoders as Foundation Models for 3D PDEs with Online Learning.* 2026. arXiv:2605.15284 |
| Width < token dim loss floor; wide DiT^DH head; dimension-dependent shift; noise-augmented decoder | B. Zheng, N. Ma, S. Tong, S. Xie. *Diffusion Transformers with Representation Autoencoders.* 2025. arXiv:2510.11690 |
| Clean-sample (x) prediction for narrow nets | T. Li, K. He. *Back to Basics: Let Denoising Generative Models Denoise.* 2025. arXiv:2511.13720 |
| Per-token adaLN MLP head (wide head widths) | T. Li, Y. Tian, H. Li, M. Deng, K. He. *Autoregressive Image Generation without Vector Quantization* (MAR). 2024. arXiv:2406.11838 |
| Encoder / velocity-decoder split | S. Wang, Z. Tian, W. Huang, L. Wang. *DDT: Decoupled Diffusion Transformer.* 2025. arXiv:2504.05741 |
| Bottleneck adapters (narrow inner width, wide residual stream) | N. Houlsby et al. *Parameter-Efficient Transfer Learning for NLP.* 2019. arXiv:1902.00751 |
| Attention width vs model width | S. Bhojanapalli et al. *Low-Rank Bottleneck in Multi-head Attention Models.* 2020. arXiv:2002.07028 |
| Narrow latent compute, wide interface | A. Jabri, D. Fleet, T. Chen. *Scalable Adaptive Computation for Iterative Generation* (RIN). 2022. arXiv:2212.11972 |
| Per-token channel-mixing MLPs | I. Tolstikhin et al. *MLP-Mixer: An all-MLP Architecture for Vision.* 2021. arXiv:2105.01601 |
| Gaussian / linear structure of learned scores | B. Wang, J. J. Vastola. *The Hidden Linear Structure in Score-Based Models and its Application.* 2023. arXiv:2311.10892 |
| Gaussian score approximation | B. Wang, J. J. Vastola. *The Unreasonable Effectiveness of Gaussian Score Approximation for Diffusion Models and its Applications.* 2024. arXiv:2412.09726 |
| Skip / output preconditioning | T. Karras, M. Aittala, T. Aila, S. Laine. *Elucidating the Design Space of Diffusion-Based Generative Models* (EDM). 2022. arXiv:2206.00364 |
| Rectified flow; resolution-dependent time shift | P. Esser et al. *Scaling Rectified Flow Transformers for High-Resolution Image Synthesis* (SD3). 2024. arXiv:2403.03206 |
| Flow matching | Y. Lipman, R. T. Q. Chen, H. Ben-Hamu, M. Nickel, M. Le. *Flow Matching for Generative Modeling.* 2022. arXiv:2210.02747 |
| Rectified flow | X. Liu, C. Gong, Q. Liu. *Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow.* 2022. arXiv:2209.03003 |
| DiT backbone | W. Peebles, S. Xie. *Scalable Diffusion Models with Transformers.* 2022. arXiv:2212.09748 |
| Latent channels vs generation quality | J. Yao, B. Yang, X. Wang. *Reconstruction vs. Generation: Taming Optimization Dilemma in Latent Diffusion Models* (VA-VAE). 2025. arXiv:2501.01423 |
| High-compression AEs, structured latents | J. Chen et al. *Deep Compression Autoencoder for Efficient High-Resolution Diffusion Models.* 2024. arXiv:2410.10733; J. Chen et al. *DC-AE 1.5.* 2025. arXiv:2508.00413 |
| Tokenizer scaling | P. Hansen-Estruch et al. *Learnings from Scaling Visual Tokenizers for Reconstruction and Generation* (ViTok). 2025. arXiv:2501.09755 |
| Diffusability of AEs (fallback) | I. Skorokhodov et al. *Improving the Diffusability of Autoencoders.* 2025. arXiv:2502.14831; T. Kouzelis et al. *EQ-VAE.* 2025. arXiv:2502.09509 |
| Representation alignment (fallback) | S. Yu et al. *REPA.* 2024. arXiv:2410.06940; X. Leng et al. *REPA-E.* 2025. arXiv:2504.10483 |
| Latent diffusion for physics emulation | F. Rozet et al. *Lost in Latent Space: An Empirical Study of Latent Diffusion Models for Physics Emulation.* 2025. arXiv:2507.02608 |
| Ordered latents | O. Rippel, M. A. Gelbart, R. P. Adams. *Learning Ordered Representations with Nested Dropout.* 2014. arXiv:1402.0915 |
| One token per variable | T. Nguyen et al. *ClimaX: A foundation model for weather and climate.* 2023. arXiv:2301.10343 |

Team reports with the full numbers (session scratch, not kept): literature
review, design study (16 prototypes), decoder-relevance test, run history.
