# Evaluating ESMDA parameter estimation for turbulent LES: metrics and figures

> **Status: design / research document** (not a maintained reference). Compiled
> 2026-07-29 from a literature survey (turbulence statistics, ensemble
> verification, urban-CFD validation practice) cross-checked against what
> `scripts/run_esmda_pipeline.sh` actually produces today. It proposes what the
> metric stage (`scripts/esmda/compute_esmda_metrics.py`) and figure stage
> (`scripts/esmda/make_esmda_figures.py`) should grow into. Verify code
> references against the tree before implementing.

## 1. Premise and framing

Parameter estimation for turbulent LES cannot be judged by instantaneous state
agreement. Two LES runs with *identical* parameters decorrelate after a
Lyapunov horizon; past it, pointwise RMSE measures chaos, not parameter
quality. The literature is explicit about this: Chang & Hanna (2004) show
stochastic turbulent variability sets a floor of roughly 20 % bias and 60 %
scatter that no model can beat, and the prescribed remedy is to **average
first, compare second**. Johnson, Wu & Ihme (2017) make the same argument for
LES specifically: a single realization must be replaced by a distribution of
outcomes.

Successful assimilation therefore means answering **two logically distinct
questions**, each with its own metric family:

1. **Statistical fidelity** — does the posterior ensemble reproduce the
   *statistical character* of the true flow (mean fields, Reynolds stresses,
   PDFs, spectra, length scales)? Compute a statistic `S` on truth and on the
   ensemble, then measure a distance in `S`-space.
2. **Statistical consistency / calibration** — is the truth a plausible draw
   from the posterior? Apply probabilistic verification (CRPS, rank
   histograms, spread–skill, innovation χ²) **to the statistics**, not to
   instantaneous fields. An instantaneous-field rank histogram is U-shaped for
   chaotic reasons regardless of parameter quality; a rank histogram on
   window-averaged velocity is the object that should be flat.

On top of these sit two more layers specific to this setup:

3. **Parameter space** — in these synthetic twin experiments the true
   parameters are known, so parameter recovery, posterior contraction, and
   identifiability can be measured directly.
4. **ESMDA health** — the algorithm itself has known pathologies
   (over-contraction, noise over-fitting, member collapse) with dedicated
   diagnostics from the history-matching literature.

### Cross-cutting principles (non-negotiable in the implementation)

- **Never compare instantaneous fields between truth and ensemble.** Every
  state-space metric and field figure operates on time averages, distributions,
  or spectra. (The current `vel_magnitude_rmse` on the ensemble-mean field and
  the instantaneous-panel animation stay useful only as qualitative sanity
  checks — do not headline them.)
- **Order of operations:** per member → time-average over the *same*
  stationary window as truth → then reduce across members (mean / quantiles /
  scores). Never time-average the ensemble-mean field and derive spread from
  it.
- **Self-distance floor.** Compute every fidelity metric between two
  independent halves of the truth run itself. That number is the irreducible
  chaotic/sampling floor; report every model-vs-truth score relative to it.
  This converts "smaller is better" into the actually meaningful criterion:
  *indistinguishable from truth's own variability, or not*.
- **Sampling error bars on every statistic.** `var(⟨u⟩_T) ≈ 2 σ_u² T_int / T`
  with `T_int` the integral time scale; use block bootstrap where the formula
  is awkward. Second moments converge 2–4× slower than means. Differences
  below the sampling error are noise, not signal.
- **Effective sample size ≪ nominal.** LES time series are autocorrelated
  (`n_eff = n·Δt / (2 T_int)`); adjacent parameter time-knots are correlated
  by the AR prior. Sub-sample at ≥ one integral time scale before any
  histogram or χ² test, and never quote KS p-values.
- **Fair (unbiased) ensemble scores only.** Pairwise-difference scores must
  divide by `M(M−1)`, not `M²` (§6.1) — the biased form is minimized by a
  collapsed ensemble, i.e. it rewards the exact failure mode being tested.
- **Prior vs posterior, always.** Every score is reported for both, ideally as
  a skill score `1 − score_post / score_prior`. A posterior number without its
  prior baseline is uninterpretable — and "posterior worse than prior" is a
  real, previously observed outcome in this repo.
- **Held-out sensors.** Scores on assimilated sensors are circular; the
  strongest single piece of evidence is skill on sensors withheld from the
  update. `xie_and_castro` already defines validation sensors; `barcelona`
  currently defines none and should.

## 2. What the pipeline produces today, and what constrains the design

Inventory (verified against a completed run and the code):

- **Artifacts:** `true_state.nc`, `true_params.nc`, `prior_params.nc`,
  `posterior_params.nc` (full parameter ensembles, tiny),
  `posterior_state_mean.nc` (ensemble-reduced: mean u/v/w[/pres], `vel_mean`,
  `vel_std`; no member dimension), and per-window
  `windows/window_{w}_posterior_state.nc` — the **only** place the full
  `(ensemble, time, z, y, x)` state exists. That file is ~1 GB for a tiny
  30×40×16 case and tens of GB at Barcelona resolution, so **every full-field
  metric must stream** over members or z-slices (existing patterns:
  `_streaming_state_summary`, `_vel_field_4z`; ~2 reader threads is the
  DRAM-bound plateau).
- **Current metrics** (`run_summary.yaml`): parameter RMSE/CRPS (+ reduction
  vs prior), a 4-z-slice `vel_magnitude_rmse` of the ensemble-mean field vs
  truth, and per-sensor vector RMSE + energy score. Nothing spectral,
  distributional, or turbulence-statistical exists anywhere in the repo — all
  of §4 is net-new code.
- **Cross-grid runs are the default** (e.g. PALM truth → surrogate ensemble):
  truth and ensemble differ in dims, staggering, and resolution. Two working
  patterns exist: interpolate truth onto the assim grid (fields), or evaluate
  both at shared physical points (sensors). Spectra and two-point statistics
  must either use probe/sensor series (grid-free) or restrict to the common
  resolved band after an explicit regrid step.
- **uDALES states are C-grid staggered** (u/v/w on different coords); the
  existing `|U|` combines components by index. For Reynolds stresses
  (`⟨u′w′⟩`) components must be co-located (interpolate to cell centers)
  rather than index-combined.
- **Not persisted today, blocking the best DA diagnostics:** observations and
  predicted observations `g(θ_m)` are computed inside the smoother and
  discarded; per-ESMDA-iteration parameter ensembles are reduced to the final
  step; the prior state ensemble is off by default (`run.save_prior_state`).
  §5 marks which metrics need which of these persisted. The filtering side
  already has `CycleDiagnostics` (innovation χ², obs-space RMSE/spread) that
  ESMDA should mirror.

## 3. Layer A — parameter-space metrics (truth known)

Notation: parameters `θ ∈ R^{Np}`, ensemble members `θ_m`, `m = 1..M`, truth
`θ*`, prior `b` / posterior `a`, ensemble std with `ddof=1`.

| Metric | Definition | Read it as |
|---|---|---|
| Bias / RMSE, normalized | `(θ̄ᵃ − θ*)/σᵇ`; member-RMSE `√(mean_m(θ_m−θ*)²)` | Error in units of prior uncertainty — comparable across heterogeneous parameters. Report the existing `rmse_reduction_vs_prior` alongside. |
| **z-score** | `z = (θ* − θ̄ᵃ)/σᵃ` | The single most informative scalar. `|z| ≲ 2`: healthy. `|z| > 3` with small `σᵃ`: **over-confident posterior**, the canonical ESMDA failure. Pool over knots/windows: the set should look ~N(0,1). |
| Coverage | fraction of knots with `θ*` inside the central-α ensemble interval (exists as `per_knot_in_band`) | ≈ α if calibrated. Use order-statistic band edges, not interpolated quantiles, at M=50; report α=0.5 as well as 0.9 (much lower variance). |
| Rank / PIT of truth | `rank = #{m: θᵃ_m < θ*}`, PIT `= rank/M` | Flat over pooled (knot, window) instances = calibrated; U = under-dispersed; ramp = biased. One run gives few samples — treat shapes as suggestive, prefer the scalar z/coverage checks. Jolliffe–Primo contrasts test slope (bias) and convexity (dispersion) separately if enough samples exist. |
| Contraction ratio | `c = σᵃ/σᵇ` per parameter; joint: geometric mean of generalized eigenvalues of `(Cᵃ, Cᵇ)` | `c ≪ 1` = data informed the parameter; `c ≈ 1` = unidentifiable (see A-sens). No universal "good" value — which is why the next row exists. |
| **Contraction vs achievable** | Linear-Gaussian prediction from the *prior* ensemble: `C^pred = C_θθ − C_θd (C_dd + C_D)⁻¹ C_dθ`; ratio `r = σᵃ / √(C^pred_ii)` | `r ≈ 1`: contraction is justified by the data. `r ≪ 1`: **posterior narrower than the observations can justify** — spurious update from sampling noise / missing localization / bad α schedule. The only test that separates legitimate from spurious contraction without repeat experiments. Needs prior predicted obs persisted (§5). |
| Correlation / identifiable directions | compare `corr(Cᵇ)` vs `corr(Cᵃ)`; generalized eigenproblem `Cᵃ v = λ Cᵇ v` | New posterior off-diagonals = only a *combination* was constrained (likelihood ridge). Eigenvalues `λ ≪ 1` are the learned directions, loadings say which combinations; `#{λ < 0.5}` is a compact identifiability count. |
| Sensitivity / observability | prior-ensemble `S_ij = cov(θ_i, g_j)/(σ_θi σ_D,j)`; per-obs `SNR_j = σ_{g_j}/σ_{D,j}`; `DFS = tr[C_dd (C_dd + C_D)⁻¹]` | `SNR ≪ 1` ⇒ that observation is pure noise and cannot inform anything (three lines of code; would have caught the documented pypalm "posterior worse than prior" case). `DFS ≳ Np` needed to constrain all parameters; note the hard cap `DFS ≤ min(Nd, M−1)`. Column-wise `S` ranks sensors → directly actionable for placement. |

Numerical care: all ensemble covariances are rank ≤ M−1; use pseudo-inverse
with eps-scaled eigen-truncation (a rank-cut helper already exists in
`libs/data-assimilation`'s reduction code) and quote the retained rank as the
χ² dof. Include the finite-ensemble factor `(1+1/M)` on ensemble covariances
when the truth is compared against them.

## 4. Layer B — statistical fidelity of the state

### 4.1 One-point statistics and mean-field error norms (the baseline)

Compute per member and for truth, in one streaming pass over time
(accumulate `Σu_i` and `Σu_i u_j`):

- mean fields `U_i = ⟨u_i⟩_t`; Reynolds stresses
  `R_ij = ⟨u_i u_j⟩ − U_i U_j`; resolved TKE `k = ½R_ii`; optionally skewness/
  kurtosis (the first genuinely "turbulent" discriminators — a Gaussianized
  flow fails here before it fails on `k`). The off-diagonal `⟨u′w′⟩` is far
  more discriminating than `k` alone: a model can match TKE with entirely
  wrong anisotropy. State whether stresses are resolved-only (they are, here)
  since SGS contributions are non-negligible inside a canopy.

Score the **time-mean** fields (cell-wise over fluid cells, or at sensor
points) with the urban-CFD standards (Chang & Hanna 2004; COST 732; VDI
3783/9):

| Metric | Formula | Acceptance |
|---|---|---|
| **Hit rate q** (VDI 3783/9 — the one designed for *velocity*) | point counts if relative error ≤ D **or** absolute error ≤ W; `D = 0.25` | `q ≥ 0.66` |
| FAC2 | fraction with `0.5 ≤ pred/obs ≤ 2` | ≥ 0.5 (dispersion), ≥ 0.3 used for urban-LES velocity. **Only valid for positive quantities** (`|U|`, TKE) — meaningless for sign-changing components; use hit rate with the absolute allowance `W` there. |
| Fractional bias FB | `(ō − p̄)/(0.5(ō + p̄))` | `|FB| ≤ 0.3` |
| NMSE, with systematic split | `⟨(o−p)²⟩/(ō·p̄)`; `NMSE_s = 4FB²/(4−FB²)`, `NMSE_u = NMSE − NMSE_s` | `NMSE ≤ 4`. The split is the conceptual payoff: **DA should collapse the systematic part; the unsystematic part is chaos and is irreducible.** |

Set `W` from the truth's own sampling uncertainty (block-bootstrap of the
truth series, `W = σ_u/√N_eff`) — the twin-experiment analogue of VDI's
wind-tunnel repeatability allowance, and it turns `q` into an
"indistinguishable within sampling error" test. Skip the log-based MG/VG
variants for velocity (blow up in recirculation zones near `|U| = 0`).

### 4.2 Distributional comparison: Wasserstein distance at probes

At each sensor/probe (and optionally pooled over statistically homogeneous
regions only), compare velocity PDFs via the **1-D Wasserstein distance** —
closed form via sorting, no binning, no bandwidth, satisfies the metric
axioms (Johnson, Wu & Ihme 2017; equal to the Roy–Oberkampf "area validation
metric" and the 2024 *Building & Environment* "overall area metric" proposed
precisely as the LES-appropriate replacement for hit rate/FAC2):

- `W₁ = ∫|F_ens − F_truth| dx` ≈ mean |sorted-sample difference|
  (`scipy.stats.wasserstein_distance`); `W₂` from a common quantile grid.
- **Normalize by the truth's σ** so `W₂ = 0.5` reads "differs at the level of
  half a standard deviation"; guideline `≲ 0.25σ` good, `≲ 0.5σ` acceptable —
  but calibrate against the truth's self-distance floor (two independent
  halves of the truth record).
- Compute **both** truth-vs-pooled-ensemble (does the ensemble cover the right
  statistics?) and truth-vs-each-member averaged (is a typical member right?).
  Their difference *is* the ensemble spread in statistics space. Note the
  pooled version conflates ensemble spread with turbulent variability — it is
  a physics check, not a calibration check; the calibration layer (§4.5/§6)
  covers that side.
- Q–Q comparisons come free from the same sorts: slope = variance bias,
  intercept = mean bias, tail departure = intermittency error.
- Skip KL/JS (binning-dependent, not a metric, undefined on disjoint support)
  and KS p-values (invalid under autocorrelation; at most report the KS `D`
  statistic descriptively).

### 4.3 Spectral / scale-by-scale

The family that catches over-smoothed, over-diffusive, or
surrogate-collapsed flows that every mean-field metric happily passes —
non-negotiable for claiming the assimilated flow is *turbulence*, not a
blurred mean.

- **Which spectrum:** frequency spectra at probes (Welch, Hann window,
  identical `nperseg`/`fs` for truth and every member — otherwise you compare
  smoothing, not physics). Wavenumber spectra only along genuinely homogeneous
  directions, i.e. above the canopy; do **not** convert between the two via
  Taylor's hypothesis inside a canopy (it fails there).
- **Convention:** premultiplied `f·E(f)/σ²` vs `log f` (normalized
  `f* = fH/U_ref`), log–log; inertial guide slope is −5/3 raw, **−2/3
  premultiplied**; expect ~2 decades of inertial range above the canopy, ~1
  inside.
- **Error metrics:** log-spectral distance
  `LSD = √(mean_k [10·log₁₀(E_t/E_m)]²)` (weights every decade equally —
  pointwise L² over-weights the energy-containing scales, which is exactly why
  blurred fields score deceptively well on RMSE), plus band-resolved relative
  energy error over three bands (energy-containing / inertial / near-cutoff:
  more interpretable and directly actionable). Fit the inertial slope but
  compare to the *truth's fitted slope*, not to the theoretical −5/3.
- **Truncate at `k < k_max/4`** — the last ~4–8 grid spacings of the spectrum
  are numerics, not physics; cross-resolution comparisons must restrict to the
  common resolved band. Detrend/de-mean records; bin-average in log-spaced
  bins; exclude spin-up.
- LES quality indices (Pope's 80 %-resolved-TKE, Celik's LES_IQ) are
  *resolution screens for a single run*, not comparison metrics — and the
  ABL-LES literature shows the naive γ > 0.8 threshold passes clearly
  under-resolved runs (calibrated thresholds are γ ≈ 0.97–0.985). Keep at most
  as a weak per-member gate; the better structural check is §4.4.

### 4.4 Two-point structure

- **Two-point correlation** `B_uu(r) = ⟨u′(x)u′(x+r)⟩/σ²` via FFT
  (Wiener–Khinchin) along horizontal directions at 2–4 z-levels. Unusually
  cheap in snapshots (≥5 snapshots ≥1 eddy-turnover apart suffice — exactly
  the limited-snapshot regime of these runs) and, per the ABL-LES
  grid-resolution literature, a more reliable structural diagnostic than
  Pope/Celik indices.
- **Scalar reductions:** integral length scale `L_int = ∫₀^{r₀} B dr`
  (integrate to first zero-crossing only), and the headline ratio
  `L_int^model / L_int^truth` — target 1; > 1 means structures too large /
  flow too diffusive, the classic over-damping signature. Check
  `B(L_domain/2) ≈ 0` before trusting `L_int` (periodic-domain contamination).
- **Third-order structure function** `S₃(r) = ⟨δu_∥(r)³⟩`, compared as
  `S₃^model/S₃^truth` — odd-order, so it tests cascade direction and increment
  skewness and **cannot be faked by a symmetric-noise or Gaussianized field**;
  independent of the spectrum in a way `S₂` is not (`S₂` ↔ spectrum are
  Fourier duals — skip `S₂` if you have spectra). Never compare against the
  theoretical −4/5 constant (isotropy/homogeneity fail in a canopy); ignore
  `r ≲ 4Δ`. Velocity-increment PDFs `δu(r)` at a few separations are the
  companion check for small-scale intermittency.
- **Mean-flow topology proxy:** volume, centroid, and extent of the
  reverse-flow region (`⟨u⟩ < 0`) — a nearly free scalar that captures the
  pattern-level question urban practitioners actually ask (is the canyon
  vortex in the right place?). Full POD comparison is deliberately excluded:
  per-member 3-D POD is prohibitively expensive, and mode-wise comparison is
  fragile; if ever needed, restrict to eigenvalue spectra and
  principal-angle subspace distances on 2-D planes.

### 4.5 Scoring statistics as an ensemble

For each member and window, reduce sensor series to statistics `T_m` (window
mean, variance/TKE, integral time scale, spectral band energy, direction
mean); truth gives `T*`. Then apply the full calibration toolkit of §6
(fair CRPS, z-score, rank, coverage) **to `{T_m}` vs `T*`**. This is the
scientifically correct verification object — the parameters being estimated
(inflow angle, magnitude, Cs, roughness) act on turbulence *statistics*, so
statistics are the identifiable quantities. Guard: the window must be ≫ the
integral time scale, else member-to-member spread in `T` is dominated by
sampling noise rather than parameter uncertainty — report the ratio
(across-member spread of `T`) / (within-member block-bootstrap std of `T`);
if it is not ≫ 1, that statistic is unidentifiable at this window length.

## 5. Layer D — ESMDA-specific diagnostics

These need artifacts not currently persisted: per-iteration parameter
ensembles, and observations + predicted observations `g(θ_m)` per iteration
(the filtering side's `CycleDiagnostics` is the in-repo precedent). The
storage cost is trivial (`Nd ≈ 180`/window, `M ≤ 100`); the plumbing change in
`run_esmda.py`/the smoother is the actual work.

- **Normalized data mismatch vs the χ² target** (the canonical ESMDA
  diagnostic; Emerick & Reynolds, Evensen):
  `O_N(θ_m) = (1/2N_d)·(d − g(θ_m))ᵀ C_D⁻¹ (d − g(θ_m))` per member per
  iteration, with the *un-inflated* `C_D`. Posterior target `E[O_N] = ½`,
  band `½ ± 3/√(2N_d)`. `O_N ≫ ½` at convergence = under-fitting (model
  error, unidentifiable parameters, `C_D` too small); `O_N ≪ ½` =
  **fitting the observation noise** — over-aggressive schedule / no
  localization, strongly associated with collapse. This asymmetric check is
  the reason to compute it: no RMSE can distinguish "great match" from
  "matched the noise". Plot per-member boxes vs iteration; healthy = monotone
  decrease to a tight cluster near ½ — and watch the *spread* of `O_N`
  across members: its collapse to ~0 is the pathology even when the median
  looks fine.
- **Innovation consistency**: per-obs
  `z_j = (y_j − ḡ_j)/√((1+1/M)·var_m(g_j) + σ_D,j²)` on the *prior*
  ensemble → should be ~N(0,1); histogram + per-sensor (mean, std) +
  `χ²_norm = mean(z²) → 1`. Localizes failure to specific sensors and
  detects mis-specified `C_D`. **Representativeness error is not optional
  here**: a point sensor in turbulent LES sees sub-filter variability the
  model cannot reproduce; if `C_D` omits it, every χ²-type target reads as
  assimilation failure when it is really `C_D` mis-specification. (Desroziers
  residual diagnostics can estimate the needed inflation but are noisy at
  ~6 sensors — direction-only.)
- **Collapse measures**: per-parameter contraction (§3) plus the
  participation ratio `N_eff = (Σλ)²/Σλ²` of the anomaly matrix eigenvalues,
  for parameters *and* predicted obs — `N_eff → 1` means the ensemble has
  collapsed onto a line. Track `σ^(l)/σ^(0)` across iterations; a step-change
  collapse flags a bad `α_l`.
- **Duplicate-member guard** (repo-specific, near-free): the pypalm
  divergence/resampling policy can duplicate members, silently inflating
  nominal `M` while every fair correction and spread estimate assumes
  distinct members. Report `n_unique/M` and diverged/resampled counts per
  window in `run_summary.yaml`; use `n_unique` in the `M(M−1)` corrections.
- **Held-out-sensor predictive check**: the highest-evidentiary-value item in
  this document; a config change (sensor split for barcelona) plus reusing
  the existing sensor machinery on the withheld set.

## 6. Layer C — probabilistic calibration metrics

### 6.1 Fixes to existing code (do these first)

1. **Biased CRPS / energy-score estimators.** Both `per_knot_crps`
   (`src/pyurbanair/utils/da_metrics.py`) and `_energy_score`
   (`scripts/esmda/_esmda_common.py`) take the pairwise mean over all `M²`
   pairs (including the zero diagonal). The fair estimator divides the
   pairwise sum by `M(M−1)` (Ferro 2014). The biased form's optimum is an
   under-dispersed ensemble — it *rewards collapse*, so using it to certify
   an ESMDA posterior is circular. ~2 % shift at M=50, but it flips the
   incentive direction, and it makes scores comparable across different
   member counts (relevant when resampling changes effective M).
2. **Spread–skill formulation.** The spread side must average *variances*
   then take the root (`√(mean var_m)`); the existing summary averages
   standard deviations, which biases spread low (Jensen) and fakes
   under-dispersion. Include the Fortin `√((M+1)/M)` factor:
   `SSR = √((M+1)/M)·SPREAD/RMSE`, target 1; `< 1` = overconfident. In
   observation space compare `√(spread² + σ_o²)` to RMSE-vs-obs.

### 6.2 The metric set

| Metric | Notes |
|---|---|
| **Fair CRPS + CRPSS vs prior** | Per parameter knot, per sensor statistic (§4.5), per mean-field cell. `CRPSS = 1 − CRPS_post/CRPS_prior`; > 0 = assimilation helped. CRPS is `W₁` between ensemble CDF and a Dirac at truth — unifies naturally with §4.2. |
| Energy score (fair) | Already exists for sensor vectors; apply the fair fix. Known weakness: dominated by mean errors, weak discrimination of variance/dependence structure — never use alone. |
| **Variogram score** `VS_{0.5}` | `Σ_{ij} w_ij(|y_i−y_j|^p − mean_m|x_{m,i}−x_{m,j}|^p)²` over sensor pairs, `w = 1/distance`. The only score here that tests *spatial dependence* between sensors — the physically meaningful multivariate structure for wakes. Blind to uniform bias → always pair with CRPS. Trivial cost at ~6 sensors. |
| Rank histogram / PIT | On held-out sensors and on statistics (§4.5), prior and posterior panels. U = under-dispersed, dome = over-dispersed, slope = bias (Hamill 2001). Needs ≳ 10·(M+1) *independent* pooled samples — sub-sample by `T_int`, prefer coarse 10-bin PIT at M=50, randomize ties. |
| Spread–skill (corrected) + binned reliability | Global SSR plus ~10 equipopulated spread-bins of RMSE-vs-spread (the version that tests whether the ensemble *knows when* it is uncertain). Finite-M attenuates the fitted slope toward 0 — plot a same-M synthetic calibrated reference as the target line instead of 1.0. |
| Honesty caveat | Flat rank histogram ≠ reliable and SSR = 1 ≠ reliable (Dirkson et al. 2025: individually insufficient). Present D-layer figures as a *set*, never one alone. |

Deprioritized: Brier/reliability diagrams (thresholds aren't the scientific
question here; 1/M probability resolution is coarse at M=50), Mahalanobis χ²
on parameters (answered more robustly by the z-score bundle),
Desroziers iteration (too few sensors for stable estimates).

## 7. Figure catalogue

Conventions applying to every figure (largely already encoded in
`scripts/figspec/style.py` and `docs/archive/figure_specs.md` — reuse, don't
reinvent): truth black; prior grey; posterior teal; ensemble bands as nested
quantiles (5–95 % + 25–75 %), not ±σ, unless near-Gaussian (parameters);
shared axis limits and one shared color `Normalize` across any prior/posterior
pair (unshared scales silently destroy the comparison); solid cells masked
grey with footprint contours; lengths as `z/H`, velocities as `u/U_ref`,
stresses as `/U_ref²`; window boundaries as dashed verticals.

**Parameter space**

- **P1 — prior vs posterior marginals with truth line** (violins/box+strip per
  parameter, physical units, truth as dashed line; annotate z-score and %
  error reduction). Y-limits must include the prior — autoscaling to the
  posterior hides the contraction, which is the point.
- **P2 — per-iteration trajectories** (x = ESMDA iteration, 0 = prior; thin
  member lines + mean ± band + truth). Diagnoses convergence stability and
  premature collapse. If resampling broke member identity, use per-iteration
  boxes and say so.
- **P3 — corner plot** (lower triangle: member scatter prior grey/posterior
  teal + 2-D KDE contours at the 39.3 %/86.5 % 2-D-sigma levels; diagonal:
  marginals; truth crosshair; annotate Pearson r). Only worth it for ≥3
  parameters or a suspected trade-off; for 2, one scatter panel with marginal
  histograms.

**Statistical state**

- **S1 — vertical profiles at stations** *(the canonical LES-validation
  figure)*: rows = quantity (`ū/U_ref`, `w̄/U_ref`, `√u′²/U_ref`,
  `⟨u′w′⟩/U_ref²` or TKE), columns = stations straddling a street canyon
  (upstream / in-canyon / wake / recovery); quantity on x, `z/H` on y;
  truth line + posterior band + prior band overlaid; roof-height line at
  `z/H = 1`; inset plan-view locating the stations. Add a first-half vs
  second-half average check — reviewers will ask whether `T` was long enough.
- **S2 — velocity PDFs at probes** (log y-axis — tails are the content;
  shared bin edges; truth line vs member-PDF envelopes; annotate
  skewness/kurtosis and sample counts).
- **S3 — Q–Q plots** (shared quantile grid, equal aspect, 1:1 line; caption
  states explicitly: quantiles, not instantaneous values, because the flow is
  chaotic — pre-empts the standard misreading).
- **S4 — energy spectra overlay** (premultiplied, log–log, truth vs posterior
  median + envelope vs prior envelope; short −5/3 (or −2/3) guide segment,
  *not* drawn through the data; dotted vertical line at the filter/grid
  cutoff — marking it defuses the "spectrum decays too fast" objection).
- **S5 — sensor time series with quantile fan + observations with ±σ_o error
  bars**, window boundaries marked, **assimilated and held-out sensors in
  clearly labeled separate column groups** — the held-out panel is the
  strongest anti-overfitting evidence in the paper.

**Field level**

- **F1 — time-averaged slice comparison**: rows = the three standard planes
  (x–z centreline, x–y in-canopy `z/H ≈ 0.1`, x–y above canopy
  `z/H ≈ 1.25`); columns = truth | prior mean | posterior mean | posterior −
  truth. First three columns share one norm and colorbar; the difference gets
  a symmetric diverging norm centred at 0 with its own colorbar. Annotate the
  averaging window (`T·U_ref/H`). **Never instantaneous** — a snapshot
  difference map manufactures a failure that does not exist.
- **F2 — ensemble-spread maps**: prior σ | posterior σ (shared norm) | ratio
  σ_post/σ_prior, computed on time-averaged fields, with **sensor positions
  overlaid** — spread collapsing around sensors while staying high in wakes
  is the observability story in one image.
- **F3 — predicted vs observed scatter**: equal aspect, 1:1 plus FAC2 lines,
  prior and posterior clouds, metric box (q, FAC2, FB, NMSE, R for both);
  FAC2 lines only for positive quantities (see §4.1 caveat).
- **F4 — hit-rate plan-view map** (optional): sensors on the footprint map
  colored by hit fraction, prior vs posterior, threshold ring at q = 0.66.
  Cheap and reads as domain fluency to a wind-engineering audience.

**DA diagnostics**

- **D1 — rank histograms** (prior | posterior, held-out sensors, uniform line
  + binomial consistency band).
- **D2 — spread vs error** (time-series version plus the binned-reliability
  version with the same-M calibrated reference line).
- **D3 — data-mismatch decay**: per-member `O_N` boxes vs iteration, log y if
  needed, horizontal target band at ½. Cheapest high-value figure in the
  set, and the one the ESMDA literature conditions readers to expect — its
  absence looks evasive.
- **D4 — innovation z-histogram vs N(0,1)** + normal Q–Q companion (more
  tail-sensitive), annotated with χ²_norm.

**Ranked by persuasiveness per implementation cost** (synthesis of the three
research rankings): S1 profiles → P1 marginals → S5 held-out sensor fans →
F1 averaged-slice comparison → D3 mismatch decay → F2 spread maps → S4
spectra → D1+D2 calibration pair; then F3, P3, S2/S3, D4, F4.

The three mistakes most likely to sink the whole suite, in order:
(1) instantaneous field comparisons; (2) unshared color scales / axis limits
across prior-posterior panels; (3) evaluating only on assimilated sensors.

## 8. Prioritized implementation roadmap

**Phase 0 — correctness fixes (hours; do regardless of everything else)**
1. Fair CRPS/energy-score estimators (`M(M−1)`), + CRPSS vs prior (§6.1).
2. Spread–skill: average variances, `√((M+1)/M)` factor (§6.1).
3. `n_unique`/duplicate-member counter into `run_summary.yaml` (§5).

**Phase 1 — pure post-processing, no run changes (days)**
4. Parameter bundle: z-scores, coverage/PIT, contraction ratios,
   prior/posterior correlation matrices (§3) — arrays already loaded by
   `compute_esmda_metrics.py`.
5. Statistics-space sensor scoring (§4.5): window mean/variance/TKE per
   member via the existing `ensemble_sensor_series`, scored with fair
   CRPS/z/rank; plus σ-normalized Wasserstein + Q–Q at sensors (§4.2), with
   the truth self-distance floor.
6. Mean-field layer (§4.1): streaming time-mean + Reynolds-stress
   accumulation over `windows/window_*_posterior_state.nc` (extend the
   `_streaming_state_summary` pattern; centre-interpolate staggered
   components first); hit rate, FAC2, FB, NMSE with systematic split.
7. Figures P1, P2, S1, S5, F1, F2, S2/S3 from the same artifacts.

**Phase 2 — persist observation-space arrays (small runner/smoother change,
big diagnostic payoff)**
8. Save `window_obs` and per-iteration `g(θ_m)` (+ per-iteration parameter
   ensembles) — mirrors the filtering side's `CycleDiagnostics`.
9. Then: `O_N` vs ½ per iteration (D3), innovation z/χ² (D4),
   contraction-vs-achievable `r_i`, SNR/sensitivity/DFS table, variogram
   score, rank histograms + binned spread–skill on obs space.

**Phase 3 — run-configuration upgrades**
10. Validation sensors for `barcelona` (held-out scoring/figures everywhere).
11. Spectra & two-point layer (§4.3–4.4): needs denser probe sampling and/or
    a few full-field snapshots ≥1 eddy-turnover apart (cross-check output
    cadence first); then Welch spectra + LSD/band errors, `B(r)`, `L_int`
    ratio, `S₃` ratio, reverse-flow-region scalars.
12. Optional: `save_prior_state=true` runs for prior-state field figures;
    representativeness-error estimate for `C_D` (block-bootstrap point-sensor
    variability) feeding §5's χ² targets.

## 9. Key references

- Chang & Hanna (2004), *Air quality model performance evaluation* — FB/NMSE
  /FAC2, the ~20 %/60 % irreducible-scatter floor, regime averaging.
- VDI 3783/9 & COST 732 — hit rate `q` (D = 0.25, q ≥ 0.66) for velocity
  validation; model-evaluation protocol.
- Johnson, Wu & Ihme (2017), arXiv:1702.05539 — Wasserstein metric for LES
  assessment; also arXiv:1801.03046.
- *Building & Environment* (2024), doi:10.1016/j.buildenv.2024.112285 —
  "overall area metric" as an LES-appropriate replacement for q/FAC2; closest
  paper in the literature to this document's goal.
- arXiv:2502.13672 (Phys. Fluids 2025) — urban-roughness LES validation:
  profile/spectra figure conventions, HR ≥ 0.66 / FAC2 ≥ 0.3, W values.
- Pope (2004); Celik et al. (2005); Boundary-Layer Meteorol.
  grid-resolution study (d-nb.info/1212785126) — LES quality indices and why
  two-point correlations beat them.
- Ferro (2014) — fair scores for finite ensembles; Gneiting & Raftery
  (2007) — proper scoring rules / energy score; Scheuerer & Hamill (2015) —
  variogram score.
- Fortin et al. (2014) — spread–skill `√((M+1)/M)`; Hamill (2001) — rank-
  histogram interpretation; Jolliffe & Primo (2008) — flatness decomposition;
  Dirkson et al. (QJRMS 2025) — insufficiency of single reliability metrics.
- Emerick & Reynolds (2013) — ES-MDA; Evensen (2019) — iterative-smoother
  convergence practice; Desroziers et al. (2005) — innovation-based error
  diagnostics.
- Hersbach (2000) — CRPS decomposition and CRPS ↔ `W₁` connection.
