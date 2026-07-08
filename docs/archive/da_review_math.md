# Data-Assimilation Library Review — Part 2: Mathematics & Numerics

Scope: the ES-MDA formulation in `smoothing/esmda.py`, the local-analysis localization in
`localization/` (following Vossepoel et al. 2025, MWR-D-24-0269.1), the online SVD/KL
reduction in `reduction.py`, and the observation operators. Verified conventions against
Emerick & Reynolds (2013) for ES-MDA and against the paper's Eqs. (6)–(10) for the
localization.

**What checks out.** Before the issues: the core math is sound and carefully done.

* The global ES-MDA update is the standard stochastic form: fresh perturbations
  `√α · C_D^{1/2} Z` per iteration, `C_MD`/`C_DD` from anomalies with `1/(N_e−1)`,
  gain solve against `C_DD + α C_D`, default `α = num_steps` so `Σ 1/α_k = 1`
  (Emerick & Reynolds 2013). Signs, shapes, and the per-step re-forecast from the
  current IC/params (the "iterated" variant) are all correct.
* The local analysis is a faithful implementation of the paper: inflation multiplies the
  perturbation (std), so error *variance* scales by `E_inf²`; the taper
  (`taper_inflation`) hits exactly `E_max` at the truncation distance and excludes
  beyond it (Eqs. 9–10 verified analytically: at `d = trunc`,
  `exp(((1−β)·trunc / b)²) = exp(log E_max) = E_max`); the `max_inflation = 1` edge case
  correctly degenerates to a hard cutoff (b → ∞ → taper ≡ 1).
* The exclusion trick — zeroing rows/columns of `C_DD + α·diag(C_D·E_inf²)` and putting 1
  on excluded diagonals, plus zeroing the corresponding innovation and `C_MD` entries —
  yields the *exact* active-submatrix solution while staying shape-stable for `vmap`.
  Nice construction, and correctly reasoned in the comments.
* One `Z` realization shared by all rows within one localized step, and by the masked
  (global-update) rows: one consistent ESMDA step, matching the reference EnKF_MS
  semantics. Block grouping via per-observation *min* inflation over the block matches
  the reference ("active if active for any member; taper driven by the strongest
  correlation").
* The correlation localization computes the exact sample correlation (consistent
  `N_e−1` in covariance and stds), clips to [−1, 1], treats zero-variance rows as
  excluded, and defaults `ρ_t = 3/√N_e` per the paper's Eq. (6).
* The reduction's encode/decode pair is self-consistent: `ξ = Σ_r⁻¹ Φ_rᵀ (u − ū)`,
  increment decode `ΦΣ Δξ` preserves each member's projection residual, and the claimed
  full-rank equivalence with the full-space update is real (the full-space update is
  confined to the ensemble anomaly span; the IC-source basis spans exactly that space) —
  and is pinned by `test_full_rank_ic_source_matches_full_space_update`.

---

## 1. Errors / mathematically questionable

### 1.1 `final_time_smoothing` uses the observations twice — `smoothing/esmda.py:895-927`

The MDA loop consumes the full likelihood: `num_steps` updates at `α = num_steps`,
`Σ 1/α_k = 1`. The post-loop trajectory smoothing then applies **one more full-weight
update (`α = 1`) with the same observation vector** to an ensemble that is already
conditioned on those observations (the trajectory is forecast from the analyzed IC and
analyzed parameters). In the linear-Gaussian limit this is conditioning on the data
twice: total likelihood weight 2, so the returned trajectory ensemble is overconfident —
its spread underestimates the posterior spread, and its mean is pulled too hard toward
the observations.

The docstring's justification ("the schedule already consumed the likelihood for the IC
and parameters; this is a single standard Kalman analysis of the trajectory") identifies
the issue but draws the wrong conclusion: the fact that the likelihood was consumed is
precisely why the forecast ensemble entering this step is *not* a prior for the same
data. The frames at t>0 were never directly updated, but they are conditioned through
the dynamics — that information is in the ensemble, not lost.

Options, in decreasing rigor:

1. **Fold the trajectory into the MDA schedule**: include the (reduced) trajectory
   coefficients in the augmented vector during the regular steps, so the same
   `Σ 1/α = 1` schedule covers them. Cost: the window-snapshot flatten per step (already
   implemented for the `window_snapshots` basis source).
2. **Keep it, but re-derive the weight**: if the goal is only a better point estimate of
   the trajectory (not calibrated spread), the current step is a pragmatic "analysis of
   the final forecast" and often looks good in RMSE. Then say exactly that in the
   docstring/config docs: *the smoothed trajectory's ensemble spread is not a posterior
   spread and must not be used for uncertainty quantification*.
3. If the spread matters, at minimum inflate back (e.g. relaxation-to-prior-spread on
   the trajectory update) — ad hoc, but bounds the damage.

This is the most important finding in this report.

### 1.2 Element-wise `sqrt(C_D)` silently assumes a diagonal covariance — `smoothing/esmda.py:57, 181`

`C_D_sqrt = jnp.sqrt(C_D)` and `perturbed_obs = obs + √α (C_D_sqrt @ Z)` produce
perturbations with covariance `α·C_D` **only when `C_D` is diagonal**. For any correlated
observation-error matrix the element-wise sqrt is not a matrix square root, and the
sampled perturbations have the wrong covariance — no error, no warning. The localized
path additionally hard-assumes diagonality (`jnp.diag(C_D)` as *the* variances,
per-observation inflation of a scalar variance).

Current callers pass `jnp.diag(σ²·1)` (`run_esmda.py:623`), so nothing is wrong *today*.
But the API type is "matrix", so the contract must be enforced: either (a) validate
diagonality at construction and document it, or (b) accept a 1-D variance vector (the
honest API — also saves the O(N_d²) storage), or (c) support full covariances properly
via `jnp.linalg.cholesky` on the global path (localization genuinely requires diagonal —
per-observation inflation of correlated errors is not defined in the paper's scheme —
so (b) is the recommendation).

### 1.3 No validation of the MDA consistency condition `Σ 1/α_k = 1` — `smoothing/esmda.py:55`

`alpha` is a free scalar applied at every step. The default (`α = num_steps`) is
consistent, but any user-supplied value with `num_steps / α ≠ 1` silently breaks the
ES-MDA identity: the posterior corresponds to a likelihood tempered by `num_steps/α` —
over-conditioned if `α < num_steps`, under-conditioned if larger. In the linear-Gaussian
case this is *exactly* equivalent to scaling the observation error, i.e. a different
inference problem. Either validate (`abs(num_steps/alpha - 1) < tol` → error or loud
warning) or generalize to a per-step schedule with normalization (see §3.1). Note the
final-time smoothing step deliberately passes `α = 1` through the same code path, so the
check belongs where `self.alpha` is set, not inside `_compute_kalman_update`.

### 1.4 Block grouping mis-co-locates staggered-grid variables — `smoothing/esmda.py:668-683`

Mathematical face of the code report's §1.7: the "grid block" joint update (paper
sec. 3b) is only meaningful when the block's rows are physically co-located. Sharing the
within-variable flat index across variables co-locates `u/v/w` on a collocated grid
(pylbm) but pairs *different physical cells* on staggered grids (uDALES: `u` on `xm`,
`v` on `ym`; PALM: `xu`/`yv`), where per-variable sizes and positions differ. The block
minimum then mixes inflation vectors of unrelated locations, and the joint transition
updates non-co-located quantities with one another's observation selections. Restrict to
collocated states (validate and raise) or derive block ids from rounded physical
coordinates.

### 1.5 Singular / ill-conditioned analysis systems fail silently as NaN — `smoothing/esmda.py:186-189`

(Numerics face of code report §1.1.) `C_DD` is rank ≤ N_e−1; positive definiteness of
`C_DD + α C_D` rests entirely on every observation-error variance being strictly
positive. A zero (or denormal) variance — e.g. a config typo, or a "perfect" synthetic
sensor — makes the system singular in the directions outside the ensemble span, JAX
returns NaN without raising, and the NaNs propagate into the next forecast. Validate
`C_D` variances > 0 at construction and check the solve output for finiteness.

### 1.6 Interval binning drops empty bins, breaking absolute alignment — `observation_operator.py:283-301`

`unique_bins` enumerates *populated* bins, so element `k` of the predicted-observation
vector is "the k-th non-empty interval", not "interval k". The real observations the
smoother is given must be aligned to the same convention; if the sensor data are binned
by absolute interval index (the natural convention for real data) and the model output
ever skips an interval (irregular output cadence, a crashed segment, a window shorter
than expected), predicted and real observations shift against each other — a silent
misalignment of the entire innovation vector, which the update will happily absorb as
signal. Fix: bin against `range(num_intervals)` computed from the window span and raise
on empty bins (or emit NaN and combine with missing-data masking, code report §3.1).

### 1.7 Half-cell extrapolation at grid edges — `interpolation.py:55-75` (minor, by design)

Sensors up to half a median cell outside a variable's native coordinate range are
accepted, with clipped indices producing weights outside [0,1] — i.e. *linear
extrapolation* from the edge cell pair. Deliberate and documented (staggered-grid
support), and bounded to half a cell, so fine — but note the median spacing rule makes
the accepted margin depend on interior grid stretching, not on the local edge spacing.
On strongly stretched vertical grids the allowed z-extrapolation at the wall could be
much larger than half the *local* cell. Using the local edge spacing
(`spacing[0]`/`spacing[-1]` per side) would match intent better.

---

## 2. Improvements (no new features)

### 2.1 Solve the analysis system in a numerically stronger and cheaper form

`C_DD + α C_D` is symmetric positive definite by construction; use a Cholesky solve
(`jax.scipy.linalg.cho_factor/cho_solve`) instead of generic LU — ~2× faster, better
error behavior, and failure (non-SPD) is *detectable* (NaN in the factor) rather than
silent. For N_d meaningfully larger than N_e, prefer the subspace form: with
`S = pred_obs_dev/√(N_e−1)` (N_d × N_e), apply Woodbury to
`(α C_D + S Sᵀ)⁻¹` so the dense solve is N_e × N_e instead of N_d × N_d — the standard
large-N_d ES-MDA implementation (Emerick's TSVD variant). With diagonal `C_D` (§1.2)
this needs no N_d × N_d matrix at all, dropping the analysis cost from O(N_d³) to
O(N_d N_e² + N_e³).

### 2.2 Run the analysis step in float64

JAX defaults to float32 and nothing in the library enables x64 (verified:
`jax.config.jax_enable_x64` is `False` in the pixi env). The forward model in float32 is
fine, but the analysis solves and the SVD are exactly where float32 hurts: `C_DD` formed
as a Gram product squares the condition number, and correlation/inflation pipelines
amplify rounding near the truncation threshold. Cheapest robust option: cast
`pred_obs_dev`/`aug_dev`/`C_D` to float64 inside `_compute_kalman_update` /
`localized_update` / `OnlineStateReduction.fit` and cast the increment back — the arrays
are small relative to the state, and it decouples analysis precision from the global JAX
config.

### 2.3 Center (and optionally orthogonalize) the observation perturbations

The stochastic-EnKF perturbation matrix `E = √α C_D^{1/2} Z` has a nonzero sample mean of
order `1/√N_e`, which biases the analysis mean; subtracting the row mean of `Z` (and, if
one wants the exact sample covariance, scaling to match `α C_D`) is the standard,
essentially free correction (Evensen 2004). At N_e = 50–100 this is a visible bias
reduction per step, compounded over `num_steps × num_windows` updates.

### 2.4 Deduplicate block solves in the local analysis

(Math-side note of code report §2's efficiency item.) The block-grouped inflation rows
are identical within a block, so the per-row transitions are identical; solving once per
block is *exactly* equivalent, not an approximation. Worth stating here because it means
the fix has zero methodological risk.

### 2.5 Document the localization/likelihood interaction

Per-row observation-error inflation means different rows assimilate *different tempered
likelihoods*; the ESMDA `Σ 1/α = 1` argument holds per row only in the linear-Gaussian
limit and with the same inflation pattern each step. Correlation localization re-selects
observations *every step* from the current (shrinking) ensemble — so the effective data
weight per row over the schedule is not controlled. This is inherent to the paper's
method (not a bug here), but a short note in `localization/base.py` would prevent future
readers from assuming exact MDA consistency under localization. Related: as the ensemble
collapses over MDA steps, sample correlations *increase* in magnitude spuriously is not
the failure mode — rather spread shrinks and `ρ_t = 3/√N_e` stays fixed; consider noting
that adaptive thresholds per step are an open choice.

### 2.6 Clarify whitened-coefficient conditioning in the reduction

`encode` divides by retained singular values; the rank rule keeps modes down to
`s > s₀·1e-12`, so with `energy_fraction = 1.0` the last coefficients can be scaled by
~1e12. In exact arithmetic the Kalman update is invariant to this diagonal scaling (it
is a linear change of variables and `decode_increment` multiplies it back), but in
float32 (§2.2) the dynamic range is exhausted. A relative floor tied to the energy
criterion (e.g. drop modes with `s < s₀·√ε_machine`) or simply documenting "use
`energy_fraction < 1` or `max_rank`" closes it.

### 2.7 Reorganize the taper docs around one formula

`taper_inflation` is the mathematical heart of both strategies and its correctness
argument currently lives across three docstrings. A short derivation note (b from Eq. 10;
endpoint values; the `E_max = 1` limit) in `localization/base.py` — essentially what §0
of this report verifies — would make the next audit trivial.

---

## 3. Recommended methodological extensions

Ordered by expected value for the urban-flow use case:

1. **Per-step α schedules with validated normalization.** Geometrically decreasing α
   (large first step, small last) is the best-documented cheap improvement over uniform
   ES-MDA for nonlinear problems (Emerick's ES-MDA-GEO; Le & Reynolds). Accept
   `alpha: float | Sequence[float]`, validate `Σ 1/α_k = 1` (optionally auto-normalize),
   keep the default uniform. Pairs with §1.3.
2. **Adaptive α selection.** Choose each α_k from the current data mismatch
   (discrepancy-principle / regularizing-EKI style: Iglesias 2015, Iglesias & Yang 2021):
   α_k set so the tempered update stays within a trust region, iterate until
   `Σ 1/α = 1`. Removes `num_steps` as a tuned hyperparameter — attractive here because
   the forward model is expensive and window conditions vary (some windows are nearly
   linear, some are not).
3. **Temporal-distance localization for time-varying parameters.** The natural analog of
   `DistanceLocalization` in the time dimension: observation interval `k` should mainly
   inform inflow knots near time `k` (advective causality even gives an asymmetry — a
   knot cannot influence observations before it). The whole machinery exists
   (`taper_inflation` against `|t_knot − t_obs|`, `requires_coordinates`-style plumbing);
   only the "coordinates" are 1-D times. This is a physically-motivated complement to
   the correlation strategy, immune to sampling noise, and directly targets the library's
   distinctive object (time-varying inflow parameters).
4. **Innovation-consistency diagnostics.** Per step, test
   `E[dᵀ(C_DD + αC_D)⁻¹d] ≈ N_d` (χ² diagnostic) and per-sensor innovation statistics.
   This is the standard tool for detecting a mis-specified `obs_error_std` — currently a
   hand-tuned config scalar — and for catching the overconfidence that localization/
   final-time smoothing can introduce. Cheap: all quantities are already computed in the
   update.
5. **Covariance inflation / relaxation.** With N_e ~ 50 and repeated windows, sampling
   error systematically deflates spread across windows (each posterior seeds the next
   prior). Relaxation-to-prior-spread (RTPS, Whitaker & Hamill 2012) is a two-line
   post-update step and the standard companion to localization. Multiplicative inflation
   as the simpler fallback.
6. **Schur-product (Gaspari–Cohn) covariance localization as an alternative backend.**
   Element-wise tapering of `C_MD` (and optionally `C_DD`) is O(N_aug·N_d) with *no
   per-row solves* — one global solve serves all rows. It is a different approximation
   than the local analysis (it tapers the gain rather than selecting observations) but
   is dramatically cheaper for full-state updates and is the field's default. Having
   both behind `BaseLocalization` would let you choose per problem size.
7. **Bounded-parameter transforms.** Parameters are updated in native space; Gaussian
   increments can push magnitudes negative or angles out of range (whatever clamping
   happens today lives in the run scripts, which biases the ensemble). A per-parameter
   bijection (log for positive, logit for bounded, identity otherwise) applied at
   flatten/unflatten keeps the Kalman update in an unconstrained space — standard
   practice in reservoir ES-MDA and cheap to add exactly where
   `_flatten_time_varying_params` already sits.
8. **Sensor-bias estimation.** Augment the parameter vector with a per-sensor additive
   bias with a tight prior — the classic treatment of miscalibrated field sensors, and in
   this codebase it is literally "add parameters", which the machinery already supports;
   the extension is the observation operator adding the bias to `pred_obs`.
9. **Iterative ensemble smoothers (IES/LM-EnRML)** (Chen & Oliver 2013; Evensen 2018) as
   an alternative to ES-MDA for strongly nonlinear windows: Gauss–Newton/Levenberg-style
   iterations with step control rather than a fixed tempering schedule. Larger lift;
   worth it only if ES-MDA shows iteration-to-iteration oscillation in practice.
10. **Principled trajectory smoothing** — the rigorous replacement for
    `final_time_smoothing` (§1.1, option 1): carry reduced trajectory coefficients in
    the augmented vector through the MDA schedule, so the full window state estimate is
    conditioned exactly once.
