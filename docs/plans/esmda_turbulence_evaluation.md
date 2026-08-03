# Evaluating ESMDA parameter estimation for turbulent LES: metrics and figures

> **Status: design / research document** (not a maintained reference). Compiled
> 2026-07-29 from a literature survey, cross-checked against what
> `scripts/run_esmda_pipeline.sh` actually produces; slimmed 2026-08-03 to the
> essential metric set (the original comprehensive survey lives in git
> history). It proposes what the metric stage
> (`scripts/esmda/compute_esmda_metrics.py`) and figure stage
> (`scripts/esmda/make_esmda_figures.py`) should grow into. Verify code
> references against the tree before implementing.



## 1. Premise

Parameter estimation for turbulent LES cannot be judged by instantaneous state
agreement: two LES runs with *identical* parameters decorrelate after a
Lyapunov horizon, so pointwise RMSE past it measures chaos, not parameter
quality (Chang & Hanna 2004; Johnson, Wu & Ihme 2017). The remedy is to
**average first, compare second**, and to ask two distinct questions:

1. **Fidelity** — does the posterior reproduce the *statistics* of the true
  flow (mean fields, second moments, spectra)?
2. **Calibration** — is the truth a plausible draw from the posterior? Apply
  probabilistic scores (CRPS, z-scores, rank histograms) **to statistics**,
   never to instantaneous fields.

On top of these: **parameter recovery** (truth known in twin experiments) and
**ESMDA health** (the algorithm's own pathologies: over-contraction, noise
over-fitting).

### Principles (non-negotiable)

- **Never compare instantaneous fields.** Every metric and field figure
operates on time averages or statistics. Order of operations: per member →
time-average over the *same* stationary window as truth → then reduce
across members.
- **Self-distance floor.** Compute each fidelity metric between two
independent halves of the truth run; report scores relative to that floor.
The meaningful criterion is *indistinguishable from truth's own
variability*, not "small".
- **Fair ensemble scores.** Pairwise scores divide by `M(M−1)`, not `M²`
(§6) — the biased form rewards a collapsed ensemble, the exact failure
being tested.
- **Prior vs posterior, always** — ideally as a skill score
`1 − score_post/score_prior`. "Posterior worse than prior" is a real,
previously observed outcome in this repo.
- **Held-out sensors.** Scores on assimilated sensors are circular; skill on
withheld sensors is the strongest single piece of evidence.
`xie_and_castro` defines validation sensors; `barcelona` should too.



## 2. Pipeline constraints

- The full `(ensemble, time, z, y, x)` state exists only in per-window
`windows/window_{w}_posterior_state.nc` (~GBs) — **field metrics must
stream** over members/z-slices (existing patterns:
`_streaming_state_summary`, `_vel_field_4z`; ~2 reader threads is the
DRAM-bound plateau). Parameter ensembles (`prior_params.nc`,
`posterior_params.nc`, `true_params.nc`) are tiny.
- **Cross-grid runs are the default** (e.g. PALM truth → surrogate ensemble):
interpolate truth onto the assim grid for fields, or evaluate both at
shared physical points for sensors. uDALES states are C-grid staggered —
co-locate components at cell centers before forming second moments.
- **Not persisted today:** observations and predicted observations `g(θ_m)`
per ESMDA iteration. §5 needs them; the filtering side's
`CycleDiagnostics` is the in-repo precedent, and the storage cost is
trivial (`Nd ≈ 180`/window).



## 3. Parameter metrics (truth `θ*` known)

Notation: members `θ_m`, prior `b` / posterior `a`, ensemble std `ddof=1`.


| Metric            | Definition                                                               | Read it as                                                                                                                                                                                           |
| ----------------- | ------------------------------------------------------------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Normalized error  | `(θ̄ᵃ − θ*)/σᵇ`; report the existing `rmse_reduction_vs_prior` alongside | Error in units of prior uncertainty — comparable across parameters.                                                                                                                                  |
| **z-score**       | `z = (θ* − θ̄ᵃ)/σᵃ`                                                      | The single most informative scalar. `|z| ≲ 2`: healthy. `|z| > 3` with small `σᵃ`: **over-confident posterior**, the canonical ESMDA failure. Pooled over knots/windows the set should look ~N(0,1). |
| Contraction ratio | `c = σᵃ/σᵇ` per parameter                                                | `c ≪ 1` = data informed the parameter; `c ≈ 1` = unidentifiable. Interpret jointly with the z-score: small `c` **and** large `|z|` = spurious contraction.                                           |
| Fair CRPS + CRPSS | per knot, fair estimator (§6); `CRPSS = 1 − CRPS_post/CRPS_prior`        | > 0 = assimilation helped. Combines accuracy and calibration in one proper score.                                                                                                                    |




## 4. State metrics



### 4.1 Mean fields (the baseline)

Per member and for truth, one streaming pass (accumulate `Σu_i`, `Σu_i u_j`):
time-mean velocity `U_i`, and one second moment — resolved TKE
`k = ½⟨u_i′u_i′⟩` plus the off-diagonal `⟨u′w′⟩` where stations make it
meaningful (`⟨u′w′⟩` discriminates anisotropy that `k` alone cannot; note
stresses are resolved-only).

Score the time-mean velocity with a single standards-based number, the
**hit rate q** (VDI 3783/9 — designed for velocity): a point counts if
relative error ≤ `D = 0.25` **or** absolute error ≤ `W`; acceptance
`q ≥ 0.66`. Set `W` from the truth's own sampling uncertainty
(block-bootstrap, `W = σ_u/√N_eff`), which turns `q` into an
"indistinguishable within sampling error" test. Report prior and posterior.

### 4.2 Sensor statistics as the verification object

For each member and window, reduce sensor series to statistics `T_m` (window
mean, variance or TKE); truth gives `T*`. Score `{T_m}` vs `T*` with fair
CRPS, z-score, and rank — on assimilated **and held-out** sensors, prior and
posterior. This is the scientifically correct object: the estimated
parameters act on turbulence *statistics*, so statistics are the
identifiable quantities. Guard: report (across-member spread of `T`) /
(within-member block-bootstrap std of `T`); if not ≫ 1, that statistic is
unidentifiable at this window length.

### 4.3 One spectral check

The one item that catches an over-smoothed / surrogate-collapsed flow that
passes every mean-field metric. Frequency spectra at probes only (Welch,
Hann window, identical `nperseg`/`fs` for truth and every member),
premultiplied `f·E(f)/σ²` vs `log f`, truncated to the common resolved band
(`k < k_max/4`), spin-up excluded. One scalar: log-spectral distance
`LSD = √(mean_k [10·log₁₀(E_t/E_m)]²)`, truth vs posterior median, with the
truth self-distance floor. Needs denser probe output cadence — check before
implementing. No wavenumber spectra, no Taylor's hypothesis in-canopy.

## 5. ESMDA health

Requires persisting, per iteration: observations, predicted observations
`g(θ_m)`, and parameter ensembles (small plumbing change in
`run_esmda.py`/the smoother, mirroring `CycleDiagnostics`).

- **Normalized data mismatch vs the χ² target** (Emerick & Reynolds;
Evensen): `O_N(θ_m) = (1/2N_d)·(d − g(θ_m))ᵀ C_D⁻¹ (d − g(θ_m))` per
member per iteration, un-inflated `C_D`. Posterior target `E[O_N] = ½`,
band `½ ± 3/√(2N_d)`. `O_N ≫ ½` = under-fitting; `O_N ≪ ½` = **fitting
the observation noise** (over-aggressive schedule / no localization) —
no RMSE can make this distinction. Watch the across-member *spread* of
`O_N` too: its collapse is the pathology even when the median looks fine.
- **Duplicate-member guard** (near-free): the pypalm divergence/resampling
policy can duplicate members. Report `n_unique/M` per window in
`run_summary.yaml`; use `n_unique` in the `M(M−1)` corrections.



## 6. Correctness fixes to existing code (do first, regardless)

1. **Biased CRPS / energy-score estimators.** `crps_ensemble` and
  `_energy_score` (both now in `libs/evaluation/src/evaluation/scores.py`
   after WP0.2; formerly `da_metrics.per_knot_crps` /
   `plotting._crps_ensemble` and `_esmda_common._energy_score`)
   average over all `M²` pairs including
   the zero diagonal; the fair estimator divides the pairwise sum by
   `M(M−1)` (Ferro 2014). The biased form's optimum is a collapsed
   ensemble — using it to certify an ESMDA posterior is circular.
2. **Spread–skill.** Average *variances* then take the root (the current
  mean-of-stds biases spread low), include the Fortin `√((M+1)/M)` factor:
   `SSR = √((M+1)/M)·SPREAD/RMSE`, target 1, `< 1` = overconfident. In
   observation space compare `√(spread² + σ_o²)` to RMSE-vs-obs.



## 7. Figures

Conventions (already encoded in `libs/evaluation/src/evaluation/style.py`
— reuse): truth
black, prior grey, posterior teal; ensemble bands as nested quantiles
(5–95 % + 25–75 %); **shared axis limits and one shared color** `Normalize`
**across every prior/posterior pair**; lengths `z/H`, velocities `u/U_ref`.

1. **P1 — parameter marginals**: prior vs posterior violins/box+strip per
  parameter, truth as dashed line; annotate z-score. Y-limits must include
   the prior — autoscaling to the posterior hides the contraction.
2. **S1 — vertical profiles at stations** *(the canonical LES-validation
  figure)*: rows = `ū/U_ref` and TKE (or `⟨u′w′⟩/U_ref²`), columns =
   stations straddling a canyon (upstream / in-canyon / wake); truth line +
   posterior band + prior band; roof line at `z/H = 1`; inset plan view.
3. **S5 — sensor time series**: quantile fan + observations with ±σ_o bars,
  window boundaries marked, **assimilated and held-out sensors in labeled
   separate columns** — the held-out panel is the strongest
   anti-overfitting evidence.
4. **F1 — time-averaged slice comparison**: 2–3 planes × (truth | prior
  mean | posterior mean | posterior − truth); first three columns share one
   norm, difference gets a symmetric diverging norm. **Never
   instantaneous.**
5. **S4 — energy spectra overlay**: premultiplied, log–log, truth vs
  posterior median + envelope vs prior envelope; guide slope segment not
   drawn through the data; dotted line at the grid cutoff.
6. **D3 — data-mismatch decay**: per-member `O_N` boxes vs iteration,
  target band at ½. The figure the ESMDA literature conditions readers to
   expect — its absence looks evasive.
7. **D1 — rank histogram** of truth within the ensemble, on held-out sensor
  statistics (§4.2), prior | posterior, uniform line + consistency band.
   Per-window statistics are already ~independent samples; pooled counts are
   small, so coarsen to ~10 rank bins rather than M+1.

The three mistakes most likely to sink the suite: (1) instantaneous field
comparisons; (2) unshared color scales across prior/posterior panels;
(3) evaluating only on assimilated sensors.

## 8. Implementation order

1. **Correctness fixes** (§6) + `n_unique` counter — hours.
2. **Pure post-processing**: parameter bundle (§3), sensor-statistics
  scoring (§4.2), streaming mean-field pass + hit rate (§4.1); figures P1,
   S1, S5, F1, D1 — days, no run changes.
3. **Persist obs-space arrays** (§5) → `O_N`/D3.
4. **Run-config upgrades**: validation sensors for `barcelona`; denser probe
  cadence → spectra (§4.3, S4).



## 9. Deliberately excluded (and why)

Recorded so the pruning isn't re-litigated; the full survey is in git
history (pre-2026-08-03 versions of this file).

- **Instantaneous-field RMSE / animations** — chaos, not parameter quality;
keep at most as unheadlined sanity checks.
- **Wasserstein/PDF distances, Q–Q plots** — the CRPS on sensor statistics
covers the distributional question; full-PDF machinery adds little for
the cost.
- **Two-point correlations, integral length scales, structure functions**
`S₃`**, velocity-increment PDFs, POD** — valuable turbulence structure
diagnostics, but the single spectrum check (§4.3) covers the
over-smoothing failure mode at a fraction of the cost.
- **FAC2 / FB / NMSE battery** — hit rate `q` alone is the velocity-appropriate
standard; the rest add numbers, not decisions.
- **Contraction-vs-achievable, sensitivity/SNR/DFS tables, posterior
correlation eigen-analysis** — genuine diagnostics for *why* an update
failed; add back only when debugging an unhealthy run, not for routine
evaluation.
- **Variogram score, binned spread–skill reliability, Jolliffe–Primo
decomposition, Brier scores, Desroziers, innovation z-histograms (D4),
LES quality indices (Pope/Celik)** — either statistically underpowered at
~6 sensors / M=50, or answered more robustly by the retained set.



## 10. Key references

- Chang & Hanna (2004) — irreducible-scatter floor; average-first principle.
- VDI 3783/9 & COST 732 — hit rate `q` (D = 0.25, q ≥ 0.66) for velocity.
- Johnson, Wu & Ihme (2017), arXiv:1702.05539 — single LES realizations must
be replaced by distributions.
- Ferro (2014) — fair scores for finite ensembles; Fortin et al. (2014) —
spread–skill factor; Hamill (2001) — rank-histogram interpretation.
- Emerick & Reynolds (2013) — ES-MDA; Evensen (2019) — iterative-smoother
convergence practice and the `O_N` target.
