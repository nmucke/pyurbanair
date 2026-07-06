# Data-Assimilation Review — Part 3: Why State Estimation Underperforms, and How to Improve It

Companion to `da_review_code.md` and `da_review_math.md`. This report addresses the
observed problem: **the state update adds little or no skill, with or without the online
SVD reduction or the correlation/distance local-analysis localization.** It combines an
analysis grounded in this repo's actual configuration with a literature sweep
(sources and their verification status in §5).

The setup the analysis refers to (Barcelona case, current defaults):

| Quantity | Value |
|---|---|
| State dimension | 224 × 224 × 32 grid × 3 components ≈ **4.8 M DOF** (4 m resolution) |
| Ensemble size | **50** |
| Sensors | **6**, all at z = 3 m in street canyons, ~220–360 m apart |
| Observations per window | 6 sensors × (u,v,w) × 10 intervals = **180**, each a **30 s time-mean** |
| Window | 300 s, warm-started from the previous posterior's final frame |
| MDA | 3 steps, α = 3; obs error std 0.25 m/s |
| Truth vs assimilation model | different solvers (e.g. PALM truth vs LBM/neural surrogate) → real model error |

---

## 1. General analysis: six compounding reasons the state update struggles

None of these is a bug. They compound: each one alone degrades the update; together they
make the *instantaneous full-state* estimation problem, as currently posed, close to
unsolvable — while leaving several reformulations (§3) very solvable.

### 1.1 The information budget: a rank-49 update against a high-dimensional error space

Every ensemble update lives in the span of the 50 forecast anomalies — at most 49
directions in a 4.8 M-dimensional space, before localization. Theory for chaotic systems
makes precise what those 49 directions must cover: in the linear/perfect-model limit the
forecast error covariance provably collapses onto the **unstable–neutral subspace** of
the dynamics, and ensemble filters/smoothers work when the ensemble spans it — its
dimension n₀ is effectively a lower bound on the ensemble size for satisfactory
performance without localization (Carrassi et al. 2022; Bocquet & Carrassi 2017 —
verified claims). For a 3-D turbulent urban canopy at 4 m resolution, the number of
positive Lyapunov exponents is not 50; it plausibly runs to thousands or more (it grows
with Reynolds number and domain size). The ensemble cannot span the growing error
directions, so most of the true error is invisible to the update, and what the update
*does* see is contaminated by sampling noise (§1.3).

Two immediate corollaries for the current code:

* **The SVD reduction cannot fix this** — and it is behaving exactly as designed. The
  online basis is fitted to the *same 50 anomalies* (the `initial_condition` source is
  exactly the ensemble span; `window_snapshots` spans the same members' trajectories).
  Reduction changes the conditioning and cost of the update, not its information
  content. Expecting skill gains from it was never on the table; it is a compression
  tool.
* **Localization increases the effective rank but not the sample quality.** The local
  analysis lets different rows use different linear combinations of the 49 directions,
  which raises the global update rank — but each row's gain is still estimated from 50
  samples, and with N_e = 50 raw sample covariances are significantly wrong *at all
  separation distances* (Ying, Zhang & Anderson 2018). Localization removes far-field
  spurious correlations; it cannot repair the near-field estimate.

### 1.2 Observability: 6 street-level time-averaging sensors cannot determine a 3-D instantaneous field

This is the hardest constraint, and it is physical, not algorithmic. The turbulence
state-estimation literature has mapped it well:

* Reconstruction accuracy decays sharply with distance from the sensing surface;
  estimates from wall data become essentially uncorrelated with the true state away
  from the wall, and the sensitivity of wall observations to high-wavenumber interior
  structures vanishes (JFM, "What is observable from wall data in turbulent channel
  flow?"; arXiv:2011.03711). All six sensors sit at z = 3 m — the bottom of a 128 m
  domain, *inside* street canyons. The bulk of the domain above roof level is close to
  unobservable in the instantaneous sense.
* Full-state reconstruction has a hard sensor-spacing threshold on the order of the
  **Taylor microscale** (arXiv:2011.03711) — metres to tens of metres here, versus
  actual spacing of hundreds of metres. Structures smaller than the sensor spacing are
  not estimable; nudging experiments find synchronization of a coherent structure only
  once observation spacing shrinks to that structure's own length scale (JFM
  square-cylinder nudging).
* Small scales are **not slaved** to large scales in 3-D turbulence: even with the
  energy-containing scales perfectly assimilated, the error at unconstrained scales
  remains at the order of the energy at those scales (Di Leoni, Mazzino & Biferale,
  spectral-nudging study). So even a perfect large-scale update leaves instantaneous
  point-wise RMSE looking poor — part of the "state does not perform well" impression
  may be the *metric*, not the method (§3.1).
* The 30 s averaging compounds this: time-means carry no phase information about the
  fluctuations, so the observed quantity constrains the slowly varying / mean component
  of the flow near the sensors — a few hundred numbers per window about a specific
  functional of the state, versus 4.8 M unknowns.

Rule of thumb from the observability literature applied here: what is actually
estimable from this network is, roughly, **the interval-mean, canopy-layer, large-scale
(≳ sensor-spacing) flow — plus anything the model can regenerate from
better-constrained boundary conditions.** That is precisely what the inflow-parameter
estimation already delivers, and why it works while the state update does not.

### 1.3 The IC-to-observation map is chaotic: the update targets the least useful variable

The augmented vector holds the window's **initial condition**, but the observations are
interval means throughout a 300 s window. In-canopy eddy turnover is tens of seconds,
so the sensitivity of a late-window observation to an IC perturbation has decorrelated
through chaotic divergence — the true IC–obs correlation is near zero for most of the
window, and the 50-sample estimate of it is dominated by noise. Distance-based
localization does not help because the failure is in *time*, not space: a sensor 10 m
from a grid cell is still uninformative about that cell's velocity 200 s earlier.
Evensen et al. (2024, MWR) demonstrate the general form of this failure: when the
window exceeds the model's predictability time, updating the IC and rerunning saturates
to climatology by the window end, and **no assimilated information survives into the
next window's warm start** — which is exactly the carry mechanism `run_esmda.py` relies
on (`state_input = posterior_state.isel(time=-1)`). Their remedy is directly
implementable here (§3.3).

### 1.4 Analysis increments are unbalanced states the solver must digest

The Kalman increment is a statistical object with no divergence/boundary/realizability
constraints; the analyzed IC is handed straight to the forward model three times per
window. Ensemble DA is known to generate imbalance through sampling error and
localization (He et al. 2020, JAMES — verified), and in turbulence specifically the
signature is **spurious energy injection at high wavenumbers**, escalating in the worst
case to the (rigorously proven) catastrophic filter divergence of ensemble filters
(Kelly, Majda & Tong, PNAS 2015 — verified; LBM-LETKF study arXiv:2308.03972 diagnoses
it via analysis-increment spectra, a diagnostic worth copying — §2). In an LBM solver
the insult is compounded: the increment perturbs macroscopic fields, and the
distribution functions must re-equilibrate, generating acoustic transients. Some of the
"state update makes things worse" effect is likely insertion shock, not estimation
error.

### 1.5 No spread maintenance across a chained, tempered pipeline

Each window consumes the full likelihood (Σ 1/α = 1), the posterior seeds the next
window's prior, and there is no inflation anywhere. Small ensembles systematically
underestimate variance, and underestimation feeds back into further underestimation —
the canonical spiral toward ignoring observations (Tandeo et al. 2020 review; Whitaker &
Hamill 2012). After a few windows the state ensemble's spread reflects neither the true
error nor the model error, so even the observable directions stop updating. (The
parameter block partially escapes this because the AR(2) prior extrapolation re-injects
spread between windows — another reason parameters behave better than the state.)

### 1.6 Model error makes the state increment absorb bias it cannot keep

Truth and assimilation models are different solvers. Under structural model bias,
direct state updates are what reduces the data mismatch (parameter-only estimation is
provably insufficient when the model cannot reproduce the truth at any parameter
value), but the gain is not retained once the increment stops — the rerun drifts back
along the model's bias (arXiv:2501.18262, EnKF+IBM channel study). In the current
pipeline the state increment is applied at the window IC and must survive 300 s of
biased integration before it is evaluated — most of it will not.

**Bottom line of the analysis.** The current design asks a rank-49, noisily estimated,
unbalanced, once-per-window update to determine the instantaneous 4.8 M-DOF state of a
chaotic flow from 180 time-averaged street-level numbers, then judges it by how well
the full field matches instantaneous truth. Each element of that sentence can be
improved independently, and several can be side-stepped by re-posing the problem.

---

## 2. Diagnostics to run before adding machinery

Cheap, in rough order; each discriminates between the failure modes of §1. (The
diagnostics module proposed in the code-review report would make these routine.)

1. **Spread–skill ratio per window** (state block, in obs space and in state space):
   spread ≪ error ⟹ §1.5 (inflation needed); spread ≈ error but no skill gain ⟹
   observability (§1.2–1.3).
2. **Innovation χ²**: `dᵀ(C_DD + αC_D)⁻¹d / N_d` per MDA step. Drifting ≫ 1 across
   windows confirms the underdispersion spiral; ≪ 1 says obs error is overestimated.
3. **Analysis-increment energy spectra** (the arXiv:2308.03972 diagnostic): compare the
   spectrum of the state increment to the flow's spectrum. Increment energy at high
   wavenumbers comparable to (or above) the flow's dissipation-range energy ⟹ §1.4
   (spurious injection / insertion shock).
4. **State-update ablation**: `state_and_dynamic` vs `dynamic` with identical seeds and
   priors. If parameter posteriors are *worse* with the state update on, the state rows
   are stealing innovation via spurious cross-correlations — strong evidence for
   §1.1/§1.3.
5. **Holdout-sensor validation**: assimilate 4 sensors, verify on 2 (Sousa & Gorlé's
   protocol). Skill at held-out sensors is the honest measure of field improvement,
   robust to the unobservable-scale issue in §1.2.
6. **IC informativeness horizon**: from the truth run, correlate the window-IC field
   with each interval's observations. The lag at which this dies is the fraction of the
   window that can inform the IC at all — if it is ~1–2 intervals (likely), the IC
   update is using ~20 of the 180 observations and fitting noise to the rest.
7. **Ensemble-size scaling probe**: one experiment at N_e = 100–200 (cheap case /
   surrogate). If state skill barely moves, the binding constraint is observability
   (§1.2), not sampling — which redirects effort from localization/inflation tuning
   toward re-posing the problem (§3.4–3.6).

---

## 3. Suggestions

Ordered from specific-and-cheap to general research directions. Tiers 1–2 stay within
the current ES-MDA architecture; tier 3 re-poses the state-estimation problem.

### Tier 1 — within the current architecture, low effort

**3.1 Fix the success metric before the method.** Under chaos, instantaneous point-wise
state RMSE is bounded below by the energy of the unobservable scales no matter how good
the DA is (§1.2). Evaluate: interval-mean fields, spectra, canopy-layer statistics, and
held-out-sensor time series. The turbulence-DA literature explicitly argues for
targeting statistics rather than instantaneous fields in chaotic flows (JFM
statistics-closure line of work). It is entirely possible the state update is already
adding skill *on the estimable component* and the metric is hiding it.

**3.2 Inflation (RTPS) on the state block + spread re-injection across windows.**
Directly treats §1.5. RTPS is two lines at the analysis site (posterior anomalies
rescaled toward prior spread) and is the standard companion of localization at N_e = 50;
combining relaxation with a small additive term helps when model error dominates
(Whitaker & Hamill 2012 — verified pattern of results). Also give the *state* carry the
analog of what the parameter GP extrapolation already does for params: add a small
perturbation (or blend toward a climatological/spun-up ensemble) when warm-starting
window k+1, so the state ensemble does not enter each window pre-collapsed.

**3.3 Change what carries across windows: update the window-END state, not (only) the
IC.** Evensen et al. (2024) show, specifically for ES-MDA over windows in chaotic
multiscale systems, that a direct ES-style update of the whole window (including the
end state that warm-starts the next forecast) in the *final* step stabilizes long
windows, improves the estimate, **and saves one ensemble integration** (the final
rerun). The machinery is 90 % present: `final_time_smoothing` already updates the full
trajectory in the reduced basis — repointing the cross-window carry at its smoothed
final frame (instead of the rerun's final frame) implements the recommendation, and
resolves the double-counting concern from the math review in the direction the
literature endorses (replace the last rerun rather than add an extra update). This is
the single most promising *specific* change for making assimilated information survive
into the next window.

**3.4 Localization tuning against the actual network geometry.** With 6 sensors, a
radius smaller than the inter-sensor spacing leaves most rows constrained by ≤ 1 sensor;
sparser networks favor *larger* radii, and small ensembles are very sensitive to
mistuning (Ying et al. 2018). Concretely: sweep the distance radius on a cheap case
(the metrics pipeline already exists); consider horizontal-only distance (already
implemented, `horizontal_only`) since all sensors share one height — vertical
correlations from z = 3 m sensors to the outer layer are exactly the spurious ones to
suppress; and cap the update above roof level (a vertical localization mask), where
sensors carry essentially no instantaneous information (§1.2).

### Tier 2 — moderate changes, same library

**3.5 Soften the insertion: IAU-style / mollified increments.** Instead of jumping the
IC and re-spinning, apply the state increment distributed over the first part of the
forecast as a body-force term (incremental analysis update). This suppresses the
insertion shock of §1.4 (He et al. 2020 — with the documented caveat that IAU's
low-pass character trades against very frequent updates). The forward models already
support body-force terms (cf. the inflow body-force machinery), so this is plumbing,
not research.

**3.6 Nudging as the state-estimation workhorse (continuous DA).** Add a feedback term
`−k(H(u) − y)` toward sensor readings *during* integration — negligible cost, no
ensemble needed for the state part (JFM square-cylinder URANS nudging; the
Azouani–Olson–Titi line gives the theory). Realistic division of labor for this
codebase: **nudge the state inside each member's forecast; keep ES-MDA for the
parameters.** This (a) keeps every member near the observed canopy flow, which also
tightens the predicted-obs spread the parameter update sees, (b) side-steps the rank
and insertion-shock problems entirely for the state, and (c) degrades gracefully with
sensor sparsity: it synchronizes the scales the network can see and leaves the rest to
the model. Expectations must match §1.2 — with 6 sensors, nudging reconstructs the
large-scale canopy flow near sensors, not the full field (LETKF needed ~an order of
magnitude fewer observation points than nudging for equal RMSE in 2-D turbulence,
arXiv:2308.03972 — but at 50× the cost).

**3.7 Sequential filtering at interval cadence (ties into the planned `filtering`
module).** For the *state*, a 30 s-cycle LETKF is better matched to the chaotic
decorrelation time than a 300 s-window IC smoother: each analysis targets the state
*now*, when the observations are actually informative about it (§1.3 disappears by
construction). The LBM-LETKF study (arXiv:2308.03972) is a direct proof of concept for
lattice-Boltzmann turbulence, including the divergence diagnostics; its observability
result (good accuracy requires observations resolving the energy-injection scale)
again tempers what 6 sensors can do. The filtering-module plan
(`da_filtering_module_plan.md`) covers the implementation; run it state-only or joint
with slowly-varying inflow parameters.

**3.8 Raise the effective rank with a climatological/hybrid component.** The repo
already owns large libraries of training trajectories (surrogate training data). Build
a static covariance (or a large frozen POD basis) offline from them and blend:
`P = β P_ens + (1−β) P_clim` — the hybrid-gain family (Houtekamer et al. 2019). This is
the one cheap way to give the update directions the 50 members cannot span, with
correlations learned from *real dynamics* rather than sampling noise. Implementation
fits the existing seams: a fixed precomputed basis in `OnlineStateReduction`'s slot
(fit once offline, rank 100–1000) plus the ensemble span, instead of the online
rank-≤49 basis.

**3.9 Scale-aware updating.** Since only large scales are estimable (§1.2), project the
state increment onto a large-scale band (spectral or coarse-grid filter) before
applying it — a poor man's scale-dependent localization (Buehner & Shlyaeva 2015 —
20–50 % error-std reduction from scale-dependent localization in small-ensemble
settings). This also removes most of the high-wavenumber insertion noise of §1.4 at the
same time.

### Tier 3 — re-pose the problem (research directions)

**3.10 Estimate fields that generate the state, not the state itself.** The strongest
lesson from both the urban-flow literature and this project's own results: with sparse
urban sensors, infer the **boundary conditions/forcing** and let the model regenerate
the interior. Sousa & Gorlé validated exactly this (EnKF inflow inference for urban
flow) against field data — with *six sonic anemometers*, the same count as this network
— doubling hit rates vs weather-station-driven BCs, with the caveat that improvement is
conditional on the inflow being identifiable from the sensors, and that **roof-level
sensors identify inflow much better than in-canopy ones**. Concrete extensions of the
current parameter vector, in increasing ambition: vertical inflow profile parameters
(shear/veer), spanwise-varying inflow (a few modes along the inlet), a low-dimensional
volumetric forcing field (weak-constraint model-error term, estimated as parameters).
Each stays in the well-behaved parameter-estimation regime the library already handles,
while widening how much of the state the observations can shape. And: **revisit sensor
placement** (add roof-level units) — per the observability analysis this buys more than
any algorithm change.

**3.11 Assimilate statistics instead of instantaneous states.** Where the goal is the
urban wind climate (means, gust statistics, exceedance maps), target statistics
directly: observations are already 30 s means; make the estimated quantities the mean
flow / low-order moments (per interval or window), with the LES supplying fluctuations
around them. The statistics-nudging closure literature (PRFluids 2025-line: fixed-gain
Kalman–Bucy nudging of spectral-coefficient statistics toward reference moments) shows
statistically steady turbulence can be held to reference statistics at coarse
resolution with per-step costs orders of magnitude below the reference — a natural fit
for pairing with the neural surrogate.

**3.12 Latent-space DA with a *learned* (nonlinear) reduction.** The observability of a
turbulent system from sparse data depends on the coordinates of assimilation; in 2-D
turbulence, variational DA in an autoencoder latent space beat state-space DA by ~two
orders of magnitude in reconstruction error and suppressed the spurious small-scale
artifacts of state-space updates (arXiv:2512.15470); Latent-EnSF handles 10⁵-DOF states
with 100 members under extreme sparsity where vanilla ensembles fail
(arXiv:2409.00127). Unlike the linear online SVD (span-limited by construction, §1.1),
a nonlinear latent space learned from the existing training-trajectory library encodes
the attractor: updates decoded from it are approximately realizable states, killing
§1.4, and the latent dimension (10²–10³) is commensurate with N_e. This is the natural
research direction for this project specifically, because the autoencoder
infrastructure and training data already exist for the surrogate work. Caveat from the
same literature: latent ensemble-*Kalman* variants can be fragile as latent dimension
grows and under temporal sparsity — start with latent ETKF on modest latent dims, keep
score-based filters in view.

**3.13 Iterative ensemble Kalman smoother (IEnKS) with quasi-static windows.** The
principled endpoint of the current window-iterated design: Gauss–Newton in-window
iterations, ensemble-derived sensitivities (no adjoint — compatible with LBM), MDA and
quasi-static variants to stabilize long chaotic windows. Reported to outperform
EnKF/4D-Var/EnKS on chaotic benchmarks (Bocquet & Sakov 2014 — extraction upheld for
the method properties; the strongest performance claims did not survive verification
unqualified, and the authors themselves caveat that results assume perfect low-dim
models). Given model error and cost here, treat as a benchmark/exploration item behind
3.3/3.6/3.10, not a first move.

### Suggested sequencing

1. Diagnostics §2 (one week of runs, mostly existing plumbing) — establishes *which*
   failure dominates.
2. Tier 1: metric fix + RTPS + window-end carry (3.1–3.3), localization sweep (3.4).
3. Tier 2, chosen by diagnostics: nudging hybrid (3.6) if insertion shock/observability
   dominates; hybrid covariance (3.8) if rank dominates; filtering cadence (3.7)
   alongside the planned filtering module.
4. Tier 3: inflow-field enrichment + sensor placement (3.10) is low-risk and
   high-value regardless; latent-space DA (3.12) as the research track.

---

## 5. Sources and verification status

The deep-research pass fetched and extracted claims from the sources below;
adversarial verification (3 independent voters per claim) completed for a subset before
the session budget ran out. Status: **[V]** = verified (≥2/3 upheld), **[E]** =
extracted from the fetched source but not independently re-verified, **[R]** = refuted
(not used above, listed for transparency).

* [V] Carrassi, Bocquet, Demaeyer, Grudzien, Raanes, Vannitsem — *Data assimilation for
  chaotic dynamics* (arXiv:2010.07063): covariance collapse onto the unstable–neutral
  subspace; convergence requires the ensemble to project on it and the network to
  control it; error "upwelling" under model noise justifies inflation.
* [V] Kelly, Majda & Tong (PNAS 2015): catastrophic filter divergence is a rigorous
  dynamical property of ensemble filters. [R] the specific perpendicular-alignment
  mechanism claim as universally stated. [E] additive inflation averts it in their model.
* [V] He, Lei, Whitaker & Tan (JAMES 2020, 10.1029/2020MS002187): ensemble DA generates
  imbalance via sampling error and localization; IAU vs assimilation-frequency
  trade-offs.
* [V] Bocquet & Sakov (QJRMS 2014, 10.1002/qj.2236): IEnKS = 4D-Var-like smoothing
  without adjoints, flow-dependent. [R] the unqualified "systematically outperforms"
  claim.
* [E] Evensen (Comput. Geosci. 2018): ES-MDA ≠ Bayesian posterior for nonlinear models
  even at huge N_e; step-schedule trade-offs.
* [E] Evensen et al. (MWR 2024, 10.1175/MWR-D-23-0239.1): IC-update+rerun saturates to
  climatology beyond the predictability time; final whole-window ES update stabilizes
  long windows and saves an integration; the opposite holds for IES.
* [E] Bocquet & Carrassi (Tellus A 2017): 4DEnVar/IEnKS and the unstable subspace;
  ensemble-size lower bound n₀; anomalies align with unstable subspace as windows
  lengthen.
* [E] Fillion, Bocquet & Gratton (NPG 2018): quasi-static minimization for long chaotic
  windows; analytic bound on usable window length.
* [E] arXiv:2308.03972 (LBM + LETKF, 2-D turbulence): LETKF beats nudging per
  observation; spurious high-wavenumber energy injection and increment-spectrum
  diagnostic; Nyquist-style observability criterion at the energy-injection scale.
* [E] JFM: *What is observable from wall data in turbulent channel flow?* and
  arXiv:2011.03711 (Wang & Zaki line): estimation accuracy decays away from sensed
  surfaces; Taylor-microscale sensor-spacing threshold; space–time resolution trade via
  mean advection.
* [E] arXiv:2512.15470: latent-space variational DA in 2-D turbulence — ~two orders of
  magnitude error reduction vs state-space DA; observability depends on assimilation
  coordinates.
* [E] arXiv:2409.00127 (Latent-EnSF): sparse-observation failure of vanilla filters in
  10⁵-dim states; coupled-VAE latent filtering; fragility notes for latent EnKF/LETKF.
* [E] JFM square-cylinder nudging (Re = 22 000): URANS nudged to sparse velocity data;
  synchronization when spacing ~ structure scale; negligible added cost.
* [E] Di Leoni, Mazzino & Biferale (spectral nudging): unnudged scales are not slaved to
  nudged large scales; nudging sharpens parameter identifiability.
* [E] PRFluids 10.1103/PhysRevFluids.10.013801: statistics-nudging closure ≡ fixed-gain
  Kalman–Bucy filter on spectral statistics; large speed-ups for statistically steady
  turbulence.
* [E] Sousa, García-Sánchez & Gorlé (Build. Environ. 2018) and Sousa & Gorlé (2019 field
  validation): urban inflow-BC inference by (i)EnKF from ~6 sensors; hit rates ×2;
  improvement conditional on identifiability; roof-level sensors preferred.
* [E] arXiv:2501.18262 (EnKF + IBM channel, Re_τ ≈ 550): state updates needed under
  structural model bias; gains not retained after observations stop; ML corrective model
  amortizes the ensemble cost.
* [E] Whitaker & Hamill (MWR 2012): RTPS; multiplicative inflation ↔ sampling error,
  additive ↔ model error, combination best.
* [E] Kotsuki et al. (QJRMS 2017): adaptive RTPS/RTPP beat adaptive multiplicative
  inflation in NICAM-LETKF; over/under-dispersion caveat vs observation density.
* [E] Buehner & Shlyaeva (Tellus A 2015) and Ying, Zhang & Anderson (MWR 2018):
  scale-dependent localization; radius↔ensemble-size↔network-density trade-offs; 50-member
  sample covariances significantly wrong at all distances.
* [E] Tandeo et al. (MWR 2020 review): innovation-based (Desroziers) diagnostics;
  underdispersion as the primary ensemble-DA failure mode.
* [E] Houtekamer et al. (QJRMS 2019): hybrid-gain blending of ensemble and static
  covariances.
* [R] "RMSE drops abruptly at exactly N = n₀+1 with no further improvement" (QG
  experiment over-reading) — the qualitative ensemble-size threshold stands (see [V]
  Carrassi et al.), the sharp quantitative version does not.
