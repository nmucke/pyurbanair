# D — Synthesis of the ensemble-DA evidence (pyurbanair, ISDA 2026 campaign)

Sources read: `presentations/isda_crps/latex/main.tex` (full), `.../scripts/{numbers.json,
make_figures.py, run_bias_cases.sh, run_barcelona_cases.sh}`, ~12 figure PNGs in
`presentations/isda_crps/latex/figures/`, `docs/plans/isda2026_talk_experiments.md`,
`experiments_report/{main.tex,sections/*.tex,scripts/comparison_data.json}`,
`experiments_report/figures/comparison/cmp_generalisation.png`, the raw run dirs under
`presentations/isda_new/experiments/{esmda,filtering,filter_smoothing}/*/run_summary.yaml`
+ `config.yaml`, `conf/{run_filtering,run_esmda,run_filter_smoothing}.yaml`,
`conf/case/{xie_and_castro,barcelona}.yaml`, `conf/params/*`, `conf/{esmda,filtering}/*`,
`libs/pyudales/.../inlet_turbulence_utils.py`,
`libs/data-assimilation/.../filtering/parameter_evolution.py`, and the truth/bias NetCDFs.

Numbers I computed myself (from `presentations/isda_new/experiments/truth_states/*.nc` and
`presentations/isda_crps/latex/results_bias/*/run/pyudales_time_varying/state.nc`) are marked
**[computed]**. Inferences are marked **[speculation]**.

---

## 1. Experimental setup — the facts

**Geometry / grid.** Xie & Castro (2008) staggered cube array,
`examples/xie_and_castro/xie_castro_2008_STL.stl`. Domain 40×60×16 cells over
x∈[−20,40], y∈[0,80], z∈[0,32] m → **Δ = 2 m isotropic**, 38 400 cells. Tallest roof 17.2 m
(18 m voxelised). Both truth and assimilation model are uDALES with the **Vreman** closure,
`sgs_constant (c_vreman) = 0.24` — chosen for stability, not physics (0.07 and 0.15 diverge on
this domain; `conf/model/pyudales.yaml` and auto-memory `project_udales_cvreman_stability`).

**Sensors.** 6 assimilated, all at z = 2 m in the open N–S lanes: (10,20), (10,60), (20,10),
(20,50), (30,30), (30,70). 4 held-out (`conf/case/xie_and_castro.yaml`):
(2,30,2) = **undisturbed approach flow upstream of the array**, (20,55,2) = open lane,
(17,30,2) = **wake/recirculation ~2 m behind a 13–14 m block**, (10,30,20) = **the only
above-canopy sensor**, one cell above an 18 m roof (in the roughness sublayer, not free stream —
the config comment says so explicitly). So "held-out" is *not* simply "downstream": it spans
approach flow, lane flow, near-wake and above-canopy — genuinely different flow regimes.
Observed components are **u and v only** (`obs.states: [u,v]`), but *every reported sensor RMSE
is a 3-component vector RMSE including the never-observed w*. σ_obs = 0.1 m/s per component
(so the observed 2-component noise vector is 0.141 m/s).

**Ensemble / cadence.** N = 50 for every run (`ensemble_size: 50` in every campaign
`config.yaml`; note `conf/run_filter_smoothing.yaml` still ships the old default 40).
3 windows × 120 s = **360 s horizon**; model output every 2 s → 180 frames.
- Filter (F*): **one analysis every 2 s = 180 analyses**, `assimilate_every_n_step=1`, 12 obs
  values per analysis, 720 per window.
- ESMDA (E*): **one analysis per 120 s window**, observations aggregated into 15 s bins →
  6 sensors × 2 comps × 8 bins = **96 obs per window**; N_a = 3 MDA steps (4 forward passes).
- Hybrid (H*): both (ESMDA parameters + EnKF state inside the same window).
`spinup_time: 150.0` s per window in every campaign config; `run.truth_start_time: null`
(no spin-up is trimmed from the scored record — see §3.4).

**Parameters.** Time-varying inflow angle α and speed |U| (1 knot / 30 s → 4 knots per window,
12–15 over the horizon), plus a static vertical exponent (0.25, not estimated).
- Truth: `params=dynamic_sine`, **deterministic** α = 20° sin(2πt/300 s),
  |U| = 7.5 + 1.0 sin(2πt/150 s) m/s.
- Smoother prior: AR(2) relaxation, L_corr = 200 s, α ~ N(0°,10°), |U| ~ N(7.5, 0.5) m/s.
- Filter prior: **static** N(0°,10°) / N(7.5,0.5) with a **random walk** between cycles,
  `RandomWalkEvolution(std=0.5)` — one scalar applied to **every** parameter in raw units
  (see §4, this is the single most consequential tuning fact in the campaign).
- Inflation: RTPS α = 0.6. Localization: correlation-based, truncation ρ_t = 0.35, taper
  β = 0.5, max inflation 8, block grouping (identical in `conf/esmda/localization/correlation.yaml`
  and `conf/filtering/localization/correlation.yaml` — *not* tuned per case).

**Cases.** All three share the geometry and differ only in the BC:
`inflow` = laminar power-law inlet + interior nudging; `inflow_turb` = same inlet + synthetic
digital-filter eddies (intensity 0.05, L = 6 m in x/y/z, dtdriver 0.5 s) — and **turning the
inlet turbulence on turns interior nudging OFF** (`lnudge=.false.`, documented in
`conf/model/pyudales.yaml`), so the parameters act only through the driver planes;
`periodic` = cyclic in x, **interior nudging is the only momentum source**.

**Truth generation.** Same model, same closure, same c_vreman, one truth per case stored in
`presentations/isda_new/experiments/truth_states/true_state_pyudales_<case>.nc`, shared by all
methods; `seed: 42` everywhere, **one seed per configuration, never repeated**. A PALM-truth
model-error arm exists (E1–E10, F1–F6, H1–H10); both PALM *laminar* cells failed at truth
generation. **Synthetic-eddy seeds are NOT shared**: `inlet_turbulence.seed: null` →
`derive_seed(experiment_name)` (blake2b of the member's experiment name,
`libs/pyudales/src/pyudales/utils/inlet_turbulence_utils.py:319`), so the truth and each of the
50 members get an independent inlet realisation. This is the crux of §3.3.

---

## 2. Results, with baselines

### 2.1 Trivial baselines I computed **[computed]**

RMSE of the *oracle climatology* predictor (predict each sensor's own time-mean over the scored
360 s), 3-component, pooled over the sensor set — this is the score a completely uninformative
ensemble mean would get, because a decorrelated 50-member ensemble mean collapses to the mean
field:

| case | assimilated (6) | held-out (4) | held-out, excl. t<50 s |
|---|---|---|---|
| `inflow` | 0.942 | **1.378** | 1.428 |
| `inflow_turb` | 1.009 | **1.413** | 1.461 |
| `periodic` | 1.432 | **1.611** | 1.681 |

Per-sensor held-out climatology (`inflow`/`inflow_turb`/`periodic`):
(2,30,2) 0.85/0.88/1.43 · (20,55,2) 1.76/1.76/1.78 · (17,30,2) 1.02/1.10/0.92 ·
(10,30,20) 1.65/1.71/2.08.

Parameter climatology: α prior RMSE 16–20° (RMS of the truth sine alone is 20/√2 = 14.1°),
|U| prior RMSE 0.73–1.0 m/s.

### 2.2 Master table — matched uDALES truth, all reported metrics + skill vs climatology

`assim`/`valid` = 3-component vector RMSE [m/s]; `ES` = energy score (multivariate CRPS) — these
are the numbers the CRPS deck's tables quote; `field` = domain |U| RMSE; α/|U| = knot-mean
parameter RMSE with skill vs that run's own prior; `χ²` innovation (1 = calibrated); `z_pool`
parameter z-std (1.03 = calibrated); `q` field hit rate. `skill_V` = 1 − valid/climatology.

| ID | method | case | loc | assim | valid | ES_a | ES_v | **valid/assim** | **skill_V** | field | α° (skill) | \|U\| (skill) | χ² | z_pool | q |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| E11 | ESMDA | inflow | on | 0.665 | 0.748 | 0.498 | 0.601 | 1.12 | 46% | 0.477 | 7.74 (61%) | 0.529 (31%) | — | 12.1 | 0.72 |
| E18 | ESMDA | inflow | off | 0.694 | 0.754 | 0.539 | 0.611 | 1.09 | 45% | 0.521 | 7.59 (62%) | 0.671 (18%) | — | 13.6 | 0.72 |
| F14 | EnKF state, no infl | inflow | off | 0.236 | 0.570 | 0.174 | 0.365 | 2.41 | 59% | 0.299 | †19.29 (0) | †0.919 (0) | 3.98 | †1.88 | 0.53 |
| F15 | EnKF state+RTPS | inflow | off | 0.157 | 0.539 | 0.105 | 0.304 | 3.44 | 61% | 0.229 | †19.29 | †0.919 | 0.95 | †1.88 | 0.54 |
| F8 | EnKF state+RTPS | inflow | on | 0.167 | 0.626 | 0.113 | 0.360 | 3.75 | 55% | 0.261 | †19.29 | †0.919 | 0.92 | †1.88 | 0.48 |
| **F13** | EnKF joint+RTPS | inflow | off | **0.144** | **0.319** | 0.097 | 0.203 | 2.21 | **77%** | 0.221 | 3.90 (87%) | 0.818 (14%) | 0.99 | 2.23 | 0.63 |
| F7 | EnKF joint+RTPS | inflow | on | 0.155 | 0.394 | 0.105 | 0.250 | 2.54 | 71% | 0.247 | 5.26 (82%) | 0.910 (9%) | 0.90 | 1.76 | 0.61 |
| H11 | Filter smoothing | inflow | on | 0.506 | 0.593 | 0.418 | 0.473 | 1.17 | 57% | 0.486 | 5.81 (66%) | 0.616 (24%) | 13.3 | 10.6 | 0.43 |
| H18 | Filter smoothing | inflow | off | 0.524 | 0.620 | 0.445 | 0.505 | 1.18 | 55% | 0.438 | 7.87 (53%) | 0.506 (38%) | 15.0 | 15.7 | 0.43 |
| **E12** | ESMDA | inflow_turb | on | 0.401 | **0.365** | 0.245 | 0.234 | **0.91** | **74%** | 0.372 | **2.31** (87%) | **0.369** (53%) | — | 4.6 | 0.62 |
| E19 | ESMDA | inflow_turb | off | 0.456 | 0.407 | 0.284 | 0.269 | 0.89 | 71% | 0.421 | 2.45 (87%) | 0.439 (42%) | — | 7.8 | 0.62 |
| E16 | ESMDA (obs 7.5 s) | inflow_turb | on | 0.439 | 0.422 | — | — | 0.96 | 70% | 0.408 | 2.42 (86%) | 0.379 (51%) | — | 6.1 | 0.74 |
| E14 | ESMDA (obs 30 s) | inflow_turb | on | 0.418 | 0.366 | — | — | 0.88 | 74% | 0.388 | 2.35 (87%) | 0.414 (48%) | — | 4.1 | 0.80 |
| F17 | EnKF state, no infl | inflow_turb | off | 0.390 | 0.991 | 0.294 | 0.707 | 2.54 | 30% | 0.687 | †19.29 | †0.919 | 8.79 | †1.88 | 0.46 |
| F18 | EnKF state+RTPS | inflow_turb | off | 0.261 | 0.772 | 0.166 | 0.478 | 2.96 | 45% | 0.476 | †19.29 | †0.919 | 1.82 | †1.88 | 0.49 |
| F10 | EnKF state+RTPS | inflow_turb | on | 0.240 | 0.753 | 0.151 | 0.459 | 3.13 | 47% | 0.445 | †19.29 | †0.919 | 1.13 | †1.88 | 0.44 |
| F16 | EnKF joint+RTPS | inflow_turb | off | 0.260 | 0.624 | 0.167 | 0.421 | 2.40 | 56% | 0.500 | 4.88 (83%) | 0.955 (−13%) | 1.95 | 3.29 | 0.50 |
| F9 | EnKF joint+RTPS | inflow_turb | on | 0.206 | 0.559 | 0.134 | 0.358 | 2.71 | 60% | 0.422 | 3.36 (88%) | 0.912 (3%) | 1.04 | **0.74** | 0.49 |
| H12 | Filter smoothing | inflow_turb | on | 0.272 | 0.497 | 0.182 | 0.343 | 1.83 | 65% | 0.533 | 2.61 (84%) | 0.433 (46%) | 3.68 | 5.0 | 0.54 |
| H19 | Filter smoothing | inflow_turb | off | 0.256 | 0.441 | 0.178 | 0.310 | 1.72 | 69% | 0.430 | 3.45 (82%) | 0.365 (54%) | 3.43 | 9.1 | 0.73 |
| E13 | ESMDA | periodic | on | 1.091 | 1.408 | 0.698 | 0.871 | 1.29 | **13%** | 1.040 | 13.62 (18%) | 0.717 (9%) | — | 2.6 | 0.71 |
| E20 | ESMDA | periodic | off | 1.163 | 1.671 | 0.748 | 1.062 | 1.44 | **−4%** | 1.075 | 14.84 (22%) | 0.764 (4%) | — | **89.8** | 0.58 |
| F20 | EnKF state, no infl | periodic | off | 0.735 | 2.391 | 0.605 | 1.795 | 3.25 | **−48%** | 1.968 | †19.29 | †0.919 | 27.7 | †1.88 | 0.66 |
| F21 | EnKF state+RTPS | periodic | off | 0.707 | 2.362 | 0.507 | 1.714 | 3.34 | −47% | 2.061 | †19.29 | †0.919 | 14.1 | †1.88 | 0.66 |
| F12 | EnKF state+RTPS | periodic | on | 0.315 | 1.261 | 0.198 | 0.773 | 4.00 | 22% | 1.029 | †19.29 | †0.919 | 1.76 | †1.88 | 0.83 |
| F19 | EnKF joint+RTPS | periodic | off | 0.826 | 3.653 | 0.598 | 2.645 | 4.42 | **−127%** | 2.951 | 19.12 (−8%) | 2.173 (−3%) | 17.8 | 11.5 | 0.48 |
| F11 | EnKF joint+RTPS | periodic | on | 0.317 | 1.551 | 0.200 | 0.913 | **4.89** | 4% | 1.733 | 19.03 (−17%) | **4.221 (−38%)** | 1.85 | 2.2 | 0.89 |
| **H13** | Filter smoothing | periodic | on | 0.314 | **1.246** | 0.196 | 0.764 | 3.96 | **23%** | 1.019 | 14.58 (12%) | 0.756 (7%) | 1.81 | 2.7 | 0.87 |
| H20 | Filter smoothing | periodic | off | 0.706 | 1.766 | 0.549 | 1.397 | 2.50 | −10% | 1.529 | 8.89 (45%) | 0.928 (−27%) | 23.1 | 41.5 | 0.76 |

† = carried static prior; the parameter ensemble is never updated in state-only mode.

**PALM-truth (model error), condensed** — held-out RMSE 1.30–1.43 (`inflow_turb`) and 1.75–4.08
(`periodic`) for every method; both laminar cells failed at truth generation. F3 (joint EnKF,
periodic, loc on) is a full blow-up: α RMSE **138.5°**, |U| RMSE **21.3 m/s**, z_val 42.5.

### 2.3 What the tables say once baselines are in

- `inflow`: best held-out 0.319 (F13) = **77% skill** vs climatology (95% of the variance).
  Real, unambiguous skill.
- `inflow_turb`: best held-out 0.365 (E12) = **74% skill** (93% of variance). Also real.
- `periodic`: best held-out 1.246 (H13) = **23% skill** (40% of variance). Every unlocalized
  variant is **worse than climatology** (skill −4% to −127%). Localization is what keeps the
  periodic runs on the useful side of "do nothing", and even then the margin is thin.
- The `assim` column tells a completely different story (78–85% skill for every filter, in every
  case) and is **not** a skill measure: it is the fit at points that were just nudged.

---

## 3. What the evidence establishes — and what it does not

### 3.1 Where held-out error plateaus, relative to noise and to signal

Held-out error is 2.3–11× the observed-component noise vector (0.141 m/s) and never approaches
it. The useful comparison is against the *signal* at those sensors:

| case | best held-out | climatology | obs noise (2-comp) | held-out / climatology |
|---|---|---|---|---|
| inflow | 0.319 | 1.378 | 0.141 | 0.23 |
| inflow_turb | 0.365 | 1.413 | 0.141 | 0.26 |
| periodic | 1.246 | 1.611 | 0.141 | **0.77** |

So: **the periodic case is the only one where "the method is doing almost nothing at held-out
sensors" is literally true** — 1.246 vs 1.611 is roughly the residual you get from a
mean-field prediction. `valid2_E13.png` and `valid2_H13.png` show this directly: the posterior
mean at (20,55,2) and (17,30,2) is a nearly flat line at ~1.5 and ~1.1 m/s while the truth swings
0.2–4.5 m/s. The ensemble *brackets* the truth (q = 0.87, χ² = 1.8) but the mean carries no
information. The turbulent-inflow case is the opposite: at those same sensors ESMDA's mean tracks
the low-frequency envelope well and misses only the fast wiggle (`valid2_E12.png`, t = 50–130 s).

**Mechanistic evidence for the plateau (from `run_summary.yaml` `filter_diagnostics`):**
each EnKF analysis reduces observation misfit ~5–9× but the **state spread by only 1–15%**:

| run | obs_prior→obs_post RMSE | state spread prior→post | contraction |
|---|---|---|---|
| F13 laminar | 0.161 → 0.072 | 0.367 → 0.314 | 14.4% |
| F9 turbulent | 0.250 → 0.049 | 0.657 → 0.627 | 4.6% |
| F11 periodic | 0.433 → 0.050 | 1.124 → 1.112 | **1.1%** |

Note also `obs_posterior_rmse ≈ 0.05 < σ_obs = 0.1` in the turbulent and periodic runs: the
analysis is drawn *inside* the observation noise at its own sensors. That is over-fitting the
noise at 12 points while 99% of the state is untouched — exactly the valid/assim = 2.4–4.9
signature. **[computed/inference]** For the periodic run the 3-component assim RMSE 0.317 is
almost entirely the *unobserved* w: the (u,v) part contributes only √2 × 0.0497 = 0.070, leaving
w-error ≈ 0.309, against a truth w-std of 0.371 at those sensors. The deck's headline
"periodic assim 1.09 → 0.31" is therefore mostly a statement about vertical velocity, not about
data fit.

**ESMDA's own convergence diagnostic is the cleanest single result in the campaign**
(`esmda_diagnostics.data_mismatch.per_step_median`, target O_N = 0.5):

| case | step 0 → 1 → 2 → 3 | verdict |
|---|---|---|
| E11 laminar | 46.8 → 21.0 → 18.1 → **17.5** | stalls at 35× the noise floor after one effective step |
| E12 turbulent | 33.7 → 4.89 → 2.57 → **2.37** | converges to 4.7× the floor |
| E13 periodic | 37.3 → 36.2 → 36.7 → **37.2** | **zero reduction across all three MDA steps** |

All three carry `underfit_final: true`. E13 is a null result in the strongest possible form: the
smoother cannot move the data misfit at all.

### 3.2 Periodic: identifiability, not (only) state estimation

**The parameters are not inert in the periodic case.** I ran the diff on the bias experiment
(`presentations/isda_crps/latex/results_bias/`, truth vs +5° angle / +0.5 m/s, same case,
`run_bias_cases.sh`) **[computed]**:

| case | domain-mean flow angle truth → biased | domain-mean u truth → biased |
|---|---|---|
| inflow | 1.60° → 5.88° | 5.92 → 6.26 |
| inflow_turb | 1.51° → 5.83° | 5.93 → 6.28 |
| **periodic** | **0.34° → 5.72°** | **3.85 → 4.10** |

The nudging target rotates the periodic domain mean essentially 1:1 with the parameter. So this
is *not* "the parameters do nothing under periodic BCs".

**But the instantaneous observation operator is ~91% chaos.** Same runs, at the held-out sensors:

| case | total response to the bias (vec RMSE) | of which a *mean shift* | coherent variance fraction |
|---|---|---|---|
| inflow | 0.676 | 0.498 | **54%** |
| inflow_turb | 0.736 | ~0.50 | ~46% |
| **periodic** | 1.775 | 0.527 | **8.8%** |

For `periodic` the response to a 5°/0.5 m/s perturbation (1.78) is *larger than the truth's own
climatological variability* (1.61) — i.e. it is dominated by chaotic trajectory divergence, not
by a parameter signal. Only ~9% of that response is a reproducible mean shift.

**Conclusion:** the periodic failure is an **identifiability failure at the timescale the
methods use**, not an absence of parameter influence and not purely a state-estimation problem.
The parameters are recoverable only from a *time-averaged* statistic over many eddy turnovers;
a 120 s window with 15 s bins does not average enough. This is consistent with (a) ESMDA's
data mismatch not moving at all, (b) α RMSE 13–15° for every method against a 16–19° prior
(12–22% reduction), (c) the ESMDA parameter contraction ratio 0.59–0.75 on periodic vs 0.22–0.30
on the inflow cases (`\EsmdaCalibTable`) — the smoother itself reports "I learned almost nothing",
and (d) the obs-interval sweep being flat (E17/E13/E15 → 12.94/13.62/13.63°), which is what you
see when there is no information to trade against.

The *state* half is a separate, partly-solvable problem: F12/H13 (state, localized) reach
held-out 1.246–1.261, which is 23% skill — so the localized state update does buy something,
just far less than in the inflow cases.

### 3.3 Turbulent inflow: how much error is irreducible

The premise is confirmed in code: `inlet_turbulence.seed: null` → `derive_seed(experiment_name)`,
a blake2b digest of each member's directory name. The truth run and all 50 members have
**independent** synthetic-eddy realisations. The ensemble mean averages 50 independent
fluctuation fields to ≈ 0, so the truth's *own* inlet-driven fluctuation is irreducible unless the
inlet planes themselves are estimated.

Three independent estimates of that floor:

1. **From the truth data [computed].** Held-out pooled climatology is 1.378 (`inflow`) vs 1.413
   (`inflow_turb`). Attributing the extra variance to the inlet fluctuations:
   √(1.413² − 1.378²) = **0.31 m/s**. (Caveat: the two truths are different chaotic
   trajectories, so this is indicative, not a clean decomposition. **[speculation]**)
2. **From the configuration.** intensity 0.05 → u′_rms ≈ 0.05 × |U(z)| ≈ 0.38 m/s at the inlet,
   v′,w′ 0.7× that; band-limited to ~1 Hz by dtdriver = 0.5 s.
3. **From ESMDA's residual.** O_N = 2.37 at convergence → obs-space residual
   σ√(2·O_N) = 0.1 × √4.74 = **0.218 m/s** on the *15-s-averaged* observations at the assimilated
   sensors; the un-averaged residual is larger.

ESMDA's held-out RMSE is 0.365. Against a floor of ≈0.31 that leaves roughly
√(0.365² − 0.31²) ≈ 0.19 m/s of *reducible* error — i.e. **ESMDA is already within ~20% (rms) of
the realisation floor, and no amount of parameter estimation can close the last 0.31.**
**[speculation, but three independent estimates agree to within 30%.]** The visual signature is
`valid2_E12.png` panel 2: the ensemble mean is flat at ~1.15 through t = 50–130 s while the truth
oscillates ±0.5 — that oscillation is the inlet realisation, and it is unlearnable from 6 sensors
without estimating the inlet field.

The practical implication: **the "turbulent case is worse than it should be" intuition is largely
wrong for ESMDA** (74% skill, near the floor); it is right for the *filters* (F9 0.559, F16 0.624
— 40–44% above the floor), which is a filter problem, not an irreducibility problem.

### 3.4 Confounds (all real, several load-bearing)

1. **The executed campaign is not the planned one.** `docs/plans/isda2026_talk_experiments.md`
   specifies obs interval 10 s = filter cycle, ESMDA window 30 s, N_a = 4, horizon 300 s, and
   **3 seeds for the headline head-to-heads**. What ran: filter cycle **2 s**, ESMDA window
   **120 s**, N_a = 3, horizon 360 s, **1 seed everywhere** (`seed: 42`). The plan's E7–E11
   (ETKF, LETKF, state reduction, N_e and interval sweeps) were never run.
2. **Unequal observation counts.** ESMDA sees 96 aggregated values per window; the filters see
   720 raw ones. The report flags this (caveat 1, `comparison.tex:687`) — the turbulent result
   (ESMDA best on held-out from less data) survives it in sign; the laminar result does not.
3. **`assim` is a different quantity per method.** For the filter it is scored at the analysis
   (post-update) at the same instants it just fitted; for ESMDA it is a free re-simulated
   trajectory. "ESMDA assim 1.09 vs filter 0.31" is a fit-vs-forecast comparison, not a skill
   comparison. The report says so in the table caption; the deck's periodic slide does not.
4. **Spin-up is inside the scored record for `periodic` [computed].** The periodic truth's
   street-level sensors are at |U| ≈ 0.01 m/s until t ≈ 40 s and only reach statistical steady
   state around t ≈ 50–60 s (`ramp` check on `true_state_pyudales_periodic.nc`; visible in
   `valid2_E13.png`, `valid2_H13.png`, `assim2_F11.png`). That is ~22 of 180 frames (12%) in which
   truth ≈ ensemble ≈ 0 and every method scores ≈ 0 error. All periodic RMSEs are therefore
   deflated by ~6% in rms (common-mode, so rankings survive, but the absolute numbers are
   optimistic). `run.truth_start_time` exists precisely to trim this and was left `null`.
   Also: `setup.tex` says "50 s spin-up"; the run configs say `spinup_time: 150.0`.
5. **Nothing was tuned per case, which is its own confound.** RTPS α = 0.6, ρ_t = 0.35 and
   random-walk std = 0.5 are the same in all 45 cells. The paired-run "technique effect" claims
   (localization −47% on periodic, +23% on laminar) are therefore *one setting's* effect, not the
   technique's — a per-case ρ_t was never tried.
6. **Filter figures use a cycle-index x-axis, E/F figures use seconds.** Documented in
   `make_figures.py`'s docstring, but the H* deck panels (0–180) sit next to E*/F* panels (0–360)
   in the same talk with no axis label difference visible to the audience.
7. **Two RMSE definitions circulate**: the tables use the 3-component vector RMSE; every
   pipeline figure legend prints a magnitude-only |U| RMSE about half as large. Flagged in the
   report (caveat 6), invisible in the deck.
8. **Localization is confounded with spread preservation.** Every `locnone` run with parameter
   updating has a collapsed parameter ensemble (contraction 0.16–0.23, z_pool 11.5–263) while the
   localized run has contraction 0.63–0.75. So "localization improves periodic held-out RMSE by
   47%" is partly "localization stops the parameter ensemble from collapsing".

---

## 4. Suspicious / inconsistent numbers

1. **The unscaled random walk is, in my reading, a bug-grade mis-tuning.**
   `conf/filtering/evolution/random_walk.yaml` sets `std: 0.5`, and
   `RandomWalkEvolution` applies **one scalar to every parameter in raw physical units**
   (`parameter_evolution.py:69–72`). Per 2 s cycle that is 0.05 prior-σ on the angle
   (prior σ = 10°) but **1.0 prior-σ on the velocity magnitude** (prior σ = 0.5 m/s) — a 20×
   mis-scaling. Over one 60-cycle window the free walk grows to 3.9° and **3.9 m/s**. This
   predicts exactly what the tables show: the joint EnKF's α skill is 82–88% on the inflow cases
   while its **|U| skill is ≈ 0 everywhere** (+14%, +9%, −13%, +3%, −38%, −3%), and it *diverges*
   to |U| RMSE 4.22 m/s (F11) / 21.3 m/s (F3, PALM) exactly where the observations carry no |U|
   information. `params_F11.png` shows the posterior |U| mean swinging 2–12 m/s against a truth
   of 6.5–8.5. **This alone is a plausible explanation for "the joint filter can't do speed",
   and it is a tuning artefact, not physics.**
2. **Identical carried-prior entries across 9 runs.** α = 19.29°, |U| = 0.919 m/s and
   z_pool = 1.8775 are byte-identical in F14/F15/F8/F17/F18/F10/F20/F21/F12 because state-only
   mode never touches the parameter ensemble. The report daggers them; **`numbers.json` does
   not**, and the deck's plain/RTPS table rows would read as parameter results to a reader who
   only sees the deck.
3. **χ² null for every ESMDA run** is correct by construction (a smoother has no sequential
   innovation) but means the cross-method calibration claim rests on z-scores alone — and the
   z-scores are computed from **n = 3 knot evaluations per parameter** (`z_score.n: 3`,
   pooled n = 6) in the filter runs. A "z_pool = 89.8" from 6 samples is not a stable statistic.
4. **z_pool spans 0.74 → 263** across the campaign, and the extremes are *all* `locnone` runs
   (E20 89.8, E10 263.3, H20 41.5, H10 101.8). This is parameter-ensemble collapse, not a
   method property; treating z_pool as a comparable calibration metric across loc on/off is
   unsafe.
5. **F11 (periodic joint EnKF, localization ON) still diverges** in |U| (RMSE 4.22 vs prior 3.06,
   −38%). The deck presents F11 as the "filter parameters diverge" slide, which is right, but the
   report's storyboard attributes the periodic fix to localization — localization did *not*
   prevent this divergence.
6. **Deck-internal metric mixing.** `main.tex` is the "CRPS variant": its tables quote energy
   scores (I verified every one against `sensor_metrics.*.velocity_vector_energy_score` — F14
   0.174→0.17, F13 0.097/0.203→0.10/0.20, E12 0.245/0.234→0.25/0.23, E13 0.698/0.871→0.70/0.87,
   H13 0.196/0.764→0.20/0.76 ✓). But the **storyboard cards on the same slides quote RMSE**:
   "held-out 0.57→0.32" (= F14→F13 RMSE), "0.56→0.37" (= F9→E12 RMSE), "assim. 1.09→0.31"
   (= E13→H13 RMSE). Two metric scales in one deck, unlabelled.
7. **The storyboard and the slides cite different runs for the same claim.** Case 2's card uses
   F9 (localized joint EnKF, valid 0.559) as "the joint filter", while the case-2 slides show
   F16 (unlocalized, valid 0.624) and then E19/E12.
8. **`presentations/isda_crps/latex/scripts/numbers.json` is stale.** It contains only RMSE, is
   not the source of any table in `main.tex`, and includes runs (E11, F10, F17, F18, F21) the deck
   never shows. The deck's tables are not reproducible from the committed numbers file.
9. **`setup.tex` vs configs**: "50 s spin-up" vs `spinup_time: 150.0`; "one knot per 30 s,
   i.e. 15 knots over 360 s" vs 4 knots per 120 s window with the window's first knot pinned
   (`pin_initial_time_point: true`).
10. **`edge frac` under-detects bias.** F15's held-out rank histogram
    (`rank_valid_F15.png`) has a pronounced left spike (~760 counts in the 0–5 bin vs ~282
    uniform) — the truth sits systematically at the low end of the ensemble — yet its
    `edge frac` reads 0.024 (nominal 0.04, "calibrated"), because the metric looks only at the
    two outermost of 51 bins. The deck uses this pair of histograms as the evidence that RTPS
    fixes calibration.
11. **Obs-interval sweeps do nothing** (E17/E13/E15 α = 12.94/13.62/13.63°; valid
    1.394/1.408/1.417) — consistent with §3.2's "no information", but the report and deck present
    these as a tuning axis.
12. **`experiments_report/main.tex` says "Generated from `presentations/isda_new/experiments/`"**
    while the deck lives in `isda_crps` and there are six near-duplicate deck trees
    (`isda`, `isda_final`, `isda_final_crps`, `isda_new`, `isda_new_sensors`, `isda_crps`). Only
    `isda_new/experiments/` holds run data.

---

## 5. Five experiments that would discriminate the hypotheses

Each is one run/config change, with the outcome expected under each competing hypothesis.

1. **Rescale the random walk per parameter** — `filtering/evolution/random_walk` with
   `std: {inflow_angle: 0.5, velocity_magnitude: 0.025}` (both ≈ 0.05 prior-σ per cycle), rerun
   F13/F16/F11. *H1 (mis-tuning, §4.1):* |U| skill jumps from ~0% to 30–50% on the inflow cases
   and F11's |U| RMSE drops from 4.22 toward the 0.92 prior; held-out RMSE improves by 10–25%.
   *H2 (the observations genuinely carry no |U| information for a filter):* |U| RMSE tracks the
   prior (~0.9) with no divergence — still a win (no blow-up) but no skill gain. Cheapest and
   highest-value experiment in the list.
2. **Perfect-inlet twin for the turbulent case** — one `inflow_turb` run in which the truth and
   every ensemble member share one synthetic-eddy seed (`inlet_turbulence.seed: 12345` on both
   `truth_model` and `assim_model`), everything else identical to E12/F9.
   *H1 (irreducible realisation error ≈ 0.31 dominates, §3.3):* held-out RMSE falls from 0.365
   (E12) / 0.559 (F9) to ≈ 0.20 and 0.35 — i.e. by roughly √(x²−0.31²).
   *H2 (the error is method-limited, not realisation-limited):* held-out barely moves (<10%).
   This is the single decisive test of the "can DA ever beat it" question.
3. **Long-window / time-averaged periodic identifiability probe** — one `periodic` ESMDA run with
   `time.simulation_time=600`, `esmda.interval_seconds=120` (5 bins), `num_assimilation_windows=1`,
   `truth_start_time=60` (drop the ramp). *H1 (identifiable only in the time mean, §3.2):* the
   data mismatch finally moves (O_N 37 → <10) and α RMSE drops below 8°.
   *H2 (structurally unidentifiable — nudging influence is too weak at the sensors):* O_N stays
   ~37 and α stays 13–15°, and the honest conclusion is "do not estimate inflow parameters under
   periodic BC", which is what the report already recommends.
4. **Free-run control ensemble (the missing baseline)** — 50 members drawn from the prior, no
   assimilation, one per case, scored with the same metric pipeline. *H1:* it reproduces my
   computed climatology (held-out 1.38/1.41/1.61) and confirms the periodic methods are at 23%
   skill. *H2:* it is *worse* than climatology (likely, since the prior mean |U| is not the truth
   mean), in which case every reported skill number is understated and the periodic result is less
   damning than §3.1 says. Either way the campaign currently has **no no-DA control**, and every
   "skill" claim in the deck is implicitly against a baseline nobody ran.
5. **Trim the spin-up and re-seed** — rerun the three headline cells (F13, E12, H13) with
   `run.truth_start_time=60` and seeds {42, 43, 44}. *H1 (results are robust):* held-out RMSEs
   rise ~5–10% uniformly (the ramp deflation of §3.4.4) and the seed spread is <10%, so the
   rankings and the H13-vs-F12 "tie" stand. *H2 (single-seed noise is driving the story):* the
   seed spread exceeds the 1–5% differences the report calls wins (H13 1.246 vs F12 1.261), and
   several storyboard claims collapse into ties.

---

### One-line bottom line

The campaign is honestly reported and the laminar/turbulent skill is real (74–77% vs
climatology, and turbulent ESMDA is within ~20% of a realisation floor it cannot cross), but the
periodic result is 23% skill against an oracle-climatology baseline that nobody ran, the joint
filter's inability to estimate speed is traceable to a single unscaled random-walk σ rather than
to physics, and no configuration was repeated, tuned per case, or scored with the spin-up removed.
