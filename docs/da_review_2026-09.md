# Data-assimilation state review — September 2026

> **Status: review record, written 2026-09-04** on branch `isda_experiments`
> at `7ab1c6d`. Read-only analysis; no code was changed. Supporting audits
> (one per area, with `file:line` anchors) live in
> [`docs/da_review_2026-09/`](da_review_2026-09/). This is a snapshot, not a
> maintained reference — verify against the tree before acting on a line
> number.

The question asked: *what is implemented and why; what is unused, outdated,
badly implemented, or unlikely to help; which three indicators are missing
from the analysis; and which routes would improve held-out state estimates,
especially for turbulent inflow and periodic boundary conditions.*

The evidence base is the ISDA 2026 campaign
(`presentations/isda_new/experiments/`, 61 runs: E\*/F\*/H\* IDs as in
`experiments_report/`), the `isda_crps` deck, and the full source tree.

---

## 0. Summary

1. **The library is sound; the configuration is not.** The ETKF/LETKF kernel,
   the stochastic update, the localization plumbing and the evaluation library
   are carefully written and tested. The results are limited by five
   configuration-level defects and one modelling gap, all confirmed in code:
   * The observation error is pure instrument noise (`σ_o = 0.1 m/s`) with
     **no representativeness term**, although truth and members are different
     turbulence realisations. ESMDA's own data-mismatch diagnostic ends at
     `O_N = 2.4–77` against a target of 0.5 in every campaign run, and the
     filter drives its observation residual to 0.05 m/s, *below* `σ_o`. This
     is the over-fitting that shows up as good assimilated-sensor and poor
     held-out-sensor error.
   * The filter's parameter random walk is one scalar, `std = 0.5`, applied
     to both angle (prior σ 10°) and speed (prior σ 0.5 m/s) every 2 s cycle.
     Speed is re-randomised by a full prior σ per cycle. This alone explains
     why the joint filter never learns |U| and why it diverges on periodic
     (F11: 4.2 m/s; F3 under PALM: 21 m/s).
   * In ESMDA, observations are perturbed per raw frame and then 15 s-averaged,
     while `C_D` is sized after averaging: the injected noise is `σ²/7.5`,
     `C_D` says `σ²`. The two error models disagree, and neither describes the
     actual mismatch.
   * With `block_grouping: true` (the shipped default), all time knots of a
     parameter share one localization block, so the taper follows the
     *strongest* knot and knot-wise temporal localization is lost.
   * Truth and all members share uDALES `irandom = 43`; no config writes it.
     On the laminar and periodic cases the ensemble is a pure
     two-parameter family with no realisation spread.
   * In the periodic case the "inflow" parameters nudge only the horizontal
     slab mean above 16 m (`nnudge_meters: 16`, dz = 2 m), while all six
     sensors sit at z = 2 m in canyons. The parameters are not inert — the
     bias runs rotate the domain mean 1:1 — but only ~9 % of the sensor
     response to a parameter change is a coherent mean shift. That is an
     observability problem, not a filter problem.
2. **The performance picture, once baselines are added**, is: laminar and
   turbulent inflow are genuinely solved at ~75 % skill against
   climatology; turbulent-inflow ESMDA is within ~20 % rms of the
   realisation floor set by the unobserved inlet eddies (~0.31 m/s); the
   periodic case reaches 23 % skill at best and every unlocalized periodic run
   is *worse than climatology*. No run in the campaign has a no-assimilation
   control, a perfect-parameter free run, or a repeated seed.
3. **Roughly 12 k lines are unused, stale, or cannot help**: the whole
   state-reduction path (provably identity-or-loss), the observation TSVD
   variants, `final_time_smoothing` (assimilates the data twice by its own
   docstring), `compare_models.py`, `run_probe_series.py` (pylbm-only), the
   `figspec`/`figure_creation` pipeline bolted to a dead HPC root, several
   broken `adjust_simulations` scripts, and an 805-line nudging test file that
   pytest has never collected.
4. **Three missing indicators**: (i) skill relative to reference runs
   (climatology, prior free run, perfect-parameter free run, realisation-floor
   ensemble); (ii) spread–skill ratio over time per sensor set; (iii) the
   per-cycle forecast/analysis decomposition with error-regrowth rate. The
   library already contains most of the code for all three.
5. **Highest-value routes**, in order: measure the floor and add controls;
   make `R` honest (instrument + representativeness, checked with Desroziers);
   fix the parameter forecast model (per-parameter σ, mean reversion, log-|U|);
   move two sensors to roof level; re-pose the periodic control as bulk
   forcing; estimate the inlet realisation (statistics first, driver-plane
   latent state later). Re-tuning localization, inflation or reduction on the
   current setup is capped: the ensembles are already calibrated at held-out
   sensors (validation z-score std 0.88–1.22), so the residual is missing
   information, not mis-weighted information.

---

## 1. What is implemented, and what it is for

### 1.1 The DA library (`libs/data-assimilation`, ~8 k lines)

| Module | Purpose | Reached by the campaign? |
|---|---|---|
| `smoothing/esmda.py` | ESMDA: five variants (parameter, time-varying parameter, state, state+parameter, state+time-varying). MDA loop with tempered stochastic updates; on-disk per-step forecast management. | `dynamic` (time-varying parameter) only. The three state-bearing smoothers never ran. |
| `filtering/base.py` | Sequential EnKF cycle: flatten → optional basis → prior inflation → serial frame sweep → posterior inflation → diagnostics → parameter evolution. 40-field `CycleDiagnostics`. | Yes; `mode=state` (29 runs) and `joint` (12). `mode=parameter` never. |
| `filtering/analysis.py` | The one stochastic-EnKF update (`x + C_MD (C_DD + αC_D)⁻¹(y + √α ε − Hx)`), Cholesky solve, centred perturbations. Shared by ESMDA and the filter. | Yes. |
| `filtering/etkf.py` | ETKF / LETKF, observation TSVD, whitening, block dedup. The best-written module in the library. | **Never run.** The two benchmark records in `docs/temp/` are still "not run". |
| `filter_smoothing/base.py` | Hybrid: ESMDA on the window's parameter trajectory (no final forecast) + filter on state, with the filtered state feeding the next window's prior. | Yes (20 runs, `dynamic × state`). |
| `localization/` | Vossepoel-style taper (`E_inf = exp(((d−βT)/b)²)`), correlation distance (`1−|ρ|`) or Euclidean distance, block grouping, row-wise localized stochastic update via `vmap`. | Correlation only. Distance never. |
| `inflation.py` | Multiplicative, RTPS, RTPP. Filter only; the smoother has no inflation hook. | RTPS (38 runs) or none. |
| `reduction.py` | Online SVD basis on the current anomalies; streaming POD with forgetting. Projects the increment. | Never. |
| `augmentation.py` | Flatten/unflatten of parameters (knots → scalars) and state; block ids for localization; row coordinates. | Yes. |
| `observation_operator.py` | Trilinear sensor sampling, temporal operator, interval aggregation (mean/median/max/min), flattening. | Aggregation only on the ESMDA/hybrid path, `mean` only. |
| `filtering/parameter_evolution.py` | `RandomWalkEvolution` (scalar or per-name std), `IdentityEvolution` (no config). | Random walk with the scalar default. |

### 1.2 Entry points and evaluation

| Script | Role | Campaign use |
|---|---|---|
| `scripts/run_forward_model.py` | Truth generation (`params=dynamic_sine`), bias and Barcelona forward runs. | Yes. |
| `scripts/esmda/run_esmda.py` + `compute_esmda_metrics.py` + `make_esmda_figures.py` | Smoother pipeline → `run_summary.yaml`, `eval_fields.nc`, figures. | Yes (20 runs). |
| `scripts/filtering/run_filtering.py` + metric/figure stages | Filter pipeline, plus an `esmda_view/` symlink farm so the ESMDA stages can score filter windows. | Yes (21 runs). |
| `scripts/filter_smoothing/run_filter_smoothing.py` | Hybrid; reuses the filtering stages. 59 % of its `run()` body is identical to `run_filtering.py`. | Yes (20 runs). |
| `libs/evaluation` | Fair CRPS / energy score, rank histograms, finite-M z-scores, contraction ratio, VDI hit rate, `O_N` data mismatch, Welch spectra, spread–skill (implemented, never called). | Partly surfaced (see §5). |
| `scripts/compare_models.py`, `scripts/esmda/run_probe_series.py`, `scripts/figure_creation/*`, `scripts/figspec/*` | Older campaigns' tooling. | No. |

### 1.3 What the campaign actually exercised

The plan in `docs/plans/isda2026_talk_experiments.md` was not what ran. All 61
runs use `case=xie_and_castro`, `params@truth_params=dynamic_sine`,
`esmda/smoother=dynamic`, `filtering/analysis=stochastic`, N = 50, one seed
(42), a 2 s filter cycle (180 analyses over 360 s) versus one ESMDA analysis
per 120 s window on 96 15-s-binned observations with three MDA steps.
Everything else that was built for the campaign — ETKF/LETKF/TSVD, both
state-reduction options, distance localization, the state-bearing smoothers,
`filtering.mode=parameter`, the planned three-seed repeats — did not run.
Six assimilated sensors (z = 2 m, u and v only, `σ_o = 0.1`), four held-out
sensors spanning approach flow, lane, near-wake and one roof-level point.

### 1.4 Backends from the DA side

Estimated everywhere: `inflow_angle`, `velocity_magnitude`. Wired and
estimable but excluded by `params_to_estimate`: `sgs_constant` (uDALES
`c_vreman`, pylbm, PALM `km_constant`), `vertical_inflow_exponent`. Sampled
but **inert on every backend**: `pressure_gradient_magnitude` (the body force
is written as zeros for every boundary condition). One small change from
estimable: uDALES synthetic-eddy `intensity` and `length_scale_{x,y,z}`.
Needs real code: nudging strength/height, roughness.

---

## 2. The performance picture

### 2.1 Reported numbers with the baselines nobody ran

Held-out (validation) 3-component vector RMSE in m/s, matched uDALES truth.
"Climatology" is the RMSE of predicting each sensor's own time mean, computed
from the stored truth files. `valid/assim` is the ratio of held-out to
assimilated-sensor RMSE.

| Case | Climatology | Best method | Held-out | Skill vs clim. | valid/assim | Notes |
|---|---|---|---|---|---|---|
| Laminar inflow | 1.378 | F13 joint EnKF + RTPS, no loc. | 0.319 | 77 % | 2.2 | ESMDA stalls (E11: 0.748); localization *hurts* the filter here (F7: 0.394). |
| Turbulent inflow | 1.413 | E12 ESMDA + loc. | 0.365 | 74 % | 0.9 | Best filter F9: 0.559. ESMDA is the only method whose held-out error is not worse than its fitted error. |
| Periodic | 1.611 | H13 hybrid + loc. | 1.246 | 23 % | 4.0 | State-only filter F12 ties (1.261). Every unlocalized periodic run is worse than climatology (−4 % to −127 %). |

Parameter results: angle is recovered by every method on the inflow cases
(2.3–5.3° from a 16–20° prior). |U| is recovered **only** by smoother-based
methods (E12: 0.37 m/s, 53 % skill); every joint-filter run reproduces its
0.92 m/s prior or diverges. On periodic no method moves either parameter.

### 2.2 What the evidence establishes

* **Over-fitting at the assimilated sensors is measured, not inferred.** Each
  filter analysis cuts observation misfit 5–9× (to 0.05 m/s, below `σ_o`)
  while reducing state spread by only 1–15 % (periodic: 1.1 %). ESMDA's
  `O_N` never approaches its target (laminar 17.5, turbulent 2.4, periodic
  37.2 with zero movement across the MDA steps).
* **Turbulent inflow is near its floor for ESMDA.** Truth and every member
  draw independent synthetic-eddy seeds (`derive_seed(experiment_name)`).
  Three independent estimates of the resulting irreducible held-out error
  agree at ≈0.31 m/s (variance decomposition of the two truths; configured
  `u'_rms ≈ 0.38`; ESMDA residual 0.22 on 15 s means). E12's 0.365 leaves
  ≈0.19 m/s reducible. The filters (0.56–0.62) are far above the floor, which
  is a filter problem (`R`, random walk), not an irreducibility problem.
* **Periodic is an identifiability failure at the assimilation timescale.**
  The bias runs show the parameters rotate the domain mean 1:1, but the
  instantaneous sensor response to a 5°/0.5 m/s change is 1.78 m/s of which
  0.53 is a coherent shift — larger than the truth's own variability, and 91 %
  chaotic divergence. A 120 s window with 15 s bins cannot average that out.
  The state half is separately, partly solvable: localized state updates buy
  the 23 %.
* **The ensembles are calibrated where they are wrong.** Validation-sensor
  z-score std is 0.88 (F13), 0.99 (F9), 1.22 (F12). Spread ≈ error at held-out
  sensors means the remaining error is absent information, so re-weighting
  the same 50 anomalies (localization radii, inflation constants, reduction
  rank) cannot remove it.

### 2.3 Confounds that weaken current conclusions

* Single seed everywhere; the plan asked for three on the headline cells.
  Differences of 1–5 % (H13 vs F12) are inside plausible seed noise.
* The periodic truth is still ramping from rest for the first ~40 s *inside*
  the scored 360 s (`run.truth_start_time` left null); all periodic RMSEs are
  deflated ~6 %.
* RTPS α = 0.6, ρ_t = 0.35, random-walk σ = 0.5 are identical in all 45
  cells. "Localization hurts laminar / helps periodic" is one setting's
  effect. Localization is also confounded with spread preservation: every
  unlocalized parameter-updating run has a collapsed parameter ensemble.
* ESMDA sees 96 aggregated observations per window, the filter 720 raw ones.
* `assim` RMSE is a post-update fit for the filter and a free re-simulation
  for ESMDA; the periodic slide's "1.09 → 0.31" compares the two.
* Static-parameter runs carry byte-identical prior values (19.29°, 0.919 m/s)
  as if they were estimates in `numbers.json`.
* The deck labels the energy score "CRPS" and quotes RMSE on the same slides'
  storyboard cards.

---

## 3. Root causes in the code, ranked

All items marked **confirmed** were verified by reading the code in this
review, not only reported by the audits.

| # | Finding | Where | Status | Effect |
|---|---|---|---|---|
| 1 | `R` is instrument noise only; no representativeness term; `O_N` 2.4–77 vs 0.5; filter residual 0.05 < σ_o. | `conf/run_*.yaml` `obs_error_std: 0.1`; `run_summary.yaml` `esmda_diagnostics.data_mismatch`, `filter_diagnostics` | confirmed | Over-fitting at assimilated sensors, `valid/assim` 2.2–4.9 for every filter. |
| 2 | Random walk `std: 0.5` is one scalar for angle (°) and speed (m/s). Speed gets 1.0 prior-σ of noise per 2 s cycle; angle gets 0.05. | `conf/filtering/evolution/random_walk.yaml`; `parameter_evolution.py:70-73` | confirmed | Joint filter never learns |U|; diverges on periodic; members below zero. The per-name mapping form already exists. |
| 3 | ESMDA perturbs raw frames then 15 s-averages, `C_D` sized post-aggregation: injected variance `σ²/7.5`, assumed `σ²`. Bins are unequal (7 vs 8 frames). | `run_esmda.py:876-885`, `:801` | confirmed | Two inconsistent error models. The accidental slack is the only representativeness allowance in the system, which is why ESMDA generalises better than it fits. Fix by making `R` explicit (item 1), not by shrinking it. |
| 4 | `block_grouping: true` puts all knots of a parameter in one block; the block takes `segment_min` of the inflation, so the taper follows the strongest knot. Knot-wise temporal localization is lost. | `augmentation.py:136-153`, `localization/base.py:64-95`, both `correlation.yaml` | confirmed | Correlation localization on the dynamic smoother is much weaker than reported. Observed E12 vs E19 difference (~10 %) is consistent with partial neutralisation. One-line config test: `block_grouping: false`. |
| 5 | Periodic parameters act through slab-mean nudging above 16 m only; body force zeroed for all BCs; `use_nudging` hard-coded `True`. | `forward_model.py:728-732`, `nudging_utils.py:384-390`, `conf/model/pyudales.yaml:53`, `modforces.f90:849-883` | confirmed | Two scalars controlling the top half of the domain, observed from z = 2 m. Not identifiable at 120 s. |
| 6 | Truth and every member share `irandom = 43`; nothing writes it. Initial state is copied into every member. | `modstartup.f90:42`, `random_utils.py:44` (never called from config), `hydra_helpers.py:158-164` | confirmed | Laminar/periodic prior covariance is a parameter-sensitivity covariance, not a model-error covariance. |
| 7 | RTPS alone does not maintain static-parameter spread (geometric decay); the spread guard accepts it, and `IdentityEvolution` passes the guard while doing nothing. | `filtering/base.py:415-425`; `run_filtering.yaml` defaults `evolution: none` + `rtps` for `mode: joint` | confirmed | Parameter collapse in unlocalized joint runs (z_pool 11–263). |
| 8 | The hybrid conditions parameters (ESMDA, full likelihood) and state (filter, full weight) on the same window's observations; total likelihood weight 2. Undocumented. | `filter_smoothing/base.py:581-592`, `:634`, `:716` | design property | Overconfident joint posterior; hybrid χ² 13–23 on laminar. |
| 9 | Parameter evolution noise is applied after the analysis and stored as the posterior; the reported spread is pre-evolution. The hybrid's persistent correction inherits the noise. | `filtering/base.py:1194-1196`, `:1149`; `filter_smoothing/base.py:729` | confirmed | Reported posterior and reported spread describe different objects. |
| 10 | Diverged members are replaced by donor clones *after* the covariance is formed; duplicates enter `C_MD`/`C_DD` with no diagnostic. | `esmda.py:394-413`, `filtering/base.py:870-921` | smell | Understated spread when failures occur. |
| 11 | Localized stochastic update materialises `(N_aug, N_d, N_d)`: 8.5 GB for a state-bearing ESMDA at `N_d = 96`, 120 GB on Barcelona. No guard. | `localization/base.py:386-396` | smell | The unused state-bearing smoothers will OOM if switched on with localization. |
| 12 | Periodic spin-up inside the scored record; `spinup_time` documented as 50 s, configured 150 s; `truth_start_time` null. | campaign `config.yaml`, `setup.tex` | confirmed | Periodic numbers deflated ~6 %. |

---

## 4. Unused, outdated, badly implemented, or unlikely to help

### 4.1 Cannot help on this problem (opinion, with the argument)

| Item | Why | Recommendation |
|---|---|---|
| State-space reduction (`reduction.py`, 734 lines; `esmda/state_reduction=svd`, `filtering/state_reduction=svd_current|svd_streaming`) | The basis is fitted on the same anomalies it projects, so `U_r ⊆ col(X)`. At full rank it is the identity; below full rank it strictly removes update directions. It cannot suppress sampling error, and the analysis cost it saves is negligible next to 50 CFD runs. | Drop, or keep the seat for an offline climatological basis (which *can* add rank). Do not spend the planned six-run "reduction ladder". |
| Observation TSVD (`etkf_tsvd`, `letkf_tsvd`) | `N_d = 12`, `N_e = 50`; nothing to regularise; the YAML comments say so. | Delete or fold behind a flag. |
| LETKF at six sensors | ~95 % of rows see no observation; dedup collapses to a handful of blocks ≈ global ETKF. | Low priority to benchmark. |
| Correlation localization as tuned | ρ_t = 0.35 vs noise floor 0.14 at N = 50 keeps thousands of spurious links; regime-dependent (+23 % harm on laminar, −47 % on periodic) is the signature of a mistuned free threshold. Plus item 4 of §3. | Replace with a sampling-error-corrected table (Anderson 2012) or a temporal kernel on the knots. |
| `final_time_smoothing` | Assimilates the window's data a second time at α = 1; the docstring says it must not be used for UQ; requires state reduction and in-memory mode. | Delete. |
| `basis_source=window_snapshots` | SVD of `230k × 3000`; infeasible at production size and a projector cannot add rank. | Delete. |
| Hybrid observation-interval sweep | Only changes the smoother's bin while the filter still assimilates every 2 s; the report calls it a placebo. | Stop. |
| `assim` RMSE as a headline | Rewards exactly the over-fitting mode identified here. | Move to an appendix next to `valid/assim`. |

### 4.2 Dead or unreachable code

* `hydra_helpers.{create_C_D, make_time_coords, create_initial_state_ensemble}`,
  `get_ensemble_mean_field` + `_BaseESMDA.get_state`, the index-based
  `ObservationOperator` mode, `IdentityEvolution` (no config),
  `AggregateObservations` `median/max/min`, `variable_scales` / `row_scales`.
* `scripts/_common.py:198-370` (173 lines), `scripts/esmda/_esmda_common.py:170-256`
  (87 lines), `run.ground_truth_dir`, `obs.mode=grid`, `import pdb` in
  `run_forward_model.py`.
* `src/pyurbanair/base_rollout_forward_model.py` and
  `pyudales/utils/rollout_utils.py` (zero importers; documented legacy).
* uDALES static/periodic inflow branch `forward_model.py:776-782` and 50
  commented-out lines in `nudging_utils.py:391-440`.
* `conf/params/static.yaml` `pressure_gradient_magnitude` (inert on every
  backend), `params/dynamic_cosine` (only reachable via `compare_models.yaml`),
  `inflation/{rtpp,multiplicative}` config groups (never selected).

### 4.3 Whole tools that are stale or orphaned

| Target | Lines | Status |
|---|---|---|
| `scripts/compare_models.py` + `conf/compare_models.yaml` | ~2600 | Zero callers, zero tests. |
| `scripts/esmda/run_probe_series.py` + config + `probe_*` helpers | ~1400 | pylbm-only; cannot run on a uDALES/PALM artifact, so `spectral_metrics` is empty in 0/55 runs. |
| `scripts/figspec/` + `figure_creation/make_*` (7 scripts) | ~3500 | One pipeline bound to `/projects/prjs2075` and a run-naming scheme the campaign does not use. |
| `figure_creation/{compute_sweep_metrics,compare_sweep_results,compare_state_runs,compare_param_vs_state,visualize_state_run}.py`, `compare_localization.sh` | ~3000 | Read stores that no longer exist or a retired campaign; `compare_localization.sh` never runs the metric stage so its table always prints n/a. |
| `adjust_simulations/{regenerate_ground_truth_params,convert_ground_truth_to_32bit,make_state_small}.py` | ~250 | Broken imports/paths after the directory move (four independent breakages in the first). |
| `presentations/isda_final_crps/latex/scripts/` | ~600 | Byte-identical fork of `isda_final`'s script that **writes into `isda_final`'s figure dir**. |
| Six `job_scripts/*` entries | — | Call pre-move `scripts/*.py` paths and fail immediately. |
| `tests/_nudging_utils.py` | 805 | 24 tests of the uDALES nudging path, never collected (name does not match `test_*.py`) since April. |

### 4.4 Duplication and drift

* `run_filtering.py` vs `run_filter_smoothing.py`: 252 identical lines (59 %
  of the smaller body) including a 37-line verbatim truth/dirs block that has
  already drifted on which keys land in `run_info.yaml`. The two pipeline
  `.sh` files are ~95 % identical.
* Three copies of on-disk step/cycle plumbing and four copies of "sort files,
  load, concat" across `esmda.py`, `filtering/base.py`,
  `filter_smoothing/base.py`, `smoothing/base.py`.
* `make_{esmda,filtering}_figures.py` share 66 byte-identical helper lines.
* The three inlined run configs drift: `run_filter_smoothing.yaml` ships
  `ensemble_size: 40` (others 50) and defaults the smoother half to
  `localization=none` (pure smoother defaults `correlation`); three names for
  the window count.
* Commit `7ab1c6d` baked campaign tuning into the shared entry-point and case
  configs; `conf/run_filtering.yaml:192` now ships `save_forecast_history: true`
  against its own comment and `tests/test_run_filtering.py:554`, so the suite
  is red on the current tree.

### 4.5 Docs

Current: `docs/data_assimilation.md`, `docs/ensemble_transform_filters.md`,
`docs/pyudales.md` §6. Stale: `docs/codebase_guide.md` ("the two assimilation
entry points", no mention of `filter_smoothing`), `conf/README.md` and
`README.md` (`filtering.num_cycles`, a key that does not exist), nearly every
default value in `docs/scripts_and_configs.md`,
`docs/plans/isda2026_talk_experiments.md` (superseded, not implemented as
written), `docs/plans/srst_sgs_parameterization.md` ("cs is inert" is fixed).

---

## 5. Three crucial indicators you are not using

The evaluation library already computes far more than the deck shows (fair
CRPS/energy score, rank histograms, z-scores, contraction ratio, `O_N`,
Desroziers-ready `pred_obs` files, TKE fields, a tested `spread_skill()` with
no caller). The three below are the ones whose absence makes the current
conclusions uninterpretable. Each is stated as: definition, question it
answers, expected reading per case, implementation.

### 5.1 Skill against reference runs, including a realisation-floor ensemble

```
R_clim     = RMSE of the truth's own time mean at each sensor           (from truth.nc alone)
R_prior    = held-out RMSE of the prior parameter ensemble, no DA         (50 forward runs)
R_perfect  = held-out RMSE of a free run with the TRUE parameters         (1 run, or M runs)
R_floor    = held-out RMSE across members with TRUE parameters and
             varied inlet seed / irandom                                   (10–20 runs)
SS_ref     = 1 − RMSE_method / RMSE_ref ;  CRPSS = 1 − CRPS_method / CRPS_R_prior
```

*Question.* Is 0.36 m/s good? For a parameter-only ESMDA, `R_perfect` is the
ceiling of the method class by construction. `R_floor` is the ceiling of any
method that does not estimate the realisation. Without them, "held-out ≈
prior" and "modest accuracy" are unquantified. Today 0 of 55 runs store a
prior state reference (`save_prior_state: false` in config, hard-coded
`False` in the filter and hybrid).

*Expected reading.* Laminar: large positive skill against everything.
Turbulent: `SS_floor ≈ 0` for ESMDA, `SS_prior ≫ 0` — the residual is
irreducible, which reframes "modest" as "at the floor". Periodic:
`R_prior ≈ R_perfect` (no parameter signal), so the only skill available is
from the state update, and the hybrid's advantage becomes expressible.

*Implementation.* `R_clim` and persistence need no run. `R_prior` /
`R_perfect` / `R_floor` are `run_forward_model.py` runs (the floor needs a
~10-line `irandom` write in `_apply_inflow_settings` and
`inlet_turbulence.seed` per member), scored by a small script that reuses
`build_sensor_sets`, `truth_sensor_series`, `vector_sensor_metrics`,
`streaming_state_rmse` verbatim. Flip `save_prior_state=true` and un-hard-code
it; `crps_reduction_vs_prior` and `hit_rate_prior` then populate for free.

### 5.2 Spread–skill ratio over time, per sensor set, forecast and analysis separately

```
spread(t) = sqrt( mean_s mean_c var_ens v_c(t,s) )
rmse(t)   = sqrt( mean_s mean_c (⟨v_c⟩ − v*_c)² )
SSR       = sqrt((M+1)/M) · RMS_t spread / RMS_t rmse        (Fortin et al. 2014)
```

*Question.* Did a method fail because it is wrong or because it is
over-confident, and *when* did it become over-confident? The current
calibration numbers cannot say: `zpool` is the sample std of 6 parameter
z-values in the filter runs (statistically empty at that n), `zval` is one
number per run pooled over window means, and the rank histograms pool
statistics and quantities so shapes cancel. SSR puts spread and error on the
same m/s axis as the RMSE the deck reports, and turns "RTPS helps" from a
0.37 → 0.30 RMSE claim into a calibration claim.

*Expected reading.* Laminar without inflation: SSR ≪ 1 and falling (χ² 3.98
already hints). Turbulent: ≈1 at assimilated sensors, ≪1 at held-out — the
signature of missing model error. Periodic: ≪1 everywhere, with the hybrid's
parameter side least collapsed.

*Implementation.* ~25 lines. `compute_sensor_metrics` already returns
`members (E,T,S)`, `ens_mean`, `truth`, `rmse (T,)`; add the spread series in
`vector_sensor_metrics` (`scores.py:708`), call the existing `spread_skill()`
(`scores.py:322`), store `sensor_metrics[set].spread_skill`, add a panel to
`sensor_fans.png`. No rerun for any run whose `windows/` survives
(`.temp/filtering_pyudales_to_pyudales/` does).

### 5.3 Forecast/analysis decomposition per cycle and error-regrowth rate

```
E_set(k, τ)  = held-out / assimilated RMSE at lead τ inside cycle k   (τ=0 analysis, τ=T next forecast)
r            = E(k,T) / E(k,0)        regrowth ratio, averaged over cycles
τ_e          = T / ln r               e-folding time of the analysis increment
```

*Question.* The central puzzle — fit 0.05 m/s at assimilated sensors,
0.3–1.5 m/s at held-out — has two explanations the current metrics cannot
separate: (i) the increment never reaches the held-out sensors (localization
or `B` has no correlation there); (ii) the increment is global but transient,
regrowing within one 2 s cycle because the cycle exceeds the predictability
horizon of the resolved eddies. `E(τ)` separates them: (i) gives
`E_heldout(0) ≈ E_heldout(T)` with `E_assim(0) ≪ E_assim(T)`; (ii) gives both
sets dropping at τ = 0 and both recovering by τ = T. It also answers whether a
2 s cycle is sane, and the "sawtooth" it draws is the standard first figure of
any sequential-DA paper.

*Expected reading.* Laminar: `r ≈ 1.2–1.5`, long `τ_e`. Turbulent: `r → 1`
at held-out sensors even when the analysis lands — fresh stochastic forcing
every cycle. Periodic: `E_heldout(0) ≈ E_heldout(T)` — the increment is local
and spurious. That would be the strongest single slide in the deck.

*Implementation.* Tier 0, no rerun: `cycle_diagnostics.yaml` already holds
180 × (`obs_prior_rmse`, `obs_posterior_rmse`); plot them and report the
ratio (measured now: 0.33 laminar, 0.20 turbulent, **0.12** periodic). Label
the posterior rows "unlocalized ride-along". Tier 1: 37 runs already store
every forecast frame of every member in `windows/window_*_forecast_state.nc`;
`cycle_sensor_series` extracts `(component, ensemble, time, sensor)` and
today hands it to `window_statistics`, which bins time away. Reshape
`time → (cycle, lead)` instead. ~120 lines across the two metric stages.

### 5.4 Already computed, never shown (surface these first, zero compute)

* ESMDA `O_N` per step and the `underfit_final` flag (`data_mismatch_decay.png`
  is drawn in every run dir and appears in no deck).
* `esmda_view/run_summary.yaml` for the filter/hybrid runs
  (`extract_comparison.py` reads the run root only, so the `onp` column is
  empty for every non-ESMDA run).
* Held-out energy score, `sensor_statistics[set].posterior.variance_*` (is the
  filter destroying resolved variance?), per-component VDI hit rate (w fails
  at 0.36 where the pooled q reads 0.61).
* Desroziers consistency and the representativeness residual
  `RMS(H(x^b) − obs_clean)` — everything needed is in
  `windows/window_*_{obs,pred_obs}.nc`; ~80 lines, no rerun.
* Held-out identifiability map: `max_j |ρ(v_heldout, H_j x)|` is computed
  inside `CorrelationLocalization` and discarded.
* TKE ratio posterior/truth off `eval_fields.nc`.

Also fix the definitional issues: parameter RMSE is a per-member RMSE
(floor = posterior spread) while sensor/state RMSE are ensemble-mean errors;
the deck prints both in one table. Rename the energy score in
`numbers.json`; dagger the static-parameter rows; exclude spin-up.

---

## 6. Routes to better held-out state estimates

Ranked by expected gain per unit of effort. The first three are
prerequisites for interpreting anything else.

### Tier 1 — cheap, do first

| # | Route | Hypothesis it tests | Change | Effort | Expected effect |
|---|---|---|---|---|---|
| 1 | **Realisation floor + no-DA control + prior baseline** (§5.1) | How much of held-out error is achievable at all. | `irandom` write in `pyudales`; `inlet_turbulence.seed` per member; baseline script. | 2–3 days incl. compute | No accuracy gain; every other route is measured against it. |
| 2 | **Honest `R`: instrument + representativeness**, calibrated from the floor ensemble and cross-checked with Desroziers on stored `pred_obs`; same `C_D` semantics on both paths (fix the raw-frame/aggregated mismatch). | Over-fitting at assimilated sensors (`valid/assim` 2–5, `O_N` ≫ 0.5). | `conf/*` `obs_error_std` → spec; `C_D` construction in `filtering/base.py` and `run_esmda.py:801`. | 1–2 days | Turbulent filter should approach ESMDA's 0.365; periodic largest relative change. Success = `valid/assim → 1` at lower held-out RMSE, χ² ≈ 1. |
| 3 | **Fix the parameter forecast model**: per-parameter std as a fraction of prior σ (mapping form already supported); then an OU/AR(1) evolution reverting to the climatological prior with `L_corr ≈ 200 s`; estimate `log|U|` (anamorphosis) so speed cannot go negative. | The unit-blind random walk (§3 item 2). | `random_walk.yaml` string (immediate); `OUEvolution` + transform hook in `augmentation.py` (2–3 days). | trivial → 3 days | Joint filter learns |U| for the first time (target ≤ 0.4 vs 0.92 prior); no periodic divergence; joint ≥ state-only. |
| 4 | **`block_grouping: false`** on the dynamic smoother, then a temporal localization kernel on `|t_knot − t_obs|` via the existing `taper_inflation`. | Knot-wise localization is neutralised (§3 item 4). | Config; ~50 lines. | trivial → 1 day | Sharper time-varying parameter tracking; test on E12/E19. |
| 5 | **Experiment hygiene**: `truth_start_time` past the periodic ramp; three seeds on headline cells; per-case ρ_t/α at least once; `save_prior_state=true`; `irandom` varied per member so laminar/periodic ensembles have realisation spread. | Single-seed, spin-up, and confounded technique claims. | Config + compute. | compute only | Conclusions that survive a referee. |

### Tier 2 — change what is estimated or observed

| # | Route | Hypothesis | Change | Effort | Expected effect |
|---|---|---|---|---|---|
| 6 | **Move two of six sensors to roof level** (z ≈ 20–25 m), chosen by an ensemble observability score (the correlation matrix `CorrelationLocalization` already forms); for turbulent inflow, one sensor just downstream of the inlet. | Six z = 2 m canyon sensors are nearly orthogonal to the nudged slab mean and deep inside the canopy. The one roof-level held-out sensor is tracked by every method. | Case config + a placement script. | low; re-runs are the cost | Periodic parameter RMSE below the prior for the first time; large gain on turbulent. |
| 7 | **Re-pose the periodic control**: bulk-velocity forcing (`luvolflowr`/`lvvolflowr`) or the large-scale pressure gradient `(dpdx, dpdy)` (`angle_to_pressure_gradient` exists and is dead), or nudging-profile EOF coefficients; lower `nnudge_meters` toward the canopy. | The current periodic parameters are indirect and unobservable (§3 item 5). | `nudging_utils.py` (stop zeroing the body force), `forward_model.py`, `conf/params`. | 2–4 days | Converts an unidentifiable control into an identifiable one; prerequisite for any parameter-borne periodic gain. Confirm with the `σ_θ/σ_floor` signal-to-noise from route 1 first. |
| 8 | **Assimilate statistics** on the filter path: route filter observations through the existing `AggregateObservations`, add a second-moment mode so the observed vector is `(mean, rms)` per sensor per window, with `R` matched to the aggregation. | Instantaneous point values are mostly realisation noise; the ESMDA interval sweep already shows 7.5 s worst, 15–30 s best. | New `conf/filtering/observations` group; `_cycle_observations`. | low | All cases; largest on periodic. Success = interval-mean held-out RMSE improves while instantaneous is flat. |
| 9 | **Estimate the inlet realisation** (turbulent case). Tier a: estimate `intensity`, `length_scale_*` from sensor statistics (one `_resolve_inlet_turbulence(params)` mirroring the nudging resolver). Tier b: augment the state with the driver-plane AR(1) latent field projected onto 20–50 y–z modes; the generator `b_n = a b_{n−1} + √(1−a²) η_n` is a known linear-Gaussian forecast model, the textbook augmented-state case. Needs the seed-replay in `inlet_turbulence_utils.py:456-482` to become injectable. | The ≈0.31 m/s floor. | `inlet_turbulence_utils.py`, a third augmentation block, filter mode handling. | a: low; b: 1–2 weeks | The only route that can beat the floor on turbulent inflow. |
| 10 | **Run the state-bearing ESMDA that already exists** (`smoother=state_and_dynamic`), *without* `final_time_smoothing` and, until the `(N_aug, N_d, N_d)` intermediate is chunked or deduplicated as LETKF does, without the localized stochastic update. | Neither method performs a window-consistent 4-D state update; ESMDA leaves periodic state at climatology, the filter does not generalise. | Config; memory guard. | near zero to run | Periodic ESMDA held-out < 1.25 at `valid/assim < 1.5`; laminar phase drift fixed. |

### Tier 3 — algorithmic, only after the diagnostics say sampling is binding

| # | Route | Notes |
|---|---|---|
| 11 | Sampling-error-corrected localization (Anderson 2012 table for N = 50) as a drop-in `BaseLocalization` subclass; scale-dependent localization (low-pass the increment) to stop injecting unbalanced small scales; localize parameter rows separately from state rows. | Removes the free ρ_t and the regime-dependence. |
| 12 | Adaptive inflation closing the already-computed `innovation_chi2` loop (Anderson 2009 / El Gharamti 2018); additive inflation from a climatological library (surrogate training data exists); hybrid covariance `βP_ens + (1−β)P_clim` in the seat state reduction currently occupies. | Mostly for the PALM model-error cells and unlocalized periodic. On matched twins the ensembles are already calibrated at held-out sensors, so returns are capped. |
| 13 | Non-uniform / adaptive MDA schedule (`α = [8,4,2,1.6]`, Rafiee–Reynolds); currently forbidden by the `num_steps/alpha == 1` check. | Standard remedy for a first over-correction on a nonlinear model. |
| 14 | Model-error parameters: re-enable `sgs_constant` (already wired; `c_vreman` sits at its stability floor), roughness, or a low-dimensional additive bias field. | PALM-truth cells only; risk of extra spurious DOF at N = 50. |
| 15 | Ensemble-size probe at N = 100–200 on one case, possibly via pylbm/surrogate multi-fidelity. | If held-out skill barely moves, observability (routes 6–9) is binding, not sampling (11–12). Worth doing early because it redirects effort decisively. |

### What to stop doing

The state-reduction ladder, the TSVD variants, the hybrid interval sweep,
per-case localization/inflation sweeps as results, `assim` RMSE as a headline,
and the 45-cell method zoo at one seed. The same compute spent on routes 1,
5 and 6 with two methods would produce conclusions that hold.

---

## 7. Suggested sequence

1. **Week 1.** Route 1 (floor, control, prior baseline) and the zero-compute
   surfacing in §5.4 (O_N, sawtooth from `cycle_diagnostics.yaml`, Desroziers
   off stored `pred_obs`). Also the one-line hygiene items: rename
   `tests/_nudging_utils.py`, fix `save_forecast_history`, move campaign
   tuning out of the shared configs.
2. **Week 2.** Routes 2 and 3 (honest `R`, parameter forecast model) — they
   interact, so tune together; add the SSR metric (§5.2) and re-run the three
   headline cells with three seeds and the spin-up trimmed.
3. **Week 3.** Route 4 (block grouping / temporal kernel), route 6 (sensor
   placement study + one re-run per case), route 10 (state-bearing ESMDA).
   Add the lead-time sawtooth (§5.3).
4. **Then, chosen by the numbers.** Route 7 if `σ_θ/σ_floor` says the periodic
   control is dead as posed; routes 8–9 if turbulent inflow is at its floor;
   routes 11–12 only if the N-probe (15) says sampling is binding.
5. **Cleanup PR, any time.** §4.2–4.4: ~12 k lines removable or
   consolidatable with no effect on the campaign path; docs in §4.5.

---

## 8. Appendices

| File | Content |
|---|---|
| [A_da_library_audit.md](da_review_2026-09/A_da_library_audit.md) | Module-by-module inventory of `libs/data-assimilation`, dead code, 17 correctness/quality items, proof that state reduction cannot help, gaps. |
| [B_scripts_configs_tests_docs_audit.md](da_review_2026-09/B_scripts_configs_tests_docs_audit.md) | Every entry point, config group and option with campaign/test/doc usage; drift between the three run configs; stale tooling; test coverage; ranked deletion list. |
| [C_metrics_and_diagnostics_audit.md](da_review_2026-09/C_metrics_and_diagnostics_audit.md) | Exact definition of every stored metric, definitional problems, computed-and-dropped diagnostics, the three missing indicators with implementation sketches, ten secondary indicators. |
| [D_campaign_results_synthesis.md](da_review_2026-09/D_campaign_results_synthesis.md) | Setup facts, master result table with climatology baselines and skill, the periodic identifiability and turbulent floor arguments, confounds, suspicious numbers, five discriminating experiments. |
| [E_improvement_routes.md](da_review_2026-09/E_improvement_routes.md) | Twelve routes with hypothesis, method, code touch-points, effort, expected effect per case, and success criteria; the periodic and turbulent-inflow questions worked out against the uDALES source. |

Where the appendices disagree, this document gives the reconciled reading:
the ESMDA raw-frame/aggregated noise mismatch is a defect (A) *and* the
accidental reason ESMDA generalises better than the filter (E); the fix is an
explicit representativeness term, not a smaller `C_D`. `final_time_smoothing`
is recommended in E's route B2 and condemned in A §3.7; this document sides
with A and recommends the state-bearing smoother without it.
