# Audit C — evaluation metrics and diagnostics for the ESMDA / EnKF / hybrid campaign

Read-only audit of `pyurbanair` @ `isda_experiments` (2026-09-04). Paths are repo-relative.
Opinions are labelled **[opinion]**; everything else is code or stored data.

Headline: the *library* (`libs/evaluation`) is far richer than what the deck reports. The
gap is almost entirely **surfacing and referencing**, not implementation — with three
genuine holes (time-resolved forecast/analysis decomposition, spread–skill, and any
baseline to normalise against).

---

## 0. Where the numbers come from

| Layer | File | Role |
|---|---|---|
| Scores | `libs/evaluation/src/evaluation/scores.py` (1601 L) | every estimator |
| Sensor reductions | `libs/evaluation/src/evaluation/sensors.py` | window mean/variance, bootstrap floor |
| Turbulence | `libs/evaluation/src/evaluation/turbulence.py` | state RMSE, moments, Welch/LSD |
| Figures | `libs/evaluation/src/evaluation/figures.py` (2763 L) | P1/S1/S4/S5/F1/D1/D3 |
| ESMDA stage | `scripts/esmda/compute_esmda_metrics.py` → `run_summary.yaml` |
| Filter stage | `scripts/filtering/compute_filtering_metrics.py` → `run_summary.yaml` |
| Hybrid | `scripts/run_filter_smoothing_pipeline.sh` — runs the *filtering* stages at the run root **and** the ESMDA stages in `<run>/esmda_view/` |
| Campaign table | `experiments_report/scripts/extract_comparison.py` |
| Deck | `presentations/isda_crps/latex/scripts/{make_figures.py,numbers.json}`, `main.tex` |

Run dirs: `presentations/isda_new/experiments/{esmda,filtering,filter_smoothing}/<run>/`.
**55 `run_summary.yaml` at the run root + 37 in `esmda_view/`.** The `*.nc` artifacts have
been pruned from all of them (`find … -name '*.nc'` → 6 files, 5 of them truths). Live
artifact sets survive under `.temp/` (e.g. `.temp/filtering_pyudales_to_pyudales/` with
`windows/window_*_{obs,pred_obs,forecast_state,posterior_state,prior_params,posterior_params}.nc`,
`eval_fields.nc`, `params_history.nc`, `state_history.nc`, `cycle_diagnostics.yaml`).
**Any new metric that needs member states requires a rerun; anything derived from
`run_summary.yaml`, `cycle_diagnostics.yaml` or the `windows/*obs*.nc` files is free.**

Structural fact that colours everything below: the ESMDA campaign runs are
**parameter-only** (`configuration.joint_state_and_parameter: false`,
`smoother: TimeVaryingParameterESMDA`). ESMDA never touches the state; its window
rollouts are free runs under estimated parameters. The EnKF/hybrid runs *do* update the
state (`mode: joint` / `state`). So "assimilated-sensor RMSE" is not the same kind of
quantity for the two families.

---

## 1. Inventory: what is computed now

### 1.1 Parameter metrics — `run_summary.yaml: parameter_metrics`

`parameter_metric_summary` (scores.py:1216) → per parameter (`inflow_angle`,
`velocity_magnitude`, …) and a `pooled` entry.

| key | definition as coded | notes |
|---|---|---|
| `rmse.{mean,final,max,min}` | scores.py:460 — **per-knot RMSE over MEMBERS**: `sqrt(mean_i (θ_i − θ*)²)`, then `series_stats` over knots | **not** the RMSE of the ensemble mean |
| `crps.*` | scores.py:462, fair `M(M−1)` CRPS (scores.py:73), per knot, reduced over knots | correct Ferro-2014 estimator, median-centred for float safety |
| `prior_rmse_mean`, `rmse_reduction_vs_prior` | `_skill_score` (scores.py:867), means intersected on knots finite in both | |
| `prior_crps_mean`, `crps_reduction_vs_prior` | as above (this is the CRPSS) | |
| `z_score.{n,mean,std,expected_std,max_abs}` | `(θ* − θ̄ᵃ)/σᵃ` per knot (scores.py:530), pooled; `expected_std = sqrt((1+1/M)(M−1)/(M−3))` = **1.031 at M=50** (scores.py:769) | correct finite-M reference |
| `normalized_error`, `contraction_ratio` | `(θ̄ᵃ−θ*)/σᵇ`, `σᵃ/σᵇ` per knot | needs a prior |
| `pooled.z_score.std` | **this is the deck's "zpool"** (`extract_comparison.py:147`) | see 1.7(b) |

Truth alignment: `_aligned_parameter_members` (scores.py:393) interpolates the truth onto
the estimate's knot axis — a time-varying truth is compared over the **whole trajectory**,
not at final time. Good.

### 1.2 State metric — `state_metrics.vel_magnitude_rmse`

`streaming_state_rmse` (turbulence.py:137): RMSE of |U| between the **ensemble-mean**
posterior state and the truth, on **4 evenly spaced z-levels only** (turbulence.py:103;
campaign `z_levels: [1, 11, 21, 31]` m), all frames, **no solid-cell mask**, then
`series_stats` over frames. ESMDA reads `posterior_state_mean.nc`; the filter reads the
analysed end-of-cycle states and compares against the truth's end-of-cycle frames.

### 1.3 Sensor metrics — `sensor_metrics[assimilation|validation]`

`vector_sensor_metrics` (scores.py:708):
* `velocity_vector_rmse(t) = sqrt(Σ_c mean_s (⟨v_c⟩_ens − v_c*)²)` — **ensemble-mean**
  error of the full (u,v,w) vector, averaged over sensors, one value per frame/cycle
  (scores.py:637–740). Truth is the **clean** truth (no observation noise added;
  perturbation lives only in `perturb_observations`, `_esmda_common.py:182`).
* `velocity_vector_energy_score(t)` — fair multivariate energy score (scores.py:664),
  `M(M−1)` pairwise term. **This is the number the CRPS deck prints as "CRPS".**
Both reduced by `series_stats` → `{mean,final,max,min}`. Held-out set is
`create_validation_points(cfg.obs)` (`_esmda_common.py:61`), never assimilated.

### 1.4 Sensor statistics — `sensor_statistics[set][posterior|prior][stat_quantity]`

The statistics-space view (`window_statistics_summary`, scores.py:1091). Per window/cycle
bin, per sensor: mean and variance (ddof=1) of `u,v,w,|U|` → 8 keys. Each scored with
fair CRPS (series over bins), pooled `z_score`, `rank_counts` (M+1 bins,
`ensemble_rank` scores.py:944, seeded tie-break), and `identifiability` = spread ÷ own
block-bootstrap sampling std (scores.py:1072).
Figure D1 (`rank_histogram.png`) pools **over windows, sensors, statistics and quantities**
(docs/temp/rank_histogram_math.md §7).
`extract_comparison.py:148–155` reads only `mean_magnitude` → `zval` and `edge`.

### 1.5 Field metrics — `field_metrics`

VDI 3783/9 hit rate `q` (scores.py:1294): `|p−o|/|o| ≤ 0.25 OR |p−o| ≤ W`, on the
time-mean velocity over the z-slabs, per component and pooled; `W` block-bootstrapped
from the truth's own sampling error. Present in 35/55 summaries. Reduced fields
(mean, **TKE**, **⟨u'w'⟩**, slabs + station columns, truth/prior/posterior) are written to
`eval_fields.nc` (`compute_esmda_metrics.py:1076`) and drawn by S1/F1.

### 1.6 Health / consistency diagnostics

* **Filter** — `filter_diagnostics` from `cycle_diagnostics.yaml` (180 cycles):
  `innovation_chi2 = dᵀ(C_DD + C_D)⁻¹d / N_d` with `d = y − mean H(x_f)` and **raw
  pre-inflation** anomalies (`filtering/base.py:1494–1503`); `obs_prior_rmse`,
  `obs_posterior_rmse`, `state_spread_{prior,posterior}`, `param_spread_{prior,posterior}`
  — each reduced to `{mean,final,max,min}`. `chi2` in the deck = `innovation_chi2.mean`.
* **ESMDA** — `esmda_diagnostics.data_mismatch`: `O_N = (1/2N_d)(d−g)ᵀC_D⁻¹(d−g)` per
  member per iteration (scores.py:1394), target ½, band `3/√(2N_d)`; plus
  `underfit_final`/`overfit_final`/`collapsed` flags (scores.py:1517). Figure D3
  `data_mismatch_decay.png`.
* `ensemble_health`: bitwise `n_unique`, `n_unique_per_cycle` (180 values),
  `min_over_median_pairwise`.
* `spectral_metrics`: log-spectral distance of the posterior-median Welch spectrum vs the
  truth's, with the truth's self-distance floor. **Requires an explicit
  `run_probe_series.py` rerun — present in 0/55 campaign summaries.**

### 1.7 Definitional problems

**(a) Two different RMSE estimators are tabled side by side.** Parameter RMSE is
`sqrt(mean over MEMBERS)` (scores.py:460) ≈ `sqrt(bias² + σ²)`; sensor and state RMSE are
errors of the **ensemble mean** (scores.py:639, turbulence.py:137). The code says so
(scores.py:352–355 "a *different* estimator family"), the deck does not. Consequence: a
parameter RMSE can never fall below the posterior spread, so `angle 2.4°` at
`σᵃ ≈ 2°` is already at its floor, while `assim 0.17` has no such floor. **[opinion]**
This is the single most misleading pairing in the tables.

**(b) "zpool" is the sample std of 6–30 numbers.** `parameter_metrics.pooled.z_score`
pools knots × parameters. Measured: filtering runs have `n = 6` (2 params × 3 window
knots), ESMDA runs `n = 30` (2 × 15). At n = 6 the sampling spread of a sample std from a
calibrated t-distribution is roughly ×0.4–×1.9, so F-run `zpool = 1.88` vs
`expected_std = 1.031` is **not** distinguishable from calibrated, while E19's `7.81`
(n = 30) is decisive. The deck prints them in the same column. The code warns that the
knots are strongly autocorrelated so the effective n is smaller still (scores.py:1290).

**(c) Static-parameter runs report a carried prior as their parameter RMSE.** In
`numbers.json`, F14/F15/F12/F17/F18/F10/F21 all carry byte-identical
`angle 19.29297022273086`, `vmag 0.9188727357384049`, `zpool 1.8775207589725202` — these
are state-only EnKF runs whose parameter ensemble is never updated.
`extract_comparison.py:78–82` guards this (`STATIC_PARAM_COLS`, `static_params`);
`numbers.json` does not, so those cells are prior draws, not estimates.

**(d) "full-state RMSE" is a 4-slice, unmasked, ensemble-mean |U| RMSE.**
`streaming_state_rmse` takes no solid mask, unlike `field_rmse` (scores.py:249). Harmless
in the current campaign (`solid_fraction: 0.0`, `solid_cell_source: none`) but it will
silently flatter Barcelona runs, where solid cells are zero in both fields.

**(e) χ² and O_N see assimilated observations only, and only the forecast side.**
`innovation_chi2` uses `C_DD + C_D` at forecast time; there is no analysis-residual
statistic, no per-sensor breakdown, and `C_D` is one scalar (`observation_error_std: 0.1`)
with no representativeness term — the code flags this itself
(`caveat: no_representativeness_error`, scores.py:1540). A misspecified `R` and an
under-dispersed ensemble are therefore not separable with what is stored.

**(f) The ESMDA O_N says every run grossly under-fits, and it is nowhere in the deck.**
`per_step_median[-1]` (target 0.5, band 0.217), measured across the 18 ESMDA summaries:

| case | pyudales→pyudales | pypalm→pyudales |
|---|---|---|
| inflow | 17.5 | — |
| inflow_turb | 2.37 | 30.1 |
| periodic | 37.2 | 57.4 |

Residual RMS = `σ_o·sqrt(2·O_N)`: 0.59 / 0.22 / 0.86 m/s (twin) and 0.78 / 1.07 m/s
(model error), against `σ_o = 0.1`. **R is misspecified by 2–11× in std.** Also monotone
in observation density (obs7p5 36.3 > obs15 30.1 > obs30 25.7 for pypalm/turb).

**(g) `obs_posterior_rmse` is a "ride-along" approximation under localization.**
`obs_posterior_rmse_kind: unlocalized_ride_along` in every campaign run
(`filtering/base.py:1509–1514`): the appended observation rows got a *global* update while
the state rows were localized. The analysis half of any sawtooth built from it is
optimistic and must be labelled.

**(h) The deck calls the energy score "CRPS", and `numbers.json` holds RMSE under the
same keys.** `main.tex:356,419,666,732` label tables "CRPS"; the values (0.17/0.37 for
F14) are `velocity_vector_energy_score.mean`, whereas `numbers.json` `assim/valid` are
`velocity_vector_rmse.mean` (F14 0.236/0.570). Two decks, same key names, different
quantities. **[opinion]** rename to `energy_score` in `numbers.json` before anything gets
mixed.

**(i) No spin-up or burn-in exclusion.** Every `series_stats` mean includes window/cycle 0,
where the ensemble is still essentially prior. For ESMDA that is 1/3 of the sample.
(The truth *is* correctly sliced past its own spin-up via `start_idx`/`t_offset`,
`_esmda_common.py:94`.)

**(j) Assimilated-sensor "success" is partly noise-fitting.** With `σ_o = 0.1`,
measured `obs_posterior_rmse.mean` is 0.050–0.061 in the twin inflow/periodic filtering
runs — *below* the observation-noise level. Nothing in the reporting flags fitting below R.

**(k) Rank histograms pool statistics and quantities**, so an over-dispersed mean and an
under-dispersed variance cancel (docs/temp/rank_histogram_math.md caveat 1). The deck's
`rank_valid_F14/F15` inherit this without saying so.

**(l) `identifiability` is absent on production-shaped runs** (3/55 summaries) — the
block-bootstrap floor needs ≳15 integral time scales and a 300 s window holds ~2
(master_plan.md, "Cross-cutting cautions"). So for 52/55 runs there is no evidence the
sensor statistics being ranked are identifiable at all.

**CRPS implementation itself is correct.** Fair `M(M−1)` estimator, O(n log n) sorted
form, median-centred against catastrophic cancellation (scores.py:112–128); energy score
likewise (scores.py:700–705). `metrics_version: 2` marks the boundary against the old
biased numbers. No issues found.

---

## 2. Computed and dropped

1. **The per-cycle obs-space sawtooth.** `cycle_diagnostics.yaml` holds 180 entries ×
   {`obs_prior_rmse`, `obs_posterior_rmse`, `innovation_chi2`, `state_spread_prior/posterior`,
   `param_spread_prior/posterior`, `analysis_time`}. All are collapsed to 4 scalars
   (`compute_filtering_metrics.py:414`) and **never plotted**. The ratio
   `obs_posterior/obs_prior` (both already in `run_summary.yaml`) is a free over-fitting
   indicator; measured across the campaign:

   | run | obs_pri | obs_post | ratio | χ² | assim | held-out | state |
   |---|---|---|---|---|---|---|---|
   | F, twin, loccorr, inflow | 0.185 | 0.061 | 0.33 | 0.90 | 0.155 | 0.394 | 0.247 |
   | F, twin, loccorr, inflow_turb | 0.250 | 0.049 | 0.20 | 1.04 | 0.206 | 0.559 | 0.422 |
   | F, twin, loccorr, periodic | 0.433 | 0.050 | **0.12** | 1.85 | 0.317 | **1.551** | **1.733** |
   | F, pypalm→ud, loccorr, periodic | 0.833 | 0.129 | 0.15 | **15.9** | 0.704 | **4.072** | 5.905 |
   | H, twin, loccorr, periodic | 0.412 | 0.053 | 0.13 | 1.81 | 0.314 | 1.246 | 1.019 |

   The periodic rows are the whole story in one line: the analysis drives the obs residual
   *below* `σ_o` while held-out error is 1.2–4.1 m/s. Nobody has drawn it.
2. **`esmda_diagnostics.data_mismatch` in `esmda_view/`.** 37 filtering/hybrid runs have it;
   `extract_comparison.py` reads `run_summary.yaml` at the **run root only** (line 116), so
   its `onp` column is `None` for every non-ESMDA run. The dropped numbers are the sharpest
   in the campaign — `per_step_median` = `[prior, posterior]`:
   `[35.9 → 0.75]` (pypalm periodic) vs `[34.1 → 9.46]` (pypalm inflow_turb): the filter
   fits the periodic observations to within their noise and still has held-out RMSE 4.07.
3. **Figure D3** (`data_mismatch_decay.png`, `make_esmda_figures.py:517`) is drawn in every
   ESMDA and every `esmda_view/` run dir and appears in no deck figure.
4. **`velocity_vector_energy_score` for the held-out set** — computed everywhere;
   `extract_comparison.py:135` keeps `valid_es`, `numbers.json` drops it.
5. **All eight `sensor_statistics` keys except `mean_magnitude`.** The four
   `variance_{u,v,w,magnitude}` entries measure directly whether the filter is destroying
   resolved velocity variance at the sensors — computed for both sets, both halves, never read
   by `extract_comparison.py` or the deck.
6. **Per-component hit rates** `field_metrics.hit_rate_posterior.{u,v,w}` (measured
   0.934/0.524/0.359 in the F7 run — the pooled `q = 0.606` hides that `w` fails badly)
   and `hit_rate_tolerance_w`.
7. **TKE and ⟨u'w'⟩.** Accumulated for truth/prior/posterior on slabs and station columns
   (`MomentAccumulator`, turbulence.py:781), written to `eval_fields.nc`, plotted in
   `station_profiles.png` / `mean_slices.png` — and **never scored**. There is no TKE number
   anywhere in `run_summary.yaml`.
8. **`spread_skill()`** (scores.py:322) — the Fortin-corrected SSR, delivered in WP1.1,
   unit-tested (`tests/test_evaluation_fair_scores.py:229`), **called by no production script**.
9. **`scripts/_common.py:250 compute_time_varying_metrics` / `:322 plot_time_varying_metrics`**
   (per-knot error / spread / CRPS / 90 % coverage per ESMDA iteration) — no caller anywhere
   in the repo. Dead code.
10. **`ensemble_health.n_unique_per_cycle`** (180 values) and `min_over_median_pairwise`
    (0.235 in F7 — the closest pair is 4× closer than the median) — never plotted.
11. **Spectra.** `welch_spectrum`, `log_spectral_distance`, `probe_spectra`,
    `spectral_metric_summary`, figure S4 (`plot_spectra`, figures.py:2164) — a complete,
    tested subsystem gated behind a `run_probe_series.py` rerun that was never done:
    0/55 summaries carry `spectral_metrics`.
12. **`window_{w}_obs.nc` carries `obs_clean`** (noise-free truth projection) beside `obs`
    and `obs_error_std` (`run_filtering.py:397–411`). Nothing computes the representativeness
    residual `H(x) − obs_clean` against `σ_o`.
13. **An entire second analysis pipeline** in `scripts/figure_creation/` is unused by the
    campaign: `compute_sweep_metrics.py` (per-component u/v/w/|U| sensor metrics for both
    sets + `sensor_timeseries_<set>.nc` holding truth, **prior** and posterior ensembles),
    `compare_sweep_results.py`, `make_figures_block_{a,b,c}.py`,
    `visualize_state_run.py:181 plot_state_spread_reduction` and `:228 plot_window_increment`
    (prior/posterior spread and increment maps — i.e. increment-vs-innovation diagnostics
    already written).
14. **`data_mismatch_summary`'s `underfit_final` / `overfit_final` / `collapsed` flags** —
    computed, stored, never read by any consumer.

---

## 3. The three most crucial missing indicators

### #1 — Forecast/analysis decomposition and error-regrowth rate at held-out sensors and in state space (the "sawtooth" + predictability horizon)

**Definition.** Per assimilation cycle *k* and lead time τ within the cycle
(τ = 0 at the analysis, τ = T_cycle at the next forecast):

```
E_set(k, τ)      = sqrt( mean_s || ⟨v⟩_ens(k,τ,s) − v*(k,τ,s) ||² )      set ∈ {assim, held-out}
E_state(k, τ)    = sqrt( mean_cells ( |U|_ens-mean − |U|* )² )
r(k)             = E(k, T_cycle) / E(k, 0)                                regrowth ratio
τ_e              = T_cycle / ln r                                        e-folding time
skill_retained   = 1 − E(k,T)/E_free(k,T)                                vs the free run (#3)
```
Report `E(τ)` averaged over cycles (one curve per case per method), plus scalars
`E_analysis = ⟨E(k,0)⟩`, `E_forecast = ⟨E(k,T)⟩`, `r`, `τ_e`.

**Question it answers.** The campaign's central puzzle is that the analysis fits the
assimilated sensors superbly (obs residual 0.05 m/s < σ_o) while held-out error stays at
0.32–1.55 m/s. There are exactly two explanations and the current metrics cannot
distinguish them: (i) the increment is *local* — it never reaches the held-out sensors
(localization too tight, or B has no correlation there); (ii) the increment is *global but
transient* — the error regrows within one 2 s cycle because the assimilation interval
exceeds the predictability horizon of the resolved eddies. `E(τ)` separates them
immediately: (i) gives `E_heldout(0) ≈ E_heldout(T)` with `E_assim(0) ≪ E_assim(T)`;
(ii) gives both sets dropping at τ = 0 and both recovering by τ = T.

**Expected reading.**
* *Laminar inflow* — small `r` (≈1.2–1.5), long `τ_e`: the flow is predictable, the
  increment should persist. If it does not, that is a localization/`B` problem, not physics.
* *Turbulent synthetic-eddy inflow* — `r` close to 1 at held-out sensors even when the
  analysis lands: the inflow noise is a fresh stochastic forcing every cycle and the
  achievable held-out skill is bounded. **[opinion]** this is the most likely explanation
  for the modest held-out numbers, and it is *not* a method failure.
* *Periodic* — the sharpest case: obs ratio already measured at 0.12 with held-out RMSE
  1.55. Expect `E_heldout(0) ≈ E_heldout(T)`, i.e. the increment is entirely spurious and
  local. That would be the strongest single slide in the deck.

**Implementation.** Two tiers.
*Tier 0, zero cost, no rerun:* `cycle_diagnostics.yaml` already holds 180 ×
(`obs_prior_rmse`, `obs_posterior_rmse`) per run. Plot them as a 180-cycle sawtooth and
report `ratio = obs_post.mean / obs_pri.mean` (numbers in §2.1 above). Label it
"assimilated observations only, posterior rows are an unlocalized ride-along"
(`filtering/base.py:1509`). ~40 lines in `scripts/filtering/make_filtering_figures.py`
plus one key in `filter_diagnostics`.
*Tier 1, the real metric:* the data is already on disk for the 37 runs whose
`cycle_states.kind == "forecast"` — `windows/window_*_forecast_state.nc` holds *every
frame of every cycle's forecast segment for every member*
(`compute_filtering_metrics.py:_cycle_evaluation_blocks`, description string in
`run_summary.yaml`). `cycle_sensor_series(run_dir, ta, source, sensor_sets, on_member=…)`
already extracts `(component, ensemble, time, sensor)` over that whole axis; today it is
handed straight to `window_statistics` which **bins time away**. Add a sibling block that
instead reshapes `time → (cycle, lead)` with `cycle_seconds(ta)` and reduces over cycles →
`sawtooth[set] = {lead_seconds, rmse_mean, spread_mean, r, tau_e}`. For ESMDA the analogue
is lead time *within a window* off `windows/window_*_posterior_state.nc`, extracted by the
existing `ensemble_sensor_series` (`_esmda_common.py:311`). Truth at full cadence is
`truth_sensor_series` (`_esmda_common.py:389`), already used. Estimated ~120 lines across
`compute_filtering_metrics.py`, `compute_esmda_metrics.py` and one new figure function.

### #2 — Spread–skill ratio over time, per sensor set, forecast and analysis separately

**Definition** (Fortin et al. 2014; already coded as `spread_skill`, scores.py:322):

```
SSR = sqrt((M+1)/M) · RMS_t(spread(t)) / RMS_t(rmse(t))
spread(t) = sqrt( mean_s mean_c var_ens( v_c(t,s) ) )
rmse(t)   = sqrt( mean_s mean_c ( ⟨v_c⟩ − v_c* )² )
```
Report `SSR` per set (assim / held-out) per run, plus the two series so the *time*
behaviour is visible, and split at forecast vs analysis time once #1 exists.
SSR < 1 → over-confident; > 1 → over-dispersive; ≈ 1 → calibrated.

**Question it answers.** "Did the method fail because it is wrong, or because it is
over-confident?" Today the only calibration numbers are `zpool` (std of 6–30 parameter
z-scores — statistically empty at n = 6, §1.7b), `zval` (pooled over ~500–800 window-mean
knots, one number for the whole run, no time axis), and rank histograms (pooled over
statistics and quantities, so shapes cancel). None of them says *when* the ensemble became
over-confident, and none is measured in the same units as the RMSE the deck reports. SSR
puts spread and error on one axis in m/s and is directly comparable across the three
methods and the three cases. It is also the metric that makes the RTPS story quantitative:
"inflation helps" is currently argued from held-out RMSE 0.37 → 0.30, which cannot
distinguish "better analysis" from "less collapsed".

**Expected reading.**
* *Laminar, no inflation* (F14) — SSR ≪ 1 at both sets and falling with cycle; χ² 3.98
  already hints at it. With RTPS (F15) SSR should recover toward ~0.7–1.
* *Turbulent* — SSR near 1 at assimilated sensors but ≪ 1 at held-out: the ensemble knows
  its error where it is constrained and not where it is not. That is the calibration
  signature of missing model error.
* *Periodic* — expect SSR ≪ 1 everywhere and a strong contrast with the hybrid, whose
  ESMDA-side parameter spread is not collapsed by 180 sequential updates.

**Implementation.** Cheapest of the three. `compute_sensor_metrics` (scores.py:601) already
returns `members (E,T,S)`, `ens_mean`, `truth`, `rmse (T,)`. Add
`spread = sqrt(mean_s var(members, axis=0, ddof=1))` in `vector_sensor_metrics`
(scores.py:708) — combine components as `sqrt(Σ_c spread_c²)`, exactly as the RMSE is
combined at scores.py:740 — then `sensor_metrics[name]["spread_skill"] =
spread_skill(spread_ts, rmse_ts, n_members)` and `["velocity_vector_spread"] =
series_stats(spread_ts)`. Both call sites (`compute_esmda_metrics.py:1490`,
`compute_filtering_metrics.py:~490`) are one-liners. ~25 lines total, uses only arrays
already in memory, needs **no rerun** for any run whose `windows/` still exists — and the
`.temp/` runs are enough to validate it. Add the two series to `sensor_fans.png` as a
second panel.

### #3 — Reference baselines and skill scores: free run with *true* parameters, free run with the *prior* ensemble, climatology, persistence

**Definition.** Four references per case, each evaluated with the identical
`vector_sensor_metrics` / `streaming_state_rmse` machinery on the same held-out sensors
and horizon:

```
R_perfect  : M-member free run with the TRUE parameter trajectory      → irreducible floor
R_prior    : M-member free run with the PRIOR parameter ensemble       → no-assimilation reference
R_clim     : RMSE of the truth's own time-mean               = std_t(v*)  → "know nothing"
R_persist  : RMSE of the truth held at each window start                 → "know the last frame"

SS  = 1 − RMSE_method / RMSE_ref            (per set, per case)
CRPSS = 1 − CRPS_method / CRPS_R_prior
```

**Question it answers.** The one the owner actually asked: *is a held-out RMSE of 0.32 m/s
good?* Nothing in the campaign can answer it. `R_clim` sets the ceiling; `R_perfect` sets
the floor — and for a **parameter-only** ESMDA (which is what the E-runs are,
`joint_state_and_parameter: false`) `R_perfect` is *exactly* the best that route can ever
achieve, because a perfect parameter estimate gives you a free run and nothing more. If
`R_perfect ≈ 0.3` in the turbulent case, then ESMDA's 0.27 held-out is at the ceiling of
its own method class and no amount of localization tuning will move it; if
`R_perfect ≈ 0.05`, the parameter estimate is still far off and there is real headroom.
**[opinion]** This is the highest-value missing number in the entire campaign, and it is
one forward run per case.

Related structural gap: **0 of 55 runs have any prior reference for the state or sensor
numbers.** `conf/run_esmda.yaml:160` sets `save_prior_state: false`, and the filter and
hybrid hard-code it (`run_filtering.py:972`, `run_filter_smoothing.py:1011`), so the
`prior` half of `sensor_statistics` and `field_metrics`, the whole
`crps_reduction_vs_prior` skill machinery (already implemented, `_skill_score`
scores.py:867), and `plot_state_spread_reduction` / `plot_window_increment`
(`visualize_state_run.py:181,228`) all have nothing to consume. Only the *parameters* have
a prior reference — which is why `angle 19.29 → 3.90` reads as a triumph while
`held-out 0.57 → 0.32` reads as ambiguous.

**Expected reading.**
* *Laminar* — `R_clim` large, `R_perfect` small: large positive skill scores for every
  method; the case is genuinely solved.
* *Turbulent synthetic-eddy* — `R_perfect` close to the reported held-out numbers
  (turbulent phase error dominates). Expect `SS vs R_perfect ≈ 0` and
  `SS vs R_prior ≫ 0`: the methods extract everything the parameters carry, and the
  residual is irreducible. This turns "held-out accuracy is only modest" from a weakness
  into a result.
* *Periodic* — `R_prior ≈ R_perfect` (no inflow signal to estimate, as the deck already
  says ESMDA "cannot move"), so the *only* skill available is from the state update; the
  hybrid's advantage should show up as skill against `R_perfect`, which no current number
  can express.

**Implementation.** Two halves.
*Half A (config only, retro-fittable to future runs):* flip `run.save_prior_state=true` in
`conf/run_esmda.yaml:160` and expose it for the filter/hybrid instead of hard-coding
`False`. Everything downstream already exists: `_prior_sensor_series`
(`compute_esmda_metrics.py:228`), the prior `MeanFieldCollector`, `prior_crps_mean` /
`crps_reduction_vs_prior` in `window_statistics_summary` (scores.py:1195–1210), and
`hit_rate_prior` in `_mean_field_block`. Cost: roughly doubles the on-disk state footprint
(the reason it was turned off).
*Half B (the baselines):* three `scripts/run_forward_model.py` runs per case —
`params@truth_params=…` (perfect), the prior parameter ensemble (no-assimilation), and a
single-member run for a deterministic reference — writing `state.nc` in the same schema
the truth uses, then a small `scripts/esmda/compute_baseline_metrics.py` that reuses
`build_sensor_sets` + `truth_sensor_series` + `vector_sensor_metrics` + `streaming_state_rmse`
verbatim and writes `baseline_summary.yaml` per case. `R_clim` and `R_persist` need no run
at all — they come from the truth file alone (`std_t` over the horizon at the sensor
points, and the window-start value held constant). Then add
`skill_vs_{perfect,prior,clim}` columns in `extract_comparison.py`.
Estimated: 3 forward runs × 3 cases (the expensive part), ~150 lines of new script,
zero new estimator code.

---

## 4. Secondary indicators (in rough priority order)

1. **Desroziers consistency, per sensor and per component.** With
   `d_ob = y − H(x^b)`, `d_oa = y − H(x^a)`, `d_ab = H(x^a) − H(x^b)`:
   `⟨d_oa d_obᵀ⟩ ≈ R`, `⟨d_ab d_obᵀ⟩ ≈ HBHᵀ`, `⟨d_ob d_obᵀ⟩ ≈ HBHᵀ + R`. **Everything
   needed is already on disk**: `windows/window_{w}_obs.nc` (`obs`, `obs_clean`,
   `obs_error_std`) and `windows/window_{w}_pred_obs.nc` (`pred_obs`, dims
   `(esmda_step, obs_index, ensemble)`, step 0 = forecast, step −1 = analysis) with
   `obs_sensor` / `obs_state` / `obs_interval` coordinates (`run_esmda.py:396–430`). Gives
   the *effective* `R` per sensor, which O_N only gives in aggregate. ~80 lines, no rerun.
   Ranked below the top three only because O_N already delivers the headline (R is 2–11×
   too small) — Desroziers refines rather than reveals.
2. **Representativeness residual.** `RMS(H(x^b) − obs_clean)` vs `σ_o`, from the same two
   files. Directly quantifies the sub-grid/phase error that `C_D` omits, and is the number
   to set a defensible `esmda.obs_error_std` from.
3. **Held-out identifiability map.** Sample correlation across members between each
   held-out sensor's predicted value and the assimilated observation vector,
   `max_j |ρ(v_heldout, H_j(x))|` per held-out sensor. Separates "this sensor was never
   observable from the assimilated set" from "the update was wrong". `CorrelationLocalization`
   (`localization/correlation.py:96–109`) already computes exactly this correlation matrix
   internally and discards it — record `rho_max` per row and the fraction of pairs
   truncated below `rho_t = 0.35`.
4. **Localization severity.** Fraction of (state row, observation) pairs excluded and mean
   tapering inflation, per cycle. Currently `cycle_diagnostics`' `local_*` group is `null`
   for `CorrelationLocalization` (only LETKF fills it). Answers "is loc. off vs loc. on"
   quantitatively instead of by run name.
5. **Parameter sensitivity ∂H/∂θ.** Ensemble regression of `pred_obs` on the prior
   parameters (`windows/window_{w}_prior_params.nc` + `pred_obs` step 0). Together with
   `contraction_ratio` (already computed) this distinguishes *unidentifiable*
   (∂H/∂θ ≈ 0, ratio ≈ 1) from *collapsed* (∂H/∂θ large, ratio ≪ 1, |z| ≫ 1) — precisely
   the periodic-vs-turbulent distinction the deck asserts qualitatively.
6. **Resolved-TKE ratio.** `TKE_posterior / TKE_truth` at the station columns and per
   z-slab, plus a hit rate on TKE with its own bootstrapped `W`. One scalar read off
   `eval_fields.nc`, which already carries `{truth,prior,posterior}_{slab,station}_tke`
   and `_uw`. Answers "is the filter destroying turbulence?" without the probe rerun the
   spectral metric needs. **[opinion]** cheapest high-value addition after SSR.
7. **Surface the stored variance statistics.** `sensor_statistics[set].posterior.variance_*`
   (CRPS, z-score, rank counts) already exist for both sets in all 55 runs — add
   `variance_magnitude` columns to `extract_comparison.py` beside `mean_magnitude`. Zero
   compute.
8. **Analysis-increment statistics.** `‖x^a − x^f‖` per cycle vs the innovation norm, and
   the increment's horizontal spectrum — the direct test for spurious long-range increments
   under weak localization. `plot_window_increment` (`visualize_state_run.py:228`) already
   draws the map; it needs `save_prior_state` (see #3, Half A).
9. **CRPS decomposition** (Hersbach 2000 reliability + potential CRPS) at held-out sensors,
   or simply `CRPSS` against `R_prior` once #3 exists — separates "sharp but biased" from
   "reliable but unskilful" in the one number the deck already prints.
10. **Time-to-recover after the parameter jump.** For the time-varying truth: cycles until
    `|θ̄ − θ*| < σᵇ/2` and until held-out RMSE returns to its pre-jump level. Directly
    comparable across ESMDA (3 window knots) / EnKF (180 cycles) / hybrid, and currently
    invisible because everything is reduced to `{mean, final, max, min}`.

---

## 5. Quick wins, ordered by cost

| # | Action | Cost | Needs rerun? |
|---|---|---|---|
| 1 | Plot the 180-cycle `obs_prior/obs_posterior` sawtooth; add `ratio` to `filter_diagnostics` | ~40 L | no |
| 2 | Read `esmda_view/run_summary.yaml` in `extract_comparison.py` so `onp` is populated for F/H runs | ~10 L | no |
| 3 | Surface `O_N`, `valid_es`, `variance_magnitude`, per-component `q` in `numbers.json` and the deck | ~20 L | no |
| 4 | Wire `spread_skill()` into `vector_sensor_metrics` (§3 #2) | ~25 L | no (uses `.temp/` runs to validate) |
| 5 | TKE ratio scalar off `eval_fields.nc` (§4 #6) | ~30 L | no |
| 6 | Desroziers + representativeness residual off `windows/*obs*.nc` (§4 #1–2) | ~80 L | no |
| 7 | Lead-time sawtooth off `window_*_forecast_state.nc` (§3 #1, tier 1) | ~120 L | no, for the 37 forecast-source runs |
| 8 | Baselines `R_perfect` / `R_prior` / `R_clim` (§3 #3) | ~150 L + 3 runs/case | yes |
| 9 | `save_prior_state=true` everywhere | config + un-hard-code 2 lines | yes |
| 10 | `run_probe_series.py` rerun to finally get `spectral_metrics` / S4 | config | yes |

Note for 4/6/7: the campaign run dirs under `presentations/isda_new/experiments/` have had
their `*.nc` pruned. Develop and validate against `.temp/filtering_pyudales_to_pyudales/`
and `.temp/pyudales_to_pyudales_correlation_inflow_turbulence/`, which still hold the full
artifact set.
