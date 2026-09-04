# Plan: experiment campaign for the ISDA 2026 talk

> **Status: working plan.** Written 2026-08-10. Companion to the deck in
> `presentations/isda/` (which references the experiment IDs below) and to
> [filtering_state_reduction_and_transforms.md](filtering_state_reduction_and_transforms.md),
> assumed implemented **with its review amendments** (§2). Working notes, not a
> maintained reference — verify config/flag names against the tree when running.

## 1. Purpose and constraints

The deck currently has nine `\uapending` placeholders. The talk is 15–20
minutes, so at most **six result slides** fit; everything else is backup. This
plan defines the smallest run set that fills those slides, plus the
new-machinery results (ETKF / LETKF / filtering state reduction) that are the
talk's novelty for a data-assimilation audience.

The narrative the runs must support: *the same observations, two update
strategies (smoother vs. filter), on a ladder of three cases; the plain setup
carries case 1, strains in case 2, and needs state estimation plus the new
update machinery in case 3.* Every experiment exists to either confirm or
honestly refute a rung of that ladder — a clean negative result (e.g. reduction
does not help at this ensemble size) is presentable and must not be massaged.

## 2. Prerequisites (blocking)

1. **PR 1 and PR 2 of the filtering plan merged, with the review amendments:**
   unnormalized forgetting recursion `C_k = λ C_{k-1} + B_k B_k^T` (λ = 1 is
   exact equal-weight accumulation), posterior inflation applied in **physical
   space** after decoding (preserves RTPS semantics and the full-rank
   equivalence invariant), finite-column guard before streaming basis updates,
   and construction-time analysis/localization compatibility checks.
2. **No new metrics or post-processing.** The campaign consumes exactly what
   the existing pipelines emit (§6.1) plus the per-cycle reduction/transform
   diagnostics that PR 1/PR 2 of the filtering plan persist in
   `cycle_diagnostics.yaml` (retained rank, retained energy, projection
   residual, basis/analysis wall time, subspace drift). Head-to-head slide
   tables are **hand-assembled from `run_summary.yaml` values** — the deck
   already hand-types its booktabs tables — with the source run directories
   recorded in a TeX comment next to each table. Where a wanted view has no
   existing figure (the per-cycle χ² trace), the slide quotes the persisted
   numbers instead of getting a new plot.
3. **Three frozen case configs**, committed as config variants with fixed truth
   seeds (proposed names; adjust to what the config layout allows):
   * `case=xie_and_castro` — **laminar**: nudged power-law inlet (current
     default).
   * `case=xie_and_castro_turbulent` — **turbulent inflow**: synthetic driver
     planes (`BCxm=3`, digital-filter fluctuations; already implemented on the
     uDALES route), interior nudging off.
   * `case=xie_and_castro_periodic` — **periodic**: cyclic in x, interior
     nudging as the only momentum source.
4. **Truth reuse.** Generate each case's truth once with
   `run_forward_model.py`, store it, and point every assimilation run at it via
   `run.truth_dir` with pinned observation-noise seeds — all methods must see
   byte-identical truth and noise.
5. **Where runs execute:** DelftBlue/Snellius via `docs/job_scripts.md`. Do
   not run the campaign on the local Mac (known pylbm/torch aborts; uDALES
   ensembles are too slow there anyway). Production shapes only — smoke shapes
   place validation sensors outside the domain.

## 3. Common protocol

Comparability rules are binding; exact time constants may be tuned once per
case when the configs are frozen, then never varied between methods.

* **Ensemble size:** `N_e = 50` for *every* run of *both* methods. The old
  case-1 ESMDA slides used 64/128; the head-to-head rows must be re-run at 50
  (E2) so no table mixes ensemble sizes.
* **Horizon and cadence (per case, fixed across methods):** observation
  interval 10 s = one filter cycle; ESMDA window 30 s (3 observation
  intervals per window), `N_a = 4`; horizon 300 s (30 filter cycles, 10
  ESMDA windows). Truth inflow: `params@truth_params=dynamic_sine`
  (time-varying angle + speed); prior static for the filter
  (`params@prior_params=static`), AR(2) dynamic prior for the smoother.
* **Filter defaults unless the experiment says otherwise:**
  `filtering.mode=joint`, `filtering/analysis=stochastic`,
  `filtering/inflation=rtps`, `filtering/evolution=random_walk`,
  `filtering/localization=none`, `filtering/state_reduction=none`.
* **Metrics: the existing blocks, by their existing names.** From
  `run_summary.yaml`: `parameter_metrics` (per-parameter RMSE/CRPS +
  calibration), `state_metrics.vel_magnitude_rmse` (mean & final),
  `sensor_metrics` per sensor set — `assimilation` **and** `validation` —
  and `filter_diagnostics` (innovation-χ² / obs-RMSE summary stats). From
  `cycle_diagnostics.yaml`: the per-cycle χ², spreads, and (PR 1/PR 2) the
  reduction/transform diagnostics. Cost is reported from what exists:
  forward-run counts are arithmetic from the config (filter: `N_e` segment
  runs per cycle; ESMDA: `(N_a + 1) × N_e` per window) and wall time comes
  from `run_info.yaml` timing. No new metric is computed anywhere.
* **Seeds:** headline head-to-heads (E2, E5) run 3 assimilation seeds and
  report mean ± range; everything else runs 1 seed (cost control). Truth and
  noise seeds are never varied.

## 4. Experiment matrix

| ID | Case | What | Priority | Slide |
|---|---|---|---|---|
| E1 | laminar | EnKF joint baseline | **P0** | Case 1: EnKF results |
| E2 | laminar | head-to-head ESMDA / EnKF / ETKF | **P0** | Case 1: methods table |
| E3 | turbulent | ESMDA | **P0** | Case 2: ESMDA results |
| E4 | turbulent | EnKF + χ²/spread diagnostics | **P0** | Case 2: EnKF results |
| E5 | turbulent | head-to-head (analysis of E3+E4) | **P0** | Case 2: what strains |
| E6 | periodic | parameter-only vs. joint, both methods | **P0** | Case 3: results |
| E7 | laminar | ETKF vs. stochastic EnKF | **P1** | row in E2 table |
| E8 | periodic | filtering state-reduction ladder | **P1** | Case 3: reduction |
| E9 | periodic | LETKF vs. global ETKF | **P1** | Case 3: localization |
| E10 | turbulent | observation TSVD | P2 | backup only |
| E11 | laminar/turb | N_e and interval sweeps | P2 | backup only |

### E1 — case 1 EnKF baseline (P0)

The filter's debut on the friendly case; must exist before any comparison.

```
python scripts/filtering/run_filtering.py case=xie_and_castro \
  filtering.mode=joint filtering.num_cycles=30 ensemble.ensemble_size=50 \
  run.truth_dir=<case1_truth> run.save_history=true
```

Deliverables: the pipeline's `parameter_evolution.png` (analyzed angle/speed
per cycle with ensemble band + per-cycle |U| RMSE) as the slide figure; the
innovation-χ² behaviour quoted on the slide from `filter_diagnostics` /
`cycle_diagnostics.yaml` (no new plot). Expected: clean tracking of the sine
truth after a few-cycle spin-up. Risk: RTPS α and random-walk std may need
one tuning pass — tune on this case only, then freeze for every other filter
run.

### E2 — case 1 head-to-head (P0)

Rows: ESMDA (AR(2) dynamic prior), stochastic EnKF (E1), ETKF (E7). Columns
straight out of `run_summary.yaml`: angle RMSE, speed RMSE,
`state_metrics.vel_magnitude_rmse`, validation-sensor vector RMSE, plus
forward-run count (config arithmetic) and wall time (`run_info.yaml`). 3
seeds per row, hand-averaged into the table. The ESMDA row is a **re-run at
N_e = 50** on the shared truth — do not reuse the old N_e = 64 numbers.

Expected: comparable accuracy; the filter at ~`1/(N_a+1)` of the smoother's
cost but without within-window trajectory estimates. Present the trade, not a
winner.

### E3 / E4 / E5 — case 2, turbulent inflow (P0)

Same layouts as case 1 (E3 mirrors the ESMDA figures, E4 mirrors E1) on the
driver-plane case. E5 is analysis only — no new runs.

Hypothesis to test explicitly: the inlet now carries turbulence the
parameters do not determine, so (a) parameter RMSE degrades relative to case
1, (b) the filter's χ² reveals whether the irreducible spread is being
misread as parameter information (over-confidence), and (c) RTPS/random-walk
settings frozen in E1 either cope or visibly don't. The talk's "first place
the plain setup strains" slide is built from whichever of (a)–(c) actually
shows; if none shows, say so and move the strain narrative to case 3.

### E6 — case 3, periodic flow (P0)

Four runs on the periodic case: {ESMDA, EnKF} × {parameter-only,
joint/state-bearing}. The point of the case: with recirculating wakes the
state carries memory, so parameter-only updating should show persistent
innovation bias / drifting field RMSE that the state-updating variants
correct. This is the slide that motivates state estimation — it must isolate
*mode*, not method, so all four cells share truth, sensors, `N_e`, cadence.
Presented as a hand-typed 2×2 table (field |U| RMSE mean/final, validation
RMSE, χ² mean) from the four `run_summary.yaml` files, with one run's
existing `final_state_with_obs.png` as the visual if space allows.

### E7 — ETKF vs. stochastic (P1, rides on E1)

```
... filtering/analysis=etkf   # otherwise byte-identical to E1
```

Hypothesis: at `N_e = 50`, `N_d ≈ 12`, removing perturbation sampling noise
gives equal-or-better RMSE and a cleaner χ². Feeds a row of E2's table; only
gets its own backup slide if the difference is visible. Gate: the ETKF
linear-Gaussian unit tests (PR 2) pass — a talk figure is not evidence of
correctness.

### E8 — filtering state-reduction ladder (P1)

On the periodic case (where the state block matters), joint mode, stochastic
analysis, 6 runs:

```
filtering/state_reduction=none                                  # control (= E6 joint)
filtering/state_reduction=svd_current  +energy_fraction=1.0     # equivalence gate
filtering/state_reduction=svd_current  +energy_fraction={0.99,0.95,0.90}
filtering/state_reduction=svd_streaming +energy_fraction=0.99 \
  +forgetting_factor=<half-life ≈ 5 cycles>
```

The full-rank run is a **correctness gate, not a slide**: it must match the
control within float32 tolerance or E8 is void (bug, not science). Slide
content: a hand-typed ladder table — retained rank / retained energy /
projection residual / analysis wall time from the PR 1 fields in
`cycle_diagnostics.yaml` (cycle-averaged by hand), skill (field / validation
RMSE) and χ² from `run_summary.yaml` — one row per rung, streaming vs.
current-cycle. Expected honest
outcome per the amended plan: no speedup at this shape (SVD costs more than
the cross-covariance at `N_e > N_d`); the question the slide answers is
whether truncation-as-regularization or the streaming basis's cross-cycle
memory buys any skill. "No, and here is the cost" is a valid ISDA result.

### E9 — LETKF vs. global ETKF (P1)

On the periodic case, joint mode: `filtering/analysis=letkf` +
`filtering/localization=distance` (radius from the case's building spacing;
one run each at radius and 2× radius), vs. `filtering/analysis=etkf`
unlocalized, vs. stochastic + distance localization (the existing path).
Question: with ~12 observations and `N_e = 50`, does spatial localization of
the state update help or just decouple the far field? Same hand-typed table
form as E8. Watch the LETKF wall time (`run_info.yaml` timing) — the
per-block `N_e × N_e` transforms are the known cost risk; if a run exceeds
~3× the stochastic-localized wall time, record it and say so on the slide.

### E10 — observation TSVD (P2, backup)

Run only if E8/E9 diagnostics show local observation ill-conditioning
(retained-spectrum diagnostics from the TSVD helper). One `etkf_tsvd` run on
case 2. Backup slide at most — the amended plan already predicts little
benefit at ~12 observations.

### E11 — sweeps (P2, backup)

`N_e ∈ {25, 50, 100}` for the case-2 EnKF, and observation interval
`∈ {5, 10, 20}` s for the case-1 EnKF (cycle length = interval; this is the
filter-specific knob the deck's interval slide talks about). Backup slides
for questions; run only with leftover budget.

## 5. Run budget

Model-seconds per assimilation run at horizon 300 s, `N_e = 50`: filter
`1.5e4`; ESMDA `(4+1) × 1.5e4 = 7.5e4`. Totals: P0 ≈ 4 ESMDA-equivalents +
~10 filter runs (incl. 3-seed replicates), P1 ≈ 10 filter runs. Roughly
**the equivalent of ~13 single ESMDA windows-campaigns** — sized for a week
of HPC turnaround, not a weekend local run. Trim order if the budget or the
clock runs out: E11 → E10 → E9's second radius → E8's 0.90 rung → E2/E5
replicate seeds (report single-seed with a caveat). P0 is never trimmed —
without it the talk has empty slides.

## 6. Figure contract

### 6.1 What already exists (verified 2026-08-10)

The filtering entry point has the same three-stage pipeline as ESMDA
(`run_filtering.py` → `compute_filtering_metrics.py` →
`make_filtering_figures.py`, orchestrated by `run_filtering_pipeline.sh`):

* **Metrics** (`run_summary.yaml`): per-parameter RMSE/CRPS + calibration,
  |U| field RMSE per analyzed cycle, per-sensor-set vector RMSE / energy
  score / fair-CRPS statistics — **including the held-out validation set**
  (`build_sensor_sets` is shared with ESMDA, and
  `conf/case/xie_and_castro.yaml` already defines 4 validation sensors),
  innovation-χ²/obs-RMSE summary stats, VDI hit-rate field metrics.
* **Figures**: parameter trajectories per cycle, parameter error,
  final-state field with sensors, sensor time series / fans, rank
  histograms, mean slices, station profiles, marginals, rollout animation.
* **Cross-run comparison**: `scripts/figure_creation/compare_state_runs.py`
  (bar-chart panels + summary table/CSV over `run_summary.yaml` dirs) exists
  for ESMDA sweeps; per the no-new-post-processing rule it is *not* extended
  — the talk's cross-method tables are hand-typed instead.

So every experiment needs only runs: E1/E3/E4 use pipeline figures directly,
E2/E5/E6/E8/E9 use hand-typed tables of pipeline numbers, and E8/E9
additionally read the PR 1/PR 2 diagnostics fields.

### 6.2 Deck-tree contract

Nothing writes into `presentations/` and nothing new will: pipeline figures
are **hand-copied, unmodified and renamed only by directory**, from the run
dir into one directory per experiment (as was done for the existing deck
figures); tables are hand-typed booktabs in `main.tex` with the source run
directories in a TeX comment beside them.

```
figures/E1_case1_enkf/parameter_evolution.png      <- run dir, verbatim
figures/E2_case1_h2h/                              (table in main.tex; comment
                                                    lists source run dirs)
figures/E3_case2_esmda/param_traj_udales.png       <- ESMDA pipeline, verbatim
figures/E4_case2_enkf/parameter_evolution.png      <- run dir, verbatim
figures/E5_case2_h2h/                              (table in main.tex)
figures/E6_case3_modes/final_state_with_obs.png    <- one run's, verbatim
figures/E8_case3_reduction/                        (table in main.tex)
figures/E9_case3_letkf/                            (table in main.tex)
```

Every producing run records `run_info.yaml`; the slide caption cites `N_e`,
cadence, and seed count. Timings without hardware + shape fields are not
comparable (same rule as the filtering plan).

## 7. Slide mapping and decision gates

* Deck placeholders name these IDs (`E1`…`E9`) so slide ↔ run traceability
  survives the next deck reshuffle.
* Gates: E8's full-rank equivalence must pass before any truncated rung is
  interpreted; E7 needs the PR 2 unit tests green; E5/E6 conclusions must
  quote χ² alongside RMSE (a posterior that "wins" by collapsing spread is
  not a win — same calibration rule as the filtering plan's acceptance
  gates).
* If a P1 experiment produces a null result, it still gets its slide — as a
  measured negative with the cost numbers, which is a contribution at this
  venue. If a P1 experiment is *not run*, its slide is cut entirely rather
  than left as a promise.
