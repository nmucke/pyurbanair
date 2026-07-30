# Master plan: turbulence-aware evaluation in the ESMDA pipeline

> **Status: living implementation plan.** Companion to the research document
> [docs/plans/esmda_turbulence_evaluation.md](../esmda_turbulence_evaluation.md)
> (the *what and why* — metric definitions, formulas, figure conventions,
> literature). This file is the index and status board; the per-phase plans
> in this directory carry the implementation detail.

## Instructions for implementing agents

- **Before starting a work package:** read the phase plan for it *and* the
  metrics-doc sections it cites. Phase plans quote line numbers as of
  2026-07-29 — verify against the current tree; the cited function/variable
  names are the stable anchors.
- **As each WP lands:** update the status table below (status + PR number),
  and record any deviation from the plan (renamed keys, changed signatures,
  descoped items) in the **Deviations** section of that phase's plan file.
  Do not rewrite phase-plan bodies to match what was built — the deviation
  log is the record.
- **Follow CLAUDE.md workflow rules:** branch first, `pixi run -e dev
  pre-commit` before committing, keep `docs/` in sync when artifacts or
  configs change (phases 2–3 explicitly change artifacts), never commit
  large artifacts.
- **Cross-phase invariants** (repeated in each phase plan; violations are
  review-blockers):
  1. `run_summary.yaml` keys are additive only — existing key paths are
     hard-coded in `scripts/figure_creation/`. Changing an existing key's
     values or semantics requires bumping the `metrics_version` marker
     (introduced in phase 0) so tooling can detect cross-version
     comparisons; purely additive keys never bump it.
  2. Full-ensemble window state files are never `.load()`ed — stream
     member-at-a-time or z-slab-wise, ≤2 reader threads.
  3. Every new metric/figure no-ops gracefully when its inputs are absent
     (old run dirs, flags off, smoke shape).
  4. Fair estimators and corrected spread formulas (phase 0) are the only
     ones new code may use.

## Shape of the change

`scripts/run_esmda_pipeline.sh` itself stays untouched (it only resolves the
run dir and chains the stages). Work lands in:

| Stage | Files | Phase(s) |
|---|---|---|
| 1. run | `scripts/esmda/run_esmda.py` + `libs/data-assimilation` smoother | 2 (persistence), 3.2 (probe artifacts) |
| backends | `libs/pylbm` wrapper; possibly `pyudales`/`pypalm` output handling | 3.2 (high-rate probes) — the only backend-touching work in the effort |
| 2. metrics | `scripts/esmda/compute_esmda_metrics.py` | 0–3 |
| 3. figures | `scripts/esmda/make_esmda_figures.py` + `src/pyurbanair/plotting.py` | 1–3 |
| sweep | `scripts/figure_creation/compute_sweep_metrics.py` | 1 (WP1.3 only: it now shares the ESMDA stage's streaming sensor extraction instead of keeping a copy) |

Shared math goes in two new reusable modules —
`src/pyurbanair/utils/ensemble_scores.py` (probabilistic scores) and
`src/pyurbanair/utils/turbulence_stats.py` (flow statistics) — so the sweep
pipeline and `scripts/figspec/` can reuse them; `_esmda_common.py` stays
orchestration glue. Figures reuse `scripts/figspec/style.py` conventions.
As of WP1.3 the reuse also runs the other way: `_esmda_common` itself is the
shared *orchestration* the sweep stage imports (`ensemble_sensor_series`,
`truth_sensor_series`, `open_truth`), which is what removed the sweep stage's
own full-ensemble `.load()` of the window state files (one full-ensemble load
remains in the *filtering* pipeline, on `state_history.nc`, and is out of scope
— see `docs/scripts_and_configs.md` §2.4). Stage 3 also gains a sanctioned
input file as of WP1.3: it re-derives everything from raw artifacts today, and
`eval_fields.nc` exists so that WP1.4's field figures do not have to repeat the
streaming pass.

## Phases and status

| WP | Content | Plan | Size | Status | PR |
|---|---|---|---|---|---|
| 0.1 | Fair CRPS / energy-score estimators + CRPSS vs prior + `metrics_version` | [phase0](phase0_correctness_fixes.md) | S | merged | [#97](https://github.com/nmucke/pyurbanair/pull/97) |
| 0.2 | Spread–skill: RMS-of-variances + Fortin factor (callers updated) | [phase0](phase0_correctness_fixes.md) | S | merged | [#97](https://github.com/nmucke/pyurbanair/pull/97) |
| 0.3 | Duplicate-member guard (`ensemble_health`) | [phase0](phase0_correctness_fixes.md) | S | merged | [#97](https://github.com/nmucke/pyurbanair/pull/97) |
| 1.0 | `run.metrics` config block + module skeletons | [phase1](phase1_postprocessing_metrics.md) | S | merged | [#99](https://github.com/nmucke/pyurbanair/pull/99) |
| 1.1 | Parameter bundle: z-scores, PIT, coverage, contraction, joint directions | [phase1](phase1_postprocessing_metrics.md) | S | merged | [#99](https://github.com/nmucke/pyurbanair/pull/99) |
| 1.2 | Statistics-space sensor scoring + Wasserstein w/ self-distance floor | [phase1](phase1_postprocessing_metrics.md) | M | merged | [#100](https://github.com/nmucke/pyurbanair/pull/100) |
| 1.3 | Shared-pass mean-field / Reynolds-stress layer + station columns + hit rate/FAC2/FB/NMSE + `eval_fields.nc` | [phase1](phase1_postprocessing_metrics.md) | M–L | in review | — |
| 1.4 | Figures: P1, S1, S5, F1, F2, S2/S3 | [phase1](phase1_postprocessing_metrics.md) | M | not started | — |
| 2.1 | Persist obs / per-iteration + posterior pred-obs / per-iteration params | [phase2](phase2_obs_persistence.md) | M | not started | — |
| 2.2 | Diagnostics: `O_N` vs ½, innovations, contraction-vs-achievable, SNR/DFS, obs-space scores | [phase2](phase2_obs_persistence.md) | M | not started | — |
| 2.3 | Figures: D1–D4, full P2 | [phase2](phase2_obs_persistence.md) | M | not started | — |
| 3.1 | Barcelona validation (held-out) sensors | [phase3](phase3_run_upgrades.md) | XS | not started | — |
| 3.2 | High-rate probes + spectra / two-point / `S₃` / reverse-flow layer + S4 figure | [phase3](phase3_run_upgrades.md) | L | not started | — |
| 3.3 | Optional: prior-state run docs, representativeness error in `C_D` | [phase3](phase3_run_upgrades.md) | S | not started | — |

Sequencing: 0 → 1 strictly (phase 1 builds on the fair estimators); 2 after
1 (its figures reuse phase-1 plumbing); 3.1 anytime after 1; 3.2 last
(backend-touching); one PR per row unless a phase plan says otherwise.
Phases 0–1 apply retroactively to existing run dirs; phase-2+ metrics only
cover runs executed after WP2.1, older dirs degrading gracefully to the
phase-1 set.

Cross-cutting cautions:

- WP0.1 shifts all historical CRPS/energy-score numbers by ~O(1/M) — sweep
  comparisons across the boundary are invalid; the boundary is
  machine-detectable via the `metrics_version` key, and the comparison
  scripts warn on version mixing (see phase 0).
- The smoke test shape degenerates several diagnostics — guard with `null` +
  log, don't special-case. It degenerates on **two** axes, and phases 1.2/1.3
  hit both: the 2-member ensemble kills ddof=1 variances, order-statistic bands
  and binned spread–skill, while the **3 frames per window** kill every
  moving-block bootstrap (20 requested blocks against 3 frames gives a block
  length below 2), so all the sampling floors and the hit-rate allowance are
  `null` on a smoke run by construction. A wiring test on a larger synthetic run
  dir, not the smoke case, is what pins those code paths.
- **Do not benchmark a layer at the smoke shape.** The corollary of the above,
  and it cost WP1.3 a review round: a degenerate input often makes the expensive
  path return early, so the smoke measurement is of the code that *did not run*.
  WP1.3's truth bootstrap measured 0.1 ms at 3 frames and 13.2 s at the same
  cell count with 36 — a factor 1e5 the budget never saw. Measure one realistic
  shape before a level becomes a production default.
- ~28 pre-existing test failures are baseline (see auto-memory) —
  stash-verify before blaming new changes.
