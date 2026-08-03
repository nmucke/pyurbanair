# Phase 1 — correctness fixes, the slimmed metric set, and first figures

> Part of the ESMDA-evaluation effort. Master plan:
> [master_plan.md](master_plan.md). Rationale: §3, §4.1–4.2, §6 and the
> figure list of [../esmda_turbulence_evaluation.md](../esmda_turbulence_evaluation.md).
> Requires phase 0. Pure post-processing — no run-stage or artifact
> changes. One PR per WP. Much of WP1.1–1.3 exists on the rollback
> branches (see master plan) — cherry-pick the math and tests, adapt paths
> to `evaluation.*`, drop what the slimmed set no longer needs (PIT/coverage
> tables, joint directions, Wasserstein, FAC2/FB/NMSE).
>
> **Implementer: update the master_plan.md status table per WP; record
> deviations at the bottom of this file. Follow the master plan's
> Implementation process: Opus 5 agent team, tests in the same PR, two
> adversarial review rounds, CI green before merge.**

## WP1.1 Correctness fixes (do first — everything later builds on these)

All sites are in `evaluation.scores` after the phase-0 move.

1. **Fair pairwise estimators** (Ferro 2014): the CRPS spread term and the
   energy-score pairwise term currently average over all `M²` pairs
   including the zero diagonal; divide the pairwise sum by `M(M−1)`
   instead. After the WP0.2 dedup there should be exactly two sites: the
   scalar/per-knot CRPS and `energy_score`. Guard `M < 2`.
2. **CRPSS vs prior:** where `parameter_metric_summary` emits
   `rmse_reduction_vs_prior`, also emit
   `crps_reduction_vs_prior = 1 − mean(post_crps)/mean(prior_crps)`.
3. **Spread–skill:** `summary_scalars`' `time_avg_spread` becomes RMS of
   per-knot stds (root of average variance, not mean of stds); the
   `spread_skill` function gains a required `n_members` and the Fortin
   factor: `√((M+1)/M) · RMS(spread) / RMS(rmse)`.
4. **Duplicate-member guard:** `ensemble_uniqueness(members: (M, K)) ->
   dict` with `n_unique` by **exact** row match (resample clones are
   bit-identical; near-duplicates are the min/median pairwise distance's
   job). Wire into `compute_esmda_metrics.py` → `run_summary.yaml`:

   ```yaml
   ensemble_health: {n_members: 50, n_unique: 48, n_unique_per_window: [...]}
   ```

   Warn when `n_unique < n_members`; later scores use `n_unique` in the
   `M(M−1)` corrections.
5. **`metrics_version: 2`** as a top-level `run_summary.yaml` key (absent/1
   = biased-estimator semantics); `compare_sweep_results.py` and
   `compare_state_runs.py` warn on version mixing. PR description carries
   the breaking-number note (~2 % shift at M=50; historical comparisons
   invalid).

Tests (cherry-pick from the rollback branch): fair CRPS vs the analytic
Gaussian CRPS at `M=10⁴`; unbiasedness direction at small `M`;
spread–skill ≈ 1 on exchangeable synthetic data; `ensemble_uniqueness` on a
constructed clone.

## WP1.2 Parameter bundle (metrics doc §3)

New `evaluation.scores.parameter_bundle(post, prior, truth)` operating on
the `(M, K)` arrays `compute_esmda_metrics.py` already loads; per parameter
and pooled:

- normalized error `(θ̄ᵃ − θ*)/σᵇ`,
- z-score `(θ* − θ̄ᵃ)/σᵃ` (pooled mean/std should look ~N(0,1)),
- contraction ratio `σᵃ/σᵇ`,
- fair CRPS + CRPSS vs prior (from WP1.1).

`run_summary.yaml` **already has** a `parameter_metrics:` block (written by
`compute_esmda_metrics.py` from `parameter_metric_summary`) — extend its
per-parameter entries with additive subkeys (`z_score`,
`contraction_ratio`, `crps_fair`, `crpss_vs_prior`) plus a pooled summary
subkey; do not clobber or duplicate the key. Nothing else — no
PIT/coverage tables, no correlation eigen-analysis (deliberately excluded,
metrics doc §9).

## WP1.3 Sensor-statistics scoring (metrics doc §4.2)

The verification object is window statistics, not time series. Series
extraction stays in `scripts/esmda/_esmda_common.py`
(`ensemble_sensor_series` / `truth_sensor_series` — see phase 0);
`evaluation.sensors` consumes the extracted arrays.

- **Streaming rewrite of `ensemble_sensor_series`** (first subtask): it
  currently `.load()`s each multi-GB window state file whole — the one
  legacy violation of master-plan invariant 2. Rewrite member-at-a-time
  before making it load-bearing here.
- Per member, window, sensor, component: window mean and window variance
  (of the velocity magnitude and components).
- Score `{T_m}` vs `T*` with fair CRPS, z-score, and rank — separately for
  the `assimilation` and (when defined) `validation` sensor sets, prior
  and posterior (prior needs `run.save_prior_state` runs or the prior
  sensor series where available; no-op with a log line otherwise).
- **Identifiability guard:** report (across-member spread of `T`) /
  (within-member block-bootstrap std of `T`) per statistic; `< ~3` flags
  the statistic as unidentifiable at this window length. Block bootstrap
  helper goes in `evaluation.turbulence` (blocks ≥ 1 integral time scale,
  estimated from the series autocorrelation).

Emit under `sensor_statistics: {assimilation: ..., validation: ...}`.

## WP1.4 Streaming mean fields + hit rate (metrics doc §4.1)

- `evaluation.turbulence.MomentAccumulator`: the one class in the library.
  Accumulates `Σu_i`, `Σu_i u_j`, `n` over time chunks per member →
  time-mean `U_i`, TKE `k = ½⟨u_i′u_i′⟩`, and `⟨u′w′⟩`. Components are
  co-located to cell centers first for staggered (uDALES) states; extend
  the existing streaming pattern (never `.load()`, ≤2 reader threads).
- Reduce across members and write `eval_fields.nc` (truth + ensemble
  reductions only — **not** per-member fields) so figures don't re-stream.
  Keep it genuinely small at Barcelona resolution: float32, per-cell
  ensemble mean/std only; nested quantiles stored only at the S1 station
  columns.
- `evaluation.scores.hit_rate(pred_mean, true_mean, D=0.25, W)` over fluid
  cells; `W` from block-bootstrap of the truth series at a sample of
  points (`W = σ_u/√N_eff`). Emit `field_metrics: {hit_rate_prior?,
  hit_rate_posterior}` (prior only when prior state was saved).
- Cross-grid runs: interpolate truth to the assim grid (existing
  pattern) before scoring.

## WP1.5 Figures (metrics doc §7)

One function per figure in `evaluation.figures`, called from
`make_esmda_figures.py`; conventions from `evaluation.style` (truth black,
prior grey, posterior teal, nested quantile bands, **shared limits/norms
across every prior/posterior pair**). Each no-ops when inputs are absent.

| ID | Function | Content |
|---|---|---|
| P1 | `plot_parameter_marginals` | prior vs posterior violins/box+strip per parameter, truth dashed, z-score annotated; y-limits include the prior |
| S1 | `plot_station_profiles` | rows = `ū/U_ref`, TKE; columns = stations (upstream/in-canyon/wake); truth line + posterior band + prior band; roof line `z/H=1`; inset plan view. Reads `eval_fields.nc` |
| S5 | `plot_sensor_fans` | sensor time series: quantile fan + observation markers ±σ_o, window boundaries; assimilated and held-out sensors in labeled separate columns. Pre-WP2.1 the noisy assimilated obs are not persisted — use recomputed clean obs (truth + obs operator) ± `esmda.obs_error_std` and say so in the caption; switch to the persisted obs once WP2.1 lands |
| F1 | `plot_mean_slices` | 2–3 planes × (truth \| prior mean \| posterior mean \| posterior − truth); shared norm for the first three, symmetric diverging norm for the difference; averaging window annotated. Never instantaneous |
| D1 | `plot_rank_histogram` | rank of `T*` within `{T_m}` from WP1.3, pooled over sensors, windows, and statistics (per-window statistics are already ~independent samples — no sub-sampling needed), prior \| posterior, uniform line + binomial band. Pooled counts are small (~50–300 ranks) — coarsen to ~10 rank bins, never M+1 |

## Tests

- Unit tests per WP (parameter bundle on a linear-Gaussian toy where the
  z-scores are analytic; hit rate on constructed fields; accumulator vs a
  direct `.mean()`/`.var()` on a small in-memory dataset).
- Integration: metric + figure stage on a smoke run dir → new YAML blocks
  present and finite, figures render; on a pre-phase-1 run dir → no crash,
  absent blocks logged. M=2 smoke degeneracies emit `null` + log.

## Acceptance

- Fair estimators everywhere, `metrics_version: 2`, `ensemble_health`,
  `parameter_metrics`, `sensor_statistics`, `field_metrics` in a smoke
  `run_summary.yaml`; P1/S1/S5/F1/D1 render on a real run; all additive
  keys; pre-commit clean.

## Deviations

**WP1.1**

- `METRICS_VERSION` lives in `evaluation.scores`, not in the metric script.
  It marks the *estimator* semantics, which is the library's property, and
  three writers need it (ESMDA, filtering, `compute_sweep_metrics.py`);
  keeping it in one script would have meant importing across script
  packages.
- `compute_filtering_metrics.py` also emits `metrics_version`. It was not in
  the WP's scope, but it writes a `run_summary.yaml` using the same
  estimators, so leaving it unmarked would have made unmarked mean two
  different things.
- `compute_sweep_metrics.py` carried byte-identical private copies of
  `series_stats` and `parameter_metric_summary` (a WP0.2 dedup miss). They
  are deleted in favour of the library functions, so `metrics.yaml` gets the
  CRPSS as well. That file also *downgrades* its own `metrics_version` when
  it falls back to copying an old run's sensor scores instead of recomputing
  them.
- `crps_ensemble` computes the pairwise term from sorted samples
  (`sum_{i,j}|x_i - x_j| = 2 sum_i (2i - n + 1) x_(i)`) rather than from the
  `(M, M, K)` difference tensor. Algebraically identical and asserted as
  such in the tests, and O(n log n) rather than O(M²K) in a function called
  per timestep. Two consequences of the rewrite are handled explicitly and
  pinned by tests: the weights sum to zero, so the sum is centred on the
  median before accumulating (without it float32 data offset from the origin
  — an inflow angle near 270° — loses ~4 orders of magnitude), and both
  terms accumulate in a signed floating dtype (the raw form wrapped on
  unsigned input).
- `ensemble_uniqueness` counts unique rows **bitwise** rather than by value,
  so two identical all-NaN rows (a diverged member cloned by the resampling
  policy — the case the guard exists for) count as one; and it reduces only
  over finite pairwise distances, so a NaN member cannot blank out
  `min_pairwise` / `median_pairwise` for the whole ensemble. It also
  computes distances row-by-row instead of via an `(M, M, K)` tensor, which
  reached hundreds of MB on long runs.
- `_ensemble_health` degrades instead of aborting (invariant 3): an
  unreadable window costs its own count and a log line, and an assembled
  posterior it cannot read omits the block entirely.
- `_skill_score` logs when its reference is zero. At `M = 2` — the CI smoke
  shape — the fair CRPS is *identically* zero whenever the truth is
  bracketed by the two members, so `crps_reduction_vs_prior` is routinely
  `null` there; per the master plan that is guarded with null + a log, not
  special-cased.
- `compute_sweep_metrics.py` **omits** the sensor block when it cannot
  recompute it (no `truth_access.yaml`) rather than copying the run's own
  scores forward. A single `metrics_version` cannot describe a file whose
  parameter block is fair and whose sensor block may not be — the mixing
  guard would then either stay silent across the boundary or warn about runs
  that are in fact comparable.
- The three `scripts/figure_creation/` scripts touched here carry 26
  pre-existing strict-mypy errors that pre-commit only surfaces once a
  commit stages them; they get the repo's standard blanket
  `# mypy: ignore-errors` waiver (as `scripts/esmda/_esmda_common.py`
  already does) rather than an unrelated annotation pass.
- `ensemble_health` also reports `min_over_median_pairwise` (from the
  rollback branch): exact row matching cannot see a *near*-duplicate, and
  the ratio is what flags one.
- **Deferred:** scores still divide by the nominal `M`, not `n_unique`. The
  plan's "later scores use `n_unique`" is read as forward-looking — the
  count is reported and warned on now, and no metric consumes it until a WP
  that has an actual duplicated-ensemble run to validate against.
