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

_(record here as they occur)_
