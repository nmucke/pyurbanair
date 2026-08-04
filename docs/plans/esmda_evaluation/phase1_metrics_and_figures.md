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
  CRPSS as well.
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
- `compute_sweep_metrics.py` carries forward only the *estimator-independent*
  sensor scores when it cannot recompute them (no `truth_access.yaml`): the
  RMSE of the ensemble mean has no pairwise term and stays comparable, the
  CRPS keys are dropped. A single `metrics_version` cannot describe a file
  whose parameter block is fair and whose sensor block is not — the mixing
  guard would then either stay silent across the boundary or warn about runs
  that are in fact comparable — but dropping the RMSE with it would silently
  erase those runs from the comparison panels.
- The analytic-Gaussian CRPS test runs at **M = 50**, not the `M = 10^4` the
  plan names. At M=10⁴ the biased estimator's error is ~6e-5, inside any
  tolerance loose enough for the Monte-Carlo noise, so that test cannot tell
  the two estimators apart; averaging 20 000 independent M=50 ensembles is
  both tighter and discriminating (the biased form is then 0.0113 out).
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

**WP1.2**

- **No `crps_fair` / `crpss_vs_prior` keys.** WP1.1 already emits exactly
  those two numbers under the names `crps` and `crps_reduction_vs_prior`.
  Re-emitting them would put two names for one value in the block invariant
  1 forbids clobbering, and leave the comparison scripts to guess which is
  authoritative. The bundle adds only what is new: `z_score`,
  `normalized_error`, `contraction_ratio`.
- The pooled entry is `parameter_metrics.pooled`, beside the parameter
  names as the plan specifies. Safe because `_PLOTTED_PARAMS` is a closed
  whitelist that cannot contain `pooled`, and every consumer indexes this
  block by an explicit parameter name rather than iterating it (checked:
  `visualize_run`, `visualize_state_run`, `make_figures_block_a/b`,
  `compare_sweep_results`, `compare_state_runs`, `compare_param_vs_state`).
- z-scores get their own reduction, `z_score_stats` → `{n, mean, std,
  expected_std, max_abs}`, not `series_stats`. A z set is judged by its
  *distribution*, and `series_stats` has no std while its `final` (the last
  knot) means nothing for a calibration set. `std` is `null` below two finite
  knots rather than 0.
- **The z-score reference is not 1** at any ensemble size this repo runs, and
  `z_score_stats` therefore takes a required `n_members` (the `spread_skill`
  precedent from WP1.1: no size makes the missing correction right, so a
  default would bake in the misreading). `z = (θ*−θ̄ᵃ)/σᵃ` compares the truth
  against a mean *and* a spread estimated from the same `M` members, so for a
  calibrated ensemble `z ~ √(1+1/M)·t₍M₋₁₎`, with std
  `√((1+1/M)(M−1)/(M−3))` — 1.02 at `M=64`, 1.26 at `M=8`, **infinite at
  `M ≤ 3`**. `calibrated_z_std` emits that reference alongside the sample
  `std`. Every constant is confirmed against a 400k-replicate simulation of
  the estimator.
  Below four members the **whole `z_score` entry** is `null` plus a log, per
  the master plan's guard-with-null rule. Round 1 nulled only `std` and
  pointed the reader at `mean` / `max_abs`; round 2 showed that merely
  relocates the misreading, because at `M = 2` the reference distribution is
  *Cauchy* — no moment converges, so a perfectly calibrated smoke run gives
  `|mean| > 5` about 13 % of the time and `max_abs > 3` about 68 % of it. The
  variance condition (`ν > 2` ⟺ `M ≥ 4`) had been applied; the mean condition
  (`ν > 1`) had not. `contraction_ratio` is well defined at every size and is
  what the log points at.
  A `frac_abs_gt_2` key was dropped for the same reason: its calibrated value
  is also M-dependent (0.05 at `M=64`, 0.10 at `M=8`, 0.35 at `M=2`) and a
  closed-form reference for it needs a t-CDF, i.e. scipy in a leaf library.
- Degenerate scales produce `nan` → `null`, never `inf`. Both cases are
  real: a **pinned** parameter has `σᵇ = 0` by construction, and a collapsed
  posterior has `σᵃ = 0`. Note what this does *not* buy: both reductions
  filter on `np.isfinite`, which drops `inf` and `nan` alike, so the YAML
  reads `null` either way. The guard is for the per-knot **arrays** the
  WP1.5 figures plot, where one `inf` rescales an axis into uselessness —
  and it keeps numpy's divide-by-zero warning off every pinned-parameter
  run. (An earlier revision of this note claimed an `inf` would reach the
  YAML as `.inf`; review round 1 disproved it by running the unguarded
  form.) The run-through-YAML test now carries a pinned parameter, so it
  actually exercises the `null` path instead of asserting about a run that
  cannot produce one.
- `parameter_bundle` returns `posterior_std` / `prior_std` but the summary
  does not write them. They are what tells a large `|z|` caused by bias from
  one caused by collapse, which figure P1 (WP1.5) annotates; the YAML has
  `contraction_ratio` for the same reading and does not need both.
- The truth alignment moved into a shared `_aligned_parameter_members`
  generator used by both `compute_parameter_metrics` and the new
  `compute_parameter_bundles`, so the accuracy and calibration blocks can
  never end up describing different knots. It carries the pre-existing
  guard (a prior sampled on a different knot grid is dropped rather than
  broadcast), which now governs the bundle too.
- One WP1.1 test assertion was relaxed:
  `test_parameter_summary_omits_prior_keys_without_a_prior` pinned the
  no-prior key set with `==`, which an additive-only block cannot support.
  It now names the prior-derived keys it is actually about.
- `compute_filtering_metrics.py` and `compute_sweep_metrics.py` gain the
  block for free — both call `parameter_metric_summary`. Same reasoning as
  WP1.1's `metrics_version` deviation: one function, one meaning.
- **Considered and not done** (review round 1): `parameter_metric_summary`
  walks `_aligned_parameter_members` twice, once for the accuracy half and
  once for the bundles. Collapsing it into one pass would need the
  per-parameter body of `compute_parameter_metrics` extracted into yet
  another helper, to save a `np.interp` and a `.values` read over arrays of
  at most `M × few-thousand` floats. The redundancy is cheaper than the
  abstraction; the shared generator already guarantees the two halves see
  identical knots, which was the only correctness stake.
- `parameter_metric_summary` reads the ensemble size with
  `sizes.get("ensemble", 0)`, not `sizes[...]`. Invariant 3: a posterior
  without an ensemble dimension (an old ensemble-mean-only artifact) selected
  no parameters and returned `{}` before this WP, and a `KeyError` there would
  now abort the whole metric stage — costing the state and sensor blocks too,
  not just the parameter one. Found in review round 2.
- Known caveat, not fixed: `z_score.std` is a *sample* std over `n` knots and
  its own sampling range is wide when `n` is small (5th–95th percentile
  0.06–1.99 at `n = 2` for a calibrated `M = 64` ensemble). A static parameter
  over two windows has `n = 2`. Documented in `scripts_and_configs.md` with
  the advice to prefer the `pooled` entry there; an `n` floor was not added
  because a plausible-interval key needs a χ² quantile, i.e. scipy in a leaf
  library, and the honest fix for a 2-window run is more windows.
- Known caveat, not fixed: a single non-finite member blanks the whole
  calibration block for that parameter (mean and spread are poisoned at every
  knot at once), which is the *opposite* convention from WP1.1's
  `ensemble_uniqueness`, which reduces over finite pairs only. Left as is —
  a NaN member means the member is not a sample from the posterior, so
  dropping it silently would report a calibration for an ensemble that does
  not exist — but the next WP that scores members should pick one convention
  deliberately.
- Known caveat, not fixed: the parameter artifacts are float32 on disk, so a
  hard-contracted posterior (`σᵃ ~ 1e-4` on an inflow angle near 270°) gets
  an O(10 %) quantization error in `z_score`. Unlike WP1.1's CRPS
  cancellation this is not an algorithmic defect — the precision is gone
  before the library sees the data — so it needs an artifact-precision
  change, not an estimator change.

**WP1.3**

- **The streaming rewrite is a slice loop, not a chunked read.** The plan says
  "member-at-a-time"; the implementation opens the window file lazily and
  materializes `ds[["u","v","w"]].isel(ensemble=slice(m, m+1))` per member,
  because `interpolate_dataarray_at_points` reads `.values` and cannot consume
  a lazy array. Peak memory is three components of one member's window instead
  of the whole ensemble's. Both sensor sets are interpolated from the slice
  already in memory, so a validation set costs no extra read — the previous
  code re-walked the loaded dataset per set, which was free only because it had
  loaded everything. Inertness is pinned by a test that reruns the pre-WP1.3
  whole-file path and compares values, dims and the global time axis.
- **Windows are cut by the time coordinate, not by frame count.** The plan does
  not say which; frame count is the tempting one because `truth_access.yaml`
  carries `n_per_window`. It is wrong here: that count describes the *truth's*
  cadence, and the assimilation writes at its own `output_frequency`, so one
  reduction cannot serve both. Both series already carry a global axis on which
  window `w` starts at `w*sim_time` (the extraction rebases them), so
  `floor(t/sim_time)` bins either one. A frame exactly on a boundary opens its
  window, matching the run's own `[w·sim_time, (w+1)·sim_time)` convention.
- **`block_bootstrap_std` is one vectorized function, not a scalar/batch pair.**
  The rollback branch shipped both (`block_bootstrap_std` +
  `block_bootstrap_std_batch`) with a shared index matrix and a test pinning
  them equal. One function taking `(..., n_time)` and an axis-taking reducer
  removes the drift risk instead of testing for it; the 1-D case is the 0-d
  result. The cost is the statistic contract — `statistic(x, axis=-1)`, not
  `statistic(1-D)` — which is checked and raises rather than silently
  mis-reducing. Rows containing a non-finite sample return `nan` rather than
  falling back to a per-row gap-dropping path: a row with gaps has a different
  finite count, hence a different block length, which is exactly the sharing
  that makes the vectorization possible.
- **Statistic set: mean and variance of u, v, w and |U|** — eight keys, scored
  separately. The plan says "window mean and window variance (of the velocity
  magnitude and components)"; making the quantity a dimension rather than four
  copies of the reduction keeps `evaluation.sensors` to two public functions.
  No TKE key: it is `½·Σ_i var(u_i)`, i.e. a fixed combination of three keys
  already present, and the resolved-stress machinery that makes TKE worth its
  own entry lands in WP1.4.
- **The CRPS and identifiability entries are `series_stats` over *windows*, the
  z-scores pooled.** A CRPS averaged over sensors is a series whose `final` is
  the end-of-run value, which is the reading every other summary key already
  has; a z-score set is a distribution and gets `z_score_stats` (the WP1.2
  precedent, including the `expected_std` reference and the null below four
  members). The identifiability ratio is emitted per window for the same reason
  the CRPS is: a statistic can become identifiable as the flow spins up.
- **Ranks are written raw, not binned.** Figure D1 (WP1.5) wants ~10 bins over
  a pooled set, but binning at metric time bakes the choice into the artifact.
  The list is bounded — sensors × windows per statistic, ~1600 ints even at
  Barcelona scale — so the figure can pool and bin as it likes. Ties get a
  seeded uniform draw rather than a fixed rule: a collapsed ensemble is all
  ties, and a deterministic tie-break piles them at one end of the histogram
  and reads as a bias that is not there.
- **The identifiability floor is a median over members, not a mean.** One
  diverged member's window can have a wild sampling std, and the floor is meant
  to describe a typical member. `< 3` logs a warning naming the statistic,
  matching the metrics doc's "if not ≫ 1"; the threshold is a flag, not a gate —
  nothing is suppressed on it.
- **The whole `identifiability` key is absent, not null, when no floor was
  measured.** The bootstrap needs ~21 frames per window at the default 20
  blocks and the CI smoke shape has four, so this is the common case in tests.
  An unmeasured floor is *unknown*, and a ratio over it would read as
  "infinitely identifiable" — the opposite of what a short window means.
- **No config knobs.** `n_blocks` / `n_resamples` are function defaults, not
  Hydra keys. The rollback branch had a `resolve_metrics_settings` validator
  for them; nothing in the slimmed set needs to vary them per run, and the
  defaults are the only values any caller passes.
- **Prior scoring is all-or-nothing per run.** A partially written set of
  `window_*_prior_state.nc` (an interrupted job) is treated as absent rather
  than scored over the windows that exist: a prior on three windows against a
  posterior on ten would put two different horizons inside one skill score.
  Logged at INFO, not WARNING — `run.save_prior_state` is off by default, so
  absence is the normal case, not a fault.
- **Deferred:** the e2e ESMDA tests all run under `run.skip_viz=true`, so none
  of them reaches this block. The wiring is covered instead by a synthetic
  run-dir test that builds the artifacts `compute_metrics` reads and runs the
  real non-`skip_viz` path. A smoke run that exercises the sensor stages for
  real is a gap that predates this WP (see the same note in auto-memory) and is
  not closed here.
