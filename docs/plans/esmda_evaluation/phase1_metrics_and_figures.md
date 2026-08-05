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
  reduction cannot serve both. Both series carry a global time axis, so
  `floor(t/sim_time)` bins either one — though by different routes, and the
  round-1 wording "the extraction rebases them" was only half right (found in
  round 2): `ensemble_sensor_series` does rebase each window onto `w·sim_time`,
  while `truth_sensor_series` keeps the truth's own `t − t_offset` axis and
  relies on `t_offset = start_time` plus a uniform truth cadence. The
  consequence is that the first kept truth frame sits at δ ∈ [0, dt) rather
  than exactly at 0 — under one frame, so it changes no window assignment. A
  frame exactly on a boundary opens its window, matching the run's own
  `[w·sim_time, (w+1)·sim_time)` convention.
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
- **Ranks are written as `M+1` counts, not as the raw rank list** (revised in
  review round 2). The first cut emitted the raw ranks, on the reasoning that
  binning at metric time bakes the choice into the artifact and the list is
  "bounded — ~1600 ints even at Barcelona scale". Measurement killed that: the
  YAML writer runs `default_flow_style=False`, so every int gets its own line
  and the block came to **7053 lines / 94 KB** at W=10, S=20, two sensor sets,
  prior and posterior — swamping a file people read. Counts per rank are what a
  rank histogram *is*, lose nothing figure D1 uses (it pools over sensors and
  windows anyway), stay exact so D1 can still coarsen to its ~10 bins, and are
  ~3x smaller. Ties get a seeded uniform draw rather than a fixed rule: a
  collapsed ensemble is all ties, and a deterministic tie-break piles them at
  one end of the histogram and reads as a bias that is not there.
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
- **Window boundaries carry a 1e-9-of-a-window tolerance** (review round 1).
  The extraction rebases window `w` to start at exactly `w·sim_time`, but
  `(w·sim_time)/sim_time` is not exactly `w` in IEEE double for most values of
  `sim_time`: at 10.76 (200 frames at pylbm's default 0.0538 s cadence) windows
  7 and 14 come out a ULP short and a bare `floor` scores their opening frame
  into the *previous* window. About a fifth of two-decimal `sim_time` values
  misbin at least one boundary in the first ten windows, and the truth's global
  axis drifts independently of the ensemble's, so the two need not even misbin
  the same frame. Latent on the shipped case configs (`simulation_time: 300.0`
  is exact) but `time.simulation_time=` is a routine CLI override.
- **Not fixed, considered:** `window_statistics_summary` does not `xr.align`
  the truth against the members. `crps_ensemble` catches a size mismatch, but a
  sensor *permutation* would misalign silently. Both sides come from the same
  `sensor_sets[name]` tuple in the same process, so there is no path that
  produces one today; an alignment call would be guarding against a caller that
  does not exist.
- **`_skill_score`'s `nan` reference is nulled at this call site only.** An
  all-empty prior reduces to `nan` and reaches the YAML as `.nan`, against
  WP1.2's "degenerate scales produce `null`, never `inf`" convention. The same
  is true of `parameter_metric_summary`'s `prior_rmse_mean` / `prior_crps_mean`
  and predates this WP; fixing it there would change an existing key's written
  value, which invariant 1 puts behind a `metrics_version` bump. The new block
  guards locally instead.
- **Deferred:** the e2e ESMDA tests all run under `run.skip_viz=true`, so none
  of them reaches this block. The wiring is covered instead by a synthetic
  run-dir test that builds the artifacts `compute_metrics` reads and runs the
  real non-`skip_viz` path. A smoke run that exercises the sensor stages for
  real is a gap that predates this WP (see the same note in auto-memory) and is
  not closed here.

**WP1.3, review round 2.** The round-1 fixes were themselves unreviewed code;
these are findings against them, and against what round 1 missed.

- **The round-1 "absent `ensemble` axis" guard was unreachable.** It sat in
  `window_statistics_summary`, but `vector_sensor_metrics` runs first in the
  same script and raises on the same input — so the pipeline still died and
  `run_summary.yaml` was still never written. The guard now lives in
  `compute_esmda_metrics.py`, where one filtered dict drops the set from *both*
  sensor blocks; the library keeps its own return-`{}` path for other callers.
  The test that "covered" this called the library directly and bypassed the
  script, which is exactly how the gap survived the round it was written in.
- **The `try/except` round 1 added around the scoring is removed again.**
  Everything inside it was pure computation on series already in memory — the
  only I/O, the prior read, has its own guard — so `OSError` was unreachable
  there and `ValueError`/`KeyError` could essentially only be bugs.
  Demonstrated: a one-character key typo in the scorer made the stage exit 0
  and write `sensor_statistics: {}`. A broken scorer shipping a green run is
  worse than a crash. Invariant 3 is about *absent inputs*, not about absent
  correctness.
- **The identifiability warning thresholds on the value that is written.** It
  had used `nanmedian` over all W×S knots while emitting `series_stats` of the
  per-window sensor mean, so on skewed sensors the log and the artifact could
  disagree — and `scripts_and_configs.md` tells the reader to compare the
  emitted number against 3.
- **A frame-truncated prior nulls the skill score.** "All-or-nothing per run"
  was enforced at *file* granularity only; a prior file that exists but stops
  short leaves its last window empty, and the skill then averaged a 2-window
  prior against a 3-window posterior — the two-horizons comparison the
  all-or-nothing rule exists to prevent.
- **`label` now reaches the "no frames in window" warning too**, which round 1
  left unlabelled while fixing the identifiability one.
- **Kept against the reviewer's advice:** the `lambda w, fn=fn:` default-arg
  capture in `window_sampling_std`. It is dead today — `_windowed` invokes the
  callback synchronously, so `fn` cannot rebind first — but it is the standard
  idiom, costs five characters, and removing it would make correctness depend
  on a non-obvious property of a different function.
- **Four mutants survived the round-1 suite** and now have tests: the z-score
  sign; `ddof=1` in the z denominator (ddof=0 rescales by √(M/(M−1)) and breaks
  the `expected_std` reference the WP1.2 machinery is built on); the axis the
  bootstrap floor is reduced over (every earlier test used homogeneous data, so
  a floor constant across sensors scored the same); and the `isfinite` filter on
  ranks (`nan_to_num` would pile unrankable knots at rank 0 and read as
  catastrophic bias in D1). All four passed 39/39 before.
- **Measured, not asserted:** the streaming rewrite costs 2.2 MB peak RSS
  against 50.4 MB for the old whole-file path on a 151 MB window file;
  `window_sampling_std` takes 7.4 s at M=64 / S=20 / 2000 frames / 10 windows.
- **Confirmed clean in round 2** (listed so a later WP does not re-litigate):
  the `_BOUNDARY_TOLERANCE` direction and scale (still exact at `w = 1e9`; the
  reverse pull needs a cadence ~10 orders below any output frequency), the
  simplified `_replicate_spread`, the CRPS/z-score/rank math against §4.2, the
  `_flat_members` window-major flattening, the identifiability median axis, and
  all four cross-phase invariants.

**WP1.4**

- **The accumulation rides on the sensor pass, it does not get its own.** The
  plan says "extend the existing streaming pattern"; the implementation adds an
  `on_member` callback to `ensemble_sensor_series` and feeds the accumulators
  from the member slice that pass already materialises. A second pass would
  have doubled the read of files that total tens of GB at Barcelona scale for
  information the first pass already has in memory. The callback fires only
  when the state carries an `ensemble` axis (everything hanging off it is per
  member), and a failure inside it disables the mean-field layer rather than
  propagating — the sensor blocks share the loop and must not die with it.
- **The truth is accumulated directly on the assimilation grid**, by
  interpolating each time chunk onto the target region before folding it in,
  rather than accumulated on its own grid and interpolated afterwards. The
  alternative needs the truth's *reduced* fields on the target's z heights,
  which the truth grid does not carry — nearest-level selection plus horizontal
  interpolation was the fallback, and it silently mixes a half-cell vertical
  offset into a cell-against-cell comparison. Interpolating first costs one
  scipy call per chunk on a few slabs (a small fraction of the read it rides
  on) and makes the identical-grid case exact: linear interpolation at
  coincident points returns the samples themselves.
- **Region: evenly spaced z-slabs + full-depth station columns.** The plan
  names neither. Slabs because the hit rate scores cells and figure F1 draws
  planes, on the *same* levels `streaming_state_rmse` already reports so the
  two state blocks describe the same heights; columns because figure S1 wants
  profiles, and at a handful of cells they are free — which is what lets the
  quantiles the plan asks for live there and only there.
- **Station columns are the assimilation sensors' `(x, y)`.** The plan's figure
  S1 says "upstream / in-canyon / wake" without saying where those come from.
  A new config group for three hand-placed stations would be a knob per case
  that nothing else reads; the sensors are already placed where the flow is
  interesting, already in the config, and already what every other block is
  scored at. If a case wants named stations, the honest place is the obs config
  that defines the sensors.
- **No config knobs, and the two memory bounds are derived rather than set**
  (the WP1.3 precedent). Neither number is one a caller could usefully choose:
  both are "what fits". The horizontal stride bounds the *persistent*
  accumulators — `n_members × cells × 80 bytes` against 1 GB, a member's
  accumulator being 10 arrays per cell independent of the frame count — and is
  logged whenever it is not 1. The time sub-chunk bounds the *transient* of one
  accumulation step, which is a different and initially unbounded quantity (see
  the review-round-1 entry below).
- **`hit_rate` returns `{q, n_points}`, not a bare float**, and the block
  reports the pooled `q` over the three components *and* a per-component
  breakdown. One scalar is what the metrics doc asks for and what the pooled
  entry is; the breakdown costs three lines of YAML and is the difference
  between "the field is off" and "`v` is off". Each component is scored against
  its own `W`, which is why the pooled call takes a broadcast tolerance rather
  than a scalar.
- **`W` is a median over 64 sampled cells, not over every cell.** The
  bootstrap is per-cell and the reduction is a median (a cell inside a
  recirculation carries a wild floor and `W` is meant to describe a typical
  one), so the sampling error of the median over 64 draws is far below the
  spread between cells that the median is summarising. Keeping every cell's
  series through the pass would have cost the whole field in memory — the one
  thing the streaming design exists to avoid.
- **Non-finite `W` falls back to the relative criterion**, rather than
  disqualifying every point. The bootstrap needs ~21 frames and the CI smoke
  shape has fewer, so an unmeasured floor is the common case in tests; the
  relative test is the guideline's own, and the absolute one is the refinement.
  Logged at INFO when no component measured a floor.
- **`eval_fields.nc` stores reductions only, and NaN propagates through them.**
  A member whose field is not finite is not a sample from the posterior, so a
  `nanmean` across members would describe an ensemble that does not exist —
  the same convention WP1.2 settled on for the parameter calibration block, and
  deliberately the opposite of WP1.1's `ensemble_uniqueness`, which reduces over
  finite pairs because it is counting duplicates rather than estimating a
  moment.
- **The prior's mean fields reuse the prior sensor pass** (`_prior_sensor_series`
  gained the same callback), so the prior half costs no read of its own either.
  The all-or-nothing rule it inherits took two attempts to get right — see the
  review-round-1 entry below, which is where it became true of the *failed-read*
  path and not only of the missing-file one.
- **Dropped from the rollback branch's version** (`turbulence_stats.py`, 1339
  lines): `fac2`, `fractional_bias`, `nmse` and `nmse_split` — the slimmed
  metric set scores the mean field with the hit rate alone; the scalar/batch
  bootstrap pair, superseded by WP1.3's single vectorized `block_bootstrap_std`;
  and `resolve_metrics_settings` with its four Hydra knobs. (Two things dropped
  here came back in review round 2: `extrapolated_centre_dims` and the
  fluid-cell masking, both for reasons recorded there.) Kept and adapted: the Chan/Welford accumulator core (re-measured
  here — 7e-11 relative error against `longdouble` where the naive form loses
  4 %) and the colocation table.
- **Colocation's extrapolated edge is disclosed rather than fixed.** Every axis
  colocation moves has its *last* index filled by linear extrapolation from the
  two faces below it (weights 1.5 / −0.5) rather than interpolated between two,
  so those cells' second moments are inflated — ~20 % for a well-resolved field,
  up to ~5x for face-to-face white noise — and `evenly_spaced_levels` (now
  shared with `_vel_field_4z` rather than copied into it) always includes the
  last index. It is **not** only the vertical, as an earlier revision of this
  entry said: uDALES moves x, y and z, PALM moves x and y, so the artefact is in
  every slab and no level selection can avoid it. `eval_fields.nc` therefore
  carries an `extrapolated_edges` attribute (round 2) naming the affected axes,
  which is what a WP1.5 figure needs to exclude them from an aggregate.
- **Deferred:** the block scores the mean *velocity* only. TKE and `<u'w'>` are
  accumulated and written to `eval_fields.nc` — figure S1 plots the TKE;
  `<u'w'>` has no reader in the phase-1 figure set and is stored because the
  WP's charter names it and the anisotropy ratio it enables is what the metrics
  doc calls more discriminating than TKE alone — but they get no scalar score — the metrics doc names the hit rate as the single standards-based
  number for the mean field and gives no threshold for a second-moment one, and
  inventing one here would put a number in `run_summary.yaml` that nothing can
  be read against.
- **The filtering side is not wired.** WP1.1 and WP1.2 reached
  `compute_filtering_metrics.py` for free because both went through
  `parameter_metric_summary`, one function with one meaning. Nothing is shared
  here: the mean fields hang off the ESMDA window-state layout and its sensor
  pass, so the filtering stage would need its own driver rather than an import.
  Out of scope for this WP, and the library half (accumulator, colocation, hit
  rate) is ready for it.

**WP1.4, review round 1.** Two adversarial reviewers were launched; the
integration/scope one completed and the correctness one died partway (an API
session limit) with its mutation-testing sweep unfinished — so its findings are
**not** in this record and round 2 has to cover the correctness lens.

- **A partially-read prior was scored, against this plan's own claim.** The
  guard compared the prior's *shape* against the posterior's — and a per-member
  mean over three windows has exactly the same shape as one over ten. So a prior
  state file that existed but could not be read (a job killed mid-`to_netcdf`)
  left the windows already folded into the accumulators, and `hit_rate_prior`
  was computed over half the horizon it was compared against, with nothing in
  either artifact recording the discrepancy. Demonstrated by the reviewer on a
  corrupted two-window run. Fixed at the root rather than with another guard:
  the prior read moved out of `_sensor_statistics` up into `compute_metrics`, so
  **one** place owns the read, and a prior it could not read reaches neither
  block — the split ownership is how the bug happened. A frame-count check backs
  it up, since it also catches a prior that is readable but short, which the
  shape check cannot. Note what that check is *not*: it gates the field block
  only, so a readable prior on a shorter horizon still reaches
  `sensor_statistics`, where WP1.3's own per-window guard governs it (round 2
  corrected an earlier wording here that claimed one gate for both blocks).
  Regression tests: `test_field_metrics_drop_a_prior_written_over_a_shorter_horizon`
  (the frame check) and the unreadable-prior case (the hoist).
- **The blanket `except` around accumulation is narrowed to colocation only.**
  It had wrapped the whole `_add`, which re-introduced exactly what WP1.3 round 2
  removed: a typo in the reduction would have shipped a green run with the block
  silently missing. Invariant 3 is about absent inputs — colocation refusing a
  layout — not about absent correctness.
- **The memory budget was calibrated against the wrong term**, which mattered
  because that budget is the entire argument for shipping no knob. It bounded
  the *persistent* accumulators (80 bytes per cell, frame-count independent),
  while the actual peak was the *transient*: measured 512 MB for a 200-frame
  64³ member against 1.3 MB of persistent state. Two fixes. (1) The station
  columns no longer hand the whole 3-D field to `.interp` — each station is
  bracketed between the two cells around it first, which is a 16x saving on the
  dominant term (223 MB → 13.0 MB measured, pinned by
  `test_station_columns_match_a_whole_field_interpolation`). The bracketed and
  whole-field interpolations agree to ~1 ULP, not bit-for-bit — the test asserts
  `allclose`, and an earlier wording here overstated it. (2) Time is sub-chunked
  to a 256 MB transient bound, which is what the chunk-wise accumulator is
  *for*; round 2 found two shapes that escaped the first cut of it, fixed there.
- **`MeanFieldCollector` moved from `_esmda_common.py` into
  `compute_esmda_metrics.py`** and is annotated. `_esmda_common.py` carries a
  file-level mypy waiver as legacy untyped code; ~250 statement lines of new
  code had landed inside it, escaping the strict config, for a module whose
  reason to exist is *sharing* between the metric and figure stages — and the
  figure stage reads `eval_fields.nc`, not the collector. Only the `on_member`
  parameter stays behind. The `target` dict became a `NamedTuple` in the move:
  it is shared by aliasing across three collectors, so immutability is worth the
  twelve lines.
- **Station columns now cover the validation sensors too**, labelled by a
  `station_set` coordinate. The original argument (sensors are already placed
  where the flow is interesting) survives, but figure S1's profiles are worth
  most at the *held-out* columns — a profile drawn only where the assimilation
  was fitted is the least informative one available — and the columns are a
  handful of cells either way.
- **A test now pins `_STAGGERED_TO_CENTRE` against
  `ObservationOperator.dim_mapping`.** Invariant 5 forbids the import, so the
  table is a restatement, and nothing failed if the two drifted — silently, in a
  way that would have the sensor series and the mean fields read one component
  off different axes *in the same pass over the same slice*. Writing the test
  found that the table is a strict superset, not a copy: pypalm's postprocess
  already moves `w` from `zw_3d` onto `z`, so that pair is a no-op on a shipped
  state and correct on a raw one. The assertion pins the invariant that actually
  holds — every staggered axis the operator reads has a colocation pair, and
  every pair names an axis the operator agrees exists.
- **Dead weight removed:** the `neural_surrogate` staggering entry (the surrogate
  records its spin-up backend's name, `pylbm`, and `ObservationOperator` rejects
  the other spelling outright, so it was unreachable), the `add_member` alias for
  `add`, and `MomentAccumulator`'s unread `n_components`/`cell_shape`
  properties. `evenly_spaced_levels` is now genuinely shared with
  `_vel_field_4z`, which had kept its own copy — the drift the function was
  introduced to prevent was still open.
- **`eval_fields.nc` gained `long_name` on every variable and `t_start`/`t_end`
  attributes**, so figure F1 can annotate its averaging window without reopening
  `truth_access.yaml`, and a `station_set` coordinate so S1 can label its
  columns. The file's stated purpose is cross-WP consumption; being
  self-describing is cheap insurance for that.
- **Accepted as-is, with the documentation corrected instead of the code:** the
  truth is streamed a second time (it can only be sampled once the ensemble pass
  has fixed the target, and it is one member's worth of frames against the
  ensemble's M). The reviewer's alternative — reordering `truth_sensor_series`
  after the ensemble pass and giving it the same callback — is a real saving and
  a reasonable follow-up, but it moves the sensor extraction's control flow for
  a term that is 1/M of the read this WP already avoids.

**WP1.4, review round 2.** A fresh adversarial sweep over the round-1 fixes
(which were themselves unreviewed code) and over what round 1 missed. The
correctness reviewer died on a session limit twice, so this pass carried both
lenses.

- **The hit rate scored solid cells, and that is the WP's headline number.** The
  spec says "over fluid cells"; nothing masked. A solid cell holds ~0 in the
  truth *and* in every member — every backend fills obstacle interiors rather
  than marking them (pylbm near-zeros, pypalm replaces PALM's NaN with 0.0,
  uDALES junk) — so `|p−o| ≈ 0 ≤ W` and each one counts as a **hit**. The
  reviewer demonstrated a posterior 100 % wrong in every fluid cell reporting
  `q_u = 0.50` on a half-solid domain: in general `q = f_solid + (1−f_solid)·
  q_fluid`, so at a Barcelona built-up fraction of ~0.3 a fluid hit rate of 0.52
  reports as 0.66 and *crosses the acceptance threshold on a field that fails
  it*. The round-1 deviation claiming "the masking rule reduces to score the
  cells that are finite" was simply false, and this PR's own colocation
  docstring said so two files away.
  Fixed with a two-rule mask: the backend's `blanking` indicator when the state
  files carry one (read once from a window file, not through every member slice
  — the sensor pass hands out `ds[["u","v","w"]]` and the indicator is not in
  it), and otherwise the truth's own resolved TKE, since a cell held at a
  constant has *exactly* zero variance and a fluid cell in a turbulent flow does
  not. `solid_cell_source`, `n_fluid_cells` and `solid_fraction` record which
  rule ran; `none` means every cell was scored and says so. The fallback stands
  down when it would mask everything (a laminar or single-frame truth), because
  a rule that masks the whole domain is not separating anything.
- **`W` was sampled over solid cells too**, the same root cause pointing the
  other way: a cell inside a building holds a constant, so its bootstrap floor
  is ~0, and in a slab more than half solid the *median* collapses to zero and
  the absolute criterion silently switches off. The sample is now filtered by
  the fluid mask at reduction time — the mask is not known during the truth
  pass, but the retained cells' flat indices are, which is enough.
- **The narrowed `except` did not match its own docstring.** `_centre_dims`
  raises on a state with no vertical dim and sat *outside* the guard, so that
  layout took the whole metric stage down — parameter, health, state and sensor
  blocks with it. Moved inside, with the no-time-axis check. Zero sensors had
  the same shape of bug (`concat([])`, `np.concatenate([])`); both now
  short-circuit, since the slabs do not depend on the stations.
- **The 256 MB transient bound was not enforced on two shapes.** (1) Colocation
  ran *before* the sub-chunk loop and materialises every axis it moves, so a
  staggered backend copied the whole member window regardless — measured 429 MB
  against 129 MB native. Colocation now happens inside the loop, on each piece;
  the layout probe that has to run first is one frame. (2) The block was derived
  from the *target* cells, but a cross-grid `.interp` scales with the **source**
  grid — measured 766 MB for a 96³ truth against a 32³ assimilation grid, the
  shipped default direction. It is now sized on whichever grid the step touches.
- **Four of the six round-1 fixes had no test that would fail if reverted**, and
  the drift test pinned coverage rather than the mapping it was written for: of
  six mutated `_STAGGERED_TO_CENTRE` entries (a component co-located onto the
  *wrong axis*, which is exactly the drift that would have the sensor series and
  the mean fields read one component off two different axes) **five survived**.
  The assertions now pin the destination per spatial letter — each pair must land
  on the centre axis of the letter the operator reads that face on — and all six
  mutants are caught. The pairs no operator axis can vouch for (pypalm
  pre-interpolates `w` off `zw`, so `dim_mapping` has the same blind spot) are
  stated outright.
  New regression tests, each verified to fail with its fix reverted: the solid-cell
  mask (both rules), the `W` filter, the frame-count prior check, the narrowed
  `except`, the no-vertical-dim and no-time-axis guards, sub-chunk equivalence at
  one frame per step, and the horizontal stride — which no grid in the suite was
  large enough to reach, despite being the whole "no config knob" argument.
- **Reported but deliberately not changed.** The truth is still read a second
  time (it can only be sampled once the ensemble pass has fixed the grid; it is
  1/M of the read the WP already avoids). `<u'w'>` still has no phase-1 reader.
  `_ensemble_reduction` still runs twice over the posterior slab (once for the
  netCDF, once for the score) — one array, and collapsing it would couple the
  writer to the scorer.

**WP1.5**

- **P1 draws two different knot pairs, chosen by the truth.** The spec says
  "prior vs posterior" and the first implementation read both marginals at the
  final knot — which is wrong on the *shipped default*
  (`esmda.num_assimilation_windows: 2`), because `prior_params.nc` stacks
  **every window's** prior along `time` (`run_esmda.py:365-376`). The final
  column of the prior is therefore window *W−1*'s prior, i.e. window *W−2*'s
  posterior, and the panel showed a nearly-converged "prior" beside the
  posterior: no visible contraction, which is the one thing the figure exists
  to show. Measured on a synthetic 3-window run (per-window prior stds
  2.0/0.6/0.25): the fixed static branch spans `(1.955, 8.122)`, the old code
  `(4.748, 5.226)`.
  The fix cannot be "always use knot 0", because for a genuinely time-varying
  parameter knot 0 is a *different physical time* and the comparison would be
  between two different quantities. So the discriminator is the truth itself
  (`np.allclose` across its finite knots): a **static** truth takes the prior
  at knot 0 against the posterior at the final knot (the total contraction the
  run achieved); a **time-varying** truth keeps the same-knot pair (per-window
  contraction, the only thing available). Both marginals are labelled with the
  knot they came from in either branch, and the panel title names the branch.
  Two limits, deliberate: with `true_params=None` there is no discriminator, so
  the same-knot pair is drawn and labelled as such; and a time-varying truth
  that happens to be flat reads as static, where the comparison is still valid
  and only the word "window" in the label is loose.
- **P1 and `run_summary.yaml` describe different things on a static
  multi-window run, on purpose.** WP1.2's `contraction_ratio` is per-knot, so
  its `final` keeps the per-window meaning. The figure's subject is the run,
  the YAML's is the window. Both are correct; neither was changed to match the
  other, and `docs/scripts_and_configs.md` states the difference where a reader
  meets it.
- **`U_ref` is the truth's `velocity_magnitude` parameter**, and the canopy
  height is an optional `geometry.building_height`. The spec names `u/U_ref`
  and `z/H` as conventions but no source for either. `U_ref` falls back to
  plotting in m/s. No shipped case defines a building height (the geometry is
  an STL, not a scalar), so S1's `z/H` axis and its roof line are reachable by
  config but off by default — the key is read the same no-op-when-absent way as
  `esmda.obs_error_std`, and adding it to the case configs belongs with the
  phase-3 run upgrades, not here.
- **S5 draws a `±σ_o` envelope, not observation markers.** The spec says
  "observations with ±σ_o bars". Pre-WP2.1 the realized noisy observations are
  not persisted, and the run assimilates the truth at *every* frame in a window
  (`run_esmda.py:606-619`) — so a marker-per-observation rendering is not the
  right shape for this pipeline even once WP2.1 lands. The first implementation
  carried an unreachable `obs_times`/`errorbar` branch for it; the branch was
  deleted in round 1 and the caption now describes the ribbon it actually draws.
- **Two spec details were read differently.** S1's columns are ordered
  held-out-first rather than "upstream / in-canyon / wake" (the sensor sets
  carry no spatial role, and a profile drawn only where the assimilation was
  fitted is the least informative one available); D1's rows split by sensor set
  rather than pooling every set into one panel (the held-out ranks are the
  anti-overfitting evidence and pooling them with the assimilated ones would
  hide exactly that).
- **`style` gained `finite_limits` and `nested_bands`.** Two helpers with two
  and four call sites — short of the plan's "third caller" rule, but the plan's
  own charter for `style.py` names "quantile bands, shared norms" as its
  contents, and a shared norm that each figure computed for itself is the
  failure the module exists to prevent. F1's masked colormap stayed private to
  `figures.py`: one caller, no charter mandate.
- **`figures.py` imports three underscore-private names from `scores.py`**
  (`_aligned_parameter_members`, `_param_members_and_x`, `_plotted_param_names`);
  the WP0.2 deviation recorded two. Intra-library, so no invariant-5 issue.
- **F1 does not apply the `extrapolated_edges` exclusion, while S1 does.** The
  artefact inflates *second* moments; F1 plots a first moment, where it is
  small. Recorded in the docstring as a choice rather than left as an omission.
- **The new figures return `pathlib.Path | None`; the WP0.2 legacy plots return
  `None` unconditionally.** Two contracts in one module, explained at the WP1.5
  block header and pinned by tests. Unifying them means touching the moved code,
  which WP0.2 deliberately did not.

**WP1.5, review round 1.** Two adversarial reviewers, one per lens; the
correctness one had to be relaunched after dying on a connection error before
reading anything. The scope lens' headline: ~2.6k lines is earned (~700 of the
new `figures.py` lines are code, ~140 per multi-panel figure), no new config
groups, no new artifacts, no registries — the problems were a content bug, a
false claim, an untested layer and ~90 lines of dead code.

- **The blocker was P1's prior knot** (above). Found by the scope reviewer
  tracing `prior_params.nc` back to its writer, not by any test — the suite was
  green on a figure that misrepresented its headline quantity.
- **Six mutants survived all 53 tests.** The correctness reviewer mutation-tested
  the suite: P1 annotating `z_score[0]` instead of the drawn knot's, S1 reading
  the `v` component under a streamwise label, F1 computing its difference on
  *unmasked* fields, `_station_order` dropping held-out columns first, S5 drawing
  the truth on a frame index, and `_VERTICAL_CENTRE_DIMS` forgetting the uDALES
  `zt` spelling — each passed every test. Two named tests were vacuous:
  `test_station_profiles_keep_the_validation_columns_when_truncating` asserted
  only that a PNG appeared and a log fired, so the exact inversion it is named
  for passed; and the solid-cell test could not catch an unmasked difference,
  because both sentinels were `1e6` and their difference is 0. The `zt` mutant is
  the one that matters most in practice — PALM does not extrapolate the vertical
  at all, so `zt` is the spelling for the one backend where the trim ever fires,
  and it was untested. Each mutant now has a test verified to fail against it.
- **Two NaN paths degraded into plausible-looking lies.** S5 guarded with
  `.any()` and then called `np.quantile`, which *propagates* NaN — so one
  diverged member produced empty bands and an all-NaN median, rendered as a
  clean truth line with no fan and no log. F1's all-NaN no-op tested the *union*
  of its columns, while the case that occurs is a single all-NaN source (WP1.4's
  `_ensemble_reduction` propagates NaN by design); that column rendered entirely
  in the solid-cell grey, under a caption saying grey means solid — so it read as
  "the whole domain is a building". S5 now drops non-finite members, logs the
  count and annotates each panel `M = kept/total`; F1 checks finiteness per
  column and drops one with a log.
- **The wiring had zero test coverage**, repeating the lesson WP1.3's round 2
  recorded verbatim: every e2e ESMDA test runs `run.skip_viz=true`, so the figure
  stage never executed in CI and `_rank_counts`, `_reference_velocity` and
  `_note_skipped` were exercised by nothing. Closed with unit tests plus
  synthetic-run-dir tests that drive `make_figures` on the non-`skip_viz` path,
  on a complete run dir and on an old one.
- **Every no-op *reason* was invisible on a real run.** No script under
  `scripts/` calls `logging.basicConfig`, so the root logger has no handler and
  the library's `logger.info` lines were dropped — the operator saw
  `Skipped station_profiles.png` and never why. The tests passed only because
  `caplog` installs a handler. `main()` now configures logging (the entry point,
  not `make_figures`, so importing the module stays side-effect free).
- **S5 lost its window boundaries on static multi-window runs.** `window_edges`
  was gated on `is_dynamic and num_windows > 1`, which is right for the legacy
  parameter plots (a static parameter's x-axis is a window index) and wrong for
  S5, whose x-axis is physical time on every run — `ensemble_sensor_series`
  rebases window *w* onto `w·sim_time` regardless of dynamics, and
  `conf/run_esmda.yaml` documents a static-truth 3-window mode. S5 now gets its
  own dynamics-independent edge list.
- **The `mypy: ignore-errors` comment asserted something false.** It claimed
  deleting the waiver left only legacy errors; measured, `figures.py` had 24
  errors of which 6 were WP1.5's. Fixed to make the claim true: `figures.py` is
  down to 18, all above the WP1.5 block header, and `style.py` to 6, all legacy.
  Note the reviewer's *itemisation* was partly wrong even though its headline was
  right — `_param_axis_label` does not error and the five `save_png` return
  errors never existed, since `save_png` was already annotated. Measuring beat
  believing the list.
- **~90 lines of dead code deleted**, each confirmed callerless first:
  `plot_sensor_fans`'s `obs_times`/`errorbar` branch and `quantiles` parameter,
  `plot_mean_slices`'s `component` parameter (now the shared `_STREAMWISE`
  constant), `nested_bands`' `alphas`/`lw`/`zorder` arguments and its unused
  return, `_sensor_time_axis`'s length-mismatch fallback, and
  `_pool_rank_counts`' modal-length guard — the last only after failing to
  construct a case that reaches it (`rank_counts` is `bincount(minlength=M+1)`
  from one ensemble, so a cell's vectors cannot differ in length). S5's invented
  `np.linspace` truth axis went too: it assumed the truth was uniform over
  exactly the ensemble's span, and the reviewer demonstrated a 1.897 error on a
  unit-amplitude signal when it is not. A length mismatch now logs and skips the
  truth line rather than drawing a wrong one.
- **`z = n/a` covered three opposite diagnoses** — no truth saved, posterior
  collapsed, parameter pinned. Each now has its own string.
- **Smaller fixes:** P1's `parameter_bundle` call is wrapped in
  `catch_warnings` (an `inf` member printed a numpy `RuntimeWarning` to the
  operator's console for a value then displayed as `n/a`);
  `_station_profile`/`_slab_component` return `None` instead of silently falling
  back to `isel(component=0)`, which could draw `u` under a `w` label; a
  zero-length `zlev` no-ops instead of raising (invariant 3); S1's streamwise
  assumption (`u` is streamwise only because the shipped configs put the inflow
  along +x) is documented rather than knobbed.
- **Reported and deliberately not changed.** `compute_sensor_metrics` runs twice
  per sensor set — once inside `plot_sensor_timeseries`, once for S5's numpy
  payload — because the two take different argument types and the legacy
  signature is WP0.2's; measured at 0.06 s. The duplication is now stated in the
  comment that used to justify only the alignment.
