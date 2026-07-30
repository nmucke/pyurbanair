# Phase 1 — post-processing metrics and figures (no run-stage changes)

> Part of the ESMDA-evaluation effort. Master plan:
> [master_plan.md](master_plan.md). Rationale: §3, §4.1–4.2, §4.5, §7 of
> [docs/plans/esmda_turbulence_evaluation.md](../esmda_turbulence_evaluation.md).
> Requires phase 0. Four PRs (WP1.0+1.1, WP1.2, WP1.3, WP1.4).
>
> **Implementer: update the master_plan.md status table per WP as each PR
> lands; record deviations at the bottom of this file.**

Everything in this phase reads only the artifacts a run already writes
(`posterior_params.nc`, `prior_params.nc`, `true_params.nc`,
`posterior_state_mean.nc`, `windows/window_*_posterior_state.nc`, truth via
`truth_access.yaml`), so it works retroactively on existing run dirs.

Hard constraints, non-negotiable:

- `windows/window_*_posterior_state.nc` is ~1 GB at smoke scale and tens of
  GB at Barcelona scale — **never `.load()` it**. Stream member-at-a-time
  (`.isel(ensemble=m)`) or z-slab-wise, following the existing
  `_vel_field_4z` / `streaming_state_rmse` patterns
  (`scripts/esmda/_esmda_common.py:141–221`); ≤2 concurrent readers.
- `run_summary.yaml` keys are **additive only** — existing key paths are
  hard-coded in `scripts/figure_creation/{compare_state_runs,
  compare_sweep_results, make_figures_summary, visualize_run}.py`.
- Truth and ensemble can be on different grids and time cadences
  (cross-model default). Fields: average in time first, then `.interp` truth
  onto the assim grid. Sensors: interpolate both at shared physical points
  (existing machinery). Truth time is global; window files have local time
  axes rebased by `(t − t[0]) + w·sim_time` (see `ensemble_sensor_series`).

## WP1.0 Config block + module skeletons

`conf/run_esmda.yaml`, under the existing `run:` block:

```yaml
run:
  metrics:
    level: standard      # basic = pre-phase-1 set | standard | full
    n_z_slices: 4        # z-levels for the mean-field layer
    mean_field_stride: 1 # spatial stride for hit-rate/NMSE layer (full grid = 1)
    bootstrap_blocks: 20 # block count for sampling-error bars
    stations: null       # [[x, y], ...] for profile figures; null -> sensor x/y
```

Gating in `compute_esmda_metrics.py`: `level: basic` reproduces today's
behavior exactly; `standard` adds WP1.1–1.3; `full` reserved for phase 3
layers. The `skip_viz` early-return keeps its current meaning (parameter
metrics only) and additionally the phase-0/WP1.1 parameter extensions (they
are cheap and need no truth).

New modules (math lives here so figspec and the sweep pipeline can reuse it;
`_esmda_common.py` stays orchestration-only):

- `src/pyurbanair/utils/ensemble_scores.py` — pure numpy, ensemble axis
  first. Functions to provide (signatures indicative):
  `fair_crps(ens, truth)`, `crpss(post_score, prior_score)`,
  `fair_energy_score(members, truth)` (move/wrap `_energy_score`),
  `zscore(ens, truth)`, `pit_rank(ens, truth)` (ties randomized, seeded),
  `coverage(ens, truth, alpha)` using order-statistic band edges
  (members `ceil(q(M+1))`, not interpolated quantiles),
  `rank_histogram(ranks, n_members, n_bins=10)`,
  `spread_skill_ratio(variances, sq_errors, n_members)`.
  Fold the phase-0 fair estimators in here and re-export from `da_metrics.py`
  (keep old import paths working).
- `src/pyurbanair/utils/turbulence_stats.py` — see WP1.3.

Unit-test both modules against synthetic/closed-form cases (see Tests).

## WP1.1 Parameter bundle (metrics doc §3)

Inputs are already open in `compute_esmda_metrics.py:64–66`. Alignment rule:
truth is `np.interp`-aligned onto the posterior x-axis exactly as
`plotting.compute_parameter_metrics` does (~plotting.py:717) — reuse that
helper rather than re-implementing (static params: truth constant; knot
grids may differ, e.g. truth 11 vs posterior 12 knots).

Per parameter, per knot, then `series_stats`-style reduction:

- `zscore = (truth − mean_m)/std_m(ddof=1)` → report `{mean, std, max_abs}`
  pooled over knots; flag `overconfident: max_abs > 3`.
- PIT/rank pooled over knots (time-varying) and windows (static smoother
  stacks windows on the x-axis — see `_param_members_and_x`,
  plotting.py:107); emit 10-bin PIT counts. Effective-sample-size caveat,
  branch-dependent: for **dynamic** params (the only ones carrying
  `correlation_length`/`seconds_per_knot` — `conf/params/dynamic*.yaml`)
  emit `n_knots_effective = min(n_knots, ceil(n_knots · seconds_per_knot /
  correlation_length))`, clamped because sub-knot correlation lengths
  cannot create more independent samples than there are knots; for
  **static** params pooling is over windows, so emit
  `n_knots_effective = num_windows` together with
  `pooling: windows_correlated` (windows are linked by the cross-window GP
  carry-over, so this is an upper bound, not an independent count). Both
  branches: treat histogram shape as indicative in figure captions.
- Coverage at α = 0.5 and 0.9 (order-statistic edges).
- `contraction_ratio = std_post/std_prior` per knot → `{mean, min}`
  (prior file guaranteed present).
- Joint: flatten each member's parameters to a vector (same `(M, K)`
  flattening as WP0.3); prior and posterior correlation matrices; generalized
  eigenvalues of `(C_post, C_prior)` via `scipy.linalg.eigh(C_a, C_b)` on
  eps-regularized matrices (`C + eps·tr(C)/K·I`, reuse the eps-scaled rank-cut
  convention from `libs/data-assimilation/.../reduction.py`);
  `n_constrained_directions = #{λ < 0.5}` plus the top/bottom eigenvector
  loadings.

Schema (under the existing `parameter_metrics.<name>` mapping, additive):

```yaml
parameter_metrics:
  inflow_angle:
    zscore: {mean: ..., std: ..., max_abs: ..., overconfident: false}
    pit_counts: [...]
    coverage: {alpha_50: ..., alpha_90: ...}
    contraction_ratio: {mean: ..., min: ...}
  joint:
    n_constrained_directions: 2
    generalized_eigenvalues: [...]
    posterior_corr: [[...]]
    prior_corr: [[...]]
```

## WP1.2 Statistics-space sensor scoring (metrics doc §4.5, §4.2)

Reuses `truth_sensor_series` / `ensemble_sensor_series`
(`_esmda_common.py:276–332`). **First step: inspect their return structure**
(per sensor set: per-component arrays for truth `(time, sensor)` and ensemble
`(ensemble, time, sensor)`, plus the magnitude helper `sensor_magnitude`) and
write the reduction against what is actually there — do not trust this plan's
shape guesses over the code.

New orchestration helper in `_esmda_common.py`:

```python
def sensor_statistic_scores(truth_series, ensemble_series, window_edges,
                            n_members, bootstrap_blocks) -> dict
```

Per sensor set, per assimilation window, per sensor:

1. Slice both series to the window's time indices.
2. Reduce to statistics per member (and for truth): window mean of `u, v, w`,
   window variance of each, TKE `½Σvar`, mean `|U|`.
3. Score `{T_m}` vs `T*` with `ensemble_scores`: `fair_crps`, `zscore`,
   `pit_rank`, `coverage(0.9)`. Aggregate over sensors/windows into
   `series_stats`-style summaries; compute `crpss` posterior-vs-prior only
   where a prior series exists (prior state is usually not saved — guard and
   omit).
4. Identifiability guard: within-member sampling std of each statistic via
   block bootstrap (`run.metrics.bootstrap_blocks` blocks of the member's own
   window series) → `identifiability_ratio = across_member_std /
   median_within_member_std`. Ratios ≲ 3 mean the statistic is
   sampling-noise-dominated at this window length; still report, but flag.
5. Wasserstein layer (pooled samples, not statistics): per sensor,
   `W₁(pooled ensemble samples, truth samples)/σ_truth` via
   `scipy.stats.wasserstein_distance`, plus the **truth self-distance floor**:
   `W₁(first half of truth window, second half)/σ_truth`. Emit
   `w1_over_floor = w1 / max(floor, tiny)`. Also per-member-mean `W₁` (score
   each member's own samples vs truth, average) — the pooled and per-member
   numbers answer different questions (coverage vs typical-member fidelity);
   emit both.

Schema:

```yaml
sensor_statistics:
  assimilation:          # and `validation` when the case defines held-out sensors
    window_mean:  {crps: ..., zscore_max_abs: ..., coverage_90: ...,
                   identifiability_ratio: ...}
    window_variance: {...}
    tke: {...}
    wasserstein: {w1_over_sigma_pooled: ..., w1_over_sigma_member_mean: ...,
                  self_floor: ..., w1_over_floor: ...}
```

## WP1.3 Mean-field / Reynolds-stress layer (metrics doc §4.1)

New module `src/pyurbanair/utils/turbulence_stats.py`:

- `class StreamingMoments`: accumulates `n`, `Σu_i`, `Σu_i u_j` (i ≤ j)
  over time chunks; `.mean()`, `.reynolds_stress()` (`⟨u_iu_j⟩ − U_iU_j`),
  `.tke()`. Operates on numpy arrays of co-located components at the selected
  z-levels.
- `colocate_components(ds, solver_name) -> (u, v, w)` on cell centers:
  - `udales`: u `(zt,yt,xm)→xt`, v `(zt,ym,xt)→yt`, w `(zm,yt,xt)→zt` by
    linear interp of the staggered coordinate (see the dim mapping in
    `libs/data-assimilation/.../observation_operator.py:66–87`). **Do not**
    reuse the by-index `|U|` shortcut (`_vel_field_4z`) — fine for
    magnitude summaries, wrong for stresses.
  - `pylbm` / `neural_surrogate`: pass-through (uniform grid).
  - `palm`: w already on z; handle `xu`/`yv` for u/v analogous to udales.
- Mean-field scores, fluid cells only (`~np.isnan` after masking; solid
  cells are the NaN/blocked cells — verify how each backend marks them
  before assuming): `hit_rate(pred, obs, D=0.25, W)`,
  `fac2(pred, obs)` (positive quantities only — apply to `|U|`, never to
  signed components), `fractional_bias`, `nmse`, and
  `nmse_split(fb, nmse) -> (systematic = 4FB²/(4−FB²), unsystematic)`.
- `block_bootstrap_std(series, n_blocks)` for the sampling floor and the
  hit-rate `W` allowance.

Driver in `compute_esmda_metrics.py` (new function, `level >= standard`):

1. Select `run.metrics.n_z_slices` evenly-spaced z-levels (reuse the
   `_vel_field_4z` selection convention so the layer matches the existing
   RMSE slices), **plus full-z station columns** at the
   `run.metrics.stations` (x, y) locations (default: sensor positions).
   A handful of vertical columns is negligible extra cost, and it is the
   only way S1 gets true vertical profiles — a 4-slice sampling is not a
   profile.
2. **Truth pass:** stream truth time chunks (its own grid), colocate,
   accumulate `StreamingMoments` for the z-slabs *and* the station columns;
   also collect per-cell time series at the selected levels for
   `W = σ_u/√N_eff` via block bootstrap.
3. **Ensemble pass — one shared read per member (production-scale
   requirement, not an optimization):** at Barcelona scale the window files
   total tens of GB and this stage is DRAM/IO-bound, so a second full read
   pass is not acceptable. Add
   `_esmda_common.stream_window_members(state_paths)` — a generator that
   opens each member once (`.isel(ensemble=m)`, ≤2 reader threads), applies
   `colocate_components`, and feeds *all* consumers in one pass: the WP1.2
   sensor-series extraction (refactor `ensemble_sensor_series` to accept an
   already-open member Dataset) and the slab + station-column
   `StreamingMoments` accumulators (per-member accumulators are small: a
   few z-slabs and columns × components). Net effect: window files are read
   once per member total — approximately today's sensor-pass I/O, so the
   incremental cost of this WP is arithmetic, not reads. `level: standard`
   **is the intended default for production runs**; `n_z_slices` /
   `mean_field_stride` are the relief valves if profiling ever shows
   otherwise. Accumulate across windows so statistics cover the full run,
   and record the window-count so truth uses the identical time range.
   Adding a second full read pass over the window files is a
   review-blocker.
4. Reduce: per-member time-mean fields → ensemble mean and std maps
   (feeds WP1.4 F1/F2); truth time-mean interped onto the assim grid
   (after averaging); score ensemble-mean-of-means vs truth with
   `q, fac2(|U|), fb, nmse ± split` per component and per z-level; TKE and
   `⟨u′w′⟩` RMSE with the bootstrap sampling floor attached.

Schema:

```yaml
mean_field_metrics:
  hit_rate: {u: ..., v: ..., w: ..., per_z: [...]}
  fac2_velmag: ...
  fb: {u: ..., v: ..., w: ...}
  nmse: {u: {total: ..., systematic: ..., unsystematic: ...}, ...}
  tke_rmse: {value: ..., sampling_floor: ...}
  uw_stress_rmse: {value: ..., sampling_floor: ...}
  averaging: {n_time: ..., n_windows: ..., z_levels: [...]}
```

Persist the reduced fields the figures need (small: `n_z_slices` × 2-D maps
per member-aggregate, **plus the full-z station columns** — truth and
per-member time-mean and second moments at each station) to
`run_dir/eval_fields.nc` (truth mean, ensemble mean,
ensemble std, per-quantity) so stage 3 does not recompute the streaming pass —
note stage 3 currently re-derives inputs from raw artifacts; this file is the
sanctioned handoff. Document it in `docs/data_assimilation.md`.

## WP1.4 Figures, first wave (metrics doc §7)

New functions in `src/pyurbanair/plotting.py` (split out a `plotting_eval.py`
if it passes ~300 new lines), one call each from
`make_esmda_figures.py::make_figures`, gated on `run.metrics.level` and on
input availability. Reuse `scripts/figspec/style.py` (`COLORS`,
`CMAP_FIELD/DIFF/STD`, window shading helpers) — conventions are already
encoded there and in `docs/archive/figure_specs.md`.

| ID | Function | Inputs | Key requirements |
|---|---|---|---|
| P1 | `plot_parameter_marginals` | param datasets | box+strip per parameter (violin/KDE invents structure at M=50), truth dashed line, y-lims include the prior, annotate z-score + % reduction |
| S1 | `plot_station_profiles` | `eval_fields.nc` station columns (full-z) | rows = quantity (`ū`, `w̄`, `√u′²`, `⟨u′w′⟩` or TKE), cols = stations; quantity on x, z on y; truth line + posterior band; station-location inset; shared x-lims per row |
| S5 | extend `plot_sensor_timeseries` | existing series | quantile fans (5–95 + 25–75) instead of member spaghetti, window boundaries, assimilated vs validation column groups labeled; obs error bars arrive with phase 2 |
| F1 | `plot_mean_field_comparison` | `eval_fields.nc` | truth \| posterior mean \| difference per plane; one shared `Normalize` + colorbar for the first two, symmetric diverging norm for the diff; buildings grey via `set_bad`; annotate averaging window; **time-averaged only** |
| F2 | `plot_spread_maps` | `eval_fields.nc` | posterior std maps with sensor positions overlaid; prior column only when `window_*_prior_state.nc` exists |
| S2/S3 | `plot_probe_distributions` | WP1.2 pooled samples | PDFs on log-y with shared bin edges + Q–Q (equal aspect, 1:1 line); captions state "quantiles, not instantaneous values" |

## Tests

- Unit (`tests/test_ensemble_scores.py`, `tests/test_turbulence_stats.py`):
  `StreamingMoments` vs direct `np.mean`/`np.cov` on random data; colocation
  on a tiny synthetic staggered grid (assert center values); hit rate /
  FAC2 / FB / NMSE on constructed fields with known answers; NMSE split
  identity `NMSE_s(FB) ≤ NMSE`; Wasserstein vs
  `scipy.stats.wasserstein_distance`; order-statistic coverage on a
  calibrated synthetic ensemble ≈ α.
- Integration (extend the existing ESMDA pipeline test via
  `compose_test_cfg`): at `level: standard` on the smoke case, assert the new
  `run_summary.yaml` keys exist and are finite, `eval_fields.nc` exists, and
  the new figures render (file-exists assertions). One test at
  `level: basic` asserting the summary is key-identical to pre-phase-1
  (minus phase-0 additions).
- Runtime budget: stage 2 at `level: standard` on the smoke case ≤ ~2× the
  `basic` runtime; if the mean-field pass busts it, drop smoke
  `n_z_slices` to 2 via `_SMOKE_OVERRIDES`.

## Acceptance

- All WP schemas present on a smoke run; figures render with shared
  norms/limits verified by review; no `.load()` of window state files
  anywhere (grep); old run dirs still process (`level: basic`); docs updated
  (`docs/data_assimilation.md` artifact table gains `eval_fields.nc`).

## Deviations

- **WP1.0 — `--metrics-level` CLI flag (addition).** `compute_esmda_metrics.py`
  gained a `--metrics-level basic|standard|full` argparse flag that overrides
  `run.metrics.level` from the run dir's saved config for one invocation. Not in
  the plan text: without it, re-processing an existing run dir at another depth
  means editing that dir's saved `config.yaml`, which is the record of how the
  run was executed. The resolution order is CLI override → saved config →
  shipped defaults (`resolve_metrics_settings`), and configs predating the
  `run.metrics` block resolve every key to the shipped default (`standard`),
  since phases 0–1 are meant to apply retroactively.

- **WP1.1 — dynamic/static is per *parameter*, and the discriminator is the
  `time` *coordinate* (correction to the plan text).** The plan describes
  dynamic params as "the only ones carrying `correlation_length` /
  `seconds_per_knot` — `conf/params/dynamic*.yaml`", which reads as a run-level
  split. It is not: `conf/params/dynamic.yaml` mounts time-varying
  `external_parameters` *and* a `static_parameters` block
  (`vertical_inflow_exponent`) into the same Dataset, so both PIT branches fire
  in one run. The implemented test is `_param_members_and_x`'s own
  (`dims == ["time"]` **and** a `time` coordinate exists), not `"time" in dims`:
  `run_esmda._concat_windows` stacks per-window parameter files along `time`, so
  in a purely static run every parameter comes out with a length-`num_windows`
  `time` *dimension* and no `time` coordinate, and its x-axis is the window
  index. Keying on the dimension alone would misroute that run into the GP
  branch and apply a `seconds_per_knot` formula to a window axis.

- **WP1.1 — `n_knots_effective` is additionally clamped by the number of
  piecewise-constant segments (addition).** A static parameter in a *dynamic*
  run is broadcast by `_concat_windows` to every knot, so it reaches the
  time-coordinate branch with `n_knots` knots but only `num_windows` distinct
  values. `min(n_knots, n_segments, ceil(n_knots·seconds_per_knot/L))` recovers
  the window count without needing to know which config block the parameter came
  from, and is a no-op for a genuinely time-varying parameter.

- **WP1.1 — the joint eigenproblem is rank-truncated before
  `scipy.linalg.eigh` (departure).** The plan prescribes
  `eigh(C_post, C_prior)` on eps-regularized matrices. Applied literally to real
  run data this is numerically meaningless: ensembles are routinely *smaller*
  than the joint parameter vector (M = 32 against K = 42 on a routine
  2-parameter/21-knot run), so both covariances have rank M−1 = 31 with
  *different* null spaces. Measured on `.temp/pyudales_to_pyudales`, the literal
  recipe returns a **negative** eigenvalue from a positive-definite pair and a
  spread of 1e−6 … 1e+11 (cond(B) = 2.6e15). The implementation therefore
  projects both covariances onto the prior's retained eigenbasis first — rank cut
  `λ_max · finfo.eps · max(shape)`, the `numpy.linalg.matrix_rank` convention
  `data_assimilation.reduction` also uses — and runs the prescribed eps-ridged
  `eigh` inside that r = min(M−1, K) subspace, where it is well-posed (same run:
  r = 31, cond = 2.9e6, all λ > 0, spread 0 … 124). `n_sample_directions` and
  `rank_deficient` are emitted so a reader can see the truncation happened.

- **WP1.1 — generalized eigenvalues come from *covariances*, the reported
  matrices are *correlations* (plan ambiguity resolved).** The plan lists both in
  one bullet. Only the covariance pencil makes `λ < 0.5` mean "this direction's
  spread at least halved" — correlation matrices both have trace K, which
  destroys the variance-contraction reading. The λ are invariant under any
  invertible rescaling of the parameter vector, so the mixed units (degrees vs
  m/s) of the joint vector are harmless.

- **WP1.1 — full correlation matrices are capped at K ≤ 8
  (`JOINT_CORR_MAX_K`).** `write_yaml` uses `default_flow_style=False`, i.e. one
  number per line, so the K = 42 of a routine run would add ~3.5k lines to a
  ~100-line `run_summary.yaml`, and production cases are larger. Above the cap
  `posterior_corr` / `prior_corr` are `null`, a `corr_matrices_omitted` note says
  why, and `corr_summary` (off-diagonal `|corr|` mean and max, both matrices)
  carries the signal instead. `generalized_eigenvalues` is capped the same way at
  r ≤ 64, with `eigenvalue_quantiles` always emitted.
  *(Revised in review from K ≤ 16.* Measured YAML cost of the two matrices:
  K = 6 → 74 lines, K = 8 → 130, K = 16 → **514**, K = 42 → 3530. The cap's own
  justification is a ~100-line summary, which K = 16 misses by ~5×. K = 8 keeps
  the matrices for the small joint vectors where a human can actually read them
  and still roughly doubles the file; a sidecar was considered and rejected —
  nothing consumes the full matrices yet, and when something does, the sanctioned
  handoff is WP1.3's `eval_fields.nc`, not a second new artifact.)*

- **WP1.1 — M < 3 nulls the whole calibration bundle.** Per the master plan's
  degenerate-shape rule (`null` + a log line, never special-cased math), the
  2-member smoke shape emits every key with a `null` value rather than a number:
  a ddof=1 spread has one degree of freedom, the widest order-statistic band is
  `[x_(1), x_(2)]` (nominal 1/3, so a "90%" coverage is unattainable), a 10-bin
  PIT has 3 possible ranks, and every sample correlation is exactly ±1. The
  threshold is one constant (`MIN_MEMBERS_CALIBRATION` / `MIN_MEMBERS_JOINT`)
  rather than a per-metric rule.

- **WP1.1 — keys emitted beyond the plan's schema sketch (additive).** The
  sketch shows `zscore`, `pit_counts`, `coverage`, `contraction_ratio` and a
  small `joint`. Also emitted: a sibling `pit` mapping holding the PIT metadata
  the plan asks for in prose (`n_bins`, `n_samples`, `n_knots_effective`,
  `pooling`, `tie_seed` — the seed so the tie-broken counts are reproducible);
  `coverage.max_nominal_alpha = (M−1)/(M+1)`, the highest nominal level an
  order-statistic band can offer at this M, so a clamped `alpha_90` does not read
  as a calibration failure (it bounds the *nominal* level, not the realized
  fraction); and in `joint`: `n_members`, `n_parameters`, `n_sample_directions`,
  `rank_deficient`, `eigenvalue_quantiles`, `most_constrained` /
  `least_constrained` (eigenvalue + the 5 largest-magnitude parameter-space
  loadings, named via `parameter_vector_labels`, instead of raw length-K
  vectors), `corr_summary`, and a `reason` string on every degraded path.

- **WP1.1 — the `(M, K)` flattening is passed in, not recomputed.**
  `parameter_bundle_summary` takes `posterior_flat` / `prior_flat` as arguments
  so the pipeline keeps exactly one flattener
  (`compute_esmda_metrics._flatten_parameter_members`, shared with
  `_ensemble_health`) rather than a second copy in `_esmda_common`. The label
  helper `parameter_vector_labels` mirrors that flattening's ordering and is
  pinned to it by a test.

- **WP1.0/1.1 (review round 1) — `metrics_level` is written to the summary
  (addition).** `compute_metrics` logged the resolved level but persisted
  nothing, so a consumer seeing no `parameter_metrics.joint` could not tell a
  pre-phase-1 run dir from one processed at `basic` from a layer that no-op'd on
  missing inputs — and `--metrics-level` makes mixed-depth reprocessing of one
  sweep easy, which is exactly the version-mixing hazard the master plan's
  invariant #1 exists to prevent. `summary["metrics_level"]` is emitted at every
  level, next to `metrics_version`; **no `metrics_version` bump**, since the
  estimator semantics are unchanged — it records how much of the suite ran, not
  how it was computed. Documented in `docs/scripts_and_configs.md`.

- **WP1.0 (review round 1) — the numeric `run.metrics` knobs are validated
  (addition).** `resolve_metrics_settings` rejected an unknown `level` but
  accepted `n_z_slices: 0`, `mean_field_stride: 0`, `bootstrap_blocks: -5`; the
  config comments stated the constraints and nothing enforced them, so a typo
  would surface as a crash (or an empty result) deep inside a WP1.3 streaming
  pass that had already read GBs. All three are now `>= 1`-checked beside the
  level check, same failure mode, same place.

- **WP1.1 (review round 1) — `pit.ranks_per_bin` (addition).** `rank_histogram`
  maps `bin = rank * n_bins // (M + 1)`, which is only uniform when `n_bins`
  divides `M + 1`. At the production M = 32 the 33 rank values split
  `[4, 3, 3, 4, 3, 3, 4, 3, 3, 3]`, so a *perfectly calibrated* ensemble shows a
  fixed three-bin comb of +21% / −9% against a flat `len(ranks)/n_bins`
  reference; at M < `n_bins` some bins are unreachable entirely. WP1.4 plots
  these counts, so the reference has to travel with them: the new
  `ensemble_scores.rank_histogram_weights` computes the per-bin rank-value
  counts and `_pit_block` emits them as `pit.ranks_per_bin`. Purely additive —
  the binning itself is unchanged, and the M = 19 (exactly-divisible) unit test
  is kept alongside the new M = 32 one.

- **WP1.1 (review round 1) — `joint.posterior_variance_retained` (addition).**
  The rank truncation onto the *prior's* eigenbasis (the departure logged above,
  endorsed in review and unchanged) is lossless only when the posterior lives in
  the prior's span. That holds within one ESMDA update, but `posterior_params.nc`
  is a multi-window **concatenation** and each window block carries its own M × M
  transform, so the joint row space is not contained in the prior's. Measured on
  a 3-window / M = 32 / K = 42 construction: ~98.5% of posterior variance
  retained (1.00 for the single-window case), dropping to ~80% when the posterior
  grows a direction the prior never had. `rank_deficient` cannot see any of this.
  The ratio `tr(Qᵀ C_post Q) / tr(C_post)` is therefore emitted next to it, so
  `n_constrained_directions` can be read as covering a stated fraction of the
  posterior spread. Computed pre-ridge; `null` on every degraded path.

- **WP1.1 (review round 1) — the WP1.1 integration criterion is met on a
  synthetic run dir, not the smoke case.** The acceptance text asks for "new keys
  present and finite at `level: standard` on the smoke case". That is
  structurally unmeetable: the smoke shape is a 2-member ensemble, so every
  numeric key takes the `MIN_MEMBERS_CALIBRATION` null path and the bundle's math
  never runs — the pipeline test would assert only that nulls appear. The
  intent (does the *wiring* work end to end?) is met instead by
  `tests/test_esmda_metrics_wiring.py`, which drives `compute_metrics` on a
  synthetic run dir at M = 8 with a multi-window parameter artifact. It is cheap
  and needs no solver because the WP1.1 block sits before the `skip_viz` early
  return, and it covers precisely what unit tests could not: that
  `ta.get("num_windows")` is spelled the way the run stage writes it (a typo
  would silently null `n_knots_effective` for every static parameter), that
  `_flatten_parameter_members` agrees in shape between prior and posterior on
  concatenated artifacts, and the emitted key **set** — not merely finiteness —
  at both `basic` and `standard`.

- **WP1.1 (review round 1) — `_aligned_parameter_arrays` re-implements the
  alignment rather than calling `compute_parameter_metrics` (departure).** The
  plan says to reuse that helper. It returns *scores* and never exposes the
  aligned members the bundle needs, so the two `np.interp` lines are repeated
  instead. To keep that from growing a private-import surface, the two pieces
  actually shared — `plotting.param_members_and_x` and
  `plotting.plotted_param_names` — were promoted to public API (old `_`-prefixed
  names kept as aliases). They define a parameter's x-axis, its dynamic/static
  discriminator and the parameter iteration order; figures and metrics disagreeing
  about any of those would be a silent error, so one public definition is the
  point.

- **WP1.1 (review round 1) — `_n_constant_segments` tolerance is scaled to the
  parameter's own spread.** It used `np.isclose` defaults, i.e. `rtol = 1e-5`
  *relative to the magnitude*: 2.7e-3° against an `inflow_angle` near 270 but
  1e-7 against an `sgs_constant` near 0.01 — a threshold set by where the
  parameter's origin happens to be, which could clamp a genuinely time-varying
  parameter's effective sample size. The step test is now
  `|Δ| > 1e-6 · std(members)`. The two cases it must separate are decades apart
  (a broadcast static parameter repeats bitwise-identical values, Δ = 0 exactly;
  a time-varying one steps by O(spread)), so the fraction is not delicate.

- **WP1.1 — per-key schema documentation deferred.** `run_summary.yaml`'s key
  list lives in `docs/scripts_and_configs.md` (the
  `compute_esmda_metrics.py` section), which is outside this WP's file scope;
  WP1.0 already documented there that levels above `basic` "add keys on top
  (never change existing ones)", so that file is not left incorrect. The exact
  WP1.1 key set is the list in the two entries above; fold it into
  `docs/scripts_and_configs.md` together with WP1.2/1.3's schemas.
  *(Done in review round 2 — the WP1.1 key set is now written out there.)*

### Review round 2

The round-1 entries above stand; these five record what round 2 changed.
Common thread, and worth stating once because it is the same bug three times:
**every calibration diagnostic in this bundle was being compared against a
large-sample reference that does not hold at production `M`**, and each of
those errors pointed the same way — toward reporting a perfectly calibrated
ensemble as broken. `metrics_version` stays 2 throughout: no estimator
changed, only what each number is reported next to.

- **WP1.0 (review round 2) — `ensemble_scores` gained two reference functions
  (addition; logged here because that module has no plan doc of its own).**
  `zscore_exceedance` / `zscore_nominal_exceedance` / `max_abs_zscore_reference`
  and `coverage_nominal_alpha` / `max_nominal_alpha`. The first family exists
  because **the z-score null is not standard normal**: `zscore` estimates both
  the mean and the spread from the same `M` members, so under exchangeability
  the null is `sqrt((M+1)/M) · t(M−1)` — the `t` from the `ddof=1` denominator,
  the Fortin factor from `truth − mean_m` having variance `σ²(1 + 1/M)`.
  Measured over 4e5 calibrated draws at `M = 32, c = 3`: empirical 0.00579,
  scaled `t` 0.00594, plain `t` 0.00529, normal 0.00270 — so dropping the
  Fortin scale is wrong by ~10% at `M = 32` and by 22% at `M = 4`, and reading
  a normal table is wrong by a factor 2.2. The second family exists because an
  order-statistic band cannot hit an arbitrary `alpha` (see the coverage entry
  below). Both are *functions* rather than inline arithmetic so the metrics and
  figure layers read one definition instead of re-deriving it.

- **WP1.1 (review round 2) — `zscore.overconfident` no longer flags sample
  size; `zscore.exceedance` and `zscore.max_abs_calibrated_median` added.**
  The flag was `max |z| > 3`. The maximum of `n` draws is an order statistic
  that grows with `n`, so this measured **how many knots were pooled**, not
  calibration: on a perfectly calibrated `M = 32` ensemble it fires 11.8% /
  31.6% / 45.1% / 85.2% of the time at 21 / 63 / 105 / 315 pooled knots
  (4000 trials each), and 315 is the routine 2-parameter/21-knot/3-window
  shape — i.e. the flag was very nearly a constant `true` on real runs.
  It is now `exceedance.observed[0] > 2 × exceedance.nominal[0]` (the `|z| > 2`
  cut), whose false-positive rate on the same ensembles is 11.4% / 2.9% / 0.7%
  / 0.0% — **decreasing** with sample size, which is the property that was
  missing. `|z| > 3` was rejected as the cut: at these `n` its expected count is
  O(1), so the rule reduces to "did any single knot exceed 3" and the rate
  stays flat at ~12%. The ×2 multiplier rather than ×3 is a measured power
  trade at `M = 32` / 315 knots against an ensemble whose spread is a factor
  `s` too small — ×2 detects `s = 0.8` 65% of the time and `s = 0.7` 99.8%,
  ×3 detects `s = 0.8` 0.3% of the time, i.e. not at all.
  Kept, per the "reference travels with the number" rule that `ranks_per_bin`
  set: `mean` / `std` / `max_abs` are unchanged, `exceedance` carries the
  observed tail fractions **with their nominal levels** (this is what WP1.4
  plots; it must not re-derive them), `max_abs_calibrated_median` says where
  `max |z|` sits for a *calibrated* ensemble of this size over this many knots
  so the retained `max_abs` is readable at all, and `overconfident_rule` names
  the exact keys the boolean is computed from so the verdict is reproducible
  from the summary alone. The boolean is deliberately computed *here* and not
  in `ensemble_scores`: the shared layer emits no verdict because deciding
  "is this miscalibrated" needs an effective sample size only this layer has,
  and even here it is a screen, not a test — correlated knots degrade it toward
  its ~11% small-`n` behaviour, which is why `sampling` sits next to it.

- **WP1.1 (review round 2) — coverage is scored against the *realizable* level:
  `coverage.nominal_alpha_50` / `nominal_alpha_90` (addition).** Band edges are
  member order statistics, so the attainable nominal levels are the `M + 1`
  multiples of `1/(M+1)` and the requested `alpha` is rounded to one of them. At
  `M = 32`, `alpha = 0.5` is the band `[x_(9), x_(25)]` with nominal level
  0.4848 and measured empirical coverage 0.4841 (2e5 samples) — so a consumer
  holding that against the requested 0.5 reads ~13 sampling sigma of **pure
  discretization** as miscalibration. The realized level is now emitted beside
  every coverage number. `max_nominal_alpha` answers the different question it
  always did (the widest band this `M` offers) but is now the imported
  `ensemble_scores.max_nominal_alpha` rather than an inline `(M−1)/(M+1)`, so
  the two references cannot drift; `_band_indices` stays private and these two
  functions remain its only public consequence.

- **WP1.1 (review round 2) — contraction and the joint pencil now report
  per-window *and* cumulative contraction (the substantive finding).**
  `run_esmda.py:772-786` sets each window's prior to the previous window's
  posterior — GP-extrapolated when the parameter is dynamic, `prior_params =
  posterior_params` bitwise when it is static. So in the concatenated
  `prior_params.nc`, **block `w` is posterior `w−1` and only block 0 is a
  genuine prior**, and an elementwise `std_post/std_prior` measures what the
  *last* update did while its name and this plan's schema read as "how much did
  assimilation shrink the uncertainty". Measured on a 3-window / `M = 32`
  construction with a true per-window ratio of 0.6: the reported number is
  `{mean: 0.600, min: 0.600}`, the cumulative truth `{mean: 0.392, min: 0.216}`
  — a run that cut spread by 78% reporting 40%.
  Both are now emitted. `contraction_ratio.vs_window_prior` is the existing
  number, unchanged; `contraction_ratio.vs_initial_prior` is the same posterior
  against window 0's prior block, tiled across the windows, with a `reason` that
  is non-null exactly when its numbers are. `mean` / `min` stay at the top level
  as aliases of `vs_window_prior`, assigned from the same mapping so they cannot
  drift — the schema sketch above indexes them directly. One slice expression
  covers both artifact layouts (a dynamic parameter's `n_knots =
  num_windows · knots_per_window`; a static parameter's x-axis *is* the window
  index, so `knots_per_window` comes out 1 and the tile degenerates to "column
  0, repeated"), so there is no branch on the parameter's kind; a knot count
  that does not divide by `num_windows` emits `null` plus a log line rather than
  mis-slicing.
  **`joint` gets the same treatment rather than the ambiguity it was left with.**
  The shift applies to the generalized pencil too — `prior_flat` spans every
  window, so `n_constrained_directions` counts directions the *per-window*
  updates halved. That is now stated in `joint.prior_reference`, and the
  cumulative question is answered by `joint.vs_initial_prior`: the same
  `joint_parameter_directions` on the **final posterior window block** against
  the **window-0 prior block**, reduced to a scalar summary (no loadings, no
  matrices — ~12 YAML lines). It was worth doing rather than documenting away
  because it is free: the blocks are one window wide, so at the routine `M = 32`
  / `K = 42` / 3-window shape the pencil is 14-dimensional against 31 sample
  directions — **full rank**, where the parent block is rank-truncated. Its
  soundness rests on the knot prior being the stationary GP (window 0's prior
  covariance is the prior covariance for any window's knots), which is the same
  assumption the run stage already makes when it extrapolates. Both per-window
  blocks are the pipeline's one flattener applied to a `time`-sliced Dataset
  (`compute_esmda_metrics._window_block_flat`), not a second flattener.
  The regression is pinned in *both* test layers, because the old fixture could
  not have caught it: `tests/test_esmda_metrics_wiring.py::_dynamic_run_artifacts`
  built a prior uniformly wider than the posterior at every knot — the one shape
  a real multi-window run never produces, and the shape that makes the two
  ratios coincide, so its `assert 0.0 < contraction_ratio["mean"] < 1.0` passed
  whichever number was reported. It now builds the real chain.

- **WP1.1 (review round 2) — minors: `crpss` wiring, `sampling`, and two
  corrected claims.** (a) `parameter_metric_summary` computed
  `rmse_reduction_vs_prior` / `crps_reduction_vs_prior` by hand, duplicating
  `ensemble_scores.crpss`; now wired to it. Verified bit-identical over 2e5
  random positive `(post, prior)` pairs. Only the non-finite corners move, each
  from a wrong number to a `null`, because the old guard tested the denominator
  alone: `post = nan` gave a bare `nan`, `post = inf` gave `-inf`, and
  `prior = inf` reported a **100% reduction**. (b) The effective-sample-size
  caveat was attached only to `pit`, which reads as though PIT alone needed it —
  `zscore.exceedance` and `coverage.alpha_*` pool over the identical correlated
  knots. It is hoisted to a per-parameter `sampling` block (`n_samples`,
  `n_knots_effective`, `pooling`) and mirrored under `pit` from the same
  computation for schema stability. (c) `_flatten_parameter_members` spans every
  var with an `ensemble` dim while the per-parameter bundle iterates the fixed
  `_PLOTTED_PARAMS` tuple; they coincide today, but a future estimated parameter
  missing from that tuple would enter `joint` while silently getting no
  calibration entry, so such vars are now named in a `logger.info`. (d)
  `_energy_score`'s docstring claimed the shared implementation "keeps the same
  memory bound". Half true, and not in the way previously written: the
  **pairwise** term is unchanged (one `(E, E, S, C)` slab, same element count as
  the old `(C, E, E, S)`), but the distance-to-truth term is now taken over the
  whole `(E, T, S, C)` batch where the old code held `(C, E, S)` inside a time
  loop — a factor `n_time`, not `n_components`, on that term. Corrected, with
  the peak stated as `max(E·T·S·C, E²·S·C)` so a future field-shaped caller can
  see it must chunk.

### WP1.2

- **WP1.2 — `block_bootstrap_std` landed in `turbulence_stats.py` under WP1.2,
  not WP1.3.** The module was a skeleton whose docstring already advertised that
  exact name, and WP1.2's identifiability guard (step 4) is the first consumer:
  the within-member sampling std of a window statistic has to come from a
  *moving-block* bootstrap, because a window's frames are serially correlated and
  the iid `std/sqrt(n)` understates it — which would inflate the identifiability
  ratio in the flattering direction, i.e. exactly the direction the guard exists
  to catch. Only that one function shipped (plus `__all__`); the rest of the
  module stays WP1.3's. It returns `nan` rather than raising when `n < 4` or the
  block length collapses to `L < 2`, which **fires at smoke scale** (20 requested
  blocks against 3 frames per window) and is the reason `identifiability` has its
  own null path independent of the block around it.

- **WP1.2 — `sensor_statistic_scores` takes `num_windows` / `n_per_window` /
  `sim_time`, not the plan's `window_edges` / `n_members` (departure).** The two
  series are windowed by *different rules* and neither is recoverable from a
  single edge list: `ensemble_sensor_series` rebases window `w` onto
  `[w·sim_time, (w+1)·sim_time)`, so the ensemble is sliced by **time value**,
  while `truth_sensor_series` concatenates `slice(w·n_per_window,
  (w+1)·n_per_window)` of a globally-timed series, so the truth is sliced by
  **frame index**. They routinely differ in both length and cadence (the truth is
  saved on its own schedule), so a shared edge list would either mis-slice one
  side or force a pointwise alignment the statistics-space comparison exists to
  avoid. `n_members` is dropped because it is `ensemble_da.sizes["ensemble"]` —
  passing it in would create a second source of truth for a number already on the
  data. The ensemble falls back to contiguous equal blocks (with one
  `logger.info`) when the series carries no `time` coordinate or `sim_time` is
  not a positive duration.

- **WP1.2 — the Wasserstein layer scores `|U|`, pooled over the whole run
  (resolution of a plan ambiguity).** The plan does not say which quantity the
  distribution distance is taken over. `|U|` is used: it is positive (so a
  distance normalized by its own `sigma` is readable), it is the quantity the
  plan's S2/S3 probe figures compare, and a per-component distance would need
  three times the numbers to say the same thing. It is taken over the **whole
  run** rather than per window — a window holds too few frames for a distance
  between empirical distributions to mean anything. Two reductions are emitted,
  not one: `w1_over_sigma_pooled` (all members pooled into one predictive
  distribution) and `w1_over_sigma_member_mean` (each member scored alone, then
  averaged). `W1` is convex in its first argument, so pooled ≤ member-mean
  always; they coincide under a shared bias and separate when the ensemble
  spread brackets the truth, which is a distinction neither number makes alone.

- **WP1.2 — `velmag_mean` added to the plan's three statistics (addition).** The
  plan lists window mean, window variance and TKE. `⟨|U|⟩ ≥ |⟨U⟩|`, with the gap
  being exactly the fluctuation a directionally-wandering flow carries, so a run
  can match `window_mean` component by component and still get the scalar wind
  speed wrong — and the scalar speed is what the sensor figures and the case
  references actually report. It is also the statistic whose raw samples the
  Wasserstein layer scores, so emitting its mean keeps the two layers reading the
  same quantity.

- **WP1.2 — the statistics-space z-score block is a strict subset of WP1.1's
  (departure).** Only `mean` / `std` / `max_abs` / `max_abs_calibrated_median`
  are emitted; `exceedance` / `overconfident` / `overconfident_rule` are not. The
  measured false-alarm table behind that multiplier rule (round-2 entry above)
  was calibrated on pooled **independent** knots, and nothing pooled here is
  independent — one window's sensors see a single realization of a single flow,
  so the `component × sensor × window` elements are strongly cross-correlated and
  the rule's operating point does not transfer. The raw moments and the
  size-aware `max_abs` reference survive that correlation (they stay unbiased;
  only their sampling error inflates), so those are what ships.

- **WP1.2 — `crpss_vs_prior` is `null` on nearly every run, by configuration.**
  It needs the prior ensemble rolled out at the same sensors, i.e.
  `windows/window_{w}_prior_state.nc` for *every* window, and
  `conf/run_esmda.yaml` ships `run.save_prior_state: false`.
  `compute_esmda_metrics._prior_sensor_series` therefore checks all paths exist
  before calling `ensemble_sensor_series` (which opens each path unconditionally
  and would raise `FileNotFoundError` on a partial set) and emits one
  `logger.info`, never a warning — absent is the default, not a fault. `null`
  there means "not saved", never "no skill", and the docs say so.

- **WP1.2 — the integration criterion is met on the same synthetic run dir as
  WP1.1, extended past the `skip_viz` early return.** WP1.2 consumes the truth,
  so unlike the WP1.1 bundle it cannot be driven by parameter artifacts alone.
  Rather than add a solver-backed test, `tests/test_esmda_metrics_wiring.py`
  fabricates the remaining NetCDF artifacts by hand (`posterior_state_mean.nc`,
  a truth state, per-window ensemble state files, an obs config with both
  assimilation and `validation_*_points`) on a `pylbm` identity grid — the only
  solver name a hand-written fixture can use, since udales/palm need staggered
  coordinates and `neural_surrogate` is rejected by `ObservationOperator`. Two
  fixture properties are load-bearing: the truth and ensemble are saved at
  **different cadences** (6 vs 8 frames per window), which is the case the two
  slicing rules exist for, and the window files carry **window-local** time
  coordinates, so the rebasing is exercised rather than assumed. The run also
  sets `run.metrics.bootstrap_blocks: 4` against 8 frames per window, so the
  identifiability bootstrap resolves (`L = 2`) instead of taking its
  smoke-scale null path.

#### Review round 1

The WP1.2 entries above stand except where a bullet below supersedes them (the
Wasserstein layer's pooling axis, and `crpss_vs_prior`'s window alignment). The
common thread across the two substantive findings is the same one WP1.1's round 2
had, pointed the other way: **every number in this layer is a comparison against
a reference, and two of those references were being computed off a different
sample than the number they normalize** — a whole-run distance against a
half-window floor, and a posterior pool against a prior pool built from a
different window subset. Both read as *flattering* results (a deflated
`w1_over_floor`, a spurious non-zero `crpss_vs_prior`), which is why neither
showed up as an obviously broken number. `metrics_version` stays 2 (see the last
entry).

- **WP1.2 (review round 1) — the Wasserstein layer is computed per
  `(sensor, window)` and then reduced, superseding the whole-run pooling logged
  above (the substantive finding).** `wasserstein_self_floor` splits the truth
  into two *contiguous* halves, which is right for a stationary series (that
  choice is unchanged, and was endorsed) but makes the floor absorb any
  deterministic trend inside the split window: the halves are then drawn from
  different parts of the trend and the split measures the trend, not the series'
  own sampling variability. Measured by the reviewer on a 108-frame `|U|` at
  σ_turb = 0.5, with "perfect" an independent realization of the same law and
  "bad" a +20% `|U|` carrying half the turbulence — floor 0.168 → 0.366 → 0.575
  and the bad model's `w1_over_floor` 18.2 → 5.8 → **1.9** as the truth goes
  stationary → magnitude-cosine-only → both cosines. The last row is the shipped
  default (`params@truth_params: dynamic_cosine`, a 400 s inflow-angle cosine and
  a 200 s magnitude cosine over the 540 s run), i.e. **a clearly-wrong model was
  reading as indistinguishable from perfect on the configuration we actually
  ship.**
  The fix is to compute both the distance and the floor per `(sensor, window)`
  and reduce `{median, max}` over the `S × W` elements instead of over `S`.
  Reasons, in the lead's order: (a) it is the only option that keeps the floor and
  the distance **like-for-like on sample count** — a per-window floor against a
  whole-run pooled distance would put a 36-frame floor under a 108-frame distance
  and deflate `w1_over_floor` *further*, the opposite of the fix; (b) it removes
  the cross-window component of the deterministic forcing, which is the dominant
  term in the table above; (c) the schema does not change — the reduction is
  still `{median, max}`, only over more elements. **Detrending was rejected** as
  the primary fix: it needs a trend model the metrics layer has no business
  choosing, and it would silently change what the number means. **Demotion**
  (shipping the ratio as diagnostic-only) **was rejected** because per-window is a
  real fix and WP1.4 wants a usable headline. The reduction now reuses the window
  index lists `sensor_statistic_scores` already computes rather than re-deriving
  the windowing, and elements whose floor or distance is `nan` — a window with
  fewer than the four finite samples `wasserstein_self_floor` needs, which **is
  the smoke shape** at 3 frames — drop out of the reduction rather than nulling
  the block.
  **Residual, documented rather than fixed:** a 180 s window against a 200 s
  magnitude cosine is still not stationary *within* the window, so on the shipped
  default the floor remains somewhat inflated. Per-window shrinks the effect; it
  does not eliminate it. A follow-up measurement sharpens *why*, and the sharper
  statement is what the docs carry: what inflates the floor is the split interval
  containing a **net excursion** of the trend, so the criterion is the forcing's
  period against the split length and not the split length as such. A cosine of
  period ≈ 2× the interval inflates the floor ~3× at *both* 108 frames
  (0.223 → 0.724) and 36 frames (0.369 → 1.426), while a cosine cycling several
  times inside the interval does not inflate it at all (0.294 against a
  stationary 0.369). "Use a shorter window" is therefore **not** a universal
  remedy — a window landing near half the forcing period is the worst case, and
  the shipped 180 s window sits at 0.9× the 200 s magnitude cosine and 0.45× the
  400 s inflow-angle cosine. The per-window change is still the right one on
  argument (a) above (floor and distance like-for-like on sample count) and it
  does remove the cross-window term; it is not a general fix for a trending
  truth, and the docs say so rather than implying that a shorter window would be.
  Both the failure mode and the reviewer's table are now in
  `docs/scripts_and_configs.md` next to the Wasserstein rows, and in
  `wasserstein_self_floor`'s docstring as a second, separately-labelled failure
  mode beside its (correct, retained) AR(1) validation table.

- **WP1.2 (review round 1) — `crpss_vs_prior` could compare two different window
  subsets; alignment is now explicit.** `_pooled_statistic` drops an unusable
  window independently per series, so the posterior and the prior could each
  arrive at an equal-sized pool built from *different* windows and the
  `prior_members.shape == members.shape` guard would pass. Reproduced by the
  reviewer with identical member values and the prior shifted by one window: the
  correct answer is `0.0` and the reported skill was **−0.26 / −0.07 / −0.61 /
  −0.10** across the four statistics. Fixed by making the alignment explicit
  rather than by tightening the shape guard — which would only have converted a
  wrong number into a vanished one: `_pooled_statistic` now also returns the
  ordered list of window indices it used and accepts an explicit `windows=` subset,
  and the caller intersects the two lists. Equal lists score as before; a strict
  subset **re-pools both sides on the intersection** so the metric survives; an
  empty intersection gives `crpss_vs_prior = None` plus one `logger.info`. The
  re-pooled comparison additionally uses a **joint** finiteness filter (truth and
  every posterior member and every prior member finite), so the two CRPS values
  in the ratio are means over exactly the same elements and not merely over the
  same windows. The posterior's own keys (`crps`, `zscore`, `coverage`, `pit`,
  `identifiability`, `n_samples`, `n_windows_scored`) keep using the posterior's
  full window list — a short prior must never degrade the posterior's own
  numbers.

- **WP1.2 (review round 1) — `crpss_vs_prior` is named as one-window-ahead
  skill (documentation).** `windows/window_{w}_prior_state.nc` is window `w`'s
  ESMDA step-0 forecast, and `run_esmda.py` seeds window `w`'s prior from window
  `w−1`'s posterior — the same chaining WP1.1's round 2 found under
  `contraction_ratio`. So pooling all windows makes this **one-window-ahead**
  skill, not skill against the run's initial prior, and it is therefore a
  different quantity from the `crps_reduction_vs_prior` sitting in the parameter
  bundle of the *same* summary file. WP1.1 built its whole
  `vs_window_prior` / `vs_initial_prior` split around exactly this distinction
  (`compute_esmda_metrics.py:376-385`), so leaving it implicit here would have
  invited a cross-bundle comparison that is not meaningful. Stated in the
  `sensor_statistic_scores` docstring and in the key table. No
  `vs_initial_prior` counterpart is added: it would need the initial prior rolled
  out at the sensors for every window, which no run saves.

- **WP1.2 (review round 1) — `n_windows_scored` per statistic (addition).**
  `_pooled_statistic`'s bare `continue` was the one degradation path in this
  layer with neither a `reason` nor a log line: with only window 0 of 3
  contributing, the summary read `n_windows: 3`, `reason: null`, and the sole
  trace was `n_samples` quietly dropping 18 → 6 — a number a reader has no
  independent expectation for. The count now travels with the block. It is
  attached **per statistic, not per sensor set** (the review suggested the set),
  because windows drop per statistic: a 1-frame window kills `window_variance`
  and `tke` while `window_mean` survives it, so one per-set number would be wrong
  for at least one block. It is emitted on `_null_statistic_block`'s degraded path
  too — as `0`, not `null`, breaking this layer's "every key null when degraded"
  rule deliberately: it is a *count of what contributed*, nothing did, and a
  consumer dividing by it must see the zero rather than an absent key. Each
  statistic also emits one `logger.info` naming the dropped window indices when
  any drop. `n_windows` is correspondingly documented as the *configured* count
  and points at the new key for the contributing one, and the wiring test's
  `STATISTIC_KEYS` frozenset pins the addition alongside an
  `n_windows_scored == n_windows` assertion on the healthy synthetic fixture, so
  a future silent drop fails loudly instead of moving `n_samples` to another
  plausible number.

- **WP1.2 (review round 1) — the within-member bootstrap is vectorized
  (`block_bootstrap_std_batch`).** `_within_member_std` was a Python double loop
  over `(member, element)` at ~1.0 ms per `block_bootstrap_std` call: ~4.6 s at
  the shipped shape (M = 32, C = 3, S = 8, W = 3) but **~3.4 min** at
  M = 128 / S = 20 / W = 10 — on the shipped default `level: standard`, i.e. a
  cost paid by every production run and growing with exactly the knobs a real
  campaign turns up. A batched sibling in `turbulence_stats.py` builds the
  resample index matrix once and applies it to every element, so the cost is one
  `statistic(..., axis=-1)` call on `(n_resamples, n_elements, n_time)`. The
  scalar `block_bootstrap_std` is kept **unchanged** (WP1.3 wants it, and it is
  already documented and tested), and both are built on one shared private
  index-matrix helper so the two cannot drift; a single-element batch is pinned
  by test to equal the scalar function **exactly**, which is what lets a reader
  trust that the fast path and the documented path are the same estimator. The
  batched statistic is called as `statistic(x, axis=-1)` rather than the scalar
  form's `statistic(1-D) -> float` — that contract is what makes the
  vectorization possible and is documented as such. `np.mean` and the `ddof=1`
  variance are re-expressed as axis-taking reducers; the TKE Bessel-factor-on-the-
  integrand construction (endorsed in review) survives the refactor bit-for-bit,
  asserted by test rather than assumed — a new test pins `_within_member_std`
  against the scalar double loop with `np.array_equal(..., equal_nan=True)` for
  both reducers, and the identifiability numbers are bit-identical across the
  change (`window_mean` `ratio_median` 1.0421934131499127 before and after, all
  four statistics unchanged to the last digit). Measured end to end at the
  shipped shape (36 frames per window, 20 blocks, 200 resamples): the
  identifiability pass **3.709 s → 0.146 s, 25.4×** — 6144 scalar
  `block_bootstrap_std` calls replaced by 12 batched ones (4 statistics × 3
  windows) — and the whole of `sensor_statistic_scores` 3.733 s → 0.171 s, i.e.
  the bootstrap was ~99% of this layer's runtime and is no longer its dominant
  cost. On the bootstrap alone,
  with exact equality against the scalar function asserted at every shape: at 288
  rows `np.mean` 0.100 s → 0.006 s (**17.7×**) and the `ddof=1` variance
  0.245 s → 0.007 s (**35.7×**); at 768 rows 18.5×, and at 7680 rows — the
  M = 128 / S = 20 scale that motivated this — 2.94 s → 0.183 s, 16.0×.
  Non-finite rows are the one behavioural difference and it is deliberate: a row
  containing any non-finite sample returns `nan` outright rather than falling back
  to the scalar path, because the scalar form drops non-finite samples *before*
  blocking, so a gappy row has a different finite count, a different block length
  and a different index matrix — and one shared index matrix is the entire
  speedup. WP1.2's rows are fully finite or absent (a masked sensor is dropped
  upstream, never passed as a row of gaps), so the `nan` reports an input this
  path does not serve rather than silently approximating it; other rows in the
  same batch are unaffected, pinned by test.

- **WP1.2 (review round 1) — `bootstrap_blocks: 1` yields a measured `0.0`;
  documented, not raised — but the zero had to be made real first.** With
  `n_blocks = 1` the block length is the whole series, so every bootstrap
  replicate is the identical series and the spread is zero by construction, not
  the `nan` the Returns section documented for `L < 2`. The lead's ruling is to
  **document the zero and not raise**: `bootstrap_blocks: 1` passes
  `resolve_metrics_settings`'s `>= 1` validation, and WP1.0's round-1 entry above
  says in as many words that the validator exists so a bad knob is reported
  cheaply *instead of* crashing deep inside a streaming pass that has already read
  GBs; raising from `block_bootstrap_std` would reintroduce precisely that failure
  mode one layer down.
  **The ruling's stated consequence did not actually hold as shipped, which is the
  finding here.** `np.std(..., ddof=1)` over 200 *bit-identical* replicates does
  not return `0.0` — the mean subtraction leaves float rounding, measured
  **2.78e-17** at `n = 60` with `np.mean` and up to ~1.3e-15 for a `ddof=1`
  variance. That is strictly positive, so it **passes** `_identifiability_block`'s
  `within > 0` filter and `identifiability` would have come out at ~1e16–1e17
  rather than the clean `null` the ruling relies on: the degradation path the lead
  reasoned from was reachable only in principle. The shared reduction step now
  collapses the identical-replicate case to a true `0.0` before calling `np.std`
  (a point-mass bootstrap distribution has zero spread by construction, not by
  measurement), which restores the ruling's outcome without a raise. Verified
  against `git show HEAD` that this is the *only* behaviour change to the scalar
  function: over `n ∈ {21, 36, 50, 108, 400}` × `n_blocks ∈ {1, 3, 7, 20, 60}` ×
  {mean, `ddof=1` variance, median} × seeds {default, 0, 7}, every combination is
  bit-identical except `n_blocks = 1`, which moved from ~1e-16 to exactly `0.0`;
  the gappy and explicit-generator paths are bit-identical too. Pinned by a test
  asserting `== 0.0` exactly, for both the scalar and the batch function.

- **WP1.2 (review round 1) — `w1_over_floor`'s ordering was correct and its
  documentation was not; `metrics_version` stays 2.** The ratio was already
  computed per element and then reduced, which is the right ordering — the review
  endorsed it — but the key table called it "the pooled distance in units of that
  floor", which invites reading it as `w1_over_sigma_pooled.median /
  self_floor.median`. That ratio-of-medians would pair one element's distance with
  another element's floor, and the two medians need not even be attained at the
  same sensor. The row now says both what the number is and what it is not.
  On the version question: **nothing was reprocessed at `standard` between #99
  merging (2026-07-30T05:55Z) and this PR**, so no artifact carries the old
  semantics under the new number. The only `run_summary.yaml` on this machine is
  `.temp/pyudales_to_pyudales/run_summary.yaml` from 2026-07-29 18:42 — *before*
  the merge — and it carries `metrics_version: 2` with **no `metrics_level` key
  at all**, which makes it a phase-0 artifact rather than a phase-1 one. So
  `metrics_version` correctly stays 2: the WP1.2 keys have never been written to
  a persisted summary under semantics different from the ones shipping here. The
  check covered the local `.temp/` tree only and cannot see HPC scratch; a run dir
  reprocessed at `standard` on Snellius or DelftBlue between those two timestamps
  would not have been found by it.

#### Review round 2

Round 1's entries stand — the reviewer re-verified all seven against the code
(56-case scalar/batch equivalence with 0 mismatches, bit-identical output from
4 KB to 1 GB chunk budgets, the isolated `crpss` case returning exactly `0.0`,
`n_windows_scored`, 4.6 s → 0.15 s) and none is reopened here. The round-2 thread
is again this layer's recurring one, now pointed at the reference that round 1
introduced rather than at the one it fixed: **`w1_over_floor` was documented as
being calibrated at 1, and it is not.** Round 1 made the floor and the distance
like-for-like on *sample count* per element; it did not make the ratio's *value*
comparable to 1, because the floor is still `n/2` truth samples against `n/2`
while the numerator is `M·n` pooled member samples against `n`. So the one round-1
sentence that reads as a calibration claim — "~1 is indistinguishable from
perfect" — was wrong by ~2×, in the flattering direction, on the number WP1.4 is
about to plot. `metrics_version` stays **2**: `w1_over_floor`'s definition and
ordering are unchanged, the round-2 work only *adds* a key beside it, and the
round-1 audit establishing that no persisted `standard` summary carries the older
semantics still holds.

- **WP1.2 (review round 2) — `w1_over_floor` is not calibrated at 1; a computed
  perfect-model reference now ships beside it (the substantive finding).**
  Measured on `ensemble_scores` directly, no windowing involved (AR(1) φ = 0.6,
  stationary mean, M = 32, median of 200 trials):

  | `n` scored | perfect model | +0.5σ bias | σ × 0.5 |
  |---|---|---|---|
  | 18 | 0.77 | 0.93 | 0.75 |
  | 36 | 0.56 | 0.99 | 0.80 |
  | 108 | 0.53 | 1.48 | 1.25 |
  | 216 | 0.50 | 2.02 | 1.72 |
  | 432 | 0.54 | 3.00 | 2.47 |

  A perfect model sits at **~0.55 and is flat in `n`** (numerator and floor both
  go as `1/√n`); a genuine error has an `n`-independent numerator, so its score
  **grows as `√n`**. Two things follow. (a) A model reading exactly `1.0` is
  already ~2× worse than perfect, so the old wording invited reading a real error
  as a clean pass — and at the *shipped* 36-frame window the raw ratio separates a
  +0.5σ bias (0.99) from a perfect model (0.56) by so little that it is unreadable
  without a reference. (b) The reference has to be *computed*, not stated:
  **the lead's ruling was a computed reference over a prose constant**, because
  0.55 was measured at one φ and one `M`, the real value moves with the series'
  autocorrelation and the ensemble size, and a prose constant cannot travel with a
  number that WP1.4 plots. This is also exactly the device the rest of this module
  already uses for every nominal-vs-calibrated trap — `coverage.nominal_alpha_*`,
  `zscore.max_abs_calibrated_median`, `zscore_nominal_exceedance` — and WP1.1's
  round 2 states the rule as "a calibration number ships with its reference".
  `_esmda_common` emits it as `wasserstein.w1_over_floor_calibrated_median`, per
  `(sensor, window)` on the truth arrays `_wasserstein_block` already extracts and
  reduced by the same `{median, max}`, so it is comparable to `w1_over_floor`
  element for element; it is in `_null_wasserstein_block` too, so the key set is
  stable.

- **WP1.2 (review round 2) — the reference is a *two-sample* block bootstrap, not
  the one-sided construction the implementation contract sketched (departure, and
  the reason the code reads differently from the plan).** The obvious estimator —
  the floor is deterministic given the truth, so resample only the model side and
  score the pooled `M·n` samples against the truth array itself — is **wrong by
  4–6× and gets worse as `M` grows**, returning 0.05–0.13 at `M = 32` where the
  answer is ~0.55. The pooled sample is `M·n` draws from the truth's *own
  empirical distribution*, so its ECDF converges to that ECDF and
  `W1(pooled, truth) → 0`: it measures the model's sampling noise, which pooling
  averages away, and misses the term that actually dominates — the truth window's
  own deviation from the law it was drawn from. What shipped instead draws
  `n_members + 1` moving-block resamples per replicate, pools `n_members` of them
  as the synthetic model and scores them against the remaining one as a **stand-in
  truth window**, dividing by *that stand-in's* floor. Both sides are then samples
  of one law, which is the comparison the number is supposed to make.
  Resampling still reuses `turbulence_stats._block_resample_indices` rather than
  adding a second block-resampler — the single-source rule round 1 established for
  the scalar and batch bootstraps.
  **The price is conditioning, and it changes what the key means:** the median now
  runs over the sampling variability of the truth *window* as well as the model's,
  so it is "the ratio a perfect model of a series like this, at this length and
  `M`, typically scores", **not** "the ratio conditional on this exact window".
  That is unavoidable rather than a shortcut — the dominant term *is* the window's
  own sampling error, and no estimator conditioned on that one window can see it.
  The docs row is worded accordingly.
  **Validated, not asserted** (the bootstrap reuses the truth's values, so neither
  synthetic side is a fully independent realization). Ratio of the reference to a
  directly simulated independent perfect model, paired per truth window (AR(1),
  `M = 32`, 120 windows, 5 model realizations each): 1.02 / 0.99 / 0.98 at φ = 0,
  1.05 / 0.91 / 1.06 at φ = 0.6, 0.55 / 0.88 / 1.08 at φ = 0.9, for
  `n = 36 / 108 / 432`. Within 12% at eight of nine. The exception has a clean
  mechanism — at φ = 0.9, `n = 36` the block length is 2 and cannot carry a
  correlation spanning ten frames, so the resampled sides under-inherit the slow
  excursions that make a short strongly-correlated window a poor sample of its own
  law — and **it errs low, which is *not* the safe direction**: an understated
  reference makes a good model look bad, unlike the floor's own conservatisms. The
  docs therefore tell a reader to treat a score moderately above its reference on
  a short strongly-correlated probe as inconclusive rather than as a failure.
  Cost stayed inside the budget round 1's vectorization bought: at the shipped
  shape (`M = 32`, `n = 36`, 24 elements) the default `n_resamples = 64` is 7.3 ms
  per call / **0.18 s** for the layer, against 0.05 s at 16 and 0.35 s at 128, with
  per-element Monte-Carlo spread on the returned median of 19% / 12% / 8%
  respectively. 64 keeps a 12% per-element error well under the ~2× effect being
  diagnosed, and the `{median, max}` reduction over 24 elements averages the
  median's down further; a consumer plotting a *per-element* reference should raise
  it, since the error falls as `1/√n_resamples` while the cost rises linearly. No
  silent cap was added.

- **WP1.2 (review round 2) — per-window is retained, and the interpretation
  question it raised is now answered by the reference rather than by the pooling
  axis.** Round 1's change divided every *real* error's score by ~`√W` (108 frames
  → 36 at the shipped cadence, ~1.7×) while leaving a perfect model's score flat,
  which cost sensitivity: the reviewer measured a +20% mean bias reading **2.02
  whole-run → 0.85 per-window**, a perfect model **0.44 → 0.27**, i.e. bad/perfect
  separation **4.6× → 3.1×**. **Provenance caveat, load-bearing:** those
  end-to-end numbers come from a *synthetic stand-in for a probe series*, not from
  a real run, so the magnitude is indicative; the `√n` mechanism producing it is
  structural and applies to any run. Per-window still stands on round 1's
  argument (a) — floor and distance like-for-like on sample count, and the
  cross-window trend term removed — and reverting would reintroduce the
  `n`-frame-floor-under-a-`W·n`-frame-distance mismatch. What changes is that the
  choice now **matters less for interpretation**: score and reference are computed
  on the same elements at the same sample count, so either pooling axis is
  readable, and the residual sensitivity cost is visible in the gap between the
  two numbers instead of being silent. The reference does **not** rescue the
  round-1 residual (a within-window deterministic trend), and it calibrates sample
  count rather than stationarity: its stand-in window is moving-block resampled (a
  2-frame block at the shipped shape), so it cannot inherit a trend spanning the
  window, and the reference is effectively computed on a de-trended series while
  the score itself still divides by the inflated floor. Reasoning from the
  construction rather than from a measurement, that points the *score* below its
  reference on a trending truth — so the docs say to read a score well below its
  reference as "the floor ate it" rather than "better than perfect", and note that
  the pair is not a trend test in either direction.

- **WP1.2 (review round 2) — the whole-run trend table is relabelled as the
  pre-change baseline it actually is (documentation).** The "Reading the
  Wasserstein rows" table (floors 0.168 / 0.366 / 0.575; bad-model ratios
  18.2 / 5.8 / 1.9) is a **whole-run** measurement taken *before* round 1 made
  every Wasserstein key per `(sensor, window)`, but it sat inside a section that
  now describes the per-window metric, so a reader would take those numbers as
  current output. They are now explicitly labelled the pre-change whole-run
  baseline **that motivated windowing** — their real role — rather than
  re-measured, the table header carries the label too, and the surrounding prose
  no longer implies any of them is comparable to a key in a current
  `run_summary.yaml`. The round-1 entry above describes that same table as a
  clearly-wrong model "reading as indistinguishable from perfect"; with the
  calibration finding in hand the accurate statement is narrower — 1.9 is ~3.5× a
  ~0.55 perfect-model reference, and the perfect model of that same trending truth
  read 0.12, so the pair still separated. What the trend destroyed was the
  *absolute* readability of the number (against the wrong 1.0 anchor it looks like
  a pass), which is what both round 1's fix and round 2's reference address from
  their two different sides.

- **WP1.2 (review round 2) — `crpss_vs_prior` is documented as not recoverable
  from the reported keys (documentation).** Round 1 made the posterior keep its
  **full** window list and its own finiteness filter while `crpss_vs_prior` is
  computed on the **intersection** under a **joint** filter (truth and every
  posterior member and every prior member finite) — both halves of that are
  deliberate and unchanged. The consequence was left implicit: the two CRPS values
  inside the skill score are means over a different element set than the reported
  `<stat>.crps`, so `crpss_vs_prior` is **not** reproducible as
  `1 − crps/prior_crps` from the summary, and the prior's CRPS cannot be backed
  out of it either. Now said in the key table, so nobody derives a prior CRPS that
  never existed.

- **WP1.2 (review round 2) — the wiring test's `WASSERSTEIN_KEYS` frozenset gains
  the new key.** Same role as round 1's `STATISTIC_KEYS` addition: the key set is
  pinned in the wiring layer and asserted present *and finite* on the healthy
  synthetic fixture, so a reference that silently stops being emitted (or starts
  coming out `nan` on a case that scores fine) fails loudly instead of quietly
  removing the only thing that makes `w1_over_floor` readable.

### WP1.3

Common thread, and worth stating once because it is the same shape three times:
**three of the premises this WP's plan text rests on turned out to be false
against the actual backends and the actual tree**, and each was found by
measuring rather than by reading. (a) "fluid cells only (`~np.isnan` after
masking; solid cells are the NaN/blocked cells)" — **no shipped backend marks
solid cells that way**; measured NaN fraction of `u`/`v`/`w` is 0.0 in every
state artifact inspected. (b) "**Do not** reuse the by-index `|U|` shortcut —
fine for magnitude summaries, wrong for stresses" — the conclusion is right and
the mechanism is not the one that wording implies: colocation does not recover
grid-scale stress *amplitude* at all, it makes the tensor *self-consistent*.
(c) "if the mean-field pass busts [the runtime budget], drop smoke `n_z_slices`
to 2 via `_SMOKE_OVERRIDES`" — measured, that changes nothing, because at the
smoke shape's 1600 cells this layer's cost is entirely fixed. A fourth,
smaller one belongs beside them: §Acceptance's "no `.load()` of window state
files anywhere (grep)" **was not true of the tree before this WP** —
`ensemble_sensor_series` loaded them whole — so step 3's shared read is a repair
as much as an optimization.

Only (a) moves a number, and it moves it in the flattering direction, which is
why it is the one carrying a `logger.warning` and a `masking` block rather than
a docstring. `metrics_version` stays **2**: WP1.3 is purely additive — one new
top-level `mean_field_metrics` mapping plus one new artifact — and the WP1.2
keys' byte-identity across the streaming refactor is measured, not assumed (see
below), so no existing key changed value or semantics.

#### The math module (`src/pyurbanair/utils/turbulence_stats.py`)

- **WP1.3 — `hit_rate(pred, obs, *, allowance, relative_tolerance=0.25)`
  (signature reshaped).** The plan writes `hit_rate(pred, obs, D=0.25, W)`,
  which is not valid Python — a positional parameter cannot follow a defaulted
  one. `W` became `allowance`, **keyword-only and required with no default**,
  because there is no universal value for it: it is the truth's own bootstrap
  sampling floor `σ_u/√N_eff`, which makes `q` an "indistinguishable within
  sampling error" test rather than an arbitrary threshold, and a default would
  invite scoring against a number nobody chose. `D` became
  `relative_tolerance` (`DEFAULT_HIT_RATE_TOLERANCE = 0.25`). `allowance=nan`
  is a **documented legal value** meaning "floor unavailable" — the smoke
  shape, where `block_bootstrap_std` is undefined at 3 frames against 20 blocks
  — and it skips the absolute clause entirely, degrading `q` to the pure
  relative test, which can only be *lower*: a missing floor never flatters a
  run. A **negative** allowance raises, since it admits no points and is always
  a bug, unlike `nan`.

- **WP1.3 — the plan's "solid cells are the NaN/blocked cells" premise is false
  for every backend (the substantive finding).** Verified in the backend code
  and on the `.temp` artifacts: measured NaN fraction of `u`/`v`/`w` = **0.0**
  in all five state files inspected, independently re-verified by the lead on
  three of them. What each backend actually does: **pylbm** writes a `blanking`
  variable (1 = solid) into the same Dataset as the velocities, in both its
  truth state and its window state files; **pypalm** explicitly `fillna(0.0)`s
  PALM's NaN (`libs/pypalm/src/pypalm/forward_model.py:1019-1026`) and keeps no
  mask, so a PALM mask is **not recoverable from the artifact at all**;
  **uDALES** fielddumps carry small non-zero junk inside buildings and ship no
  mask (the training-data scripts attach `blanking` from `solid_c.txt` after the
  fact, which needs case-dir paths this stage does not have). Two consequences,
  both load-bearing: masking is the **driver's** job and not the math layer's
  (`hit_rate`'s docstring says so — it drops non-finite pairs and states that
  the caller must have masked already), and an unmasked `fac2` scores the
  zero-vs-zero pair inside a building as a hit, so it is optimistic by roughly
  the building fraction. What the driver does with this is the `masking` entry
  below.

- **WP1.3 — `StreamingMoments` does not accumulate `Σu_i u_j` literally
  (departure from the plan's wording, on measurement).** The plan specifies
  `n`, `Σu_i`, `Σu_i u_j`; the class uses the chunk-wise Chan/Welford combine
  instead (per-chunk co-moment about the chunk mean, plus `δᵢδⱼ·nm/(n+m)`).
  Relative error of the `ddof=1` variance against an exact (longdouble
  two-pass) reference, `n = 360` in 10 chunks of 36, naive `Σuu − (Σu)²/n`
  against this class, as the mean/fluctuation ratio grows:

  | mean / fluctuation | naive sums | this class |
  |---|---|---|
  | 25× | 8.0e-14 | 1.7e-16 |
  | 2500× | 1.6e-09 | 4.4e-15 |
  | 1e7× | 3.9e-02 | 7.1e-11 |

  The cross moment follows the same pattern (1.1e-02 vs 7.7e-11 at 1e7×), and
  the class is within ~2× of a whole-series two-pass — i.e. chunking costs
  nothing numerically, which is what makes the chunk length a free memory knob.
  This is not a hypothetical regime: **the layer's regime *is* a small
  fluctuation on a large mean** (urban flow, 5 m/s mean against 0.2 m/s
  fluctuation), which is exactly where a literal sum of squares loses its
  digits.

- **WP1.3 — NaN policy: per-cell `int64` count, casewise deletion (plan
  silent).** A frame contributes to a cell only if **all** components are finite
  there. Per cell rather than one scalar count because a blocked cell and the
  NaN edge of an interpolated truth field are per-cell facts, not per-frame
  ones. Casewise rather than per-pair because available-case counts mix sample
  sets *within one matrix*: the entries of `R` are then computed on different
  frame subsets, which can make `R` indefinite and `tke` negative. A test
  asserts a PSD matrix on an adversarial mask, so the choice is pinned rather
  than assumed.

- **WP1.3 — Bessel: `ddof=1` by default, applied to the reduced moment
  (cross-WP consistency, resolved deliberately).** `tke()` is then exactly
  `0.5·Σᵢ var_ddof1(uᵢ)`, i.e. WP1.2's sensor TKE estimator, to the last digit
  rather than approximately. WP1.2 put the `n/(n−1)` on the *integrand* only so
  that the moving-block bootstrap resampled the reported estimator; nothing is
  resampled inside this class, so the factor belongs on the reduced value and
  the two definitions coincide. `ddof=0` is exposed for a caller who wants the
  biased moment.

- **WP1.3 — `colocate_components` returns `xarray.DataArray`s, not bare arrays
  (plan ambiguity resolved).** The plan says "arrays/DataArrays". DataArrays,
  because they carry the **centre coordinates**, which is what the driver needs
  for the cross-grid truth `.interp` — handing back bare arrays would force the
  caller to re-derive the axes the function just constructed. Output dims are
  `(…, zt, yt, xt)` for udales and `(…, z, y, x)` for palm/pylbm; `ensemble` and
  `time` pass through untouched.

- **WP1.3 — domain edge: linear extrapolation, not column-dropping
  (departure; the plan does not say).** The staggered axes are *lower* faces
  with the **same length** as the centres (measured `xm − xt = −dx/2`), so the
  last centre has no upper face and is extrapolated with weights (1.5, −0.5).
  That is the same choice pypalm's own `zw→z` postprocessing makes, it is exact
  for a linear field, and it keeps the returned grid identical to the centre
  axes so nothing downstream has to special-case a short axis. The costs are
  documented rather than hidden: variance amplification **2.5×** in the last
  column, and a periodic domain would prefer wrapping — but the boundary
  condition is not recoverable from the state file, so wrapping cannot be
  chosen here.

- **WP1.3 — new guard `_check_face_centre_alignment` (addition).**
  `colocate_components` raises unless `centre[i] == ½(face[i] + face[i+1])`
  within 1e-3 of the local spacing. It exists for one specific
  silent-wrong-number case: a caller who selects z-levels **before** colocating
  leaves both axes uniformly spaced, so a spacing check would pass while the
  interpolation blends cells five apart. It also raises on a single-level
  staggered axis, where one point defines no interpolant. This guard is what
  sets the driver's memory floor (see the `_TRUTH_CHUNK_BYTES` entry below) —
  it is the reason a truth chunk cannot be colocated one z-level at a time.

- **WP1.3 — the plan's "by-index is wrong for stresses" is right, but not for
  the reason the wording implies (correction, measured, and the docstring says
  so).** On a homogeneous synthetic field (Gaussian correlation length `L`,
  6000 frames), colocation does **not** recover grid-scale stress amplitude —
  `⟨u'w'⟩` error is −22.1% by index against −22.2% colocated at `L = dx` — and
  it *adds* a −19.7% loss on the diagonal, where by-index was unbiased. What it
  buys is tensor **consistency**: the anisotropy ratio `⟨u'w'⟩/k` goes
  −22.1% → **−3.1%** at `L = dx` and −6.0% → **−0.2%** at `L = 2dx`, because the
  interpolation filter is identical for every entry of the tensor while the
  by-index lag is not. Plus the unambiguous half of the claim: a linear mean
  profile is displaced by exactly `0.5·dx·dU/dx` (0.05 measured) by index and is
  exact to 2e-16 colocated. Recording this matters because a reader told only
  "by-index is wrong for stresses" will expect the colocated stress
  *magnitudes* to be right, and at `L ≈ dx` they are ~20% low either way — the
  layer resolves what its grid resolves.

- **WP1.3 — `nmse` is `nan` when `ō·p̄ ≤ 0`, `nmse_split` is `(nan, nan)` for
  `|FB| ≥ 2` (documentation of a routine regime, not an error path).** Those
  are exactly the arguments where the formulas divide by zero or return a
  negative systematic part. Both are unreachable for a positive field and
  **routine** for a signed velocity component whose mean passes through zero —
  which is why it matters here specifically: the schema emits per-component
  `nmse`. On `pylbm_to_pylbm` it fires in practice (`nmse.v` → `null`,
  `nmse.w` = 382), and `nmse.velmag` / `fb.velmag` were added (below) so the
  layer always carries one well-posed number regardless.

- **WP1.3 — `fac2` raises `ValueError` on any negative paired value
  (enforcement, not documentation).** The plan says "positive quantities only —
  apply to `|U|`, never to signed components". Once the number is in
  `run_summary.yaml` that instruction is unenforceable and unfalsifiable, so it
  is enforced at the call instead: the error names how many of the fluid points
  were negative and points the caller at `hit_rate`'s absolute allowance, which
  is the metric built for a sign-changing quantity.

- **WP1.3 — descoped: optional skewness/kurtosis, and any accumulator
  `merge()` / `__iadd__`.** The metrics doc offers the higher moments
  "optionally" and the WP1.3 bullet list does not ask for them. A merge
  operation has no consumer either: windows are simply more `update` calls on
  the same instance, which is what makes the statistics span the run. A
  parallel-member reduce would need one — that is the shape to add it for, not
  this one.

- **WP1.3 — the metrics doc's path was checked rather than corrected blind
  (documentation).** The real file is
  `docs/plans/esmda_turbulence_evaluation.md`, and the new docstrings cite it in
  full. The `docs/esmda_turbulence_evaluation.md` spelling that prompted this
  item does **not** appear anywhere in the tree — `grep -rn
  'docs/esmda_turbulence_evaluation'` over `docs/ scripts/ src/ tests/` returns
  nothing; this plan file and `master_plan.md` both reference it as the relative
  link `../esmda_turbulence_evaluation.md`, which resolves correctly from
  `docs/plans/esmda_evaluation/`. What *was* wrong is that the rendered link
  **text** showed a bare `../` path, so a reader (or an agent) copying it out of
  the prose produced the non-existent root-level path. Both files now spell the
  repo-root path in the link text and keep the relative target.

#### The shared streaming pass (`scripts/esmda/_esmda_common.py`)

- **WP1.3 — the shared pass repairs a pre-existing violation of invariant #2;
  it does not merely avoid creating one (context, and the reason this refactor
  is in scope at all).** `ensemble_sensor_series` did
  `xarray.open_dataset(path).load()` on `windows/window_*_posterior_state.nc` —
  ~1 GB at smoke scale, tens of GB at Barcelona scale. So §Acceptance's "no
  `.load()` of window state files anywhere (grep)" did **not** hold before this
  WP, in the one place phase 1 reads those files most. The plan's step 3 reads
  as an optimization for the new layer; it is also the fix for the old one, and
  the two consumers now hang off one `stream_window_members` pass.

- **WP1.3 — the yielded member is materialised, not lazy (departure from the
  lead's own pinned design, accepted on measurement).** xarray does **not**
  cache reads taken through `.isel`, so a lazy yield makes `N` consumers cost
  `N×` the bytes — which defeats the single-shared-read requirement the shape
  exists to satisfy. Measured by counting `NetCDF4ArrayWrapper._getitem` on a
  fabricated `(ensemble 8, time 8, z 6, y 7, x 9)` window file, where one full
  ensemble is 72576 elements:

  | shape | reads | elements |
  |---|---|---|
  | HEAD's full-ensemble `.load()` + 2 sensor sets | 3 | 72576 |
  | lazy member, 1 consumer | 24 | 72576 |
  | lazy member, **2 consumers** | 48 | **145152** |
  | materialised member, 2 consumers | 24 | 72576 |

  Materialising bounds the read at once-per-member for **any** number of
  consumers, at a peak of one member's velocity fields instead of the
  ensemble's. **Stated plainly rather than hidden behind the acceptance grep:**
  `Dataset.compute()` is `.load()` on a shallow copy, so this *is* a load — of
  a member, which is precisely the granularity invariant #2 prescribes ("stream
  member-at-a-time"). The generator's docstring says this in as many words, so
  a future reader auditing the grep result is not misled by the spelling.

- **WP1.3 — the generator does *not* colocate (departure from the plan's
  wording, directed by the lead).** The plan has `stream_window_members`
  applying `colocate_components`. It does not, because the two consumers need
  different things from the same bytes: the sensor extraction interpolates each
  component **on its own staggered grid** (which is what makes the sensor series
  grid- and solver-independent) and needs the raw staggered dims, while the
  moment accumulators need cell-centre arrays. Colocating in the generator would
  destroy the former and charge its cost to every consumer. Confirmed necessary
  by test rather than argued: a staggered uDALES fixture (u on `xm`, v on `ym`,
  w on `zm`) round-trips bit-identically only because the member arrives raw.

- **WP1.3 — `_stack_window_members` reproduces the old path's *memory layout*,
  not just its values (the surprise).** Every consumer of the window series
  reduces over some of its axes, numpy's pairwise summation walks a reduction in
  memory order, and floating-point addition is not associative — so identical
  contents in a different layout move `run_summary.yaml` in the 16th digit.
  Measured by diffing all 587 leaves (260 floats) of the summary computed on the
  WP1.2 wiring fixture against the old code path:

  | how the window's members are stacked | leaves changed |
  |---|---|
  | `concat(dim="ensemble")` then `transpose` | 4 (`sensor_metrics.*.velocity_vector_energy_score`) |
  | …then forced C-contiguous | 20 (incl. `sensor_statistics.*.crps`, `zscore.*`, `identifiability.*`, `crpss_vs_prior`) |
  | reproducing the old `(component, sensor, ensemble, time)` layout | **0** |

  All the moves were 1–2 ULP (largest relative 2e-14, on `crpss_vs_prior`, a
  ratio of near-equal CRPS values) and no estimator changed — but WP1.2's
  deviation log pins its calibration numbers to the last digit, so byte-identity
  was judged worth the ~4 lines. The layout being matched comes from numpy's
  advanced indexing inside `interpolate_dataarray_at_points` (which lays the
  gathered trilinear corners out with the sensor axis outermost), and **nothing
  documents that**, so the code guards on the member dims and falls back to a
  plain concat — giving up only those last ULPs — when they are not exactly
  `(component, time, sensor)`.

- **WP1.3 — `stream_window_members` reads only `u, v, w` by default
  (narrowing; `variables=None` widens).** Window files carry every variable the
  solver wrote; the old full-ensemble `.load()` read them all and used three.
  Output is unaffected, and the driver asks for `blanking` explicitly — after a
  **header-only** open of one window file to check it exists there, since the
  generator raises on a missing variable and pypalm/uDALES states have none.

- **WP1.3 — `ensemble_sensor_series` now requires an `ensemble` dimension
  (behaviour narrowing).** It raises `ValueError` on a window file without one,
  where previously it would silently return a series with no ensemble axis and
  let a downstream reduction produce a plausible wrong number. No current caller
  can hit this; it is recorded because it *is* a behaviour change.

- **WP1.3 — bit-identity of the sensor series was verified, not assumed.**
  Reference = `git show HEAD:scripts/esmda/_esmda_common.py` loaded as a module
  and run side by side. Five shapes (the WP1.2 wiring fixture; per-window
  cadences of 5/9/4 frames with offset local time axes; no `time` coord; no
  `ensemble` coord variable; an extra data variable in the file): all `dims`
  equal, `np.array_equal(..., equal_nan=True)` True, raw-bit `view("u8")` equal,
  `.identical()` True. End to end, `compute_metrics` run twice on one synthetic
  run dir with HEAD's `ensemble_sensor_series` monkeypatched in gives **587
  `run_summary.yaml` leaves (260 floats) equal bit for bit**, with a null
  experiment confirming the harness was actually sensitive to a change.

#### The mean-field driver (`compute_esmda_metrics.py` + `_esmda_common.py`)

- **WP1.3 — masking is `blanking`-only, and what that leaves unmasked is
  reported rather than papered over (the lead's ruling, resolving the false
  premise above).** The `masking` block emits `source` / `truth_source` /
  `ensemble_source` / `fluid_fraction` / `ensemble_fluid_fraction` /
  `truth_finite_fraction` / `note`, and a `logger.warning` fires when neither
  side carries a mask. Parsing uDALES `solid_c.txt` (it needs case-dir paths
  this stage does not have) and preserving pypalm's NaN (backend-touching, which
  phase 1 forbids) are both **descoped**, recorded in the `note` string and in
  `BLANKING_VAR`'s comment. The honest consequence, stated in the docs where a
  reader of the number will meet it: on an unmasked case — **uDALES and PALM
  today** — FAC2 and the hit rate include building interiors and are optimistic
  by roughly the building fraction.

- **WP1.3 — cross-grid masking NaN-masks the truth *before* `.interp`
  (substantive tradeoff).** Linear interpolation propagates NaN, so any target
  cell whose stencil touched a building drops out. That is the conservative
  threshold, obtained for free and with no second mask interpolation to keep in
  sync. **Measured cost:** on `pylbm_to_pylbm`, whose truth is half-cell-shifted
  by `x_offset: -0.5`, the fluid fraction goes 0.75 → 0.6475 — i.e. **13.7% of
  genuine fluid cells lost to perimeter erosion**, and the number is reported
  (`masking.fluid_fraction`, `averaging.n_scored_cells`) rather than absorbed.
  **Rejected alternative:** interpolate the field and the mask separately, which
  keeps those cells but scores them partly against building interiors — a wrong
  number instead of a missing one.
  *(Corrected in review round 1: the 13.7% is the total drop, not the erosion.
  Only 44 of the 164 lost cells — 3.7% — are eroded; the other 120 are genuinely
  solid in the truth's own shifted geometry and are correctly excluded. The
  erosion is also **not** neutral. See the round-1 entry below; the design and
  the rejected alternative both stand.)*

- **WP1.3 — vertical alignment for the slabs is nearest-level, not
  interpolated (departure).** Post-averaging z-interpolation would need the
  bracketing truth levels carried through the whole streaming pass, i.e. the
  full 3-D truth moments, which is exactly the cost the streaming shape exists
  to avoid. The truth slab is therefore accumulated at the truth levels nearest
  the assimilation grid's, and the residual is reported as
  `averaging.z_offset_max` (**0.0 on all three real run dirs**) rather than
  hidden. The station columns have no such constraint — they are a handful of
  cells — so they keep the truth's full z axis and **are** z-interpolated after
  averaging.

- **WP1.3 — `hit_rate`'s `W` is a per-cell bootstrap reduced by *median*; the
  TKE / `⟨u′w′⟩` floors reduce by *RMS* (forced by the signature, not a free
  choice).** `hit_rate(allowance=...)` takes one scalar for the whole
  comparison, so the per-cell bootstrap has to be reduced: the median is the
  robust choice for a quantity whose per-cell values span a wake. The two stress
  floors sit beside RMSEs, and the RMS of the per-cell sampling errors is the
  RMSE a perfect model would still score, so RMS is the matching reduction.
  Both bootstrap per-timestep **integrands** (`_fluctuation_energy` and its new
  sibling `_fluctuation_covariance`, both Bessel-on-the-integrand) so that the
  statistic being resampled is a plain mean — the only shape
  `block_bootstrap_std_batch`'s row contract admits — and this reuses WP1.2's
  existing helper rather than introducing a second TKE definition.

- **WP1.3 — the truth pass runs *after* the ensemble pass (plan orders them 2
  then 3).** Reversed so the scored z-levels come off the **actual** colocated
  assimilation grid rather than a re-derived probe of it: `mean_field_scores`
  takes the filled accumulator and reads `z_levels` / `station_x` / `station_y`
  off it. The metric is unaffected; the ordering removes a way for the two
  passes to disagree about which levels were scored.

- **WP1.3 — keys emitted beyond the plan's schema sketch (additive).**
  `hit_rate.per_z` is a list of `{z, u, v, w, fac2_velmag}` mappings (the sketch
  says only `per_z: [...]`); `hit_rate.allowance` with its own `reason`, and
  `hit_rate.relative_tolerance`, so the two constants a `q` was computed with
  travel with it; `fb.velmag` and `nmse.velmag`, so the layer always carries a
  well-posed bias and NMSE even when every signed component takes the `nan`
  path above; `averaging` gains `n_time_truth` / `n_members` / `n_stations` /
  `stride` / `z_indices` / `truth_z_levels` / `z_offset_max` / `n_cells` /
  `n_scored_cells` / `n_bootstrap_cells`; plus the whole `masking` block,
  `eval_fields` (the artifact's file name) and `reason`. All of it is present as
  `null` on the degraded path too, so the key set is stable — the same contract
  `_null_statistic_block` set in WP1.2.

- **WP1.3 — `ensemble_velmag` and `ensemble_velmag_std` are different
  reductions, and both say so in their `long_name` (disambiguation).**
  `ensemble_velmag` is `|mean of the member means|` — the quantity actually
  scored, and what the plan's "ensemble-mean-of-means vs truth" names — while
  `ensemble_velmag_std` is the across-member std of each member's **own** `|U|`.
  They differ by the ensemble spread of the *direction*, so a consumer that
  assumed one was the moment of the other would be wrong by exactly the
  quantity F2 is drawn to show.

- **WP1.3 — the memory floor is stated rather than optimized away.** Because
  `colocate_components` refuses axis-subset input (the alignment guard above),
  every truth chunk is colocated at **full 3-D resolution** before the z-slabs
  are taken, so one frame is the smallest possible chunk: **~0.8 GB at
  512×512×128** for the velocity triple plus the interpolation's temporaries.
  That is the same order as the member `stream_window_members` already
  materialises, so it is the accepted shape of this stage rather than a
  regression — but it is a floor, not a knob. Documented at
  `_TRUTH_CHUNK_BYTES`; a truth grown past it needs colocation to gain a
  **level-wise mode**, not a smaller constant.

- **WP1.3 — validated on the three real run dirs, including that WP1.2's
  numbers did not move.** Byte-identity of the accumulator-produced
  `ensemble_series` (with the widened `variables`) against
  `ensemble_sensor_series`: **True on all three**, and `sensor_statistics`
  recomputed through the old dedicated-pass path is **identical** to what the
  shared pass produces.

  | run dir | `q` (u/v/w) | FAC2 `\|U\|` | NMSE `\|U\|` | masking | scored cells |
  |---|---|---|---|---|---|
  | `pylbm_to_pylbm` | .723/.149/.347 | 0.954 | 0.0282 | `blanking` both sides | 1036/1600 |
  | `pyudales_to_pyudales` | .944/.821/.798 | 0.999 | 0.0102 | **none** (warned) | 1600/1600 |
  | `pylbm_to_pyudales` (cross-grid) | .731/.167/.144 | 0.943 | 0.0400 | truth `blanking`, ensemble none | 1036/1600 |

  The sampling floors and the hit-rate allowance are `null` on all three (3
  frames against 20 blocks → `nan`, the sanctioned path), and are real numbers
  on the `M = 8` synthetic fixture (18 truth frames, 4 blocks) where a test
  asserts them `> 0`. Read the `pyudales_to_pyudales` row against its `masking:
  none` — 1600/1600 scored cells is the unmasked-backend gap, not a cleaner run.

- **WP1.3 — the runtime budget is overshot, and the plan's stated relief valve
  does not work (accepted and documented, per the lead's ruling).** Stage 2,
  best of 3–5 runs, on the three real smoke-shaped dirs (M = 2, 3 frames,
  20×20×4): `pylbm_to_pylbm` 39 → 83 ms (**2.12×**), `pyudales_to_pyudales`
  46 → 95 ms (**2.06×**), `pylbm_to_pyudales` 38 → 87 ms (**2.28×**), against
  the plan's "≤ ~2× the `basic` runtime". Attribution: standard-without-WP1.3 is
  already 1.08–1.21×, so this WP is 43–48% of the standard stage — shared pass
  +10.7–15.4 ms, truth pass and reduction +12.9–14.9 ms, **`eval_fields.nc`
  write +11.2–34.3 ms**. The plan's remedy is to drop smoke `n_z_slices` to 2
  via `_SMOKE_OVERRIDES`; **measured, that changes nothing** (2.07× / 2.10× /
  2.26×), because at 1600 cells the cost is entirely fixed — file opens, the
  netCDF write, xarray overhead — and not per cell. So the remedy is ineffective
  *for the stated reason*, which is why `_SMOKE_OVERRIDES` was left alone rather
  than given a knob that buys nothing. **Lead's ruling: accept and document.**
  The ratio was a proxy for "does this layer blow up the test suite"; 90 ms
  absolute answers that question, the overshoot is fixed cost at a shape 1600
  cells wide, and the per-cell scaling the budget was really about is not
  exercised at smoke scale at all.
  *(Superseded in review round 1, and the last sentence was closer to right than
  the ruling it was attached to: the benchmark measured this layer at the one
  shape where its dominant cost is **identically zero**, so "fixed cost" was an
  artefact of the measurement, not a property of the layer. The truth pass is a
  13.2 s operation at 4×128×128 × 36 frames and was capped in response — see the
  round-1 entry below for the measured scaling and the cap.)*

#### The sweep-pipeline port and the acceptance grep

- **WP1.3 — the sweep port was widened to `_truth_series` (scope change).** The
  task assigned only `_ensemble_series`, whose `.load()` is the invariant-#2
  violation. The truth path was deduped in the same pass, deleting `_open_truth`,
  `_sensor_components`, `_concat` and the `_X_COORDS` constant: `_open_truth` was
  **character-for-character** `_esmda_common.open_truth` — including the
  offset/slicing rules that are the contract with `truth_access.yaml` — and
  `_sensor_components` was `_sensor_component_timeseries` in dict form. Keeping
  both copies only created somewhere for the two stages to disagree about what a
  sensor series is. This half is **not** a `.load()` fix: that path was already
  window-at-a-time. Bit-identity is pinned by a dedicated test.

- **WP1.3 — new helper `_split_quantities(components)` (addition).** The one
  structural difference between the two copies: ESMDA carries
  `(component, ensemble, time, sensor)` on a `component` dim, while the sweep
  stage wants `{q: DataArray}` over `QUANTITIES = ("u", "v", "w", "vel")`. It
  uses `.sel(component=q, drop=True).rename(q)` — deliberately a **view** (see
  the layout entry below) — and builds `vel` with the verbatim old elementwise
  `sqrt(u² + v² + w²)`.

- **WP1.3 — `compute_sweep_metrics.py` gained the repo's standard `sys.path`
  shim (addition).** The file had none and the new cross-package import needs
  one; it matches `compute_esmda_metrics.py` and `visualize_ground_truth.py`
  exactly. Verified by running `--help` (exit 0), since an import-time shim is
  the kind of thing a unit test can miss.

- **WP1.3 — the memory-layout hazard is real at this call site too, and the
  view is load-bearing.** Measured by diffing every leaf of `metrics.yaml`: the
  shipped `.sel` view changes **0 of 92** leaves; forcing
  `np.ascontiguousarray` per quantity changes **16 of 92**, largest relative
  move **2.079e-15**, all of them sensor CRPS entries. So preserving the view of
  `_stack_window_members`' `(component, sensor, ensemble, time)` buffer is a
  requirement, not a defensive habit — sweep comparisons are cross-run, and
  `metrics_version` exists so historical numbers only move deliberately. One
  benign stride note, recorded so it is not mistaken for a difference: on a
  single-sensor fixture the view reports strides `(40, 8, 160)` where the old
  path reported `(40, 8, 8)` — numpy normalizes the stride of a size-1 axis;
  values, bytes and every reduction are identical.

- **WP1.3 — a claim the implementing agent retracted after measuring it (kept,
  because it is the house rule in action).** The first draft asserted that
  `_esmda_common.sensor_magnitude`'s `sqrt((x**2).sum("component"))` would
  re-lay-out the buffer and so could not be shared. Measured, it does **not** —
  0 of 92 leaves change, and the bytes *and* strides are identical on every
  fixture, because the reduction is over three elements in the same order. The
  actual and only reason the shared helper is not called is that its result
  inherits the stacked array's `name` (`"u"`), which the old `vel` did not carry
  and `DataArray.rename` cannot clear. The docstring records the measurement,
  not the guess.

- **WP1.3 — the no-`.load()` regression test whitelists receivers by name
  rather than banning `.load()` outright.** `OmegaConf.load` legitimately reads
  `config.yaml`, so a blanket ban would fail on correct code. The AST-based
  check keeps failing on `xr.open_dataset(...).load()` **however it is spelled**,
  which a "`.load()` on an `xr` call" rule would not, and it parses rather than
  greps so the modules' own docstring mentions of `.load()` do not trip it.

- **WP1.3 — the acceptance grep passes, with one known remaining hit that is
  deliberately not fixed.** `grep -rn --include='*.py' '\.load()' scripts/ src/
  libs/` (excluding vendored `palm_model_system/`, `LBM/`, `_tadpole/`) returns
  **26 hits, 0 in scope**: every hit under `scripts/esmda/` is prose in a
  docstring or comment, and the rest load a single member's solver output, one
  time frame, one z-plane, an already-reduced file, or a small parameter
  artifact. The exception is `scripts/filtering/_filtering_common.py:180`, where
  `ensemble_cycle_sensor_series` does `analyzed_states.load()` on
  `state_history.nc` — which **is** a `(cycle, ensemble, …)` full-ensemble state
  file. It is not a `windows/window_*_state.nc`, so it falls outside the letter
  of phase 1's acceptance criterion, and it is much smaller (one analyzed frame
  per cycle rather than a whole rollout). **Lead's ruling: out of scope for
  WP1.3** — it belongs to the EnKF/filtering pipeline. It is recorded in
  `docs/scripts_and_configs.md` beside the filtering pipeline so the next person
  running that grep is not misled into thinking the tree is clean of
  full-ensemble loads everywhere.

#### Not a deviation: pre-existing breakage found while validating

The three `.temp/` run dirs are **not processable as-is**, for reasons predating
this WP, and this is recorded because the validation numbers above were
therefore taken on scratchpad copies. `pylbm_to_pylbm/truth_access.yaml` points
`true_state_path` at a deleted pytest tmpdir, and **all three** saved configs
carry `validation_*_points` at y = 30/55 on a 20-unit domain, which makes
`build_sensor_sets` → `interpolate_dataarray_at_points` raise in the *sensor*
stage — before WP1.3's layer is reached at all. Both were repaired in scratchpad
copies only; the user's `.temp` is untouched (verified: `run_summary.yaml` mtime
unchanged, no `eval_fields.nc` written there).
*(Review round 1 unexpectedly rescued this: the sweep stage now degrades on a
`ValueError` from the sensor series instead of losing the run — see below.)*

#### Review round 1

An adversarial review returned **REQUEST CHANGES**. The entries above stand
except where a bullet below supersedes them (the erosion magnitude, and the
runtime budget). Common thread, and it is worth naming because it is now three
rounds running: **every round-1 blocker was a number reported next to a
description of a different sample.** `n_scored_cells` counted cells that were
never scored (B1); the sampling floors were bootstrapped over building interiors
while the scores they anchor were fluid-only (B2); and the fix itself very nearly
shipped a `fluid_fraction` that had quietly become a *scored* fraction, which is
why the rename below happened before anything merged. That is exactly WP1.2's
round-1 thread ("two references computed off a different sample than the number
they normalize") and WP1.1's round-2 thread pointed at samples instead of at
sample *sizes* — the recurring failure mode of this whole phase is a name that
describes one set attached to a number computed on another.

The reviewer also independently **confirmed** a great deal, none of which is
reopened: WP1.2's numbers are byte-identical (587 shared leaves, 0 differing,
with a 1-ULP `np.nextafter` null experiment proving the harness sensitive); the
keys are additive-only with `metrics_version` still 2 and all four
`scripts/figure_creation/` consumers resolving unchanged; graceful degradation
holds across nine degenerate shapes; and the Chan/Welford precision, the Bessel
consistency with WP1.2's TKE, the colocation-vs-by-index measurement, the `nmse`
`nan` regime and the `eval_fields.nc` round-trip all check out.
`metrics_version` stays **2** throughout: every WP1.3 key was introduced on this
unmerged branch, so nothing here changes an estimator's semantics on a key any
shipped run has persisted.

- **WP1.3 (review round 1) — one diverged member nulled the whole layer with no
  `reason`, no log line, and an `n_scored_cells` that counted cells nobody
  scored (blocker).** Reproduced on the wiring fixture at `M = 8` with member 0's
  `u/v/w` set to NaN in every window — a diverged CFD member is routine, not
  exotic. `MeanFieldAccumulator.result` took a plain `np.stack(values).mean(axis=0)`
  over members, so one NaN member made the ensemble mean NaN *everywhere*, while
  `scored` was built only from `blanking` and the truth's finiteness and knew
  nothing about it. Before: `hit_rate u/v/w = None/None/None`, `fac2 = None`,
  both RMSEs `None`, **`reason = None`** — a silent null block, in direct
  violation of invariant #3 — with `n_members: 8` and `n_scored_cells: 104` on a
  run where **zero** pairs were scored. The partial case was worse than the total
  one: a member blowing up over a sub-region silently removed those cells from
  every score while `n_scored_cells` kept counting them.
  Three separate changes, per the lead's ruling, because they are not one fix.
  (1) `result()` now drops members whose time-mean **slab** is non-finite
  anywhere and reports `averaging.n_members_scored` beside `n_members`, a
  non-null block `reason` naming the excluded members, a `logger.warning`, and an
  `n_members_scored` attr in `eval_fields.nc`; all members non-finite → `None` →
  null block plus reason. The exclusion is **whole-member and casewise**, the
  same rule `StreamingMoments` already applies in time and for the same reason:
  an ensemble mean and a `ddof=1` spread taken over *different member sets at
  neighbouring cells* are not a field, and the spread is what `eval_fields.nc`
  publishes. Finiteness is judged on the slab alone — a station column is
  legitimately NaN outside the domain or inside a building, which is a station's
  problem and never a member's. (2) `n_scored_cells` and every fraction that
  describes the *scores* now come from a new `_fluid_pairs` (one retained set,
  cells with a finite pair on both sides), so the reported sample **is** the
  scored sample by construction rather than by coincidence; one shared set rather
  than per-score dropping, because `nmse_split`'s identity and the per-component
  hit rates are only comparable if they are computed on the same cells and the
  summary reports one `n_scored_cells`. (3) Both the all-NaN and the
  partial-region cases are pinned by test. After: `reason = "1 of 8 ensemble
  members (0) have a non-finite time-mean field and were excluded…"`,
  `n_members: 8`, `n_members_scored: 7`, every score real.

- **WP1.3 (review round 1) — the sampling floors and the hit-rate allowance were
  bootstrapped over the *unmasked, unstrided* truth slab (blocker).**
  `truth_mean_field_stats` built its retained series from `_slab(..., stride=1)`
  with no fluid mask, and computed `fluid` three lines later without ever
  applying it. Measured: the allowance was **byte-identical with and without
  `blanking`** — which is precisely the regression a test now catches. On a truth
  with pylbm-like near-zero building interiors at 25% solid fraction:

  | | allowance `u` | `tke_floor` | `uw_floor` |
  |---|---|---|---|
  | shipped (all cells) | 0.04581 | 0.012873 | 0.010462 |
  | fluid cells only | 0.04990 | 0.014865 | 0.012080 |
  | ratio | **0.918** | **0.866** | **0.866** |

  **The direction is what makes this a blocker.** A low allowance is
  conservative for `q`, but floors 13.4% low are **anti-conservative for their
  stated purpose** — the docs say "an RMSE below its floor is not skill, it is
  the truth's window length", and an RMSE genuinely inside sampling noise was
  being reported as above the floor. The bias scales with the solid fraction,
  which near the ground at Barcelona is larger than this 25%. It was stride-blind
  too: at `mean_field_stride: 3` the run reported `n_cells: 16`,
  `n_scored_cells: 8`, `n_bootstrap_cells: 120` with an allowance identical to
  the stride-1 run.
  The series is now built from the truth's **fluid** cells at the scored
  **stride** (new `stride` argument on `truth_mean_field_stats`). Measured after:
  allowance `u/v/w` `0.11269 / 0.09605 / 0.10668` masked against
  `0.11284 / 0.10145 / 0.11218` unmasked; stride-3 allowance `0.09446` against
  stride-1 `0.11269`; `n_bootstrap_cells` 104 fluid against 120 unmasked.
  **The cell subsample was replaced as well**, on a hazard the reviewer flagged
  without measuring: `_bootstrap_cell_stride` strided a *raveled* `(zlev, y, x)`
  index, so at 4×512×512 / 360 frames the stride is 68 and `gcd(68, 512) = 4` —
  the retained cells sit on a sub-lattice hitting only `x ≡ 0 (mod 4)`, which on
  a regular building array can lock onto one geometric phase. `_bootstrap_cells`
  now takes a **seeded random draw** of the fluid cells instead: it costs
  nothing, cannot phase-lock, and the seed and realised count both travel
  (`averaging.bootstrap_seed`, `n_bootstrap_cells`, `n_bootstrap_cells_max`).
  **Known consequence, documented rather than fixed:** at `stride > 1`
  `n_bootstrap_cells` can *exceed* `n_scored_cells` (16 against 8 on the
  fixture), because the truth-side stride lands on the truth's own cells while
  the erosion happens on the assimilation grid. The floors are a property of the
  truth's sampling **at the same density**, not a cell-for-cell alignment; said
  so in `truth_mean_field_stats`.

- **WP1.3 (review round 1) — `blanking` was costing a second full
  ensemble-sized read, and the "arithmetic, not reads" claim was false at
  `standard`.** Instrumented `NetCDF4ArrayWrapper._getitem` on a 3-window /
  8-member run dir (one member's `u+v+w` = 2880 elements, a full ensemble =
  23,040):

  | | `window_0` elements | truth elements |
  |---|---|---|
  | `basic` | 23,071 (`u/v/w` = 23,040 exactly — one read per member) | 19,580 |
  | `standard` + `blanking` (before) | **30,782** (`blanking` = 7,680, another full ensemble) | **34,735** (+77%) |
  | `standard` + `blanking` (after) | **23,222** (`blanking` = 120, one frame) | 32,695 |

  §WP1.3 step 3's actual requirement **held all along** — `u/v/w` are read
  exactly once per member, so the materialisation design works and the truth pass
  is legitimately a separate file. What was false was the sentence attached to
  it. `blanking` is *static geometry* that `run_esmda` writes replicated over
  `(ensemble, time)` (confirmed on `.temp/pylbm_to_pylbm`:
  `blanking ('ensemble','time','z','y','x') float32`), and the code's own
  `_fluid_indicator` docstring already said "the geometry is static in every
  shipped case" — so at Barcelona scale with pylbm this was **+33% on the
  window-state bytes for one 3-D frame's worth of information**, plus
  `_record_mask` re-deriving and re-interpolating the indicator `M × W` times.
  The accumulator now takes `state_paths` and reads the mask from
  `isel(ensemble=0, time=0)` on its own open; `stream_window_members` is back to
  `u/v/w` only and `_state_read_variables` is deleted. The claim is now true
  rather than aspirational, and the comment says the measured thing.

- **WP1.3 (review round 1) — the cross-grid erosion figure was overstated ~3.7×,
  and the erosion is *not* neutral (documentation; the design stands).**
  Measured on the real `.temp/pylbm_to_pylbm` artifacts, scoring the retained set
  `R` and the dropped-but-fluid set `D` against the same unmasked-interp truth
  (ensemble-fluid 1200, truth-fluid by majority rule 1080, `R` = 1036, `D` = 44):

  | set | `q(u)` | `q(v)` | `q(w)` | FAC2 | NMSE(`\|U\|`) | mean `\|∇\|U\|\|` |
  |---|---|---|---|---|---|---|
  | `R` retained | 0.804 | 0.277 | 0.444 | 0.968 | 0.0128 | 2.323 |
  | `D` eroded | 0.682 | 0.136 | 0.205 | 1.000 | 0.0680 | 6.165 |

  (a) **The correction.** Only **44 of the 164 lost cells — 3.7% of the
  ensemble-fluid set — are lost to erosion**; the other 120 are genuinely solid
  in the truth's own half-cell-shifted geometry and are correctly excluded. The
  13.7% recorded above is the total `0.75 → 0.6475` drop, which is not all
  erosion. It was the lead's figure and it was wrong.
  (b) **The original worry was justified anyway.** `D` carries **2.65× the mean
  shear** of `R` and the run scores materially worse there, so dropping it is not
  a neutral thinning of the sample: it **flatters `q` by +0.005…+0.010 and
  understates NMSE(`|U|`) by 6%** at this shape, and the bias grows with
  perimeter/area — a dense Barcelona array has far more perimeter per fluid cell
  than this fixture, so the effect is *larger* on the case that matters.
  The design is unchanged and the **rejected alternative stays rejected**: a
  missing number still beats a wrong one, and interpolating field and mask
  separately would score those same high-shear cells partly against building
  interiors. What changed is that the corrected magnitude, the measured direction
  and the perimeter/area scaling are now in the log and in
  `docs/scripts_and_configs.md` instead of a single overstated percentage.

- **WP1.3 (review round 1) — station truth columns were silently empty at
  stations near buildings.** On the blanking fixture with stations defaulting to
  sensor x/y, `station_truth_tke` had finite values at **2 of 5** stations
  (`[4 0 4 0 0]`) while `station_ensemble_tke` was finite at all five
  (`[4 4 4 4 4]`) — drawn straight through the building. `_column_indicator`
  requires the interpolated indicator `>= 1 − 1e-9`, so a station whose 2×2
  horizontal stencil touches any solid cell loses its **entire** truth column;
  stations default to sensor x/y and urban sensors sit on facades and roofs, so
  this is the common case, not the corner. Nothing in `run_summary.yaml` said so
  — `masking` carries slab fractions only and `averaging.n_stations` counted all
  five — and WP1.4's S1 would have rendered three panels with a posterior band
  and no truth line, silently.
  **The strict all-fluid stencil rule is retained deliberately** (relaxing it
  would contaminate profiles with building values, which is the wrong trade); it
  is made *visible* instead: a `logger.warning` naming the stations and their
  coordinates, and `averaging.n_stations_with_truth` beside `n_stations`.
  The same finding surfaced a second gap: the file round-trips exactly and
  satisfies S1's layout, but with only mean and std persisted the "posterior
  band" could only ever be `mean ± kσ`, so an empirical 5–95 fan would have
  forced exactly the recomputation `eval_fields.nc` exists to prevent. Four
  `station_ensemble_*_quantile` variables (`STATION_QUANTILES` = 5/25/50/75/95,
  with a `quantile` coord) are now persisted — stations are a handful of columns,
  so the cost is trivial.

- **WP1.3 (review round 1) — the runtime budget measured nothing, because it was
  taken at the one shape where this layer's dominant cost is identically zero
  (supersedes the accepted-overshoot entry above).** `block_bootstrap_std_batch`
  returns all-`nan` immediately below 4 frames and the smoke truth has 3, so
  `_truth_sampling_floors` — the layer's expensive part — never ran at all in the
  benchmark. "`n_z_slices: 2` changes nothing" was itself the tell. Measured
  `_truth_sampling_floors`, 3 components, 20 blocks:

  | cells | frames | wall time |
  |---|---|---|
  | 1600 | 3 | 0.1 ms (the smoke shape: `nan` before any work) |
  | 1600 | 36 | 157.5 ms |
  | 4096 | 36 | 448 ms (the cap) |
  | 16384 | 36 | 1.9 s |
  | 65536 (4×128×128) | 36 | **13.2 s** |

  `_TRUTH_SERIES_MAX_BYTES` bounds *memory*, not time — at 4×128×128 the cell
  stride it implies is 1 — so the truth pass is a tens-of-seconds operation at
  production shapes and had never been exercised. Per the lead's ruling it was
  measured at one realistic shape and then **bounded explicitly**:
  `_TRUTH_BOOTSTRAP_MAX_CELLS = 4096`, in the same spirit as
  `_TRUTH_SERIES_MAX_BYTES` bounding memory. End to end at 4×128×128 × 36 frames,
  `M = 8`, 3 windows: **`basic` 0.35 s against `standard` 1.62 s**, where
  uncapped the truth pass alone would have added the 13.2 s.
  **The cap does not materially move what it bounds**, and that was measured
  rather than assumed: over 8 seeds at that shape, capped-against-full is rms
  **0.50%** (allowance), **0.39%** (tke), **0.64%** (uw), worst case 1.29% —
  against a floor that is an order-of-magnitude statement about the truth's
  window length. The realised count, the maximum and the seed all travel
  (`averaging.n_bootstrap_cells`, `n_bootstrap_cells_max`, `bootstrap_seed`), so
  the number is reproducible rather than merely stable.
  The same measurement corrected a second smoke-scale illusion: `eval_fields.nc`
  is ~11 kB at smoke but **9.1 MB at 4×128×128**, implying **~150 MB at
  4×512×512**. The file's size grows with the grid, the 11–34 ms write does not
  stay fixed, and the artifact table now says so.

- **WP1.3 (review round 1) — the accumulator memory table stopped 15× short of
  the plan's own target (documentation).** The read-side ruling is confirmed
  (exactly one read per member for `u/v/w`, within invariant #2's letter), but
  the arithmetic the docstring published stopped at the wrong shape:
  `StreamingMoments` slab accumulators are 10 arrays × 8 B × cells × M, i.e.
  **0.62 GB at `M = 32` / 4×256×256** — the largest number stated — against
  **10.00 GB at `M = 128` / 4×512×512**, which is the plan's actual Barcelona
  target, held for the whole pass on top of the materialised member and its
  float64 colocation temporaries (`colocate_components` promotes float32 →
  float64). The `M = 128` row is now in both `MeanFieldAccumulator`'s and
  `StreamingMoments`' tables so a reader reaches for `mean_field_stride` before
  discovering it the hard way.

- **WP1.3 (review round 1) — the extrapolated edge level always lands on a
  scored plane, and it is now excluded from the aggregates rather than
  documented at.** The earlier entry recorded a "2.5× variance amplification" in
  the last column; that was measured against the *unfiltered* variance, and the
  number that matters is against the **interior colocated** estimate:
  `(2.5 − 1.5ρ)/(0.5 + 0.5ρ)` = **5.06× at ρ = 0**, 2.33× at ρ = 0.5, 1.21× at
  ρ = 0.9 — so ~20% for a well-resolved field, not 5×, and the docstring figure
  was wrong in both directions depending on the field. The consequential half is
  that it is **not avoidable by luck**: `evenly_spaced_levels` uses
  `np.linspace(0, nz−1, k)`, which always includes `nz−1`, and for uDALES/PALM
  `w` that is exactly the extrapolated `zm→zt` level (verified on
  `.temp/pyudales_to_pyudales`: `zt = [1.25, 3.75, 6.25, 8.75]`,
  `zm = [0, 2.5, 5, 7.5]`, `evenly_spaced_levels(4, 4) = [0, 1, 2, 3]`). That is
  **1 of 4 scored z-levels** contaminating `hit_rate.per_z[-1]`, `tke_rmse` and
  `uw_stress_rmse`, and the docstring's own mitigation ("a caller that cares
  should drop the last index") was not applied by any caller.
  New public `turbulence_stats.extrapolated_centre_dims` reports which centre
  axes carry the edge (dims only, never values); `averaging.z_levels_extrapolated`
  and a per-row `extrapolated` flag in `hit_rate.per_z` report which levels were
  affected; and extrapolated levels are **excluded from every aggregate** (hit
  rate, FAC2, FB, NMSE, both RMSEs) while still being reported per z — so nothing
  is hidden and nothing is silently biased. `averaging.n_aggregate_cells` sits
  beside `n_scored_cells` for the two samples. The amplification figure is
  corrected in the docstring and pinned by a Monte-Carlo test.
  *(One claim in this entry was false and is corrected in review round 2: "the
  aggregate can never empty, because `linspace(0, n−1, k)` always includes index
  0" — at `nz == 1`, index 0 **is** `nz−1`, and the exclusion emptied every
  aggregate score with no `reason`. See the round-2 entry below. The exclusion
  itself, and its scope, stand.)*

- **WP1.3 (review round 1) — `masking` gained `scored_fraction` rather than
  letting `fluid_fraction` change meaning (schema freeze; the fix agent's
  proposed deviation is withdrawn).** B1's repair initially left
  `masking.fluid_fraction` meaning "the fraction that survived into the scored
  pairs" while its siblings `ensemble_fluid_fraction` / `truth_finite_fraction`
  still meant mask fractions — the round's own thread reappearing as a *naming*
  choice instead of as arithmetic. The lead's ruling was to split them, and since
  nothing has shipped the rename is free. Final: **`fluid_fraction`** is the
  *mask's* number (assimilation-side fluid ∧ truth-side resolvable), unchanged
  from before the fix and consistent with its siblings; **`scored_fraction`** is
  the fraction carrying a scoreable pair, i.e. the sample this block's numbers
  were computed on, equal to `n_scored_cells / n_cells` by construction and
  `≤ fluid_fraction`. The change is therefore **purely additive** — one new leaf
  — and the fix agent's proposed "deviation 1" (an existing key changing meaning)
  is **withdrawn**.
  Worth recording, because it looks like redundancy and is not: with B1's
  whole-member exclusion in place the two fractions are **equal on every
  currently reachable path** (measured 0.8667 on both a healthy run and a
  partial-divergence run). That is the correct outcome — the only remaining
  sources of non-finiteness are the mask and diverged members, and the latter are
  now removed member-wise. The two names exist so that a *future* source shows up
  as a divergence between them instead of one number standing in for the other,
  and the tests pin the `≤` relation and the `n_scored_cells` identity rather
  than equality, so they catch that case rather than lock it out.

- **WP1.3 (review round 1) — keys, names and signatures the fixes added (all
  additive or internal).** New keys: `averaging.{n_members_scored,
  n_stations_with_truth, z_levels_extrapolated, n_aggregate_cells,
  n_bootstrap_cells_max, bootstrap_seed}`, `hit_rate.per_z[*].extrapolated`,
  `masking.scored_fraction`. New public names:
  `turbulence_stats.extrapolated_centre_dims` and
  `_esmda_common.STATION_QUANTILES`. `eval_fields.nc` gains four
  `station_ensemble_*_quantile` variables, a `quantile` coord and an
  `n_members_scored` attr, **and its `truth_*` variables are now NaN outside the
  scored set** — so `isfinite(truth_velmag)` *is* the scored set while
  `fluid_mask` alone is strictly larger *(corrected in round 2: it is a superset,
  `≥` not `>`, measured equal on a matched-grid run and strictly larger only
  when the truth's mask or grid differs)*. That point is called out in
  `docs/data_assimilation.md` because it is a WP1.4 trap: a figure masking on
  `fluid_mask` and a metric scored on the pairs would disagree.
  Signature changes: `truth_mean_field_stats(..., stride=1, ...)`;
  `MeanFieldAccumulator(..., state_paths=None)` (falling back to reading the mask
  from the member when absent, so an in-memory caller still works);
  `_bootstrap_cell_stride` → `_bootstrap_cells`; `_state_read_variables` deleted.
  One behavioural note the lead endorsed: **`block["reason"]` can now be non-null
  on a block that still carries numbers** (a partial degradation, e.g. some
  members excluded). The key set is unchanged; partial degradation should be
  legible rather than either silent or fatal.

- **WP1.3 (review round 1) — the sweep stage: D5's stated blocker was false, and
  it is retracted in code rather than only in prose.** The earlier entry recorded
  that `sensor_magnitude` could not be shared because its result inherits the
  stacked array's `name` and "`DataArray.rename` cannot clear it". Measured on
  this env (xarray 2025.12.0): `sensor_magnitude(components).name == "u"` but
  `sensor_magnitude(components).rename().name is None` — the impossibility was
  invented. The sweep stage now calls the shared `sensor_magnitude` instead of
  duplicating the `|U|` formula, so there is one definition across both stages as
  the master plan intends; the docstring records the measured fact, and the
  "0 of 92 vs 16 of 92 leaves" layout table is retained, since that finding was
  real and independent of this one.

- **WP1.3 (review round 1) — `_split_quantities` is loudly three-component
  rather than silently so, and the sweep stage degrades instead of vanishing.**
  It hard-coded `("u","v","w")` and would have dropped a fourth component
  without a word; it now derives `QUANTITIES = _COMPONENTS + ("vel",)` from one
  constant and raises a `ValueError` naming the actual component set otherwise.
  **Loud was chosen over general deliberately** — `_Q_KEY`, the `metrics.yaml`
  key names and the three-term `|U|` sum are all per-component, so a "general"
  splitter would have moved the silent drop one layer down rather than removing
  it. Implementation note worth keeping: `components.coords.get("component")` on
  a dim with no coordinate returns a *virtual* `0..n−1` range in this xarray, so
  the guard tests `"component" in components.coords`.
  Separately, `ensemble_sensor_series`'s new `ValueError` (the behaviour
  narrowing logged above) would have become **no `metrics.yaml` at all** inside
  `compute_sweep_metrics.main()`'s per-run `except: continue` — unreachable via
  `run_esmda`, reachable for legacy files. `process_run` now wraps the three
  series calls in a targeted `except ValueError`, logs a warning naming the run
  and the cause, sets `status["note"]`, and still writes `metrics.yaml` with the
  parameter/state metrics and the `num_sensors` skeleton, degrading exactly like
  the existing missing-`truth_access` branch. `main()`'s per-run handler is
  unchanged in scope but now does `logger.exception(...)`, and `main()` calls
  `logging.basicConfig`. **Unplanned benefit:** this also rescues the `.temp`
  breakage recorded above — an out-of-domain sensor point raises `ValueError`, so
  such a run now yields a `metrics.yaml` plus a warning naming the cause instead
  of disappearing from the sweep.

- **WP1.3 (review round 1) — minors, fixed where cheap.**
  `resolve_metrics_settings` accepted `stations: [[5.0]]` and `_station_points`
  then raised an uncaught `IndexError` that killed the metrics stage; the station
  *shape* is now validated where the other `run.metrics` knobs are, same failure
  mode, same place — which is the WP1.0 round-1 rule ("a bad knob is reported
  cheaply instead of crashing deep inside a pass that has already read GBs")
  applied to the one knob that had escaped it. Configured stations *are* deduped
  despite the comment claiming they "win outright"; the comment now matches the
  code. Dead per-member work removed (the station `velmag` and the slab `rms`
  were computed for every member and never read; the slab `velmag` is kept
  because its across-member spread is published). Three stale
  `scripts/compute_sweep_metrics.py` / `compare_sweep_results.py` paths in the
  sweep module's own docstring corrected to `scripts/figure_creation/`.

- **WP1.3 (review round 1) — verification after the fixes.** **131 passed, 0
  failed** across the five targeted suites (`test_esmda_metrics_wiring` 20,
  `test_turbulence_stats` 37, `test_esmda_stream_members` 20,
  `test_esmda_metrics_levels` 21, `test_sensor_statistics` 33), plus
  `test_da_metrics` 20 on the sweep side. WP1.2 byte-identity re-verified with
  `blanking` both on and off: **587 shared leaves, 0 differing, 0 old-only**, 90
  new-only and all of them under `mean_field_metrics`, with an `np.nextafter`
  null experiment detecting exactly 1 differing leaf so the harness is proven
  sensitive. That last point is not ceremony — **both fix agents independently
  wrote, and caught, a false-negative null experiment** (`*= 1 + 1e-16` is a
  float no-op, and `reshape(-1)` on a non-contiguous view silently returns a
  copy), so a byte-identity claim made without a working null is worth nothing.
  Degradation: 11 shapes same-keys. A missing truth and a corrupt window file
  still abort `compute_metrics` — both **pre-existing at HEAD** in the earlier
  state/sensor stage, not WP1.3 regressions, and not fixed here. black/isort
  clean; no new mypy errors (the 13 reported under a 4-file invocation are
  pre-existing and verified identical against `HEAD`; each file alone
  type-checks clean).

- **WP1.3 (review round 1) — out of scope, reported not fixed.**
  `job_scripts/local/eval_sweep.sh:85`, `job_scripts/snellius/eval_sweep.slurm:66`
  and `job_scripts/delftblue/eval_sweep.slurm:69` all invoke
  `scripts/compute_sweep_metrics.py`, a path that no longer exists (the review
  found two; there are three, plus matching comment lines in each file and in
  `job_scripts/local/README.md`). Pre-existing breakage unrelated to this WP,
  deliberately left for its own one-line change rather than widening an already
  large diff. Recorded in `docs/scripts_and_configs.md` beside the script so the
  next person to run a sweep on HPC finds it.

#### Review round 2

Round-2 verdict: **APPROVE**. Every round-1 finding was verified closed by
independent re-measurement rather than by reading the transcript — B1, B2, S1,
S3 and A7 all confirmed, the masking rename confirmed honest-but-currently-
redundant, the `state_paths=None` fallback confirmed bit-identical, the
stride-corrected floor confirmed both the right scale *and* biased in the
conservative direction, and the sweep port confirmed byte-identical after the
`sensor_magnitude` swap. Standing invariants intact throughout (WP1.2 587 shared
leaves / 0 differing with `blanking` on and off, `metrics_version` 2, all four
`figure_creation/` consumers resolving, 11 degenerate shapes same-keys). The
reviewer classified the follow-ups as non-blocking; **the lead's ruling was to
fix them all in this PR anyway**, on the grounds that each is a one-line change
or a test and a follow-up list attached to a merged PR is how these become
permanent.

The round's character is different from round 1's, and the difference is the
lesson. Round 1 found *numbers* reported against the wrong sample. Round 2 found
**three false or stale claims in comments and docstrings** — "the aggregate can
never empty", "`fluid_mask` alone is strictly larger", and an illustrative cause
that B1 had already made unreachable — plus one branch with no CI coverage at
all. None of the three was a wrong number today; two of them were *load-bearing*
anyway, because **a comment asserting an invariant is what stops the next person
checking it**. That is precisely what happened at `nz == 1`: the "can never
empty" sentence had been written, reviewed and carried through a round-1 fix
without anyone testing the one shape where it is false. Where round 1's rule was
"the sample a number describes must be the sample it was computed on", round 2's
is "an invariant asserted in prose is a claim, and claims get tested".
`metrics_version` stays **2**: one new leaf, no estimator touched.

- **WP1.3 (review round 2) — A7's aggregate exclusion had no CI coverage; it now
  has the contrast pinned rather than the key set (the priority finding, a test
  gap and not a defect).** `test_esmda_metrics_wiring.py` asserted only that
  `z_levels_extrapolated` and `n_aggregate_cells` were *present*, and the wiring
  fixture is **pylbm**, where `z_edge_extrapolated` is always `False` — so the
  exclusion branch never executed in CI at all. B1, B2 and S3 each got a proper
  pin in round 1; A7's correctness rested on the reviewer's hand-check. Two new
  tests now drive a genuinely uDALES-staggered state (`u` on `xm`, `v` on `ym`,
  `w` on `zm`) through a real `MeanFieldAccumulator`, so the flag comes off the
  real `extrapolated_centre_dims` path rather than a fixture constant. The
  fixture is built so the *contrast* is the assertion: the truth matches the
  ensemble exactly on the three interior levels and is wrong only on the
  extrapolated one, giving aggregate `hit_rate.u` **1.0** with the exclusion on
  against **0.75** with it off, `tke_rmse` **0.0** against **5.0**, with
  `per_z u = [1.0, 1.0, 1.0, 0.0]` and `extrapolated = [F, F, F, T]`. Pinning
  the contrast rather than the key set is the point: a regression that silently
  stopped excluding would leave every key present and every value plausible.

- **WP1.3 (review round 2) — the two stress RMSEs were reported beside a cell
  count that overstates their sample; `averaging.n_stress_cells` added.** The
  round-1 thread surviving in exactly two numbers: `_fluid_pairs` builds the
  retained set from `mean` and `velmag` only, while `tke` and `⟨u′w′⟩` keep
  their own pairwise finiteness inside `_masked_rmse` — they are `ddof=1`
  moments, so they are `nan` wherever a cell's own sample count is below 2.
  Measured with member 2 losing all but one finite frame over a 3×3 region for
  the whole run (a blow-up confined to a sub-region): its slab *mean* stays
  finite so B1 correctly keeps the member, but the summary read
  `n_scored_cells: 104`, `n_aggregate_cells: 104`, `reason: None`,
  `n_members_scored: 8` while the two RMSEs were computed on **76** cells.
  **The finiteness design is deliberately unchanged** — folding `tke`/`uw` into
  the base sample would let a one-frame window empty the hit rate's sample too,
  which is the wrong trade and is disclosed in `_fluid_pairs`' docstring. Only
  the count was missing, and only the count was added: the RMSE values are
  byte-identical across the change (`tke 0.022994175913767668`,
  `uw 0.019789352691558938`). `n_stress_cells` comes from a new `_stress_pairs`
  and is the subset of `n_aggregate_cells`, not of `n_scored_cells`. One leaf
  rather than two is exact rather than a compromise here — `tke` and `⟨u′w′⟩`
  come from the same `StreamingMoments` per-cell count and the same mask, so
  they are `nan` in identical cells (measured equal on the divergence fixture)
  — and the helper intersects the two anyway, so the number can only ever
  understate, never overstate.

- **WP1.3 (review round 2) — "the exclusion cannot empty the aggregate" was
  false, and where it emptied it reproduced B1's exact signature one layer
  down.** `_aggregated_levels` dropped `z_index == nz−1`, and round 1's own
  entry argued the aggregate survives because `linspace(0, n−1, k)` always
  includes 0. **At `nz == 1`, index 0 *is* `nz−1`.** Reachable rather than
  hypothetical: `_check_face_centre_alignment` uses
  `n_check = min(face.size−1, centre.size)`, so a uDALES-staggered state with a
  2-level `zm` and a 1-level `zt` colocates cleanly. Built and pushed through
  `mean_field_summary`, it produced `hit u/v/w = None/None/None`, `fac2 = None`,
  `tke_rmse.value = None`, `n_aggregate_cells: 0`, **`reason: None`, no log
  line** — the null-without-a-reason that B1 was blocked for.
  The ruling offered two repairs and **the lead confirmed the first**: keep every
  level, and say so, rather than nulling. A contaminated number that announces
  its contamination beats a silent null, and nulling would have reintroduced
  B1's signature at a different address instead of removing it. Now:
  `hit 1.0/1.0/1.0`, `n_aggregate_cells: 9` (`== n_scored_cells`),
  `z_levels_extrapolated: [0.5]`, a `logger.warning`, and a `reason` stating that
  the only scored z-level is the extrapolated one, that it could not be excluded,
  and that the second moments carry that edge. `_aggregated_levels` now returns
  `(aggregated, extrapolated)` — **the grid fact is reported whatever the
  aggregate decides**, which is the structural half of the fix: the previous
  shape let one boolean stand for both "which levels are extrapolated" and
  "which levels are aggregated", and those are different questions. The false
  "can never" sentence is gone from the comment; it is the sentence that stopped
  anyone checking.

- **WP1.3 (review round 2) — a member NaN only *inside buildings* was excluded
  as a diverged member.** `_member_is_finite` tested the raw slab mean even
  though `self._fluid_slab` is populated by the time `result()` runs. Measured
  with NaN in member 3's `u/v/w` only where `blanking == 1`: `n_members_scored:
  7` and a "diverged member" `reason`, despite every *scored* cell of that
  member being finite; with all eight members doing it, the whole layer nulls and
  no `eval_fields.nc` is written. Not reachable today — no shipped backend
  NaN-marks solid cells, which is this WP's own headline finding — but **the
  WP1.3 plan text assumes exactly that convention**, so this was one pypalm
  change away from nulling the layer on the backend whose native output the plan
  expects. The test now masks first: `n_members_scored: 8`, `reason: None`. It
  fails loud rather than quiet either way, which is why it was a follow-up and
  not a blocker.

- **WP1.3 (review round 2) — three documentation corrections, and one scope
  statement.** (a) `_masking_block`'s docstring cited "a member that diverged
  over part of the domain" as the case where `fluid_fraction` and
  `scored_fraction` differ — B1 removes exactly that case member-wise, so the
  illustrative cause was unreachable; the names and the `≤` relation are
  unaffected and the stale example is gone. (b) `_eval_fields_dataset` said
  `fluid_mask` "alone is **strictly larger**" than `isfinite(truth_velmag)`.
  Measured **equal** (104 == 104) on the healthy matched-grid fixture, and
  strictly larger only when the truth's mask or grid differs (16 against 8 at
  stride 3). It is a **superset (`≥`)**, not a strict one; corrected here and in
  `docs/data_assimilation.md`. The WP1.4 hand-off warning is unaffected — a
  figure masking on `fluid_mask` still risks showing cells the metric never
  scored, and `≥` is precisely why "sometimes equal" is not something a consumer
  may rely on. (c) `extrapolated_centre_dims` returns all three centre axes
  (`('xt','yt','zt')` for uDALES) while the driver reads only the z entry, so
  the last x-row and y-column carry the same edge extrapolation and **remain in
  every aggregate**. A7's ruling was z-scoped and stays z-scoped: the docstring
  is now scoped to what the caller actually does and the x/y residual is stated
  as a **known limitation** rather than implied away. Widening the exclusion is
  a WP1.4-or-later decision and was deliberately not taken here. Also
  `extrapolated_centre_dims` is now in `turbulence_stats.__all__` (11 names) —
  it was public and imported by name while being absent from the export list.

- **WP1.3 (review round 2) — verification.** **133 passed, 0 failed** (wiring 22,
  up 2 for N7). WP1.2 byte-identity re-run: **587 shared leaves, 0 differing, 0
  old-only**, new-only 90 → **91** — exactly the one new leaf, confirmed against
  the key set rather than assumed from the count. 11 degenerate shapes
  same-keys. black/isort clean. mypy back to the pre-existing 13: the two new
  `no-any-return` errors the round introduced were **fixed properly rather than
  suppressed or left for the commit hook**, which is worth recording because a
  `# type: ignore` here would have been invisible in the diff and permanent in
  the file.
