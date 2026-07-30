# Phase 1 — post-processing metrics and figures (no run-stage changes)

> Part of the ESMDA-evaluation effort. Master plan:
> [master_plan.md](master_plan.md). Rationale: §3, §4.1–4.2, §4.5, §7 of
> [../esmda_turbulence_evaluation.md](../esmda_turbulence_evaluation.md).
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
