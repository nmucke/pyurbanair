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

_(record here as they occur)_
