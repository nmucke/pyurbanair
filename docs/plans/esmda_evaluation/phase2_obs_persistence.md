# Phase 2 — persist observation-space arrays + the data-mismatch diagnostic

> Part of the ESMDA-evaluation effort. Master plan:
> [master_plan.md](master_plan.md). Rationale: §5 of
> [../esmda_turbulence_evaluation.md](../esmda_turbulence_evaluation.md).
> Requires WP1.1. Two PRs (WP2.1 persistence; WP2.2 diagnostic + figure).
> Slimmed 2026-08-03: the only diagnostic built on these arrays is the
> normalized data mismatch `O_N`; innovations, contraction-vs-achievable,
> SNR/DFS, variogram and obs-space rank/spread–skill were cut (metrics doc
> §9) — the persisted arrays still make them possible later if a run ever
> needs debugging.
>
> **Implementer: update the master_plan.md status table per WP; record
> deviations at the bottom of this file. WP2.1 changes stage-1 artifacts —
> update `docs/data_assimilation.md` in the same PR. Follow the master
> plan's Implementation process: Opus 5 agent team, tests in the same PR,
> two adversarial review rounds, CI green before merge.**

## Context: what exists and where (anchors as of 2026-07-29 — verify)

- Observations are built per window in `scripts/esmda/run_esmda.py:683–691`
  (`window_obs = truth_obs_op(window_true_state)` + `√C_D·noise`) and
  passed to the smoother; never saved.
- Predicted observations `pred_obs` (`(N_d, N_e)` after the transpose) are
  materialized inside each `_one_step` in
  `libs/data-assimilation/src/data_assimilation/smoothing/esmda.py`
  (`ParameterESMDA` ~399, `StateAndParameterESMDA` ~827,
  `StateAndTimeVaryingParameterESMDA` ~946; check whether
  `TimeVaryingParameterESMDA` ~512 delegates). Instrument wherever
  `pred_obs` is actually materialized, once per iteration. The
  `_final_time_smoothing_step` pred_obs (~888) is a different operation —
  **exclude it**.
- Step 0's `pred_obs` come from the forecast under the prior parameters —
  they are the prior predicted observations.
- Per-iteration parameter ensembles are already returned
  (`return_params_history=True`, run_esmda.py:707–726) and discarded via
  `.isel(esmda_step=-1)` — saving them is one line.
- Precedent for the plumbing: the filtering side's `CycleDiagnostics`
  (`libs/data-assimilation/.../filtering/base.py:56–92, 634–680`).

## WP2.1 Persistence

Config: `esmda.save_obs_diagnostics: true` in `conf/run_esmda.yaml`
(`false` reproduces the pre-phase-2 artifact set byte-identically; files
are KB-scale so default-on is fine).

Smoother change (minimal, no return-signature changes): in
`_BaseESMDA.__init__`, `self.collect_obs_diagnostics = False` and
`self.pred_obs_history: list = []`; reset the list at `_analysis` entry; in
each `_one_step` that materializes `pred_obs`, append
`np.asarray(pred_obs)` when the flag is set (asarray detaches from
JAX/device memory). After the final posterior forecast and **before**
`_final_time_smoothing_step`, compute the posterior forecast's predicted
obs with the same `_observation_step` call the `_one_step` implementations
use and append it — the history then holds `num_steps + 1` entries, entry
0 = prior, entry −1 = posterior forecast (note in file attrs that with
`final_time_smoothing` active the last entry is pre-smoothing). Same
attribute-plumbing pattern as `esmda.prune_disk_steps`.

Runner change (window loop, after ~726), gated on the flag; per window `w`
into `windows/`:

1. `window_{w}_obs.nc` — `obs`, `obs_clean` (pre-noise, line 686),
   `obs_error_std` (from `C_D` diagonal), with whatever sensor/component/
   interval metadata the observation operator exposes as coordinates (at
   minimum the flat obs index; document the ordering in attrs).
2. `window_{w}_pred_obs.nc` — `pred_obs (esmda_step, obs, ensemble)` from
   `pred_obs_history`.
3. `window_{w}_params_steps.nc` — `result_params` saved before the
   `.isel(esmda_step=-1)` (one line; kept for future debugging, no
   diagnostic builds on it in this plan).

Record the flag in `run_info.yaml`; update the artifact table in
`docs/data_assimilation.md` and the layout comment atop `run_esmda.py`.

## WP2.2 Normalized data mismatch + figure D3

In `compute_esmda_metrics.py` via `evaluation.scores`; no-ops with a log
line when the phase-2 files are absent. With `d = obs`,
`g_l (N_d, M)` = pred_obs at step `l`, un-inflated `σ_D`:

- Per member and step: `O_N = mean_j[((d_j − g_j)/σ_D,j)²]/2`. Emit
  per-step `{median, iqr, min}` across members, the target band
  `0.5 ± 3/√(2N_d)`, and advisory flags `underfit_final` /
  `overfit_final` / `collapsed` (across-member IQR → 0 while the median is
  off-target). Caveat, emitted in the block itself: the χ² target assumes
  `C_D` includes representativeness error, which it currently does not —
  the flags are advisory, the trend and the member spread are the signal.

```yaml
esmda_diagnostics:
  data_mismatch: {per_step_median: [...], per_step_iqr: [...],
                  target: 0.5, target_band: ..., overfit_final: false,
                  underfit_final: false, collapsed: false,
                  caveat: no_representativeness_error}
```

- Figure **D3** (`evaluation.figures.plot_data_mismatch_decay`): per-member
  `O_N` boxes vs iteration (0 = prior), horizontal target band at ½, log-y
  when the drop spans decades, caveat annotated.

## Tests

- Unit: `O_N` on a linear-Gaussian toy (posterior samples → mean ≈ ½).
- Integration: smoke run, flag on → files exist with
  `esmda_step == num_steps + 1`; flag off → artifact set identical to
  pre-phase-2 (directory-listing comparison); stage 2 on a phase-1-era run
  dir → no `esmda_diagnostics` key, no crash.

## Acceptance

- Flag-off byte-compatibility demonstrated; D3 renders on a smoke run;
  schema exactly as above (additive); docs updated.

## Deviations

Recorded during the WP2.1 + WP2.2 implementation (2026-08-07). Both WPs
landed as **one PR**, not two: WP2.2 is ~150 lines that only exercise WP2.1's
files, and splitting them would have merged a persistence format with no
reader.

1. **The observation dimension is `obs_index`, not `obs`.** The plan asks for
   "at minimum the flat obs index" on a dimension whose observation variable is
   also called `obs`. xarray accepts that in memory, but a variable whose name
   equals its dimension is silently promoted to an *index coordinate* on the
   netCDF round-trip — so a file written with an `obs` dimension reads back with
   `obs` as a coordinate rather than the data variable the docs describe.
   Naming the dimension `obs_index` avoids the promotion and leaves `obs`,
   `obs_clean` and `obs_error_std` as plain data variables. The `obs_sensor` /
   `obs_state` / `obs_interval` labels are as planned, and the flattening order
   is in the file attrs.
   (An earlier revision claimed xarray *refuses* the collision. It does not —
   it refuses only a name given as both a data variable and a coordinate in the
   same constructor call, which is a different thing.)
2. **`data_mismatch_summary` emits three keys beyond the sketched schema**:
   `per_step_min` (the plan's prose asks for `min`, its YAML sketch omits it),
   `num_observations` (what the band was computed from) and `final_step_index`
   (which iteration the `*_final` flags actually describe — a run whose
   posterior forecast failed for every member falls back to an earlier one, and
   nothing else in the block would reveal that). Additive within a new block, so
   invariant 1 is unaffected.
3. **The three flags are `None`, not `False`, when unjudgeable** — no
   observations means no band, and a `False` there would read as "checked, and
   fine". `collapsed` fires only when a vanishing IQR is paired with an
   off-target median (identical members *on* target are converged, not
   collapsed) and abstains below 8 pooled values: at the smoke shape's `M = 2`
   the "quartiles" are just the two members scaled, so every CI run would
   otherwise publish `collapsed: true`. That follows the master plan's
   cross-cutting caution to guard the smoke shape with `null` + a log line
   rather than a special case.
4. **The bundle loader lives in `scripts/esmda/_esmda_common.py`**
   (`obs_diagnostics_bundle`), mirroring `probe_spectra_bundle`, and both the
   metric and figure stages call it. The alternative — the metric stage writing
   raw per-member `O_N` values into `run_summary.yaml` for D3 to read back —
   would have put `W·L·M` floats in the summary for no gain; this way the boxes
   and the YAML come off one reduction.
5. **D3 draws per-window boxes rather than pooling the windows.** Window 0's
   prior is a cold-start draw and a later window's is an extrapolated
   posterior, so one pooled step-0 box would conflate two different objects.
   The `run_summary.yaml` block still pools (the plan's schema is per-step, not
   per-window-per-step); the bundle carries both. D3 takes `per_window` and
   `num_observations` as plain arguments rather than the bundle dict, so the
   leaf library does not bind itself to a script's key names (invariant 5).
6. **The `esmda_diagnostics` block is computed above the `skip_viz` gate**, on
   the same reasoning as WP3's `spectral_metrics`: that flag exists to avoid
   reading the multi-GB truth, and this block reads only the KB-scale
   observation-space files.
7. **`_BaseESMDA._results_dir_or_none()` was extracted** while instrumenting
   the three `_one_step` sites — the `results_dir if save_on_disk else None`
   expression already existed verbatim in three places and the new
   posterior-forecast call site would have been a fourth.

8. **`obs_diagnostics_bundle` is bounded by `truth_access.yaml`'s
   `num_windows`**, like every other consumer in `compute_esmda_metrics`.
   `paths.results_dir` is a fixed, non-timestamped path and the window loop
   never clears `windows/`, so a rerun with fewer windows — the normal case
   under this repo's "retune between live runs" workflow — leaves the earlier
   run's files in place. Globbing alone would pool them into this run's
   diagnostic while every other block in the same `run_summary.yaml` covered
   only the current windows. A window whose `N_d` differs from the first one
   kept is also dropped, since the target band cannot be shared across them.
9. **`collapsed` is judged per window, not on the pooled `per_step`.** It is an
   *across-member* verdict; pooling a rollout's windows measures the drift of
   `O_N` from window to window, which is not a spread. `data_mismatch_summary`
   therefore takes an optional `per_window`, and reports `collapsed: null`
   without it rather than a pooled approximation. This also keeps the
   `_COLLAPSE_MIN_VALUES` guard honest: pooling would sail a 2-member smoke run
   past a count threshold once it had four windows.
10. **The D3 band is drawn at its true position rather than clipped to the
   data.** For every `N_d >= 19` the lower edge `1/2 - 3/sqrt(2 N_d)` is
   positive and needs no floor at all, so the axis simply grows to include the
   band — which is what a reader of an off-target run needs, and removes the
   inversion at its source rather than clamping around it. Only at `N_d <= 18`
   (`3/sqrt(2*18)` is exactly `1/2`) is a floor needed, and only on a log axis.
11. **The bundle carries `window_indices`** so D3's legend names the actual
   windows: a run that lost window 1 to a read error would otherwise present
   windows 0 and 2 as though they were 0 and 1.

Two acceptance-criterion notes, recorded rather than fixed:

- The plan asks for flag-off **byte-compatibility** against the *pre-phase-2*
  artifact set. The test compares flag-on against flag-off within this branch,
  which cannot catch a file added or dropped in both modes; a true pre/post
  comparison would need a fixture built from the parent commit. Also, flag-off
  is not literally byte-identical — `run_info.yaml` gains the
  `save_obs_diagnostics` key unconditionally, which the plan itself asks for.
- `window_{w}_params_steps.nc` records a failed member's *un-substituted*
  parameters, while its posterior `pred_obs` column holds the donor clone that
  `apply_failure_substitutions_to_params` wrote. The final forecast is not
  followed by that substitution (pre-existing, unchanged here), so the two
  sides of the debugging artifact disagree for failed members. `O_N` is
  unaffected — it reads `pred_obs` only.

Not done, and deliberately: nothing reads `window_{w}_params_steps.nc`. The
plan says so explicitly ("kept for future debugging, no diagnostic builds on
it in this plan").

**Branch review round 3 (2026-08-07, `be1433e`).** Not a WP review: an
eight-agent adversarial sweep over the *whole* `esmda-evaluation` branch after
every WP had merged, which confirmed the plan is fully implemented and found six
blockers. One of them is phase 2's; the rest are recorded in the phase-1 and
phase-3 plans and in the master plan's filtering section.

12. **`obs_diagnostics_bundle` reads the run's own `save_obs_diagnostics` flag,
   not just the files.** Deviation 8's window-count bound catches a rerun with
   *fewer* windows; it cannot catch one that merely turned the flag off — which
   `conf/run_esmda.yaml` explicitly offers as the way to reproduce the
   pre-phase-2 artifact set — because the leftover `windows/*_obs.nc` then still
   match this run's window count exactly. On such a rerun the previous
   assimilation's data mismatch was republished into this run's
   `run_summary.yaml` and D3 was redrawn from it, beside this run's own
   parameter and state metrics. The flag is now read from `run_info.yaml` and a
   leftover set is skipped with a warning. An **absent** flag (a run dir written
   before WP2.1) means *unknown*, not false, and falls through to the files, so
   invariant 3 is intact. The guard lives in the bundle loader, which is why one
   change covers both the metric and the figure stage.

**Branch review round 4 (2026-08-07, `a843f00`).** A four-agent review of round
3's own fixes. Phase 2's fix was verified sound as landed and is unchanged; the
round's findings fall in phase 1 and in the master plan's filtering section.
Round 4 did document the `save_obs_diagnostics` flag-versus-files rule in
`docs/scripts_and_configs.md`, so the deviation above is now stated where a
reader of the artifact docs will meet it and not only here.
