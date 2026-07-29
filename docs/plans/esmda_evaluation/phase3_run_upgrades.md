# Phase 3 — run-configuration upgrades (held-out sensors, spectra, two-point)

> Part of the ESMDA-evaluation effort. Master plan:
> [master_plan.md](master_plan.md). Rationale: §4.3–4.4 and the held-out
> principle of §1 in
> [../esmda_turbulence_evaluation.md](../esmda_turbulence_evaluation.md).
> WP3.1 is independent of phase 2; WP3.2 depends on phase 1 modules;
> WP3.3 is optional follow-up. Three PRs.
>
> **Implementer: update the master_plan.md status table per WP; record
> deviations at the bottom of this file. WP3.2 touches backend output
> handling — read the matching backend doc (`docs/pylbm.md` etc.) first, and
> respect the "no-op when a param is absent" rule: default runs must stay
> byte-identical.**

## WP3.1 Validation (held-out) sensors for barcelona  — size XS, do first

Everything downstream (S5 validation panels, D1 held-out rank histograms,
`sensor_statistics.validation`) activates automatically once the case
defines held-out sensors: `build_sensor_sets` (`_esmda_common.py:56`) returns
a `validation` set iff the config carries validation points
(`create_validation_points`, `hydra_helpers.py:188`), and every consumer
already loops over sensor-set names.

- Copy the schema from `conf/case/xie_and_castro.yaml` (it defines 6
  assimilation + 2 validation sensors) into `conf/case/barcelona.yaml`.
- Placement guidance (from the identifiability results once phase 2 is in,
  or heuristically until then): put held-out sensors in flow regimes the
  assimilated ones do not cover — at least one in a wake/recirculation zone
  and one above roof level, not co-located with assimilation sensors.
- No code changes expected; the PR is config + a smoke-config counterpart if
  the test case should exercise the validation path (recommended: add one
  validation sensor to the smoke overrides so CI covers the branch).
- Note in the PR: runs started after this change have different
  `run_info`/sensor metadata; metric comparisons against older barcelona
  runs must restrict to the assimilation set.

## WP3.2 Spectra + two-point layer — size L, backend-touching

### Blocking constraint (verify before coding)

Welch spectra need ≳ 10³ samples per sensor; the default assim output
cadence (`output_frequency=10 s`, 300 s windows → 30 frames/window) is ~30×
too coarse. Two sub-efforts, deliberately separable:

### (a) High-rate probe output (`probes.nc`)

Grid-free sensor-point sampling at (or near) solver cadence, per member and
for the truth run:

- Config: `model.probe_output` (new, optional) with
  `{points: [[x,y,z],...], every_n_steps: N}`. **Absent ⇒ exact current
  behavior** (the repo's no-op rule). Default the point list to the union of
  assimilation + validation sensors.
- Backends: implement for `pylbm` first (uniform grid; sample in the Python
  wrapper as fields stream through, or at solver-output granularity if
  sub-output sampling requires Fortran changes — if so, record the achievable
  cadence and stop there; do not modify the pinned LBM Fortran). Then assess
  `pyudales`/`palm` (both have native probe/timeseries facilities —
  investigate `fielddump`/timeseries namelists before writing any Python
  resampling).
- Artifact: `windows/window_{w}_probes.nc` — `u,v,w (ensemble, time, sensor)`
  at the probe cadence — plus `truth_probes.nc` for an inline-simulated
  truth (for pre-existing truth datasets, fall back to the truth's stored
  cadence and record it; the spectra comparison must then truncate to the
  common resolved frequency band).
- Update `docs/job_scripts.md`/backend docs if run invocations change.

### (b) Spectral + two-point metrics and figures (post-processing)

Extends `turbulence_stats.py`; runs at `run.metrics.level: full`, no-ops
without the needed inputs.

- `welch_spectrum(series, fs, nperseg)` — Hann, 50 % overlap, linear
  detrend; **one shared `(fs, nperseg)` for truth and every member**
  (choose `nperseg = n_truth//8` and force the ensemble to match; if
  cadences differ, restrict to the common band instead of resampling).
- `log_spectral_distance(E_t, E_m)` and `band_energy_errors` over three
  bands (energy-containing / inertial / near-cutoff); truncate all
  comparisons at `f < f_Nyquist/4` (grid/filter roll-off is numerics).
  Inertial slope fitted on truth and members alike — report both, never
  score against the theoretical −5/3.
- Two-point correlation `B_uu(r)` via FFT (Wiener–Khinchin) along horizontal
  directions at 2–4 z-levels from full-field snapshots. Snapshot need is
  modest (≥5 snapshots ≥1 eddy-turnover apart) — check whether the stored
  window cadence already satisfies this before adding any output; it likely
  does. Cross-grid runs: compute on each grid separately (correlations are
  grid-relative), compare only the scalar reductions.
- `integral_length_scale` (integrate to first zero crossing; assert
  `B(L_domain/2) < 0.1` else flag periodic-domain contamination) and the
  headline `L_int_ratio = L_int_member/L_int_truth`.
- `s3_ratio`: third-order longitudinal structure function via array
  shifts, member/truth ratio over `4Δ < r < L_int`; increment PDFs `δu(r)`
  at 2–3 separations feed a figure panel.
- `reverse_flow_stats`: volume/centroid/extent of `⟨u⟩ < 0` from the
  WP1.3 `eval_fields.nc` means — **no new inputs; may land with phase 1 if
  convenient**.
- Figures: S4 spectra overlay (premultiplied, log–log, envelope + truth,
  −2/3 guide segment, cutoff line — conventions in metrics doc §7) and a
  two-point/`L_int` panel.
- Schema additions: `spectral_metrics: {lsd, band_errors, slope_truth,
  slope_ensemble_median, f_cutoff}`,
  `structure_metrics: {l_int_ratio, s3_ratio_median,
  reverse_flow: {volume_ratio, centroid_offset}}`.

### Tests

- Unit: Welch on synthetic colored noise with known slope; LSD zero on
  identical spectra; `B(r)` and `L_int` on a synthetic field with prescribed
  correlation length; `S₃` sign on a skewed synthetic increment field;
  reverse-flow stats on a constructed mean field.
- Integration: smoke run with probes enabled → `probes.nc` dims correct;
  with probes absent → artifact set unchanged (no-op rule);
  `level: full` on smoke → schema keys present (spectra will be
  noise-dominated at smoke scale — assert presence/finiteness only, not
  values).

## WP3.3 Optional follow-ups

- **Prior-state runs**: no code — document the
  `run.save_prior_state=true` override (needed for F1/F2 prior columns and
  prior-vs-posterior state spread figures) and its disk cost in
  `docs/data_assimilation.md`.
- **Representativeness error in `C_D`**: new
  `esmda.obs_error_representativeness: null | auto | <float>` —
  `null` (default) = today's behavior; `<float>` = added in quadrature to
  `obs_error_std`; `auto` = block-bootstrap point-sensor variability from
  the truth window (helper already in `turbulence_stats.py`). Without this,
  the χ²-type targets of phase 2 (data mismatch, innovations) sit off-target
  for reasons unrelated to assimilation quality — which is why phase 2
  emits `caveat: no_representativeness_error` in those blocks and treats
  its quality flags as advisory. When this WP lands and a run uses a
  non-null setting, drop the caveat from the emitted YAML; the flags then
  become authoritative.

## Acceptance

- WP3.1: validation panels/keys appear on a barcelona smoke variant.
- WP3.2: pylbm probe path lands with the no-op guarantee demonstrated;
  spectra/two-point metrics + S4 figure on a real (non-smoke) pylbm run
  reviewed by eye before merging (smoke-scale turbulence is not a meaningful
  visual check).
- Docs updated wherever configs/artifacts changed.

## Deviations

_(record here as they occur)_
