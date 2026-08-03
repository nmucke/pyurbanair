# Phase 3 — run upgrades: held-out sensors and the single spectrum check

> Part of the ESMDA-evaluation effort. Master plan:
> [master_plan.md](master_plan.md). Rationale: §4.3 and the held-out
> principle of [../esmda_turbulence_evaluation.md](../esmda_turbulence_evaluation.md).
> WP3.1 is independent of phase 2 and can land anytime after WP1.3;
> WP3.2 last (backend-touching). Two PRs. Slimmed 2026-08-03: two-point
> correlations, structure functions, increment PDFs, reverse-flow stats,
> band energies and representativeness-error config were cut (metrics doc
> §9) — the spectrum + LSD is the only structural check retained.
>
> **Implementer: update the master_plan.md status table per WP; record
> deviations at the bottom of this file. WP3.2 touches backend output
> handling — read `docs/pylbm.md` first; default runs must stay
> byte-identical (no-op rule).**

## WP3.1 Validation (held-out) sensors for barcelona — size XS, do first

Everything downstream (S5 validation columns, D1 held-out rank histograms,
`sensor_statistics.validation`) activates automatically once the case
defines held-out sensors: `build_sensor_sets` (`_esmda_common.py:56`)
returns a `validation` set iff the config carries validation points
(`create_validation_points`, `hydra_helpers.py:188`), and every consumer
loops over sensor-set names.

- Copy the schema from `conf/case/xie_and_castro.yaml` (6 assimilation + 2
  validation sensors) into `conf/case/barcelona.yaml`.
- Placement: held-out sensors in regimes the assimilated ones don't cover —
  at least one in a wake/recirculation zone and one above roof level, not
  co-located with assimilation sensors.
- Add one validation sensor to the smoke overrides so CI exercises the
  branch. No code changes expected.
- PR note: metric comparisons against older barcelona runs must restrict
  to the assimilation set.

## WP3.2 High-rate probes + spectrum — size M, backend-touching

Blocking constraint (verify before coding): Welch spectra need ≳ 10³
samples per sensor; the default assim output cadence
(`output_frequency=10 s`, 300 s windows → 30 frames/window) is ~30× too
coarse. Hence two parts:

### (a) High-rate probe series

Mechanism reality check (this is why the earlier "sample in the Python
wrapper as fields stream through" idea is dead): the pylbm wrapper runs
the compiled Fortran binary as a single `subprocess.run`
(`libs/pylbm/src/pylbm/forward_model.py:479–527`) and Python sees nothing
until it exits; fields exist only as the snapshot files Fortran writes at
`output_frequency`. Without touching the pinned Fortran (out of bounds),
the only route to high-rate series is **dedicated probe re-runs**:

- New script `scripts/esmda/run_probe_series.py`: re-run the truth and the
  posterior members (pylbm only) over one chosen window with
  `output_frequency` lowered to ~1 s, extract the probe points from each
  snapshot file as it accumulates, write
  `windows/window_{w}_probes.nc` (`u,v,w (ensemble, time, sensor)`) and
  `truth_probes.nc`, then **delete the high-rate snapshot files** — the
  full-field disk cost (~GBs/member at Barcelona resolution) is transient.
  Probe points default to the union of assimilation + validation sensors.
- This is a rerun cost (one extra window of forward runs), not a solver
  change; the assimilation pipeline and its artifacts are untouched.
  Prior-member probe runs are optional — without them S4 shows truth vs
  posterior only.
- For pre-existing truth datasets that cannot be re-run, fall back to the
  truth's stored cadence and record it; the comparison then truncates to
  the common resolved band (and may be too coarse to be worth it — check
  before running the ensemble).
- Update `docs/job_scripts.md` / backend docs with the new invocation.

### (b) Spectrum metric + figure (post-processing)

In `evaluation.turbulence`; no-ops without `probes.nc`.

- `welch_spectrum(series, fs, nperseg)` — Hann, 50 % overlap, linear
  detrend; **one shared `(fs, nperseg)` for truth and every member**
  (`nperseg = n_truth//8`; if cadences differ, restrict to the common band
  instead of resampling). Spin-up excluded; truncate all comparisons at
  `f < f_Nyquist/4`.
- `log_spectral_distance(E_t, E_m)` — the one scalar:
  `LSD = √(mean_k [10·log₁₀(E_t/E_m)]²)`, truth vs posterior median,
  reported next to the truth self-distance floor (LSD between the two
  halves of the truth record). Emit
  `spectral_metrics: {lsd_posterior_median, lsd_truth_floor, f_cutoff}`.
- Figure **S4** (`evaluation.figures.plot_spectra`): premultiplied
  `f·E(f)/σ²`, log–log, truth vs posterior median + envelope vs prior
  envelope; short −2/3 guide segment not drawn through the data; dotted
  line at the cutoff.

### Tests

- Unit: Welch slope on synthetic colored noise; LSD = 0 on identical
  spectra.
- Integration: `run_probe_series.py` on the smoke shape → `probes.nc` dims
  correct and high-rate snapshots deleted afterwards; smoke-scale spectra
  are noise-dominated — assert key presence/finiteness only. S4 reviewed
  by eye on a real (non-smoke) pylbm run before merging.

## Acceptance

- WP3.1: validation panels/keys appear on a barcelona smoke variant.
- WP3.2: probe reruns leave the assimilation pipeline and its artifacts
  untouched; `spectral_metrics` + S4 on a real pylbm run; docs updated
  where configs/artifacts changed.

## Deviations

_(record here as they occur)_
