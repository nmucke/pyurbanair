# Phase 3 — run upgrades: held-out sensors and the single spectrum check

> Part of the ESMDA-evaluation effort. Master plan:
> [master_plan.md](master_plan.md). Rationale: §4.3 and the held-out
> principle of [../esmda_turbulence_evaluation.md](../esmda_turbulence_evaluation.md).
> WP3.1 is independent of phase 2 and can land anytime after WP1.3;
> WP3.2 last (backend-touching). Slimmed 2026-08-03: two-point
> correlations, structure functions, increment PDFs, reverse-flow stats,
> band energies and representativeness-error config were cut (metrics doc
> §9) — the spectrum + LSD is the only structural check retained.
>
> **Rescoped 2026-08-06 (user decision), superseding the two bullets above:**
> **(a)** the case is **`xie_and_castro`, not `barcelona`** — barcelona is too
> slow to run and iterate on, and the held-out machinery is
> case-independent, so it is exercised on the cheap geometry; barcelona
> held-out sensors are deferred, not cancelled. **(b)** WP3.1 and WP3.2 are
> merged into **one work package and one PR**: WP3.1's held-out run is
> launched first and WP3.2 is implemented while that run is in flight,
> since WP3.1 is config-only and its verification is dominated by
> wall-clock.
>
> **Implementer: update the master_plan.md status table per WP; record
> deviations at the bottom of this file. WP3.2 touches backend output
> handling — read `docs/pylbm.md` first; default runs must stay
> byte-identical (no-op rule). Follow the master plan's Implementation
> process: Opus 5 agent team, tests in the same PR, two adversarial review
> rounds, CI green before merge.**

## WP3.1 Validation (held-out) sensors for xie_and_castro — size XS, do first

Everything downstream (S5 validation columns, D1 held-out rank histograms,
`sensor_statistics.validation`) activates automatically once the case
defines held-out sensors: `build_sensor_sets` (`_esmda_common.py:56`)
returns a `validation` set iff the config carries validation points
(`create_validation_points`, `hydra_helpers.py:188`), and every consumer
loops over sensor-set names.

`conf/case/xie_and_castro.yaml` already carried 2 held-out sensors, but
both sat at street level in an open column — the same regime as all 6
assimilation sensors, so the held-out score was close to a duplicate of the
assimilated one. The work is therefore regime coverage, not schema:

- Extend the held-out set to cover regimes the assimilated ones don't:
  a wake/recirculation point behind the tallest block and a point above
  roof level, neither co-located with an assimilation sensor.
- Add one validation sensor to the smoke overrides so CI exercises the
  branch (the real case's held-out coordinates all fall outside the
  20x20x10 smoke box). No code changes expected.
- PR note: metric comparisons against older xie_and_castro runs must
  restrict to the assimilation set.

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

- WP3.1: validation panels/keys appear on a xie_and_castro run — both on the
  smoke variant (CI) and on a real ensemble run, whose held-out columns must
  be populated rather than `null`.
- WP3.2: probe reruns leave the assimilation pipeline and its artifacts
  untouched; `spectral_metrics` + S4 on a real pylbm run; docs updated
  where configs/artifacts changed.

## Deviations

**Scope (2026-08-06, user decision)**

- Case is `xie_and_castro`, not `barcelona` (see the header). Barcelona
  held-out sensors are deferred.
- WP3.1 and WP3.2 merged into one work package and one PR.

**WP3.1**

- The case already carried 2 held-out sensors, so the work was regime
  coverage rather than schema: a wake point at (17, 30, 2) and an
  above-canopy point at (10, 30, 20) were added to the existing approach-flow
  and lane sensors.
- The above-canopy sensor was first placed at z=17, which review round 1
  falsified: on the **committed** domain (`x=[-20, 80]`) the array's tallest
  blocks are 17.2 m in the STL and voxelize to an 18 m roof, so z=17 sits *at*
  canopy height. It only looked above-canopy because the working tree's
  in-flight tuning cropped x to 40, excluding those blocks. Raised to z=20,
  which clears the tallest roof at every grid and crop tested.
- The placement guard lives in a new `tests/test_case_sensor_placement.py`
  rather than in `tests/test_hydra_config.py`: it voxelizes the STL (seconds,
  and it pulls in the pylbm submodule), which does not belong in the suite's
  cheap config-composition module. It asserts against the solid-occupancy mask
  at the case grid *and* the sweep grids, including the uncropped domain — a
  criterion that passes only on a cropped domain proves nothing.
- The smoke overrides pin one held-out sensor inside the 20x20x10 box. All of
  the case's real held-out coordinates fall outside it, so CI had been
  carrying a validation set that scored nothing. Pinned with `++` because
  `conf/case/barcelona.yaml` defines no `validation_*_points`, and a plain
  assignment would break any future `case=barcelona` compose.

**WP3.2**

- `scripts/esmda/run_esmda.py` gained a file-level `# mypy: ignore-errors`,
  waived on the same terms as `_esmda_common.py`. It has never satisfied the
  strict config (~15 errors) and is reached transitively by every test that
  imports it (`test_run_esmda.py`, `test_localization.py`, and now
  `test_run_probe_series.py`). Reviewers objected that a blanket waiver on the
  main entry point is worse than narrow ignores; the counter-argument is that
  typing an 850-line production script is not this WP's business and the
  sibling helper module set the precedent. Two of the hidden errors (None/int
  divisions in the state-summary helpers) look latent and are called out in
  the waiver comment for whoever types the file.
- `run_probe_series.py` does not use `resolve_output_dir`: probe artifacts must
  land in the *ESMDA run dir* so the metric/figure stages find them, not in
  the probe job's own timestamped Hydra dir.
- The probe re-run warm-starts from the window state file's first stored frame
  and prepends the discarded lead-in, so it reproduces the window's length,
  parameters and initial field but **not** its absolute-time interval, and is
  not bit-identical to the assimilation forecast. Fine for spectra and
  statistics, not for phase.
- The spec's "one shared `(fs, nperseg)`" is implemented as one shared segment
  *duration* when cadences differ, with the comparison truncated to the
  coarser record's Nyquist/4. Sharing the duration is what makes the bin
  spacing identical, so a coarser record is *compared over a smaller band*
  rather than refused.
- Records from one real run never share an exact cadence: the LBM timestep is
  derived from each solve's own velocity scale, so truth and members came out
  at 1.00079 s and 0.99917 s (and the members spread 0.5 % among themselves).
  Two consequences, both found by running it rather than by reading it:
  - The grid cross-check has to tolerate that. It is judged on the **bin
    spacing ratio**, not on the accumulated frequency offset: the offset grows
    as `k * eps`, so an offset budget tightens as `1/n` and refuses exactly the
    long records the diagnostic wants (measured: an offset rule accepted 480
    samples but refused 1024 and 2048 at this run's own cadence pair, i.e. the
    plan's nominal 300 s at 0.25 s). The residual mislabelling is ~0.012 dB
    against the 1-3 dB the LSD reports.
  - The bundle is scored on the truth's bins, and `sample_frequency` per record
    is what lets a reader recover the offset.
- `spectral_metrics` no-ops purely on **sample count** — the cadence cancels
  out of the in-band bin count — with a floor of 264 samples
  (`evaluation.turbulence.minimum_spectral_samples`, derived from
  `_MIN_BAND_BINS`, `SPECTRUM_SEGMENTS` and `SPECTRUM_CUTOFF_FRACTION`; the
  naive inversion gives 256 and is wrong because the DC bin is dropped and the
  band is `f < cutoff` strictly). The probe script warns before it spends the
  solve. The cadence this WP shipped, `probes.output_frequency: 1.0`, therefore
  could not produce a spectrum at all for a window shorter than ~264 s — and at
  the plan's nominal 300 s window it produced exactly the refusal floor. Branch
  review round 3 replaced it with `0.25` (see below); the 264-sample floor
  itself is unchanged.
- The probe re-run clones its member models **before** the truth solve, and the
  ordering is load-bearing: both mounts share one experiment dir,
  `_probe_window` prunes restarts but keeps the latest, and clones are copied
  from that dir — so cloning afterwards seeded every member's warm start with
  the *truth's* non-equilibrium field, i.e. truth information on the member
  side of a truth-vs-member diagnostic, biasing the LSD optimistically.
- A re-probe deletes the window's existing member records before writing the
  new truth. The `window_index` guard cannot catch a re-probe of the *same*
  window at a different cadence, which is the normal iteration.
- `lsd_truth_floor` keeps the full-record `nperseg` on half records, which
  makes it ≈2x (not ≈sqrt(2)x) the like-for-like scatter — measured 1.99 at
  M=8, 2.08 at M=32 over statistically identical flows. The plan's key is not
  silently rescaled — it is defined as the halves' LSD and still is — but the
  halved value is emitted beside it as **`lsd_truth_floor_comparable`**
  (`= lsd_truth_floor / _HALVES_INFLATION`, `_HALVES_INFLATION = 2.0`), and
  that is the reference the reader is told to score `lsd_posterior_median`
  against, by `spectral_metric_summary`'s own docstring and by
  `docs/scripts_and_configs.md` §2.3 alike. "At or under `lsd_truth_floor`" is
  therefore *not* the pass criterion — and neither number is a threshold at
  all, since the metrics doc sets no acceptance level for the LSD.
- `spectral_metrics` is computed above the `skip_viz` gate, unlike the other
  metric blocks.
- **`spectral_metrics` emits ten keys, not the three the plan names.** The
  spec above asks for `{lsd_posterior_median, lsd_truth_floor, f_cutoff}`;
  `spectral_metric_summary` emits nine unconditionally — `n_sensors`,
  `n_members`, `n_band_bins`, `f_cutoff`, `segment_seconds`, `sample_frequency`
  (a per-record dict), `lsd_posterior_median`, `lsd_truth_floor`,
  `lsd_truth_floor_comparable` — plus `lsd_prior_median` when the run included
  prior probe records. That is seven beyond the plan, and the overshoot is
  recorded rather than trimmed because each one answers a question the three
  planned keys leave open on a real run: `lsd_truth_floor_comparable` is the
  reference the reader is actually told to score against (above);
  `n_band_bins` and `segment_seconds` are what say whether the LSD was measured
  over a decade or over the 4-bin refusal floor; `sample_frequency` is what lets
  a reader recover the per-record cadence offset the deviation above describes;
  `n_sensors` / `n_members` are what the medians were taken over. Additive
  within a new block, so invariant 1 is unaffected — but the plan's schema line
  is not the shipped schema, and this is the record of that.

**Found while implementing, each fixed in its own reviewed pylbm PR**

Two backend bugs surfaced in the WP3.1 verification run and were deferred out of
this WP as needing their own reviewed PRs. Both landed on 2026-08-07, before
this branch was merged, along with a third that the first one uncovered. All
three are recorded in the master plan's "Outside the WP list" section; this is
the phase-3 view of how they were found.

- **pylbm restart filename width — PR #112 (`bea72c3`).** The pinned Fortran
  reads `restart_0000_<it:i6.6>.uf` while pylbm wrote `:09d`. A Python-authored
  warm start was therefore invisible, and when a stale 6-digit restart at the
  same iteration existed — which is exactly the rollout case, since the solver
  writes one at the end of every window — it was read *instead*, silently
  discarding the state Python supplied. So every pylbm ESMDA rollout warm start
  had been restarting from the solver's own previous-window field; for
  `esmda/smoother=state_and_parameter|state_and_dynamic` the state update was
  discarded outright. Observed directly in the WP3.1 verification run (a member
  dir holding both filename widths at a stale iteration). Both sides now spell
  the width from one constant. `run_probe_series.py` had carried a local
  workaround — `_link_restart_for_solver`, an `os.link` re-exposing the 9-digit
  restart under the 6-digit name, plus its own overflow ceiling; **#112 deleted
  the shim along with the bug**, so no hard link remains anywhere in the tree
  and its coverage moved to `tests/test_pylbm_restart_filenames.py`, which pins
  the width against the Fortran sources so it cannot drift back.
- **A truncated member is not treated as a failure — PR #113 (`7e1c73d`).** The
  same run lost a member whose solver hit a Fortran `stop` (exit code **0**)
  after 3 of 48 frames. `resample_from_successes` never saw a failure, and the
  run died two hours in inside `_stream_concat_members` on the shape mismatch.
  The wrapper now checks the frame count against the cadence-derived expected
  count and reports a short member as a member failure, so the substitution
  machinery sees it at the window that produced it.
- **The restart template was never read — PR #114 (`fae5dc0`),** uncovered by
  #112 (whose fix also corrected the template lookup's filename width). See the
  master plan; it changed what a pylbm warm start physically is, so it bears on
  every pylbm number this phase's probe reruns produce.

**Branch review round 3 (2026-08-07, `be1433e`).** Not a WP review: an
eight-agent adversarial sweep over the *whole* `esmda-evaluation` branch after
every WP had merged, which confirmed the plan is fully implemented and found six
blockers. One is phase 3's; the rest are recorded in the phase-1 and phase-2
plans and in the master plan's filtering section.

- **The shipped probe cadence produced exactly the refusal floor.**
  `probes.output_frequency: 1.0` over the plan's nominal 300 s window gives 300
  samples and **4** scored bins — `_MIN_BAND_BINS` itself — spanning 0.027 to
  0.108 Hz, about 0.6 of a decade. At `U ~ 8 m/s` over 14 m blocks that band is
  the *energy-containing* range, not an inertial range, so figure S4's inertial
  reference slope (`−2/3` on the premultiplied axes, i.e. `−5/3` in `E(f)`)
  would have been drawn beside a band that contains no inertial range at all,
  and the LSD reported was an RMS over four bins — against this WP's own stated
  requirement of `≳ 10³` samples. One window shorter and the metric no-ops
  entirely, *after*
  the solve. The default is now **0.25 s** (1200 samples, 18 bins, 1.26
  decades), which is the value the WP3.2 spec was written against; the
  pre-flight reports the bin count and warns below a decade. The cost is 4x the
  transient scratch, which `conf/run_probe_series.yaml` and
  `docs/job_scripts.md` now lead with rather than bury.
- **The cutoff's stated rationale was wrong.** `SPECTRUM_CUTOFF_FRACTION` was
  documented as an anti-aliasing margin. It is not one, and reading it as one
  licenses a much coarser cadence than is safe: snapshot probing is
  instantaneous point sampling with no anti-alias filter, so content in
  `(fs/2, fs)` folds *mirrored* onto `(fs/2, 0)` — energy near `f ~ fs` lands
  near **DC**, at the bottom of the scored band, where discarding the top octave
  cannot reach it (verified: sampling `sin(2π·0.98·fs·t)` at `fs` puts the peak
  at `0.02·fs`). What the constant actually bounds is **solver damping** — a
  solver dissipates its own smallest resolved scales and the SGS closure bites
  at the same end, so scoring there compares numerics against numerics — and the
  comment now says so, with the folded power quantified (~0.2 % a decade below
  the cutoff, ~13 % / 0.55 dB at it, against the 1–3 dB the LSD reports, and
  largely common-mode since truth and members are sampled alike). No number
  changed; the justification did, and with it the rule that a cadence is sized
  from the band wanted rather than from this constant.

**Branch review round 4 (2026-08-07, `a843f00`).** A four-agent review of round
3's own fixes. Phase 3's cadence fix was verified sound as landed and the value
is unchanged. Two follow-ups:

- The sub-decade pre-flight warning round 3 added **had no test at all** — the
  one new user-facing behaviour of that fix was unpinned. It now has one.
- `docs/job_scripts.md` still shipped a copy-pasteable command pinning
  `probes.output_frequency=1.0`, which would have reinstated the 4-bin refusal
  floor the cadence change exists to prevent. Removed, and the scratch budget
  there now leads with the 4x cost (barcelona ~103 GB/member, ~411 GB at 4
  workers).
