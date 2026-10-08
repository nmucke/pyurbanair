# Turbulence-aware metrics and figures for assimilation runs

Status: proposed, not implemented. Written 2026-10-08 against `5eba7bc`. This
adds what [sgs_discrepancy_twin_tests.md](sgs_discrepancy_twin_tests.md) needs
to judge its runs. It is also needed for any assimilation run with turbulent
inflow.

## Why

The truth and every member are driven by different turbulent realisations.
They decorrelate within a few eddy turnovers, which for this case (\|U\| about
5 m/s, buildings about 10 m) is seconds. Any **instantaneous** comparison is
then dominated by unpredictable eddies, whatever the parameters are. That
covers `state.vel_magnitude_rmse`, the `sensors` RMSE and energy score, the
`spread_skill` ratio and the RMSE panel of `parameter_evolution.png`. The
instantaneous spread–skill ratio reads about 1 even when the parameters are
wrong. Parameters act on the flow's **statistics**: means, variances,
stresses, distributions and spectra. `sensor_statistics` already scores window
means and variances at the sensors. What is missing:

1. Statistic errors over the **whole field**, and canopy profiles including
   the momentum flux u′w′.
2. A **noise floor**: the score a perfect model gets against one turbulent
   realisation.
3. **Sensor value distributions**, with distances between predicted and truth
   distributions.
4. **Energy spectra**, the only view of the small scales where the SGS
   correction acts.
5. The **joint posterior** (parameter correlations) and **SGS health**
   (multiplier saturation).
6. **Per-window values**, so that forecast windows can be read separately from
   the first window.

## Simplicity rules for this work

The point is to measure the right things, not to grow the evaluation code.

- **No new modules, scripts or config groups.** The pure array reductions go
  into the existing `libs/evaluation` modules: distributions into `scores.py`,
  field statistics and spectra into `turbulence.py`, plots into `figures.py`.
  Pipeline code goes into the existing `scripts/compute_metrics.py` and
  `scripts/visualize_assimilation.py`. Shared run-dir helpers stay in
  `scripts/utils/helper_functions.py`.
- **One read pass.** `compute_metrics.py` already streams every window file
  one member at a time. Accumulate the new statistics in that same loop. Write
  the small arrays the figures need (profiles, spectra, sensor quantiles,
  parameter correlations) to one `diagnostics.nc` next to `metrics.yaml`.
  `visualize_assimilation.py` draws the new figures from that file instead of
  re-reading tens of GB.
- **One new config key:** `assimilation.replica_dir` (item 2). Everything else
  is a module constant with a one-line reason: quantile count, bin count,
  spectral bands. Do not add knobs for them.
- **Few functions.** Prefer a handful of short functions with array in, array
  out over classes or registries. Each new function gets a unit test against
  an analytic answer.
- **Replace rather than pile up.** The field-statistics block supersedes the
  instantaneous `state.vel_magnitude_rmse` and the RMSE panel of
  `parameter_evolution.png`: remove both. Keep the instantaneous `sensors` and
  `spread_skill` blocks, but document them as sanity checks.
  `docs/evaluation.md` says which numbers are the scores.
- **No-op when absent.** Without `replica_dir`, the replica entries are simply
  missing. A case with no fully fluid lines has no in-canopy spectra and logs
  why. A run without prior states has no prior entries, as today.

## The changes

### 1. Per-window values and parameter diagnostics

- **Per-window values:** for every window-indexed series in `metrics.yaml`
  (parameter RMSE and CRPS, `sensor_statistics` CRPS, the new blocks), also
  write `per_window: [...]` next to `mean/final/max/min`. Do it in the few
  call sites; leave `series_stats` alone.
- **Parameter correlations:** the posterior correlation matrix per window
  (prior too), from the window parameter files. A time-varying parameter enters
  as its window-mean per member. In `metrics.yaml` keep only the
  final-window matrix, as a nested mapping. The full set goes into
  `diagnostics.nc`.
- **SGS health:** when the assimilation model has `model_discrepancy`
  enabled, read the `model_discrepancy_by_member` JSON attribute of each
  posterior and prior state file. Report the multiplier min and max and the
  maximum saturation fraction over members, per window. Verify the JSON keys
  against `libs/pyudales` first.
- **Figure `parameter_pairs.png`:** a corner plot of the final-window posterior
  (member scatter) over the prior, with the truth marked when it is static.
  Make it one function in `figures.py`.

### 2. Noise floor: a truth replica

A forward run of the truth configuration that differs from the truth only in
the turbulence seed (for uDALES,
`model.forward_model.inlet_turbulence.seed`; for PALM, check which random
seed `pypalm` exposes). Its time axis must cover the same horizon. Make it with
`scripts/run_forward.py` (`params=<truth params>`,
`time.simulation_time=num_windows·simulation_time`). For a time-varying truth,
check that the same sampler seed and horizon reproduce the truth's trajectory.

- **Config:** `assimilation.replica_dir: null` in `configs/assimilation.yaml`.
  Open the replica with the same code path as `open_truth`: generalise that
  helper to take the directory, honouring `truth_start_time`. `check_config`
  checks that the directory has a `state.nc` long enough for the horizon.
- **Use:** score the replica as a one-member prediction in every statistics
  block below (and in `sensor_statistics`), under a `replica` key. Each
  posterior score is then read against `replica`: posterior ≈ replica means as
  good as a perfect model can be.
- **Free secondary floor:** for the sensor distributions, also score the
  truth's first half-window against its second half (`truth_halves`). This
  needs no extra run, but it assumes stationarity.

### 3. Field statistics and canopy profiles

These statistics are per member, per window, on cell centres:

- **Statistics:** time-mean u, v and w; TKE = ½(var u + var v + var w); and
  resolved u′w′.
- **Colocation:** use `colocate_components`, since TKE and u′w′ are one-point
  moments.
- **Fluid mask:** `stl_solid_mask`, dilated by one cell, so that centres whose
  colocation touches a solid face are dropped.

Average each member's statistic over the ensemble. Never take the statistic
of the ensemble-mean field, which removes the turbulence. In the loop, keep
only running per-member sums and ensemble totals, so memory stays one member
plus totals, even at Barcelona size. For a truth on another grid (Round 2),
interpolate the truth statistics onto the ensemble centres the way
`streaming_state_rmse` does.

- **`field_statistics` block:** per quantity, the RMSE over fluid cells for
  the posterior, the prior and the replica against the truth, as
  `per_window` plus a per-level RMSE profile. The per-level profile shows
  where in height the error sits (for SGS, the 5–20 m band) without the code
  knowing about bands. This block replaces `state`.
- **`canopy_profiles` block:** intrinsic horizontal averages (over the fluid
  cells of each level) of time-mean u, TKE and u′w′. Report the profile RMSE
  over z for each source. These are resolved stresses only: the outputs
  carry no SGS stress. That is fair within one solver; in Round 2 read it
  with that caveat.
- **Figure `canopy_profiles.png`:** three panels (⟨ū⟩, ⟨TKE⟩, ⟨u′w′⟩ against
  z) showing the truth, the replica, and prior/posterior member bands with
  their mean. The building-height range is shaded. Reuse the style of
  `_plot_profiles`; the existing per-station `station_profiles.png` stays.

### 4. Sensor value distributions

For each sensor set (assimilated and validation), each quantity (u, v, w,
\|U\|) and each window, flatten the values at the sensors:

- **Truth sample:** all frames × sensors, on the posterior's time stamps, as
  now.
- **Prediction sample:** frames × sensors × members, for the posterior, the
  prior and the replica.

Scores, as two functions in `scores.py`:

- **Wasserstein-2 (primary).** In 1-D it is exact from quantile functions.
  On a shared grid of `N_QUANTILES = 99` levels,
  `W2² = mean_p (Q_truth(p) − Q_pred(p))²`. Also report its split into
  location `(Δμ)²`, scale `(Δσ)²` and shape (the rest). This tells whether a
  mismatch is a bias, a wrong turbulence intensity or a wrong shape (skewness,
  tails). W2 is in m/s and stays finite when the supports do not overlap.
- **KL(truth ‖ pred) (secondary).** Use shared histogram bins over the union
  range (`N_BINS = 30`) with additive smoothing of 0.5 counts, in nats. Document
  the caveats: it depends on the binning, it diverges as the supports separate,
  and it is asymmetric.

Also, compute both scores in two ways:

- **Pooled over sensors:** the headline number and the figure.
- **Per sensor:** averaged over sensors, normalised by each sensor's truth
  std. Pooling over sensors makes a mixture, so a too-fast sensor and a
  too-slow one can cancel. The per-sensor number exposes that.

Report the ensemble two ways as well: the pooled ensemble, and the median over
members of each member's own distance. Pooling members with different
parameters widens the distribution. A pooled ensemble can therefore match the
truth's spread through parameter uncertainty rather than turbulence; the
per-member median shows whether that happened.

Sample size is the realisation's, not the frame count: frames are
autocorrelated. Read the distances against the `replica` and `truth_halves`
floors, never alone.

- **`sensor_distributions` block:** for each set and quantity, W2 (with its
  split) and KL, per source, as `per_window`.
- **Figure `sensor_distributions_<set>.png`:** one row per quantity. The left
  column overlays densities (histograms on the shared bins) of the truth, the
  prior, the posterior and the replica, pooled over sensors and windows. The
  right column is a quantile–quantile plot of each prediction against the
  truth, which is the picture of what W2 measures.

### 5. Energy spectra despite the buildings

Two facts make spectra tractable here:

- The **y direction is periodic** in both backends (uDALES `BCym=1`, PALM
  `bc_ns=cyclic`). An FFT along y needs no window or detrending.
- **Spatial** spectra come from the existing 1 s field output at no extra
  cost. Each frame is a snapshot, and averaging over frames, lines and members
  beats down the noise. Temporal spectra at the sensors would need a
  high-cadence re-run (the archived `run_probe_series.py` route); they are
  out of scope here.

The single setup is spanwise spectra on **fully fluid lines**. One function in
`turbulence.py` takes a component on its native grid and the fluid mask. It
FFTs along y every (x, z) line that is entirely fluid, and averages
`|FFT|²` over lines, frames and the x-range downstream of the first building
(dropping the last two cells before the outflow). Group the lines by height:

- **Above canopy:** levels with no solid cell, below the top two cells, to
  avoid the top boundary.
- **In canopy:** lines through open streets parallel to y. In Xie–Castro
  these are the N–S lanes at x = 10, 20 and 30 m. The mask finds them
  automatically, so no case configuration is needed. A case without such
  lines (likely Barcelona) gets no in-canopy spectrum, and a log line says so.

Use native component grids with no interpolation. Interpolation is a low-pass
filter and would bias exactly the high-wavenumber tail being measured.

Scores, in the same block:

- **Band energy ratios** in dB (prediction / truth) over three fixed
  wavelength bands, where Δ is the grid spacing: large `λ > 8Δ`, mid
  `4Δ–8Δ` and near cutoff `2Δ–4Δ`. Report them per component, per height
  group and per source, including the replica. The near-cutoff band is where
  an SGS change shows first; the large band is where inflow errors show.
- **Log-spectral distance** over the resolved band, reusing
  `log_spectral_distance`.

Spectra are averaged per member first. Then take the ensemble median and the
10–90 % range, matching `median_spectrum`.

Cross-solver note: compare in physical wavenumber, up to the coarser grid's
Nyquist. Near the cutoff the two codes' numerical dissipation differs. A
Round 2 mismatch in that band partly measures numerics; say so in the docs.

The domain is small: 80 m in y gives 20 wavenumbers. Expect coarse spectra on
Xie–Castro; the averaging over lines, frames and members is what makes them
usable.

- **Figure `spectra.png`:** rows u, v, w; columns above canopy and in canopy.
  Draw the truth, the replica, the posterior median with its 10–90 % band, and
  the prior median. Shade the three bands and add a `k^(-5/3)` guide.
  Consider reusing `plot_spectra` (S4) if it fits wavenumber axes without
  contortions; otherwise write one plain function and do not generalise S4.

**Deferred: structure functions.** Second-order structure functions over
fluid-cell pairs, `D(r) = ⟨(q(y+r) − q(y))²⟩`, are the masked alternative for
cases with no fully fluid canopy lines. Add them only when such a case needs
in-canopy small-scale scores.

## Tests

- **`tests/evaluation`, analytic checks:**
  - W2 between two Gaussians, against `(Δμ)² + (Δσ)²`, with shape ≈ 0.
  - KL between two Gaussians, against the closed form, within a binning
    tolerance.
  - Spanwise spectra of a synthetic periodic field with known modes, with a
    masked block, checking that only fully fluid lines are used.
  - Intrinsic profile averaging with a mask.
  - u′w′ of a field with a known covariance.
- **`tests/scripts`:** the existing tiny assimilation workflow test checks
  that the new blocks and figures are produced. Add one case with
  `replica_dir` set (the tiny truth rerun with another seed) and one without
  it, checking that the replica keys are absent.

## Docs

- **`docs/evaluation.md`:** list the new functions and the new metrics and
  figures. Say plainly that the window and field statistics, distributions
  and spectra are the scores, the instantaneous blocks are sanity checks, and
  every score is read against the replica floor.
- **`docs/scripts_and_configs.md`:** document `assimilation.replica_dir`.
- **`configs/assimilation.yaml`:** the key, with a short comment.

## Order and size

Make one PR with commits in the order 1 → 2 → 3 → 4 → 5. Items 1–4 are small
changes in the existing loop; item 5 is the only new numerics. The
`compute_metrics.py` and `visualize_assimilation.py` diffs should stay modest:
the computations live in the library functions. If they grow large, the loop
is being duplicated; fix that before adding more.

The overnight twin runs do **not** wait for this. Every new metric is
computed from saved run directories. The runs must keep their window state
files (posterior and prior), and one replica forward run per truth
configuration should be made alongside them. Re-run `compute_metrics.py` and
`visualize_assimilation.py` on them once this lands.
