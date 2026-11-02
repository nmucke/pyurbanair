# Surrogate comparison: paper metrics, figures and tables

Replaces the outputs of `scripts/surrogate_comparison/compute_metrics.py` and
`visualize_comparison.py`, and adds `make_tables.py`. Sources: Qin et al. 2025
(Local-FNO), Park et al. 2026 (3DSwinUrbanNet), Park & Lee 2026
(SSRollingUrbanNet). `run_models.py` is unchanged. The definitions as built
are in `scripts/surrogate_comparison/metrics_reference.pdf`.

Goal: every number and figure goes straight into the paper, and each metric is
one small function that can be reviewed on its own.

## Files

| File | Holds |
|---|---|
| `scripts/utils/surrogate_metrics.py` | one pure numpy function per metric, the formula in its docstring |
| `scripts/surrogate_comparison/compute_metrics.py` | reads the rollouts and the truth, calls the metrics, writes `metrics.nc` + `metrics.yaml` |
| `scripts/surrogate_comparison/visualize_comparison.py` | one function per figure, the shared paper style; reads `metrics.nc` (+ rollouts for the snapshot figures) |
| `scripts/surrogate_comparison/make_tables.py` | the LaTeX tables, from `metrics.nc` + `metrics.yaml` |
| `tests/scripts/test_surrogate_metrics.py` | unit tests of every metric on analytic fields |
| `tests/scripts/test_comparison.py` | the scripts end to end on the tiny trained steppers |

## Conventions

Fields are `(T, C, nz, ny, nx)` float arrays, `C = (u, v, w)`; `fluid` is the
`(nz, ny, nx)` bool mask; spacing `(dz, dy, dx)` in m. Per sample `s`:

- **window**: lead steps 1..N of the compared rollout (N: the shortest rollout).
- **ū**: the truth's per-cell time mean over the window. Every *per-lead*
  fluctuation (ACC, spectra, Λ, PDFs) is about ū, the **same reference for
  every source**, so a model is never rewarded for its own drift.
  Window statistics (TKE fields, profiles, `tke` curve) keep each source's
  **own** time mean, as now.
- **H**: mean building height, mean over columns with a solid cell of
  `(solid cells in the column) * dz` [m].
- **canopy-top level `k_c`**: the lowest building-free level (so z_c / H varies by sample), with no solid cell (fully fluid
  spanwise lines everywhere); `z_c` its centre height. Spectra, Λ and PDFs are
  taken there.
- **pedestrian level `k_p`**: `metrics.pedestrian_level` (default 0).
- **u_ref**: the plane mean of ū_u over the fluid cells at `k_c` [m/s].
- **T**: the truth's integral time scale, Local-FNO §3.1: per fluid cell and
  component, R(τ) the autocorrelation of `truth - ū` over the window
  (biased estimator), `T_cell = Δt * (Σ_{τ=0}^{τ0-1} R(τ) − 1/2)` (trapezoid
  rule; the rectangle rule is biased high by Δt/2) with τ0 the first lag where
  R ≤ 0 (the whole window if none); `T` the mean over cells and components. `T̄` = mean of T over the compared samples.
- Everything is made dimensionless **per sample** before averaging over
  samples (the inflow speed differs per trajectory): velocities / u_ref,
  vorticity and divergence × H / u_ref, TKE and stresses / u_ref², lengths / H,
  spectra E / (u_ref² H) against κ·H, κ = 2π/λ the angular wavenumber [rad/m]
  with Σ E dκ = variance. Lead time is shown as t / T̄.
- Persistence (start frame repeated) is a source everywhere; it is shown but
  never bolded. `truth` is the reference curve where it has one.
- An unstable (non-finite) rollout: NaN curves from its first non-finite frame,
  NaN window statistics, `stable: false`.

## Metrics (`metrics.nc`, dims `source x lead` unless noted; sample mean)

| Name | Definition | Truth curve |
|---|---|---|
| `rmse`, `rmse_{u,v,w}` | RMSE over fluid cells, pooled / per component, / u_ref | – |
| `acc`, `acc_{u,v,w}` | anomaly correlation about ū over fluid cells, pooled / per component | – |
| `rmse_pedestrian`, `acc_pedestrian` | the same, pooled over components, at `k_p` only | – |
| `vorticity_rmse`, `vorticity_rmse_{x,y,z}` | RMSE of ω (central differences, cells with six fluid neighbours), × H / u_ref | – |
| `divergence_rms` | RMS of ∇·u on the same cells, × H / u_ref (existing) | yes |
| `tke` | fluid-mean resolved TKE about the source's own time mean, / u_ref² (existing) | yes |
| `integral_length` | Λ_y / H at `k_c`: spanwise autocorrelation R(Δy) of u' = u − ū over the fully fluid y-lines, each line's spanwise mean removed per frame (the κ = 0 bin), `Λ = dy (Σ R − 1/2)` (trapezoid) up to the first zero crossing | yes |
| `spectrum_{u,w}` (`source x lead x kh`) | spanwise spectrum of u' (w') at `k_c`, / (u_ref² H), on the fully fluid y-lines, binned onto shared log-spaced κ·H bins, averaged over the samples (the ratio figure takes the ratio of these means) | yes |
| `pdf` (`source x pdf_lead x var x bin`), `pdf_bin_centre` (`var x bin`) | density of u'/u_ref, v', w' at `k_c` at the `metrics.snapshot_leads`, pooled over samples, 60 bins over the truth's 0.05–99.95 percentile range (the same for every source; values outside fall out but count in the normalisation) | yes |
| `profile_{mean_u,tke,uw}` (`source x z`) | fluid-plane-averaged window profiles, / u_ref, / u_ref² (existing) | yes |

Coordinates: `source`, `lead`, `lead_time` (s), `lead_over_T` (t / T̄), `kh`,
`pdf_lead` (steps), `var`, `bin`, `z`, `z_over_H` (z / H̄). Per-sample
references as data vars on dim `sample`: `reference_H`, `reference_u`,
`reference_T`, `canopy_level`. Attrs `T_mean`, `H_mean`.

## Scalars (`metrics.yaml`, per source `{mean, per_sample}`)

`rmse` (RMS of the curve over the window), `rmse_vs_persistence`,
`acc_horizon_seconds` (first lead with pooled ACC < `metrics.acc_threshold`,
censored at the window's last lead time if never), `acc_horizon_censored`
(1 if ACC never drops, else 0; the mean is the censored fraction), `stable`, `vorticity_rmse`, `rmse_pedestrian` (window RMS),
`mean_u_rmse`, `tke_rmse`, `profile_rmse_{mean_u,tke,uw}` (window
statistics, dimensionless), `spectra_db_{u,w}_{large,mid,near_cutoff}`
(10 log10 of the window-mean, band-summed spectrum against the truth's,
`evaluation.turbulence.SPECTRAL_BANDS`), `integral_length` (window mean of
Λ/H; truth too), `wasserstein_{u,v,w}` (W2 of the fluctuations at `k_c`,
pooled over the window, / u_ref), `divergence_ratio` (window mean
divergence_rms / the truth's). Plus `seconds_per_step`, `n_parameters`, and
`speedup` = `metrics.solver_seconds_per_step / seconds_per_step` when that is
set. `truth_halves` (the truth's first half against its second; not a noise floor when the inflow varies in time) stays. A
top-level `reference` entry holds per sample H, u_ref, T, k_c, z_c.

## Tables (`make_tables.py`, `table_<name>.tex` + `.csv`)

booktabs, a `table` environment with caption and label, ready to `\input`.
Rows: models in config order (display names), then persistence (and the truth
for Λ/H), separated by `\midrule`; `truth_halves` is not tabulated. The best
model value per column is `\textbf` (min, max, or closest to the truth / to
0 dB). Fixed decimals per column.

- `accuracy`: RMSE/u_ref (RMS) and ACC (mean) over leads `t ≤ h T̄` for each
  `metrics.horizons` and the whole window; ACC horizon (t/T̄); vorticity RMSE;
  pedestrian RMSE. A horizon some sample never drops below is a lower bound (≥).
- `physics`: mean-u RMSE, TKE RMSE, −u'w' profile RMSE, spectral dB of u' (mid,
  near-cutoff), Λ/H (truth row), W2 of u', divergence ratio.
- `cost`: parameters (M), ms per step, speedup (if set).

## Figures (`visualize_comparison.py`, PDF + PNG)

One style for all: serif text at 8 pt, single-column 3.5 in and double-column
7.2 in widths, a colour-blind-safe colour per model (config order, the same in
every figure), truth solid black, persistence dashed grey, one legend above
the panels, panel letters (a), (b), …, light grid, dimensionless axes.

1. `rmse` – RMSE/u_ref vs t/T̄: pooled, u, v, w.
2. `acc` – ACC vs t/T̄: pooled, u, v, w; threshold line.
3. `vorticity` – vorticity RMSE vs t/T̄: pooled, x, y, z.
4. `physics` – TKE, divergence, Λ/H vs t/T̄, truth as reference.
5. `pedestrian` – RMSE and ACC at `k_p` vs t/T̄.
6. `spectra` – premultiplied k·H·E vs k·H, at the snapshot leads; rows u', w'.
7. `spectral_ratio` – 10 log10(E_model/E_truth) of u' over (k·H, t/T̄), one panel per model, diverging map, contours at 0.8 and 1.2.
8. `pdfs` – u', v', w' densities at the snapshot leads, log scale.
9. `profiles` – mean u, TKE, −u'w' vs z/H̄.
10. `station_profiles` – the same window statistics at columns behind the most downstream building of `figures.sample` (x = lee face + {0.5, 1, 2, 4} H_b, H_b its own height, at its spanwise centre).
11. `slices` – |U|/u_ref at `k_p`, rows truth + models, columns the snapshot leads.
12. `vertical_slices` – x–z plane through the most downstream building of `figures.sample`: u/u_ref, w/u_ref and spanwise vorticity, rows truth + models, at `figures.vertical_lead`.
13. `statistics_slices` – window mean u/u_ref and TKE/u_ref² at `k_p`.
14. `tke_heights` – window TKE/u_ref² on horizontal slices at `figures.tke_heights` (units of H) of `figures.sample`, rows truth + models.
15. `animation_speed.mp4` – unchanged (review aid, not a paper figure); `animation_tke_z<h>H.mp4` – the instantaneous TKE at each of those heights.

Dropped: `energy.png` (std ratio), `tke_slices.png`, the raw-velocity PDFs and
Wasserstein, `rmse_step<k>`, `final_rmse`.

## Config (`configs/comparison.yaml`)

```yaml
display_names: {}           # rollout label -> name in figures and tables, e.g. {dft_s: Ours}
metrics:
  acc_threshold: 0.5
  pedestrian_level: 0
  snapshot_leads: [1, 10, 50, -1]  # PDFs, spectra columns and slices (-1: last)
  horizons: [1, 5]          # table lead windows in units of T̄, plus the whole window
  max_pdf_values: 200000
  solver_seconds_per_step: null    # solver wall time per output interval; null: no speedup
figures:
  sample: 0
  vertical_lead: 10
  tke_heights: [0.5, 1.0, 1.5, 2.0]  # in units of H
  formats: [pdf, png]
  animation: {max_frames: 150, fps: 10, dpi: 100}
```
