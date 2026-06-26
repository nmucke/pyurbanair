# Results — figure & table specification (EnKF 2026 talk)

This document specifies **every figure, table, and animation** to produce from
the finalized simulations, so a follow-up agent with data access can generate
them and they drop straight into the beamer deck (`latex/main.tex`).

Neural surrogates are **out of scope** here.

---

## 0. How to read this

* Each item has an **ID**, a **priority** (`[ESSENTIAL]`, `[RECOMMENDED]`,
  `[OPTIONAL/BACKUP]`), the **filename(s)** to write, and a precise content
  spec.
* Put everything under a single `figures/` folder in this work directory
  (layout in §2.4). Keep the **exact filenames** — they are referenced from
  the slides.
* Where a metric/number is needed, also emit a **CSV** and a **ready-to-paste
  LaTeX `booktabs` snippet** (`.tex`) so I don't re-key numbers.
* Anything you must assume about the data, list it in a `figures/NOTES.md` so I
  can reconcile (see §10).

---

## 1. The experiments (recap) & notation

Ground truth for **all** runs: one **high-resolution uDALES** simulation
(`truth`).

**Block A — Parameter-only ESMDA, model × resolution.**
Estimate the time-varying **inflow parameters** only; reconstruct the state by
running the model forward with the parameter ensemble. Models:

* `palm`, `udales`, `lbm` (Lattice Boltzmann)
* each at **several resolutions** R1 (coarsest) … Rk (finest). Use the actual
  grid spacing / cell count in labels (e.g. `Δx = 4 m`, `50×40×8`).

**Block B — Initial-condition (IC) + parameters, uDALES only.**
Estimate, **per window**, the window's initial condition **and** the inflow
parameters. Three update strategies:

* `corr` — correlation-based localization
* `dist` — distance-based localization
* `red`  — state-reduction method

Shorter horizon / shorter windows than Block A, **but same `truth`** → directly
comparable on the overlapping horizon. (The state-reduction method is not yet
explained in the slides — that's a later task; just produce its figures/numbers
now and label it `state reduction`.)

**Block C — IC + full state-in-window + parameters, uDALES, `red` only.**
A single run that additionally updates the state **throughout** each window
(not just the IC). Treat as one extra "method" alongside Block B.

**Parameters** (time series on knots, per window): primarily
`inflow_angle` α(t) [deg] and `velocity_magnitude` |U|(t) [m/s]; include
`pressure_gradient` if present for uDALES. We have **prior and posterior** of
**states and parameters in every window** for every run.

**Method labels** (use verbatim in legends/tables):
`param-only`, `IC+param (corr)`, `IC+param (dist)`, `IC+param (reduction)`,
`IC+state+param (reduction)`.

---

## 2. Global conventions

### 2.1 Colors (UrbanAIR palette)
* truth → **black** (`#000000`), solid, lw≈2.
* posterior → **teal** `#009BC2` (mean solid; ±1σ band same hue, alpha 0.25).
* prior → **grey** `#9AA6AC` (mean dashed; band alpha 0.18).
* accents for method/model series: orange `#D0661C`, amber `#DC911B`,
  light blue `#92C7DF`, charcoal `#343434`.
* assimilation **window boundaries** → thin dashed vertical lines, grey.
* field heatmaps: sequential perceptual map (`viridis`) with a **shared color
  scale** across panels being compared; **difference** maps: diverging
  (`RdBu_r`) centered at 0 with a symmetric range.

### 2.2 Plot style
* No chart titles baked in (the slide gives the title); keep concise axis
  labels with units. Legends inside the axes, small.
* Fonts ≥ 9 pt at final size; line plots exported so text is legible at
  ~11 cm width. Transparent background.
* Mark window boundaries on every time-axis plot; annotate window indices.

### 2.3 Formats
* **Line / scatter / bar** plots → **vector PDF** (`.pdf`).
* **Field heatmaps / many-panel image grids** → **PNG at 300 dpi**
  (`.png`) (vector PDFs of pixel fields get huge).
* **Animations** → deliver an **`.mp4`** (H.264, yuv420p) per animation at the
  spec'd size/fps; I convert to frames for the `animate` package. Also fine to
  additionally drop a `frames/` subfolder (`frame_%05d.png`, starting at 0) if
  easy.
* **Tables** → `*.csv` **and** `*.tex` (booktabs snippet).

### 2.4 Folder layout & naming
```
figures/
  NOTES.md                      # assumptions, resolution list, sensor coords, masks
  conventions_check.png         # 1 sample of each style for sign-off (optional)
  A_param_only/
    A1_param_traj_<model>.pdf            # one per model (palm,udales,lbm)
    A1_param_traj_grid.pdf               # combined 2×3 grid (params × models)
    A2_param_err_vs_res.pdf
    A3_state_fields_truth_vs_models.png
    A4_valsensors_<model>.pdf            # one per model
    A5_state_err_vs_res.pdf
    A6_window_prior_post_<model>.pdf     # optional
    anim_A_<model>_<res>.mp4             # truth | estimate | error triptych
    tables/
      A_param_accuracy.csv / .tex
      A_state_accuracy.csv / .tex
  B_state_estimation/
    B1_param_traj_methods.pdf
    B2_state_err_vs_time_methods.pdf
    B3_state_fields_methods.png
    B4_valsensors_methods.pdf
    B5_window_prior_post_methods.pdf
    anim_B_methods.mp4                   # side-by-side methods over a window
    tables/
      B_method_comparison.csv / .tex
  C_full_joint/
    C1_state_fields_full.png
    anim_C_full.mp4
  summary/
    S1_cost_vs_accuracy.pdf
    S2_headline_metric.pdf
    tables/ (optional combined)
```

---

## 3. Quantities & metrics (definitions)

Compute all field/sensor metrics on a **common evaluation grid** (the `truth`
grid, or a fixed reference grid) by interpolating each model / reduced state
onto it. **Mask out solid (building) cells.** Where useful, report **canopy**
(z ≤ building height H) and **above-canopy** separately.

**Parameters** (posterior-mean trajectory θ̂ vs θ_true):
* `RMSE_α` [deg], `RMSE_|U|` [m/s] — RMS over time (and per window).
* `bias` (signed mean error), per parameter.
* **prior→posterior error reduction** [%].
* **spread** = posterior ensemble std (time-mean); **calibration** = fraction
  of time θ_true within posterior ±1σ (target ≈0.68) and ±2σ (≈0.95).

**State** (posterior state ensemble vs truth field):
* `RMSE_U` per component (u,v,w) and for **|U|**; time-mean and per window.
* **normalized RMSE** (by truth RMS or std of |U|).
* **validation-sensor RMSE** at held-out sensors (not assimilated), per
  component and |U|.
* **spread–skill ratio** (ensemble std / RMSE) at sensors — calibration.
* **prior vs posterior** state RMSE (to show the update working).
* `[OPTIONAL]` CRPS at sensors; energy-spectrum / structural similarity of |U|.

**Cost** (for the method/cost figures):
* updated state-vector dimension, ensemble size `N_e`, # forward model runs,
  wall-clock per window and total. (Per model/resolution and per method.)

Always state whether a metric is **per window**, **final window**, or
**horizon-averaged**.

---

## 4. Block A — Parameter-only: model × resolution

> Story: with `palm`, `udales`, `lbm` at varying resolution (truth = hi-res
> uDALES), (1) how well are the **inflow parameters** recovered, and (2) how
> well is the **flow state reconstructed** from them?

### A1 — Parameter trajectories vs truth `[ESSENTIAL]`
`A1_param_traj_<model>.pdf` (one per model) **and** `A1_param_traj_grid.pdf`.

* **Per-model figure:** 2 stacked panels (top `inflow_angle` α(t) [deg],
  bottom `velocity_magnitude` |U|(t) [m/s]) across the **full multi-window
  horizon**. Each panel: truth (black), **posterior** mean + ±1σ band (teal),
  **prior** mean + band (grey), window boundaries dashed. Overlay the curves
  for the model's **resolutions** as distinct line styles/shades of teal (or
  small inset legend) — or, if cluttered, show only the finest + coarsest and
  defer the rest to the table.
* **Grid figure:** rows = {α, |U|}, columns = {palm, udales, lbm} at a single
  **representative resolution** (state which) — for a compact "all models at a
  glance" slide. Same color conventions; shared y-limits per row.

### A2 — Parameter error vs resolution `[ESSENTIAL]`
`A2_param_err_vs_res.pdf`.
* Two panels (α, |U|). x = resolution (Δx [m] decreasing → finer, or cell
  count); y = parameter RMSE. One line+markers **per model** (palm/udales/lbm).
* This is the core **model × resolution** comparison for parameters. Log x if
  spacing is geometric. Mark the truth resolution on the axis.

### A3 — Reconstructed state fields: truth vs models `[ESSENTIAL]`
`A3_state_fields_truth_vs_models.png`.
* Horizontal slice of **|U|** at pedestrian height (z ≈ 2–10 m; state which)
  at a **representative time** (state which window/time).
* Panel grid: **columns** = {truth, palm, udales, lbm} (each at its finest
  resolution, posterior-mean reconstruction); **two rows**: row 1 = |U| field
  (shared color scale), row 2 = **error** (model − truth, diverging map,
  symmetric scale, shared). Buildings masked/hatched.
* Add a thin marker for assimilation sensors (○) and validation sensors (△) on
  the truth panel.

### A4 — Validation-sensor time series `[ESSENTIAL]`
`A4_valsensors_<model>.pdf` (one per model).
* Pick **3–4 held-out validation sensors** (deeper in the canopy / wake, not
  used in assimilation; list coords in NOTES). For each sensor a small panel:
  truth (black) vs posterior reconstruction mean ±1σ (teal) vs prior (grey),
  for **|U|** (or the dominant component), over the full horizon, window
  boundaries dashed.
* Purpose: shows the reconstruction generalizes **beyond** the assimilated
  near-inflow obs. Use the finest resolution; mention it.

### A5 — State reconstruction error vs resolution `[RECOMMENDED]`
`A5_state_err_vs_res.pdf`.
* Same x-axis as A2 (resolution). y = **state |U| field RMSE** (horizon-mean,
  fluid cells). One line per model. `[OPTIONAL]` second panel with
  **validation-sensor RMSE** vs resolution.
* Together with A2 this answers "how do models perform at different
  resolutions" for **both** parameters and state.

### A6 — Windowed prior→posterior illustration `[OPTIONAL/BACKUP]`
`A6_window_prior_post_<model>.pdf`.
* For one model/resolution: per window, show the parameter ensemble **prior**
  (window start) vs **posterior** (after ESMDA) as box/violin or mean±σ,
  illustrating the windowed update and how spread re-opens across windows.

### Table A-I — Parameter accuracy `[ESSENTIAL]`
`A_param_accuracy.csv` / `.tex`.
Columns: `Model`, `Resolution`, `RMSE α [deg]`, `RMSE |U| [m/s]`,
`α reduction [%]`, `|U| reduction [%]`, `α coverage ±1σ`, `|U| coverage ±1σ`.
Rows: every (model × resolution). Bold the best per column in the `.tex`.

### Table A-II — State reconstruction accuracy `[ESSENTIAL]`
`A_state_accuracy.csv` / `.tex`.
Columns: `Model`, `Resolution`, `|U| field RMSE`, `u/v/w RMSE` (or just |U|),
`norm. RMSE`, `val-sensor |U| RMSE`. Rows: every (model × resolution).

### Animation A-anim — per-model state evolution `[RECOMMENDED]`
`anim_A_<model>_<res>.mp4` (finest res; at least uDALES + LBM, ideally all 3).
* **Triptych** (3 stacked or side-by-side panels): **truth | posterior-mean
  estimate | error (diverging)** of |U| on the slice, evolving over the full
  horizon. Window boundaries flashed/annotated. ~10 fps, ≤ ~15 s, width 2000 px
  (4:1 ok), shared color scales. (These supersede the current
  `*_state_animation.mp4` placeholders.)

---

## 5. Block B — Does state estimation help? (uDALES)

> Story: holding the model (uDALES) and truth fixed on the shorter horizon,
> compare `param-only` vs `IC+param (corr/dist/reduction)` vs
> `IC+state+param (reduction)`. Does estimating the IC/state improve the flow,
> and which localization/reduction wins?

Include `param-only` (uDALES, restricted to the overlapping horizon) as the
**baseline** in every comparison here.

### B1 — Parameter trajectories by method `[RECOMMENDED]`
`B1_param_traj_methods.pdf`.
* 2 panels (α, |U|) over the shorter horizon. One colored line (mean; thin
  ±1σ band optional) **per method** + truth (black). Window boundaries dashed.
* Shows whether adding state/IC estimation changes parameter recovery.

### B2 — State error vs time, by method `[ESSENTIAL]` ← key figure
`B2_state_err_vs_time_methods.pdf`.
* x = time over the shorter horizon; y = **state |U| field RMSE** (fluid cells,
  common grid). One line **per method** + window boundaries dashed.
* Expect the characteristic **saw-tooth** (error drops at each window's
  assimilation, grows within the window). This is the clearest "does IC/state
  estimation help" visual: `param-only` should jump/stay high at window starts;
  IC methods should pull error down at each window start; full joint lowest.
* `[RECOMMENDED]` companion `B2b_valsensor_err_vs_time_methods.pdf` with the
  same lines but **validation-sensor** RMSE.

### B3 — State field snapshots by method `[ESSENTIAL]`
`B3_state_fields_methods.png`.
* |U| slice at a representative time (ideally **near end of a window**, where
  methods differ most). Columns = {truth, param-only, IC+param (corr),
  IC+param (dist), IC+param (reduction), IC+state+param (reduction)}; rows =
  {field (shared scale), error (diverging, shared)}. Buildings masked; sensors
  marked on truth.

### B4 — Validation-sensor time series by method `[RECOMMENDED]`
`B4_valsensors_methods.pdf`.
* 2–3 held-out sensors (same as A4 where possible). Per sensor: truth (black)
  + each method (colored mean, optional band) for |U| over the shorter horizon.

### B5 — Prior vs posterior state error per window `[OPTIONAL/BACKUP]`
`B5_window_prior_post_methods.pdf`.
* Per window, bar/point of **state RMSE before vs after** the window update for
  each method — quantifies the per-window correction the IC/state update buys.

### Table B-I — Method comparison (uDALES) `[ESSENTIAL]`
`B_method_comparison.csv` / `.tex`.
Rows = the 5 methods. Columns:
`RMSE α [deg]`, `RMSE |U| [m/s]`, `|U| field RMSE`, `val-sensor |U| RMSE`,
`updated state dim`, `N_e`, `wall-time / window [s]` (or total), and a
calibration column (spread–skill or ±1σ coverage). Bold best per column.
* This table is the quantitative core of the "state estimation benefit"
  message.

### Animation B-anim — methods side by side `[RECOMMENDED]`
`anim_B_methods.mp4`.
* |U| slice over **one (or two) window(s)**, panels = {truth, param-only,
  IC+param (best loc), IC+state+param (reduction)} (4 panels keeps it legible),
  shared color scale, window boundary annotated. Shows the IC correction at the
  window start visually. Same encoding spec as A-anim.

---

## 6. Block C — Full joint run highlight `[RECOMMENDED]`

The single `IC+state+param (reduction)` run is already included as a method in
Block B. Additionally provide a **showcase**:

* `C1_state_fields_full.png` — truth vs full-joint estimate vs error of |U| at
  a few times (e.g. window-start, mid, end) in a small grid — the "best we can
  do" result.
* `anim_C_full.mp4` — truth | estimate | error triptych over the full (short)
  horizon. (May reuse the B-anim panel for this method.)

---

## 7. Cross-cutting summary `[RECOMMENDED]`

### S1 — Cost vs accuracy `[RECOMMENDED]`
`S1_cost_vs_accuracy.pdf`.
* Scatter: x = computational cost (wall-time, or # forward runs, or updated
  state dim — pick the most honest; state which), y = **state |U| RMSE**.
* Points: every Block-A (model × resolution) run + every Block-B/C method
  (uDALES). Encode model by color, resolution by marker size, method by marker
  shape. A legend maps the encodings. This is the "what should you actually
  run" slide.

### S2 — Headline metric bar `[OPTIONAL]`
`S2_headline_metric.pdf`.
* Grouped bar of one headline number (e.g. val-sensor |U| RMSE): group 1 =
  models at finest res (param-only); group 2 = uDALES methods. One glance:
  model choice vs estimation-strategy choice.

---

## 8. Suggested slide mapping (Part 4 "Results")

1. **Parameters recovered across models** → `A1_param_traj_grid.pdf` (+ per-model
   in backup).
2. **Parameter accuracy vs resolution** → `A2_param_err_vs_res.pdf` + Table A-I.
3. **State reconstructed from parameters** → `A3_state_fields_truth_vs_models.png`.
4. **Generalization to validation sensors** → `A4_valsensors_udales.pdf`
   (+ `A5_state_err_vs_res.pdf`) + Table A-II.
5. **(animation)** `anim_A_udales` / `anim_A_lbm` on the existing "state
   estimation" slides.
6. **Does state estimation help?** → `B2_state_err_vs_time_methods.pdf` (key).
7. **Method comparison fields** → `B3_state_fields_methods.png` + Table B-I.
8. **Full joint showcase** → `C1` / `anim_C_full` (later, once state-reduction
   is introduced).
9. **Cost vs accuracy wrap-up** → `S1_cost_vs_accuracy.pdf`.

(Items 6–8 land in/after the localization section; the state-reduction method
still needs its own explanatory slide first — flagged for later.)

---

## 9. File manifest (checklist)

```
A_param_only/A1_param_traj_palm.pdf
A_param_only/A1_param_traj_udales.pdf
A_param_only/A1_param_traj_lbm.pdf
A_param_only/A1_param_traj_grid.pdf
A_param_only/A2_param_err_vs_res.pdf
A_param_only/A3_state_fields_truth_vs_models.png
A_param_only/A4_valsensors_palm.pdf
A_param_only/A4_valsensors_udales.pdf
A_param_only/A4_valsensors_lbm.pdf
A_param_only/A5_state_err_vs_res.pdf
A_param_only/A6_window_prior_post_udales.pdf            (optional)
A_param_only/anim_A_udales_<res>.mp4
A_param_only/anim_A_lbm_<res>.mp4
A_param_only/anim_A_palm_<res>.mp4                      (optional)
A_param_only/tables/A_param_accuracy.{csv,tex}
A_param_only/tables/A_state_accuracy.{csv,tex}
B_state_estimation/B1_param_traj_methods.pdf
B_state_estimation/B2_state_err_vs_time_methods.pdf
B_state_estimation/B2b_valsensor_err_vs_time_methods.pdf   (recommended)
B_state_estimation/B3_state_fields_methods.png
B_state_estimation/B4_valsensors_methods.pdf
B_state_estimation/B5_window_prior_post_methods.pdf        (optional)
B_state_estimation/anim_B_methods.mp4
B_state_estimation/tables/B_method_comparison.{csv,tex}
C_full_joint/C1_state_fields_full.png
C_full_joint/anim_C_full.mp4
summary/S1_cost_vs_accuracy.pdf
summary/S2_headline_metric.pdf                          (optional)
NOTES.md
```

---

## 10. Assumptions to verify against the data (please confirm in NOTES.md)

1. **Parameters present:** is it `{inflow_angle, velocity_magnitude}` for all
   models, plus `pressure_gradient` for uDALES only? Adjust panels accordingly.
2. **Resolutions:** list the actual set per model (Δx and/or cell counts) and
   which is "finest/representative" — used in labels and A1-grid / A3.
3. **Truth resolution** and whether the finest uDALES assimilation run equals
   the truth grid (inverse-crime caveat → metrics on a common grid regardless).
4. **Sensors:** coordinates of assimilation sensors and a set of **held-out
   validation sensors** (place some in wakes / deeper canopy). If no held-out
   sensors exist, evaluate at all sensors and label them "assimilated".
5. **Evaluation slice/region:** the z-level and any sub-region (canopy vs
   above-canopy) for fields; the building mask.
6. **Horizons/windows:** number and length of windows for Block A vs B/C, and
   the overlapping horizon used for the A-vs-B baseline comparison.
7. **Cost numbers:** are wall-times available? If not, use updated state
   dimension and/or # forward runs as the cost axis in S1 / Table B-I.
8. **Units & sign conventions:** angle in degrees, speed in m/s, error sign
   (model − truth) for diverging maps.

> Keep the **filenames and folder names exactly** as above so the slides can
> reference them without edits. Emit tables as both `.csv` and a `booktabs`
> `.tex` snippet.