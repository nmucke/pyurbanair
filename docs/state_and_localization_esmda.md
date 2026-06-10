# Joint state + time-varying-parameter ESMDA, localization, and the IC fix

Summary of the work on branch `feature/state-tv-esmda`. It adds a joint
state + time-varying-parameter smoother, two state-only localization strategies
selectable from config, a full-vector sensor error metric, a comparison driver
script, and — critically — fixes the analysis loop so the estimated initial
condition is actually used (previously it was computed and discarded, which made
state estimation and state localization a no-op).

See also: [`docs/codebase_guide.md`](codebase_guide.md) §6 (data-assimilation
flow) for the surrounding architecture.

---

## 1. `StateAndTimeVaryingParameterESMDA`

A new smoother
([`smoothing/esmda.py`](../libs/data-assimilation/src/data_assimilation/smoothing/esmda.py))
that **jointly estimates the window's `time=0` initial-condition state and the
time-varying (AR(2)) parameters**. It combines:

- `StateAndParameterESMDA` — flattens the `time=0` state ensemble into the
  augmented Kalman vector;
- `TimeVaryingParameterESMDA` — flattens each time-varying parameter into
  per-time-knot scalars `{name}_{t}` (respecting `pin_initial_time_point` for
  cross-window continuity).

It is built by multiple inheritance (`class …(StateAndParameterESMDA,
TimeVaryingParameterESMDA)`) with an overridden `_one_step`; `isinstance(…,
StateAndParameterESMDA)` is `True`, so `run_esmda.py`'s state-history path picks
it up unchanged. Both state-bearing variants share
`StateAndParameterESMDA._augmented_state_update`, which builds `[state | params]`,
applies the (state-only) update, and splits it back.

**Config / selection:** `conf/esmda/smoother/state_and_dynamic.yaml`, chosen with
`esmda/smoother=state_and_dynamic` (pair with `params@prior_params=dynamic`,
`params@truth_params=dynamic_truth`). `num_time_points` interpolates from
`params.time_coords.num`.

---

## 2. Localization — state-only, two strategies, config-selectable

Localization is applied to the **state rows only**; parameter rows always receive
the global Kalman update. This is implemented with a per-row `localize_mask` in
[`localization/base.py`](../libs/data-assimilation/src/data_assimilation/localization/base.py):
a masked-out (parameter) row is forced to all-ones inflation, which is provably
the exact global update for that row (sharing the same ESMDA perturbation
realization as the localized state rows).

Two strategies, both expressed as an observation-error **inflation matrix**
`(N_aug, N_d)` (`1` = keep, `>1` = taper, `inf` = exclude), reusing one shared
`taper_inflation()` helper (Vossepoel et al. 2025, Eqs. 9–10):

- **`CorrelationLocalization`** — selects observations by ensemble correlation
  `|ρ|` between a row and a predicted measurement; correlation distance
  `d_c = 1 − |ρ|`, truncation `d_t = 1 − ρ_t`. Needs no coordinates.
- **`DistanceLocalization`** (new,
  [`localization/distance.py`](../libs/data-assimilation/src/data_assimilation/localization/distance.py))
  — selects by **physical Euclidean distance** between each state grid point and
  the sensor (metres / domain units — **not** grid-point count), excluding beyond
  `localization_radius` and tapering the rest. Requires geometry (`requires_coordinates
  = True`); the smoother supplies `row_coords` (per-row `x/y/z`, built to match
  `_flatten_state`'s order) and `obs_coords` (sensor `xyz`, tiled — the sensor is
  the innermost index of the observation vector). Optional `horizontal_only`.

Grid-block grouping (`block_grouping`, Vossepoel §3b) co-locates the `u/v/w`
state at a cell; parameters are kept in singleton blocks (moot, since masked).

**Config group:** `esmda/localization=none|correlation|distance`
([`conf/esmda/localization/`](../conf/esmda/localization/)), default `none`
(global update). Every smoother wires `localization: ${esmda.localization}`.
Override a strategy's fields on the CLI, e.g.
`esmda.localization.localization_radius=40`.

> Adding a strategy: subclass `BaseLocalization`, implement
> `inflation_factors(aug_dev, pred_obs_dev, row_coords=None, obs_coords=None)`,
> set `requires_coordinates` if it needs geometry, and add a
> `conf/esmda/localization/<name>.yaml` option.

---

## 3. Full-vector sensor error metric (energy score + vector RMSE)

The per-run sensor error in [`run_esmda.py`](../scripts/run_esmda.py) was changed
from a velocity-**magnitude** comparison to the **full `(u, v, w)` vector** — one
scalar per sensor per timestep:

- `velocity_vector_rmse` — `√(mean_s ‖⟨v⟩_ens − v_truth‖²)`, obtained as
  `√(Σ_c rmse_c²)` by calling the shared `compute_sensor_metrics` per component
  (so the time-aligning metric is reused unchanged).
- `velocity_vector_energy_score` — the multivariate CRPS generalization
  (`mean_m‖v_m − v‖ − ½·mean_{m,m'}‖v_m − v_{m'}‖`), reduced over sensors.

The change is confined to `run_esmda.py`: `compute_sensor_metrics`
(`plotting.py`) and `compute_sweep_metrics.py` are untouched, and the sensor
figures still draw the `|U|` magnitude series (a vector *error* is not a state to
plot). The streamed *field* RMSE (`state_metrics.vel_magnitude_rmse`) remains
magnitude-based.

---

## 4. The initial-condition fix (the important one)

**Symptom:** every localization mode — and even parameter-only vs.
state-and-parameter — produced statistically identical results.

**Root cause (two compounding reasons):**

1. **The analyzed initial condition was discarded.** `_analysis` re-forecast the
   posterior every iteration from the *pinned* `initial_state`, using only the
   updated parameters. The Kalman-updated state from `_one_step` was overwritten
   the next iteration and never used, returned, or carried to the next window.
   So localizing the (state) rows changed a quantity that was thrown away.
2. **The global parameter update is independent of the state.** In the global
   ESMDA update each augmented row updates via its own cross-covariance with the
   observations, so adding the state to the augmented vector does not change the
   parameter update — joint state+param ≡ param-only for the parameters.

The only visible variation between modes was the forward model's run-to-run
non-determinism (~`1e-4`): two *identical* runs differed by the same amount as
two *different* modes.

**Fix** (`_analysis` in `smoothing/esmda.py`): feed the Kalman-updated IC
forward. Each ESMDA iteration now warm-starts from the *current* IC estimate, so
the analyzed IC flows into the next iteration's Kalman update, the posterior
forecast, and (via the cross-window carry-over) the next window. It is surgical:
parameter-only variants return `None` for the state and keep the caller's pinned
IC (behavior unchanged); only the state-bearing variants feed forward.

**Verified effect** (pylbm/pylbm, localization off, same seed):

| difference in… | noise floor (run vs. itself) | state+param vs. param-only |
|---|---|---|
| posterior params | `5.9e-5` | **`1.98`** |
| posterior state  | `2.7e-5` | **`1.85`** |

i.e. state estimation now moves the answer ~5 orders of magnitude above noise.
Model error (different truth/assim solvers) amplifies it further, and localization
now genuinely shapes the state analysis.

**Caveat:** feeding the analyzed IC forward warm-starts the next forecast from a
Kalman-blended field, intentionally skipping a fresh spin-up after iteration 0.
The analyzed field must be a usable warm-start for the forward model (the
cross-window carry-over already assumes this). A rough analyzed field (large
model error) can make members diverge — the ensemble failure policy (resample)
catches them; watch the logs on a first real run. This replaces the old
"pin the IC" behavior described in the codebase guide (now updated).

---

## 5. `scripts/compare_localization.sh` — comparison driver

Runs `run_esmda.py` across **five modes** — `none`, `correlation`,
`distance_small`, `distance_large`, and a parameter-only baseline (the `dynamic`
smoother) — on an otherwise identical joint state + time-varying-parameter
config, then prints a comparison table (per-parameter RMSE/CRPS and held-out
validation-sensor vector RMSE / energy score, ≤4 decimals) and writes the
figures + `run_summary.yaml` per mode under `.temp/loc_<mode>/`.

Env-var knobs (with current defaults): `SIZE` (`test` tiny grid | `large`
domain x[-20,40] y[0,40] z[0,32], nz=4 in all sizes), `TRUTH_MODEL`/`ASSIM_MODEL`
(`pyudales`/`pylbm` — different solvers inject model error),
`NNUDGE_M`, `ENSEMBLE_SIZE`, `NPROC`, `NUM_STEPS`, `NUM_WINDOWS` (3),
`RADIUS_SMALL`/`RADIUS_LARGE` (distance radii, metres), `RHO_T`, `MAX_INFL_C/D`.
Window length is 30 s with 2 parameter knots per window.

---

## 6. Pre-existing breakage fixed along the way

The branch started from a mid-refactor `main` where `run_esmda.py` could not run
via the testable `run(cfg)` path. Fixed: missing `paths.base_results_dir`; an
`experiment_dir` Hydra-resolution crash; stale `esmda/smoother` names in the
tests (`parameter`/`time_varying` → `static`/`dynamic`); `run.truth_dir` wiring;
and the pyudales `nnudge_meters` vs. test-grid-height collision (`udales`/`cross`
cases).

---

## 7. Tests

- `tests/test_localization.py` — unit tests for the taper, the state-only mask,
  grid-block grouping, distance exclusion/taper/`horizontal_only`, the
  observation-coordinate tiling, and the state-row coordinate ordering (tied to
  `_flatten_state`).
- `tests/test_run_esmda.py` — e2e smoke tests for every mode incl. the new
  `state_and_dynamic` and a distance-localized case.

Run them via the project's pixi env:

```
pixi run --environment dev python -m pytest tests/test_localization.py -q
pixi run --environment dev python -m pytest tests/test_run_esmda.py -q
```

---

## 8. Files touched

| Area | Files |
|---|---|
| Smoother + geometry + IC fix | `libs/data-assimilation/src/data_assimilation/smoothing/esmda.py` |
| Localization | `localization/{base,correlation,distance}.py` |
| Vector sensor metric | `scripts/run_esmda.py` |
| Config | `conf/esmda.yaml`, `conf/esmda/localization/{none,correlation,distance}.yaml`, `conf/esmda/smoother/state_and_dynamic.yaml`, `conf/run_esmda.yaml`, `conf/paths.yaml` |
| Comparison driver | `scripts/compare_localization.sh` |
| Tests | `tests/test_localization.py`, `tests/test_run_esmda.py` |
| Docs | `docs/codebase_guide.md`, this file |
