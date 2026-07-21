# Atmospheric Forcing Across Backends — Capability Survey

> Working note (docs/temp): a scoping report on what atmospheric forcing each
> solver supports, what the Python wrappers currently expose, and what dynamic
> (time-varying) forcing would take. **No implementation** — reporting only.
> Verify against code before relying on any file:line reference.

## TL;DR

Every solver supports far more atmospheric forcing than the wrappers expose, and
**dynamic inflow forcing already works end-to-end in all three** via the ESMDA
per-window warm-start + `extrapolate` machinery. The gap is almost entirely
wrapper-side variable emission and namelist toggles, not missing solver
capability. **PALM is the best fit for dynamic forcing**: it has the richest
physics *and* the wrapper already ships the exact NetCDF machinery
(PIDS_DYNAMIC) its most powerful dynamic mode (offline nesting) needs.

## Capability matrix

| Forcing | LBM | uDALES | PALM |
|---|---|---|---|
| Inflow speed + direction | ✅ wrapped | ✅ wrapped (via nudging) | ✅ wrapped |
| Vertical shear profile | ✅ wrapped | ⚠️ columns only | ✅ wrapped (synthetic) |
| **Time-varying inflow** | ✅ wrapped (dynamic) | ✅ wrapped (dynamic) | ✅ wrapped (dynamic) |
| SGS closure knob | ✅ wrapped | solver only | ✅ wrapped (`km_constant`) |
| Geostrophic wind (ug/vg) | ❌ absent | ✅ solver only | ✅ init only |
| Coriolis | ❌ absent | ✅ solver only | ✅ solver only (`omega=0`) |
| Large-scale forcing / tendencies | ❌ absent | ✅ solver only | ✅ solver only |
| Subsidence | ❌ absent | ✅ solver only | ✅ solver only |
| Nudging (T/q) | ❌ absent | ✅ solver only¹ | ✅ solver only |
| Surface heat/moisture flux | ⚠️ ABL path only | ✅ solver only | ✅ solver only |
| Buoyancy / stratification | ✅ solver only | ✅ solver only | ✅ solver only |
| Offline/mesoscale nesting | ❌ absent | via driver sim | ✅ solver only² |
| Precursor/driver inflow | ❌ absent | ✅ solver only | (turbulent inflow) |

¹ uDALES nudging file already carries `thl/qt` columns — dynamic T/q forcing is
one step away. ² PALM offline nesting reuses the wrapper's existing
PIDS_DYNAMIC writer — lowest-effort major extension in the codebase.

## Per-backend detail

### LBM (`pylbm`)
- **Solver supports:** inflow speed/direction (`uini`, `udir`), vertical shear
  (`uvel_shear.dat`), time-varying inflow (`uvel_time.dat`, precomputed at
  startup), inflow turbulence (`lturb`), buoyancy/ABL (`iablvisc`, `istable`,
  `ablheight`; Boussinesq, vertical-only), surface heat flux (unstable ABL),
  SGS (Vreman/Smagorinsky).
- **Wrapped:** inflow speed/direction, shear profile, time-varying inflow, SGS.
- **Not wrapped:** buoyancy/ABL and inflow turbulence exist in Fortran but are
  only *read* by the wrapper, never set.
- **Absent entirely:** Coriolis / geostrophic (no code); pressure gradient
  `rhoa` marked "NOT USED".
- **Dynamics:** solver never re-reads forcing mid-run; per-window warm-start
  rollout is the only cross-time mechanism (already used).
- **Write hook:** `libs/pylbm/src/pylbm/forward_model.py:332` (`_apply_inflow_settings`).

### uDALES (`pyudales`) — richest namelist, thinnest wrapper
- **Solver supports:** geostrophic + Coriolis (`lcoriol`, `ug/vg`, `lprofforc`),
  large-scale tendencies + subsidence (`lstend`, `wfls`), nudging, surface
  heat/moisture fluxes (`wtsurf`, `wqsurf`, `thls`), free-stream/mass-flow
  forcing, `ltimedep` (4 ASCII-driven channels, interpolated at runtime),
  precursor/driver simulations (`idriver`).
- **Wrapped:** constant body-force pressure gradient (`dpdx/dpdy`), reference
  velocity, and inflow driven through **time-dependent nudging**
  (`ltimedepnudge` + `timedepnudge.inp`) — dynamic velocity works today.
- **Not wrapped:** Coriolis, geostrophic, subsidence, surface fluxes, driver.
  Nudging file already carries `thl/qt` columns → dynamic T/q is nearly free.
- **Gotcha:** `INFLOW_PARAM_NAMES` whitelist silently drops unlisted params.
- **Write hooks:** `utils/params_utils.py:181`, `utils/nudging_utils.py`,
  `forward_model.py:510`.

### PALM (`pypalm`) — best fit for dynamic forcing
- **Solver supports:** geostrophic, LSF, nudging, subsidence, surface fluxes,
  Coriolis, initial soundings, and **offline/mesoscale nesting via PIDS_DYNAMIC**
  (time-dependent `ls_forcing_*` boundary forcing interpolated in time).
- **Wrapped:** `ug_surface`/`vg_surface` init, synthetic shear profile,
  `km_constant`, and — already working — **time-varying turbulent inflow planes
  + LOD-2 warm-start**, both written into the same `_dynamic` NetCDF file.
- **Not wrapped:** Coriolis/thermal disabled by templates (`omega=0.0`,
  `neutral=.T.`); offline nesting `ls_forcing_*` not emitted.
- **Strategic insight:** offline nesting uses the *same* PIDS_DYNAMIC file and
  the *same* `switch_off_module` gating pattern the wrapper already uses for
  turbulent inflow. Adding it = new variable emitter + one namelist toggle.
- **Write hooks:** `forward_model.py:341` (`_apply_inflow_settings`),
  `utils/dynamic_driver_utils.py` (PIDS_DYNAMIC writer),
  `utils/warm_start_utils.py`, `direct_palm.py:46` (`INPUT_FILE_MAP` staging).

## Dynamic forcing: the common enabler already exists

Dynamic forcing rides one existing pathway, generic across backends:

1. **`params` xarray Dataset** with a `time` dim → detected by
   `is_time_varying_params` → written in each backend's `_apply_inflow_settings`.
2. **Per-window warm-start:** end-of-window state seeds the next window's IC
   (`state_input = posterior_state.isel(time=-1)`, `scripts/esmda/run_esmda.py:745`).
3. **Forcing hand-off:** `prior_sampler.extrapolate(...)` (AR(2) knots) seeds the
   next window's prior, re-written on the solver's absolute clock (pylbm
   `time_offset=nt0*dt` is the reference), `run_esmda.py:776-785`.

No changes to `BaseForwardModel`, `BaseEnsembleForwardModel`, ESMDA, or the
window loops are needed for scalar forcing — the base classes are already
generic over `params` contents.

## Recommended strategy (for a future implementation)

- **Ride the existing `params` + `_apply_inflow_settings` channel**, not a
  parallel system — inherits ensemble slicing, failure resampling, ESMDA
  estimability, and the window `extrapolate` hand-off; auto-satisfies the
  no-op-when-absent invariant.
- **Shared physical name, per-backend write site** (the `sgs_constant` pattern),
  but configure prior ranges *per backend* — forcing units differ across solvers.
- **A dedicated `conf/forcing/` group** is only justified when forcing carries
  non-scalar structure (profiles, fields, or a discrete type switch); prefer
  extending the existing `params` samplers first.

### Effort ranking to add real atmospheric forcing
1. **PALM offline nesting** — lowest effort, highest payoff; reuses PIDS_DYNAMIC writer.
2. **uDALES temp/humidity nudging** — one step from working (columns exist).
3. **uDALES Coriolis/geostrophic/surface fluxes** — namelist writes via `NamoptionsFile`.
4. **LBM buoyancy/ABL** — wire `iablvisc/istable/ablheight` via `_set_infile_value`.
5. **LBM Coriolis/geostrophic** — not feasible without new Fortran; absent entirely.
