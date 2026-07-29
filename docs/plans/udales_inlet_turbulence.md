# Plan: uDALES inlet turbulence via synthetic driver planes

**Status: IMPLEMENTED 2026-07-29** (steps 1–5; step 6, calibration, is still
open). Historical design record — the maintained reference is
[docs/pyudales.md §6.1](../pyudales.md); verify against the code before relying
on details here (see the `docs/plans/` disclaimer in CLAUDE.md).

Two things came out differently from the design below, both found during
implementation:

1. **`btime` is always 0, so `window_start_time` cannot come from `state.time`**
   (§3.2, §5). `ForwardModel._prepare_warmstart` deliberately strips the time
   coordinate so the restart's `timee` is written as 0 — uDALES otherwise dies
   with `timee >= runtime`. The state's own time coordinate is also rebased per
   window, so it is not a global clock. Implemented instead as
   `ForwardModel._elapsed_time`, a physical-seconds counter advanced by the
   window runtime after each successful run; it selects which slice of the
   seed-replayed AR(1) history the window gets, and the written record times
   always start at 0. `_apply_inflow_settings` therefore takes `warm_start`
   (which also fixes the spinup: `run_single` zeroes `spinup_time` only *after*
   the inflow settings are written).
2. **`prof.inp`/`lscale.inp` are written as zeros, not the mean profile** (§2.3).
   That is what today's inflow-outflow nudging path actually does — starting the
   flow from rest rather than stagnating a full-speed profile against the
   building walls before the pressure solver settles. `u0`/`v0` are still
   written to `&INPS` as reference scalars, and `dpdx`/`dpdy` zeroed.

Also worth recording: zeroing the plane mean of `u'` (§3.2) costs ~2% of the
target rms at length scales small against the inlet plane, but 40–50% once
`length_scale_y/z` approach the plane's own dimensions — relevant to step 6.

Goal: make `inlet_turbulence.enabled: true` work for pyudales — today it raises
(`forward_model.py`, `validate_inlet_turbulence`) because the Lund recycling
generator (`iinletgen`) is dead code in uDALES v2.2.0 (docs/pyudales.md §6.1).
The route below needs **no Fortran changes**: we feed uDALES's fully-wired
precursor/driver inlet (`BCxm=3` / `idriver=2`) with driver files synthesized in
Python — mean profile from the same ESMDA parameters the nudging path uses, plus
digital-filter turbulent fluctuations (Xie & Castro 2008, the same authors as
the benchmark case).

---

## 1. Verified solver facts (all in `libs/pyudales/u-dales/src/`)

| Fact | Evidence |
|---|---|
| `&DRIVER` namelist is declared: `idriver, tdriverstart, driverjobnr, dtdriver, driverstore, iplane, iangledeg, lchunkread, chunkread_size` | `modstartup.f90:145-148` |
| A namoptions file *without* an `&DRIVER` block still parses (only `iostat > 0` aborts; a missing group reads as EOF, `iostat < 0`) | `modstartup.f90:242-247` |
| `BCxm=3` auto-sets `idriver=2` and `linoutflow=.true.`; `BCtopm` forced to 3 | `modstartup.f90:831-835`, `:866-869` |
| The driver path is live end-to-end: `initdriver` (`program.f90:79`), planes read at startup (cold `modstartup.f90:1449-1457`, warm `:1883-1891`), `drivergen` interpolates in time, `xmi_driver` sets `u/um` at `ib` and `ib-1`, `v/w` at the `ib-1` ghost face (`modboundary.f90:677-720`), pressure step enforces the u-face exactly: `pup(ib,:,:) = u0driver*rk3coefi`, `up(ib,:,:)=0` (`modboundary.f90:1239-1247`) | — |
| File format: `unformatted` + `access='direct'` = **raw float64 stream, no record markers** (gfortran build uses `-fdefault-real-8`, `CMakeLists.txt:42`). Byte offset of record *n* is `(n-1) * record_bytes` regardless of compiler `recl` units | `moddriver.f90:520-940` |
| File names read (`idriver=2`): `tdriver_DDD.NNN`, `udriver_DDD.NNN`, `vdriver_DDD.NNN`, `wdriver_DDD.NNN`, where `DDD` = `driverid` (i3.3) and `NNN` = `driverjobnr` (i3.3). With the wrapper-enforced `nprocy=1`, `driverid = mod(myidy, nprocy) = 0` on every rank → **one file set**, read only by the x-rank owning the inlet (`ibrank`) | `moddriver.f90:786-872`, `initdriver:62` |
| Record layout: each u/v/w record is one inlet plane `(j = jb-jh … je+jh, k = kb-kh … ke+kh)`, **j fastest** (Fortran order). With cd2 advection `jh = kh = 1` (`modglobal.f90:575-582`), so the plane is `(jtot+2) × (ktot+2)` float64. `tdriver` records are a single float64 each (ascending times) | `moddriver.f90:833-871` |
| Time handling: `drivergen` linearly interpolates between the two bracketing records; it **hard-stops** (`stop 'Time in simulation has exceeded the inlet information…'`) if `timee > max(storetdriver)`. Cold start: `timee` starts at 0; warm start: at `btime` from the restart file | `moddriver.f90:219-241` |
| `iangledeg` rotates the planes solver-side every step (default 0) | `moddriver.f90:469-473`, `modstartup.f90:546-547` |
| Temperature/moisture/scalars may stay on profile BCs under `BCxm=3` (solver only warns); `lhdriver/lqdriver/lsdriver` stay false, no h/q/s driver files needed | `modstartup.f90:837-856` |
| Outflow convective velocity `uouttot` is recomputed from `u0av` every step in the default (`.not. luvolflowr`) branch — no extra config. (`ubulk` under the `idriver=2` cold-start branch reads a stale `uaverage` — `modstartup.f90:1487`, slabsum commented out — but `ubulk` is only consumed when `luvolflowr/lvolflowr` are on. Keep those off.) | `modboundary.f90:143-162` |
| `initdriver` (`idriver=2`, no `lchunkread`) allocates ~6 full-history plane arrays: `(jtot+2)·(ktot+2)·driverstore·8 B` each. `lchunkread` bounds this if it ever matters | `moddriver.f90:89-130` |
| `local_execute.sh` copies the experiment dir into the output dir and runs `mpiexec` there → driver files written next to `namoptions.<expnr>` are found via their bare relative names | `shell_scripts/local_execute.sh` |

## 2. Design decisions

1. **Synthetic planes, not a real precursor run.** A per-member precursor would
   double ensemble cost and disconnect the ESMDA parameters. Python generates
   the planes from the *same* parameters (`velocity_magnitude`, `inflow_angle`,
   `vertical_inflow_exponent`) the nudging path uses, so the estimated inflow
   keeps reaching the solver. A "precursor library" (record one periodic run,
   replay/rescale) stays a future option — the file interface is identical.
2. **Fluctuations must be space-time correlated** (digital filter), not white
   noise: the pressure projection + SGS dissipation kill uncorrelated noise
   within a few cells; correlated vortical structures survive and develop into
   real turbulence over an adaptation fetch.
3. **When enabled, volume nudging is turned off** (`lnudge=.false.`,
   `ltimedepnudge=.false.`, no `timedepnudge.inp`). Under `BCxm=3` the inlet no
   longer comes from `uprof`, and nudging toward mean profiles would damp the
   injected fluctuations on the `tnudge=15 s` scale. Time-varying inflow is
   instead encoded in the plane sequence itself (the driver is time-resolved).
   `dpdx`/`dpdy` are zeroed and `u0`/`v0` + `prof.inp`/`lscale.inp` still get
   the mean values (initial condition / reference), mirroring today's
   inflow-outflow handling.
4. **Inflow angle is baked into the planes in Python** (per-snapshot, so
   time-varying angle works); `iangledeg` stays unwritten (solver default 0).
5. **Deterministic per-member noise, regenerated from t=0 each window.** Seed
   derived from the member's experiment name. Each `run_single` regenerates the
   full sequence `[0, btime + runtime]` and writes all records; the AR(1)
   recursion from a fixed seed makes window *n*'s planes the exact continuation
   of window *n-1*'s — temporal continuity without persisting filter state.
   Generation is 2-D and cheap; revisit only if horizons grow very long.
6. **`driverjobnr: 998` default** (must be a 3-digit int; distinct from the
   experiment number to avoid any confusion with `idriver=1` outputs, which we
   never produce).
7. **`lchunkread` off by default**; expose it in the config for large cases.
8. **Strict no-op when disabled** (repo rule): absent / `{}` / `enabled: false`
   writes nothing — namoptions stays byte-identical, no driver files appear.

## 3. New modules

### 3.1 `libs/pyudales/src/pyudales/utils/driver_file_utils.py`

Binary I/O, no physics. Everything float64 (`<f8`; both build platforms are
little-endian — assert `sys.byteorder == "little"` at write time).

```python
PLANE_HALO_J = 1  # jh under cd2 advection (modglobal.f90:576)
PLANE_HALO_K = 1  # kh under cd2 advection

def plane_shape(jtot: int, ktot: int) -> tuple[int, int]:
    # (ktot + 2*PLANE_HALO_K, jtot + 2*PLANE_HALO_J) in C order;
    # .tofile() then emits j-fastest, matching the Fortran read loop.

def write_driver_files(experiment_dir, driverjobnr, times, u, v, w) -> None:
    # times: (n,) float64; u/v/w: (n, ktot+2, jtot+2) float64.
    # Writes tdriver_000.NNN + u/v/wdriver_000.NNN atomically
    # (tmp file + os.replace) so a killed member can't leave torn files.

def read_driver_files(experiment_dir, driverjobnr, jtot, ktot) -> ...:
    # Inverse, for tests and debugging.
```

Layout contract (unit-tested): file size == `n_records * record_bytes`;
`record_bytes = (jtot+2)*(ktot+2)*8` for planes, 8 for `tdriver`; record *n*
starts at `(n-1)*record_bytes`.

### 3.2 `libs/pyudales/src/pyudales/utils/inlet_turbulence_utils.py`

Mirrors the pylbm/pypalm module shape: `validate_inlet_turbulence(...)` +
`apply_inlet_turbulence(...)` orchestrator.

**Mean profile** — reuse existing pieces: `build_profile_shape`
(`vertical_profile.py`) with the resolved `profile_config` (α override from
`vertical_inflow_exponent` via `_resolve_nudging_config`), scaled by
`velocity_magnitude` and decomposed by `inflow_angle` (`angle_to_velocity` in
`inflow_utils.py`). Time-varying params are interpolated onto the driver time
grid; the spinup plateau prepends initial values, exactly like
`apply_time_varying_inflow` does for nudging.

**Fluctuations (phase 1 — simple, calibratable):**

- Target rms: `sigma(z) = intensity * |U_mean(z)|` for u; `0.7 * sigma` for v
  and w (rough neutral-ABL anisotropy; full Lund/Cholesky Reynolds-stress
  matching is a phase-2 option).
- Spatial correlation: per record, filter white noise on the (y, z) inlet grid
  with the Xie & Castro exponential kernel `exp(-pi*r / (2*L))`, length scales
  `length_scale_y`, `length_scale_z`. Periodic convolution in y (FFT) — this
  also makes the j ghost rows the natural periodic wrap, matching `BCym=1`.
- Time correlation: AR(1) across records,
  `b_n = a * b_{n-1} + sqrt(1 - a**2) * eta_n` with
  `a = exp(-pi * dtdriver / (2 * T))`, `T = length_scale_x / U_ref` (Taylor).
- Zero the plane-mean of `u'` per record so the instantaneous bulk inflow
  equals the mean-profile bulk (keeps the outflow correction from fighting the
  inlet); leave `v'`, `w'` means alone (w mean is 0 by construction).
- Staggering: u and v fluctuation rows are generated at cell-centre z levels
  (`zt`), w at face levels (`zm`); the half-cell y offset of v is ignored
  (irrelevant at synthetic-noise fidelity). Only rows `k = kb..ke`
  (u, v) / `kb..ke+1` (w) are consumed by `xmi_driver`; fill the k halo rows
  by edge-copy, with `u' = v' = w' = 0` and `U_mean` extrapolated flat at
  `k = kb-1` (below ground).
- Time axis: records at `t = 0, dtdriver, …` covering
  `[0, t_end + 2*dtdriver]` margin, where `t_end = btime + runtime_total`
  (cold: `runtime_total = spinup_time + simulation_time`, `btime = 0`; warm:
  `runtime_total = simulation_time`, `btime = float(state.time)` — the same
  value `update_warmstart_file_from_xarray` writes into the restart's `timee`
  record). Coverage is a hard constraint: `drivergen` stops the run if
  `timee` exceeds the last record.

**Namoptions writes (enabled path only):**

| Section | Key | Value |
|---|---|---|
| `&BC` | `BCxm` | `3` (replaces the `2` from `_apply_boundary_condition`) |
| `&DRIVER` | `driverjobnr` | config (default 998) |
| `&DRIVER` | `driverstore` | exact number of records written |
| `&DRIVER` | `dtdriver` | config `time_step` |
| `&DRIVER` | `tdriverstart` | `0` |
| `&DRIVER` | `lchunkread` / `chunkread_size` | only when configured |
| `&PHYSICS` | `lnudge`, `ltimedepnudge` | `.false.` (decision 3) |
| `&INPS` | `u0`, `v0`, `dpdx=0`, `dpdy=0` | as today's inflow-outflow path |

`NamoptionsFile.set_value` already creates missing sections, so templates
without `&DRIVER` need no changes. `idriver` itself need not be written
(`BCxm=3` forces it), but write `idriver = 2` anyway for greppability — it is
a declared key.

## 4. Config schema (`conf/model/pyudales.yaml`)

```yaml
  inlet_turbulence:
    enabled: false        # default: strict no-op, byte-identical namoptions
    intensity: 0.1        # u'_rms / |U_mean(z)|
    length_scale_y: 25.0  # m — integral length scales of the synthetic eddies
    length_scale_z: 25.0  # m   (default ≈ building height; calibrate, §8)
    length_scale_x: 50.0  # m — sets the AR(1) time scale via Taylor
    time_step: 0.5        # s — dtdriver; injected turbulence is band-limited
                          #     to ~1/(2*dtdriver) by linear interpolation
    driverjobnr: 998
    seed: null            # null → derived from member experiment name
    # lchunkread: false / chunkread_size: N   (optional, large cases only)
```

Cross-backend parity: `enabled` everywhere; `intensity` plays the role of
pylbm's `amplitude` / pypalm's `disturbance_amplitude` (unit-alike knobs,
per-backend names as with `sgs_constant`). Unknown keys: warn + ignore
(existing convention).

## 5. `forward_model.py` changes

- `INLET_TURBULENCE_KEYS`: extend with the new fields; delete
  `UDALES_INLET_GENERATOR_UNAVAILABLE` and the unconditional raise. New
  validation: `enabled: true` requires `boundary_condition='inflow_outflow'`
  (keep the periodic-BC error), `time_step > 0`, `intensity >= 0`, length
  scales > 0, `0 <= driverjobnr <= 999` and `!= int(experiment_name)`.
  Delegate to `inlet_turbulence_utils.validate_inlet_turbulence` (module owns
  its schema, as in pylbm/pypalm).
- `_apply_inflow_settings(params, window_start_time=0.0)`: gains the window
  start time. Branch:

  ```python
  if is_inlet_turbulence_enabled(self.inlet_turbulence):
      apply_inlet_turbulence(
          params=self.params, dirs=self.dirs,
          config=self.inlet_turbulence,
          nudging_profile_config=nudging_config["profile_config"],
          spinup_time=self.spinup_time,
          simulation_time=self._simulation_time,
          window_start_time=window_start_time,
      )
  elif use_nudging:  # existing path, unchanged
      ...
  ```

  The SGS write (`_apply_sgs_setting`) stays outside the branch, as today.
- `run_single`: compute `window_start_time = float(state.time.values) if state
  is not None else 0.0` and pass it. Note `_apply_inflow_settings` runs
  *before* `_snapshot_namoptions`, so the driver keys persist across the
  post-run `_restore_namoptions` — same lifecycle as the nudging writes.
- Driver files are regenerated (overwritten in place, same names) on every
  `run_single`, so stale planes from a previous window/donor can't leak.

Ensemble layer: no changes. `create_new_forward_model` deep-copies the
template and renames experiment-suffixed files — driver files carry the
`driverjobnr` extension, not the experiment number, so they are *not* renamed;
harmless because they are regenerated per run. Failure substitution
(`copy_carry`) also needs nothing: the substituted member regenerates its own
planes (its own seed) next window.

## 6. Tests

Unit (fast, no solver):

- `driver_file_utils` round-trip: write → `read_driver_files` → identical;
  byte-offset/record-size contract against hand-computed values.
- Generator statistics: per-z rms within tolerance of `intensity * U(z)`;
  plane-mean of `u'` ≈ 0; lagged autocorrelation ≈ `exp(-pi*lag/(2T))`;
  j ghost rows equal the periodic wrap.
- Continuation property: planes for `[0, t2]` sliced at `[t1, t2]` equal
  planes generated for a window starting at `t1` with the same seed.
- Namoptions: enabled path writes the §3.2 table (section auto-created);
  **disabled path byte-identical** (adapt `tests/test_udales_inlet_turbulence.py`
  — its `&INLET` dead-code documentation tests still hold and stay).
- Validator: periodic-BC raise, bad values, driverjobnr collision.
- Time-axis coverage: last record time ≥ `btime + runtime_total` for cold and
  warm windows.

E2E smoke (CI — local pytest collection crashes on this Mac, OMP #15):

- Forward run, Xie & Castro smoke shape, `inlet_turbulence.enabled=true`:
  completes; near-inlet u has nonzero temporal variance; `run.<expnr>.log`
  shows the "Inputs interpolated from driver tsteps" lines.
- Two-window warm-start rollout: second window runs (proves the `btime`
  offset; the drivergen hard-stop makes a coverage bug loud, not silent).

## 7. Docs

Same PR as the code (repo rule):

- `docs/pyudales.md` §6.1: retitle to "Turbulent inlet — synthetic driver
  planes"; keep the `iinletgen` dead-code table as history; document the new
  schema, the driver-file format, the nudging-off decision, and the ESMDA
  consequence (params now reach the solver through the plane means — the §6.1
  disconnect argument no longer applies). Update the Quick-reference row.
- `conf/model/pyudales.yaml` comment block (currently says "must stay false").
- Auto-memory `udales-driver-inlet-turbulence` already records the format;
  add a pointer to this plan.

## 8. Implementation order

1. `driver_file_utils.py` + unit tests (layout contract first — everything
   depends on it).
2. Generator (`inlet_turbulence_utils.py`) + statistical unit tests.
3. Namoptions writes + validator swap in `forward_model.py`; disabled-path
   byte-identity test green.
4. Manual single run on the smoke domain: confirm startup reads the planes
   (log lines), no `stop` at the namelist or time-coverage checks, inlet
   fluctuates. This is the point where any wrong assumption (halo widths,
   record order) surfaces — fix against `moddriver.f90` before proceeding.
5. Warm-start window chaining + e2e tests.
6. **Calibration runs** (separate follow-up, like the c_vreman study):
   downstream decay of resolved TKE vs. fetch on the Xie & Castro domain;
   tune default `intensity` / length scales; the classic failure mode is
   too-short correlation lengths (fluctuations die before the buildings), and
   the fix is longer scales, not more amplitude. Record findings in memory.

## 9. Risks / open questions

- **Adaptation fetch**: the upstream fetch before the building array is
  limited on this domain; synthetic turbulence needs some distance to become
  "real". If insufficient, escalate to the precursor-library variant (same
  file interface) or a small Fortran disturbance strip (out of scope).
- **`dtdriver` band-limit vs. file size**: fluctuation content above
  `~1/(2*dtdriver)` is lost to linear interpolation. Smoke shape is tiny; for
  production (e.g. 64×64 planes, 0.1 s, 300 s window ≈ 100 MB × 3 files per
  member) watch disk and the `local_execute.sh` copy into the output dir, plus
  the resident `driverstore` arrays (→ `lchunkread`).
- **dt-watchdog interplay**: turbulent inflow lowers the stable dt; the
  instability watchdog (`min_dt`, `patience`) may need retuning with the SGS
  constant findings (memory: `c_vreman` ≈ 0.25 on this case).
- **Observation operator / outputs**: unchanged — fielddump shapes and the
  staggered-grid handling don't depend on the inlet BC.
- Assumptions to re-verify in step 4 (cheap, loud failures): `jh=kh=1` under
  the case template's advection settings; gfortran `recl`/iolength in bytes;
  cwd of the mpiexec invocation.
