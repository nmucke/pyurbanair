# pypalm — PALM LES Wrapper

Reference for AI coding assistants and developers. Read
[codebase_guide.md](codebase_guide.md) first for the shared forward-model
abstraction, ensemble orchestration, and Hydra config conventions. This doc
covers everything specific to the `pypalm` sub-library at
[libs/pypalm/src/pypalm/](../libs/pypalm/src/pypalm/).

---

## 1. What it wraps and how PALM is obtained

`pypalm` wraps [PALM (Parallelised Large-eddy Simulation Model)](https://palm.muk.uni-hannover.de),
a Fortran large-eddy-simulation solver for atmospheric and urban-canopy flows.
PALM supports turbulent-inflow / dirichlet-radiation boundary conditions suitable
for wind-environment studies; the wrapper drives it in that mode
(`boundary_condition: inflow_outflow`, multigrid pressure solver).

### Source acquisition

Importing pypalm never downloads or builds anything. `install_palm()` in
[libs/pypalm/src/pypalm/__init__.py](../libs/pypalm/src/pypalm/__init__.py)
runs before every PALM run and is a no-op once PALM is built. The first time,
it downloads the pinned release (`PALM_COMMIT`, the commit of tag `v25.10`) as a
tarball from GitLab and runs `install_palm.sh` against the pixi env, logging to
`libs/pypalm/palm_install.log`, to produce
`palm_model_system/MAKE_DEPOSITORY_default/palm`. A failed download or build
raises with the log tail and the next run retries; a tree from another commit
is replaced. Linux and macOS (osx-arm64) build the same way: the installer
runs with the pixi env first on `CMAKE_PREFIX_PATH` (so PALM's CMake finds the
env's FFTW and NetCDF, never e.g. Homebrew's), with `HOME` set to
`palm_model_system` (the installer otherwise writes `~/.palm/palmtest*.yml`),
and on macOS with Apple's linker (`apple_linker_flags`, through `LDFLAGS` and
Open MPI's `OMPI_LDFLAGS`, plus header padding for the `install_name_tool`
fix-up in `install_palm.sh`). Unlike pylbm,
**PALM does not need to be recompiled when the grid changes** — `nx/ny/nz` are
read from the `_p3d` namelist at runtime (PALM needs `nz >= 14`).

Palmrun resolution priority:
1. `PALM_BIN` env var
2. `palmrun` on `PATH`
3. `$PALM_ROOT/bin/palmrun`
4. `palm_model_system/bin/palmrun` built by `install_palm`

Both launch paths run through the shared
`pyurbanair.utils.solver_process.run_solver`: a failure carries the tail of
PALM's output, and no `mpirun`/`palm` process outlives its Python owner.

### Lazy-import invariant

`pypalm` is **lazy-imported**. All `pypalm.*` `_target_` blocks live
exclusively in [configs/model/pypalm.yaml](../configs/model/pypalm.yaml).
Composing a config with `model=pylbm` or `model=pyudales` never imports
`pypalm` and never triggers PALM's download/compile.

---

## 2. Class structure

### `ForwardModel`

[libs/pypalm/src/pypalm/forward_model.py](../libs/pypalm/src/pypalm/forward_model.py)

Subclasses `BaseForwardModel` from
[src/pyurbanair/base_forward_model.py](../src/pyurbanair/base_forward_model.py).
Key constructor args (all wired from Hydra via `configs/model/pypalm.yaml`):

| Arg | Purpose |
|---|---|
| `case_dir` | Source of template `_p3d` / `_topo` / `_static` files — copied to `INPUT/` at `__init__` |
| `stl_path` | STL geometry; rasterized to `_topo` via `stl_to_palm_topography` |
| `experiment_name` | PALM job identifier (default `urban_run`) |
| `ncpu` | MPI ranks; `npex=ncpu, npey=1` for inflow-outflow slab decomp |
| `nx, ny, nz, bounds` | Grid; written to `_p3d` at init; required for topography + warm-start. **`nx`/`ny` must be even for `periodic` runs** — the default `poisfft` solver rejects an odd grid-point count on a cyclic boundary (PALM `PAC0071`/`PAC0072`); pypalm raises at staging instead. |
| `simulation_time`, `output_frequency`, `spinup_time` | Written to `_p3d` (`end_time`, `dt_data_output`, `averaging_interval`) |
| `boundary_condition` | `"periodic"` (cyclic) or `"inflow_outflow"` (dirichlet/radiation + multigrid solver) |
| `nudging_config` | Periodic nudging driver + inflow profile shape: `enabled` (default `true`), `tnudge` (15.0 s), `nnudge_meters` (16.0 m), `profile_config` (default `power_law` with `alpha=0.25`, `z_ref=64` m). See §8. |
| `inlet_turbulence` | Switchable inlet turbulence (`enabled`, default `false`). `method: digital_filter` (default) imposes synthetic turbulent inflow planes; `random_disturbances` turns on PALM's own inflow disturbances. Absent (`None`) is a strict no-op. `inflow_outflow` only. See §8. |
| `random_seed` | Seed of PALM's random disturbances → `&initialization_parameters ensemble_member_nr`, an int in `[0, 2000]`. `None` (default) writes nothing. See §7. |
| `verbose` | When `False`, stdout/stderr captured; failures surface the last 80 lines |

On construction, `__init__` copies case files to `INPUT/`, writes grid+time
into `_p3d`, applies boundary conditions (and calls
`_apply_processor_topology` for inflow-outflow), and rasterizes the STL into
`_topo`.

#### `run_single(state, params, sim_name)`

The single-run entry point called by `BaseForwardModel.__call__`.

- **Cold start** (`state is None`): PALM initialises from the analytic inflow
  profile. Under `periodic` that is `initializing_actions =
  'set_constant_profiles'`; under `inflow_outflow`, `_reset_cold_init` writes
  the same cell-centre profile as LOD=1 `init_atmosphere_u/v/w/pt/qv` into the
  `_dynamic` NetCDF (merged into the time-varying driver when there is one)
  and sets `initializing_actions = 'read_from_file'`, because
  `set_constant_profiles` zeroes the first fluid level, inflow face included
  (§8 "Near-wall inflow zeroed by set_constant_profiles").
- **Warm start** (`state` provided): calls `_apply_warmstart(state)`, which
  writes `init_atmosphere_u/v/w/pt/qv` (LOD=2) into the `_dynamic` NetCDF via
  `write_warmstart_driver` and sets `initializing_actions = 'read_from_file'`.
  `u`/`v` come back on PALM's own staggered points (§8 "Warm-start grid
  contract"); **`w` does not**: PALM's `read_from_file` path sets `w = 0`
  after reading the driver, LOD=2 included (`init_3d_model.f90:593`), and
  the initial pressure solve rebuilds a divergence-free `w` from `u`/`v`.
  PALM then needs an initial field for every prognostic quantity it carries
  (`DRV0006`): `pt` unless `neutral`, `qv` whenever `humidity` is on, which the
  periodic nudging driver forces (see §8). Both are always written; `qv = 0`
  is exact because q stays identically 0 in these runs.
  The initial velocity-perturbation kick is suppressed (`create_disturbances =
  .false.`) to avoid shocking the injected field — mirroring what PALM's own
  restart path does. **No SGS-TKE is carried** across windows; PALM re-derives
  it from the mean field. **Nor is `w`**: the `read_from_file` init sets
  `w = 0` after reading (`init_3d_model.f90:593`) and the initial pressure
  solve then projects (u, v), so each window boundary carries a small jump
  (§8 "Warm starts under inflow_outflow").
- Calls `_apply_inflow_settings(params)`, then `_clean_output()`, then `run()`,
  then `_load_and_postprocess_state()`.
- `disable_spinup()` zeroes `spinup_time` and rewrites `end_time`; called
  automatically on warm-start windows so only the cold start pays the spinup
  cost.

#### `_apply_inflow_settings(params)`

Consumes `inflow_angle` and `velocity_magnitude` from `params`; both can be
time-varying or static. The driver is selected by `boundary_condition` first,
then by whether the params carry a `time` dim:

| `boundary_condition` | params | driver |
|---|---|---|
| `periodic` | static **or** time-varying | **nudging** — `_nudge` (NUDGING_DATA) + inert `_lsf` (LSF_DATA), see §8 |
| `inflow_outflow` | static **or** time-varying | dynamic driver / `turbulent_inflow` (static params as a constant schedule) |
| `periodic` + `nudging_config.enabled: false` | static | static path |

- **Nudging path** (periodic, default): calls `apply_nudging_driver` from
  [`utils/nudging_utils.py`](../libs/pypalm/src/pypalm/utils/nudging_utils.py),
  turns on the four `&initialization_parameters` switches PALM's nudging
  requires, and clears the inflow_outflow apparatus. `nudging_config.enabled:
  false` restores the old un-driven periodic staging.
- **Static path** (un-nudged periodic only): calls `disable_turbulent_inflow`
  (idempotent — prevents a stale `&turbulent_inflow_parameters` block from
  leaking across runs) and removes any leftover `_dynamic` file.
- **Dynamic-driver path** (every `inflow_outflow` run): static params become a
  two-knot constant schedule (`as_inflow_schedule`); then `apply_time_varying_inflow` from
  [`utils/dynamic_driver_utils.py`](../libs/pypalm/src/pypalm/utils/dynamic_driver_utils.py),
  which writes `inflow_plane_u/v/w/e/pt(time_inflow, z, y)` into the
  `_dynamic` NetCDF and flips `&turbulent_inflow_parameters
  switch_off_module=.false.` in `_p3d`.
- The `inflow_outflow` path also runs symmetric hygiene — remove stale
  `_nudge`/`_lsf` files and force the nudging switches off *if the template
  has them* — so a prior periodic run in the same experiment dir cannot leak
  the LSF apparatus into an inflow run. On a clean template the switches are
  left absent, so inflow staging is unchanged.
- **Digital-filter inflow** (`inlet_turbulence.method: digital_filter`)
  replaces both `inflow_outflow` rows: `apply_digital_filter_inflow` writes
  mean-plus-fluctuation `inflow_plane_u/v/w` for static and time-varying params
  alike and switches the `turbulent_inflow` reader on. See §8.
- **Random disturbances** (`method: random_disturbances`) are applied
  separately, in `run_single` *after* the warm/cold init block
  (`_apply_inlet_turbulence`), because both write `create_disturbances`. They
  compose with any driver row. See §8.
- All paths then write `ug_surface`/`vg_surface` (inert with `omega = 0`;
  initial condition only) and `u_profile`/`v_profile`/`uv_heights` (the
  vertical shear profile prepended with a z=0 no-slip anchor), so the run
  starts consistent with the initial nudging target. The cell-centre part of
  that profile is kept for the inflow_outflow cold-start LOD=1 driver.
- Model-error knobs are applied via `_resolve_profile_config` and
  `_apply_sgs_setting` (see §5).

#### `save_results` / `_clean_output`

- `save_results` delegates to `BaseForwardModel._save_results` (in-memory or
  on-disk depending on `results_dir`).
- `_clean_output` calls
  [`clean_palm_output_dir`](../libs/pypalm/src/pypalm/utils/clean_up_utils.py),
  which wipes `OUTPUT/`, `MONITORING/`, and `RESTART/` but leaves `INPUT/`
  intact so the staged `_p3d`/`_topo` survive between runs.

### `EnsembleForwardModel`

[libs/pypalm/src/pypalm/ensemble_forward_model.py](../libs/pypalm/src/pypalm/ensemble_forward_model.py)

Subclasses `BaseEnsembleForwardModel`. Only override is
`_create_new_forward_model`, which calls
[`utils/forward_model_utils.create_new_forward_model`](../libs/pypalm/src/pypalm/utils/forward_model_utils.py):
deep-copies the template model, copies the `INPUT/` tree into a fresh
per-member directory, and renames `_p3d`/`_topo` files to the new
`experiment_name`. This gives each member an isolated directory and prevents
parallel palmrun invocations from colliding on CWD or `fast_io_catalog`.

Failure policy is wired via `failure: ${ensemble.failure}` in `pypalm.yaml`.
When PALM diverges it terminates without writing a 3D output;
`_locate_3d_output` catches the missing file and raises `CalledProcessError`
so the `resample_from_successes` policy can replace the failed member from a
successful donor (see §8 — roughly 15% of members can diverge intrinsically).

---

## 3. Runtime config — the `_p3d` namelist

PALM's input is a Fortran namelist file (`<experiment_name>_p3d`). The wrapper
edits it in-place via
[`P3DFile`](../libs/pypalm/src/pypalm/utils/p3d_utils.py), which parses the
`&section ... /` blocks into a nested dict and rewrites them preserving existing
lines (structurally identical to pyudales's `NamoptionsFile`).

Key methods on `P3DFile`:
- `set_value(section, key, value)` — scalar; auto-formats booleans as
  `.true.`/`.false.`
- `set_string(section, key, value)` — wraps in single quotes for Fortran strings
- `set_array(section, key, values)` — formats a float iterable as
  comma-separated `%.7f`
- `write()` — rewrites the file; new keys are appended before the section `/`

Fields written at construction: `nx/ny/nz`, `dx/dy/dz`, `end_time`,
`dt_data_output`, `dt_data_output_av`, `averaging_interval`,
`bc_lr`, `bc_ns`, `psolver` (multigrid for inflow-outflow),
`npex/npey`, `topography`.

Fields written per-run (in `_apply_inflow_settings`): `ug_surface`,
`vg_surface`, `u_profile`, `v_profile`, `uv_heights`, `km_constant`,
`constant_flux_layer`, `initializing_actions`, `create_disturbances`.

### Inflow driver / `u_profile`

For **static inflow**, `_apply_inflow_settings` writes
`initialization_parameters / u_profile` and `v_profile` as `nz+1`-element
arrays (`z=0` no-slip anchor prepended) shaped by
[`utils/vertical_profile.build_profile_shape`](../libs/pypalm/src/pypalm/utils/vertical_profile.py)
(`uniform` or `power_law`).

For **every inflow_outflow run** (time-varying `inflow_angle`/
`velocity_magnitude`, or static ones as a constant `[0, simulation_time]`
schedule), `apply_time_varying_inflow` builds a `_dynamic` NetCDF with
`inflow_plane_u/v/w/e/pt(time_inflow, z, y)`. The profile shape is evaluated
once and broadcast across time; the spinup plateau is prepended by
`_prepend_spinup_plateau` (t=0 values held constant for `[0, spinup_time]`).
PALM's `turbulent_inflow` module reads this file and linearly interpolates in
time between snapshots at the dirichlet boundary. The dynamic driver also
carries the `init_atmosphere_*` fields — LOD=2 when warm-starting, LOD=1 on
an inflow_outflow cold start (the two mechanisms coexist in one PIDS_DYNAMIC
file sharing the same 0-based vertical axes).

---

## 4. Postprocessing — unifying vertical staggers

PALM writes `u`/`v` on `zu_3d` and `w` on `zw_3d`. The
`_load_and_postprocess_state` method in `forward_model.py`:

1. **Renames** dims: `zu_3d → z`, `zw_3d → zw` (and `zs_3d → zs` if present).
2. **Shifts coordinates** onto the physical domain: PALM's native NetCDF axes
   start at 0; `xmin`/`ymin`/`zmin` offsets from `bounds` are added so sensor
   coords from the `obs` block of `configs/case/*.yaml` resolve correctly (especially for
   `xmin < 0` inflow regions).
3. **Fills NaN with 0** in `u/v/w` — PALM writes NaN at topography-occluded
   cells (no-slip BC); leaving NaN would poison Kalman updates.
4. **Drops the lowest z level** (`isel(z=slice(1, None))`).
5. **Interpolates `w` from `zw` onto `z`** (linear, with extrapolation) and
   drops the `zw` dim — so all three velocity components share a single `z`
   axis for downstream aggregation and observation operators.
6. **Trims spinup outputs** (first `spinup_time / output_frequency` frames).
7. **Clips or pads** to the expected `simulation_time / output_frequency` count
   (PALM's adaptive timestep occasionally produces one fewer output; missing
   frames are padded by repeating the last).
8. **Assigns a seconds-based `time` coord** (`dt, 2·dt, …, simulation_time`),
   the `(0, simulation_time]` axis every backend uses.

A `_assert_combine_succeeded` guard checks that `u/v/w` are not identically
zero with no fill values — the sentinel for a missing `combine_plot_fields.x`
step that would otherwise produce a silent dead field.

---

## 5. Parameters consumed

Standard parameters (shared across backends):
- `inflow_angle` (degrees from +x axis, CCW) → `ug_surface`/`vg_surface` and
  `u_profile`/`v_profile` via `angle_to_velocity`
- `velocity_magnitude` (m/s) → same

### Model-error compensation knobs

Two extra parameters (and `roughness_length`, below) let ESMDA absorb truth↔assim solver misspecification
(see [docs/archive/esmda_model_error_parameters.md](archive/esmda_model_error_parameters.md)).
Both are no-ops when absent, so single-model runs are unaffected.

#### `vertical_inflow_exponent` → `profile_config` / `u_profile`

`_resolve_profile_config` in `ForwardModel` overrides the construction-time
`profile_config.alpha` (power-law shear exponent) with the per-member value of
`vertical_inflow_exponent` from `params`. This makes the inlet shear
ESMDA-estimable per member. When absent, the construction-time `alpha` (default
`0.25`) is used unchanged.

Write site: `initialization_parameters / u_profile` (via `P3DFile.set_array`)
and, for time-varying inflow, into the `inflow_plane_u/v` arrays in the
`_dynamic` NetCDF.

#### `sgs_constant` → `km_constant` (Option A proxy)

**Where `sgs_constant` comes from.** Two sources, in precedence order:

1. `sgs_constant` in the params Dataset (from the `configs/params/*.yaml` sampler) —
   used when ESMDA estimates or pins it.
2. `forward_model.sgs_constant` in the backend's own `configs/model/*.yaml` — the
   per-backend default.

Absent from both is a strict no-op: the solver's own closure/template value
stands. The per-backend default exists because the three backends' `sgs_constant`
are **different physical quantities** (uDALES/pylbm take a dimensionless
Smagorinsky-family constant; PALM takes an eddy diffusivity in m²/s), so a single
value in the shared params sampler cannot be correct for all three at once.


PALM's LES TKE closure has no Smagorinsky-style namelist multiplier (`c_0` is
hardcoded), so `sgs_constant` is mapped to `km_constant` — a **constant eddy
diffusivity in m²/s**. This replaces the prognostic SGS-TKE closure with a
constant-Km model, accepted purely as a bias-absorbing knob.

**This is a different physical quantity from the dimensionless Smagorinsky
constant used by pylbm and pyudales.** A uDALES↔PALM cross-model ESMDA needs
PALM-appropriate `sgs_constant` ranges in the prior config (not the uDALES `cs`
defaults, which are dimensionless ~0.1–0.2).

A fixed `km_constant` also forces `constant_flux_layer = .false.` because PALM
rejects the combination at startup (check_parameters PAC0149). This is a full
constant-Km regime switch.

Write site: `_apply_sgs_setting` in `ForwardModel` →
`initialization_parameters / km_constant` and
`initialization_parameters / constant_flux_layer` via `P3DFile.set_value`.

### Roughness (`roughness_length`)

z0 for momentum (m), shared with pyudales: from the params Dataset, else
`forward_model.roughness_length` (`flow.roughness_length`: null in
`common.yaml`, which leaves the `_p3d` alone; xie_and_castro sets 0.05).
`_apply_roughness` writes it to `&initialization_parameters roughness_length`
on every `_apply_inflow_settings`, so on every member and window. Without the
land/urban surface models every surface (ground, roofs, walls) is the default
surface type, so this one scalar is the roughness of all of them. A
model-config value equal to the `_p3d`'s is not written. Default runs are
unchanged: xie_and_castro's 0.05 equals its template, and barcelona and the
urbantales templates (0.1, while uDALES generates 0.05) keep their own values
through the null default (`generate_data.yaml` sets null too).

PALM applies z0 through Monin-Obukhov similarity between the surface and the
**first grid level, zu(1) = dz/2** (a half cell, unlike uDALES' true facet
distance). It needs `0 < z0 < 0.5·dz` (PAC0366, PAC0173); `_apply_roughness`
raises outside that. `sgs_constant` (`km_constant`) forces
`constant_flux_layer = .false.` (PAC0149), which switches off the wall model
and so z0: a z0 from params together with `sgs_constant` raises, while the
model-config z0 is then skipped with a log line, so a `sgs_constant` run keeps
working on any template. On a flat 2 m grid
(laminar inflow, 40 m fetch) z0 0.01 -> 0.2 lowers the first-level u from
1.72 to 1.45 m/s.

---

## 6. The `utils/` subpackage

[libs/pypalm/src/pypalm/utils/](../libs/pypalm/src/pypalm/utils/)

| Module | Summary |
|---|---|
| [`p3d_utils.py`](../libs/pypalm/src/pypalm/utils/p3d_utils.py) | `P3DFile` — parse/edit PALM `_p3d` Fortran namelist in-place. Mirrors pyudales `NamoptionsFile`. |
| [`dir_utils.py`](../libs/pypalm/src/pypalm/utils/dir_utils.py) | `PALMDirectoryPaths` dataclass + `get_palm_directory_paths` factory. Lays out `experiment_dir/{INPUT,OUTPUT,MONITORING,RESTART}`. |
| [`forward_model_utils.py`](../libs/pypalm/src/pypalm/utils/forward_model_utils.py) | `create_new_forward_model` — deep-copies a template `ForwardModel` and retargets its dirs to a fresh per-member directory. Used by `EnsembleForwardModel._create_new_forward_model`. |
| [`inflow_utils.py`](../libs/pypalm/src/pypalm/utils/inflow_utils.py) | `angle_to_velocity(angle_deg, wind_speed) -> (u, v)`. Mirrors pyudales; duplicated deliberately so pypalm has no runtime dependency on pyudales. |
| [`vertical_profile.py`](../libs/pypalm/src/pypalm/utils/vertical_profile.py) | `build_profile_shape(profile_config, heights, zsize)` — returns a dimensionless `s(z)` shape. Supports `"uniform"` and `"power_law"` (with `alpha`). |
| [`dynamic_driver_utils.py`](../libs/pypalm/src/pypalm/utils/dynamic_driver_utils.py) | Time-varying inflow driver: `write_dynamic_driver_file` (mean profiles or full planes), `apply_time_varying_inflow`, `enable_turbulent_inflow`/`disable_turbulent_inflow`, `is_time_varying_params`. Writes the PALM PIDS_DYNAMIC NetCDF with `inflow_plane_u/v/w/e/pt`. |
| [`inlet_turbulence_utils.py`](../libs/pypalm/src/pypalm/utils/inlet_turbulence_utils.py) | Switchable inlet turbulence: `apply_digital_filter_inflow` / `build_digital_filter_planes` (turbulent inflow planes from the shared generator in [`pyurbanair/utils/digital_filter.py`](../src/pyurbanair/utils/digital_filter.py)), `apply_inlet_turbulence` (random-disturbance namelist keys), `validate_inlet_turbulence`. Module docstring carries the PALM `file:line` evidence for why recycling / STG are not usable here. See §8. |
| [`nudging_utils.py`](../libs/pypalm/src/pypalm/utils/nudging_utils.py) | Periodic nudging driver: `apply_nudging_driver`, `write_nudging_data`, `write_inert_lsf_data`, `remove_nudging_files`. Writes the ASCII `NUDGING_DATA` + inert `LSF_DATA`. Shares the schedule builders with `dynamic_driver_utils`. See §8. |
| [`warm_start_utils.py`](../libs/pypalm/src/pypalm/utils/warm_start_utils.py) | Warm-start via PALM LOD=2: `build_init_atmosphere_dataset`, `write_warmstart_driver`. Merges `init_atmosphere_*` fields into the same PIDS_DYNAMIC file as the inflow planes, using 0-based PALM-native vertical axes (required by DRV0005 value checks). |
| [`clean_up_utils.py`](../libs/pypalm/src/pypalm/utils/clean_up_utils.py) | `clean_palm_output_dir` (wipes OUTPUT/MONITORING/RESTART, leaves INPUT). `clean_palm_input_dir` (optional, keeps `_p3d`/`_topo`). |
| [`compile_utils.py`](../libs/pypalm/src/pypalm/utils/compile_utils.py) | `compile_palm` — shells out to `palmbuild -c <config>`. No-op when `compile=False`. |
| [`ncpu_utils.py`](../libs/pypalm/src/pypalm/utils/ncpu_utils.py) | `derive_npex_npey(ncpu, nx_points)` — slab decomposition (`npex=ncpu, npey=1`); raises with a helpful list of valid values when `nx_points % ncpu != 0`. |

---

## 7. Config wiring — `configs/model/pypalm.yaml`

[configs/model/pypalm.yaml](../configs/model/pypalm.yaml)

```
name: pypalm
solver_name: palm
compile: false

forward_model:
  _target_: pypalm.forward_model.ForwardModel
  ...

prepare:
  _target_: pyurbanair.config.hydra_helpers.prepare_compile
  compile: ${..compile}

ensemble_model:
  _target_: pypalm.ensemble_forward_model.EnsembleForwardModel
  ...
  failure: ${ensemble.failure}
```

The `_target_` blocks are the **only** place in the codebase that name
`pypalm.*` classes. When a run uses `model=pylbm` or `model=pyudales`, Hydra
never resolves these targets and Python never imports `pypalm`. The invariant
is enforced by a test.

Key field notes:
- `ncpu: 14` — default for the Barcelona grid (`nx=225`; valid divisors include
  1, 3, 5, 9, 15, 25, 45, 75, 225; 14 is not listed as Barcelona's 225 is not
  divisible by 14, so check per-case).
- `boundary_condition: ${flow.boundary_condition}` — `inflow_outflow` (the
  `configs/common.yaml` default; until 2026-10 pypalm defaulted to `periodic`)
  triggers the multigrid solver + npex/npey slab decomposition. `periodic`
  (set by `case=xie_and_castro_periodic`) selects the nudging driver (§8).
  Every `${flow.*}` value is shared with pyudales; see §8 "Cross-model parity
  with pyudales".
- `outflow_damping:` — `factor` (default `1.0`, 1/s) and `width` (default
  `4.0`, metres from the outflow edge), written as `&initialization_parameters
  outflow_damping_factor` / `outflow_damping_width` on `inflow_outflow` runs
  only. Periodic runs never get them (any a template carries are zeroed);
  `outflow_damping: null` writes nothing. Staging raises on `factor < 0`,
  `width <= 0` or a width longer than the domain, and warns on a width under
  two cells. See §8 "inflow_outflow diverges at the outflow without damping".
- `outflow_buffer: ${flow.outflow_buffer}` — `inflow_outflow` only: metres of
  flat ground PALM adds downstream of `domain.bounds` (rounded up to whole
  cells; `ncpu` must divide the extended `nx`). `xie_and_castro` sets 20 m,
  and pyudales adds the same buffer. The topography beyond `xmax` is zeroed, the
  damping layer then sits inside the buffer, and the returned state is cropped
  back to `domain.bounds`; a warm start fills the buffer by repeating the
  state's last column. `null` changes nothing. Staging raises on `<= 0` or
  under `periodic`, and warns when the damping width exceeds the buffer. See
  §8.
- `nudging_config:` — `enabled` (default `true`; `false` restores un-driven
  periodic staging), `tnudge` (relaxation timescale, s), `nnudge_meters` (no
  nudging below this height), `profile_config` (vertical shear shape); all but
  `enabled` from `flow.nudging` / `flow.profile`. Only
  `profile_config` applies under `inflow_outflow`; the rest drive the periodic
  nudging path. See §8.
- `inlet_turbulence:` — `enabled` (`flow.inlet_turbulence.enabled`, default
  `true` since 2026-10; the periodic case sets `false`), `method`
  (`digital_filter`, the default, or `random_disturbances`), `initial_seed`.
  `digital_filter` takes pyudales' keys and values, both read from
  `flow.inlet_turbulence`: `intensity` (0.15), `length_scale_x/y/z`
  (24/12/12 m), `time_step` (0.5 s, must divide the window), `seed` (`null` →
  derived from the member name and `random_seed`).
  `random_disturbances` takes `dt_disturb` (5.0 s), `amplitude` (0.25 m/s), and
  the optional `begin`/`end` (grid points) and `level_b`/`level_t` (m) bounds
  of the perturbed region (`null` → PALM's own auto-derivation). Requires
  `boundary_condition: inflow_outflow`; staging raises otherwise. **`enabled:
  false` does not disable PALM's `turbulent_inflow` module** — see §8.
- `random_seed: null` — the seed of PALM's random disturbances; see below.
- `compile: false` — PALM is compiled once at install time via
  `__init__.py`, not per-run.
- `failure: ${ensemble.failure}` — pulls the shared `resample_from_successes`
  policy from the inlined `ensemble:` block.

Select pypalm for forward or assimilation runs:
```bash
python scripts/run_forward.py model=pypalm
python scripts/run_smoother.py model@assim_model=pypalm model@truth_model=pylbm \
    'smoothing.smoother=${smoother.static}' params@truth_params=static_truth params@prior_params=static
```

---


### `inlet_turbulence`: `initial_seed` and the random disturbances

`initial_seed` is independent of `enabled` and `method`. With `method:
digital_filter` the random disturbances stay off and only `initial_seed`
decides whether the cold-start kick fires. With `method: random_disturbances`
PALM has two distinct disturbance mechanisms, both gated by the single
`&runtime_parameters create_disturbances` flag:

| Sub-key | Mechanism | PALM site |
|---|---|---|
| `initial_seed` | one-off random kick on u/v at cold-start init | `init_3d_model.f90:1488` |
| `enabled` (with `random_disturbances`) | perturbations re-injected near the inflow all run, paced by `dt_disturb` | `time_integration.f90:966-989` |

`create_disturbances` is written unconditionally by the cold-start init path
(`_reset_cold_init`), so before this split `enabled: false` **still fired the
t=0 kick** — the knob only ever controlled the in-run perturbations. All four
combinations are now expressible:

| `initial_seed` | `enabled` | `create_disturbances` | `disturbance_energy_limit` | `dt_disturb` |
|---|---|---|---|---|
| true | false | `.true.` | default | default (huge) → no in-run |
| true | true | `.true.` | 0.01 | set |
| false | false | `.false.` | — | — |
| false | true | `.true.` | **0.0** | set |

The last row works because the initial kick requires `energy_limit /= 0`
(`init_3d_model.f90:1488`) while the in-run non-cyclic branch does not
(`time_integration.f90:976`) — so zeroing the limit suppresses only the kick. A
warm start forces the same suppression regardless of `initial_seed`, so the
injected field is not shocked.

`initial_seed` defaults to **true**: it is what breaks symmetry so turbulence can
develop from PALM's smooth analytic initial profile. With it false, transition
has to be tripped by the topography alone and develops later.

### `random_seed`: PALM's one seed

PALM has no seed key of its own. Its random disturbances are seeded by
`ensemble_member_nr` (integer, default 0, `modules.f90:638`), a key of
`&initialization_parameters` (`parin.f90:213`). Each init calls
`init_parallel_random_generator` (`init_3d_model.f90:822-826`), which gives
every grid column the start ID
`j*(nx+1) + i + 1 + 1E6*(ensemble_member_nr - 1000)`
(`random_generator_parallel_mod.f90:176-177`; the 1D variant at `:129`). The
same per-column streams then draw both the initial kick
(`init_3d_model.f90:1488-1490`) and the in-run inflow disturbances
(`time_integration.f90:966-989`), both through `disturb_field`
(`disturb_field.f90:126-139`).

This holds only for `random_generator = 'random-parallel'`, PALM's default
(`modules.f90:582`; also an `&initialization_parameters` key, `parin.f90:271`).
`'numerical-recipes'` always starts from the fixed `iran = -1234567`
(`modules.f90:657`, `disturb_field.f90:115-125`), and `'system-specific'`
uses Fortran's `RANDOM_NUMBER` (`:140-150`); neither reads the seed. Apart from
seeding, `ensemble_member_nr` only adds an `en-no:` tag to the run header
(`check_parameters.f90:248-249`, `header.f90:296-298`).

`random_seed` (in `forward_model`) writes that key:

- `null` (the default) writes nothing, so the `_p3d` is byte-identical to a
  run without the knob, and PALM uses 0.
- An int is written once to `&initialization_parameters` at construction.
  It must lie in `[0, 2000]`: the column ID is a 32-bit integer, and this keeps
  the offset within ±1E9. Out-of-range values, non-integers and a template
  that selects another `random_generator` raise before PALM starts.
- `0` reproduces the default run. Any other value gives a different
  realisation of both the kick and the inflow disturbances.
- It is one value per run. Ensemble members are copies of the staged model,
  so they all get the same seed (as today, when they all get 0). pyudales
  derives per-member seeds from the experiment name instead; pypalm does not.
- Every warm-started window re-initialises the generator from the same seed,
  as PALM does without the knob.
- It needs a disturbance to act on. With `inlet_turbulence.initial_seed` and
  `enabled` both false PALM draws no random number at all, so every seed gives
  the bit-identical run (checked: seeds 0 and 1 on a periodic run, max |Δu| =
  0 at 60 s). Construction then raises. Under `periodic` only `initial_seed:
  true` can carry the seed, and it only reaches cold starts.
- The digital-filter inflow (§8) has its own seed (`inlet_turbulence.seed`,
  else hashed from the member name); `random_seed` is mixed into that hash, so
  a replica with another `random_seed` also gets a new inflow realisation.

## 8. Known gotchas

### Nudging driver for periodic runs

Periodic (`bc_lr = bc_ns = 'cyclic'`) runs are **driven by PALM's nudging
scheme**, the counterpart of pyudales's periodic nudging path. Without it a
cyclic run is un-driven and decays: `ug_surface`/`vg_surface` and
`u_profile`/`v_profile` are written, but with `omega = 0` they are pure initial
conditions. This is *not* geostrophic forcing.

**How it works.** `apply_nudging_driver`
([`utils/nudging_utils.py`](../libs/pypalm/src/pypalm/utils/nudging_utils.py))
writes two ASCII files into `INPUT/`:

- `<name>_nudge` → `NUDGING_DATA`. One block per time snapshot, `# <time>`
  followed by rows `height tnudge u v w pt q`. PALM relaxes the
  *horizontal-mean* profile toward the u/v targets on the `tnudge` timescale,
  at every fluid grid point, interpolating the target linearly in time.
- `<name>_lsf` → `LSF_DATA`, physically **inert**. PALM requires
  `large_scale_forcing = .T.` whenever `nudging = .T.` (LSF0001), so this file
  exists only to satisfy that constraint. Its single surface row sits past
  `end_time`, which turns `lsf_surf` off (`LSF0012`, warning); otherwise
  `ls_forcing_surf` would overwrite `pt_surface` and `surface_pressure` with the
  file's zeros. Its profile half is one all-zero block per NUDGING_DATA time:
  it must share that time axis because of a PALM bug (below, #165). That keeps
  `lsf_vert` on, but with zero content, so `ug`/`vg` are set to 0 (felt only
  through Coriolis, off with `omega = 0`), `w_subs` is unused without
  `large_scale_subsidence`, and the large-scale `pt`/`q` tendencies add zero.
  Confirmed bit-for-bit (below). The nudging term is then the only large-scale
  forcing. If a periodic run ever turns Coriolis on, `ug = vg = 0` is no
  longer inert.

**u and v only.** The `w`/`pt`/`q` columns carry PALM's `-999999` sentinel in
every row, which switches nudging off for those quantities — PALM confirms with
`LSF0020: Initial profiles of u, v, from NUDGING_DATA are used`. uDALES also
relaxes `thl`/`qt` when `ltempeq`/`lmoist` are on (both off in our neutral runs
— no practical gap).

**Near-wall cutoff.** uDALES exempts the bottom `nnudge` levels outright; PALM
has no such switch, so `nnudge_meters` is emulated by writing a huge `tnudge`
(1e9 s) below the cutoff height, with a paired row at `cutoff ∓ ε` so the step
is sharp rather than linearly ramped. A cutoff above the top cell centre raises
(as in pyudales). The default, 16 m, is pyudales's and sits above most of the
Xie & Castro canopy (cubes 2.8-17.2 m). Below it the canopy is driven by the
shear layer; nudged, it is held to the target instead, with the steady
deficit set by drag × `tnudge`. At the old 4 m default the canopy slab-mean u
at z = 5 m was 1.59 m/s, against 0.51 m/s at 16 m and 0.53 m/s in uDALES at
16 m (periodic unit below, 6 m/s at 10°).

**Profile reference height.** `profile_config.z_ref` defaults to 64 m, as in
pyudales, so `velocity_magnitude` is the same wind in both backends. Unset, it
falls back to the domain top: at a 32 m lid that is a 19 % stronger wind than
pyudales's for the same parameter value.

**The constraint chain** (all PALM-fatal if violated) forces four
`&initialization_parameters` switches on together: `nudging`,
`large_scale_forcing`, `lsf_exception` (required for non-flat topography —
an upstream bypass slated for revision), and `humidity` (LSF0003).

Costs and limitations:

- `humidity = .T.` on all periodic runs: one extra prognostic equation,
  physically inert at q ≡ 0 with zero fluxes (`q_surface = 0`, no surface or
  wall water flux, q nudging off; a cold-start `q` output is exactly 0). A warm
  start must still supply `init_atmosphere_qv` (`DRV0006`); the driver writes
  zeros.
- **Passive scalars are unavailable** under the nudging driver — PALM forbids
  `large_scale_forcing` with `passive_scalar` (LSF0004). Staging raises a
  `ValueError` naming the conflict rather than letting PALM abort mid-run. If
  PALM-side pollutant dispersion is needed, run under `inflow_outflow`.
- `lsf_exception = .T.` disables an upstream guard; LSF-with-topography is not
  an upstream-supported combination. Our use is benign (the LSF content is zero;
  only the nudging term is active), but it is the feature's main external risk.
- **PALM's `nudge_ref` reads the nudging profiles on the LSF time axis
  (issue #165, fixed on our side).** `nudge_ref`
  (`large_scale_forcing_nudging_mod.f90:1457-1485`, called every step from
  `time_integration.f90:754`) interpolates the NUDGING_DATA profiles on
  `time_vert`, the LSF_DATA time axis, instead of `timenudge` (which `nudge`
  uses correctly), and writes them into `u_init`/`v_init`. Before the fix our
  LSF_DATA had only one time, past `end_time`, so `nt` ended at 0 and
  `nudge_ref` read `unudge(:,0)`/`vnudge(:,0)`, one column before the arrays
  (allocated `1:ntnudge`). When that memory was unmapped (heap layout, so
  ASLR) the run died with SIGBUS, about 1 run in 20 on the 30×40×16 Xie &
  Castro grid on macOS. LSF_DATA's profile half now shares NUDGING_DATA's time
  axis, so `time_vert == timenudge` and the read stays in bounds. The
  upstream report is drafted in the #165 PR.
  - **What it did to past results.** `u_init`/`v_init` enter the run only
    through the top Dirichlet boundary (`u_p(nzt+1) = u_init(nzt+1)`,
    `dynamics_mod.f90:1110`); Rayleigh damping would use them too, but
    `rayleigh_damping_factor` is 0 (PALM's default, never set by pypalm). On
    macOS the out-of-bounds column held zeros, so old periodic runs had a
    **u = v = 0 lid** instead of the nudging target (4.8 m/s at the top
    of the default periodic Xie & Castro run). Fixing it changes the 180 s mean
    profile by +1.7 m/s at z = 31 m, +0.2 m/s at 27 m, +0.03 m/s at 23 m and
    ≤ 0.003 m/s at or below 19 m (canopy included); the time-mean u at the
    z = 2 m sensors moves ≤ 0.01 m/s. Results that use the upper quarter of
    the domain (full-state metrics, profiles) need rerunning; results built
    from near-ground sensors barely change. This holds only for the macOS
    build: elsewhere the column may hold other values or crash.
  - **Inertness check** (scratch PALM build with `nudge_ref` patched to
    `timenudge`, same inputs): patched build with the old LSF_DATA (nudging
    only) and with the new one give bit-identical 3D and time-series output,
    for static and time-varying params, and so does the stock build with the
    new LSF_DATA.

**Escape hatch.** `nudging_config.enabled: false` restores the old un-driven
periodic staging exactly.

**Cross-backend meaning.** Periodic runs are nudging-driven in *both* pypalm
and pyudales, with the same relaxation (fluid slab mean, every level above the
cutoff, same `tnudge`, cutoff, shear profile and `z_ref` in the default
configs) and the same parameter interpretation: `inflow_angle`/
`velocity_magnitude` are "the mean wind the domain is held to". Both accept
time-varying params under periodic BCs; both interpolate the u/v components,
not angle and speed, linearly between knots. Remaining differences: PALM's lid
was a Dirichlet wall at the nudging target (`bc_uv_t = 'dirichlet'`),
uDALES's is free-slip (`BCtopm = 1`); switching PALM to `'neumann'` changed the
300-600 s mean u by ≤ 0.07 m/s at every level. Since 2026-10 the xie_and_castro
`_p3d` sets `bc_uv_t = 'neumann'` (§ "Cross-model parity with pyudales"). A
pyudales cold start begins from rest, a pypalm one from the target profile.

### Periodic runs: domain, spin-up and verification

Checked on `case=xie_and_castro_periodic` (one 80 m × 80 m repeat unit, 2 m
grid, 40×40×16, lid 32 m; see §9), `inflow_angle = 10°`, `velocity_magnitude =
6 m/s`, default nudging, one rank. The default `xie_and_castro` domain is not
periodic (a 25 m empty fetch plus two of the eight cube columns); use the
periodic case for cyclic runs.

- **Nudged mean wind.** Above the cutoff the slab-mean u sits 3-9 % below the
  target (4.07 vs 4.47 m/s at 21 m, 4.60 vs 4.85 at 29 m): the nudging has to
  supply the canopy drag, so a steady deficit remains. The lid holds the target
  exactly (`#165` fix; e.g. u = 5.95 vs 5.95 m/s at z = 33 m). After a change of
  wind (10→40°, 6→8 m/s over one 180 s window) the nudged levels follow 2-4°
  behind in direction (about `tnudge` of lag), and the speed deficit grows to
  0.3-0.7 m/s.
- **Spin-up.** A cold start begins at the target profile everywhere, so the
  canopy decelerates: the canopy slab-mean u reaches its stationary value
  (1.3-1.4 m/s) within ~40 s, the canopy kinetic energy within ~120 s.
  `xie_and_castro_periodic` sets `spinup_time: 180`, which pyudales needs
  from rest (§ "Cross-model parity with pyudales"). The first-level zeroing of
  `set_constant_profiles` (below) also hits periodic cold starts, but under
  cyclic BCs it is gone in ~20 s.
- **Turbulence.** The cube array trips the flow without any disturbance:
  resolved TKE 0.3-0.45 m²/s² at street level, peaking at 0.9-1.0 m²/s² at
  the canopy top (z = 13-17 m); canopy w² ≈ 0.3 m²/s².
- **Seeds.** With `initial_seed: true`, seeds 1 and 2 differ by 0.4 m/s after
  2 s and decorrelate within a minute (max |Δu| 4.4 m/s at 60 s), with the
  same slab-mean profiles (≤ 0.15 m/s apart over 300-600 s).
- **Warm starts.** Three 180 s warm windows after a 300 s cold one against an
  840 s continuous run: slab-mean u per window within 0.1 m/s at every level,
  and the 3-window sensor means within 0.26 m/s (the spread of single 180 s
  sensor means between realisations is 0.02-0.19 m/s). Each boundary still has
  a shock, because PALM zeroes the injected `w` (above): canopy w² falls from
  0.26-0.36 to 0.06-0.08 m²/s² in the first 2 s and is back in its range after
  15-30 s; canopy kinetic energy dips 4-7 %, mostly that lost `w`.
  Window-mean resolved TKE is 10-25 % below the continuous run's near the
  ground and 5-10 % at the canopy top. The stagger fix (§ "Warm-start grid
  contract") is a separate, larger error that is gone. So a PALM **truth**
  should be one continuous run (`forward.rollout_steps=0` with the whole
  horizon as `simulation_time`), not warm-started windows.
- **uDALES parity** (pyudales `closure: vreman`, `sgs_constant 0.24`, both at
  the default nudging, 300-600 s means; PALM: mean of three realisations).
  Slab-mean u agrees within 0.1 m/s from 3 to 25 m (PALM − uDALES: −0.08,
  −0.05, +0.03, +0.02, +0.03, +0.02, +0.09 at 3, 5, 9, 13, 17, 21, 25 m);
  PALM is 0.16-0.18 m/s faster at 29-31 m (its lid) and −0.10 vs +0.12 m/s at
  1 m. Resolved TKE agrees at the canopy top (0.74-0.90 vs 0.81 at 13 m,
  0.88-0.99 vs 1.03 at 17 m); PALM has more at street level (0.32-0.45 vs
  0.22 at 1 m) and less above the canopy (0.25-0.40 vs 0.60 at 29 m). 180 s
  sensor means agree within 0.35 m/s; above the canopy that is within the
  realisation spread, but at the two street-level sensors in the wakes the
  difference is 3-4 realisation std (e.g. u −0.71 vs −0.36..−0.44 m/s), a
  real model difference near the walls (wall model, SGS closure).
  uDALES needs ~120 s from rest; it runs ~1.5x faster on one rank.

### Cross-model parity with pyudales

Since 2026-10 a uDALES and a PALM run of the same case are configured to
simulate the same flow wherever the two solvers allow it. The shared values
live once, in `flow` (`configs/common.yaml`; the cases override it), and both
`configs/model/pyudales.yaml` and `pypalm.yaml` interpolate them. Three
setups, all on xie_and_castro with `params=static_truth` (10°, 5 m/s,
α 0.25):

```bash
# 1. inflow_outflow, laminar inflow
python scripts/run_forward.py model=pyudales params=static_truth flow.inlet_turbulence.enabled=false
# 2. inflow_outflow, turbulent inflow (the default); the same seed shares the realisation
python scripts/run_forward.py model=pypalm params=static_truth flow.inlet_turbulence.seed=7
# 3. periodic, nudged
python scripts/run_forward.py model=pypalm params=static_truth case=xie_and_castro_periodic
```

(the same overrides with the other `model=`; `model.forward_model.ncpu=5` for
both at nx = 60 with the buffer, or 5 for uDALES and 2-4 for PALM periodic).

| item | uDALES | PALM | status |
|---|---|---|---|
| boundary condition | `flow.boundary_condition` | same | aligned (pypalm defaulted to periodic before) |
| mean inflow | power law α 0.25, z_ref 64 m, at cell centres; u on the inflow face, v so its face mean is the profile | same profile through the `turbulent_inflow` planes; LOD=1 cold start | aligned: face time-mean u identical to < 0.001 m/s |
| inlet turbulence | shared digital filter, driver planes | same planes, `turbulent_inflow` reader | aligned; face u correlation 0.99 for the same seed, see below for what happens behind the face |
| seeds | `seed`, else hash of the member name | `seed`, else hash of the member name and `random_seed` | members `000`, `001`, … match; single runs differ (`999` vs `urban_run`), so set `flow.inlet_turbulence.seed` |
| interior forcing, inflow_outflow | none (`interior_nudging: false` since 2026-10) | none | aligned |
| periodic forcing | slab-mean nudging, tnudge 15 s, none below 16 m | same (huge tnudge below 16 m) | aligned |
| pressure gradient, Coriolis | dpdx = dpdy = 0, `lcoriol` off | `omega = 0` | aligned |
| top | free-slip u, v; w open (`BCtopm = 3`, forced by uDALES under inflow-outflow; 1 periodic) | free-slip u, v (`bc_uv_t = 'neumann'` since 2026-10); w = 0 | u, v aligned; w cannot be (PALM has no open lid outside nesting) |
| lateral | cyclic y | cyclic y | aligned |
| outflow | convective, 20 m flat buffer (since 2026-10) | radiation + damping (1/s, 4 m), 20 m buffer | different schemes, both outside the bounds |
| upstream ground | wall (STL ground extended over the fetch since 2026-10; before it was free-slip) | wall | aligned |
| walls, roofs, ground | log-law wall function, z0 = 0.05 m (`factypes` type 1) | MOST, `roughness_length = 0.05` m | same law and z0; first point 1 m (dz/2) in both |
| SGS | Vreman, c = 0.12 (0.24 before 2026-10-14) | Deardorff 1.5-order TKE | cannot be aligned, see below |
| advection, time | CD2, RK3, CFL 0.7, dt_max 1 s | ws-scheme (5th order upwind, since 2026-10; pw-scheme before), RK3, CFL 0.7, dt_max 1 s | different on purpose: uDALES has only CD2, and the undamped pw-scheme left grid-scale streaks; see "Grid-scale streaks" |
| pressure solver | FFT (cosine in x under inflow-outflow) | multigrid (inflow_outflow), FFT (periodic) | both converged; not a parity issue |
| initial state | rest (+0.01 m/s noise) | the profile (no kick, `initial_seed: false`) | different, covered by the spin-up |
| output | instantaneous u, v, w (+p) every `output_frequency`, labelled dt, 2dt, … | instantaneous u, v, w, same labels (uDALES used actual dump times before 2026-10) | aligned; uDALES u, v on faces (`xm`, `ym`), w on `zm`; PALM `xu`, `yv`, w interpolated to `z`, plus a ghost level at 33 m |
| solids | 0 | NaN, filled with 0 | aligned; voxelisation identical (0 of 2000 columns differ at 2 m on x = [-20, 80]) |

**Evidence** (`ncpu` 5 / 3, 420 s from a cold start, statistics over
120-420 s, i.e. the case's 120 s spin-up and 300 s window):

- *Inflow face*: identical time-mean u at every level in all setups; with
  turbulent inflow and the same seed the face TKE agrees to 2 %.
- *Upstream ground*: extending uDALES's ground over x < 0 lowered its
  time-mean u at z = 1 m by 0.10 m/s at x = -10 and 0.18 m/s at x = 0, from
  0.07 m/s above PALM to 0.03-0.05 m/s below it at x = -10.
- *Interior nudging off* (laminar): above-canopy array-mean u at 21 m went from
  0.22 to 0.03 m/s below PALM (25-31 m: from +0.08..+0.12 to -0.13..-0.07);
  the domain rms of the time-mean u difference is unchanged (0.54 m/s), being
  dominated by the canopy (below).
- *Free-slip lid in PALM*: time-mean u changes ≤ 0.03 m/s (laminar).
- *Outflow buffer*: see § "inflow_outflow diverges at the outflow"; domain rms
  difference 0.547 → 0.496 m/s (laminar).
- *What remained at c = 0.24* (turbulent, seeds 7 and 8): domain rms of the time-mean u
  difference 0.44-0.45 m/s against 0.09 (uDALES) and 0.11 m/s (PALM) between
  seeds. It sits in the canopy: the array-mean u at z = 1 m is 0.82 (uDALES)
  vs 0.37 m/s (PALM), and the resolved TKE in the array is 0.02-0.04 vs
  0.25 m²/s² near the ground. The injected turbulence decays faster in uDALES
  right behind the face (TKE at 9 m, x = -19: 0.17 vs 0.28 m²/s²; x = -1:
  0.12 vs 0.25). Above the canopy the array-mean u agrees within 0.23 m/s.
  With laminar inflow uDALES reaches a steady state (TKE < 0.001 m²/s²
  everywhere) while PALM's wakes shed (array TKE 0.16 m²/s² at 1 m).
  These TKE and wake numbers are from the pw-scheme. Much of the excess
  PALM TKE was grid noise, and with the ws-scheme laminar PALM is steady too;
  see "Grid-scale streaks".
  Periodic: slab-mean u within 0.17 m/s below 13 m and 0.08 m/s above; TKE
  agrees near the ground and PALM has 20-60 % less above 13 m.

**SGS closure.** uDALES runs Vreman with c = 0.12, the Vreman equivalent
of its own Smagorinsky constant 0.217; PALM runs its prognostic SGS-TKE
closure, whose only knob, `km_constant`, would replace it by a constant
viscosity. Until 2026-10-14 uDALES ran c = 0.24, because lower values
collapsed dt. That was backflow through uDALES' convective outflow while it
sat in the array's wake, gone with the 20 m buffer; why 0.12 and not the
literature 0.07, and what c does to the comparison, is in docs/pyudales.md §7
"Outflow backflow and the SGS constant". In short: c moves the resolved TKE
and spectra, but not the near-ground canopy mean (array-mean u at 1 m
1.33 m/s in uDALES for c = 0.12-0.24 against 0.87-0.88 in PALM with its
ws-scheme), so that gap is not an SGS-constant effect.

**Spin-up** (20-s blocks of the canopy-mean u and kinetic energy, cold
start): PALM starts from the profile and is stationary within 20-40 s in
every setup, ~60 s periodic. uDALES starts from rest: laminar inflow_outflow
2 % low at 120-140 s, 1 % at 140-160 s; turbulent ~3 % low at 120-140 s,
within the block noise from ~140-160 s; periodic 9 % low at 120-140 s,
stationary from ~140 s. Recommended `time.spinup_time`: 150 s for
inflow_outflow (the case's 120 s leaves uDALES's canopy u ~3 % low in the
first frames), 180 s periodic (`xie_and_castro_periodic` sets 180).

### Switchable inlet turbulence: digital-filter inflow planes

`inlet_turbulence.enabled` is the cross-backend inlet-turbulence switch.
`method: digital_filter` (the default) gives PALM the same synthetic turbulent
inflow as uDALES; `method: random_disturbances` keeps PALM's own disturbances.
Line numbers below refer to the vendored PALM 25.10
`packages/palm/model/src/`.

**How `digital_filter` works.** `apply_digital_filter_inflow` writes
`inflow_plane_u/v/w(time_inflow, z, y)` into the `_dynamic` driver: the mean
profile from `inflow_angle`/`velocity_magnitude`/`vertical_inflow_exponent`
(static or time-varying, spinup plateau included) plus Xie & Castro (2008)
digital-filter fluctuations, one plane every `time_step` s over PALM's
`end_time` plus two margin planes. It then switches `turbulent_inflow` on with
`turbulent_inflow_method = 'read_from_file'`. That reader is built for
precursor planes: at every Runge-Kutta substep it interpolates the planes
linearly in time and imposes u at the inflow face and v, w, e at the ghost
column (`turbulent_inflow_mod.f90:1006-1230`), and it corrects the mass flux
at the radiation outflow (`:476-536`). The fluctuations come from
[`pyurbanair/utils/digital_filter.py`](../src/pyurbanair/utils/digital_filter.py),
the generator pyudales' inlet uses (docs/pyudales.md §6.1), with its keys and
statistics: `sigma_u = intensity * |U_mean(z)|`, v' and w' at 0.7x, the plane
mean of u' removed per record, AR(1) time scale `length_scale_x /
velocity_magnitude`. For the same seed, grid and record grid the PALM planes
equal the uDALES inlet planes value for value
(`test_matches_the_udales_inlet_planes`), so a PALM truth and a uDALES model
can be driven by the *same* inflow realisation. The seed is
`inlet_turbulence.seed`, else a hash of the member name with `random_seed`
mixed in, so ensemble members and replicas differ.

Windows continue each other: the member's elapsed time is persisted to
`<experiment_dir>/inlet_turbulence_clock.json` (as in pyudales), each window
writes the slice of the member's turbulence history from that time on, and the
clock is reset when a model or ensemble member is built. `time_step` must
divide the window length. A warm-started window also gets a correct inflow face
on this path, because the reader re-imposes it every substep (contrast the
open warm-start issue in "Near-wall inflow zeroed by set_constant_profiles").
Not handled: a member substituted after a failure keeps its own clock rather
than the donor's (statistically harmless, unlike pyudales which copies it).

**Measured** (10°, 6.0 m/s, `z_ref` 64 m, α 0.25, 2 m grid, `ny` 40, `nz`
16, 600 s after 60 s spinup, frames from t = 100 s; empty domain x = -20..140 m
unless stated). Integral scales at z = 9 m from u'; the Taylor length uses the
local mean speed (3.7 m/s), whereas the generator converts `length_scale_x`
with `velocity_magnitude`.

| plane | mean-profile error (rms, z ≤ 25 m) | TI u / v / w at 9 m | T_int | L_y | spectral slope 0.08-0.4 Hz |
|---|---|---|---|---|---|
| inflow face (x = -20) | 0.5 % | 0.131 / 0.102 / 0.091 | 2.7 s | 10.6 m | -1.7 |
| 20 m fetch (x = 0) | 2.8 % | 0.115 / 0.108 / 0.076 | 4.1 s | 9.9 m | -3.4 |
| 80 m fetch (x = 60) | 5.6 % | 0.132 / 0.118 / 0.066 | 5.2 s | 7.6 m | -2.9 |
| 120 m fetch (x = 100) | 5.1 % | 0.148 / 0.117 / 0.072 | 6.0 s | 6.8 m | -2.6 |

- The face reproduces the planes (time correlation 0.99 with the written
  u-plane at 9 m; planes' own TI_u 0.134). TI_u at the face is 0.17 at 1 m and
  0.12 above 9 m, below the nominal 0.15 because the u' plane-mean removal costs
  energy with `length_scale_z` 12 m on a 32 m plane — exactly as on uDALES.
- Over the fetch the resolved turbulence neither dies nor blows up: TI_u at
  9 m dips to 0.115 within 20 m (the grid damps the smallest injected scales,
  slope steepens to -3.4) and then grows back as wall shear produces
  turbulence near the ground; TI_w decays from 0.09 to 0.07. The mean profile
  stays within 1 % above 5 m; near the ground it adapts to PALM's wall model
  (z = 1 m speed -10 % at 20 m fetch, -20 % at 80 m).
- A second seed (`random_seed: 17`) gives the same statistics (TI_u at 9 m
  0.126 vs 0.131 at the face, length scales within 10 %) and an uncorrelated
  realisation (u' correlation with the first < 0.03 at every plane); the 180 s
  means of the two differ by 0.12 m/s rms — the realisation noise.
- On xie_and_castro with the inflow moved upstream (x = -60..40, 50x40x16, two
  300 s windows, the second warm-started), the plane 25 m in front of the
  array (x = -20) has TI_u / TI_v / TI_w = 0.17 / 0.16 / 0.10 at 9 m, L_y 6 m,
  and turbulence up to the domain top (TI_u 0.11 at 17-23 m). The warm window
  keeps the face mean (4.04 m/s) and its fluctuations.

**`random_disturbances` is not an inflow turbulence.** It re-injects
uncorrelated per-grid-point u/v kicks of `amplitude` m/s into the strip
`[inflow_disturbance_begin, inflow_disturbance_end]` every `dt_disturb`
(`time_integration.f90:966-989`; off by default through `dt_disturb =
9999999.9`, `modules.f90:939`). Intensity and length scales are not
prescribed, `disturbance_level_b` cannot go below `zu(3)` = 5 m (PAC0153,
`check_parameters.f90:2698-2701`), and the auto strip
(`check_parameters.f90:2755-2768`) on xie_and_castro is x = 0-22 m, z = 5-9 m,
inside the first building row. In the SGS Round-2 PALM truth (that strip,
`dt_disturb` 5 s) the inflow face carried no fluctuation at all, and at x =
-10 m TI_u at 9 m was 0.18 but with L_y ≈ 1 m (grid noise) and TI_u < 0.04
above 17 m. Moving the strip upstream does not fix it: on the empty domain
with the strip at the inflow (`begin` 1, `end` 10, i.e. x = -18..0 m,
`level_b` 5 m, `level_t` 25 m, 0.25 m/s every 5 s) TI_u at 9 m is 0.02 at
x = 0 and 0.036 at x = 100 m, with L_y ≈ 1.3 m and T_int < 1 s: the
grid-scale kicks are dissipated before they organise, so 120 m of fetch is
not enough to develop an inflow turbulence comparable with uDALES'.

The disturbances *do* act: in the Round-2 pilot the `palm_smooth` (off) and
`palm_turb` (on) states differ by 0.24 m/s rms. The pilot's report showing them
identical came from its analysis script (`round2_pilot_v2.py` mapped both names
to the same `palm_truth` run), not from PALM.

**Neither PALM alternative fits pypalm's staging.**

- *Recycling* (`turbulent_inflow_method = 'recycle_*'`, the analogue of
  uDALES `iinletgen=1`) needs `initializing_actions = 'cyclic_fill'` or
  `'read_restart_data'` (`turbulent_inflow_mod.f90:337-343`, `TUI0006`), i.e. a
  precursor run plus restart plumbing, and `dx < recycling_width < nx·dx`
  (`:346-349`, `TUI0007`). It recycles toward the precursor's mean, so the
  estimated inflow would not reach the solver, and it doubles the cost.
- The *synthetic turbulence generator* (the same digital-filter family)
  requires, for idealized setups, `initializing_actions =
  'set_constant_profiles'` or `'read_restart_data'`
  (`synthetic_turbulence_generator_mod.f90:390-396`, `STG0004`) — which rules
  out pypalm's LOD=1 cold start (see "Near-wall inflow zeroed by
  set_constant_profiles") and every warm start — plus an `STG_PROFILES` file
  (`:412-417`, `STG0007`) and `random_generator = 'random-parallel'`
  (`STG0010`). It also excludes `read_from_file` inflow
  (`turbulent_inflow_mod.f90:323-328`, `TUI0005`), so time-varying inflow
  parameters could not reach the solver, and its mean is the static initial
  profile.

**`turbulent_inflow` is not a turbulence switch.** On every `inflow_outflow` run
(static or time-varying params, laminar or `digital_filter`) the module is the
*reader* of the `_dynamic` planes
(`turbulent_inflow_mod.f90:456-459`), the only thing connecting the
ESMDA-estimated `inflow_angle`/`velocity_magnitude` to the solver.
**`inlet_turbulence.enabled: false` therefore never touches
`turbulent_inflow`.**

**Precedence.**

1. `inlet_turbulence` absent, or `enabled: false` → nothing is written on a
   clean template (the teardown only resets disturbance keys that are already
   present). No clock file is written.
2. `method: digital_filter` → the planes replace both `inflow_outflow` driver
   rows in `_apply_inflow_settings`; the disturbance keys are handled as in 1
   (`initial_seed` still decides the kick).
3. `method: random_disturbances` → applied in `run_single` after the warm/cold
   init block, so it overrides the `create_disturbances` value written there.
   On a **warm start** `disturbance_energy_limit` is forced to `0.0`, which
   suppresses the one-off initial kick (`init_3d_model.f90:1488` is gated on
   `create_disturbances .AND. disturbance_energy_limit /= 0.0`) while keeping
   the in-run inflow perturbations alive via the `ELSEIF` branch at
   `time_integration.f90:976`. On a cold start the limit stays at PALM's
   default `0.01`. PALM's `HEADER` reports `Disturbance impulse (u,v) every :
   5.00 s` when they are on.
4. `periodic` + `enabled: true` → **rejected at construction** with a
   `ValueError`: there is no inflow face for either method
   (`turbulent_inflow_mod.f90:316-319`, `TUI0004`; `check_parameters.f90:2754`,
   `time_integration.f90:976-978`).

**Setup for a PALM truth in cross-model tests** — the defaults since 2026-10:
`inflow_outflow`, `digital_filter` inflow with the shared `flow.inlet_turbulence`
values, `initial_seed: false`, outflow damping inside the outflow buffer, and
at least ~25 m (two `length_scale_y`) of empty fetch before the first
building; vary `random_seed` for replicas. Set `flow.inlet_turbulence.seed` to
give uDALES the same realisation, or leave it null to keep it as noise.

### STL topography and the height-map limitation

The STL geometry is valid for pylbm (full 3D voxelization) and pyudales (3D
solid_c approach), but `stl_to_palm_topography` rasterizes it into PALM's 2D
height map via **downward ray casting** — one highest-intersection height per
(x, y) cell. For the Barcelona case, the STL has an elevated merged ground
surface. PALM's top-down rasterizer interprets the elevated ground as "one big
building": every cell in that region gets a large non-zero height, blocking
airflow over the entire base rather than only the actual buildings. Pylbm and
pyudales handle the same geometry correctly with their 3D approaches.

### Grid-scale streaks with the pw-scheme: xie_and_castro uses the ws-scheme

Until 2026-10 xie_and_castro ran `momentum_advec = 'pw-scheme'` (2nd-order
Piacsek-Williams, chosen in 55574ce to match uDALES CD2). PALM's time-mean |U|
then had steady spanwise streaks, 2-4 cells wide, from the inflow face to the
array (and one at y ≈ 47 m at z = 23 m), even with a laminar, y-homogeneous
inflow. The time-mean w alternated sign cell by cell in x and y in front of
the array.

**Cause.** Under central differences a 2Δx wave has group velocity -U: the
grid-scale wiggles that the building edges generate travel *upstream* and
stand between the face and the array. Nothing damps them. The pw-scheme has no
numerical dissipation, and in front of the array PALM's SGS TKE is nearly zero
(e = 0 on the inflow planes; 0.006-0.014 m²/s² at x = -19..-15, z = 9 m), so
Km ≈ 0.02 m²/s and the cell Reynolds number is ~400. uDALES runs the same CD2,
but its Vreman c = 0.24 (§ "Cross-model parity") reacts to the wiggles' own
gradients and damps them.

**Bisection** (laminar inflow, 2 m grid, x = [-20, 80] + 20 m buffer, 420 s,
statistics over 120-420 s; metric: rms over x = -18..-6 of the y-deviation of
the time-mean u, m/s; one ingredient changed at a time from the first row):

| variant | z = 1 | 9 | 15 | 23 m |
|---|---|---|---|---|
| pw, as in the comparison (kick on, npex 5) | 0.27 | 0.51 | 0.45 | 0.088 |
| npex 1 | 0.26 | 0.51 | 0.45 | 0.085 |
| no initial kick | 0.26 | 0.51 | 0.47 | 0.088 |
| no buildings | 0.000 | 0.000 | 0.000 | 0.000 |
| multigrid 12 cycles (default 4) | 0.26 | 0.51 | 0.44 | 0.085 |
| static inflow (`turbulent_inflow` off, face from `u_init`) | 0.28 | 0.50 | 0.41 | 0.081 |
| no outflow damping | 0.27 | 0.50 | 0.46 | 0.083 |
| no-slip lid | 0.28 | 0.50 | 0.46 | 0.087 |
| pw + `km_constant = 0.2` m²/s (diagnostic only) | 0.10 | 0.17 | 0.15 | 0.026 |
| **ws-scheme** | 0.09 | 0.05 | 0.03 | 0.005 |
| pw, turbulent inflow | 0.22 | 0.42 | 0.39 | 0.092 |
| ws, turbulent inflow | 0.11 | 0.06 | 0.09 | 0.076 |
| uDALES laminar / turbulent | 0.08 / 0.08 | 0.06 / 0.09 | 0.05 / 0.10 | 0.015 / 0.11 |

The inflow planes, the multigrid convergence, the decomposition, the kick,
the damping and the lid do not matter. Two side findings: with `mg_cycles =
-1` (iterate to `residual_limit` 1e-6) PALM aborts with PAC0265 because the
mean residual stalls at 1.8e-6. With 4 or 12 cycles the divergence after the
solve is the same (~1.5e-3), so it is not what drives the streaks. The
pw-scheme also biased the upstream y-mean u at 9 m by +0.2 m/s (3.21 against
the 3.02 m/s profile; ws 2.99, uDALES 3.03).

**Fix.** `geometries/xie_and_castro/_p3d` sets `momentum_advec` and
`scalar_advec` to `'ws-scheme'`, PALM's default (5th-order upwind-biased
Wicker-Skamarock). It is also what Barcelona runs. Its implicit dissipation
acts mainly at 2-4Δ. **This is a parity trade-off:** uDALES only implements
CD2 for momentum (`iadv_mom = 2`), so the advection schemes now differ. Yet
pw-scheme "parity" was only nominal: CD2 in uDALES is damped by a 3x-typical
Vreman viscosity, whereas pw in PALM is not damped at all. Matched by the
scheme names, the two models had unmatched grid-scale dissipation. With the
ws-scheme the two behave alike (comparison workflow, 2 replicas, `ncpu` 5,
uDALES vs PALM):

| | pw | ws |
|---|---|---|
| laminar: domain rms of time-mean u difference | 0.413 | 0.376 |
| laminar: PALM replica noise of time-mean u | 0.065 | 0.006 (uDALES 0.006) |
| laminar: field TKE distance | 0.133 | 0.008 |
| laminar: sensor mean speed distance | 0.578 | 0.484 |
| laminar: array resolved TKE at 1 m (uDALES 0.004) | 0.40 | 0.009 |
| laminar: profile u distance | 0.130 | 0.141 |
| turbulent: domain rms of time-mean u difference (replica noise 0.13-0.15) | 0.372 | 0.351 |
| turbulent: field / profile TKE distance | 0.173 / 0.140 | 0.076 / 0.041 |
| turbulent: sensor mean speed / sensor std distance | 0.536 / 0.214 | 0.455 / 0.155 |
| turbulent: profile u distance | 0.147 | 0.158 |
| turbulent: upstream streak metric at 9 / 15 m (above) | 0.33 / 0.37 | 0.06 / 0.11 |
| turbulent: upstream TKE at 9 m, x = -19..-7 (uDALES 0.15) | 0.27 | 0.13 |
| turbulent: array resolved TKE at 1 m (uDALES 0.04) | 0.59 | 0.14 |

So the "PALM's wakes shed with laminar inflow" and "injected turbulence
decays slower in PALM" differences recorded before (§ "Cross-model parity")
were mostly pw-scheme grid noise. With laminar inflow, PALM-ws is steady like
uDALES. The array-mean u at 1 m rises from 0.70 to 0.92 m/s (uDALES 1.32). The
remaining canopy difference is the SGS closure. Periodic (FFT, no inflow
face, never streaky) moves closer too: field mean-u distance 0.215 → 0.179
(replica noise 0.13-0.14), profile u 0.126 → 0.064, profile TKE 0.108 →
0.056, sensor mean speed 0.194 → 0.098 (now within the noise).
The ws-scheme costs ~1.2-1.5x the wall time (laminar 420 s run: 28 → 43 s
at npex 4).

Tests: `tests/pypalm/test_pypalm_advection.py` (staging, and an `integration`
run on a 30 x 20 x 16 grid where the upstream y-std of the time-mean u is
0.10-0.34 m/s with pw and ≤ 0.013 m/s with ws).

### inflow_outflow diverges at the outflow without damping (pw-scheme)

With the pw-scheme, xie_and_castro under `boundary_condition: inflow_outflow`
without damping diverges within ~10 s. Behind the last cube the flow reverses
at the radiation outflow boundary, and the pw-scheme adds no numerical
dissipation to absorb it. PALM's outflow damping layer fixes it, so
`configs/model/pypalm.yaml` turns it on by default (`outflow_damping`, §7).
The ws-scheme the case runs now is stable without damping: 360 s on the case
domain without buffer or damping, min u at the outflow -1.1 m/s. The damping
stays on by default because it keeps reversed flow out of the outflow column;
inside the buffer it changes nothing upstream.

PALM semantics (vendored source, `packages/palm/model/src/`):
- `outflow_damping_factor` and `outflow_damping_width` are
  `&initialization_parameters` keys (`parin.f90:253-254`), both defaulting to
  `0.0` (`modules.f90:975-976`), i.e. off.
- `width` is in **metres**: the layer covers `i*dx > nx*dx - width` with a
  `factor * sin^2` ramp up to `factor` at the outflow edge
  (`init_3d_model.f90:1712-1744`). `factor` is a relaxation rate (1/s): the
  prognostic equations add `- MAX(rdf, odf_x, odf_y) * (u - u_init)` to u, v
  and w (`prognostic_equations.f90:391`, `:474`, `:554`) and to pt (`:638`).
- PAC0351 (`check_parameters.f90:2603-2615`): width must be in
  `[0, (nx+1)*dx]`.
- PAC0352 (`check_parameters.f90:2621-2657`): a positive factor needs a
  `dirichlet/radiation` (or reverse) `bc_lr`/`bc_ns`; under cyclic BCs it
  aborts. Hence the periodic branch zeroes any keys.
- PAC0353 (`check_parameters.f90:2665-2668`): topography height must be zero
  at the inflow column. Cases with a building on the inflow edge cannot use
  damping (pypalm does not pre-check this; PALM aborts at start).

Evidence (2026-10 diagnosis): factor 1.0, width 4 m is stable for 600 s on
30x40x16, 50x40x16, 64x40x16 (domain x to 80 m) and 64x80x32, with inlet
disturbances on and off. The flow stays unsteady: temporal std of u ~0.22 m/s,
the same as the undamped run before it blows up. Factor 0.1 still diverges.

**The damping layer distorts the flow of interest on the case domain.** It
relaxes u, v, w toward `u_init`, i.e. the inflow profile, in the last two
columns (x = 36-40 m), which is the near wake of the second cube column
(x = 25-35 m). `outflow_buffer` moves the outflow and the layer into empty
ground behind `domain.bounds` (the case array is cropped at x = 40, inside
the gap before the next column at x = 45, so the geometry inside the bounds is
unchanged). Laminar runs (10°, 5 m/s, power law 0.25 with `z_ref` = the
32 m domain top, no inlet disturbances,
30 s spin-up, 540 s, `ncpu=1`), 540-s mean at the case sensors, buffered
40 m with the default damping inside the buffer as reference:

| sensor (x, y, z) | buffer 40 m u, v | case domain Δu, Δv | buffer 80 m Δu, Δv | buffer 40 m, no damping Δu, Δv |
|---|---|---|---|---|
| (10, 20, 2) | 3.45, 0.37 | +0.32, +0.02 | -0.03, -0.04 | -0.02, -0.02 |
| (10, 60, 2) | 3.17, 0.57 | +0.43, +0.13 | +0.07, -0.00 | -0.01, -0.05 |
| (20, 10, 2) | 0.17, 1.55 | -0.17, -0.01 | -0.03, -0.00 | -0.04, +0.03 |
| (20, 50, 2) | -0.83, -0.53 | -0.06, -0.89 | -0.02, -0.21 | -0.02, -0.20 |
| (30, 30, 2) | 2.66, 0.38 | **+1.21**, -0.20 | -0.05, +0.01 | -0.04, -0.09 |
| (30, 70, 2) | 3.26, -0.25 | **+1.27**, +0.05 | -0.01, -0.01 | +0.02, +0.01 |
| (2, 30, 2) | -0.14, 0.22 | +0.05, +0.01 | -0.02, -0.06 | -0.04, -0.01 |
| (20, 55, 2) | -0.06, -1.93 | -0.19, -0.23 | +0.02, -0.09 | +0.02, -0.14 |
| (17, 30, 2) | -0.48, 1.03 | +0.17, -1.29 | +0.01, -0.06 | +0.02, -0.10 |
| (10, 30, 20) | 5.46, 0.82 | -0.05, +0.02 | -0.01, -0.01 | -0.01, -0.00 |

Realisation noise, as the rms change of 180-s window means within one run, is
0.17-0.20 m/s. Buffers of 40 and 80 m agree within it (rms of window-mean
differences 0.11 / 0.20 m/s for u / v), and so does dropping the damping
inside a 40 m buffer, but then the outflow column still sees u down to
-3.3 m/s (1.06 m/s with damping), so keep the damping. The case domain is far
outside the noise at both x = 30 sensors (+1.2 m/s) and biased at x = 10
(+0.3-0.4 m/s): it is not a trustworthy truth. The time-mean wake at
x = 30-38, z = 1 m is 0.8 m/s with the buffer and 1.3-1.9 m/s on the case
domain. A 20 m longer upstream fetch (`xmin = -40`) slows the approach flow
at z = 1 m by 0.2-0.4 m/s at x = -20..-4 (the power-law profile is not in
equilibrium with the z0 = 0.05 m wall and keeps decelerating near the
ground), and shifts the sensors by an rms 0.31 m/s, about its own window
noise (0.32). The upstream sensor (2, 30) moves most (u -0.14 -> +0.54);
fetch matters when that sensor is used.

On the x = [-20, 80] case domain (2026-10) the bounds again end 5 m behind a
cube column. Without a buffer PALM's time-mean u at x = 78 m, z = 1-9 m was
1.0-1.5 m/s above uDALES's (the damping layer pulls the wake toward the inflow
profile); with 20 m of buffer in both models the difference is -0.2 m/s, and
20 vs 40 m changes uDALES by ≤ 0.06 m/s and PALM by its realisation noise
(array rms 0.10 m/s). `xie_and_castro` therefore sets `flow.outflow_buffer: 20`
for both backends.

The ~15% member divergence on this case (next section, noted 2026-06, 13-23%
across runs) is very likely the same outflow instability; not yet confirmed by
a rerun with damping.

### Near-wall inflow zeroed by set_constant_profiles

Before 2026-10, inflow_outflow cold starts used `initializing_actions =
'set_constant_profiles'`. The profile itself is right: `check_parameters.f90:672-697`
builds `u_init` from `u_profile`/`uv_heights` (`u_init(1)` = 2.089 m/s at
`zu(1)` = 1 m for 6 m/s at 10°). But `init_3d_model.f90:707-720`
(`set_constant_profiles`, `ibc_uv_b /= 1`, no LSM/USM spinup) then zeroes u
and v wherever `topo_flags` bits 20/21 are unset, and those need fluid at
k-1..k+1 (`topography_mod.f90:1575-1590`). So the whole k=1 level is zeroed,
boundary columns included. With `bc_lr = 'dirichlet/radiation'` the inflow
face is never re-imposed from `u_init` (`dynamics_mod.f90:1287-1288` copies
`u_p(:,:,nxl) = u_p(:,:,nxl-1)`), so u = v = 0 at z = 1 m on the inflow face
for the whole run (zero in every raw output frame).

The `read_from_file` path (`init_3d_model.f90:~566-610`) sets u = `u_init`
for every column, ghosts included, and masks only topography. pypalm now
writes the same cell-centre profile as LOD=1 `init_atmosphere_u/v/w/pt/qv`
(`warm_start_utils.build_init_profile_dataset`; LOD=1 still needs the
`x`/`xu`/`y`/`yv` dims, else DRV0003) and reads it with `read_from_file`.
`u_profile`/`uv_heights` stay in `_p3d`. Periodic cold starts (cyclic, so the
zeroed level refills from its neighbours within ~20 s) and warm starts (LOD=2, already
`read_from_file`) are unchanged. This changes inflow_outflow cold-start runs
by design.

Evidence (xie_and_castro, 10°/6.0 m/s, inlet disturbances on, against a uDALES
run with the same inflow): before the fix, the y-mean time-mean inflow |U|
ratio uDALES/PALM was 3.56 at z = 1 m and 1.45 at z = 2 m, and the fitted
profile exponent 0.43 (target 0.25). With the LOD=1 driver the ratio is
≈1.00 near the wall, the exponent 0.24-0.26, and the run is stable.
Through `run_forward.py` (540 s after 30 s spinup, `ncpu=1`, frames ≥ 30 s),
ratio uDALES/PALM of the y-mean time-mean |U|:

| x \ z (m) | 1 | 3 | 5 | 9 | 15 | 21 |
|---|---|---|---|---|---|---|
| −19 m | 1.01 | 1.02 | 1.01 | 0.97 | 0.98 | 1.01 |
| −5 m | 1.10 | 0.98 | 0.97 | 0.96 | 0.98 | 1.03 |

**Laminar inflow check** (2026-10, empty flat domain and the case domain,
inlet disturbances off, `initial_seed: false`): the inflow-face u equals the
prescribed cell-centre profile at every level z = 1-31 m, every y and every
output frame, to < 1e-6 m/s, through spin-up and for 180-540 s; v at the
first interior column is within 0.02 m/s of it. The `_p3d` carries
`create_disturbances = .false.`, so neither the kick nor the in-run
disturbances fire, and the empty domain stays steady (temporal std of u
< 0.004 m/s). With `initial_seed: true` the t=0 kick perturbs u by up to
0.34 m/s, mostly gone after 20 s; the face is restored after the first step.
In the empty domain the near-wall flow still decelerates downstream (z = 1 m:
2.07 m/s at the inflow, 1.70 m/s at x = 30): the power-law profile is not in
equilibrium with the rough wall. Static and time-varying params impose the
same face values; since static params now use the inflow planes too, the
paths are identical (before, the static path kept PALM's Neumann SGS-TKE at
the inflow instead of `inflow_plane_e = 0`, which moved u at z = 1 m by up to
0.06 m/s).

### Warm starts under inflow_outflow

Three bugs made warm starts unusable as truth; all fixed 2026-10:

- **Inflow face zeroed.** The LOD=2 reader fills `u` from `nxlu` (= 1) and
  copies it to `nxlu-1` = `nxl` (`netcdf_data_input_mod.f90:2339-2351`), so the
  ghost column `nxl-1` keeps the `u = 0` it was reset to, and
  `dynamics_mod.f90:1287-1288` copies that onto the inflow face every step
  (domain-mean u fell from 4.0 to 0.35 m/s in the warm window). No driver
  field reaches the ghost column, so every inflow_outflow run now imposes the
  face through the `turbulent_inflow` inflow planes, static params as a
  constant schedule (`turbulent_inflow_rff_bc` sets `u(nxl)` and `u(nxl-1)`
  every step).
- **One-cell shift.** The staggered fields were sampled one cell off (see
  "Warm-start grid contract"): each window boundary moved u one cell
  downstream and v one cell in y.
- **Initial kick re-enabled.** With `inlet_turbulence.enabled: false` and
  `initial_seed: true`, the knob rewrote `create_disturbances = .true.` after
  the warm start had turned it off.

Verification (laminar, 3 x 180 s windows vs one 540 s run, same config; `ncpu=1`
runs are bit-reproducible, so window 0 is identical): the inflow face stays
on the profile in every warm window. At the 10 sensors, warm-vs-continuous
window means differ by a mean |Δ| of 0.11 m/s on the case domain and 0.09 m/s
with `outflow_buffer: 40`, against 0.06 / 0.14 m/s between consecutive
windows of the continuous run (max 0.67 vs 0.83 m/s, at the wake sensor
(17, 30)). Plane-mean u profiles below z = 31 m agree within 0.012 m/s and there is
no drift across windows. What remains:

- The window boundary is a jump: the first warm frame differs from the last
  frame by 0.16 m/s (mean |Δu|, z <= 15 m), three times a normal 1 s change
  (0.29 before the shift fix), because PALM resets `w = 0` and re-projects.
- The top ghost level (z = 33 m, above the lid) reads 5.6 instead of 4.9 m/s
  in warm windows, which raises the domain mean u (including that level) by
  0.02-0.04 m/s; interior levels are unaffected.
- In warm windows `u_init`, the damping target and the top boundary value,
  is the horizontal mean of the injected field (`init_3d_model.f90:493-509`),
  not the inflow profile. With the damping in an empty buffer this did not
  show at the sensors.
- The last output frame is often a padded copy (PALM writes 179 of 180), so
  a window starts from the state 1 s before its predecessor ended.

### Member divergence in ensembles (~15%)

PALM's own divergence detection terminates the run (exit 0) before the first
`dt_data_output` dump, leaving no 3D output. `_locate_3d_output` converts this
into a `CalledProcessError` so the ensemble's `resample_from_successes` policy
can clone a successful donor. In practice roughly 15% of PALM members diverge
intrinsically (not parameter-driven). The resample policy masks this, but
downstream diagnostics (parameter RMSE, CRPS) should be read knowing that some
"member results" are donor copies — the posterior spread may be artificially
compressed. Set `failure: raise` to expose all divergences when debugging.

### Posterior can be worse than prior

When all sensors are placed in a region where PALM's predictions are
insensitive to the estimated parameters (e.g. deep in the sheltered wake of a
building), the predicted-vs-observed spread is 10× below the observation error
standard deviation. The Kalman update then effectively regularises toward the
prior rather than improving it. This is a sensor-placement issue, not a model
bug. Moving sensors to regions of higher flow sensitivity (approaching or
lateral faces) dramatically improves the posterior.

### Warm-start grid contract (DRV0005)

**Staggered u/v.** The LOD=2 reader puts file element `i` of `xu` at PALM's
`u(i+1)`, i.e. at x = (i+1)·dx (it starts at `nxlu = 1`,
`netcdf_data_input_mod.f90:2132,2339`), and likewise `yv` at `v(j+1)`. The
driver used to sample element `i` at i·dx, so every warm start shifted `u` one
cell along x and `v` one cell along y; the pressure solve then removed the
divergence this created. It now samples at (i+1)·dx. The `u` column at x = 0
(`v` row at y = 0) is not in the file; PALM copies it from its neighbour
(`:2344,2393`). This applies to both boundary conditions.

**`w` is not injected.** `read_from_file` zeroes `w` after reading
(`init_3d_model.f90:593`), so `init_atmosphere_w` is written but has no
effect (checked: replacing it changes no output bit). The initial pressure
solve rebuilds `w`; see §8 "Periodic runs" for the cost.


The `init_atmosphere_*` fields in the `_dynamic` NetCDF must use **0-based
PALM-native vertical axes** (`z[k] = (k+0.5)*dz`, `zw[k] = (k+1)*dz`) — not
the physical-bounds offset that `_load_and_postprocess_state` adds to the
returned state. `write_warmstart_driver` handles this by sampling the
(offset) state at the matching physical heights but labelling the file axis
0-based. Violating this causes PALM's DRV0005 check to abort.

The horizontal axes are only length-checked, but the staggered fields are read
from index 1: file element `n` of `init_atmosphere_u` lands in `u(i=n+1)`
(start `nxlu`) and of `init_atmosphere_v` in `v(j=n+1)` (start `nysv`,
`netcdf_data_input_mod.f90:2339,2388`), so the writer samples u at
`xmin + (n+1)*dx` and v at `ymin + (n+1)*dy`. PALM copies `u(1)` into `u(0)`
and `v(1)` into `v(0)`. `init_atmosphere_w` is read but discarded (`w = 0`).

### `PYPALM_FAST_IO_CATALOG` for cluster runs

palmrun copies the full build tree (~750 files) to `fast_io_catalog` before
each run. On networked scratch (BeeGFS) this is slow for many ensemble members.
Set `PYPALM_FAST_IO_CATALOG` to a node-local `/tmp` path; each member's
working dir will be isolated under it.

### `PYPALM_USE_DIRECT_RUN=0` fallback

The default (`PYPALM_USE_DIRECT_RUN=1`) bypasses palmrun + palmbuild and runs
the prebuilt `palm` binary directly (~16x faster on small grids). Set to `0` to
fall back to the historical palmrun path. On macOS the palmrun path's combine
step silently yields all-zero fields due to a dyld `rrtmg.so` load failure;
`_assert_combine_succeeded` catches this but the direct path is preferred.

---

## 9. Example configs

Case inputs live in [geometries/](../geometries/), one folder per case:

- [geometries/xie_and_castro/](../geometries/xie_and_castro/) — the
  Xie & Castro 2008 benchmark geometry. `case=xie_and_castro` crops it with an
  upstream fetch for inflow_outflow runs; `case=xie_and_castro_periodic` is
  one 80 m × 80 m repeat unit for periodic runs (§8).
- [geometries/barcelona/](../geometries/barcelona/) — the Barcelona urban
  case. Each contains a `_p3d` namelist template that `ForwardModel.__init__`
  copies into `INPUT/` and edits. The folder also holds the uDALES inputs;
  PALM copies only its `_p3d`/`_topo`/`_static`/`_dynamic` files.

`pypalm.yaml` reads the folder as `case_dir: ${geometry.case_dir}`. The case
bundle (`configs/case/{xie_and_castro,barcelona}.yaml`) sets
`geometry.case_dir` and `geometry.stl_path`.

---

## 10. Quick reference

| You want to change… | Look here |
|---|---|
| PALM version pinned | `PALM_COMMIT` constant in [`__init__.py`](../libs/pypalm/src/pypalm/__init__.py) |
| Grid / bounds / time | `configs/case/<name>.yaml` (`domain` + `time` blocks) |
| Inflow profile shape (`alpha`) | `nudging_config.profile_config.alpha` in `pypalm.yaml` (or via `vertical_inflow_exponent` ESMDA parameter) |
| SGS knob | `sgs_constant` parameter prior in `configs/params/` (maps to `km_constant` m²/s — not dimensionless) |
| ncpu / processor topology | `ncpu` in `pypalm.yaml`; `derive_npex_npey` validates divisibility |
| Namelist key editing | [`utils/p3d_utils.P3DFile`](../libs/pypalm/src/pypalm/utils/p3d_utils.py) |
| Time-varying inflow driver | [`utils/dynamic_driver_utils.apply_time_varying_inflow`](../libs/pypalm/src/pypalm/utils/dynamic_driver_utils.py) |
| Inlet turbulence on/off, method | `inlet_turbulence` in `pypalm.yaml`; [`utils/inlet_turbulence_utils.py`](../libs/pypalm/src/pypalm/utils/inlet_turbulence_utils.py); generator in [`pyurbanair/utils/digital_filter.py`](../src/pyurbanair/utils/digital_filter.py) |
| Outflow damping / empty outflow buffer | `outflow_damping`, `outflow_buffer` in `pypalm.yaml` (§7, §8) |
| Seed of the random disturbances | `random_seed` in `pypalm.yaml` (→ `ensemble_member_nr`, §7) |
| Warm-start (state injection) | [`utils/warm_start_utils.write_warmstart_driver`](../libs/pypalm/src/pypalm/utils/warm_start_utils.py) |
| Vertical stagger unification | `_load_and_postprocess_state` in [`forward_model.py`](../libs/pypalm/src/pypalm/forward_model.py) |
| Topography from STL | [`stl_to_palm.stl_to_palm_topography`](../libs/pypalm/src/pypalm/stl_to_palm.py) |
| Member clone (ensemble) | [`utils/forward_model_utils.create_new_forward_model`](../libs/pypalm/src/pypalm/utils/forward_model_utils.py) |
| Member failure / resample | `failure: ${ensemble.failure}` in `pypalm.yaml`; base class in [`base_ensemble_forward_model.py`](../src/pyurbanair/base_ensemble_forward_model.py) |
