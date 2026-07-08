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

[libs/pypalm/src/pypalm/__init__.py](../libs/pypalm/src/pypalm/__init__.py)
downloads the PALM source tree as a tarball from GitLab on first import and
runs `install_palm.sh` to produce the compiled binary
`palm_model_system/MAKE_DEPOSITORY_default/palm`. Unlike pylbm, **PALM does not
need to be recompiled when the grid changes** — `nx/ny/nz` are read from the
`_p3d` namelist at runtime.

Palmrun resolution priority:
1. `PALM_BIN` env var
2. `palmrun` on `PATH`
3. `$PALM_ROOT/bin/palmrun`
4. Auto-installed `palm_model_system/bin/palmrun`

Skip auto-install with `PYPALM_SKIP_AUTOINSTALL=1`.

The version is pinned by `PALM_VERSION` (default `master`; set the env var or
edit the module constant to pin a release tag like `v25.10`).

### Lazy-import invariant

`pypalm` is **lazy-imported**. All `pypalm.*` `_target_` blocks live
exclusively in [conf/model/pypalm.yaml](../conf/model/pypalm.yaml).
Composing a config with `model=pylbm` or `model=pyudales` never imports
`pypalm` and never triggers PALM's download/compile. This invariant is
asserted by a regression test:
`tests/test_hydra_config.py::test_palm_target_does_not_import_for_non_palm_composition`.

---

## 2. Class structure

### `ForwardModel`

[libs/pypalm/src/pypalm/forward_model.py](../libs/pypalm/src/pypalm/forward_model.py)

Subclasses `BaseForwardModel` from
[src/pyurbanair/base_forward_model.py](../src/pyurbanair/base_forward_model.py).
Key constructor args (all wired from Hydra via `conf/model/pypalm.yaml`):

| Arg | Purpose |
|---|---|
| `case_dir` | Source of template `_p3d` / `_topo` / `_static` files — copied to `INPUT/` at `__init__` |
| `stl_path` | STL geometry; rasterized to `_topo` via `stl_to_palm_topography` |
| `experiment_name` | PALM job identifier (default `urban_run`) |
| `ncpu` | MPI ranks; `npex=ncpu, npey=1` for inflow-outflow slab decomp |
| `nx, ny, nz, bounds` | Grid; written to `_p3d` at init; required for topography + warm-start |
| `simulation_time`, `output_frequency`, `spinup_time` | Written to `_p3d` (`end_time`, `dt_data_output`, `averaging_interval`) |
| `boundary_condition` | `"periodic"` (cyclic) or `"inflow_outflow"` (dirichlet/radiation + multigrid solver) |
| `nudging_config` | Inflow profile shape; default `power_law` with `alpha=0.25` |
| `verbose` | When `False`, stdout/stderr captured; failures surface the last 80 lines |

On construction, `__init__` copies case files to `INPUT/`, writes grid+time
into `_p3d`, applies boundary conditions (and calls
`_apply_processor_topology` for inflow-outflow), and rasterizes the STL into
`_topo`.

#### `run_single(state, params, sim_name)`

The single-run entry point called by `BaseForwardModel.__call__`.

- **Cold start** (`state is None`): PALM initialises from analytic profiles
  (`initializing_actions = 'set_constant_profiles'`).
- **Warm start** (`state` provided): calls `_apply_warmstart(state)`, which
  writes `init_atmosphere_u/v/w/pt` (LOD=2) into the `_dynamic` NetCDF via
  `write_warmstart_driver` and sets `initializing_actions = 'read_from_file'`.
  The initial velocity-perturbation kick is suppressed (`create_disturbances =
  .false.`) to avoid shocking the injected field — mirroring what PALM's own
  restart path does. **No SGS-TKE is carried** across windows; PALM re-derives
  it from the mean field.
- Calls `_apply_inflow_settings(params)`, then `_clean_output()`, then `run()`,
  then `_load_and_postprocess_state()`.
- `disable_spinup()` zeroes `spinup_time` and rewrites `end_time`; called
  automatically on warm-start windows so only the cold start pays the spinup
  cost.

#### `_apply_inflow_settings(params)`

Consumes `inflow_angle` and `velocity_magnitude` from `params`; both can be
time-varying or static:

- **Static path**: calls `disable_turbulent_inflow` (idempotent — prevents a
  stale `&turbulent_inflow_parameters` block from leaking across runs) and
  removes any leftover `_dynamic` file.
- **Time-varying path**: calls `apply_time_varying_inflow` from
  [`utils/dynamic_driver_utils.py`](../libs/pypalm/src/pypalm/utils/dynamic_driver_utils.py),
  which writes `inflow_plane_u/v/w/e/pt(time_inflow, z, y)` into the
  `_dynamic` NetCDF and flips `&turbulent_inflow_parameters
  switch_off_module=.false.` in `_p3d`.
- Both paths then write `ug_surface`/`vg_surface` (geostrophic-wind init) and
  `u_profile`/`v_profile`/`uv_heights` (the vertical shear profile prepended
  with a z=0 no-slip anchor).
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

For **time-varying inflow** (`inflow_angle`/`velocity_magnitude` have a `time`
dim), `apply_time_varying_inflow` builds a `_dynamic` NetCDF with
`inflow_plane_u/v/w/e/pt(time_inflow, z, y)`. The profile shape is evaluated
once and broadcast across time; the spinup plateau is prepended by
`_prepend_spinup_plateau` (t=0 values held constant for `[0, spinup_time]`).
PALM's `turbulent_inflow` module reads this file and linearly interpolates in
time between snapshots at the dirichlet boundary. The dynamic driver also
carries the `init_atmosphere_*` fields when warm-starting (the two mechanisms
coexist in one PIDS_DYNAMIC file sharing the same 0-based vertical axes).

---

## 4. Postprocessing — unifying vertical staggers

PALM writes `u`/`v` on `zu_3d` and `w` on `zw_3d`. The
`_load_and_postprocess_state` method in `forward_model.py`:

1. **Renames** dims: `zu_3d → z`, `zw_3d → zw` (and `zs_3d → zs` if present).
2. **Shifts coordinates** onto the physical domain: PALM's native NetCDF axes
   start at 0; `xmin`/`ymin`/`zmin` offsets from `bounds` are added so sensor
   coords from `conf/case/*/obs.yaml` resolve correctly (especially for
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
8. **Assigns a seconds-based `time` coord** (`0, dt, 2·dt, …`) matching
   pylbm/pyudales convention.

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

Two extra parameters let ESMDA absorb truth↔assim solver misspecification
(see [docs/esmda_model_error_parameters.md](temp/esmda_model_error_parameters.md)).
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
| [`dynamic_driver_utils.py`](../libs/pypalm/src/pypalm/utils/dynamic_driver_utils.py) | Time-varying inflow driver: `write_dynamic_driver_file`, `apply_time_varying_inflow`, `disable_turbulent_inflow`, `is_time_varying_params`. Writes the PALM PIDS_DYNAMIC NetCDF with `inflow_plane_u/v/w/e/pt`. |
| [`warm_start_utils.py`](../libs/pypalm/src/pypalm/utils/warm_start_utils.py) | Warm-start via PALM LOD=2: `build_init_atmosphere_dataset`, `write_warmstart_driver`. Merges `init_atmosphere_*` fields into the same PIDS_DYNAMIC file as the inflow planes, using 0-based PALM-native vertical axes (required by DRV0005 value checks). |
| [`clean_up_utils.py`](../libs/pypalm/src/pypalm/utils/clean_up_utils.py) | `clean_palm_output_dir` (wipes OUTPUT/MONITORING/RESTART, leaves INPUT). `clean_palm_input_dir` (optional, keeps `_p3d`/`_topo`). |
| [`compile_utils.py`](../libs/pypalm/src/pypalm/utils/compile_utils.py) | `compile_palm` — shells out to `palmbuild -c <config>`. No-op when `compile=False`. |
| [`ncpu_utils.py`](../libs/pypalm/src/pypalm/utils/ncpu_utils.py) | `derive_npex_npey(ncpu, nx_points)` — slab decomposition (`npex=ncpu, npey=1`); raises with a helpful list of valid values when `nx_points % ncpu != 0`. |

---

## 7. Config wiring — `conf/model/pypalm.yaml`

[conf/model/pypalm.yaml](../conf/model/pypalm.yaml)

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
- `boundary_condition: inflow_outflow` — triggers the multigrid solver +
  npex/npey slab decomposition.
- `compile: false` — PALM is compiled once at install time via
  `__init__.py`, not per-run.
- `failure: ${ensemble.failure}` — pulls the shared `resample_from_successes`
  policy from the inlined `ensemble:` block.

Select pypalm for forward or assimilation runs:
```bash
python scripts/run_forward_model.py model=pypalm
python scripts/esmda/run_esmda.py model@assim_model=pypalm model@truth_model=pylbm \
    esmda/smoother=static params@truth_params=static_truth params@prior_params=static
```

---

## 8. Known gotchas

### STL topography and the height-map limitation

The STL geometry is valid for pylbm (full 3D voxelization) and pyudales (3D
solid_c approach), but `stl_to_palm_topography` rasterizes it into PALM's 2D
height map via **downward ray casting** — one highest-intersection height per
(x, y) cell. For the Barcelona case, the STL has an elevated merged ground
surface. PALM's top-down rasterizer interprets the elevated ground as "one big
building": every cell in that region gets a large non-zero height, blocking
airflow over the entire base rather than only the actual buildings. Pylbm and
pyudales handle the same geometry correctly with their 3D approaches.

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

The `init_atmosphere_*` fields in the `_dynamic` NetCDF must use **0-based
PALM-native vertical axes** (`z[k] = (k+0.5)*dz`, `zw[k] = (k+1)*dz`) — not
the physical-bounds offset that `_load_and_postprocess_state` adds to the
returned state. `write_warmstart_driver` handles this by sampling the
(offset) state at the matching physical heights but labelling the file axis
0-based. Violating this causes PALM's DRV0005 check to abort.

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

Experiment configs live in
[examples/palm/](../examples/palm/), one directory per case:

- [examples/palm/xie_and_castro/](../examples/palm/xie_and_castro/) — the
  Xie & Castro 2008 benchmark geometry.
- [examples/palm/barcelona/](../examples/palm/barcelona/) — the Barcelona urban
  case. Each contains a `_p3d` namelist template that `ForwardModel.__init__`
  copies into `INPUT/` and edits.

These are the files referenced by `case_dir: ${geometry.palm_case_dir}` in
`pypalm.yaml`. The case bundle (`conf/case/{xie_and_castro,barcelona}/`) sets
`geometry.palm_case_dir` and `geometry.stl_path`.

---

## 10. Quick reference

| You want to change… | Look here |
|---|---|
| PALM version pinned | `PALM_VERSION` constant in [`__init__.py`](../libs/pypalm/src/pypalm/__init__.py) |
| Grid / bounds / time | `conf/case/<name>/` (domain + time groups) |
| Inflow profile shape (`alpha`) | `nudging_config.profile_config.alpha` in `pypalm.yaml` (or via `vertical_inflow_exponent` ESMDA parameter) |
| SGS knob | `sgs_constant` parameter prior in `conf/params/` (maps to `km_constant` m²/s — not dimensionless) |
| ncpu / processor topology | `ncpu` in `pypalm.yaml`; `derive_npex_npey` validates divisibility |
| Namelist key editing | [`utils/p3d_utils.P3DFile`](../libs/pypalm/src/pypalm/utils/p3d_utils.py) |
| Time-varying inflow driver | [`utils/dynamic_driver_utils.apply_time_varying_inflow`](../libs/pypalm/src/pypalm/utils/dynamic_driver_utils.py) |
| Warm-start (state injection) | [`utils/warm_start_utils.write_warmstart_driver`](../libs/pypalm/src/pypalm/utils/warm_start_utils.py) |
| Vertical stagger unification | `_load_and_postprocess_state` in [`forward_model.py`](../libs/pypalm/src/pypalm/forward_model.py) |
| Topography from STL | [`stl_to_palm.stl_to_palm_topography`](../libs/pypalm/src/pypalm/stl_to_palm.py) |
| Member clone (ensemble) | [`utils/forward_model_utils.create_new_forward_model`](../libs/pypalm/src/pypalm/utils/forward_model_utils.py) |
| Member failure / resample | `failure: ${ensemble.failure}` in `pypalm.yaml`; base class in [`base_ensemble_forward_model.py`](../src/pyurbanair/base_ensemble_forward_model.py) |
