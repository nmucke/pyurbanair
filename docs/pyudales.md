# pyudales — Python wrapper for uDALES v2.2.0

pyudales adapts the uDALES large-eddy simulation (LES) / CFD Fortran solver to the
`pyurbanair` ensemble and data-assimilation stack. It exposes the same three-class
shape — [`ForwardModel`](../libs/pyudales/src/pyudales/forward_model.py),
[`EnsembleForwardModel`](../libs/pyudales/src/pyudales/ensemble_forward_model.py),
and a `prepare` step — that every backend provides (§3, §7 of
[codebase_guide.md](codebase_guide.md)). The Fortran code is uDALES v2.2.0, a
boundary-layer LES solver on a staggered C-grid.

---

## 1. Obtaining the Fortran source

[`__init__.py`](../libs/pyudales/src/pyudales/__init__.py) runs at first import. It
reads `.gitmodules` to locate the `libs/pyudales/u-dales` submodule entry, then:

1. Attempts `git submodule update --init --recursive libs/pyudales/u-dales`.
2. Falls back to a direct `git clone` of the URL extracted from `.gitmodules`.
3. Checks out the tag/branch recorded in `.gitmodules` (currently v2.2.0).

After the source is present the `__init__` runs `build_udales_macos.sh release`
(builds the Fortran binary into `u-dales/build/release/`) and
`build_preprocessing_macos.sh` (builds the IBM preprocessing tools including
`u-dales/tools/View3D/`). These are skipped when the corresponding
`CMakeCache.txt` already exists.

The top-level script path is in `LOCAL_EXECUTE_SCRIPT`
(`libs/pyudales/shell_scripts/local_execute.sh`), kept outside the submodule so
local modifications survive submodule re-initialisation.

**Known build constraint.** The uDALES build implements only the `cd2` (second-
order centred differencing) momentum advection scheme. Setting `iadv_mom=1` in
namoptions causes a crash at startup. SGS momentum dissipation is the only knob
available — `&NAMSUBGRID cs` under `lsmagorinsky=.true.`, `c_vreman` under
`lvreman=.true.` (see §4).

---

## 2. Class structure

### `ForwardModel`

[`forward_model.py`](../libs/pyudales/src/pyudales/forward_model.py) —
`ForwardModel(BaseForwardModel)`

Key constructor arguments (all wired from
[`conf/model/pyudales.yaml`](../conf/model/pyudales.yaml)):

| Argument | Purpose |
|---|---|
| `case_dir` | Source of the namoptions / STL template files (from `${geometry.udales_case_dir}`) |
| `experiment_name` | uDALES experiment number string (default `"999"`) |
| `ncpu` | Total MPI ranks; always decomposed as `nprocx=ncpu, nprocy=1` (x-strips only) |
| `simulation_time` | Window length in seconds; written to `&RUN runtime` |
| `spinup_time` | Prepended spin-up in seconds; effective runtime = `simulation_time + spinup_time` |
| `nx/ny/nz`, `bounds` | Domain overrides → `itot/jtot/ktot`, `xlen/ylen/zsize` in namoptions |
| `boundary_condition` | `"periodic"` or `"inflow_outflow"` (sets `BCxm`, `BCym`, `BCtopm`) |
| `closure` | SGS closure: `"smagorinsky"` / `"vreman"` → exclusive `&NAMSUBGRID` switches; `None` (default) keeps the template's. `"oneeqn"` is rejected — see §4 |
| `nudging_config` | Nudging tunables dict; see §6 |
| `inlet_turbulence` | Turbulent-inlet block. `None`/`false` (default) is a strict no-op; `true` switches the inlet to synthetic driver planes (`BCxm=3`) and turns nudging off. See §6.1 |
| `instability_check` | dt-watchdog config dict; see §7 |
| `precomputed_geom_dir` | Skip STL→IBM Fortran step by reusing prior geometry bundle |
| `verbose` | `False` (default) suppresses all subprocess stdout/stderr |

On construction, the case files are copied into a per-member experiment directory
and the namoptions file is renamed to `namoptions.<experiment_name>`. Domain
and runtime overrides are applied immediately via `NamoptionsFile`.

**Inflow settings are deferred.** `_apply_inflow_settings` is NOT called in
`__init__`. It runs inside `run_single` after any preprocessing (which would
otherwise wipe generated files) and with the final per-call parameters.

**Implemented abstract methods**

| Method | What it does |
|---|---|
| `run_single(state, params, sim_name)` | Applies inflow, runs uDALES, loads/postprocesses output |
| `_apply_inflow_settings(params, warm_start)` | Merges params, writes nudging files **or** driver planes, sets SGS and α |
| `save_results(state, sim_name)` | Calls `_save_results` from base class |
| `_clean_output()` | Deletes the output directory (calls `clean_output_dir`) |

### `EnsembleForwardModel`

[`ensemble_forward_model.py`](../libs/pyudales/src/pyudales/ensemble_forward_model.py) —
`EnsembleForwardModel(BaseEnsembleForwardModel)`

`_create_new_forward_model` delegates to
[`utils/forward_model_utils.create_new_forward_model`](../libs/pyudales/src/pyudales/utils/forward_model_utils.py),
which deep-copies the template `ForwardModel`, mirrors its experiment directory,
and renames all experiment-suffixed files to the new member name.

After a parallel run, `run_ensemble` iterates `_last_failure_substitutions`
(the donor-map set by the base class) and calls `copy_carry(donor_dirs,
failed_dirs)` so the failed member inherits the donor's warmstart restart — the
subgrid fields (e120, ekm, thl0, …) stay consistent with the warm-start state
that was resampled from the donor.

---

## 3. Staggered C-grid and grid utilities

uDALES solves on a staggered C-grid. The output `xarray.Dataset` carries **six
coordinate arrays**:

| Coordinate | Field(s) located there |
|---|---|
| `xt`, `yt`, `zt` | Cell centres — `pres`, scalars |
| `xm`, `yt`, `zt` | U-velocity face (`u @ xm`) |
| `xt`, `ym`, `zt` | V-velocity face (`v @ ym`) |
| `xt`, `yt`, `zm` | W-velocity face (`w @ zm`) |

The observation operator's `dim_mapping` for `solver_name="udales"` selects the
correct axis per variable (see §4 of [codebase_guide.md](codebase_guide.md)).

**Grid collocation.** To bring all variables onto a single regular grid use
[`utils/grid_utils.interpolate_grid`](../libs/pyudales/src/pyudales/utils/grid_utils.py).
It linearly interpolates:

- `u: (zt, yt, xm) → (zt, yt, xt)` via `interp(xm=ds.xt)`
- `v: (zt, ym, xt) → (zt, yt, xt)` via `interp(ym=ds.yt)`
- `w: (zm, yt, xt) → (zt, yt, xt)` via `interp(zm=ds.zt)`

`pres` (already at cell centres) passes through unchanged. The returned dataset
carries only `(time, zt, yt, xt)` coords. This is called by the neural-surrogate
training-data pipeline before saving each sample.

**Multi-CPU stitch.** When `ncpu > 1`, uDALES writes one fielddump slab per MPI
rank (`fielddump.<procx>.<procy>.<expnr>.nc`). The shell `gather_outputs.sh` is
meant to concatenate them but is a no-op when `nprocy=1` (the y-pass short-
circuits the x-pass), so multi-rank runs never produce a merged file. The Python
`_read_fielddump` method handles this: if a merged `fielddump.<expnr>.nc` exists
it is used; otherwise `_stitch_x_decomposition` concatenates the per-rank slabs.

**Why not `combine_by_coords`?** Because the staggered grid has two x-dimensions
(`xt` and `xm`). `combine_by_coords` sees a 2-D `xt × xm` tiling and broadcasts
every variable across the *other* x-axis, exploding memory and producing garbled
fields. The stitch instead iterates over `("xt", "xm")` and concatenates each
group of variables along the single x-dimension it actually carries, then merges
the two groups. See `ForwardModel._stitch_x_decomposition`.

**Stale fielddump padding.** A failed worker can leave partial fielddumps (including
files from an older grid) because the normal post-run cleanup is not reached.
`run_single` therefore calls `clean_output_dir` at the start of every attempt,
before any warmstart carry is restored into the execution directory. This keeps
retries from mixing stale and current output without deleting the authoritative
carry stored under the member's experiment directory.

---

## 4. Runtime configuration — `NamoptionsFile`

[`utils/namoptions_utils.py`](../libs/pyudales/src/pyudales/utils/namoptions_utils.py)
provides `NamoptionsFile`, a stateful parser/editor. Usage pattern throughout the
codebase:

```python
namoptions = NamoptionsFile(namoptions_path)
namoptions.set_value("SECTION", "key", value)
namoptions.write()
```

It preserves formatting, comments, and section order. Key namoptions sections used
by the wrapper:

| Section | Notable keys written by the wrapper |
|---|---|
| `&RUN` | `runtime`, `trestart`, `lwarmstart`, `startfile`, `nprocx`, `nprocy` |
| `&DOMAIN` | `itot`, `jtot`, `ktot` |
| `&INPS` | `xlen`, `ylen`, `zsize`, `u0`, `v0`, `dpdx`, `dpdy`, `stl_file`, `gen_geom`, `geom_path` |
| `&BC` | `BCxm`, `BCym`, `BCtopm` |
| `&INLET` | **nothing** — the wrapper never writes this section. `iinletgen` is *not* a declared key and writing it aborts the run (§6.1) |
| `&PHYSICS` | `lnudge`, `nnudge`, `tnudge`, `ltimedepnudge`, `ntimedepnudge` |
| `&NAMSUBGRID` | `lsmagorinsky`/`lvreman`/`loneeqn` (closure switches) and `cs` **or** `c_vreman` — whichever the active closure reads (see below) |

**SGS closure — selected by config, constant follows automatically.**
`&NAMSUBGRID` carries one constant per closure and uDALES reads only the one
belonging to the closure that is on (`u-dales/src/modsubgrid.f90`, subroutine
`closure`):

| Closure | Switch | Constant | Where it enters |
|---|---|---|---|
| Smagorinsky | `lsmagorinsky=.true.` | `cs` | seeds `csz`, hence the mixing length |
| Vreman (2004) | `lvreman=.true.` | `c_vreman` | `ekm = c_vreman*sqrt(max(bb/aa, 0.))` |
| One-equation | `loneeqn=.true.` | `cm`/`ce1`/`ce2` — **not** in the namelist | prognostic SGS TKE |

The closure itself is chosen by the `closure` constructor arg
(`conf/model/pyudales.yaml`). `_apply_closure` writes **all three** switches — the
chosen one `.true.`, the others `.false.` — so the active closure is fully
determined by the config rather than by whatever the case template shipped. Only
the keys uDALES declares in the NAMSUBGRID namelist may ever be written: an
undeclared key aborts the namelist read with `stop 1`. `closure: null` (the
Python default) skips the write entirely, so a run that doesn't ask for a closure
is byte-identical.

`"oneeqn"` is **recognised but rejected** with an explanatory `ValueError`. It
initialises (and injects at the inlet) prognostic SGS TKE from the `tke` column of
`prof.inp`, which the wrapper's preprocessing hardcodes to zero
(`python_udgeom/preprocessing.py`, `obj.tke = 0`); uDALES then merely warns and
clamps it to `e12min = 5e-5`, silently producing a degenerate SGS field. Its
constants are not namelist-reachable either, so `sgs_constant` would be inert.

`_apply_sgs_setting` then reads both switches back from the member's namoptions
(via `NamoptionsFile.get_value_as_bool`, which parses `.true.`/`.t.`/`true`/`T`
and the false forms case-insensitively) and writes `sgs_constant` to the matching
key. An **absent** switch falls back to the Fortran defaults in
`modsubgriddata.f90` — `lsmagorinsky=.false.`, `lvreman=.true.`, i.e. Vreman —
because a missing key means the compiled-in default applies, not `.false.`. If
both switches are `.true.` Smagorinsky wins, matching the
`if(lsmagorinsky) … elseif(lvreman) …` branch order in `closure`. If neither is
active the write is skipped with a warning (nothing would read the value). The
key and value actually written are logged, so a diverging run can be diagnosed
from `run.<expnr>.log`'s companion Python log.

**Where `sgs_constant` comes from.** Two sources, in precedence order:

1. `sgs_constant` in the params Dataset (from the `conf/params/*.yaml` sampler) —
   used when ESMDA estimates or pins it.
2. `forward_model.sgs_constant` in the backend's own `conf/model/*.yaml` — the
   per-backend default.

Absent from both is a strict no-op: the solver's own closure/template value
stands. The per-backend default exists because the three backends' `sgs_constant`
are **different physical quantities** (uDALES/pylbm take a dimensionless
Smagorinsky-family constant; PALM takes an eddy diffusivity in m²/s), so a single
value in the shared params sampler cannot be correct for all three at once.


> **The two constants are not on the same scale.** uDALES defaults are `cs = -1.`
> (→ the derived `(cm³/ceps)^0.25 ≈ 0.17`) for Smagorinsky and `c_vreman = 0.07`
> for Vreman. A prior tuned for `cs` is roughly 2–3× too large for `c_vreman`;
> retune `conf/params/*.yaml` when switching a case's closure.

**Initial inflow speed.** Static runs write `u0`/`v0` (and `dpdx`/`dpdy`) directly
into namoptions `&INPS` and the `prof.inp`/`lscale.inp` files via
`apply_inflow_settings`. The `uini` key is not written (that is a pylbm-specific
convention); uDALES uses `u0`/`v0` as the reference velocity scalars independent
of the initial condition.

---

## 5. Preprocessing — Python vs Matlab

uDALES requires preprocessing that converts an STL geometry into IBM (Immersed
Boundary Method) input files (`solid_*.txt`, `fluid_boundary_*.txt`,
`facet_sections_*.txt`, `facets.inp`, `facetarea.inp`). Two paths exist:

**Python preprocessor** (default):
[`python_udgeom/`](../libs/pyudales/src/pyudales/python_udgeom/) — a pure-Python
reimplementation of the Matlab `write_inputs.m` workflow.

| Module | Role |
|---|---|
| [`preprocessing.py`](../libs/pyudales/src/pyudales/python_udgeom/preprocessing.py) | `Preprocessing` class — reads namoptions, sets domain/grid properties |
| [`ibm.py`](../libs/pyudales/src/pyudales/python_udgeom/ibm.py) | IBM geometry: solid/fluid point classification + facet-to-cell matching (calls Fortran `IBM_preproc.exe`) |
| [`seb.py`](../libs/pyudales/src/pyudales/python_udgeom/seb.py) | Surface energy balance input generation |
| [`write_inputs.py`](../libs/pyudales/src/pyudales/python_udgeom/write_inputs.py) | Orchestrator — calls ibm/seb, handles `precomputed_geom_dir` bypass |

The Python preprocessor is invoked via `shell_scripts/write_inputs.sh`, which sets
`DA_EXPDIR` and `DA_TOOLSDIR` environment variables and runs the Python script
against the experiment directory. It uses `trimesh` for STL loading.

**Matlab preprocessor** (legacy, optional):
Calls `u-dales/tools/write_inputs.sh` and requires `matlab_bin` to be on `PATH`.
The Matlab path sleeps 90 s after subprocess launch to wait for MATLAB to finish.

**Selector.** The `prepare._target_` in
[`conf/model/pyudales.yaml`](../conf/model/pyudales.yaml) points at
`pyurbanair.config.hydra_helpers.prepare_udales`, which receives
`python_or_matlab: python` (the config default) and passes it to
`forward_model.run_preprocessing(python_or_matlab=...)`.

**Precomputed geometry.** Preprocessing is expensive (STL→IBM classification).
For a fixed STL + grid, the output files are deterministic. Set
`precomputed_geom_dir` (or `geometry.udales_precomputed_geom_dir` in the case
config) to a directory of pre-computed files. The wrapper flips `gen_geom=.false.`
and sets `geom_path` in namoptions; `write_inputs` then copies the files instead
of running the Fortran classifier. A `geom_meta.json` (written by
`save_precomputed_geometry`) records the grid dimensions; a mismatch raises.

**Geometry blanking.** Shipped from `solid_c.txt` (fluid fraction 0.9328 for the
Xie & Castro case). The nonzero-fallback and STL-voxelisation approaches both fail
for pyudales geometry detection — use only the `solid_c.txt`-sourced blanking.

---

## 6. Inflow / nudging

pyudales applies inflow via **nudging** (`use_nudging=True` is hardcoded in
`_apply_inflow_settings`) unless `inlet_turbulence.enabled` is set, which
replaces both the nudged inlet face and the interior relaxation with synthetic
driver planes (§6.1). The nudging generates a
`timedepnudge.inp.<expnr>` file and enables `&PHYSICS lnudge=.true.`,
`ltimedepnudge=.true.` in namoptions.

**`apply_time_varying_inflow`**
([`utils/nudging_utils.py`](../libs/pyudales/src/pyudales/utils/nudging_utils.py)):

- **Time-varying params** (`"time"` dim present): uses the time array directly.
- **Scalar/constant params**: synthesises a 2-snapshot constant schedule spanning
  `[0, simulation_time]` so that nudging holds the values fixed — required to
  stabilise inflow-outflow BCs.
- **Spinup**: when `spinup_time > 0`, a constant plateau at the initial parameter
  values is prepended before the time-varying schedule begins.
- Under inflow-outflow BCs, `dpdx`/`dpdy` are zeroed in namoptions (`&INPS`) so
  the inlet face is the sole streamwise driver (no body-force conflict with pylbm,
  which has no body force).

**`profile_config`** (nested under `nudging_config`):

| Field | Values | Effect |
|---|---|---|
| `type` | `"uniform"` (default) or `"power_law"` | Shape of vertical inflow profile |
| `alpha` | float | Power-law exponent `(z/z_ref)^alpha` |
| `z_ref` | float (m) | Reference height; defaults to domain top (`zsize`) |

Profile shape is computed by
[`utils/vertical_profile.build_profile_shape`](../libs/pyudales/src/pyudales/utils/vertical_profile.py)
and applied column-wise to u/v at each time snapshot.

**`nnudge_meters`**: height in metres below which nudging is NOT applied. Converted
to a grid-level count `nnudge` via:

```python
nnudge = int(np.count_nonzero(heights < nnudge_meters))
```

where `heights` are cell centres `(0.5*dz, 1.5*dz, …)`. This takes precedence
over the raw `nnudge` key. Config default: `nnudge_meters: 4.0`.

**`tnudge`**: nudging relaxation timescale in seconds (default 15.0). Xie & Castro
"divergence" in practice is marginal dt-collapse at end-of-window, not a nudging
failure; it was resolved by raising the Smagorinsky constant from cs 0.20 → 0.24
(see §8).

### 6.1 Turbulent inlet — synthetic driver planes

uDALES v2.2.0 has two inlet-turbulence routes. The Lund (1998) recycling
generator is dead code (documented below, and still asserted against the Fortran
source by `tests/test_udales_inlet_turbulence.py`). The **precursor/driver**
route — `BCxm=3` → `idriver=2`, `moddriver.f90` — is wired end to end, and that
is what `inlet_turbulence.enabled: true` drives, fed by driver planes
**synthesised in Python** rather than by a precursor run.

```yaml
  inlet_turbulence:
    enabled: false        # default: strict no-op, byte-identical namoptions
    intensity: 0.1        # u'_rms / |U_mean(z)|; v' and w' get 0.7x this
    length_scale_y: 25.0  # m — spanwise integral length scale
    length_scale_z: 25.0  # m — vertical
    length_scale_x: 50.0  # m — streamwise; sets the AR(1) time scale (Taylor)
    time_step: 0.5        # s — &DRIVER dtdriver
    driverjobnr: 998      # 3-digit file suffix; must differ from the expnr
    seed: null            # null -> derived from the member's experiment name
    # lchunkread: true / chunkread_size: 100   (large cases only)
```

The block is wired through the `inlet_turbulence: Optional[dict]` constructor
arg, validated by `validate_inlet_turbulence` and implemented by
[`utils/inlet_turbulence_utils.py`](../libs/pyudales/src/pyudales/utils/inlet_turbulence_utils.py)
(physics + orchestration) on top of
[`utils/driver_file_utils.py`](../libs/pyudales/src/pyudales/utils/driver_file_utils.py)
(binary I/O). **Absent, `{}`, or `enabled: false` writes nothing at all** —
byte-identical namoptions and no driver files. Unknown keys are warned about and
ignored. `enabled: true` requires `boundary_condition: inflow_outflow`; under
`periodic` there is no inlet face and it raises.

**What is generated.** Per `run_single`, the whole record sequence for the
window is regenerated and the files overwritten in place (so stale planes from a
previous window — or from a failure-substitution donor — cannot leak):

- **Mean profile** from the *same* ESMDA parameters the nudging path uses:
  `build_profile_shape` with the resolved `profile_config` (α from
  `vertical_inflow_exponent`), scaled by `velocity_magnitude` and decomposed by
  `inflow_angle`. Time-varying params are interpolated onto the driver time grid
  with the same spinup plateau `apply_time_varying_inflow` builds. The angle is
  baked into the planes, so `iangledeg` is left at its default 0.
- **Fluctuations** from a Xie & Castro (2008) digital filter: white noise
  correlated in y (periodic FFT-free circular convolution, matching `BCym=1`),
  in z (truncated at the ground and the top, renormalised per level), and in
  time by an AR(1) recursion with `a = exp(-pi*dtdriver/(2T))`,
  `T = length_scale_x / U_ref`. Uncorrelated noise would be annihilated by the
  pressure projection within a couple of cells; correlated structures survive.
  The plane mean of `u'` is zeroed per record so the instantaneous bulk inflow
  equals the mean-profile bulk.

**Namoptions written on the enabled path:** `&BC BCxm=3`; `&DRIVER idriver`,
`driverjobnr`, `driverstore`, `dtdriver`, `tdriverstart` (plus
`lchunkread`/`chunkread_size` only when configured — `NamoptionsFile.set_value`
creates the missing section, so templates need no `&DRIVER` block); `&PHYSICS
lnudge=.false.`, `ltimedepnudge=.false.`; `&INPS u0`, `v0`, `dpdx=0`, `dpdy=0`.

**Nudging is turned off deliberately.** Under `BCxm=3` the inlet no longer comes
from `uprof`, so `ltimedepnudge` has nothing left to drive, and interior nudging
would damp the injected fluctuations on the `tnudge=15 s` scale. Time-varying
inflow is instead encoded in the plane sequence itself, which is time-resolved.

**Driver-file format** (`moddriver.f90:786-872`, unit-tested in
`driver_file_utils`): four files `tdriver_000.NNN`, `udriver_000.NNN`,
`vdriver_000.NNN`, `wdriver_000.NNN`, where `NNN` is `driverjobnr` (i3.3) and
`000` is `driverid = mod(myidy, nprocy)` — always 0 because the wrapper pins
`nprocy=1`. They are `unformatted`+`access='direct'`, i.e. a **raw little-endian
float64 stream with no record markers** (the build uses `-fdefault-real-8`), so
record *n* starts at exactly `(n-1)*record_bytes`. Each velocity record is one
inlet plane `(ktot+2) x (jtot+2)` with **j fastest**; `tdriver` records are one
float64 each, ascending. The halos are `jh = kh = 1` under this build's cd2
advection (`modglobal.f90:575-582`); the j halo columns are the periodic wrap
and the k halo rows are an edge copy (`xmi_driver` reads `k = kb..ke` for u/v
and `kb..ke+1` for w). `local_execute.sh` copies the experiment dir into the
output dir before `mpiexec`, so bare relative names resolve.

**Window continuity.** `_prepare_warmstart` writes `timee = 0` into every
restart (uDALES otherwise dies with `timee >= runtime`), so the solver clock —
and `btime` — restart at 0 each window. `ForwardModel._elapsed_time` therefore
tracks the member's *physical* time and selects which slice of its turbulence
history the window gets: the AR(1) recursion is replayed from a fixed
name-derived seed up to that point, making window *n*'s planes the continuation
of window *n-1*'s with no persisted filter state.

That clock is **persisted to `<experiment_dir>/inlet_turbulence_clock.json`**,
not merely held on the object. `BaseEnsembleForwardModel._run_parallel` submits
`model.__call__` to a `ProcessPoolExecutor`, so under
`ensemble.num_parallel_processes > 1` the member is pickled into a forkserver
worker and every attribute it mutates dies with that process — an in-memory
counter would silently reset to 0 each window and restart the turbulence
history. `run_single` reloads it at the top and writes it back after a
successful run; `EnsembleForwardModel.run_ensemble` copies a donor's clock to a
substituted member alongside its warmstart carry, and refreshes the parent's
in-memory copies.

Replay cost is bounded: the AR(1) recursion forgets its history geometrically,
so only `log(eps)/log(a)` records before the window are replayed
(`ar1_burn_in_records`). Each record's white noise comes from a counter-based
Philox stream keyed on `(seed, record index)` — with a sequential generator the
draw order *is* the state, so truncating the prefix would change every
subsequent record. Without this bound a 20-window rollout would cost
O(windows²).

Coverage is a hard constraint — `drivergen` stops the run outright once `timee`
exceeds the last record — so the grid overshoots by two records, or by `dtmax`
where that is larger (uDALES' final step can overrun `runtime` by up to
`dtmax`). `time_step` must divide the window length, or the record grid would
slip against physical time every window; `apply_inlet_turbulence` raises rather
than let that accumulate silently.

**Consequence for the ESMDA path.** The estimated `inflow_angle` /
`velocity_magnitude` / `vertical_inflow_exponent` still reach the solver, now
through the plane *means* rather than through `uprof`. This is why a real
per-member precursor was rejected: it would have replaced the inlet face and
disconnected those parameters (the same objection that made `iinletgen` a
`ValueError`). A "precursor library" — record one periodic run, replay and
rescale — remains a future option behind the identical file interface.

**Spin-up is longer than on the nudging path.** `prof.inp`/`lscale.inp` are
written as zeros (start from rest) *and* `lnudge=.false.`, so the interior fills
from the inlet face alone with no relaxation pulling it toward the target
profile. A `spinup_time` sized from nudging-path experience will be too short.
Measure it during calibration.

**Calibration is still open.** The shipped `intensity` and length scales are
starting points (≈ building height), not tuned values. Three things bias the
realised turbulence below its nominal value, all of which calibration should
account for before reaching for more amplitude:

- *Too-short correlation lengths* are the classic failure mode — the
  fluctuations decay before reaching the buildings, and the fix is longer
  scales, not larger `intensity`.
- *Plane-mean removal* costs ~2% of the target rms at length scales small
  against the inlet plane, but **40–50%** once `length_scale_y/z` approach the
  plane's own dimensions (which the shipped 25 m defaults do on this domain).
  The generator logs a warning past 30% of the plane extent.
- *Near-wall rms* is low by construction: `sigma(z) = intensity * |U_mean(z)|`
  with a power-law shape sends `sigma → 0` at the ground, whereas a real ABL has
  its largest `u'_rms/U` there. A z-dependent intensity profile is the phase-2
  refinement if the canyon flow turns out to care.

See §8 of
[docs/plans/udales_inlet_turbulence.md](plans/udales_inlet_turbulence.md).

#### Why not the Lund generator

uDALES ships the Lund (1998) recycling/rescaling inlet generator (`iinletgen=1`,
`modinlet.f90`) but never connects it. Three independent facts, each on its own
decisive:

| # | Finding | Evidence |
|---|---|---|
| 1 | `iinletgen` is **not a member of any namelist**, and is never assigned anywhere in the source — it keeps its `modglobal.f90:167` default of `0` and is not even `MPI_BCAST` to the other ranks | Namelist blocks are `modstartup.f90:108-173`; `&INLET` (`:141-144`) declares only `Uinf, Vinf, di, dti, inletav, linletRA, lstoreplane, lreadminl, lfixinlet, lfixutauin, lwallfunc` |
| 2 | `call initinlet` — which allocates the generator's arrays — is **commented out** | `program.f90:77`, `modstartup.f90:627` |
| 3 | `inletgen`/`inletgennotemp` have **no call site anywhere**, and `modboundary.f90` never reads the generator's output `u0inletbc` | `modinlet.f90:204`, `:952`; `modboundary.f90` has zero `iinletgen` references |

Writing the key is not a silent no-op — uDALES namelist reads are strict, so it
aborts at startup:

```
 ERROR: Problem in namoptions INLET
 iostat error:         5010
STOP 1
```

(reproduced directly against `build/release/u-dales`; the same file with `Uinf`
instead of `iinletgen` gets past the `&INLET` read.) The wrapper therefore never
writes the key on any path — `test_wrapper_never_writes_iinletgen` enforces it.

**`iinletgen` vs `BCxm`.** They do not interact at all in v2.2.0. The inlet face
is set purely by `select case(BCxm)` in `modboundary.f90` — `xmi`/`bcpup`
(`:255-262`, `:1204-1262`) branch over `BCxm_periodic` / `BCxm_profile` /
`BCxm_driver` only. Under `BCxm=2` (`BCxm_profile`) the inlet is `uprof(k)` and
under `BCxm=3` it is the driver planes; the generator's `u0inletbc` array has no
consumer in either case. So even a patched-in `iinletgen=1` would produce
nothing at the inlet without also adding a `BCxm` case for it. That is why the
synthetic-plane route above rides on `BCxm=3` instead of trying to revive the
generator: it needs no Fortran changes at all.

---

## 7. Instability watchdog

[`utils/run_monitor.py`](../libs/pyudales/src/pyudales/utils/run_monitor.py)

`run_with_dt_watchdog` replaces the bare `subprocess.run` used to launch uDALES.
It spawns the process in its own session (so killing the session kills the entire
`bash → mpiexec → MPI-rank` tree), then tails `run.<expnr>.log` in a polling loop:

```
line pattern: "... dt:  0.242654880"   ← parsed by _DT_RE
```

`InstabilityCheck.from_config(config_dict)` builds the watchdog config from the
`instability_check:` block in the forward model config. Unknown keys are warned
and ignored.

| Field | Default | Meaning |
|---|---|---|
| `enabled` | `True` | When False, falls back to bare `subprocess.run(check=True)` |
| `min_dt` | `1e-4` s | Absolute timestep floor |
| `patience` | `20` | Consecutive sub-floor steps before killing |
| `warmup_steps` | `20` | Ignore the first N steps (legitimate ramp-up) |
| `poll_interval_s` | `2.0` | Log re-read interval |

When the patience criterion trips, the process group receives `SIGTERM` (with a
`SIGKILL` fallback after 5 s), and `CalledProcessError` is raised — the same
exception the ensemble layer already treats as a member failure. The
`resample_from_successes` failure policy then substitutes the diverging member
from a random successful donor without waiting for the slow dt-collapse crash.

**Known gotchas:**
- Xie & Castro "divergence" is typically marginal dt-collapse near end-of-window,
  not a genuine flow instability. It is fixed by raising the SGS constant — under
  Smagorinsky `cs 0.20 → 0.24`; under Vreman the equivalent lever is `c_vreman`,
  where the 0.07 default diverges and ~0.25 runs clean on this case.
- Stale fielddumps can give NaN-padded or duplicate z-coordinates after a failed
  worker is retried. `run_single` automatically cleans the member output directory
  before restoring/staging any warmstart files for each attempt.

---

## 8. Parameters consumed

`INFLOW_PARAM_NAMES` in
[`utils/params_utils.py`](../libs/pyudales/src/pyudales/utils/params_utils.py) is
the whitelist that `extract_inflow_params` and `merge_params` use. Any variable
**not** in this tuple is **silently dropped** before reaching the solver:

```python
INFLOW_PARAM_NAMES = (
    "inflow_angle",
    "velocity_magnitude",
    "pressure_gradient_magnitude",
    "vertical_inflow_exponent",
    "sgs_constant",
)
```

| Parameter | Where written | Notes |
|---|---|---|
| `inflow_angle` | `u0`/`v0`/`dpdx`/`dpdy` in namoptions + nudging files | Degrees from +x axis |
| `velocity_magnitude` | `u0`/`v0` in namoptions + nudging profiles | m/s reference speed |
| `pressure_gradient_magnitude` | `dpdx`/`dpdy` in namoptions | Pa/m; decomposed by angle. uDALES-only parameter. |
| `vertical_inflow_exponent` (α) | Overrides `profile_config["alpha"]` in `_resolve_nudging_config` | Estimated by ESMDA; per-member shear |
| `sgs_constant` | `&NAMSUBGRID cs` (Smagorinsky) or `c_vreman` (Vreman) via `_apply_sgs_setting` | Dimensionless SGS constant; the target key is picked from the case's `lsmagorinsky`/`lvreman` switches (§4). Note the two constants have different natural magnitudes (~0.17 vs 0.07). |

**`pressure_gradient_magnitude`** is the third parameter unique to pyudales.
`resolve_parameter_schema` in `hydra_helpers.py` adds it to the uDALES parameter
schema; the sampler configs include it as a `Constant` which other backends ignore.

**Model-error knobs** (`vertical_inflow_exponent`, `sgs_constant`) are applied
unconditionally (outside the static/time-varying branch), so they affect every
run where they are present. Each is a no-op when absent, keeping default runs
byte-identical. See [codebase_guide.md §7](codebase_guide.md) "Model-error
compensation knobs".

---

## 9. `utils/` subpackage

| Module | Purpose |
|---|---|
| [`clean_up_utils.py`](../libs/pyudales/src/pyudales/utils/clean_up_utils.py) | `clean_output_dir` (delete output), `clean_temp_dir` (wipe experiment dir except namoptions/STL/config.sh) |
| [`config_utils.py`](../libs/pyudales/src/pyudales/utils/config_utils.py) | `create_config_sh` — writes `config.sh` with `DA_EXPDIR`, `DA_NCPU`, `MATLAB_BIN` |
| [`dir_utils.py`](../libs/pyudales/src/pyudales/utils/dir_utils.py) | `DirectoryPaths` dataclass, `get_udales_directory_paths`, `get_project_root` |
| [`file_update_utils.py`](../libs/pyudales/src/pyudales/utils/file_update_utils.py) | `update_prof_file`, `update_lscale_file`, `…_profile` variants — patch `prof.inp` and `lscale.inp` in-place |
| [`file_utils.py`](../libs/pyudales/src/pyudales/utils/file_utils.py) | `copy_files`, `change_file_extensions` (rename experiment-suffix files) |
| [`forward_model_utils.py`](../libs/pyudales/src/pyudales/utils/forward_model_utils.py) | `create_new_forward_model` — deep-copy template into per-member directory |
| [`grid_utils.py`](../libs/pyudales/src/pyudales/utils/grid_utils.py) | `interpolate_grid` — staggered → cell-centred collocation (see §3) |
| [`driver_file_utils.py`](../libs/pyudales/src/pyudales/utils/driver_file_utils.py) | `write_driver_files`/`read_driver_files` — raw float64 `*driver_000.NNN` inlet planes for `idriver=2` (see §6.1) |
| [`inflow_utils.py`](../libs/pyudales/src/pyudales/utils/inflow_utils.py) | `angle_to_velocity`, `angle_to_pressure_gradient` — decompose angle+magnitude into u/v and dpdx/dpdy components |
| [`inlet_turbulence_utils.py`](../libs/pyudales/src/pyudales/utils/inlet_turbulence_utils.py) | `validate_inlet_turbulence`, `apply_inlet_turbulence` — synthetic turbulent inlet on the `BCxm=3` driver route (see §6.1) |
| [`namoptions_utils.py`](../libs/pyudales/src/pyudales/utils/namoptions_utils.py) | `NamoptionsFile` editor (see §4), `parse_fortran_logical`, `rename_namoptions_file` |
| [`ncpu_utils.py`](../libs/pyudales/src/pyudales/utils/ncpu_utils.py) | `validate_and_sync_ncpu` — sets `nprocx=ncpu, nprocy=1` and checks divisibility |
| [`nudging_utils.py`](../libs/pyudales/src/pyudales/utils/nudging_utils.py) | `apply_time_varying_inflow`, `compute_nudging_profiles`, `write_timedepnudge_file`, `enable_nudging_in_namoptions` (see §6) |
| [`params_utils.py`](../libs/pyudales/src/pyudales/utils/params_utils.py) | `INFLOW_PARAM_NAMES` whitelist, `extract_inflow_params`, `merge_params`, `apply_inflow_settings`, `get_param_value`, `is_time_varying_params` |
| [`random_utils.py`](../libs/pyudales/src/pyudales/utils/random_utils.py) | `apply_random_initial_condition` — sets `irandom`/`randu`/`lrandomize` in `&RUN` |
| [`rollout_utils.py`](../libs/pyudales/src/pyudales/utils/rollout_utils.py) | `collect_rollout_results` — concatenate per-window result files along time |
| [`run_monitor.py`](../libs/pyudales/src/pyudales/utils/run_monitor.py) | `run_with_dt_watchdog`, `InstabilityCheck` (see §7) |
| [`save_frequency_utils.py`](../libs/pyudales/src/pyudales/utils/save_frequency_utils.py) | `apply_output_frequency` (writes `tfielddump`), `apply_save_only_last_timestep` (sets `tfielddump=runtime`) |
| [`vertical_profile.py`](../libs/pyudales/src/pyudales/utils/vertical_profile.py) | `build_profile_shape` — `uniform` or `power_law` shear profile `s(z)` |
| [`warm_start_utils.py`](../libs/pyudales/src/pyudales/utils/warm_start_utils.py) | `set_warm_start`, `set_trestart`, `fetch_carry`/`store_carry`/`copy_carry`, `update_warmstart_file_from_xarray`, `identify_generated_warmstart_file` |

---

## 10. Config wiring — `conf/model/pyudales.yaml`

[`conf/model/pyudales.yaml`](../conf/model/pyudales.yaml) is the complete model
config entry. Notable fields:

```yaml
name: pyudales
solver_name: udales          # selects dim_mapping in ObservationOperator

forward_model:
  _target_: pyudales.forward_model.ForwardModel
  case_dir: ${geometry.udales_case_dir}
  precomputed_geom_dir: ${oc.select:geometry.udales_precomputed_geom_dir,null}
  temp_dir: ${paths.experiment_dir}
  experiment_name: "999"
  matlab_bin: /opt/sw/matlab-2023b/bin/matlab  # unused when python_or_matlab: python
  ncpu: 25
  boundary_condition: inflow_outflow
  closure: vreman             # smagorinsky | vreman | null (keep template)
  nudging_config:
    tnudge: 15.0
    nnudge_meters: 4.0          # skip nudging below 4 m (near-wall cells)
    profile_config:
      type: power_law
      alpha: 0.25
  instability_check:
    enabled: true
    min_dt: 1.0e-4
    patience: 20
    warmup_steps: 20
    poll_interval_s: 2.0
  inlet_turbulence:
    enabled: false            # true -> synthetic driver planes, BCxm=3 (§6.1)
    intensity: 0.1            # u'_rms / |U_mean(z)|
    length_scale_x/y/z: 50/25/25   # m, digital-filter integral length scales
    time_step: 0.5            # s, &DRIVER dtdriver
    driverjobnr: 998
  nx/ny/nz: ${domain.nx/ny/nz}
  bounds: ${domain.bounds}
  simulation_time: ${time.simulation_time}
  output_frequency: ${time.output_frequency}
  spinup_time: ${time.spinup_time}

prepare:
  _target_: pyurbanair.config.hydra_helpers.prepare_udales
  python_or_matlab: python          # python (default) or matlab

ensemble_model:
  _target_: pyudales.ensemble_forward_model.EnsembleForwardModel
  ensemble_size: ${ensemble.ensemble_size}
  num_parallel_processes: ${ensemble.num_parallel_processes}
  num_cpus_per_process: ${ensemble.num_cpus_per_process}
  failure: ${ensemble.failure}      # inherits resample_from_successes / raise
```

The `prepare._target_` (`prepare_udales` in
[`hydra_helpers.py`](../src/pyurbanair/config/hydra_helpers.py)) calls
`forward_model.run_preprocessing(python_or_matlab=...)`. For the Python path the
script is `shell_scripts/write_inputs.sh`; for Matlab it is
`u-dales/tools/write_inputs.sh`.

---

## 11. Warm start / spinup

`run_single` manages a two-track execution path:

**Cold start** (`state=None`): uDALES runs with spinup from the template initial
condition. `set_trestart` enables restart-file writing. After the run,
`store_carry` saves the end-of-run restart file as this member's **carry** — a
persisted binary with subgrid fields (e120, ekm, thl0, …) that avoids turbulence
re-spin-up on the next window.

**Warm start** (`state` provided): `spinup_time` is zeroed for this call.
`fetch_carry` retrieves the member's persisted carry. If available it is used as
the restart template (only u/v/w/pres are overwritten from `state`); otherwise a
tiny cold-start run is triggered to produce the template. After the warm run,
`store_carry` captures the new end-of-run restart for the next window.

On failure substitution, `EnsembleForwardModel.run_ensemble` copies the donor's
carry to the failed member's slot so the resampled state and the subgrid carry are
consistent.

`disable_spinup()` zeros `spinup_time` and rewrites `runtime` in namoptions.
Called by `BaseRolloutForwardModel` after step 0 when
`spinup_first_step_only=True`.

---

## 12. Multi-CPU considerations

`validate_and_sync_ncpu` enforces `nprocx = ncpu, nprocy = 1`. This means the
domain is always decomposed in x-strips only.

**The `gather_outputs.sh` no-op.** With `nprocy=1` the shell script's NCO y-pass
is a no-op, which starves the x-pass, so a merged `fielddump.<expnr>.nc` is never
written. `_read_fielddump` detects this and stitches per-rank slabs in Python via
`_stitch_x_decomposition` (see §3).

**`combine_by_coords` bloats staggered fields.** When slabs are passed to
`xarray.combine_by_coords` the staggered `xt`/`xm` pair causes a 2-D broadcast,
exploding memory. Always use `_stitch_x_decomposition`.

**Ensemble parallelism.** Each ensemble member runs its own uDALES instance with
its own process group (`start_new_session=True`). With `ncpu=25` per member and
`num_parallel_processes=4`, 100 MPI ranks run simultaneously. The DRAM-bandwidth
ceiling on the development box is ~4–8 parallel processes (see
[ensemble_scaling.md](temp/ensemble_scaling.md)).

---

## Quick reference

| You want to… | Look here |
|---|---|
| Change the advection scheme | Not possible — only `cd2` (`iadv_mom` hardcoded by the build) |
| Switch the SGS closure | `model.forward_model.closure=smagorinsky\|vreman` (writes the exclusive `&NAMSUBGRID` switches; §4) |
| Turn on a turbulent inlet | `model.forward_model.inlet_turbulence.enabled=true` — synthetic driver planes on the `BCxm=3` route (needs `inflow_outflow`; turns nudging off). uDALES' own `iinletgen` stays unreachable dead code (§6.1) |
| Change the SGS constant | `&NAMSUBGRID cs` (Smagorinsky) / `c_vreman` (Vreman) in namoptions, or pass `sgs_constant` in params — it targets the active closure automatically (§4) |
| Add a new inflow parameter | Add to `INFLOW_PARAM_NAMES` in `params_utils.py` first |
| Skip expensive preprocessing | Set `precomputed_geom_dir` / `geometry.udales_precomputed_geom_dir` |
| Debug a silent crash | Set `verbose: true` on the forward model (or `model.forward_model.verbose=true` CLI) |
| Tune the instability watchdog | `instability_check:` block in `conf/model/pyudales.yaml` |
| Interpolate staggered → centred | `pyudales.utils.grid_utils.interpolate_grid(ds)` |
| Change nudging height cutoff | `nudging_config.nnudge_meters` in model config |
| Understand ncpu → nprocx mapping | `utils/ncpu_utils.validate_and_sync_ncpu` |
