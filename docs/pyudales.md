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

Importing pyudales does not download, checkout or compile solver code. Normal
`prepare_udales` / `run_preprocessing` prepares the solver and preprocessing tools;
direct forecasts and ensemble construction also ensure the solver is prepared
before any workers start.

[`utils/solver_build.py`](../libs/pyudales/src/pyudales/utils/solver_build.py)
exports pristine commit `b84916ac60cecd1da54dd09df76c15e30dcaabe9` from the
submodule's Git objects (or a cached direct clone), leaving local edits untouched.
Stock and discrepancy variants live in separate, locked, atomically published
cache entries. The default cache is `.cache/pyudales`; `PYUDALES_CACHE_DIR` can
select another writable location. Package resources may be read-only. Cache keys
include source/extension hashes, build scripts, toolchain, dependencies, flags and
platform. Reuse requires a matching capability manifest and executable hash.
Failed or partial builds are rebuilt; `CMakeCache.txt` alone is insufficient.
A failed configure or compile raises with the build output, publishes nothing,
and the next preparation retries it.

**Platforms.** The same scripts build on Linux and macOS
([`shell_scripts/build_udales.sh`](../libs/pyudales/shell_scripts/build_udales.sh),
`build_preprocessing.sh`). `build_udales.sh` points CMake at the pixi env
(`CMAKE_PREFIX_PATH`, `NETCDF_DIR`, `NETCDF_FORTRAN_DIR`, `FFTW_ROOT` default to
`$CONDA_PREFIX`), so a system NetCDF or FFTW can never be picked up instead;
values you set yourself still win. On macOS with a pixi compiler, the shared
`pyurbanair.utils.toolchain.apple_linker_flags` selects Apple's `/usr/bin/ld`
for both native builds and the IBM geometry compiler: conda's ld64 cannot read
a current SDK's `libSystem.tbd`. An explicit caller-supplied `-B` flag takes
precedence. The active SDK and effective linker flags enter the build cache key.

The selected executable travels in `DirectoryPaths.solver_executable` and
`config.sh`'s `DA_BUILD`; its source tree supplies the corresponding preprocessing
tools. No manual patch, solver fork, or executable-path override is required.
The launch script remains a pyudales resource outside the upstream source.

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
[`configs/model/pyudales.yaml`](../configs/model/pyudales.yaml)):

| Argument | Purpose |
|---|---|
| `case_dir` | Case folder whose files (namoptions, the STL its `stl_file` names) are copied into the run (from `${geometry.case_dir}`) |
| `experiment_name` | uDALES experiment number string (default `"999"`) |
| `ncpu` | Total MPI ranks; always decomposed as `nprocx=ncpu, nprocy=1` (x-strips only) |
| `simulation_time` | Window length in seconds; written to `&RUN runtime` |
| `spinup_time` | Prepended spin-up in seconds; effective runtime = `simulation_time + spinup_time` |
| `nx/ny/nz`, `bounds` | Domain overrides → `itot/jtot/ktot`, `xlen/ylen/zsize` in namoptions |
| `boundary_condition` | `"periodic"` or `"inflow_outflow"` (sets `BCxm`, `BCym`, `BCtopm`); the config reads `flow.boundary_condition` |
| `outflow_buffer` | `inflow_outflow` only: metres of flat ground added downstream of `bounds`, the output cropped back (§5); `None` (default) no buffer |
| `closure` | SGS closure: `"smagorinsky"` / `"vreman"` → exclusive `&NAMSUBGRID` switches; `None` (default) keeps the template's. `"oneeqn"` is rejected — see §4 |
| `nudging_config` | Nudging tunables dict; see §6 |
| `inlet_turbulence` | Turbulent-inlet block. `None`/`false` (default) is a strict no-op; `true` switches the inlet to synthetic driver planes (`BCxm=3`) and turns nudging off. See §6.1 |
| `instability_check` | dt-watchdog config dict; see §7 |
| `per_member_irandom` | `False` (default) is a no-op. `True` gives each run its own initial-condition seed; see §6.2 |
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
(`configs/model/pyudales.yaml`). `_apply_closure` writes **all three** switches — the
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

1. `sgs_constant` in the params Dataset (from the `configs/params/*.yaml` sampler) —
   used when ESMDA estimates or pins it.
2. `forward_model.sgs_constant` in the backend's own `configs/model/*.yaml` — the
   per-backend default.

Absent from both is a strict no-op: the solver's own closure/template value
stands. The per-backend default exists because the three backends' `sgs_constant`
are **different physical quantities** (uDALES/pylbm take a dimensionless
Smagorinsky-family constant; PALM takes an eddy diffusivity in m²/s), so a single
value in the shared params sampler cannot be correct for all three at once.


> **The two constants are not on the same scale.** Vreman (2004) relates them
> by `c_vreman ≈ 2.5 cs²`. uDALES' Fortran defaults are `cs = -1.` (→ the
> derived `(cm³/ceps)^0.25 = 0.217`, from `cf = 2.5`, `alpha_kolm = 1.5`) and
> `c_vreman = 0.07` (`cs ≈ 0.17`). The config default is `c_vreman = 0.12`, the
> Vreman equivalent of that `cs = 0.217` (§7). Retune `configs/params/*.yaml`
> when switching a case's closure.

**Initial inflow speed.** Static runs write `u0`/`v0` (and `dpdx`/`dpdy`) directly
into namoptions `&INPS` and the `prof.inp`/`lscale.inp` files via
`apply_inflow_settings`. The `uini` key is not written (that is a pylbm-specific
convention); uDALES uses `u0`/`v0` as the reference velocity scalars independent
of the initial condition.

---

### 4.1 Strain/rotation discrepancy

`model.forward_model.model_discrepancy` enables a native Vreman viscosity
correction. Fixed coefficients are supported in single and ensemble forward
runs. The backend also provides the window checkpoint/replay protocol below.
The runners also support parameter-only ESMDA, parameter/joint filtering, and
parameter-only ESMDA followed by state-only filtering. ESMDA and the hybrid may
estimate dynamic inflow parameters alongside static SGS coefficients; the
coefficients have no time dimension and carry their analyzed values between
windows. There are no within-forecast coefficient schedules or coefficient
process noise.

The backend requires explicit feature settings. The shipped model config currently
enables the SGS bias test with a 10 m canopy height, a height band of `[0.5, 2.0]`,
regularization of 0.01 s⁻¹, and a log cap of `log(1.5)`. The dynamic prediction
prior prescribes coefficients of `-20`; these are test settings, not calibration.
Set `model_discrepancy.enabled=false` to use the stock solver.

The following numbers illustrate
the interface and are **not calibrated defaults**:

```yaml
model_discrepancy:
  enabled: true
  kind: sgs_strain_rotation
  coefficient_model: persistent
  canopy_height: 10.0
  height_band_over_H: [0.5, 1.5]
  gradient_regularization: 1.0e-6
  log_multiplier_cap: 1.0986122886681098
  prior_std: null
```

Height is measured in metres from the native solver's vertical datum (`zf`);
`canopy_height` is a fixed representative building height. The regularization is
in s⁻¹ and the logarithmic cap is dimensionless. Coefficient priors live in
`configs/params/static.yaml` or the `static_parameters` block of
`configs/params/dynamic.yaml`; truth values live in the regular truth configs.
Explicit distributions take precedence.
`prior_std` is an optional fallback: its three positive finite scales supply
zero-centred Gaussian priors for selected coefficients missing from a custom
sampler, in the static block of either sampler type. Leave it null when all
selected coefficients have explicit priors.
In a forward-only run this field does not sample anything.

All SGS coefficient entries in the truth and prediction parameter configs are
sampled and applied independently of `params_to_estimate`. That setting only
selects prediction parameters for DA updates. For example, `Constant` entries
of `-20` under the dynamic prior's `static_parameters` apply those coefficients
even when only `inflow_angle` and `velocity_magnitude` are estimated. A coefficient
absent from the config falls back to its constructor value (zero by default),
unless selected inference supplies a missing prior via `prior_std`.

Supply static scalar Dataset fields `sgs_bias_b0`, `sgs_bias_b1`, `sgs_bias_b2`
through `params` (or `Constant` entries in the parameter sampler). For example:

```python
params = xarray.Dataset({"sgs_bias_b0": 0.1, "sgs_bias_b1": -0.2, "sgs_bias_b2": 0.05})
state = forward_model(params=params)
```

Coefficients are routed separately from inlet controls. Each call resolves
against constructor coefficients (zero by default), then applies that call's
provided fields; previous call values never become defaults. All three are
written after preprocessing. Time-array and nonfinite coefficients are rejected.
Disabling a reused case strips extension-only namelist keys before selecting the
stock executable; untouched disabled inputs are a no-op.
When enabled, the native `c_vreman` value must be finite and nonnegative, including
any per-call `sgs_constant` override. A positive multiplier cannot repair a
negative native eddy viscosity.

The correction is `exp(L*tanh((b0+b1*phi+b2*q)/L))`, where `phi` is the compact
sine-squared height band and `q` compares squared rotation and strain norms.
The native patch reuses Vreman gradients every closure evaluation, preserves
scalar diffusivity by applying the multiplier after `ekh` is calculated, and
adds molecular viscosity afterward. The existing momentum stress divergence and
boundary filling remain in place. Enabled zero-gradient cells have zero SGS
viscosity; the disabled branch retains upstream arithmetic. Before the first
step, the enabled extension refreshes viscosity from the staged state with
velocity halos/boundaries populated. Enabled warm starts stage the full rank
restart set, inject each rank's velocity/pressure slab, synchronize clocks and
preserve that rank's hidden records byte-for-byte. This staging supports the
pinned `cd2` layout and an evenly divided x-only decomposition. Auxiliary restart
reads (`lreadmean`, `lreadminl`, or `nsv>0` with `lreadscal`) are rejected until
those additional files are carried too. Unsupported layouts or incomplete rank
sets fail before launch. Native startup also rejects inconsistent
rank clocks before entering thermodynamic collectives.

The returned/saved state carries JSON in its `model_discrepancy` attribute:
feature settings, coefficients, executable/build provenance and native multiplier
extrema/saturation diagnostics. The native diagnostic text file is collected
before output cleanup. Ensemble metadata is aggregated as
`model_discrepancy_by_member`. General native physics, multi-rank equivalence,
wall/energy budgets, continuous-versus-segmented forecasts, and coefficient
recovery remain separate acceptance gates; compiling the extension does not
demonstrate transfer skill.

### 4.2 Window checkpoints for repeated forecasts

Discrepancy-enabled forward and ensemble models implement
`begin_forecast_window()`, `restore_forecast_window()`, and
`end_forecast_window(commit=...)`. Begin captures each member's full native
carry, runtime input files, template, mutable parameter defaults, and physical
clock in a separate temporary directory. Restore verifies the snapshot hashes
and stages its contents before replacing the live inputs. Checkpoints include
every MPI rank; incomplete or corrupt carry fails instead of silently falling
back to a cold start. Snapshot storage is additional to forecast output storage
and is released when the window closes.

Each replay starts from these inputs, then injects the current velocity/pressure
state and coefficients. The native startup viscosity refresh still runs before
timestep selection. A successful `commit=True` retains the final forecast's
carry and clock once; `commit=False` restores the original inputs and clock.
The ESMDA loop uses this lifecycle automatically on supporting backends, rolling
back on errors or `final_forecast=False`. Ordinary sequential forward calls
outside this lifecycle continue carrying their previous endpoint.

Failure substitution copies the donor's window-start identity as well as its
endpoint carry and clock, including the synthetic-inlet seed. Discrepancy
parameter substitution clones the donor exactly: failure jitter would associate
the accepted forecast with coefficients it never used. The failed member's
original snapshot remains available for whole-window rollback. The disabled
backend path does not capture checkpoints or change failure jitter.

A cold checkpoint has no native carry. Repeated cold calls restart from the
same inputs; a later analyzed warm start without a native carry constructs its
template from that attempt's inputs. This is not an archived post-spinup hidden
state. Joint initial-state ESMDA and cross-model initial-state injection remain outside
the supported discrepancy runner modes. Sequential joint filtering instead
updates each forecast endpoint before the next warm segment.

For the hybrid, the filter and smoother retain separate experiment directories.
Before each window, the smoother imports the filter's validated native carry,
physical clock, and inlet realization, while preserving its own window runtime.
The smoother's analysis-only call rolls back its intermediate endpoints; the
state-only filter then advances with the estimated coefficient vector. Hybrid
SGS runs require `ensemble.failure.policy=raise`: donor remapping across these
separate stacks is not supported yet. Ordinary ESMDA/filter donor handling is
unchanged.

The small native regression case verifies exact repeated cold and warm forecasts
on one and two MPI ranks, including replay after an intervening state/coefficient
change. With zero coefficients the diagnostic multiplier is exactly one, but
the enabled startup refresh can change the first adaptive timestep, so stock
comparison uses a numerical tolerance. The rank regression compares the
extension-minus-stock effect separately at each rank count; it does not assert
that independently initialized stock runs are decomposition-independent.

These checkpoints support replay within a live Python run, not resuming an
interrupted assimilation job. They do not checkpoint the assimilation RNG or
restore forecast output artifacts. Nonzero scalar top-flux cases and discrete
energy/wall budgets remain unverified: startup boundary gradients can depend on
the diffusivity that the enabled refresh subsequently changes. A neutral-case
velocity regression cannot establish that boundary budget.

---

## 5. Preprocessing — Python vs Matlab

uDALES requires preprocessing that converts an STL geometry into IBM (Immersed
Boundary Method) input files (`solid_*.txt`, `fluid_boundary_*.txt`,
`facet_sections_*.txt`, `facets.inp`, `facetarea.inp`). Two paths exist:

**The floor is a wall only under STL ground facets.** With an STL the
domain bottom is not a wall: `modibm.f90::bottom` applies its wall function
only with `lbottom`, which stays `.false.`, and the ghost `u0(:,:,kb-1)` is
never set, so an uncovered floor carries molecular stress only (free-slip).
`ForwardModel._fit_stl_geometry` therefore adds ground facets (two triangles
per strip, `utils/stl_utils.fit_stl_to_domain`) where the STL's ground does
not reach along x: xie_and_castro's upstream fetch x < 0 had a free-slip floor
until 2026-10 (PALM's is a rough wall; extending it lowered the time-mean u at
z = 1 m by 0.10-0.18 m/s at x = -10..0). With `outflow_buffer` the grid is
extended by whole cells, buildings beyond `bounds` are dropped (one that
straddles the edge raises), the ground is clipped there and the buffer gets
flat ground, as `stl_to_palm_topography(x_max=...)` does for PALM. The state is
cropped back to `bounds`; a warm start fills the buffer by repeating the last
column, which makes the window-boundary jump about twice as large (mean |Δu|
0.18 vs 0.12 m/s without the buffer, one frame). STLs that already cover the
floor are not rewritten.

**Keep the domain top 1.5 cells above the buildings.** Upstream's
`matchFacetsCells.f90` does not clamp a facet's top cell in z, so geometry
ending in the top cell of the w-grid makes it read past its arrays: the facet
sections then depend on uninitialised memory (zeroed on Linux, random on
macOS, where uDALES stops in `wallfunmom`). `ibm.py` refuses such geometry
with an error naming the highest allowed roof. The tests' tiny grid uses a 15 m
lid over the 10 m roof for this reason.

**Python preprocessor** (default):
[`python_udgeom/`](../libs/pyudales/src/pyudales/python_udgeom/) — a pure-Python
reimplementation of the Matlab `write_inputs.m` workflow.

| Module | Role |
|---|---|
| [`preprocessing.py`](../libs/pyudales/src/pyudales/python_udgeom/preprocessing.py) | `Preprocessing` class — reads namoptions, sets domain/grid properties |
| [`ibm.py`](../libs/pyudales/src/pyudales/python_udgeom/ibm.py) | IBM geometry: solid/fluid point classification + facet-to-cell matching (calls Fortran `IBM_preproc.exe`) |
| [`seb.py`](../libs/pyudales/src/pyudales/python_udgeom/seb.py) | Surface energy balance input generation |
| [`write_inputs.py`](../libs/pyudales/src/pyudales/python_udgeom/write_inputs.py) | Orchestrator — calls ibm/seb, handles `precomputed_geom_dir` bypass |

The generated `config.sh` shell-quotes paths, preserving spaces and treating
shell metacharacters as literal path characters.

The Python preprocessor is invoked via `shell_scripts/write_inputs.sh`, which sets
`DA_EXPDIR` and `DA_TOOLSDIR` environment variables and runs the Python script
against the experiment directory. It uses `trimesh` for STL loading.

**Matlab preprocessor** (legacy, optional):
Calls `u-dales/tools/write_inputs.sh` and needs `matlab_bin` (unset by default)
to point at your MATLAB binary.
The Matlab path sleeps 90 s after subprocess launch to wait for MATLAB to finish.

**Selector.** The `prepare._target_` in
[`configs/model/pyudales.yaml`](../configs/model/pyudales.yaml) points at
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

pyudales applies inflow via **nudging** (under both boundary conditions)
unless `inlet_turbulence.enabled` is set, which
replaces both the nudged inlet face and the interior relaxation with synthetic
driver planes (§6.1). The nudging generates a
`timedepnudge.inp.<expnr>` file and enables `&PHYSICS ltimedepnudge=.true.`
(the inlet face follows it) and `lnudge` (the interior relaxation) in
namoptions. `nudging_config.interior_nudging` decides `lnudge` under
`inflow_outflow`: `false` since 2026-10, as PALM has no interior forcing
there (it was `true`; see docs/pypalm.md §8 "Cross-model parity with
pyudales"). Periodic runs always nudge the interior, their only driver.
`tnudge`, `nnudge_meters` and `profile_config` come from `flow` in
`configs/common.yaml`, shared with pypalm.

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
  which has no body force). Under periodic BCs the case template's `dpdx`/`dpdy`
  are kept: the `pressure_gradient_magnitude` parameter is not written by any
  uDALES path (it stays in the parameter schema as a neural-surrogate input).

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
over the raw `nnudge` key.

**`tnudge`**: nudging relaxation timescale in seconds (default 15.0).

### 6.1 Turbulent inlet — synthetic driver planes

uDALES v2.2.0 has two inlet-turbulence routes. The Lund (1998) recycling
generator is dead code (documented below, and still asserted against the Fortran
source by `tests/pyudales/test_udales_inlet_turbulence.py`). The **precursor/driver**
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
(plane layout + orchestration) on top of
[`utils/driver_file_utils.py`](../libs/pyudales/src/pyudales/utils/driver_file_utils.py)
(binary I/O). The digital filter itself, the per-member seed and the clock
below live in
[`pyurbanair/utils/digital_filter.py`](../src/pyurbanair/utils/digital_filter.py),
shared with pypalm's turbulent inflow (docs/pypalm.md §8). **Absent, `{}`, or `enabled: false` writes nothing at all** —
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
profile. A `spinup_time` sized from nudging-path experience will be too short — it is a
flow-through-time requirement, roughly 6 x `Lx / (bulk streamwise speed)`.
Measured on a 764 m domain given 2 flow-throughs, the first saved frame still
carried a 27% inlet->outlet velocity deficit. The random-geometry generator
sizes it per geometry (`training_data.adaptive_spinup`, see
[neural_surrogates.md](neural_surrogates.md)).

**`profile_config.z_ref` defaults to the domain top.** `build_profile_shape`
falls back to `zsize` when `z_ref` is unset (`vertical_profile.py`
`_power_law`), so `velocity_magnitude` means "the speed at the domain top" and
silently changes meaning whenever `z_size` does — the same nominal 7.5 m/s is a
16% slower canopy wind at `z_size` 128 than at 64. Pin `z_ref` explicitly on
any dataset that will be compared with, or fine-tuned from, another generated
at a different height.

**`intensity` is bounded above by a dead safety net.** The generated field is
Gaussian and unclipped, and uDALES' own inlet guard is commented out — the
`max(0., …)` in `modboundary.f90:687-706` is dead on all six lines (u0/um,
v0/vm, w0/wm). A negative driver sample is therefore written verbatim onto the
west face *and* into the pinned pressure BC `bcpup`. Since `sigma_u =
intensity * |U(z)|` with constant TI in z, the backflow rate is
`Phi(-1/intensity)` everywhere: ~4e-11 at `intensity: 0.15`, but **3e-5 at
0.25** — hundreds of backflow cells per window on a 200x32 inlet plane. Keep
`intensity` at or below ~0.15 unless you first restore that guard.

**Shipped values.** `configs/model/pyudales.yaml` carries values chosen for
the `realistic` STL pool at 4 m spacing; its comments say which are still test
values. Three things
still bias the realised turbulence below its nominal value, all of which
calibration should account for before reaching for more amplitude:

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
[docs/plans/implemented/udales_inlet_turbulence.md](plans/implemented/udales_inlet_turbulence.md).

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

### 6.2 Per-member initial-condition seed (`per_member_irandom`)

uDALES perturbs the cold-start velocity field (`&RUN lrandomize`, default
`.true.`, amplitude `randu`) with the seed `irandom`, which defaults to 43. The
truth and every member therefore share one realisation, so a laminar or
periodic ensemble at fixed parameters has no realisation spread.
`forward_model.per_member_irandom: true` writes
`irandom = derive_seed(experiment_name) % 134456` into each run's namoptions
(truth included) through `random_utils.apply_random_initial_condition`, from
`_apply_inflow_settings`. It uses the same stable per-member digest as the
inlet-turbulence seed (§6.1). The modulus is uDALES' own generator period
(`randomnize`: `ir = mod(ir*8121 + 28411, 134456)` in 32-bit integers); a
larger seed overflows `ir*8121` and biases the perturbations. It only matters on a cold start: a warm-started
run inherits its copied state, and uDALES does not re-randomize. Off by
default, so default runs are unchanged.

### 6.3 Parity with pypalm

The boundary condition, outflow buffer, inflow profile, nudging and inlet
turbulence values are shared with pypalm through `flow` in
`configs/common.yaml` (cases override it: `xie_and_castro` sets a 40 m buffer,
`xie_and_castro_periodic` periodic without inlet turbulence). The item-by-item
audit, the residual differences (SGS closure above all, the open lid, the
initial state) and the recommended spin-up per setup are in
[docs/pypalm.md §8 "Cross-model parity with pyudales"](pypalm.md#cross-model-parity-with-pyudales).
Output frames are labelled with their target times (dt, 2·dt, …,
`simulation_time`) like pypalm's and pylbm's, not with uDALES' dump times
(the first step past each target, e.g. 1.076 s); before 2026-10 they carried
the dump times.

## 7. Instability watchdog

[`utils/run_monitor.py`](../libs/pyudales/src/pyudales/utils/run_monitor.py)

`run_with_dt_watchdog` launches uDALES through the shared
[`pyurbanair.utils.solver_process.run_solver`](../src/pyurbanair/utils/solver_process.py),
which every backend uses: the `bash → mpiexec → MPI-rank` tree runs in its own
process group, is killed on return or error, and dies with its owner even when
that Python process is killed without cleanup. A failure raises a
`CalledProcessError` carrying the tail of `run.<expnr>.log`. The watchdog tails
that log in a polling loop:

```
line pattern: "... dt:  0.242654880"   ← parsed by _DT_RE
```

`InstabilityCheck.from_config(config_dict)` builds the watchdog config from the
`instability_check:` block in the forward model config. Unknown keys are warned
and ignored.

| Field | Default | Meaning |
|---|---|---|
| `enabled` | `True` | When False, falls back to bare `subprocess.run(check=True)` |
| `min_dt` | `1e-3` s | Absolute timestep floor (config value; a collapsed run sits near 1.6e-4 s, normal steps are 0.05-0.3 s at 2 m) |
| `patience` | `20` | Consecutive sub-floor steps before killing |
| `warmup_steps` | `20` | Ignore the first N steps (legitimate ramp-up) |
| `poll_interval_s` | `2.0` | Log re-read interval |

When the patience criterion trips, the process group receives `SIGTERM` (with a
`SIGKILL` fallback after 5 s), and `CalledProcessError` is raised — the same
exception the ensemble layer already treats as a member failure. The
`resample_from_successes` failure policy then substitutes the diverging member
from a random successful donor without waiting for the slow dt-collapse crash.

**Known gotchas:**
- A dt collapse under `inflow_outflow` is a real instability at the **outflow
  face**, not a marginal time step (see "Outflow backflow and the SGS
  constant" below). The watchdog catches it only because `min_dt` is above
  the ~1.6e-4 s the collapsed run settles at; at the old 1e-4 floor such a
  run crawled on for hours.
- Stale fielddumps can give NaN-padded or duplicate z-coordinates after a failed
  worker is retried. `run_single` automatically cleans the member output directory
  before restoring/staging any warmstart files for each attempt.

### Outflow backflow and the SGS constant

The default `c_vreman` is **0.12**: Vreman's (2004) `c = 2.5 Cs²` for
`Cs = 0.217`, the Smagorinsky constant uDALES derives from its own Deardorff
constants (§4). Until 2026-10 it was 0.24 (`Cs = 0.31`, the old Smagorinsky
`cs = 0.24` carried over to `c_vreman`), because lower values "collapsed dt" on
xie_and_castro. That collapse is the outflow, not the closure:

- The outflow (`xmo_convective` for v, w and `bcpup` for u, in
  `modboundary.f90`) advects every variable out at one speed, the bulk mean
  `uouttot`, with a Neumann u in the lowest cell. Reversed flow reaching the
  outflow face turns it into an inflow the condition cannot handle. CD2
  momentum advection (the only one built, §Quick reference) adds no
  dissipation, so the SGS viscosity is the only damping there.
- With the outflow at x = 80 m (5 m behind the last cube column, no buffer)
  the wake reaches the face. At c = 0.10 with turbulent inflow, dt drops from
  0.15 s to 0.04 s at 85 s and settles near 1.6e-4 s by 110 s. Over that time
  u on the last interior u face (x = 78 m, z = 3-9 m) goes from -9.6 to
  -6000 m/s, while |u| stays below 7 m/s everywhere else until the blow-up
  spreads. The run never NaNs; the adaptive step shrinks with the velocity.
  Without the buffer dt falls below 0.01 s at ~80 s for c = 0.07, ~110 s for
  0.10 and 120-230 s for 0.12 (laminar and turbulent inflow); 0.24 runs clean.
- The 20 m flat `flow.outflow_buffer` of xie_and_castro (since 2026-10)
  removes the trigger. With it c = 0.12 is stable for 1020 s, at 10 m/s, and
  over three warm-started `params=dynamic_truth` windows. So is c = 0.07,
  also at 30° and periodic. Backflow still reaches the face (5 % of the
  samples, 29 % at z = 1 m, where the mean u is 0.3 m/s), so the margin is
  finite: c = 0.05 runs, and c = 0.04 with laminar inflow diverges at the
  face (z = 1 m) after ~380 s; a 40 m buffer makes that run stable. A case
  without a buffer, or with buildings close to `xmax`, needs one before its
  SGS constant is lowered.
- 20 m is not enough at c = 0.12 (2026-10-21; 6 m/s, z0 0.1, 120 s spin-up +
  600 s, 4 seeds per arm). Backflow bursts still got through in 3 of 40 runs:
  u at the face fell to about -25 m/s, |U| reached 21 m/s at x = 73-79 m, and
  the run recovered by itself in ~50 s without tripping the dt watchdog. With
  40 or 60 m there were none in 36 runs. Without bursts, 20 m also inflated
  street-level TKE at x = 65-79 m 2-5x at 40° (+10-40 % at 10°); 40 m and 60 m
  agree within the replica noise there, and upstream of x ≈ 55 m the buffer
  changes nothing. The case default is therefore 40 m (+18 % wall time;
  60 m costs +45 %).
- Why not the literature 0.07 (`Cs ≈ 0.17`, Lilly's value for a sharp
  spectral cut-off)? CD2 does not dissipate at the grid scale, and at 0.07
  energy piles up there. Spanwise spectra of u' at z = 9 m (inflow_outflow,
  turbulent) and in the periodic case, compensated by k^5/3: they rise toward
  the 2Δ cut-off for c ≤ 0.10, are flat to slightly falling at 0.12, and fall
  to a third of the inertial level at 0.24. The share of u' variance at
  wavelengths under 2.7Δ is 7-10 % at 0.07, 3-5 % at 0.12, 1-2 % at 0.24 and
  ~3 % for a k^-5/3 spectrum. Same criterion for any grid: the lowest c
  whose spectrum does not rise at the cut-off.
- Effect on the flow (2 m grid, 5 m/s, statistics over 150-420 s, against
  PALM with its ws-scheme). Laminar inflow: uDALES' wakes are nearly steady
  at 0.12 and 0.24 (canopy TKE 0.013 and 0.001 m²/s², PALM 0.003); they shed
  at 0.07 (0.078). Turbulent inflow: canopy TKE 0.10 (0.24), 0.15 (0.12),
  0.22 (0.07), PALM 0.13-0.15. The compare_models TKE-profile distance to
  PALM is 0.041, 0.017 and 0.068 m²/s². The near-ground canopy mean is NOT an
  SGS-constant effect: array-mean u at z = 1 m is 1.33-1.34 m/s for c =
  0.12-0.24 and 1.20 at 0.07, against PALM's 0.87-0.88. The sensor-mean
  distance stays 0.42-0.46 m/s for every c. Periodic runs barely depend on c.
- The `model_discrepancy` patch at zero coefficients matches the stock
  closure. The runs differ by 1e-3 m/s after the 120 s spin-up (the chaotic
  growth of round-off), and their statistics agree (canopy TKE 0.2259 vs
  0.2265 at c = 0.07).
- The closure stays Vreman. In this build Smagorinsky has its Piomelli wall
  damping commented out (`damp = 1` in `closure`), and the one-equation model
  cannot be used (§4). Vreman's viscosity vanishes in pure shear, so it needs
  no wall damping.

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
    "roughness_length",
    "roughness_length_wall",
    "roughness_length_ground",
)
```

| Parameter | Where written | Notes |
|---|---|---|
| `inflow_angle` | `u0`/`v0`/`dpdx`/`dpdy` in namoptions + nudging files | Degrees from +x axis |
| `velocity_magnitude` | `u0`/`v0` in namoptions + nudging profiles | m/s reference speed |
| `pressure_gradient_magnitude` | `dpdx`/`dpdy` in namoptions | Pa/m; decomposed by angle. uDALES-only parameter. |
| `vertical_inflow_exponent` (α) | Overrides `profile_config["alpha"]` in `_resolve_nudging_config` | Estimated by ESMDA; per-member shear |
| `roughness_length` | z0 column of `factypes.inp.<expnr>` via `_apply_roughness` | m; see "Roughness" below |
| `roughness_length_wall`, `roughness_length_ground` | same, per facet class (walls get their own facet type) | m; both instead of `roughness_length`; uDALES only |
| `sgs_constant` | `&NAMSUBGRID cs` (Smagorinsky) or `c_vreman` (Vreman) via `_apply_sgs_setting` | Dimensionless SGS constant; the target key is picked from the case's `lsmagorinsky`/`lvreman` switches (§4). The two constants are on different scales (`c_vreman ≈ 2.5 cs²`: 0.217 ↔ 0.12). |

**`pressure_gradient_magnitude`** is the third parameter unique to pyudales.
`resolve_parameter_schema` in `hydra_helpers.py` adds it to the uDALES parameter
schema; the sampler configs include it as a `Constant` which other backends ignore.

**Model-error knobs** (`vertical_inflow_exponent`, `sgs_constant`) are applied
unconditionally (outside the static/time-varying branch), so they affect every
run where they are present. Each is a no-op when absent, keeping default runs
byte-identical. See [codebase_guide.md §7](codebase_guide.md) "Model-error
compensation knobs".

### Roughness (`roughness_length`)

z0 for momentum (m), one value for every wall, shared with pypalm. It comes
from the params Dataset, else from `forward_model.roughness_length`
(`flow.roughness_length`: null in `common.yaml`, which leaves `factypes.inp`
alone; xie_and_castro sets 0.05). `_apply_roughness` writes it, as `%10.6f`,
into the z0 column (3) of the member's `factypes.inp.<expnr>` for every facet
type that `facets.inp` uses and whose z0 is > 0 (type 1 today; the free-slip
bounding walls, z0 = 0, are left alone; z0h is unused in neutral runs). uDALES
reads the file at every launch (`initfac.f90`), so z0 changes per member and
window without redoing the IBM preprocessing. The write runs in
`_apply_inflow_settings`, so every member and window gets it, also after a
window-checkpoint restore. When the file already holds the value (the default
0.05 against the preprocessing's `%6.2f` 0.05) nothing is written. z0 must be
at least 1e-6 m: uDALES treats z0 = 0 as free slip, and anything smaller would
be written as 0.000000 and stay free slip in every later window.

**Wall and ground z0.** Instead of `roughness_length`, the params may hold both
`roughness_length_wall` and `roughness_length_ground` (one of them alone, or
either with `roughness_length`, raises; so does `check_config`). Walls are the
facets whose `facets.inp` normal has |n_z| < 0.5; **ground is every other
facet, the ground and the roofs**. `set_facet_roughness` moves the walls of a
type shared with horizontal facets to a new type (id = largest in
`factypes.inp` + 1, its row a copy of the original's) by rewriting their type
in `facets.inp`, then writes each class's z0; a rerun of the preprocessing in
the same experiment dir reuses the unused copy an earlier split left instead
of appending another. In a neutral run uDALES uses a facet's type only to look
up its `factypes.inp` row (`initfac.f90` `initfac`: z0, z0h, lGR, emissivity
and the layer properties; the copy keeps all but z0 equal), and the IBM files
reference facet ids, not types, so no preprocessing is redone. Thermal runs
(`lEB`, `iwallmom`/`iwalltemp`/`iwallmoist` = 2) also read the type's sign
(`initfac.f90`: > 0 is not a floor for the inner-layer facet temperature,
< -100 skips the energy-balance layers), so there splitting a negative floor
type that also covers walls would turn those walls into "building" facets.
After the split every type is all-wall or all-horizontal, so later windows
only rewrite z0. The fallback warning uses the larger of the two. PALM has a
single z0 and raises with the pair ([pypalm.md](pypalm.md) "Roughness").

The wall function is the neutral log law evaluated at the **true distance d**
of each section's fluid point from its facet (`facet_sections_*.txt`, column 4),
not at a half cell as in PALM. If `ln(d/z0) <= 1` (d <= e·z0) the velocity is
taken from a reconstruction point further out, and if that is still too close
the section gets **no wall stress at all** (`modibm.f90` `ibmwallfun`,
silently). On the 2 m xie_and_castro grid the ground sections have d = 1 m,
but **every wall section has d = 2 m** (u, v and w) and the roofs 0.2-1.8 m
(2.8 m roof: 0.2, 6.4: 0.6, 10: 1.0, 13.6: 1.4, 17.2: 1.8). The cube faces
(x, y = 5 + 10n) and those roofs lie off the grid faces, while the IBM blocks
the flow at the voxel faces, 1 m from the first fluid point, exactly as PALM's
topography does (same 12 m blocks, same roofs 2/6/10/14/18 m). So uDALES
applies the wall law 1 m further out than the wall it actually resolves, and
over the true facet areas (10 m faces), not the blocked ones. On the same flow
that is ~0.5x PALM's wall friction and ~0.65x its roof friction at the same
z0; ground is the same. It is what makes uDALES' canopy faster than PALM's and
what a cross-model z0 estimate absorbs (`docs/pypalm.md` § "Cross-model
parity", wall treatment). For a wall, ln(2/z0') = ln(1/z0) at z0' = 2 z0. The
reconstruction fallback hits the 0.2 m roof sections from z0 = 0.074 m (6% of
the roof area), and 41% of all sections from z0 = 0.37 m (1 m/e). `_apply_roughness` warns when more than a quarter of the
sections lie within e·z0. In the tiny test case (1 m cells) a z0 of 0.01 -> 0.1
lowers the time-mean first-cell u by ~4% (1.00 -> 0.97 m/s).

---

## 9. `utils/` subpackage

| Module | Purpose |
|---|---|
| [`clean_up_utils.py`](../libs/pyudales/src/pyudales/utils/clean_up_utils.py) | `clean_output_dir` (delete output), `clean_temp_dir` (wipe experiment dir except namoptions/STL/config.sh) |
| [`config_utils.py`](../libs/pyudales/src/pyudales/utils/config_utils.py) | `create_config_sh` — writes `config.sh` with `DA_EXPDIR`, `DA_TOOLSDIR`, `DA_BUILD`, `DA_WORKDIR`, `NCPU`, `MATLAB_BIN` |
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
| [`random_utils.py`](../libs/pyudales/src/pyudales/utils/random_utils.py) | `apply_random_initial_condition` — sets `irandom`/`randu`/`lrandomize` in `&RUN` (used by `per_member_irandom`, §6.2) |
| [`rollout_utils.py`](../libs/pyudales/src/pyudales/utils/rollout_utils.py) | `collect_rollout_results` — concatenate per-window result files along time |
| [`run_monitor.py`](../libs/pyudales/src/pyudales/utils/run_monitor.py) | `run_with_dt_watchdog`, `InstabilityCheck` (see §7) |
| [`save_frequency_utils.py`](../libs/pyudales/src/pyudales/utils/save_frequency_utils.py) | `apply_output_frequency` (writes `tfielddump`), `apply_save_only_last_timestep` (sets `tfielddump=runtime`) |
| [`vertical_profile.py`](../libs/pyudales/src/pyudales/utils/vertical_profile.py) | `build_profile_shape` — `uniform` or `power_law` shear profile `s(z)` |
| [`warm_start_utils.py`](../libs/pyudales/src/pyudales/utils/warm_start_utils.py) | `set_warm_start`, `set_trestart`, `fetch_carry`/`store_carry`/`copy_carry`, `update_warmstart_file_from_xarray`, `identify_generated_warmstart_file` |

---

## 10. Config wiring — `configs/model/pyudales.yaml`

[`configs/model/pyudales.yaml`](../configs/model/pyudales.yaml) is the complete model
config entry. Notable fields (values illustrative: the file is the source of
truth and is tuned between runs):

```yaml
name: pyudales
solver_name: udales          # selects dim_mapping in ObservationOperator

forward_model:
  _target_: pyudales.forward_model.ForwardModel
  case_dir: ${geometry.case_dir}
  precomputed_geom_dir: ${oc.select:geometry.udales_precomputed_geom_dir,null}
  temp_dir: ${paths.experiment_dir}
  experiment_name: "999"
  matlab_bin: null            # only python_or_matlab: matlab needs it
  ncpu: 1
  boundary_condition: ${flow.boundary_condition}   # shared flow setup, configs/common.yaml
  outflow_buffer: ${flow.outflow_buffer}
  closure: vreman             # smagorinsky | vreman | null (keep template)
  sgs_constant: 0.12          # c_vreman = 2.5 * 0.217^2; see §7
  per_member_irandom: false   # true -> one cold-start seed per run (§6.2)
  nudging_config:
    tnudge: 15.0
    nnudge_meters: 16.0         # skip nudging below this height (m)
    interior_nudging: false     # true -> also relax the interior (inflow_outflow)
    profile_config:
      type: power_law
      alpha: 0.25
  instability_check:
    enabled: true
    min_dt: 1.0e-3
    patience: 20
    warmup_steps: 20
    poll_interval_s: 2.0
  inlet_turbulence:
    enabled: ${flow.inlet_turbulence.enabled}  # synthetic driver planes, BCxm=3 (§6.1)
    intensity: 0.15           # u'_rms / |U_mean(z)|
    length_scale_x/y/z: 24/12/12   # m, digital-filter integral length scales
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
Its only caller is the neural surrogate's `disable_spinup`, which forwards to
its `spinup_forward_model`; warm-start windows zero `spinup_time` in
`run_single` themselves.

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
its own process group (`run_solver`). With `ncpu=25` per member and
`num_parallel_processes=4`, 100 MPI ranks run simultaneously. The DRAM-bandwidth
ceiling on the development box is ~4–8 parallel processes (see
[ensemble_scaling.md](archive/ensemble_scaling.md)).

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
| Tune the instability watchdog | `instability_check:` block in `configs/model/pyudales.yaml` |
| Interpolate staggered → centred | `pyudales.utils.grid_utils.interpolate_grid(ds)` |
| Change nudging height cutoff | `nudging_config.nnudge_meters` in model config |
| Understand ncpu → nprocx mapping | `utils/ncpu_utils.validate_and_sync_ncpu` |
