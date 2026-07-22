# Implementation Plan: Nudging-Driven Periodic Runs in pypalm (no geostrophic forcing, no Fortran changes)

> Working note (docs/plans): drafted 2026-07-21 against `main` (the
> `feat/geostrophic-forcing-ug-vg` branch was **not** merged; pypalm has no
> geostrophic code). Design mirrors pyudales's nudging driver for periodic
> runs so the two backends drive the flow with the same physics and the same
> parameter meaning. Solver claims verified line-by-line in the vendored
> `libs/pypalm/palm_model_system` source. Reviewed and re-verified against
> source 2026-07-21: line citations corrected, the inert-`LSF_DATA` parsing
> walkthrough confirmed, and the `nudge_ref` / dt-vs-tnudge open questions
> settled (see "Risks").

## Goal

Give pypalm a flow driver for **periodic** (`boundary_condition: periodic`)
runs that relaxes the domain toward the wind prescribed by the existing
`inflow_angle`/`velocity_magnitude` params (static **or** time-varying, plus
the `vertical_inflow_exponent` shear knob) — the exact counterpart of
pyudales's periodic nudging path. Explicitly **not** geostrophic forcing: the
template's `omega = 0` stays, ug/vg remain inert initializers.

Why this matters on `main` today:

- **Periodic PALM runs are currently un-driven.** `ug_surface`/`vg_surface`
  and `u_profile`/`v_profile` are written (`forward_model.py:380-401`) but
  with `omega = 0` they are pure initial conditions — a cyclic run decays.
- **Periodic + time-varying params are mis-routed.** `_apply_inflow_settings`
  sends any time-dimmed params to `apply_time_varying_inflow`
  (`forward_model.py:349-369`), which stages a dynamic driver and enables
  `&turbulent_inflow_parameters` — an inflow_outflow mechanism, wrong under
  cyclic BCs. The new periodic branch supersedes this routing.

## Verified solver facts

All in `palm_model_system/packages/palm/model/src/` unless noted.

1. **PALM's nudging is the same scheme as uDALES's.** `nudge`
   (`large_scale_forcing_nudging_mod.f90:1201-1333`) adds
   `−(hom(k,1,·,0) − target(k,t))/tmp_tnudge(k)` at every fluid grid point
   (the tendency is masked by `topo_flags`, so points inside buildings are
   excluded and the loop runs `nzb+1..nzt`) — a relaxation of the
   *horizontal-mean* profile toward the target, applied volume-wide, target
   linearly interpolated in time. The `hom` statistics column differs per
   quantity (u: `hom(k,1,1,0)`, v: `hom(k,1,2,0)`, …). The divisor is
   `tmp_tnudge = MAX(dt_3d, tnudge(k,t))` (`calc_tnudge`, `:1191`). Called
   per prognostic quantity from `prognostic_equations.f90`
   (`IF ( nudging ) CALL nudge(...)`).
2. **Native time dependence.** Targets come from an ASCII `NUDGING_DATA` file
   of time blocks (reader loop `:1045-1112` in `nudge_init`, `:990-1161`);
   each block is `# <time>` followed by rows
   `height  tnudge  u  v  w  pt  q`, height-interpolated onto the grid at
   init and time-interpolated every step. (No header lines — unlike
   `LSF_DATA`.) `tnudge` is a **per-height, per-time column** — uDALES's
   `nnudge_meters` near-wall exemption is emulated by a huge `tnudge` below
   the cutoff height.
3. **Selective nudging via sentinels.** Any quantity whose column is
   `-999999` in every row/time has its nudging disabled entirely
   (`nudge_u/v/w/pt/q` flags, `:1120-1144`). We nudge **u and v only**; w,
   pt, q stay untouched — cleaner than uDALES, where scalars ride along under
   `ltempeq`/`lmoist`. `nudge_ref` (`:1457`) guards every assignment with
   these flags, so sentinel columns cannot leak into `pt_init`/`q_init`
   (and `nudge_init` guards the initial-profile overwrite the same way,
   `:1148-1152`).
4. **The constraint chain** (`lsf_nudging_check_parameters`, `:210-250`, all
   PALM-fatal):
   - `nudging` requires `large_scale_forcing = .T.` (LSF0001);
   - LSF requires cyclic lateral BCs (LSF0002) — satisfied, that's our case;
   - LSF requires `humidity = .T.` (LSF0003);
   - LSF is incompatible with `passive_scalar` (LSF0004);
   - non-flat topography (our buildings) requires `lsf_exception = .T.`
     (LSF0005), an upstream bypass (the module-header `@todo` at `:30` marks
     the whole lsf_exception/lsf_surf/lsf_vert flag set for revision).
   (There is also LSF0006 — LSF incompatible with `ocean_mode` — irrelevant
   here.) All four switches (`nudging`, `large_scale_forcing`,
   `lsf_exception`, `humidity`) are `&initialization_parameters` namelist
   entries (`parin.f90:226-246`).
5. **`LSF_DATA` must exist but can be inert.** `lsf_init` (`:452-652`) reads
   3 header lines, then surface rows `time shf qsws pt q p`, then `# <time>`
   profile blocks of `height ug vg w_subs td_lsa_lpt td_lsa_q td_sub_lpt
   td_sub_q`. Both halves have **non-fatal** disable paths: a first surface
   time beyond `end_time` sets `lsf_surf = .FALSE.` (LSF0012, warning-level
   `message(..., 0, 1, ...)`, `:541-547`), and a first profile time beyond
   `end_time` exits before reading any rows (`:575`) and sets
   `lsf_vert = .FALSE.` (LSF0016, `message(..., 0, 0, ...)`, `:642-648`).
   So an "inert" `LSF_DATA` keeps
   `large_scale_forcing = .T.` legal while contributing zero physics — the
   nudging is then the *only* large-scale term. (The ug/vg columns in a
   non-inert LSF_DATA would anyway be dynamically dead with `omega = 0`.)
6. **Reader mechanics that constrain the inert-file shape** (`:523-573`),
   **confirmed by walkthrough**: the surface loop's `DO WHILE` tests
   `time_surf(0) = 0` first, so it always reads exactly one surface row
   before the LSF0012 check — it does not skip the read. The skip loop
   `READ (finput,*) r_dummy` (`:551-553`) then consumes records until a
   read error — the record that *errors* is still consumed, and a bare `#`
   errors a list-directed REAL read. The profile search then needs its own
   `# <time>` line. The inert file therefore needs one sacrificial
   separator line (a bare `#`) between the surface row and the `# <time>`
   marker (exact layout below); without it the skip loop would swallow the
   `# <time>` marker and the profile search would hit fatal LSF0013. The
   walkthrough confirms the layout below reaches LSF0016 with no fatal
   error; the first smoke test remains as insurance.
7. **Staging is already half-wired.** palmrun's `.palm.iofiles` maps
   `<name>_nudge → NUDGING_DATA` and `<name>_lsf → LSF_DATA` as optional
   inputs (`palm_model_system/packages/palm/model/share/config/.palm.iofiles:12-13`).
   The default direct-run path needs two entries added to
   `INPUT_FILE_MAP` (`src/pypalm/direct_palm.py:46-51`; absent source files
   are already skipped, matching the optional `inopt` semantics).

## Design

### Driver selection

| `boundary_condition` | params | driver |
|---|---|---|
| `periodic` | static or time-varying | **nudging** (new): `NUDGING_DATA` + inert `LSF_DATA` + the four namelist switches; stale dynamic driver removed, `turbulent_inflow` disabled |
| `inflow_outflow` | static | current static path, unchanged; stale `_nudge`/`_lsf` files removed, switches forced `.F.` |
| `inflow_outflow` | time-varying | current dynamic-driver/turbulent-inflow path, unchanged (plus the same `_nudge`/`_lsf` cleanup) |

**One deliberate behavior change, flagged for sign-off:** periodic runs stop
being un-driven. This deviates from the strict no-op invariant, on the
grounds that a decaying cyclic flow is not a baseline any experiment relies
on, and uDALES parity (its static periodic runs are nudged) is the point of
the feature. Escape hatch: `nudging_config.enabled: false` restores today's
undriven staging exactly.

The initial condition keeps the existing writes untouched:
`ug_surface`/`vg_surface` (inert with `omega=0`) and
`u_profile`/`v_profile`/`uv_heights` from the t=0 parameter values, so the
run starts consistent with the initial nudging target.

### New writers — `src/pypalm/utils/nudging_utils.py`

**`write_nudging_data(path, times, heights, tnudge_column, u_profiles,
v_profiles)`** emits, per time snapshot:

```
# <time_seconds>
<z_0>   <tnudge(z_0)>   <u(z_0)>   <v(z_0)>   -999999.0  -999999.0  -999999.0
...
<z_top> <tnudge(z_top)> <u(z_top)> <v(z_top)> -999999.0  -999999.0  -999999.0
```

- Heights: PALM's **0-based native grid** (`np.arange(nz)*dz + 0.5*dz`, no
  `zmin` offset), plus a `z=0` anchor and a top row comfortably above
  `zu(nzt+1)` (top + 2·dz) — the reader height-interpolates against its own
  0-based `zu` and errors if the profile tops out below the model grid
  (LSF0019, `nudge_init:1077-1096`, fatal). The profile *shape* is still
  evaluated at physical heights (`cell_heights`, which includes `zmin`) —
  the same 0-based-axis / physical-shape split `dynamic_driver_utils.py`
  already makes for the dynamic driver (`:264-283`), so a non-zero `zmin`
  is honoured without misaligning the file against `zu`.
- `tnudge_column`: `tnudge` (default 15.0 s, uDALES parity) above the
  `nnudge_meters` cutoff (default 4.0 m); `1.0e9` below it, with paired rows
  at `cutoff ∓ ε` for a sharp transition.
- u/v values: `angle_to_velocity(inflow_angle(t), velocity_magnitude(t))`
  shaped by `build_profile_shape(profile_config)` — identical construction to
  pyudales's `compute_nudging_profiles`, including the α override from
  `vertical_inflow_exponent` (already plumbed via `_resolve_profile_config`).
- Time schedule: **reuse pypalm's existing builders** — `_extract_schedule`
  and `_prepend_spinup_plateau` in `utils/dynamic_driver_utils.py` already
  implement exactly these semantics (static params → a 2-snapshot constant
  schedule; time-varying → the params' `time` array; `spinup_time > 0` → a
  constant plateau at the initial values prepended). Do not re-implement
  them in `nudging_utils.py`; promote them to shared helpers if the import
  direction is awkward. On top of that schedule, always append a terminal
  snapshot past `end_time = simulation_time + spinup_time` (the time
  interpolator needs a bracketing upper snapshot).
- w, pt, q columns: sentinel `-999999.0` everywhere (fact 3).

**`write_inert_lsf_data(path, end_time)`** emits exactly:

```
# pyurbanair inert LSF_DATA — exists only to satisfy nudging's
# large_scale_forcing requirement; lsf_surf and lsf_vert both disable.
# columns(surface): time shf qsws pt q p
<end_time + 1e6>  0.0  0.0  0.0  0.0  0.0
#
# <end_time + 1e6>
```

Line-by-line against the reader (facts 5–6, confirmed by walkthrough): 3
header lines; one surface row beyond `end_time` → `lsf_surf=.F.` (LSF0012,
warning); the bare `#` is consumed by the skip loop; the `# <time>` marker
is found by the profile search, its time is beyond `end_time` → immediate
exit, `lsf_vert=.F.` (LSF0016, info). Fallback, should the vendored source
ever drift: zero-valued surface rows and profile blocks spanning
`[0, end_time+pad]` (physically inert too, just wordier).

### Wrapper changes — `src/pypalm/forward_model.py`

Restructure `_apply_inflow_settings` around the driver table:

- Periodic branch (when `nudging_config.enabled`, default true): resolve the
  t=0 scalars for the init writes (reuse the existing "returns a scalar
  params Dataset holding the t=0 values" convention), call both writers into
  `dirs.input_dir` as `<experiment_name>_nudge` / `<experiment_name>_lsf`,
  set `nudging/large_scale_forcing/lsf_exception/humidity = .T.` via
  `P3DFile`, call `disable_turbulent_inflow` + `remove_dynamic_driver_file`
  (today's static-path hygiene). Log one line naming the driver, tnudge, and
  the snapshot count. The writers need `bounds` and `nz` for the heights
  column — raise a `ValueError` naming them when absent (mirroring the
  time-varying branch's existing check at `forward_model.py:350-354`);
  today's static periodic path silently skips the `u_profile` block instead,
  and that silent degradation must not extend to the nudging driver.
- inflow_outflow branches: unchanged, plus symmetric hygiene — remove stale
  `_nudge`/`_lsf` files and force the four switches `.F.` so a template (or
  a previous periodic run in the same experiment dir) cannot leak the LSF
  apparatus into an inflow run.
- Validation: raise at staging time if the template has `passive_scalar =
  .T.` alongside the nudging driver (PALM would abort with LSF0004 anyway —
  fail with a message naming the conflict and the `nudging_config.enabled`
  escape hatch instead).

`src/pypalm/direct_palm.py`: add `"_nudge": "NUDGING_DATA"` and
`"_lsf": "LSF_DATA"` to `INPUT_FILE_MAP`. palmrun path: nothing (fact 7).

`_stage_input_dir` (`forward_model.py:200`): add `_nudge`/`_lsf` to the
copied suffixes only if we want case-dir templates to be able to ship them —
not required; the writers generate both files per run. Skip for now.

### Config — `conf/model/pypalm.yaml`

Extend the existing `nudging_config` to full uDALES parity:

```yaml
nudging_config:
  enabled: true          # false -> today's un-driven periodic staging
  tnudge: 15.0           # s; matches pyudales
  nnudge_meters: 4.0     # no nudging below this height (via huge tnudge)
  profile_config:
    type: power_law
    alpha: 0.25
```

`DEFAULT_NUDGING_CONFIG` in `forward_model.py` (used when no
`nudging_config` is passed) has no `enabled` key today — either add
`enabled`/`tnudge`/`nnudge_meters` defaults there or read every key via
`.get(key, default)`; pick one convention and use it for all four keys so
the constructor default and the YAML stay in sync.

No params changes anywhere: the driver consumes the existing
`inflow_angle`/`velocity_magnitude`/`vertical_inflow_exponent`, so every
sampler config, `params_to_estimate` selection, and the ESMDA window channel
work as-is. `resolve_parameter_schema` in `hydra_helpers.py` is untouched.

## Cross-backend meaning

With this, periodic runs are nudging-driven in **both** backends with the
same relaxation physics (slab-mean, volume-wide, same tnudge/cutoff/shear
construction) and the same parameter interpretation ("the mean wind the
domain is held to"). Differences that remain, to document in
`docs/pypalm.md`:

- PALM nudges u/v only (sentinels); uDALES also relaxes thl/qt when
  `ltempeq`/`lmoist` are on (both off in our neutral runs — no practical gap).
- PALM's `humidity = .T.` runs a (zero-valued, passive) moisture equation
  that uDALES runs don't — small cost, no dynamics with q ≡ 0 and zero
  fluxes.
- uDALES applies nudging above a *level index* (`nnudge`); PALM via the
  tnudge column — the cutoff is a height in both configs (`nnudge_meters`).

## Costs / limitations

- `humidity = .T.` on all periodic PALM runs (constraint, fact 4): memory +
  one prognostic equation; physically inert at q=0.
- `lsf_exception = .T.` is an upstream bypass slated for revision (fact 4) —
  it disables a guard, and LSF-with-topography is not an upstream-supported
  combination.
  Our use is benign (the LSF file is inert; only the nudging term is active,
  and it has no topography interaction beyond `hom` averaging), but this is
  the plan's main external risk. The first smoke test exercises exactly this.
- **Passive scalars become unavailable in nudging-driven runs** (LSF0004).
  If PALM-side pollutant dispersion is ever needed, it must run under
  inflow_outflow, or this driver needs replacing (e.g. by upstreaming a
  standalone-nudging patch). Worth a prominent note in `docs/pypalm.md`.
- Undocumented corner: `NUDGING_DATA`+`LSF_DATA` are the old FORTRAN-ASCII
  LSF route; upstream's focus has moved to PIDS_DYNAMIC. The files are still
  read and the module still maintained (present in current source), but
  examples are scarce — hence the writer unit tests pin the exact format the
  reader parses.

## Risks / verify during implementation

- **Inert `LSF_DATA` parsing** (fact 6): *settled* — the walkthrough against
  the reader source confirms the layout reaches time integration with
  LSF0012 + LSF0016 and no fatal LSF error. Keep the staging smoke run as
  the first implementation step anyway (buildings present, `lsf_exception`
  path) — it is the only check against a vendored-source drift.
- **`nudge_ref` call sites**: *settled* — it **is** called unconditionally
  whenever `nudging` is on (`time_integration.f90:753-754`), independent of
  subsidence / Rayleigh damping. With only `nudge_u/v` set it re-writes
  `u_init`/`v_init` with the time-interpolated targets each step; those are
  only consumed post-init by Rayleigh damping, which our runs don't enable —
  benign.
- **Interaction with `_apply_warmstart`**: ordering is already safe on both
  counts — `disable_spinup()` runs *before* `_apply_inflow_settings`
  (`run_single:816-825`), so the nudging schedule correctly sees
  `spinup_time = 0` on warm windows and skips the plateau; and
  `_apply_inflow_settings` runs before `_apply_warmstart` writes its LOD=2
  init driver, so the nudging branch's `remove_dynamic_driver_file` cannot
  clobber it. Keep both orderings pinned by the warm-start + periodic +
  nudging regression test.
- **`hom` averaging over buildings**: PALM's slab mean at building-occupied
  levels averages over fluid cells per its statistics conventions — sanity
  check the equilibrium mean wind against the target above the canopy in the
  smoke run (it should sit close to the target above roof level, below it the
  huge-tnudge cutoff applies).
- **dt vs tnudge**: *settled* — PALM does **not** warn; `calc_tnudge`
  silently floors the effective timescale to `dt_3d`
  (`MAX(dt_3d, tnudge...)`, `:1191`). With tnudge=15 s and urban dt ≪ 1 s
  the floor never binds, but keep tnudge ≥ a few dt in tests with coarse
  smoke grids, and don't expect a solver-side warning if it's violated.

## Phases

1. **Writers + format tests** (`utils/nudging_utils.py`): emit both files;
   unit tests assert exact line shapes (block markers, sentinel columns,
   tnudge cutoff rows, terminal snapshot past end_time, inert-LSF layout).
2. **Wrapper branch** (`forward_model.py`, `direct_palm.py`): driver table,
   namelist switches, hygiene on both branches, `INPUT_FILE_MAP` entries,
   the `bounds`/`nz` guard, passive-scalar guard.
3. **Config + smoke validation**: extend `nudging_config` in
   `conf/model/pypalm.yaml`; run the staging smoke test (buildings +
   `lsf_exception` + inert LSF + two-snapshot nudging) through PALM far
   enough to confirm no fatal LSF messages and a driven (non-decaying) mean
   wind.
4. **Tests** (smoke shape, serial):
   - Staging assertions per driver-table row: files present/absent, the four
     switches `.T.`/`.F.`, turbulent_inflow disabled, dynamic driver removed.
   - Time-varying periodic run stages one block per param time + spinup
     plateau + terminal pad; static run stages exactly two blocks.
   - inflow_outflow regressions: staging byte-identical to today apart from
     the (previously absent) `_nudge`/`_lsf` cleanup; dynamic-driver path
     untouched.
   - `nudging_config.enabled: false` → staging byte-identical to today's
     periodic path.
   - Periodic + nudging without `bounds`/`nz` → `ValueError` naming them.
   - Warm-start + periodic + nudging ordering test.
5. **Docs**: `docs/pypalm.md` (new §"Nudging driver for periodic runs":
   constraint chain, escape hatch, passive-scalar limitation, cross-backend
   meaning) and `docs/scripts_and_configs.md` (nudging_config keys). Note
   both backends can then be nudging-driven and time-varying — the old
   PALM-asymmetry caveat is superseded wherever it still appears. (An
   earlier draft pointed at `docs/temp/udales_time_dependent_forcing_problems.md`,
   which does not exist in this repo.)
