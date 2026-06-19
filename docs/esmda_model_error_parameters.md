# Plan — model-error compensation parameters for cross-model ESMDA

Status: **proposal / not yet implemented**. This document is the implementation
plan for adding two new estimable parameters to the ESMDA pipeline so that, when
the truth and assimilation models differ (e.g. `model@truth_model=pyudales`,
`model@assim_model=pylbm`), the smoother has internal physics knobs to absorb the
**model misspecification** instead of corrupting the inflow estimate.

The two new parameters:

| Name | Physical meaning | Was | Becomes |
|---|---|---|---|
| `vertical_inflow_exponent` | power-law exponent `α` of the incident vertical shear `u(z) = u_ref (z/z_ref)^α` | hardcoded `0.25`, identical in every model, fixed at construction | per-model, per-ensemble-member, estimated |
| `sgs_constant` | sub-grid-scale eddy-viscosity constant (turbulent mixing / wake-recovery rate) | template-fixed solver constant | per-model, per-ensemble-member, estimated |

Both are **static** scalars (one value per member, no `time` dim) even in the
time-varying inflow runs — they are model coefficients, not inflow schedules. The
typical run mixes them with dynamic inflow: `inflow_angle` / `velocity_magnitude`
carry a `time` dim while `vertical_inflow_exponent` / `sgs_constant` do not, **all
in one `params` Dataset**, sampled, assimilated and consumed together. See
[§6 Mixing static and dynamic parameters](#6-mixing-static-and-dynamic-parameters-in-one-xarray-dataset)
— this turns out to need no ESMDA-core change.

Companion analysis: the reasoning for *why* these two (vs. roughness `z0`,
nudging `tnudge`, viscosity) is in the chat that produced this plan; the short
version is that `α` covers the near-ground shear bias and `sgs_constant` covers
the wake-recovery / mixing bias, which are the two dominant cross-model error
modes, and both should differ between solvers by design (different
discretizations and closures), which is exactly what a per-model free parameter
expresses.

---

## 1. The core design change

Today neither knob is per-member:

- **`α`** is consumed once in each backend's `__init__` (or template) and baked
  into a static input file (`uvel_shear.dat`, `prof.inp`/`lscale.inp`,
  `u_profile`/`v_profile`). Every ensemble member that clones the template
  inherits the *same* shear.
- **`sgs_constant`** lives only in the static solver template (LBM `infile.in`
  `ivreman smagor` line, uDALES `namoptions` `&NAMSUBGRID cs`, PALM closure
  default).

To make either estimable, the value must be **written per member, at run time**,
from the member's `params` Dataset — i.e. inside (or just before) each backend's
`_apply_inflow_settings(params)`, which is already the per-member hook the
ensemble calls with that member's parameters. The new parameters ride in the
*same* `params` Dataset that already carries `inflow_angle` /
`velocity_magnitude`, so no new plumbing is needed between the sampler, the
smoother and the forward model — only the **consumption** end changes.

Principle: **move the `α` write out of `__init__` into the per-call path, and add
an `sgs_constant` write to the same path.** Keep a sensible default when the
parameter is absent (so single-model forward runs and existing tests are
unaffected).

---

## 2. Per-model mapping

### 2.1 `vertical_inflow_exponent` (α) — clean in all three backends

All three already import `build_profile_shape(profile_config, heights, zsize)`
(`(z/z_ref)^α`); the plan only redirects where `α` comes from and *when* the
shape is written.

| Backend | Current write site | Change |
|---|---|---|
| **pylbm** | `__init__` → `write_uvel_shear_file(...)` ([forward_model.py:124-138](../libs/pylbm/src/pylbm/forward_model.py#L124-L138)) | Move the `write_uvel_shear_file` call into `_apply_inflow_settings`, overriding `profile_config["alpha"]` with `get_param_value(params, "vertical_inflow_exponent")` when present; fall back to the construction-time `profile_config` otherwise. |
| **pyudales** | nudging path rewrites `prof.inp`/`lscale.inp` every call via `apply_time_varying_inflow(..., **self._nudging_config)` ([forward_model.py:531-540](../libs/pyudales/src/pyudales/forward_model.py#L531-L540)) | In `_apply_inflow_settings`, build a shallow copy of `self._nudging_config` with `profile_config["alpha"]` overridden from `params` before the call. The static (`apply_inflow_settings`) branch takes an optional `profile_shape` already — build it from the param there too. |
| **pypalm** | `_apply_inflow_settings` already calls `build_profile_shape(self._nudging_config.get("profile_config"), ...)` per call ([forward_model.py:361-372](../libs/pypalm/src/pypalm/forward_model.py#L361-L372)) | Override `profile_config["alpha"]` from `params` immediately before that call. Smallest change of the three. |

Helper to add (one per backend `params_utils`, or shared): given `params` and the
default `profile_config`, return a `profile_config` dict with `alpha` replaced by
the param when present. Keep `type`/`z_ref` from the default.

### 2.2 `sgs_constant` — clean in LBM & uDALES, proxy in PALM

| Backend | Solver input | Write |
|---|---|---|
| **pylbm** | `infile.in` line `1 0.15  ! ivreman smagor` — key is `ivreman`, value is the whole `"<ivreman> <smagorinsky>"` token string (confirmed via `Infile._parse_file`) | `Infile.set_value("ivreman", f"1 {sgs:.4f}")` (mirror the existing `uini`/`udir` two-token pattern). Consumed by [m_vreman.F90](../libs/pylbm/LBM/src/m_vreman.F90) as `const = 2.5*smagorinsky**2`. |
| **pyudales** | `namoptions` `&NAMSUBGRID cs` (`lsmagorinsky=.true.`) | `NamoptionsFile.set_value("NAMSUBGRID", "cs", f"{sgs:.4f}")`. |
| **pypalm** | LES Deardorff `c_0` is **hardcoded** `0.1` ([turbulence_closure_mod.f90:542](../libs/pypalm/palm_model_system/packages/palm/model/src/turbulence_closure_mod.f90#L542)); not a namelist key. `km_constant` *is* a namelist key but switches to constant-viscosity (regime change). | See [§2.3](#23-the-palm-sgs-problem). |

PALM `c_0`, LBM `smagorinsky`, and uDALES `cs` are *not* the same number and must
not be tied — they have different definitions per closure. This is intentional:
each model gets its own `sgs_constant` value (separate truth/prior configs
already enforce this; see [§3](#3-config-changes)).

### 2.3 The PALM SGS problem

PALM's LES TKE closure does not expose a Smagorinsky-style multiplier in the
namelist (`c_0` is fixed at `0.1`). Two ways to honour "works for all models":

- **Option A (no Fortran change): map `sgs_constant` → `km_constant`.** Writes a
  constant eddy diffusivity into `initialization_parameters`
  ([parin.f90:225](../libs/pypalm/palm_model_system/packages/palm/model/src/parin.f90#L225)).
  Pros: pure namelist, immediate. Cons: replaces the prognostic SGS-TKE closure
  with a constant-`Km` model — a different turbulence regime, so the PALM
  `sgs_constant` would mean "constant eddy viscosity [m²/s]", not "Smagorinsky
  constant". Acceptable *as a model-error knob* (its only job is to absorb bias),
  but document the unit/meaning divergence loudly.
- **Option B (one-line patch): make `c_0` namelist-readable.** Add `c_0` to the
  `initialization_parameters` namelist in `parin.f90` and skip the hardcoded
  `c_0 = 0.1` assignment when the user set it. Pros: keeps the LES regime,
  semantically parallel to LBM/uDALES. Cons: patches vendored PALM source
  (`libs/pypalm/palm_model_system/...`), which must survive PALM re-fetch/compile.

**Recommendation: ship Option A first** (no source patch, unblocks LBM↔uDALES↔PALM
cross-runs immediately), and treat Option B as a follow-up only if PALM-side
estimates prove poorly identifiable with constant-`Km`. The mapping is isolated to
PALM's `_apply_inflow_settings`, so swapping A→B later is a one-function change.

---

## 3. Config changes

### 3.1 Parameter schema

Extend `resolve_parameter_schema` in
[hydra_helpers.py](../src/pyurbanair/config/hydra_helpers.py) so all three models
advertise the two new names:

```python
base = ("inflow_angle", "velocity_magnitude",
        "vertical_inflow_exponent", "sgs_constant")
if model_name == "pyudales":
    return base + ("pressure_gradient_magnitude",)
return base
```

(`pressure_gradient_magnitude` stays uDALES-only and remains a no-op under
`inflow_outflow` — orthogonal to this change.)

### 3.2 Samplers (static first)

Add both parameters to the **static** sampler configs. Because the truth model
and the assimilation model are physically different solvers, give them their own
ranges — the truth config holds the truth-model's plausible value, the prior
holds the assim-model's prior.

`conf/params/static.yaml` (assimilation prior — tight, physically-motivated):

```yaml
  vertical_inflow_exponent:
    _target_: pyurbanair.static_parameters.Normal
    mean: 0.25
    std: 0.05
    min: 0.05
  sgs_constant:
    _target_: pyurbanair.static_parameters.Normal
    mean: 0.15        # assim model's nominal (e.g. LBM smagorinsky); set per assim model
    std: 0.04
    min: 0.01
```

`conf/params/static_truth.yaml` (fixed truth realization — Constants):

```yaml
  vertical_inflow_exponent:
    _target_: pyurbanair.static_parameters.Constant
    value: 0.25
  sgs_constant:
    _target_: pyurbanair.static_parameters.Constant
    value: 0.20        # truth model's value (e.g. uDALES cs); set per truth model
```

The sampler picks up any key in the mapping with no Python change (per the guide's
"Add a new parameter" recipe). **Use tight priors** — with few sensors, each extra
free parameter risks an underdetermined inverse and ensemble collapse.

---

## 4. File-by-file change list

1. **`src/pyurbanair/config/hydra_helpers.py`** — extend `resolve_parameter_schema`
   (both names for all models).
2. **`libs/pylbm/src/pylbm/forward_model.py`** — move `write_uvel_shear_file` into
   `_apply_inflow_settings`; override `α` from params; add the `ivreman`/`smagor`
   `Infile.set_value` write.
3. **`libs/pylbm/src/pylbm/utils/params_utils.py`** — `α`-override helper; optional
   `apply_sgs_setting(params, dirs)`.
4. **`libs/pyudales/src/pyudales/forward_model.py`** — override
   `_nudging_config["profile_config"]["alpha"]` from params per call; add
   `NamoptionsFile.set_value("NAMSUBGRID", "cs", ...)`.
5. **`libs/pyudales/src/pyudales/utils/params_utils.py`** — thread the param-derived
   `profile_shape` into the static `apply_inflow_settings` branch.
6. **`libs/pypalm/src/pypalm/forward_model.py`** — override `α` before
   `build_profile_shape`; add `sgs_constant` write (Option A: `km_constant`).
7. **`conf/params/static.yaml`**, **`conf/params/static_truth.yaml`** — add both
   parameter blocks (per [§3.2](#32-samplers-static-first)).
8. **`docs/codebase_guide.md`** — note the two new parameters in §4 (Data
   contracts) and §7 (backend gotchas: the SGS write site + PALM caveat).

Default-absent behaviour everywhere: when a key is missing from `params`,
`get_param_value` returns `None` → keep the construction-time/template value. This
keeps `run_forward_model.py`, single-model runs and the existing smoke tests
byte-identical.

---

## 5. Testing

- **Unit (per backend):** with `params` carrying a non-default
  `vertical_inflow_exponent` / `sgs_constant`, assert the rendered input file
  changed (`uvel_shear.dat` slope; `infile.in` `ivreman` line; `namoptions` `cs`;
  PALM `_p3d` `km_constant`/`u_profile`). Two distinct members → two distinct
  files (the per-member guarantee).
- **Schema:** `resolve_parameter_schema("pylbm"|"pyudales"|"pypalm")` contains both
  names; uDALES still has `pressure_gradient_magnitude`.
- **ESMDA smoke (`run_esmda`, static smoother):** compose with the extended
  samplers and `run(cfg)` end-to-end on the tiny smoke shape; assert the posterior
  for both new parameters is finite and the augmented vector width grew by 2.
  Reuse `compose_test_cfg(config_name="run_esmda")` per [tests/conftest.py](../tests/conftest.py).
- **Identifiability check (manual, before trusting joint runs):** estimate one new
  parameter at a time on a real cross-model pair (`pyudales`↔`pylbm`) and confirm
  the held-out **validation sensors** improve, not just the assimilated ones.
- Run serially — `run_esmda` e2e sessions race on the LBM build/`.temp` (see the
  serial-e2e memory).

---

## 6. Mixing static and dynamic parameters in one xarray Dataset

**Requirement.** A typical run keeps `vertical_inflow_exponent` and `sgs_constant`
constant in time while `inflow_angle` / `velocity_magnitude` are dynamic — but all
four must travel in **one** `params` Dataset, be **estimated together** by ESMDA,
and be **processed at run time** (not frozen at construction).

**Key finding — the ESMDA core already does this.**
`TimeVaryingParameterESMDA._flatten_time_varying_params`
([esmda.py:419-447](../libs/data-assimilation/src/data_assimilation/smoothing/esmda.py#L419-L447))
expands **only** variables that have a `time` dim into contiguous `{name}_0…{name}_T`
scalars and **passes time-less variables through unchanged**; `_unflatten_params`
([esmda.py:449-478](../libs/data-assimilation/src/data_assimilation/smoothing/esmda.py#L449-L478))
reverses it symmetrically. So a mixed Dataset is updated jointly with **no smoother
change**: each dynamic knot becomes one augmented row, each static scalar becomes a
single augmented row. `pin_initial_time_point` only affects the time vars (the
`{name}_0` knot), so statics are always estimated. `num_time_points`, which
`run_esmda.py` derives from `prior_params.sizes["time"]`
([run_esmda.py:432-433](../scripts/run_esmda.py#L432-L433)), is **unaffected** by
static vars (they add no `time` dim). `is_time_varying_params` inspects only
`inflow_angle` / `velocity_magnitude`, so adding static extras never mis-routes the
time-varying detection.

The work therefore reduces to the **sampler** (emit the mixed Dataset) and the
**consumption end** (already covered in [§1-2](#1-the-core-design-change)) — no
change to the augmented-state math.

### 6.1 Sampler — emit one mixed Dataset

The shared `_build_dataset` already accepts a `passthrough` dict of time-less vars
([base.py](../src/pyurbanair/dynamic_parameters/base.py)), and `extrapolate`
already forwards non-time posterior vars. Add a **static block** to
`AR2RelaxationModel`:

- New constructor arg `static_parameters: dict[str, Distribution] | None` — the
  same `Normal`/`Uniform`/`Constant` objects the static sampler uses
  (`Distribution.sample(rng_key, ensemble_size) -> (ensemble,)`).
- `sample()`: after building the dynamic `arrays`, draw each static distribution
  **once** (its own RNG split) into an `(ensemble,)` array and pass them as
  `passthrough` to `_build_dataset` → variables with no `time` dim.
- `extrapolate()`: the existing `passthrough = {n: posterior[n] for n in
  posterior.data_vars if "time" not in posterior[n].dims}` already carries the
  statics forward — and because `posterior` holds the **ESMDA-updated** static
  values, the next window's prior inherits the *refined* statics (constant in time,
  refined across windows). This is the desired behavior; do **not** re-randomize
  them per window. Ensure the new static draw happens only in `sample()` (window 0).

### 6.2 Forward models — read statics on both inflow branches

In each backend's `_apply_inflow_settings`, read `sgs_constant` /
`vertical_inflow_exponent` via `get_param_value(params, …)` **outside** the
`is_time_varying_params(...)` if/else, since they apply identically to the static
and time-varying inflow paths. (The α and SGS writes from [§2](#2-per-model-mapping)
are the same; this just notes they must not sit inside the time-varying-only
branch.)

### 6.3 Gotcha — the uDALES param whitelist drops unknown vars

`pyudales/utils/params_utils.py` `extract_inflow_params` / `merge_params` /
`create_params_dataset` hardcode a 3-name whitelist (`inflow_angle`,
`velocity_magnitude`, `pressure_gradient_magnitude`). New variables are **silently
dropped** there before reaching the solver. Extend those lists to include
`vertical_inflow_exponent` / `sgs_constant` (or bypass the whitelist for model
coefficients). pylbm and pypalm read `params` directly, so **only uDALES** needs
this fix — and it is easy to miss because it fails *silently* (the parameter
estimates would move but never affect the run).

### 6.4 Config

`conf/params/dynamic.yaml` (and `dynamic_truth.yaml`) gain a sibling
`static_parameters:` block next to `external_parameters:`:

```yaml
external_parameters:        # time-varying (gets a `time` dim)
  inflow_angle: {_target_: pyurbanair.static_parameters.Normal, mean: 25.0, std: 6.0}
  velocity_magnitude: {_target_: pyurbanair.static_parameters.Normal, mean: 6.0, std: 0.5, min: 0.1}
static_parameters:          # constant-in-time, still ESMDA-estimated
  vertical_inflow_exponent:
    _target_: pyurbanair.static_parameters.Normal
    mean: 0.25
    std: 0.05
    min: 0.05
  sgs_constant:
    _target_: pyurbanair.static_parameters.Normal
    mean: 0.15
    std: 0.04
    min: 0.01
```

`dynamic_truth.yaml` uses `Constant`s for the two static blocks (true values for
the truth model).

### 6.5 Extra file changes for the mixed case (on top of [§4](#4-file-by-file-change-list))

- **`src/pyurbanair/dynamic_parameters/ar2_relaxation.py`** — `static_parameters`
  arg + one-time draw in `sample()`, merged via `passthrough`.
- **`src/pyurbanair/dynamic_parameters/base.py`** — only if a shared helper for the
  static draw is preferred; `_build_dataset` passthrough already suffices.
- **`libs/pyudales/src/pyudales/utils/params_utils.py`** — extend the three
  whitelists ([§6.3](#63-gotcha--the-udales-param-whitelist-drops-unknown-vars)).
- **`conf/params/dynamic.yaml`**, **`conf/params/dynamic_truth.yaml`** — add the
  `static_parameters:` blocks.

---

## 7. Phasing

- **Phase 1 — static smoother** (`esmda/smoother=static`,
  `params@prior_params=static`). Both new parameters are static scalars;
  `ParameterESMDA` handles extra scalar variables with no change. Where cross-model
  misspecification studies naturally start.
- **Phase 2 — mixed dynamic+static** (`esmda/smoother=dynamic`,
  `params@prior_params=dynamic`). Per [§6](#6-mixing-static-and-dynamic-parameters-in-one-xarray-dataset):
  sampler `static_parameters` block + uDALES whitelist fix + config. **No
  smoother/flatten change** — the time-varying ESMDA already passes time-less vars
  through. Small, well-scoped.
- **Phase 3 — PALM Option B** (namelist `c_0`) only if Phase 1/2 show PALM's
  constant-`Km` proxy is poorly identifiable.

---

## 8. Risks / watch-items

- **Over-parameterization.** Few sensors + more free parameters → ensemble
  collapse / overfitting. Mitigate with tight priors ([§3.2](#32-samplers-static-first)),
  one-at-a-time identifiability checks, and the validation-sensor score as the
  arbiter.
- **Cross-model value confusion.** `sgs_constant=0.20` means uDALES `cs`, LBM
  `smagorinsky`, or PALM `km_constant` depending on which solver consumes it. The
  truth/prior split already keeps them separate; the configs must set
  model-appropriate ranges, and the YAML comments must say which solver each value
  targets.
- **PALM regime change (Option A).** `km_constant` disables prognostic SGS-TKE.
  Acceptable as a bias knob; flag the unit/meaning divergence in the config and
  guide.
- **Stability.** Very small `sgs_constant` (low dissipation) can destabilize a run.
  Keep the prior `min` above zero; the uDALES instability watchdog and the
  ensemble failure/resample policy already catch divergent draws.
- **Vendored PALM source (Option B only).** A `parin.f90` patch must survive PALM
  re-fetch/compile — out of scope for Phase 1.
