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
time-varying inflow runs — they are model coefficients, not inflow schedules.
See [§6 Phasing](#6-phasing) for the dynamic-sampler wrinkle.

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

## 6. Phasing

- **Phase 1 — static smoother only** (`esmda/smoother=static`,
  `params@prior_params=static`). Both new parameters are static scalars; the
  `ParameterESMDA` flatten/unflatten handles extra scalar variables with no change.
  This is where cross-model misspecification studies naturally start.
- **Phase 2 — dynamic / time-varying inflow.** Wrinkle: the dynamic sampler
  (`AR2RelaxationModel`) gives a `time` dim to **every** `external_parameters`
  entry, and `TimeVaryingParameterESMDA` flattens per `(time, member)`. `α` and
  `sgs_constant` must stay static (one value per member). Two sub-options:
  (a) carry them as a separate static block the dynamic sampler concatenates
  without a `time` dim, and teach the flatten/unflatten to leave non-time vars
  alone; or (b) keep them out of the dynamic vector and only co-estimate them in
  the static/joint variants. Decide in Phase 2; do not block Phase 1 on it.
- **Phase 3 — PALM Option B** (namelist `c_0`) only if Phase 1 shows PALM's
  constant-`Km` proxy is poorly identifiable.

---

## 7. Risks / watch-items

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
