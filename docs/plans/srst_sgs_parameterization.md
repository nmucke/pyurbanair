# Two-channel closure parameterization (wall model + SGS stress) for PALM & uDALES with ESMDA

> Design record, 2026-07-27 (rev. 2 — wall-model channel promoted to co-equal
> focus after source verification). Origin: porting the *stochastic Reynolds
> stress tensor* (SRST) of Jewell, Farhat & Soize, J. Comput. Phys. 536 (2025)
> 114067 to this repo's incompressible LES backends, with hyperparameters
> estimated by **ESMDA** instead of the paper's loss minimization. The audit
> (three-way: PALM SGS internals, uDALES SGS internals, ESMDA plumbing, plus a
> follow-up wall-model source verification) showed that in these
> building-resolving cases the *wall closure* is the dominant parameterizable
> momentum channel and is namelist/input-file-reachable on both backends,
> while the SGS *tensor* structure is not. The plan therefore parameterizes
> **two channels**: the wall model (primary) and the interior SGS stress
> (secondary, SRST-derived). Departing from the paper here is accepted and
> deliberate; the paper's own bipartite split (SRST for the turbulence
> closure, NPM for "other MFU, e.g. wall functions") maps onto exactly this
> two-channel structure. Line numbers are from the working tree on the date
> above — re-verify before implementing.

---

## 1. Background: the SRST and what survives incompressibility

The SRST decomposes the kinematic SGS stress R = 2k(A + I/3), A = ΦΛΦᵀ, and
randomizes the three spectral components independently: eigenvalues Λ̌
(componentiality, hyperparameters μ_z, δ_z, μ_J, p₀), eigenvectors Ψ̌ (SO(3)
rotation, μ_ζ, μ_η, μ_θ, μ_ι, δ_Ψ), and TKE amplitude ǩ = k·μ_k(1+δ_k ξ).
Each draw is one "admissible turbulence model"; the paper fits the 11
hyperparameters to data by loss minimization over ROM-accelerated Monte
Carlo. Two of their results steer this plan: QoIs were most sensitive to
eigenvalues and least to eigenvectors (their Fig. 6), and turbulence-model
UQ alone could not explain their data misfit (their Figs. 7–9) — other
model-form error (they name wall functions explicitly) had to carry the rest.

Incompressible adaptation (both backends confirmed structurally):

1. **The isotropic (2/3)k I part is dynamically inert** — absorbed into the
   modified pressure; neither code ever forms it. Only the deviatoric stress
   affects velocity, so ǩ and the eigenvalue-magnitude direction are
   degenerate: parameterize the deviatoric amplitude once.
2. The compressible terms vanish; the realizability/barycentric/rotation
   machinery carries over verbatim.
3. k is prognostic in PALM (Deardorff `e`); under uDALES's current Vreman
   closure there is no k anywhere (equilibrium reconstruction would be
   needed).
4. Hyperparameters are per-member constants per assimilation window; the
   ensemble replaces the paper's Monte Carlo draws and ESMDA replaces the
   optimizer (no ROM needed).

## 2. The structural constraint on the SGS channel

Neither backend ever assembles the SGS stress as a tensor. Both compute a
scalar eddy viscosity and fuse τ^dev = −2ν_sgs S into the momentum stencils:

- **PALM**: `km = c_0·ℓ·√e` with `c_0 = 0.1` hardcoded for all LES branches
  (`turbulence_closure_mod.f90:542, 4896`); fluxes assembled in
  `diffusion_u/v/w.f90`. No namelist multiplier on LES `km` exists;
  `km_constant` is a regime switch (freezes km uniform, disables the TKE
  equation *and* the wall model), not a multiplier.
- **uDALES**: `closure` (`modsubgrid.f90:165-415`) produces only `ekm`
  (Smagorinsky / **Vreman, current case default** / one-equation TKE);
  `diffu/v/w` (`modsubgrid.f90:677-1010`) fuse it into the divergence.
  `&NAMSUBGRID` accepts exactly 13 declared keys; undeclared keys abort.

Under a scalar closure the stress eigenvectors are locked to the strain-rate
eigenvectors and the anisotropy eigenvalue ratios to those of S. Externally
reachable degree of freedom: **magnitude only**. Componentiality (Λ̌) and
rotation (Ψ̌) require Fortran on either backend — on PALM achievable inside
the sanctioned `user_module.f90` (§6); on uDALES only via heavy edits to the
fused stencils plus their byte-for-byte duplicates in `modibm.f90`
`diff*_corr` (§7).

Moreover, **at solid surfaces both codes bypass the SGS closure entirely** —
the interior flux is replaced by a wall-model flux. In a dense canopy with
sensors at z = 2 m that path carries the dominant share of the tangential
momentum budget. That observation motivates the wall channel.

## 3. Channel A (primary): the wall model — `roughness_length`

### 3.1 PALM side (verified)

MOST surface-layer fluxes overwrite the interior SGS flux at every
wall-adjacent face (`diffusion_u.f90:180-225`;
`surface_layer_fluxes_mod.f90:1722, 1825`: `usws = −κ·u/ln(z_mo/z0)·u*`,
κ = 0.4). The roughness is the plain `&initialization_parameters` namelist
scalar `roughness_length`, applied **uniformly to all surfaces** (ground,
walls, roofs). Related knobs: `z0h_factor` (scalars only),
`constant_flux_layer` (master switch, default `.TRUE.`), `bc_uv_b`.
Spatially varying z0 would require a static-driver surface field — out of
scope initially.

- Current value, Xie & Castro: `roughness_length = 0.05`
  (`examples/palm/xie_and_castro/_p3d:30`). Barcelona: 0.1.
- pypalm does **not** expose it yet, but `P3DFile.set_value` writes it with
  no new utils code — same shape as the `sgs_constant` precedent.

### 3.2 uDALES side (verified — and the namelist route is a trap)

The IBM momentum wall function `wallfunmom` (`modibm.f90:1309`) uses
**per-facet roughness `facz0(fac)`**, not the global namelist `z0`:

- With `ltempeq = .false.` (our case), `iwallmom` (default 2) is **forced to
  3 = neutral** at `modstartup.f90:792-794`, giving
  `ctm = (fkar/ln(dist/facz0(fac)))²`, `stress = ctm·utan²`
  (`modibm.f90:1403-1405, 1938`). Only `facz0` enters; `facz0h` is unused
  for neutral momentum.
- `facz0` is filled in `initfac.f90:215` from **column 3 of
  `factypes.inp.<expnr>`**, indexed per facet by the type ids in
  `facets.inp.<expnr>`.
- ⚠️ The global `&BC z0` (`modsurfdata.f90:72`, default −1) is consumed
  **only** by the legacy `subroutine bottom` (`modibm.f90:2021`), which is
  gated by `lbottom` — default `.false.` (`modibm.f90:49`), not set in our
  namoptions. **Writing `z0` into namoptions would be a silent no-op.** The
  `roughness_length` parameter must map to the `factypes.inp` z0 column.
- Where `factypes.inp` comes from: the example dir ships only the STL +
  namoptions; pyudales' geometry preprocessing generates it
  (`python_udgeom/preprocessing.py:303 generate_factypes`,
  `write_inputs.py:180-186`). All building facets default to **type 1
  "Concrete", z0 = 0.05 m** (`write_inputs.py:339`,
  `preprocessing.py: id_c = 1, z0_c = 0.05`); floors are type −1 with
  z0 = 0.05; bounding walls (type −101) and dummy (0) have z0 = 0 and are
  skipped by the wall function (`modibm.f90:439`).
- von Kármán constant: `fkar = 0.41` (`modglobal.f90:300`, `&WALLS`
  namelist) vs PALM's 0.4 — a ~2.5% systematic wall-stress difference
  between backends at identical z0. Note it; optionally align via
  `fkar = 0.4` in namoptions.

### 3.3 Cross-backend consistency check (verified)

**The Xie & Castro cases currently agree: effective roughness is 0.05 m on
both backends** (PALM uniform `roughness_length = 0.05`; uDALES
`facz0 = 0.05` on every active facet — floors and default-type buildings).
So a shared `roughness_length` parameter has identical units, semantics, and
current baseline value on both codes. The wall-model *forms* differ (MOST
with iterated u* and z_mo = half first grid spacing vs neutral log-law
transfer coefficient at the reconstructed boundary distance, κ 0.4 vs 0.41)
— that residual discrepancy is precisely the kind of model error the
estimated parameter is allowed to absorb in cross-model experiments.

### 3.4 The parameter

`roughness_length` [m], per ensemble member, estimated as **log z0**
(smooth, monotone, log-natural since stress ∝ (κ/ln(z₁/z0))²; keeps z0 > 0
under the Kalman update with no DA-library changes):

- **PALM write site**: `_apply_roughness_setting(p3d, params)` sibling of
  `_apply_sgs_setting` → `p3d.set_value("initialization_parameters",
  "roughness_length", float(z0))`. No-op when absent.
- **uDALES write site**: rewrite column 3 of the member's
  `factypes.inp.<expnr>` for the active types (floors −1 and type 1; leave
  the z0 = 0 rows untouched), in a helper called from
  `_apply_inflow_settings` — which already runs **after** preprocessing
  (`forward_model.py:364` documents that ordering), and per-member file
  copies already exist. Plain text file, 3 header rows. No Fortran, no
  namoptions involvement.
- **No-op contract**: parameter absent → neither file touched → byte-identical
  runs, matching the repo convention.
- Later extension (uDALES-only, still no Fortran): split into 2–3 parameters
  by facet class (floor vs wall vs roof) via distinct type ids — spatially
  structured roughness estimation that PALM's namelist route cannot match.

### 3.5 Why this channel leads

It is where the sensitivity is (wall fluxes dominate at the 2 m sensors); it
is reachable with zero Fortran on both backends; z0 is an uncertain physical
input — ESMDA's home turf, better-posed than closure constants; and MOST /
log-law assumptions are badly violated in canopies, so it is plausibly the
largest single closure-form error in these flows. Shared caveat: bluff-body
form drag is resolved pressure, untouched by either channel — the Phase 1
sensitivity sweep must confirm the sensors respond at all.

## 4. Channel B (secondary): interior SGS amplitude — `sgs_stress_multiplier`

The SRST collapsed to its one namelist-reachable direction: a dimensionless
multiplier μ on the deviatoric SGS stress (the ǩ·|Λ̌| amplitude; μ = 1 is
the unmodified closure on every backend), estimated as **log μ**:

- **uDALES**: pure namelist. Vreman: `c_vreman = 0.07·μ` (linear in ν_t).
  Smagorinsky: `cs` with μ = (cs/cs₀)². The 1-eq `cf` is *not* a clean
  multiplier (it co-rescales dissipation constants).
- **PALM**: one multiply of `km` (+`kh`) in
  `user_actions('before_prognostic_equations')` in `user_module.f90`, read
  from a new `&user_parameters` namelist block (a real namelist in the same
  `_p3d`; `user_module_enabled` flips on only when the block exists, so
  absence is bit-identical — the no-op contract for free). One in-place
  rebuild of `MAKE_DEPOSITORY_default/` serves all members
  (`direct_palm.py:168-201` symlinks that binary); register the rebuild in
  `install_palm.sh`.
- Closed-loop caveat (PALM): scaled `km` feeds back through the SGS-TKE
  production, so the equilibrium stress scales ≈ μ^{3/2}, not μ. Fine for
  ESMDA (smooth, monotone); interpret posteriors accordingly. The exact
  open-loop form is the Phase 2 tendency correction.
- Large μ shrinks the diffusion-limited timestep on both codes; keep the
  prior tight around 1 (e.g. log μ ~ N(0, 0.2²)) — uDALES already runs near
  its stability margin (`cd2`-only advection, courant 0.7) and its dt
  watchdog will kill diverging members.

## 5. Feasibility matrix (both channels)

| Perturbation | uDALES | PALM |
|---|---|---|
| **Wall roughness** (log z0) | ✅ input-file only (`factypes.inp` col 3; per-facet-class split possible) | ✅ namelist (`roughness_length`, uniform) |
| **Wall flux rotation / non-log-law form** | ❌ Fortran (`wallfunmom`) | ❌ Fortran (`surface_layer_fluxes_mod`) |
| **SGS deviatoric amplitude** (log μ) | ✅ namelist (`c_vreman` / `cs`) | ⚠️ one-line `user_module` hook |
| **SGS eigenvalue componentiality Λ̌** | ❌ heavy Fortran | ⚠️ `user_module` tendency correction (no core edits) |
| **SGS eigenvector rotation Ψ̌** | ❌ heavy Fortran | ⚠️ same route; wall stress stays velocity-aligned regardless |

## 6. ESMDA integration

Per-member scalar parameters are fully supported (the
`sgs_constant`/`vertical_inflow_exponent` precedent —
`docs/archive/esmda_model_error_parameters.md`; recipe in
`docs/codebase_guide.md` §8). ESMDA auto-discovers any `(ensemble,)` scalar;
no DA-library changes. Checklist for the two Phase 1 parameters
(`roughness_length`, `sgs_stress_multiplier`):

1. Prior/truth blocks in `conf/params/static.yaml` + `static_truth.yaml`,
   and under **`static_parameters:`** in `dynamic.yaml`/`dynamic_truth.yaml`
   (constant-in-time; passes through the time-varying flatten unchanged,
   refined across windows). Priors: log z0 ~ N(ln 0.05, σ²) with σ ≈ 0.5;
   log μ ~ N(0, 0.2²).
2. `params_to_estimate` in `conf/run_esmda.yaml:89-92` (default off = no-op).
3. uDALES: add names to `INFLOW_PARAM_NAMES`
   (`libs/pyudales/src/pyudales/utils/params_utils.py:27-33`) — unlisted
   names are silently dropped; write sites per §3.4/§4 called from
   `_apply_inflow_settings` outside the nudging branch.
4. PALM: `_apply_roughness_setting` / `_apply_srst_settings` siblings at
   `libs/pypalm/src/pypalm/forward_model.py:~443` (pypalm has no whitelist).
5. **Transforms live at the write site**: ESMDA estimates the log variables;
   the backend applies `exp(·)` before writing. No clipping in the DA loop
   (none exists — bounds act only at prior sampling; the failure-resampling
   jitter is unbounded too).
6. `_PLOTTED_PARAMS` in `src/pyurbanair/plotting.py:55-60`, else invisible
   in all metrics/figures.
7. Tests mirroring `tests/test_model_error_parameters.py` (whitelist
   round-trip, per-member file rendering — for uDALES assert the factypes z0
   column changed and z0 = 0 rows didn't, no-time-dim, byte-identical no-op
   regression); smoke shape per `tests/conftest.py`; ESMDA e2e serial.
8. Doc sync (`codebase_guide.md` §§4/7/8, `scripts_and_configs.md` §1.4,
   backend docs) in the same PR.

Identifiability: start with these two parameters only. They have distinct
spatial signatures (near-surface wall stress vs elevated shear layers), and
the 2/10/32 m sensor levels give leverage to separate them; validate on the
held-out `validation_{x,y,z}_points`, expand only if held-out misfit
improves. The SRST corner selector μ_J ∈ {1,2,3} has no Gaussian home —
treat corners as fixed scenario axes via `compare_models.py` (zero code) or
a softmax relaxation if ever needed.

## 7. Plan

**Phase 0 — repairs (small PR, do regardless).**
(a) Fix the inert `sgs_constant`: pyudales `_apply_sgs_setting` must write
`c_vreman` under the working-tree Vreman switch (or revert the case to
Smagorinsky); retune its prior (0.22/0.03 targeted `cs = 0.24`; Vreman
default is 0.07). (b) Decide on κ alignment (`fkar = 0.4` in namoptions vs
accept the 2.5% offset) and record the decision.

**Phase 1 — two-knob estimation: `roughness_length` (lead) +
`sgs_stress_multiplier`.**
Wire both per §6. Before assimilating, run a `compare_models.py` sensitivity
sweep on both axes (scenario configs with `Constant` values in
`static_parameters:` blocks) to confirm the z = 2 m sensors respond;
expected outcome is roughness ≫ multiplier at 2 m, more balanced aloft.
Then: (a) same-solver twin experiments (truth = perturbed value, prior
centered on baseline) for each parameter separately; (b) joint estimation;
(c) the headline cross-model experiment — uDALES truth / PALM model (or
vice versa) — where the pair absorbs wall-model-form + closure discrepancy
between solvers. Fortran cost of this phase: one line (PALM user_module).

**Phase 2 — structured extensions, still cheap.**
(a) uDALES facet-class roughness: split `roughness_length` into
floor/wall/roof parameters via facet type ids — spatial structure in the
wall channel with zero Fortran; PALM keeps the uniform value (documented
asymmetry). (b) Full SRST on PALM via the `user_module`
tendency-correction (deviatoric-only, reduced continuous set: amplitude —
shared with Phase 1 — one barycentric interior coordinate via logits, one
orientation spread δ_Ψ). Validate no-op bit identity, stability, sweeps,
then ESMDA one hyperparameter at a time. Wall-model flux left untouched
(document this).

**Phase 3 (deferred, likely never) — uDALES tensor SRST.**
Only if Phase 2 shows the tensorial components carry information the two
scalars cannot: requires materializing τ_ij, rewriting three fused stencils,
lockstep `modibm.f90 diff*_corr` edits, and a `wallfunmom` decision.

## 8. Incidental findings to act on (independent of this plan)

- 🐛 `sgs_constant` inert under the working-tree Vreman switch (Phase 0a).
- ⚠️ uDALES `&BC z0` is dead in IBM cases (`lbottom = .false.`): any future
  "roughness via namoptions" attempt is a silent no-op — roughness lives in
  `factypes.inp`. Worth a comment in `docs/pyudales.md`.
- `fkar = 0.41` (uDALES) vs κ = 0.4 (PALM) systematic wall-stress offset.
- `pressure_gradient_magnitude` is in the default `params_to_estimate` but
  pypalm never reads it (uDALES-only).
- pypalm `self.params` accumulates across `run_single` calls
  (`forward_model.py:385`) — a knob set in window 0 persists into window 1
  even if absent from that window's Dataset.
- `HarmonicParameterModel`'s `profiles:` block is missing from
  `_PARAM_CONFIG_BLOCKS` (`hydra_helpers.py:125`) — `params_to_estimate`
  cannot filter harmonic profiles.
- Stale doc links: `codebase_guide.md` L42/676/793 and `pypalm.md` point to
  `docs/temp/esmda_model_error_parameters.md`; the file moved to
  `docs/archive/`.
- barcelona `_p3d` does not set `neutral` (runs buoyant) while
  xie_and_castro sets `neutral = .T.`.
- uDALES `lmason`/`nmason` namelist knobs are dead code in v2.2.0.
