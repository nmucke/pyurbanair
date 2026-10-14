# Why the SGS correction does not help, and where to go next

October 2026 · Follows the SGS twin study (Rounds 1, 1b and 2; report on
branch `exp/sgs-twin-tests`, `docs/research/sgs_twin_tests_2026-10/`).

**Conclusion.** Correcting the uDALES SGS viscosity is a dead end for
cross-model data assimilation. The uDALES–PALM mismatch comes from flow next to
walls and around under-resolved buildings. The SGS term neither causes it nor
reaches it. Future work should first make PALM a trustworthy truth, then give
the error model an honest structural-error term, and estimate only parameters
that also exist in the real world.

**Scope.** The goal is assimilation of real observations, where neither the
true state nor the true parameters are known. PALM only stands in for "a truth
from a different source than the DA model". So nothing below learns or fits the
PALM–uDALES difference itself: methods must work with sensor data alone.

## 1. What the study found

| Setting | SGS correction (3 coefficients) | Single SGS constant |
|---|---|---|
| uDALES truth, no inlet turbulence | Recovered; held-out errors −40 to −90 % | — |
| uDALES truth, turbulent inlet | Not identifiable, no value | — |
| PALM truth (Round 2) | No value; coefficients flip sign or saturate | about −4 % |

In Round 2, every arm was miscalibrated: the Desroziers ratio was about 3.5.
The mismatch at the true inflow is 0.48 m/s in 180 s sensor means, 4–6× PALM's
realisation noise. It is spatial and of mixed sign.

## 2. Why the correction cannot work

**The mean flow does not respond to the knob.** In the Round-2 pilot runs:
- switching the uDALES inlet turbulence on or off moves the 22 sensor means by
  0.08 m/s RMS;
- the Round-1 coefficients move them by about 0.005 m/s;
- raising the Vreman constant by 67 % improves the fit by about 4 %.

The literature agrees. Urban mean flow is set by form drag at sharp edges, and
the choice of SGS model barely changes it (Tseng et al. 2006). Estimating an
SGS-coefficient profile by DA did not move the mean flow; a separate momentum
forcing was needed (Mons et al. 2021).

**"More dissipation" is compensation, not physics.** Both arms pushed uDALES
toward more SGS dissipation. Yet uDALES already uses about 3× PALM's equivalent
eddy viscosity: Vreman c = 0.24 against PALM's Deardorff model at about c = 0.08.
A tuned SGS constant absorbs numerical and resolution error (Meyers et al. 2003).
Here that error is most likely the under-resolved separated shear layers:
- the roofs are 1–7 cells tall, and the 2.8 m cube is one cell;
- PALM carries an advected, prognostic SGS kinetic energy through those shear
  layers, which no local algebraic multiplier can reproduce.

**The error is where the correction does not act.**

| Region | Mean-flow error | Error / PALM noise | Correction acts there? |
|---|---|---|---|
| 14 m, roof level (PALM separates over the 14 m roofs, uDALES does not) | u 0.54 m/s RMS | about 10× | Weakly: only rescales a viscosity already about 3× PALM's |
| 2 m, street level (channelling, up to 1.3 m/s) | u 0.46, v 0.72 m/s RMS | about 3–4× | No: the band is zero below 5 m, and wall stress ignores the eddy viscosity |
| Cells next to walls | about 0.9 m/s RMS | — | No |
| Above 15 m | 0.12–0.19 m/s | small | Not needed |

**Geometry is not the cause.** The rasterised buildings are identical in both
models, column for column. The setups differ elsewhere:
- **Wall model.** uDALES uses a log-law wall function at the distance to the true
  facet. PALM's surface layer sits at a fixed 1 m. Wall stress differs by −30 % to
  +50 %, and by about 5× on the 2.8 m roof.
- **SGS model type.** Algebraic Vreman against prognostic Deardorff.
- **PALM-only outflow damping.** A layer relaxes the last building column's whole
  near wake toward the inflow profile.
- **PALM's inflow disturbances** were injected inside the first building column,
  not upstream of the array.

Neither model is grid-converged at 2 m, so the "truth" carries its own artefacts.

## 3. Is SGS correction a dead end?

**As a DA parameter, yes.** No reformulation fixes this: the mean flow is not
sensitive to the SGS term, and the error is not in it. In canonical flows, SGS
estimation improves turbulence statistics (variances, spectra), not means.

A better closure would be a model change, not a DA knob. Two options:
- a one-equation SGS kinetic-energy model in uDALES;
- wall-adapting treatment near walls.

Revisit SGS only if turbulence statistics become a target, and then as a model
improvement.

## 4. Directions for real-world cross-model DA

Ordered by priority. Each applies with real observations, where no truth field
is available.

1. **Make PALM a trustworthy truth (in progress).** Verify laminar inflow,
   turbulent inflow, periodic runs and the outflow:
   - turbulent inflow upstream of the array, not inside it;
   - outflow treatment outside the region with sensors;
   - fix the inflow face that warm starts zero.

   Re-check the pilot anomaly in which the "smooth" and "turbulent" PALM runs
   are identical to four decimals. Until this is done, part of the 0.48 m/s may
   be PALM artefact rather than uDALES error.
2. **Locate the mismatch with forward runs, without DA.** With the fixed PALM,
   compare uDALES at the true inflow against:
   - uDALES at 1 m;
   - uDALES with the wall-function distance at the half-cell, as PALM does;
   - PALM at 1 m.

   This shows how much of the gap is resolution, how much is wall treatment, and
   how much remains irreducible code difference. It decides which parameters are
   worth estimating.
3. **An honest observation-error model, built from data alone.** The current R
   holds only realisation noise, so every posterior is overconfident and the
   parameters compensate for unmodelled error. Add a structural term, estimated
   without a truth field, in one of two ways:
   - **Innovation statistics:** iterate the Desroziers relation, per sensor
     height, on past windows.
   - **Physics-ensemble spread:** run uDALES variants that differ in resolution
     and wall treatment.

   Give the term a spatial correlation. The Desroziers ratio near 1 and the
   held-out spread/skill are the checks. Only then are parameter posteriors
   interpretable (Brynjarsdóttir & O'Hagan 2014; Evensen 2019).
4. **Estimate parameters that exist in reality and act where the error is.**
   - inflow angle, speed and shear;
   - inflow turbulence intensity;
   - wall and ground roughness, per surface type (ground, walls, roofs).

   Roughness reaches the street level and the roof separation; the SGS band does
   not. In urban DA, inflow and boundary conditions are the controls that sparse
   sensors can identify (Sousa & Gorlé 2019; Defforge et al. 2019, 2021). Keep the
   Vreman constant at most as a nuisance parameter.
5. **Choose observations the model can represent.**
   - Prefer sensors and statistics that a 2 m model resolves: 180 s window means
     (Round 1b), locations away from roof edges and wall-adjacent cells.
   - Alternatively, give poorly represented locations a larger error in R
     instead of dropping them.

   Sensor placement should be judged by sensitivity to the estimated parameters
   relative to the structural error.
6. **Reduce the structural error at the source.** If step 2 shows resolution
   dominates, run the DA model at 1 m in the region of interest, using the
   surrogates to keep the ensemble affordable. This beats compensating with
   parameters.

**Ruled out:**
- learning or fitting the model–truth difference, online or offline;
- additive forcing fields built from the PALM–uDALES difference;
- tuning parameters with no real-world counterpart until they absorb the
  mismatch.

All of these need the truth field or fail to transfer.

## References

- Brynjarsdóttir & O'Hagan 2014, Inverse Problems 30:114007, doi:10.1088/0266-5611/30/11/114007
- Defforge et al. 2019, J. Wind Eng. Ind. Aerodyn., doi:10.1016/j.jweia.2019.03.030; 2021, Boundary-Layer Meteorol., doi:10.1007/s10546-020-00588-9
- Desroziers et al. 2005, Q. J. R. Meteorol. Soc., doi:10.1256/qj.05.108
- Evensen 2019, Comput. Geosci., doi:10.1007/s10596-019-9819-z
- Meyers, Geurts & Baelmans 2003, Phys. Fluids 15:2740, doi:10.1063/1.1597683
- Mons, Du & Zaki 2021, Phys. Rev. Fluids 6:104607, doi:10.1103/PhysRevFluids.6.104607
- Sousa & Gorlé 2019, Build. Environ., doi:10.1016/j.buildenv.2019.02.028
- Tseng, Meneveau & Parlange 2006, Environ. Sci. Technol. 40:2653, doi:10.1021/es051708m
