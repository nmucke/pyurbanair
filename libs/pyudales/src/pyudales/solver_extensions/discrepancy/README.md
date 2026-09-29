# Native Vreman discrepancy, capability `sgs_strain_rotation_v1`

`manifest.json` pins pristine uDALES commit
`b84916ac60cecd1da54dd09df76c15e30dcaabe9`. The build helper verifies resource
and source SHA256 hashes, applies `discrepancy.patch`, copies the independent
Fortran module into `src/`, and checks the final hashes. uDALES' existing CMake
source glob includes the helper; no upstream checkout edits are necessary.

The extension recomputes the dimensionless strain/rotation feature from all
nine native Vreman derivatives on every closure call, including RK stages.
Scaled nonnegative squared norms avoid overflow while preserving the configured
regularization in inverse seconds. The height feature uses native `zf(k)` and
configured dimensional endpoints. A scratch multiplier array exists only when
enabled. The native buoyancy adjustment and scalar diffusivity run first; the
multiplier then changes only interior turbulent momentum viscosity, before
molecular viscosity is added and `closurebc` fills/exchanges boundaries. All
three existing momentum stress-divergence routines are unchanged.

Pristine Vreman's `bb/aa` traps for a zero gradient. The enabled branch gives
zero turbulent viscosity when `aa == 0`; the disabled branch retains the
original expression. Tests reproduce both behaviors using the shipped branch.

Before the initial enabled timestep, all ranks verify finite and identical
`timee`/`dt` clocks; inconsistent checkpoint shards abort collectively before
rank-dependent thermodynamic diagnostics can deadlock. Halo exchange precedes the existing
startup boundary call. Thermodynamic diagnostic fields and closure are then
refreshed without a time advance or momentum tendency. This prevents stale
restart viscosity from determining the first adaptive timestep. Enabled zero
coefficients give the native algebraic closure at a fixed state, but can change
that first timestep compared with a stock executable that uses stale viscosity.
This is deliberately not a claim of bitwise forecast equivalence.

Rank zero writes `sgs_discrepancy.<experiment>.txt` at normal completion.
It contains coefficient and feature settings, multiplier extrema across all
ranks/evaluations (including the initial refresh), and the fraction of evaluated
interior cells with `abs(tanh(g/L)) >= 0.95`. Counts include each RK evaluation.
No per-cell diagnostic or MPI collective is added to the disabled closure.

Fast tests check resource provenance, patch application, preserved stress
routines and application order, float32/float64 kernel behavior with floating
point traps, bounded multipliers, finite input validation and zero-gradient
handling. Full MPI agreement, wall/energy budgets, cold/warm restart replay,
and continuous-versus-segmented forecast equivalence remain integration gates;
these resources alone do not establish physical validity or inference skill.
