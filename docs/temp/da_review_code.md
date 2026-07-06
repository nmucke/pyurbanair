# Data-Assimilation Library Review — Part 1: Code

Scope: all of `libs/data-assimilation/src/data_assimilation/` — `smoothing/` (`base.py`,
`esmda.py`), `localization/` (`base.py`, `correlation.py`, `distance.py`),
`observation_operator.py`, `interpolation.py`, `reduction.py`. Cross-checked against the
consumers (`scripts/run_esmda.py`, `scripts/_esmda_common.py`) and the test suite
(`tests/test_localization.py`, `test_state_reduction.py`, `test_observation_operator.py`,
`test_run_esmda.py`).

---

## 1. Bugs to fix

### 1.1 Dead exception handler hides singular-solve failures — `smoothing/esmda.py:186-189`

```python
try:
    x = jnp.linalg.solve(C_DD_alpha, innovation)
except jnp.linalg.LinAlgError:
    x = jnp.linalg.lstsq(C_DD_alpha, innovation, rcond=None)[0]
```

Two problems, verified against the installed JAX (0.7.x):

* `jnp.linalg.solve` **never raises** on a singular matrix — it silently returns
  NaN/Inf. The `except` branch is unreachable; the lstsq fallback never runs.
* `jnp.linalg.LinAlgError` **does not exist** (`hasattr(jnp.linalg, "LinAlgError")` is
  `False`). If an exception ever did propagate here, evaluating the except clause would
  itself raise `AttributeError`.

Net effect: a singular `C_DD + alpha * C_D` silently NaN-poisons the whole ensemble, and
the NaNs propagate through subsequent forecasts before anyone notices. Fix: delete the
try/except, and add an explicit finiteness check on `x` (raise with a diagnostic), or use
a Cholesky solve and check for NaN. (In practice `alpha * C_D` with positive variances
keeps the system SPD, so the check should essentially never fire — which is exactly why a
silent NaN would otherwise go undetected for a long time.)

### 1.2 `Any` imported from pandas internals — `observation_operator.py:6`

```python
from pandas.core.indexing import Any
```

This should be `from typing import Any`. It currently resolves to the same object only
because `pandas.core.indexing` happens to import `Any` at module top level — a private,
unstable internal. It also makes `data_assimilation` import pandas at module load for no
reason, and pandas is not a declared dependency of the package (it arrives transitively
via xarray). One-line fix; almost certainly an auto-import gone wrong. The annotation it
feeds (`num_obs -> int | Any`) should just be `int`.

### 1.3 `_observation_step` silently returns `None` — `smoothing/base.py:32-60`

When both `state is None` and `results_dir is None` the function falls off the end and
returns `None`, which then fails much later as a confusing
`jnp.asarray(None)` / transpose error in `_one_step`. Add an explicit
`raise ValueError("Either state or results_dir must be provided.")` (the sibling
`_get_states` in `esmda.py` already does this). While there: the docstring's claimed
return shape "(num_observations, num_sensors)" is wrong — the disk path returns
`(N_e, num_obs)` and the in-memory path returns whatever the operator returns
(`(N_e, num_obs)` for ensembles).

### 1.4 `TemporalObservationOperator.num_obs` crashes with `TypeError` in "full" mode — `observation_operator.py:232-244`

For `mode="full"` with `num_time_steps=None` (documented as OK, "detected from the first
observed state"), querying `num_obs` before the first call evaluates
`self.observation_operator.num_obs * None` → `TypeError`. The "intervals" mode handles
the identical situation with a clean `RuntimeError`; "full" should do the same.

### 1.5 Interval count is cached once and never re-validated — `observation_operator.py:286-289`

`_num_intervals` is set on the first call and then assumed forever. If a later window
produces a different number of *populated* bins (a short final window, a missing frame,
a different output cadence), the operator silently returns an observation vector of a
different length than `num_obs` claims — and downstream the `C_D` built from the first
window no longer matches, either failing with an opaque shape error or (worse) matching
by coincidence with wrong semantics. Cheap fix: after computing `num_intervals` on every
call, raise if it differs from the cached value.

Related structural weakness (same code region): bins are keyed by `unique_bins`, so
*empty* bins are dropped and interval `k` of the returned vector is "k-th populated bin",
not "absolute interval k". If the real sensor data are aligned by absolute interval
index, an empty bin would shift everything after it. With a uniform output cadence this
cannot happen, but nothing checks that assumption — see the math report (§2.6).

### 1.6 Name-based block grouping can merge unrelated parameters — `smoothing/esmda.py:14-27`

`_group_ids_by_base_name` reconstructs the time-knot blocks by stripping a trailing
`_<int>` from the flattened names. A *static* parameter whose real name ends in
`_<digits>` (e.g. `pm2_5`, `sensor_3_bias`) would be silently merged into the block of a
same-base time-varying parameter, or merged with another static parameter. The flattener
(`_flatten_time_varying_params`) knows the true name→(param, knot) mapping at the moment
it builds the flat names — return group ids from there instead of re-parsing names with a
regex. That removes the failure mode entirely and deletes the regex.

### 1.7 `_state_group_ids` co-location is wrong on staggered grids — `smoothing/esmda.py:668-683`

Block ids are the within-variable flat cell index, shared across variables. That
co-locates `u`/`v`/`w` only when all state variables live on the **same grid shape**
(true for pylbm's collocated grid). For uDALES/PALM staggered output the per-variable
shapes differ (`xm` vs `xt`, `yv` vs `y`, …): index `i` of `u` is not physically
co-located with index `i` of `v`, and when sizes differ the id ranges only partially
overlap — the "joint block update" then jointly updates physically unrelated cells.
Minimal fix: validate all state vars share one shape and raise otherwise ("block grouping
requires collocated state variables"); the honest fix is to build ids from rounded
physical coordinates (which `_state_row_coords` already knows how to compute).

### 1.8 Datasets opened and never closed — `smoothing/esmda.py:103-107, 591, 624` and `smoothing/base.py:57`

`get_state`, `_get_states`, `_get_window_states`, and `_observation_step` all call
`xarray.open_dataset(...)` without `load()`/`close()`/context manager. With
`ensemble_size × (num_steps+1)` NetCDF files per window, long rollouts accumulate open
handles; the netCDF4 backend keeps them until GC, and this is a classic source of
`Too many open files` / HDF5 locking errors on clusters. Use
`with xarray.open_dataset(f) as ds: ds.load()` (the arrays are consumed immediately in
every call site, so eager loading costs nothing).

### 1.9 `rng_key` default: shared, import-time, and falsely `Optional` — `smoothing/esmda.py:50`

```python
rng_key: Optional[jax.random.PRNGKey] = jax.random.PRNGKey(42),
```

* The default is evaluated at import time — it initializes the JAX backend as a side
  effect of importing the module, which can pin the platform before the caller configures
  it (`JAX_PLATFORMS`, x64, etc.).
* Every smoother constructed without an explicit key gets the *same* key, so e.g. two
  experiments in one process draw identical observation-perturbation streams.
* The annotation says `Optional`, but passing `None` crashes at the first
  `jax.random.split(None)`.
* `jax.random.PRNGKey` is a function, not a type; the annotation should be `jax.Array`.

Fix: `rng_key: Optional[jax.Array] = None` and `self.rng_key = jax.random.PRNGKey(42) if
rng_key is None else rng_key`.

---

## 2. Improvements (no new features)

### Structure

* **Imports after code in `esmda.py:1-37`.** The two module-level helpers are defined
  *between* the import blocks (line 33 resumes imports after `def` statements). Move the
  helpers below the imports. Cosmetic but jarring, and it defeats isort/ruff.
* **Return-type polymorphism of `_analysis`.** Depending on `save_on_disk` /
  `return_state_history`, `__call__` returns `Dataset`, `(Dataset, Dataset)`, or
  `(Dataset, concat Dataset)`. Every caller must branch on mode
  (`run_esmda.py` does). A small result object (`ESMDAResult(params, state,
  params_history, state_history)` with `None` for absent pieces) removes the branching
  and the docstring caveats at near-zero cost.
* **The 5-class MRO diamond.** `StateAndTimeVaryingParameterESMDA(StateAndParameterESMDA,
  TimeVaryingParameterESMDA)` works, but the constructor contract is only discoverable
  through a 10-line docstring explaining the MRO chain — a durable maintenance tax. The
  two orthogonal concerns are already cleanly separable: (a) how parameters
  flatten/unflatten (static vs time-varying with pinning), and (b) whether the window IC
  joins the augmented vector. Two small strategy objects (`ParamFlattener`,
  optional `StateAugmentation`) composed into *one* smoother class would replace four
  subclasses and the MRO note. Recommended next time this file is touched, not as a
  drive-by.
* **Empty `__init__.py`s / no public API.** Consumers import from deep module paths and
  the package re-exports nothing. Populate `data_assimilation/__init__.py` (and the
  subpackage inits) with the public names (`ParameterESMDA`, …, `CorrelationLocalization`,
  `DistanceLocalization`, `OnlineStateReduction`, `ObservationOperator`,
  `TemporalObservationOperator`) so refactors of file layout don't break callers.
* **`print` → `logging`.** `esmda.py:331` ("ESMDA step i completed") and
  `reduction.py:111-115` (rank report) print unconditionally; in ensemble sweeps this is
  noise the caller can't silence. Use a module logger.
* **Validate `C_D` at construction.** `C_D_sqrt = jnp.sqrt(C_D)` (element-wise) and
  `jnp.diag(C_D)` in the localized update are both only correct for a diagonal matrix,
  but nothing checks this (see math report §1.2). Cheapest robust API: accept a 1-D
  variance vector and build what's needed internally; otherwise assert
  off-diagonals are zero.
* **`_flatten_time_varying_params` trusts `self.num_time_points`** (`esmda.py:469`)
  instead of reading `params.sizes["time"]`. If config and data disagree, time knots are
  silently dropped (config smaller) or it raises an opaque out-of-bounds error (config
  larger). Read the size from the data and (optionally) validate against the configured
  value once.
* **Duck-typed localization attributes.** `getattr(localization, "block_grouping",
  False)` and `getattr(..., "requires_coordinates", False)` — `BaseLocalization` exists
  precisely so these can be declared class attributes (`requires_coordinates` already
  is). Declare `block_grouping: bool = False` on the base and drop the `getattr`s.
* **Fail fast on invalid strategy/smoother pairings.** `DistanceLocalization` with a
  parameter-only smoother only errors deep inside the first Kalman update (it needs
  `row_coords`). The smoother knows at construction whether it can supply coordinates —
  check `localization.requires_coordinates` in `ParameterESMDA.__init__` and raise there.
* **Stale docstrings**: `_observation_ensemble` "num_obs = NUM_OBS * 3"
  (`observation_operator.py:149`); the "(time, sensor) -> (time * sensor,)" ravel comment
  at `observation_operator.py:133` (time was already selected away at line 103);
  `_observation_step` return shape (see §1.3).

### Computational efficiency

* **The localized update is O(N_aug · N_d³) time and O(N_aug · N_d²) memory —
  `localization/base.py:256-293`.** `jax.vmap(update_row)` materializes a batched
  `C_DD_alpha` of shape `(N_aug, N_d, N_d)` and solves each. For parameter-only vectors
  (N_aug ~ tens) this is fine; for a state-bearing smoother N_aug is the full flattened
  grid — e.g. 128³ × 3 components ≈ 6.3M rows, which at N_d = 50 is a ~630 GB batched
  intermediate. Three stacked remedies, in order of payoff:
  1. **Solve once per unique inflation row, not per row.** With block grouping the
     inflation rows within a block are identical *by construction* (the comment even says
     the solves "collapse to one transition per block" — but the code still performs
     every redundant solve). With distance localization, all rows of all variables at the
     same cell share a row, and rows far from every sensor share the all-excluded row.
     `jnp.unique(inflation, axis=0)` (or the group ids directly) + one solve per unique
     row + a gather back is exact and typically orders of magnitude smaller.
  2. **Chunk the vmap** with `jax.lax.map(..., batch_size=...)` over rows so peak memory
     is `O(chunk · N_d²)` regardless of N_aug.
  3. **Short-circuit fully-excluded rows** (`~active.any()`): they get the identity
     update; no solve needed. After distance truncation this is often most of the grid.
* **`update_row` recomputes row-independent work per row.** `perturbed_obs` depends only
  on `e_inf`; for the common case where most rows share `e_inf ≡ 1` this is the same
  `(N_d, N_e)` matrix rebuilt N_aug times. Falls out for free with the unique-row
  dedup above.
* **Vectorize the ensemble loop in the observation operators.**
  `ObservationOperator._observation_ensemble` (`observation_operator.py:154-157`) loops
  Python-level over members and calls `_observation_single` per member;
  `interpolate_dataarray_at_points` already handles arbitrary leading dims (it's written
  for `(..., z, y, x)`), and the index path's vectorized `isel` also broadcasts over
  `ensemble`. Calling the operator once on the full ensemble Dataset removes an
  N_e-fold Python/xarray overhead. Same for `TemporalObservationOperator.
  _observation_ensemble`.
* **`state_template` allocates a full ensemble-state copy of empties —
  `esmda.py:775-784`.** `_unflatten_state` only reads dims/sizes/coords/dtype from the
  template; pass `states_array` itself and drop the `jnp.empty` allocations (which are
  real device allocations, the size of the entire state ensemble).
* **No `jit` anywhere.** Every `jnp` op in the update dispatches eagerly. The pure-array
  core (`_compute_kalman_update` global path; `localized_update`) is an ideal jit target
  — shapes are stable across the `num_steps` iterations of a window and across windows.
  Passing `rng_key` in and returning the split key out (instead of mutating `self`)
  makes `_compute_kalman_update` a pure function; jitting `localized_update` also lets
  XLA fuse the taper/masking pipeline. Likely a several-fold speedup of the analysis
  step for in-memory runs, more on GPU.
* **`_flatten_window_snapshots` + `_flatten_state` produce host→device copies per
  variable** (`.values` then `jnp.concatenate`). Minor next to the forward model cost;
  only worth touching if the analysis step ever shows up in profiles.

### Testing

Coverage of `localization/`, `reduction.py`, and the observation operators is genuinely
good (equivalence tests against the global update, taper endpoint tests, flatten-order
consistency tests). Gaps worth closing:

* No direct unit test of `_compute_kalman_update`'s global path against a hand-computed
  linear-Gaussian posterior (the standard "ESMDA with G linear reproduces the exact
  Bayesian posterior mean/cov as N_e → ∞" sanity check, or a fixed-seed small-matrix
  regression). It's exercised only through e2e runs.
* No test for `TimeVaryingParameterESMDA._flatten/_unflatten` round-trip with
  `pin_initial_time_point=True` (the pinning path is only covered indirectly).
* Nothing pins the on-disk pruning behavior (`prune_disk_steps`,
  `keep_prior_disk_step`).

---

## 3. Recommended extensions (code-level)

Ordered roughly by value/effort:

1. **Missing-observation masking.** Real sensor networks drop out. Accept NaNs in `obs`
   and mask the corresponding rows out of the update (the localized-update machinery
   already knows how to exclude observations exactly — the same decoupling trick applies
   to the global path). Today a single NaN observation poisons the whole update.
2. **Diagnostics module.** Per-iteration scalars the calling scripts currently improvise
   or skip: ensemble spread (state & params), data mismatch
   `(obs - pred_obs)ᵀ C_D⁻¹ (obs - pred_obs)` per member, effective number of active
   observations per row after localization, reduction rank/energy. Returned as a small
   Dataset alongside the result; makes divergence and overconfidence visible *during* a
   window instead of post-mortem.
3. **Checkpoint/restart of the MDA loop.** A window with `num_steps=4` and an expensive
   forward model that dies at step 3 currently restarts from scratch. The state to
   persist is tiny (params Dataset, IC Dataset, rng_key, step index).
4. **Per-step α schedules** (list-valued `alpha`, validated; see math report §3.1) —
   the plumbing change is trivial, the value is methodological.
5. **Posterior predicted observations in the result** — callers re-run the observation
   operator today to make fit plots; `_one_step` already has `pred_obs` in hand at the
   final step.
6. **Structured logging/callbacks** (`on_step_end(step, diagnostics)`) so run scripts can
   stream progress to their own sinks instead of parsing prints.
