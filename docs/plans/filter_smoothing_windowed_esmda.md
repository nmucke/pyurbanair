# Plan: `filter_smoothing` — windowed parameter-trajectory ESMDA with inner EnKF state filtering

Implementation plan for the hybrid algorithm in
`docs/filter_smoothing/parameter_trajectory_esmda_state_filter.pdf`: the
high-dimensional state is estimated by a **sequential EnKF** (never smoothed),
while the low-dimensional **parameter trajectory** `Θ = [θ_0 … θ_{L-1}]` over a
window of `L` cycles is updated jointly by an **outer ESMDA loop**. Each ESMDA
iteration runs one inner EnKF pass through the window (storing forecast
observations `d_k = H(x_f_k)` *before* each analysis), stacks them into
`D ∈ R^{L·N_d × N_e}`, and applies one tempered Kalman update to the flattened
trajectory. A final EnKF pass makes the state consistent with the final `Θ`.

New module: `libs/data-assimilation/src/data_assimilation/filter_smoothing/`.

---

## 1. Paper → code mapping

| Paper concept | Implementation |
|---|---|
| Inner EnKF pass (§4.1, Eq. 6–8) | Existing `BaseFilter` cycle loop in `mode="state"` (filtering/base.py), via a thin subclass that feeds `θ_{k}` to cycle `k`'s forecast |
| Forecast observations `d_k` stored pre-analysis (Eq. 7) | `BaseFilter` already computes raw pre-inflation `pred_obs` per cycle; add an opt-in recording hook (mirrors the smoother's `collect_obs_diagnostics` pattern) |
| Stacking `D`, `Y`, `R_W` (§4.2, Eq. 9–10) | Plain concatenation of the recorded per-cycle `(N_d, N_e)` arrays / the `(L, N_d)` observation batches / `jnp.tile(C_D_diag, L)` |
| ESMDA trajectory update (§4.3, Eq. 12) | Existing `stochastic_enkf_update(..., alpha=a)` from `filtering/analysis.py` — the exact perturbed-observation tempered update, including localization plumbing |
| `Θ` flatten/unflatten (Eq. 3) | Existing `ParamAugmentation(num_time_points=L)` — one flattening order shared with `TimeVaryingParameterESMDA` |
| Alpha schedule (Eq. 5) | Same convention as `_BaseESMDA`: `alpha = num_steps` default (equal weights), scalar override allowed |
| Reset state ensemble per iteration (§4.4) | The outer loop calls the inner filter's `run()` with the *same* initial state each iteration (state trajectories are discarded) |
| Final consistency pass (§5) | One more inner `run()` with the final `Θ`; its `FilterResult` is the returned filtered state + diagnostics |
| Temporal localization (§7, Eq. 17) | New `TemporalLocalization(BaseLocalization)` reusing `taper_inflation` + `localized_update` with time encoded as a coordinate |
| Random-walk / AR trajectory prior (Eq. 4) | Existing `params@prior_params=dynamic` samplers (`AR2RelaxationModel`, `HarmonicParameterModel`) — knot spacing `seconds_per_knot = time.simulation_time` |
| Moving window (§8) | Phase 2 (see §8 below); existing `ParameterTimeSeries.extrapolate` is the append-at-leading-edge mechanism |

Not needed: process noise `η_k` (the CFD backends are deterministic; the paper
allows `Q_k = 0`), and any change to the ESMDA smoother package.

---

## 2. What is reused unchanged

- `stochastic_enkf_update` — both the inner state analysis (via the injected
  `AnalysisScheme`, α=1) and the outer trajectory update (α=`num_steps`).
  ETKF/LETKF inner analyses come for free through the existing
  `filtering/analysis` schemes.
- `BaseFilter` / `EnsembleKalmanFilter` — the entire inner pass: forecast
  segmentation, augmentation, inflation, localization plumbing, failure
  substitution, on-disk `cycle_{k}/` management, per-cycle diagnostics.
- `ParamAugmentation` — trajectory flatten/unflatten (`{name}_{t}` scalars) and
  `group_ids` (block grouping for correlation localization of the outer update).
- `taper_inflation`, `localized_update`, `resolve_row_inflation` — the outer
  temporal localization is one more `BaseLocalization` strategy.
- Prior samplers under `conf/params/` (`dynamic`, `dynamic_sine`,
  `dynamic_truth`, …) and the truth/observation-batch construction logic of
  `scripts/filtering/run_filtering.py`.
- `InflationScheme` (RTPS etc.) for the inner state filter.

## 3. Small, no-op-preserving extensions to `filtering/base.py`

Two hooks, both byte-identical for existing configurations:

1. **Per-cycle parameter selection.** In `BaseFilter.run`, replace the direct
   forecast call with

   ```python
   forecast = self._forecast_step(
       state=analysis_state, params=self._params_for_cycle(cycle, params)
   )
   ```

   with the default `def _params_for_cycle(self, cycle, params): return params`.
   The filter_smoothing inner filter overrides it to slice knot `cycle` from the
   trajectory (`isel(time=cycle)`, static vars passed through), so each segment
   runs with its `θ_k` as plain scalar params — no backend changes.
   Note `mode="state"` already skips `_check_static_params`, so a time-varying
   params Dataset can legally ride through the state-only filter.

2. **Forecast-observation recording.** Attribute-plumbed opt-in, mirroring the
   smoother's `collect_obs_diagnostics`: when `self.collect_pred_obs` is True,
   `run()` rebinds `self.pred_obs_history = []` on entry and appends the **raw,
   pre-inflation** `(N_d, N_e)` `pred_obs` each cycle *before*
   `_analysis_cycle`. This is exactly the paper's "stored before assimilating
   `y_k`" requirement (§9, first bullet). Off by default: no recording, no
   behavior change.

Nothing else in the filtering package changes.

## 4. Module layout

```
libs/data-assimilation/src/data_assimilation/filter_smoothing/
  __init__.py                # re-export FilterSmoothingESMDA, TemporalLocalization, result types
  base.py                    # FilterSmoothingESMDA + result/diagnostics dataclasses
                             #   + _TrajectoryStateFilter (the thin inner-filter subclass)
  temporal_localization.py   # TemporalLocalization(BaseLocalization)
```

### 4.1 `FilterSmoothingESMDA` (base.py)

Composition, not inheritance from `_BaseESMDA` (the smoother's `_analysis` loop
does whole-window re-forecasts, which is precisely what this method replaces).

```python
class FilterSmoothingESMDA:
    def __init__(
        self,
        observation_operator,            # per-segment operator (same as the filter's)
        forward_model,                   # BaseEnsembleForwardModel
        C_D,                             # per-cycle 1-D variance vector (N_d,)
        num_steps: int = 4,              # ESMDA iterations N_a
        alpha: float | None = None,      # default num_steps (equal weights)
        inner_analysis: AnalysisScheme | None = None,   # default StochasticEnKFAnalysis
        inner_localization: BaseLocalization | None = None,
        inner_inflation: InflationScheme | None = None,
        temporal_localization: TemporalLocalization | None = None,
        common_inner_noise: bool = True, # reuse one inner rng key across iterations (§9 CRN)
        rng_key: jax.Array | None = None,
    ): ...

    def run(self, state, params, observations, *, return_history=False)
        -> FilterSmoothingResult
```

`params` is the trajectory prior ensemble: a Dataset with `time` (knots) and
`ensemble` dims (plus optional static, no-`time` variables — see §9).
`observations` has shape `(L, N_d)`, one batch per cycle, exactly as the
existing filter consumes them. `L` (`num_cycles`) is read from
`observations.shape[0]` and validated against the params' knot layout.

Constructor builds the inner filter once:

```python
self._inner = _TrajectoryStateFilter(
    observation_operator=..., forward_model=..., C_D=...,
    analysis=inner_analysis or StochasticEnKFAnalysis(),
    mode="state", localization=inner_localization, inflation=inner_inflation,
)
self._inner.collect_pred_obs = True
self._param_augmentation = ParamAugmentation()   # num_time_points set at run()
```

`run()` (Algorithm 1):

```python
L, N_d = observations.shape
Y = observations.reshape(L * N_d)                  # cycle-major stacking
C_D_stacked = jnp.tile(self.C_D_diag, L)
aug = ParamAugmentation(num_time_points=n_knots)   # n_knots from params.sizes["time"]

for a in range(self.num_steps):                    # outer ESMDA loop
    self._inner.rng_key = inner_key(a)             # common_inner_noise -> same key each a
    inner_result = self._inner.run(
        state=initial_state,                       # SAME x_0 every iteration (reset)
        params=params, observations=observations,  # inner filter slices knot per cycle
    )
    D = jnp.concatenate(self._inner.pred_obs_history, axis=0)   # (L*N_d, N_e)

    flat = aug.flatten(params)                     # {name}_{t} scalar Dataset
    theta = ParamAugmentation.to_array(flat)       # (n_knots*n_theta, N_e)
    self.rng_key, subkey = jax.random.split(self.rng_key)
    theta = stochastic_enkf_update(
        theta, D, Y, C_D_stacked, subkey, alpha=self.alpha,
        localization=self.temporal_localization,
        row_coords=knot_time_coords, obs_coords=obs_time_coords,
        group_ids=..., localize_mask=...,          # only when localized
    )
    params = aug.unflatten(ParamAugmentation.from_array(theta, flat), params)
    # per-iteration diagnostics: windowed obs RMSE of mean(D) vs Y,
    # trajectory spread, innovation chi2 on the stacked system

# final consistency pass with Θ^{N_a} (§5)
final = self._inner.run(state=initial_state, params=params, ...)
return FilterSmoothingResult(
    params=params,                    # smoothed trajectory ensemble, Eq. (15)
    state=final.state,                # filtered end-of-window ensemble, Eq. (16)
    iteration_diagnostics=[...],      # one record per ESMDA iteration
    final_pass=final,                 # full FilterResult of the last pass
    params_history=...,               # per-iteration trajectories (return_history)
)
```

Key semantics pinned by the paper and enforced here:

- **Reset between iterations** — every inner `run()` receives the identical
  `state` argument; the analyzed trajectories of previous passes never leak in
  (§9, second bullet). A `state=None` cold start is legal (window 0): each
  iteration then cold-starts identically.
- **Forecast, not analysis, observations** — `D` comes from the recording hook,
  which fires before the analysis and before prior inflation.
- **Final pass** — the state returned is always from the extra pass run with the
  *final* trajectory (§9, third bullet).
- **Common random numbers** (`common_inner_noise=True` default) — the inner
  stochastic analysis + any inflation noise reuse one key per window across
  iterations, so the map `Θ → D` is (pseudo-)deterministic and the ESMDA
  cross-covariances are not diluted by fresh Monte Carlo noise (§9, last
  bullet). Set False for independent draws. The *outer* perturbed-observation
  draws always use fresh subkeys.

### 4.2 `_TrajectoryStateFilter` (base.py, private)

```python
class _TrajectoryStateFilter(EnsembleKalmanFilter):
    """State-only EnKF whose cycle-k forecast uses knot k of a trajectory."""
    def _params_for_cycle(self, cycle, params):
        if params is None:
            return None
        sliced = params.isel(time=min(cycle, params.sizes["time"] - 1), drop=True)
        return sliced      # scalar (ensemble,) vars; static vars pass through
```

Piecewise-constant `θ_k` over segment `k` matches the paper's discrete dynamics
(Eq. 6). (A two-knot slice for within-segment interpolation can be added later
without touching the outer loop.)

### 4.3 `TemporalLocalization` (temporal_localization.py)

`BaseLocalization` subclass; Eq. 17 with the shared Vossepoel taper:

- `requires_coordinates = True`, `localizes_parameters = True`,
  `block_grouping` supported (knots of one parameter share a block via
  `ParamAugmentation.group_ids` — same convention as `TimeVaryingParameterESMDA`).
- `inflation_factors(aug_dev, pred_obs_dev, row_coords, obs_coords)`:
  `distance = |t_row − t_obs|` from the first coordinate component (rows carry
  `(t_j, 0, 0)`, observations `(t_k, 0, 0)`), then
  `taper_inflation(distance, truncation=temporal_radius, tapering_beta, max_inflation)`.
- Knot `j` sits at `t_j = j · Δt` (segment start); observation batch `k` at its
  segment end `t_{k+1} = (k+1) · Δt` (the coordinates are built by
  `FilterSmoothingESMDA`, so the strategy stays geometry-agnostic).
- Deliberately symmetric in time (future obs may update earlier knots): the
  taper suppresses spurious long-range sampling correlations, not causality (§7).

The whole localized solve then reuses `localized_update` unchanged.

## 5. Script and configs

Mirror the filtering entry point (same three-stage shape; metrics/figures can
initially reuse the filtering pipeline helpers):

- `scripts/filter_smoothing/run_filter_smoothing.py` — `def run(cfg)` + thin
  `@hydra.main` wrapper. Borrows from `run_filtering.py` verbatim: truth
  inline-or-disk, per-cycle observation batches `(L, N_d)`, obs noise, output
  artifacts (`posterior_params.nc` = trajectory ensemble, `posterior_state.nc`,
  `prior_params.nc`, `true_params.nc`, `iteration_diagnostics.yaml`,
  `cycle_diagnostics.yaml` from the final pass, `truth_access.yaml`,
  `run_info.yaml`). Key differences:
  - the **prior must be a dynamic (time-varying) sampler** — the inverse of
    run_filtering's guard: sampled once over the full window with
    `time.seconds_per_knot = time.simulation_time` so knots align with cycles
    (validated loudly: knot spacing must equal the cycle length, knot count
    must cover `num_cycles` — the sampler emits `L+1` knots for an `L·Δt`
    horizon; the trailing knot rides along in `Θ` and is updated only through
    prior temporal correlations, becoming the leading edge of a future moving
    window).
  - instantiates `FilterSmoothingESMDA` from a new config subtree.
- `conf/run_filter_smoothing.yaml` — self-contained entry point modeled on
  `run_filtering.yaml`. Groups:
  - `filter_smoothing/inner_analysis: stochastic|etkf|etkf_tsvd` (reuse the
    existing `filtering/analysis` option files' `_target_`s; LETKF variants need
    an inner localization, same policy validation as today),
  - `filter_smoothing/inner_localization: none|correlation|distance` (reuse
    strategies),
  - `filter_smoothing/inner_inflation: none|multiplicative|rtps|rtpp`,
  - `filter_smoothing/temporal_localization: none|taper` (new group;
    `taper.yaml` exposes `temporal_radius`, `tapering_beta`, `max_inflation`,
    `block_grouping`),
  - `params@truth_params: dynamic_sine`, `params@prior_params: dynamic` (AR(2)),
  - scalars: `filter_smoothing.num_cycles`, `num_steps`, `alpha: null`,
    `obs_error_std`, `seed`, `common_inner_noise`.
- Export the new classes from `data_assimilation.filter_smoothing.__init__` (and
  the package `__init__` if that is the convention used by the other groups'
  `_target_` strings — match `data_assimilation.filtering.EnsembleKalmanFilter`).

## 6. Tests (`tests/test_filter_smoothing.py`)

Follow the smoke-shape conventions (`compose_test_cfg`, tiny domain, 2 members,
short window; local torch/pylbm caveats per auto-memory — CI is the arbiter):

Unit tests with a cheap stub ensemble forward model (linear/scalar dynamics, in
memory — no CFD):

1. **Pre-analysis forecast observations**: recorded `D` equals `H` of the
   forecast, not of the analyzed state (inject an analysis that visibly shifts
   the state).
2. **Reset semantics**: with `num_steps=2`, the second iteration's first-cycle
   forecast is a function of the same `x_0` (stub records the states it was
   called with).
3. **Final consistency pass**: the returned state comes from a pass run with
   the returned `params` (stub records the params sequence; last inner pass sees
   the post-update trajectory).
4. **Temporal credit assignment**: a linear-Gaussian toy where only `y_L` is
   informative about `θ_0` through the dynamics — the smoothed `θ_0` mean moves
   toward the value implied by `y_L` (the defining feature, §4.3), and with a
   tight `TemporalLocalization` radius it does not.
5. **`TemporalLocalization.inflation_factors`**: exact taper values at distance
   0, inside/outside the radius; block grouping over one parameter's knots.
6. **Alpha schedule**: default `alpha == num_steps`; `common_inner_noise=True`
   gives bit-identical `D` for identical `Θ` across iterations.
7. **`_params_for_cycle`** slicing (knot per cycle, static vars carried, final
   knot clamp) and the `BaseFilter` hook's no-op default (existing filter tests
   already pin unchanged behavior; add one explicit identity check).

E2E smoke (mirroring `tests/test_run_filtering.py`): compose
`run_filter_smoothing` with smoke overrides, `num_cycles=2`, `num_steps=2`,
2 members; assert the artifacts exist and the trajectory posterior has the
expected dims. Serial with the other e2e sessions (auto-memory: they race on
`.temp`).

## 7. Docs (same PR)

- `docs/data_assimilation.md`: new §"Filter smoothing — windowed
  parameter-trajectory ESMDA" (source-tree entry, cycle/iteration semantics,
  temporal localization, config groups) + add the module to the §1 source tree.
- `docs/scripts_and_configs.md`: the new entry point and its groups.
- `docs/codebase_guide.md`: mention the third assimilation entry point.
- `CLAUDE.md` command list: one line for the new script (it lists the other two).

## 8. Phasing

**Phase 1 (this PR series, single window):** the two `BaseFilter` hooks + unit
tests pinning their no-op defaults; `filter_smoothing/base.py` +
`temporal_localization.py` + unit tests; script + configs + e2e smoke; docs.
In-memory ensembles only in the first PR is acceptable — on-disk mode should
already work (each iteration's `cycle_{k}/` dirs are cleared and rewritten by
the existing `_set_cycle_results_dir`, and only the final pass's files matter),
but verify with a test before advertising it; if pruning interacts badly,
raise loudly on `save_on_disk` and defer.

**Phase 2 (moving window, §8 of the paper):** an outer window loop in the run
script — after a window converges, finalize `θ` of the oldest knot, shift, and
append a leading-edge knot via the prior model's `extrapolate` (the
`ParameterTimeSeries` interface already exists for exactly this). Member pairing
between the carried state ensemble and trajectory ensemble is preserved by
construction (both are indexed by `ensemble`).

**Phase 2 candidates, only if experiments demand:** ensemble-space/batched
formulation for long windows (§9 observation-space cost — `L·N_d` here is small:
~12 sensors), parameter-basis reduction over knots, within-segment parameter
interpolation, deterministic (ETKF) outer update.

## 8a. Phase 2 contract (moving window, as agreed before implementation)

Supersedes the Phase 2 sketch above where they differ. Paper §8: sliding
window of length `L`, `[t_{n-L} … t_n] → [t_{n-L+1} … t_{n+1}]`; the oldest
`θ` is **finalized** on leaving the window; the overlapping parameter ensemble
*is* the next window's prior; the new leading-edge value comes from the
**parameter evolution model**.

**Deviation from the sketch, with reason:** the leading edge is appended with
the library's own `ParameterEvolution` (`filtering/parameter_evolution.py` —
`IdentityEvolution` / `RandomWalkEvolution`, applied per member to the last
carried knot), *not* `ParameterTimeSeries.extrapolate`. `extrapolate` rebuilds
a whole-window prior from a posterior (run_esmda's non-overlapping-window
regime) and for e.g. AR(2) synthesizes a fresh trajectory, which would discard
the carried members' posterior information; `evolve` is exactly the paper's
per-member append and keeps `data_assimilation` free of `pyurbanair` imports.

### Orchestrator: `filter_smoothing/moving_window.py`

A pure function (no class state), plus result dataclasses:

```python
run_moving_window(
    smoother,                      # a FilterSmoothingESMDA
    *, state, params, observations,  # window-0 x_0, window-0 trajectory prior,
                                     # observations over the FULL horizon (T, N_d)
    num_windows, window_shift=1,     # s in [1, L]; s = L → non-overlapping
    evolution=None,                  # ParameterEvolution; default IdentityEvolution
    rng_key=None,                    # evolution draws; default smoother-derived
    return_history=False,
) -> MovingWindowResult
```

- **Horizon**: `T = observations.shape[0]` must equal
  `L + (num_windows − 1)·s`, with `L = smoother`'s cycles per window
  (validated loudly). Window `w` (0-based) consumes observation rows
  `[w·s, w·s + L)`.
- **State carry**: window `w+1`'s initial ensemble is window `w`'s
  final-consistency-pass analysis state after cycle `s`:
  `final_pass.state_history.isel(cycle=s−1)` (drop the `cycle` dim). The
  orchestrator therefore runs each window with history collection on
  internally, discarding heavy histories after extraction when the caller
  didn't ask for them. `s = L` is equivalent to carrying `result.state`.
- **Parameter carry**: window `w`'s posterior knots `[s:]` (positional) become
  window `w+1`'s prior knots `[:−s]`; `s` new leading-edge knots are appended
  by chaining `evolution.evolve` per knot with fresh subkeys, each new knot's
  `time` coordinate continuing the incoming grid (`t_last + i·Δt`). Static
  (no-`time`) variables carry through unchanged. Member pairing is positional
  (`ensemble` index) for both state and params — never reordered.
- **Finalization**: when window `w` advances, its posterior knots `[:s]` are
  finalized (paper: "fixed after leaving the window") and recorded; the last
  window contributes its entire posterior. `MovingWindowResult.params` is the
  concatenation along `time` — the full-horizon smoothed trajectory ensemble.
  `MovingWindowResult.state` is the last window's filtered end state;
  `window_results` keeps each window's `FilterSmoothingResult` (histories
  stripped unless `return_history`), and per-window `WindowDiagnostics`
  records the window index, cycle span, and that window's iteration
  diagnostics.
- **Degenerate case**: `num_windows = 1` must return exactly the Phase 1
  `run()` result (bit-identical `params`/`state`), wrapped — pinned by a test.
- `FilterSmoothingESMDA` and `filtering/base.py` are **not modified**. The
  orchestrator uses only public API (`run`, `num_cycles` from the observation
  batches it slices, `FilterResult.state_history`).

### Script & config

- `filter_smoothing.num_windows: 1` (default — byte-identical Phase 1 path)
  and `filter_smoothing.window_shift: 1` scalars; new group
  `filter_smoothing/evolution: none|random_walk` mirroring
  `conf/filtering/evolution/*` (`none` → `IdentityEvolution` at the
  orchestrator default; `random_walk` → `RandomWalkEvolution(std=…)`).
- **Truth vs prior horizons differ**: the truth (and its observations) must be
  simulated over all `T = L + (num_windows−1)·s` cycles, while the prior
  trajectory is sampled over the *first window only* (`L` cycles, as today).
  The knot-alignment validation applies to the prior's window, and the truth
  sampler's knot grid must span the full horizon.
- Artifacts: `posterior_params.nc` becomes the full-horizon assembled
  trajectory; `window_diagnostics.yaml` (per window: span + iteration
  diagnostics); `posterior_state.nc` stays the final filtered state;
  `run_info.yaml` gains `num_windows`, `window_shift`, `total_cycles`, and
  per-window timing. Single-window runs keep today's artifact set unchanged.

### Tests

1. `num_windows=1` ≡ Phase 1 `run()` (bit-identical params and state).
2. Observation slicing: stub records which obs rows each window's inner passes
   saw; window `w` sees exactly rows `[w·s, w·s+L)`.
3. State carry: window 2's first-cycle forecast starts from window 1's
   final-pass analysis after cycle `s` (stub records states; test `s=1` and
   `s=L`).
4. Parameter carry + leading edge: with `IdentityEvolution`, window 2's prior
   knots = window 1's posterior knots shifted, and the appended knot equals
   the last carried knot; with `RandomWalkEvolution`, the appended knot
   differs per member and its ensemble spread grows by ~`std`.
5. Finalization/assembly: `result.params` time axis spans the full horizon;
   early knots equal the posterior of the window they left.
6. Validation: `T` mismatch, `window_shift ∉ [1, L]`, `num_windows < 1` raise.
7. E2E smoke: `num_windows=2, window_shift=1` — artifacts exist,
   `posterior_params` spans `T` cycles' knots, `window_diagnostics` has 2
   entries; plus a compose/guard test for the new group and scalars.

## 9. Decisions taken / open points

- **Static (no-`time`) parameters in the prior Dataset** (e.g.
  `vertical_inflow_exponent`): Phase 1 carries them through forecasts unchanged
  and *excludes* them from the outer update (they pass through
  `ParamAugmentation.flatten` as single scalars, so including them later is a
  mask away). Estimating them jointly is a config knob to add when needed.
- **Trailing knot**: kept in `Θ` (updated only via prior temporal correlation);
  documented above.
- **Observation time coordinate** for temporal localization: segment end.
  Midpoint is a one-line change if the taper proves asymmetric in practice.
- **`num_time_points` wiring**: derived from the sampled prior's `time` dim at
  run time (same pattern as `run_esmda.py`'s smoother override), never a config
  literal.
- Naming: module `filter_smoothing`, class `FilterSmoothingESMDA`,
  configs `conf/filter_smoothing/*`, script
  `scripts/filter_smoothing/run_filter_smoothing.py`. (Rename to
  `WindowedTrajectoryESMDA` if preferred — decide before the first commit.)

---

## 9. Amendment: within-segment interpolation, knots decoupled from cycles

Adopted after Phase 2, superseding §4.2 (`_TrajectoryStateFilter`), the
"knot spacing = `time.simulation_time`" requirement of §5, and the positional
carry of §8a wherever they differ. It **changes the numbers** of existing
configurations, including at the old one-knot-per-cycle spacing — deliberately.

### What changed

1. **Segment restriction instead of knot selection.** `_params_for_cycle` no
   longer slices knot `k`. Cycle `k` receives the trajectory restricted to
   `[k·Δt_cycle, (k+1)·Δt_cycle]` on the trajectory's own seconds-valued clock:
   the linearly interpolated value at the segment start, every knot strictly
   inside the segment, and the interpolated value at the segment end, re-based
   onto a segment-local axis `[0, Δt_cycle]`. Outside the knot range the value
   is **clamped** (`np.interp` semantics), so a segment past the last knot holds
   it and a one-knot trajectory is constant. The result still carries a `time`
   dimension, and the backends consume it through the path
   `params@prior_params=dynamic` ESMDA already uses (`uvel_time.dat` for pylbm,
   the nudging schedule for pyudales, interpolated by the Fortran); the schedule
   is call-relative, hence the local axis.

2. **`time.seconds_per_knot` decoupled from `time.simulation_time`.** The knot
   grid is the estimated trajectory's resolution and nothing else: any grid from
   one knot upward, coarser or finer than the cycle, is legal. The
   `num_knots >= num_cycles` guards in `FilterSmoothingESMDA.run` and
   `run_moving_window`, and the `seconds_per_knot == simulation_time` guard in
   the run script, are gone; what is validated instead is the *shape* of the
   sampled grid (origin, uniform spacing, reach).

3. **`cycle_length` is now an explicit input.** Knot times are physical seconds,
   so placing cycle `k` on the trajectory needs the segment length.
   `FilterSmoothingESMDA(cycle_length=…)` (config:
   `cycle_length: ${time.simulation_time}`), falling back to
   `forward_model.simulation_time`; a time-varying `params` with neither is
   rejected at `run()`.

4. **Temporal localization in fractional cycles.** Knot `j`'s row coordinate is
   `knot_time / cycle_length` instead of the knot index `j`. `temporal_radius`
   stays in cycles, and the mapping reduces to the old integers exactly when the
   spacing equals the cycle length.

5. **Time-based moving window.** The window advances
   `shift_seconds = s · cycle_length`; knots before it are finalized, the rest
   are carried with their times re-based by `−shift_seconds` (so each window's
   clock again opens at its first segment), and as many knots as left are
   appended at `t_last + i·Δt`. Finalized knots are put back on the horizon's
   clock (`+ w·shift_seconds`) when the result is assembled. A new **alignment
   guard** rejects a `shift_seconds` that is not a whole number of knot
   spacings — the knot grid would otherwise drift out of phase with the windows,
   so a different set of knots would leave each time and the finalized pieces
   would not tile the horizon. At `Δt = cycle_length` every step reduces to the
   old positional `[:s]` / `[s:]` slicing, and the `num_windows = 1`
   object-identity path is untouched.

### Why

The paper's Eq. (6) makes `θ_k` piecewise-constant over segment `k`, and §4.2
implemented exactly that. But the **truth** trajectory in this repo is generated
on a knot grid that the solver *interpolates* — the same `uvel_time.dat` /
nudging path — so a piecewise-constant estimate fits a different model of the
forcing than the one that produced the data. Estimating the trajectory the
solver actually consumes removes that inconsistency (and the associated
representation error), at no cost in the outer update: the map `Θ → D` stays as
linear as it was, and `ParamAugmentation` already handled an arbitrary knot
count.

Decoupling the grids follows from the same change: once a segment is a piece of
a continuous trajectory rather than one knot's value, nothing ties the knot
count to the cycle count. That buys two things the fixed coupling could not —
a trajectory resolved *finer* than the assimilation cycle (the cycle length is
set by the observation cadence and the cost of a forecast, not by how fast the
forcing varies), and a *coarser*, better-conditioned control vector when the
cycles are short.

### Consequences worth knowing

- Numbers change for every existing filter-smoothing configuration, including
  `seconds_per_knot == simulation_time`: a segment now ramps between its
  bracketing knots instead of holding knot `k`.
- The knot at the window's end is no longer "updated only through prior temporal
  correlations": it is the last segment's end value, so observations reach it
  directly.
- A knot grid coarser than a cycle constrains `window_shift` (the shift must
  cover whole knots); a grid finer than a cycle by an integer factor never does.
- Non-uniform grids — which `build_knot_times` emits when the window span is not
  a multiple of the spacing — are tolerated for a single window and rejected for
  a moving one.
