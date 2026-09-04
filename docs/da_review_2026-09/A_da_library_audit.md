# Audit: `libs/data-assimilation` (pyurbanair)

Read-only audit of every module under
`libs/data-assimilation/src/data_assimilation/` (~7,970 LOC), against
`docs/data_assimilation.md`, `docs/ensemble_transform_filters.md`, the `conf/`
tree, the three entry points, and `tests/`.

Paths are relative to the repo root. `DA/` abbreviates
`libs/data-assimilation/src/data_assimilation/`.

Labels: **[bug]** = confirmed wrong or provably inert; **[smell]** = suspicious /
fragile / mis-specified but defensible; **[opinion]** = my judgement call.

Shipped shape used throughout: `N_e = 50`; `xie_and_castro` = 6 assimilation
sensors × `[u, v]` = **12 obs per frame**, `output_frequency = 2 s`,
`simulation_time = 120 s`; `barcelona` = `[u, v, w]`, `output_frequency = 10 s`,
`simulation_time = 300 s`. ESMDA aggregates at `interval_seconds = 15` →
`N_d ≈ 8 × 12 = 96` per window (xie); the filter's `N_d = 12` per cycle.

---

## 1. Inventory

### 1.1 `DA/observation_operator.py` (441 L)

| Symbol | What / math | Wired from |
|---|---|---|
| `ObservationOperator` (`:10`) | `H(x)`: sample `obs_states` at sensors; flat layout `[var0 all sensors, var1 all sensors, …]` | `create_observation_operator` in `src/pyurbanair/config/hydra_helpers.py:200`, called by all three run scripts |
| — index mode `obs_ids_x/y/z` (`:52`) | `isel` vectorized indexing | **Not reachable**: the helper only ever passes `obs_x/y/z` (`hydra_helpers.py:204`). Tests only. |
| — coord mode `obs_x/y/z` (`:45`) | trilinear interpolation | The production path (`obs.mode: points` / `grid`) |
| — `dim_mapping` (`:66-89`) | per-solver staggered dim names (`udales`/`pylbm`/`palm`) | `cfg.*_model.solver_name` |
| `TemporalObservationOperator` (`:168`) | applies `H` per frame → `("[ensemble,] time, obs")` | built when `obs.temporal_mode: full` (both shipped cases) |
| `AggregateObservations` (`:258`) | bins frames by absolute `floor((t−t0)/Δ)`, reduces with mean/median/max/min; locks interval count after first call (`:347-361`) | `create_aggregate_observations(cfg.esmda)`; used by `run_esmda.py:787` and `run_filter_smoothing.py:587`. **Never by `run_filtering.py`** (by design) |
| `flatten_observations` (`:389`) | time-major flatten `(T,obs) → (T·N_obs,)` | `BaseSmoothing._get_observations` (`DA/smoothing/base.py:56`) and `run_esmda.py:448` |
| `sensor_observation_coords` (`:406`) | tiles sensor xyz so obs `j` ↦ sensor `j % num_sensors` | `_BaseESMDA._observation_coords` (`DA/smoothing/esmda.py:285`), `BaseFilter._localization_plumbing` (`DA/filtering/base.py:1448`) — distance localization only |

### 1.2 `DA/interpolation.py` (174 L)

`interpolate_dataarray_at_points` (`:93`): trilinear interpolation with
staggered-dim alias resolution (`_resolve_axis_dim_name:23`) and a half-*edge*-cell
extrapolation margin (`:68-72`). Reached on every `H(x)` evaluation. Clean; the
edge-local margin (rather than a grid median) is correct for stretched grids.

### 1.3 `DA/augmentation.py` (351 L)

| Symbol | Math | Wired |
|---|---|---|
| `ParamAugmentation.flatten/unflatten` (`:72`, `:103`) | `(time, ensemble)` → `{name}_{t}` scalars; `pin_initial_time_point` drops `t=0` from the augmented vector and re-inserts it per member | `TimeVaryingParameterESMDA` (`esmda.py:637-647`); `BaseFilter` uses the static (identity) config (`filtering/base.py:436`) |
| `ParamAugmentation.group_ids` (`:136`) | **all knots of one parameter share one block id** | `esmda.py:633` → `_augmented_state_update(param_group_ids=…)` |
| `StateAugmentation.flatten/unflatten` (`:183`,`:194`) | sorted-var, C-order flatten to `(N_s, N_e)` | ESMDA state variants + `BaseFilter._analysis_cycle:1029` |
| `.flatten_snapshots` (`:220`) | `(N_s, N_e·N_t)` | only `basis_source="window_snapshots"` |
| `.group_ids` (`:242`) | co-located cells share a block, keyed on the exact coordinate bytes per axis | block-grouped localization |
| `.row_scales` (`:291`) | per-variable scale vector | `OnlineStateReduction.resolve_row_scales` (filter only) |
| `.row_coords` (`:313`) | physical `(x,y,z)` per row; axis from first char of dim name | distance localization only |

### 1.4 `DA/inflation.py` (101 L)

`InflationScheme` (`:31`, no-op), `MultiplicativeInflation(factor)` (`:53`,
`dev ← f·dev`), `RTPS(alpha)` (`:63`, `σ_a ← σ_a + α(σ_f − σ_a)`),
`RTPP(alpha)` (`:86`, `dev_a ← α dev_f + (1−α) dev_a`).
Wired only into `BaseFilter` via `conf/filtering/inflation/{none,multiplicative,rtpp,rtps}.yaml`.
**The ESMDA smoothers have no `inflation` argument at all** — the module
docstring's claim "it lives at the package root because the smoother can use the
same schemes" (`inflation.py:7-9`) is aspirational; grep `esmda.py` for
`inflation` returns only the `alpha` comments.

### 1.5 `DA/io.py` (44 L)

`load_dataset` (eager open+close, avoids fd exhaustion), `get_sorted_state_files`
(regex `state_<int>.nc`). Used by both packages. Fine.

### 1.6 `DA/localization/`

* `taper_inflation` (`base.py:166`) — Vossepoel Eqs. 9–10:
  `b = (1−β)T/√(ln E_max)`, `E_inf = exp(((d−βT)/b)²)` for `d > βT`, `inf` for `d > T`.
* `_group_inflation` (`base.py:64`) — `segment_min` of `E_inf` over a block.
* `resolve_row_inflation` (`base.py:97`) — mask→`inf`, group, then mask→`1.0`.
  Shared by the stochastic local update and `LETKFAnalysis`.
* `active_observations` (`base.py:142`) — `isfinite & > 0`.
* `BaseLocalization.localized_update` (`base.py:268`) — `vmap` over `N_aug` rows,
  each solving its own `(N_d, N_d)` system with excluded obs decoupled by
  zeroing rows/cols and putting 1 on the diagonal (`:393-396`).
* `CorrelationLocalization` (`correlation.py:27`) — `d = 1−|ρ|`, `T = 1−ρ_t`,
  `ρ` with `ddof=1`; default `ρ_t = min(3/√N_e, .99)`.
  Config `conf/esmda/localization/correlation.yaml`, `conf/filtering/localization/correlation.yaml`.
  **This is the `run_esmda.yaml` default** (`conf/run_esmda.yaml:46`).
* `DistanceLocalization` (`distance.py:27`) — Euclidean `‖x_row − x_sensor‖` via
  the `|a|²+|b|²−2a·b` identity; `requires_coordinates = True`,
  `localizes_parameters = False`. Configs in both trees; default radius 10 m.

### 1.7 `DA/reduction.py` (734 L)

* `_numerical_rank` (`:57`), `_select_rank` (`:95`) — rank policy; `conservative`
  cut = `eps·max(shape)` (whitened path), non-conservative = `eps·max(min(shape), 64)`.
* `OnlineStateReduction` (`:121`) — thin SVD of `(scaled − mean)/√(n−1)`;
  `encode = Σ_r⁻¹ Φ_rᵀ(u−ū)` (whitened) or `Φ_rᵀ(u−ū)`;
  `decode_increment = Φ_r Σ_r dξ` / `Φ_r dξ`.
  Configs: `conf/esmda/state_reduction/svd.yaml` (whiten default `True`),
  `conf/filtering/state_reduction/svd_current.yaml` (`whiten: false`).
* `StreamingStateReduction` (`:461`) — incremental POD with
  `C_k = λ C_{k−1} + B_k B_kᵀ`, CGS2 re-projection (`:606-613`), rare
  re-orthogonalization (`:670-681`), principal-angle drift (`:710`).
  Config `conf/filtering/state_reduction/svd_streaming.yaml` (`λ=0.9`, `max_rank=50`).

### 1.8 `DA/smoothing/base.py` (165 L)

`BaseSmoothing`: `_get_observations` (aggregate → flatten, `:41`), `_forecast_step`,
`_observation_step` (in-memory or per-member on-disk with `join="override"`),
abstract `_analysis`, `__call__` with the `final_forecast` seam (`:139`).

### 1.9 `DA/smoothing/esmda.py` (1174 L)

* `_BaseESMDA.__init__` (`:50`) — validates diagonal, strictly-positive `C_D`;
  **enforces `num_steps/alpha == 1`** (`:104-111`); rejects a coordinate-based
  localization on a parameter-only smoother (`:116-127`); creates/clears
  `step_{i}/` dirs.
* `_compute_kalman_update` (`:219`) — thin wrapper over
  `filtering/analysis.py::stochastic_enkf_update` with the tempered `alpha`.
* `_analysis` (`:320`) — the MDA loop: forecast → failure substitution →
  `_one_step` → feed analyzed IC forward → optional prune; then the posterior
  forecast (unless `final_forecast=False`) and `_final_time_smoothing_step`.
* Variants: `ParameterESMDA` (`:484`), `TimeVaryingParameterESMDA` (`:575`),
  `StateAndParameterESMDA` (`:680`), `StateESMDA` (`:1054`),
  `StateAndTimeVaryingParameterESMDA` (`:1115`).
  All five reachable via `conf/esmda/smoother/*.yaml`; `run_esmda.yaml` defaults to `dynamic`.
* Opt-in attributes set post-construction: `prune_disk_steps`/`keep_prior_disk_step`
  (`run_esmda.py:829-830`, `run_filter_smoothing.py:686-687`),
  `collect_obs_diagnostics` (`run_esmda.py:837`, `run_filter_smoothing.py:692`).

### 1.10 `DA/filtering/analysis.py` (248 L)

`validate_variances` (`:35`); `stochastic_enkf_update` (`:61`) — the single
implementation of
`x ← x + C_MD (C_DD + α C_D)⁻¹ (y + √α √C_D Z_c − H(x))`, with **centered**
perturbations (`:152`) and a Cholesky solve + finiteness guard (`:165-172`);
`AnalysisScheme` ABC with `localization_policy` (`:177`);
`StochasticEnKFAnalysis` (`:220`, `α = 1`). Config
`conf/filtering/analysis/stochastic.yaml` (default).

### 1.11 `DA/filtering/base.py` (1560 L)

`CycleDiagnostics` (`:69`, 40 fields), `FilterResult` (`:159`),
`BaseFilter` (`:184`) / `EnsembleKalmanFilter` (`:1531`).
Constructor validation: mode, `C_D` 1-D-or-diagonal, distance-localization vs
`mode="parameter"`, `localization_policy`, reduction vs
`parameter`/localization/`basis_source`, `parameter_evolution` vs `mode="state"`,
and the spread-maintenance guard (`:415-425`).
Cycle loop `run` (`:749`) → `_analysis_cycle` (`:975`): flatten → basis fit →
prior inflation → encode → append `H(x)` ride-along rows → `_localization_plumbing`
→ serial frame sweep (`_assimilate_frames:1213`) → decode increment → posterior
inflation → diagnostics → split → parameter evolution.
Opt-in attributes: `collect_pred_obs` (`run_filtering.py:774`,
`run_filter_smoothing.py:713`), `assimilate_every_n_step`,
`collect_forecast_frames` (`run_filtering.py:779`).

### 1.12 `DA/filtering/etkf.py` (1275 L)

`ObservationTSVD` (`:83`), `ObservationTransform` (`:157`),
`whiten_observations`/`_whiten` (`:210`,`:264`), `_retention_mask` (`:297`),
`_transform_factors` (`:438`), `ensemble_transform` (`:472`),
`apply_ensemble_transform` (`:580`), `ETKFAnalysis` (`:607`, policy `forbidden`),
`_resolve_chunk_size` (`:711`), `LocalTransformDiagnostics` (`:755`),
`LETKFAnalysis` (`:830`, policy `required`, dedup on the canonical inflation
vector at `:1140-1147`).
Configs `conf/filtering/analysis/{etkf,etkf_tsvd,letkf,letkf_tsvd}.yaml`.
The math checks out: symmetric root, `W_a 1 = 1` structurally (`Y_w 1 = 0 ⇒ Vᵀ1 = 0`),
damped weights `s/((N−1)+s²)` and `√((N−1)/((N−1)+s²))`, suffix-energy rank cut.
This is the best-written module in the library.

### 1.13 `DA/filtering/parameter_evolution.py` (84 L)

`ParameterEvolution` ABC, `IdentityEvolution` (`:42`), `RandomWalkEvolution` (`:52`,
`θ ← θ + N(0, std²)` per variable, sorted-name key split).
Config: only `conf/filtering/evolution/{none,random_walk}.yaml` — `IdentityEvolution`
has **no** config.

### 1.14 `DA/filter_smoothing/base.py` (783 L)

Pure helpers `knot_times` (`:90`), `_interpolate_knots` (`:121`),
`params_for_segment` (`:144`), `trajectory_values_at` (`:209`),
`segment_bounds` (`:247`); `FilterSmoothingResult` (`:309`);
`FilterSmoothing` (`:355`) with `_run_static` (`:617`) and `_run_dynamic` (`:661`,
the "correction on the ESMDA schedule" loop). Reached from
`scripts/filter_smoothing/run_filter_smoothing.py` /
`conf/run_filter_smoothing.yaml` (defaults: `esmda/smoother=dynamic`,
`filtering.mode=state`, `esmda.interval_seconds=15`, both localizations `none`).

---

## 2. Dead / unused / vestigial

**Hard dead (no caller anywhere in the repo):**

1. `src/pyurbanair/config/hydra_helpers.py:259` `create_C_D` — zero callers.
   `run_esmda.py:801` builds `jnp.diag(...)` inline; `run_filtering.py:750`
   builds the 1-D vector. Only referenced in a comment (`DA/filtering/base.py:320`).
   The docs still present it as the API (`docs/data_assimilation.md` §11).
2. `src/pyurbanair/utils/run_utils.py:32` `get_ensemble_mean_field` — zero
   callers. It is the *only* consumer of `_BaseESMDA.get_state`
   (`DA/smoothing/esmda.py:213`), which is therefore also dead in production
   (`run_esmda.py` re-assembles states by streaming files itself).
3. `ObservationOperator`'s index-based mode (`DA/observation_operator.py:52-59`,
   `:122-133`) — `create_observation_operator` never constructs it.

**Unreachable from any shipped config (constructible, no YAML):**

4. `IdentityEvolution` (`DA/filtering/parameter_evolution.py:42`) — no config
   option. Worse, it is a **hole in the spread-maintenance guard** (see §3.6).
5. `OnlineStateReduction(basis_source="window_snapshots")` — the only YAML
   (`conf/esmda/state_reduction/svd.yaml:32`) ships `initial_condition`. Its
   whole support chain is therefore cold: `_get_window_states` (`esmda.py:761`),
   `_flatten_window_snapshots` (`:783`), `_basis_snapshots` (`:797`),
   `StateAugmentation.flatten_snapshots` (`augmentation.py:220`),
   `snapshot_stride`. It is also computationally infeasible at production size
   (SVD of `230k × (50·60)`), so it can never be turned on as shipped.
6. `AggregateObservations` modes `median`/`max`/`min` (`:288-290`) — every
   config uses `mean`, and the class docstring concedes only `mean` is
   equivalent to aggregating the state.
7. `OnlineStateReduction.variable_scales` / `resolve_row_scales` /
   `StateAugmentation.row_scales` — both filtering YAMLs ship `null`; the ESMDA
   `svd.yaml` does not even expose the key.
8. `StreamingStateReduction.subspace_warning_threshold` — no YAML key.
9. `LETKFAnalysis.block_chunk_size` — documented as "for tests and benchmarks,
   not for configs" (`etkf.py:735`), and indeed absent from both LETKF YAMLs.

**Vestigial / doc-drift:**

10. `DA/inflation.py:7-9` claims the smoother can use these schemes. It cannot —
    `_BaseESMDA` has no inflation hook.
11. `docs/data_assimilation.md` lines 36–60 print a source tree with
    `esmda.py` at the top level; the real layout is `smoothing/esmda.py`. §11's
    worked example calls `create_C_D`, which no script does.
12. `docs/data_assimilation.md:486-488` links `docs/reduced_state_da.md`, which
    does not exist (the doc says so itself). `esmda.py:686` references it too.
13. `docs/ensemble_transform_filters.md` and
    `docs/temp/filtering_ensemble_transform_benchmark.md`: the ETKF/LETKF family
    ships **unbenchmarked** by the authors' own statement. ~1,275 LOC of
    production code with no measured accuracy/cost claim.

**Duplicated logic (three near-copies of the same plumbing):**

14. On-disk step/cycle management: `esmda.py:175-217` (`_set_step_results_dir`,
    `_prune_step_results_dir`, `get_state`) vs `filtering/base.py:715-743`
    (`_set_cycle_results_dir`, `_prune_cycle_results_dir`, `get_state`) vs
    `filter_smoothing/base.py:475-525` (`_staging_dir`, `_collect_segment_dir`,
    which re-implements the filter's pruning semantics because the filter's own
    never fires with one cycle per call).
15. Per-member on-disk observation/state assembly: `smoothing/base.py:86-118`
    vs `filtering/base.py:570-591` and `:663-675` / `:697-712` — four copies of
    "sort files, load, concat with `join='override'`".
16. `_record_pred_obs` exists twice with the same doc (`esmda.py:192`,
    `filtering/base.py:476`).
17. Localization row-descriptor assembly: `esmda.py:910-940` vs
    `filtering/base.py:1412-1451` — same mask/group/coords construction, two
    implementations that must be kept in sync by hand.
18. `_BaseESMDA._observation_coords` (`esmda.py:277`) is a one-line wrapper
    around `sensor_observation_coords`, which the filter calls directly.

---

## 3. Correctness and quality concerns

### 3.1 **[bug]** Block grouping silently disables correlation localization for time-varying parameters — and this is the shipped default

`ParamAugmentation.group_ids` (`augmentation.py:136-153`) assigns **one block id
to every time knot of a parameter**. `_group_inflation` (`localization/base.py:91`)
then takes `segment_min` of `E_inf` over the block. So for `esmda/smoother=dynamic`
with `esmda/localization=correlation` and `block_grouping: true` — the exact
default of `conf/run_esmda.yaml` (`:46`, `:48`) — every knot of `inflow_angle`
receives the *union* of observations correlated with *any* knot, tapered at the
*strongest* correlation over all knots.

With 4–5 knots per parameter, `N_e = 50`, `ρ_t = 0.35` and ~96 observations, the
probability that at least one knot clears the threshold for a given observation is
close to 1, and the block minimum then also collapses the taper toward `E_inf = 1`.
The net effect is that the default configuration runs an essentially **global
update while reporting `localization: correlation`** in `run_summary.yaml`.

This also destroys the one thing correlation localization could genuinely buy for
a time-varying inflow: automatic *temporal* localization (knot `t` should only see
observations from around time `t`). Setting `block_grouping: false` restores it.
I would test that as a first experiment — it is a one-line config change.

### 3.2 **[bug]** ESMDA observation noise is inconsistent with `C_D` by a factor of ~7.5 in variance

`run_esmda.py:876-885` perturbs every **raw** frame with `obs_error_std`, then
`AggregateObservations` interval-means them (`run_esmda.py:787`), while
`C_D = σ² I` is built at the aggregated size (`run_esmda.py:801`). With
`interval_seconds = 15` and `output_frequency = 2` each bin holds 7–8 frames, so
the actual aggregated noise variance is `σ²/7.5`, not `σ²`.

The comment at `run_esmda.py:878-881` calls this "deliberate (the assimilation
stays mildly conservative)". It is not mild: it under-weights the data by 7.5×,
which for a smoother with `num_steps = 2` (`alpha = 2`) means an effective
likelihood tempering of ~15 relative to the data actually available. **[opinion]**
This is a plausible first-order contributor to "modest accuracy" on the ESMDA side.
The bins are also unequal (7 vs 8 frames), so the effective error is
heteroscedastic while `C_D` is uniform.

The filter has no such issue (`run_filtering.py:731`, no aggregation).

### 3.3 **[bug]** `IdentityEvolution` defeats the spread-maintenance guard

`filtering/base.py:415-425` refuses `mode ∈ {parameter, joint}` when both
`parameter_evolution is None` and `inflation is None`. `IdentityEvolution()`
(`parameter_evolution.py:42`) satisfies the check while doing nothing — and
`tests/test_filtering.py:1421` deliberately uses it for exactly that purpose. The
guard tests for the *presence of an object*, not for the *property* it is meant to
enforce.

### 3.4 **[smell]** `inflation=RTPS` alone does not maintain parameter spread

The guard also accepts an inflation as sufficient. For static parameters the
forecast does not change them, so the prior spread of cycle `k` **is** the
posterior spread of cycle `k−1`. RTPS then gives
`σ_k = σ_{k−1}(g + α(1−g))` with `g < 1` the analysis shrinkage — a geometric
decay, slower but still monotone. `conf/run_filtering.yaml` ships
`filtering/evolution: none` + `filtering/inflation: rtps` with `mode: joint`
(`:59-63`, `:151`), i.e. the default filtering configuration lets the parameter
ensemble collapse over 30 cycles. Only `RandomWalkEvolution` (or additive
inflation, not implemented) genuinely maintains it. `RTPP(α=1)` would, but at the
cost of never updating the anomalies at all.

### 3.5 **[bug]** `RandomWalkEvolution` default `std: 0.5` is mis-scaled by ~20× in both directions

`conf/filtering/evolution/random_walk.yaml:9-10` applies one scalar `0.5` to
*every* parameter. Prior stds (`conf/params/static.yaml`): `inflow_angle`
`std = 10` (degrees), `velocity_magnitude` `std = 0.5` (m/s).

* velocity: the per-cycle random walk equals the **entire prior std**. Over 30
  cycles it injects `√30 × 0.5 ≈ 2.7 m/s` of variance — 5× the prior. The filter
  cannot converge on speed; it re-randomizes it faster than it learns.
* angle: `0.5°` per cycle is 1/20 of the prior std — near-zero, so angle spread
  still collapses.

`RandomWalkEvolution` supports a per-name mapping (`parameter_evolution.py:74`)
and the YAML comment even shows the syntax — it is just not used.
**[opinion]** With `docs/plans/isda2026_talk_experiments.md:76` declaring
`filtering/evolution=random_walk` the filter default for the whole campaign, this
is the single most likely cause of poor speed estimation in the filter runs.

### 3.6 **[smell]** Parameter evolution noise is added to the *reported* posterior, but not to the reported spread

`filtering/base.py:1194-1196` evolves `params` *after* the analysis, and `run`
stores that evolved object into `params_history` and `FilterResult.params`
(`:927-928`, `:934`). Meanwhile `param_spread_posterior` is computed at `:1149`
from `dev_post`, i.e. **before** evolution. So the saved parameter ensemble and
the reported parameter spread describe different objects, and every reported
posterior carries one extra random-walk step of noise (with `std = 0.5` on
velocity, that dominates the analysis increment). Ordering the evolution as the
*forecast* of the next cycle — i.e. applying it at the top of the next
`_analysis_cycle` — would fix both.

The same noise leaks into the hybrid: `filter_smoothing/base.py:729`
`correction = result.params − schedule` is computed from the post-evolution
params, so the persistent correction accumulates random-walk noise.

### 3.7 **[bug, documented]** `final_time_smoothing` assimilates the observations twice

`esmda.py:996-1051`. The docstring itself (`:1008-1024`) states it: the MDA loop
consumed `Σ 1/α_k = 1`, and this adds one more `α = 1` update of a trajectory
already conditioned on the same data. It also runs a warning at `:1029`. The
resulting ensemble is overconfident by construction. It is off by default
(`conf/run_esmda.yaml:129`) and requires `state_reduction`. **[opinion]** Delete
it; a knob whose docstring says "must not be used for uncertainty quantification"
next to a codebase whose whole point is ensemble UQ is a trap.

### 3.8 **[bug]** The hybrid assimilates each window's observations twice

`filter_smoothing/base.py:581-592` hands the *whole window* to the ESMDA phase
(full likelihood, `Σ1/α_k = 1`), and `:634` / `:716` then hand the *same raw
frames* to the filter (full weight, `α = 1`) for the state. So parameters and
state are each conditioned on the same data at full weight. In a linear-Gaussian
setting the total likelihood weight per window is 2 and the joint posterior is
overconfident; the state increments also partly re-explain innovations the
parameter update already absorbed.

This is a genuine algorithmic property of the design, not an implementation slip,
but it is **not mentioned anywhere** in `docs/data_assimilation.md §9`, which
discusses only the coupling of the filtered state into the next window's ESMDA
prior. It deserves a line in the docs and, ideally, an `α`-split between the two
phases. **[opinion]** This is the kind of thing that shows up as "the ensemble
looks confident but held-out sensors are mediocre".

### 3.9 **[smell]** Diverged ensemble members enter the covariance before repair

Both loops repair *after* the update: `esmda.py:394` substitutes params after the
forecast but `_one_step` (`:399`) uses `state` — the raw forecast — for both
`pred_obs` and `_get_states`; `apply_failure_substitutions_to_state` runs at
`:413`, after. Same in `filtering/base.py:870-921`.

`BaseEnsembleForwardModel._resolve_failures`
(`src/pyurbanair/base_ensemble_forward_model.py:336-344`) does clone donor states
into the returned forecast, so the arrays are finite — but that means the
ensemble contains **exact duplicate members**. `C_MD`/`C_DD` are then computed at
an effective ensemble size below `N_e` with an underestimated spread, and nothing
reports how many duplicates a given analysis saw. There is no diagnostic field
for it in `CycleDiagnostics`.

### 3.10 **[smell]** Localized update is `O(N_aug · N_d²)` memory and will OOM for a state-bearing ESMDA smoother

`localization/base.py:386-396` forms `C_DD_alpha` per row under `vmap`, i.e. a
`(N_aug, N_d, N_d)` intermediate. For the filter (`N_d = 12`, `N_s = 230k`) that
is 132 MB — fine. For `esmda/smoother=state_and_parameter` with the aggregated
`N_d = 96` it is **8.5 GB** in float32, and for `barcelona` (`N_d = 20 × 18 =
360`) it is 120 GB. Nothing guards this; the combination composes cleanly and
then dies. `LETKFAnalysis` solves exactly this problem by deduplicating on the
inflation vector (`etkf.py:1140-1147`), but that machinery exists only on the
filtering side and only for the deterministic scheme.

### 3.11 **[smell]** Inconsistent linear algebra between the global and localized paths

`analysis.py:165` uses `cho_factor`/`cho_solve` with a documented SPD argument and
an explicit finiteness guard. `localization/base.py:396` uses a generic
`jnp.linalg.solve` (LU) with **no** finiteness check, on a matrix that has been
deliberately made non-symmetric-looking by the `keep` mask (it is still SPD, so
Cholesky would work). A localized update that goes non-finite is not caught.

### 3.12 **[smell]** Prior/posterior inflation compose incorrectly if a scheme implements both hooks

`filtering/base.py:1062` overwrites `dev_prior` with the *inflated* anomalies, and
`:1142` passes that same inflated `dev_prior` to `inflate_posterior`. So a scheme
implementing both hooks would relax toward its own inflated spread (double
counting). No shipped scheme does both, so this is latent, not active.

### 3.13 **[smell]** `whiten` has no effect on the ESMDA result

`OnlineStateReduction.encode` divides by `σ` (`reduction.py:373-375`) and
`decode_increment` multiplies back (`:388-390`). A Kalman update is equivariant
under an invertible diagonal transform of the augmented rows (the increment is
`C_MD (…)⁻¹ d` and `C_MD` scales linearly), so the decoded state increment is
**identical** with `whiten=True` and `whiten=False` — up to float error. The only
real effect of the flag is which numerical-rank cut is applied
(`_select_rank(conservative_rank=self.whiten)`, `:337`). The docs present it as a
substantive ESMDA-vs-filtering distinction (`docs/data_assimilation.md:459-464`,
`reduction.py:5-7`); it is not.

### 3.14 **[smell]** Serial frame sweep treats within-segment frames as independent

`_assimilate_frames` (`filtering/base.py:1213-1250`) applies one full-weight
analysis per frame, each with the same per-frame `C_D`. That is only correct if
observation errors are independent *and* representativeness errors are
independent across frames a few seconds apart in an LES — they are strongly
correlated. In the shipped configuration `T = 1` per cycle, so this is currently
inert; it becomes wrong the moment anyone assimilates a multi-frame segment.

### 3.15 **[smell]** `_record_reduction_diagnostics` runs the analysis a second time per cycle

`filtering/base.py:1304-1309`. It is cheap (a `(k, N_e)` array) and the ordering
hazard is handled explicitly (`:1160-1175`), but it means the analysis object's
`last_transform`/`last_diagnostics` are mutated by a *diagnostic*. That is a
fragile contract held together by a comment and one test.

### 3.16 **[smell]** `_BaseESMDA` forbids any non-uniform MDA schedule

`esmda.py:104-111` requires `num_steps/alpha == 1` for a single scalar `alpha`.
There is no way to express a decreasing schedule (`α = [8, 4, 2, 1.6]`,
Emerick & Reynolds 2013 / Rafiee & Reynolds 2017 adaptive), which is the standard
remedy when the first MDA step over-corrects a nonlinear model. See §5.

### 3.17 Things I checked and found **correct**

* ESMDA per-step perturbation: `y + √α √C_D Z_c` with denominator `C_DD + α C_D`
  (`analysis.py:153-156`) — textbook, and the centering of `Z` (`:152`) is the
  right `O(1/√N_e)` bias fix.
* Perturbed-obs vs deterministic consistency: both use `R_eff = E_inf² C_D`
  (`localization/base.py:378`, `etkf.py:289`) via the same
  `resolve_row_inflation`/`active_observations`, so the localization radius means
  the same thing in both.
* The ETKF kernel (`etkf.py:472-577`) — symmetric root, structural mean
  preservation, damped weights, suffix-energy rank criterion. Correct and unusually
  carefully argued.
* `AggregateObservations` absolute binning and the interval-count lock
  (`observation_operator.py:343-361`) — the right call; a silent gap would shift
  the whole innovation vector.
* `resolve_row_inflation`'s mask→group→restore ordering
  (`localization/base.py:132-139`) — correct, and the failure mode it prevents is
  real.
* `sensor_observation_coords`' `j % num_sensors` tiling matches
  `flatten_observations`' time-major layout (`observation_operator.py:389-441`).
* Filter windowing really is mathematically inert (`rng_key` mutated in place,
  noise drawn for the whole horizon before the window loop,
  `run_filtering.py:706-737`).

---

## 4. Features that probably do not help this problem

**[opinion]** throughout, but §4.1 is a proof, not a preference.

### 4.1 State-space reduction (both `OnlineStateReduction` and `StreamingStateReduction`) — cannot improve the estimate, only degrade it

The EnKF/ESMDA increment is `ΔX = X W` where `X` is the `(N_s, N_e)` forecast
anomaly matrix and `W` an ensemble-space weight matrix. Every increment therefore
already lies in `col(X)`, of dimension ≤ `N_e − 1 = 49`.

`OnlineStateReduction` fits its basis on **the same `X`** it then projects
(`esmda.py:887-890`, `filtering/base.py:1048-1051`), so `U_r ⊆ col(X)`.
Consequently:

* with `r = rank(X)` the reduction is exactly the identity;
* with `r < rank(X)` it discards part of an increment that was already confined
  to the ensemble span.

There is **no spurious-correlation suppression** — unlike localization, projection
onto a subspace of the ensemble's own span cannot remove sampling error, because
the sampling error lives in that same span. The shipped `energy_fraction: 0.99`
(`conf/esmda/state_reduction/svd.yaml:24`, `conf/filtering/state_reduction/svd_current.yaml:14`)
therefore *strictly removes update directions*.

The compute argument does not rescue it either: the analysis cost is
`O(N_s N_e N_d + N_d³)` — utterly negligible next to `N_e` CFD runs. The reduction
saves nothing that matters and costs an SVD.

`StreamingStateReduction` spans more than the current ensemble (accumulated over
cycles with `λ = 0.9`), but it is used only as a *projector on the increment*
(`filtering/base.py:1131-1136`), never to augment the covariance — so it cannot
add rank to the update either, and with `max_rank: 50` against a growing
accumulator the current ensemble's span may not even be contained in the retained
basis. Strictly a loss.

E8 in `docs/plans/isda2026_talk_experiments.md` is a 6-run "state-reduction
ladder". My prediction: `energy_fraction = 1.0` reproduces the unreduced run to
float32, and everything below it is monotonically worse. If that is the result, it
is worth a slide as a negative result — but it should not be a P1 spend.

### 4.2 Correlation localization at `N_e = 50` with 6 sensors

`ρ_t = 0.35` against a sampling noise floor of `1/√50 ≈ 0.14` is only ~2.5σ, so
roughly 1 % of genuinely-uncorrelated `(row, obs)` pairs survive — across
`N_s ≈ 230k` rows that is thousands of spurious retained links per observation.
More importantly, with only 6 sensors there is very little *to* localize: with
`N_d = 12` per frame and `N_e = 50` the covariance is already over-determined
(`N_e ≫ N_d`), which is the regime where localization buys the least. And per
§3.1 the shipped block grouping neutralizes it anyway.

### 4.3 Observation TSVD (`etkf_tsvd`, `letkf_tsvd`)

The authors say it themselves (`etkf.py:88-90`, both YAMLs): with `N_d ≈ 12` and
`N_e = 50` there is nothing to regularize. `min(N_d, N_e−1) = 12` and the weights
are damped at `s → 0`, so truncation can only remove information. E10 is correctly
marked P2/backup. **[opinion]** It should not be run at all at this sensor count.

### 4.4 LETKF at 6 sensors

`localization_policy = "required"` (`etkf.py:898`) makes LETKF the only localized
deterministic option, but with `N_d = 12` and a 10 m radius on a domain whose
`docs/plans/…` note says ~95 % of rows see no observation, most blocks are
identity and the rest are near-global. The dedup machinery (`etkf.py:1140-1147`)
is well-built and will collapse the block count to a handful — meaning LETKF
≈ global ETKF for most rows. E9 will probably show "no difference", which is a
fine result but a cheap one.

### 4.5 `final_time_smoothing`

See §3.7. Statistically invalid by its own docstring, requires an equally
questionable state reduction, and cannot run on-disk (`esmda.py:722-728`).

### 4.6 `basis_source: window_snapshots`

Infeasible at production size (§2 item 5) and, per §4.1, a projector that spans
*more* than the ensemble does not add rank to the update — it only changes which
part of the increment survives truncation.

---

## 5. Gaps — standard ensemble-DA ingredients that are absent

Ordered by my estimate of expected payoff for *this* problem (chaotic urban LES,
`N_e = 50`, 6 sensors, inflow-parameter + state estimation).

### High value

1. **Observation-error / representativeness-error estimation, and any prior
   consistency check on `C_D`.** `obs_error_std = 0.1 m/s` is a hardcoded config
   scalar in all three run configs, applied uniformly to `u/v/w` at street level
   in an LES. The true model-minus-truth mismatch at a sensor is dominated by
   turbulence the parameters do not control, and is very likely 10–50× larger in
   variance. **[opinion]** Together with §3.2 this is the most likely single cause
   of mediocre held-out-sensor scores: an over-tight `C_D` makes the filter
   overfit assimilated sensors and collapse, which shows up precisely as good
   assimilation-sensor / poor validation-sensor RMSE.
   *Missing:* Desroziers (2005) diagnostics (`⟨d_a d_bᵀ⟩ → R`), which are ~10
   lines given the `pred_obs`/`pred_obs_post` rows already recorded
   (`filtering/base.py:1122`), and any innovation-χ²-driven `R` rescaling.
2. **Adaptive inflation.** `MultiplicativeInflation` is a fixed scalar; there is
   no Anderson (2007/2009) spatially-varying adaptive inflation, no Miyoshi (2011)
   adaptive multiplicative inflation, and no χ²-driven feedback — even though
   `innovation_chi2` is computed every cycle (`filtering/base.py:1498`) and then
   only *written to a YAML file*. Closing that loop is the cheapest real
   improvement available.
3. **Additive inflation / model-error stochastic terms.** Nothing adds structured
   noise to the *state*. In a chaotic LES with a 10 s cycle the dominant error is
   unresolved turbulence, which is additive model error, not a multiplicative
   rescaling of a 50-member spread. `RandomWalkEvolution` is the only additive
   noise anywhere and it only touches parameters.
4. **Per-block inflation / per-parameter evolution scaling.** `RTPS(α)` and
   `RTPP(α)` apply one `α` uniformly to state and parameter rows
   (`filtering/base.py:1142`, which passes the whole `dev_post`), and
   `RandomWalkEvolution` ships a single scalar (§3.5). Both blocks have entirely
   different dynamics and units.
5. **A non-uniform / adaptive MDA schedule.** §3.16. Standard ESMDA practice is a
   decreasing `α` (or Rafiee–Reynolds adaptive `α` from the data mismatch), which
   is exactly the remedy for a first over-correction on a nonlinear model.
   `num_steps = 2` uniform is a very short and very aggressive schedule.
6. **ESMDA innovation-consistency diagnostics inside the library.** The filter has
   `innovation_chi2` per cycle; ESMDA has nothing — the normalized data mismatch
   is computed downstream in `scripts/esmda/_esmda_common.py:629`. There is no
   ESMDA-side "is `C_D` consistent with the actual innovations" number.

### Medium value

7. **Iterative EnKF / IEnKS / running-in-place** for the state. Nothing here
   iterates the *state* analysis against the nonlinear model within a cycle; the
   filter is a single linear update per cycle and the smoother iterates only the
   window IC. For a strongly nonlinear urban wake, IEnKF/IEnKS (Bocquet & Sakov
   2014) is the standard step up from EnKF and is a natural fit given the ESMDA
   machinery already re-forecasts.
8. **4D / asynchronous localization.** `_assimilate_frames` is a 4D-ish serial
   sweep, but there is no time-dependent localization (an observation at `t−T`
   should influence the state at `t` less, and along the advected path). With
   `interval_seconds = 15` on the ESMDA side, all 8 intervals influence the `t=0`
   IC equally.
9. **Parameter localization by physical influence.** Distance localization
   explicitly refuses to localize parameters (`distance.py:55`,
   `localizes_parameters = False`), and correlation localization is neutralized by
   block grouping (§3.1). For a *time-varying* inflow schedule, a knot at `t=90 s`
   should not be updated by an observation at `t=10 s` — a purely temporal
   localization kernel on `|t_knot − t_obs|` would be trivial to add via the
   existing `taper_inflation` and is the single most physically-motivated
   localization for this problem. **[opinion]** Highest-payoff missing feature after
   the `R`/inflation items.
10. **Sampling-error-corrected localization** (Anderson 2012 SEC / Ying & Zhang
    2015 GC-tuning). The correlation strategy uses a fixed `ρ_t` with a hard cut;
    an SEC table would be better-calibrated at `N_e = 50` and is a drop-in
    `BaseLocalization` subclass.
11. **Hybrid / climatological covariance.** No `α B_clim + (1−α) P_ens` anywhere.
    With `N_e = 50` against `N_s ≈ 230k` this is the other standard rank-deficiency
    remedy besides localization — and unlike the state reduction (§4.1) a
    climatological term *can* add rank to the update.

### Lower value here

12. **Gaussian anamorphosis / rank-based transforms.** Velocity fields in a wake
    are skewed and `velocity_magnitude` is positivity-constrained
    (`conf/params/static.yaml:21` clips at 0.1) — the Kalman update can and does
    push members past that bound and the clip is applied outside the DA. A log or
    rank transform on `velocity_magnitude` would be cheap. **[opinion]** Worth
    trying for the parameter block; probably not worth it for the state.
13. **Rank histograms / CRPS inside the library.** They exist downstream
    (`docs/temp/rank_histogram_math.md`, the evaluation package) but not as
    per-cycle diagnostics.
14. **Particle / Gaussian-mixture filters, localized particle filters.** Correctly
    absent — at `N_e = 50` with `N_s ≈ 230k` they are not viable, and the
    `AnalysisScheme` interface leaves the door open.
15. **Restart / checkpointing of the DA state.** `StreamingStateReduction`'s basis
    is explicitly "in-memory run state and not a restart checkpoint"
    (`docs/data_assimilation.md:717`); `BaseFilter.rng_key` likewise. A long HPC
    run cannot resume.

---

## 6. If I could change five things

**[opinion]**, ordered by expected effect on held-out-sensor accuracy per unit of work:

1. Fix the observation-error specification: either stop aggregating with a
   mismatched `C_D` (§3.2) or scale `C_D` by the bin count; then run a Desroziers
   or χ²-based check to find the `σ` the innovations actually imply. Expect the
   honest `σ` to be several times 0.1 m/s.
2. Fix `RandomWalkEvolution` to a per-parameter mapping scaled to each prior's
   std (§3.5), and move the evolution to the start of the next cycle (§3.6).
3. Set `block_grouping: false` for correlation localization on the dynamic
   smoother, or add a temporal-distance localization for the knots (§3.1, §5.9).
4. Drop `final_time_smoothing`, `basis_source=window_snapshots`, `create_C_D`,
   `get_ensemble_mean_field`/`get_state`, and (after one confirming run) the whole
   state-reduction path (§2, §3.7, §4.1). That is ~900 LOC of maintained surface
   that cannot help.
5. Close the χ² → inflation loop (§5.2) rather than only logging it.
