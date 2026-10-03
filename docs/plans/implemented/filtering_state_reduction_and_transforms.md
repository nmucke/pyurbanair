# Plan: filtering state reduction, ETKF, LETKF, and observation TSVD

> **Status: design plan.** Written 2026-08-10 after tracing the implemented
> ESMDA state-reduction and sequential-filter paths. This is a working plan,
> not a maintained reference; verify file and function anchors against the
> current tree before implementation. Implementation is split into exactly two
> PRs: PR 1 contains steps 1–3, and PR 2 contains ETKF, LETKF, and observation
> TSVD.
>
> **Review incorporated 2026-08-10.** The revision fixes the streaming
> forgetting convention and physical-space posterior-inflation semantics, and
> adds finite basis updates, filtering-pipeline benchmark provenance, LETKF
> chunking/resource gates, fail-fast analysis/localization capabilities,
> localized-diagnostic provenance, realistic float32 tests, and stricter reuse
> of the existing reduction implementation.

## 1. Goal

Add optional reduced state analyses to the sequential filter, then add the
deterministic ensemble-transform family and observation-space spectral
regularization. The work must support the filter's three modes without changing
their meaning:

- `mode="state"`: reduce and update the end-of-cycle state; parameters remain
  fixed.
- `mode="parameter"`: state reduction is not applicable; configuring it must
  raise instead of appearing to do work.
- `mode="joint"`: reduce the state block, keep scalar parameters in their
  existing full representation, and update `[state coefficients | parameters]`.

The work is about the **analysis representation**, not a reduced CFD forecast.
Every ensemble member still runs through the full forward model. Because the
ordinary EnKF increment already lies in the span of the current ensemble
anomalies, truncated state SVD is purely a projection filter: it can remove
update directions but cannot add directions the ensemble did not contain. At
the shipped shape (`N_e=50`, about `N_d=12`), fitting a current-state SVD is
expected to make the analysis slightly slower, not faster. The candidate
benefits are truncation as regularization and, for the streaming variant,
cross-cycle memory of recurring forecast-error directions. Any speed or memory
benefit is an empirical result to demonstrate, never an assumed goal; none of
this reduces the dominant solver cost.

## 2. Current behavior and the data choice

`BaseFilter.run()` forecasts a complete segment, applies the temporal
observation operator to that segment, extracts its final frame, and analyzes
that final frame. The relevant cross-covariance is therefore between the final
forecast state and the segment observation:

$$
C_{x_T y} = X_{k,T}Y_k^T,
\qquad
X_{k,T} = \frac{1}{\sqrt{N_e-1}}
\left[x_{k,T}^{f,(e)}-\bar{x}_{k,T}^{f}\right]_{e=1}^{N_e}.
$$

The default state-reduction input for filtering must consequently be the
**current cycle's final forecast-state anomalies**. The basis is constructed
before the current analysis, so it contains no current-observation leakage.
Past observations influence it only through the analyzed state propagated into
the current forecast, which is the correct filtering recursion.

Do not copy ESMDA's `window_snapshots` behavior verbatim. A basis built from
raw frames across a segment and centered around one member-time grand mean
mixes deterministic mean-flow evolution with forecast uncertainty. If a future
trajectory source is added, construct it as

$$
B_k = \frac{1}{\sqrt{N_t}}
[X_{k,1}, X_{k,2}, \ldots, X_{k,N_t}],
$$

where each `X_{k,t}` is centered across ensemble members separately. Weight
each cycle equally so output cadence does not silently change the covariance.
This trajectory option is deliberately deferred from the first two PRs: the
current Xie-and-Castro cadence provides only about two saved frames per cycle,
and the filter analyzes only the final frame.

### State inner product

The SVD is only meaningful after defining the state norm. PR 1 must support a
diagonal per-variable scaling applied before fitting/encoding and inverted when
decoding the increment. Default scale `1.0` preserves the current Euclidean
flattening for the velocity-only state. There is no reliable automatic
mixed-unit rejection in PR 1: existing datasets do not consistently carry
`units` metadata, and multiple variables (`u`, `v`, `w`) legitimately share
one unit. Validate configured scale keys and require finite positive values;
record the resolved scales. When complete, conflicting non-empty `units` attrs
are present and scales were omitted, emit a warning rather than guessing.
Grid-volume weighting for nonuniform meshes is a documented follow-up, not part
of these PRs.

## 3. Cross-cutting invariants

1. With all nonzero current-ensemble modes retained, a current-cycle state SVD
   must reproduce the existing global stochastic EnKF update—including the
   configured prior/posterior inflation—to realistic JAX float32 tolerance
   (start around `rtol=1e-5` and scale `atol` to the test data; do not demand
   machine-epsilon equality).
2. Decode only the coefficient **increment**. Projection residuals remain on
   each member, so zero gain leaves the full state numerically unchanged.
3. Predicted observations always come from the full forecast segment. Reduction
   never projects the observation operator or the CFD model.
4. State reduction is valid only in `state` and `joint` modes. Parameter-only
   construction with a reduction configured must raise rather than no-op.
5. PR 1 supports only an unlocalized reduced state update. Global POD
   coefficients have no physical coordinates, so combining them with existing
   distance localization is invalid. Correlation screening of modal
   coefficients is possible in principle but changes the localization meaning
   and is also deferred. Reject `state_reduction != None` together with
   `localization != None`.
6. In joint mode, state coefficients are reduced while parameter rows retain
   their current analysis, inflation, and evolution behavior.
7. Physical-state diagnostics must remain comparable between reduced and
   unreduced runs. Do not report coefficient spread under the existing
   `state_spread_*` names.
8. In-memory and on-disk ensemble modes must give the same analysis. The on-disk
   path may stream/load final frames but must not assemble all historical
   trajectories in memory.
9. Existing stochastic filtering and all ESMDA behavior remain unchanged when
   the new options are disabled.
10. Reuse as much existing code as possible. In particular, extend or compose
    the existing `StateAugmentation`, `ParamAugmentation`, `AnalysisScheme`,
    `BaseLocalization`, inflation, diagnostics, filter cycle, on-disk I/O, and
    Hydra-group plumbing instead of creating parallel versions. New helpers or
    classes need a concrete semantic difference that existing code cannot
    express cleanly; reducing duplication and avoiding unnecessary module,
    config, and abstraction clutter are review requirements, not optional
    cleanup.
11. Basis construction is transactional and finite-only. The ensemble model's
    existing failure policy normally clones successful forecast states into
    failed slots before returning in both in-memory and on-disk modes. Validate
    every candidate anomaly column anyway; a non-finite column raises before
    analysis and must leave an already initialized streaming basis unchanged.
    Never let one bad forecast permanently NaN-poison subsequent cycles.
12. Analysis schemes declare their localization capability and `BaseFilter`
    validates it at construction: stochastic EnKF is `optional`, ETKF is
    `forbidden`, and LETKF is `required`. Invalid combinations fail before the
    first forecast, not inside cycle 0.

## 4. Implementation and review process

Apply this process independently to each PR:

1. Implement the scoped steps and their tests together. Before adding a new
   abstraction, search the data-assimilation library and scripts for an
   existing setup that can be extended or composed. Refactoring shared code is
   preferable to copying it, provided ESMDA and existing stochastic-filter
   behavior remain covered and unchanged by default.
2. Once implementation and focused tests are complete, spawn a **new agent**
   for adversarial review round 1. Give it the plan, the PR diff, relevant
   maintained docs, and explicit instructions to verify claims against the
   implementation. It must focus first on mathematical and software
   correctness, including array shapes, covariance/transform identities,
   mode semantics, localization/reduction interactions, numerical edge cases,
   configuration wiring, persistence, and tests that could pass without
   exercising the intended behavior. It must also identify unnecessary
   complexity, duplicated machinery, and missed opportunities to reuse the
   existing augmentation, analysis, localization, filter-loop, I/O, config, or
   diagnostics setups.
3. Resolve every round-1 blocker and add regression coverage for correctness
   defects. Record any review recommendation deliberately rejected, with the
   technical reason.
4. After the round-1 fixes are complete, spawn a **different new agent** for an
   independent adversarial review round 2. Do not reuse the first reviewer or
   prime the second reviewer with the first review's conclusions. Give it the
   updated plan/diff/docs and the same correctness-first instructions, plus an
   explicit request to challenge whether the final design is more complex than
   necessary and whether newly added code can be deleted, collapsed, or
   replaced by existing repository mechanisms.
5. Resolve every round-2 blocker, rerun the required tests and pre-commit, and
   summarize both reviews and their resolutions in the PR description. A PR is
   not ready to merge until both independent reviews are complete and the final
   diff has been rechecked for avoidable duplication and abstraction.

## 5. PR 1 — filtering state SVD, evaluation, and streaming update

One PR contains all three steps below. Suggested branch:
`filtering-state-reduction`.

### Step 1 — current-cycle final-state SVD baseline

Implement the smallest scientifically defensible baseline before introducing
history or forgetting.

#### Library design

- Reuse `OnlineStateReduction` for the current-cycle implementation. Extend it
  minimally with an orthonormal/non-whitened coefficient option and optional
  row scaling while keeping its existing whitened ESMDA behavior as the
  default. Fitting on a final-frame versus an initial-condition ensemble is a
  caller choice, not a reason for a parallel class hierarchy.
- Add only one genuinely new reduction class for streaming state, sharing the
  existing rank selection, encoding/decoding, scaling, validation, and
  diagnostics helpers with `OnlineStateReduction`. Do not introduce a
  filtering-specific abstract interface unless implementation demonstrates a
  third behavior that cannot be expressed by these two classes.
- Prefer orthonormal coefficients
  `a = U_r.T @ (x - current_forecast_mean)` and
  `delta_x = U_r @ delta_a`. Historical singular values are useful for rank
  selection and diagnostics, but they must not be assumed to whiten the current
  cycle's ensemble.
- Add `state_reduction=None` to `BaseFilter` and
  `EnsembleKalmanFilter`. Validate the mode/localization combinations in the
  constructor.
- In `_analysis_cycle`, flatten the final forecast state, validate it is finite,
  fit/update the basis, and replace only the state block with coefficients.
  Append parameters and predicted-observation diagnostic rows exactly as
  today.
- Preserve inflation semantics in **physical state space**. Fit the basis on
  raw forecast anomalies; apply prior inflation to physical state anomalies
  before encoding and separately to parameter and predicted-observation
  anomalies. After the reduced analysis, decode the state increment onto that
  physical prior ensemble, then apply RTPS/RTPP to physical prior/posterior
  state anomalies. Apply the same posterior hook separately to the parameter
  block. The existing inflation schemes operate row-wise, so this block split
  reproduces today's augmented-array behavior while avoiding coefficient-space
  RTPS/RTPP, which does not commute with a basis rotation.
- Calculate `state_spread_prior` and `state_spread_posterior` from physical
  flattened states after the corresponding inflation stages. The physical
  prior ensemble is therefore an intentional materialization on the reduced
  path, not an implicit coefficient diagnostic. Coefficient spread gets a new
  explicitly named diagnostic if retained.

#### Configuration

- Add `conf/filtering/state_reduction/none.yaml` and
  `conf/filtering/state_reduction/svd_current.yaml`.
- Mount the group from `conf/run_filtering.yaml` and pass it into
  `filtering.filter.state_reduction`.
- Initial knobs:
  `energy_fraction`, `max_rank`, and optional `variable_scales`.
- Record the resolved reduction configuration in `run_info.yaml`.

#### Required tests

- Full-rank current-cycle SVD equals the existing global update for state and
  joint modes with the same RNG key, both without inflation and with each
  shipped inflation scheme (including the default RTPS).
- Zero predicted-observation spread leaves every full-state member unchanged.
- Truncation returns finite states, preserves shapes/coords/dtypes as currently
  guaranteed by `StateAugmentation`, and never exceeds `N_e - 1` current
  statistical rank.
- Joint mode changes both state and parameters; state mode leaves parameters
  unchanged; parameter mode plus reduction raises.
- Reduction plus localization raises with actionable guidance.
- Multiplicative prior inflation and RTPS/RTPP posterior inflation reproduce
  their full-space physical-state semantics; add a regression that would fail
  if posterior relaxation were mistakenly applied per coefficient row.
- A non-finite forecast member raises before analysis and does not mutate an
  existing streaming basis. A forecast repaired by the ensemble model's normal
  donor-substitution path remains valid input.
- In-memory and on-disk paths are equivalent on the toy ensemble model.
- Hydra composition and one smoke-sized `run_filtering.py` execution cover the
  new group.

### Step 2 — evaluation and decision instrumentation

Land the measurements in the same PR as the baseline; a reduction without a
way to determine whether it helped is incomplete.

#### Per-cycle diagnostics

Extend `CycleDiagnostics` and the persisted YAML additively with:

- retained rank and nonzero available rank (current-cycle rank is at most
  `N_e-1`; accumulated streaming rank may legitimately exceed it);
- retained singular-value energy;
- projection residual of the **current final forecast anomalies**;
- norm of the decoded state increment and the discarded increment fraction;
- basis construction and total analysis wall time;
- condition indicator such as retained `sigma_max / sigma_min`;
- for streaming mode, principal-angle/subspace drift from the previous cycle.

Use `None` for these keys when reduction is disabled so downstream readers can
handle one stable schema.

#### Comparison matrix

Run the same truth/prior/seed through:

1. full stochastic EnKF;
2. current SVD at full rank;
3. current SVD at energy fractions `0.90`, `0.95`, and `0.99`;
4. at least two hard rank caps below the default observation dimension.

Compare state RMSE/statistics, assimilated and held-out sensor scores,
innovation chi-square, physical state spread, parameter recovery in joint mode,
wall time, and peak memory. Full rank is a correctness control, not a candidate
optimization.

This seven-run matrix is a production/offline benchmark campaign, not a CI
test. Execute every run through `scripts/run_filtering_pipeline.sh` (or run
`compute_filtering_metrics.py` explicitly afterward): held-out validation
scores already come from the filtering post-processing pipeline's
`build_sensor_sets(cfg)` and saved state/config artifacts, not from a validation
operator in `run_filtering.py`. Smoke-shape configurations may place validation
sensors outside their reduced domain, so smoke tests cover wiring/correctness
only and are not used for held-out acceptance.

Record commands, git commit, resolved configs, hardware, run-directory paths,
failures, metric tables, timing, and peak memory in
`docs/temp/filtering_state_reduction_benchmark.md`; keep large run artifacts in
the normal gitignored results directories. The PR cannot claim scientific or
performance benefit without this reproducible record, but the campaign itself
must not run in CI.

For the default Xie-and-Castro shape (`N_e=50`, about `N_d=12`), explicitly
report whether batch SVD is slower than the existing
`O(N_s N_e N_d)` cross-covariance. Do not claim an analysis speedup unless the
end-to-end measurement shows one.

#### Acceptance gates

- Full-rank equivalence passes before interpreting truncated runs.
- A proposed default rank must not materially degrade held-out sensor skill or
  innovation consistency relative to the full filter.
- The reduced posterior must not look better only because its spread collapsed;
  evaluate skill and calibration together.
- Keep `filtering/state_reduction=none` as the default regardless of the first
  benchmark. Promotion requires results on more than the smoke case.

### Step 3 — exponentially forgotten incremental SVD/POD

After the current-cycle baseline and diagnostics are working, add the streaming
implementation in the same PR.

Use the unnormalized exponentially weighted accumulator

$$
C_k = \lambda C_{k-1} + B_kB_k^T,
$$

where `B_k` is the current final forecast-anomaly block. Given the previous
factor `U Sigma`, update the SVD of

$$
[\sqrt{\lambda}U\Sigma,\;B_k]
$$

by projecting `B_k` onto `U`, QR-factorizing only its residual, taking an SVD
of the resulting small matrix, and truncating immediately. Never materialize or
retain all previous state snapshots.

#### Streaming semantics and knobs

- `forgetting_factor` in `(0, 1]`. `1.0` means exact equal-weight accumulation
  of all accepted anomaly blocks; it must never suppress the new block.
  Document the old-block covariance half-life in cycles as
  `log(0.5) / log(forgetting_factor)` when it is below one.
- `energy_fraction`, `max_rank`, `update_every_n_cycles`, and
  `variable_scales`.
- Include the current finite forecast block at unit weight before its analysis.
  On cycle zero, fit `B_0` directly, so the basis and singular values exactly
  reduce to the current-cycle construction without a spurious lambda-dependent
  scale.
- Track the accumulator weight `w_k = lambda * w_{k-1} + 1`. Retained-energy
  ratios are scale invariant; absolute spectrum diagnostics use
  `Sigma / sqrt(w_k)` so they remain comparable across cycles, especially when
  `lambda=1` makes the unnormalized accumulator grow.
- Re-orthogonalize when an orthogonality diagnostic exceeds tolerance.
- If subspace projection error or drift exceeds a configured warning threshold,
  log it but do not silently change the rank policy.
- Keep the basis in memory for the current `run()` call. Persist compact
  metadata, not the potentially huge `U`, until filter restart/checkpointing is
  itself implemented.
- Validate the complete anomaly block before mutating `U`, `Sigma`, or `w_k`.
  If any column is non-finite, raise with the member indices and retain the
  previous streaming state unchanged. Do not drop individual members and
  silently change the ensemble covariance.

#### Streaming tests

- With `forgetting_factor=1` and no truncation, incremental singular values and
  subspace projection match a batch SVD of concatenated toy anomaly blocks.
- With forgetting, the represented covariance matches the explicitly formed
  exponentially weighted covariance on small matrices.
- Cycle zero matches a batch SVD of `B_0`, including its singular values.
- Rank and memory remain bounded across many cycles.
- Repeated runs are deterministic.
- A rotating low-rank linear system demonstrates that forgetting adapts faster
  than an equal-history basis.
- Current and streaming reductions obey the same state/joint mode and
  zero-increment contracts.
- Current-cycle tests enforce rank `<= N_e-1`; streaming tests do not make that
  invalid assertion because accumulated forecast subspaces can exceed the
  instantaneous ensemble rank.

### PR 1 files expected to change

- `libs/data-assimilation/src/data_assimilation/reduction.py`
- `libs/data-assimilation/src/data_assimilation/filtering/base.py`
- `libs/data-assimilation/src/data_assimilation/filtering/__init__.py`
- `conf/filtering/state_reduction/{none,svd_current,svd_streaming}.yaml`
- `conf/run_filtering.yaml`
- `scripts/filtering/run_filtering.py`
- `tests/test_state_reduction.py`
- `tests/test_filtering.py`
- `tests/test_run_filtering.py`
- `docs/temp/filtering_state_reduction_benchmark.md` (new; reproducible
  production/offline comparison record, no large artifacts)
- `docs/data_assimilation.md`
- `docs/scripts_and_configs.md`

## 6. PR 2 — ETKF, LETKF, and observation-space TSVD

This is one PR after PR 1 has merged. Suggested branch:
`filtering-ensemble-transforms`.

Keep **observation TSVD** distinct from PR 1's **state SVD**:

- state SVD chooses a state-space basis `U_r` and projects the analyzed state;
- observation TSVD truncates weak directions of the whitened predicted-
  observation anomaly matrix and regularizes the Kalman transform.

### Step 4 — global ETKF

- Add `ETKFAnalysis`, using a symmetric square-root transform in ensemble
  space. Dense decompositions must be at most `N_e x N_e` (or the smaller
  observation rank), never `N_s x N_s`.
- Preserve the forecast ensemble mean/anomaly identities and use the same
  diagonal observation-error variance contract as the stochastic analysis.
- Support state, parameter, and joint modes, existing prior/posterior
  inflation, appended observation-space diagnostic rows, and PR 1's reduced
  state representation on the global path.
- Add one declarative localization policy to `AnalysisScheme` (`optional`,
  `forbidden`, or `required`) and validate it in `BaseFilter.__init__`.
  `ETKFAnalysis` declares `forbidden`; localized deterministic behavior belongs
  to the explicit LETKF class so config names cannot lie. The existing
  stochastic scheme declares `optional`, preserving its current behavior.
- Add `conf/filtering/analysis/etkf.yaml`.

Tests must compare mean and covariance against the exact linear Kalman filter,
verify deterministic repeatability, check zero innovation/zero spread edge
cases, and verify invalid localization fails at construction. The exact
linear-KF comparison using the sampled forecast mean/covariance is the
load-bearing CI test. A stochastic-EnKF-versus-ETKF large-ensemble comparison
is statistical: pin seeds and use deliberately generous tolerances, or keep it
in the offline benchmark rather than making CI flaky.

### Step 5 — reusable observation-space TSVD

Implement one internal helper used by global ETKF and every local LETKF
analysis. For an effective diagonal observation covariance `R_eff`, form

$$
Y_w = R_{eff}^{-1/2}Y
$$

and truncate its SVD/eigendecomposition before constructing the ensemble
transform. Configuration is nested on the ETKF/LETKF analysis object rather
than exposed as a separate filter class:

- disabled by default;
- `energy_fraction` and optional `max_rank`;
- a numerical relative singular-value tolerance independent of the scientific
  truncation;
- diagnostics for available rank, retained rank, retained energy, and discarded
  spectrum.

Use the same thin SVD of `Y_w` for the untruncated low-rank transform when the
active observation dimension is smaller than `N_e`; disabling scientific TSVD
then means retaining every numerically nonzero singular direction, not falling
back to a mandatory dense `N_e x N_e` eigendecomposition. This shared kernel is
both the exact ETKF/LETKF implementation and the place where optional spectral
truncation occurs.

When disabled, the helper must reproduce the untruncated ETKF/LETKF path. TSVD
must not alter the physical observation error variances; it removes weak
**linear combinations** of observation anomalies after whitening.

Add config variants or nested settings that make the choice explicit, for
example `filtering/analysis=etkf_tsvd` and `letkf_tsvd`, while keeping
`etkf`/`letkf` untruncated.

### Step 6 — LETKF driven by existing localization strategies

- Add `LETKFAnalysis` with `AnalysisScheme.localization_policy="required"`, so
  a missing localization strategy fails in `BaseFilter.__init__`.
- Reuse `BaseLocalization.inflation_factors()` to obtain per-row observation-
  error inflation. For a local row/block, apply
  `R_eff = diag(E_inf**2 * R)`; infinite inflation excludes the observation.
- Reuse `group_ids` so co-located state components share one local transform.
  Compute once per unique block and scatter the result; do not perform an
  eigendecomposition independently for every raw state row.
- Unique blocks can still number roughly one per grid cell. Process them in
  bounded chunks and use the thin local `Y_w` SVD from step 5 when
  `N_d_active < N_e`; never materialize an
  `(n_blocks, N_e, N_e)` transform tensor for the full domain. Make chunk size
  a measured implementation detail/config knob only if one fixed safe value
  does not cover supported grids.
- Respect the existing `localize_mask`: distance localization updates state
  blocks locally and parameter/diagnostic rows globally; correlation
  localization can update both state and parameter blocks locally; appended
  predicted-observation diagnostic rows remain global.
- Preserve but explicitly label the existing observation-posterior diagnostic
  limitation: appended predicted-observation rows take a global ride-along
  update. Under LETKF/localized stochastic analyses,
  `obs_posterior_rmse` is therefore a global-analysis proxy, not `H` applied to
  the row-wise localized posterior. Add provenance such as
  `obs_posterior_rmse_kind` and do not use this value to rank ETKF versus LETKF
  in the benchmark.
- Apply the TSVD helper **after** local observation selection and whitening,
  because each LETKF block has a different effective observation matrix.
- Initially reject LETKF together with PR 1's global state reduction. Supporting
  both requires local POD bases with spatial support and is a separate project.
- Add `conf/filtering/analysis/letkf.yaml` and its TSVD-enabled counterpart.

#### Corrections to this section, measured 2026-08-11

Recorded during PR 2 implementation, against the canonical Xie-and-Castro case
and a completed 60-cycle uDALES filtering run. Where the bullets above conflict
with the measurements below, the measurements win; the deviations are
deliberate, not oversights.

1. **`group_ids` deduplication buys nothing on the shipped default.** uDALES is
   staggered, so `pres`/`u`/`v`/`w` each carry a distinct grid signature
   (`xt` vs `xm`, `yt` vs `ym`, `zt` vs `zm`) and
   `StateAugmentation.group_ids` gives each its own id range. Unique state
   blocks therefore equal `N_s` exactly (230,400 on the working-tree grid),
   not "roughly one per grid cell". Dedup is worth implementing for clarity
   and for collocated grids such as pylbm's (3:1 there), but it is **not** a
   performance lever and the resource gate must not assume it is.
2. **The binding constraint is the transform's representation, not chunk
   size.** With `N_d = 12` and `N_e = 50`, an `(n_blocks, N_e, N_e)` tensor is
   2.30 GB in float32 while the `(n_blocks, N_d, N_d)` form is 133 MB, at
   roughly one-thirteenth the arithmetic. The local transform must be built
   from the `min(N_d, N_e)` thin factors and applied in factored form; it must
   never be assembled at `N_e x N_e`, not even per chunk. Bounded chunking
   remains, but as a safety bound rather than the mechanism that makes LETKF
   affordable.
3. **"Use the thin `Y_w` SVD when `N_d_active < N_e`" cannot be a shape
   decision.** `N_d_active` varies per block, so any rank that changes array
   shape is not vectorizable. Fix the rank at `min(N_d, N_e)` and express both
   observation exclusion and TSVD truncation as zero weights / boolean masks,
   mirroring the shape-stable decoupling the existing localized stochastic
   update already uses instead of extracting active submatrices.
4. **Two facts the plan omits, both favorable.** At `localization_radius=7.5`
   only ~5% of blocks have any active observation; the rest are provably
   unchanged and can be partitioned out host-side, since nothing in
   `data-assimilation` is jitted. And run 1 of the resource table — the
   localized stochastic baseline — is already measured on this hardware at
   full size: `analysis_time` mean 1.289 s/cycle over 60 cycles.
5. **Already satisfied, no new work needed.** `obs_posterior_rmse_kind`
   provenance exists from PR 1 and already emits `unlocalized_ride_along`
   under localization; and LETKF-plus-state-reduction is rejected structurally,
   because `BaseFilter` already refuses `state_reduction` together with any
   localization and LETKF requires localization.

#### LETKF performance gate

A naive LETKF performs one `N_e x N_e` decomposition per spatial block and can
be much more expensive than the current localized stochastic update, whose
default local systems are only about `12 x 12`. Correctness tests are not
enough. On at least the canonical Xie-and-Castro grid, record wall time, peak
memory, active-observation counts, unique block count, chunk size, and time per
cycle for:

1. localized stochastic EnKF;
2. LETKF retaining all numerical local modes;
3. LETKF with configured TSVD.

The implementation passes the resource gate only if memory remains bounded by
the documented chunking design and the runtime is usable for the intended
experiment. Otherwise keep LETKF explicitly experimental and record the
limitation; do not compensate by changing defaults or weakening localization.
Record the commands, commit/configs, hardware, metrics, and resource table in
`docs/temp/filtering_ensemble_transform_benchmark.md`; the realistic-grid runs
are an offline campaign, not CI jobs.

Required LETKF tests:

- all-ones localization equals global ETKF;
- an infinite-inflation observation has no effect;
- block-grouped output equals the equivalent repeated-row calculation;
- chunked and unchunked toy calculations are identical;
- distance localization keeps joint parameter rows on the global ETKF path;
- correlation localization acts on both joint blocks;
- local TSVD-disabled results equal full-rank local transforms;
- local TSVD rank never exceeds the active-observation or ensemble-anomaly
  rank;
- no-active-observation rows remain unchanged;
- scheme localization policies fail fast for ETKF+localization and
  LETKF-without-localization;
- smoke-sized Hydra runs cover ETKF, LETKF, ETKF+TSVD, and LETKF+TSVD.

### How LETKF and TSVD compose

LETKF and observation TSVD act on different axes and are complementary:

1. **LETKF is spatial/statistical localization.** For each state row or grid
   block, it decides which observations apply and how strongly, producing a
   block-specific `R_eff` and local `Y`.
2. **TSVD is spectral regularization.** Within that already selected and
   whitened local observation matrix, it decides which observable ensemble
   directions are sufficiently energetic/informative to retain.
3. The required order is therefore:
   `localize -> form R_eff -> whiten Y -> TSVD -> ensemble transform`.

TSVD alone remains a global ETKF with weak observation-space directions
removed. LETKF alone retains every numerically nonzero direction in each local
analysis. Enabling both gives a spectrally truncated local transform; neither
feature substitutes for the other. Because the present default has only about
12 global observations and fewer active observations per local block, TSVD may
provide little benefit inside LETKF and must remain off by default until the
diagnostics show persistent local ill-conditioning.

### PR 2 files expected to change

- `libs/data-assimilation/src/data_assimilation/filtering/etkf.py` (new;
  shared ETKF/TSVD kernels and ETKF/LETKF analysis classes)
- `libs/data-assimilation/src/data_assimilation/filtering/analysis.py`
- `libs/data-assimilation/src/data_assimilation/filtering/base.py`
- `libs/data-assimilation/src/data_assimilation/filtering/__init__.py`
- `conf/filtering/analysis/{etkf,etkf_tsvd,letkf,letkf_tsvd}.yaml`
- `conf/run_filtering.yaml` comments/examples
- `tests/test_filtering_etkf.py` (new)
- `tests/test_filtering.py`
- `tests/test_run_filtering.py`
- `docs/temp/filtering_ensemble_transform_benchmark.md` (new; realistic-grid
  ETKF/LETKF/TSVD resource and skill record)
- `docs/data_assimilation.md`
- `docs/scripts_and_configs.md`

## 7. Deliberately deferred

- Segment-trajectory or across-cycle raw-state snapshot bases.
- Local/tiled POD that can coexist with distance-localized LETKF.
- Correlation localization of global POD coefficients.
- DMD/SPOD bases; current cycles are too short and these target dynamics rather
  than the forecast-error covariance used by the filter.
- Nonlinear autoencoder/generative latent-state filters.
- Reduced-order/Galerkin CFD forecasts and multifidelity ensembles.
- Filter checkpoint/restart persistence of the full streaming basis.

## 8. Verification and merge gates for both PRs

- Tests land with implementation in each PR.
- Run focused data-assimilation/filtering tests and smoke-sized filtering
  entry-point tests locally. Run the full dev suite in CI: the current macOS
  development machine can abort in the unrelated torch/libomp stack, so a
  local full-suite crash is not evidence about these PRs and must not be hidden
  or misreported as a test result.
- Run `pixi run -e dev pre-commit` before commit and judge new failures against
  the target branch's recorded baseline.
- Update maintained data-assimilation/config docs in the same PR.
- Complete the two independent adversarial-agent review rounds in §4 and apply
  their required correctness and simplification/reuse fixes.
- Record benchmark hardware, state shape, `N_e`, `N_d`, rank, localization,
  save mode, and solver; timings without these fields are not comparable.
- Do not change the default analysis (`stochastic`) or default state reduction
  (`none`) in either PR.

## 9. Primary references

- M. Brand (2002), *Incremental Singular Value Decomposition of Uncertain Data
  with Missing Values*: https://www.merl.com/publications/TR2002-24
- D. Matsumoto et al. (2019), *Application of Incremental Proper Orthogonal
  Decomposition for the Reduction of Very Large Transient Flow Field Data*:
  https://doi.org/10.20485/jsaeijae.10.1_117
- B. Hunt, E. Kostelich, and I. Szunyogh (2007), *Efficient Data Assimilation
  for Spatiotemporal Chaos: A Local Ensemble Transform Kalman Filter*:
  https://arxiv.org/abs/physics/0511236
