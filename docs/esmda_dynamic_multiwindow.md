# ES-MDA for Time-Varying Inflow Estimation — Dynamic, Multi-Window Setup

Background notes for a presentation. Scope is deliberately narrow: the
**dynamic (time-varying) parameter** case run over **multiple assimilation
windows**, as driven by [`scripts/run_esmda.py`](../scripts/run_esmda.py) with

```
esmda/smoother=dynamic   params@prior_params=dynamic   esmda.num_assimilation_windows=3
```

Every section is written so a slide can be lifted from it directly. Symbols are
collected in the glossary at the end.

---

## 1. What problem are we solving?

We observe a turbulent urban flow at a handful of fixed sensors and want to
infer the **time-varying upstream inflow forcing** that produced it — here two
scalar time series:

- `inflow_angle(t)` — wind direction (degrees),
- `velocity_magnitude(t)` — wind speed (m/s).

These two series are the *parameters*. They are not constant: they drift over
the simulation horizon, so we estimate a **trajectory**, not a scalar. The flow
solver (pylbm, a lattice-Boltzmann LES) maps a parameter trajectory to a state
field; the observation operator maps the state to sensor readings. ES-MDA
inverts that chain.

**One-line framing for a slide:** *Given sparse velocity sensors, recover the
time-varying wind speed and direction at the domain inflow.*

---

## 2. The end-to-end pipeline

```
 truth params (AR2, seed 7)  ──►  truth solver  ──►  truth state  ──►  obs operator  ──►  observations d
                                                                                              │  (+ noise)
 prior params (AR2, seed 2)  ──►  ES-MDA  ◄───────────────────────────────────────────────────┘
       (ensemble of 32)            │
                                   ├─ forecast ensemble through pylbm
                                   ├─ predict observations
                                   ├─ localized Kalman update of params
                                   └─ repeat over windows (rollout)
```

Two deliberate design choices:

- **Twin experiment with an anti–inverse-crime gap.** Truth and prior are *both*
  AR(2) processes but with **different seeds and external means** (truth seed 7,
  prior seed 2; see [`conf/params/dynamic_truth.yaml`](../conf/params/dynamic_truth.yaml)
  vs [`conf/params/dynamic.yaml`](../conf/params/dynamic.yaml)). The truth is
  therefore *not* a draw from the prior's generative process.
- **Truth-from-disk.** The truth is a pre-simulated, spun-up artifact
  (`ground_truth_spunup/{state,params}.nc`) loaded via `run.truth_dir`, so the
  assimilation starts from a fully developed flow rather than a cold start.

---

## 3. The dynamic parameter model — AR(2) relaxation

Both truth and prior trajectories come from a **critically-damped AR(2)
process** ([`AR2RelaxationModel`](../src/pyurbanair/dynamic_parameters/ar2_relaxation.py),
following Evensen 2024):

$$\frac{dz}{dt}=w,\qquad \frac{dw}{dt}=-2\lambda w-\lambda^2 z+\eta(t),\qquad \lambda=\frac{\sqrt 3}{\ell}$$

- `z(t)` is a **C¹-smooth, unit-variance, zero-mean** anomaly with correlation
  length `ℓ = 100 s`.
- The physical parameter is the anomaly wrapped in an external envelope:
  $$x(t) = x_{\text{ext}} + \Sigma_{\text{ext}}\, z(t).$$
  For `inflow_angle`: `x_ext = 25°`, `Σ_ext = 6°`. For `velocity_magnitude`:
  `x_ext = 6 m/s`, `Σ_ext = 0.5 m/s` (clipped at `min = 0.1`).
- Integrated **exactly** with the closed-form transition matrix
  `F = exp(A·dt)` and exact one-step process-noise covariance
  `Q = P_stat − F P_stat Fᵀ` — no substepping, stationary covariance preserved
  per step.

### Knots / control points

Each window carries the trajectory on a small set of **time knots**
(`time_coords = linspace(0, simulation_time, num)`):

- **Prior / assimilation grid:** `num = 6` knots per window → `t = 0, 36, 72,
  108, 144, 180 s`.
- **Truth grid:** finer, `num = 18` over the full horizon (independent sampling).

These 6 knots per parameter are exactly the scalars ES-MDA updates (Section 5).

### Between-window extrapolation (the rollout link)

At a window boundary the next window's prior is **not** re-drawn cold — it is
*anchored* to the previous window's posterior so the trajectory is C¹-continuous
across windows ([`extrapolate`](../src/pyurbanair/dynamic_parameters/ar2_relaxation.py)):

- Per member, the AR(2) state is re-initialised from the **normalized
  end-of-window posterior**: `z₀ = (x_post(t_end) − μ_end)/Σ_ext`, with `w₀` the
  finite-difference slope.
- The new prior blends the posterior-anchored draw toward the external mean with
  an exponential relaxation:
  $$x(t) = \big[\alpha(t)\,\mu_{\text{end}} + (1-\alpha(t))\,x_{\text{ext}}\big] + \Sigma_{\text{ext}}\,z(t),\qquad \alpha(t)=e^{-(t-t_0)/\ell}.$$
- **Consequence:** at the boundary (`t = t₀`, `α = 1`) the prior matches each
  member's own posterior value; deep into the window (`α → 0`) it relaxes back
  to `x_ext` with full external spread. So the **spread grows across the window**
  — tight at the boundary, loose at the far end. (Observed window-1 prior std for
  angle: `[2.0, 2.8, 4.1, 5.8, 5.4, 5.3]°`.)

**Slide takeaway:** *Each new window's prior inherits the previous posterior at
its left edge and re-opens uncertainty toward its right edge.*

---

## 4. ES-MDA — the core algorithm

ES-MDA (Ensemble Smoother with Multiple Data Assimilation, Emerick & Reynolds
2013) assimilates the **whole window of observations at once**, but in several
tempered iterations instead of a single Kalman update.

For each of `Na = num_steps = 3` iterations `i`:

1. **Forecast** every ensemble member through pylbm from a *fixed* initial state
   (the iterations all restart from the same IC — they do not chain).
2. **Predict observations** `g(m_j)` for each member `j`.
3. **Kalman update** the parameters with an **inflated** observation-error
   covariance `α_i C_D`:

$$m_j^{a} = m_j^{f} + C_{MD}\,\big(C_{DD} + \alpha\,C_D\big)^{-1}\big(d + \sqrt{\alpha}\,C_D^{1/2} z_j - g(m_j^{f})\big)$$

where
- `C_MD = (1/(N_e-1)) Σ (m_j-\bar m)(g_j-\bar g)ᵀ` — param/obs cross-covariance,
- `C_DD = (1/(N_e-1)) Σ (g_j-\bar g)(g_j-\bar g)ᵀ` — predicted-obs covariance,
- `z_j ~ N(0,I)` — fresh observation perturbation per member.

The inflation factors satisfy `Σ_i 1/α_i = 1`. Here `α` is **constant = 3** over
3 steps (`3 × 1/3 = 1`), the standard equal-weight schedule.

After the `Na` updates a **final forecast** is run with the converged params; its
last frame is the state handed to the next window.

Implementation: [`_BaseESMDA._analysis`](../libs/data-assimilation/src/data_assimilation/smoothing/esmda.py)
(loop), [`_compute_kalman_update`](../libs/data-assimilation/src/data_assimilation/smoothing/esmda.py)
(the equation above), [`TimeVaryingParameterESMDA`](../libs/data-assimilation/src/data_assimilation/smoothing/esmda.py)
(the dynamic specialization).

**Why MDA and not a single update?** Tempering with `α C_D` makes the otherwise
strongly nonlinear flow→obs map easier to invert: each step takes a gentler step
and the ensemble is re-forecast in between, reducing the linear-Gaussian bias of
a one-shot ensemble smoother.

---

## 5. Augmented state for time-varying parameters

The Kalman update acts on a flat vector. For the dynamic case each **(parameter,
time-knot)** pair is treated as its own scalar
([`_flatten_time_varying_params`](../libs/data-assimilation/src/data_assimilation/smoothing/esmda.py)):

```
inflow_angle_0 … inflow_angle_5 , velocity_magnitude_0 … velocity_magnitude_5
```

So the augmented state is `2 parameters × 6 knots = 12` rows × `N_e = 32`
columns. The flatten keeps all knots of one parameter contiguous, which matters
for block grouping (Section 7).

### Pinning the boundary knot

From **window 1 onward**, `pin_initial_time_point = True`: the `t=0` knot of each
parameter is **removed from the augmented state and reinserted unchanged**. It is
already fixed by the cross-window extrapolation (it equals the previous window's
last knot), so freezing it preserves C¹ continuity across the window seam. In
window 0 the `t=0` knot is free (cold GP draw over a spun-up flow).

- Window 0 augmented state: `2 × 6 = 12` rows.
- Windows 1+: `2 × 5 = 10` rows (knot 0 pinned out).

**Slide takeaway:** *Window 0 fits all 6 knots; later windows fit knots 1–5 and
inherit knot 0 from the previous window.*

---

## 6. The observation operator

[`conf/case/xie_and_castro/obs.yaml`](../conf/case/xie_and_castro/obs.yaml) +
[`ObservationOperator`](../libs/data-assimilation/src/data_assimilation/observation_operator.py).

**Spatial.** 7 point sensors, interpolated from the grid:

| field | value |
|---|---|
| x | `-10 m` (all 7) — just downstream of the `x=-20` inflow plane |
| y | `10, 20, 30, 40, 50, 60, 70 m` |
| z | `7 m` (all 7) |
| components | `u, v, w` |

So per time the operator returns `7 sensors × 3 components = 21` values. The
sensors sit near the inflow, so they measure something close to the imposed
profile — directly informative about `velocity_magnitude` and `inflow_angle`.

**Temporal.** `temporal_mode = intervals`, `interval_seconds = 20`,
`aggregation_mode = mean` ([`TemporalObservationOperator`](../libs/data-assimilation/src/data_assimilation/observation_operator.py)).
Each window (`180 s`) is binned into `180/20 = 9` intervals; frames within a bin
are mean-averaged. Final observation-vector length:

$$N_d = 9\ \text{intervals} \times 7\ \text{sensors} \times 3\ \text{components} = 189.$$

**Observation error.** Diagonal, `C_D = σ² I` with `σ = obs_error_std = 0.25`
(see the failure-mode note in Section 9). The truth observations are perturbed by
`√C_D · N(0,I)` before assimilation.

**Slide takeaway:** *189 observations per window = 7 near-inflow sensors × 3
velocity components × 9 time-averaged intervals.*

---

## 7. Localization — adaptive correlation truncation

Localization is the most consequential and most tunable piece. We use
**correlation-based localization with error-variance tapering** (Vossepoel et al.
2025), [`CorrelationLocalization`](../libs/data-assimilation/src/data_assimilation/localization/correlation.py).
Unlike distance localization it needs **no spatial coordinates** for the
parameter rows — only ensemble correlations — so it applies naturally to abstract
parameter knots.

For each augmented row `l` and each observation `j` it estimates the ensemble
correlation `ρ(l,j)` and builds an **inflation factor** `E_inf[l,j]`:

| `E_inf[l,j]` | meaning |
|---|---|
| `= 1` | observation fully influences the row |
| `1 < E_inf < ∞` | observation **tapered** (its error variance scaled by `E_inf²`) |
| `= ∞` | observation **excluded** from the row |

Rules:

- **Truncation.** Exclude observation `j` from row `l` when `|ρ(l,j)| < ρ_t`. The
  threshold defaults to the theoretical first guess `ρ_t = 3/√N_e` (for `N_e=32`
  → `≈ 0.53`); the paper recommends tuned values in **[0.3, 0.4]**.
- **Tapering.** For surviving observations, inflate the error variance smoothly
  from `1` (at correlation distance `d_c = β d_t`) up to `E_max²` (at the
  truncation distance `d_t = 1 − ρ_t`), with `β = tapering_beta = 0.5`,
  `E_max = max_inflation = 8`. Near-cutoff observations are down-weighted rather
  than switched off abruptly.
- A row whose observations are **all** excluded receives the **identity update**
  (left unchanged).

### Per-row vs. block grouping

`block_grouping` selects the granularity of the local analysis:

- **`false` (per-row).** Each knot is localized independently. A knot is updated
  only by observations whose correlation *with that specific knot* clears `ρ_t`.
- **`true` (grid-block, paper sec. 3b).** All knots of one parameter are grouped
  (by base name, [`_group_ids_by_base_name`](../libs/data-assimilation/src/data_assimilation/smoothing/esmda.py))
  and updated **jointly with one shared observation selection**. Mechanically
  ([`_group_inflation`](../libs/data-assimilation/src/data_assimilation/localization/base.py)),
  the block's inflation vector is the **per-observation minimum** over its rows:
  - an observation is active for the block if it is active for **any** knot;
  - the taper is driven by the **strongest** correlation in the block.

  Each knot still updates through its **own** cross-covariance `C_MD`; grouping
  only decides *which observations are admissible*. The net effect: a strongly
  constrained knot lets its siblings "see" the same observations, so an
  individual knot is no longer frozen just because its own marginal correlation
  dipped below the cutoff.

**Slide takeaway:** *Per-row localization can freeze weakly-correlated knots;
block grouping lets all knots of one parameter share the observations any of them
correlates with.*

---

## 8. The multi-window rollout loop

[`run_esmda.py`](../scripts/run_esmda.py), `run()`. For `num_windows = 3`:

```
state_input = None
for window in range(3):
    save prior
    pin knot 0  ⟺  window > 0
    window_obs = truth_obs[window block]  +  noise          # 189-vector
    posterior, final_state = ESMDA(state=state_input,
                                   params=prior, observations=window_obs)
    state_input = final_state[last frame]                   # warm-start next window
    if window < 2:
        prior = AR2.extrapolate(posterior, next_window_times)   # Section 3
```

Key points:

- **Observation slicing.** The truth is sliced into `num_windows` contiguous,
  *half-open* frame blocks (`n_per_window` frames each), so the boundary frame is
  not double-counted and the obs vector always matches the assimilation model's
  per-window output.
- **Warm start.** The final state of each window's last forecast becomes the
  initial condition of the next window — the flow is continuous, only the
  parameters are re-estimated.
- **Outputs.** Per window: `prior_params`, `posterior_params`, `posterior_state`.
  Plus final rollout plots (`rollout_time_evolution.png`, `parameter_error.png`,
  `rollout_animation.mp4`, `final_state_with_obs.png`).

---

## 9. Known failure mode (why this report exists)

With the **default** localization settings the rollout fails after window 0:
window 0 fits all 6 knots well, but in windows 1–2 the interior/late
`inflow_angle` knots (2–5) receive the **identity update** — they do not move at
all from the prior, even though the truth direction rises to ~33° while the prior
sits near 22°.

**Mechanism (verified on the saved ensembles).** Every observation's correlation
with the late angle knots falls **below the truncation threshold** `ρ_t ≈ 0.53`,
so per-row localization (`block_grouping=false`) excludes them all → identity:

```
window 1  inflow_angle   knot1: max|ρ|=0.54  surviving=  2/189   (only this updates)
                         knot2: max|ρ|=0.48  surviving=  0/189   (frozen)
                         knot3: max|ρ|=0.32  surviving=  0/189   (frozen)
                         knot4: max|ρ|=0.22  surviving=  0/189   (frozen)
                         knot5: max|ρ|=0.26  surviving=  0/189   (frozen)
```

`velocity_magnitude` survives better (it scales all of u/v/w uniformly — a robust
signal), confirming this is a localization-gating issue, not a forward-model bug.

**Three coupled causes, in order of impact:**

1. **`block_grouping=false`** — each knot is gated alone, so only the knot
   adjacent to the strongly-constrained window boundary clears the threshold.
2. **`obs_error_std=0.25` is too small** for m/s velocities — window 0
   over-collapses the ensemble (angle std `6° → ~1°`), so later windows start
   near-degenerate and produce weak, noisy correlations.
3. **`ρ_t ≈ 0.53` is high for `N_e=32`** — the sampling-noise floor is `~0.18`
   and the paper's tuned range is `0.3–0.4`.

**Mitigations (complementary):** `block_grouping=true`,
`truncation_correlation≈0.35`, larger `obs_error_std`, larger ensemble.

---

## 10. Configuration at a glance

| Quantity | Symbol | Value | Source |
|---|---|---|---|
| Assimilation windows | — | 3 | `esmda.num_assimilation_windows` |
| Window length | — | 180 s | `time.simulation_time` |
| Output cadence | — | 1 s (~180 frames/window) | `time.output_frequency` |
| Ensemble size | `N_e` | 32 | `ensemble.ensemble_size` |
| ES-MDA iterations | `Na` | 3 | `esmda.num_steps` |
| Inflation schedule | `α` | 3 (constant) | `esmda.alpha` |
| Parameters | — | `inflow_angle`, `velocity_magnitude` | `params/dynamic` |
| Knots per window | — | 6 | `params.time_coords.num` |
| AR(2) correlation length | `ℓ` | 100 s | `correlation_length` |
| Prior ext. mean / std (angle) | `x_ext,Σ_ext` | 25° / 6° | `params/dynamic` |
| Prior ext. mean / std (speed) | `x_ext,Σ_ext` | 6 / 0.5 m/s | `params/dynamic` |
| Sensors | — | 7 (× u,v,w) | `obs.{x,y,z}_points` |
| Temporal intervals | — | 9 (20 s mean) | `obs.interval_seconds` |
| Observation vector | `N_d` | 189 | derived |
| Obs error std | `σ` | 0.25 | `esmda.obs_error_std` |
| Trunc. correlation | `ρ_t` | `3/√N_e ≈ 0.53` (default) | `localization.truncation_correlation` |
| Tapering fraction | `β` | 0.5 | `localization.tapering_beta` |
| Max inflation | `E_max` | 8 | `localization.max_inflation` |
| Block grouping | — | false (default) | `localization.block_grouping` |
| Domain grid | — | 50 × 40 × 8 | `domain.nx/ny/nz` |
| Domain bounds | — | x∈[-20,80], y∈[0,80], z∈[0,32] m | `domain.bounds` |
| Truth seed / mean | — | 7 / (25°, 6 m/s), 18 knots | `params/dynamic_truth` |
| Prior seed | — | 2 | `params/dynamic` |

---

## 11. Glossary

| Symbol | Meaning |
|---|---|
| `m` | parameter vector (the flattened knots being estimated) |
| `d` | observation vector (length `N_d = 189`) |
| `g(m)` | forward map: params → predicted observations (pylbm + obs operator) |
| `N_e` | ensemble size (32) |
| `Na` | number of ES-MDA iterations per window (3) |
| `α` | per-iteration error-inflation factor (`Σ 1/α_i = 1`) |
| `C_D` | observation-error covariance (`σ² I`) |
| `C_MD` | param–predicted-obs cross-covariance |
| `C_DD` | predicted-obs covariance |
| `ρ(l,j)` | ensemble correlation between augmented row `l` and observation `j` |
| `ρ_t` | correlation truncation threshold |
| `E_inf` | localization inflation factor (1 keep, >1 taper, ∞ exclude) |
| `z(t)` | unit-variance AR(2) anomaly |
| `x_ext, Σ_ext` | external (relaxation-target) mean and spread of a parameter |
| `ℓ` | AR(2) correlation length (100 s) |
| `α(t)` | cross-window relaxation weight `e^{-(t-t₀)/ℓ}` |

---

### Primary references
- Emerick & Reynolds (2013) — Ensemble Smoother with Multiple Data Assimilation.
- Vossepoel et al. (2025), MWR-D-24-0269.1 — adaptive correlation localization.
- Evensen (2024) — AR(2) inflow-forcing priors for LES data assimilation.
