# Routes to better held-out state estimates in `pyurbanair`

Review of the ISDA campaign (`experiments_report/`, runs `E*`/`F*`/`H*` under
`presentations/isda_new/experiments/`) against the code. Read-only; no runs performed.
All line references are to the tree as of this review.

---

## 0. Six load-bearing findings from the code (these drive the ranking)

**F1 — The filter's error model contains no representativeness term, and this is the
single best predictor of who generalises.**
Truth and members are *different realisations of the same turbulent flow*, but the
observation error is pure instrument noise. The filter perturbs raw 2 s frames with
`obs_error_std = 0.1` and uses `C_D = 0.01·I` — exactly consistent, i.e. zero allowance
(`conf/run_filtering.yaml:148`). ESMDA perturbs *raw* frames and then assimilates 15 s
**means** while keeping `C_D = diag(0.1²)` on the aggregated vector
(`scripts/esmda/run_esmda.py:801`, `:877-884`): the true noise on a mean of 8 frames is
`0.1/√8 ≈ 0.035`, so ESMDA's R is **8× too large in variance** — deliberately, per the
comment at `run_esmda.py:877-882`. That accidental allowance is the only
representativeness term anywhere in the system, and ESMDA is the only method whose
held-out error is *not* worse than its fitted error (`valid/assim` 0.89–1.48, and <1 on
both `inflow_turb` cells, vs 2.2–4.9 for every filter variant —
`experiments_report/sections/comparison_tables.tex`, `\CmpGapTable`). The runs are
already flagged `underfit_final: true, caveat: no_representativeness_error`
(`experiments_report/sections/esmda.tex`).

**F2 — The filter's parameter random walk is unit-blind and mis-scaled by ~40×.**
`conf/filtering/evolution/random_walk.yaml` ships a *scalar* `std: 0.5`, and
`RandomWalkEvolution._std_for` returns that same scalar for every parameter name
(`libs/data-assimilation/src/data_assimilation/filtering/parameter_evolution.py:70-73`).
The F11 run config confirms `std: 0.5` was used. A cycle is `time.output_frequency = 2 s`,
so 180 cycles over the 360 s horizon: free diffusion of `velocity_magnitude` is
`0.5·√180 ≈ 6.7 m/s` against a prior std of 0.5 m/s, while the truth's own |U| drift is at
most `1.0·2π/150 · 2 s ≈ 0.084 m/s` per cycle. This is a complete explanation for three
reported failures at once: the joint EnKF *never* improves |U| (F7 0.910, F9 0.912,
F13 0.818, F16 0.955 against a 0.919 prior); it diverges to 4.22 m/s on periodic (F11) and
21.3 m/s under PALM truth (F3); and marginals show **members below zero**
(`experiments_report/sections/filtering.tex`, Fig. `filt-marg-F11`). The angle
(0.5°/cycle → 6.7° over the run) is coincidentally in the right range, which is why the
angle *is* learned and |U| is not.

**F3 — In the periodic case the inflow parameters DO force the flow — but only the
horizontal slab mean above 16 m.**
`use_nudging` is computed and then unconditionally overwritten with `True`
(`libs/pyudales/src/pyudales/forward_model.py:728-732`), so nudging is applied under
`boundary_condition: periodic` too. The solver's `nudge` relaxes only the *slab mean*:
`up(:,:,k) -= (u0av(k) - uprof(k))/tnudge` for `k = kb+nnudge .. ke`
(`libs/pyudales/u-dales/src/modforces.f90:849-883`). With `tnudge: 15.0`,
`nnudge_meters: 16.0` (`conf/model/pyudales.yaml`) and dz = 2 m on the 32 m domain, that
is `nnudge = 8` levels — **only z > 16 m is forced**, i.e. just above the tallest roofs.
The body force is zeroed for every BC: `lscale.inp` is written with
`dpdx_profile = dpdy_profile = zeros` unconditionally
(`libs/pyudales/src/pyudales/utils/nudging_utils.py:384-390`) and the case template has
`&INPS dpdx = 0.0` (`examples/udales/xie_and_castro/namoptions.300`). So under periodic,
`{inflow_angle, velocity_magnitude, vertical_inflow_exponent}` control exactly **two
numbers per vertical level of a horizontally averaged profile in the top half of the
domain**, on a 15 s relaxation. Nothing about horizontal structure. The 6 sensors sit at
z = 2 m, 14 m below the lowest forced level. That is the mechanism behind
"unidentifiable" — the parameters are not *inert*, they are *nearly orthogonal to what the
sensors see*, and the estimator has no way to tell the two apart.

**F4 — Truth and every member share the same initial random field; the only genuine
realisation spread comes from the turbulent inlet.**
uDALES defaults `lrandomize = .true.` (`modglobal.f90:237`) with a hard-coded
`irandom = 43, randu = 0.01` (`modstartup.f90:42-44`), and `pyudales` never writes
`irandom` (no occurrence anywhere in `libs/pyudales/src/`). `create_initial_state_ensemble`
copies the *same* state into every member
(`src/pyurbanair/config/hydra_helpers.py:158-164`). So on `inflow` and `periodic` the prior
ensemble is a **pure two-parameter family plus chaotic divergence** — its covariance is a
parameter-sensitivity covariance, not a model-error covariance, and it structurally cannot
represent the truth-vs-ensemble realisation mismatch. Only `inflow_turb` has physically
generated spread, via `derive_seed(experiment_name)`
(`libs/pyudales/src/pyudales/utils/inlet_turbulence_utils.py:319`, `:854-855`) — and that
is the case where ESMDA generalises best (valid 0.365 < assim 0.401, E12).

**F5 — The ensemble is *calibrated* at the held-out sensors and still wrong.**
Validation-sensor z-score std: F13 0.88 (laminar), F9 0.99 (turbulent), F12 1.22
(periodic) — all ≈ 1 (`\CmpMasterUDALES`). Spread ≈ error means the residual held-out
error is **not** under-dispersion and **not** a tuning failure: the ensemble correctly
reports that the information is absent from the 6 sensors. Any route that only re-weights
the existing 50 anomalies (localization radii, inflation constants, reduction ranks) is
therefore capped. Routes that add *information* (sensors, statistics, honest R, a new
control variable) are not.

**F6 — Capabilities that exist and were never used, and capabilities that do not exist.**
Unused-but-implemented: `StateESMDA` / `StateAndParameterESMDA` /
`StateAndTimeVaryingParameterESMDA` (`smoothing/esmda.py:680`, `:1054`;
`conf/esmda/smoother/state_and_dynamic.yaml`) with `final_time_smoothing`
(`smoothing/esmda.py:289`, `:996-1042`) — the campaign ran
`joint_state_and_parameter = False` everywhere; `AggregateObservations`
(`observation_operator.py:258-386`) is wired only into the ESMDA path; the per-parameter
mapping form of the random walk (`parameter_evolution.py:60-73`).
**Absent entirely** (grep over `libs/data-assimilation/src`): adaptive inflation, additive
/ climatological inflation, hybrid gain or any static covariance, Gaussian anamorphosis or
bounded parameter transforms, a bias / model-error state, IAU or nudging, fixed-lag state
smoothing, sampling-error-corrected localization. `inflation.py` holds only
Multiplicative / RTPS / RTPP; `localization/` only distance + correlation.

---

## A. Fixing the problem formulation

### A1 (rank 1) — Split the observation error: instrument + representativeness
**Hypothesis addressed.** `valid/assim = 2.2–4.9` for every filter (F5, `\CmpGapTable`) is
the classic signature of fitting noise: the filter drives the six assimilated sensors to
0.15–0.32 m/s against `σ_obs = 0.1`, while the held-out sensors sit at 0.32–1.55. Because
truth and ensemble are different realisations (F4), the instantaneous point mismatch has a
large component that is *not* observation noise and *not* estimable state error. ESMDA's
accidental 8× R inflation (F1) is the only reason it does not do the same thing.

**Method.** `R = σ_inst² I + R_repr`, with `R_repr` estimated two ways and cross-checked:
(i) directly from the realisation-floor ensemble of A2 (`R_repr = cov` of the
same-parameter, different-seed sensor differences — the honest definition here); (ii) from
Desroziers et al. (2005) consistency statistics, `E[d_a^o d_b^oᵀ] ≈ R`, computable per
cycle from quantities the filter already stores (`_record_pred_obs` /
`_record_pred_obs_post`, `filtering/base.py:476-509`).

**What it changes.** `conf/run_filtering.yaml:148` becomes a matrix/spec; `C_D`
construction in `filtering/base.py::_analysis_cycle`; a new `conf/filtering/obs_error`
group; the same for `scripts/esmda/run_esmda.py:801`.
**Effort.** Low (1–2 days), plus A2 to calibrate it.
**Expected.** laminar: small (no realisation mismatch, F4). turbulent: moderate — the
filter should stop over-fitting and land nearer ESMDA's 0.365. periodic: **large** — this
is where `assim = 0.315` vs `valid = 1.261` is most extreme.
**Shows it worked.** `valid/assim → ~1` with held-out RMSE *lower* than today, at
χ² ≈ 1. If held-out RMSE does not move while `assim` degrades, the error was never
over-fitting and A2's floor is binding — that is also a result.

### A2 (rank 2) — Measure the realisation-error floor and decompose the held-out error
**Hypothesis addressed.** Nobody in the campaign knows how much of the held-out RMSE is
achievable. There is *no free-running / no-DA control* in the metric set at all
(`run_summary.yaml` has only `sensor_metrics.{assimilation,validation}`), so "held-out ≈
prior" is asserted, never measured.

**Method.** Three cheap forward-only ensembles, no DA:
1. **Realisation floor** `σ_r`: 10–20 members with parameters *pinned at the truth
   trajectory*, varying only the inlet seed
   (`inlet_turbulence.seed`, already a config key). RMSE of member-vs-truth at the 4
   validation sensors is the floor no parameter-only method can beat. For `periodic` and
   `inflow`, vary `&RUN irandom` instead — a ~10-line write in
   `ForwardModel._apply_inflow_settings` (`forward_model.py:701-782`); today it is
   hard-coded to 43 for everyone (F4).
2. **Parameter-signal amplitude** `σ_θ`: 5–7 members sweeping `inflow_angle` across the
   prior at *fixed* seed. `σ_θ/σ_r` at the sensors is the identifiability signal-to-noise;
   below ~1 the parameter is not estimable, whatever the estimator.
3. **Prior baseline**: score the untouched prior ensemble at the validation sensors and
   persist it as `sensor_metrics.prior.*`.

**What it changes.** A script under `scripts/tools/`; one `irandom` write in `pyudales`;
one extra block in `scripts/*/compute_*_metrics.py`.
**Effort.** Low (2–3 days incl. compute).
**Expected.** No accuracy gain — it is the measuring stick every other route needs, and it
answers the two questions posed in the brief quantitatively.
**Shows it worked.** Three numbers per case: `σ_r`, `σ_θ`, prior RMSE. See §E for what I
can already infer without them.

### A3 (rank 5) — Re-pose the periodic control: estimate the *forcing*, not "inflow angle"
**Hypothesis addressed.** F3: the parameters control a slab-mean profile above 16 m via a
15 s relaxation, which is a poor, indirect, and nearly unobservable control. All three
methods land at 13.6–19.0° against a ~16.6° prior; the hybrid fits the aggregated
observations to **0.053 against σ_obs = 0.1** while the parameters move *away* from truth
(`experiments_report/sections/filter_smoothing.tex`, takeaway 3) — textbook overfitting of
an unidentifiable control.

**Method.** Replace the borrowed inflow parameterisation with a control that is native to a
periodic LES, in increasing order of well-posedness:
1. **Constant body force / large-scale pressure gradient.** Estimate
   `(dpdx, dpdy) = |∇p|·(cos φ, sin φ)`. `angle_to_pressure_gradient` already exists
   (`libs/pyudales/src/pyudales/utils/inflow_utils.py:28-74`) and
   `pressure_gradient_magnitude` is already in the sampler
   (`conf/params/static.yaml`) — both are **dead code today** because `lscale.inp` is
   written with zeros (`nudging_utils.py:384-390`) and `params_to_estimate` drops the
   parameter. Two numbers, applied uniformly (`modforces.f90:85-86`,
   `dpdxl(k) = -pgx(k) - dpdx`, `modstartup.f90:2100`).
2. **Fixed volume-flow-rate forcing.** uDALES has `luvolflowr` / `lvvolflowr`
   (`modforces.f90:414-440`), which adjusts a uniform body force so the *domain-mean*
   velocity matches `uflowrate`. Estimating the bulk velocity vector is the cleanest
   possible periodic control: it *is* the domain-mean momentum, i.e. the leading term of
   what a canopy sensor measures.
3. **The nudging target profile itself**, as 3–5 EOF/knot coefficients of `U(z), V(z)`
   plus `tnudge` and `nnudge`, instead of `{|U|, φ, α}`. Higher-dimensional but honest.
Also worth testing: lower `nnudge_meters` so the nudged layer reaches nearer the canopy,
or nudge the domain-mean momentum only.

**What it changes.** `libs/pyudales/src/pyudales/utils/nudging_utils.py` (stop zeroing the
body force under periodic), `forward_model.py:718-782`, `conf/model/pyudales.yaml`,
`conf/params/*`, `params_to_estimate`.
**Effort.** Medium (2–4 days for option 1 or 2).
**Expected.** periodic only, but potentially decisive: it converts an unidentifiable
control into an identifiable one, which is a precondition for any parameter-borne state
improvement and removes the joint-filter divergence mode. laminar/turbulent unaffected.
**Shows it worked.** Periodic parameter RMSE drops below the climatological prior for the
first time (any reduction > ~30 % would be new), the posterior contracts, and the joint
filter stops diverging.
*Speculation flag:* whether the bulk forcing is identifiable from six z = 2 m sensors is
untested; A2 step 2 settles it before any implementation.

### A4 (rank 8) — Estimate the inlet realisation, not just the inlet statistics
**Hypothesis addressed.** In `inflow_turb` the parameters cannot represent the specific
eddy field entering the domain. The posterior absorbs it as bias (`z_par = 4.6` for E12,
`z_val = 2.71` — over-confident) and held-out error saturates at ~0.36–0.44 for every
method. This is the residual A1 makes honest but cannot remove.

**Method.** Three tiers.
1. *Cheap, do first:* estimate the inlet **statistics** — `intensity`, `length_scale_y/z/x`
   — which are identifiable from sensor variance and spectra but not from instantaneous
   values, so pair with C2. All four are already config keys
   (`conf/model/pyudales.yaml`, `inlet_turbulence`).
2. *The real move:* augment the state with the **driver-plane latent field**. The
   generator is an AR(1) recursion `b_n = a·b_{n-1} + √(1-a²)·η_n` with
   `a = exp(-π·dtdriver/(2T))` over a spatially filtered `(ktot+2)×(jtot+2)` plane
   (`inlet_turbulence_utils.py`, `_record_noise:425-442`, `generate_fluctuations:448-490`).
   That is a bona-fide state with a *known linear Gaussian forecast model* — the textbook
   augmented-state case. Project it onto ~20–50 y–z Fourier/POD modes and put those
   coefficients in the augmented vector. Requires making the generator accept an injected
   latent state rather than always replaying from the name-derived seed (the seed-replay
   design at `inlet_turbulence_utils.py:456-482` is exactly what has to become optional).
3. Combine with C1: one roof-level sensor near the inlet makes the plane partially
   observable at all (Sousa & Gorlé 2019: roof-level sensors identify inflow far better
   than in-canopy ones).

**What it changes.** `libs/pyudales/src/pyudales/utils/inlet_turbulence_utils.py`,
`data_assimilation/augmentation.py` (a third augmentation block alongside
`ParamAugmentation` / `StateAugmentation`), `filtering/base.py` mode handling.
**Effort.** Tier 1 low; tier 2 high (1–2 weeks, and it is genuinely new).
**Expected.** turbulent only, and it is the *only* route in this list that can beat the
realisation floor there. laminar/periodic: n/a (no inlet).
**Shows it worked.** Held-out RMSE on `inflow_turb` below the `σ_r` measured in A2.

### A5 (rank 11) — Model-error parameters and an additive bias term
**Hypothesis addressed.** Under PALM truth every method compresses to valid 1.30–1.43
(turbulent) / 1.75–2.13 (periodic) and every posterior becomes over-confident — no
parameter in the current vector can absorb structural bias. Note that
`params_to_estimate: [inflow_angle, velocity_magnitude]` *filters the sampler itself*
(`filter_parameter_config`, `src/pyurbanair/config/hydra_helpers.py:129-155`), so
`sgs_constant` and `vertical_inflow_exponent` are absent from prior **and** truth: the
model-error-parameter machinery documented in
`docs/archive/esmda_model_error_parameters.md` is currently **inert** in every ISDA run.
**Method.** (i) Re-enable `sgs_constant` — auto-memory records that `c_vreman` sits near
the stability floor at 0.24 (0.07/0.15 diverge), so it strongly controls canopy mixing and
is a plausible bias absorber; (ii) add wall roughness / `z0` as a second knob;
(iii) weak-constraint: a low-dimensional **additive bias field** (a handful of vertical or
POD modes) estimated as parameters and added as a body force — the standard
model-error/weak-constraint formulation, and the only thing that can absorb what no
physical parameter can.
**What it changes.** `conf/params/static.yaml`, `params_to_estimate`, a new forcing hook in
`pyudales`. **Effort.** (i)-(ii) low; (iii) medium-high.
**Expected.** PALM-truth cells only; roughly neutral on matched twins (and a risk of extra
spurious DOF at N_e = 50, hence the low rank here).

---

## B. Algorithmic changes in the DA

### B1 (rank 3) — Fix the parameter forecast model: per-parameter scale, mean reversion, bounds
**Hypothesis addressed.** F2. This is the highest-value/lowest-effort item in the whole
list.
**Method.**
1. Use the *mapping* form the class already supports
   (`parameter_evolution.py:60-73`): `std = {inflow_angle: 0.4, velocity_magnitude: 0.02}`,
   set from `max|dθ/dt|·Δt_cycle` of the truth process.
2. Replace the pure random walk with an **Ornstein–Uhlenbeck / AR(1) evolution** that mean-
   reverts to the climatological prior with a correlation time matching the ESMDA AR(2)
   prior's `L_corr = 200 s`: `θ_{k+1} = θ̄ + ρ(θ_k − θ̄) + √(1−ρ²)σ_clim ξ`,
   `ρ = exp(−Δt/L_corr)`. This bounds the forecast prior at the climatological width
   instead of letting it diffuse to 30° (the report's own number for the joint filter's
   forecast prior) — the standard fix for augmented-state parameter estimation.
3. **Gaussian anamorphosis / bounded transform** (Bertino et al. 2003; Simon & Bertino
   2009): estimate `log|U|` rather than `|U|`, so the unbounded Gaussian update cannot
   produce negative speeds. A transform hook belongs in `augmentation.py`
   (`ParamAugmentation.flatten/unflatten:72-135`) and would apply to smoother and filter
   alike.
**What it changes.** `filtering/parameter_evolution.py` (add `OUEvolution`), a new
`conf/filtering/evolution/ou.yaml`, transform hooks in `augmentation.py`.
**Effort.** Trivial for (1) — a config string. 2–3 days for (2)+(3).
**Expected.** laminar and turbulent: the joint filter finally learns |U| (target ≤ 0.4 m/s
against the 0.919 prior it currently reproduces exactly). periodic: removes the divergence
(F11 4.22 m/s, F3 21.3 m/s) and, given B1 alone, the joint filter should become at worst
equal to state-only rather than 23–55 % worse.
**Shows it worked.** Any `F*` run with |U| RMSE meaningfully below 0.919; no member below
zero in `marginals_F11.png`; joint ≥ state-only on periodic held-out RMSE.

### B2 (rank 6) — Run the joint state+parameter ESMDA with final-time smoothing
**Hypothesis addressed.** The two working methods fail in complementary ways: ESMDA
generalises (`valid/assim ≈ 1`) but leaves the state at climatology on periodic and drifts
out of phase on laminar; the filter fixes the state locally and does not generalise. A
smoother that updates a *window-consistent trajectory* is the 4-D update that neither
currently performs — the ESMDA analogue of 4D-LETKF (Hunt et al. 2007) / IEnKS (Bocquet &
Sakov 2014), without an adjoint.
**Method.** `esmda/smoother=state_and_dynamic` + `esmda/state_reduction=svd` +
`final_time_smoothing: true`. The last implements exactly the Evensen et al. (2024, MWR
10.1175/MWR-D-23-0239.1) recommendation — update the window-*end* state that warm-starts
the next window rather than only the IC, which also saves one ensemble pass
(`smoothing/esmda.py:996-1042`; note the guard: it requires `state_reduction` and
in-memory mode, `:716-725`). This is **already implemented and was never run in the
campaign** (`joint_state_and_parameter = False` in every `run_summary.yaml`).
**What it changes.** Configuration only. Effort: near zero to run; a few days to interpret.
**Expected.** laminar: fixes the phase drift that pins E11 at valid 0.748. periodic: gives
the smoother a state to move — the honest target is beating the best filter's 1.246 while
keeping ESMDA's ~1.0–1.3 valid/assim ratio. turbulent: neutral to modest.
**Shows it worked.** Periodic ESMDA valid < 1.25 at `valid/assim < 1.5`.

### B3 (rank 9) — Sampling-error-corrected and scale-aware localization
**Hypothesis addressed.** The correlation localization is a hand-tuned threshold
(`truncation_correlation: 0.35` vs the theoretical `3/√50 = 0.42`,
`localization/correlation.py:78-83`, `conf/filtering/localization/correlation.yaml`), and
its effect is strongly regime-dependent: −47 to −69 % on periodic, **+16/+23 % (harmful)**
on laminar. That is the signature of a mistuned estimator, not of a physical result. At
N_e = 50 sample correlations are wrong *at all separations* (Ying, Zhang & Anderson 2018).
**Method.** (i) **Anderson (2012) sampling-error correction**: precompute the
`E[ρ_true | ρ_sample]` table for N_e = 50 offline once and multiply the gain by it — this
removes the free threshold entirely and drops straight into the existing
`BaseLocalization.inflation_factors` / `taper_inflation` contract
(`localization/base.py:354`, `:368`). (ii) **Scale-dependent localization**
(Buehner & Shlyaeva 2015): low-pass the state increment (coarse-grid or spectral) before
applying it — only large scales are estimable from 6 sensors hundreds of grid cells apart,
and this is precisely the mechanism behind the 2.2–4.9 `valid/assim` gap. ~10 lines at the
analysis site in `filtering/base.py::_analysis_cycle`. (iii) Localize *parameters*
separately from state rows — today correlation localization is applied to every row
including parameters, which is why the "best" unlocalized periodic angle comes from a
collapsed ensemble (`z_pool = 41`, the report calls this out itself).
**Effort.** (i) medium; (ii) low; (iii) low.
**Expected.** periodic and turbulent; laminar should *stop being hurt*.
**Shows it worked.** One localization setting that is never harmful across the three cases,
with the `valid/assim` ratio down at equal `assim`.

### B4 (rank 10) — Adaptive inflation, additive/climatological inflation, hybrid covariance
**Hypothesis addressed.** RTPS `α = 0.6` is a single constant serving six regimes; the
report shows it buys calibration and *costs* generalisation (`valid/assim` F14 2.41 → F15
3.44: "inflation buys calibration at the sensors without adding any information away from
them"). Multiplicative relaxation is the wrong tool when model error, not sampling error,
dominates (Whitaker & Hamill 2012).
**Method.** (i) Adaptive inflation (Anderson 2009; El Gharamti 2018) — a scalar or
per-observation inflation with its own Bayesian update, replacing the fixed α;
(ii) **additive inflation from a climatological ensemble**: draw increments from a library
of differences between spun-up trajectories (the surrogate-training data already exists),
which injects directions the 50 members cannot span; (iii) **hybrid gain / hybrid
covariance** (Houtekamer et al. 2019): `P = β P_ens + (1−β) P_clim` with `P_clim` a fixed
offline POD basis of rank 100–1000 — this slots into the existing
`state_reduction` seat (`reduction.py`), which today fits its basis to the *same 50
anomalies* and so cannot add information (see `docs/temp/da_review_state_estimation.md`
§1.1).
**What it changes.** `data_assimilation/inflation.py` (currently only Multiplicative /
RTPS / RTPP), `reduction.py`, `conf/filtering/inflation/*`.
**Effort.** (i) medium, (ii) medium, (iii) medium-high.
**Expected.** Mostly the PALM model-error cells and periodic-without-localization. On
matched twins F5 says the return is capped — hence the rank.

---

## C. Observation and experiment design

### C1 (rank 4) — Move sensors to where the estimable signal is, chosen by an observability criterion
**Hypothesis addressed.** All six assimilated sensors are at z = 2 m inside canyons
(`conf/case/xie_and_castro.yaml`), 14 m below the *only* forced layer in the periodic case
(F3) and deep inside the canopy in the inflow cases. The observability literature is
unambiguous that reconstruction skill decays sharply away from the sensed surface and that
roof-level sensors identify inflow far better than in-canopy ones (Sousa & Gorlé 2019).
The one validation sensor at z = 20 m is tracked by *all* methods
(`comparison.tex`, Fig. `cmp:valid-ud-turb`) — evidence that the above-canopy layer is the
part of the flow the sensors *can* constrain.
**Method.** (i) Use the ensemble itself: `CorrelationLocalization.inflation_factors`
already forms `ρ(row, obs)` from ensemble anomalies
(`localization/correlation.py:88-108`). Reuse it to score every candidate location by its
correlation with (a) the parameters and (b) the held-out targets, then pick the network by
greedy A-optimality / ensemble mutual information. (ii) Re-run all three cases with 2 of
the 6 sensors moved to z ≈ 20–25 m and, for `inflow_turb`, one placed just downstream of
the inlet. (iii) Consider a different observable — a passive scalar released at a known
source is far more informative about transport than a point velocity, and uDALES carries
scalars natively.
**Effort.** Low-medium (the placement study is a script; the re-runs are the cost).
**Expected.** turbulent: large. periodic: **large** — a roof-level pair observes the nudged
slab mean *directly*, which is the only quantity the parameters control. laminar: small.
**Shows it worked.** Periodic parameter RMSE below the 19.29° prior for the first time;
held-out RMSE down in all three cases at unchanged algorithm.

### C2 (rank 7) — Assimilate statistics, not instantaneous velocities
**Hypothesis addressed.** F4/F5: truth and ensemble are decorrelated realisations, so the
instantaneous point value is largely uninformative while its 30–60 s mean and its variance
are not. The campaign already contains supporting evidence: in the ESMDA interval sweep
7.5 s is the **worst** interval on both accuracy and calibration and 15–30 s the best
(`\CmpObsIntervalTable`), which is the wrong way round for an information-count argument
and the right way round for a representativeness argument.
**Method.** Route the filter's observations through the existing `AggregateObservations`
(`observation_operator.py:258-386`, currently wired only into the ESMDA path), and add a
second-moment mode so the assimilated vector is `(mean, rms)` per sensor per window. Pair
with A1 so R matches the aggregation. Deeper version: target statistics as the *estimand*
(the statistics-nudging / moment-matching line), which is also the right framing if the
end use is the urban wind climate rather than an instantaneous field.
**What it changes.** A new `conf/filtering/observations` group; `_cycle_observations`
(`filtering/base.py:629`); `AggregateObservations.__init__` mode list (`:268`).
**Effort.** Low (the aggregator exists).
**Expected.** All three cases; largest on periodic, where the instantaneous signal is
essentially all realisation noise.
**Shows it worked.** Held-out *interval-mean* RMSE improves while instantaneous RMSE stays
flat — which is the honest success criterion, not a failure.

### C3 (rank 12) — Experiment hygiene: a control run, a longer horizon, replicate seeds
**Hypothesis addressed.** Several conclusions are currently unfalsifiable.
**Method.** (i) The no-DA baseline of A2 step 3 — without it "held-out ≈ prior" cannot be
checked. (ii) A longer horizon: the campaign is 150 s spin-up + 3×120 s, and the report
itself notes the periodic truth "starts from rest, so the first ≈ 40–50 s carries no
assimilation information" — a large fraction of a 360 s experiment is spin-up transient,
not the statistically steady regime the conclusions are stated about. (iii) ≥3 assimilation
seeds on the headline cells; the ISDA plan asked for this
(`docs/plans/isda2026_talk_experiments.md` §3) and the campaign is single-seed, so
differences of 10–20 % between methods are not yet distinguishable from noise.
(iv) Vary `irandom` per member (A2) so that the laminar and periodic ensembles have any
realisation spread at all (F4).
**Effort.** Low code, real compute.
**Expected.** No accuracy gain; it is what makes every other route interpretable.

---

## D. Diagnostics to run before any of the above (except A2, which is one of them)

Cheap, mostly reusing stored quantities; each discriminates between the failure modes.

1. **Desroziers consistency per cycle** — free: `_record_pred_obs` /
   `_record_pred_obs_post` already store forecast and analysis predicted observations
   (`filtering/base.py:476-509`). Gives `R̂`, `HPHᵀ`, and hence A1's `R_repr` directly.
2. **Analysis-increment energy spectrum vs the flow's spectrum.** If increment energy at
   high wavenumbers is comparable to the flow's, the state update is injecting unbalanced
   small scales — the mechanism behind the `valid/assim` gap, and the trigger for B3(ii).
3. **Identifiability signal-to-noise `σ_θ/σ_r`** per case (A2 steps 1–2). This is the
   *decisive* number for the periodic question and costs ~25 forward runs.
4. **Spread–skill per cycle in obs and state space**, separately for assimilated and
   held-out sensors. (Partly present as the z-score columns; F5 already reads it.)
5. **IC informativeness horizon**: correlate the window-IC field against each interval's
   observations in the truth run. If it dies within 1–2 intervals, the smoother's IC update
   is fitting noise over most of the window.
6. **Ensemble-size probe** at N_e = 100–200 on one cheap case. If held-out skill barely
   moves, the binding constraint is observability (C1) and not sampling (B3/B4) — this
   redirects effort decisively and is worth doing early.

---

## E. Answers to the two specific questions

### The periodic case: what do the inflow parameters actually do?
They are **not** inert. `use_nudging` is hard-coded `True`
(`forward_model.py:728-732`), so under `boundary_condition: periodic` the parameters are
written into `timedepnudge.inp` and drive `modforces.f90::nudge`, which relaxes the
**horizontal slab mean** `u0av(k), v0av(k)` toward `|U|·(z/z_ref)^α·(cos φ, sin φ)` at
`1/tnudge = 1/15 s⁻¹`, over `k = kb+nnudge .. ke` only. With `nnudge_meters: 16.0` and
dz = 2 m that is **z > 16 m** — the top half of a 32 m domain, just above the tallest
roofs. The body force is zeroed for every BC (`nudging_utils.py:384-390`;
`&INPS dpdx = 0.0` in the case template), so the nudging is the *only* momentum source.

So the estimation problem as posed is: *infer two scalars that set the horizontally
averaged wind vector above roof level, from six instantaneous point velocities at z = 2 m
inside canyons, in a flow whose realisation is uncorrelated with the truth's.* The
parameters are nearly orthogonal to the observed quantity, and the estimator cannot
distinguish "no signal" from "noise" — which is exactly what the numbers show: 13.6–19.0°
against a 16.6° prior, posteriors that barely contract, and a hybrid that fits the
aggregated observations to 0.053 (half the noise floor) while moving *away* from truth.

**Well-posed targets, best first.**
1. **Domain-mean momentum** (bulk velocity vector) via uDALES's fixed-volume-flow-rate
   forcing, `luvolflowr` / `lvvolflowr` (`modforces.f90:414-440`). Two numbers, directly
   the leading term of what a canopy sensor measures, and the standard way to drive a
   periodic urban LES.
2. **Large-scale pressure gradient** `(dpdx, dpdy)` as a uniform body force
   (`modforces.f90:85-86`). `angle_to_pressure_gradient` and the
   `pressure_gradient_magnitude` prior already exist and are dead code
   (`inflow_utils.py:28-74`, `conf/params/static.yaml`).
3. **The nudging target profile** as 3–5 EOF coefficients of `U(z), V(z)`, plus `tnudge`
   and `nnudge` as tunables rather than fixed constants.
4. **A low-dimensional field of the state itself** — the canopy-layer, interval-mean, large
   scale — which is what the observability analysis says is estimable, and what C2's
   statistics-observations naturally target.
Note that (1)–(3) will *still* not fix held-out instantaneous RMSE by much: F5 shows the
periodic ensemble is calibrated at the held-out sensors, so the residual is genuine
unobservability. The gain from A3 is that the estimator stops confidently reporting
nonsense and stops diverging (F11 |U| 4.22 m/s), which is a prerequisite for anything else.

### The turbulent-inflow case: what fraction of held-out variance is the unobserved inlet realisation?
**It has not been measured, and cannot be computed from the stored artifacts** — there is
no pinned-parameter/varied-seed ensemble and no free-run baseline in the metric set. A2
step 1 is the two-day experiment that answers it. What the existing numbers support:

*Order-of-magnitude estimate (speculation, flagged as such).* The generator injects
`u'_rms = intensity·|U| = 0.05·7.5 = 0.375 m/s` with `v', w' = 0.7×` that
(`conf/model/pyudales.yaml`; `docs/pyudales.md` §6.1), i.e. a vector rms of ≈ 0.53 m/s at
the inlet plane. Two *independent* realisations differ by `√2 ×` that ≈ 0.75 m/s at the
inlet — already larger than the best held-out vector RMSE anyone achieved on that case
(0.365, E12). The in-canopy attenuation is unknown (and partly offset by wake-generated
turbulence), so this is an upper bound, not the floor.

*Bracketing from the campaign itself.* `inflow` (laminar) has **zero** realisation
mismatch by construction (F4: fixed `irandom`, no inlet turbulence), and its best held-out
RMSE is 0.319 (F13). `inflow_turb` adds an independent inlet realisation and its best is
0.365 (E12). The 0.32 → 0.37 gap is a crude bracket on the realisation contribution — but
it is confounded, because the two numbers come from different methods and the laminar case
has its own defect (the smoother stalls on phase drift). Read together with F5 (the
turbulent ensemble is calibrated at held-out sensors, `z_val = 0.99` for F9), the honest
reading is: **the turbulent case is plausibly already close to its realisation floor for
any method that does not estimate the inlet field**, and the remaining headroom is in
calibration (E12 is over-confident, `z_par = 4.6`) rather than in RMSE.

**How to estimate the realisation itself**, in order of cost:
1. A sensor near the inlet / above roof level (C1) — makes the plane partially observable
   at all; cheapest by a wide margin.
2. Estimate the **SEM/digital-filter parameters** (`intensity`, `length_scale_{x,y,z}`)
   from sensor *statistics* (C2) — low effort, and it fixes the ensemble's second moment
   even if it cannot fix the realisation.
3. Augment the state with the **driver-plane AR(1) latent field**, projected onto ~20–50
   y–z modes (A4 tier 2). It has a known linear Gaussian forecast model
   (`inlet_turbulence_utils.py:456-482`), which makes it an unusually clean augmented-state
   problem, and it is the only route that can beat the floor.

---

## F. What I would drop or simplify

* **The filtering state-reduction ladder** (`reduction.py`, 734 lines, plus
  `conf/filtering/state_reduction/` and `conf/esmda/state_reduction/`). Its basis is fitted
  to the same 50 anomalies, so it changes conditioning and cost but not information
  content — `docs/temp/da_review_state_estimation.md` §1.1 says so, and at N_e = 50 > N_d = 12
  the SVD costs more than the update it replaces. **Keep the seat, change the occupant**:
  it is the natural home for a fixed, offline, climatological basis (B4 iii).
* **`etkf_tsvd` and `letkf_tsvd`.** With N_d ≈ 12 there is nothing to regularise; the
  config comments say so themselves. Two of four analysis variants, unbenchmarked and
  unused. Delete or fold into the untruncated variants behind a flag.
* **The scalar default of `RandomWalkEvolution`.** A unit-blind scalar applied across
  parameters with different physical dimensions is a live foot-gun that demonstrably
  produced two published failure modes (F2). Make the mapping form mandatory.
* **The hybrid's observation-interval sweep.** The report already calls it "a placebo" —
  it changes only the smoother's aggregation bin while the filter still assimilates every
  2 s. Six runs of no information.
* **The joint EnKF on periodic**, and more generally **reporting `assim` RMSE as a headline
  metric.** It is a fit diagnostic that rewards exactly the over-fitting mode this review
  identifies; keep it in an appendix next to `valid/assim`.
* **Per-case localization and inflation sweeps.** Regime-dependence of a hand-tuned
  threshold is a symptom (B3), not a result. Pick one principled scheme
  (sampling-error-corrected localization, adaptive inflation) and stop sweeping.
* **Simplify the method zoo.** Five method variants × 3 cases × 2 localizations × 2 truths
  is 45 cells at 1–3 h each, single-seed, on a 360 s horizon that is largely spin-up. The
  same compute spent on A2 + C1 + C3 (floor, sensors, replicates) on *two* methods would
  produce conclusions that survive a referee.
* Either **use** the model-error parameters (`sgs_constant`, `vertical_inflow_exponent`) or
  stop documenting them as active: `params_to_estimate` filters them out of prior *and*
  truth (`hydra_helpers.py:129-155`), so the machinery in
  `docs/archive/esmda_model_error_parameters.md` is inert in every ISDA run.

---

## G. Suggested sequencing

1. **Week 1** — A2 (realisation floor + parameter-signal + prior baseline) and D1
   (Desroziers). Two numbers per case; everything below is measured against them.
2. **Week 2** — B1 (parameter forecast model; a config string gets most of it) and A1
   (representativeness R). Cheapest large expected gain, and they interact.
3. **Week 3** — C1 (sensor placement study + one re-run per case) and B2 (turn on the
   state-and-parameter ESMDA that already exists). Both are near-free to attempt.
4. **Then, chosen by the diagnostics** — A3 if periodic identifiability is confirmed dead;
   C2 + A4 if the turbulent case is at its realisation floor; B3/B4 only if D6
   (ensemble-size probe) says sampling, not observability, is binding.
