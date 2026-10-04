# Handover: DA review fixes as opt-in options

The September DA review (`docs/research/da_review_2026-09/summary.md`, audits
A–E) lists defects behind good fits at the assimilated sensors but poor skill at
held-out ones. This PR fixes the code defects and adds the missing options.
**Every change is either a bug fix in a path default runs never reach, or a new
option that is a no-op when absent**: default runs stay byte-identical. Choosing
new default values (R, localization, spin-up trimming, hybrid allocation) is
experiment work for the user, not this PR. Read `AGENTS.md` and
`docs/data_assimilation.md` first.

Branch from `main`, open the PR into `main`, don't merge it. The references
below were checked against `main` at 02b582c; re-check each before editing.

The default run, for orientation: dynamic smoother with correlation localization
and 20 s mean aggregation, state-mode filter with RTPS and no parameter
evolution, uDALES `inflow_outflow` with inlet turbulence on (so nudging is off).

## 1. Representativeness error correlated in time (review rank 1)

`observation.error` has `instrument_std`, `representation_std` and
`propagation: propagate_mean` (`configs/assimilation.yaml` ~90–100,
`data_assimilation/observation_error.py`). Only
`representation_time_model: "independent"` is accepted (~197–201), so averaging
over a 20-frame bin shrinks the representativeness variance by 1/20. The review
argues it is persistent in time (audit A §3.14).

Add `representation_time_model: persistent`: the representativeness error is
fully correlated within a bin, so its variance is not divided by the bin count
(and across raw frames it is one shared draw, not independent ones). The code
currently refuses this on purpose ("persistent errors need a calibrated
temporal covariance"); the fully-correlated limit needs no calibration, so
replace that error with this option and keep refusing anything else. Test in
`tests/data_assimilation/test_observation_error.py`.

## 2. Filter parameter evolution (rank 3)

`RandomWalkEvolution` (`filtering/parameter_evolution.py` ~58–70) accepts one
scalar std for every parameter, blind to units (the review's `std: 0.5` on both
angle in degrees and |U| in m/s). The per-name mapping form exists.
- Require the per-name mapping (or scale a scalar by each parameter's prior
  std); document a sensible example in `configs/assimilation.yaml` next to the
  commented-out block (~126–128).
- Apply the evolution at the start of the next forecast, not after the analysis:
  today `evolve()` runs in `_analysis_cycle` after the update
  (`filtering/base.py` ~1485–1489), so the saved `posterior_params` already
  carry the next cycle's noise.
- The spread guard accepts `IdentityEvolution` as if it maintained spread
  (`filtering/base.py` ~509–519, `scripts/utils/inconsistency_check.py`
  ~164–174); treat it like `None`.

Default runs use `mode: state`, which can't use evolution, so none of this
changes them. Tests next to `tests/data_assimilation/test_filtering.py` ~539 and
~1951.

## 3. Knot-wise temporal localization (rank 4)

`block_grouping: true` (`configs/assimilation_settings/localization.yaml` ~19,
~30) puts all time knots of a parameter in one localization block
(`augmentation.py` ~136–153, `smoothing/esmda.py` ~889–892, ~998–1000), and the
block takes the strongest knot's taper (`localization/base.py` ~64–95).
Add an option that keeps the u/v/w state grouping but gives each knot its own
block (e.g. `group_parameter_knots`, default true = today's behaviour), read in
`_time_varying_group_ids`. Tests in `tests/data_assimilation/test_localization.py`
and `test_esmda_smoother.py`.

## 4. Per-member uDALES `irandom`

uDALES hard-codes `irandom = 43`, shared by the truth and every member, so a
laminar or periodic ensemble at fixed parameters has no realisation spread. The
writer `apply_random_initial_condition` (`pyudales/utils/random_utils.py`)
is reachable only through a constructor argument nothing sets, and members are
copies, so they'd share it anyway. The inlet-turbulence seed is already per
member via `derive_seed(experiment_name)` (`inlet_turbulence_utils.py`
~316–326).

Add an opt-in `model.forward_model` key that writes
`irandom = derive_seed(experiment_name) mod 2**31` per member (truth included).
Don't edit `libs/pyudales/u-dales/`. Test like
`tests/pyudales/test_udales_inlet_turbulence.py`: two members' namoptions differ
only in `irandom`.

## 5. Guards (no behaviour change)

- `check_config`: refuse localized state-bearing smoother updates whose
  `(N_aug, N_d, N_d)` array (`localization/base.py` ~363–401) would exceed a
  memory bound; say which knob to change.
- Delete dead code the review lists, if `git grep` confirms nothing uses it: the
  periodic static-inflow `else` branch (`pyudales/forward_model.py` ~796–802),
  and `pressure_gradient_magnitude` in `configs/params/static.yaml` if the
  periodic case zeroes it anyway (`nudging_utils.py` ~378–390). Ask the user
  before deleting a sampled parameter: it changes the prior draw.

## Out of scope (experiments for the user)

New default values for R, aggregation, `block_grouping`, `truth_start_time`,
`hybrid.likelihood_allocation`/`beta`, RTPS α, `truncation_correlation`; roof-level
sensors; periodic forcing (`nnudge_meters`, bulk pressure gradient); the
review's proposed experiments (floor ensemble, perfect-inlet twin, ensemble-size
probe). List these in the PR as the next steps, citing the review's order
(summary §7).

## Done when

- [ ] Items 1–4 added, each a no-op when absent; item 5 guards and deletions
      done (deletions approved by the user).
- [ ] A test per item; default runs byte-identical (show it for one tiny run per
      method: same `metrics.yaml` before and after).
- [ ] `docs/data_assimilation.md` (and `docs/pyudales.md` for item 4) document
      the new options.
- [ ] `tests/data_assimilation`, `tests/pyudales`, `tests/scripts`, `pre-commit`
      pass; the uDALES integration tests pass (item 4 touches solver input).
- [ ] This file moved to `docs/plans/implemented/`.
