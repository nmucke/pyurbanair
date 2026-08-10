# Master plan: ESMDA evaluation — slimmed

> **Status: every work package implemented and merged into
> `esmda-evaluation`** as of 2026-08-07 (slimmed 2026-08-03 after the July
> rollback; see [Rollback](#rollback)), and two branch-wide adversarial review
> rounds have since run over the merged result and had their blockers fixed
> (PR **#116**, merged 2026-08-10, CI green — see [Implementation
> process](#implementation-process)). The branch is complete; the only step
> left is the single reviewed merge into `main`, opened 2026-08-10 as PR
> **#117** — see [Branching model](#branching-model). **Companion to** the
> research document
> [../esmda_turbulence_evaluation.md](../esmda_turbulence_evaluation.md)
> (the *what and why* — metric definitions, formulas, figure conventions).
> This file is the index and status board; the per-phase plans carry the
> implementation detail. The pre-slim plans (14 WPs, full metric survey)
> live in git history before 2026-08-03.

## Branching model

All work integrates through a central branch, **`esmda-evaluation`**, cut
from `main` when the effort starts. Every WP PR targets `esmda-evaluation`,
not `main`; the branch is merged into `main` once at the end (or at an
agreed intermediate milestone, e.g. after phase 1), as a single reviewed
PR. Keep `esmda-evaluation` current by merging `main` into it periodically
— at minimum before starting each phase — so the final merge is small.
CLAUDE.md's "branch first" rule still applies per WP: branch off
`esmda-evaluation`, PR back into it.

## Implementation process

Applies to every work package:

- **Tests land alongside the code, in the same PR.** A WP without its
  tests is incomplete — the Tests section of each phase plan is part of
  the WP's scope, not follow-up work. **All tests must pass (CI green)
  before a PR is merged** into `esmda-evaluation`; judge against the
  current CI baseline on the target branch.
- **Implementation is done by a team of Opus 5 agents** with roles
  appropriate to the WP (e.g. implementer, test author, docs/config;
  scale the team to the WP size — an XS config change does not need a
  full team).
- **Two adversarial review rounds after implementation:**
  1. Launch an adversarial reviewer agent on the finished WP; apply the
     fixes its review demands.
  2. After the fixes, launch a *second* adversarial reviewer agent for a
     fresh sweep; apply its fixes too.
  Reviewers verify claims against the actual code, focusing **first on
  correctness of the implementation**, and additionally flag unnecessary
  complexity and abstraction (the library design rules in invariant 5 are
  the yardstick). Only after both rounds and green CI is the PR ready to
  merge.
- **Two further adversarial rounds over the whole branch, once every WP had
  merged** (2026-08-07), merged as PR **#116** on 2026-08-10 with CI green.
  Not in the original process, which reviews a WP at a
  time; these read the assembled branch. **Round 3** (`be1433e`) was an
  eight-agent sweep: it confirmed every work package in the table below is
  implemented as planned, and found six blockers — all fixed in that one
  commit, each with a regression test verified to fail against the unfixed
  code. **Round 4** (`a843f00`) was a four-agent review of round 3's own
  fixes, which were themselves unreviewed code; five of the six were sound,
  the sixth had over-applied and is corrected. Each finding is recorded in
  the phase plan it touches (phase 1: the skill-score knot sets and the
  mean-field memory bound; phase 2: the stale obs-diagnostics bundle; phase
  3: the probe cadence) or, for the filtering port, in the section below.
- **Four non-blocker findings from those rounds were also fixed** (`bb4aa09`,
  in the same PR #116), on the user's election rather than because the review
  demanded them: the block-bootstrap block length (a fixed *count* of blocks
  shrank `L` as the window shortened, under-reporting the identifiability
  floor 2.6–4.2×; now floored at 3τ from a measured integral time scale, and
  refused outright rather than guessed — see the caution below); a probe-run
  restart leak that warm-started every member from the *truth's* restart on
  any second invocation of `run_probe_series`; five test mutations that had
  survived the whole suite; and the cleanup of 406 lines of callerless figure
  code plus all four blanket mypy waivers in `libs/evaluation` (which exposed
  three latent defects, including an unguarded `Dataset | None` dereference).

## Instructions for implementing agents

- **Before starting a work package:** read its phase plan *and* the
  metrics-doc sections it cites; verify cited file/function anchors against
  the current tree.
- **As each WP lands:** update the status table below (status + PR number)
  and record deviations in that phase plan's Deviations section. Do not
  rewrite plan bodies to match what was built.
- **Follow CLAUDE.md workflow rules:** branch first, pre-commit before
  committing, keep `docs/` in sync when artifacts/configs change.
- **Cross-phase invariants** (violations are review-blockers):
  1. `run_summary.yaml` keys are additive only; changing an existing key's
     values or semantics bumps the `metrics_version` marker (reintroduced in
     WP1.1). **One exemption was taken deliberately** — see the cross-cutting
     cautions below.
  2. Full-ensemble window state files are never `.load()`ed by new code —
     stream member-at-a-time or z-slab-wise, ≤2 reader threads. (The one legacy
     violation, `ensemble_sensor_series` loading whole window files, was
     rewritten in WP1.3; no known violation remains.)
  3. Every metric/figure no-ops gracefully when its inputs are absent
     (old run dirs, flags off, smoke shape).
  4. Only the fair estimators and corrected spread formulas (WP1.1) may be
     used by new code.
  5. `libs/evaluation` stays a leaf: plain functions, arrays/datasets in →
     numbers/dicts/figures out; no Hydra, no run-dir layout knowledge, no
     imports from `pyurbanair` or the backends.

## The evaluation library

All metric and figure code lives in a new editable lib, **`libs/evaluation`**
(phase 0). Scripts (`scripts/esmda/`, `scripts/filtering/`,
`scripts/figure_creation/`) become thin orchestration: resolve run dirs, open
artifacts, call `evaluation`, write YAML/PNGs.

```
libs/evaluation/
├── pyproject.toml          # same hatchling + pixi shape as libs/data-assimilation
└── src/evaluation/
    ├── __init__.py
    ├── scores.py           # probabilistic ensemble scores (fair CRPS/CRPSS, energy
    │                       # score, z-score, rank, spread–skill, hit rate,
    │                       # parameter/sensor metric bundles)
    ├── turbulence.py       # flow statistics: streaming moment accumulation over
    │                       # state files, block bootstrap; phase 3: Welch + LSD
    ├── sensors.py          # reductions of pre-extracted sensor/probe series
    │                       # (window statistics); extraction from state files
    │                       # stays in scripts — it needs the observation-operator
    │                       # machinery from data-assimilation (jax)
    ├── style.py            # figure conventions: colors, quantile bands, shared
    │                       # norms, solid-cell masking
    └── figures.py          # one function per figure ID (P1, S1, S5, F1, D1;
                            # later D3, S4) + the general state/parameter plots
```

Design rules (deliberate, keep them): five flat modules, no subpackages; no
base classes or registries; the only class is the streaming moment
accumulator (genuinely stateful); matplotlib is imported only by
`style`/`figures`. Add abstraction only when a third caller would otherwise
copy-paste.

## Phases and status

| WP | Content | Plan | Size | Status | PR |
|---|---|---|---|---|---|
| 0.1 | `libs/evaluation` skeleton + pixi wiring | [phase0](phase0_evaluation_library.md) | XS | done | #103 |
| 0.2 | Move existing metric/plot code into it (pure refactor) | [phase0](phase0_evaluation_library.md) | M | done | #104 |
| 1.1 | Correctness: fair CRPS/energy + CRPSS, spread–skill, `n_unique`, `metrics_version` | [phase1](phase1_metrics_and_figures.md) | S | done | #105 |
| 1.2 | Parameter bundle: z-score, contraction ratio, fair CRPS/CRPSS | [phase1](phase1_metrics_and_figures.md) | S | done | #106 |
| 1.3 | Sensor-statistics scoring (assimilated + held-out) | [phase1](phase1_metrics_and_figures.md) | S–M | done | #107 |
| 1.4 | Streaming mean fields + TKE + hit rate `q` | [phase1](phase1_metrics_and_figures.md) | M | done | #108 |
| 1.5 | Figures P1, S1, S5, F1, D1 | [phase1](phase1_metrics_and_figures.md) | M | done | #109 |
| 2.1 | Persist obs + per-iteration predicted obs / params | [phase2](phase2_obs_persistence.md) | S | done | #115 |
| 2.2 | `O_N` vs ½ + figure D3 | [phase2](phase2_obs_persistence.md) | S | done | #115 |
| 3 | xie_and_castro validation (held-out) sensors **+** high-rate probes + Welch spectrum + LSD + figure S4 (WP3.1 and WP3.2 merged) | [phase3](phase3_run_upgrades.md) | M | done | #110 |

Sequencing: 0 → 1 strictly; 2 after 1.1; 3 after 1.3, last (backend-touching).
One PR per WP unless a phase plan says otherwise, all
targeting the central `esmda-evaluation` branch (see Branching model).
Phase 2 is likewise one PR (2026-08-07): WP2.2 is small and reads nothing but
WP2.1's own files, so splitting them would have merged a persistence format
with no reader — see the phase-2 plan's Deviations.
Phase 3 is one PR by the 2026-08-06 rescope: its held-out half is
config-only and gated on wall-clock, so the run is launched first and the
probe/spectrum half is implemented while it is in flight. That rescope also
moved the held-out sensors from `barcelona` to `xie_and_castro` (barcelona
is too slow to iterate on; the machinery is case-independent) — see the
phase-3 plan header.
Phases 0–1 apply retroactively to existing run dirs; phase-2 metrics only
cover runs executed after WP2.1, older dirs degrading gracefully.

**Outside the WP list: the filtering pipeline** (PR #111, not a planned WP).
The WP1.5 figures and the metric blocks behind them were ported to
`scripts/run_filtering_pipeline.sh` on request, so the sequential filter
(EnKF) is evaluated with the same instruments as the smoothers. The one
structural difference is what stands in for ESMDA's per-window state files:
the filter keeps only one analyzed frame per cycle unless
`run.ensemble_save_on_disk=true`, in which case it keeps every member's full
forecast segment. Both are supported and which one a run used is recorded in
`run_summary.yaml`'s `cycle_states`; the weaker one nulls the per-cycle
variance and takes the TKE moments across cycles instead of within them. See
[scripts_and_configs.md](../../scripts_and_configs.md) §2.4. S4 has no
filtering counterpart — the probe records need a dedicated solver rerun, which
the ESMDA *pipeline* script does not run either.

Branch review round 3 (see [Implementation
process](#implementation-process)) found that this port's **S1 TKE row and F1 time
mean were labelled as continuous time averages** while, on the shipped default
`cycle_states` source, they are moments over one analyzed frame per cycle — an
across-cycle variance carrying the analysis increments, which is invisible in
the numbers. The caveat existed but only reached `run_summary.yaml`, and the
figure stage reads `eval_fields.nc`. That file now carries a `moment_sampling`
attribute and the figures take a `sampling_note` (a plain string, so invariant 5
is intact). S5's fan legend said "Posterior median" while drawing forecast
segments and now names what it draws. Round 4 then found the fix had
**over-applied**: it gated the qualification on the *presence of a note* rather
than on sparseness, so the `forecast` cycle-state source — every frame of every
cycle's forecast segment, tiling the run, a genuine continuous time average —
was stamped "sample-mean", "sampled over t = a–b s" and "NOT a continuous time
average" one line above a note saying it saw every frame. The same PNG
contradicted itself, and F1's wording was byte-identical between the two sources
apart from the note, so the qualification carried no discriminating signal.
Fixed by recording sparseness as what it is — a property of the frames —
in a separate `moment_sampling_is_sparse` (`0`/`1`) attribute on
`eval_fields.nc` and a separate `sampling_is_sparse` figure parameter: the note
is provenance and prints on both sources, the flag alone drives the
time-mean/sample-mean wording and S1's TKE-row marker. Encoding a caveat in the
presence of prose was the defect. See
[scripts_and_configs.md](../../scripts_and_configs.md) §2.3.

**Outside the WP list: three pylbm backend fixes** (PRs **#112**, **#113**,
**#114**, all merged 2026-08-07). Phase 3's verification run turned up two
backend bugs that were deferred out of that WP as needing their own reviewed
PRs, and fixing the first uncovered a third. Together they add ~350 lines under
`libs/pylbm` (plus ~50 in `pyurbanair`'s ensemble base class) and 1,022 lines of
new test files — none of it planned work, all of it on this branch.
Cross-referenced from
[data_assimilation.md](../../data_assimilation.md) and documented in
[pylbm.md](../../pylbm.md).

- **#112 — restart/output filename width** (`bea72c3`). The pinned Fortran
  declares `character(len=6) cit` and opens `restart_0000_<it:i6.6>.uf`; pylbm
  wrote a 9-digit field, introduced for a newer LBM that a later submodule pin
  moved away from. The solver therefore never saw the restart Python wrote — and
  because the wrapper writes at the iteration the solver itself last wrote, the
  6-digit name it *did* open was the solver's own restart from the previous
  window, sitting in the same directory. **The consequence: for
  `esmda/smoother=state_and_parameter` and `state_and_dynamic`, every pylbm
  rollout produced before 2026-08-07 discarded the Kalman state update at every
  window boundary** and continued on the solver's own free trajectory. Nothing
  looked wrong — the run completed and the fields were physically consistent,
  just not conditioned. The filter's cycle-to-cycle warm start is affected the
  same way. Both sides now spell the width from one constant, pinned in tests
  against the Fortran sources. This also deleted `run_probe_series.py`'s local
  `os.link` workaround (see the phase-3 plan).
- **#113 — a truncated run is a member failure** (`7e1c73d`). The LBM's error
  paths call Fortran `stop`, which exits **0**, so `subprocess.run(check=True)`
  returned cleanly on a partial run and the trims in `run_single` only ever
  shorten. A member that wrote 3 of 48 frames passed straight through
  `resample_from_successes` and killed a two-window ESMDA run hours later inside
  `_stream_concat_members`, nowhere near the cause. Frame counts are now checked
  against the cadence-derived expected count, so a short member is a member
  failure at the window that produced it.
- **#114 — actually read the restart template** (`fae5dc0`). Uncovered by #112,
  whose fix also corrected the template lookup's filename width.
  `_try_load_restart_distribution` passed scipy a list of *scalar* dtypes, which
  it reads as one repeating 20-byte compound and then demands the record be an
  exact multiple of — an LBM restart never is, so the read raised on **every**
  call into a bare `except Exception: return None`. The template was treated as
  absent always, and **every pylbm warm start was rebuilt from a pure-equilibrium
  distribution**, discarding the non-equilibrium stress it exists to carry. This
  is the one that changed what a pylbm warm start physically *is*. It did not
  discard the state update — the macroscopic fields were built from the analyzed
  ρ, u, v, w either way, which is why nothing ever looked wrong; the solver
  simply re-established the stress as a startup transient each time. Measured:
  the first warm frame sat 3.17 % (RMS) from the state handed in with a `max|u|`
  excursion at frame 0, and 0.12 % with no excursion after. Two things outlive
  the fix — the blanking mask is live for the first time, and peak memory per
  member roughly doubles.

  Note for anyone reading the two bugs together: it is **#112**, not #114, that
  discarded the state update. #114's effect is a lost stress tensor and a
  startup transient on a state that did arrive.

**Any pylbm ESMDA or filtering result from before 2026-08-07 should be
re-checked before it is read**, for both reasons. No other backend is affected
and nothing in `libs/data-assimilation` changed.

Cross-cutting cautions:

- WP1.1 shifts historical CRPS/energy-score numbers by ~O(1/M) — cross-
  boundary sweep comparisons are invalid; `metrics_version: 2` marks the
  boundary and the comparison scripts warn on mixing.
- **Invariant 1 has exactly one recorded exemption on this branch.** Round 3's
  `_skill_score` fix changes the written value of `prior_rmse_mean`,
  `prior_crps_mean`, `rmse_reduction_vs_prior` and `crps_reduction_vs_prior` on
  any run whose prior and posterior differ in their NaN pattern across knots,
  without a `metrics_version` bump. The old value averaged the two sides of the
  ratio over different knot sets and was never a valid measurement, so a bump
  would invalidate cross-run comparison across the whole branch in order to
  retire numbers that were already wrong. This is a user decision, and it is
  written out in full in the phase-1 plan's WP1.3 deviations. Nothing else on
  the branch changes an existing key's value.
- **The `identifiability` key now disappears from `run_summary.yaml` on
  production-shaped runs** (round 4's block-bootstrap fix, `bb4aa09`). The
  floor needs a window spanning ≳15 integral time scales; in-canopy velocity
  decorrelates over ~140 s, so a 300 s window holds ~2 independent samples and
  the floor is refused in 5/5 windows at both shipped case shapes. Absent
  means unmeasured, not identifiable — this is the honest verdict, not a
  regression, and it replaces a number that was silently 2.6–4.2× optimistic.
  See [scripts_and_configs.md](../../scripts_and_configs.md) §`identifiability`
  and the phase-3 note.
- The smoke shape (2-member ensemble) degenerates several diagnostics
  (ddof=1 variances, rank histograms) — guard with `null` + log, don't
  special-case.
- Test baseline: compare against the current CI run on `main` (full local
  collection crashes on this Mac — see auto-memory), not against an assumed
  failure count.

## Rollback

On 2026-07-31 the first implementation (WP0.1–1.2 of the old plan, PRs
#97/#99/#100) was reverted from `main` for scope, not correctness: ~11.3k
lines for three of fourteen work packages. This plan is the slimmed
replacement. The reverted code survives on
`origin/agent/esmda-phase0-correctness`,
`origin/agent/esmda-phase1-metrics-foundation`,
`origin/agent/esmda-wp12-sensor-statistics`, and
`origin/agent/esmda-wp13-mean-field-metrics` — **cherry-pick the small,
targeted pieces** (the fair-estimator math, the analytic-CRPS tests, the
streaming accumulator core) rather than rewriting them, but adapt paths to
`libs/evaluation` and drop everything the slimmed metric set no longer
needs. Run dirs produced while #97–#100 were on `main` carry
`metrics_version: 2` and fair-estimator scores; current code writes the
biased values and no marker — those dirs are not comparable with new ones
until WP1.1 reintroduces the marker.
