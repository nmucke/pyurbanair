# Master plan: ESMDA evaluation — slimmed

> **Status: ready to implement** (slimmed 2026-08-03 after the July rollback;
> see [Rollback](#rollback)). **Companion to** the research document
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
     WP1.1).
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
| 2.1 | Persist obs + per-iteration predicted obs / params | [phase2](phase2_obs_persistence.md) | S | not started | — |
| 2.2 | `O_N` vs ½ + figure D3 | [phase2](phase2_obs_persistence.md) | S | not started | — |
| 3.1 | Barcelona validation (held-out) sensors | [phase3](phase3_run_upgrades.md) | XS | not started | — |
| 3.2 | High-rate probes + Welch spectrum + LSD + figure S4 | [phase3](phase3_run_upgrades.md) | M | not started | — |

Sequencing: 0 → 1 strictly; 2 after 1.1; 3.1 anytime after 1.3; 3.2 last
(backend-touching). One PR per WP unless a phase plan says otherwise, all
targeting the central `esmda-evaluation` branch (see Branching model).
Phases 0–1 apply retroactively to existing run dirs; phase-2 metrics only
cover runs executed after WP2.1, older dirs degrading gracefully.

Cross-cutting cautions:

- WP1.1 shifts historical CRPS/energy-score numbers by ~O(1/M) — cross-
  boundary sweep comparisons are invalid; `metrics_version: 2` marks the
  boundary and the comparison scripts warn on mixing.
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
