# Ensemble-transform filters (ETKF and variants)

A short orientation to the deterministic analysis schemes added to the
sequential filter. The reference detail — diagnostics, config wiring, how these
compose with localization and the state reduction — is in
[data_assimilation.md](data_assimilation.md#analysis-schemes); the
implementation is
[filtering/etkf.py](../libs/data-assimilation/src/data_assimilation/filtering/etkf.py).

> **Status.** These ship *disabled by default* (`filtering/analysis=stochastic`).
> They are tested but **not yet benchmarked**: no accuracy, memory or speed
> claim is made. The campaign record
> [temp/filtering_ensemble_transform_benchmark.md](temp/filtering_ensemble_transform_benchmark.md)
> is deliberately unpopulated.

## Why a second analysis scheme

The default `StochasticEnKFAnalysis` draws perturbed observations, so the
posterior sample covariance equals the Kalman covariance only *in expectation
over the perturbation draw*. At `N_e = 50` that sampling noise is not small.

The ensemble-transform family removes the draw. It computes one ensemble-space
weight matrix and right-multiplies the forecast anomalies with it, giving the
Kalman covariance exactly, given the sample forecast moments. It is
deterministic: the `rng_key` is accepted and ignored.

## The kernel

Everything below runs through one function, `ensemble_transform`. With `N = N_e`
members, `R_eff = diag(E_inf² · C_D)` the effective observation-error variances,
and the whitened anomalies / innovation

```
Y_w = R_eff^{-1/2} (pred_obs - mean(pred_obs))        (N_d, N_e)
d_w = R_eff^{-1/2} (obs      - mean(pred_obs))        (N_d,)
```

take the thin SVD `Y_w = U S Vᵀ` (Hunt et al. 2007, in SVD form):

```
C    = (N-1) I + Y_wᵀ Y_w
W_a  = sqrt(N-1) · C^{-1/2}  =  I + V diag( sqrt((N-1)/((N-1)+s²)) - 1 ) Vᵀ
wbar = C^{-1} Y_wᵀ d_w       =      V diag( s / ((N-1)+s²) ) Uᵀ d_w
```

and the posterior is `mean + X @ (wbar 1ᵀ + W_a)`, with `X` the raw forecast
anomalies.

Three properties of this form are contracts, not incidental:

**The square root is the symmetric one.** Any `W_a Q` with orthogonal `Q` gives
the same posterior covariance, so the choice looks free. It is not: RTPP blends
posterior anomalies against *prior* anomalies member by member, and a rotated
root scrambles member identity, so RTPP would blend a member against an
unrelated perturbation. Every moment-based test would still pass. (A rotated-root
mutation is in the suite and is killed.)

**Mean preservation is structural.** `Y_w` has centered rows, so `Y_w 1 = 0`,
hence `Vᵀ 1 = 0` and `W_a 1 = 1`. The anomaly transform contributes nothing to
the posterior mean — including under truncation. This is asserted, not assumed.

**The update is pure right-multiplication.** The weights depend only on
`(pred_obs, obs, C_D)`; nothing may depend on the analyzed rows. That is what
lets `BaseFilter` call the analysis a second time on the reduction's small
coordinate array to measure the discarded increment, and what makes the state
reduction work here for free (encoding is a left multiplication, and left and
right multiplication commute).

The transform is stored **factored** as `(V, scale, wbar)` and applied as
`X + (X V) diag(scale - 1) Vᵀ`. The dense `N_e × N_e` matrix is a convenience
for tests and a single global analysis, never a required intermediate.

## The four variants

| `filtering/analysis=` | Class | Localization | What it changes |
|---|---|---|---|
| `etkf` | `ETKFAnalysis` | **forbidden** | One global transform per cycle |
| `etkf_tsvd` | `ETKFAnalysis` | **forbidden** | …plus observation-space truncation |
| `letkf` | `LETKFAnalysis` | **required** | One transform per local block |
| `letkf_tsvd` | `LETKFAnalysis` | **required** | …plus per-block truncation |

They differ along exactly two axes.

### Global vs local

**`ETKFAnalysis`** applies one transform to every augmented row — state,
parameters and appended diagnostic rows alike — so it supports `state` /
`parameter` / `joint` modes and the filtering state reduction without knowing
about any of them. It refuses to run with a localization strategy, because a
localized deterministic analysis is a different estimator and a config name must
not claim localization the analysis silently ignores.

**`LETKFAnalysis`** computes one transform per *distinct local observation
selection*, from the same `inflation_factors` the stochastic localized update
uses (`R_eff = C_D · E_inf²`, `E_inf = inf` excludes). Switching
`filtering/analysis` between the two therefore changes the estimator and nothing
about what "localization radius" means. It requires a localization strategy —
without one it is a slower global ETKF — which also rules out combining it with
the global state reduction, whose POD basis has no spatial support.

Three choices make LETKF affordable at production size (`N_s ≈ 230k`, `N_e = 50`,
`N_d ≈ 12`): blocks are deduplicated on the **canonical inflation vector** rather
than on `group_ids` (which subsumes block grouping, and additionally collapses
every observation-free row into one block and every globally-updated row into
the all-ones block that *is* the global ETKF); blocks with no active observation
are partitioned out host-side and returned bit-identical; and no `N_e × N_e`
transform is ever assembled. How far the dedup actually collapses the block count
on a real run is one of the things the benchmark exists to measure.

### With and without the observation TSVD

The `*_tsvd` variants enable `ObservationTSVD`, nested on the analysis object
rather than exposed as a separate class, because it regularizes the transform
the analysis already computes. It truncates weak **linear combinations of the
whitened observation anomalies** — a different axis from
`filtering/state_reduction`, which acts on state rows, and from localization,
which decides *which* observations reach a block. The order is
`localize → form R_eff → whiten Y → TSVD → transform`.

Truncation means "do not assimilate this combination of observations". It leaves
`R` untouched and can only *remove* update directions, which is why it is off by
default: with the shipped `N_d ≈ 12` there is little to regularize.

The rank criterion cuts on the **suffix** energy — retain the smallest prefix
whose discarded tail holds at most `1 - energy_fraction` of the squared spectrum.
Mathematically identical to the usual prefix form, but a float32 prefix sum
saturates at `1.0` after ~7 significant digits and would let dtype decide the
rank. There is exactly one criterion, shared by the global ETKF (which reads a
Python `int`) and the LETKF block loop (which consumes a traced mask); an earlier
revision had two, and they disagreed on 36% of a random sweep at
`energy_fraction = 1.0`.

Retaining a round-off direction is harmless here: the weights `s/((N-1)+s²)` and
`sqrt((N-1)/((N-1)+s²))` are both *damped* as `s → 0`, never a `1/s`
amplification — the opposite of ESMDA's whitened `encode`. So disabling the TSVD
keeps every thin-SVD direction rather than truncating at the numerical rank.

## Diagnostics

Both schemes publish per-cycle rank/energy readouts into `cycle_diagnostics.yaml`
(`transform_*` for the global ETKF, `local_*` summaries for LETKF, `None` for the
stochastic scheme). `retained_rank` says nothing about the ensemble when the TSVD
is off — `available_rank`, the numerically nonzero count bounded by
`min(N_d, N_e - 1)`, is the meaningful one. The LETKF block/row counts are what
the resource gate in
[plans/filtering_state_reduction_and_transforms.md](plans/filtering_state_reduction_and_transforms.md)
§6 asks to be reported. Full field list:
[data_assimilation.md](data_assimilation.md#filtering-state-reduction).

## Running one

```bash
# Global deterministic update — select localization explicitly, since the
# scheme refuses anything but `none`.
python scripts/filtering/run_filtering.py filtering.mode=state \
  filtering/analysis=etkf filtering/localization=none

# Localized deterministic update.
python scripts/filtering/run_filtering.py filtering.mode=state \
  filtering/analysis=letkf filtering/localization=distance
```
