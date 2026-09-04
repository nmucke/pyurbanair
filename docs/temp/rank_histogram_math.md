# Rank histogram (figure D1) — the mathematics

How `rank_histogram.png` is computed in the ESMDA, filtering and
filter-smoothing pipelines. Working note, not a maintained reference.

**Two things it is not:** it is not a count of "truth inside vs outside the
spread" (that would be 2 bins — this has $M+1$), and it is not per time step
(time is reduced away in step 4, *before* any ranking happens).

Code: [`evaluation/scores.py`](../../libs/evaluation/src/evaluation/scores.py)
(steps 4–6), [`evaluation/sensors.py`](../../libs/evaluation/src/evaluation/sensors.py)
(step 3), [`evaluation/figures.py`](../../libs/evaluation/src/evaluation/figures.py)
(steps 7–10).

## Notation

| Symbol | Meaning |
|---|---|
| $M$ | ensemble size (members $m = 1 \dots M$) |
| $W$ | number of windows (ESMDA) or cycles (filtering) |
| $n_s$ | number of sensors in the set |
| $i$ | sensor index, $i = 1 \dots n_s$ |
| $T$ | window length in seconds (`sim_time`) or cycle length (`bin_seconds`) |
| $x_m(t,i)$ | member $m$'s series at sensor $i$ |
| $x^*(t,i)$ | truth series at sensor $i$ |

---

## Step 1 — Split into panels

Run everything below independently for each combination of

- **sensor set**: `assimilation` (in-sample) and `validation` (held out) → figure rows
- **half**: `prior` and `posterior` → figure columns

## Step 2 — Extract series

Member series $x_m(t,i)$ and truth series $x^*(t,i)$ at the set's sensors.

## Step 3 — Bin in time

Assign frame time $t$ to window

$$
w(t) \;=\; \operatorname{clip}\!\left(\left\lfloor \frac{t}{T} + \varepsilon \right\rfloor,\; 0,\; W-1\right),
\qquad \varepsilon = 10^{-9}
$$

so window $w$ collects the frames with $t \in [wT,\ (w+1)T)$. The tolerance
$\varepsilon$ keeps a boundary frame from falling one ULP into the previous
window.

## Step 4 — Reduce each bin to statistics

For each window $w$, sensor $i$, and quantity $q \in \{u, v, w, |\mathbf{u}|\}$,
compute two statistics over the $N_w$ frames in that bin:

$$
X^{\text{mean},q}_m[w,i] \;=\; \frac{1}{N_w}\sum_{t \in w} x^q_m(t,i)
$$

$$
X^{\text{var},q}_m[w,i] \;=\; \frac{1}{N_w - 1}\sum_{t \in w}\Big(x^q_m(t,i) - X^{\text{mean},q}_m[w,i]\Big)^2
$$

Identically for the truth, giving $X^{*S,q}[w,i]$. The variance uses
$\mathrm{ddof}=1$: it estimates the flow's variance from a finite window, not
the window's own second moment. A one-frame bin therefore yields
$\mathrm{NaN}$, which is correct rather than zero.

**Time no longer exists after this step.** Every downstream quantity is indexed
by $(w, i)$.

## Step 5 — Rank the truth within the ensemble

Fix one statistic–quantity pair $(S,q)$. Define a **knot** $k = (w,i)$, so
there are $K = W \cdot n_s$ of them. The rank is the number of members below
the truth, plus a tie-break draw:

$$
r_k \;=\; \underbrace{\#\{\, m : X_m[k] < X^*[k] \,\}}_{\text{members below}} \;+\; U_k,
\qquad
U_k \sim \mathrm{Unif}\{0, 1, \dots, \tau_k\}
$$

$$
\tau_k \;=\; \#\{\, m : X_m[k] = X^*[k] \,\}
$$

so $r_k \in \{0, 1, \dots, M\}$.

- $r_k = 0$ → every member lies **above** the truth.
- $r_k = M$ → every member lies **below** the truth.
- Untied knots have $\tau_k = 0$, hence $U_k = 0$ deterministically.
- The draw is seeded (`_DEFAULT_TIE_SEED = 0`) so re-running the metric stage
  reproduces `run_summary.yaml` byte for byte.
- Knots where $X^*[k]$ or any $X_m[k]$ is non-finite give $r_k = \mathrm{NaN}$
  and are **dropped**.

## Step 6 — Count into rank bins

$$
c^{S,q}_j \;=\; \#\{\, k : r_k = j \,\}, \qquad j = 0, 1, \dots, M
$$

A vector of length $M+1$. **These are the bins:** one per attainable rank.
Written to `run_summary.yaml` as
`sensor_statistics[set][half][statistic]["rank_counts"]`.

## Step 7 — Pool

Sum the count vectors over every statistic–quantity key in the panel:

$$
C_j \;=\; \sum_{S,q} c^{S,q}_j, \qquad N \;=\; \sum_{j=0}^{M} C_j
$$

So the counts are pooled over **windows, sensors, statistics and quantities** —
$N$ is large enough to read a shape from, at the cost of all locality.

## Step 8 — Coarsen for display

Split $\{0, \dots, M\}$ into $G = \min(10,\, M+1)$ contiguous groups via
`np.array_split`. Group $g$ has size $n_g$ (**unequal** when $M+1$ is not
divisible by $G$):

$$
\hat{C}_g \;=\; \sum_{j \in g} C_j
$$

## Step 9 — Uniform reference and consistency band

Under calibration, $r_k \sim \mathrm{Unif}\{0,\dots,M\}$, so group $g$ has
probability $p_g$ and its count is binomial:

$$
p_g = \frac{n_g}{M+1}, \qquad
\mathbb{E}[\hat{C}_g] = N p_g, \qquad
\mathrm{sd}[\hat{C}_g] = \sqrt{N p_g (1 - p_g)}
$$

The dashed step is $N p_g$ and the shaded band is

$$
N p_g \;\pm\; 2\sqrt{N p_g (1 - p_g)}
$$

The reference is **per group**, not a flat line — with unequal $n_g$, a flat
line would flag a calibrated ensemble as biased.

## Step 10 — Plot

Bars $\hat{C}_g$ (counts, not densities) over the dashed step and its band, one
panel per (sensor set, half), with $N$ annotated.

---

## Reading it

| Shape | Conclusion |
|---|---|
| Flat | Spread consistent with error |
| U | Under-dispersed / over-confident |
| Dome | Over-dispersed |
| Mass at $j=0$ | Ensemble biased **high** (truth below all members) |
| Mass at $j=M$ | Ensemble biased **low** (truth above all members) |

The two comparisons that carry the information: **prior vs posterior**
(flat → U means assimilation cut the spread more than the error) and
**assimilation vs validation** (only the validation row is out-of-sample).

## Caveats that change the conclusion

1. **Step 7 can cancel shapes.** An over-dispersed mean and an under-dispersed
   variance sum to something flat. Cross-check the per-statistic `z_score`
   entries in `run_summary.yaml`.
2. **The band assumes independent knots.** Neighbouring sensors and consecutive
   windows are correlated, so the effective sample is $< N$ and the true band is
   wider than drawn. Small excursions are over-read.
3. **Filtering at default `run.ensemble_save_on_disk=false`** ranks the
   *analyzed* frames — in-sample by construction, and a one-frame bin, so the
   variance statistic is $\mathrm{NaN}$ and only `mean_*` contributes. Set it to
   `true` to rank forecast segments, which is the honest out-of-sample view.
4. **Small $M$** → $M+1 \le 10$ means step 8 does nothing and the bars are raw
   noise.
5. **Identifiability.** `_score_window_statistic` warns when across-member
   spread is under $3\times$ its own bootstrap sampling noise. Where it fires,
   the ranks are noise, not calibration.
6. **Hybrid runs have two histograms** answering different questions: the run
   root is the filter's per-cycle calibration, `esmda_view/` is the MDA
   posterior's per-window calibration.
