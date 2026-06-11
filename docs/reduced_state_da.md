# ESMDA in a Reduced SVD/Karhunen–Loève Parameterisation

## Setting

An unknown field $u \in \mathbb{R}^n$ (parameter field, initial condition, or augmented state — for a smoother, typically the initial condition and/or static parameters of a window), observations $d_{\text{obs}} \in \mathbb{R}^m$ collected over a time window $[t_0, t_K]$, and a forward map $\mathcal{G}$ that runs the model across the window and applies the observation operators at the observation times:

$$d = \mathcal{G}(u) + \varepsilon, \qquad \varepsilon \sim \mathcal{N}(0, R).$$

ESMDA updates $u$ given the whole window at once (the smoothing aspect); the SVD/KL machinery makes the update act on reduced coefficients $\xi$ rather than on $u$ directly.

## Stage 0 — Build the reduced basis (offline)

### 1. Assemble prior samples

Take $N_s$ prior realisations $\{u^{(1)}, \dots, u^{(N_s)}\}$ — draws from a specified GRF prior, snapshots from a long free run, or a climatological archive. Form the anomaly matrix

$$A = \frac{1}{\sqrt{N_s - 1}} \left[\, u^{(1)} - \bar{u}, \;\dots,\; u^{(N_s)} - \bar{u} \,\right] \in \mathbb{R}^{n \times N_s}, \qquad \bar{u} = \frac{1}{N_s}\sum_j u^{(j)}.$$

If instead an analytic kernel $C(s,s')$ is available (e.g. Matérn), replace this stage by a Nyström/FEM eigensolve of the covariance operator; everything downstream is identical.

### 2. Thin SVD and truncation

Compute

$$A = \Phi \Sigma V^\top, \qquad \Phi \in \mathbb{R}^{n \times N_s},\; \Sigma = \mathrm{diag}(\sigma_1 \ge \sigma_2 \ge \dots).$$

Then $C \approx AA^\top = \Phi \Sigma^2 \Phi^\top$, i.e. the KL eigenpairs are $(\lambda_i, \varphi_i) = (\sigma_i^2, \Phi_{:,i})$. Choose the rank $r$ by an energy criterion,

$$r = \min \Big\{ k : \sum_{i \le k} \sigma_i^2 \,\Big/\, \sum_i \sigma_i^2 \;\ge\; 1 - \tau \Big\},$$

and keep $\Phi_r \in \mathbb{R}^{n \times r}$, $\Sigma_r \in \mathbb{R}^{r \times r}$. This is the step where SVD (not QR) is essential: the truncation needs the ordered spectrum.

### 3. KL parameterisation

Define the decoder

$$u(\xi) = \bar{u} + \Phi_r \Sigma_r\, \xi, \qquad \xi \sim \mathcal{N}(0, I_r).$$

By construction the prior on $\xi$ is whitened: $\mathrm{Cov}(u) = \Phi_r \Sigma_r^2 \Phi_r^\top \approx C$, and all prior correlation structure now lives in the basis, not in the coefficients.

## Stage 1 — ESMDA on the coefficients (online)

### 4. Schedule

Choose $N_a$ assimilation iterations and inflation coefficients $\alpha_1, \dots, \alpha_{N_a}$ with the consistency condition

$$\sum_{k=1}^{N_a} \frac{1}{\alpha_k} = 1$$

(simplest choice $\alpha_k = N_a$; decreasing schedules front-load small steps). This condition guarantees that in the linear-Gaussian case the composition of the $N_a$ tempered updates reproduces the single exact Bayesian update.

### 5. Initial ensemble

Draw $N_e$ coefficient vectors directly:

$$\xi_0^{(j)} \sim \mathcal{N}(0, I_r), \qquad j = 1, \dots, N_e.$$

There is no sampling error in the prior covariance representation — unlike a standard EnKF prior ensemble, $\mathrm{Cov}(\xi_0) \to I_r$ is exact in distribution, and it can even be enforced exactly with second-order sampling (SEIK-style: $\xi_0 = \sqrt{N_e - 1}\,\Omega^\top$ rows, with $\Omega$ random orthogonal with zero column sums).

Then iterate steps 6–9 for $k = 1, \dots, N_a$.

### 6. Forward pass (the expensive part)

Decode and simulate each member across the full window:

$$d_k^{(j)} = \mathcal{G}\big(\bar{u} + \Phi_r \Sigma_r\, \xi_{k-1}^{(j)}\big), \qquad j = 1, \dots, N_e.$$

Note $\mathcal{G}$ is the *nonlinear* solver — the reduction is in the parameterisation only; there is no Galerkin-projected surrogate dynamics and hence no closure problem.

### 7. Ensemble anomalies

With $\bar{\xi}_{k-1} = \frac{1}{N_e}\sum_j \xi_{k-1}^{(j)}$ and $\bar{d}_k$ analogous,

$$X = \frac{1}{\sqrt{N_e - 1}}\big[\xi_{k-1}^{(j)} - \bar{\xi}_{k-1}\big]_{j} \in \mathbb{R}^{r \times N_e}, \qquad S = \frac{1}{\sqrt{N_e - 1}}\big[d_k^{(j)} - \bar{d}_k\big]_{j} \in \mathbb{R}^{m \times N_e},$$

giving the empirical cross- and data covariances $C_{\xi d} = X S^\top$ and $C_{dd} = S S^\top$.

### 8. Observation perturbation

Perturb the data with inflated noise:

$$d_{\text{obs}}^{(j)} = d_{\text{obs}} + \sqrt{\alpha_k}\, \varepsilon^{(j)}, \qquad \varepsilon^{(j)} \sim \mathcal{N}(0, R).$$

### 9. Update with TSVD inversion — the second place SVD enters

The update is

$$\xi_k^{(j)} = \xi_{k-1}^{(j)} + X S^\top \big(S S^\top + \alpha_k R\big)^{-1} \big(d_{\text{obs}}^{(j)} - d_k^{(j)}\big),$$

and the $m \times m$ inverse is computed by Evensen's subspace pseudo-inversion rather than directly. Scale into whitened data space, $\tilde{S} = R^{-1/2} S$, take the thin SVD

$$\tilde{S} = U_p \Lambda_p W_p^\top, \qquad p \le \min(m, N_e - 1),$$

optionally truncating $p$ at, say, 99% of $\sum \Lambda_p^2$ (this discards noise-level directions in data space and regularises the inversion — TSVD as regulariser, the third role SVD plays), and use

$$\big(S S^\top + \alpha_k R\big)^{-1} \approx R^{-1/2}\, U_p \big(\Lambda_p^2 + \alpha_k I_p\big)^{-1} U_p^\top R^{-1/2}.$$

Everything is now small: the solve is $p \times p$ diagonal, the gain application costs $O((r + m) N_e p)$, and nothing of size $n$ appears in the update at all.

### 10. Loop and decode

Return to step 6 with $\xi_k^{(j)}$. After $N_a$ iterations, decode the posterior ensemble:

$$u_{\text{post}}^{(j)} = \bar{u} + \Phi_r \Sigma_r\, \xi_{N_a}^{(j)},$$

whose spread is the reduced-rank posterior; pushing each member through the model once more gives the smoothed trajectory ensemble over the window.

## Why this composition is coherent

The whitening is what makes the pieces lock together. In $\xi$-coordinates the prior is $\mathcal{N}(0, I_r)$ exactly, so the prior-related sampling noise that plagues raw-space EnKF/ESMDA (spurious long-range correlations, the usual motivation for localisation) is confined to the *likelihood* side — the $S$ matrix — rather than contaminating the prior too. The update can never push $u$ outside $\mathrm{span}(\Phi_r) + \bar{u}$, so posterior fields inherit the prior's smoothness class and physical balances encoded in the modes for free. ESMDA's tempering also interacts well with the reduction: each small-$1/\alpha_k$ step keeps the ensemble closer to the Gaussian regime where the KL coefficients' uncorrelatedness actually implies independence.

## Practical caveats

**Rank vs ensemble size.** You need $N_e \gtrsim r$ for $X$ to resolve the coefficient space, or the rank deficiency has merely moved from $n$ to $r$. If $r$ must be large, localise *in coefficient space* using the spectral decay (taper updates to high-index $\xi_i$, which carry small-scale, weakly observed structure) — spatial localisation is no longer directly available since the $\xi_i$ are nonlocal.

**Fixed basis.** $\Phi_r$ is frozen from the prior, so if observations demand structure orthogonal to the prior's dominant modes the method cannot produce it. Standard fixes: re-expand $C$ from the current ensemble between MDA passes (re-anchoring, at the price of reintroducing sampling noise), or augment $\Phi_r$ with a few residual directions from $d_{\text{obs}} - \bar{d}$ mapped back through $X S^\top$.

**Gaussianisation.** ESMDA is a linear-update method and the KL prior is Gaussian by construction, so for genuinely non-Gaussian fields (channelised permeability, intermittent turbulence statistics) the principled upgrade is to replace the linear decoder $\xi \mapsto \bar{u} + \Phi_r \Sigma_r \xi$ with a learned generative decoder and run exactly the same ESMDA loop on its latent space — the algorithm above is unchanged except step 6's decode, which is precisely what makes it a good baseline to build on.

---

## Implementation in this repo (online basis)

The scheme above is implemented for the **state-bearing smoothers**
(`StateAndParameterESMDA`, `StateAndTimeVaryingParameterESMDA`) with one key
difference: **the basis is built online, refitted each ESMDA iteration from
the current forecast ensemble**, not offline from prior samples (Stage 0 is
folded into the loop). This is the "re-anchoring" fix from the *Fixed basis*
caveat applied by construction. Parameter rows always keep the exact global
update; the parameter-only smoothers are untouched.

- **Class:** `OnlineStateReduction`
  ([`reduction.py`](../libs/data-assimilation/src/data_assimilation/reduction.py)),
  passed as the `state_reduction=` argument of the state-bearing smoothers
  (default `None` = the full-space update, exactly the previous behavior).
  Knobs: `energy_fraction` (retained-energy truncation, step 2),
  `max_rank`, `basis_source`, `snapshot_stride`.
- **Basis source** (`basis_source`): `"initial_condition"` fits the SVD to the
  `time=0` ensemble anomalies (rank ≤ N_e−1; the encoded coefficients are
  exactly whitened, and at full rank the update is identical to the
  full-space one). `"window_snapshots"` fits it to every output frame of
  every member (N_e·N_t samples; richer basis).
- **Increment decoding:** the update is applied as
  `u += Φ_r Σ_r (ξ_post − ξ_prior)` rather than a full decode, so each
  member's projection residual is preserved (a zero Kalman update leaves the
  state untouched for either source).
- **Localization** is incompatible with the reduction (the ξ are nonlocal);
  the constructor raises. Step 9's TSVD data-space inversion is not
  implemented — the observation counts here keep the direct m×m solve cheap.
- **Optional final trajectory smoothing** (`final_time_smoothing=True`,
  requires `state_reduction` and an in-memory forward model): after the ESMDA
  loop's final forecast, one extra *un-tempered* (`alpha=1`) Kalman update of
  the state at **all time steps of the window jointly**, reusing that
  forecast (no extra solve) with the parameters frozen. Only feasible in the
  reduced basis — the space-time vector is N_s·N_t, but the trajectory
  ensemble collapses it to ≤ N_e−1 coefficients. The result is a smoothing
  estimate per frame, not a model integration; in multi-window rollout the
  carried-over last frame is then a Kalman-blended field (consistent with the
  analyzed-IC warm starts the loop already performs).
- **Config:** the `esmda/state_reduction` group
  ([`conf/esmda/state_reduction/`](../conf/esmda/state_reduction/)), default
  `none`. Enable with `esmda/state_reduction=svd` and tweak fields on the CLI
  (e.g. `esmda.state_reduction.basis_source=window_snapshots`); the final
  smoothing step is `esmda.final_time_smoothing=true`. Both are wired into
  the state-bearing smoother YAMLs only.
- **Tests:** [`tests/test_state_reduction.py`](../tests/test_state_reduction.py)
  (incl. the exact full-rank ↔ full-space equivalence) and the
  `state_reduction` cases in
  [`tests/test_run_esmda.py`](../tests/test_run_esmda.py).