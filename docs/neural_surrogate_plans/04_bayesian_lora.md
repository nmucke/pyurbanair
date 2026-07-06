# Plan 04 — Bayesian LoRA (BaLoRA) variant + stochastic inference

**Goal:** a second LoRA variant, switchable via `lora.variant: balora` in the
plan-01/03 fine-tuning configs, implementing BaLoRA
([arXiv:2605.08110](https://arxiv.org/abs/2605.08110), Coscia/Löwe/Welling):
stochastic adapters trained with an ELBO, giving calibrated predictive
uncertainty via weight sampling at inference. No library ships this (checked:
PEFT has no Bayesian LoRA method) → implemented in-repo, but as a
**PEFT-compatible custom layer** so it shares all plan-01 machinery.

## 0. The method (concrete, from the paper)

Per adapted layer with base weight `θ₀`, LoRA factors `θ_A ∈ R^{r×d}`
(down), `θ_B ∈ R^{k×r}` (up, zero-init):

- Only `θ_A` is stochastic, with input-adaptive multiplicative noise:
  `ω_A;ij = θ_A;ij + sqrt(α(x)) · |θ_A;ij| · ε_ij`, `ε ~ N(0,1)`, i.e.
  `q(ω_A|x) = N(θ_A, α(x)·θ_A²)` (element-wise). `θ_B` stays deterministic.
- `α(x) ∈ R^L` (one positive scalar per adapted layer) comes from a small
  **inference network**: pooled features of the input → MLP → softplus.
- **Training** (ELBO, single MC sample per step):
  `L = E_q[log p(y|ω_A,x)] − D_KL[q ‖ p]`, prior
  `p(ω_A;ij) = N(0, p/(1−p)·θ_A;ij²)` with dropout-rate hyperparameter `p`,
  giving the closed-form per-layer KL
  `D_KL = ½·((α+1)(1+p)/p + log(p/(1−p)) − log α − 1)` — a scalar function of
  `α(x)` and `p` only (cheap).
- **Inference**, two modes:
  - *deterministic*: use means, i.e. exactly standard LoRA (mergeable, zero
    overhead);
  - *sampling*: low-rank reparameterization — per layer compute
    `d(x) = α(x)·(θ_A²)(x²) ∈ R^r`, then
    `y = θ₀x + θ_Bθ_Ax + θ_B(sqrt(d(x)) ⊙ ε_r)`, `ε_r ~ N(0, I_r)`;
    run S forward passes, aggregate sample mean and variance.
- Init: `θ_A` small isotropic normal, `θ_B` zero (⇒ output-identity at init,
  same guarantee as standard LoRA). Paper ranks 8–64 with `alpha ≈ 2r`.

## 1. Module: `finetuning/balora.py`

### Layers

`class BaLoRALinear(nn.Module, peft.tuners.lora.layer.LoraLayer)` and
`class BaLoRAConv3d(...)`, registered via PEFT's custom dispatch:

```python
config = LoraConfig(target_modules=..., rank=..., ...)
config._register_custom_module({nn.Linear: BaLoRALinear, nn.Conv3d: BaLoRAConv3d})
peft_model = get_peft_model(base_model, config)
```

Constraints from PEFT's custom-module API (documented): `__init__(base_layer,
adapter_name, **kwargs)`, learnable params in `ModuleDict/ParameterDict` keyed
by adapter name, attribute names starting with `lora_`; `merge`/`unmerge`
implemented for the deterministic path. Custom mappings are **not persisted**
by `save_pretrained` — our `finetuning.load_adapter` re-registers the mapping
before `PeftModel.from_pretrained` (plan 01's loader already owns this seam).

Layer behavior, controlled by a module-level mode flag (see "runtime modes"):

- `train`: sample `ω_A` once per forward via the element-wise
  reparameterization above; expose `self.last_kl` (scalar) computed from the
  layer's current `α`.
- `eval_mean`: standard LoRA forward (no noise) — also what `merge` uses.
- `eval_sample`: low-rank trick (`d(x)` in R^r) — cheaper than element-wise
  when sampling repeatedly at inference.

For `Conv3d`, `θ_A x` is the rank-r down-conv; the noise term needs
`(θ_A²)(x²)` — implement as a conv of `x²` with `θ_A²` (same
stride/padding), which is the exact diagonal-covariance propagation. Cover
with a unit test against a brute-force MC estimate on a tiny layer.

### Inference network (α(x))

`class AlphaNet(nn.Module)`: pooled summary of the model input → MLP
(`d → 256 → 256 → L`) → softplus → per-layer `α`. Design decisions for our
setting (paper used frozen-backbone CLS tokens):

- Input summary: global-average-pool of the **normalized input state over
  fluid cells**, per channel, concatenated with normalized params →
  a ~`(C+P)`-vector. Cheap, permutation-safe, and available identically at
  train and rollout time.
- Ownership: `AlphaNet` hangs off the `PeftModel` wrapper (a
  `BaLoRAController` module that also keeps the ordered list of adapted
  layers); a pre-forward hook computes `α(x)` once per batch and distributes
  `α_l` to each layer. Saved inside `adapter/` (it *is* adapter state).

### Controller / public API

`class BaLoRAController`: `set_mode("train"|"eval_mean"|"eval_sample")`,
`kl_loss()` (sum of `last_kl` over layers), `sample_weights(seed)` (for
frozen-noise rollouts, below), `num_adapted_layers`. `inject_lora(...,
variant="balora", balora_cfg={prior_p, alphanet_hidden, ...})` from plan 01
returns `(peft_model, controller)`.

## 2. Training integration (ELBO)

Small, opt-in extension of the existing trainers — not a new trainer class:

- `BaseTraining` gains an optional `aux_loss_fn: Callable[[], Tensor] | None`
  (default `None` → byte-identical behavior). When set, `_final_loss`'s
  result gets `+ kl_weight * aux_loss_fn() / n_train_samples` (the 1/N ELBO
  scaling; expose `kl_weight` for tempering, default 1.0).
- The fine-tune script passes `aux_loss_fn=controller.kl_loss` when
  `variant == "balora"`.
- Single MC sample per step = just the layer's `train` mode; no other loop
  changes. Log the KL term as an extra `metrics.csv` column.
- AMP note: sampling uses `sqrt` of variances — keep the noise math in fp32
  inside the layer (autocast-exempt region) to avoid bf16 underflow of
  `θ_A²`.

Applies unchanged to both targets: plan-01 next-step P3D fine-tunes and
plan-03 DFT fine-tunes (the encoder/decoder adapters there are BaLoRA layers;
the fully-trained sub-network/γ modules stay deterministic).

## 3. Inference design (the stochastic-model question)

### Artifacts

`weights.pt` = merged **mean** weights (deterministic mode) — ESMDA and all
existing tooling work unchanged, zero risk. `adapter/` additionally holds
`θ_A, θ_B, AlphaNet` (+ `balora_config.json`: prior_p, mode defaults). So a
BaLoRA model degrades gracefully to its mean model anywhere the adapter isn't
loaded.

### Modes in `NeuralSurrogateForwardModel`

New optional config block in `conf/model/neural_surrogate.yaml`
(default absent → current behavior):

```yaml
balora:
  mode: deterministic        # deterministic | sample
  resample: rollout          # rollout | step   (when mode: sample)
  seed: null                 # base seed; member index is folded in
```

- `deterministic`: load `weights.pt` as today. Nothing else.
- `sample`: `_load_weights` additionally rebuilds the adapter
  (re-register custom modules → `load_adapter`) and sets `eval_sample`.
  - `resample: step`: fresh `ε_r` every model call — i.i.d. noise per step,
    the paper's UQ setting (S rollouts → predictive spread).
  - `resample: rollout` (**default for DA**): draw one weight sample per
    member per rollout (`sample_weights(seed=base_seed + member_idx)`, noise
    frozen via cached `ε`), so each ensemble member integrates a *consistent*
    perturbed model. This is the natural ESMDA fit: BaLoRA's posterior over
    adapters becomes the model-error term of the ensemble, replacing/adding to
    additive inflation.
- Ensemble wiring: `clone_for_member` currently shallow-shares `self.model`.
  For per-member weight samples, sharing stays fine as long as sampling is
  *stateless per call* (`resample: step`) or the frozen sample is applied
  per-chunk in `rollout_batched` (members are already batched — draw a
  per-member `ε_r` batch dimension inside the layer: noise shaped
  `(B_members, r)` instead of `(r,)`). Prefer the batched-noise design — no
  cloning, no serialization issues with `forkserver`.
- Aggregation: for standalone UQ (outside DA) add a small
  `scripts/neural_surrogate/rollout_uncertainty.py` or a flag in the existing
  testing script: S rollouts → per-cell mean/std NetCDF. S default 32
  (paper used ~100 for scalar tasks; fields are expensive — make it a knob).

## 4. Tests

- Unit: mean-mode ≡ standard-LoRA forward (exact); `train`-mode expectation ≈
  mean-mode (MC test, loose tol); closed-form KL matches a numerical
  `torch.distributions.kl_divergence` on sampled cases; Conv3d variance
  propagation vs brute-force MC; low-rank `eval_sample` covariance matches
  element-wise sampling covariance (MC).
- Integration: fine-tune smoke with `variant: balora` (2 epochs, fixture
  data) → deterministic export loads in ESMDA path; `mode: sample` rollout
  produces member-dependent trajectories with fixed seed reproducibility.
- No-op guard: absent `balora:` block in the model config ⇒
  `NeuralSurrogateForwardModel` code path byte-identical (existing e2e tests
  already enforce this de facto).

## 5. Open questions (defaults chosen)

- **Which layers get BaLoRA** in P3D: start with the same `attention` preset
  as standard LoRA (paper adapts attention+MLP projections in LLMs; QKV-only
  for ViTs). Rank per plan 01 (`r=32, alpha=64`) until swept.
- **`prior_p`**: paper's dropout-rate prior parameter; default 0.1, expose in
  config, sweep with calibration (spread/skill) on held-out rollouts.
- **Interaction with `torch.compile`**: stochastic branches + per-batch α are
  recompile bait; fine-tuning already defaults `compile_model: false`
  (plan 01). For sampled *inference* measure first; eager is likely fine.
- **BaLoRA on the AE stage** (plan 02) is out of scope — pre-training is not
  parameter-efficient fine-tuning; uncertainty enters at the predictor stage.
