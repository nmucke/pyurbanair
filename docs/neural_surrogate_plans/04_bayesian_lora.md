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
  giving the closed-form **per-element** KL
  `D_KL;ij = ½·((α+1)(1−p)/p + log(p/(1−p)) − log α − 1)` — θ-independent,
  so the per-layer KL is that scalar **× numel(θ_A) = r·d**, a cheap function
  of `α(x)` and `p` only. `α(x)` carries a batch dimension; the layer reduces
  it (batch mean) before the per-layer sum. (An earlier draft had `(1+p)/p`
  and no element count; the form above follows directly from the stated `q`
  and prior — re-verify against the paper before implementation, and unit-test
  it numerically either way, see §4.)
- **Inference**, two modes:
  - *deterministic*: use means, i.e. exactly standard LoRA (mergeable, zero
    overhead);
  - *sampling*: low-rank reparameterization — per layer compute
    `d(x) = α(x)·(θ_A²)(x²) ∈ R^r`, then
    `y = θ₀x + θ_Bθ_Ax + θ_B(sqrt(d(x)) ⊙ ε_r)`, `ε_r ~ N(0, I_r)`;
    run S forward passes, aggregate sample mean and variance.
- Init: `θ_A` small isotropic normal, `θ_B` zero (⇒ output-identity at init,
  same guarantee as standard LoRA — and because the noise term also passes
  through `θ_B`, even the *sampled* forward is exactly the identity at init).
  Paper ranks 8–64 with `alpha ≈ 2r`.

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
implemented for the deterministic path. Two loose ends to nail down here:

- Custom mappings are **not persisted** by `save_pretrained`. Plan-01's
  `finetuning.load_adapter` is today a bare `PeftModel.from_pretrained`
  (`finetuning/inject.py`) with **no config-injection seam** — extend it with a
  variant-aware path: when `balora_config.json` is present in the adapter dir,
  load the `LoraConfig`, call `_register_custom_module` on it, and pass it via
  `PeftModel.from_pretrained(..., config=...)`.
- `_register_custom_module` is *private* PEFT API and the dependency pin is
  only `peft>=0.11` (`libs/neural-surrogates/pyproject.toml`). Pin tighter
  (the dev env carries 0.19) and add a smoke test that fails loud on upstream
  drift.

Layer behavior — **sampling keys off the standard `nn.Module` training flag**,
not a bespoke train-mode toggle, so `BaseTraining`'s existing `model.train()` /
`model.eval()` calls do the right thing with zero trainer changes; the
controller flag below only selects between the two *eval* behaviors:

- `self.training == True`: sample `ω_A` once per forward via the element-wise
  reparameterization above; expose `self.last_kl` (the per-layer scalar:
  per-element KL × `r·d`, batch-mean over `α(x)`).
- `self.training == False`, controller mode `mean` (default): standard LoRA
  forward (no noise) — also what `merge` uses. On a **merged base** (§3) the
  deterministic delta is skipped entirely (it is already folded into the base
  weight), making this a strict no-op.
- `self.training == False`, controller mode `sample`: low-rank trick (`d(x)`
  in R^r) — cheaper than element-wise when sampling repeatedly at inference.
  On a merged base only the zero-mean noise term is added (§3).

For `Conv3d`, `θ_A x` is the rank-r down-conv; the noise term needs
`(θ_A²)(x²)` — implement as a conv of `x²` with `θ_A²` (same
stride/padding), which is the exact diagonal-covariance propagation. Cover
with a unit test against a brute-force MC estimate on a tiny layer.

The shared target presets (`finetuning/targets.py`) apply unchanged. Their
1x1x1-Conv3d / grouped-conv exclusions exist because *peft's own* Conv merge is
broken (0.19, verified); BaLoRA's hand-written `merge` could in principle lift
that, but keep the presets and exclusions shared across variants —
`variant: standard ↔ balora` must stay a pure config flip.

### Inference network (α(x))

`class AlphaNet(nn.Module)`: pooled summary of the model input → MLP
(`d → 256 → 256 → L`) → softplus → per-layer `α`. Design decisions for our
setting (paper used frozen-backbone CLS tokens):

- Input summary: global-average-pool of the **normalized input state over
  fluid cells**, per channel, concatenated with normalized params →
  a ~`(C+P)`-vector. Cheap, permutation-safe, and available identically at
  train and rollout time. (Normalization happens inside our models — read the
  `state_mean/std`, `param_mean/std` buffers off the base model rather than
  re-deriving stats.)
- Ownership: the `BaLoRAController` (holding the `AlphaNet` and the ordered
  list of adapted layers) is registered as a **submodule of the `PeftModel`
  wrapper** (`peft_model.balora_controller`), attached after injection with
  its params trainable. This placement is load-bearing four ways:
  1. the fine-tune script's `trainable = [p for p in peft_model.parameters()
     if p.requires_grad]` picks the AlphaNet up (and grad clipping covers it);
  2. `checkpoint.pt` / `best_model_state` snapshot `_eager_model.state_dict()`
     (`training/base.py`), so resume and the best-epoch restore carry it — a
     side object outside the module tree would silently lose the AlphaNet on
     resume;
  3. `merge_and_unload()` returns the unwrapped *base* model, so controller
     keys can never leak into the merged `weights.pt` (the ESMDA-purity
     invariant from plan 01);
  4. `save_adapter` runs after `fit()` has restored the best-val state, so the
     exported AlphaNet is the best epoch, consistent with `weights.pt`.
- A pre-forward hook on the `PeftModel` computes `α(x)` once per model call
  and distributes `α_l` to each layer. The hook must reach the controller
  *through the module tree* (no external closures): `merge_to_state_dict`
  deepcopies the whole `PeftModel` (`finetuning/inject.py`), and a closure
  over an outside object would leave the copy's hook pointing at the
  original's controller.
- Persistence: PEFT's `save_pretrained` persists only the `lora_*` adapter
  tensors — it does **not** capture the AlphaNet. The balora path explicitly
  writes `adapter/alphanet.pt` and `adapter/balora_config.json` alongside the
  PEFT files (see §3 artifacts).

### Controller / public API

`class BaLoRAController(nn.Module)`: `set_eval_mode("mean"|"sample")`
(training behavior follows `self.training`, see above), `kl_loss()` (sum of
`last_kl` over layers), `set_member_noise(seed, member_offset, n_members)`
(frozen rank-r rollout noise, §3), `num_adapted_layers`.

`inject_lora(..., variant="balora", balora_cfg={prior_p, alphanet_hidden,
...})` from plan 01 keeps its implemented return type — a bare `PeftModel` —
with the controller reachable as `peft_model.balora_controller`. (A
variant-dependent `(peft_model, controller)` tuple would break every existing
call site in `finetune_neural_surrogate.py`; the submodule attachment makes it
unnecessary.)

## 2. Training integration (ELBO)

Small, opt-in extension of the existing trainers — not a new trainer class:

- `BaseTraining` gains an optional `aux_loss_fn: Callable[[], Tensor] | None`
  (default `None` → byte-identical behavior). When set, `_final_loss`'s
  result gets `+ kl_weight * aux_loss_fn()` **during training only**
  (`if self.model.training`; expose `kl_weight` for tempering, default 1.0).
  Scaling: because `q(ω_A|x)` is input-adaptive the KL is a *per-sample*
  quantity, so `kl_loss()` returns the batch-mean KL and is added to the
  batch-mean data loss with **no extra `1/N`** (the `1/n_train_samples`
  scaling applies only to an input-independent weight posterior — verify the
  paper's minibatch treatment before implementation and document the choice).
- **Validation stays the mean-model data loss.** `_validate` runs under
  `model.eval()`, which with the `self.training` keying above means: no
  sampling, no KL. `best_val`, early stopping and the merged
  `weights_transform` export are therefore deterministic and directly
  comparable to a standard-LoRA fine-tune of the same data.
- The fine-tune script passes
  `aux_loss_fn=peft_model.balora_controller.kl_loss` when
  `variant == "balora"`.
- Multiple model calls per batch: `finetuning.yaml` defaults
  `grad_unroll_steps: 2`, so `_forward` makes *two* gradient-bearing calls per
  batch (plus `no_grad` prefix calls if a pushforward curriculum is enabled).
  Convention: every call in training mode samples — the prefix included, so
  the unroll matches a sampled-weights deployment rollout (moot at the default
  `dataset.pushforward_steps: 1`) — and `kl_loss()` reads the `α` of the
  final, loss-bearing call (`last_kl` is simply the most recent write).
- Log the KL term through the existing `_aux_terms` mechanism
  (`training/base.py`), which already flows into `metrics.csv` — no new
  plumbing.
- AMP note: sampling uses `sqrt` of variances — keep the noise math in fp32
  inside the layer (autocast-exempt region) to avoid bf16 underflow of
  `θ_A²`.

Applies unchanged to both targets: plan-01 next-step P3D fine-tunes and
plan-03 DFT fine-tunes (the encoder/decoder adapters there are BaLoRA layers;
the fully-trained sub-network/γ modules stay deterministic MAP — the ELBO's
KL covers only the BaLoRA layers).

## 3. Inference design (the stochastic-model question)

### Artifacts

`weights.pt` = merged **mean** weights (deterministic mode) — ESMDA and all
existing tooling work unchanged, zero risk. `adapter/` additionally holds the
PEFT adapter (`θ_A, θ_B`), `alphanet.pt`, and `balora_config.json` (prior_p,
mode defaults, and the merged-base marker below). So a BaLoRA model degrades
gracefully to its mean model anywhere the adapter isn't loaded.

**The merged-base convention (correctness-critical).** The only base weights
in a fine-tuned `model_dir` are the *merged* ones — the unmerged pretrained
base is provenance only (`config.yaml`'s `pretrained:` block; the dir may be
gone, and the DFT path even stamps `skip_pretrained_load: true`). Re-attaching
the adapter on top of merged weights and running the standard LoRA forward
would add `θ_Bθ_A` **twice**. So sample mode loads `weights.pt` as-is and
attaches the adapter in a declared *merged-base* state (recorded in
`balora_config.json`): the layers skip the deterministic delta and contribute
**only the zero-mean noise term** `θ_B(sqrt(d(x)) ⊙ ε)` — mathematically
identical to mean-weights + sampled perturbation, with no extra artifacts.
(`eval_mean` on a merged base is then a strict no-op, which is exactly the
degrade-gracefully guarantee.)

### Modes in `NeuralSurrogateForwardModel`

New optional config block in `conf/model/neural_surrogate.yaml`
(default absent → current behavior). It arrives as a new optional constructor
kwarg (`balora: dict | None = None`) — the class takes explicit kwargs, not a
config blob — so `None` keeps the code path byte-identical:

```yaml
balora:
  mode: deterministic        # deterministic | sample
  resample: rollout          # rollout | step   (when mode: sample)
  seed: null                 # base seed; global member index is folded in
```

- `deterministic`: load `weights.pt` as today. Nothing else — `peft` is not
  even imported.
- `sample`: `_load_weights` additionally rebuilds the adapter (re-register
  custom modules → the extended `load_adapter`, §1) in merged-base state and
  sets eval mode `sample`. `peft` is the optional `[finetuning]` extra and the
  forward model does not currently import it — make it a lazy import with a
  clear "install neural_surrogates[finetuning]" error, since sample-mode DA
  runs now need it.
  - `resample: step`: fresh `ε_r` every model call — i.i.d. noise per step,
    the paper's UQ setting (S rollouts → predictive spread).
  - `resample: rollout` (**default for DA**): each member gets **frozen
    rank-r noise** — a per-member `ε_r ∈ R^r` drawn once per rollout from
    `seed + global_member_index` and held fixed across steps, applied via the
    low-rank term. Be precise about what this is: *not* a frozen weight-space
    sample `ω_A` — the perturbation magnitude `sqrt(α(x_t)·(θ_A²)(x_t²))`
    still varies with the state each step, and the paper's low-rank identity
    is exact only for fresh per-input noise, so freezing `ε_r` across steps is
    *our* extension of the method. Each member still integrates a fixed,
    self-consistent perturbed model, which is what ESMDA needs: BaLoRA's
    posterior over adapters becomes the ensemble's model-error term,
    replacing/adding to additive inflation.
- Ensemble wiring: `NeuralSurrogateEnsembleForwardModel.run_ensemble` calls
  the *shared* model's `rollout_batched` in the parent process
  (`ensemble_forward_model.py`) — members are already batched and the network
  never crosses the `forkserver` boundary, so batched noise shaped
  `(B_members, r)` inside the layer needs no cloning and no serialization
  care (`clone_for_member` shallow-shares `self.model`, which stays fine).
  One trap: `rollout_batched` splits members into `rollout_batch_size` chunks
  (default 10 in `conf/model/neural_surrogate.yaml`) and the layers only see
  chunk-local batches — thread the **global member offset** through
  `rollout_batched → _rollout_chunk` into `set_member_noise`, otherwise
  members 0 and 10 draw identical noise.
- Aggregation: for standalone UQ (outside DA) add a small
  `scripts/neural_surrogate/rollout_uncertainty.py` or a flag in the existing
  testing script: S rollouts → per-cell mean/std NetCDF. S default 32
  (paper used ~100 for scalar tasks; fields are expensive — make it a knob).

## 4. Tests

- Unit: mean-mode ≡ standard-LoRA forward (exact); **sampled-at-init ≡
  deterministic** (`θ_B = 0` zeroes the noise path too, so BaLoRA preserves
  plan-03's stepper==AE identity in every mode); `train`-mode expectation ≈
  mean-mode (MC test, loose tol); closed-form KL (including the `r·d` element
  count) matches a numerical `torch.distributions.kl_divergence` on sampled
  cases; Conv3d variance propagation vs brute-force MC; low-rank
  `eval_sample` covariance matches element-wise sampling covariance **for a
  fixed input** (the identity is per-input; it does not hold across a
  frozen-ε rollout).
- Integration: fine-tune smoke with `variant: balora` (2 epochs, fixture
  data) → deterministic export loads in the ESMDA path AND matches the
  sample-mode adapter path with noise forced to zero (this guards the
  merged-base delta skip); `mode: sample` rollout produces member-dependent,
  fixed-seed-reproducible trajectories; chunked ≡ unchunked (same members,
  same seeds, `rollout_batch_size` set vs `null`).
- No-op guard: absent `balora:` block in the model config ⇒
  `NeuralSurrogateForwardModel` code path byte-identical (existing e2e tests
  already enforce this de facto).

## 5. Open questions (defaults chosen)

- **Which layers get BaLoRA** in P3D: start with the same `attention` preset
  as standard LoRA (paper adapts attention+MLP projections in LLMs; QKV-only
  for ViTs). Rank per plan 01 (`r=32, alpha=64`) until swept.
- **`prior_p`**: paper's dropout-rate prior parameter; default 0.1, expose in
  config, sweep with calibration (spread/skill) on held-out rollouts.
- **ELBO bookkeeping vs the paper**: the KL closed form and the minibatch
  scaling in §0/§2 are derived from the stated `q`/prior — confirm both
  against the paper (and the numerical KL test) before freezing the trainer
  hook.
- **Interaction with `torch.compile`**: stochastic branches + per-batch α are
  recompile bait; fine-tuning already defaults `compile_model: false`
  (plan 01). For sampled *inference* measure first; eager is likely fine.
- **BaLoRA on the AE stage** (plan 02) is out of scope — pre-training is not
  parameter-efficient fine-tuning; uncertainty enters at the predictor stage.
