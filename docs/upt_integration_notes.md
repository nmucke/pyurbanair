# UPT Integration Notes (research + design spec)

Audience: the implementer who will write the `UPT` architecture module and its
Hydra preset configs for `pyurbanair`'s neural-surrogate library. This document
is the result of inspecting the UPT tutorial repo
(https://github.com/BenediktAlkin/upt-tutorial), the `kappamodules` package it
depends on, and the existing contract in
`libs/neural-surrogates/src/neural_surrogates/`.

The wrapper must satisfy the EXACT forward contract used by the other
architectures (`unet_convnext.py`, `simple_conv.py`):

```python
def forward(self, state: torch.Tensor, params: torch.Tensor, geometry: torch.Tensor) -> torch.Tensor
```
- `state`:    `(B, C, D, H, W)` velocity channels, regular grid `(D,H,W)=(nz,ny,nx)`.
- `params`:   `(B, P)` scalar inflow params.
- `geometry`: `(B, D, H, W)` or `(B, 1, D, H, W)` binary mask, 1 = fluid, 0 = obstacle.
- returns:    `(B, C, D, H, W)` predicted next state on the SAME regular grid.

Constructor must accept at least `n_state_channels: int` and `n_params: int` as
the first kwargs; everything else has defaults (Hydra does
`instantiate(cfg.architecture, n_state_channels=..., n_params=...)`).

---

## A. Vendor-vs-install decision + exact file/dep list

### Decision summary

| Component | Source of the tutorial impl | Decision |
|---|---|---|
| Transformer blocks (`PrenormBlock`, `PerceiverBlock`, `PerceiverPoolingBlock`, `VitBlock`, `DitBlock`, `DitPerceiverBlock`, `DitPerceiverPoolingBlock`) | `kappamodules.transformer` / `kappamodules.vit` | **pip install `kappamodules`** |
| `LinearProjection`, `Sequential`, `ContinuousSincosEmbed` | `kappamodules.layers` | **pip install `kappamodules`** |
| `Approximator`, `DecoderPerceiver`, `EncoderSupernodes`, `ConditionerTimestep` | `upt/models/*.py` | **vendor** (tiny glue around kappamodules) |
| `SupernodePooling` (`radius_graph` + `segment_csr`) | `upt/modules/supernode_pooling.py` | **vendor + REWRITE in pure torch** (drop `torch_geometric` / `torch_scatter`) |
| `einops` | used throughout | **pip install `einops`** |

Rationale:

- `kappamodules` (latest 0.1.112) is **pure-torch**: its only runtime deps are
  `torch`, `einops`, `numpy`. I installed it in a throwaway venv and exercised
  every block UPT needs (`ContinuousSincosEmbed`, `PrenormBlock`,
  `PerceiverPoolingBlock`, `PerceiverBlock`, `VitBlock`, `DitBlock`) — all
  import and run. So we install it rather than vendoring ~dozen block files.
- `einops` is NOT currently in the env and is needed by both kappamodules and
  the vendored UPT modules. Add it explicitly.
- The ONLY heavy/fragile dependency is the supernode-pooling graph op:
  `torch_geometric.nn.pool.radius_graph` + `torch_scatter.segment_csr`. Neither
  is installed. `torch_scatter` ships as wheels compiled against an exact
  torch ABI; the project is on **torch 2.10.0**, for which PyG's wheel index
  (`https://data.pyg.org/whl/`) has **no prebuilt wheels** (it tops out well
  below 2.10), so installing `torch_scatter` would force a source compile —
  exactly the kind of brittle native build we want to avoid in a research repo
  run through pixi. Since `SupernodePooling` only does a radius graph + mean
  aggregation over a SHARED, small, regular point set, it is trivial to
  reimplement in ~30 lines of pure torch (prototyped and verified differentiable
  during research). **Vendor + rewrite it; do NOT add torch_geometric/torch_scatter.**
- The four UPT model files (`approximator.py`, `decoder_perceiver.py`,
  `encoder_supernodes.py`, `conditioner_timestep.py`) are thin wiring over
  kappamodules (≈ 65 / 127 / 124 / 28 lines). Vendor them so we control the
  forward signatures and can strip unused branches (image unbatch modes, etc.).

### Exact files to create under `libs/neural-surrogates/src/neural_surrogates/architectures/_upt/`

```
_upt/
  __init__.py
  supernode_pooling.py    # PURE-TORCH rewrite (see §A.1)
  encoder.py              # EncoderSupernodes  (vendored from upt/models/encoder_supernodes.py)
  approximator.py         # Approximator       (vendored from upt/models/approximator.py, verbatim)
  decoder.py              # DecoderPerceiver   (vendored from upt/models/decoder_perceiver.py,
                          #                      keep only unbatch_mode="dense_to_sparse_unpadded";
                          #                      our wrapper actually keeps it DENSE — see §B)
```
The public wrapper class `UPT(nn.Module)` lives in
`architectures/upt.py` (sibling of `unet_convnext.py`) and imports from `_upt`.
Register it in `architectures/__init__.py` and the top-level
`neural_surrogates/__init__.py` next to `UNetConvNeXt` so
`_target_: neural_surrogates.UPT` resolves (matches how `UNetConvNeXt` is
exported — see `forward_model.py` which references `neural_surrogates.UNetConvNeXt`).

The vendored files should keep their kappamodules imports
(`from kappamodules.layers import ...`, `from kappamodules.transformer import ...`,
`from kappamodules.vit import VitBlock`). Only `supernode_pooling.py` changes its
imports (drop torch_geometric/torch_scatter). Add a short header comment in each
vendored file citing the upstream path + commit and the MIT/equivalent license of
upt-tutorial.

### Exact dependencies to add

In `pyproject.toml` under
`[tool.pixi.feature.neural-surrogates.pypi-dependencies]` (alongside the editable
`neural-surrogates` entry):

```toml
einops = ">=0.7"
kappamodules = ">=0.1.112"
```

`kappamodules` and `einops` are pure-Python/torch and resolve from PyPI. **Do
NOT add `torch_geometric` or `torch_scatter`.** No conda-channel pins are
needed. (Note: the pixi `dev` env has no `pip`/`python -m pip`; deps must be
declared in `pyproject.toml` and resolved by `pixi`, not pip-installed
ad hoc.)

### A.1 The pure-torch `SupernodePooling` rewrite (drop torch_geometric/torch_scatter)

Upstream does, on a *sparse* `(B*N, ·)` layout: build a radius graph over all
input points, keep only edges whose destination is a supernode, message-pass
(`concat[src, dst] -> MLP`), and `segment_csr(..., reduce="mean")` to aggregate
each supernode's incoming messages, then reshape to `(B, num_supernodes, dim)`.

Because in our setting the point set (fluid cells) is **identical across the
batch** (geometry is shared — see §C), we operate on a SINGLE point set of `N`
fluid points and batch only the features. Replace the graph op with dense
neighbor selection computed once per forward:

```
# pos: (N, ndim) normalized fluid-cell coords (shared across batch)
# supernode_pos = pos[supernode_idxs]            # (S, ndim)
# feat: (B, N, hidden) per-sample embedded point features
#
# 1. neighbor index set per supernode (precompute once; cache on the geometry):
#    dist = torch.cdist(supernode_pos, pos)      # (S, N)
#    within = dist <= radius                      # self-loop guaranteed (dist=0)
#    for each supernode keep up to max_degree nearest True entries
#    -> a padded (S, max_degree) LongTensor `nbr_idx` + a (S, max_degree) bool `nbr_mask`
# 2. messages: m = MLP(concat([feat[:, nbr_idx], feat[:, supernode_idxs, None]])) # broadcast dst
#    shape (B, S, max_degree, hidden)
# 3. masked mean over the max_degree axis using nbr_mask -> (B, S, hidden)
```

Notes:
- Self-loop is always present (a supernode is within radius 0 of itself), so no
  empty-neighborhood division-by-zero — but still guard the mean with the mask
  count clamped to >= 1.
- `cdist` over `S x N` with `S = num_supernodes` (≈ 64–256) and `N` = number of
  fluid cells (≤ D·H·W; for the 4×8×8 test grid N ≤ 256) is negligible.
- Precompute `nbr_idx`/`nbr_mask` ONCE per forward from the geometry and reuse
  for every autoregressive step / every batch element. Optionally cache keyed by
  `id(geometry)`/mask-hash so repeated rollout calls skip the recompute (pure
  perf; not required for correctness).
- This is fully differentiable w.r.t. `feat` (verified in a prototype). The
  neighbor selection (argsort/topk on distances) is constant per geometry, so no
  gradient flows through it — correct, matching the original (graph topology is
  not learned).
- Keep the constructor signature compatible:
  `SupernodePooling(radius, max_degree, input_dim, hidden_dim, ndim)` with
  `input_proj = LinearProjection(input_dim, hidden_dim)`,
  `pos_embed = ContinuousSincosEmbed(dim=hidden_dim, ndim=ndim)`,
  `message = Seq(LinearProjection(2*hidden, hidden), GELU, LinearProjection(hidden, hidden))`,
  and `x = input_proj(feat) + pos_embed(pos)` before messaging, exactly as upstream.
  Change only `forward` to take dense `(B, N, input_dim)` feats + the precomputed
  neighbor tensors instead of `(supernode_idxs, batch_idx)` sparse args.

---

## B. The wrapper class design

### Constructor

```python
class UPT(nn.Module):
    def __init__(
        self,
        n_state_channels: int,      # C  (Hydra-injected)
        n_params: int,              # P  (Hydra-injected)
        # --- latent / token sizing ---
        dim: int = 192,             # latent token dim (enc_dim == approx dim == dec dim)
        num_latent_tokens: int = 64,
        # --- supernodes (input pooling) ---
        num_supernodes: int = 64,
        radius: float = 2.5,        # in NORMALIZED grid units (see §B positions)
        max_degree: int = 16,
        gnn_dim: int = 96,          # supernode-pooling hidden dim
        # --- depths / heads ---
        enc_depth: int = 2,
        approx_depth: int = 4,
        dec_depth: int = 2,
        num_heads: int = 3,
        # --- conditioning ---
        cond_dim: int | None = None,  # if None, params injected via per-point feature concat
        ndim: int = 3,
    ) -> None:
```

`dim` and `num_heads` must satisfy `dim % num_heads == 0`. `ndim` is the spatial
dimensionality (3 for these grids). All non-`n_*` args have defaults so Hydra
presets only override what differs.

Submodules (built from the vendored `_upt` files + kappamodules):
- `self.pos_embed_in = ContinuousSincosEmbed(dim=gnn_dim, ndim=ndim)` is inside
  `SupernodePooling`; the encoder/decoder also embed positions internally.
- `self.encoder = EncoderSupernodes(input_dim=feat_dim, ndim=ndim, radius=radius,
   max_degree=max_degree, gnn_dim=gnn_dim, enc_dim=dim, enc_depth=enc_depth,
   enc_num_heads=num_heads, perc_dim=dim, perc_num_heads=num_heads,
   num_latent_tokens=num_latent_tokens, cond_dim=cond_dim)`
- `self.approximator = Approximator(input_dim=dim, depth=approx_depth,
   num_heads=num_heads, dim=dim, cond_dim=cond_dim)`
- `self.decoder = DecoderPerceiver(input_dim=dim, output_dim=n_state_channels,
   ndim=ndim, dim=dim, depth=dec_depth, num_heads=num_heads,
   perc_dim=dim, perc_num_heads=num_heads, cond_dim=cond_dim,
   unbatch_mode="dense_to_sparse_unpadded")`  (we keep decoder output DENSE — see below)

where `feat_dim = n_state_channels + (n_params if cond_dim is None else 0)`.

**Conditioning choice.** Two viable options; pick (1) for simplicity and to
avoid the timestep-conditioner machinery:

1. **Concatenate `params` to per-point features (recommended default,
   `cond_dim=None`).** Each fluid point's input feature is
   `[u, v, w, p_1, ..., p_P]` (params broadcast to every point). Then encoder /
   approximator / decoder all run WITHOUT DiT conditioning (`cond_dim=None` ->
   `PrenormBlock` / `VitBlock` / `PerceiverBlock`). This is the cleanest mapping
   onto our contract (no timestep token; the surrogate is a pure one-step
   `state,params -> next_state` map). `feat_dim = C + P`.
2. **DiT conditioning (`cond_dim` set).** Project `params` `(B,P)` through a small
   MLP to `(B, cond_dim)` and pass as `condition=` to encoder/approximator/decoder
   (which then use `DitBlock` / `DitPerceiverBlock`). Do NOT use the upstream
   `ConditionerTimestep` — it embeds an integer timestep, which we don't have
   (our one-step model is autoregressive at the forward-model level, not inside
   the net). If you enable this path, add `self.param_proj = nn.Sequential(
   nn.Linear(P, cond_dim*?), nn.SiLU(), ...)` to produce the condition.

   Default to (1); expose `cond_dim` so (2) is available without code changes.

### Position handling

- Build normalized coordinates for ALL grid cells once: for a `(D,H,W)` grid,
  `coords[z,y,x] = (z/(D-1), y/(H-1), x/(W-1))` (or center-normalized; just be
  consistent), flattened to `(D*H*W, ndim)` with `ndim=3`, ordered as
  `(z, y, x)` so it round-trips to the `(D,H,W)` layout. Scale so that a
  one-cell spacing is ~1.0 unit, then `radius` is interpretable in cell units
  (e.g. `radius=2.5` ≈ a 5-cell-wide neighborhood). Concretely, use integer
  index coords `(z, y, x)` directly (spacing 1.0) — simplest and makes `radius`
  literally "cells".
- **Input points** = fluid cells only: `fluid_idx = geometry.flatten().nonzero()`
  (computed from the shared mask). `input_pos = coords[fluid_idx]` `(N, 3)`.
- **Supernodes**: choose `num_supernodes` indices among the fluid points. Use a
  DETERMINISTIC farthest-point or strided selection (NOT random — see §D
  determinism). Simplest deterministic rule: evenly strided over `fluid_idx`
  (`fluid_idx[:: max(1, N // num_supernodes)][:num_supernodes]`). If
  `N < num_supernodes`, clamp `S = min(num_supernodes, N)`. Supernode count must
  be constant within a forward so the perceiver pooling sees a fixed token grid;
  it is, because geometry is shared.
- **Decoder query positions** = the SAME fluid-cell positions (`input_pos`), so
  the decoder predicts a feature at every fluid cell. Pass them as
  `output_pos` shaped `(B, N, 3)` (broadcast the shared `(N,3)` to batch).
- **Scatter back to the grid**: allocate `out = zeros(B, C, D, H, W)`; write the
  decoder's per-fluid-point predictions `(B, N, C)` into `out` at `fluid_idx`
  (obstacle cells stay 0, matching the data convention where obstacle cells are
  0 and the geometry mask is the fluid indicator). Use
  `out.view(B, C, D*H*W)[:, :, fluid_idx] = pred.transpose(1,2)` (or
  `index_copy`/scatter). Multiply by the mask at the end for safety so obstacle
  cells are exactly 0.

  IMPORTANT: keep the decoder DENSE per sample. Upstream's
  `dense_to_sparse_unpadded` rearranges `(B, N, C) -> (B*N, C)`; our wrapper does
  NOT want the cross-batch flatten because we scatter per-sample. Either pass
  `unbatch_mode="dense_to_sparse_unpadded"` and immediately
  `rearrange(..., "(b n) c -> b n c", b=B)`, or (cleaner) edit the vendored
  decoder to return the dense `(B, N, C)` directly. Since `N` is identical for
  all batch members (shared geometry), there is no padding ambiguity.

### Forward-pass pseudocode

```python
def forward(self, state, params, geometry):
    # --- normalize geometry to (B, D, H, W) ---
    if geometry.dim() == state.dim():            # (B,1,D,H,W) -> (B,D,H,W)
        geometry = geometry.squeeze(1)
    B, C, D, H, W = state.shape

    # --- shared fluid point set (geometry identical across batch; see §C) ---
    mask0 = geometry[0]                           # (D,H,W); assert all equal (see §D)
    coords = self._grid_coords(D, H, W, device, dtype)   # (D*H*W, 3) cached
    fluid_idx = mask0.reshape(-1).nonzero(as_tuple=False).squeeze(1)   # (N,)
    input_pos = coords[fluid_idx]                 # (N, 3)
    supernode_local = self._select_supernodes(input_pos)  # indices into [0,N)
    nbr_idx, nbr_mask = self._neighbors(input_pos, input_pos[supernode_local])

    # --- per-point input features (B, N, feat_dim) ---
    flat = state.reshape(B, C, -1)[:, :, fluid_idx].transpose(1, 2)   # (B, N, C)
    if self.cond_dim is None:                     # concat params to every point
        feats = torch.cat([flat, params[:, None, :].expand(B, flat.size(1), -1)], dim=-1)
        condition = None
    else:
        feats = flat
        condition = self.param_proj(params)       # (B, cond_dim)

    # --- encode -> approximate -> decode ---
    latent = self.encoder(feats, input_pos, supernode_local, nbr_idx, nbr_mask,
                          condition=condition)     # (B, num_latent_tokens, dim)
    latent = self.approximator(latent, condition=condition)
    query_pos = input_pos[None].expand(B, -1, -1)  # (B, N, 3)
    pred = self.decoder(latent, query_pos, condition=condition)   # (B, N, C) dense

    # --- scatter back to grid; obstacles stay 0 ---
    out = state.new_zeros(B, C, D * H * W)
    out[:, :, fluid_idx] = pred.transpose(1, 2)    # (B, C, N) -> placed
    out = out.reshape(B, C, D, H, W) * geometry.unsqueeze(1)
    return out
```

(The encoder signature above assumes the vendored `EncoderSupernodes.forward` is
adapted to take the dense `(B,N,feat)` + precomputed neighbor tensors instead of
the sparse `(input_feat, input_pos, supernode_idxs, batch_idx)` layout. Adapt
`EncoderSupernodes` and `SupernodePooling` forwards together; the transformer /
perceiver tail is unchanged.)

### Differentiability + repeated calls

- The full path (feature gather, supernode mean-aggregation, transformer/
  perceiver blocks, decoder cross-attention, scatter) is differentiable w.r.t.
  `state` and `params`. The final `* geometry` is a constant mask multiply.
  Verified the pooling + dense gather/scatter pattern is differentiable in a
  prototype.
- No internal state is mutated across calls (any neighbor cache is keyed on the
  geometry and read-only after build), so it supports arbitrary batch sizes and
  repeated autoregressive calls under `no_grad` (as `rollout_batched` and the
  trainer's pushforward unroll require). The scatter uses index assignment on a
  fresh tensor each call.

---

## C. Confirming shared geometry across the batch

This is the load-bearing assumption that lets us use ONE point set + batched
features. Confirmed from the contract code:

- `forward_model.py::rollout_batched`:
  `geom = torch.stack([self._build_geometry(t) for t in templates], dim=0)` —
  each `_build_geometry` voxelizes the SAME `stl_path` (or the same nonzero
  convention) onto the SAME trained grid, so all members' masks are identical;
  only `state`/`params` differ per member. The whole point of `rollout_batched`
  is that members share the trained network and grid.
- `data.py::TransitionDataset`: `self._geometry` is a SINGLE tensor returned for
  every `__getitem__` ("Same tensor for every item"); the default collate stacks
  identical masks into `(B, D, H, W)`.
- `training.py`: passes that stacked `geometry` straight through.

So within any single forward call the per-sample masks are identical.

**Design rule:** use `geometry[0]` as the shared mask and build the fluid index
set / supernodes / neighbor graph from it. **Add a cheap guard** (in `forward`,
under an `if self.training`-independent check or a one-time assert) that
`geometry` is constant across the batch, e.g.
`assert torch.equal(geometry, geometry[0].expand_as(geometry))` — or, to be
robust and avoid per-step cost in long rollouts, check only that each sample's
fluid-cell COUNT matches `geometry[0]`'s. If the assumption is ever violated, the
correct fallback is a per-sample Python loop over the batch (build point set per
sample, run the net, scatter) — document this as the explicit, slower fallback
path. For all current call sites the shared-geometry fast path applies.

---

## D. Gotchas / test-compatibility notes

1. **`test_rollout_batched_matches_per_member` — NO cross-sample leakage.**
   The batched rollout output for member `b` must byte-match
   (`rtol=1e-5, atol=1e-5`) rolling member `b` alone. Therefore the network must
   process each batch element independently:
   - All attention (encoder transformer, perceiver pooling, approximator,
     decoder cross-attention) operates within a sample's own token set — the
     kappamodules blocks attend within `(B, seq, dim)` along `seq` only, with no
     cross-`B` mixing. Good (standard attention does not mix the batch axis).
   - The supernode pooling must NOT mix points across samples. In our dense
     `(B, N, ·)` rewrite, aggregation is along the neighbor axis per sample, so
     no cross-sample mixing. (The upstream sparse version used `batch_idx` +
     `radius_graph(batch=...)` precisely to prevent cross-sample edges; our
     dense-per-sample layout gets this for free.)
   - Do NOT introduce BatchNorm anywhere (kappamodules blocks use LayerNorm —
     good; keep it that way). Any norm that couples the batch would break the
     equality test.
   - Supernode selection and the neighbor graph must be DETERMINISTIC and
     independent of batch size (derive purely from the shared geometry). Use the
     strided / farthest-point rule, NOT `torch.randperm`. A random selection
     would still be per-forward-consistent but would make the test flaky across
     the two calls only if re-seeded differently — avoid randomness entirely so
     the single-member and batched calls pick identical supernodes.

2. **`tiny` preset must run on CPU on a 4×8×8 grid in a unit test.**
   `N <= 256` fluid cells. Keep `tiny` small: `dim=32`, `num_latent_tokens=16`,
   `num_supernodes=16`, `num_heads=2`, depths `1/1/1`, `gnn_dim=16`,
   `radius=2.5`, `max_degree=8`. The test helper `_architecture()` builds the
   instance directly with `n_state_channels=3, n_params=2`; mirror that — make
   sure `UPT(n_state_channels=3, n_params=2, **tiny)` constructs and a forward on
   `(2,3,4,8,8)` returns `(2,3,4,8,8)` with finite values.

3. **`num_supernodes` / `num_latent_tokens` vs tiny grids.** With `N` fluid cells
   possibly `< num_supernodes`, clamp `S = min(num_supernodes, N)`. The perceiver
   pooling's query count is `num_latent_tokens` (independent of `N`/`S`), so it
   is safe even when `N` is tiny; only the supernode count needs clamping.

4. **`radius` must guarantee non-empty neighborhoods.** With integer cell coords
   (spacing 1.0), `radius >= 1.0` guarantees each supernode reaches its
   6-neighborhood; the self-loop guarantees at least one neighbor regardless.
   Defaults use `radius=2.5`. If a user sets `radius < 1`, only self-loops
   survive (still valid, degenerate). Clamp the masked-mean denominator to
   `>= 1`.

5. **`ContinuousSincosEmbed` dim divisibility.** It pads when `dim % ndim != 0`,
   so any `dim`/`gnn_dim`/`perc_dim` works, but prefer `dim` divisible by
   `ndim*2` (=6 for 3D) to avoid wasted padding channels. The preset `dim`
   values below are chosen divisible by 6 where convenient.

6. **`dim % num_heads == 0`** is required by attention. All presets respect it.

7. **dtype / device.** `forward_model.py` moves the model to `device`/`dtype` and
   calls under `no_grad`. Build `coords` and neighbor tensors on
   `state.device`/`state.dtype` lazily inside `forward` (cache by
   `(D,H,W,device,dtype)`), so the module follows `.to(...)` without registering
   huge buffers. Index tensors (`fluid_idx`, `nbr_idx`, `supernode_local`) are
   `long` regardless of model dtype.

8. **`einops` newly required.** Vendored decoder/pooling and kappamodules import
   `einops`; ensure it's in the dependency set (it is not currently installed).

9. **No `pip` in the pixi env.** Don't try `python -m pip install` at runtime;
   declare `kappamodules` + `einops` in `pyproject.toml` and let pixi resolve.
   (During research these were validated in a throwaway `uv` venv only.)

10. **License/provenance.** upt-tutorial and KappaModules are MIT-licensed by
    Benedikt Alkin. Add a header to each vendored file crediting the source
    file + repo + commit and the MIT license; consider a `_upt/LICENSE` copy.

---

## E. Preset hyperparameters (Hydra configs)

Create `conf/neural_surrogate_architectures/upt/{tiny,small,medium,large,xlarge}.yaml`
mirroring the `unet_convnext/` preset pattern. Each sets `_target_:
neural_surrogates.UPT` and overrides only the sizing fields (`n_state_channels` /
`n_params` are injected by `instantiate`). Suggested values:

| preset  | dim | num_latent_tokens | num_supernodes | gnn_dim | enc_depth | approx_depth | dec_depth | num_heads | radius | max_degree |
|---------|-----|-------------------|----------------|---------|-----------|--------------|-----------|-----------|--------|------------|
| tiny    | 32  | 16                | 16             | 16      | 1         | 1            | 1         | 2         | 2.5    | 8          |
| small   | 96  | 32                | 32             | 48      | 2         | 2            | 2         | 3         | 2.5    | 16         |
| medium  | 192 | 64                | 64             | 96      | 2         | 4            | 2         | 6         | 3.0    | 16         |
| large   | 384 | 128               | 128            | 192     | 3         | 6            | 3         | 6         | 3.0    | 24         |
| xlarge  | 768 | 256               | 256            | 384     | 4         | 8            | 4         | 12        | 4.0    | 32         |

Example `tiny.yaml`:

```yaml
# UPT — tiny preset. Smoke-test sizing; runs on CPU on a 4x8x8 grid.
_target_: neural_surrogates.UPT
dim: 32
num_latent_tokens: 16
num_supernodes: 16
gnn_dim: 16
enc_depth: 1
approx_depth: 1
dec_depth: 1
num_heads: 2
radius: 2.5
max_degree: 8
# cond_dim: null   # default: params concatenated to per-point features
```

Notes on the table:
- `cond_dim` omitted (None) on all presets -> params-as-feature-concat path
  (recommended default). To try DiT conditioning, add e.g. `cond_dim: 192`.
- `dim` divisible by `num_heads` everywhere; medium+ also divisible by 6 for
  clean sincos embedding.
- `num_supernodes`/`num_latent_tokens` scale with capacity; `tiny` is sized so a
  4×8×8 (N≤256) forward is sub-second on CPU.

---

## F. Quick build sketch of the vendored modules (for the implementer)

- `approximator.py`: copy verbatim from `upt/models/approximator.py`. No changes
  (it's pure kappamodules + a `condition` kwarg).
- `decoder.py`: copy `upt/models/decoder_perceiver.py`; drop the `"image"`
  unbatch branch (and the `math` import) and make the default path return the
  DENSE `(B, N, C)` tensor (skip the final `(b n) c` rearrange) — our wrapper
  scatters per sample.
- `encoder.py`: copy `upt/models/encoder_supernodes.py`; change its `forward` to
  accept the dense `(B, N, feat)` features + precomputed
  `supernode_local, nbr_idx, nbr_mask` and call the rewritten
  `SupernodePooling`. Keep `enc_proj` / `blocks` / `perceiver` tail unchanged.
- `supernode_pooling.py`: rewrite per §A.1 (pure torch; no
  torch_geometric/torch_scatter). Keep `input_proj`, `pos_embed`, `message`
  submodules identical so weights/semantics match upstream.
- `conditioner_timestep.py`: do NOT vendor unless the DiT-conditioning path
  (option 2) is implemented with a real timestep; the recommended default
  (params-concat) doesn't need it. If DiT is desired, build a small
  `param -> cond_dim` MLP in the wrapper instead.
