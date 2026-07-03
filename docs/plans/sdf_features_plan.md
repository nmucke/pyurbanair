# Implementation plan: SDF + ∇SDF geometry features for the P3D surrogate

*Design record, 2026-07-02. Follows from
[docs/multi_geometry_surrogate_research.md](../multi_geometry_surrogate_research.md)
(§3, phase 1). Scope: add a signed-distance field and its gradient as
geometry input channels to `P3D`, with dataloader-side computation during
training and model-side (cached, once-per-rollout) computation at inference.*

## 1. Goal and requirements

Replace "geometry = one binary mask channel" with "geometry = mask + 4
derived channels": the clamped signed distance function and its (unit)
gradient. Two hard requirements:

- **Training:** the features are supplied through the dataloader (computed
  once per geometry on CPU at dataset init, shipped once per batch like the
  mask) — never recomputed inside the training step.
- **Inference:** callers keep the existing contract — they pass only
  `(state, params, geometry)` and the model derives the SDF features
  itself. Because geometry is constant over a rollout, the model computes
  them **once at rollout start** and caches them for every subsequent step.

Non-goals (deferred, §8): other architectures (`UNetConvNeXt`, `UPT`,
`SimpleConv`), the `DomainDecomposed` wrapper, MDDF directional distances,
and the multi-geometry dataset work (separate plan; this design is written
so per-trajectory features slot in naturally later).

## 2. Feature definition

From the existing binary fluid mask `m` (`1` = fluid, `0` = solid,
buildings + ground), define on the voxel grid:

```
d_out = EDT(m)          # distance (in cells) to nearest solid, inside fluid
d_in  = EDT(1 - m)      # distance to nearest fluid, inside solid
sdf   = d_out - d_in    # > 0 in fluid, < 0 in solid, ~0 at the interface
```

The 4 feature channels, in tensor-layout order `(B, C, z, y, x)`:

| Channel | Definition | Range |
|---|---|---|
| `sdf_n` | `clip(sdf, -L, L) / L`, clamp radius `L` in **cells** (config knob, default 32) | `[-1, 1]` |
| `g_z, g_y, g_x` | central differences of the **unclamped** `sdf`, normalised to unit length (`g / max(‖g‖, ε)`); replicate edges | each `[-1, 1]` |

Decisions and rationale:

- **Cell units, not physical units.** The model only receives a voxel mask
  at inference — it has no grid spacing. Working in cells makes training
  and inference trivially consistent, and the strict domain check (grid +
  bounds equality; spacing equality for `domain_flexible`) already
  guarantees the trained cell spacing is the deployed cell spacing. An
  optional `spacing=(dz, dy, dx)` argument on the helper (mapped to
  scipy's `sampling`) is kept for future anisotropic-grid use but defaults
  off.
- **Gradient from the unclamped SDF** so direction information survives
  beyond the clamp radius (for exact EDT the gradient is a unit vector
  a.e. anyway; the explicit normalisation cleans up the finite-difference
  approximation and plateau cells). Sign convention: positive-in-fluid SDF
  means ∇SDF points **away from** the nearest wall.
- **No z-scoring.** All 4 channels are bounded in `[-1, 1]` by
  construction; like the mask they enter the stem raw and are excluded
  from `_compute_normalization_stats` (no change to that function).
- **No geometry masking of the features.** Inside solids the channels are
  well-defined (negative SDF) and carry "how deep into the building" —
  harmless given the output hard-mask, and it keeps the interface channel
  smooth.

Cost: `scipy.ndimage.distance_transform_edt` is `O(N)` in grid cells,
independent of building count. Two EDTs + one gradient on a 128³ grid is
tens of milliseconds on CPU — negligible once per geometry.

## 3. New module: `neural_surrogates/sdf.py`

Single source of truth used by *both* the dataset (training path) and the
model (inference path), so the two can never drift:

```python
def sdf_features(
    mask: torch.Tensor,          # (*grid,) or (B, *grid), 1 = fluid
    clamp_cells: float = 32.0,
    spacing: tuple[float, float, float] | None = None,
    eps: float = 1e-6,
) -> torch.Tensor:               # (4, *grid) or (B, 4, *grid), same dtype
```

- Accepts torch, computes via numpy/scipy on CPU, returns torch on the
  input's device/dtype (the one `.cpu()` round-trip is paid once per
  geometry, never per step).
- Batched input loops over members (see §5 for the shared-geometry
  shortcut).
- Exported from `neural_surrogates/__init__.py`.

## 4. Training path

**`TransitionDataset`** (`datasets/transition.py`):

- New ctor arg `sdf_features: bool = False` (+ `sdf_clamp_cells: float = 32.0`).
- When on, compute `self._geom_features = sdf_features(self._geometry, ...)`
  once in `__init__`, right after `_load_geometry`, and add
  `"geom_features": self._geom_features` to every `__getitem__` dict —
  same single-shared-tensor pattern as `geometry`. (When the multi-root
  dataset lands, this becomes per-trajectory alongside the per-trajectory
  mask; nothing in this design changes.)

**`transition_collate`:** ship `geom_features` once per batch as
`(1, 4, *grid)` exactly as it does `geometry` (pop from items, re-attach
`batch[0]`'s tensor unsqueezed).

**`BaseTraining`** (`training/base.py`):

- `_prepare_batch` additionally pulls `batch.get("geom_features")`, caches
  the device copy next to the existing `self._geometry` cache, and the
  train/val unroll passes it to the model:
  `self.model(state, params_i, geometry, geom_features=feat)` when the
  batch carries it; the call stays exactly as today when it doesn't. Guard:
  if the model advertises `n_geom_feature_channels > 0` but the batch has
  no features, raise with a pointer at `dataset.sdf_features` (fail loud,
  not silently-zero).
- Note: the existing first-batch geometry caching (base.py:251) is a known
  single-geometry assumption; this plan mirrors it deliberately (features
  cached the same way) so the multi-geometry fix later changes both in one
  place.

## 5. Inference path (model-side, cached)

**`P3D`** (`architectures/p3d.py`):

- New ctor args: `sdf_features: bool = False`, `sdf_clamp_cells: float = 32.0`.
  When on, `in_channels += 4` and expose `self.n_geom_feature_channels = 4`
  (0 otherwise — the trainer keys off this).
- `forward(state, params, geometry, geom_features=None, extra=None)`:
  - features provided (training) → use as-is (shape-check `C == 4`);
  - features absent and `sdf_features=True` (inference) → fetch from a
    small per-instance cache, computing on miss:

    ```python
    key = (geometry.data_ptr(), geometry.shape, geometry.device)
    ```

    This works because `NeuralSurrogateForwardModel.rollout_batched` builds
    the stacked geometry tensor **once**
    ([forward_model.py:785](../../libs/neural-surrogates/src/neural_surrogates/forward_model.py#L785))
    and passes the *same tensor object* to every step's model call
    (line 801) — so step 1 computes, steps 2..T hit the cache. Keep the
    cache size-1 (a rollout has one geometry; a new key evicts the old),
    mirroring the `UPT._geom_cache` precedent. Store it outside
    `state_dict` (plain attribute, not a buffer).
  - Batched geometry `(B, *grid)`: compute member 0, then reuse for members
    whose mask equals member 0 (`O(B·N)` compare, once per rollout) — the
    ensemble stacks identical masks per window, so this collapses B EDTs
    to 1. Fall back to per-member EDT when they differ.
  - Stem assembly order (documented in the docstring):
    `[state, geometry, sdf_n, g_z, g_y, g_x, (param channels), (extra)]`.
    Features are inserted **right after the mask** so the "geometry block"
    is contiguous; params/extra keep their relative positions.
- **Checkpoint compatibility:** default `sdf_features=False` keeps
  `in_channels` — and the entire state dict — byte-identical to today's
  models, per the repo's no-op-when-absent rule. Existing
  `model_weights/p3d_barcelona` continues to load and run untouched.

**`NeuralSurrogateForwardModel` / rollout & test scripts / ESMDA path:**
**zero changes.** They already pass `(state, params, geometry)`; the model
self-computes on the first step. This is the payoff of putting the
inference-side computation in the model rather than the wrapper.

**torch.compile:** during training the features always arrive from the
batch, so the compiled graph never contains the scipy path; the
`geom_features is None` check is a cheap dynamo guard on a stable branch.
The inference path (uncompiled) may compute freely. Do **not** call the
EDT inside a compiled region — if someone compiles an inference rollout
later, the cache lookup must sit in a thin uncompiled wrapper (note this
in the code comment).

## 6. Config and validation

- `conf/neural_surrogate/architectures/p3d/*.yaml`: add commented-off
  `# sdf_features: true` to the presets (default stays off); enable per
  run via CLI `architecture.sdf_features=true` or a new preset variant if
  it earns one.
- `conf/neural_surrogate/training.yaml`: `dataset.sdf_features: false`
  (+ `dataset.sdf_clamp_cells: 32`) documented next to `geometry_var`.
- `train_neural_surrogate.py`: after instantiating model + datasets,
  cross-check `getattr(model, "n_geom_feature_channels", 0) > 0` ⇔
  `train_ds` was built with `sdf_features=True`, and that the two
  `sdf_clamp_cells` values match — raise on mismatch. The resolved config
  saved to `model_weights/<name>/config.yaml` then carries both flags, so
  the forward model rebuilds the architecture with the right settings for
  free (its config-driven reconstruction path is untouched).

## 7. Tests

1. **`test_sdf_features.py`** — helper correctness on a hand-built box in a
   small grid: sign inside/outside, exact distances along axes, gradient
   unit-norm and direction (points away from the box), clamp behaviour,
   batched == looped, dtype/device round-trip.
2. **Dataset/collate** — items carry `(4, *grid)` features; collate ships
   `(1, 4, *grid)`; `sdf_features=False` items are byte-identical to
   today's (no new key).
3. **P3D wiring** — (a) `sdf_features=False` model: state dict keys and a
   fixed-seed forward output identical to pre-change (regression guard);
   (b) `sdf_features=True`: forward accepts provided features; (c)
   self-compute path: two calls with the same geometry tensor → EDT runs
   once (count via monkeypatch) and output equals the explicitly-provided
   path; new geometry tensor → recompute.
4. **End-to-end smoke** — extend the `surrogate_model_dir_factory` case in
   `tests/test_run_esmda.py` (or a lighter forward-model rollout test) with
   `sdf_features=true` to prove the untouched inference plumbing works.
5. **Trainer guard** — model-wants-features + dataset-without raises.

## 8. Follow-ups (out of scope here)

- **Other architectures:** `UNetConvNeXt`/`UPT` can adopt the same
  `n_geom_feature_channels` + `geom_features=` convention later; the
  trainer-side plumbing built here already serves them.
- **`DomainDecomposed`:** features must be computed **globally** and then
  tiled with the patches (a per-patch EDT would see patch borders as
  walls); wire through the same `restrict` path as state/geometry, and the
  fine net's widened stem. Deferred with the DD work.
- **Multi-geometry datasets:** per-trajectory features keyed like the
  per-trajectory mask; batch sampler keeps batches geometry-homogeneous so
  the ship-once collate survives (see the multi-geometry research doc,
  phase 0).
- **MDDF** (directional distances) as extra channels if ∇SDF saturates —
  same helper module, same plumbing, `n_geom_feature_channels = 4 + K`.

## 9. Suggested implementation order

1. `sdf.py` + unit tests (pure, no integration risk).
2. Dataset + collate + trainer plumbing (feature-flagged off by default).
3. P3D ctor/forward + cache + regression tests.
4. Config knobs + train-script validation + smoke test.
5. A/B experiment: retrain `p3d_barcelona` with `sdf_features=true`
   (fresh run, `init_weights_path=null` — the stem width changes) and
   compare val loss + rollout RMSE (`test_neural_surrogate.py`) against
   the current checkpoint. This is the go/no-go gate before the
   multi-geometry phases build on it.
