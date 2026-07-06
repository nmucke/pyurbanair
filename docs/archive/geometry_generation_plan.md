# Scoping: procedural city geometries → STL for multi-geometry training data

*Design record, 2026-07-02. Third piece of the multi-geometry effort (see
[multi_geometry_surrogate_research.md](../multi_geometry_surrogate_research.md)
phase 3 and [sdf_features_plan.md](sdf_features_plan.md)). Goal: automatically
generate synthetic urban geometries as `.stl` files consumable by all three
conventional backends (pylbm, pyudales, pypalm), so `generate_training_data.py`
can produce per-geometry training datasets at scale.*

## 1. Requirements (from discussion)

- Three **complexity tiers**:
  - `simple` — equal x/y footprints (cubes/square towers), varying heights;
    Xie & Castro-like staggered/aligned arrays.
  - `medium` — rectangular footprints with varying x/y extents and heights.
  - `advanced` — complex footprints including **inner courtyards open to the
    sky** (wind can enter from above), Barcelona-like perimeter blocks.
- **No overlapping buildings**, ever.
- **Realistic streets of varying width** between buildings.
- **Placement modes**: `grid` (regular lattice, optionally staggered) or
  `random`.
- Output: one binary **STL in the domain frame** per geometry (the shared
  single-STL contract all backends consume via `geometry.stl_path`).

Literature precedent (verified in the earlier research pass): every published
multi-geometry urban-flow result trains on procedurally generated layouts —
randomized building arrays (AB-SWIFT, arXiv:2603.25635), 10k procedural 2D
layouts (arXiv:2603.21210), 163 synthetic geometries (Clarke et al. 2025),
UrbanTALES (arXiv:2510.27101). Random block/array generation with controlled
density is the standard recipe; nobody needs more than that to get
generalization moving. *(The killed follow-up research pass would have pulled
exact per-paper generation recipes; treat that as an optional refinement, not
a blocker — the approach below is standard urban-canopy practice.)*

## 2. General approach: 2D layout → extrude → STL

Two clean stages, both in already-available dependencies (`shapely` comes with
`trimesh`'s ecosystem; `trimesh>=4.10` is a root dependency and all three
backends depend on it):

**Stage A — 2D layout (shapely, pure geometry).** Work on the domain footprint
`[0, Lx] × [0, Ly]` in metres.

1. **Street skeleton first, buildings second** — streets are then guaranteed
   by construction, no rejection sampling needed for the primary structure:
   - `grid` mode: a lattice of rows/columns where **each street's width is
     sampled** from a configured range (so streets vary but stay axis-aligned);
     optional stagger offset per row reproduces the Xie & Castro pattern.
   - `random` mode: **recursive block subdivision** (BSP-style): start from
     the whole footprint minus a boundary margin, repeatedly split the largest
     block at a random position with a random street width, stop when blocks
     reach a target size range. Simple (~50 lines), produces realistic
     irregular blocks and naturally varying street widths, and is inherently
     overlap-free.
2. **Building synthesis inside each block**, per tier:
   - `simple`: one square footprint of fixed side `b` per lattice cell (grid
     mode) or per block (random mode), height sampled per building.
   - `medium`: rectangle with sampled x/y extents (bounded by the block minus
     a sampled setback), height sampled.
   - `advanced`: perimeter block — the block polygon shrunk by a street
     setback, with a **courtyard hole** cut by an inward buffer
     (`footprint.buffer(-depth)` → shapely `Polygon` with an interior ring);
     probability mix with L/U-shapes (rectangle minus corner rectangle) and
     plain rectangles. Courtyards open to the sky fall out of the extrusion
     for free (2.5D: a hole in the footprint is a hole at every z).
   - Heights: lognormal (the standard urban-canopy distribution) with
     configurable mean/σ, clipped to `[h_min, h_max]`.
3. **Overlap guarantee**: in both modes buildings live strictly inside
   disjoint blocks/cells with positive setbacks — non-overlap and minimum
   street width hold by construction. A final `STRtree` pairwise check runs
   as an assertion, not a mechanism.
4. **Morphology targets** (the knobs urban meteorology actually uses): plan
   area density λ_p = built-area fraction (Xie & Castro's array is λ_p = 0.25)
   is monitored and reported; the generator resamples or rescales setbacks to
   land within a configured λ_p band, so density is controlled rather than
   emergent.

**Stage B — 3D mesh (trimesh).**

1. `trimesh.creation.extrude_polygon(footprint, height)` per building —
   handles interior rings (courtyards) via its triangulation backend.
2. Append a **flat ground sheet at z = 0** covering the domain footprint —
   verified convention: both existing case STLs carry one (Xie-Castro: 438
   z=0 faces; Barcelona: a flattened ground merged by
   `tools/prepare_case_stl.py` — "open streets in PALM, a strippable z=0
   sheet for pylbm, a standard flat floor for uDALES").
3. `trimesh.util.concatenate(...)` and export **binary STL**. No boolean
   union needed: buildings never touch (positive street widths), and the
   existing STLs prove the voxelizers accept multi-body, non-watertight
   meshes (Barcelona: 314 bodies, not watertight, 347k faces). Rectilinear
   extrusions keep facet counts trivial (Xie-Castro scale: ~1k faces), which
   also keeps uDALES facet processing fast (`nfcts` = face count).

## 3. Backend + pipeline constraints the generator must respect

These come from the repo's configs, tools and accumulated memory:

| Constraint | Consequence for the generator |
|---|---|
| Single-STL-in-domain-frame contract (`tools/prepare_case_stl.py` docstring): geometry spans `(0,0,0) → (Lx, Ly, ·)`, floor at z=0, consistent winding | Generate directly in the domain frame; `mesh.fix_normals()` before export |
| pylbm SIGFPE when domain z < building heights (memory) | Hard cap `h_max ≤ margin · Lz` (default ~0.6·Lz), assert before export |
| PALM topography is a top-down height map (memory: Barcelona ground issue) | 2.5D extrusions only (no overhangs) — courtyards are fine (height-map representable); flat ground at exactly z=0, never elevated/merged terrain |
| uDALES is y-periodic (spanwise) | Option `clear_periodic_y: true` — keep a street of at least the boundary margin at y=0/y=Ly (default), or tile footprints periodically (later) |
| Open inflow region in x (xie case: array starts at x=5, domain from x=−20) | Configurable upstream/downstream margins along x |
| Voxelization cleanliness + multi-geometry training wants one shared grid spacing (research doc phase 0) | Snap all footprint edges and street positions to multiples of (dx, dy); one fixed domain/grid shared by all generated cities (also keeps P3D's divisible-by-16 grid rule satisfied once) |
| uDALES precomputed IBM geometry is grid- *and* geometry-specific (barcelona.yaml) | Small facet counts make on-the-fly preprocessing cheap; no precomputed bundle needed per city |
| `obs` sensors must sit outside buildings (case yamls) | Out of scope for forward-only training data (ignored by those scripts); auto-placing street-level sensors is a follow-up for ESMDA use |
| Reproducibility (repo-wide seeded-RNG convention, `training_data.seed`) | Everything driven by one seed; per-city seeds derived by index |

## 4. Fit into the codebase

**New module** `src/pyurbanair/training_data/city_generator.py` (next to
`samplers.py` — geometry generation is training-data machinery, and keeping it
in `pyurbanair` avoids any backend/lib dependency):

```python
class CityGenerator:            # tier + placement + morphology knobs, one rng
    def generate(self, seed) -> CityLayout    # shapely footprints + heights
def layout_to_stl(layout, path) -> None       # stage B
```

**New script** `scripts/neural_surrogate/generate_geometries.py` (repo shape:
`def run(cfg)` + thin `@hydra.main` wrapper, output via `resolve_output_dir`),
driven by a new config `conf/neural_surrogate/geometry_generation.yaml`:

```yaml
geometry_generation:
  num_geometries: 32
  tier: medium              # simple | medium | advanced
  placement: random         # grid | random
  seed: 0
  output_dir: examples/generated/${geometry_generation.tier}
  domain: {...}             # one shared grid for all cities (divisible by 16)
  morphology:
    lambda_p: [0.15, 0.35]  # plan-area density band
    height: {dist: lognormal, mean: 12.0, sigma: 0.4, max_frac_lz: 0.6}
    street_width: [4.0, 16.0]
    block_size: [24.0, 80.0]     # random mode
    courtyard_prob: 0.7          # advanced tier
```

Per city `i` it writes `examples/generated/<tier>/city_{i:03d}/buildings.stl`
plus a `layout.png` top-down preview (footprints colored by height, λ_p in the
title) and a `manifest.yaml` (seed, tier, λ_p achieved, height stats) — the
same at-a-glance QA idea as `sampled_params.png` in the data-gen pipeline.

**Case wiring** — the key simplification: because all generated cities share
**one domain**, a single new `conf/case/generated.yaml` (domain block +
`geometry.stl_path` placeholder + shared `udales_case_dir`/`palm_case_dir`
copied from the xie templates with the generated domain) serves every city.
Producing a dataset per city is then just:

```bash
python scripts/neural_surrogate/generate_geometries.py geometry_generation.tier=medium
python scripts/neural_surrogate/generate_training_data.py \
    case=generated model=pylbm \
    geometry.stl_path=examples/generated/medium/city_000/buildings.stl \
    training_data.output_dir=training_data/pylbm_gen_medium_000
```

A thin batch driver (loop or SLURM array over city index) is a later
convenience; phase 1 keeps the two scripts decoupled so each city's CFD run
stays independently restartable — one ensemble run per city is the expensive
step (hours), so babysitting granularity matters more than orchestration
sugar.

**QA hook**: after export, voxelize each STL with the existing
`neural_surrogates.geometry.stl_to_fluid_mask` against the shared grid and
assert fluid fraction is in a sane band and max solid height < Lz — this
catches contract violations at generation time instead of mid-ensemble (the
memory-documented failure mode).

## 5. Suggested implementation order

1. `CityLayout` + `CityGenerator` for `simple`/grid (reproduce a Xie-Castro-
   like array as the golden test) — layout logic + unit tests, no I/O.
2. `layout_to_stl` + ground sheet + export; QA voxelization check; visual
   check via the existing `dataloading.py` geometry plot.
3. `random` placement (block subdivision) + `medium` tier.
4. `advanced` tier (courtyard/L/U footprints via buffers + interior rings).
5. Script + config + `conf/case/generated.yaml`; verify one small
   `generate_training_data.py` run per backend (pylbm first — cheapest).
6. Pilot: N=8 medium cities → 8 small datasets → feed the multi-geometry
   training work (phase 0 of the research doc) its first real input.

## 6. Open questions / later

- **Backend template dirs**: confirm the xie `udales_case_dir`/`palm_case_dir`
  configs contain nothing geometry-specific beyond the domain (namoptions
  grid fields, PALM `_p3d`) — if they do, the generated case dir needs its own
  templated copies (step 5 will surface this).
- **Periodic tiling in y** (buildings crossing the seam consistently) vs the
  default clear-margin approach.
- **Sensor auto-placement** in streets for ESMDA on generated cities.
- **Exact recipes from UrbanTALES / AB-SWIFT** (the follow-up research pass
  that was cut short) — worth a quick look before locking morphology defaults,
  purely to steal validated parameter ranges.
- **Non-flat terrain** — explicitly out of scope (PALM height-map constraint;
  `prepare_case_stl.py` flattens real terrain for the same reason).
