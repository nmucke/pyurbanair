# PALM Barcelona topography: the merged ground fills the footprint

**Status:** root cause confirmed; fix not yet applied.

## TL;DR
`examples/barcelona/buildings.stl` is **valid** geometry — it opens correctly in
the VS Code STL viewer and the **pylbm** model produces correct output. But the
**PALM** run looked like "one big building". The cause is the **extruded ground
mesh that `tools/prepare_case_stl.py` merges into the STL**: it sits elevated
(~15 m) and covers the whole footprint, so PALM's 2.5-D top-down topography
turns it into a solid raised plateau with no open streets.

## Symptom
- PALM (`model=pypalm case=barcelona`) flow field shows essentially one solid
  obstacle; no street canyons.
- pylbm on the *same* STL is correct. VS Code STL viewer looks correct.

## Why it's PALM-specific (method matters)
| Backend | Geometry method | Result |
|---|---|---|
| pylbm | Splits the mesh into components and **filters out the large ground/base** (the component covering >90 % of the domain), so it effectively runs on **buildings only** — it does not take the ground into account at all | ✅ correct (ground simply ignored) |
| PALM | **2.5-D** top-down height map: one downward ray per (x,y) cell, take the **highest** hit (`stl_to_palm._vertical_ray_heights`); no ground filtering | ❌ the elevated ground fills every cell → plateau |

pylbm being correct is therefore **not** evidence the ground is harmless — pylbm
just discards it. PALM has no equivalent filter, so it ingests the ground sheet.
A top-down ray-cast "footprint" diagnostic mimics PALM's method, so it *also*
shows a solid filled diamond — that is **not** evidence the STL lacks buildings.
Use a 3-D / cross-section view (or pylbm) to judge the buildings themselves.

## Root cause (measured)
`prepare_case_stl.py` merges **raw buildings + an extruded ground/terrain mesh**,
then yaws 45° (`--rotate`, hence the diamond footprint) and seats it with
`--z-datum min` (lowest point → z=0).

On the merged `examples/barcelona/buildings.stl`:
- Interior is ~100 % covered top-down.
- **Lowest surface in the central window averages ~15 m** (range 0–27 m), and
  most cells are a *single* open surface (not a closed volume) → an elevated
  ground sheet, not buildings, is what fills the map.

So PALM topography = ground plateau (~15 m) + buildings on top. pylbm is immune
because its 3-D test treats everything below the ground surface as solid and the
air above as fluid.

## Fix options (for the PALM / height-map path)
1. **Flatten the ground to z=0 for PALM** — strip the ground's vertical variation
   so streets read as open ground. Cleanest if the elevation is unwanted.
2. **Feed PALM the buildings-only mesh** — exclude the extruded ground from the
   height map; PALM puts its own flat floor at z=0. (pylbm/udales keep the
   merged STL since they need the ground as a 3-D floor.) This likely means
   PALM should *not* share the merged `stl_path` — give it a buildings-only STL.
3. **`--z-datum ground`** in `prepare_case_stl.py` — references mean ground
   level instead of the lowest point; helps only if the ground is roughly flat
   (won't fix genuine terrain relief).

Decision needed first: **is the ~15 m ground elevation intended terrain, or an
artifact** of how the ground was extruded / datum'd? That determines whether to
flatten (1/2) or to keep terrain and accept/handle it in PALM.

## Requirement: independent building vs. ground rotation
`prepare_case_stl.py` currently applies a **single `--rotate`** to the *merged*
mesh, so buildings and ground spin together. **I want the two rotations to be
independent and to be able to set both** — i.e. rotate the buildings relative to
the ground (e.g. align building blocks to the wind/grid) and *optionally* rotate
the ground as well, each by its own angle. Plan:
- Add separate angles, e.g. `--rotate-buildings` and `--rotate-ground` (keep
  `--rotate` as a shared default / back-compat if useful).
- Rotate each mesh about its own (or a shared, explicit) pivot **before** the
  merge + domain-frame translation, so the relative yaw between them is exactly
  the difference of the two angles.

## Relevant code
- `tools/prepare_case_stl.py` → `prepare()` (merge at ~L173, rotate ~L176,
  z-datum translate ~L196). Writes one STL to `examples/<case>/` (no longer
  symlinks into the uDALES case dir).
- `libs/pypalm/src/pypalm/stl_to_palm.py` → `_vertical_ray_heights` (~L39),
  `stl_to_palm_topography` (~L70). This is where PALM could be pointed at a
  buildings-only mesh or have the ground flattened.
- `conf/case/barcelona/geometry.yaml` → `stl_path` (shared STL, used by all
  backends). A PALM-specific buildings-only path would go here.

## How to verify a candidate fix
Rasterize the height map and check for open (z≈0) street cells in the centre:
```python
import numpy as np, trimesh
m = trimesh.load("examples/barcelona/buildings.stl")
if isinstance(m, trimesh.Scene): m = trimesh.util.concatenate(list(m.geometry.values()))
b = m.bounds; cx, cy = (b[0,0]+b[1,0])/2, (b[0,1]+b[1,1])/2
N, W = 120, 200
xs = np.linspace(cx-W/2, cx+W/2, N); ys = np.linspace(cy-W/2, cy+W/2, N)
xx, yy = np.meshgrid(xs, ys, indexing="xy")
o = np.column_stack([xx.ravel(), yy.ravel(), np.full(xx.size, b[1,2]+1)])
d = np.tile([0,0,-1.0], (o.shape[0],1))
loc, idr, _ = trimesh.ray.ray_triangle.RayMeshIntersector(m).intersects_location(o, d, multiple_hits=False)
h = np.zeros(o.shape[0])
for p, r in zip(loc, idr): h[r] = max(h[r], p[2])
print("central built fraction:", (h > 2).mean())   # want << 1.0 (streets open) after fix
```
A correct PALM topography shows a clear street grid (open cells), not ~100 %
coverage. Final confidence check: re-run `model=pypalm case=barcelona` and look
at the flow field.

## Loose ends from the prior session (context, not blocking)
- uDALES precomputed-geometry feature landed (`precomputed_geom_dir` config,
  `tools/preprocess_udales_geometry.py`, `geom_meta.json` grid guard); validated
  on `xie_and_castro`, **not yet generated for Barcelona** (was waiting on this).
- `prepare_case_stl.py` no longer writes into `examples/udales/<case>/`; uDALES
  STL sourcing at run time still needs rewiring (it loads the STL for
  `facets.inp`/`nfcts` even with a precomputed bundle).
- Temp diagnostic scripts in `scripts/` (`_diag_*`, `_time_*`,
  `_gen_barcelona_geom.py`, `_validate_precomputed_geom.py`) and the
  `examples/udales/barcelona/namoptions.300` `nprocx/nprocy` edits are pending
  cleanup. A stray raw `./buildings.stl` sits at the repo root.
