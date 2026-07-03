# UrbanTALES geometries

Urban building geometries harvested from the **UrbanTALES** dataset and
converted into the domain-frame `.stl` contract that every backend in this repo
consumes (pylbm, uDALES, PALM). Intended as multi-geometry input for neural
surrogate training (see `docs/plans/geometry_generation_plan.md`).

UrbanTALES is a large-eddy-simulation study of 538 urban configurations run in
PALM. Each case ships a `*_topo` file: a 2-D building-height raster in plain-text
ASCII at 1 m resolution (`0` = ground/street, positive = building height in
metres). Because it is a top-down 2.5-D height map, it maps cleanly onto our STL
contract — flat ground at z=0, extruded footprints, no overhangs; courtyards
(holes in a footprint) are fine.

- **224 idealized** cases — aligned & staggered arrays (Xie & Castro lineage),
  fixed mean height 16 m, varied height spread / max height, wind at 0° and 90°.
- **314 realistic** cases — real neighbourhoods worldwide (OpenStreetMap
  footprints via the authors' `OSM2LES` tool), plan-area density λ_p ∈ [0.06, 0.64],
  heights ≈ [4, 50] m, domains 300×300 m … 1000×1000 m.

Only the geometries are pulled here (~350 MB of rasters). The UrbanTALES CFD flow
data is **time-averaged mean fields**, not time-resolved trajectories, so it is
not usable as autoregressive-surrogate training data — see the evaluation notes
for that discussion.

## Layout

```
examples/geometries/
├── download_urbantales_geometries.py   # fetch the *_topo rasters
├── rasters_to_stl.py                   # convert rasters -> domain-frame STL
├── raw/          {idealized,realistic}/  *_topo rasters + metadata.json   (gitignored)
└── processed/    {idealized,realistic}/  *.stl + manifest.csv             (gitignored)
```

`raw/` and `processed/` are gitignored — the data is bulky (~1 GB total) and
fully regenerable from the two scripts below.

## Reproduce

Run inside the `dev` Pixi env (both scripts need no arguments for the full set):

```bash
# 1. download the height rasters into raw/{idealized,realistic}
python examples/geometries/download_urbantales_geometries.py

# 2. convert them into watertight STLs in processed/{idealized,realistic}
pixi run -e dev python examples/geometries/rasters_to_stl.py
```

Both are stdlib-friendly, parallel, and resumable (skip files already present;
`--force` to overwrite). Useful flags: `--set {idealized,realistic}`,
`--workers N`. See each script's `--help`.

## Mesh contract & design notes

Each raster is meshed by greedy-decomposing every height level into rectangles,
extruding each to a box from z=0 to its height, and **boolean-unioning** the
boxes (via `manifold3d`) into one **watertight manifold** per city. The union
removes the coincident interior walls between abutting buildings, so ray-based
`trimesh.mesh.contains` — and hence `neural_surrogates.geometry.stl_to_fluid_mask` —
voxelises the geometry exactly (verified 100% vs the raster).

- **No separate ground sheet.** An open sheet reintroduces ray ambiguity at its
  free edges; building undersides already sit at z=0, pylbm strips the sheet and
  PALM ignores it, and the floor is a z=0 boundary condition. Pass `--multibody`
  for the older fast box-soup mesh *with* a full ground sheet (matches the repo's
  Barcelona STL) if you don't need exact `contains`.
- **`is_watertight` reads `False` on reload** for large meshes — a binary-STL
  artifact (STL is a triangle soup with no shared-vertex topology, so trimesh's
  reload-merge can't restitch big meshes). The in-memory union *is* watertight
  and `contains` is exact regardless.
- **Exact obstacle mask without the mesh.** For these height maps, cell (i,j,k)
  is solid iff `raster[i,j] > z_k` — often simpler than voxelising the STL.

`manifest.csv` records per case: grid size (nx, ny), physical extent (Lx, Ly),
z_max, plan-area density λ_p, and face count. `raw/**/metadata.json` carries the
upstream morphology (config, wind direction, height stats; city/country/lat-lon
for realistic cases).

## Provenance, license & citation

- **Data platform:** <https://urbantales.vercel.app/> (files served from a
  Nextcloud instance; `/api/metadata_{ideal,rea}` are the manifests the
  downloader reads).
- **Generator (realistic footprints):** `OSM2LES` / `OSM2PALM` — MIT,
  <https://github.com/jiachenlu95/BF2PALM>, <https://zenodo.org/records/6566346>.
- **License:** the UrbanTALES data is **CC BY-NC 4.0** (non-commercial; attribution
  required). Cite the paper in any work that uses these geometries.

> Nazarian, N., Lu, J., Lipson, M., Liu, S., Hart, M., Krayenhoff, E. S., Blunn, L.,
> & Martilli, A. (2025). *UrbanTALES: A Large-Eddy Simulation Dataset for Urban
> Canopy Layer Turbulence and Parameterization.* Bulletin of the American
> Meteorological Society, 106(12), E2461–E2478.
> https://doi.org/10.1175/BAMS-D-25-0061.1 (preprint: https://doi.org/10.31223/X58X3B)
