# pycity4cfd: a simulation case from an area on a map

`libs/pycity4cfd` turns a bbox (west, south, east, north in EPSG:4326) into a
case folder and the Hydra overrides that run a forward model on it. Prototype:
buildings only (no terrain, trees or point clouds), heights from OSM tags.

```text
bbox -> osm.fetch_buildings -> buildings.geojson (UTM, "height")
     -> city4cfd.write_config + run_city4cfd -> city4cfd/city4cfd_Buildings.stl
     -> case.make_case -> buildings.stl (domain frame), namoptions.300 (uDALES),
                          _p3d (PALM), case.json + overrides
```

From Claude Code the MCP tools drive it: `open_area_selector` ->
`get_selected_area` -> `prepare_city_case` -> `prepare_forward_run` /
`launch_forward_run` with `model=pylbm`, `pypalm` or `pyudales` (see
[mcp.md](mcp.md)). By hand:

```bash
pixi install -e city4cfd          # first time: fetches and compiles City4CFD, ~2 min
pixi run -e city4cfd python -c "from pycity4cfd.case import make_case; \
  print(make_case((4.3555, 52.0095, 4.3600, 52.0122), '.temp/city_cases/delft'))"
pixi run -e dev python scripts/run_forward.py model=pylbm <overrides>  # or pypalm, pyudales
```

pycity4cfd and City4CFD live only in the `city4cfd` pixi environment; `dev`,
`mcp` and `ui` don't have them (the MCP tools and the browser app build cases
as jobs in `city4cfd`).

## Modules

- **`osm`**: `fetch_buildings(bbox)` downloads footprints with osmnx (Overpass),
  keeps polygons, clips to the bbox and reprojects to the local UTM zone.
  Height chain: OSM `height` (parsed: m, ft, decimal commas), else
  `building:levels` x `floor_height` (3 m), else `default_height` (10 m);
  `height_source` records which. Few buildings carry `height`: in central Delft
  4 of 439, about half fall back to the default. Overpass can stall for
  minutes (504 + retry). An area with no building footprint (none in OSM, or
  only nodes, slivers below `min_area` or parts clipped away) raises a
  `ValueError`.
- **`city4cfd`**: City4CFD 0.8.1 (AGPL-3.0, tudelft3d/City4CFD) is not on
  conda-forge. `libs/pycity4cfd/recipe/pixi.toml` packages it as a pixi source
  package (`pixi-build-cmake`, upstream git tag `0.8.1`, conda-forge CGAL,
  Boost, Eigen 3.4, GMP, MPFR, GDAL and OpenMP), so the first
  `pixi install -e city4cfd` fetches it, compiles it with conda-forge's
  compilers (about 2 min; on macOS with the Xcode command line tools' SDK) and
  installs `$CONDA_PREFIX/bin/city4cfd`; later installs reuse pixi's build
  cache. The package's build config is not part of that cache key: after
  editing `recipe/pixi.toml`, run `pixi clean --build` and reinstall. Each building is an LoD1.2
  extrusion to its `height`. City4CFD cannot run without a ground point cloud,
  so `write_config` writes a flat `ground.xyz` grid; every footprint must lie
  inside the `half_width` square around `point_of_interest`, or City4CFD
  aborts. Output coordinates are relative to `point_of_interest`; buildings
  with no valid height are skipped (see `city4cfd_log`).
- **`case`**: `make_case(bbox, case_dir, resolution=4.0, margin=50.0)` sizes
  the domain to the bbox's UTM envelope plus `margin` of open ground on every
  side (clear inflow/outflow), top at 3x the tallest building (at least 15
  cells), even nz and nx/ny rounded up to even counts with no prime factor
  above 5 (PALM's Temperton FFT, PAC0317; the extra cells widen the +x/+y
  margin). It shifts the STL into the domain frame (the
  [geometries/README.md](../geometries/README.md) contract: metres,
  `(0, 0)` corner, flat ground at z = 0, no ground mesh), picks six
  street-level sensors outside buildings and three more as held-out
  validation sensors (`++obs.validation_*_points`, so the overrides compose
  on any case). `case.json` and the returned dict record `crs` (the UTM
  EPSG) and `origin`, the projected coordinates of the domain's `(0, 0)`
  corner: domain `(x, y)` + `origin` is a point in `crs` (to put sensors back
  on the map). It also writes the uDALES `namoptions.300` and PALM `_p3d`
  from the `geometries/xie_and_castro/` templates: what the wrappers do not
  overwrite there (courant/CFL 0.7, neutral, no Coriolis, CD2 advection,
  roughness 0.05 m) is generic and matched between the two solvers, while
  barcelona's and urbantales' leave Coriolis on in PALM. The grid is written
  for consistency (both wrappers overwrite it from `domain`, as they do
  runtime, BCs, inflow and closure). uDALES reads `stl_file =
  buildings.stl`; `nfcts` is the STL's triangle count, which its IBM
  preprocessing recounts, and the other `&WALLS` counts are zeroed because
  that preprocessing regenerates them. uDALES needs no ground facets (the floor is its z = 0 wall, as for
  the UrbanTALES STLs). The overrides also null
  `geometry.udales_precomputed_geom_dir`, which barcelona sets to its own
  case folder.
- `CASE_FORMAT` (also written to `case.json` as `case_format`) versions the
  folder layout, `case.json` keys and overrides. Bump it when a change makes
  older folders unusable: it is part of the MCP server's case-cache key
  ([mcp.md](mcp.md)), so folders from an older format are rebuilt, not
  reused.

## Limits

All three backends run these cases. On the ~300 x 300 m Delft bbox above
(439 buildings, ~20k STL triangles, 108 x 108 x 16 cells at 4 m, 60 s
window + 10 s spin-up, one core, macOS M-series): PALM ~18 s, uDALES ~75 s of
which its IBM preprocessing (STL -> solid/boundary points) ~9 s, pylbm ~170 s
including a fresh compile. The preprocessing grows with grid size and
triangle count and runs once per prepare. Short windows hit two backend limits that are not
city-specific: uDALES' turbulent inlet fails for `time.simulation_time` below
`time.seconds_per_knot`, and PALM returned all-zero fields with
`time.output_frequency=5` over a 30 s window. Sensors need street points
`2 * resolution` clear of buildings, so coarse grids in dense areas (8 m in
central Delft) raise "Too few open street-level points". Cases built before
the solver templates were added lack `namoptions.300`/`_p3d`: rebuild them.
The selector caps the area at 4 km²; at 4 m that is a ~550 x 550 grid, so
check `domain` before launching.
