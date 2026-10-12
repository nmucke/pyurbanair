# pycity4cfd: a simulation case from an area on a map

`libs/pycity4cfd` turns a bbox (west, south, east, north in EPSG:4326) into a
case folder and the Hydra overrides that run a forward model on it. Prototype:
buildings only (no terrain, trees or point clouds), heights from OSM tags.

```text
bbox -> osm.fetch_buildings -> buildings.geojson (UTM, "height")
     -> city4cfd.write_config + run_city4cfd -> city4cfd/city4cfd_Buildings.stl
     -> case.make_case -> buildings.stl (domain frame) + case.json + overrides
```

From Claude Code the MCP tools drive it: `open_area_selector` ->
`get_selected_area` -> `prepare_city_case` -> `prepare_forward_run` /
`launch_forward_run` with `model=pylbm` (see [mcp.md](mcp.md)). By hand:

```bash
pixi run -e dev build-city4cfd    # once, ~2 min
pixi run -e dev python -c "from pycity4cfd.case import make_case; \
  print(make_case((4.3555, 52.0095, 4.3600, 52.0122), '.temp/city_cases/delft'))"
pixi run -e dev python scripts/run_forward.py model=pylbm <overrides>
```

## Modules

- **`osm`**: `fetch_buildings(bbox)` downloads footprints with osmnx (Overpass),
  keeps polygons, clips to the bbox and reprojects to the local UTM zone.
  Height chain: OSM `height` (parsed: m, ft, decimal commas), else
  `building:levels` x `floor_height` (3 m), else `default_height` (10 m);
  `height_source` records which. Few buildings carry `height`: in central Delft
  4 of 439, about half fall back to the default. Overpass can stall for
  minutes (504 + retry); an empty area raises osmnx's
  `InsufficientResponseError`.
- **`city4cfd`**: City4CFD 0.8.1 (`libs/pycity4cfd/City4CFD`, AGPL-3.0
  submodule, vendored: don't edit) is not on conda-forge; `build-city4cfd`
  compiles it with the `city4cfd` pixi feature's conda-forge deps into
  `libs/pycity4cfd/build/` (not in `setup-dev`). Each building is an LoD1.2
  extrusion to its `height`. City4CFD cannot run without a ground point cloud,
  so `write_config` writes a flat `ground.xyz` grid; every footprint must lie
  inside the `half_width` square around `point_of_interest`, or City4CFD
  aborts. Output coordinates are relative to `point_of_interest`; buildings
  with no valid height are skipped (see `city4cfd_log`).
- **`case`**: `make_case(bbox, case_dir, resolution=4.0, margin=50.0)` sizes
  the domain to the bbox's UTM envelope plus `margin` of open ground on every
  side (clear inflow/outflow), top at 3x the tallest building (at least 15
  cells), even nx/ny/nz. It shifts the STL into the domain frame (the
  [geometries/README.md](../geometries/README.md) contract: metres,
  `(0, 0)` corner, flat ground at z = 0, no ground mesh), picks six
  street-level sensors outside buildings and drops the default case's
  held-out sensors.

## Limits

Only pylbm has been run on these cases. PALM should work (its STL rasteriser
takes the same STL; nx/ny are even); uDALES needs a namoptions with a matching
`nfcts` and a slow IBM preprocessing, so it is not wired. The selector caps the
area at 4 km²; at 4 m that is a ~550 x 550 grid, so check `domain` before
launching.
