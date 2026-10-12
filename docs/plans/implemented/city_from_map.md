# Plan: simulate a city area picked on a map

Prototype. In Claude Code the user asks to simulate a city area; an MCP tool
opens a map, the user draws a rectangle, the buildings come from OpenStreetMap
(heights from OSM tags), City4CFD turns them into an STL, and pylbm runs on it.
Point clouds, terrain and trees are out of scope for now. Implemented; the
maintained reference is [docs/pycity4cfd.md](../../pycity4cfd.md).

## Pipeline and owners

| Stage | Module | Branch |
|---|---|---|
| Map selection (bbox from a browser page) | `libs/mcp-server`: `open_area_selector`, `get_selected_area` | `feat/city-map-selector` |
| OSM buildings with heights | `pycity4cfd.osm` | `feat/city-osm` |
| City4CFD build + run | `pycity4cfd.city4cfd`, City4CFD submodule, pixi build task | `feat/city4cfd-build` |
| bbox -> case dir -> pylbm run | `pycity4cfd.case`, MCP `prepare_city_case` | `feat/city-from-map` (integration) |

All branch off `feat/city-from-map` and merge back into it.

## Interfaces

A bbox is `(west, south, east, north)` in EPSG:4326 degrees.

```python
# pycity4cfd.osm
def fetch_buildings(bbox, *, floor_height=3.0, default_height=10.0) -> geopandas.GeoDataFrame
    # Polygons in the local UTM CRS (gdf.estimate_utm_crs()), clipped to bbox,
    # columns: "id" (str), "height" (float, m, > 0), "height_source"
    # ("height" | "levels" | "default"). Height chain: OSM `height` tag, else
    # `building:levels` * floor_height, else default_height.
def write_buildings(gdf, path) -> Path   # GeoJSON, keeps the projected CRS

# pycity4cfd.city4cfd
def city4cfd_executable() -> Path        # the binary built by the pixi task
def write_config(buildings_path, out_dir, *, point_of_interest, half_width,
                 top_height, ...) -> Path  # City4CFD JSON: height_attribute
                                           # "height", flat terrain, LoD1.2, STL
def run_city4cfd(config_path) -> Path    # runs via pyurbanair.utils.solver_process.run_solver,
                                         # returns the buildings STL
```

The case stage (integration) puts the STL in the domain frame (metres,
`(xmin, ymin) -> (0, 0)`, flat ground at `z = 0`, z up; see
`scripts/tools/prepare_case_stl.py`) and passes the case to pylbm with
`geometry.case_dir=...` plus domain overrides.
