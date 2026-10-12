"""Building footprints with heights from OpenStreetMap.

``fetch_buildings`` downloads the ``building=*`` features in a bbox with osmnx
and returns one polygon per row in the local UTM CRS, clipped to the bbox, with
a height for each: the OSM ``height`` tag, else ``building:levels`` times
``floor_height``, else ``default_height``. ``roof:levels`` is ignored: City4CFD
extrudes flat LoD1.2 roofs, and ``height`` (when present) already includes the
roof. ``min_height`` / ``building:min_level`` (overhangs, bridges) are ignored
too, so every footprint is extruded from the ground.

``write_buildings`` writes them as GeoJSON in that projected CRS (GDAL writes
the CRS as a ``crs`` member and reads it back; City4CFD takes the coordinates
as they are, so they must be metric).
"""

import re
from pathlib import Path

import geopandas as gpd
import numpy as np
import osmnx as ox
import pandas as pd
from shapely.geometry import box

# Don't leave osmnx's response cache (./cache) in whatever directory we run from.
ox.settings.use_cache = False

COLUMNS = ["id", "height", "height_source", "geometry"]

# "12", "12.5", "12,5 m", "40 ft", "40'"; anything else is not a height.
_HEIGHT = re.compile(r"^\s*(\d+(?:[.,]\d+)?)\s*(m|meters?|metres?|ft|feet|')?\s*$")


def parse_height(value: object) -> float | None:
    """Metres from an OSM ``height`` tag, or None if it doesn't parse."""
    if not isinstance(value, str):
        return None
    match = _HEIGHT.match(value.lower())
    if match is None:
        return None
    height = float(match.group(1).replace(",", "."))
    if match.group(2) in ("ft", "feet", "'"):
        height *= 0.3048
    return height if height > 0 else None


def parse_levels(value: object) -> float | None:
    """Number of levels from an OSM ``building:levels`` tag, or None."""
    try:
        levels = float(str(value).replace(",", "."))
    except ValueError:
        return None
    return levels if np.isfinite(levels) and levels > 0 else None


def _polygons(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Valid polygons only, one per row (multipolygons split up)."""
    gdf = gdf.set_geometry(gdf.geometry.make_valid()).explode(index_parts=False)
    return gdf[gdf.geom_type == "Polygon"]


def fetch_buildings(
    bbox: tuple[float, float, float, float],
    *,
    floor_height: float = 3.0,
    default_height: float = 10.0,
    min_area: float = 1.0,
) -> gpd.GeoDataFrame:
    """OSM building footprints in ``bbox`` = (west, south, east, north), EPSG:4326.

    Returns polygons in the local UTM CRS, clipped to the bbox, without slivers
    smaller than ``min_area`` m², with columns ``id`` (``"way/123"``, suffixed
    ``"_1"``, ``"_2"`` ... for the parts of a multipolygon), ``height`` (m) and
    ``height_source`` (``"height"`` | ``"levels"`` | ``"default"``).
    """
    raw = ox.features_from_bbox(bbox, tags={"building": True})
    raw = raw.reset_index()
    raw["id"] = raw["element"].astype(str) + "/" + raw["id"].astype(str)

    height = raw.get("height", pd.Series(None, index=raw.index)).map(parse_height)
    levels = raw.get("building:levels", pd.Series(None, index=raw.index))
    levels = levels.map(parse_levels)
    raw["height_source"] = np.where(
        height.notna(), "height", np.where(levels.notna(), "levels", "default")
    )
    raw["height"] = height.fillna(levels * floor_height).fillna(default_height)
    raw["height"] = raw["height"].astype(float)

    gdf = _polygons(raw[COLUMNS])
    gdf = _polygons(gdf.clip(box(*bbox)))
    gdf = gdf.to_crs(gdf.estimate_utm_crs())
    gdf = gdf[gdf.area >= min_area].reset_index(drop=True)

    # Unique ids: suffix the parts of a split multipolygon (or clipped polygon).
    part = gdf.groupby("id").cumcount() + 1
    split = gdf["id"].duplicated(keep=False)
    gdf.loc[split, "id"] = gdf.loc[split, "id"] + "_" + part[split].astype(str)
    return gdf


def write_buildings(gdf: gpd.GeoDataFrame, path: str | Path) -> Path:
    """Write ``fetch_buildings`` output as GeoJSON, keeping its projected CRS."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.unlink(missing_ok=True)
    gdf[COLUMNS].to_file(path, driver="GeoJSON")
    return path
