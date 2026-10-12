"""pycity4cfd.osm: height chain, clipping, CRS, ids and the GeoJSON round trip.

osmnx is mocked with a hand-built GeoDataFrame shaped like its output (tags as
columns, an (element, id) index), so no test touches the network.
"""

from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from pycity4cfd import osm
from shapely.geometry import MultiPolygon, Point, Polygon, box

# A ~300 m box in central Delft.
BBOX = (4.3550, 52.0100, 4.3594, 52.0127)
D = 0.0002  # ~14-22 m


def _square(x: float, y: float, d: float = D) -> Polygon:
    return box(x, y, x + d, y + d)


@pytest.fixture  # type: ignore[misc]
def raw() -> gpd.GeoDataFrame:
    """osmnx-like features: mixed geometries, odd tags, one straddling the bbox."""
    w, s = BBOX[0] + 0.0005, BBOX[1] + 0.0005
    rows = [
        # (element, id, geometry, height, building:levels)
        ("way", 1, _square(w, s), "12", None),
        ("way", 2, _square(w + 0.0005, s), "12.5 m", "7"),
        ("way", 3, _square(w + 0.001, s), "40 ft", None),
        ("way", 4, _square(w + 0.0015, s), "tall", "4"),
        ("way", 5, _square(w + 0.002, s), None, "2,5"),
        ("way", 6, _square(w + 0.0025, s), "0", "many"),
        (
            "relation",
            7,
            MultiPolygon([_square(w, s + 0.0005), _square(w + 0.0005, s + 0.0005)]),
            "20",
            None,
        ),
        ("node", 8, Point(w, s + 0.001), "30", None),
        # Half outside the east edge of the bbox.
        ("way", 9, _square(BBOX[2] - D / 2, s), None, None),
        # A ~0.1 m² sliver.
        ("way", 10, _square(w, s + 0.0015, 1e-6), "5", None),
    ]
    index = pd.MultiIndex.from_tuples([r[:2] for r in rows], names=["element", "id"])
    return gpd.GeoDataFrame(
        {
            "building": "yes",
            "height": [r[3] for r in rows],
            "building:levels": [r[4] for r in rows],
        },
        geometry=[r[2] for r in rows],
        index=index,
        crs="EPSG:4326",
    )


@pytest.fixture  # type: ignore[misc]
def buildings(
    raw: gpd.GeoDataFrame, monkeypatch: pytest.MonkeyPatch
) -> gpd.GeoDataFrame:
    calls = []

    def features_from_bbox(bbox: tuple, tags: dict) -> gpd.GeoDataFrame:
        calls.append((bbox, tags))
        return raw

    monkeypatch.setattr(osm.ox, "features_from_bbox", features_from_bbox)
    gdf = osm.fetch_buildings(BBOX, floor_height=3.0, default_height=10.0)
    assert calls == [(BBOX, {"building": True})]
    return gdf


@pytest.mark.parametrize(  # type: ignore[misc]
    "value, expected",
    [
        ("12", 12.0),
        ("12 m", 12.0),
        ("12.5m", 12.5),
        ("12,5", 12.5),
        (" 7 metres ", 7.0),
        ("40 ft", 12.192),
        ("40'", 12.192),
        ("0", None),
        ("tall", None),
        ("10;12", None),
        ("~10", None),
        (None, None),
        (np.nan, None),
    ],
)
def test_parse_height(value: object, expected: float | None) -> None:
    assert osm.parse_height(value) == pytest.approx(expected)


def test_height_chain(buildings: gpd.GeoDataFrame) -> None:
    by_id = buildings.set_index("id")
    expected = {
        "way/1": (12.0, "height"),
        "way/2": (12.5, "height"),  # height wins over levels
        "way/3": (12.192, "height"),
        "way/4": (12.0, "levels"),  # garbage height falls through
        "way/5": (7.5, "levels"),
        "way/6": (10.0, "default"),  # zero height, garbage levels
        "way/9": (10.0, "default"),
    }
    for id_, (height, source) in expected.items():
        assert by_id.loc[id_, "height"] == pytest.approx(height)
        assert by_id.loc[id_, "height_source"] == source
    assert (buildings["height"] > 0).all()
    assert buildings["height"].dtype == float


def test_polygons_ids_and_crs(buildings: gpd.GeoDataFrame) -> None:
    assert list(buildings.columns) == osm.COLUMNS
    assert set(buildings.geom_type) == {"Polygon"}
    assert buildings.is_valid.all()
    # The point and the sliver are gone, the multipolygon is split in two.
    assert buildings["id"].is_unique
    assert set(buildings["id"]) == {
        *(f"way/{i}" for i in (1, 2, 3, 4, 5, 6, 9)),
        "relation/7_1",
        "relation/7_2",
    }
    assert buildings.crs.to_epsg() == 32631  # UTM 31N
    assert buildings.crs.axis_info[0].unit_name == "metre"


def test_clipped_to_bbox(buildings: gpd.GeoDataFrame, raw: gpd.GeoDataFrame) -> None:
    area = box(*BBOX)
    bounds = gpd.GeoSeries([area], crs="EPSG:4326").to_crs(buildings.crs)
    # 1 cm slack: a projected meridian chord is not exactly straight in UTM.
    assert buildings.within(bounds.buffer(0.01).iloc[0]).all()
    # The straddling building keeps about half its area.
    full = raw.loc[("way", 9)].geometry
    full_area = gpd.GeoSeries([full], crs="EPSG:4326").to_crs(buildings.crs).area[0]
    clipped = buildings.set_index("id").area["way/9"]
    assert clipped == pytest.approx(full_area / 2, rel=0.05)


def test_write_buildings_round_trip(
    buildings: gpd.GeoDataFrame, tmp_path: Path
) -> None:
    path = osm.write_buildings(buildings, tmp_path / "out" / "buildings.geojson")
    assert path == tmp_path / "out" / "buildings.geojson"
    # Writing again overwrites rather than failing or appending.
    osm.write_buildings(buildings, path)
    back = gpd.read_file(path)
    assert back.crs == buildings.crs
    assert list(back["id"]) == list(buildings["id"])
    np.testing.assert_allclose(back["height"], buildings["height"])
    assert list(back["height_source"]) == list(buildings["height_source"])
    assert back.geom_equals_exact(buildings.geometry, tolerance=1e-6).all()
