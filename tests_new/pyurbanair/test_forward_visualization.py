"""Scientific visualization contracts use synthetic fields, never CFD setup."""

import json
import subprocess
from pathlib import Path
from typing import Sequence
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import numpy as np
import pytest
import xarray as xr

from pyurbanair.visualization import (
    ArtifactReader,
    BundleAssetServer,
    normalize,
    render,
)
from pyurbanair.visualization.data import fingerprint


def regular(times: Sequence[float] = (0.0, 2.5, 7.0)) -> xr.Dataset:
    shape = (len(times), 3, 4, 5)
    ds = xr.Dataset(
        {
            name: (("time", "z", "y", "x"), np.full(shape, value, dtype=float))
            for name, value in (("u", 3), ("v", 4), ("w", 12))
        },
        coords={
            "time": list(times),
            "x": [1, 2, 4, 7, 11],
            "y": [2, 4, 8, 12],
            "z": [1, 3, 6],
        },
    )
    for name in ("u", "v", "w"):
        ds[name].attrs["units"] = "m/s"
    return ds


def save_run(tmp_path: Path, ds: xr.Dataset | None = None) -> Path:
    run = tmp_path / "run"
    run.mkdir()
    (regular() if ds is None else ds).to_netcdf(run / "state.nc")
    return run


def test_default_dashboard_has_maps_section_and_matching_height_probes(
    tmp_path: Path,
) -> None:
    ds = regular((0, 2.5))
    ds["w"] = ds.w * 0 + ds.z - 3
    ds["blanking"] = xr.zeros_like(ds.u)
    ds.blanking.loc[dict(x=4, y=8, z=1)] = 1
    root = save_run(tmp_path, ds)
    bundle = tmp_path / "bundle"
    manifest = render(root, bundle, {"movie": False, "width": 320, "height": 240})
    low, high, side = manifest["views"]
    assert [view["slice"]["axis"] for view in manifest["views"]] == ["z", "z", "y"]
    assert low["slice"]["actual"] != high["slice"]["actual"]
    assert low["field"] == high["field"] == "horizontal_speed"
    assert low["color_limits"] == high["color_limits"]
    assert side["field"] == "w" and side["cmap"] == "RdBu_r"
    assert side["color_limits"] == [-3, 3]
    probes = json.loads((bundle / "probes.json").read_text())["probes"]
    assert len(probes) == 6
    for a, b in zip(probes[:3], probes[3:]):
        assert a["label"] == b["label"]
        assert a["color"] == b["color"]
        assert a["actual"]["x"] == b["actual"]["x"]
        assert a["actual"]["y"] == b["actual"]["y"] == side["slice"]["actual"]
    assert probes[0]["values"] == [None, None]  # Never move a solid sample into fluid.
    assert probes[3]["values"] == [5, 5]
    with BundleAssetServer() as server:
        url = server.register(bundle)
        for view in manifest["views"]:
            assert [frame["simulation_time"] for frame in view["snapshots"]] == [0, 2.5]
            with urlopen(url + view["snapshots"][1]["path"]) as response:
                assert response.read().startswith(b"\x89PNG")


def test_regular_magnitudes_masks_and_axis_orientation() -> None:
    ds = regular((0,)).isel(time=0, drop=True).sortby("x", ascending=False)
    ds["blanking"] = xr.zeros_like(ds.u)
    ds["blanking"].loc[dict(z=1, y=2, x=1)] = 1
    ds["u"].loc[dict(z=1, y=2, x=2)] = 0
    ds["v"].loc[dict(z=1, y=2, x=2)] = 0
    output = normalize(ds)
    assert output.speed.dims == ("z", "y", "x")
    assert output.x.values.tolist() == [1, 2, 4, 7, 11]
    assert np.isnan(output.speed.sel(x=1, y=2, z=1))
    assert output.horizontal_speed.sel(x=2, y=2, z=1) == 0
    assert output.speed.sel(x=4, y=2, z=1) == 13
    assert output.horizontal_speed.sel(x=4, y=2, z=1) == 5


@pytest.mark.parametrize(  # type: ignore[misc]
    "names", [("xt", "yt", "zt", "xm", "ym", "zm"), ("x", "y", "z", "xu", "yv", "zw")]
)
def test_staggered_components_interpolate_physical_positions(
    names: tuple[str, ...]
) -> None:
    x, y, z, xu, yv, zw = names
    coords = {
        x: [1.0, 2.0, 3.0],
        y: [1.0, 2.0, 3.0],
        z: [1.0, 2.0, 3.0],
        xu: [0.0, 2.0, 4.0],
        yv: [0.0, 2.0, 4.0],
        zw: [0.0, 2.0, 4.0],
    }
    ds = xr.Dataset(
        {
            "u": ((z, y, xu), np.broadcast_to(np.array(coords[xu]), (3, 3, 3))),
            "v": (
                (z, yv, x),
                np.broadcast_to(np.array(coords[yv])[None, :, None], (3, 3, 3)),
            ),
            "w": (
                (zw, y, x),
                np.broadcast_to(np.array(coords[zw])[:, None, None], (3, 3, 3)),
            ),
        },
        coords=coords,
    )
    normalized = normalize(ds)
    assert normalized.speed.sel(x=1, y=2, z=3) == pytest.approx(np.sqrt(14))
    assert normalized.u.sel(x=1, y=2, z=3) == 1


def test_missing_boundary_not_extrapolated() -> None:
    ds = regular((0,)).isel(time=0, drop=True)
    ds = (
        ds.drop_vars("u")
        .assign(u=(("z", "y", "xu"), np.ones((3, 4, 5))))
        .assign_coords(xu=[2, 3, 4, 7, 11])
    )
    assert np.isnan(normalize(ds).speed.sel(x=1)).all()


def test_ensemble_quantity_reduction_order(tmp_path: Path) -> None:
    ds = regular().expand_dims(ensemble=[4, 8]).copy(deep=True)
    ds["u"].loc[dict(ensemble=8)] *= -1
    ds["v"].loc[dict(ensemble=8)] *= -1
    ds["w"].loc[dict(ensemble=8)] *= -1
    root = save_run(tmp_path, ds)
    with pytest.raises(ValueError, match="explicit member"):
        ArtifactReader(root)
    with pytest.raises(ValueError, match="Unknown ensemble member"):
        ArtifactReader(root, member=0)
    assert np.all(ArtifactReader(root, member=8).frame(0).speed == 13)
    assert np.all(ArtifactReader(root, reduction="mean_velocity").frame(0).speed == 0)
    assert np.all(ArtifactReader(root, reduction="mean_speed").frame(0).speed == 13)


def test_index_windows_deduplicate_and_check_hashes(tmp_path: Path) -> None:
    root = tmp_path / "run"
    root.mkdir()
    entries = []
    for window, times in enumerate(((0, 2), (2, 5))):
        path = root / f"window-{window}.nc"
        regular(times).expand_dims(ensemble=[9]).to_netcdf(path)
        entries.append(
            {
                "kind": "state",
                "path": path.name,
                "window": window,
                "member": 9,
                "sha256": fingerprint(path),
            }
        )
    index = {
        "version": 1,
        "status": "complete",
        "total_windows": 2,
        "artifacts": entries,
    }
    (root / "artifact_index.json").write_text(json.dumps(index))
    reader = ArtifactReader(root, member=9)
    assert reader.times == [0, 2, 5]
    assert float(reader.frame(2).speed.mean()) == 13
    entries[0]["sha256"] = "bad"
    (root / "artifact_index.json").write_text(json.dumps(index))
    with pytest.raises(ValueError, match="fingerprint changed"):
        ArtifactReader(root, member=9)


def test_conflicting_window_endpoint_rejected(tmp_path: Path) -> None:
    root = tmp_path / "run"
    root.mkdir()
    regular((0, 2)).to_netcdf(root / "a.nc")
    different = regular((2, 5))
    different["u"] += 1
    different.to_netcdf(root / "b.nc")
    (root / "artifact_index.json").write_text(
        json.dumps(
            {
                "status": "complete",
                "artifacts": [
                    {"kind": "state", "path": "a.nc", "window": 0},
                    {"kind": "state", "path": "b.nc", "window": 1},
                ],
            }
        )
    )
    reader = ArtifactReader(root)
    with pytest.raises(ValueError, match="Conflicting"):
        reader.frame(2)


def test_render_bundle_probes_actual_coordinates_and_source_unchanged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import importlib

    monkeypatch.setattr(
        importlib.import_module("pyurbanair.visualization.render").shutil,
        "which",
        lambda _: None,
    )
    root = save_run(tmp_path)
    original = fingerprint(root / "state.nc")
    bundle = tmp_path / "bundle"
    result = render(
        root,
        bundle,
        {
            "slices": [{"axis": "z", "position": 2.6}],
            "probes": [
                {"id": "<script>alert(1)</script>", "x": 4.0, "y": 8.0, "z": 3.0}
            ],
            "width": 320,
            "height": 240,
        },
    )
    assert result["views"][0]["slice"]["actual"] == 3
    assert result["views"][0]["frames"] == [
        {"video_time": 0.0, "simulation_time": 0.0},
        {"video_time": 1 / 12, "simulation_time": 2.5},
        {"video_time": 2 / 12, "simulation_time": 7.0},
    ]
    probes = json.loads((bundle / "probes.json").read_text())
    assert probes["probes"][0]["values"] == [5, 5, 5]
    assert probes["times"] == [0, 2.5, 7]
    assert (bundle / result["views"][0]["poster"]).read_bytes().startswith(b"\x89PNG")
    assert result["views"][0]["media"] is None
    assert any("ffmpeg unavailable" in warning for warning in result["warnings"])
    assert fingerprint(root / "state.nc") == original
    assert "innerHTML" not in (bundle / "viewer.js").read_text()
    with pytest.raises(FileExistsError):
        render(root, bundle, {"movie": False})


def test_solid_probe_gaps_and_outside_errors(tmp_path: Path) -> None:
    ds = regular((0,))
    ds["blanking"] = xr.zeros_like(ds.u)
    ds.blanking.loc[dict(x=4, y=8, z=3)] = 1
    root = save_run(tmp_path, ds)
    output = tmp_path / "bundle"
    render(
        root,
        output,
        {
            "movie": False,
            "width": 320,
            "height": 240,
            "probes": [{"x": 4, "y": 8, "z": 3}],
        },
    )
    assert json.loads((output / "probes.json").read_text())["probes"][0]["values"] == [
        None
    ]
    with pytest.raises(ValueError, match="outside"):
        render(root, tmp_path / "bad", {"probes": [{"x": -100, "y": 8, "z": 3}]})
    assert not (tmp_path / "bad" / "viewer_manifest.json").exists()


def test_asset_server_ranges_and_containment(tmp_path: Path) -> None:
    root = save_run(tmp_path, regular((0,)))
    bundle = tmp_path / "bundle"
    manifest = render(root, bundle, {"movie": False, "width": 320, "height": 240})
    (bundle / "secret.txt").write_text("private")
    with BundleAssetServer() as server:
        url = server.register(bundle)
        with urlopen(url) as response:
            assert response.headers["Content-Type"] == "text/html"
            assert response.read().startswith(b"<!doctype html>")
        poster = url + manifest["views"][0]["poster"]
        with urlopen(Request(poster, headers={"Range": "bytes=0-7"})) as response:
            assert response.status == 206
            assert response.read() == b"\x89PNG\r\n\x1a\n"
        for suffix in ("secret.txt", "../secret.txt", "%2e%2e/secret.txt"):
            with pytest.raises(HTTPError) as error:
                urlopen(url + suffix)
            assert error.value.code == 404
        for headers in (
            {"Host": "attacker.example"},
            {"Origin": "http://attacker.example"},
        ):
            with pytest.raises(HTTPError) as error:
                urlopen(Request(url, headers=headers))
            assert error.value.code == 403
        with pytest.raises(HTTPError) as error:
            urlopen(Request(poster, headers={"Range": "bytes=999999999-"}))
        assert error.value.code == 416
        # Revalidate containment at request time, including changed symlinks.
        image = bundle / manifest["views"][0]["poster"]
        image.unlink()
        image.symlink_to(root / "state.nc")
        with pytest.raises(HTTPError) as error:
            urlopen(poster)
        assert error.value.code == 404


def test_timeline_nonuniform_mapping_and_hours() -> None:
    import shutil

    node = shutil.which("node")
    if not node:
        pytest.skip("Node.js unavailable")
    assert node is not None
    path = Path(__file__).parents[2] / "src/pyurbanair/visualization/web/viewer.js"
    code = f"""
const assert = require('assert');
const t = require({json.dumps(str(path))});
const frames = [{{video_time:0,simulation_time:10}},{{video_time:1,simulation_time:13}},{{video_time:2,simulation_time:21}}];
assert.equal(t.physicalTime(frames,1.5),13);
assert.equal(t.videoTime(frames,20),1);
assert.equal(t.clock(3661),'1:01:01');
assert.equal(t.clock(NaN),'0:00');
"""
    subprocess.run([node, "-e", code], check=True, timeout=10)
