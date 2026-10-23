"""The 3D view is a numpy scene export that the browser draws; no renderer here."""

import json
import shutil
import sys
from importlib import import_module
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import urlopen

import numpy as np
import pytest
import xarray as xr
from evaluation.turbulence import read_binary_stl
from visualization import BundleAssetServer, render
from visualization.__main__ import main
from visualization.render_3d import SPEED_BINS, clip_triangles

from tests.visualization.test_forward_visualization import XIE_CASTRO, regular, save_run


def varied(times: tuple[float, ...] = (0, 2.5, 7)) -> xr.Dataset:
    ds = regular(times)
    ds["u"] = ds.u * 0 + 0.37 * ds.x + ds.time
    ds["v"] = ds.v * 0 - 0.11 * ds.y
    ds["w"] = ds.w * 0 + 0.05 * ds.z * ds.x
    ds["blanking"] = xr.zeros_like(ds.u)
    ds.blanking.loc[dict(x=4, y=8, z=1)] = 1
    return ds


def decode(scene_dir: Path, scene: dict, index: int) -> np.ndarray:
    shape = (len(scene["z"]), len(scene["y"]), len(scene["x"]))
    raw = np.fromfile(scene_dir / scene["velocity"][index], "<i2")
    return raw.reshape(3, *shape) * scene["velocity_scale"][index]


def test_scene_round_trip_mask_and_manifest_view(tmp_path: Path) -> None:
    ds = varied()
    root = save_run(tmp_path, ds)
    (root / "config.yaml").write_text("case_name: cubes\nmodel:\n  name: pylbm\n")
    bundle = tmp_path / "bundle"
    manifest = render(
        root,
        bundle,
        {"render_3d": True, "movie": False, "width": 320, "height": 240},
    )
    assert "pyvista" not in sys.modules and "vtk" not in sys.modules
    flow = manifest["views"][-1]
    assert flow["kind"] == "3d" and flow["id"] == "flow-3d"
    assert flow["scene"] == "media/flow-3d/scene.json"
    assert flow["geometry"] is None
    assert flow["frames"] == manifest["views"][0]["frames"]
    assert flow["duration"] == manifest["views"][0]["duration"]
    scene_dir = bundle / "media" / "flow-3d"
    scene = json.loads((scene_dir / "scene.json").read_text())
    assert (scene["case"], scene["backend"]) == ("cubes", "pylbm")
    assert scene["x"] == [1, 2, 4, 7, 11] and scene["z"] == [1, 3, 6]
    assert scene["times"] == [0, 2.5, 7]
    assert scene["speed_limits"] == flow["color_limits"]
    assert scene["buildings"] is None and scene["camera"] is None
    fluid = np.fromfile(scene_dir / scene["fluid"], np.uint8).reshape(3, 4, 5)
    solid = ds.blanking.isel(time=0).values != 0
    np.testing.assert_array_equal(fluid, ~solid)
    for index in range(3):
        decoded = decode(scene_dir, scene, index)
        expected = np.stack([ds[name].isel(time=index).values for name in "uvw"])
        assert np.all(decoded[:, solid] == 0)
        error = np.abs(decoded - expected)[:, ~solid]
        assert error.max() <= scene["velocity_scale"][index] / 2 * 1.0001
    # [0, 99.5th percentile] of the fluid speed, within one histogram bin.
    speed = np.sqrt(sum(ds[name] ** 2 for name in "uvw")).values[:, ~solid]
    low, high = flow["color_limits"]
    assert low == 0
    exact = np.percentile(speed, 99.5, method="inverted_cdf")
    assert -1e-3 <= high - exact <= speed.max() / SPEED_BINS + 1e-3
    full = np.sqrt(sum(ds[name] ** 2 for name in "uvw")).values
    assert len(scene["level_speed_limits"]) == 3
    for level, (low, high) in enumerate(scene["level_speed_limits"]):
        values = full[:, level][:, ~solid[level]]
        exact = np.percentile(values, 99, method="inverted_cdf")
        assert low == 0
        assert -1e-3 <= high - exact <= speed.max() / SPEED_BINS + 1e-3
    with BundleAssetServer() as server:
        url = server.register(bundle)
        with urlopen(url + "media/flow-3d/velocity/00002.bin") as response:
            assert response.headers["Content-Type"] == "application/octet-stream"
        with urlopen(url + "viewer.js") as response:
            assert response.headers["Content-Type"] == "text/javascript"
        with urlopen(url + "vendor/three.module.min.js") as response:
            assert response.headers["Content-Type"] == "text/javascript"


def test_3d_frames_follow_the_2d_pacing(tmp_path: Path) -> None:
    root = save_run(tmp_path, varied())
    manifest = render(
        root,
        tmp_path / "bundle",
        {
            "render_3d": True,
            "movie": False,
            "fps": 2,
            "width": 320,
            "height": 240,
            "simulation_seconds_per_video_second": 2.5,
        },
    )
    flat, flow = manifest["views"][0], manifest["views"][-1]
    first_2d: dict[float, float] = {}
    for frame in flat["frames"]:
        first_2d.setdefault(frame["simulation_time"], frame["video_time"])
    assert flow["frames"] == [
        {"video_time": video, "simulation_time": time}
        for time, video in first_2d.items()
    ]
    assert flow["duration"] == flat["duration"]


def test_clip_triangles_to_box() -> None:
    lo, hi = np.zeros(3), np.ones(3)
    inside = np.array([[[0.1, 0.1, 0.1], [0.9, 0.1, 0.1], [0.1, 0.9, 0.1]]])
    outside = inside + 5
    # Half of this triangle lies beyond x = 1: a quad, as two triangles.
    crossing = np.array([[[0.5, 0.2, 0.5], [1.5, 0.2, 0.5], [0.5, 0.8, 0.5]]])
    clipped = clip_triangles(np.concatenate([inside, outside, crossing]), lo, hi)
    np.testing.assert_array_equal(clipped[0], inside[0])
    assert len(clipped) == 3
    assert clipped.min() >= 0 and clipped.max() <= 1

    def area(tri: np.ndarray) -> float:
        return float(np.linalg.norm(np.cross(tri[1] - tri[0], tri[2] - tri[0])) / 2)

    # The part with x <= 1 of the triangle (0.5,0.2)-(1.5,0.2)-(0.5,0.8).
    assert sum(area(t) for t in clipped[1:]) == pytest.approx(0.3 - 0.3 / 4)


def test_buildings_cut_to_the_domain(tmp_path: Path) -> None:
    (tmp_path / "geometries").mkdir()
    shutil.copy(XIE_CASTRO, tmp_path / "geometries" / "blocks.stl")
    ds = regular((0, 1)).assign_coords(
        x=[-10, 5, 10, 15, 30], y=[2, 10, 25, 40], z=[1, 5, 25]
    )
    root = save_run(tmp_path / "results", ds)
    (root / "config.yaml").write_text("geometry:\n  stl_path: geometries/blocks.stl\n")
    bundle = tmp_path / "bundle"
    manifest = render(
        root,
        bundle,
        {"render_3d": True, "movie": False, "width": 320, "height": 240},
    )
    flow = manifest["views"][-1]
    bounds = flow["geometry"]["display_bounds"]
    assert bounds == [-17.5, 37.5, -2, 47.5, -1, 35]
    scene_dir = bundle / "media" / "flow-3d"
    scene = json.loads((scene_dir / "scene.json").read_text())
    tri = np.fromfile(scene_dir / scene["buildings"], "<f4").reshape(-1, 3, 3)
    assert len(tri) == flow["geometry"]["triangles"] > 0
    assert len(tri) != len(read_binary_stl(XIE_CASTRO))
    tolerance = 1e-4
    assert np.all(tri.min((0, 1)) >= np.array(bounds[::2]) - tolerance)
    assert np.all(tri.max((0, 1)) <= np.array(bounds[1::2]) + tolerance)
    # Solid cells come from the STL mask and are stored as zeros.
    fluid = np.fromfile(scene_dir / scene["fluid"], np.uint8).reshape(3, 4, 5)
    assert 0 < fluid.sum() < fluid.size
    assert np.all(decode(scene_dir, scene, 0)[:, fluid == 0] == 0)


def test_scene_size_bound_refuses_before_writing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        import_module("visualization.render_3d"), "MAX_SCENE_BYTES", 500
    )
    root = save_run(tmp_path, varied())
    with pytest.raises(ValueError, match="increase stride or shorten"):
        render(root, tmp_path / "bundle", {"render_3d": True, "movie": False})
    assert not (tmp_path / "bundle").exists()
    # 60 cells: 60 + 6 * 60 bytes per frame, so one frame fits in 500.
    render(
        root,
        tmp_path / "strided",
        {"render_3d": True, "movie": False, "stride": 3, "width": 320, "height": 240},
    )


def test_speed_limits_ignore_rare_peaks(tmp_path: Path) -> None:
    ds = varied((0, 1, 2, 3, 4))
    ds["u"] = xr.ones_like(ds.u)
    ds["v"] = xr.zeros_like(ds.v)
    ds["w"] = xr.zeros_like(ds.w)
    ds.u.loc[dict(time=1, x=11, y=12, z=6)] = 50.0  # 1 of 200 fluid samples
    ds.blanking.loc[dict(z=1)] = 1  # a level without fluid
    root = save_run(tmp_path, ds)
    options = {"render_3d": True, "movie": False, "width": 320, "height": 240}
    manifest = render(root, tmp_path / "bundle", options)
    low, high = manifest["views"][-1]["color_limits"]
    assert low == 0 and 1 <= high <= 1 + 50 / SPEED_BINS
    scene = json.loads((tmp_path / "bundle/media/flow-3d/scene.json").read_text())
    none, middle, top = scene["level_speed_limits"]
    assert none is None
    for low, high in (middle, top):  # 1 of 100 at the top: below its p99
        assert low == 0 and 1 <= high <= 1 + 50 / SPEED_BINS
    explicit = render(
        root,
        tmp_path / "explicit",
        {**options, "variable": "speed", "color_limits": [0.5, 2]},
    )
    assert explicit["views"][-1]["color_limits"] == [0.5, 2]
    scene = json.loads((tmp_path / "explicit/media/flow-3d/scene.json").read_text())
    assert scene["speed_limits"] == [0.5, 2]
    assert scene["level_speed_limits"] == [None, [0.5, 2], [0.5, 2]]


def test_bad_stl_keeps_the_2d_bundle(tmp_path: Path) -> None:
    # The STL lies far outside the state domain: the 2D views are published,
    # the 3D view is dropped with a warning and leaves no partial files.
    root = save_run(
        tmp_path, varied((0, 2.5)).assign_coords(x=[1e4, 2e4, 3e4, 4e4, 5e4])
    )
    shutil.copy(XIE_CASTRO, root / "blocks.stl")
    bundle = tmp_path / "bundle"
    manifest = render(
        root,
        bundle,
        {
            "render_3d": True,
            "movie": False,
            "width": 320,
            "height": 240,
            "geometry": str(root / "blocks.stl"),
        },
    )
    assert manifest == json.loads((bundle / "viewer_manifest.json").read_text())
    assert manifest["status"] == "complete"
    assert [view["kind"] for view in manifest["views"]] == ["2d", "2d", "2d"]
    assert any(
        "3D view unavailable" in warning and "does not overlap" in warning
        for warning in manifest["warnings"]
    )
    assert not (bundle / "media" / "flow-3d").exists()
    for view in manifest["views"]:
        for snapshot in view["snapshots"]:
            assert (bundle / snapshot["path"]).read_bytes().startswith(b"\x89PNG")
    with BundleAssetServer() as server:
        server.register(bundle)


def test_cli_skips_3d_without_two_cells_per_axis(tmp_path: Path) -> None:
    root = save_run(tmp_path, varied().isel(z=[1]))
    main(root, tmp_path / "viewer")
    manifest = json.loads((tmp_path / "viewer" / "viewer_manifest.json").read_text())
    assert "3d" not in [view["kind"] for view in manifest["views"]]


def test_cli_strides_to_fit_the_scene(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # 60 cells: 60 + 6 * 60 bytes per frame, so two frames fit in 800.
    monkeypatch.setattr(
        import_module("visualization.render_3d"), "MAX_SCENE_BYTES", 800
    )
    root = save_run(tmp_path, varied((0, 1, 2, 3)))
    main(root, tmp_path / "viewer")
    scene = json.loads((tmp_path / "viewer/media/flow-3d/scene.json").read_text())
    assert scene["times"] == [0, 2]


def get(url: str) -> tuple[int, str | None]:
    try:
        with urlopen(url) as response:
            return response.status, response.headers["Content-Type"]
    except HTTPError as error:
        return error.code, None


def test_server_allows_only_the_bundle_files(tmp_path: Path) -> None:
    # A bundle of an older renderer lacks today's viewer files; it still serves.
    root = save_run(tmp_path, varied())
    options = {"movie": False, "width": 320, "height": 240}
    old = tmp_path / "old"
    render(root, old, options)
    for name in ("flow3d.js", "flow3d_slice.js", "flow3d_volume.js"):
        (old / name).unlink()
    shutil.rmtree(old / "vendor")
    bundle = tmp_path / "bundle"
    render(root, bundle, {**options, "render_3d": True})
    (bundle / "media/flow-3d/velocity/99999.bin").write_bytes(b"stray")
    with BundleAssetServer() as server:
        url = server.register(old)
        assert get(url) == (200, "text/html")
        assert get(url + "flow3d.js")[0] == 404
        url = server.register(bundle)
        scene = url + "media/flow-3d/"
        assert get(scene + "scene.json") == (200, "application/json")
        for name in ("fluid.bin", "velocity/00000.bin"):
            assert get(scene + name) == (200, "application/octet-stream")
        assert get(url + "flow3d.js") == (200, "text/javascript")
        assert get(scene + "velocity/99999.bin")[0] == 404
    # Scene binaries are named relative to scene.json and may not leave the bundle.
    (tmp_path / "x").write_text("private")
    path = bundle / "media/flow-3d/scene.json"
    for outside in ("../../../x", "../../../../x"):
        path.write_text(json.dumps({**json.loads(path.read_text()), "fluid": outside}))
        with BundleAssetServer() as server, pytest.raises(ValueError, match="escapes"):
            server.register(bundle)
