"""Tests for the Unreal Engine stage (les_render.unreal).

Unreal is not available here, so this covers (a) the pure helpers shared by
``prepare.py`` and ``build_scene.py`` (frame conversion, look-at rotators,
camera baking, placement maths), (b) ``prepare_unreal`` on a synthetic bundle,
(c) that the editor scripts compile and only import allowed modules, and (d)
an end-to-end smoke run of ``build_scene.main`` / ``render.main`` against a
fake ``unreal`` module (catches Python-level bugs, not UE API semantics).
"""

from __future__ import annotations

import ast
import importlib.util
import json
import math
import pathlib
import py_compile
import struct
import subprocess
import sys
import types
import zlib
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest
from les_render import colormaps
from les_render.cameras import make_shots
from les_render.cameras import sample_camera as ref_sample_camera
from les_render.timeline import Timeline
from les_render.unreal import build_scene as bs
from les_render.unreal import prepare
from les_render.unreal import render as rnd

UNREAL_DIR = pathlib.Path(bs.__file__).parent


# -- synthetic bundle ---------------------------------------------------------------


def _geometry() -> dict[str, Any]:
    footprints = [
        {"min": [20.0, 10.0, 0.0], "max": [30.0, 20.0, 18.0]},
        {"min": [20.0, 30.0, 0.0], "max": [30.0, 40.0, 12.0]},
        {"min": [45.0, 10.0, 0.0], "max": [55.0, 20.0, 25.0]},
        {"min": [45.0, 30.0, 0.0], "max": [55.0, 40.0, 9.0]},
    ]
    return {
        "buildings": {"obj": "geometry/buildings.obj", "glb": "geometry/buildings.glb"},
        "ground": {"obj": "geometry/ground.obj", "glb": "geometry/ground.glb"},
        "buildings_bounds": [[20.0, 10.0, 0.0], [55.0, 40.0, 25.0]],
        "max_building_height": 25.0,
        "footprints": footprints,
    }


def _color(name: str, lo: float, hi: float, var: str) -> dict[str, Any]:
    return dict(colormaps.layer_color_spec(name, lo, hi, var))


def make_manifest(
    n_frames: int = 40, fps: float = 30.0, look: str = "dark"
) -> dict[str, Any]:
    domain = {
        "lower": [0.0, 0.0, 0.0],
        "upper": [80.0, 50.0, 40.0],
        "spacing": [1.0, 1.0, 1.0],
        "shape": [80, 50, 40],
    }
    geometry = _geometry()
    tl = Timeline(fps, 20.0, 0.0, n_frames)
    layers = [
        {
            "name": "speed_glow",
            "type": "volume",
            "pattern": "volumes/speed_glow/speed_glow.{frame:04d}.vdb",
            "frame_step": 4,
            "n_files": math.ceil(n_frames / 4),
            "grid": "speed_glow",
            "voxel_size": [0.5, 0.5, 0.5],
            "origin": [0.25, 0.25, 0.25],
            "shape": [160, 100, 80],
            "density_range": [0.2, 1.5],
            "emission_strength": 1.5,
            "density_scale": 1.0,
            **_color("inferno", 0.2, 1.5, "speed"),
        },
        {
            "name": "streaklines",
            "type": "particles",
            "kind": "streaklines",
            "pattern": "particles/streaklines/streaklines.{frame:04d}.npz",
            "frame_step": 1,
            "n_files": n_frames,
            "n_lines": 100,
            "points_per_line": 16,
            "radius": 0.05,
            "emission_strength": 2.0,
            **_color("viridis", 0.0, 5.0, "speed"),
        },
        {
            "name": "vortices",
            "type": "isosurface",
            "pattern": "isosurfaces/vortices/vortices.{frame:04d}.ply",
            "frame_step": 2,
            "n_files": n_frames // 2,
            "iso_variable": "q_criterion",
            "level": 0.1,
            **_color("magma", 0.0, 5.0, "speed"),
        },
        {
            "name": "ground_speed",
            "type": "slice",
            "pattern": "slices/ground_speed/ground_speed.{frame:04d}.png",
            "frame_step": 1,
            "n_files": n_frames,
            "axis": "z",
            "position": 2.0,
            "extent": [[0.0, 0.0], [80.0, 50.0]],
            "resolution": [320, 200],
            **_color("inferno", 0.0, 5.0, "speed"),
        },
    ]
    manifest = {
        "version": 1,
        "case": {
            "name": "test case-01",
            "state": "state.nc",
            "geometry": None,
            "params": None,
        },
        "frame": {"units": "m", "handedness": "right", "up": "z"},
        "domain": domain,
        "timeline": tl.to_dict(),
        "render": {"width": 960, "height": 540, "look": look, "preset": "test"},
        "geometry": geometry,
        "inflow": None,
        "layers": layers,
        "hud": {"pattern": "hud/hud.{frame:04d}.png", "frame_step": 1},
    }
    manifest["shots"] = make_shots(geometry, domain, tl)
    return manifest


def make_bundle(root: pathlib.Path, **kw: Any) -> tuple[pathlib.Path, dict[str, Any]]:
    manifest = make_manifest(**kw)
    root.mkdir(parents=True, exist_ok=True)
    (root / "manifest.json").write_text(json.dumps(manifest))
    for rel in (
        "geometry/buildings.glb",
        "geometry/ground.glb",
        "alembic/streaklines.abc",
        "alembic/vortices.abc",
    ):
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"stub")
    for layer in manifest["layers"]:
        for f in range(min(layer["n_files"], 3)):
            p = root / layer["pattern"].format(frame=f)
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(b"stub")
    return root, manifest


@pytest.fixture()  # type: ignore[misc]
def bundle(tmp_path: pathlib.Path) -> tuple[pathlib.Path, dict[str, Any]]:
    return make_bundle(tmp_path / "bundle")


# -- coordinate helpers ---------------------------------------------------------------


def test_sim_to_ue_scales_and_mirrors_y() -> None:
    assert bs.sim_to_ue((1.0, 2.0, 3.0)) == (100.0, -200.0, 300.0)
    assert bs.sim_dir_to_ue((1.0, 2.0, 3.0)) == (1.0, -2.0, 3.0)
    assert prepare.sim_to_ue is bs.sim_to_ue


def test_look_at_rotator_axes() -> None:
    # +x in sim == +X in Unreal: yaw 0, level
    assert bs.look_at_rotator((0, 0, 10), (10, 0, 10)) == pytest.approx((0.0, 0.0, 0.0))
    # +y in sim is -Y in Unreal (mirrored): yaw -90
    assert bs.look_at_rotator((0, 0, 10), (0, 10, 10)) == pytest.approx(
        (0.0, -90.0, 0.0)
    )
    assert bs.look_at_rotator((0, 0, 10), (0, -10, 10)) == pytest.approx(
        (0.0, 90.0, 0.0)
    )
    assert bs.look_at_rotator((0, 0, 10), (-10, 0, 10))[1] == pytest.approx(180.0)
    # looking down 45 degrees
    assert bs.look_at_rotator((0, 0, 10), (10, 0, 0)) == pytest.approx(
        (-45.0, 0.0, 0.0)
    )
    # straight down keeps the previous yaw
    p, y, r = bs.look_at_rotator((5, 5, 10), (5, 5, 0), prev_yaw=33.0)
    assert (p, y, r) == pytest.approx((-90.0, 33.0, 0.0))


def test_look_at_rotator_roundtrip_forward() -> None:
    rng = np.random.default_rng(0)
    for _ in range(200):
        loc, tgt = rng.normal(size=3) * 50, rng.normal(size=3) * 50
        pitch, yaw, roll = bs.look_at_rotator(loc, tgt)
        fwd = np.array(bs.rotator_forward(pitch, yaw))
        want = np.array(bs.sim_dir_to_ue(tgt - loc))
        np.testing.assert_allclose(fwd, want / np.linalg.norm(want), atol=1e-9)
        assert roll == 0.0


def test_yaw_unwrap_is_continuous_through_180() -> None:
    # orbit behind the target: sim yaw crosses +-180 in Unreal
    shots = [
        {
            "name": "orbit",
            "start": 0,
            "end": 60,
            "keys": [
                {
                    "frame": 0,
                    "location": [10.0, -3.0, 5.0],
                    "target": [0, 0, 0],
                    "focal_length_mm": 35,
                    "fstop": 4,
                },
                {
                    "frame": 30,
                    "location": [10.0, 0.0, 5.0],
                    "target": [0, 0, 0],
                    "focal_length_mm": 35,
                    "fstop": 4,
                },
                {
                    "frame": 60,
                    "location": [10.0, 3.0, 5.0],
                    "target": [0, 0, 0],
                    "focal_length_mm": 35,
                    "fstop": 4,
                },
            ],
        }
    ]
    keys = bs.bake_camera(shots, 1)["orbit"]
    yaws = [k["rotation"][1] for k in keys]
    assert max(abs(a - b) for a, b in zip(yaws, yaws[1:])) < 5.0
    assert max(abs(y) for y in yaws) > 179.0  # really passes behind the target
    assert keys[0]["focus_distance_cm"] == pytest.approx(
        100 * math.dist([10, -3, 5], [0, 0, 0])
    )


def test_fps_fraction_and_file_index() -> None:
    assert bs.fps_to_fraction(30) == (30, 1)
    assert bs.fps_to_fraction(29.97) == (30000, 1001)
    assert bs.fps_to_fraction(23.976) == (24000, 1001)
    assert bs.fps_to_fraction(12.5) == (12500, 1000)
    assert [bs.file_index(f, 4, 3) for f in range(14)] == [0] * 4 + [1] * 4 + [2] * 6
    layer = {"frame_step": 4, "n_files": 3}
    assert bs.volume_frame_keys(layer, 14) == [(0, 0), (4, 1), (8, 2)]


# -- cameras ---------------------------------------------------------------------------


def test_sample_camera_matches_reference(
    bundle: tuple[pathlib.Path, dict[str, Any]]
) -> None:
    _, manifest = bundle
    shots = manifest["shots"]
    for f in range(manifest["timeline"]["n_frames"]):
        ours = bs.sample_camera(shots, f)
        loc, tgt, focal = ref_sample_camera(shots, f)
        np.testing.assert_allclose(ours["location"], loc, atol=1e-4)
        np.testing.assert_allclose(ours["target"], tgt, atol=1e-4)
        assert ours["focal_length_mm"] == pytest.approx(focal)


def test_sample_camera_matches_blender_preview(
    bundle: tuple[pathlib.Path, dict[str, Any]]
) -> None:
    path = UNREAL_DIR.parent / "blender" / "bundle.py"
    if not path.exists():
        pytest.skip("blender bundle module not present")
    spec = importlib.util.spec_from_file_location("_les_blender_bundle", path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
    except ImportError as exc:
        pytest.skip(f"blender bundle module not importable here: {exc}")
    _, manifest = bundle
    for f in range(manifest["timeline"]["n_frames"]):
        a, b = bs.sample_camera(manifest["shots"], f), mod.sample_camera(
            manifest["shots"], f
        )
        np.testing.assert_allclose(a["location"], b["location"], atol=1e-9)
        np.testing.assert_allclose(a["target"], b["target"], atol=1e-9)
        assert a["fstop"] == pytest.approx(b["fstop"])
        assert a["shot"] == b["shot"]


def test_bake_camera_covers_every_shot(
    bundle: tuple[pathlib.Path, dict[str, Any]]
) -> None:
    _, manifest = bundle
    baked = bs.bake_camera(manifest["shots"], step=3)
    for shot in manifest["shots"]:
        frames = [k["frame"] for k in baked[shot["name"]]]
        assert frames[0] == shot["start"] and frames[-1] == shot["end"]
        assert frames == sorted(set(frames))


def test_layer_visibility_keys(bundle: tuple[pathlib.Path, dict[str, Any]]) -> None:
    _, manifest = bundle
    vis = bs.layer_visibility_keys(manifest)
    assert set(vis) == {layer["name"] for layer in manifest["layers"]}
    shots = sorted(manifest["shots"], key=lambda s: s["start"])
    for name, keys in vis.items():
        assert keys[0][0] == shots[0]["start"]
        for shot in shots:  # the state in force at each shot start is right
            state = [v for f, v in keys if f <= shot["start"]][-1]
            assert state == ("layers" not in shot or name in shot["layers"])
    # a shot without "layers" shows everything
    m = {"layers": [{"name": "a"}], "shots": [{"name": "s", "start": 0, "end": 9}]}
    assert bs.layer_visibility_keys(m) == {"a": [(0, True)]}


# -- placement maths ---------------------------------------------------------------------


def test_volume_transform_modes() -> None:
    layer = {
        "voxel_size": [0.5, 0.5, 0.5],
        "origin": [0.25, 0.25, 0.25],
        "shape": [160, 100, 80],
    }
    w = bs.volume_transform(layer, "world")
    assert w["location"] == (0.0, 0.0, 0.0) and w["scale"] == (100.0, -100.0, 100.0)
    i = bs.volume_transform(layer, "index")
    assert i["scale"] == (50.0, -50.0, 50.0) and i["location"] == (25.0, -25.0, 25.0)
    # auto: bounds in VDB world units (metres) -> world; in voxel units -> index
    assert (
        bs.volume_transform(layer, "auto", ((10.0, 5.0, 0.0), (60.0, 45.0, 30.0)))[
            "mode"
        ]
        == "world"
    )
    assert (
        bs.volume_transform(layer, "auto", ((0.0, 0.0, 0.0), (150.0, 90.0, 70.0)))[
            "mode"
        ]
        == "index"
    )
    assert bs.volume_transform(layer, "auto", None)["mode"] == "world"
    # engine version decides; contradicting bounds win
    assert bs.volume_transform(layer, "auto", None, (5, 5))["mode"] == "world"
    assert bs.volume_transform(layer, "auto", None, (5, 3))["mode"] == "index"
    assert (
        bs.volume_transform(
            layer, "auto", ((0.0, 0.0, 0.0), (150.0, 90.0, 70.0)), (5, 5)
        )["mode"]
        == "index"
    )
    assert bs.parse_engine_version("5.5.1-37573402+++UE5+Release-5.5") == (5, 5)
    assert bs.parse_engine_version("garbage") is None
    lo, hi = bs.volume_world_box(layer)
    assert lo == (0.0, 0.0, 0.0) and hi == (80.0, 50.0, 40.0)


@pytest.mark.parametrize(  # type: ignore[misc]
    "mapping",
    [
        lambda x, y, z: (x, y, z),  # importer already right
        lambda x, y, z: (x / 100, y / 100, z / 100),  # metres read as cm
        lambda x, y, z: (x, -y, z),  # missing Y mirror
        lambda x, y, z: (-y, x, z),  # x/y swapped (other glTF axis convention)
        lambda x, y, z: (y / 100, x / 100, z / 100),  # swap + mirror + x100
    ],
)
def test_geometry_fix_recovers_expected_bounds(mapping: Any) -> None:
    lo_m, hi_m = (20.0, 10.0, 0.0), (55.0, 40.0, 25.0)
    corners = [
        bs.sim_to_ue((x, y, z))
        for x in (lo_m[0], hi_m[0])
        for y in (lo_m[1], hi_m[1])
        for z in (lo_m[2], hi_m[2])
    ]
    e_lo = tuple(min(c[k] for c in corners) for k in range(3))
    e_hi = tuple(max(c[k] for c in corners) for k in range(3))
    actual = [mapping(*c) for c in corners]
    a_lo = tuple(min(c[k] for c in actual) for k in range(3))
    a_hi = tuple(max(c[k] for c in actual) for k in range(3))
    fix = bs.geometry_fix(e_lo, e_hi, a_lo, a_hi)
    assert fix["error"] < 1e-6
    for c, a in zip(corners, actual):  # every corner lands where it should
        np.testing.assert_allclose(bs.apply_fix(a, fix), c, atol=1e-6)


def test_geometry_fix_identity_note() -> None:
    fix = bs.geometry_fix((0, -10, 0), (10, 0, 5), (0, -10, 0), (10, 0, 5))
    assert fix["note"] == "ok" and fix["scale"] == (1.0, 1.0, 1.0) and fix["yaw"] == 0.0


def test_slice_quad_uv_orientation() -> None:
    layer = {"axis": "z", "position": 2.0, "extent": [[0.0, 0.0], [80.0, 50.0]]}
    corners, uvs = bs.slice_quad(layer)
    assert corners == [
        (0.0, 0.0, 2.0),
        (80.0, 0.0, 2.0),
        (80.0, 50.0, 2.0),
        (0.0, 50.0, 2.0),
    ]
    # glTF UV origin is top-left; PNG row 0 (V = 0) is the max-v (max y) edge
    assert uvs[3] == (0.0, 0.0) and uvs[0] == (0.0, 1.0)
    corners_x, _ = bs.slice_quad(
        {"axis": "x", "position": 5.0, "extent": [[1.0, 2.0], [3.0, 4.0]]}
    )
    assert corners_x[0] == (5.0, 1.0, 2.0) and corners_x[2] == (5.0, 3.0, 4.0)


# -- manifest / plan / args ---------------------------------------------------------------------


def test_parse_manifest_validates() -> None:
    m = make_manifest()
    assert bs.parse_manifest(json.loads(json.dumps(m)))["version"] == 1
    bad = json.loads(json.dumps(m))
    bad["version"] = 2
    with pytest.raises(ValueError):
        bs.parse_manifest(bad)
    bad = json.loads(json.dumps(m))
    del bad["shots"]
    with pytest.raises(ValueError):
        bs.parse_manifest(bad)


def test_build_plan(bundle: tuple[pathlib.Path, dict[str, Any]]) -> None:
    root, manifest = bundle
    plan = bs.build_plan(bs.parse_manifest(manifest), root)
    assert plan["case"] == "test_case_01"
    assert plan["map"] == "/Game/LES/test_case_01/Maps/LES_test_case_01"
    assert plan["fps"] == (30, 1) and plan["resolution"] == (960, 540)
    by = {i["name"]: i for i in plan["layers"]}
    assert (
        by["speed_glow"]["source"].endswith("volumes/speed_glow/speed_glow.0000.vdb")
        and by["speed_glow"]["exists"]
    )
    assert (
        by["streaklines"]["source"].endswith("alembic/streaklines.abc")
        and by["streaklines"]["mesh_source"] is None
    )
    assert by["ground_speed"]["source"].endswith("slices/ground_speed")


def test_parse_args() -> None:
    o = bs.parse_args(["--bundle", "/b", "--quit", "--skip", "mrq,save"], env={})
    assert o == {
        "bundle": "/b",
        "quit": True,
        "renderer": None,
        "skip": {"mrq", "save"},
    }
    assert (
        bs.parse_args([], env={"LES_BUNDLE": "/env"}, default_bundle="/baked")["bundle"]
        == "/env"
    )
    assert bs.parse_args([], env={}, default_bundle="/baked")["bundle"] == "/baked"
    r = rnd.parse_args(
        ["--bundle=/b", "--project", "P.uproject", "--dry-run", "--extra", "-foo -bar"],
        env={},
    )
    assert (
        r["bundle"] == "/b"
        and r["project"] == "P.uproject"
        and r["dry_run"]
        and r["extra"] == ["-foo", "-bar"]
    )


# -- prepare_unreal ---------------------------------------------------------------------------------


def _read_png(data: bytes) -> tuple[int, int, bytes]:
    assert data[:8] == b"\x89PNG\r\n\x1a\n"
    pos, idat = 8, b""
    w = h = 0
    while pos < len(data):
        n = struct.unpack(">I", data[pos : pos + 4])[0]
        tag, body = data[pos + 4 : pos + 8], data[pos + 8 : pos + 8 + n]
        crc = struct.unpack(">I", data[pos + 8 + n : pos + 12 + n])[0]
        assert crc == zlib.crc32(tag + body) & 0xFFFFFFFF
        if tag == b"IHDR":
            w, h = struct.unpack(">II", body[:8])
        elif tag == b"IDAT":
            idat += body
        pos += 12 + n
    raw = zlib.decompress(idat)
    return w, h, raw


def test_lut_png_roundtrip() -> None:
    lut = colormaps.lut_list("inferno")
    w, h, raw = _read_png(prepare.lut_png(lut))
    assert (w, h) == (256, 1) and raw[0] == 0
    px = np.frombuffer(raw[1:], dtype=np.uint8).reshape(256, 4)
    want = np.rint(colormaps.lut("inferno", linear=False) * 255)
    assert np.abs(px[:, :3].astype(int) - want).max() <= 1
    assert (px[:, 3] == 255).all()


def _read_glb(path: pathlib.Path) -> tuple[dict[str, Any], bytes]:
    data = path.read_bytes()
    assert data[:4] == b"glTF"
    jlen = struct.unpack("<I", data[12:16])[0]
    doc = json.loads(data[20 : 20 + jlen])
    blen = struct.unpack("<I", data[20 + jlen : 24 + jlen])[0]
    return doc, data[28 + jlen : 28 + jlen + blen]


def _accessor(doc: dict[str, Any], binary: bytes, idx: int) -> np.ndarray:
    acc = doc["accessors"][idx]
    view = doc["bufferViews"][acc["bufferView"]]
    ncomp = {"VEC2": 2, "VEC3": 3, "SCALAR": 1}[acc["type"]]
    start = view.get("byteOffset", 0) + acc.get("byteOffset", 0)
    arr = np.frombuffer(
        binary[start : start + 4 * ncomp * acc["count"]], dtype=np.float32
    )
    return arr.reshape(acc["count"], ncomp)


def test_prepare_unreal(bundle: tuple[pathlib.Path, dict[str, Any]]) -> None:
    root, manifest = bundle
    res = prepare.prepare_unreal(root)
    out = root / "unreal"
    assert pathlib.Path(res["dir"]) == out
    # scripts copied with the bundle path baked in, still compile
    for name in ("build_scene.py", "render.py"):
        text = (out / name).read_text()
        assert f"DEFAULT_BUNDLE = {str(root.resolve())!r}" in text
        py_compile.compile(str(out / name), doraise=True)
    # LUTs
    assert set(res["luts"]) == {layer["name"] for layer in manifest["layers"]}
    for p in res["luts"].values():
        assert _read_png(pathlib.Path(p).read_bytes())[:2] == (256, 1)
    # slice plane: positions in glTF y-up, V = 0 on the max-y edge
    doc, binary = _read_glb(out / "meshes" / "ground_speed_plane.glb")
    prim = doc["meshes"][0]["primitives"][0]
    pos = _accessor(doc, binary, prim["attributes"]["POSITION"])
    uv = _accessor(doc, binary, prim["attributes"]["TEXCOORD_0"])
    sim = np.stack([pos[:, 0], -pos[:, 2], pos[:, 1]], axis=1)  # y-up -> z-up
    np.testing.assert_allclose(sim[:, 2], 2.0, atol=1e-5)
    for p_sim, t in zip(sim, uv):
        assert t[0] == pytest.approx(p_sim[0] / 80.0, abs=1e-5)
        assert t[1] == pytest.approx(1.0 - p_sim[1] / 50.0, abs=1e-5)
    # camera bake: one sample per video frame, from the reference sampler
    bake = json.loads((out / "camera_bake.json").read_text())["frames"]
    assert [s["frame"] for s in bake] == list(range(manifest["timeline"]["n_frames"]))
    loc, _, _ = ref_sample_camera(manifest["shots"], 7)
    np.testing.assert_allclose(bake[7]["location"], loc)
    # docs + commands
    readme = (out / "README.md").read_text()
    assert "{{" not in readme and "test_case_01" in readme and "streaklines" in readme
    sh = (out / "ue_commands.sh").read_text()
    assert "-ExecutePythonScript=" in sh and "render.py" in sh and "hud.%04d.png" in sh
    assert (out / "ue_commands.sh").stat().st_mode & 0o111
    assert b"\r\n" in (out / "ue_commands.bat").read_bytes()
    assert subprocess.run(["bash", "-n", str(out / "ue_commands.sh")]).returncode == 0


# -- editor scripts: compile + import hygiene ------------------------------------------------------

ALLOWED = {
    "build_scene.py": {"unreal", "json", "math", "os", "pathlib", "sys", "__future__"},
    "render.py": {
        "unreal",
        "json",
        "math",
        "os",
        "pathlib",
        "sys",
        "shlex",
        "subprocess",
        "__future__",
    },
}


@pytest.mark.parametrize("name", sorted(ALLOWED))  # type: ignore[misc]
def test_editor_scripts_compile_and_import_only_allowed(name: str) -> None:
    path = UNREAL_DIR / name
    py_compile.compile(str(path), doraise=True)
    tree = ast.parse(path.read_text())
    mods = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            mods |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom):
            assert node.level == 0, "no relative imports in editor scripts"
            mods.add((node.module or "").split(".")[0])
    assert mods <= ALLOWED[name], mods - ALLOWED[name]


def test_editor_scripts_have_no_side_effects_on_import() -> None:
    # importing without unreal must not raise nor run main()
    assert bs.unreal is None and rnd.unreal is None
    with pytest.raises(SystemExit):
        bs.main([])


# -- render.py outside Unreal ----------------------------------------------------------------------


def _scene(root: pathlib.Path) -> dict[str, Any]:
    scene = {
        "case": "c",
        "map": "/Game/LES/c/Maps/LES_c",
        "sequence": "/Game/LES/c/Sequences/LS_c.LS_c",
        "mrq_config": "/Game/LES/c/Render/MRQ_c.MRQ_c",
        "output_dir": str(root / "unreal" / "render"),
        "resolution": [960, 540],
        "fps": [30, 1],
        "n_frames": 3,
        "output_format": "png",
    }
    (root / "unreal").mkdir(parents=True, exist_ok=True)
    (root / "unreal" / "ue_scene.json").write_text(json.dumps(scene))
    return scene


def test_render_command(tmp_path: pathlib.Path) -> None:
    scene = _scene(tmp_path)
    cmd = rnd.build_render_command(
        scene, "/p/P.uproject", "/ue/UnrealEditor-Cmd", offscreen=True
    )
    assert cmd[:4] == [
        "/ue/UnrealEditor-Cmd",
        "/p/P.uproject",
        "/Game/LES/c/Maps/LES_c",
        "-game",
    ]
    assert "-LevelSequence=/Game/LES/c/Sequences/LS_c.LS_c" in cmd
    assert "-MoviePipelineConfig=/Game/LES/c/Render/MRQ_c.MRQ_c" in cmd
    assert "-ResX=960" in cmd and "-ResY=540" in cmd and cmd[-1] == "-RenderOffscreen"
    assert [p.name for p in rnd.expected_frames(scene)] == [
        "LS_c.0000.png",
        "LS_c.0001.png",
        "LS_c.0002.png",
    ]


def test_render_dry_run_cli(tmp_path: pathlib.Path) -> None:
    scene = _scene(tmp_path)
    proc = subprocess.run(
        [
            sys.executable,
            str(UNREAL_DIR / "render.py"),
            "--bundle",
            str(tmp_path),
            "--project",
            "P.uproject",
            "--dry-run",
        ],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip().splitlines()[-1] == scene["output_dir"]


# -- smoke run against a fake `unreal` module ------------------------------------------------------


class _Vec:
    def __init__(self, x: float = 0.0, y: float = 0.0, z: float = 0.0) -> None:
        self.x, self.y, self.z = x, y, z


class _Meta(type):
    def __getattr__(cls, name: str) -> Any:
        if name.startswith("_"):
            raise AttributeError(name)
        if name.upper() == name:  # enum member
            return f"{cls.__name__}.{name}"
        m = MagicMock(name=f"{cls.__name__}.{name}")
        setattr(cls, name, m)
        return m


class _Obj(metaclass=_Meta):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.__dict__["_props"] = dict(kwargs)
        self.__dict__["_args"] = args

    def set_editor_property(self, k: str, v: Any) -> None:
        self._props[k] = v

    def get_editor_property(self, k: str) -> Any:
        if k not in self._props:
            self._props[k] = (
                []
                if k in ("tags", "imported_object_paths", "static_materials")
                else MagicMock(name=k)
            )
        return self._props[k]

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            raise AttributeError(name)
        m = MagicMock(name=name)
        self.__dict__[name] = m
        return m


class _Actor(_Obj):
    def get_actor_bounds(self, only_colliding: bool) -> tuple[_Vec, _Vec]:
        return _Vec(3750.0, -2500.0, 1250.0), _Vec(1750.0, 1500.0, 1250.0)


def _fake_unreal(created: dict[str, list]) -> types.ModuleType:
    classes: dict[str, type] = {"Vector": _Vec}
    mod = types.ModuleType("unreal")

    def cls(name: str) -> type:
        if name not in classes:
            classes[name] = _Meta(name, (_Obj,), {})
        return classes[name]

    def mod_getattr(name: str) -> Any:
        if name.startswith("_"):
            raise AttributeError(name)
        return cls(name)

    setattr(mod, "__getattr__", mod_getattr)
    mod.log = lambda msg: created["log"].append(msg)  # type: ignore[attr-defined]
    mod.log_warning = lambda msg: created["warn"].append(msg)  # type: ignore[attr-defined]
    mod.log_error = lambda msg: created["error"].append(msg)  # type: ignore[attr-defined]

    def imported(task: Any) -> list:
        fn = str(task._props["filename"])
        factory = type(task._props.get("factory")).__name__
        if fn.endswith(".png"):
            return [cls("Texture2D")()]
        if fn.endswith(".glb"):
            return [cls("StaticMesh")()]
        if fn.endswith(".vdb"):
            return [cls("StreamingSparseVolumeTexture")()]
        if factory == "HairStrandsFactory":
            return [cls("GroomAsset")(), cls("GroomCache")()]
        if factory == "AlembicImportFactory":
            return [cls("GeometryCache")()]
        return []

    class _Task(_Obj):
        def get_objects(self) -> list:
            return imported(self)

    classes["AssetImportTask"] = _Meta("AssetImportTask", (_Task,), {})

    seq = MagicMock(name="asset")
    seq.get_bindings.return_value = []
    seq.get_tracks.return_value = []
    chans = [MagicMock(name=f"chan{i}") for i in range(9)]
    for c in chans:
        c.get_editor_property.return_value = "?"
    binding = seq.add_possessable.return_value
    binding.add_track.return_value.add_section.return_value.get_all_channels.return_value = (
        chans
    )
    tools = MagicMock(name="asset_tools")
    tools.create_asset.return_value = seq
    tools.import_asset_tasks.side_effect = lambda tasks: created["imports"].extend(
        t._props["filename"] for t in tasks
    )
    setattr(cls("AssetToolsHelpers"), "get_asset_tools", MagicMock(return_value=tools))

    eal = cls("EditorAssetLibrary")
    setattr(eal, "does_asset_exist", MagicMock(return_value=False))
    setattr(eal, "does_directory_exist", MagicMock(return_value=True))
    setattr(eal, "list_assets", MagicMock(return_value=[]))

    subsystems: dict[str, Any] = {}

    def get_editor_subsystem(c: type) -> Any:
        name = c.__name__
        if name not in subsystems:
            m = MagicMock(name=name)
            if name == "EditorActorSubsystem":

                def spawn(*a: Any, **k: Any) -> _Actor:
                    actor = _Actor()
                    created["actors"].append(actor)
                    return actor

                m.spawn_actor_from_class.side_effect = spawn
                m.spawn_actor_from_object.side_effect = spawn
                m.get_all_level_actors.return_value = []
            if name == "MoviePipelineQueueSubsystem":
                m.get_queue.return_value.get_jobs.return_value = []
                m.is_rendering.return_value = False
            subsystems[name] = m
        return subsystems[name]

    mod.get_editor_subsystem = get_editor_subsystem  # type: ignore[attr-defined]
    mod._subsystems = subsystems  # type: ignore[attr-defined]
    mod._seq = seq  # type: ignore[attr-defined]
    return mod


def _load_with_fake(
    name: str, fake: types.ModuleType, monkeypatch: pytest.MonkeyPatch
) -> Any:
    monkeypatch.setitem(sys.modules, "unreal", fake)
    spec = importlib.util.spec_from_file_location(
        f"_fake_ue_{name}", UNREAL_DIR / f"{name}.py"
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_build_scene_smoke_with_fake_unreal(
    bundle: tuple[pathlib.Path, dict[str, Any]], monkeypatch: pytest.MonkeyPatch
) -> None:
    root, manifest = bundle
    prepare.prepare_unreal(root)
    created: dict[str, list] = {
        "log": [],
        "warn": [],
        "error": [],
        "imports": [],
        "actors": [],
    }
    fake = _fake_unreal(created)
    mod = _load_with_fake("build_scene", fake, monkeypatch)
    assert mod.unreal is fake
    ctx = mod.main(["--bundle", str(root)])
    statuses = {stage: (status, msg) for stage, status, msg in mod.REPORT}
    failed = {k: v for k, v in statuses.items() if v[0] != "ok"}
    assert not failed, failed
    assert len(statuses) == len(mod.STAGES)
    assert not created["error"]
    # every layer got an actor, plus geometry, lights, post-process and one camera per shot
    assert set(ctx["actors"]) >= {
        "buildings",
        "ground",
        "speed_glow",
        "streaklines",
        "vortices",
        "ground_speed",
    }
    n_cams = len(manifest["shots"])
    assert len(created["actors"]) >= 2 + 4 + 2 + n_cams
    # every bundle source went through an import task
    imported = {pathlib.Path(p).name for p in created["imports"]}
    assert {
        "buildings.glb",
        "ground.glb",
        "speed_glow.0000.vdb",
        "streaklines.abc",
        "vortices.abc",
        "ground_speed_plane.glb",
        "speed_glow_lut.png",
    } <= imported
    # the camera transform channels got one key per baked frame
    chans = (
        fake._seq.add_possessable.return_value.add_track.return_value.add_section.return_value.get_all_channels.return_value
    )
    assert chans[0].add_key.call_count >= manifest["timeline"]["n_frames"]
    scene = json.loads((root / "unreal" / "ue_scene.json").read_text())
    assert (
        scene["sequence"].endswith("LS_test_case_01.LS_test_case_01")
        and scene["n_frames"] == manifest["timeline"]["n_frames"]
    )
    assert all(r["status"] == "ok" for r in scene["report"])


def test_build_scene_stage_failure_is_contained(
    bundle: tuple[pathlib.Path, dict[str, Any]], monkeypatch: pytest.MonkeyPatch
) -> None:
    root, _ = bundle
    prepare.prepare_unreal(root)
    created: dict[str, list] = {
        "log": [],
        "warn": [],
        "error": [],
        "imports": [],
        "actors": [],
    }
    fake = _fake_unreal(created)
    mod = _load_with_fake("build_scene", fake, monkeypatch)
    # make every heterogeneous-volume spawn blow up
    actor_sys = fake.get_editor_subsystem(fake.EditorActorSubsystem)
    original = actor_sys.spawn_actor_from_class.side_effect

    def spawn(c: Any, *a: Any) -> Any:
        if c.__name__ == "HeterogeneousVolume":
            raise RuntimeError("boom")
        return original(c, *a)

    actor_sys.spawn_actor_from_class.side_effect = spawn
    mod.main(["--bundle", str(root)])
    statuses = {stage: status for stage, status, _ in mod.REPORT}
    assert statuses["volumes"] == "failed"
    assert sum(1 for s in statuses.values() if s == "ok") == len(mod.STAGES) - 1
    assert any("Manual fallback" in w for w in created["warn"])


def test_render_in_editor_with_fake_unreal(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    scene = _scene(tmp_path)
    created: dict[str, list] = {
        "log": [],
        "warn": [],
        "error": [],
        "imports": [],
        "actors": [],
    }
    fake = _fake_unreal(created)
    mod = _load_with_fake("render", fake, monkeypatch)
    executor = mod.main(["--bundle", str(tmp_path)])
    assert executor is not None and not created["error"]
    q = fake._subsystems["MoviePipelineQueueSubsystem"]
    q.render_queue_with_executor.assert_called_once()
    assert any(scene["output_dir"] in m for m in created["log"])


# -- VDB compatibility checks -------------------------------------------------------------------------


def test_vdb_file_version_header(tmp_path: pathlib.Path) -> None:
    p = tmp_path / "a.vdb"
    p.write_bytes(struct.pack("<qI", 0x56444220, 224) + b"\x00" * 8)
    assert prepare.vdb_file_version(p) == 224
    (tmp_path / "b.vdb").write_bytes(b"not a vdb at all")
    assert prepare.vdb_file_version(tmp_path / "b.vdb") is None


def test_vdb_file_version_real_file(tmp_path: pathlib.Path) -> None:
    try:
        import pyopenvdb as openvdb  # OpenVDB <= 11 bindings
    except ImportError:
        openvdb = pytest.importorskip("openvdb")
    grid = openvdb.FloatGrid()
    grid.name = "g"
    path = tmp_path / "g.vdb"
    openvdb.write(str(path), grids=[grid])
    version = prepare.vdb_file_version(path)
    assert version is not None and version >= 220
    assert not bs.volume_warnings(
        {"name": "g", "voxel_size": [1.0, 1.0, 1.0]}, version
    ), version


def test_volume_warnings() -> None:
    ok = {"name": "v", "voxel_size": [0.5, 0.5, 0.5]}
    assert bs.volume_warnings(ok, 224) == []
    assert any(
        "non-uniform" in w
        for w in bs.volume_warnings({"name": "v", "voxel_size": [0.5, 0.5, 1.0]})
    )
    assert any("225" in w for w in bs.volume_warnings(ok, 225))
