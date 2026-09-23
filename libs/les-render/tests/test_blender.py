"""Blender stage: pure-python helpers + an end-to-end smoke render / Alembic export.

The helpers in ``les_render/blender/bundle.py`` run inside Blender's Python
but only need numpy, so they are tested directly here. The smoke tests build
a tiny contract-conformant bundle (``blender_fixture.py``), then run the real
``blender -b`` stage through ``blender_runner``; they are skipped when Blender
is not installed.
"""

from __future__ import annotations

import importlib.util
import json
import pathlib
import shutil
import sys
import types

import numpy as np
import pytest

HERE = pathlib.Path(__file__).resolve().parent
BLENDER_PKG = HERE.parent / "src" / "les_render" / "blender"
STATE = HERE.parents[2] / "training_data/pyudales_idealized/state/train/sample_0000.nc"


def _load(name: str, path: pathlib.Path) -> types.ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


bmod = _load("les_blender_bundle", BLENDER_PKG / "bundle.py")


# -- pure helpers ------------------------------------------------------------------


def test_file_index_frame_step() -> None:
    layer = {"frame_step": 3, "n_files": 4}
    assert [bmod.Bundle.file_index(layer, f) for f in range(14)] == [
        0,
        0,
        0,
        1,
        1,
        1,
        2,
        2,
        2,
        3,
        3,
        3,
        3,
        3,
    ]


def test_visible_runs_follow_segment_rule() -> None:
    alpha = np.array(
        [
            [1.0, 0.5, 0.0, 0.3, 0.2, 0.1],  # runs [0,1] and [3,4,5]
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # nothing
            [0.0, 0.4, 0.0, 0.2, 0.0, 0.0],  # isolated points -> no segment
            [0.1, 0.1, 0.1, 0.1, 0.1, 0.1],  # whole line
        ],
        dtype=np.float32,
    )
    sizes, idx = bmod.visible_runs(alpha)
    assert sizes.tolist() == [2, 3, 6]
    assert idx.tolist() == [0, 1, 3, 4, 5] + list(range(18, 24))
    # every segment drawn has both ends visible, every visible segment is drawn
    flat = alpha.reshape(-1)
    drawn = set()
    o = 0
    for s in sizes:
        run = idx[o : o + s]
        drawn |= {(int(a), int(b)) for a, b in zip(run[:-1], run[1:])}
        o += s
    expect = {tuple(e) for e in bmod.visible_segments(alpha).tolist()}
    assert drawn == expect
    assert all(flat[a] > 0 and flat[b] > 0 for a, b in drawn)


def test_collapse_hidden_zero_length_gaps() -> None:
    rng = np.random.default_rng(1)
    pts = rng.normal(size=(3, 7, 3)).astype(np.float32)
    alpha = np.array(
        [[0, 0, 1, 1, 0, 0, 1], [1, 1, 1, 1, 1, 1, 1], [0, 0, 0, 0, 0, 0, 0]],
        dtype=np.float32,
    )
    out = bmod.collapse_hidden(pts, alpha)
    np.testing.assert_array_equal(out[1], pts[1])  # untouched when all visible
    np.testing.assert_array_equal(
        out[0, 0], pts[0, 2]
    )  # leading hidden -> next visible
    np.testing.assert_array_equal(out[0, 4], pts[0, 3])  # gap: nearest side
    np.testing.assert_array_equal(out[0, 5], pts[0, 6])
    assert np.all(out[2] == pts[2, 0])  # fully hidden -> head
    # hidden->hidden spans inside a gap connect the two visible ends; each
    # hidden point itself sits exactly on a visible point
    vis0 = {tuple(p) for p in pts[0][alpha[0] > 0]}
    assert all(tuple(p) in vis0 for p in out[0])


def test_smooth_runs_keeps_samples_and_does_not_overshoot() -> None:
    # one run with very uneven spacing (the case where uniform Catmull-Rom loops)
    pts = np.array(
        [[0, 0, 0], [0.2, 0, 0], [10, 1, 0], [10.3, 1.2, 0], [20, 0, 0]], np.float32
    )
    pts2 = np.concatenate([pts, pts + [0, 5, 0]])
    spd = np.arange(10, dtype=np.float32)
    out, (s2,), sizes = bmod.smooth_runs(pts2, [spd], np.array([5, 5]), 4)
    assert sizes.tolist() == [17, 17]
    np.testing.assert_allclose(out[::4][:5], pts)  # original samples are kept
    np.testing.assert_allclose(out[17::4], pts + [0, 5, 0])
    first = out[:17]
    assert (
        first[:, 0].min() >= -1e-6 and first[:, 0].max() <= 20 + 1e-6
    )  # no loops outside the hull
    assert np.all(np.diff(s2[:17]) >= 0)  # attributes interpolate monotonically


def test_chunk_runs_splits_into_short_connected_curves() -> None:
    sizes = np.array([20, 8, 2, 9])
    pts = np.arange(int(sizes.sum()) * 3, dtype=np.float32).reshape(-1, 3)
    val = np.arange(int(sizes.sum()), dtype=np.float32)
    out, (v,), cs = bmod.chunk_runs(pts, [val], sizes, 8)
    assert cs.max() <= 8 and cs.min() >= 2
    assert cs.tolist() == [8, 8, 6, 8, 2, 8, 2]
    # every original segment appears exactly once, chunks share joint points
    o, segs = 0, []
    for n in cs:
        segs += [(v[o + j], v[o + j + 1]) for j in range(n - 1)]
        o += n
    starts = np.concatenate([[0], np.cumsum(sizes)[:-1]])
    expect = [
        (float(s + j), float(s + j + 1))
        for s, n in zip(starts, sizes)
        for j in range(n - 1)
    ]
    assert [(float(a), float(b)) for a, b in segs] == expect
    np.testing.assert_array_equal(out[:, 0], v * 3)


def test_sample_camera_matches_cameras_module() -> None:
    cameras = pytest.importorskip("les_render.cameras")
    shots = [
        {
            "name": "a",
            "start": 0,
            "end": 49,
            "keys": [
                {
                    "frame": 0,
                    "location": [0, -100, 50],
                    "target": [0, 0, 0],
                    "focal_length_mm": 24,
                    "fstop": 8,
                },
                {
                    "frame": 20,
                    "location": [30, -90, 40],
                    "target": [10, 0, 5],
                    "focal_length_mm": 30,
                    "fstop": 4,
                },
                {
                    "frame": 35,
                    "location": [60, -60, 30],
                    "target": [20, 5, 5],
                    "focal_length_mm": 35,
                    "fstop": 4,
                },
                {
                    "frame": 49,
                    "location": [80, -20, 25],
                    "target": [25, 5, 5],
                    "focal_length_mm": 50,
                    "fstop": 2.8,
                },
            ],
        },
        {
            "name": "b",
            "start": 50,
            "end": 80,
            "keys": [
                {
                    "frame": 50,
                    "location": [0, 0, 200],
                    "target": [0, 1, 0],
                    "focal_length_mm": 35,
                    "fstop": 11,
                },
            ],
        },
    ]
    for f in range(-3, 85):
        ours = bmod.sample_camera(shots, f)
        loc, tgt, focal = cameras.sample_camera(shots, f)
        np.testing.assert_allclose(ours["location"], loc, atol=1e-4)
        np.testing.assert_allclose(ours["target"], tgt, atol=1e-4)
        assert ours["focal_length_mm"] == pytest.approx(focal)


def test_camera_quaternion_aims_at_target() -> None:
    loc, tgt = np.array([10.0, -50.0, 30.0]), np.array([40.0, 20.0, 5.0])
    w, x, y, z = bmod.camera_quaternion(loc, tgt)
    # rotate camera -Z and +Y into world
    q = np.array([w, x, y, z])

    def rot(v: np.ndarray) -> np.ndarray:
        u = q[1:]
        return v + 2 * np.cross(u, np.cross(u, v) + q[0] * v)

    fwd = rot(np.array([0.0, 0.0, -1.0]))
    d = (tgt - loc) / np.linalg.norm(tgt - loc)
    np.testing.assert_allclose(fwd, d, atol=1e-9)
    right = rot(np.array([1.0, 0.0, 0.0]))
    assert abs(right[2]) < 1e-9  # no roll: camera X stays horizontal
    assert rot(np.array([0.0, 1.0, 0.0]))[2] > 0  # camera up points skyward


def test_read_ply_roundtrip(tmp_path: pathlib.Path) -> None:
    trimesh = pytest.importorskip("trimesh")
    mesh = trimesh.creation.icosphere(subdivisions=2)
    colors = np.tile(np.array([[200, 30, 10, 255]], np.uint8), (len(mesh.vertices), 1))
    mesh.visual.vertex_colors = colors
    path = tmp_path / "m.ply"
    mesh.export(path)
    verts, tris, props = bmod.read_ply(path)
    np.testing.assert_allclose(verts, mesh.vertices, atol=1e-5)
    np.testing.assert_array_equal(tris, mesh.faces)
    assert props["red"][0] == 200 and props["blue"][0] == 10
    # ascii too
    path2 = tmp_path / "a.ply"
    path2.write_bytes(trimesh.exchange.ply.export_ply(mesh, encoding="ascii"))
    v2, t2, _ = bmod.read_ply(path2)
    np.testing.assert_allclose(v2, mesh.vertices, atol=1e-5)
    np.testing.assert_array_equal(t2, mesh.faces)


# -- end to end (needs Blender) ------------------------------------------------------

needs_blender = pytest.mark.skipif(
    shutil.which("blender") is None and not __import__("os").environ.get("BLENDER"),
    reason="Blender not installed",
)


@pytest.fixture(scope="module")  # type: ignore[misc]
def tiny_bundle(tmp_path_factory: pytest.TempPathFactory) -> pathlib.Path:
    if not STATE.is_file():
        pytest.skip(f"test data missing: {STATE}")
    pytest.importorskip("skimage")
    sys.path.insert(0, str(HERE))
    try:
        import blender_fixture
    finally:
        sys.path.remove(str(HERE))
    out: pathlib.Path = tmp_path_factory.mktemp("bundle")
    blender_fixture.make_fixture_bundle(
        out,
        STATE,
        n_frames=4,
        n_lines=400,
        points_per_line=8,
        width=192,
        height=108,
        heavy_step=2,
    )
    return out


@needs_blender  # type: ignore[misc]
def test_blender_smoke_render(tiny_bundle: pathlib.Path) -> None:
    from les_render.blender_runner import run_blender

    frames = run_blender(
        tiny_bundle, render=True, frames="0,3", samples=4, width=192, height=108
    )
    pngs = sorted(frames.glob("*.png"))
    assert [p.name for p in pngs] == ["0000.png", "0003.png"]
    import matplotlib.image

    img = matplotlib.image.imread(pngs[0])
    assert img.shape[:2] == (108, 192)
    assert float(img[..., :3].max()) > 0.2  # not a black frame
    timings = json.loads((frames.parent / "timings_eevee.json").read_text())
    assert [t["frame"] for t in timings["frames"]] == [0, 3]


@needs_blender  # type: ignore[misc]
def test_blender_alembic_export(tiny_bundle: pathlib.Path) -> None:
    from les_render.blender_runner import run_blender

    abc_dir = run_blender(tiny_bundle, render=False, export_alembic=True)
    report = json.loads((abc_dir / "alembic_report.json").read_text())
    manifest = json.loads((tiny_bundle / "manifest.json").read_text())
    for layer in manifest["layers"]:
        if layer["type"] in ("particles", "isosurface"):
            assert (abc_dir / f"{layer['name']}.abc").stat().st_size > 0
    assert (abc_dir / "buildings.abc").is_file()
    # particles re-import as constant-topology animated curves
    part = next(x for x in manifest["layers"] if x["type"] == "particles")
    checks = next(iter(report[f"{part['name']}_verify"].values()))
    assert {c["curves"] for c in checks} == {part["n_lines"]}
    assert {c["points"] for c in checks} == {part["n_lines"] * part["points_per_line"]}
    assert len({c["pos_hash"] for c in checks}) == len(
        checks
    )  # animated, not a static first sample
    iso = next(x for x in manifest["layers"] if x["type"] == "isosurface")
    ichecks = next(iter(report[f"{iso['name']}_verify"].values()))
    assert all(c["verts"] > 0 for c in ichecks)
