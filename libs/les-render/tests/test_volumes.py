"""Tests for the volumes / isosurfaces / slices exporters.

Uses a small synthetic in-memory dataset (no on-disk source -> the
exporters' process-pool path never engages; ``workers=1`` is used
throughout so everything runs sequentially in-process, keeping the whole
module well under the 30 s budget).
"""

from __future__ import annotations

import pathlib

import numpy as np
import pytest
import xarray as xr
from PIL import Image

try:
    import pyopenvdb as vdb
except ImportError:  # pragma: no cover - depends on the openvdb build in the env
    import openvdb as vdb

from les_render.fields import FieldSeries
from les_render.isosurfaces import (
    _limit_faces,
    default_isosurface_specs,
    export_isosurfaces,
)
from les_render.slices import default_slice_specs, export_slices
from les_render.timeline import Timeline, make_timeline
from les_render.volumes import default_volume_specs, export_volumes

_COMMON_LAYER_KEYS = ("name", "type", "pattern", "frame_step", "n_files")
_COLOR_KEYS = ("variable", "range", "colormap", "lut_linear_rgb")


@pytest.fixture  # type: ignore[misc]
def synthetic_dataset() -> xr.Dataset:
    nx, ny, nz, nt = 8, 8, 4, 3
    xt = np.arange(nx) * 2.0  # 0..14
    yt = np.arange(ny) * 2.0  # 0..14
    zt = np.arange(nz) * 2.0 + 1.0  # 1,3,5,7
    time = np.arange(nt) * 2.0  # 0, 2, 4

    # v depends only on y (positive, monotonic) so speed = |v| is brighter
    # for larger y -- used by the slice-orientation test. u is a constant
    # "inflow", w is zero.
    v3 = np.broadcast_to(yt[None, :, None], (nz, ny, nx)).astype(np.float32)
    u = np.full((nt, nz, ny, nx), 7.0, dtype=np.float32)
    v = np.broadcast_to(v3, (nt, nz, ny, nx)).astype(np.float32)
    w = np.zeros((nt, nz, ny, nx), dtype=np.float32)

    # A single-z-layer "building" footprint at the lowest z (zt[0]) spanning
    # x indices [3, 5) and all y -- used by the alpha-in-solids test. Upper
    # z layers are solid-free.
    blanking = np.zeros((nz, ny, nx), dtype=np.int8)
    blanking[0, :, 3:5] = 1

    ds = xr.Dataset(
        {
            "u": (("time", "zt", "yt", "xt"), u),
            "v": (("time", "zt", "yt", "xt"), v),
            "w": (("time", "zt", "yt", "xt"), w),
            "blanking": (("zt", "yt", "xt"), blanking),
        },
        coords={"time": time, "zt": zt, "yt": yt, "xt": xt},
    )
    return ds


@pytest.fixture  # type: ignore[misc]
def fields(synthetic_dataset: xr.Dataset) -> FieldSeries:
    return FieldSeries(synthetic_dataset)


@pytest.fixture  # type: ignore[misc]
def fields_no_solid(synthetic_dataset: xr.Dataset) -> FieldSeries:
    """Same grid/velocity as `fields`, but with an all-fluid blanking mask --
    for isosurface-shape tests where a solid block would locally cut the
    engineered analytic sphere and make an exact geometric check fail for a
    reason unrelated to the code under test."""
    ds = synthetic_dataset.copy()
    ds["blanking"] = xr.zeros_like(ds["blanking"])
    return FieldSeries(ds)


@pytest.fixture  # type: ignore[misc]
def timeline(fields: FieldSeries) -> Timeline:
    # dt=1s, 5 video frames covering sim t=0..4 (the full stored range).
    return make_timeline(fields.times, fps=2.0, playback_speed=2.0, duration=2.0)


# -- volumes -------------------------------------------------------------


def test_volume_manifest_contract_keys(
    fields: FieldSeries, timeline: Timeline, tmp_path: pathlib.Path
) -> None:
    for spec in default_volume_specs():
        spec = dict(spec, workers=1, frame_step=2, upsample=1)
        layer = export_volumes(fields, timeline, spec, tmp_path)
        for key in (
            _COMMON_LAYER_KEYS
            + _COLOR_KEYS
            + (
                "grid",
                "voxel_size",
                "origin",
                "shape",
                "density_range",
                "emission_strength",
                "density_scale",
            )
        ):
            assert (
                key in layer
            ), f"missing manifest key {key!r} for volume spec {spec['name']!r}"
        assert layer["type"] == "volume"
        assert len(layer["voxel_size"]) == 3
        assert len(layer["origin"]) == 3


def test_volume_transform_roundtrip_and_sparsity(
    fields: FieldSeries, timeline: Timeline, tmp_path: pathlib.Path
) -> None:
    spec = {"name": "speed_glow", "workers": 1, "frame_step": 2, "upsample": 1}
    layer = export_volumes(fields, timeline, spec, tmp_path)

    path = tmp_path / "volumes" / "speed_glow" / "speed_glow.0000.vdb"
    assert path.exists()
    grid = vdb.read(str(path), "speed_glow")

    total_voxels = int(np.prod(layer["shape"]))
    active = grid.activeVoxelCount()
    assert (
        0 < active < total_voxels
    ), "expected the density floor to inactivate some voxels (sparsity)"

    ox, oy, oz = layer["origin"]
    wx0, wy0, wz0 = grid.transform.indexToWorld((0, 0, 0))
    assert np.allclose(
        (wx0, wy0, wz0), (ox, oy, oz)
    ), "index (0,0,0) must map to manifest origin"

    vx, vy, vz = layer["voxel_size"]
    wx1, wy1, wz1 = grid.transform.indexToWorld((1, 1, 1))
    assert np.allclose((wx1 - wx0, wy1 - wy0, wz1 - wz0), (vx, vy, vz))


def test_volume_default_density_grid_omitted_by_default(
    fields: FieldSeries, timeline: Timeline, tmp_path: pathlib.Path
) -> None:
    """Unreal's SVT importer is picky about extra grids in a file; the
    normalized-density second grid must be opt-in, not default."""
    spec = {"name": "speed_glow", "workers": 1, "frame_step": 2, "upsample": 1}
    export_volumes(fields, timeline, spec, tmp_path)
    path = tmp_path / "volumes" / "speed_glow" / "speed_glow.0000.vdb"
    with pytest.raises(Exception):
        vdb.read(str(path), "density")


def test_volume_all_null_spec_uses_defaults(
    fields: FieldSeries, timeline: Timeline, tmp_path: pathlib.Path
) -> None:
    """A preset/render.yaml key set to ``null`` (Hydra ``~``) must fall back
    to the default, not reach a caster (``float(None)`` etc.) as a literal
    None -- every documented volume spec key, all null at once."""
    spec = {
        "name": None,
        "variable": None,
        "transform": None,
        "gamma": None,
        "reference": None,
        "vscale": None,
        "upsample": None,
        "density_range": None,
        "density_floor_percentile": None,
        "density_ceiling_percentile": None,
        "colormap": None,
        "color_range": None,
        "emission_strength": None,
        "density_scale": None,
        "write_normalized_density": None,
        "half": None,
        "frame_step": None,
        "workers": None,
    }
    layer = export_volumes(fields, timeline, spec, tmp_path)
    assert layer["type"] == "volume"
    assert layer["name"] == "speed_glow"
    assert layer["n_files"] > 0
    for f in range(layer["n_files"]):
        assert (tmp_path / layer["pattern"].format(frame=f)).is_file()


# -- isosurfaces -----------------------------------------------------------


def _patch_sphere_field(
    monkeypatch: pytest.MonkeyPatch, radius_scale: float = 3.0
) -> None:
    """Replace isosurfaces.scalar_field with a synthetic field: a smooth,
    radially-symmetric bump (positive core, negative tail -> a clean
    zero-level-crossing sphere) for the iso variable, and plain radius for
    everything else (used as the colour variable)."""
    import les_render.isosurfaces as iso_mod
    from les_render.fields import scalar_field as real_scalar_field
    from les_render.fields import upsample_mask

    def fake(
        fields: FieldSeries, name: str, t: float, upsample: int
    ) -> tuple[np.ndarray, object]:
        _, grid = real_scalar_field(fields, "speed", t, upsample)
        X, Y, Z = np.meshgrid(grid.x, grid.y, grid.z, indexing="ij")
        center = grid.origin + grid.spacing * (np.array(grid.shape) - 1) / 2.0
        r2 = (X - center[0]) ** 2 + (Y - center[1]) ** 2 + (Z - center[2]) ** 2
        sigma = radius_scale
        if name == "q_criterion":
            amp = 10.0
            vals = amp * np.exp(-r2 / (2 * sigma**2)) - 0.05 * amp
        else:
            vals = np.sqrt(r2)
        solid = upsample_mask(fields.solid, upsample) if upsample > 1 else fields.solid
        vals = vals.astype(np.float32)
        vals[solid] = 0.0
        return vals, grid

    monkeypatch.setattr(iso_mod, "scalar_field", fake)


def test_isosurface_manifest_contract_keys(
    fields: FieldSeries,
    timeline: Timeline,
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_sphere_field(monkeypatch)
    for spec in default_isosurface_specs():
        spec = dict(spec, workers=1, frame_step=2, upsample=1)
        layer = export_isosurfaces(fields, timeline, spec, tmp_path)
        for key in _COMMON_LAYER_KEYS + _COLOR_KEYS + ("iso_variable", "level"):
            assert key in layer
        assert layer["type"] == "isosurface"


def _read_ply(path: pathlib.Path) -> tuple[np.ndarray, np.ndarray]:
    with open(path, "rb") as fh:
        data = fh.read()
    header_end = data.index(b"end_header\n") + len(b"end_header\n")
    header = data[:header_end].decode("ascii")
    n_vertex = int(
        [ln for ln in header.splitlines() if ln.startswith("element vertex")][
            0
        ].split()[-1]
    )
    n_face = int(
        [ln for ln in header.splitlines() if ln.startswith("element face")][0].split()[
            -1
        ]
    )

    vdt = np.dtype(
        [
            ("x", "<f4"),
            ("y", "<f4"),
            ("z", "<f4"),
            ("red", "u1"),
            ("green", "u1"),
            ("blue", "u1"),
            ("value", "<f4"),
        ]
    )
    offset = header_end
    verts = np.frombuffer(data, dtype=vdt, count=n_vertex, offset=offset)
    offset += n_vertex * vdt.itemsize

    fdt = np.dtype([("n", "u1"), ("idx", "<i4", (3,))])
    faces = np.frombuffer(data, dtype=fdt, count=n_face, offset=offset)
    return verts, faces


def test_isosurface_known_field_sphere_and_ply_readable(
    fields_no_solid: FieldSeries,
    timeline: Timeline,
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fields = fields_no_solid
    _patch_sphere_field(monkeypatch)
    spec = {
        "name": "q_criterion",
        "workers": 1,
        "frame_step": 2,
        "upsample": 1,
        "color_variable": "speed",
        "min_component_faces": 0,
    }
    layer = export_isosurfaces(fields, timeline, spec, tmp_path)
    path = tmp_path / "isosurfaces" / "q_criterion" / "q_criterion.0000.ply"
    assert path.exists()

    verts, faces = _read_ply(path)
    assert (
        verts.size > 0
    ), "expected a non-empty isosurface for the engineered sphere field"
    assert faces.size > 0
    assert np.all(faces["n"] == 3)
    assert faces["idx"].max() < verts.size

    # every vertex must sit near the analytic isosurface radius of
    # amp*exp(-r^2/2sigma^2) - 0.05*amp == level
    amp, sigma = 10.0, 3.0
    level = layer["level"]
    inner = level / amp + 0.05
    assert 0 < inner < 1, "test isn't exercising a real crossing; check the fixture"
    r_expected = sigma * np.sqrt(-2 * np.log(inner))
    grid_center = (
        fields.grid.origin
        + fields.grid.spacing * (np.array(fields.grid.shape) - 1) / 2.0
    )
    r = np.sqrt(
        (verts["x"] - grid_center[0]) ** 2
        + (verts["y"] - grid_center[1]) ** 2
        + (verts["z"] - grid_center[2]) ** 2
    )
    assert np.allclose(r, r_expected, atol=max(fields.grid.spacing) * 1.5)

    # colour: value property should equal the radius (our fake colour
    # variable, sampled trilinearly on a coarse 2 m grid -- some
    # discretisation error vs. the exact analytic radius is expected), and
    # red/green/blue should be a valid LUT sample of it.
    assert np.allclose(verts["value"], r, atol=0.15)
    assert verts["red"].dtype == np.uint8 or verts["red"].dtype.kind == "u"


def test_isosurface_all_null_spec_uses_defaults(
    fields: FieldSeries, timeline: Timeline, tmp_path: pathlib.Path
) -> None:
    """Every documented isosurface spec key set to ``null`` must fall back to
    its default rather than reach a caster as a literal None."""
    spec = {
        "name": None,
        "iso_variable": None,
        "level": None,
        "level_percentile": None,
        "level_fraction": None,
        "upsample": None,
        "smooth_sigma": None,
        "color_variable": None,
        "colormap": None,
        "color_range": None,
        "max_faces": None,
        "min_component_faces": None,
        "frame_step": None,
        "workers": None,
    }
    layer = export_isosurfaces(fields, timeline, spec, tmp_path)
    assert layer["type"] == "isosurface"
    assert layer["name"] == "q_criterion"
    assert layer["n_files"] > 0
    for f in range(layer["n_files"]):
        assert (tmp_path / layer["pattern"].format(frame=f)).is_file()


def _strip_mesh(n_quads: int, x_offset: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    """A single connected component of ``2 * n_quads`` triangles: two rows of
    ``n_quads + 1`` vertices at y=0/y=1, x=``x_offset``..``x_offset+n_quads``,
    triangulated into a ladder strip (each pair of adjacent triangles shares
    an edge, so ``trimesh.graph.connected_components`` reports it as one
    component)."""
    n = n_quads + 1
    xs = np.arange(n, dtype=np.float64) + x_offset
    row0 = np.stack([xs, np.zeros(n), np.zeros(n)], axis=1)
    row1 = np.stack([xs, np.ones(n), np.zeros(n)], axis=1)
    verts = np.concatenate([row0, row1])
    faces = []
    for i in range(n_quads):
        faces.append([i, i + 1, n + i])
        faces.append([i + 1, n + i + 1, n + i])
    return verts, np.asarray(faces, dtype=np.int64)


def _disjoint_components(*n_quads: int) -> tuple[np.ndarray, np.ndarray]:
    """Concatenate several :func:`_strip_mesh` strips, spaced far enough
    apart in x that they share no vertices/edges (so each stays its own
    connected component)."""
    verts_parts, face_parts = [], []
    offset = 0.0
    voffset = 0
    for n in n_quads:
        v, f = _strip_mesh(n, x_offset=offset)
        face_parts.append(f + voffset)
        verts_parts.append(v)
        voffset += v.shape[0]
        offset += n + 100.0  # far more than 1 unit apart -> no shared vertices
    return np.concatenate(verts_parts), np.concatenate(face_parts)


def test_limit_faces_never_exceeds_max_faces_across_components() -> None:
    """Regression for the off-by-one that let the cap be exceeded: two
    60-face components with max_faces=100 used to return 120 faces (both
    kept, since the check ran *before* adding each component) instead of
    honouring the cap."""
    verts, faces = _disjoint_components(30, 30)  # 60 + 60 faces
    _, kept_faces, _ = _limit_faces(verts, faces, min_component_faces=0, max_faces=100)
    assert kept_faces.shape[0] <= 100
    assert kept_faces.shape[0] == 60, "expected exactly the first component kept"


def test_limit_faces_keeps_oversized_single_component_whole() -> None:
    """The documented exception: a single component already over max_faces is
    kept whole (this pass only drops whole components, never splits one)."""
    verts, faces = _strip_mesh(75)  # 150 faces, one component
    _, kept_faces, _ = _limit_faces(verts, faces, min_component_faces=0, max_faces=100)
    assert kept_faces.shape[0] == faces.shape[0] == 150


# -- slices ----------------------------------------------------------------


def test_slice_manifest_contract_keys(
    fields: FieldSeries, timeline: Timeline, tmp_path: pathlib.Path
) -> None:
    for spec in default_slice_specs():
        spec = dict(spec, workers=1, frame_step=2, upsample=1, resolution=[32, 32])
        layer = export_slices(fields, timeline, spec, tmp_path)
        for key in (
            _COMMON_LAYER_KEYS
            + _COLOR_KEYS
            + ("axis", "position", "extent", "resolution")
        ):
            assert key in layer
        assert layer["type"] == "slice"


def test_slice_orientation_row0_is_max_v(
    fields: FieldSeries, timeline: Timeline, tmp_path: pathlib.Path
) -> None:
    # speed = |v| increases with y; row 0 must be the max-y (brightest,
    # under the monotonic "inferno" colormap) edge.
    spec = {
        "name": "speed_slice",
        "axis": "z",
        "position": 1.0,
        "variable": "speed",
        "workers": 1,
        "frame_step": 2,
        "upsample": 1,
        "resolution": [16, 32],
        "lic": False,
    }
    layer = export_slices(fields, timeline, spec, tmp_path)
    path = tmp_path / "slices" / "speed_slice" / "speed_slice.0000.png"
    img = np.asarray(Image.open(path).convert("RGBA"))
    assert img.shape == (32, 16, 4)

    row0_luma = img[0, :, :3].astype(np.float64).mean()
    row_last_luma = img[-1, :, :3].astype(np.float64).mean()
    assert (
        row0_luma > row_last_luma
    ), "row 0 (max-v edge) should be brighter for a y-increasing field"

    # sanity: extent's v1 really is the domain's max y
    assert layer["extent"][1][1] == pytest.approx(float(fields.grid.upper[1]))


def test_slice_alpha_zero_in_solids(
    fields: FieldSeries, timeline: Timeline, tmp_path: pathlib.Path
) -> None:
    # position=1.0 (zt[0]) intersects the synthetic building footprint at
    # x in [6, 8) (xt indices 3,4 -> 6.0, 8.0); position=7.0 (zt[-1]) is
    # solid-free everywhere.
    common = dict(
        axis="z",
        variable="speed",
        workers=1,
        frame_step=2,
        upsample=1,
        resolution=[16, 16],
        lic=False,
    )
    solid_layer = export_slices(
        fields, timeline, dict(common, name="z_low", position=1.0), tmp_path
    )
    clear_layer = export_slices(
        fields, timeline, dict(common, name="z_high", position=7.0), tmp_path
    )

    img_solid = np.asarray(
        Image.open(tmp_path / "slices" / "z_low" / "z_low.0000.png").convert("RGBA")
    )
    img_clear = np.asarray(
        Image.open(tmp_path / "slices" / "z_high" / "z_high.0000.png").convert("RGBA")
    )

    assert (
        img_solid[..., 3] == 0
    ).any(), "expected some fully-transparent pixels over the building"
    assert (img_solid[..., 3] == 255).any(), "expected some opaque (fluid) pixels too"
    assert (
        img_clear[..., 3] == 255
    ).all(), "top slice has no solids, should be fully opaque"
    del solid_layer, clear_layer


def test_slice_resolution_capped_and_lic_runs(
    fields: FieldSeries, timeline: Timeline, tmp_path: pathlib.Path
) -> None:
    spec = {
        "name": "capped",
        "axis": "z",
        "position": 1.0,
        "variable": "speed",
        "workers": 1,
        "frame_step": 2,
        "upsample": 1,
        "px_per_metre": 200.0,
        "max_resolution": 64,
        "lic": True,
        "lic_length": 4,
    }
    layer = export_slices(fields, timeline, spec, tmp_path)
    assert max(layer["resolution"]) <= 64
    path = tmp_path / "slices" / "capped" / "capped.0000.png"
    img = np.asarray(Image.open(path).convert("RGBA"))
    assert img.shape[:2] == (layer["resolution"][1], layer["resolution"][0])
    assert img.dtype == np.uint8


def test_slice_all_null_spec_uses_defaults(
    fields: FieldSeries, timeline: Timeline, tmp_path: pathlib.Path
) -> None:
    """Every documented slice spec key set to ``null`` must fall back to its
    default rather than reach a caster as a literal None."""
    spec = {
        "name": None,
        "axis": None,
        "position": None,
        "variable": None,
        "colormap": None,
        "range": None,
        "upsample": None,
        "extent": None,
        "px_per_metre": None,
        "max_resolution": None,
        "resolution": None,
        "lic": None,
        "lic_length": None,
        "lic_kernel": None,
        "lic_noise_seed": None,
        "lic_strength": None,
        "animate_noise": None,
        "animate_speed": None,
        "frame_step": None,
        "workers": None,
    }
    layer = export_slices(fields, timeline, spec, tmp_path)
    assert layer["type"] == "slice"
    assert layer["name"] == "pedestrian_speed"
    assert layer["axis"] == "z"
    assert layer["n_files"] > 0
    for f in range(layer["n_files"]):
        assert (tmp_path / layer["pattern"].format(frame=f)).is_file()
