"""Tests for les_render.particles (streaklines + trails export)."""

from __future__ import annotations

import pathlib
from typing import Any

import numpy as np
import pytest
import xarray as xr
from les_render.fields import FieldSeries
from les_render.particles import (
    SEGMENT_RULE,
    VelocitySampler,
    default_particle_specs,
    detect_periodic,
    export_particles,
)
from les_render.timeline import Timeline, make_timeline

U = 5.0  # m/s, uniform inflow along +x
H = 2.0  # m, cell size
REPO = pathlib.Path(__file__).resolve().parents[3]
REAL_SAMPLE = REPO / "training_data/pyudales_idealized/state/train/sample_0000.nc"


def _dataset(
    block: bool = True, v: float = 0.0, noise: float = 0.0, seed: int = 0
) -> xr.Dataset:
    """24 x 12 x 8 cells of 2 m, 3 snapshots 10 s apart, optional 4 x 4 x 3 solid block."""
    nx, ny, nz, nt = 24, 12, 8, 3
    xt = (np.arange(nx) + 0.5) * H
    yt = (np.arange(ny) + 0.5) * H
    zt = (np.arange(nz) + 0.5) * H
    rng = np.random.default_rng(seed)
    shape = (nt, nz, ny, nx)
    ds = xr.Dataset(
        {
            "u": (
                ("time", "zt", "yt", "xt"),
                (U + noise * rng.standard_normal(shape)).astype(np.float32),
            ),
            "v": (
                ("time", "zt", "yt", "xt"),
                (v + noise * rng.standard_normal(shape)).astype(np.float32),
            ),
            "w": (
                ("time", "zt", "yt", "xt"),
                (noise * rng.standard_normal(shape)).astype(np.float32),
            ),
        },
        coords={"time": np.array([0.0, 10.0, 20.0]), "zt": zt, "yt": yt, "xt": xt},
    )
    blanking = np.zeros((nz, ny, nx), dtype=np.int8)
    if block:
        blanking[0:3, 4:8, 10:14] = 1  # z < 6 m, 8 <= y < 16, 20 <= x < 28
    ds["blanking"] = (("zt", "yt", "xt"), blanking)
    return ds


def _timeline(n_frames: int = 8) -> Timeline:
    return Timeline(
        fps=10.0, playback_speed=10.0, t_start=2.0, n_frames=n_frames
    )  # dt = 1 s


def _load(out: pathlib.Path, layer: dict[str, Any]) -> list[dict[str, np.ndarray]]:
    frames = []
    for i in range(layer["n_files"]):
        with np.load(out / layer["pattern"].format(frame=i)) as d:
            frames.append({k: d[k] for k in ("points", "speed", "alpha")})
    return frames


def _streak_spec(**kw: Any) -> dict[str, Any]:
    spec = default_particle_specs()["streaklines"]
    spec.update(
        name="streaks",
        seeding=["inlet_rake", "ground_line", "building_corners"],
        counts={"rake_y": 5, "rake_z": 3, "ground_line": 6},
        points_per_line=24,
        corner_heights=[0.5],
    )
    spec.update(kw)
    return spec


def _trail_spec(**kw: Any) -> dict[str, Any]:
    spec = default_particle_specs()["trails"]
    spec.update(
        name="trails", counts=400, points_per_line=8, fade_frames=2, max_age=6.0, seed=3
    )
    spec.update(kw)
    return spec


def _check_contract(
    layer: dict[str, Any], frames: list[dict[str, np.ndarray]], fields: FieldSeries
) -> None:
    n, k = layer["n_lines"], layer["points_per_line"]
    for fr in frames:
        assert fr["points"].shape == (n, k, 3) and fr["points"].dtype == np.float32
        assert fr["speed"].shape == (n, k) and fr["speed"].dtype == np.float16
        assert fr["alpha"].shape == (n, k) and fr["alpha"].dtype == np.float16
        a = fr["alpha"].astype(np.float32)
        assert np.all((a >= 0) & (a <= 1)) and np.isfinite(fr["points"]).all()
        vis = a > 0
        pts = fr["points"].astype(np.float64)
        assert not fields.is_solid(pts[vis]).any(), "visible point inside a building"
        # drawn segments (both ends visible) never cross a solid; the exporter
        # probes segments at discrete points, so diagonal grazes of a building
        # corner shallower than ~0.2 m are tolerated (invisible when rendered)
        both = vis[:, :-1] & vis[:, 1:]
        p0, p1 = pts[:, :-1][both], pts[:, 1:][both]
        for s in np.linspace(0.1, 0.9, 9):
            assert not _deep_in_solid(
                fields, p0 + s * (p1 - p0)
            ).any(), "drawn segment crosses a building"


def _deep_in_solid(
    fields: FieldSeries, pts: np.ndarray, margin: float = 0.2
) -> np.ndarray:
    """Solid at ``pts`` and at every +-``margin`` offset along each axis."""
    deep = fields.is_solid(pts)
    for ax in range(3):
        for sign in (-1.0, 1.0):
            off = np.zeros(3)
            off[ax] = sign * margin
            deep &= fields.is_solid(pts + off)
    return deep


# -- sampler ---------------------------------------------------------------------


def test_sampler_matches_fieldseries() -> None:
    fields = FieldSeries(_dataset(noise=1.0))
    sampler = VelocitySampler(fields)
    rng = np.random.default_rng(1)
    g = fields.grid
    pts = rng.uniform(
        g.lower - 3.0, g.upper + 3.0, size=(500, 3)
    )  # includes clamped outside points
    for t in (0.0, 3.7, 10.0, 19.2, 25.0):
        np.testing.assert_allclose(
            sampler(pts, t), fields.sample_velocity(pts, t), atol=1e-5
        )


def test_sampler_periodic_wraps() -> None:
    fields = FieldSeries(_dataset(block=False, noise=1.0))
    sampler = VelocitySampler(fields, periodic=(False, True, False))
    rng = np.random.default_rng(2)
    pts = rng.uniform(fields.grid.lower, fields.grid.upper, size=(200, 3))
    shifted = pts + np.array([0.0, fields.grid.upper[1] - fields.grid.lower[1], 0.0])
    np.testing.assert_allclose(sampler(pts, 4.0), sampler(shifted, 4.0), atol=1e-5)
    # between the last cell centre and the upper face, blends toward cell 0
    g = fields.grid
    p = np.array([[g.x[5], g.upper[1] - 0.25 * H, g.z[3]]])
    v_last, v_first = (
        fields.velocity_snapshot(0)[1, 5, -1, 3],
        fields.velocity_snapshot(0)[1, 5, 0, 3],
    )
    np.testing.assert_allclose(
        sampler(p, 0.0)[0, 1], 0.75 * v_last + 0.25 * v_first, atol=1e-5
    )


def test_detect_periodic() -> None:
    assert detect_periodic(FieldSeries(_dataset(block=False, noise=1.0))) == (
        False,
        False,
        False,
    )
    ds = _dataset(block=False)
    rng = np.random.default_rng(0)
    # smooth, y-periodic fluctuation: cos(2 pi y / Ly) + noise along x
    y = ds["yt"].values
    phase = rng.uniform(0, 2 * np.pi, size=(3, 8, 1, 24))
    fluct = np.cos(2 * np.pi * y[None, None, :, None] / (12 * H) + phase)
    ds["u"] = ds["u"] + fluct.astype(np.float32)
    assert detect_periodic(FieldSeries(ds))[:2] == (False, True)


# -- streaklines ---------------------------------------------------------------------


def test_streaklines_uniform_flow_spacing(tmp_path: pathlib.Path) -> None:
    fields = FieldSeries(_dataset(block=False))
    tau = 0.25
    spec = _streak_spec(
        seeding="custom",
        emitters=[[3.0, 12.0, 5.0], [3.0, 5.0, 9.0]],
        release_interval=tau,
        periodic=[False, False],
    )
    layer = export_particles(fields, _timeline(4), spec, tmp_path)
    frames = _load(tmp_path, layer)
    _check_contract(layer, frames, fields)
    for fr in frames:
        pts, a = fr["points"].astype(np.float64), fr["alpha"] > 0
        assert a[:, 0].all()
        # consecutive releases are U * tau apart (older = further downstream), no y, z motion
        d = np.diff(pts, axis=1)
        both = a[:, 1:] & a[:, :-1]
        np.testing.assert_allclose(d[..., 0][both], U * tau, atol=1e-3)
        np.testing.assert_allclose(d[..., 1:][both], 0.0, atol=1e-4)
        # head released within the last tau, so it sits within U * tau of the emitter
        assert np.all(
            (pts[:, 0, 0] >= 3.0 - 1e-6) & (pts[:, 0, 0] <= 3.0 + U * tau + 1e-6)
        )
        # speed is the flow speed
        np.testing.assert_allclose(fr["speed"][a].astype(np.float32), U, rtol=2e-3)
    # frames are dt apart; a chain spans points_per_line releases
    assert layer["release_interval"] == pytest.approx(tau)


@pytest.mark.parametrize("integrator", ["rk2", "rk4"])  # type: ignore[misc]
def test_streaklines_solids_and_contract(
    tmp_path: pathlib.Path, integrator: str
) -> None:
    fields = FieldSeries(_dataset(block=True, noise=0.3))
    tl = _timeline(6)
    spec = _streak_spec(integrator=integrator, footprints=None)
    layer = export_particles(fields, tl, spec, tmp_path)
    frames = _load(tmp_path, layer)
    _check_contract(layer, frames, fields)
    # the block kills some tracers and the chain behind it is hidden, but most is drawn
    vis = np.mean([(fr["alpha"] > 0).mean() for fr in frames])
    assert 0.2 < vis < 1.0

    # manifest layer contract
    for key in (
        "name",
        "type",
        "pattern",
        "frame_step",
        "n_files",
        "kind",
        "n_lines",
        "points_per_line",
        "radius",
        "emission_strength",
        "variable",
        "range",
        "colormap",
        "lut_linear_rgb",
    ):
        assert key in layer, key
    assert layer["type"] == "particles" and layer["kind"] == "streaklines"
    assert layer["frame_step"] == 1 and layer["n_files"] == tl.n_frames
    assert layer["pattern"] == "particles/streaks/streaks.{frame:04d}.npz"
    assert layer["variable"] == "speed" and layer["colormap"] == "inferno"
    assert np.asarray(layer["lut_linear_rgb"]).shape == (256, 3)
    assert layer["range"][0] < layer["range"][1]
    assert layer["segment_rule"] == SEGMENT_RULE
    # 5 x 3 rake + 6 ground-line + 2 corners x 1 height of the single block
    assert layer["n_lines"] == 15 + 6 + 2


def test_streaklines_broken_segments_hidden_on_both_ends(
    tmp_path: pathlib.Path,
) -> None:
    fields = FieldSeries(_dataset(block=True, noise=0.3))
    spec = _streak_spec(max_segment_length=3.0)
    layer = export_particles(fields, _timeline(4), spec, tmp_path)
    for fr in _load(tmp_path, layer):
        a = fr["alpha"].astype(np.float32)
        d = np.linalg.norm(np.diff(fr["points"].astype(np.float64), axis=1), axis=2)
        long_ = d > 3.0 + 1e-3
        assert np.all(a[:, :-1][long_] == 0) and np.all(a[:, 1:][long_] == 0)
        # a segment touching a hidden point never jumps: zero length or a normal step
        one = (a[:, :-1] > 0) ^ (a[:, 1:] > 0)
        assert np.all(d[one] <= 3.0 + 1e-3)


# -- trails ------------------------------------------------------------------------------


def test_trails_uniform_flow(tmp_path: pathlib.Path) -> None:
    fields = FieldSeries(_dataset(block=False))
    spec = _trail_spec(samples_per_frame=2, periodic=[False, False])
    tl = _timeline(6)
    layer = export_particles(fields, tl, spec, tmp_path)
    frames = _load(tmp_path, layer)
    _check_contract(layer, frames, fields)
    step = U * tl.dt / 2
    for fr in frames:
        a = fr["alpha"].astype(np.float32) > 0
        d = np.diff(fr["points"].astype(np.float64), axis=1)
        both = a[:, :-1] & a[:, 1:]
        assert both.any()
        # history samples are U * dt / samples_per_frame apart (head first => -x), never a jump
        np.testing.assert_allclose(d[..., 0][both], -step, atol=1e-3)
        np.testing.assert_allclose(d[..., 1:][both], 0.0, atol=1e-4)
        # alpha tapers toward the tail
        alpha = fr["alpha"].astype(np.float32)
        full = a.all(axis=1)
        if full.any():
            assert np.all(np.diff(alpha[full], axis=1) <= 1e-3)
    # a tracer moves U * dt between frames: its old head is now 2 samples down its trail
    f2, f3 = frames[2], frames[3]
    same = (f2["alpha"][:, 0] > 0) & (f3["alpha"][:, 2] > 0)
    assert same.sum() > 100
    np.testing.assert_allclose(f3["points"][same, 2], f2["points"][same, 0], atol=1e-4)
    moved = f3["points"][same, 0].astype(np.float64) - f2["points"][same, 0]
    np.testing.assert_allclose(
        moved, np.tile([U * tl.dt, 0.0, 0.0], (same.sum(), 1)), atol=1e-3
    )
    assert layer["n_lines"] == 400 and layer["kind"] == "trails"


def test_trails_solids_recycling_and_fade(tmp_path: pathlib.Path) -> None:
    fields = FieldSeries(_dataset(block=True, noise=0.5))
    spec = _trail_spec(counts={"particles": 600}, max_age=3.0)
    layer = export_particles(fields, _timeline(10), spec, tmp_path)
    frames = _load(tmp_path, layer)
    _check_contract(layer, frames, fields)
    # recycling with fades: head alpha changes gradually, never 0 -> 1 in one frame
    heads = np.stack([fr["alpha"][:, 0].astype(np.float32) for fr in frames])
    assert np.max(np.abs(np.diff(heads, axis=0))) < 0.9


def test_trails_periodic_stay_inside(tmp_path: pathlib.Path) -> None:
    fields = FieldSeries(_dataset(block=False, v=3.0))
    spec = _trail_spec(periodic=[True, True], max_age=100.0)
    layer = export_particles(fields, _timeline(6), spec, tmp_path)
    g = fields.grid
    for fr in _load(tmp_path, layer):
        p = fr["points"].astype(np.float64).reshape(-1, 3)
        assert np.all((p >= g.lower - 1e-4) & (p <= g.upper + 1e-4))
        assert (fr["alpha"] > 0).mean() > 0.5  # wrap segments hidden, the rest drawn
    assert layer["periodic"] == [True, True]


# -- determinism ---------------------------------------------------------------------------


@pytest.mark.parametrize("make_spec", [_streak_spec, _trail_spec])  # type: ignore[misc]
def test_deterministic(tmp_path: pathlib.Path, make_spec: Any) -> None:
    fields = FieldSeries(_dataset(block=True, noise=0.3))
    a = _load(
        tmp_path / "a",
        export_particles(fields, _timeline(3), make_spec(), tmp_path / "a"),
    )
    b = _load(
        tmp_path / "b",
        export_particles(fields, _timeline(3), make_spec(), tmp_path / "b"),
    )
    for fa, fb in zip(a, b):
        for key in fa:
            np.testing.assert_array_equal(fa[key], fb[key])
    if make_spec is _trail_spec:
        c = _load(
            tmp_path / "c",
            export_particles(fields, _timeline(3), make_spec(seed=99), tmp_path / "c"),
        )
        assert not np.array_equal(a[0]["points"], c[0]["points"])


def test_default_specs() -> None:
    specs = default_particle_specs()
    assert set(specs) == {"streaklines", "trails"}
    for kind, spec in specs.items():
        assert spec["kind"] == kind and spec["type"] == "particles"
        assert spec["colormap"] == "inferno" and spec["color_variable"] == "speed"
    specs["streaklines"]["counts"]["rake_y"] = -1  # returns fresh copies
    assert default_particle_specs()["streaklines"]["counts"]["rake_y"] > 0


# -- real data smoke ------------------------------------------------------------------------


@pytest.mark.skipif(not REAL_SAMPLE.is_file(), reason="training data not available")  # type: ignore[misc]
def test_real_sample_smoke(tmp_path: pathlib.Path) -> None:
    from les_render.case import discover_case

    case = discover_case(REAL_SAMPLE)
    fields = FieldSeries(case.open_state())
    tl = make_timeline(
        fields.times, fps=30, playback_speed=20, t_start=300.0, duration=3 / 30 - 1e-6
    )
    streak = _streak_spec(
        counts={"rake_y": 6, "rake_z": 4, "ground_line": 8},
        preroll=20.0,
        line_duration=20.0,
    )
    trail = _trail_spec(counts=2000, preroll=10.0)
    for spec in (streak, trail):
        layer = export_particles(fields, tl, spec, tmp_path)
        frames = _load(tmp_path, layer)
        assert len(frames) == tl.n_frames == 3
        _check_contract(layer, frames, fields)
        assert (frames[-1]["alpha"] > 0).mean() > 0.3


def test_rk2_close_to_rk4() -> None:
    from les_render.particles import _Tracer

    fields = FieldSeries(_dataset(block=False, noise=1.0))
    rng = np.random.default_rng(5)
    p0 = rng.uniform(fields.grid.lower + 8.0, fields.grid.upper - 8.0, size=(300, 3))
    out = {}
    for integ, cfl in (("rk2", 0.5), ("rk4", 0.05)):
        tr = _Tracer(
            fields,
            {
                **_trail_spec(),
                "integrator": integ,
                "cfl": cfl,
                "periodic": [True, True],
            },
        )
        p, t = p0.copy(), 1.0
        for _ in range(4):
            n = tr.n_sub(t, t + 0.5)
            for s in range(n):
                p, _ = tr.step(p, t + s * 0.5 / n, 0.5 / n)
            t += 0.5
        out[integ] = p
    err = np.linalg.norm(out["rk2"] - out["rk4"], axis=1)
    assert np.median(err) < 0.02 * H and err.max() < 0.25 * H
