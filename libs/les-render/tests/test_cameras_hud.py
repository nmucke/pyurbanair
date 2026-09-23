"""Fast tests for cameras.py, hud.py and compose.py.

No solver run needed: geometry/domain/timeline/manifest are hand-built
("smoke shape" fixtures below), matching the ``training_data/pyudales_idealized``
layout in scale (a handful of ~30 m buildings on a ~250x120 m footprint) so
the shot heuristics (margins, orbit radii, canyon search) exercise realistic
code paths without touching disk.
"""

from __future__ import annotations

import math
import pathlib
import shutil
from typing import Any

import numpy as np
import pytest
import xarray as xr
from les_render import cameras, compose, hud
from les_render.timeline import Timeline, make_timeline

# -- fixtures -----------------------------------------------------------


def _footprint(x0: float, x1: float, y0: float, y1: float, h: float) -> dict[str, Any]:
    return {"min": [float(x0), float(y0), 0.0], "max": [float(x1), float(y1), float(h)]}


@pytest.fixture  # type: ignore[misc]
def geometry() -> dict[str, Any]:
    # A 3x2 grid of buildings, 32 m square footprints on a 44 m pitch (12 m
    # streets), heights 13-25 m -- mirrors training_data/pyudales_idealized.
    footprints = []
    heights = [[13, 16, 16], [16, 25, 16]]
    for row, y0 in enumerate((7.0, 51.0)):
        for col, x0 in enumerate((47.0, 91.0, 135.0)):
            footprints.append(_footprint(x0, x0 + 32, y0, y0 + 32, heights[row][col]))
    xs = [fp["min"][0] for fp in footprints] + [fp["max"][0] for fp in footprints]
    ys = [fp["min"][1] for fp in footprints] + [fp["max"][1] for fp in footprints]
    zs = [fp["max"][2] for fp in footprints]
    bounds = [[min(xs), min(ys), 0.0], [max(xs), max(ys), max(zs)]]
    return {
        "buildings": {"obj": "geometry/buildings.obj", "glb": "geometry/buildings.glb"},
        "ground": {"obj": "geometry/ground.obj", "glb": "geometry/ground.glb"},
        "buildings_bounds": bounds,
        "max_building_height": float(max(zs)),
        "footprints": footprints,
    }


@pytest.fixture  # type: ignore[misc]
def domain() -> dict[str, Any]:
    return {
        "lower": [-32.0, -28.0, 0.0],
        "upper": [352.0, 164.0, 64.0],
        "spacing": [4.0, 4.0, 4.0],
        "shape": [96, 48, 16],
    }


@pytest.fixture  # type: ignore[misc]
def timeline() -> Timeline:
    return Timeline(fps=30.0, playback_speed=20.0, t_start=0.0, n_frames=600)


@pytest.fixture  # type: ignore[misc]
def shots(
    geometry: dict[str, Any], domain: dict[str, Any], timeline: Timeline
) -> list[dict[str, Any]]:
    return cameras.make_shots(geometry, domain, timeline)


LAYER_NAMES = ["streaklines", "trails", "speed_glow", "vortices", "ground_speed"]

# Flat colour keys mirroring the real manifest contract (variable/type/
# colormap/range, no hand-authored "name [units]" strings). streaklines and
# speed_glow deliberately share (variable, colormap, range) so a merge is
# exercised; trails and ground_speed also read "speed" but with a different
# colormap/range each, so both must come out tagged by origin; vortices
# reads a variable ("q_criterion") nothing else uses, so it stays untagged.
_LAYER_VARIANTS: dict[str, dict] = {
    "streaklines": {
        "type": "particles",
        "variable": "speed",
        "range": [0.0, 9.0],
        "colormap": "viridis",
    },
    "speed_glow": {
        "type": "volume",
        "variable": "speed",
        "range": [0.0, 9.0],
        "colormap": "viridis",
    },
    "trails": {
        "type": "particles",
        "variable": "speed",
        "range": [0.0, 6.0],
        "colormap": "plasma",
    },
    "vortices": {
        "type": "isosurface",
        "variable": "q_criterion",
        "range": [0.0, 3.0],
        "colormap": "plasma",
    },
    "ground_speed": {
        "type": "slice",
        "variable": "speed",
        "range": [0.0, 9.0],
        "colormap": "cividis",
        "axis": "z",
        "position": 2.0,
    },
}


def _fake_lut(n: int = 256) -> list[list[float]]:
    return [[i / (n - 1), 0.2, 1.0 - i / (n - 1)] for i in range(n)]


@pytest.fixture  # type: ignore[misc]
def manifest(
    geometry: dict[str, Any],
    domain: dict[str, Any],
    timeline: Timeline,
    shots: list[dict[str, Any]],
) -> dict[str, Any]:
    time = np.linspace(0.0, timeline.t_end, 20)
    inflow = {
        "time": [float(t) for t in time],
        "angle_deg": [float(-15.0 + 0.02 * t) for t in time],
        "speed": [float(6.0 + 0.5 * math.sin(t / 50.0)) for t in time],
    }
    layers = [
        {
            "name": name,
            "pattern": f"{name}/{name}.{{frame:04d}}.npz",
            "frame_step": 1,
            "n_files": timeline.n_frames,
            "lut_linear_rgb": _fake_lut(),
            **_LAYER_VARIANTS[name],
        }
        for name in LAYER_NAMES
    ]
    return {
        "version": 1,
        "render": {"width": 320, "height": 180, "look": "dark"},
        "timeline": timeline.to_dict(),
        "domain": domain,
        "geometry": geometry,
        "inflow": inflow,
        "layers": layers,
        "shots": shots,
    }


# -- cameras: tiling ------------------------------------------------------


def test_shots_tile_timeline_exactly(
    shots: list[dict[str, Any]], timeline: Timeline
) -> None:
    assert shots[0]["start"] == 0
    assert shots[-1]["end"] == timeline.n_frames - 1
    ordered = sorted(shots, key=lambda s: s["start"])
    for a, b in zip(ordered[:-1], ordered[1:]):
        assert (
            a["end"] + 1 == b["start"]
        ), f"gap/overlap between {a['name']} and {b['name']}"


def test_shots_cover_every_frame_exactly_once(
    shots: list[dict[str, Any]], timeline: Timeline
) -> None:
    covered = np.zeros(timeline.n_frames, dtype=int)
    for s in shots:
        covered[s["start"] : s["end"] + 1] += 1
    assert np.all(covered == 1)


def test_custom_fractions_still_tile(
    geometry: dict[str, Any], domain: dict[str, Any], timeline: Timeline
) -> None:
    spec = {
        "shots": [
            {"type": "establishing", "fraction": 1},
            {"type": "plan", "fraction": 2},
            {"type": "wake", "fraction": 1},
        ]
    }
    out = cameras.make_shots(geometry, domain, timeline, spec)
    assert out[0]["start"] == 0
    assert out[-1]["end"] == timeline.n_frames - 1
    for a, b in zip(out[:-1], out[1:]):
        assert a["end"] + 1 == b["start"]


# -- cameras: key sanity ---------------------------------------------------


def test_keys_have_required_fields(shots: list[dict[str, Any]]) -> None:
    for s in shots:
        assert s["keys"], s["name"]
        for k in s["keys"]:
            assert set(k) >= {"frame", "location", "target", "focal_length_mm", "fstop"}
            assert len(k["location"]) == 3
            assert len(k["target"]) == 3
            assert k["focal_length_mm"] > 0
            assert k["fstop"] > 0


def test_keys_roughly_near_domain(
    shots: list[dict[str, Any]], domain: dict[str, Any]
) -> None:
    lo, hi = np.array(domain["lower"]), np.array(domain["upper"])
    span = hi - lo
    # generous box: cinematic shots may fly a bit outside the simulated
    # domain (e.g. a wide establishing orbit), but not wildly so.
    pad = 1.5 * span
    for s in shots:
        for k in s["keys"]:
            loc = np.array(k["location"])
            assert np.all(loc >= lo - pad) and np.all(loc <= hi + pad), (s["name"], loc)


def test_street_shot_never_inside_a_footprint(
    shots: list[dict[str, Any]], geometry: dict[str, Any]
) -> None:
    footprints = geometry["footprints"]
    street = next(s for s in shots if s["name"] == "street")
    for f in range(street["start"], street["end"] + 1):
        loc, _target, _focal = cameras.sample_camera(shots, f)
        assert not cameras.point_in_building(loc, footprints), (f, loc)


def test_no_shot_ever_inside_a_footprint(
    shots: list[dict[str, Any]], geometry: dict[str, Any]
) -> None:
    footprints = geometry["footprints"]
    for s in shots:
        for f in range(s["start"], s["end"] + 1, 5):  # subsample for speed
            loc, _target, _focal = cameras.sample_camera(shots, f)
            assert not cameras.point_in_building(loc, footprints), (s["name"], f, loc)


def test_street_shot_beside_packed_row_avoids_buildings() -> None:
    """Regression for the street-shot fallback: three buildings in a row with
    no gap between them (no canyon) used to send the dolly straight down the
    row's centreline, inside every footprint. The fallback must now fly
    beside the row instead."""
    footprints = [
        {"min": [x, 40.0, 0.0], "max": [x + 20.0, 60.0, 15.0]}
        for x in (40.0, 80.0, 120.0)
    ]
    geometry = {
        "buildings_bounds": [[40.0, 40.0, 0.0], [140.0, 60.0, 15.0]],
        "max_building_height": 15.0,
        "footprints": footprints,
    }
    domain = {"lower": [0.0, 0.0, 0.0], "upper": [200.0, 100.0, 60.0]}
    timeline = Timeline(fps=30.0, playback_speed=20.0, t_start=0.0, n_frames=300)
    shots = cameras.make_shots(
        geometry, domain, timeline, {"shots": [{"type": "street", "fraction": 1.0}]}
    )
    hits = [
        f
        for f in range(timeline.n_frames)
        if cameras.point_in_building(cameras.sample_camera(shots, f)[0], footprints)
    ]
    assert not hits, hits


def test_street_shot_fallback_raises_camera_when_it_cannot_clear(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """When widening the cross clearance still can't clear every footprint
    (an adjoining building the fallback isn't aware of via ``buildings_bounds``),
    the fallback must raise the camera above the buildings and log a warning,
    rather than silently leaving the camera inside a building."""
    # `buildings_bounds` (used to place the fallback dolly) only spans the
    # near building, but `footprints` (used for collision checks) also has a
    # second, much wider building starting exactly where the fallback's
    # "beside the row" placement would land -- wide enough that widening by a
    # margin at a time for a few iterations still doesn't clear it.
    footprints = [
        {"min": [40.0, 40.0, 0.0], "max": [60.0, 60.0, 15.0]},
        {"min": [40.0, 60.0, 0.0], "max": [60.0, 300.0, 15.0]},
    ]
    geometry = {
        "buildings_bounds": [[40.0, 40.0, 0.0], [60.0, 60.0, 15.0]],
        "max_building_height": 15.0,
        "footprints": footprints,
    }
    domain = {"lower": [0.0, 0.0, 0.0], "upper": [200.0, 400.0, 60.0]}
    timeline = Timeline(fps=30.0, playback_speed=20.0, t_start=0.0, n_frames=120)
    with caplog.at_level("WARNING", logger="les_render.cameras"):
        shots = cameras.make_shots(
            geometry, domain, timeline, {"shots": [{"type": "street", "fraction": 1.0}]}
        )
    assert any("raising the camera" in r.message for r in caplog.records)
    z_vals = [k["location"][2] for k in shots[0]["keys"]]
    expected_z = 15.0 + cameras._margin(geometry, cameras._merge_spec(None))
    assert z_vals == pytest.approx([expected_z] * len(z_vals))
    for f in range(timeline.n_frames):
        loc, _target, _focal = cameras.sample_camera(shots, f)
        assert not cameras.point_in_building(loc, footprints), (f, loc)


# -- cameras: sample_camera --------------------------------------------


def test_sample_camera_holds_at_shot_endpoints(shots: list[dict[str, Any]]) -> None:
    for s in shots:
        loc0, tgt0, focal0 = cameras.sample_camera(shots, s["start"])
        first_key = s["keys"][0]
        if first_key["frame"] == s["start"]:
            assert loc0 == pytest.approx(tuple(first_key["location"]), abs=1e-3)
            assert focal0 == pytest.approx(first_key["focal_length_mm"], abs=1e-6)


def test_sample_camera_continuity(
    shots: list[dict[str, Any]], timeline: Timeline
) -> None:
    # Frame-to-frame motion should be smooth *within* a shot (no teleports).
    # Shot cuts are intentionally hard (no blending across a cut -- a plan
    # shot at high altitude can neighbour a street-level shot), so jumps are
    # only bounded inside each shot's own [start, end] range.
    for s in shots:
        prev_loc: np.ndarray | None = None
        max_jump = 0.0
        for f in range(s["start"], s["end"] + 1):
            loc, _target, _focal = cameras.sample_camera(shots, f)
            loc = np.array(loc)
            if prev_loc is not None:
                max_jump = max(max_jump, float(np.linalg.norm(loc - prev_loc)))
            prev_loc = loc
        assert max_jump < 20.0, (s["name"], max_jump)


def test_sample_camera_interior_keys_stay_at_speed(
    shots: list[dict[str, Any]],
) -> None:
    """Regression: easing per key segment used to decelerate the camera to a
    near-stop at every interior key. Easing should happen once per shot, so a
    multi-key shot keeps cruising through interior keys (speed only drops
    near the shot's own start/end)."""
    checked_a_multi_key_shot = False
    for s in shots:
        keys = s["keys"]
        if len(keys) < 3:
            continue  # no interior keys to check
        frames = list(range(s["start"], s["end"] + 1))
        if len(frames) < 10:
            continue
        locs = np.array([cameras.sample_camera(shots, f)[0] for f in frames])
        speeds = np.linalg.norm(np.diff(locs, axis=0), axis=1)
        median_speed = float(np.median(speeds))
        if median_speed <= 1e-9:
            continue
        checked_a_multi_key_shot = True

        # Interior keys (excluding the shot's first/last) must not force a
        # near-stop: speed around each one stays a healthy fraction of the
        # shot's cruising speed.
        for k in keys[1:-1]:
            idx = k["frame"] - s["start"]
            lo, hi = max(idx - 1, 0), min(idx, len(speeds) - 1)
            speed_here = max(speeds[lo], speeds[hi])
            assert speed_here > 0.3 * median_speed, (
                s["name"],
                k["frame"],
                speed_here,
                median_speed,
            )

        # The shot as a whole still eases in/out at its own start and end.
        assert speeds[0] < 0.5 * median_speed, (s["name"], "start", speeds[0])
        assert speeds[-1] < 0.5 * median_speed, (s["name"], "end", speeds[-1])

    assert checked_a_multi_key_shot  # sanity: the default sequence has one


def test_sample_camera_clamps_outside_range(shots: list[dict[str, Any]]) -> None:
    loc_first, _, _ = cameras.sample_camera(shots, -100)
    loc0, _, _ = cameras.sample_camera(shots, 0)
    assert loc_first == pytest.approx(loc0)

    loc_last, _, _ = cameras.sample_camera(shots, 10**9)
    loc_end, _, _ = cameras.sample_camera(shots, shots[-1]["end"])
    assert loc_last == pytest.approx(loc_end)


def test_establishing_loop_end_matches_start(shots: list[dict[str, Any]]) -> None:
    first = next(s for s in shots if s["name"].startswith("establishing"))
    last = shots[-1]
    assert last["name"].startswith("establishing")
    assert last is not first
    assert tuple(last["keys"][-1]["location"]) == pytest.approx(
        tuple(first["keys"][0]["location"]), abs=1e-3
    )
    assert tuple(last["keys"][-1]["target"]) == pytest.approx(
        tuple(first["keys"][0]["target"]), abs=1e-3
    )


# -- cameras: single_shot presets ------------------------------------------


@pytest.mark.parametrize("kind", ["orbit", "plan", "street", "wake", "establishing"])  # type: ignore[misc]
def test_single_shot_presets(
    geometry: dict[str, Any], domain: dict[str, Any], timeline: Timeline, kind: str
) -> None:
    shot = cameras.single_shot(kind, geometry, domain, timeline)
    assert shot["start"] == 0
    assert shot["end"] == timeline.n_frames - 1
    assert len(shot["keys"]) >= 2
    footprints = geometry["footprints"]
    for f in range(shot["start"], shot["end"] + 1, 10):
        loc, _t, _foc = cameras.sample_camera([shot], f)
        assert not cameras.point_in_building(loc, footprints), (kind, f, loc)


def test_single_shot_custom_frame_range(
    geometry: dict[str, Any], domain: dict[str, Any], timeline: Timeline
) -> None:
    shot = cameras.single_shot("wake", geometry, domain, timeline, frame_range=(10, 40))
    assert shot["start"] == 10
    assert shot["end"] == 40
    for k in shot["keys"]:
        assert 10 <= k["frame"] <= 40


# -- hud ------------------------------------------------------------


def test_render_hud_writes_rgba_transparent_pngs(
    manifest: dict[str, Any], tmp_path: pathlib.Path
) -> None:
    from PIL import Image

    out = hud.render_hud(manifest, {"workers": 1}, tmp_path)
    assert out["pattern"] == "hud/hud.{frame:04d}.png"
    frame_step = out["frame_step"]
    n_files = math.ceil(manifest["timeline"]["n_frames"] / frame_step)

    files = sorted((tmp_path / "hud").glob("hud.*.png"))
    assert len(files) == n_files

    im = Image.open(files[0])
    assert im.mode == "RGBA"
    assert im.size == (manifest["render"]["width"], manifest["render"]["height"])
    alpha = np.array(im)[..., 3]
    assert alpha.min() == 0  # background stays transparent
    assert alpha.max() > 0  # but something was actually drawn


def test_render_hud_frame_step(
    manifest: dict[str, Any], tmp_path: pathlib.Path
) -> None:
    out = hud.render_hud(manifest, {"frame_step": 10}, tmp_path)
    assert out["frame_step"] == 10
    n_files = math.ceil(manifest["timeline"]["n_frames"] / 10)
    files = list((tmp_path / "hud").glob("hud.*.png"))
    assert len(files) == n_files


def test_render_hud_respects_shot_layers(
    manifest: dict[str, Any], tmp_path: pathlib.Path
) -> None:
    # First shot ("establishing") only shows a subset of layers per
    # cameras.DEFAULT_LAYERS; render_hud must not crash and must still
    # produce a valid frame for it.
    from PIL import Image

    out = hud.render_hud(manifest, {"workers": 1}, tmp_path)
    files = sorted((tmp_path / "hud").glob("hud.*.png"))
    im = Image.open(files[0]).convert("RGBA")
    assert im.size == (320, 180)


def test_render_hud_no_inflow_hides_compass(
    manifest: dict[str, Any], tmp_path: pathlib.Path
) -> None:
    manifest = dict(manifest)
    manifest["inflow"] = None
    out = hud.render_hud(manifest, {"workers": 1}, tmp_path)
    assert out["frame_step"] >= 1  # just check it doesn't crash without inflow


def test_render_hud_n_workers_is_a_deprecated_alias_for_workers(
    manifest: dict[str, Any], tmp_path: pathlib.Path
) -> None:
    merged = hud._merge_spec({"n_workers": 4})
    assert merged["workers"] == 4
    # explicit "workers" wins over the legacy alias if both are given
    merged2 = hud._merge_spec({"n_workers": 4, "workers": 2})
    assert merged2["workers"] == 2


def test_inflow_block_from_params_dataset() -> None:
    ds = xr.Dataset(
        {
            "inflow_angle": ("time", np.array([-10.0, -12.0, -14.0])),
            "velocity_magnitude": ("time", np.array([5.0, 5.5, 6.0])),
        },
        coords={"time": np.array([0.0, 5.0, 10.0])},
    )
    block = hud.inflow_block(ds)
    assert block is not None
    assert block["angle_deg"] == [-10.0, -12.0, -14.0]
    assert block["speed"] == [5.0, 5.5, 6.0]
    assert block["time"] == [0.0, 5.0, 10.0]


def test_inflow_block_none() -> None:
    assert hud.inflow_block(None) is None


def test_compass_stays_within_margin() -> None:
    # Regression: the old "+X"/"+Y" labels were drawn outside the dial's
    # rim and ran past the canvas edge. The dial (circle + axis ticks) must
    # now stay fully inside the margin on the top and right edges.
    spec = hud._merge_spec(None)
    layout = hud._Layout.make(320, 180, spec["margin_frac"])
    face, _centre, _radius = hud._compass_face(layout, spec)
    arr = np.array(face)
    margin = layout.margin
    assert arr[:margin, :, 3].max() == 0, "content above the top margin"
    assert (
        arr[:, layout.width - margin :, 3].max() == 0
    ), "content past the right margin"


def test_split_variable_units() -> None:
    assert hud._split_variable_units("wind speed [m/s]") == ("wind speed", "m/s")
    assert hud._split_variable_units("vorticity magnitude [1/s]") == (
        "vorticity magnitude",
        "1/s",
    )
    assert hud._split_variable_units("no units here") == ("no units here", "")


def test_colorbar_label_keeps_units_in_plain_case() -> None:
    # "wind speed [m/s]" must render as small-caps "WIND SPEED" followed by
    # plain-case "m/s" -- not an all-caps "[M/S]". Rendering isn't OCR-able
    # from a PNG, so exercise the drawing helper directly and check the two
    # pieces are laid out side by side (units bbox starts after the name's).
    from PIL import Image, ImageDraw

    spec = hud._merge_spec(None)
    im = Image.new("RGBA", (400, 40), (0, 0, 0, 0))
    draw = ImageDraw.Draw(im)
    font = hud._find_font("regular", 14)
    hud._draw_variable_label(draw, (0, 0), "wind speed [m/s]", font, spec)
    assert np.array(im)[..., 3].max() > 0  # something was drawn

    name_bbox = draw.textbbox(
        (0, 0), hud._small_caps("wind speed"), font=font, anchor="la"
    )
    units_bbox = draw.textbbox((0, 0), "m/s", font=font, anchor="la")
    # sanity: the plain-case unit string is narrower than an all-caps
    # bracketed one would be, and definitely doesn't start at x=0 (it's
    # drawn after the name, not uppercased into the name itself).
    assert units_bbox[2] - units_bbox[0] < name_bbox[2] - name_bbox[0]


def test_fmt_num() -> None:
    assert hud._fmt_num(0.016) == "0"  # near-zero noise collapses to a clean 0
    assert hud._fmt_num(0.0) == "0"
    assert hud._fmt_num(7.4) == "7.4"
    assert hud._fmt_num(9.0) == "9"  # no redundant trailing ".0"
    assert hud._fmt_num(0.42) == "0.42"
    assert hud._fmt_num(130.0) == "130"
    assert hud._fmt_num(2.0) == "2"
    assert hud._fmt_num(-5.3) == "-5.3"


def test_variable_label_known_keys() -> None:
    assert hud._variable_label({"variable": "speed"}) == ("wind speed", "m/s")
    assert hud._variable_label({"variable": "w"}) == ("vertical velocity", "m/s")
    assert hud._variable_label({"variable": "vorticity_magnitude"}) == (
        "vorticity",
        "1/s",
    )
    assert hud._variable_label({"variable": "q_criterion"}) == ("Q-criterion", "1/s²")
    assert hud._variable_label({"variable": "pressure"}) == (
        "pressure (kinematic)",
        "m²/s²",
    )
    assert hud._variable_label({"variable": "gamma_norm"}) == (
        "vorticity (normalised)",
        "",
    )


def test_variable_label_transform_override() -> None:
    # A volume of "speed" with an abs_excess transform is a deviation field,
    # not raw wind speed -- the transform must win over the bare variable.
    label = hud._variable_label({"variable": "speed", "transform": "abs_excess"})
    assert label[0].startswith("speed deviation")
    assert label[1] == "m/s"


def test_variable_label_unknown_falls_back_to_bracket_parsing() -> None:
    assert hud._variable_label({"variable": "custom thing [widgets]"}) == (
        "custom thing",
        "widgets",
    )


def test_layer_tag_ground_speed_slice_suffix() -> None:
    layer = {"name": "ground_speed", "type": "slice", "axis": "z", "position": 2.0}
    word, suffix = hud._layer_tag(layer)
    assert word == "ground"
    assert suffix == " (z = 2 m)"


def test_layer_tag_known_and_unknown_names() -> None:
    assert hud._layer_tag({"name": "streaklines"})[0] == "smoke"
    assert hud._layer_tag({"name": "trails"})[0] == "tracers"
    assert hud._layer_tag({"name": "vortices"})[0] == "vortex cores"
    assert hud._layer_tag({"name": "custom_layer"})[0] == "custom layer"


def test_colorbar_entries_merge_identical_and_tag_divergent(
    manifest: dict[str, Any]
) -> None:
    # streaklines + speed_glow share (variable, colormap, range) exactly and
    # must merge into one bar; trails and ground_speed both also read
    # "speed" but with a different colormap/range each, so all three
    # surviving "wind speed" entries must be tagged; vortices reads
    # q_criterion, which nothing else uses, so it stays untagged.
    entries = hud._build_colorbar_entries(manifest["layers"])
    assert len(entries) == 4  # 5 layers -> streaklines+speed_glow merged

    def joined(entry: dict[str, Any]) -> str:
        return "".join(text for text, _is_caps in entry["segments"])

    texts = [joined(e) for e in entries]
    q_entries = [t for t in texts if "Q-criterion" in t or "q-criterion" in t.lower()]
    assert len(q_entries) == 1
    assert "·" not in q_entries[0]  # untagged: nothing else is q_criterion

    speed_entries = [t for t in texts if "wind speed" in t.lower()]
    assert (
        len(speed_entries) == 3
    )  # merged(streaklines+speed_glow), trails, ground_speed
    assert all("·" in t for t in speed_entries)  # all disambiguated
    assert any("ground" in t.lower() and "z = 2 m" in t.lower() for t in speed_entries)
    assert any("tracers" in t.lower() for t in speed_entries)


def test_colorbar_stack_renders_without_crashing(manifest: dict[str, Any]) -> None:
    spec = hud._merge_spec(None)
    layout = hud._Layout.make(320, 180, spec["margin_frac"])
    im = hud._colorbar_stack(layout, manifest["layers"], spec)
    assert im.size == (320, 180)
    assert np.array(im)[..., 3].max() > 0


# -- compose --------------------------------------------------------


def _make_synthetic_frames(
    tmp_path: pathlib.Path, n: int = 5, size: tuple = (64, 36)
) -> tuple[pathlib.Path, pathlib.Path]:
    from PIL import Image, ImageDraw

    frames_dir = tmp_path / "frames"
    hud_dir = tmp_path / "hud"
    frames_dir.mkdir()
    hud_dir.mkdir()
    for i in range(n):
        frame = Image.new("RGB", size, (20 + i * 20, 50, 80))
        frame.save(frames_dir / f"frame.{i:04d}.png")
        overlay = Image.new("RGBA", size, (0, 0, 0, 0))
        ImageDraw.Draw(overlay).text((2, 2), str(i), fill=(255, 255, 255, 255))
        overlay.save(hud_dir / f"hud.{i:04d}.png")
    return frames_dir, hud_dir


_HAVE_FFMPEG = shutil.which("ffmpeg") is not None


@pytest.mark.skipif(not _HAVE_FFMPEG, reason="ffmpeg not on PATH")  # type: ignore[misc]
def test_compose_video_from_synthetic_frames(tmp_path: pathlib.Path) -> None:
    frames_dir, hud_dir = _make_synthetic_frames(tmp_path)
    out_mp4 = tmp_path / "out.mp4"
    result = compose.compose_video(
        str(frames_dir / "frame.%04d.png"),
        out_mp4,
        fps=5,
        hud_pattern=str(hud_dir / "hud.%04d.png"),
    )
    assert result == out_mp4
    assert out_mp4.is_file()
    assert out_mp4.stat().st_size > 0


@pytest.mark.skipif(not _HAVE_FFMPEG, reason="ffmpeg not on PATH")  # type: ignore[misc]
def test_compose_video_without_hud(tmp_path: pathlib.Path) -> None:
    frames_dir, _hud_dir = _make_synthetic_frames(tmp_path)
    out_mp4 = tmp_path / "out_nohud.mp4"
    compose.compose_video(str(frames_dir / "frame.%04d.png"), out_mp4, fps=5)
    assert out_mp4.is_file()
    assert out_mp4.stat().st_size > 0


@pytest.mark.skipif(not _HAVE_FFMPEG, reason="ffmpeg not on PATH")  # type: ignore[misc]
def test_make_contact_sheet(tmp_path: pathlib.Path) -> None:
    frames_dir, _hud_dir = _make_synthetic_frames(tmp_path)
    out_png = tmp_path / "sheet.png"
    compose.make_contact_sheet(
        str(frames_dir / "frame.%04d.png"), out_png, columns=5, rows=1, tile_width=32
    )
    assert out_png.is_file()
    assert out_png.stat().st_size > 0


def test_compose_video_missing_ffmpeg_raises_clear_error(
    tmp_path: pathlib.Path,
) -> None:
    frames_dir, _hud_dir = _make_synthetic_frames(tmp_path, n=1)
    with pytest.raises(compose.FFmpegNotFoundError):
        compose.compose_video(
            str(frames_dir / "frame.%04d.png"),
            tmp_path / "out.mp4",
            fps=5,
            ffmpeg="/definitely/not/a/real/ffmpeg/binary",
        )
