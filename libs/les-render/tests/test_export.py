"""Integration tests for the les-render pipeline.

Two things are exercised end to end, against the real ``conf/render_les.yaml``
+ ``conf/render_preset/*.yaml`` and the real ``les_render.export`` stages (no
mocking): Hydra composition of every preset, and a full bundle export driven
through ``scripts/visualization/render_les.py`` on a tiny synthetic case (24 x
12 x 8 cells @ 4 m, 6 snapshots 5 s apart, uniform +x flow, one solid block,
geometry from ``blanking``). See ``docs/les_render.md`` for the bundle
contract this checks against.

``scripts`` is not on ``sys.path`` here (this tests dir has no
``__init__.py``, so pytest's rootdir-insertion import mode puts
``libs/les-render/tests`` on ``sys.path``, not the repo root -- unlike
``tests/`` at the repo root, which *is* a package and does get the root
inserted). ``render_les.py`` is therefore loaded directly by file path via
``importlib``, which is enough since it only imports ``hydra``,
``omegaconf`` and ``les_render`` (all installed in the ``viz`` env), never
anything from its own ``scripts`` package.

Runs in the ``viz`` pixi environment::

    pixi run -e viz pytest libs/les-render/tests/test_export.py -q
"""

from __future__ import annotations

import importlib.util
import json
import math
import pathlib
import shutil
from typing import Any, Optional

import numpy as np
import pytest
import xarray as xr
import yaml
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

REPO = pathlib.Path(__file__).resolve().parents[3]
CONF_DIR = str((REPO / "conf").resolve())
RENDER_LES_SCRIPT = REPO / "scripts" / "visualization" / "render_les.py"

PRESETS = ("cinematic", "smoke_tunnel", "vortex", "comfort_map")
LAYER_TYPES = {"particles", "volume", "isosurface", "slice"}

# Grid: 24 x 12 x 8 cells @ 4 m -> a 96 x 48 x 32 m domain.
H = 4.0
NX, NY, NZ, NT = 24, 12, 8, 6
U = 5.0  # m/s, uniform +x inflow

# Overrides that shrink the cinematic preset down to a fast, tiny export:
# small particle counts, a 1 px/m slice, one worker (no forkserver pools --
# see docs/les_render.md's cost table; slices/volumes/isosurfaces spin up
# process pools above workers=1), a small frame and 1 s of video at 10 fps,
# and the heavy downstream stages (Blender preview, ffmpeg, alembic) off so
# only the bundle export + HUD run.
_SHRINK_OVERRIDES = [
    "time.fps=10",
    "time.duration=1.0",
    "render_preset.layers.trails.counts=200",
    "+render_preset.layers.streaklines.counts={rake_y:3,rake_z:2,ground_line:4}",
    "+render_preset.layers.streaklines.points_per_line=16",
    "render_preset.layers.ground_speed.px_per_metre=1",
    "workers=1",
    "render.width=320",
    "render.height=180",
    "stages.blender=false",
    "stages.video=false",
    "stages.alembic=false",
]


def _load_render_les_module() -> Any:
    """Load scripts/visualization/render_les.py by path (see module docstring
    for why a plain ``import scripts...`` doesn't work here)."""
    spec = importlib.util.spec_from_file_location(
        "render_les_under_test", RENDER_LES_SCRIPT
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _make_case(
    tmp_path: pathlib.Path, name: str = "case", render_yaml: Optional[dict] = None
) -> pathlib.Path:
    """A tiny case folder: ``state.nc`` (u, v, w, blanking) + ``params.nc``,
    no STL, so geometry comes from ``blanking`` (docs/les_render.md's "Input:
    the case folder" table). Uniform +x flow with a bit of noise, one solid
    block in the middle of the domain.
    """
    case_dir = tmp_path / name
    case_dir.mkdir(parents=True, exist_ok=True)

    xt = (np.arange(NX) + 0.5) * H
    yt = (np.arange(NY) + 0.5) * H
    zt = (np.arange(NZ) + 0.5) * H
    time = np.arange(NT) * 5.0
    rng = np.random.default_rng(0)
    shape = (NT, NZ, NY, NX)
    noise = 0.3
    ds = xr.Dataset(
        {
            "u": (
                ("time", "zt", "yt", "xt"),
                (U + noise * rng.standard_normal(shape)).astype(np.float32),
            ),
            "v": (
                ("time", "zt", "yt", "xt"),
                (noise * rng.standard_normal(shape)).astype(np.float32),
            ),
            "w": (
                ("time", "zt", "yt", "xt"),
                (noise * rng.standard_normal(shape)).astype(np.float32),
            ),
        },
        coords={"time": time, "zt": zt, "yt": yt, "xt": xt},
    )
    blanking = np.zeros((NZ, NY, NX), dtype=np.int8)
    blanking[0:3, 4:8, 10:14] = 1  # one solid block: z < 12 m, mid-domain in x/y
    ds["blanking"] = (("zt", "yt", "xt"), blanking)
    ds.to_netcdf(case_dir / "state.nc")

    params = xr.Dataset(
        {
            "inflow_angle": (("time",), np.zeros(NT)),
            "velocity_magnitude": (("time",), np.full(NT, U)),
        },
        coords={"time": time},
    )
    params.to_netcdf(case_dir / "params.nc")

    if render_yaml is not None:
        (case_dir / "render.yaml").write_text(yaml.dump(render_yaml))

    return case_dir


def _compose_cfg(
    input_path: pathlib.Path,
    output_dir: pathlib.Path,
    preset: str = "cinematic",
    overrides: tuple[str, ...] = (),
) -> Any:
    with initialize_config_dir(config_dir=CONF_DIR, version_base=None):
        return compose(
            config_name="render_les",
            overrides=[
                f"render_preset={preset}",
                f"input={input_path}",
                f"output_dir={output_dir}",
                *overrides,
            ],
        )


def _assert_layer_files_exist(bundle: pathlib.Path, layer: dict[str, Any]) -> None:
    for f in range(layer["n_files"]):
        path = bundle / layer["pattern"].format(frame=f)
        assert path.is_file(), (layer["name"], f, path)


def _assert_particle_shapes_constant(
    bundle: pathlib.Path, layer: dict[str, Any]
) -> None:
    expected = (layer["n_lines"], layer["points_per_line"], 3)
    for f in range(layer["n_files"]):
        with np.load(bundle / layer["pattern"].format(frame=f)) as d:
            assert d["points"].shape == expected, (layer["name"], f, d["points"].shape)


def _assert_shots_tile_exactly(shots: list[dict[str, Any]], n_frames: int) -> None:
    ordered = sorted(shots, key=lambda s: s["start"])
    assert ordered[0]["start"] == 0
    assert ordered[-1]["end"] == n_frames - 1
    for a, b in zip(ordered[:-1], ordered[1:]):
        assert a["end"] + 1 == b["start"], (a["name"], b["name"])


# ================================================================
# 1. Hydra composition: conf/render_les.yaml x every render_preset
# ================================================================


class TestHydraComposition:
    @pytest.mark.parametrize("preset", PRESETS)  # type: ignore[misc]
    def test_composes_and_resolves(self, preset: str) -> None:
        with initialize_config_dir(config_dir=CONF_DIR, version_base=None):
            cfg = compose(
                config_name="render_les",
                overrides=[f"render_preset={preset}", "input=/does/not/need/to/exist"],
            )
        container = OmegaConf.to_container(cfg, resolve=True)
        assert isinstance(container, dict)
        assert container["render_preset"]["name"] == preset

        layers = container["render_preset"]["layers"]
        assert layers, f"{preset} preset defines no layers"
        for name, spec in layers.items():
            assert spec["type"] in LAYER_TYPES, (preset, name, spec.get("type"))


# ================================================================
# 2. End to end: run(cfg) -> bundle, then the reuse path
# ================================================================


class TestEndToEndExport:
    def test_bundle_contract_and_reuse(self, tmp_path: pathlib.Path) -> None:
        case_dir = _make_case(tmp_path)
        bundle = tmp_path / "bundle"
        cfg = _compose_cfg(case_dir, bundle, overrides=tuple(_SHRINK_OVERRIDES))

        module = _load_render_les_module()
        out_dir = module.run(cfg)
        assert out_dir == bundle

        manifest_path = bundle / "manifest.json"
        assert manifest_path.is_file()
        manifest = json.loads(manifest_path.read_text())

        # -- manifest contract (docs/les_render.md, "manifest.json (version 1)") --
        assert manifest["version"] == 1
        timeline = manifest["timeline"]
        assert timeline["n_frames"] == len(timeline["frame_times"])
        n_frames = timeline["n_frames"]
        assert n_frames >= 2

        # geometry came from blanking: no STL in the case folder.
        assert manifest["case"]["geometry"] is None
        assert manifest["inflow"] is not None  # params.nc was present

        assert manifest["layers"], "no layers exported"
        particle_layers = []
        for layer in manifest["layers"]:
            _assert_layer_files_exist(bundle, layer)
            if layer["type"] == "particles":
                particle_layers.append(layer)
        assert (
            particle_layers
        ), "expected at least one particles layer (cinematic preset)"
        for layer in particle_layers:
            _assert_particle_shapes_constant(bundle, layer)

        _assert_shots_tile_exactly(manifest["shots"], n_frames)

        hud = manifest["hud"]
        assert hud is not None
        hud_frame_step = hud["frame_step"]
        n_hud_files = math.ceil(n_frames / hud_frame_step)
        hud_files = [f for f in range(n_frames) if f % hud_frame_step == 0]
        assert len(hud_files) == n_hud_files
        for f in hud_files:
            path = bundle / hud["pattern"].format(frame=f)
            assert path.is_file(), (f, path)

        unreal_dir = bundle / "unreal"
        for name in ("build_scene.py", "render.py", "README.md"):
            assert (unreal_dir / name).is_file(), name

        # -- reuse path: export=false, hud=false must read the existing
        # manifest rather than recompute it (les_render.export.build_bundle:
        # read_manifest(out_dir) when stages.export is off).
        reuse_cfg = _compose_cfg(
            case_dir,
            bundle,
            overrides=tuple(_SHRINK_OVERRIDES)
            + ("stages.export=false", "stages.hud=false"),
        )
        mtime_before = manifest_path.stat().st_mtime_ns
        reused_out_dir = module.run(reuse_cfg)
        assert reused_out_dir == bundle
        mtime_after = manifest_path.stat().st_mtime_ns
        assert mtime_after == mtime_before, "reuse path rewrote manifest.json"
        reused_manifest = json.loads(manifest_path.read_text())
        assert reused_manifest == manifest


# ================================================================
# 3. Per-case render.yaml overrides the preset
# ================================================================


class TestRenderYamlOverride:
    def test_render_yaml_look_override_lands_in_manifest(
        self, tmp_path: pathlib.Path
    ) -> None:
        case_dir = _make_case(tmp_path, render_yaml={"render": {"look": "daylight"}})
        bundle = tmp_path / "bundle"
        # cinematic's own look is "dark" (conf/render_preset/cinematic.yaml),
        # so seeing "daylight" in the manifest proves render.yaml was merged.
        cfg = _compose_cfg(case_dir, bundle, overrides=tuple(_SHRINK_OVERRIDES))

        module = _load_render_les_module()
        out_dir = module.run(cfg)
        manifest = json.loads((out_dir / "manifest.json").read_text())
        assert manifest["render"]["look"] == "daylight"


class TestCaseOverridePrecedence:
    """Command line > render.yaml > preset defaults."""

    def test_cli_beats_render_yaml(self, tmp_path: pathlib.Path) -> None:
        mod = _load_render_les_module()
        cfg = _compose_cfg(
            tmp_path, tmp_path / "bundle", overrides=("time.duration=8",)
        )
        merged = mod.apply_case_overrides(
            cfg,
            {"time": {"duration": 20, "fps": 12}, "render": {"look": "daylight"}},
            ["time.duration=8", "render_preset=cinematic", "+extra=1"],
        )
        assert merged.time.duration == 8  # command line wins
        assert merged.time.fps == 12  # render.yaml beats the config default
        assert merged.render.look == "daylight"

    def test_render_yaml_cannot_switch_preset_by_name(
        self, tmp_path: pathlib.Path
    ) -> None:
        mod = _load_render_les_module()
        cfg = _compose_cfg(tmp_path, tmp_path / "bundle", overrides=())
        with pytest.raises(ValueError, match="cannot switch presets"):
            mod.apply_case_overrides(cfg, {"render_preset": "vortex"}, [])


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="needs ffmpeg")  # type: ignore[misc, unused-ignore]
def test_encode_preview_partial_range_ignores_stale_frames(
    tmp_path: pathlib.Path,
) -> None:
    import json
    import subprocess

    from les_render.blender_runner import encode_preview
    from PIL import Image

    bundle = tmp_path / "bundle"
    frames = bundle / "preview" / "frames"
    hud = bundle / "hud"
    frames.mkdir(parents=True)
    hud.mkdir()
    for f in range(10):  # 0-9 on disk: 0-1 and 5-9 are stale leftovers
        Image.new("RGB", (64, 36), (f * 20, 0, 0)).save(frames / f"{f:04d}.png")
        # HUD drawn at a different resolution than the preview -> scaled
        Image.new("RGBA", (128, 72), (0, 0, 0, 0)).save(hud / f"hud.{f:04d}.png")
    manifest = {
        "case": {"name": "demo"},
        "timeline": {"fps": 10, "n_frames": 10},
        "hud": {"pattern": "hud/hud.{frame:04d}.png", "frame_step": 1},
    }
    (bundle / "manifest.json").write_text(json.dumps(manifest))

    mp4 = encode_preview(bundle, frames="2:4")
    n = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-count_frames",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=nb_read_frames",
            "-of",
            "csv=p=0",
            str(mp4),
        ],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    assert int(n) == 3

    with pytest.raises(ValueError, match="contiguous"):
        encode_preview(bundle, frames="0:8:2")
    (frames / "0003.png").unlink()
    with pytest.raises(FileNotFoundError, match="missing"):
        encode_preview(bundle, frames="2:4")
