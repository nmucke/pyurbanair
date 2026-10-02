"""Optional 3D failures must still publish the completed 2D bundle."""

import json
import subprocess
from importlib import import_module
from pathlib import Path

import pytest
from visualization import render

from tests.visualization.test_forward_visualization import regular, save_run


@pytest.mark.parametrize(  # type: ignore[misc]
    ("error", "reason"),
    [
        (subprocess.CalledProcessError(1, ["ffmpeg"]), "exit status 1"),
        (subprocess.TimeoutExpired(["ffmpeg"], 300), "timed out after 300 seconds"),
    ],
    ids=["encoder-exit", "encoder-timeout"],
)
def test_subprocess_failure_in_optional_3d_publishes_2d(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    error: subprocess.SubprocessError,
    reason: str,
) -> None:
    render_3d_module = import_module("visualization.render_3d")

    def fail_after_partial_products(*args: object) -> None:
        output = args[1]
        assert isinstance(output, Path)
        (output / "media" / "flow-3d").mkdir()
        (output / "media" / "flow-3d" / "00000.png").write_bytes(b"partial")
        (output / "media" / "flow-3d.mp4").write_bytes(b"partial")
        (output / "previews" / "flow-3d.png").write_bytes(b"partial")
        raise error

    monkeypatch.setattr(render_3d_module, "render_3d", fail_after_partial_products)
    root = save_run(tmp_path, regular((0, 2.5)))
    bundle = tmp_path / "bundle"
    manifest = render(
        root,
        bundle,
        {
            "render_3d": True,
            "movie": False,
            "width": 320,
            "height": 240,
            "slices": [{"axis": "z", "fraction": 0}],
            "probes": [{"id": "P", "x": 2, "y": 4, "z": 3}],
        },
    )
    assert manifest == json.loads((bundle / "viewer_manifest.json").read_text())
    assert manifest["status"] == "complete"
    assert [view["kind"] for view in manifest["views"]] == ["2d"]
    assert any(
        "Optional 3D rendering unavailable" in warning and reason in warning
        for warning in manifest["warnings"]
    )
    assert not (bundle / "media" / "flow-3d").exists()
    assert not (bundle / "media" / "flow-3d.mp4").exists()
    assert not (bundle / "previews" / "flow-3d.png").exists()
    assert len(manifest["views"][0]["snapshots"]) == 2
    for snapshot in manifest["views"][0]["snapshots"]:
        assert (bundle / snapshot["path"]).read_bytes().startswith(b"\x89PNG")
    assert json.loads((bundle / "probes.json").read_text())["probes"][0]["values"] == [
        5,
        5,
    ]
