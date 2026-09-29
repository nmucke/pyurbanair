"""Optional real encoder, browser, and offscreen VTK acceptance tests."""

import json
import shutil
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from pyurbanair.visualization import BundleAssetServer, render
from tests.test_forward_visualization import regular, save_run

pytestmark = pytest.mark.integration


def test_vtk_order_mask_and_offscreen(tmp_path: Path) -> None:
    pv = pytest.importorskip("pyvista")
    from pyurbanair.visualization.render_3d import probe_offscreen, rectilinear_field

    assert probe_offscreen()["available"]
    ds = regular((0,)).isel(time=0, drop=True)
    ds["u"] = ds.u * 0 + ds.x
    ds["v"] = ds.v * 0 + ds.y
    ds["w"] = ds.w * 0 + ds.z
    field = rectilinear_field(ds)
    np.testing.assert_allclose(field["velocity"], field.points)
    # Solid plane splits mesh; streamlines launched to its left cannot cross it.
    masked = ds.copy(deep=True)
    for name in ("u", "v", "w"):
        masked[name] = xr.full_like(masked[name], 1.0 if name == "u" else 0.0)
        masked[name].loc[dict(x=4)] = np.nan
    fluid = rectilinear_field(masked)
    lines = fluid.streamlines_from_source(
        pv.PolyData([[1.2, 4.0, 3.0]]),
        vectors="velocity",
        integration_direction="forward",
        max_steps=200,
        compute_vorticity=False,
    )
    assert lines.n_points > 0
    assert np.all(lines.points[:, 0] <= 2.0001)
    root = save_run(tmp_path, regular((0,)))
    manifest = render(
        root,
        tmp_path / "bundle",
        {
            "render_3d": True,
            "movie": False,
            "variable": "speed",
            "width": 320,
            "height": 240,
            "seeds": [[2, 4, 3]],
        },
    )
    assert len(manifest["views"]) == 2, manifest["warnings"]
    assert (
        (tmp_path / "bundle" / "previews" / "flow-3d.png")
        .read_bytes()
        .startswith(b"\x89PNG")
    )


def test_browser_playback_seek_switch_missing_media_and_narrow_layout(
    tmp_path: Path,
) -> None:
    browser_api = pytest.importorskip("playwright.sync_api")
    if not shutil.which("ffmpeg"):
        pytest.skip("ffmpeg unavailable")
    root = save_run(tmp_path)
    bundle = tmp_path / "bundle"
    manifest = render(
        root,
        bundle,
        {
            "fps": 2,
            "movie": True,
            "width": 320,
            "height": 240,
            "slices": [{"axis": "z", "position": 1}, {"axis": "z", "position": 3}],
            "probes": [{"id": "<img src=x onerror=alert(1)>", "x": 4, "y": 4, "z": 3}],
        },
    )
    assert all(view["media"] for view in manifest["views"]), manifest["warnings"]
    # Different view frame intervals verify switching by physical time.
    manifest["views"][1]["frames"] = [
        {"video_time": 0, "simulation_time": 0},
        {"video_time": 0.25, "simulation_time": 2.5},
        {"video_time": 0.5, "simulation_time": 7},
    ]
    (bundle / "viewer_manifest.json").write_text(json.dumps(manifest))
    with BundleAssetServer() as server, browser_api.sync_playwright() as playwright:
        url = server.register(bundle)
        browser = playwright.chromium.launch(headless=True, args=["--no-sandbox"])
        page = browser.new_page(viewport={"width": 1200, "height": 850})
        errors = []
        page.on("pageerror", lambda error: errors.append(str(error)))
        page.goto(url)
        page.wait_for_function(
            "() => document.querySelector('#status').textContent === 'Ready'"
        )
        assert page.locator(".probe-chart h3").inner_text().startswith("<img src=x")
        assert page.locator(".probe-chart img").count() == 0
        page.locator("#seek").fill("0.6")
        page.locator("#seek").dispatch_event("input")
        page.wait_for_function(
            "() => document.querySelector('#physical').textContent.includes('2.500')"
        )
        page.locator("#view").select_option("1")
        page.wait_for_function(
            "() => document.querySelector('#status').textContent === 'Ready'"
        )
        assert page.locator("#video").evaluate(
            "video => video.currentTime"
        ) == pytest.approx(0.25, abs=0.01)
        page.locator("#restart").click()
        page.wait_for_function(
            "() => document.querySelector('#physical').textContent.includes('0.000')"
        )
        page.locator("#rate").select_option("2")
        assert page.locator("#video").evaluate("video => video.playbackRate") == 2
        page.locator("#play").click()
        page.wait_for_function("() => !document.querySelector('#video').paused")
        page.locator("#play").click()
        assert page.locator("#video").evaluate("video => video.paused")
        page.set_viewport_size({"width": 375, "height": 700})
        assert page.evaluate(
            "document.documentElement.scrollWidth <= window.innerWidth"
        )
        assert page.locator("#download").get_attribute("href").endswith(".mp4")
        # Invalid media produces an accessible still fallback, not dead controls.
        page.route("**/*.mp4", lambda route: route.abort())
        page.locator("#view").select_option("0")
        page.wait_for_function(
            "() => document.querySelector('#status').textContent.includes('showing PNG preview')"
        )
        assert page.locator("#poster").is_visible()
        assert page.locator("#download").get_attribute("href").endswith(".png")
        assert not errors
        browser.close()


def test_browser_single_frame_view(tmp_path: Path) -> None:
    browser_api = pytest.importorskip("playwright.sync_api")
    root = save_run(tmp_path, regular((8,)))
    bundle = tmp_path / "bundle"
    render(root, bundle, {"width": 320, "height": 240})
    with BundleAssetServer() as server, browser_api.sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True, args=["--no-sandbox"])
        page = browser.new_page()
        page.goto(server.register(bundle))
        page.wait_for_function(
            "() => document.querySelector('#status').textContent.includes('Still preview')"
        )
        assert page.locator("#poster").is_visible()
        assert not page.locator("#playback").is_visible()
        assert "8.000" in page.locator("#physical").inner_text()
        browser.close()
