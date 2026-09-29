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
    state = regular((0,))
    state["blanking"] = xr.zeros_like(state.u)
    state.blanking.loc[dict(x=[4, 7], y=[4, 8], z=[1, 3])] = 1
    geometry = tmp_path / "geometry.stl"
    pv.Box(bounds=[4, 7, 4, 8, -10, 3]).triangulate().save(geometry)
    root = save_run(tmp_path, state)
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
            "geometry": str(geometry),
        },
    )
    assert len(manifest["views"]) == 4, manifest["warnings"]
    flow = manifest["views"][-1]
    assert flow["geometry"]["display_bounds"] == [0.5, 13, 1, 14, 0, 7.5]
    assert flow["slice"]["actual"] == 1
    assert (
        flow["slice"]["rendered_points"] > 0
    )  # Coincident outer grid face stays visible.
    assert (
        (tmp_path / "bundle" / "previews" / "flow-3d.png")
        .read_bytes()
        .startswith(b"\x89PNG")
    )


def test_browser_panels_seek_modes_missing_media_and_narrow_layout(
    tmp_path: Path,
) -> None:
    browser_api = pytest.importorskip("playwright.sync_api")
    pytest.importorskip("pyvista")
    if not shutil.which("ffmpeg"):
        pytest.skip("ffmpeg unavailable")
    from pyurbanair.visualization.render import encode_movie

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
            "render_3d": True,
            "seeds": [[2, 4, 3]],
        },
    )
    assert len(manifest["views"]) == 4, manifest["warnings"]
    assert all(view["media"] for view in manifest["views"]), manifest["warnings"]
    # A real 3D clip at twice the FPS must preserve physical time on mode switches.
    flow = manifest["views"][3]
    assert encode_movie(bundle / "media/flow-3d", bundle / flow["media"], 4)
    flow["frames"] = [
        {"video_time": i / 4, "simulation_time": t} for i, t in enumerate((0, 2.5, 7))
    ]
    flow["duration"] = 0.75
    (bundle / "viewer_manifest.json").write_text(json.dumps(manifest))
    probes = json.loads((bundle / "probes.json").read_text())
    probes["probes"][0]["label"] = "<img src=x onerror=alert(1)>"
    (bundle / "probes.json").write_text(json.dumps(probes))
    with BundleAssetServer() as server, browser_api.sync_playwright() as playwright:
        url = server.register(bundle)
        browser = playwright.chromium.launch(headless=True, args=["--no-sandbox"])
        page = browser.new_page(viewport={"width": 1440, "height": 1000})
        errors = []
        page.on("pageerror", lambda error: errors.append(str(error)))
        page.goto(url)
        page.wait_for_selector('.cinema[data-ready="true"]')
        assert page.locator("#views-2d .view-panel:visible").count() == 3
        assert page.locator("#views-3d .view-panel:visible").count() == 0
        assert page.locator('.probe-chart[data-trace-count="3"]').count() == 2
        assert "<img src=x" in page.locator(".sensor-legend").first.inner_text()
        assert page.locator(".sensor-legend img").count() == 0
        maps = page.locator('#views-2d [data-axis="z"]')
        left, right = maps.nth(0).bounding_box(), maps.nth(1).bounding_box()
        section = page.locator(".vertical-section").bounding_box()
        assert left and right and section
        assert abs(left["y"] - right["y"]) < 2 and left["x"] < right["x"]
        assert section["y"] >= left["y"] + left["height"]
        assert section["width"] > left["width"] * 1.8
        page.wait_for_function(
            "() => [...document.querySelectorAll('.panel-video')].every(v => v.readyState >= 1)"
        )
        page.locator("#seek").fill("2.5")
        page.locator("#seek").dispatch_event("input")
        page.wait_for_function(
            "() => [...document.querySelectorAll('#views-2d .view-panel')].every(p => p.dataset.sampleTime === '2.5' && p.dataset.seeking === 'false')"
        )
        assert page.locator("#physical").get_attribute("data-time") == "2.5"
        assert all(
            "2.500" in label
            for label in page.locator(".sensor-time").all_text_contents()
        )
        page.locator("#mode-3d").click()
        page.wait_for_selector(
            '#views-3d .view-panel[data-sample-time="2.5"][data-seeking="false"]'
        )
        assert page.locator("#views-3d video").evaluate(
            "video => video.currentTime"
        ) == pytest.approx(0.25, abs=0.01)
        assert page.locator("#views-2d .view-panel:visible").count() == 0
        page.locator("#mode-2d").click()
        assert page.locator("#physical").get_attribute("data-time") == "2.5"
        page.locator("#restart").click()
        assert page.locator("#physical").get_attribute("data-time") == "0"
        page.locator("#rate").select_option("0.5")
        page.locator("#play").click()
        page.wait_for_selector('#playback[data-playing="true"]')
        page.wait_for_function(
            "() => Number(document.querySelector('#physical').dataset.time) > 0"
        )
        page.locator("#play").click()
        assert page.locator("#play").get_attribute("aria-pressed") == "false"
        page.set_viewport_size({"width": 375, "height": 700})
        assert page.evaluate(
            "document.documentElement.scrollWidth <= window.innerWidth"
        )
        assert page.locator("#views-2d .view-panel:visible").count() == 3
        assert page.locator("#views-2d .panel-downloads a[download]").count() == 6
        with page.expect_download() as download:
            page.locator("#views-2d .panel-downloads a").first.click()
        assert download.value.suggested_filename.endswith(".png")
        # Block every movie: all three PNG sequences must still seek together.
        page.route("**/*.mp4", lambda route: route.abort())
        page.reload()
        page.wait_for_selector('.cinema[data-ready="true"]')
        page.wait_for_function(
            "() => [...document.querySelectorAll('#views-2d .panel-status')].every(p => p.textContent.includes('Movie unavailable'))"
        )
        page.locator("#seek").fill("7")
        page.locator("#seek").dispatch_event("input")
        page.wait_for_function(
            "() => [...document.querySelectorAll('#views-2d .view-panel')].every(p => p.dataset.sampleTime === '7' && p.querySelector('img').complete)"
        )
        assert page.locator("#views-2d .panel-poster:visible").count() == 3
        assert page.locator("#views-2d a:visible").count() == 3
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
        page.wait_for_selector('.cinema[data-ready="true"]')
        assert page.locator("#views-2d .panel-poster:visible").count() == 3
        assert page.locator("#play").is_disabled()
        assert page.locator("#seek").is_disabled()
        assert page.locator("#mode-3d").is_disabled()
        assert "unavailable" in page.locator("#mode-help").inner_text()
        assert "8.000" in page.locator("#physical").inner_text()
        browser.close()
