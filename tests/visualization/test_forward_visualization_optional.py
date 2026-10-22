"""Optional real encoder and browser acceptance tests."""

import json
import shutil
from io import BytesIO
from pathlib import Path

import numpy as np
import pytest
from matplotlib.image import imread
from visualization import BundleAssetServer, render

from tests.visualization.test_forward_visualization import regular, save_run
from tests.visualization.test_forward_visualization_3d import varied

pytestmark = pytest.mark.integration


def test_browser_panels_seek_modes_missing_media_and_narrow_layout(
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
        },
    )
    assert len(manifest["views"]) == 3, manifest["warnings"]
    assert all(view["media"] for view in manifest["views"]), manifest["warnings"]
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


def test_browser_3d_panel_renders_rotates_and_switches(tmp_path: Path) -> None:
    browser_api = pytest.importorskip("playwright.sync_api")
    root = save_run(tmp_path, varied())
    bundle = tmp_path / "bundle"
    render(
        root,
        bundle,
        {"render_3d": True, "movie": False, "width": 320, "height": 240},
    )
    with BundleAssetServer() as server, browser_api.sync_playwright() as playwright:
        # Software WebGL, so the check runs on machines without a GPU.
        browser = playwright.chromium.launch(
            headless=True,
            args=[
                "--no-sandbox",
                "--use-angle=swiftshader",
                "--enable-unsafe-swiftshader",
            ],
        )
        page = browser.new_page(viewport={"width": 1280, "height": 900})
        errors: list[str] = []
        page.on("pageerror", lambda error: errors.append(str(error)))
        page.on(
            "console",
            lambda message: errors.extend([message.text] * (message.type == "error")),
        )
        page.goto(server.register(bundle))
        page.wait_for_selector('.cinema[data-ready="true"]')
        assert page.locator("#mode-3d").is_enabled()
        page.locator("#mode-3d").click()
        host = page.locator("#views-3d .flow3d")
        page.wait_for_selector('#views-3d .flow3d[data-ready="true"]')
        canvas = host.locator("canvas.flow3d-canvas")

        def settled() -> np.ndarray:
            # Paused, the view only redraws on a change: wait until it is still.
            last: np.ndarray | None = None
            for _ in range(20):
                page.wait_for_timeout(150)
                shot = imread(BytesIO(canvas.screenshot()))[..., :3]
                if last is not None and np.array_equal(shot, last):
                    return shot
                last = shot
            raise AssertionError("The 3D view never stopped changing")

        before = settled()
        # Brighter than the near-black background: the sheet or streaks drew.
        assert before.max() > 0.3
        box = canvas.bounding_box()
        assert box
        x, y = box["x"] + box["width"] / 2, box["y"] + box["height"] / 2
        page.mouse.move(x, y)
        page.mouse.down()
        page.mouse.move(x + 150, y + 40, steps=8)
        page.mouse.up()
        assert not np.array_equal(settled(), before)

        assert host.get_attribute("data-mode") == "slice"
        level = host.locator(".flow3d-level")
        label = level.locator("span")
        first = label.inner_text()
        assert first.startswith("z = ") and first.endswith(" m")
        slider = level.locator("input")
        top = slider.get_attribute("max")
        slider.fill("0" if slider.input_value() != "0" else str(top))
        assert label.inner_text() != first
        host.get_by_role("button", name="Full 3D").click()
        assert host.get_attribute("data-mode") == "volume"
        assert level.is_hidden()
        host.get_by_role("button", name="Height slice").click()
        assert host.get_attribute("data-mode") == "slice"
        assert level.is_visible()
        assert not errors, errors
        browser.close()
