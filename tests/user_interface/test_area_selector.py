"""The map page that picks a city area: token, validation and polling.

The MCP tools that wrap it are tested in tests/mcp/test_city_tools.py.
"""

from __future__ import annotations

import json
import threading
import time
import urllib.error
import urllib.request
from typing import Any

import pytest
from user_interface.area_selector import AreaSelector, bbox_area_km2, validate_bbox

# About 0.7 km x 0.6 km in Delft.
DELFT = {"west": 4.350, "south": 52.008, "east": 4.360, "north": 52.0135}


def post(url: str, payload: Any) -> tuple[int, dict[str, Any]]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            return response.status, json.loads(response.read())
    except urllib.error.HTTPError as error:
        body = error.read()
        is_json = error.headers.get_content_type() == "application/json"
        return error.code, json.loads(body) if is_json else {}


@pytest.fixture  # type: ignore[misc]
def selector() -> Any:
    with AreaSelector() as selector:
        yield selector


def test_submitted_area_wakes_the_wait(selector: AreaSelector) -> None:
    selection_id, url = selector.open()
    assert url.startswith("http://127.0.0.1:") and selection_id in url

    # A submission while the caller waits wakes it up.
    threading.Timer(0.2, post, (url + "area", DELFT)).start()
    started = time.monotonic()
    result = selector.wait(selection_id, 10)
    assert time.monotonic() - started < 5
    assert result["status"] == "selected"
    bbox = result["bbox"]
    assert {key: bbox[key] for key in DELFT} == DELFT
    assert bbox["area_km2"] == pytest.approx(0.42, abs=0.03)


def test_pending_returns_after_short_wait(selector: AreaSelector) -> None:
    selection_id, _ = selector.open()
    started = time.monotonic()
    assert selector.wait(selection_id, 0.2)["status"] == "pending"
    assert time.monotonic() - started < 2
    with pytest.raises(ValueError, match="Unknown or expired"):
        selector.wait("x", 0)


def test_serves_page_and_rejects_bad_token(selector: AreaSelector) -> None:
    _, url = selector.open(max_area_km2=2.5)
    with urllib.request.urlopen(url, timeout=10) as response:
        page = response.read().decode()
    assert response.headers.get_content_type() == "text/html"
    assert 'data-max-area-km2="2.5"' in page and "leaflet@1.9.4" in page

    bad = url.replace("/select/", "/select/x")
    with pytest.raises(urllib.error.HTTPError) as error:
        urllib.request.urlopen(bad, timeout=10)
    assert error.value.code == 404
    assert post(bad + "area", DELFT)[0] == 404


def test_rejects_invalid_and_oversized_areas(selector: AreaSelector) -> None:
    selection_id, url = selector.open(max_area_km2=1.0)
    big = {"west": 4.30, "south": 52.00, "east": 4.40, "north": 52.05}
    for payload, message in (
        ({**DELFT, "east": DELFT["west"]}, "west must be smaller"),
        ({**DELFT, "north": 95.0}, "Latitudes"),
        ({**DELFT, "west": "4.35"}, "finite numbers"),
        (big, "maximum is 1 km²"),
    ):
        status, body = post(url + "area", payload)
        assert status == 400 and message in body["error"]
    assert selector.wait(selection_id, 0)["status"] == "pending"


def test_rejects_foreign_origin_and_host() -> None:
    with AreaSelector() as selector:
        _, url = selector.open()
        request = urllib.request.Request(
            url + "area",
            data=json.dumps(DELFT).encode(),
            headers={"Content-Type": "application/json", "Origin": "http://evil"},
            method="POST",
        )
        with pytest.raises(urllib.error.HTTPError) as error:
            urllib.request.urlopen(request, timeout=10)
        assert error.value.code == 403
        with pytest.raises(urllib.error.HTTPError) as error:
            urllib.request.urlopen(url.replace("127.0.0.1", "localhost"), timeout=10)
        assert error.value.code == 403


def test_validate_bbox_accepts_a_pasted_sequence() -> None:
    bbox = validate_bbox([4.350, 52.008, 4.360, 52.0135])
    assert bbox["area_km2"] == pytest.approx(
        bbox_area_km2(4.350, 52.008, 4.360, 52.0135)
    )
    with pytest.raises(ValueError):
        validate_bbox([4.35, 52.0])
