"""The map page that picks a city area: token, validation and polling."""

from __future__ import annotations

import json
import pathlib
import threading
import time
import urllib.error
import urllib.request
from typing import Any

import pytest
from mcp_server.area_selector import AreaSelector, bbox_area_km2, validate_bbox
from mcp_server.tools import Tools

REPO = pathlib.Path(__file__).resolve().parents[2]
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
def tools(tmp_path: pathlib.Path) -> Any:
    tools = Tools(REPO, tmp_path)
    yield tools
    if tools.area_selector is not None:
        tools.area_selector.close()


def test_submitted_area_is_returned(tools: Tools) -> None:
    opened = tools.open_area_selector()
    assert opened["status"] == "pending" and opened["url"] in opened["instructions"]
    assert opened["url"].startswith("http://127.0.0.1:")

    # A submission while the agent waits wakes it up.
    threading.Timer(0.2, post, (opened["url"] + "area", DELFT)).start()
    started = time.monotonic()
    result = tools.get_selected_area(opened["selection_id"], wait_s=10)
    assert time.monotonic() - started < 5
    assert result["status"] == "selected"
    bbox = result["bbox"]
    assert {key: bbox[key] for key in DELFT} == DELFT
    assert bbox["area_km2"] == pytest.approx(0.42, abs=0.03)


def test_pending_returns_after_short_wait(tools: Tools) -> None:
    opened = tools.open_area_selector()
    started = time.monotonic()
    result = tools.get_selected_area(opened["selection_id"], wait_s=0.2)
    assert result["status"] == "pending"
    assert time.monotonic() - started < 2


def test_serves_page_and_rejects_bad_token(tools: Tools) -> None:
    url = tools.open_area_selector(max_area_km2=2.5)["url"]
    with urllib.request.urlopen(url, timeout=10) as response:
        page = response.read().decode()
    assert response.headers.get_content_type() == "text/html"
    assert 'data-max-area-km2="2.5"' in page and "leaflet@1.9.4" in page

    bad = url.replace("/select/", "/select/x")
    with pytest.raises(urllib.error.HTTPError) as error:
        urllib.request.urlopen(bad, timeout=10)
    assert error.value.code == 404
    assert post(bad + "area", DELFT)[0] == 404


def test_rejects_invalid_and_oversized_areas(tools: Tools) -> None:
    opened = tools.open_area_selector(max_area_km2=1.0)
    url = opened["url"] + "area"
    big = {"west": 4.30, "south": 52.00, "east": 4.40, "north": 52.05}
    for payload, message in (
        ({**DELFT, "east": DELFT["west"]}, "west must be smaller"),
        ({**DELFT, "north": 95.0}, "Latitudes"),
        ({**DELFT, "west": "4.35"}, "finite numbers"),
        (big, "maximum is 1 km²"),
    ):
        status, body = post(url, payload)
        assert status == 400 and message in body["error"]
    result = tools.get_selected_area(opened["selection_id"], wait_s=0)
    assert result["status"] == "pending"


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


def test_unknown_or_replaced_selection_is_expired(tools: Tools) -> None:
    expired = {"selection_id": "x", "status": "expired"}
    assert tools.get_selected_area("x", wait_s=0) == expired
    old_id = tools.open_area_selector()["selection_id"]
    assert tools.get_selected_area("x", wait_s=0) == expired
    assert tools.area_selector is not None
    tools.area_selector.close()
    # A new selector replaces the closed one; the old id is expired, not an error.
    tools.open_area_selector()
    result = tools.get_selected_area(old_id, wait_s=0)
    assert result == {"selection_id": old_id, "status": "expired"}


@pytest.fixture  # type: ignore[misc]
def fake_make_case(monkeypatch: pytest.MonkeyPatch) -> list[pathlib.Path]:
    """Replace the OSM + City4CFD build by one that only writes case.json."""
    import pycity4cfd.case

    calls: list[pathlib.Path] = []

    def make_case(bbox: Any, case_dir: pathlib.Path, **kwargs: Any) -> Any:
        calls.append(case_dir)
        result = {"case_dir": str(case_dir), "overrides": ["domain.nx=8"]}
        case_dir.mkdir(parents=True)
        (case_dir / "case.json").write_text(json.dumps({"bbox": list(bbox), **result}))
        return result

    monkeypatch.setattr(pycity4cfd.case, "make_case", make_case)
    return calls


def test_city_case_area_limit(tmp_path: pathlib.Path, fake_make_case: Any) -> None:
    # About 7 km x 5.5 km: above the default 4 km² but under 50 km².
    big = {"west": 4.30, "south": 52.00, "east": 4.40, "north": 52.05}
    tools = Tools(REPO, tmp_path)
    tools.repo_root = tmp_path
    with pytest.raises(ValueError, match="maximum is 4 km²"):
        tools.prepare_city_case(big)
    assert fake_make_case == []
    case = tools.prepare_city_case(big, max_area_km2=50)
    assert case["overrides"] == ["domain.nx=8"] and len(fake_make_case) == 1


def test_city_case_is_reused(tmp_path: pathlib.Path, fake_make_case: Any) -> None:
    tools = Tools(REPO, tmp_path)
    tools.repo_root = tmp_path
    first = tools.prepare_city_case(DELFT, resolution=4)
    # Same inputs (4 == 4.0) map to the same folder and reuse its case.json.
    again = tools.prepare_city_case(DELFT, resolution=4.0, default_height=10)
    assert again == first and len(fake_make_case) == 1
    assert "bbox" not in again
    tools.prepare_city_case(DELFT, resolution=2.0)
    assert len(fake_make_case) == 2 and fake_make_case[0] != fake_make_case[1]


def test_older_city_case_is_not_reused(
    tmp_path: pathlib.Path, fake_make_case: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    import hashlib

    import pycity4cfd.case

    tools = Tools(REPO, tmp_path)
    tools.repo_root = tmp_path
    # A folder built before the case format was versioned, at its old key.
    values = (*DELFT.values(), 4.0, 10.0)
    old_key = ",".join(f"{value:.5f}" for value in values)
    old_dir = (
        tmp_path
        / ".temp"
        / "city_cases"
        / ("area_" + hashlib.sha1(old_key.encode()).hexdigest()[:10])
    )
    old_dir.mkdir(parents=True)
    (old_dir / "case.json").write_text(json.dumps({"overrides": ["old"]}))
    case = tools.prepare_city_case(DELFT, resolution=4.0)
    assert case["overrides"] == ["domain.nx=8"] and fake_make_case[0] != old_dir
    # A new case format builds a new folder instead of reusing this one.
    monkeypatch.setattr(pycity4cfd.case, "CASE_FORMAT", pycity4cfd.case.CASE_FORMAT + 1)
    tools.prepare_city_case(DELFT, resolution=4.0)
    assert len(fake_make_case) == 2 and fake_make_case[0] != fake_make_case[1]
