"""The city tools: map selection wrappers and case builds as supervisor jobs.

The supervisor is replaced by its registry in-process (no worker starts), and
the worker step is called directly with a fake `pycity4cfd.case`.
"""

from __future__ import annotations

import concurrent.futures
import hashlib
import json
import pathlib
import sys
import types
from typing import Any

import pytest
from mcp_server.jobs import city_case
from mcp_server.jobs.registry import Registry
from mcp_server.jobs.worker import execute
from mcp_server.tools import Tools

REPO = pathlib.Path(__file__).resolve().parents[2]
# About 0.7 km x 0.6 km in Delft.
DELFT = {"west": 4.350, "south": 52.008, "east": 4.360, "north": 52.0135}
CASE = {"case_dir": "unused", "overrides": ["domain.nx=8"], "n_buildings": 3}


@pytest.fixture  # type: ignore[misc]
def tools(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> Any:
    """Tools on a checkout in tmp_path with the city4cfd env and a local queue."""
    tools: Any = Tools(REPO, tmp_path / "store")
    tools.repo_root = tmp_path
    python = tmp_path / ".pixi/envs/city4cfd/bin/python"
    python.parent.mkdir(parents=True)
    python.touch()
    registry = Registry(tmp_path / "store")
    calls: list[str] = []

    def request(operation: str, **args: Any) -> Any:
        calls.append(operation)
        if operation == "submit":
            return registry.submit(args["payload"], args["key"])
        if operation == "status":
            return registry.get(args["job_id"])
        raise AssertionError(operation)

    monkeypatch.setattr(tools.jobs, "request", request)
    tools.registry, tools.calls = registry, calls
    yield tools
    if tools.area_selector is not None:
        tools.area_selector.close()


def finish(tools: Any, job_id: str, case: dict[str, Any] | None) -> None:
    """Settle a job as the supervisor would after the worker's completion."""
    job = tools.registry.get(job_id)
    if case is None:
        tools.registry.update(job_id, "failed", error="ValueError: no buildings")
        return
    case_dir = pathlib.Path(job["payload"]["case_dir"])
    case_dir.mkdir(parents=True)
    (case_dir / "case.json").write_text(json.dumps({"bbox": [0, 0, 1, 1], **case}))
    completion = {"state": "succeeded", "result": case}
    tools.registry.update(job_id, "succeeded", completion=completion)


def test_selector_tools(tools: Tools) -> None:
    from user_interface.area_selector import MAX_AREA_KM2

    expired = {"selection_id": "x", "status": "expired"}
    assert tools.get_selected_area("x", wait_s=0) == expired
    opened = tools.open_area_selector()
    assert opened["status"] == "pending" and opened["url"] in opened["instructions"]
    assert opened["max_area_km2"] == MAX_AREA_KM2
    assert tools.get_selected_area("x", wait_s=0) == expired
    selection = tools.get_selected_area(opened["selection_id"], wait_s=0)
    assert selection["status"] == "pending"
    assert tools.area_selector is not None
    tools.area_selector.close()
    # A new selector replaces the closed one; the old id is expired, not an error.
    tools.open_area_selector(max_area_km2=2.0)
    result = tools.get_selected_area(opened["selection_id"], wait_s=0)
    assert result == {"selection_id": opened["selection_id"], "status": "expired"}


def test_city_case_area_limit(tools: Any) -> None:
    # About 7 km x 5.5 km: above the default 4 km² but under 50 km².
    big = {"west": 4.30, "south": 52.00, "east": 4.40, "north": 52.05}
    with pytest.raises(ValueError, match="maximum is 4 km²"):
        tools.prepare_city_case(big)
    assert tools.calls == []
    assert tools.prepare_city_case(big, max_area_km2=50)["status"] == "building"


def test_city_case_builds_once_then_is_reused(tools: Any) -> None:
    # Concurrent and repeated requests for the same inputs (4 == 4.0) share a job.
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        started = list(
            pool.map(
                lambda resolution: tools.prepare_city_case(DELFT, resolution),
                [4, 4.0, 4.000001, 4],
            )
        )
    assert {s["status"] for s in started} == {"building"}
    (job_id,) = {s["case_job_id"] for s in started}
    job = tools.registry.get(job_id)
    assert job["kind"] == "city_case" and job["payload"]["environment"] == "city4cfd"
    assert job["payload"]["bbox"] == list(DELFT.values())
    assert tools.get_city_case(job_id) == {"case_job_id": job_id, "status": "building"}
    other = tools.prepare_city_case(DELFT, resolution=2.0)["case_job_id"]
    assert other != job_id

    finish(tools, job_id, CASE)
    assert tools.get_city_case(job_id) == {
        "case_job_id": job_id,
        "status": "ready",
        **CASE,
    }
    # A finished folder is returned at once, without the queue.
    tools.calls.clear()
    assert tools.prepare_city_case(DELFT) == {"status": "ready", **CASE}
    assert tools.calls == []


def test_failed_or_deleted_case_is_rebuilt(tools: Any) -> None:
    first = tools.prepare_city_case(DELFT)["case_job_id"]
    finish(tools, first, None)
    failed = tools.get_city_case(first)
    assert failed["status"] == "failed" and "no buildings" in failed["error"]
    second = tools.prepare_city_case(DELFT)["case_job_id"]
    assert second != first
    finish(tools, second, CASE)
    case_dir = pathlib.Path(tools.registry.get(second)["payload"]["case_dir"])
    (case_dir / "case.json").unlink()
    third = tools.prepare_city_case(DELFT)["case_job_id"]
    assert third not in {first, second}


def test_city_case_needs_the_city4cfd_env(tools: Any) -> None:
    (tools.repo_root / ".pixi/envs/city4cfd/bin/python").unlink()
    with pytest.raises(ValueError, match="pixi install -e city4cfd"):
        tools.prepare_city_case(DELFT)
    assert tools.calls == []


def test_get_city_case_rejects_other_jobs(tools: Any) -> None:
    job = tools.registry.submit({"kind": "forward"}, "run")
    with pytest.raises(ValueError, match="city case"):
        tools.get_city_case(job["id"])


def test_older_city_case_is_not_reused(
    tools: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A folder built before the case format was versioned, at its old key.
    values = (*DELFT.values(), 4.0, 10.0)
    old_key = ",".join(f"{value:.5f}" for value in values)
    name = "area_" + hashlib.sha1(old_key.encode()).hexdigest()[:10]
    old_dir = tools.repo_root / ".temp" / "city_cases" / name
    old_dir.mkdir(parents=True)
    (old_dir / "case.json").write_text(json.dumps({"overrides": ["old"]}))
    job_id = tools.prepare_city_case(DELFT)["case_job_id"]
    case_dir = tools.registry.get(job_id)["payload"]["case_dir"]
    assert case_dir != str(old_dir)
    # A new case format builds a new folder instead of reusing this one.
    monkeypatch.setattr(city_case, "CASE_FORMAT", city_case.CASE_FORMAT + 1)
    newer = tools.prepare_city_case(DELFT)["case_job_id"]
    assert tools.registry.get(newer)["payload"]["case_dir"] != case_dir


def test_worker_builds_the_case(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The city_case worker step calls make_case and records its result."""
    calls: list[tuple[Any, ...]] = []

    def make_case(bbox: Any, case_dir: Any, **kwargs: Any) -> dict[str, Any]:
        calls.append((bbox, case_dir, kwargs))
        if kwargs["resolution"] < 0:
            raise ValueError("bad resolution")
        return {**CASE, "case_dir": str(case_dir)}

    package = types.ModuleType("pycity4cfd")
    module = types.ModuleType("pycity4cfd.case")
    module.make_case = make_case  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "pycity4cfd", package)
    monkeypatch.setitem(sys.modules, "pycity4cfd.case", module)
    payload = {
        "kind": "city_case",
        "bbox": [4.35, 52.008, 4.36, 52.0135],
        "resolution": 4.0,
        "default_height": 10.0,
        "case_dir": str(tmp_path / "case"),
    }
    root = tmp_path / "run"
    execute({"run_root": str(root), "payload": payload})
    assert calls == [
        (
            (4.35, 52.008, 4.36, 52.0135),
            str(tmp_path / "case"),
            {"resolution": 4.0, "default_height": 10.0},
        )
    ]
    completion = json.loads((root / "completion.json").read_text())
    assert completion["state"] == "succeeded"
    assert completion["result"] == {**CASE, "case_dir": str(tmp_path / "case")}

    with pytest.raises(SystemExit):
        execute({"run_root": str(root), "payload": {**payload, "resolution": -1}})
    completion = json.loads((root / "completion.json").read_text())
    assert completion["state"] == "failed" and "bad resolution" in completion["error"]
    assert "result" not in completion


def test_case_dir_key_is_stable() -> None:
    # Folders built before the move of CASE_FORMAT keep their names.
    values = (*DELFT.values(), 4.0, 10.0)
    key = f"v{city_case.CASE_FORMAT}," + ",".join(f"{v:.5f}" for v in values)
    name = "area_" + hashlib.sha1(key.encode()).hexdigest()[:10]
    west, south, east, north = DELFT.values()
    assert city_case.case_dir("/r", (west, south, east, north), 4.0, 10.0) == (
        pathlib.Path("/r/.temp/city_cases") / name
    )
