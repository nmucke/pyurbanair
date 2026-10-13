"""The browser app's server: security checks, options, case and run lifecycles.

`FakeTools` keeps the real config composition (`inspect_config`) and replaces
everything that downloads, builds or runs: no network, no solver, no
supervisor. Its case build is job "case1", held while `release` is clear.
"""

from __future__ import annotations

import json
import pathlib
import threading
import time
import urllib.error
import urllib.request
from typing import Any, cast

import geopandas as gpd
import pytest
from mcp_server.tools import Tools
from shapely.geometry import box
from user_interface.app import FIELDS, MODEL_IDS, App, lookup

REPO = pathlib.Path(__file__).resolve().parents[2]
DELFT = {"west": 4.3555, "south": 52.0095, "east": 4.3600, "north": 52.0122}


class FakeTools(Tools):
    def __init__(self, store: pathlib.Path) -> None:
        super().__init__(REPO, store / "jobs")
        self.store = store
        self.prepared: list[list[str]] = []
        self.jobs_state: dict[str, str] = {}
        self.fail_case = False
        self.cached = False
        self.prepare_error: Exception | None = None
        self.release = threading.Event()
        self.release.set()
        self.frame_ancestor: str | None = None

    def prepare_city_case(self, bbox: Any, **kwargs: Any) -> dict[str, Any]:  # type: ignore[override]
        if self.prepare_error is not None:
            raise self.prepare_error
        if self.cached:
            return {"status": "ready", **self._case()}
        return {"status": "building", "case_job_id": "case1"}

    def get_city_case(self, case_job_id: str) -> dict[str, Any]:
        assert case_job_id == "case1"
        if not self.release.is_set():
            return {"case_job_id": case_job_id, "status": "building"}
        if self.fail_case:
            error = "ValueError: No buildings in this area"
            return {"case_job_id": case_job_id, "status": "failed", "error": error}
        return {"case_job_id": case_job_id, "status": "ready", **self._case()}

    def _case(self) -> dict[str, Any]:
        case_dir = self.store / "case"
        case_dir.mkdir(exist_ok=True)
        # One building in UTM 31N near the bbox's south-west corner.
        corner = gpd.GeoSeries.from_xy([DELFT["west"]], [DELFT["south"]], crs=4326)
        x, y = corner.to_crs(32631).iloc[0].coords[0]
        buildings = gpd.GeoDataFrame(
            {"height": [12.0], "height_source": ["height"]},
            geometry=[box(x + 60, y + 60, x + 80, y + 80)],
            crs=32631,
        )
        buildings.to_file(case_dir / "buildings.geojson", driver="GeoJSON")
        case = {
            "case_dir": str(case_dir),
            # Sensor (50, 50) is the bbox's UTM corner: the origin is 50 m off.
            "crs": "EPSG:32631",
            "origin": [x - 50.0, y - 50.0],
            "overrides": [
                "domain.nx=100",
                "obs.x_points=[50.0, 70.0]",
                "obs.y_points=[50.0, 50.0]",
            ],
            "n_buildings": 1,
            "height_sources": {"height": 1},
            "max_height": 12.0,
            "domain": {"nx": 100, "ny": 100, "nz": 16, "lengths_m": [400.0] * 3},
        }
        (case_dir / "case.json").write_text(
            json.dumps({"bbox": [DELFT[k] for k in DELFT], **case})
        )
        return case

    def prepare_forward_run(self, overrides: Any = None, **_: Any) -> dict[str, Any]:  # type: ignore[override]
        self.prepared.append(list(overrides))
        issues = []
        if "model.forward_model.ncpu=3" in overrides:
            issues = [{"field": "ncpu", "message": "too many", "kind": "configuration"}]
        return {
            "plan_id": "plan1",
            "validation": {
                "configuration_valid": not issues,
                "prerequisites_present": True,
                "issues": issues,
            },
        }

    def _job(self, job_id: str, kind: str = "forward") -> dict[str, Any]:
        root = self.store / "runs" / job_id
        root.mkdir(parents=True, exist_ok=True)
        (root / "worker.log").write_text("".join(f"line {i}\n" for i in range(100)))
        return {
            "run_id": job_id,
            "state": self.jobs_state[job_id],
            "kind": kind,
            "run_root": str(root),
            "details": {"phase": "running"},
        }

    def launch_forward_run(self, plan_id: str, idempotency_key: str, **_: Any) -> Any:  # type: ignore[override]
        self.jobs_state["run1"] = "queued"
        return self._job("run1")

    def get_run_status(self, run_id: str) -> dict[str, Any]:
        if run_id not in self.jobs_state:
            raise ValueError(f"Unknown job: {run_id}")
        return self._job(run_id)

    def get_run_logs(self, run_id: str, cursor: int = 0, limit: int = 32768) -> Any:
        data = (self.store / "runs" / run_id / "worker.log").read_bytes()
        return {"text": data[cursor : cursor + limit].decode(), "cursor": cursor}

    def cancel_run(self, run_id: str) -> Any:
        self.jobs_state[run_id] = "cancelling"
        return self._job(run_id)

    def inspect_run_results(  # type: ignore[override]
        self, run_id: str, artifact_id: Any = None, **_: Any
    ) -> Any:
        if artifact_id is None:
            return {"artifacts": [{"artifact_id": 0, "path": "state.nc"}]}
        return {
            "dimensions": {"time": 3},
            "variables": {"v": {}, "u": {}},
            "coordinates": {"time": {"size": 3, "first": "0.0", "last": "2.0"}},
        }

    def render_simulation(self, run_id: str, idempotency_key: str, **_: Any) -> Any:  # type: ignore[override]
        self.jobs_state["vis1"] = "succeeded"
        return {"visualization_id": "vis1"}

    def visualization(
        self, visualization_id: str, preview: Any = None, frame_ancestor: Any = None
    ) -> Any:
        self.frame_ancestor = frame_ancestor
        return {"state": "succeeded", "viewer_url": "http://127.0.0.1:1/v/"}, None


@pytest.fixture  # type: ignore[misc]
def app(tmp_path: pathlib.Path) -> Any:
    web = tmp_path / "web"
    web.mkdir()
    (web / "index.html").write_text("<!doctype html><title>t</title>")
    (web / "app.js").write_text("console.log(1)")
    app = App(REPO, tools=FakeTools(tmp_path), web_dir=web).start()
    yield app
    app.close()


def request(
    app: App,
    path: str,
    payload: Any = None,
    headers: dict[str, str] | None = None,
    raw: bytes | None = None,
) -> tuple[int, Any, str]:
    data = raw if raw is not None else None
    if payload is not None:
        data = json.dumps(payload).encode()
    sent = {"Content-Type": "application/json"} if data is not None else {}
    req = urllib.request.Request(
        app.url + path, data=data, headers={**sent, **(headers or {})}
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as response:
            body, kind = response.read(), response.headers.get_content_type()
            status = response.status
    except urllib.error.HTTPError as error:
        body, kind, status = error.read(), error.headers.get_content_type(), error.code
    return status, json.loads(body) if kind == "application/json" else body, kind


def wait_for(app: App, path: str, key: str, pending: str) -> Any:
    for _ in range(100):
        status, body, _ = request(app, path)
        if body[key] != pending:
            return body
        time.sleep(0.1)
    raise AssertionError(f"{path} stayed {pending}")


def test_static_files_and_security(app: App) -> None:
    status, body, kind = request(app, "")
    assert status == 200 and kind == "text/html" and b"doctype" in body
    assert request(app, "static/app.js")[2] == "text/javascript"
    assert request(app, "static/missing.js")[0] == 404
    assert request(app, "static/..%2Fweb%2Findex.html")[0] == 404
    # Wrong token, Host or Origin.
    wrong = urllib.request.Request(app.url.replace(app.token, "x" * 32) + "api/options")
    with pytest.raises(urllib.error.HTTPError) as error:
        urllib.request.urlopen(wrong, timeout=10)
    assert error.value.code == 404
    assert request(app, "api/options", headers={"Host": "evil.test"})[0] == 403
    status, body, _ = request(app, "api/runs", headers={"Origin": "http://evil.test"})
    assert status == 403 and "error" in body


def test_post_must_be_json_and_bounded(app: App) -> None:
    status, body, _ = request(
        app, "api/cases", raw=b"bbox=1", headers={"Content-Type": "text/plain"}
    )
    assert status == 415 and "error" in body
    assert request(app, "api/cases", raw=b"x" * (65 * 1024))[0] == 413
    assert request(app, "api/cases", raw=b"{nope")[0] == 400
    assert request(app, "api/cases", payload=[1, 2])[0] == 400
    assert request(app, "", payload={})[0] == 405
    assert request(app, "api/nothing")[0] == 404


def test_options_fields_resolve_for_each_model(app: App) -> None:
    status, options, _ = request(app, "api/options")
    assert status == 200
    assert {model["id"] for model in options["models"]} == MODEL_IDS
    assert options["max_area_km2"] > 0
    assert set(options["case_defaults"]) == {"resolution", "default_height"}
    assert [field["key"] for field in options["fields"]] == [f["key"] for f in FIELDS]
    for field in options["fields"]:
        assert field["type"] in {"number", "int", "bool", "choice"}
        assert field["label"] and field["help"]
        for model in field.get("models", MODEL_IDS):
            config = app.tools.inspect_config([f"model={model}", "params=dynamic"])
            value = lookup(config["config"], field["key"])
            if model == field.get("models", ["pylbm"])[0]:
                assert value == field["default"]
            assert isinstance(value, bool) == (field["type"] == "bool")


def test_case_lifecycle(app: App) -> None:
    status, body, _ = request(app, "api/cases", {"bbox": {**DELFT, "east": 999}})
    assert status == 400 and "Longitudes" in body["error"]
    status, body, _ = request(app, "api/cases", {"bbox": DELFT, "resolution": -1})
    assert status == 400 and "resolution" in body["error"]

    cast(
        FakeTools, app.tools
    ).release.clear()  # hold the build until "building" is seen
    status, body, _ = request(app, "api/cases", {"bbox": DELFT})
    assert status == 202
    case_id = body["case_id"]
    assert request(app, f"api/cases/{case_id}")[1]["status"] == "building"
    cast(FakeTools, app.tools).release.set()
    case = wait_for(app, f"api/cases/{case_id}", "status", "building")
    assert case["status"] == "ready", case
    assert case["summary"]["n_buildings"] == 1
    assert case["summary"]["domain"]["nx"] == 100
    (feature,) = case["buildings"]["features"]
    assert feature["properties"] == {"height": 12.0, "height_source": "height"}
    lon, lat = feature["geometry"]["coordinates"][0][0]
    assert DELFT["west"] < lon < DELFT["east"] and DELFT["south"] < lat < DELFT["north"]
    (sw, east), corner = case["sensors"], [DELFT["west"], DELFT["south"]]
    assert sw == pytest.approx(corner, abs=1e-6)
    assert east[0] > sw[0] and east[1] == pytest.approx(sw[1], abs=1e-5)
    assert request(app, "api/cases/unknown")[0] == 404

    cast(FakeTools, app.tools).fail_case = True
    case_id = request(app, "api/cases", {"bbox": DELFT})[1]["case_id"]
    case = wait_for(app, f"api/cases/{case_id}", "status", "building")
    assert case == {"case_id": case_id, "status": "failed", "error": case["error"]}
    assert "No buildings" in case["error"]

    # A case built before is ready at once, without a job.
    cast(FakeTools, app.tools).cached = True
    case_id = request(app, "api/cases", {"bbox": DELFT})[1]["case_id"]
    case = request(app, f"api/cases/{case_id}")[1]
    assert case["status"] == "ready" and case["summary"]["n_buildings"] == 1


def test_case_failures_while_preparing(app: App) -> None:
    tools = cast(FakeTools, app.tools)
    # A reused folder without its buildings: a failed case, not a 500.
    tools.cached = True
    original = tools._case
    tools._case = lambda: {**original(), "case_dir": str(tools.store / "nowhere")}  # type: ignore[method-assign]
    status, body, _ = request(app, "api/cases", {"bbox": DELFT})
    assert status == 202
    case = request(app, f"api/cases/{body['case_id']}")[1]
    assert case["status"] == "failed" and "nowhere" in case["error"]
    # The supervisor failing to start: a failed case; bad input: a 400.
    tools.prepare_error = RuntimeError("supervisor did not start")
    status, body, _ = request(app, "api/cases", {"bbox": DELFT})
    assert status == 202
    case = request(app, f"api/cases/{body['case_id']}")[1]
    assert case == {
        "case_id": body["case_id"],
        "status": "failed",
        "error": "supervisor did not start",
    }
    tools.prepare_error = ValueError("needs the city4cfd env")
    status, body, _ = request(app, "api/cases", {"bbox": DELFT})
    assert status == 400 and "city4cfd" in body["error"]


def test_run_lifecycle(app: App) -> None:
    tools = cast(FakeTools, app.tools)
    case_id = request(app, "api/cases", {"bbox": DELFT})[1]["case_id"]
    wait_for(app, f"api/cases/{case_id}", "status", "building")
    run = {"case_id": case_id, "model": "pylbm"}

    for bad, message in [
        ({**run, "model": "fluent"}, "model must be"),
        ({**run, "case_id": "nope"}, "Unknown case"),
        ({**run, "settings": {"domain.nx": 4}}, "Unknown setting"),
        ({**run, "settings": {"time.simulation_time": "x"}}, "must be a number"),
        ({**run, "settings": {"time.simulation_time": -5}}, "at least"),
        ({**run, "settings": {"forward.ensemble": 1}}, "true or false"),
        ({**run, "settings": {"ensemble.ensemble_size": 2.5}}, "whole number"),
        # Too large for a float: a 400, not an OverflowError.
        ({**run, "settings": {"ensemble.ensemble_size": 10**400}}, "at most"),
        ({**run, "settings": {"time.simulation_time": -(10**400)}}, "at least"),
        ({**run, "settings": {"model.forward_model.ncpu": 3}}, "cannot start"),
        ({**run, "extra_overrides": "a=b"}, "list of strings"),
        (
            {
                **run,
                "model": "pyudales",
                "settings": {"model.forward_model.inlet_turbulence.enabled": True},
            },
            "does not apply",
        ),
    ]:
        status, body, _ = request(app, "api/runs", bad)
        assert status in (400, 404) and message in body["error"], (bad, body)

    status, body, _ = request(
        app,
        "api/runs",
        {
            **run,
            "settings": {"time.simulation_time": 2, "forward.ensemble": False},
            "extra_overrides": ["forward.rollout_steps=1"],
        },
    )
    assert status == 200 and body == {"run_id": "run1"}
    assert tools.prepared[-1] == [
        "model=pylbm",
        "params=dynamic",
        "domain.nx=100",
        "obs.x_points=[50.0, 70.0]",
        "obs.y_points=[50.0, 50.0]",
        "time.simulation_time=2.0",
        "forward.ensemble=false",
        "forward.rollout_steps=1",
    ]
    listed = request(app, "api/runs")[1]["runs"]
    assert [(r["run_id"], r["case_id"], r["model"]) for r in listed] == [
        ("run1", case_id, "pylbm")
    ]
    assert listed[0]["state"] == "queued" and listed[0]["created"] > 0

    status, body, _ = request(app, "api/runs/run1")
    assert body["state"] == "queued" and body["phase"] == "running"
    assert body["log_tail"][-1] == "line 99" and len(body["log_tail"]) == 40
    assert "results" not in body
    assert request(app, "api/runs/run1/visualization", {})[0] == 409

    tools.jobs_state["run1"] = "succeeded"
    body = request(app, "api/runs/run1")[1]
    assert body["results"]["variables"] == ["u", "v"]
    assert body["results"]["times"]["size"] == 3
    (listed,) = request(app, "api/runs")[1]["runs"]
    assert set(listed) == {"run_id", "case_id", "model", "state", "created"}
    status, body, _ = request(app, "api/runs/run1/visualization", raw=b"")
    assert status == 200 and body == {"visualization_id": "vis1"}
    assert request(app, "api/visualizations/vis1")[1] == {
        "state": "succeeded",
        "viewer_url": "http://127.0.0.1:1/v/",
    }
    assert tools.frame_ancestor == f"http://{app.authority}"  # the page embeds it

    tools.jobs_state["run1"] = "running"
    assert request(app, "api/runs/run1/cancel", {})[1] == {
        "run_id": "run1",
        "state": "cancelling",
    }
    status, body, _ = request(app, "api/runs/zzz")
    assert status == 400 and "Unknown job" in body["error"]
