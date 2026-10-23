"""Loopback HTTP server: the web/ page plus a JSON API over `mcp_server.tools.Tools`.

Mirrors `user_interface/area_selector.py`: a stdlib ThreadingHTTPServer on
127.0.0.1, Host/Origin checks, an unguessable token as the first URL path
segment and JSON-only POSTs with a size cap. Every action is a `Tools` call
(case builds and runs are supervisor jobs); this module only validates the
requests and keeps the cases, runs and renders of this session in memory. The
routes are listed in docs/user_interface.md.
"""

from __future__ import annotations

import inspect
import json
import math
import pathlib
import re
import secrets
import threading
import time
import traceback
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Callable
from urllib.parse import urlsplit

from mcp_server.tools import Tools
from user_interface.area_selector import MAX_AREA_KM2, validate_bbox
from visualization import RenderOptions

WEB_DIR = pathlib.Path(__file__).with_name("web")
MAX_BODY_BYTES = 64 * 1024
LOG_TAIL_BYTES = 16 * 1024
LOG_TAIL_LINES = 40
# Always composed under the run's overrides: the inflow fields below live in
# the dynamic sampler, whatever configs/forward.yaml selects by default.
BASE_OVERRIDES = ["params=dynamic"]
CONTENT_TYPES = {
    ".html": "text/html; charset=utf-8",
    ".js": "text/javascript; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".json": "application/json",
    ".svg": "image/svg+xml",
    ".png": "image/png",
    ".ico": "image/x-icon",
}
FINISHED = {"succeeded", "failed", "cancelled", "interrupted"}

MODELS = [
    {
        "id": "pylbm",
        "label": "Lattice Boltzmann (LBM)",
        "description": "Fastest: a lattice Boltzmann LES, compiled on the first "
        "run; the quickest look at the flow.",
    },
    {
        "id": "pypalm",
        "label": "PALM",
        "description": "Established atmospheric LES; slower, and builds the "
        "solver on first use (minutes).",
    },
    {
        "id": "pyudales",
        "label": "uDALES",
        "description": "Urban LES with immersed boundaries and inlet "
        "turbulence; the most detailed and the slowest to set up.",
    },
]
MODEL_IDS = {model["id"] for model in MODELS}

# Curated run settings. `key` is the Hydra override path; the default is read
# from the composed forward config of the first model in `models` (all models
# when absent), unless the field sets one. A choice is a plain value or
# {value, label}; the value None sends no override.
FIELDS: list[dict[str, Any]] = [
    {
        "key": "time.simulation_time",
        "label": "Run time",
        "type": "number",
        "min": 1,
        "max": 7200,
        "step": 1,
        "unit": "s",
        "help": "Simulated time after the spin-up.",
    },
    {
        "key": "time.spinup_time",
        "label": "Spin-up",
        "type": "number",
        "min": 0,
        "max": 3600,
        "step": 1,
        "unit": "s",
        "help": "Constant-inflow time before the run, not saved.",
    },
    {
        "key": "time.output_frequency",
        "label": "Output every",
        "type": "number",
        "min": 0.1,
        "max": 600,
        "step": 0.1,
        "unit": "s",
        "help": "Interval between saved flow fields.",
    },
    {
        "key": "params.external_parameters.velocity_magnitude.mean",
        "label": "Wind speed",
        "type": "number",
        "min": 0.1,
        "max": 30,
        "step": 0.1,
        "unit": "m/s",
        "help": "Mean inflow speed.",
    },
    {
        "key": "params.external_parameters.velocity_magnitude.std",
        "label": "Wind speed variability",
        "type": "number",
        "min": 0,
        "max": 10,
        "step": 0.1,
        "unit": "m/s",
        "help": "Standard deviation of the inflow speed over time and "
        "members; 0 keeps it steady.",
    },
    {
        "key": "params.external_parameters.inflow_angle.mean",
        "label": "Wind direction",
        "type": "number",
        "min": -180,
        "max": 180,
        "step": 1,
        "unit": "°",
        "help": "Direction the wind blows towards, counter-clockwise from "
        "east (0 = west-to-east flow, 90 = south-to-north).",
    },
    {
        "key": "params.external_parameters.inflow_angle.std",
        "label": "Wind direction variability",
        "type": "number",
        "min": 0,
        "max": 90,
        "step": 1,
        "unit": "°",
        "help": "Standard deviation of the direction over time and members; "
        "0 keeps it steady.",
    },
    {
        "key": "forward.ensemble",
        "label": "Ensemble",
        "type": "bool",
        "help": "Run several members with different inflow draws instead of one.",
    },
    {
        "key": "ensemble.ensemble_size",
        "label": "Members",
        "type": "int",
        "min": 2,
        "max": 256,
        "step": 1,
        "help": "Number of ensemble members (only with Ensemble on).",
    },
    {
        "key": "ensemble.num_parallel_processes",
        "label": "Members in parallel",
        "type": "int",
        "min": 1,
        "max": 64,
        "step": 1,
        "help": "Members run at once; memory bandwidth limits the gain past 4-8.",
    },
    {
        "key": "model.forward_model.ncpu",
        "label": "CPU cores per run",
        "type": "int",
        "min": 1,
        "max": 64,
        "step": 1,
        "help": "OpenMP threads (LBM) or MPI ranks (PALM, uDALES) per member.",
    },
    {
        "key": "model.forward_model.inlet_turbulence.enabled",
        "label": "Inlet turbulence",
        "type": "choice",
        # None keeps each model's own config value: on in uDALES and PALM (the
        # shared `flow.inlet_turbulence`), off in LBM.
        "default": None,
        "choices": [
            {"value": None, "label": "Model default"},
            {"value": True, "label": "On"},
            {"value": False, "label": "Off"},
        ],
        "help": "Perturb the inflow to trigger turbulence. Needs inflow / outflow.",
    },
    {
        "key": "model.forward_model.boundary_condition",
        "label": "Boundary condition",
        "type": "choice",
        # None keeps the config value, `flow.boundary_condition` (the case's).
        "default": None,
        "choices": [
            {"value": None, "label": "Model default"},
            {"value": "inflow_outflow", "label": "Inflow / outflow"},
            {"value": "periodic", "label": "Periodic"},
        ],
        "help": "Inflow at the upwind side and outflow at the other, or "
        "periodic in x. Inlet turbulence needs inflow / outflow.",
    },
]
FIELDS_BY_KEY = {field["key"]: field for field in FIELDS}
_CASE_PARAMETERS = inspect.signature(Tools.prepare_city_case).parameters
CASE_DEFAULTS = {
    key: _CASE_PARAMETERS[key].default for key in ("resolution", "default_height")
}


class RequestError(Exception):
    """An error answered as `{"error": message}` with an HTTP status."""

    def __init__(self, status: int, message: str) -> None:
        super().__init__(message)
        self.status = status


def lookup(config: Any, key: str) -> Any:
    """The value at dotted `key` in a composed config; KeyError if absent."""
    for segment in key.split("."):
        if not isinstance(config, dict) or segment not in config:
            raise KeyError(key)
        config = config[segment]
    return config


def field_override(key: str, value: Any) -> str | None:
    """Validate one settings entry and return its Hydra override, if any."""
    field = FIELDS_BY_KEY.get(key)
    if field is None:
        raise RequestError(400, f"Unknown setting: {key}")
    kind = field["type"]
    if kind == "bool":
        if not isinstance(value, bool):
            raise RequestError(400, f"{field['label']} must be true or false")
        return f"{key}={'true' if value else 'false'}"
    if kind == "choice":
        values = [c["value"] if isinstance(c, dict) else c for c in field["choices"]]
        # By type too: 1 == True, but 1 is not an inlet-turbulence choice.
        if not any(type(value) is type(v) and value == v for v in values):
            raise RequestError(400, f"{field['label']} must be one of {values}")
        if value is None:
            return None
        if isinstance(value, bool):
            value = "true" if value else "false"
        return f"{key}={value}"
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RequestError(400, f"{field['label']} must be a number")
    # The range first: a huge JSON integer overflows the float conversions.
    if "min" in field and value < field["min"]:
        raise RequestError(400, f"{field['label']} must be at least {field['min']}")
    if "max" in field and value > field["max"]:
        raise RequestError(400, f"{field['label']} must be at most {field['max']}")
    if not math.isfinite(value):
        raise RequestError(400, f"{field['label']} must be finite")
    if kind == "int" and not float(value).is_integer():
        raise RequestError(400, f"{field['label']} must be a whole number")
    return f"{key}={int(value) if kind == 'int' else float(value)!r}"


def _positive(payload: dict[str, Any], key: str, default: float, top: float) -> float:
    value = payload.get(key, default)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RequestError(400, f"{key} must be a number")
    if not (0 < value <= top and math.isfinite(value)):
        raise RequestError(400, f"{key} must lie in (0, {top:g}]")
    return float(value)


def case_geometry(case_dir: pathlib.Path) -> dict[str, Any]:
    """Buildings as EPSG:4326 GeoJSON and the sensors as [lon, lat].

    The sensors are in the domain frame of `make_case`; case.json records its
    CRS and `origin`, the projected coordinates of the domain's (0, 0) corner.
    """
    import geopandas as gpd

    buildings = gpd.read_file(case_dir / "buildings.geojson")
    shown = buildings[["height", "height_source", "geometry"]].to_crs("EPSG:4326")
    case = json.loads((case_dir / "case.json").read_text(encoding="utf-8"))
    x0, y0 = case["origin"]
    points: dict[str, list[float]] = {}
    for override in case["overrides"]:
        match = re.fullmatch(r"obs\.([xy])_points=\[(.*)\]", override)
        if match:
            points[match[1]] = [float(v) for v in match[2].split(",") if v.strip()]
    sensors = gpd.points_from_xy(
        [x + x0 for x in points.get("x", [])],
        [y + y0 for y in points.get("y", [])],
        crs=case["crs"],
    ).to_crs("EPSG:4326")
    return {
        "buildings": json.loads(shown.to_json(drop_id=True)),
        "sensors": [[round(p.x, 7), round(p.y, 7)] for p in sensors],
    }


class App:
    """The server and this session's cases, runs and renders.

    `tools` defaults to `Tools(repo_root)`; tests pass a fake. `web_dir`
    holds index.html and the static files.
    """

    def __init__(
        self,
        repo_root: str | pathlib.Path = ".",
        port: int = 0,
        tools: Tools | None = None,
        web_dir: pathlib.Path = WEB_DIR,
    ) -> None:
        self.tools = tools if tools is not None else Tools(repo_root)
        self.web_dir = pathlib.Path(web_dir)
        self.token = secrets.token_urlsafe(24)
        self.cases: dict[str, dict[str, Any]] = {}
        self.runs: dict[str, dict[str, Any]] = {}
        self.renders: dict[str, str] = {}  # run_id -> visualization_id
        self._lock = threading.Lock()
        self._options: dict[str, Any] | None = None
        self.server = ThreadingHTTPServer(("127.0.0.1", port), self._handler())
        self.server.daemon_threads = True
        self.authority = f"127.0.0.1:{self.server.server_port}"
        self.url = f"http://{self.authority}/{self.token}/"

    # Routes -----------------------------------------------------------------

    def options(self) -> dict[str, Any]:
        """Models, the fields with their defaults, and the case defaults."""
        with self._lock:
            if self._options is not None:
                return self._options
        configs = {
            model: self.tools.inspect_config([f"model={model}", *BASE_OVERRIDES])[
                "config"
            ]
            for model in MODEL_IDS
        }
        fields = []
        for field in FIELDS:
            first = field.get("models", [MODELS[0]["id"]])[0]
            default = (
                field["default"]
                if "default" in field
                else lookup(configs[first], field["key"])
            )
            fields.append({**field, "default": default})
        options = {
            "models": MODELS,
            "fields": fields,
            "case_defaults": CASE_DEFAULTS,
            "max_area_km2": MAX_AREA_KM2,
        }
        with self._lock:
            self._options = options
        return options

    def create_case(self, payload: dict[str, Any]) -> tuple[int, dict[str, Any]]:
        try:
            bbox = validate_bbox(payload.get("bbox"))
        except ValueError as exc:
            raise RequestError(400, str(exc)) from exc
        resolution = _positive(payload, "resolution", CASE_DEFAULTS["resolution"], 50.0)
        default_height = _positive(
            payload, "default_height", CASE_DEFAULTS["default_height"], 500.0
        )
        # A finished case is ready at once; a new one builds as a job. Bad
        # input (ValueError) is a 400; any other failure fails the case.
        try:
            built = self.tools.prepare_city_case(
                bbox, resolution=resolution, default_height=default_height
            )
        except ValueError:
            raise
        except Exception as exc:  # reported to the page, not raised
            traceback.print_exc()
            built = {"status": "failed", "error": str(exc) or repr(exc)}
        case_id = uuid.uuid4().hex[:12]
        record: dict[str, Any] = {"case_id": case_id, "status": "building"}
        record.update(self._settle(built))
        with self._lock:
            self.cases[case_id] = record
        return 202, {"case_id": case_id}

    @classmethod
    def _settle(cls, built: dict[str, Any]) -> dict[str, Any]:
        """The case record update for a `prepare_city_case`/`get_city_case` result."""
        if built["status"] == "ready":
            try:
                return cls._ready(built)
            except Exception as exc:  # reported to the page, not raised
                traceback.print_exc()
                return {"status": "failed", "error": str(exc)[:2000] or repr(exc)}
        if built["status"] == "failed":
            return {"status": "failed", "error": built["error"][:2000]}
        return {"case_job_id": built["case_job_id"]}

    @staticmethod
    def _ready(case: dict[str, Any]) -> dict[str, Any]:
        """A finished case as this app keeps it: summary, overrides, geometry."""
        return {
            "status": "ready",
            "summary": {
                key: case[key]
                for key in ("n_buildings", "height_sources", "max_height", "domain")
            },
            "overrides": list(case["overrides"]),
            **case_geometry(pathlib.Path(case["case_dir"])),
        }

    def case(self, case_id: str) -> dict[str, Any]:
        with self._lock:
            record = self.cases.get(case_id)
            if record is None:
                raise RequestError(404, f"Unknown case: {case_id}")
            job_id = (
                record.get("case_job_id") if record["status"] == "building" else None
            )
        if job_id is not None:
            update = self._settle(self.tools.get_city_case(job_id))
            with self._lock:
                record.update(update)
        with self._lock:
            hidden = ("overrides", "case_job_id")
            return {k: v for k, v in record.items() if k not in hidden}

    def create_run(self, payload: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            case = self.cases.get(str(payload.get("case_id")))
        if case is None:
            raise RequestError(404, "Unknown case; build the area first")
        if case["status"] != "ready":
            raise RequestError(409, "The case is not ready yet")
        model = payload.get("model")
        if model not in MODEL_IDS:
            raise RequestError(400, f"model must be one of {sorted(MODEL_IDS)}")
        settings = payload.get("settings") or {}
        extra = payload.get("extra_overrides") or []
        if not isinstance(settings, dict):
            raise RequestError(400, "settings must be an object")
        if not (isinstance(extra, list) and all(isinstance(e, str) for e in extra)):
            raise RequestError(400, "extra_overrides must be a list of strings")
        for key in settings:
            models = FIELDS_BY_KEY.get(key, {}).get("models")
            if models is not None and model not in models:
                raise RequestError(400, f"{key} does not apply to {model}")
        overrides = [
            f"model={model}",
            *BASE_OVERRIDES,
            *case["overrides"],
            *(
                override
                for key, value in settings.items()
                if (override := field_override(key, value)) is not None
            ),
            *extra,
        ]
        plan = self.tools.prepare_forward_run(overrides)
        validation = plan["validation"]
        if not (
            validation["configuration_valid"] and validation["prerequisites_present"]
        ):
            issues = validation["issues"]
            raise RequestError(
                400,
                "The run cannot start: "
                + "; ".join(f"{i['field']}: {i['message']}" for i in issues),
            )
        launched = self.tools.launch_forward_run(plan["plan_id"], uuid.uuid4().hex)
        run = {
            "run_id": launched["run_id"],
            "case_id": case["case_id"],
            "model": model,
            "state": launched["state"],
            "created": time.time(),
        }
        with self._lock:
            self.runs[run["run_id"]] = run
        return {"run_id": run["run_id"]}

    def list_runs(self) -> dict[str, Any]:
        with self._lock:
            runs = [
                {k: v for k, v in run.items() if k != "results"}
                for run in self.runs.values()
            ]
        for run in runs:
            if run["state"] not in FINISHED:
                run["state"] = self._status(run["run_id"])["state"]
        return {"runs": sorted(runs, key=lambda run: -run["created"])}

    def _status(self, run_id: str) -> dict[str, Any]:
        status = dict(self.tools.get_run_status(run_id))
        with self._lock:
            if run_id in self.runs:
                self.runs[run_id]["state"] = status["state"]
        return status

    def run(self, run_id: str) -> dict[str, Any]:
        status = self._status(run_id)
        details = status.get("details") or {}
        result: dict[str, Any] = {"run_id": run_id, "state": status["state"]}
        if details.get("phase"):
            result["phase"] = details["phase"]
        if details.get("error"):
            result["message"] = str(details["error"])
        log = pathlib.Path(status["run_root"]) / "worker.log"
        size = log.stat().st_size if log.is_file() else 0
        cursor = max(0, size - LOG_TAIL_BYTES)
        text = self.tools.get_run_logs(run_id, cursor=cursor, limit=LOG_TAIL_BYTES)[
            "text"
        ]
        lines = text.splitlines()
        if cursor and lines:
            lines = lines[1:]  # the first line is cut
        result["log_tail"] = lines[-LOG_TAIL_LINES:]
        if status["state"] == "succeeded":
            result["results"] = self._results(run_id)
        return result

    def _results(self, run_id: str) -> dict[str, Any]:
        with self._lock:
            cached = self.runs.get(run_id, {}).get("results")
        if cached is not None:
            return dict(cached)
        listing = self.tools.inspect_run_results(run_id)
        artifacts = listing.get("artifacts", [])
        results: dict[str, Any] = {"files": [a["path"] for a in artifacts]}
        state = next((a for a in artifacts if a["path"] == "state.nc"), None)
        if state is not None:
            inspected = self.tools.inspect_run_results(run_id, state["artifact_id"])
            results["variables"] = sorted(inspected["variables"])
            results["dimensions"] = inspected["dimensions"]
            results["times"] = inspected["coordinates"].get("time")
        with self._lock:
            if run_id in self.runs:
                self.runs[run_id]["results"] = results
        return results

    def cancel(self, run_id: str) -> dict[str, Any]:
        cancelled = self.tools.cancel_run(run_id)
        with self._lock:
            if run_id in self.runs:
                self.runs[run_id]["state"] = cancelled["state"]
        return {"run_id": run_id, "state": cancelled["state"]}

    def render(self, run_id: str) -> dict[str, Any]:
        with self._lock:
            existing = self.renders.get(run_id)
        if existing is not None:
            state = self.tools.get_run_status(existing)["state"]
            if state not in FINISHED - {"succeeded"}:
                return {"visualization_id": existing}
        if self._status(run_id)["state"] != "succeeded":
            raise RequestError(409, "The run has not succeeded (yet)")
        # The smallest stride that fits every saved time in the frame budget.
        frames = self._results(run_id).get("dimensions", {}).get("time", 1)
        options = {"stride": max(1, math.ceil(frames / RenderOptions.max_frames))}
        job = self.tools.render_simulation(run_id, uuid.uuid4().hex, options)
        with self._lock:
            self.renders[run_id] = job["visualization_id"]
        return {"visualization_id": job["visualization_id"]}

    def visualization(self, visualization_id: str) -> dict[str, Any]:
        # The page embeds the viewer: allow this app's origin to frame it.
        metadata, _ = self.tools.visualization(
            visualization_id, frame_ancestor=f"http://{self.authority}"
        )
        result = {"state": metadata["state"]}
        if metadata.get("viewer_url"):
            result["viewer_url"] = metadata["viewer_url"]
        message = (metadata.get("details") or {}).get("error") or metadata.get(
            "viewer_warning"
        )
        if message:
            result["message"] = str(message)
        return result

    def route(
        self, method: str, parts: list[str], payload: dict[str, Any]
    ) -> tuple[int, dict[str, Any]]:
        """Dispatch `api/...` (`parts` without the token and "api")."""
        routes: dict[tuple[str, ...], Callable[..., Any]] = {
            ("GET", "options"): self.options,
            ("POST", "cases"): lambda: self.create_case(payload),
            ("GET", "cases", "*"): self.case,
            ("POST", "runs"): lambda: self.create_run(payload),
            ("GET", "runs"): self.list_runs,
            ("GET", "runs", "*"): self.run,
            ("POST", "runs", "*", "cancel"): self.cancel,
            ("POST", "runs", "*", "visualization"): self.render,
            ("GET", "visualizations", "*"): self.visualization,
        }
        for pattern, handler in routes.items():
            if pattern[0] == method and len(pattern) == len(parts) + 1:
                ids = []
                for expected, part in zip(pattern[1:], parts):
                    if expected == "*" and re.fullmatch(r"[\w-]{1,128}", part):
                        ids.append(part)
                    elif expected != part:
                        break
                else:
                    result = handler(*ids)
                    return result if isinstance(result, tuple) else (200, result)
        raise RequestError(404, "Unknown API route")

    # Server -----------------------------------------------------------------

    def _handler(self) -> type[BaseHTTPRequestHandler]:
        app = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, format: str, *args: Any) -> None:
                pass

            def setup(self) -> None:
                super().setup()
                self.connection.settimeout(30)

            def _send(self, status: int, content_type: str, body: bytes) -> None:
                self.send_response(status)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.send_header("X-Content-Type-Options", "nosniff")
                # Map tiles and geocoding need a Referer; send the origin only,
                # never the token path.
                self.send_header("Referrer-Policy", "strict-origin-when-cross-origin")
                self.end_headers()
                self.wfile.write(body)

            def _json(self, status: int, payload: dict[str, Any]) -> None:
                body = json.dumps(payload, allow_nan=False).encode()
                self._send(status, "application/json", body)

            def _parts(self) -> list[str] | None:
                """The path after the token, after Host/Origin checks."""
                if self.headers.get("Host") != app.authority:
                    self._json(403, {"error": "Invalid Host"})
                    return None
                origin = self.headers.get("Origin")
                if origin is not None and origin != f"http://{app.authority}":
                    self._json(403, {"error": "Cross-origin access is not allowed"})
                    return None
                parts = urlsplit(self.path).path.split("/")
                if len(parts) < 3 or not secrets.compare_digest(parts[1], app.token):
                    self._json(404, {"error": "Not found"})
                    return None
                return parts[2:]

            def _static(self, parts: list[str]) -> None:
                if parts == [""]:
                    name = "index.html"
                elif len(parts) == 2 and parts[0] == "static":
                    name = parts[1]
                else:
                    self._json(404, {"error": "Not found"})
                    return
                path = app.web_dir / name
                content_type = CONTENT_TYPES.get(path.suffix)
                if (
                    content_type is None
                    or name.startswith(".")
                    or "/" in name
                    or "\\" in name
                    or not path.is_file()
                ):
                    self._json(404, {"error": "Not found"})
                    return
                self._send(200, content_type, path.read_bytes())

            def _api(self, method: str, parts: list[str]) -> None:
                try:
                    payload: dict[str, Any] = {}
                    if method == "POST":
                        payload = self._body()
                    status, result = app.route(method, parts, payload)
                except RequestError as exc:
                    status, result = exc.status, {"error": str(exc)}
                except ValueError as exc:
                    # Tools reports bad input (unknown run, invalid override) so.
                    status, result = 400, {"error": str(exc)[:2000]}
                except Exception as exc:
                    traceback.print_exc()
                    status, result = 500, {"error": str(exc)[:2000] or repr(exc)}
                self._json(status, result)

            def _body(self) -> dict[str, Any]:
                try:
                    length = int(self.headers.get("Content-Length", "0"))
                except ValueError as exc:
                    raise RequestError(400, "Bad Content-Length") from exc
                if length == 0:
                    return {}
                if self.headers.get_content_type() != "application/json":
                    raise RequestError(415, "Send the request body as JSON")
                if not 0 < length <= MAX_BODY_BYTES:
                    raise RequestError(413, "Request body must be at most 64 KiB")
                try:
                    payload = json.loads(self.rfile.read(length))
                except (ValueError, UnicodeDecodeError) as exc:
                    raise RequestError(400, "Bad JSON") from exc
                if not isinstance(payload, dict):
                    raise RequestError(400, "Send a JSON object")
                return payload

            def do_GET(self) -> None:
                parts = self._parts()
                if parts is None:
                    return
                if parts[0] == "api":
                    self._api("GET", parts[1:])
                else:
                    self._static(parts)

            def do_POST(self) -> None:
                parts = self._parts()
                if parts is None:
                    return
                if parts[0] != "api":
                    self._json(405, {"error": "Only the API accepts POST"})
                    return
                self._api("POST", parts[1:])

        return Handler

    def serve_forever(self) -> None:
        self.server.serve_forever()

    def start(self) -> App:
        """Serve on a daemon thread (tests)."""
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        return self

    def close(self) -> None:
        self.server.shutdown()
        self.server.server_close()
