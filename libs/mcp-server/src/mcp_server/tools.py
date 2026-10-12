"""Thin synchronous adapters; simulation and rendering always run in workers."""

from __future__ import annotations

import hashlib
import json
import os
import pathlib
import shutil
import sys
import threading
from typing import Any
from urllib.parse import quote

from mcp_server.area_selector import MAX_AREA_KM2, AreaSelector, validate_bbox
from mcp_server.jobs.preparation import PreparationService
from mcp_server.jobs.rendering_environment import select_render_environment
from mcp_server.jobs.results import contained_file, inspect_results
from mcp_server.jobs.supervisor import SupervisorClient
from mcp_server.schemas import (
    MAX_MANIFEST_BYTES,
    MAX_METADATA_BYTES,
    MAX_PNG_BYTES,
    summary,
)


class Tools:
    def __init__(
        self,
        repo_root: str | pathlib.Path,
        store_root: str | pathlib.Path | None = None,
    ) -> None:
        self.repo_root = pathlib.Path(repo_root).expanduser().resolve()
        if not (self.repo_root / "configs" / "forward.yaml").is_file():
            raise ValueError("--repo-root must identify a pyurbanair checkout")
        self.store_root = (
            pathlib.Path(store_root).expanduser().resolve()
            if store_root
            else self.repo_root / ".temp" / "local-jobs"
        )
        self.preparation = PreparationService(self.repo_root, self.store_root)
        self.jobs = SupervisorClient(self.repo_root, self.store_root)
        self.area_selector: AreaSelector | None = None
        # MCP sync tools run on worker threads: guard the selector swap and
        # give each city case one builder at a time.
        self._lock = threading.Lock()
        self._case_locks: dict[str, threading.Lock] = {}

    def get_capabilities(self) -> dict[str, Any]:
        """Inspect available configurations and installed environments without building solvers."""
        from mcp_server.jobs.composition import list_config_options
        from mcp_server.jobs.native import native_coverage

        return {
            "workflow": "forward",
            "backends": ["pylbm", "pyudales", "pypalm", "neural_surrogate"],
            "modes": ["single", "ensemble", "static", "dynamic", "rollout"],
            "total_windows": "1 + forward.rollout_steps",
            "max_active_jobs": 1,
            "python": sys.executable,
            "repo_root": str(self.repo_root),
            "store_root": str(self.store_root),
            "environments": {
                name: (
                    self.repo_root / ".pixi" / "envs" / name / "bin" / "python"
                ).exists()
                for name in ("dev", "cuda", "rendering")
            },
            "pixi": shutil.which(os.environ.get("PYURBANAIR_PIXI", "pixi")),
            "rendering": {
                "standard": True,
                "ffmpeg": shutil.which("ffmpeg"),
                "three_dimensional_environment": (
                    self.repo_root / ".pixi" / "envs" / "rendering" / "bin" / "python"
                ).exists(),
            },
            "prerequisites": self.preparation.capabilities(),
            "native_settings": {
                name: native_coverage(name)
                for name in ("pylbm", "pyudales", "pypalm", "neural_surrogate")
            },
            "visualization_presets": list_config_options(
                self.repo_root, group="visualization"
            ),
        }

    def list_config_options(
        self,
        group: str | None = None,
        search: str | None = None,
        page: int = 0,
        page_size: int = 50,
    ) -> dict[str, Any]:
        """Discover the forward config options (case, model, params) and render presets."""
        from mcp_server.jobs.composition import list_config_options

        return list_config_options(
            self.repo_root, group=group, search=search, page=page, page_size=page_size
        )

    def inspect_config(
        self, overrides: list[str] | None = None, subtree: str | None = None
    ) -> dict[str, Any]:
        """Compose configs/forward.yaml with ordered Hydra overrides, without importing a backend."""
        from mcp_server.jobs.composition import inspect_config

        return inspect_config(
            self.repo_root, overrides=overrides or [], subtree=subtree
        )

    def prepare_forward_run(
        self,
        overrides: list[str] | None = None,
        native_overrides: dict[str, Any] | None = None,
        initial_state: dict[str, Any] | None = None,
        execution_limits: dict[str, Any] | None = None,
        environment: str = "dev",
    ) -> dict[str, Any]:
        """Freeze a forward configuration (configs/forward.yaml + overrides, e.g. model=pylbm forward.ensemble=true forward.rollout_steps=2), input fingerprints and validation before launch. initial_state is a NetCDF path or {path, member, time_index}."""
        return self.preparation.prepare(
            overrides=overrides or [],
            native_overrides=native_overrides,
            initial_state=initial_state,
            execution_limits=execution_limits,
            environment=environment,
        )

    def launch_forward_run(
        self,
        plan_id: str,
        idempotency_key: str,
        post_render: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Queue a prepared run promptly; repeated keys return the same job."""
        if post_render is not None:
            from visualization import validate_options

            validate_options(post_render)
        plan = self.preparation.load(plan_id)
        payload = {
            "kind": "forward",
            "plan_id": plan_id,
            "plan_digest": plan["digest"],
            "post_render": post_render,
            "store_root": str(self.store_root),
            "environment": plan.get("environment", "dev"),
        }
        existing = self.jobs.request("existing", payload=payload, key=idempotency_key)
        if existing is not None:
            return dict(summary(existing))
        self.preparation.verify(plan_id)
        validation = plan["validation"]
        if (
            not validation["configuration_valid"]
            or not validation["prerequisites_present"]
        ):
            raise ValueError(f"Prepared run is not launchable: {validation['issues']}")
        return dict(
            summary(self.jobs.request("submit", payload=payload, key=idempotency_key))
        )

    def list_runs(
        self, state: str | None = None, offset: int = 0, limit: int = 50
    ) -> dict[str, Any]:
        """List persistent jobs across client sessions."""
        return {
            "runs": [
                summary(job)
                for job in self.jobs.request(
                    "list", state=state, offset=offset, limit=limit
                )
            ]
        }

    def get_run_status(self, run_id: str) -> dict[str, Any]:
        """Return persisted phase, heartbeat, process identity and outcome."""
        job = self.jobs.request("status", job_id=run_id)
        return {
            **summary(job),
            "created": job["created"],
            "updated": job["updated"],
            "parent_run_id": job["parent_run_id"],
        }

    def get_run_logs(
        self, run_id: str, cursor: int = 0, limit: int = 32768
    ) -> dict[str, Any]:
        """Read a bounded UTF-8 log chunk using a byte cursor."""
        return dict(
            self.jobs.request("logs", job_id=run_id, cursor=cursor, limit=limit)
        )

    def cancel_run(self, run_id: str) -> dict[str, Any]:
        """Request cooperative cancellation and teardown of all owned descendants."""
        return dict(summary(self.jobs.request("cancel", job_id=run_id)))

    def inspect_run_results(
        self,
        run_id: str,
        artifact_id: int | None = None,
        variable: str | None = None,
        selection: dict[str, Any] | None = None,
        offset: int = 0,
        limit: int = 100,
    ) -> dict[str, Any]:
        """List a run's NetCDF results (state.nc, params.nc, per-window files) or inspect at most 4096 selected values of one."""
        job = self.jobs.request("status", job_id=run_id)
        return inspect_results(
            pathlib.Path(job["run_root"]) / "artifacts",
            artifact_id,
            variable,
            selection,
            offset,
            limit,
        )

    def open_area_selector(self, max_area_km2: float = MAX_AREA_KM2) -> dict[str, Any]:
        """Open a local map page where the user draws the rectangle of a city area to simulate.

        Give the user the returned `url` to open in their browser, then call
        `get_selected_area(selection_id)` until it returns the bbox. If the
        user would rather type the area, they may paste a bbox as west, south,
        east, north in degrees (EPSG:4326); use that directly instead, without
        this page. Areas above `max_area_km2` are rejected on the page.
        """
        with self._lock:
            if self.area_selector is None or self.area_selector.closed:
                self.area_selector = AreaSelector()
            selection_id, url = self.area_selector.open(max_area_km2)
        return {
            "selection_id": selection_id,
            "url": url,
            "status": "pending",
            "max_area_km2": max_area_km2,
            "instructions": (
                f"Open {url} in a browser on this machine, click 'Draw rectangle', "
                "drag over the area, then press 'Use this area'."
            ),
        }

    def get_selected_area(
        self, selection_id: str, wait_s: float = 30
    ) -> dict[str, Any]:
        """Return the area drawn on the `open_area_selector` page.

        Blocks up to `wait_s` seconds (at most 60). Returns status "selected"
        with `bbox` {west, south, east, north, area_km2} in EPSG:4326 degrees,
        or status "pending": ask the user to finish on the page and call again.
        Status "expired" means the page server shut down or the id is unknown
        to it; open a new selector.
        """
        selector = self.area_selector
        expired = {"selection_id": selection_id, "status": "expired"}
        if selector is None:
            return expired
        try:
            selection = selector.wait(selection_id, min(max(wait_s, 0), 60))
        except ValueError:
            return expired
        result: dict[str, Any] = {"selection_id": selection_id, **selection}
        if result["status"] == "pending" and selector.closed:
            result["status"] = "expired"
        return result

    def prepare_city_case(
        self,
        bbox: dict[str, float] | list[float],
        resolution: float = 4.0,
        default_height: float = 10.0,
        max_area_km2: float = MAX_AREA_KM2,
    ) -> dict[str, Any]:
        """Turn a city area into a simulation case: OSM buildings -> City4CFD STL.

        `bbox` is {west, south, east, north} (or a 4-list) in EPSG:4326, e.g.
        the one `get_selected_area` returned. Building heights come from OSM
        `height`, else `building:levels` x 3 m, else `default_height` (m).
        Takes minutes (the OpenStreetMap download dominates). Returns
        `overrides`: pass `["model=<m>", *overrides]` to `prepare_forward_run`
        with m = pylbm, pypalm or pyudales (pyudales first runs its IBM
        preprocessing on the STL: seconds to minutes, growing with the grid
        and building count). `resolution` is the grid spacing (m); check
        `domain` before launching, the run cost grows with nx * ny * nz.
        `max_area_km2` is the area limit; pass the one the selector was
        opened with. A finished case for the same inputs is reused.
        """
        from pycity4cfd.case import CASE_FORMAT, make_case

        area = validate_bbox(bbox, max_area_km2)
        west, south, east, north = (area[k] for k in ("west", "south", "east", "north"))
        # The case format is in the key, so folders built by an older make_case
        # are never reused.
        key = ",".join(
            [f"v{CASE_FORMAT}"]
            + [
                f"{value:.5f}"
                for value in (west, south, east, north, resolution, default_height)
            ]
        )
        name = "area_" + hashlib.sha1(key.encode()).hexdigest()[:10]
        case_dir = self.repo_root / ".temp" / "city_cases" / name
        with self._lock:
            case_lock = self._case_locks.setdefault(name, threading.Lock())
        # A retry waits for a running build of the same case, then reuses it.
        with case_lock:
            case_json = case_dir / "case.json"
            if case_json.is_file():
                case: dict[str, Any] = json.loads(case_json.read_text(encoding="utf-8"))
                case.pop("bbox", None)
                case.pop("case_format", None)
            else:
                case = make_case(
                    (west, south, east, north),
                    case_dir,
                    resolution=resolution,
                    default_height=default_height,
                )
        return case

    def render_simulation(
        self, run_id: str, idempotency_key: str, options: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """Queue visualization of saved numerical artifacts, without rerunning the solver."""
        from visualization import validate_options

        validate_options(options or {})
        parent = self.jobs.request("status", job_id=run_id)
        if parent["state"] != "succeeded" or parent["kind"] != "forward":
            raise ValueError("Rendering requires a successful forward run")
        options = options or {}
        payload = {
            "kind": "visualization",
            "parent_run_id": run_id,
            "source_root": str(pathlib.Path(parent["run_root"]) / "artifacts"),
            "options": options,
            "environment": select_render_environment(self.repo_root, options),
        }
        job = self.jobs.request("submit", payload=payload, key=idempotency_key)
        return {**summary(job), "visualization_id": job["id"]}

    def visualization(
        self,
        visualization_id: str,
        preview: str | None = None,
        frame_ancestor: str | None = None,
    ) -> tuple[dict[str, Any], bytes | None]:
        """Metadata, a viewer URL and an optional preview PNG of a render.

        `frame_ancestor` is the origin allowed to embed the viewer in a frame
        (the browser app's own); by default none is.
        """
        job = self.jobs.request("status", job_id=visualization_id)
        if job["kind"] != "visualization":
            raise ValueError("ID does not identify a visualization")
        metadata: dict[str, Any] = dict(summary(job))
        if job["state"] != "succeeded":
            return metadata, None
        root = pathlib.Path(job["run_root"]) / "bundle"
        manifest_path = contained_file(root, "viewer_manifest.json")
        with manifest_path.open("rb") as stream:
            manifest_bytes = stream.read(MAX_MANIFEST_BYTES + 1)
        if len(manifest_bytes) > MAX_MANIFEST_BYTES:
            raise ValueError(
                "Viewer manifest exceeds 4 MiB; inspect its local artifact"
            )
        manifest = json.loads(manifest_bytes)
        if manifest.get("version") != 1 or manifest.get("status") != "complete":
            raise ValueError("Visualization bundle is not a completed version-1 bundle")
        # The HTML viewer consumes the complete manifest; tool output is bounded.
        transported_manifest = dict(manifest)
        sources = manifest.get("sources", [])
        if len(sources) > 100:
            transported_manifest["sources"] = sources[:100]
            transported_manifest["sources_total"] = len(sources)
            transported_manifest["sources_truncated"] = True
        metadata["manifest"] = transported_manifest
        try:
            metadata["viewer_url"] = self.jobs.request(
                "viewer", job_id=visualization_id, frame_ancestor=frame_ancestor
            )["url"]
        except (OSError, RuntimeError, ValueError) as exc:
            # PNG delivery works even where the local browser server cannot bind.
            metadata["viewer_warning"] = (
                f"Browser serving unavailable: {str(exc)[:500]}"
            )
        previews = sorted(
            str(path.relative_to(root)) for path in (root / "previews").glob("*.png")
        )[:32]
        metadata["previews"] = previews
        artifacts = []
        if metadata.get("viewer_url"):
            listed = [
                ("viewer_manifest.json", "application/json"),
                ("probes.csv", "text/csv"),
            ]
            for view in manifest.get("views", [])[:7]:
                if view.get("poster"):
                    listed.append((view["poster"], "image/png"))
                if view.get("media"):
                    listed.append((view["media"], "video/mp4"))
            for relative, mime_type in listed:
                try:
                    asset = contained_file(root, relative)
                except ValueError:
                    continue
                artifacts.append(
                    {
                        "name": asset.name,
                        "uri": metadata["viewer_url"] + quote(relative, safe="/"),
                        "mime_type": mime_type,
                        "size": asset.stat().st_size,
                    }
                )
        metadata["artifacts"] = artifacts
        if len(json.dumps(metadata).encode()) > MAX_METADATA_BYTES:
            # Avoid transporting an unbounded timeline/provenance table. The full
            # file remains available through its artifact resource link.
            metadata["manifest"] = {
                "version": manifest.get("version"),
                "status": manifest.get("status"),
                "view_count": len(manifest.get("views", [])),
            }
            metadata["details"] = {}
            metadata["metadata_warning"] = (
                "Manifest metadata omitted to stay within 512 KiB; use the manifest artifact link"
            )
        if not previews:
            return metadata, None
        first_view: dict[str, Any] = next(iter(manifest.get("views", [])), {})
        preferred = first_view.get("poster")
        selected = preview or (preferred if preferred in previews else previews[0])
        if selected not in previews:
            raise ValueError("Select one of the available PNG previews")
        path = contained_file(root, selected)
        with path.open("rb") as stream:
            png = stream.read(MAX_PNG_BYTES + 1)
        if len(png) > MAX_PNG_BYTES:
            metadata["image_warning"] = (
                "Preview exceeds 2 MiB; request a smaller render"
            )
            return metadata, None
        if not png.startswith(b"\x89PNG\r\n\x1a\n"):
            metadata["image_warning"] = "Selected preview is not a PNG image"
            return metadata, None
        metadata["selected_preview"] = selected
        return metadata, png
