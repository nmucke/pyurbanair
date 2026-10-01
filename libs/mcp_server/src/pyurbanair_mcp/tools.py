"""Thin synchronous adapters; simulation and rendering always run in workers."""

from __future__ import annotations

import json
import os
import pathlib
import shutil
import sys
from typing import Any
from urllib.parse import quote

from pyurbanair_mcp.schemas import (
    MAX_MANIFEST_BYTES,
    MAX_METADATA_BYTES,
    MAX_PNG_BYTES,
    summary,
)

from pyurbanair.jobs.preparation import PreparationService
from pyurbanair.jobs.rendering_environment import select_render_environment
from pyurbanair.jobs.results import contained_file, inspect_results
from pyurbanair.jobs.supervisor import SupervisorClient


class Tools:
    def __init__(
        self,
        repo_root: str | pathlib.Path,
        store_root: str | pathlib.Path | None = None,
    ) -> None:
        self.repo_root = pathlib.Path(repo_root).expanduser().resolve()
        if not (self.repo_root / "conf" / "run_forward_model.yaml").is_file():
            raise ValueError("--repo-root must identify a pyurbanair checkout")
        self.store_root = (
            pathlib.Path(store_root).expanduser().resolve()
            if store_root
            else self.repo_root / ".temp" / "local-jobs"
        )
        self.preparation = PreparationService(self.repo_root, self.store_root)
        self.jobs = SupervisorClient(self.repo_root, self.store_root)

    def get_capabilities(self) -> dict[str, Any]:
        """Inspect available configurations and installed environments without building solvers."""
        from pyurbanair.config.composition import list_config_options
        from pyurbanair.jobs.native import native_coverage

        return {
            "workflow": "forward",
            "backends": ["pylbm", "pyudales", "pypalm", "neural_surrogate"],
            "modes": ["single", "ensemble", "static", "dynamic", "rollout"],
            "total_windows": "1 + run.rollout_steps",
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
        """Discover repository config groups and options with bounded pagination."""
        from pyurbanair.config.composition import list_config_options

        return list_config_options(
            self.repo_root, group=group, search=search, page=page, page_size=page_size
        )

    def inspect_config(
        self, overrides: list[str] | None = None, subtree: str | None = None
    ) -> dict[str, Any]:
        """Compose ordered Hydra overrides without importing a backend."""
        from pyurbanair.config.composition import inspect_config

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
        """Freeze a forward configuration, input fingerprints and validation before launch."""
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
            from pyurbanair.visualization import validate_options

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
        """Inspect numerical artifact metadata or at most 4096 selected values."""
        job = self.jobs.request("status", job_id=run_id)
        return inspect_results(
            job["run_root"], artifact_id, variable, selection, offset, limit
        )

    def render_simulation(
        self, run_id: str, idempotency_key: str, options: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """Queue visualization of saved numerical artifacts, without rerunning the solver."""
        from pyurbanair.visualization import validate_options

        validate_options(options or {})
        parent = self.jobs.request("status", job_id=run_id)
        if parent["state"] != "succeeded" or parent["kind"] != "forward":
            raise ValueError("Rendering requires a successful forward run")
        options = options or {}
        payload = {
            "kind": "visualization",
            "parent_run_id": run_id,
            "source_root": parent["run_root"],
            "options": options,
            "environment": select_render_environment(self.repo_root, options),
        }
        job = self.jobs.request("submit", payload=payload, key=idempotency_key)
        return {**summary(job), "visualization_id": job["id"]}

    def visualization(
        self, visualization_id: str, preview: str | None = None
    ) -> tuple[dict[str, Any], bytes | None]:
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
                "viewer", job_id=visualization_id
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
