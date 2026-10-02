"""Optional SDK tests, including real stdio framing and entry-point packaging."""

from __future__ import annotations

import asyncio
import importlib.metadata
import json
import os
import pathlib
import selectors
import shutil
import subprocess
import sys
from typing import Any

import pytest

pytest.importorskip("mcp")
pytest.importorskip("pyurbanair_mcp")

from mcp import Client
from pyurbanair_mcp.server import create_server

pytestmark = pytest.mark.skip(
    reason="MCP port pending: docs/plans/mcp_server_refactor_handover.md"
)

REPO = pathlib.Path(__file__).resolve().parents[2]


def test_packaging_and_backend_free_discovery(tmp_path: pathlib.Path) -> None:
    distribution = importlib.metadata.distribution("pyurbanair-mcp")
    assert any(entry.name == "pyurbanair-mcp" for entry in distribution.entry_points)
    code = """
import json, sys
from pyurbanair_mcp.tools import Tools
capabilities = Tools(sys.argv[1], sys.argv[2]).get_capabilities()
assert not {'jax','torch','pylbm','pypalm','pyudales','neural_surrogates'} & sys.modules.keys()
print(json.dumps(capabilities))
"""
    process = subprocess.run(
        [sys.executable, "-c", code, str(REPO), str(tmp_path)],
        capture_output=True,
        text=True,
        check=True,
    )
    assert len(json.loads(process.stdout)["backends"]) == 4


def test_tools_via_sdk(tmp_path: pathlib.Path) -> None:
    async def exercise() -> None:
        server = create_server(REPO, tmp_path)
        async with Client(server, raise_exceptions=True) as client:
            tools = await client.list_tools()
            names = {tool.name for tool in tools.tools}
            assert {
                "launch_forward_run",
                "get_visualization",
                "render_simulation",
                "prepare_forward_run",
            } <= names
            result = await client.call_tool("list_config_options", {"group": "model"})
            assert not result.is_error
            assert result.structured_content["total"] == 4
            invalid = await client.call_tool(
                "inspect_config",
                {"overrides": ["++model.forward_model._target_=os.system"]},
            )
            assert invalid.is_error

    asyncio.run(exercise())


@pytest.mark.parametrize("entrypoint", ["module", "console", "launcher"])  # type: ignore[misc]
def test_real_stdio_handshake_no_stdout_contamination(
    tmp_path: pathlib.Path, entrypoint: str
) -> None:
    command = {
        "module": [sys.executable, "-m", "pyurbanair_mcp"],
        "console": [str(pathlib.Path(sys.executable).with_name("pyurbanair-mcp"))],
        "launcher": [str(REPO / "scripts/start_mcp")],
    }[entrypoint]
    environment = dict(os.environ)
    if entrypoint == "launcher":
        pixi = shutil.which("pixi")
        assert pixi is not None
        environment.update(PATH=os.defpath, PYURBANAIR_PIXI=pixi)
    process = subprocess.Popen(
        [
            *command,
            "--repo-root",
            str(REPO),
            "--store-root",
            str(tmp_path),
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
        env=environment,
    )
    assert process.stdin is not None and process.stdout is not None

    def send(message: dict[str, Any]) -> None:
        assert process.stdin is not None
        process.stdin.write(json.dumps(message) + "\n")
        process.stdin.flush()

    def receive() -> dict[str, Any]:
        assert process.stdout is not None
        with selectors.DefaultSelector() as selector:
            selector.register(process.stdout, selectors.EVENT_READ)
            if not selector.select(timeout=15):
                process.terminate()
                _, diagnostic = process.communicate(timeout=5)
                raise AssertionError(f"MCP server did not respond: {diagnostic}")
        line = process.stdout.readline()
        assert line, "MCP server closed its protocol stream"
        return dict(json.loads(line))  # Any startup/log chatter fails here.

    try:
        send(
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {
                    "protocolVersion": "2025-11-25",
                    "capabilities": {},
                    "clientInfo": {"name": "pyurbanair-test", "version": "1"},
                },
            }
        )
        result = receive()
        assert result["id"] == 1 and "result" in result
        send({"jsonrpc": "2.0", "method": "notifications/initialized"})
        send(
            {
                "jsonrpc": "2.0",
                "id": 2,
                "method": "tools/call",
                "params": {"name": "get_capabilities", "arguments": {}},
            }
        )
        result = receive()
        assert result["id"] == 2
        assert not result["result"].get("isError", False)
        assert "pylbm" in json.dumps(result)
    finally:
        process.terminate()
        process.communicate(timeout=10)


def test_visualization_sdk_image_links_and_bounded_fallback(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Exercise SDK serialization through the actual artifact-reading adapter."""
    import base64

    from mcp.types import ImageContent, ResourceLink
    from pyurbanair_mcp.schemas import MAX_METADATA_BYTES, MAX_PNG_BYTES

    from pyurbanair.jobs.supervisor import SupervisorClient

    run_root = tmp_path / "render"
    bundle = run_root / "bundle"
    (bundle / "previews").mkdir(parents=True)
    # A complete 1x1 PNG fixture, not just the signature or a path string.
    png = base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+aXioAAAAASUVORK5CYII="
    )
    (bundle / "previews" / "slice-0.png").write_bytes(png)
    (bundle / "previews" / "probes.png").write_bytes(png + b"probe-plot")
    (bundle / "probes.csv").write_text("simulation_time_seconds\n0\n")
    (bundle / "movie.mp4").write_bytes(b"movie resource must never be embedded")
    manifest = {
        "version": 1,
        "status": "complete",
        "sources": [],
        "views": [{"poster": "previews/slice-0.png", "media": "movie.mp4"}],
    }
    manifest_path = bundle / "viewer_manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    job = {
        "id": "render-1",
        "kind": "visualization",
        "state": "succeeded",
        "run_root": str(run_root),
        "details": {},
    }
    fail_viewer = False

    def request(self: SupervisorClient, operation: str, **kwargs: Any) -> Any:
        assert kwargs["job_id"] == "render-1"
        if operation == "status":
            return job
        assert operation == "viewer"
        if fail_viewer:
            raise RuntimeError("loopback socket unavailable")
        return {"url": "http://127.0.0.1:8123/view/opaque-token/"}

    monkeypatch.setattr(SupervisorClient, "request", request)

    async def exercise() -> None:
        nonlocal fail_viewer
        async with Client(
            create_server(REPO, tmp_path / "store"), raise_exceptions=True
        ) as client:
            result = await client.call_tool(
                "get_visualization", {"visualization_id": "render-1"}
            )
            assert not result.is_error
            images = [item for item in result.content if isinstance(item, ImageContent)]
            assert len(images) == 1
            assert images[0].mime_type == "image/png"
            assert base64.b64decode(images[0].data) == png
            assert (
                result.structured_content["selected_preview"] == "previews/slice-0.png"
            )
            links = [item for item in result.content if isinstance(item, ResourceLink)]
            assert any(
                item.mime_type == "video/mp4" and item.uri.endswith("movie.mp4")
                for item in links
            )
            assert all(
                "movie resource must never be embedded" not in str(item)
                for item in result.content
            )
            invalid = await client.call_tool(
                "get_visualization",
                {"visualization_id": "render-1", "preview": "../outside.png"},
            )
            assert invalid.is_error
            # PNG inspection remains available without a local browser server.
            fail_viewer = True
            result = await client.call_tool(
                "get_visualization", {"visualization_id": "render-1"}
            )
            assert not result.is_error
            assert any(isinstance(item, ImageContent) for item in result.content)
            assert "viewer_warning" in result.structured_content
            # Large metadata stays on disk; oversize PNGs are never embedded.
            manifest["warnings"] = ["x" * (MAX_METADATA_BYTES + 1)]
            manifest_path.write_text(json.dumps(manifest))
            (bundle / "previews" / "slice-0.png").write_bytes(
                png + b"x" * MAX_PNG_BYTES
            )
            result = await client.call_tool(
                "get_visualization", {"visualization_id": "render-1"}
            )
            assert not result.is_error
            assert not any(isinstance(item, ImageContent) for item in result.content)
            assert "image_warning" in result.structured_content
            assert "metadata_warning" in result.structured_content
            assert (
                len(json.dumps(result.structured_content).encode()) < MAX_METADATA_BYTES
            )

    asyncio.run(exercise())


def test_launch_retry_preserves_existing_job_after_inputs_change(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from pyurbanair_mcp.tools import Tools

    tools = Tools(REPO, tmp_path)
    job = {
        "id": "saved-run",
        "kind": "forward",
        "state": "succeeded",
        "run_root": str(tmp_path / "saved-run"),
        "details": {},
    }
    monkeypatch.setattr(
        tools.preparation,
        "load",
        lambda plan_id: {"digest": "saved-digest", "environment": "dev"},
    )

    def stale(plan_id: str) -> None:
        raise ValueError("Prepared input changed")

    monkeypatch.setattr(tools.preparation, "verify", stale)

    def request(operation: str, **kwargs: Any) -> Any:
        assert operation == "existing", "A retry must not submit another job"
        assert kwargs["payload"]["plan_digest"] == "saved-digest"
        if kwargs["key"] == "original-key":
            return job
        if kwargs["key"] == "conflicting-key":
            raise ValueError(
                "Idempotency key already refers to different launch inputs"
            )
        return None

    monkeypatch.setattr(tools.jobs, "request", request)
    assert (
        tools.launch_forward_run("saved-plan", "original-key")["run_id"] == "saved-run"
    )
    with pytest.raises(ValueError, match="different launch inputs"):
        tools.launch_forward_run("saved-plan", "conflicting-key")
    with pytest.raises(ValueError, match="Prepared input changed"):
        tools.launch_forward_run("saved-plan", "new-key")
