"""Official MCP SDK v2 registration; the core remains SDK independent."""

from __future__ import annotations

import base64
import json
import pathlib

from mcp.server import MCPServer
from mcp.types import CallToolResult, ImageContent, ResourceLink, TextContent
from pyurbanair_mcp.tools import Tools


def create_server(
    repo_root: str | pathlib.Path, store_root: str | pathlib.Path | None = None
) -> MCPServer:
    tools = Tools(repo_root, store_root)
    server = MCPServer(
        "pyurbanair",
        instructions="Discover and inspect forward settings, prepare a resolved plan, then launch within the user's request. Poll persistent run IDs. Rendering reads saved results and does not rerun the solver.",
    )
    for name in (
        "get_capabilities",
        "list_config_options",
        "inspect_config",
        "prepare_forward_run",
        "launch_forward_run",
        "list_runs",
        "get_run_status",
        "get_run_logs",
        "cancel_run",
        "inspect_run_results",
        "render_simulation",
    ):
        server.tool()(getattr(tools, name))

    def get_visualization(
        visualization_id: str, preview: str | None = None
    ) -> CallToolResult:
        """Retrieve visualization metadata, browser URL and bounded PNG image content."""
        metadata, png = tools.visualization(visualization_id, preview)
        result = CallToolResult(
            content=[TextContent(type="text", text=json.dumps(metadata))],
            structured_content=metadata,
        )
        if png is not None:
            result.content.append(
                ImageContent(
                    type="image",
                    data=base64.b64encode(png).decode("ascii"),
                    mime_type="image/png",
                )
            )
        for artifact in metadata.get("artifacts", []):
            result.content.append(
                ResourceLink(
                    type="resource_link",
                    name=artifact["name"],
                    uri=artifact["uri"],
                    mime_type=artifact["mime_type"],
                    size=artifact["size"],
                )
            )
        return result

    def plan_snapshot(plan_id: str) -> str:
        """The immutable resolved preparation snapshot."""
        return json.dumps(tools.preparation.load(plan_id), indent=2)

    server.tool()(get_visualization)
    server.resource("pyurbanair://plans/{plan_id}")(plan_snapshot)
    return server
