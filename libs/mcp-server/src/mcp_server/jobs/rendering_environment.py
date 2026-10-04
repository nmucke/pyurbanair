"""Select provisioned render dependencies without importing a renderer."""

from pathlib import Path
from typing import Any


def select_render_environment(repo_root: str | Path, options: dict[str, Any]) -> str:
    if options.get("render_3d", False):
        return "rendering"
    if (
        options.get("movie", True)
        and (Path(repo_root) / ".pixi/envs/rendering/bin/python").is_file()
    ):
        return "rendering"
    return "dev"
