"""A throwaway checkout for the MCP job-preparation tests.

`checkout` copies the real configs/ with the tests/configs/ overlays merged in,
so a test selects the tiny run with the usual `+test=forward` or
`model=pyudales_tiny`. It also holds what preparation reads from a checkout:
scripts/utils/ (check_config), the stdlib-only native editors and the Xie &
Castro case inputs.
"""

from __future__ import annotations

import importlib.util
import shutil
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]

# These tests run in the mcp env, the only one that installs mcp-server.
if importlib.util.find_spec("mcp_server") is None:
    collect_ignore_glob = ["test_*.py"]


@pytest.fixture  # type: ignore[misc]
def checkout(tmp_path: Path) -> Path:
    root = tmp_path / "checkout"
    ignore = shutil.ignore_patterns("__pycache__")
    shutil.copytree(REPO / "configs", root / "configs", ignore=ignore)
    shutil.copytree(REPO / "tests/configs/test", root / "configs/test", ignore=ignore)
    for overlay in sorted((REPO / "tests/configs/model").glob("*.yaml")):
        shutil.copyfile(overlay, root / "configs/model" / overlay.name)
    shutil.copytree(REPO / "scripts/utils", root / "scripts/utils", ignore=ignore)
    for backend, filename in (
        ("pyudales", "namoptions_utils.py"),
        ("pypalm", "p3d_utils.py"),
    ):
        relative = Path("libs") / backend / "src" / backend / "utils" / filename
        (root / relative).parent.mkdir(parents=True)
        shutil.copyfile(REPO / relative, root / relative)
    for relative_case in (
        "examples/udales/xie_and_castro/namoptions.300",
        "examples/palm/xie_and_castro/_p3d",
    ):
        destination = root / relative_case
        destination.parent.mkdir(parents=True)
        shutil.copyfile(REPO / relative_case, destination)
    geometry = root / "examples/xie_and_castro/xie_castro_2008_STL.stl"
    geometry.parent.mkdir(parents=True)
    geometry.write_text("solid test\nendsolid test\n")
    return root
