"""City case builds as supervisor jobs: cache key, environment and worker step.

The MCP and browser-app processes do not install pycity4cfd; they compute the
case folder from the inputs and queue a `city_case` job, whose worker runs
`pycity4cfd.case.make_case` in the `city4cfd` Pixi environment.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

ENVIRONMENT = "city4cfd"
# Bump when a change to pycity4cfd.case makes older case folders unusable
# (files, case.json keys, overrides): it is part of the folder key, so folders
# of an older format are rebuilt, not reused.
CASE_FORMAT = 2


def case_dir(
    repo_root: str | Path,
    bbox: tuple[float, float, float, float],
    resolution: float,
    default_height: float,
) -> Path:
    """The folder `.temp/city_cases/area_<hash>` of these inputs."""
    key = ",".join(
        [f"v{CASE_FORMAT}"]
        + [f"{value:.5f}" for value in (*bbox, resolution, default_height)]
    )
    name = "area_" + hashlib.sha1(key.encode()).hexdigest()[:10]
    return Path(repo_root) / ".temp" / "city_cases" / name


def select_case_environment(repo_root: str | Path) -> str:
    """The `city4cfd` environment, or a ValueError saying how to install it."""
    if not (Path(repo_root) / ".pixi/envs" / ENVIRONMENT / "bin/python").is_file():
        raise ValueError(
            f"Building a city case needs the {ENVIRONMENT} Pixi environment; "
            f"run `pixi install -e {ENVIRONMENT}`"
        )
    return ENVIRONMENT


def build_case(payload: dict[str, Any]) -> dict[str, Any]:
    """The worker step: build the case folder and return its summary."""
    from pycity4cfd.case import make_case

    result: dict[str, Any] = make_case(
        tuple(payload["bbox"]),
        payload["case_dir"],
        resolution=payload["resolution"],
        default_height=payload["default_height"],
    )
    return result
