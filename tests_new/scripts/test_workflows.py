"""The shell workflows in workflows/, run on the tiny surrogate."""

from __future__ import annotations

import os
import pathlib
import subprocess
import sys
from typing import Any

from tests_new.conftest import REPO, TEST_CONFIGS, surrogate


def _workflow(script: str, *args: str) -> None:
    # The workflows call `python`: make it this interpreter.
    env = {
        **os.environ,
        "PATH": f"{pathlib.Path(sys.executable).parent}:{os.environ['PATH']}",
    }
    subprocess.run(
        ["bash", str(REPO / "workflows" / script), *args], env=env, check=True
    )


def _overrides(root: pathlib.Path, session_root: pathlib.Path) -> list[str]:
    return [
        "--config-dir",
        str(TEST_CONFIGS),
        f"paths.results_root={root}",
        f"paths.scratch.local={root / 'scratch'}",
        *surrogate(session_root),
    ]


def test_forward_workflow(
    tmp_path: pathlib.Path, session_root: pathlib.Path, trained: Any
) -> None:
    _workflow(
        "forward_workflow.sh",
        *_overrides(tmp_path, session_root),
        "+test=forward",
        "model=neural_surrogate_tiny",
    )
    assert (tmp_path / "neural_surrogate" / "figures" / "parameters.png").exists()


def test_assimilation_workflow(
    tmp_path: pathlib.Path, session_root: pathlib.Path, trained: Any
) -> None:
    _workflow(
        "assimilation_workflow.sh",
        "smoother",
        *_overrides(tmp_path, session_root),
        "+test=assimilation",
        "model@truth_model=neural_surrogate_tiny",
        "model@assim_model=neural_surrogate_tiny",
    )
    run_dir = tmp_path / "neural_surrogate_to_neural_surrogate" / "smoother"
    assert (run_dir / "metrics.yaml").exists()
    assert (run_dir / "figures" / "parameter_evolution.png").exists()
