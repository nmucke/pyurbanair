import pathlib
import sys

import pytest

# Patch scripts.config to use tests.config before any script imports
PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import tests.config as tests_config

sys.modules["scripts.config"] = tests_config


def test_run_time_varying_forward_model(tmp_path: pathlib.Path) -> None:
    """Test run_time_varying_forward_model.py runs to completion."""
    from scripts.run_time_varying_forward_model import main

    results_dir = tmp_path / "results"
    results_dir.mkdir()

    original_argv = sys.argv
    sys.argv = [
        "run_time_varying_forward_model",
        "--skip-viz",
        "--results-dir",
        str(results_dir),
    ]
    try:
        main()
    finally:
        sys.argv = original_argv
