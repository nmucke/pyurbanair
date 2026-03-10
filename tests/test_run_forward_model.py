import pathlib
import sys

import pytest

# Patch scripts.config to use tests.config before any script imports
PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import tests.config as tests_config

sys.modules["scripts.config"] = tests_config


@pytest.mark.parametrize(
    "model",
    [
        pytest.param("pylbm", id="pylbm"),
        pytest.param("pyudales", id="pyudales", marks=pytest.mark.udales),
    ],
)
@pytest.mark.parametrize(
    "use_results_dir",
    [
        pytest.param(False, id="default_results_dir"),
        pytest.param(True, id="custom_results_dir"),
    ],
)
def test_run_forward_model(model: str, use_results_dir: bool, tmp_path: pathlib.Path) -> None:
    """Test run_forward_model.py with pylbm and pyudales backends."""
    from scripts.run_forward_model import main

    # Set argv for argparse
    original_argv = sys.argv
    argv = ["run_forward_model", "--model", model, "--skip-viz"]
    if use_results_dir:
        results_dir = tmp_path / "results"
        results_dir.mkdir()
        argv.extend(["--results-dir", str(results_dir)])
    sys.argv = argv
    try:
        main()
    finally:
        sys.argv = original_argv
