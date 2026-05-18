import pathlib


def test_run_time_varying_forward_model(
    tmp_path: pathlib.Path,
    compose_test_cfg,
) -> None:
    """Test run_time_varying_forward_model.py runs to completion."""
    from scripts.run_time_varying_forward_model import run

    results_dir = tmp_path / "results"
    results_dir.mkdir()

    overrides = [
        "run.skip_viz=true",
        f"run.results_dir={results_dir}",
        f"paths.base_results_dir={tmp_path}",
        "model.forward_model.cuda=false",
    ]

    run(compose_test_cfg(overrides))
