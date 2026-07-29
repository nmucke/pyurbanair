"""Regression tests for per-attempt uDALES output cleanup."""

import pathlib
from typing import Any

import pytest
import xarray
from pyudales.utils.dir_utils import DirectoryPaths
from pyudales.utils.run_monitor import InstabilityCheck


def _make_dirs(root: pathlib.Path, exp: str) -> DirectoryPaths:
    experiment_dir = root / "experiment" / exp
    output_dir = root / "outputs"
    experiment_dir.mkdir(parents=True)
    (output_dir / exp).mkdir(parents=True)
    (experiment_dir / f"namoptions.{exp}").write_text(
        "&RUN\nruntime = 3.0\n/\n" "&DOMAIN\nitot = 8\njtot = 6\nktot = 4\n/\n"
    )
    return DirectoryPaths(
        udales_root_path=root,
        cwd=root,
        temp_dir=root,
        experiment_base_dir=root / "experiment",
        experiment_dir=experiment_dir,
        output_dir=output_dir,
        case_dir=root,
        experiment_name=exp,
    )


def test_run_single_cleans_output_before_staging_warmstart(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A retry removes stale fielddumps without deleting its restored carry."""
    import pyudales.forward_model as forward_model_module
    from pyudales.forward_model import ForwardModel

    dirs = _make_dirs(tmp_path, "000")
    output_dir: pathlib.Path = dirs.output_dir / dirs.experiment_name
    stale_fielddump = output_dir / "fielddump.000.nc"
    stale_fielddump.write_bytes(b"wrong-grid-output")

    model = ForwardModel.__new__(ForwardModel)
    model.dirs = dirs
    model._elapsed_time = 0.0
    model.spinup_time = 2.0
    model._simulation_time = 3.0
    model.inlet_turbulence = {}
    model._instability_check = InstabilityCheck(enabled=False)
    model.stdout = None
    model.stderr = None

    staged_restart: pathlib.Path = output_dir / "initd00000005_000_000.000"

    def fake_apply_inflow_settings(
        params: xarray.Dataset | None, warm_start: bool = False
    ) -> None:
        assert warm_start is True
        assert not stale_fielddump.exists()

    def fake_fetch_carry(_dirs: DirectoryPaths) -> pathlib.Path:
        assert not stale_fielddump.exists()
        staged_restart.write_bytes(b"carried-subgrid-fields")
        return staged_restart

    def fake_prepare_warmstart(
        state: xarray.Dataset, template_file: pathlib.Path | None = None
    ) -> None:
        assert template_file == staged_restart
        assert staged_restart.exists()

    def fake_run_with_dt_watchdog(*args: Any, **kwargs: Any) -> None:
        # This is the actual _run_executable boundary. The restart staged above
        # must still be present when uDALES would be launched.
        assert staged_restart.exists()
        assert not stale_fielddump.exists()

    monkeypatch.setattr(model, "_apply_inflow_settings", fake_apply_inflow_settings)
    monkeypatch.setattr(model, "_window_runtime", lambda warm_start: 3.0)
    monkeypatch.setattr(model, "_snapshot_namoptions", lambda: "original")
    monkeypatch.setattr(model, "_rewrite_runtime", lambda runtime: None)
    monkeypatch.setattr(model, "_prepare_warmstart", fake_prepare_warmstart)
    monkeypatch.setattr(
        model,
        "_load_and_postprocess_state",
        lambda: xarray.Dataset({"u": ("time", [1.0])}),
    )
    monkeypatch.setattr(model, "_restore_namoptions", lambda text: None)
    monkeypatch.setattr(
        forward_model_module, "read_elapsed_time", lambda dirs, default: default
    )
    monkeypatch.setattr(forward_model_module, "set_trestart", lambda dirs: None)
    monkeypatch.setattr(forward_model_module, "fetch_carry", fake_fetch_carry)
    monkeypatch.setattr(forward_model_module, "store_carry", lambda dirs: None)
    monkeypatch.setattr(
        forward_model_module,
        "clean_output_except_warmstart_files",
        lambda dirs: None,
    )
    monkeypatch.setattr(
        forward_model_module, "remove_old_warmstart_files", lambda dirs: None
    )
    monkeypatch.setattr(
        forward_model_module, "run_with_dt_watchdog", fake_run_with_dt_watchdog
    )

    result = model.run_single(state=xarray.Dataset({"u": ("time", [0.0])}))

    assert result["u"].item() == 1.0
    assert staged_restart.exists()
    assert not stale_fielddump.exists()
