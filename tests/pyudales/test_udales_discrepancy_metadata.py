"""Member provenance survives memory concatenation and disk loading."""

from __future__ import annotations

import json
import pathlib
from types import SimpleNamespace
from typing import Any, cast

import pytest
import xarray as xr
from pyudales.ensemble_forward_model import EnsembleForwardModel

from pyurbanair.base_ensemble_forward_model import BaseEnsembleForwardModel


def _make_ensemble(tmp_path: pathlib.Path, enabled: bool = True) -> Any:
    ensemble = cast(Any, EnsembleForwardModel).__new__(EnsembleForwardModel)
    members = []
    for index in range(2):
        experiment = tmp_path / str(index)
        experiment.mkdir()
        metadata = {
            "coefficients": {"sgs_bias_b0": float(index)},
            "native_diagnostics": f"member {index}",
        }
        if enabled:
            (experiment / "model_discrepancy.json").write_text(json.dumps(metadata))
        state = xr.Dataset({"u": ("time", [float(index)])})
        if enabled:
            state.attrs["model_discrepancy"] = json.dumps(metadata)
        filename = tmp_path / f"state_{index}.nc"
        state.to_netcdf(filename)

        def get_states(sim_name: str, root: pathlib.Path = tmp_path) -> xr.Dataset:
            with xr.open_dataset(root / f"{sim_name}.nc") as stored:
                return stored.load()

        members.append(
            SimpleNamespace(
                dirs=SimpleNamespace(experiment_dir=experiment),
                model_discrepancy={"enabled": enabled},
                _elapsed_time=0.0,
                get_states=get_states,
            )
        )
    ensemble.ensemble_forward_models = members
    ensemble.ensemble_size = 2
    ensemble._last_failure_substitutions = {}
    return ensemble


@pytest.mark.parametrize("donor_substitution", [False, True])  # type: ignore[misc]
def test_memory_metadata_preserves_each_member_and_failure_donor(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, donor_substitution: bool
) -> None:
    import pyudales.ensemble_forward_model as module

    ensemble = _make_ensemble(tmp_path)
    if donor_substitution:
        ensemble._last_failure_substitutions = {1: 0}
    result = xr.Dataset({"u": (("ensemble", "time"), [[0.0], [1.0]])})
    result.attrs["model_discrepancy"] = json.dumps({"first_member_only": True})
    monkeypatch.setattr(
        BaseEnsembleForwardModel, "run_ensemble", lambda *args, **kwargs: result
    )
    monkeypatch.setattr(module, "copy_carry", lambda *args: None)
    monkeypatch.setattr(module, "copy_elapsed_time", lambda *args: True)
    monkeypatch.setattr(module, "read_elapsed_time", lambda dirs, default: default)
    output = ensemble.run_ensemble()
    assert "model_discrepancy" not in output.attrs
    metadata = json.loads(output.attrs["model_discrepancy_by_member"])
    assert [item["coefficients"]["sgs_bias_b0"] for item in metadata] == [
        0.0,
        0.0 if donor_substitution else 1.0,
    ]
    assert metadata[1]["native_diagnostics"] == (
        "member 0" if donor_substitution else "member 1"
    )


def test_disk_metadata_comes_from_saved_states_not_current_sidecars(
    tmp_path: pathlib.Path,
) -> None:
    ensemble = _make_ensemble(tmp_path)
    for member in ensemble.ensemble_forward_models:
        (member.dirs.experiment_dir / "model_discrepancy.json").unlink()
    output = ensemble.get_states()
    metadata = json.loads(output.attrs["model_discrepancy_by_member"])
    assert "model_discrepancy" not in output.attrs
    assert [item["coefficients"]["sgs_bias_b0"] for item in metadata] == [0.0, 1.0]
    path = tmp_path / "combined.nc"
    output.to_netcdf(path)
    with xr.open_dataset(path) as stored:
        assert json.loads(stored.attrs["model_discrepancy_by_member"]) == metadata


def test_disabled_ensemble_has_no_new_metadata(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import pyudales.ensemble_forward_model as module

    ensemble = _make_ensemble(tmp_path, enabled=False)
    expected = ensemble.get_states()
    monkeypatch.setattr(
        BaseEnsembleForwardModel, "run_ensemble", lambda *args, **kwargs: expected
    )
    monkeypatch.setattr(module, "read_elapsed_time", lambda dirs, default: default)
    output = ensemble.run_ensemble()
    assert output.attrs == {}
    assert not list(tmp_path.glob("*/model_discrepancy.json"))
    # A previous enabled run can leave a sidecar; disabled members ignore it.
    stale = tmp_path / "0" / "model_discrepancy.json"
    stale.write_text(json.dumps({"coefficients": {"sgs_bias_b0": 99.0}}))
    assert ensemble.run_ensemble().attrs == {}
