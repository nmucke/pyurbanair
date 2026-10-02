"""Pure configuration and immutable preparation contracts; no solver imports."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import pytest

from pyurbanair.config.composition import (
    compose_forward_config,
    inspect_config,
    list_config_options,
)
from pyurbanair.jobs.native import native_coverage
from pyurbanair.jobs.paths import bind_job_paths, owned_path
from pyurbanair.jobs.preparation import PreparationService

REPO = Path(__file__).resolve().parents[2]


@pytest.fixture  # type: ignore[misc]
def checkout(tmp_path: Path) -> Path:
    root = tmp_path / "checkout"
    shutil.copytree(REPO / "tests/conf", root / "conf")
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


@pytest.mark.parametrize("backend", ["pylbm", "pyudales", "pypalm", "neural_surrogate"])  # type: ignore[misc]
def test_composition_is_backend_free(checkout: Path, backend: str) -> None:
    script = """
import json, sys
from pyurbanair.config.composition import compose_forward_config
result = compose_forward_config(sys.argv[1], [f'model@model={sys.argv[2]}'])
assert result['config']['model']['name'] == sys.argv[2]
for name in ('jax', 'torch', 'pylbm', 'pyudales', 'pypalm', 'neural_surrogates', 'pyurbanair.config.hydra_helpers'):
    assert name not in sys.modules, name
print(json.dumps(result['config']['paths']))
"""
    result = subprocess.run(
        [sys.executable, "-c", script, str(checkout), backend],
        capture_output=True,
        text=True,
        check=True,
        cwd="/tmp",
    )
    assert str(checkout) in json.loads(result.stdout)["experiment_dir"]
    assert not list(checkout.glob(".temp*"))


def test_ordered_nested_add_delete_overrides(checkout: Path) -> None:
    result = compose_forward_config(
        checkout,
        [
            "params=static",
            "domain.nx=10",
            "domain.nx=12",
            "domain.bounds=[[0,10],[0,20],[0,5]]",
            "+model.forward_model.random_initial_condition_args={seed:17}",
            "~params.parameters.sgs_constant",
        ],
    )["config"]
    assert result["domain"]["nx"] == 12
    assert result["domain"]["bounds"] == [[0, 10], [0, 20], [0, 5]]
    assert result["model"]["forward_model"]["random_initial_condition_args"] == {
        "seed": 17
    }
    assert "sgs_constant" not in result["params"]["parameters"]


@pytest.mark.parametrize(  # type: ignore[misc]
    "override",
    [
        "hydra.run.dir=/tmp",
        "+hydra.searchpath=[file:///tmp]",
        "model.forward_model._target_=os.system",
        "+model.forward_model.extra={_target_:builtins.eval}",
        "model.forward_model.experiment_name=${oc.env:HOME}",
        "+model.forward_model.extra=[{_target_:os.system}]",
    ],
)
def test_untrusted_overrides_fail(checkout: Path, override: str) -> None:
    with pytest.raises(ValueError):
        compose_forward_config(checkout, [override])


def test_discovery_and_subtree(checkout: Path) -> None:
    options = list_config_options(checkout, group="model", page_size=2)
    assert options["total"] == 4 and len(options["options"]) == 2
    result = inspect_config(checkout, ["domain.nx=16"], "domain.nx")
    assert result["config"] == 16
    assert "conf/run_forward_model.yaml" in result["source_yaml"]


def test_parallel_composition_is_serialized(checkout: Path) -> None:
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(
            pool.map(
                lambda size: compose_forward_config(checkout, [f"domain.nx={size}"])[
                    "config"
                ]["domain"]["nx"],
                (8, 10, 12, 14),
            )
        )
    assert results == [8, 10, 12, 14]


def test_native_staging_and_plan_integrity(checkout: Path, tmp_path: Path) -> None:
    service = PreparationService(checkout, tmp_path / "store")
    original = checkout / "examples/udales/xie_and_castro/namoptions.300"
    before = original.read_bytes()
    plan = service.prepare(native_overrides={"RUN": {"dtmax": 0.25}})
    assert plan["validation"]["configuration_valid"]
    staged = (
        Path(plan["config"]["model"]["forward_model"]["case_dir"]) / "namoptions.300"
    )
    assert "0.25" in staged.read_text()
    assert original.read_bytes() == before
    assert service.verify(plan["plan_id"])["digest"] == plan["digest"]
    staged.write_text(staged.read_text().replace("0.25", "0.1"))
    with pytest.raises(ValueError, match="input changed"):
        service.verify(plan["plan_id"])


@pytest.mark.parametrize(  # type: ignore[misc]
    "backend,field",
    [
        ("pyudales", "DOMAIN.itot"),
        ("pypalm", "initialization_parameters.nx"),
        ("pylbm", "iout"),
    ],
)
def test_native_managed_fields_direct_to_hydra(
    checkout: Path, tmp_path: Path, backend: str, field: str
) -> None:
    with pytest.raises(ValueError, match="wrapper-managed; use"):
        PreparationService(checkout, tmp_path / "store").prepare(
            [f"model={backend}"], native_overrides={field: 2}
        )
    assert native_coverage(backend)["unlisted_fields"] == "unsupported"


def test_unknown_native_field_is_not_silently_ignored(
    checkout: Path, tmp_path: Path
) -> None:
    with pytest.raises(ValueError, match="unsupported"):
        PreparationService(checkout, tmp_path / "store").prepare(
            native_overrides={"RUN.magic": 1}
        )


def test_inputs_and_code_are_verified(checkout: Path, tmp_path: Path) -> None:
    service = PreparationService(checkout, tmp_path / "store")
    plan = service.prepare()
    original = checkout / "examples/udales/xie_and_castro/namoptions.300"
    original.write_text(original.read_text() + "\n! changed\n")
    with pytest.raises(ValueError, match="input changed"):
        service.verify(plan["plan_id"])
    plan = service.prepare()
    (checkout / "conf/model/pyudales.yaml").write_text(
        (checkout / "conf/model/pyudales.yaml").read_text() + "\n# changed\n"
    )
    with pytest.raises(ValueError, match="code or configuration changed"):
        service.verify(plan["plan_id"])


def test_modified_plan_is_rejected(checkout: Path, tmp_path: Path) -> None:
    service = PreparationService(checkout, tmp_path / "store")
    plan = service.prepare()
    destination = service.store_root / "plans" / plan["plan_id"] / "plan.json"
    destination.chmod(0o600)
    plan["config"]["domain"]["nx"] = 1
    destination.write_text(json.dumps(plan))
    with pytest.raises(ValueError, match="plan was modified"):
        service.load(plan["plan_id"])
    with pytest.raises(ValueError, match="invalid plan ID"):
        service.load("../outside")


@pytest.mark.parametrize(  # type: ignore[misc]
    "overrides,field",
    [
        (["domain.nx=-1"], "domain.nx"),
        (["time.output_frequency=0"], "time.output_frequency"),
        (["run.rollout_steps=-1"], "run.rollout_steps"),
        (["run.rollout_steps=100"], "windows"),
        (["run.ensemble_save_on_disk=true"], "run.ensemble_save_on_disk"),
        (["model.forward_model.ncpu=3"], "model.forward_model.ncpu"),
        (["model=pypalm", "domain.nx=21"], "domain"),
    ],
)
def test_scientific_validation(
    checkout: Path, tmp_path: Path, overrides: list[str], field: str
) -> None:
    plan = PreparationService(checkout, tmp_path / "store").prepare(overrides)
    assert not plan["validation"]["configuration_valid"]
    assert any(item["field"] == field for item in plan["validation"]["issues"])


def test_requests_can_only_tighten_limits(checkout: Path, tmp_path: Path) -> None:
    service = PreparationService(checkout, tmp_path / "store", {"max_workers": 2})
    with pytest.raises(ValueError, match="only tighten"):
        service.prepare(execution_limits={"max_workers": 3})
    plan = service.prepare(
        ["run.ensemble=true", "ensemble.num_parallel_processes=2"],
        execution_limits={"max_workers": 1},
    )
    assert not plan["validation"]["configuration_valid"]


def test_all_managed_paths_are_private(checkout: Path, tmp_path: Path) -> None:
    config = compose_forward_config(
        checkout,
        [
            "model=neural_surrogate",
            "model.forward_model.spinup_source=forward_model",
            "+model.forward_model.spinup_forward_model.output_dir=/tmp/shared",
        ],
    )["config"]
    root = tmp_path / "run"
    bound, environment = bind_job_paths(config, root)
    assert Path(bound["paths"]["experiment_dir"]).is_relative_to(root)
    spinup = bound["model"]["forward_model"]["spinup_forward_model"]
    assert Path(spinup["output_dir"]).is_relative_to(root)
    assert spinup["verbose"]
    assert all(Path(value).is_relative_to(root) for value in environment.values())
    root.mkdir()
    (root / "scratch").symlink_to(tmp_path / "outside")
    with pytest.raises(ValueError, match="escapes"):
        bind_job_paths(config, root)
    with pytest.raises(ValueError, match="escapes"):
        owned_path(root, "../outside")


def test_initial_state_missing_and_surrogate_cold_start(
    checkout: Path, tmp_path: Path
) -> None:
    plan = PreparationService(checkout, tmp_path / "store").prepare(
        ["model=neural_surrogate"], initial_state={"path": "missing.nc"}
    )
    assert not plan["validation"]["prerequisites_present"]
    assert any(
        issue["field"] == "initial_state.path" for issue in plan["validation"]["issues"]
    )
    plan = PreparationService(checkout, tmp_path / "store2").prepare(
        ["model=neural_surrogate"]
    )
    assert any(
        issue["field"] == "initial_state" for issue in plan["validation"]["issues"]
    )


def test_constructor_grid_cannot_bypass_resource_limits(
    checkout: Path, tmp_path: Path
) -> None:
    plan = PreparationService(checkout, tmp_path / "store").prepare(
        ["model.forward_model.nx=1000000"]
    )
    assert not plan["validation"]["configuration_valid"]
    assert any(
        issue["field"] == "model.forward_model.nx"
        for issue in plan["validation"]["issues"]
    )
