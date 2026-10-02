"""Pure configuration and immutable preparation contracts; no solver imports."""

from __future__ import annotations

import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
from mcp_server.jobs.composition import (
    compose_forward_config,
    inspect_config,
    list_config_options,
)
from mcp_server.jobs.native import native_coverage
from mcp_server.jobs.paths import bind_job_paths, owned_path
from mcp_server.jobs.preparation import PreparationService

REPO = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("backend", ["pylbm", "pyudales", "pypalm", "neural_surrogate"])  # type: ignore[misc]
def test_composition_is_backend_free(checkout: Path, backend: str) -> None:
    script = """
import json, sys
from mcp_server.jobs.composition import compose_forward_config
result = compose_forward_config(sys.argv[1], [f'model={sys.argv[2]}'])
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


def test_test_overlays_compose(checkout: Path) -> None:
    config = compose_forward_config(checkout, ["+test=forward"])["config"]
    assert config["model"]["name"] == "pyudales"
    assert [config["domain"][key] for key in ("nx", "ny", "nz")] == [20, 20, 4]
    assert config["model"]["forward_model"]["nudging_config"]["nnudge_meters"] == 4.0
    config = compose_forward_config(checkout, ["+test=forward", "model=pylbm_tiny"])[
        "config"
    ]
    assert config["model"]["name"] == "pylbm"
    assert config["model"]["forward_model"]["cuda"] is False


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
            "forward.rollout_steps=2",
        ],
    )["config"]
    assert result["domain"]["nx"] == 12
    assert result["model"]["forward_model"]["nx"] == 12
    assert result["domain"]["bounds"] == [[0, 10], [0, 20], [0, 5]]
    assert result["model"]["forward_model"]["random_initial_condition_args"] == {
        "seed": 17
    }
    assert "sgs_constant" not in result["params"]["parameters"]
    assert result["forward"]["rollout_steps"] == 2


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
    # The four backends plus the three tests/configs/model overlays.
    options = list_config_options(checkout, group="model", page_size=2)
    assert options["total"] == 7 and len(options["options"]) == 2
    second = list_config_options(checkout, group="model", page=1, page_size=2)
    assert second["options"] != options["options"]
    assert {entry["group"] for entry in list_config_options(checkout)["options"]} == {
        "case",
        "model",
        "params",
        "visualization",
    }
    found = list_config_options(checkout, search="tiny")["options"]
    assert found and all("tiny" in entry["name"] for entry in found)
    with pytest.raises(ValueError):
        list_config_options(checkout, page_size=0)
    result = inspect_config(checkout, ["domain.nx=16"], "domain.nx")
    assert result["config"] == 16
    assert "configs/forward.yaml" in result["source_yaml"]
    assert "configs/model/pyudales.yaml" in result["source_yaml"]
    with pytest.raises(ValueError, match="unknown configuration subtree"):
        inspect_config(checkout, subtree="domain.missing")


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


def test_plan_records_forward_run(checkout: Path, tmp_path: Path) -> None:
    plan = PreparationService(checkout, tmp_path / "store").prepare(
        ["+test=forward", "forward.ensemble=true", "forward.rollout_steps=1"]
    )
    assert plan["validation"]["configuration_valid"]
    assert plan["resources"]["members"] == 2
    assert plan["resources"]["windows"] == 2
    assert plan["resources"]["grid"] == {"nx": 20, "ny": 20, "nz": 4}
    assert "artifacts/state.nc" in plan["expected_artifacts"]
    assert {entry["field"] for entry in plan["diff_from_defaults"]} >= {
        "domain.nx",
        "forward.ensemble",
        "forward.rollout_steps",
    }


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
    for relative in (
        "configs/model/pyudales.yaml",
        "scripts/utils/inconsistency_check.py",
    ):
        plan = service.prepare()
        changed = checkout / relative
        changed.write_text(changed.read_text() + "\n# changed\n")
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


def test_check_config_rejection_is_a_configuration_issue(
    checkout: Path, tmp_path: Path
) -> None:
    plan = PreparationService(checkout, tmp_path / "store").prepare(
        [
            "+test=forward",
            "model=pylbm_tiny",
            "+model.forward_model.model_discrepancy={enabled:true}",
        ]
    )
    assert not plan["validation"]["configuration_valid"]
    issue = next(
        item for item in plan["validation"]["issues"] if item["field"] == "config"
    )
    assert issue["kind"] == "configuration"
    assert "model_discrepancy needs pyudales" in issue["message"]


@pytest.mark.parametrize(  # type: ignore[misc]
    "overrides,field",
    [
        (["forward.rollout_steps=100"], "windows"),
        (["forward.ensemble=true", "ensemble.ensemble_size=1000"], "members"),
        (
            ["forward.ensemble=true", "ensemble.num_parallel_processes=16"],
            "workers",
        ),
        (["model.forward_model.ncpu=256"], "cpu_threads"),
        (["domain.nx=4000", "domain.ny=4000"], "estimated_output_bytes"),
    ],
)
def test_resource_limits(
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
        ["forward.ensemble=true", "ensemble.num_parallel_processes=2"],
        execution_limits={"max_workers": 1},
    )
    assert not plan["validation"]["configuration_valid"]
    assert plan["limits"]["max_workers"] == 1


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
    assert Path(bound["paths"]["results_dir"]) == root.resolve() / "artifacts"
    assert bound["forward"]["save_windows"] is True
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


def test_experiment_name_must_be_a_plain_name(checkout: Path, tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="plain name"):
        PreparationService(checkout, tmp_path / "store").prepare(
            ["model.forward_model.experiment_name=../escape"]
        )


def test_initial_state_is_normalized_into_the_config(
    checkout: Path, tmp_path: Path
) -> None:
    state = checkout / "states/start.nc"
    state.parent.mkdir()
    state.write_bytes(b"netcdf placeholder")
    service = PreparationService(checkout, tmp_path / "store")
    plan = service.prepare(initial_state={"path": "states/start.nc", "member": 1})
    expected = {"path": str(state.resolve()), "member": 1}
    assert plan["config"]["forward"]["initial_state"] == expected
    assert plan["initial_state"] == expected
    assert any(
        item["path"] == str(state.resolve()) for item in plan["provenance"]["inputs"]
    )
    # A config-selected state is normalized the same way.
    plan = service.prepare(["forward.initial_state=states/start.nc"])
    assert plan["config"]["forward"]["initial_state"] == {"path": str(state.resolve())}
    state.write_bytes(b"changed")
    with pytest.raises(ValueError, match="input changed"):
        service.verify(plan["plan_id"])
    with pytest.raises(ValueError, match="path, member and time_index"):
        service.prepare(initial_state={"path": "states/start.nc", "extra": 1})


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
