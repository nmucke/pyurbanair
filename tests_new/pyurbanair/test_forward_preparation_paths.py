"""Regression guards for constructor overrides and selected runtime budgets."""

from pathlib import Path

import pytest
from omegaconf import OmegaConf

from pyurbanair.config.composition import compose_forward_config
from pyurbanair.jobs.paths import bind_job_paths, ensure_private_directory
from pyurbanair.jobs.preparation import PreparationService, _artifact_architecture
from tests_new.pyurbanair.test_forward_preparation import checkout


def test_surrogate_checks_selected_cfd_spinup_prerequisites(
    checkout: Path, tmp_path: Path
) -> None:
    worker_bin = checkout / ".pixi/envs/dev/bin"
    worker_bin.mkdir(parents=True)
    (worker_bin / "python").write_text("test interpreter identity")
    (worker_bin / "mpirun").write_text("test MPI identity")
    (checkout / "libs/pyudales/u-dales/tools/IBM/IBM_preproc_fortran").mkdir(
        parents=True
    )
    original = compose_forward_config(checkout)["config"]
    training = tmp_path / "training"
    training.mkdir()
    OmegaConf.save(
        OmegaConf.create({"domain": original["domain"], "time": original["time"]}),
        training / "config.yaml",
    )
    exported = tmp_path / "export"
    exported.mkdir()
    OmegaConf.save(
        OmegaConf.create(
            {
                "architecture": {
                    "_target_": "pyurbanair.static_parameters.Normal",
                    "mean": 0,
                    "std": 1,
                },
                "dataset": {"root_dir": str(training)},
            }
        ),
        exported / "config.yaml",
    )
    (exported / "weights.pt").write_bytes(b"test checkpoint identity")
    service = PreparationService(checkout, tmp_path / "store")
    assert service.capabilities()["backends"]["neural_surrogate"][
        "prerequisites_present"
    ]
    plan = service.prepare(
        [
            "model=neural_surrogate",
            "model.forward_model.device=cpu",
            "model.forward_model.spinup_source=forward_model",
            f"model.forward_model.model_dir={exported}",
        ]
    )
    assert not plan["validation"]["prerequisites_present"]
    assert any(
        issue["field"] == "backend.pyudales" and "mpif90" in issue["message"]
        for issue in plan["validation"]["issues"]
    )


def test_udales_readiness_uses_managed_build_prerequisites(
    checkout: Path, tmp_path: Path
) -> None:
    worker_bin = checkout / ".pixi/envs/dev/bin"
    worker_bin.mkdir(parents=True)
    for name in ("python", "mpirun", "mpif90", "cmake", "nc-config", "nf-config"):
        (worker_bin / name).write_text("test tool identity")
    source = checkout / "libs/pyudales/u-dales"
    source.mkdir(parents=True)
    (source / ".git").write_text("gitdir: test source repository")
    service = PreparationService(checkout, tmp_path / "store")
    assert service.capabilities()["backends"]["pyudales"]["prerequisites_present"]
    (worker_bin / "mpif90").unlink()
    assert not service.capabilities()["backends"]["pyudales"]["prerequisites_present"]


@pytest.mark.parametrize("kind", ["surrogate", "generator"])  # type: ignore[misc]
@pytest.mark.parametrize("nested", [False, True])  # type: ignore[misc]
def test_exported_architecture_references_cannot_hide_targets(
    checkout: Path, tmp_path: Path, kind: str, nested: bool
) -> None:
    exported = tmp_path / "export"
    exported.mkdir()
    architecture = (
        "architecture:\n  _target_: pyurbanair.static_parameters.Normal\n  extra: ${other_arch}\n"
        if nested
        else "architecture: ${other_arch}\n"
    )
    (exported / "config.yaml").write_text(
        architecture
        + "other_arch:\n  _target_: os.system\n  command: false\ndataset:\n  root_dir: missing\n"
    )
    overrides = ["model=neural_surrogate", f"model.forward_model.model_dir={exported}"]
    if kind == "generator":
        benign_surrogate = tmp_path / "benign_surrogate"
        benign_surrogate.mkdir()
        (benign_surrogate / "config.yaml").write_text(
            "architecture:\n  _target_: pyurbanair.static_parameters.Normal\n  mean: 0\n  std: 1\ndataset:\n  root_dir: missing\n"
        )
        overrides.extend(
            [
                f"model.forward_model.model_dir={benign_surrogate}",
                "model.forward_model.spinup_source=generative",
                f"model.forward_model.generative_spinup.model_dir={exported}",
            ]
        )
    with pytest.raises(ValueError, match="untrusted executable target"):
        PreparationService(checkout, tmp_path / "store").prepare(overrides)


def test_exported_architecture_safe_references_resolve(checkout: Path) -> None:
    resolved = _artifact_architecture(
        {
            "architecture": "${other_arch}",
            "other_arch": {
                "_target_": "pyurbanair.static_parameters.Normal",
                "mean": "${value}",
                "std": 1,
            },
            "value": 3,
        },
        checkout,
    )
    assert resolved["mean"] == 3


def test_exported_architecture_resolvers_rejected_before_resolution(
    checkout: Path,
) -> None:
    with pytest.raises(ValueError, match="custom resolvers"):
        _artifact_architecture({"architecture": "${oc.create:bad}"}, checkout)


def test_preparation_creates_private_store(checkout: Path, tmp_path: Path) -> None:
    store = tmp_path / "private_store"
    PreparationService(checkout, store).prepare()
    assert store.stat().st_mode & 0o777 == 0o700


def test_nonprivate_store_is_rejected_without_chmod(
    checkout: Path, tmp_path: Path
) -> None:
    store = tmp_path / "shared_store"
    store.mkdir(mode=0o755)
    store.chmod(0o755)
    with pytest.raises(ValueError, match="private directory"):
        PreparationService(checkout, store).prepare()
    assert store.stat().st_mode & 0o777 == 0o755
    assert not (store / "plans").exists()


def test_private_directory_rejects_symlink(tmp_path: Path) -> None:
    actual = tmp_path / "actual"
    actual.mkdir(mode=0o700)
    alias = tmp_path / "alias"
    alias.symlink_to(actual, target_is_directory=True)
    with pytest.raises(ValueError, match="private directory"):
        ensure_private_directory(alias)


@pytest.mark.parametrize(  # type: ignore[misc]
    "override",
    [
        "model.forward_model.matlab_bin=/tmp/custom-program",
        "+nested={matlab_bin:/tmp/custom-program}",
        "+nested={_target_:[os.system]}",
    ],
)
def test_executable_path_and_nonstring_targets_are_rejected(
    checkout: Path, override: str
) -> None:
    with pytest.raises(ValueError):
        compose_forward_config(checkout, [override])


@pytest.mark.parametrize("backend", ["pylbm", "pyudales", "pypalm", "neural_surrogate"])  # type: ignore[misc]
def test_ensemble_and_forward_result_overrides_are_owned(
    checkout: Path, tmp_path: Path, backend: str
) -> None:
    config = compose_forward_config(
        checkout,
        [
            f"model={backend}",
            "+model.ensemble_model.temp_dir=/outside/scratch",
            "+model.ensemble_model.results_dir=/outside/results",
            "+model.forward_model.results_dir=/outside/forward",
        ],
    )["config"]
    root = tmp_path / "run"
    bound, _ = bind_job_paths(config, root)
    for component, field in (
        ("ensemble_model", "temp_dir"),
        ("ensemble_model", "results_dir"),
        ("forward_model", "results_dir"),
    ):
        assert Path(bound["model"][component][field]).is_relative_to(root)


def test_nested_constructor_outside_model_is_still_bound(
    checkout: Path, tmp_path: Path
) -> None:
    config = compose_forward_config(checkout)["config"]
    config["extra"] = [
        {
            "_target_": "pyudales.forward_model.ForwardModel",
            "temp_dir": "/outside",
            "output_dir": "/outside/output",
        }
    ]
    root = tmp_path / "run"
    bound, _ = bind_job_paths(config, root)
    assert Path(bound["extra"][0]["output_dir"]).is_relative_to(root)


@pytest.mark.parametrize(  # type: ignore[misc]
    "overrides,field",
    [
        (
            ["model.ensemble_model.ensemble_size=1000000"],
            "model.ensemble_model.ensemble_size",
        ),
        (["+run.max_retained_bytes=1"], "run.max_retained_bytes"),
        (["model=pypalm", "model.compile=true"], "model.compile"),
        (["model=pylbm", "model.forward_model.cuda=true"], "model.forward_model.cuda"),
    ],
)
def test_readiness_and_budget_guards(
    checkout: Path, tmp_path: Path, overrides: list[str], field: str
) -> None:
    plan = PreparationService(checkout, tmp_path / "store").prepare(overrides)
    assert any(issue["field"] == field for issue in plan["validation"]["issues"])
