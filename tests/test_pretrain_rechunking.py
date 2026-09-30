"""Pretraining prepares its complete dataset before constructing training objects."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest
from omegaconf import DictConfig, OmegaConf

torch = pytest.importorskip("torch")

_SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts/neural_surrogate/pretrain_autoencoder.py"
)


_Preparations = list[tuple[Path, Path, dict[str, Any]]]


@pytest.fixture
def pretrain() -> ModuleType:
    spec = importlib.util.spec_from_file_location("pretrain_rechunking_test", _SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def cfg(tmp_path: Path) -> DictConfig:
    return OmegaConf.create(
        {
            "dataset": {
                "root_dir": str(tmp_path / "source"),
                "dtype": "float32",
                "state_vars": ["u", "v", "w"],
                "time_stride": 1,
            },
            "dataloader": {"num_workers": 0, "drop_last": False},
            "batch_sampler": {"drop_last": False},
            "architecture": {"_target_": "unused.model"},
            "prechunk": {
                "enabled": True,
                "output_root": str(tmp_path / "prepared"),
                "time_chunk": 1,
                "spatial_chunks": [16, 64, 64],
                "compression_level": 1,
                "max_buffer_mb": 64,
                "prepare_only": False,
            },
        }
    )


@pytest.fixture
def preparations(monkeypatch: pytest.MonkeyPatch) -> _Preparations:
    from neural_surrogates.datasets import rechunk

    calls: _Preparations = []

    def prepare(
        source_root: str | Path, output_root: str | Path, **kwargs: Any
    ) -> Path:
        calls.append((Path(source_root), Path(output_root), kwargs))
        return Path(output_root)

    monkeypatch.setattr(rechunk, "prepare_rechunked_dataset", prepare)
    return calls


@pytest.mark.parametrize("block", [None, {"enabled": False}])
def test_disabled_or_absent_prechunk_is_noop(
    pretrain: ModuleType, block: dict[str, bool] | None
) -> None:
    cfg = OmegaConf.create({} if block is None else {"prechunk": block})
    assert pretrain._prepare_dataset_root(cfg) is None


def test_prepare_only_requires_enabled_prechunk(
    pretrain: ModuleType, cfg: DictConfig, preparations: _Preparations
) -> None:
    cfg.prechunk.enabled = False
    cfg.prechunk.prepare_only = True
    with pytest.raises(ValueError, match="enabled"):
        pretrain._prepare_dataset_root(cfg)
    assert preparations == []


@pytest.mark.parametrize("output_root", [None, ""])
def test_enabled_prechunk_requires_destination(
    pretrain: ModuleType,
    cfg: DictConfig,
    preparations: _Preparations,
    output_root: str | None,
) -> None:
    cfg.prechunk.output_root = output_root
    with pytest.raises(ValueError, match="output_root"):
        pretrain._prepare_dataset_root(cfg)
    assert preparations == []


def test_preparation_passes_layout_without_replacing_source(
    pretrain: ModuleType, cfg: DictConfig, preparations: _Preparations
) -> None:
    original_source = cfg.dataset.root_dir
    cfg.prechunk.time_chunk = 4
    cfg.prechunk.spatial_chunks = [8, 32, 48]
    cfg.prechunk.compression_level = 0
    cfg.prechunk.max_buffer_mb = 12

    assert pretrain._prepare_dataset_root(cfg) == Path(cfg.prechunk.output_root)
    source, destination, options = preparations[0]
    assert source == Path(original_source)
    assert destination == Path(cfg.prechunk.output_root)
    assert options == {
        "time_chunk": 4,
        "spatial_chunks": [8, 32, 48],
        "compression_level": 0,
        "max_buffer_mb": 12,
    }
    assert cfg.dataset.root_dir == original_source


def test_prepare_only_never_constructs_training_objects(
    pretrain: ModuleType,
    cfg: DictConfig,
    preparations: _Preparations,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg.prechunk.prepare_only = True

    def unexpected(*args: Any, **kwargs: Any) -> None:
        pytest.fail("prepare_only constructed a dataset, loader, or model")

    monkeypatch.setattr(pretrain, "instantiate", unexpected)
    monkeypatch.setattr(pretrain, "build_loader", unexpected)
    pretrain.run(cfg)
    assert len(preparations) == 1


@pytest.mark.parametrize("enabled", [False, True])
def test_both_splits_use_selected_root_before_model_construction(
    pretrain: ModuleType,
    cfg: DictConfig,
    preparations: _Preparations,
    monkeypatch: pytest.MonkeyPatch,
    enabled: bool,
) -> None:
    cfg.prechunk.enabled = enabled
    source = cfg.dataset.root_dir
    splits: list[dict[str, Any]] = []
    loaders: list[tuple[str, bool]] = []

    class ModelConstructionReached(Exception):
        pass

    def instantiate(config: DictConfig, **kwargs: Any) -> Any:
        if config is cfg.architecture:
            raise ModelConstructionReached
        assert config is cfg.dataset
        assert len(preparations) == int(enabled)
        splits.append(kwargs)
        return kwargs["split"]

    def build_loader(config: DictConfig, dataset: str, *, train: bool) -> object:
        loaders.append((dataset, train))
        return object()

    monkeypatch.setattr(pretrain, "instantiate", instantiate)
    monkeypatch.setattr(pretrain, "build_loader", build_loader)
    with pytest.raises(ModelConstructionReached):
        pretrain.run(cfg)

    assert [split["split"] for split in splits] == ["train", "val"]
    assert loaders == [("train", True), ("val", False)]
    for split in splits:
        assert split["dtype"] == torch.float32
        if enabled:
            assert Path(split["root_dir"]) == Path(cfg.prechunk.output_root)
        else:
            assert "root_dir" not in split
    assert cfg.dataset.root_dir == source


@pytest.mark.parametrize("stride", [0, 2, 3])
def test_coverage_guard_rejects_skipped_frames(
    pretrain: ModuleType, cfg: DictConfig, preparations: _Preparations, stride: int
) -> None:
    cfg.dataset.time_stride = stride
    with pytest.raises(ValueError, match="time_stride"):
        pretrain._prepare_dataset_root(cfg)
    assert preparations == []


@pytest.mark.parametrize("use_sampler", [False, True])
def test_coverage_guard_rejects_dropped_tail_batches(
    pretrain: ModuleType,
    cfg: DictConfig,
    preparations: _Preparations,
    use_sampler: bool,
) -> None:
    if use_sampler:
        cfg.batch_sampler.drop_last = True
    else:
        cfg.batch_sampler = None
        cfg.dataloader.drop_last = True
    with pytest.raises(ValueError, match="drop_last"):
        pretrain._prepare_dataset_root(cfg)
    assert preparations == []


def test_sampler_overrides_inactive_dataloader_drop_last(
    pretrain: ModuleType, cfg: DictConfig, preparations: _Preparations
) -> None:
    cfg.dataloader.drop_last = True
    assert pretrain._prepare_dataset_root(cfg) == Path(cfg.prechunk.output_root)
    assert len(preparations) == 1
