"""Lossless snapshot rechunking and complete training-sample coverage."""

import os
from pathlib import Path
from typing import Any, cast

import netCDF4
import numpy as np
import pytest
import torch
import xarray as xr
from neural_surrogates.datasets.rechunk import prepare_rechunked_dataset
from neural_surrogates.datasets.sampler import TrajectoryBatchSampler
from neural_surrogates.datasets.snapshot import SnapshotDataset


def _write_trajectory(path: Path, frames: int, offset: int = 0) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with netCDF4.Dataset(path, "w") as ds:
        ds.title = "rechunking fixture"
        for name, size in (("time", frames), ("zt", 3), ("yt", 5), ("xt", 7)):
            ds.createDimension(name, size)
            coord = ds.createVariable(name, "f8", (name,))
            coord[:] = np.arange(size)
            coord.units = "seconds since 2000-01-01" if name == "time" else "m"
        values = np.arange(frames * 3 * 5 * 7, dtype=np.float32).reshape(
            frames, 3, 5, 7
        )
        for index, name in enumerate(("u", "v", "w")):
            var = ds.createVariable(
                name,
                "f4",
                ("time", "zt", "yt", "xt"),
                zlib=True,
                chunksizes=(min(frames, 3), 3, 5, 7),
                fill_value=-9999.0,
            )
            var.units = "m/s"
            var[:] = values + offset + index
        mask = ds.createVariable("blanking", "i1", ("zt", "yt", "xt"))
        mask[:] = 0
        mask[0, 0, 0] = 1
        packed = ds.createVariable(
            "packed", "i2", ("time",), fill_value=np.int16(-32768)
        )
        packed.scale_factor = 0.25
        packed.add_offset = 1.0
        packed.set_auto_maskandscale(False)
        packed[:] = np.arange(frames, dtype=np.int16)
        packed[-1] = -32768
        auxiliary = ds.createVariable("auxiliary", "f4", ("time",))
        auxiliary[:] = np.arange(frames, dtype=np.float32) + 0.123456
        auxiliary[-1] = np.nan
        auxiliary.least_significant_digit = 2
        scalar = ds.createVariable("reference", "f8")
        scalar.assignValue(1.25)


@pytest.fixture
def source_root(tmp_path: Path) -> Path:
    root = tmp_path / "source"
    for split, lengths in (("train", (5, 7)), ("val", (3,)), ("test", (2,))):
        for index, frames in enumerate(lengths):
            _write_trajectory(
                root / "state" / split / f"sample_{index}.nc", frames, index * 100
            )
    return root


def test_rechunk_preserves_every_file_value_and_temporal_remainder(
    source_root: Path, tmp_path: Path
) -> None:
    output = prepare_rechunked_dataset(
        source_root,
        tmp_path / "prepared",
        time_chunk=2,
        spatial_chunks=(2, 3, 4),
        max_buffer_mb=1,
    )
    relative_files = sorted(
        p.relative_to(source_root) for p in source_root.rglob("*.nc")
    )
    assert sorted(p.relative_to(output) for p in output.rglob("*.nc")) == relative_files
    for relative in relative_files:
        for decode in (False, True):
            with xr.open_dataset(source_root / relative, decode_cf=decode) as original:
                with xr.open_dataset(output / relative, decode_cf=decode) as converted:
                    xr.testing.assert_identical(original, converted)
        with netCDF4.Dataset(output / relative) as ds:
            assert ds["u"].chunking() == [2, 2, 3, 4]
            assert ds["u"].filters()["zlib"]
            assert ds["packed"].dtype == np.dtype("int16")


# 0.0005 MB holds less than one frame: blocks then split an output chunk.
@pytest.mark.parametrize("max_buffer_mb", [1, 0.0005])
def test_whole_frame_rechunk_preserves_every_value(
    source_root: Path, tmp_path: Path, max_buffer_mb: float
) -> None:
    output = prepare_rechunked_dataset(
        source_root,
        tmp_path / "prepared",
        spatial_chunks=None,
        max_buffer_mb=max_buffer_mb,
    )
    for path in source_root.rglob("*.nc"):
        relative = path.relative_to(source_root)
        with xr.open_dataset(path, decode_cf=False) as original:
            with xr.open_dataset(output / relative, decode_cf=False) as converted:
                xr.testing.assert_identical(original, converted)
        with netCDF4.Dataset(output / relative) as ds:
            assert ds["u"].chunking() == [1, 3, 5, 7]
            assert ds["blanking"].chunking() == [3, 5, 7]


@pytest.mark.parametrize("crop_size", [None, 2])
@pytest.mark.parametrize("time_stride", [1, 2])
def test_rechunk_keeps_sample_index_crops_and_all_sampler_remainders(
    source_root: Path, tmp_path: Path, crop_size: int | None, time_stride: int
) -> None:
    output = prepare_rechunked_dataset(source_root, tmp_path / "prepared", time_chunk=2)
    datasets = [
        SnapshotDataset(
            root,
            "train",
            time_stride=time_stride,
            random_crop_size=crop_size,
            sdf_features="sdf",
        )
        for root in (source_root, output)
    ]
    original, converted = datasets
    expected_index = [
        (traj, frame)
        for traj, frames in enumerate((5, 7))
        for frame in range(0, frames, time_stride)
    ]
    assert original.sample_index == converted.sample_index == expected_index
    batches = []
    for dataset in datasets:
        # Shared sampler accepts SnapshotDataset through its dataset interface.
        sampler = TrajectoryBatchSampler(
            cast(Any, dataset), batch_size=4, shuffle=True, drop_last=False, seed=12
        )
        batches.append(list(sampler))
    assert batches[0] == batches[1]
    indices = [index for batch in batches[1] for index in batch]
    assert sorted(indices) == list(range(len(expected_index)))
    try:
        for index in indices:
            torch.manual_seed(100 + index)
            expected = original[index]
            torch.manual_seed(100 + index)
            actual = converted[index]
            assert expected.keys() == actual.keys()
            for name in expected:
                torch.testing.assert_close(actual[name], expected[name], rtol=0, atol=0)
    finally:
        for dataset in datasets:
            for ds in (dataset._state_cache or {}).values():
                ds.close()


def test_completed_cache_reused_without_rewriting(
    source_root: Path, tmp_path: Path
) -> None:
    output = prepare_rechunked_dataset(source_root, tmp_path / "prepared")
    before = {p: p.stat().st_mtime_ns for p in output.rglob("*.nc")}
    assert prepare_rechunked_dataset(source_root, output) == output
    assert before == {p: p.stat().st_mtime_ns for p in output.rglob("*.nc")}


@pytest.mark.parametrize("change", ["source", "added_source", "options", "output"])
def test_cache_drift_is_refused(source_root: Path, tmp_path: Path, change: str) -> None:
    output = prepare_rechunked_dataset(source_root, tmp_path / "prepared")
    kwargs: dict[str, Any] = {}
    if change in {"source", "output"}:
        root = source_root if change == "source" else output
        path = root / "state/train/sample_0.nc"
        stat = path.stat()
        os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000))
    elif change == "added_source":
        _write_trajectory(source_root / "state/train/sample_2.nc", 2)
    else:
        kwargs["time_chunk"] = 2
    with pytest.raises(ValueError):
        prepare_rechunked_dataset(source_root, output, **kwargs)


@pytest.mark.parametrize("location", ["same", "inside", "parent"])
def test_overlapping_roots_are_refused(source_root: Path, location: str) -> None:
    output = {
        "same": source_root,
        "inside": source_root / "cache",
        "parent": source_root.parent,
    }[location]
    with pytest.raises(ValueError):
        prepare_rechunked_dataset(source_root, output)


@pytest.mark.parametrize(
    "options",
    [
        {"time_chunk": 0},
        {"spatial_chunks": (0, 2, 2)},
        {"spatial_chunks": (2, 2)},
        {"compression_level": 10},
        {"max_buffer_mb": 0},
    ],
)
def test_invalid_options_are_refused(
    source_root: Path, tmp_path: Path, options: dict[str, Any]
) -> None:
    with pytest.raises(ValueError):
        prepare_rechunked_dataset(source_root, tmp_path / "prepared", **options)


def test_unowned_output_is_not_overwritten(source_root: Path, tmp_path: Path) -> None:
    output = tmp_path / "prepared"
    output.mkdir()
    marker = output / "keep.txt"
    marker.write_text("unrelated data")
    with pytest.raises(ValueError):
        prepare_rechunked_dataset(source_root, output)
    assert marker.read_text() == "unrelated data"


def test_interrupted_preparation_recovers_without_rewriting_completed_files(
    source_root: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from neural_surrogates.datasets import rechunk

    output = tmp_path / "prepared"
    original_copy = rechunk._copy_file
    calls: list[Path] = []

    def interrupt_second_file(source: Path, destination: Path, **kwargs: Any) -> int:
        calls.append(destination)
        if len(calls) == 2:
            destination.write_bytes(b"interrupted NetCDF write")
            raise OSError("simulated interrupted write")
        return int(original_copy(source, destination, **kwargs))

    monkeypatch.setattr(rechunk, "_copy_file", interrupt_second_file)
    with pytest.raises(OSError, match="simulated interrupted"):
        prepare_rechunked_dataset(source_root, output)
    completed = list(output.rglob("*.nc"))
    assert len(completed) == 1
    before = completed[0].stat().st_mtime_ns
    assert len(list(output.rglob("*.rechunking"))) == 1
    monkeypatch.setattr(rechunk, "_copy_file", original_copy)
    assert prepare_rechunked_dataset(source_root, output) == output
    assert completed[0].stat().st_mtime_ns == before
    assert not list(output.rglob("*.rechunking"))
    for source in source_root.rglob("*.nc"):
        with xr.open_dataset(source) as original:
            with xr.open_dataset(output / source.relative_to(source_root)) as converted:
                xr.testing.assert_identical(original, converted)


def test_test_split_is_optional(tmp_path: Path) -> None:
    source = tmp_path / "source"
    for split in ("train", "val"):
        _write_trajectory(source / "state" / split / "sample_0.nc", 3)
    output = prepare_rechunked_dataset(source, tmp_path / "prepared")
    assert len(list(output.rglob("*.nc"))) == 2
    assert not (output / "state/test").exists()


def test_concurrent_preparation_is_refused(source_root: Path, tmp_path: Path) -> None:
    import fcntl

    output = tmp_path / "prepared"
    output.mkdir()
    with (output / ".rechunk.lock").open("a") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(RuntimeError, match="Another process"):
            prepare_rechunked_dataset(source_root, output)
    assert prepare_rechunked_dataset(source_root, output) == output


def test_prepared_cache_reused_after_training_writes_normalization_stats(
    source_root: Path, tmp_path: Path
) -> None:
    from neural_surrogates.training.data_utils import get_normalization_stats

    output = prepare_rechunked_dataset(source_root, tmp_path / "prepared")
    # Shared normalization reads the same attributes from both dataset classes.
    expected = get_normalization_stats(cast(Any, SnapshotDataset(source_root, "train")))
    actual = get_normalization_stats(cast(Any, SnapshotDataset(output, "train")))
    for left, right in zip(expected, actual):
        np.testing.assert_array_equal(left, right)
    assert (output / "normalization_stats/train.npz").is_file()
    assert prepare_rechunked_dataset(source_root, output) == output


def test_normalization_stats_stream_from_source_root(
    source_root: Path, tmp_path: Path
) -> None:
    from neural_surrogates.training.data_utils import (
        _compute_normalization_stats,
        get_normalization_stats,
    )

    output = prepare_rechunked_dataset(source_root, tmp_path / "prepared")
    # A copy is lossless, so stats from the copy and from the source agree:
    # stream from a different "source" to see that `source_root` is read.
    other = tmp_path / "other"
    for split, lengths in (("train", (5, 7)), ("val", (3,)), ("test", (2,))):
        for index, frames in enumerate(lengths):
            _write_trajectory(
                other / "state" / split / f"sample_{index}.nc", frames, 1000 + index
            )
    copy = cast(Any, SnapshotDataset(output, "train"))
    expected = _compute_normalization_stats(cast(Any, SnapshotDataset(other, "train")))
    from_other = _compute_normalization_stats(copy, source_root=other)
    for left, right in zip(expected, from_other):
        np.testing.assert_array_equal(left, right)
    assert not np.array_equal(_compute_normalization_stats(copy)[0], expected[0])

    actual = get_normalization_stats(copy, source_root=source_root)
    reference = get_normalization_stats(
        cast(Any, SnapshotDataset(source_root, "train"))
    )
    for left, right in zip(reference, actual):
        np.testing.assert_array_equal(left, right)
    assert (output / "normalization_stats/train.npz").is_file()


def test_complete_cache_validates_while_another_process_holds_the_lock(
    source_root: Path, tmp_path: Path
) -> None:
    import fcntl

    output = prepare_rechunked_dataset(source_root, tmp_path / "prepared")
    manifest = output / ".rechunk-manifest.json"
    before = manifest.stat().st_mtime_ns
    with (output / ".rechunk.lock").open("a") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert prepare_rechunked_dataset(source_root, output) == output
        assert prepare_rechunked_dataset(source_root, output) == output
    assert manifest.stat().st_mtime_ns == before


def test_read_only_complete_cache_validates(source_root: Path, tmp_path: Path) -> None:
    output = prepare_rechunked_dataset(source_root, tmp_path / "prepared")
    # Everything read-only (chmod moves the ctime, which the check ignores),
    # the source's permissions changed too.
    paths = sorted(output.rglob("*"), key=lambda p: len(p.parts), reverse=True)
    modes = {p: p.stat().st_mode for p in [*paths, output]}
    try:
        for path in modes:
            path.chmod(0o555 if path.is_dir() else 0o444)
        for path in source_root.rglob("*.nc"):
            path.chmod(0o640)
        assert prepare_rechunked_dataset(source_root, output) == output
    finally:
        for path in reversed(list(modes)):
            path.chmod(modes[path])
    assert prepare_rechunked_dataset(source_root, output) == output


def test_manifest_with_ctime_and_stats_temporaries_still_validates(
    source_root: Path, tmp_path: Path
) -> None:
    """Manifests written before ctime was dropped record it; a crashed stats
    write leaves a temporary file. Neither invalidates the copy."""
    import json

    output = prepare_rechunked_dataset(source_root, tmp_path / "prepared")
    path = output / ".rechunk-manifest.json"
    manifest = json.loads(path.read_text())
    for record in [*manifest["sources"].values(), *manifest["files"].values()]:
        (record.get("output") or record)["ctime_ns"] = 1
    path.write_text(json.dumps(manifest))
    (output / "normalization_stats").mkdir()
    (output / "normalization_stats" / ".train.npz.123.tmp").write_bytes(b"x")
    with (output / ".rechunk.lock").open("a") as lock:
        import fcntl

        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert prepare_rechunked_dataset(source_root, output) == output
