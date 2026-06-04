"""Unit tests for the pyudales warmstart carry store.

These exercise the disk-level carry helpers (store/fetch/copy) without running
uDALES: the carry mechanism is plain file bookkeeping keyed by a member's
``DirectoryPaths``. The behaviour they pin down is what fixes the warm-start
turbulence re-spin-up bias — the real end-of-run restart (with its subgrid
fields) is persisted and reused instead of a zeroed cold-start template.
"""

import json
import pathlib

import pytest

from pyudales.utils.dir_utils import DirectoryPaths
from pyudales.utils.warm_start_utils import (
    CARRY_META_NAME,
    clear_carry,
    copy_carry,
    fetch_carry,
    store_carry,
)


def _make_dirs(root: pathlib.Path, exp: str, itot=8, jtot=6, ktot=4) -> DirectoryPaths:
    """Build a DirectoryPaths plus a minimal namoptions with a DOMAIN grid."""
    experiment_dir = root / "experiment" / exp
    output_dir = root / "outputs"
    (experiment_dir).mkdir(parents=True, exist_ok=True)
    (output_dir / exp).mkdir(parents=True, exist_ok=True)
    namoptions = experiment_dir / f"namoptions.{exp}"
    namoptions.write_text(
        "&DOMAIN\n"
        f"itot = {itot}\n"
        f"jtot = {jtot}\n"
        f"ktot = {ktot}\n"
        "/\n"
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


def _write_restart(dirs: DirectoryPaths, timestamp: int, content: bytes) -> pathlib.Path:
    """Drop a fake single-processor restart file in output_dir/{exp}."""
    name = f"initd{timestamp:08d}_000_000.{dirs.experiment_name}"
    path = dirs.output_dir / dirs.experiment_name / name
    path.write_bytes(content)
    return path


def test_store_and_fetch_roundtrip(tmp_path):
    dirs = _make_dirs(tmp_path, "000")
    _write_restart(dirs, 190, b"real-subgrid-fields")

    carry_dir = store_carry(dirs)
    assert carry_dir is not None and carry_dir.exists()
    meta = json.loads((carry_dir / CARRY_META_NAME).read_text())
    assert meta["grid"] == {"itot": 8, "jtot": 6, "ktot": 4}

    # Simulate the post-run output wipe; the carry must survive it.
    for f in (dirs.output_dir / "000").iterdir():
        f.unlink()

    restored = fetch_carry(dirs)
    assert restored is not None
    assert restored.read_bytes() == b"real-subgrid-fields"
    assert restored.parent == dirs.output_dir / "000"


def test_store_picks_newest_by_mtime_not_timestamp(tmp_path):
    """A leftover restored carry with a larger {ts} must not shadow the fresh one."""
    dirs = _make_dirs(tmp_path, "000")
    # Stale carry from a cold start that ran to sim+spinup (ts=190)...
    stale = _write_restart(dirs, 190, b"stale")
    # ...and the freshly written warm-window restart (ts=180), newer on disk.
    import os, time

    old = stale.stat().st_mtime - 100
    os.utime(stale, (old, old))
    fresh = _write_restart(dirs, 180, b"fresh")

    carry_dir = store_carry(dirs)
    meta = json.loads((carry_dir / CARRY_META_NAME).read_text())
    assert meta["files"] == [fresh.name]
    assert (carry_dir / fresh.name).read_bytes() == b"fresh"


def test_fetch_rejects_grid_mismatch(tmp_path):
    dirs = _make_dirs(tmp_path, "000", itot=8, jtot=6, ktot=4)
    _write_restart(dirs, 180, b"x")
    store_carry(dirs)

    # Rewrite namoptions with a different grid; the carry should be refused.
    (dirs.experiment_dir / "namoptions.000").write_text(
        "&DOMAIN\nitot = 16\njtot = 6\nktot = 4\n/\n"
    )
    assert fetch_carry(dirs) is None


def test_fetch_without_carry_returns_none(tmp_path):
    dirs = _make_dirs(tmp_path, "000")
    assert fetch_carry(dirs) is None


def test_copy_carry_renames_for_destination_member(tmp_path):
    src = _make_dirs(tmp_path, "003")
    dst = _make_dirs(tmp_path, "007")
    _write_restart(src, 180, b"donor-fields")
    store_carry(src)

    assert copy_carry(src, dst) is True

    restored = fetch_carry(dst)
    assert restored is not None
    assert restored.name == "initd00000180_000_000.007"
    assert restored.read_bytes() == b"donor-fields"


def test_clear_carry(tmp_path):
    dirs = _make_dirs(tmp_path, "000")
    _write_restart(dirs, 180, b"x")
    store_carry(dirs)
    clear_carry(dirs)
    assert fetch_carry(dirs) is None
