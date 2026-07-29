"""
Per-run LBM build tree.

The Fortran solver bakes its grid dimensions in at compile time, so this wrapper
has to rewrite ``mod_dimensions.F90`` and ``m_solid_objects_init.F90`` for every
experiment, and the makefile's own rules rewrite ``depends.file`` /
``source.files`` on every build. All four are files *tracked* by the LBM
submodule. Editing them in place meant:

* ``git status`` in the parent repo permanently showed ``M libs/pylbm/LBM``, and
  that dirty gitlink was repeatedly swept into unrelated commits (it broke CI);
* every run in the repo, whatever its config, contended for one source tree;
* the ``bin/boltzmann`` left behind carried whichever dimensions were baked in
  last, so a later run could silently pick up a binary built for another grid.

**This narrows the race, it does not eliminate it.** ``build_root`` defaults to
``<temp_dir>/lbm_build`` and ``temp_dir`` is ``paths.experiment_dir``, which is a
*fixed* path per entry point (``$PWD/.temp`` in ``conf/run_esmda.yaml``,
``$PWD/.temp_<model>`` for forward runs) -- not per-run. So two concurrent runs of
the same entry point still share one tree, now with this mirror's prune/refresh
pass racing the other process's ``make``. Concurrent e2e runs must still be
serialized; give genuinely parallel jobs distinct trees via ``PYLBM_BUILD_ROOT``
(or ``PYLBM_LBM_PATH``). What is fixed is that a run no longer corrupts the
*submodule*, and that differently-configured entry points no longer collide.

So the submodule is treated as read-only and every run builds in a private copy
of the tree under its own scratch directory. ``materialize_build_tree`` mirrors
``src/`` and the ``bin/`` helper scripts into that copy; the makefile's
``../build`` and ``../bin`` relative paths keep working because the copy has the
same layout. The submodule stays byte-identical to its checked-out commit.

``PYLBM_LBM_PATH`` still bypasses all of this: that copy is managed by the
caller (e.g. rsynced onto node-local scratch by a batch job), so it is used in
place and never mirrored.
"""

import hashlib
import json
import logging
import os
import pathlib
import shutil
from typing import Any, Optional, Union

logger = logging.getLogger(__name__)

# Everything in ``src/`` is mirrored except build output, so the copy cannot go
# wrong by omission -- the makefile pulls in more than the Fortran sources
# (``target.mk`` defines TARGET, ``source.files``/``depends.file`` are included).
# A deny-list of generated artifacts is the safe direction here: missing one of
# those merely leaves a stale file in scratch, whereas missing a *source* breaks
# the build in obscure ways (an empty TARGET silently turns `make install` into
# `cp ../build/ <bindir>`).
_GENERATED_SUFFIXES = (".o", ".mod", ".a", ".bak", ".old", ".smod")
_GENERATED_NAMES = ("tags", ".DS_Store")

# Sources the wrapper rewrites for the configured experiment. These are seeded
# from the submodule when absent and then never overwritten again.
#
# They must survive a *second* mirror pass. `get_lbm_directory_paths` runs once
# per `ForwardModel`, and `create_new_forward_model` calls it again for every
# ensemble member -- after the binary has been built. Members are deepcopies, so
# nothing regenerates these files on that second pass; refreshing them from the
# submodule would silently restore upstream's own grid (nx=200, ny=120, nz=2)
# under a binary compiled for the configured one, and `_prepare_warmstart` then
# fails with a state/grid shape mismatch.
_WRAPPER_OWNED = ("mod_dimensions.F90", "m_solid_objects_init.F90")

# Records what the last mirror pass put in the build tree, so pruning can remove
# exactly the files it placed there and nothing else. Without it, pruning is
# "delete anything absent from the submodule", which would also delete files the
# wrapper generates into the tree (e.g. a per-experiment `m_<experiment>.F90`
# from the legacy geometry path) on the second pass.
MIRROR_MANIFEST_NAME = ".pylbm_mirror_manifest.json"

# Helper scripts the makefile shells out to (``../bin/mkdepend.pl`` and friends).
_BIN_SUFFIXES = (".pl", ".sh")

BUILD_STAMP_NAME = ".pylbm_build_stamp.json"


def _is_mirrored(path: pathlib.Path) -> bool:
    """Whether a file in ``src/`` is a source to keep in sync (not build output)."""
    return path.suffix not in _GENERATED_SUFFIXES and path.name not in _GENERATED_NAMES


def _copy_if_changed(source: pathlib.Path, dest: pathlib.Path) -> bool:
    """Copy ``source`` onto ``dest`` unless they already match in size and mtime."""
    if dest.exists():
        src_stat, dest_stat = source.stat(), dest.stat()
        if (
            src_stat.st_size == dest_stat.st_size
            and src_stat.st_mtime_ns == dest_stat.st_mtime_ns
        ):
            return False
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, dest)
    return True


def materialize_build_tree(
    lbm_path: pathlib.Path,
    build_root: pathlib.Path,
) -> pathlib.Path:
    """
    Mirror the LBM sources into a private build tree and return its root.

    The mirror is incremental: only files that differ from the submodule in size
    or mtime are copied, so repeated runs cost a stat per source file.

    Two exceptions keep the mirror from fighting the wrapper:

    * the files in ``_WRAPPER_OWNED`` are seeded only when absent, never
      refreshed -- they carry the configured grid and geometry case;
    * pruning removes only files a previous pass actually mirrored (tracked in
      ``MIRROR_MANIFEST_NAME``), so files the wrapper generates into the tree are
      left alone.

    Both matter because this runs again for every ensemble member, after the
    binary is built.

    Args:
        lbm_path: Root of the LBM submodule checkout (contains ``src/``, ``bin/``).
        build_root: Directory to materialize the private tree into.

    Returns:
        ``build_root``, containing ``src/`` and ``bin/``.

    Raises:
        FileNotFoundError: If ``lbm_path/src`` does not exist.
    """
    source_src = lbm_path / "src"
    if not source_src.is_dir():
        raise FileNotFoundError(
            f"LBM sources not found at {source_src}. The submodule at {lbm_path} "
            "is missing or empty -- run 'git submodule update --init --recursive'."
        )

    dest_src = build_root / "src"
    dest_src.mkdir(parents=True, exist_ok=True)
    (build_root / "build").mkdir(parents=True, exist_ok=True)

    mirrored: set[str] = set()
    copied = 0
    seeded = 0
    for source_file in sorted(source_src.iterdir()):
        if not source_file.is_file() or not _is_mirrored(source_file):
            continue
        mirrored.add(source_file.name)
        dest_file = dest_src / source_file.name
        if source_file.name in _WRAPPER_OWNED:
            # Seed once, then leave to set_experiment / update_solid_objects_init.
            if not dest_file.exists():
                shutil.copy2(source_file, dest_file)
                seeded += 1
            continue
        if _copy_if_changed(source_file, dest_file):
            copied += 1

    # Drop sources that vanished upstream. Leaving one behind is not harmless:
    # the makefile rebuilds source.files from `ls *.F90`, so a stale module gets
    # compiled back into the binary. Only files a previous pass mirrored are
    # candidates, so anything the wrapper generated into the tree survives.
    previously_mirrored = _read_mirror_manifest(build_root)
    pruned = 0
    for name in sorted(previously_mirrored - mirrored):
        stale = dest_src / name
        if stale.is_file():
            stale.unlink()
            pruned += 1

    _write_mirror_manifest(build_root, mirrored)

    source_bin = lbm_path / "bin"
    if source_bin.is_dir():
        dest_bin = build_root / "bin"
        dest_bin.mkdir(parents=True, exist_ok=True)
        for script in sorted(source_bin.iterdir()):
            if script.is_file() and script.suffix in _BIN_SUFFIXES:
                dest_script = dest_bin / script.name
                if _copy_if_changed(script, dest_script):
                    dest_script.chmod(dest_script.stat().st_mode | 0o111)

    logger.info(
        "LBM build tree at %s (%d synced, %d seeded, %d pruned, source %s)",
        build_root,
        copied,
        seeded,
        pruned,
        lbm_path,
    )
    return build_root


def _read_mirror_manifest(build_root: pathlib.Path) -> set[str]:
    """Names the previous mirror pass placed in ``src/`` (empty if unknown)."""
    manifest = build_root / MIRROR_MANIFEST_NAME
    if not manifest.exists():
        return set()
    try:
        loaded = json.loads(manifest.read_text())
    except (OSError, ValueError):
        return set()
    if not isinstance(loaded, list):
        return set()
    return {name for name in loaded if isinstance(name, str)}


def _write_mirror_manifest(build_root: pathlib.Path, mirrored: set[str]) -> None:
    """Record what this mirror pass placed in ``src/``."""
    manifest = build_root / MIRROR_MANIFEST_NAME
    manifest.write_text(json.dumps(sorted(mirrored), indent=0))


def resolve_build_root(
    lbm_path: pathlib.Path,
    temp_dir: pathlib.Path,
    isolated: bool,
) -> pathlib.Path:
    """
    Decide where the build tree lives and make sure it is populated.

    Args:
        lbm_path: Root of the LBM checkout.
        temp_dir: The run's scratch directory (``paths.experiment_dir``).
        isolated: True when ``PYLBM_LBM_PATH`` already points at a private tree,
            in which case it is built in place and nothing is mirrored.

    Returns:
        The root of the tree to build in.
    """
    if isolated:
        return lbm_path

    override = os.environ.get("PYLBM_BUILD_ROOT")
    build_root = (
        pathlib.Path(override).expanduser().resolve()
        if override
        else pathlib.Path(temp_dir).resolve() / "lbm_build"
    )
    return materialize_build_tree(lbm_path=lbm_path, build_root=build_root)


def build_stamp_path(build_root: pathlib.Path) -> pathlib.Path:
    """Path of the stamp describing what the binary in this tree was built from."""
    return build_root / "bin" / BUILD_STAMP_NAME


def compute_build_signature(
    src_path: pathlib.Path,
    experiment_name: str,
    enable_cuda: Union[bool, str],
    enable_netcdf: bool,
) -> dict[str, Any]:
    """
    Describe the inputs that are baked into the compiled binary.

    ``mod_dimensions.F90`` carries the grid dimensions and
    ``m_solid_objects_init.F90`` the geometry case, both of which are compiled in.
    Hashing their contents catches a binary built for a different grid or a
    different experiment -- which otherwise produces wrong-shaped or all-NaN
    output with no error at all.

    ``enable_cuda`` is recorded for diagnostics only and is deliberately *not*
    part of the staleness check: it changes neither the numerics nor the array
    shapes, and ``cuda="auto"`` legitimately resolves differently per host.
    """
    digests: dict[str, str] = {}
    for name in ("mod_dimensions.F90", "m_solid_objects_init.F90"):
        candidate = src_path / name
        if candidate.exists():
            digests[name] = hashlib.sha256(candidate.read_bytes()).hexdigest()[:16]
    return {
        "experiment": experiment_name,
        "cuda": enable_cuda if isinstance(enable_cuda, bool) else str(enable_cuda),
        "netcdf": bool(enable_netcdf),
        "sources": digests,
    }


def write_build_stamp(
    build_root: pathlib.Path,
    signature: dict[str, Any],
) -> None:
    """Record what the binary now in ``build_root/bin`` was built from."""
    stamp = build_stamp_path(build_root)
    stamp.parent.mkdir(parents=True, exist_ok=True)
    stamp.write_text(json.dumps(signature, indent=2, sort_keys=True))


def read_build_stamp(build_root: pathlib.Path) -> Optional[dict[str, Any]]:
    """Read the build stamp, or None when it is missing or unreadable."""
    stamp = build_stamp_path(build_root)
    if not stamp.exists():
        return None
    try:
        loaded = json.loads(stamp.read_text())
    except (OSError, ValueError):
        return None
    return loaded if isinstance(loaded, dict) else None
