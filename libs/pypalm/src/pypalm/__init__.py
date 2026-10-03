"""pypalm - Python wrapper for the PALM LES model.

Importing never downloads or builds PALM. ``install_palm`` runs before every
PALM run: the first time it downloads the pinned PALM source tree
(``palm_model_system``) as a tarball and builds it with ``install_palm.sh``
against the active pixi environment; after that it only checks the binary.
Unlike pylbm, PALM does not need to be recompiled when the grid changes --
nx/ny/nz are read from the ``_p3d`` namelist at runtime.
"""

import contextlib
import fcntl
import logging
import os
import pathlib
import shutil
import subprocess
import tarfile
import urllib.request
from typing import Iterator

from pyurbanair.utils.solver_process import log_tail

__version__ = "0.1.0"

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

_project_root = pathlib.Path(__file__).parent.parent.parent

LOCAL_EXECUTE_SCRIPT = _project_root / "shell_scripts" / "execute.sh"
LOCAL_INSTALL_SCRIPT = _project_root / "shell_scripts" / "install_palm.sh"

# The PALM release this wrapper is tested against: the commit of tag v25.10.
# A commit, unlike a branch or tag, cannot move under a fresh clone.
PALM_COMMIT = "27f42650ec9ba885ddead8cbe50f736540d45e71"
PALM_TARBALL_URL = (
    "https://gitlab.palm-model.org/releases/palm_model_system/-/archive/"
    f"{PALM_COMMIT}/palm_model_system-{PALM_COMMIT}.tar.gz"
)

PALM_MODEL_SYSTEM_PATH = _project_root / "palm_model_system"
# What install_palm.sh builds. bin/palmrun is not proof of a build: that perl
# wrapper ships in the source tarball.
PALM_BINARY = PALM_MODEL_SYSTEM_PATH / "MAKE_DEPOSITORY_default" / "palm"
INSTALL_LOG = _project_root / "palm_install.log"


def _download_tarball(url: str, dest: pathlib.Path) -> None:
    logger.info("Downloading PALM tarball from %s …", url)
    dest.parent.mkdir(parents=True, exist_ok=True)
    # A stalled connection must fail, not hang the run: the timeout bounds each
    # blocking read, not the whole download.
    with urllib.request.urlopen(url, timeout=60) as resp, open(dest, "wb") as f:
        shutil.copyfileobj(resp, f)


def _extract_tarball(tarball: pathlib.Path, target: pathlib.Path) -> None:
    """Extract ``tarball`` so that its top-level contents land directly in ``target``.

    GitLab archives have a single top-level directory named
    ``palm_model_system-<ref>``; we strip that component.
    """
    target.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tarball, "r:gz") as tf:
        members = tf.getmembers()
        if not members:
            raise RuntimeError(f"PALM tarball {tarball} is empty")
        top = members[0].name.split("/", 1)[0]
        for m in members:
            if m.name == top:
                continue
            if not m.name.startswith(f"{top}/"):
                continue
            m.name = m.name[len(top) + 1 :]
            tf.extract(m, target)


@contextlib.contextmanager
def _lock() -> Iterator[None]:
    """Serialize installs across processes (e.g. parallel ensemble members)."""
    with open(_project_root / ".palm_install.lock", "a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def install_palm() -> None:
    """Make sure the pinned PALM is downloaded and built.

    A no-op once ``PALM_BINARY`` exists for ``PALM_COMMIT``. A tree from
    another commit is replaced. A failed download or build raises with the
    tail of ``INSTALL_LOG`` and leaves no binary, so the next call retries.
    """
    stamp = PALM_MODEL_SYSTEM_PATH / ".pypalm_commit"
    with _lock():
        pinned = stamp.is_file() and stamp.read_text().strip() == PALM_COMMIT
        if pinned and os.access(PALM_BINARY, os.X_OK):
            return
        if not pinned:
            shutil.rmtree(PALM_MODEL_SYSTEM_PATH, ignore_errors=True)
            logger.info("Downloading PALM %s from %s", PALM_COMMIT, PALM_TARBALL_URL)
            tarball = _project_root / "palm_model_system.tar.gz"
            try:
                _download_tarball(PALM_TARBALL_URL, tarball)
                _extract_tarball(tarball, PALM_MODEL_SYSTEM_PATH)
                stamp.write_text(PALM_COMMIT + "\n")
            except Exception as error:
                raise RuntimeError(
                    f"Could not download PALM from {PALM_TARBALL_URL}: {error}"
                ) from error
            finally:
                tarball.unlink(missing_ok=True)
        logger.info("Building PALM; this takes several minutes (log: %s)", INSTALL_LOG)
        with open(INSTALL_LOG, "w") as log:
            result = subprocess.run(
                ["bash", str(LOCAL_INSTALL_SCRIPT), str(PALM_MODEL_SYSTEM_PATH)],
                stdout=log,
                stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,
            )
        if not os.access(PALM_BINARY, os.X_OK):
            raise RuntimeError(
                f"PALM build failed (install_palm.sh exited {result.returncode}); "
                f"the next run retries it.\n{log_tail(INSTALL_LOG)}"
            )


def resolve_palmrun() -> pathlib.Path:
    """Locate the palmrun executable.

    Preference order:
      1. ``PALM_BIN`` env var pointing at the palmrun script.
      2. ``palmrun`` on ``PATH``.
      3. ``$PALM_ROOT/bin/palmrun`` when ``PALM_ROOT`` is set.
      4. ``<libs/pypalm/palm_model_system>/bin/palmrun`` (``install_palm``).
    """
    explicit = os.environ.get("PALM_BIN")
    if explicit and pathlib.Path(explicit).exists():
        return pathlib.Path(explicit)

    found = shutil.which("palmrun")
    if found:
        return pathlib.Path(found)

    palm_root = os.environ.get("PALM_ROOT")
    if palm_root:
        candidate = pathlib.Path(palm_root) / "bin" / "palmrun"
        if candidate.exists():
            return candidate

    return PALM_MODEL_SYSTEM_PATH / "bin" / "palmrun"
