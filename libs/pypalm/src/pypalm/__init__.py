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
import shlex
import shutil
import subprocess
import tarfile
import urllib.request
from typing import Iterator

from pyurbanair.utils.solver_process import log_tail
from pyurbanair.utils.toolchain import apple_linker_flags

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


def _install_environment() -> dict[str, str]:
    """The environment ``install_palm.sh`` builds PALM in.

    - ``CMAKE_PREFIX_PATH`` puts the pixi env first, so PALM's CMake finds its
      FFTW and NetCDF instead of a system copy (e.g. Homebrew's FFTW).
    - ``HOME`` keeps the installer's ``~/.palm/palmtest*.yml`` (and pip caches)
      inside ``palm_model_system`` instead of the user's home.
    - On macOS, Apple's linker (``apple_linker_flags``) for both of PALM's link
      paths: ``LDFLAGS`` for its CMake checks, ``OMPI_LDFLAGS`` for the links
      through ``mpif90``. ``OMPI_LDFLAGS`` replaces the wrapper's own linker
      flags, so those are kept. The header padding leaves room for the
      ``install_name_tool`` fix-up in ``install_palm.sh``.
    """
    env = os.environ.copy()
    prefixes = [env.get("CONDA_PREFIX", ""), env.get("CMAKE_PREFIX_PATH", "")]
    env["CMAKE_PREFIX_PATH"] = os.pathsep.join(p for p in prefixes if p)
    env["HOME"] = str(PALM_MODEL_SYSTEM_PATH)
    apple = apple_linker_flags("mpif90", env)
    if apple:

        def showme(part: str) -> list[str]:
            command = ["mpif90", f"--showme:{part}"]
            output = subprocess.run(
                command, capture_output=True, text=True, check=True, env=env
            ).stdout
            return shlex.split(output)

        compile_flags = set(showme("compile"))
        wrapper = [
            flag
            for flag in showme("link")
            if flag not in compile_flags and not flag.startswith("-l")
        ]
        link = [*apple, "-Wl,-headerpad_max_install_names"]
        env["LDFLAGS"] = " ".join([env.get("LDFLAGS", "").strip(), *link]).strip()
        env["OMPI_LDFLAGS"] = " ".join([*wrapper, *link])
    return env


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
                env=_install_environment(),
            )
        if result.returncode != 0:
            # A binary from a failed install (e.g. its macOS fix-up) is unusable.
            PALM_BINARY.unlink(missing_ok=True)
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
