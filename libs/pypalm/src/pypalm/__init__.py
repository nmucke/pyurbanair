"""pypalm - Python wrapper for the PALM LES model.

On first import, downloads the PALM source tree (``palm_model_system``) as a
tarball and runs its ``install`` script against the active pixi environment.
After that, ``palmrun`` is available at ``palm_model_system/bin/palmrun``
regardless of whether PALM was installed system-wide. Unlike pylbm, PALM does
not need to be recompiled when the grid changes — nx/ny/nz are read from the
``_p3d`` namelist at runtime.
"""

import logging
import os
import pathlib
import shutil
import subprocess
import tarfile
import urllib.request

__version__ = "0.1.0"

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

_project_root = pathlib.Path(__file__).parent.parent.parent

LOCAL_EXECUTE_SCRIPT = _project_root / "shell_scripts" / "execute.sh"
LOCAL_INSTALL_SCRIPT = _project_root / "shell_scripts" / "install_palm.sh"

# Pin a tag here when a specific PALM release is required. "master" pulls the
# current tip; a named release tag (e.g. "v25.10") pins to that release.
PALM_VERSION = os.environ.get("PYPALM_PALM_VERSION", "master")
PALM_TARBALL_URL = (
    "https://gitlab.palm-model.org/releases/palm_model_system/-/archive/"
    f"{PALM_VERSION}/palm_model_system-{PALM_VERSION}.tar.gz"
)

PALM_MODEL_SYSTEM_PATH = _project_root / "palm_model_system"


def _download_tarball(url: str, dest: pathlib.Path) -> None:
    logger.info("Downloading PALM tarball from %s …", url)
    dest.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url) as resp, open(dest, "wb") as f:
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


def _ensure_palm_source() -> bool:
    """Ensure ``palm_model_system/`` contains the PALM source tree.

    Returns True when the source is present (either pre-existing or freshly
    downloaded); False when download failed. Network failure is logged but
    non-fatal — the user can still install PALM manually.
    """
    install_script = PALM_MODEL_SYSTEM_PATH / "install"
    if install_script.exists():
        return True

    logger.info("PALM source not found at %s — downloading …", PALM_MODEL_SYSTEM_PATH)
    tarball = _project_root / "palm_model_system.tar.gz"
    try:
        _download_tarball(PALM_TARBALL_URL, tarball)
        _extract_tarball(tarball, PALM_MODEL_SYSTEM_PATH)
    except Exception as e:
        logger.warning("Failed to download/extract PALM source: %s", e)
        return False
    finally:
        if tarball.exists():
            tarball.unlink()

    return install_script.exists()


def _install_palm() -> bool:
    """Run the PALM install script if ``bin/palmrun`` is missing.

    Returns True when ``bin/palmrun`` is present after this call. Install
    failure is logged but non-fatal.
    """
    palmrun = PALM_MODEL_SYSTEM_PATH / "bin" / "palmrun"
    if palmrun.exists():
        return True

    if not LOCAL_INSTALL_SCRIPT.exists():
        logger.warning("install_palm.sh missing at %s", LOCAL_INSTALL_SCRIPT)
        return False

    logger.info("Installing PALM — this may take several minutes …")
    try:
        subprocess.run(
            ["bash", str(LOCAL_INSTALL_SCRIPT), str(PALM_MODEL_SYSTEM_PATH)],
            check=True,
            env=os.environ.copy(),
        )
    except subprocess.CalledProcessError as e:
        logger.warning("PALM install failed (exit %s).", e.returncode)
        return False
    except Exception as e:
        logger.warning("PALM install raised: %s", e)
        return False

    return palmrun.exists()


def _resolve_palmrun() -> pathlib.Path | None:
    """Locate the palmrun executable.

    Preference order:
      1. ``PALM_BIN`` env var pointing at the palmrun script.
      2. ``palmrun`` on ``PATH``.
      3. ``$PALM_ROOT/bin/palmrun`` when ``PALM_ROOT`` is set.
      4. ``<libs/pypalm/palm_model_system>/bin/palmrun`` (auto-installed).
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

    bundled = PALM_MODEL_SYSTEM_PATH / "bin" / "palmrun"
    if bundled.exists():
        return bundled

    return None


# Skip auto-install when the user has opted out (e.g. CI that installs PALM
# separately) or already has palmrun on PATH / via env vars.
_preinstalled = (
    os.environ.get("PALM_BIN")
    or shutil.which("palmrun")
    or (os.environ.get("PALM_ROOT") and (pathlib.Path(os.environ["PALM_ROOT"]) / "bin" / "palmrun").exists())
)
_skip_autoinstall = os.environ.get("PYPALM_SKIP_AUTOINSTALL") == "1"

if not _preinstalled and not _skip_autoinstall:
    if _ensure_palm_source():
        _install_palm()

PALMRUN_BIN = _resolve_palmrun()
if PALMRUN_BIN is None:
    logger.info(
        "palmrun not found. Auto-install may have failed — set PALM_BIN/PALM_ROOT "
        "or install palm_model_system manually; ForwardModel.run() will raise."
    )
else:
    logger.info("palmrun resolved to: %s", PALMRUN_BIN)
