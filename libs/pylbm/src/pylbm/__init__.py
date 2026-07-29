"""pylbm - Python wrapper for LBM."""

import logging
import os
import pathlib
import subprocess
import sys
from typing import Optional

__version__ = "0.1.0"

logger = logging.getLogger(__name__)

# Get paths
_project_root = pathlib.Path(__file__).parent.parent.parent
# Find repo root by looking for .git directory or .gitmodules file
_repo_root = _project_root
while _repo_root != _repo_root.parent:
    if (_repo_root / ".git").exists() or (_repo_root / ".gitmodules").exists():
        break
    _repo_root = _repo_root.parent
_gitmodules_path = _repo_root / ".gitmodules"

# Parse .gitmodules to get LBM path and URL
LBM_PATH = None
_lbm_path = None
_lbm_url = None

# Per-job isolation hook. When PYLBM_LBM_PATH is set, use that copy of the LBM
# tree directly and skip the shared in-repo submodule discovery/clone. The LBM
# build mutates its own source tree (mod_dimensions.F90, generated m_*.F90, the
# makefile) and writes object files plus the boltzmann binary in place, so two
# processes building the shared submodule concurrently corrupt each other. HPC
# jobs give each run a private copy (e.g. rsynced onto node-local scratch) and
# point pylbm at it via this variable. Unset -> unchanged single-process default.
_lbm_path_override = os.environ.get("PYLBM_LBM_PATH")

# True when LBM_PATH already is a caller-owned private tree, so the build runs in
# place. False -> the shared submodule, which is treated as read-only and
# mirrored into a per-run build tree (see utils/build_tree_utils.py).
LBM_PATH_IS_ISOLATED = bool(_lbm_path_override)

if _lbm_path_override:
    LBM_PATH = pathlib.Path(_lbm_path_override).resolve()
    logger.info("LBM_PATH overridden via PYLBM_LBM_PATH: %s", LBM_PATH)
elif _gitmodules_path.exists():
    try:
        gitmodules_content = _gitmodules_path.read_text()
        logger.info("Reading .gitmodules from: %s", _gitmodules_path)
        # Parse .gitmodules by sections
        in_lbm_section = False
        for line in gitmodules_content.splitlines():
            stripped = line.strip()
            # Check if we're entering the LBM submodule section
            if stripped.startswith("[submodule") and "lbm" in stripped.lower():
                in_lbm_section = True
            # Check if we're entering a different submodule section
            elif stripped.startswith("[submodule"):
                in_lbm_section = False
            # Parse path and URL only within LBM section
            elif in_lbm_section:
                if stripped.startswith("path = ") or stripped.startswith("path="):
                    if "=" in stripped:
                        submodule_path = stripped.split("=", 1)[1].strip()
                        _lbm_path = _repo_root / submodule_path
                        logger.info(
                            "Found LBM path in .gitmodules: %s -> %s",
                            submodule_path,
                            _lbm_path,
                        )
                elif stripped.startswith("url = ") or stripped.startswith("url="):
                    if "=" in stripped:
                        _lbm_url = stripped.split("=", 1)[1].strip()
                        logger.info("Found LBM URL in .gitmodules: %s", _lbm_url)
    except Exception as e:
        logger.exception("Error reading .gitmodules: %s", e)
else:
    logger.warning(".gitmodules not found at: %s", _gitmodules_path)


def _git(args: list[str], cwd: pathlib.Path) -> Optional[str]:
    """Run a git command, returning stripped stdout or None if it failed."""
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=str(cwd),
            check=False,
            capture_output=True,
            text=True,
        )
    except Exception as e:  # git missing, cwd gone, ...
        logger.debug("git %s failed in %s: %s", " ".join(args), cwd, e)
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip()


def _verify_submodule_pin(repo_root: pathlib.Path, lbm_path: pathlib.Path) -> None:
    """
    Check the LBM working copy against the commit the parent repo records.

    The submodule is a *build* tree as well as a source tree, so it drifts easily:
    a plain ``git clone`` fallback lands on the default branch tip, and a manual
    ``git checkout`` inside it is never noticed by the parent repo. A tree at the
    wrong commit fails in ways that look nothing like a version mismatch -- the
    Fortran makefile regenerates ``depends.file`` from the ``use`` statements in
    ``src/``, so a source file this wrapper injects a ``use`` for but which does
    not exist at that commit surfaces as a bare
    ``make: *** No rule to make target 'm_read_bathymetry.o'``.

    Policy: **report, do not touch.** This runs at import time, in every
    forkserver child as well as the parent, and a checkout there would silently
    detach a branch somebody deliberately checked out. Set
    ``PYLBM_AUTOSYNC_SUBMODULE=1`` to opt into self-healing; even then a *dirty*
    working copy is never touched, since it may hold work in progress.
    """
    # `<rev>:<path>` resolves from the repo root, so the path must be given
    # relative to it (git itself hints `HEAD:./LBM` for the repo-relative form).
    rel = lbm_path.relative_to(repo_root).as_posix()
    recorded = _git(["rev-parse", f"HEAD:{rel}"], cwd=repo_root)
    actual = _git(["rev-parse", "HEAD"], cwd=lbm_path)

    if recorded is None or actual is None:
        logger.warning(
            "Could not determine the LBM submodule commit (recorded=%s, actual=%s); "
            "building whatever is checked out at %s",
            recorded,
            actual,
            lbm_path,
        )
        return

    if recorded == actual:
        logger.info("LBM submodule at recorded commit %s", actual[:12])
        return

    dirty = _git(["status", "--porcelain", "--untracked-files=no"], cwd=lbm_path)
    autosync = os.environ.get("PYLBM_AUTOSYNC_SUBMODULE", "").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )

    if dirty or not autosync:
        logger.warning(
            "LBM submodule is at %s but the parent repo records %s%s. The build "
            "will use the checked-out sources, which may not match this wrapper "
            "(a source it injects a `use` for may not exist at that commit, which "
            "surfaces as a bare 'No rule to make target'). Resolve with:\n"
            "    git submodule update --checkout %s\n"
            "%s",
            actual[:12],
            recorded[:12],
            " and has uncommitted changes" if dirty else "",
            lbm_path,
            (
                "    (stash the submodule's local changes first)"
                if dirty
                else "    (or set PYLBM_AUTOSYNC_SUBMODULE=1 to do this automatically)"
            ),
        )
        return

    logger.warning(
        "LBM submodule is at %s but the parent repo records %s; the working copy "
        "is clean and PYLBM_AUTOSYNC_SUBMODULE is set, so checking it out onto "
        "the recorded commit.",
        actual[:12],
        recorded[:12],
    )
    if _git(["checkout", "--force", recorded], cwd=lbm_path) is None:
        logger.error(
            "Failed to check out LBM submodule commit %s in %s. Run "
            "'git submodule update --checkout %s' manually.",
            recorded,
            lbm_path,
            lbm_path,
        )
    else:
        logger.info("LBM submodule checked out at recorded commit %s", recorded[:12])


# Initialize git submodule from .gitmodules (skipped entirely when an explicit
# PYLBM_LBM_PATH override is in effect -- that copy is managed by the caller).
_repo_just_downloaded = False
if _lbm_path and not _lbm_path_override:
    # Check if submodule needs to be initialized
    # Repository is considered downloaded if it exists, has content, and is a valid git repo
    is_repo_downloaded = (
        _lbm_path.exists()
        and any(_lbm_path.iterdir())
        and (_lbm_path / ".git").exists()
    )
    needs_init = not is_repo_downloaded

    if needs_init:
        logger.info("Initializing LBM git submodule...")
        submodule_success = False

        # Try git submodule first
        try:
            result = subprocess.run(
                [
                    "git",
                    "submodule",
                    "update",
                    "--init",
                    "--recursive",
                    "libs/pylbm/LBM",
                ],
                cwd=str(_repo_root),
                check=False,
                capture_output=True,
                text=True,
            )
            if result.returncode == 0:
                logger.info("LBM submodule initialized successfully.")
                submodule_success = True
                _repo_just_downloaded = True
            else:
                logger.warning(
                    "Git submodule init failed (code %s), trying direct clone...",
                    result.returncode,
                )
        except Exception as e:
            logger.warning(
                "Exception during submodule init: %s, trying direct clone...", e
            )

        # Fallback to direct clone if submodule failed
        if not submodule_success and _lbm_url:
            try:
                logger.info("Cloning LBM from %s...", _lbm_url)
                # Remove empty directory if it exists
                if _lbm_path.exists():
                    import shutil

                    shutil.rmtree(_lbm_path)

                # Create parent directory
                _lbm_path.parent.mkdir(parents=True, exist_ok=True)

                # Clone the repository
                result = subprocess.run(
                    ["git", "clone", "--recursive", _lbm_url, str(_lbm_path)],
                    check=False,
                    capture_output=True,
                    text=True,
                )
                if result.returncode == 0:
                    logger.info("LBM cloned successfully.")
                    _repo_just_downloaded = True
                    # A bare clone lands on the default branch tip, which is not
                    # what the parent repo pins. _verify_submodule_pin below moves
                    # it onto the recorded commit (the clone is clean, so it will).
                else:
                    logger.warning("git clone failed (code %s)", result.returncode)
                    if result.stderr:
                        logger.error("Error: %s", result.stderr)
            except Exception as e:
                logger.exception("Exception during git clone: %s", e)
    else:
        logger.info("LBM repository already downloaded, skipping initialization.")

    # Whether it was just fetched or already on disk, make the commit we are
    # about to build explicit and reconcile it with the parent repo's pin.
    _verify_submodule_pin(_repo_root, _lbm_path)

    # Set LBM_PATH from gitmodules path (always set it)
    LBM_PATH = _lbm_path.resolve()
    logger.info("LBM_PATH set to: %s", LBM_PATH)
elif not _lbm_path_override:
    logger.warning("Could not find LBM path in .gitmodules")
