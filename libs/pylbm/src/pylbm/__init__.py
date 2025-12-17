"""pylbm - Python wrapper for LBM."""

import os
import pathlib
import subprocess
import sys

__version__ = "0.1.0"

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

if _gitmodules_path.exists():
    try:
        gitmodules_content = _gitmodules_path.read_text()
        print(f"Reading .gitmodules from: {_gitmodules_path}", file=sys.stderr)
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
                        print(
                            f"Found LBM path in .gitmodules: {submodule_path} -> {_lbm_path}",
                            file=sys.stderr,
                        )
                elif stripped.startswith("url = ") or stripped.startswith("url="):
                    if "=" in stripped:
                        _lbm_url = stripped.split("=", 1)[1].strip()
                        print(
                            f"Found LBM URL in .gitmodules: {_lbm_url}",
                            file=sys.stderr,
                        )
    except Exception as e:
        print(f"Error reading .gitmodules: {e}", file=sys.stderr)
        import traceback

        traceback.print_exc(file=sys.stderr)
else:
    print(f".gitmodules not found at: {_gitmodules_path}", file=sys.stderr)

# Initialize git submodule from .gitmodules
_repo_just_downloaded = False
if _lbm_path:
    # Check if submodule needs to be initialized
    # Repository is considered downloaded if it exists, has content, and is a valid git repo
    is_repo_downloaded = (
        _lbm_path.exists()
        and any(_lbm_path.iterdir())
        and (_lbm_path / ".git").exists()
    )
    needs_init = not is_repo_downloaded

    if needs_init:
        print("Initializing LBM git submodule...", file=sys.stderr)
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
                print("LBM submodule initialized successfully.", file=sys.stderr)
                submodule_success = True
                _repo_just_downloaded = True
            else:
                print(
                    f"Git submodule init failed (code {result.returncode}), trying direct clone...",
                    file=sys.stderr,
                )
        except Exception as e:
            print(
                f"Exception during submodule init: {e}, trying direct clone...",
                file=sys.stderr,
            )

        # Fallback to direct clone if submodule failed
        if not submodule_success and _lbm_url:
            try:
                print(f"Cloning LBM from {_lbm_url}...", file=sys.stderr)
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
                    print("LBM cloned successfully.", file=sys.stderr)
                    _repo_just_downloaded = True
                else:
                    print(
                        f"Warning: git clone failed (code {result.returncode})",
                        file=sys.stderr,
                    )
                    if result.stderr:
                        print(f"Error: {result.stderr}", file=sys.stderr)
            except Exception as e:
                print(f"Exception during git clone: {e}", file=sys.stderr)
    else:
        print(
            "LBM repository already downloaded, skipping initialization.",
            file=sys.stderr,
        )

    # Set LBM_PATH from gitmodules path (always set it)
    LBM_PATH = _lbm_path.resolve()
    print(f"LBM_PATH set to: {LBM_PATH}", file=sys.stderr)
else:
    print("Warning: Could not find LBM path in .gitmodules", file=sys.stderr)
