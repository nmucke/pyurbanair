"""pyudales - Python wrapper for uDALES."""

import logging
import pathlib
import subprocess
import sys

__version__ = "0.1.0"

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Get paths
_project_root = pathlib.Path(__file__).parent.parent.parent

# Path to local_execute.sh script (stored in pyudales package, not u-dales submodule)
# This ensures modifications to the script persist when the u-dales submodule is re-initialized
LOCAL_EXECUTE_SCRIPT = _project_root / "shell_scripts" / "local_execute.sh"
# Find repo root by looking for .git directory or .gitmodules file
_repo_root = _project_root
while _repo_root != _repo_root.parent:
    if (_repo_root / ".git").exists() or (_repo_root / ".gitmodules").exists():
        break
    _repo_root = _repo_root.parent
_gitmodules_path = _repo_root / ".gitmodules"

# Parse .gitmodules to get u-dales path and URL
UDALES_PATH = None
_udales_path = None
_udales_url = None
_udales_tag = None

if _gitmodules_path.exists():
    try:
        gitmodules_content = _gitmodules_path.read_text()
        print(f"Reading .gitmodules from: {_gitmodules_path}", file=sys.stderr)
        # Track which submodule section we're currently in
        current_submodule = None
        for line in gitmodules_content.splitlines():
            stripped = line.strip()
            # Check if this is a submodule section header
            if stripped.startswith("[submodule"):
                # Extract submodule name from [submodule "name"]
                if '"' in stripped:
                    current_submodule = stripped.split('"')[1]
                else:
                    current_submodule = None
            elif stripped.startswith("path = ") or stripped.startswith("path="):
                # Handle both "path = " and "path=" formats
                if "=" in stripped:
                    submodule_path = stripped.split("=", 1)[1].strip()
                    if "u-dales" in submodule_path:
                        _udales_path = _repo_root / submodule_path
                        logger.info(
                            f"Found u-dales path in .gitmodules: {submodule_path} -> {_udales_path}",
                        )
            elif stripped.startswith("url = ") or stripped.startswith("url="):
                # Only set URL if we're in the u-dales submodule section
                if (
                    "=" in stripped
                    and current_submodule
                    and "u-dales" in current_submodule
                ):
                    _udales_url = stripped.split("=", 1)[1].strip()
                    logger.info(
                        f"Found u-dales URL in .gitmodules: {_udales_url}",
                    )
            elif stripped.startswith("branch = ") or stripped.startswith("branch="):
                # Only set branch/tag if we're in the u-dales submodule section
                if (
                    "=" in stripped
                    and current_submodule
                    and "u-dales" in current_submodule
                ):
                    _udales_tag = stripped.split("=", 1)[1].strip()
                    logger.info(
                        f"Found u-dales branch/tag in .gitmodules: {_udales_tag}",
                    )
    except Exception as e:
        logger.error(f"Error reading .gitmodules: {e}")
        import traceback

        traceback.print_exc()
else:
    logger.info(f".gitmodules not found at: {_gitmodules_path}")

# Initialize git submodule from .gitmodules
if _udales_path:
    # Check if submodule needs to be initialized
    # Repository is considered downloaded if it exists, has content, and is a valid git repo
    is_repo_downloaded = (
        _udales_path.exists()
        and any(_udales_path.iterdir())
        and (_udales_path / ".git").exists()
    )

    # Validate that the existing repo is the correct one if it exists
    if is_repo_downloaded and _udales_url:
        try:
            result = subprocess.run(
                ["git", "remote", "get-url", "origin"],
                cwd=str(_udales_path),
                check=False,
                capture_output=True,
                text=True,
            )
            if result.returncode == 0:
                actual_url = result.stdout.strip()
                expected_url = _udales_url.rstrip("/")
                actual_url_normalized = actual_url.rstrip("/")
                # Check if URLs match (handle .git suffix differences)
                if not (
                    expected_url in actual_url_normalized
                    or actual_url_normalized in expected_url
                ):
                    logger.warning(
                        f"Warning: Existing repository has wrong remote URL: {actual_url}",
                    )
                    logger.warning(
                        f"Expected: {expected_url}. Removing incorrect repository...",
                    )
                    import shutil

                    shutil.rmtree(_udales_path)
                    is_repo_downloaded = False
        except Exception as e:
            logger.warning(f"Warning: Could not validate repository URL: {e}")

    needs_init = not is_repo_downloaded

    if needs_init:
        logger.info("Initializing u-dales git submodule...")
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
                    "libs/pyudales/u-dales",
                ],
                cwd=str(_repo_root),
                check=False,
                capture_output=True,
                text=True,
            )
            if result.returncode == 0:
                logger.info("u-dales submodule initialized successfully.")
                submodule_success = True
                # Checkout the specified tag if provided
                if _udales_tag and _udales_path.exists():
                    try:
                        checkout_result = subprocess.run(
                            ["git", "checkout", _udales_tag],
                            cwd=str(_udales_path),
                            check=False,
                            capture_output=True,
                            text=True,
                        )
                        if checkout_result.returncode == 0:
                            logger.info(f"Checked out u-dales tag: {_udales_tag}")
                        else:
                            logger.warning(
                                f"Warning: Failed to checkout tag {_udales_tag}: {checkout_result.stderr}"
                            )
                    except Exception as e:
                        logger.warning(f"Warning: Exception checking out tag: {e}")
            else:
                logger.warning(
                    f"Git submodule init failed (code {result.returncode}), trying direct clone...",
                )
        except Exception as e:
            logger.warning(
                f"Exception during submodule init: {e}, trying direct clone...",
            )

        # Fallback to direct clone if submodule failed
        if not submodule_success and _udales_url:
            try:
                print(f"Cloning u-dales from {_udales_url}...", file=sys.stderr)
                # Remove empty directory if it exists
                if _udales_path.exists():
                    import shutil

                    shutil.rmtree(_udales_path)

                # Create parent directory
                _udales_path.parent.mkdir(parents=True, exist_ok=True)

                # Clone the repository
                result = subprocess.run(
                    ["git", "clone", "--recursive", _udales_url, str(_udales_path)],
                    check=False,
                    capture_output=True,
                    text=True,
                )
                if result.returncode == 0:
                    print("u-dales cloned successfully.", file=sys.stderr)
                    # Checkout the specified tag if provided
                    if _udales_tag and _udales_path.exists():
                        try:
                            checkout_result = subprocess.run(
                                ["git", "checkout", _udales_tag],
                                cwd=str(_udales_path),
                                check=False,
                                capture_output=True,
                                text=True,
                            )
                            if checkout_result.returncode == 0:
                                print(
                                    f"Checked out u-dales tag: {_udales_tag}",
                                    file=sys.stderr,
                                )
                            else:
                                print(
                                    f"Warning: Failed to checkout tag {_udales_tag}: {checkout_result.stderr}",
                                    file=sys.stderr,
                                )
                        except Exception as e:
                            print(
                                f"Warning: Exception checking out tag: {e}",
                                file=sys.stderr,
                            )
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
            "u-dales repository already downloaded, skipping initialization.",
            file=sys.stderr,
        )
        # Verify that the correct tag is checked out
        if _udales_tag and _udales_path.exists():
            try:
                result = subprocess.run(
                    ["git", "describe", "--tags", "--exact-match", "HEAD"],
                    cwd=str(_udales_path),
                    check=False,
                    capture_output=True,
                    text=True,
                )
                current_tag = result.stdout.strip() if result.returncode == 0 else None
                if current_tag != _udales_tag:
                    logger.info(
                        f"Current tag ({current_tag}) differs from expected ({_udales_tag}), checking out {_udales_tag}..."
                    )
                    checkout_result = subprocess.run(
                        ["git", "checkout", _udales_tag],
                        cwd=str(_udales_path),
                        check=False,
                        capture_output=True,
                        text=True,
                    )
                    if checkout_result.returncode == 0:
                        logger.info(f"Checked out u-dales tag: {_udales_tag}")
                    else:
                        logger.warning(
                            f"Warning: Failed to checkout tag {_udales_tag}: {checkout_result.stderr}"
                        )
            except Exception as e:
                logger.warning(f"Warning: Exception verifying/checking out tag: {e}")

    # Set UDALES_PATH from gitmodules path (always set it)
    UDALES_PATH = _udales_path.resolve()
    print(f"UDALES_PATH set to: {UDALES_PATH}", file=sys.stderr)
else:
    print("Warning: Could not find u-dales path in .gitmodules", file=sys.stderr)

# Run build scripts
if _udales_path and _udales_path.exists() and any(_udales_path.iterdir()):
    build_udales_script = _project_root / "shell_scripts/build_udales_macos.sh"
    build_preprocessing_script = (
        _project_root / "shell_scripts/build_preprocessing_macos.sh"
    )

    # Check if uDALES build is already complete
    # Build artifacts are in u-dales/build/release (build_type is "release")
    udales_build_dir = _udales_path / "build" / "release"
    udales_build_complete = (
        udales_build_dir.exists() and (udales_build_dir / "CMakeCache.txt").exists()
    )

    if build_udales_script.exists():
        if udales_build_complete:
            print("uDALES build already complete, skipping build.", file=sys.stderr)
        else:
            print("Building uDALES...", file=sys.stderr)
            subprocess.run(
                ["bash", str(build_udales_script), "release"],
                cwd=str(_project_root),
                check=False,
                env=None,  # Inherit environment (including pixi PATH)
            )

    # Check if preprocessing build is already complete
    # Build artifacts are in u-dales/tools/View3D/build
    preprocessing_build_dir = _udales_path / "tools" / "View3D" / "build"
    preprocessing_build_complete = (
        preprocessing_build_dir.exists()
        and (preprocessing_build_dir / "CMakeCache.txt").exists()
    )

    if build_preprocessing_script.exists():
        if preprocessing_build_complete:
            print(
                "Preprocessing tools build already complete, skipping build.",
                file=sys.stderr,
            )
        else:
            print("Building preprocessing tools...", file=sys.stderr)
            subprocess.run(
                ["bash", str(build_preprocessing_script)],
                cwd=str(_project_root),
                check=False,
                env=None,  # Inherit environment (including pixi PATH)
            )
