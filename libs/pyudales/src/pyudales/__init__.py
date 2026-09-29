"""Python wrapper for uDALES; importing never downloads or builds the solver."""

from pathlib import Path

__version__ = "0.1.0"

_project_root = Path(__file__).resolve().parents[2]
UDALES_PATH = _project_root / "u-dales"
_script_dir = Path(__file__).resolve().parent / "shell_scripts"
if not _script_dir.is_dir():
    _script_dir = _project_root / "shell_scripts"
LOCAL_EXECUTE_SCRIPT = _script_dir / "local_execute.sh"
