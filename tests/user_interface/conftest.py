"""The browser app runs in the ui env (mcp also installs it); skipped elsewhere."""

from __future__ import annotations

import importlib.util

if importlib.util.find_spec("user_interface") is None:
    collect_ignore_glob = ["test_*.py"]
