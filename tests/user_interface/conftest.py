"""The browser app runs in the mcp env, the only one that installs user-interface."""

from __future__ import annotations

import importlib.util

if importlib.util.find_spec("user_interface") is None:
    collect_ignore_glob = ["test_*.py"]
