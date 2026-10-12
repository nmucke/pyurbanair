"""pycity4cfd runs in the city4cfd env, the only one that installs it."""

from __future__ import annotations

import importlib.util

if importlib.util.find_spec("pycity4cfd") is None:
    collect_ignore_glob = ["test_*.py"]
