"""Entry points import pyurbanair.quiet_jax before jax, or it silences nothing."""

from __future__ import annotations

import subprocess
import sys

import pytest

from tests.conftest import REPO

ENTRY_POINTS = sorted(
    p.relative_to(REPO).as_posix()
    for p in (REPO / "scripts").rglob("*.py")
    if "utils" not in p.parts
)

# Imports the script in a fresh interpreter; sys.modules keeps import order.
CHECK = """
import importlib.util, sys
spec = importlib.util.spec_from_file_location("entry_point", sys.argv[1])
module = sys.modules["entry_point"] = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
names = list(sys.modules)
if "jax" in names:
    assert "pyurbanair.quiet_jax" in names, "quiet_jax not imported"
    assert names.index("pyurbanair.quiet_jax") < names.index("jax"), "jax first"
"""


@pytest.mark.parametrize("script", ENTRY_POINTS)  # type: ignore[misc]
def test_quiet_jax_before_jax(script: str) -> None:
    result = subprocess.run(
        [sys.executable, "-c", CHECK, str(REPO / script)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr[-2000:]
