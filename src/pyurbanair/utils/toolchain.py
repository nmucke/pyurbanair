"""Compiler settings shared by every solver build (uDALES, LBM, test kernels)."""

from __future__ import annotations

import os
import platform
import shlex
import shutil
from pathlib import Path
from typing import Mapping


def apple_linker_flags(
    compiler_command: str, env: Mapping[str, str] | None = None
) -> list[str]:
    """Return ``["-B/usr/bin/"]`` when a pixi compiler links on macOS.

    Conda's ld64 lags new macOS SDKs: ld64-956.6 cannot parse a current SDK's
    ``libSystem.tbd`` ("unknown architecture arm64e.x1") and the link then
    misses symbols such as ``expf`` and ``memcpy``. ``-B/usr/bin/`` makes the
    compiler driver run Apple's ld instead; compilation is unchanged. Empty on
    Linux, without Apple's ld, and for compilers outside the pixi env.
    """
    env = os.environ if env is None else env
    if platform.system() != "Darwin" or not os.access("/usr/bin/ld", os.X_OK):
        return []
    tokens = shlex.split(compiler_command)
    compiler = shutil.which(tokens[0], path=env.get("PATH")) if tokens else None
    if compiler is None:
        return []
    compiler_path = Path(compiler)
    prefix = env.get("CONDA_PREFIX")
    in_conda = prefix is not None and compiler_path.resolve().is_relative_to(
        Path(prefix).resolve()
    )
    in_pixi = ".pixi" in compiler_path.parts and "envs" in compiler_path.parts
    return ["-B/usr/bin/"] if in_conda or in_pixi else []
