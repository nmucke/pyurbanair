"""Thin wrapper around PALM's build tooling.

PALM is built with ``palmbuild`` (part of the palm_model_system installation).
``compile_palm`` shells out to it when ``compile=True``; otherwise it is a
no-op. A real palm_model_system installation is required.

This mirrors ``pylbm.utils.compile_utils.compile_lbm`` — same shape, same
gating, different build tool.
"""

import logging
import os
import pathlib
import subprocess

logger = logging.getLogger(__name__)


def compile_palm(
    palm_root: pathlib.Path | None = None,
    config_identifier: str = "default",
    verbose: bool = True,
) -> None:
    """Invoke ``palmbuild -c <config>`` against the user's PALM install.

    ``palm_root`` defaults to the ``PALM_ROOT`` environment variable. If
    neither is set, raise a clear error pointing the user at the expected
    installation step.
    """
    root = pathlib.Path(palm_root) if palm_root else None
    if root is None:
        env_root = os.environ.get("PALM_ROOT")
        if env_root:
            root = pathlib.Path(env_root)

    if root is None or not root.exists():
        raise RuntimeError(
            "compile_palm requires a PALM installation. Set $PALM_ROOT (path "
            "to palm_model_system) or pass palm_root explicitly. See PALM's "
            "install docs: https://palm.muk.uni-hannover.de"
        )

    palmbuild = root / "trunk" / "SCRIPTS" / "palmbuild"
    if not palmbuild.exists():
        # Some PALM layouts put palmbuild directly under bin/
        palmbuild = root / "bin" / "palmbuild"
    if not palmbuild.exists():
        raise RuntimeError(
            f"palmbuild not found under {root}. "
            "Expected trunk/SCRIPTS/palmbuild or bin/palmbuild."
        )

    logger.info("Running palmbuild -c %s …", config_identifier)
    stdout = None if verbose else subprocess.DEVNULL
    stderr = None if verbose else subprocess.DEVNULL
    subprocess.run(
        [str(palmbuild), "-c", config_identifier],
        check=True,
        stdout=stdout,
        stderr=stderr,
    )
