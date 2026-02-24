import logging
import os
import pathlib
import sys

logger = logging.getLogger(__name__)


def identify_environment(repo_root: pathlib.Path, verbose: bool = True) -> pathlib.Path:
    """
    Identify the current pixi environment path.

    Checks in order:
    1. CONDA_PREFIX (set by pixi shell/run)
    2. PIXI_ENVIRONMENT environment variable (path or environment name)
    3. PIXI_PROJECT_ENVIRONMENT environment variable
    4. Active Python prefix (sys.prefix)
    5. Checks for known environments in .pixi/envs/
    6. Defaults to .pixi/envs/default

    Args:
        repo_root: Root directory of the repository

    Returns:
        Path to the pixi environment directory
    """
    # Check CONDA_PREFIX first (represents the currently active pixi env)
    conda_prefix = os.environ.get("CONDA_PREFIX")
    if conda_prefix:
        conda_env_path = pathlib.Path(conda_prefix)
        if conda_env_path.exists():
            if verbose:
                logger.info(
                    "Using pixi environment from CONDA_PREFIX: %s",
                    conda_env_path,
                )
            return conda_env_path

    # Check PIXI_ENVIRONMENT (may be a path OR an environment name)
    pixi_env = os.environ.get("PIXI_ENVIRONMENT")
    if pixi_env:
        pixi_env_candidate = pathlib.Path(pixi_env)
        if pixi_env_candidate.exists():
            if verbose:
                logger.info(
                    "Using pixi environment from PIXI_ENVIRONMENT path: %s",
                    pixi_env_candidate,
                )
            return pixi_env_candidate

        named_env_path = repo_root / ".pixi" / "envs" / pixi_env
        if named_env_path.exists():
            if verbose:
                logger.info(
                    "Using pixi environment from PIXI_ENVIRONMENT name: %s",
                    named_env_path,
                )
            return named_env_path

    # Check PIXI_PROJECT_ENVIRONMENT (set by pixi when activating an environment)
    pixi_proj_env = os.environ.get("PIXI_PROJECT_ENVIRONMENT")
    if pixi_proj_env:
        pixi_env_path = pathlib.Path(pixi_proj_env)
        if pixi_env_path.exists():
            if verbose:
                logger.info(
                    "Using pixi environment from PIXI_PROJECT_ENVIRONMENT: %s",
                    pixi_env_path,
                )
            return pixi_env_path

    # Check the current Python prefix if it looks like a pixi environment
    py_prefix = pathlib.Path(sys.prefix)
    if py_prefix.exists() and ".pixi/envs" in str(py_prefix):
        if verbose:
            logger.info("Using pixi environment from sys.prefix: %s", py_prefix)
        return py_prefix

    # Check for .pixi/envs directory
    pixi_envs_dir = repo_root / ".pixi" / "envs"
    if pixi_envs_dir.exists():
        # Check for common environment names in priority order.
        # delftblue/dev include lbm feature deps; default often does not.
        for env_name in ["delftblue", "dev", "default"]:
            env_path = pixi_envs_dir / env_name
            if env_path.exists():
                if verbose:
                    logger.info("Using pixi environment: %s", env_name)
                return env_path

    # Default fallback
    default_env = repo_root / ".pixi" / "envs" / "default"
    if verbose:
        logger.info("Using default pixi environment: %s", default_env)
    return default_env
