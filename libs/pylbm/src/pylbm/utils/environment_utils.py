import logging
import os
import pathlib

logger = logging.getLogger(__name__)


def identify_environment(repo_root: pathlib.Path, verbose: bool = True) -> pathlib.Path:
    """
    Identify the current pixi environment path.

    Checks in order:
    1. PIXI_ENVIRONMENT environment variable
    2. PIXI_PROJECT_ENVIRONMENT environment variable
    3. Checks for active environment in .pixi/envs/
    4. Defaults to .pixi/envs/default

    Args:
        repo_root: Root directory of the repository

    Returns:
        Path to the pixi environment directory
    """
    # Check PIXI_ENVIRONMENT environment variable first
    pixi_env = os.environ.get("PIXI_ENVIRONMENT")
    if pixi_env:
        pixi_env_path = pathlib.Path(pixi_env)
        if pixi_env_path.exists():
            if verbose:
                logger.info(
                    "Using pixi environment from PIXI_ENVIRONMENT: %s",
                    pixi_env_path,
                )
            return pixi_env_path

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

    # Check for .pixi/envs directory
    pixi_envs_dir = repo_root / ".pixi" / "envs"
    if pixi_envs_dir.exists():
        # Check for common environment names
        for env_name in ["dev", "default"]:
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
