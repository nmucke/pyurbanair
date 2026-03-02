"""Utilities for parsing and editing mod_dimensions.F90 (experiment dimensions)."""

import logging
import pathlib
import re
from typing import Optional, Union

from pylbm.utils import DirectoryPaths

logger = logging.getLogger(__name__)

# Parameter names we know how to parse and write.
DIMENSION_PARAMS = ("nx", "ny", "ntiles", "nyg", "nz", "ntracer")

# Default comment suffix for each parameter (for writing).
PARAM_COMMENTS = {
    "nx": "! grid dimension x-dir (east)",
    "ny": "! grid dimension y-dir (north)",
    "ntiles": "! Number of tiles in y direction",
    "nyg": "! global grid dimension y-dir (north)",
    "nz": "! grid dimension z-dir (up)",
    "ntracer": "! Number of tracer fields (potential temperature etc)",
}


def _normalize_experiment_name(line: str) -> Optional[str]:
    """Extract experiment name from a comment line like '!city2' or '! runcase' or '!windfarm big D=32'."""
    stripped = line.strip()
    if not stripped.startswith("!"):
        return None
    rest = stripped.lstrip("!").strip()
    if not rest:
        return None
    return rest.split()[0].strip()


def _parse_param_line(line: str) -> Optional[tuple[str, Union[int, None]]]:
    """Parse 'integer, parameter :: nx = 160' or 'nyg = ntiles*ny'. Returns (name, value) or (name, None) for expressions."""
    stripped = line.strip().lstrip("!").strip()
    m = re.match(
        r"integer\s*,\s*parameter\s*::\s*(\w+)\s*=\s*(\S+)", stripped, re.IGNORECASE
    )
    if not m:
        return None
    name = m.group(1).lower()
    value_part = m.group(2).strip()
    try:
        return (name, int(value_part))
    except ValueError:
        if "ntiles*ny" in value_part or "ntiles* ny" in value_part:
            return (name, None)  # Will be computed from ntiles and ny
        return None


class ModDimensions:
    """
    A parser and editor for mod_dimensions.F90.

    The file contains multiple "experiments" (e.g. city2, runcase), each with dimension
    parameters (nx, ny, nz, and optionally ntiles, nyg, ntracer). One experiment is
    "active" (uncommented); the rest are commented out. This class allows
    getting/setting experiment parameters, adding new experiments, and switching the active experiment.
    """

    def __init__(self, file_path: pathlib.Path) -> None:
        """
        Initialize ModDimensions by reading and parsing the file.

        Args:
            file_path: Path to mod_dimensions.F90.
        """
        self.file_path = pathlib.Path(file_path)
        self._experiments: dict[str, dict[str, int]] = {}
        self._experiment_order: list[str] = []
        self._experiment_line_ranges: dict[str, tuple[int, int]] = {}
        self._active_experiment: Optional[str] = None
        self.raw_lines: list[str] = []
        self._header_end: int = 0
        self._end_module_line: int = 0

        if self.file_path.exists():
            self._parse_file()

    def _parse_file(self) -> None:
        """Parse mod_dimensions.F90 into experiment blocks and active experiment."""
        if not self.file_path.exists():
            return

        with open(self.file_path, "r") as f:
            self.raw_lines = f.readlines()

        i = 0
        # Find the first experiment block: a "!word" line followed by at least one integer, parameter line
        while i < len(self.raw_lines):
            line = self.raw_lines[i]
            stripped = line.strip()
            if stripped.startswith("end module"):
                self._end_module_line = i
                return
            if stripped.startswith("!"):
                start = i
                experiment_name = _normalize_experiment_name(line)
                if experiment_name:
                    params: dict[str, int] = {}
                    j = i + 1
                    while j < len(self.raw_lines):
                        next_line = self.raw_lines[j]
                        next_stripped = next_line.strip()
                        if next_stripped.startswith("end module"):
                            self._end_module_line = j
                            break
                        if (
                            next_stripped.startswith("!")
                            and "integer" not in next_stripped
                        ):
                            break
                        parsed = _parse_param_line(next_line)
                        if parsed:
                            name, value = parsed
                            if value is not None:
                                params[name] = value
                            elif (
                                name == "nyg" and "ntiles" in params and "ny" in params
                            ):
                                params["nyg"] = params["ntiles"] * params["ny"]
                            if not next_stripped.startswith("!"):
                                self._active_experiment = experiment_name
                        j += 1
                    if "ntiles" in params and "ny" in params and "nyg" not in params:
                        params["nyg"] = params["ntiles"] * params["ny"]
                    if params:
                        if self._header_end == 0:
                            self._header_end = start
                        self._experiments[experiment_name] = params
                        if experiment_name not in self._experiment_order:
                            self._experiment_order.append(experiment_name)
                        self._experiment_line_ranges[experiment_name] = (start, j)
                    i = j
                    continue
            i += 1

    def get_experiments(self) -> dict[str, dict[str, int]]:
        """Return a copy of all experiments (name -> {nx, ny, nz, ...})."""
        return {k: dict(v) for k, v in self._experiments.items()}

    def has_experiment(self, experiment_name: str) -> bool:
        """Return True if the experiment exists."""
        return experiment_name in self._experiments

    def get_active_experiment_name(self) -> Optional[str]:
        """Return the name of the currently active experiment (the one with uncommented parameters)."""
        return self._active_experiment

    def get_experiment_params(self, experiment_name: str) -> Optional[dict[str, int]]:
        """Return a copy of the parameters for an experiment, or None if not found."""
        if experiment_name not in self._experiments:
            return None
        return dict(self._experiments[experiment_name])

    def set_experiment_params(
        self,
        experiment_name: str,
        nx: Optional[int] = None,
        ny: Optional[int] = None,
        nz: Optional[int] = None,
        ntiles: Optional[int] = None,
        nyg: Optional[int] = None,
        ntracer: Optional[int] = None,
    ) -> None:
        """
        Update parameters for an existing experiment. Omitted arguments are left unchanged.

        Args:
            experiment_name: Name of the experiment to update.
            nx, ny, nz, ntiles, nyg, ntracer: New values (only provided ones are updated).
        """
        if experiment_name not in self._experiments:
            raise KeyError(
                f"Experiment '{experiment_name}' not found. Use add_experiment() to add it."
            )

        p = self._experiments[experiment_name]
        if nx is not None:
            p["nx"] = nx
        if ny is not None:
            p["ny"] = ny
        if nz is not None:
            p["nz"] = nz
        if ntiles is not None:
            p["ntiles"] = ntiles
        if nyg is not None:
            p["nyg"] = nyg
        if ntracer is not None:
            p["ntracer"] = ntracer

    def add_experiment(
        self,
        experiment_name: str,
        nx: int,
        ny: int,
        nz: int,
        ntiles: int = 1,
        ntracer: int = 0,
        set_active: bool = True,
    ) -> None:
        """
        Add a new experiment with the given dimensions. Optionally set it as the active experiment.

        Args:
            experiment_name: Name of the new experiment (e.g. "runcase", "city2").
            nx, ny, nz: Grid dimensions.
            ntiles: Number of tiles in y (default 1). If 1, nyg is set to ny.
            ntracer: Number of tracer fields (default 0).
            set_active: If True, make this experiment the active one (uncommented).
        """
        nyg = ntiles * ny
        self._experiments[experiment_name] = {
            "nx": nx,
            "ny": ny,
            "nz": nz,
            "ntiles": ntiles,
            "nyg": nyg,
            "ntracer": ntracer,
        }
        if experiment_name not in self._experiment_order:
            self._experiment_order.append(experiment_name)
        if set_active:
            self._active_experiment = experiment_name

    def set_active_experiment(self, experiment_name: str) -> None:
        """
        Set the active experiment (the one that will be written as uncommented).

        Args:
            experiment_name: Name of an existing experiment.
        """
        if experiment_name not in self._experiments:
            raise KeyError(f"Experiment '{experiment_name}' not found.")
        self._active_experiment = experiment_name

    def _format_experiment_block(self, experiment_name: str, active: bool) -> list[str]:
        """Generate the lines for one experiment block."""
        params = self._experiments[experiment_name]
        prefix = "" if active else "!"
        lines = [f"!{experiment_name}\n", "\n"]
        for var in ("nx", "ny", "ntiles", "nyg", "nz", "ntracer"):
            if var not in params:
                continue
            comment = PARAM_COMMENTS.get(var, "")
            lines.append(
                f"{prefix}  integer, parameter :: {var} = {params[var]:<10} {comment}\n"
            )
        lines.append("\n")
        return lines

    def write(self, file_path: Optional[pathlib.Path] = None) -> None:
        """
        Write mod_dimensions.F90 back to disk. Header and end module are preserved;
        all experiment blocks are rewritten with the active experiment uncommented and others commented.

        Args:
            file_path: Optional path to write to. If None, writes to original path.
        """
        output_path = file_path if file_path is not None else self.file_path

        new_lines: list[str] = []
        new_lines.extend(self.raw_lines[: self._header_end])

        for experiment_name in self._experiment_order:
            active = experiment_name == self._active_experiment
            new_lines.extend(self._format_experiment_block(experiment_name, active))

        if self._end_module_line < len(self.raw_lines):
            new_lines.append(self.raw_lines[self._end_module_line])

        with open(output_path, "w") as f:
            f.writelines(new_lines)

        if file_path is not None:
            self.file_path = output_path
            self._parse_file()


def set_experiment(
    dirs: DirectoryPaths,
    nx: int,
    ny: int,
    nz: int,
) -> None:
    """
    Set experiment dimensions in mod_dimensions.F90 (add or update experiment, set active).

    This function will:
    - Add the experiment if it doesn't exist
    - Update the experiment parameters if it already exists
    - Set the experiment as active
    - Write the changes to the file

    Args:
        dirs: DirectoryPaths object containing all relevant paths (including experiment_dir
              and executable_path).
        nx: Grid dimension in x-direction.
        ny: Grid dimension in y-direction.
        nz: Grid dimension in z-direction.

    Raises:
        FileNotFoundError: If dirs.mod_dimensions_path doesn't exist.
    """
    if not dirs.mod_dimensions_path.exists():
        raise FileNotFoundError(
            f"mod_dimensions.F90 not found at {dirs.mod_dimensions_path}"
        )

    mod_dims = ModDimensions(dirs.mod_dimensions_path)

    if mod_dims.has_experiment(dirs.experiment_name):
        update_kwargs: dict[str, int] = {"nx": nx, "ny": ny, "nz": nz}

        # ForwardModel updates "runcase" dimensions; keep nyg consistent with ny.
        if dirs.experiment_name == "runcase":
            update_kwargs["nyg"] = ny

        mod_dims.set_experiment_params(dirs.experiment_name, **update_kwargs)
        mod_dims.set_active_experiment(dirs.experiment_name)
    else:
        mod_dims.add_experiment(
            dirs.experiment_name, nx=nx, ny=ny, nz=nz, set_active=True
        )

    mod_dims.write()
