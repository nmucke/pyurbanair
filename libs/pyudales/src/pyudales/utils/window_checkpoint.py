"""Immutable native window inputs shared by parent and forkserver workers."""

from __future__ import annotations

import copy
import hashlib
import json
import pathlib
import re
import shutil
import tempfile
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
from scipy.io import FortranEOFError, FortranFile

from .config_utils import create_config_sh
from .inlet_turbulence_utils import (
    ELAPSED_TIME_FILENAME,
    derive_seed,
    read_elapsed_time,
    write_elapsed_time,
)
from .namoptions_utils import NamoptionsFile, rename_namoptions_file
from .warm_start_utils import CARRY_DIRNAME, CARRY_META_NAME

if TYPE_CHECKING:
    from ..forward_model import ForwardModel


def validate_carry(model: ForwardModel, *, required: bool = False) -> None:
    """Reject partial or corrupt native checkpoints instead of falling back."""
    directory = model.dirs.experiment_dir / CARRY_DIRNAME
    if not directory.exists():
        if required:
            raise ValueError("Discrepancy forecast did not produce a native carry")
        return
    try:
        metadata = json.loads((directory / CARRY_META_NAME).read_text())
        nam = NamoptionsFile(
            model.dirs.experiment_dir / f"namoptions.{model.dirs.experiment_name}"
        )
        grid = {
            key: int(nam.get_value("DOMAIN", key) or 0)
            for key in ("itot", "jtot", "ktot")
        }
        nprocx = int(nam.get_value("RUN", "nprocx") or 1)
        nprocy = int(nam.get_value("RUN", "nprocy") or 1)
        if (
            metadata["grid"] != grid
            or metadata["experiment_name"] != model.dirs.experiment_name
        ):
            raise ValueError("carry grid/member mismatch")
        names = metadata["files"]
        if (
            not names
            or nprocy != 1
            or min(grid.values()) <= 0
            or nprocx <= 0
            or grid["itot"] % nprocx
        ):
            raise ValueError("unsupported or empty carry")
        match = re.fullmatch(r"initd(\d+)_000_000\.(.+)", sorted(names)[0])
        if match is None:
            raise ValueError("missing rank zero")
        expected = {
            f"initd{match.group(1)}_{rank:03d}_000.{model.dirs.experiment_name}"
            for rank in range(nprocx)
        }
        if set(names) != expected or len(names) != len(expected):
            raise ValueError("incomplete per-rank carry")
        cells = (grid["itot"] // nprocx + 2) * (grid["jtot"] + 2) * (grid["ktot"] + 1)
        rank_clocks: list[tuple[float, float]] = []
        for name in names:
            records = []
            with FortranFile(directory / name, "r") as handle:
                while True:
                    try:
                        records.append(handle.read_record(np.uint8))
                    except FortranEOFError:
                        break
            if len(records) != 13 or len(records[2]) != 8 * cells:
                raise ValueError("invalid native restart layout")
            size = 8  # The supported cached cd2 build uses default-real-8.
            interior = grid["itot"] // nprocx * grid["jtot"] * grid["ktot"]
            if (
                len(records[0]) != interior * size
                or len(records[1]) != interior * 5 * 4
            ):
                raise ValueError("invalid native geometry records")
            if not np.isfinite(records[0].view(np.float64)).all():
                raise ValueError("nonfinite native geometry")
            if (
                any(len(record) != cells * size for record in records[2:12])
                or len(records[12]) != 2 * size
            ):
                raise ValueError("inconsistent native restart records")
            if any(
                not np.isfinite(record.view(f"f{size}")).all() for record in records[2:]
            ):
                raise ValueError("nonfinite native restart")
            clock = records[12].view(np.float64)
            if clock[0] < 0 or clock[1] <= 0:
                raise ValueError("invalid native restart clock")
            clock_values = (float(clock[0]), float(clock[1]))
            if rank_clocks and clock_values != rank_clocks[0]:
                raise ValueError("inconsistent per-rank native restart clocks")
            rank_clocks.append(clock_values)
    except (OSError, KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            f"Invalid discrepancy window carry in {directory}: {exc}"
        ) from exc


def _hashes(directory: pathlib.Path) -> dict[str, str]:
    result = {}
    for path in sorted(directory.rglob("*")):
        if path.is_file():
            with path.open("rb") as handle:
                result[str(path.relative_to(directory))] = hashlib.file_digest(
                    handle, "sha256"
                ).hexdigest()
    return result


@dataclass(frozen=True)
class WindowCheckpoint:
    """The source directory is immutable until the owning window is closed."""

    root: pathlib.Path
    experiment_name: str
    experiment_dir: pathlib.Path
    attributes: dict[str, Any]
    hashes: dict[str, str]

    @classmethod
    def capture(cls, model: ForwardModel) -> WindowCheckpoint:
        validate_carry(model)
        elapsed = read_elapsed_time(model.dirs, model._elapsed_time)
        if not np.isfinite(elapsed) or elapsed < 0:
            raise ValueError("Invalid discrepancy window physical clock")
        # Fail on a malformed persisted clock; the legacy reader intentionally
        # falls back, which is unsuitable for deterministic replay.
        clock = model.dirs.experiment_dir / ELAPSED_TIME_FILENAME
        if clock.exists():
            data = json.loads(clock.read_text())
            if (
                data.get("experiment_name") != model.dirs.experiment_name
                or float(data["elapsed_time"]) != elapsed
            ):
                raise ValueError("Invalid discrepancy window physical clock")
        if model.warmstart_template_file is not None:
            model.warmstart_template_file.relative_to(model.dirs.experiment_dir)
        root = pathlib.Path(
            tempfile.mkdtemp(
                prefix=".window_checkpoint_", dir=model.dirs.experiment_dir.parent
            )
        )
        try:
            shutil.copytree(model.dirs.experiment_dir, root / "inputs")
            attributes = {
                "_elapsed_time": elapsed,
                "spinup_time": model.spinup_time,
                "_simulation_time": model._simulation_time,
                "inlet_turbulence": copy.deepcopy(model.inlet_turbulence),
                "_discrepancy_metadata": copy.deepcopy(model._discrepancy_metadata),
                "params": copy.deepcopy(model.params),
                "_discrepancy_defaults": copy.deepcopy(model._discrepancy_defaults),
                "warmstart_template_file": model.warmstart_template_file,
            }
            return cls(
                root,
                model.dirs.experiment_name,
                model.dirs.experiment_dir,
                attributes,
                _hashes(root / "inputs"),
            )
        except BaseException:
            shutil.rmtree(root)
            raise

    def restore(self, model: ForwardModel) -> None:
        source = self.root / "inputs"
        if _hashes(source) != self.hashes:
            raise ValueError(f"Corrupt discrepancy window checkpoint: {self.root}")
        destination = model.dirs.experiment_dir
        donor = self.experiment_name != model.dirs.experiment_name
        template = self.attributes["warmstart_template_file"]
        relative_template = None
        if template is not None:
            relative_template = template.relative_to(self.experiment_dir)
            if donor:
                relative_template = relative_template.with_suffix(
                    f".{model.dirs.experiment_name}"
                )
        with tempfile.TemporaryDirectory(
            prefix=".window_restore_", dir=destination.parent
        ) as temporary:
            staged = pathlib.Path(temporary) / "inputs"
            shutil.copytree(source, staged)
            if donor:
                for path in sorted(staged.rglob("*"), reverse=True):
                    if path.is_file() and path.name.endswith(
                        f".{self.experiment_name}"
                    ):
                        path.rename(path.with_suffix(f".{model.dirs.experiment_name}"))
                rename_namoptions_file(staged, model.dirs.experiment_name)
                meta_path = staged / CARRY_DIRNAME / CARRY_META_NAME
                if meta_path.exists():
                    metadata = json.loads(meta_path.read_text())
                    metadata["experiment_name"] = model.dirs.experiment_name
                    metadata["files"] = [
                        str(
                            pathlib.Path(name).with_suffix(
                                f".{model.dirs.experiment_name}"
                            )
                        )
                        for name in metadata["files"]
                    ]
                    meta_path.write_text(json.dumps(metadata))
                # Namoptions can hold absolute paths into member-local inputs.
                nam = staged / f"namoptions.{model.dirs.experiment_name}"
                nam.write_text(
                    nam.read_text().replace(str(self.experiment_dir), str(destination))
                )
            # Copy/validation completed before replacing any live input. Keep
            # the old directory available if publication itself fails.
            previous = pathlib.Path(temporary) / "previous"
            destination.rename(previous)
            try:
                staged.rename(destination)
            except BaseException:
                previous.rename(destination)
                raise
        for name, value in self.attributes.items():
            setattr(model, name, copy.deepcopy(value))
        model._warmstart_template_dir = destination / "warmstart_template"
        if relative_template is not None:
            model.warmstart_template_file = destination / relative_template
        if donor:
            if model.inlet_turbulence.get("seed") is None:
                model.inlet_turbulence["seed"] = derive_seed(self.experiment_name)
            write_elapsed_time(model.dirs, model._elapsed_time)
            create_config_sh(model.dirs, model.matlab_bin, model.ncpu)

    def remove(self) -> None:
        shutil.rmtree(self.root)
