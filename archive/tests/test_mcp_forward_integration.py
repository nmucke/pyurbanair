"""Actual MCP tool adapters -> persistent supervisor -> solver -> PNG.

Uses physical settings from tests/conf, never production numerical tuning. The
surrogate export is a generated test fixture, not a claim of predictive quality.
"""

from __future__ import annotations

import json
import os
import pathlib
import subprocess
import sys
import time
from collections.abc import Iterator
from typing import Any

import pytest
from omegaconf import OmegaConf

pytest.importorskip("mcp")
pytest.importorskip("pyurbanair_mcp")

from pyurbanair_mcp.tools import Tools

from pyurbanair.jobs.registry import TERMINAL
from tests.config_loader import compose_test_config

pytestmark = pytest.mark.integration
REPO = pathlib.Path(__file__).resolve().parents[1]


def _leaves(node: dict[str, Any], prefix: str = "") -> Iterator[str]:
    for key, value in node.items():
        path = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            yield from _leaves(value, path)
        else:
            yield f'++{path}={json.dumps(value, separators=(",", ":"))}'


def _overrides(backend: str) -> tuple[list[str], dict[str, Any]]:
    cfg = compose_test_config([f"model={backend}", "params=static"])
    frozen = OmegaConf.to_container(cfg, resolve=True)
    assert isinstance(frozen, dict)
    selected = {
        key: frozen[key]
        for key in ("domain", "geometry", "time", "params", "model", "ensemble", "run")
    }
    selected["run"].update(
        skip_viz=True, rollout_steps=0, ensemble=backend == "pypalm", results_dir=None
    )
    selected["ensemble"].update(
        ensemble_size=2, num_parallel_processes=1, num_cpus_per_process=1
    )
    if backend == "pylbm":
        selected["model"]["forward_model"]["cuda"] = False
    if backend == "pypalm":
        selected["domain"]["nz"] = 16
        selected["time"].update(simulation_time=4.0, output_frequency=2.0)
        selected["model"]["forward_model"].update(
            nz=16, simulation_time=4.0, output_frequency=2.0
        )
        selected["params"]["parameters"]["inflow_angle"]["mean"] = 12.0
    return [f"model={backend}", "params=static", *list(_leaves(selected))], selected


def _wait(tools: Tools, run_id: str, timeout: float = 240) -> dict[str, Any]:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        status = tools.get_run_status(run_id)
        if status["state"] in TERMINAL:
            return dict(status)
        time.sleep(0.2)
    tools.cancel_run(run_id)
    raise AssertionError(
        f"Job {run_id} did not complete in {timeout}s: {tools.get_run_logs(run_id)}"
    )


@pytest.fixture  # type: ignore[misc, unused-ignore]
def tool_service(tmp_path: pathlib.Path) -> Iterator[Tools]:
    store = tmp_path / "jobs"
    store.mkdir(mode=0o700)
    with (tmp_path / "supervisor.log").open("wb") as log:
        process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "pyurbanair.jobs.supervisor",
                "--repo-root",
                str(REPO),
                "--root",
                str(store),
            ],
            cwd=REPO,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
        tools = Tools(REPO, store)
        deadline = time.monotonic() + 10
        while True:
            try:
                tools.jobs._request("ping", {})
                break
            except (ConnectionError, FileNotFoundError):
                if process.poll() is not None or time.monotonic() > deadline:
                    raise AssertionError((tmp_path / "supervisor.log").read_text())
                time.sleep(0.05)
        try:
            yield tools
        finally:
            for run in tools.list_runs()["runs"]:
                if run["state"] not in TERMINAL:
                    tools.cancel_run(run["run_id"])
                    _wait(tools, run["run_id"], timeout=20)
            process.terminate()
            process.wait(timeout=10)


def _checkpoint(
    tmp_path: pathlib.Path, settings: dict[str, Any]
) -> tuple[pathlib.Path, pathlib.Path]:
    # Only the worker environment needs torch or the surrogate runtime.
    script = """
import json, pathlib, sys
import numpy as np
import torch
import xarray as xr
from omegaconf import OmegaConf
from hydra.utils import instantiate
root = pathlib.Path(sys.argv[1])
settings = json.loads(sys.argv[2])
root.mkdir()
training = root / 'training'
training.mkdir()
OmegaConf.save(OmegaConf.create({'domain': settings['domain'], 'time': {'output_frequency': 1.0}}), training / 'config.yaml')
architecture = {'_target_': 'neural_surrogates.UNetConvNeXt', 'base_channels': 4, 'channel_mults': [1, 2], 'depths': [1, 1], 'kernel_size': 3, 'expansion': 2, 'num_history_steps': 2}
OmegaConf.save(OmegaConf.create({'architecture': architecture, 'dataset': {'root_dir': str(training), 'state_vars': ['u','v','w'], 'param_vars': ['inflow_angle','velocity_magnitude'], 'num_history_steps': 2}}), root / 'config.yaml')
model = instantiate(architecture, n_state_channels=3, n_params=2)
torch.save(model.state_dict(), root / 'weights.pt')
coords = {'time': [0., 1.]}
for axis, bounds in zip('xyz', settings['domain']['bounds']):
    count = settings['domain']['n' + axis]
    coords[axis] = np.linspace(*bounds, count, endpoint=False) + (bounds[1]-bounds[0]) / (2 * count)
shape = (2, *[settings['domain']['n'+axis] for axis in 'zyx'])
xr.Dataset({v: (('time','z','y','x'), np.ones(shape, dtype=np.float32)) for v in ('u','v','w')}, coords=coords).to_netcdf(root / 'initial.nc')
"""
    directory = tmp_path / "fixture_checkpoint"
    subprocess.run(
        [
            str(REPO / ".pixi/envs/dev/bin/python"),
            "-c",
            script,
            str(directory),
            json.dumps(settings),
        ],
        cwd=REPO,
        env={**os.environ, "OMP_NUM_THREADS": "1"},
        check=True,
        capture_output=True,
        text=True,
    )
    return directory, directory / "initial.nc"


@pytest.mark.parametrize("backend", ["pylbm", "pyudales", "pypalm", "neural_surrogate"])  # type: ignore[misc, unused-ignore]
def test_solver_to_png_through_tools(
    backend: str, tmp_path: pathlib.Path, tool_service: Tools
) -> None:
    overrides, settings = _overrides(backend)
    initial_state = None
    if backend == "neural_surrogate":
        directory, initial = _checkpoint(tmp_path, settings)
        overrides.extend(
            [
                f"model.forward_model.model_dir={directory}",
                "model.forward_model.device=cpu",
            ]
        )
        initial_state = {"path": str(initial)}
    # Source checks deliberately reject edits made between preparation and launch.
    # Allow a bounded retry when this integration test runs alongside development.
    for attempt in range(3):
        plan = tool_service.prepare_forward_run(overrides, initial_state=initial_state)
        assert plan["validation"]["configuration_valid"], plan["validation"]
        assert plan["validation"]["prerequisites_present"], plan["validation"]
        try:
            launched = tool_service.launch_forward_run(
                plan["plan_id"], f"{backend}-{attempt}"
            )
        except ValueError as error:
            if attempt < 2 and "changed" in str(error):
                continue
            raise
        status = _wait(tool_service, launched["run_id"])
        if (
            status["state"] != "succeeded"
            and attempt < 2
            and "changed" in json.dumps(status)
        ):
            continue
        break
    assert status["state"] == "succeeded", (
        status,
        tool_service.get_run_logs(launched["run_id"]),
    )
    results = tool_service.inspect_run_results(launched["run_id"])
    assert results, results
    if backend != "neural_surrogate":
        inputs = [
            entry for entry in results["artifacts"] if entry["kind"] == "solver_input"
        ]
        assert len(inputs) == (2 if backend == "pypalm" else 1)
        assert all(
            entry["sha256"] and entry["source_relative_path"] for entry in inputs
        )
    options = {
        "movie": False,
        "width": 320,
        "height": 240,
        "max_frames": 2,
        "stride": 2,
        "slices": [{"axis": "z", "position": 3.0}],
        "probes": [{"id": "center", "x": 10.0, "y": 10.0, "z": 3.0}],
    }
    if backend == "pypalm":
        options["member"] = 0
    render = tool_service.render_simulation(
        launched["run_id"], f"{backend}-render", options
    )
    rendered = _wait(tool_service, render["run_id"])
    assert rendered["state"] == "succeeded", (
        rendered,
        tool_service.get_run_logs(render["run_id"]),
    )
    metadata, png = tool_service.visualization(render["run_id"])
    assert png is not None and png.startswith(b"\x89PNG\r\n\x1a\n")
    assert metadata["viewer_url"].startswith("http://127.0.0.1:")
    assert tool_service.get_run_status(launched["run_id"])["state"] == "succeeded"
