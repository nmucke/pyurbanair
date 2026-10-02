"""MCP tools over the SDK -> persistent supervisor -> run_forward.py -> PNG.

Each backend runs on the tiny grid of tests/configs/test/tiny.yaml, passed as
plain overrides of configs/forward.yaml (the server composes the real configs
only). The surrogate export is a generated fixture, not a trained model.
"""

from __future__ import annotations

import asyncio
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
pytest.importorskip("mcp_server")

from mcp import Client
from mcp_server.jobs.registry import TERMINAL
from mcp_server.server import create_server
from mcp_server.tools import Tools

pytestmark = pytest.mark.integration
REPO = pathlib.Path(__file__).resolve().parents[2]
TINY = REPO / "tests" / "configs" / "test" / "tiny.yaml"


def _leaves(node: dict[str, Any], prefix: str = "") -> Iterator[str]:
    for key, value in node.items():
        path = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            yield from _leaves(value, path)
        else:
            yield f"{path}={json.dumps(value, separators=(',', ':'))}"


def _overrides(backend: str) -> list[str]:
    tiny = OmegaConf.to_container(OmegaConf.load(TINY))
    assert isinstance(tiny, dict)
    overrides = [
        f"model={backend}",
        "params=static",
        *_leaves({"domain": tiny["domain"], "time": tiny["time"]}),
        "ensemble.ensemble_size=2",
        "ensemble.num_parallel_processes=1",
    ]
    return (
        overrides
        + {
            "pylbm": ["model.forward_model.cuda=false"],
            "pyudales": ["model.forward_model.nudging_config.nnudge_meters=4.0"],
            "pypalm": [
                "forward.ensemble=true",
                "domain.nz=16",
                "time.simulation_time=4.0",
                "time.output_frequency=2.0",
                "params.parameters.inflow_angle.mean=12.0",
            ],
            "neural_surrogate": [
                "model.forward_model.device=cpu",
                "model.forward_model.spinup_source=training_data",
            ],
        }[backend]
    )


async def _call(client: Client, tool: str, **arguments: Any) -> dict[str, Any]:
    result = await client.call_tool(tool, arguments)
    assert not result.is_error, result.content
    assert result.structured_content is not None
    return dict(result.structured_content)


async def _wait(client: Client, run_id: str, timeout: float = 300) -> dict[str, Any]:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        status = await _call(client, "get_run_status", run_id=run_id)
        if status["state"] in TERMINAL:
            return status
        await asyncio.sleep(0.5)
    await _call(client, "cancel_run", run_id=run_id)
    logs = await _call(client, "get_run_logs", run_id=run_id)
    raise AssertionError(f"Job {run_id} did not finish in {timeout}s: {logs}")


@pytest.fixture  # type: ignore[misc]
def store(tmp_path: pathlib.Path) -> Iterator[pathlib.Path]:
    store = tmp_path / "jobs"
    store.mkdir(mode=0o700)
    with (tmp_path / "supervisor.log").open("wb") as log:
        process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "mcp_server.jobs.supervisor",
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
            yield store
        finally:
            for run in tools.list_runs()["runs"]:
                if run["state"] not in TERMINAL:
                    tools.cancel_run(run["run_id"])
            process.terminate()
            process.wait(timeout=10)


def _surrogate_export(tmp_path: pathlib.Path) -> tuple[pathlib.Path, pathlib.Path]:
    """An untrained one-frame export plus a matching initial state (dev env)."""
    tiny = OmegaConf.to_container(OmegaConf.load(TINY), resolve=True)
    script = """
import json, pathlib, sys
import numpy as np
import torch
import xarray as xr
from omegaconf import OmegaConf
from hydra.utils import instantiate
root = pathlib.Path(sys.argv[1])
domain = json.loads(sys.argv[2])
training = root / 'training'
training.mkdir(parents=True)
OmegaConf.save(OmegaConf.create({'domain': domain, 'time': {'output_frequency': 1.0}}), training / 'config.yaml')
architecture = {'_target_': 'neural_surrogates.UNetConvNeXt', 'base_channels': 4, 'channel_mults': [1, 2], 'depths': [1, 1], 'kernel_size': 3, 'expansion': 2}
params = ['inflow_angle', 'velocity_magnitude']
OmegaConf.save(OmegaConf.create({'architecture': architecture, 'dataset': {'root_dir': str(training), 'state_vars': ['u', 'v', 'w'], 'param_vars': params, 'num_history_steps': 1}}), root / 'config.yaml')
torch.save(instantiate(architecture, n_state_channels=3, n_params=2).state_dict(), root / 'weights.pt')
coords = {'time': [0.0, 1.0]}
for axis, bounds in zip('xyz', domain['bounds']):
    count = domain['n' + axis]
    coords[axis] = np.linspace(*bounds, count, endpoint=False) + (bounds[1] - bounds[0]) / (2 * count)
shape = (2, *[domain['n' + axis] for axis in 'zyx'])
fields = {v: (('time', 'z', 'y', 'x'), np.ones(shape, dtype=np.float32)) for v in ('u', 'v', 'w')}
xr.Dataset(fields, coords=coords).to_netcdf(root / 'initial.nc')
"""
    assert isinstance(tiny, dict)
    directory = tmp_path / "surrogate"
    subprocess.run(
        [
            str(REPO / ".pixi/envs/dev/bin/python"),
            "-c",
            script,
            str(directory),
            json.dumps(tiny["domain"]),
        ],
        cwd=REPO,
        env={**os.environ, "OMP_NUM_THREADS": "1"},
        check=True,
        capture_output=True,
        text=True,
    )
    return directory, directory / "initial.nc"


@pytest.mark.parametrize("backend", ["pylbm", "pyudales", "pypalm", "neural_surrogate"])  # type: ignore[misc]
def test_forward_run_to_png_over_the_protocol(
    backend: str, tmp_path: pathlib.Path, store: pathlib.Path
) -> None:
    overrides = _overrides(backend)
    initial_state = None
    if backend == "neural_surrogate":
        directory, initial = _surrogate_export(tmp_path)
        overrides.append(f"model.forward_model.model_dir={directory}")
        initial_state = {"path": str(initial), "time_index": -1}

    async def exercise() -> None:
        async with Client(create_server(REPO, store), raise_exceptions=True) as client:
            options = await _call(client, "list_config_options", group="model")
            assert backend in {option["name"] for option in options["options"]}
            plan = await _call(
                client,
                "prepare_forward_run",
                overrides=overrides,
                initial_state=initial_state,
            )
            assert plan["validation"]["configuration_valid"], plan["validation"]
            assert plan["validation"]["prerequisites_present"], plan["validation"]
            launched = await _call(
                client,
                "launch_forward_run",
                plan_id=plan["plan_id"],
                idempotency_key=f"{backend}-run",
            )
            status = await _wait(client, launched["run_id"])
            logs = await _call(client, "get_run_logs", run_id=launched["run_id"])
            assert status["state"] == "succeeded", (status, logs["text"][-4000:])
            results = await _call(
                client, "inspect_run_results", run_id=launched["run_id"]
            )
            paths = [entry["path"] for entry in results["artifacts"]]
            assert {"state.nc", "params.nc", "windows/state_0000.nc"} <= set(paths)
            state = await _call(
                client,
                "inspect_run_results",
                run_id=launched["run_id"],
                artifact_id=paths.index("state.nc"),
            )
            assert {"u", "v", "w"} <= set(state["variables"])
            render_options: dict[str, Any] = {
                "movie": False,
                "width": 320,
                "height": 240,
                "max_frames": 2,
                "stride": 2,
                "slices": [{"axis": "z", "position": 3.0}],
                "probes": [{"id": "center", "x": 10.0, "y": 10.0, "z": 3.0}],
            }
            if backend == "pypalm":
                render_options["member"] = 0
            render = await _call(
                client,
                "render_simulation",
                run_id=launched["run_id"],
                idempotency_key=f"{backend}-render",
                options=render_options,
            )
            rendered = await _wait(client, render["run_id"])
            assert rendered["state"] == "succeeded", rendered
            view = await client.call_tool(
                "get_visualization", {"visualization_id": render["run_id"]}
            )
            assert not view.is_error
            assert any(item.type == "image" for item in view.content)
            assert view.structured_content is not None
            assert view.structured_content["viewer_url"].startswith("http://127.0.0.1:")

    asyncio.run(exercise())


def test_cancel_keeps_finished_windows(store: pathlib.Path) -> None:
    overrides = [
        *_overrides("pyudales"),
        "time.simulation_time=30.0",
        "forward.rollout_steps=3",
    ]

    async def exercise() -> None:
        async with Client(create_server(REPO, store), raise_exceptions=True) as client:
            plan = await _call(client, "prepare_forward_run", overrides=overrides)
            assert plan["validation"]["configuration_valid"], plan["validation"]
            launched = await _call(
                client,
                "launch_forward_run",
                plan_id=plan["plan_id"],
                idempotency_key="cancel-run",
            )
            run_id = launched["run_id"]
            deadline = time.monotonic() + 300
            while time.monotonic() < deadline:
                results = await _call(client, "inspect_run_results", run_id=run_id)
                if results["artifacts"]:
                    break
                await asyncio.sleep(0.5)
            cancelled = await _call(client, "cancel_run", run_id=run_id)
            assert cancelled["state"] in {"cancelling", "cancelled"}
            status = await _wait(client, run_id, timeout=60)
            assert status["state"] == "cancelled", status
            results = await _call(client, "inspect_run_results", run_id=run_id)
            paths = {entry["path"] for entry in results["artifacts"]}
            assert "windows/state_0000.nc" in paths and "state.nc" not in paths

    asyncio.run(exercise())
